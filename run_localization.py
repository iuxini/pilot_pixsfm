# 用途：仅接收图像、地图、输出路径和内参，每次全新生成中间产物并自动完成六图定位。
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import rasterio
from run_joint_map import PROJECT_ROOT, PIPELINE_STAGES, check_config, check_environment
from joint_map.export import read_geographic_poses
from joint_map.joint import load_confirmed_handoff
from joint_map.optimization import confirm_prior_candidate, get_sha256, write_validation_json

def prepare_config(image_dir, dom_path, dsm_path, intrinsics):
    """Validate raw six-image inputs and replace every data-dependent path from the protocol template."""
    images, dom_path, dsm_path = [Path(value).expanduser().resolve() for value in [image_dir, dom_path, dsm_path]]
    names = sorted(path.name for path in images.iterdir() if path.is_file() and path.suffix.lower() in ['.png', '.jpg', '.jpeg'])
    config = json.loads((PROJECT_ROOT/'configs/joint_map.json').read_text())
    config.update(image_dir=str(images), image_names=names, dom_path=str(dom_path), dsm_path=str(dsm_path), camera={'model': 'PINHOLE', 'width': 3840, 'height': 2160, 'params': list(intrinsics), 'fixed': True})
    for key in ['native_sfm_dir', 'native_dense_dir', 'output_dir']:
        config.pop(key)
    config['frontend'] = {}
    config['prior_validation'].pop('selected_model_path')
    check_config(config)
    if len({Path(name).stem for name in names}) != len(names) or any(any(character.isspace() for character in name) for name in names):
        raise ValueError('Use six unique image stems and filenames without whitespace')
    for name in names:
        with Image.open(images/name) as image:
            if image.size != (3840, 2160) or image.mode != 'RGB':
                raise ValueError('Expected six 3840x2160 RGB images: '+name)
            image.verify()
    with rasterio.open(dom_path) as dom, rasterio.open(dsm_path) as dsm:
        if dom.crs != rasterio.crs.CRS.from_epsg(4326) or dsm.crs != dom.crs or min(dsm.shape) < 2 or dsm.shape != dom.shape or dsm.transform != dom.transform or not np.isfinite(list(dsm.transform)).all() or dsm.transform.determinant == 0 or dom.count != 3 or dom.dtypes != ('int16',)*3 or dom.nodata != 255 or dsm.count != 1 or dsm.dtypes != ('float32',) or dsm.nodata != -9999:
            raise ValueError('Expected common-grid EPSG:4326 DOM int16 RGB/NoData255 and DSM float32/NoData-9999')
        if not np.any(np.isfinite(dsm.read(1, masked=True)).filled(False)):
            raise ValueError('DSM has no valid finite height')
    return config

def select_automatic_prior(output):
    """Choose only validated candidates using a declared deterministic policy without reading GT."""
    shared = output/'shared'
    report = json.loads((shared/'prior_validation.json').read_text())
    passed = {row['candidate_id']: row for row in report['candidates'] if row['status'] == 'passed'}
    if report['status'] != 'complete' or not passed:
        raise ValueError('No completed passing prior validation; localization cannot continue')
    proposed = report.get('candidate_id') if report.get('selection_status') == 'proposed' else None
    candidate = proposed if proposed in passed else next(name for name in ['weight_1p0', 'weight_0p1', 'weight_10p0'] if name in passed)
    prior = confirm_prior_candidate(output, candidate, '用户授权无人值守定位；按automatic_no_gt_v1策略选择本批已通过候选，非用户手选权重。')
    prior.update(selection_basis='automatic_no_gt_v1', automation_policy={'prefer_validated_dominance_proposal': True, 'fallback_order': ['weight_1p0', 'weight_0p1', 'weight_10p0'], 'gt_inputs_read': False})
    write_validation_json(shared/'prior_model.json', prior)
    return prior

def run_localization(image_dir, dom_path, dsm_path, output_dir, intrinsics):
    """Start a fresh run and generate every intermediate and final artifact without user intervention."""
    check_environment()
    config = prepare_config(image_dir, dom_path, dsm_path, intrinsics)
    output = Path(output_dir).expanduser().resolve()/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'cache/tmp').mkdir(parents=True)
    (output/'logs').mkdir()
    native = output/'native'
    config.update(output_dir=str(output), native_sfm_dir=str(native/'sfm'), native_dense_dir=str(native/'dense'))
    config['frontend'] = {'features_path': str(native/'keypoints.h5'), 'adjusted_features_path': str(native/'sfm/refined_keypoints.h5'), 'matches_path': str(native/'matches.h5')}
    config['prior_validation']['selected_model_path'] = str(output/'shared/prior_model.json')
    config_path = output/'reconstruction.json'
    write_validation_json(config_path, config)
    result = {'status': 'running', 'output_dir': str(output), 'data_artifacts_reused': False, 'gt_inputs_read': False, 'accuracy_evaluated': False, 'stages': []}
    environment = dict(os.environ, TMPDIR=str(output/'cache/tmp'))
    print('本次全新运行目录：'+str(output), flush=True)
    def run_step(label, command):
        """Persist each subprocess log and stop the workflow immediately on failure."""
        log = output/'logs'/('localize_'+label+'.log')
        result['current_stage'] = label
        write_validation_json(output/'localization_result.json', result)
        print('正在运行：'+label, flush=True)
        with log.open('w') as stream:
            process = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        result['stages'].append({'stage': label, 'exit_code': process.returncode, 'log_path': str(log)})
        if process.returncode:
            result.update(status='failed', error='Stage '+label+' failed; see '+str(log))
            write_validation_json(output/'localization_result.json', result)
            raise RuntimeError(result['error'])
        write_validation_json(output/'localization_result.json', result)
    run_step('native', [sys.executable, '-B', '-u', '-m', 'joint_map.native', '--config', str(config_path)])
    for stage, branch in PIPELINE_STAGES:
        if stage == 'evaluate':
            continue
        if stage == 'register':
            code = 'import sys; from pathlib import Path; from run_localization import select_automatic_prior; select_automatic_prior(Path(sys.argv[1]))'
            run_step('select_prior', [sys.executable, '-B', '-u', '-c', code, str(output)])
            prior = json.loads((output/'shared/prior_model.json').read_text())
            result['prior_selection'] = {'candidate_id': prior['candidate_id'], 'lambda_map': prior['lambda_map'], 'selection_basis': prior['selection_basis']}
        command = [sys.executable, '-B', '-u', str(PROJECT_ROOT/'run_joint_map.py'), '--config', str(config_path), '--stage', stage]
        if stage in ['dense', 'update', 'export']:
            command += ['--branch', branch]
        run_step(stage+('_'+branch if stage in ['dense', 'update', 'export'] else ''), command)
    load_confirmed_handoff(output)
    export = json.loads((output/'joint/export/summary.json').read_text())
    for path, digest in export['input_sha256'].items():
        if get_sha256(Path(path)) != digest:
            raise ValueError('Localization evidence changed: '+path)
    for name, path in export['artifacts'].items():
        if get_sha256(Path(path)) != export['output_sha256'][name]:
            raise ValueError('Localization output changed: '+path)
    pose_path = output/'joint/export/poses_wgs84.txt'
    poses = read_geographic_poses(pose_path, config['image_names'])
    fields = ['longitude_deg', 'latitude_deg', 'ellipsoidal_height_m', 'roll_deg', 'pitch_deg', 'yaw_deg']
    result.update(status='complete', current_stage='complete', poses=[dict(image_name=name, **dict(zip(fields, values.tolist()))) for name, values in zip(config['image_names'], poses)], poses_wgs84=str(pose_path), sparse_cloud=str(output/'joint/export/sparse_wgs84.ply'), dense_cloud=str(output/'joint/export/dense_wgs84.ply'), updated_dsm=str(output/'joint/map/updated_dsm.tif'))
    write_validation_json(output/'localization_result.json', result)
    print('定位完成。位姿文件：'+str(pose_path), flush=True)
    print('运行结果：'+str(output/'localization_result.json'), flush=True)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='六张照片+DOM/DSM自动定位；每次重新生成全部数据产物，不需要配置文件或GT')
    parser.add_argument('--image-dir', type=Path, required=True, help='仅含本批六张RGB图像的目录')
    parser.add_argument('--dom', type=Path, required=True, help='DOM GeoTIFF路径')
    parser.add_argument('--dsm', type=Path, required=True, help='DSM GeoTIFF路径')
    parser.add_argument('--output-dir', type=Path, required=True, help='输出根目录；每次自动建立新的run子目录')
    parser.add_argument('--intrinsics', nargs=4, type=float, required=True, metavar=('FX', 'FY', 'CX', 'CY'))
    args = parser.parse_args()
    run_localization(args.image_dir, args.dom, args.dsm, args.output_dir, args.intrinsics)
