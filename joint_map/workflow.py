# 用途：提供三个独立入口共用的输入检查、批次交接、日志及无GT候选选择。
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import rasterio
from run_joint_map import PROJECT_ROOT, check_config, check_environment
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

def check_mode(mode):
    """Accept only the explicit development mode or the default uncached real mode."""
    if mode not in ['real', 'modify']:
        raise ValueError('Expected mode real or modify')

def guard_real_paths(batch_dir, paths):
    """Reject generated data or input aliases that resolve into the development cache namespace."""
    root = Path(batch_dir).expanduser().resolve()
    for path in paths:
        actual = Path(path).resolve()
        if any(name in actual.parts for name in ['modify', 'modify_cache']) or actual.is_relative_to(root/'modify'):
            raise ValueError('real mode cannot read modify data or caches: '+str(actual))

def mode_fields(mode, fingerprint):
    """Record execution policy and the actual data identity independently of algorithm outputs."""
    check_mode(mode)
    return {'mode': mode, 'input_fingerprint': fingerprint, 'cache_allowed': mode == 'modify', 'cache_used': False, 'cache_hits': [], 'cache_events': []}

def create_run(batch_dir, part, mode='real'):
    """Create a fresh result directory for one independently invoked workflow part."""
    batch_dir = Path(batch_dir).expanduser().resolve()
    check_mode(mode)
    output = batch_dir/mode/part/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    if mode == 'real':
        guard_real_paths(batch_dir, [output])
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    (output/'cache/tmp').mkdir(parents=True)
    return batch_dir, output

def read_batch(batch_dir, required, mode='real'):
    """Require complete, unmodified outputs from the active batch instead of searching older runs."""
    batch_dir = Path(batch_dir).expanduser().resolve()
    check_mode(mode)
    state = json.loads((batch_dir/'batch.json').read_text())
    if state.get('mode', 'real') != mode:
        raise ValueError('Batch mode differs from '+mode+'; run the first step in that mode or use the matching --mode')
    records = {}
    for part in required:
        item = state.get(part)
        if not item or not item.get('result_sha256'):
            raise ValueError('Run '+part+' successfully for this batch first')
        output = Path(item['run_dir'])
        allowed = [batch_dir/mode/part]
        if 'mode' not in state:
            allowed.append(batch_dir/part)
        if mode == 'real':
            guard_real_paths(batch_dir, [output])
        if not any(output.resolve().is_relative_to(folder.resolve()) for folder in allowed) or get_sha256(output/'result.json') != item['result_sha256']:
            raise ValueError('Batch result changed: '+part)
        result = json.loads((output/'result.json').read_text())
        if result.get('mode', 'real') != mode or (mode == 'real' and result.get('cache_used', False)):
            raise ValueError('Result mode or cache provenance differs from '+mode)
        if result['status'] != 'complete' or result['batch_id'] != state['batch_id'] or Path(result['output_dir']).resolve() != output.resolve() or get_sha256(output/'reconstruction.json') != result['config_sha256']:
            raise ValueError('Incomplete, modified or different-batch result: '+part)
        if mode == 'real':
            config = json.loads((output/'reconstruction.json').read_text())
            if config.get('workflow_mode', 'real') != 'real':
                raise ValueError('real mode rejects a modify reconstruction configuration')
            paths = [output/name for name in ['shared', 'native', 'baseline', 'joint', 'baseline/model', 'joint/model']]
            paths += [PROJECT_ROOT/config[key] for key in ['image_dir', 'dom_path', 'dsm_path'] if key in config]
            paths += [config[key] for key in ['native_sfm_dir', 'native_dense_dir'] if key in config]
            paths += list(config.get('frontend', {}).values())
            guard_real_paths(batch_dir, paths)
        records[part] = result
    return state, records

def publish_batch(batch_dir, state):
    """Reject a stale stage attempting to overwrite another active batch's handoff."""
    current = json.loads((batch_dir/'batch.json').read_text())
    if current['batch_id'] != state['batch_id']:
        raise ValueError('Another original run started; preserve this result and use its own batch')
    write_validation_json(batch_dir/'batch.json', state)

def run_step(output, result, label, command):
    """Run only the requested part's subprocess and retain its failure without starting later parts."""
    log = output/'logs'/('step_'+label+'.log')
    result['current_stage'] = label
    result['cache_hits'] = [row['stage'] for row in result.get('cache_events', []) if row['hit']]
    result['cache_used'] = bool(result['cache_hits'])
    write_validation_json(output/'result.json', result)
    print('正在运行：'+label, flush=True)
    with log.open('w') as stream:
        process = subprocess.run(command, cwd=PROJECT_ROOT, env=dict(os.environ, TMPDIR=str(output/'cache/tmp')), stdout=stream, stderr=subprocess.STDOUT)
    result['stages'].append({'stage': label, 'exit_code': process.returncode, 'log_path': str(log)})
    if process.returncode:
        result.update(status='failed', error='Stage '+label+' failed; see '+str(log))
    write_validation_json(output/'result.json', result)
    if process.returncode:
        raise RuntimeError(result['error'])

def run_backend_step(output, result, config_path, label, module, function, branch=None):
    """Execute one backend in a fresh process while preserving immutable preparation from another run."""
    extra = ', '+repr(branch) if branch else ''
    code = "import json,sys; from pathlib import Path; from joint_map."+module+" import "+function+"; from run_joint_map import write_json; c=json.loads(Path(sys.argv[1]).read_text()); r="+function+"(c,Path(sys.argv[2]),Path(sys.argv[3])"+extra+",write_json); assert r['status']=='complete', r.get('error')"
    import sys
    run_step(output, result, label, [sys.executable, '-B', '-u', '-c', code, str(config_path), str(PROJECT_ROOT), str(output)])

def finish_reconstruction(output, config, branch, result):
    """Verify this part's exports and publish its poses without reading any ground truth."""
    load_confirmed_handoff(output)
    export = json.loads((output/branch/'export/summary.json').read_text())
    for path, digest in export['input_sha256'].items():
        if get_sha256(Path(path)) != digest:
            raise ValueError('Reconstruction evidence changed: '+path)
    for name, path in export['artifacts'].items():
        if get_sha256(Path(path)) != export['output_sha256'][name]:
            raise ValueError('Reconstruction output changed: '+path)
    pose_path = output/branch/'export/poses_wgs84.txt'
    values = read_geographic_poses(pose_path, config['image_names'])
    fields = ['longitude_deg', 'latitude_deg', 'ellipsoidal_height_m', 'roll_deg', 'pitch_deg', 'yaw_deg']
    prior = json.loads((output/'shared/prior_model.json').read_text())
    result['cache_hits'] = [row['stage'] for row in result.get('cache_events', []) if row['hit']]
    result['cache_used'] = bool(result['cache_hits'])
    result.update(status='complete', current_stage='complete', poses=[dict(image_name=name, **dict(zip(fields, row.tolist()))) for name, row in zip(config['image_names'], values)], poses_wgs84=str(pose_path), sparse_cloud=str(output/branch/'export/sparse_wgs84.ply'), dense_cloud=str(output/branch/'export/dense_wgs84.ply'), updated_dsm=str(output/branch/'map/updated_dsm.tif'), selected_prior=prior['candidate_id'], config_sha256=get_sha256(output/'reconstruction.json'))
    write_validation_json(output/'result.json', result)
    print(result['part']+'完成。位姿文件：'+str(pose_path), flush=True)
    return result

