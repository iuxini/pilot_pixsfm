# 用途：独立执行原版PixSfM及外部地图配准，仅导出原版分支，不执行正式改进版或GT评价。
import argparse
from pathlib import Path
import sys
from joint_map import workflow

def run_original(image_dir, dom_path, dsm_path, output_dir, intrinsics):
    """Start a new batch from raw inputs and preserve native relative geometry in the map baseline."""
    workflow.check_environment()
    batch_dir, output = workflow.create_run(output_dir, 'original')
    state = {'schema_version': 1, 'batch_id': output.name, 'original': {'run_dir': str(output)}}
    workflow.write_validation_json(batch_dir/'batch.json', state)
    config = workflow.prepare_config(image_dir, dom_path, dsm_path, intrinsics)
    native = output/'native'
    config.update(output_dir=str(output), native_sfm_dir=str(native/'sfm'), native_dense_dir=str(native/'dense'))
    config['frontend'] = {'features_path': str(native/'keypoints.h5'), 'adjusted_features_path': str(native/'sfm/refined_keypoints.h5'), 'matches_path': str(native/'matches.h5')}
    config['prior_validation']['selected_model_path'] = str(output/'shared/prior_model.json')
    config_path = output/'reconstruction.json'
    workflow.write_validation_json(config_path, config)
    result = {'status': 'running', 'part': 'original', 'batch_id': state['batch_id'], 'output_dir': str(output), 'branch': 'baseline', 'gt_inputs_read': False, 'accuracy_evaluated': False, 'data_artifacts_reused': False, 'external_map_registration': True, 'formal_joint_executed': False, 'stages': []}
    print('原版本次目录：'+str(output), flush=True)
    workflow.run_step(output, result, 'native', [sys.executable, '-B', '-u', '-m', 'joint_map.native', '--config', str(config_path)])
    stages = [('check', None), ('baseline', None), ('initialize', None), ('validate_priors', None), ('register', None), ('dense', 'baseline'), ('update', 'baseline'), ('export', 'baseline')]
    for stage, branch in stages:
        if stage == 'register':
            code = 'import sys; from pathlib import Path; from joint_map.workflow import select_automatic_prior; select_automatic_prior(Path(sys.argv[1]))'
            workflow.run_step(output, result, 'select_prior', [sys.executable, '-B', '-u', '-c', code, str(output)])
        command = [sys.executable, '-B', '-u', str(workflow.PROJECT_ROOT/'run_joint_map.py'), '--config', str(config_path), '--stage', stage]
        if branch:
            command += ['--branch', branch]
        workflow.run_step(output, result, stage+('_'+branch if branch else ''), command)
    workflow.finish_reconstruction(output, config, 'baseline', result)
    state['original']['result_sha256'] = workflow.get_sha256(output/'result.json')
    workflow.publish_batch(batch_dir, state)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='原版PixSfM + 独立外部地图配准；不执行正式改进版和评价')
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--dom', type=Path, required=True)
    parser.add_argument('--dsm', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='本批输出根目录，后两个入口使用相同目录')
    parser.add_argument('--intrinsics', nargs=4, type=float, required=True, metavar=('FX', 'FY', 'CX', 'CY'))
    args = parser.parse_args()
    run_original(args.image_dir, args.dom, args.dsm, args.output_dir, args.intrinsics)
