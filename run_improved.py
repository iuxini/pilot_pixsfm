# 用途：独立执行改进版联合优化及后处理，只读取同批原版交接，不重跑原版或读取GT。
import argparse
import json
from pathlib import Path
from joint_map import modify_cache, workflow

def run_improved(output_dir, mode='real'):
    """Generate fresh improved outputs using the exact same batch's immutable native and map initialization."""
    workflow.check_environment()
    batch_dir = Path(output_dir).expanduser().resolve()
    state, records = workflow.read_batch(batch_dir, ['original'], mode)
    original = Path(records['original']['output_dir'])
    config = json.loads((original/'reconstruction.json').read_text())
    if mode == 'modify':
        current = workflow.prepare_config(config['image_dir'], config['dom_path'], config['dsm_path'], config['camera']['params'])
        keys = modify_cache.original_keys(current, workflow.PROJECT_ROOT)
        if records['original'].get('dependency_fingerprints') != keys:
            raise ValueError('Preparation dependencies changed; rerun run_original.py --mode modify to regenerate affected stages')
    elif records['original'].get('input_fingerprint') and modify_cache.digest(modify_cache.input_identity(config)) != records['original']['input_fingerprint']:
        raise ValueError('Raw inputs changed after the original run; start a new real batch')
    workflow.load_confirmed_handoff(original)
    _, output = workflow.create_run(batch_dir, 'improved', mode)
    state.update(improved={'run_dir': str(output)})
    state.pop('evaluation', None)
    workflow.publish_batch(batch_dir, state)
    config.update(output_dir=str(output), workflow_mode=mode)
    config['prior_validation']['selected_model_path'] = str(output/'shared/prior_model.json')
    config_path = output/'reconstruction.json'
    workflow.write_validation_json(config_path, config)
    (output/'shared').symlink_to(original/'shared', target_is_directory=True)
    (output/'baseline').mkdir()
    (output/'baseline/model').symlink_to(original/'baseline/model', target_is_directory=True)
    for name in ['check.log', 'register.log']:
        (output/'logs'/name).symlink_to(original/'logs'/name)
    (output/'joint/model').mkdir(parents=True)
    result = {'status': 'running', 'part': 'improved', 'batch_id': state['batch_id'], 'output_dir': str(output), 'original_run_dir': str(original), 'original_result_sha256': state['original']['result_sha256'], 'branch': 'joint', 'gt_inputs_read': False, 'accuracy_evaluated': False, 'initialization_source': 'same_batch_original_and_frozen_map_registration', 'stages': []}
    result.update(workflow.mode_fields(mode, records['original'].get('input_fingerprint')))
    if mode == 'modify':
        result['cache_events'] = [{'stage': 'shared_preparation', 'hit': True, 'reason': 'same_batch_verified', 'source': str(original)}]
    print('改进版本次目录：'+str(output), flush=True)
    jobs = [('joint', 'joint', 'run_joint_optimization', False), ('dense_joint', 'dense', 'run_dense_preparation', True), ('update_joint', 'dsm', 'run_dsm_update', True), ('export_joint', 'export', 'run_geographic_export', True)]
    for label, module, function, branch in jobs:
        workflow.run_backend_step(output, result, config_path, label, module, function, 'joint' if branch else None)
    workflow.read_batch(batch_dir, ['original'], mode)
    workflow.finish_reconstruction(output, config, 'joint', result)
    state['improved']['result_sha256'] = workflow.get_sha256(output/'result.json')
    workflow.publish_batch(batch_dir, state)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='仅执行同批改进版；不重跑原版，不读取GT')
    parser.add_argument('--output-dir', type=Path, required=True, help='第一步使用的同一批输出根目录')
    parser.add_argument('--mode', choices=['real', 'modify'], default='real')
    args = parser.parse_args()
    run_improved(args.output_dir, args.mode)
