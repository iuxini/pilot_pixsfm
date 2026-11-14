# 用途：独立执行原版PixSfM及外部地图配准，仅导出原版分支，不执行正式改进版或GT评价。
import argparse
from pathlib import Path
import sys
from joint_map import modify_cache, workflow

def run_original(image_dir, dom_path, dsm_path, output_dir, intrinsics, mode='real'):
    """Start a new batch from raw inputs and preserve native relative geometry in the map baseline."""
    workflow.check_mode(mode)
    workflow.check_environment()
    if mode == 'real':
        workflow.guard_real_paths(output_dir, [image_dir, dom_path, dsm_path])
    batch_dir, output = workflow.create_run(output_dir, 'original', mode)
    state = {'schema_version': 2, 'mode': mode, 'batch_id': output.name, 'original': {'run_dir': str(output)}}
    workflow.write_validation_json(batch_dir/'batch.json', state)
    config = workflow.prepare_config(image_dir, dom_path, dsm_path, intrinsics)
    fingerprint = modify_cache.digest(modify_cache.input_identity(config))
    keys = modify_cache.original_keys(config, workflow.PROJECT_ROOT) if mode == 'modify' else None
    cache = modify_cache.ModifyCache(batch_dir, 'reconstruction') if mode == 'modify' else None
    native = output/'native'
    config.update(output_dir=str(output), native_sfm_dir=str(native/'sfm'), native_dense_dir=str(native/'dense'), workflow_mode=mode)
    config['frontend'] = {'features_path': str(native/'keypoints.h5'), 'adjusted_features_path': str(native/'sfm/refined_keypoints.h5'), 'matches_path': str(native/'matches.h5')}
    config['prior_validation']['selected_model_path'] = str(output/'shared/prior_model.json')
    config_path = output/'reconstruction.json'
    workflow.write_validation_json(config_path, config)
    result = {'status': 'running', 'part': 'original', 'batch_id': state['batch_id'], 'output_dir': str(output), 'branch': 'baseline', 'gt_inputs_read': False, 'accuracy_evaluated': False, 'data_artifacts_reused': False, 'external_map_registration': True, 'formal_joint_executed': False, 'stages': []}
    result.update(workflow.mode_fields(mode, fingerprint), dependency_fingerprints=keys)
    if cache:
        result['cache_events'] = cache.events
    print('原版本次目录：'+str(output), flush=True)
    baseline = cache.get('baseline', keys['baseline']) if cache else None
    prefix = cache.get('prefix', keys['prefix']) if cache and not baseline else None
    if baseline or prefix:
        source = Path((baseline or prefix)['source_dir'])
        for name in ['native', 'shared']+(['baseline'] if baseline else ['baseline/model']):
            (output/name).parent.mkdir(parents=True, exist_ok=True)
            (output/name).symlink_to(source/name, target_is_directory=True)
        for name in ['check.log', 'register.log']:
            (output/'logs'/name).symlink_to(source/'logs'/name)
        if not baseline:
            for label, module, function in [('dense_baseline', 'dense', 'run_dense_preparation'), ('update_baseline', 'dsm', 'run_dsm_update'), ('export_baseline', 'export', 'run_geographic_export')]:
                workflow.run_backend_step(output, result, config_path, label, module, function, 'baseline')
        result['data_artifacts_reused'] = True
        workflow.finish_reconstruction(output, config, 'baseline', result)
        # Keep the first complete source as the prefix; thin cache-hit runs omit its provisional files.
        if not baseline:
            cache.put('baseline', keys['baseline'], output, ['native', 'shared', 'baseline', 'logs/check.log', 'logs/register.log', 'reconstruction.json', 'result.json'])
        state['original']['result_sha256'] = workflow.get_sha256(output/'result.json')
        workflow.publish_batch(batch_dir, state)
        return result
    native_entry = cache.get('native', keys['native']) if cache else None
    if native_entry:
        native.symlink_to(native_entry['source_dir'], target_is_directory=True)
    else:
        workflow.run_step(output, result, 'native', [sys.executable, '-B', '-u', '-m', 'joint_map.native', '--config', str(config_path)])
        if cache:
            cache.put('native', keys['native'], native, ['.'])
    stages = [('check', None), ('baseline', None), ('initialize', None), ('validate_priors', None), ('register', None), ('dense', 'baseline'), ('update', 'baseline'), ('export', 'baseline')]
    for stage, branch in stages:
        if cache and stage == 'initialize':
            entry = cache.get('initialize', keys['initialize'])
            if entry:
                modify_cache.restore_initialize(entry, output)
                continue
        if cache and stage == 'validate_priors':
            entry = cache.get('features', keys['features'])
            if entry:
                (output/'shared/holdout_dense_features.h5').symlink_to(Path(entry['source_dir'])/'holdout_dense_features.h5')
        if stage == 'register':
            code = 'import sys; from pathlib import Path; from joint_map.workflow import select_automatic_prior; select_automatic_prior(Path(sys.argv[1]))'
            workflow.run_step(output, result, 'select_prior', [sys.executable, '-B', '-u', '-c', code, str(output)])
        command = [sys.executable, '-B', '-u', str(workflow.PROJECT_ROOT/'run_joint_map.py'), '--config', str(config_path), '--stage', stage]
        if branch:
            command += ['--branch', branch]
        workflow.run_step(output, result, stage+('_'+branch if branch else ''), command)
        if cache and stage == 'initialize':
            cache.snapshot_initialize(keys['initialize'], output)
        if cache and stage == 'validate_priors':
            cache.put('features', keys['features'], output/'shared', ['holdout_dense_features.h5'])
        if cache and stage == 'register':
            cache.put('prefix', keys['prefix'], output, modify_cache.PREFIX_FILES)
    result['data_artifacts_reused'] = bool(cache and any(row['hit'] for row in cache.events))
    workflow.finish_reconstruction(output, config, 'baseline', result)
    if cache:
        cache.put('baseline', keys['baseline'], output, ['native', 'shared', 'baseline', 'logs/check.log', 'logs/register.log', 'reconstruction.json', 'result.json'])
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
    parser.add_argument('--mode', choices=['real', 'modify'], default='real', help='real每次从输入重新生成；modify按阶段校验并复用开发缓存')
    args = parser.parse_args()
    run_original(args.image_dir, args.dom, args.dsm, args.output_dir, args.intrinsics, args.mode)
