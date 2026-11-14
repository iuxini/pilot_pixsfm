# 用途：独立读取同批两版完整结果和GT计算误差，不启动重建、配准、优化或DSM更新。
import argparse
import json
from pathlib import Path
from joint_map import modify_cache, workflow
from joint_map.evaluation import run_evaluation
from run_joint_map import get_latest_stage, write_json

def resolve_evaluation_sources(batch_dir, mode='real'):
    """Resolve a verified split batch or one explicit complete legacy run without fabricating handoffs."""
    batch_dir = Path(batch_dir).expanduser().resolve()
    workflow.check_mode(mode)
    if (batch_dir/'batch.json').exists():
        state, records = workflow.read_batch(batch_dir, ['original', 'improved'], mode)
        original, improved = [Path(records[part]['output_dir']) for part in ['original', 'improved']]
        if records['improved']['original_result_sha256'] != state['original']['result_sha256'] or Path(records['improved']['original_run_dir']) != original:
            raise ValueError('The two outputs do not share the same original initialization and map registration')
        return {'layout': 'split_batch', 'state': state, 'batch_id': state['batch_id'], 'original': original, 'improved': improved, 'source_sha256': {}}
    if not batch_dir.is_dir():
        raise FileNotFoundError('Result directory does not exist: '+str(batch_dir))
    if (batch_dir/'localization_result.json').is_file():
        candidates = [batch_dir]
    else:
        candidates = sorted(path for path in batch_dir.iterdir() if path.is_dir() and path.name.startswith('run_') and (path/'localization_result.json').is_file())
    if not candidates:
        raise FileNotFoundError('No batch.json or legacy localization_result.json found; specify a completed result directory')
    if len(candidates) != 1:
        raise ValueError('Multiple legacy runs found; set --output-dir to the exact run directory instead of choosing an older or newer run automatically')
    original = candidates[0]
    if mode == 'real':
        workflow.guard_real_paths(batch_dir, [original]+[original/name for name in ['baseline', 'joint', 'shared']])
    marker = json.loads((original/'localization_result.json').read_text())
    if mode == 'real' and (marker.get('mode', 'real') != 'real' or marker.get('cache_used', False)):
        raise ValueError('real mode rejects modify legacy results')
    if marker.get('status') != 'complete' or Path(marker['output_dir']).resolve() != original:
        raise ValueError('Legacy localization run is incomplete or belongs to a different directory')
    required = ['localization_result.json', 'reconstruction.json', 'run_manifest.json', 'shared/origin.json']
    required += [branch+'/'+name+'/summary.json' for branch in ['baseline', 'joint'] for name in ['export', 'map']]
    for name in required:
        if not (original/name).is_file():
            raise ValueError('Legacy localization artifact is missing: '+str(original/name))
    config = json.loads((original/'reconstruction.json').read_text())
    if mode == 'real':
        if config.get('workflow_mode', 'real') != 'real':
            raise ValueError('real mode rejects a modify legacy configuration')
        workflow.guard_real_paths(batch_dir, [workflow.PROJECT_ROOT/config[key] for key in ['image_dir', 'dom_path', 'dsm_path'] if key in config])
    manifest = json.loads((original/'run_manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Legacy localization manifest is incomplete')
    if 'config' in manifest and manifest['config']['sha256'] != workflow.get_sha256(original/'reconstruction.json'):
        raise ValueError('Legacy reconstruction configuration changed')
    for branch in ['baseline', 'joint']:
        stage = get_latest_stage(manifest, 'export', branch)
        if not stage or stage['status'] != 'complete' or any(json.loads((original/branch/name/'summary.json').read_text())['status'] != 'complete' for name in ['export', 'map']):
            raise ValueError('Complete both legacy branch exports before evaluation')
    source_sha256 = {str(original/name): workflow.get_sha256(original/name) for name in required}
    return {'layout': 'legacy_combined', 'state': None, 'batch_id': original.name, 'original': original, 'improved': original, 'source_sha256': source_sha256}

def evaluate_results(output_dir, gt_poses, gt_depth_dir, mode='real'):
    """Score two complete same-batch outputs read-only using the frozen full-pixel evaluation protocol."""
    workflow.check_environment()
    batch_dir = Path(output_dir).expanduser().resolve()
    sources = resolve_evaluation_sources(batch_dir, mode)
    state, original, improved = sources['state'], sources['original'], sources['improved']
    gt_poses, gt_depth_dir = [Path(value).expanduser().resolve() for value in [gt_poses, gt_depth_dir]]
    if not gt_poses.is_file() or not gt_depth_dir.is_dir():
        raise FileNotFoundError('Provide GT poses and the six corresponding camera-z depth files')
    if mode == 'real':
        workflow.guard_real_paths(batch_dir, [gt_poses, gt_depth_dir])
    _, output = workflow.create_run(batch_dir, 'evaluation', mode)
    if state is not None:
        state['evaluation'] = {'run_dir': str(output)}
        workflow.publish_batch(batch_dir, state)
    print('评价输入：'+sources['layout']+' | '+str(original)+' | '+str(improved), flush=True)
    for name, source in [('baseline', original/'baseline'), ('joint', improved/'joint'), ('shared', original/'shared')]:
        (output/name).symlink_to(source, target_is_directory=True)
    config = json.loads((original/'reconstruction.json').read_text())
    config.update(output_dir=str(output), workflow_mode=mode)
    config['prior_validation']['selected_model_path'] = str(output/'shared/prior_model.json')
    config_path = output/'reconstruction.json'
    workflow.write_validation_json(config_path, config)
    evaluation = json.loads((workflow.PROJECT_ROOT/'configs/joint_map_evaluation.json').read_text())
    evaluation.update(gt_pose_path=str(gt_poses), gt_depth_dir=str(gt_depth_dir), depth_rule_confirmation='独立评价入口固定协议：本次六组输入使用finite且depth>0的camera-z米制深度；不排除far或最大值。')
    evaluation_path = output/'evaluation_config.json'
    workflow.write_validation_json(evaluation_path, evaluation)
    key = modify_cache.reference_key(config, evaluation, original, workflow.PROJECT_ROOT)
    fields = workflow.mode_fields(mode, key)
    cache = modify_cache.ModifyCache(batch_dir, 'evaluation') if mode == 'modify' else None
    if cache:
        entry = cache.get('reference', key)
        if entry:
            modify_cache.restore_reference(entry, config, evaluation_path, config_path, workflow.PROJECT_ROOT, output)
        fields.update(cache_events=cache.events, cache_hits=[row['stage'] for row in cache.events if row['hit']], cache_used=entry is not None)
    report = run_evaluation(config, evaluation_path, config_path, workflow.PROJECT_ROOT, output, write_json)
    report.update(fields)
    write_json(output/'evaluation/report.json', report)
    if state is not None:
        workflow.read_batch(batch_dir, ['original', 'improved'], mode)
    for path, digest in sources['source_sha256'].items():
        if workflow.get_sha256(path) != digest:
            raise ValueError('Legacy localization metadata changed during evaluation: '+path)
    result = {'status': report['status'], 'part': 'evaluation', 'batch_id': sources['batch_id'], 'input_layout': sources['layout'], 'source_run_dirs': {'original': str(original), 'improved': str(improved)}, 'output_dir': str(output), 'config_sha256': workflow.get_sha256(config_path), 'gt_inputs_read': True, 'reconstruction_executed': False, 'alignment': 'same_batch_frozen_map_registration_no_GT_alignment', 'report_json': str(output/'evaluation/report.json'), 'report_markdown': str(output/'evaluation/report.md'), 'primary_acceptance_passed': report['primary_acceptance_passed'], 'delivery_complete': report['delivery_complete']}
    result.update(fields)
    workflow.write_validation_json(output/'result.json', result)
    if cache and report['status'] == 'complete':
        cache.put('reference', key, output/'evaluation', modify_cache.REFERENCE_FILES+['baseline_metrics.json'])
    if state is not None:
        state['evaluation']['result_sha256'] = workflow.get_sha256(output/'result.json')
        workflow.publish_batch(batch_dir, state)
    print('评价完成。报告：'+str(output/'evaluation/report.md'), flush=True)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只评价原版与改进版结果；GT仅在本入口读取')
    parser.add_argument('--output-dir', type=Path, required=True, help='新版批次根目录，或旧版已完成run目录/仅含一个旧run的父目录')
    parser.add_argument('--gt-poses', type=Path, required=True, help='六行 image_name lon lat ellipsoidal_height roll pitch yaw')
    parser.add_argument('--gt-depth-dir', type=Path, required=True, help='六份同名stem.npy，float32 camera-z深度，单位米')
    parser.add_argument('--mode', choices=['real', 'modify'], default='real')
    args = parser.parse_args()
    evaluate_results(args.output_dir, args.gt_poses, args.gt_depth_dir, args.mode)
