# 用途：独立读取同批两版完整结果和GT计算误差，不启动重建、配准、优化或DSM更新。
import argparse
import json
from pathlib import Path
from joint_map import workflow
from joint_map.evaluation import run_evaluation
from run_joint_map import write_json

def evaluate_results(output_dir, gt_poses, gt_depth_dir):
    """Score two complete same-batch outputs read-only using the frozen full-pixel evaluation protocol."""
    workflow.check_environment()
    batch_dir = Path(output_dir).expanduser().resolve()
    state, records = workflow.read_batch(batch_dir, ['original', 'improved'])
    original, improved = [Path(records[part]['output_dir']) for part in ['original', 'improved']]
    if records['improved']['original_result_sha256'] != state['original']['result_sha256'] or Path(records['improved']['original_run_dir']) != original:
        raise ValueError('The two outputs do not share the same original initialization and map registration')
    gt_poses, gt_depth_dir = [Path(value).expanduser().resolve() for value in [gt_poses, gt_depth_dir]]
    if not gt_poses.is_file() or not gt_depth_dir.is_dir():
        raise FileNotFoundError('Provide GT poses and the six corresponding camera-z depth files')
    _, output = workflow.create_run(batch_dir, 'evaluation')
    state['evaluation'] = {'run_dir': str(output)}
    workflow.publish_batch(batch_dir, state)
    for name, source in [('baseline', original/'baseline'), ('joint', improved/'joint'), ('shared', original/'shared')]:
        (output/name).symlink_to(source, target_is_directory=True)
    config = json.loads((original/'reconstruction.json').read_text())
    config['output_dir'] = str(output)
    config['prior_validation']['selected_model_path'] = str(output/'shared/prior_model.json')
    config_path = output/'reconstruction.json'
    workflow.write_validation_json(config_path, config)
    evaluation = json.loads((workflow.PROJECT_ROOT/'configs/joint_map_evaluation.json').read_text())
    evaluation.update(gt_pose_path=str(gt_poses), gt_depth_dir=str(gt_depth_dir), depth_rule_confirmation='独立评价入口固定协议：本次六组输入使用finite且depth>0的camera-z米制深度；不排除far或最大值。')
    evaluation_path = output/'evaluation_config.json'
    workflow.write_validation_json(evaluation_path, evaluation)
    report = run_evaluation(config, evaluation_path, config_path, workflow.PROJECT_ROOT, output, write_json)
    workflow.read_batch(batch_dir, ['original', 'improved'])
    result = {'status': report['status'], 'part': 'evaluation', 'batch_id': state['batch_id'], 'output_dir': str(output), 'config_sha256': workflow.get_sha256(config_path), 'gt_inputs_read': True, 'reconstruction_executed': False, 'alignment': 'same_batch_frozen_map_registration_no_GT_alignment', 'report_json': str(output/'evaluation/report.json'), 'report_markdown': str(output/'evaluation/report.md'), 'primary_acceptance_passed': report['primary_acceptance_passed'], 'delivery_complete': report['delivery_complete']}
    workflow.write_validation_json(output/'result.json', result)
    state['evaluation']['result_sha256'] = workflow.get_sha256(output/'result.json')
    workflow.publish_batch(batch_dir, state)
    print('评价完成。报告：'+str(output/'evaluation/report.md'), flush=True)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只评价原版与改进版结果；GT仅在本入口读取')
    parser.add_argument('--output-dir', type=Path, required=True, help='前两步使用的同一批输出根目录')
    parser.add_argument('--gt-poses', type=Path, required=True, help='六行 image_name lon lat ellipsoidal_height roll pitch yaw')
    parser.add_argument('--gt-depth-dir', type=Path, required=True, help='六份同名stem.npy，float32 camera-z深度，单位米')
    args = parser.parse_args()
    evaluate_results(args.output_dir, args.gt_poses, args.gt_depth_dir)
