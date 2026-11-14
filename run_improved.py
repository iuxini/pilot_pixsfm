# 用途：独立执行改进版联合优化及后处理，只读取同批原版交接，不重跑原版或读取GT。
import argparse
import json
from pathlib import Path
import sys
from joint_map import workflow

def run_improved(output_dir):
    """Generate fresh improved outputs using the exact same batch's immutable native and map initialization."""
    workflow.check_environment()
    batch_dir = Path(output_dir).expanduser().resolve()
    state, records = workflow.read_batch(batch_dir, ['original'])
    original = Path(records['original']['output_dir'])
    workflow.load_confirmed_handoff(original)
    _, output = workflow.create_run(batch_dir, 'improved')
    state.update(improved={'run_dir': str(output)})
    state.pop('evaluation', None)
    workflow.publish_batch(batch_dir, state)
    config = json.loads((original/'reconstruction.json').read_text())
    config.update(output_dir=str(output))
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
    print('改进版本次目录：'+str(output), flush=True)
    jobs = [('joint', 'joint', 'run_joint_optimization', False), ('dense_joint', 'dense', 'run_dense_preparation', True), ('update_joint', 'dsm', 'run_dsm_update', True), ('export_joint', 'export', 'run_geographic_export', True)]
    for label, module, function, branch in jobs:
        extra = ", 'joint'" if branch else ''
        code = "import json,sys; from pathlib import Path; from joint_map."+module+" import "+function+"; from run_joint_map import write_json; c=json.loads(Path(sys.argv[1]).read_text()); r="+function+"(c,Path(sys.argv[2]),Path(sys.argv[3])"+extra+",write_json); assert r['status']=='complete', r.get('error')"
        workflow.run_step(output, result, label, [sys.executable, '-B', '-u', '-c', code, str(config_path), str(workflow.PROJECT_ROOT), str(output)])
    workflow.read_batch(batch_dir, ['original'])
    workflow.finish_reconstruction(output, config, 'joint', result)
    state['improved']['result_sha256'] = workflow.get_sha256(output/'result.json')
    workflow.publish_batch(batch_dir, state)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='仅执行同批改进版；不重跑原版，不读取GT')
    parser.add_argument('--output-dir', type=Path, required=True, help='第一步使用的同一批输出根目录')
    args = parser.parse_args()
    run_improved(args.output_dir)
