# 用途：验证三个独立入口的边界、同批交接、只读评价及旧批次拒绝。
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from joint_map import workflow
from run_improved import run_improved
from evaluate_results import evaluate_results
import test_joint_map

class TestWorkflowParts(unittest.TestCase):
    """Verify stage isolation with independent small six-image ground-truth fixtures."""
    def setUp(self):
        """Create two completed same-batch outputs without historical files or reconstruction jobs."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.batch = Path(self.temporary.name)
        fixture = self.batch/'original/run_fixture'
        fixture.mkdir(parents=True)
        config, _, _, self.original = test_joint_map.TestGroundTruthEvaluation().create_fixture(fixture)
        config.update(image_dir=str(fixture/'images'), dsm_path=str(fixture/'original.tif'), output_dir=str(self.original), prior_validation={'selected_model_path': str(self.original/'shared/prior_model.json')})
        workflow.write_validation_json(self.original/'reconstruction.json', config)
        workflow.write_validation_json(self.original/'shared/prior_model.json', {'candidate_id': 'weight_1p0'})
        for name in ['check.log', 'register.log']:
            workflow.write_validation_json(self.original/'logs'/name, {'status': 'complete'})
        original = {'status': 'complete', 'part': 'original', 'branch': 'baseline', 'batch_id': 'fixture_batch', 'output_dir': str(self.original), 'config_sha256': workflow.get_sha256(self.original/'reconstruction.json')}
        workflow.write_validation_json(self.original/'result.json', original)
        self.improved = self.batch/'improved/run_fixture'
        self.improved.mkdir(parents=True)
        (self.improved/'joint').symlink_to(self.original/'joint', target_is_directory=True)
        workflow.write_validation_json(self.improved/'reconstruction.json', config)
        improved = {'status': 'complete', 'part': 'improved', 'branch': 'joint', 'batch_id': 'fixture_batch', 'output_dir': str(self.improved), 'config_sha256': workflow.get_sha256(self.improved/'reconstruction.json'), 'original_run_dir': str(self.original), 'original_result_sha256': workflow.get_sha256(self.original/'result.json')}
        workflow.write_validation_json(self.improved/'result.json', improved)
        self.state = {'schema_version': 1, 'batch_id': 'fixture_batch', 'original': {'run_dir': str(self.original), 'result_sha256': workflow.get_sha256(self.original/'result.json')}, 'improved': {'run_dir': str(self.improved), 'result_sha256': workflow.get_sha256(self.improved/'result.json')}}
        workflow.write_validation_json(self.batch/'batch.json', self.state)
        self.gt_poses, self.gt_depth = fixture/'truth.txt', fixture/'images'

    def fingerprint_source(self):
        """Fingerprint both producer directories to catch evaluator or optimizer writes through links."""
        return {str(path): workflow.get_sha256(path) for root in [self.original, self.improved] for path in root.rglob('*') if path.is_file()}

    def test_evaluation_scores_all_metrics_without_reconstruction_or_input_writes(self):
        """Run real evaluation on all valid fixture pixels and leave both producer outputs unchanged."""
        before = self.fingerprint_source()
        with patch.object(workflow, 'check_environment'), patch.object(workflow.subprocess, 'run', side_effect=AssertionError('Evaluation must not launch reconstruction')):
            result = evaluate_results(self.batch, self.gt_poses, self.gt_depth)
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['reconstruction_executed'])
        self.assertEqual(before, self.fingerprint_source())
        report = json.loads(Path(result['report_json']).read_text())
        self.assertEqual(report['reference_count'], 148)
        self.assertEqual(len(report['primary_comparisons']), 8)
        self.assertEqual(report['baseline_metrics']['valid_pose_count'], 6)
        self.assertEqual(report['joint_metrics']['valid_pose_count'], 6)
        self.assertTrue(report['primary_acceptance_passed'])

    def test_evaluation_invocations_generate_fresh_score_directories(self):
        """Never reuse earlier GT references or frozen baseline errors on another evaluation invocation."""
        with patch.object(workflow, 'check_environment'):
            first = evaluate_results(self.batch, self.gt_poses, self.gt_depth)
            second = evaluate_results(self.batch, self.gt_poses, self.gt_depth)
        self.assertNotEqual(first['output_dir'], second['output_dir'])
        self.assertTrue(Path(first['report_json']).is_file())
        self.assertTrue(Path(second['report_json']).is_file())

    def test_modified_or_incomplete_original_blocks_later_parts(self):
        """Reject the latest incomplete batch rather than silently finding an older completed run."""
        self.state['original'].pop('result_sha256')
        workflow.write_validation_json(self.batch/'batch.json', self.state)
        with patch.object(workflow, 'check_environment'), patch.object(workflow.subprocess, 'run') as process:
            with self.assertRaisesRegex(ValueError, 'original successfully'):
                run_improved(self.batch)
            with self.assertRaisesRegex(ValueError, 'original successfully'):
                evaluate_results(self.batch, self.gt_poses, self.gt_depth)
        process.assert_not_called()

    def test_different_batch_or_initialization_blocks_evaluation(self):
        """Prevent mixing complete results with another original cohort even if their files look valid."""
        data = json.loads((self.improved/'result.json').read_text())
        data['original_result_sha256'] = 'different_original'
        workflow.write_validation_json(self.improved/'result.json', data)
        self.state['improved']['result_sha256'] = workflow.get_sha256(self.improved/'result.json')
        workflow.write_validation_json(self.batch/'batch.json', self.state)
        with patch.object(workflow, 'check_environment'), self.assertRaisesRegex(ValueError, 'same original initialization'):
            evaluate_results(self.batch, self.gt_poses, self.gt_depth)

    def test_improved_runs_only_its_four_jobs_and_preserves_original(self):
        """Run only joint/dense/update/export for the improved version using immutable shared inputs."""
        before = self.fingerprint_source()
        commands = []
        def simulate(command, **kwargs):
            commands.append(command)
            code = command[command.index('-c')+1]
            output = Path(command[-1])
            if 'run_geographic_export' in code:
                folder = output/'joint/export'
                folder.mkdir(parents=True)
                pose_path = folder/'poses_wgs84.txt'
                pose_path.write_bytes((self.original/'baseline/export/poses_wgs84.txt').read_bytes())
                workflow.write_validation_json(folder/'summary.json', {'input_sha256': {}, 'artifacts': {'poses': str(pose_path)}, 'output_sha256': {'poses': workflow.get_sha256(pose_path)}})
            return SimpleNamespace(returncode=0)
        with patch.object(workflow, 'check_environment'), patch.object(workflow, 'load_confirmed_handoff'), patch.object(workflow.subprocess, 'run', side_effect=simulate):
            result = run_improved(self.batch)
        self.assertEqual(len(commands), 4)
        self.assertEqual([row['stage'] for row in result['stages']], ['joint', 'dense_joint', 'update_joint', 'export_joint'])
        self.assertFalse(result['gt_inputs_read'])
        self.assertEqual(before, self.fingerprint_source())
        self.assertEqual((Path(result['output_dir'])/'shared').resolve(), self.original/'shared')
        self.assertEqual(result['original_result_sha256'], self.state['original']['result_sha256'])

if __name__ == '__main__':
    unittest.main()
