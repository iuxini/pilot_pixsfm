# 用途：验证裸输入自动定位、新内参、每次重新生成、自动候选选择和失败停点。
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from joint_map import workflow as entry
from run_original import run_original

class TestRawLocalization(unittest.TestCase):
    """Check the raw-input delivery contract without creating reusable data artifacts."""
    @classmethod
    def setUpClass(cls):
        """Generate independent raw fixtures without requiring the historical dataset or native models."""
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root/'images').mkdir()
        for index in range(6):
            Image.new('RGB', (3840, 2160), (index, 20, 30)).save(root/'images'/('frame_'+str(index)+'.png'))
        for name, count, dtype, nodata in [('dom', 3, 'int16', 255), ('dsm', 1, 'float32', -9999)]:
            with rasterio.open(root/(name+'.tif'), 'w', driver='GTiff', height=5, width=7, count=count, dtype=dtype, nodata=nodata, crs='EPSG:4326', transform=from_origin(112.99, 28.29, 0.00001, 0.00001)) as dataset:
                dataset.write(np.full((count, 5, 7), 70, dtype=dtype))
        cls.inputs = [root/'images', root/'dom.tif', root/'dsm.tif']

    def setUp(self):
        """Create isolated per-test outputs for the synthetic raw inputs."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.intrinsics = [2711., 2699., 1917., 1082.]

    def test_prepare_raw_inputs_uses_only_supplied_data_and_intrinsics(self):
        """Remove old intermediate paths and accept another fixed camera calibration."""
        config = entry.prepare_config(*self.inputs, self.intrinsics)
        self.assertEqual(config['camera']['params'], self.intrinsics)
        self.assertEqual(config['image_names'], sorted(config['image_names']))
        self.assertEqual(len(config['image_names']), 6)
        for key in ['native_sfm_dir', 'native_dense_dir', 'output_dir']:
            self.assertNotIn(key, config)
        self.assertEqual(config['frontend'], {})
        self.assertNotIn('selected_model_path', config['prior_validation'])

    def test_reject_invalid_intrinsics_before_creating_run(self):
        """Reject invalid focal lengths and nonfinite or malformed calibration."""
        for params in [[0, 2700, 1920, 1080], [-1, 2700, 1920, 1080], [2700, 2700, float('nan'), 1080], [2700, 2700, 1920]]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                entry.prepare_config(*self.inputs, params)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_reject_non_six_image_input_before_extraction(self):
        """Fail on an incorrect raw image count before invoking any GPU model."""
        for index in range(5):
            (self.output/(str(index)+'.png')).write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'six unique'):
            entry.prepare_config(self.output, *self.inputs[1:], self.intrinsics)

    def test_refuse_existing_native_workspace(self):
        """Never reuse a completed or partial native workspace in the raw delivery path."""
        from joint_map.native import prepare_native_inputs
        (self.output/'native').mkdir()
        with self.assertRaisesRegex(ValueError, 'must be new'):
            prepare_native_inputs({'native_sfm_dir': str(self.output/'native/sfm')}, entry.PROJECT_ROOT)

    def test_automatic_selection_uses_passed_candidates_and_no_gt(self):
        """Apply the declared proposal and fallback order without asking for extra user inputs."""
        shared = self.output/'shared'
        shared.mkdir()
        for proposed, passed, expected in [(None, ['weight_0p1', 'weight_1p0', 'weight_10p0'], 'weight_1p0'), ('weight_10p0', ['weight_1p0', 'weight_10p0'], 'weight_10p0'), (None, ['weight_0p1', 'weight_10p0'], 'weight_0p1'), (None, ['weight_10p0'], 'weight_10p0')]:
            with self.subTest(proposed=proposed, passed=passed):
                report = {'status': 'complete', 'selection_status': 'proposed' if proposed else 'indistinguishable', 'candidate_id': proposed, 'candidates': [{'candidate_id': name, 'status': 'passed' if name in passed else 'failed'} for name in ['weight_0p1', 'weight_1p0', 'weight_10p0']]}
                (shared/'prior_validation.json').write_text(json.dumps(report))
                with patch.object(entry, 'confirm_prior_candidate', return_value={'candidate_id': expected}) as confirm:
                    prior = entry.select_automatic_prior(self.output)
                self.assertEqual(confirm.call_args.args[1], expected)
                self.assertEqual(prior['selection_basis'], 'automatic_no_gt_v1')
                self.assertFalse(prior['automation_policy']['gt_inputs_read'])

    def test_tensor_multiprocessing_supports_long_output_paths(self):
        """Transfer a real tensor across processes while keeping sockets in a long output path."""
        import subprocess
        import sys
        output = self.output/('long_output_'*12)
        temporary = output/'cache/tmp'
        temporary.mkdir(parents=True)
        code = '''import multiprocessing as mp, sys, torch
from joint_map.native import configure_native_process
configure_native_process(sys.argv[1])
context = mp.get_context('fork')
queue, event = context.Queue(), context.Event()
def send_tensor():
    queue.put(torch.arange(3))
    event.wait(15)
process = context.Process(target=send_tensor)
process.start()
value = queue.get(timeout=15)
event.set()
process.join(15)
assert value.tolist() == [0, 1, 2] and process.exitcode == 0
print('Long output path tensor IPC: PASS')
'''
        process = subprocess.run([sys.executable, '-B', '-c', code, str(output)], cwd=entry.PROJECT_ROOT, env=dict(os.environ, TMPDIR=str(temporary)), capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        self.assertIn('Long output path tensor IPC: PASS', process.stdout)

    def test_automatic_selection_rejects_no_valid_candidate(self):
        """Stop rather than use an unvalidated prior or bypass failed evidence."""
        (self.output/'shared').mkdir()
        (self.output/'shared/prior_validation.json').write_text(json.dumps({'status': 'complete', 'candidates': [{'candidate_id': 'weight_1p0', 'status': 'failed'}]}))
        with patch.object(entry, 'confirm_prior_candidate') as confirm, self.assertRaisesRegex(ValueError, 'No completed passing'):
            entry.select_automatic_prior(self.output)
        confirm.assert_not_called()

    def test_native_failure_stops_and_reports_failed_run(self):
        """Never start downstream stages after native preparation fails."""
        with patch.object(entry, 'check_environment'), patch.object(entry.subprocess, 'run', return_value=SimpleNamespace(returncode=7)) as process:
            with self.assertRaisesRegex(RuntimeError, 'native failed'):
                run_original(*self.inputs, self.output, self.intrinsics)
        process.assert_called_once()
        runs = list(self.output.glob('real/original/run_*'))
        result = json.loads((runs[0]/'result.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['current_stage'], 'native')
        self.assertEqual(process.call_args.kwargs['env']['TMPDIR'], str(runs[0]/'cache/tmp'))

    def test_original_invocations_are_fresh_and_never_start_improved_or_gt(self):
        """Generate separate original runs without starting formal joint or evaluating either version."""
        calls = []
        def simulate(command, **kwargs):
            calls.append(command)
            if '--config' in command:
                config_path = Path(command[command.index('--config')+1])
                config = json.loads(config_path.read_text())
                output = Path(config['output_dir'])
                self.assertEqual(config['camera']['params'], self.intrinsics)
                for value in config['frontend'].values():
                    self.assertTrue(Path(value).is_relative_to(output))
                self.assertEqual(config['prior_validation']['selected_model_path'], str(output/'shared/prior_model.json'))
                if '--stage' in command and command[command.index('--stage')+1] == 'export' and command[-1] == 'baseline':
                    folder = output/'baseline/export'
                    folder.mkdir(parents=True)
                    (folder/'summary.json').write_text(json.dumps({'input_sha256': {}, 'artifacts': {}, 'output_sha256': {}}))
                    (folder/'poses_wgs84.txt').write_text(''.join(name+' 113 28 100 0 0 0\n' for name in config['image_names']))
            if '-c' in command:
                shared = Path(command[-1])/'shared'
                shared.mkdir()
                (shared/'prior_model.json').write_text(json.dumps({'candidate_id': 'weight_1p0', 'lambda_map': 1., 'selection_basis': 'automatic_no_gt_v1'}))
            return SimpleNamespace(returncode=0)
        with patch.object(entry, 'check_environment'), patch.object(entry.subprocess, 'run', side_effect=simulate), patch.object(entry, 'load_confirmed_handoff'):
            first = run_original(*self.inputs, self.output, self.intrinsics)
            second = run_original(*self.inputs, self.output, self.intrinsics)
        self.assertNotEqual(first['output_dir'], second['output_dir'])
        self.assertEqual(len(calls), 20)
        self.assertFalse(any('--stage' in call and call[call.index('--stage')+1] in ['joint', 'evaluate', 'all'] for call in calls))
        self.assertFalse(any('--branch' in call and call[-1] == 'joint' for call in calls))
        for result in [first, second]:
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['stages']), 10)
            self.assertEqual(len(result['poses']), 6)
            self.assertFalse(result['data_artifacts_reused'])
            self.assertFalse(result['accuracy_evaluated'])
            self.assertFalse(result['gt_inputs_read'])

if __name__ == '__main__':
    unittest.main()
