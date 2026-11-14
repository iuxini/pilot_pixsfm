# 用途：用独立小数据验证开发缓存依赖、真实模式隔离和只复用基线评价参考。
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from joint_map import evaluation, modify_cache, workflow
from run_original import run_original
from run_improved import run_improved
from evaluate_results import evaluate_results, resolve_evaluation_sources
import test_workflow_parts

class TestModifyReconstruction(unittest.TestCase):
    """Verify orchestration and byte-bound invalidation with mocked expensive backends."""
    def setUp(self):
        """Build independent tiny raw inputs and simulate only the expensive reconstruction kernels."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.batch = self.root/'batch'
        self.images = self.root/'images'
        self.images.mkdir()
        self.config = json.loads((workflow.PROJECT_ROOT/'configs/joint_map.json').read_text())
        self.config.update(image_dir=str(self.images), image_names=['frame_'+str(i)+'.png' for i in range(6)], dom_path=str(self.root/'dom.tif'), dsm_path=str(self.root/'dsm.tif'))
        for name in self.config['image_names']:
            (self.images/name).write_bytes(name.encode())
        for name in ['dom', 'dsm']:
            (self.root/(name+'.tif')).write_bytes(name.encode())
        self.tokens = {name: name+'_v1' for name in ['native', 'initialize', 'features', 'prior', 'baseline']}
        self.commands = []
        for context in [patch.object(workflow, 'check_environment'), patch.object(workflow, 'load_confirmed_handoff'), patch.object(workflow, 'prepare_config', side_effect=self.prepare), patch.object(modify_cache, 'dependency_tokens', side_effect=lambda *args: copy.deepcopy(self.tokens)), patch.object(workflow.subprocess, 'run', side_effect=self.simulate)]:
            context.start()
            self.addCleanup(context.stop)

    def prepare(self, images, dom, dsm, intrinsics):
        """Keep input identities real while bypassing full-resolution image/map validation in backend mocks."""
        config = copy.deepcopy(self.config)
        config.update(image_dir=str(images), dom_path=str(dom), dsm_path=str(dsm))
        config['camera']['params'] = list(intrinsics)
        return config

    def write(self, output, name, value=b'fixture'):
        """Write one small artifact with the same dependency location as its actual producer."""
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, dict):
            workflow.write_validation_json(path, value)
        else:
            path.write_bytes(value)

    def simulate(self, command, **kwargs):
        """Publish tiny completed-stage artifacts and enforce predecessor and mutable-snapshot behavior."""
        if '--config' in command:
            config = json.loads(Path(command[command.index('--config')+1]).read_text())
            output = Path(config['output_dir'])
            label = command[command.index('--stage')+1] if '--stage' in command else 'native'
        else:
            output = Path(command[-1])
            config = json.loads((output/'reconstruction.json').read_text())
            code = command[command.index('-c')+1]
            label = next((name for text, name in [('select_automatic_prior', 'select_prior'), ('run_joint_optimization', 'joint'), ('run_dense_preparation', 'dense'), ('run_dsm_update', 'update'), ('run_geographic_export', 'export')] if text in code))
        self.commands.append(label)
        if label == 'native':
            for name in ['sfm/images.bin', 'sfm/refined_keypoints.h5', 'dense/summary.json', 'keypoints.h5', 'matches.h5']:
                self.write(output/'native', name)
        elif label == 'check':
            self.write(output, 'shared/origin.json', {'origin': 'same_map'})
            self.write(output, 'run_manifest.json', {'stages': [{'stage': 'check', 'status': 'complete'}]})
            self.write(output, 'logs/check.log', {'status': 'complete'})
        elif label == 'baseline':
            self.write(output, 'logs/baseline.log', {'status': 'complete'})
        elif label == 'initialize':
            for name in modify_cache.INITIAL_FILES:
                self.write(output, name, b'provisional')
            self.write(output, 'logs/initialize.log', {'status': 'complete'})
            manifest = json.loads((output/'run_manifest.json').read_text())
            manifest['stages'].append({'stage': 'initialize', 'status': 'complete'})
            self.write(output, 'run_manifest.json', manifest)
        elif label == 'validate_priors':
            manifest = json.loads((output/'run_manifest.json').read_text())
            assert any(row['stage'] == 'initialize' and row['status'] == 'complete' for row in manifest['stages'])
            assert (output/'shared/track_map_links.npz').read_bytes() == b'provisional'
            self.write(output, 'shared/track_map_links.npz', b'training_frozen')
            if not (output/'shared/holdout_dense_features.h5').exists():
                self.write(output, 'shared/holdout_dense_features.h5')
            self.write(output, 'logs/validate_priors.log', {'status': 'complete'})
        elif label == 'select_prior':
            self.write(output, 'shared/prior_model.json', {'candidate_id': 'weight_1p0'})
        elif label == 'register':
            self.write(output, 'shared/registration.json', {'status': 'confirmed'})
            for branch in ['baseline', 'joint']:
                self.write(output, branch+'/model/images.bin', b'confirmed')
            self.write(output, 'logs/register.log', {'status': 'complete'})
        elif label in ['joint', 'dense', 'update']:
            branch = 'joint' if config['output_dir'].find('/improved/') >= 0 else 'baseline'
            self.write(output, branch+'/'+{'joint': 'model/images.bin', 'dense': 'dense/summary.json', 'update': 'map/updated_dsm.tif'}[label])
        elif label == 'export':
            branch = 'joint' if '/improved/' in str(output) else 'baseline'
            self.write(output, branch+'/export/poses_wgs84.txt', ''.join(name+' 113 28 100 0 0 0\n' for name in config['image_names']).encode())
            self.write(output, branch+'/export/summary.json', {'input_sha256': {}, 'artifacts': {}, 'output_sha256': {}})
        return SimpleNamespace(returncode=0)

    def run_original(self, mode='modify', intrinsics=None):
        """Invoke the public entry using only raw paths, K and the explicit execution mode."""
        return run_original(self.images, self.root/'dom.tif', self.root/'dsm.tif', self.batch, intrinsics or [2700., 2700., 1920., 1080.], mode)

    def test_modify_repeat_hits_baseline_without_any_backend(self):
        """A second identical invocation uses checked artifacts while preserving a new batch/result directory."""
        first = self.run_original()
        before = {str(path): workflow.get_sha256(path) for path in Path(first['output_dir']).rglob('*') if path.is_file()}
        self.commands.clear()
        second = self.run_original()
        self.assertEqual(self.commands, [])
        self.assertNotEqual(first['output_dir'], second['output_dir'])
        self.assertFalse(first['cache_used'])
        self.assertEqual(second['cache_hits'], ['baseline'])
        for path, value in before.items():
            self.assertEqual(workflow.get_sha256(path), value)
        workflow.read_batch(self.batch, ['original'], 'modify')

    def test_map_change_keeps_native_and_features_but_rebuilds_registration(self):
        """Map bytes invalidate matching, common calibration and baseline while image-only stages remain reusable."""
        self.run_original()
        (self.root/'dsm.tif').write_bytes(b'new map')
        self.commands.clear()
        result = self.run_original()
        self.assertEqual(result['cache_hits'], ['native', 'features'])
        self.assertNotIn('native', self.commands)
        self.assertIn('initialize', self.commands)
        self.assertIn('validate_priors', self.commands)

    def test_prior_change_restores_mutable_initialization_snapshot(self):
        """Calibration changes reuse pre-validation associations without reusing the mutated final models."""
        self.run_original()
        self.tokens['prior'] = 'prior_v2'
        self.commands.clear()
        result = self.run_original()
        self.assertEqual(result['cache_hits'], ['native', 'initialize', 'features'])
        self.assertNotIn('initialize', self.commands)
        self.assertIn('validate_priors', self.commands)

    def test_baseline_change_reuses_prefix_and_only_repeats_postprocessing(self):
        """Backend edits regenerate baseline outputs without repeating native work or prior calibration."""
        self.run_original()
        self.tokens['baseline'] = 'baseline_v2'
        self.commands.clear()
        result = self.run_original()
        self.assertEqual(result['cache_hits'], ['prefix'])
        self.assertEqual(self.commands, ['dense', 'update', 'export'])
        self.commands.clear()
        self.assertEqual(self.run_original()['cache_hits'], ['baseline'])
        self.assertEqual(self.commands, [])

    def test_image_or_intrinsics_changes_invalidate_native_and_downstream(self):
        """Input bytes and K participate in stage identities rather than using saved image counts or names."""
        self.run_original()
        (self.images/self.config['image_names'][0]).write_bytes(b'changed image')
        self.commands.clear()
        result = self.run_original()
        self.assertIn('native', self.commands)
        self.assertFalse(result['cache_used'])
        self.commands.clear()
        result = self.run_original(intrinsics=[2800., 2700., 1920., 1080.])
        self.assertIn('native', self.commands)
        self.assertEqual(result['cache_hits'], ['features'])

    def test_real_ignores_modify_cache_and_regenerates_every_invocation(self):
        """Poisoning the cache constructor proves real reconstruction never accesses development cache logic."""
        self.run_original()
        self.commands.clear()
        with patch.object(modify_cache, 'ModifyCache', side_effect=AssertionError('real must not look up caches')):
            first = self.run_original(mode='real')
            second = self.run_original(mode='real')
        self.assertEqual(self.commands.count('native'), 2)
        self.assertEqual(len(self.commands), 20)
        self.assertNotEqual(first['output_dir'], second['output_dir'])
        self.assertFalse(second['cache_used'])
        self.assertFalse(second['cache_allowed'])
        with self.assertRaisesRegex(ValueError, 'Batch mode differs'):
            workflow.read_batch(self.batch, ['original'], 'modify')

    def test_improved_modify_always_runs_four_jobs_and_rejects_stale_preparation(self):
        """Only common preparation is reused; each improved model and its dense/DSM outputs are new."""
        self.run_original()
        self.commands.clear()
        first = run_improved(self.batch, 'modify')
        second = run_improved(self.batch, 'modify')
        self.assertEqual(self.commands, ['joint', 'dense', 'update', 'export']*2)
        self.assertNotEqual(first['output_dir'], second['output_dir'])
        self.commands.clear()
        self.tokens['prior'] = 'prior_v2'
        with self.assertRaisesRegex(ValueError, 'rerun run_original.py --mode modify'):
            run_improved(self.batch, 'modify')
        self.assertEqual(self.commands, [])

    def test_real_rejects_modify_batch_before_reading_its_results(self):
        """A mode mismatch blocks both downstream entries before any reconstruction or evaluation."""
        self.run_original()
        self.commands.clear()
        with self.assertRaisesRegex(ValueError, 'Batch mode differs'):
            run_improved(self.batch)
        with self.assertRaisesRegex(ValueError, 'Batch mode differs'):
            resolve_evaluation_sources(self.batch)
        self.assertEqual(self.commands, [])

    def test_failed_native_is_not_published_as_reusable_cache(self):
        """A failed new batch blocks downstream stages and leaves no complete native cache entry."""
        with patch.object(workflow.subprocess, 'run', return_value=SimpleNamespace(returncode=7)):
            with self.assertRaisesRegex(RuntimeError, 'native failed'):
                self.run_original()
        cache_root = self.batch/'modify_cache/reconstruction'
        self.assertEqual(list(cache_root.rglob('*.json')), [])
        with self.assertRaisesRegex(ValueError, 'original successfully'):
            workflow.read_batch(self.batch, ['original'], 'modify')

    def test_modified_input_failure_does_not_leave_old_success_active(self):
        """Reject an invalid latest raw-input invocation instead of handing the old successful batch forward."""
        self.run_original()
        with patch.object(workflow, 'prepare_config', side_effect=ValueError('bad input')):
            with self.assertRaisesRegex(ValueError, 'bad input'):
                self.run_original()
        with self.assertRaisesRegex(ValueError, 'original successfully'):
            workflow.read_batch(self.batch, ['original'], 'modify')

    def test_cache_integrity_scope_and_real_path_guards(self):
        """Missing/modified artifacts miss; reconstruction cannot consume GT caches or point at real outputs."""
        result = self.run_original()
        cache = modify_cache.ModifyCache(self.batch, 'reconstruction')
        key = result['dependency_fingerprints']['baseline']
        self.assertIsNotNone(cache.get('baseline', key))
        (Path(result['output_dir'])/'baseline/export/poses_wgs84.txt').write_bytes(b'tampered')
        self.assertIsNone(cache.get('baseline', key))
        (Path(result['output_dir'])/'native/keypoints.h5').unlink()
        self.assertIsNone(cache.get('native', result['dependency_fingerprints']['native']))
        with self.assertRaisesRegex(ValueError, 'GT reference'):
            cache.get('reference', key)
        with self.assertRaisesRegex(ValueError, 'outside'):
            cache.put('native', key, self.batch/'real/original', ['.'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            cache.put('native', key, Path(result['output_dir']), ['absent.bin'])
        alias = self.root/'alias'
        alias.symlink_to(Path(result['output_dir']), target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'real mode cannot read'):
            workflow.guard_real_paths(self.batch, [alias])

    def test_solver_function_changes_do_not_invalidate_selected_handoff_code(self):
        """Only formal solver bodies are omitted; shared imports, globals and handoff functions remain bound."""
        source = self.root/'stage.py'
        source.write_text('import json\nLIMIT = 1\ndef handoff():\n    return LIMIT\ndef solve():\n    return 1\n')
        first = modify_cache.source_digest(source, ['handoff'])
        source.write_text(source.read_text().replace('return 1', 'return 2'))
        self.assertEqual(first, modify_cache.source_digest(source, ['handoff']))
        source.write_text(source.read_text().replace('LIMIT = 1', 'LIMIT = 2'))
        self.assertNotEqual(first, modify_cache.source_digest(source, ['handoff']))

class TestModifyEvaluation(unittest.TestCase):
    """Exercise the actual evaluator on all 148 fixture pixels, without any reconstruction jobs."""
    def setUp(self):
        """Use an independent legacy fixture so evaluation compatibility and mode provenance stay testable."""
        self.fixture = test_workflow_parts.TestWorkflowParts()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.legacy = self.fixture.create_legacy_run()
        context = patch.object(workflow, 'check_environment')
        context.start()
        self.addCleanup(context.stop)

    def score(self, mode='modify'):
        """Run actual metrics using the explicit requested mode and the tiny GT dataset."""
        return evaluate_results(self.legacy, self.fixture.gt_poses, self.fixture.gt_depth, mode)

    def test_modify_reuses_reference_and_recomputes_joint_errors(self):
        """A changed improved DSM keeps baseline references but always generates new improved error arrays."""
        import rasterio
        first = self.score()
        source = self.fixture.original/'joint/map/updated_dsm.tif'
        with rasterio.open(source, 'r+') as raster:
            raster.write(raster.read(1)+1, 1)
        summary = self.fixture.original/'joint/export/summary.json'
        data = json.loads(summary.read_text())
        for name, path in data['artifacts'].items():
            data['output_sha256'][name] = workflow.get_sha256(path)
        data['input_sha256'] = {path: workflow.get_sha256(path) for path in data['input_sha256']}
        workflow.write_validation_json(summary, data)
        with patch.object(evaluation, 'stream_baseline_references', side_effect=AssertionError('Baseline reference must be reused')):
            second = self.score()
        self.assertFalse(first['cache_used'])
        self.assertEqual(second['cache_hits'], ['reference'])
        old = json.loads(Path(first['report_json']).read_text())
        new = json.loads(Path(second['report_json']).read_text())
        self.assertEqual(old['reference_count'], 148)
        self.assertEqual(old['baseline_metrics'], new['baseline_metrics'])
        self.assertNotEqual(old['joint_metrics']['dsm_median_m'], new['joint_metrics']['dsm_median_m'])
        self.assertTrue((Path(second['output_dir'])/'evaluation/reference_points.bin').is_symlink())
        self.assertFalse((Path(second['output_dir'])/'evaluation/dsm_errors_joint.npy').is_symlink())

    def test_gt_change_invalidates_reference_key(self):
        """A changed depth or baseline invalidates only evaluator reference caching, without GT in reconstruction."""
        self.score()
        depth_path = next(self.fixture.gt_depth.glob('*.npy'))
        depth = np.load(depth_path)
        depth[1, 1] += 0.5
        np.save(depth_path, depth)
        with patch.object(evaluation, 'stream_baseline_references', wraps=evaluation.stream_baseline_references) as stream:
            second = self.score()
        stream.assert_called_once()
        self.assertFalse(second['cache_used'])
        self.assertFalse((self.legacy/'modify_cache/reconstruction').exists())

    def test_baseline_and_camera_changes_invalidate_reference_identity(self):
        """Bind reference caching to baseline outputs, fixed K and source DSM as well as GT bytes."""
        config = json.loads((self.legacy/'reconstruction.json').read_text())
        protocol = json.loads((workflow.PROJECT_ROOT/'configs/joint_map_evaluation.json').read_text())
        protocol.update(gt_pose_path=str(self.fixture.gt_poses), gt_depth_dir=str(self.fixture.gt_depth))
        first = modify_cache.reference_key(config, protocol, self.legacy, workflow.PROJECT_ROOT)
        changed = copy.deepcopy(config)
        changed['camera']['params'][0] += 1
        self.assertNotEqual(first, modify_cache.reference_key(changed, protocol, self.legacy, workflow.PROJECT_ROOT))
        path = self.legacy/'baseline/export/poses_wgs84.txt'
        path.write_text(path.read_text()+'\n')
        self.assertNotEqual(first, modify_cache.reference_key(config, protocol, self.legacy, workflow.PROJECT_ROOT))

    def test_real_evaluation_never_loads_modify_reference(self):
        """Real evaluation recreates references even when a complete modify reference exists for these inputs."""
        self.score()
        with patch.object(modify_cache, 'ModifyCache', side_effect=AssertionError('real must never access modify cache')), patch.object(evaluation, 'stream_baseline_references', wraps=evaluation.stream_baseline_references) as stream:
            first = self.score('real')
            second = self.score('real')
        self.assertEqual(stream.call_count, 2)
        self.assertNotEqual(first['output_dir'], second['output_dir'])
        self.assertFalse(second['cache_used'])
        self.assertEqual(json.loads(Path(second['report_json']).read_text())['mode'], 'real')

    def test_real_rejects_legacy_modify_marker(self):
        """Old result adaptation cannot bypass execution-mode isolation."""
        path = self.legacy/'localization_result.json'
        data = json.loads(path.read_text())
        workflow.write_validation_json(path, dict(data, mode='modify'))
        with self.assertRaisesRegex(ValueError, 'real mode rejects modify'):
            resolve_evaluation_sources(self.legacy)

if __name__ == '__main__':
    unittest.main()
