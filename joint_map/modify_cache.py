# 用途：仅在modify模式校验和复用分阶段产物；不缓存改进版求解结果，不供real模式调用。
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
from joint_map.optimization import get_sha256, write_validation_json

INITIAL_FILES = ['shared/'+name for name in ['dom_features.h5', 'dom_variants.json', 'image_dom_matches.h5', 'track_map_links.npz', 'registration.json', 'prior_model.json']]+['logs/initialize.log']+[branch+'/model/'+name for branch in ['baseline', 'joint'] for name in ['cameras.bin', 'images.bin', 'points3D.bin']]
PREFIX_FILES = ['native', 'shared', 'baseline/model', 'joint/model', 'reconstruction.json']+['logs/'+name+'.log' for name in ['check', 'initialize', 'validate_priors', 'register']]
REFERENCE_FILES = ['reference_points.bin', 'dsm_errors_baseline.npy', 'dsm_errors_original.npy', 'baseline_update_flags.npy']

def digest(value):
    """Fingerprint structured semantic inputs independently of generated output locations."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_identity(path):
    """Bind an explicit input path to its current bytes rather than its name or modification time."""
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': get_sha256(path)}

def input_identity(config):
    """Describe only actual raw image/map inputs and the supplied camera without reading any cache."""
    return {'images': {name: file_identity(Path(config['image_dir'])/name) for name in config['image_names']}, 'camera': config['camera'], 'dom': file_identity(config['dom_path']), 'dsm': file_identity(config['dsm_path'])}

def source_digest(path, functions=None):
    """Hash selected function bodies and globals so formal solver edits do not invalidate baseline preparation."""
    tree = ast.parse(Path(path).read_text())
    if functions is not None:
        tree.body = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or getattr(node, 'name', None) in functions]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()

def dependency_tokens(config, root):
    """Identify the existing model weights, loaded native libraries and relevant stage implementations."""
    from hloc import extract_features, match_features, reconstruction
    from hloc.extractors.superpoint import SuperPoint
    from hloc.matchers.superglue import SuperGlue
    from pixsfm.features.models import s2dnet
    from pixsfm.features.extractor import FeatureExtractor
    from pixsfm.refine_hloc import PixSfM
    import pyceres
    import pycolmap
    import torch
    weights = Path(inspect.getfile(SuperPoint)).resolve().parents[2]/'third_party/SuperGluePretrainedNetwork/models/weights'
    checkpoint = Path(s2dnet.__file__).parent/'checkpoints/s2dnet_weights.pth'
    libraries = {name: get_sha256(module.__file__) for name, module in list(sys.modules.items()) if name.startswith(('_pixsfm', 'pixsfm')) and (getattr(module, '__file__', '') or '').endswith('.so')}
    native_files = [root/'joint_map/native.py', Path(inspect.getfile(PixSfM)), root/'pixsfm/refine_colmap.py', Path(extract_features.__file__), Path(match_features.__file__), Path(reconstruction.__file__), Path(inspect.getfile(SuperPoint)), Path(inspect.getfile(SuperGlue)), weights/'superpoint_v1.pth', weights/'superglue_outdoor.pth', checkpoint, root/config['colmap_binary'], Path(pycolmap.__file__)]
    native = {str(path): get_sha256(path) for path in native_files}
    features = {str(path): get_sha256(path) for path in [Path(inspect.getfile(FeatureExtractor)), Path(s2dnet.__file__), checkpoint]}
    prior = {str(path): get_sha256(path) for path in [root/'joint_map/optimization.py', root/'joint_map/registration.py', root/'PROJECT_PLAN.md', root/'run_joint_map.py', Path(pyceres.__file__)]}
    prior['selection'] = source_digest(root/'joint_map/workflow.py', ['select_automatic_prior'])
    baseline = {name: get_sha256(root/'joint_map'/name) for name in ['dense.py', 'dsm.py', 'export.py', 'geometry.py']}
    baseline['handoff'] = source_digest(root/'joint_map/joint.py', ['get_formal_fingerprint', 'check_serialized_model', 'check_registered_visual_model', 'load_confirmed_handoff'])
    return {'native': [native, libraries, torch.__version__], 'initialize': [get_sha256(root/'joint_map/registration.py'), get_sha256(root/'joint_map/geometry.py')], 'features': [features, libraries, torch.__version__], 'prior': [prior, libraries], 'baseline': baseline}

def original_keys(config, root, dependencies=None):
    """Build a dependency graph with independent native, matching, feature, prior and baseline keys."""
    inputs = input_identity(config)
    tokens = dependency_tokens(config, root) if dependencies is None else dependencies
    native = digest([inputs['images'], inputs['camera'], config['dense'], tokens['native']])
    initialize = digest([native, inputs['dom'], inputs['dsm'], config['map_association'], tokens['initialize']])
    features = digest([inputs['images'], tokens['features'], tokens['native']])
    validation = {key: value for key, value in config['prior_validation'].items() if key != 'selected_model_path'}
    prefix = digest([initialize, features, validation, config['optimization'], tokens['prior']])
    return {'input': digest(inputs), 'native': native, 'initialize': initialize, 'features': features, 'prefix': prefix, 'baseline': digest([prefix, config['update'], tokens['baseline']])}

class ModifyCache:
    """Store validated references to immutable modify artifacts in a reconstruction or evaluation namespace."""
    def __init__(self, batch_dir, scope):
        """Create only the explicitly requested development cache namespace."""
        if scope not in ['reconstruction', 'evaluation']:
            raise ValueError('Unknown modify cache scope')
        self.batch_dir = Path(batch_dir).resolve()
        self.scope = scope
        self.root = self.batch_dir/'modify_cache'/scope
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = []

    def check_path(self, path):
        """Reject cache references to real outputs or arbitrary files outside this modify namespace."""
        path = Path(path).resolve()
        if not any(path.is_relative_to(folder) for folder in [self.batch_dir/'modify', self.batch_dir/'modify_cache'/self.scope]):
            raise ValueError('Modify cache artifact is outside its development namespace: '+str(path))
        return path

    def get(self, stage, key):
        """Reuse only complete matching entries whose recorded artifact bytes are still intact."""
        if (self.scope == 'evaluation') != (stage == 'reference'):
            raise ValueError('GT reference caches are isolated from reconstruction caches')
        path = self.root/stage/(key+'.json')
        entry, reason = None, 'missing'
        if path.is_file():
            candidate = json.loads(path.read_text())
            reason = 'identity_changed'
            if candidate.get('key') == key and candidate.get('stage') == stage and candidate.get('mode') == 'modify' and candidate.get('scope') == self.scope and candidate.get('complete'):
                self.check_path(candidate['source_dir'])
                reason = 'artifact_changed'
                if candidate['files'] and all(self.check_path(name).is_file() and get_sha256(name) == checksum for name, checksum in candidate['files'].items()):
                    entry, reason = candidate, 'verified'
        self.events.append({'stage': stage, 'key': key, 'hit': entry is not None, 'reason': reason})
        if entry is not None:
            print('modify缓存命中：'+stage, flush=True)
        return entry

    def put(self, stage, key, source, groups):
        """Publish completed artifacts by checksum without duplicating large feature or depth arrays."""
        if (self.scope == 'evaluation') != (stage == 'reference'):
            raise ValueError('GT reference caches are isolated from reconstruction caches')
        source = self.check_path(source)
        files = {}
        for name in groups:
            path = self.check_path(source/name)
            if not path.exists():
                raise ValueError('Cannot cache an incomplete stage: '+str(path))
            selected = [path] if path.is_file() else [item for item in path.rglob('*') if item.is_file()]
            for item in selected:
                files[str(self.check_path(item))] = get_sha256(item)
        if not files:
            raise ValueError('Cannot cache an empty stage')
        entry = {'mode': 'modify', 'scope': self.scope, 'stage': stage, 'key': key, 'complete': True, 'source_dir': str(source), 'files': files}
        path = self.root/stage/(key+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        write_validation_json(path, entry)
        return entry

    def snapshot_initialize(self, key, output):
        """Copy the small provisional map state before prior validation and registration mutate it."""
        destination = self.root/'initialize'/key/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        destination.mkdir(parents=True)
        for name in INITIAL_FILES:
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output/name, target)
        return self.put('initialize', key, destination, INITIAL_FILES)

def restore_initialize(entry, output):
    """Restore provisional files as writable copies and record the verified cached stage in this manifest."""
    for name in INITIAL_FILES:
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(entry['source_dir'])/name, target)
    report = json.loads((output/'logs/initialize.log').read_text())
    manifest = json.loads((output/'run_manifest.json').read_text())
    manifest['stages'].append({'stage': 'initialize', 'branch': None, 'status': 'complete', 'cache_used': True, 'artifacts': {'diagnostics': str(output/'logs/initialize.log')}, 'counts': {'registered_images': report.get('registered_images'), 'sparse_points': report.get('sparse_points'), 'map_priors': report.get('accepted_track_count')}, 'error': None})
    write_validation_json(output/'run_manifest.json', manifest)

def reference_key(config, evaluation, original, root):
    """Key baseline GT references without depending on the changing improved model or its output paths."""
    names = config['image_names']
    inputs = {'camera': config['camera'], 'images': {name: file_identity(root/config['image_dir']/name) for name in names}, 'dsm': file_identity(root/config['dsm_path']), 'gt_pose': file_identity(root/evaluation['gt_pose_path']), 'depths': {name: file_identity(root/evaluation['gt_depth_dir']/(Path(name).stem+evaluation['depth_suffix'])) for name in names}}
    baseline = {name: get_sha256(original/name) for name in ['shared/origin.json', 'baseline/export/poses_wgs84.txt', 'baseline/map/updated_dsm.tif', 'baseline/map/update_mask.tif']}
    protocol = {key: value for key, value in evaluation.items() if key not in ['gt_pose_path', 'gt_depth_dir', 'depth_rule_confirmation']}
    code = {name: get_sha256(root/'joint_map'/name) for name in ['evaluation.py', 'geometry.py', 'dsm.py']}
    return digest([inputs, baseline, protocol, code])

def restore_reference(entry, config, evaluation_path, config_path, root, output):
    """Bind verified read-only baseline arrays to the new evaluation's exact current input audit."""
    from joint_map.evaluation import collect_evaluation_inputs
    inputs, _, _ = collect_evaluation_inputs(config, json.loads(evaluation_path.read_text()), config_path, evaluation_path, root, output)
    folder = output/'evaluation'
    folder.mkdir()
    source = Path(entry['source_dir'])
    frozen = json.loads((source/'baseline_metrics.json').read_text())
    for name in REFERENCE_FILES:
        (folder/name).symlink_to(source/name)
    frozen.update(input_sha256=inputs, modify_reference_cache={'key': entry['key'], 'source': str(source)})
    write_validation_json(folder/'baseline_metrics.json', frozen)
