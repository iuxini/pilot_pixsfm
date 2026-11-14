# 用途：每次从本批照片重新生成原版特征、PixSfM及CUDA稠密工作区，不复用数据产物。
import argparse
import inspect
import itertools
import json
import multiprocessing
import os
import subprocess
from pathlib import Path
import numpy as np
import pycolmap
import torch
from omegaconf import OmegaConf
from hloc import extract_features, match_features
from hloc.extractors.superpoint import SuperPoint
from pixsfm.features.models import s2dnet
from pixsfm.refine_hloc import PixSfM
from joint_map.dense import check_dense_workspace, read_fused_cloud
from joint_map.optimization import get_sha256, write_validation_json

def configure_native_process(output):
    """Keep multiprocessing sockets relative to this isolated process's output directory."""
    output = Path(output).resolve()
    (output/'cache/tmp').mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    multiprocessing.current_process()._config['tempdir'] = 'cache/tmp'

def prepare_native_inputs(config, root):
    """Build native inputs in an empty workspace using existing local model weights."""
    directory = Path(config['native_sfm_dir']).parent
    if directory.exists():
        raise ValueError('Native workspace must be new; this entry never reuses data artifacts')
    images = root/config['image_dir']
    names = config['image_names']
    weights = Path(inspect.getfile(SuperPoint)).resolve().parents[2]/'third_party/SuperGluePretrainedNetwork/models/weights'
    weight_paths = [weights/'superpoint_v1.pth', weights/'superglue_outdoor.pth', Path(s2dnet.__file__).parent/'checkpoints/s2dnet_weights.pth']
    for path in weight_paths:
        if not path.is_file():
            raise FileNotFoundError('Existing local model weight required; no download: '+str(path))
    if not torch.cuda.is_available():
        raise RuntimeError('Native preparation requires access to the existing CUDA GPU')
    directory.mkdir(parents=True)
    feature_conf = {'output': 'superpoint', 'model': {'name': 'superpoint', 'nms_radius': 3, 'max_keypoints': 4096}, 'preprocessing': {'grayscale': True, 'resize_max': 1024}}
    match_conf = {'output': 'superglue', 'model': {'name': 'superglue', 'weights': 'outdoor', 'sinkhorn_iterations': 50}}
    mapper = {'ba_refine_focal_length': False, 'ba_refine_principal_point': False, 'ba_refine_extra_params': False}
    refiner = PixSfM({'dense_features': {'max_edge': 1600, 'dtype': 'half', 'use_cache': True, 'cache_format': 'chunked'}, 'KA': {'optimizer': {'solver': {'num_threads': 1}}}, 'BA': {'optimizer': {'refine_focal_length': False, 'refine_principal_point': False, 'refine_extra_params': False}}})
    provenance = {'image_dir': str(images), 'image_names': names, 'camera_model': 'PINHOLE', 'width': config['camera']['width'], 'height': config['camera']['height'], 'camera_params': config['camera']['params'], 'features': feature_conf, 'matching': match_conf, 'mapping': mapper, 'pixsfm': OmegaConf.to_container(refiner.conf, resolve=True), 'coordinate_system': 'arbitrary_local_sfm', 'input_sha256': {str(images/name): get_sha256(images/name) for name in names}, 'weights_sha256': {str(path): get_sha256(path) for path in weight_paths}, 'gt_pose_depth_used': False}
    write_validation_json(directory/'run_config.json', provenance)
    features, matches, pairs = directory/'keypoints.h5', directory/'matches.h5', directory/'pairs.txt'
    extract_features.main(feature_conf, images, image_list=names, feature_path=features)
    pairs.write_text(''.join(first+' '+second+'\n' for first, second in itertools.combinations(names, 2)))
    match_features.main(match_conf, pairs, features=features, matches=matches)
    model, _ = refiner.reconstruction(directory/'sfm', images, pairs, features, matches, cache_path=directory/'dense_features.h5', camera_mode=pycolmap.CameraMode.SINGLE, image_list=names, image_options={'camera_model': 'PINHOLE', 'camera_params': ','.join(map(str, config['camera']['params']))}, mapper_options=mapper)
    if model.num_reg_images() != len(names) or not model.num_points3D():
        raise ValueError('Native SfM did not register all input images with nonempty geometry')
    for camera in model.cameras.values():
        np.testing.assert_array_equal(camera.params, config['camera']['params'])
    workspace = directory/'dense'
    binary = str(root/config['colmap_binary'])
    commands = {
        'undistort': [binary, 'image_undistorter', '--image_path', str(images), '--input_path', str(directory/'sfm'), '--output_path', str(workspace), '--output_type', 'COLMAP', '--num_patch_match_src_images', str(len(names)-1)],
        'patch_match': [binary, 'patch_match_stereo', '--workspace_path', str(workspace), '--workspace_format', 'COLMAP', '--PatchMatchStereo.max_image_size', str(config['dense']['max_image_size']), '--PatchMatchStereo.gpu_index', '0', '--PatchMatchStereo.geom_consistency', '1', '--PatchMatchStereo.write_consistency_graph', '1', '--PatchMatchStereo.cache_size', '2'],
        'fusion': [binary, 'stereo_fusion', '--workspace_path', str(workspace), '--workspace_format', 'COLMAP', '--input_type', 'geometric', '--output_path', str(workspace/'fused.ply'), '--StereoFusion.max_image_size', str(config['dense']['max_image_size']), '--StereoFusion.num_threads', '8']}
    for name, command in commands.items():
        with (directory/('dense_'+name+'.log')).open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    points = read_fused_cloud(workspace/'fused.ply')
    _, checks = check_dense_workspace(workspace, points, config, 1.)
    summary = {'status': 'PASS', 'sparse_model': str(directory/'sfm'), 'registered_images': model.num_reg_images(), 'sparse_points': model.num_points3D(), 'dense_points': len(points), 'max_image_size': config['dense']['max_image_size'], 'commands': commands, 'depth_maps': checks, 'coordinate_system': 'arbitrary_local_sfm', 'gt_pose_depth_used': False, 'data_artifacts_reused': False}
    write_validation_json(workspace/'summary.json', summary)
    write_validation_json(directory/'summary.json', summary)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate this run native inputs from its raw photographs')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text())
    root = Path(__file__).resolve().parents[1]
    configure_native_process(config['output_dir'])
    prepare_native_inputs(config, root)
