# 用途：验证共享可动 XYZ、先验尺度、条件留出稳定性及确认后的共同地图配准。
import copy
import hashlib
import inspect
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import h5py
import pyceres
import pycolmap
import rasterio
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pixsfm import features, bundle_adjustment as ba, residuals as native_residuals
from pixsfm.base import InterpolationConfig, interpolation_default_conf
from pixsfm.features.extractor import FeatureExtractor
from pixsfm.features.models import s2dnet
from PIL import Image
import torch
from joint_map.geometry import convert_pixels_to_geodetic, convert_geodetic_to_enu
from joint_map.registration import apply_sim3, check_track_geometry, estimate_sim3, screen_homography, select_map_candidates, write_initialized_model

class RecordSolverIterations(pyceres.IterationCallback):
    """Observe native Ceres iteration summaries without changing solver decisions or parameters."""
    def __init__(self):
        """Initialize an owned list of iteration values copied from native summaries."""
        super().__init__()
        self.iterations = []
        self.label = None

    def __call__(self, summary):
        """Copy diagnostics and always let the native solver continue unchanged."""
        row = {"iteration": int(summary.iteration), "step_is_valid": bool(summary.step_is_valid), "step_is_successful": bool(summary.step_is_successful), "step_is_nonmonotonic": bool(summary.step_is_nonmonotonic), "linear_solver_iterations": int(summary.linear_solver_iterations)}
        nonfinite = []
        for name in ["cost", "cost_change", "gradient_norm", "gradient_max_norm", "step_norm", "step_size", "trust_region_radius", "relative_decrease", "iteration_time_in_seconds", "cumulative_time_in_seconds"]:
            value = float(getattr(summary, name))
            row[name] = value if np.isfinite(value) else None
            if not np.isfinite(value):
                nonfinite.append({"field": name, "value": str(value)})
        row["nonfinite_values"] = nonfinite
        self.iterations.append(row)
        if self.label is not None and row["iteration"] % 10 == 0:
            print(f"{self.label}: iteration={row['iteration']}, cost={row['cost']}, gradient={row['gradient_norm']}, radius={row['trust_region_radius']}", flush=True)
        return pyceres.CallbackReturnType.SOLVER_CONTINUE

def get_observations(model):
    """List the complete current graph once without copying image point lists per observation."""
    points = {image_id: image.points2D for image_id, image in model.images.items()}
    return [(int(point_id), int(entry.image_id), int(entry.point2D_idx), points[entry.image_id][entry.point2D_idx].xy.copy()) for point_id, point in model.points3D.items() for entry in point.track.elements]

def project_observations(model, observations):
    """Project shared XYZ to observation pixels and reject nonpositive camera depths."""
    poses = {image_id: (image.rotmat(), image.tvec, model.cameras[image.camera_id].params) for image_id, image in model.images.items()}
    output = []
    for point_id, image_id, feature_id, measured in observations:
        rotation, translation, intrinsics = poses[image_id]
        camera_point = rotation @ model.points3D[point_id].xyz + translation
        if camera_point[2] <= 0 or not np.isfinite(camera_point).all():
            raise ValueError("Invalid camera depth in validation model")
        output.append(camera_point[:2] / camera_point[2] * intrinsics[:2] + intrinsics[2:])
    return np.asarray(output)

def create_holdout(model, fraction, seed):
    """Remove one observation from a seeded fraction of tracks having at least three views."""
    training = copy.deepcopy(model)
    observations = get_observations(model)
    eligible = sorted(point_id for point_id, point in model.points3D.items() if len(point.track.elements) >= 3)
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, size=int(np.floor(len(eligible) * fraction)), replace=False)
    heldout = []
    for point_id in sorted(selected):
        entries = sorted(model.points3D[int(point_id)].track.elements, key=lambda entry: (entry.image_id, entry.point2D_idx))
        entry = entries[int(rng.integers(len(entries)))]
        heldout.append(next(observation for observation in observations if observation[:3] == (point_id, entry.image_id, entry.point2D_idx)))
        training.delete_observation(entry.image_id, entry.point2D_idx)
    if not heldout or set(training.points3D) != set(model.points3D):
        raise ValueError("Holdout requires nonempty evaluation and at least two training views per affected track")
    return training, heldout

def create_training_links(config, root, output_dir, model, heldout):
    """Exclude heldout map votes and recompute planar and 3D selection on training observations."""
    with np.load(output_dir / "shared/track_map_links.npz") as archive:
        links = {key: archive[key] for key in archive.files}
    pairs = {(image_id, feature_id) for point_id, image_id, feature_id, xy in heldout}
    keep = np.array([(int(image_id), int(feature_id)) not in pairs for image_id, feature_id in zip(links["image_ids"], links["feature_ids"])])
    original_indices = np.flatnonzero(keep)
    links = {key: values[keep] for key, values in links.items()}
    observations = {image_id: image.points2D for image_id, image in model.images.items()}
    reliable = {int(point_id): check_track_geometry(model, observations, point_id, config["update"]["max_reprojection_error_px"], config["update"]["min_triangulation_angle_deg"]) for point_id in model.points3D}
    links["rejection_codes"][:] = "pending_3d"
    for index, track_id in enumerate(links["track_ids"]):
        if track_id < 0:
            links["rejection_codes"][index] = "pending_untriangulated_feature"
        elif not np.isfinite(links["means_enu_m"][index]).all():
            links["rejection_codes"][index] = "invalid_dsm_support"
        elif not reliable[int(track_id)]:
            links["rejection_codes"][index] = "unreliable_image_track"
    with h5py.File(root / config["frontend"]["features_path"], "r") as images:
        for image_id, image in model.images.items():
            pixels = images[image.name]["keypoints"][:]
            for variant_id in np.unique(links["dom_variant_ids"]):
                indices = np.flatnonzero((links["image_ids"] == image_id) & (links["dom_variant_ids"] == variant_id))
                errors, matrix = screen_homography(pixels[links["feature_ids"][indices]], links["dom_pixels"][indices], config["map_association"]["homography_forward_threshold_dom_px"])
                links["homography_pass"][indices] = errors <= config["map_association"]["homography_forward_threshold_dom_px"]
    source = np.array([model.points3D[int(track_id)].xyz if track_id >= 0 else np.full(3, np.nan) for track_id in links["track_ids"]])
    accepted, codes, errors, reference_counts, selected, evidence = select_map_candidates(links, source, config["map_association"]["sim3_inlier_threshold_m"])
    if len(selected) < config["map_association"]["min_tracks"]:
        raise ValueError("Insufficient independently verified training-only map associations")
    return {"point_ids": links["track_ids"][selected], "means": links["means_enu_m"][selected], "dom_pixels": links["dom_pixels"][selected], "map_feature_ids": links["dom_feature_ids"][selected], "selected_link_indices": original_indices[selected], "source_prior_ids": links["prior_ids"][selected], "excluded_map_votes": int((~keep).sum()), "independent_models": evidence}

def compute_robust_scale(values, floor):
    """Estimate a MAD working scale with an explicit numeric or quantization floor."""
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Scale estimation requires finite training residuals")
    return float(max(1.482602218505602 * np.median(np.abs(values - np.median(values))), floor))

def create_feature_context(model, cache_path):
    """Keep native feature storage, views and training-only references alive together."""
    manager = features.FeatureManager_f16(str(cache_path), False, "")
    feature_set = manager.fset(0)
    view = features.FeatureView(feature_set, model)
    interpolation = InterpolationConfig(interpolation_default_conf)
    extractor = ba.ReferenceExtractor(ba.ReferenceConfig({"num_threads": 1, "keep_observations": True, "compute_offsets3D": False}), interpolation)
    labels = [-1] * (max(model.points3D) + 1)
    for point_id in model.points3D:
        labels[point_id] = point_id // 10
    references = extractor.run(labels, model, feature_set)
    return {"manager": manager, "feature_set": feature_set, "view": view, "interpolation": interpolation, "extractor": extractor, "references": references, "interpolator": features.PatchInterpolator(interpolation)}

def get_sha256(path):
    """Hash a file in bounded chunks without changing its contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def get_observation_digest(observations):
    """Identify an ordered observation set including its measured pixel coordinates."""
    digest = hashlib.sha256()
    for point_id, image_id, feature_id, measured in observations:
        digest.update(np.asarray([point_id, image_id, feature_id], dtype="<i8").tobytes())
        digest.update(np.asarray(measured, dtype="<f8").tobytes())
    return digest.hexdigest()

def get_dense_provenance(config, root):
    """Fingerprint the original images, checkpoint, extractor source and resolved extraction settings."""
    directory = root / config["native_sfm_dir"] / ".."
    original = json.loads((directory / "run_config.json").read_text())["pixsfm"]
    extraction = original["dense_features"]
    if extraction["max_edge"] != 1600 or extraction["dtype"] != "half" or extraction["pyr_scales"] != [1.0] or extraction["model"]["name"] != "s2dnet" or not extraction["l2_normalize"]:
        raise ValueError("Original extraction differs from the approved 1600-pixel float16 S2DNet setup")
    checkpoint = Path(s2dnet.__file__).parent / "checkpoints/s2dnet_weights.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("Original S2DNet checkpoint unavailable; no download allowed")
    images = {}
    for name in config["image_names"]:
        path = root / config["image_dir"] / name
        with Image.open(path) as image:
            size = list(image.size)
        images[name] = {"sha256": get_sha256(path), "original_size": size, "feature_shape": [900, 1600, 128]}
        if size != [3840, 2160]:
            raise ValueError("Unexpected original image dimensions")
    return {"images": images, "checkpoint": {"path": str(checkpoint), "sha256": get_sha256(checkpoint)}, "extraction": extraction, "model_defaults": s2dnet.S2DNet.default_conf, "interpolation": original["interpolation"], "normalization": {"mean": s2dnet.S2DNet.mean, "std": s2dnet.S2DNet.std, "l2_before_float16": True}, "storage": "whole_image_float16_HWC_evaluation_only", "extractor_source_sha256": get_sha256(inspect.getfile(FeatureExtractor)), "model_source_sha256": get_sha256(s2dnet.__file__), "torch_version": torch.__version__}

def check_dense_cache(path, provenance):
    """Validate every full-image tensor, coordinate scale, finite value and content hash before reuse."""
    summaries = {}
    with h5py.File(path, "r") as cache:
        if not cache.attrs.get("complete", False) or json.loads(cache.attrs["provenance"]) != provenance or set(cache) != set(provenance["images"]):
            raise ValueError("Existing holdout cache is incomplete or has different inputs; refusing overwrite or reuse")
        for name, expected in provenance["images"].items():
            group = cache[name]
            dataset = group["features"]
            if list(dataset.shape) != expected["feature_shape"] or dataset.dtype != np.float16 or not group.attrs.get("complete", False):
                raise ValueError(f"Invalid full feature tensor: {name}")
            scale = np.array([dataset.shape[1] / expected["original_size"][0], dataset.shape[0] / expected["original_size"][1]])
            if not np.array_equal(group.attrs["scale"], scale) or not np.array_equal(group.attrs["original_size"], expected["original_size"]):
                raise ValueError(f"Feature coordinate mapping mismatch: {name}")
            digest = hashlib.sha256()
            for start in range(0, dataset.shape[0], 32):
                block = dataset[start:start + 32]
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite full feature values: {name}")
                digest.update(block.tobytes())
            if digest.hexdigest() != group.attrs["content_sha256"]:
                raise ValueError(f"Full feature content hash mismatch: {name}")
            summaries[name] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype), "scale": scale.tolist(), "original_size": expected["original_size"], "content_sha256": digest.hexdigest(), "raw_bytes": int(dataset.size * dataset.dtype.itemsize)}
    return summaries

def prepare_dense_holdout_cache(config, root, path):
    """Extract six whole S2DNet maps sequentially, or reuse only a complete matching cache."""
    provenance = get_dense_provenance(config, root)
    required = sum(int(np.prod(item["feature_shape"])) * 2 for item in provenance["images"].values())
    available_memory = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    resources = {"disk_free_bytes": shutil.disk_usage(path.parent).free, "memory_available_bytes": available_memory, "expected_feature_bytes": required}
    reused = path.exists()
    if not reused:
        if resources["disk_free_bytes"] < required + 1024 ** 3 or available_memory < 8 * 1024 ** 3:
            raise RuntimeError("Insufficient resources for approved full-resolution float16 cache; no downsampling or precision change")
        if not torch.cuda.is_available():
            raise RuntimeError("Host GPU access required for original whole-image extraction; no environment change allowed")
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        resources.update(gpu_free_bytes=free_gpu, gpu_total_bytes=total_gpu, gpu_name=torch.cuda.get_device_name(0))
        if free_gpu < 6 * 1024 ** 3:
            raise RuntimeError("Insufficient GPU memory for full-image S2DNet extraction")
        extraction = copy.deepcopy(provenance["extraction"])
        extraction.update(sparse=False, use_cache=False)
        extractor = FeatureExtractor(extraction)
        with h5py.File(path, "x") as cache:
            cache.attrs["complete"] = False
            cache.attrs["provenance"] = json.dumps(provenance, sort_keys=True)
            for name in config["image_names"]:
                maps = extractor(root / config["image_dir"] / name, as_dict=True, overwrite_sparse=False)
                if len(maps) != 1:
                    raise ValueError("Expected exactly one full S2DNet feature level")
                data = maps[0]["patches"][0]
                if list(data.shape) != provenance["images"][name]["feature_shape"] or data.dtype != np.float16:
                    raise ValueError("Extraction changed full-map shape or precision")
                group = cache.create_group(name)
                dataset = group.create_dataset("features", data=data, chunks=(32, 128, 128), fletcher32=True)
                group.attrs["scale"] = maps[0]["metadata"]["scale"]
                group.attrs["original_size"] = provenance["images"][name]["original_size"]
                group.attrs["content_sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
                group.attrs["complete"] = True
                cache.flush()
                print(f"Whole-image holdout cache: {name} {data.shape} float16", flush=True)
                del maps, data
            cache.attrs["complete"] = True
    summaries = check_dense_cache(path, provenance)
    return {"path": str(path), "reused": reused, "provenance": provenance, "resources": resources, "complete": True, "images": summaries, "file_bytes": path.stat().st_size, "coordinate_mapping": "feature_uv = original_xy * scale - 0.5; whole-map corner=(0,0)"}

def compare_dense_and_patch_cache(dense_path, patch_path, interpolation):
    """Check all cached patch values and interior subpixel interpolation against whole-image features."""
    interpolator = features.PatchInterpolator(interpolation)
    raw_max, interpolation_max, scalar_count, sample_count = 0., 0., 0, 0
    with h5py.File(dense_path, "r") as dense, h5py.File(patch_path, "r") as sparse:
        for name in dense:
            data = dense[name]["features"][:]
            scale = dense[name].attrs["scale"]
            full_patch = features.FeaturePatch(data, np.array([0, 0], dtype=np.int32), np.asarray(scale, dtype=np.float64), False)
            group = sparse["0"][name]
            if not np.allclose(group["scales"][:], scale, rtol=0, atol=1e-15):
                raise ValueError("Sparse and dense feature scales differ")
            for index, corner in enumerate(group["corners"][:]):
                old = group["patches"][index]
                x, y = corner
                new = data[y:y + old.shape[0], x:x + old.shape[1]]
                difference = np.abs(new.astype(np.float32) - old.astype(np.float32))
                if not np.array_equal(new, old):
                    raise ValueError(f"Whole-image features differ from original patch extraction: {name}, patch {index}, max={difference.max()}")
                raw_max = max(raw_max, float(difference.max()))
                scalar_count += old.size
                old_patch = features.FeaturePatch(np.ascontiguousarray(old), np.asarray(corner, dtype=np.int32), np.asarray(scale, dtype=np.float64), False)
                pixel = (np.asarray(corner) + [5.75, 8.0]) / scale
                before = interpolator.interpolate(old_patch, pixel)
                after = interpolator.interpolate(full_patch, pixel)
                error = float(np.max(np.abs(before - after)))
                if not np.all(np.abs(before-after) <= 1e-10+1e-8*np.maximum(np.abs(before), np.abs(after))):
                    raise ValueError(f"Native interpolation mismatch between caches: {name}, patch {index}")
                interpolation_max = max(interpolation_max, error)
                sample_count += 1
            print(f"Cache consistency: {name} passed", flush=True)
    return {"passed": True, "compared_feature_scalars": scalar_count, "interpolation_samples": sample_count, "max_raw_absolute_difference": raw_max, "max_interpolation_absolute_difference": interpolation_max, "raw_tolerance": "exact same float16 values", "interpolation_tolerance": "atol=1e-10, rtol=1e-8", "float16_quantization_accounted": True}

def audit_training_references(model, heldout, context):
    """Re-evaluate each saved native reference input and verify actual contributors exclude holdout."""
    observations = get_observations(model)
    projected = project_observations(model, observations)
    expected = {(point_id, image_id, feature_id): pixel for (point_id, image_id, feature_id, measured), pixel in zip(observations, projected)}
    excluded = {(point_id, image_id, feature_id) for point_id, image_id, feature_id, measured in heldout}
    participants, maximum_error = [], 0.
    digest = hashlib.sha256()
    for point_id, reference in context["references"].items():
        entries = reference.track.elements
        saved = reference.observations
        if len(entries) != len(saved):
            raise ValueError("Reference contributor list does not describe its saved descriptor inputs")
        source_index = None
        for index, entry in enumerate(entries):
            identity = (int(point_id), int(entry.image_id), int(entry.point2D_idx))
            if identity not in expected or identity in excluded:
                raise ValueError("Reference input is absent from training or includes holdout")
            patch = context["view"].fpatch(entry.image_id, entry.point2D_idx)
            sampled = context["interpolator"].interpolate(patch, expected[identity]).reshape(-1)
            actual = np.asarray(saved[index]).reshape(-1)
            error = float(np.max(np.abs(sampled - actual)))
            if error > 1e-12:
                raise ValueError("Saved reference inputs do not match interpolation of the declared training initial model")
            maximum_error = max(maximum_error, error)
            participants.append(identity)
            digest.update(actual.astype("<f8").tobytes())
            if entry.image_id == reference.source.image_id and entry.point2D_idx == reference.source.point2D_idx:
                source_index = index
        if source_index is None or not np.allclose(np.asarray(reference.descriptor).reshape(-1), np.asarray(saved[source_index]).reshape(-1), rtol=0, atol=1e-12):
            raise ValueError("Reference descriptor is not its native selected training source")
    if set(participants) != set(expected) or len(participants) != len(expected):
        raise ValueError("Reference computation did not use exactly the current training observations")
    return {"recomputed": True, "source": "native ReferenceExtractor with keep_observations=True on the current training initial model", "participants": [list(identity) for identity in participants], "training_observations_sha256": get_observation_digest(observations), "heldout_observations_sha256": get_observation_digest(heldout), "actual_input_descriptors_sha256": digest.hexdigest(), "max_resampling_difference": maximum_error, "actual_participant_overlap_with_holdout": 0, "reference_count": len(context["references"]), "reference_config": {"keep_observations": True, "compute_offsets3D": False, "closest_to_robust_mean": True, "num_threads": 1}}

def sample_dense_holdout(model, observations, cache_path, context):
    """Evaluate every predicted holdout descriptor with explicit image and bicubic boundary failures."""
    residuals = np.full((len(observations), 128), np.nan)
    pixel_errors = np.full(len(observations), np.nan)
    invalid, old_missing = [], []
    with h5py.File(cache_path, "r") as cache:
        for image_id, image in model.images.items():
            group = cache[image.name]
            data = group["features"][:]
            patch = features.FeaturePatch(data, np.array([0, 0], dtype=np.int32), np.asarray(group.attrs["scale"], dtype=np.float64), False)
            camera = model.cameras[image.camera_id]
            for index, (point_id, current_id, feature_id, measured) in enumerate(observations):
                if current_id != image_id:
                    continue
                xyz = image.rotmat() @ model.points3D[point_id].xyz + image.tvec
                reason, pixel = None, None
                if not np.isfinite(xyz).all() or xyz[2] <= 0:
                    reason = "nonfinite_or_nonpositive_camera_depth"
                else:
                    pixel = xyz[:2] / xyz[2] * camera.params[:2] + camera.params[2:]
                    if not np.isfinite(pixel).all():
                        reason = "nonfinite_projection"
                    elif not (0 <= pixel[0] < camera.width and 0 <= pixel[1] < camera.height):
                        reason = "outside_full_image"
                    else:
                        local = patch.get_pixel_coords(pixel)
                        if not (np.all(local >= 1) and local[0] < patch.width - 2 and local[1] < patch.height - 2):
                            reason = "outside_full_bicubic_support"
                if reason is not None:
                    invalid.append({"point3D_id": point_id, "image_id": image_id, "image_name": image.name, "feature_id": feature_id, "predicted_xy": None if pixel is None else [float(value) if np.isfinite(value) else None for value in pixel], "camera_depth": float(xyz[2]) if np.isfinite(xyz[2]) else None, "reason": reason})
                    continue
                pixel_errors[index] = np.linalg.norm(pixel - measured)
                descriptor = context["interpolator"].interpolate(patch, pixel).reshape(-1)
                residuals[index] = descriptor - np.asarray(context["references"][point_id].descriptor).reshape(-1)
                old = context["view"].fpatch(image_id, feature_id)
                local = old.get_pixel_coords(pixel)
                if not (np.all(local >= 1) and local[0] < old.width - 2 and local[1] < old.height - 2):
                    old_missing.append({"point3D_id": point_id, "image_id": image_id, "feature_id": feature_id, "predicted_xy": pixel.tolist(), "reason": "inside_image_outside_old_patch"})
    return residuals, pixel_errors, {"sample_count": len(observations), "valid_count": len(observations) - len(invalid), "invalid_count": len(invalid), "invalid_predictions": invalid, "old_patch_misses_filled": old_missing, "sampling_location": "optimized_camera_and_shared_XYZ_prediction", "evaluation_only": True}

def sample_feature_residuals(model, observations, context):
    """Sample native feature residuals only inside valid bicubic patches."""
    projected = project_observations(model, observations)
    residuals, valid = [], []
    for pixel, (point_id, image_id, feature_id, measured) in zip(projected, observations):
        patch = context["view"].fpatch(image_id, feature_id)
        local = patch.get_pixel_coords(pixel)
        inside = bool(np.all(local >= 1) and local[0] < patch.width - 2 and local[1] < patch.height - 2)
        valid.append(inside)
        if inside:
            descriptor = context["interpolator"].interpolate(patch, pixel)
            reference = np.asarray(context["references"][point_id].descriptor).reshape(-1)
            residuals.append(descriptor.reshape(-1) - reference)
        else:
            residuals.append(np.full(context["feature_set"].channels, np.nan))
    return np.asarray(residuals), np.asarray(valid)

def estimate_working_scales(config, root, model, anchors, context, origin):
    """Estimate conditional map, pixel and feature scales exclusively from training evidence."""
    ids = anchors["point_ids"]
    source = np.array([model.points3D[int(point_id)].xyz for point_id in ids])
    transform, inliers = estimate_sim3(source, anchors["means"], config["map_association"]["sim3_inlier_threshold_m"])
    if transform is None or inliers.sum() < config["map_association"]["min_tracks"]:
        raise ValueError("Training-only map fit is not observable")
    residuals = apply_sim3(source[inliers], transform) - anchors["means"][inliers]
    with rasterio.open(root / config["dsm_path"]) as dsm:
        pixels = np.array([[dsm.width / 2, dsm.height / 2], [dsm.width / 2 + 1, dsm.height / 2], [dsm.width / 2, dsm.height / 2 + 1]])
        geographic = convert_pixels_to_geodetic(pixels, dsm.transform)
        enu = convert_geodetic_to_enu(np.column_stack([geographic, np.full(3, origin["ellipsoidal_height_m"])]), origin)
        cell_lengths = np.linalg.norm(enu[1:, :2] - enu[0, :2], axis=1)
        horizontal_floor = float(np.min(cell_lengths) / np.sqrt(12))
    height_floor = float(np.spacing(np.float32(origin["ellipsoidal_height_m"])))
    observations = get_observations(model)
    pixel_residuals = project_observations(model, observations) - np.array([row[3] for row in observations])
    pixel_floor = float(np.finfo(np.float32).eps * max(config["camera"]["width"], config["camera"]["height"]))
    feature_residuals, valid = sample_feature_residuals(model, observations, context)
    if not valid.any():
        raise ValueError("No valid training feature patches for scale estimation")
    valid_features = feature_residuals[valid]
    feature_scale = compute_robust_scale(valid_features.ravel(), np.finfo(np.float16).eps * np.median(np.abs(valid_features)))
    horizontal = compute_robust_scale(residuals[:, :2].ravel(), horizontal_floor)
    height = compute_robust_scale(residuals[:, 2], height_floor)
    pixel = compute_robust_scale(pixel_residuals.ravel(), pixel_floor)
    feature_norm_scale = feature_scale * np.sqrt(feature_residuals.shape[1])
    return {"sigma_horizontal_m": horizontal, "sigma_height_m": height, "sigma_pixel_px": pixel, "sigma_feature_component": feature_scale, "sigma_feature_norm": float(feature_norm_scale), "lambda_feature": float(2 / feature_norm_scale ** 2), "feature_scale_valid_observations": int(valid.sum()), "feature_scale_invalid_observations": int((~valid).sum()), "floors": {"horizontal_m": horizontal_floor, "height_m": height_floor, "pixel_px": pixel_floor}, "source": "MAD of training-only robust map-fit, image and valid feature residuals; optimization still requires every feature projection valid after geometric preoptimization", "limitations": "Conditional working scales after association selection and 2m truncation; not independently calibrated map noise or absolute datum accuracy", "feature_scaling": "2/(channels*sigma_component^2) normalizes expected per-observation feature energy to two pixel components", "robust_loss": {"name": "huber", "normalized_threshold": 3.0}}

def check_gauge_information(positions, covariance):
    """Check seven similarity freedoms using centered, spatially normalized anchor information."""
    centered = np.asarray(positions) - np.mean(positions, axis=0)
    span = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Degenerate map anchor span")
    matrix = np.zeros((7, 7))
    whitening = np.diag(1 / np.sqrt(np.diag(covariance)))
    for x, y, z in centered / span:
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        jacobian = whitening @ np.column_stack([np.eye(3), -skew, [x, y, z]])
        matrix += jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= np.finfo(float).eps * eigenvalues[-1] * 7:
        raise ValueError("Map anchors cannot constrain all seven similarity freedoms")
    return {"eigenvalues": eigenvalues.tolist(), "condition": float(eigenvalues[-1] / eigenvalues[0]), "coordinate_normalization_span_m": float(span)}

def build_native_problem(model, anchors, scales, information_scale, context=None, fixed_points=False, gauge_control=False):
    """Build native image factors and finite map factors on the same XYZ memory blocks."""
    if not all(np.isfinite(scales[key]) and scales[key] > 0 for key in ["sigma_horizontal_m", "sigma_height_m", "sigma_pixel_px"]) or not np.isfinite(information_scale) or information_scale < 0:
        raise ValueError("Working scales must be finite positive and information scale nonnegative")
    setup = ba.BundleAdjustmentSetup()
    setup.add_images(set(model.images))
    if gauge_control:
        image_ids = sorted(model.images)
        setup.set_constant_pose(image_ids[0])
        setup.set_constant_tvec(image_ids[1], [0])
    options = ba.BundleOptimizerOptions({"refine_focal_length": False, "refine_principal_point": False, "refine_extra_params": False, "refine_extrinsics": True, "min_track_length": 2, "print_summary": False, "solver": {"use_inner_iterations": False, "max_num_iterations": 0, "function_tolerance": 1e-6, "gradient_tolerance": 1e-10, "parameter_tolerance": 1e-8, "num_threads": 1, "minimizer_progress_to_stdout": False}})
    iteration_recorder = RecordSolverIterations()
    options.solver.callbacks = [iteration_recorder]
    pixel_sigma = scales["sigma_pixel_px"]
    geometric_loss = pyceres.LossFunction({"name": "huber", "params": [3 * pixel_sigma], "magnitude": 1 / pixel_sigma ** 2})
    losses, costs = [geometric_loss], []
    observations = get_observations(model)
    if context is None:
        solver = ba.GeometricBundleOptimizer(options, setup)
        solver.set_up(model, geometric_loss)
        feature_count = 0
    else:
        feature_loss = pyceres.LossFunction({"name": "huber", "params": [3 * scales["sigma_feature_norm"]], "magnitude": scales["lambda_feature"]})
        losses.append(feature_loss)
        solver = ba.FeatureReferenceBundleOptimizer(options, setup, context["interpolation"])
        solver.set_up(model, feature_loss, context["view"], context["references"])
        feature_count = solver.problem.num_residual_blocks()
        for point_id, image_id, feature_id, measured in observations:
            image = model.images[image_id]
            camera = model.cameras[image.camera_id]
            cost = pyceres.factors.BundleAdjustmentCost(camera.model_id, measured)
            costs.append(cost)
            solver.problem.add_residual_block(cost, geometric_loss, [image.qvec, image.tvec, model.points3D[point_id].xyz, camera.params])
    problem = solver.problem
    before = problem.num_residual_blocks()
    prior_loss = pyceres.LossFunction({"name": "huber", "params": [3.0], "magnitude": information_scale})
    losses.append(prior_loss)
    covariance = np.diag([scales["sigma_horizontal_m"] ** 2] * 2 + [scales["sigma_height_m"] ** 2])
    point_ids = np.asarray(anchors["point_ids"], dtype=np.int64)
    if len(set(point_ids)) != len(point_ids) or not np.all(np.isfinite(covariance)) or np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise ValueError("Priors must have unique XYZ identities and finite positive covariance")
    pointers, views = [], []
    for point_id, mean in zip(point_ids, anchors["means"]):
        xyz = model.points3D[int(point_id)].xyz
        if xyz.dtype != np.float64 or not xyz.flags.c_contiguous or not problem.has_parameter_block(xyz) or problem.parameter_block_size(xyz) != 3 or problem.is_parameter_block_constant(xyz):
            raise ValueError("Map XYZ is absent, copied or fixed in native image factors")
        if problem.has_parameter_block(xyz.copy()):
            raise ValueError("An independent XYZ copy unexpectedly shares the native parameter block")
        pointers.append(int(xyz.ctypes.data))
        views.append(xyz)
        original = xyz[0]
        xyz[0] += 0.001
        if model.points3D[int(point_id)].xyz[0] != xyz[0]:
            raise ValueError("XYZ view does not mutate native model memory")
        xyz[0] = original
        if information_scale > 0:
            cost = pyceres.factors.NormalPrior(np.asarray(mean, dtype=np.float64), covariance)
            costs.append(cost)
            problem.add_residual_block(cost, prior_loss, [xyz])
        if fixed_points:
            xyz[:] = mean
            problem.set_parameter_block_constant(xyz)
    pose_states = []
    for image_id, image in model.images.items():
        pose_states.append({"image_id": image_id, "qvec_size": problem.parameter_block_size(image.qvec) if problem.has_parameter_block(image.qvec) else None, "qvec_tangent": problem.parameter_block_tangent_size(image.qvec) if problem.has_parameter_block(image.qvec) else None, "quaternion_manifold": problem.has_manifold(image.qvec) if problem.has_parameter_block(image.qvec) else None, "pose_variable": problem.has_parameter_block(image.qvec) and not problem.is_parameter_block_constant(image.qvec) and not problem.is_parameter_block_constant(image.tvec)})
    if not gauge_control and not all(row["pose_variable"] and row["qvec_size"] == 4 and row["qvec_tangent"] == 3 and row["quaternion_manifold"] for row in pose_states):
        raise ValueError("All six poses must be variable with quaternion manifolds")
    if not all(problem.is_parameter_block_constant(camera.params) for camera in model.cameras.values()):
        raise ValueError("Intrinsics must remain constant")
    prior_count = len(point_ids) if information_scale > 0 else 0
    if problem.num_residual_blocks() - before != prior_count:
        raise ValueError("Map prior residual count is not one per unique shared XYZ")
    return {"model": model, "solver": solver, "problem": problem, "setup": setup, "losses": losses, "costs": costs, "views": views, "context": context, "observations": observations, "anchors": anchors, "scales": scales, "information_scale": information_scale, "covariance": covariance, "iteration_recorder": iteration_recorder, "audit": {"point_ids": point_ids.tolist(), "xyz_addresses": pointers, "feature_blocks": feature_count, "geometric_blocks": before - feature_count, "map_blocks": prior_count, "total_blocks": problem.num_residual_blocks(), "pose_states": pose_states, "intrinsics_fixed": True, "use_inner_iterations": False, "xyz_shared_and_variable": not fixed_points}}

def get_candidate_projection(parameters):
    """Project the current Ceres buffers after finite-value and positive-depth checks."""
    if not all(np.isfinite(block).all() for block in parameters):
        return None, "nonfinite_parameter"
    quaternion, translation, point, intrinsics = parameters
    w, x, y, z = quaternion
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    camera_point = rotation @ point + translation
    if not np.isfinite(camera_point).all() or camera_point[2] <= 0:
        return None, "nonfinite_or_nonpositive_camera_depth"
    pixel = camera_point[:2] / camera_point[2] * intrinsics[:2] + intrinsics[2:]
    return (pixel, None) if np.isfinite(pixel).all() else (None, "nonfinite_projection")

def check_feature_domain(patch, pixel):
    """Require every configured bicubic sample to have its complete four-pixel support."""
    coordinate = patch.get_pixel_coords(pixel)
    return bool(np.isfinite(coordinate).all() and np.all(coordinate >= 1) and coordinate[0] < patch.width-2 and coordinate[1] < patch.height-2)

class GuardImageCost(pyceres.CostFunction):
    """Guard each native image evaluation using the actual Ceres trial buffers."""
    def __init__(self, native, identity, patch=None, reference=None, restore_sigma=None):
        """Declare fixed parameter blocks and retain the native factor and feature storage."""
        super().__init__()
        self.native, self.identity, self.patch = native, tuple(identity), patch
        self.reference, self.restore_sigma = reference, restore_sigma
        self.set_num_residuals(4 if restore_sigma is not None else (128 if patch is not None else 2))
        self.set_parameter_block_sizes([4, 3, 3, 4])
        self.calls, self.samples, self.rejected = 0, 0, 0

    def Evaluate(self, parameters, output, jacobians):
        """Reject invalid projections before sampling and copy only requested native derivatives."""
        self.calls += 1
        pixel, reason = get_candidate_projection(parameters)
        if reason is not None or (self.patch is not None and self.restore_sigma is None and not check_feature_domain(self.patch, pixel)):
            self.rejected += 1
            return False
        values, derivatives = self.native.evaluate(*parameters)
        if values is None or not np.isfinite(values).all():
            return False
        values = values.reshape(-1)
        if self.restore_sigma is not None:
            coordinate = self.patch.get_pixel_coords(pixel)
            scale = np.asarray(self.patch.scale)
            distances = np.array([1.25-coordinate[0], coordinate[0]-(self.patch.width-2.25), 1.25-coordinate[1], coordinate[1]-(self.patch.height-2.25)])
            output[:] = np.maximum(distances, 0) / np.repeat(scale*self.restore_sigma, 2)
            multipliers = (distances > 0) * np.array([-1., 1., -1., 1.]) / self.restore_sigma
            if jacobians is not None:
                for target, derivative in zip(jacobians, derivatives):
                    if target is not None:
                        target[:] = (derivative[[0, 0, 1, 1]] * multipliers[:, None]).ravel()
        else:
            output[:] = values if self.reference is None else values-self.reference
            if self.patch is not None:
                self.samples += 1
            if jacobians is not None:
                for target, derivative in zip(jacobians, derivatives):
                    if target is not None:
                        if not np.isfinite(derivative).all():
                            return False
                        target[:] = derivative.ravel()
        return bool(np.isfinite(output).all())

def create_dense_feature_context(model, cache_path, compute_references=True):
    """Adapt immutable full-image float16 maps and optionally recompute strict training references."""
    feature_set = features.FeatureSet_f16(128)
    arrays, maps = [], []
    with h5py.File(cache_path, "r") as cache:
        for image_id, image in model.images.items():
            group = cache[image.name]
            array = group["features"][:][None]
            feature_map = features.FeatureMap(array, [0], [np.zeros(2, dtype=np.int32)], {"is_sparse": False, "scale": np.asarray(group.attrs["scale"], dtype=np.float64)})
            feature_set.add_fmap(image.name, feature_map)
            arrays.append(array)
            maps.append(feature_map)
    feature_set.lock()
    interpolation = InterpolationConfig(interpolation_default_conf)
    view = features.FeatureView(feature_set, model)
    context = {"feature_set": feature_set, "view": view, "arrays": arrays, "maps": maps, "interpolation": interpolation, "interpolator": features.PatchInterpolator(interpolation), "references": {}, "full_image": True}
    if compute_references:
        domain = check_model_domain(model, get_observations(model), context)
        if domain["invalid_count"]:
            raise ValueError("Training visual reference inputs are outside strict full-image support")
        loss = pyceres.LossFunction({"name": "cauchy", "params": [0.25], "magnitude": 1.})
        config = ba.ReferenceConfig({"iters": 10, "keep_observations": True, "compute_offsets3D": False, "num_threads": 1})
        config.loss = loss
        extractor = ba.ReferenceExtractor(config, interpolation)
        labels = [-1] * (max(model.points3D)+1)
        for point_id in model.points3D:
            labels[point_id] = point_id // 10
        context.update(references=extractor.run(labels, model, feature_set), extractor=extractor, reference_loss=loss)
    return context

def check_model_domain(model, observations, context=None):
    """Classify every fixed observation without clipping, sampling or dropping invalid predictions."""
    invalid, outside_image = [], []
    for point_id, image_id, feature_id, measured in observations:
        image = model.images[image_id]
        camera = model.cameras[image.camera_id]
        pixel, reason = get_candidate_projection([image.qvec, image.tvec, model.points3D[point_id].xyz, camera.params])
        if reason is None:
            outside = not (0 <= pixel[0] < camera.width and 0 <= pixel[1] < camera.height)
            if outside:
                outside_image.append([point_id, image_id, feature_id])
            if context is not None:
                patch = context["view"].fpatch(image_id, feature_id)
                if outside:
                    reason = "outside_full_image"
                elif not check_feature_domain(patch, pixel):
                    reason = "outside_full_bicubic_support"
        if reason is not None:
            invalid.append({"point3D_id": point_id, "image_id": image_id, "image_name": image.name, "feature_id": feature_id, "predicted_xy": None if pixel is None else pixel.tolist(), "reason": reason})
    return {"sample_count": len(observations), "valid_count": len(observations)-len(invalid), "invalid_count": len(invalid), "outside_image_count": len(outside_image), "outside_image_observations": outside_image, "invalid_predictions": invalid, "domain": "geometry" if context is None else "full_bicubic"}

def build_shared_problem(model, anchors, scales, information_scale, context=None, fixed_points=False, gauge_control=False, restore_context=None, beta=None):
    """Explicitly assemble guarded image factors and one movable map prior per shared point."""
    if not all(np.isfinite(scales[key]) and scales[key] > 0 for key in ["sigma_horizontal_m", "sigma_height_m", "sigma_pixel_px"]) or not np.isfinite(information_scale) or information_scale < 0:
        raise ValueError("Working scales must be finite positive and information scale nonnegative")
    ids = np.asarray(anchors["point_ids"], dtype=np.int64)
    if len(set(ids)) != len(ids):
        raise ValueError("Priors must have unique XYZ identities")
    problem = pyceres.Problem()
    options = pyceres.SolverOptions()
    options.max_num_iterations = 100 if context is not None or restore_context is not None else 300
    options.function_tolerance, options.gradient_tolerance, options.parameter_tolerance = 1e-6, 1e-10, 1e-8
    options.use_inner_iterations, options.num_threads, options.minimizer_progress_to_stdout = False, 1, False
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    recorder = RecordSolverIterations()
    options.callbacks = [recorder]
    sigma = scales["sigma_pixel_px"]
    geometric_loss = pyceres.LossFunction({"name": "huber", "params": [3*sigma], "magnitude": 1/sigma**2})
    prior_loss = pyceres.LossFunction({"name": "huber", "params": [3.], "magnitude": information_scale})
    feature_loss = None if context is None else pyceres.LossFunction({"name": "huber", "params": [3*scales["sigma_feature_norm"]], "magnitude": scales["lambda_feature"]})
    restore_loss = None if beta is None else pyceres.LossFunction({"name": "trivial", "params": [], "magnitude": beta})
    losses = [geometric_loss, prior_loss, feature_loss, restore_loss]
    costs, factors, views, manifolds = [], [], [], []
    observations = get_observations(model)
    for point_id, image_id, feature_id, measured in observations:
        image = model.images[image_id]
        camera = model.cameras[image.camera_id]
        blocks = [image.qvec, image.tvec, model.points3D[point_id].xyz, camera.params]
        identity = (point_id, image_id, feature_id)
        native = pyceres.factors.BundleAdjustmentCost(camera.model_id, measured)
        cost = GuardImageCost(native, identity)
        problem.add_residual_block(cost, geometric_loss, blocks)
        costs.append(cost)
        factors.append({"kind": "geometric", "identity": list(identity), "addresses": [int(block.ctypes.data) for block in blocks]})
        views.extend(blocks)
        if context is not None:
            patch = context["view"].fpatch(image_id, feature_id)
            reference = np.asarray(context["references"][point_id].descriptor).reshape(-1)
            native = native_residuals.FeatureReferenceCostFunctor(camera.model_id, patch, reference.reshape(1, 128), context["interpolation"])
            cost = GuardImageCost(native, identity, patch, reference)
            problem.add_residual_block(cost, feature_loss, blocks)
            costs.append(cost)
            factors.append({"kind": "feature", "identity": list(identity), "reference_point_id": point_id, "addresses": [int(block.ctypes.data) for block in blocks]})
        if restore_context is not None:
            cost = GuardImageCost(native, identity, restore_context["view"].fpatch(image_id, feature_id), restore_sigma=sigma)
            problem.add_residual_block(cost, restore_loss, blocks)
            costs.append(cost)
    pose_states = []
    for image_id, image in model.images.items():
        manifold = pyceres.QuaternionManifold()
        manifolds.append(manifold)
        problem.set_manifold(image.qvec, manifold)
        pose_states.append({"image_id": image_id, "qvec_size": problem.parameter_block_size(image.qvec), "qvec_tangent": problem.parameter_block_tangent_size(image.qvec), "quaternion_manifold": problem.has_manifold(image.qvec), "pose_variable": True})
    for camera in model.cameras.values():
        problem.set_parameter_block_constant(camera.params)
    if gauge_control:
        images = [model.images[index] for index in sorted(model.images)]
        problem.set_parameter_block_constant(images[0].qvec)
        problem.set_parameter_block_constant(images[0].tvec)
        manifold = pyceres.SubsetManifold(3, [0])
        manifolds.append(manifold)
        problem.set_manifold(images[1].tvec, manifold)
    covariance = np.diag([scales["sigma_horizontal_m"]**2]*2+[scales["sigma_height_m"]**2])
    pointers = []
    for point_id, mean in zip(ids, anchors["means"]):
        xyz = model.points3D[int(point_id)].xyz
        if not problem.has_parameter_block(xyz) or problem.has_parameter_block(xyz.copy()) or problem.is_parameter_block_constant(xyz):
            raise ValueError("Shared native XYZ is absent, copied or fixed")
        pointers.append(int(xyz.ctypes.data))
        views.append(xyz)
        if information_scale > 0:
            cost = pyceres.factors.NormalPrior(np.asarray(mean, dtype=np.float64), covariance)
            costs.append(cost)
            problem.add_residual_block(cost, prior_loss, [xyz])
        if fixed_points:
            xyz[:] = mean
            problem.set_parameter_block_constant(xyz)
    feature_count = len(observations) if context is not None else 0
    prior_count = len(ids) if information_scale > 0 else 0
    expected = len(observations)+feature_count+prior_count+(len(observations) if restore_context is not None else 0)
    if problem.num_residual_blocks() != expected:
        raise ValueError("Actual factor count differs from explicit observation and prior sets")
    return {"model": model, "problem": problem, "options": options, "costs": costs, "losses": losses, "views": views, "manifolds": manifolds, "context": context, "observations": observations, "anchors": anchors, "scales": scales, "information_scale": information_scale, "covariance": covariance, "iteration_recorder": recorder, "restore_context": restore_context, "beta": beta, "audit": {"point_ids": ids.tolist(), "xyz_addresses": pointers, "feature_blocks": feature_count, "geometric_blocks": len(observations), "map_blocks": prior_count, "restore_blocks": len(observations) if restore_context is not None else 0, "total_blocks": expected, "pose_states": pose_states, "intrinsics_fixed": True, "use_inner_iterations": False, "xyz_shared_and_variable": not fixed_points, "actual_factors": factors}}

def compute_huber_cost(squared_norms, threshold):
    """Evaluate radial Huber rho with the same squared-norm convention as Ceres."""
    squared_norms = np.asarray(squared_norms)
    return np.where(squared_norms <= threshold ** 2, squared_norms, 2 * threshold * np.sqrt(squared_norms) - threshold ** 2)

def compute_problem_cost(state):
    """Compute each weighted factor family independently for comparison with Ceres cost."""
    model, observations, scales = state["model"], state["observations"], state["scales"]
    residuals = project_observations(model, observations) - np.array([row[3] for row in observations])
    geometric = np.sum(compute_huber_cost(np.sum(residuals ** 2, axis=1), 3 * scales["sigma_pixel_px"])) / scales["sigma_pixel_px"] ** 2 / 2
    point_ids = state["anchors"]["point_ids"]
    residuals = np.array([model.points3D[int(point_id)].xyz for point_id in point_ids]) - state["anchors"]["means"]
    mahalanobis = np.sum(residuals ** 2 / np.diag(state["covariance"]), axis=1)
    prior = state["information_scale"] * np.sum(compute_huber_cost(mahalanobis, 3)) / 2
    feature = 0.
    if state["context"] is not None:
        residuals, valid = sample_feature_residuals(model, observations, state["context"])
        if not valid.all():
            raise ValueError("Projection left valid feature patches; refusing zero or clamped residuals")
        feature = scales["lambda_feature"] * np.sum(compute_huber_cost(np.sum(residuals ** 2, axis=1), 3 * scales["sigma_feature_norm"])) / 2
    result = {"geometric": float(geometric), "map": float(prior), "feature": float(feature)}
    if state.get("restore_context") is not None:
        total = 0.
        for cost in state["costs"]:
            if isinstance(cost, GuardImageCost) and cost.restore_sigma is not None:
                point_id, image_id, feature_id = cost.identity
                image = model.images[image_id]
                blocks = [image.qvec, image.tvec, model.points3D[point_id].xyz, model.cameras[image.camera_id].params]
                values, derivatives = cost.evaluate(*blocks)
                if values is None:
                    raise ValueError("Invalid accepted restoration state")
                total += float(np.sum(values**2))
        result["restore"] = state["beta"]*total/2
    return result

def solve_shared_problem(state):
    """Solve the audited Ceres problem and verify costs, factor counts and shared XYZ identity."""
    initial = compute_problem_cost(state)
    if "options" in state:
        summary = pyceres.SolverSummary()
        pyceres.solve(state["options"], state["problem"], summary)
    else:
        state["solver"].solve_problem(state["model"])
        summary = state["solver"].summary()
    if not summary.IsSolutionUsable():
        return {"usable": False, "converged": False, "termination_type": str(summary.termination_type), "termination_message": summary.message, "solver_full_report": summary.FullReport(), "iterations": state["iteration_recorder"].iterations, "audit": state["audit"]}
    final = compute_problem_cost(state)
    if not np.isclose(sum(initial.values()), summary.initial_cost, rtol=1e-9, atol=1e-7) or not np.isclose(sum(final.values()), summary.final_cost, rtol=1e-9, atol=1e-7):
        raise ValueError("Independently computed weighted factor costs disagree with Ceres")
    if state["problem"].num_residual_blocks() != state["audit"]["total_blocks"]:
        raise ValueError("Added factors disappeared during solving")
    return {"solver_report": summary.BriefReport(), "solver_full_report": summary.FullReport(), "termination_message": summary.message, "usable": True, "converged": summary.termination_type == pyceres.TerminationType.CONVERGENCE, "termination_type": str(summary.termination_type), "initial_costs": initial, "final_costs": final, "ceres_initial_cost": summary.initial_cost, "ceres_final_cost": summary.final_cost, "successful_steps": summary.num_successful_steps, "unsuccessful_steps": summary.num_unsuccessful_steps, "iterations": state["iteration_recorder"].iterations, "iteration_field_definitions": {"cost": "raw Ceres iteration objective value", "cost_change": "raw Ceres cost change; positive means a reduction", "gradient_norm": "gradient L2 norm", "gradient_max_norm": "gradient infinity norm", "step_norm": "step L2 norm", "step_size": "line-search step size; not the trust-region step norm", "trust_region_radius": "trust-region radius at the end of the iteration", "relative_decrease": "actual versus predicted cost reduction ratio", "step_is_successful": "whether the solver accepted this step", "iteration_zero": "initial evaluation, not an optimization update", "nonfinite_values": "nonfinite raw values are named explicitly and represented as JSON null"}, "audit": state["audit"]}

def create_synthetic_scene():
    """Create six image-only cameras, noisy observations and known noisy map means for controls."""
    rng = np.random.default_rng(12)
    truth = rng.uniform([-1.2, -1.2, 8], [1.2, 1.2, 12], (24, 3))
    means = truth + rng.normal(size=truth.shape) * [0.3, 0.3, 0.2]
    model = pycolmap.Reconstruction()
    model.add_camera(pycolmap.Camera("PINHOLE", 640, 480, [800, 800, 320, 240], id=1))
    for index in range(6):
        center = np.array([(index - 2.5) * 0.5, 0.3 * np.sin(index), 0])
        camera_points = truth - center
        pixels = camera_points[:, :2] / camera_points[:, 2:] * 800 + [320, 240]
        pixels += rng.normal(0, 0.2, pixels.shape)
        model.add_image(pycolmap.Image(f"{index}.png", keypoints=list(pixels), tvec=-center, camera_id=1, id=index + 1))
        model.register_image(index + 1)
    point_ids = []
    for index, xyz in enumerate(truth):
        track = pycolmap.Track([pycolmap.TrackElement(image_id, index) for image_id in range(1, 7)])
        point_ids.append(model.add_point3D(xyz + rng.normal(0, 0.05, 3), track))
    return model, {"point_ids": np.asarray(point_ids), "means": means}, truth

def run_synthetic_controls():
    """Compare no, weak, candidate, strong and fixed-map controls with synthetic truth only."""
    initial, anchors, truth = create_synthetic_scene()
    scales = {"sigma_horizontal_m": 0.3, "sigma_height_m": 0.2, "sigma_pixel_px": 0.2}
    results = []
    for label, weight, fixed in [("none", 0., False), ("weak", 0.1, False), ("candidate", 1., False), ("strong", 10., False), ("fixed_map", 1., True)]:
        model = copy.deepcopy(initial)
        state = build_shared_problem(model, anchors, scales, weight, fixed_points=fixed, gauge_control=weight == 0)
        before = np.array([model.points3D[int(point_id)].xyz.copy() for point_id in anchors["point_ids"]])
        result = solve_shared_problem(state)
        points = np.array([model.points3D[int(point_id)].xyz for point_id in anchors["point_ids"]])
        transform, inliers = estimate_sim3(points, truth, 2.)
        result.update(label=label, information_scale=weight, shape_rmse_m=float(np.sqrt(np.mean(np.sum((apply_sim3(points, transform) - truth) ** 2, axis=1)))), displacement_median_m=float(np.median(np.linalg.norm(points - before, axis=1))), map_residual_median_m=float(np.median(np.linalg.norm(points - anchors["means"], axis=1))))
        results.append(result)
    finite = next(row for row in results if row["label"] == "candidate")
    fixed = next(row for row in results if row["label"] == "fixed_map")
    if finite["shape_rmse_m"] >= fixed["shape_rmse_m"] or finite["displacement_median_m"] <= 0:
        raise ValueError("Synthetic controls do not demonstrate image correction of movable map-linked points")
    return results

def get_error_statistics(values):
    """Report fixed-sample median, population std, maximum and count without dropping invalids."""
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Validation statistics require every evaluation sample to be finite")
    return {"median": float(np.median(values)), "std": float(np.std(values, ddof=0)), "max": float(np.max(values)), "count": len(values)}

def get_array_digest(values):
    """Hash an array's exact dtype, dimensions and ordered contents."""
    array = np.ascontiguousarray(values)
    return hashlib.sha256(str((array.dtype.str, array.shape)).encode()+array.tobytes()).hexdigest()

def get_model_fingerprint(model):
    """Fingerprint the actual camera, pose, XYZ and reciprocal observation state."""
    return {"poses": get_array_digest(np.array([np.r_[index, model.images[index].qvec, model.images[index].tvec] for index in sorted(model.images)])), "points": get_array_digest(np.array([np.r_[index, model.points3D[index].xyz] for index in sorted(model.points3D)])), "cameras": get_array_digest(np.array([np.r_[index, model.cameras[index].params] for index in sorted(model.cameras)])), "observations": get_observation_digest(get_observations(model))}

def compute_gauge_information(anchors, covariances, positions, information_scale, state_name):
    """Compute the prescribed seven-dimensional whitened robust Sim(3) IRLS information."""
    means = np.asarray(anchors["means"], dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    covariances = np.broadcast_to(covariances, (len(means), 3, 3))
    center = np.mean(means, axis=0)
    span = float(np.sqrt(np.mean(np.sum((means-center)**2, axis=1))))
    if not np.isfinite(span) or span <= 0 or not np.isfinite(positions).all() or not np.isfinite(covariances).all() or not np.isfinite(information_scale) or information_scale <= 0:
        raise ValueError("Invalid gauge inputs, map span or weight")
    diagonal = np.diagonal(covariances, axis1=1, axis2=2)
    if np.any(diagonal <= 0) or not np.array_equal(covariances, np.eye(3)[None]*diagonal[:, None, :]):
        raise ValueError("Gauge covariance must be diagonal and positive definite")
    whitening = 1/np.sqrt(diagonal)
    residual = (positions-means)*whitening
    squared = np.sum(residual**2, axis=1)
    weights = np.ones(len(means))
    outside = squared > 9
    weights[outside] = 3/np.sqrt(squared[outside])
    jacobians = np.zeros((len(means), 3, 7), dtype=np.float64)
    jacobians[:, :, :3] = span*np.eye(3)
    for index, (x, y, z) in enumerate(positions-center):
        jacobians[index, :, 3:6] = [[0, z, -y], [-z, 0, x], [y, -x, 0]]
    jacobians[:, :, 6] = positions-center
    jacobians *= whitening[:, :, None]*np.sqrt(information_scale*weights)[:, None, None]
    matrix = np.einsum("nki,nkj->ij", jacobians, jacobians)/len(means)
    matrix = (matrix+matrix.T)/2
    eigenvalues = np.linalg.eigvalsh(matrix)
    valid = bool(np.isfinite(matrix).all() and np.isfinite(eigenvalues).all() and eigenvalues[-1] > 0 and eigenvalues[0] > 1e-10*eigenvalues[-1])
    return {"state_name": state_name, "passed": valid, "center_enu_m": center.tolist(), "span_m": span, "parameter_order": ["tau_x", "tau_y", "tau_z", "omega_x", "omega_y", "omega_z", "kappa"], "lambda_map": information_scale, "huber_delta": 3., "robust_weights": weights.tolist(), "matrix": matrix.tolist(), "eigenvalues": eigenvalues.tolist(), "eigenvalue_ratio": float(eigenvalues[0]/eigenvalues[-1]) if eigenvalues[-1] > 0 else None, "xyz_sha256": get_array_digest(positions), "mu_sha256": get_array_digest(means), "covariance_sha256": get_array_digest(covariances), "prior_ids_sha256": get_array_digest(np.arange(len(means), dtype=np.int64))}

def select_reference_source(descriptors):
    """Reproduce native ten-step Cauchy IRLS and its first nearest actual descriptor."""
    descriptors = np.asarray(descriptors, dtype=np.float64)
    weights = np.ones(len(descriptors))
    for iteration in range(10):
        weights /= np.sum(weights)
        mean = np.sum(descriptors*weights[:, None], axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean /= norm
        squared = np.sum((descriptors-mean)**2, axis=1)
        losses = np.array([0.0625*math.log(1+float(value)*16) for value in squared])
        exact = np.flatnonzero(losses == 0)
        if len(exact):
            mean = descriptors[exact[0]]
            break
        weights = 1/losses
    distances = np.sum((descriptors-mean)**2, axis=1)
    return int(np.argmin(distances)), distances

def get_reference_policy(root, context):
    """Freeze effective native reference parameters, boundary semantics and implementation identity."""
    files = ["pixsfm/bundle_adjustment/src/reference_extractor.h", "pixsfm/base/src/irls_optim.h", "pixsfm/features/src/featurepatch.h", "pixsfm/base/src/interpolation.h", "pixsfm/base/src/projection.h"]
    extension = Path(__import__("pixsfm._pixsfm", fromlist=["x"]).__file__)
    interpolation = context["interpolation"]
    fields = ["nodes", "mode", "l2_normalize", "ncc_normalize", "use_float_simd", "check_bounds"]
    resolved = {field: str(getattr(interpolation, field)) for field in fields}
    if "bool fill_channel_differences = true;" not in (root/"pixsfm/base/src/interpolation.h").read_text():
        raise ValueError("Unexposed native interpolation default differs from reference policy")
    resolved["fill_channel_differences"] = {"value": True, "source": "native default; not exposed to Python"}
    resolved["upsampling_factor"] = sorted({float(feature_map.fpatch(0).upsampling_factor) for feature_map in context["maps"]})
    return {"algorithm": "native RobustMeanIRLS then closest input; first exact tie in track order", "iters": 10, "keep_observations": True, "compute_offsets3D": False, "closest_to_robust_mean": True, "num_threads": 1, "loss": {"name": "cauchy", "a": 0.25, "magnitude": 1}, "interpolation": resolved, "boundary_policy": "external strict 1<=scale*projection-0.5<size-2 before native check_bounds=False", "sampling_model": "public_training_visual_reference_before_map_assignment", "dtype": "float16 input; float64 interpolation and references", "source_sha256": {name: get_sha256(root/name) for name in files}, "extension_path": str(extension), "extension_sha256": get_sha256(extension)}

def save_training_references(model, heldout, context, path, root):
    """Audit and save the exact ordered native reference inputs and selected training sources."""
    provenance = audit_training_references(model, heldout, context)
    ids = sorted(context["references"])
    participants, pixels, coordinates, inputs, sources, descriptors, offsets, costs, scales, corners = [], [], [], [], [], [], [0], [], [], []
    projected = {row[:3]: pixel for row, pixel in zip(get_observations(model), project_observations(model, get_observations(model)))}
    for point_id in ids:
        reference = context["references"][point_id]
        expected = [(entry.image_id, entry.point2D_idx) for entry in model.points3D[point_id].track.elements]
        actual = [(entry.image_id, entry.point2D_idx) for entry in reference.track.elements]
        if actual != expected:
            raise ValueError("Native reference changed ordered training contributors")
        values = np.array([np.asarray(value).reshape(-1) for value in reference.observations])
        source_prediction, distances = select_reference_source(values)
        native_distances = np.asarray(reference.costs, dtype=np.float64)
        require_numeric_match(distances, native_distances, 1e-10, 1e-8, "native reference IRLS distances")
        source_index = int(np.argmin(native_distances))
        sorted_distances = np.sort(native_distances)
        if len(sorted_distances) > 1 and sorted_distances[1]-sorted_distances[0] > 2e-10+2e-8*max(sorted_distances[1], sorted_distances[0]) and source_prediction != source_index:
            raise ValueError("Native unique reference source differs from IRLS algorithm")
        source_identity = (reference.source.image_id, reference.source.point2D_idx)
        if source_identity != actual[source_index] or not np.allclose(np.asarray(reference.descriptor).reshape(-1), values[source_index], atol=1e-10, rtol=1e-8):
            raise ValueError("Native reference source disagrees with frozen Cauchy IRLS policy")
        costs.extend(native_distances.tolist())
        sources.append(source_identity)
        descriptors.append(np.asarray(reference.descriptor).reshape(-1))
        for entry, value in zip(reference.track.elements, values):
            identity = (point_id, entry.image_id, entry.point2D_idx)
            patch = context["view"].fpatch(entry.image_id, entry.point2D_idx)
            pixel = projected[identity]
            if not check_feature_domain(patch, pixel):
                raise ValueError("Reference input lacks strict full bicubic support")
            participants.append(identity)
            pixels.append(pixel)
            coordinates.append(patch.get_pixel_coords(pixel))
            scales.append(np.asarray(patch.scale))
            corners.append(np.asarray(patch.corner))
            inputs.append(value)
        offsets.append(len(participants))
    arrays = {"point3D_ids": np.array(ids, dtype=np.int64), "descriptors": np.array(descriptors, dtype=np.float64), "source_identities": np.array(sources, dtype=np.int64), "participant_offsets": np.array(offsets, dtype=np.int64), "participants": np.array(participants, dtype=np.int64), "predicted_xy": np.array(pixels), "feature_coordinates": np.array(coordinates), "input_descriptors": np.array(inputs), "native_reference_distances": np.array(costs), "scales": np.array(scales), "corners": np.array(corners)}
    np.savez_compressed(path, **arrays)
    provenance.update(recomputed=True, migration="strict full-image reference recomputation", model_fingerprint=get_model_fingerprint(model), ordered_arrays_sha256={name: get_array_digest(value) for name, value in arrays.items()}, reference_policy=get_reference_policy(root, context), path=str(path), file_sha256=get_sha256(path))
    return provenance

def load_training_references(path):
    """Load immutable worker descriptors without recomputing them on candidate geometry."""
    with np.load(path) as archive:
        return {int(point_id): SimpleNamespace(descriptor=descriptor.copy()) for point_id, descriptor in zip(archive["point3D_ids"], archive["descriptors"])}

def initialize_map_points(model, anchors):
    """Set each reliable shared XYZ to its immutable map mean and preserve visual lineage."""
    registered = np.array([model.points3D[int(index)].xyz.copy() for index in anchors["point_ids"]])
    for point_id, mean in zip(anchors["point_ids"], anchors["means"]):
        model.points3D[int(point_id)].xyz[:] = mean
    initial = np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]])
    if not np.array_equal(initial, anchors["means"]):
        raise ValueError("Shared XYZ did not retain exact map initialization")
    return {"point_ids": np.asarray(anchors["point_ids"]).tolist(), "registered_sfm_xyz_enu_m": registered.tolist(), "map_xyz_enu_m": anchors["means"].tolist(), "initial_xyz_enu_m": initial.tolist(), "initialization_source": "map_mu", "model_fingerprint": get_model_fingerprint(model)}

def select_weight_candidates(candidates):
    """Apply the declared material-difference dominance rule without tie-breaking by weight."""
    valid = [row for row in candidates if row["status"] == "passed"]
    if len(valid) < 2:
        return {"selection_status": "insufficient_candidates" if valid else "no_valid_candidate", "proposed_weight": None, "candidate_id": None}
    metrics = np.array([[row["heldout_reprojection_px"]["median"], row["heldout_feature_normalized"]["median"], row["heldout_reprojection_px"]["std"], row["heldout_feature_normalized"]["std"]] for row in valid])
    if not np.isfinite(metrics).all():
        raise ValueError("Candidate comparison contains nonfinite metrics")
    proposed, has_tradeoff = [], False
    for index, values in enumerate(metrics):
        dominates = True
        for other_index, other in enumerate(metrics):
            if index == other_index:
                continue
            delta = np.maximum([0.01, 0.01, 0.02, 0.02], 0.01*np.maximum(values, other))
            difference = values-other
            has_tradeoff = has_tradeoff or bool(np.any(difference > delta) and np.any(difference < -delta))
            dominates = dominates and bool(np.all(difference <= delta) and np.any(difference < -delta))
        if dominates:
            proposed.append(index)
    if len(proposed) == 1:
        selected = valid[proposed[0]]
        return {"selection_status": "proposed", "proposed_weight": selected["information_scale"], "candidate_id": selected["candidate_id"]}
    return {"selection_status": "tradeoff" if has_tradeoff else "indistinguishable", "proposed_weight": None, "candidate_id": None}

def require_numeric_match(first, second, atol, rtol, label):
    """Enforce the frozen symmetric elementwise tolerance without masking nonfinite values."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all() or not np.all(np.abs(first-second) <= atol+rtol*np.maximum(np.abs(first), np.abs(second))):
        raise ValueError(f"Fixed numerical equivalence failed: {label}")
    return float(np.max(np.abs(first-second))) if first.size else 0.

def audit_factor_identity(state):
    """Verify actual guarded factors and native buffers against every expected training observation."""
    model = state["model"]
    expected = [list(row[:3]) for row in state["observations"]]
    for kind in ["geometric"]+(["feature"] if state["context"] is not None else []):
        rows = [row for row in state["audit"]["actual_factors"] if row["kind"] == kind]
        if [row["identity"] for row in rows] != expected:
            raise ValueError("Actual factor identities differ from ordered training observations")
        for row in rows:
            point_id, image_id, feature_id = row["identity"]
            image = model.images[image_id]
            blocks = [image.qvec, image.tvec, model.points3D[point_id].xyz, model.cameras[image.camera_id].params]
            if row["addresses"] != [int(block.ctypes.data) for block in blocks]:
                raise ValueError("Actual factor addresses do not reference current native model buffers")
            if kind == "feature" and row["reference_point_id"] != point_id:
                raise ValueError("Feature factor uses a different point's reference")
    if state["problem"].num_residual_blocks() != state["audit"]["total_blocks"]:
        raise ValueError("Actual residual count changed")
    for image_id, image in model.images.items():
        if state["problem"].is_parameter_block_constant(image.qvec) or state["problem"].is_parameter_block_constant(image.tvec) or state["problem"].parameter_block_tangent_size(image.qvec) != 3:
            raise ValueError("A formal camera pose is fixed or lacks its quaternion manifold")
    for point_id in model.points3D:
        if state["problem"].is_parameter_block_constant(model.points3D[point_id].xyz):
            raise ValueError("A shared point was fixed")
    if not all(state["problem"].is_parameter_block_constant(camera.params) for camera in model.cameras.values()):
        raise ValueError("Intrinsics became variable")
    return {"passed": True, "observation_count": len(expected), "identity_sha256": get_array_digest(np.asarray(expected, dtype=np.int64)), "parameter_addresses_verified": True, "map_prior_count": state["audit"]["map_blocks"]}

def get_derivative_policy():
    """Return the fixed multiscale reference policy independently of any analytic derivative."""
    return {"version": "multiscale_richardson_roundguard_v1", "rotation_h0": 1e-6, "translation_xyz_h0_m": 1e-4, "level_indices": list(range(7)), "step_ratio": 0.5, "window_indices": [2, 3, 4, 5], "channels": 128, "shared_window": True, "reference_function": "native descriptor before frozen reference subtraction", "epsilon": 2.220446049250313e-16, "rounding_coefficient": 8., "representation_rtol": 1e-6, "comparison_atol": 1e-5, "comparison_rtol": 1e-4, "stability_fraction": 0.1, "error": "E_diff+E_round; estimate, not a strict bound", "selection": "first common stable same-cell window; no analytic input"}

def check_actual_perturbation(original, negative, positive, block_index, axis, step):
    """Check actual buffer displacements and native quaternion chord lengths before sampling."""
    original, negative, positive = np.asarray(original), np.asarray(negative), np.asarray(positive)
    record = {"original": original.tolist(), "minus": negative.tolist(), "plus": positive.tolist(), "minus_displacement": (negative-original).tolist(), "plus_displacement": (positive-original).tolist(), "step": step}
    distinct = not np.array_equal(negative, original) and not np.array_equal(positive, original) and not np.array_equal(negative, positive)
    if block_index == 0:
        norm = np.linalg.norm(original)
        expected = np.array([2*norm*np.sin(step/2), 2*norm*np.sin(step/2), 2*norm*np.sin(step)])
        actual = np.array([np.linalg.norm(negative-original), np.linalg.norm(positive-original), np.linalg.norm(positive-negative)])
        errors = np.abs(actual-expected)/expected if np.all(expected > 0) else np.full(3, np.inf)
        record.update(expected_chords=expected.tolist(), actual_chords=actual.tolist(), relative_errors=[float(value) if np.isfinite(value) else None for value in errors])
        valid = distinct and np.isfinite(errors).all() and np.all(errors <= 1e-6)
    else:
        index = int(np.argmax(axis))
        others = np.arange(3) != index
        displacements = np.array([negative[index]-original[index], positive[index]-original[index]])
        errors = np.abs(displacements-np.array([-step, step]))/step
        valid = distinct and displacements[0] < 0 and displacements[1] > 0 and np.all(errors <= 1e-6) and np.array_equal(negative[others], original[others]) and np.array_equal(positive[others], original[others])
        record["relative_errors"] = errors.tolist()
    record.update(valid=bool(valid), reason=None if valid else "unrepresentable_perturbation")
    return record

def select_multiscale_reference(center, levels):
    """Select the first shared numerical window using only samples and rounding estimates."""
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (128,) or len(levels) != 7 or not np.isfinite(center).all():
        raise ValueError("Multiscale reference requires seven levels and 128 finite center values")
    epsilon = 2.220446049250313e-16
    rows, extrapolations, windows = [], [], []
    for index, level in enumerate(levels):
        row = dict(level, k=index, D=None, B_D=None)
        if row["valid"]:
            minus, plus = np.asarray(row["values_minus"]), np.asarray(row["values_plus"])
            if minus.shape != (128,) or plus.shape != (128,) or not np.isfinite(minus).all() or not np.isfinite(plus).all():
                row.update(valid=False, reason="nonfinite_or_invalid_function_values")
            else:
                derivative = (plus-minus)/(2*row["step"])
                magnitude = np.maximum.reduce([np.ones(128), np.abs(center), np.abs(minus), np.abs(plus)])
                rounding = 8*epsilon*magnitude
                bound = rounding/row["step"]+8*epsilon*np.abs(derivative)
                if not np.isfinite(derivative).all() or not np.isfinite(bound).all():
                    row.update(valid=False, reason="nonfinite_difference_or_rounding")
                else:
                    row.update(D=derivative.tolist(), magnitude=magnitude.tolist(), rounding_floor_minus=rounding.tolist(), rounding_floor_plus=rounding.tolist(), B_D=bound.tolist())
        rows.append(row)
    for index in range(6):
        row = {"k": index, "valid": False, "R": None, "B_R": None}
        if rows[index]["valid"] and rows[index+1]["valid"]:
            value = (4*np.asarray(rows[index+1]["D"])-np.asarray(rows[index]["D"]))/3
            bound = (4*np.asarray(rows[index+1]["B_D"])+np.asarray(rows[index]["B_D"]))/3+8*epsilon*np.abs(value)
            if np.isfinite(value).all() and np.isfinite(bound).all():
                row.update(valid=True, R=value.tolist(), B_R=bound.tolist())
        extrapolations.append(row)
    result = {"strategy": "multiscale_richardson_roundguard_v1", "status": "inconclusive", "selected_window": None, "reference": None, "error": None, "center_values": center.tolist(), "levels": rows, "extrapolations": extrapolations, "windows": windows}
    reasons = []
    for j in [2, 3, 4, 5]:
        raw = rows[j-2:j+2]
        window = {"j": j, "eligible": False, "accepted": False}
        if not all(row["valid"] for row in raw):
            window["reason"] = "invalid_raw_layer"
            reasons.extend(row.get("reason") for row in raw if not row["valid"])
        elif not all(row["same_segment"] for row in raw):
            window["reason"] = "cross_segment"
            reasons.append("cross_segment")
        elif not all(row["valid"] for row in extrapolations[j-2:j+1]):
            window["reason"] = "nonfinite_extrapolation"
            reasons.append(window["reason"])
        else:
            previous2, previous, current = [np.asarray(row["R"]) for row in extrapolations[j-2:j+1]]
            delta1, delta2 = np.abs(current-previous), np.abs(previous-previous2)
            first = (1e-5+1e-4*np.maximum(np.abs(current), np.abs(previous)))/10
            second = (1e-5+1e-4*np.maximum(np.abs(previous), np.abs(previous2)))/10
            difference = np.maximum(delta1, delta2)
            rounding = np.max([row["B_R"] for row in extrapolations[j-2:j+1]], axis=0)
            error = difference+rounding
            stable = bool(np.all(delta1 <= first) and np.all(delta2 <= second))
            resolved = bool(np.all(error <= np.minimum(first, second)))
            window.update(eligible=True, accepted=stable and resolved, S_current=first.tolist(), S_previous=second.tolist(), E_diff=difference.tolist(), E_round=rounding.tolist(), E=error.tolist(), reason=None if stable and resolved else ("rounding_limited" if stable else "unstable_estimate"))
            if window["accepted"]:
                windows.append(window)
                result.update(status="ready", selected_window=j, reference=current.tolist(), error=error.tolist(), reason=None)
                return result
            reasons.append(window["reason"])
        windows.append(window)
    result["reason"] = sorted({reason for reason in reasons if reason is not None})
    return result

def classify_multiscale_derivative(numerical, analytic):
    """Compare a frozen numerical reference with an analytic derivative using three outcomes."""
    if numerical["status"] != "ready":
        return {"status": "inconclusive", "reason": numerical.get("reason"), "channels": ["inconclusive"]*128}
    analytic = np.asarray(analytic)
    reference, error = np.asarray(numerical["reference"]), np.asarray(numerical["error"])
    if analytic.shape != (128,) or not np.isfinite(analytic).all():
        return {"status": "mismatch", "reason": "nonfinite_or_invalid_analytic_derivative", "channels": ["mismatch"]*128}
    difference = np.abs(analytic-reference)
    tolerance = 1e-5+1e-4*np.maximum(np.abs(analytic), np.abs(reference))
    statuses = np.full(128, "inconclusive", dtype="U12")
    statuses[difference+error <= tolerance] = "passed"
    statuses[np.maximum(0, difference-error) > tolerance] = "mismatch"
    status = "mismatch" if np.any(statuses == "mismatch") else ("passed" if np.all(statuses == "passed") else "inconclusive")
    return {"status": status, "analytic": analytic.tolist(), "difference": difference.tolist(), "tolerance": tolerance.tolist(), "channels": statuses.tolist(), "reason": None if status == "passed" else "analytic_comparison"}

def verify_multiscale_direction(feature, blocks, block_index, axis, manifold, interpolation):
    """Sample all fixed scales safely and read the analytic derivative only after window selection."""
    patch = feature.patch
    nodes = np.asarray(interpolation.nodes, dtype=np.float64)
    pixel, reason = get_candidate_projection(blocks)
    if reason is not None or not check_feature_domain(patch, pixel):
        return {"status": "inconclusive", "reason": "invalid_center_projection", "identity": list(feature.identity), "comparison": None}
    center_local = patch.get_pixel_coords(pixel)[None]+nodes
    center_cells = np.floor(center_local+np.asarray(patch.corner)).astype(np.int64)
    center = feature.native.evaluate(*blocks)[0]
    if center is None or not np.isfinite(center).all():
        return {"status": "inconclusive", "reason": "nonfinite_center_descriptor", "identity": list(feature.identity), "comparison": None}
    levels = []
    for k in range(7):
        step = (1e-6 if block_index == 0 else 1e-4)/(2**k)
        pair = []
        for sign in [-1, 1]:
            candidate = [block.copy() for block in blocks]
            if block_index == 0:
                candidate[0] = manifold.plus(blocks[0], sign*axis*step)
            else:
                candidate[block_index] += sign*axis*step
            pair.append(candidate)
        displacement = check_actual_perturbation(blocks[block_index], pair[0][block_index], pair[1][block_index], block_index, axis, step)
        row = {"step": step, "valid": displacement["valid"], "reason": displacement["reason"], "same_segment": False, "displacement": displacement, "parameters_minus": [block.tolist() for block in pair[0]], "parameters_plus": [block.tolist() for block in pair[1]], "values_minus": None, "values_plus": None, "cells_minus": None, "cells_plus": None}
        if row["valid"]:
            samples, cells = [], []
            for candidate in pair:
                projected, invalid = get_candidate_projection(candidate)
                if invalid is not None:
                    row.update(valid=False, reason=invalid)
                    break
                local = patch.get_pixel_coords(projected)[None]+nodes
                supported = np.isfinite(local).all() and np.all(local >= 1) and np.all(local[:, 0] < patch.width-2) and np.all(local[:, 1] < patch.height-2)
                if not supported:
                    row.update(valid=False, reason="outside_bicubic_support")
                    break
                values = feature.native.evaluate(*candidate)[0]
                if values is None or not np.isfinite(values).all():
                    row.update(valid=False, reason="nonfinite_descriptor")
                    break
                samples.append(values.reshape(-1))
                cells.append(np.floor(local+np.asarray(patch.corner)).astype(np.int64))
            if row["valid"]:
                row.update(values_minus=samples[0].tolist(), values_plus=samples[1].tolist(), cells_minus=cells[0].tolist(), cells_plus=cells[1].tolist(), same_segment=bool(np.array_equal(cells[0], center_cells) and np.array_equal(cells[1], center_cells)))
        levels.append(row)
    numerical = select_multiscale_reference(center.reshape(-1), levels)
    result = {"identity": list(feature.identity), "block_index": block_index, "axis": axis.tolist(), "center_parameters": [block.tolist() for block in blocks], "center_cells": center_cells.tolist(), "numerical": numerical, "comparison": None}
    if numerical["status"] != "ready":
        result.update(status="inconclusive", reason=numerical["reason"])
        return result
    jacobians = feature.native.evaluate(*blocks)[1]
    derivative = jacobians[0]@manifold.plus_jacobian(blocks[0]) if block_index == 0 else jacobians[block_index]
    comparison = classify_multiscale_derivative(numerical, derivative@axis)
    result.update(status=comparison["status"], reason=comparison["reason"], comparison=comparison)
    return result

def audit_native_equivalence(model, anchors, scales, context, patch_path):
    """Compare frozen-state native costs, all common residuals and prescribed tangent perturbations."""
    state = build_shared_problem(model, anchors, scales, 1., context=context)
    audit_factor_identity(state)
    native = build_native_problem(copy.deepcopy(model), anchors, scales, 1., context=context)
    independent = compute_problem_cost(state)
    native["solver"].solve_problem(native["model"])
    summary = native["solver"].summary()
    require_numeric_match(sum(independent.values()), summary.initial_cost, 1e-7, 1e-9, "native total objective")
    old_manager = features.FeatureManager_f16(str(patch_path), False, "")
    old_view = features.FeatureView(old_manager.fset(0), model)
    common, maxima = [], {"geometric": 0., "feature": 0., "jacobian": 0., "perturbation": 0., "directional_derivative": 0.}
    image_costs = {(cost.identity, cost.patch is not None): cost for cost in state["costs"] if isinstance(cost, GuardImageCost)}
    for point_id, image_id, feature_id, measured in state["observations"]:
        identity = (point_id, image_id, feature_id)
        image = model.images[image_id]
        camera = model.cameras[image.camera_id]
        blocks = [image.qvec, image.tvec, model.points3D[point_id].xyz, camera.params]
        pixel, reason = get_candidate_projection(blocks)
        geometric = image_costs[(identity, False)]
        values, jacobians = geometric.evaluate(*blocks)
        expected, expected_jacobians = geometric.native.evaluate(*blocks)
        maxima["geometric"] = max(maxima["geometric"], require_numeric_match(values, expected, 1e-9, 1e-10, "geometric residual"))
        feature = image_costs[(identity, True)]
        values, jacobians = feature.evaluate(*blocks)
        sampled = context["interpolator"].interpolate(feature.patch, pixel).reshape(-1)-feature.reference
        maxima["feature"] = max(maxima["feature"], require_numeric_match(values.reshape(-1), sampled, 1e-10, 1e-8, "full feature residual"))
        old_patch = old_view.fpatch(image_id, feature_id)
        if not check_feature_domain(old_patch, pixel):
            continue
        old_native = native_residuals.FeatureReferenceCostFunctor(camera.model_id, old_patch, feature.reference.reshape(1, 128), context["interpolation"])
        previous, previous_jacobians = old_native.evaluate(*blocks)
        maxima["feature"] = max(maxima["feature"], require_numeric_match(values.reshape(-1), previous.reshape(-1)-feature.reference, 1e-10, 1e-8, "old/full common feature residual"))
        for first, second in zip(jacobians, previous_jacobians):
            maxima["jacobian"] = max(maxima["jacobian"], require_numeric_match(first, second, 1e-8, 1e-6, "old/full ambient Jacobian"))
        common.append((identity, feature, old_patch, old_native, [block.copy() for block in blocks]))
    perturb_counts, perturb_samples, direction_reports = {}, [], []
    manifolds = dict(zip(model.images, state["manifolds"]))
    for image_id in model.images:
        eligible = []
        for identity, feature, patch, old_native, blocks in sorted(common, key=lambda row: row[0]):
            if identity[1] != image_id:
                continue
            trials = []
            valid = True
            for block_index in [0, 1, 2]:
                for axis in np.eye(3):
                    pair = []
                    for sign in [-1, 1]:
                        trial = [block.copy() for block in blocks]
                        if block_index == 0:
                            trial[0] = manifolds[image_id].plus(blocks[0], sign*axis*1e-6)
                        else:
                            trial[block_index] += sign*axis*1e-4
                        pixel, reason = get_candidate_projection(trial)
                        valid = valid and reason is None and check_feature_domain(patch, pixel) and check_feature_domain(feature.patch, pixel)
                        pair.append(trial)
                    trials.append((block_index, axis, pair))
            if valid:
                eligible.append((identity, feature, old_native, blocks, trials))
        if not eligible:
            raise ValueError("No common-domain perturbation observations in an image")
        selected = np.unique(np.linspace(0, len(eligible)-1, min(32, len(eligible)), dtype=int))
        perturb_counts[str(image_id)] = [list(eligible[index][0]) for index in selected]
        perturb_samples.extend(eligible[index] for index in selected)
    audit = {"passed": True, "all_training_observations": len(state["observations"]), "common_old_patch_observations": len(common), "perturbation_observations": perturb_counts, "maximum_absolute_differences": maxima, "independent_costs": independent, "native_initial_cost": float(summary.initial_cost), "factor_identity": audit_factor_identity(state), "manifold": "same owned native QuaternionManifold instances used in Problem", "reference_policy": "both implementations use the same newly frozen references", "derivative_policy": get_derivative_policy(), "planned_direction_count": len(perturb_samples)*9, "checked_direction_count": 0, "direction_results": direction_reports}
    for identity, feature, old_native, blocks, trials in perturb_samples:
        base = feature.evaluate(*blocks)[0]
        old_base = old_native.evaluate(*blocks)[0]-feature.reference.reshape(-1, 1)
        for block_index, axis, pair in trials:
            for trial in pair:
                current = feature.evaluate(*trial)[0]
                previous = old_native.evaluate(*trial)[0]-feature.reference.reshape(-1, 1)
                maxima["perturbation"] = max(maxima["perturbation"], require_numeric_match(current-base, previous-old_base, 2e-10, 1e-6, "perturbed residual change"))
            result = verify_multiscale_direction(feature, blocks, block_index, axis, manifolds[identity[1]], context["interpolation"])
            direction_reports.append(result)
            audit["checked_direction_count"] = len(direction_reports)
            if result["comparison"] is not None:
                maxima["directional_derivative"] = max(maxima["directional_derivative"], max(result["comparison"].get("difference", [0.])))
            if result["status"] != "passed":
                audit.update(passed=False, reason=result["status"], failure_identity=list(identity), failure_block_index=block_index, failure_axis=axis.tolist())
                return audit
    return audit

def write_map_priors(model, anchors, scales, path):
    """Write one finite covariance and complete observation lineage for each accepted track."""
    ids = np.asarray(anchors["point_ids"], dtype=np.int64)
    image_ids, feature_ids, prior_indices = [], [], []
    for index, point_id in enumerate(ids):
        for entry in model.points3D[int(point_id)].track.elements:
            image_ids.append(entry.image_id)
            feature_ids.append(entry.point2D_idx)
            prior_indices.append(index)
    covariance = np.diag([scales["sigma_horizontal_m"] ** 2] * 2 + [scales["sigma_height_m"] ** 2])
    np.savez_compressed(path, source_link_indices=np.asarray(anchors["selected_link_indices"], dtype=np.int64), source_prior_ids=np.asarray(anchors["source_prior_ids"], dtype=np.int64), shared_point_ids=ids, track_ids=ids, prior_ids=np.arange(len(ids), dtype=np.int64), map_feature_ids=anchors["map_feature_ids"].astype(np.int64), means_enu_m=anchors["means"].astype(np.float64), covariances_m2=np.repeat(covariance[None], len(ids), axis=0), dom_pixels=anchors["dom_pixels"].astype(np.float64), observation_image_ids=np.asarray(image_ids, dtype=np.int64), observation_point2D_ids=np.asarray(feature_ids, dtype=np.int64), observation_prior_indices=np.asarray(prior_indices, dtype=np.int64))

def write_validation_json(path, data):
    """Write finite checkpoint JSON atomically without exception suppression."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)

def restore_feature_domain(model, anchors, scales, information_scale, context, label):
    """Attempt only the three declared feasibility stages while keeping every training factor."""
    records = []
    for beta in [1., 10., 100.]:
        state = build_shared_problem(model, anchors, scales, information_scale, restore_context=context, beta=beta)
        state["iteration_recorder"].label = f"{label} restore beta={beta}"
        solved = solve_shared_problem(state)
        domain = check_model_domain(model, state["observations"], context)
        records.append({"beta": beta, "solver": solved, "domain": domain, "positions_sha256": get_array_digest(np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]]))})
        if solved["usable"] and not domain["invalid_count"]:
            return {"restored": True, "stages": records}
        if not solved["usable"]:
            return {"restored": False, "stages": records}
    return {"restored": False, "stages": records}

def read_frozen_candidate_inputs(run_dir):
    """Verify every immutable input before constructing any candidate state."""
    metadata = json.loads((run_dir/"run.json").read_text())
    for filename, expected in metadata["frozen_files"].items():
        if get_sha256(filename) != expected:
            raise ValueError(f"Frozen candidate input changed: {filename}")
    return metadata

def run_candidate_worker(run_dir, information_scale):
    """Solve one isolated fixed candidate and persist normal failures without try-except."""
    run_dir = Path(run_dir)
    metadata = read_frozen_candidate_inputs(run_dir)
    label = {0.1: "weight_0p1", 1.: "weight_1p0", 10.: "weight_10p0"}[information_scale]
    directory = run_dir/label
    directory.mkdir(exist_ok=True)
    result_path = directory/"result.json"
    result = {"candidate_id": label, "run_id": metadata["run_id"], "policy_sha256": metadata["policy_sha256"], "run_inputs_sha256": get_sha256(run_dir/"run.json"), "information_scale": information_scale, "status": "running", "failure_stage": None, "geometry": {"executed": False, "status": "not_run"}, "feature": {"executed": False, "status": "not_run"}, "coverage": {"sample_count": metadata["heldout_count"], "valid_count": None, "invalid_count": None, "evaluated": False}, "heldout_reprojection_px": None, "heldout_feature_norm": None, "heldout_feature_normalized": None}
    write_validation_json(result_path, result)
    model = pycolmap.Reconstruction(str(run_dir/"training_visual_model"))
    with np.load(run_dir/"map_priors.npz") as priors:
        anchors = {"point_ids": priors["shared_point_ids"], "means": priors["means_enu_m"]}
        covariances = priors["covariances_m2"]
    scales = metadata["scales"]
    result["initialization"] = initialize_map_points(model, anchors)
    result["geometry_initial_domain"] = check_model_domain(model, get_observations(model))
    initial_positions = np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]])
    result["gauge_checkpoints"] = [compute_gauge_information(anchors, covariances, initial_positions, information_scale, "map_initial")]
    require_numeric_match(result["gauge_checkpoints"][0]["matrix"], information_scale*np.asarray(metadata["base_gauge"]["matrix"]), 1e-9, 1e-10, "initial gauge weight scaling")
    if result["geometry_initial_domain"]["invalid_count"]:
        result.update(status="failed", failure_stage="geometry_initial_domain")
        write_validation_json(result_path, result)
        return
    geometry = build_shared_problem(model, anchors, scales, information_scale)
    audit_factor_identity(geometry)
    geometry["iteration_recorder"].label = f"{label} geometry"
    result["geometry"] = {"executed": True, "status": "running"}
    write_validation_json(result_path, result)
    solved = solve_shared_problem(geometry)
    result["geometry"] = dict(solved, executed=True, status="passed" if solved["converged"] else "failed")
    if solved["usable"]:
        (directory/"geometry_model").mkdir(exist_ok=True)
        model.write(str(directory/"geometry_model"))
    write_validation_json(result_path, result)
    if not solved["converged"]:
        result.update(status="failed", failure_stage="geometry_convergence")
        write_validation_json(result_path, result)
        return
    positions = np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]])
    result["gauge_checkpoints"].append(compute_gauge_information(anchors, covariances, positions, information_scale, "after_geometry"))
    del geometry
    context = create_dense_feature_context(model, metadata["dense_path"], compute_references=False)
    context["references"] = load_training_references(run_dir/"training_references.npz")
    result["feature_start_domain"] = check_model_domain(model, get_observations(model), context)
    write_validation_json(result_path, result)
    if result["feature_start_domain"]["invalid_count"]:
        result["restoration"] = restore_feature_domain(model, anchors, scales, information_scale, context, label)
        write_validation_json(result_path, result)
        if not result["restoration"]["restored"]:
            result.update(status="failed", failure_stage="feature_feasibility_restoration")
            write_validation_json(result_path, result)
            return
        positions = np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]])
        result["gauge_checkpoints"].append(compute_gauge_information(anchors, covariances, positions, information_scale, "after_restoration"))
    feature = build_shared_problem(model, anchors, scales, information_scale, context=context)
    result["factor_identity"] = audit_factor_identity(feature)
    if feature["audit"]["restore_blocks"] != 0:
        raise ValueError("Temporary feasibility factors leaked into final objective")
    result["feature"] = {"executed": True, "status": "running"}
    write_validation_json(result_path, result)
    feature["iteration_recorder"].label = f"{label} feature"
    solved = solve_shared_problem(feature)
    result["feature"] = dict(solved, executed=True, status="passed" if solved["converged"] else "failed")
    write_validation_json(result_path, result)
    if not solved["usable"]:
        result.update(status="failed", failure_stage="feature_solver")
        write_validation_json(result_path, result)
        return
    result["final_training_domain"] = check_model_domain(model, get_observations(model), context)
    with np.load(run_dir/"heldout.npz") as archive:
        heldout = [(int(identity[0]), int(identity[1]), int(identity[2]), pixel.copy()) for identity, pixel in zip(archive["identities"], archive["pixels"])]
    result["coverage"] = dict(check_model_domain(model, heldout, context), evaluated=True)
    if not result["coverage"]["invalid_count"]:
        descriptors, valid = sample_feature_residuals(model, heldout, context)
        if not valid.all():
            raise ValueError("Final coverage disagrees with independent descriptor sampling")
        pixels = np.linalg.norm(project_observations(model, heldout)-np.array([row[3] for row in heldout]), axis=1)
        norms = np.linalg.norm(descriptors, axis=1)
        result["heldout_reprojection_px"] = get_error_statistics(pixels)
        result["heldout_feature_norm"] = get_error_statistics(norms)
        result["heldout_feature_normalized"] = get_error_statistics(norms/scales["sigma_feature_norm"])
    positions = np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]])
    gauge = compute_gauge_information(anchors, covariances, positions, information_scale, "final_joint")
    result["gauge_checkpoints"].append(gauge)
    result["map_residual_m"] = get_error_statistics(np.linalg.norm(positions-anchors["means"], axis=1))
    result["xyz_displacement_m"] = get_error_statistics(np.linalg.norm(positions-initial_positions, axis=1))
    result["final_factor_identity"] = audit_factor_identity(feature)
    result["trial_evaluations"] = {"calls": sum(cost.calls for cost in feature["costs"] if isinstance(cost, GuardImageCost)), "feature_samples": sum(cost.samples for cost in feature["costs"] if isinstance(cost, GuardImageCost)), "rejected_before_sampling": sum(cost.rejected for cost in feature["costs"] if isinstance(cost, GuardImageCost))}
    valid = solved["converged"] and not result["final_training_domain"]["invalid_count"] and not result["coverage"]["invalid_count"] and gauge["passed"]
    result.update(status="passed" if valid else "failed", failure_stage=None if valid else "final_convergence_domain_or_gauge")
    (directory/"joint_model").mkdir(exist_ok=True)
    model.write(str(directory/"joint_model"))
    result["final_model_fingerprint"] = get_model_fingerprint(model)
    read_frozen_candidate_inputs(run_dir)
    write_validation_json(result_path, result)
    print(f"{label}: {result['status']}, heldout={result['coverage']['valid_count']}/{len(heldout)}, geometry={result['geometry']['termination_type']}, feature={solved['termination_type']}", flush=True)

def execute_candidate_loop(report, run_dir, root, report_path, write_json):
    """Run all normal candidate failures but stop globally on worker or identity errors."""
    worker_code = "import sys; from pathlib import Path; from joint_map.optimization import run_candidate_worker; run_candidate_worker(Path(sys.argv[1]), float(sys.argv[2]))"
    for index, candidate in enumerate(report["candidates"]):
        candidate.update(status="running")
        write_json(report_path, report)
        directory = run_dir/candidate["candidate_id"]
        directory.mkdir(exist_ok=True)
        with (directory/"worker.log").open("w") as stream:
            process = subprocess.run([sys.executable, "-B", "-c", worker_code, str(run_dir), str(candidate["information_scale"])], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode != 0 or not (directory/"result.json").is_file():
            candidate.update(status="aborted", exit_code=process.returncode, checkpoint_path=str(directory/"result.json"))
            report.update(status="failed", stop_scope="global", error="Candidate worker exited unexpectedly; this is an execution failure")
            for remaining in report["candidates"][index+1:]:
                remaining.update(status="not_run", reason="global execution stop", heldout_reprojection_px=None, heldout_feature_norm=None)
            break
        result = json.loads((directory/"result.json").read_text())
        if result["run_id"] != report["run_id"] or result["policy_sha256"] != report["policy_sha256"] or result["candidate_id"] != candidate["candidate_id"] or result["run_inputs_sha256"] != get_sha256(run_dir/"run.json") or result["information_scale"] != candidate["information_scale"] or result["status"] not in ["passed", "failed"]:
            candidate.update(status="aborted", reason="stale, mismatched or incomplete worker result")
            report.update(status="failed", stop_scope="global", error="Worker result identity validation failed")
            for remaining in report["candidates"][index+1:]:
                remaining.update(status="not_run", reason="global worker identity stop", heldout_reprojection_px=None, heldout_feature_norm=None)
            break
        report["candidates"][index] = result
        write_json(report_path, report)
        print(f"Candidate {candidate['information_scale']}: {result['status']}, coverage={result['coverage']}", flush=True)

def run_prior_validation(config, root, output_dir, write_json):
    """Contain global native or preparation failures without suppressing exceptions in workers."""
    code = "import json,sys; from pathlib import Path; from run_joint_map import write_json; from joint_map.optimization import run_prior_validation_in_process; run_prior_validation_in_process(json.loads(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),write_json)"
    process = subprocess.run([sys.executable, "-B", "-c", code, json.dumps(config), str(root), str(output_dir)], cwd=root)
    report_path = output_dir/"shared/prior_validation.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"schema_version": 3, "stage": "validate_priors", "candidates": []}
    if process.returncode != 0:
        report.update(status="failed", prior_status="unvalidated", stop_scope="global", exit_code=process.returncode, error="Validation preparation or implementation exited unexpectedly; see process traceback", proposed_weight=None, candidate_id=None)
        for candidate in report["candidates"]:
            if candidate["status"] in ["pending", "running"]:
                candidate.update(status="aborted" if candidate["status"] == "running" else "not_run", reason="global validation execution stop", heldout_reprojection_px=None, heldout_feature_norm=None)
        write_json(report_path, report)
        if "run_dir" in report:
            write_json(Path(report["run_dir"])/"validation_snapshot.json", report)
        write_json(output_dir/"logs/validate_priors.log", report)
        write_json(output_dir/"shared/prior_model.json", {"schema_version": 3, "status": "unvalidated", "lambda_map": None, "candidate_id": None})
    return report

def run_prior_validation_in_process(config, root, output_dir, write_json):
    """Freeze training evidence, validate the guarded bridge and compare isolated fixed candidates."""
    shared = output_dir/"shared"
    if json.loads((output_dir/"logs/initialize.log").read_text())["status"] != "complete":
        raise ValueError("Initialization must be complete before prior validation")
    if config["prior_validation"]["candidate_information_scales"] != [0.1, 1., 10.]:
        raise ValueError("Candidate weights differ from the frozen development plan")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = shared/"candidates"/run_id
    run_dir.mkdir(parents=True)
    report_path = shared/"prior_validation.json"
    for name in ["prior_validation.json", "prior_model.json", "map_priors.npz", "training_references.npz"]:
        if (shared/name).exists():
            shutil.copy2(shared/name, run_dir/("previous_"+name))
    model = pycolmap.Reconstruction(str(output_dir/"joint/model"))
    training, heldout = create_holdout(model, config["prior_validation"]["holdout_fraction"], config["prior_validation"]["holdout_seed"])
    report = {"schema_version": 3, "stage": "validate_priors", "run_id": run_id, "run_dir": str(run_dir), "status": "running", "error": None, "registered_images": model.num_reg_images(), "sparse_points": model.num_points3D(), "initialization_uses_heldout": True, "interpretation": "conditional holdout stability and sensitivity; not independent generalization or absolute accuracy", "gt_inputs_read": False, "training_observation_count": len(get_observations(training)), "heldout_observation_count": len(heldout), "training_observations_sha256": get_observation_digest(get_observations(training)), "holdout_observations_sha256": get_observation_digest(heldout), "heldout_observations": [list(row[:3]) for row in heldout], "visual_reference_model_fingerprint": get_model_fingerprint(training), "prior_status": "unvalidated", "candidates": [{"candidate_id": label, "information_scale": weight, "status": "pending"} for label, weight in [("weight_0p1", 0.1), ("weight_1p0", 1.), ("weight_10p0", 10.)]]}
    write_json(report_path, report)
    write_json(shared/"prior_model.json", {"schema_version": 3, "status": "unvalidated", "lambda_map": None, "candidate_id": None})
    print(f"Validation {run_id}: training={report['training_observation_count']}, heldout={len(heldout)}", flush=True)
    available_memory = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    report["resources"] = {"memory_available_bytes": available_memory, "disk_free_bytes": shutil.disk_usage(shared).free, "device": "CPU Ceres and immutable float16 cache; no network inference"}
    if available_memory < 8*1024**3 or report["resources"]["disk_free_bytes"] < 512*1024**2:
        raise RuntimeError("Insufficient resources for approved full-resolution validation")
    if len(heldout) != 777 or report["training_observation_count"] != 18013:
        raise ValueError("The fixed 777/18013 observation split changed")
    dense_path = shared/"holdout_dense_features.h5"
    if not dense_path.exists():
        raise FileNotFoundError("Approved dense cache is missing; automatic extraction is forbidden")
    patch_path = (root/config["native_sfm_dir"]/"../dense_features.h5").resolve()
    report["holdout_dense_cache"] = {"path": str(dense_path), "sha256": get_sha256(dense_path), "provenance": get_dense_provenance(config, root)}
    report["holdout_dense_cache"]["images"] = check_dense_cache(dense_path, report["holdout_dense_cache"]["provenance"])
    report["cache_consistency"] = compare_dense_and_patch_cache(dense_path, patch_path, InterpolationConfig(interpolation_default_conf))
    report["cache_usage"] = "same immutable full-image storage for training and evaluation; heldout observations excluded from all fitting"
    write_json(report_path, report)
    anchors = create_training_links(config, root, output_dir, training, heldout)
    if len(anchors["point_ids"]) != 942:
        raise ValueError("The frozen training map collection differs from the planned 942 anchors")
    order = np.argsort(anchors["point_ids"])
    for key in ["point_ids", "means", "dom_pixels", "map_feature_ids", "selected_link_indices", "source_prior_ids"]:
        anchors[key] = anchors[key][order]
    context = create_dense_feature_context(training, dense_path)
    reference_path = shared/"training_references.npz"
    report["reference_provenance"] = save_training_references(training, heldout, context, reference_path, root)
    report["reference_policy"] = report["reference_provenance"]["reference_policy"]
    report["isolation_checks"] = {"training_residual_overlap": len({row[:3] for row in get_observations(training)} & {row[:3] for row in heldout}), "reference_actual_participant_overlap": report["reference_provenance"]["actual_participant_overlap_with_holdout"], "excluded_map_candidate_votes": anchors["excluded_map_votes"], "map_selection": "heldout image/feature votes excluded before homography and independent 2m geometry revalidation"}
    if report["isolation_checks"]["training_residual_overlap"] or report["isolation_checks"]["reference_actual_participant_overlap"]:
        raise ValueError("Heldout leakage detected")
    scales = estimate_working_scales(config, root, training, anchors, context, json.loads((shared/"origin.json").read_text()))
    report["scales"] = scales
    report["scales_frozen_on"] = "training visual geometry before map assignment"
    write_map_priors(training, anchors, scales, shared/"map_priors.npz")
    report["map_prior_scope"] = "training_frozen"
    report["accepted_track_count"] = len(anchors["point_ids"])
    report["map_selection_models"] = anchors["independent_models"]
    covariance = np.diag([scales["sigma_horizontal_m"]**2]*2+[scales["sigma_height_m"]**2])
    report["base_gauge"] = compute_gauge_information(anchors, covariance, anchors["means"], 1., "frozen_map_base")
    if not report["base_gauge"]["passed"]:
        raise ValueError("Frozen map set cannot constrain seven similarity freedoms")
    report["synthetic"] = run_synthetic_controls()
    bridge_path = root/"build/pyceres_module3/bridge_validation.json"
    report["bridge_validation"] = json.loads(bridge_path.read_text())
    loaded_sha = get_sha256(pyceres.__file__)
    if loaded_sha != report["bridge_validation"]["new_extension"]["sha256"]:
        raise ValueError("Loaded pyceres does not match tested extension")
    report["bridge_validation"].update(loaded_path=pyceres.__file__, loaded_sha256=loaded_sha)
    write_json(report_path, report)
    report["policy"] = {"plan_sha256": get_sha256(root/"PROJECT_PLAN.md"), "code_sha256": get_sha256(__file__), "extension_sha256": loaded_sha, "weights": [0.1, 1., 10.], "geometry_max_iterations": 300, "feature_max_iterations": 100, "tolerances": {"function": 1e-6, "gradient": 1e-10, "parameter": 1e-8}, "recovery": {"beta": [1., 10., 100.], "max_iterations": 100, "feature_margin": 0.25}, "comparison": {"metrics": ["P", "F", "SP", "SF"], "absolute_deltas": [0.01, 0.01, 0.02, 0.02], "relative_delta": 0.01}, "reference_policy": report["reference_policy"], "derivative_verification": get_derivative_policy()}
    policy_sha = hashlib.sha256(json.dumps(report["policy"], sort_keys=True).encode()).hexdigest()
    report["policy_sha256"] = policy_sha
    for name in ["map_priors.npz", "training_references.npz"]:
        shutil.copy2(shared/name, run_dir/name)
    write_json(report_path, report)
    report["native_equivalence"] = audit_native_equivalence(training, anchors, scales, context, patch_path)
    if not report["native_equivalence"]["passed"]:
        report.update(status="failed", stop_scope="global", error="Multiscale derivative verification "+report["native_equivalence"]["reason"], prior_status="unvalidated", selection_status="not_run", proposed_weight=None, candidate_id=None)
        for candidate in report["candidates"]:
            candidate.update(status="not_run", reason=report["error"], heldout_reprojection_px=None, heldout_feature_norm=None)
        write_json(report_path, report)
        write_json(run_dir/"validation_snapshot.json", report)
        write_json(output_dir/"logs/validate_priors.log", report)
        return report
    if get_model_fingerprint(training) != report["visual_reference_model_fingerprint"]:
        raise ValueError("Reference extraction or native controls changed the frozen visual geometry")
    initialized = copy.deepcopy(training)
    report["map_initialization"] = initialize_map_points(initialized, anchors)
    report["geometry_initial_domain"] = check_model_domain(initialized, get_observations(initialized))
    report["feature_initial_domain_diagnostic"] = check_model_domain(initialized, get_observations(initialized), context)
    report["input_fingerprints"] = json.loads((output_dir/"logs/check.log").read_text())["input_fingerprints"]
    (run_dir/"training_visual_model").mkdir()
    training.write(str(run_dir/"training_visual_model"))
    for name in ["map_priors.npz", "training_references.npz"]:
        shutil.copy2(shared/name, run_dir/name)
    np.savez_compressed(run_dir/"heldout.npz", identities=np.array([row[:3] for row in heldout], dtype=np.int64), pixels=np.array([row[3] for row in heldout]))
    frozen_paths = [run_dir/name for name in ["map_priors.npz", "training_references.npz", "heldout.npz"]]+[run_dir/"training_visual_model"/name for name in ["cameras.bin", "images.bin", "points3D.bin"]]+[Path(__file__), Path(pyceres.__file__), dense_path, patch_path, shared/"track_map_links.npz", shared/"map_priors.npz", root/"PROJECT_PLAN.md"]
    metadata = {"run_id": run_id, "policy_sha256": policy_sha, "policy": report["policy"], "scales": scales, "base_gauge": report["base_gauge"], "dense_path": str(dense_path), "heldout_count": len(heldout), "frozen_files": {str(path.resolve()): get_sha256(path) for path in frozen_paths}}
    write_json(run_dir/"run.json", metadata)
    write_json(run_dir/"validation_snapshot.json", report)
    write_json(report_path, report)
    del context, initialized
    execute_candidate_loop(report, run_dir, root, report_path, write_json)
    read_frozen_candidate_inputs(run_dir)
    if report["status"] != "failed":
        report.update(status="complete", **select_weight_candidates(report["candidates"]))
        report["prior_status"] = "validated" if any(row["status"] == "passed" for row in report["candidates"]) else "unvalidated"
    else:
        report.update(selection_status="aborted", proposed_weight=None, candidate_id=None)
    report["error"] = report.get("error")
    write_json(report_path, report)
    write_json(run_dir/"validation_snapshot.json", report)
    write_json(output_dir/"logs/validate_priors.log", report)
    prior_model = {key: scales[key] for key in ["sigma_horizontal_m", "sigma_height_m", "sigma_pixel_px", "sigma_feature_norm", "lambda_feature", "robust_loss"]}
    prior_model.update(schema_version=3, status=report["prior_status"], lambda_map=report["proposed_weight"], candidate_id=report["candidate_id"], map_prior_scope="training_frozen", map_priors_sha256=get_sha256(shared/"map_priors.npz"), reference_sha256=get_sha256(reference_path), validation_report=str(report_path), validation_sha256=get_sha256(report_path), policy_sha256=policy_sha, selection_basis=report["selection_status"], limitations=scales["limitations"])
    write_json(shared/"prior_model.json", prior_model)
    return report

def require_confirmed_prior(model):
    """Require explicit confirmation of a concrete finite positive validated candidate."""
    if model["status"] != "confirmed":
        raise ValueError("Prior model awaits explicit user confirmation; register has not run")
    weight = model.get("lambda_map")
    if not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0 or not model.get("candidate_id") or not model.get("user_confirmation") or model.get("map_prior_scope") != "training_frozen":
        raise ValueError("Confirmed prior lacks a concrete validated candidate or user confirmation")

def confirm_prior_candidate(output_dir, candidate_id, user_confirmation):
    """Record an explicit user choice without rewriting the frozen validation report."""
    shared = Path(output_dir)/"shared"
    model = json.loads((shared/"prior_model.json").read_text())
    report = json.loads((shared/"prior_validation.json").read_text())
    if model["status"] != "validated" or report["status"] != "complete" or not user_confirmation:
        raise ValueError("Confirmation requires completed validation and explicit user text")
    for field, filename in [("validation_sha256", "prior_validation.json"), ("map_priors_sha256", "map_priors.npz"), ("reference_sha256", "training_references.npz")]:
        if get_sha256(shared/filename) != model[field]:
            raise ValueError("Validation evidence changed before confirmation")
    candidate = next((row for row in report["candidates"] if row["candidate_id"] == candidate_id and row["status"] == "passed"), None)
    if candidate is None:
        raise ValueError("User must choose an actually passed candidate")
    model.update(status="confirmed", candidate_id=candidate_id, lambda_map=candidate["information_scale"], selection_basis="user_selected", user_confirmation=user_confirmation)
    require_confirmed_prior(model)
    write_validation_json(shared/"prior_model.json", model)
    return model

def run_confirmed_registration(config, root, output_dir, write_json):
    """Fit the shared robust map binding with confirmed scales while preserving native relative geometry."""
    shared = output_dir / "shared"
    prior_model = json.loads((shared / "prior_model.json").read_text())
    require_confirmed_prior(prior_model)
    for field, filename in [("validation_sha256", "prior_validation.json"), ("map_priors_sha256", "map_priors.npz"), ("reference_sha256", "training_references.npz")]:
        if hashlib.sha256((shared / filename).read_bytes()).hexdigest() != prior_model[field]:
            raise ValueError("Confirmed validation inputs changed")
    validation = json.loads((shared/"prior_validation.json").read_text())
    selected = [row for row in validation["candidates"] if row["candidate_id"] == prior_model["candidate_id"] and row["status"] == "passed" and row["information_scale"] == prior_model["lambda_map"]]
    if len(selected) != 1 or validation["status"] != "complete" or validation["policy_sha256"] != prior_model["policy_sha256"]:
        raise ValueError("Confirmed candidate does not match the actual frozen validation")
    read_frozen_candidate_inputs(Path(validation["run_dir"]))
    with np.load(shared / "map_priors.npz") as archive:
        ids, means, covariances = archive["shared_point_ids"], archive["means_enu_m"], archive["covariances_m2"]
    native = pycolmap.Reconstruction(str(root / config["native_sfm_dir"]))
    points = np.array([native.points3D[int(point_id)].xyz for point_id in ids])
    initial = json.loads((shared / "registration.json").read_text())
    parameters = np.r_[np.log(initial["scale"]), Rotation.from_matrix(initial["rotation"]).as_rotvec(), initial["translation_enu_m"]]
    def compute_registration_residuals(parameters):
        """Encode the confirmed radial Huber map objective for finite-difference least squares."""
        transform = (np.exp(parameters[0]), Rotation.from_rotvec(parameters[1:4]).as_matrix(), parameters[4:7])
        residuals = (apply_sim3(points, transform) - means) / np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
        squared = np.sum(residuals ** 2, axis=1)
        factors = np.sqrt(np.divide(compute_huber_cost(squared, 3), squared, out=np.ones_like(squared), where=squared > 0))
        return (residuals * factors[:, None] * np.sqrt(prior_model["lambda_map"])).ravel()
    result = least_squares(compute_registration_residuals, parameters, max_nfev=200)
    if not result.success or np.linalg.matrix_rank(result.jac) != 7:
        raise ValueError("Confirmed map registration did not converge with seven observable freedoms")
    transform = (np.exp(result.x[0]), Rotation.from_rotvec(result.x[1:4]).as_matrix(), result.x[4:7])
    joint_initialization = None
    for branch in ["baseline", "joint"]:
        model = copy.deepcopy(native)
        write_initialized_model(model, transform, output_dir / branch / "model")
        if branch == "joint":
            lineage = initialize_map_points(model, {"point_ids": ids, "means": means})
            domain = check_model_domain(model, get_observations(model))
            if domain["invalid_count"]:
                raise ValueError("Confirmed map initialization violates geometric domain")
            model.write(str(output_dir/branch/"model"))
            joint_initialization = {"lineage": lineage, "geometry_initial_domain": domain, "map_priors_sha256": prior_model["map_priors_sha256"]}
    registration = {"schema_version": 3, "status": "confirmed", "scale": transform[0], "rotation": transform[1].tolist(), "translation_enu_m": transform[2].tolist(), "inlier_point3D_ids": ids.tolist(), "priors_confirmed": True, "prior_model": prior_model, "robust_cost": float(result.cost), "joint_map_initialization": joint_initialization}
    write_json(shared / "registration.json", registration)
    report = {"schema_version": 3, "stage": "register", "status": "complete", "registered_images": native.num_reg_images(), "sparse_points": native.num_points3D(), "accepted_track_count": len(ids), "baseline_relative_geometry_preserved": True, "joint_map_points_initialized": True, "map_prior_scope": "training_frozen", "gt_inputs_read": False, "error": None}
    write_json(output_dir / "logs/register.log", report)
    return report
