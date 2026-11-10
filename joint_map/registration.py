# 用途：建立同特征索引的 DOM 关联并独立检验地图 Sim(3) 初始化。
import itertools
import json
import cv2
import h5py
import numpy as np
import pycolmap
import rasterio
import torch
from hloc.extractors.superpoint import SuperPoint
from hloc.matchers.superglue import SuperGlue
from hloc.utils.parsers import names_to_pair
from joint_map.geometry import convert_geodetic_to_enu, convert_pixels_to_geodetic, sample_dsm, transform_colmap_pose

def transform_pixels(pixels, matrix):
    """Apply a homogeneous pixel-center transform without adding a half pixel."""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))]) @ np.asarray(matrix).T
    return homogeneous[:, :2] / homogeneous[:, 2:]

def create_dom_variant(gray, valid, angle_deg, scale):
    """Rotate and scale DOM centers with expanded borders and conservative masks."""
    height, width = gray.shape
    matrix = np.eye(3)
    matrix[:2] = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle_deg, scale)
    corners = transform_pixels([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], matrix)
    lower = np.floor(corners.min(axis=0) + 1e-10)
    upper = np.ceil(corners.max(axis=0) - 1e-10)
    matrix[:2, 2] -= lower
    size = tuple((upper - lower + 1).astype(int))
    image = cv2.warpAffine(gray, matrix[:2], size, flags=cv2.INTER_LINEAR)
    coverage = cv2.warpAffine(valid.astype(np.float32), matrix[:2], size, flags=cv2.INTER_LINEAR)
    mask = coverage >= 1 - 1e-6
    image[~mask] = 0
    return image, mask, matrix

def check_feature_identity(model, feature_path, adjusted_path):
    """Verify all feature rows, float32 KA coordinates and reciprocal track entries."""
    registered = 0
    observations = {image_id: image.points2D for image_id, image in model.images.items()}
    with h5py.File(feature_path, "r") as features, h5py.File(adjusted_path, "r") as adjusted:
        for image_id, image in model.images.items():
            group = features[image.name]
            count = len(group["keypoints"])
            if len(observations[image_id]) != count or group["descriptors"].shape != (256, count) or group["scores"].shape != (count,):
                raise ValueError(f"Feature row identity mismatch: {image.name}")
            descriptors = group["descriptors"][:].astype(np.float32)
            if not np.isfinite(descriptors).all() or not np.allclose(np.linalg.norm(descriptors, axis=0), 1, atol=0.002):
                raise ValueError(f"Invalid original descriptors: {image.name}")
            expected = (adjusted[image.name]["keypoints"][:] + 0.5).astype(np.float32)
            actual = np.array([point.xy for point in observations[image_id]])
            if not np.array_equal(actual, expected):
                raise ValueError(f"KA/COLMAP coordinate identity mismatch: {image.name}")
            for feature_id, point in enumerate(observations[image_id]):
                if not point.has_point3D():
                    continue
                entries = model.points3D[point.point3D_id].track.elements
                if not any(entry.image_id == image_id and entry.point2D_idx == feature_id for entry in entries):
                    raise ValueError(f"Missing reciprocal track entry: {image.name}, {feature_id}")
                registered += 1
    for point_id, point in model.points3D.items():
        for entry in point.track.elements:
            if observations[entry.image_id][entry.point2D_idx].point3D_id != point_id:
                raise ValueError(f"Track points to another XYZ: {point_id}")
    return {"registered_observations": registered, "max_coordinate_identity_error_px": 0.0}

def check_camera_connectivity(model, anchor_ids):
    """Require all cameras to connect to map anchors through complete shared tracks."""
    reached = {entry.image_id for point_id in anchor_ids for entry in model.points3D[int(point_id)].track.elements}
    changed = True
    while changed:
        previous = len(reached)
        for point in model.points3D.values():
            images = {entry.image_id for entry in point.track.elements}
            if images & reached:
                reached.update(images)
        changed = len(reached) != previous
    return sorted(reached), reached == set(model.images)

def check_track_geometry(model, observations, point_id, max_error_px, min_angle_deg):
    """Check positive depths, full-track reprojection and multiview triangulation angle."""
    point = model.points3D[int(point_id)]
    centers, errors = [], []
    for entry in point.track.elements:
        image = model.images[entry.image_id]
        camera = model.cameras[image.camera_id]
        xyz_camera = image.rotmat() @ point.xyz + image.tvec
        if xyz_camera[2] <= 0:
            return False
        fx, fy, cx, cy = camera.params
        predicted = xyz_camera[:2] / xyz_camera[2] * [fx, fy] + [cx, cy]
        errors.append(np.linalg.norm(predicted - observations[entry.image_id][entry.point2D_idx].xy))
        centers.append(-image.rotmat().T @ image.tvec)
    if len(centers) < 2 or max(errors) > max_error_px:
        return False
    rays = point.xyz - np.asarray(centers)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    angle = np.rad2deg(np.arccos(np.clip(rays @ rays.T, -1, 1))).max()
    return bool(angle >= min_angle_deg)

def fit_sim3(source, target):
    """Fit a positive-scale proper Sim(3), returning None for degenerate anchors."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        return None
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source - source_center, target - target_center
    if np.linalg.matrix_rank(source_zero, tol=1e-8) < 2 or np.linalg.matrix_rank(target_zero, tol=1e-8) < 2:
        return None
    left, singular, right = np.linalg.svd(target_zero.T @ source_zero / len(source))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(left @ right)
    rotation = (left * signs) @ right
    scale = np.sum(singular * signs) / np.mean(np.sum(source_zero ** 2, axis=1))
    if not np.isfinite(scale) or scale <= 0:
        return None
    return scale, rotation, target_center - scale * rotation @ source_center

def apply_sim3(points, transform):
    """Transform XYZ rows by scale, rotation and translation."""
    scale, rotation, translation = transform
    return scale * np.asarray(points) @ rotation.T + translation

def estimate_sim3(source, target, threshold_m, seed=0):
    """Estimate deterministic robust Sim(3) with a fixed 2000-hypothesis budget."""
    source, target = np.asarray(source), np.asarray(target)
    count = len(source)
    if count < 3:
        return None, np.zeros(count, dtype=bool)
    samples = itertools.combinations(range(count), 3) if count <= 23 else (np.random.default_rng(seed + index).choice(count, 3, replace=False) for index in range(2000))
    best, best_mask, best_key = None, np.zeros(count, dtype=bool), (0, -np.inf)
    for indices in samples:
        indices = np.asarray(indices)
        transform = fit_sim3(source[indices], target[indices])
        if transform is None:
            continue
        residuals = np.linalg.norm(apply_sim3(source, transform) - target, axis=1)
        mask = residuals <= threshold_m
        if mask.sum() < 3:
            continue
        key = (int(mask.sum()), -float(np.median(residuals[mask])))
        if key > best_key:
            best, best_mask, best_key = transform, mask, key
    if best_mask.sum() < 3:
        return None, best_mask
    for iteration in range(5):
        refined = fit_sim3(source[best_mask], target[best_mask])
        if refined is None:
            return None, best_mask
        mask = np.linalg.norm(apply_sim3(source, refined) - target, axis=1) <= threshold_m
        best = refined
        if np.array_equal(mask, best_mask):
            break
        best_mask = mask
        if mask.sum() < 3:
            return None, mask
    return best, best_mask

def screen_homography(image_pixels, dom_pixels, threshold_px):
    """Measure only forward image-to-original-DOM errors for the planar coarse screen."""
    errors = np.full(len(image_pixels), np.inf)
    if len(image_pixels) < 4:
        return errors, None
    matrix, mask = cv2.findHomography(np.asarray(image_pixels, dtype=np.float64), np.asarray(dom_pixels, dtype=np.float64), cv2.RANSAC, threshold_px)
    if matrix is not None:
        predicted = transform_pixels(image_pixels, matrix)
        errors = np.linalg.norm(predicted - dom_pixels, axis=1)
    return errors, matrix

def select_map_candidates(links, source_xyz, threshold_m):
    """Cross-fit by track so validation transforms exclude every candidate of tested tracks."""
    count = len(links["track_ids"])
    accepted = np.zeros(count, dtype=bool)
    codes = np.array(links["rejection_codes"], dtype="U64")
    residuals = np.full(count, np.nan)
    reference_counts = np.zeros(count, dtype=np.int64)
    eligible = codes == "pending_3d"
    track_ids = np.unique(links["track_ids"][eligible])
    references = {}
    for track_id in track_ids:
        indices = np.flatnonzero(eligible & (links["track_ids"] == track_id) & links["homography_pass"])
        if not len(indices):
            continue
        positions = links["means_enu_m"][indices]
        if np.max(np.linalg.norm(positions[:, None] - positions[None, :], axis=2)) > threshold_m:
            continue
        references[int(track_id)] = indices[np.argmax(links["scores"][indices])]
    selected = []
    folds = {int(track_id): index % min(5, len(track_ids)) for index, track_id in enumerate(track_ids)}
    fits, validation_models = {}, []
    for track_id in track_ids:
        indices = np.flatnonzero(eligible & (links["track_ids"] == track_id))
        fold = folds[int(track_id)]
        reference_indices = np.array([index for other_id, index in references.items() if folds[other_id] != fold], dtype=int)
        reference_counts[indices] = len(reference_indices)
        if fold not in fits:
            fits[fold] = estimate_sim3(source_xyz[reference_indices], links["means_enu_m"][reference_indices], threshold_m)
            fitted, support = fits[fold]
            validation_models.append({"fold": fold, "excluded_track_ids": [track for track, group in folds.items() if group == fold], "reference_track_ids": links["track_ids"][reference_indices].tolist(), "inlier_track_ids": links["track_ids"][reference_indices[support]].tolist(), "transform": None if fitted is None else {"scale": fitted[0], "rotation": fitted[1].tolist(), "translation_enu_m": fitted[2].tolist()}})
        transform, inliers = fits[fold]
        if transform is None:
            codes[indices] = "pending_independent_anchors"
            continue
        # A candidate's whole track is absent from fitting, consensus and refinement.
        residuals[indices] = np.linalg.norm(apply_sim3(source_xyz[indices], transform) - links["means_enu_m"][indices], axis=1)
        codes[indices] = "independent_3d_outlier"
        passed = indices[residuals[indices] <= threshold_m]
        if not len(passed):
            continue
        positions = links["means_enu_m"][passed]
        if np.max(np.linalg.norm(positions[:, None] - positions[None, :], axis=2)) > threshold_m:
            codes[passed] = "pending_inconsistent_track"
            continue
        accepted[passed] = True
        codes[passed] = "accepted_independent_3d"
        # Multiple variants of one image never increase support or prior weight.
        selected.append(passed[np.argmax(links["scores"][passed])])
    return accepted, codes, residuals, reference_counts, np.asarray(selected, dtype=int), validation_models

def extract_dom_features(config, root, shared_dir):
    """Extract SuperPoint on the fixed DOM variants while preserving masks and resize maps."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; use the already verified host GPU access without changing the environment")
    extractor = SuperPoint({"nms_radius": 3, "max_keypoints": 4096}).eval().cuda()
    with rasterio.open(root / config["dom_path"]) as dataset:
        bands = dataset.read(masked=True)
        valid = ~np.ma.getmaskarray(bands).any(axis=0)
        rgb = bands.filled(0).astype(np.float32).transpose(1, 2, 0)
        lower, upper = np.percentile(rgb[valid], [1, 99])
        if upper <= lower:
            raise ValueError("DOM has no usable texture range")
        rgb = np.clip((rgb - lower) * 255 / (upper - lower), 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray[~valid] = 0
    variants = []
    with h5py.File(shared_dir / "dom_features.h5", "w") as features:
        for angle in config["map_association"]["dom_rotations_deg"]:
            for scale in config["map_association"]["dom_scales"]:
                name = f"dom_r{angle:03d}_s{scale}"
                image, mask, matrix = create_dom_variant(gray, valid, angle, scale)
                height, width = image.shape
                resize_scale = min(1., 1024 / max(height, width))
                size = (round(width * resize_scale), round(height * resize_scale))
                resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                resized_mask = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
                with torch.no_grad():
                    prediction = extractor({"image": torch.from_numpy(resized.astype(np.float32) / 255)[None, None].cuda()})
                keypoints = prediction["keypoints"][0].cpu().numpy()
                support = cv2.remap(resized_mask, keypoints[:, 0].astype(np.float32)[:, None], keypoints[:, 1].astype(np.float32)[:, None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).ravel() >= 1 - 1e-6
                ratios = np.array([width / size[0], height / size[1]])
                keypoints = (keypoints + 0.5) * ratios - 0.5
                group = features.create_group(name)
                group.create_dataset("keypoints", data=keypoints[support].astype(np.float32))
                group.create_dataset("descriptors", data=prediction["descriptors"][0].cpu().numpy()[:, support])
                group.create_dataset("scores", data=prediction["scores"][0].cpu().numpy()[support])
                group.create_dataset("image_size", data=[width, height])
                group.create_dataset("valid_mask", data=mask.astype(np.uint8), compression="gzip")
                variants.append({"id": len(variants), "name": name, "angle_deg": angle, "scale": scale, "original_to_variant": matrix.tolist(), "variant_to_original": np.linalg.inv(matrix).tolist(), "network_to_variant": [[ratios[0], 0, (ratios[0] - 1) / 2], [0, ratios[1], (ratios[1] - 1) / 2], [0, 0, 1]], "image_size": [width, height], "network_size": list(size), "valid_mask": f"dom_features.h5:{name}/valid_mask", "feature_count": int(support.sum())})
                print(f"DOM features {len(variants)}/36: {name}, points={support.sum()}", flush=True)
    return {"schema_version": 3, "normalization_percentiles": [1, 99], "normalization_values": [float(lower), float(upper)], "variants": variants}

def match_dom_features(config, root, shared_dir, variants):
    """Match unmodified original image feature rows against DOM features with SuperGlue."""
    matcher = SuperGlue({"weights": "outdoor", "sinkhorn_iterations": 50, "match_threshold": config["map_association"]["min_match_score"]}).eval().cuda()
    with h5py.File(root / config["frontend"]["features_path"], "r") as images, h5py.File(shared_dir / "dom_features.h5", "r") as maps, h5py.File(shared_dir / "image_dom_matches.h5", "w") as matches:
        for name in config["image_names"]:
            for variant in variants["variants"]:
                reference = variant["name"]
                data = {}
                for suffix, group in [("0", images[name]), ("1", maps[reference])]:
                    for key in ["keypoints", "descriptors", "scores"]:
                        data[key + suffix] = torch.from_numpy(group[key][:].astype(np.float32))[None].cuda()
                    width, height = group["image_size"][:]
                    data["image" + suffix] = torch.empty((1, 1, int(height), int(width)), device="cuda")
                with torch.no_grad():
                    prediction = matcher(data)
                group = matches.create_group(names_to_pair(name, reference))
                group.create_dataset("matches0", data=prediction["matches0"][0].cpu().numpy().astype(np.int32))
                group.create_dataset("matching_scores0", data=prediction["matching_scores0"][0].cpu().numpy())
            print(f"DOM matching complete: {name} ({len(variants['variants'])} variants)", flush=True)

def collect_map_candidates(config, root, shared_dir, variants, model, origin):
    """Record descriptor candidates, coordinate backtraces and exact feature-to-track links."""
    records, homographies = [], []
    name_to_id = {image.name: image_id for image_id, image in model.images.items()}
    observations = {image_id: image.points2D for image_id, image in model.images.items()}
    reliability = {point_id: check_track_geometry(model, observations, point_id, config["update"]["max_reprojection_error_px"], config["update"]["min_triangulation_angle_deg"]) for point_id in model.points3D}
    with rasterio.open(root / config["dsm_path"]) as dsm, h5py.File(root / config["frontend"]["features_path"], "r") as images, h5py.File(shared_dir / "dom_features.h5", "r") as maps, h5py.File(shared_dir / "image_dom_matches.h5", "r") as matches:
        heights = dsm.read(1, masked=True)
        for name in config["image_names"]:
            image_id = name_to_id[name]
            image = model.images[image_id]
            for variant in variants["variants"]:
                group = matches[names_to_pair(name, variant["name"])]
                indices = group["matches0"][:]
                scores = group["matching_scores0"][:]
                feature_ids = np.flatnonzero((indices >= 0) & (scores >= config["map_association"]["min_match_score"]))
                dom_feature_ids = indices[feature_ids]
                dom_pixels = transform_pixels(maps[variant["name"]]["keypoints"][:][dom_feature_ids], variant["variant_to_original"])
                errors, matrix = screen_homography(images[name]["keypoints"][:][feature_ids], dom_pixels, config["map_association"]["homography_forward_threshold_dom_px"])
                homographies.append({"image_name": name, "variant_id": variant["id"], "matrix_image_to_original_dom": None if matrix is None else matrix.tolist(), "matches": len(feature_ids), "coarse_inliers": int(np.sum(errors <= 3))})
                positions = convert_pixels_to_geodetic(dom_pixels, dsm.transform)
                sampled, valid = sample_dsm(heights, dsm.transform, positions)
                xyz = convert_geodetic_to_enu(np.column_stack([positions, sampled]), origin)
                for index, feature_id in enumerate(feature_ids):
                    point2d = observations[image_id][int(feature_id)]
                    track_id = int(point2d.point3D_id) if point2d.has_point3D() else -1
                    code = "pending_3d"
                    if not valid[index]:
                        code = "invalid_dsm_support"
                    elif track_id < 0:
                        code = "pending_untriangulated_feature"
                    elif not reliability[track_id]:
                        code = "unreliable_image_track"
                    records.append((image_id, feature_id, track_id, variant["id"], dom_feature_ids[index], dom_pixels[index], scores[feature_id], errors[index], xyz[index], code))
    fields = ["image_ids", "feature_ids", "track_ids", "dom_variant_ids", "dom_feature_ids", "dom_pixels", "scores", "homography_errors_dom_px", "means_enu_m", "rejection_codes"]
    links = {}
    for column, field in enumerate(fields):
        dtype = "U64" if field == "rejection_codes" else np.int64 if column < 5 else np.float64
        links[field] = np.asarray([record[column] for record in records], dtype=dtype)
    links["dom_pixels"] = links["dom_pixels"].reshape(-1, 2)
    links["means_enu_m"] = links["means_enu_m"].reshape(-1, 3)
    links["homography_pass"] = links["homography_errors_dom_px"] <= config["map_association"]["homography_forward_threshold_dom_px"]
    links["shared_point_ids"] = links["track_ids"].copy()
    links["point2D_indices"] = links["feature_ids"].copy()
    source_xyz = np.array([model.points3D[int(track_id)].xyz if track_id >= 0 else np.full(3, np.nan) for track_id in links["track_ids"]]).reshape(-1, 3)
    return links, source_xyz, homographies

def write_initialized_model(model, transform, path):
    """Transform every camera and shared XYZ while preserving IDs, intrinsics and full tracks."""
    for image in model.images.values():
        rotation, translation = transform_colmap_pose(image.rotmat(), image.tvec, *transform)
        image.qvec = pycolmap.rotmat_to_qvec(rotation)
        image.tvec = translation
    for point in model.points3D.values():
        point.xyz = apply_sim3(point.xyz, transform)
    path.mkdir(parents=True, exist_ok=True)
    model.write(str(path))

def run_initialization(config, root, output_dir, write_json):
    """Build DOM links and provisional ENU models, returning explicit failure evidence."""
    shared_dir = output_dir / "shared"
    origin = json.loads((shared_dir / "origin.json").read_text())
    model = pycolmap.Reconstruction(str(root / config["native_sfm_dir"]))
    identity = check_feature_identity(model, root / config["frontend"]["features_path"], root / config["frontend"]["adjusted_features_path"])
    variants = extract_dom_features(config, root, shared_dir)
    write_json(shared_dir / "dom_variants.json", variants)
    match_dom_features(config, root, shared_dir, variants)
    links, source_xyz, homographies = collect_map_candidates(config, root, shared_dir, variants, model, origin)
    threshold = config["map_association"]["sim3_inlier_threshold_m"]
    accepted, codes, residuals, reference_counts, selected, validation_models = select_map_candidates(links, source_xyz, threshold)
    links.update(accepted=accepted, rejection_codes=codes, independent_errors_m=residuals, independent_reference_counts=reference_counts, selected_for_prior=np.zeros(len(accepted), dtype=bool), prior_ids=np.full(len(accepted), -1, dtype=np.int64))
    links["selected_for_prior"][selected] = True
    for prior_id, index in enumerate(selected):
        links["prior_ids"][accepted & (links["track_ids"] == links["track_ids"][index])] = prior_id
    np.savez_compressed(shared_dir / "track_map_links.npz", **links)
    reached, connected = check_camera_connectivity(model, links["track_ids"][selected])
    transform, inliers = estimate_sim3(source_xyz[selected], links["means_enu_m"][selected], threshold)
    report = {"schema_version": 3, "stage": "initialize", "status": "failed", "identity": identity, "candidate_count": len(accepted), "accepted_candidate_count": int(accepted.sum()), "accepted_track_count": len(selected), "single_image_dom_track_count": int(sum(len(np.unique(links["image_ids"][accepted & (links["track_ids"] == links["track_ids"][index])])) == 1 for index in selected)), "rejection_counts": {str(code): int(np.sum(codes == code)) for code in np.unique(codes)}, "homographies": homographies, "connected_image_ids": reached, "all_cameras_connected": connected, "homography_threshold_dom_px": 3., "sim3_threshold_m": threshold, "independent_validation": "track_grouped_5_fold_crossfit", "initialization_uses_heldout": True, "gt_inputs_read": False, "prior_status": "unvalidated", "error": None}
    report["independent_models"] = validation_models
    if len(selected) < config["map_association"]["min_tracks"] or transform is None or inliers.sum() < config["map_association"]["min_tracks"] or not connected:
        report["error"] = "Insufficient independently validated nondegenerate map anchors or disconnected cameras; no threshold relaxation or GT fallback"
        write_json(output_dir / "logs/initialize.log", report)
        return report
    if not inliers.all():
        report["error"] = "Independently accepted anchors are inconsistent under the common final Sim(3); pending review"
        write_json(output_dir / "logs/initialize.log", report)
        return report
    scale, rotation, translation = transform
    registration = {"schema_version": 3, "status": "provisional", "scale": scale, "rotation": rotation.tolist(), "translation_enu_m": translation.tolist(), "inlier_point3D_ids": links["track_ids"][selected].tolist(), "independent_validation": "track_grouped_5_fold_crossfit", "priors_confirmed": False}
    write_json(shared_dir / "registration.json", registration)
    for branch in ["baseline", "joint"]:
        branch_model = pycolmap.Reconstruction(str(root / config["native_sfm_dir"]))
        write_initialized_model(branch_model, transform, output_dir / branch / "model")
    report.update(status="complete", registered_images=model.num_reg_images(), sparse_points=model.num_points3D(), scale=scale, anchor_errors_m=np.linalg.norm(apply_sim3(source_xyz[selected], transform) - links["means_enu_m"][selected], axis=1).tolist())
    write_json(shared_dir / "prior_model.json", {"schema_version": 3, "status": "unvalidated", "sigma_horizontal_m": None, "sigma_height_m": None, "sigma_pixel_px": None, "lambda_feature": None, "lambda_map": None, "robust_loss": None, "validation_report": None, "selection_basis": "Pending module three; no production covariance or weight assigned"})
    write_json(output_dir / "logs/initialize.log", report)
    return report
