# 用途：按多视图与单像元分层证据筛选稠密点，生成可追溯的局部 DSM 更新。
import json
from itertools import combinations
from pathlib import Path
import shutil
import numpy as np
import pycolmap
import rasterio
from pyproj import Geod
from joint_map.dense import read_depth_map, read_fused_cloud, read_image_order, read_visibility
from joint_map.geometry import convert_enu_to_geodetic
from joint_map.optimization import get_sha256

def get_grid_cells(positions, transform, shape):
    """Map finite geographic positions to containing raster cells without clipping outside coordinates."""
    positions = np.asarray(positions)
    columns, rows = ~transform @ (positions[:, 0], positions[:, 1])
    valid = np.isfinite(columns) & np.isfinite(rows) & (columns >= 0) & (rows >= 0) & (columns < shape[1]) & (rows < shape[0])
    cells = np.full((len(positions), 2), -1, dtype=np.int64)
    cells[valid] = np.floor(np.column_stack([rows, columns])[valid]).astype(np.int64)
    return cells, valid

def get_projection(positions, view):
    """Project metric world positions with intrinsics scaled to the actual depth resolution."""
    local = positions@view["rotation"].T+view["translation"]
    positive = np.isfinite(local).all(axis=1) & (local[:, 2] > 0)
    pixels = np.full((len(local), 2), np.nan)
    pixels[positive] = local[positive, :2]/local[positive, 2:]*view["params"][:2]+view["params"][2:]
    height, width = view["depth"].shape
    inside = positive & np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return local, pixels, inside

def sample_view(positions, view, relative_limit):
    """Sample nearest depth pixels and retain invalid and occluded predictions as explicit states."""
    local, pixels, inside = get_projection(positions, view)
    coordinates = np.full((len(local), 2), -1, dtype=np.int64)
    coordinates[inside] = np.floor(pixels[inside]+0.5).astype(np.int64)
    height, width = view["depth"].shape
    inside &= (coordinates[:, 0] >= 0) & (coordinates[:, 1] >= 0) & (coordinates[:, 0] < width) & (coordinates[:, 1] < height)
    depth = np.full(len(local), np.nan)
    normals = np.full((len(local), 3), np.nan)
    depth[inside] = view["depth"][coordinates[inside, 1], coordinates[inside, 0]]
    normals[inside] = view["normals"][coordinates[inside, 1], coordinates[inside, 0]]@view["rotation"]
    valid = inside & np.isfinite(depth) & (depth > 0)
    residual = np.full(len(local), np.nan)
    relative = np.full(len(local), np.nan)
    residual[valid] = local[valid, 2]-depth[valid]
    relative[valid] = residual[valid]/local[valid, 2]
    occluded = valid & (relative > relative_limit)
    recovered = np.full((len(local), 3), np.nan)
    rays = (coordinates[valid]-view["params"][2:])/view["params"][:2]
    recovered[valid] = (np.column_stack([rays, np.ones(len(rays))])*depth[valid, None]-view["translation"])@view["rotation"]
    return {"pixels": pixels, "sample_pixels": coordinates, "inside": inside, "valid_depth": valid, "occluded": occluded, "visible": valid & ~occluded, "residual_m": residual, "relative_residual": relative, "predicted_depth_m": local[:, 2], "recovered_xyz": recovered, "world_normals": normals}

def compute_vector_angles(first, second):
    """Compute signed-direction normal or ray angles while leaving invalid vectors undefined."""
    denominator = np.linalg.norm(first, axis=1)*np.linalg.norm(second, axis=1)
    valid = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1) & (denominator > 0)
    angles = np.full(len(first), np.nan)
    cosine = np.sum(first[valid]*second[valid], axis=1)/denominator[valid]
    angles[valid] = np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))
    return angles

def check_point_reliability(positions, normals, native_support, views, limits):
    """Check native support, occlusion, depth, normals, pairwise reprojection and triangulation angles."""
    count, view_count = native_support.shape
    visible = np.zeros((count, view_count), dtype=bool)
    depth_support = np.zeros_like(visible)
    consistent = np.zeros_like(visible)
    depth_error = np.full((count, view_count), np.nan)
    normal_error = np.full_like(depth_error, np.nan)
    samples = []
    for index, view in enumerate(views):
        sample = sample_view(positions, view, limits["max_relative_depth_error"])
        samples.append(sample)
        visible[:, index] = native_support[:, index] & sample["visible"]
        depth_error[:, index] = np.abs(sample["relative_residual"])
        depth_support[:, index] = visible[:, index] & (depth_error[:, index] <= limits["max_relative_depth_error"])
        normal_error[:, index] = compute_vector_angles(normals, sample["world_normals"])
        consistent[:, index] = depth_support[:, index] & (normal_error[:, index] <= limits["max_normal_angle_deg"])
    reprojection = np.full(count, -np.inf)
    pair_normal = np.full(count, -np.inf)
    parallax = np.full(count, -np.inf)
    has_pair = np.zeros(count, dtype=bool)
    for first, second in combinations(range(view_count), 2):
        selected = consistent[:, first] & consistent[:, second]
        if not selected.any():
            continue
        has_pair |= selected
        pair_errors = []
        for source, target in [(first, second), (second, first)]:
            local, pixels, valid = get_projection(samples[source]["recovered_xyz"][selected], views[target])
            error = np.linalg.norm(pixels-samples[target]["sample_pixels"][selected], axis=1)
            pair_errors.append(np.where(valid, error, np.inf))
        reprojection[selected] = np.maximum(reprojection[selected], np.maximum(*pair_errors))
        angles = compute_vector_angles(samples[first]["world_normals"][selected], samples[second]["world_normals"][selected])
        pair_normal[selected] = np.maximum(pair_normal[selected], angles)
        rays = [positions[selected]-views[index]["center"] for index in [first, second]]
        parallax[selected] = np.maximum(parallax[selected], compute_vector_angles(*rays))
    for values in [reprojection, pair_normal, parallax]:
        values[~has_pair] = np.nan
    supported = consistent.sum(axis=1) >= limits["min_support_views"]
    reprojected = supported & (reprojection <= limits["max_reprojection_error_px"])
    aligned = reprojected & (pair_normal <= limits["max_normal_angle_deg"])
    reliable = aligned & (parallax >= limits["min_triangulation_angle_deg"])
    counts = {"input": count, "visible": int((visible.sum(axis=1) >= 1).sum()), "overlap": int((visible.sum(axis=1) >= 2).sum()), "depth_support": int((depth_support.sum(axis=1) >= limits["min_support_views"]).sum()), "normal_support": int(supported.sum()), "reprojection": int(reprojected.sum()), "pair_normal": int(aligned.sum()), "parallax_reliable": int(reliable.sum())}
    return {"native_support": native_support, "visible_support": visible, "consistent_support": consistent, "relative_depth_error": depth_error, "normal_error_deg": normal_error, "max_reprojection_error_px": reprojection, "max_pair_normal_angle_deg": pair_normal, "max_triangulation_angle_deg": parallax, "reliable": reliable, "counts": counts}

def check_height_groups(heights, min_points, max_mad):
    """Apply fixed one-meter gap groups and priority before the unscaled single-layer MAD check."""
    heights = np.sort(np.asarray(heights, dtype=np.float64))
    if not np.isfinite(heights).all():
        raise ValueError("Cell heights must all be finite")
    if not len(heights):
        return "insufficient_reliable_points", None, None
    groups = np.split(heights, np.flatnonzero(np.diff(heights) >= 1.0)+1)
    if sum(len(group) >= 3 for group in groups) >= 2:
        return "multiple_height_layers", None, None
    if len(groups) > 1:
        return "ambiguous_height_support", None, None
    median = float(np.median(heights))
    mad = float(np.median(np.abs(heights-median)))
    reason = "insufficient_reliable_points" if len(heights) < min_points else "height_mad_exceeded" if mad > max_mad else "accepted"
    return reason, median, mad

def build_dsm_update(original, cells, heights, evidence, conflict_mask, limits):
    """Build four evidence masks and replace only supported single-layer raster cells."""
    shape = original.shape
    coverage, overlap, reliability = [np.zeros(shape, dtype=bool) for _ in range(3)]
    inside = (cells[:, 0] >= 0) & (cells[:, 1] >= 0) & (cells[:, 0] < shape[0]) & (cells[:, 1] < shape[1])
    for mask, selected in [(coverage, evidence["visible_support"].sum(axis=1) >= 1), (overlap, evidence["visible_support"].sum(axis=1) >= 2)]:
        positions = cells[inside & selected]
        mask[positions[:, 0], positions[:, 1]] = True
    selected = np.flatnonzero(inside & evidence["reliable"])
    indices = cells[selected, 0]*shape[1]+cells[selected, 1]
    order = np.argsort(indices, kind="stable")
    groups = np.split(selected[order], np.flatnonzero(np.diff(indices[order]))+1) if len(selected) else []
    reasons = ["not_evaluated", "insufficient_reliable_points", "multiple_height_layers", "ambiguous_height_support", "height_mad_exceeded", "accepted"]
    codes = np.zeros(shape, dtype=np.uint8)
    medians = np.full(shape, np.nan)
    mads = np.full(shape, np.nan)
    point_counts = np.zeros(shape, dtype=np.int64)
    for group in groups:
        row, column = cells[group[0]]
        reason, median, mad = check_height_groups(heights[group], limits["min_points_per_cell"], limits["max_height_mad_m"])
        codes[row, column] = reasons.index(reason)
        point_counts[row, column] = len(group)
        if median is not None:
            medians[row, column], mads[row, column] = median, mad
        reliability[row, column] = reason == "accepted"
    reliability &= ~conflict_mask
    update = coverage & overlap & reliability
    updated = original.copy()
    updated[update] = medians[update].astype(original.dtype)
    return {"updated": updated, "coverage_mask": coverage, "overlap_mask": overlap, "reliability_mask": reliability, "update_mask": update, "height_reason_codes": codes, "height_reason_names": np.array(reasons), "height_medians": medians, "height_mads": mads, "reliable_point_counts": point_counts, "sparse_conflict_mask": conflict_mask}

def load_update_views(output_dir, branch, dense_report):
    """Load verified branch cameras with metric depths and camera-frame normal maps."""
    model = pycolmap.Reconstruction(str(output_dir/branch/"model"))
    workspace = Path(dense_report["workspace_path"])
    order = json.loads(Path(dense_report["image_order_path"]).read_text())
    if order["images"] != read_image_order(workspace/"sparse") or get_sha256(workspace/"sparse/images.bin") != order["images_bin_sha256"]:
        raise ValueError("Native MVS image order changed after dense preparation")
    hashes, views = {}, []
    for row in order["images"]:
        image = model.images[row["image_id"]]
        if image.name != row["name"] or image.camera_id != row["camera_id"]:
            raise ValueError("Dense-to-optimized image identity changed")
        paths = [workspace/f"stereo/{folder}/{image.name}.geometric.bin" for folder in ["depth_maps", "normal_maps"]]
        for path in paths:
            hashes[str(path)] = get_sha256(path)
        depth, normals = [read_depth_map(path) for path in paths]
        camera = model.cameras[image.camera_id]
        params = camera.params.copy()*np.tile([depth.shape[1]/camera.width, depth.shape[0]/camera.height], 2)
        rotation, translation = image.rotmat(), image.tvec.copy()
        if depth.shape[2] != 1 or normals.shape != (*depth.shape[:2], 3):
            raise ValueError("Depth and normal dimensions do not match")
        views.append(dict(row, rotation=rotation, translation=translation, center=-rotation.T@translation, params=params, depth=depth[:, :, 0].astype(np.float64)*dense_report["depth_scale_to_m"], normals=normals))
    return model, views, hashes

def check_sparse_dense(model, base, views, origin, transform, shape, limits):
    """Audit actual sparse track projections and mark conflicts only in their optimized raster cells."""
    result = {key: value.copy() for key, value in base.items()}
    point_ids = result["shared_point_ids"]
    positions = np.array([model.points3D[int(index)].xyz for index in point_ids])
    np.testing.assert_array_equal(positions, result["optimized_xyz_enu_m"])
    result["old_cells"], old_inside = get_grid_cells(convert_enu_to_geodetic(result["map_xyz_enu_m"], origin), transform, shape)
    result["optimized_cells"], inside = get_grid_cells(convert_enu_to_geodetic(positions, origin), transform, shape)
    image_ids, feature_ids, point_indices = [result[key] for key in ["observation_image_ids", "observation_point2D_ids", "observation_prior_indices"]]
    expected = {(index, entry.image_id, entry.point2D_idx) for index, point_id in enumerate(point_ids) for entry in model.points3D[int(point_id)].track.elements}
    actual = set(zip(point_indices.tolist(), image_ids.tolist(), feature_ids.tolist()))
    if actual != expected or len(actual) != len(image_ids):
        raise ValueError("Sparse evidence participants differ from the actual complete tracks")
    count = len(image_ids)
    residuals, relatives, reprojections, bands, distances = [np.full(count, np.nan) for _ in range(5)]
    recovered = np.full((count, 3), np.nan)
    valid, consistent, conflict, occluded = [np.zeros(count, dtype=bool) for _ in range(4)]
    supported = np.zeros((len(point_ids), len(views)), dtype=bool)
    conflict_views = np.zeros_like(supported)
    for view_index, view in enumerate(views):
        rows = np.flatnonzero(image_ids == view["image_id"])
        points = positions[point_indices[rows]]
        sample = sample_view(points, view, limits["max_relative_depth_error"])
        image = model.images[view["image_id"]]
        camera = model.cameras[image.camera_id]
        height, width = view["depth"].shape
        image_points = image.points2D
        measured = np.array([image_points[int(index)].xy for index in feature_ids[rows]])*[width/camera.width, height/camera.height]
        reprojections[rows] = np.linalg.norm(sample["pixels"]-measured, axis=1)
        valid[rows] = sample["visible"] & (reprojections[rows] <= limits["max_reprojection_error_px"])
        occluded[rows] = sample["occluded"]
        residuals[rows] = sample["residual_m"]
        relatives[rows] = sample["relative_residual"]
        bands[rows] = sample["predicted_depth_m"]*limits["max_relative_depth_error"]
        recovered[rows] = sample["recovered_xyz"]
        distances[rows] = np.linalg.norm(sample["recovered_xyz"]-points, axis=1)
        consistent[rows] = valid[rows] & (np.abs(relatives[rows]) <= limits["max_relative_depth_error"])
        conflict[rows] = valid[rows] & (np.abs(relatives[rows]) > limits["max_relative_depth_error"])
        supported[point_indices[rows[consistent[rows]]], view_index] = True
        conflict_views[point_indices[rows[conflict[rows]]], view_index] = True
    conflicts = conflict_views.sum(axis=1) >= 2
    good = supported.sum(axis=1) >= 2
    conflict_mask = np.zeros(shape, dtype=bool)
    cells = result["optimized_cells"][conflicts & inside]
    conflict_mask[cells[:, 0], cells[:, 1]] = True
    states = np.full(len(point_ids), "insufficient_evidence", dtype="U40")
    states[good] = "consistent"
    states[conflicts] = "conflict"
    result.update(dense_agreement_residuals=residuals, dense_agreement_relative_residuals=relatives, dense_agreement_limit_m=bands, dense_recovered_xyz_enu_m=recovered, dense_position_residuals_m=distances, dense_reprojection_errors_px=reprojections, dense_valid_observations=valid, dense_consistent_observations=consistent, dense_conflict_observations=conflict, dense_conflict_views=conflict_views, dense_occluded_observations=occluded, dense_consistency_states=states, dense_supported_views=supported, dense_support_status=np.array("checked"), cell_mapping_status=np.array("mapped_without_clipping"), raster_shape=np.array(shape))
    return result, conflict_mask

def link_dense_support(lineage, cells, evidence, update_mask):
    """Link reliable same-cell dense IDs and common views to actual output cells without inventing support."""
    good = np.flatnonzero(evidence["reliable"] & (cells[:, 0] >= 0))
    width = update_mask.shape[1]
    linear = cells[good, 0]*width+cells[good, 1]
    order = np.argsort(linear, kind="stable")
    sorted_cells, sorted_ids = linear[order], good[order]
    dense_indices, update_indices = [], []
    dense_offsets, update_offsets = [0], [0]
    for index, cell in enumerate(lineage["optimized_cells"]):
        linked = np.empty(0, dtype=np.int64)
        if cell[0] >= 0 and lineage["dense_consistency_states"][index] == "consistent":
            cell_index = cell[0]*width+cell[1]
            lower, upper = np.searchsorted(sorted_cells, [cell_index, cell_index+1])
            candidates = sorted_ids[lower:upper]
            shared = (evidence["consistent_support"][candidates] & lineage["dense_supported_views"][index]).sum(axis=1) >= 2
            linked = candidates[shared]
            dense_indices.extend(linked)
            if len(linked) and update_mask[tuple(cell)]:
                update_indices.append(cell_index)
        dense_offsets.append(len(dense_indices))
        update_offsets.append(len(update_indices))
    lineage.update(dense_support_indices=np.array(dense_indices, dtype=np.int64), dense_support_offsets=np.array(dense_offsets, dtype=np.int64), update_cell_indices=np.array(update_indices, dtype=np.int64), update_cell_offsets=np.array(update_offsets, dtype=np.int64))

def compute_updated_area(mask, transform):
    """Sum WGS84 ellipsoidal cell areas instead of treating angular grid sizes as square meters."""
    geod = Geod(ellps="WGS84")
    area = 0.
    for row, column in np.argwhere(mask):
        longitude, latitude = transform@(np.array([column, column+1, column+1, column]), np.array([row, row, row+1, row+1]))
        cell_area, perimeter = geod.polygon_area_perimeter(longitude, latitude)
        area += abs(cell_area)
    return area

def write_update_rasters(folder, profile, tags, result):
    """Write and reread the original-grid float32 DSM and four uint8 evidence masks."""
    for name in ["updated_dsm", "coverage_mask", "overlap_mask", "reliability_mask", "update_mask"]:
        values = result["updated"] if name == "updated_dsm" else result[name].astype(np.uint8)
        current = profile.copy()
        if name != "updated_dsm":
            current.update(dtype="uint8", nodata=255)
        path = folder/(name+".tif")
        with rasterio.open(path, "w", **current) as output:
            output.write(values, 1)
            output.update_tags(**tags)
        with rasterio.open(path) as output:
            if output.transform != profile["transform"] or output.crs != profile["crs"] or output.nodata != current["nodata"]:
                raise ValueError("Output raster georeferencing or NoData changed")
            np.testing.assert_array_equal(output.read(1), values)

def create_baseline_lineage(output_dir, model):
    """Describe original registered shared points without claiming map-assigned or optimized baseline XYZ."""
    with np.load(output_dir/"shared/map_priors.npz") as priors:
        ids, means, prior_ids = [priors[key].copy() for key in ["shared_point_ids", "means_enu_m", "prior_ids"]]
    positions = np.array([model.points3D[int(index)].xyz for index in ids])
    rows = [(entry.image_id, entry.point2D_idx, index) for index, point in enumerate(ids) for entry in model.points3D[int(point)].track.elements]
    return {"schema_version": np.array(3), "track_ids": ids, "shared_point_ids": ids, "prior_ids": prior_ids, "registered_sfm_xyz_enu_m": positions.copy(), "map_xyz_enu_m": means, "initialization_source": np.full(len(ids), "registered_original_visual"), "initial_xyz_enu_m": positions.copy(), "optimized_xyz_enu_m": positions.copy(), "observation_image_ids": np.array([row[0] for row in rows]), "observation_point2D_ids": np.array([row[1] for row in rows]), "observation_prior_indices": np.array([row[2] for row in rows])}

def run_dsm_update(config, root, output_dir, branch, write_json):
    """Verify dense inputs and execute evidence-only DSM updating with an explicit empty-update outcome."""
    folder = output_dir/branch/"map"
    folder.mkdir(parents=True, exist_ok=True)
    log_path = output_dir/"logs"/f"update_{branch}.log"
    report = {"schema_version": 3, "stage": "update", "branch": branch, "status": "running", "error": None, "gt_inputs_read": False}
    write_json(log_path, report)
    dense = output_dir/branch/"dense"
    dense_report = json.loads((dense/"summary.json").read_text())
    if dense_report["status"] != "complete":
        raise ValueError("DSM update requires completed branch dense preparation")
    for name, digest in dense_report["output_sha256"].items():
        if get_sha256(dense/name) != digest:
            raise ValueError("Dense preparation output changed before DSM update")
    for name, digest in dense_report["source_model_sha256"].items():
        if get_sha256(output_dir/branch/"model"/name) != digest:
            raise ValueError("Optimized camera or sparse model changed before DSM update")
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < 6*1024**3 or shutil.disk_usage(folder).free < 1024**3:
        raise RuntimeError("Insufficient resources for complete point evidence and depth maps")
    policy = {"version": "dsm_reliability_single_cell_layers_v1", "limits": config["update"], "height_gap_m": 1.0, "height_gap_comparison": ">=", "min_points_per_height_layer": 3, "height_reason_priority": ["multiple_height_layers", "ambiguous_height_support"], "mad": "median(abs(height-median(height))); unscaled", "conflict_scope": "optimized shared-point cell only; no dilation", "sampling": "nearest pixel floor(uv+0.5), no clipping", "occlusion": "predicted_z-depth > relative_limit*predicted_z", "reprojection": "maximum bidirectional depth-backprojection error across all depth/normal-consistent support pairs", "depth_uncertainty": "relative_limit*predicted_z is an engineering consistency band, not an estimated covariance", "sparse_dense_link": "same optimized cell and at least two common consistent image IDs"}
    report.update(policy=policy, resources={"memory_available_bytes": available, "disk_free_bytes": shutil.disk_usage(folder).free}, source_dense_summary_sha256=get_sha256(dense/"summary.json"))
    write_json(log_path, report)
    origin = json.loads((output_dir/"shared/origin.json").read_text())
    dsm_path = root/config["dsm_path"]
    dsm_hash = get_sha256(dsm_path)
    with rasterio.open(dsm_path) as source:
        original, profile, tags = source.read(1), source.profile, source.tags()
    model, views, view_hashes = load_update_views(output_dir, branch, dense_report)
    points = read_fused_cloud(dense/"fused_metric.ply")
    positions, normals = [np.column_stack([points[key] for key in keys]).astype(np.float64) for keys in [["x", "y", "z"], ["nx", "ny", "nz"]]]
    point_indices, image_indices = read_visibility(dense/"fused_metric.ply.vis", len(points), len(views))
    native_support = np.zeros((len(points), len(views)), dtype=bool)
    native_support[point_indices, image_indices] = True
    print(f"update {branch}: checking {len(points)} dense points and {len(views)} views", flush=True)
    evidence = check_point_reliability(positions, normals, native_support, views, config["update"])
    geographic = convert_enu_to_geodetic(positions, origin)
    if not np.isfinite(geographic).all():
        raise ValueError("Dense ENU to geographic conversion produced invalid coordinates")
    cells, inside = get_grid_cells(geographic, profile["transform"], original.shape)
    if branch == "joint":
        joint = json.loads((output_dir/"joint/diagnostics.json").read_text())
        source_lineage = Path(joint["run_dir"])/"map_point_lineage.npz"
        with np.load(source_lineage) as archive:
            base = {key: archive[key].copy() for key in archive.files}
        report["source_lineage_sha256"] = get_sha256(source_lineage)
    else:
        base = create_baseline_lineage(output_dir, model)
    lineage, conflicts = check_sparse_dense(model, base, views, origin, profile["transform"], original.shape, config["update"])
    result = build_dsm_update(original, cells, geographic[:, 2], evidence, conflicts, config["update"])
    link_dense_support(lineage, cells, evidence, result["update_mask"])
    write_update_rasters(folder, profile, tags, result)
    np.savez_compressed(folder/"point_evidence.npz", dense_point_ids=np.arange(len(points)), geographic_positions=geographic, raster_cells=cells, inside_raster=inside, **{key: value for key, value in evidence.items() if key != "counts"}, **{key: value for key, value in result.items() if key not in ["updated", "coverage_mask", "overlap_mask", "reliability_mask", "update_mask"]})
    lineage_path = output_dir/"joint/map_point_lineage.npz" if branch == "joint" else folder/"sparse_evidence.npz"
    np.savez_compressed(lineage_path, **lineage)
    with rasterio.open(folder/"updated_dsm.tif") as output:
        written = output.read(1)
    np.testing.assert_array_equal(written[~result["update_mask"]], original[~result["update_mask"]])
    for path, digest in {**view_hashes, str(dsm_path): dsm_hash}.items():
        if get_sha256(path) != digest:
            raise ValueError("Input DSM or MVS evidence changed during updating")
    count = int(result["update_mask"].sum())
    area = compute_updated_area(result["update_mask"], profile["transform"])
    reason_counts = {str(name): int((result["height_reason_codes"] == index).sum()) for index, name in enumerate(result["height_reason_names"])}
    states, state_counts = np.unique(lineage["dense_consistency_states"], return_counts=True)
    artifacts = {name: str(folder/(name+".tif")) for name in ["updated_dsm", "coverage_mask", "overlap_mask", "reliability_mask", "update_mask"]}
    artifacts.update(point_evidence=str(folder/"point_evidence.npz"), sparse_evidence=str(lineage_path))
    report.update(status="complete", registered_images=model.num_reg_images(), sparse_points=model.num_points3D(), dense_points=len(points), update_status="updated" if count else "no_valid_update", dsm_update_goal_complete=bool(count), updated_cell_count=count, updated_area_m2=area, numerically_changed_cell_count=int(np.count_nonzero(written[result["update_mask"]] != original[result["update_mask"]])), source_dsm_sha256=dsm_hash, source_view_sha256=view_hashes, point_filter_counts=evidence["counts"], points_inside_raster=int(inside.sum()), cell_height_reason_counts=reason_counts, sparse_states={str(key): int(value) for key, value in zip(states, state_counts)}, sparse_conflict_cell_count=int(conflicts.sum()), mask_counts={key: int(result[key].sum()) for key in ["coverage_mask", "overlap_mask", "reliability_mask", "update_mask"]}, artifacts=artifacts, output_sha256={name: get_sha256(path) for name, path in artifacts.items()})
    write_json(folder/"summary.json", report)
    write_json(log_path, report)
    print(f"update {branch}: complete | update_status={report['update_status']} | cells={count} | area_m2={area:.6f}", flush=True)
    return report
