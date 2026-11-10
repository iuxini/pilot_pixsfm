# 用途：导出六行 WGS84 位姿及 double 经纬高点云，核验序列化和坐标往返。
import json
from pathlib import Path
import shutil
import numpy as np
import pycolmap
from joint_map.dense import read_fused_cloud
from joint_map.geometry import convert_enu_to_geodetic, convert_geodetic_to_enu, convert_pose_from_colmap, convert_pose_to_colmap
from joint_map.optimization import get_sha256

def check_geographic_values(values):
    """Require finite longitude, latitude and ellipsoidal height in the declared geographic order."""
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite geographic XYZ array")
    if (np.abs(values[:, 0]) > 180).any() or (np.abs(values[:, 1]) > 90).any():
        raise ValueError("Geographic longitude or latitude is outside its angular range")

def write_geographic_cloud(path, positions, colors, origin):
    """Write ordered float64 geographic XYZ and unchanged uint8 RGB with explicit coordinate metadata."""
    geographic = convert_enu_to_geodetic(positions, origin)
    check_geographic_values(geographic)
    if colors.shape != geographic.shape or colors.dtype != np.uint8:
        raise ValueError("Expected one unchanged uint8 RGB triple per point")
    values = np.empty(len(geographic), dtype=[(key, "<f8") for key in ["x", "y", "z"]]+[(key, "u1") for key in ["red", "green", "blue"]])
    for index, key in enumerate(["x", "y", "z"]):
        values[key] = geographic[:, index]
    for index, key in enumerate(["red", "green", "blue"]):
        values[key] = colors[:, index]
    header = ["ply", "format binary_little_endian 1.0", "comment horizontal_crs EPSG:4326", "comment x longitude degrees; y latitude degrees; z ellipsoidal_height meters", "comment vertical_datum WGS84 ellipsoidal height", f"element vertex {len(values)}"]
    header.extend(f"property double {key}" for key in ["x", "y", "z"])
    header.extend(f"property uchar {key}" for key in ["red", "green", "blue"])
    with Path(path).open("wb") as stream:
        stream.write(("\n".join(header)+"\nend_header\n").encode("ascii"))
        values.tofile(stream)

def read_geographic_cloud(path):
    """Read the declared geographic double-only XYZ/RGB schema and verify mandatory metadata."""
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply" or stream.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Expected geographic binary little-endian PLY")
        comments, properties, count = [], [], None
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("Incomplete geographic PLY header")
            line = raw.decode("ascii").strip()
            if line == "end_header":
                break
            words = line.split()
            if words[:1] == ["comment"]:
                comments.append(line)
            elif words[:2] == ["element", "vertex"]:
                count = int(words[2])
            elif words[:1] == ["property"]:
                properties.append(line)
            else:
                raise ValueError("Unsupported geographic PLY header")
        expected = [f"property double {key}" for key in ["x", "y", "z"]]+[f"property uchar {key}" for key in ["red", "green", "blue"]]
        required = ["comment horizontal_crs EPSG:4326", "comment x longitude degrees; y latitude degrees; z ellipsoidal_height meters", "comment vertical_datum WGS84 ellipsoidal height"]
        if properties != expected or count is None or count <= 0 or not all(line in comments for line in required):
            raise ValueError("Geographic PLY precision, fields or coordinate metadata differ from the protocol")
        values = np.fromfile(stream, dtype=[(key, "<f8") for key in ["x", "y", "z"]]+[(key, "u1") for key in ["red", "green", "blue"]])
    if len(values) != count:
        raise ValueError("Geographic PLY point count differs from its header")
    check_geographic_values(np.column_stack([values[key] for key in ["x", "y", "z"]]))
    return values

def check_geographic_cloud(path, positions, colors, origin):
    """Check all persisted points and colors exactly before a micrometer-level ENU round trip."""
    values = read_geographic_cloud(path)
    geographic = np.column_stack([values[key] for key in ["x", "y", "z"]])
    np.testing.assert_array_equal(geographic, convert_enu_to_geodetic(positions, origin))
    np.testing.assert_array_equal(np.column_stack([values[key] for key in ["red", "green", "blue"]]), colors)
    difference = np.abs(convert_geodetic_to_enu(geographic, origin)-positions)
    maximum = float(np.max(difference))
    if maximum > 1e-6:
        raise ValueError("Persisted geographic cloud failed the fixed ENU round-trip tolerance")
    return {"passed": True, "point_count": len(values), "xyz_storage": "float64", "rgb_and_point_order": "exact", "max_abs_coordinate_error_m": maximum, "tolerance_m": 1e-6}

def read_geographic_poses(path, names):
    """Read exactly the six ordered headerless pose rows and reject missing or nonfinite values."""
    rows = [line.split() for line in Path(path).read_text().splitlines()]
    if len(names) != 6 or len(set(names)) != 6 or len(rows) != 6 or any(len(row) != 7 for row in rows) or [row[0] for row in rows] != names:
        raise ValueError("Expected exactly six ordered image_name/lon/lat/height/roll/pitch/yaw rows")
    values = np.array([[float(value) for value in row[1:]] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Exported pose contains nonfinite numeric values")
    check_geographic_values(values[:, :3])
    return values

def export_geographic_poses(path, model, names, origin):
    """Serialize existing camera poses with local geographic ENU bases and verify the inverse transform."""
    images = {image.name: image for image in model.images.values()}
    if set(images) != set(names) or model.num_reg_images() != 6:
        raise ValueError("Geographic pose export requires exactly the six registered input images")
    rows = []
    for name in names:
        image = images[name]
        position, angles = convert_pose_from_colmap(image.rotmat(), image.tvec, origin)
        rows.append(name+" "+" ".join(f"{value:.17f}" for value in np.r_[position, angles]))
    Path(path).write_text("\n".join(rows)+"\n", encoding="utf-8")
    values = read_geographic_poses(path, names)
    errors = []
    for name, row in zip(names, values):
        image = images[name]
        rotation, translation = convert_pose_to_colmap(row[:3], row[3:], origin)
        center = -image.rotmat().T@image.tvec
        coordinate_error = float(np.max(np.abs(convert_geodetic_to_enu(row[:3], origin)-center)))
        rotation_error = float(np.max(np.abs(rotation-image.rotmat())))
        translation_error = float(np.max(np.abs(translation-image.tvec)))
        if coordinate_error > 1e-6 or rotation_error > 1e-10 or translation_error > 1e-6:
            raise ValueError("Persisted geographic pose failed the fixed coordinate/rotation round-trip tolerance")
        errors.append({"image_name": name, "max_abs_center_error_m": coordinate_error, "max_abs_rotation_matrix_error": rotation_error, "max_abs_tvec_error_m": translation_error})
    return {"passed": True, "pose_count": 6, "coordinate_tolerance_m": 1e-6, "rotation_matrix_tolerance": 1e-10, "rows": errors}

def run_geographic_export(config, root, output_dir, branch, write_json):
    """Verify branch handoffs and export full sparse/dense geometry without reading evaluation inputs."""
    folder = output_dir/branch/"export"
    folder.mkdir(parents=True, exist_ok=True)
    log_path = output_dir/"logs"/f"export_{branch}.log"
    report = {"schema_version": 3, "stage": "export", "branch": branch, "status": "running", "error": None, "gt_inputs_read": False}
    write_json(log_path, report)
    dense_dir, map_dir, model_dir = [output_dir/branch/name for name in ["dense", "map", "model"]]
    dense = json.loads((dense_dir/"summary.json").read_text())
    update = json.loads((map_dir/"summary.json").read_text())
    if dense["status"] != "complete" or update["status"] != "complete" or update["update_status"] not in ["updated", "no_valid_update"]:
        raise ValueError("Geographic export requires complete dense/update execution with explicit update status")
    has_update = update["updated_cell_count"] > 0
    if has_update != (update["update_status"] == "updated") or has_update != update["dsm_update_goal_complete"] or (not has_update and update["updated_area_m2"] != 0):
        raise ValueError("DSM execution and actual update status are inconsistent")
    if get_sha256(dense_dir/"summary.json") != update["source_dense_summary_sha256"]:
        raise ValueError("Dense summary changed after DSM updating")
    inputs = {str(model_dir/name): digest for name, digest in dense["source_model_sha256"].items()}
    inputs.update({str(dense_dir/name): digest for name, digest in dense["output_sha256"].items()})
    inputs.update({str(update["artifacts"][name]): digest for name, digest in update["output_sha256"].items()})
    for path in [output_dir/"shared/origin.json", dense_dir/"summary.json", map_dir/"summary.json"]:
        inputs[str(path)] = get_sha256(path)
    for path, digest in inputs.items():
        if get_sha256(path) != digest:
            raise ValueError("Geographic export handoff changed: "+path)
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < 512*1024**2 or shutil.disk_usage(folder).free < 2*(dense_dir/"fused_metric.ply").stat().st_size+1024**2:
        raise RuntimeError("Insufficient resources for complete double-precision geographic export")
    origin = json.loads((output_dir/"shared/origin.json").read_text())
    model = pycolmap.Reconstruction(str(model_dir))
    names = config["image_names"]
    policy = {"horizontal_crs": "EPSG:4326", "height_datum": "WGS84_ellipsoidal", "xyz_order": ["longitude_deg", "latitude_deg", "ellipsoidal_height_m"], "pose_columns": ["image_name", "longitude_deg", "latitude_deg", "ellipsoidal_height_m", "roll_deg", "pitch_deg", "yaw_deg"], "pose_convention": "pitch=0 is nadir; xyz Euler uses [pitch,roll,yaw] at each camera's geographic ENU basis", "pose_decimal_places": 17, "cloud_xyz_storage": "float64", "sparse_point_order": "ascending COLMAP point3D_id", "dense_point_scope": "all native fused points in unchanged fused_metric.ply order", "coordinate_roundtrip_tolerance_m": 1e-6, "rotation_matrix_roundtrip_tolerance": 1e-10, "verification_scope": "coordinate and serialization correctness; not GT accuracy"}
    report.update(policy=policy, input_sha256=inputs, source_update_status=update["update_status"], dsm_update_goal_complete=update["dsm_update_goal_complete"], resources={"memory_available_bytes": available, "disk_free_bytes": shutil.disk_usage(folder).free})
    write_json(log_path, report)
    pose_check = export_geographic_poses(folder/"poses_wgs84.txt", model, names, origin)
    point_ids = sorted(model.points3D)
    positions = np.array([model.points3D[index].xyz for index in point_ids])
    colors = np.array([model.points3D[index].color for index in point_ids], dtype=np.uint8)
    write_geographic_cloud(folder/"sparse_wgs84.ply", positions, colors, origin)
    sparse_check = check_geographic_cloud(folder/"sparse_wgs84.ply", positions, colors, origin)
    cloud = read_fused_cloud(dense_dir/"fused_metric.ply")
    positions = np.column_stack([cloud[key] for key in ["x", "y", "z"]])
    colors = np.column_stack([cloud[key] for key in ["red", "green", "blue"]])
    if len(cloud) != dense["dense_points"]:
        raise ValueError("Metric dense point count differs from its completed branch summary")
    write_geographic_cloud(folder/"dense_wgs84.ply", positions, colors, origin)
    dense_check = check_geographic_cloud(folder/"dense_wgs84.ply", positions, colors, origin)
    for path, digest in inputs.items():
        if get_sha256(path) != digest:
            raise ValueError("Export mutated an immutable source or its inputs changed during export")
    artifacts = {name: str(folder/name) for name in ["poses_wgs84.txt", "sparse_wgs84.ply", "dense_wgs84.ply"]}
    report.update(status="complete", registered_images=model.num_reg_images(), sparse_points=model.num_points3D(), dense_points=len(cloud), checks={"poses": pose_check, "sparse_cloud": sparse_check, "dense_cloud": dense_check}, artifacts=artifacts, output_sha256={name: get_sha256(path) for name, path in artifacts.items()})
    write_json(folder/"summary.json", report)
    write_json(log_path, report)
    print(f"export {branch}: complete | poses=6 | sparse_points={model.num_points3D()} | dense_points={len(cloud)}", flush=True)
    return report
