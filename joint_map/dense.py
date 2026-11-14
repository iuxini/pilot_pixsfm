# 用途：准备两分支米制稠密点云，核验深度、相机和原生可见性索引。
import json
import shutil
import struct
import subprocess
from pathlib import Path
import numpy as np
import pycolmap
from joint_map.joint import check_registered_visual_model, load_confirmed_handoff
from joint_map.optimization import get_observations, get_sha256

def read_depth_map(path):
    """Read an unchanged native COLMAP depth array with explicit dimensions."""
    with Path(path).open("rb") as stream:
        header = bytearray()
        while header.count(b"&") < 3:
            value = stream.read(1)
            if not value or len(header) > 100:
                raise ValueError("Invalid COLMAP array header")
            header.extend(value)
        width, height, channels = map(int, header[:-1].split(b"&"))
        data = np.fromfile(stream, dtype="<f4")
    if min(width, height, channels) <= 0 or data.size != width*height*channels:
        raise ValueError("Incomplete COLMAP array")
    return data.reshape((width, height, channels), order="F").transpose(1, 0, 2)

def read_fused_cloud(path):
    """Read the binary XYZ, normal and RGB records without changing their order."""
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply" or stream.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Expected native binary little-endian PLY")
        count, fields = None, []
        types = {"float": "<f4", "double": "<f8", "uchar": "u1"}
        while True:
            line = stream.readline().decode("ascii").strip()
            if line == "end_header":
                break
            words = line.split()
            if words[:2] == ["element", "vertex"]:
                count = int(words[2])
            elif words[:1] == ["property"] and len(words) == 3 and words[1] in types:
                fields.append((words[2], types[words[1]]))
            else:
                raise ValueError("Unsupported PLY header")
        if count is None or count <= 0 or [row[0] for row in fields] != ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"]:
            raise ValueError("Incomplete native fused-cloud schema")
        points = np.fromfile(stream, dtype=np.dtype(fields))
    if len(points) != count or not all(np.isfinite(points[key]).all() for key in ["x", "y", "z", "nx", "ny", "nz"]):
        raise ValueError("Incomplete or nonfinite fused point cloud")
    return points

def write_metric_cloud(path, points, scale, rotation, translation):
    """Transform ordered XYZ and normals while preserving original point and RGB identities."""
    xyz = np.column_stack([points[key] for key in ["x", "y", "z"]]).astype(np.float64)
    normals = np.column_stack([points[key] for key in ["nx", "ny", "nz"]]).astype(np.float64)
    transformed = scale*xyz@rotation.T+translation
    rotated = normals@rotation.T
    lengths = np.linalg.norm(rotated, axis=1)
    if not np.isfinite(rotated).all() or np.any(lengths <= 0):
        raise ValueError("Fused cloud contains invalid or zero normals")
    rotated /= lengths[:, None]
    dtype = np.dtype([(key, "<f8") for key in ["x", "y", "z", "nx", "ny", "nz"]]+[(key, "u1") for key in ["red", "green", "blue"]])
    result = np.empty(len(points), dtype=dtype)
    for index, key in enumerate(["x", "y", "z"]):
        result[key] = transformed[:, index]
    for index, key in enumerate(["nx", "ny", "nz"]):
        result[key] = rotated[:, index]
    for key in ["red", "green", "blue"]:
        result[key] = points[key]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}"]
    header.extend(f"property {'uchar' if key in ['red', 'green', 'blue'] else 'double'} {key}" for key in result.dtype.names)
    with Path(path).open("wb") as stream:
        stream.write(("\n".join(header)+"\nend_header\n").encode("ascii"))
        result.tofile(stream)
    return result

def read_image_order(path):
    """Recover MVS image_idx from binary record order used by native Reconstruction::AddImage."""
    rows = []
    with (Path(path)/"images.bin").open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for index in range(count):
            data = struct.unpack("<i7di", stream.read(64))
            name = bytearray()
            while True:
                value = stream.read(1)
                if not value:
                    raise ValueError("Incomplete binary image name")
                if value == b"\0":
                    break
                name.extend(value)
            points = struct.unpack("<Q", stream.read(8))[0]
            stream.seek(24*points, 1)
            rows.append({"image_idx": index, "image_id": data[0], "name": name.decode("utf-8"), "camera_id": data[-1]})
        if stream.tell() != (Path(path)/"images.bin").stat().st_size or len({row["image_id"] for row in rows}) != count:
            raise ValueError("Invalid binary image ordering")
    return rows

def read_visibility(path, point_count, image_count):
    """Validate every native visibility index and retain its exact point association."""
    image_indices, point_indices = [], []
    with Path(path).open("rb") as stream:
        if struct.unpack("<Q", stream.read(8))[0] != point_count:
            raise ValueError("Visibility point count differs from PLY")
        for point_index in range(point_count):
            count = struct.unpack("<I", stream.read(4))[0]
            if count < 1 or count > image_count:
                raise ValueError("Invalid visibility support count")
            indices = np.frombuffer(stream.read(4*count), dtype="<u4")
            if len(indices) != count or np.any(indices >= image_count) or len(set(indices)) != count:
                raise ValueError("Invalid or repeated native visibility image_idx")
            image_indices.extend(indices)
            point_indices.extend([point_index]*count)
        if stream.read(1):
            raise ValueError("Unexpected bytes after native visibility records")
    return np.asarray(point_indices, dtype=np.int64), np.asarray(image_indices, dtype=np.int64)

def check_sparse_handoff(source, workspace):
    """Check unchanged sparse XYZ, tracks, camera calibration and serialized pose geometry."""
    if set(source.images) != set(workspace.images) or set(source.cameras) != set(workspace.cameras) or set(source.points3D) != set(workspace.points3D):
        raise ValueError("MVS sparse identities differ from their input model")
    for index, image in source.images.items():
        other = workspace.images[index]
        if (image.name, image.camera_id) != (other.name, other.camera_id):
            raise ValueError("MVS image identity differs from its input model")
        np.testing.assert_allclose(image.rotmat(), other.rotmat(), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(image.tvec, other.tvec)
    for index, point in source.points3D.items():
        np.testing.assert_array_equal(point.xyz, workspace.points3D[index].xyz)
    for index, camera in source.cameras.items():
        other = workspace.cameras[index]
        if (camera.model_name, camera.width, camera.height) != (other.model_name, other.width, other.height):
            raise ValueError("MVS camera model or dimensions changed")
        np.testing.assert_array_equal(camera.params, other.params)
    original = {tuple(row[:3]): row[3] for row in get_observations(source)}
    actual = {tuple(row[:3]): row[3] for row in get_observations(workspace)}
    if original.keys() != actual.keys():
        raise ValueError("MVS sparse observation identities changed")
    changes = []
    for identity, pixel in original.items():
        camera = source.cameras[source.images[identity[1]].camera_id]
        expected = camera.world_to_image(camera.image_to_world(pixel))
        np.testing.assert_array_equal(expected, actual[identity])
        changes.append(float(np.max(np.abs(actual[identity]-pixel))))
    return {"passed": True, "pose_rotation_atol": 1e-12, "xyz_calibration_translation_identities": "exact", "pixel_policy": "exact_native_image_to_world_then_world_to_image", "observation_count": len(original), "max_undistortion_pixel_change": max(changes)}

def check_dense_workspace(path, points, config, depth_scale):
    """Audit six geometric depth maps and reproject native visibility associations at depth resolution."""
    path = Path(path)
    model = pycolmap.Reconstruction(str(path/"sparse"))
    order = read_image_order(path/"sparse")
    if sorted(row["name"] for row in order) != sorted(config["image_names"]):
        raise ValueError("MVS workspace does not contain exactly the six requested images")
    point_indices, image_indices = read_visibility(path/"fused.ply.vis", len(points), len(order))
    xyz = np.column_stack([points[key] for key in ["x", "y", "z"]]).astype(np.float64)
    checks = []
    for row in order:
        image = model.images[row["image_id"]]
        camera = model.cameras[image.camera_id]
        if (image.name, image.camera_id) != (row["name"], row["camera_id"]):
            raise ValueError("Native binary image order does not match loaded camera identities")
        depth = read_depth_map(path/"stereo/depth_maps"/(image.name+".geometric.bin"))
        normal = read_depth_map(path/"stereo/normal_maps"/(image.name+".geometric.bin"))
        if not (path/"stereo/consistency_graphs"/(image.name+".geometric.bin")).is_file():
            raise ValueError("Missing native geometric consistency graph")
        height, width, channels = depth.shape
        valid_depth = np.isfinite(depth[:, :, 0]) & (depth[:, :, 0] > 0)
        if channels != 1 or normal.shape != (height, width, 3) or max(height, width) != config["dense"]["max_image_size"] or not valid_depth.any():
            raise ValueError("Missing valid full-resolution geometric depths or matching normals")
        valid_normals = np.isfinite(normal).all(axis=2) & (np.linalg.norm(normal, axis=2) > 0)
        if not np.all(valid_normals[valid_depth]):
            raise ValueError("Geometric depths contain missing, nonfinite or zero surface normals")
        selected = point_indices[image_indices == row["image_idx"]]
        local = xyz[selected]@image.rotmat().T+image.tvec
        positive = np.isfinite(local).all(axis=1) & (local[:, 2] > 0)
        pixels = np.full((len(local), 2), np.nan)
        pixels[positive] = local[positive, :2]/local[positive, 2:]*camera.params[:2]+camera.params[2:]
        pixels *= [width/camera.width, height/camera.height]
        inside = positive & np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        coords = np.floor(pixels[inside]+0.5).astype(np.int64)
        nearest_inside = (coords[:, 0] < width) & (coords[:, 1] < height)
        depths = np.full(np.count_nonzero(inside), np.nan)
        depths[nearest_inside] = depth[coords[nearest_inside, 1], coords[nearest_inside, 0], 0]
        relative = np.full(len(depths), np.inf)
        usable = np.isfinite(depths) & (depths > 0)
        relative[usable] = np.abs(depths[usable]-local[inside, 2][usable])/local[inside, 2][usable]
        agreement = relative <= config["update"]["max_relative_depth_error"]
        if not len(selected) or not agreement.any():
            raise ValueError("Visibility index mapping has no depth-consistent support for an image")
        checks.append(dict(row, width=width, height=height, valid_depth_pixels=int(valid_depth.sum()), valid_normal_pixels=int(np.count_nonzero(valid_depth & valid_normals)), visibility_associations=len(selected), positive_depth_count=int(positive.sum()), inside_image_count=int(inside.sum()), sampled_valid_depth_count=int(usable.sum()), depth_agreement_count=int(agreement.sum()), depth_disagreement_or_missing_count=int(len(selected)-agreement.sum()), depth_scale_to_m=depth_scale, depth_relative_threshold=config["update"]["max_relative_depth_error"]))
    return order, checks

def get_dense_workspace_fingerprints(path, image_names):
    """Bind reused or published MVS images, geometry, depths, normals and visibility by content."""
    names = ["fused.ply", "fused.ply.vis", "stereo/patch-match.cfg", "stereo/fusion.cfg"]
    names.extend("sparse/"+name for name in ["cameras.bin", "images.bin", "points3D.bin"])
    names.extend("images/"+name for name in image_names)
    names.extend(f"stereo/{folder}/{name}.geometric.bin" for folder in ["depth_maps", "normal_maps", "consistency_graphs"] for name in image_names)
    return {name: get_sha256(path/name) for name in names}

def run_dense_preparation(config, root, output_dir, branch, write_json):
    """Prepare one dense branch and publish its summary only after all required artifact checks pass."""
    destination = output_dir/branch/"dense"
    destination.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 3, "stage": "dense", "branch": branch, "status": "running", "error": None, "gt_inputs_read": False}
    log_path = output_dir/"logs"/f"dense_{branch}.log"
    write_json(log_path, report)
    registration = load_confirmed_handoff(output_dir)[1]
    native_dir = root/config["native_dense_dir"]
    original = json.loads((native_dir/"summary.json").read_text())
    if original["status"] != "PASS" or original["max_image_size"] != config["dense"]["max_image_size"]:
        raise ValueError("Original MVS settings or completion evidence do not match")
    source_model = output_dir/branch/"model"
    hashes = {name: get_sha256(source_model/name) for name in ["cameras.bin", "images.bin", "points3D.bin"]}
    model = pycolmap.Reconstruction(str(source_model))
    identity = {"source_model_sha256": hashes, "registration_sha256": get_sha256(output_dir/"shared/registration.json"), "colmap_binary_sha256": get_sha256(root/config["colmap_binary"]), "image_sha256": {name: get_sha256(root/config["image_dir"]/name) for name in config["image_names"]}, "original_commands": original["commands"]}
    identity["native_workspace_sha256"] = get_dense_workspace_fingerprints(native_dir, config["image_names"])
    if any(identity["native_workspace_sha256"]["images/"+name] != digest for name, digest in identity["image_sha256"].items()):
        raise ValueError("Original PINHOLE MVS images differ from the configured input images")
    summary_path = destination/"summary.json"
    if summary_path.is_file() and json.loads(summary_path.read_text()).get("input_identity") != identity:
        raise ValueError("Existing dense workspace has different or unverified inputs; refusing stale reuse")
    if not summary_path.is_file() and any(destination.iterdir()):
        raise ValueError("Existing dense workspace has no input provenance; refusing stale reuse")
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < 6*1024**3:
        raise RuntimeError("Insufficient memory for the unchanged native MVS cache and fusion")
    report.update(registered_images=model.num_reg_images(), sparse_points=model.num_points3D(), source_model=str(source_model), source_model_sha256=hashes, max_image_size=config["dense"]["max_image_size"], commands={}, update_status="not_run")
    report.update(input_identity=identity, resources={"memory_available_bytes": available, "disk_free_bytes": shutil.disk_usage(destination).free})
    write_json(summary_path, report)
    if branch == "baseline":
        native = pycolmap.Reconstruction(str(root/config["native_sfm_dir"]))
        check_registered_visual_model(model, native, registration)
        workspace = native_dir
        report["sparse_handoff"] = check_sparse_handoff(native, pycolmap.Reconstruction(str(workspace/"sparse")))
        transform = (registration["scale"], np.asarray(registration["rotation"]), np.asarray(registration["translation_enu_m"]))
    else:
        joint = json.loads((output_dir/"joint/diagnostics.json").read_text())
        if joint["status"] != "complete" or joint["output_model_sha256"] != hashes:
            raise ValueError("Joint MVS input differs from the completed module-four model")
        required = 3*sum(path.stat().st_size for path in native_dir.rglob("*") if path.is_file())
        if shutil.disk_usage(destination).free < required:
            raise RuntimeError("Insufficient disk space for unchanged-resolution dense reconstruction")
        workspace = destination
        for name in ["undistort", "patch_match", "fusion"]:
            command = list(original["commands"][name])
            command[0] = str(root/config["colmap_binary"])
            replacements = {"--image_path": root/config["image_dir"], "--input_path": source_model, "--output_path": destination if name == "undistort" else destination/"fused.ply", "--workspace_path": destination}
            for flag, value in replacements.items():
                if flag in command:
                    command[command.index(flag)+1] = str(value)
            report["commands"][name] = command
            write_json(log_path, report)
            print(f"dense {branch}: running {name}", flush=True)
            with (destination/f"{name}.log").open("w") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode:
                report.update(status="failed", error=f"COLMAP {name} exited with code {process.returncode}; see {destination/(name+'.log')}")
                write_json(log_path, report)
                write_json(destination/"summary.json", report)
                return report
        report["sparse_handoff"] = check_sparse_handoff(model, pycolmap.Reconstruction(str(workspace/"sparse")))
        transform = (1., np.eye(3), np.zeros(3))
    points = read_fused_cloud(workspace/"fused.ply")
    order, checks = check_dense_workspace(workspace, points, config, transform[0])
    workspace_hashes = get_dense_workspace_fingerprints(workspace, config["image_names"])
    if any(workspace_hashes["images/"+name] != digest for name, digest in identity["image_sha256"].items()):
        raise ValueError("MVS workspace image content differs from its input images")
    write_json(destination/"image_order.json", {"schema_version": 3, "source": "native images.bin record order / Reconstruction::RegImageIds", "sparse_model": str(workspace/"sparse"), "images_bin_sha256": get_sha256(workspace/"sparse/images.bin"), "images": order})
    result = write_metric_cloud(destination/"fused_metric.ply", points, *transform)
    shutil.copy2(workspace/"fused.ply.vis", destination/"fused_metric.ply.vis")
    restored = read_fused_cloud(destination/"fused_metric.ply")
    if not np.array_equal(restored, result) or get_sha256(workspace/"fused.ply.vis") != get_sha256(destination/"fused_metric.ply.vis"):
        raise ValueError("Metric PLY or visibility identities changed during serialization")
    for name, digest in hashes.items():
        if get_sha256(source_model/name) != digest:
            raise ValueError("Sparse source model changed during MVS")
    if get_sha256(output_dir/"shared/registration.json") != identity["registration_sha256"] or get_dense_workspace_fingerprints(native_dir, config["image_names"]) != identity["native_workspace_sha256"]:
        raise ValueError("Frozen registration or reused native MVS evidence changed during preparation")
    report["workspace_sha256"] = workspace_hashes
    report.update(status="complete", dense_points=len(points), coordinate_system="local_ENU_m", workspace_path=str(workspace), workspace_sparse_model=str(workspace/"sparse"), depth_scale_to_m=transform[0], depth_maps=checks, image_order_path=str(destination/"image_order.json"), visibility_point_order_preserved=True, dense_filtering="native geometric fusion; additional reliability filtering belongs to module six", output_sha256={name: get_sha256(destination/name) for name in ["fused_metric.ply", "fused_metric.ply.vis", "image_order.json"]}, artifacts={"dense_cloud": str(destination/"fused_metric.ply"), "visibility": str(destination/"fused_metric.ply.vis"), "image_order": str(destination/"image_order.json"), "summary": str(destination/"summary.json")})
    write_json(destination/"summary.json", report)
    write_json(log_path, report)
    print(f"dense {branch}: complete | images={len(order)} | points={len(points)}", flush=True)
    return report
