# 用途：调度地图重建、可靠性更新、地理导出及独立 GT 评价验收。
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pycolmap
import rasterio
from PIL import Image
from joint_map.geometry import create_origin

PROJECT_ROOT = Path(__file__).resolve().parent

def write_json(path, data):
    """Write strict JSON with readable indentation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def get_file_fingerprint(path):
    """Return an absolute path, byte count and streamed SHA-256 fingerprint."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}

def check_config(config):
    """Validate the fixed reconstruction protocol and reject evaluation fields."""
    expected = {"model": "PINHOLE", "width": 3840, "height": 2160, "params": [2700.0, 2700.0, 1920.0, 1080.0], "fixed": True}
    if config["schema_version"] != 3 or config["camera"] != expected:
        raise ValueError("Expected schema v3 and the planned fixed PINHOLE camera")
    if config["image_names"] != [f"{index}_0.png" for index in range(6)]:
        raise ValueError("Expected the six planned images in fixed order")
    pending = [config]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if any(key.lower().startswith("gt_") or "evaluation" in key.lower() for key in value):
                raise ValueError("Reconstruction configuration must not contain evaluation fields")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)

def check_environment():
    """Verify the existing pixsfm interpreter and import required installed libraries."""
    if Path(sys.prefix) != Path("/opt/anaconda3/envs/pixsfm"):
        raise RuntimeError("Use /opt/anaconda3/envs/pixsfm/bin/python without changing the environment")
    modules = {}
    for name in ["numpy", "scipy", "rasterio", "pyproj", "h5py", "cv2", "torch", "pycolmap", "pyceres", "pixsfm", "hloc", "PIL"]:
        module = importlib.import_module(name)
        modules[name] = {"version": str(getattr(module, "__version__", "unreported")), "path": module.__file__}
    return {"python": sys.executable, "prefix": sys.prefix, "modules": modules}

def check_inputs(config):
    """Inspect reconstruction inputs read-only and derive the map based ENU origin."""
    fingerprints = {}
    paths = [PROJECT_ROOT / config["image_dir"] / name for name in config["image_names"]]
    paths.extend(PROJECT_ROOT / config[key] for key in ["dom_path", "dsm_path", "colmap_binary"])
    paths.extend(PROJECT_ROOT / value for value in config["frontend"].values())
    paths.extend(PROJECT_ROOT / config["native_sfm_dir"] / name for name in ["cameras.bin", "images.bin", "points3D.bin"])
    paths.extend(PROJECT_ROOT / config["native_dense_dir"] / name for name in ["fused.ply", "fused.ply.vis"])
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        fingerprints[str(path.relative_to(PROJECT_ROOT))] = get_file_fingerprint(path)
    if not os.access(PROJECT_ROOT / config["colmap_binary"], os.X_OK):
        raise ValueError("The planned COLMAP binary is not executable")
    camera = config["camera"]
    for name in config["image_names"]:
        with Image.open(PROJECT_ROOT / config["image_dir"] / name) as image:
            if image.size != (camera["width"], camera["height"]) or image.mode != "RGB":
                raise ValueError(f"Unexpected image size or color mode: {name}")
            image.verify()
    with rasterio.open(PROJECT_ROOT / config["dom_path"]) as dom, rasterio.open(PROJECT_ROOT / config["dsm_path"]) as dsm:
        if dom.crs != rasterio.crs.CRS.from_epsg(4326) or dsm.crs != dom.crs:
            raise ValueError("Both maps must have the planned EPSG:4326 horizontal CRS")
        if dom.shape != (395, 404) or dsm.shape != dom.shape or dsm.transform != dom.transform:
            raise ValueError("Expected the planned common 395 by 404 map grid")
        if dom.count != 3 or dom.dtypes != ("int16",) * 3 or dom.nodata != 255:
            raise ValueError("Unexpected DOM bands, data type or NoData")
        if dsm.count != 1 or dsm.dtypes != ("float32",) or dsm.nodata != -9999:
            raise ValueError("Unexpected DSM bands, data type or NoData")
        heights = dsm.read(1, masked=True)
        valid = ~np.ma.getmaskarray(heights) & np.isfinite(heights.data)
        if not valid.any():
            raise ValueError("DSM has no valid finite heights")
        longitude = (dsm.bounds.left + dsm.bounds.right) / 2
        latitude = (dsm.bounds.bottom + dsm.bounds.top) / 2
        origin = create_origin(longitude, latitude, np.median(heights.data[valid]))
        maps = {"crs": str(dsm.crs), "height_datum": "WGS84_ellipsoidal", "height_datum_basis": "user_agreement", "shape": list(dsm.shape), "transform": list(dsm.transform), "valid_dsm_pixels": int(valid.sum()), "dom_nodata": dom.nodata, "dsm_nodata": dsm.nodata}
    model = pycolmap.Reconstruction(str(PROJECT_ROOT / config["native_sfm_dir"]))
    if sorted(image.name for image in model.images.values()) != config["image_names"]:
        raise ValueError("Native model must contain exactly the six planned registered images")
    for native_camera in model.cameras.values():
        if native_camera.model_name != camera["model"] or (native_camera.width, native_camera.height) != (camera["width"], camera["height"]) or not np.array_equal(native_camera.params, camera["params"]):
            raise ValueError("Native model intrinsics differ from the fixed input camera")
    return origin, {"input_fingerprints": fingerprints, "camera": camera, "maps": maps, "registered_images": model.num_reg_images(), "sparse_points": model.num_points3D()}

def run_check(config_path):
    """Run module one and mark complete only after all input checks succeed."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    check_config(config)
    environment = check_environment()
    output_dir = PROJECT_ROOT / config["output_dir"]
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"schema_version": 3, "stages": []}
    for record in manifest["stages"]:
        if record["status"] == "running":
            record.update(status="failed", error=f"Stage {record['stage']} did not complete during its previous invocation")
    record = {"stage": "check", "branch": None, "status": "running", "artifacts": {}, "counts": {}, "error": None}
    manifest["stages"].append(record)
    manifest.update(status="running", config=get_file_fingerprint(config_path))
    write_json(manifest_path, manifest)
    origin, report = check_inputs(config)
    origin_path = output_dir / "shared/origin.json"
    log_path = output_dir / "logs/check.log"
    report.update(schema_version=3, stage="check", status="complete", environment=environment, config=manifest["config"], origin=origin, gt_inputs_read=False)
    write_json(origin_path, origin)
    write_json(log_path, report)
    record.update(status="complete", artifacts={"origin": str(origin_path), "diagnostics": str(log_path)}, counts={"registered_images": report["registered_images"], "sparse_points": report["sparse_points"], "map_priors": 0})
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"check: PASS | images={report['registered_images']} | sparse_points={report['sparse_points']} | valid_dsm_pixels={report['maps']['valid_dsm_pixels']}")
    print(f"Input fingerprints: {len(report['input_fingerprints'])} | GT inputs read: false")
    print(f"Origin: {origin_path}\nReport: {log_path}\nManifest: {manifest_path}")

def run_main():
    """Parse and execute the currently implemented pipeline stages."""
    parser = argparse.ArgumentParser(description="PixSfM map pipeline: input checks, registration and joint optimization")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["check", "baseline", "initialize", "validate_priors", "register", "joint", "dense", "update", "export", "evaluate"], required=True)
    parser.add_argument("--branch", choices=["baseline", "joint"])
    parser.add_argument("--evaluation-config", type=Path)
    arguments = parser.parse_args()
    if (arguments.stage in ["dense", "update", "export"]) != (arguments.branch is not None):
        parser.error("--branch is required only for dense/update/export")
    if (arguments.stage == "evaluate") != (arguments.evaluation_config is not None):
        parser.error("--evaluation-config is required only for evaluate")
    config_path = (PROJECT_ROOT / arguments.config).resolve()
    if arguments.stage == "check":
        run_check(config_path)
    elif arguments.stage in ["dense", "update", "export"]:
        run_map_stage(config_path, arguments.stage, arguments.branch)
    elif arguments.stage == "evaluate":
        run_map_stage(config_path, arguments.stage, evaluation_config_path=(PROJECT_ROOT/arguments.evaluation_config).resolve())
    else:
        run_map_stage(config_path, arguments.stage)

def run_map_stage(config_path, stage, branch=None, evaluation_config_path=None):
    """Verify predecessor fingerprints and execute the selected reconstruction stage."""
    from joint_map.registration import check_feature_identity, run_initialization
    config = json.loads(config_path.read_text())
    check_config(config)
    check_environment()
    output_dir = PROJECT_ROOT / config["output_dir"]
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {"baseline": "check", "initialize": "baseline", "validate_priors": "initialize", "register": "validate_priors", "joint": "register", "dense": "register" if branch == "baseline" else "joint", "update": "dense", "export": "update", "evaluate": "export"}[stage]
    predecessors = [record for record in manifest["stages"] if record["stage"] == required and (stage not in ["update", "export"] or record["branch"] == branch)]
    if not predecessors or predecessors[-1]["status"] != "complete":
        raise RuntimeError(f"Run {required} successfully before {stage}")
    if stage == "evaluate":
        for name in ["baseline", "joint"]:
            records = [record for record in manifest["stages"] if record["stage"] == "export" and record["branch"] == name]
            if not records or records[-1]["status"] != "complete":
                raise RuntimeError("Complete both branch exports before evaluation")
    if manifest["config"]["sha256"] != get_file_fingerprint(config_path)["sha256"]:
        raise ValueError("Configuration changed; repeat check before continuing")
    check_report = json.loads((output_dir / "logs/check.log").read_text())
    if check_report["status"] != "complete":
        raise ValueError("Input check report is incomplete")
    for saved in check_report["input_fingerprints"].values():
        if get_file_fingerprint(Path(saved["path"])) != saved:
            raise ValueError(f"Input fingerprint changed: {saved['path']}")
    if json.loads((output_dir / "shared/origin.json").read_text()) != check_report["origin"]:
        raise ValueError("Map origin changed after input checks")
    if stage == "initialize":
        baseline_report = json.loads((output_dir / "logs/baseline.log").read_text())
        if baseline_report["status"] != "complete":
            raise ValueError("Baseline confirmation report is incomplete")
    if stage in ["register", "joint", "dense", "update", "export", "evaluate"]:
        from joint_map.optimization import require_confirmed_prior
        require_confirmed_prior(json.loads((output_dir / "shared/prior_model.json").read_text()))
    for previous in manifest["stages"]:
        if previous["status"] == "running":
            previous.update(status="failed", error=f"Previous {previous['stage']} invocation did not complete")
    record = {"stage": stage, "branch": branch if stage in ["dense", "update", "export"] else "joint" if stage == "joint" else None, "status": "running", "artifacts": {}, "counts": {}, "error": None}
    manifest["stages"].append(record)
    manifest["status"] = "running"
    if stage == "update":
        manifest.setdefault("update_status_by_branch", {})[branch] = None
        manifest["dsm_update_goal_complete"] = False
    write_json(manifest_path, manifest)
    if stage == "baseline":
        model = pycolmap.Reconstruction(str(PROJECT_ROOT / config["native_sfm_dir"]))
        identity = check_feature_identity(model, PROJECT_ROOT / config["frontend"]["features_path"], PROJECT_ROOT / config["frontend"]["adjusted_features_path"])
        report = {"schema_version": 3, "stage": stage, "status": "complete", "registered_images": model.num_reg_images(), "sparse_points": model.num_points3D(), "identity": identity, "coordinate_system": "arbitrary_local_sfm", "map_registration": "pending_initialize_and_confirmed_priors", "gt_inputs_read": False, "error": None}
        write_json(output_dir / "logs/baseline.log", report)
    elif stage == "initialize":
        report = run_initialization(config, PROJECT_ROOT, output_dir, write_json)
    elif stage == "validate_priors":
        from joint_map.optimization import run_prior_validation
        report = run_prior_validation(config, PROJECT_ROOT, output_dir, write_json)
    elif stage == "register":
        from joint_map.optimization import run_confirmed_registration
        report = run_confirmed_registration(config, PROJECT_ROOT, output_dir, write_json)
    elif stage == "joint":
        from joint_map.joint import run_joint_optimization
        report = run_joint_optimization(config, PROJECT_ROOT, output_dir, write_json)
    elif stage == "dense":
        from joint_map.dense import run_dense_preparation
        report = run_dense_preparation(config, PROJECT_ROOT, output_dir, branch, write_json)
    elif stage == "update":
        from joint_map.dsm import run_dsm_update
        report = run_dsm_update(config, PROJECT_ROOT, output_dir, branch, write_json)
    elif stage == "export":
        from joint_map.export import run_geographic_export
        report = run_geographic_export(config, PROJECT_ROOT, output_dir, branch, write_json)
    else:
        from joint_map.evaluation import run_evaluation
        report = run_evaluation(config, evaluation_config_path, config_path, PROJECT_ROOT, output_dir, write_json)
    log_name = f"{stage}_{branch}" if stage in ["dense", "update", "export"] else stage
    record.update(status=report["status"], artifacts={"diagnostics": str(output_dir / f"logs/{log_name}.log")}, counts={"registered_images": report.get("registered_images"), "sparse_points": report.get("sparse_points"), "map_priors": report.get("accepted_track_count", 0)}, error=report["error"])
    if stage in ["joint", "dense", "update", "export", "evaluate"] and report["status"] == "complete":
        record["artifacts"].update(report["artifacts"])
    if stage in ["dense", "export"]:
        record["counts"]["dense_points"] = report.get("dense_points", 0)
    if stage == "export" and report["status"] == "complete":
        record.update(source_update_status=report["source_update_status"], dsm_update_goal_complete=report["dsm_update_goal_complete"])
    if stage == "evaluate" and report["status"] == "complete":
        record.update(primary_acceptance_passed=report["primary_acceptance_passed"], delivery_complete=report["delivery_complete"])
        record["artifacts"]["report_json"] = str(output_dir/"evaluation/report.json")
        record["counts"].update(reference_count=report["reference_count"], acceptance_reference_count=report["acceptance_reference_count"])
        manifest.update(accuracy_acceptance_passed=report["primary_acceptance_passed"], delivery_complete=report["delivery_complete"])
    if stage == "update" and report["status"] == "complete":
        record.update(update_status=report["update_status"], dsm_update_goal_complete=report["dsm_update_goal_complete"])
        record["counts"].update(updated_cell_count=report["updated_cell_count"], updated_area_m2=report["updated_area_m2"])
        manifest.setdefault("update_status_by_branch", {})[branch] = report["update_status"]
        manifest["dsm_update_goal_complete"] = all(manifest["update_status_by_branch"].get(name) == "updated" for name in ["baseline", "joint"])
    manifest["status"] = report["status"]
    write_json(manifest_path, manifest)
    print(f"{stage}: {report['status']} | images={report.get('registered_images')} | map_tracks={report.get('accepted_track_count', 0)}", flush=True)
    if report["status"] != "complete":
        raise RuntimeError(report["error"])

if __name__ == "__main__":
    run_main()
