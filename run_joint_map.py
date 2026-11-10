# 用途：调度分阶段或完整地图流水线，核验交接并保留独立精度验收状态。
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
PIPELINE_STAGES = [("check", None), ("baseline", None), ("initialize", None), ("validate_priors", None), ("register", None), ("joint", "joint"), ("dense", "baseline"), ("dense", "joint"), ("update", "baseline"), ("update", "joint"), ("export", "baseline"), ("export", "joint"), ("evaluate", None)]

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
    if (output_dir/"evaluation/baseline_metrics.json").is_file():
        raise ValueError("GT baseline is frozen; use --stage all to verify this experiment instead of rebuilding check")
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
    parser.add_argument("--stage", choices=["check", "baseline", "initialize", "validate_priors", "register", "joint", "dense", "update", "export", "evaluate", "all"], required=True)
    parser.add_argument("--branch", choices=["baseline", "joint"])
    parser.add_argument("--evaluation-config", type=Path)
    arguments = parser.parse_args()
    if (arguments.stage in ["dense", "update", "export"]) != (arguments.branch is not None):
        parser.error("--branch is required only for dense/update/export")
    if (arguments.stage in ["evaluate", "all"]) != (arguments.evaluation_config is not None):
        parser.error("--evaluation-config is required only for evaluate/all")
    config_path = (PROJECT_ROOT / arguments.config).resolve()
    if arguments.stage == "check":
        run_check(config_path)
    elif arguments.stage in ["dense", "update", "export"]:
        run_map_stage(config_path, arguments.stage, arguments.branch)
    elif arguments.stage == "evaluate":
        run_map_stage(config_path, arguments.stage, evaluation_config_path=(PROJECT_ROOT/arguments.evaluation_config).resolve())
    elif arguments.stage == "all":
        run_all(config_path, (PROJECT_ROOT/arguments.evaluation_config).resolve())
    else:
        run_map_stage(config_path, arguments.stage)

def run_map_stage(config_path, stage, branch=None, evaluation_config_path=None):
    """Verify predecessor fingerprints and execute the selected reconstruction stage."""
    from joint_map.registration import check_feature_identity, run_initialization
    config = json.loads(config_path.read_text())
    check_config(config)
    check_environment()
    output_dir = PROJECT_ROOT / config["output_dir"]
    if stage != "evaluate" and (output_dir/"evaluation/baseline_metrics.json").is_file():
        raise ValueError("GT baseline is frozen; use --stage all for verification, not reconstruction-stage reruns")
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

def get_latest_stage(manifest, stage, branch):
    """Return only the latest attempt for the exact stage and branch, preserving earlier failure history."""
    records = [row for row in manifest["stages"] if row["stage"] == stage and row.get("branch") == branch]
    return records[-1] if records else None

def audit_pipeline(config_path, evaluation_config_path, output_dir):
    """Verify all completed handoffs and current artifacts without rerunning reconstruction or GT scoring."""
    from joint_map.joint import load_confirmed_handoff, get_formal_fingerprint
    from joint_map.optimization import get_observations
    from joint_map.export import read_geographic_cloud, read_geographic_poses
    from joint_map.evaluation import compare_primary_metrics
    config = json.loads(config_path.read_text())
    manifest = json.loads((output_dir/"run_manifest.json").read_text())
    verified = {}
    def check_file(path, expected=None):
        """Hash each artifact once and compare every recorded handoff digest against the same bytes."""
        path = Path(path).resolve()
        key = str(path)
        if key not in verified:
            verified[key] = get_file_fingerprint(path)["sha256"]
        if expected is not None and verified[key] != expected:
            raise ValueError("Pipeline artifact fingerprint changed: "+key)
        return verified[key]
    if manifest["config"]["sha256"] != check_file(config_path):
        raise ValueError("Reconstruction configuration changed after input checking")
    stages = []
    for stage, branch in PIPELINE_STAGES:
        row = get_latest_stage(manifest, stage, branch)
        if row is None or row["status"] != "complete":
            raise ValueError(f"Incomplete pipeline stage: {stage}/{branch}")
        for path in row["artifacts"].values():
            if not Path(path).exists():
                raise FileNotFoundError(path)
        stages.append({"stage": stage, "branch": branch, "status": row["status"], "counts": row["counts"]})
    initial = json.loads((output_dir/"logs/check.log").read_text())
    for saved in initial["input_fingerprints"].values():
        if get_file_fingerprint(Path(saved["path"])) != saved:
            raise ValueError("Original pipeline input changed: "+saved["path"])
    if json.loads((output_dir/"shared/origin.json").read_text()) != initial["origin"]:
        raise ValueError("Shared ENU origin changed after checking")
    prior, registration, anchors, handoff = load_confirmed_handoff(output_dir)
    joint = json.loads((output_dir/"joint/diagnostics.json").read_text())
    if joint["status"] != "complete" or not joint["geometry"]["converged"] or not joint["feature"]["converged"] or not joint["final_factor_identity"]["passed"]:
        raise ValueError("Joint solving or shared-parameter verification is incomplete")
    branches = {}
    with rasterio.open(PROJECT_ROOT/config["dsm_path"]) as source:
        original, profile = source.read(1), source.profile
    for branch in ["baseline", "joint"]:
        model = pycolmap.Reconstruction(str(output_dir/branch/"model"))
        if model.num_reg_images() != 6 or sorted(image.name for image in model.images.values()) != config["image_names"]:
            raise ValueError("Branch camera identities are incomplete")
        for camera in model.cameras.values():
            if camera.model_name != config["camera"]["model"] or (camera.width, camera.height) != (config["camera"]["width"], config["camera"]["height"]):
                raise ValueError("Branch camera calibration dimensions changed")
            np.testing.assert_array_equal(camera.params, config["camera"]["params"])
        dense = json.loads((output_dir/branch/"dense/summary.json").read_text())
        update = json.loads((output_dir/branch/"map/summary.json").read_text())
        export = json.loads((output_dir/branch/"export/summary.json").read_text())
        if any(row["status"] != "complete" for row in [dense, update, export]):
            raise ValueError("Branch dense/update/export execution is incomplete")
        for name, digest in dense["source_model_sha256"].items():
            check_file(output_dir/branch/"model"/name, digest)
        for name, digest in dense["output_sha256"].items():
            check_file(output_dir/branch/"dense"/name, digest)
        check_file(output_dir/"shared/registration.json", dense["input_identity"]["registration_sha256"])
        for name, digest in update["output_sha256"].items():
            check_file(update["artifacts"][name], digest)
        for path, digest in update["source_view_sha256"].items():
            check_file(path, digest)
        for name, digest in export["output_sha256"].items():
            check_file(export["artifacts"][name], digest)
        for path, digest in export["input_sha256"].items():
            check_file(path, digest)
        if len(dense["depth_maps"]) != 6 or any(row["valid_depth_pixels"] <= 0 for row in dense["depth_maps"]):
            raise ValueError("Branch geometric depths are incomplete")
        masks = {}
        for name in ["coverage_mask", "overlap_mask", "reliability_mask", "update_mask"]:
            with rasterio.open(output_dir/branch/"map"/(name+".tif")) as source:
                if source.shape != original.shape or source.transform != profile["transform"] or source.crs != profile["crs"] or source.nodata != 255 or source.dtypes != ("uint8",):
                    raise ValueError("Evidence mask metadata changed")
                values = source.read(1)
                if not np.isin(values, [0, 1]).all():
                    raise ValueError("Evidence mask is not binary")
                masks[name] = values.astype(bool)
        selected = masks["update_mask"]
        np.testing.assert_array_equal(selected, masks["coverage_mask"] & masks["overlap_mask"] & masks["reliability_mask"])
        with rasterio.open(output_dir/branch/"map/updated_dsm.tif") as source:
            if source.shape != original.shape or source.transform != profile["transform"] or source.crs != profile["crs"] or source.nodata != profile["nodata"] or source.dtypes != ("float32",):
                raise ValueError("Updated DSM metadata changed")
            np.testing.assert_array_equal(source.read(1)[~selected], original[~selected])
        if int(selected.sum()) != update["updated_cell_count"] or (bool(selected.any()) != (update["update_status"] == "updated")):
            raise ValueError("Actual update mask and update delivery status disagree")
        if not selected.any() and update["updated_area_m2"] != 0:
            raise ValueError("Empty update has nonzero area")
        read_geographic_poses(output_dir/branch/"export/poses_wgs84.txt", config["image_names"])
        for kind, expected in [("sparse", model.num_points3D()), ("dense", dense["dense_points"])]:
            if len(read_geographic_cloud(output_dir/branch/"export"/(kind+"_wgs84.ply"))) != expected:
                raise ValueError("Geographic point count differs from the branch source")
        with np.load(update["artifacts"]["sparse_evidence"]) as lineage:
            np.testing.assert_array_equal(lineage["shared_point_ids"], anchors["point_ids"])
            np.testing.assert_array_equal(lineage["optimized_xyz_enu_m"], [model.points3D[int(index)].xyz for index in anchors["point_ids"]])
            if lineage["dense_support_status"].item() != "checked" or len(lineage["dense_support_offsets"]) != len(anchors["point_ids"])+1:
                raise ValueError("Sparse-to-dense evidence chain is incomplete")
        if branch == "joint" and get_formal_fingerprint(model) != joint["serialization_check"]["persisted_model_fingerprint"]:
            raise ValueError("Current joint model differs from the solved persisted model")
        branches[branch] = {"registered_images": 6, "sparse_points": model.num_points3D(), "observations": len(get_observations(model)), "dense_points": dense["dense_points"], "update_status": update["update_status"], "updated_cell_count": update["updated_cell_count"], "updated_area_m2": update["updated_area_m2"]}
    evaluation = json.loads((output_dir/"evaluation/report.json").read_text())
    if evaluation["status"] != "complete" or str(evaluation_config_path) not in evaluation["input_sha256"]:
        raise ValueError("Evaluation is incomplete or belongs to another evaluation configuration")
    for path, digest in evaluation["input_sha256"].items():
        check_file(path, digest)
    for name, digest in evaluation["output_sha256"].items():
        check_file(evaluation["artifacts"][name], digest)
    reference = json.loads((output_dir/"evaluation/reference_manifest.json").read_text())
    baseline = json.loads((output_dir/"evaluation/baseline_metrics.json").read_text())
    check_file(output_dir/"evaluation/baseline_metrics.json", evaluation["baseline_metrics_sha256"])
    for name, digest in baseline["artifact_sha256"].items():
        check_file(output_dir/"evaluation"/name, digest)
    for branch in ["baseline", "joint", "original"]:
        errors = np.load(output_dir/"evaluation"/f"dsm_errors_{branch}.npy", mmap_mode="r")
        if errors.shape != (evaluation["reference_count"],) or errors.dtype != np.float64:
            raise ValueError("Full-domain error arrays changed shape or precision")
    if reference["reference_count"] != evaluation["reference_count"] or reference["acceptance_reference_count"] != evaluation["acceptance_reference_count"]:
        raise ValueError("Fixed reference cardinality differs between reports")
    comparisons, passed = compare_primary_metrics(evaluation["baseline_metrics"], evaluation["joint_metrics"], evaluation["invalid_regression_count"])
    if comparisons != evaluation["primary_comparisons"] or passed != evaluation["primary_acceptance_passed"]:
        raise ValueError("Primary metric decisions disagree with acceptance status")
    delivery = passed and all(row["update_status"] == "updated" for row in branches.values())
    if delivery != evaluation["delivery_complete"] or manifest.get("accuracy_acceptance_passed") != passed or manifest.get("delivery_complete") != delivery:
        raise ValueError("Execution, accuracy and delivery flags disagree")
    documentation_sha = check_file(PROJECT_ROOT/"USAGE_JOINT_MAP.md")
    integration_code_sha = check_file(Path(__file__))
    return {"schema_version": 3, "status": "complete", "implementation_complete": True, "primary_acceptance_passed": passed, "delivery_complete": delivery, "accepted_map_priors": len(anchors["point_ids"]), "confirmed_map_weight": prior["lambda_map"], "user_confirmation": prior["user_confirmation"], "validation_run_dir": handoff["validation_run_dir"], "stages": stages, "branches": branches, "reference_count": evaluation["reference_count"], "acceptance_reference_count": evaluation["acceptance_reference_count"], "failed_primary_metrics": [key for key, row in evaluation["primary_comparisons"].items() if not row["passed"]], "documentation_sha256": documentation_sha, "integration_code_sha256": integration_code_sha, "verified_file_sha256": verified, "gt_scoring_performed_in_audit": False, "previous_failed_attempts_retained": sum(row["status"] == "failed" for row in manifest["stages"])}

def run_all(config_path, evaluation_config_path):
    """Resume the ordered pipeline, reuse completed stages and prohibit rebuilding a GT-frozen experiment."""
    config = json.loads(config_path.read_text())
    check_config(config)
    check_environment()
    output_dir = PROJECT_ROOT/config["output_dir"]
    manifest_path = output_dir/"run_manifest.json"
    log_path = output_dir/"logs/all.log"
    report = {"schema_version": 3, "stage": "all", "status": "running", "error": None, "reused_stages": [], "executed_stages": []}
    write_json(log_path, report)
    for stage, branch in PIPELINE_STAGES:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"stages": []}
        previous = get_latest_stage(manifest, stage, branch)
        if previous is not None and previous["status"] == "complete":
            report["reused_stages"].append({"stage": stage, "branch": branch})
            continue
        if stage != "evaluate" and (output_dir/"evaluation/baseline_metrics.json").exists():
            report.update(status="failed", error=f"Frozen GT baseline forbids rebuilding {stage}/{branch}; preserve this experiment")
            write_json(log_path, report)
            raise ValueError(report["error"])
        if stage == "register":
            prior = json.loads((output_dir/"shared/prior_model.json").read_text())
            if prior["status"] != "confirmed":
                report.update(status="failed", error="Confirm an eligible prior candidate explicitly before register/joint; no weight is selected automatically")
                write_json(log_path, report)
                raise ValueError(report["error"])
        if stage == "check":
            run_check(config_path)
        elif stage in ["dense", "update", "export"]:
            run_map_stage(config_path, stage, branch)
        elif stage == "evaluate":
            run_map_stage(config_path, stage, evaluation_config_path=evaluation_config_path)
        else:
            run_map_stage(config_path, stage)
        report["executed_stages"].append({"stage": stage, "branch": branch})
        write_json(log_path, report)
    audit = audit_pipeline(config_path, evaluation_config_path, output_dir)
    report.update(audit, stage="all", error=None)
    write_json(output_dir/"integration.json", report)
    write_json(log_path, report)
    manifest = json.loads(manifest_path.read_text())
    manifest["stages"].append({"stage": "all", "branch": None, "status": "complete", "artifacts": {"integration": str(output_dir/"integration.json"), "diagnostics": str(log_path)}, "counts": {"verified_stages": len(audit["stages"]), "verified_files": len(audit["verified_file_sha256"])}, "primary_acceptance_passed": audit["primary_acceptance_passed"], "delivery_complete": audit["delivery_complete"], "error": None})
    manifest.update(status="complete", implementation_complete=True, accuracy_acceptance_passed=audit["primary_acceptance_passed"], delivery_complete=audit["delivery_complete"])
    write_json(manifest_path, manifest)
    print(f"all: complete | reused={len(report['reused_stages'])} | executed={len(report['executed_stages'])} | primary_acceptance={'passed' if audit['primary_acceptance_passed'] else 'failed'} | delivery_complete={audit['delivery_complete']}", flush=True)
    return report

if __name__ == "__main__":
    run_main()
