# 用途：复用冻结先验与原生保护因子，执行正式全观测联合优化并保存稀疏点身份链。
import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation
from joint_map.optimization import (
    audit_factor_identity, build_shared_problem, check_dense_cache,
    check_model_domain, compute_gauge_information, create_dense_feature_context,
    get_dense_provenance, get_error_statistics,
    get_model_fingerprint, get_observation_digest, get_observations, get_sha256,
    initialize_map_points, project_observations, read_frozen_candidate_inputs,
    require_confirmed_prior, restore_feature_domain, sample_feature_residuals,
    save_training_references, solve_shared_problem, write_validation_json,
)
from joint_map.geometry import transform_colmap_pose
from joint_map.registration import apply_sim3, check_camera_connectivity

def get_formal_fingerprint(model):
    """Fingerprint numeric state and canonical observation identities across model serialization."""
    result = get_model_fingerprint(model)
    result["observations"] = get_observation_digest(sorted(get_observations(model), key=lambda row: row[:3]))
    return result

def check_serialized_model(model, persisted):
    """Require exact round-trip state after the native writer and reader quaternion normalizations."""
    expected = copy.deepcopy(model)
    for image in expected.images.values():
        image.normalize_qvec()
        image.normalize_qvec()
    if get_formal_fingerprint(persisted) != get_formal_fingerprint(expected):
        raise ValueError("Persisted formal model differs from the expected native serialization")
    names = lambda current: [(index, image.name, image.camera_id) for index, image in sorted(current.images.items())]
    cameras = lambda current: [(index, camera.model_id, camera.width, camera.height) for index, camera in sorted(current.cameras.items())]
    if names(model) != names(persisted) or cameras(model) != cameras(persisted):
        raise ValueError("Persisted formal image or camera identities changed")
    return {"passed": True, "policy": "exact_native_write_read_normalization_v1", "quaternion_normalizations": 2, "max_quaternion_change": max(float(np.max(np.abs(image.qvec-persisted.images[index].qvec))) for index, image in model.images.items()), "persisted_model_fingerprint": get_formal_fingerprint(persisted)}

def check_registered_visual_model(visual, native, registration):
    """Verify that the baseline is exactly the registered original geometry with unchanged identities."""
    expected = copy.deepcopy(native)
    transform = (registration["scale"], np.asarray(registration["rotation"]), np.asarray(registration["translation_enu_m"]))
    for image in expected.images.values():
        rotation, translation = transform_colmap_pose(image.rotmat(), image.tvec, *transform)
        image.qvec = pycolmap.rotmat_to_qvec(rotation)
        image.tvec = translation
    for point in expected.points3D.values():
        point.xyz = apply_sim3(point.xyz, transform)
    names = [(index, image.name, image.camera_id) for index, image in sorted(visual.images.items())]
    expected_names = [(index, image.name, image.camera_id) for index, image in sorted(expected.images.items())]
    if names != expected_names or get_formal_fingerprint(visual) != get_formal_fingerprint(expected):
        raise ValueError("Registered visual model changed geometry, calibration or observation identities")
    for camera_id, camera in visual.cameras.items():
        original = native.cameras[camera_id]
        if (camera.model_id, camera.width, camera.height) != (original.model_id, original.width, original.height):
            raise ValueError("Registered visual camera model or image dimensions changed")

def load_confirmed_handoff(output_dir):
    """Verify the selected candidate, immutable training priors and completed common registration."""
    shared = output_dir/"shared"
    prior = json.loads((shared/"prior_model.json").read_text())
    require_confirmed_prior(prior)
    for field, filename in [("validation_sha256", "prior_validation.json"), ("map_priors_sha256", "map_priors.npz"), ("reference_sha256", "training_references.npz")]:
        if get_sha256(shared/filename) != prior[field]:
            raise ValueError("Confirmed module-three evidence changed")
    validation = json.loads((shared/"prior_validation.json").read_text())
    selected = [row for row in validation["candidates"] if row["candidate_id"] == prior["candidate_id"] and row["status"] == "passed" and row["information_scale"] == prior["lambda_map"]]
    if validation["status"] != "complete" or len(selected) != 1 or validation["policy_sha256"] != prior["policy_sha256"]:
        raise ValueError("Confirmed candidate does not match completed validation")
    read_frozen_candidate_inputs(Path(validation["run_dir"]))
    registration = json.loads((shared/"registration.json").read_text())
    if registration["status"] != "confirmed" or not registration["priors_confirmed"] or registration["prior_model"] != prior:
        raise ValueError("Common registration does not use the frozen confirmed prior")
    if json.loads((output_dir/"logs/register.log").read_text())["status"] != "complete":
        raise ValueError("Run register successfully before joint optimization")
    with np.load(shared/"map_priors.npz") as archive:
        anchors = {"point_ids": archive["shared_point_ids"].copy(), "means": archive["means_enu_m"].copy(), "prior_ids": archive["prior_ids"].copy(), "covariances": archive["covariances_m2"].copy()}
        training_prior_observations = len(archive["observation_image_ids"])
    expected = np.diag([prior["sigma_horizontal_m"]**2]*2+[prior["sigma_height_m"]**2])
    if len(set(anchors["point_ids"])) != len(anchors["point_ids"]) or not np.array_equal(anchors["covariances"], np.broadcast_to(expected, anchors["covariances"].shape)):
        raise ValueError("Frozen map IDs or covariances differ from the validated working model")
    return prior, registration, anchors, {"validation_run_dir": validation["run_dir"], "training_observation_count": validation["training_observation_count"], "training_prior_observations": training_prior_observations}

def prepare_formal_initialization(visual, anchors, registration):
    """Rebuild the registered map start from visual geometry without reusing an optimized result."""
    model = copy.deepcopy(visual)
    lineage = initialize_map_points(model, anchors)
    expected = registration["joint_map_initialization"]["lineage"]["model_fingerprint"]
    actual = get_model_fingerprint(model)
    if any(actual[field] != expected[field] for field in ["poses", "points", "cameras"]) or get_formal_fingerprint(model)["observations"] != get_formal_fingerprint(visual)["observations"]:
        raise ValueError("Formal map initialization differs from the registered handoff")
    return model, lineage

def solve_formal_problem(model, anchors, scales, context, report, save_checkpoint):
    """Run the frozen geometry and feature stages on the complete registered observation set."""
    observations = get_observations(model)
    observation_digest = get_observation_digest(observations)
    initial_points = {point_id: point.xyz.copy() for point_id, point in model.points3D.items()}
    initial_poses = {image_id: (image.qvec.copy(), image.tvec.copy(), image.rotmat().copy()) for image_id, image in model.images.items()}
    weight = scales["lambda_map"]
    covariance = np.diag([scales["sigma_horizontal_m"]**2]*2+[scales["sigma_height_m"]**2])
    report["geometry_initial_domain"] = check_model_domain(model, observations)
    report["gauge_checkpoints"] = [compute_gauge_information(anchors, covariance, np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]]), weight, "formal_map_initial")]
    if report["geometry_initial_domain"]["invalid_count"]:
        report.update(status="failed", error="Formal initialization has invalid camera depth or nonfinite projection")
        save_checkpoint(report)
        return report
    geometry = build_shared_problem(model, anchors, scales, weight)
    report["geometry_factor_identity"] = audit_factor_identity(geometry)
    geometry["iteration_recorder"].label = "formal geometry"
    report["geometry"] = {"executed": True, "status": "running"}
    save_checkpoint(report)
    result = solve_shared_problem(geometry)
    report["geometry"] = dict(result, executed=True, status="passed" if result["converged"] else "failed")
    save_checkpoint(report)
    if not result["converged"]:
        report.update(status="failed", error="Formal geometry did not converge within 300 iterations at the original tolerances")
        save_checkpoint(report)
        return report
    report["gauge_checkpoints"].append(compute_gauge_information(anchors, covariance, np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]]), weight, "formal_after_geometry"))
    del geometry
    report["feature_start_domain"] = check_model_domain(model, observations, context)
    if report["feature_start_domain"]["invalid_count"]:
        report["restoration"] = restore_feature_domain(model, anchors, scales, weight, context, "formal")
        save_checkpoint(report)
        if not report["restoration"]["restored"]:
            report.update(status="failed", error="Formal feature feasibility restoration failed within the fixed budgets")
            save_checkpoint(report)
            return report
        report["gauge_checkpoints"].append(compute_gauge_information(anchors, covariance, np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]]), weight, "formal_after_restoration"))
    feature = build_shared_problem(model, anchors, scales, weight, context=context)
    report["initial_factor_identity"] = audit_factor_identity(feature)
    if feature["audit"]["restore_blocks"]:
        raise ValueError("Temporary restoration factors remain in the final objective")
    feature["iteration_recorder"].label = "formal feature"
    report["feature"] = {"executed": True, "status": "running"}
    save_checkpoint(report)
    result = solve_shared_problem(feature)
    report["feature"] = dict(result, executed=True, status="passed" if result["converged"] else "failed")
    save_checkpoint(report)
    if not result["converged"]:
        report.update(status="failed", error="Formal feature problem did not converge within 100 iterations at the original tolerances")
        save_checkpoint(report)
        return report
    report["final_domain"] = check_model_domain(model, observations, context)
    if report["final_domain"]["invalid_count"]:
        report.update(status="failed", error="Formal final predictions do not all satisfy the full feature domain")
        save_checkpoint(report)
        return report
    if get_observation_digest(get_observations(model)) != observation_digest:
        raise ValueError("Formal solving changed the complete observation identities or measured pixels")
    report["final_factor_identity"] = audit_factor_identity(feature)
    points = np.array([model.points3D[int(index)].xyz for index in anchors["point_ids"]])
    final_gauge = compute_gauge_information(anchors, covariance, points, weight, "formal_final_joint")
    report["gauge_checkpoints"].append(final_gauge)
    if not final_gauge["passed"]:
        report.update(status="failed", error="Formal final map information does not constrain all seven similarity freedoms")
        save_checkpoint(report)
        return report
    descriptors, valid = sample_feature_residuals(model, observations, context)
    if not valid.all():
        raise ValueError("Independent final feature sampling disagrees with domain verification")
    report["reprojection_px"] = get_error_statistics(np.linalg.norm(project_observations(model, observations)-np.array([row[3] for row in observations]), axis=1))
    report["feature_norm"] = get_error_statistics(np.linalg.norm(descriptors, axis=1))
    report["map_residual_m"] = get_error_statistics(np.linalg.norm(points-anchors["means"], axis=1))
    report["all_xyz_displacement_m"] = get_error_statistics(np.array([np.linalg.norm(point.xyz-initial_points[point_id]) for point_id, point in model.points3D.items()]))
    report["pose_changes"] = []
    for image_id, image in model.images.items():
        qvec, tvec, rotation = initial_poses[image_id]
        center_before, center_after = -rotation.T@tvec, -image.rotmat().T@image.tvec
        report["pose_changes"].append({"image_id": image_id, "image_name": image.name, "initial_qvec": qvec.tolist(), "initial_tvec": tvec.tolist(), "optimized_qvec": image.qvec.tolist(), "optimized_tvec": image.tvec.tolist(), "camera_center_displacement_m": float(np.linalg.norm(center_after-center_before)), "rotation_change_deg": float(np.rad2deg(Rotation.from_matrix(image.rotmat()@rotation.T).magnitude()))})
    report["trial_evaluations"] = {"calls": sum(cost.calls for cost in feature["costs"] if hasattr(cost, "calls")), "rejected_before_sampling": sum(cost.rejected for cost in feature["costs"] if hasattr(cost, "rejected"))}
    report.update(optimization_complete=True, error=None, final_model_fingerprint=get_formal_fingerprint(model))
    save_checkpoint(report)
    return report

def save_sparse_lineage(path, visual, model, anchors):
    """Save sparse identity and movement while explicitly leaving dense and update evidence pending."""
    ids = np.asarray(anchors["point_ids"], dtype=np.int64)
    image_ids, point2D_ids, prior_indices = [], [], []
    for index, point_id in enumerate(ids):
        for entry in model.points3D[int(point_id)].track.elements:
            image_ids.append(entry.image_id)
            point2D_ids.append(entry.point2D_idx)
            prior_indices.append(index)
    np.savez_compressed(path, schema_version=np.array(3), stage=np.array("joint"), track_ids=ids, shared_point_ids=ids, prior_ids=anchors["prior_ids"], registered_sfm_xyz_enu_m=np.array([visual.points3D[int(index)].xyz for index in ids]), map_xyz_enu_m=anchors["means"], initialization_source=np.full(len(ids), "map_mu"), initial_xyz_enu_m=anchors["means"], optimized_xyz_enu_m=np.array([model.points3D[int(index)].xyz for index in ids]), observation_image_ids=np.array(image_ids, dtype=np.int64), observation_point2D_ids=np.array(point2D_ids, dtype=np.int64), observation_prior_indices=np.array(prior_indices, dtype=np.int64), old_cells=np.empty((0, 2), dtype=np.int64), optimized_cells=np.empty((0, 2), dtype=np.int64), dense_agreement_residuals=np.empty(0), dense_support_offsets=np.zeros(len(ids)+1, dtype=np.int64), dense_support_indices=np.empty(0, dtype=np.int64), update_cell_offsets=np.zeros(len(ids)+1, dtype=np.int64), update_cell_indices=np.empty(0, dtype=np.int64), dense_support_status=np.array("not_run"), cell_mapping_status=np.array("pending_dense_update"))

def run_joint_optimization_in_process(config, root, output_dir, run_dir):
    """Build formal full-observation references and publish only a completely verified joint model."""
    joint = output_dir/"joint"
    report = {"schema_version": 3, "stage": "joint", "branch": "joint", "status": "running", "error": None, "run_dir": str(run_dir), "gt_inputs_read": False, "geometry": {"executed": False, "status": "not_run"}, "feature": {"executed": False, "status": "not_run"}}
    def save_checkpoint(current):
        """Persist this run and the canonical diagnostics without rewriting validated module-three evidence."""
        for path in [run_dir/"diagnostics.json", joint/"diagnostics.json", output_dir/"logs/joint.log"]:
            write_validation_json(path, current)
    save_checkpoint(report)
    prior, registration, anchors, handoff = load_confirmed_handoff(output_dir)
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < 8*1024**3 or shutil.disk_usage(joint).free < 512*1024**2:
        raise RuntimeError("Insufficient resources for full-resolution formal optimization")
    visual = pycolmap.Reconstruction(str(output_dir/"baseline/model"))
    check_registered_visual_model(visual, pycolmap.Reconstruction(str(root/config["native_sfm_dir"])), registration)
    model, lineage = prepare_formal_initialization(visual, anchors, registration)
    observations = get_observations(model)
    reached, connected = check_camera_connectivity(model, anchors["point_ids"])
    if not connected or model.num_reg_images() != len(config["image_names"]):
        raise ValueError("Formal camera and point graph is not completely map-connected")
    report.update(registered_images=model.num_reg_images(), sparse_points=model.num_points3D(), accepted_track_count=len(anchors["point_ids"]), observation_count=len(observations), restored_image_observation_count=len(observations)-handoff["training_observation_count"], connected_image_ids=reached, prior_model=prior, map_prior_scope="training_frozen", map_initialization=lineage, visual_reference_model_fingerprint=get_formal_fingerprint(visual), resources={"memory_available_bytes": available, "disk_free_bytes": shutil.disk_usage(joint).free}, solver_policy={"geometry_max_iterations": 300, "feature_max_iterations": 100, "function_tolerance": 1e-6, "gradient_tolerance": 1e-10, "parameter_tolerance": 1e-8, "num_threads": 1, "use_inner_iterations": False}, implementation_sha256=get_sha256(__file__))
    (run_dir/"initial_model").mkdir()
    model.write(str(run_dir/"initial_model"))
    dense_path = output_dir/"shared/holdout_dense_features.h5"
    report["cache"] = {"path": str(dense_path), "sha256": get_sha256(dense_path), "images": check_dense_cache(dense_path, get_dense_provenance(config, root))}
    context = create_dense_feature_context(visual, dense_path)
    provenance = save_training_references(visual, [], context, run_dir/"full_observation_references.npz", root)
    provenance["source"] = "native ReferenceExtractor on the registered full-observation visual model"
    provenance["role"] = "formal_full_observation_reference; validation holdout is restored only for formal solving"
    provenance["full_observations_sha256"] = provenance.pop("training_observations_sha256")
    provenance["reference_policy"]["sampling_model"] = "registered_full_visual_model_before_map_assignment"
    report["reference_provenance"] = provenance
    save_checkpoint(report)
    solve_formal_problem(model, anchors, prior, context, report, save_checkpoint)
    if not report.get("optimization_complete", False):
        return report
    if get_formal_fingerprint(visual) != report["visual_reference_model_fingerprint"]:
        raise ValueError("Formal optimization mutated its frozen visual reference model")
    load_confirmed_handoff(output_dir)
    if get_sha256(dense_path) != report["cache"]["sha256"]:
        raise ValueError("Immutable dense feature cache changed during formal optimization")
    (run_dir/"final_model").mkdir()
    model.write(str(run_dir/"final_model"))
    persisted = pycolmap.Reconstruction(str(run_dir/"final_model"))
    report["serialization_check"] = check_serialized_model(model, persisted)
    report["persisted_final_domain"] = check_model_domain(persisted, get_observations(persisted), context)
    if report["persisted_final_domain"]["invalid_count"]:
        raise ValueError("Persisted formal predictions do not satisfy the full feature domain")
    save_sparse_lineage(run_dir/"map_point_lineage.npz", visual, model, anchors)
    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        shutil.copy2(run_dir/"final_model"/name, joint/"model"/name)
    shutil.copy2(run_dir/"map_point_lineage.npz", joint/"map_point_lineage.npz")
    report["artifacts"] = {"model_dir": str(joint/"model"), "lineage": str(joint/"map_point_lineage.npz"), "reference_path": str(run_dir/"full_observation_references.npz")}
    report["output_model_sha256"] = {name: get_sha256(joint/"model"/name) for name in ["cameras.bin", "images.bin", "points3D.bin"]}
    report.update(status="complete", error=None)
    save_checkpoint(report)
    print(f"formal joint complete: images={model.num_reg_images()}, points={model.num_points3D()}, observations={len(observations)}, map_priors={len(anchors['point_ids'])}", flush=True)
    return report

def run_joint_optimization(config, root, output_dir, write_json):
    """Contain implementation failures in an isolated process while retaining its last checkpoint."""
    run_dir = output_dir/"joint/runs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    path = output_dir/"joint/diagnostics.json"
    report = {"schema_version": 3, "stage": "joint", "branch": "joint", "status": "running", "run_dir": str(run_dir), "error": None}
    write_validation_json(path, report)
    write_validation_json(run_dir/"diagnostics.json", report)
    code = "import json,sys; from pathlib import Path; from joint_map.joint import run_joint_optimization_in_process; run_joint_optimization_in_process(json.loads(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]))"
    process = subprocess.run([sys.executable, "-B", "-c", code, json.dumps(config), str(root), str(output_dir), str(run_dir)], cwd=root)
    report = json.loads(path.read_text())
    if process.returncode != 0:
        report.update(status="failed", error="Formal joint implementation exited unexpectedly; see traceback and last checkpoint", exit_code=process.returncode)
        write_json(path, report)
        write_json(output_dir/"logs/joint.log", report)
        if "run_dir" in report:
            write_json(Path(report["run_dir"])/"diagnostics.json", report)
    return report
