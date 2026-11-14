# 用途：隔离读取 GT，冻结全像素基线参考域，报告八项主验收及 DSM 分区诊断。
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import rasterio
from PIL import Image
from joint_map.dsm import compute_updated_area, get_grid_cells
from joint_map.export import read_geographic_poses
from joint_map.geometry import backproject_pixels, convert_enu_to_geodetic, convert_geodetic_to_enu, convert_pose_to_colmap, sample_dsm
from joint_map.optimization import get_sha256

REFERENCE_DTYPE = np.dtype([("image_index", "u1"), ("pixel_index", "<u4"), ("longitude", "<f8"), ("latitude", "<f8"), ("height", "<f8"), ("cell_index", "<i4")])
METRIC_NAMES = ["translation_median_m", "translation_std_m", "rotation_median_deg", "rotation_std_deg", "dsm_median_m", "dsm_std_m", "recall_1m_1deg", "recall_3m_3deg"]

def check_evaluation_config(config):
    """Require the approved complete-pixel protocol and the explicit six-pair depth override."""
    expected = {"schema_version": 3, "depth_suffix": ".npy", "depth_type": "camera_z_m", "height_datum": "WGS84_ellipsoidal", "reference_sampling": "all_valid_pixels", "dsm_reference_region": "fixed_gt_domain_with_update_partitions", "acceptance_dsm_region": "fixed_reference_domain", "partition_metrics_role": "diagnostic", "std_ddof": 0, "valid_depth_rule": "finite_positive_six_pairs_v1"}
    if any(config.get(key) != value for key, value in expected.items()) or not config.get("depth_rule_confirmation"):
        raise ValueError("Evaluation protocol differs from the approved fixed six-pair rules")

def get_valid_depth(depth):
    """Apply only finite positive depth validity without a far or maximum-value exclusion."""
    return np.isfinite(depth) & (depth > 0)

def load_evaluation_raster(path, profile=None):
    """Load a masked DSM and require the same original grid, CRS and NoData convention."""
    with rasterio.open(path) as source:
        if profile is not None and (source.shape != (profile["height"], profile["width"]) or source.transform != profile["transform"] or source.crs != profile["crs"] or source.nodata != profile["nodata"]):
            raise ValueError("Evaluation DSM differs from the original raster grid or NoData convention")
        return source.read(1, masked=True).astype(np.float64), source.profile

def load_update_mask(path, profile):
    """Read a binary update mask on the exact original grid without using errors to define regions."""
    with rasterio.open(path) as source:
        values = source.read(1)
        if source.shape != (profile["height"], profile["width"]) or source.transform != profile["transform"] or source.crs != profile["crs"] or not np.isin(values, [0, 1]).all():
            raise ValueError("Evaluation update mask is not a binary original-grid mask")
    return values.astype(bool)

def summarize_error_values(values, selected):
    """Report exact float64 median and population std with explicit valid and invalid counts."""
    chosen = np.asarray(values)[selected]
    finite = np.isfinite(chosen)
    count = int(finite.sum())
    return {"reference_count": len(chosen), "valid_count": count, "invalid_count": int((~finite).sum()), "median": float(np.median(chosen[finite])) if count else None, "std": float(np.std(chosen[finite], ddof=0)) if count else None}

def summarize_pose_errors(translations, rotations):
    """Keep six-image recall denominators and withhold aggregate pose errors when any pose is invalid."""
    translations, rotations = np.asarray(translations), np.asarray(rotations)
    if translations.shape != (6,) or rotations.shape != (6,):
        raise ValueError("Pose evaluation requires the fixed denominator of six images")
    valid = np.isfinite(translations) & np.isfinite(rotations)
    complete = bool(valid.all())
    return {"translation_median_m": float(np.median(translations)) if complete else None, "translation_std_m": float(np.std(translations, ddof=0)) if complete else None, "rotation_median_deg": float(np.median(rotations)) if complete else None, "rotation_std_deg": float(np.std(rotations, ddof=0)) if complete else None, "recall_1m_1deg": float(np.count_nonzero(valid & (translations < 1) & (rotations < 1))/6), "recall_3m_3deg": float(np.count_nonzero(valid & (translations < 3) & (rotations < 3))/6), "valid_pose_count": int(valid.sum())}

def compute_pose_metrics(estimated, ground_truth, names, origin):
    """Compare centers in fixed ENU and rotations by relative matrix angle without GT alignment."""
    translations, rotations, rows = np.full(6, np.nan), np.full(6, np.nan), []
    for index, name in enumerate(names):
        if name in estimated and np.isfinite(estimated[name]).all():
            prediction, truth = estimated[name], ground_truth[name]
            translations[index] = np.linalg.norm(convert_geodetic_to_enu(prediction[:3], origin)-convert_geodetic_to_enu(truth[:3], origin))
            predicted_rotation, predicted_translation = convert_pose_to_colmap(prediction[:3], prediction[3:], origin)
            true_rotation, true_translation = convert_pose_to_colmap(truth[:3], truth[3:], origin)
            rotations[index] = np.rad2deg(np.arccos(np.clip((np.trace(true_rotation.T@predicted_rotation)-1)/2, -1., 1.)))
        valid = bool(np.isfinite(translations[index]) and np.isfinite(rotations[index]))
        rows.append({"image_name": name, "status": "valid" if valid else "missing_or_invalid", "translation_error_m": float(translations[index]) if valid else None, "rotation_error_deg": float(rotations[index]) if valid else None})
    return summarize_pose_errors(translations, rotations), rows

def get_partition_masks(baseline_updated, joint_updated, baseline_valid, joint_valid, original_valid, original_cell_nodata):
    """Define all diagnostic regions only from fixed cell masks and retain every validity flag."""
    return {"all": np.ones(len(baseline_updated), dtype=bool), "union": baseline_updated | joint_updated, "common": baseline_updated & joint_updated, "new": joint_updated & ~baseline_updated, "lost": baseline_updated & ~joint_updated, "D0": baseline_valid, "baseline_updated": baseline_updated, "joint_updated": joint_updated, "valid_baseline": baseline_valid, "valid_joint": joint_valid, "valid_original": original_valid, "original_cell_nodata": original_cell_nodata, "original_nodata_new_region": joint_updated & ~baseline_updated & original_cell_nodata, "original_nodata_new_fill": joint_updated & ~baseline_updated & original_cell_nodata & joint_valid}

def compare_primary_metrics(baseline, joint, invalid_regression_count):
    """Apply only the eight unrounded primary inequalities and required validity conditions."""
    comparisons = {}
    for name in METRIC_NAMES:
        first, second = baseline[name], joint[name]
        comparable = first is not None and second is not None
        passed = bool(comparable and (second >= first if name.startswith("recall_") else second <= first))
        comparisons[name] = {"baseline": first, "joint": second, "relation": ">=" if name.startswith("recall_") else "<=", "comparable": comparable, "passed": passed}
    passed = all(row["passed"] for row in comparisons.values()) and baseline["valid_pose_count"] == joint["valid_pose_count"] == 6 and invalid_regression_count == 0
    return comparisons, bool(passed)

def get_partition_statistics(errors, masks, references, profile):
    """Report per-method validity and strictly paired two-way and three-way partition comparisons."""
    result = {}
    for name in ["all", "union", "common", "new", "lost"]:
        selected = masks[name]
        paired = selected & masks["valid_baseline"] & masks["valid_joint"]
        triple = paired & masks["valid_original"]
        grid = np.zeros((profile["height"], profile["width"]), dtype=bool)
        grid.ravel()[references["cell_index"][selected]] = True
        individual = {method: summarize_error_values(values, selected) for method, values in errors.items()}
        paired_statistics = {method: summarize_error_values(errors[method], paired) for method in ["baseline", "joint"]}
        common_statistics = {method: summarize_error_values(values, triple) for method, values in errors.items()}
        degraded = any(paired_statistics["joint"][key] > paired_statistics["baseline"][key] for key in ["median", "std"]) if paired.any() else None
        result[name] = {"reference_count": int(selected.sum()), "cell_count": int(grid.sum()), "area_m2": compute_updated_area(grid, profile["transform"]), "per_method_valid_subset": individual, "paired_baseline_joint": paired_statistics, "paired_with_original": common_statistics, "paired_diagnostic_degradation": degraded, "unpaired_joint_valid_count": int((selected & masks["valid_joint"] & ~masks["valid_baseline"]).sum())}
    return result

def collect_evaluation_inputs(config, evaluation, config_path, evaluation_path, root, output_dir):
    """Verify completed branch artifacts and fingerprint the exact six image/depth pairs before scoring."""
    inputs, declared = {}, []
    updates = {}
    for branch in ["baseline", "joint"]:
        export = json.loads((output_dir/branch/"export/summary.json").read_text())
        update = json.loads((output_dir/branch/"map/summary.json").read_text())
        if export["status"] != "complete" or update["status"] != "complete":
            raise ValueError("Both branches must complete update and export before evaluation")
        for name, digest in export["output_sha256"].items():
            declared.append((export["artifacts"][name], digest))
        declared.extend(export["input_sha256"].items())
        declared.append((str(output_dir/branch/"export/summary.json"), get_sha256(output_dir/branch/"export/summary.json")))
        updates[branch] = update["update_status"]
    paths = [config_path, evaluation_path, Path(__file__), Path(__file__).with_name("geometry.py"), root/config["dsm_path"], output_dir/"shared/origin.json", root/evaluation["gt_pose_path"]]
    pairs = []
    for name in config["image_names"]:
        image_path = root/config["image_dir"]/name
        depth_path = root/evaluation["gt_depth_dir"]/(Path(name).stem+evaluation["depth_suffix"])
        with Image.open(image_path) as image:
            if image.size != (config["camera"]["width"], config["camera"]["height"]):
                raise ValueError("Evaluation image dimensions differ from fixed intrinsics")
        depth = np.load(depth_path, mmap_mode="r")
        if depth.shape != (config["camera"]["height"], config["camera"]["width"]) or depth.dtype != np.float32:
            raise ValueError("GT depth must be same-resolution float32 camera-z data")
        pairs.append({"image_name": name, "image_path": str(image_path), "image_sha256": get_sha256(image_path), "depth_path": str(depth_path), "depth_sha256": get_sha256(depth_path), "shape": list(depth.shape), "dtype": str(depth.dtype)})
        paths.extend([image_path, depth_path])
    declared.extend((str(path), get_sha256(path)) for path in paths)
    for path, digest in declared:
        if path in inputs and inputs[path] != digest:
            raise ValueError("Conflicting frozen evaluation source: "+path)
        inputs[path] = digest
    for path, digest in inputs.items():
        if get_sha256(path) != digest:
            raise ValueError("Evaluation source fingerprint changed: "+path)
    return inputs, pairs, updates

def stream_baseline_references(config, pairs, ground_truth, origin, original, baseline, baseline_mask, profile, folder):
    """Backproject every valid GT pixel in fixed row-major order and stream D and baseline-only errors."""
    width = config["camera"]["width"]
    params = config["camera"]["params"]
    intrinsics = np.array([[params[0], 0., params[2]], [0., params[1], params[3]], [0., 0., 1.]])
    digest, acceptance_digest = hashlib.sha256(), hashlib.sha256()
    per_image, count = [], 0
    reference_path = folder/"reference_points.bin"
    with reference_path.open("wb") as reference_stream, (folder/"baseline_errors.raw").open("wb") as baseline_stream, (folder/"original_errors.raw").open("wb") as original_stream, (folder/"baseline_update_flags.raw").open("wb") as mask_stream:
        for image_index, pair in enumerate(pairs):
            depth = np.load(pair["depth_path"], mmap_mode="r")
            pose = ground_truth[pair["image_name"]]
            rotation, translation = convert_pose_to_colmap(pose[:3], pose[3:], origin)
            valid_count, inside_count = 0, 0
            for first in range(0, len(depth), 64):
                block = np.asarray(depth[first:first+64])
                indices = np.flatnonzero(get_valid_depth(block))
                pixels = indices+first*width
                valid_count += len(pixels)
                if not len(pixels):
                    continue
                coordinates = np.column_stack([pixels % width, pixels // width])
                positions = backproject_pixels(coordinates, block.ravel()[indices], intrinsics, rotation, translation)
                geographic = convert_enu_to_geodetic(positions, origin)
                if not np.isfinite(geographic).all():
                    raise ValueError("Finite positive GT depths produced invalid geographic references")
                cells, inside = get_grid_cells(geographic, profile["transform"], original.shape)
                geographic, pixels, cells = geographic[inside], pixels[inside], cells[inside]
                size = len(pixels)
                if not size:
                    continue
                rows = np.empty(size, dtype=REFERENCE_DTYPE)
                rows["image_index"], rows["pixel_index"] = image_index, pixels
                rows["longitude"], rows["latitude"], rows["height"] = geographic.T
                rows["cell_index"] = cells[:, 0]*original.shape[1]+cells[:, 1]
                rows.tofile(reference_stream)
                original_values, original_valid = sample_dsm(original, profile["transform"], geographic)
                baseline_values, baseline_valid = sample_dsm(baseline, profile["transform"], geographic)
                np.where(original_valid, np.abs(original_values-geographic[:, 2]), np.nan).astype("<f8").tofile(original_stream)
                np.where(baseline_valid, np.abs(baseline_values-geographic[:, 2]), np.nan).astype("<f8").tofile(baseline_stream)
                baseline_mask[cells[:, 0], cells[:, 1]].astype(np.uint8).tofile(mask_stream)
                identities = np.column_stack([np.full(size, image_index, dtype="<u8"), pixels.astype("<u8")])
                digest.update(identities.tobytes())
                acceptance_digest.update(identities[baseline_valid].tobytes())
                count += size
                inside_count += size
            per_image.append({"image_name": pair["image_name"], "valid_depth_count": valid_count, "in_map_reference_count": inside_count, "outside_map_count": valid_count-inside_count})
            print(f"evaluate reference {pair['image_name']}: valid_depth={valid_count}, in_map={inside_count}", flush=True)
    if not count:
        raise ValueError("The fixed GT/map domain has no reference points")
    for stem, dtype in [("baseline_errors", "<f8"), ("original_errors", "<f8"), ("baseline_update_flags", "u1")]:
        path = folder/(stem+".raw")
        values = np.fromfile(path, dtype=dtype)
        if len(values) != count:
            raise ValueError("Baseline streamed reference and error counts differ")
        target = {"baseline_errors": "dsm_errors_baseline.npy", "original_errors": "dsm_errors_original.npy", "baseline_update_flags": "baseline_update_flags.npy"}[stem]
        np.save(folder/target, values)
        path.unlink()
    return {"reference_count": count, "reference_identity_sha256": digest.hexdigest(), "acceptance_reference_sha256": acceptance_digest.hexdigest(), "per_image": per_image, "reference_points_sha256": get_sha256(reference_path), "reference_dtype": REFERENCE_DTYPE.descr, "pixel_order": "row_major", "domain_definition": "finite positive GT depth, finite backprojection, containing cell inside input map; no method mask or error selection"}

def write_evaluation_report(folder, report):
    """Write the full-precision primary table and separate diagnostic limitations for human review."""
    lines = ["# GT 评价结果", "", f"执行状态：{report['status']}", f"八项主验收：{'通过' if report['primary_acceptance_passed'] else '未通过'}", f"完整交付条件：{'满足' if report['delivery_complete'] else '未满足'}", "", "| 指标 | 基线 | 联合 | 通过 |", "|---|---:|---:|---|"]
    for name, row in report["primary_comparisons"].items():
        lines.append(f"| {name} | {row['baseline']} | {row['joint']} | {row['passed']} |")
    lines.extend(["", f"固定参考域 D：{report['reference_count']}；主验收 D0：{report['acceptance_reference_count']}。", "", "## 分区诊断", "", "| 分区 | 参考点数 | 像元数 | 面积 m² | 配对 median（基线/联合） | 配对 std（基线/联合） | 退化 |", "|---|---:|---:|---:|---|---|---|"])
    for name, values in report["partitions"].items():
        baseline, joint = values["paired_baseline_joint"]["baseline"], values["paired_baseline_joint"]["joint"]
        lines.append(f"| {name} | {values['reference_count']} | {values['cell_count']} | {values['area_m2']} | {baseline['median']} / {joint['median']} | {baseline['std']} / {joint['std']} | {values['paired_diagnostic_degradation']} |")
    lines.extend(["", "下表为各方法在相同分区内自身有效子集的统计；方法间配对比较使用上表及 JSON 的配对字段。", "", "| 分区 | 方法 | 有效数 | 无效数 | median m | std m |", "|---|---|---:|---:|---:|---:|"])
    for name, values in report["partitions"].items():
        for method, statistics in values["per_method_valid_subset"].items():
            lines.append(f"| {name} | {method} | {statistics['valid_count']} | {statistics['invalid_count']} | {statistics['median']} | {statistics['std']} |")
    fill = report["original_nodata_new_fill"]
    statistics = fill["joint_absolute_error"]
    lines.extend(["", "## 原始 NoData 新增区", "", f"参考点 {fill['reference_count']}，有效 {statistics['valid_count']}，无效 {statistics['invalid_count']}，有效覆盖率 {fill['valid_coverage_ratio']}，像元 {fill['cell_count']}，面积 {fill['area_m2']} m²。", f"联合绝对高程误差 median/std：{statistics['median']} / {statistics['std']} m；无基线配对参考点：{fill['baseline_unavailable_count']}。"])
    lines.extend(["", "分区退化为独立诊断，不增加八项主指标之外的精度门槛。无基线配对的新填补区仅报告联合绝对误差。", "GT 为渲染位姿与深度参考，不宣称独立实测精度。未按 GT 回调重建参数或掩膜。", "本批仅用有限正深度，不使用 far，不删除最大深度；仅适用审计中的六组文件。", "原初始化使用完整图像观测；此前留出验证仅解释为条件稳定性，不作为独立泛化精度。"])
    (folder/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

def run_evaluation(config, evaluation_path, config_path, root, output_dir, write_json):
    """Freeze baseline metrics before joint scoring and preserve exact full-domain primary and partition evidence."""
    evaluation = json.loads(evaluation_path.read_text())
    check_evaluation_config(evaluation)
    folder = output_dir/"evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    log_path = output_dir/"logs/evaluate.log"
    report = {"schema_version": 3, "stage": "evaluate", "status": "running", "error": None, "gt_inputs_read": True}
    write_json(log_path, report)
    inputs, pairs, updates = collect_evaluation_inputs(config, evaluation, config_path, evaluation_path, root, output_dir)
    total_pixels = len(pairs)*config["camera"]["width"]*config["camera"]["height"]
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < 8*1024**3 or shutil.disk_usage(folder).free < total_pixels*(REFERENCE_DTYPE.itemsize+36)+512*1024**2:
        raise RuntimeError("Insufficient resources for exact full-pixel reference, median and error artifacts")
    origin = json.loads((output_dir/"shared/origin.json").read_text())
    names = config["image_names"]
    truth = dict(zip(names, read_geographic_poses(root/evaluation["gt_pose_path"], names)))
    original, profile = load_evaluation_raster(root/config["dsm_path"])
    baseline_path = folder/"baseline_metrics.json"
    if baseline_path.is_file():
        frozen = json.loads(baseline_path.read_text())
        if frozen["input_sha256"] != inputs:
            raise ValueError("Frozen baseline inputs or evaluation code changed; refusing to overwrite baseline")
        for name, digest in frozen["artifact_sha256"].items():
            if get_sha256(folder/name) != digest:
                raise ValueError("Frozen baseline reference or errors changed")
    else:
        baseline, baseline_profile = load_evaluation_raster(output_dir/"baseline/map/updated_dsm.tif", profile)
        baseline_mask = load_update_mask(output_dir/"baseline/map/update_mask.tif", profile)
        reference = stream_baseline_references(config, pairs, truth, origin, original, baseline, baseline_mask, profile, folder)
        baseline_errors = np.load(folder/"dsm_errors_baseline.npy", mmap_mode="r")
        valid = np.isfinite(baseline_errors)
        if not valid.any():
            raise ValueError("Baseline has no finite samples in the fixed reference domain D")
        poses = dict(zip(names, read_geographic_poses(output_dir/"baseline/export/poses_wgs84.txt", names)))
        pose_metrics, pose_rows = compute_pose_metrics(poses, truth, names, origin)
        dsm = summarize_error_values(baseline_errors, valid)
        metrics = dict(pose_metrics, dsm_median_m=dsm["median"], dsm_std_m=dsm["std"])
        artifacts = ["reference_points.bin", "dsm_errors_baseline.npy", "dsm_errors_original.npy", "baseline_update_flags.npy"]
        frozen = {"schema_version": 3, "input_sha256": inputs, "metrics": metrics, "pose_rows": pose_rows, "reference_metadata": reference, "acceptance_reference_count": int(valid.sum()), "artifact_sha256": {name: get_sha256(folder/name) for name in artifacts}}
        write_json(baseline_path, frozen)
    baseline_sha = get_sha256(baseline_path)
    reference = frozen["reference_metadata"]
    count = reference["reference_count"]
    print(f"evaluate baseline frozen: D={count}, D0={frozen['acceptance_reference_count']}", flush=True)
    depth_validity = {"rule": "np.isfinite(depth) & (depth > 0)", "rule_version": evaluation["valid_depth_rule"], "user_confirmation": evaluation["depth_rule_confirmation"], "assistant_confirmation": "Accepted for precisely these six fingerprinted image/depth pairs; no far filtering or maximum deletion", "pairs": pairs}
    reference_manifest = {"schema_version": 3, **reference, "image_order": names, "depth_validity": depth_validity, "baseline_update_mask_sha256": get_sha256(output_dir/"baseline/map/update_mask.tif"), "joint_update_mask_sha256": get_sha256(output_dir/"joint/map/update_mask.tif"), "gt_pose_sha256": get_sha256(root/evaluation["gt_pose_path"]), "gt_depth_sha256": {pair["image_name"]: pair["depth_sha256"] for pair in pairs}, "evaluation_count": count, "acceptance_reference_count": frozen["acceptance_reference_count"], "baseline_metrics_sha256": baseline_sha}
    write_json(folder/"reference_manifest.json", reference_manifest)
    report.update(phase="baseline_frozen_before_joint_scores", baseline_metrics_sha256=baseline_sha, reference_count=count)
    write_json(log_path, report)
    references = np.memmap(folder/"reference_points.bin", dtype=REFERENCE_DTYPE, mode="r", shape=(count,))
    baseline_errors = np.load(folder/"dsm_errors_baseline.npy", mmap_mode="r")
    original_errors = np.load(folder/"dsm_errors_original.npy", mmap_mode="r")
    baseline_flags = np.load(folder/"baseline_update_flags.npy", mmap_mode="r").astype(bool)
    joint, joint_profile = load_evaluation_raster(output_dir/"joint/map/updated_dsm.tif", profile)
    joint_mask = load_update_mask(output_dir/"joint/map/update_mask.tif", profile)
    joint_errors = np.lib.format.open_memmap(folder/"dsm_errors_joint.npy", mode="w+", dtype="<f8", shape=(count,))
    joint_flags = np.zeros(count, dtype=bool)
    original_cell_nodata = np.zeros(count, dtype=bool)
    for first in range(0, count, 262144):
        rows = references[first:first+262144]
        geographic = np.column_stack([rows["longitude"], rows["latitude"], rows["height"]])
        values, valid = sample_dsm(joint, profile["transform"], geographic)
        joint_errors[first:first+len(rows)] = np.where(valid, np.abs(values-rows["height"]), np.nan)
        joint_flags[first:first+len(rows)] = joint_mask.ravel()[rows["cell_index"]]
        original_cell_nodata[first:first+len(rows)] = np.ma.getmaskarray(original).ravel()[rows["cell_index"]]
    joint_errors.flush()
    masks = get_partition_masks(baseline_flags, joint_flags, np.isfinite(baseline_errors), np.isfinite(joint_errors), np.isfinite(original_errors), original_cell_nodata)
    if int(masks["D0"].sum()) != frozen["acceptance_reference_count"]:
        raise ValueError("Fixed D0 cardinality changed during joint evaluation")
    dsm = summarize_error_values(joint_errors, masks["D0"])
    estimates = dict(zip(names, read_geographic_poses(output_dir/"joint/export/poses_wgs84.txt", names)))
    pose_metrics, joint_pose_rows = compute_pose_metrics(estimates, truth, names, origin)
    joint_metrics = dict(pose_metrics, dsm_median_m=dsm["median"] if not dsm["invalid_count"] else None, dsm_std_m=dsm["std"] if not dsm["invalid_count"] else None)
    regressions = int(((masks["valid_baseline"] | masks["valid_original"]) & ~masks["valid_joint"]).sum())
    comparisons, passed = compare_primary_metrics(frozen["metrics"], joint_metrics, regressions)
    errors = {"baseline": baseline_errors, "joint": joint_errors, "original": original_errors}
    partitions = get_partition_statistics(errors, masks, references, profile)
    np.savez_compressed(folder/"dsm_partition_masks.npz", **masks)
    fill = masks["original_nodata_new_region"]
    fill_grid = np.zeros(original.shape, dtype=bool)
    fill_grid.ravel()[references["cell_index"][fill]] = True
    fill_statistics = summarize_error_values(joint_errors, fill)
    new_fill = {"reference_count": int(fill.sum()), "joint_absolute_error": fill_statistics, "valid_coverage_ratio": fill_statistics["valid_count"]/int(fill.sum()) if fill.any() else None, "cell_count": int(fill_grid.sum()), "area_m2": compute_updated_area(fill_grid, profile["transform"]), "baseline_unavailable_count": int((fill & ~masks["valid_baseline"]).sum()), "limitation": "No baseline pair where baseline sampling is invalid; no additional absolute accuracy gate"}
    covered_cells = np.zeros(original.shape, dtype=bool)
    covered_cells.ravel()[references["cell_index"]] = True
    outside_reference = {}
    for branch, mask in [("baseline", load_update_mask(output_dir/"baseline/map/update_mask.tif", profile)), ("joint", joint_mask)]:
        outside = mask & ~covered_cells
        outside_reference[branch] = {"updated_cells_outside_D": int(outside.sum()), "area_m2": compute_updated_area(outside, profile["transform"]), "accuracy_evaluated": False}
    reference_manifest.update(valid_reference_counts={name: int(masks["valid_"+name].sum()) for name in errors}, partition_counts={name: value["reference_count"] for name, value in partitions.items()})
    write_json(folder/"reference_manifest.json", reference_manifest)
    with (folder/"pose_errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["branch", "image_name", "status", "translation_error_m", "rotation_error_deg"])
        writer.writeheader()
        for branch, rows in [("baseline", frozen["pose_rows"]), ("joint", joint_pose_rows)]:
            writer.writerows(dict(row, branch=branch) for row in rows)
    for path, digest in inputs.items():
        if get_sha256(path) != digest:
            raise ValueError("Frozen reconstruction or GT inputs changed during evaluation")
    if get_sha256(baseline_path) != baseline_sha:
        raise ValueError("Frozen baseline metrics changed after joint GT scoring")
    for name, digest in frozen["artifact_sha256"].items():
        if get_sha256(folder/name) != digest:
            raise ValueError("Frozen D, D0 source or baseline errors changed after joint scoring")
    delivery = passed and all(value == "updated" for value in updates.values())
    report.update(status="complete", phase="complete", registered_images=6, reference_count=count, acceptance_reference_count=frozen["acceptance_reference_count"], baseline_metrics=frozen["metrics"], joint_metrics=joint_metrics, primary_comparisons=comparisons, primary_acceptance_passed=passed, invalid_regression_count=regressions, partitions=partitions, original_nodata_new_fill=new_fill, updated_outside_reference=outside_reference, source_update_status=updates, delivery_complete=delivery, input_sha256=inputs, initialization_uses_heldout=True, prior_validation_scope="conditional holdout stability; not independent generalization", artifacts={name: str(folder/name) for name in ["baseline_metrics.json", "reference_points.bin", "reference_manifest.json", "pose_errors.csv", "dsm_errors_baseline.npy", "dsm_errors_joint.npy", "dsm_errors_original.npy", "dsm_partition_masks.npz", "report.md"]})
    write_evaluation_report(folder, report)
    report["output_sha256"] = {name: get_sha256(path) for name, path in report["artifacts"].items()}
    write_json(folder/"report.json", report)
    write_json(log_path, report)
    print(f"evaluate: complete | D={count} | D0={frozen['acceptance_reference_count']} | primary_acceptance={'passed' if passed else 'failed'}", flush=True)
    return report
