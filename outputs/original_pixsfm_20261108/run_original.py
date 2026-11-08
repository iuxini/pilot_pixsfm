# Reproduce the README PixSfM reconstruction on six images with fixed intrinsics.
import json
import time
from pathlib import Path
import pycolmap
import torch
from hloc import extract_features, match_features, pairs_from_exhaustive
from omegaconf import OmegaConf
from pixsfm.refine_hloc import PixSfM

def run_reconstruction():
    """Run original featuremetric SfM and save its native model and statistics."""
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).resolve().parent
    image_dir = root / "datasets/images/DJI_20251221132525_0004_V"
    image_names = [f"{index}_0.png" for index in range(6)]
    assert all((image_dir / name).is_file() for name in image_names)
    assert torch.cuda.is_available(), "GPU access is required for this run."
    features_path = output_dir / "keypoints.h5"
    matches_path = output_dir / "matches.h5"
    pairs_path = output_dir / "pairs.txt"
    sfm_dir = output_dir / "sfm"
    feature_conf = extract_features.confs["superpoint_aachen"]
    match_conf = match_features.confs["superglue"]
    conf = {
        "dense_features": {"use_cache": True},
        "BA": {"optimizer": {
            "refine_focal_length": False,
            "refine_principal_point": False,
            "refine_extra_params": False,
        }},
    }
    mapper_options = {
        "ba_refine_focal_length": False,
        "ba_refine_principal_point": False,
        "ba_refine_extra_params": False,
    }
    refiner = PixSfM(conf)
    run_config = {
        "image_dir": str(image_dir),
        "image_names": image_names,
        "camera_model": "PINHOLE",
        "width": 3840,
        "height": 2160,
        "camera_params": [2700.0, 2700.0, 1920.0, 1080.0],
        "features": feature_conf,
        "matching": match_conf,
        "mapping": mapper_options,
        "pixsfm": OmegaConf.to_container(refiner.conf, resolve=True),
        "coordinate_system": "arbitrary_local_sfm",
        "gpu": torch.cuda.get_device_name(0),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n")
    extract_features.main(feature_conf, image_dir, image_list=image_names,
                          feature_path=features_path)
    pairs_from_exhaustive.main(pairs_path, image_list=image_names)
    match_features.main(match_conf, pairs_path, features=features_path,
                        matches=matches_path)
    model, outputs = refiner.reconstruction(
        sfm_dir, image_dir, pairs_path, features_path, matches_path,
        cache_path=output_dir / "dense_features.h5",
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_list=image_names,
        image_options={"camera_model": "PINHOLE",
                       "camera_params": "2700,2700,1920,1080"},
        mapper_options=mapper_options,
        verbose=True,
    )
    model.write_text(str(sfm_dir))
    model.export_PLY(str(sfm_dir / "sparse.ply"))
    summary = {
        "num_input_images": len(image_names),
        "num_registered_images": model.num_reg_images(),
        "registered_images": sorted(image.name for image in model.images.values()),
        "num_points3D": model.num_points3D(),
        "colmap_stored_mean_reprojection_error_px": model.compute_mean_reprojection_error(),
        "mean_track_length": model.compute_mean_track_length(),
        "ka_executed": outputs["KA"] is not None,
        "ba_executed": outputs["BA"] is not None,
        "elapsed_seconds": time.monotonic() - started,
        "coordinate_system": "arbitrary_local_sfm",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    assert model.num_reg_images() == len(image_names)
    assert model.num_points3D() > 0
    assert summary["ka_executed"] and summary["ba_executed"]

if __name__ == "__main__":
    run_reconstruction()
