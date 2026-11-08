# Densify the original PixSfM model with CUDA COLMAP and verify the fused cloud.
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pycolmap

def read_depth_map(path):
    """Read a COLMAP binary depth map without changing its values."""
    raw = path.read_bytes()
    offset = 0
    for _ in range(3):
        offset = raw.index(b"&", offset) + 1
    width, height, channels = map(int, raw[:offset - 1].split(b"&"))
    values = np.frombuffer(raw, dtype="<f4", offset=offset)
    return values.reshape((width, height, channels), order="F").transpose(1, 0, 2).squeeze()

def read_fused_cloud(path):
    """Read and validate the standard binary COLMAP point cloud format."""
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline().decode("ascii").strip()
            if not line:
                raise ValueError("Invalid PLY header")
            header.append(line)
            if line == "end_header":
                break
        assert "format binary_little_endian 1.0" in header
        count = int(next(line.split()[2] for line in header
                         if line.startswith("element vertex ")))
        dtype = np.dtype([(name, "<f4") for name in ["x", "y", "z", "nx", "ny", "nz"]]
                         + [(name, "u1") for name in ["red", "green", "blue"]])
        points = np.fromfile(stream, dtype=dtype, count=count)
    assert count > 0 and len(points) == count
    assert all(np.all(np.isfinite(points[name])) for name in ["x", "y", "z", "nx", "ny", "nz"])
    return points

def run_dense_reconstruction():
    """Run undistortion, GPU PatchMatch, fusion and artifact verification."""
    started = time.monotonic()
    output_dir = Path(__file__).resolve().parent
    root = output_dir.parents[1]
    binary = root / "build/colmap_cuda/install_ptx90/bin/colmap"
    image_dir = root / "datasets/images/DJI_20251221132525_0004_V"
    sparse_dir = output_dir / "sfm"
    dense_dir = output_dir / "dense"
    max_image_size = 2000
    model = pycolmap.Reconstruction(str(sparse_dir))
    assert binary.is_file() and model.num_reg_images() == 6
    commands = [
        ("undistort", [str(binary), "image_undistorter",
                       "--image_path", str(image_dir),
                       "--input_path", str(sparse_dir),
                       "--output_path", str(dense_dir),
                       "--output_type", "COLMAP",
                       "--num_patch_match_src_images", "5"]),
        ("patch_match", [str(binary), "patch_match_stereo",
                         "--workspace_path", str(dense_dir),
                         "--workspace_format", "COLMAP",
                         "--PatchMatchStereo.max_image_size", str(max_image_size),
                         "--PatchMatchStereo.gpu_index", "0",
                         "--PatchMatchStereo.geom_consistency", "1",
                         "--PatchMatchStereo.write_consistency_graph", "1",
                         "--PatchMatchStereo.cache_size", "2"]),
        ("fusion", [str(binary), "stereo_fusion",
                    "--workspace_path", str(dense_dir),
                    "--workspace_format", "COLMAP",
                    "--input_type", "geometric",
                    "--output_path", str(dense_dir / "fused.ply"),
                    "--StereoFusion.max_image_size", str(max_image_size),
                    "--StereoFusion.num_threads", "8"]),
    ]
    for name, command in commands:
        print("Running", name, flush=True)
        with (output_dir / f"dense_{name}.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    depth_checks = []
    for image in sorted(model.images.values(), key=lambda image: image.name):
        depth = read_depth_map(dense_dir / "stereo/depth_maps" / f"{image.name}.geometric.bin")
        valid = np.isfinite(depth) & (depth > 0)
        assert np.count_nonzero(valid) > 0
        depth_checks.append({"image": image.name, "width": depth.shape[1],
                             "height": depth.shape[0], "valid_pixels": int(valid.sum()),
                             "total_pixels": int(depth.size)})
    points = read_fused_cloud(dense_dir / "fused.ply")
    summary = {
        "status": "PASS",
        "sparse_model": str(sparse_dir),
        "registered_images": model.num_reg_images(),
        "sparse_points": model.num_points3D(),
        "dense_points": len(points),
        "max_image_size": max_image_size,
        "depth_maps": depth_checks,
        "colmap_binary": str(binary),
        "cuda_architecture": "90-virtual",
        "commands": {name: command for name, command in commands},
        "coordinate_system": "arbitrary_local_sfm",
        "gt_pose_depth_used": False,
        "accuracy_evaluation": "not_performed",
        "elapsed_seconds": time.monotonic() - started,
    }
    (dense_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    run_dense_reconstruction()
