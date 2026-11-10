# 用途：验证地图联合优化的几何、共享参数、留出隔离和特征缓存。
import copy
import json
import hashlib
import h5py
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.spatial.transform import Rotation
from joint_map.geometry import (
    backproject_pixels, convert_ecef_to_geodetic, convert_enu_to_geodetic,
    convert_geodetic_to_ecef, convert_geodetic_to_enu, convert_pixels_to_geodetic,
    convert_pose_from_colmap, convert_pose_to_colmap, create_origin,
    get_ecef_to_enu_rotation, sample_dsm, transform_colmap_pose,
)
from run_joint_map import PROJECT_ROOT, check_config, get_file_fingerprint
from joint_map.registration import (
    apply_sim3, check_camera_connectivity, check_feature_identity,
    create_dom_variant, estimate_sim3, fit_sim3, screen_homography,
    select_map_candidates, transform_pixels, write_initialized_model,
)
from joint_map.optimization import (
    build_shared_problem, check_gauge_information, compute_huber_cost,
    compute_robust_scale, create_holdout, create_synthetic_scene,
    get_observations, require_confirmed_prior, run_synthetic_controls,
    audit_training_references, check_dense_cache, compare_dense_and_patch_cache,
    get_observation_digest, sample_dense_holdout,
    RecordSolverIterations, solve_shared_problem,
)
import pycolmap
import pyceres
from pixsfm import features
from pixsfm.base import InterpolationConfig, interpolation_default_conf

class TestCoordinates(unittest.TestCase):
    """Check geographic axes, units and numerical round trips."""
    def test_known_wgs84_axes(self):
        """Check ECEF against WGS84 equatorial and polar reference positions."""
        positions = [[0, 0, 0], [90, 0, 100], [0, 90, 0]]
        expected = [[6378137, 0, 0], [0, 6378237, 0], [0, 0, 6356752.314245179]]
        np.testing.assert_allclose(convert_geodetic_to_ecef(positions), expected, atol=1e-7, rtol=0)

    def test_geographic_round_trip(self):
        """Preserve degree angles and meter heights across ECEF and ENU."""
        origin = create_origin(112.99, 28.29, 70)
        positions = np.array([[112.99, 28.29, 70], [112.991, 28.292, 180], [112.989, 28.288, -10]])
        restored = convert_ecef_to_geodetic(convert_geodetic_to_ecef(positions))
        np.testing.assert_allclose(restored[:, :2], positions[:, :2], atol=1e-11, rtol=0)
        np.testing.assert_allclose(restored[:, 2], positions[:, 2], atol=1e-6, rtol=0)
        restored = convert_enu_to_geodetic(convert_geodetic_to_enu(positions, origin), origin)
        np.testing.assert_allclose(restored, positions, atol=1e-6, rtol=0)
        np.testing.assert_allclose(convert_geodetic_to_enu(positions[0], origin), 0, atol=1e-8)

    def test_enu_axes_and_height(self):
        """Verify east, north, up orientation and distinguish ENU z from height."""
        rotation = get_ecef_to_enu_rotation(0, 0)
        np.testing.assert_allclose(rotation, [[0, 1, 0], [0, 0, 1], [1, 0, 0]], atol=1e-15)
        origin = create_origin(112.99, 28.29, 70)
        offsets = convert_geodetic_to_enu([[112.99001, 28.29, 70], [112.99, 28.29001, 70], [112.99, 28.29, 71]], origin)
        self.assertGreater(offsets[0, 0], 0.9)
        self.assertGreater(offsets[1, 1], 1.0)
        np.testing.assert_allclose(offsets[2], [0, 0, 1], atol=1e-8)
        point = convert_enu_to_geodetic([10000, 10000, 0], origin)
        self.assertGreater(point[2] - origin["ellipsoidal_height_m"], 10)
        np.testing.assert_allclose(np.asarray(origin["R_ecef_to_enu"]) @ np.asarray(origin["R_ecef_to_enu"]).T, np.eye(3), atol=1e-15)

class TestCameraGeometry(unittest.TestCase):
    """Verify the planned renderer and COLMAP camera conventions."""
    def setUp(self):
        """Create synthetic camera inputs without loading real GT."""
        self.origin = create_origin(112.99, 28.29, 70)
        self.intrinsics = np.array([[2700., 0, 1920], [0, 2700, 1080], [0, 0, 1]])
        self.position = np.array([112.99, 28.29, 170])
        self.pixels = np.array([[1920, 1080], [0, 0], [3839, 0], [0, 2159], [3839, 2159]])

    def test_nonzero_pose_round_trip(self):
        """Round trip nonzero attitudes at varying camera geographic positions."""
        for position, angles in [([112.991, 28.292, 170], [13, -21, 37]), ([112.989, 28.288, 125], [-18, 34, -123])]:
            with self.subTest(position=position, angles=angles):
                rotation, translation = convert_pose_to_colmap(position, angles, self.origin)
                restored_position, restored_angles = convert_pose_from_colmap(rotation, translation, self.origin)
                np.testing.assert_allclose(restored_position, position, atol=1e-6, rtol=0)
                np.testing.assert_allclose(restored_angles, angles, atol=1e-10, rtol=0)
                np.testing.assert_allclose(rotation @ convert_geodetic_to_enu(position, self.origin) + translation, 0, atol=1e-10)

    def test_nadir_center_and_corners(self):
        """Verify right, down, forward camera axes and positive z-depth rays."""
        rotation, translation = convert_pose_to_colmap(self.position, [0, 0, 0], self.origin)
        points = backproject_pixels(self.pixels, np.full(5, 100.), self.intrinsics, rotation, translation)
        expected = np.column_stack([(self.pixels[:, 0] - 1920) * 100 / 2700, -(self.pixels[:, 1] - 1080) * 100 / 2700, np.zeros(5)])
        np.testing.assert_allclose(points, expected, atol=1e-8, rtol=0)
        self.assertGreater(points[1, 1], points[3, 1])
        self.assertGreater(points[2, 0], points[1, 0])

    def test_nonzero_attitude_rays(self):
        """Check Euler ordering against separately constructed axis rotations."""
        roll, pitch, yaw = 13., -21., 37.
        x, y, z = np.deg2rad([pitch, roll, yaw])
        rotate_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        rotate_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rotate_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        camera_to_enu = rotate_z @ rotate_y @ rotate_x @ np.diag([1, -1, -1])
        rotation, translation = convert_pose_to_colmap(self.position, [roll, pitch, yaw], self.origin)
        depths = np.array([80., 90., 100., 110., 120.])
        rays = np.column_stack([(self.pixels[:, 0] - 1920) / 2700, (self.pixels[:, 1] - 1080) / 2700, np.ones(5)])
        expected = (rays * depths[:, None]) @ camera_to_enu.T + [0, 0, 100]
        points = backproject_pixels(self.pixels, depths, self.intrinsics, rotation, translation)
        np.testing.assert_allclose(points, expected, atol=1e-8, rtol=0)
        camera_points = points @ rotation.T + translation
        projected = camera_points @ self.intrinsics.T
        np.testing.assert_allclose(projected[:, :2] / projected[:, 2:], self.pixels, atol=1e-9)

    def test_sim3_preserves_projection(self):
        """Transform points, camera centers and z-depth consistently under Sim(3)."""
        rotation, translation = convert_pose_to_colmap(self.position, [13, -21, 37], self.origin)
        points = backproject_pixels(self.pixels, np.full(5, 100.), self.intrinsics, rotation, translation)
        scale = 2.7
        align = Rotation.from_euler("xyz", [23, -15, 64], degrees=True).as_matrix()
        offset = np.array([12., -8., 40.])
        rotation_new, translation_new = transform_colmap_pose(rotation, translation, scale, align, offset)
        points_new = scale * points @ align.T + offset
        camera_old = points @ rotation.T + translation
        camera_new = points_new @ rotation_new.T + translation_new
        np.testing.assert_allclose(camera_new, scale * camera_old, atol=1e-10)
        center_old = -rotation.T @ translation
        np.testing.assert_allclose(-rotation_new.T @ translation_new, scale * align @ center_old + offset, atol=1e-10)
        projected = camera_new @ self.intrinsics.T
        np.testing.assert_allclose(projected[:, :2] / projected[:, 2:], self.pixels, atol=1e-9)

    def test_invalid_depth_and_scale(self):
        """Reject nonpositive or nonfinite depth and Sim(3) scale."""
        for value in [0, -1, np.nan, np.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    backproject_pixels([[1920, 1080]], [value], self.intrinsics, np.eye(3), np.zeros(3))
                with self.assertRaises(ValueError):
                    transform_colmap_pose(np.eye(3), np.zeros(3), value, np.eye(3), np.zeros(3))

class TestMapSampling(unittest.TestCase):
    """Check half pixel placement, bilinear boundaries, masks and GeoTIFF metadata."""
    def setUp(self):
        """Create a planar height field with a rotated geographic affine grid."""
        self.transform = Affine(0.00001, 0.000002, 112.99, 0.000001, -0.00001, 28.29)
        rows, columns = np.indices((4, 5))
        self.heights = np.ma.array(7 + 2 * columns + 3 * rows, dtype=np.float64, mask=False)

    def test_pixel_centers_and_fractional_sample(self):
        """Locate pixel centers and interpolate a known plane without half pixel drift."""
        pixels = np.array([[0, 0], [4, 3], [1.25, 2.5]])
        positions = convert_pixels_to_geodetic(pixels, self.transform)
        np.testing.assert_allclose(positions[0], [112.990006, 28.2899955], atol=1e-13, rtol=0)
        values, valid = sample_dsm(self.heights, self.transform, positions)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(values, 7 + 2 * pixels[:, 0] + 3 * pixels[:, 1], atol=1e-8, rtol=0)

    def test_outside_and_nonfinite_positions(self):
        """Reject extrapolation beyond center boundaries and invalid coordinates."""
        positions = convert_pixels_to_geodetic([[-0.1, 1], [4.1, 1], [1, -0.1], [1, 3.1]], self.transform)
        positions = np.concatenate([positions, [[np.nan, 28.29], [np.inf, 28.29]]])
        values, valid = sample_dsm(self.heights, self.transform, positions)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(values).all())

    def test_invalid_four_neighbor_support(self):
        """Reject masked or nonfinite neighbors instead of interpolating through holes."""
        positions = convert_pixels_to_geodetic([[1.25, 1.5], [3.2, 2.5]], self.transform)
        self.heights.mask[1, 1] = True
        values, valid = sample_dsm(self.heights, self.transform, positions)
        np.testing.assert_array_equal(valid, [False, True])
        self.assertTrue(np.isnan(values[0]))
        self.heights[2, 3] = np.nan
        self.assertFalse(sample_dsm(self.heights, self.transform, positions)[1].any())

    def test_geotiff_round_trip(self):
        """Preserve CRS, affine grid and NoData when reopening a synthetic GeoTIFF."""
        self.heights.mask[1, 1] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dsm.tif"
            with rasterio.open(path, "w", driver="GTiff", width=5, height=4, count=1, dtype="float32", crs="EPSG:4326", transform=self.transform, nodata=-9999) as dataset:
                dataset.write(self.heights.filled(-9999).astype(np.float32), 1)
            with rasterio.open(path) as dataset:
                self.assertEqual(dataset.crs, rasterio.crs.CRS.from_epsg(4326))
                self.assertEqual(dataset.transform, self.transform)
                self.assertEqual(dataset.nodata, -9999)
                restored = dataset.read(1, masked=True)
            np.testing.assert_array_equal(restored.mask, self.heights.mask)
            np.testing.assert_array_equal(restored.compressed(), self.heights.compressed())
            positions = convert_pixels_to_geodetic([[1.25, 1.5], [3.2, 2.5]], self.transform)
            np.testing.assert_array_equal(sample_dsm(restored, self.transform, positions)[1], [False, True])

class TestInputProtocol(unittest.TestCase):
    """Check fixed camera requirements, GT exclusion and exact input fingerprints."""
    def test_fixed_camera_and_nested_gt_rejection(self):
        """Reject changed intrinsics and evaluation fields at any configuration depth."""
        config = json.loads((PROJECT_ROOT / "configs/joint_map.json").read_text())
        check_config(config)
        altered = copy.deepcopy(config)
        altered["camera"]["params"][0] = 2701
        with self.assertRaises(ValueError):
            check_config(altered)
        altered = copy.deepcopy(config)
        altered["frontend"]["gt_pose_path"] = "forbidden.txt"
        with self.assertRaises(ValueError):
            check_config(altered)

    def test_input_fingerprint(self):
        """Record exact file size and the known SHA-256 of synthetic content."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"abc")
            fingerprint = get_file_fingerprint(path)
        self.assertEqual(fingerprint["size_bytes"], 3)
        self.assertEqual(fingerprint["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

class TestMapRegistration(unittest.TestCase):
    """Verify DOM coordinate recovery, robust binding and exclusion of candidate evidence."""
    def create_candidates(self, count=16):
        """Create independent synthetic image tracks with one DOM candidate per track."""
        source = np.random.default_rng(4).normal(size=(count, 3)) * 10
        transform = (2.3, Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix(), np.array([30., -15., 6.]))
        links = {"track_ids": np.arange(count), "image_ids": np.zeros(count, dtype=int), "scores": np.full(count, 0.8), "homography_pass": np.ones(count, dtype=bool), "means_enu_m": apply_sim3(source, transform), "rejection_codes": np.full(count, "pending_3d", dtype="U64")}
        return source, links, transform

    def test_dom_variant_round_trip_and_mask(self):
        """Recover native DOM centers through identity, rotations, scales and resize maps."""
        gray = np.arange(63, dtype=np.uint8).reshape(7, 9)
        valid = np.ones_like(gray, dtype=bool)
        valid[3, 4] = False
        image, mask, matrix = create_dom_variant(gray, valid, 0, 1)
        np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)
        self.assertFalse(mask[3, 4])
        self.assertEqual(image[3, 4], 0)
        for angle, scale in [(30, 2), (90, 4), (330, 1)]:
            image, mask, matrix = create_dom_variant(gray, valid, angle, scale)
            pixels = np.array([[0., 0.], [8., 6.], [2.5, 4.25]])
            transformed = transform_pixels(pixels, matrix)
            np.testing.assert_allclose(transform_pixels(transformed, np.linalg.inv(matrix)), pixels, atol=1e-12)
            ratio = np.array([2.2, 1.8])
            resized = (transformed + 0.5) / ratio - 0.5
            restored = (resized + 0.5) * ratio - 0.5
            np.testing.assert_allclose(transform_pixels(restored, np.linalg.inv(matrix)), pixels, atol=1e-12)
            self.assertTrue((transformed >= -1e-10).all())
            self.assertTrue((transformed[:, 0] <= image.shape[1] - 1 + 1e-10).all())
            self.assertFalse(mask.all())

    def test_original_dom_forward_homography_error(self):
        """Compute forward errors after undoing variant scale rather than in variant pixels."""
        source = np.random.default_rng(0).uniform(0, 100, (30, 2))
        expected = np.array([[0.5, 0.03, 4], [0.02, 0.6, 7], [0.0001, -0.0002, 1]])
        dom = transform_pixels(source, expected)
        variant = np.array([[4., 0, 20], [0, 4, 10], [0, 0, 1]])
        restored = transform_pixels(transform_pixels(dom, variant), np.linalg.inv(variant))
        restored[-1] += [10, 0]
        errors, matrix = screen_homography(source, restored, 3.)
        self.assertTrue((errors[:-1] < 1e-4).all())
        self.assertGreater(errors[-1], 9.9)
        self.assertEqual(screen_homography(source[:3], dom[:3], 3)[1], None)

    def test_robust_sim3_and_degeneracy(self):
        """Recover proper scale and rotation despite gross outliers, allowing planar anchors."""
        source, links, transform = self.create_candidates()
        target = links["means_enu_m"].copy()
        target[:3] += 80
        fitted, mask = estimate_sim3(source, target, 2.)
        np.testing.assert_allclose(apply_sim3(source[3:], fitted), links["means_enu_m"][3:], atol=1e-10)
        np.testing.assert_array_equal(mask[:3], False)
        self.assertGreater(np.linalg.det(fitted[1]), 0.999)
        planar = source.copy()
        planar[:, 2] = 0
        self.assertIsNotNone(fit_sim3(planar, apply_sim3(planar, transform)))
        line = np.column_stack([np.arange(5), np.zeros((5, 2))])
        self.assertIsNone(fit_sim3(line, line))

    def test_single_image_dom_and_nonplanar_rescue(self):
        """Accept a single-image DOM association using other tracks even if planar screening fails."""
        source, links, transform = self.create_candidates()
        links["homography_pass"][0] = False
        accepted, codes, errors, counts, selected, models = select_map_candidates(links, source, 2.)
        self.assertTrue(accepted.all())
        self.assertEqual(len(selected), len(source))
        self.assertLess(errors.max(), 1e-10)
        for model in models:
            self.assertFalse(set(model["reference_track_ids"]) & set(model["excluded_track_ids"]))
            self.assertGreaterEqual(len(model["inlier_track_ids"]), 3)

    def test_reject_self_supported_high_leverage_match(self):
        """Reject a wrong high-leverage candidate that could distort its own fitted transform."""
        source, links, transform = self.create_candidates()
        source[-1] = [1000, 1000, 1000]
        links["means_enu_m"][-1] = apply_sim3(source[-1], transform) + [10, 0, 0]
        accepted, codes, errors, counts, selected, models = select_map_candidates(links, source, 2.)
        self.assertFalse(accepted[-1])
        self.assertEqual(codes[-1], "independent_3d_outlier")
        self.assertGreater(errors[-1], 9.9)
        validation = next(model for model in models if 15 in model["excluded_track_ids"])
        self.assertNotIn(15, validation["reference_track_ids"])

    def test_insufficient_independent_evidence_stays_pending(self):
        """Keep candidates pending when exclusion leaves too few nondegenerate anchors."""
        source, links, transform = self.create_candidates(3)
        accepted, codes, errors, counts, selected, models = select_map_candidates(links, source, 2.)
        self.assertFalse(accepted.any())
        self.assertTrue((codes == "pending_independent_anchors").all())
        self.assertEqual(len(selected), 0)

    def test_duplicate_variants_do_not_duplicate_priors(self):
        """Exclude all copies of a track and emit only one prior representative."""
        source, links, transform = self.create_candidates()
        source = np.concatenate([source, source[:1]])
        for key, values in links.items():
            links[key] = np.concatenate([values, values[:1]])
        accepted, codes, errors, counts, selected, models = select_map_candidates(links, source, 2.)
        self.assertTrue(accepted.all())
        self.assertEqual(len(selected), 16)
        self.assertEqual(len(set(links["track_ids"][selected])), 16)
        validation = next(model for model in models if 0 in model["excluded_track_ids"])
        self.assertNotIn(0, validation["reference_track_ids"])

    def test_conflicting_track_candidates_remain_pending(self):
        """Avoid averaging incompatible positions even when each passes the two meter test."""
        source, links, transform = self.create_candidates()
        source = np.concatenate([source, source[:1]])
        for key, values in links.items():
            links[key] = np.concatenate([values, values[:1]])
        links["means_enu_m"][0] += [1.5, 0, 0]
        links["means_enu_m"][-1] -= [1.5, 0, 0]
        accepted, codes, errors, counts, selected, models = select_map_candidates(links, source, 2.)
        self.assertFalse(accepted[[0, -1]].any())
        self.assertTrue((codes[[0, -1]] == "pending_inconsistent_track").all())

    def test_indirect_camera_connection(self):
        """Connect a camera with no direct DOM match through shared image tracks."""
        points = {index: SimpleNamespace(track=SimpleNamespace(elements=[SimpleNamespace(image_id=image_id) for image_id in images])) for index, images in enumerate([[1, 2], [2, 3], [3, 4]])}
        model = SimpleNamespace(points3D=points, images={index: None for index in [1, 2, 3, 4]})
        reached, connected = check_camera_connectivity(model, [0])
        self.assertTrue(connected)
        self.assertEqual(reached, [1, 2, 3, 4])
        self.assertFalse(check_camera_connectivity(model, [])[1])

    def test_real_identity_and_model_transform(self):
        """Verify native feature rows and preserve complete real tracks under metric conversion."""
        directory = PROJECT_ROOT / "outputs/original_pixsfm_20261108"
        model = pycolmap.Reconstruction(str(directory / "sfm"))
        identity = check_feature_identity(model, directory / "keypoints.h5", directory / "sfm/refined_keypoints.h5")
        self.assertEqual(identity["registered_observations"], 18790)
        point_ids = set(model.points3D)
        source_point = np.array(model.points3D[next(iter(point_ids))].xyz)
        point_id = next(iter(point_ids))
        track = [(entry.image_id, entry.point2D_idx) for entry in model.points3D[point_id].track.elements]
        image_id = track[0][0]
        old_image = model.images[image_id]
        old_camera_point = old_image.rotmat() @ source_point + old_image.tvec
        transform = (2.5, Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix(), np.array([10., -20., 30.]))
        with tempfile.TemporaryDirectory() as directory:
            write_initialized_model(model, transform, Path(directory))
            restored = pycolmap.Reconstruction(directory)
        self.assertEqual(set(restored.points3D), point_ids)
        self.assertEqual(restored.num_reg_images(), 6)
        self.assertEqual([(entry.image_id, entry.point2D_idx) for entry in restored.points3D[point_id].track.elements], track)
        new_image = restored.images[image_id]
        new_camera_point = new_image.rotmat() @ restored.points3D[point_id].xyz + new_image.tvec
        np.testing.assert_allclose(new_camera_point, transform[0] * old_camera_point, atol=1e-10)

class TestSharedMapPriors(unittest.TestCase):
    """Verify actual native parameter sharing, prior whitening and synthetic behavior."""
    def test_native_shared_xyz_and_manifolds(self):
        """Check native image and map factors share one movable XYZ with fixed K and open poses."""
        model, anchors, truth = create_synthetic_scene()
        scales = {"sigma_horizontal_m": 0.3, "sigma_height_m": 0.2, "sigma_pixel_px": 0.2}
        state = build_shared_problem(model, anchors, scales, 1.)
        audit = state["audit"]
        self.assertEqual(audit["geometric_blocks"], 6 * 24)
        self.assertEqual(audit["map_blocks"], 24)
        self.assertEqual(audit["total_blocks"], 6 * 24 + 24)
        self.assertTrue(audit["intrinsics_fixed"])
        self.assertTrue(all(row["pose_variable"] and row["quaternion_manifold"] and row["qvec_tangent"] == 3 for row in audit["pose_states"]))
        point_id = int(anchors["point_ids"][0])
        view = model.points3D[point_id].xyz
        self.assertEqual(int(view.ctypes.data), audit["xyz_addresses"][0])
        self.assertTrue(state["problem"].has_parameter_block(view))
        self.assertFalse(state["problem"].has_parameter_block(view.copy()))
        self.assertFalse(state["problem"].is_parameter_block_constant(view))

    def test_covariance_whitening_and_loss_magnitude(self):
        """Check diagonal NormalPrior and scaled Huber cost numerically against formulas."""
        import pyceres
        mean = np.array([1., 2., 3.])
        cost = pyceres.factors.NormalPrior(mean, np.diag([4., 4., 9.]))
        residual, jacobians = cost.evaluate(mean + [2, 4, 6])
        np.testing.assert_allclose(residual.ravel(), [1, 2, 2])
        for weight in [0.1, 1, 10]:
            loss = pyceres.LossFunction({"name": "huber", "params": [3.], "magnitude": weight})
            for squared in [1., 9., 25.]:
                self.assertAlmostEqual(loss.evaluate(squared)[0], weight * compute_huber_cost(squared, 3.))

    def test_duplicate_and_invalid_priors_rejected(self):
        """Reject repeated XYZ priors and nonpositive working scales before map factor creation."""
        model, anchors, truth = create_synthetic_scene()
        scales = {"sigma_horizontal_m": 0.3, "sigma_height_m": 0.2, "sigma_pixel_px": 0.2}
        anchors["point_ids"][1] = anchors["point_ids"][0]
        with self.assertRaises(ValueError):
            build_shared_problem(model, anchors, scales, 1.)
        scales["sigma_height_m"] = 0
        with self.assertRaises(ValueError):
            build_shared_problem(model, anchors, scales, 1.)

    def test_holdout_removed_from_complete_training_graph(self):
        """Remove heldout residuals reciprocally without losing shared XYZ or full remaining tracks."""
        model, anchors, truth = create_synthetic_scene()
        training, heldout = create_holdout(model, 0.2, 0)
        repeated, same_heldout = create_holdout(model, 0.2, 0)
        self.assertEqual([row[:3] for row in heldout], [row[:3] for row in same_heldout])
        remaining = {row[:3] for row in get_observations(training)}
        self.assertFalse(remaining & {row[:3] for row in heldout})
        self.assertEqual(len(remaining) + len(heldout), 24 * 6)
        self.assertEqual(set(training.points3D), set(model.points3D))
        self.assertEqual(len(heldout), 4)

    def test_gauge_information_and_scale_floors(self):
        """Accept distributed anchors, reject line degeneracy and honor recorded scale floors."""
        model, anchors, truth = create_synthetic_scene()
        result = check_gauge_information(truth, np.diag([0.09, 0.09, 0.04]))
        self.assertGreater(min(result["eigenvalues"]), 0)
        with self.assertRaises(ValueError):
            check_gauge_information(np.column_stack([np.arange(10), np.zeros((10, 2))]), np.eye(3))
        self.assertEqual(compute_robust_scale(np.zeros(10), 0.05), 0.05)

    def test_synthetic_finite_and_fixed_map_controls(self):
        """Demonstrate movable finite priors allow image evidence to improve noisy map geometry."""
        results = {row["label"]: row for row in run_synthetic_controls()}
        self.assertLess(results["candidate"]["shape_rmse_m"], results["fixed_map"]["shape_rmse_m"])
        self.assertGreater(results["candidate"]["displacement_median_m"], 0)
        self.assertEqual(results["none"]["audit"]["map_blocks"], 0)
        self.assertLess(results["strong"]["final_costs"]["map"] / results["strong"]["information_scale"], results["weak"]["final_costs"]["map"] / results["weak"]["information_scale"])
        self.assertTrue(all(row["usable"] for row in results.values()))

    def test_registration_waits_for_confirmation(self):
        """Block formal registration for both unvalidated and merely validated model states."""
        for status in ["unvalidated", "validated"]:
            with self.assertRaises(ValueError):
                require_confirmed_prior({"status": status})
        with self.assertRaises(ValueError):
            require_confirmed_prior({"status": "confirmed"})
        require_confirmed_prior({"status": "confirmed", "lambda_map": 1., "candidate_id": "weight_1p0", "user_confirmation": "explicit test user choice", "map_prior_scope": "training_frozen"})

    def test_native_iteration_diagnostics_and_termination_message(self):
        """Record every native solve iteration with gradient, step, radius and actual termination text."""
        model, anchors, truth = create_synthetic_scene()
        scales = {"sigma_horizontal_m": 0.3, "sigma_height_m": 0.2, "sigma_pixel_px": 0.2}
        state = build_shared_problem(model, anchors, scales, 1.)
        result = solve_shared_problem(state)
        rows = result["iterations"]
        self.assertGreater(len(rows), 1)
        self.assertEqual([row["iteration"] for row in rows], list(range(len(rows))))
        self.assertEqual(len(rows), result["successful_steps"] + result["unsuccessful_steps"])
        self.assertAlmostEqual(rows[0]["cost"], result["ceres_initial_cost"], places=7)
        self.assertTrue(result["termination_message"])
        self.assertIn("Termination", result["solver_full_report"])
        for row in rows:
            self.assertGreaterEqual(row["gradient_norm"] + 1e-12, row["gradient_max_norm"])
            self.assertGreaterEqual(row["step_norm"], 0)
            self.assertGreater(row["trust_region_radius"], 0)
            self.assertEqual(row["nonfinite_values"], [])
        json.dumps(result, allow_nan=False)

    def test_iteration_observer_continues_and_preserves_nonfinite_evidence(self):
        """Keep callback observation passive and serialize nonfinite diagnostics without losing their identity."""
        import pyceres
        fields = {name: 1. for name in ["cost", "cost_change", "gradient_norm", "gradient_max_norm", "step_norm", "step_size", "trust_region_radius", "relative_decrease", "iteration_time_in_seconds", "cumulative_time_in_seconds"]}
        fields.update(iteration=0, step_is_valid=True, step_is_successful=True, step_is_nonmonotonic=False, linear_solver_iterations=0, gradient_norm=np.inf)
        summary = SimpleNamespace(**fields)
        recorder = RecordSolverIterations()
        self.assertEqual(recorder(summary), pyceres.CallbackReturnType.SOLVER_CONTINUE)
        self.assertEqual(vars(summary), fields)
        self.assertIsNone(recorder.iterations[0]["gradient_norm"])
        self.assertEqual(recorder.iterations[0]["nonfinite_values"], [{"field": "gradient_norm", "value": "inf"}])
        json.dumps(recorder.iterations, allow_nan=False)

class TestDenseHoldoutFeatures(unittest.TestCase):
    """Check full-map identity, numerical equivalence and strict predicted-position coverage."""
    def test_complete_matching_cache_and_content_hash(self):
        """Refuse incomplete, mismatched or corrupted caches without overwriting them."""
        data = np.ones((12, 16, 128), dtype=np.float16)
        provenance = {"images": {"test.png": {"original_size": [32, 24], "feature_shape": list(data.shape)}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.h5"
            with h5py.File(path, "w") as cache:
                cache.attrs["provenance"] = json.dumps(provenance)
                cache.attrs["complete"] = True
                group = cache.create_group("test.png")
                group.create_dataset("features", data=data)
                group.attrs.update(complete=True, scale=[0.5, 0.5], original_size=[32, 24], content_sha256=hashlib.sha256(data.tobytes()).hexdigest())
            self.assertEqual(check_dense_cache(path, provenance)["test.png"]["raw_bytes"], data.nbytes)
            with self.assertRaises(ValueError):
                check_dense_cache(path, {"images": {}})
            with h5py.File(path, "r+") as cache:
                cache.attrs["complete"] = False
            with self.assertRaises(ValueError):
                check_dense_cache(path, provenance)
            with h5py.File(path, "r+") as cache:
                cache.attrs["complete"] = True
                cache["test.png/features"][0, 0, 0] = 2
            with self.assertRaises(ValueError):
                check_dense_cache(path, provenance)

    def test_patch_dense_interpolation_equivalence(self):
        """Match identical float16 patch crops and native subpixel interpolation on a synthetic map."""
        data = np.random.default_rng(0).normal(size=(24, 24, 128)).astype(np.float16)
        with tempfile.TemporaryDirectory() as directory:
            dense_path, sparse_path = Path(directory) / "dense.h5", Path(directory) / "sparse.h5"
            with h5py.File(dense_path, "w") as cache:
                group = cache.create_group("test.png")
                group.create_dataset("features", data=data)
                group.attrs["scale"] = [0.5, 0.5]
            with h5py.File(sparse_path, "w") as cache:
                group = cache.create_group("0/test.png")
                group.create_dataset("corners", data=[[3, 4]])
                group.create_dataset("patches", data=data[4:20, 3:19][None])
                group.create_dataset("scales", data=[[0.5, 0.5]])
            result = compare_dense_and_patch_cache(dense_path, sparse_path, InterpolationConfig(interpolation_default_conf))
            self.assertTrue(result["passed"])
            self.assertEqual(result["max_raw_absolute_difference"], 0)
            self.assertLess(result["max_interpolation_absolute_difference"], 1e-12)
            with h5py.File(sparse_path, "r+") as cache:
                cache["0/test.png/patches"][0, 0, 0, 0] += 1
            with self.assertRaises(ValueError):
                compare_dense_and_patch_cache(dense_path, sparse_path, InterpolationConfig(interpolation_default_conf))

    def test_predicted_sampling_and_boundary_failures(self):
        """Fill an old patch miss and preserve invalid depth, image and bicubic predictions as failures."""
        model, anchors, truth = create_synthetic_scene()
        observation = get_observations(model)[:1]
        point_id, image_id, feature_id, measured = observation[0]
        image = model.images[image_id]
        data = np.zeros((16, 16, 128), dtype=np.float16)
        data[:, :, 0] = 1
        old_data = data[:6, :6].copy()
        old_patch = features.FeaturePatch(old_data, np.zeros(2, dtype=np.int32), np.array([16 / 640, 16 / 480]), False)
        context = {"view": SimpleNamespace(fpatch=lambda image_id, feature_id: old_patch), "references": {point_id: SimpleNamespace(descriptor=np.r_[1., np.zeros(127)])}, "interpolator": features.PatchInterpolator(InterpolationConfig(interpolation_default_conf))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.h5"
            with h5py.File(path, "w") as cache:
                for current in model.images.values():
                    group = cache.create_group(current.name)
                    group.create_dataset("features", data=data)
                    group.attrs["scale"] = [16 / 640, 16 / 480]
            residuals, errors, coverage = sample_dense_holdout(model, observation, path, context)
            self.assertEqual(coverage["valid_count"], 1)
            self.assertEqual(len(coverage["old_patch_misses_filled"]), 1)
            np.testing.assert_allclose(residuals, 0, atol=1e-12)
            # The measured holdout location does not select the descriptor sampling position.
            changed = [(point_id, image_id, feature_id, measured + [1000, 0])]
            repeated, changed_errors, repeated_coverage = sample_dense_holdout(model, changed, path, context)
            np.testing.assert_array_equal(repeated, residuals)
            self.assertGreater(changed_errors[0], errors[0] + 900)
            for xyz, reason in [(np.array([0., 0., -1.]), "nonfinite_or_nonpositive_camera_depth"), (np.array([10000., 0., 10.]), "outside_full_image"), (np.array([(1 - 320) * 10 / 800, 0., 10.]) - image.tvec, "outside_full_bicubic_support")]:
                model.points3D[point_id].xyz = xyz
                residuals, errors, coverage = sample_dense_holdout(model, observation, path, context)
                self.assertEqual(coverage["sample_count"], 1)
                self.assertEqual(coverage["valid_count"], 0)
                self.assertEqual(coverage["invalid_predictions"][0]["reason"], reason)
                self.assertTrue(np.isnan(residuals).all())

    def test_reference_labels_cannot_hide_wrong_descriptor_inputs(self):
        """Reject saved descriptor values that do not come from the declared training projections."""
        model, anchors, truth = create_synthetic_scene()
        point_id = int(anchors["point_ids"][0])
        entry = model.points3D[point_id].track.elements[0]
        reference = SimpleNamespace(track=SimpleNamespace(elements=[entry]), observations=[np.ones(128)], source=entry, descriptor=np.ones(128))
        context = {"references": {point_id: reference}, "view": SimpleNamespace(fpatch=lambda image_id, feature_id: None), "interpolator": SimpleNamespace(interpolate=lambda patch, pixel: np.zeros(128))}
        with self.assertRaisesRegex(ValueError, "Saved reference inputs"):
            audit_training_references(model, [], context)
        heldout = get_observations(model)[:1]
        self.assertNotEqual(get_observation_digest(heldout), get_observation_digest(get_observations(model)))

class RecordBridgeCost(pyceres.CostFunction):
    """Exercise real Ceres buffers, optional derivatives and failed trial evaluations."""
    def __init__(self):
        """Declare 128 residuals and two scalar blocks before any native evaluation."""
        super().__init__()
        self.set_num_residuals(128)
        self.set_parameter_block_sizes([1, 1])
        self.calls = []
        self.reject = False

    def Evaluate(self, parameters, residuals, jacobians):
        """Record requested buffers and reject deliberately invalid test candidates."""
        self.calls.append((float(parameters[0][0]), None if jacobians is None else [block is None for block in jacobians]))
        if self.reject and parameters[0][0] > 0.5:
            return False
        residuals[:] = parameters[0][0] + parameters[1][0] - 1
        if jacobians is not None:
            for block in jacobians:
                if block is not None:
                    block[:] = 1
        return True

class TestModule3Bridge(unittest.TestCase):
    """Check the rebuilt bridge using genuine native cost-only and partial-Jacobian calls."""
    def test_native_quaternion_plus_and_jacobian(self):
        """Check native shapes, zero increments, norm preservation and tangent central differences."""
        manifold = pyceres.QuaternionManifold()
        for q in [np.array([1., 0, 0, 0]), np.array([0.7, -0.2, 0.3, 0.6]) / np.sqrt(0.98)]:
            original = q.copy()
            delta = np.array([0.13, -0.07, 0.11])
            saved_delta = delta.copy()
            zero = manifold.plus(q, np.zeros(3))
            np.testing.assert_allclose(zero, q, atol=1e-15, rtol=0)
            output = manifold.plus(q, delta)
            jacobian = manifold.plus_jacobian(q)
            self.assertEqual(output.shape, (4,))
            self.assertEqual(jacobian.shape, (4, 3))
            self.assertEqual(output.dtype, np.float64)
            self.assertEqual(jacobian.dtype, np.float64)
            self.assertTrue(output.flags.c_contiguous and jacobian.flags.c_contiguous)
            self.assertFalse(np.shares_memory(output, q))
            self.assertLess(abs(np.linalg.norm(output)-1), 1e-12)
            difference = np.column_stack([(manifold.plus(q, axis*1e-6)-manifold.plus(q, -axis*1e-6))/(2e-6) for axis in np.eye(3)])
            np.testing.assert_allclose(jacobian, difference, atol=1e-8, rtol=1e-6)
            np.testing.assert_array_equal(q, original)
            np.testing.assert_array_equal(delta, saved_delta)
        with self.assertRaises(ValueError):
            manifold.plus(np.ones(3), np.zeros(3))
        with self.assertRaises(ValueError):
            manifold.plus_jacobian(np.array([1., np.nan, 0, 0]))

    def test_native_ambient_jacobian_chain_rule(self):
        """Compare ambient native reprojection derivatives along the same manifold's Plus."""
        manifold = pyceres.QuaternionManifold()
        cost = pyceres.factors.BundleAdjustmentCost(1, np.array([320., 240.]))
        for q in [np.array([1., 0, 0, 0]), np.array([0.98, 0.02, -0.04, 0.05])/np.linalg.norm([0.98, 0.02, -0.04, 0.05])]:
            blocks = [q, np.array([0.1, -0.1, 0.2]), np.array([0.3, -0.4, 10.]), np.array([800., 800., 320., 240.])]
            residual, jacobians = cost.evaluate(*blocks)
            local = jacobians[0] @ manifold.plus_jacobian(q)
            columns = []
            for axis in np.eye(3):
                positive = cost.evaluate(manifold.plus(q, axis*1e-6), *blocks[1:])[0]
                negative = cost.evaluate(manifold.plus(q, -axis*1e-6), *blocks[1:])[0]
                columns.append(((positive-negative)/(2e-6)).reshape(-1))
            np.testing.assert_allclose(local, np.column_stack(columns), atol=1e-5, rtol=1e-4)

    def test_first_cost_only_then_partial_jacobians(self):
        """Start with all constant blocks and then request only one block's derivatives."""
        cost = RecordBridgeCost()
        self.assertEqual(cost.num_residuals(), 128)
        first, second = np.array([0.]), np.array([0.])
        problem = pyceres.Problem()
        problem.add_residual_block(cost, None, [first, second])
        problem.set_parameter_block_constant(first)
        problem.set_parameter_block_constant(second)
        options, summary = pyceres.SolverOptions(), pyceres.SolverSummary()
        options.num_threads = 1
        options.minimizer_progress_to_stdout = False
        pyceres.solve(options, problem, summary)
        self.assertIsNone(cost.calls[0][1])
        problem.set_parameter_block_variable(first)
        pyceres.solve(options, problem, summary)
        self.assertTrue(summary.IsSolutionUsable())
        self.assertTrue(any(row[1] == [False, True] for row in cost.calls))
        self.assertAlmostEqual(first[0], 1, places=6)
        residual, jacobians = cost.evaluate(np.array([2.]), np.array([3.]))
        np.testing.assert_array_equal(residual.reshape(-1), np.full(128, 4.))
        self.assertEqual([matrix.shape for matrix in jacobians], [(128, 1), (128, 1)])
        self.assertEqual(second[0], 0)

    def test_failed_trial_is_rejected_without_zero_residual(self):
        """Preserve False across the bridge and let Ceres reject and recover from invalid steps."""
        cost = RecordBridgeCost()
        cost.reject = True
        self.assertEqual(cost.evaluate(np.array([1.]), np.array([0.])), (None, None))
        first, second = np.array([0.]), np.array([0.])
        problem = pyceres.Problem()
        problem.add_residual_block(cost, None, [first, second])
        problem.set_parameter_block_constant(second)
        options, summary = pyceres.SolverOptions(), pyceres.SolverSummary()
        options.max_num_iterations = 20
        options.num_threads = 1
        options.minimizer_progress_to_stdout = False
        pyceres.solve(options, problem, summary)
        self.assertTrue(summary.IsSolutionUsable())
        self.assertGreater(summary.num_unsuccessful_steps, 0)
        self.assertGreater(first[0], 0)
        self.assertLessEqual(first[0], 0.5)
        self.assertTrue(any(row[0] > 0.5 for row in cost.calls))
        self.assertGreater(summary.final_cost, 0)

class TestModule3Policy(unittest.TestCase):
    """Check fixed gauge normalization, guarded feature derivatives and selection semantics."""
    def test_gauge_matrix_invariance_and_robust_weighting(self):
        """Verify common translations, units, lambda scaling and actual-state robust weights."""
        from joint_map.optimization import compute_gauge_information
        model, anchors, truth = create_synthetic_scene()
        covariance = np.diag([0.09, 0.09, 0.04])
        base = compute_gauge_information(anchors, covariance, anchors["means"], 1., "base")
        self.assertTrue(base["passed"])
        doubled = compute_gauge_information(anchors, covariance, anchors["means"], 2., "scaled")
        np.testing.assert_allclose(doubled["matrix"], 2*np.array(base["matrix"]), atol=1e-9, rtol=1e-10)
        shifted = dict(anchors, means=anchors["means"]+[100., -200., 300.])
        changed = compute_gauge_information(shifted, covariance, shifted["means"], 1., "shifted")
        np.testing.assert_allclose(changed["matrix"], base["matrix"], atol=1e-9, rtol=1e-10)
        scaled = dict(anchors, means=anchors["means"]*1000)
        changed = compute_gauge_information(scaled, covariance*1e6, scaled["means"], 1., "units")
        np.testing.assert_allclose(changed["matrix"], base["matrix"], atol=1e-9, rtol=1e-10)
        positions = anchors["means"].copy()
        positions[0] += [100, -100, 200]
        changed = compute_gauge_information(anchors, covariance, positions, 1., "final")
        self.assertLess(changed["robust_weights"][0], 0.01)
        self.assertFalse(np.allclose(changed["matrix"], base["matrix"]))
        degenerate = dict(anchors, means=np.column_stack([np.arange(24), np.zeros((24, 2))]))
        self.assertFalse(compute_gauge_information(degenerate, covariance, degenerate["means"], 1., "line")["passed"])

    def test_gauge_geometric_jacobian_central_difference(self):
        """Check the seven audit columns against the prescribed physical Sim(3) perturbations."""
        point, center, span = np.array([2., 4., -1.]), np.array([0.4, -0.3, 0.1]), 3.2
        x, y, z = point-center
        jacobian = np.column_stack([span*np.eye(3), [[0, z, -y], [-z, 0, x], [y, -x, 0]], point-center])
        def apply_audit_transform(delta):
            """Evaluate only the specified centered physical similarity transformation."""
            return center+span*delta[:3]+np.exp(delta[6])*(Rotation.from_rotvec(delta[3:6]).as_matrix()@(point-center))
        difference = np.column_stack([(apply_audit_transform(axis*1e-6)-apply_audit_transform(-axis*1e-6))/(2e-6) for axis in np.eye(7)])
        np.testing.assert_allclose(jacobian, difference, atol=1e-8, rtol=1e-6)

    def test_selection_preserves_ties_tradeoffs_and_single_candidate(self):
        """Require material dominance and keep incomparable or insufficient candidates unselected."""
        from joint_map.optimization import select_weight_candidates
        def create_candidate(weight, pixel, feature):
            """Construct fixed-sample candidate metrics for decision-rule testing."""
            return {"candidate_id": str(weight), "information_scale": weight, "status": "passed", "heldout_reprojection_px": {"median": pixel, "std": pixel}, "heldout_feature_normalized": {"median": feature, "std": feature}}
        first, same, worse = create_candidate(0.1, 1., 1.), create_candidate(1., 1.001, 1.001), create_candidate(10., 2., 2.)
        self.assertEqual(select_weight_candidates([first])["selection_status"], "insufficient_candidates")
        self.assertEqual(select_weight_candidates([first, same])["selection_status"], "indistinguishable")
        self.assertEqual(select_weight_candidates([first, same, worse])["selection_status"], "indistinguishable")
        self.assertIsNone(select_weight_candidates([first, same, worse])["proposed_weight"])
        self.assertEqual(select_weight_candidates([worse, same, first])["selection_status"], "indistinguishable")
        self.assertEqual(select_weight_candidates([first, worse])["proposed_weight"], 0.1)
        self.assertEqual(select_weight_candidates([first, create_candidate(1., 0.5, 2.)])["selection_status"], "tradeoff")
        self.assertEqual(select_weight_candidates([])["selection_status"], "no_valid_candidate")

    def test_full_reference_and_guarded_feature_chain_rule(self):
        """Use native full maps to audit ordered references, frozen inputs and tangent derivatives."""
        from joint_map.optimization import create_dense_feature_context, save_training_references, GuardImageCost, build_shared_problem, project_observations
        from pixsfm import residuals as native_residuals
        model, anchors, truth = create_synthetic_scene()
        training, heldout = create_holdout(model, 0.2, 0)
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory)/"full.h5"
            with h5py.File(cache_path, "w") as cache:
                yy, xx, channels = np.indices((48, 64, 128))
                for image_id, image in model.images.items():
                    group = cache.create_group(image.name)
                    data = (0.1+np.sin(channels*0.07+xx*0.015+yy*0.011+image_id*0.02)).astype(np.float16)
                    group.create_dataset("features", data=data)
                    group.attrs["scale"] = [0.1, 0.1]
            context = create_dense_feature_context(training, cache_path)
            provenance = save_training_references(training, heldout, context, Path(directory)/"refs.npz", PROJECT_ROOT)
            self.assertEqual(provenance["actual_participant_overlap_with_holdout"], 0)
            self.assertEqual(len(provenance["participants"]), len(get_observations(training)))
            row = get_observations(training)[0]
            point_id, image_id, feature_id, measured = row
            image = training.images[image_id]
            camera = training.cameras[image.camera_id]
            blocks = [image.qvec.copy(), image.tvec.copy(), training.points3D[point_id].xyz.copy(), camera.params.copy()]
            patch = context["view"].fpatch(image_id, feature_id)
            reference = np.asarray(context["references"][point_id].descriptor).reshape(-1)
            native = native_residuals.FeatureReferenceCostFunctor(camera.model_id, patch, reference.reshape(1, 128), context["interpolation"])
            guard = GuardImageCost(native, row[:3], patch, reference)
            value, jacobians = guard.evaluate(*blocks)
            direct = context["interpolator"].interpolate(patch, project_observations(training, [row])[0]).reshape(-1)-reference
            np.testing.assert_allclose(value.reshape(-1), direct, atol=1e-10, rtol=1e-8)
            manifold = pyceres.QuaternionManifold()
            difference = np.column_stack([((guard.evaluate(manifold.plus(blocks[0], axis*1e-6), *blocks[1:])[0]-guard.evaluate(manifold.plus(blocks[0], -axis*1e-6), *blocks[1:])[0])/(2e-6)).reshape(-1) for axis in np.eye(3)])
            np.testing.assert_allclose(jacobians[0]@manifold.plus_jacobian(blocks[0]), difference, atol=1e-5, rtol=1e-4)
            saved = reference.copy()
            guard.reference = reference+0.02
            changed = guard.evaluate(*blocks)[0]
            np.testing.assert_allclose(changed-value, -0.02, atol=1e-10, rtol=1e-8)
            guard.reference = saved
            samples = guard.samples
            for bad in [np.array([0., 0., -2.]), np.array([1e8, 0., 10.]), np.array([np.nan, 0., 10.])]:
                self.assertEqual(guard.evaluate(blocks[0], blocks[1], bad, blocks[3]), (None, None))
            self.assertEqual(guard.samples, samples)
            scales = {"sigma_horizontal_m": 0.3, "sigma_height_m": 0.2, "sigma_pixel_px": 0.2, "sigma_feature_norm": 0.1, "lambda_feature": 200.}
            state = build_shared_problem(training, anchors, scales, 1., context=context)
            self.assertEqual(state["audit"]["feature_blocks"], len(get_observations(training)))
            self.assertEqual(state["audit"]["restore_blocks"], 0)

class TestModule3Execution(unittest.TestCase):
    """Check candidate failure isolation, frozen input gates and explicit user confirmation."""
    def test_global_preparation_exception_is_recorded_without_try_except(self):
        """Contain a nonzero preparation process and mark untouched candidates not_run."""
        from unittest.mock import patch
        from joint_map.optimization import run_prior_validation, write_validation_json
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/"shared").mkdir()
            (output/"logs").mkdir()
            report = {"status": "running", "run_dir": str(output), "candidates": [{"status": "pending", "candidate_id": "weight_0p1"}]}
            write_validation_json(output/"shared/prior_validation.json", report)
            with patch("joint_map.optimization.subprocess.run", return_value=SimpleNamespace(returncode=1)):
                failed = run_prior_validation({}, PROJECT_ROOT, output, write_validation_json)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["stop_scope"], "global")
            self.assertEqual(failed["candidates"][0]["status"], "not_run")
            self.assertIsNone(failed["candidates"][0]["heldout_reprojection_px"])
            self.assertEqual(json.loads((output/"shared/prior_model.json").read_text())["status"], "unvalidated")

    def test_candidate_failures_continue_and_execution_errors_stop(self):
        """Distinguish normal failures, native crashes and stale worker identities in the parent."""
        from unittest.mock import patch
        from joint_map.optimization import execute_candidate_loop, write_validation_json, get_sha256
        for mode in ["normal_failure", "crash", "stale"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                write_validation_json(run_dir/"run.json", {"frozen_files": {}})
                labels = ["weight_0p1", "weight_1p0", "weight_10p0"]
                report = {"run_id": "test_run", "policy_sha256": "policy", "status": "running", "candidates": [{"candidate_id": label, "information_scale": weight, "status": "pending"} for label, weight in zip(labels, [0.1, 1., 10.])]}
                calls = []
                def run_test_worker(command, **kwargs):
                    """Emit one controlled worker outcome without invoking a numerical solve."""
                    index = len(calls)
                    calls.append(command)
                    result = {"candidate_id": labels[index], "run_id": "wrong_run" if mode == "stale" else "test_run", "policy_sha256": "policy", "run_inputs_sha256": get_sha256(run_dir/"run.json"), "information_scale": float(command[-1]), "status": "failed" if index == 0 else "passed", "coverage": {"valid_count": None if index == 0 else 777}, "heldout_reprojection_px": None if index == 0 else {"median": 1.}}
                    write_validation_json(run_dir/labels[index]/"result.json", result)
                    return SimpleNamespace(returncode=139 if mode == "crash" else 0)
                with patch("joint_map.optimization.subprocess.run", side_effect=run_test_worker):
                    execute_candidate_loop(report, run_dir, PROJECT_ROOT, run_dir/"report.json", write_validation_json)
                if mode == "normal_failure":
                    self.assertEqual(len(calls), 3)
                    self.assertEqual([row["status"] for row in report["candidates"]], ["failed", "passed", "passed"])
                    self.assertIsNone(report["candidates"][0]["heldout_reprojection_px"])
                else:
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(report["stop_scope"], "global")
                    self.assertEqual([row["status"] for row in report["candidates"]], ["aborted", "not_run", "not_run"])

    def test_guarded_native_feature_rejects_trials_before_sampling(self):
        """Run genuine Ceres feature steps that leave support and recover without sampling invalid pixels."""
        from joint_map.optimization import GuardImageCost, get_candidate_projection, check_feature_domain
        from pixsfm import residuals as native_residuals
        data = np.ones((8, 8, 128), dtype=np.float16)
        data[:, :, 0] = np.arange(8, dtype=np.float16)[None, :]
        patch = features.FeaturePatch(data, np.zeros(2, dtype=np.int32), np.ones(2), False)
        interpolation = InterpolationConfig(interpolation_default_conf)
        reference = np.ones(128)
        reference[0] = 100
        reference /= np.linalg.norm(reference)
        native = native_residuals.FeatureReferenceCostFunctor(1, patch, reference.reshape(1, 128), interpolation)
        sampled = []
        def evaluate_native(*blocks):
            """Count only real native feature calls and independently check their support."""
            pixel, reason = get_candidate_projection(blocks)
            self.assertIsNone(reason)
            self.assertTrue(check_feature_domain(patch, pixel))
            sampled.append(pixel.copy())
            return native.evaluate(*blocks)
        cost = GuardImageCost(SimpleNamespace(evaluate=evaluate_native), (1, 1, 1), patch, reference)
        q, translation, point, intrinsics = np.array([1., 0, 0, 0]), np.zeros(3), np.array([0., 0., 10.]), np.array([1., 1., 4., 4.])
        problem = pyceres.Problem()
        problem.add_residual_block(cost, None, [q, translation, point, intrinsics])
        for block in [q, point, intrinsics]:
            problem.set_parameter_block_constant(block)
        manifold = pyceres.SubsetManifold(3, [1, 2])
        problem.set_manifold(translation, manifold)
        options, summary = pyceres.SolverOptions(), pyceres.SolverSummary()
        options.num_threads, options.max_num_iterations, options.minimizer_progress_to_stdout = 1, 25, False
        pyceres.solve(options, problem, summary)
        self.assertTrue(summary.IsSolutionUsable())
        self.assertGreater(cost.rejected, 0)
        self.assertEqual(len(sampled), cost.samples)
        self.assertEqual(cost.calls, cost.samples+cost.rejected)
        self.assertGreater(translation[0], 0)
        self.assertLess(summary.final_cost, summary.initial_cost)
        pixel, reason = get_candidate_projection([q, translation, point, intrinsics])
        self.assertTrue(check_feature_domain(patch, pixel))
        sampled.clear()
        point[2] = -10
        pyceres.solve(options, problem, summary)
        self.assertFalse(summary.IsSolutionUsable())
        self.assertEqual(len(sampled), 0)

    def test_frozen_prior_lineage_and_single_shared_map_initialization(self):
        """Keep original link indices and train-only mappings while mutating only anchored XYZ views."""
        from joint_map.optimization import initialize_map_points, write_map_priors
        model, anchors, truth = create_synthetic_scene()
        training, heldout = create_holdout(model, 0.2, 0)
        anchors = {"point_ids": anchors["point_ids"][:12], "means": anchors["means"][:12], "map_feature_ids": np.arange(12), "dom_pixels": np.zeros((12, 2)), "selected_link_indices": np.arange(12)+100, "source_prior_ids": np.arange(12)+200}
        original = {point_id: point.xyz.copy() for point_id, point in training.points3D.items()}
        addresses = [training.points3D[int(point_id)].xyz.ctypes.data for point_id in anchors["point_ids"]]
        lineage = initialize_map_points(training, anchors)
        self.assertEqual(lineage["initialization_source"], "map_mu")
        for point_id, mean, address in zip(anchors["point_ids"], anchors["means"], addresses):
            np.testing.assert_array_equal(training.points3D[int(point_id)].xyz, mean)
            self.assertEqual(training.points3D[int(point_id)].xyz.ctypes.data, address)
        for point_id in set(training.points3D)-set(anchors["point_ids"]):
            np.testing.assert_array_equal(training.points3D[point_id].xyz, original[point_id])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"priors.npz"
            write_map_priors(training, anchors, {"sigma_horizontal_m": 0.3, "sigma_height_m": 0.2}, path)
            with np.load(path) as priors:
                np.testing.assert_array_equal(priors["source_link_indices"], anchors["selected_link_indices"])
                np.testing.assert_array_equal(priors["source_prior_ids"], anchors["source_prior_ids"])
                np.testing.assert_array_equal(priors["prior_ids"], np.arange(12))
                actual = {(int(priors["shared_point_ids"][index]), int(image_id), int(feature_id)) for index, image_id, feature_id in zip(priors["observation_prior_indices"], priors["observation_image_ids"], priors["observation_point2D_ids"])}
                self.assertFalse(actual & {row[:3] for row in heldout})

    def test_input_fingerprint_and_confirmation_gates(self):
        """Reject changed evidence and preserve the immutable report during concrete confirmation."""
        from joint_map.optimization import read_frozen_candidate_inputs, write_validation_json, get_sha256, confirm_prior_candidate
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            shared = output/"shared"
            shared.mkdir()
            for name in ["map_priors.npz", "training_references.npz"]:
                (shared/name).write_bytes(b"fixed test evidence")
            report = {"status": "complete", "candidates": [{"candidate_id": "weight_1p0", "information_scale": 1., "status": "passed"}, {"candidate_id": "weight_10p0", "information_scale": 10., "status": "failed"}]}
            write_validation_json(shared/"prior_validation.json", report)
            model = {"status": "validated", "map_prior_scope": "training_frozen", "validation_sha256": get_sha256(shared/"prior_validation.json"), "map_priors_sha256": get_sha256(shared/"map_priors.npz"), "reference_sha256": get_sha256(shared/"training_references.npz")}
            write_validation_json(shared/"prior_model.json", model)
            with self.assertRaises(ValueError):
                confirm_prior_candidate(output, "weight_10p0", "test user text")
            confirmed = confirm_prior_candidate(output, "weight_1p0", "test user explicitly selects weight 1")
            self.assertEqual(confirmed["status"], "confirmed")
            self.assertEqual(confirmed["lambda_map"], 1.)
            self.assertEqual(get_sha256(shared/"prior_validation.json"), model["validation_sha256"])
            write_validation_json(output/"run.json", {"frozen_files": {str(shared/"map_priors.npz"): model["map_priors_sha256"]}})
            read_frozen_candidate_inputs(output)
            (shared/"map_priors.npz").write_bytes(b"changed evidence")
            with self.assertRaises(ValueError):
                read_frozen_candidate_inputs(output)

    def test_feasibility_recovery_and_removal_from_final_problem(self):
        """Recover an off-image training point and keep all temporary factors out of the final BA."""
        from joint_map.optimization import restore_feature_domain, check_model_domain
        model, anchors, truth = create_synthetic_scene()
        scales = {"sigma_horizontal_m": 0.3, "sigma_height_m": 0.2, "sigma_pixel_px": 0.2, "sigma_feature_norm": 0.1, "lambda_feature": 200.}
        feature_set = features.FeatureSet_f16(128)
        arrays, maps = [], []
        for image in model.images.values():
            array = np.ones((1, 48, 64, 128), dtype=np.float16)
            feature_map = features.FeatureMap(array, [0], [np.zeros(2, dtype=np.int32)], {"is_sparse": False, "scale": np.array([0.1, 0.1])})
            feature_set.add_fmap(image.name, feature_map)
            arrays.append(array)
            maps.append(feature_map)
        context = {"feature_set": feature_set, "arrays": arrays, "maps": maps, "view": features.FeatureView(feature_set, model), "interpolation": InterpolationConfig(interpolation_default_conf), "references": {point_id: SimpleNamespace(descriptor=np.ones(128)/np.sqrt(128)) for point_id in model.points3D}}
        point_id = int(anchors["point_ids"][0])
        model.points3D[point_id].xyz[:] = [100., 0., 10.]
        count = len(get_observations(model))
        self.assertGreater(check_model_domain(model, get_observations(model), context)["invalid_count"], 0)
        restored = restore_feature_domain(model, anchors, scales, 1., context, "test")
        self.assertTrue(restored["restored"])
        self.assertEqual(check_model_domain(model, get_observations(model), context)["invalid_count"], 0)
        self.assertEqual(len(get_observations(model)), count)
        final = build_shared_problem(model, anchors, scales, 1., context=context)
        self.assertEqual(final["audit"]["restore_blocks"], 0)
        self.assertEqual(final["audit"]["total_blocks"], 2*count+len(anchors["point_ids"]))
        from unittest.mock import patch
        model.points3D[point_id].xyz[:] = [100., 0., 10.]
        with patch("joint_map.optimization.solve_shared_problem", return_value={"usable": True, "converged": False}):
            failed = restore_feature_domain(model, anchors, scales, 1., context, "test_budget")
        self.assertFalse(failed["restored"])
        self.assertEqual([row["beta"] for row in failed["stages"]], [1., 10., 100.])
        self.assertEqual(len(get_observations(model)), count)

class TestMultiscaleDerivative(unittest.TestCase):
    """Verify numerical reference independence, rounding protection and actual native regressions."""
    def create_sample_levels(self, function):
        """Generate the fixed seven-level 128-channel samples for controlled scalar functions."""
        center = np.broadcast_to(function(0.), (128,)).copy()
        levels = [{"step": 1e-6/(2**k), "valid": True, "reason": None, "same_segment": True, "values_minus": np.broadcast_to(function(-1e-6/(2**k)), (128,)).tolist(), "values_plus": np.broadcast_to(function(1e-6/(2**k)), (128,)).tolist()} for k in range(7)]
        return center, levels

    def create_native_feature(self, principal_x, corner=(0, 0)):
        """Retain every native owner for a smooth normalized 128-channel feature field."""
        from pixsfm import residuals as native_residuals
        data = np.ones((8, 8, 128), dtype=np.float16)
        data[:, :, 0] = np.arange(8, dtype=np.float16)[None]
        patch = features.FeaturePatch(data, np.array(corner, dtype=np.int32), np.ones(2), False)
        interpolation = InterpolationConfig(interpolation_default_conf)
        reference = np.zeros((1, 128))
        native = native_residuals.FeatureReferenceCostFunctor(1, patch, reference, interpolation)
        feature = SimpleNamespace(native=native, patch=patch, identity=(1, 1, 1), owners=(data, interpolation, reference))
        blocks = [np.array([1., 0, 0, 0]), np.zeros(3), np.array([0., 0., 1.]), np.array([1., 1., principal_x, corner[1]+3.5])]
        return feature, blocks, pyceres.QuaternionManifold(), interpolation

    def test_richardson_reference_is_independent_of_analytic_values(self):
        """Cancel the central-difference quadratic error and preserve the same window for wrong derivatives."""
        from joint_map.optimization import select_multiscale_reference, classify_multiscale_derivative
        center, levels = self.create_sample_levels(lambda value: 2*value+1e9*value**3)
        numerical = select_multiscale_reference(center, levels)
        self.assertEqual(numerical["status"], "ready")
        self.assertEqual(numerical["selected_window"], 2)
        np.testing.assert_allclose(numerical["reference"], 2., atol=1e-12, rtol=0)
        before = json.dumps(numerical, sort_keys=True)
        self.assertEqual(classify_multiscale_derivative(numerical, np.full(128, 2.))["status"], "passed")
        self.assertEqual(classify_multiscale_derivative(numerical, np.full(128, 3.))["status"], "mismatch")
        self.assertEqual(json.dumps(numerical, sort_keys=True), before)

    def test_legitimate_zero_derivatives_pass(self):
        """Accept exact low-magnitude constants and a symmetric quadratic at its zero derivative."""
        from joint_map.optimization import select_multiscale_reference, classify_multiscale_derivative
        for function in [lambda value: 0., lambda value: value**2]:
            with self.subTest(function=function):
                numerical = select_multiscale_reference(*self.create_sample_levels(function))
                self.assertEqual(numerical["status"], "ready")
                self.assertEqual(classify_multiscale_derivative(numerical, np.zeros(128))["status"], "passed")
                self.assertGreater(min(numerical["error"]), 0)

    def test_high_bias_roundoff_platform_is_inconclusive(self):
        """Reject false certainty when both a large constant and a sloped function round to identical samples."""
        from joint_map.optimization import select_multiscale_reference, classify_multiscale_derivative
        for function in [lambda value: 1e12, lambda value: 1e12+value]:
            numerical = select_multiscale_reference(*self.create_sample_levels(function))
            self.assertEqual(numerical["status"], "inconclusive")
            self.assertIsNone(numerical["selected_window"])
            self.assertIn("rounding_limited", numerical["reason"])
            self.assertEqual(classify_multiscale_derivative(numerical, None)["status"], "inconclusive")
            self.assertGreater(numerical["windows"][0]["E_round"][0], 1)

    def test_one_unresolved_channel_blocks_the_shared_window(self):
        """Keep all channels when only one has an unresolvable rounding budget."""
        from joint_map.optimization import select_multiscale_reference
        center, levels = self.create_sample_levels(lambda value: np.r_[np.full(127, value), 1e12+value])
        numerical = select_multiscale_reference(center, levels)
        self.assertEqual(numerical["status"], "inconclusive")
        self.assertIsNone(numerical["reference"])
        self.assertLess(numerical["windows"][0]["E"][0], 1e-6)
        self.assertGreater(numerical["windows"][0]["E"][127], 1e-6)

    def test_unrepresentable_parameter_steps_never_become_zero_derivatives(self):
        """Invalidate collapsed actual input increments without fabricating numerical samples."""
        from joint_map.optimization import check_actual_perturbation, select_multiscale_reference
        original = np.array([1e20, 0., 0.])
        axis = np.array([1., 0., 0.])
        checked = check_actual_perturbation(original, original-axis*1e-4, original+axis*1e-4, 1, axis, 1e-4)
        self.assertFalse(checked["valid"])
        center, levels = self.create_sample_levels(lambda value: 0.)
        for level in levels:
            level.update(valid=False, reason=checked["reason"], values_minus=None, values_plus=None)
        numerical = select_multiscale_reference(center, levels)
        self.assertEqual(numerical["status"], "inconclusive")
        self.assertTrue(all(row["D"] is None for row in numerical["levels"]))

    def test_native_quaternion_chord_representation_at_every_scale(self):
        """Validate the native quaternion convention for identity and general unit inputs at all seven steps."""
        from joint_map.optimization import check_actual_perturbation
        manifold = pyceres.QuaternionManifold()
        for q in [np.array([1., 0, 0, 0]), np.array([0.7, -0.2, 0.3, 0.6])/np.sqrt(0.98)]:
            for axis in np.eye(3):
                for k in range(7):
                    step = 1e-6/(2**k)
                    result = check_actual_perturbation(q, manifold.plus(q, -axis*step), manifold.plus(q, axis*step), 0, axis, step)
                    self.assertTrue(result["valid"])

    def test_coarse_cell_crossing_uses_the_first_finer_common_window(self):
        """Use global cell indices and retain the same observation when a coarse stencil crosses a knot."""
        from joint_map.optimization import verify_multiscale_direction
        feature, blocks, manifold, interpolation = self.create_native_feature(5.50006, (2, 3))
        result = verify_multiscale_direction(feature, blocks, 1, np.array([1., 0., 0.]), manifold, interpolation)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["center_cells"], [[5, 6]])
        self.assertFalse(result["numerical"]["levels"][0]["same_segment"])
        self.assertEqual(result["numerical"]["selected_window"], 3)
        json.dumps(result, allow_nan=False)

    def test_all_windows_crossing_a_knot_remain_inconclusive(self):
        """Stop at unresolved interpolation segments without switching to one-sided differences."""
        from joint_map.optimization import verify_multiscale_direction
        feature, blocks, manifold, interpolation = self.create_native_feature(3.5)
        result = verify_multiscale_direction(feature, blocks, 1, np.array([1., 0., 0.]), manifold, interpolation)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("cross_segment", result["reason"])
        self.assertIsNone(result["comparison"])
        blocks[2][2] = -1
        invalid = verify_multiscale_direction(feature, blocks, 1, np.array([1., 0., 0.]), manifold, interpolation)
        self.assertEqual(invalid["reason"], "invalid_center_projection")
        self.assertIsNone(invalid["comparison"])

    def test_uncertainty_crossing_the_tolerance_is_not_a_mismatch(self):
        """Separate borderline uncertainty from a confirmed discrepancy and an accepted match."""
        from joint_map.optimization import classify_multiscale_derivative
        numerical = {"status": "ready", "reference": [0.]*128, "error": [3e-7]*128}
        analytic = np.zeros(128)
        analytic[0] = 9.9e-6
        result = classify_multiscale_derivative(numerical, analytic)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["channels"][0], "inconclusive")
        analytic[0] = 1e-3
        self.assertEqual(classify_multiscale_derivative(numerical, analytic)["status"], "mismatch")
        self.assertEqual(classify_multiscale_derivative(numerical, np.zeros(128))["status"], "passed")

    def test_original_failed_native_direction_under_the_new_fixed_policy(self):
        """Reproduce the saved old failure and verify all 128 channels with retained native owners."""
        from pixsfm import residuals as native_residuals
        from joint_map.optimization import verify_multiscale_direction
        case_path = PROJECT_ROOT/"outputs/joint_map/shared/candidates/20261109_155215_383991/derivative_case.npz"
        with np.load(case_path) as case:
            blocks = [case[name].copy() for name in ["q", "translation", "point", "intrinsics"]]
            identity = case["identity"].copy()
            axis = case["axis"].copy()
            saved = case["analytic"].copy()
        model = pycolmap.Reconstruction(str(PROJECT_ROOT/"outputs/joint_map/joint/model"))
        name = model.images[int(identity[1])].name
        with h5py.File(PROJECT_ROOT/"outputs/joint_map/shared/holdout_dense_features.h5", "r") as cache:
            data = cache[name]["features"][:]
            scale = np.asarray(cache[name].attrs["scale"], dtype=np.float64)
        patch = features.FeaturePatch(data, np.zeros(2, dtype=np.int32), scale, False)
        interpolation = InterpolationConfig(interpolation_default_conf)
        reference = np.zeros((1, 128))
        native = native_residuals.FeatureReferenceCostFunctor(1, patch, reference, interpolation)
        feature = SimpleNamespace(native=native, patch=patch, identity=identity.tolist())
        result = verify_multiscale_direction(feature, blocks, 0, axis, pyceres.QuaternionManifold(), interpolation)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["numerical"]["selected_window"], 2)
        np.testing.assert_array_equal(result["comparison"]["analytic"], saved)
        self.assertEqual(result["comparison"]["channels"], ["passed"]*128)

class TestFormalJointOptimization(unittest.TestCase):
    """Check confirmed full-observation solving, serialization and the sparse-only handoff."""
    def get_scales(self):
        """Provide finite synthetic scales without estimating or tuning the formal model."""
        return {"sigma_horizontal_m": 0.3, "sigma_height_m": 0.2, "sigma_pixel_px": 0.2, "sigma_feature_norm": 0.1, "lambda_feature": 200., "lambda_map": 1.}

    def create_report(self):
        """Create explicit not_run states for the two required solver stages."""
        return {"status": "running", "error": None, "geometry": {"executed": False, "status": "not_run"}, "feature": {"executed": False, "status": "not_run"}}

    def test_canonical_fingerprint_survives_serialization(self):
        """Preserve exact geometry and observation identities independently of native map iteration order."""
        from joint_map.joint import get_formal_fingerprint, check_serialized_model
        model, anchors, truth = create_synthetic_scene()
        model.images[1].qvec = [0.9993436263458957, 0.033605434720501745, -0.01190332192579901, 0.006426676140319738]
        with tempfile.TemporaryDirectory() as directory:
            model.write(directory)
            restored = pycolmap.Reconstruction(directory)
            self.assertTrue(check_serialized_model(model, restored)["passed"])
            self.assertNotEqual(get_formal_fingerprint(model)["poses"], get_formal_fingerprint(restored)["poses"])
            restored.images[1].qvec[1] += 1e-7
            with self.assertRaises(ValueError):
                check_serialized_model(model, restored)
            restored = pycolmap.Reconstruction(directory)
            image = restored.images[sorted(restored.images)[0]]
            image.points2D[0].xy[0] += 0.1
            self.assertNotEqual(get_formal_fingerprint(model), get_formal_fingerprint(restored))
            with self.assertRaises(ValueError):
                check_serialized_model(model, restored)

    def test_registered_visual_geometry_and_map_start_are_verified(self):
        """Reject real geometry changes while reconstructing exactly the registered map initialization."""
        from joint_map.joint import check_registered_visual_model, prepare_formal_initialization
        from joint_map.optimization import initialize_map_points
        native, anchors, truth = create_synthetic_scene()
        visual = copy.deepcopy(native)
        rotation = Rotation.from_euler("xyz", [8., -4., 17.], degrees=True).as_matrix()
        transform = (1.2, rotation, np.array([4., -2., 1.]))
        with tempfile.TemporaryDirectory() as directory:
            write_initialized_model(visual, transform, Path(directory))
            visual = pycolmap.Reconstruction(directory)
        registration = {"scale": transform[0], "rotation": rotation.tolist(), "translation_enu_m": transform[2].tolist()}
        check_registered_visual_model(visual, native, registration)
        initial = copy.deepcopy(visual)
        lineage = initialize_map_points(initial, anchors)
        registration["joint_map_initialization"] = {"lineage": lineage}
        model, actual = prepare_formal_initialization(visual, anchors, registration)
        np.testing.assert_array_equal(actual["initial_xyz_enu_m"], anchors["means"])
        visual.points3D[int(anchors["point_ids"][0])].xyz[0] += 0.001
        with self.assertRaises(ValueError):
            check_registered_visual_model(visual, native, registration)

    def test_full_joint_solve_preserves_shared_variables_and_lineage(self):
        """Solve three factor families on all observations and persist no fabricated dense evidence."""
        from joint_map.joint import solve_formal_problem, save_sparse_lineage, get_formal_fingerprint, check_serialized_model
        from joint_map.optimization import create_dense_feature_context, initialize_map_points
        visual, anchors, truth = create_synthetic_scene()
        anchors["prior_ids"] = np.arange(len(anchors["point_ids"]), dtype=np.int64)
        source = get_formal_fingerprint(visual)
        model = copy.deepcopy(visual)
        initialize_map_points(model, anchors)
        report = self.create_report()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cache_path = directory/"features.h5"
            with h5py.File(cache_path, "w") as cache:
                for image in visual.images.values():
                    group = cache.create_group(image.name)
                    group.create_dataset("features", data=np.ones((48, 64, 128), dtype=np.float16))
                    group.attrs["scale"] = [0.1, 0.1]
            context = create_dense_feature_context(visual, cache_path)
            solve_formal_problem(model, anchors, self.get_scales(), context, report, lambda current: None)
            self.assertTrue(report["optimization_complete"])
            self.assertEqual(report["status"], "running")
            self.assertTrue(report["geometry"]["converged"])
            self.assertTrue(report["feature"]["converged"])
            self.assertEqual(report["final_domain"]["valid_count"], 144)
            self.assertEqual(report["feature"]["audit"]["feature_blocks"], 144)
            self.assertEqual(report["feature"]["audit"]["geometric_blocks"], 144)
            self.assertEqual(report["feature"]["audit"]["map_blocks"], 24)
            self.assertEqual(report["feature"]["audit"]["restore_blocks"], 0)
            self.assertEqual(get_formal_fingerprint(visual), source)
            self.assertGreater(report["all_xyz_displacement_m"]["median"], 0)
            (directory/"model").mkdir()
            model.write(str(directory/"model"))
            self.assertTrue(check_serialized_model(model, pycolmap.Reconstruction(str(directory/"model")))["passed"])
            save_sparse_lineage(directory/"lineage.npz", visual, model, anchors)
            with np.load(directory/"lineage.npz") as lineage:
                np.testing.assert_array_equal(lineage["initial_xyz_enu_m"], anchors["means"])
                self.assertGreater(np.max(np.abs(lineage["optimized_xyz_enu_m"]-lineage["initial_xyz_enu_m"])), 0)
                self.assertEqual(lineage["observation_image_ids"].size, 144)
                self.assertEqual(lineage["dense_agreement_residuals"].size, 0)
                self.assertEqual(lineage["dense_support_status"].item(), "not_run")
                self.assertEqual(lineage["cell_mapping_status"].item(), "pending_dense_update")
            json.dumps(report, allow_nan=False)

    def test_nonconvergence_stops_before_feature_solving(self):
        """Keep the feature stage not_run when the required geometric stage is not converged."""
        from unittest.mock import patch
        from joint_map.joint import solve_formal_problem
        from joint_map.optimization import initialize_map_points
        model, anchors, truth = create_synthetic_scene()
        initialize_map_points(model, anchors)
        report = self.create_report()
        with patch("joint_map.joint.solve_shared_problem", return_value={"usable": True, "converged": False, "termination_message": "controlled test budget"}) as solver:
            solve_formal_problem(model, anchors, self.get_scales(), None, report, lambda current: None)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["feature"]["executed"])
        self.assertEqual(solver.call_count, 1)
        self.assertFalse(report.get("optimization_complete", False))

    def test_invalid_initial_depth_stops_without_solving(self):
        """Reject nonpositive depth before constructing or solving the formal geometry problem."""
        from unittest.mock import patch
        from joint_map.joint import solve_formal_problem
        from joint_map.optimization import initialize_map_points
        model, anchors, truth = create_synthetic_scene()
        anchors["means"][0, 2] = -10
        initialize_map_points(model, anchors)
        report = self.create_report()
        with patch("joint_map.joint.solve_shared_problem") as solver:
            solve_formal_problem(model, anchors, self.get_scales(), None, report, lambda current: None)
        self.assertEqual(report["status"], "failed")
        self.assertGreater(report["geometry_initial_domain"]["invalid_count"], 0)
        solver.assert_not_called()
        self.assertFalse(report["feature"]["executed"])

    def test_unconfirmed_prior_and_abnormal_process_are_rejected(self):
        """Block missing confirmation and record a controlled native process failure honestly."""
        from unittest.mock import patch
        from joint_map.joint import load_confirmed_handoff, run_joint_optimization
        from joint_map.optimization import write_validation_json
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/"shared").mkdir()
            (output/"joint").mkdir()
            (output/"logs").mkdir()
            write_validation_json(output/"shared/prior_model.json", {"status": "validated"})
            with self.assertRaises(ValueError):
                load_confirmed_handoff(output)
            write_validation_json(output/"joint/diagnostics.json", {"status": "running", "run_dir": str(output), "error": None})
            with patch("joint_map.joint.subprocess.run", return_value=SimpleNamespace(returncode=139)):
                failed = run_joint_optimization({}, PROJECT_ROOT, output, write_validation_json)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["exit_code"], 139)

    def test_joint_cli_stage_is_available(self):
        """Dispatch the explicit joint stage through the existing public command parser."""
        from unittest.mock import patch
        from run_joint_map import run_main
        with patch("sys.argv", ["run_joint_map.py", "--config", "configs/joint_map.json", "--stage", "joint"]), patch("run_joint_map.run_map_stage") as dispatch:
            run_main()
        dispatch.assert_called_once_with((PROJECT_ROOT/"configs/joint_map.json").resolve(), "joint")

class TestDensePreparation(unittest.TestCase):
    """Check metric transforms, native binary identities and the explicit dense branch interface."""
    def test_metric_cloud_preserves_order_normals_and_color(self):
        """Apply Sim(3) only to XYZ and rotation only to normals with exact RGB identity."""
        from joint_map.dense import write_metric_cloud, read_fused_cloud
        dtype = [(key, "<f4") for key in ["x", "y", "z", "nx", "ny", "nz"]]+[(key, "u1") for key in ["red", "green", "blue"]]
        points = np.array([(1, 2, 3, 0, 0, 1, 10, 20, 30), (-4, 5, 6, 1, 0, 0, 40, 50, 60)], dtype=dtype)
        rotation = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        translation = np.array([20., -30., 40.])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cloud.ply"
            result = write_metric_cloud(path, points, 2., rotation, translation)
            restored = read_fused_cloud(path)
        np.testing.assert_array_equal(restored, result)
        np.testing.assert_array_equal(np.column_stack([result[key] for key in ["x", "y", "z"]]), 2*np.array([[1, 2, 3], [-4, 5, 6]])@rotation.T+translation)
        np.testing.assert_array_equal(np.column_stack([result[key] for key in ["nx", "ny", "nz"]]), np.array([[0, 0, 1], [1, 0, 0]])@rotation.T)
        for key in ["red", "green", "blue"]:
            np.testing.assert_array_equal(result[key], points[key])

    def test_image_order_uses_binary_records(self):
        """Keep noncontiguous IDs and record order independently of image names and numeric sorting."""
        import struct
        from joint_map.dense import read_image_order
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory)/"images.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", 2))
                for index, name in [(42, "z.png"), (7, "a.png")]:
                    stream.write(struct.pack("<i7di", index, 1., 0, 0, 0, 0, 0, 0, 3))
                    stream.write(name.encode()+b"\0"+struct.pack("<Q", 1)+struct.pack("<ddq", 2., 4., -1))
            rows = read_image_order(Path(directory))
        self.assertEqual([(row["image_idx"], row["image_id"], row["name"]) for row in rows], [(0, 42, "z.png"), (1, 7, "a.png")])

    def test_visibility_rejects_corrupt_identities(self):
        """Reject mismatched point counts, duplicate indices and indices outside the native image order."""
        import struct
        from joint_map.dense import read_visibility
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cloud.vis"
            path.write_bytes(struct.pack("<QI2I", 1, 2, 5, 0))
            points, images = read_visibility(path, 1, 6)
            np.testing.assert_array_equal(points, [0, 0])
            np.testing.assert_array_equal(images, [5, 0])
            with self.assertRaises(ValueError):
                read_visibility(path, 2, 6)
            for indices in [(2, 2), (6, 0)]:
                path.write_bytes(struct.pack("<QI2I", 1, 2, *indices))
                with self.assertRaises(ValueError):
                    read_visibility(path, 1, 6)

    def test_depth_array_preserves_native_layout_and_rejects_truncation(self):
        """Restore native planar channel layout without coordinate clipping or invalid-value replacement."""
        from joint_map.dense import read_depth_map
        array = np.arange(36, dtype="<f4").reshape(3, 4, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"normal.bin"
            raw = b"4&3&3&"+array.transpose(1, 0, 2).tobytes(order="F")
            path.write_bytes(raw)
            np.testing.assert_array_equal(read_depth_map(path), array)
            path.write_bytes(raw[:-4])
            with self.assertRaises(ValueError):
                read_depth_map(path)

    def test_sparse_handoff_rejects_different_optimized_geometry(self):
        """Prevent using an older point model or different camera calibration for dense reconstruction."""
        from joint_map.dense import check_sparse_handoff
        model, anchors, truth = create_synthetic_scene()
        other = copy.deepcopy(model)
        for image in other.images.values():
            camera = other.cameras[image.camera_id]
            for point in image.points2D:
                point.xy = camera.world_to_image(camera.image_to_world(point.xy))
        self.assertTrue(check_sparse_handoff(model, other)["passed"])
        other.images[1].points2D[0].xy[0] += 1e-6
        with self.assertRaises(AssertionError):
            check_sparse_handoff(model, other)
        other.images[1].points2D[0].xy[0] -= 1e-6
        other.points3D[next(iter(other.points3D))].xyz[0] += .01
        with self.assertRaises(AssertionError):
            check_sparse_handoff(model, other)

    def test_dense_cli_requires_explicit_branch(self):
        """Dispatch both declared dense branches and refuse an omitted or unrelated branch argument."""
        from unittest.mock import patch
        from run_joint_map import run_main
        for branch in ["baseline", "joint"]:
            with patch("sys.argv", ["run_joint_map.py", "--config", "configs/joint_map.json", "--stage", "dense", "--branch", branch]), patch("run_joint_map.run_map_stage") as dispatch:
                run_main()
            dispatch.assert_called_once_with((PROJECT_ROOT/"configs/joint_map.json").resolve(), "dense", branch)
        with patch("sys.argv", ["run_joint_map.py", "--config", "configs/joint_map.json", "--stage", "dense"]), patch("sys.stderr"), self.assertRaises(SystemExit):
            run_main()

class TestDsmUpdate(unittest.TestCase):
    """Check physical rejection cases, single-cell conflicts, exact masks and empty-update reporting."""
    def get_limits(self):
        """Return the approved fixed point and raster reliability limits."""
        return {"min_support_views": 3, "min_triangulation_angle_deg": 1., "max_reprojection_error_px": 2., "max_relative_depth_error": .01, "max_normal_angle_deg": 10., "min_points_per_cell": 3, "max_height_mad_m": .5}

    def create_views(self, centers=(-1., 0., 1.), focal=100., size=101):
        """Create independent parallel cameras observing a ten-meter plane with valid normals."""
        views = []
        for index, center in enumerate(centers):
            normals = np.zeros((size, size, 3), dtype=np.float32)
            normals[:, :, 2] = -1
            views.append({"image_idx": index, "image_id": index+1, "name": f"{index}.png", "camera_id": 1, "rotation": np.eye(3), "translation": np.array([-center, 0., 0.]), "center": np.array([center, 0., 0.]), "params": np.array([focal, focal, (size-1)/2, (size-1)/2]), "depth": np.full((size, size), 10.), "normals": normals})
        return views

    def create_fixture(self, root):
        """Write a complete six-view metric MVS fixture and original DSM without external tools."""
        import struct
        from joint_map.dense import write_metric_cloud, read_image_order
        from joint_map.optimization import get_sha256
        from run_joint_map import write_json
        out = root/"outputs"
        dense = out/"baseline/dense"
        for folder in [out/"baseline/model", dense/"sparse", out/"shared", out/"logs", dense/"stereo/depth_maps", dense/"stereo/normal_maps"]:
            folder.mkdir(parents=True, exist_ok=True)
        views = self.create_views((-2.5, -1.5, -.5, .5, 1.5, 2.5))
        positions = np.array([[-.01, 0, 10.], [0., 0., 10.], [.01, 0., 10.]])
        model = pycolmap.Reconstruction()
        model.add_camera(pycolmap.Camera("PINHOLE", 101, 101, [100., 100., 50., 50.], id=1))
        for view in views:
            local = positions+view["translation"]
            pixels = local[:, :2]/local[:, 2:]*100+50
            model.add_image(pycolmap.Image(view["name"], keypoints=list(pixels), tvec=view["translation"], camera_id=1, id=view["image_id"]))
            model.register_image(view["image_id"])
        ids = []
        for index, position in enumerate(positions):
            ids.append(model.add_point3D(position, pycolmap.Track([pycolmap.TrackElement(view["image_id"], index) for view in views])))
        model.write(str(out/"baseline/model"))
        model.write(str(dense/"sparse"))
        for view in views:
            for folder, array in [("depth_maps", view["depth"][:, :, None]), ("normal_maps", view["normals"])]:
                raw = b"101&101&"+str(array.shape[2]).encode()+b"&"+array.astype("<f4").transpose(1, 0, 2).tobytes(order="F")
                (dense/"stereo"/folder/(view["name"]+".geometric.bin")).write_bytes(raw)
        points = np.zeros(3, dtype=[(key, "<f4") for key in ["x", "y", "z", "nx", "ny", "nz"]]+[(key, "u1") for key in ["red", "green", "blue"]])
        for index, key in enumerate(["x", "y", "z"]):
            points[key] = positions[:, index]
        points["nz"] = -1
        write_metric_cloud(dense/"fused_metric.ply", points, 1., np.eye(3), np.zeros(3))
        with (dense/"fused_metric.ply.vis").open("wb") as stream:
            stream.write(struct.pack("<Q", 3))
            for _ in range(3):
                stream.write(struct.pack("<I6I", 6, *range(6)))
        write_json(dense/"image_order.json", {"images": read_image_order(dense/"sparse"), "images_bin_sha256": get_sha256(dense/"sparse/images.bin")})
        np.savez_compressed(out/"shared/map_priors.npz", shared_point_ids=np.array(ids), means_enu_m=positions, prior_ids=np.arange(3))
        write_json(out/"shared/origin.json", create_origin(112.99, 28.29, 70))
        original = np.full((3, 3), 72., dtype=np.float32)
        original[0, 0] = -9999
        transform = Affine(1e-4, 0, 112.99-1.5e-4, 0, -1e-4, 28.29+1.5e-4)
        with rasterio.open(root/"dsm.tif", "w", driver="GTiff", width=3, height=3, count=1, dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999) as output:
            output.write(original, 1)
        report = {"status": "complete", "dense_points": 3, "source_model_sha256": {name: get_sha256(out/"baseline/model"/name) for name in ["cameras.bin", "images.bin", "points3D.bin"]}, "output_sha256": {name: get_sha256(dense/name) for name in ["fused_metric.ply", "fused_metric.ply.vis", "image_order.json"]}, "workspace_path": str(dense), "image_order_path": str(dense/"image_order.json"), "depth_scale_to_m": 1.}
        write_json(dense/"summary.json", report)
        return out, {"dsm_path": "dsm.tif", "update": self.get_limits()}, original, model, views

    def test_height_layers_use_fixed_gap_and_reason_priority(self):
        """Reject supported layers before undersupported groups even when raw MAD is zero."""
        from joint_map.dsm import check_height_groups
        for heights in [[10.]*7+[12.]*3, [10.]*3+[11.]*3+[15.]]:
            self.assertEqual(check_height_groups(heights, 3, .5)[0], "multiple_height_layers")
        self.assertEqual(check_height_groups([10.]*7+[12.]*2, 3, .5)[0], "ambiguous_height_support")
        self.assertEqual(check_height_groups([10., 11.], 3, .5)[0], "ambiguous_height_support")
        reason, median, mad = check_height_groups([10., 10.5, 11.], 3, .5)
        self.assertEqual((reason, median, mad), ("accepted", 10.5, .5))
        self.assertEqual(check_height_groups([0., .75, 1.5, 2.25, 3.], 3, .5)[0], "height_mad_exceeded")
        self.assertEqual(check_height_groups([10., 10.1], 3, .5)[0], "insufficient_reliable_points")

    def test_grid_boundaries_do_not_clip_outside_points(self):
        """Keep outer right and bottom boundaries, nonfinite coordinates and outside points unmapped."""
        from joint_map.dsm import get_grid_cells
        cells, valid = get_grid_cells(np.array([[.1, 1.9], [2., 1.], [-.1, 1.], [1., 0.], [np.nan, 1.]]), Affine(1, 0, 0, 0, -1, 2), (2, 2))
        np.testing.assert_array_equal(valid, [True, False, False, False, False])
        np.testing.assert_array_equal(cells[0], [0, 0])
        np.testing.assert_array_equal(cells[1:], np.full((4, 2), -1))

    def test_point_reliability_requires_actual_multiview_support(self):
        """Accept a physically consistent plane and reject absent overlap or insufficient support views."""
        from joint_map.dsm import check_point_reliability
        positions, normals = np.array([[0., 0., 10.]]), np.array([[0., 0., -1.]])
        for support, expected in [(np.ones((1, 3), dtype=bool), True), (np.array([[1, 0, 0]], dtype=bool), False), (np.array([[1, 1, 0]], dtype=bool), False)]:
            result = check_point_reliability(positions, normals, support, self.create_views(), self.get_limits())
            self.assertEqual(bool(result["reliable"][0]), expected)

    def test_bad_depth_normal_parallax_and_invalid_predictions_are_rejected(self):
        """Reject each independent point failure without replacing invalid errors by zero."""
        from joint_map.dsm import check_point_reliability
        positions, normals = np.array([[0., 0., 10.]]), np.array([[0., 0., -1.]])
        support = np.ones((1, 3), dtype=bool)
        for mode in ["depth", "normal", "parallax", "negative_depth", "nonfinite"]:
            views = self.create_views((-.001, 0., .001) if mode == "parallax" else (-1., 0., 1.))
            points = positions.copy()
            if mode == "depth":
                views[0]["depth"][:] = 10.2
            if mode == "normal":
                views[0]["normals"][:] = [1., 0., 0.]
            if mode == "negative_depth":
                points[:, 2] = -10
            if mode == "nonfinite":
                points[:, 0] = np.nan
            result = check_point_reliability(points, normals, support, views, self.get_limits())
            self.assertFalse(result["reliable"][0], mode)
            if mode in ["negative_depth", "nonfinite"]:
                self.assertTrue(np.isnan(result["max_reprojection_error_px"][0]))

    def test_pairwise_reprojection_and_normals_are_not_tautological(self):
        """Reject cross-view errors even when individual sampled depths and fused-normal angles pass."""
        from joint_map.dsm import check_point_reliability
        positions, normals = np.array([[0., 0., 10.]]), np.array([[0., 0., -1.]])
        views = self.create_views((-2., 0., 2.), 1000., 1001)
        views[0]["depth"][:] = 10.09
        result = check_point_reliability(positions, normals, np.ones((1, 3), dtype=bool), views, self.get_limits())
        self.assertTrue(result["consistent_support"].all())
        self.assertGreater(result["max_reprojection_error_px"][0], 2.)
        self.assertFalse(result["reliable"][0])
        views = self.create_views()
        for index, angle in [(0, 9.), (2, -9.)]:
            views[index]["normals"][:] = Rotation.from_euler("y", angle, degrees=True).apply([0., 0., -1.])
        result = check_point_reliability(positions, normals, np.ones((1, 3), dtype=bool), views, self.get_limits())
        self.assertTrue(result["consistent_support"].all())
        self.assertGreater(result["max_pair_normal_angle_deg"][0], 10.)
        self.assertFalse(result["reliable"][0])

    def test_raster_masks_preserve_unobserved_conflicted_and_nodata_cells(self):
        """Update only reliable observed cells while leaving a conflict local to its one cell."""
        from joint_map.dsm import build_dsm_update
        original = np.full((3, 3), 5., dtype=np.float32)
        original[0, 0] = -9999
        cells = np.array([[1, 1]]*3+[[1, 2]]*3)
        evidence = {"visible_support": np.ones((6, 3), dtype=bool), "reliable": np.ones(6, dtype=bool)}
        conflicts = np.zeros((3, 3), dtype=bool)
        conflicts[1, 1] = True
        result = build_dsm_update(original, cells, np.array([10., 10.1, 10.2]*2), evidence, conflicts, self.get_limits())
        self.assertEqual(result["updated"][1, 1], 5.)
        self.assertAlmostEqual(result["updated"][1, 2], 10.1, places=5)
        np.testing.assert_array_equal(np.argwhere(result["update_mask"]), [[1, 2]])
        np.testing.assert_array_equal(result["updated"][~result["update_mask"]], original[~result["update_mask"]])
        conflicts[:] = False
        evidence["visible_support"][:, 1:] = False
        result = build_dsm_update(original, cells, np.array([10.]*6), evidence, conflicts, self.get_limits())
        self.assertFalse(result["update_mask"].any())

    def test_raster_multiple_and_ambiguous_layers_do_not_drop_minor_groups(self):
        """Preserve original raster values for both layer rejection reasons and the low-point case."""
        from joint_map.dsm import build_dsm_update
        for heights, reason in [([10.]*7+[12.]*3+[15.], "multiple_height_layers"), ([10.]*7+[12.], "ambiguous_height_support"), ([10., 10.1], "insufficient_reliable_points")]:
            count = len(heights)
            result = build_dsm_update(np.array([[72.]], dtype=np.float32), np.zeros((count, 2), dtype=int), np.array(heights), {"visible_support": np.ones((count, 3), dtype=bool), "reliable": np.ones(count, dtype=bool)}, np.zeros((1, 1), dtype=bool), self.get_limits())
            self.assertFalse(result["update_mask"].any())
            self.assertEqual(result["updated"][0, 0], 72.)
            self.assertEqual(result["height_reason_names"][result["height_reason_codes"][0, 0]], reason)

    def test_sparse_conflicts_are_local_and_occlusion_is_not_conflict(self):
        """Use actual track identities and distinguish front conflicts from occluded sparse predictions."""
        from joint_map.dsm import create_baseline_lineage, check_sparse_dense
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out, config, original, model, views = self.create_fixture(root)
            origin = json.loads((out/"shared/origin.json").read_text())
            base = create_baseline_lineage(out, model)
            base["map_xyz_enu_m"][:, 0] -= 12
            with rasterio.open(root/"dsm.tif") as dsm:
                transform = dsm.transform
            wrong = copy.deepcopy(base)
            wrong["observation_point2D_ids"][0] = 100
            with self.assertRaises(ValueError):
                check_sparse_dense(model, wrong, views, origin, transform, original.shape, self.get_limits())
            views[0]["depth"][:] = 11.
            views[1]["depth"][:] = 11.
            result, mask = check_sparse_dense(model, base, views, origin, transform, original.shape, self.get_limits())
            self.assertTrue(np.all(result["dense_consistency_states"] == "conflict"))
            np.testing.assert_array_equal(np.argwhere(mask), [[1, 1]])
            self.assertTrue(np.all(result["old_cells"][:, 1] == 0))
            self.assertTrue(np.all(result["optimized_cells"][:, 1] == 1))
            views[0]["depth"][:] = 9.
            views[1]["depth"][:] = 9.
            result, mask = check_sparse_dense(model, base, views, origin, transform, original.shape, self.get_limits())
            self.assertFalse(mask.any())
            self.assertEqual(int(result["dense_occluded_observations"].sum()), 6)
            self.assertFalse(result["dense_conflict_observations"].any())

    def test_sparse_movement_without_dense_support_cannot_force_an_update(self):
        """Keep lineage links empty when no reliable same-cell dense evidence supports moved points."""
        from joint_map.dsm import link_dense_support
        lineage = {"optimized_cells": np.array([[0, 0]]), "dense_consistency_states": np.array(["consistent"]), "dense_supported_views": np.ones((1, 3), dtype=bool)}
        evidence = {"reliable": np.array([False]), "consistent_support": np.ones((1, 3), dtype=bool)}
        link_dense_support(lineage, np.array([[0, 0]]), evidence, np.ones((1, 1), dtype=bool))
        self.assertEqual(lineage["dense_support_indices"].size, 0)
        self.assertEqual(lineage["update_cell_indices"].size, 0)

    def test_complete_pipeline_and_empty_update_preserve_original_values(self):
        """Run real raster and point I/O, then prove empty evidence gives no_valid_update and zero area."""
        from joint_map.dsm import run_dsm_update
        from joint_map.optimization import get_sha256
        from run_joint_map import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out, config, original, model, views = self.create_fixture(root)
            before = get_sha256(root/"dsm.tif")
            report = run_dsm_update(config, root, out, "baseline", write_json)
            self.assertEqual(report["update_status"], "updated")
            self.assertEqual(report["updated_cell_count"], 1)
            self.assertGreater(report["updated_area_m2"], 0.)
            with rasterio.open(out/"baseline/map/updated_dsm.tif") as output:
                written = output.read(1)
                self.assertEqual(output.nodata, -9999)
                self.assertEqual(written[0, 0], -9999)
                self.assertAlmostEqual(written[1, 1], 80., places=5)
            for path in (out/"baseline/dense/stereo/normal_maps").glob("*.bin"):
                path.write_bytes(b"101&101&3&"+np.zeros(101*101*3, dtype="<f4").tobytes())
            report = run_dsm_update(config, root, out, "baseline", write_json)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["update_status"], "no_valid_update")
            self.assertFalse(report["dsm_update_goal_complete"])
            self.assertEqual(report["updated_cell_count"], 0)
            self.assertEqual(report["updated_area_m2"], 0.)
            with rasterio.open(out/"baseline/map/updated_dsm.tif") as output:
                np.testing.assert_array_equal(output.read(1), original)
            self.assertEqual(get_sha256(root/"dsm.tif"), before)

    def test_update_cli_requires_branch_and_empty_update_is_not_goal_completion(self):
        """Keep stage execution separate from actual DSM delivery in the public manifest."""
        from unittest.mock import patch
        from run_joint_map import run_main, run_map_stage, write_json
        for branch in ["baseline", "joint"]:
            with patch("sys.argv", ["run_joint_map.py", "--config", "configs/joint_map.json", "--stage", "update", "--branch", branch]), patch("run_joint_map.run_map_stage") as dispatch:
                run_main()
            dispatch.assert_called_once_with((PROJECT_ROOT/"configs/joint_map.json").resolve(), "update", branch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/"outputs"
            write_json(root/"config.json", {"output_dir": "outputs"})
            write_json(out/"run_manifest.json", {"config": get_file_fingerprint(root/"config.json"), "stages": [{"stage": "dense", "branch": "baseline", "status": "complete"}]})
            write_json(out/"logs/check.log", {"status": "complete", "input_fingerprints": {}, "origin": {}})
            write_json(out/"shared/origin.json", {})
            write_json(out/"shared/prior_model.json", {})
            report = {"status": "complete", "error": None, "artifacts": {}, "update_status": "no_valid_update", "updated_cell_count": 0, "updated_area_m2": 0., "dsm_update_goal_complete": False}
            with patch("run_joint_map.PROJECT_ROOT", root), patch("run_joint_map.check_config"), patch("run_joint_map.check_environment"), patch("joint_map.optimization.require_confirmed_prior"), patch("joint_map.dsm.run_dsm_update", return_value=report):
                run_map_stage(root/"config.json", "update", "baseline")
            manifest = json.loads((out/"run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["update_status_by_branch"]["baseline"], "no_valid_update")
            self.assertFalse(manifest["dsm_update_goal_complete"])

class TestGeographicExport(unittest.TestCase):
    """Check geographic storage, ordered pose conventions, immutable handoffs and empty-update propagation."""
    def test_cloud_double_fields_axes_colors_and_roundtrip(self):
        """Preserve full point identity and double geographic coordinates including a distant ENU position."""
        from joint_map.export import write_geographic_cloud, read_geographic_cloud, check_geographic_cloud
        origin = create_origin(112.99, 28.29, 70)
        positions = np.array([[.123456789, -2.987654321, 10.1], [10000., -20000., 301.23456789]])
        colors = np.array([[1, 100, 255], [240, 60, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cloud.ply"
            write_geographic_cloud(path, positions, colors, origin)
            result = check_geographic_cloud(path, positions, colors, origin)
            values = read_geographic_cloud(path)
            self.assertEqual(result["point_count"], 2)
            self.assertLessEqual(result["max_abs_coordinate_error_m"], 1e-6)
            self.assertEqual(values.dtype.names, ("x", "y", "z", "red", "green", "blue"))
            self.assertEqual(values.dtype["x"].itemsize, 8)
            np.testing.assert_array_equal(np.column_stack([values[key] for key in ["x", "y", "z"]]), convert_enu_to_geodetic(positions, origin))
            np.testing.assert_array_equal(np.column_stack([values[key] for key in ["red", "green", "blue"]]), colors)
            with self.assertRaises(AssertionError):
                check_geographic_cloud(path, positions+np.array([.1, 0, 0]), colors, origin)
            raw = path.read_bytes()
            for corrupted in [raw.replace(b"property double x", b"property float x"), raw.replace(b"EPSG:4326", b"EPSG:4978"), raw[:-1]]:
                path.write_bytes(corrupted)
                with self.assertRaises(ValueError):
                    read_geographic_cloud(path)

    def test_six_nonzero_poses_use_camera_local_geographic_basis(self):
        """Round trip decimal pose rows with independently declared nonzero angles and varying locations."""
        from joint_map.export import export_geographic_poses, read_geographic_poses
        origin = create_origin(112.99, 28.29, 70)
        model = pycolmap.Reconstruction()
        model.add_camera(pycolmap.Camera("PINHOLE", 640, 480, [800., 800., 320., 240.], id=1))
        names, expected = [], []
        for index in range(6):
            name = f"{index}_0.png"
            position = np.array([112.99+.0001*index, 28.29-.0002*index, 125.+index])
            angles = np.array([13.-index, -21.+2*index, 37.+index])
            rotation, translation = convert_pose_to_colmap(position, angles, origin)
            model.add_image(pycolmap.Image(name, qvec=pycolmap.rotmat_to_qvec(rotation), tvec=translation, camera_id=1, id=index+1))
            model.register_image(index+1)
            names.append(name)
            expected.append(np.r_[position, angles])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"poses.txt"
            result = export_geographic_poses(path, model, names, origin)
            self.assertTrue(result["passed"])
            values = read_geographic_poses(path, names)
            np.testing.assert_allclose(values, expected, rtol=0, atol=1e-8)
            rows = path.read_text().splitlines()
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(len(row.split()) == 7 for row in rows))
            self.assertTrue(all("e" not in token.lower() for row in rows for token in row.split()[1:]))
            for text in ["\n".join(rows[:-1]), "\n".join(rows[::-1]), "\n".join(rows).replace(rows[0].split()[1], "nan", 1)]:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_geographic_poses(path, names)

    def test_complete_export_preserves_sources_and_no_valid_update_status(self):
        """Export real synthetic stage artifacts without promoting an empty DSM update into delivery."""
        from joint_map.dsm import run_dsm_update
        from joint_map.export import run_geographic_export
        from joint_map.optimization import get_sha256
        from run_joint_map import write_json
        for empty in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                out, config, original, model, views = TestDsmUpdate().create_fixture(root)
                config["image_names"] = sorted(image.name for image in model.images.values())
                if empty:
                    for path in (out/"baseline/dense/stereo/normal_maps").glob("*.bin"):
                        path.write_bytes(b"101&101&3&"+np.zeros(101*101*3, dtype="<f4").tobytes())
                update = run_dsm_update(config, root, out, "baseline", write_json)
                report = run_geographic_export(config, root, out, "baseline", write_json)
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["source_update_status"], "no_valid_update" if empty else "updated")
                self.assertEqual(report["dsm_update_goal_complete"], not empty)
                self.assertEqual(report["registered_images"], 6)
                self.assertEqual(report["sparse_points"], 3)
                self.assertEqual(report["dense_points"], 3)
                self.assertFalse(report["gt_inputs_read"])
                for key, path in update["artifacts"].items():
                    self.assertEqual(get_sha256(path), update["output_sha256"][key])
                cloud = out/"baseline/dense/fused_metric.ply"
                cloud.write_bytes(cloud.read_bytes()+b"broken")
                with self.assertRaises(ValueError):
                    run_geographic_export(config, root, out, "baseline", write_json)

    def test_export_cli_requires_branch(self):
        """Dispatch only explicit export branches and reject missing branch selection."""
        from unittest.mock import patch
        from run_joint_map import run_main
        for branch in ["baseline", "joint"]:
            with patch("sys.argv", ["run_joint_map.py", "--config", "configs/joint_map.json", "--stage", "export", "--branch", branch]), patch("run_joint_map.run_map_stage") as dispatch:
                run_main()
            dispatch.assert_called_once_with((PROJECT_ROOT/"configs/joint_map.json").resolve(), "export", branch)
        with patch("sys.argv", ["run_joint_map.py", "--config", "configs/joint_map.json", "--stage", "export"]), patch("sys.stderr"), self.assertRaises(SystemExit):
            run_main()

class TestGroundTruthEvaluation(unittest.TestCase):
    """Check exact GT validity, fixed-domain scoring, paired diagnostics and baseline-first freezing."""
    def create_fixture(self, root, invalid_joint=False):
        """Write small six-pair GT and branch artifacts with a known constant-depth reference plane."""
        from PIL import Image
        from joint_map.optimization import get_sha256
        from run_joint_map import write_json
        out = root/"outputs"
        (root/"images").mkdir()
        (out/"shared").mkdir(parents=True)
        (out/"logs").mkdir()
        names = [f"{index}_0.png" for index in range(6)]
        config = {"image_names": names, "image_dir": "images", "dsm_path": "original.tif", "camera": {"width": 5, "height": 5, "params": [100., 100., 2., 2.]}}
        evaluation = json.loads((PROJECT_ROOT/"configs/joint_map_evaluation.json").read_text())
        evaluation.update(gt_pose_path="truth.txt", gt_depth_dir="images")
        write_json(root/"config.json", config)
        write_json(root/"evaluation.json", evaluation)
        write_json(out/"shared/origin.json", create_origin(112.99, 28.29, 70.))
        pose_text = "\n".join(name+" 112.99 28.29 80 0 0 0" for name in names)+"\n"
        (root/"truth.txt").write_text(pose_text)
        for index, name in enumerate(names):
            Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(root/"images"/name)
            depth = np.full((5, 5), 10., dtype=np.float32)
            if index == 0:
                depth[0, 0], depth[0, 1] = np.nan, 0.
            np.save(root/"images"/(Path(name).stem+".npy"), depth)
        profile = {"driver": "GTiff", "height": 5, "width": 5, "count": 1, "dtype": "float32", "crs": "EPSG:4326", "transform": Affine(1e-5, 0, 112.99-2.5e-5, 0, -1e-5, 28.29+2.5e-5), "nodata": -9999.}
        with rasterio.open(root/"original.tif", "w", **profile) as source:
            source.write(np.full((5, 5), 72., dtype=np.float32), 1)
        for branch in ["baseline", "joint"]:
            folder = out/branch
            (folder/"map").mkdir(parents=True)
            (folder/"export").mkdir()
            value = -9999. if branch == "joint" and invalid_joint else 71.
            with rasterio.open(folder/"map/updated_dsm.tif", "w", **profile) as source:
                source.write(np.full((5, 5), value, dtype=np.float32), 1)
            mask_profile = dict(profile, dtype="uint8", nodata=255)
            with rasterio.open(folder/"map/update_mask.tif", "w", **mask_profile) as source:
                source.write(np.ones((5, 5), dtype=np.uint8), 1)
            artifacts = {name: str(folder/"map"/(name+".tif")) for name in ["updated_dsm", "update_mask"]}
            update = {"status": "complete", "update_status": "updated", "artifacts": artifacts, "output_sha256": {key: get_sha256(path) for key, path in artifacts.items()}}
            write_json(folder/"map/summary.json", update)
            (folder/"export/poses_wgs84.txt").write_text(pose_text)
            inputs = {path: get_sha256(path) for path in artifacts.values()}
            inputs[str(folder/"map/summary.json")] = get_sha256(folder/"map/summary.json")
            write_json(folder/"export/summary.json", {"status": "complete", "artifacts": {"poses_wgs84.txt": str(folder/"export/poses_wgs84.txt")}, "output_sha256": {"poses_wgs84.txt": get_sha256(folder/"export/poses_wgs84.txt")}, "input_sha256": inputs})
        return config, root/"evaluation.json", root/"config.json", out

    def get_metric_template(self):
        """Return explicit finite primary values for testing acceptance separately from diagnostic regions."""
        return {"translation_median_m": .5, "translation_std_m": .1, "rotation_median_deg": .5, "rotation_std_deg": .1, "dsm_median_m": 1., "dsm_std_m": .1, "recall_1m_1deg": 1., "recall_3m_3deg": 1., "valid_pose_count": 6}

    def test_depth_rule_keeps_all_finite_positive_maxima(self):
        """Preserve repeated maximum depths and exclude only nonfinite or nonpositive values."""
        from joint_map.evaluation import get_valid_depth
        values = np.array([np.nan, np.inf, -np.inf, -1., 0., 1., 55., 55.])
        np.testing.assert_array_equal(get_valid_depth(values), [False, False, False, False, False, True, True, True])

    def test_strict_recalls_fixed_denominator_and_population_std(self):
        """Exclude exact threshold boundaries and do not remove missing images from the denominator."""
        from joint_map.evaluation import summarize_pose_errors
        result = summarize_pose_errors([.999, 1., 2.999, 3., np.nan, 0.], [.999, .1, 2.999, 0., np.nan, 3.])
        self.assertEqual(result["recall_1m_1deg"], 1/6)
        self.assertEqual(result["recall_3m_3deg"], 3/6)
        self.assertIsNone(result["translation_median_m"])
        self.assertEqual(result["valid_pose_count"], 5)
        result = summarize_pose_errors(np.arange(6.), np.zeros(6))
        self.assertEqual(result["translation_median_m"], 2.5)
        self.assertEqual(result["translation_std_m"], float(np.std(np.arange(6.), ddof=0)))

    def test_rotation_matrix_angle_and_missing_estimate(self):
        """Measure relative rotations and mark missing poses without fabricating zero errors."""
        from joint_map.evaluation import compute_pose_metrics
        names = [f"{index}_0.png" for index in range(6)]
        origin = create_origin(112.99, 28.29, 70)
        truth = {name: np.array([112.99, 28.29, 80., 0., 0., 0.]) for name in names}
        prediction = {name: np.array([112.99, 28.29, 80., 0., 0., 10.]) for name in names}
        metrics, rows = compute_pose_metrics(prediction, truth, names, origin)
        self.assertAlmostEqual(metrics["rotation_median_deg"], 10., places=10)
        self.assertEqual(metrics["translation_median_m"], 0.)
        prediction.pop(names[-1])
        metrics, rows = compute_pose_metrics(prediction, truth, names, origin)
        self.assertEqual(rows[-1]["status"], "missing_or_invalid")
        self.assertIsNone(rows[-1]["rotation_error_deg"])
        self.assertEqual(metrics["valid_pose_count"], 5)

    def test_partition_identity_nodata_and_paired_samples(self):
        """Keep fixed mask regions while separating invalid samples and newly filled unpaired references."""
        from joint_map.evaluation import get_partition_masks, get_partition_statistics, REFERENCE_DTYPE
        masks = get_partition_masks(np.array([1, 0, 0, 1], dtype=bool), np.array([1, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1], dtype=bool), np.array([1, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 1], dtype=bool), np.array([0, 0, 1, 0], dtype=bool))
        np.testing.assert_array_equal(masks["new"], [False, True, True, False])
        np.testing.assert_array_equal(masks["lost"], [False, False, False, True])
        np.testing.assert_array_equal(masks["D0"], [True, True, False, True])
        self.assertEqual(int(masks["original_nodata_new_fill"].sum()), 1)
        unpaired = get_partition_masks(np.zeros(2, dtype=bool), np.ones(2, dtype=bool), np.zeros(2, dtype=bool), np.array([1, 0], dtype=bool), np.zeros(2, dtype=bool), np.ones(2, dtype=bool))
        self.assertEqual(int(unpaired["original_nodata_new_region"].sum()), 2)
        self.assertEqual(int(unpaired["original_nodata_new_fill"].sum()), 1)
        references = np.zeros(4, dtype=REFERENCE_DTYPE)
        references["cell_index"] = np.arange(4)
        errors = {"baseline": np.array([1., 1., np.nan, 1.]), "joint": np.array([2., 2., 10., np.nan]), "original": np.array([2., 2., np.nan, 2.])}
        statistics = get_partition_statistics(errors, masks, references, {"height": 2, "width": 2, "transform": Affine(1e-5, 0, 112., 0, -1e-5, 28.)})
        new = statistics["new"]
        self.assertEqual(new["per_method_valid_subset"]["baseline"]["invalid_count"], 1)
        self.assertEqual(new["unpaired_joint_valid_count"], 1)
        self.assertEqual(new["paired_baseline_joint"]["baseline"]["reference_count"], 1)
        self.assertEqual(new["paired_baseline_joint"]["joint"]["reference_count"], 1)
        self.assertGreater(new["area_m2"], 0.)

    def test_primary_pass_and_new_region_degradation_remain_separate(self):
        """Allow eight primary metrics to pass while reporting an independently degraded new region."""
        from joint_map.evaluation import compare_primary_metrics, get_partition_masks, get_partition_statistics, REFERENCE_DTYPE
        baseline, joint = self.get_metric_template(), self.get_metric_template()
        baseline.update(dsm_median_m=5.5, dsm_std_m=4.5)
        joint.update(dsm_median_m=2., dsm_std_m=0.)
        comparisons, passed = compare_primary_metrics(baseline, joint, 0)
        self.assertTrue(passed)
        self.assertEqual(len(comparisons), 8)
        masks = get_partition_masks(np.array([1, 1, 0, 0], dtype=bool), np.ones(4, dtype=bool), np.ones(4, dtype=bool), np.ones(4, dtype=bool), np.ones(4, dtype=bool), np.zeros(4, dtype=bool))
        references = np.zeros(4, dtype=REFERENCE_DTYPE)
        references["cell_index"] = np.arange(4)
        partitions = get_partition_statistics({"baseline": np.array([10., 10., 1., 1.]), "joint": np.array([2., 2., 2., 2.]), "original": np.ones(4)}, masks, references, {"height": 2, "width": 2, "transform": Affine(1e-5, 0, 112., 0, -1e-5, 28.)})
        self.assertTrue(partitions["new"]["paired_diagnostic_degradation"])
        joint["dsm_std_m"] = None
        self.assertFalse(compare_primary_metrics(baseline, joint, 1)[1])

    def test_complete_reference_order_and_baseline_freeze_before_joint(self):
        """Evaluate all finite pixels, prove baseline-first access and reuse the identical immutable baseline."""
        from unittest.mock import patch
        from joint_map.evaluation import run_evaluation, load_evaluation_raster, REFERENCE_DTYPE
        from joint_map.optimization import get_sha256
        from run_joint_map import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, evaluation_path, config_path, out = self.create_fixture(root)
            def load_checked(path, profile=None):
                """Require the baseline snapshot to exist before opening the joint DSM for scoring."""
                if Path(path) == out/"joint/map/updated_dsm.tif":
                    self.assertTrue((out/"evaluation/baseline_metrics.json").is_file())
                return load_evaluation_raster(path, profile)
            with patch("joint_map.evaluation.load_evaluation_raster", side_effect=load_checked):
                report = run_evaluation(config, evaluation_path, config_path, root, out, write_json)
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["primary_acceptance_passed"])
            self.assertEqual(report["reference_count"], 148)
            self.assertEqual(report["acceptance_reference_count"], 148)
            references = np.fromfile(out/"evaluation/reference_points.bin", dtype=REFERENCE_DTYPE)
            expected = [(index, pixel) for index in range(6) for pixel in range(25) if not (index == 0 and pixel in [0, 1])]
            self.assertEqual(list(zip(references["image_index"], references["pixel_index"])), expected)
            digest = get_sha256(out/"evaluation/baseline_metrics.json")
            mtime = (out/"evaluation/baseline_metrics.json").stat().st_mtime_ns
            report = run_evaluation(config, evaluation_path, config_path, root, out, write_json)
            self.assertEqual(get_sha256(out/"evaluation/baseline_metrics.json"), digest)
            self.assertEqual((out/"evaluation/baseline_metrics.json").stat().st_mtime_ns, mtime)
            evaluation_path.write_text(evaluation_path.read_text()+"\n")
            with self.assertRaises(ValueError):
                run_evaluation(config, evaluation_path, config_path, root, out, write_json)
            self.assertEqual(get_sha256(out/"evaluation/baseline_metrics.json"), digest)

    def test_joint_nodata_does_not_shrink_D0_or_hide_primary_failure(self):
        """Keep the complete baseline-valid reference set when every joint DSM sample becomes invalid."""
        from joint_map.evaluation import run_evaluation
        from run_joint_map import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, evaluation_path, config_path, out = self.create_fixture(root, invalid_joint=True)
            report = run_evaluation(config, evaluation_path, config_path, root, out, write_json)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["reference_count"], 148)
            self.assertEqual(report["acceptance_reference_count"], 148)
            self.assertEqual(report["invalid_regression_count"], 148)
            self.assertIsNone(report["joint_metrics"]["dsm_median_m"])
            self.assertFalse(report["primary_acceptance_passed"])
            self.assertFalse(report["delivery_complete"])
            values = np.load(out/"evaluation/dsm_errors_joint.npy")
            self.assertEqual(values.shape, (148,))
            self.assertTrue(np.isnan(values).all())

    def test_evaluation_config_cannot_enter_reconstruction_stages(self):
        """Expose the isolated evaluation entry and reject evaluation data on reconstruction invocations."""
        from unittest.mock import patch
        from run_joint_map import run_main
        arguments = ["run_joint_map.py", "--config", "configs/joint_map.json", "--stage", "evaluate", "--evaluation-config", "configs/joint_map_evaluation.json"]
        with patch("sys.argv", arguments), patch("run_joint_map.run_map_stage") as dispatch:
            run_main()
        dispatch.assert_called_once_with((PROJECT_ROOT/"configs/joint_map.json").resolve(), "evaluate", evaluation_config_path=(PROJECT_ROOT/"configs/joint_map_evaluation.json").resolve())
        for values in [arguments[:-2], ["run_joint_map.py", "--config", "configs/joint_map.json", "--stage", "joint", "--evaluation-config", "configs/joint_map_evaluation.json"]]:
            with patch("sys.argv", values), patch("sys.stderr"), self.assertRaises(SystemExit):
                run_main()

if __name__ == "__main__":
    unittest.main()
