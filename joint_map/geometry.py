# 用途：实现 WGS84/ECEF/ENU、相机姿态、反投影和 DSM 像元中心采样。
import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation

GEODETIC_TO_ECEF = Transformer.from_crs(4979, 4978, always_xy=True)
ECEF_TO_GEODETIC = Transformer.from_crs(4978, 4979, always_xy=True)
CAMERA_TO_RENDER = np.diag([1.0, -1.0, -1.0])

def get_ecef_to_enu_rotation(longitude_deg, latitude_deg):
    """Return the ECEF to ENU rotation at a geographic position."""
    longitude, latitude = np.deg2rad([longitude_deg, latitude_deg])
    return np.array([
        [-np.sin(longitude), np.cos(longitude), 0.0],
        [-np.sin(latitude) * np.cos(longitude), -np.sin(latitude) * np.sin(longitude), np.cos(latitude)],
        [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)],
    ])

def convert_geodetic_to_ecef(positions):
    """Convert (..., 3) longitude, latitude, ellipsoidal height to ECEF meters."""
    positions = np.asarray(positions, dtype=np.float64)
    return np.stack(GEODETIC_TO_ECEF.transform(*np.moveaxis(positions, -1, 0)), axis=-1)

def convert_ecef_to_geodetic(positions):
    """Convert (..., 3) ECEF meters to longitude, latitude, ellipsoidal height."""
    positions = np.asarray(positions, dtype=np.float64)
    return np.stack(ECEF_TO_GEODETIC.transform(*np.moveaxis(positions, -1, 0)), axis=-1)

def create_origin(longitude_deg, latitude_deg, ellipsoidal_height_m):
    """Create the serializable local ENU frame without evaluation inputs."""
    return {
        "schema_version": 3,
        "longitude_deg": float(longitude_deg),
        "latitude_deg": float(latitude_deg),
        "ellipsoidal_height_m": float(ellipsoidal_height_m),
        "ecef_origin_m": convert_geodetic_to_ecef([longitude_deg, latitude_deg, ellipsoidal_height_m]).tolist(),
        "R_ecef_to_enu": get_ecef_to_enu_rotation(longitude_deg, latitude_deg).tolist(),
    }

def convert_geodetic_to_enu(positions, origin):
    """Convert geographic positions to the fixed local ENU frame in meters."""
    return (convert_geodetic_to_ecef(positions) - origin["ecef_origin_m"]) @ np.asarray(origin["R_ecef_to_enu"]).T

def convert_enu_to_geodetic(positions, origin):
    """Convert local ENU meters to longitude, latitude, ellipsoidal height."""
    ecef = np.asarray(positions, dtype=np.float64) @ np.asarray(origin["R_ecef_to_enu"]) + origin["ecef_origin_m"]
    return convert_ecef_to_geodetic(ecef)

def convert_pose_to_colmap(position, roll_pitch_yaw_deg, origin):
    """Convert geographic pose with nadir pitch zero to COLMAP R_cw and t_cw."""
    roll, pitch, yaw = roll_pitch_yaw_deg
    render_to_enu = Rotation.from_euler("xyz", [pitch, roll, yaw], degrees=True).as_matrix()
    camera_to_ecef = get_ecef_to_enu_rotation(*position[:2]).T @ render_to_enu @ CAMERA_TO_RENDER
    rotation_cw = (np.asarray(origin["R_ecef_to_enu"]) @ camera_to_ecef).T
    return rotation_cw, -rotation_cw @ convert_geodetic_to_enu(position, origin)

def convert_pose_from_colmap(rotation_cw, translation_cw, origin):
    """Recover geographic position and roll, pitch, yaw from a COLMAP pose."""
    rotation_cw = np.asarray(rotation_cw, dtype=np.float64)
    position = convert_enu_to_geodetic(-rotation_cw.T @ np.asarray(translation_cw), origin)
    camera_to_ecef = np.asarray(origin["R_ecef_to_enu"]).T @ rotation_cw.T
    render_to_enu = get_ecef_to_enu_rotation(*position[:2]) @ camera_to_ecef @ CAMERA_TO_RENDER
    pitch, roll, yaw = Rotation.from_matrix(render_to_enu).as_euler("xyz", degrees=True)
    return position, np.array([roll, pitch, yaw])

def backproject_pixels(pixels, depths_z, intrinsics, rotation_cw, translation_cw):
    """Backproject pixel coordinates and positive camera z-depth to world XYZ."""
    pixels = np.asarray(pixels, dtype=np.float64)
    depths_z = np.asarray(depths_z, dtype=np.float64)
    if not np.all(np.isfinite(depths_z) & (depths_z > 0)):
        raise ValueError("Camera z-depth must be finite and positive")
    homogeneous = np.concatenate([pixels, np.ones(pixels.shape[:-1] + (1,))], axis=-1)
    camera_points = (homogeneous @ np.linalg.inv(intrinsics).T) * depths_z[..., None]
    return (camera_points - translation_cw) @ np.asarray(rotation_cw)

def transform_colmap_pose(rotation_cw, translation_cw, scale, rotation_align, translation_align):
    """Apply a positive Sim(3) consistently to a COLMAP camera pose."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Sim(3) scale must be finite and positive")
    rotation_new = np.asarray(rotation_cw) @ np.asarray(rotation_align).T
    return rotation_new, scale * np.asarray(translation_cw) - rotation_new @ np.asarray(translation_align)

def convert_pixels_to_geodetic(pixels, transform):
    """Convert zero based continuous pixel center indices to map coordinates."""
    pixels = np.asarray(pixels, dtype=np.float64) + 0.5
    longitude, latitude = transform @ (pixels[..., 0], pixels[..., 1])
    return np.stack([longitude, latitude], axis=-1)

def sample_dsm(heights, transform, positions):
    """Bilinearly sample valid four-neighbor DSM centers without extrapolation."""
    heights = np.ma.asarray(heights, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    columns, rows = ~transform @ (positions[..., 0], positions[..., 1])
    columns, rows = np.asarray(columns) - 0.5, np.asarray(rows) - 0.5
    height, width = heights.shape
    if height < 2 or width < 2:
        raise ValueError("Bilinear sampling requires at least two rows and columns")
    valid = np.isfinite(columns) & np.isfinite(rows)
    valid &= (columns >= -1e-8) & (columns <= width - 1 + 1e-8)
    valid &= (rows >= -1e-8) & (rows <= height - 1 + 1e-8)
    columns = np.clip(np.where(valid, columns, 0.0), 0, width - 1)
    rows = np.clip(np.where(valid, rows, 0.0), 0, height - 1)
    left = np.minimum(np.floor(columns).astype(int), width - 2)
    top = np.minimum(np.floor(rows).astype(int), height - 2)
    dx, dy = columns - left, rows - top
    neighbors = np.ma.stack([heights[top, left], heights[top, left + 1], heights[top + 1, left], heights[top + 1, left + 1]], axis=-1)
    values = neighbors.filled(np.nan)
    valid &= np.all(~np.ma.getmaskarray(neighbors) & np.isfinite(values), axis=-1)
    weights = np.stack([(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy], axis=-1)
    sampled = np.sum(values * weights, axis=-1)
    return np.where(valid, sampled, np.nan), valid
