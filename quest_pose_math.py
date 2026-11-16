"""Quaternion-native Cartesian poses: [x_mm, y_mm, z_mm, qw, qx, qy, qz].

RPY is accepted only for legacy callers/files and the xArm SDK boundary.
Angular velocities and errors remain spatial rotation vectors, NOT Euler rates.
"""
import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def orientation_rotation(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('orientation must be finite')
    if value.shape == (3,):
        return Rotation.from_euler('xyz', value)  # legacy/SDK boundary only
    if value.shape != (4,):
        raise ValueError('expected nonzero wxyz quaternion')
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('expected finite nonzero quaternion norm')
    value = value / norm
    return Rotation.from_quat(value[[1, 2, 3, 0]])


def quaternion_wxyz(rotation, reference=None):
    q = rotation.as_quat()[[3, 0, 1, 2]]
    if reference is not None and np.dot(q, reference) < 0:
        q = -q  # same rotation, continuous quaternion hemisphere
    return q


def orientation_values(rotation, template):
    """Preserve old helper API at its boundary; live control uses quaternion4."""
    if np.asarray(template).shape == (4,):
        return quaternion_wxyz(rotation, template)
    return rotation_rpy(rotation)


def rotation_rpy(rotation):
    # Euler is only an SDK/display representation; at gimbal lock scipy picks
    # an equivalent orientation. Never use its components as control errors.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected.*', category=UserWarning)
        return rotation.as_euler('xyz')


def pose_vector(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape not in ((6,), (7,)) or not np.isfinite(pose).all():
        raise ValueError('expected finite XYZ+RPY or XYZ+wxyz pose')
    if pose.shape == (7,):
        pose = pose.copy()
        pose[3:] = quaternion_wxyz(orientation_rotation(pose[3:]), pose[3:])
    return pose


def as_quaternion_pose(value):
    pose = pose_vector(value)
    return np.r_[pose[:3], quaternion_wxyz(orientation_rotation(pose[3:]),
                 pose[3:] if len(pose) == 7 else None)]


def pose_to_sdk(value):
    pose = pose_vector(value)
    return np.r_[pose[:3], rotation_rpy(orientation_rotation(pose[3:]))]


def pose_rpy_deg(value):
    return np.rad2deg(pose_to_sdk(value)[3:]).tolist()
