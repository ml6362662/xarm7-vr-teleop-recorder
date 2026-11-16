"""Six-dimensional mapping for stored matrices.

Features are [dx_mm, dy_mm, dz_mm, 100*spatial_rotation_vector_rad].
They are relative to a local anchor, NOT Euler angles or per-frame rates.
Endpoint fits must not be applied as a differential law: those are not
equivalent for general cross-coupled rotations. The live mapper automatically
changes local anchors, carrying the mapped goal, before this chart expires.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from quest_pose_math import orientation_rotation

ANGULAR_LENGTH_MM = 100.


def features(start, end):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if start.shape != (7,) or end.shape != (7,) or not np.isfinite([start, end]).all():
        raise ValueError('六维标定端点须为有限 XYZ + wxyz')
    rv = (orientation_rotation(end[3:]) * orientation_rotation(start[3:]).inv()).as_rotvec()
    if np.linalg.norm(rv) >= np.deg2rad(150):
        raise ValueError('局部六维补偿接近 180° 分支：请松开扳机重新接管；标定请缩小转角')
    return np.r_[end[:3] - start[:3], ANGULAR_LENGTH_MM * rv]


def matrix6(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise ValueError('六维补偿必须是有限 6×6 矩阵')
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[0] > 5 or singular[-1] < .03:
        raise ValueError('六维补偿放大过大或退化（归一化奇异值需在 0.03–5）；请重新采样')
    return matrix.copy()
