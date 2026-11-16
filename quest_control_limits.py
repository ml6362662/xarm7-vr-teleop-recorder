"""Shared GUI/backend tuning ranges (no numerical or GUI dependencies)."""

import math

HOME_SPEED_DEFAULT = 5.
HOME_SPEED_LIMITS = (5., 30.)


def home_speed(value):
    """Validate the explicitly selected home-return joint speed, in deg/s."""
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not HOME_SPEED_LIMITS[0] <= value <= HOME_SPEED_LIMITS[1]):
        raise ValueError('初始位姿复位速度必须为 5–30°/s 的有限数值')
    return float(value)

TUNING_LIMITS = {
    'speed_mm_s': (1, 200),
    'translation_scale': (.05, 10),
    'joint_speed_rad_s': (.05, 1),
    'acceleration_mm_s2': (50, 1000),
    'tracking_lead_mm': (1, 20),
    'rotation_scale': (.05, 5),
    'rotation_speed_deg_s': (1, 120),
    'rotation_acceleration_deg_s2': (10, 720),
}

# Gains are independent: raising a gain also raises the target's required speed.
RESPONSIVE_SETTINGS = dict(speed_mm_s=150., joint_speed_rad_s=1.,
    acceleration_mm_s2=600., tracking_lead_mm=12.,
    rotation_speed_deg_s=60., rotation_acceleration_deg_s2=240.)
