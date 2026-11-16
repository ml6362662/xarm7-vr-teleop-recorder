"""One Euro filtering of measured hand position and SO(3) orientation.

No robot I/O. Positions enter/leave in metres; tuning uses mm/s and deg/s.
Each timestamp is consumed once, independent of the controller tick rate.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation

from quest_pose_math import orientation_rotation, quaternion_wxyz


DEFAULT_FILTER = dict(
    enabled=False,  # Existing installations stay unchanged until explicitly enabled.
    position_min_cutoff_hz=2.0,
    position_beta_hz_per_mm_s=0.08,
    position_max_cutoff_hz=15.0,
    rotation_min_cutoff_hz=2.0,
    rotation_beta_hz_per_deg_s=0.15,
    rotation_max_cutoff_hz=15.0,
    derivative_cutoff_hz=2.0,
)


def filter_settings(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError('input_filter 必须是配置对象')
    if value and set(value) - set(DEFAULT_FILTER):
        raise ValueError('input_filter 包含未知参数')
    result = dict(DEFAULT_FILTER, **(value or {}))
    if type(result['enabled']) is not bool:
        raise ValueError('input_filter.enabled 必须是布尔值')
    for key in DEFAULT_FILTER:
        if key == 'enabled':
            continue
        number = result[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f'input_filter.{key} 必须为有限数值')
        low, high = (0., 10.) if 'beta' in key else (.1, 60.)
        if not low <= number <= high:
            raise ValueError(f'input_filter.{key} 必须在 {low}–{high} 范围内')
        result[key] = float(number)
    for kind in ('position', 'rotation'):
        if result[f'{kind}_min_cutoff_hz'] > result[f'{kind}_max_cutoff_hz']:
            raise ValueError('滤波最低截止频率不能大于最高截止频率')
    return result


def alpha(cutoff_hz, dt):
    product = 2 * math.pi * cutoff_hz * dt
    return product / (1 + product)


class AdaptivePoseFilter:
    MAX_GAP_S = .25

    def __init__(self, config=None):
        self.config = filter_settings(config)
        self.reset()

    def reset(self):
        self.sample_time = None
        self.frame = None
        self.raw_position = self.raw_rotation = None
        self.position = self.rotation = None
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.updates = 0
        self.feedback = {}

    def update(self, position_m, quaternion, sample_time_s, frame, *, reanchor=False):
        position = np.asarray(position_m, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('滤波输入位置无效')
        quaternion = np.asarray(quaternion, dtype=float)
        if quaternion.shape != (4,):
            raise ValueError('滤波输入必须为 wxyz 四元数')
        rotation = orientation_rotation(quaternion)
        if (isinstance(sample_time_s, bool) or not isinstance(sample_time_s, (int, float))
                or not math.isfinite(sample_time_s)):
            raise ValueError('缺少有效采样时刻，请重启新版手柄接收器')
        if not isinstance(frame, tuple) or len(frame) != 2 or not all(isinstance(x, str) and x for x in frame):
            raise ValueError('缺少手柄来源或参考坐标标识')
        dt = None if self.sample_time is None else sample_time_s - self.sample_time
        if dt is not None:
            if frame != self.frame:
                raise ValueError('滤波参考坐标已改变，请松开后重新接管')
            if dt < 0 or (dt > self.MAX_GAP_S and not reanchor):
                raise ValueError('采样时刻倒退或中断，请松开后重新接管')
            if dt == 0:
                if (not np.allclose(position, self.raw_position, rtol=0, atol=1e-12)
                        or (rotation * self.raw_rotation.inv()).magnitude() > 1e-10):
                    raise ValueError('同一采样时刻出现不同位姿')
                if not reanchor:
                    return self.position.copy(), quaternion_wxyz(self.rotation)
        if reanchor:
            # A checked zero-distance SDK initialization can outlast MAX_GAP_S
            # while input continues arriving. Start at its NEWEST raw sample,
            # rather than filter across the initialization. Source freshness,
            # trigger continuity and raw jump checks belong to the controller;
            # clock reversal, repeated-pose conflicts and frame changes above
            # are still rejected before clearing history.
            self.reset()
            dt = None
        cfg = self.config
        p_cutoff, r_cutoff = cfg['position_min_cutoff_hz'], cfg['rotation_min_cutoff_hz']
        if dt is None:
            self.position, self.rotation = position.copy(), rotation
        else:
            # Filter the vector derivative before its norm, so rapidly
            # alternating noise does not automatically open the passband.
            a_d = alpha(cfg['derivative_cutoff_hz'], dt)
            velocity = (position - self.raw_position) * 1000 / dt
            angular_velocity = np.rad2deg((rotation * self.raw_rotation.inv()).as_rotvec()) / dt
            self.velocity += a_d * (velocity - self.velocity)
            self.angular_velocity += a_d * (angular_velocity - self.angular_velocity)
            p_cutoff = min(cfg['position_max_cutoff_hz'], p_cutoff +
                           cfg['position_beta_hz_per_mm_s'] * float(np.linalg.norm(self.velocity)))
            r_cutoff = min(cfg['rotation_max_cutoff_hz'], r_cutoff +
                           cfg['rotation_beta_hz_per_deg_s'] * float(np.linalg.norm(self.angular_velocity)))
            self.position += alpha(p_cutoff, dt) * (position - self.position)
            # Short-arc SLERP expressed with a spatial rotation increment.
            error = (rotation * self.rotation.inv()).as_rotvec()
            self.rotation = Rotation.from_rotvec(alpha(r_cutoff, dt) * error) * self.rotation
        self.sample_time, self.frame = sample_time_s, frame
        self.raw_position, self.raw_rotation = position.copy(), rotation
        self.updates += 1
        self.feedback = dict(
            algorithm='one-euro-so3-v1', sample_monotonic_s=sample_time_s,
            sample_dt_s=dt, updates=self.updates,
            raw_position_m=position.tolist(), raw_quaternion_wxyz=quaternion_wxyz(rotation).tolist(),
            filtered_position_m=self.position.tolist(), filtered_quaternion_wxyz=quaternion_wxyz(self.rotation).tolist(),
            estimated_speed_mm_s=float(np.linalg.norm(self.velocity)),
            estimated_angular_speed_deg_s=float(np.linalg.norm(self.angular_velocity)),
            position_cutoff_hz=float(p_cutoff), rotation_cutoff_hz=float(r_cutoff),
            position_residual_mm=float(np.linalg.norm(position-self.position)*1000),
            rotation_residual_deg=float(np.rad2deg((rotation*self.rotation.inv()).magnitude())),
        )
        return self.position.copy(), quaternion_wxyz(self.rotation)

    def snapshot(self):
        # Lists in feedback are replaced, never mutated by subsequent updates.
        return dict(self.feedback)
