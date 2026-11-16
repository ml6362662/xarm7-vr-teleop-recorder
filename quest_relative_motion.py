"""Pure relative-position mapping and continuous joint-step planning."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from quest_control_limits import TUNING_LIMITS
from quest_pose_math import (pose_vector, orientation_rotation, orientation_values)
from quest_empirical_mapping import features, matrix6, ANGULAR_LENGTH_MM
from quest_pose_filter import filter_settings


# Quest Unity world: X right, Y up, Z forward. Robot base: X forward, Y left, Z up.
QUEST_TO_ROBOT = np.array([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.]])
DEFAULT_SETTINGS = {"speed_mm_s": 50.0, "translation_scale": 1.0,
                    "joint_speed_rad_s": 0.5, "acceleration_mm_s2": 300.0,
                    "tracking_lead_mm": 5.0,
                    "range_mm": None, "direction_matrix": QUEST_TO_ROBOT.tolist(),
                    "rotation_enabled": True, "rotation_axes": [True, True, True],
                    "rotation_scale": 1.0, "rotation_speed_deg_s": 30.0,
                    "rotation_acceleration_deg_s2": 90.0, "gripper_enabled": False,
                    "gripper_open_mm": 84.0, "gripper_close_mm": 0.0,
                    "gripper_speed_mm_s": 50.0, "gripper_force": 30.0,
                    "teleop_button": "trigger", "gripper_button": "a",
                    "empirical_matrix": None, "empirical_frame": None,
                    "saved_empirical_matrix": None, "saved_empirical_frame": None}
CONTROL_HZ = 100
CONTROL_PERIOD = 1 / CONTROL_HZ


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"expected {size} finite values")
    return result


def direction_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("方向矩阵必须是 3×3 有限数值")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4) or abs(abs(np.linalg.det(matrix)) - 1) > 1e-4:
        raise ValueError("方向矩阵各轴必须正交且长度为 1；倍率请单独设置")
    return matrix.copy()


def settings(value):
    result = dict(DEFAULT_SETTINGS)
    result.update(value or {})
    result['input_filter'] = filter_settings(result.get('input_filter'))
    # Migrate existing trial settings without changing the matrix or user tuning.
    for key in ('empirical_trial', 'trial_radius_mm', 'trial_angle_deg'):
        result.pop(key, None)
    result['range_mm'] = None
    if result["teleop_button"] not in ("trigger", "b"):
        raise ValueError("teleop_button 必须是 trigger 或 b")
    if result["gripper_button"] not in ("a","b"):
        raise ValueError("gripper_button 必须是 a 或 b；扳机专用于遥操")
    if result['teleop_button']=='b' and result['gripper_button']=='b':
        raise ValueError("B 不能同时用于遥操和夹爪；B 夹爪时遥操必须使用食指扳机")
    limits = dict(TUNING_LIMITS, gripper_open_mm=(0,84), gripper_close_mm=(0,84),
                  gripper_speed_mm_s=(15,225), gripper_force=(1,100))
    for name, (low, high) in limits.items():
        number = float(result[name])
        if not np.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{name} 必须在 {low}–{high} 范围内")
        result[name] = number
    for name in ("rotation_enabled", "gripper_enabled"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} 必须是布尔值")
    axes = result["rotation_axes"]
    if not isinstance(axes, (list, tuple)) or len(axes) != 3 or any(type(axis) is not bool for axis in axes):
        raise ValueError("rotation_axes 必须为三个布尔值")
    result["rotation_axes"] = list(axes)
    if result["gripper_close_mm"] >= result["gripper_open_mm"]:
        raise ValueError("夹爪闭合位置必须小于打开位置")
    result["direction_matrix"] = direction_matrix(result["direction_matrix"]).tolist()
    for prefix in ('empirical', 'saved_empirical'):
        if result[prefix + '_matrix'] is None:
            continue
        result[prefix + '_matrix'] = matrix6(result[prefix + '_matrix']).tolist()
        frame = result[prefix + '_frame']
        if not isinstance(frame, (list, tuple)) or len(frame) != 2 or not all(isinstance(s, str) and s for s in frame):
            raise ValueError('六维补偿必须绑定输入来源和参考坐标')
        result[prefix + '_frame'] = list(frame)
    if result['empirical_matrix'] is not None:
        result['saved_empirical_matrix'] = result['empirical_matrix']
        result['saved_empirical_frame'] = result['empirical_frame']
    return result


def yaw_matrix(degrees):
    angle = np.deg2rad(float(degrees))
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ QUEST_TO_ROBOT


def tether(target, actual, lead_mm):
    target = pose_vector(target).copy()
    delta = target[:3] - pose_vector(actual)[:3]
    norm = np.linalg.norm(delta)
    if norm > lead_mm:
        target[:3] = np.asarray(actual)[:3] + delta * lead_mm / norm
    return target


class RelativeTranslation:
    def __init__(self, scale=0.25, radius_mm=None, speed_mm_s=10.0, matrix=None,
                 rotation_enabled=False, rotation_scale=1, rotation_axes=(True, True, True), empirical_matrix=None):
        if (not TUNING_LIMITS['translation_scale'][0] <= scale <= TUNING_LIMITS['translation_scale'][1]
                or not TUNING_LIMITS['speed_mm_s'][0] <= speed_mm_s <= TUNING_LIMITS['speed_mm_s'][1]
                or not TUNING_LIMITS['rotation_scale'][0] <= rotation_scale <= TUNING_LIMITS['rotation_scale'][1]
                or (radius_mm is not None and (not np.isfinite(radius_mm) or radius_mm <= 0))):
            raise ValueError("control limits: scale 0.05–10, rotation scale 0.05–5, optional positive radius, speed 1–200 mm/s")
        self.scale, self.radius, self.speed = scale, radius_mm, speed_mm_s
        self.matrix = direction_matrix(QUEST_TO_ROBOT if matrix is None else matrix)
        self.empirical_matrix = None if empirical_matrix is None else matrix6(empirical_matrix)
        self.origin = self.hand_anchor = self.tcp_anchor = None
        self.rotation_enabled, self.rotation_scale = rotation_enabled, rotation_scale
        self.rotation_axes = np.asarray(rotation_axes, dtype=bool)
        self.hand_rotation_anchor = None
        self.hand_rotation_previous = self.target_rotation = None
        self.aligned_rotation = Rotation.identity()
        self.empirical_anchor = None
        self.empirical_robot_anchor = None
        self.empirical_delta = np.zeros(6)
        self.empirical_previous = None
        self.empirical_linear_base = np.zeros(3)
        self.empirical_aligned_base = Rotation.identity()
        self.empirical_output_orientation = None
        self.empirical_reanchors = 0
        self.last_linear_increment = self.last_angular_increment = np.zeros(3)

    def reset_origin(self, tcp):
        self.origin = pose_vector(tcp).copy()
        self.release()

    def release(self):
        self.hand_anchor = self.tcp_anchor = None
        self.hand_rotation_anchor = None
        self.hand_rotation_previous = self.target_rotation = None
        self.aligned_rotation = Rotation.identity()
        self.empirical_anchor = None
        self.empirical_delta = np.zeros(6)
        self.empirical_previous = None
        self.empirical_linear_base = np.zeros(3)
        self.empirical_aligned_base = Rotation.identity()
        self.empirical_output_orientation = None
        self.empirical_reanchors = 0
        self.last_linear_increment = self.last_angular_increment = np.zeros(3)

    def engage(self, hand, tcp, quaternion_wxyz=None):
        if self.origin is None:
            self.reset_origin(tcp)
        self.hand_anchor = vector(hand, 3).copy()
        self.tcp_anchor = pose_vector(tcp).copy()
        self.hand_rotation_anchor = hand_rotation(quaternion_wxyz) if self.rotation_enabled or self.empirical_matrix is not None else None
        self.hand_rotation_previous = self.hand_rotation_anchor
        self.target_rotation = orientation_rotation(self.tcp_anchor[3:])
        self.aligned_rotation = Rotation.identity()
        if self.empirical_matrix is not None:
            self.empirical_anchor = np.r_[self.hand_anchor * 1000, vector(quaternion_wxyz, 4)]
            self.empirical_previous = self.empirical_anchor.copy()
            self.empirical_robot_anchor = self.target_rotation
            self.empirical_delta = np.zeros(6)
            self.empirical_linear_base = np.zeros(3)
            self.empirical_aligned_base = Rotation.identity()
            self.empirical_output_orientation = self.tcp_anchor[3:].copy()
            self.empirical_reanchors = 0

    def target(self, hand, quaternion_wxyz=None):
        if self.hand_anchor is None:
            raise RuntimeError("relative position has no takeover anchor")
        if self.empirical_matrix is not None:
            return self.empirical_target(hand, quaternion_wxyz)
        displacement = self.matrix @ (vector(hand, 3) - self.hand_anchor) * 1000 * self.scale
        requested = self.tcp_anchor.copy()
        requested[:3] += displacement
        requested[3:] = self.tcp_anchor[3:]
        if self.rotation_enabled:
            current_rotation = hand_rotation(quaternion_wxyz)
            # Apply gain to EACH spatial increment, not the principal logarithm
            # of the whole gesture (which wraps at 180 degrees and changes the
            # effective gain when combined axes rotate).
            delta = (current_rotation * self.hand_rotation_previous.inv()).as_rotvec()
            # Angular velocity is an axial vector. An improper (LH/RH) basis
            # transform needs det(M) * M, not the position map M alone.
            robot_delta = np.linalg.det(self.matrix) * self.matrix @ delta
            self.aligned_rotation = Rotation.from_rotvec(robot_delta) * self.aligned_rotation
            self.target_rotation = Rotation.from_rotvec(robot_delta * self.rotation_axes * self.rotation_scale) * self.target_rotation
            self.hand_rotation_previous = current_rotation
            requested[3:] = orientation_values(self.target_rotation, requested[3:])
        bounded = requested.copy()
        if self.radius is not None:
            bounded[:3] = np.clip(requested[:3], self.origin[:3] - self.radius, self.origin[:3] + self.radius)
        return requested, bounded

    def empirical_target(self, hand, quaternion_wxyz):
        current = np.r_[vector(hand, 3) * 1000, vector(quaternion_wxyz, 4)]
        current_rotation = hand_rotation(quaternion_wxyz)
        previous_rotation = orientation_rotation(self.empirical_previous[3:])
        if (current_rotation * previous_rotation.inv()).magnitude() > np.deg2rad(45):
            raise ValueError('手柄单帧旋转突变超过 45°：请松开扳机重新接管')
        local_angle = (current_rotation * orientation_rotation(self.empirical_anchor[3:]).inv()).magnitude()
        if local_angle >= np.pi/2:
            # Change the local chart BEFORE approaching log(R)'s pi branch.
            # Carry the previous MAPPED goal, including anti-windup changes,
            # never measured TCP (which would lose motion while it lags).
            # K remains an endpoint map within each chart, not a rate law.
            self.tcp_anchor[:3] += self.empirical_delta[:3] * self.scale
            self.tcp_anchor[3:] = self.empirical_output_orientation
            self.empirical_robot_anchor = self.target_rotation
            self.empirical_linear_base += self.empirical_delta[:3]
            self.empirical_aligned_base = self.aligned_rotation
            self.empirical_anchor = self.empirical_previous.copy()
            self.empirical_delta = np.zeros(6)
            self.empirical_reanchors += 1
        mapped = self.empirical_matrix @ features(self.empirical_anchor, current)
        rv = mapped[3:] / ANGULAR_LENGTH_MM
        previous_rv = self.empirical_delta[3:] / ANGULAR_LENGTH_MM
        self.last_linear_increment = (mapped[:3] - self.empirical_delta[:3]) * self.scale
        relative = Rotation.from_rotvec(rv * self.rotation_axes * self.rotation_scale)
        previous = Rotation.from_rotvec(previous_rv * self.rotation_axes * self.rotation_scale)
        self.last_angular_increment = ((relative * previous.inv()).as_rotvec() if self.rotation_enabled else np.zeros(3))
        self.empirical_delta = mapped
        self.empirical_previous = current.copy()
        self.aligned_rotation = Rotation.from_rotvec(rv) * self.empirical_aligned_base
        requested = self.tcp_anchor.copy()
        requested[:3] += mapped[:3] * self.scale
        if self.rotation_enabled:
            self.target_rotation = relative * self.empirical_robot_anchor
            requested[3:] = orientation_values(self.target_rotation, self.empirical_output_orientation)
        self.empirical_output_orientation = requested[3:].copy()
        bounded = requested.copy()
        if self.radius is not None:
            bounded[:3] = np.clip(requested[:3], self.origin[:3] - self.radius, self.origin[:3] + self.radius)
        return requested, bounded

    def aligned_hand_delta(self, hand):
        """Unit-gain gesture in robot-base axes for coordinate diagnostics."""
        if self.hand_anchor is None:
            return None, None
        if self.empirical_matrix is not None:
            return (self.empirical_linear_base + self.empirical_delta[:3]).tolist(), np.rad2deg(self.aligned_rotation.as_rotvec()).tolist()
        delta = self.matrix @ (vector(hand,3)-self.hand_anchor)*1000
        return delta.tolist(), np.rad2deg(self.aligned_rotation.as_rotvec()).tolist() if self.rotation_enabled else None

    def discard_target_excess(self, requested, bounded):
        """Back-calculate only discarded target motion, preserving clutch/input history.

        Unlike clipping the output alone, this prevents a held hand from
        regenerating the discarded goal and allows reversal from the new goal.
        No change to the workspace origin, direction map, or saved settings.
        """
        requested, bounded = pose_vector(requested), pose_vector(bounded)
        self.tcp_anchor[:3] += bounded[:3]-requested[:3]
        if self.rotation_enabled:
            self.target_rotation = orientation_rotation(bounded[3:])
            if self.empirical_matrix is not None:
                relative = Rotation.from_rotvec(self.empirical_delta[3:] / ANGULAR_LENGTH_MM * self.rotation_axes * self.rotation_scale)
                self.empirical_robot_anchor = relative.inv() * self.target_rotation
                self.empirical_output_orientation = bounded[3:].copy()

    def contains(self, tcp, tolerance=0):
        return self.radius is None or bool(np.all(np.abs(pose_vector(tcp)[:3] - self.origin[:3]) <= self.radius + tolerance))

    def limited_target(self, target, current, dt):
        target, current = pose_vector(target).copy(), pose_vector(current)
        delta = target[:3] - current[:3]
        step = self.speed * min(max(dt, 0), CONTROL_PERIOD)
        norm = np.linalg.norm(delta)
        if norm > step and norm > 0:
            target[:3] = current[:3] + delta * step / norm
        return target


def continuous_solution(solution, current, lower, upper):
    current, solution = vector(current, 7), vector(solution, 7)
    delta = np.arctan2(np.sin(solution - current), np.cos(solution - current))
    if np.max(np.abs(delta)) > 0.035:
        raise RuntimeError("IK 分支跳变超过 2°，已释放控制")
    result = current + delta
    if np.any(result < lower) or np.any(result > upper):
        raise RuntimeError("IK 目标接近关节限位，已释放控制")
    return result


class PressEdge:
    """Debounced press, with no edge from an initially held/unknown button."""
    def __init__(self, press_delay=.06):
        self.press_delay = press_delay

    def reset(self, pressed, now):
        self.stable = self.candidate = pressed if type(pressed) is bool else None
        self.since = now
        self.ready = pressed is False

    def update(self, pressed, now):
        if type(pressed) is not bool:
            raise ValueError("按键状态未上报或无效")
        if pressed != self.candidate:
            self.candidate, self.since = pressed, now
        if now - self.since < self.press_delay or pressed == self.stable:
            return False
        self.stable = pressed
        if not pressed:
            self.ready = True
            return False
        edge, self.ready = self.ready, False
        return edge


def interpolate_pose(start, end, fraction):
    """Short-arc Cartesian interpolation, including the +/-180-degree seam."""
    start, end = pose_vector(start), pose_vector(end)
    result = start.copy()
    result[:3] += (end[:3] - start[:3]) * fraction
    origin = orientation_rotation(start[3:])
    delta = (orientation_rotation(end[3:]) * origin.inv()).as_rotvec()
    result[3:] = orientation_values(Rotation.from_rotvec(delta * fraction) * origin, start[3:])
    return result


def limited_joints(current, target, dt, max_speed_rad_s=0.15):
    delta = vector(target, 7) - vector(current, 7)
    maximum = np.max(np.abs(delta))
    # Limit the whole vector together, preserving the IK branch and direction.
    step = max_speed_rad_s * min(max(dt, 0), CONTROL_PERIOD)
    return np.asarray(current) + delta * min(1, step / maximum) if maximum > 0 else np.asarray(current).copy()


def smooth_target(target, current, velocity, dt, speed, acceleration):
    """Acceleration-bounded command interpolation, with discrete braking."""
    target, current = pose_vector(target), pose_vector(current)
    velocity = vector(velocity, 3)
    dt = min(max(dt, 0), CONTROL_PERIOD)
    if dt == 0:
        return current.copy(), velocity.copy()
    delta = target[:3] - current[:3]
    distance = np.linalg.norm(delta)
    braking = np.sqrt(2 * acceleration * distance + (acceleration * dt / 2) ** 2) - acceleration * dt / 2
    desired = delta * min(speed, braking) / distance if distance > 1e-9 else np.zeros(3)
    change = desired - velocity
    norm = np.linalg.norm(change)
    new_velocity = velocity + change * min(1, acceleration * dt / norm) if norm > 0 else velocity.copy()
    result = current.copy()
    result[:3] += new_velocity * dt
    # Do not clip to the goal: abrupt target reversals must decelerate instead
    # of dropping velocity instantly. Speed/acceleration remain bounded.
    return result, new_velocity


def hand_rotation(quaternion_wxyz):
    q = vector(quaternion_wxyz, 4)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < .5:
        raise ValueError("手柄旋转四元数无效")
    q = q / norm
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def rotation_distance(a, b):
    return float((orientation_rotation(a) * orientation_rotation(b).inv()).magnitude())


def smooth_rotation(target_orientation, current_orientation, velocity, dt, speed, acceleration):
    current = orientation_rotation(current_orientation)
    delta = (orientation_rotation(target_orientation) * current.inv()).as_rotvec()
    # Reuse the translational ramp for a spatial rotation vector, with an
    # identity quaternion placeholder (never interpreted as Euler rates).
    goal = np.r_[delta, 1., 0., 0., 0.]
    step, new_velocity = smooth_target(goal, np.array([0., 0., 0., 1., 0., 0., 0.]), velocity, dt, speed, acceleration)
    result = orientation_values(Rotation.from_rotvec(step[:3]) * current, current_orientation)
    return result, new_velocity


def bound_follow_goal(goal, command, linear_velocity, angular_velocity, settings,
                      linear_increment=None, angular_increment=None):
    """Bound pending gestures to avoid seconds of catch-up after saturation.

    Keep only the tracking allowance plus current braking distance. Rebase
    discarded motion and clear opposing backlog on an intentional reversal.
    Gain is preserved while the robot can follow; above its capability,
    responsiveness takes priority over retaining the entire displacement.
    """
    goal, command = pose_vector(goal), pose_vector(command)
    velocity, omega = vector(linear_velocity, 3), vector(angular_velocity, 3)
    speed = min(np.linalg.norm(velocity), settings['speed_mm_s'])
    angular_speed = min(np.linalg.norm(omega), np.deg2rad(settings['rotation_speed_deg_s']))
    distance_limit = settings['tracking_lead_mm'] + speed**2/(2*settings['acceleration_mm_s2'])
    angle_limit = np.deg2rad(3) + angular_speed**2/(2*np.deg2rad(settings['rotation_acceleration_deg_s2']))
    delta = goal[:3]-command[:3]
    angle = (orientation_rotation(goal[3:]) * orientation_rotation(command[3:]).inv()).as_rotvec()

    def bounded(value, limit, increment, threshold):
        reverse = False
        if increment is not None:
            increment = vector(increment, 3)
            size = np.linalg.norm(increment)
            # Ignore sub-threshold tracking noise and mostly transverse moves.
            if size > threshold and np.linalg.norm(value) > limit*.5:
                direction = increment/size
                projection = float(np.dot(value,direction))
                if projection < -.5*np.linalg.norm(value):
                    value = value-direction*projection+increment
                    reverse = True
        norm = np.linalg.norm(value)
        if norm > limit:
            value = value*(limit/norm)
        return value, reverse

    delta, linear_reverse = bounded(delta,distance_limit,linear_increment,.25)
    if settings['rotation_enabled']:
        angle, angular_reverse = bounded(angle,angle_limit,angular_increment,np.deg2rad(.15))
    else:
        angular_reverse = False
    result = goal.copy()
    result[:3] = command[:3]+delta
    if settings['rotation_enabled']:
        result[3:] = orientation_values(Rotation.from_rotvec(angle)*orientation_rotation(command[3:]), goal[3:])
    return result, dict(status='bounded', translation_policy='responsive', limit_mm=float(distance_limit),
        pending_translation_mm=float(np.linalg.norm(delta)),
        limit_deg=float(np.rad2deg(angle_limit)),
        discarded_mm=float(np.linalg.norm(result[:3]-goal[:3])),
        discarded_deg=float(np.rad2deg(rotation_distance(result[3:],goal[3:]))),
        reverse_translation=linear_reverse, reverse_rotation=angular_reverse)


def command_velocities(previous, committed, dt, speed, angular_speed):
    """Back-calculate planner state AFTER IK/joint/lead constraints.

    The unconstrained planned velocity is not the velocity of a shortened
    command. Keeping it winds up the next step near a kinematic boundary.
    Rotations use the same spatial/world-frame convention as smooth_rotation.
    """
    previous, committed = pose_vector(previous), pose_vector(committed)
    if dt <= 0:
        return np.zeros(3), np.zeros(3)
    linear = (committed[:3] - previous[:3]) / dt
    angular = (orientation_rotation(committed[3:]) *
               orientation_rotation(previous[3:]).inv()).as_rotvec() / dt
    for velocity, limit in ((linear, speed), (angular, angular_speed)):
        norm = np.linalg.norm(velocity)
        if norm > limit:
            velocity *= limit / norm
    return linear, angular
