"""Read-only local dexterity checks; no inverse, enabling or motion commands."""
import numpy as np
from scipy.spatial.transform import Rotation

from quest_relative_motion import vector, CONTROL_PERIOD
from quest_pose_math import as_quaternion_pose, orientation_rotation

ANGULAR_LENGTH_MM = 100.0
SIGMA_SLOW = 20.0
SIGMA_STOP = 8.0
SIGMA_RECOVERY_FLOOR = 7.5
SIGMA_APPROACH_FLOOR = 8.5
JOINT_ACCELERATION = 2.0  # rad/s² for acceleration/reversal, not emergency braking


def joint_acceleration_step(current, target, previous_velocity, dt, speed):
    current, target = vector(current, 7), vector(target, 7)
    if not np.isfinite([dt, speed]).all() or dt < 0 or speed < 0:
        raise ValueError('关节平滑参数必须有限且非负')
    dt = min(max(dt, 0.), CONTROL_PERIOD)
    if dt == 0:
        return current.copy()
    desired = (target-current)/dt
    peak = np.max(np.abs(desired))
    if peak > speed:
        desired *= speed/peak
    previous_velocity = vector(previous_velocity, 7)
    # Coordinate velocity changes as a whole vector. Scaling every desired
    # joint velocity by the slowest accelerating joint repeatedly brakes
    # already-moving axes when another joint makes a small correction.
    # Never clip joints independently: retain their coupled velocity change.
    if np.dot(desired, previous_velocity) < -1e-12:
        # Do not insert an instantaneous zero-speed frame on reversal. Brake
        # along the previous coordinated direction; the caller MUST FK-check
        # this candidate (including joint/workspace/lead/pose guards). It can
        # lie outside the new IK segment. Safety clipping still takes priority.
        peak_previous = np.max(np.abs(previous_velocity))
        braking_speed = max(0., min(speed, peak_previous-JOINT_ACCELERATION*dt))
        return current + previous_velocity*(braking_speed/peak_previous)*dt
    if not np.any(np.abs(desired) > 1e-12):
        return current.copy()
    change = desired-previous_velocity
    peak_change = float(np.max(np.abs(change)))
    fraction = min(1., JOINT_ACCELERATION*dt/peak_change) if peak_change>0 else 1.
    velocity = previous_velocity + change*fraction
    peak_velocity = float(np.max(np.abs(velocity)))
    if peak_velocity>speed:
        velocity *= speed/peak_velocity  # tightened safety speed wins over smoothing
    # A gradual turn, like braking, can leave the new IK segment. The caller
    # MUST validate this final candidate by FK, joint/lead/speed/pose guards.
    return current + velocity*dt


class PoseGuard:
    """Finite-difference Jacobian from this controller's calibrated TCP FK.

    Rotation rows use 100 mm/rad so thresholds have a fixed documented scale.
    Cache only healthy neighborhoods; warning-region candidates are checked
    individually. Thresholds are conservative software heuristics, not a
    manufacturer-certified safety envelope or a collision detector.
    """
    def __init__(self, arm, clock):
        self.arm, self.clock = arm, clock
        self.cached = None
        self.feedback = {}

    def evaluate(self, joints, deadline, force=False):
        joints = vector(joints, 7)
        cached = self.cached
        if (not force and cached is not None and cached['sigma_min'] >= SIGMA_SLOW
                and self.clock()-cached['sample_time'] < .05
                and np.max(np.abs(joints-cached['joints'])) < np.deg2rad(.5)):
            return dict(cached)

        def fk(q):
            if self.clock()+.001 >= deadline:
                raise RuntimeError('姿态保护计算预算不足，停止本帧，不发送迟到指令')
            code, pose = self.arm.get_forward_kinematics(q.tolist(), input_is_radian=True, return_is_radian=True)
            if code:
                raise RuntimeError(f'姿态保护正解失败（SDK {code}）')
            if self.clock() >= deadline:
                raise RuntimeError('姿态保护正解超时，停止本帧')
            return as_quaternion_pose(pose)

        base = fk(joints)
        rotation = orientation_rotation(base[3:])
        step = .0005
        columns = []
        for axis in range(7):
            perturbed = joints.copy(); perturbed[axis] += step
            pose = fk(perturbed)
            angle = (orientation_rotation(pose[3:])*rotation.inv()).as_rotvec()
            columns.append(np.r_[pose[:3]-base[:3], angle*ANGULAR_LENGTH_MM]/step)
        singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
        sigma = float(singular[-1])
        if not np.isfinite(singular).all():
            raise RuntimeError('姿态保护指标无效，拒绝运动')
        ratio = float(np.clip((sigma-SIGMA_STOP)/(SIGMA_SLOW-SIGMA_STOP), 0., 1.))
        self.cached = dict(joints=joints.copy(), sample_time=self.clock(), sigma_min=sigma,
            condition=float(singular[0]/max(sigma, 1e-9)), speed_ratio=ratio,
            status='blocked' if sigma <= SIGMA_RECOVERY_FLOOR else 'recovery_only' if sigma <= SIGMA_APPROACH_FLOOR else 'slowing' if sigma < SIGMA_SLOW else 'normal')
        return dict(self.cached)

    def publish(self, metric, message=''):
        self.feedback = {k: v for k,v in metric.items() if k not in ('joints', 'sample_time')}
        self.feedback.update(message=message, sigma_slow=SIGMA_SLOW, sigma_stop=SIGMA_RECOVERY_FLOOR,
            sigma_approach_stop=SIGMA_APPROACH_FLOOR,
            angular_length_mm=ANGULAR_LENGTH_MM)

    def snapshot(self):
        return dict(self.feedback)
