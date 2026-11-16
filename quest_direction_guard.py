"""Read-only local direction probes, never sends or enables motion."""
import numpy as np
from scipy.spatial.transform import Rotation

from quest_relative_motion import vector, continuous_solution, rotation_distance, interpolate_pose
from quest_pose_math import as_quaternion_pose, orientation_rotation, quaternion_wxyz, pose_to_sdk
from quest_pose_guard import SIGMA_SLOW, SIGMA_RECOVERY_FLOOR, SIGMA_APPROACH_FLOOR

# A narrow recovery band around the old threshold, not unrestricted operation
# below it. Approaching paths stop early; only proven retreats may use the band.


def choose_direction(arm, guard, joints, pose, goal, metric, lower, upper, deadline, clock):
    joints, pose, goal = vector(joints,7), as_quaternion_pose(pose), as_quaternion_pose(goal)
    translation = goal[:3]-pose[:3]
    angular = (orientation_rotation(goal[3:])*orientation_rotation(pose[3:]).inv()).as_rotvec()
    distance, angle = np.linalg.norm(translation), np.linalg.norm(angular)
    result = dict(direction='neutral', translation_only=False, probes=[])
    if distance < .01 and angle < np.deg2rad(.01):
        return result

    def probe(rotation_enabled):
        # Same 6D direction as the goal, not six independent axis tests.
        scale = min(1., .5/max(distance,1e-12), np.deg2rad(.1)/max(angle,1e-12) if rotation_enabled else 1.)
        target = pose.copy(); target[:3] += translation*scale
        if rotation_enabled:
            target[3:] = quaternion_wxyz(Rotation.from_rotvec(angular*scale)*orientation_rotation(pose[3:]), pose[3:])
        full_target = target.copy()
        for contraction in (1., .25):
            target = interpolate_pose(pose,full_target,contraction)
            if clock()+.002 >= deadline:
                raise RuntimeError('方向探测预算不足，未发送运动')
            code, solution = arm.get_inverse_kinematics(pose_to_sdk(target).tolist(), input_is_radian=True,
                return_is_radian=True, limited=False, ref_angles=joints.tolist())
            if clock() >= deadline:
                raise RuntimeError('方向探测逆解超时，未发送运动')
            if code == 10:
                continue
            if code:
                raise RuntimeError(f'方向探测逆解失败（SDK {code}）')
            try:
                solution = continuous_solution(solution,joints,lower,upper)
                break
            except RuntimeError:
                continue
        else:
            result['probes'].append(dict(rotation=rotation_enabled, direction='blocked'))
            return 'blocked'
        code, fk = arm.get_forward_kinematics(solution.tolist(),input_is_radian=True,return_is_radian=True)
        if code:
            raise RuntimeError(f'方向探测正解失败（SDK {code}）')
        fk = as_quaternion_pose(fk)
        if np.linalg.norm(fk[:3]-target[:3]) > .1 or rotation_distance(fk[3:],target[3:]) > np.deg2rad(.05):
            result['probes'].append(dict(rotation=rotation_enabled, direction='blocked'))
            return 'blocked'
        predicted = guard.evaluate(solution,deadline,force=True)
        change = predicted['sigma_min']-metric['sigma_min']
        direction = 'retreating' if change > .05 else 'approaching' if change < -.05 else 'neutral'
        result['probes'].append(dict(rotation=rotation_enabled, direction=direction,
                                    sigma=predicted['sigma_min'], sigma_change=change))
        return direction

    result['direction'] = probe(True)
    # Explicit translation priority only when the full 6D request worsens
    # conditioning (or cannot be solved), and translation alone improves it.
    if (result['direction'] in ('approaching','blocked') or
            (metric['sigma_min'] <= SIGMA_APPROACH_FLOOR and result['direction'] != 'retreating')):
        if distance >= .01 and angle >= np.deg2rad(.01) and probe(False) == 'retreating':
            result.update(direction='retreating', translation_only=True)
    return result


def direction_speed_ratio(metric, direction, speed, angular_speed_deg, joint_speed):
    sigma = metric['sigma_min']
    if sigma <= SIGMA_RECOVERY_FLOOR or direction == 'blocked':
        return 0.
    if direction == 'retreating':
        # Do not multiply an escape by a vanishing near-boundary factor.
        slow_exit = min(1.,20./speed,8./angular_speed_deg,.15/joint_speed)
        return slow_exit if sigma <= 10 else max(slow_exit, metric['speed_ratio'])
    # Approaching and ambiguous motion stop BEFORE the recovery-only band.
    return float(np.clip((sigma-SIGMA_APPROACH_FLOOR)/(SIGMA_SLOW-SIGMA_APPROACH_FLOOR),0.,1.))
