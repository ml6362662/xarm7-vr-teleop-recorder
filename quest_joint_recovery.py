"""TCP-pose home targets and joint-space safety paths; checks never move."""

import json
import time
from pathlib import Path

import numpy as np
from quest_control_limits import HOME_SPEED_DEFAULT, home_speed

from quest_relative_motion import vector, rotation_distance
from quest_pose_math import as_quaternion_pose, pose_to_sdk


def closest_joint_target(solution, current, lower, upper):
    """Select physical angle equivalents near CURRENT joints, not saved joints."""
    solution, current = vector(solution, 7), vector(current, 7)
    result = np.empty(7)
    for axis, angle in enumerate(solution):
        first = int(np.ceil((lower[axis]-angle)/(2*np.pi)))
        last = int(np.floor((upper[axis]-angle)/(2*np.pi)))
        choices = [angle+2*np.pi*k for k in range(first,last+1)]
        if not choices:
            raise RuntimeError("Pose 逆解超出硬件关节限位，拒绝复位")
        result[axis] = min(choices,key=lambda q:abs(q-current[axis]))
    return result


def robot_identity(arm):
    return {"serial": getattr(arm, "sn", None), "axis": getattr(arm, "axis", 7),
            "device_type": getattr(arm, "device_type", 7),
            "tcp_offset": np.round(vector(getattr(arm, "tcp_offset", np.zeros(6)),6), 6).tolist(),
            "world_offset": np.round(vector(getattr(arm, "world_offset", np.zeros(6)),6), 6).tolist()}


class JointRecovery:
    def __init__(self, arm, lower, upper, home_path=None):
        self.arm, self.lower, self.upper = arm, np.array(lower), np.array(upper)
        self.home_path = None if home_path is None else Path(home_path)
        self.home = None
        self.motion = None
        self.feedback = {"status": "idle", "message": "未设置初始位姿；先手动调整到合适姿态再记录"}
        if self.home_path is not None and self.home_path.exists():
            try:
                profile = json.loads(self.home_path.read_text())
                self.validate_profile(profile)
                self.home = profile
                self.feedback = {"status": "saved", "message": "已载入初始位姿；复位前需检查并确认路径净空"}
            except Exception as error:
                self.feedback = {"status": "rejected", "message": f"初始位姿文件不可用：{error}"}

    def validate_profile(self, profile):
        if profile.get("version") not in (1,2,3) or profile.get("robot") != robot_identity(self.arm):
            raise RuntimeError("初始点的机器人/工具坐标配置不匹配，请重新记录")
        if not np.isfinite(float(profile.get('saved_at', float('nan')))):
            raise RuntimeError("初始点记录时间无效，请重新记录")
        grip = profile.get('gripper_mm')
        if grip is not None and (not np.isfinite(float(grip)) or not 0 <= float(grip) <= 84):
            raise RuntimeError("初始夹爪开度无效")
        # v1 files remain readable. Their saved joints are historical diagnostics
        # only: home is a six-dimensional TCP pose, not a joint configuration.
        if profile['version'] == 3:
            goal = as_quaternion_pose(vector(profile.get('tcp_pose_quat'), 7))
            legacy = as_quaternion_pose(vector(profile.get('tcp'), 6))
            if np.linalg.norm(goal[:3]-legacy[:3]) > 1e-6 or rotation_distance(goal[3:], legacy[3:]) > 1e-6:
                raise RuntimeError('初始位姿四元数与兼容字段不一致，请重新记录')
            return goal
        return as_quaternion_pose(vector(profile.get("tcp"), 6))

    def save(self, joints, tcp, gripper_mm=None):
        profile = {"version": 3, "target_type": "tcp_pose", "saved_at": time.time(), "robot": robot_identity(self.arm),
                   "joints_rad": vector(joints, 7).tolist(), "tcp": pose_to_sdk(tcp).tolist(),
                   "tcp_pose_quat": as_quaternion_pose(tcp).tolist(),
                   "gripper_mm": gripper_mm}
        self.validate_profile(profile)
        if self.home_path is not None:
            temporary = self.home_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.home_path)
        self.home = profile
        self.feedback = {"status": "saved", "message": "初始 TCP Pose 已保存；按 XYZ/RxRyRz 复位，不要求原关节组合"}

    def check_health(self):
        code, codes = self.arm.get_err_warn_code()
        if code or codes[0] or codes[1] not in (0, 14):
            raise RuntimeError(f"机器人有错误/警告 {codes}（SDK {code}）；先按厂家流程处理，不自动清故障")
        code, state = self.arm.get_state()
        if code or state in (1, 3, 5):
            raise RuntimeError(f"机器人未停稳或处于保护状态（state {state}），请先停止/处理急停")
        code, joint_states = self.arm.get_joint_states(is_radian=True, num=2)
        if code:
            raise RuntimeError(f"不能读取实际关节速度（SDK {code}），拒绝复位")
        velocity = vector(joint_states[1], 7)
        if np.max(np.abs(velocity)) > np.deg2rad(.5):
            raise RuntimeError("机械臂关节仍在移动；请先停止其他控制源")
        return codes[1]

    def plan(self, current, target, label, check_cancel=None):
        warning = self.check_health()
        current, target = vector(current, 7), vector(target, 7)
        # Recovery is allowed inside the hardware limits, including the normal
        # teleoperation's 5-degree margin. Never move an already out-of-limit arm.
        if np.any(current < self.lower) or np.any(current > self.upper):
            raise RuntimeError("实际关节超出硬件限位；禁止自动恢复，请用厂家示教/故障处理流程")
        if np.any(target < self.lower + np.deg2rad(.2)) or np.any(target > self.upper - np.deg2rad(.2)):
            raise RuntimeError("复位目标太接近硬件关节限位")
        delta = target - current
        if np.max(np.abs(delta)) > np.pi:
            raise RuntimeError("单段关节路径超过 180°，请先分段手动恢复，不自动绕圈")
        samples = max(2, int(np.ceil(np.rad2deg(np.max(np.abs(delta))) / 2)) + 1)
        poses = []
        for fraction in np.linspace(0, 1, samples):
            if check_cancel is not None:
                check_cancel()
            q = current + delta * fraction
            code, fk = self.arm.get_forward_kinematics(q.tolist(), input_is_radian=True, return_is_radian=True)
            if code:
                raise RuntimeError(f"关节复位路径正解失败（SDK {code}）")
            poses.append(as_quaternion_pose(fk))
        xyz = np.array(poses)[:, :3]
        return {"type": "joint", "label": label, "start": current.tolist(), "target": target.tolist(),
                "start_tcp": pose_to_sdk(poses[0]).tolist(), "target_tcp": pose_to_sdk(poses[-1]).tolist(),
                "start_pose_quat": poses[0].tolist(), "target_pose_quat": poses[-1].tolist(),
                "max_joint_travel_deg": float(np.rad2deg(np.max(np.abs(delta)))),
                "tcp_path_min_mm": xyz.min(axis=0).tolist(), "tcp_path_max_mm": xyz.max(axis=0).tolist(),
                "samples": samples, "warning": warning, "speed_deg_s": 5,
                "acceleration_deg_s2": 10, "status": "checked",
                "message": f"{label}关节路径检查通过（{samples} 点）；不含碰撞检查，确认净空后再执行"}

    def plan_home_pose(self, current, check_cancel=None, target_lower=None, target_upper=None,
                       *, speed_deg_s=HOME_SPEED_DEFAULT):
        """Solve the saved TCP goal from the current branch, then check its path.

        The endpoint is POSE-based; a checked joint path is used for recovery
        through awkward postures. This does NOT promise a straight TCP return.
        No fallback to recorded joint angles, no mode/enable/warning writes.
        """
        speed_deg_s = home_speed(speed_deg_s)
        if self.home is None:
            raise RuntimeError("尚未记录初始 TCP Pose")
        goal = self.validate_profile(self.home)
        self.check_health()
        current = vector(current,7)
        if np.any(current < self.lower) or np.any(current > self.upper):
            raise RuntimeError("实际关节超出硬件限位，禁止自动 Pose 复位")
        code, actual = self.arm.get_forward_kinematics(current.tolist(),input_is_radian=True,return_is_radian=True)
        if code:
            raise RuntimeError(f"Pose 复位当前位置正解失败（SDK {code}）")
        actual = as_quaternion_pose(actual)
        if check_cancel is not None:
            check_cancel()
        if np.linalg.norm(actual[:3]-goal[:3]) < .05 and rotation_distance(actual[3:],goal[3:]) < np.deg2rad(.02):
            target = current.copy()  # already at pose; do not rearrange redundancy
        else:
            code, solution = self.arm.get_inverse_kinematics(pose_to_sdk(goal).tolist(),input_is_radian=True,
                return_is_radian=True,limited=False,ref_angles=current.tolist())
            if code:
                raise RuntimeError(f"初始 Pose 逆解失败（SDK {code}）；请先小步关节恢复，不回退到旧关节目标")
            margin = np.deg2rad(.2)
            lower=self.lower+margin if target_lower is None else np.maximum(self.lower+margin,target_lower)
            upper=self.upper-margin if target_upper is None else np.minimum(self.upper-margin,target_upper)
            target = closest_joint_target(solution,current,lower,upper)
        plan = self.plan(current,target,"返回初始 TCP Pose",check_cancel)
        endpoint = as_quaternion_pose(plan['target_pose_quat'])
        if np.linalg.norm(endpoint[:3]-goal[:3]) > .5 or rotation_distance(endpoint[3:],goal[3:]) > np.deg2rad(.25):
            raise RuntimeError("Pose 逆解正解验证不匹配，拒绝执行近似/错误目标")
        plan.update(type="pose", target_tcp=pose_to_sdk(goal).tolist(), target_pose_quat=goal.tolist(),
                    planned_endpoint_tcp=pose_to_sdk(endpoint).tolist(),
                    speed_deg_s=speed_deg_s, acceleration_deg_s2=2*speed_deg_s,
                    message=f"初始 Pose 逆解及关节路径检查通过 · {speed_deg_s:g}°/s；不要求原关节组合，不保证 TCP 直线返回，不含碰撞检查")
        return plan

    def snapshot(self):
        return {"home": self.home, "active": self.motion is not None, "feedback": self.feedback,
                "hardware_lower_deg": np.rad2deg(self.lower).tolist(), "hardware_upper_deg": np.rad2deg(self.upper).tolist()}
