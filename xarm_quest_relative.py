#!/usr/bin/env python3
"""Shared Quest input -> preview or bounded, explicitly armed xArm7 movement."""

from __future__ import annotations

import argparse
import threading
import fcntl
import json
import os
import signal
import socket
import time
from collections import deque
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from quest_relative_motion import RelativeTranslation, continuous_solution, limited_joints, settings, tether, smooth_target, smooth_rotation, command_velocities, hand_rotation, rotation_distance, vector, CONTROL_HZ, CONTROL_PERIOD, PressEdge, interpolate_pose
from quest_gripper_control import GripperWorker
from quest_joint_recovery import JointRecovery
from quest_control_limits import HOME_SPEED_DEFAULT, home_speed
from quest_pose_guard import PoseGuard, joint_acceleration_step, SIGMA_STOP, SIGMA_SLOW
from quest_relative_motion import bound_follow_goal
from quest_pose_math import as_quaternion_pose, pose_to_sdk, pose_rpy_deg, orientation_rotation, quaternion_wxyz
from quest_direction_guard import choose_direction, direction_speed_ratio, SIGMA_RECOVERY_FLOOR, SIGMA_APPROACH_FLOOR
from quest_recording_stream import RecordingPublisher, recording_sample
from quest_pose_filter import AdaptivePoseFilter


ROOT = Path(__file__).resolve().parent
COMMAND_PATH = ROOT / ".state" / "xarm_relative.sock"
SETTINGS_PATH = ROOT / "relative_control_settings.json"


def make_mapper(values):
    return RelativeTranslation(scale=values["translation_scale"], radius_mm=values["range_mm"],
                               speed_mm_s=values["speed_mm_s"], matrix=values["direction_matrix"],
                               rotation_enabled=values["rotation_enabled"], rotation_scale=values["rotation_scale"],
                               rotation_axes=values["rotation_axes"], empirical_matrix=values.get('empirical_matrix'))


class TakeoverCancelled(RuntimeError):
    """The operator/input cancelled a hold initialization, not an SDK failure."""


def prepare_servo_hold(arm, allowed=None):
    """Initialize a zero-distance joint goal before entering servo mode."""
    def check(code, text):
        if code:
            raise RuntimeError(f"{text}失败（SDK {code}）")

    def check_allowed():
        if allowed is not None and not allowed():
            raise TakeoverCancelled("接管初始化时输入失效/扳机松开，已停止")

    def wait_mode(mode):
        deadline = time.monotonic() + 0.6
        while arm.mode != mode:
            if time.monotonic() > deadline:
                raise RuntimeError(f"机器人未切换到模式 {mode}")
            time.sleep(0.01)

    code, joints = arm.get_servo_angle(is_radian=True, is_real=True)
    check(code, "读取实际关节")
    code, before = arm.get_forward_kinematics(joints, input_is_radian=True, return_is_radian=True)
    check(code, "保持位置正解")
    # Stop state 4 cannot be directly changed into paused state 3. Enter position
    # mode first so state 0 cannot reactivate an old servo goal.
    check(arm.set_state(4), "准备停机")
    check(arm.set_mode(0), "设置保持初始化模式")
    wait_mode(0)
    check_allowed()
    check(arm.motion_enable(enable=True), "使能机械臂")
    check_allowed()
    check(arm.set_state(0), "进入位置保持状态")
    check_allowed()
    check(arm.set_servo_angle(angle=joints, is_radian=True, speed=0.1, mvacc=0.3, wait=True, timeout=2), "初始化当前关节保持目标")
    check(arm.set_state(3), "暂停以初始化伺服保持目标")
    code, paused = arm.get_state()
    check(code, "核验暂停状态")
    if paused != 3:
        raise RuntimeError(f"机器人未进入暂停态（{paused}），拒绝启动伺服")
    check_allowed()
    check(arm.set_mode(1), "设置伺服模式")
    wait_mode(1)
    code, actual = arm.get_servo_angle(is_radian=True, is_real=True)
    check(code, "核验实际关节")
    code, after = arm.get_forward_kinematics(actual, input_is_radian=True, return_is_radian=True)
    check(code, "核验保持位置")
    if np.linalg.norm(np.asarray(after[:3]) - before[:3]) > 0.5:
        raise RuntimeError("保持初始化发生超过 0.5 mm 的变化，拒绝接管")
    # Firmware 2.8 rejects servo commands while paused (SDK 9). The preceding
    # completed zero-distance MoveJoint initializes the controller's goal first;
    # immediately refresh it with actual joints after entering ready state.
    check_allowed()
    check(arm.set_state(0), "恢复已初始化的保持目标")
    check_allowed()
    check(arm.set_servo_angle_j(actual, is_radian=True, speed=0.35, mvacc=0.75), "刷新实际关节伺服保持目标")
    return as_quaternion_pose(after), np.asarray(actual)


def wait_initial_reports(arm, timeout=2.):
    """Wait for live reports before validating cached TCP/world identity.

    SDK connection return precedes the first rich report; cached offsets can
    still be zeros. Two report callbacks ensure a preceding report has fully
    completed (some SDK callback phases precede rich-tail parsing).
    """
    ready = threading.Event()
    count = 0
    def reported(_):
        nonlocal count
        count += 1
        if count >= 2:
            ready.set()
    if not arm.register_report_callback(reported):
        raise RuntimeError('无法订阅机器人初始状态，拒绝初始化控制')
    try:
        if not ready.wait(timeout):
            raise RuntimeError('等待机器人工具/坐标状态上报超时，拒绝初始化控制')
    finally:
        arm.release_report_callback(reported)


class RelativeArm:
    def __init__(self, arm, lower, upper, clock=time.monotonic, control_settings=None, gripper=None,
                 hardware_lower=None, hardware_upper=None, home_path=None):
        self.arm, self.lower, self.upper, self.clock = arm, np.asarray(lower), np.asarray(upper), clock
        self.settings = settings(control_settings)
        self.mapper = make_mapper(self.settings)
        self.input_filter = AdaptivePoseFilter(self.settings['input_filter'])
        self.previous_raw_hand = self.previous_raw_rotation = None
        self.gripper = gripper
        self.recovery = JointRecovery(arm, lower if hardware_lower is None else hardware_lower,
                                      upper if hardware_upper is None else hardware_upper, home_path)
        self.motion_button = PressEdge(.02)
        self.gripper_button = PressEdge()
        self.motion_button.reset(None, clock())
        self.gripper_button.reset(None, clock())
        self.gripper_held = False
        self.gripper_generation = 0
        self.last_gripper_release = 0
        self.gripper_input_seen = False
        self.gripper_request_from_controller = False
        self.previous_hand_rotation = None
        self.mode = "preview"
        self.source = None
        self.source_received = 0
        self.heartbeat = clock()
        self.release_seen = False
        self.servo_started = False
        self.motion_started = None
        self.last_tick = clock()
        self.last_invalid = self.last_pause = self.last_count = 0
        self.last_trigger_release = 0
        self.previous_hand = None
        self.command_count = 0
        self.hold_count = 0
        self.poll_source = None
        self.status = "只读目标预览"
        self.last_error = ""
        self.last_stop_event = {}
        self.run_id = f"{time.time():.6f}"
        self.stopping = False
        self.settings_revision = 0
        self.limit_reason = ""
        self.actual_speed = 0.0
        self.actual_angular_speed = 0.0
        self.pause_settle = None
        self.pause_feedback = {}
        self.pose_guard = PoseGuard(arm, clock)
        self.command_guard = {}
        self.backlog_feedback = {}
        self.smoothing_feedback = {}
        self.translation_priority = False
        self.joint_velocity = np.zeros(7)
        self.previous_tcp = None
        self.previous_tcp_time = clock()
        self.last_command_joints = None
        self.tcp, self.joints = self.read_robot()
        self.mapper.reset_origin(self.tcp)
        self.command_tcp = self.tcp.copy()
        self.last_command_joints = None
        self.requested = self.bounded = self.tcp.copy()
        self.hold_tcp = self.tcp.copy()
        self.cartesian_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.cycle_times = deque(maxlen=100)
        self.control_hz = 0.0
        self.ik_subdivision = 0
        self.ik_query_times = deque(maxlen=16)
        self.ik_feedback = {}
        self.ik_updated = None

    @staticmethod
    def checked(code, label):
        if code != 0:
            raise RuntimeError(f"{label}失败（SDK {code}）")

    def read_robot(self):
        self.observation_read_start_s = self.clock()
        code, joints = self.arm.get_servo_angle(is_radian=True, is_real=True)
        self.checked(code, "读取关节")
        code, tcp = self.arm.get_forward_kinematics(joints, input_is_radian=True, return_is_radian=True)
        self.checked(code, "实际关节计算 TCP")
        tcp, joints = np.asarray(tcp, dtype=float), np.asarray(joints, dtype=float)
        if tcp.shape != (6,) or joints.shape != (7,) or not np.isfinite(tcp).all() or not np.isfinite(joints).all():
            raise RuntimeError("机器人状态无效")
        self.observation_read_end_s = self.clock()
        return as_quaternion_pose(tcp), joints

    def update_source(self, state):
        self.source, self.source_received = state, self.clock()

    def hand_pose(self, reanchor=False):
        """Check raw motion before filtering; never smooth buttons or faults."""
        s = self.source
        hand = vector(s['position_m'], 3)
        quaternion = vector(s['quaternion_wxyz'], 4)
        rotation = hand_rotation(quaternion)
        if self.previous_raw_hand is not None and np.linalg.norm(hand-self.previous_raw_hand) > .2:
            raise ValueError('手柄位置突变 · 已释放，请重新接管')
        if (self.previous_raw_rotation is not None and
                (rotation*self.previous_raw_rotation.inv()).magnitude() > np.deg2rad(45)):
            raise ValueError('手柄旋转突变超过 45° · 已释放，请重新接管')
        self.previous_raw_hand, self.previous_raw_rotation = hand.copy(), rotation
        if not self.settings['input_filter']['enabled']:
            return hand, quaternion
        stamp = s.get('sample_monotonic_s')
        # Source freshness remains based on RAW sample/transport age, not on
        # the fact that a filter produced an output on this control tick.
        if type(stamp) not in (int, float) or not np.isfinite(stamp):
            raise ValueError('缺少有效采样时刻，请重启新版手柄接收器')
        if stamp > self.clock()+.03 or self.clock()-stamp >= .25:
            raise ValueError('滤波输入采样时刻已过期或超前')
        frame = (s.get('input_backend', 'legacy'), s.get('frame_id', 'legacy'))
        # Only dt uses the source clock. Host-clock RAW age checks above and
        # in fresh()/the input adapter remain mandatory, including on repeats.
        headset = s.get('sample_headset_s') if s.get('input_backend') == 'unitree' else None
        if headset is not None and (type(headset) not in (int, float) or not np.isfinite(headset) or headset < 0):
            raise ValueError('头显采样时刻无效')
        filter_stamp = stamp if headset is None else headset
        clock_source = 'server_monotonic' if headset is None else 'headset_monotonic'
        frame = (frame[0], frame[1] + '/' + clock_source if isinstance(frame[1], str) and frame[1] else frame[1])
        # After a checked hold initialization, seed a new filter at the fresh
        # pose instead of first filtering across the SDK's blocking interval.
        filtered = self.input_filter.update(hand, quaternion, filter_stamp, frame, reanchor=reanchor)
        self.input_filter.feedback.update(sample_time_source=clock_source, sample_server_monotonic_s=stamp)
        return filtered

    def fresh(self):
        s = self.source
        elapsed = self.clock() - self.source_received
        return bool(s and s.get("valid") and elapsed < 0.25 and s.get("valid_age_s") is not None and elapsed + s["valid_age_s"] < 0.25)

    def pause_fresh(self):
        s = self.source
        elapsed = self.clock() - self.source_received
        return bool(s and elapsed < 0.25 and s.get("pause_age_s") is not None and elapsed + s["pause_age_s"] < 0.25)

    def release(self, message, disarm=False, keep_gripper=False):
        self.translation_priority = False
        was_armed = self.mode == 'motion'
        was_following = self.servo_started
        if self.gripper is not None and not keep_gripper and (
                self.servo_started or disarm or self.gripper_held or self.gripper_request_from_controller):
            self.gripper.cancel_pending()
            self.gripper_request_from_controller = False
        self.mapper.release()
        self.input_filter.reset()
        self.previous_raw_hand = self.previous_raw_rotation = None
        self.previous_hand = None
        self.previous_hand_rotation = None
        self.motion_button.reset((self.source or {}).get("trigger_pressed"), self.clock())
        if not keep_gripper:
            self.gripper_input_seen = False
            self.gripper_held = False
            self.gripper_generation += 1
            key = self.settings['gripper_button']
            self.gripper_button.reset((self.source or {}).get(key+'_pressed'), self.clock())
            self.last_gripper_release = (self.source or {}).get(key+'_release_generation',0)
        self.release_seen = False
        if self.servo_started or self.pause_settle is not None or self.recovery.motion is not None:
            self.hold_tcp = self.tcp.copy()
            # Cancel the controller's last servo target, rather than continuing to chase it.
            self.servo_started = False
            if self.recovery.motion is not None:
                self.recovery.feedback = dict(self.recovery.feedback, status="stopped", message=message)
                self.recovery.motion = None
            code = self.arm.set_state(4)
            if code != 0:
                self.last_error = f"停机请求未确认（SDK {code}），请检查机器人状态"
                disarm = True
            elif was_following and not disarm:
                # A stop cancels the goal, but feedback/motor braking is not
                # instantaneous. Freeze a bounded envelope at release; never
                # enlarge it using motion observed AFTER the stop request.
                lead_limit = min(self.settings['tracking_lead_mm'], max(1., self.mapper.speed*.08))
                lead = min(float(np.linalg.norm(self.command_tcp[:3]-self.tcp[:3])), lead_limit)
                speed = min(self.actual_speed, self.mapper.speed)
                distance_limit = min(30., max(10., lead + speed*.1 + 2.))
                self.pause_settle = {"started": self.clock(), "tcp": self.tcp.copy(),
                                     "joints": self.joints.copy(), "last_joints": self.joints.copy(),
                                     "last_time": self.clock(), "distance_limit_mm": distance_limit}
                self.pause_feedback = dict(status='settling', started_at=time.time(),
                    release_speed_mm_s=self.actual_speed, release_lead_mm=lead,
                    distance_limit_mm=distance_limit, displacement_mm=0., elapsed_s=0.)
        if disarm:
            self.hold_tcp = self.tcp.copy()
            self.mode = "preview"
            self.motion_started = None
            self.pause_settle = None
        if was_following or (was_armed and disarm):
            self.last_stop_event = dict(timestamp=time.time(), reason=message,
                requires_enable=disarm, error=self.last_error)
        self.status = message
        self.command_tcp = self.tcp.copy()
        self.last_command_joints = None
        self.cartesian_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.ik_subdivision = 0
        self.joint_velocity = np.zeros(7)
        # A slow/disconnected query must not poison every subsequent takeover
        # with an estimate larger than the entire budget. Keep the failure
        # diagnostics, but learn timing afresh after an explicit new session.
        self.ik_query_times.clear()

    def settle_pause(self, now):
        """Stop the ARM but retain the session and capture its settled anchor."""
        pause = self.pause_settle
        if pause is None:
            return False
        if self.mode != 'motion':
            self.pause_settle = None
            return False
        displacement = np.linalg.norm(self.tcp[:3]-pause['tcp'][:3])
        angle = rotation_distance(self.tcp[3:],pause['tcp'][3:])
        joint_coast = np.max(np.abs(self.joints-pause['joints']))
        self.pause_feedback.update(displacement_mm=float(displacement), angle_deg=float(np.rad2deg(angle)),
            joint_coast_deg=float(np.rad2deg(joint_coast)), elapsed_s=now-pause['started'])
        if (displacement > pause['distance_limit_mm'] or angle > np.deg2rad(4.5) or
                joint_coast > self.settings['joint_speed_rad_s']*.2+np.deg2rad(2)):
            self.pause_feedback['status'] = 'rejected'
            self.release('暂停停稳期间位姿变化异常 · 遥操已禁用', disarm=True)
            self.last_error = (f"停稳变化超限：位移 {displacement:.1f}/{pause['distance_limit_mm']:.1f} mm，"
                f"旋转 {np.rad2deg(angle):.1f}/4.5°；请检查其他控制源/负载后重新启用")
            return True
        if np.any(self.joints < self.lower) or np.any(self.joints > self.upper):
            self.pause_feedback['status'] = 'rejected'
            self.release('停稳期间关节接近限位 · 遥操已禁用', disarm=True)
            self.last_error = '停稳期间实测关节超出遥操安全余量'
            return True
        code, state = self.arm.get_state()
        self.checked(code, '核验暂停停止状态')
        if getattr(self.arm,'has_error',False) or state==5:
            self.pause_feedback['status'] = 'rejected'
            self.release('暂停期间机器人故障/保护停止 · 遥操已禁用', disarm=True)
            return True
        if now-pause['started'] > 1:
            self.pause_feedback['status'] = 'timeout'
            self.release('暂停停稳超时 · 遥操已禁用', disarm=True)
            self.last_error = '1秒内未确认停稳，请检查机器人'
            return True
        if hasattr(self.arm,'get_joint_states'):
            code, values = self.arm.get_joint_states(is_radian=True,num=2)
            self.checked(code,'核验暂停实际关节速度')
            speed = np.max(np.abs(vector(values[1],7)))
        else:  # deterministic fake/older adapters: actual feedback derivative
            dt = max(now-pause['last_time'],CONTROL_PERIOD)
            speed = np.max(np.abs(self.joints-pause['last_joints']))/dt
        pause['last_joints'], pause['last_time'] = self.joints.copy(), now
        if state==4 and speed <= np.deg2rad(.5):
            self.pause_feedback['status'] = 'settled'
            self.hold_tcp = self.tcp.copy()
            self.command_tcp = self.tcp.copy()
            self.pause_settle = None
            return False
        self.status = '已暂停 · 等待机械臂停稳，遥操许可保留'
        return True

    def validate_robot(self, clear_warning=True):
        self.tcp, self.joints = self.read_robot()
        code, codes = self.arm.get_err_warn_code()
        self.checked(code, "读取错误状态")
        if codes[0] != 0 or codes[1] not in (0, 14):
            raise RuntimeError(f"机器人错误/警告 {codes}，请先处理")
        if np.any(self.joints < self.lower) or np.any(self.joints > self.upper):
            raise RuntimeError("实测关节太接近限位")
        metric = self.pose_guard.evaluate(self.joints, self.clock()+.04, force=True)
        self.pose_guard.publish(metric)
        if metric['sigma_min'] <= SIGMA_RECOVERY_FLOOR:
            raise RuntimeError('当前位置接近奇异姿态，拒绝遥操；请检查 Pose 复位或低速关节恢复路径')
        # Holding the actual pose must have a continuous IK solution. It is
        # NOT necessary that all six +/-1 mm directions be feasible: that old
        # requirement made takeover impossible at Cartesian workspace edges.
        for axis, offset in [(0, 0)]:
            pose = self.tcp.copy()
            pose[axis] += offset
            # SDK limited=True imposes +/-180 degrees, not this robot's actual
            # joint limits. Keep unwrapped physical limits/continuity below.
            code, solution = self.arm.get_inverse_kinematics(pose_to_sdk(pose).tolist(), input_is_radian=True, return_is_radian=True, limited=False, ref_angles=self.joints.tolist())
            self.checked(code, "当前实际位姿逆解检查")
            continuous_solution(solution, self.joints, self.lower, self.upper)
        if codes[1] == 14:
            if not clear_warning:
                raise RuntimeError("机器人有警告 14；只读检查不会清除警告，请先处理后重试")
            self.checked(self.arm.clean_warn(), "清除已核查的无解警告 14")
        code, remaining = self.arm.get_err_warn_code()
        self.checked(code, "复核错误状态")
        if remaining != [0, 0]:
            raise RuntimeError(f"机器人仍有错误/警告 {remaining}")

    def command(self, name, parameters=None):
        if name == "heartbeat":
            self.heartbeat = self.clock()
            return
        if name in ("stop", "release", "preview"):
            if name == "stop" and not self.servo_started and self.recovery.motion is None:
                self.checked(self.arm.set_state(4), "停止机械臂")
            self.release("已释放 · 只读预览", disarm=True)
            self.mapper.reset_origin(self.tcp)
            if name == "stop":
                deadline = time.monotonic() + .3
                while True:
                    code, state = self.arm.get_state()
                    self.checked(code, "核验机械臂停止状态")
                    if state in (4, 5):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"机械臂停止状态未确认（state {state}），拒绝后续操作")
                    time.sleep(.01)
            return
        if self.stopping:
            raise RuntimeError("控制器正在退出，拒绝新的运动/校准命令")
        if self.recovery.motion is not None:
            raise RuntimeError("关节复位正在运动；请等待完成或先停止机械臂")
        if name.startswith("home_") or name == "joint_recover":
            self.recovery_command(name, parameters or {})
            return
        if name in ("gripper_open", "gripper_close"):
            self.request_gripper(self.settings["gripper_open_mm"] if name == "gripper_open" else self.settings["gripper_close_mm"], controller=False)
            return
        if name == "matrix_select":
            self.select_matrix(parameters or {})
            return
        if name == "configure":
            if self.gripper is not None:
                self.gripper.cancel_pending()
            self.release("参数修改 · 已暂停，请重新按遥操按键接管")
            parameters = dict(parameters or {})
            parameters['input_filter'] = dict(self.settings['input_filter'], **parameters.get('input_filter', {}))
            for key in ('direction_matrix', 'empirical_matrix', 'empirical_frame', 'saved_empirical_matrix', 'saved_empirical_frame'):
                parameters.setdefault(key, self.settings.get(key))
            if parameters['empirical_matrix'] is not None and not np.allclose(
                    parameters.get('direction_matrix', self.settings['direction_matrix']), self.settings['direction_matrix'], atol=1e-5):
                raise ValueError('六维补偿启用时不使用旧 3×3 矩阵；请先停用六维补偿再修改方向矩阵')
            new_settings = settings(parameters)
            origin = self.mapper.origin.copy()
            self.settings = new_settings
            self.mapper = make_mapper(new_settings)
            self.input_filter = AdaptivePoseFilter(new_settings['input_filter'])
            self.mapper.reset_origin(origin)
            self.settings_revision += 1
            return
        if name in ("enable", "allow_controller"):
            if self.settings.get('empirical_matrix') is not None and self.settings.get('empirical_frame') != [
                    (self.source or {}).get('input_backend'), (self.source or {}).get('frame_id')]:
                raise RuntimeError('六维补偿的头显参考坐标已失效；请禁用遥操后重新选择矩阵，或切回旧映射')
            if name == "enable" and (not self.fresh() or not self.pause_fresh()):
                raise RuntimeError("需要实时有效的手柄位姿和 B 状态后才能启用试动")
            self.release("检查机器人…", disarm=True)
            self.validate_robot()
            self.mapper.reset_origin(self.tcp)
            self.hold_tcp = self.tcp.copy()
            self.mode = "motion"
            self.motion_started = None
            source = self.source or {}
            self.last_invalid = source.get("invalid_count", 0)
            self.last_pause = source.get("pause_generation", 0)
            self.last_count = source.get("packet_count", 0)
            self.last_trigger_release = source.get("trigger_release_generation", 0)
            self.release_seen = self.settings["teleop_button"] == "b" and source.get("motion_enabled") is False and self.pause_fresh()
            self.status = ("就绪 · 按住扳机遥操，松开暂停 · B 单击切换夹爪开合" if self.settings['gripper_button']=='b' else
                           "就绪 · 按住扳机遥操，松开暂停") if self.settings["teleop_button"] == "trigger" else "已允许手柄 B 控制 · B 暂停/继续"
            return
        if name == "takeover":
            if self.mode != "motion":
                raise RuntimeError("请先启用低速试动")
            if not self.fresh() or not self.pause_fresh() or not self.source.get("motion_enabled"):
                raise RuntimeError("手柄必须有效追踪并处于绿色/继续状态")
            self.release("重新接管…")
            self.release_seen = self.settings["teleop_button"] == "b"
            if not self.release_seen:
                self.status = "起点已释放 · 松开后重新按住扳机"
            return
        raise ValueError(f"未知控制命令 {name}")

    def gripper_allowed(self, controller):
        if self.stopping or not self.settings["gripper_enabled"] or self.recovery.motion is not None:
            return False
        if controller and self.gripper is not None and self.gripper.snapshot().get('command_fault'):
            return False  # queued toggles must not open after an earlier SDK failure
        s = self.source or {}
        return not controller or bool(self.mode == "motion" and self.fresh() and self.pause_fresh()
            and s.get('motion_enabled') is True and self.clock()-self.heartbeat <= .8
            and not getattr(self.arm,'has_error',False)
            and s.get('invalid_count',0) <= self.last_invalid and s.get('pause_generation',0) <= self.last_pause)

    def motion_input_allowed(self):
        s = self.source or {}
        return bool(not self.stopping and self.fresh() and self.pause_fresh()
                    and s.get("motion_enabled") is True
                    and s.get("invalid_count", 0) <= self.last_invalid
                    and s.get("pause_generation", 0) <= self.last_pause
                    and self.clock() - self.heartbeat <= .8
                    and (self.settings["teleop_button"] == "b" or
                         (s.get("trigger_pressed") is True and
                          s.get("trigger_release_generation", 0) <= self.last_trigger_release)))

    def refresh_motion_permission(self):
        if self.poll_source is not None:
            self.poll_source()
        return self.motion_input_allowed()

    def request_gripper(self, target, controller, generation=None):
        if not self.gripper_allowed(controller):
            raise RuntimeError("请先勾选允许夹爪控制并应用；手柄控制还需有效遥操接管")
        if self.gripper is None:
            raise RuntimeError("未连接 G2 夹爪控制通道")
        self.gripper.submit(target, self.settings["gripper_speed_mm_s"], self.settings["gripper_force"],
                            allowed=lambda: self.gripper_allowed(controller) and
                                (generation is None or generation == self.gripper_generation))
        self.gripper_request_from_controller = controller

    def update_gripper_button(self):
        if not self.gripper_allowed(True) or self.gripper is None:
            return
        key = self.settings['gripper_button']
        pressed = self.source.get(key+'_pressed')
        if type(pressed) is not bool:
            if self.gripper_held or self.gripper_request_from_controller:
                self.gripper.cancel_pending()
                self.gripper_request_from_controller = False
            self.gripper_generation += 1
            self.gripper_held = False
            self.gripper_button.reset(None,self.clock())
            self.gripper_input_seen = False
            return
        generation = self.source.get(key+'_release_generation',0)
        if not self.gripper_input_seen:
            self.gripper_button.reset(pressed,self.clock())
            self.last_gripper_release = generation
            self.gripper_input_seen = True
            return
        released = generation > self.last_gripper_release
        self.last_gripper_release = generation
        if key == 'b':
            # A release never opens the gripper. The release counter can
            # expose a release/repress hidden by a conflated snapshot; wait
            # for the new press debounce before accepting another toggle.
            if released and pressed and self.gripper_button.candidate is True:
                self.gripper_button.reset(False, self.clock())
                self.gripper_button.update(True, self.clock())
                return
            if self.gripper_button.update(pressed, self.clock()):
                self.toggle_gripper()
            return
        if self.gripper_held and (not pressed or released):
            # Only an observed button release opens. Trigger clutching, data
            # loss, B's legacy stream pause, and explicit stop do NOT open.
            self.gripper_generation += 1
            self.gripper_held = False
            self.request_gripper(self.settings['gripper_open_mm'],True,self.gripper_generation)
            self.gripper_button.reset(False,self.clock())
            if pressed:
                self.gripper_button.update(True,self.clock())
            return
        if self.gripper_button.update(pressed,self.clock()):
            self.gripper_generation += 1
            self.request_gripper(self.settings['gripper_close_mm'],True,self.gripper_generation)
            self.gripper_held = True

    def toggle_gripper(self):
        """B press toggles intent; a cancelled/failed close is not a closed claw.

        Prefer the newest queued/in-flight target, then the last SDK-accepted
        target. Actual opening is only used before any command was accepted;
        a grasped object may prevent reaching the requested closed position.
        """
        grip = self.gripper.snapshot()
        if grip.get('command_fault'):
            self.status = '夹爪上次指令失败 · 请检查后使用界面明确开/合'
            return
        current = next((grip.get(key) for key in
                        ('pending_target_mm', 'active_target_mm', 'committed_target_mm')
                        if grip.get(key) is not None), None)
        if current is None:
            age = grip.get('position_age_s')
            if age is None or not np.isfinite(age) or not 0 <= age <= .75:
                self.status = '夹爪反馈未就绪 · 等待有效开度后重新按 B'
                return
            current = grip.get('position_mm')
        if current is None or not np.isfinite(current) or not 0 <= current <= 84:
            self.status = '夹爪目标/开度无效 · 本次 B 未执行'
            return
        opened, closed = self.settings['gripper_open_mm'], self.settings['gripper_close_mm']
        target = opened if current <= (opened+closed)/2 else closed
        self.gripper_generation += 1
        self.request_gripper(target, True, self.gripper_generation)

    def solve_follow_target(self, target, previous_command, previous_joints, started):
        # Continuity is between COMMANDS, not target vs delayed actual joints.
        # Do not increase the 2-degree threshold: shorten the Cartesian step
        # when a valid local IK solution needs more joint travel than allowed.
        entered = self.clock()
        self.ik_updated = entered
        self.ik_feedback = {"status": "solving", "attempts": [], "elapsed_ms": 0.,
                            "cycle_elapsed_ms": 0., "fraction": None, "last_failure": "",
                            "target_mm": target[:3].tolist(), "target_rpy_deg": pose_rpy_deg(target),
                            "previous_command_mm": previous_command[:3].tolist(),
                            "previous_command_rpy_deg": pose_rpy_deg(previous_command),
                            "reference_joints_deg": np.rad2deg(previous_joints).tolist()}
        failure = ""

        def finish(status):
            self.ik_feedback.update(status=status, elapsed_ms=(self.clock()-entered)*1000,
                                    cycle_elapsed_ms=(self.clock()-started)*1000, last_failure=failure)

        # Recover one level at a time rather than redoing every known-failing
        # large step each 10 ms cycle. This is local step size, not a workspace
        # fence; a new takeover resets it and never executes a rejected branch.
        attempt = max(0, self.ik_subdivision-1)
        while attempt < 8:
            remaining = .04 - (self.clock() - started)
            expected = max(.003, max(self.ik_query_times, default=0)*1.25)
            if remaining <= 0 or remaining < expected:
                finish("budget_exhausted")
                raise RuntimeError(f"逆解细分超时/预算不足，已释放控制（{len(self.ik_feedback['attempts'])} 次查询，"
                                   f"本帧 {self.ik_feedback['cycle_elapsed_ms']:.1f} ms；最后原因：{failure or '剩余周期预算不足'}）。"
                                   "请松开扳机，必要时禁用遥操并检查关节复位路径")
            candidate = interpolate_pose(previous_command, target, .5 ** attempt)
            query_started = self.clock()
            code, solution = self.arm.get_inverse_kinematics(pose_to_sdk(candidate).tolist(), input_is_radian=True,
                return_is_radian=True, limited=False, ref_angles=previous_joints.tolist())
            duration = max(0, self.clock()-query_started)
            self.ik_query_times.append(duration)
            detail = {"subdivision": attempt, "fraction": .5**attempt, "sdk_code": code,
                      "duration_ms": duration*1000, "max_joint_delta_deg": None, "rejection": "",
                      "candidate_mm": candidate[:3].tolist(), "candidate_rpy_deg": pose_rpy_deg(candidate)}
            self.ik_feedback['attempts'].append(detail)
            if self.clock()-started > .04:
                failure = f"单次逆解查询耗时 {duration*1000:.1f} ms（SDK {code}）"
                detail['rejection'] = failure
                finish("deadline")
                raise RuntimeError(f"逆解细分超时，已释放控制：{failure}")
            if code:
                failure = f"目标逆解失败（SDK {code}）"
                detail['rejection'] = failure
                if code != 10:
                    finish("sdk_error")
                    raise RuntimeError(failure)
                # Invalid IK queries cost ~10 ms on firmware 2.8. Shrink by
                # four after code 10 so a feasible small step can be reached
                # without spending the whole cycle on known-invalid targets.
                next_attempt = min(attempt+2, 7) if attempt < 7 else 8
                if attempt < self.ik_subdivision < next_attempt:
                    next_attempt = self.ik_subdivision
                attempt = next_attempt
                continue
            # Malformed/non-finite results must fail immediately, not retry.
            try:
                solution = vector(solution, 7)
            except ValueError:
                failure = detail['rejection'] = "逆解返回无效/非有限关节数据"
                finish("invalid_result")
                raise
            detail['max_joint_delta_deg'] = float(np.rad2deg(np.max(np.abs(
                np.arctan2(np.sin(solution-previous_joints), np.cos(solution-previous_joints))))))
            try:
                result = continuous_solution(solution, previous_joints, self.lower, self.upper)
            except RuntimeError as error:
                failure = str(error)
                detail['rejection'] = failure
                attempt += 1
                continue
            self.ik_subdivision = attempt
            self.ik_feedback['fraction'] = .5**attempt
            finish("ok")
            return result, attempt
        finish("blocked")
        raise RuntimeError(f"当前方向不可连续/不可达，请松开扳机；可反向移动或使用关节复位恢复。详情：{failure}")

    def recovery_command(self, name, parameters):
        if self.mode != "preview" or self.servo_started:
            raise RuntimeError("请先禁用遥操，再记录/检查/执行关节恢复")
        self.tcp, self.joints = self.read_robot()
        if self.actual_speed > 2:
            raise RuntimeError("机械臂未停稳，请先停止其他控制程序")
        try:
            if name == "home_save":
                if parameters.get("confirmed") is not True:
                    raise RuntimeError("请确认当前姿态适合每次采集的初始点；不能把异常姿态设为初始点")
                self.recovery.check_health()
                if np.any(self.joints < self.lower) or np.any(self.joints > self.upper):
                    raise RuntimeError("当前关节接近遥操限位，不适合记录初始点")
                # A good DATA-COLLECTION home should have local freedom. This
                # stricter test belongs to saving a home, not enabling a hold.
                for axis in range(3):
                    for offset in (-.5, .5):
                        pose = self.tcp.copy(); pose[axis] += offset
                        code, q = self.arm.get_inverse_kinematics(pose_to_sdk(pose).tolist(), input_is_radian=True,
                            return_is_radian=True, limited=False, ref_angles=self.joints.tolist())
                        if code:
                            raise RuntimeError("当前点接近笛卡尔可达边界，不适合初始位姿；请先关节恢复/手动示教")
                        try:
                            continuous_solution(q, self.joints, self.lower, self.upper)
                        except RuntimeError as error:
                            raise RuntimeError("当前点局部逆解不连续，不适合初始位姿；请先关节恢复/手动示教") from error
                if self.settings['rotation_enabled']:
                    base = orientation_rotation(self.tcp[3:])
                    for axis in range(3):
                        for sign in (-1,1):
                            rv = np.zeros(3); rv[axis] = sign*np.deg2rad(.25)
                            pose = self.tcp.copy()
                            pose[3:] = quaternion_wxyz(Rotation.from_rotvec(rv)*base, self.tcp[3:])
                            code, q = self.arm.get_inverse_kinematics(pose_to_sdk(pose).tolist(), input_is_radian=True,
                                return_is_radian=True, limited=False, ref_angles=self.joints.tolist())
                            if code:
                                raise RuntimeError("当前点小角度旋转无解，不适合六维遥操初始点")
                            try:
                                continuous_solution(q,self.joints,self.lower,self.upper)
                            except RuntimeError as error:
                                raise RuntimeError("当前点小角度旋转不连续，不适合六维遥操初始点") from error
                grip = {} if self.gripper is None else self.gripper.snapshot()
                grip_mm = grip.get("position_mm") if grip.get("connected") and grip.get("position_age_s", 999) is not None and grip.get("position_age_s", 999) < 2 else None
                self.recovery.save(self.joints, self.tcp, grip_mm)
                self.status = "初始位姿已保存 · 未移动机械臂"
                return
            if name in ("home_check", "home_start"):
                speed = home_speed(parameters.get("speed_deg_s", HOME_SPEED_DEFAULT))
                if self.recovery.home is None:
                    raise RuntimeError("尚未记录初始位姿；先用关节恢复或 xArm Studio 调整，再设当前为初始点")
                self.recovery.validate_profile(self.recovery.home)
                if name == "home_start":
                    if parameters.get("confirmed") is not True:
                        raise RuntimeError("复位会真实运动；请先确认整条关节路径净空")
                    if parameters.get("home_saved_at") != self.recovery.home["saved_at"]:
                        raise RuntimeError("初始点已改变，请重新检查并确认")
                    expected = vector(parameters.get("checked_start"), 7)
                    if np.max(np.abs(expected - self.joints)) > np.deg2rad(.2):
                        raise RuntimeError("检查后关节位置已改变，请重新检查复位路径")
                    if self.recovery.feedback.get("status") != "checked":
                        raise RuntimeError("请先检查复位路径，不能重复发送复位")
                    if self.recovery.feedback.get("speed_deg_s") != speed:
                        raise RuntimeError("复位速度已改变，请重新检查复位路径")
                plan = self.recovery.plan_home_pose(self.joints,self.recovery_check_cancel,self.lower,self.upper,
                                                    speed_deg_s=speed)
                plan["home_saved_at"] = self.recovery.home["saved_at"]
                self.recovery.feedback = plan
                if name == "home_check":
                    return
                restore_gripper = parameters.get("restore_gripper") is True
                if restore_gripper and (not self.settings["gripper_enabled"] or self.recovery.home.get("gripper_mm") is None):
                    raise RuntimeError("恢复夹爪需启用夹爪，且初始点记录中必须有有效开度")
                if restore_gripper:
                    grip = {} if self.gripper is None else self.gripper.snapshot()
                    if not grip.get('connected') or grip.get('command_fault') or grip.get('position_age_s') is None or grip['position_age_s'] > 2:
                        raise RuntimeError("夹爪状态无效/有故障，拒绝带夹爪复位；可取消该选项仅复位机械臂")
                    if not 0 <= float(self.recovery.home['gripper_mm']) <= 84:
                        raise RuntimeError("初始夹爪开度超范围，请重新记录")
                plan["gripper_target"] = self.recovery.home.get("gripper_mm") if restore_gripper else None
            elif name == "joint_recover":
                if parameters.get("confirmed") is not True:
                    raise RuntimeError("单关节恢复会真实运动，请确认净空")
                axis, step = parameters.get("joint"), float(parameters.get("step_deg", 0))
                if type(axis) is not int or not 0 <= axis < 7 or not np.isfinite(step) or not .5 <= abs(step) <= 5:
                    raise ValueError("单关节恢复：J1–J7，步长 0.5–5°")
                expected = np.deg2rad(vector(parameters.get("start_joints_deg"), 7))
                if np.max(np.abs(expected - self.joints)) > np.deg2rad(.2):
                    raise RuntimeError("确认期间机械臂关节已改变，请重新确认恢复动作")
                target = self.joints.copy(); target[axis] += np.deg2rad(step)
                plan = self.recovery.plan(self.joints, target, f"J{axis+1} {'+' if step>0 else '−'}{abs(step):g}°", self.recovery_check_cancel)
                self.recovery.feedback = plan
            else:
                raise ValueError(f"未知恢复命令 {name}")
            self.start_recovery(plan)
        except Exception as error:
            self.recovery.feedback = dict(self.recovery.feedback, status="rejected", message=str(error))
            raise

    def recovery_check_cancel(self):
        if self.stopping or self.clock() - self.heartbeat > .8:
            raise RuntimeError("界面心跳超时/已关闭，取消复位检查")

    def start_recovery(self, plan):
        self.recovery_check_cancel()
        # Acquire stop ownership BEFORE any mode/enable command. All failures
        # in this block cancel the trajectory; the gripper stays untouched.
        self.recovery.motion = dict(plan, started=self.clock(), deadline=self.clock()+
            max(10, plan["max_joint_travel_deg"]/plan["speed_deg_s"]*2+5))
        if self.gripper is not None:
            self.gripper.cancel_pending()
        try:
            self.checked(self.arm.set_state(4), "准备关节复位停机")
            code, stopped = self.arm.get_state()
            self.checked(code, "核验复位停机")
            if stopped != 4:
                raise RuntimeError("停止状态未确认，拒绝关节复位")
            self.checked(self.arm.set_mode(0), "进入关节位置模式")
            deadline = time.monotonic()+.6
            while self.arm.mode != 0:
                if time.monotonic() > deadline:
                    raise RuntimeError("关节位置模式切换超时")
                time.sleep(.01)
            tcp, q = self.read_robot()
            if np.max(np.abs(q - plan["start"])) > np.deg2rad(.2):
                raise RuntimeError("复位预检期间关节被其他控制源改变，拒绝运动")
            self.recovery_check_cancel()
            code, codes = self.arm.get_err_warn_code()
            self.checked(code, "复核复位错误状态")
            if codes[0] or codes[1] not in (0, 14):
                raise RuntimeError(f"复位前有错误/警告 {codes}，不自动清故障")
            if codes[1] == 14:
                # Explicitly confirmed joint recovery, AFTER checking the path.
                self.checked(self.arm.clean_warn(), "清除已确认的历史无解警告14")
            self.checked(self.arm.motion_enable(enable=True), "关节复位使能")
            self.checked(self.arm.set_state(0), "进入复位保持状态")
            self.checked(self.arm.set_servo_angle(angle=q.tolist(), is_radian=True, speed=.1,
                mvacc=.3, wait=True, timeout=2), "初始化复位当前位置保持目标")
            after_tcp, after_q = self.read_robot()
            if np.linalg.norm(after_tcp[:3]-tcp[:3]) > .5 or np.max(np.abs(after_q-q)) > np.deg2rad(.2):
                raise RuntimeError("复位保持初始化发生位移，已停止")
            self.recovery_check_cancel()
            self.checked(self.arm.set_servo_angle(angle=plan["target"], is_radian=True,
                speed=np.deg2rad(plan['speed_deg_s']), mvacc=np.deg2rad(plan['acceleration_deg_s2']),
                wait=False, radius=None), "发送关节复位")
            self.recovery.feedback = dict(plan, status="moving", message=f"{plan['label']} · {plan['speed_deg_s']:g}°/s · 空格/Esc 停止")
            self.last_tick = self.clock()
        except Exception:
            self.release("关节复位失败 · 已停止", disarm=True)
            raise

    def tick_recovery(self, now):
        motion = self.recovery.motion
        if self.stopping or now-self.heartbeat > .8:
            self.release("复位心跳超时/窗口关闭 · 已停止", disarm=True)
            return
        if getattr(self.arm, "has_error", False) or self.arm.mode != 0 or self.arm.state in (3, 4, 5):
            raise RuntimeError("复位期间机器人错误或模式/状态改变，已停止")
        if now > motion["deadline"]:
            raise RuntimeError("Pose/关节恢复超时，已停止")
        start, target = np.array(motion["start"]), np.array(motion["target"])
        if np.any(self.joints < self.recovery.lower) or np.any(self.joints > self.recovery.upper):
            raise RuntimeError("复位实测关节超出硬件限位，已停止")
        if np.any(self.joints < np.minimum(start,target)-np.deg2rad(.5)) or np.any(self.joints > np.maximum(start,target)+np.deg2rad(.5)):
            raise RuntimeError("复位关节偏离检查路径，已停止")
        delta = target-start
        progress = float(np.dot(self.joints-start, delta)/np.dot(delta,delta)) if np.dot(delta,delta)>1e-12 else 1.
        if np.max(np.abs(self.joints-(start+np.clip(progress,0,1)*delta))) > np.deg2rad(2):
            raise RuntimeError("复位同步关节轨迹偏离，已停止")
        self.requested = self.bounded = as_quaternion_pose(motion.get("target_pose_quat", motion["target_tcp"]))
        self.status = f"{motion['label']} · {np.clip(progress,0,1)*100:.0f}% · {motion['speed_deg_s']:g}°/s"
        self.recovery.feedback.update(progress=float(np.clip(progress,0,1)))
        goal = as_quaternion_pose(motion.get('target_pose_quat', motion['target_tcp']))
        joint_reached = motion.get('type')=='pose' or np.max(np.abs(self.joints-target)) < np.deg2rad(.1)
        if (joint_reached and np.linalg.norm(self.tcp[:3]-goal[:3]) < 1
                and rotation_distance(self.tcp[3:],goal[3:]) < np.deg2rad(.5) and now-motion['started'] > .2):
            code, states = self.arm.get_joint_states(is_radian=True, num=2)
            self.checked(code, "核验复位关节速度")
            if np.max(np.abs(vector(states[1],7))) > np.deg2rad(.5):
                return
            grip_target = motion.get("gripper_target")
            self.release("复位完成 · 已停止，重新启用遥操后按住扳机", disarm=True)
            self.mapper.reset_origin(self.tcp)
            if grip_target is not None:
                self.request_gripper(grip_target, controller=False)
            self.recovery.feedback = dict(self.recovery.feedback, status="completed", progress=1,
                message="复位完成；机械臂保持只读预览，不自动恢复遥操" + ("；已请求恢复初始夹爪开度" if grip_target is not None else "；夹爪保持"))

    def select_matrix(self, parameters):
        """Select a stored map while stopped; never enable robot movement."""
        if self.mode != 'preview' or self.servo_started or self.pause_settle is not None:
            raise RuntimeError('请先禁用遥操并等待停稳，再选择矩阵')
        choice = parameters.get('choice')
        updated = dict(self.settings)
        if choice == 'old':
            updated.update(empirical_matrix=None, empirical_frame=None)
        elif choice == 'new':
            matrix = self.settings['saved_empirical_matrix']
            frame = self.settings['saved_empirical_frame']
            if matrix is None:
                raise ValueError('尚无已保存的 6×6 矩阵')
            current = [(self.source or {}).get('input_backend'), (self.source or {}).get('frame_id')]
            if current[0] is not None and current[0] != frame[0]:
                raise ValueError('已保存矩阵的输入来源与当前手柄不同')
            # Explicit selection associates the stored map with this session.
            # Choosing settings is allowed without live tracking. Starting
            # motion still checks freshness and the reference frame separately.
            # With no known session, keep the saved binding until reselected.
            binding = current if all(isinstance(v, str) and v for v in current) else frame
            updated.update(empirical_matrix=matrix, empirical_frame=binding)
        else:
            raise ValueError('矩阵选择必须是 old 或 new')
        self.command('configure', updated)
        self.status = '矩阵已应用 · 只读预览，手动启用遥操后接管'

    def tick(self):
        now = self.clock()
        elapsed = max(now - self.last_tick, 0)
        dt = min(elapsed, CONTROL_PERIOD)
        if 0 < elapsed < .5:
            self.cycle_times.append(elapsed)
            self.control_hz = len(self.cycle_times) / sum(self.cycle_times)
        self.last_tick = now
        self.tcp, self.joints = self.read_robot()
        if self.previous_tcp is not None and now > self.previous_tcp_time:
            self.actual_speed = float(np.linalg.norm(self.tcp[:3] - self.previous_tcp[:3]) / (now - self.previous_tcp_time))
            self.actual_angular_speed = float(np.rad2deg(rotation_distance(self.tcp[3:],self.previous_tcp[3:])) / (now-self.previous_tcp_time))
        self.previous_tcp, self.previous_tcp_time = self.tcp.copy(), now
        self.limit_reason = ""
        self.smoothing_feedback = {}
        if self.stopping:
            self.release("控制器退出 · 已停止", disarm=True)
            return
        if self.recovery.motion is not None:
            self.tick_recovery(now)
            return
        if (self.servo_started or self.gripper_held) and self.settings["gripper_enabled"] and self.gripper is not None and self.gripper.snapshot()["command_fault"]:
            self.release("夹爪指令失败 · 已暂停机械臂，保持抓持，不自动松开", disarm=True)
            self.last_error = self.gripper.snapshot()["error"]
            return
        if self.mode == "motion" and now - self.heartbeat > 0.8:
            self.release("界面心跳超时 · 已停止", disarm=True)
        if self.mode=='motion':
            self.update_gripper_button()  # independent of trigger and stop settling
        if self.settle_pause(now):
            return
        if self.servo_started and elapsed > .05:
            self.release("控制周期超过 50 ms · 已停止，请检查连接后重新允许遥操", disarm=True)
            self.last_error = f"控制周期超时 {elapsed * 1000:.0f} ms"
            return
        if self.mode == "motion" and not self.mapper.contains(self.tcp, .5):
            self.release("机器人位置越出原始范围 · 已停止，请重新允许手柄遥操", disarm=True)
            self.last_error = "当前 TCP 已改变，原始试动起点失效；需要在当前位置重新检查并允许遥操"
            return
        if self.mode == "motion" and not self.servo_started and self.mapper.radius is None:
            hold_angle = rotation_distance(self.tcp[3:], self.hold_tcp[3:])
            if np.linalg.norm(self.tcp[:3] - self.hold_tcp[:3]) > 6 or hold_angle > np.deg2rad(.5):
                self.release("暂停期间机械臂被其他控制源移动 · 已释放，请重新允许遥操", disarm=True)
                self.last_error = "暂停保持位置已改变，需要重新检查当前位置"
                return
        if self.servo_started and getattr(self.arm, "has_error", False):
            raise RuntimeError(f"机械臂报告错误 {self.arm.error_code}")
        if self.servo_started and (getattr(self.arm, "mode", 1) != 1 or getattr(self.arm, "state", 0) in (3, 4, 5)):
            self.release("机器人模式/状态被改变 · 已释放控制", disarm=True)
            return
        s = self.source or {}
        invalid = s.get("invalid_count", 0)
        pause = s.get("pause_generation", 0)
        count = s.get("packet_count", 0)
        trigger_release = s.get("trigger_release_generation", 0)
        trigger_release_edge = trigger_release > self.last_trigger_release
        self.last_trigger_release = trigger_release
        lost_frame = invalid > self.last_invalid or count < self.last_count
        pause_edge = pause > self.last_pause
        self.last_invalid, self.last_pause, self.last_count = invalid, pause, count
        self.requested = self.bounded = self.tcp.copy()
        if not self.pause_fresh():
            self.release("只读预览 · 等待有效手柄追踪（机械臂不跟随）" if self.mode == "preview" else "数据丢失/无效 · 已释放，恢复后重新按遥操按键")
            return
        if pause_edge or s.get("motion_enabled") is False:
            if (s.get('input_backend') == 'webxr' or
                    (self.settings['teleop_button']=='trigger' and self.settings['gripper_button']=='b')):
                cause = s.get('reason') if s.get('valid') is False else s.get('last_invalid_reason') if lost_frame else None
                message = (("遥操输入中断" + (f"：{cause}" if cause else "") + " · 恢复后松开再按住扳机接管") if self.mode == 'motion' else
                           "等待头显遥操输入 · 只读预览，机械臂不移动")
            else:
                message = (("B 暂停 · 恢复绿色后重新按扳机接管" if self.settings["teleop_button"] == "trigger" else "B 暂停 · 已允许按 B 继续遥操") if self.mode == "motion" else "B 暂停 · 机械臂已停止")
            self.release(message)
            self.release_seen = self.mode == "motion" and self.settings["teleop_button"] == "b"
            if s.get("motion_enabled") is False:
                return
        if lost_frame and not pause_edge:
            self.release("遥操输入曾中断：" + str(s.get('last_invalid_reason') or '收到无效帧') +
                         " · 已释放，恢复后松开再按住扳机")
            return
        if not self.fresh():
            self.release("位姿超时 · 已释放，恢复后重新按遥操按键")
            return
        if (self.mode=='motion' and self.settings['gripper_enabled'] and self.settings['gripper_button']=='b'
                and type(s.get('b_pressed')) is not bool):
            self.release('等待独立 B 按键输入 · 请点“打开头显输入”；旧 App 的 B 不能作为夹爪键')
            return
        if self.mode == "motion" and self.settings["teleop_button"] == "trigger":
            pressed = s.get("trigger_pressed")
            if type(pressed) is not bool:
                self.release("扳机状态未上报/无效 · 已释放，不能接管")
                return
            if trigger_release_edge:
                self.release("已暂停 · 遥操仍启用；移动手柄调整握姿，再按住继续",keep_gripper=True)
                if pressed is True:
                    # A short release/repress between snapshots must stop first.
                    # Then debounce the NEW press, never continue the old goal.
                    self.motion_button.reset(False, now)
                    self.motion_button.update(True, now)
                return
            # Dead-man clutch: release is immediate on the first received
            # False frame, not debounced; press alone needs 20 ms stability.
            if not pressed:
                self.release("已暂停 · 遥操仍启用；移动手柄调整握姿，再按住继续",keep_gripper=True)
                return
            if self.motion_button.update(pressed, now):
                self.release_seen = True
        try:
            hand, quaternion = self.hand_pose()
        except ValueError as error:
            self.release(str(error))
            return
        if self.settings.get('empirical_matrix') is not None and self.settings['empirical_frame'] != [s.get('input_backend'), s.get('frame_id')]:
            self.release('六维矩阵参考坐标已改变 · 重新选择矩阵后再启用', disarm=True)
            return
        hand_orientation = hand_rotation(quaternion) if self.settings["rotation_enabled"] or self.mapper.empirical_matrix is not None else None
        hand_increment = np.zeros(3) if self.previous_hand is None else hand-self.previous_hand
        angular_increment = (np.zeros(3) if hand_orientation is None or self.previous_hand_rotation is None
            else (hand_orientation*self.previous_hand_rotation.inv()).as_rotvec())
        self.previous_hand = hand.copy()
        self.previous_hand_rotation = hand_orientation
        if self.mode == "motion" and not self.release_seen:
            self.status = "就绪 · 松开后重新按住扳机" if self.settings["teleop_button"] == "trigger" else "等待接管 · 按 B 暂停→继续，或点击接管"
            return
        if self.mapper.hand_anchor is None:
            self.status = "只读预览 · 起点已建立" if self.mode == "preview" else "已接管 · 相对位移跟随"
            if self.mode == "motion":
                metric = self.pose_guard.evaluate(self.joints, self.clock()+.04, force=True)
                self.pose_guard.publish(metric)
                if metric['sigma_min'] <= SIGMA_RECOVERY_FLOOR:
                    self.protective_pause('当前位置接近奇异姿态，请检查复位/关节恢复路径')
                    return
                code, codes = self.arm.get_err_warn_code()
                self.checked(code, "读取错误状态")
                if codes != [0, 0]:
                    raise RuntimeError(f"机器人错误/警告 {codes}")
                self.servo_started = True
                try:
                    self.tcp, self.joints = prepare_servo_hold(self.arm, allowed=self.refresh_motion_permission)
                except TakeoverCancelled:
                    # A release/tracking interruption during initialization is a
                    # cancelled takeover. Still stop and verify settling, and
                    # require a new press; never resume a held/stale trigger.
                    source = self.source or {}
                    faults = []
                    if self.stopping:
                        faults.append('控制器正在退出')
                    if self.clock()-self.heartbeat > .8:
                        faults.append('GUI 心跳超时')
                    if getattr(self.arm,'has_error',False):
                        faults.append('机器人报告错误')
                    if self.settings.get('empirical_matrix') is not None and self.settings['empirical_frame'] != [source.get('input_backend'),source.get('frame_id')]:
                        faults.append('参考坐标已改变')
                    critical = bool(faults)
                    reason = '、'.join(faults) if critical else (
                        source.get('reason','位姿无效或过期') if not self.fresh() or source.get('motion_enabled') is not True else
                        '扳机松开' if source.get('trigger_pressed') is False else '初始化期间发生追踪中断或扳机松开')
                    self.release('接管已取消（' + reason + '） · ' + ('请重新检查并启用遥操' if critical else
                        '遥操仍启用；松开后再按住扳机继续'), disarm=bool(critical), keep_gripper=not critical)
                    return
                self.hold_count += 1
                self.ik_feedback = {}
                self.ik_updated = None
                self.last_error = ""
                self.command_tcp = self.tcp.copy()
                self.last_command_joints = self.joints.copy()
                self.last_command_time_s = self.clock()
                if self.poll_source is not None:
                    self.poll_source()
                if not self.motion_input_allowed():
                    self.release("初始化时手柄失效或暂停 · 已停止，请重新按遥操按键")
                    return
                try:
                    hand, quaternion = self.hand_pose(reanchor=True)
                except ValueError as error:
                    self.release(str(error))
                    return
                if not self.mapper.contains(self.tcp, .5):
                    raise RuntimeError("初始化位置越出原始试动范围")
                self.motion_started = self.motion_started or now
            self.mapper.engage(hand, self.tcp, quaternion)
            self.previous_hand = hand.copy()
            self.previous_hand_rotation = hand_rotation(quaternion) if self.settings["rotation_enabled"] or self.mapper.empirical_matrix is not None else None
            self.cartesian_velocity = np.zeros(3)
            self.angular_velocity = np.zeros(3)
            self.joint_velocity = np.zeros(7)
            self.last_tick = self.clock()  # hold initialization is intentionally blocking
            return
        self.requested, self.bounded = self.mapper.target(hand, quaternion)
        self.backlog_feedback = {}
        reasons = []
        if self.mode == "preview":
            self.status = "只读目标预览 · 机械臂不移动"
            return
        if not self.mapper.contains(self.tcp, .5):
            raise RuntimeError("实测 TCP 越出试动范围")
        metric = self.pose_guard.evaluate(self.joints, now+.04)
        command_metric = metric
        if self.last_command_joints is not None:
            command_metric = self.pose_guard.evaluate(self.last_command_joints, now+.04)
            metric = min((metric, command_metric), key=lambda item: item['sigma_min'])
        self.pose_guard.publish(metric)
        if metric['sigma_min'] <= SIGMA_RECOVERY_FLOOR:
            self.protective_pause('接近奇异/伸直姿态 · 已暂停；检查恢复路径，不继续向该方向推')
            return
        ratio = metric['speed_ratio']
        max_lead = min(self.settings['tracking_lead_mm'], max(1.0, self.mapper.speed * 0.08))
        # Prioritize response: keep only a short braking allowance rather
        # than accumulating a distant goal when the hand outruns the robot.
        if self.mapper.empirical_matrix is not None:
            linear_step, angular_step = self.mapper.last_linear_increment, self.mapper.last_angular_increment
        else:
            linear_step = self.mapper.matrix @ hand_increment*1000*self.mapper.scale
            angular_step = np.linalg.det(self.mapper.matrix)*self.mapper.matrix @ angular_increment *self.mapper.rotation_axes*self.mapper.rotation_scale
        bounded_goal, backlog = bound_follow_goal(self.bounded,self.command_tcp,
            self.cartesian_velocity,self.angular_velocity,self.settings, linear_step, angular_step)
        self.backlog_feedback = backlog
        if backlog['discarded_mm'] > 1e-6 or backlog['discarded_deg'] > 1e-6:
            self.mapper.discard_target_excess(self.bounded,bounded_goal)
            self.bounded = bounded_goal
            if backlog['discarded_mm'] > 1e-6:
                reasons.append('低延迟平移 · 超速位移已压缩')
            if backlog['discarded_deg'] > 1e-6:
                reasons.append('旋转跟随 · 已丢弃超前旋转手势')
        direction_info = None
        if metric['sigma_min'] >= SIGMA_SLOW+2:
            self.translation_priority = False
        if self.translation_priority:
            revised = self.bounded.copy(); revised[3:] = self.command_tcp[3:]
            self.mapper.discard_target_excess(self.bounded,revised)
            self.bounded = revised
            self.angular_velocity = np.zeros(3)
            reasons.append('旋转暂锁，优先平移退出 · 松开扳机可重选')
        if metric['sigma_min'] < SIGMA_SLOW:
            if self.limit_target_backlog(max_lead):
                reasons.append('受限目标有界跟随 · 多余手势已丢弃')
            ref = self.joints if self.last_command_joints is None else self.last_command_joints
            direction_info = choose_direction(self.arm,self.pose_guard,ref,self.command_tcp,self.bounded,
                command_metric,self.lower,self.upper,now+.04,self.clock)
            ratio = direction_speed_ratio(metric,direction_info['direction'],self.mapper.speed,
                self.settings['rotation_speed_deg_s'],self.settings['joint_speed_rad_s'])
            if direction_info['translation_only']:
                self.translation_priority = True
                revised = self.bounded.copy(); revised[3:] = self.command_tcp[3:]
                self.mapper.discard_target_excess(self.bounded,revised)
                self.bounded = revised
                self.angular_velocity = np.zeros(3)
                reasons.append('旋转受限，优先平移退出')
            self.pose_guard.feedback.update(direction_info=direction_info, effective_speed_ratio=ratio)
            if ratio == 0:
                self.hold_constrained_target('当前手势趋向受限姿态 · 保持；反向移动手柄可重新探测',direction_info)
                return
            reasons.append(f'方向保护限速 {ratio*100:.1f}%')
        # Integrate a bounded command trajectory, instead of restarting every
        # step from delayed measured feedback (which made the follower sluggish).
        previous_command = self.command_tcp.copy()
        if np.linalg.norm(self.bounded[:3] - previous_command[:3]) > self.mapper.speed * dt + 0.02:
            reasons.append("TCP 限速")
        target, planned_velocity = smooth_target(self.bounded, previous_command, self.cartesian_velocity, dt,
                                                 self.mapper.speed*ratio, self.settings["acceleration_mm_s2"])
        if np.linalg.norm(planned_velocity - self.cartesian_velocity) > .01:
            reasons.append("加减速平滑")
        max_lead = min(self.settings['tracking_lead_mm'], max(1.0, self.mapper.speed * 0.08))
        if np.linalg.norm(target[:3] - self.tcp[:3]) > max_lead + 0.01:
            reasons.append("目标超前限制")
        target = tether(target, self.tcp, max_lead)
        angular_speed = np.deg2rad(self.settings["rotation_speed_deg_s"])
        if self.settings["rotation_enabled"]:
            target[3:], planned_angular_velocity = smooth_rotation(self.bounded[3:], previous_command[3:], self.angular_velocity, dt,
                                                                  angular_speed*ratio, np.deg2rad(self.settings["rotation_acceleration_deg_s2"]))
            angular_error = (orientation_rotation(target[3:]) * orientation_rotation(self.tcp[3:]).inv()).as_rotvec()
            if np.linalg.norm(angular_error) > np.deg2rad(3):
                target[3:] = quaternion_wxyz(Rotation.from_rotvec(angular_error * np.deg2rad(3) / np.linalg.norm(angular_error)) * orientation_rotation(self.tcp[3:]), target[3:])
                reasons.append("姿态超前保护")
            if rotation_distance(self.bounded[3:], previous_command[3:]) > angular_speed * dt + np.deg2rad(.02):
                reasons.append("旋转限速/平滑")
        else:
            planned_angular_velocity = np.zeros(3)
        if (np.linalg.norm(target[:3] - previous_command[:3]) < 1e-6 and rotation_distance(target[3:], previous_command[3:]) < 1e-8
                and np.max(np.abs(self.joint_velocity)) < 1e-8):
            # Nothing is sent: do not integrate an imaginary command velocity.
            # Only skip an identical COMMAND, not a small step near actual
            # feedback: dropping the first .009-degree acceleration step
            # prevents a 90-degree/s^2 rotation from ever starting.
            self.cartesian_velocity = np.zeros(3)
            self.angular_velocity = np.zeros(3)
            self.status = "已接管 · 保持当前位置"
            return
        previous_joints = self.joints if self.last_command_joints is None else self.last_command_joints
        solution, subdivisions = self.solve_follow_target(target, previous_command, previous_joints, now)
        if subdivisions:
            reasons.append(f"IK 自适应缩步 ×{.5 ** subdivisions:g}")
            if self.limit_target_backlog(max_lead):
                reasons.append('逆解受限目标有界跟随 · 多余手势已丢弃')
        command = joint_acceleration_step(previous_joints, solution, self.joint_velocity, dt,
                                         self.settings["joint_speed_rad_s"]*ratio)
        speed_limited = limited_joints(previous_joints, solution, dt, self.settings['joint_speed_rad_s']*ratio)
        # Record each stage before FK fitting so a high Cartesian ceiling
        # cannot hide downstream joint-step attenuation in diagnostics.
        joint_step = speed_limited-previous_joints
        step_norm = float(np.linalg.norm(joint_step))
        self.smoothing_feedback = dict(
            planned_tcp_speed_mm_s=float(np.linalg.norm(target[:3]-previous_command[:3])/max(dt,1e-9)),
            joint_step_fraction=float(np.linalg.norm(command-previous_joints)/step_norm) if step_norm>1e-12 else 1.,
            dt_s=float(dt),
            previous_joint_velocity_rad_s=self.joint_velocity.tolist(),
            desired_joint_velocity_rad_s=(joint_step/max(dt,1e-9)).tolist(),
            smoothed_joint_velocity_rad_s=((command-previous_joints)/max(dt,1e-9)).tolist())
        if np.linalg.norm(speed_limited - solution) > 1e-6:
            joint = int(np.argmax(np.abs(solution-previous_joints)))
            required = np.rad2deg(abs(solution[joint]-previous_joints[joint])/max(dt,1e-9))
            reasons.append(f"关节限速 J{joint+1}（需 {required:.0f}°/s，上限 {np.rad2deg(self.settings['joint_speed_rad_s'])*ratio:.0f}°/s）")
        if np.linalg.norm(command-speed_limited) > 1e-6:
            reasons.append('关节加速/换向平滑')
        self.limit_reason = " · ".join(reasons)
        fitted = self.fit_follow_command(command, previous_joints, previous_command, dt,
                                         max_lead, angular_speed*ratio, now+.04,
                                         speed_limit=self.mapper.speed*ratio)
        if fitted is None:
            self.protective_pause('下发目标无法满足跟随边界 · 已暂停并清除旧目标，请松开扳机后检查')
            return
        command, fk, fraction = fitted
        if fraction < 1:
            reasons.append(f'下发目标边界回缩 ×{fraction:g}')
            self.limit_reason = ' · '.join(reasons)
        predicted = self.pose_guard.evaluate(command, now+.04, force=True)
        # Different linear/angular acceleration histories can make the next
        # smoothed step point somewhere other than the long-term 6D goal.
        # Recheck this case: never trust a retreat label on the goal alone.
        if (direction_info is not None and direction_info['direction']=='retreating'
                and predicted['sigma_min'] < command_metric['sigma_min']-.03
                and self.settings['rotation_enabled']):
            translation_goal = self.bounded.copy(); translation_goal[3:] = previous_command[3:]
            translation_info = choose_direction(self.arm,self.pose_guard,previous_joints,previous_command,
                translation_goal,command_metric,self.lower,self.upper,now+.04,self.clock)
            if translation_info['direction']=='retreating':
                translation_target = target.copy(); translation_target[3:] = previous_command[3:]
                solution, subdivisions = self.solve_follow_target(translation_target,previous_command,previous_joints,now)
                command = joint_acceleration_step(previous_joints,solution,self.joint_velocity,dt,
                    self.settings['joint_speed_rad_s']*ratio)
                fitted = self.fit_follow_command(command,previous_joints,previous_command,dt,max_lead,
                    angular_speed*ratio,now+.04,speed_limit=self.mapper.speed*ratio)
                if fitted is not None:
                    command,fk,fraction=fitted
                    predicted=self.pose_guard.evaluate(command,now+.04,force=True)
                    direction_info=translation_info
                    direction_info['translation_only']=True
                    self.translation_priority=True
                    self.mapper.discard_target_excess(self.bounded,translation_goal)
                    self.bounded=translation_goal
                    self.angular_velocity=np.zeros(3)
                    reasons.append('平滑后旋转受限，优先平移退出')
                else:
                    self.hold_constrained_target('平移退出步长仍受限 · 保持，请调整手势',translation_info)
                    return
        if predicted['sigma_min'] <= SIGMA_RECOVERY_FLOOR:
            self.pose_guard.publish(predicted)
            self.protective_pause('下一步将进入奇异/伸直保护区 · 未下发，松开扳机后检查恢复路径')
            return
        retreat = direction_info is not None and direction_info['direction']=='retreating'
        if ((not retreat and predicted['sigma_min'] <= SIGMA_APPROACH_FLOOR) or
                (retreat and predicted['sigma_min'] < command_metric['sigma_min']-.03)):
            self.hold_constrained_target('候选动作未通过方向保护 · 保持；请反向移动或调整手腕',direction_info)
            return
        self.pose_guard.publish(min((metric, predicted), key=lambda item: item['sigma_min']))
        if direction_info is not None:
            self.pose_guard.feedback.update(direction_info=direction_info,effective_speed_ratio=ratio)
        if ratio < 1:
            # Compare against the previous COMMAND, not delayed measured
            # joints. FK/Jacobian quantization has a small deadband; do not
            # call an ambiguous change an escape or relax any safety limits.
            improvement = predicted['sigma_min']-command_metric['sigma_min']
            direction = 'retreating' if improvement > .05 else 'approaching' if improvement < -.05 else 'neutral'
            self.pose_guard.feedback.update(direction=direction, sigma_change=improvement)
            reasons.append({'retreating':'正在远离奇异区', 'approaching':'正在接近奇异区 · 限速',
                            'neutral':'可解性变化较小 · 有界跟随'}[direction])
            self.limit_reason = ' · '.join(reasons)
        if np.linalg.norm(fk[:3] - previous_command[:3]) > self.mapper.speed * ratio * dt + 0.04:
            raise RuntimeError("命令 TCP 步长超过低速限制")
        if np.linalg.norm(fk[:3] - self.tcp[:3]) > max_lead + 0.1:
            raise RuntimeError("目标超前实测位置过多，已释放控制")
        if not self.mapper.contains(fk, .05):
            raise RuntimeError("命令越出试动范围")
        if rotation_distance(fk[3:], previous_command[3:]) > angular_speed * ratio * dt + np.deg2rad(.05):
            raise RuntimeError("命令旋转步长超过限速")
        if rotation_distance(fk[3:], self.tcp[3:]) > np.deg2rad(3.1):
            raise RuntimeError("姿态目标超前实测过多，已释放")
        if not self.settings["rotation_enabled"] and rotation_distance(fk[3:], self.mapper.tcp_anchor[3:]) > np.deg2rad(.5):
            raise RuntimeError("命令姿态偏离固定姿态")
        # Recheck input after SDK round trips; never execute a stale target.
        if self.poll_source is not None:
            self.poll_source()
        if self.clock() - now > .05 or not self.motion_input_allowed():
            self.release("扳机松开或输入失效 · 已停止")
            return
        self.checked(self.arm.set_servo_angle_j(command.tolist(), is_radian=True, speed=max(0.35, self.settings["joint_speed_rad_s"]), mvacc=0.75), "发送关节伺服目标")
        self.command_count += 1
        self.last_command_time_s = self.clock()
        self.command_tcp = fk.copy()
        self.last_command_joints = command.copy()
        self.joint_velocity = (command-previous_joints)/max(dt, 1e-9)
        self.cartesian_velocity, self.angular_velocity = command_velocities(
            previous_command, fk, dt, self.mapper.speed*ratio, angular_speed*ratio)
        self.smoothing_feedback['committed_tcp_speed_mm_s'] = float(np.linalg.norm(self.cartesian_velocity))
        self.smoothing_feedback['final_fit_fraction'] = float(fraction)
        self.status = f"相对位移跟随中 · TCP 上限 {self.mapper.speed*ratio:g} mm/s / {self.mapper.scale:g}×"

    def hold_constrained_target(self, message, direction_info=None):
        # The previously validated servo target remains unchanged. Do not send
        # an unsafe candidate, drop the unreachable gesture, retain the clutch
        # so the NEXT reversed hand increment can be evaluated immediately.
        self.mapper.discard_target_excess(self.bounded,self.command_tcp)
        self.bounded = self.command_tcp.copy()
        self.cartesian_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.joint_velocity = np.zeros(7)
        self.status = self.limit_reason = message
        self.pose_guard.feedback.update(status='direction_hold',message=message,
            direction_info=direction_info,effective_speed_ratio=0.)

    def limit_target_backlog(self, max_lead):
        """Keep the warning/IK-limited goal local without releasing the clutch."""
        old = self.bounded.copy()
        local = tether(old, self.tcp, max_lead)
        delta = (orientation_rotation(old[3:])*orientation_rotation(self.tcp[3:]).inv()).as_rotvec()
        angle = np.linalg.norm(delta)
        if self.settings['rotation_enabled'] and angle > np.deg2rad(3):
            local[3:] = quaternion_wxyz(Rotation.from_rotvec(delta*np.deg2rad(3)/angle)*
                         orientation_rotation(self.tcp[3:]), local[3:])
        dropped = float(np.linalg.norm(local[:3]-old[:3]))
        dropped_angle = float(np.rad2deg(rotation_distance(local[3:], old[3:])))
        if dropped < 1e-6 and dropped_angle < 1e-6:
            return False
        self.mapper.discard_target_excess(old, local)
        self.bounded = local
        self.backlog_feedback = dict(status='bounded', discarded_mm=dropped,
            discarded_deg=dropped_angle, limit_mm=max_lead, limit_deg=3.)
        return True

    def fit_follow_command(self, candidate, previous_joints, previous_tcp, dt, max_lead, angular_speed, deadline, speed_limit=None):
        """FK-check the FINAL smoothed target, not just the pre-IK Cartesian one.

        Joint-segment interpolation can change the nonlinear TCP path even
        when the IK target was tethered. Backtrack only on the short candidate
        joint segment; never enlarge limits or send an unverified correction.
        Safety braking overrides soft acceleration smoothing, as at a stop.
        """
        self.command_guard = {'status': 'checking', 'attempts': [], 'lead_limit_mm': max_lead}
        speed_limit = self.mapper.speed if speed_limit is None else speed_limit
        for fraction in (1., .5, .25, .125, .0625, .03125, .015625, 0.):
            if self.clock()+.001 >= deadline:
                self.command_guard['status'] = 'timeout'
                raise RuntimeError('下发目标边界检查预算不足，未发送本帧')
            command = previous_joints + (candidate-previous_joints)*fraction
            code, pose = self.arm.get_forward_kinematics(command.tolist(), input_is_radian=True, return_is_radian=True)
            self.checked(code, '下发目标边界正解')
            if self.clock() >= deadline:
                self.command_guard['status'] = 'timeout'
                raise RuntimeError('下发目标边界正解超时，未发送本帧')
            fk = as_quaternion_pose(pose)
            lead = float(np.linalg.norm(fk[:3]-self.tcp[:3]))
            angle = float(rotation_distance(fk[3:], self.tcp[3:]))
            valid = (lead <= max_lead+.1 and angle <= np.deg2rad(3.1)
                and np.linalg.norm(fk[:3]-previous_tcp[:3]) <= speed_limit*dt+.04
                and rotation_distance(fk[3:], previous_tcp[3:]) <= angular_speed*dt+np.deg2rad(.05)
                and self.mapper.contains(fk, .05)
                and np.all(command >= self.lower) and np.all(command <= self.upper)
                and (self.settings['rotation_enabled'] or
                     rotation_distance(fk[3:], self.mapper.tcp_anchor[3:]) <= np.deg2rad(.5)))
            self.command_guard['attempts'].append(dict(fraction=fraction, lead_mm=lead,
                rotation_lead_deg=float(np.rad2deg(angle)), valid=bool(valid)))
            if valid:
                self.command_guard.update(status='ok' if fraction == 1 else 'shortened', fraction=fraction)
                return command, fk, fraction
        self.command_guard['status'] = 'blocked'
        return None

    def protective_pause(self, message):
        """No chasing/re-arming: stop, drop anchors, require a new press."""
        self.pose_guard.feedback.update(status='paused', message=message)
        self.release(message, keep_gripper=True)
        self.requested = self.bounded = self.tcp.copy()

    def telemetry(self):
        filtered_hand = (self.input_filter.position if self.settings['input_filter']['enabled']
                         else (self.source or {}).get('position_m'))
        hand_delta, hand_rotation_delta = (self.mapper.aligned_hand_delta(filtered_hand)
            if self.fresh() and filtered_hand is not None else (None,None))
        source_age = (max(0,self.clock()-self.source_received)+self.source['valid_age_s']
            if self.fresh() else None)
        return {"type": "robot", "timestamp": time.time(), "run_id": self.run_id, "mode": self.mode,
                "last_stop_event": self.last_stop_event,
                "follow_algorithm": "responsive-coordinated-velocity-v4",
                "empirical_rotation_mode": "automatic-local-charts-v1",
                "empirical_reanchors": self.mapper.empirical_reanchors,
                "active": self.servo_started or self.recovery.motion is not None, "status": self.status, "error": self.last_error,
                "tcp_mm": self.tcp[:3].tolist(), "target_mm": self.bounded[:3].tolist(),
                "orientation_representation": "quaternion_wxyz",
                "tcp_quaternion_wxyz": self.tcp[3:].tolist(),
                "target_quaternion_wxyz": self.bounded[3:].tolist(),
                "command_quaternion_wxyz": self.command_tcp[3:].tolist(),
                "tcp_rpy_deg": pose_rpy_deg(self.tcp), "target_rpy_deg": pose_rpy_deg(self.bounded),
                "requested_mm": self.requested[:3].tolist(), "command_tcp_mm": self.command_tcp[:3].tolist(),
                "command_rpy_deg": pose_rpy_deg(self.command_tcp),
                "command_valid": self.servo_started and self.last_command_joints is not None,
                "command_tracking_error_mm": float(np.linalg.norm(self.command_tcp[:3]-self.tcp[:3])) if self.servo_started and self.last_command_joints is not None else None,
                "command_rotation_error_deg": float(np.rad2deg(rotation_distance(self.command_tcp[3:],self.tcp[3:]))) if self.servo_started and self.last_command_joints is not None else None,
                "ik_feedback": self.ik_feedback,
                "smoothing_feedback": self.smoothing_feedback,
                "input_filter": dict(self.input_filter.snapshot(), enabled=self.settings['input_filter']['enabled'],
                                     valid=self.fresh() and self.input_filter.sample_time is not None),
                "ik_feedback_age_s": None if self.ik_updated is None else max(0,self.clock()-self.ik_updated),
                "origin_mm": self.mapper.origin[:3].tolist(), "joints_deg": np.rad2deg(self.joints).tolist(),
                "max_displacement_mm": np.abs(self.tcp[:3] - self.mapper.origin[:3]).tolist(),
                "hand_anchor_m": None if self.mapper.hand_anchor is None else self.mapper.hand_anchor.tolist(),
                "hand_position_m": None if self.source is None else self.source.get("position_m"),
                "hand_aligned_delta_mm": hand_delta, "hand_aligned_rotation_deg": hand_rotation_delta,
                "source_age_ms": None if source_age is None else source_age*1000,
                "source_hz": None if not self.fresh() else self.source.get('valid_hz'),
                "sent_commands": self.command_count, "hold_commands": self.hold_count, "source_valid": self.fresh(), "speed_mm_s": self.mapper.speed,
                "range_mm": self.mapper.radius, "control_hz": self.control_hz, "control_target_hz": CONTROL_HZ,
                "effective_tracking_lead_mm": min(self.settings['tracking_lead_mm'],max(1.,self.mapper.speed*.08)),
                "translation_scale": self.mapper.scale, "settings": self.settings,
                "settings_revision": self.settings_revision, "limit_reason": self.limit_reason,
                "actual_speed_mm_s": self.actual_speed, "following_error_mm": float(np.linalg.norm(self.bounded[:3] - self.tcp[:3])),
                "actual_angular_speed_deg_s": self.actual_angular_speed,
                "recovery": self.recovery.snapshot(),
                "rotation_error_deg": float(np.rad2deg(rotation_distance(self.bounded[3:], self.tcp[3:]))),
                "teleop_button": self.settings["teleop_button"],
                "teleop_armed": self.mode=='motion',
                "teleop_paused": self.mode=='motion' and not self.servo_started,
                "pause_settling": self.pause_settle is not None,
                "pause_feedback": self.pause_feedback,
                "pose_guard": self.pose_guard.snapshot(),
                "command_guard": self.command_guard,
                "backlog_feedback": self.backlog_feedback,
                "translation_priority": self.translation_priority,
                "trigger_pressed": None if not self.fresh() else self.source.get("trigger_pressed"),
                "a_pressed": None if not self.fresh() else self.source.get("a_pressed"),
                "b_pressed": None if not self.fresh() else self.source.get("b_pressed"),
                "gripper_button": self.settings['gripper_button'], "gripper_held": self.gripper_held,
                "gripper_button_mode": 'toggle' if self.settings['gripper_button']=='b' else 'hold',
                "gripper": None if self.gripper is None else self.gripper.snapshot()}


def serve(ip, source_port=8130, duration=None, command_path=COMMAND_PATH):
    from xarm.core.config.x_config import XCONF
    from xarm.wrapper import XArmAPI

    ROOT.joinpath(".state").mkdir(exist_ok=True)
    lock = (ROOT / ".state" / "xarm_relative.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if command_path.exists():
        if not command_path.is_socket():
            raise RuntimeError("控制接口路径不是 socket，拒绝覆盖")
        command_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(command_path))
    os.chmod(command_path, 0o600)
    server.listen(8)
    server.setblocking(False)
    context = zmq.Context()
    source = context.socket(zmq.SUB)
    source.setsockopt(zmq.SUBSCRIBE, b"")
    source.setsockopt(zmq.CONFLATE, 1)
    source.connect(f"tcp://127.0.0.1:{source_port}")
    arm = controller = gripper = publisher = None
    running = True
    started = time.monotonic()

    def shutdown(*_):
        nonlocal running
        running = False
        if controller is not None:
            controller.stopping = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    trace = (ROOT / ".state" / "relative_control_trace.jsonl").open("a", buffering=1)
    previous_error = ""
    previous_pause = None
    next_report = 0.0
    try:
        arm = XArmAPI(ip, is_radian=True)
        wait_initial_reports(arm)
        limits = np.asarray(XCONF.Robot.JOINT_LIMITS[arm.axis][arm.device_type])
        loaded = json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else None
        gripper = GripperWorker(lambda: XArmAPI(ip, is_radian=True))
        controller = RelativeArm(arm, limits[:, 0] + np.deg2rad(5), limits[:, 1] - np.deg2rad(5), control_settings=loaded, gripper=gripper,
            hardware_lower=limits[:,0], hardware_upper=limits[:,1], home_path=ROOT/"relative_home.json")
        # Cached rich-report properties; no new control or feedback query.
        controller.recording_robot_info = {
            "ip": ip, "firmware_version": list(arm.version_number), "serial": arm.sn,
            "tcp_offset_mm_rad": list(arm.tcp_offset), "world_offset_mm_rad": list(arm.world_offset),
            "tcp_load_kg_cog_mm": arm.tcp_load, "gravity_direction": list(arm.gravity_direction),
            "axis": arm.axis, "device_type": arm.device_type,
        }
        publisher = RecordingPublisher()
        def poll_source():
            try:
                controller.update_source(source.recv_json(zmq.NOBLOCK))
            except zmq.Again:
                pass
        controller.poll_source = poll_source
        print(json.dumps({"type": "robot_ready", "message": "xArm 已连接 · 默认只读预览"}, ensure_ascii=False), flush=True)
        while running and (duration is None or time.monotonic() - started < duration):
            tick_started = time.monotonic()
            try:
                controller.update_source(source.recv_json(zmq.NOBLOCK))
            except zmq.Again:
                pass
            for _ in range(8):
                try:
                    connection, _ = server.accept()
                except BlockingIOError:
                    break
                with connection:
                    connection.settimeout(0.1)
                    try:
                        raw = connection.recv(4096)
                        request = json.loads(raw)
                        controller.command(request["command"], request.get("settings"))
                        if request["command"] in ("configure", "matrix_select"):
                            temporary_settings = SETTINGS_PATH.with_suffix(".json.tmp")
                            temporary_settings.write_text(json.dumps(controller.settings, ensure_ascii=False, indent=2), encoding="utf-8")
                            temporary_settings.replace(SETTINGS_PATH)
                        reply = {"ok": True}
                        if request["command"] != "heartbeat":
                            controller.last_error = ""
                    except Exception as error:
                        controller.last_error = str(error)
                        reply = {"ok": False, "error": str(error)}
                    try:
                        connection.sendall(json.dumps(reply, ensure_ascii=False).encode())
                    except OSError:
                        pass
            try:
                controller.tick()
            except Exception as error:
                controller.last_error = str(error)
                controller.release(f"控制已停止：{error}", disarm=True)
            publisher.publish(recording_sample(controller))
            payload = controller.telemetry()
            pause_event = (payload['pause_feedback'].get('started_at'), payload['pause_feedback'].get('status'))
            if time.monotonic() >= next_report or payload["error"] != previous_error or pause_event != previous_pause:
                encoded = json.dumps(payload, ensure_ascii=False)
                print(encoded, flush=True)
                path = ROOT / ".state" / "relative_control_telemetry.json"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(encoded, encoding="utf-8")
                temporary.replace(path)
                if payload["active"] or payload['pause_settling'] or payload["error"] != previous_error or pause_event != previous_pause:
                    trace.write(encoded + "\n")
                previous_error = payload["error"]
                previous_pause = pause_event
                next_report = time.monotonic() + .05
            signal_elapsed = time.monotonic() - tick_started
            if signal_elapsed < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD - signal_elapsed)
    finally:
        if controller is not None:
            controller.release("控制器已退出", disarm=True)
        if gripper is not None:
            gripper.close()
        if publisher is not None:
            publisher.close()
        if arm is not None:
            arm.disconnect()
        trace.close()
        source.close(linger=0)
        context.term()
        server.close()
        command_path.unlink(missing_ok=True)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.244")
    parser.add_argument("--source-port", type=int, default=8130)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    serve(args.robot_ip, args.source_port, args.duration)
