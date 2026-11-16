#!/usr/bin/env python3
"""GTK six-dimensional Quest input viewer and explicit xArm teleoperation UI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import select
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk

from quest_app_control import QuestAppControl, PACKAGE
from quest_control_limits import TUNING_LIMITS, RESPONSIVE_SETTINGS, HOME_SPEED_DEFAULT, HOME_SPEED_LIMITS
from quest_lan import lan_host as validate_lan_host, ensure_tls, request_browser_reopen
from quest_process_adoption import AdoptedInput
from quest_recording_ui import RecordingPanel


ROOT = Path(__file__).resolve().parent
COLORS = ((0.94, 0.31, 0.32), (0.22, 0.75, 0.52), (0.26, 0.58, 0.95))


class PoseWindow(Gtk.Window):
    def __init__(self, demo=False, controller_control=False, lan_host=None):
        super().__init__(title="Quest 3 → xArm7 · 相对位移联调" + (" [演示数据]" if demo else ""))
        self.set_default_size(1180, 820)
        self.set_border_width(12)
        self.demo = demo
        self.lan_host = validate_lan_host(lan_host) if lan_host else None
        self.lan_ready = None
        self.process = None
        self.robot_process = None
        self.webxr_process = None
        self.webxr_url = None
        self.robot_mode = "preview"
        self.robot_demo_active = False
        self.recovery_pending = False
        self.recovery_local_error = None
        self.recovery_active = False
        self.last_recovery_item = None
        self.robot_last_snapshot = None
        self.auto_arm_pending = controller_control and not demo
        self.settings_revision = -1
        self.updating_settings = False
        self.messages = queue.Queue(maxsize=200)
        self.history = deque(maxlen=1200)
        self.samples = deque(maxlen=36000)
        self.reference = None
        self.quaternion_reference = None
        self.quaternion_frame = None
        self.position = None
        self.last_sample = None
        self.last_snapshot = None
        self.last_state_save = 0
        self.latest_live = False
        self.started = time.monotonic()
        self.closing = False
        self.quest_app_busy = False
        self.quest_status_pending = False
        self.quest_device_state = None
        self.last_source_state = None
        self.connect("destroy", self.close)

        self.gripper_can_command = False
        self.matrix_pending = False
        self.matrix_error = ''
        self.matrix_last_item = None
        self.settings_dirty = False
        from quest_dashboard_layout import build_dashboard
        build_dashboard(self)
        self.recording_panel = RecordingPanel(self, self.workspace_root)
        self.connect('key-press-event', self.key_pressed)
        GLib.timeout_add(250, self.robot_heartbeat)
        GLib.timeout_add(5000, self.poll_quest_status)
        GLib.timeout_add(50, self.refresh)

    def show_robot_pose(self, item):
        filtered = item.get('input_filter', {})
        if not filtered.get('enabled'):
            self.input_filter_feedback.set_text('输入滤波：关闭')
        elif filtered.get('valid') and item.get('source_valid'):
            self.input_filter_feedback.set_text(
                f"输入滤波：运行 · 平移 {filtered.get('position_cutoff_hz',0):.1f} Hz / "
                f"旋转 {filtered.get('rotation_cutoff_hz',0):.1f} Hz")
        else:
            self.input_filter_feedback.set_text('输入滤波：已启用，等待有效接管 / 预览')
        actual = [*item['tcp_mm'], *item.get('tcp_rpy_deg', [None]*3)]
        target = [*item['target_mm'], *item.get('target_rpy_deg', [None]*3)]
        command = ([*item.get('command_tcp_mm', [None]*3), *item.get('command_rpy_deg', [None]*3)]
                   if item.get('command_valid') is True else [None]*6)
        for fields, values in ((self.robot_pose_actual, actual), (self.robot_pose_target, target),
                               (self.robot_pose_command, command)):
            for field, value in zip(fields, values):
                field.set_text("—" if value is None or not math.isfinite(value) else f"{value:+.2f}")
        def number(value, suffix):
            return "—" if value is None else f"{value:.1f}{suffix}"
        def triple(values):
            return '—' if values is None else ' / '.join(f'{v:+.1f}' for v in values)
        delta = item.get('hand_aligned_delta_mm') if item.get('source_valid') else None
        angle = item.get('hand_aligned_rotation_deg') if item.get('source_valid') else None
        self.aligned_hand_feedback.set_text(f"对齐后手柄 ΔXYZ：{triple(delta)} mm · Δ旋转：{triple(angle)} °（基座轴，1×）")
        self.motion_feedback.set_text(f"实速 {item.get('actual_speed_mm_s', 0):.1f} mm/s · "
            f"{item.get('actual_angular_speed_deg_s',0):.1f}°/s · 输入 {number(item.get('source_age_ms'),' ms')} · "
            f"{item.get('control_hz', 0):.0f} Hz · "
            f"目标→实测 {item.get('following_error_mm', 0):.1f} mm / {item.get('rotation_error_deg', 0):.1f}° · "
            f"下发→实测 {number(item.get('command_tracking_error_mm') if item.get('command_valid') else None, ' mm')} / "
            f"{number(item.get('command_rotation_error_deg') if item.get('command_valid') else None, '°')}")
        guard = item.get('pose_guard') or {}
        guard_text = ''
        if guard:
            labels = {'normal':'正常', 'slowing':'提前减速', 'blocked':'禁止接管', 'paused':'保护暂停',
                      'recovery_only':'仅允许低速退出', 'direction_hold':'方向受限，等待反向手势'}
            prefix = '姿态保护' if item.get('active') or guard.get('status') in ('paused', 'blocked') else '上次姿态检查'
            guard_text = f"{prefix}：{labels.get(guard.get('status'), '检查中')}"
            if guard.get('sigma_min') is not None:
                guard_text += f" · 可解性 {guard['sigma_min']:.1f}（≤{guard.get('sigma_stop',8):g} 停止）"
                if guard.get('sigma_approach_stop') is not None:
                    guard_text += f" · ≤{guard['sigma_approach_stop']:g} 仅允许验证后的退出"
            if item.get('translation_priority'):
                guard_text += ' · 旋转暂锁，优先平移；松开扳机可重选'
            if guard.get('message'):
                guard_text += ' · '+guard['message']
        limit_text = ' · '.join(filter(None, [guard_text, item.get('limit_reason', '')]))
        self.limit_feedback.set_text(limit_text)
        self.limit_feedback.set_visible(bool(limit_text))
        ik = item.get('ik_feedback') or {}
        if ik:
            attempts = ik.get('attempts', [])
            fraction = "—" if ik.get('fraction') is None else f"{ik['fraction']:g}"
            age = item.get('ik_feedback_age_s')
            prefix = "逆解" if age is not None and age < .25 and item.get('active') else "上次逆解"
            status = {'ok': '连续解通过', 'blocked': '不可连续/不可达', 'budget_exhausted': '预算不足',
                      'deadline': '查询超时', 'sdk_error': 'SDK 异常', 'invalid_result': '返回数据异常'}.get(ik.get('status'), '求解中')
            text = f"{prefix}：{status} · {len(attempts)} 次 / {ik.get('elapsed_ms', 0):.1f} ms · 步长 ×{fraction}"
            if ik.get('last_failure') and ik.get('status') != 'ok':
                text += "\n最后原因：" + ik['last_failure']
            self.ik_feedback_label.set_text(text)
        self.ik_feedback_label.set_visible(bool(ik))

    def clear_robot_pose(self, message):
        self.input_filter_feedback.set_text('输入滤波：无实时状态')
        for field in (*self.robot_pose_actual, *self.robot_pose_target, *self.robot_pose_command):
            field.set_text("—")
        self.motion_feedback.set_text(message)
        self.aligned_hand_feedback.set_text("对齐后手柄 ΔXYZ / Δ旋转：—（机器人基座轴，1×）")
        self.limit_feedback.set_text("")
        self.limit_feedback.hide()
        self.ik_feedback_label.set_text("")
        self.ik_feedback_label.hide()

    def confirm_recovery(self, message):
        dialog = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.WARNING,
                                  buttons=Gtk.ButtonsType.OK_CANCEL, text="确认复位/关节恢复")
        dialog.format_secondary_text(message)
        def stop_key(_widget, event):
            if event.keyval in (Gdk.KEY_space, Gdk.KEY_Escape):
                self.robot_command("stop")
                dialog.response(Gtk.ResponseType.CANCEL)
                return True
            return False
        dialog.connect("key-press-event", stop_key)
        result = dialog.run()
        dialog.destroy()
        return result == Gtk.ResponseType.OK

    def home_speed_changed(self, *_):
        self.update_recovery_buttons()
        if not self.recovery_pending and not self.recovery_active:
            self.home_feedback.set_text('复位速度已调整，请重新检查路径后执行')
        panel = getattr(self, 'recording_panel', None)
        if panel is not None:
            panel.update_home_buttons()

    def recovery_action(self, command, direction=1):
        if self.demo or self.quest_app_busy or self.recovery_pending or self.recovery_active:
            self.home_feedback.set_text("演示/设备操作/复位处理中禁止重复发送恢复指令")
            return
        item = self.last_recovery_item
        if item is None or self.robot_process is None or self.robot_process.poll() is not None or self.robot_mode != "preview":
            self.home_feedback.set_text("请先连接机械臂并禁用遥操")
            return
        if self.robot_last_snapshot is None or time.monotonic()-self.robot_last_snapshot > .8:
            self.home_feedback.set_text("机器人状态过期，请重新连接后检查")
            return
        recovery = item.get("recovery", {})
        profile, feedback = recovery.get("home"), recovery.get("feedback", {})
        parameters = {"confirmed": True}
        if command in ('home_check', 'home_start'):
            self.home_speed_setting.update()
            self.update_recovery_buttons()
            parameters['speed_deg_s'] = self.home_speed_setting.get_value()
        if command == "home_save":
            text = "确认当前姿态是合适的采集初始点，不是需要恢复的异常姿态。\n记录 TCP 的 XYZ/RxRyRz 作为 Pose 目标，不移动；复位不要求原关节组合。"
            if profile is not None:
                text += "\n将覆盖已保存的初始点；如仅想复位，请点取消并使用“返回初始位姿”。"
        elif command == "home_start":
            if profile is None or not self.home_start_button.get_sensitive():
                self.home_feedback.set_text("请先检查复位路径；速度/关节变化或上次执行后需要重新检查")
                return
            parameters.update(home_saved_at=profile["saved_at"], checked_start=feedback["start"],
                              restore_gripper=self.restore_gripper.get_active())
            speed = parameters['speed_deg_s']
            text = ("机械臂将返回记录的 XYZ/RxRyRz Pose，不要求原关节组合。\n"
                    f"按检查过的关节路径运动，速度 {speed:g}°/s，加速度 {2*speed:g}°/s²，TCP 不一定走直线。\n")
            if 'max_joint_travel_deg' in feedback:
                text += f"本次最大单关节行程 {feedback['max_joint_travel_deg']:.1f}°；不是末端小幅直线移动。\n"
            text += "确认整个扫掠空间净空、负载/TCP 设置正确、急停可触达。\n若只有历史无解警告14，将在核查后清除；不清其他错误。"
            if parameters['restore_gripper']:
                text += f"\n会恢复初始夹爪开度 {profile.get('gripper_mm')} mm，注意夹手/物体掉落！"
            else:
                text += "\n夹爪保持当前抓持，不自动松开。"
        elif command == "joint_recover":
            parameters.update(joint=int(self.recovery_joint.get_active_id()),
                              step_deg=direction*self.recovery_step.get_value(), start_joints_deg=item['joints_deg'])
            text = f"J{parameters['joint']+1} 移动 {parameters['step_deg']:+g}°，速度 5°/s；不依赖笛卡尔 IK。\n"
            text += "确认该关节及整条机械臂扫掠空间净空。历史无解警告14可核查清除，其他故障不清除。\n夹爪保持；空格/Esc 或停止按钮中止。"
        else:
            text = None  # path check is read-only and needs no motion confirmation
        if text is not None and not self.confirm_recovery(text):
            return
        self.auto_arm_pending = False
        self.recovery_pending = True
        self.recovery_local_error = None
        self.home_feedback.set_text("正在核查恢复请求…不要重复点击；空格/Esc 可停止")
        self.update_recovery_buttons()
        self.robot_command(command, parameters)

    def show_recovery(self, item):
        self.last_recovery_item = item
        recovery = item.get("recovery", {})
        self.recovery_active = recovery.get("active", False)
        feedback = recovery.get("feedback", {})
        if self.recovery_local_error:
            self.home_feedback.set_text(self.recovery_local_error)
        elif not self.recovery_pending:
            if (feedback.get('status') == 'checked' and
                    feedback.get('speed_deg_s') != self.home_speed_setting.get_value()):
                self.home_feedback.set_text('复位速度已调整，请重新检查路径后执行')
            else:
                self.home_feedback.set_text(feedback.get("message", "未设置初始位姿；异常姿态可使用关节恢复"))
        profile = recovery.get("home")
        current = "当前关节 °：" + " / ".join(f"{q:+.1f}" for q in item.get('joints_deg', []))
        details = current + "\n"
        axis = int(self.recovery_joint.get_active_id())
        lower, upper = recovery.get('hardware_lower_deg',[]), recovery.get('hardware_upper_deg',[])
        if len(lower)==7 and len(upper)==7:
            details += f"所选 J{axis+1} 硬件范围 [{lower[axis]:.1f}, {upper[axis]:.1f}]°；目标不可越限。\n"
        if profile is None:
            details += "初始点尚未记录：先恢复到合适姿态，再设当前为初始点。"
        else:
            details += "初始 Pose · TCP mm：" + " / ".join(f"{v:.1f}" for v in profile['tcp'][:3])
            details += "\n初始 Rx/Ry/Rz °：" + " / ".join(f"{math.degrees(v):.1f}" for v in profile['tcp'][3:])
            details += f" · 记录夹爪 {profile.get('gripper_mm')} mm"
            details += "\n以 TCP Pose 为目标，不以记录的七关节角为目标；关节仅用于安全路径检查。"
        if 'tcp_path_min_mm' in feedback:
            low = ' / '.join(f'{v:.1f}' for v in feedback['tcp_path_min_mm'])
            high = ' / '.join(f'{v:.1f}' for v in feedback['tcp_path_max_mm'])
            details += f'\n检查路径 XYZ 包络 mm：[{low}] → [{high}]（不是碰撞检测）'
        if 'max_joint_travel_deg' in feedback:
            details += f"\n最大单关节行程 {feedback['max_joint_travel_deg']:.1f}° · 确认整臂扫掠空间，不保证 TCP 直线返回。"
        self.home_details.set_text(details)

    def update_recovery_buttons(self):
        item = self.last_recovery_item or {}
        recovery = item.get('recovery', {})
        feedback, profile = recovery.get('feedback', {}), recovery.get('home')
        fresh = self.robot_last_snapshot is not None and time.monotonic()-self.robot_last_snapshot < .8
        available = bool(not self.demo and fresh and self.robot_process is not None and not self.quest_app_busy
                         and not self.recovery_pending and not self.recovery_active and not self.robot_demo_active
                         and self.robot_mode == 'preview')
        self.home_save_button.set_sensitive(available)
        self.home_check_button.set_sensitive(available and profile is not None)
        self.home_speed_setting.set_sensitive(available)
        self.joint_minus.set_sensitive(available)
        self.joint_plus.set_sensitive(available)
        start = feedback.get('start', [])
        same_start = len(start)==7 and len(item.get('joints_deg',[]))==7 and max(abs(math.radians(a)-b) for a,b in zip(item['joints_deg'],start)) <= math.radians(.2)
        self.home_start_button.set_sensitive(bool(available and profile is not None and same_start and
            feedback.get('status') == 'checked' and feedback.get('home_saved_at') == profile['saved_at'] and
            feedback.get('speed_deg_s') == self.home_speed_setting.get_value()))

    def connect_quest(self, *_):
        if self.process is not None:
            return
        self.reference = self.position = self.last_sample = None
        self.quaternion_reference = self.quaternion_frame = None
        self.history.clear()
        self.samples.clear()
        self.last_snapshot = None
        self.connect_button.set_sensitive(False)
        self.status.set_text("正在检查 USB 并打开接收端口…")
        if self.demo:
            self.status.set_text("演示模式 · 当前显示模拟数据，不能用于确认硬件正常")
            return
        config = json.loads((ROOT / "config.json").read_text())
        self.process = subprocess.Popen([config["openteach_python"], "-u", str(ROOT / "quest_pose_client.py")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=self.read_stream, args=(self.process,), daemon=True).start()
        if not self.lan_host:
            threading.Thread(target=self.setup_usb, daemon=True).start()
        else:
            self.status.set_text('局域网接收已就绪 · 点击“打开头显输入”，无需 USB 授权')

    def enqueue(self, value):
        try:
            self.messages.put_nowait(value)
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            self.messages.put_nowait(value)

    def read_stream(self, process):
        for line in process.stdout:
            try:
                item = json.loads(line)
                item["process"] = process
                self.enqueue(item)
            except json.JSONDecodeError:
                self.enqueue({"type": "error", "message": line.strip()})
        error = process.stderr.read().strip()
        code = process.wait()
        self.enqueue({"type": "exit", "process": process, "code": code, "error": error})

    def quest_control(self):
        config = json.loads((ROOT/'config.json').read_text())
        package='com.oculus.browser' if config.get('quest_input_backend') in ('webxr','unitree') else config.get('quest_package',PACKAGE)
        return QuestAppControl(cancelled=lambda: self.closing, package=package)

    def ensure_webxr(self,restart=False):
        process=self.webxr_process
        if process is not None and process.poll() is None:
            if not restart and self.webxr_url:
                return self.webxr_url
            process.terminate();process.wait(timeout=3)
        if self.closing:
            raise RuntimeError('窗口已关闭，取消头显输入')
        config=json.loads((ROOT/'config.json').read_text())
        unitree=config.get('quest_input_backend')=='unitree'
        python=str(ROOT/'.venv-unitree310/bin/python') if unitree else str(Path(config['lerobot_root'])/'.venv/bin/python')
        script='quest_unitree_bridge.py' if unitree else 'quest_webxr_bridge.py'
        command=[python,'-u',str(ROOT/script)]
        fingerprint=None
        if self.lan_host:
            if not unitree:
                raise ValueError('局域网 HTTPS 目前仅支持宇树输入')
            cert,key,fingerprint=ensure_tls(self.lan_host,ROOT/'.state'/'quest_tls')
            command += ['--host',self.lan_host,'--cert',str(cert),'--key',str(key)]
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.webxr_process=process
        if not select.select([process.stdout],[],[],12)[0]:
            process.terminate();raise RuntimeError('WebXR 本地服务启动超时')
        line=process.stdout.readline()
        if not line:
            raise RuntimeError('WebXR 服务启动失败：'+process.stderr.read()[-500:])
        item=json.loads(line)
        if item.get('type')!=('unitree_ready' if unitree else 'webxr_ready'):
            process.terminate();raise RuntimeError('WebXR 服务未确认就绪')
        self.webxr_url=item['url']
        if self.lan_host:
            self.lan_ready=dict(url=self.webxr_url,pairing_code=item['pairing_code'],fingerprint=fingerprint,
                               launch_url=item.get('launch_url'))
            self.enqueue(dict(type='lan_ready',**self.lan_ready))
        session_path=ROOT/'.state'/'webxr_session.json'
        # Restrict before writing pairing credentials, including first creation.
        session_path.touch(mode=0o600,exist_ok=True)
        session_path.chmod(0o600)
        session_path.write_text(json.dumps({'pid':process.pid,'url':self.webxr_url,'started_at':time.time(),
            'transport':'lan' if self.lan_host else 'usb',**(self.lan_ready or {})}),encoding='utf-8')
        # Drain upstream logs; otherwise a full pipe can freeze input delivery.
        def drain(stream):
            for _ in stream:
                pass
        threading.Thread(target=drain,args=(process.stdout,),daemon=True).start()
        threading.Thread(target=drain,args=(process.stderr,),daemon=True).start()
        return self.webxr_url

    def adopt_lan_input(self):
        if not self.lan_host or self.demo:
            raise ValueError('只能接管当前 LAN 模式的真实采集服务')
        session = json.loads((ROOT / '.state/webxr_session.json').read_text())
        if session.get('transport') != 'lan' or session.get('url') != f'https://{self.lan_host}:8766/':
            raise ValueError('已有采集服务地址与 GUI 配置不一致')
        process = AdoptedInput(session['pid'], ROOT, self.lan_host)
        self.webxr_process = process
        self.webxr_url = session['url']
        self.lan_ready = {key: session[key] for key in ('url', 'pairing_code', 'fingerprint')}
        self.lan_ready['launch_url'] = session.get('launch_url')
        self.enqueue(dict(type='lan_ready', **self.lan_ready))
        def drain(stream):
            for _ in stream:
                pass
        for stream in (process.stdout, process.stderr):
            threading.Thread(target=drain, args=(stream,), daemon=True).start()

    def setup_usb(self):
        try:
            state = self.quest_control().prepare()
            self.enqueue({"type": "usb", "message": f"USB 已连接：{state['serial']} · 8100 / 8125 已映射"})
            self.enqueue({"type": "quest_device", "state": state})
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.enqueue({"type": "error", "message": f"USB 设置失败：{error}"})

    def stop_for_quest_app(self):
        path = ROOT / ".state" / "xarm_relative.sock"
        robot_alive = self.robot_process is not None and self.robot_process.poll() is None
        if not robot_alive and not path.exists():
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(2)
                connection.connect(str(path))
                connection.sendall(json.dumps({"command": "stop"}).encode())
                reply = json.loads(connection.recv(4096))
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "停止请求被拒绝"))
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(f"无法确认机器人停止，已取消 Quest 操作：{error}") from error

    def quest_app_action(self, action):
        if self.demo or self.closing or self.quest_app_busy:
            return
        self.quest_app_busy = True
        self.auto_arm_pending = False
        self.quest_app_note.set_text("正在确认机器人释放控制，再操作 Quest；不会自动恢复遥操…")
        def work():
            try:
                config=json.loads((ROOT/'config.json').read_text())
                backend=config.get('quest_input_backend')
                webxr=backend in ('webxr','unitree') and action!='wake'
                if webxr:
                    self.stop_for_quest_app()
                    url=self.ensure_webxr(restart=action=='restart')
                    if self.lan_host:
                        launch = (self.lan_ready or {}).get('launch_url')
                        message = '局域网 HTTPS 已启动；请在 Quest 浏览器打开 GUI 地址与配对码，再点 Pass-through。机械臂仍只读。'
                        if launch:
                            try:
                                cert=ROOT/'.state'/'quest_tls'/self.lan_host.replace('.','_')/'cert.pem'
                                result=request_browser_reopen(launch,cert)
                                message='已发送 Wi-Fi 页面恢复请求；头显网页重新加载后点 Pass-through，等实时位姿恢复，再启用遥操并按住扳机。'
                                if not result.get('clock_clients'):
                                    try:
                                        self.quest_control().perform_lan_webxr(launch,self.lan_host,self.stop_for_quest_app)
                                        message='已在头显打开并配对 Wi-Fi 采集网页；点 Pass-through，等实时位姿恢复，再启用遥操并按住扳机。'
                                    except (OSError, RuntimeError, subprocess.SubprocessError):
                                        message='恢复请求已排队；请唤醒头显并打开之前配对的浏览器页，再点 Pass-through。若浏览器已关闭，手动打开 GUI 地址并配对（USB 自动打开不可用）。'
                            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                                message += ' 页面恢复未完成：'+str(error)
                        self.enqueue({'type':'quest_action_result','action':action,
                            'message':message})
                        return
                    state=self.quest_control().perform_webxr(url,self.stop_for_quest_app,backend=backend)
                else:
                    state = self.quest_control().perform(action, self.stop_for_quest_app)
                verb = {"start": "打开", "restart": "重启", "wake": "唤醒"}[action]
                note = (f"{verb}头显输入页面完成，机器人保持只读。戴上 Quest 点击“开始手柄采集”；核对 XYZ/扳机/B 和方向，再允许遥操。" if webxr else
                        f"{verb}命令完成，机器人保持释放。请戴上头显，进入 Menu → Stream，确认有效 XYZ/按键后再允许遥操。")
                if webxr and backend=='unitree':
                    note=f'{verb}宇树采集页面完成，机器人只读。戴上 Quest，点击底部“pass-through / 透视”；核对位姿、扳机和 B 后再启用遥操。'
                if action == "restart":
                    note += "重启可能重置手柄坐标基准，请重新核对方向，未完成校准需重新采样。"
                self.enqueue({"type": "quest_action_result", "action": action, "state": state, "message": note})
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                self.enqueue({"type": "quest_action_result", "error": str(error)})
        threading.Thread(target=work, daemon=True).start()

    def poll_quest_status(self):
        if self.closing:
            return False
        if self.lan_host:
            self.show_quest_device()
            return True
        if self.demo or self.quest_app_busy or self.quest_status_pending:
            return True
        self.quest_status_pending = True
        def work():
            try:
                state = self.quest_control().inspect()
                self.enqueue({"type": "quest_device", "state": state})
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                self.enqueue({"type": "quest_device", "error": str(error)})
        threading.Thread(target=work, daemon=True).start()
        return True

    def show_quest_device(self):
        if self.lan_host:
            source = self.last_source_state or {}
            elapsed = time.monotonic()-self.last_snapshot if self.last_snapshot is not None else float('inf')
            live = source.get('valid') is True and source.get('valid_age_s') is not None and source['valid_age_s']+elapsed < .25
            self.device_summary.set_text('头显：局域网 HTTPS · 无需 USB 调试授权 · '+('正在接收数据' if live else '等待浏览器连接/进入透视'))
            self.instructions.set_text('Quest 浏览器手动打开电脑 GUI 中的 HTTPS 地址 → 输入配对码 → Pass-through。\n先只读核对 XYZ/姿态和扳机/B；现场安全确认后才启用遥操。')
            return
        state = self.quest_device_state
        if state is None:
            return
        self.device_summary.set_text("头显：" + ("已连接" if state['installed'] else "App 未安装")
            + (" · 休眠，请佩戴唤醒" if state['wakefulness'] in ("Asleep", "Dozing", "Dreaming") else "")
            + (" · App 未运行" if state['installed'] and not state['running'] else ""))
        app_name = 'WebXR 浏览器' if state.get('package')=='com.oculus.browser' else 'FrankaBot' if state.get('package','com.Xigbee.FrankaBot')=='com.Xigbee.FrankaBot' else 'Quest 控制器'
        if state.get('package')=='com.oculus.browser':
            self.instructions.set_text('操作：点“打开头显输入”→在 Quest 浏览器点“开始手柄采集”。\n先只读核对 XYZ/RxRyRz、扳机和 B；确认方向及现场安全后启用遥操。扳机按住跟随/松开暂停，B 单击切换开合、松开保持。')
            if json.loads((ROOT/'config.json').read_text()).get('quest_input_backend')=='unitree':
                app_name='宇树 TeleVuer 浏览器'
                self.device_summary.set_text(self.device_summary.get_text()+' · 宇树输入（Quest 中点 Pass-through）')
                self.instructions.set_text('宇树输入：打开头显输入 → 戴上 Quest，在页面底部点击 pass-through（透视）。\n先只读核对 XYZ/姿态和方向；扳机按住跟随/松开暂停，B 单击切换开合、松开保持。xArm 控制保持原设置。')
        webxr = state.get('package') == 'com.oculus.browser'
        app = f"未安装 {app_name}" if not state["installed"] else f"{app_name} 进程运行（PID {state['pid']}）" if state["running"] else f"{app_name} 未运行：点“打开头显输入”"
        wake = state["wakefulness"]
        power = "头显已唤醒" if wake == "Awake" else "头显休眠：请佩戴/唤醒" if wake in ("Asleep", "Dozing", "Dreaming") else "头显唤醒状态未知"
        data = ("尚无有效 XYZ：请在头显页面进入采集/透视模式并保持手柄追踪" if webxr else
                "尚无有效 XYZ：请进入 Stream 并保持手柄追踪")
        s = self.last_source_state or {}
        elapsed = 0 if self.last_snapshot is None else time.monotonic() - self.last_snapshot
        if s.get("valid") and s.get("valid_age_s") is not None and s["valid_age_s"] + elapsed < .25:
            data = f"有效手柄数据 {s.get('valid_hz', 0):g} Hz"
        elif state["running"] and wake == "Awake" and (s.get("pause_age_s") is None or s["pause_age_s"] + elapsed > 1):
            data = ("浏览器运行但无实时数据：采集未开始、后台暂停或卡住，请检查头显页面" if webxr else
                    "进程存在但无实时数据：未进入 Stream、后台暂停或卡住，可尝试重启")
        self.quest_app_status.set_text(f"USB：{state['serial']} · {app} · {power}\n{data}（进程运行不等于画面/追踪正常）")

    def stop(self):
        if self.robot_process is not None:
            self.robot_command("stop")
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self.connect_button.set_sensitive(True)
        self.latest_live = False
        self.button_status.set_text("扳机 / A：未知（已停止接收）")
        self.status.set_text("已停止接收 · 曲线保留，可导出")

    def close(self, *_):
        self.closing = True
        self.recording_panel.close()
        if self.webxr_process is not None and self.webxr_process.poll() is None:
            self.webxr_process.terminate()
        process, self.robot_process = self.robot_process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Its SIGTERM handler and heartbeat watchdog still request a stop.
                # Do not SIGKILL the process before its hardware cleanup returns.
                pass
        self.stop()
        Gtk.main_quit()

    def connect_robot(self, *_):
        if self.demo:
            self.robot_error.set_text("演示模式不连接实体机械臂")
            return
        if self.robot_process is not None:
            return
        config = json.loads((ROOT / "config.json").read_text())
        self.robot_status.set_text("连接 xArm 中 · 只读目标预览")
        self.robot_process = subprocess.Popen([str(Path(config["lerobot_root"]) / ".venv/bin/python"), "-u", str(ROOT / "xarm_quest_relative.py"), "--robot-ip", config["robot_ip"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=self.read_robot_stream, args=(self.robot_process,), daemon=True).start()

    def read_robot_stream(self, process):
        for line in process.stdout:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # xArm SDK startup information is not telemetry.
            item["process"] = process
            self.enqueue(item)
        error = process.stderr.read().strip()
        code = process.wait()
        self.enqueue({"type": "robot_exit", "process": process, "code": code, "error": error})

    def robot_command(self, command, settings=None):
        if self.quest_app_busy and command not in ("heartbeat", "stop", "release", "preview"):
            self.robot_error.set_text("Quest 操作期间禁止接管/示教，请等待完成")
            return
        if self.robot_process is None or self.robot_process.poll() is not None:
            if command != "heartbeat":
                self.robot_error.set_text("请先连接 xArm")
            return
        if command == 'matrix_select':
            if self.matrix_pending:
                return
            self.matrix_pending = True
            self.matrix_error = ''
            self.matrix_choice_note.set_text('正在应用矩阵…')

        def send():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(6.0 if command in ("paired_demo", "home_start", "joint_recover") else 2.0 if command != "heartbeat" else 0.4)
                    connection.connect(str(ROOT / ".state" / "xarm_relative.sock"))
                    request = {"command": command}
                    if settings is not None:
                        request["settings"] = settings
                    connection.sendall(json.dumps(request).encode())
                    reply = json.loads(connection.recv(4096))
                if not reply.get("ok") and command != "heartbeat":
                    self.enqueue({"type": "robot_command_error", "message": reply.get("error", "控制命令失败")})
                if command.startswith('home_') or command == 'joint_recover':
                    self.enqueue({"type": "recovery_reply", "ok": reply.get('ok', False), 'error': reply.get('error','')})
                if command == 'matrix_select':
                    self.enqueue(dict(type='matrix_reply', error='' if reply.get('ok') else reply.get('error', '矩阵应用失败')))
            except (OSError, ValueError) as error:
                if command != "heartbeat":
                    self.enqueue({"type": "robot_command_error", "message": str(error)})
                if command.startswith('home_') or command == 'joint_recover':
                    self.enqueue({"type": "recovery_reply", "ok": False, "error": f"恢复命令通信失败：{error}；不要重复点击，请先停止并检查状态"})
                if command == 'matrix_select':
                    self.enqueue(dict(type='matrix_reply', error=f'矩阵通信失败：{error}'))
        threading.Thread(target=send, daemon=True).start()

    def matrix_choice_changed(self, *_):
        if not self.updating_settings:
            self.matrix_choice_dirty = True
        if self.matrix_last_item is not None:
            self.show_matrix_choice(self.matrix_last_item)

    def confirm_matrix_choice(self, *_):
        self.robot_command('matrix_select', {'choice': self.matrix_choice.get_active_id()})

    def show_matrix_choice(self, item):
        self.matrix_last_item = item
        values = item.get('settings', {})
        active = values.get('empirical_matrix') is not None
        self.mapping_mode_note.set_text('当前：' + ('新 6×6 矩阵' if active else '旧 3×3 方向映射'))
        ready = (item.get('mode') == 'preview' and not item.get('active', False)
                 and not item.get('demo_active', False) and not item.get('recovery', {}).get('active', False)
                 and not item.get('pause_settling', False) and not self.matrix_pending)
        self.robot_demo_active = item.get('demo_active', False)
        chosen_new = self.matrix_choice.get_active_id() == 'new'
        available = values.get('saved_empirical_matrix') is not None or active
        self.matrix_choice.set_sensitive(ready)
        self.matrix_select_button.set_sensitive(bool(ready and (not chosen_new or available)))
        note = '应用后保持只读；手动启用遥操后接管。'
        if chosen_new:
            note = ('将已保存的 6×6 矩阵用于当前手柄会话。' + note) if available else '尚无已保存的 6×6 矩阵。'
        if item.get('mode') != 'preview':
            note = '请先点击“禁用遥操”，再选择和应用矩阵；松开扳机暂停仍保留遥操许可。'
        elif chosen_new and available and not item.get('source_valid', False):
            note = '可以应用已保存的 6×6 矩阵；追踪恢复且参考坐标匹配后才能接管。'
        self.matrix_choice_note.set_text('正在应用矩阵…' if self.matrix_pending else self.matrix_error or note)
        matrix = values.get('empirical_matrix') if active else values.get('direction_matrix')
        self.matrix_values.set_text('\n'.join('  '.join(f'{x:+.6f}' for x in row) for row in matrix) if matrix is not None else '等待读取矩阵')

    def apply_responsive_settings(self, *_):
        if self.recovery_active or self.recovery_pending or self.robot_demo_active or self.quest_app_busy:
            return
        self.speed_setting.set_value(RESPONSIVE_SETTINGS['speed_mm_s'])
        self.joint_speed_setting.set_value(math.degrees(RESPONSIVE_SETTINGS['joint_speed_rad_s']))
        self.acceleration_setting.set_value(RESPONSIVE_SETTINGS['acceleration_mm_s2'])
        self.tracking_lead_setting.set_value(RESPONSIVE_SETTINGS['tracking_lead_mm'])
        self.rotation_speed_setting.set_value(RESPONSIVE_SETTINGS['rotation_speed_deg_s'])
        self.rotation_acceleration_setting.set_value(RESPONSIVE_SETTINGS['rotation_acceleration_deg_s2'])
        self.apply_settings()

    def settings_edited(self, *_):
        if self.updating_settings:
            return
        self.settings_dirty=True
        self.settings_edit_note.set_text('有未应用修改 · 点击“应用并暂停”后生效。')
        self.settings_edit_note.get_style_context().add_class('pending-text')

    def apply_settings(self, *_):
        try:
            values = {"speed_mm_s": self.speed_setting.get_value(), "translation_scale": self.scale_setting.get_value(),
                      "input_filter": {"enabled": self.input_filter_setting.get_active()},
                      "teleop_button": self.teleop_button_setting.get_active_id(), "gripper_button": self.gripper_button_setting.get_active_id(),
                      "joint_speed_rad_s": math.radians(self.joint_speed_setting.get_value()),
                      "acceleration_mm_s2": self.acceleration_setting.get_value(),
                      "range_mm": None,
                      "tracking_lead_mm": self.tracking_lead_setting.get_value(),
                      "rotation_enabled": self.rotation_enabled_setting.get_active(),
                      "rotation_axes": [axis.get_active() for axis in self.rotation_axis_settings],
                      "rotation_scale": self.rotation_scale_setting.get_value(),
                      "rotation_speed_deg_s": self.rotation_speed_setting.get_value(),
                      "rotation_acceleration_deg_s2": self.rotation_acceleration_setting.get_value(),
                      "gripper_enabled": self.gripper_enabled_setting.get_active(),
                      "gripper_open_mm": self.gripper_open_setting.get_value(), "gripper_close_mm": self.gripper_close_setting.get_value(),
                      "gripper_speed_mm_s": self.gripper_speed_setting.get_value(), "gripper_force": self.gripper_force_setting.get_value(),
                      "direction_matrix": (self.matrix_last_item or {}).get("settings", {}).get("direction_matrix", [[0,0,1],[-1,0,0],[0,1,0]])}
            self.robot_command("configure", values)
        except ValueError:
            self.robot_error.set_text("矩阵中存在非数值项")

    def show_settings(self, item):
        values = item.get("settings")
        if values is None or item.get("settings_revision") == self.settings_revision:
            return
        self.settings_revision = item["settings_revision"]
        self.updating_settings = True
        self.input_filter_setting.set_active(values.get('input_filter', {}).get('enabled', False))
        if not self.matrix_choice_dirty:
            self.matrix_choice.set_active_id('new' if values.get('empirical_matrix') is not None else 'old')
        self.gain_limit_note.set_text('平移 ≤10× / 旋转 ≤5×')
        for key, field in self.tuning_fields.items():
            low, high = TUNING_LIMITS[key]
            if key == 'joint_speed_rad_s':
                low, high = math.degrees(low), math.degrees(high)
            field.set_range(low, high)
        teleop_button = values.get("teleop_button", "trigger")
        self.teleop_button_setting.set_active_id(teleop_button)
        self.robot_enable.set_label("启用遥操" if teleop_button == "trigger" else "启用 B 遥操")
        self.robot_takeover.set_label("重设起点" if teleop_button == "trigger" else "接管 / 重设起点")
        self.speed_setting.set_value(values["speed_mm_s"])
        self.scale_setting.set_value(values["translation_scale"])
        self.joint_speed_setting.set_value(math.degrees(values["joint_speed_rad_s"]))
        self.acceleration_setting.set_value(values.get("acceleration_mm_s2", 300))
        self.tracking_lead_setting.set_value(values.get('tracking_lead_mm',5))
        self.rotation_enabled_setting.set_active(values.get("rotation_enabled", True))
        for field, enabled in zip(self.rotation_axis_settings, values.get("rotation_axes", [True]*3)):
            field.set_active(enabled)
        self.rotation_scale_setting.set_value(values.get("rotation_scale", 1))
        self.rotation_speed_setting.set_value(values.get("rotation_speed_deg_s", 30))
        self.rotation_acceleration_setting.set_value(values.get("rotation_acceleration_deg_s2", 90))
        self.gripper_enabled_setting.set_active(values.get("gripper_enabled", False))
        self.gripper_button_setting.set_active_id(values.get('gripper_button','a'))
        for field, key, default in ((self.gripper_open_setting, "gripper_open_mm", 84), (self.gripper_close_setting, "gripper_close_mm", 0),
                                    (self.gripper_speed_setting, "gripper_speed_mm_s", 50), (self.gripper_force_setting, "gripper_force", 30)):
            field.set_value(values.get(key, default))
        self.updating_settings = False
        self.settings_dirty = False
        self.settings_edit_note.set_text('已同步后台参数；修改后需点击“应用并暂停”。')
        self.settings_edit_note.get_style_context().remove_class('pending-text')
        self.effective_settings_note.set_text(
            f"后台已生效：平移 {values['speed_mm_s']:g} mm/s ×{values['translation_scale']:g}；"
            f"旋转 {values.get('rotation_speed_deg_s',30):g}°/s ×{values.get('rotation_scale',1):g}；"
            f"关节 {math.degrees(values['joint_speed_rad_s']):.1f}°/s\n"
            f"加速度 {values.get('acceleration_mm_s2',300):g} mm/s² / {values.get('rotation_acceleration_deg_s2',90):g}°/s²；"
            f"跟随容差上限 {values.get('tracking_lead_mm',5):g} mm；夹爪力 {values.get('gripper_force',30):g}%"
            + ('；输入抑抖开启' if values.get('input_filter', {}).get('enabled') else '；输入抑抖关闭')
            + '。'
            + '\n输入框是待应用值；修改后点“应用并暂停”，不会自动启用遥操。')
        rows = ["[" + ", ".join(f"{v:+.3f}" for v in row) + "]" for row in values["direction_matrix"]]
        self.mapping.set_text("生效矩阵 M = " + "  ".join(rows) + f"    倍率 {values['translation_scale']:g}×")
        if values.get('empirical_matrix') is not None:
            self.mapping.set_text(f"已启用 6×6 经验补偿（含标定倍率/串扰） · 用户平移 {values['translation_scale']:g}× / 旋转 {values['rotation_scale']:g}× 另行叠加；旧 3×3 未使用")

    def robot_heartbeat(self):
        self.robot_command("heartbeat")
        return True

    def key_pressed(self, _widget, event):
        if event.keyval == Gdk.KEY_space:
            panel = getattr(self, 'recording_panel', None)
            if panel is not None and self.get_focus() in (panel.task, panel.folder, panel.dataset):
                return False  # allow task text/path spaces; Esc remains global
            focus = self.get_focus()
            if (panel is not None and panel.notebook.get_current_page() != 0 and
                    focus is not None and focus.is_ancestor(panel.notebook)):
                return False  # recording/library keyboard actions are not robot commands
            self.robot_command("stop")
            return True
        if event.keyval == Gdk.KEY_Escape:
            self.robot_command("release")
            return True
        return False

    def zero(self, *_):
        if self.position is not None and self.latest_live:
            self.reference = list(self.position)
            self.quaternion_reference = None
            self.detail.set_text("已设置位移/旋转诊断零点；绝对位姿不变，不改变遥操标定")
        else:
            self.detail.set_text("还没有实时有效位姿，无法设置零点")

    def refresh(self):
        while True:
            try:
                item = self.messages.get_nowait()
            except queue.Empty:
                break
            if "process" in item and item["process"] is not self.process and item["process"] is not self.robot_process:
                continue
            kind = item.get("type")
            if kind == "snapshot":
                self.last_snapshot = time.monotonic()
                self.show_snapshot(item)
            elif kind == 'lan_ready':
                self.lan_info.set_text(f"头显浏览器：{item['url']}    配对码：{item['pairing_code']}\n无需 USB · 首次证书提示请核对地址；完整指纹见“高级参数 / 设备维护”")
                self.lan_info.set_tooltip_text('证书 SHA-256：'+item['fingerprint'])
                self.quest_app_status.set_text('本机 HTTPS 证书 SHA-256：'+item['fingerprint'])
                self.lan_info.show()
            elif kind == "error":
                self.detail.set_text(item["message"])
                self.device_summary.set_text(item["message"])
            elif kind == "usb":
                self.detail.set_text(item["message"])
            elif kind in ("quest_device", "quest_action_result"):
                if kind == "quest_device":
                    self.quest_status_pending = False
                else:
                    self.quest_app_busy = False
                    self.quest_app_note.set_text(item.get("message", "Quest 操作失败：" + item.get("error", "")))
                    if "state" in item and item.get("action") in ("start", "restart"):
                        self.reference = None
                        self.quaternion_reference = None
                if "state" in item:
                    self.quest_device_state = item["state"]
                    self.show_quest_device()
                elif kind == "quest_device":
                    self.quest_device_state = None
                    self.quest_app_status.set_text(item["error"])
                    self.device_summary.set_text("头显检查失败 · " + item['error'])
            elif kind == "robot":
                event = item.get('last_stop_event') or {}
                self.last_stop_note.set_text('历史停止记录（非当前状态）：' + event['reason'] if event.get('reason') else '')
                self.show_settings(item)
                self.show_matrix_choice(item)
                self.show_robot_pose(item)
                self.show_recovery(item)
                self.robot_mode = item["mode"]
                self.robot_last_snapshot = time.monotonic()
                color = "#38bf85" if item["active"] else "#fbbf24" if item.get('teleop_paused') else "#60a5fa"
                self.robot_status.set_markup(f'<span foreground="{color}">' + GLib.markup_escape_text(item["status"]) + '</span>')
                xyz = lambda values: "[" + ", ".join(f"{v:.2f}" for v in values) + "]"
                self.robot_tcp.set_text(f"实际 TCP XYZ：{xyz(item['tcp_mm'])} mm    目标：{xyz(item['target_mm'])} mm")
                self.robot_rotation.set_text(f"实际 Rx/Ry/Rz：{xyz(item.get('tcp_rpy_deg', [0]*3))} °    目标：{xyz(item.get('target_rpy_deg', [0]*3))} °    角误差：{item.get('rotation_error_deg', 0):.2f}°")
                grip = item.get("gripper") or {}
                position = "—" if grip.get("position_mm") is None or grip.get("position_age_s") is None or grip["position_age_s"] > 2 else f"{grip['position_mm']:.1f}"
                target = "—" if grip.get("target_mm") is None else f"{grip['target_mm']:.1f}"
                self.gripper_status.set_text(f"G2 开度：{position} mm    目标：{target} mm    已发指令：{grip.get('commands', 0)}\n{grip.get('status', '未连接')} {grip.get('error', '')}")
                key=item.get('gripper_button',item.get('settings',{}).get('gripper_button','b')).upper()
                self.gripper_summary.set_text(f"开度 {position} mm" + (" · " + grip['error'] if grip.get('error') else (f" · {key} 单击切换开合 · 松开保持" if key=='B' else f" · {key} 按住闭合 / 松开打开")))
                self.gripper_can_command = bool(item.get("settings", {}).get("gripper_enabled") and grip.get("connected"))
                self.robot_stats.set_text(f"会话起点：{xyz(item['origin_mm'])} mm    实测位移：{xyz(item['max_displacement_mm'])} mm    已发指令：{item['sent_commands']}")
                self.robot_error.set_text(item.get("error", ""))
                command_error = item.get('command_tracking_error_mm') if item.get('command_valid') else None
                command_error_text = '—' if command_error is None else f'{command_error:.2f} mm'
                self.speed_feedback.set_text(f"实测速率：{item.get('actual_speed_mm_s', 0):.1f} mm/s    映射目标差：{item.get('following_error_mm', 0):.2f} mm    下发跟随差：{command_error_text}\n控制：{item.get('control_hz', 0):.1f}/{item.get('control_target_hz', 100)} Hz · {item.get('limit_reason', '')}")
                self.range_status.set_text("移动范围：无起点小围栏；请确认现场净空" if item.get("range_mm") is None else f"移动范围：配置围栏 ±{item['range_mm']:g} mm")
            elif kind == "robot_command_error":
                self.robot_error.set_text(item["message"])
            elif kind == 'matrix_reply':
                self.matrix_pending = False
                self.matrix_error = item['error']
                if self.matrix_last_item is not None:
                    self.show_matrix_choice(self.matrix_last_item)
            elif kind == "recovery_reply":
                self.recovery_pending = False
                self.recovery_local_error = item['error'] if not item['ok'] else None
                if self.recovery_local_error:
                    self.home_feedback.set_text(self.recovery_local_error)
            elif kind == "robot_exit":
                self.robot_process = None
                self.robot_status.set_text(f"xArm 控制器已退出：{item['code']}")
                self.robot_error.set_text(item["error"][-400:])
                self.clear_robot_pose("机器人连接已断开 · 无实时位姿")
            elif kind == "exit" and item["process"] is self.process:
                self.process = None
                self.connect_button.set_sensitive(True)
                self.latest_live = False
                self.status.set_text(f"接收器已退出，退出码 {item['code']}")
                if item["error"]:
                    self.detail.set_text(item["error"][-400:])
        if self.demo and not self.connect_button.get_sensitive():
            t = time.monotonic() - self.started
            self.show_snapshot({"valid": True, "valid_age_s": 0, "position_m": [0.2 + 0.15 * math.sin(t), 1.1 + 0.1 * math.cos(t * 0.7), -0.3 + 0.12 * math.sin(t * 0.5)], "euler_deg": [0, 0, 0], "valid_hz": 20, "packet_count": int(t * 20), "valid_count": int(t * 20), "invalid_count": 0, "motion_enabled": True, "sample_timestamp": time.time()})
        elif self.process is not None and self.last_snapshot is not None and time.monotonic() - self.last_snapshot > 0.5:
            self.latest_live = False
            self.button_status.set_text("扳机 / A：未知（接收器未更新）")
            self.status.set_text("接收器未更新 · 当前数值为历史值")
            for field in (*self.values, *self.hand_angles):
                field.set_text("—")
        self.plot.queue_draw()
        robot_connected = self.robot_process is not None
        self.robot_connect.set_sensitive(not robot_connected and not self.demo)
        source=self.last_source_state or {}
        source_elapsed=time.monotonic()-self.last_snapshot if self.last_snapshot is not None else float('inf')
        input_ready=bool(self.latest_live and source.get('motion_enabled') is True and
                         source.get('valid_age_s') is not None and source['valid_age_s']+source_elapsed < .25)
        self.robot_enable.set_sensitive(robot_connected and input_ready and not self.quest_app_busy and not self.robot_demo_active and not self.recovery_pending and not self.matrix_pending and self.robot_mode != "motion")
        if self.robot_mode=='motion':
            self.robot_enable.set_label('遥操已启用')
        elif self.teleop_button_setting.get_active_id()=='trigger':
            self.robot_enable.set_label('启用遥操' if input_ready else '等待头显有效位姿')
        self.robot_takeover.set_sensitive(robot_connected and not self.quest_app_busy and not self.robot_demo_active and not self.recovery_pending and not self.matrix_pending and self.latest_live and self.robot_mode == "motion")
        self.robot_release.set_sensitive(robot_connected)
        self.robot_stop.set_sensitive(robot_connected)
        self.apply_button.set_sensitive(not self.recovery_active and not self.recovery_pending and not self.robot_demo_active)
        self.responsive_button.set_sensitive(not self.recovery_active and not self.recovery_pending and not self.robot_demo_active and not self.quest_app_busy)
        for button in (self.gripper_open_button, self.gripper_close_button):
            button.set_sensitive(robot_connected and self.gripper_can_command and not self.quest_app_busy and not self.robot_demo_active
                                 and self.robot_last_snapshot is not None and time.monotonic() - self.robot_last_snapshot < .8)
        if self.quest_app_busy or not robot_connected or self.robot_last_snapshot is None or time.monotonic() - self.robot_last_snapshot > .8:
            self.matrix_choice.set_sensitive(False)
            self.matrix_select_button.set_sensitive(False)
        for button in (self.quest_open, self.quest_restart, self.quest_wake, self.quest_inspect):
            button.set_sensitive(not self.demo and not self.quest_app_busy and not self.closing)
        if self.auto_arm_pending and robot_connected and self.robot_last_snapshot is not None:
            self.auto_arm_pending = False
            self.robot_command("allow_controller")
        if robot_connected and self.robot_last_snapshot is not None and time.monotonic() - self.robot_last_snapshot > 0.8:
            self.robot_status.set_text("机器人遥测超时 · 请停止并检查连接")
            self.clear_robot_pose("机器人遥测超时 · 无实时位姿")
        self.robot_error.set_visible(bool(self.robot_error.get_text()))
        self.update_recovery_buttons()
        return True

    def show_snapshot(self, state):
        self.last_source_state = state
        self.show_quest_device()
        age = state.get("valid_age_s")
        self.latest_live = bool(state.get("valid") and age is not None and age < 0.25)
        def button(value):
            return "按下" if value is True else "松开" if value is False else "未上报"
        trigger = state.get("trigger_pressed") if self.latest_live else None
        a = state.get("a_pressed") if self.latest_live else None
        b = state.get('b_pressed') if self.latest_live else None
        self.button_status.set_text(f"右手食指扳机：{button(trigger)}    B：{button(b)}    A：{button(a)}"
            + (" · 旧输入无独立 B，请点“打开头显输入”" if self.latest_live and b is None else ""))
        if self.latest_live:
            self.position = state["position_m"]
            if self.reference is None:
                self.reference = list(self.position)
            for i in range(3):
                self.values[i].set_text(f"{self.position[i]:+.5f}")
                self.deltas[i].set_text(f"相对零点 Δ：{(self.position[i] - self.reference[i]) * 1000:+.1f} mm")
                self.hand_angles[i].set_text(f"{state['euler_deg'][i]:+.2f}")
            if state.get("sample_timestamp") != self.last_sample:
                self.last_sample = state["sample_timestamp"]
                self.history.append((time.monotonic(), list(self.position)))
                self.samples.append([self.last_sample, *self.position, *state["euler_deg"],
                                     *state.get('quaternion_wxyz', [None]*4)])
            stream = "继续" if state.get('motion_enabled') is True else "暂停"
            source = {'webxr':'WebXR 采集','unitree':'宇树 TeleVuer'}.get(state.get('input_backend'),'Stream')
            self.status.set_markup(f'<span foreground="#38bf85">● 追踪正常 · {state["valid_hz"]:g} Hz · {age*1000:.0f} ms · {source} {stream}</span>' if not self.demo else '<span foreground="#fbbf24">● 演示数据 · 不连接机械臂</span>')
            self.orientation.set_text("姿态 Rx / Ry / Rz：{} °    有效更新率：{} Hz    数据年龄：{:.0f} ms".format(" / ".join(f"{v:+.1f}" for v in state["euler_deg"]), state["valid_hz"], age * 1000))
            q = state.get('quaternion_wxyz')
            if q is not None and len(q) == 4 and all(math.isfinite(v) for v in q) and math.hypot(*q) > 1e-12:
                norm = math.hypot(*q)
                q = [v/norm for v in q]
                frame = (state.get('input_backend'), state.get('frame_id'))
                if self.quaternion_reference is None or frame != self.quaternion_frame or self.reference is None:
                    self.quaternion_reference, self.quaternion_frame = q, frame
                # q and -q represent the same orientation; do not subtract RPY.
                dot = abs(sum(a*b for a,b in zip(q,self.quaternion_reference)))
                angle = math.degrees(2*math.acos(min(1., dot)))
                self.orientation.set_text("四元数 w/x/y/z：{}\n相对诊断零点的测得转角：{:.3f}° · RPY 仅显示，不参与控制".format(
                    ' / '.join(f'{v:+.5f}' for v in q), angle))
        else:
            packet_age = state.get("packet_age_s")
            if packet_age is not None and packet_age < 0.5:
                text = "● 已收到数据，但位姿无效 · " + state.get("reason", "")
            elif state.get('input_backend')=='unitree' and packet_age is not None and packet_age >= 2:
                text = '● 头显已停止发送 XR 帧 · 点击“打开头显输入”，然后在头显点 Pass-through'
            elif state.get("pause_age_s") is not None and state["pause_age_s"] < 0.5:
                text = "● 只收到采集状态，没有有效位置 · 请检查手柄追踪及头显输入页面"
            else:
                text = ("● 等待数据 · Quest 浏览器输入上方 HTTPS 地址及配对码，再点 Pass-through" if self.lan_host else
                        "● 等待数据 · 请戴上 Quest，打开并开始头显输入，移动右手柄")
            self.status.set_markup('<span foreground="#fbbf24">' + GLib.markup_escape_text(text) + '</span>')
            self.orientation.set_text("无实时有效姿态" + (f" · 最后有效数据：{age:.1f} 秒前" if age is not None else ""))
            for value, delta in zip(self.values, self.deltas):
                value.set_text("—")
                delta.set_text("相对零点 Δ：— mm")
            for field in self.hand_angles:
                field.set_text("—")
        mode = "绿色 / 继续" if state.get("motion_enabled") else "红色 / 暂停" if state.get("motion_enabled") is False else "未知"
        self.counters.set_text(f"位姿包：{state.get('packet_count', 0)}    有效：{state.get('valid_count', 0)}    无效：{state.get('invalid_count', 0)}    应用状态：{mode}")
        if "position_span_mm" in state:
            self.movement.set_text("本次观测位置变化范围 ΔX / ΔY / ΔZ：" + " / ".join(f"{v:.1f}" for v in state["position_span_mm"]) + " mm")
        if not self.demo and time.monotonic() - self.last_state_save >= 1:
            path = ROOT / ".state" / "quest_pose_viewer.json"
            try:
                path.parent.mkdir(exist_ok=True)
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({key: value for key, value in state.items() if key != "process"}, ensure_ascii=False), encoding="utf-8")
                temporary.replace(path)
                self.last_state_save = time.monotonic()
            except OSError:
                pass

    def draw_plot(self, widget, cr):
        width, height = widget.get_allocated_width(), widget.get_allocated_height()
        cr.set_source_rgb(0.06, 0.09, 0.14)
        cr.paint()
        now = time.monotonic()
        points = [(t, p) for t, p in self.history if now - t <= 20]
        left, right = 100, width - 20
        row = height / 3
        cr.select_font_face("Sans")
        cr.set_font_size(12)
        for axis, color in enumerate(COLORS):
            top, bottom = axis * row + 16, (axis + 1) * row - 20
            cr.set_source_rgb(0.22, 0.27, 0.35)
            for f in (0, 0.5, 1):
                y = top + f * (bottom - top)
                cr.move_to(left, y)
                cr.line_to(right, y)
            cr.stroke()
            vals = [p[axis] for _, p in points]
            lo, hi = (min(vals), max(vals)) if vals else (-0.01, 0.01)
            span = max(hi - lo, 0.02)
            center = (hi + lo) / 2
            lo, hi = center - span * 0.6, center + span * 0.6
            cr.set_source_rgb(*color)
            cr.move_to(8, top + 12)
            cr.show_text(f"{'XYZ'[axis]} / m")
            cr.move_to(8, top + 31)
            cr.show_text(f"{hi:+.3f}")
            cr.move_to(8, bottom)
            cr.show_text(f"{lo:+.3f}")
            previous = None
            cr.set_line_width(2)
            for t, p in points:
                x = left + (1 - (now - t) / 20) * (right - left)
                y = bottom - (p[axis] - lo) / (hi - lo) * (bottom - top)
                if previous is None or t - previous > 0.25:
                    cr.move_to(x, y)
                else:
                    cr.line_to(x, y)
                previous = t
            cr.stroke()
            cr.set_source_rgb(0.6, 0.66, 0.74)
            cr.move_to(left, bottom + 15)
            cr.show_text("20s ago")
            cr.move_to(right - 30, bottom + 15)
            cr.show_text("now")

    def export(self, *_):
        if not self.samples:
            self.detail.set_text("没有有效位置样本，无法导出")
            return
        dialog = Gtk.FileChooserDialog(title="保存有效位置数据", transient_for=self, action=Gtk.FileChooserAction.SAVE)
        dialog.add_buttons("取消", Gtk.ResponseType.CANCEL, "保存", Gtk.ResponseType.OK)
        dialog.set_current_name("quest_position.csv")
        dialog.set_do_overwrite_confirmation(True)
        if dialog.run() == Gtk.ResponseType.OK:
            path = dialog.get_filename()
            try:
                with open(path, "w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["receive_timestamp", "x_m", "y_m", "z_m", "rx_deg", "ry_deg", "rz_deg",
                                     "qw", "qx", "qy", "qz"])
                    writer.writerows(self.samples)
                self.detail.set_text(f"已导出 {len(self.samples)} 个有效显示样本：{path}")
            except OSError as error:
                self.detail.set_text(f"保存失败：{error}")
        dialog.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--demo", action="store_true", help="明确标记的界面演示数据")
    parser.add_argument("--robot-preview", action="store_true", help="连接 xArm，仅目标预览，不使能运动")
    parser.add_argument("--open-headset",action="store_true",help="打开已配置的头显输入；确认停止后打开，不自动允许运动")
    parser.add_argument("--controller-control", action="store_true", help="连接后允许遥操；按住食指扳机跟随/松开暂停，夹爪使用配置中的独立手柄键，需真实有效数据")
    parser.add_argument("--matrix-open", "--alignment-open", dest="matrix_open", action="store_true", help="展开当前矩阵详情")
    parser.add_argument('--adopt-lan-input', action='store_true', help='接管已有本工作区 LAN 采集服务，不重启头显会话')
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--quit-after", type=float)
    parser.add_argument('--lan-host', help='只在指定私有 IP 上启动 HTTPS 采集；不使用 ADB')
    parser.add_argument('--usb', action='store_true', help='本次使用原 USB 输入连接，忽略已保存的 LAN 设置')
    args = parser.parse_args()
    config=json.loads((ROOT/'config.json').read_text())
    if args.usb and args.lan_host:
        parser.error('--usb 与 --lan-host 不能同时使用')
    host=None if args.usb else args.lan_host or (config.get('quest_lan_host') if config.get('quest_input_transport')=='lan' else None)
    window = PoseWindow(demo=args.demo, controller_control=args.controller_control, lan_host=host)
    if args.adopt_lan_input:
        window.adopt_lan_input()
        ready = ROOT / '.state/gui_input_adopted.json'
        import os
        ready.write_text(json.dumps(dict(gui_pid=os.getpid(), bridge_pid=window.webxr_process.pid)), encoding='utf-8')
    window.matrix_details.set_expanded(args.matrix_open)
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(window.destroy))
    signal.signal(signal.SIGINT, lambda *_: GLib.idle_add(window.destroy))
    window.show_all()
    if args.connect or args.demo:
        GLib.idle_add(window.connect_quest)
    if args.robot_preview or args.controller_control:
        GLib.idle_add(window.connect_robot)
    if args.open_headset and not args.demo:
        GLib.timeout_add(1000,lambda:window.quest_app_action('start') or False)
    if args.screenshot:
        def capture():
            pixbuf = Gdk.pixbuf_get_from_window(window.get_window(), 0, 0, window.get_allocated_width(), window.get_allocated_height())
            pixbuf.savev(str(args.screenshot), "png", [], [])
            return False
        GLib.timeout_add(4000, capture)
    if args.quit_after:
        GLib.timeout_add(int(args.quit_after * 1000), lambda: window.destroy() or False)
    Gtk.main()


if __name__ == "__main__":
    main()
