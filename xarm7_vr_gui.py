#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, GLib, Gtk  # noqa: E402


CLI = Path("/home/a/.local/bin/xarm7-vr")
GUIDE = Path("/home/a/xarm7-vr-lab/使用说明.md")
TELEMETRY = Path("/home/a/xarm7-vr-lab/.state/control_telemetry.json")
CONTROLLER_TELEMETRY = Path("/home/a/xarm7-vr-lab/.state/controller_telemetry.json")
CONTROLLER_MONITOR = Path("/home/a/xarm7-vr-lab/quest_controller_monitor.py")
RENDERER = Path("/home/a/xarm7-vr-lab/xarm7_renderer.py")
RENDER_IMAGE = Path("/home/a/xarm7-vr-lab/.state/xarm7_render.png")
RENDER_PYTHON = Path("/home/a/miniconda3/envs/openteach/bin/python")


class ControlWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application):
        super().__init__(application=app, title="XARM7 VR 控制台")
        self.set_wmclass("xarm7-vr-control", "XArm7VRControl")
        self.set_icon_from_file("/home/a/xarm7-vr-lab/xarm7-vr-control.svg")
        self.set_default_size(920, 980)
        self.set_border_width(18)
        self.process: subprocess.Popen[str] | None = None
        self.renderer_process: subprocess.Popen[str] | None = None
        self.controller_monitor_process: subprocess.Popen[str] | None = None
        self.controller_monitor_started_at = 0.0
        self.render_mtime_ns = 0
        self.operation = ""
        self.closing = False
        self.telemetry_age = float("inf")
        self.connect("delete-event", self.on_close)
        self.connect("destroy", self.stop_renderer)
        self.connect("destroy", self.stop_controller_monitor)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        self.add(root)

        title = Gtk.Label()
        title.set_markup("<span size='x-large' weight='bold'>XARM7 · Quest 遥操</span>")
        title.set_xalign(0)
        root.pack_start(title, False, False, 0)

        subtitle = Gtk.Label(label="XARM7 + G2 · 双 USB 相机 · LeRobot")
        subtitle.set_xalign(0)
        subtitle.get_style_context().add_class("dim-label")
        root.pack_start(subtitle, False, False, 0)

        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        status_box.get_style_context().add_class("status-box")
        self.status_dot = Gtk.Label(label="●")
        self.status_dot.get_style_context().add_class("status-idle")
        self.status_label = Gtk.Label(label="已停止 · 机械臂未由本应用控制")
        self.status_label.set_xalign(0)
        status_box.pack_start(self.status_dot, False, False, 0)
        status_box.pack_start(self.status_label, True, True, 0)
        root.pack_start(status_box, False, False, 0)

        settings = Gtk.Frame(label="安全试动设置")
        grid = Gtk.Grid(column_spacing=16, row_spacing=10, margin=14)
        settings.add(grid)
        root.pack_start(settings, False, False, 0)

        grid.attach(Gtk.Label(label="TCP 速度", xalign=0), 0, 0, 1, 1)
        self.speed = Gtk.ComboBoxText()
        for value in ("5", "10", "20", "30", "50"):
            self.speed.append_text(value)
        self.speed.set_active(4)
        grid.attach(self.speed, 1, 0, 1, 1)
        grid.attach(Gtk.Label(label="mm/s", xalign=0), 2, 0, 1, 1)

        grid.attach(Gtk.Label(label="XYZ 围栏半径", xalign=0), 0, 1, 1, 1)
        self.radius = Gtk.ComboBoxText()
        for value in ("10", "25", "50", "100", "200", "300"):
            self.radius.append_text(value)
        self.radius.set_active(0)
        grid.attach(self.radius, 1, 1, 1, 1)
        grid.attach(Gtk.Label(label="mm（以启动位置为中心）", xalign=0), 2, 1, 1, 1)

        grid.attach(Gtk.Label(label="手柄位移比例", xalign=0), 0, 2, 1, 1)
        self.scale = Gtk.ComboBoxText()
        for label, value in (("精细 0.25×", "0.25"), ("中等 0.5×", "0.5"), ("直接 1.0×", "1.0")):
            self.scale.append(value, label)
        self.scale.set_active_id("0.25")
        grid.attach(self.scale, 1, 2, 2, 1)

        note = Gtk.Label(
            label="试动模式始终禁用末端旋转；范围越大，碰撞风险越高。正式数据采集使用独立工作空间。",
            xalign=0,
        )
        note.set_line_wrap(True)
        note.get_style_context().add_class("warning-label")
        grid.attach(note, 0, 3, 3, 1)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        root.pack_start(actions, False, False, 0)
        self.check_button = Gtk.Button(label="设备检查")
        self.quest_button = Gtk.Button(label="连接 Quest")
        self.monitor_button = Gtk.Button(label="只读监视手柄")
        self.start_button = Gtk.Button(label="启动遥操")
        self.start_button.get_style_context().add_class("suggested-action")
        self.stop_button = Gtk.Button(label="停止遥操")
        self.stop_button.get_style_context().add_class("destructive-action")
        self.stop_button.set_sensitive(False)
        for button in (self.check_button, self.quest_button, self.monitor_button, self.start_button, self.stop_button):
            actions.pack_start(button, True, True, 0)
        self.check_button.connect("clicked", lambda *_: self.run_utility([str(CLI), "status"], "设备检查"))
        self.quest_button.connect(
            "clicked", lambda *_: self.run_utility([str(CLI), "quest-setup"], "Quest 连接")
        )
        self.monitor_button.connect("clicked", self.start_controller_preview)
        self.start_button.connect("clicked", self.start_teleop)
        self.stop_button.connect("clicked", lambda *_: self.stop_process())

        collection = Gtk.Frame(label="LeRobot 双相机数据采集")
        collection_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=9, margin=12)
        collection.add(collection_box)
        collection_hint = Gtk.Label(
            label="自动记录腕部/全局视频、关节、TCP、夹爪和动作；每回合 60 秒，回合间重置 15 秒。",
            xalign=0,
        )
        collection_hint.set_line_wrap(True)
        collection_box.pack_start(collection_hint, False, False, 0)
        collection_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.record_start_button = Gtk.Button(label="开始采集")
        self.record_start_button.get_style_context().add_class("suggested-action")
        self.record_stop_button = Gtk.Button(label="停止采集")
        self.record_stop_button.get_style_context().add_class("destructive-action")
        self.record_stop_button.set_sensitive(False)
        self.record_start_button.connect("clicked", self.start_recording)
        self.record_stop_button.connect("clicked", lambda *_: self.stop_process())
        collection_actions.pack_start(self.record_start_button, True, True, 0)
        collection_actions.pack_start(self.record_stop_button, True, True, 0)
        collection_box.pack_start(collection_actions, False, False, 0)
        root.pack_start(collection, False, False, 0)

        controls = Gtk.Label(
            label="手柄：B 暂停/继续 · 0.8 秒内快速双按 B 切换夹爪 · 跟踪丢失自动保持",
            xalign=0,
        )
        controls.set_line_wrap(True)
        root.pack_start(controls, False, False, 0)

        controller = Gtk.Frame(label="Quest 右手柄 · 实时状态与 6D 位姿（Quest 世界坐标系）")
        controller_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5, margin=10)
        controller.add(controller_box)
        self.controller_status = Gtk.Label(label="等待手柄数据", xalign=0)
        self.controller_pose = Gtk.Label(label="位置 XYZ：—　姿态 Rx Ry Rz：—", xalign=0)
        self.controller_status.set_selectable(True)
        self.controller_pose.set_selectable(True)
        controller_box.pack_start(self.controller_status, False, False, 0)
        controller_box.pack_start(self.controller_pose, False, False, 0)
        root.pack_start(controller, False, False, 0)

        planner = Gtk.Frame(label="XARM7 IK + MPC 实时监视")
        planner_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7, margin=10)
        planner.add(planner_box)
        self.planner_status = Gtk.Label(label="等待控制数据", xalign=0)
        self.planner_status.set_selectable(True)
        planner_box.pack_start(self.planner_status, False, False, 0)
        self.tcp_status = Gtk.Label(label="TCP：—", xalign=0)
        self.tcp_status.set_selectable(True)
        planner_box.pack_start(self.tcp_status, False, False, 0)
        joint_grid = Gtk.Grid(column_spacing=14, row_spacing=3)
        for column, heading in enumerate(("关节", "当前角°", "IK目标°", "MPC指令°", "本周期Δ°", "速度°/s")):
            label = Gtk.Label()
            label.set_markup(f"<b>{heading}</b>")
            label.set_xalign(1 if column else 0)
            joint_grid.attach(label, column, 0, 1, 1)
        self.joint_labels: list[list[Gtk.Label]] = []
        for index in range(7):
            row_labels = []
            name = Gtk.Label(label=f"J{index + 1}", xalign=0)
            joint_grid.attach(name, 0, index + 1, 1, 1)
            for column in range(1, 6):
                value = Gtk.Label(label="—", xalign=1)
                value.set_selectable(True)
                value.set_width_chars(9)
                joint_grid.attach(value, column, index + 1, 1, 1)
                row_labels.append(value)
            self.joint_labels.append(row_labels)
        planner_body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)
        planner_body.pack_start(joint_grid, False, False, 0)
        animation_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        self.robot_image = Gtk.Image()
        self.robot_image.set_size_request(350, 215)
        self.robot_image.set_from_icon_name("image-loading", Gtk.IconSize.DIALOG)
        animation_box.pack_start(self.robot_image, True, True, 0)
        legend = Gtk.Label(label="官方 XARM7 + G2 三维模型 · 跟随实测关节实时运动", xalign=0.5)
        legend.get_style_context().add_class("dim-label")
        animation_box.pack_start(legend, False, False, 0)
        planner_body.pack_start(animation_box, True, True, 0)
        planner_box.pack_start(planner_body, False, False, 0)
        root.pack_start(planner, False, False, 0)
        GLib.timeout_add(100, self.refresh_telemetry)

        log_frame = Gtk.Frame(label="运行日志")
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.log = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.log.set_left_margin(8)
        self.log.set_right_margin(8)
        scroll.add(self.log)
        log_frame.add(scroll)
        root.pack_start(log_frame, True, True, 0)

        footer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.clear_button = Gtk.Button(label="清空日志")
        self.clear_button.connect("clicked", lambda *_: self.log.get_buffer().set_text(""))
        guide_button = Gtk.Button(label="打开使用说明")
        guide_button.connect("clicked", self.open_guide)
        footer.pack_start(self.clear_button, False, False, 0)
        footer.pack_end(guide_button, False, False, 0)
        root.pack_start(footer, False, False, 0)

        css = b"""
        .status-box { padding: 10px; border-radius: 8px; background: alpha(@theme_fg_color, 0.06); }
        .status-idle { color: #6b7280; }
        .status-busy { color: #f59e0b; }
        .status-running { color: #22c55e; }
        .status-error { color: #ef4444; }
        .warning-label { color: #b45309; }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            self.get_screen(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self.start_renderer()

    def append_log(self, text: str) -> bool:
        buffer = self.log.get_buffer()
        buffer.insert(buffer.get_end_iter(), text)
        mark = buffer.create_mark(None, buffer.get_end_iter(), False)
        self.log.scroll_mark_onscreen(mark)
        return False

    def refresh_telemetry(self) -> bool:
        try:
            payload = json.loads(TELEMETRY.read_text(encoding="utf-8"))
            age = max(0.0, time.time() - float(payload["timestamp"]))
            self.telemetry_age = age
            status = str(payload.get("status", "未知"))
            horizon = payload.get("mpc_horizon", "—")
            freshness = "实时" if age < 0.6 else f"已停止/数据陈旧 {age:.1f}s"
            self.planner_status.set_text(f"IK：{status}　MPC：H={horizon}　{freshness}")
            current_tcp = payload.get("current_tcp", [])
            target_tcp = payload.get("target_tcp", [])
            command_tcp = payload.get("command_tcp", [])
            self.tcp_status.set_text(
                "TCP XYZ mm　当前 {}　Quest目标 {}　MPC一步 {}".format(
                    self._xyz_text(current_tcp),
                    self._xyz_text(target_tcp),
                    self._xyz_text(command_tcp),
                )
            )
            columns = (
                payload.get("current_deg"),
                payload.get("ik_target_deg"),
                payload.get("mpc_command_deg"),
                payload.get("delta_deg"),
                payload.get("velocity_deg_s"),
            )
            for joint_index, labels in enumerate(self.joint_labels):
                for values, label in zip(columns, labels, strict=True):
                    value = None if values is None or len(values) <= joint_index else values[joint_index]
                    label.set_text("—" if value is None else f"{float(value):.3f}")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.planner_status.set_text("等待 IK/MPC 控制数据")
        self.refresh_controller()
        self.refresh_robot_image()
        return True

    def refresh_controller(self) -> None:
        process = self.controller_monitor_process
        if process is None or process.poll() is not None:
            self.controller_status.set_text("手柄监视未运行")
            self.controller_pose.set_text("位置 XYZ：—　姿态 Rx Ry Rz：—")
            return
        try:
            payload = json.loads(CONTROLLER_TELEMETRY.read_text(encoding="utf-8"))
            if CONTROLLER_TELEMETRY.stat().st_mtime < self.controller_monitor_started_at:
                raise ValueError("stale controller telemetry")
            frame_time = payload.get("frame_timestamp")
            pause_time = payload.get("pause_timestamp")
            trigger_time = payload.get("trigger_timestamp")
            frame_age = float("inf") if frame_time is None else max(0.0, time.time() - float(frame_time))
            pause_age = float("inf") if pause_time is None else max(0.0, time.time() - float(pause_time))
            trigger_age = float("inf") if trigger_time is None else max(0.0, time.time() - float(trigger_time))
            tracking = "跟踪正常" if frame_age < 0.25 else "暂停中" if payload.get("motion_enabled") is False and pause_age < 1.0 else "无实时位姿"
            motion = "继续" if payload.get("motion_enabled") else "暂停"
            if pause_age > 1.0:
                motion = "状态未更新"
            trigger = "按下" if payload.get("trigger_pressed") else "松开"
            if trigger_age > 1.0:
                trigger = "未上报"
            hint = "　请在头显进入 Menu → Stream，再按 B" if frame_age >= 0.25 else ""
            self.controller_status.set_text(f"B/运动：{motion}　跟踪：{tracking}　食指扳机：{trigger}{hint}")
            position = payload.get("position_m")
            euler = payload.get("euler_xyz_deg")
            if frame_age < 0.25 and isinstance(position, list) and isinstance(euler, list) and len(position) == len(euler) == 3:
                self.controller_pose.set_text(
                    "位置 XYZ (m)：[{}]　姿态 Rx Ry Rz (°)：[{}]".format(
                        ", ".join(f"{float(value):.3f}" for value in position),
                        ", ".join(f"{float(value):.1f}" for value in euler),
                    )
                )
            else:
                self.controller_pose.set_text("位置 XYZ：—　姿态 Rx Ry Rz：—（无新位姿）")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.controller_status.set_text("等待手柄数据")
            self.controller_pose.set_text("位置 XYZ：—　姿态 Rx Ry Rz：—")

    @staticmethod
    def _xyz_text(values: list[float]) -> str:
        if len(values) < 3:
            return "—"
        return "[" + ", ".join(f"{float(value):.1f}" for value in values[:3]) + "]"

    def start_renderer(self) -> None:
        try:
            self.renderer_process = subprocess.Popen(
                [str(RENDER_PYTHON), str(RENDERER)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self.append_log(f"三维模型渲染器启动失败：{exc}\n")

    def refresh_robot_image(self) -> None:
        try:
            mtime_ns = RENDER_IMAGE.stat().st_mtime_ns
            if mtime_ns == self.render_mtime_ns:
                return
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                str(RENDER_IMAGE), 350, 215, True
            )
            self.robot_image.set_from_pixbuf(pixbuf)
            self.render_mtime_ns = mtime_ns
        except (OSError, GLib.Error):
            pass

    def stop_renderer(self, *_args) -> None:
        process = self.renderer_process
        self.renderer_process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def start_controller_monitor(self) -> None:
        try:
            self.controller_monitor_started_at = time.time()
            self.controller_monitor_process = subprocess.Popen(
                [str(RENDER_PYTHON), str(CONTROLLER_MONITOR), "--output", str(CONTROLLER_TELEMETRY)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self.append_log(f"手柄监视启动失败：{exc}\n")

    def stop_controller_monitor(self, *_args) -> None:
        process = self.controller_monitor_process
        self.controller_monitor_process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def set_status(self, text: str, style: str) -> None:
        context = self.status_dot.get_style_context()
        for name in ("status-idle", "status-busy", "status-running", "status-error"):
            context.remove_class(name)
        context.add_class(style)
        self.status_label.set_text(text)

    def set_controls_busy(self, busy: bool, teleop: bool = False, recording: bool = False, monitor: bool = False) -> None:
        self.check_button.set_sensitive(not busy)
        self.quest_button.set_sensitive(not busy)
        self.monitor_button.set_sensitive(not busy)
        self.start_button.set_sensitive(not busy)
        self.speed.set_sensitive(not busy)
        self.radius.set_sensitive(not busy)
        self.scale.set_sensitive(not busy)
        self.stop_button.set_sensitive(busy and (teleop or monitor))
        self.record_start_button.set_sensitive(not busy)
        self.record_stop_button.set_sensitive(busy and recording)

    def run_utility(self, command: list[str], operation: str) -> None:
        if self.process is not None:
            return
        self.launch(command, operation, teleop=False)

    def start_controller_preview(self, _button: Gtk.Button) -> None:
        if self.process is None:
            self.launch([str(CLI), "bridge"], "只读手柄监视", monitor=True)

    def start_teleop(self, _button: Gtk.Button) -> None:
        if self.process is not None:
            return
        speed = self.speed.get_active_text()
        radius = self.radius.get_active_text()
        scale = self.scale.get_active_id()
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.NONE,
            text="即将使能真实 XARM7",
        )
        dialog.format_secondary_text(
            f"速度 {speed} mm/s，XYZ 围栏半径 ±{radius} mm，手柄比例 {scale}×。\n\n"
            "请确认工作区无人、急停可触达、线缆无缠绕。"
        )
        dialog.add_button("取消", Gtk.ResponseType.CANCEL)
        dialog.add_button("确认启动", Gtk.ResponseType.OK)
        response = dialog.run()
        dialog.destroy()
        if response != Gtk.ResponseType.OK:
            return
        command = [
            str(CLI),
            "teleop",
            "--enable-motion",
            "--yes",
            f"--speed-mm-s={speed}",
            f"--range-mm={radius}",
            f"--translation-scale={scale}",
        ]
        self.launch(command, "Quest 遥操", teleop=True)

    def start_recording(self, _button: Gtk.Button) -> None:
        if self.process is not None:
            return
        speed = self.speed.get_active_text()
        radius = self.radius.get_active_text()
        scale = self.scale.get_active_id()
        dataset_name = f"paper_cup_handover_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_path = f"/home/a/lerobot_datasets/{dataset_name}"
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.NONE,
            text="即将使能机械臂并开始数据采集",
        )
        dialog.format_secondary_text(
            f"数据集：{dataset_name}\n速度：{speed} mm/s\n动态围栏：启动点周围 ±{radius} mm\n"
            f"手柄比例：{scale}×\n\n"
            "将同步保存两路相机、机器人状态、动作和夹爪数据。请确认工作区无人、急停可触达。"
        )
        dialog.add_button("取消", Gtk.ResponseType.CANCEL)
        dialog.add_button("确认并开始采集", Gtk.ResponseType.OK)
        response = dialog.run()
        dialog.destroy()
        if response != Gtk.ResponseType.OK:
            return
        self.append_log(f"\n数据将保存到：{dataset_path}\n")
        command = [
            str(CLI),
            "record",
            dataset_name,
            "--task=抓起面前的空纸杯，移动到黑色方框上方并停稳，递给客人",
            "--episodes=50",
            "--episode-seconds=60",
            "--reset-seconds=15",
            "--enable-motion",
            "--yes",
            f"--speed-mm-s={speed}",
            f"--dynamic-range-mm={radius}",
            f"--translation-scale={scale}",
            "--disable-rotation",
        ]
        self.launch(command, f"采集 {dataset_name}", recording=True)

    def launch(
        self,
        command: list[str],
        operation: str,
        *,
        teleop: bool = False,
        recording: bool = False,
        monitor: bool = False,
    ) -> None:
        self.operation = operation
        self.append_log(f"\n$ {' '.join(command)}\n")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            self.append_log(f"启动失败：{exc}\n")
            self.set_status("启动失败", "status-error")
            self.process = None
            return
        if teleop or recording or monitor:
            self.start_controller_monitor()
        self.set_controls_busy(True, teleop=teleop, recording=recording, monitor=monitor)
        if recording:
            status = "正在采集 · 点击“停止采集”保存当前回合并安全停止"
            style = "status-running"
        elif teleop:
            status = "遥操运行中 · 点击红色按钮安全停止"
            style = "status-running"
        elif monitor:
            status = "只读手柄监视中 · 不连接机械臂"
            style = "status-busy"
        else:
            status = f"正在执行：{operation}"
            style = "status-busy"
        self.set_status(status, style)
        threading.Thread(
            target=self.read_process, args=(self.process, teleop, recording), daemon=True
        ).start()

    def read_process(self, process: subprocess.Popen[str], teleop: bool, recording: bool) -> None:
        assert process.stdout is not None
        failure = ""
        for line in process.stdout:
            GLib.idle_add(self.append_log, line)
            if line.startswith("FAIL:") or "端口已被占用" in line or "桥已由另一项任务运行" in line:
                failure = line.removeprefix("FAIL:").strip()
        code = process.wait()
        GLib.idle_add(self.process_finished, process, code, teleop, recording, failure)

    def process_finished(
        self, process: subprocess.Popen[str], code: int, teleop: bool, recording: bool, failure: str
    ) -> bool:
        if self.process is process:
            self.process = None
        self.stop_controller_monitor()
        self.set_controls_busy(False)
        if code in (0, 130, -signal.SIGINT):
            message = "采集已停止并保存 · 机械臂已断开" if recording else "已停止 · 机械臂未由本应用控制"
            self.set_status(message, "status-idle")
        else:
            detail = f"：{failure}" if failure else "；请查看运行日志"
            self.set_status(f"{self.operation}失败（退出码 {code}）{detail}", "status-error")
        if self.closing:
            self.destroy()
        return False

    def stop_process(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.set_status("正在安全停止…", "status-busy")
        self.stop_button.set_sensitive(False)
        self.append_log("\n正在请求安全停止…\n")
        try:
            os.killpg(self.process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def open_guide(self, _button: Gtk.Button) -> None:
        subprocess.Popen(["xdg-open", str(GUIDE)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def on_close(self, *_args) -> bool:
        if self.process is None:
            return False
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.YES_NO,
            text="机械臂任务仍在运行，是否安全停止并退出？",
        )
        response = dialog.run()
        dialog.destroy()
        if response == Gtk.ResponseType.YES:
            self.closing = True
            self.stop_process()
        return True


class ControlApp(Gtk.Application):
    def __init__(self) -> None:
        # A new ID lets this console open beside an older running instance.
        super().__init__(application_id="com.local.XArm7VRControl4")

    def do_activate(self) -> None:
        window = self.props.active_window or ControlWindow(self)
        window.show_all()
        window.present()


if __name__ == "__main__":
    raise SystemExit(ControlApp().run(sys.argv))
