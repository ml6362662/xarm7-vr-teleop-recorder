#!/usr/bin/env python3
"""Independent Quest/xArm recorder. No robot SDK; JSON commands on stdin.

Run with the configured LeRobot Python (numpy, OpenCV). Startup opens cameras
only. Commands: start(task=...), save(success=true/false/null), discard, shutdown.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import sys
import threading
import time
from collections import deque

from quest_recording_stream import RECORD_SOCKET, ROOT, STREAM_VERSION
from recording_data import EpisodeWriter, RAW_VERSION, atomic_json, json_text, training_vectors
from recording_gate import TriggerCaptureGate


class CameraWorker:
    RETRY_SECONDS = 1.

    def __init__(self, config, inbox, stop, preview_dir):
        self.config, self.inbox, self.stop = config, inbox, stop
        self.preview_dir = preview_dir
        self.state = dict(ready=False, error="", frames=0, dropped=0)
        self.thread = threading.Thread(target=self.run, daemon=True, name="camera-"+config["name"])
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            self.capture_once()
            if self.stop.wait(self.RETRY_SECONDS):
                break

    def capture_once(self):
        import cv2
        c = self.config
        cap = None
        try:
            if not Path(c['path']).exists():
                raise RuntimeError(f"设备未连接：{c['path']}；请检查 USB 连接，正在自动重试")
            cap = cv2.VideoCapture(str(c["path"]), cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开 {c['path']}；请检查设备占用/权限，正在自动重试")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.get("fourcc", "MJPG")))
            for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH,c["width"]),
                                (cv2.CAP_PROP_FRAME_HEIGHT,c["height"]),
                                (cv2.CAP_PROP_FPS,c["fps"]), (cv2.CAP_PROP_BUFFERSIZE,1)):
                cap.set(prop, value)
            controls = {"auto_exposure":cv2.CAP_PROP_AUTO_EXPOSURE, "exposure":cv2.CAP_PROP_EXPOSURE,
                        "gain":cv2.CAP_PROP_GAIN,
                        "autofocus":cv2.CAP_PROP_AUTOFOCUS, "focus":cv2.CAP_PROP_FOCUS}
            control_results = {}
            for key, prop in controls.items():
                if key in c:
                    control_results[key] = dict(requested=c[key], accepted=cap.set(prop,float(c[key])), actual=cap.get(prop))
            self.state.update(controls=control_results, negotiated_fps=cap.get(cv2.CAP_PROP_FPS),
                              timestamp_kind="host_read_completion_not_exposure")
            sequence, last_preview = self.state['frames'], 0.
            initial_sequence = sequence
            started = time.monotonic()
            while not self.stop.is_set():
                read_start = time.monotonic()
                ok, bgr = cap.read()
                stamp = time.monotonic()
                if not ok:
                    raise RuntimeError("相机读取失败")
                if bgr.shape[:2] != (c["height"], c["width"]):
                    raise RuntimeError(f"相机分辨率不符：{bgr.shape}")
                # Archive the decoded sensor frame before configured orientation.
                ok, png = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION,1])
                if not ok:
                    raise RuntimeError("PNG 编码失败")
                item = dict(camera=c["name"], sequence=sequence, t=stamp, read_start=read_start,
                            width=bgr.shape[1], height=bgr.shape[0], png=png.tobytes())
                try:
                    self.inbox.put_nowait(item)
                except queue.Full:
                    self.state["dropped"] += 1
                sequence += 1
                self.state.update(ready=stamp-started >= 1., error="", frames=sequence, last_frame_s=stamp,
                                  measured_fps=(sequence-initial_sequence)/max(.001,stamp-started))
                if stamp-last_preview >= .3:
                    rotation = c.get("rotation",0)
                    if rotation:
                        bgr = cv2.rotate(bgr, {90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
                    thumb = cv2.resize(bgr, (320,240))
                    ok, jpg = cv2.imencode(".jpg",thumb)
                    if ok:
                        path = self.preview_dir/(c["name"]+".jpg")
                        tmp = path.with_suffix(".tmp")
                        tmp.write_bytes(jpg.tobytes())
                        tmp.replace(path)
                    last_preview = stamp
        except Exception as error:
            self.state.update(ready=False, error=str(error), measured_fps=0.)
        finally:
            if cap is not None:
                cap.release()


class Recorder:
    def __init__(self, root, config, socket_path=RECORD_SOCKET, status_path=None):
        self.root, self.config = Path(root), config
        self.socket_path = Path(socket_path)
        self.status_path = Path(status_path or ROOT/".state"/"recorder_status.json")
        self.preview_dir = self.status_path.parent/(self.status_path.stem+"_preview")
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.commands, self.camera_queue = queue.Queue(maxsize=32), queue.Queue(maxsize=120)
        self.cameras = []
        self.episode = None
        self.latest = None
        self.last_sequence = None
        self.last_message = "正在准备相机和机器人数据订阅"
        self.completed = 0
        self.last_path = None
        self.sequence_gaps = 0
        self.shutdown_requested = False
        self.gate = None
        self.pending_images = deque()
        self.camera_last_sequences = {}
        self.camera_gaps = 0
        self.last_sample_segment = None
        self.reviewing = False
        self.input_pause = None
        self.awaiting_input_resume = False

    def command_reader(self):
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("命令必须为 JSON 对象")
                self.commands.put(request, timeout=1)
            except (ValueError, queue.Full):
                self.last_message = "采集命令无效/命令队列已满"
        self.shutdown_requested = True  # GUI disappeared: preserve as incomplete.

    def ready_error(self):
        errors = []
        control_error = self.control_ready_error()
        if control_error:
            errors.append(control_error)
        now = time.monotonic()
        for camera in self.cameras:
            state = camera.state
            if not state['ready'] or now-state.get('last_frame_s',0) > .3:
                errors.append(f"{camera.config['name']}：" + (state.get('error') or '相机预热中或图像过期'))
        if not self.cameras:
            errors.append("未配置相机")
        if shutil.disk_usage(self.root).free < 2*1024**3:
            errors.append("可用磁盘空间不足 2 GiB")
        return "\n".join(errors)

    def control_ready_error(self):
        now = time.monotonic()
        if self.latest is None or now-self.latest["monotonic_s"] > .25:
            return "等待新版遥操控制器数据（已运行的旧进程需重新启动）"
        if self.latest.get('recovery_active'):
            return "机械臂正在复位，请等待复位完成后采集"
        if self.latest.get('error'):
            return "控制器错误：" + str(self.latest['error'])
        if not self.latest.get('source_valid'):
            return "右手柄追踪无效：请戴上头显、保持 VR 会话在前台，并让右手柄进入头显视野"
        if self.latest.get('mode') != 'motion':
            return "遥操未启用，请先启用遥操后采集"
        if self.latest.get('pause_settling'):
            return "机械臂正在停止，请等待停稳后采集"
        if type((self.latest.get('hand') or {}).get('trigger_pressed')) is not bool:
            return "等待独立食指扳机状态"
        if self.latest.get('settings',{}).get('teleop_button','trigger') != 'trigger':
            return "请将遥操按键设为食指扳机后采集"
        try:
            training_vectors(self.latest)
        except ValueError as error:
            return str(error)
        return ""

    def handle(self, request):
        command = request.get("command")
        if command == "start":
            if self.episode:
                raise ValueError("已有录制中的 episode")
            task = str(request.get("task", "")).strip()
            if not task:
                raise ValueError("请填写任务描述")
            error = self.ready_error()
            if error:
                raise ValueError(error)
            metadata = dict(task=task, robot_type="xarm7_g2", nominal_fps=30,
                capture_mode="trigger", capture_intervals=[],
                input_interruptions=[],
                input_interruption_policy='pause same task; omit invalid interval; require fresh release/repress; controller faults remain terminal',
                cameras=self.config["cameras"], run_id=self.latest["run_id"],
                robot_info=self.latest.get("robot_info",{}),
                settings=self.latest["settings"], settings_revision=self.latest["settings_revision"],
                camera_status={c.config["name"]: dict(c.state) for c in self.cameras},
                clock="host CLOCK_MONOTONIC seconds", timestamp_kind="host acquisition, not hardware synchronized",
                units=dict(joints="rad", tcp_position="m", quaternion="wxyz", gripper="mm"),
                gripper_normalization=dict(min_mm=0,max_mm=84,zero="closed",one="open"),
                image_encoding="lossless PNG of decoded native BGR, stored before rotation",
                calibration={"intrinsics":None,"extrinsics":None,"status":"not calibrated"},
                code_hashes=code_hashes())
            self.episode = EpisodeWriter(self.root, metadata)
            self.reviewing = False
            self.input_pause = None
            self.awaiting_input_resume = False
            hand = self.latest.get('hand') or {}
            # Starting a recorder never asks the operator to interrupt teleop.
            # Capture begins at the next eligible sample, even if already held.
            self.gate = TriggerCaptureGate(False,hand.get('trigger_release_generation'))
            self.pending_images.clear()
            self.camera_last_sequences.clear()
            self.camera_gaps = 0
            self.last_sample_segment = None
            self.sequence_gaps = 0
            self.last_sequence = None
            self.last_message = "本条已准备 · 自动记录后续有效遥操，遥操控制不变"
        elif command in ("review", "save", "discard"):
            if not self.episode:
                raise ValueError("没有正在录制的 episode")
            if command == 'review':
                if not self.reviewing:
                    self.freeze_capture()
                    self.drain_images(final=True)
                    self.reviewing = True
                    self.episode.event('review_started',dict(samples=self.episode.samples))
                    self.episode.flush()
                self.last_message = "本条录制已冻结，请人工判定成功/失败，并选择保留或丢弃"
                return
            success = request.get("success")
            if success is not None and type(success) is not bool:
                raise ValueError("success 必须为 true / false / null")
            if command == 'save' and type(success) is not bool:
                raise ValueError("保存前必须人工选择成功或失败")
            annotation = dict(success=success, outcome='unknown' if success is None else
                              'success' if success else 'failure',
                              retained=command=='save', source='operator_review',
                              annotated_utc_s=time.time())
            self.episode.metadata['review'] = annotation
            self.episode.metadata['outcome'] = annotation['outcome']
            self.episode.metadata['retained'] = annotation['retained']
            self.episode.event('operator_review',annotation)
            self.finish("complete" if command == "save" else "discarded", success,
                        "operator_save" if command == "save" else "operator_discard")
        elif command == "shutdown":
            self.shutdown_requested = True
        else:
            raise ValueError("未知采集命令")

    def freeze_capture(self):
        # Stop at the last observed controller watermark; never include unseen
        # motion while the user judges an episode. Keep the final sample inside
        # the half-open interval, including when recording ends while following.
        if self.gate and self.gate.current is not None:
            self.gate.close(self.gate.last_time + 1e-6)

    def finish(self, status, success=None, reason=""):
        if self.episode is None:
            return
        if self.gate:
            self.freeze_capture()
            self.drain_images(final=True)
            self.episode.metadata['capture_intervals'] = self.gate.intervals
            self.episode.metadata['recorded_duration_s'] = self.gate.duration(time.monotonic())
        quality = dict(telemetry_gaps=self.sequence_gaps, camera_queue_drops=self.camera_gaps)
        if status == "complete" and (quality["telemetry_gaps"] or quality["camera_queue_drops"]):
            status, reason = "incomplete", "数据丢包/写入跟不上，保留原始记录，拒绝标成完整轨迹"
        if status == "complete" and (self.episode.samples < 2 or any(self.episode.images.get(c["name"],0)<2 for c in self.config["cameras"])):
            status, reason = "incomplete", "轨迹太短/缺少相机帧"
        path = self.episode.finish(status, success, reason, quality)
        self.episode = None
        self.gate = None
        self.reviewing = False
        self.awaiting_input_resume = False
        self.pending_images.clear()
        self.last_path = str(path)
        self.completed += status == "complete"
        self.last_message = {"complete":"已保存", "discarded":"已标记丢弃（原始记录仍保留）", "incomplete":"未完成，原始记录已保留"}[status] + "：" + reason

    def status(self):
        now = time.monotonic()
        capture_state = ('idle' if self.episode is None else 'review' if self.reviewing else 'input_paused' if self.awaiting_input_resume else 'recording' if self.gate.current is not None
                         else 'paused' if self.gate.intervals else 'waiting')
        return dict(type="recorder_status", pid=os.getpid(), monotonic_s=now,
            recording=self.episode is not None, root=str(self.root), message=self.last_message,
            capture_state=capture_state, trigger_pressed=(self.latest.get('hand') or {}).get('trigger_pressed') if self.latest else None,
            last_input_pause=self.input_pause,
            can_review=self.episode is not None,
            recorded_s=0 if self.gate is None else self.gate.duration(now),
            capture_segments=0 if self.gate is None else len(self.gate.intervals),
            ready_error=self.ready_error(), completed=self.completed, last_episode=self.last_path,
            episode=None if self.episode is None else self.episode.id,
            elapsed_s=0 if self.episode is None else now-self.episode.metadata["started_monotonic_s"],
            samples=0 if self.episode is None else self.episode.samples, telemetry_gaps=self.sequence_gaps,
            images={} if self.episode is None else self.episode.images,
            cameras={c.config["name"]: dict(c.state) for c in self.cameras}, preview_dir=str(self.preview_dir),
            control_hz=None if not self.latest else self.latest.get("control_hz"))

    @staticmethod
    def input_fault(item, previous):
        current = item.get('input_status') or {}
        old = (previous or {}).get('input_status') or {}
        # Counters retain a short invalid frame even if the newest pose is valid.
        edges = {k for k in ('invalid_count', 'pause_generation')
                 if isinstance(current.get(k), int) and isinstance(old.get(k), int)
                 and current[k] > old[k]}
        if (not item.get('source_valid') or current.get('valid') is False
                or current.get('motion_enabled') is False or edges):
            reason = current.get('reason') if current.get('valid') is False else None
            if not reason and 'invalid_count' in edges:
                reason = current.get('last_invalid_reason')
            return reason or item.get('status') or '手柄输入无效/过期'
        return None

    def pause_for_input(self, item, reason):
        self.gate.interrupt_input()
        if not self.awaiting_input_resume:
            self.input_pause = dict(start_s=item['monotonic_s'], reason=reason, resumed_s=None)
            self.episode.metadata.setdefault('input_interruptions', []).append(self.input_pause)
            self.episode.event('input_interruption', dict(self.input_pause,
                input_status=item.get('input_status'), controller_status=item.get('status'),
                source_valid=item.get('source_valid'), source_received_s=item.get('source_received_s'),
                hand=item.get('hand')))
        self.awaiting_input_resume = True
        self.last_message = '输入中断，已保留本条：' + self.input_pause['reason'] + '；恢复后松开再按住扳机继续'

    def ingest_sample(self, item):
        previous_sample = self.latest
        self.latest = item
        if not self.episode or self.reviewing or item['monotonic_s'] < self.episode.metadata['started_monotonic_s']:
            return
        reason = None
        if item['run_id'] != self.episode.metadata['run_id']:
            reason = '控制器会话发生变化'
        elif item['settings_revision'] != self.episode.metadata['settings_revision']:
            reason = '录制中遥操参数发生变化'
        elif item.get('recovery_active'):
            reason = '机械臂进入复位'
        elif item.get('error'):
            reason = '控制器故障：' + str(item['error'])
        elif item.get('mode') != 'motion':
            reason = '遥操许可已撤销：' + str(item.get('status', '已停止遥操'))
        if reason:
            self.episode.event('control_discontinuity',item)
            self.finish('incomplete',reason=reason)
            return
        fault = self.input_fault(item, previous_sample)
        if fault:
            self.pause_for_input(item, fault)
            return
        previous = None if self.gate.current is None else self.gate.current['segment_id']
        segment = self.gate.update(item)
        if segment != previous:
            self.episode.event('capture_gate',dict(segment_id=segment,t=item['monotonic_s'],
                intervals=self.gate.intervals))
            self.last_message = '按住扳机 · 正在记录' if segment is not None else '扳机已松开/跟随已暂停 · 本条暂停记录，可继续或保存'
        if segment is not None:
            if self.awaiting_input_resume:
                self.input_pause['resumed_s'] = item['monotonic_s']
                self.episode.event('input_resumed', dict(self.input_pause, segment_id=segment))
                self.awaiting_input_resume = False
                self.last_message = '已重新接管，继续本条；上次输入中断：' + self.input_pause['reason']
            if segment == self.last_sample_segment and self.last_sequence is not None:
                self.sequence_gaps += max(0,item['sequence']-self.last_sequence-1)
            self.last_sequence = item['sequence']
            self.last_sample_segment = segment
            self.episode.add_sample(dict(item,capture_segment=segment))

    def drain_images(self, final=False):
        """Classify by acquisition time, not by the gate state when encoding ends.

        A frame newer than the controller watermark waits for a later sample;
        delayed frames from a paused interval are never written into a resumed one.
        """
        while True:
            try:
                self.pending_images.append(self.camera_queue.get_nowait())
            except queue.Empty:
                break
        watermark = (self.latest or {}).get('monotonic_s',0)
        pending = deque()
        for image in self.pending_images:
            if not self.episode or image['t'] < self.episode.metadata['started_monotonic_s']:
                continue
            if image['t'] > watermark:
                if not final:
                    pending.append(image)
                continue
            segment = self.gate.segment_at(image['t'])
            if segment is not None:
                key = (segment,image['camera'])
                previous = self.camera_last_sequences.get(key)
                if previous is not None:
                    self.camera_gaps += max(0,image['sequence']-previous-1)
                self.camera_last_sequences[key] = image['sequence']
                self.episode.add_image(image)
        self.pending_images = pending

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        with (self.socket_path.with_suffix(".lock")).open("a+") as lock, (self.root/".recorder.lock").open("a+") as dataset_lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(dataset_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            manifest = self.root/"dataset.json"
            if manifest.exists():
                if json.loads(manifest.read_text()).get("schema_version") != RAW_VERSION:
                    raise ValueError("目录中已有不同格式的数据集")
            else:
                atomic_json(manifest, dict(schema_version=RAW_VERSION,created_utc_s=time.time(),robot_type="xarm7_g2"))
            if self.socket_path.exists():
                if not self.socket_path.is_socket():
                    raise ValueError("录制接口路径不是 socket")
                self.socket_path.unlink()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.bind(str(self.socket_path))
            os.chmod(self.socket_path,0o600)
            sock.setblocking(False)
            self.cameras = [CameraWorker(c,self.camera_queue,self.stop,self.preview_dir) for c in self.config["cameras"]]
            threading.Thread(target=self.command_reader,daemon=True).start()
            next_status = 0
            try:
                while not self.shutdown_requested:
                    # Bounded work per iteration keeps command/status handling responsive.
                    for _ in range(256):
                        try:
                            raw = sock.recv(131072)
                        except BlockingIOError:
                            break
                        item = json.loads(raw)
                        if item.get("stream_version") != STREAM_VERSION:
                            continue
                        self.ingest_sample(item)
                    self.drain_images()
                    if self.episode:
                        self.episode.flush()
                        now=time.monotonic()
                        if not self.reviewing and (not self.latest or now-self.latest["monotonic_s"] > .5):
                            self.finish("incomplete",reason="机器人数据超时")
                        elif not self.reviewing and any(c.state.get("error") or now-c.state.get("last_frame_s",now)>.5 for c in self.cameras):
                            self.finish("incomplete",reason="相机数据中断")
                    try:
                        request = self.commands.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        try:
                            self.handle(request)
                        except (ValueError,OSError) as error:
                            self.last_message = str(error)
                    if time.monotonic() >= next_status:
                        if self.episode and shutil.disk_usage(self.root).free < 1024**3:
                            self.finish("incomplete",reason="可用磁盘不足 1 GiB")
                        status=self.status()
                        atomic_json(self.status_path,status)
                        print(json_text(status),flush=True)
                        next_status=time.monotonic()+.25
                    time.sleep(.003)
            finally:
                self.stop.set()
                self.finish("incomplete",reason="录制进程退出/界面关闭")
                for camera in self.cameras:
                    camera.thread.join(timeout=1)
                sock.close()
                self.socket_path.unlink(missing_ok=True)
                status=self.status()
                status.update(stopped=True)
                atomic_json(self.status_path,status)


def code_hashes():
    import hashlib
    names=("quest_recorder.py","recording_data.py","recording_gate.py","quest_recording_stream.py",
           "xarm_quest_relative.py","quest_gripper_control.py","quest_pose_math.py",
           "quest_pose_filter.py","quest_pose_stream.py","quest_relative_motion.py",
           "quest_unitree_quality.js","quest_unitree_realtime.js","quest_unitree_bridge.py","quest_unitree_input.py")
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--status",type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    names=[c["name"] for c in config["cameras"]]
    if len(set(names))!=len(names) or any(not n.replace('_','').isalnum() for n in names):
        parser.error("相机名称必须唯一且仅包含字母数字和下划线")
    recorder=Recorder(args.root,config,status_path=args.status)
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_: setattr(recorder,"shutdown_requested",True))
    recorder.run()


if __name__=="__main__":
    main()
