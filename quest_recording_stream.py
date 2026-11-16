"""Best-effort recording feed. Never waits for a recorder or sends robot commands."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import socket
import threading

ROOT = Path(__file__).resolve().parent
RECORD_SOCKET = ROOT / ".state" / "recorder_samples.sock"
STREAM_VERSION = "xarm7.telemetry.v1"


class RecordingPublisher:
    def __init__(self, path=RECORD_SOCKET):
        self.path = str(path)
        self.queue = queue.Queue(maxsize=64)
        self.dropped = 0
        self.sequence = 0
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, name="record-feed", daemon=True)
        self.thread.start()

    def publish(self, sample):
        sample.update(stream_version=STREAM_VERSION, sequence=self.sequence,
                      publisher_dropped=self.dropped)
        self.sequence += 1
        try:
            self.queue.put_nowait(sample)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.setblocking(False)
            while not self.closed.is_set():
                try:
                    sample = self.queue.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    sock.sendto(json.dumps(sample, allow_nan=False, separators=(",", ":")).encode(), self.path)
                except (OSError, ValueError):
                    self.dropped += 1

    def close(self):
        self.closed.set()
        self.thread.join(timeout=.2)


def recording_sample(controller):
    """Own immutable numerical copies, including only committed servo commands.

    read timestamps are host SDK read intervals, NOT hardware exposure/report times.
    Paused frames have command_valid=False; exporters must select a hold policy.
    """
    c = controller
    now = c.clock()
    source = c.source or {}
    return dict(
        run_id=c.run_id, monotonic_s=now,
        robot_info=dict(getattr(c,"recording_robot_info",{})),
        observation_read_start_s=getattr(c, "observation_read_start_s", now),
        observation_read_end_s=getattr(c, "observation_read_end_s", now),
        joints_rad=c.joints.tolist(), tcp_position_m=(c.tcp[:3] / 1000).tolist(),
        tcp_quaternion_wxyz=c.tcp[3:].tolist(),
        command_joints_rad=None if c.last_command_joints is None else c.last_command_joints.tolist(),
        command_tcp_position_m=(c.command_tcp[:3] / 1000).tolist(),
        command_tcp_quaternion_wxyz=c.command_tcp[3:].tolist(),
        command_time_s=getattr(c, "last_command_time_s", None),
        command_valid=c.servo_started and c.last_command_joints is not None,
        command_count=c.command_count, hold_count=c.hold_count,
        requested_tcp_position_m=(c.requested[:3] / 1000).tolist(),
        requested_tcp_quaternion_wxyz=c.requested[3:].tolist(),
        target_tcp_position_m=(c.bounded[:3] / 1000).tolist(),
        target_tcp_quaternion_wxyz=c.bounded[3:].tolist(),
        gripper=None if c.gripper is None else c.gripper.snapshot(),
        mode=c.mode, active=c.servo_started, pause_settling=c.pause_settle is not None,
        recovery_active=c.recovery.motion is not None, source_valid=c.fresh(),
        error=c.last_error, status=c.status, settings_revision=c.settings_revision,
        input_status={key: source.get(key) for key in (
            "valid", "reason", "motion_enabled", "invalid_count", "pause_generation",
            "packet_count", "frame_id", "sample_age_s", "packet_age_s",
            "last_invalid_reason", "last_invalid_monotonic_s")},
        settings=dict(c.settings), control_hz=c.control_hz,
        input_filter=dict(c.input_filter.snapshot(), enabled=c.settings['input_filter']['enabled'],
                          valid=c.fresh() and c.input_filter.sample_time is not None),
        hand={key: source.get(key) for key in (
            "position_m", "quaternion_wxyz", "trigger_pressed", "a_pressed", "b_pressed",
            "sample_timestamp", "sample_monotonic_s", "sample_headset_s", "sample_time_source", "valid_age_s", "frame_id", "input_backend",
            "trigger_release_generation", "b_release_generation")},
        source_received_s=c.source_received,
    )
