#!/usr/bin/env python3
"""Publish a read-only view of the Open-Teach Quest controller stream."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation


TOPICS = {
    "pose": (8089, "quest_controller_pose"),
    "pause": (8102, "pause"),
    "trigger": (8127, "gripper_msg"),
}


def decode_pose(value: object) -> tuple[list[float], list[float]] | None:
    """Convert the Quest's XYZ + wxyz quaternion to XYZ + Euler angles."""
    try:
        pose = np.asarray(value, dtype=float).reshape(7)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(pose).all() or np.linalg.norm(pose[:3]) <= 1e-4 or np.linalg.norm(pose[3:]) <= 0.5:
        return None
    quaternion_xyzw = pose[[4, 5, 6, 3]]
    return pose[:3].round(5).tolist(), Rotation.from_quat(quaternion_xyzw).as_euler("xyz", degrees=True).round(2).tolist()


def monitor(output: Path) -> None:
    context = zmq.Context()
    poller = zmq.Poller()
    sockets = {}
    state: dict[str, object] = {"frame_timestamp": None, "pause_timestamp": None, "trigger_timestamp": None}
    try:
        for name, (port, topic) in TOPICS.items():
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
            socket.connect(f"tcp://127.0.0.1:{port}")
            poller.register(socket, zmq.POLLIN)
            sockets[socket] = (name, topic)
        output.parent.mkdir(parents=True, exist_ok=True)
        while True:
            changed = False
            for socket in dict(poller.poll(timeout=100)):
                name, topic = sockets[socket]
                message = socket.recv()
                prefix = f"{topic} ".encode()
                if not message.startswith(prefix):
                    continue
                value = pickle.loads(message[len(prefix) :])  # noqa: S301 - trusted local bridge stream
                now = time.time()
                if name == "pose":
                    pose = decode_pose(value)
                    if pose is None:
                        continue
                    state.update(position_m=pose[0], euler_xyz_deg=pose[1], frame_timestamp=now)
                elif name == "pause":
                    state.update(motion_enabled=bool(np.asarray(value).reshape(-1)[0]), pause_timestamp=now)
                else:
                    state.update(trigger_pressed=bool(np.asarray(value).reshape(-1)[0]), trigger_timestamp=now)
                changed = True
            if changed:
                temporary = output.with_suffix(output.suffix + ".tmp")
                temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, output)
    finally:
        for socket in sockets:
            socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    monitor(parser.parse_args().output)
