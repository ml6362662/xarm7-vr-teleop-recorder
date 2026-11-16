#!/usr/bin/env python3
"""Bridge the Quest controller's USB ZMQ packets to LeRobot's local topics."""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import zmq


def parse_controller_packet(message: bytes) -> tuple[list[list[float]], list[float], bool] | None:
    try:
        parts = message.decode("utf-8").strip().split("|")
        if len(parts) not in (7, 8):
            return None
        if len(parts) == 8 and parts[7].strip().lower() not in ("true", "false"):
            return None
        if parts[3].strip().lower() not in ("true", "false"):
            return None
        position = np.asarray([float(value) for value in parts[1].split(",")], dtype=float)
        quaternion = np.asarray([float(value) for value in parts[2].split(",")], dtype=float)
        offsets = [np.asarray([float(value) for value in part.split(",")], dtype=float) for part in parts[4:7]]
        if position.shape != (3,) or quaternion.shape != (4,) or any(value.shape != (3,) for value in offsets):
            return None
        axes = [offsets[0] - position, offsets[1] - position, position - offsets[2]]
        if any(np.linalg.norm(axis) < 1e-6 for axis in axes):
            return None
        axes = [axis / np.linalg.norm(axis) for axis in axes]
        rotation = np.stack(axes, axis=1)
        if not (
            np.isfinite(position).all()
            and np.isfinite(quaternion).all()
            and np.linalg.norm(position) > 1e-4
            and np.linalg.norm(quaternion) > 0.5
            and np.isfinite(np.linalg.norm(quaternion))
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=0.25)
            and abs(np.linalg.det(rotation)) > 0.5
        ):
            return None
        frame = [position.tolist(), *(axis.tolist() for axis in axes)]
        pose = [*position.tolist(), *quaternion.tolist()]
        return frame, pose, parts[3].strip().lower() == "true"
    except (UnicodeDecodeError, ValueError, OverflowError):
        return None


def publish(socket: zmq.Socket, topic: str, value: object) -> None:
    socket.send(topic.encode("ascii") + b" " + pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def run(host: str) -> None:
    context = zmq.Context()
    sockets: dict[str, zmq.Socket] = {}
    poller = zmq.Poller()
    try:
        for name, port in (("pause_in", 8100), ("pose_in", 8125)):
            socket = context.socket(zmq.PULL)
            socket.setsockopt(zmq.RCVHWM, 30)
            socket.bind(f"tcp://{host}:{port}")
            poller.register(socket, zmq.POLLIN)
            sockets[name] = socket
        for name, port in (("frame_out", 8089), ("pause_out", 8102), ("gripper_out", 8127)):
            socket = context.socket(zmq.PUB)
            socket.bind(f"tcp://{host}:{port}")
            sockets[name] = socket
        print("Quest USB bridge ready: 8100/8125 → 8089/8102/8127", flush=True)
        frame_count = 0
        invalid_count = 0
        invalid_sample = None
        pause_count = 0
        last_report = time.monotonic()
        while True:
            events = dict(poller.poll(timeout=100))
            if sockets["pause_in"] in events:
                message = sockets["pause_in"].recv()
                if message in (b"Low", b"High"):
                    publish(sockets["pause_out"], "pause", int(message == b"High"))
                    pause_count += 1
            if sockets["pose_in"] in events:
                message = sockets["pose_in"].recv()
                parsed = parse_controller_packet(message)
                if parsed is not None:
                    frame, pose, trigger = parsed
                    publish(sockets["frame_out"], "transformed_hand_frame", frame)
                    publish(sockets["frame_out"], "quest_controller_pose", pose)
                    publish(sockets["gripper_out"], "gripper_msg", int(trigger))
                    frame_count += 1
                else:
                    invalid_count += 1
                    if invalid_sample is None:
                        invalid_sample = repr(message[:220])
            if time.monotonic() - last_report >= 5.0:
                print(f"Quest packets / 5s: pose={frame_count}, invalid={invalid_count}, pause={pause_count}", flush=True)
                if invalid_sample is not None:
                    print(f"First invalid Quest packet: {invalid_sample}", flush=True)
                frame_count = invalid_count = pause_count = 0
                invalid_sample = None
                last_report = time.monotonic()
    finally:
        for socket in sockets.values():
            socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    run(parser.parse_args().host)
