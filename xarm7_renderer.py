#!/usr/bin/env python3
"""Render the official XARM7 + G2 meshes from live controller telemetry."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw


ROOT = Path("/home/a/xarm7-vr-lab")
URDF = ROOT / "assets/xarm7_g2_visual.urdf"
TELEMETRY = ROOT / ".state/control_telemetry.json"
OUTPUT = ROOT / ".state/xarm7_render.png"
WIDTH = 440
HEIGHT = 270


def load_telemetry() -> tuple[list[float], float, float]:
    payload = json.loads(TELEMETRY.read_text(encoding="utf-8"))
    degrees = payload.get("current_deg")
    if not isinstance(degrees, list) or len(degrees) != 7:
        raise ValueError("telemetry current_deg must contain seven values")
    gripper = float(payload.get("gripper", 1.0))
    timestamp = float(payload.get("timestamp", 0.0))
    return [math.radians(float(value)) for value in degrees], gripper, timestamp


def joint_indices(body: int) -> dict[str, int]:
    return {
        p.getJointInfo(body, index)[1].decode("utf-8"): index
        for index in range(p.getNumJoints(body))
    }


def apply_xarm_colors(body: int, indices: dict[str, int]) -> None:
    """Use high-contrast colors close to the physical white/silver XARM7."""
    p.changeVisualShape(body, -1, rgbaColor=[0.82, 0.85, 0.89, 1.0])
    for number in range(1, 8):
        shade = 0.96 if number % 2 else 0.82
        p.changeVisualShape(
            body,
            indices[f"joint{number}"],
            rgbaColor=[shade, shade + 0.01, min(1.0, shade + 0.025), 1.0],
        )
    for name in (
        "gripper_fix",
        "drive_joint",
        "left_finger_joint",
        "left_inner_joint",
        "right_outer_joint",
        "right_finger_joint",
        "right_inner_joint",
    ):
        p.changeVisualShape(body, indices[name], rgbaColor=[0.28, 0.31, 0.35, 1.0])


def render(body: int, indices: dict[str, int], joints: list[float], gripper: float, timestamp: float) -> None:
    for number, value in enumerate(joints, start=1):
        p.resetJointState(body, indices[f"joint{number}"], value)
    grip_angle = max(0.0, min(1.0, gripper)) * 0.85
    for name in (
        "drive_joint",
        "left_finger_joint",
        "left_inner_joint",
        "right_outer_joint",
        "right_finger_joint",
        "right_inner_joint",
    ):
        p.resetJointState(body, indices[name], grip_angle)

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.05, 0.0, 0.38],
        distance=1.35,
        yaw=132,
        pitch=-24,
        roll=0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(fov=42, aspect=WIDTH / HEIGHT, nearVal=0.05, farVal=4.0)
    _, _, rgba, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=[-0.4, -0.7, -1.0],
        lightColor=[1.0, 1.0, 1.0],
        lightDistance=2.0,
        shadow=1,
    )
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").convert("RGB")
    draw = ImageDraw.Draw(image)
    stale = time.time() - timestamp >= 0.6
    color = (165, 55, 55) if stale else (25, 145, 75)
    draw.rounded_rectangle((10, 10, 104, 34), radius=8, fill=(250, 250, 250), outline=color, width=2)
    draw.text((21, 16), "STALE" if stale else "LIVE XARM7", fill=color)
    temporary = OUTPUT.with_suffix(".png.tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, OUTPUT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    client = p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf", basePosition=[0, 0, -0.012], useFixedBase=True)
    body = p.loadURDF(str(URDF), useFixedBase=True, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    indices = joint_indices(body)
    apply_xarm_colors(body, indices)
    last_signature: tuple[float, ...] | None = None
    try:
        while True:
            try:
                joints, gripper, timestamp = load_telemetry()
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                joints, gripper, timestamp = [0.0] * 7, 1.0, 0.0
            signature = tuple(round(value, 5) for value in (*joints, gripper, timestamp))
            # Refresh stale/live badge even when the arm is stationary.
            if signature != last_signature or int(time.time() * 2) % 2 == 0:
                render(body, indices, joints, gripper, timestamp)
                last_signature = signature
            if args.once:
                return 0
            time.sleep(0.08)
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    raise SystemExit(main())
