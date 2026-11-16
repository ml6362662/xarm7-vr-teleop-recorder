#!/usr/bin/env python3
"""Capture a short real-camera episode without connecting to the robot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("camera")
    args = parser.parse_args()

    camera = OpenCVCamera(
        OpenCVCameraConfig(
            index_or_path=args.camera,
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
            warmup_s=1,
        )
    )
    dataset = LeRobotDataset.create(
        repo_id="local/xarm7_camera_smoke",
        fps=30,
        robot_type="xarm7",
        root=args.root,
        use_videos=True,
        image_writer_threads=2,
        features={
            "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
            "action": {"dtype": "float32", "shape": (7,), "names": None},
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
        },
    )
    camera.connect()
    try:
        next_frame = time.monotonic()
        for _ in range(30):
            dataset.add_frame(
                {
                    "observation.state": np.zeros(14, dtype=np.float32),
                    "action": np.zeros(7, dtype=np.float32),
                    "observation.images.wrist": camera.read_latest(),
                    "task": "camera smoke test",
                }
            )
            next_frame += 1 / 30
            time.sleep(max(0, next_frame - time.monotonic()))
        dataset.save_episode()
        dataset.finalize()
    finally:
        camera.disconnect()
    print(args.root)


if __name__ == "__main__":
    main()
