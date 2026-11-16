#!/usr/bin/env python3
"""Create a tiny visual LeRobotDataset for an end-to-end training smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"state_{index}" for index in range(14)],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"action_{index}" for index in range(7)],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (96, 96, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/xarm7_visual_smoke",
        fps=30,
        features=features,
        root=args.root,
        robot_type="xarm7",
        use_videos=False,
        image_writer_threads=2,
    )
    yy, xx = np.mgrid[:96, :96]
    for episode in range(2):
        for frame_index in range(16):
            phase = (episode * 16 + frame_index) / 31.0
            image = np.empty((96, 96, 3), dtype=np.uint8)
            image[..., 0] = (xx + frame_index * 4) % 256
            image[..., 1] = (yy + episode * 80) % 256
            image[..., 2] = round(phase * 255)
            dataset.add_frame(
                {
                    "observation.state": np.linspace(0, phase, 14, dtype=np.float32),
                    "action": np.linspace(phase, phase + 0.1, 7, dtype=np.float32),
                    "observation.images.wrist": image,
                    "task": "move the test object",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    print(args.root)


if __name__ == "__main__":
    main()
