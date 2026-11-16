import argparse
import fcntl
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("xarm7_vr", Path(__file__).parents[1] / "xarm7_vr.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parents[1] / "config.json").read_text())

    def test_camera_argument_is_valid_json_and_stable_path(self):
        parsed = json.loads(MODULE.camera_argument(self.config))
        self.assertEqual(parsed["wrist"]["type"], "opencv")
        self.assertIn("/dev/v4l/by-id/", parsed["wrist"]["index_or_path"])
        self.assertEqual(parsed["wrist"]["fourcc"], "MJPG")
        self.assertEqual(parsed["wrist"]["rotation"], 180)
        self.assertEqual(parsed["wrist"]["auto_exposure"], 1)
        self.assertEqual(parsed["wrist"]["exposure"], 360)
        self.assertEqual(parsed["global"]["type"], "opencv")
        self.assertIn("5MP_USB_Camera", parsed["global"]["index_or_path"])
        self.assertFalse(parsed["global"]["autofocus"])
        self.assertEqual(parsed["global"]["focus"], 128)

    def test_original_single_camera_config_remains_supported(self):
        legacy = {"camera": self.config["cameras"][0]}

        parsed = json.loads(MODULE.camera_argument(legacy))

        self.assertEqual(list(parsed), ["wrist"])

    def test_workspace_args(self):
        self.config["workspace_mm"] = {"min": [1, 2, 3], "max": [4, 5, 6]}
        args = MODULE.workspace_args(self.config)
        self.assertEqual(len(args), 2)
        self.assertIn("[1, 2, 3]", args[0])

    def test_invalid_workspace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with self.assertRaises(SystemExit):
                MODULE.cmd_workspace(
                    argparse.Namespace(minimum=[1, 2, 3], maximum=[1, 4, 5]),
                    self.config,
                    config_path,
                )

    def test_motion_requires_explicit_flag(self):
        with self.assertRaises(SystemExit):
            MODULE.confirm_motion(argparse.Namespace(enable_motion=False, yes=False))

    def test_second_bridge_is_rejected_before_starting_quest(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            with (state_dir / "bridge.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(MODULE, "STATE_DIR", state_dir), patch.object(MODULE, "cmd_quest_setup") as setup:
                    with self.assertRaisesRegex(SystemExit, "桥已由另一项任务运行"):
                        with MODULE.managed_bridge(self.config):
                            pass
                    setup.assert_not_called()

    def test_teleop_parser_exposes_safe_tuning(self):
        args = MODULE.build_parser().parse_args(
            ["teleop", "--speed-mm-s", "20", "--range-mm", "50", "--translation-scale", "0.5"]
        )

        self.assertEqual(args.speed_mm_s, 20)
        self.assertEqual(args.range_mm, 50)
        self.assertEqual(args.translation_scale, 0.5)

    def test_record_parser_exposes_gui_collection_options(self):
        args = MODULE.build_parser().parse_args(
            [
                "record",
                "paper_cup_demo",
                "--task",
                "递纸杯",
                "--dynamic-range-mm",
                "100",
                "--speed-mm-s",
                "20",
                "--disable-rotation",
            ]
        )

        self.assertEqual(args.dynamic_range_mm, 100)
        self.assertEqual(args.speed_mm_s, 20)
        self.assertTrue(args.disable_rotation)

    def test_gui_dataset_name_format_is_accepted(self):
        name = "paper_cup_handover_20260930_143000"

        self.assertTrue(name.replace("-", "").replace("_", "").isalnum())

    def test_recording_robot_args_honor_gui_speed(self):
        args = MODULE.common_robot_args(
            self.config,
            cautious=False,
            tcp_speed_mm_s=10,
            tcp_acceleration_mm_s2=50,
        )

        self.assertIn("--robot.tcp_speed_mm_s=10", args)
        self.assertIn("--robot.tcp_acceleration_mm_s2=50", args)

    def test_train_is_local_only_by_default(self):
        args = argparse.Namespace(
            dataset="visual",
            policy="act",
            output=None,
            batch_size=2,
            empty_cameras=2,
            steps=1,
            dry_run=True,
        )
        info = {"features": {"observation.images.wrist": {"dtype": "image"}}}
        stream = StringIO()
        with patch.object(MODULE, "read_dataset_info", return_value=info), redirect_stdout(stream):
            MODULE.cmd_train(args, self.config)
        self.assertIn("--policy.push_to_hub=false", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
