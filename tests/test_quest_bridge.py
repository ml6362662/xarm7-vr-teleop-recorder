import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_quest_packet_maps_to_lerobot_frame():
    bridge = load_module("quest_usb_bridge", "quest_usb_bridge.py")
    packet = (
        b"absolute|-0.1958634,0.5487109,0.1997189|0.9726847,0.1625274,-0.05004814,0.1580023"
        b"|False|-0.2003626,0.5154874,0.2939312|-0.1013651,0.5778831,0.2145192"
        b"|-0.2282644,0.6384059,0.229802"
    )
    frame, pose, trigger = bridge.parse_controller_packet(packet)
    assert len(frame) == 4
    assert all(len(row) == 3 for row in frame)
    assert frame[0] == pose[:3]
    assert len(pose) == 7
    assert trigger is False


def test_invalid_tracking_frame_is_rejected():
    bridge = load_module("quest_usb_bridge", "quest_usb_bridge.py")
    assert bridge.parse_controller_packet(b"absolute|0,0,0|1,0,0,0|False|0,0,0|0,0,0|0,0,0") is None


def test_invalid_trigger_never_silently_becomes_gripper_release():
    bridge = load_module("quest_usb_bridge", "quest_usb_bridge.py")
    assert bridge.parse_controller_packet(b"absolute|1,2,3|1,0,0,0|bad|1.1,2,3|1,2.1,3|1,2,2.9") is None


def test_monitor_uses_quaternion_for_orientation():
    monitor = load_module("quest_controller_monitor", "quest_controller_monitor.py")
    position, euler = monitor.decode_pose([1, 2, 3, 1, 0, 0, 0])
    assert position == [1.0, 2.0, 3.0]
    assert euler == [0.0, 0.0, 0.0]
