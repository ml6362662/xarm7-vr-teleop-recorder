#!/usr/bin/env python3
"""Safe, single-command launcher for the local XARM7 + Quest + LeRobot stack."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
STATE_DIR = ROOT / ".state"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = ("robot_ip", "lerobot_root", "openteach_root", "openteach_python", "dataset_root")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise SystemExit(f"配置缺少字段：{', '.join(missing)}")
    return config


def save_config(path: Path, config: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_text(command: Sequence[str]) -> str:
    import shlex

    return shlex.join([str(item) for item in command])


def run(command: Sequence[str], *, cwd: Path | None = None, env: dict | None = None, check: bool = True) -> int:
    print(f"+ {command_text(command)}", flush=True)
    completed = subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=False)
    if check and completed.returncode:
        raise SystemExit(completed.returncode)
    return completed.returncode


def output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        [str(item) for item in command], cwd=cwd, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(detail or f"命令失败：{command_text(command)}")
    return completed.stdout.strip()


def paths(config: dict) -> tuple[Path, Path, Path]:
    lerobot = Path(config["lerobot_root"])
    openteach = Path(config["openteach_root"])
    lerobot_python = lerobot / ".venv/bin/python"
    for path in (lerobot, openteach, lerobot_python, Path(config["openteach_python"])):
        if not path.exists():
            raise SystemExit(f"所需路径不存在：{path}")
    return lerobot, openteach, lerobot_python


def configured_cameras(config: dict) -> list[dict]:
    """Return all configured cameras while accepting the original one-camera config."""
    cameras = config.get("cameras")
    if cameras is None and "camera" in config:
        cameras = [config["camera"]]
    if not cameras:
        raise SystemExit("配置中至少需要一台相机")
    names = [camera["name"] for camera in cameras]
    if len(names) != len(set(names)):
        raise SystemExit("相机名称不能重复")
    return cameras


def camera_argument(config: dict) -> str:
    result = {}
    for camera in configured_cameras(config):
        entry = {
            "type": "opencv",
            "index_or_path": camera["path"],
            "width": camera["width"],
            "height": camera["height"],
            "fps": camera["fps"],
            "fourcc": camera["fourcc"],
        }
        if "rotation" in camera:
            entry["rotation"] = camera["rotation"]
        for control in ("auto_exposure", "exposure", "autofocus", "focus"):
            if control in camera:
                entry[control] = camera[control]
        result[camera["name"]] = entry
    return json.dumps(result, separators=(",", ":"))


def workspace_args(config: dict) -> list[str]:
    workspace = config.get("workspace_mm")
    if not workspace:
        return []
    return [
        f"--robot.workspace_min_mm={json.dumps(workspace['min'])}",
        f"--robot.workspace_max_mm={json.dumps(workspace['max'])}",
    ]


def require_workspace(config: dict) -> None:
    if not config.get("workspace_mm"):
        raise SystemExit(
            "尚未配置任务工作空间。请先测量 xArm 基座坐标系下安全 XYZ 范围，再运行：\n"
            "  xarm7-vr workspace --min X Y Z --max X Y Z"
        )


def confirm_motion(args: argparse.Namespace) -> None:
    if not getattr(args, "enable_motion", False):
        raise SystemExit("拒绝使能机械臂：请检查现场安全后显式加 --enable-motion")
    if getattr(args, "yes", False):
        return
    print("\n即将使能真实 XARM7。清空工作区、降低控制柜速度，并让急停保持可触达。")
    answer = input("请输入 MOVE XARM7 继续：").strip()
    if answer != "MOVE XARM7":
        raise SystemExit("已取消；机械臂未使能。")


def preflight_command(config: dict, *, skip_quest: bool = False) -> list[str]:
    lerobot, _, python = paths(config)
    command = [
        str(python),
        str(lerobot / "scripts/xarm7_quest_preflight.py"),
        "--robot-ip",
        config["robot_ip"],
    ]
    workspace = config.get("workspace_mm")
    if workspace:
        command += [
            "--workspace-min-mm",
            json.dumps(workspace["min"]),
            "--workspace-max-mm",
            json.dumps(workspace["max"]),
        ]
    if skip_quest:
        command.append("--skip-quest")
    return command


def check_camera(camera: dict, lerobot_python: Path) -> int:
    script = """
import cv2, json, sys
p, width, height, fps, fourcc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
c = cv2.VideoCapture(p)
c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
c.set(cv2.CAP_PROP_FRAME_WIDTH, width); c.set(cv2.CAP_PROP_FRAME_HEIGHT, height); c.set(cv2.CAP_PROP_FPS, fps)
ok, frame = c.read()
actual = {"opened": c.isOpened(), "frame_ok": ok, "shape": None if frame is None else list(frame.shape), "fps": c.get(cv2.CAP_PROP_FPS)}
c.release(); print(json.dumps(actual)); raise SystemExit(0 if ok else 1)
"""
    print(f"[{camera['name']}] {camera['path']}")
    return run(
        [
            str(lerobot_python),
            "-c",
            script,
            camera["path"],
            str(camera["width"]),
            str(camera["height"]),
            str(camera["fps"]),
            camera["fourcc"],
        ],
        check=False,
    )


def cmd_status(args: argparse.Namespace, config: dict) -> int:
    lerobot, _, lerobot_python = paths(config)
    failures = 0
    print("=== XARM7（只读，不发送运动命令） ===")
    failures += run(preflight_command(config, skip_quest=True), cwd=lerobot, check=False) != 0
    print("\n=== USB 相机 ===")
    for camera in configured_cameras(config):
        failures += check_camera(camera, lerobot_python) != 0
    print("\n=== Quest / ADB ===")
    adb = shutil.which("adb") or str(Path.home() / ".local/bin/adb")
    adb_result = subprocess.run([adb, "devices", "-l"], text=True, capture_output=True, check=False)
    print(adb_result.stdout, end="")
    if adb_result.stderr:
        print(adb_result.stderr, end="", file=sys.stderr)
    authorized = [
        line for line in adb_result.stdout.splitlines()[1:] if "\tdevice" in line or " device " in line
    ]
    if adb_result.returncode or len(authorized) != 1:
        failures += 1
        print("Quest 尚未连接并授权（接入前这是正常状态）。")
    print("\n=== NVIDIA GPU ===")
    failures += run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], check=False) != 0
    training_check = (
        "import torch,transformers,accelerate; "
        "assert torch.cuda.is_available(), 'PyTorch CUDA unavailable'; "
        "print('PyTorch',torch.__version__,'| Transformers',transformers.__version__,"
        "'| CUDA',torch.cuda.get_device_name(0))"
    )
    failures += run([str(lerobot_python), "-c", training_check], check=False) != 0
    if config.get("workspace_mm"):
        print(f"\n任务工作空间（mm）：{config['workspace_mm']}")
    else:
        print("\n注意：尚未配置任务工作空间；正式采集和模型推理将被拒绝。")
    print("\n状态：" + ("存在未就绪项" if failures else "已就绪"))
    return int(bool(failures))


def cmd_workspace(args: argparse.Namespace, config: dict, config_path: Path) -> int:
    low, high = list(args.minimum), list(args.maximum)
    if any(a >= b for a, b in zip(low, high, strict=True)):
        raise SystemExit("每个最小值必须严格小于对应最大值")
    config["workspace_mm"] = {"min": low, "max": high}
    save_config(config_path, config)
    print(f"已保存工作空间：min={low}, max={high} mm")
    return 0


def cmd_quest_install(args: argparse.Namespace, config: dict) -> int:
    _, openteach, _ = paths(config)
    apk = openteach / "VR/APK/FrankaRemoteTrackingV2.apk"
    if not apk.exists():
        raise SystemExit(f"APK 不存在：{apk}")
    if not args.yes and input("这会在已连接的 Quest 上安装/更新控制器应用。输入 INSTALL 继续：").strip() != "INSTALL":
        raise SystemExit("已取消。")
    return run(["adb", "install", "-r", str(apk)])


def cmd_quest_setup(args: argparse.Namespace, config: dict) -> int:
    _, openteach, _ = paths(config)
    return run([config["openteach_python"], "tools/setup_quest_usb.py"], cwd=openteach)


def bridge_command(config: dict) -> list[str]:
    return [
        config["openteach_python"],
        str(ROOT / "quest_usb_bridge.py"),
        "--host=127.0.0.1",
    ]


def wait_for_ports(ports: Sequence[int], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    missing = list(ports)
    while time.monotonic() < deadline:
        missing = []
        for port in ports:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    pass
            except OSError:
                missing.append(port)
        if not missing:
            return
        time.sleep(0.25)
    raise RuntimeError(f"Open-Teach 桥未能监听端口：{missing}")


def listening_ports(ports: Sequence[int]) -> list[int]:
    occupied = []
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                occupied.append(port)
        except OSError:
            pass
    return occupied


@contextmanager
def managed_bridge(config: dict, *, setup_quest: bool = True) -> Iterator[None]:
    _, openteach, _ = paths(config)
    STATE_DIR.mkdir(exist_ok=True)
    log_path = STATE_DIR / "bridge.log"
    with (STATE_DIR / "bridge.lock").open("a+b") as lock, log_path.open("ab", buffering=0) as log:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Open-Teach 桥已由另一项任务运行；请先停止只读监视或遥操。") from None
        occupied = listening_ports((8125, 8089, 8102, 8127))
        if occupied:
            raise SystemExit(f"Open-Teach 端口已被占用：{occupied}；请先停止旧桥进程。")
        if setup_quest:
            cmd_quest_setup(argparse.Namespace(), config)
        process = subprocess.Popen(
            bridge_command(config), cwd=openteach, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_for_ports((8125, 8089, 8102, 8127))
            print(f"Open-Teach 桥已启动（日志：{log_path}）")
            yield
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)


def cmd_bridge(args: argparse.Namespace, config: dict) -> int:
    with managed_bridge(config, setup_quest=not args.no_setup):
        print("只读手柄桥运行中；在 Quest 的 Menu 中进入 Stream 并按 B 继续。按 Ctrl+C 停止。")
        while True:
            time.sleep(0.5)


def common_robot_args(
    config: dict,
    *,
    cautious: bool,
    tcp_speed_mm_s: float | None = None,
    tcp_acceleration_mm_s2: float | None = None,
) -> list[str]:
    safety = config["safety"]
    prefix = [
        "--robot.type=xarm7",
        f"--robot.ip={config['robot_ip']}",
        "--robot.id=xarm7-quest-usb",
        f"--robot.gripper_open_position={config['gripper']['open_position']}",
        f"--robot.gripper_close_position={config['gripper']['close_position']}",
        "--robot.disable_motion_on_disconnect=true",
        "--robot.telemetry_path=/home/a/xarm7-vr-lab/.state/control_telemetry.json",
    ]
    if cautious:
        speed = tcp_speed_mm_s or safety["first_test"]["tcp_speed_mm_s"]
        acceleration = tcp_acceleration_mm_s2 or safety["first_test"]["tcp_acceleration_mm_s2"]
        prefix += [
            f"--robot.max_translation_step_mm={safety['first_test']['max_translation_step_mm']}",
            f"--robot.max_rotation_step_rad={safety['first_test']['max_rotation_step_rad']}",
            f"--robot.tcp_speed_mm_s={speed}",
            f"--robot.tcp_acceleration_mm_s2={acceleration}",
        ]
    else:
        speed = tcp_speed_mm_s or safety["recording"]["tcp_speed_mm_s"]
        acceleration = tcp_acceleration_mm_s2 or safety["recording"]["tcp_acceleration_mm_s2"]
        prefix += [
            f"--robot.max_translation_step_mm={safety['recording']['max_translation_step_mm']}",
            f"--robot.max_rotation_step_rad={safety['recording']['max_rotation_step_rad']}",
            f"--robot.tcp_speed_mm_s={speed}",
            f"--robot.tcp_acceleration_mm_s2={acceleration}",
        ]
    return prefix


def teleop_args(config: dict) -> list[str]:
    return [
        "--teleop.type=openteach_quest_controller",
        "--teleop.host=127.0.0.1",
        "--teleop.id=quest3-usb",
    ]


def read_tcp_xyz(config: dict, python: Path) -> list[float]:
    script = """
import json, sys
from xarm.wrapper import XArmAPI
a=XArmAPI(sys.argv[1], is_radian=False)
try:
 c,p=a.get_position(is_radian=False)
 if c: raise SystemExit(c)
 print(json.dumps(p[:3]))
finally: a.disconnect()
"""
    return json.loads(output([str(python), "-c", script, config["robot_ip"]]).splitlines()[-1])


def cmd_teleop(args: argparse.Namespace, config: dict) -> int:
    confirm_motion(args)
    speed = float(getattr(args, "speed_mm_s", None) or config["safety"]["first_test"]["tcp_speed_mm_s"])
    radius = float(getattr(args, "range_mm", 10.0))
    translation_scale = float(getattr(args, "translation_scale", 0.25))
    if not 1.0 <= speed <= 50.0:
        raise SystemExit("测试遥操速度必须在 1–50 mm/s 之间")
    if not 2.0 <= radius <= 300.0:
        raise SystemExit("动态围栏半径必须在 2–300 mm 之间")
    if not 0.05 <= translation_scale <= 1.0:
        raise SystemExit("手柄位移比例必须在 0.05–1.0 之间")
    acceleration = min(200.0, max(25.0, speed * 5.0))
    lerobot, _, python = paths(config)
    xyz = read_tcp_xyz(config, python)
    low = [round(value - radius, 3) for value in xyz]
    high = [round(value + radius, 3) for value in xyz]
    command = [str(lerobot / ".venv/bin/lerobot-teleoperate")]
    command += common_robot_args(
        config,
        cautious=True,
        tcp_speed_mm_s=speed,
        tcp_acceleration_mm_s2=acceleration,
    )
    command += [f"--robot.workspace_min_mm={json.dumps(low)}", f"--robot.workspace_max_mm={json.dumps(high)}"]
    command += teleop_args(config)
    command += [
        f"--teleop.translation_scale={translation_scale}",
        "--teleop.control_rotation=false",
        "--fps=30",
    ]
    print(f"测试遥操动态围栏：{low} 到 {high} mm；速度 {speed:g} mm/s；旋转已禁用。")
    if args.dry_run:
        print(command_text(command))
        return 0
    # Report robot faults before opening the Quest bridge or claiming live control.
    run(preflight_command(config, skip_quest=True), cwd=lerobot)
    with managed_bridge(config, setup_quest=not args.no_setup):
        run(preflight_command(config), cwd=lerobot)
        return run(command, cwd=lerobot)


def cmd_record(args: argparse.Namespace, config: dict) -> int:
    confirm_motion(args)
    speed = float(getattr(args, "speed_mm_s", None) or config["safety"]["recording"]["tcp_speed_mm_s"])
    translation_scale = float(getattr(args, "translation_scale", 1.0))
    dynamic_radius = getattr(args, "dynamic_range_mm", None)
    if not 1.0 <= speed <= 100.0:
        raise SystemExit("采集速度必须在 1–100 mm/s 之间")
    if not 0.05 <= translation_scale <= 1.0:
        raise SystemExit("手柄位移比例必须在 0.05–1.0 之间")
    if not args.name.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("数据集名称只能包含字母、数字、下划线和连字符")
    lerobot, _, python = paths(config)
    workspace = config.get("workspace_mm")
    if workspace:
        workspace_command = workspace_args(config)
    elif dynamic_radius is not None:
        dynamic_radius = float(dynamic_radius)
        if not 10.0 <= dynamic_radius <= 300.0:
            raise SystemExit("试采动态围栏半径必须在 10–300 mm 之间")
        xyz = read_tcp_xyz(config, python)
        low = [round(value - dynamic_radius, 3) for value in xyz]
        high = [round(value + dynamic_radius, 3) for value in xyz]
        workspace_command = [
            f"--robot.workspace_min_mm={json.dumps(low)}",
            f"--robot.workspace_max_mm={json.dumps(high)}",
        ]
        print(f"试采动态围栏：{low} 到 {high} mm。")
    else:
        require_workspace(config)
    root = Path(config["dataset_root"]) / args.name
    if root.exists() and not args.resume:
        raise SystemExit(f"数据集已存在：{root}；如要追加请加 --resume")
    command = [str(lerobot / ".venv/bin/lerobot-record")]
    command += common_robot_args(
        config,
        cautious=False,
        tcp_speed_mm_s=speed,
        tcp_acceleration_mm_s2=min(300.0, max(25.0, speed * 5.0)),
    ) + workspace_command
    command += [f"--robot.cameras={camera_argument(config)}"] + teleop_args(config)
    command += [
        f"--teleop.translation_scale={translation_scale}",
        f"--teleop.control_rotation={str(not args.disable_rotation).lower()}",
        f"--dataset.repo_id=local/{args.name}",
        f"--dataset.root={root}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.episodes}",
        f"--dataset.episode_time_s={args.episode_seconds}",
        f"--dataset.reset_time_s={args.reset_seconds}",
        "--dataset.fps=30",
        "--dataset.video=true",
        "--dataset.no_stamp=true",
        "--dataset.push_to_hub=false",
        f"--resume={str(args.resume).lower()}",
        "--play_sounds=true",
    ]
    if args.dry_run:
        print(command_text(command))
        return 0
    with managed_bridge(config, setup_quest=not args.no_setup):
        run(preflight_command(config), cwd=lerobot)
        return run(command, cwd=lerobot)


def read_dataset_info(path: Path) -> dict:
    info = path / "meta/info.json"
    if not info.exists():
        raise SystemExit(f"不是有效的 LeRobot 数据集：{path}")
    return json.loads(info.read_text(encoding="utf-8"))


def cmd_dataset(args: argparse.Namespace, config: dict) -> int:
    path = Path(config["dataset_root"]) / args.name
    info = read_dataset_info(path)
    image_features = [name for name, value in info.get("features", {}).items() if value.get("dtype") in {"video", "image"}]
    summary = {
        "path": str(path),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "image_features": image_features,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not image_features:
        print("警告：该数据集没有图像，不能用于视觉语言动作（VLA）训练。", file=sys.stderr)
        return 2
    return 0


def cmd_train(args: argparse.Namespace, config: dict) -> int:
    lerobot, _, _ = paths(config)
    dataset_path = Path(config["dataset_root"]) / args.dataset
    info = read_dataset_info(dataset_path)
    image_features = [value for value in info.get("features", {}).values() if value.get("dtype") in {"video", "image"}]
    if not image_features:
        raise SystemExit("拒绝启动训练：数据集不含相机图像；请使用本工具重新采集视觉数据。")
    output_dir = Path(args.output or (ROOT / "outputs" / f"{args.policy}_{args.dataset}"))
    command = [
        str(lerobot / ".venv/bin/lerobot-train"),
        f"--dataset.repo_id=local/{args.dataset}",
        f"--dataset.root={dataset_path}",
        f"--output_dir={output_dir}",
        f"--job_name={args.policy}_{args.dataset}",
        "--policy.device=cuda",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]
    if args.policy == "act":
        command.append("--policy.type=act")
    elif args.policy == "smolvla":
        command += [
            "--policy.path=lerobot/smolvla_base",
            f"--policy.empty_cameras={args.empty_cameras}",
        ]
    elif args.policy == "pi0":
        command += [
            "--policy.type=pi0",
            "--policy.pretrained_path=lerobot/pi0_base",
            f"--policy.empty_cameras={args.empty_cameras}",
            "--policy.dtype=bfloat16",
            "--policy.gradient_checkpointing=true",
        ]
    if args.dry_run:
        print(command_text(command))
        return 0
    return run(command, cwd=lerobot)


def cmd_run_policy(args: argparse.Namespace, config: dict) -> int:
    require_workspace(config)
    confirm_motion(args)
    lerobot, _, _ = paths(config)
    command = [str(lerobot / ".venv/bin/lerobot-rollout"), "--strategy.type=base"]
    command += common_robot_args(config, cautious=False) + workspace_args(config)
    command += [
        f"--robot.cameras={camera_argument(config)}",
        f"--policy.path={args.policy_path}",
        f"--task={args.task}",
        "--fps=30",
        f"--duration={args.duration}",
        "--display_data=true",
    ]
    if args.dry_run:
        print(command_text(command))
        return 0
    run(preflight_command(config, skip_quest=True), cwd=lerobot)
    return run(command, cwd=lerobot)


def cmd_menu(args: argparse.Namespace, config: dict, parser: argparse.ArgumentParser) -> int:
    choices = {"1": ["status"], "2": ["quest-setup"], "3": ["teleop"], "4": ["dataset"]}
    print("1) 设备状态（安全、只读）\n2) 建立 Quest USB 链路\n3) 首次小范围遥操作\n4) 检查数据集\nq) 退出")
    choice = input("请选择：").strip().lower()
    if choice == "q":
        return 0
    if choice not in choices:
        raise SystemExit("无效选择")
    if choice in {"3", "4"}:
        print("这些命令需要参数；请运行 xarm7-vr --help 查看示例。")
        return 0
    nested = parser.parse_args(choices[choice])
    return dispatch(nested, config, DEFAULT_CONFIG, parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xarm7-vr", description="XARM7 + G2 + Quest + LeRobot 一体化工具")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="只读检查机械臂、相机、Quest 和 GPU")
    workspace = sub.add_parser("workspace", help="设置任务绝对 XYZ 安全边界（单位 mm）")
    workspace.add_argument("--min", dest="minimum", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    workspace.add_argument("--max", dest="maximum", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    install = sub.add_parser("quest-install", help="在 Quest 安装/更新控制器 APK")
    install.add_argument("--yes", action="store_true")
    sub.add_parser("quest-setup", help="建立 ADB reverse 并启动 Quest 应用")
    bridge = sub.add_parser("bridge", help="前台运行 Open-Teach Quest 数据桥（不会连接机械臂）")
    bridge.add_argument("--no-setup", action="store_true")
    teleop = sub.add_parser("teleop", help="可调动态围栏、无旋转的安全试动")
    teleop.add_argument("--enable-motion", action="store_true")
    teleop.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    teleop.add_argument("--dry-run", action="store_true")
    teleop.add_argument("--no-setup", action="store_true")
    teleop.add_argument("--speed-mm-s", type=float, default=50.0, help="TCP 速度，允许 1–50 mm/s")
    teleop.add_argument("--range-mm", type=float, default=10.0, help="以启动 TCP 为中心的 XYZ 围栏半径")
    teleop.add_argument(
        "--translation-scale", type=float, default=0.25, help="Quest 位移比例，允许 0.05–1.0"
    )
    record = sub.add_parser("record", help="遥操作并直接采集 LeRobotDataset v3")
    record.add_argument("name")
    record.add_argument("--task", required=True)
    record.add_argument("--episodes", type=int, default=50)
    record.add_argument("--episode-seconds", type=float, default=60)
    record.add_argument("--reset-seconds", type=float, default=20)
    record.add_argument("--resume", action="store_true")
    record.add_argument("--enable-motion", action="store_true")
    record.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    record.add_argument("--dry-run", action="store_true")
    record.add_argument("--no-setup", action="store_true")
    record.add_argument("--speed-mm-s", type=float, default=50.0)
    record.add_argument("--translation-scale", type=float, default=1.0)
    record.add_argument(
        "--dynamic-range-mm",
        type=float,
        help="没有正式工作空间时，以启动 TCP 为中心建立仅供试采的动态围栏",
    )
    record.add_argument("--disable-rotation", action="store_true")
    dataset = sub.add_parser("dataset", help="检查数据集是否含视觉数据")
    dataset.add_argument("name")
    train = sub.add_parser("train", help="训练 ACT 或 VLA")
    train.add_argument("dataset")
    train.add_argument("--policy", choices=("act", "smolvla", "pi0"), default="act")
    train.add_argument("--steps", type=int, default=100000)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument(
        "--empty-cameras",
        type=int,
        default=2,
        help="VLA 零填充相机数量；当前单腕部相机对应默认值 2",
    )
    train.add_argument("--output")
    train.add_argument("--dry-run", action="store_true")
    policy = sub.add_parser("run-policy", help="让训练后的策略短时驱动真机")
    policy.add_argument("policy_path")
    policy.add_argument("--task", required=True)
    policy.add_argument("--duration", type=float, default=10)
    policy.add_argument("--enable-motion", action="store_true")
    policy.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    policy.add_argument("--dry-run", action="store_true")
    sub.add_parser("menu", help="显示交互式入门菜单")
    return parser


def dispatch(args: argparse.Namespace, config: dict, config_path: Path, parser: argparse.ArgumentParser) -> int:
    handlers = {
        "status": cmd_status,
        "workspace": lambda a, c: cmd_workspace(a, c, config_path),
        "quest-install": cmd_quest_install,
        "quest-setup": cmd_quest_setup,
        "bridge": cmd_bridge,
        "teleop": cmd_teleop,
        "record": cmd_record,
        "dataset": cmd_dataset,
        "train": cmd_train,
        "run-policy": cmd_run_policy,
        "menu": lambda a, c: cmd_menu(a, c, parser),
    }
    return handlers[args.command](args, config)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    return dispatch(args, config, args.config, parser)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已安全停止。")
        raise SystemExit(130) from None
