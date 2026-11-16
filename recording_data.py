"""Versioned, crash-recoverable raw episodes and model-neutral alignment.

SQLite contains timestamped telemetry JSON and lossless PNG camera frames.
Only finalized episodes may be exported. No SDK or control connection exists here.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

RAW_VERSION = "xarm7.raw.v1"


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value), encoding="utf-8")
    temporary.replace(path)


class EpisodeWriter:
    def __init__(self, root, metadata):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:10]
        self.path = root / "episodes" / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, schema_version=RAW_VERSION, episode_id=self.id,
                             status="recording", started_utc_s=time.time(),
                             started_monotonic_s=time.monotonic(), success=None)
        atomic_json(self.path / "episode.json", self.metadata)
        self.db = sqlite3.connect(self.path / "raw.sqlite3")
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE telemetry(sequence INTEGER PRIMARY KEY, t REAL NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX telemetry_time ON telemetry(t);
            CREATE TABLE images(camera TEXT NOT NULL, sequence INTEGER NOT NULL,
                t REAL NOT NULL, read_start REAL NOT NULL, width INTEGER NOT NULL,
                height INTEGER NOT NULL, png BLOB NOT NULL, PRIMARY KEY(camera,sequence));
            CREATE INDEX images_time ON images(camera,t);
            CREATE TABLE events(t REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
        """)
        self.last_commit = time.monotonic()
        self.samples = 0
        self.images = {}

    def add_sample(self, item):
        self.db.execute("INSERT INTO telemetry VALUES(?,?,?)",
                        (item["sequence"], item["monotonic_s"], json_text(item)))
        self.samples += 1

    def add_image(self, item):
        self.db.execute("INSERT INTO images VALUES(?,?,?,?,?,?,?)", (
            item["camera"], item["sequence"], item["t"], item["read_start"],
            item["width"], item["height"], item["png"]))
        self.images[item["camera"]] = self.images.get(item["camera"], 0) + 1

    def event(self, kind, value):
        self.db.execute("INSERT INTO events VALUES(?,?,?)", (time.monotonic(), kind, json_text(value)))

    def flush(self):
        if time.monotonic() - self.last_commit >= .5:
            self.db.commit()
            self.last_commit = time.monotonic()

    def finish(self, status="complete", success=None, reason="", quality=None):
        self.event("finish", dict(status=status, success=success, reason=reason))
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()
        self.metadata.update(status=status, success=success, reason=reason,
                             ended_utc_s=time.time(), ended_monotonic_s=time.monotonic(),
                             telemetry_samples=self.samples, image_counts=self.images,
                             quality=quality or {})
        atomic_json(self.path / "episode.json", self.metadata)
        return self.path


def open_episode(path):
    path = Path(path)
    if path.is_symlink() or any((path/name).is_symlink() for name in ('episode.json', 'raw.sqlite3')):
        raise ValueError('不支持链接轨迹或数据库')
    metadata = json.loads((path / "episode.json").read_text())
    if metadata.get("schema_version") != RAW_VERSION:
        raise ValueError("不支持的原始数据版本")
    if metadata.get("status") != "complete":
        raise ValueError(f"未完成/已丢弃的 episode 不可用于训练：{path.name}")
    if not (path/'raw.sqlite3').is_file() or (path/'raw.sqlite3-wal').exists():
        raise ValueError('数据库缺失或尚未关闭，暂不导出')
    db = sqlite3.connect(f"{(path / 'raw.sqlite3').resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    return db, metadata


def nearest_index(times, t):
    if not times:
        raise ValueError("数据流为空")
    pos = bisect.bisect_left(times, t)
    return min((max(0, pos - 1), min(pos, len(times)-1)), key=lambda i: abs(times[i]-t))


def finite_vector(value, size, label):
    if not isinstance(value, list) or len(value) != size or not all(
            type(x) in (int, float) and math.isfinite(x) for x in value):
        raise ValueError(f"无效字段：{label}")
    return value


def quaternion_vector(value, label):
    value = finite_vector(value, 4, label)
    if abs(sum(x*x for x in value)-1) > .02:
        raise ValueError('无效单位四元数：'+label)
    return value


def training_vectors(sample, action_space="joint", max_gripper_age=.75):
    """Keep pause periods continuous; settled holds use measured pose explicitly.

    Never infer gripper command from a pending button/request. Only acknowledged
    target is usable; before first command the fresh measured aperture is held.
    """
    grip = sample.get("gripper") or {}
    age, aperture = grip.get("position_age_s"), grip.get("position_mm")
    if type(age) not in (int,float) or not 0 <= age <= max_gripper_age or type(aperture) not in (int,float) or not math.isfinite(aperture):
        raise ValueError("夹爪实测位置缺失/过期")
    if not 0 <= aperture <= 84 or grip.get("command_fault"):
        raise ValueError("夹爪数据无效/命令失败")
    target = grip.get("committed_target_mm")
    if target is None:
        target = aperture
    if type(target) not in (int,float) or not math.isfinite(target) or not 0 <= target <= 84:
        raise ValueError("夹爪已下发目标无效")
    if type(sample.get('command_valid')) is not bool:
        raise ValueError('命令有效标记缺失或类型错误')
    held = not sample['command_valid']
    if sample.get("recovery_active") or sample.get("error") or sample.get("source_valid") is not True:
        raise ValueError("包含复位、控制错误或无效追踪；请检查原始记录")
    if sample.get("mode") != "motion":
        raise ValueError("包含未启用遥操的片段")
    # During physical braking no 'hold current' command was issued; do not invent
    # one. An episode with these rows is split by the export alignment routine.
    if sample.get("pause_settling"):
        raise ValueError("pause_settling")
    if action_space in ("joint", "joint_delta"):
        state = finite_vector(sample.get("joints_rad"), 7, "joints_rad")
        action = state if held else finite_vector(sample.get("command_joints_rad"), 7, "command_joints_rad")
        if action_space == "joint_delta":
            action = [a-b for a, b in zip(action, state)]
    elif action_space == "tcp_pose":
        state = finite_vector(sample.get("tcp_position_m"), 3, "tcp_position_m") + quaternion_vector(
            sample.get("tcp_quaternion_wxyz"), "tcp_quaternion_wxyz")
        action = state if held else finite_vector(sample.get("command_tcp_position_m"), 3, "command_tcp_position_m") + quaternion_vector(
            sample.get("command_tcp_quaternion_wxyz"), "command_tcp_quaternion_wxyz")
    else:
        raise ValueError("未知动作空间")
    return state + [aperture/84], list(action) + [target/84], held


def aligned_segments(db, metadata, fps=30, action_space="joint", camera_tolerance=.04,
                     robot_tolerance=.025, _window=None):
    """Yield contiguous, uniform-FPS segments; don't close time gaps by concatenation.

    Paused braking samples are omitted by splitting, not interpreted as actions.
    All other data defects reject the episode. Every source row remains in raw.
    """
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("导出 FPS 必须在 1–60 之间")
    if metadata.get('capture_mode') == 'trigger' and _window is None:
        intervals = metadata.get('capture_intervals',[])
        previous_end = -math.inf
        for interval in intervals:
            start,end = interval['start_s'],interval['end_s']
            if end is None or not math.isfinite(start) or not math.isfinite(end) or end<start or start<previous_end:
                raise ValueError('扳机采集时间范围无效')
            previous_end=end
            # Never drop the final tap and attach its outcome to an earlier clip.
            if end-start < 2/fps:
                if os.environ.get("XARM_ALLOW_SHORT_INTERVALS") == "1":
                    continue
                raise ValueError('按压片段过短，无法对齐双相机；请复核原始记录')
            yield from aligned_segments(db,metadata,fps,action_space,camera_tolerance,robot_tolerance,interval)
        return
    bounds = (-1e30,1e30) if _window is None else (_window['start_s'],_window['end_s'])
    indexed = [(seq,t,json.loads(p)) for seq,t,p in db.execute(
        "SELECT sequence,t,payload FROM telemetry WHERE t>=? AND t<? ORDER BY t",bounds)]
    if any(sample.get('sequence') != seq for seq,_,sample in indexed):
        raise ValueError('机器人采样序号与数据库索引不一致')
    samples = [(t,sample) for _,t,sample in indexed]
    if len(samples) < 2:
        raise ValueError("机器人样本不足")
    times = [row[0] for row in samples]
    if not all(type(t) in (int,float) and math.isfinite(t) for t in times):
        raise ValueError('机器人时间戳无效')
    if any(b <= a or b-a > .1 for a, b in zip(times, times[1:])):
        raise ValueError("机器人数据时间倒退或间断超过 100 ms")
    if len({s["run_id"] for _, s in samples}) != 1:
        raise ValueError("同一条轨迹包含多个控制会话")
    if len({s.get('settings_revision') for _,s in samples}) != 1:
        raise ValueError('同一片段内遥操参数发生变化')
    sequences = [s["sequence"] for _, s in samples]
    if any(b != a+1 for a,b in zip(sequences,sequences[1:])):
        raise ValueError("机器人原始数据有丢包，拒绝默认训练导出")
    # Validate every recorded sample, including those between export-grid ticks.
    # Otherwise downsampling can silently hide an invalid state or bad command.
    for stamp, sample in samples:
        if sample.get('monotonic_s') != stamp:
            raise ValueError('机器人采样时间与数据库索引不一致')
        if metadata.get('capture_mode') == 'trigger' and (
                sample.get('active') is not True or sample.get('command_valid') is not True or
                sample.get('hand',{}).get('trigger_pressed') is not True or sample.get('pause_settling')):
            raise ValueError('扳机采集区间包含非遥操/暂停样本')
        if not sample.get('pause_settling'):
            training_vectors(sample, action_space)
    cameras = [c["name"] for c in metadata["cameras"]]
    image_rows = {c: list(db.execute("SELECT sequence,t FROM images WHERE camera=? AND t>=? AND t<? ORDER BY t", (c,*bounds))) for c in cameras}
    if any(len(rows) < 2 for rows in image_rows.values()):
        raise ValueError("相机样本不足")
    image_times = {c: [t for _, t in rows] for c,rows in image_rows.items()}
    for camera, rows in image_rows.items():
        if any(b[0]!=a[0]+1 or b[1]<=a[1] or b[1]-a[1]>.1 for a,b in zip(rows,rows[1:])):
            raise ValueError(f"相机 {camera} 存在丢帧或超过 100 ms 的间断")
    start = max(times[0], *(ts[0] for ts in image_times.values()))
    end = min(times[-1], *(ts[-1] for ts in image_times.values()))
    segment = []
    selected_images={c:[] for c in cameras}
    for i in range(max(0, int((end-start)*fps)+1)):
        t = start + i/fps
        j = nearest_index(times, t)
        sample = samples[j][1]
        if abs(times[j]-t) > robot_tolerance:
            raise ValueError("机器人采样无法对齐导出时刻")
        if sample.get("pause_settling"):
            if segment:
                yield segment
                segment = []
            continue
        state, action, held = training_vectors(sample, action_space)
        images, offsets = {}, {}
        for camera in cameras:
            k = nearest_index(image_times[camera], t)
            seq, stamp = image_rows[camera][k]
            if abs(stamp-t) > camera_tolerance:
                raise ValueError(f"相机 {camera} 时间偏差超过 {camera_tolerance*1000:g} ms")
            images[camera], offsets[camera] = seq, stamp-t
            selected_images[camera].append(seq)
        segment.append(dict(t=t, state=state, action=action, held=held,
                            capture_segment=sample.get('capture_segment'),
                            sample_sequence=sample["sequence"], sample_offset_s=times[j]-t,
                            images=images, camera_offsets_s=offsets))
    if segment:
        yield segment
    # Repeated aligned camera frames are retained by explicit user policy.


def image_rgb(db, camera, sequence):
    import cv2
    import numpy as np
    row = db.execute("SELECT png FROM images WHERE camera=? AND sequence=?", (camera, sequence)).fetchone()
    if row is None:
        raise ValueError("缺少原始图像")
    bgr = cv2.imdecode(np.frombuffer(row[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("原始 PNG 解码失败")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def source_fingerprint(path):
    # Manifest + immutable DB contents link each export back to exact raw bytes.
    digest = hashlib.sha256()
    for name in ("episode.json", "raw.sqlite3"):
        with (Path(path)/name).open("rb") as file:
            for block in iter(lambda: file.read(1024*1024), b""):
                digest.update(block)
    return digest.hexdigest()
