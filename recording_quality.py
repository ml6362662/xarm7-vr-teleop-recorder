"""Conservative episode screening. Never rewrites raw samples or outcome labels."""
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import time

from recording_data import aligned_segments, atomic_json, image_rgb, open_episode

DATA_ERRORS = (ValueError, KeyError, TypeError, OSError, sqlite3.DatabaseError)
REPORT_VERSION = 'xarm7.quality.v1'


def episode_candidates(source, include_unsuccessful=False, include_failures=False):
    source = Path(source)
    if not source.is_dir() or source.is_symlink() or (source/'episodes').is_symlink():
        raise ValueError('原始数据集路径不存在或为链接')
    for path in sorted((source/'episodes').iterdir()):
        if not path.is_dir():
            continue
        item = dict(episode_id=path.name, decision='excluded', reason='', frames=0)
        try:
            if path.is_symlink() or (path/'episode.json').is_symlink():
                raise ValueError('不支持链接轨迹')
            meta = json.loads((path/'episode.json').read_text())
            item.update(outcome=('success' if meta.get('success') is True else
                                 'failure' if meta.get('success') is False else 'unknown'),
                        task=meta.get('task', ''), status=meta.get('status'))
            if meta.get('status') != 'complete' or not meta.get('retained', True):
                item['reason'] = '未完成、录制中或已丢弃'
            elif not (meta.get('success') is True or include_unsuccessful or
                      (include_failures and meta.get('success') is False)):
                item['reason'] = '按结果筛选排除（失败/未标注）'
            else:
                item['decision'] = 'pending'
        except DATA_ERRORS as error:
            item.update(decision='rejected', reason=str(error))
        yield path, item


def inspect_episode(db, meta, fps, action_space):
    """All retained intervals must pass; no truncation or fabricated success."""
    if not isinstance(meta.get('task'), str) or not meta['task'].strip():
        raise ValueError('缺少任务描述')
    cameras = meta['cameras']
    if not isinstance(cameras, list) or not cameras:
        raise ValueError('缺少相机配置')
    names = [c['name'] for c in cameras]
    if len(set(names)) != len(names):
        raise ValueError('相机名称重复')
    for aliases in ({'global': 'cam_high', 'wrist': 'cam_left_wrist'},
                    {'global': 'cam_head', 'wrist': 'cam_left_wrist'}):
        if len({aliases.get(n,n) for n in names}) != len(names):
            raise ValueError('导出相机名称冲突')
    for c in cameras:
        if not isinstance(c['name'], str) or not c['name'] or any(
                ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in c['name']):
            raise ValueError('相机名称无效')
        if any(type(c[k]) is not int or c[k] <= 0 or c[k] % 2 for k in ('width','height')):
            raise ValueError('相机分辨率必须为正偶数')
        if c.get('rotation', 0) not in (0,90,180,270):
            raise ValueError('相机旋转无效')
    segments = list(aligned_segments(db, meta, fps, action_space))
    if not segments or any(len(segment) < 2 for segment in segments):
        raise ValueError('有效帧不足或按压片段过短，不能删除结尾后沿用成功标签')
    rows = [row for segment in segments for row in segment]
    checked = set()
    for c in cameras:
        for row in rows:
            key = (c['name'], row['images'][c['name']])
            if key in checked:
                continue
            image = image_rgb(db, *key)
            if image.shape != (c['height'], c['width'], 3):
                raise ValueError(f"相机 {c['name']} 图像分辨率与元数据不一致")
            checked.add(key)
    metrics = dict(frames=len(rows), segments=len(segments),
                   playback_duration_s=len(rows)/fps,
                   max_robot_offset_ms=max(abs(r['sample_offset_s']) for r in rows)*1000,
                   max_camera_offset_ms=max(abs(v) for r in rows for v in r['camera_offsets_s'].values())*1000)
    metrics['repeated_frame_policy']='retain; statistics only'
    metrics['repeated_camera_frames']={c['name']:len(rows)-len({row['images'][c['name']] for row in rows}) for c in cameras}
    # Timing quality uses the recorded SDK read window, not a claim of hardware sync.
    ages, read_windows, hand_ages = [], [], []
    sessions, revisions = set(), set()
    selected = {r['sample_sequence'] for r in rows}
    for seq, stamp, payload in db.execute('SELECT sequence,t,payload FROM telemetry ORDER BY sequence'):
        if seq not in selected:
            continue
        sample = json.loads(payload)
        sessions.add(sample['run_id'])
        revisions.add(sample.get('settings_revision'))
        if sample.get('sequence') != seq:
            raise ValueError('机器人采样序号与数据库索引不一致')
        start, end = sample.get('observation_read_start_s'), sample.get('observation_read_end_s')
        if start is not None or end is not None:
            if not all(type(v) in (int,float) and math.isfinite(v) for v in (start,end)) or not start <= end <= stamp+1e-6:
                raise ValueError('机器人观测读取时间范围无效')
            ages.append((stamp-end)*1000)
            read_windows.append((end-start)*1000)
        hand = sample.get('hand') or {}
        hand_stamp = hand.get('sample_monotonic_s')
        if type(hand_stamp) in (int,float) and math.isfinite(hand_stamp):
            hand_ages.append((stamp-hand_stamp)*1000)
    if len(sessions) != 1 or len(revisions) != 1:
        raise ValueError('同一轨迹跨片段的控制会话或参数版本不同')
    warnings = []
    if ages:
        metrics.update(max_observation_age_ms=max(ages), max_observation_read_window_ms=max(read_windows))
        if max(ages)+max(read_windows) > 25:
            warnings.append('SDK 观测读取/发布时间偏差超过 25 ms；仅按主机记录时间对齐，需复核')
    else:
        warnings.append('缺少 SDK 观测读取时间，不能核实实测状态的新鲜度')
    if hand_ages:
        metrics['max_headset_age_ms'] = max(hand_ages)
        if max(hand_ages) > 180:
            warnings.append('存在头显输入年龄超过 180 ms 的采样，需复核遥操质量')
    if meta.get('capture_mode') != 'trigger':
        warnings.append('旧记录无扳机区间标记，无法保证已排除全部暂停')
    metrics['warnings'] = warnings
    return segments, metrics


def new_report(source, fps, action_space, include_failures=False, include_unsuccessful=False):
    return dict(schema_version=REPORT_VERSION, source=str(Path(source).resolve()),
                created_at=time.time(), fps=fps, action_space=action_space,
                include_failures=include_failures, include_unsuccessful=include_unsuccessful,
                policy='按整条轨迹筛选；原始数据不修改；不裁掉坏片段后继承成功标签；警告需人工复核',
                episodes=[])


def finalize_report(report):
    counts = Counter(item['decision'] for item in report['episodes'])
    report['summary'] = dict(total=len(report['episodes']),
                            accepted=counts['accepted'], rejected=counts['rejected'], excluded=counts['excluded'],
                            warnings=sum(bool(item.get('warnings')) for item in report['episodes']))
    return report


def write_report(path, report):
    path = Path(path)
    finalize_report(report)
    atomic_json(path, report)
    lines = [f"数据清洗报告 · {report['fps']} FPS", report['policy'],
             '汇总：'+json.dumps(report['summary'], ensure_ascii=False)]
    labels = dict(accepted='合格', rejected='不合格', excluded='未纳入')
    for item in report['episodes']:
        lines.append(f"{item['episode_id']} · {labels.get(item['decision'], item['decision'])} · {item.get('reason','')}")
        lines.extend('  注意：'+w for w in item.get('warnings', []))
    path.with_suffix('.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def scan_dataset(source, fps=15, action_space='joint', include_failures=False, include_unsuccessful=False):
    from recording_library import library_lock
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError('导出 FPS 必须在 1–60 之间')
    report = new_report(source, fps, action_space, include_failures, include_unsuccessful)
    signature = None
    with library_lock(source):
        for path, item in episode_candidates(source, include_unsuccessful, include_failures):
            if item['decision'] == 'pending':
                try:
                    db, meta = open_episode(path)
                    try:
                        _, metrics = inspect_episode(db, meta, fps, action_space)
                        current = camera_signature(meta)
                        if signature is not None and signature != current:
                            raise ValueError('相机布局与本批已通过轨迹不同，请分开导出')
                        signature = current
                        item.update(metrics, decision='accepted', reason='结构、对齐和图像校验通过')
                    finally:
                        db.close()
                except DATA_ERRORS as error:
                    item.update(decision='rejected', reason=str(error))
            report['episodes'].append(item)
            print(f"已检查 {path.name}：{item['decision']} · {item['reason']}", flush=True)
    return finalize_report(report)


def camera_signature(meta):
    return [(c['name'],c['width'],c['height'],c.get('rotation',0)) for c in meta['cameras']]
