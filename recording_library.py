"""Raw episode browsing and reversible moves. No SDK, camera or robot control."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3

from recording_data import RAW_VERSION


@contextmanager
def library_lock(root, exclusive=False):
    path = Path(root)/'.library.lock'
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'rb') as lock:
        try:
            fcntl.flock(lock, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('数据正在导出、预览或整理，请稍后重试') from None
        yield


def dataset_root(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('不支持链接数据集目录')
    root = root.resolve(strict=True)
    if json.loads((root/'dataset.json').read_text()).get('schema_version') != RAW_VERSION:
        raise ValueError('请选择当前采集程序的原始数据集')
    return root


def episode_parent(root, trash=False):
    parent = root
    for part in (('.trash', 'episodes') if trash else ('episodes',)):
        parent = parent/part
        if parent.is_symlink():
            raise ValueError('不支持链接轨迹目录')
    return parent


def episode_path(root, episode_id, trash=False):
    if not episode_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in episode_id):
        raise ValueError('轨迹编号无效')
    path = episode_parent(root, trash)/episode_id
    if path.is_symlink():
        raise ValueError('不支持链接轨迹')
    return path


def read_metadata(path):
    file = path/'episode.json'
    if file.is_symlink():
        raise ValueError('不支持链接元数据')
    meta = json.loads(file.read_text())
    if meta.get('schema_version') != RAW_VERSION or meta.get('episode_id') != path.name:
        raise ValueError('轨迹元数据版本或编号不匹配')
    return meta


def category(meta):
    status = meta.get('status')
    if status == 'complete' and meta.get('retained', True):
        return 'success' if meta.get('success') is True else 'failure' if meta.get('success') is False else 'unlabeled'
    return {'recording':'recording', 'incomplete':'incomplete', 'discarded':'discarded'}.get(status, 'invalid')


def list_datasets(parent):
    parent = Path(parent).expanduser()
    if not parent.is_dir():
        return []
    result = []
    for path in sorted(parent.iterdir()):
        try:
            if path.is_dir() and not path.is_symlink() and json.loads((path/'dataset.json').read_text()).get('schema_version') == RAW_VERSION:
                result.append(path.name)
        except (OSError, ValueError):
            continue
    return result


def list_episodes(root, trash=False):
    root = dataset_root(root)
    result = []
    for path in sorted(episode_parent(root, trash).glob('*'), reverse=True):
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            meta = read_metadata(path)
            db = path/'raw.sqlite3'
            size = db.stat().st_size if db.exists() and not db.is_symlink() else 0
            result.append(dict(id=path.name, category=category(meta), metadata=meta, bytes=size,
                               duration=float(meta.get('recorded_duration_s') or 0),
                               samples=int(meta.get('telemetry_samples') or 0), trash=trash))
        except (OSError, ValueError, TypeError) as error:
            result.append(dict(id=path.name, category='invalid', metadata={}, bytes=0,
                               duration=0, samples=0, trash=trash, error=str(error)))
    return result


def move_episode(root, episode_id, *, restore=False):
    """Move one finalized directory atomically; never overwrite or delete bytes."""
    root = dataset_root(root)
    fd = os.open(root/'.recorder.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'rb') as recorder_lock:
        try:
            fcntl.flock(recorder_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('请先关闭此数据集的采集器，再整理轨迹') from None
        with library_lock(root, exclusive=True):
            source = episode_path(root, episode_id, trash=restore)
            meta = read_metadata(source)
            if meta.get('status') not in ('complete', 'incomplete', 'discarded'):
                raise ValueError('正在录制或状态未知的轨迹不能移动')
            if (source/'raw.sqlite3-wal').exists():
                raise ValueError('轨迹仍有数据库日志，请先确认采集器已正常退出')
            destination = episode_path(root, episode_id, trash=not restore)
            if destination.exists() or destination.is_symlink():
                raise ValueError('目标位置存在同名轨迹，未覆盖')
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            return destination


def source_time(meta, elapsed):
    intervals = meta.get('capture_intervals', [])
    if not intervals:
        raise ValueError('本条没有可预览的有效片段')
    elapsed = max(0., float(elapsed))
    for index, interval in enumerate(intervals):
        start, end = interval['start_s'], interval['end_s']
        if end is None or not all(math.isfinite(v) for v in (start, end)) or end < start:
            raise ValueError('片段时间无效')
        length = end-start
        if elapsed < length or index == len(intervals)-1:
            return min(start+elapsed, max(start, end-1e-6)), index
        elapsed -= length
    raise ValueError('片段时间无效')


def preview(root, episode_id, elapsed=0., trash=False):
    root = dataset_root(root)
    path = episode_path(root, episode_id, trash)
    # Immutable reads never create WAL/SHM alongside archived files. Only closed
    # episodes without outstanding WAL are eligible for this mode.
    with library_lock(root):
        meta = read_metadata(path)
        if meta.get('status') not in ('complete', 'incomplete', 'discarded'):
            raise ValueError('本条尚未结束，预览请使用采集页')
        file = path/'raw.sqlite3'
        if file.is_symlink() or (path/'raw.sqlite3-wal').exists():
            raise ValueError('数据库尚未关闭或为链接，暂不预览')
        stamp, segment = source_time(meta, elapsed)
        interval = meta['capture_intervals'][segment]
        bounds = (interval['start_s'], interval['end_s'])
        def nearest(db, table, fields, prefix='', args=()):
            rows = []
            for comparison, order in (('<=','DESC'),('>','ASC')):
                row = db.execute(f'SELECT {fields} FROM {table} WHERE {prefix}t>=? AND t<? AND t{comparison}? ORDER BY t {order} LIMIT 1', (*args,*bounds,stamp)).fetchone()
                if row is not None:
                    rows.append(row)
            return min(rows, key=lambda r: abs(r[0]-stamp)) if rows else None
        db = sqlite3.connect(file.resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
        try:
            frames = []
            for camera in meta['cameras']:
                row = nearest(db, 'images', 't,png', 'camera=? AND ', (camera['name'],))
                frames.append(dict(name=camera['name'], rotation=camera.get('rotation',0),
                    png=None if row is None or abs(row[0]-stamp)>.1 else row[1],
                    offset_ms=None if row is None else (row[0]-stamp)*1000))
            row = nearest(db, 'telemetry', 't,payload')
            sample = json.loads(row[1]) if row and abs(row[0]-stamp)<=.1 else None
            return dict(frames=frames, sample=sample, segment=segment+1, segments=len(meta['capture_intervals']))
        finally:
            db.close()
