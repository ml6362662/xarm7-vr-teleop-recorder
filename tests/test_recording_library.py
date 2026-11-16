"""Data management works on isolated archives, never on the user's recordings."""
import fcntl
import json
from pathlib import Path

import pytest

from recording_data import RAW_VERSION, source_fingerprint
from recording_library import (list_datasets, list_episodes, library_lock, move_episode,
                               preview, source_time)
from test_recording_gate import trigger_episode


def archive(tmp_path):
    root = tmp_path/'dataset'
    path = trigger_episode(root)
    (root/'dataset.json').write_text(json.dumps(dict(schema_version=RAW_VERSION)))
    meta = json.loads((path/'episode.json').read_text())
    meta['recorded_duration_s'] = .61
    (path/'episode.json').write_text(json.dumps(meta))
    return root, path


def test_list_and_reversible_move_preserve_all_raw_bytes(tmp_path):
    root, path = archive(tmp_path)
    fingerprint = source_fingerprint(path)
    assert list_datasets(tmp_path) == ['dataset']
    rows = list_episodes(root)
    assert rows[0]['category']=='success' and rows[0]['duration']==.61
    moved = move_episode(root, path.name)
    assert not path.exists() and not list_episodes(root)
    assert list_episodes(root, trash=True)[0]['id']==path.name
    assert source_fingerprint(moved)==fingerprint
    restored = move_episode(root, path.name, restore=True)
    assert restored==path and source_fingerprint(path)==fingerprint
    assert not list_episodes(root, trash=True)


@pytest.mark.parametrize('lock_name', ['.recorder.lock', '.library.lock'])
def test_active_recorder_or_export_blocks_move(tmp_path, lock_name):
    root, path = archive(tmp_path)
    with (root/lock_name).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError):
            move_episode(root, path.name)
    assert path.is_dir()


def test_export_shared_lock_prevents_management_move(tmp_path):
    root, path = archive(tmp_path)
    with library_lock(root):
        with pytest.raises(ValueError, match='导出'):
            move_episode(root, path.name)
    assert path.is_dir()


def test_restore_collision_never_overwrites(tmp_path):
    root, path = archive(tmp_path)
    moved = move_episode(root, path.name)
    path.mkdir()
    with pytest.raises(ValueError, match='同名'):
        move_episode(root, path.name, restore=True)
    assert moved.is_dir() and list(path.iterdir())==[]


@pytest.mark.parametrize('status', ['recording', 'unknown'])
def test_unfinalized_data_cannot_be_moved_or_previewed(tmp_path, status):
    root, path = archive(tmp_path)
    meta = json.loads((path/'episode.json').read_text())
    meta['status']=status
    (path/'episode.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        move_episode(root, path.name)
    with pytest.raises(ValueError):
        preview(root, path.name)


def test_preview_seeks_across_pause_without_cross_segment_frames(tmp_path):
    root, path = archive(tmp_path)
    meta = json.loads((path/'episode.json').read_text())
    assert source_time(meta, .35)[0]==pytest.approx(100.75)
    before = source_fingerprint(path)
    result = preview(root, path.name, .35)
    assert result['segment']==2 and len(result['frames'])==2
    assert all(f['png'].startswith(b'\x89PNG') for f in result['frames'])
    assert result['sample']['monotonic_s']>=100.7
    assert source_fingerprint(path)==before
    moved = move_episode(root, path.name)
    assert preview(root, path.name, .35, trash=True)['segment']==2
    assert moved.is_dir()


def test_missing_frames_and_invalid_metadata_are_visible(tmp_path):
    root, path = archive(tmp_path)
    import sqlite3
    db=sqlite3.connect(path/'raw.sqlite3')
    db.execute("DELETE FROM images WHERE camera='global'")
    db.commit(); db.close()
    result=preview(root, path.name)
    assert next(f for f in result['frames'] if f['name']=='global')['png'] is None
    (path/'episode.json').write_text('{broken')
    assert list_episodes(root)[0]['category']=='invalid'


@pytest.mark.parametrize('link', ['episodes', '.trash', 'raw.sqlite3', 'episode.json'])
def test_symlinks_never_redirect_management_or_preview(tmp_path, link):
    root, path = archive(tmp_path)
    external=tmp_path/'external';external.mkdir()
    if link in ('episodes','.trash'):
        if link=='episodes':
            (root/'episodes').rename(root/'saved_episodes')
        (root/link).symlink_to(external,target_is_directory=True)
        with pytest.raises(ValueError):
            move_episode(root,path.name)
    else:
        file=path/link
        file.rename(external/link)
        file.symlink_to(external/link)
        with pytest.raises(ValueError):
            preview(root,path.name)
    assert list(external.iterdir()) == ([] if link in ('episodes','.trash') else [external/link])


def test_traversal_rejected_and_wal_not_read_as_immutable(tmp_path):
    root, path=archive(tmp_path)
    with pytest.raises(ValueError):
        move_episode(root,'../external')
    (path/'raw.sqlite3-wal').touch()
    with pytest.raises(ValueError):
        preview(root,path.name)
    with pytest.raises(ValueError):
        move_episode(root,path.name)


def test_trashed_episodes_are_excluded_from_export(tmp_path):
    from export_recording import export_dataset
    root,path=archive(tmp_path)
    move_episode(root,path.name)
    with pytest.raises(ValueError,match='没有可导出'):
        export_dataset(root,tmp_path/'export')


def test_actual_export_holds_source_lock_until_finished(tmp_path, monkeypatch):
    import export_recording
    root, path = archive(tmp_path)
    original = export_recording.V21Exporter.add
    checked = []
    def add(exporter, *args):
        with pytest.raises(ValueError, match='导出'):
            move_episode(root, path.name)
        checked.append(True)
        return original(exporter, *args)
    monkeypatch.setattr(export_recording.V21Exporter, 'add', add)
    export_recording.export_dataset(root, tmp_path/'export')
    assert checked == [True] and path.is_dir()
