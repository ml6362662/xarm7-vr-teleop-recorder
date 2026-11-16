"""Trigger boundaries, delayed camera frames, and intentional pause gaps."""
import json

import pytest

from recording_gate import TriggerCaptureGate
from recording_data import EpisodeWriter, aligned_segments, open_episode
from quest_recorder import Recorder
from test_recording import sample, make_episode


def active_sample(i,t=None,pressed=True,**changes):
    value=sample(i,t)
    value.update(active=pressed,command_valid=pressed,
                 hand=dict(trigger_pressed=pressed,trigger_release_generation=0))
    value.update(changes)
    return value


def test_gate_requires_release_if_armed_while_trigger_held():
    gate=TriggerCaptureGate(True)
    assert gate.update(active_sample(0)) is None
    assert gate.update(active_sample(1,pressed=False)) is None
    assert gate.update(active_sample(2))==0
    assert gate.update(active_sample(3,pressed=False)) is None
    assert gate.update(active_sample(4))==1
    gate.close(100.05)
    assert gate.duration(999)==pytest.approx(.02)
    assert gate.segment_at(100.03) is None
    assert gate.segment_at(100.041)==1


@pytest.mark.parametrize('change',[
    dict(active=False),dict(command_valid=False),dict(pause_settling=True),
    dict(source_valid=False),dict(mode='preview'),dict(recovery_active=True),dict(error='SDK'),
])
def test_gate_excludes_initialization_braking_and_invalid_states(change):
    gate=TriggerCaptureGate()
    assert gate.update(active_sample(0,**change)) is None
    assert not gate.intervals


def test_missed_release_splits_interval_even_when_latest_trigger_is_pressed():
    gate=TriggerCaptureGate()
    assert gate.update(active_sample(0))==0
    assert gate.update(active_sample(1))==0
    assert gate.update(active_sample(5,hand=dict(trigger_pressed=True,trigger_release_generation=1)))==1
    assert gate.segment_at(100.03) is None


def test_release_counter_can_arm_capture_after_a_short_release_repress():
    gate=TriggerCaptureGate(True,7)
    assert gate.update(active_sample(0,hand=dict(trigger_pressed=True,trigger_release_generation=7))) is None
    assert gate.update(active_sample(1,hand=dict(trigger_pressed=True,trigger_release_generation=8)))==0


def test_delayed_images_are_classified_by_acquisition_time(tmp_path):
    recorder=Recorder(tmp_path,dict(cameras=[dict(name='global')]),status_path=tmp_path/'status.json')
    recorder.episode=EpisodeWriter(tmp_path,{})
    recorder.episode.metadata['started_monotonic_s']=99
    recorder.gate=TriggerCaptureGate()
    for item in (active_sample(0),active_sample(10,pressed=False),active_sample(20)):
        recorder.gate.update(item)
        recorder.latest=item
    for i,t in enumerate((100.05,100.15,100.21)):
        recorder.camera_queue.put(dict(camera='global',sequence=i,t=t,read_start=t-.001,
                                       width=1,height=1,png=b'dummy'))
    recorder.drain_images()
    assert recorder.episode.images=={'global':1}
    assert len(recorder.pending_images)==1  # Needs a controller watermark beyond acquisition.
    recorder.latest=active_sample(25)
    recorder.gate.update(recorder.latest)
    recorder.drain_images()
    assert recorder.episode.images=={'global':2}
    assert recorder.camera_gaps==0
    recorder.finish('incomplete')


def trigger_episode(root,drop_inside=False):
    path=make_episode(root)
    import sqlite3
    db=sqlite3.connect(path/'raw.sqlite3')
    db.execute('DELETE FROM telemetry WHERE t>=100.3 AND t<100.7')
    db.execute('DELETE FROM images WHERE t>=100.3 AND t<100.7')
    if drop_inside:
        db.execute('DELETE FROM telemetry WHERE sequence=10')
    for seq,payload in db.execute('SELECT sequence,payload FROM telemetry').fetchall():
        value=json.loads(payload)
        value['capture_segment']=0 if seq<30 else 1
        value.update(active=True,command_valid=True)
        value['hand']['trigger_pressed']=True
        db.execute('UPDATE telemetry SET payload=? WHERE sequence=?',(json.dumps(value),seq))
    db.commit()
    db.close()
    meta=json.loads((path/'episode.json').read_text())
    meta.update(capture_mode='trigger',capture_intervals=[
        dict(segment_id=0,start_s=100,end_s=100.3),dict(segment_id=1,start_s=100.7,end_s=101.01)])
    (path/'episode.json').write_text(json.dumps(meta))
    return path


def test_pause_gaps_export_as_one_task_with_explicit_boundaries(tmp_path):
    from export_recording import export_dataset
    path=trigger_episode(tmp_path/'raw')
    db,meta=open_episode(path)
    try:
        segments=list(aligned_segments(db,meta))
        assert len(segments)==2
        assert {r['capture_segment'] for r in segments[0]}=={0}
        assert {r['capture_segment'] for r in segments[1]}=={1}
    finally:
        db.close()
    export_dataset(tmp_path/'raw',tmp_path/'lerobot')
    info=json.loads((tmp_path/'lerobot'/'meta'/'info.json').read_text())
    assert info['total_episodes']==1
    import pyarrow.parquet as pq
    table=pq.read_table(tmp_path/'lerobot/data/chunk-000/episode_000000.parquet').to_pydict()
    assert set(table['episode_index'])=={0}
    assert set(table['capture.segment_index'])=={0,1}
    assert sum(table['next.discontinuity'])==1
    assert sum(table['next.done'])==1 and table['next.done'][-1]
    assert sum(table['next.success'])==1 and table['next.success'][-1]
    assert table['frame_index']==list(range(len(table['frame_index'])))
    boundary=table['next.discontinuity'].index(True)
    assert table['source.monotonic_s'][boundary+1]-table['source.monotonic_s'][boundary]>.3
    assert table['timestamp'][boundary+1]-table['timestamp'][boundary]==pytest.approx(1/30)
    assert not table['next.done'][boundary] and not table['next.success'][boundary]
    from recording_training import training_indices
    indices=training_indices(tmp_path/'lerobot',horizon=3,history=1)
    assert indices
    for i in indices:
        assert len(set(table['capture.segment_index'][i-1:i+3]))==1
    assert boundary not in indices and boundary+1 not in indices


def test_real_packet_loss_inside_pressed_interval_is_still_rejected(tmp_path):
    path=trigger_episode(tmp_path,drop_inside=True)
    db,meta=open_episode(path)
    try:
        with pytest.raises(ValueError,match='丢包'):
            list(aligned_segments(db,meta))
    finally:
        db.close()
