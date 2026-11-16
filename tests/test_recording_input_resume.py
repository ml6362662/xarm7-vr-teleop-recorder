"""Offline input interruptions: fresh re-clutch, boundaries, diagnostics and faults."""
import json

import pytest

from test_recorder_runtime import prepared_recorder
from test_recording_gate import active_sample


def running(tmp_path):
    recorder, start = prepared_recorder(tmp_path)
    recorder.ingest_sample(active_sample(5, start+.05,
        hand=dict(trigger_pressed=True, trigger_release_generation=1),
        input_status=dict(valid=True, invalid_count=0, pause_generation=0)))
    return recorder, start


def interruption(recorder, start, **changes):
    item = active_sample(6, start+.06, source_valid=False, active=False,
        command_valid=False, pause_settling=True,
        hand=dict(trigger_pressed=True, trigger_release_generation=1),
        input_status=dict(valid=False, reason='等待电脑时钟同步', invalid_count=1,
                          pause_generation=0))
    item.update(changes)
    recorder.ingest_sample(item)


def healthy(i, start, *, pressed=True, generation=1, **changes):
    return active_sample(i, start+i*.01, pressed=pressed,
        hand=dict(trigger_pressed=pressed, trigger_release_generation=generation),
        input_status=dict(valid=True, invalid_count=1, pause_generation=0), **changes)


def test_clock_loss_requires_new_release_and_preserves_first_cause(tmp_path):
    r, start = running(tmp_path)
    episode = r.episode
    count = episode.samples
    interruption(r, start)
    interruption(r, start, sequence=7, monotonic_s=start+.07,
                 input_status=dict(valid=False, reason='后来又丢失追踪', invalid_count=2))
    assert len(episode.metadata['input_interruptions']) == 1
    assert r.status()['last_input_pause']['reason'] == '等待电脑时钟同步'
    assert r.gate.current is None
    # A release counter changed during the outage does not authorize capture.
    r.ingest_sample(healthy(8, start, generation=4))
    r.ingest_sample(healthy(9, start, generation=4))
    assert episode.samples == count
    r.ingest_sample(healthy(10, start, pressed=False, generation=5))
    r.ingest_sample(healthy(11, start, generation=5, pause_settling=True))
    assert episode.samples == count
    r.ingest_sample(healthy(12, start, generation=5))
    r.ingest_sample(healthy(13, start, generation=5))
    assert r.episode is episode and episode.samples == count+2
    assert r.status()['capture_state'] == 'recording'
    assert r.input_pause['resumed_s'] == start+.12
    assert r.sequence_gaps == 0
    assert r.gate.segment_at(start+.06) is None
    r.ingest_sample(healthy(14, start, pressed=False, generation=6))
    r.handle(dict(command='save', success=True))
    meta = json.loads((episode.path/'episode.json').read_text())
    assert meta['status'] == 'complete'
    assert meta['input_interruptions'][0]['reason'] == '等待电脑时钟同步'
    assert meta['input_interruptions'][0]['resumed_s'] == start+.12


def test_delayed_camera_frames_from_outage_are_excluded(tmp_path):
    r, start = running(tmp_path)
    interruption(r, start)
    r.ingest_sample(healthy(20, start, pressed=False, generation=2))
    r.ingest_sample(healthy(21, start, generation=2))
    r.ingest_sample(healthy(22, start, generation=2))
    for seq, stamp in enumerate([start+.05, start+.055, start+.1, start+.215]):
        r.camera_queue.put(dict(camera='global', sequence=seq, t=stamp,
            read_start=stamp-.001, width=1, height=1, png=b'dummy'))
    r.drain_images()
    rows = list(r.episode.db.execute('SELECT t FROM images ORDER BY t'))
    assert [t for t, in rows] == [start+.05, start+.215]
    assert r.camera_gaps == 0
    r.finish('incomplete', reason='test cleanup')


def test_hidden_invalid_packet_still_splits_capture(tmp_path):
    r, start = running(tmp_path)
    interruption(r, start, source_valid=True,
        input_status=dict(valid=True, invalid_count=1, pause_generation=0,
                          reason='有效位姿', last_invalid_reason='数据过期'))
    assert r.awaiting_input_resume and r.gate.current is None
    assert r.input_pause['reason'] == '数据过期'
    r.finish('incomplete', reason='test cleanup')


def test_recovered_release_counter_allows_new_press_only_after_baseline(tmp_path):
    r, start = running(tmp_path)
    interruption(r, start)
    r.ingest_sample(healthy(7, start, generation=4))
    assert r.gate.current is None
    # A release/repress after recovery can occur between snapshots.
    r.ingest_sample(healthy(8, start, generation=5))
    assert r.gate.current is not None and not r.awaiting_input_resume
    r.finish('incomplete', reason='test cleanup')


@pytest.mark.parametrize('fault,reason', [
    ({'run_id':'new'}, '控制器会话发生变化'),
    ({'settings_revision':99}, '录制中遥操参数发生变化'),
    ({'recovery_active':True}, '机械臂进入复位'),
    ({'error':'SDK error'}, '控制器故障：SDK error'),
    ({'mode':'preview'}, '遥操许可已撤销'),
])
def test_actual_faults_still_end_episode_even_during_input_pause(tmp_path, fault, reason):
    r, start = running(tmp_path)
    interruption(r, start)
    path = r.episode.path
    interruption(r, start, sequence=7, monotonic_s=start+.07, **fault)
    assert r.episode is None
    meta = json.loads((path/'episode.json').read_text())
    assert meta['status']=='incomplete' and reason in meta['reason']


def test_repeated_interruptions_keep_separate_causes(tmp_path):
    r, start = running(tmp_path)
    interruption(r, start)
    r.ingest_sample(healthy(7, start, pressed=False, generation=2))
    r.ingest_sample(healthy(8, start, generation=2))
    interruption(r, start, sequence=9, monotonic_s=start+.09,
                 input_status=dict(valid=False, reason='手柄被遮挡', invalid_count=2))
    assert [i['reason'] for i in r.episode.metadata['input_interruptions']] == [
        '等待电脑时钟同步', '手柄被遮挡']
    r.finish('incomplete', reason='test cleanup')
