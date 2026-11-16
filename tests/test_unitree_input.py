import copy
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quest_unitree_input import UnitreeInput
from quest_pose_stream import inspect_packet, merge_release_counters
import json
from pathlib import Path


def event(now=None, sequence=1):
    matrix = np.eye(4)
    matrix[:3, 3] = [.2, 1., -.3]
    return {'right': matrix.flatten(order='F').tolist(),
            'rightState': {'trigger': False, 'aButton': False, 'bButton': False},
            'xarmMeta': {'valid': True, 'frame_id': 'unitree-test', 'sequence': sequence,
                         'sample_server_s': time.monotonic() if now is None else now,
                         'buttons': dict(trigger=False, a=False, b=False),
                         'release_counters': dict(trigger=0, a=0, b=0)}}


def test_coordinate_conversion_and_source_label():
    value = event()
    matrix = np.array(value['right']).reshape(4, 4, order='F')
    matrix[:3, :3] = Rotation.from_euler('xyz', [20, 35, -42], degrees=True).as_matrix()
    value['right'] = matrix.flatten(order='F').tolist()
    packet = UnitreeInput().decode(value)
    assert packet['valid']
    np.testing.assert_allclose(packet['position_m'], [.2, 1., .3])
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    np.testing.assert_allclose(packet['quaternion_wxyz'], [q[3], -q[0], -q[1], q[2]])
    assert inspect_packet(json.dumps(packet).encode())['input_backend'] == 'unitree'


def test_translation_does_not_rotate():
    adapter = UnitreeInput()
    first = adapter.decode(event())
    value = event(sequence=2)
    value['right'][12] += .5
    second = adapter.decode(value)
    assert first['quaternion_wxyz'] == second['quaternion_wxyz']
    assert second['position_m'][0] == .7


def test_pinned_client_changes_only_transport_and_input_quality():
    from quest_unitree_bridge import CHUNK, patch_client
    root = Path(__file__).resolve().parents[1]
    raw = (root/'third_party/vuer-0.0.60/vuer/client_build'/CHUNK).read_bytes()
    patched = patch_client(raw)
    assert 'window.xarmUnitreeMeta(Ae,fe,le)' in patched
    assert 'createXRStore({hand:!1,handTracking:!1,emulate:!1})' in patched
    assert 'children:"Virtual Reality"' not in patched
    assert 'children:"进入遥操（Pass-through）"' in patched
    assert 'ie&&window.xarmUnitreeSend({' in patched
    assert '(window.location.origin+"/unitree-profiles/")' in patched
    assert 'function getTargetRayTransform' in patched
    with pytest.raises(RuntimeError):
        patch_client(raw+b'changed upstream version')


@pytest.mark.parametrize('change', [
    lambda e: e.pop('xarmMeta'),
    lambda e: e['xarmMeta'].update(valid=False),
    lambda e: e['xarmMeta'].update(sample_server_s=time.monotonic()-.25),
    lambda e: e['xarmMeta'].update(sample_server_s=float('nan')),
    lambda e: e['right'].__setitem__(0, 2.),
    lambda e: e['right'].__setitem__(12, float('nan')),
    lambda e: e['rightState'].pop('bButton'),
    lambda e: e['rightState'].update(bButton=True),
    lambda e: e['rightState'].update(trigger=1),
    lambda e: e['xarmMeta']['release_counters'].update(b=-1),
])
def test_invalid_data_fail_closed(change):
    value = event()
    change(value)
    assert not UnitreeInput().decode(value)['valid']


def test_sequence_and_session_lease():
    adapter = UnitreeInput(clock=lambda: 10.)
    value = event(10.)
    assert adapter.decode(value)['valid']
    assert not adapter.decode(value)['valid']
    value['xarmMeta']['frame_id'] = 'competing-session'
    assert not adapter.decode(value)['valid']


def test_independent_buttons_and_release_generation():
    adapter = UnitreeInput()
    previous, state = {}, {'invalid_count': 0}
    value = event()
    value['rightState']['bButton'] = value['xarmMeta']['buttons']['b'] = True
    packet = adapter.decode(value)
    assert packet['b_pressed'] and not packet['trigger_pressed']
    merge_release_counters(state, inspect_packet(json.dumps(packet).encode()), previous)
    value = event(sequence=3)  # skipped frame still preserves releases
    value['xarmMeta']['release_counters']['b'] = 1
    packet = adapter.decode(value)
    merge_release_counters(state, inspect_packet(json.dumps(packet).encode()), previous)
    assert state['b_release_generation'] == 1


def test_diagnostics_capture_brief_invalid_frame_even_after_recovery():
    now = [10.]
    adapter = UnitreeInput(clock=lambda: now[0])
    assert adapter.decode(event(10., 1))['valid']
    now[0] += .01
    bad = event(now[0], 2)
    bad['xarmMeta'].update(valid=False, reason='未知按键映射：generic-trigger')
    assert not adapter.decode(bad)['valid']
    now[0] += .01
    good = event(now[0], 3)
    good['xarmMeta']['clock'] = dict(age_ms=1020, rtt_ms=8, consecutive_failures=1,
                                   total_failures=2, last_error='clock RTT >100ms')
    assert adapter.decode(good)['valid']
    now[0] += .3
    info = adapter.diagnostics()
    assert info['valid_packets'] == 2 and info['invalid_packets'] == 1
    assert info['invalid_reasons']['宇树输入无效：未知按键映射：generic-trigger'] == 1
    assert info['receive_age_ms'] == pytest.approx(300)
    assert info['max_receive_gap_ms'] == pytest.approx(10)
    assert [e['valid'] for e in info['recent_transitions']] == [True, False, True]
    assert info['clock']['consecutive_failures'] == 1
    json.dumps(info, allow_nan=False)


def test_diagnostics_do_not_accept_malformed_optional_clock_metadata():
    adapter = UnitreeInput(clock=lambda: 10.)
    value = event(10.)
    value['xarmMeta']['clock'] = dict(age_ms=float('nan'), rtt_ms='bad',
                                    consecutive_failures=-1, total_failures=True)
    assert adapter.decode(value)['valid']
    assert adapter.diagnostics()['clock'] == {}
    for count in range(2, 50):
        value = event(10., count)
        value['xarmMeta'].update(valid=count % 2 == 0, reason='XR 非前台')
        adapter.decode(value)
    assert len(adapter.diagnostics()['recent_transitions']) == 20


def test_clock_fallback_diagnostics_keep_websocket_failure_visible():
    adapter = UnitreeInput(clock=lambda: 10.)
    value = event(10.)
    value['xarmMeta']['clock'] = dict(transport='http', ws_failures=50, http_failures=0,
        ws_last_error='clock connection timeout', http_last_error='', rtt_ms=8)
    assert adapter.decode(value)['valid']
    clock = adapter.diagnostics()['clock']
    assert clock['transport']=='http' and clock['ws_failures']==50
    assert clock['ws_last_error']=='clock connection timeout'


def timed_event(server, headset, sequence=1):
    value = event(server, sequence)
    value['xarmMeta'].update(client_version='input-timing-v5', sample_headset_s=headset)
    return value


def test_headset_clock_survives_server_offset_change_and_stream(monkeypatch):
    now = [10.]
    adapter = UnitreeInput(clock=lambda: now[0])
    assert adapter.decode(timed_event(9.99, 100.))['valid']
    now[0] = 10.01
    # A better sync estimate revises server time backwards by 60 ms.
    packet = adapter.decode(timed_event(9.93, 100.01, 2))
    assert packet['valid']
    assert packet['sample_age_s'] == pytest.approx(.08)
    import quest_pose_stream
    monkeypatch.setattr(quest_pose_stream.time, 'monotonic', lambda: now[0])
    parsed = inspect_packet(json.dumps(packet).encode())
    assert parsed['valid'] and parsed['sample_headset_s'] == 100.01
    assert parsed['sample_monotonic_s'] == 9.93
    assert parsed['sample_time_source'] == 'synchronized_headset_earliest'
    now[0] = 10.14
    assert not inspect_packet(json.dumps(packet).encode())['valid'], 'transport queue age still enforced'


@pytest.mark.parametrize('stamp', [None, True, '1', -1, float('nan'), float('inf')])
def test_new_client_requires_valid_headset_clock(stamp):
    assert not UnitreeInput(clock=lambda: 10.).decode(timed_event(10., stamp))['valid']


def test_new_client_missing_headset_clock_is_not_legacy_fallback():
    value = timed_event(10., 100.)
    value['xarmMeta'].pop('sample_headset_s')
    assert not UnitreeInput(clock=lambda: 10.).decode(value)['valid']


def test_reversed_headset_clock_and_real_expiry_still_fail_closed():
    adapter = UnitreeInput(clock=lambda: 10.)
    assert adapter.decode(timed_event(10., 100.))['valid']
    for seq, stamp in enumerate((100., 99.), 2):
        assert not adapter.decode(timed_event(10., stamp, seq))['valid']
    old = adapter.decode(timed_event(9.81, 101., 4))
    assert not old['valid'] and '180 ms' in old['reason']
    assert adapter.diagnostics()['timing']['sample_age_ms'] == pytest.approx(190)
    future = adapter.decode(timed_event(10.04, 102., 5))
    assert not future['valid'] and '超前' in future['reason']
    assert adapter.diagnostics()['timing']['sample_age_ms'] == pytest.approx(-40)


def test_short_tracking_fault_survives_latest_packet_replacement():
    now = [10.]
    adapter = UnitreeInput(clock=lambda: now[0])
    old, state = {}, {'invalid_count': 0}
    good = timed_event(10., 100.)
    good['xarmMeta'].update(client_version='realtime-input-v6', input_interruptions=0)
    first = adapter.decode(good)
    merge_release_counters(state, dict(first, input_backend='unitree',
        remote_input_interruptions=first['input_interruptions'], remote_release_counters=first['release_counters']), old)
    # No invalid packet was sent while the uplink was full. Client counter
    # carries the interruption into the next VALID pose, preventing old goals.
    now[0] += .01
    good['xarmMeta'].update(sequence=2, sample_server_s=now[0], sample_headset_s=100.01,
        input_interruptions=1, last_input_error='tracking lost between transmitted frames')
    second = adapter.decode(good)
    assert second['valid']
    merge_release_counters(state, dict(second, input_backend='unitree',
        remote_input_interruptions=second['input_interruptions'], remote_input_error=second['last_input_error'],
        remote_release_counters=second['release_counters']), old)
    assert state['invalid_count']==1
    assert 'tracking lost' in state['last_invalid_reason']


def test_server_rejected_packet_is_latched_into_next_valid_packet():
    adapter=UnitreeInput(clock=lambda: 10.)
    assert adapter.decode(event(10.,1))['input_interruptions']==0
    assert not adapter.decode(event(9.,2))['valid']
    packet=adapter.decode(event(10.,3))
    assert packet['valid'] and packet['input_interruptions']==1
    assert '180 ms' in packet['last_input_error']
