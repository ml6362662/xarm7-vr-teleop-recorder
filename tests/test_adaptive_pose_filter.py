"""Deterministic, offline-only input and FakeArm integration checks."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quest_pose_filter import AdaptivePoseFilter, filter_settings
from quest_pose_math import quaternion_wxyz, orientation_rotation
from quest_relative_motion import settings, rotation_distance
from quest_pose_stream import inspect_packet
from quest_recording_stream import recording_sample
from test_relative_motion import Clock, FakeArm, state
from xarm_quest_relative import RelativeArm


FRAME = ('unitree', 'offline-test')
IDENTITY = [1., 0., 0., 0.]


def quat(degrees, axis='z'):
    return quaternion_wxyz(Rotation.from_euler(axis, degrees, degrees=True))


@pytest.mark.parametrize('bad', [dict(enabled=1), dict(position_min_cutoff_hz=0),
    dict(position_beta_hz_per_mm_s=-1), dict(rotation_max_cutoff_hz=float('nan')),
    dict(derivative_cutoff_hz=True), dict(rotation_min_cutoff_hz=20, rotation_max_cutoff_hz=10),
    dict(typo=1), []])
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        filter_settings(bad)


def test_first_pose_and_repeated_snapshot_do_not_walk_filter():
    f = AdaptivePoseFilter()
    p, q = f.update([.2, 1, .3], quat(179), 10., FRAME)
    assert p == pytest.approx([.2, 1, .3])
    assert rotation_distance(q, quat(179)) < 1e-12
    p, q = f.update([.21, 1, .3], quat(181), 10.01, FRAME)
    for _ in range(100):
        same, same_q = f.update([.21, 1, .3], -quat(181), 10.01, FRAME)
        assert same == pytest.approx(p)
        assert rotation_distance(same_q, q) < 1e-12
    assert f.updates == 2


def test_identical_timestamp_with_different_pose_is_rejected():
    f = AdaptivePoseFilter()
    f.update([.2, 1, .3], IDENTITY, 1, FRAME)
    with pytest.raises(ValueError, match='同一采样'):
        f.update([.21, 1, .3], IDENTITY, 1, FRAME)


@pytest.mark.parametrize('dt', [-.001, .251, float('nan'), float('inf')])
def test_bad_time_never_advances_state(dt):
    f = AdaptivePoseFilter()
    f.update([.2, 1, .3], IDENTITY, 10, FRAME)
    with pytest.raises(ValueError):
        f.update([.21, 1, .3], IDENTITY, 10+dt, FRAME)
    assert f.updates == 1


def test_session_change_requires_reset_and_reset_has_no_tail():
    f = AdaptivePoseFilter()
    f.update([.2, 1, .3], IDENTITY, 10, FRAME)
    with pytest.raises(ValueError, match='参考坐标'):
        f.update([.3, 1, .3], quat(5), 10.01, ('unitree', 'new'))
    f.reset()
    p, q = f.update([.3, 1, .3], quat(5), 11, ('unitree', 'new'))
    assert p == pytest.approx([.3, 1, .3])
    assert rotation_distance(q, quat(5)) < 1e-12


@pytest.mark.parametrize('hz', [90, 100, 120])
def test_stationary_jitter_reduced_in_position_and_orientation(hz):
    f = AdaptivePoseFilter()
    raw, output, raw_angle, output_angle = [], [], [], []
    for i in range(hz*4):
        t = i/hz
        noise_mm = .5*np.sin(2*np.pi*11*t)
        noise_deg = .3*np.sin(2*np.pi*13*t)
        p, q = f.update([.2+noise_mm/1000, 1, .3], quat(noise_deg), 10+t, FRAME)
        if t > 1:
            raw.append(noise_mm); output.append((p[0]-.2)*1000)
            raw_angle.append(noise_deg)
            output_angle.append(np.rad2deg(orientation_rotation(q).as_rotvec()[2]))
    assert np.std(output) < np.std(raw)*.3
    assert np.std(output_angle) < np.std(raw_angle)*.3


def test_adaptation_reduces_fast_motion_lag_without_coupling_rotation():
    adaptive = AdaptivePoseFilter()
    fixed = AdaptivePoseFilter(dict(position_beta_hz_per_mm_s=0))
    for i in range(201):
        t = i*.01
        p = [.2+.1*t, 1, .3]  # 100 mm/s hand motion
        a, _ = adaptive.update(p, IDENTITY, 10+t, FRAME)
        b, _ = fixed.update(p, IDENTITY, 10+t, FRAME)
    assert p[0]-a[0] < (p[0]-b[0])*.3
    assert adaptive.feedback['position_cutoff_hz'] == pytest.approx(10, rel=.001)
    assert adaptive.feedback['rotation_cutoff_hz'] == 2
    assert a[0] <= p[0]


def test_slow_fine_motion_is_not_a_deadband_and_eventually_reaches_endpoint():
    f = AdaptivePoseFilter()
    for i in range(401):
        t = i*.01
        # 1 mm and 0.5 degrees over 2 s, then hold for 2 s.
        progress = min(t/2, 1)
        p, q = f.update([.2+.001*progress, 1, .3], quat(.5*progress), 10+t, FRAME)
        if i == 100:
            assert p[0] > .2004
    assert p[0] == pytest.approx(.201, abs=1e-9)
    assert rotation_distance(q, quat(.5)) < 1e-8


def test_rotation_crosses_180_and_360_without_euler_or_sign_jumps():
    f = AdaptivePoseFilter()
    previous = None
    for i in range(501):
        raw = quat(170+i*.5)
        if i % 2: raw = -raw
        _, q = f.update([.2, 1, .3], raw, 10+i*.01, FRAME)
        assert np.linalg.norm(q) == pytest.approx(1)
        if previous is not None:
            # Adaptive bandwidth can catch up slightly faster than the input
            # during acceleration. It must not create a seam-sized jump.
            assert np.rad2deg(rotation_distance(q, previous)) < .75
            assert (orientation_rotation(q)*orientation_rotation(previous).inv()).as_rotvec()[2] > 0
        assert np.rad2deg(rotation_distance(q, raw)) < 2
        previous = q
    assert f.feedback['position_cutoff_hz'] == 2


def test_filter_is_equivariant_under_fixed_rotation_of_input_axes():
    a, b = AdaptivePoseFilter(), AdaptivePoseFilter()
    basis = Rotation.from_euler('xyz', [.4, -.5, .2])
    for i in range(101):
        p = np.array([.2+i*.0002, 1, .3])
        r = Rotation.from_rotvec([.001*i, .0003*i, -.0002*i])
        pa, qa = a.update(p, quaternion_wxyz(r), 10+i*.01, FRAME)
        pb, qb = b.update(basis.apply(p), quaternion_wxyz(basis*r), 10+i*.01, FRAME)
        assert pb == pytest.approx(basis.apply(pa), abs=1e-12)
        assert rotation_distance(qb, quaternion_wxyz(basis*orientation_rotation(qa))) < 1e-12


def make_controller(empirical=False):
    clock, arm = Clock(), FakeArm()
    cfg = dict(input_filter={'enabled': True}, translation_scale=1, speed_mm_s=100,
               acceleration_mm_s2=600, gripper_button='b')
    if empirical:
        cfg.update(empirical_matrix=np.eye(6).tolist(), empirical_frame=list(FRAME))
    c = RelativeArm(arm, [-3]*7, [3]*7, clock, control_settings=cfg)
    return clock, arm, c


def feed(clock, c, pressed=True, frames=1, **changes):
    for _ in range(frames):
        clock.now += .01
        payload = dict(state(count=c.last_count+1), trigger_pressed=pressed, b_pressed=False,
                       sample_monotonic_s=clock(), input_backend=FRAME[0], frame_id=FRAME[1])
        payload.update(changes)
        c.update_source(payload)
        c.command('heartbeat')
        c.tick()


def start(clock, c):
    feed(clock, c, False)
    c.command('allow_controller')
    feed(clock, c, False, frames=5)
    feed(clock, c, True, frames=8)
    assert c.servo_started


@pytest.mark.parametrize('empirical', [False, True])
def test_filtered_controller_small_move_then_clutch_has_no_old_tail(empirical):
    clock, arm, c = make_controller(empirical)
    start(clock, c)
    feed(clock, c, position_m=[.2,1,.302])
    info = c.input_filter.snapshot()
    assert .3 < info['filtered_position_m'][2] < .302
    expected = c.mapper.matrix @ [0,0,2.] if not empirical else np.array([0,0,2.])
    feed(clock, c, frames=250, position_m=[.2,1,.302])
    assert arm.tcp[:3] == pytest.approx(arm.home[:3]+expected, abs=.02)
    feed(clock, c, position_m=[.2,1,.31])
    feed(clock, c, False, position_m=[.2,1,.31])
    assert not c.servo_started and c.input_filter.sample_time is None
    stopped = arm.tcp.copy()
    feed(clock, c, False, frames=30, position_m=[.4,1,.5])
    feed(clock, c, True, frames=60, position_m=[.4,1,.5])
    assert c.servo_started and c.mode == 'motion'
    assert arm.tcp == pytest.approx(stopped, abs=1e-6)


@pytest.mark.parametrize('fault', ['release', 'invalid', 'heartbeat', 'raw_jump',
                                   'raw_angle_jump', 'session', 'time_reverse', 'time_missing', 'time_stale'])
def test_filter_cannot_delay_stop_or_hide_bad_raw_input(fault):
    clock, arm, c = make_controller()
    start(clock, c)
    feed(clock, c, position_m=[.2,1,.31])
    count = c.command_count
    changes = dict(position_m=[.2,1,.31])
    if fault == 'release': changes['trigger_pressed'] = False
    if fault == 'invalid': changes.update(valid=False, invalid_count=1)
    if fault == 'raw_jump': changes['position_m'] = [.8,1,.31]
    if fault == 'raw_angle_jump': changes['quaternion_wxyz'] = quat(60)
    if fault == 'session': changes['frame_id'] = 'new-session'
    if fault == 'time_reverse': changes['sample_monotonic_s'] = c.input_filter.sample_time-.001
    if fault == 'time_missing': changes['sample_monotonic_s'] = None
    if fault == 'time_stale': changes['sample_monotonic_s'] = clock()-.3
    if fault == 'heartbeat':
        clock.now += 1
        c.tick()
    else:
        feed(clock, c, **changes)
    assert not c.servo_started and arm.state == 4
    assert c.command_count == count
    assert c.input_filter.sample_time is None


def test_duplicate_snapshot_is_only_one_measurement_and_raw_recording_is_preserved():
    clock, arm, c = make_controller()
    start(clock, c)
    feed(clock, c, position_m=[.2,1,.31])
    original = c.input_filter.snapshot()
    for _ in range(5):
        clock.now += .01
        c.command('heartbeat')
        c.update_source(dict(c.source, valid_age_s=clock()-c.source['sample_monotonic_s']))
        c.tick()
    assert c.input_filter.snapshot() == original
    assert c.source['position_m'] == [.2,1,.31]
    sample = recording_sample(c)
    assert sample['hand']['position_m'] == [.2,1,.31]
    assert sample['input_filter']['filtered_position_m'][2] < .31
    assert sample['input_filter']['valid']
    json.dumps(sample, allow_nan=False)


def test_trigger_release_on_duplicate_pose_bypasses_filter():
    clock, arm, c = make_controller()
    start(clock, c)
    feed(clock, c, position_m=[.2,1,.31])
    source = dict(c.source, trigger_pressed=False)
    count = c.command_count
    clock.now += .01
    c.update_source(source)
    c.tick()
    assert not c.servo_started and c.command_count == count
    assert arm.state == 4


def test_input_filter_code_is_in_recording_provenance():
    from quest_recorder import code_hashes
    import hashlib
    import quest_pose_filter
    from pathlib import Path
    assert code_hashes()['quest_pose_filter.py'] == hashlib.sha256(Path(quest_pose_filter.__file__).read_bytes()).hexdigest()


def test_configure_preserves_filter_tuning_and_cancels_pending_motion():
    clock, arm, c = make_controller()
    start(clock, c)
    feed(clock, c, position_m=[.2,1,.31])
    c.command('configure', dict(c.settings, input_filter={'position_min_cutoff_hz': 3.}))
    assert not c.servo_started
    assert c.settings['input_filter']['enabled']
    assert c.settings['input_filter']['position_min_cutoff_hz'] == 3
    c.command('configure', {'input_filter': {'enabled': False}})
    assert not c.settings['input_filter']['enabled']
    assert c.settings['input_filter']['position_min_cutoff_hz'] == 3


def test_webxr_parser_keeps_acquisition_time_not_publication_time(monkeypatch):
    import quest_pose_stream
    monkeypatch.setattr(quest_pose_stream.time, 'monotonic', lambda: 10.02)
    packet = dict(type='quest_unitree_v1', valid=True, position_m=[.2,1,.3],
                  quaternion_wxyz=IDENTITY, frame_id='test', sample_age_s=.02,
                  sample_server_s=10., trigger_pressed=True, a_pressed=False, b_pressed=False,
                  release_counters=dict(trigger=0, a=0, b=0))
    parsed = inspect_packet(json.dumps(packet).encode())
    assert parsed['valid']
    assert parsed['sample_monotonic_s'] == 10.
    assert parsed['sample_time_source'] == 'synchronized_headset'


def test_saved_candidate_enables_filter_without_changing_existing_tuning():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'relative_control_settings.json').read_text())
    assert settings(config)['input_filter']['enabled']
    # Defaults alone preserve old integrations, while this installation opts in.
    assert not settings({})['input_filter']['enabled']


def test_source_clock_filters_continuously_across_server_clock_corrections():
    clock, arm, c = make_controller()
    # Use the real takeover/clutch flow with independent browser clock.
    def sample(pressed=True, correction=0., **changes):
        feed(clock, c, pressed, sample_headset_s=100+clock()+.01,
             sample_monotonic_s=clock()+.01-correction, **changes)
    sample(False); c.command('allow_controller')
    for _ in range(5): sample(False)
    for _ in range(8): sample(True)
    assert c.servo_started
    previous = c.input_filter.sample_time
    for i in range(200):
        # Sync jitter greater than frame interval used to reverse filter dt.
        sample(correction=.08 if i%2 else .005, position_m=[.2,1,.3+i*.00001])
        assert c.servo_started
        assert c.input_filter.sample_time > previous
        previous = c.input_filter.sample_time
        assert c.input_filter.feedback['sample_dt_s'] == pytest.approx(.01)
    assert c.input_filter.feedback['sample_time_source'] == 'headset_monotonic'
    recorded = recording_sample(c)
    assert recorded['hand']['sample_headset_s'] == c.source['sample_headset_s']
    assert recorded['hand']['sample_monotonic_s'] == c.source['sample_monotonic_s']
    assert recorded['input_filter']['sample_server_monotonic_s'] == c.source['sample_monotonic_s']
    commands = c.command_count
    sample(False)
    assert not c.servo_started and c.command_count == commands and arm.state == 4


@pytest.mark.parametrize('fault', ['expired', 'future', 'missing_server', 'reversed_headset',
                                   'invalid', 'pause', 'raw_jump', 'session'])
def test_headset_filter_clock_cannot_hide_expiry_or_tracking_loss(fault):
    clock, arm, c = make_controller()
    feed(clock, c, False, sample_headset_s=100.)
    c.command('allow_controller')
    for i in range(15):
        feed(clock, c, i>=5, sample_headset_s=100.+(i+1)*.01)
    assert c.servo_started
    changes = dict(sample_headset_s=100.16)
    if fault == 'expired': changes['sample_monotonic_s'] = clock()-.3
    if fault == 'future': changes['sample_monotonic_s'] = clock()+.1
    if fault == 'missing_server': changes['sample_monotonic_s'] = None
    if fault == 'reversed_headset': changes['sample_headset_s'] = 100.14
    if fault == 'invalid': changes.update(valid=False, invalid_count=1)
    if fault == 'pause': changes.update(motion_enabled=False, pause_generation=1)
    if fault == 'raw_jump': changes['position_m'] = [.8,1,.3]
    if fault == 'session': changes['frame_id'] = 'new-session'
    commands = c.command_count
    feed(clock, c, **changes)
    assert not c.servo_started and arm.state == 4
    assert c.command_count == commands


@pytest.mark.parametrize('headset_clock', [False, True])
@pytest.mark.parametrize('initialization_s', [.1, .26, .35, .55])
def test_single_press_after_slow_hold_initialization_reanchors_without_jump(headset_clock, initialization_s):
    clock, arm, c = make_controller()
    original = arm.set_servo_angle
    calls = []
    def slow_hold(angle, **kwargs):
        result = original(angle, **kwargs)
        if kwargs.get('wait'):
            calls.append('hold')
            clock.now += initialization_s
            # Real SDK checks poll the latest subscriber frame after waiting.
            fresh = dict(c.source, sample_monotonic_s=clock(), packet_count=c.last_count+1,
                         position_m=[.23,1,.32], quaternion_wxyz=quat(10))
            if headset_clock:
                fresh['sample_headset_s'] = 100+clock()
            c.update_source(fresh)
        return result
    arm.set_servo_angle = slow_hold
    def frame(pressed, **changes):
        if headset_clock:
            changes['sample_headset_s'] = 100+clock()+.01
        feed(clock, c, pressed, **changes)
    frame(False)
    c.command('allow_controller')
    frame(False)
    before = arm.tcp.copy()
    # One rising edge and a continuous hold; no second click/release generation.
    for _ in range(6):
        frame(True, position_m=[.23,1,.32] if calls else [.2,1,.3],
              quaternion_wxyz=quat(10) if calls else IDENTITY)
    assert calls == ['hold']
    assert c.servo_started and c.mode == 'motion'
    assert c.mapper.hand_anchor == pytest.approx([.23,1,.32])
    assert arm.tcp == pytest.approx(before, abs=1e-6)
    assert c.input_filter.sample_time is not None
    frame(False, position_m=[.23,1,.32], quaternion_wxyz=quat(10))
    assert not c.servo_started and arm.state == 4


@pytest.mark.parametrize('fault', ['stale', 'released', 'missed_release', 'invalid', 'pause',
                                  'session', 'time_reversed', 'raw_jump', 'heartbeat'])
def test_slow_initialization_reanchor_still_rejects_unsafe_input(fault):
    clock, arm, c = make_controller()
    original = arm.set_servo_angle
    def slow_hold(angle, **kwargs):
        result = original(angle, **kwargs)
        if kwargs.get('wait'):
            old = dict(c.source)
            clock.now += .35 if fault!='heartbeat' else 1.
            fresh = dict(old, sample_monotonic_s=clock(), packet_count=c.last_count+1)
            if fault=='stale': fresh.update(sample_monotonic_s=clock()-.3, valid_age_s=.3)
            if fault=='released': fresh['trigger_pressed']=False
            if fault=='missed_release': fresh['trigger_release_generation']=1
            if fault=='invalid': fresh.update(valid=False,invalid_count=1)
            if fault=='pause': fresh.update(motion_enabled=False,pause_generation=1)
            if fault=='session': fresh['frame_id']='changed-during-hold'
            if fault=='time_reversed': fresh['sample_headset_s']=100.+old['sample_monotonic_s']-.001
            else: fresh['sample_headset_s']=100.+clock()
            if fault=='raw_jump': fresh['position_m']=[.8,1,.3]
            c.update_source(fresh)
        return result
    arm.set_servo_angle=slow_hold
    feed(clock,c,False,sample_headset_s=100.+clock()+.01)
    c.command('allow_controller')
    for _ in range(4):
        feed(clock,c,True,sample_headset_s=100.+clock()+.01)
    assert not c.servo_started and arm.state==4
    assert c.command_count==0


def test_reanchor_only_exempts_positive_gap_not_reversal_or_frame_conflicts():
    f=AdaptivePoseFilter()
    f.update([.2,1,.3],IDENTITY,10.,FRAME)
    with pytest.raises(ValueError,match='倒退或中断'):
        f.update([.2,1,.3],IDENTITY,10.35,FRAME)
    with pytest.raises(ValueError,match='倒退或中断'):
        f.update([.2,1,.3],IDENTITY,9.99,FRAME,reanchor=True)
    with pytest.raises(ValueError,match='参考坐标'):
        f.update([.2,1,.3],IDENTITY,10.35,('unitree','changed'),reanchor=True)
    with pytest.raises(ValueError,match='同一采样'):
        f.update([.21,1,.3],IDENTITY,10.,FRAME,reanchor=True)
    p,q=f.update([.21,1,.3],quat(10),10.35,FRAME,reanchor=True)
    assert p==pytest.approx([.21,1,.3]) and rotation_distance(q,quat(10))<1e-12
    assert f.feedback['sample_dt_s'] is None and f.updates==1
