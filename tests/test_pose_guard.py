import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quest_pose_guard import PoseGuard, joint_acceleration_step, JOINT_ACCELERATION
from test_relative_motion import Clock, FakeArm, trigger_setup, button_frames


@pytest.mark.parametrize('index,expected', [(0,'normal'),(1,'slowing'),(2,'blocked'),(3,'blocked')])
def test_historical_calibrated_fk_detects_risk_before_failure(index, expected):
    fixture = json.loads((Path(__file__).parent/'fixtures/pose_guard_historical_fk.json').read_text())
    sample = fixture['samples'][index]
    q = np.array(sample['joints_rad'])

    class RecordedFK:
        def get_forward_kinematics(self, joints, **kwargs):
            inputs = [q] + [q+np.eye(7)[i]*fixture['perturbation_rad'] for i in range(7)]
            matches = [i for i,v in enumerate(inputs) if np.allclose(v,joints,rtol=0,atol=1e-12)]
            assert len(matches) == 1
            return 0, sample['fk'][matches[0]]

    clock = Clock()
    guard = PoseGuard(RecordedFK(), clock)
    metric = guard.evaluate(q, clock()+.04, force=True)
    assert metric['status'] == expected
    if expected == 'normal':
        assert metric['speed_ratio'] == 1 and metric['sigma_min'] > 35
    elif expected == 'slowing':
        assert 0 < metric['speed_ratio'] < 1
    else:
        assert metric['speed_ratio'] == 0


def test_guard_cache_does_not_hide_warning_candidates_or_nonfinite_fk():
    clock, arm = Clock(), FakeArm()
    count = []
    original = arm.get_forward_kinematics
    def tracked(*a, **kw):
        count.append(1); return original(*a, **kw)
    arm.get_forward_kinematics = tracked
    guard = PoseGuard(arm, clock)
    guard.evaluate(arm.joints, clock()+.04)
    assert len(count) == 8
    guard.evaluate(arm.joints, clock()+.04)
    assert len(count) == 8
    guard.evaluate(arm.joints, clock()+.04, force=True)
    assert len(count) == 16
    arm.get_forward_kinematics = lambda *a, **kw: (0, [float('nan')]*6)
    with pytest.raises(ValueError, match='finite'):
        guard.evaluate(arm.joints, clock()+.04, force=True)


def test_guard_deadline_and_sdk_errors_fail_closed():
    clock, arm = Clock(), FakeArm()
    guard = PoseGuard(arm, clock)
    with pytest.raises(RuntimeError, match='预算'):
        guard.evaluate(arm.joints, clock())
    original = arm.get_forward_kinematics
    def slow(*args, **kw):
        clock.now += .045
        return original(*args, **kw)
    arm.get_forward_kinematics = slow
    with pytest.raises(RuntimeError, match='超时'):
        guard.evaluate(arm.joints, clock()+.04)
    arm.get_forward_kinematics = lambda *a, **kw: (7, [])
    with pytest.raises(RuntimeError, match='SDK 7'):
        guard.evaluate(arm.joints, clock()+.04)


def test_joint_acceleration_reversal_and_safety_braking():
    q = np.zeros(7); velocity = np.zeros(7); dt = .01
    for _ in range(30):
        next_q = joint_acceleration_step(q, np.ones(7), velocity, dt, 1.)
        v = (next_q-q)/dt
        assert np.max(np.abs(v-velocity)) <= JOINT_ACCELERATION*dt+1e-10
        assert np.max(np.abs(v)) <= 1.+1e-10
        q, velocity = next_q, v
    # Reversal brakes continuously along the previous coordinated direction.
    # The controller's FINAL FK guard validates the braking candidate too.
    braking = joint_acceleration_step(q, q-np.ones(7), velocity, dt, 1.)
    assert (braking-q)/dt == pytest.approx(velocity-JOINT_ACCELERATION*dt)
    reverse = joint_acceleration_step(q, q-np.ones(7), np.zeros(7), dt, 1.)
    assert (reverse-q)/dt == pytest.approx(np.full(7,-.02))
    assert joint_acceleration_step(q, q, velocity, dt, 1.) == pytest.approx(q)


def test_joint_safety_speed_reduction_overrides_soft_braking():
    q = np.zeros(7)
    step = joint_acceleration_step(q, np.ones(7), np.ones(7), .01, .1)
    assert step/.01 == pytest.approx(np.full(7, .1))
    for dt, speed in [(float('nan'),1),(.01,float('inf')),(-.01,1),(.01,-1)]:
        with pytest.raises(ValueError):
            joint_acceleration_step(q, q, q, dt, speed)


def test_small_wrist_correction_does_not_repeatedly_brake_other_moving_joints():
    q=np.zeros(7);dt=.01
    previous=np.array([.2,.2,.2,0,0,0,0])
    desired=np.array([.2,.2,.2,0,0,.05,0])
    step=joint_acceleration_step(q,q+desired*dt,previous,dt,1.)
    velocity=(step-q)/dt
    assert velocity[:3] == pytest.approx(previous[:3])
    assert velocity[5] == pytest.approx(.02)
    assert np.max(np.abs(velocity-previous)) <= JOINT_ACCELERATION*dt+1e-12


def test_partial_turn_interpolates_coupled_velocity_change_and_respects_acceleration():
    q=np.zeros(7);dt=.01
    previous=np.array([.1,.2,.15,.05,0,-.08,0])
    desired=np.array([.1,.25,.17,-.05,.03,-.02,0])
    for _ in range(10):
        step=joint_acceleration_step(q,q+desired*dt,previous,dt,1.)
        velocity=(step-q)/dt
        change=desired-previous
        fraction=min(1.,JOINT_ACCELERATION*dt/max(np.max(np.abs(change)),1e-12))
        assert velocity==pytest.approx(previous+fraction*change)
        assert np.max(np.abs(velocity-previous)) <= JOINT_ACCELERATION*dt+1e-10
        q,previous=step,velocity
    assert previous==pytest.approx(desired)


def metric(sigma):
    return dict(sigma_min=sigma, condition=100./max(sigma,.01),
        speed_ratio=float(np.clip((sigma-8)/12,0,1)),
        status='blocked' if sigma<=8 else 'slowing' if sigma<20 else 'normal')


def test_predicted_risk_stops_before_command_and_requires_new_clutch():
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True)
    count = controller.command_count
    controller.pose_guard.evaluate = lambda q,*a,**kw: metric(1 if q[0]>.00001 else 100)
    button_frames(clock, controller, True, frames=1, position=(.2,1,.31))
    assert controller.command_count == count and not controller.servo_started
    assert controller.mode == 'motion' and arm.state == 4
    assert controller.mapper.hand_anchor is None
    assert controller.requested == pytest.approx(controller.tcp)
    assert controller.telemetry()['pose_guard']['status'] == 'paused'
    button_frames(clock, controller, True, frames=20, position=(.2,1,.31))
    assert controller.command_count == count and not controller.servo_started
    button_frames(clock, controller, False, position=(.7,.8,.6))
    button_frames(clock, controller, True, position=(.7,.8,.6))
    assert controller.servo_started
    assert controller.mapper.hand_anchor == pytest.approx([.7,.8,.6])
    assert controller.command_tcp == pytest.approx(controller.tcp)


@pytest.mark.parametrize('kind', ['translation', 'rotation', 'ik_subdivision'])
def test_constrained_backlog_is_dropped_not_chased(kind):
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True)
    count = controller.command_count
    controller.pose_guard.evaluate = lambda *a, **kw: metric(100 if kind=='ik_subdivision' else 14)
    changes = {'position':(.2,1,.4)}
    if kind == 'rotation':
        q = Rotation.from_euler('z',35,degrees=True).as_quat()
        changes = {'quaternion_wxyz':[q[3],*q[:3]]}
    if kind == 'ik_subdivision':
        controller.solve_follow_target = lambda *a: (arm.joints.copy(), 2)
    button_frames(clock, controller, True, frames=1, **changes)
    assert controller.command_count > count and controller.servo_started
    assert controller.mode == 'motion'
    assert controller.backlog_feedback['status'] == 'bounded'
    assert np.linalg.norm(controller.bounded[:3]-controller.tcp[:3]) <= 4.000001
    from quest_relative_motion import rotation_distance
    assert rotation_distance(controller.bounded[3:],controller.tcp[3:]) <= np.deg2rad(3.000001)
    assert controller.mapper.hand_anchor is not None
    # Holding the same hand cannot regenerate discarded motion.
    same = controller.mapper.target(changes.get('position',(.2,1,.3)),
                                    changes.get('quaternion_wxyz',[1,0,0,0]))[1]
    assert same == pytest.approx(controller.bounded)


def test_unsafe_takeover_is_rejected_before_enabling():
    clock, arm, controller = trigger_setup()
    controller.pose_guard.evaluate = lambda *a, **kw: metric(1)
    before = arm.calls.copy()
    button_frames(clock, controller, True)
    assert 'enable' not in arm.calls[len(before):]
    assert not controller.servo_started
    assert '奇异' in controller.pose_guard.snapshot()['message']


def test_retreat_continues_and_leaving_warning_does_not_restore_old_goal():
    clock, arm, c = trigger_setup()
    button_frames(clock,c,True)
    c.pose_guard.evaluate = lambda q,*a,**kw: metric(14+2000*q[0])
    directions = set()
    start = arm.tcp.copy()
    # A 100 mm gesture exceeds the old 45 mm stop threshold. It now leaves
    # only the local goal; crossing back into normal must not revive 100 mm.
    for _ in range(150):
        button_frames(clock,c,True,frames=1,position=(.2,1,.4))
        directions.add(c.pose_guard.snapshot().get('direction'))
        assert c.servo_started and c.mode=='motion'
        assert c.bounded[0] <= start[0]+4.000001
    assert arm.tcp[0]-start[0] == pytest.approx(4,abs=.1)
    assert 'retreating' in directions
    assert c.pose_guard.snapshot()['sigma_min'] >= 20
    # A small reversal immediately changes the bounded goal, without first
    # taking back all the discarded 96 mm of hand displacement.
    button_frames(clock,c,True,frames=1,position=(.2,1,.399))
    assert c.bounded[0] == pytest.approx(start[0]+3,abs=.01)
    button_frames(clock,c,False,position=(.2,1,.399))
    assert not c.servo_started and arm.state == 4
    button_frames(clock,c,True,position=(.2,1,.399))
    assert c.servo_started
    assert c.mapper.tcp_anchor == pytest.approx(c.tcp)


def test_approaching_warning_remains_speed_limited_and_identified():
    clock, arm, c = trigger_setup()
    button_frames(clock,c,True)
    c.pose_guard.evaluate = lambda q,*a,**kw: metric(16-1000*q[0])
    directions=set()
    for _ in range(20):
        button_frames(clock,c,True,frames=1,position=(.2,1,.4))
        directions.add(c.pose_guard.snapshot().get('direction'))
        assert c.servo_started
        assert c.pose_guard.snapshot()['speed_ratio'] < 1
        assert c.pose_guard.snapshot()['sigma_min'] > 8
    assert 'approaching' in directions


def test_rotation_backcalculation_preserves_increment_history_and_gain():
    from quest_relative_motion import RelativeTranslation, hand_rotation
    m=RelativeTranslation(rotation_enabled=True,rotation_scale=2)
    hand=np.array([.2,1,.3]); tcp=np.array([400.,0,300,0,0,0])
    m.engage(hand,tcp,[1,0,0,0])
    r=Rotation.from_euler('xyz',[.1,.2,.15]); q=r.as_quat();wxyz=[q[3],*q[:3]]
    requested,_=m.target(hand,wxyz)
    local=requested.copy();local[3:]=[.01,.02,.03]
    m.discard_target_excess(requested,local)
    assert m.target(hand,wxyz)[0] == pytest.approx(local)
    next_r=Rotation.from_rotvec([.005,-.004,.003])*r
    q=next_r.as_quat();next_wxyz=[q[3],*q[:3]]
    robot_delta=np.linalg.det(m.matrix)*m.matrix@(hand_rotation(next_wxyz)*hand_rotation(wxyz).inv()).as_rotvec()
    expected=Rotation.from_rotvec(robot_delta*2)*Rotation.from_euler('xyz',local[3:])
    actual=m.target(hand,next_wxyz)[0]
    assert (Rotation.from_euler('xyz',actual[3:])*expected.inv()).magnitude()<1e-10
