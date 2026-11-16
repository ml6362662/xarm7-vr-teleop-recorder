"""Offline only: fake robot and deterministic input, no hardware connection."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quest_relative_motion import bound_follow_goal, settings, rotation_distance
from quest_pose_guard import joint_acceleration_step, JOINT_ACCELERATION
from test_relative_motion import Clock, FakeArm, engage, state, tick_100
from xarm_quest_relative import RelativeArm
from test_relative_motion import button_frames


def test_braking_allowance_bounds_backlog_and_preserves_small_moves():
    config = settings(dict(speed_mm_s=150,tracking_lead_mm=12,acceleration_mm_s2=600,
        rotation_speed_deg_s=60,rotation_acceleration_deg_s2=240))
    origin = np.zeros(6)
    goal = np.array([200.,0,0,0,0,.5])
    result, info = bound_follow_goal(goal,origin,[150,0,0],[0,0,np.pi/3],config)
    assert info['limit_mm'] == pytest.approx(30.75)
    assert result[0] == pytest.approx(30.75)
    assert info['pending_translation_mm'] == pytest.approx(30.75)
    assert info['limit_deg'] == pytest.approx(10.5)
    assert np.rad2deg(result[5]) == pytest.approx(10.5)
    small = np.array([1.,2.,3.,.01,-.01,.02])
    result, info = bound_follow_goal(small,origin,np.zeros(3),np.zeros(3),config)
    assert result == pytest.approx(small)
    assert info['discarded_mm'] == 0


def test_reverse_clears_opposing_backlog_but_ignores_noise():
    config = settings(dict(tracking_lead_mm=12))
    goal = np.array([10.,2.,0,0,0,np.deg2rad(2.5)])
    result, info = bound_follow_goal(goal,np.zeros(6),np.zeros(3),np.zeros(3),config,
        [-1,0,0],[0,0,-.01])
    assert result[:3] == pytest.approx([-1,2,0])
    assert result[5] == pytest.approx(-.01)
    assert info['reverse_translation'] and info['reverse_rotation']
    result, info = bound_follow_goal(goal,np.zeros(6),np.zeros(3),np.zeros(3),config,
        [-.01,0,0],[0,0,-.0001])
    assert result == pytest.approx(goal)
    assert not info['reverse_translation'] and not info['reverse_rotation']


def test_joint_reversal_brakes_without_zero_speed_impulse():
    q = np.zeros(7)
    velocity = np.array([.5,.25,-.1,0,0,0,0])
    original_direction = velocity.copy()
    for _ in range(25):
        candidate = joint_acceleration_step(q,q-original_direction,velocity,.01,1.)
        next_velocity = (candidate-q)/.01
        assert np.max(np.abs(next_velocity-velocity)) <= JOINT_ACCELERATION*.01+1e-10
        assert np.linalg.norm(np.cross(next_velocity[:3],original_direction[:3])) < 1e-10
        q,velocity=candidate,next_velocity
    assert np.max(np.abs(velocity)) < 1e-10
    candidate = joint_acceleration_step(q,q-original_direction,velocity,.01,1.)
    assert np.dot(candidate-q,original_direction) < 0


def test_braking_candidate_still_obeys_final_fk_and_joint_limits():
    from test_follow_command_guard import setup_guard
    clock, arm, c = setup_guard()
    q = np.array([.011,0,0,0,0,0,0])
    previous_tcp = np.asarray(arm.get_forward_kinematics(q)[1])
    candidate = joint_acceleration_step(q,q-np.array([.01,0,0,0,0,0,0]),
        [.5,0,0,0,0,0,0],.01,1.)
    assert candidate[0] > q[0]  # braking can continue in the old direction
    command,fk,fraction = c.fit_follow_command(candidate,q,previous_tcp,.01,12,np.pi/3,clock()+.04)
    assert fraction < 1
    assert np.linalg.norm(fk[:3]-c.tcp[:3]) <= 12.1
    c.upper = c.upper.astype(float)
    c.upper[0] = q[0]+1e-6
    command,fk,fraction = c.fit_follow_command(candidate,q,previous_tcp,.01,12,np.pi/3,clock()+.04)
    assert fraction == 0 and command == pytest.approx(q)
    assert arm.calls == []


def run_fast_gesture():
    class LagArm(FakeArm):
        def set_servo_angle_j(self,joints,**kwargs):
            self.calls.append('servo')
            self.joints += (np.asarray(joints)-self.joints)*.2
            self.tcp = np.asarray(self.get_forward_kinematics(self.joints)[1])
            return 0
    clock,arm=Clock(),LagArm()
    c=RelativeArm(arm,[-3]*7,[3]*7,clock,control_settings=dict(speed_mm_s=200,
        translation_scale=3,acceleration_mm_s2=1000,tracking_lead_mm=12,
        rotation_enabled=False,joint_speed_rad_s=1,teleop_button='b'))
    engage(clock,c)
    rows=[]
    for i in range(600):
        # 200 mm/s hand ramp, then reverse, then hold. Mapping requests
        # 600 mm/s: above the selected 200 mm/s robot speed ceiling.
        z=.3+.002*(i if i<100 else 200-i if i<200 else 0)
        tick_100(clock,c,state(position=(.2,1,z),invalid=1,pause=1,count=i+4))
        assert c.servo_started and c.mode=='motion'
        assert np.linalg.norm(c.command_tcp[:3]-c.tcp[:3]) <= 12.1
        assert np.max(np.abs(c.joint_velocity)) <= 1.+1e-8
        rows.append([c.command_tcp[0],np.linalg.norm(c.bounded[:3]-c.command_tcp[:3]),
            np.linalg.norm(c.cartesian_velocity),arm.tcp[0]])
    return np.array(rows),c


def test_fast_out_and_back_keeps_short_backlog_and_does_not_chase_old_motion():
    rows,c=run_fast_gesture()
    assert rows[:,1].max() <= 32.1
    assert np.max(np.abs(np.diff(rows[:,0],n=2))) <= .105
    assert rows[:,2].max() <= 200.0001
    reverse=np.flatnonzero(np.diff(rows[100:150,0])<-.001)[0]+1
    assert reverse < 30
    assert abs(rows[250,0]-rows[-1,0]) < .1
    same = c.mapper.target((.2,1,.3),[1,0,0,0])[1]
    assert same == pytest.approx(c.bounded)
    c.command('stop')
    assert not c.servo_started


@pytest.mark.parametrize('empirical', [False, True])
@pytest.mark.parametrize('axis,sign', [(a,s) for a in range(3) for s in (-1,1)])
def test_small_translation_preserves_three_times_gain_in_six_directions(empirical,axis,sign):
    endpoints=[]
    for ramp_frames in (5,100):
        clock,arm=Clock(),FakeArm()
        config=dict(speed_mm_s=200,translation_scale=3,acceleration_mm_s2=1000,
            tracking_lead_mm=12,rotation_enabled=False,joint_speed_rad_s=1,teleop_button='b')
        k=np.eye(6)
        k[:3,:3]=[[0,0,.6],[-.95,0,0],[0,.6,0]]
        if empirical:config.update(empirical_matrix=k.tolist(),empirical_frame=['unitree','test'])
        c=RelativeArm(arm,[-3]*7,[3]*7,clock,control_settings=config)
        def sample(position=(.2,1,.3),**kwargs):
            return dict(state(position=position,**kwargs),input_backend='unitree',frame_id='test')
        c.update_source(sample());c.command('enable')
        tick_100(clock,c,sample(enabled=False,valid=False,invalid=1,pause=1,count=2))
        tick_100(clock,c,sample(invalid=1,pause=1,count=3))
        delta=np.zeros(3);delta[axis]=sign*.002
        mapping=k[:3,:3] if empirical else c.mapper.matrix
        expected=arm.home[:3]+mapping@delta*3000
        for i in range(1,301):
            hand=np.array([.2,1,.3])+delta*min(1,i/ramp_frames)
            previous=c.command_tcp.copy()
            tick_100(clock,c,sample(hand,invalid=1,pause=1,count=i+3))
            assert c.servo_started
            assert np.linalg.norm(c.command_tcp[:3]-previous[:3]) <= 2.000001
            assert np.linalg.norm(c.command_tcp[:3]-c.tcp[:3]) <= 12.1
            if i>=ramp_frames:assert c.bounded[:3] == pytest.approx(expected,abs=1e-6)
        assert arm.tcp[:3] == pytest.approx(expected,abs=.1)
        endpoints.append(arm.tcp[:3].copy())
    assert endpoints[0] == pytest.approx(endpoints[1],abs=.1)


@pytest.mark.parametrize('interrupt', ['trigger_release','tracking_loss','heartbeat_loss'])
@pytest.mark.parametrize('empirical', [False,True])
def test_pending_translation_is_cancelled_and_never_resumes_after_interruption(interrupt,empirical):
    clock,arm=Clock(),FakeArm()
    config=dict(translation_scale=3,speed_mm_s=200,acceleration_mm_s2=1000,
        tracking_lead_mm=12,joint_speed_rad_s=1,rotation_enabled=False)
    if empirical:config.update(empirical_matrix=np.eye(6).tolist(),empirical_frame=['unitree','test'])
    c=RelativeArm(arm,[-3]*7,[3]*7,clock,control_settings=config)
    source=dict(input_backend='unitree',frame_id='test')
    c.update_source(dict(state(),**source));c.command('allow_controller')
    button_frames(clock,c,False,**source)
    button_frames(clock,c,True,**source)
    hand=(.2,1,.38)
    button_frames(clock,c,True,frames=1,position=hand,**source)
    assert 10 < np.linalg.norm(c.bounded[:3]-c.tcp[:3]) < 33
    commands=c.command_count
    if interrupt=='trigger_release':
        button_frames(clock,c,False,frames=1,position=hand,**source)
    elif interrupt=='tracking_loss':
        button_frames(clock,c,True,frames=1,valid=False,invalid_count=1,position=hand,**source)
    else:
        clock.now+=1;c.tick()
    assert not c.servo_started and arm.state==4 and c.mapper.hand_anchor is None
    assert c.command_count==commands
    stopped=arm.tcp.copy()
    # Stop settling, then a fresh clutch at the same hand position. The
    # discarded gesture and remaining short backlog must not reappear.
    button_frames(clock,c,False,frames=30,position=hand,**source)
    if c.mode=='preview':c.command('allow_controller')
    button_frames(clock,c,True,frames=30,position=hand,**source)
    assert c.servo_started
    assert arm.tcp==pytest.approx(stopped,abs=1e-6)
    assert c.bounded[:3]==pytest.approx(stopped[:3],abs=1e-6)
