import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quest_direction_guard import choose_direction, direction_speed_ratio
from test_pose_guard import metric
from test_relative_motion import Clock, FakeArm, trigger_setup, button_frames


class TestMetric:
    __test__ = False
    def evaluate(self, q, *a, **kw):
        return metric(8+1000*q[0]-1000*q[4])


def test_mixed_wrist_request_projects_only_when_translation_really_improves():
    arm, clock = FakeArm(), Clock()
    goal=arm.tcp.copy();goal[0]+=1;goal[4]+=.004
    result=choose_direction(arm,TestMetric(),arm.joints,arm.tcp,goal,metric(8),
                            np.full(7,-3),np.full(7,3),clock()+.04,clock)
    assert result['translation_only']
    assert result['direction']=='retreating'
    assert result['probes'][0]['direction']=='approaching'
    assert result['probes'][1]['direction']=='retreating'
    goal[0]-=2
    result=choose_direction(arm,TestMetric(),arm.joints,arm.tcp,goal,metric(8),
                            np.full(7,-3),np.full(7,3),clock()+.04,clock)
    assert not result['translation_only'] and result['direction']=='approaching'
    assert arm.calls==[]


def test_escape_has_nonzero_capped_speed_not_vanishing_scalar():
    for sigma in [7.51,7.99,8.0,8.01,8.5,10]:
        ratio=direction_speed_ratio(metric(sigma),'retreating',150,60,1)
        assert 0 < ratio <= 20/150
        assert 60*ratio <= 8 and ratio <= .15
    for direction in ['approaching','neutral','blocked']:
        assert direction_speed_ratio(metric(8),direction,150,60,1)==0
    assert direction_speed_ratio(metric(7.5),'retreating',150,60,1)==0


def test_boundary_bad_direction_holds_then_reverses_without_reclutch():
    clock,arm,c=trigger_setup()
    c.pose_guard.evaluate=lambda q,*a,**kw:metric(7.99+4000*q[0])
    button_frames(clock,c,True)
    assert c.servo_started
    count=c.command_count
    button_frames(clock,c,True,frames=1,position=(.2,1,.29))
    assert c.command_count==count and c.servo_started
    assert c.pose_guard.snapshot()['status']=='direction_hold'
    assert c.bounded==pytest.approx(c.command_tcp)
    # Reverse by only 1 mm, not the entire discarded 10 mm. Trigger stays down.
    button_frames(clock,c,True,frames=100,position=(.2,1,.291))
    assert c.servo_started and c.command_count>count
    assert arm.tcp[0]>400.5
    button_frames(clock,c,False,frames=1,position=(.2,1,.291))
    assert not c.servo_started and arm.state==4


def test_deep_boundary_remains_blocked_before_motion_enable():
    clock,arm,c=trigger_setup()
    c.pose_guard.evaluate=lambda *a,**kw:metric(7.4)
    old=len(arm.calls)
    button_frames(clock,c,True)
    assert not c.servo_started and 'enable' not in arm.calls[old:]


def test_translation_priority_latches_until_reclutch_without_rotation_backlog():
    clock,arm,c=trigger_setup()
    c.pose_guard.evaluate=lambda q,*a,**kw:metric(8+1000*q[0]-1000*q[4])
    button_frames(clock,c,True)
    robot_delta=np.array([0,.04,0])
    hand_delta=np.linalg.det(c.mapper.matrix)*c.mapper.matrix.T@robot_delta/c.mapper.rotation_scale
    q=Rotation.from_rotvec(hand_delta).as_quat();wxyz=[q[3],*q[:3]]
    button_frames(clock,c,True,frames=1,position=(.2,1,.31),quaternion_wxyz=wxyz)
    assert c.servo_started and c.translation_priority
    held_rotation=c.command_tcp[3:].copy()
    button_frames(clock,c,True,frames=20,position=(.2,1,.31),quaternion_wxyz=wxyz)
    assert c.servo_started and c.translation_priority
    assert c.bounded[3:] == pytest.approx(held_rotation)
    assert c.command_tcp[0]>400
    button_frames(clock,c,False,frames=1,position=(.2,1,.31),quaternion_wxyz=wxyz)
    assert not c.translation_priority
    button_frames(clock,c,True,position=(.2,1,.31),quaternion_wxyz=wxyz)
    assert c.bounded == pytest.approx(c.tcp)


def test_actual_smoothed_step_is_rechecked_not_just_goal_direction(monkeypatch):
    import xarm_quest_relative
    clock,arm,c=trigger_setup()
    c.pose_guard.evaluate=lambda q,*a,**kw:metric(12+1000*q[0]-2000*q[4])
    button_frames(clock,c,True)
    # The long-term goal may be safe while the next accelerated mixture isn't.
    monkeypatch.setattr(xarm_quest_relative,'choose_direction',lambda *a,**kw:
                        dict(direction='retreating',translation_only=False,probes=[]))
    hand_delta=np.linalg.det(c.mapper.matrix)*c.mapper.matrix.T@np.array([0,.04,0])/c.mapper.rotation_scale
    q=Rotation.from_rotvec(hand_delta).as_quat()
    button_frames(clock,c,True,frames=1,position=(.2,1,.31),quaternion_wxyz=[q[3],*q[:3]])
    assert c.servo_started and c.translation_priority
    assert abs(arm.tcp[4])<1e-9 and arm.tcp[0]>400
    assert c.pose_guard.snapshot()['direction_info']['translation_only']


@pytest.mark.parametrize('kind',['sdk','nonfinite','late'])
def test_probe_failures_do_not_produce_permission(kind):
    arm,clock=FakeArm(),Clock();goal=arm.tcp.copy();goal[0]+=1
    if kind=='sdk':arm.get_inverse_kinematics=lambda *a,**kw:(7,[])
    elif kind=='nonfinite':arm.get_inverse_kinematics=lambda *a,**kw:(0,[float('nan')]*7)
    else:
        def late(*a,**kw):
            clock.now+=.05
            return 0,arm.joints.tolist()
        arm.get_inverse_kinematics=late
    with pytest.raises((RuntimeError,ValueError)):
        choose_direction(arm,TestMetric(),arm.joints,arm.tcp,goal,metric(8),
                         np.full(7,-3),np.full(7,3),clock()+.04,clock)
    assert arm.calls==[]
