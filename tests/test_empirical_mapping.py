import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quest_pose_math import quaternion_wxyz, orientation_rotation
from quest_relative_motion import RelativeTranslation, settings


def pose(position, rotation):
    return np.r_[position, quaternion_wxyz(rotation)]


def endpoint(start, delta):
    return pose(start[:3] + delta[:3], Rotation.from_rotvec(delta[3:]/100) * orientation_rotation(start[3:]))


def sample_matrix():
    k = np.diag([.6, .7, .65, .4, .35, .45])
    k[:3, 3:] = [[0,.1,0], [.1,0,0], [0,0,-.2]]
    k[3:, :3] = [[0,-.08,0], [.12,0,0], [0,0,-.1]]
    return k


def test_cross_talk_cancellation_and_gains_are_post_calibration():
    k=sample_matrix()
    start=pose([0,0,0],Rotation.identity())
    goal=np.array([50.,0,0,0,0,0])
    measured=np.linalg.solve(k,goal)
    hand=endpoint(start,measured)
    mapper=RelativeTranslation(scale=2,rotation_enabled=True,rotation_scale=3,empirical_matrix=k)
    mapper.engage(start[:3],start,start[3:])
    result,_=mapper.target(hand[:3]/1000,hand[3:])
    assert result[:3] == pytest.approx([100,0,0],abs=1e-7)
    assert orientation_rotation(result[3:]).magnitude()<1e-7


def test_anti_windup_does_not_regenerate_discarded_pose_and_reclutch_is_zero():
    k=sample_matrix()
    start=pose([400,0,300],Rotation.from_euler('z',.3))
    hs=pose([0,0,0],Rotation.identity())
    h=endpoint(hs,np.array([80.,0,0,20,0,0]))
    m=RelativeTranslation(scale=1,rotation_enabled=True,empirical_matrix=k)
    m.engage(hs[:3],start,hs[3:])
    requested,_=m.target(h[:3]/1000,h[3:])
    bounded=endpoint(start,np.array([5,0,0,1,0,0]))
    m.discard_target_excess(requested,bounded)
    next_target,_=m.target(h[:3]/1000,h[3:])
    assert next_target[:3] == pytest.approx(bounded[:3])
    assert (orientation_rotation(next_target[3:])*orientation_rotation(bounded[3:]).inv()).magnitude()<1e-8
    assert np.linalg.norm(m.last_linear_increment)<1e-8
    assert np.linalg.norm(m.last_angular_increment)<1e-8
    m.release()
    m.engage(h[:3]/1000,bounded,h[3:])
    next_target,_=m.target(h[:3]/1000,h[3:])
    assert next_target == pytest.approx(bounded)


def test_single_frame_jump_still_fails_closed_and_disabled_rotation_stays_fixed():
    start=pose([0,0,0],Rotation.identity())
    m=RelativeTranslation(scale=1,rotation_enabled=False,empirical_matrix=sample_matrix())
    m.engage(start[:3],start,start[3:])
    hand=endpoint(start,np.array([30,0,0,20,0,0]))
    result,_=m.target(hand[:3]/1000,hand[3:])
    assert result[3:] == pytest.approx(start[3:])
    with pytest.raises(ValueError,match='重新接管'):
        m.target([0,0,0],quaternion_wxyz(Rotation.from_euler('x',151,degrees=True)))


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('gain', [.95,2.5])
def test_cross_coupled_map_crosses_both_pi_branches_and_reverses_continuously(axis,gain):
    k=sample_matrix()
    start=pose([400,0,300],Rotation.from_euler('xyz',[.2,-.4,.3]))
    m=RelativeTranslation(scale=3,rotation_enabled=True,rotation_scale=gain,empirical_matrix=k)
    m.engage([0,0,0],start,[1,0,0,0])
    previous=start.copy()
    angles=np.r_[np.arange(0,362,2),np.arange(358,-362,-2)]
    for i,angle in enumerate(angles):
        rv=np.zeros(3);rv[axis]=np.deg2rad(angle)
        q=quaternion_wxyz(Rotation.from_rotvec(rv))
        if i%3==0:q=-q  # an input quaternion sign change is not physical motion
        target,_=m.target([0,0,0],q)
        mapped=k@np.r_[np.zeros(3),100*rv]
        expected=Rotation.from_rotvec(mapped[3:]/100*gain)*orientation_rotation(start[3:])
        assert target[:3] == pytest.approx(start[:3]+3*mapped[:3],abs=1e-7)
        assert (orientation_rotation(target[3:])*expected.inv()).magnitude()<1e-8
        assert np.dot(target[3:],previous[3:])>0
        assert np.linalg.norm(target[:3]-previous[:3])<3
        assert np.linalg.norm(m.last_angular_increment)<np.deg2rad(3)
        previous=target
    assert m.empirical_reanchors>=8


def test_noncommuting_rotations_keep_exact_identity_map_through_chart_changes():
    k=np.eye(6)
    start=pose([400,0,300],Rotation.from_euler('xyz',[.7,-.2,.5]))
    initial=Rotation.from_euler('xyz',[-.5,.3,.8]);hand=initial
    m=RelativeTranslation(scale=2,rotation_enabled=True,empirical_matrix=k)
    m.engage([0,0,0],start,quaternion_wxyz(hand))
    previous=start
    for i in range(800):
        step=np.array([.014*np.cos(i*.03),.018*np.sin(i*.013),.016])
        hand=Rotation.from_rotvec(step)*hand
        position=np.array([.01*np.sin(i*.02),.01*np.cos(i*.015),.0001*i])
        target,_=m.target(position,quaternion_wxyz(hand))
        expected=hand*initial.inv()*orientation_rotation(start[3:])
        assert target[:3] == pytest.approx(start[:3]+2000*position,abs=1e-7)
        assert (orientation_rotation(target[3:])*expected.inv()).magnitude()<1e-8
        assert np.dot(target[3:],previous[3:])>0
        previous=target
    assert m.empirical_reanchors>=5


@pytest.mark.parametrize('rotation_enabled',[True,False])
def test_chart_carries_adjusted_target_not_original_or_measured_pose(rotation_enabled):
    k=sample_matrix();start=pose([400,0,300],Rotation.identity())
    m=RelativeTranslation(scale=3,rotation_enabled=rotation_enabled,rotation_scale=.95,empirical_matrix=k)
    m.engage([0,0,0],start,[1,0,0,0])
    for angle in range(0,89,2):
        requested,_=m.target([0,0,0],quaternion_wxyz(Rotation.from_euler('x',angle,degrees=True)))
    bounded=pose([405,-3,302],Rotation.from_euler('z',.02) if rotation_enabled else Rotation.identity())
    m.discard_target_excess(requested,bounded)
    q=quaternion_wxyz(Rotation.from_euler('x',90,degrees=True))
    target,_=m.target([0,0,0],q)
    mapped=k@np.array([0,0,0,100*np.deg2rad(2),0,0])
    expected=(Rotation.from_rotvec(mapped[3:]/100*.95)*orientation_rotation(bounded[3:])
        if rotation_enabled else Rotation.identity())
    assert target[:3] == pytest.approx(bounded[:3]+3*mapped[:3],abs=1e-8)
    assert (orientation_rotation(target[3:])*expected.inv()).magnitude()<1e-8
    assert m.empirical_reanchors==1
    held,_=m.target([0,0,0],-q)
    assert held==pytest.approx(target,abs=1e-8)
    assert m.last_linear_increment==pytest.approx(np.zeros(3),abs=1e-8)
    assert m.last_angular_increment==pytest.approx(np.zeros(3),abs=1e-8)
    m.release();assert m.empirical_reanchors==0
    m.engage([0,0,0],bounded,q)
    assert m.target([0,0,0],q)[0]==pytest.approx(bounded,abs=1e-8)


def test_jump_is_rejected_even_near_automatic_chart_boundary():
    start=pose([400,0,300],Rotation.identity())
    m=RelativeTranslation(empirical_matrix=np.eye(6))
    m.engage([0,0,0],start,[1,0,0,0])
    for angle in (20,40,60,80):
        m.target([0,0,0],quaternion_wxyz(Rotation.from_euler('z',angle,degrees=True)))
    with pytest.raises(ValueError,match='单帧旋转突变'):
        m.target([0,0,0],quaternion_wxyz(Rotation.from_euler('z',140,degrees=True)))
    assert m.empirical_reanchors==0


def test_json_settings_roundtrip_requires_frame_binding():
    d=settings(dict(empirical_matrix=sample_matrix().tolist(),empirical_frame=['unitree','one']))
    assert settings(json.loads(json.dumps(d)))==d
    with pytest.raises(ValueError,match='绑定'):
        settings(dict(empirical_matrix=np.eye(6).tolist()))


def test_runtime_reverse_detection_uses_compensated_output_increments():
    k=sample_matrix()
    start=pose([0,0,0],Rotation.identity())
    m=RelativeTranslation(scale=1,rotation_enabled=True,empirical_matrix=k)
    m.engage(start[:3],start,start[3:])
    for goal in ([50.,0,0,0,0,0],[40.,0,0,0,0,0]):
        hand=endpoint(start,np.linalg.solve(k,np.array(goal)))
        m.target(hand[:3]/1000,hand[3:])
    assert m.last_linear_increment==pytest.approx([-10,0,0],abs=1e-7)
    assert m.last_angular_increment==pytest.approx([0,0,0],abs=1e-7)
