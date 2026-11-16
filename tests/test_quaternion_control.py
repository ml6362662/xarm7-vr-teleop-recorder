"""Offline quaternion pipeline regression. All robot calls use FakeArm."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import quest_pose_math as pm
from quest_relative_motion import (RelativeTranslation, interpolate_pose, smooth_rotation,
    smooth_target, bound_follow_goal, command_velocities, rotation_distance, settings, tether)
from test_relative_motion import trigger_setup, button_frames, FakeArm
from test_quest_joint_recovery import recovery_setup, save_and_displace


@pytest.mark.parametrize('rpy', [[.1, np.pi/2, .3], [.1, -np.pi/2, .3], [0, 0, np.pi]])
def test_sdk_boundary_preserves_physical_orientation_at_euler_singularities(rpy):
    pose = pm.as_quaternion_pose([400, 0, 300, *rpy])
    assert pose.shape == (7,)
    assert np.linalg.norm(pose[3:]) == pytest.approx(1)
    sdk = pm.pose_to_sdk(pose)
    assert sdk.shape == (6,)
    assert rotation_distance(sdk[3:], pose[3:]) < 1e-12


@pytest.mark.parametrize('q', [[0,0,0,0], [float('nan'),0,0,1], [float('inf'),0,0,1]])
def test_invalid_quaternions_rejected(q):
    with pytest.raises(ValueError):
        pm.as_quaternion_pose([1,2,3,*q])


def test_pure_quaternion_pipeline_never_round_trips_through_euler(monkeypatch):
    tcp = pm.as_quaternion_pose([400,0,300,.3,np.pi/2,-.7])
    hand = pm.quaternion_wxyz(Rotation.from_rotvec([.2,.3,.1]))
    original = pm.orientation_rotation
    def quaternion_only(value):
        assert np.asarray(value).shape == (4,), 'internal legacy RPY encountered'
        return original(value)
    def forbidden(*args, **kwargs):
        raise AssertionError('Euler conversion inside quaternion pipeline')
    monkeypatch.setattr(pm, 'orientation_rotation', quaternion_only)
    monkeypatch.setattr(pm, 'rotation_rpy', forbidden)
    import quest_relative_motion as motion
    monkeypatch.setattr(motion, 'orientation_rotation', quaternion_only)
    mapper = RelativeTranslation(scale=2, rotation_enabled=True, rotation_scale=2)
    mapper.engage([0,0,0], tcp, hand)
    requested, bounded = mapper.target([.01,.02,.03], hand)
    assert rotation_distance(requested[3:], tcp[3:]) < 1e-12
    # A sign flip is not rotation, even at an Euler singularity.
    requested, bounded = mapper.target([.02,.03,.04], -hand)
    assert rotation_distance(requested[3:], tcp[3:]) < 1e-12
    bounded, _ = bound_follow_goal(bounded, tcp, np.zeros(3), np.zeros(3), settings({}))
    mapper.discard_target_excess(requested, bounded)
    target, _ = smooth_target(bounded, tcp, np.zeros(3), .01, 50, 300)
    target[3:], _ = smooth_rotation(bounded[3:], tcp[3:], np.zeros(3), .01, 1, 2)
    target = interpolate_pose(tcp, target, .5)
    target = tether(target, tcp, 5)
    assert target.shape == (7,)
    assert command_velocities(tcp, target, .01, 50, 1)[1] == pytest.approx(np.zeros(3), abs=1e-12)


@pytest.mark.parametrize('axis,start,end', [('z',179,181), ('y',89,91), ('y',-89,-91)])
def test_quaternion_interpolation_and_smoothing_cross_seams(axis, start, end):
    p = np.r_[400,0,300,pm.quaternion_wxyz(Rotation.from_euler(axis,start,degrees=True))]
    goal = np.r_[400,0,300,pm.quaternion_wxyz(Rotation.from_euler(axis,end,degrees=True))]
    halfway = interpolate_pose(p, goal, .5)
    assert np.rad2deg(rotation_distance(p[3:], halfway[3:])) == pytest.approx(1)
    velocity = np.zeros(3)
    current = p[3:].copy()
    for _ in range(250):
        following, next_velocity = smooth_rotation(goal[3:], current, velocity, .01, .5, 1)
        assert following.shape == (4,)
        assert np.linalg.norm(following) == pytest.approx(1)
        assert np.dot(current, following) > 0
        assert np.linalg.norm(next_velocity-velocity) <= .010000001
        assert rotation_distance(following, current) <= .005000001
        current, velocity = following, next_velocity
    # Existing acceleration ramp converges asymptotically; <0.00006 degrees.
    assert rotation_distance(current, goal[3:]) < 1e-6


def test_local_delta_does_not_remove_measured_rotation_error():
    previous = Rotation.from_euler('xyz', [.4,.8,-.5])
    current = Rotation.from_rotvec([0,np.deg2rad(5),0])*previous
    local = previous.inv()*current
    world = current*previous.inv()
    assert np.rad2deg(local.magnitude()) == pytest.approx(5)
    assert local.magnitude() == pytest.approx(world.magnitude())
    # 0.999 is near 1, yet represents a significant 5-degree change.
    assert abs(np.dot(previous.as_quat(),current.as_quat())) == pytest.approx(np.cos(np.deg2rad(2.5)))


def test_backend_states_and_sdk_boundary_remain_distinct_during_follow_and_pause(monkeypatch):
    original = FakeArm.get_inverse_kinematics
    def checked(self, pose, **kwargs):
        assert np.asarray(pose).shape == (6,), 'quaternion pose leaked into SDK RPY API'
        return original(self, pose, **kwargs)
    monkeypatch.setattr(FakeArm, 'get_inverse_kinematics', checked)
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True)
    button_frames(clock, controller, True, position=(.2,1,.32), frames=15,
                  quaternion_wxyz=pm.quaternion_wxyz(Rotation.from_rotvec([.03,0,0])))
    assert controller.servo_started and controller.command_count > 0
    for name in ('tcp','command_tcp','bounded','requested','hold_tcp'):
        pose = getattr(controller, name)
        assert pose.shape == (7,), name
        assert np.linalg.norm(pose[3:]) == pytest.approx(1), name
    assert controller.mapper.tcp_anchor.shape == (7,)
    button_frames(clock, controller, False, frames=1)
    assert not controller.servo_started
    assert controller.mode == 'motion'  # release pauses, does not disable
    assert controller.telemetry()['orientation_representation'] == 'quaternion_wxyz'


def test_new_home_quaternion_and_legacy_pose_compatibility(tmp_path):
    _, arm, controller = recovery_setup(tmp_path)
    save_and_displace(controller, arm)
    recovery = controller.recovery
    assert recovery.home['version'] == 3
    assert len(recovery.home['tcp_pose_quat']) == 7
    assert len(recovery.feedback['target_pose_quat']) == 7
    for version in (1,2):
        legacy = dict(recovery.home, version=version)
        del legacy['tcp_pose_quat']
        assert rotation_distance(recovery.validate_profile(legacy)[3:], recovery.home['tcp'][3:]) < 1e-12
    recovery.home['tcp_pose_quat'][0] += 1
    with pytest.raises(RuntimeError, match='不一致'):
        recovery.validate_profile(recovery.home)
