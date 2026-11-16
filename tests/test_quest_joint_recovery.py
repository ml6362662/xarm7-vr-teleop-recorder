import json
import numpy as np
import pytest

from test_relative_motion import FakeArm, Clock, state, tick_100
from xarm_quest_relative import RelativeArm
from quest_joint_recovery import JointRecovery


class RecoveryArm(FakeArm):
    def __init__(self):
        super().__init__()
        self.velocity = np.zeros(7)
        self.last_recovery_target = None
        self.expected_speed = 5
        self.expected_acceleration = 10
        self.last_motion_kwargs = None
        self.warning = 0
        self.axis, self.device_type, self.sn = 7, 7, 'test-robot'
        self.tcp_offset = [0]*6
        self.world_offset = [0]*6

    def get_joint_states(self, **kwargs):
        return 0, [self.joints.tolist(), self.velocity.tolist()]

    def get_inverse_kinematics(self, pose, **kwargs):
        code, solution = super().get_inverse_kinematics(pose, **kwargs)
        # Model the redundant seventh joint: use the CURRENT IK reference.
        solution[6] = kwargs.get('ref_angles',self.joints)[6]
        return code,solution

    def set_servo_angle(self, angle, **kwargs):
        if kwargs.get('wait'):
            return super().set_servo_angle(angle, **kwargs)
        assert self.mode == 0
        assert kwargs['is_radian']
        assert kwargs['speed'] == pytest.approx(np.deg2rad(self.expected_speed))
        assert kwargs['mvacc'] == pytest.approx(np.deg2rad(self.expected_acceleration))
        assert kwargs['radius'] is None
        self.calls.append('joint_recover')
        self.last_recovery_target = np.array(angle)
        self.last_motion_kwargs = kwargs
        return 0

    def arrive(self):
        self.joints = self.last_recovery_target.copy()
        self.tcp = np.array(self.get_forward_kinematics(self.joints)[1])


def recovery_setup(tmp_path=None):
    clock, arm = Clock(), RecoveryArm()
    controller = RelativeArm(arm, [-3]*7, [3]*7, clock,
        hardware_lower=[-3.1]*7, hardware_upper=[3.1]*7,
        home_path=None if tmp_path is None else tmp_path/'home.json')
    return clock, arm, controller


def save_and_displace(controller, arm):
    controller.command('home_save', {'confirmed': True})
    arm.joints[1] = .2
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    controller.command('home_check')
    feedback = controller.recovery.feedback
    return dict(confirmed=True, home_saved_at=controller.recovery.home['saved_at'], checked_start=feedback['start'])


def test_enable_only_requires_current_pose_not_all_six_cartesian_directions():
    clock, arm, controller = recovery_setup()
    original = arm.get_inverse_kinematics
    def boundary_ik(pose, **kwargs):
        if np.linalg.norm(np.array(pose)[:3]-arm.tcp[:3]) > .1:
            return 10, []
        return original(pose, **kwargs)
    arm.get_inverse_kinematics = boundary_ik
    controller.command('allow_controller')
    assert controller.mode == 'motion'
    assert not controller.servo_started
    assert 'enable' not in arm.calls


def test_current_pose_branch_jump_still_blocks_enable():
    clock, arm, controller = recovery_setup()
    arm.get_inverse_kinematics = lambda *a, **kw: (0, [1]*7)
    with pytest.raises(RuntimeError, match='IK 分支'):
        controller.command('allow_controller')
    assert 'enable' not in arm.calls


def test_home_record_persists_without_motion_and_reloads_with_identity(tmp_path):
    clock, arm, controller = recovery_setup(tmp_path)
    controller.command('home_save', {'confirmed': True})
    assert arm.calls == []
    profile = json.loads((tmp_path/'home.json').read_text())
    assert profile['joints_rad'] == pytest.approx(arm.joints)
    loaded = JointRecovery(arm, [-3.1]*7, [3.1]*7, tmp_path/'home.json')
    assert loaded.home == profile
    arm.sn = 'another-robot'
    rejected = JointRecovery(arm, [-3.1]*7, [3.1]*7, tmp_path/'home.json')
    assert rejected.home is None and rejected.feedback['status'] == 'rejected'


def test_rerecord_home_replaces_persistent_target_without_motion(tmp_path):
    _,arm,controller=recovery_setup(tmp_path)
    controller.command('home_save',{'confirmed':True})
    original=controller.recovery.home['tcp_pose_quat']
    arm.joints[1]=.2
    arm.tcp=np.array(arm.get_forward_kinematics(arm.joints)[1])
    controller.command('home_save',{'confirmed':True})
    loaded=JointRecovery(arm,[-3.1]*7,[3.1]*7,tmp_path/'home.json')
    assert loaded.home==controller.recovery.home
    assert loaded.home['tcp_pose_quat']!=original
    assert arm.calls==[]


def test_saving_bad_initial_pose_or_unconfirmed_pose_is_refused():
    clock, arm, controller = recovery_setup()
    with pytest.raises(RuntimeError, match='确认'):
        controller.command('home_save')
    original = arm.get_inverse_kinematics
    arm.get_inverse_kinematics = lambda pose, **kw: (10, [])
    with pytest.raises(RuntimeError, match='可达边界'):
        controller.command('home_save', {'confirmed': True})
    assert controller.recovery.home is None and arm.calls == []
    arm.get_inverse_kinematics = original
    arm.joints[1] = 3.05
    with pytest.raises(RuntimeError, match='接近遥操限位'):
        controller.command('home_save', {'confirmed': True})


def test_home_path_check_is_readonly_even_with_warning_14():
    clock, arm, controller = recovery_setup()
    controller.command('home_save', {'confirmed': True})
    arm.warning = 14
    controller.command('home_check')
    assert arm.calls == []
    assert arm.warning == 14
    assert controller.recovery.feedback['status'] == 'checked'


def test_home_return_uses_saved_pose_and_current_ik_branch_not_recorded_joints():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    arm.joints[6] = .7  # same TCP, a different redundant joint configuration
    controller.command('home_check')
    request['checked_start'] = controller.recovery.feedback['start']
    controller.command('home_start', request)
    assert controller.recovery.motion is not None
    assert controller.recovery.motion['type'] == 'pose'
    assert arm.last_recovery_target[6] == pytest.approx(.7)
    assert controller.recovery.home['joints_rad'][6] == 0
    assert arm.calls.index('planned_hold') < arm.calls.index('joint_recover')
    assert controller.mode == 'preview'
    arm.arrive()
    clock.now += .3
    controller.command('heartbeat')
    controller.tick()
    assert controller.recovery.motion is None and arm.state == 4
    assert controller.recovery.feedback['status'] == 'completed'
    assert controller.mode == 'preview' and not controller.servo_started


def test_old_joint_diagnostics_are_not_used_as_a_pose_target():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller,arm)
    controller.recovery.home['version'] = 1
    controller.recovery.home['joints_rad'] = [99]*7
    controller.command('home_check')
    plan = controller.recovery.feedback
    assert plan['target'] == pytest.approx(np.zeros(7))
    assert plan['target_tcp'] == pytest.approx(arm.home)
    assert 'joint_recover' not in arm.calls


def test_pose_home_rejects_no_ik_and_wrong_fk_without_motion():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller,arm)
    arm.get_inverse_kinematics = lambda *a,**kw: (10,[])
    with pytest.raises(RuntimeError,match='Pose 逆解失败'):
        controller.command('home_start',request)
    assert 'joint_recover' not in arm.calls and 'enable' not in arm.calls
    arm.get_inverse_kinematics = lambda *a,**kw: (0,[0,.1,0,0,0,0,0])
    with pytest.raises(RuntimeError,match='正解验证不匹配'):
        controller.command('home_check')
    assert 'joint_recover' not in arm.calls


def test_already_at_pose_does_not_rearrange_joints_and_checks_orientation():
    clock, arm, controller = recovery_setup()
    controller.command('home_save',{'confirmed':True})
    arm.joints[6] = .8
    arm.get_inverse_kinematics = lambda *a,**kw: (_ for _ in ()).throw(AssertionError('already at pose'))
    controller.command('home_check')
    assert controller.recovery.feedback['target'][6] == pytest.approx(.8)
    # A small orientation residual within tolerance may complete.
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller,arm)
    controller.command('home_start',request)
    arm.arrive();arm.joints[3]=np.deg2rad(.4)
    arm.tcp=np.array(arm.get_forward_kinematics(arm.joints)[1])
    # .4 degree is in the checked corridor and orientation completion tolerance.
    clock.now+=.3;controller.command('heartbeat');controller.tick()
    assert controller.recovery.feedback['status']=='completed'


def test_pose_completion_requires_orientation_not_only_translation():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller,arm)
    controller.command('home_start',request)
    arm.arrive()
    # Both axes stay within the .5-degree checked joint corridor, but their
    # combined quaternion angle exceeds the .5-degree POSE tolerance.
    arm.joints[3:5] = np.deg2rad(.4)
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    clock.now += .3; controller.command('heartbeat'); controller.tick()
    assert controller.recovery.motion is not None
    assert controller.recovery.feedback['status'] == 'moving'
    arm.joints[3:5] = np.deg2rad(.3)
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    clock.now += .1; controller.command('heartbeat'); controller.tick()
    assert controller.recovery.feedback['status'] == 'completed'


def test_home_start_requires_confirmation_same_profile_and_same_start():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    for bad, reason in ((dict(request, confirmed=False),'确认'),
                        (dict(request, home_saved_at=0),'初始点'),
                        (dict(request, checked_start=[1]*7),'关节位置')):
        with pytest.raises(RuntimeError, match=reason):
            controller.command('home_start', bad)
        assert 'joint_recover' not in arm.calls


def test_joint_jog_can_recover_within_hardware_but_outside_teleop_margin():
    clock, arm, controller = recovery_setup()
    arm.joints[1] = 3.05
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    request = dict(confirmed=True, joint=1, step_deg=-2, start_joints_deg=np.rad2deg(arm.joints).tolist())
    controller.command('joint_recover', request)
    assert arm.last_recovery_target[1] < 3.05
    assert controller.recovery.home is None


def test_joint_jog_outside_hardware_limits_is_refused_and_no_fault_is_cleared():
    clock, arm, controller = recovery_setup()
    arm.joints[1] = 3.2
    request = dict(confirmed=True, joint=1, step_deg=-2, start_joints_deg=np.rad2deg(arm.joints).tolist())
    with pytest.raises(RuntimeError, match='硬件限位'):
        controller.command('joint_recover', request)
    assert arm.calls == []
    arm.joints[1] = 0
    arm.error = 31
    with pytest.raises(RuntimeError, match='不自动清故障'):
        controller.command('joint_recover', dict(request, start_joints_deg=[0]*7))
    assert arm.calls == []


def test_recovery_refuses_moving_robot_tool_change_and_long_rotation():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    arm.velocity[1] = .1
    with pytest.raises(RuntimeError, match='仍在移动'):
        controller.command('home_check')
    arm.velocity[:] = 0
    arm.tcp_offset[0] = 1
    with pytest.raises(RuntimeError, match='配置不匹配'):
        controller.command('home_check')
    with pytest.raises(RuntimeError, match='180'):
        JointRecovery(arm, [-7]*7,[7]*7).plan([0]*7,[4]*7,'test')


def test_recovery_stop_and_heartbeat_failure_cancel_motion():
    for stop in ('stop','heartbeat'):
        clock, arm, controller = recovery_setup()
        request = save_and_displace(controller, arm)
        controller.command('home_start', request)
        if stop == 'stop':
            controller.command('stop')
        else:
            clock.now += 1
            controller.tick()
        assert arm.state == 4
        assert controller.recovery.motion is None
        assert controller.mode == 'preview'


def test_recovery_precheck_is_rerun_and_fk_nan_never_sends_motion():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    original = arm.get_forward_kinematics
    arm.get_forward_kinematics = lambda *a, **kw: (0, [float('nan')]*6)
    with pytest.raises(RuntimeError, match='状态无效'):
        controller.command('home_start', request)
    assert 'joint_recover' not in arm.calls
    arm.get_forward_kinematics = original
    controller.recovery.lower[1] = .1
    with pytest.raises(RuntimeError, match='限位'):
        controller.command('home_start', request)
    assert 'joint_recover' not in arm.calls


def test_recovery_initialization_failure_stops_and_does_not_leave_owner_active():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    arm.motion_enable = lambda **kw: 9
    with pytest.raises(RuntimeError, match='复位使能'):
        controller.command('home_start', request)
    assert arm.state == 4 and controller.recovery.motion is None
    assert 'joint_recover' not in arm.calls


def test_command_rejection_during_recovery_and_runtime_deviation_are_stopped():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    controller.command('home_start', request)
    for command in ('allow_controller','home_save','joint_recover','gripper_open','configure'):
        with pytest.raises(RuntimeError,match='正在运动'):
            controller.command(command, {'confirmed':True})
    arm.joints[0] = .1
    with pytest.raises(RuntimeError,match='偏离检查路径'):
        controller.tick()
    controller.release('故障停止', disarm=True)
    assert arm.state == 4 and controller.recovery.motion is None


def test_completed_recovery_keeps_grasp_unless_restore_was_explicitly_selected():
    class Grip:
        def __init__(self): self.jobs=[]
        def snapshot(self):
            return dict(connected=True, command_fault=False, position_mm=84, position_age_s=0)
        def cancel_pending(self): pass
        def submit(self,target,speed,force,allowed):
            assert allowed()
            self.jobs.append(target)
    for restore in (False,True):
        clock, arm, controller = recovery_setup()
        controller.gripper = grip = Grip()
        controller.settings['gripper_enabled'] = True
        request = save_and_displace(controller, arm)
        request['restore_gripper'] = restore
        controller.command('home_start',request)
        assert grip.jobs == []
        arm.arrive(); clock.now += .3
        controller.command('heartbeat'); controller.tick()
        assert grip.jobs == ([84] if restore else [])


def test_recovery_rechecks_every_path_sample_and_rejects_nonfinite_fk():
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    original = arm.get_forward_kinematics
    def invalid_path(joints,**kw):
        if .01 < joints[1] < .19: return 0,[float('nan')]*6
        return original(joints,**kw)
    arm.get_forward_kinematics = invalid_path
    with pytest.raises(ValueError,match='finite'):
        controller.command('home_start',request)
    assert 'joint_recover' not in arm.calls


@pytest.mark.parametrize('speed', [5, 15, 30])
def test_selected_home_speed_reaches_sdk_and_stop_still_works(speed):
    clock, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    arm.joints[1] = 1.
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    controller.command('home_check', {'speed_deg_s': speed})
    assert arm.calls == []
    feedback = controller.recovery.feedback
    request.update(speed_deg_s=speed, checked_start=feedback['start'])
    arm.expected_speed, arm.expected_acceleration = speed, 2*speed
    controller.command('home_start', request)
    motion = controller.recovery.motion
    assert arm.last_motion_kwargs['speed'] == pytest.approx(np.deg2rad(speed))
    assert motion['deadline']-motion['started'] == pytest.approx(max(10, np.rad2deg(1.)/speed*2+5))
    clock.now += .1
    controller.command('heartbeat'); controller.tick()
    assert f'{speed}°/s' in controller.status
    controller.command('stop')
    assert arm.state == 4 and controller.recovery.motion is None


@pytest.mark.parametrize('speed', [None, True, '15', 0, 4.9, 30.1, float('nan'), float('inf')])
@pytest.mark.parametrize('command', ['home_check', 'home_start'])
def test_invalid_home_speed_never_enables_or_moves_robot(speed, command):
    _, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    with pytest.raises(ValueError, match='复位速度'):
        controller.command(command, dict(request, speed_deg_s=speed))
    assert arm.calls == []


def test_changed_speed_requires_recheck_before_home_start():
    _, arm, controller = recovery_setup()
    request = save_and_displace(controller, arm)
    with pytest.raises(RuntimeError, match='速度已改变'):
        controller.command('home_start', dict(request, speed_deg_s=15))
    assert arm.calls == []
    controller.command('home_check', {'speed_deg_s': 15})
    # Omitting the speed must not silently run a previously checked faster plan.
    with pytest.raises(RuntimeError, match='速度已改变'):
        controller.command('home_start', request)
    assert arm.calls == []
    controller.command('home_check', {'speed_deg_s': 15})
    arm.expected_speed, arm.expected_acceleration = 15, 30
    controller.command('home_start', dict(request, speed_deg_s=15))
    assert controller.recovery.motion['speed_deg_s'] == 15


def test_faster_home_speed_keeps_joint_jog_at_five():
    _, arm, controller = recovery_setup()
    save_and_displace(controller, arm)
    controller.command('home_check', {'speed_deg_s': 30})
    controller.command('joint_recover', dict(confirmed=True, joint=1, step_deg=-2,
        start_joints_deg=np.rad2deg(arm.joints).tolist(), speed_deg_s=30))
    assert controller.recovery.motion['speed_deg_s'] == 5
