import numpy as np
import pytest

from quest_relative_motion import RelativeTranslation, continuous_solution, direction_matrix, yaw_matrix, settings, tether, smooth_target, smooth_rotation, command_velocities, rotation_distance, CONTROL_PERIOD
from scipy.spatial.transform import Rotation
from xarm_quest_relative import RelativeArm, prepare_servo_hold
from quest_pose_math import pose_to_sdk, pose_rpy_deg


class Clock:
    def __init__(self):
        self.now = 10.

    def __call__(self):
        return self.now


class FakeArm:
    def __init__(self):
        self.home = np.array([400., 0., 300., 0., 0., 0.])
        self.tcp = self.home.copy()
        self.joints = np.zeros(7)
        self.calls = []
        self.warning = 14
        self.error = 0
        self.mode = 0
        self.state = 4

    def get_position(self, **kwargs):
        return 0, self.tcp.tolist()

    def get_servo_angle(self, **kwargs):
        return 0, self.joints.tolist()

    def get_err_warn_code(self):
        return 0, [self.error, self.warning]

    def clean_warn(self):
        self.calls.append("clean_warn")
        self.warning = 0
        return 0

    def get_inverse_kinematics(self, pose, **kwargs):
        result = np.zeros(7)
        result[:3] = (np.asarray(pose)[:3] - self.home[:3]) / 1000
        result[3:6] = np.asarray(pose)[3:] - self.home[3:]
        return 0, result.tolist()

    def get_forward_kinematics(self, joints, **kwargs):
        result = self.home.copy()
        result[:3] += np.asarray(joints)[:3] * 1000
        result[3:] += np.asarray(joints)[3:6]
        return 0, result.tolist()

    def motion_enable(self, **kwargs):
        self.calls.append("enable")
        return 0

    def set_mode(self, mode):
        self.calls.append(("mode", mode))
        self.mode = mode
        return 0

    def set_state(self, state):
        self.calls.append(("state", state))
        self.state = state
        return 0

    def get_state(self):
        return 0, self.state

    def set_servo_angle(self, angle, **kwargs):
        assert self.mode == 0
        assert np.asarray(angle) == pytest.approx(self.joints)
        self.calls.append("planned_hold")
        return 0

    def set_servo_angle_j(self, joints, **kwargs):
        self.calls.append("hold_seed" if self.state == 3 else "servo")
        self.joints = np.asarray(joints)
        _, tcp = self.get_forward_kinematics(joints)
        self.tcp = np.asarray(tcp)
        return 0

    def set_position(self, *target, **kwargs):
        assert kwargs["motion_type"] == 0
        assert kwargs["speed"] == 20
        assert kwargs["wait"] is False
        self.calls.append("demo_linear")
        self.tcp = np.asarray(target)
        self.joints[:3] = (self.tcp[:3] - self.home[:3]) / 1000
        return 0


def state(position=(0.2, 1., 0.3), enabled=True, valid=True, invalid=0, pause=0, count=1):
    return {"valid": valid, "position_m": list(position), "motion_enabled": enabled,
            "quaternion_wxyz": [1, 0, 0, 0], "trigger_pressed": False, "a_pressed": False,
            "valid_age_s": 0, "pause_age_s": 0, "invalid_count": invalid,
            "pause_generation": pause, "packet_count": count}


def setup():
    clock, arm = Clock(), FakeArm()
    controller = RelativeArm(arm, [-3] * 7, [3] * 7, clock, control_settings={"speed_mm_s": 10, "translation_scale": .25, "joint_speed_rad_s": .15, "range_mm": 10, "teleop_button": "b"})
    return clock, arm, controller


def tick(clock, controller, payload):
    clock.now += 1 / 30
    controller.command("heartbeat")
    controller.update_source(payload)
    controller.tick()


def engage(clock, controller):
    controller.update_source(state())
    controller.command("enable")
    tick(clock, controller, state(enabled=False, valid=False, invalid=1, pause=1, count=2))
    tick(clock, controller, state(invalid=1, pause=1, count=3))


def test_mapping_and_clipping_are_relative_and_pose_is_fixed():
    mapper = RelativeTranslation(radius_mm=10)
    mapper.reset_origin([400, 0, 300, .1, .2, .3])
    mapper.engage([.2, 1, .3], [400, 0, 300, .1, .2, .3])
    requested, bounded = mapper.target([.24, 1.02, .34])
    assert requested[:3] == pytest.approx([410, -10, 305])
    assert requested[3:] == pytest.approx([.1, .2, .3])
    _, bounded = mapper.target([2, 3, 4])
    assert bounded[:3] == pytest.approx([410, -10, 310])


def test_step_limit_cannot_catch_up_after_a_long_delay():
    mapper = RelativeTranslation()
    result = mapper.limited_target([410, 0, 300, 0, 0, 0], [400, 0, 300, 0, 0, 0], 10)
    assert result[0] - 400 == pytest.approx(10 * CONTROL_PERIOD)


def test_reengage_does_not_jump_and_cannot_walk_out_of_original_box():
    mapper = RelativeTranslation(radius_mm=10)
    mapper.reset_origin([400, 0, 300, 0, 0, 0])
    mapper.engage([.2, 1, .3], [409, 0, 300, 0, 0, 0])
    mapper.release()
    mapper.engage([1, 2, 3], [409, 0, 300, 0, 0, 0])
    _, bounded = mapper.target([1, 2, 3])
    assert bounded[:3] == pytest.approx([409, 0, 300])
    _, bounded = mapper.target([1, 2, 4])
    assert bounded[:3] == pytest.approx([410, 0, 300])


def test_preview_never_calls_motion_methods():
    clock, arm, controller = setup()
    tick(clock, controller, state())
    tick(clock, controller, state(position=(.2, 1, .34), count=2))
    assert controller.bounded[0] == pytest.approx(410)
    assert arm.calls == []


def test_enable_alone_does_not_enable_motors_or_move():
    clock, arm, controller = setup()
    controller.update_source(state())
    controller.command("enable")
    tick(clock, controller, state(position=(.2, 1, .4), count=2))
    assert arm.calls == ["clean_warn"]
    assert not controller.servo_started


def test_b_cycle_engages_and_hand_delta_moves_robot_at_limited_speed():
    clock, arm, controller = setup()
    engage(clock, controller)
    assert controller.servo_started
    assert controller.command_count == 0
    tick(clock, controller, state(position=(.2, 1, .32), invalid=1, pause=1, count=4))
    assert 400 < arm.tcp[0] <= 400 + 10 / 30 + 1e-6
    assert controller.command_count == 1
    assert arm.tcp[1:] == pytest.approx([0, 300, 0, 0, 0])


def test_invalid_frame_cancels_last_servo_target_and_requires_new_takeover():
    clock, arm, controller = setup()
    engage(clock, controller)
    tick(clock, controller, state(valid=False, invalid=2, pause=1, count=4))
    assert ("state", 4) in arm.calls
    assert not controller.servo_started
    tick(clock, controller, state(position=(.2, 1, .5), invalid=2, pause=1, count=5))
    assert not controller.servo_started
    assert controller.command_count == 0


def test_tracking_timeout_stops_even_if_no_new_packet_arrives():
    clock, arm, controller = setup()
    engage(clock, controller)
    clock.now += .4
    controller.command("heartbeat")
    controller.tick()
    assert not controller.servo_started
    assert ("state", 4) in arm.calls


def test_ui_heartbeat_loss_disarms_motion():
    clock, arm, controller = setup()
    engage(clock, controller)
    clock.now += 1
    controller.update_source(state(invalid=1, pause=1, count=4))
    controller.tick()
    assert controller.mode == "preview"
    assert not controller.servo_started
    assert ("state", 4) in arm.calls


def test_stop_command_holds_and_disarms():
    clock, arm, controller = setup()
    engage(clock, controller)
    controller.command("stop")
    assert controller.mode == "preview"
    assert not controller.servo_started
    assert ("state", 4) in arm.calls


def test_robot_error_is_not_cleared_automatically():
    clock, arm, controller = setup()
    controller.update_source(state())
    arm.error = 21
    with pytest.raises(RuntimeError, match="错误/警告"):
        controller.command("enable")
    assert arm.calls == []


def test_wrong_or_stale_hand_blocks_enable():
    clock, arm, controller = setup()
    controller.update_source(state(valid=False))
    with pytest.raises(RuntimeError, match="实时有效"):
        controller.command("enable")
    assert arm.calls == []


def test_controller_permission_without_pose_never_enables_motors():
    clock, arm, controller = setup()
    controller.command("allow_controller")
    controller.tick()
    assert controller.mode == "motion"
    assert arm.calls == ["clean_warn"]
    assert not controller.servo_started


def test_starting_from_paused_only_needs_one_b_continue():
    clock, arm, controller = setup()
    controller.update_source(state(enabled=False, valid=False, pause=1))
    controller.command("allow_controller")
    tick(clock, controller, state(pause=1, count=2))
    assert controller.servo_started
    assert controller.command_count == 0


def test_external_reposition_disarms_before_any_takeover():
    clock, arm, controller = setup()
    controller.command("allow_controller")
    arm.joints[2] = .05
    tick(clock, controller, state(enabled=False, valid=False, pause=1))
    assert controller.mode == "preview"
    assert not controller.servo_started
    assert "暂停保持位置已改变" in controller.last_error


def test_singularity_branch_jump_is_rejected():
    with pytest.raises(RuntimeError, match="IK 分支"):
        continuous_solution([.5, 0, 0, 0, 0, 0, 0], np.zeros(7), [-3] * 7, [3] * 7)


def test_continuous_teleop_does_not_expire_at_sixty_seconds():
    clock, arm, controller = setup()
    engage(clock, controller)
    clock.now += 60.1
    controller.last_tick = clock.now - CONTROL_PERIOD  # control kept running, not a 60-second stall
    controller.command("heartbeat")
    controller.update_source(state(invalid=1, pause=1, count=4))
    controller.tick()
    assert controller.mode == "motion"
    assert controller.servo_started


def test_startup_preloads_hold_before_resuming_servo():
    arm = FakeArm()
    prepare_servo_hold(arm)
    assert arm.state == 0
    assert arm.calls.index("planned_hold") < arm.calls.index(("mode", 1))
    assert arm.calls.index(("state", 3)) < arm.calls.index(("mode", 1))
    assert arm.calls[-2:] == [("state", 0), "servo"]
    assert arm.tcp == pytest.approx(arm.home)


def test_direction_yaw_and_reflections_preserve_unit_axes():
    matrix = yaw_matrix(90)
    assert matrix @ [0, 0, 1] == pytest.approx([0, 1, 0])
    assert direction_matrix(matrix) == pytest.approx(matrix)
    flipped = matrix.copy()
    flipped[0] *= -1
    assert abs(np.linalg.det(direction_matrix(flipped))) == pytest.approx(1)


def test_invalid_direction_matrix_cannot_scale_or_duplicate_axes():
    with pytest.raises(ValueError, match="正交"):
        direction_matrix(np.eye(3) * 2)
    with pytest.raises(ValueError, match="正交"):
        direction_matrix([[1, 0, 0], [1, 0, 0], [0, 0, 1]])


def test_parameter_change_stops_and_waits_for_fresh_takeover_without_jump():
    clock, arm, controller = setup()
    engage(clock, controller)
    before = arm.tcp.copy()
    controller.command("configure", {"speed_mm_s": 50, "translation_scale": 1, "direction_matrix": yaw_matrix(180).tolist()})
    assert not controller.servo_started
    assert ("state", 4) in arm.calls
    tick(clock, controller, state(position=(.2, 1, .34), invalid=1, pause=1, count=4))
    assert arm.tcp == pytest.approx(before)
    assert controller.mapper.speed == 50
    assert controller.mapper.scale == 1
    assert not controller.servo_started


def test_invalid_settings_are_rejected_after_pausing():
    clock, arm, controller = setup()
    engage(clock, controller)
    with pytest.raises(ValueError):
        controller.command("configure", {"speed_mm_s": float("nan")})
    assert not controller.servo_started
    assert controller.mapper.speed == 10


def test_command_trajectory_reduces_lag_without_unbounded_lead():
    class LagArm(FakeArm):
        def set_servo_angle_j(self, joints, **kwargs):
            self.calls.append("servo")
            self.joints += (np.asarray(joints) - self.joints) * .2
            _, tcp = self.get_forward_kinematics(self.joints)
            self.tcp = np.asarray(tcp)
            return 0

    clock, arm = Clock(), LagArm()
    controller = RelativeArm(arm, [-3] * 7, [3] * 7, clock, control_settings={"speed_mm_s": 10, "translation_scale": .25, "joint_speed_rad_s": .15, "teleop_button": "b"})
    engage(clock, controller)
    for count in range(4, 104):
        # A ramp within robot speed retains exact displacement. An instant
        # large step now intentionally drops excess unexecuted gesture.
        tick(clock, controller, state(position=(.2, 1, .3+.04*min(1,(count-3)/70)), invalid=1, pause=1, count=count))
        assert np.linalg.norm(controller.command_tcp[:3] - controller.tcp[:3]) <= 1.1
    # Delayed measured-position-relative stepping advanced ~2 mm in 100
    # cycles. The interpolated command trajectory advances almost 10 mm.
    assert arm.tcp[0] > 408
    assert arm.tcp[0] <= 410


def test_tether_caps_cartesian_command_lead():
    assert tether([410, 0, 300, 0, 0, 0], [400, 0, 300, 0, 0, 0], 4)[0] == 404


def test_configured_tracking_tolerance_removes_speed_ceiling_but_stays_bounded():
    class DelayedArm(FakeArm):
        def set_servo_angle_j(self,joints,**kwargs):
            self.joints += (np.asarray(joints)-self.joints)*.08
            self.tcp=np.array(self.get_forward_kinematics(self.joints)[1])
            return 0
    travel=[]
    for limit in (5.,12.):
        clock,arm=Clock(),DelayedArm()
        controller=RelativeArm(arm,[-3]*7,[3]*7,clock,control_settings={
            'speed_mm_s':150,'translation_scale':2,'acceleration_mm_s2':1000,
            'joint_speed_rad_s':1,'tracking_lead_mm':limit,'teleop_button':'b'})
        engage(clock,controller)
        for count in range(4,204):
            tick_100(clock,controller,state(position=(.2,1,.45),invalid=1,pause=1,count=count))
            assert np.linalg.norm(controller.command_tcp[:3]-controller.tcp[:3])<=limit+.1
        travel.append(arm.tcp[0]-arm.home[0])
        tick_100(clock,controller,state(enabled=False,pause=2,count=204))
        assert not controller.servo_started and arm.state==4
    assert travel[1]>travel[0]*1.7,travel
    assert travel[1]<150*2
    for bad in (0,20.1,float('nan')):
        with pytest.raises(ValueError):settings({'tracking_lead_mm':bad})


def test_speed_and_gain_limits_are_independent():
    fast = settings({"speed_mm_s": 200, "translation_scale": 10, "joint_speed_rad_s": 1, 'rotation_scale':5})
    assert fast["speed_mm_s"] == 200
    assert fast["translation_scale"] == 10
    mapper = RelativeTranslation(scale=fast['translation_scale'],speed_mm_s=fast['speed_mm_s'])
    mapper.engage([.2,1,.3],[400,0,300,0,0,0])
    target,_ = mapper.target([.2,1,.4])
    assert target[0] == pytest.approx(1400)  # 10 cm at 10x, no small fence
    with pytest.raises(ValueError):
        settings({"speed_mm_s": 201})
    with pytest.raises(ValueError):
        settings({'translation_scale':10.01})
    with pytest.raises(ValueError):
        settings({'rotation_scale':5.01})


def test_explicit_stop_in_preview_requests_and_verifies_hardware_stop():
    clock, arm, controller = setup()
    arm.state = 0
    controller.command("stop")
    assert arm.state == 4
    assert controller.mode == "preview"
    assert ("state", 4) in arm.calls


def test_failed_stop_does_not_acknowledge_success():
    clock, arm, controller = setup()
    arm.set_state = lambda state: 9
    with pytest.raises(RuntimeError, match="停止机械臂"):
        controller.command("stop")


def test_stop_failure_during_active_teleoperation_is_not_acknowledged():
    clock, arm, controller = setup()
    engage(clock, controller)
    arm.set_state = lambda state: 9
    with pytest.raises(RuntimeError, match="停止状态未确认"):
        controller.command("stop")


def unbounded_setup():
    clock, arm = Clock(), FakeArm()
    controller = RelativeArm(arm, [-3] * 7, [3] * 7, clock, control_settings={"teleop_button": "b"})
    engage(clock, controller)
    assert controller.mapper.radius is None
    return clock, arm, controller


def tick_100(clock, controller, payload):
    clock.now += CONTROL_PERIOD
    controller.command("heartbeat")
    controller.update_source(payload)
    controller.tick()


def test_default_mapping_has_no_ten_millimeter_clipping():
    mapper = RelativeTranslation(scale=1)
    mapper.reset_origin([400, 0, 300, .1, .2, .3])
    mapper.engage([.2, 1, .3], mapper.origin)
    requested, target = mapper.target([.35, 1.12, .5])
    assert target == pytest.approx(requested)
    assert target[:3] == pytest.approx([600, -150, 420])
    assert mapper.contains([1000, -300, 800, 0, 0, 0])
    assert target[3:] == pytest.approx([.1, .2, .3])


def test_unbounded_simulated_servo_reaches_150mm_and_b_resumes_there():
    clock, arm, controller = unbounded_setup()
    for count in range(4, 1004):
        previous = controller.command_tcp.copy()
        tick_100(clock, controller, state(position=(.2, 1, .3+.15*min(1,(count-3)/600)), invalid=1, pause=1, count=count))
        assert np.linalg.norm(controller.command_tcp[:3] - previous[:3]) <= .54
        assert controller.mode == "motion"
    assert arm.tcp[0] == pytest.approx(550, abs=.1)
    assert controller.requested[0] == pytest.approx(550)
    assert controller.bounded[0] == pytest.approx(550)
    tick_100(clock, controller, state(position=(.2, 1, .45), enabled=False, invalid=1, pause=2, count=1004))
    assert not controller.servo_started
    tick_100(clock, controller, state(position=(.2, 1, .3), invalid=1, pause=2, count=1005))
    assert controller.servo_started
    assert arm.tcp[0] == pytest.approx(550, abs=.1)
    for count in range(1006, 1306):
        tick_100(clock, controller, state(position=(.2, 1, .3+.05*min(1,(count-1005)/200)), invalid=1, pause=2, count=count))
    assert arm.tcp[0] == pytest.approx(600, abs=.1)
    assert controller.mode == "motion"


def test_unbounded_input_loss_still_stops():
    clock, arm, controller = unbounded_setup()
    tick_100(clock, controller, state(valid=False, invalid=2, pause=1, count=4))
    assert not controller.servo_started
    assert arm.state == 4


def test_unbounded_joint_limit_still_blocks_command():
    clock, arm, controller = unbounded_setup()
    controller.upper[0] = .00001
    before = controller.command_count
    with pytest.raises(RuntimeError, match="限位"):
        tick_100(clock, controller, state(position=(.2, 1, .4), invalid=1, pause=1, count=4))
    assert controller.command_count == before


def test_unbounded_external_reposition_while_paused_still_disarms():
    clock, arm = Clock(), FakeArm()
    controller = RelativeArm(arm, [-3] * 7, [3] * 7, clock, control_settings={"teleop_button": "b"})
    controller.command("allow_controller")
    arm.joints[2] = .05
    tick_100(clock, controller, state(enabled=False, valid=False, pause=1))
    assert controller.mode == "preview"
    assert "保持位置已改变" in controller.last_error
    assert not controller.servo_started


def test_control_stall_stops_instead_of_large_catch_up_step():
    clock, arm, controller = unbounded_setup()
    before = controller.command_count
    clock.now += .1
    controller.command("heartbeat")
    controller.update_source(state(invalid=1, pause=1, count=4))
    controller.tick()
    assert controller.mode == "preview"
    assert arm.state == 4
    assert controller.command_count == before
    assert "周期超时" in controller.last_error


def test_smooth_interpolation_bounds_speed_acceleration_and_reversal():
    current = np.array([400., 0, 300, .1, .2, .3])
    target = current.copy()
    target[0] += 150
    velocity = np.zeros(3)
    for count in range(800):
        if count == 100:
            target[0] = 350  # direction reversal is not an instantaneous velocity flip
        previous = velocity.copy()
        new, velocity = smooth_target(target, current, velocity, CONTROL_PERIOD, 50, 300)
        assert np.linalg.norm(velocity) <= 50 + 1e-8
        assert np.linalg.norm(velocity - previous) <= 300 * CONTROL_PERIOD + 1e-8
        assert np.linalg.norm(new[:3] - current[:3]) <= 50 * CONTROL_PERIOD + 1e-8
        assert new[3:] == pytest.approx(current[3:])
        current = new
    assert current[0] == pytest.approx(350, abs=.1)


def test_smoothing_never_uses_a_long_delay_to_jump():
    current = np.array([400., 0, 300, 0, 0, 0])
    target = current.copy(); target[0] += 100
    result, velocity = smooth_target(target, current, np.zeros(3), 10, 50, 300)
    assert result[0] - current[0] <= .5
    assert velocity[0] <= 3


def test_default_settings_remove_radius_without_changing_direction_or_gain():
    matrix = yaw_matrix(23).tolist()
    result = settings({"direction_matrix": matrix, "translation_scale": 1})
    assert result["range_mm"] is None
    assert result["direction_matrix"] == matrix
    assert result["translation_scale"] == 1
    assert result["acceleration_mm_s2"] == 300


def test_nonfinite_command_fk_is_rejected_before_unbounded_motion():
    clock, arm, controller = unbounded_setup()
    original = arm.get_forward_kinematics
    def invalid_command_fk(joints, **kwargs):
        code, pose = original(joints, **kwargs)
        if np.linalg.norm(np.asarray(joints) - arm.joints) > 1e-6:
            pose[0] = float("nan")
        return code, pose
    arm.get_forward_kinematics = invalid_command_fk
    with pytest.raises(ValueError, match="finite"):
        tick_100(clock, controller, state(position=(.2, 1, .4), invalid=1, pause=1, count=4))
    assert controller.command_count == 0


def quaternion(rotation):
    q = rotation.as_quat()
    return [q[3], q[0], q[1], q[2]]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rotation_uses_handedness_correct_matrix_and_relative_anchor(axis):
    matrix = yaw_matrix(23)
    mapper = RelativeTranslation(scale=1, matrix=matrix, rotation_enabled=True)
    tcp = np.array([400, 0, 300, .3, -.2, .7])
    initial = Rotation.from_euler('xyz', [.1, .2, -.1])
    mapper.reset_origin(tcp)
    mapper.engage([.2, 1, .3], tcp, quaternion(initial))
    _, zero = mapper.target([.2, 1, .3], quaternion(initial))
    assert rotation_distance(zero[3:], tcp[3:]) < 1e-9
    hand_delta = Rotation.from_rotvec(np.eye(3)[axis] * .2)
    _, target = mapper.target([.2, 1, .3], quaternion(hand_delta * initial))
    mapped = matrix @ hand_delta.as_matrix() @ matrix.T
    expected = mapped @ Rotation.from_euler('xyz', tcp[3:]).as_matrix()
    assert Rotation.from_euler('xyz', target[3:]).as_matrix() == pytest.approx(expected)
    assert target[:3] == pytest.approx(tcp[:3])


def test_rotation_crossing_180_and_quaternion_sign_do_not_jump():
    mapper = RelativeTranslation(matrix=np.eye(3), rotation_enabled=True)
    tcp = [400, 0, 300, 0, 0, 0]
    initial = Rotation.from_euler('z', 179, degrees=True)
    mapper.reset_origin(tcp)
    mapper.engage([.2, 1, .3], tcp, quaternion(initial))
    _, target = mapper.target([.2, 1, .3], quaternion(Rotation.from_euler('z', -179, degrees=True)))
    assert np.rad2deg(rotation_distance(target[3:], [0, 0, 0])) == pytest.approx(2)
    _, same = mapper.target([.2, 1, .3], [-v for v in quaternion(initial)])
    assert rotation_distance(same[3:], [0, 0, 0]) < 1e-9


@pytest.mark.parametrize('gain',[1.5,2.,3.5,5.])
def test_continuous_rotation_gain_crosses_full_gesture_180_without_jump(gain):
    mapper = RelativeTranslation(matrix=np.eye(3),rotation_enabled=True,rotation_scale=gain)
    tcp = [400,0,300,.2,-.3,.4]
    mapper.engage([.2,1,.3],tcp,[1,0,0,0])
    previous = Rotation.from_euler('xyz',tcp[3:])
    for angle in range(1,241):
        target,_ = mapper.target([.2,1,.3],quaternion(Rotation.from_euler('z',angle,degrees=True)))
        current = Rotation.from_euler('xyz',target[3:])
        step = (current*previous.inv()).as_rotvec()
        assert np.rad2deg(step) == pytest.approx([0,0,gain],abs=1e-9)
        previous = current


@pytest.mark.parametrize('axis',[0,1,2])
def test_combined_orientation_has_equal_incremental_gain_on_each_world_axis(axis):
    matrix = yaw_matrix(31)
    mapper = RelativeTranslation(matrix=matrix,rotation_enabled=True,rotation_scale=2)
    origin=Rotation.from_euler('xyz',[.4,-.7,.3])
    tcp=[400,0,300,.2,-.1,1.]
    mapper.engage([.2,1,.3],tcp,quaternion(origin))
    prior=Rotation.from_euler('xyz',tcp[3:])
    hand=origin
    for i in [1,2,0,axis]:
        delta=np.eye(3)[i]*.08
        hand=Rotation.from_rotvec(delta)*hand
        target,_=mapper.target([.2,1,.3],quaternion(hand))
        actual=Rotation.from_euler('xyz',target[3:])
        assert (actual*prior.inv()).as_rotvec() == pytest.approx(2*np.linalg.det(matrix)*matrix@delta)
        prior=actual
    # A new clutch uses current actual pose; no accumulated old rotation.
    mapper.release();mapper.engage([.2,1,.3],tcp,quaternion(hand))
    target,_=mapper.target([.2,1,.3],quaternion(hand))
    assert rotation_distance(target[3:],tcp[3:]) < 1e-9


def test_disabled_rotation_axes_do_not_follow():
    mapper = RelativeTranslation(matrix=np.eye(3), rotation_enabled=True, rotation_axes=[False, True, False])
    mapper.reset_origin([400, 0, 300, 0, 0, 0])
    mapper.engage([.2, 1, .3], mapper.origin, [1, 0, 0, 0])
    _, target = mapper.target([.2, 1, .3], quaternion(Rotation.from_euler('x', .3)))
    assert rotation_distance(target[3:], [0, 0, 0]) < 1e-9


def test_smooth_rotation_limits_speed_acceleration_and_converges():
    current = np.zeros(3); velocity = np.zeros(3)
    target = [.1, .2, -.1]
    for _ in range(600):
        previous = velocity.copy()
        result, velocity = smooth_rotation(target, current, velocity, CONTROL_PERIOD, .5, 1.5)
        assert np.linalg.norm(velocity) <= .5 + 1e-9
        assert np.linalg.norm(velocity - previous) <= 1.5 * CONTROL_PERIOD + 1e-9
        assert rotation_distance(result, current) <= .5 * CONTROL_PERIOD + 1e-9
        current = result
    assert rotation_distance(current, target) < .002


def test_simulated_six_dof_follow_and_resume_do_not_reset_orientation():
    clock, arm, controller = unbounded_setup()
    q = quaternion(Rotation.from_euler('xyz', [.1, .05, -.1]))
    for count in range(4, 604):
        fraction = min(1,(count-3)/300)
        payload = state(position=(.2, 1, .3+.05*fraction), invalid=1, pause=1, count=count)
        payload['quaternion_wxyz'] = quaternion(Rotation.from_euler('xyz', np.array([.1,.05,-.1])*fraction))
        tick_100(clock, controller, payload)
    assert arm.tcp[0] == pytest.approx(450, abs=.1)
    assert rotation_distance(arm.tcp[3:], controller.bounded[3:]) < .003
    before = arm.tcp.copy()
    tick_100(clock, controller, state(enabled=False, invalid=1, pause=2, count=604))
    tick_100(clock, controller, state(invalid=1, pause=2, count=605))
    assert arm.tcp == pytest.approx(before)
    tick_100(clock, controller, state(invalid=1, pause=2, count=606))
    assert rotation_distance(controller.bounded[3:], before[3:]) < 1e-9


def test_rotation_discontinuity_releases_before_servo_command():
    clock, arm, controller = unbounded_setup()
    payload = state(invalid=1, pause=1, count=4)
    payload['quaternion_wxyz'] = quaternion(Rotation.from_euler('x', 60, degrees=True))
    tick_100(clock, controller, payload)
    assert not controller.servo_started
    assert arm.state == 4
    assert controller.command_count == 0


class FakeGripperWorker:
    def __init__(self):
        self.jobs = []
        self.cancelled = 0
        self.target = None
    def snapshot(self):
        return {"command_fault": False, "position_mm": 84, "position_age_s": 0,
                "target_mm": self.target, "committed_target_mm": self.target, "error": ""}
    def submit(self, target, speed, force, allowed):
        self.jobs.append((target, speed, force, allowed))
        self.target = target
    def cancel_pending(self):
        self.cancelled += 1


def test_a_hold_closes_release_opens_and_stream_pause_never_opens():
    clock, arm, controller = unbounded_setup()
    grip = FakeGripperWorker()
    controller.gripper = grip
    controller.settings['gripper_enabled'] = True
    tick_100(clock,controller,state(invalid=1,pause=1,count=4))
    for count in range(5, 24):
        payload = state(invalid=1, pause=1, count=count)
        payload['a_pressed'] = True
        tick_100(clock, controller, payload)
    assert [j[0] for j in grip.jobs] == [0]
    for count in range(24, 34):
        tick_100(clock, controller, state(invalid=1, pause=1, count=count))
    assert [j[0] for j in grip.jobs] == [0,84]
    for count in range(34, 54):
        payload = state(invalid=1, pause=1, count=count)
        payload['a_pressed'] = True
        tick_100(clock, controller, payload)
    assert [j[0] for j in grip.jobs] == [0,84,0]
    tick_100(clock, controller, state(enabled=False, invalid=1, pause=2, count=54))
    assert [j[0] for j in grip.jobs] == [0,84,0]
    assert grip.cancelled >= 1
    assert not grip.jobs[-1][3]()


def test_manual_gripper_requires_explicit_enable_and_does_not_move_arm():
    clock, arm, controller = setup()
    controller.gripper = FakeGripperWorker()
    with pytest.raises(RuntimeError, match="勾选"):
        controller.command('gripper_open')
    controller.settings['gripper_enabled'] = True
    controller.command('gripper_close')
    assert controller.gripper.jobs[0][0] == 0
    assert arm.calls == []


def test_rotation_and_gripper_setting_validation():
    for bad in ({'rotation_axes': [True, False]}, {'rotation_enabled': 1},
                {'gripper_open_mm': 100}, {'gripper_close_mm': 84}, {'rotation_speed_deg_s': 0}):
        with pytest.raises(ValueError):
            settings(bad)


def test_held_a_at_takeover_requires_release_then_new_press():
    clock, arm = Clock(), FakeArm()
    grip = FakeGripperWorker()
    controller = RelativeArm(arm, [-3]*7, [3]*7, clock, control_settings={'gripper_enabled': True, 'teleop_button': 'b'}, gripper=grip)
    controller.update_source(state(enabled=False, pause=1))
    controller.command('allow_controller')
    payload = state(pause=1, count=2)
    payload['a_pressed'] = True
    tick_100(clock, controller, payload)
    for count in range(3, 23):
        payload['packet_count'] = count
        tick_100(clock, controller, payload)
    assert grip.jobs == []
    for count in range(23, 33):
        tick_100(clock, controller, state(pause=1, count=count))
    assert grip.jobs == []
    for count in range(33, 53):
        payload['packet_count'] = count
        tick_100(clock, controller, payload)
    assert [j[0] for j in grip.jobs] == [0]


def test_gripper_fault_pauses_arm_without_opening_gripper():
    clock, arm, controller = unbounded_setup()
    grip = FakeGripperWorker()
    grip.snapshot = lambda: {'command_fault': True, 'error': 'G2 error'}
    controller.gripper = grip
    controller.settings['gripper_enabled'] = True
    tick_100(clock, controller, state(invalid=1, pause=1, count=4))
    assert controller.mode == 'preview' and arm.state == 4
    assert grip.jobs == []
    assert controller.last_error == 'G2 error'


def trigger_setup(held=False):
    clock, arm = Clock(), FakeArm()
    controller = RelativeArm(arm, [-3]*7, [3]*7, clock)
    payload = state()
    payload['trigger_pressed'] = held
    controller.update_source(payload)
    controller.command('allow_controller')
    return clock, arm, controller


def button_frames(clock, controller, pressed, frames=10, **changes):
    changes = dict(changes)
    if 'enabled' in changes:
        changes['motion_enabled'] = changes.pop('enabled')
    if 'position' in changes:
        changes['position_m'] = list(changes.pop('position'))
    if 'pause' in changes:
        changes['pause_generation'] = changes.pop('pause')
    if 'invalid' in changes:
        changes['invalid_count'] = changes.pop('invalid')
    for _ in range(frames):
        payload = state(count=controller.last_count + 1)
        payload.update(trigger_pressed=pressed, **changes)
        tick_100(clock, controller, payload)


def test_trigger_hold_follows_and_single_release_frame_stops_without_gripper_action():
    clock, arm, controller = trigger_setup()
    grip = FakeGripperWorker()
    controller.gripper = grip
    controller.settings['gripper_enabled'] = True
    button_frames(clock, controller, False)
    assert not controller.servo_started
    button_frames(clock, controller, True)
    assert controller.servo_started
    button_frames(clock, controller, True, position=(.2, 1, .32))
    assert controller.servo_started and controller.command_count > 0
    assert grip.jobs == []
    commands = controller.command_count
    button_frames(clock, controller, False, frames=1, position=(.2, 1, .32))
    assert not controller.servo_started and arm.state == 4
    assert controller.command_count == commands
    assert controller.mode == 'motion'
    before = arm.tcp.copy()
    button_frames(clock, controller, False, position=(.4, 1, .32))
    button_frames(clock, controller, True, position=(.4, 1, .32))
    assert controller.servo_started
    assert arm.tcp == pytest.approx(before)
    assert controller.mapper.hand_anchor == pytest.approx([.4, 1, .32])
    assert grip.jobs == []


def test_held_trigger_on_enable_does_not_start_until_release_and_new_press():
    clock, arm, controller = trigger_setup(held=True)
    button_frames(clock, controller, True)
    assert not controller.servo_started
    button_frames(clock, controller, False)
    assert not controller.servo_started
    button_frames(clock, controller, True)
    assert controller.servo_started


def test_trigger_short_bounce_is_rejected_but_stable_press_does_not_wait_60_ms():
    clock,arm,controller=trigger_setup()
    button_frames(clock,controller,True,frames=2)
    assert not controller.servo_started
    button_frames(clock,controller,True,frames=2)
    assert controller.servo_started
    button_frames(clock,controller,False,frames=1)
    assert not controller.servo_started and arm.state==4


def test_responsive_profile_reduces_ramp_lag_with_speed_and_stop_guards():
    from quest_control_limits import RESPONSIVE_SETTINGS
    errors=[]
    for profile in ({},RESPONSIVE_SETTINGS):
        clock,arm,controller=trigger_setup()
        controller.command('configure',profile)
        controller.command('allow_controller')
        button_frames(clock,controller,False)
        button_frames(clock,controller,True)
        for i in range(1,101):
            button_frames(clock,controller,True,frames=1,
                position=(.2,1,.3+.0006*i),
                quaternion_wxyz=quaternion(Rotation.from_euler('y',.4*i,degrees=True)))
            assert np.linalg.norm(controller.command_tcp[:3]-controller.tcp[:3])<=5.1
        target=controller.bounded.copy()
        errors.append([np.linalg.norm(target[:3]-arm.tcp[:3]),rotation_distance(target[3:],arm.tcp[3:])])
        button_frames(clock,controller,False,frames=1)
        assert arm.state==4 and not controller.servo_started
    assert np.all(np.array(errors[1])<np.array(errors[0])*.6),errors


def test_trigger_bounce_and_gui_takeover_do_not_start_motion():
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True, frames=2)
    button_frames(clock, controller, False)
    assert not controller.servo_started
    controller.command('takeover')
    button_frames(clock, controller, False)
    assert not controller.servo_started
    button_frames(clock, controller, True)
    assert controller.servo_started


def test_b_resume_alone_never_starts_trigger_teleop():
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, False, enabled=False, pause=1)
    button_frames(clock, controller, False, pause=1)
    assert not controller.servo_started
    button_frames(clock, controller, True, pause=1)
    assert controller.servo_started
    button_frames(clock, controller, True, enabled=False, pause=2)
    assert not controller.servo_started
    button_frames(clock, controller, True, pause=2)
    assert not controller.servo_started
    button_frames(clock, controller, False, pause=2)
    button_frames(clock, controller, True, pause=2)
    assert controller.servo_started


def test_invalid_trigger_releases_and_tracking_recovery_does_not_autostart():
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True)
    button_frames(clock, controller, None)
    assert not controller.servo_started
    button_frames(clock, controller, True)
    assert not controller.servo_started
    button_frames(clock, controller, False)
    button_frames(clock, controller, True)
    assert controller.servo_started
    button_frames(clock, controller, True, valid=False, invalid=1)
    assert not controller.servo_started
    button_frames(clock, controller, True, invalid=1)
    assert not controller.servo_started


def test_a_while_trigger_teleop_hold_controls_only_gripper():
    clock, arm, controller = trigger_setup()
    grip = FakeGripperWorker()
    controller.gripper = grip
    controller.settings['gripper_enabled'] = True
    button_frames(clock, controller, True)
    button_frames(clock, controller, True, a_pressed=True)
    assert controller.servo_started
    assert [job[0] for job in grip.jobs] == [0]
    button_frames(clock, controller, True, a_pressed=False)
    button_frames(clock, controller, True, a_pressed=True)
    assert [job[0] for job in grip.jobs] == [0,84,0]
    button_frames(clock, controller, False, a_pressed=True)
    assert not controller.servo_started
    assert [job[0] for job in grip.jobs] == [0,84,0]


def test_legacy_apk_missing_a_does_not_invent_a_gripper_press():
    clock, arm, controller = trigger_setup()
    grip = FakeGripperWorker()
    controller.gripper = grip
    controller.settings['gripper_enabled'] = True
    button_frames(clock, controller, True, a_pressed=None)
    button_frames(clock, controller, True, a_pressed=None)
    assert controller.servo_started and grip.jobs == []
    controller.command('gripper_close')
    assert [job[0] for job in grip.jobs] == [0]


def test_follow_ik_checks_previous_command_not_delayed_actual_joint_angles():
    clock, arm, controller = trigger_setup()
    previous = arm.tcp.copy()
    previous[4] = np.deg2rad(3)
    _, joints = arm.get_inverse_kinematics(previous)
    target = previous.copy(); target[4] += np.deg2rad(.2)
    with pytest.raises(RuntimeError, match='2°'):
        _, old_solution = arm.get_inverse_kinematics(target)
        continuous_solution(old_solution, controller.joints, controller.lower, controller.upper)
    result, shrinks = controller.solve_follow_target(target, previous, np.array(joints), clock())
    assert shrinks == 0
    assert result[4] == pytest.approx(target[4])
    assert arm.calls == ['clean_warn']


def test_valid_rotation_ik_adapts_step_instead_of_releasing_at_two_degrees():
    clock, arm, controller = trigger_setup()
    target = arm.tcp.copy(); target[4] = np.deg2rad(3)
    result, shrinks = controller.solve_follow_target(target, arm.tcp, arm.joints, clock())
    assert shrinks == 1
    assert np.rad2deg(result[4]) == pytest.approx(1.5)
    assert np.max(np.abs(result - arm.joints)) <= .035


def test_true_ik_branch_jump_still_fails_all_subdivisions_without_motion():
    clock, arm, controller = trigger_setup()
    arm.get_inverse_kinematics = lambda *a, **kw: (0, [2]*7)
    target = arm.tcp.copy(); target[4] = .01
    with pytest.raises(RuntimeError, match='当前方向不可连续'):
        controller.solve_follow_target(target, arm.tcp, arm.joints, clock())
    assert not controller.servo_started
    assert 'servo' not in arm.calls


def test_follow_ik_nonfinite_and_timeout_are_not_bypassed():
    clock, arm, controller = trigger_setup()
    arm.get_inverse_kinematics = lambda *a, **kw: (0, [float('nan')]*7)
    with pytest.raises(ValueError, match='finite'):
        controller.solve_follow_target(arm.tcp, arm.tcp, arm.joints, clock())
    with pytest.raises(RuntimeError, match='超时'):
        controller.solve_follow_target(arm.tcp, arm.tcp, arm.joints, clock()-.1)


def test_control_button_settings_reject_conflicting_or_unknown_mappings():
    assert settings({})['teleop_button'] == 'trigger'
    assert settings({})['gripper_button'] == 'a'
    for bad in ({'teleop_button': 'a'}, {'gripper_button': 'trigger'}, {'teleop_button': None}):
        with pytest.raises(ValueError):
            settings(bad)


def test_trigger_release_after_sdk_queries_prevents_next_servo_command():
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True)
    commands = controller.command_count
    def release_during_query():
        controller.update_source(state(count=controller.last_count + 1))
    controller.poll_source = release_during_query
    button_frames(clock, controller, True, frames=1, position=(.2, 1, .32))
    assert not controller.servo_started
    assert controller.command_count == commands
    assert arm.state == 4


def test_release_during_hold_initialization_cancels_servo_start():
    clock, arm, controller = trigger_setup()
    hold = arm.set_servo_angle
    def release_during_hold(*args, **kwargs):
        result = hold(*args, **kwargs)
        controller.update_source(state(count=controller.last_count + 1))
        return result
    arm.set_servo_angle = release_during_hold
    button_frames(clock, controller, True, frames=4)
    assert 'servo' not in arm.calls
    assert arm.state == 4
    assert controller.mode == 'motion'
    assert not controller.servo_started
    assert controller.last_stop_event['requires_enable'] is False
    assert '接管已取消' in controller.last_stop_event['reason']
    arm.set_servo_angle = hold
    button_frames(clock, controller, False, frames=15)
    button_frames(clock, controller, True)
    assert controller.servo_started


def test_hold_initialization_heartbeat_loss_still_disarms():
    clock, arm, controller = trigger_setup()
    hold = arm.set_servo_angle
    def lost_heartbeat(*args, **kwargs):
        result = hold(*args, **kwargs)
        controller.heartbeat = clock() - 1
        return result
    arm.set_servo_angle = lost_heartbeat
    button_frames(clock, controller, True, frames=4)
    assert controller.mode == 'preview' and arm.state == 4
    assert not controller.servo_started
    assert controller.last_stop_event['requires_enable'] is True


def test_hold_initialization_sdk_failure_is_not_treated_as_trigger_release():
    _, arm, controller = trigger_setup()
    arm.set_servo_angle = lambda **_: 9
    with pytest.raises(RuntimeError, match='SDK 9'):
        button_frames(controller.clock, controller, True)


def test_a_and_release_in_same_frame_never_start_gripper_action():
    clock, arm, controller = trigger_setup()
    grip = FakeGripperWorker()
    controller.gripper = grip
    controller.settings['gripper_enabled'] = True
    button_frames(clock, controller, True)
    button_frames(clock, controller, True, a_pressed=True, frames=5)
    button_frames(clock, controller, False, a_pressed=True, frames=1)
    assert grip.jobs == []
    assert not controller.servo_started


def test_release_repress_between_snapshots_stops_old_goal_and_reanchors():
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True)
    button_frames(clock, controller, True, position=(.2, 1, .32))
    count = controller.command_count
    before = arm.tcp.copy()
    button_frames(clock, controller, True, frames=1, position=(.35, 1, .4), trigger_release_generation=1)
    assert not controller.servo_started and arm.state == 4
    assert controller.command_count == count
    button_frames(clock, controller, True, position=(.35, 1, .4), trigger_release_generation=1)
    assert controller.servo_started
    assert arm.tcp == pytest.approx(before)
    assert controller.mapper.hand_anchor == pytest.approx([.35, 1, .4])


def test_release_generation_after_sdk_queries_prevents_next_command():
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True)
    count = controller.command_count
    def short_release_repress():
        payload = state(count=controller.last_count + 1)
        payload.update(trigger_pressed=True, trigger_release_generation=1)
        controller.update_source(payload)
    controller.poll_source = short_release_repress
    button_frames(clock, controller, True, frames=1, position=(.2, 1, .32))
    assert not controller.servo_started and arm.state == 4
    assert controller.command_count == count


def test_pause_bounded_motor_settling_does_not_disable_and_resume_reanchors():
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True)
    button_frames(clock, controller, True, position=(.2,1,.32))
    button_frames(clock, controller, False, frames=1, position=(.2,1,.32))
    assert controller.mode=='motion' and controller.pause_settle is not None
    # Recorded real stops settled 0.7-1.5 degrees after the stop request.
    arm.joints[4] += np.deg2rad(1.2)
    arm.joints[0] += .002
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    button_frames(clock,controller,False,frames=10,position=(.7,.8,.6))
    assert controller.mode=='motion'
    assert not controller.servo_started
    assert controller.pause_settle is None
    assert not controller.last_error
    parked = arm.tcp.copy()
    button_frames(clock,controller,True,frames=10,position=(.7,.8,.6))
    assert controller.servo_started
    assert arm.tcp == pytest.approx(parked)
    assert controller.mapper.hand_anchor == pytest.approx([.7,.8,.6])
    assert controller.hold_count==2


def test_pause_keeps_session_and_source_live_through_repeated_clutches():
    clock, arm, controller = trigger_setup()
    for iteration in range(5):
        position = (.2+iteration*.15,1,.3)
        button_frames(clock,controller,False,frames=10,position=position)
        assert controller.mode=='motion'
        assert controller.telemetry()['teleop_armed']
        assert controller.telemetry()['teleop_paused']
        button_frames(clock,controller,True,frames=10,position=position)
        assert controller.servo_started
        assert controller.fresh()
    assert controller.hold_count==5


def fast_pause():
    clock, arm, controller = trigger_setup()
    controller.settings.update(speed_mm_s=150., tracking_lead_mm=12., joint_speed_rad_s=1.)
    controller.mapper.speed = 150.
    button_frames(clock, controller, True)
    controller.actual_speed = 150.
    controller.command_tcp[0] += 12.
    controller.release('扳机松开', keep_gripper=True)
    return clock, arm, controller


def test_fast_pause_allows_bounded_braking_and_reanchors_without_old_target():
    clock, arm, controller = fast_pause()
    assert controller.pause_settle['distance_limit_mm'] == pytest.approx(29.)
    commands = controller.command_count
    for mm in (5, 10.1, 15, 17, 17, 17):
        arm.joints[0] = mm/1000
        arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
        button_frames(clock, controller, False, frames=1)
    assert controller.mode == 'motion' and controller.pause_settle is None
    assert not controller.last_error and controller.command_count == commands
    assert controller.pause_feedback['status'] == 'settled'
    assert controller.pause_feedback['displacement_mm'] == pytest.approx(17.)
    parked = arm.tcp.copy()
    button_frames(clock, controller, True, position=(.6, .8, .4))
    assert controller.servo_started and arm.tcp == pytest.approx(parked)
    assert controller.mapper.hand_anchor == pytest.approx([.6, .8, .4])


def test_pause_envelope_is_frozen_not_expanded_by_post_stop_motion():
    clock, arm, controller = fast_pause()
    controller.actual_speed = 9999.
    controller.command_tcp[0] += 999.
    arm.joints[0] = .031
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    stops = arm.calls.count(('state', 4))
    button_frames(clock, controller, False, frames=1)
    assert controller.mode == 'preview'
    assert '31.0/29.0 mm' in controller.last_error
    assert controller.pause_feedback['status'] == 'rejected'
    assert arm.calls.count(('state', 4)) == stops+1  # reassert stop on settle fault


def test_stationary_pause_does_not_inherit_fast_configuration_allowance():
    clock, arm, controller = trigger_setup()
    controller.settings.update(speed_mm_s=200., tracking_lead_mm=20.)
    controller.mapper.speed = 200.
    button_frames(clock, controller, True)
    controller.release('暂停')
    assert controller.pause_settle['distance_limit_mm'] == 10.
    arm.joints[0] = .011
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    button_frames(clock, controller, False, frames=1)
    assert controller.mode == 'preview'


@pytest.mark.parametrize('failure', ['angle', 'joint_coast', 'joint_limit', 'post_settle'])
def test_fast_pause_retains_non_translation_and_post_settle_protections(failure):
    clock, arm, controller = fast_pause()
    if failure == 'angle':
        arm.joints[3] += np.deg2rad(5)
    elif failure == 'joint_coast':
        arm.joints[6] += .3  # redundant joint, does not change fake TCP
    elif failure == 'joint_limit':
        controller.upper[6] = .005
        arm.joints[6] = .01
    else:
        button_frames(clock, controller, False, frames=10)
        assert controller.pause_settle is None
        arm.joints[0] += .007  # settled external-movement limit remains 6 mm
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    button_frames(clock, controller, False, frames=1)
    assert controller.mode == 'preview' and not controller.servo_started
    assert arm.state == 4


def test_pause_large_or_delayed_external_robot_movement_still_disables():
    for settling in (True,False):
        clock, arm, controller = trigger_setup()
        button_frames(clock,controller,True)
        button_frames(clock,controller,False,frames=1 if settling else 10)
        arm.joints[0] += .05  # 50 mm, NOT motor settling
        arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
        button_frames(clock,controller,False,frames=1)
        assert controller.mode=='preview'
        assert not controller.telemetry()['teleop_armed']


def test_stop_button_withdraws_permission_unlike_trigger_pause():
    clock, arm, controller = trigger_setup()
    button_frames(clock,controller,True)
    button_frames(clock,controller,False,frames=10)
    assert controller.mode=='motion'
    controller.command('stop')
    button_frames(clock,controller,True,frames=10)
    assert controller.mode=='preview' and not controller.servo_started


def test_pause_unconfirmed_stop_and_one_second_settle_timeout_disable():
    clock, arm, controller = trigger_setup()
    button_frames(clock,controller,True)
    arm.set_state = lambda *_: 9
    button_frames(clock,controller,False,frames=1)
    assert controller.mode=='preview'
    assert '停机请求未确认' in controller.last_error
    clock, arm, controller = trigger_setup()
    button_frames(clock,controller,True)
    button_frames(clock,controller,False,frames=1)
    arm.state=0  # accepted request but never actually entered stopped state
    for _ in range(120):
        button_frames(clock,controller,False,frames=1)
    assert controller.mode=='preview'
    assert '未确认停稳' in controller.last_error


def test_rotation_follower_with_feedback_lag_and_wrist_gain_keeps_running():
    class LagArm(FakeArm):
        def __init__(self):
            super().__init__()
            self.pending = []
            self.references = []

        def get_inverse_kinematics(self, pose, **kwargs):
            code, result = super().get_inverse_kinematics(pose, **kwargs)
            result[4] *= 1.35
            if 'ref_angles' in kwargs:
                self.references.append(np.array(kwargs['ref_angles']))
            return code, result

        def get_forward_kinematics(self, joints, **kwargs):
            joints = np.array(joints).copy()
            joints[4] /= 1.35
            return super().get_forward_kinematics(joints, **kwargs)

        def set_servo_angle_j(self, joints, **kwargs):
            self.calls.append('servo')
            self.pending.append(np.array(joints).copy())
            return 0

        def feedback_tick(self):
            if len(self.pending) > 12:
                self.joints = self.pending.pop(0)
                self.tcp = np.array(self.get_forward_kinematics(self.joints)[1])

    clock, arm = Clock(), LagArm()
    controller = RelativeArm(arm, [-3]*7, [3]*7, clock, control_settings={'teleop_button': 'b'})
    engage(clock, controller)
    reached_old_guard = False
    for count in range(4, 304):
        payload = state(invalid=1, pause=1, count=count)
        # Default reflected map sends Quest X rotation to robot +Y.
        payload['quaternion_wxyz'] = quaternion(Rotation.from_euler('x', min(.004*(count-4), .6)))
        tick_100(clock, controller, payload)
        assert controller.servo_started
        # The previous actual-vs-goal test would reject valid delayed tracking.
        _, requested_q = arm.get_inverse_kinematics(pose_to_sdk(controller.command_tcp))
        reached_old_guard |= np.max(np.abs(np.array(requested_q) - arm.joints)) > .035
        if controller.last_command_joints is not None:
            assert np.max(np.abs(controller.last_command_joints - arm.references[-1])) <= .005 + 1e-9
        arm.feedback_tick()
    assert reached_old_guard
    assert controller.command_count > 150
    assert arm.tcp[4] > .5


def test_physical_joint_branch_can_cross_180_without_sdk_artificial_limit():
    class WrappedArm(FakeArm):
        def get_inverse_kinematics(self, pose, **kwargs):
            assert kwargs.get('limited') is False
            code, result = super().get_inverse_kinematics(pose, **kwargs)
            result = np.arctan2(np.sin(result), np.cos(result))
            return code, result.tolist()
    clock, arm = Clock(), WrappedArm()
    arm.joints[3] = np.deg2rad(179.5)
    arm.tcp = np.array(arm.get_forward_kinematics(arm.joints)[1])
    lower, upper = np.full(7, -2*np.pi), np.full(7, 2*np.pi)
    lower[3], upper[3] = np.deg2rad(-6), np.deg2rad(220)
    controller = RelativeArm(arm, lower, upper, clock)
    controller.validate_robot()
    target = arm.tcp.copy(); target[3] += np.deg2rad(1.2)
    q, depth = controller.solve_follow_target(target, arm.tcp, arm.joints, clock())
    assert depth == 0
    assert np.rad2deg(q[3]) == pytest.approx(180.7)
    assert not controller.servo_started and 'servo' not in arm.calls
    with pytest.raises(RuntimeError, match='限位'):
        continuous_solution(q, arm.joints, lower, np.array([*upper[:3], np.deg2rad(180), *upper[4:]]))


def test_hold_and_home_preflights_use_physical_ik_limits(tmp_path):
    clock, arm, controller = trigger_setup()
    controller.command('preview')
    controller.recovery.home_path = tmp_path / 'home.json'
    arm.get_joint_states = lambda **kwargs: (0, [arm.joints.tolist(), [0]*7])
    original = arm.get_inverse_kinematics
    queries = []
    def unrestricted(pose, **kwargs):
        queries.append(kwargs)
        assert kwargs['limited'] is False
        return original(pose, **kwargs)
    arm.get_inverse_kinematics = unrestricted
    controller.validate_robot()
    controller.command('home_save', {'confirmed':True})
    assert queries
    assert 'servo' not in arm.calls and 'enable' not in arm.calls


def test_slow_invalid_ik_skips_to_feasible_step_and_reuses_successful_scale():
    clock, arm, controller = trigger_setup()
    original = arm.get_inverse_kinematics
    calls = []
    def boundary(pose, **kwargs):
        delta = pose[0]-arm.tcp[0]; calls.append(delta)
        if delta > .04:
            clock.now += .009
            return 10, []
        clock.now += .0003
        return original(pose, **kwargs)
    arm.get_inverse_kinematics = boundary
    target = arm.tcp.copy(); target[0] += .5
    q, depth = controller.solve_follow_target(target, arm.tcp, arm.joints, clock())
    assert depth == 4 and len(calls) == 3
    assert controller.ik_feedback['elapsed_ms'] == pytest.approx(18.3)
    assert controller.ik_feedback['status'] == 'ok'
    calls.clear()
    _, depth = controller.solve_follow_target(target, arm.tcp, arm.joints, clock())
    assert depth == 4 and len(calls) == 2
    assert calls[0] == pytest.approx(.0625)  # not a repeat of .5, .25, .125
    assert controller.ik_feedback['elapsed_ms'] == pytest.approx(9.3)
    controller.release('reset')
    assert controller.ik_subdivision == 0
    assert controller.ik_feedback['status'] == 'ok'


def test_ik_budget_does_not_start_query_that_cannot_fit_and_preserves_failure():
    clock, arm, controller = trigger_setup()
    def too_expensive(*args, **kwargs):
        clock.now += .015
        return 10, []
    arm.get_inverse_kinematics = too_expensive
    target = arm.tcp.copy(); target[2] += .5
    with pytest.raises(RuntimeError, match='SDK 10'):
        controller.solve_follow_target(target, arm.tcp, arm.joints, clock())
    feedback = controller.ik_feedback
    assert feedback['status'] == 'budget_exhausted'
    assert len(feedback['attempts']) == 2
    assert feedback['cycle_elapsed_ms'] == pytest.approx(30)
    assert feedback['target_mm'] == pytest.approx(target[:3])
    controller.release('stop', disarm=True)
    assert controller.ik_feedback == feedback
    assert 'servo' not in arm.calls


def test_slow_successful_ik_is_not_executed_after_cycle_budget():
    clock, arm, controller = trigger_setup()
    original = arm.get_inverse_kinematics
    def slow(*args, **kwargs):
        clock.now += .045
        return original(*args, **kwargs)
    arm.get_inverse_kinematics = slow
    with pytest.raises(RuntimeError, match='超时'):
        controller.solve_follow_target(arm.tcp, arm.tcp, arm.joints, clock())
    assert controller.ik_feedback['status'] == 'deadline'
    assert len(controller.ik_feedback['attempts']) == 1
    assert 'servo' not in arm.calls


def test_protocol_errors_are_not_retried_as_unreachable_targets():
    clock, arm, controller = trigger_setup()
    arm.get_inverse_kinematics = lambda *a, **kw: (7, [])
    with pytest.raises(RuntimeError, match='SDK 7'):
        controller.solve_follow_target(arm.tcp, arm.tcp, arm.joints, clock())
    assert controller.ik_feedback['status'] == 'sdk_error'
    assert len(controller.ik_feedback['attempts']) == 1


def test_committed_velocity_uses_short_arc_and_respects_caps():
    previous = np.array([400., 0., 300., 0, 0, np.deg2rad(179.9)])
    committed = previous.copy(); committed[0] += .2; committed[5] = np.deg2rad(-179.9)
    linear, angular = command_velocities(previous, committed, .01, 50, np.deg2rad(30))
    assert linear == pytest.approx([20, 0, 0])
    assert np.rad2deg(angular) == pytest.approx([0, 0, 20])
    linear, angular = command_velocities(previous, committed, .001, 50, np.deg2rad(30))
    assert np.linalg.norm(linear) == pytest.approx(50)
    assert np.rad2deg(np.linalg.norm(angular)) == pytest.approx(30)
    assert command_velocities(previous, committed, 0, 50, 1)[0] == pytest.approx(np.zeros(3))


def test_joint_limited_command_back_calculates_velocity_instead_of_winding_up():
    class AmplifiedArm(FakeArm):
        def get_inverse_kinematics(self, pose, **kwargs):
            code, q = super().get_inverse_kinematics(pose, **kwargs)
            q[0] *= 100
            return code, q
        def get_forward_kinematics(self, joints, **kwargs):
            q = np.array(joints).copy(); q[0] /= 100
            return super().get_forward_kinematics(q, **kwargs)
    clock, arm = Clock(), AmplifiedArm()
    controller = RelativeArm(arm, [-3]*7, [3]*7, clock,
                             control_settings={'joint_speed_rad_s':.05})
    controller.update_source(state()); controller.command('allow_controller')
    button_frames(clock, controller, True)
    before = controller.command_tcp.copy()
    button_frames(clock, controller, True, frames=1, position=(.2,1,.31))
    expected = (controller.command_tcp[:3]-before[:3])/CONTROL_PERIOD
    assert controller.cartesian_velocity == pytest.approx(expected)
    # A tangential direction at sigma=10 uses the conservative approach band,
    # while still back-calculating actual commanded velocity after the cap.
    assert controller.cartesian_velocity[0] == pytest.approx(.5*(10-8.5)/(20-8.5))
    assert '关节限速' in controller.limit_reason


def test_telemetry_command_is_valid_only_during_active_following():
    clock, arm, controller = trigger_setup()
    assert controller.telemetry()['command_valid'] is False
    button_frames(clock, controller, True)
    button_frames(clock, controller, True, position=(.2,1,.32))
    data = controller.telemetry()
    assert data['command_valid'] is True
    assert data['command_rpy_deg'] == pytest.approx(pose_rpy_deg(controller.command_tcp))
    assert data['command_quaternion_wxyz'] == pytest.approx(controller.command_tcp[3:])
    assert data['command_tracking_error_mm'] >= 0
    button_frames(clock, controller, False, frames=1)
    data = controller.telemetry()
    assert data['command_valid'] is False
    assert data['command_tracking_error_mm'] is None
    assert data['command_rotation_error_deg'] is None


def test_first_small_rotation_acceleration_step_is_not_discarded():
    clock, arm, controller = trigger_setup()
    button_frames(clock, controller, True)
    assert controller.command_count == 0
    button_frames(clock, controller, True, frames=1,
                  quaternion_wxyz=quaternion(Rotation.from_euler('x', .05)))
    assert controller.command_count == 1
    assert pose_rpy_deg(controller.command_tcp)[1] == pytest.approx(.009)
    assert np.rad2deg(controller.angular_velocity[1]) == pytest.approx(.9)


def test_low_translation_speed_does_not_stall_below_old_skip_threshold():
    clock, arm, controller = trigger_setup()
    controller.command('configure', {'speed_mm_s':1, 'acceleration_mm_s2':50})
    button_frames(clock, controller, True)
    before = controller.command_tcp.copy()
    button_frames(clock, controller, True, frames=1, position=(.2,1,.32))
    assert controller.command_count == 1
    assert controller.command_tcp[0]-before[0] == pytest.approx(.005)
    assert controller.cartesian_velocity[0] == pytest.approx(.5)


def test_previous_slow_query_cannot_permanently_block_a_new_takeover():
    clock, arm, controller = trigger_setup()
    original = arm.get_inverse_kinematics
    def slow(*args, **kwargs):
        clock.now += .05
        return original(*args, **kwargs)
    arm.get_inverse_kinematics = slow
    with pytest.raises(RuntimeError, match='超时'):
        controller.solve_follow_target(arm.tcp, arm.tcp, arm.joints, clock())
    controller.release('stop', disarm=True)
    assert controller.ik_feedback['status'] == 'deadline'
    assert not controller.ik_query_times
    arm.get_inverse_kinematics = original
    _, depth = controller.solve_follow_target(arm.tcp, arm.tcp, arm.joints, clock())
    assert depth == 0 and controller.ik_feedback['status'] == 'ok'


def test_b_click_toggles_and_release_holds_while_trigger_is_paused():
    clock,arm,controller=trigger_setup()
    controller.command('configure',{'gripper_enabled':True,'gripper_button':'b'})
    grip=FakeGripperWorker();controller.gripper=grip
    button_frames(clock,controller,False,b_pressed=False)
    button_frames(clock,controller,False,b_pressed=True)
    assert [job[0] for job in grip.jobs]==[0]
    assert not controller.servo_started and controller.mode=='motion'
    button_frames(clock,controller,False,frames=10,b_pressed=False)
    assert [job[0] for job in grip.jobs]==[0]
    button_frames(clock,controller,False,b_pressed=True)
    assert [job[0] for job in grip.jobs]==[0,84]
    assert controller.command_count==0 and 'enable' not in arm.calls


def test_trigger_clutch_keeps_b_toggle_and_next_click_opens():
    clock,arm,controller=trigger_setup()
    controller.settings.update(gripper_enabled=True,gripper_button='b')
    controller.gripper=grip=FakeGripperWorker()
    button_frames(clock,controller,True,b_pressed=False)
    button_frames(clock,controller,True,b_pressed=True)
    button_frames(clock,controller,False,frames=1,b_pressed=True)
    assert not controller.servo_started
    assert [job[0] for job in grip.jobs]==[0]
    assert grip.jobs[-1][3]()  # trigger is no longer required for gripper
    button_frames(clock,controller,False,frames=10,b_pressed=False)
    assert [job[0] for job in grip.jobs]==[0]
    button_frames(clock,controller,False,b_pressed=True)
    assert [job[0] for job in grip.jobs]==[0,84]


def test_missing_b_and_data_loss_or_stop_are_not_a_button_release():
    for failure in ('missing','tracking','stop'):
        clock,arm,controller=trigger_setup()
        controller.settings.update(gripper_enabled=True,gripper_button='b')
        controller.gripper=grip=FakeGripperWorker()
        button_frames(clock,controller,False,b_pressed=False)
        button_frames(clock,controller,False,b_pressed=True)
        if failure=='stop':controller.command('stop')
        elif failure=='missing':button_frames(clock,controller,False,b_pressed=None)
        else:button_frames(clock,controller,False,b_pressed=False,valid=False,invalid=1)
        assert [job[0] for job in grip.jobs]==[0]
        assert not grip.jobs[-1][3]()


def test_b_release_generation_preserves_short_release_repress():
    clock,arm,controller=trigger_setup()
    controller.settings.update(gripper_enabled=True,gripper_button='b')
    controller.gripper=grip=FakeGripperWorker()
    button_frames(clock,controller,False,b_pressed=False)
    button_frames(clock,controller,False,b_pressed=True)
    button_frames(clock,controller,False,frames=1,b_pressed=True,b_release_generation=1)
    assert [job[0] for job in grip.jobs]==[0]
    assert grip.jobs[0][3]()  # release alone cannot supersede the close
    button_frames(clock,controller,False,b_pressed=True,b_release_generation=1)
    assert [job[0] for job in grip.jobs]==[0,84]
    assert not grip.jobs[0][3]()  # only the new accepted press supersedes it


def test_b_gripper_cannot_share_b_teleop_and_held_initial_b_never_closes():
    with pytest.raises(ValueError,match='同时'):
        settings({'teleop_button':'b','gripper_button':'b'})
    clock,arm,controller=trigger_setup()
    controller.settings.update(gripper_enabled=True,gripper_button='b')
    controller.gripper=grip=FakeGripperWorker()
    button_frames(clock,controller,False,b_pressed=True)
    button_frames(clock,controller,False,b_pressed=False)
    assert grip.jobs==[]
    button_frames(clock,controller,False,b_pressed=True)
    assert [job[0] for job in grip.jobs]==[0]
