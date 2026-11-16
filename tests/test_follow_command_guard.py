import numpy as np
import pytest

from quest_pose_guard import joint_acceleration_step
from xarm_quest_relative import RelativeArm, wait_initial_reports
from test_relative_motion import Clock, FakeArm, engage, tick_100, state


class CoupledArm(FakeArm):
    """Two joints cancel in X: independent smoothing distorts the TCP path."""
    def get_forward_kinematics(self, joints, **kwargs):
        q = np.asarray(joints)
        pose = self.home.copy()
        pose[:3] += [1000*(q[0]-q[1]), 1000*q[1], 1000*q[2]]
        pose[3:] += q[3:6]
        return 0, pose.tolist()

    def get_inverse_kinematics(self, pose, **kwargs):
        delta = np.asarray(pose)-self.home
        return 0, [float((delta[0]+delta[1])/1000), float(delta[1]/1000),
                   float(delta[2]/1000), *delta[3:].tolist(), 0.]


def setup_guard():
    clock, arm = Clock(), CoupledArm()
    c = RelativeArm(arm, [-3]*7, [3]*7, clock, control_settings={
        'speed_mm_s':150, 'tracking_lead_mm':12, 'rotation_enabled':True,
        'joint_speed_rad_s':1, 'teleop_button':'b'})
    c.mapper.engage([.2,1,.3], c.tcp, [1,0,0,0])
    return clock, arm, c


def test_coupled_joint_smoothing_is_rechecked_after_tether_and_shortened():
    clock, arm, c = setup_guard()
    previous = np.array([.011,0,0,0,0,0,0])
    previous_tcp = np.array(arm.get_forward_kinematics(previous)[1])
    target = arm.home.copy()
    target[:2] += [np.sqrt(12**2-.5**2), .5]
    solution = np.array(arm.get_inverse_kinematics(target)[1])
    velocity = (solution-previous)/.01
    velocity[1] = 0
    # Reproduce the old independent joint clamp for the final-boundary guard.
    candidate = previous + np.minimum((solution-previous)/.01, velocity+.02)*.01
    raw_fk = np.array(arm.get_forward_kinematics(candidate)[1])
    assert np.linalg.norm(target[:3]-c.tcp[:3]) == pytest.approx(12)
    assert np.linalg.norm(raw_fk[:3]-c.tcp[:3]) > 12.1
    fitted = c.fit_follow_command(candidate, previous, previous_tcp, .01, 12, np.pi/3, clock()+.04)
    command, fk, fraction = fitted
    assert fraction == .5
    assert np.linalg.norm(fk[:3]-c.tcp[:3]) <= 12.1
    assert command == pytest.approx(previous+(candidate-previous)*fraction)
    assert c.command_guard['status'] == 'shortened'
    assert arm.calls == []  # This guard only asks FK; no enabling or motion.
    # A coordinated velocity transition can leave the new IK segment while
    # braking/turning. Its FINAL FK must therefore be fitted before sending.
    coordinated = joint_acceleration_step(previous, solution, velocity, .01, 1)
    assert np.max(np.abs((coordinated-previous)/.01-velocity)) <= .02000001
    final,fk,fraction=c.fit_follow_command(coordinated,previous,previous_tcp,.01,12,np.pi/3,clock()+.04)
    assert fraction < 1
    assert np.linalg.norm(fk[:3]-c.tcp[:3]) <= 12.1
    assert np.linalg.norm(fk[:3]-previous_tcp[:3]) <= 1.54
    assert final == pytest.approx(previous+(coordinated-previous)*fraction)
    assert arm.calls == []


@pytest.mark.parametrize('axis', ['translation', 'rotation'])
def test_no_feasible_segment_does_not_return_unsafe_hold(axis):
    clock, arm, c = setup_guard()
    previous = np.zeros(7)
    previous[0 if axis=='translation' else 3] = .02 if axis=='translation' else .1
    previous_tcp = np.array(arm.get_forward_kinematics(previous)[1])
    assert c.fit_follow_command(previous, previous, previous_tcp, .01, 12, np.pi/3, clock()+.04) is None
    assert c.command_guard['status'] == 'blocked'
    assert len(c.command_guard['attempts']) == 8
    assert arm.calls == []


def test_final_guard_rejects_late_failed_and_nonfinite_fk():
    clock, arm, c = setup_guard()
    q = np.zeros(7)
    def run():
        return c.fit_follow_command(q, q, c.tcp, .01, 12, np.pi/3, clock()+.04)
    arm.get_forward_kinematics = lambda *a, **kw: (7, [])
    with pytest.raises(RuntimeError, match='正解'): run()
    arm.get_forward_kinematics = lambda *a, **kw: (0, [float('nan')]*6)
    with pytest.raises(ValueError): run()
    def late(*a, **kw):
        clock.now += .045
        return 0, c.tcp.tolist()
    arm.get_forward_kinematics = late
    with pytest.raises(RuntimeError, match='超时'): run()
    assert arm.calls == []


def test_coupled_lag_following_retains_arm_permission_and_bounded_commands():
    class LagArm(CoupledArm):
        def set_servo_angle_j(self, joints, **kwargs):
            self.calls.append('servo')
            self.joints += (np.asarray(joints)-self.joints)*.08
            self.tcp = np.array(self.get_forward_kinematics(self.joints)[1])
            return 0
    clock, arm = Clock(), LagArm()
    c = RelativeArm(arm, [-3]*7, [3]*7, clock, control_settings={
        'speed_mm_s':150, 'translation_scale':2.75, 'acceleration_mm_s2':600,
        'tracking_lead_mm':12, 'rotation_enabled':True, 'joint_speed_rad_s':1,
        'teleop_button':'b'})
    engage(clock,c)
    for i in range(400):
        p = (.2+.015*np.sin(i*.04), 1+.008*np.sin(i*.07), .3+.0002*i)
        s = state(position=p, invalid=1, pause=1, count=i+4)
        tick_100(clock,c,s)
        assert c.servo_started and c.mode=='motion'
        assert np.linalg.norm(c.command_tcp[:3]-c.tcp[:3]) <= 12.1
    assert c.command_count > 300
    assert arm.tcp[0]-arm.home[0] > 100


@pytest.mark.parametrize('reports,passed', [(0,False),(1,False),(2,True)])
def test_startup_waits_for_completed_report_and_unsubscribes(reports, passed):
    class ReportingArm:
        released = False
        def register_report_callback(self, callback):
            self.callback = callback
            for _ in range(reports): callback({})
            return True
        def release_report_callback(self, callback):
            assert callback is self.callback
            self.released = True
    arm = ReportingArm()
    if passed:
        wait_initial_reports(arm, timeout=0)
    else:
        with pytest.raises(RuntimeError, match='上报超时'):
            wait_initial_reports(arm, timeout=0)
    assert arm.released

