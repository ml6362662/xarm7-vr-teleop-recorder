import pytest

from quest_gripper_control import GripperWorker


class FakeG2:
    def __init__(self):
        self.position = 84
        self.calls = []
        self.code = 0

    def get_gripper_g2_position(self, **kwargs):
        assert kwargs == {"check_baud": False}
        self.calls.append("read")
        return 0, self.position

    def set_gripper_g2_position(self, target, **kwargs):
        assert kwargs["wait"] is False and kwargs["wait_motion"] is False
        assert kwargs["check_baud"] is False
        assert type(kwargs["force"]) is int
        self.calls.append(("move", target, kwargs["speed"], kwargs["force"]))
        if self.code == 0:
            self.position = target
        return self.code


def setup_gripper(clock=lambda: 10):
    arm = FakeG2()
    worker = GripperWorker(lambda: arm, clock=clock, autostart=False)
    worker.arm = arm
    worker.state["connected"] = True
    worker.cycle()
    return arm, worker


def test_initial_gripper_poll_is_read_only():
    arm, worker = setup_gripper()
    assert arm.calls == ["read"]
    assert worker.snapshot()["position_mm"] == 84
    assert worker.snapshot()["commands"] == 0


def test_explicit_g2_request_uses_mm_force_and_nonblocking_mode():
    arm, worker = setup_gripper()
    worker.submit(0, 50, 30, lambda: True)
    worker.cycle()
    assert arm.calls[-1] == ("move", 0, 50, 30)
    assert worker.snapshot()["commands"] == 1


def test_cancel_pending_never_opens_or_disables_gripper():
    arm, worker = setup_gripper()
    worker.submit(0, 50, 30, lambda: True)
    worker.cancel_pending()
    worker.cycle()
    assert arm.calls == ["read"]


@pytest.mark.parametrize("target,speed,force", [(85, 50, 30), (0, 14, 30), (0, 50, 101), (float('nan'), 50, 30)])
def test_gripper_rejects_invalid_parameters(target, speed, force):
    arm, worker = setup_gripper()
    with pytest.raises(ValueError):
        worker.submit(target, speed, force, lambda: True)
    assert arm.calls == ["read"]


def test_permission_is_rechecked_before_gripper_sdk_call():
    arm, worker = setup_gripper()
    worker.submit(0, 50, 30, lambda: False)
    worker.cycle()
    assert arm.calls == ["read"]


def test_expired_gripper_command_does_not_move():
    now = [10.]
    arm, worker = setup_gripper(lambda: now[0])
    worker.submit(0, 50, 30, lambda: True)
    now[0] += .6
    worker.cycle()
    assert all(call == "read" for call in arm.calls)


def test_gripper_fault_is_reported_without_clear_or_arm_calls():
    arm, worker = setup_gripper()
    arm.code = 19
    worker.submit(0, 50, 30, lambda: True)
    worker.cycle()
    assert worker.snapshot()["command_fault"]
    assert "SDK 19" in worker.snapshot()["error"]
    assert worker.snapshot()["commands"] == 0


def test_latest_pending_gripper_target_replaces_older_unsent_target():
    arm, worker = setup_gripper()
    worker.submit(0, 50, 30, lambda: True)
    worker.submit(84, 50, 30, lambda: True)
    worker.cycle()
    assert arm.calls[-1] == ("move", 84, 50, 30)


def test_worker_close_does_not_auto_open():
    arm, worker = setup_gripper()
    worker.submit(0, 50, 30, lambda: True)
    worker.close()
    worker.cycle()
    assert arm.calls == ["read"]


def test_gui_float_parameters_are_converted_to_integer_force():
    arm, worker = setup_gripper()
    worker.submit(0.0, 50.0, 30.0, lambda: True)
    worker.cycle()
    assert worker.snapshot()['commands'] == 1
    assert not worker.snapshot()['command_fault']
    assert arm.calls[-1] == ('move', 0.0, 50.0, 30)


def test_real_installed_sdk_g2_serializer_offline_accepts_gui_values():
    # Execute the actual SDK serializer against an intercepted transport,
    # not a robot connection. The former fake accepted the invalid float.
    from xarm.x3.gripper import Gripper
    import struct

    class SDKSerializer(FakeG2):
        connected = True
        _gripper_error_code = 0
        def __init__(self):
            super().__init__()
            self.frames = []
        def check_is_simulation_robot(self): return False
        def log_api_info(self, *a, **kw): pass
        def _get_modbus_gripper_err_code(self): return 0, 0
        def set_rs485_data(self, frame, **kw):
            self.frames.append(bytes(frame))
            return 0, b'\x08\x10\x0c\x00\x00\x05'
        def set_gripper_g2_position(self, target, **kw):
            return Gripper.set_gripper_g2_position(self, target, **kw)

    arm = SDKSerializer()
    worker = GripperWorker(lambda: arm, autostart=False)
    worker.arm = arm
    worker.state['connected'] = True
    worker.submit(0.0, 50.0, 30.0, lambda: True)
    worker.cycle()
    assert worker.snapshot()['commands'] == 1
    assert struct.unpack('>h', arm.frames[0][11:13])[0] == 30
    assert len(arm.frames[0]) == 17
