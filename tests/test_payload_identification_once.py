import json
import threading
from tasks import identify_payload_once as task
import pytest


class FakeArm:
    axis = 7
    version_number = (2, 8, 0)
    connected = True
    sn = 'test'
    version = '2.8.0'
    gravity_direction = [0, 0, -1]
    tcp_load = [.82, [0, 0, 48]]
    tcp_offset = [0, 0, 172, 0, 0, 0]
    world_offset = [0]*6
    collision_sensitivity = 4

    def __init__(self, calls, errors, reply):
        self.calls, self.errors, self.reply = calls, errors, reply
        self.state = 4

    def get_state(self): return 0, self.state
    def get_err_warn_code(self): return 0, self.errors
    def get_joint_states(self, **kw): return 0, [[0]*7, [0]*7]
    def set_state(self, state):
        self.calls.append(('state', state)); self.state = state; return 0
    def set_mode(self, mode): self.calls.append(('mode', mode)); return 0
    def motion_enable(self, **kw): self.calls.append('enable'); return 0
    def register_iden_progress_changed_callback(self, callback): self.callback = callback
    def set_tcp_load(self, mass, center, **kw):
        self.calls.append('restore_load'); self.tcp_load = [mass, center]; return 0
    def set_collision_sensitivity(self, sensitivity, **kw):
        self.calls.append('restore_collision'); self.collision_sensitivity = sensitivity
        self.state = 0  # mirror SDK setter's ready-state side effect
        return 0
    def iden_tcp_load(self): self.calls.append('identify'); return self.reply
    def disconnect(self): self.calls.append('disconnect')


@pytest.mark.parametrize('errors,reply,expected', [
    ([0, 0], (0, [1.1, 0, 0, 60]), 0),
    ([31, 0], (0, [1.1, 0, 0, 60]), 1),
    ([0, 0], (10, []), 1),
    ([0, 0], (0, [float('nan'), 0, 0, 60]), 1),
])
def test_identification_never_applies_result_and_stops_on_exit(tmp_path, monkeypatch, errors, reply, expected):
    (tmp_path/'.state').mkdir()
    (tmp_path/'config.json').write_text('{"robot_ip":"fake"}')
    monkeypatch.setattr(task, 'ROOT', tmp_path)
    monkeypatch.setattr('sys.argv', ['identify', '--operator-confirmed-motion'])
    monkeypatch.setattr(task.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(task.time, 'sleep', lambda *a: None)
    calls = []
    monkeypatch.setattr(task, 'XArmAPI', lambda *a, **kw: FakeArm(calls, errors, reply))
    assert task.main() == expected
    data = json.loads(next((tmp_path/'.state').glob('payload_identification_*.json')).read_text())
    assert data['applied'] is False
    if errors != [0, 0]:
        assert calls == ['disconnect']
    else:
        assert calls[-2:] == [('state', 4), 'disconnect']
        assert data['before']['tcp_load'] == [.82, [0, 0, 48]]
        assert calls.count('identify') == 1


def test_completed_motion_waits_for_result_and_restores_firmware_zeroed_load(tmp_path, monkeypatch):
    (tmp_path/'.state').mkdir()
    (tmp_path/'config.json').write_text('{"robot_ip":"fake"}')
    monkeypatch.setattr(task, 'ROOT', tmp_path)
    monkeypatch.setattr('sys.argv', ['identify', '--operator-confirmed-motion'])
    monkeypatch.setattr(task.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(task.time, 'sleep', lambda *a: None)
    instances, calls = [], []

    class CompletingArm(FakeArm):
        def iden_tcp_load(self):
            calls.append('identify')
            instances[0].tcp_load = [0., [0., 0., 0.]]
            instances[0].collision_sensitivity = 0
            instances[0].state = 4
            self.callback({'progress': 100})
            threading.Event().wait(.35)  # result follows motion-complete state
            return 0, [1.1, 0., 0., 60.]

    def factory(*args, **kw):
        arm = CompletingArm(calls, [0, 0], None); instances.append(arm); return arm

    monkeypatch.setattr(task, 'XArmAPI', factory)
    assert task.main() == 0
    data = json.loads(next((tmp_path/'.state').glob('payload_identification_*.json')).read_text())
    assert data['status'] == 'identified_not_applied'
    assert data['result']['reply'][1][0] == 1.1
    assert data['restored_tcp_load'] == [.82, [0, 0, 48]]
    assert data['final_collision_sensitivity'] == 4
    assert data['final_state'] == [0, 4]
    assert data['applied'] is False and calls.count('identify') == 1
    assert calls[-2:] == [('state', 4), 'disconnect']
