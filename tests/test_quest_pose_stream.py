import quest_pose_stream as stream
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import zmq


def test_valid_recorded_packet_exposes_xyz_and_orientation():
    packet = (b"absolute|-0.1958634,0.5487109,0.1997189|0.9726847,0.1625274,-0.05004814,0.1580023"
              b"|False|-0.2003626,0.5154874,0.2939312|-0.1013651,0.5778831,0.2145192"
              b"|-0.2282644,0.6384059,0.229802")
    result = stream.inspect_packet(packet)
    assert result["valid"] is True
    assert result["position_m"] == [-0.1958634, 0.5487109, 0.1997189]
    assert len(result["euler_deg"]) == 3


def test_zero_tracking_is_reported_without_claiming_valid_xyz():
    result = stream.inspect_packet(b"absolute|0,0,0|1,0,0,0|False|0.1,0,0|0,0.1,0|0,0,-0.1")
    assert result["valid"] is False
    assert result["raw_position_m"] == [0, 0, 0]
    assert "position_m" not in result
    assert "零位置" in result["reason"]


def test_malformed_packet_does_not_crash_receiver():
    assert stream.inspect_packet(b"bad data")["valid"] is False


def test_extended_packet_reports_a_and_legacy_packet_reports_unknown_a():
    packet = b"absolute|1,2,3|1,0,0,0|True|1.1,2,3|1,2.1,3|1,2,2.9"
    assert stream.inspect_packet(packet)['a_pressed'] is None
    extended = stream.inspect_packet(packet + b'|True')
    assert extended['valid'] and extended['a_pressed'] is True
    assert extended['trigger_pressed'] is True
    assert stream.inspect_packet(packet + b'|False')['a_pressed'] is False
    assert stream.inspect_packet(packet + b'|bad')['valid'] is False


def test_xr_permission_uses_pose_not_delayed_legacy_pause_channel():
    state=dict(input_backend='unitree',motion_enabled=True,pause_generation=0,resume_generation=1)
    assert not stream.update_input_permission(state,legacy_pause=b'Low')
    assert state['motion_enabled'] and state['pause_generation']==0
    assert stream.update_input_permission(state,parsed=dict(valid=False))
    assert not state['motion_enabled'] and state['pause_generation']==1
    assert not stream.update_input_permission(state,legacy_pause=b'High')
    assert not state['motion_enabled']
    assert stream.update_input_permission(state,parsed=dict(valid=True,input_backend='unitree'))
    assert state['motion_enabled'] and state['resume_generation']==2


def test_legacy_apk_pause_still_controls_legacy_input():
    state=dict(input_backend='legacy',motion_enabled=True)
    assert stream.update_input_permission(state,legacy_pause=b'Low')
    assert not state['motion_enabled']
    assert not stream.update_input_permission(state,parsed=dict(valid=True,input_backend='legacy'))
    assert not state['motion_enabled']
    assert stream.update_input_permission(state,legacy_pause=b'High')
    assert state['motion_enabled']


def test_receiver_reports_xyz_motion_over_real_zmq_transport():
    guards = [socket.socket(), socket.socket()]
    for guard in guards:
        guard.bind(("127.0.0.1", 0))
    ports = [guard.getsockname()[1] for guard in guards]
    for guard in guards:
        guard.close()
    context = zmq.Context()
    pose, pause = [context.socket(zmq.PUSH) for _ in range(2)]
    for sock in (pose, pause):
        sock.setsockopt(zmq.SNDTIMEO, 1000)
        sock.setsockopt(zmq.LINGER, 0)
    process = subprocess.Popen([sys.executable, str(Path(stream.__file__)), "--duration", "1", "--pose-port", str(ports[0]), "--pause-port", str(ports[1]), "--pub-port", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert json.loads(process.stdout.readline())["type"] == "ready"
        pose.connect(f"tcp://127.0.0.1:{ports[0]}")
        pause.connect(f"tcp://127.0.0.1:{ports[1]}")
        pause.send(b"High")
        pose.send(b"absolute|1,2,3|1,0,0,0|False|1.1,2,3|1,2.1,3|1,2,2.9")
        pose.send(b"absolute|1.05,2.1,3.2|1,0,0,0|True|1.15,2.1,3.2|1.05,2.2,3.2|1.05,2.1,3.1|True")
        pose.send(b"absolute|1.05,2.1,3.2|1,0,0,0|False|1.15,2.1,3.2|1.05,2.2,3.2|1.05,2.1,3.1|True")
        pose.send(b"absolute|1.05,2.1,3.2|1,0,0,0|True|1.15,2.1,3.2|1.05,2.2,3.2|1.05,2.1,3.1|True")
        output, error = process.communicate(timeout=5)
        assert process.returncode == 0, error
        states = [json.loads(line) for line in output.splitlines()]
        last = states[-1]
        assert last["valid_count"] == 4
        assert last["position_m"] == [1.05, 2.1, 3.2]
        assert last["position_span_mm"] == pytest.approx([50, 100, 200])
        assert last["motion_enabled"] is True
        assert last["trigger_pressed"] is True
        assert last["a_pressed"] is True
        assert last["trigger_release_generation"] == 1
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        pose.close()
        pause.close()
        context.term()
