#!/usr/bin/env python3
"""Receive Quest WebXR or legacy APK input; never import a robot SDK."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time

import zmq
from scipy.spatial.transform import Rotation

from quest_usb_bridge import parse_controller_packet


def inspect_webxr_packet(message):
    """Independent XR buttons; never derive B from legacy High/Low state."""
    try:
        value = json.loads(message)
        if value.get('type') not in ('quest_webxr_v1', 'quest_unitree_v1') or type(value.get('valid')) is not bool:
            raise ValueError('unsupported packet')
        if not value['valid']:
            return {'valid':False,'reason':value.get('reason','WebXR 未追踪右手柄')}
        for name,size in (('position_m',3),('quaternion_wxyz',4)):
            numbers = value.get(name)
            if not isinstance(numbers,list) or len(numbers)!=size or any(type(x) not in (int,float) or not math.isfinite(x) for x in numbers):
                raise ValueError('invalid vector')
        if sum(x*x for x in value['position_m']) < 1e-8 or sum(x*x for x in value['quaternion_wxyz']) < .25:
            raise ValueError('untracked pose')
        for key in ('trigger_pressed','a_pressed','b_pressed'):
            if type(value.get(key)) is not bool:
                raise ValueError('missing independent button')
        frame = value.get('frame_id')
        if not isinstance(frame,str) or not 1<=len(frame)<=100:
            raise ValueError('invalid frame')
        age = value.get('sample_age_s',0)
        if type(age) not in (int,float) or not math.isfinite(age) or not 0<=age<.2:
            raise ValueError('stale sample')
        generated=value.get('sample_server_s')
        if type(generated) not in (int,float) or not math.isfinite(generated):
            raise ValueError('missing synchronized sample time')
        transport_age=time.monotonic()-generated
        if transport_age<-.03 or transport_age>=.2:
            raise ValueError('stale transport sample')
        age=max(age,transport_age,0)
        headset = value.get('sample_headset_s')
        if headset is not None and (type(headset) not in (int,float) or not math.isfinite(headset) or headset < 0):
            raise ValueError('invalid headset sample timestamp')
        counters = value.get('release_counters',{})
        if not isinstance(counters,dict) or any(type(counters.get(k)) is not int or not 0<=counters[k]<=2147483647 for k in ('trigger','a','b')):
            raise ValueError('invalid release counters')
        w,x,y,z = value['quaternion_wxyz']
        interruptions = value.get('input_interruptions')
        if interruptions is not None and (type(interruptions) is not int or not 0 <= interruptions < 2**53):
            raise ValueError('invalid input interruption counter')
        return dict(valid=True,position_m=value['position_m'],quaternion_wxyz=value['quaternion_wxyz'],
            euler_deg=Rotation.from_quat([x,y,z,w]).as_euler('xyz',degrees=True).tolist(),
            trigger_pressed=value['trigger_pressed'],a_pressed=value['a_pressed'],b_pressed=value['b_pressed'],
            reason='有效位姿/独立按键',input_backend=('unitree' if value['type']=='quest_unitree_v1' else 'webxr'),frame_id=frame,sample_age_s=age,
            remote_release_counters=counters, sample_monotonic_s=generated,
            sample_headset_s=headset,
            remote_input_interruptions=interruptions, remote_input_error=str(value.get('last_input_error', ''))[:160],
            sample_time_source=('synchronized_headset_earliest' if value['type']=='quest_unitree_v1' and headset is not None
                                else 'synchronized_headset'))
    except (ValueError,TypeError,KeyError,AttributeError):
        return {'valid':False,'reason':'WebXR 位姿/按键格式无效或数据过期'}


def merge_release_counters(state, parsed, previous):
    if not parsed.get('valid'):
        return
    frame = parsed.get('frame_id','legacy')
    changed = previous.get('frame') is not None and previous['frame'] != frame
    if changed:
        state['invalid_count'] += 1  # a new coordinate origin must stop old goals
        state['last_invalid_reason'] = '输入参考坐标系/会话已改变'
        state['last_invalid_monotonic_s'] = time.monotonic()
    interruptions = parsed.get('remote_input_interruptions')
    old_interruptions = previous.get('input_interruptions')
    if not changed and interruptions is not None and old_interruptions is not None and interruptions != old_interruptions:
        state['invalid_count'] += max(1, interruptions-old_interruptions)
        state['last_invalid_reason'] = parsed.get('remote_input_error') or '遥操输入曾短暂中断'
        state['last_invalid_monotonic_s'] = time.monotonic()
    remote = parsed.get('remote_release_counters')
    for key in ('trigger','a','b'):
        name = key+'_release_generation'
        state.setdefault(name,0)
        if remote is not None:
            old = previous.get('counters',{}).get(key)
            if not changed and old is not None:
                state[name] += max(0,remote[key]-old)
        elif not changed and state.get(key+'_pressed') is True and parsed.get(key+'_pressed') is False:
            state[name] += 1
    previous.update(frame=frame,counters=remote or {},input_interruptions=interruptions)


def update_input_permission(state, parsed=None, legacy_pause=None):
    """XR permission belongs to its pose packet, never the legacy High/Low channel."""
    backend = (parsed or {}).get('input_backend', state.get('input_backend'))
    xr = backend in ('unitree', 'webxr')
    if xr and parsed is not None:
        enabled = parsed.get('valid') is True
    elif not xr and legacy_pause in (b'High', b'Low'):
        enabled = legacy_pause == b'High'
    else:
        return False
    if enabled != state.get('motion_enabled'):
        key = 'resume_generation' if enabled else 'pause_generation'
        state[key] = state.get(key, 0)+1
    state['motion_enabled'] = enabled
    state['pause_count'] = state.get('pause_count', 0)+1
    return True


def inspect_packet(message: bytes) -> dict:
    if message.lstrip().startswith(b'{'):
        return inspect_webxr_packet(message)
    parsed = parse_controller_packet(message)
    if parsed is None:
        parts = message.decode("utf-8", errors="replace").strip().split("|")
        try:
            raw_position = [float(v) for v in parts[1].split(",")]
            reason = "应用发送零位置：手柄未追踪或应用未正确获取位姿" if raw_position == [0, 0, 0] else "位姿格式或旋转数据无效"
        except (IndexError, ValueError):
            raw_position = None
            reason = "收到无法解析的数据包"
        return {"valid": False, "reason": reason, "raw_position_m": raw_position}
    _, pose, trigger = parsed
    euler = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_euler("xyz", degrees=True)
    return {
        "valid": True,
        "position_m": pose[:3],
        "quaternion_wxyz": pose[3:],
        "euler_deg": euler.tolist(),
        "trigger_pressed": trigger,
        # Never invent A=False for the old seven-field APK.
        "a_pressed": (message.decode("utf-8").strip().split("|")[7].strip().lower() == "true")
                     if len(message.decode("utf-8").strip().split("|")) == 8 else None,
        "b_pressed": None, "input_backend": "legacy", "sample_age_s": 0,
        "frame_id": "legacy", "remote_release_counters": None,
        "reason": "有效位姿",
    }


def receive(duration: float | None = None, pose_port: int = 8125, pause_port: int = 8100, pub_port: int = 8130) -> int:
    context = zmq.Context()
    sockets = {}
    poller = zmq.Poller()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    publisher = None
    state = {"type": "snapshot", "packet_count": 0, "valid_count": 0, "invalid_count": 0,
             "pause_generation": 0, "resume_generation": 0,
             "trigger_release_generation": 0,
             "pause_count": 0, "motion_enabled": None, "valid": False, "reason": "等待头显发送数据"}
    valid_times = []
    previous_buttons = {}
    last_packet = last_valid = last_pause = None
    next_report = started
    try:
        for name, port in (("pause", pause_port), ("pose", pose_port)):
            socket = context.socket(zmq.PULL)
            socket.setsockopt(zmq.RCVHWM, 100)
            sockets[name] = socket
            socket.bind(f"tcp://127.0.0.1:{port}")
            poller.register(socket, zmq.POLLIN)
        if pub_port:
            publisher = context.socket(zmq.PUB)
            publisher.setsockopt(zmq.SNDHWM, 5)
            publisher.bind(f"tcp://127.0.0.1:{pub_port}")
        print(json.dumps({"type": "ready", "ports": [pause_port, pose_port]}), flush=True)
        while running and (duration is None or time.monotonic() - started < duration):
            events = dict(poller.poll(20))
            now = time.monotonic()
            for name, socket in sockets.items():
                if socket not in events:
                    continue
                for _ in range(100):
                    try:
                        message = socket.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if name == "pause":
                        if update_input_permission(state, legacy_pause=message):
                            last_pause = now
                        continue
                    last_packet = now
                    state["packet_count"] += 1
                    parsed = inspect_packet(message)
                    if update_input_permission(state, parsed=parsed):
                        last_pause = now
                    if parsed['valid'] and 'sample_monotonic_s' not in parsed:
                        # Legacy APK has no acquisition clock. Keep receipt
                        # time explicit rather than claim hardware sampling.
                        parsed.update(sample_monotonic_s=time.monotonic(),
                                      sample_time_source='host_receipt')
                    merge_release_counters(state,parsed,previous_buttons)
                    state.update(parsed)
                    if parsed["valid"]:
                        last_valid = now
                        state["valid_count"] += 1
                        valid_times.append(now)
                        state["sample_timestamp"] = time.time()
                        position = parsed["position_m"]
                        state["position_min_m"] = [min(a, b) for a, b in zip(state.get("position_min_m", position), position)]
                        state["position_max_m"] = [max(a, b) for a, b in zip(state.get("position_max_m", position), position)]
                        state["position_span_mm"] = [(b - a) * 1000 for a, b in zip(state["position_min_m"], state["position_max_m"])]
                    else:
                        state["invalid_count"] += 1
                        state['last_invalid_reason'] = parsed.get('reason', '输入无效')
                        state['last_invalid_monotonic_s'] = now
            # Control subscribers receive fresh input immediately. The GUI
            # stdout cadence stays bounded independently of control delivery.
            report_due = now >= next_report
            if report_due or (publisher is not None and events):
                valid_times = [t for t in valid_times if now - t <= 1.0]
                state.update(
                    packet_age_s=None if last_packet is None else now - last_packet,
                    valid_age_s=None if last_valid is None else now - last_valid + state.get('sample_age_s',0),
                    pause_age_s=None if last_pause is None else now - last_pause,
                    valid_hz=round(len(valid_times) / min(max(now - started, 0.1), 1.0), 1),
                    timestamp=time.time(),
                )
                if report_due:
                    print(json.dumps(state, ensure_ascii=False), flush=True)
                    next_report = now + 0.02
                if publisher is not None:
                    publisher.send_json(state)
        return 0
    except zmq.ZMQError as error:
        print(json.dumps({"type": "error", "message": f"接收端口被占用或无法打开：{error}。请先停止旧遥操/手柄桥。"}, ensure_ascii=False), flush=True)
        return 1
    finally:
        for socket in sockets.values():
            socket.close(linger=0)
        if publisher is not None:
            publisher.close(linger=0)
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--pose-port", type=int, default=8125)
    parser.add_argument("--pause-port", type=int, default=8100)
    parser.add_argument("--pub-port", type=int, default=8130, help="共享只读数据端口；0 表示禁用")
    args = parser.parse_args()
    raise SystemExit(receive(args.duration, args.pose_port, args.pause_port, args.pub_port))
