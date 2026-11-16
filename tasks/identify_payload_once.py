"""One operator-confirmed vendor payload identification; never apply its result."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import threading
import time

from xarm.wrapper import XArmAPI

ROOT = Path(__file__).resolve().parents[1]


def checked(result, label):
    code = result[0] if isinstance(result, (tuple, list)) else result
    if code != 0:
        raise RuntimeError(f'{label}: SDK {code}')
    return result[1] if isinstance(result, (tuple, list)) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-confirmed-motion', action='store_true', required=True)
    args = parser.parse_args()
    assert args.operator_confirmed_motion
    cancel = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancel.set())
    lock = (ROOT/'.state/xarm_relative.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ip = json.loads((ROOT/'config.json').read_text())['robot_ip']
    monitor = worker = None
    owned = False
    record = {'operation': 'vendor_tcp_load_identification', 'applied': False}
    path = ROOT/'.state'/f'payload_identification_{time.time_ns()}.json'

    def save():
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))

    try:
        monitor = XArmAPI(ip, is_radian=True, report_type='rich')
        time.sleep(.6)
        if checked(monitor.get_err_warn_code(), 'errors') != [0, 0]:
            raise RuntimeError('Existing robot error/warning; no automatic clearing')
        if checked(monitor.get_state(), 'state') != 4:
            raise RuntimeError('Robot must already be stopped in state 4')
        states = checked(monitor.get_joint_states(is_radian=True, num=2), 'joint states')
        if len(states[1]) != 7 or any(not math.isfinite(v) or abs(v) > math.radians(.5) for v in states[1]):
            raise RuntimeError('Robot not stationary')
        if monitor.axis != 7 or tuple(monitor.version_number) < (1, 8, 0):
            raise RuntimeError('Unexpected robot or unsupported firmware')
        if monitor.collision_sensitivity == 0:
            raise RuntimeError('Collision protection is disabled before identification; review and restore it first')
        if sum((a-b)**2 for a,b in zip(monitor.gravity_direction, [0,0,-1])) > .01:
            raise RuntimeError('Mounting configuration not upright; do not change it automatically')
        record['before'] = {k: getattr(monitor, k) for k in
            ('sn', 'version', 'tcp_load', 'tcp_offset', 'world_offset', 'gravity_direction', 'collision_sensitivity')}
        record['initial_joints_rad'] = states[0]
        save()
        print('BACKUP', path, json.dumps(record['before']), flush=True)
        if cancel.is_set():
            raise RuntimeError('Cancelled before motion')
        # A dedicated connection is necessary: the vendor identification holds
        # its command lock for up to 300 s. Stop/health use another connection.
        worker = XArmAPI(ip, is_radian=True, report_type='rich')
        progress = {'value': 0}

        def on_progress(data):
            progress['value'] = data.get('progress', 0)
            print('PROGRESS', data, flush=True)

        worker.register_iden_progress_changed_callback(on_progress)
        owned = True
        checked(monitor.set_state(4), 'clear old motion')
        checked(monitor.set_mode(0), 'position mode')
        checked(monitor.motion_enable(enable=True), 'enable')
        checked(monitor.set_state(0), 'ready')
        done = threading.Event()
        result = {}

        def identify():
            try:
                result['reply'] = worker.iden_tcp_load()
            except Exception as error:
                result['exception'] = repr(error)
            finally:
                done.set()

        if cancel.is_set():
            raise RuntimeError('Cancelled before identification')
        start = time.monotonic()
        threading.Thread(target=identify, daemon=True).start()
        print('IDENTIFICATION STARTED: real automatic motion; use physical E-stop for danger', flush=True)
        last_log = start
        completed_motion_at = None
        while not done.wait(.25):
            if cancel.is_set() or time.monotonic()-start > 310:
                raise RuntimeError('Identification cancelled or timed out')
            if not monitor.connected:
                raise RuntimeError('Monitor disconnected; use physical E-stop')
            errors = checked(monitor.get_err_warn_code(), 'monitor errors')
            if errors != [0, 0]:
                raise RuntimeError(f'Identification robot error/warning {errors}')
            state = checked(monitor.get_state(), 'monitor state')
            if state == 4 and progress['value'] >= 100:
                # Firmware can stop BEFORE the long-running SDK request
                # returns its computed payload. Do not kill that reply window.
                completed_motion_at = completed_motion_at or time.monotonic()
                if time.monotonic()-completed_motion_at > 30:
                    raise RuntimeError('Motion complete but no identification result within 30 s')
            elif state in (4, 5):
                if done.wait(.2):
                    break
                raise RuntimeError(f'Identification interrupted, state {state}')
            if time.monotonic()-last_log >= 10:
                print('RUNNING', round(time.monotonic()-start, 1), 'seconds, state', state, flush=True)
                last_log = time.monotonic()
        record['result'] = result
        record['elapsed_s'] = time.monotonic()-start
        if 'exception' in result:
            raise RuntimeError(result['exception'])
        load = checked(result['reply'], 'identification')
        if len(load) != 4 or any(not math.isfinite(v) for v in load) or load[0] <= 0:
            raise RuntimeError('Invalid identification result, NOT applied')
        record['status'] = 'identified_not_applied'
        print('RESULT_NOT_APPLIED', json.dumps(load), flush=True)
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = str(error)
        print('FAILED', str(error), flush=True)
    finally:
        if monitor is not None:
            if owned:
                try:
                    record['stop_code'] = monitor.set_state(4)
                    record['final_state'] = monitor.get_state()
                    record['final_errors'] = monitor.get_err_warn_code()
                    time.sleep(.3)
                    record['after_tcp_load'] = monitor.tcp_load
                    record['after_tcp_offset'] = monitor.tcp_offset
                    print('STOP', record['stop_code'], 'state', record['final_state'], flush=True)
                    original = record['before']['tcp_load']
                    # Identification itself can zero the running payload.
                    # Restore the known prior configuration, not an unreviewed
                    # estimate; never leave the tool configured as no-load.
                    if monitor.tcp_load != original:
                        if record['final_state'] != (0, 4) or record['final_errors'] != (0, [0, 0]):
                            raise RuntimeError('Cannot safely restore payload with robot fault/unconfirmed stop')
                        states = checked(monitor.get_joint_states(is_radian=True, num=2), 'restore stationary check')
                        if any(not math.isfinite(v) or abs(v) > math.radians(.5) for v in states[1]):
                            raise RuntimeError('Cannot restore payload until stationary')
                        try:
                            # Firmware 2.8 rejects set_tcp_load in state 4.
                            # Clear queue, enter position-ready without a motion
                            # command, restore, then immediately stop again.
                            checked(monitor.set_state(4), 'restore clear queue')
                            checked(monitor.set_mode(0), 'restore position mode')
                            checked(monitor.set_state(0), 'restore ready')
                            checked(monitor.set_tcp_load(original[0], original[1], wait=False), 'restore original payload')
                        finally:
                            checked(monitor.set_state(4), 'restore stop')
                        time.sleep(.5)
                        if monitor.tcp_load != original:
                            raise RuntimeError('Original payload restoration not confirmed')
                        record['restored_tcp_load'] = monitor.tcp_load
                        record['final_state'] = monitor.get_state()
                        record['final_errors'] = monitor.get_err_warn_code()
                    sensitivity = record['before']['collision_sensitivity']
                    if monitor.collision_sensitivity != sensitivity:
                        if monitor.get_state() != (0, 4) or monitor.get_err_warn_code() != (0, [0, 0]):
                            raise RuntimeError('Cannot restore protection with fault/unconfirmed stop')
                        try:
                            checked(monitor.set_collision_sensitivity(sensitivity, wait=False), 'restore collision protection')
                        finally:
                            # The SDK setter may make the controller ready.
                            checked(monitor.set_state(4), 'stop after restoring protection')
                        time.sleep(.5)
                        if monitor.collision_sensitivity != sensitivity:
                            raise RuntimeError('Collision protection restoration not confirmed')
                    record['final_collision_sensitivity'] = monitor.collision_sensitivity
                    record['final_state'] = monitor.get_state()
                    record['final_errors'] = monitor.get_err_warn_code()
                except Exception as error:
                    record['stop_error'] = str(error)
                    print('STOP/RESTORE ISSUE: keep robot stopped; check physical E-stop', str(error), flush=True)
            monitor.disconnect()
        # On cancellation the worker is daemonized; do not wait 300 s for its
        # blocked reply. Stop is issued through the independent monitor socket.
        save()
        print('RECORD', path, flush=True)
        lock.close()
    return 0 if (record.get('status') == 'identified_not_applied' and record.get('stop_code') == 0
        and record.get('final_state') == (0, 4) and record.get('final_errors') == (0, [0, 0])
        and not record.get('stop_error')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
