"""Strict TeleVuer CONTROLLER_MOVE adapter; no robot SDK or motion commands.

Keep WORLD poses: do not use TeleVuerWrapper's humanoid/head-relative mapping.
The quality metadata is added at the XR animation frame, not at receipt time.
"""
import math
import time
from collections import Counter, deque

import numpy as np
from scipy.spatial.transform import Rotation


class UnitreeInput:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.frame = None
        self.sequence = -1
        self.seen = -1e9
        self.received = None
        self.max_receive_gap = 0.
        self.counts = Counter()
        self.invalid_reasons = Counter()
        self.transitions = deque(maxlen=20)
        self.last_result = None
        self.clock_quality = {}
        self.client_version = None
        self.tracking = {}
        self.headset_sample = None
        self.timing = {}
        self.interrupt_generation = 0
        self.client_interruptions = None
        self.last_input_error = ''
        self.transport = {}

    def decode(self, value):
        now = self.clock()
        if self.received is not None:
            self.max_receive_gap = max(self.max_receive_gap, now - self.received)
        self.received = now
        packet = self._decode(value)
        valid = packet['valid']
        self.counts['valid' if valid else 'invalid'] += 1
        reason = '' if valid else packet['reason']
        if not valid:
            self.invalid_reasons[reason] += 1
            self.interrupt_generation += 1
            self.last_input_error = reason
        packet.update(input_interruptions=self.interrupt_generation, last_input_error=self.last_input_error)
        if (valid, reason) != self.last_result:
            self.transitions.append(dict(timestamp=time.time(), valid=valid, reason=reason, **self.timing))
            self.last_result = (valid, reason)
        meta = value.get('xarmMeta', {}) if isinstance(value, dict) else {}
        version = meta.get('client_version') if isinstance(meta, dict) else None
        self.client_version = version[:80] if isinstance(version, str) else None
        tracking = meta.get('tracking', {}) if isinstance(meta, dict) else {}
        self.tracking = {k: tracking[k] for k in ('pose_available', 'emulated_position')
                         if isinstance(tracking, dict) and type(tracking.get(k)) is bool}
        quality = meta.get('clock', {}) if isinstance(meta, dict) else {}
        transport = meta.get('transport', {}) if isinstance(meta, dict) else {}
        self.transport = {}
        if isinstance(transport, dict):
            for key in ('in_flight', 'sent', 'skipped', 'reconnects',
                        'skip_pending', 'skip_buffer', 'skip_closed'):
                number = transport.get(key)
                if type(number) is int and 0 <= number < 2**53:
                    self.transport[key] = number
            if isinstance(transport.get('last_error'), str):
                self.transport['last_error'] = transport['last_error'][:160]
        self.clock_quality = {}
        if isinstance(quality, dict):
            for key in ('age_ms', 'rtt_ms', 'consecutive_failures', 'total_failures',
                        'ws_failures', 'http_failures'):
                number = quality.get(key)
                if type(number) in (float, int) and math.isfinite(number) and number >= 0:
                    self.clock_quality[key] = number
            if isinstance(quality.get('last_error'), str):
                self.clock_quality['last_error'] = quality['last_error'][:160]
            for key in ('ws_last_error', 'http_last_error'):
                if isinstance(quality.get(key), str):
                    self.clock_quality[key] = quality[key][:160]
            if quality.get('transport') in ('http', 'websocket'):
                self.clock_quality['transport'] = quality['transport']
        return packet

    def diagnostics(self):
        return dict(valid_packets=self.counts['valid'], invalid_packets=self.counts['invalid'],
                    invalid_reasons=dict(self.invalid_reasons),
                    receive_age_ms=None if self.received is None else max(0., self.clock()-self.received)*1000,
                    max_receive_gap_ms=self.max_receive_gap*1000,
                    clock=dict(self.clock_quality), client_version=self.client_version,
                    tracking=dict(self.tracking), timing=dict(self.timing), transport=dict(self.transport),
                    recent_transitions=list(self.transitions))

    def _decode(self, value):
        now = self.clock()
        self.timing = {}
        try:
            meta = value['xarmMeta']
            frame, sequence = meta['frame_id'], meta['sequence']
            if (not isinstance(frame, str) or not 1 <= len(frame) <= 100 or
                    type(sequence) is not int or not 0 <= sequence <= 2147483647):
                raise ValueError('invalid session/sequence')
            if frame != self.frame:
                if now - self.seen < .5:
                    raise ValueError('another XR session owns input')
                self.frame, self.sequence = frame, -1
                self.headset_sample = None
                self.client_interruptions = None
            if sequence <= self.sequence:
                raise ValueError('duplicate/out-of-order XR frame')
            self.sequence = sequence
            if meta.get('valid') is not True:
                raise ValueError(meta.get('reason', 'controller tracking invalid'))
            generated = meta['sample_server_s']
            if type(generated) not in (float, int) or not math.isfinite(generated):
                raise ValueError('invalid XR sample timestamp')
            age = now - generated
            self.timing['sample_age_ms'] = age*1000
            if age < -.03:
                raise ValueError('XR 时间戳超前：电脑/头显时钟估计不一致')
            if age >= .18:
                raise ValueError('XR 数据过期：采样至接收超过 180 ms')
            headset = meta.get('sample_headset_s')
            if 'sample_headset_s' in meta or meta.get('client_version') in ('input-timing-v5', 'realtime-input-v6'):
                if type(headset) not in (int, float) or not math.isfinite(headset) or headset < 0:
                    raise ValueError('invalid headset sample timestamp')
                if self.headset_sample is not None and headset <= self.headset_sample:
                    raise ValueError('headset sample timestamp repeated/reversed')
            faults = meta.get('input_interruptions')
            if 'input_interruptions' in meta or meta.get('client_version') == 'realtime-input-v6':
                if type(faults) is not int or not 0 <= faults <= 2147483647:
                    raise ValueError('invalid input interruption counter')
                if self.client_interruptions is not None:
                    if faults < self.client_interruptions:
                        raise ValueError('input interruption counter reversed')
                    if faults > self.client_interruptions:
                        self.interrupt_generation += faults-self.client_interruptions
                        self.last_input_error = str(meta.get('last_input_error') or '头显输入短暂中断')[:160]
                self.client_interruptions = faults
            raw = value['right']
            if (not isinstance(raw, list) or len(raw) != 16 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in raw)):
                raise ValueError('invalid controller matrix')
            matrix = np.asarray(raw).reshape(4, 4, order='F')
            rotation = matrix[:3, :3]
            if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5) or
                    not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or
                    abs(np.linalg.det(rotation) - 1) > 1e-3):
                raise ValueError('non-rigid controller matrix')
            buttons = value['rightState']
            pressed = {}
            for name, upstream in (('trigger', 'trigger'), ('a', 'aButton'), ('b', 'bButton')):
                state = buttons[upstream]
                if type(state) is not bool or state != meta['buttons'][name]:
                    raise ValueError('independent button mapping mismatch')
                pressed[name + '_pressed'] = state
            counters = meta['release_counters']
            if any(type(counters.get(k)) is not int or not 0 <= counters[k] <= 2147483647
                   for k in ('trigger', 'a', 'b')):
                raise ValueError('invalid release counters')
            # Same RH -> Unity-LH convention as the existing input. No extra
            # head transform, calibration, smoothing, gain, or axis swapping.
            basis = np.diag([1., 1., -1.])
            p = basis @ matrix[:3, 3]
            q = Rotation.from_matrix(basis @ rotation @ basis).as_quat()
            self.seen = now
            self.headset_sample = headset
            return dict(type='quest_unitree_v1', valid=True, frame_id=frame,
                        sequence=sequence, position_m=p.tolist(),
                        quaternion_wxyz=[q[3], q[0], q[1], q[2]],
                        sample_server_s=generated, sample_age_s=max(0, age),
                        sample_headset_s=headset,
                        release_counters=dict(counters), **pressed)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            return dict(type='quest_unitree_v1', valid=False,
                        reason='宇树输入无效：' + str(error))
