"""Synthetic input-only measurements. No SDK import, networking or robot I/O."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest_pose_filter import AdaptivePoseFilter
from quest_pose_math import quaternion_wxyz, orientation_rotation


def run():
    rates = {}
    timings = []
    for hz in (90, 100, 120):
        f = AdaptivePoseFilter()
        raw_p, filtered_p, raw_r, filtered_r = [], [], [], []
        for i in range(hz*4):
            t = i/hz
            noise_p = .5*np.sin(2*np.pi*11*t)
            noise_r = .3*np.sin(2*np.pi*13*t)
            q = quaternion_wxyz(Rotation.from_euler('z', noise_r, degrees=True))
            start = time.perf_counter()
            p, qf = f.update([.2+noise_p/1000, 1, .3], q, 10+t, ('unitree','synthetic'))
            timings.append((time.perf_counter()-start)*1000)
            if t > 1:
                raw_p.append(noise_p); filtered_p.append((p[0]-.2)*1000)
                raw_r.append(noise_r)
                filtered_r.append(np.rad2deg(orientation_rotation(qf).as_rotvec()[2]))
        rates[str(hz)] = dict(raw_position_std_mm=float(np.std(raw_p)),
            filtered_position_std_mm=float(np.std(filtered_p)),
            raw_rotation_std_deg=float(np.std(raw_r)), filtered_rotation_std_deg=float(np.std(filtered_r)))

    ramp = {}
    for kind, config in [('adaptive', {}), ('fixed_2hz', dict(position_beta_hz_per_mm_s=0, rotation_beta_hz_per_deg_s=0))]:
        f = AdaptivePoseFilter(config)
        position_errors, angle_errors = [], []
        settle_position = settle_rotation = None
        for i in range(401):
            t = i*.01
            distance, angle = min(t,1)*100, min(t,1)*45
            q = quaternion_wxyz(Rotation.from_euler('z', angle, degrees=True))
            p, qf = f.update([.2+distance/1000,1,.3],q,10+t,('unitree','synthetic'))
            dp = abs((p[0]-.2)*1000-distance)
            dr = np.rad2deg((orientation_rotation(qf)*orientation_rotation(q).inv()).magnitude())
            if .5 <= t <= 1:
                position_errors.append(dp); angle_errors.append(dr)
            if t >= 1:
                if dp < .1 and settle_position is None: settle_position=t-1
                if dr < .1 and settle_rotation is None: settle_rotation=t-1
        ramp[kind] = dict(position_lag_p95_mm=float(np.percentile(position_errors,95)),
            rotation_lag_p95_deg=float(np.percentile(angle_errors,95)),
            time_after_stop_to_0_1mm_s=settle_position, time_after_stop_to_0_1deg_s=settle_rotation)
    return dict(scope='Synthetic input filter only; not measured hardware jitter or end-to-end latency.',
                static_noise='0.5 mm amplitude at 11 Hz; 0.3 deg amplitude at 13 Hz; first second excluded',
                ramp='100 mm/s and 45 deg/s for 1 second, then hold; 100 Hz input',
                stationary=rates, ramp_comparison=ramp,
                compute_ms=dict(zip(('p50','p95','p99'),map(float,np.percentile(timings,[50,95,99])))))


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
