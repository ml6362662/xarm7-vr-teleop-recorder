"""Manual diagnostic: real controller FK/IK only; all motion goes to FakeArm.

Run from project root with the backend Python. Not a physical tracking test.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_relative_motion import FakeArm, Clock, engage, tick_100, state
from xarm_quest_relative import RelativeArm, wait_initial_reports


def main():
    from xarm.wrapper import XArmAPI
    from xarm.core.config.x_config import XCONF
    root=Path(__file__).resolve().parents[1]
    rows=[json.loads(s) for s in (root/'.state/relative_control_trace.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r.get('run_id')=='1791623936.089955']
    boundary=min(rows,key=lambda r:abs(r['timestamp']-1791624090.12045))
    start=min(rows,key=lambda r:abs(r['timestamp']-1791624074.3427057))
    config=json.loads((root/'relative_control_settings.json').read_text())
    config.update(teleop_button='b',gripper_button='a',gripper_enabled=False,rotation_enabled=True)
    real=XArmAPI(json.loads((root/'config.json').read_text())['robot_ip'],is_radian=True)
    try:
        wait_initial_reports(real)
        code,hardware=real.get_servo_angle(is_radian=True);assert code==0
        limits=np.array(XCONF.Robot.JOINT_LIMITS[real.axis][real.device_type])
        for kind in ['boundary_retreat','boundary_approach_reverse','historical_mixed']:
            q=np.deg2rad((start if kind=='historical_mixed' else boundary)['joints_deg'])
            code,p=real.get_forward_kinematics(q.tolist(),input_is_radian=True,return_is_radian=True);assert code==0
            p=np.array(p)
            class VirtualArm(FakeArm):
                def __init__(self):
                    super().__init__();self.joints=q.copy();self.home=p.copy();self.tcp=p.copy()
                # Only these TWO kinematic queries delegate to hardware.
                def get_forward_kinematics(self,*a,**kw):return real.get_forward_kinematics(*a,**kw)
                def get_inverse_kinematics(self,*a,**kw):return real.get_inverse_kinematics(*a,**kw)
            arm=VirtualArm();clock=Clock()
            c=RelativeArm(arm,limits[:,0]+np.deg2rad(5),limits[:,1]-np.deg2rad(5),clock,control_settings=config)
            engage(clock,c);times=[];holds=0;projected=0;minimum=1000
            hand_rot=Rotation.identity();previous_rot=Rotation.from_euler('xyz',p[3:])
            initial=c.pose_guard.snapshot()['sigma_min']
            for i in range(1,251):
                z= -.2*i if kind!='boundary_approach_reverse' else (.2*i if i<75 else 15-.3*(i-75))
                hand=np.array([.2,1,.3])+c.mapper.matrix.T@np.array([0.,0.,z])/(1000*c.mapper.scale)
                payload=state(position=hand,invalid=1,pause=1,count=i+3)
                if kind=='historical_mixed':
                    desired=Rotation.from_euler('xyz',p[3:]+np.deg2rad([.012*i,-.03*i,0]))
                    rv=(desired*previous_rot.inv()).as_rotvec();previous_rot=desired
                    hand_rot=Rotation.from_rotvec(np.linalg.det(c.mapper.matrix)*c.mapper.matrix.T@rv/c.settings['rotation_scale'])*hand_rot
                    qr=hand_rot.as_quat();payload['quaternion_wxyz']=[qr[3],*qr[:3]]
                before=time.monotonic();tick_100(clock,c,payload);times.append((time.monotonic()-before)*1000)
                fb=c.pose_guard.snapshot();minimum=min(minimum,fb['sigma_min'])
                holds+=fb.get('status')=='direction_hold'
                projected+=bool((fb.get('direction_info') or {}).get('translation_only'))
                if not c.servo_started:break
            print(json.dumps(dict(case=kind,initial_sigma=initial,cycles=i,active=c.servo_started,status=c.status,
                delta_xyz_mm=(arm.tcp[:3]-p[:3]).tolist(),sigma=fb['sigma_min'],min_sigma=minimum,
                held_frames=holds,translation_priority_frames=projected,sent=c.command_count,
                compute_ms_p50_p95_max=np.percentile(times,[50,95,100]).tolist()),ensure_ascii=False))
        code,after=real.get_servo_angle(is_radian=True);assert code==0
        print('HARDWARE',dict(joint_change_deg=float(np.max(np.abs(np.rad2deg(np.array(after)-hardware)))),
                             state=real.get_state(),errors=real.get_err_warn_code()))
    finally:
        real.disconnect()


if __name__=='__main__':main()
