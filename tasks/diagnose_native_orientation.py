"""Isolated legacy Quest pose receiver. No robot SDK or command publishing."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest_usb_bridge import parse_controller_packet


def summarize(rows, elapsed, invalid):
    info = dict(elapsed_s=round(elapsed, 2), samples=len(rows), invalid_total=invalid)
    if rows:
        rotations=Rotation.from_quat(np.asarray([r['q_wxyz'] for r in rows])[:, [1,2,3,0]])
        info.update(xyz_span_mm=(np.ptp([r['p_m'] for r in rows],axis=0)*1000).round(3).tolist(),
            quaternion_span_from_first_deg=round(float(np.max(np.rad2deg((rotations*rotations[0].inv()).magnitude()))),4),
            euler_span_deg=np.ptp(rotations.as_euler('xyz',degrees=True),axis=0).round(3).tolist())
    print('BLOCK',json.dumps(info),flush=True)


def run(args):
    ctx=zmq.Context();poll=zmq.Poller();sockets={}
    first=None;last_report=time.monotonic();start=last_report;block=[];count=0;invalid=0
    try:
        for name,port in [('pose',args.pose_port),('pause',args.pause_port)]:
            s=ctx.socket(zmq.PULL);s.setsockopt(zmq.LINGER,0);s.bind(f'tcp://127.0.0.1:{port}')
            sockets[s]=name;poll.register(s,zmq.POLLIN)
        with args.output.open('x',encoding='utf-8',buffering=1) as out:
            print('ISOLATED_RECEIVER_READY',flush=True)
            while True:
                now=time.monotonic()
                if (first is None and now-start>=args.wait) or (first is not None and now-first>=args.duration):break
                for sock,event in poll.poll(100):
                    raw=sock.recv();now=time.monotonic();name=sockets[sock]
                    row={'time':time.time(),'elapsed_s':now-start,'channel':name,'raw':raw.decode('utf-8',errors='replace')}
                    if name=='pose':
                        parsed=parse_controller_packet(raw)
                        if parsed:
                            if first is None:
                                first=now;last_report=now
                                print('FIRST_VALID_NATIVE_POSE',flush=True)
                            _,pose,_=parsed
                            row.update(p_m=pose[:3],q_wxyz=pose[3:],sample_s=now-first)
                            block.append(row);count+=1
                        else:invalid+=1
                    out.write(json.dumps(row,ensure_ascii=False)+'\n')
                if now-last_report>=10:
                    summarize(block,now-(first or start),invalid);block=[];last_report=now
            summarize(block,time.monotonic()-(first or start),invalid)
            print('COMPLETE',json.dumps(dict(valid=count,invalid=invalid,output=str(args.output))),flush=True)
    finally:
        for sock in sockets:sock.close()
        ctx.term()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pose-port',type=int,default=18125)
    p.add_argument('--pause-port',type=int,default=18100)
    p.add_argument('--duration',type=float,default=60)
    p.add_argument('--wait',type=float,default=120)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not 1<=args.duration<=120 or not 1<=args.wait<=180:p.error('invalid duration/wait')
    if args.pose_port in (8100,8125,8130) or args.pause_port in (8100,8125,8130):
        p.error('diagnostic must not bind production ports')
    run(args)
