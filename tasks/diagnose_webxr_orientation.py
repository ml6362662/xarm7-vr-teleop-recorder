"""Read-only WebXR observer via an already authorized Quest DevTools socket.

No robot SDK, application handler replacements, input injection, or settings.
Adds a bounded independent XR frame observer and removes its event listener.
Saves raw diagnostics (no page URL/token) for repeatable analysis.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
from scipy.spatial.transform import Rotation


EXPRESSION = r"""
(async () => {
  if (typeof session === 'undefined' || !session || !reference) return {error:'No active XR session'};
  const s=session, ref=reference;
  const local=await s.requestReferenceSpace('local');
  let resets=0, frames=0, finished=false, rows=[];
  const onReset=()=>resets++;
  ref.addEventListener('reset',onReset);
  const start=performance.now();
  const poseData=p=>p ? {p:[p.transform.position.x,p.transform.position.y,p.transform.position.z],
    q:[p.transform.orientation.x,p.transform.orientation.y,p.transform.orientation.z,p.transform.orientation.w],
    emulated:p.emulatedPosition} : null;
  return await new Promise(resolve=>{
    let timeout;
    function finish(){
      if(finished)return;finished=true;clearTimeout(timeout);ref.removeEventListener('reset',onReset);
      resolve({rows,resets,elapsed_ms:performance.now()-start});
    }
    function sample(t,f){
      if(finished)return;
      try{
        const right=Array.from(s.inputSources).find(v=>v.handedness==='right'&&v.gripSpace);
        const left=Array.from(s.inputSources).find(v=>v.handedness==='left'&&v.gripSpace);
        if((frames++ % 3)===0){
          rows.push({t:performance.now()-start,visibility:s.visibilityState,
            profiles:right ? Array.from(right.profiles):[],hand:!!(right&&right.hand),
            left_profiles:left ? Array.from(left.profiles):[],left_hand:!!(left&&left.hand),
            left_grip:left ? poseData(f.getPose(left.gripSpace,ref)):null,
            grip:right ? poseData(f.getPose(right.gripSpace,ref)):null,
            ray:right ? poseData(f.getPose(right.targetRaySpace,ref)):null,
            grip_local:right ? poseData(f.getPose(right.gripSpace,local)):null,
            head:poseData(f.getViewerPose(ref)),resets});
        }
        if(performance.now()-start>=DURATION_MS)finish();else s.requestAnimationFrame(sample);
      }catch(e){rows.push({t:performance.now()-start,error:String(e)});finish();}
    }
    timeout=setTimeout(finish,DURATION_MS+1000);
    s.requestAnimationFrame(sample);
  });
})()
"""


def relative_pair_metrics(rows):
    """Invariant under common rigid-board motion; no physical fixation assumed."""
    paired=[r for r in rows if r.get('grip') and r.get('left_grip') and all(
        any(p.startswith(('oculus-touch','meta-quest-touch')) for p in r.get(key,[]))
        for key in ('profiles','left_profiles'))]
    if not paired:return {'paired_samples':0}
    right=Rotation.from_quat([r['grip']['q'] for r in paired])
    left=Rotation.from_quat([r['left_grip']['q'] for r in paired])
    relative=right.inv()*left
    delta=np.asarray([r['left_grip']['p'] for r in paired])-np.asarray([r['grip']['p'] for r in paired])
    local=right.inv().apply(delta)
    return dict(paired_samples=len(paired),
        relative_orientation_change_deg=float(np.max(np.rad2deg((relative*relative[0].inv()).magnitude()))),
        relative_position_span_mm=(np.ptp(local,axis=0)*1000).tolist(),
        distance_span_mm=float(np.ptp(np.linalg.norm(delta,axis=1))*1000))


def summarize(data):
    rows=data.get('rows',[])
    print('SUMMARY',json.dumps({'rows':len(rows),'resets':data.get('resets'),
        'elapsed_ms':data.get('elapsed_ms'),'error':data.get('error')},ensure_ascii=False))
    for start in range(0, int(data.get('elapsed_ms',0))+1, 10000):
        chunk=[r for r in rows if start<=r['t']<start+10000]
        valid=[r for r in chunk if r.get('grip') and any(
            p.startswith(('oculus-touch','meta-quest-touch')) for p in r.get('profiles',[]))]
        info={'start_s':start/1000,'samples':len(chunk),'touch_samples':len(valid),
              'profiles':sorted(set(p for r in chunk for p in r.get('profiles',[])))}
        info['rigid_pair']=relative_pair_metrics(chunk)
        left=[r['left_grip'] for r in chunk if r.get('left_grip') and any(
            p.startswith(('oculus-touch','meta-quest-touch')) for p in r.get('left_profiles',[]))]
        if left:
            rots=Rotation.from_quat([p['q'] for p in left])
            info['left_grip']={'samples':len(left),
                'position_span_mm':(np.ptp([p['p'] for p in left],axis=0)*1000).round(3).tolist(),
                'orientation_span_from_first_deg':round(float(np.max(np.rad2deg((rots*rots[0].inv()).magnitude()))),4),
                'emulated_count':sum(p.get('emulated') is True for p in left)}
        if valid:
            for name in ('grip','grip_local','ray','head'):
                poses=[r[name] for r in valid if r.get(name)]
                if not poses:continue
                rots=Rotation.from_quat([p['q'] for p in poses])
                info[name]={'position_span_mm':(np.ptp([p['p'] for p in poses],axis=0)*1000).round(3).tolist(),
                    'orientation_span_from_first_deg':round(float(np.max(np.rad2deg((rots*rots[0].inv()).magnitude()))),4),
                    'emulated_count':sum(p.get('emulated') is True for p in poses)}
            pair=[r for r in valid if r.get('ray') and r.get('grip_local')]
            if pair:
                grip=Rotation.from_quat([r['grip']['q'] for r in pair])
                ray=Rotation.from_quat([r['ray']['q'] for r in pair])
                local=Rotation.from_quat([r['grip_local']['q'] for r in pair])
                for name,rel in [('grip_to_ray',grip.inv()*ray),('reference_to_local',local*grip.inv())]:
                    info[name+'_variation_deg']=round(float(np.max(np.rad2deg((rel*rel[0].inv()).magnitude()))),5)
        print('BLOCK',json.dumps(info,ensure_ascii=False),flush=True)


async def capture(url,duration,output):
    expression=EXPRESSION.replace('DURATION_MS',str(int(duration*1000)))
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(url,timeout=5) as ws:
            await ws.send_json({'id':1,'method':'Runtime.evaluate','params':{
                'expression':expression,'returnByValue':True,'awaitPromise':True,
                'timeout':int((duration+5)*1000)}})
            print('OBSERVER_STARTED',flush=True)
            while True:
                message=await ws.receive_json(timeout=duration+10)
                if message.get('id')!=1:continue
                result=message.get('result',{})
                if 'exceptionDetails' in result or 'error' in message:
                    raise RuntimeError(message)
                data=result.get('result',{}).get('value',{})
                with output.open('x',encoding='utf-8') as f:
                    json.dump(data,f,ensure_ascii=False)
                summarize(data)
                print('RAW_DIAGNOSTIC',str(output),flush=True)
                break


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--websocket',required=True)
    parser.add_argument('--duration',type=float,default=60)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.duration<=120:parser.error('duration must be 1–120 s')
    asyncio.run(capture(args.websocket,args.duration,args.output))
