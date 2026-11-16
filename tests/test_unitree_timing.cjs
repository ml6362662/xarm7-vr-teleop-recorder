'use strict';
// Reproduce asymmetric clock RTT and offset replacement using production JS.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const code=fs.readFileSync(path.join(__dirname,'../quest_unitree_quality.js'),'utf8');
const flush=async()=>{for(let i=0;i<10;i++)await Promise.resolve();};
(async()=>{
  let now=10,tick,finish;
  const buttons=Array.from({length:6},()=>({pressed:false}));
  const right={handedness:'right',targetRayMode:'tracked-pointer',targetRaySpace:{},
    profiles:['meta-quest-touch-plus'],gamepad:{mapping:'xr-standard',buttons}};
  const session={visibilityState:'visible',inputSources:[right]};
  const frame={session,getPose:()=>({emulatedPosition:false})},ref={addEventListener(){}};
  const window={};
  vm.runInNewContext(code,{window,performance:{now:()=>now*1000},setInterval:f=>{tick=f;},
    fetch:()=>new Promise(resolve=>{finish=resolve;}),AbortSignal:{timeout:()=>null},Date,Math});
  const get=()=>window.xarmUnitreeMeta(frame,right,ref);
  const reply=async(server)=>{finish({ok:true,json:async()=>({monotonic_s:server})});await flush();};
  // 78 ms outbound, 2 ms return. Old midpoint estimate was 38 ms in future
  // and violated the existing -30 ms gate despite a fresh, tracked pose.
  now=10.08;await reply(60.078);
  let first=get();assert(first.valid);
  const oldMidpoint=now+60.078-(10+10.08)/2;
  assert(now+50-oldMidpoint < -.03,'reproduction must fail the old future gate');
  assert(Math.abs(first.sample_server_s-60.078)<1e-8);
  assert(now+50-first.sample_server_s>=0);
  assert.equal(first.sample_headset_s,now);
  // Better RTT but opposite asymmetry replaces clock estimate mid-movement.
  buttons[0].pressed=true;
  now=11;tick();now=11.059;
  const before=get();
  now=11.06;await reply(61.);
  const after=get();
  assert(after.valid);
  assert(after.sample_server_s<before.sample_server_s,'server clock correction reproduces the filter regression');
  assert(after.sample_headset_s>before.sample_headset_s,'filter clock must stay monotonic');
  assert(Math.abs(after.sample_headset_s-before.sample_headset_s-.001)<1e-8);
  assert(now+50-after.sample_server_s<.18);
  // Conservatively estimated age never hides a delayed packet.
  assert(now+50+.19-after.sample_server_s>=.18);
  buttons[0].pressed=false;
  assert.equal(get().release_counters.trigger,1);
  now=14.07;
  assert(!get().valid,'source time must not bypass expired synchronization');
  console.log('asymmetric RTT, backwards host correction, monotonic filter clock, release and expiry PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});
