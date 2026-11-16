'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
let now=10000,reset;
const buttons=Array.from({length:6},()=>({pressed:false}));
const right={handedness:'right',targetRayMode:'tracked-pointer',targetRaySpace:{},
  profiles:['meta-quest-touch-plus'],gamepad:{mapping:'xr-standard',buttons}};
const session={visibilityState:'visible',inputSources:[right]};
const reference={addEventListener:(name,cb)=>{if(name==='reset')reset=cb;}};
let pose={emulatedPosition:false};
const frame={session,getPose:()=>pose};
const context=vm.createContext({window:{},performance:{now:()=>now},setInterval(){},
  AbortSignal,fetch:async()=>({ok:true,json:async()=>({monotonic_s:10})}),
  Date,Math,console});
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../quest_unitree_quality.js'),'utf8'),context);
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));
  const get=()=>context.window.xarmUnitreeMeta(frame,right,reference);
  let meta=get(),frameId=meta.frame_id;
  assert(meta.valid);
  for(let i=0;i<90;i++){
    now+=1000/90;buttons[5].pressed=i>=5&&i<15;buttons[0].pressed=i>=30&&i<50;
    meta=get();assert(meta.valid);
    assert.equal(meta.buttons.b,i>=5&&i<15);
    assert.equal(meta.buttons.trigger,i>=30&&i<50);
  }
  assert.equal(meta.release_counters.trigger,1);assert.equal(meta.release_counters.b,1);
  pose.emulatedPosition=true;assert(!get().valid);
  pose.emulatedPosition=false;assert(get().valid);
  session.visibilityState='visible-blurred';assert(!get().valid);
  session.visibilityState='visible';session.inputSources=[];assert(!get().valid);
  session.inputSources=[right];reset();assert.notEqual(get().frame_id,frameId);
  now+=4000;assert(!get().valid,'expired clock must fail closed');
  console.log('Unitree XR metadata: 90 frames/buttons/releases/emulation/visibility/disconnect/reset/clock PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});

require('./unitree_clock_checks.js');
runUnitreeClockChecks(environment=>vm.runInNewContext(
  fs.readFileSync(require('node:path').join(__dirname,'../quest_unitree_quality.js'),'utf8'),environment))
  .then(message=>console.log(message)).catch(error=>{console.error(error);process.exitCode=1;});
