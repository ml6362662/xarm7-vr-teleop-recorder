/* Browser sampling regression, entirely mocked: no headset/robot/network. */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const posts=[];
let now=10000;
const buttonStates=Array.from({length:6},()=>({pressed:false}));
const elements=new Map();
const context=vm.createContext({
  document:{getElementById:id=>{if(!elements.has(id))elements.set(id,{});return elements.get(id);}},
  window:{isSecureContext:false,addEventListener(){}},navigator:{},
  performance:{now:()=>now},AbortSignal,encodeURIComponent,Float32Array,Math,Date,token:'test',
  fetch:async(url,options)=>{posts.push(JSON.parse(options.body));return {ok:true};},
  testSession:{inputSources:[{handedness:'right',gripSpace:{},profiles:['meta-quest-touch-plus'],
    gamepad:{mapping:'xr-standard',buttons:buttonStates}}],visibilityState:'visible',requestAnimationFrame(){}},
  testFrame:{getPose:()=>({transform:{position:{x:.2,y:1,z:-.3},orientation:{w:1,x:0,y:0,z:0}}}),getViewerPose:()=>null},
});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../quest_webxr.js'),'utf8'),context);
vm.runInContext("session=testSession;reference={};clockOffset=0;frameId='test-session';",context);
(async()=>{
  for(let frame=1;frame<=90;frame++){
    now=10000+frame*1000/90;
    buttonStates[5].pressed=frame>=10&&frame<20;
    buttonStates[0].pressed=frame>=30&&frame<40;
    vm.runInContext('onFrame(performance.now(),testFrame)',context);
    await new Promise(resolve=>setImmediate(resolve));
  }
  assert.equal(posts.length,90,'90-Hz input must not become 45-Hz input');
  assert(posts.every(p=>p.valid));
  assert(posts.some(p=>p.b_pressed&&!p.trigger_pressed));
  assert(posts.some(p=>!p.b_pressed&&p.trigger_pressed));
  assert.deepEqual(posts.at(-1).release_counters,{trigger:1,a:0,b:1});
  assert.deepEqual(posts.at(-1).position_m,[.2,1,.3]);
  console.log('WebXR client: 90/90 frames; independent B/trigger and releases passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
