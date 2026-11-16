'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const read=name=>fs.readFileSync(path.join(__dirname,'../'+name),'utf8');
(async()=>{
  let now=10,pose={emulatedPosition:false};
  const window={location:{href:'https://127.0.0.1:8766/'}};
  // Initialize real metadata through the HTTP clock path, then install uplink.
  vm.runInNewContext(read('quest_unitree_quality.js'),{window,performance:{now:()=>now*1000},
    setInterval(){},fetch:async()=>({ok:true,json:async()=>({monotonic_s:60})}),AbortSignal,Date,Math});
  for(let i=0;i<10;i++)await Promise.resolve();
  const sockets=[];
  class Socket{
    constructor(url){this.url=url;this.readyState=0;this.bufferedAmount=0;this.sent=[];sockets.push(this);}
    send(value){this.sent.push(JSON.parse(value));}
    close(){this.readyState=3;this.onclose?.();}
    ack(id){this.onmessage({data:JSON.stringify({id})});}
  }
  vm.runInNewContext(read('quest_unitree_realtime.js'),{window,performance:{now:()=>now*1000},WebSocket:Socket,URL,JSON,Map});
  const buttons=Array.from({length:6},()=>({pressed:false}));
  const right={handedness:'right',targetRayMode:'tracked-pointer',targetRaySpace:{},profiles:['meta-quest-touch-plus'],gamepad:{mapping:'xr-standard',buttons}};
  const frame={session:{visibilityState:'visible',inputSources:[right]},getPose:()=>pose};
  const ref={addEventListener(){}};
  const emit=()=>{
    now+=.001;
    return window.xarmUnitreeSend({right:[now],rightState:{trigger:buttons[0].pressed},
      xarmMeta:window.xarmUnitreeMeta(frame,right,ref)});
  };
  assert(!emit());assert.equal(sockets[0].url,'wss://127.0.0.1:8766/unitree-controller-ws');
  sockets[0].readyState=1;
  assert(emit());buttons[0].pressed=true;assert(emit());
  // Backpressure: never queue a third old pose, while still observing every
  // XR frame for short trigger releases/tracking interruptions.
  for(let i=0;i<50;i++)assert(!emit());
  buttons[0].pressed=false;assert(!emit());buttons[0].pressed=true;
  pose={emulatedPosition:true};assert(!emit());pose={emulatedPosition:false};
  sockets[0].ack(sockets[0].sent[0].id);
  assert.equal(sockets[0].sent.length,2,'ack alone must not replay a cached pose');
  assert(emit());const latest=sockets[0].sent.at(-1);
  assert.equal(latest.value.right[0],now);
  assert.equal(latest.value.xarmMeta.release_counters.trigger,1);
  assert.equal(latest.value.xarmMeta.input_interruptions,1);
  assert(latest.value.xarmMeta.last_input_error.includes('仅为推算'));
  assert.equal(window.xarmUnitreeTransportStatus().in_flight,2);
  now+=.2;assert(!emit());assert.equal(sockets[0].readyState,3);
  now+=.11;assert(!emit());assert.equal(sockets.length,2);
  sockets[1].readyState=1;assert(emit());
  sockets[0].ack(latest.id);
  assert.equal(window.xarmUnitreeTransportStatus().in_flight,1,'old connection ack ignored');
  assert(sockets[1].sent[0].value.xarmMeta.input_interruptions>=2);
  sockets[1].bufferedAmount=500;assert(!emit());
  assert.equal(sockets[1].sent.length,1);
  console.log('dedicated uplink: bounded backlog, no replay, timeout/reconnect, release and tracking-fault counters PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});
