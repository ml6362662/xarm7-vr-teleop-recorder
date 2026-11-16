/* Runs production JS with a deterministic WebSocket and XR frame clock. */
globalThis.runUnitreeRecoveryChecks=async function(install){
  const check=(x,s)=>{if(!x)throw new Error(s);};
  let now=10,tick;
  const sockets=[];
  class Socket{
    constructor(url){this.url=url;this.readyState=0;this.sent=[];sockets.push(this);}
    open(){this.readyState=1;this.onopen();}
    send(data){this.sent.push(JSON.parse(data));}
    close(){this.readyState=3;this.onclose?.();}
    reply(id,server=now+50){this.onmessage({data:JSON.stringify({id,monotonic_s:server})});}
  }
  const window={location:{href:'https://10.72.115.86:8766/'}};
  install({window,WebSocket:Socket,URL,performance:{now:()=>now*1000},
    setInterval:f=>{tick=f;},fetch:()=>{throw new Error('HTTP clock must not be used');},
    AbortSignal:{timeout:()=>null},Date,Math});
  const right={handedness:'right',targetRayMode:'tracked-pointer',targetRaySpace:{},
    profiles:['meta-quest-touch-plus'],gamepad:{mapping:'xr-standard',buttons:Array.from({length:6},()=>({pressed:false}))}};
  const synthetic={...right,profiles:['meta-fixed-hand','generic-hand-select']};
  const session={visibilityState:'visible',inputSources:[synthetic,right]};
  let pose={emulatedPosition:false};
  const frame={session,getPose:()=>pose},reference={addEventListener(){}};
  const get=()=>window.xarmUnitreeMeta(frame,right,reference);
  check(window.xarmUnitreeControllerSource(session,'right')===right,'synthetic hand must not hide real controller');
  session.inputSources=[synthetic];
  check(!window.xarmUnitreeControllerSource(session,'right'),'synthetic hand cannot substitute for controller');
  const replacement={...right};session.inputSources=[replacement];
  check(window.xarmUnitreeControllerSource(session,'right')===replacement,'select current input source after replacement');
  session.inputSources=[right];
  check(sockets[0].url==='wss://10.72.115.86:8766/unitree-clock-ws','same-origin port preserved');
  sockets[0].open();sockets[0].reply(sockets[0].sent[0].id);
  check(get().valid,'first clock sample');
  right.gamepad.buttons[0].pressed=true;get();
  now=11.01;tick();const late=sockets[0].sent.at(-1).id;
  now=11.4;get(); // XR frames must expire a stalled ping even without timers.
  check(sockets[0].readyState===3,'stalled connection must close');
  now=11.51;get();check(sockets.length===2,'XR frame reconnects clock');
  sockets[1].open();const fresh=sockets[1].sent[0].id;
  sockets[0].reply(late,9999);check(get().sample_server_s<100,'old socket cannot corrupt clock');
  right.gamepad.buttons[0].pressed=false;
  check(get().release_counters.trigger===1,'trigger release survives clock reconnect');
  sockets[1].reply(fresh);check(get().valid,'clock recovers without browser reload');
  now=15;check(!get().valid,'expired clock must still stop control');
  now=15.2;sockets.at(-1).reply(sockets.at(-1).sent.at(-1).id);
  check(!get().valid,'delayed reply cannot renew expired clock');
  now=15.31;get();sockets.at(-1).reply(sockets.at(-1).sent.at(-1).id);
  check(get().valid,'recovery after timer suspension');
  pose={emulatedPosition:true};check(!get().valid&&get().reason.includes('仅为推算'),'real tracking loss cannot be bypassed');
  pose=null;check(!get().valid&&get().reason.includes('未提供位姿'),'missing pose diagnosed separately');
  pose={emulatedPosition:false};session.visibilityState='visible-blurred';check(!get().valid,'foreground still required');
  return 'current controller selection, clock timeout/reconnect/expiry, stale replies, release edges and tracking checks PASS';
};
