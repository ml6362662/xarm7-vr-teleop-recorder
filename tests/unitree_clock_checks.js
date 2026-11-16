/* Portable deterministic checks; install runs the production metadata script
 * inside an isolated environment. Used by test_unitree_client.cjs. */
globalThis.runUnitreeClockChecks = async function(install) {
  const check=(condition,message)=>{if(!condition)throw new Error(message);};
  async function harness() {
    let now=10, tick, request=async()=>({ok:true,json:async()=>({monotonic_s:now+50})});
    const window={};
    const buttons=Array.from({length:6},()=>({pressed:false}));
    const right={handedness:'right',targetRayMode:'tracked-pointer',targetRaySpace:{},
      profiles:['meta-quest-touch-plus'],gamepad:{mapping:'xr-standard',buttons}};
    const session={visibilityState:'visible',inputSources:[right]};
    const frame={session,getPose:()=>({emulatedPosition:false})};
    const reference={addEventListener(){}};
    install({window,performance:{now:()=>now*1000},setInterval:fn=>{tick=fn;},
      fetch:(...args)=>request(...args),AbortSignal:{timeout:()=>null},Date,Math});
    // Flush the initial async fetch/json sequence.
    for(let i=0;i<8;i++)await Promise.resolve();
    return {get:()=>window.xarmUnitreeMeta(frame,right,reference),right,session,buttons,
      time:value=>{now=value;},tick:()=>tick(),request:fn=>{request=fn;}};
  }
  {
    const h=await harness();check(h.get().valid,'initial clock must sync');
    h.buttons[0].pressed=true;h.get();
    h.time(11);h.request(async()=>{throw new Error('temporary network timeout');});
    await h.tick();h.buttons[0].pressed=false;
    let m=h.get();check(m.valid,'one failed clock request must retain a fresh estimate');
    check(m.release_counters.trigger===1,'release edge must survive a transient sync failure');
    check(m.clock.consecutive_failures===1,'sync failure must be diagnosed');
    h.time(12);await h.tick();check(h.get().valid,'second failure inside original lifetime');
    h.time(13);m=h.get();check(!m.valid&&m.reason==='等待电脑时钟同步','failed attempts must not extend clock lifetime');
    h.request(async()=>({ok:true,json:async()=>({monotonic_s:63})}));await h.tick();
    check(h.get().valid,'successful sync must recover after expiry');
  }
  {
    const h=await harness();h.time(11);
    h.request(async()=>{h.time(11.15);return {ok:true,json:async()=>({monotonic_s:61.10})};});
    await h.tick();let m=h.get();
    check(m.valid&&Math.abs(m.sample_server_s-61.15)<1e-8,'slow RTT must not poison prior offset');
    check(m.clock.last_error==='clock RTT >100ms','slow RTT reason');
    h.time(12);h.request(async()=>({ok:true,json:async()=>({monotonic_s:NaN})}));await h.tick();
    check(h.get().valid,'malformed sync must not poison valid previous sample');
    h.time(13.01);check(!h.get().valid,'slow/malformed samples cannot renew expiry');
  }
  {
    const h=await harness();h.time(11);
    h.request(async()=>{h.time(11.08);return {ok:true,json:async()=>({monotonic_s:61.08})};});
    await h.tick();let m=h.get();
    check(Math.abs(m.sample_server_s-61.08)<1e-8,'prefer lower-RTT estimate over jittery midpoint');
    h.time(13.01);check(h.get().valid,'select a still-fresh backup sample as best sample expires');
    h.time(14.09);check(!h.get().valid,'all expired samples must stop input');
  }
  {
    const h=await harness();h.session.visibilityState='visible-blurred';
    check(!h.get().valid,'clock changes must not bypass XR foreground check');
    h.session.visibilityState='visible';h.right.profiles=['generic-trigger-squeeze-thumbstick'];
    const m=h.get();check(!m.valid&&m.reason.includes('generic-trigger-squeeze-thumbstick'),'unknown profile stays blocked with exact identity');
  }
  return 'clock jitter, failure expiry, recovery, offset selection, trigger release, visibility and profile checks PASS';
};
