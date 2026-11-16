'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const production=fs.readFileSync(path.join(__dirname,'../quest_unitree_quality.js'),'utf8');
const flush=async()=>{for(let i=0;i<10;i++)await Promise.resolve();};
function harness({throwSocket=false}={}){
  let now=10,tick,request=async()=>({ok:true,json:async()=>({monotonic_s:now+50})});
  const sockets=[],requests=[],navigations=[];
  class Socket{
    constructor(url){if(throwSocket)throw new Error('blocked socket');this.url=url;this.readyState=0;sockets.push(this);}
    send(raw){this.request=JSON.parse(raw);}
    close(){this.readyState=3;this.onclose?.();}
    open(){this.readyState=1;this.onopen();}
    reply(){this.onmessage({data:JSON.stringify({id:this.request.id,monotonic_s:now+50})});}
  }
  const window={location:{href:'https://10.72.115.86:8766/',replace:url=>navigations.push(url)}};
  const env={window,WebSocket:Socket,URL,performance:{now:()=>now*1000},setInterval:fn=>{tick=fn;},
    fetch:(url,options)=>{requests.push({url,options});return request();},AbortSignal:{timeout:ms=>({timeout:ms})},Date,Math};
  vm.runInNewContext(production,env);
  const buttons=Array.from({length:6},()=>({pressed:false}));
  const right={handedness:'right',targetRayMode:'tracked-pointer',targetRaySpace:{},profiles:['meta-quest-touch-plus'],gamepad:{mapping:'xr-standard',buttons}};
  const session={visibilityState:'visible',inputSources:[right]};
  let pose={emulatedPosition:false};
  const frame={session,getPose:()=>pose},ref={addEventListener(){}};
  return {sockets,requests,navigations,buttons,session,get:()=>window.xarmUnitreeMeta(frame,right,ref),
    time:t=>{now=t;},tick:()=>tick(),request:fn=>{request=fn;},pose:p=>{pose=p;}};
}
(async()=>{
  // Reproduce the real fault: tracking and main input live, clock WS never opens.
  for(const throwSocket of [false,true]){
    const h=harness({throwSocket});assert(!h.get().valid);assert.equal(h.requests.length,0);
    h.time(10.6);h.tick();await flush();
    assert(h.get().valid);assert.equal(h.get().clock.transport,'http');
    assert.equal(h.requests[0].url,'/unitree-clock');
    assert.equal(h.requests[0].options.credentials,'same-origin');
    assert.equal(h.requests[0].options.cache,'no-store');
    assert.equal(h.requests[0].options.signal.timeout,300);
    h.buttons[0].pressed=true;h.get();h.buttons[0].pressed=false;
    assert.equal(h.get().release_counters.trigger,1);
    h.time(13.1);h.tick();await flush();assert(h.get().valid);
    assert(h.get().clock.ws_failures>0);assert(h.get().clock.ws_last_error);
    h.pose({emulatedPosition:true});assert(!h.get().valid);
    h.pose({emulatedPosition:false});h.session.visibilityState='visible-blurred';assert(!h.get().valid);
  }
  {
    const h=harness();h.time(10.6);h.tick();await flush();assert(h.get().valid);
    // Healthy WS resumes without continuing HTTP polling.
    h.time(11);h.sockets[0].open();h.time(11.01);h.sockets[0].reply();
    const count=h.requests.length;
    h.time(12.02);h.tick();h.sockets[0].reply();
    h.time(13.03);h.tick();h.sockets[0].reply();
    h.time(13.7);h.get();await flush();
    assert.equal(h.requests.length,count);assert.equal(h.get().clock.transport,'websocket');
  }
  for(const bad of ['slow','malformed','unauthorized','failed']){
    const h=harness();h.time(10.6);h.tick();await flush();assert(h.get().valid);
    h.request(async()=>{
      if(bad==='failed')throw new Error('network timeout');
      if(bad==='slow')h.time(11.75);
      return {ok:bad!=='unauthorized',json:async()=>({monotonic_s:bad==='malformed'?NaN:61.6})};
    });
    h.time(11.6);h.tick();await flush();assert(h.get().valid);
    h.time(13.61);
    assert(!h.get().valid,'rejected fallback must not extend the last valid clock: '+bad);
    assert(h.get().clock.http_failures>0);
  }
  {
    const h=harness();let finish;
    h.request(()=>new Promise(resolve=>{finish=resolve;}));
    h.time(10.6);h.tick();h.get();h.get();h.time(12);h.tick();
    assert.equal(h.requests.length,1,'only one HTTP request may be in flight');
    finish({ok:true,json:async()=>({monotonic_s:62})});await flush();
    assert(!h.get().valid,'late HTTP reply cannot authorize input');
  }
  {
    const h=harness();
    h.request(async()=>({ok:true,json:async()=>({monotonic_s:60.6,reopen_generation:'0123456789abcdef01234567'})}));
    h.time(10.6);h.tick();await flush();
    assert.equal(h.navigations.length,1,'desktop reopen must also work over fallback');
    assert(!h.get().valid,'reloading never enables input');
  }
  // Run existing websocket/sleep checks against the same production client.
  require('./unitree_recovery_checks.js');require('./unitree_sleep_checks.js');
  // Browser checks use one realm; keep Promise identity when running in Node VM.
  const install=environment=>vm.runInNewContext(production,{Promise,...environment});
  console.log(await runUnitreeRecoveryChecks(install));
  console.log(await runUnitreeSleepChecks(install));
  console.log('HTTP fallback, WS recovery, bounded requests, expiry, auth/RTT rejection and reload checks PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});
