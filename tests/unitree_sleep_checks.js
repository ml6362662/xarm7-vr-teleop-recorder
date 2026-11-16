globalThis.runUnitreeSleepChecks=async function(install){
  const check=(ok,message)=>{if(!ok)throw new Error(message);};
  function harness(query=''){
    let now=10,wall=10000,tick;const events={},navigations=[],sockets=[],timers=new Map();let timerId=0;
    class Socket{
      constructor(){this.readyState=0;sockets.push(this);}
      send(raw){this.request=JSON.parse(raw);}
      close(){this.readyState=3;this.onclose?.();}
      open(){this.readyState=1;this.onopen();}
      reply(extra={}){this.onmessage({data:JSON.stringify({id:this.request.id,monotonic_s:now+50,...extra})});}
    }
    const location={href:'https://10.72.115.86:8766/'+query,replace:url=>navigations.push(url)};
    const document={visibilityState:'visible',addEventListener:(name,fn)=>{events[name]=fn;}};
    const window={location,document,addEventListener:(name,fn)=>{events[name]=fn;}};
    install({window,WebSocket:Socket,URL,performance:{now:()=>now*1000},setInterval:fn=>{tick=fn;},
      setTimeout:fn=>{timers.set(++timerId,fn);return timerId;},clearTimeout:id=>timers.delete(id),
      Date:{now:()=>wall},Math});
    const right={handedness:'right',targetRayMode:'tracked-pointer',targetRaySpace:{},profiles:['meta-quest-touch-plus'],
      gamepad:{mapping:'xr-standard',buttons:Array.from({length:6},()=>({pressed:false}))}};
    const session={visibilityState:'visible',inputSources:[right],addEventListener:(name,fn)=>{events['xr-'+name]=fn;}};
    const ref={addEventListener(){}};const frame={session,getPose:()=>({emulatedPosition:false})};
    return {events,navigations,sockets,document,session,timers,get:()=>window.xarmUnitreeMeta(frame,right,ref),
      newSession:()=>{frame.session={...session};},
      advance:s=>{now+=s;wall+=s*1000;},tick:()=>tick()};
  }
  const generation='0123456789abcdef01234567';
  {
    const h=harness();h.sockets[0].open();h.sockets[0].reply();check(h.get().valid,'initial frame valid');
    h.advance(1);h.events.visibilitychange();check(h.navigations.length===0,'short menu interaction must not reload');
    h.advance(10);h.events.visibilitychange();
    check(h.navigations.length===1,'wake from long sleep must reload');
    h.events.visibilitychange();check(h.navigations.length===1,'only one reload per wake');
    check(!h.get().valid,'old frames cannot continue while reloading');
  }
  {
    const h=harness();h.sockets[0].open();h.sockets[0].reply();h.get();
    h.advance(10);h.events['xr-visibilitychange']();
    check(h.navigations.length===1,'XR visibility resume must recover even if document event is absent');
  }
  {
    const h=harness();h.sockets[0].open();h.sockets[0].reply();h.get();
    h.advance(10);h.newSession();h.get();
    check(h.navigations.length===0,'a fresh user-entered XR session must not be reloaded again');
  }
  {
    const h=harness();h.sockets[0].open();h.sockets[0].reply({reopen_generation:generation});
    check(h.navigations.length===1,'desktop Wi-Fi request reloads paired page');
    check(new URL(h.navigations[0]).searchParams.get('xarm_reopen')===generation,'ack carried through navigation');
    const fresh=harness('?xarm_reopen='+generation);fresh.sockets[0].open();fresh.sockets[0].reply({reopen_generation:generation});
    check(fresh.navigations.length===0&&fresh.get().valid,'persistent request cannot cause a reload loop');
  }
  {
    const h=harness();h.events.pageshow({persisted:true});
    check(h.navigations.length===1,'bfcache restoration must not reuse suspended capture');
  }
  {
    const h=harness();h.sockets[0].open();h.sockets[0].reply();h.get();
    let ended=0,finish;
    h.session.end=()=>{ended++;return new Promise(resolve=>{finish=resolve;});};
    h.advance(10);h.events['xr-visibilitychange']();
    check(ended===1&&h.navigations.length===0,'must end XR before navigation');
    h.events.visibilitychange();check(ended===1,'repeated wake must not end twice');
    check(!h.get().valid,'pending XR end must invalidate control input');
    finish();await Promise.resolve();
    check(h.navigations.length===1&&h.timers.size===0,'navigate after XR end and clear fallback');
  }
  for(const outcome of ['reject','throw','hang']){
    const h=harness();h.sockets[0].open();h.sockets[0].reply();h.get();
    let finish;
    h.session.end=()=>{
      if(outcome==='throw')throw new Error('session already ended');
      if(outcome==='reject')return Promise.reject(new Error('session already ended'));
      return new Promise(resolve=>{finish=resolve;});
    };
    h.advance(10);h.events['xr-visibilitychange']();await Promise.resolve();
    if(outcome==='hang'){
      check(h.navigations.length===0,'hang must wait for bounded fallback');
      for(const fn of h.timers.values())fn();
      finish();await Promise.resolve();
    }
    check(h.navigations.length===1,'failed or hung XR end must recover exactly once: '+outcome);
  }
  return 'sleep/visibility/XR wake, stale frame rejection, Wi-Fi reload acknowledgment and bfcache recovery PASS';
};
