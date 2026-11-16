/* Safety metadata for Unitree TeleVuer/Vuer 0.0.60 CONTROLLER_MOVE.
 * Pose and button values remain Vuer's. No robot commands or pose filtering.
 */
'use strict';
(() => {
  // This script runs before Vuer's module. Its HTTPS default socket URL drops
  // the page's port when a paired user reopens the bare root URL.
  if(window.location && window.history){
    const page=new URL(window.location.href);
    const socket=(page.protocol==='https:'?'wss://':'ws://')+page.host+'/';
    if(page.searchParams.get('ws')!==socket){
      page.searchParams.set('ws',socket);
      window.history.replaceState(window.history.state,'',page.href);
    }
  }
  let offset=null, synced=-Infinity, session=null, reference=null;
  let syncing=false, failures=0, totalFailures=0, lastError='', selectedRtt=null;
  let clockTransport=null, wsFailures=0, httpFailures=0, wsLastError='', httpLastError='';
  let lastWsSample=-Infinity, nextHttp=0;
  const clockStarted=performance.now()/1000;
  let clockSamples=[];
  const CLOCK_MAX_AGE_S=3;
  const CLIENT_VERSION='realtime-input-v6';
  let reloading=false,lastXrAt=null;
  function reloadCapture(generation=null){
    if(reloading||!window.location?.replace)return;
    const page=new URL(window.location.href);
    if(generation){
      if(!/^[a-f0-9]{24}$/.test(generation)||page.searchParams.get('xarm_reopen')===generation)return;
      page.searchParams.set('xarm_reopen',generation);
    }
    page.searchParams.set('xarm_resume',String(Date.now()));
    reloading=true;offset=null;synced=-Infinity;clockSamples=[];previous={};
    // Release immersive presentation explicitly before navigating. Depending
    // on navigation alone can leave a suspended Quest session on screen.
    // A hung/rejected end must not permanently strand the recovery request.
    let navigated=false,timer;
    const navigate=()=>{
      if(navigated)return;
      navigated=true;
      if(timer!==undefined)clearTimeout(timer);
      window.location.replace(page.href);
    };
    if(!session?.end){navigate();return;}
    timer=setTimeout(navigate,1500);
    try{Promise.resolve(session.end()).then(navigate,navigate);}
    catch(_){navigate();}
  }
  function resumed(){
    if(lastXrAt!==null&&Date.now()-lastXrAt>5000)reloadCapture();
    else clockTick();
  }
  window.document?.addEventListener('visibilitychange',()=>{
    if(window.document.visibilityState==='visible')resumed();
  });
  window.addEventListener?.('pageshow',event=>{if(event.persisted)reloadCapture();});
  function xrVisibility(){if(session?.visibilityState==='visible')resumed();}
  const knownController=source=>!!source&&!source.hand&&source.targetRayMode==='tracked-pointer'&&
    source.profiles?.some(p=>p.startsWith('oculus-touch')||p.startsWith('meta-quest-touch'))&&
    source.gamepad?.mapping==='xr-standard'&&source.gamepad.buttons.length>=6;
  window.xarmUnitreeControllerSource=(xrSession,side)=>
    Array.from(xrSession?.inputSources||[]).find(source=>source.handedness===side&&knownController(source));
  let frameId='', sequence=0, previous={}, releases={trigger:0,a:0,b:0};
  let interruptions=0, lastInputError='', previousValid=null;
  window.xarmUnitreeInputFault=reason=>{interruptions++;lastInputError=reason;};
  function reset(){
    frameId='unitree-'+Date.now()+'-'+Math.random().toString(36).slice(2);
    sequence=0;previous={};releases={trigger:0,a:0,b:0};
    interruptions=0;lastInputError='';previousValid=null;
  }
  function selectClock(now){
    clockSamples=clockSamples.filter(sample=>now-sample.at<CLOCK_MAX_AGE_S);
    if(!clockSamples.length)return;
    const best=clockSamples.reduce((a,b)=>a.rtt<=b.rtt?a:b);
    offset=best.offset;synced=best.at;selectedRtt=best.rtt;clockTransport=best.transport;
  }
  async function sync(){
    if(syncing)return;
    syncing=true;
    try{
      const before=performance.now()/1000;
      const response=await fetch('/unitree-clock',{cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(300)});
      if(!response.ok)throw new Error('clock request failed');
      const value=await response.json(),after=performance.now()/1000;
      if(reloading)return;
      if(value.reopen_generation)reloadCapture(value.reopen_generation);
      if(reloading)return;
      if(!Number.isFinite(value.monotonic_s))throw new Error('invalid server clock');
      if(after-before>.1)throw new Error('clock RTT >100ms');
      clockSamples.push({at:after,rtt:after-before,offset:value.monotonic_s-(before+after)/2,transport:'http'});
      // Prefer the least delayed recent measurement; a Wi-Fi latency spike
      // must not move the clock estimate or invalidate an otherwise fresh pose.
      selectClock(after);
      failures=0;lastError='';httpLastError='';
    }catch(error){
      // Keep the previous estimate only within the original three-second
      // validity window. Failed attempts never renew that window.
      failures++;totalFailures++;httpFailures++;lastError=error.message;httpLastError=error.message;
    }finally{syncing=false;}
  }
  // A persistent clock channel avoids a new HTTP fetch on every sample.
  // Drive recovery from XR frames too: background timer throttling must not
  // leave a resumed session waiting for the next browser interval.
  let clockSocket=null, openingAt=0, nextConnect=0, nextPing=0, pending=null, pingId=0;
  const useClockSocket=typeof WebSocket==='function'&&!!window.location;
  function clockFailure(message){failures++;totalFailures++;wsFailures++;lastError=message;wsLastError=message;}
  function clockTick(){
    if(!useClockSocket)return;
    const now=performance.now()/1000;
    // Connection setup may be slow; it never extends a sample's 3-second life.
    if(clockSocket?.readyState===0&&now-openingAt>3){
      const old=clockSocket;clockSocket=null;old.close();pending=null;
      clockFailure('clock connection timeout');nextConnect=now+.25;
    }
    if(!clockSocket&&now>=nextConnect){
      const address=new URL('/unitree-clock-ws',window.location.href);
      address.protocol=address.protocol==='https:'?'wss:':'ws:';
      try{
      const ws=new WebSocket(address.href);clockSocket=ws;openingAt=now;
      ws.onopen=()=>{if(clockSocket===ws){nextPing=0;clockTick();}};
      ws.onclose=()=>{if(clockSocket===ws){clockSocket=null;pending=null;nextConnect=performance.now()/1000+.25;}};
      ws.onerror=()=>{if(clockSocket===ws){clockFailure('clock socket error');ws.close();}};
      ws.onmessage=event=>{
        if(clockSocket!==ws||!pending)return;
        let value;try{value=JSON.parse(event.data);}catch(_){return;}
        if(value.id!==pending.id)return; // Never accept a delayed old reply.
        const after=performance.now()/1000,before=pending.at;pending=null;
        if(value.reopen_generation)reloadCapture(value.reopen_generation);
        if(reloading)return;
        if(!Number.isFinite(value.monotonic_s)){clockFailure('invalid server clock');return;}
        if(after-before>.1){clockFailure('clock RTT >100ms');nextPing=after+.1;return;}
        clockSamples.push({at:after,rtt:after-before,offset:value.monotonic_s-(before+after)/2,transport:'websocket'});
        selectClock(after);lastWsSample=after;wsLastError='';failures=0;lastError='';nextPing=after+1;
      };
      }catch(error){
        clockSocket=null;pending=null;clockFailure('clock socket setup: '+error.message);nextConnect=now+1;
      }
    }
    if(pending&&now-pending.at>.3){
      pending=null;clockFailure('clock reply timeout');nextPing=now;
      // Discard an unresponsive connection instead of queuing new pings.
      const old=clockSocket;clockSocket=null;old?.close();nextConnect=now+.1;
    }
    if(clockSocket?.readyState===1&&!pending&&now>=nextPing){
      pending={id:++pingId,at:now};
      try{clockSocket.send(JSON.stringify({id:pending.id}));}
      catch(_){pending=null;clockSocket.close();clockFailure('clock send failed');}
    }
    // WebSocket support does not guarantee this endpoint is reachable. Use the
    // authenticated same-origin HTTP endpoint while WS is failing or stale.
    // Healthy WS suppresses fallback; both transports enforce the same limits.
    if(typeof fetch==='function'&&now-clockStarted>=.5&&now-lastWsSample>1.25&&now>=nextHttp&&!syncing){
      nextHttp=now+1;
      void sync();
    }
  }
  if(useClockSocket){clockTick();setInterval(clockTick,100);}
  else{sync();setInterval(sync,1000);}
  window.xarmUnitreeMeta=(frame,right,ref)=>{
    if(frame?.session!==session || ref!==reference){
      reference?.removeEventListener?.('reset',reset);
      session?.removeEventListener?.('visibilitychange',xrVisibility);
      if(frame?.session!==session)lastXrAt=null; // A newly entered XR session is already fresh.
      session=frame?.session;reference=ref;reset();
      ref?.addEventListener('reset',reset);
      session?.addEventListener?.('visibilitychange',xrVisibility);
    }
    if(lastXrAt!==null&&session?.visibilityState==='visible'&&Date.now()-lastXrAt>5000)reloadCapture();
    lastXrAt=Date.now();
    const now=performance.now()/1000;
    clockTick();
    selectClock(now);
    const result={frame_id:frameId,sequence:sequence++,valid:false,client_version:CLIENT_VERSION,clock:{
      transport:clockTransport||(useClockSocket?'websocket':'http'),
      age_ms:Number.isFinite(synced)?Math.max(0,now-synced)*1000:null,
      rtt_ms:selectedRtt===null?null:selectedRtt*1000,
      consecutive_failures:failures,total_failures:totalFailures,last_error:lastError,
      ws_failures:wsFailures,http_failures:httpFailures,ws_last_error:wsLastError,http_last_error:httpLastError}};
    try{
      if(reloading)throw new Error('休眠后正在重建采集页，请点 Pass-through');
      if(!frame||session?.visibilityState!=='visible')throw new Error('XR 非前台');
      if(!right||right.hand||right.handedness!=='right'||right.targetRayMode!=='tracked-pointer')throw new Error('等待实体右手柄');
      if(!Array.from(session.inputSources).includes(right))throw new Error('右手柄已断开');
      if(!right.profiles.some(p=>p.startsWith('oculus-touch')||p.startsWith('meta-quest-touch')))
        throw new Error('未知按键映射：'+Array.from(right.profiles).join(',').slice(0,160));
      const gamepad=right.gamepad;
      if(gamepad?.mapping!=='xr-standard'||gamepad.buttons.length<6)throw new Error('缺少独立按键');
      // Upstream Vuer uses targetRaySpace, not gripSpace. Check that exact pose.
      const pose=frame.getPose(right.targetRaySpace,ref);
      result.tracking={pose_available:!!pose,emulated_position:pose?.emulatedPosition??null,
        profiles:Array.from(right.profiles).join(',')};
      if(!pose)throw new Error('右手柄位置丢失：XR 未提供位姿');
      if(pose.emulatedPosition!==false)throw new Error('右手柄位置仅为推算：请让头显看见手柄');
      if(offset===null||now-synced>=CLOCK_MAX_AGE_S)throw new Error('等待电脑时钟同步');
      const buttons={trigger:gamepad.buttons[0].pressed,a:gamepad.buttons[4].pressed,b:gamepad.buttons[5].pressed};
      for(const key of Object.keys(buttons)){
        if(typeof buttons[key]!=='boolean')throw new Error('按键类型无效');
        if(previous[key]===true&&!buttons[key])releases[key]++;
      }
      previous=buttons;
      // An asymmetric clock round trip can put the midpoint estimate up to
      // RTT/2 in the future. Use the EARLIEST possible server time for age
      // checks: this never makes an old packet look newer. Keep the headset's
      // own monotonic clock separately for filter dt; clock resynchronization
      // must not reverse time or create artificial hand velocity.
      Object.assign(result,{valid:true,sample_server_s:now+offset-selectedRtt/2,
        sample_headset_s:now,buttons,release_counters:{...releases}});
    }catch(error){result.reason=error.message;previous={};}
    if(!result.valid&&previousValid!==false){interruptions++;lastInputError=result.reason;}
    previousValid=result.valid;
    result.input_interruptions=interruptions;result.last_input_error=lastInputError;
    result.transport=window.xarmUnitreeTransportStatus?.();
    return result;
  };
})();
