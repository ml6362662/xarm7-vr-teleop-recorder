/* Dedicated controller uplink. No recording state, scene messages or replay queue. */
'use strict';
(() => {
  let socket=null, openingAt=0, nextConnect=0, sequence=0;
  const pending=new Map();
  // Throughput ceiling is roughly MAX_IN_FLIGHT / ack round trip. Two made a
  // ~80 ms Wi-Fi ack path cap out near 25 Hz; four still keeps only fresh
  // frames in flight and the receiver keeps its 180 ms age gate.
  const MAX_IN_FLIGHT=4;
  let sent=0, skipped=0, reconnects=0, lastError='';
  let skipPending=0, skipBuffer=0, skipClosed=0;
  function disconnect(ws, reason){
    if(socket!==ws)return;
    socket=null;pending.clear();nextConnect=performance.now()/1000+.1;
    lastError=reason;reconnects++;
    window.xarmUnitreeInputFault?.(reason);
    ws.close();
  }
  window.xarmUnitreeSend=value=>{
    const now=performance.now()/1000;
    if(socket?.readyState===0&&now-openingAt>3)disconnect(socket,'遥操输入连接超时');
    if(pending.size&&now-pending.values().next().value>=.18)
      disconnect(socket,'遥操输入确认超时');
    if(!socket&&now>=nextConnect){
      const url=new URL('/unitree-controller-ws',window.location.href);
      url.protocol=url.protocol==='https:'?'wss:':'ws:';
      try{
        const ws=new WebSocket(url.href);socket=ws;openingAt=now;
        ws.onmessage=event=>{
          if(socket!==ws)return;
          let ack;try{ack=JSON.parse(event.data);}catch(_){return;}
          if(pending.delete(ack.id))lastError='';
        };
        ws.onclose=()=>disconnect(ws,'遥操输入连接中断');
        ws.onerror=()=>disconnect(ws,'遥操输入连接错误');
      }catch(_){nextConnect=now+1;lastError='遥操输入连接建立失败';}
    }
    // Bound the uplink independently of XR frame rate or scene traffic and
    // never queue poses for later replay. A small send-buffer allowance keeps
    // a single unflushed frame from blocking the next one.
    if(socket?.readyState!==1){skipClosed++;skipped++;return false;}
    if(pending.size>=MAX_IN_FLIGHT){skipPending++;skipped++;return false;}
    if(socket.bufferedAmount>2048){skipBuffer++;skipped++;return false;}
    const id=++sequence;
    const payload={id,value:{right:value.right,rightState:value.rightState,xarmMeta:value.xarmMeta}};
    pending.set(id,now);
    try{socket.send(JSON.stringify(payload));sent++;return true;}
    catch(_){disconnect(socket,'遥操输入发送失败');return false;}
  };
  window.xarmUnitreeTransportStatus=()=>({kind:'dedicated-ack-v1',in_flight:pending.size,
    sent,skipped,reconnects,skip_pending:skipPending,skip_buffer:skipBuffer,skip_closed:skipClosed,
    last_error:lastError});
})();
