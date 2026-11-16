"""Receive shared Quest input without creating a second port owner."""
import socket,sys,json,signal,time
import zmq

def listening(port):
 s=socket.socket();s.settimeout(.2)
 try:return s.connect_ex(('127.0.0.1',port))==0
 finally:s.close()

def main():
 ports=[listening(p) for p in (8100,8125,8130)]
 if not any(ports):
  from quest_pose_stream import receive
  return receive()
 if not all(ports):
  print(json.dumps({'type':'error','message':'输入服务端口状态不完整，请检查8100/8125/8130的占用进程'},ensure_ascii=False),flush=True);return 1
 context=zmq.Context();sub=context.socket(zmq.SUB);sub.setsockopt(zmq.SUBSCRIBE,b'');sub.setsockopt(zmq.RCVHWM,5);sub.connect('tcp://127.0.0.1:8130')
 running=True
 def stop(*_):
  nonlocal running
  running=False
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  while running:
   if sub.poll(200):
    item=sub.recv_json()
    if isinstance(item,dict) and item.get('type')=='snapshot':print(json.dumps(item,ensure_ascii=False),flush=True)
 finally:sub.close(0);context.term()
 return 0
if __name__=='__main__':sys.exit(main())
