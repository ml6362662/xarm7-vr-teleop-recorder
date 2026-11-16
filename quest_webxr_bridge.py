#!/usr/bin/env python3
"""Loopback/USB WebXR pose transport. No robot SDK, no motion commands."""

import argparse
import json
import math
import queue
import secrets
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import zmq
from quest_pose_stream import inspect_webxr_packet

ROOT = Path(__file__).resolve().parent


class InputLease:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = -1
        self.seen = -1e9
        self.latest = queue.Queue(maxsize=1)

    def accept(self, value):
        now = self.clock()
        frame, sequence = value.get('frame_id'), value.get('sequence')
        if not isinstance(frame,str) or not 1<=len(frame)<=100 or type(sequence) is not int or not 0<=sequence<=2147483647:
            raise ValueError('invalid session/sequence')
        if value.get('valid') is True:
            generated = value.get('sample_server_s')
            if type(generated) not in (int,float) or not math.isfinite(generated):
                raise ValueError('clock synchronization required')
            age = now-generated
            if not -.03<=age<.18:
                value.update(valid=False,reason='WebXR 网络样本过期或时钟未同步')
            value['sample_age_s'] = max(0,age)
        parsed = inspect_webxr_packet(json.dumps(value).encode())
        if value.get('valid') is not False and not parsed.get('valid'):
            raise ValueError(parsed['reason'])
        with self.lock:
            if frame != self.frame:
                if now-self.seen<.5:
                    raise ValueError('another XR session owns the input; stop it first')
                self.frame,self.sequence = frame,-1
            if sequence<=self.sequence:
                raise ValueError('out-of-order input')
            self.sequence,self.seen = sequence,now
            if value.get('ended') is True:
                self.seen = -1e9
            try:
                self.latest.get_nowait()
            except queue.Empty:
                pass
            self.latest.put_nowait((value,now))


def make_server(port, token, lease):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            self.connection.settimeout(2)

        def log_message(self,*args):
            pass

        def reply(self,status,data,content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            if status >= 400:
                # Error paths may not have consumed the request body.
                self.send_header('Connection','close')
                self.close_connection = True
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):
                pass

        def authorized(self):
            url = urlsplit(self.path)
            return parse_qs(url.query).get('token') == [token]

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path not in ('/','/quest_webxr.js','/clock') or not self.authorized():
                self.reply(403,b'{}');return
            if url.path=='/clock':
                self.reply(200,json.dumps({'monotonic_s':lease.clock()}).encode());return
            name = 'quest_webxr.html' if url.path=='/' else 'quest_webxr.js'
            kind = 'text/html; charset=utf-8' if url.path=='/' else 'application/javascript'
            self.reply(200,(ROOT/name).read_bytes(),kind)

        def do_POST(self):
            if urlsplit(self.path).path!='/pose' or not self.authorized():
                self.reply(403,b'{}');return
            origin = self.headers.get('Origin')
            actual_port=self.server.server_port
            if origin not in (None,f'http://127.0.0.1:{actual_port}',f'http://localhost:{actual_port}'):
                self.reply(403,b'{}');return
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384:
                    raise ValueError('invalid payload size')
                value = json.loads(self.rfile.read(size))
                lease.accept(value)
                self.reply(200,b'{"ok":true}')
            except (ValueError,TypeError,AttributeError,KeyError) as error:
                self.reply(400,json.dumps({'error':str(error)}).encode())
    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads = True
    return server


def run(port=8765,pose_port=8125,pause_port=8100):
    lease, token = InputLease(), secrets.token_urlsafe(24)
    server = make_server(port,token,lease)
    running = threading.Event();running.set()
    signal.signal(signal.SIGTERM,lambda *_:running.clear())
    signal.signal(signal.SIGINT,lambda *_:running.clear())
    threading.Thread(target=server.serve_forever,daemon=True).start()
    context = zmq.Context()
    pose, pause = context.socket(zmq.PUSH),context.socket(zmq.PUSH)
    for sock,output in ((pose,pose_port),(pause,pause_port)):
        sock.setsockopt(zmq.SNDTIMEO,5);sock.setsockopt(zmq.LINGER,0)
        sock.setsockopt(zmq.SNDHWM,3);sock.setsockopt(zmq.IMMEDIATE,1)
        sock.connect(f'tcp://127.0.0.1:{output}')
    print(json.dumps({'type':'webxr_ready','port':port,'url':f'http://127.0.0.1:{port}/?token={token}'}),flush=True)
    enabled,last_seen,next_pause = False,-1e9,0
    try:
        while running.is_set():
            try:
                value,received = lease.latest.get(timeout=.01)
                now = time.monotonic()
                value['sample_age_s'] = value.get('sample_age_s',0)+now-received
                if value['sample_age_s']>=.2:
                    value.update(valid=False,reason='WebXR 传输队列数据过期')
                enabled = value.get('valid') is True
                last_seen = now
                try:
                    pose.send_json(value)
                except zmq.Again:
                    enabled = False
            except queue.Empty:
                pass
            now = time.monotonic()
            if now>=next_pause:
                try:
                    pause.send(b'High' if enabled and now-last_seen<.2 else b'Low')
                except zmq.Again:
                    pass
                next_pause = now+.02
    finally:
        try:
            pause.send(b'Low')
        except zmq.Again:
            pass
        server.shutdown();server.server_close()
        pose.close();pause.close();context.term()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--pose-port',type=int,default=8125)
    parser.add_argument('--pause-port',type=int,default=8100)
    args=parser.parse_args()
    run(args.port,args.pose_port,args.pause_port)
