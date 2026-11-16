#!/usr/bin/env python3
"""Unitree TeleVuer input only, localhost/ADB; deliberately no robot SDK.

Vuer's upstream 30-Hz interval repeats cached matrices. A pinned, checked
in-memory client adaptation emits CONTROLLER_MOVE at the actual XR frame and
adds freshness/tracking metadata. Upstream pose/button extraction is unchanged.
"""
import argparse
import asyncio
import contextlib
import hashlib
import json
import secrets
import signal
import ssl
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

from quest_unitree_input import UnitreeInput
from quest_lan import lan_host, PairingGate, PAIR_PAGE

ROOT = Path(__file__).resolve().parent
CHUNK = 'assets/chunks/chunk-Dd3xtWba.js'
CHUNK_SHA256 = '03fa38712fe2d00b7e83648d5e9e08d6ff1202c2129037a2610718af3c9a6a11'
OLD = ('controllerState.rightPose=null,controllerState.right=null)});'
       'const he=reactExports.useCallback(()=>{!de||!ie||se({ts:Date.now(),etype:"CONTROLLER_MOVE",key:K,value:{...pe}})},[ie,de]);'
       'return useInterval(he,1e3/Z),null}')
NEW = ('controllerState.rightPose=null,controllerState.right=null);'
       'ie&&window.xarmUnitreeSend({...pe,xarmMeta:window.xarmUnitreeMeta(Ae,fe,le)})});return null}')


def patch_client(data):
    if hashlib.sha256(data).hexdigest() != CHUNK_SHA256:
        raise RuntimeError('Vuer 客户端版本/校验不匹配，禁止未经检查的输入')
    text = data.decode()
    if text.count(OLD) != 1:
        raise RuntimeError('Vuer 控制器接口不匹配')
    text = text.replace(OLD, NEW)
    # Resolve devices from THIS XR frame, not a render-hook closure which can
    # retain a removed device or select a synthetic hand gamepad first.
    frame_start = 'useFrame(function(ge,me,Ae){if(!de)return;const{transform:ye'
    if text.count(frame_start) != 1:
        raise RuntimeError('Vuer 每帧手柄选择接口不匹配')
    text = text.replace(frame_start, 'useFrame(function(ge,me,Ae){if(!de)return;'
        'const ce=window.xarmUnitreeControllerSource(Ae?.session,"left"),'
        'fe=window.xarmUnitreeControllerSource(Ae?.session,"right");const{transform:ye')
    # This deployment is controller-only. Upstream enables hand models too;
    # their lazy CDN loads can tear down the scene when hands replace a pad.
    hand_options = 'createXRStore({hand:!0,handTracking:!0,emulate:!1})'
    if text.count(hand_options) != 1:
        raise RuntimeError('Vuer XR 输入模式接口不匹配')
    text = text.replace(hand_options, 'createXRStore({hand:!1,handTracking:!1,emulate:!1})')
    # Teleoperation must show the real workspace. Do not offer the upstream
    # opaque VR entry beside passthrough, where it is easily selected by mistake.
    vr_button = 'jsxRuntimeExports.jsx("button",{className:buttonStyle$2,onClick:()=>K.enterVR(),children:"Virtual Reality"}),'
    ar_label = 'children:"Pass-through"'
    if text.count(vr_button) != 1 or text.count(ar_label) != 1:
        raise RuntimeError('Vuer 透视入口接口不匹配')
    text = text.replace(vr_button, '').replace(ar_label, 'children:"进入遥操（Pass-through）"')
    # Quest may not reach public CDNs. Serve pinned controller profiles/models
    # over the same authorized USB loopback connection as the client.
    return text.replace('"https://cdn.jsdelivr.net/npm/@webxr-input-profiles/assets@1.0/dist/profiles/"',
                        '(window.location.origin+"/unitree-profiles/")')


async def receive_controller_socket(ws, receive_value):
    """Consume/ack directly; no scene event/task queue and no recorder API."""
    from aiohttp import web
    async for message in ws:
        if message.type != web.WSMsgType.TEXT:
            continue
        try:
            payload = json.loads(message.data)
            identity, value = payload['id'], payload['value']
            if type(identity) is not int or not 0 <= identity <= 2147483647 or not isinstance(value, dict):
                raise ValueError('invalid controller request')
        except (ValueError, KeyError, TypeError):
            await ws.close(code=1008, message=b'invalid controller request')
            break
        receive_value(value)
        await ws.send_json({'id': identity})


async def serve(port, pose_port, pause_port, host='127.0.0.1', cert=None, key=None):
    import zmq
    from aiohttp import web
    from vuer import Vuer
    from vuer.base import websocket_handler
    from televuer import TeleVuer

    lan = host != '127.0.0.1' or bool(cert)
    if lan:
        if host != '127.0.0.1':
            host = lan_host(host)
        if not cert or not key:
            raise ValueError('LAN 必须使用 HTTPS 证书，不允许明文控制器输入')
    tls = None
    if cert or key:
        if not cert or not key:
            raise ValueError('证书和私钥必须一起提供')
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.minimum_version = ssl.TLSVersion.TLSv1_2
        tls.load_cert_chain(cert, key)
    token = secrets.token_urlsafe(24)
    reopen_generation = None
    clock_clients = set()
    pairing = PairingGate() if lan else None
    origin = f'{"https" if tls else "http"}://{host}:{port}'
    socket_url = f'{"wss" if tls else "ws"}://{host}:{port}/'
    app = Vuer(host=host, port=port, cert=None, key=None,
               queries={'grid': False}, queue_len=3)
    root = app.client_root.resolve()
    patched = patch_client((root / CHUNK).read_bytes())
    index = (root / 'index.html').read_text().replace('<head>',
        '<head><script src="/unitree-quality.js"></script><script src="/unitree-realtime.js"></script>')
    index = index.replace('<title>Vuer.AI</title>', '<title>Unitree · xArm 输入</title>')

    @web.middleware
    async def auth(request, handler):
        if request.headers.get('Origin') not in (None, origin):
            raise web.HTTPForbidden(text='来源校验失败。请重新打开电脑 GUI 显示的完整 HTTPS 根地址，不要刷新旧的 /pair 页面。')
        # Block Host-header rebinding as well as non-local origins.
        if request.host != f'{host}:{port}':
            raise web.HTTPForbidden()
        if lan and request.path == '/pair' and request.method == 'POST':
            return await handler(request)
        supplied = request.query.get('token') or request.cookies.get('unitree_input')
        if not supplied or not secrets.compare_digest(supplied, token):
            if lan and request.path == '/' and request.method == 'GET' and not request.headers.get('Upgrade'):
                # A navigation-mode form POST under no-referrer gets Origin:
                # null in Chromium, including a POST to this SAME origin.
                # Keep cross-site origins blocked; fix the form's policy.
                return web.Response(text=PAIR_PAGE, content_type='text/html',
                    headers={'Cache-Control':'no-store','Referrer-Policy':'same-origin','X-Frame-Options':'DENY',
                             'Content-Security-Policy':"frame-ancestors 'none'; form-action 'self'; base-uri 'none'"})
            raise web.HTTPForbidden()
        result = await handler(request)
        result.headers['Cache-Control'] = 'no-store'
        result.headers['Referrer-Policy'] = 'no-referrer'
        result.headers['X-Frame-Options'] = 'DENY'
        if request.query.get('token') == token and request.headers.get('Upgrade', '').lower() != 'websocket':
            result.set_cookie('unitree_input', token, httponly=True, samesite='Strict', secure=bool(tls))
        return result

    app.app.middlewares.append(auth)
    app.app._client_max_size = 65536

    async def pair(request):
        form = await request.post()
        result = pairing.verify(form.get('code'))
        if result != 200:
            return web.Response(status=result, text='配对码错误；失败过多请等待一分钟后重试。')
        response = web.HTTPSeeOther('/?' + urlencode({'ws':socket_url,'grid':'false'}))
        response.set_cookie('unitree_input',token,httponly=True,secure=True,samesite='Strict')
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    async def index_or_socket(request):
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            ws = web.WebSocketResponse(timeout=.25, max_msg_size=65536)
            await ws.prepare(request)
            try:
                await app.downlink(request, ws)
            except ConnectionResetError:
                pass
            finally:
                await ws.close()
            return ws
        return web.Response(text=index, content_type='text/html')

    async def assets(request):
        relative = 'assets/' + request.match_info['filename']
        path = (root / relative).resolve()
        if not path.is_relative_to(root / 'assets') or not path.is_file():
            raise web.HTTPNotFound()
        if relative == CHUNK:
            return web.Response(text=patched, content_type='application/javascript')
        return web.FileResponse(path)

    async def clock(request):
        return web.json_response({'monotonic_s': time.monotonic(),
                                  'reopen_generation': reopen_generation})

    async def clock_socket(request):
        ws = web.WebSocketResponse(max_msg_size=1024, heartbeat=10)
        await ws.prepare(request)
        clock_clients.add(ws)
        try:
            async for message in ws:
                if message.type != web.WSMsgType.TEXT:
                    continue
                try:
                    value = json.loads(message.data)
                    seq = value['id']
                    if type(seq) is not int or not 0 <= seq <= 2147483647:
                        raise ValueError('invalid clock sequence')
                except (ValueError, KeyError, TypeError):
                    await ws.close(code=1008, message=b'invalid clock request')
                    break
                await ws.send_json({'id': seq, 'monotonic_s': time.monotonic(),
                                    'reopen_generation': reopen_generation})
        finally:
            clock_clients.discard(ws)
        return ws

    async def reopen(request):
        nonlocal reopen_generation
        # Only the desktop session credential can request a reload, not a
        # cookie-only cross-site form. This endpoint never enables the robot.
        if not secrets.compare_digest(request.query.get('token', ''), token):
            raise web.HTTPForbidden()
        reopen_generation = secrets.token_hex(12)
        return web.json_response({'ok': True, 'clock_clients':len(clock_clients),
                                  'generation':reopen_generation})

    async def quality(request):
        return web.Response(text=(ROOT/'quest_unitree_quality.js').read_text(), content_type='application/javascript')

    async def realtime(request):
        return web.Response(text=(ROOT/'quest_unitree_realtime.js').read_text(), content_type='application/javascript')

    input_clients = set()

    async def controller_socket(request):
        ws = web.WebSocketResponse(max_msg_size=16384, heartbeat=10)
        await ws.prepare(request)
        input_clients.add(ws)
        try:
            await receive_controller_socket(ws, receive_value)
        finally:
            input_clients.discard(ws)
        return ws

    profiles = (ROOT/'third_party'/'webxr-input-assets'/'package'/'dist'/'profiles').resolve()
    if not (profiles/'profilesList.json').is_file():
        raise RuntimeError('缺少本地手柄 profiles 资源，禁止依赖头显外网下载')

    async def profile_assets(request):
        path = (profiles/request.match_info['filename']).resolve()
        if not path.is_relative_to(profiles) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    # Do not expose Vuer's /static (workspace files) or /relay endpoints.
    app.app.router.add_get('/', index_or_socket)
    if lan:
        app.app.router.add_post('/pair', pair)
        app.app.router.add_post('/unitree-reopen', reopen)
    app.app.router.add_get('/assets/{filename:.*}', assets)
    app.app.router.add_get('/unitree-clock', clock)
    app.app.router.add_get('/unitree-clock-ws', clock_socket)
    app.app.router.add_get('/unitree-quality.js', quality)
    app.app.router.add_get('/unitree-realtime.js', realtime)
    app.app.router.add_get('/unitree-controller-ws', controller_socket)
    app.app.router.add_get('/unitree-profiles/{filename:.*}', profile_assets)
    upstream = SimpleNamespace(use_hand_tracking=False, display_fps=60.)

    @app.spawn(start=False)
    async def session_start(session):
        await TeleVuer.main_pass_through(upstream, session)

    adapter = UnitreeInput()
    latest = None
    status = dict(backend='unitree', valid=False, reason='等待头显进入 pass-through', packets=0)

    def receive_value(value):
        nonlocal latest
        # Decode each event before replacing the latest slot: release counts
        # from the client survive dropped transport/render frames.
        latest = adapter.decode(value)

    @app.add_handler('CONTROLLER_MOVE')
    async def controller(event, session):
        # Older browser compatibility only; never mix it into the new uplink.
        if not input_clients:
            receive_value(event.value)

    context = zmq.Context()
    pose, pause = context.socket(zmq.PUSH), context.socket(zmq.PUSH)
    for sock, output in ((pose, pose_port), (pause, pause_port)):
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.setsockopt(zmq.IMMEDIATE, 1)
        sock.connect(f'tcp://127.0.0.1:{output}')
    done = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, done.set)
    # A connected XR browser must not keep an old producer alive across a GUI
    # restart (the GUI waits at most three seconds for its input child).
    runner = web.AppRunner(app.app, shutdown_timeout=.5)
    await runner.setup()
    try:
        await web.TCPSite(runner, host, port, ssl_context=tls).start()
        url = origin + '/' if lan else origin + '/?' + urlencode({'token': token, 'ws':socket_url, 'grid':'false'})
        print(json.dumps({'type': 'unitree_ready', 'port':port, 'url':url,
                         'launch_url':origin + '/?' + urlencode({'token':token,'ws':socket_url,'grid':'false'}),
                         'transport':'lan' if lan else 'usb', 'pairing_code':pairing.code if pairing else None}), flush=True)
        # Upstream debug prints must not fill the GUI's unread stdout pipe.
        enabled, seen, next_pause, next_status = False, -1e9, 0., 0.
        while not done.is_set():
            now = time.monotonic()
            if latest is not None:
                value, latest = latest, None
                enabled = value.get('valid') is True
                if enabled and now-value['sample_server_s'] >= .2:
                    value.update(valid=False, reason='宇树输入排队过期')
                    enabled = False
                seen = now
                try:
                    pose.send_json(value, zmq.NOBLOCK)
                except zmq.Again:
                    enabled = False
                status.update(value)
                if enabled:
                    status['reason'] = '宇树有效控制器输入'
                status['packets'] += 1
            if now >= next_pause:
                with contextlib.suppress(zmq.Again):
                    pause.send(b'High' if enabled and now-seen < .2 else b'Low', zmq.NOBLOCK)
                next_pause = now + .02
            if now >= next_status:
                status.update(timestamp=time.time(), live=enabled and now-seen < .2,
                              connections=len(app.ws), diagnostics=adapter.diagnostics())
                status['connection_status'] = ('disconnected' if not app.ws else
                    'no_recent_frames' if now-seen >= .2 else 'live' if enabled else 'invalid_input')
                status['clock_connections'] = len(clock_clients)
                status['controller_connections'] = len(input_clients)
                if pose_port == 8125 and pause_port == 8100:
                    (ROOT/'.state'/'unitree_input_status.json').write_text(json.dumps(status, ensure_ascii=False))
                next_status = now + .5
            await asyncio.sleep(.002)
    finally:
        with contextlib.suppress(zmq.Again):
            pause.send(b'Low', zmq.NOBLOCK)
        await runner.cleanup()
        pose.close(); pause.close(); context.term()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--pose-port', type=int, default=8125)
    parser.add_argument('--pause-port', type=int, default=8100)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--cert', type=Path)
    parser.add_argument('--key', type=Path)
    args = parser.parse_args()
    asyncio.run(serve(args.port, args.pose_port, args.pause_port, args.host, args.cert, args.key))
