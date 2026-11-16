"""Run with .venv-unitree310/bin/python. Isolated random ports; never robot ports."""
import asyncio
import json
import socket
import subprocess
import sys
import time
import ssl
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import msgpack
import zmq


def event(sequence):
    return {'right': [1.,0,0,0, 0,1.,0,0, 0,0,1.,0, .2,1.,-.3,1.],
            'rightState': dict(trigger=False, aButton=False, bButton=False),
            'xarmMeta': dict(valid=True, frame_id='isolated-test', sequence=sequence,
                             sample_server_s=time.monotonic(),
                             buttons=dict(trigger=False, a=False, b=False),
                             release_counters=dict(trigger=0, a=0, b=0))}


async def main(tls=False):
    context = zmq.Context()
    pose, pause = context.socket(zmq.PULL), context.socket(zmq.PULL)
    pp = pose.bind_to_random_port('tcp://127.0.0.1')
    bp = pause.bind_to_random_port('tcp://127.0.0.1')
    with socket.socket() as guard:
        guard.bind(('127.0.0.1', 0))
        port = guard.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    temporary = tempfile.TemporaryDirectory(prefix='quest-lan-test-')
    extra = []
    connector = None
    if tls:
        cert, key = Path(temporary.name)/'cert.pem', Path(temporary.name)/'key.pem'
        subprocess.run(['openssl','req','-x509','-nodes','-newkey','rsa:2048','-days','1',
            '-subj','/CN=isolated test','-addext','subjectAltName=IP:127.0.0.1',
            '-keyout',str(key),'-out',str(cert)],check=True,capture_output=True)
        extra=['--cert',str(cert),'--key',str(key)]
        # Trust this test certificate ONLY; verify hostname, no ssl=False.
        connector=aiohttp.TCPConnector(ssl=ssl.create_default_context(cafile=str(cert)))
    process = subprocess.Popen([sys.executable, '-u', str(root/'quest_unitree_bridge.py'),
        '--port', str(port), '--pose-port', str(pp), '--pause-port', str(bp), *extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready = json.loads(await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), 10))
        origin = f'{"https" if tls else "http"}://127.0.0.1:{port}'
        jar = aiohttp.CookieJar(unsafe=True)
        async with aiohttp.ClientSession(cookie_jar=jar, connector=connector) as client:
            async with client.get(origin) as response:
                assert response.status == (200 if tls else 403)
                if tls:
                    assert '配对码' in await response.text()
                    assert response.headers['Referrer-Policy']=='same-origin'
            if tls:
                async with client.post(origin+'/unitree-reopen') as response:
                    assert response.status == 403
                try:
                    await client.ws_connect(origin+'/unitree-clock-ws',origin=origin)
                except aiohttp.WSServerHandshakeError as error:
                    assert error.status == 403
                else:
                    raise AssertionError('unpaired clock socket accepted')
                async with client.get(origin+'/unitree-clock') as response:
                    assert response.status == 403
                async with client.post(origin+'/pair',data={'code':'wrong'}) as response:
                    assert response.status == 403
                async with client.post(origin+'/pair', data={'code':ready['pairing_code']},
                                       headers={'Origin':'https://evil.test'}) as response:
                    assert response.status == 403
                async with client.post(origin+'/pair', data={'code':ready['pairing_code']},
                                       headers={'Origin':'null'}) as response:
                    assert response.status == 403  # do not fix this by trusting opaque origins
                async with client.post(origin+'/pair',data={'code':ready['pairing_code']},allow_redirects=False) as response:
                    assert response.status == 303
                    cookie=response.cookies['unitree_input']
                    assert cookie['secure'] and cookie['httponly'] and cookie['samesite']=='Strict'
                    ready['url']=origin+response.headers['Location']
                    assert 'wss' in ready['url'] and 'token' not in ready['url']
                async with client.get(origin+'/',headers={'Host':'evil.test'}) as response:
                    assert response.status == 403
            async with client.get(ready['url']) as response:
                assert response.status == 200
                assert 'unitree-quality.js' in await response.text()
            if tls:
                async with client.post(origin+'/unitree-reopen') as response:
                    assert response.status == 403  # browser cookie alone cannot request reload
            async with client.get(origin+'/unitree-clock') as response:
                assert abs((await response.json())['monotonic_s']-time.monotonic()) < .1
            async with client.ws_connect(origin+'/unitree-clock-ws',origin=origin) as clock_ws:
                before=time.monotonic()
                await clock_ws.send_json({'id':7})
                reply=await clock_ws.receive_json(timeout=1)
                assert reply['id']==7 and before <= reply['monotonic_s'] <= time.monotonic()
                await clock_ws.send_json({'id':True})
                assert (await clock_ws.receive(timeout=1)).type == aiohttp.WSMsgType.CLOSE
            async with client.get(origin+'/unitree-profiles/meta-quest-touch-plus/profile.json') as response:
                assert response.status == 200
                assert (await response.json())['profileId'] == 'meta-quest-touch-plus'
            async with client.get(origin+'/static/config.json') as response:
                assert response.status == 404
            async with client.get(origin+'/unitree-clock', headers={'Origin': 'https://other.test'}) as response:
                assert response.status == 403
            async with client.ws_connect(origin+'/', origin=origin) as ws:
                # Upstream TeleVuer really inserts its MotionControllers schema.
                message = await ws.receive(timeout=2)
                schema = msgpack.unpackb(message.data, raw=False)
                assert 'MotionControllers' in str(schema), schema
                await asyncio.sleep(.1)
                for index in range(1, 4):
                    value = event(sequence=index)
                    value['rightState']['bButton'] = value['xarmMeta']['buttons']['b'] = True
                    await ws.send_bytes(msgpack.packb({'etype': 'CONTROLLER_MOVE', 'key': 'motionControllers', 'value': value}))
                    await asyncio.sleep(.015)
                await asyncio.sleep(.03)
                results = []
                while pose.poll(0):
                    results.append(pose.recv_json())
                assert any(p.get('valid') and p['b_pressed'] and not p['trigger_pressed'] for p in results), results
                await asyncio.sleep(.25)
                messages = []
                while pause.poll(0):
                    messages.append(pause.recv())
                assert b'High' in messages and messages[-1] == b'Low', messages
                value = event(sequence=4)
                value['xarmMeta']['sample_server_s'] -= .5
                await ws.send_bytes(msgpack.packb({'etype': 'CONTROLLER_MOVE', 'key': 'motionControllers', 'value': value}))
                await asyncio.sleep(.03)
                assert pose.poll(0) and not pose.recv_json()['valid']
                process.terminate()
                await asyncio.wait_for(asyncio.to_thread(process.wait), 2.5)
        print(f'Unitree isolated {"HTTPS/pairing" if tls else "HTTP"}/auth/WebSocket/schema/pose/buttons/stale/watchdog: PASS')
    finally:
        process.terminate()
        process.wait(timeout=5)
        pose.close(linger=0); pause.close(linger=0); context.term()
        temporary.cleanup()


if __name__ == '__main__':
    asyncio.run(main(tls='--tls' in sys.argv))
