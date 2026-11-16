"""Real Chrome form submission on isolated HTTPS/robot-free random ports.

Run with .venv-unitree310/bin/python. Never uses the user's Chrome profile.
Only the ephemeral test certificate's SPKI is trusted for this test browser.
"""
import asyncio
import base64
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


async def main(expect_blocked=False):
    root = Path(__file__).resolve().parents[1]
    processes = []
    with tempfile.TemporaryDirectory(prefix='quest-browser-pairing-') as directory:
        directory = Path(directory)
        cert, key = directory/'cert.pem', directory/'key.pem'
        subprocess.run(['openssl','req','-x509','-nodes','-newkey','rsa:2048','-days','1',
            '-subj','/CN=isolated browser test','-addext','subjectAltName=IP:127.0.0.1',
            '-keyout',str(key),'-out',str(cert)], check=True,capture_output=True)
        pem=subprocess.run(['openssl','x509','-in',str(cert),'-pubkey','-noout'],check=True,capture_output=True).stdout
        der=subprocess.run(['openssl','pkey','-pubin','-outform','DER'],input=pem,check=True,capture_output=True).stdout
        spki=base64.b64encode(hashlib.sha256(der).digest()).decode()
        try:
            bridge=subprocess.Popen([sys.executable,'-u',str(root/'quest_unitree_bridge.py'),
                '--port',str(free_port()),'--pose-port',str(free_port()),'--pause-port',str(free_port()),
                '--cert',str(cert),'--key',str(key)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
            processes.append(bridge)
            ready=json.loads(await asyncio.wait_for(asyncio.to_thread(bridge.stdout.readline),10))
            chrome=subprocess.Popen(['/opt/google/chrome/google-chrome','--headless=new',
                '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
                '--user-data-dir='+str(directory/'browser'),'--no-first-run','--no-default-browser-check',
                '--ignore-certificate-errors-spki-list='+spki,'about:blank'],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            processes.append(chrome)
            portfile=directory/'browser'/'DevToolsActivePort'
            deadline=time.monotonic()+10
            while not portfile.exists():
                if chrome.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('isolated Chrome did not start')
                await asyncio.sleep(.1)
            port=int(portfile.read_text().splitlines()[0])
            async with aiohttp.ClientSession() as client:
                async with client.get(f'http://127.0.0.1:{port}/json/list') as response:
                    tabs=await response.json()
                endpoint=next(t['webSocketDebuggerUrl'] for t in tabs if t['type']=='page')
                async with client.ws_connect(endpoint) as ws:
                    events=[];sequence=0
                    async def call(method,params=None):
                        nonlocal sequence
                        sequence+=1
                        await ws.send_json(dict(id=sequence,method=method,params=params or {}))
                        while True:
                            value=await ws.receive_json(timeout=10)
                            if value.get('id')==sequence:
                                if 'error' in value:raise RuntimeError(value['error'])
                                return value.get('result',{})
                            events.append(value)
                    async def evaluate(code):
                        result=await call('Runtime.evaluate',dict(expression=code,returnByValue=True))
                        if 'exceptionDetails' in result:raise RuntimeError(result['exceptionDetails'])
                        return result.get('result',{}).get('value')
                    await call('Network.enable')
                    await call('Page.enable')
                    await call('Page.navigate',{'url':ready['url']})
                    for _ in range(100):
                        if await evaluate('!!document.querySelector("form")'):break
                        await asyncio.sleep(.05)
                    else:raise AssertionError('pairing form missing')
                    secure=await evaluate('window.isSecureContext')
                    await evaluate('document.querySelector("input").value='+json.dumps(ready['pairing_code'])+';document.querySelector("form").requestSubmit()')
                    pair=None
                    for _ in range(100):
                        await evaluate('document.readyState')
                        pair=next((e['params'] for e in events if e.get('method')=='Network.responseReceived'
                            and e['params']['response']['url'].endswith('/pair')),None)
                        redirects=[e['params'] for e in events if e.get('method')=='Network.requestWillBeSent'
                            and e['params'].get('redirectResponse',{}).get('url','').endswith('/pair')]
                        if pair or redirects:break
                        await asyncio.sleep(.05)
                    request_ids=[e['params']['requestId'] for e in events if e.get('method')=='Network.requestWillBeSent'
                        and e['params']['request']['url'].endswith('/pair')]
                    origins=[e['params']['headers'].get('Origin') for e in events
                        if e.get('method')=='Network.requestWillBeSentExtraInfo' and e['params']['requestId'] in request_ids
                        and 'Origin' in e['params']['headers']]
                    status=pair['response']['status'] if pair else redirects[0]['redirectResponse']['status']
                    print('Browser pairing:',dict(secure_context=secure,origins=origins,status=status),flush=True)
                    assert secure is True
                    assert status==(403 if expect_blocked else 303)
                    if not expect_blocked:
                        for _ in range(100):
                            if await evaluate('document.title')=='Unitree · xArm 输入':break
                            await asyncio.sleep(.05)
                        else:raise AssertionError('authenticated Vuer page did not load')
                        result=await call('Runtime.evaluate',dict(expression='fetch("/unitree-clock").then(r=>r.status)',awaitPromise=True,returnByValue=True))
                        assert result['result']['value']==200
                        print('Real Chrome form / redirect / secure cookie / authenticated fetch: PASS')
                        # Reopening a paired HTTPS root loses the original ws
                        # query. Vuer must still connect to this test port.
                        events.clear()
                        await call('Page.navigate',{'url':ready['url'].split('?')[0]})
                        expected_socket=ready['url'].split('?')[0].replace('https://','wss://')
                        for _ in range(200):
                            await evaluate('document.readyState')
                            created={e['params']['requestId']:e['params']['url'] for e in events
                                if e.get('method')=='Network.webSocketCreated'}
                            connected=[created.get(e['params']['requestId']) for e in events
                                if e.get('method')=='Network.webSocketHandshakeResponseReceived'
                                and e['params']['response']['status']==101]
                            if expected_socket in connected:break
                            await asyncio.sleep(.05)
                        else:raise AssertionError(f'paired root did not reconnect: {created}')
                        # Exercise the actual persistent clock in Chrome; fake
                        # XR objects are confined to this isolated test server.
                        probe = """(()=>{const right={handedness:'right',targetRayMode:'tracked-pointer',
                          profiles:['meta-quest-touch-plus'],targetRaySpace:{},gamepad:{mapping:'xr-standard',
                          buttons:Array.from({length:6},()=>({pressed:false}))}};
                          return window.xarmUnitreeMeta({session:{visibilityState:'visible',inputSources:[right]},
                          getPose:()=>({emulatedPosition:false})},right,{addEventListener(){}});})()"""
                        for _ in range(100):
                            meta=await evaluate(probe)
                            if meta.get('valid'):break
                            await asyncio.sleep(.05)
                        else:raise AssertionError(f'WebSocket clock failed: {meta}')
                        assert meta['clock']['transport']=='websocket'
                        print('Real Chrome persistent WebSocket clock / XR metadata: PASS')
                        production=json.dumps((root/'quest_unitree_quality.js').read_text())
                        checks='\n'.join((root/'tests'/name).read_text() for name in
                            ('unitree_clock_checks.js','unitree_recovery_checks.js','unitree_sleep_checks.js'))
                        expression='(async()=>{'+checks+'\nconst install=env=>new Function(...Object.keys(env),'+production+')( ...Object.values(env));return [await runUnitreeClockChecks(install),await runUnitreeRecoveryChecks(install),await runUnitreeSleepChecks(install)];})()'
                        result=await call('Runtime.evaluate',dict(expression=expression,awaitPromise=True,returnByValue=True))
                        assert 'exceptionDetails' not in result,result
                        print('Deterministic clock / controller recovery:',result['result']['value'])
                        assert await evaluate('new URL(location.href).searchParams.get("ws")')==expected_socket
                        print('Paired HTTPS root reopen retains socket port and completes WebSocket handshake: PASS')
                        # Desktop recovery is authenticated, and reaches a
                        # paired browser without ADB or controller messages.
                        from urllib.parse import urlsplit,parse_qs,urlencode
                        token=parse_qs(urlsplit(ready['launch_url']).query)['token'][0]
                        reopen=urlsplit(ready['url'])._replace(path='/unitree-reopen',query=urlencode({'token':token})).geturl()
                        async with client.post(reopen,ssl=__import__('ssl').create_default_context(cafile=str(cert))) as response:
                            assert response.status==200
                            recovery=await response.json()
                        for _ in range(100):
                            if await evaluate('new URL(location.href).searchParams.get("xarm_reopen")')==recovery['generation']:break
                            await asyncio.sleep(.05)
                        else:raise AssertionError('desktop Wi-Fi recovery failed to navigate browser')
                        await asyncio.sleep(2.1)
                        assert await evaluate('new URL(location.href).searchParams.get("xarm_reopen")')==recovery['generation']
                        title=await evaluate('document.title')
                        assert title in ('Unitree · xArm 输入','Vuer (Technical Preview)'),title
                        assert await evaluate('typeof window.xarmUnitreeMeta')=='function'
                        assert not await evaluate('!!document.querySelector("form")')
                        print('Real desktop Wi-Fi recovery / paired navigation / no reload loop: PASS')
                        await call('Page.navigate',{'url':'about:blank'})
                        async with client.post(reopen,ssl=__import__('ssl').create_default_context(cafile=str(cert))) as response:
                            queued=await response.json()
                        await call('Page.navigate',{'url':ready['url'].split('?')[0]})
                        for _ in range(100):
                            if await evaluate('new URL(location.href).searchParams.get("xarm_reopen")')==queued['generation']:break
                            await asyncio.sleep(.05)
                        else:raise AssertionError('offline recovery request was not delivered on reconnect')
                        print('Queued recovery while browser closed is delivered when paired browser returns: PASS')

        finally:
            for process in reversed(processes):
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=3)  # isolated test processes only


if __name__=='__main__':
    asyncio.run(main('--expect-blocked' in sys.argv))
