import json
import http.client
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

import pytest
import zmq

from quest_webxr_bridge import InputLease,make_server
from quest_pose_stream import inspect_packet,merge_release_counters

# Local USB endpoints must not be sent through the user's HTTP proxy.
urlopen=urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def packet(**changes):
    value=dict(type='quest_webxr_v1',valid=True,frame_id='test-session',sequence=0,
        position_m=[.2,1,.3],quaternion_wxyz=[1,0,0,0],trigger_pressed=False,
        a_pressed=False,b_pressed=False,release_counters={'trigger':0,'a':0,'b':0},
        sample_server_s=time.monotonic(),sample_age_s=0)
    value.update(changes);return value


def test_independent_b_is_not_trigger_or_legacy_pause():
    result=inspect_packet(json.dumps(packet(trigger_pressed=True,b_pressed=False)).encode())
    assert result['valid'] and result['trigger_pressed'] and result['b_pressed'] is False
    result=inspect_packet(json.dumps(packet(trigger_pressed=False,b_pressed=True)).encode())
    assert result['b_pressed'] and not result['trigger_pressed']
    legacy=b'absolute|1,2,3|1,0,0,0|True|1.1,2,3|1,2.1,3|1,2,2.9'
    assert inspect_packet(legacy)['b_pressed'] is None


@pytest.mark.parametrize('changes',[
    {'b_pressed':1},{'b_pressed':None},{'position_m':[True,1,2]},
    {'quaternion_wxyz':[0,0,0,0]},{'position_m':[float('nan'),1,2]},
    {'sample_age_s':.25},{'sample_age_s':-.1},{'frame_id':''},
    {'release_counters':{'trigger':0,'a':0,'b':False}},
])
def test_invalid_pose_button_or_age_is_not_accepted(changes):
    assert not inspect_packet(json.dumps(packet(**changes)).encode())['valid']


def test_merged_button_releases_and_frame_reset_never_lose_stop_edge():
    state={'invalid_count':0};previous={}
    first=inspect_packet(json.dumps(packet(trigger_pressed=True,b_pressed=True)).encode())
    merge_release_counters(state,first,previous);state.update(first)
    latest=inspect_packet(json.dumps(packet(trigger_pressed=True,b_pressed=True,
        release_counters={'trigger':1,'a':0,'b':1})).encode())
    merge_release_counters(state,latest,previous);state.update(latest)
    assert state['trigger_release_generation']==1 and state['b_release_generation']==1
    reset=inspect_packet(json.dumps(packet(frame_id='new-origin')).encode())
    merge_release_counters(state,reset,previous)
    assert state['invalid_count']==1 and state['b_release_generation']==1


def test_input_lease_rejects_reorder_and_parallel_session_and_stale_sample():
    lease=InputLease()
    lease.accept(packet())
    with pytest.raises(ValueError,match='out-of-order'):lease.accept(packet())
    with pytest.raises(ValueError,match='another XR'):lease.accept(packet(frame_id='other'))
    lease.accept(packet(sequence=1,sample_server_s=time.monotonic()-.4))
    stale,received=lease.latest.get_nowait()
    assert stale['valid'] is False
    lease.accept(dict(type='quest_webxr_v1',valid=False,ended=True,reason='end',frame_id='test-session',sequence=2))
    lease.accept(packet(frame_id='other'))
    assert lease.latest.get_nowait()[0]['frame_id']=='other'


def test_http_requires_token_origin_and_clock_synchronization():
    lease=InputLease();server=make_server(0,'test-token',lease)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    root=f'http://127.0.0.1:{server.server_port}'
    try:
        with pytest.raises(urllib.error.HTTPError):urlopen(root+'/?token=wrong')
        clock=json.loads(urlopen(root+'/clock?token=test-token').read())
        assert abs(clock['monotonic_s']-time.monotonic())<.1
        value=packet();request=urllib.request.Request(root+'/pose?token=test-token',data=json.dumps(value).encode(),headers={'Origin':'https://untrusted.invalid'})
        with pytest.raises(urllib.error.HTTPError):urlopen(request)
        assert lease.latest.empty()
        request=urllib.request.Request(root+'/pose?token=test-token',data=json.dumps(value).encode(),headers={'Origin':root})
        assert json.loads(urlopen(request).read())['ok'] is True
        assert lease.latest.get_nowait()[0]['valid'] is True
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)


def test_usb_input_reuses_http_connection_without_delayed_ack_wait():
    lease=InputLease();server=make_server(0,'test-token',lease)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
    try:
        original_socket=None
        for sequence in range(3):
            connection.request('POST','/pose?token=test-token',json.dumps(packet(sequence=sequence)))
            response=connection.getresponse()
            assert response.status==200 and response.version==11
            response.read()
            if original_socket is None:original_socket=connection.sock
            assert connection.sock is original_socket and not response.will_close
            assert lease.latest.get_nowait()[0]['sequence']==sequence
    finally:
        connection.close();server.shutdown();server.server_close();thread.join(timeout=2)


def test_usb_bridge_posts_to_isolated_zmq_ports_never_live_robot_ports():
    context=zmq.Context();pose=context.socket(zmq.PULL);pause=context.socket(zmq.PULL)
    pose_port=pose.bind_to_random_port('tcp://127.0.0.1');pause_port=pause.bind_to_random_port('tcp://127.0.0.1')
    guard=socket.socket();guard.bind(('127.0.0.1',0));http_port=guard.getsockname()[1];guard.close()
    script=Path(__file__).parents[1]/'quest_webxr_bridge.py'
    process=subprocess.Popen([sys.executable,'-u',str(script),'--port',str(http_port),
        '--pose-port',str(pose_port),'--pause-port',str(pause_port)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        ready=json.loads(process.stdout.readline());url=ready['url'].replace('/?','/pose?')
        request=urllib.request.Request(url,data=json.dumps(packet(b_pressed=True)).encode())
        urlopen(request,timeout=2).read()
        assert pose.poll(1000)
        result=pose.recv_json()
        assert result['b_pressed'] is True and result['trigger_pressed'] is False
        assert result['valid'] is True and result['sample_age_s']<.2
        messages=[];deadline=time.monotonic()+.15
        while time.monotonic()<deadline:
            if pause.poll(20):messages.append(pause.recv())
        assert b'High' in messages  # B does not disable pose streaming
    finally:
        process.terminate();process.wait(timeout=3)
        pose.close(linger=0);pause.close(linger=0);context.term()
