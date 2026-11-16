"""Passive recorder integration with synthetic cameras/input; no SDK or hardware."""
import json
import queue
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from quest_recorder import CameraWorker, Recorder
from quest_recording_stream import RecordingPublisher
from test_recording import sample


def wait_until(condition,timeout=4):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if condition():
            return
        time.sleep(.01)
    raise AssertionError("timed out waiting for synthetic recorder")


def test_camera_recovers_after_missing_device_and_read_failure(tmp_path, monkeypatch):
    device = tmp_path/'camera'
    stop = threading.Event()
    disconnect = threading.Event()
    inbox = queue.Queue()
    captures = []

    class Capture:
        def __init__(self, *args):
            self.released = False
            captures.append(self)
        def isOpened(self):
            return True
        def set(self, *args):
            return True
        def get(self, *args):
            return 30.
        def read(self):
            time.sleep(.01)
            if disconnect.is_set():
                return False, None
            return True, np.zeros((48,64,3),np.uint8)
        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, 'VideoCapture', Capture)
    monkeypatch.setattr(CameraWorker, 'RETRY_SECONDS', .05)
    worker = CameraWorker(dict(name='wrist',path=str(device),width=64,height=48,fps=30),
                          inbox,stop,tmp_path)
    try:
        wait_until(lambda: '设备未连接' in worker.state['error'])
        assert not captures and not worker.state['ready']
        device.touch()
        wait_until(lambda: worker.state['ready'])
        assert worker.state['error'] == ''
        before = worker.state['frames']
        device.unlink()
        disconnect.set()
        wait_until(lambda: captures[0].released and not worker.state['ready'])
        assert worker.state['measured_fps'] == 0
        disconnect.clear()
        device.touch()
        wait_until(lambda: len(captures)>1 and worker.state['ready'])
        assert worker.state['error'] == '' and worker.state['frames']>before
    finally:
        stop.set()
        worker.thread.join(timeout=2)
    assert not worker.thread.is_alive() and all(c.released for c in captures)
    sequences = []
    while not inbox.empty():
        sequences.append(inbox.get_nowait()['sequence'])
    assert sequences == list(range(len(sequences)))


@pytest.mark.parametrize(('updates','message'), [
    (dict(source_valid=False),'右手柄追踪无效'),
    (dict(recovery_active=True),'正在复位'),
    (dict(error='servo failed'),'servo failed'),
    (dict(mode='preview'),'遥操未启用'),
    (dict(pause_settling=True),'正在停止'),
])
def test_readiness_reports_control_and_camera_failures_together(tmp_path, updates, message):
    recorder = Recorder(tmp_path,dict(cameras=[]),status_path=tmp_path/'status.json')
    recorder.latest = dict(sample(0,time.monotonic()),**updates)
    recorder.cameras = [SimpleNamespace(config=dict(name='wrist'),
        state=dict(ready=False,error='设备未连接'))]
    error = recorder.ready_error()
    assert message in error and 'wrist：设备未连接' in error
    with pytest.raises(ValueError, match=message):
        recorder.handle(dict(command='start',task='blocked capture'))
    assert recorder.episode is None


def prepared_recorder(tmp_path):
    recorder=Recorder(tmp_path,dict(cameras=[]),status_path=tmp_path/'status.json')
    recorder.latest=sample(0,time.monotonic())
    recorder.ready_error=lambda:''
    recorder.handle(dict(command='start',task='handover'))
    start=recorder.episode.metadata['started_monotonic_s']
    for i in range(1,4):
        value=sample(i,start+i*.01)
        value.update(active=True,hand=dict(trigger_pressed=True,trigger_release_generation=0))
        recorder.ingest_sample(value)
    value=sample(4,start+.04)
    value.update(command_valid=False,active=False,hand=dict(trigger_pressed=False,trigger_release_generation=1))
    recorder.ingest_sample(value)
    return recorder,start


def test_review_freezes_samples_and_requires_explicit_result(tmp_path):
    recorder,start=prepared_recorder(tmp_path)
    episode=recorder.episode
    with pytest.raises(ValueError,match='人工选择'):
        recorder.handle(dict(command='save'))
    # Taking off the headset after release does not lose the episode.
    value=sample(5,start+.05)
    value.update(source_valid=False)
    recorder.ingest_sample(value)
    assert recorder.episode is episode
    recorder.handle(dict(command='review'))
    assert recorder.status()['capture_state']=='review'
    count=episode.samples
    value=sample(6,start+.06)
    value.update(active=True,hand=dict(trigger_pressed=True),run_id='another-run')
    recorder.ingest_sample(value)
    assert episode.samples==count and recorder.episode is episode
    recorder.handle(dict(command='save',success=False))
    meta=json.loads((episode.path/'episode.json').read_text())
    assert meta['status']=='complete' and meta['success'] is False
    assert meta['retained'] is True and meta['outcome']=='failure'
    assert meta['review']['source']=='operator_review'
    assert len(meta['capture_intervals'])==1


def test_discard_keeps_raw_but_excludes_training(tmp_path):
    recorder,_=prepared_recorder(tmp_path)
    path=recorder.episode.path
    recorder.handle(dict(command='review'))
    recorder.handle(dict(command='discard',success=True))
    meta=json.loads((path/'episode.json').read_text())
    assert meta['status']=='discarded' and meta['success'] is True and not meta['retained']
    assert (path/'raw.sqlite3').exists()


def test_start_and_finish_while_teleop_follows_does_not_require_trigger_release(tmp_path):
    recorder=Recorder(tmp_path,dict(cameras=[]),status_path=tmp_path/'status.json')
    recorder.latest=dict(sample(0,time.monotonic()),active=True,
        hand=dict(trigger_pressed=True,trigger_release_generation=0))
    recorder.ready_error=lambda:''
    recorder.handle(dict(command='start',task='continuous teleop'))
    episode=recorder.episode
    start=episode.metadata['started_monotonic_s']
    for i in range(1,4):
        value=dict(sample(i,start+i*.01),active=True,
            hand=dict(trigger_pressed=True,trigger_release_generation=0))
        recorder.ingest_sample(value)
    assert episode.samples==3 and recorder.status()['can_review']
    recorder.handle(dict(command='review'))
    assert recorder.latest['active'] and recorder.latest['hand']['trigger_pressed']
    assert recorder.gate.segment_at(start+.03)==0
    assert recorder.gate.segment_at(start+.031) is None
    recorder.ingest_sample(dict(value,sequence=4,monotonic_s=start+.04))
    assert episode.samples==3
    recorder.handle(dict(command='save',success=True))
    assert recorder.latest['active'] and recorder.latest['hand']['trigger_pressed']
    meta=json.loads((episode.path/'episode.json').read_text())
    assert meta['status']=='complete' and meta['capture_intervals'][0]['end_s']<=start+.030002


def test_recorder_actions_do_not_interrupt_an_actively_following_controller(tmp_path,monkeypatch):
    from test_adaptive_pose_filter import make_controller,start,feed
    from quest_recording_stream import recording_sample
    clock,arm,controller=make_controller()
    start(clock,controller)
    monkeypatch.setattr('quest_recorder.time.monotonic',clock)
    recorder=Recorder(tmp_path,dict(cameras=[]),status_path=tmp_path/'status.json')
    recorder.latest=dict(recording_sample(controller),sequence=0)
    recorder.ready_error=lambda:''
    count=controller.command_count
    recorder.handle(dict(command='start',task='independent recorder'))
    assert controller.servo_started and controller.command_count==count
    for i in range(1,4):
        feed(clock,controller,position_m=[.2,1,.301+i*.0001])
        recorder.ingest_sample(dict(recording_sample(controller),sequence=i))
    assert recorder.episode.samples==3
    count=controller.command_count
    recorder.handle(dict(command='review'))
    recorder.handle(dict(command='save',success=True))
    recorder.handle(dict(command='shutdown'))
    assert controller.servo_started and controller.command_count==count
    feed(clock,controller,position_m=[.2,1,.302])
    assert controller.servo_started and controller.command_count>count


def test_tracking_failure_during_active_capture_preserves_episode(tmp_path):
    recorder,start=prepared_recorder(tmp_path)
    path=recorder.episode.path
    value=sample(5,start+.05)
    value.update(active=True,hand=dict(trigger_pressed=True,trigger_release_generation=1))
    recorder.ingest_sample(value)
    recorder.ingest_sample(dict(value,sequence=6,monotonic_s=start+.06,source_valid=False))
    assert recorder.episode is not None
    assert recorder.status()['capture_state']=='input_paused'
    assert recorder.gate.current is None
    assert recorder.gate.require_release
    assert json.loads((path/'episode.json').read_text())['status']=='recording'
    recorder.finish('incomplete',reason='test cleanup')


def test_recorder_lifecycle_and_feed_do_not_control_robot(tmp_path,monkeypatch):
    class FakeCamera:
        def __init__(self,config,inbox,stop,preview_dir):
            self.config=config
            self.state=dict(ready=True,error="",dropped=0,frames=0,last_frame_s=time.monotonic())
            ok,png=cv2.imencode(".png",np.full((48,64,3),100,np.uint8))
            assert ok
            def run():
                i=0
                while not stop.is_set():
                    now=time.monotonic()
                    inbox.put(dict(camera=config["name"],sequence=i,t=now,read_start=now-.001,
                        width=64,height=48,png=png.tobytes()))
                    self.state.update(last_frame_s=now,frames=i)
                    i+=1
                    stop.wait(1/30)
            self.thread=threading.Thread(target=run,daemon=True)
            self.thread.start()
    monkeypatch.setattr("quest_recorder.CameraWorker",FakeCamera)
    monkeypatch.setattr(Recorder,"command_reader",lambda _:None)
    config=dict(cameras=[dict(name=n,width=64,height=48,fps=30) for n in ("global","wrist")])
    sock=tmp_path/"samples.sock"
    recorder=Recorder(tmp_path/"raw",config,socket_path=sock,status_path=tmp_path/"status.json")
    errors=[]
    def run():
        try:
            recorder.run()
        except BaseException as error:
            errors.append(error)
    thread=threading.Thread(target=run,daemon=True)
    thread.start()
    publisher=RecordingPublisher(sock)
    source_stop=threading.Event()
    trigger=threading.Event()
    def feed():
        i=0
        previous=False
        generation=0
        while not source_stop.is_set():
            pressed=trigger.is_set()
            generation+=int(previous and not pressed)
            previous=pressed
            value=sample(i,time.monotonic())
            value.update(active=pressed,command_valid=pressed,
                         hand=dict(trigger_pressed=pressed,trigger_release_generation=generation))
            publisher.publish(value)
            i+=1
            source_stop.wait(.01)
    feeder=threading.Thread(target=feed,daemon=True)
    feeder.start()
    try:
        wait_until(lambda: recorder.latest is not None and not recorder.ready_error())
        recorder.commands.put(dict(command="start",task="synthetic lifecycle test"))
        wait_until(lambda: recorder.episode is not None)
        time.sleep(.15)
        assert recorder.episode.samples==0 and not recorder.episode.images
        trigger.set()
        wait_until(lambda: recorder.episode is not None and recorder.episode.samples>20)
        # Saving while held must leave the episode open.
        recorder.commands.put(dict(command="save",success=True))
        wait_until(lambda: '先松开' in recorder.last_message)
        assert recorder.episode is not None and recorder.completed==0
        trigger.clear()
        wait_until(lambda: recorder.gate.current is None)
        time.sleep(.1)
        paused_samples=recorder.episode.samples
        paused_images=dict(recorder.episode.images)
        episode_id=recorder.episode.id
        time.sleep(.15)
        assert recorder.episode.samples==paused_samples
        assert recorder.episode.images==paused_images
        trigger.set()
        wait_until(lambda: recorder.episode.samples>paused_samples+20)
        assert recorder.episode.id==episode_id
        trigger.clear()
        wait_until(lambda: recorder.gate.current is None)
        recorder.commands.put(dict(command="save",success=True))
        wait_until(lambda: recorder.completed==1)
        metadata=json.loads((__import__("pathlib").Path(recorder.last_path)/"episode.json").read_text())
        assert metadata["status"]=="complete" and metadata["success"]
        assert metadata["image_counts"]["global"]>=2
        assert len(metadata['capture_intervals'])==2
        assert metadata['quality']['telemetry_gaps']==0
        assert metadata['quality']['camera_queue_drops']==0
        from recording_data import open_episode, aligned_segments
        db,metadata=open_episode(recorder.last_path)
        try:
            segments=list(aligned_segments(db,metadata))
            assert len(segments)==2
            for t,payload in db.execute('SELECT t,payload FROM telemetry'):
                assert json.loads(payload)['hand']['trigger_pressed'] is True
                assert any(s['start_s']<=t<s['end_s'] for s in metadata['capture_intervals'])
            for (t,) in db.execute('SELECT t FROM images'):
                assert any(s['start_s']<=t<s['end_s'] for s in metadata['capture_intervals'])
        finally:
            db.close()
        recorder.commands.put(dict(command="start",task="interrupted test"))
        wait_until(lambda: recorder.episode is not None)
        recorder.shutdown_requested=True
        thread.join(timeout=4)
        assert not thread.is_alive() and not errors
        metadata=json.loads((__import__("pathlib").Path(recorder.last_path)/"episode.json").read_text())
        assert metadata["status"]=="incomplete"
    finally:
        recorder.shutdown_requested=True
        source_stop.set()
        feeder.join(timeout=1)
        publisher.close()
        thread.join(timeout=4)
