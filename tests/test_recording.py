"""No hardware: raw storage, alignment, schema, decoder and failure boundaries."""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from recording_data import EpisodeWriter, aligned_segments, open_episode, training_vectors
from export_recording import export_dataset


def sample(i,t=None):
    return dict(sequence=i,monotonic_s=100+i/100 if t is None else t,run_id="test",
        joints_rad=[.1]*7,command_joints_rad=[.2]*7,
        tcp_position_m=[.3,.1,.2],tcp_quaternion_wxyz=[1,0,0,0],
        command_tcp_position_m=[.31,.1,.2],command_tcp_quaternion_wxyz=[1,0,0,0],
        gripper=dict(position_mm=42,position_age_s=.1,committed_target_mm=84,command_fault=False),
        source_valid=True,mode="motion",command_valid=True,pause_settling=False,error="",
        active=False,hand=dict(trigger_pressed=False,trigger_release_generation=0),
        settings_revision=0,settings={})


def make_episode(root,success=True,mutate=None):
    metadata=dict(task="pick up the cup",cameras=[
        dict(name=name,width=64,height=48,fps=30,rotation=0) for name in ("global","wrist")])
    writer=EpisodeWriter(root,metadata)
    for i in range(101):
        value=sample(i)
        if mutate:
            mutate(value,i)
        writer.add_sample(value)
    for i in range(31):
        for camera in ("global","wrist"):
            # Asymmetric colour catches accidental RGB/BGR reversal.
            rgb=np.zeros((48,64,3),np.uint8)
            rgb[:]=[220,30,10] if camera=="global" else [10,30,220]
            rgb[0,0]=[i,255,0]
            ok,png=cv2.imencode(".png",cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            assert ok
            writer.add_image(dict(camera=camera,sequence=i,t=100+i/30,read_start=100+i/30-.001,
                width=64,height=48,png=png.tobytes()))
    return writer.finish(success=success)


def test_committed_gripper_target_not_unsent_request():
    value=sample(0)
    value["gripper"]["target_mm"]=0
    state,action,held=training_vectors(value)
    assert state[-1]==.5 and action[-1]==1 and not held
    value["gripper"]["position_age_s"]=1
    with pytest.raises(ValueError,match="过期"):
        training_vectors(value)


def test_recording_snapshot_does_not_invent_commands_or_query_robot():
    from test_follow_command_guard import setup_guard
    from quest_recording_stream import recording_sample
    _,arm,controller=setup_guard()
    before=list(arm.calls)
    value=recording_sample(controller)
    json.dumps(value,allow_nan=False)
    assert arm.calls==before
    assert value["joints_rad"]==controller.joints.tolist()
    if controller.last_command_joints is None:
        assert value["command_joints_rad"] is None
        assert value["command_valid"] is False


def test_pause_hold_and_delta_are_explicit():
    value=sample(0)
    _,action,_=training_vectors(value,"joint_delta")
    assert action[:7]==pytest.approx([.1]*7)
    value["command_valid"]=False
    state,action,held=training_vectors(value)
    assert held and action[:7]==state[:7]
    value["pause_settling"]=True
    with pytest.raises(ValueError,match="pause_settling"):
        training_vectors(value)


def test_raw_alignment_and_braking_split(tmp_path):
    path=make_episode(tmp_path,mutate=lambda s,i:s.update(pause_settling=40<=i<=60))
    db,meta=open_episode(path)
    segments=list(aligned_segments(db,meta))
    assert len(segments)==2
    assert segments[1][0]["t"]-segments[0][-1]["t"]>.15
    for rows in segments:
        assert np.diff([r["t"] for r in rows])==pytest.approx(1/30)
    db.close()


def test_reject_data_loss_and_never_overwrite(tmp_path):
    path=make_episode(tmp_path,mutate=lambda s,i:s.update(sequence=i+(1 if i>50 else 0)))
    db,meta=open_episode(path)
    with pytest.raises(ValueError,match="丢包"):
        list(aligned_segments(db,meta))
    db.close()
    destination=tmp_path/"existing"
    destination.mkdir()
    with pytest.raises(ValueError,match="已存在"):
        export_dataset(tmp_path,destination)


@pytest.mark.parametrize("update",[dict(source_valid=False),dict(mode="preview"),dict(recovery_active=True),
                                  dict(joints_rad=[float("nan")]*7),dict(command_joints_rad=None)])
def test_invalid_training_values_fail(update):
    value=sample(0)
    value.update(update)
    with pytest.raises(ValueError):
        training_vectors(value)


def test_unfinished_and_failed_excluded(tmp_path):
    make_episode(tmp_path,success=False)
    with pytest.raises(ValueError,match="成功完整"):
        export_dataset(tmp_path,tmp_path/"out")
    writer=EpisodeWriter(tmp_path,dict(task="test",cameras=[]))
    path=writer.finish(status="incomplete")
    with pytest.raises(ValueError,match="未完成"):
        open_episode(path)


def test_v21_export_and_video_decode(tmp_path):
    import av
    import pyarrow.parquet as pq
    make_episode(tmp_path)
    out=export_dataset(tmp_path,tmp_path/"v21")
    info=json.loads((out/"meta/info.json").read_text())
    assert info["codebase_version"]=="v2.1"
    assert info["total_frames"]==31 and info["total_videos"]==2
    table=pq.read_table(out/"data/chunk-000/episode_000000.parquet")
    assert table["observation.state"][0].as_py()==pytest.approx([.1]*7+[.5])
    assert table["action"][0].as_py()==pytest.approx([.2]*7+[1.])
    assert table["timestamp"].to_numpy()==pytest.approx(np.arange(31)/30,abs=1e-6)
    assert table['next.done'].to_pylist()==[False]*30+[True]
    assert table['next.success'].to_pylist()==[False]*30+[True]
    assert table['annotation.success_valid'].to_pylist()==[False]*30+[True]
    with av.open(str(out/"videos/chunk-000/observation.images.cam_high/episode_000000.mp4")) as video:
        frames=list(video.decode(video=0))
    assert len(frames)==31
    pixel=frames[0].to_ndarray(format="rgb24")[20,20]
    assert pixel[0]>200 and pixel[2]<20


def test_export_partial_not_presented_as_dataset(tmp_path):
    make_episode(tmp_path,mutate=lambda s,i:s["gripper"].update(position_age_s=2 if i==50 else .1))
    with pytest.raises(ValueError,match="过期"):
        export_dataset(tmp_path,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()
    assert (tmp_path/"bad.partial/EXPORT_FAILED.json").exists()


def test_official_v21_reader(tmp_path):
    validator=Path(__file__).resolve().parents[1]/".state/v21_validation"
    if not validator.exists():
        pytest.skip("isolated official LeRobot 0.3.3 not installed")
    make_episode(tmp_path)
    out=export_dataset(tmp_path,tmp_path/"v21")
    code='''
import sys,os
os.environ["HF_DATASETS_CACHE"]=str(__import__("pathlib").Path(sys.argv[2]).parent/"hf_cache")
sys.path.insert(0,sys.argv[1])
from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
assert CODEBASE_VERSION == "v2.1"
ds=LeRobotDataset("local/xarm7_test",root=sys.argv[2],video_backend="torchcodec",
    delta_timestamps={"action":[0.,1/30,2/30]})
assert len(ds)==31
for i in (0,15,30):
    row=ds[i]
    assert row["action"].shape==(3,8)
    assert row["observation.images.cam_high"].shape==(3,48,64)
    assert row["task"]=="pick up the cup"
assert ds[30]["action_is_pad"].tolist()==[False,True,True]
assert not ds[0]['next.done'].item()
assert ds[30]['next.done'].item() and ds[30]['next.success'].item()
assert ds[30]['annotation.success_valid'].item()
assert ds[30]['capture.segment_index'].item()==0
print("official v2.1 loader: images, state, action chunks and end padding OK")
'''
    result=subprocess.run([sys.executable,"-c",code,str(validator),str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_official_reader_training_windows_do_not_cross_pause(tmp_path):
    from test_recording_gate import trigger_episode
    validator=Path(__file__).resolve().parents[1]/'.state/v21_validation'
    if not validator.exists():
        pytest.skip('isolated official LeRobot 0.3.3 not installed')
    trigger_episode(tmp_path/'raw')
    out=export_dataset(tmp_path/'raw',tmp_path/'v21')
    code='''
import sys,os
os.environ['HF_DATASETS_CACHE']=str(__import__('pathlib').Path(sys.argv[2]).parent/'hf_cache')
sys.path.insert(0,sys.argv[1])
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from recording_training import restrict_training_windows
ds=LeRobotDataset('local/joined_test',root=sys.argv[2],video_backend='torchcodec',
    delta_timestamps={'action':[0.,1/30,2/30],'capture.segment_index':[0.,1/30,2/30]})
assert ds.num_episodes==1
selected=restrict_training_windows(ds,sys.argv[2],horizon=3)
assert len(selected)>0
segments=set()
for i in range(len(selected)):
    item=selected[i]
    ids=item['capture.segment_index'].flatten().tolist()
    assert len(set(ids))==1
    segments.update(ids)
assert segments=={0,1}
assert ds[len(ds)-1]['next.success'].item()
'''
    result=subprocess.run([sys.executable,'-c',code,str(validator),str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_failure_unknown_and_discarded_outcomes_are_distinct(tmp_path):
    import pyarrow.parquet as pq
    make_episode(tmp_path,success=False)
    unknown=make_episode(tmp_path,success=None)
    discarded=make_episode(tmp_path,success=True)
    meta=json.loads((discarded/'episode.json').read_text())
    meta.update(status='discarded',retained=False)
    (discarded/'episode.json').write_text(json.dumps(meta))
    out=export_dataset(tmp_path,tmp_path/'labeled',include_failures=True)
    info=json.loads((out/'meta/info.json').read_text())
    assert info['total_episodes']==1
    table=pq.read_table(out/'data/chunk-000/episode_000000.parquet').to_pydict()
    assert not any(table['next.success']) and table['next.done'][-1]
    assert table['annotation.success_valid'][-1]
    out=export_dataset(tmp_path,tmp_path/'all',include_unsuccessful=True)
    manifest=json.loads((out/'export_manifest.json').read_text())
    entry=next(e for e in manifest['episodes'] if e['source']==str(unknown))
    table=pq.read_table(out/f"data/chunk-000/episode_{entry['episode_index']:06d}.parquet").to_pydict()
    assert not any(table['annotation.success_valid']) and entry['outcome']=='unknown'


def test_robotwin_official_codec(tmp_path):
    root=Path(__file__).resolve().parents[1]
    python=Path("/home/a/miniconda3/envs/openteach/bin/python")
    if not python.exists():
        pytest.skip("HDF5 export environment not installed")
    make_episode(tmp_path)
    out=tmp_path/"robotwin"
    result=subprocess.run([str(python),str(root/"export_recording.py"),str(tmp_path),str(out),"--format","robotwin"],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    code='''
import sys,h5py,numpy as np
sys.path.insert(0,sys.argv[1])
from decode_image_bit import decode_image_bit
with h5py.File(sys.argv[2]) as f:
    assert f["state/joint_states"].shape==(31,7)
    assert f["action/left_ee_joint_states"].shape==(31,1)
    image=decode_image_bit(f["vision/cam_head/colors"][0])
    assert image[20,20,0]>200 and image[20,20,2]<20
    assert "cam_right_wrist" not in f["vision"]
print("RoboTwin official decoder: RGB and single-arm schema OK")
'''
    result=subprocess.run([str(python),"-c",code,str(root/"third_party/robotwin_image_codec"),str(out/"data/episode_0000000.hdf5")],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_openpi_adapter_preserves_rgb_masks_and_eight_dimensions():
    from openpi_xarm7 import XArm7Inputs,XArm7Outputs
    base=np.zeros((3,48,64),np.float32)
    base[0]=1
    row=XArm7Inputs()(dict(state=np.zeros(8),actions=np.ones((30,8)),
        images=dict(cam_high=base,cam_left_wrist=base),prompt="pick up cup"))
    assert row["image"]["base_0_rgb"][10,10].tolist()==[255,0,0]
    assert row["image_mask"]["right_wrist_0_rgb"]==False
    assert row["actions"].shape==(30,8)
    assert XArm7Outputs()(dict(actions=np.zeros((30,32))))["actions"].shape==(30,8)
