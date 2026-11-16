#!/usr/bin/env python3
"""Export immutable raw episodes to LeRobot v2.1 or RoboTwin/XPolicyLab HDF5.

Default: successful complete episodes, 30 Hz, absolute 7-joint + aperture actions.
No imports from the installed LeRobot: a v3 installation cannot silently write v3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from compact_export import compact_episode

from recording_data import atomic_json, image_rgb, json_text, open_episode, source_fingerprint
from recording_quality import (DATA_ERRORS, camera_signature, episode_candidates, inspect_episode,
                               new_report, scan_dataset, write_report)

DATA_PATH="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CAMERA_KEYS={"global":"cam_high","wrist":"cam_left_wrist"}
ROOT=Path(__file__).resolve().parent


def rotate_image(image,rotation):
    if rotation not in (0,90,180,270):
        raise ValueError("相机旋转必须是 0/90/180/270")
    return np.ascontiguousarray(np.rot90(image,k=-(rotation//90)))


def feature_names(action_space):
    if action_space in ("joint","joint_delta"):
        return [f"joint_{i+1}_rad" for i in range(7)]+["gripper_open_fraction"]
    return ["tcp_x_m","tcp_y_m","tcp_z_m","qw","qx","qy","qz","gripper_open_fraction"]


def numeric_stats(array):
    array=np.asarray(array)
    if array.ndim==1:
        array=array[:,None]
    return {"min":array.min(axis=0).tolist(),"max":array.max(axis=0).tolist(),
            "mean":array.mean(axis=0).tolist(),"std":array.std(axis=0).tolist(),"count":[len(array)]}


class VideoWriter:
    def __init__(self,path,fps):
        import av
        self.av=av
        path.parent.mkdir(parents=True,exist_ok=True)
        self.container=av.open(str(path),"w")
        self.stream=self.container.add_stream("libx264",rate=fps)
        self.stream.pix_fmt="yuv420p"
        self.stream.options={"crf":"18","preset":"fast","g":"2"}
        self.stream.codec_context.thread_count=2
        self.count=0
        self.sum=np.zeros(3)
        self.square=np.zeros(3)
        self.low=np.ones(3)
        self.high=np.zeros(3)
        self.pixels=0

    def add(self,image):
        h,w=image.shape[:2]
        if h%2 or w%2:
            raise ValueError("H.264 yuv420p 要求宽高为偶数")
        self.stream.width=w
        self.stream.height=h
        frame=self.av.VideoFrame.from_ndarray(image,format="rgb24")
        frame.pts=self.count
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        # Image statistics follow v2.1 per-channel [3,1,1] convention.
        pixels=image[::4,::4].reshape(-1,3).astype(np.float64)/255
        self.sum+=pixels.sum(axis=0)
        self.square+=(pixels*pixels).sum(axis=0)
        self.low=np.minimum(self.low,pixels.min(axis=0))
        self.high=np.maximum(self.high,pixels.max(axis=0))
        self.pixels+=len(pixels)
        self.count+=1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
        mean=self.sum/self.pixels
        return {k:np.asarray(v).reshape(3,1,1).tolist() for k,v in dict(
            min=self.low,max=self.high,mean=mean,std=np.sqrt(np.maximum(0,self.square/self.pixels-mean**2))).items()}|{"count":[self.count]}


class V21Exporter:
    def __init__(self,path,fps,action_space):
        self.path,self.fps,self.action_space=path,fps,action_space
        (path/"meta").mkdir(parents=True)
        self.features={key:dict(dtype="float32",shape=[8],names=feature_names(action_space))
                       for key in ("observation.state","action")}
        if action_space=="joint_delta":
            self.features["action"]["names"]=[f"joint_{i+1}_delta_rad" for i in range(7)]+["gripper_open_fraction"]
        for key in ("timestamp","frame_index","episode_index","index","task_index"):
            self.features[key]=dict(dtype="float32" if key=="timestamp" else "int64",shape=[1],names=None)
        for key,dtype in (("next.done","bool"),("next.success","bool"),
                          ("annotation.success_valid","bool"),("next.discontinuity","bool"),
                          ("capture.segment_index","int64"),("source.monotonic_s","float64")):
            self.features[key]=dict(dtype=dtype,shape=[1],names=None)
        self.tasks=[]
        self.episodes=0
        self.frames=0
        self.camera_signature=None

    def add(self,db,meta,rows):
        import pyarrow as pa
        import pyarrow.parquet as pq
        signature=[(c["name"],c["width"],c["height"],c.get("rotation",0)) for c in meta["cameras"]]
        if self.camera_signature is not None and signature!=self.camera_signature:
            raise ValueError("同一导出数据集中的相机布局或分辨率发生变化")
        self.camera_signature=signature
        if meta["task"] not in self.tasks:
            self.tasks.append(meta["task"])
        n=len(rows)
        index=self.episodes
        task_index=self.tasks.index(meta["task"])
        arrays={"observation.state":np.asarray([r["state"] for r in rows],np.float32),
                "action":np.asarray([r["action"] for r in rows],np.float32),
                "timestamp":np.arange(n,dtype=np.float32)/self.fps,
                "frame_index":np.arange(n,dtype=np.int64),
                "episode_index":np.full(n,index,np.int64),
                "index":np.arange(self.frames,self.frames+n,dtype=np.int64),
                "task_index":np.full(n,task_index,np.int64)}
        done=np.zeros(n,dtype=bool)
        done[-1]=True
        segment_ids=np.asarray([r['export_segment'] for r in rows],np.int64)
        discontinuity=np.zeros(n,dtype=bool)
        discontinuity[:-1]=segment_ids[:-1]!=segment_ids[1:]
        arrays.update({"next.done":done,"next.success":done & (meta.get('success') is True),
            "annotation.success_valid":done & (type(meta.get('success')) is bool),
            "next.discontinuity":discontinuity,"capture.segment_index":segment_ids,
            "source.monotonic_s":np.asarray([r['t'] for r in rows],np.float64)})
        table=pa.table({key:pa.array(value.tolist(),type=pa.list_(pa.float32(),8) if value.ndim==2 else
                                     pa.from_numpy_dtype(value.dtype)) for key,value in arrays.items()})
        file=self.path/DATA_PATH.format(episode_chunk=index//1000,episode_index=index)
        file.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(table,file,compression="zstd")
        stats={key:numeric_stats(value) for key,value in arrays.items()}
        for camera in meta["cameras"]:
            name=camera["name"]
            key="observation.images."+CAMERA_KEYS.get(name,name)
            video=self.path/VIDEO_PATH.format(episode_chunk=index//1000,episode_index=index,video_key=key)
            writer=VideoWriter(video,self.fps)
            try:
                for row in rows:
                    image=rotate_image(image_rgb(db,name,row["images"][name]),camera.get("rotation",0))
                    writer.add(image)
                stats[key]=writer.close()
            except BaseException:
                writer.container.close()
                raise
            h,w=image.shape[:2]
            self.features[key]=dict(dtype="video",shape=[3,h,w],names=["channels","height","width"],
                info={"video.fps":self.fps,"video.height":h,"video.width":w,"video.channels":3,
                      "video.codec":"h264","video.pix_fmt":"yuv420p","video.is_depth_map":False,"has_audio":False})
        self.append("episodes.jsonl",dict(episode_index=index,tasks=[meta["task"]],length=n,
            success=meta.get('success'),outcome=episode_outcome(meta),source_episode_id=meta['episode_id']))
        self.append("episode_annotations.jsonl",dict(episode_index=index,source_episode_id=meta['episode_id'],
            outcome=episode_outcome(meta),success=meta.get('success'),review=meta.get('review'),
            terminal_frame_index=n-1,capture_intervals=meta.get('capture_intervals',[])))
        self.append("episodes_stats.jsonl",dict(episode_index=index,stats=stats))
        self.episodes+=1
        self.frames+=n
        return index

    def append(self,name,value):
        with (self.path/"meta"/name).open("a",encoding="utf-8") as file:
            file.write(json_text(value)+"\n")

    def finish(self):
        for i,task in enumerate(self.tasks):
            self.append("tasks.jsonl",dict(task_index=i,task=task))
        atomic_json(self.path/"meta/info.json",dict(codebase_version="v2.1",robot_type="xarm7_g2",
            total_episodes=self.episodes,total_frames=self.frames,total_tasks=len(self.tasks),
            total_videos=self.episodes*sum(f["dtype"]=="video" for f in self.features.values()),
            total_chunks=(self.episodes+999)//1000,chunks_size=1000,fps=self.fps,
            splits={"train":f"0:{self.episodes}"},data_path=DATA_PATH,video_path=VIDEO_PATH,features=self.features))


class RoboTwinExporter:
    """XPolicyLab native fields plus explicit single-arm aliases for its v21 converter."""
    def __init__(self,path,fps,action_space):
        if action_space!="joint":
            raise ValueError("RoboTwin 导出当前使用绝对关节动作；其他动作空间请选 LeRobot v2.1")
        self.path,self.fps=path,fps
        self.episodes=0
        (path/"data").mkdir(parents=True)

    def add(self,db,meta,rows):
        import h5py
        sys.path.insert(0,str(ROOT/"third_party/robotwin_image_codec"))
        from decode_image_bit import encode_image_bit
        state=np.asarray([r["state"] for r in rows],np.float32)
        action=np.asarray([r["action"] for r in rows],np.float32)
        index=self.episodes
        with h5py.File(self.path/"data"/f"episode_{index:07d}.hdf5","w") as file:
            file.attrs.update(schema="XPolicyLab xspark v1.0",robot_type="xarm7_g2",fps=self.fps,
                arm_dim=7,ee_dim=1,single_arm_alias="left is the sole xArm7; no right arm exists",
                gripper_convention="0 closed, 1 open; physical aperture / 84 mm",
                action_semantics="absolute commanded joint radians; settled pause holds measured joints")
            file.create_dataset("instructions",data=json_text([meta["task"]]),dtype=h5py.string_dtype())
            for name,values in (("state",state),("action",action)):
                group=file.create_group(name)
                group.create_dataset("joint_states",data=values[:,:7])
                group.create_dataset("ee_joint_states",data=values[:,7:])
                # The current official converter reads left/right keys even for
                # single arms. Hard links adapt names without creating fake data.
                group["left_arm_joint_states"]=group["joint_states"]
                group["left_ee_joint_states"]=group["ee_joint_states"]
            for camera in meta["cameras"]:
                name=camera["name"]
                key={"global":"cam_head","wrist":"cam_left_wrist"}.get(name,name)
                group=file.create_group("vision/"+key)
                colors=group.create_dataset("colors",(len(rows),),dtype=h5py.vlen_dtype(np.dtype("uint8")))
                group.attrs.update(source_camera=name,color_order="RGB",encoding="official marked JPEG")
                for i,row in enumerate(rows):
                    image=rotate_image(image_rgb(db,name,row["images"][name]),camera.get("rotation",0))
                    colors[i]=np.frombuffer(encode_image_bit(image,quality=95),np.uint8)
            file.create_dataset("timestamps",data=np.arange(len(rows))/self.fps)
            file.create_dataset("source_monotonic_s",data=[r["t"] for r in rows])
            file.create_dataset("capture_segment_index",data=[r['export_segment'] for r in rows])
            file.attrs['outcome']=episode_outcome(meta)
            file.attrs['success_valid']=type(meta.get('success')) is bool
            file.attrs['success']=meta.get('success') is True
        self.episodes+=1
        return index

    def finish(self):
        atomic_json(self.path/"xarm7_profile.json",dict(robot_type="xarm7_g2",arm_dim=[7],ee_dim=[1],
            fps=self.fps,data_type="RoboDojo",data_version="v1.0",camera_mapping=CAMERA_KEYS,
            missing_cameras=["cam_right_wrist"], note="Configure single-arm dimensions; do not use ALOHA 14-dimensional config."))


def episode_outcome(meta):
    return 'success' if meta.get('success') is True else 'failure' if meta.get('success') is False else 'unknown'


def export_dataset(source,destination,kind="lerobot21",fps=30,action_space="joint",include_unsuccessful=False,include_failures=False,clean=False):
    # Management moves require the exclusive side of this lock. An export must
    # retain stable source paths until videos, manifests and hashes are finished.
    from recording_library import library_lock
    with library_lock(source):
        return _export_dataset(source,destination,kind,fps,action_space,include_unsuccessful,include_failures,clean)


def _export_dataset(source,destination,kind="lerobot21",fps=30,action_space="joint",include_unsuccessful=False,include_failures=False,clean=False):
    source,destination=Path(source),Path(destination)
    partial=destination.with_name(destination.name+".partial")
    if destination.exists() or destination.is_symlink() or partial.exists() or partial.is_symlink():
        raise ValueError("导出目录已存在；请选择新目录，不覆盖已有数据")
    if kind not in ('lerobot21','robotwin') or action_space not in ('joint','joint_delta','tcp_pose'):
        raise ValueError('导出格式或动作空间无效')
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError('导出 FPS 必须在 1–60 之间')
    partial.mkdir(parents=True)
    provenance=[]
    report=new_report(source,fps,action_space,include_failures,include_unsuccessful)
    signature=None
    try:
        exporter=(V21Exporter if kind=='lerobot21' else RoboTwinExporter)(partial,fps,action_space)
        candidates=list(episode_candidates(source,include_unsuccessful,include_failures))
        if not any(item['decision'] in ('pending','rejected') for _,item in candidates):
            report['episodes']=[item for _,item in candidates]
            raise ValueError('没有可导出的成功完整轨迹；可显式纳入失败或未标注轨迹')
        for path,item in candidates:
            report['episodes'].append(item)
            if item['decision']=='excluded':
                continue
            db=None
            try:
                if item['decision']=='rejected':
                    raise ValueError(item['reason'])
                db,meta=open_episode(path)
                segments,metrics=inspect_episode(db,meta,fps,action_space)
                current=camera_signature(meta)
                if signature is not None and signature!=current:
                    raise ValueError('相机布局与本批已通过轨迹不同，请分开导出')
                signature=current
                item.update(metrics,decision='accepted',reason='结构、对齐和图像校验通过')
            except DATA_ERRORS as error:
                if db is not None:
                    db.close()
                item.update(decision='rejected',reason=str(error))
                write_report(partial/'quality_report.json',report)
                if not clean:
                    raise ValueError(f'{path.name}: {error}') from error
                print(f'已跳过 {path.name}：{error}',flush=True)
                continue
            try:
                fingerprint=source_fingerprint(path)
                rows,capture_windows,windows,boundary_jumps=compact_episode(segments,action_space)
                index=exporter.add(db,meta,rows)
                provenance.append(dict(episode_index=index,source=str(path),sha256=fingerprint,
                    task=meta["task"],success=meta.get("success"),outcome=episode_outcome(meta),frames=len(rows),
                    capture_segments=len(segments),training_windows=windows,
                    capture_windows=capture_windows,boundary_jumps=boundary_jumps,robot_frame=meta.get("robot_info"),
                    start_monotonic_s=rows[0]["t"],end_monotonic_s=rows[-1]["t"]))
                # One task remains one episode; real timing and boundaries remain auditable.
                alignment=partial/"alignment"
                alignment.mkdir(exist_ok=True)
                with (alignment/f"episode_{index:06d}.jsonl").open("w") as file:
                    for row in rows:
                        file.write(json_text({k:v for k,v in row.items() if k not in ("state","action")})+"\n")
                print(f"已转换 {path.name}",flush=True)
            finally:
                db.close()
        if not exporter.episodes:
            raise ValueError("没有长度足够的连续片段")
        exporter.finish()
        write_report(partial/'quality_report.json',report)
        atomic_json(partial/"export_manifest.json",dict(format=kind,fps=fps,action_space=action_space,
            exporter_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                             for name in ("export_recording.py","recording_data.py","recording_quality.py","compact_export.py")},
            state_semantics="measured joint rad + gripper open fraction" if action_space!="tcp_pose" else "measured base-frame position m + quaternion wxyz + gripper",
            sampling_mode="compact_within_episode",sampling_schema_version=2,
            pause_policy="one raw task = one exported episode; active intervals concatenated on a compact playback clock; source.monotonic_s preserves real time; next.discontinuity marks gaps",
            outcome_policy="operator episode label at final recorded row: next.done=true, next.success=success; annotation.success_valid distinguishes unknown; no inferred reward or success onset",
            training_policy="compact_within_episode: training_windows spans the full demonstration, including internal pause joins; never cross episode boundaries; retain every anchor with masked tail padding",
            alignment="nearest host-timestamp sample; camera <=40ms, robot <=25ms; not hardware sync",
            gripper="sample-and-hold, maximum feedback age 0.75s, aperture_mm / 84",
            camera_mapping=CAMERA_KEYS,cleaning=report['summary'],quality_report="quality_report.json",episodes=provenance))
        partial.rename(destination)
    except BaseException as error:
        write_report(partial/'quality_report.json',report)
        atomic_json(partial/"EXPORT_FAILED.json",dict(error=str(error)))
        raise
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("destination",type=Path)
    parser.add_argument("--format",choices=("lerobot21","robotwin"),default="lerobot21")
    parser.add_argument("--fps",type=int,default=30)
    parser.add_argument("--action-space",choices=("joint","joint_delta","tcp_pose"),default="joint")
    parser.add_argument("--include-unsuccessful",action="store_true")
    parser.add_argument("--include-failures",action="store_true",help="同时纳入人工判定失败的完整保留轨迹，排除未标注")
    parser.add_argument('--clean',action='store_true',help='跳过不合格整条轨迹，保留报告和原始数据')
    parser.add_argument('--scan-only',action='store_true',help='只检查并生成报告，不编码视频')
    args=parser.parse_args()
    # Offline work yields CPU priority to teleoperation; no control sockets used.
    import os
    import cv2
    cv2.setNumThreads(1)
    os.nice(10)
    try:
        if args.scan_only:
            if args.destination.exists() or args.destination.is_symlink():
                raise ValueError('报告目录已存在；请选择新目录')
            report=scan_dataset(args.source,args.fps,args.action_space,args.include_failures,args.include_unsuccessful)
            args.destination.mkdir(parents=True,exist_ok=False)
            write_report(args.destination/'quality_report.json',report)
            path=args.destination
        else:
            path=export_dataset(args.source,args.destination,args.format,args.fps,args.action_space,args.include_unsuccessful,args.include_failures,args.clean)
        report=json.loads((path/'quality_report.json').read_text())
        print(json_text(dict(type='export_result',scan_only=args.scan_only,output=str(path),
                             report=str(path/'quality_report.json'),summary=report['summary'])),flush=True)
    except Exception:
        failed=args.destination.with_name(args.destination.name+'.partial')/'quality_report.json'
        if failed.is_file():
            print(json_text(dict(type='export_result',scan_only=args.scan_only,output=None,
                                 report=str(failed),summary=json.loads(failed.read_text())['summary'])),flush=True)
        raise


if __name__=="__main__":
    main()
