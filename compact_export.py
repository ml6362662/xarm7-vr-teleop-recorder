"""Compact one demonstration, retaining source boundaries and measured jumps."""
import numpy as np


def compact_episode(segments, action_space):
    rows, capture_windows = [], []
    for segment_id, segment in enumerate(segments):
        if len(segment) < 2:
            raise ValueError('按压片段有效帧不足，不能静默删除后拼接')
        capture_windows.append(dict(segment_index=segment_id, from_frame=len(rows), to_frame=len(rows)+len(segment)))
        rows.extend(dict(row, export_segment=segment_id, state=list(row['state']), action=list(row['action'])) for row in segment)
    if not rows:
        raise ValueError('没有可导出的轨迹帧')
    for row in rows:
        for key in ('state','action'):
            if len(row[key]) != 8 or not np.isfinite(row[key]).all():
                raise ValueError('状态或动作维数/有限性错误')
    if action_space == 'tcp_pose':
        previous = None
        for row in rows:
            for key in ('state','action'):
                q=np.asarray(row[key][3:7],dtype=float); norm=np.linalg.norm(q)
                if abs(norm-1)>1e-4:raise ValueError('四元数不是单位旋转')
                q/=norm
                ref=previous if key=='state' else np.asarray(row['state'][3:7])
                if ref is not None and np.dot(q,ref)<0:q=-q
                row[key][3:7]=q.tolist()
            previous=np.asarray(row['state'][3:7])
    jumps=[]
    for window in capture_windows[1:]:
        i=window['from_frame'];left,right=rows[i-1],rows[i]
        jump=dict(right_frame=i,source_gap_s=float(right['t']-left['t']))
        for key in ('state','action'):
            a,b=np.asarray(left[key]),np.asarray(right[key])
            metrics={'gripper_fraction':float(abs(b[7]-a[7]))}
            if action_space=='tcp_pose':
                metrics.update(position_m=float(np.linalg.norm(b[:3]-a[:3])),rotation_rad=float(2*np.arccos(np.clip(abs(np.dot(a[3:7],b[3:7])),0,1))))
            else:metrics['max_joint_rad']=float(np.max(abs(b[:7]-a[:7])))
            jump[key]=metrics
        jumps.append(jump)
    return rows,capture_windows,[dict(segment_index=0,from_frame=0,to_frame=len(rows))],jumps
