"""Select complete temporal windows without crossing pauses in joined episodes.

Apply to a full exported dataset, before train/validation sampling. This does not
split stored episodes, invent pause frames, or change their outcome labels.
"""
import json
from pathlib import Path


def training_indices(export_root, horizon, history=0):
    """Indices for [anchor-history, anchor+horizon), in full dataset row order."""
    if type(horizon) is not int or horizon < 1 or type(history) is not int or history < 0:
        raise ValueError('horizon 必须为正整数，history 必须为非负整数')
    manifest=json.loads((Path(export_root)/'export_manifest.json').read_text())
    indices=[]
    offset=0
    for expected,episode in enumerate(manifest['episodes']):
        if episode['episode_index']!=expected:
            raise ValueError('导出 episode 顺序不完整')
        end=0
        for window in episode['training_windows']:
            start,stop=window['from_frame'],window['to_frame']
            if start!=end or stop<=start or stop>episode['frames']:
                raise ValueError('训练窗口边界无效')
            indices.extend(range(offset+start+history,offset+stop-horizon+1))
            end=stop
        if end!=episode['frames']:
            raise ValueError('训练窗口未覆盖导出轨迹')
        offset+=episode['frames']
    return indices


def restrict_training_windows(dataset, export_root, horizon, history=0):
    """Wrap a FULL LeRobot dataset after configuring its delta_timestamps.

    horizon/history are frame counts including the current frame, not seconds.
    The caller must match them to all action/observation temporal offsets.
    Do not use an already episode-filtered dataset: indices would refer elsewhere.
    """
    from torch.utils.data import Subset
    manifest=json.loads((Path(export_root)/'export_manifest.json').read_text())
    if len(dataset)!=sum(e['frames'] for e in manifest['episodes']):
        raise ValueError('请使用完整导出数据集，再按训练窗口筛选')
    indices=training_indices(export_root,horizon,history)
    if not indices:
        raise ValueError('没有足够长的连续录制区间，请减小动作窗口或补采数据')
    return Subset(dataset,indices)
