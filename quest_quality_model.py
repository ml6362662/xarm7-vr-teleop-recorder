"""Presentation of recorded quality evidence; never changes cleaning decisions."""
import json
import math
from pathlib import Path

STATES = {'rejected':('不合格','#ff8c8c'), 'warning':('需复核','#fbbf24'),
          'accepted':('通过','#65d6a4'), 'excluded':('未纳入','#aab4c4'),
          'unknown':('未完成检查','#aab4c4')}


def severity(item):
    decision=item.get('decision')
    if decision=='accepted':
        return 'warning' if item.get('warnings') else 'accepted'
    return decision if decision in ('rejected','excluded') else 'unknown'


def read_report(path):
    report=json.loads(Path(path).read_text())
    if not isinstance(report,dict) or not isinstance(report.get('source'),str) or not isinstance(report.get('episodes'),list):
        raise ValueError('质量报告格式无效')
    seen=set()
    for item in report['episodes']:
        if not isinstance(item,dict) or not isinstance(item.get('episode_id'),str) or item['episode_id'] in seen:
            raise ValueError('轨迹标识缺失或重复')
        if not isinstance(item.get('reason',''),str) or not isinstance(item.get('warnings',[]),list) or any(not isinstance(w,str) for w in item.get('warnings',[])):
            raise ValueError('异常原因格式无效')
        seen.add(item['episode_id'])
    return report


def matches(report,root):
    return root is not None and Path(report['source']).resolve()==Path(root).resolve()


def counts(report):
    result={key:0 for key in STATES}
    for item in report['episodes']:result[severity(item)]+=1
    return result


def visible_episodes(report,mode):
    order={'rejected':0,'warning':1,'unknown':2,'excluded':3,'accepted':4}
    return sorted((item for item in report['episodes'] if mode=='all' or
        (severity(item) in ('rejected','warning','unknown') if mode=='problems' else severity(item)==mode)),
        key=lambda item:(order[severity(item)],item['episode_id']))


def reasons(item):
    return '\n'.join([item.get('reason','')]+['注意：'+w for w in item.get('warnings',[]) ]).strip() or '报告未提供原因'


def advice(item):
    text=reasons(item)
    if severity(item)=='excluded':
        return '这是筛选结果，不代表数据损坏。核对任务结果及保留状态；失败但完整的轨迹可显式勾选包含后重新检查。'
    tips=[]
    for terms,tip in (
        (('图像','相机','丢帧','缺帧','重复'), '核对相机帧率、连接和图像完整性；按相机实际帧率检查，缺失或损坏的数据需重新采集。'),
        (('对齐','间隔','时间','时间戳'), '检查图像与机械臂采样时间、采集丢帧及暂停边界，再按当前 FPS 重新检查。'),
        (('头显','180 ms'), '回看是否存在跟随迟滞，检查头显追踪和传输延迟；警告需人工复核。'),
        (('SDK','观测'), '检查机器人观测读取耗时和状态新鲜度；主机时间对齐不能证明硬件同步。'),
        (('片段过短','帧不足'), '增加连续有效操作时长，重新采集；不要删掉结尾后继续沿用成功标签。'),
        (('会话','版本'), '同一条采集保持同一控制会话和参数；调参或重连后开始新一条。'),
        (('任务描述',), '补充明确任务描述后重新检查。')):
        if any(term in text for term in terms):tips.append(tip)
    if tips:return '\n'.join('• '+tip for tip in dict.fromkeys(tips))
    return '结构检查通过；仍需人工确认任务质量。' if severity(item)=='accepted' else '按上述原始错误核对该条记录；保留原始数据，修复采集原因后重新检查或重新采集。'


def details(item):
    metrics=[]
    for key,title,unit in (
        ('frames','有效帧',''),('segments','有效片段',''),('playback_duration_s','有效回放时长','s'),
        ('max_robot_offset_ms','最大机械臂采样偏差','ms'),('max_camera_offset_ms','最大图像对齐偏差','ms'),
        ('max_observation_age_ms','最大观测发布年龄','ms'),('max_observation_read_window_ms','最大 SDK 读取窗口','ms'),
        ('max_headset_age_ms','最大头显输入年龄','ms')):
        value=item.get(key)
        if type(value) in (int,float) and math.isfinite(value):metrics.append(f'{title}：{value:g} {unit}'.rstrip())
    return (f"轨迹：{item['episode_id']}\n质量：{STATES[severity(item)][0]}\n任务：{item.get('task','—')}\n\n"
        +'问题 / 检查依据\n'+reasons(item)+'\n\n指标\n'+('\n'.join(metrics) or '检查未产生可用指标')
        +'\n\n建议\n'+advice(item)+'\n\n定位范围：整条轨迹。当前报告不含精确异常帧时间；回看从本条起点开始。'
        +'\n检查可能在首个硬错误处停止，列出的原因不保证覆盖该条所有异常。')
