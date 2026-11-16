"""UI decisions independent of GTK, robot commands, and recording transports."""
from dataclasses import dataclass
import math


def process_running(process):
    return process is not None and process.poll() is None


def fresh_timestamp(stamp, now, max_age):
    return type(stamp) in (int,float) and math.isfinite(stamp) and 0 <= now-stamp < max_age


@dataclass(frozen=True)
class RecordingActions:
    prepare: bool
    start: bool
    review: bool
    stop: bool
    export: bool
    configure: bool
    edit_task: bool
    organize: bool
    reason: str


def recording_actions(snapshot, *, running, exporting, pending, demo, closed, now):
    """No teleoperation state is an input: data controls cannot change arm state."""
    fresh = running and fresh_timestamp(snapshot.get('monotonic_s'),now,2)
    recording = running and bool(snapshot.get('recording'))
    usable = not demo and not closed
    ready = fresh and not snapshot.get('ready_error',True)
    reason = ('演示模式不操作设备' if demo else '界面已关闭' if closed else
              '正在检查 / 导出，请等待完成' if exporting else
              '请先结束本条采集' if recording else
              '采集状态未更新，请检查采集器' if running and not fresh else
              '等待采集器确认' if pending else
              str(snapshot.get('ready_error') or '') if running and not ready else '')
    return RecordingActions(
        prepare=usable and not running and not exporting,
        start=usable and ready and not recording and not pending and not exporting,
        review=usable and fresh and recording and not pending and bool(snapshot.get('can_review',False)),
        stop=usable and running,  # Closing a stuck recorder remains possible while a reply is pending.
        export=usable and not exporting and not recording and not pending and (not running or fresh),
        configure=not closed and not running and not exporting,
        edit_task=not closed and not recording and not pending,
        organize=usable and not running and not exporting,
        reason=reason)


CAPTURE_LABELS = {
    'idle':'采集待机', 'waiting':'本条已开启 · 等待扳机',
    'recording':'正在记录', 'paused':'采集暂停 · 再按扳机继续',
    'input_paused':'输入中断 · 本条已保留', 'review':'本条冻结 · 等待判定',
}


def capture_summary(snapshot):
    state=snapshot.get('capture_state','idle')
    label=CAPTURE_LABELS.get(state,'采集状态未知 · 请检查状态详情')
    seconds=snapshot.get('recorded_s',0)
    duration=f'{seconds:.1f}' if type(seconds) in (int,float) and math.isfinite(seconds) else '—'
    return f"{label} · {duration} s · 已保存 {snapshot.get('completed',0)} 条"
