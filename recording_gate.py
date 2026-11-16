"""Trigger-controlled recording intervals; no device or file I/O."""


class TriggerCaptureGate:
    def __init__(self, trigger_at_start=False, release_generation_at_start=None):
        self.require_release = trigger_at_start is not False
        self.intervals = []
        self.current = None
        self.last_time = None
        self.release_generation = release_generation_at_start

    def close(self, timestamp):
        if self.current is not None:
            self.current['end_s'] = max(self.current['start_s'], timestamp)
            self.current = None

    def interrupt_input(self):
        # Only data up to the last known-good controller watermark is trusted.
        # In particular, never include delayed camera frames from an outage.
        if self.current is not None:
            self.close(self.last_time + 1e-6)
        self.require_release = True
        # Rebase on the first valid packet. Releases seen during invalid input
        # cannot authorize a resume; a fresh release/repress is required.
        self.release_generation = None

    def update(self, sample):
        now = sample['monotonic_s']
        hand = sample.get('hand') or {}
        pressed = hand.get('trigger_pressed')
        generation = hand.get('trigger_release_generation')
        # A release/repress can occur between snapshots. Never bridge that gap.
        if generation is not None and self.release_generation is not None and generation != self.release_generation:
            self.close(self.last_time + 1e-6 if self.last_time is not None else now)
            self.require_release = False
        self.release_generation = generation
        if pressed is False:
            self.require_release = False
        permitted = (not self.require_release and pressed is True and sample.get('source_valid') is True
            and sample.get('mode') == 'motion' and sample.get('active') is True
            and sample.get('command_valid') is True and not sample.get('pause_settling')
            and not sample.get('recovery_active') and not sample.get('error'))
        if permitted:
            if self.current is None:
                self.current = dict(segment_id=len(self.intervals), start_s=now, end_s=None)
                self.intervals.append(self.current)
        else:
            self.close(now)
        self.last_time = now
        return None if self.current is None else self.current['segment_id']

    def segment_at(self, timestamp):
        for interval in reversed(self.intervals):
            if timestamp >= interval['start_s']:
                if interval['end_s'] is None or timestamp < interval['end_s']:
                    return interval['segment_id']
                return None
        return None

    def duration(self, now):
        return sum(max(0,(i['end_s'] if i['end_s'] is not None else now)-i['start_s']) for i in self.intervals)
