# FILE: v4/lumina_dsp/ml/metrics.py
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class MLMetrics:
    # queue/load
    dropped_frames: int = 0
    enqueued_frames: int = 0

    # timings
    last_enqueue_ts: float = 0.0
    last_event_ts: float = 0.0

    # debug counters
    events_sent: int = 0

    def mark_enqueue(self, ok: bool) -> None:
        self.last_enqueue_ts = time.time()
        if ok:
            self.enqueued_frames += 1
        else:
            self.dropped_frames += 1

    def mark_event(self) -> None:
        self.last_event_ts = time.time()
        self.events_sent += 1
