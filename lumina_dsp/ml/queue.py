# FILE: v4/lumina_dsp/ml/queue.py
from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class DropQueueStats:
    put_ok: int = 0
    put_drop: int = 0
    get_ok: int = 0


class DropQueue(Generic[T]):
    """
    Bounded non-blocking queue:
    - put_nowait(): drop on full, never blocks
    - get_nowait(): for worker thread (optional)
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._q: "queue.Queue[T]" = queue.Queue(maxsize=maxsize)
        self.stats = DropQueueStats()

    def put_nowait(self, item: T) -> bool:
        try:
            self._q.put_nowait(item)
            self.stats.put_ok += 1
            return True
        except queue.Full:
            self.stats.put_drop += 1
            return False

    def get_nowait(self) -> Optional[T]:
        try:
            item = self._q.get_nowait()
            self.stats.get_ok += 1
            return item
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._q.qsize()

    def maxsize(self) -> int:
        return self._q.maxsize
