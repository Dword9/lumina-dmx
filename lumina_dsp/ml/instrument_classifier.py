# FILE: v4/lumina_dsp/ml/instrument_classifier.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .instrument_classifier_stub import InstrumentClassifierStub, MLClassifierConfig
from .metrics import MLMetrics
from .queue import DropQueue


class InstrumentClassifier:
    """
    v4 ML facade.

    Шаг 2:
    - добавляем bounded queue + metrics + drop policy
    - НЕ добавляем worker / inference (пока)
    - поведение для UI не ломаем: события (если нужны) может продолжать слать stub,
      но ingestion слой уже становится "реальным" и безопасным.

    ВАЖНО: enqueue_pcm() никогда не блокирует.
    """

    def __init__(
        self,
        publish_fn: Callable[[Dict[str, Any]], None],
        cfg: Optional[MLClassifierConfig] = None,
    ) -> None:
        self.cfg = cfg or MLClassifierConfig()
        self._publish = publish_fn
        self.metrics = MLMetrics()

        # bounded queue (drop on full)
        # можно потом вынести в cfg, но сейчас безопасный дефолт
        max_q = int(getattr(self.cfg, "max_queue", 8) or 8)
        self._q: DropQueue[np.ndarray] = DropQueue(maxsize=max_q)

        # Пока оставляем stub как "processor", но вызывать его будем уже не из enqueue,
        # а на следующем шаге из worker.
        self._stub = InstrumentClassifierStub(publish_fn=publish_fn, cfg=self.cfg)

        self._started = False

    @property
    def enabled(self) -> bool:
        # cfg.enabled — канонично
        return bool(getattr(self.cfg, "enabled", False))

    def start(self) -> None:
        if self._started:
            return
        self._stub.start()
        self._started = True

    def shutdown(self) -> None:
        if not self._started:
            return
        self._stub.shutdown()
        self._started = False

    def enqueue_pcm(self, pcm_mono_f32: np.ndarray) -> bool:
        """
        Non-blocking ingestion point.
        Сейчас кладём в bounded queue и НЕ делаем inference здесь.
        """
        if not self.enabled:
            return False

        # Минимальная валидация без тяжёлых операций
        if pcm_mono_f32 is None:
            return False
        if not isinstance(pcm_mono_f32, np.ndarray):
            return False
        if pcm_mono_f32.dtype != np.float32:
            # Не конвертим тут: чтобы не делать лишнюю работу в DSP loop.
            # На следующих шагах конверт будет в worker.
            return False

        ok = self._q.put_nowait(pcm_mono_f32)
        self.metrics.mark_enqueue(ok)

        # На шаге 2 мы НЕ читаем очередь. Она будет читаться на шаге 3 (runner thread).
        return ok

    # Утилита для будущего runner'а (шаг 3)
    def _try_get_frame(self) -> Optional[np.ndarray]:
        return self._q.get_nowait()

    def debug_snapshot(self) -> Dict[str, Any]:
        """
        Для локального логирования/диагностики (UI можно не трогать).
        """
        return {
            "enabled": self.enabled,
            "queue": {"size": self._q.qsize(), "max": self._q.maxsize()},
            "metrics": {
                "enqueued_frames": self.metrics.enqueued_frames,
                "dropped_frames": self.metrics.dropped_frames,
                "events_sent": self.metrics.events_sent,
                "last_enqueue_ts": self.metrics.last_enqueue_ts,
                "last_event_ts": self.metrics.last_event_ts,
            },
        }
