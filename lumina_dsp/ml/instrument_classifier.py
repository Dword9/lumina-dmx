# FILE: v4/lumina_dsp/ml/instrument_classifier.py
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from .instrument_classifier_stub import MLClassifierConfig
from .metrics import MLMetrics
from .queue import DropQueue


class InstrumentClassifier:
    """
    v4 ML facade.

    Шаг 3:
    - bounded queue + metrics + drop policy
    - отдельный worker-thread, который читает очередь и эмитит synthetic ai_classifier_event
    - аудио/DSP поток не блокируем и не тормозим

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

        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._started = False
        self._log_ts: Dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        # cfg.enabled — канонично
        return bool(getattr(self.cfg, "enabled", False))

    def start(self) -> None:
        self._started = True
        self.maybe_start_worker()

    def shutdown(self) -> None:
        if not self._started:
            return
        self._started = False

        self._stop.set()
        th = self._th
        self._th = None
        if th is not None and th.is_alive():
            try:
                th.join(timeout=0.5)
            except Exception:
                pass

        # drain queue after stopping to avoid stale buffers
        try:
            while self._q.get_nowait() is not None:
                pass
        except Exception:
            pass

    def maybe_start_worker(self) -> None:
        """
        Ensure worker state follows cfg.enabled without blocking hot paths.
        """
        th = self._th

        if not self.enabled:
            if th is not None and th.is_alive():
                self._stop.set()
            return

        if th is None or not th.is_alive():
            self._stop.clear()
            self._th = threading.Thread(target=self._run, name="InstrumentClassifier", daemon=True)
            self._th.start()

    def enqueue_pcm(self, pcm_mono_f32: np.ndarray) -> bool:
        """
        Non-blocking ingestion point.
        Сейчас кладём в bounded queue и НЕ делаем inference здесь.
        """
        self.maybe_start_worker()

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

        if ok:
            self._rate_limited_log(
                "enqueue",
                logging.DEBUG,
                "[ML] enqueue ok size=%d/%d",  # args deferred below
                args=(self._q.qsize(), self._q.maxsize()),
            )

        return ok

    # Утилита для будущего runner'а (шаг 3)
    def _try_get_frame(self) -> Optional[np.ndarray]:
        return self._q.get_nowait()

    # ---------------- internal runner ----------------
    def _run(self) -> None:
        """
        Worker thread: читает очередь с таймаутом и делает лёгкую евристику по энергии.
        Никаких блокировок/ожиданий на аудио пути.
        """
        logging.info("[ML] worker started (enabled=%s, qmax=%d)", self.enabled, self._q.maxsize())
        try:
            while not self._stop.is_set():
                frame = self._q.get(timeout=0.1)
                if frame is None:
                    self._maybe_log_debug_snapshot()
                    continue

                try:
                    self._process_frame(frame)
                except Exception:
                    # ML side-chain не должен влиять на DSP
                    continue
                self._maybe_log_debug_snapshot()
        finally:
            logging.info("[ML] worker exiting")

    def _process_frame(self, frame: np.ndarray) -> None:
        if frame.size == 0:
            return
        if not self.enabled:
            return

        # Лёгкая метрика: RMS энергии
        energy = float(np.sqrt(np.mean(frame.astype(np.float32, copy=False) ** 2)))

        # Простая евристика: высокий энерджи -> "snare", иначе noop-события с низким confidence
        threshold = 0.15
        if energy >= threshold:
            label = "snare"
            confidence = float(min(1.0, energy / 0.8))
        else:
            label = "noop"
            confidence = float(min(0.25, energy * 2.0))

        # Rate limit (~<=15 Hz)
        now = time.time()
        max_hz = min(15.0, float(getattr(self.cfg, "max_events_hz", 10.0) or 10.0))
        min_dt = 1.0 / max(1e-6, max_hz)
        if (now - self.metrics.last_event_ts) < min_dt:
            return

        msg = {
            "type": "ai_classifier_event",
            "payload": {"label": label, "confidence": confidence, "model": self.cfg.model_name},
            "ts": now,
        }

        try:
            self._publish(msg)
            self.metrics.mark_event()
        except Exception:
            # side-chain, не ломаем основной поток
            self.metrics.last_event_ts = now

        self._rate_limited_log(
            "publish",
            logging.INFO,
            "[ML] publish label=%s conf=%0.3f q=%d/%d",  # args deferred below
            args=(label, confidence, self._q.qsize(), self._q.maxsize()),
        )

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
            "queue_stats": {
                "put_ok": self._q.stats.put_ok,
                "put_drop": self._q.stats.put_drop,
                "get_ok": self._q.stats.get_ok,
            },
        }

    def _rate_limited_log(
        self, key: str, level: int, msg: str, *, args: tuple[Any, ...] | None = None, interval: float = 5.0
    ) -> None:
        now = time.time()
        last = self._log_ts.get(key, 0.0)
        if (now - last) < interval:
            return
        self._log_ts[key] = now
        try:
            if args:
                logging.log(level, msg, *args)
            else:
                logging.log(level, msg)
        except Exception:
            pass

    def _maybe_log_debug_snapshot(self) -> None:
        if not self.enabled:
            return
        snap = None
        if (time.time() - self._log_ts.get("diag", 0.0)) >= 5.0:
            snap = self.debug_snapshot()
        if snap is None:
            return
        self._log_ts["diag"] = time.time()
        try:
            logging.debug("[ML] diag %s", snap)
        except Exception:
            pass
