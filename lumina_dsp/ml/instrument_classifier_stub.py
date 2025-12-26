# v3/lumina_dsp/ml/instrument_classifier_stub.py
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class MLClassifierConfig:
    """
    Конфиг каркаса ML.

    enabled:
      Пока мы рефакторим базу — по умолчанию False, чтобы поведение системы
      не изменилось (важно для проверки работоспособности).

    max_queue:
      Ограниченная очередь. Если ML не успевает — мы ДРОПАЕМ фреймы,
      но аудио/DSP не страдают.
    """
    enabled: bool = False
    model_name: str = "instrument_stub_v0"
    max_queue: int = 64

    # throttling telemetry, чтобы не спамить UI (когда будет настоящая модель)
    max_events_hz: float = 20.0


class InstrumentClassifierStub:
    """
    Заглушка ML-классификатора.

    Интеграционный контракт:
    - enqueue_pcm() вызывается из DSP loop (НЕ из audio callback)
    - enqueue_pcm() НЕ блокирует
    - worker работает в отдельном потоке
    - результаты публикуются через publish_fn(msg)
    """

    def __init__(
        self,
        publish_fn: Callable[[Dict[str, Any]], None],
        cfg: Optional[MLClassifierConfig] = None,
    ) -> None:
        self.cfg = cfg or MLClassifierConfig()
        self._publish_fn = publish_fn

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=int(self.cfg.max_queue))
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

        self._last_emit_t = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def start(self) -> None:
        """
        Запускает worker-thread.
        """
        if self._th is not None:
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, name="InstrumentClassifierStub", daemon=True)
        self._th.start()

    def shutdown(self) -> None:
        """
        Останавливает worker-thread.
        """
        self._stop.set()
        th = self._th
        self._th = None
        if th is not None and th.is_alive():
            try:
                th.join(timeout=0.5)
            except Exception:
                pass

        # чистим очередь
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass

    def enqueue_pcm(self, pcm_mono_f32: np.ndarray) -> bool:
        """
        Non-blocking enqueue.

        pcm_mono_f32:
          ожидается mono float32, shape (n,)
          (downmix/downsample делается снаружи, в DSP loop)

        Возвращает True если положили, False если дропнули.
        """
        if not self.enabled:
            return False

        if pcm_mono_f32 is None:
            return False

        x = np.asarray(pcm_mono_f32, dtype=np.float32)
        if x.ndim != 1:
            # пытаемся привести к mono
            x = x.reshape(-1).astype(np.float32, copy=False)

        try:
            self._q.put_nowait(x)
            return True
        except queue.Full:
            # drop, чтобы не влиять на realtime-аудио
            return False

    # ---------------- internal ----------------

    def _run(self) -> None:
        """
        Worker thread: пока без модели.
        Ничего тяжелого, просто “скелет”.
        """
        while not self._stop.is_set():
            try:
                x = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Заглушка: ничего не классифицируем.
            # Когда подключим модель — здесь будет feature extraction + inference.

            # Чтобы не менять текущее поведение системы при рефакторинге,
            # по умолчанию cfg.enabled=False, значит сюда вообще не попадем.
            # Но если включили — можно выдавать редкие "test" события для проверки пайпа.
            self._maybe_emit_test_event()

    def _maybe_emit_test_event(self) -> None:
        # throttle
        now = time.time()
        min_dt = 1.0 / max(1e-6, float(self.cfg.max_events_hz))
        if (now - self._last_emit_t) < min_dt:
            return
        self._last_emit_t = now

        msg = {
            "type": "ai_classifier_event",
            "payload": {
                "label": "test",
                "confidence": 0.0,
                "model": self.cfg.model_name,
            },
            "ts": now,
        }
        try:
            self._publish_fn(msg)
        except Exception:
            # Никогда не даём ML сломать DSP/аудио/WS
            pass
