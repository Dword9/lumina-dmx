# v3/lumina_dsp/audio/monitor.py
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore


class MonitorError(RuntimeError):
    pass


class _MonitorRingBuffer:
    """
    Lock-free-ish ring buffer для monitor output.

    IMPORTANT:
    - write() никогда не блокирует
    - read_into() никогда не блокирует (в callback)
    - при underrun callback заполняет нулями
    """

    def __init__(self, channels: int, capacity_frames: int) -> None:
        self.channels = int(channels)
        self.capacity = int(capacity_frames)
        self._buf = np.zeros((self.capacity, self.channels), dtype=np.float32)

        # Индексы в "frames" (монотонно растущие)
        self._w = 0  # write cursor
        self._r = 0  # read cursor

        # Минимальная блокировка: только для write (чтобы не порвать индексы при многопоточности)
        self._lock = threading.Lock()

    def available_to_read(self) -> int:
        return max(0, self._w - self._r)

    def available_to_write(self) -> int:
        # оставляем один frame как safety gap
        return max(0, self.capacity - self.available_to_read() - 1)

    def write(self, x: np.ndarray) -> int:
        """
        Записывает сколько может, остальное дропает.
        Возвращает число записанных frames.
        """
        if x.ndim != 2 or x.shape[1] != self.channels:
            raise ValueError(f"expected shape (frames,{self.channels}), got {x.shape}")

        n = int(x.shape[0])
        if n <= 0:
            return 0

        with self._lock:
            can = self.available_to_write()
            if can <= 0:
                return 0
            nwrite = min(n, can)

            # Пишем с wrap-around
            w0 = self._w % self.capacity
            first = min(nwrite, self.capacity - w0)
            self._buf[w0 : w0 + first] = x[:first]
            remain = nwrite - first
            if remain > 0:
                self._buf[0:remain] = x[first : first + remain]

            self._w += nwrite
            return nwrite

    def read_into(self, out: np.ndarray) -> int:
        """
        Читает в out ровно out.shape[0] frames если есть,
        иначе читает сколько есть и остальное caller заполнит нулями.

        Возвращает число реально прочитанных frames.
        """
        if out.ndim != 2 or out.shape[1] != self.channels:
            raise ValueError(f"expected out shape (frames,{self.channels}), got {out.shape}")

        need = int(out.shape[0])
        if need <= 0:
            return 0

        # Не лочим read — в callback нужно быть максимально быстрым.
        # Допускаем “мягкие” гонки индексов, это безопаснее чем блокировать callback.
        can = self.available_to_read()
        nread = min(need, can)
        if nread <= 0:
            return 0

        r0 = self._r % self.capacity
        first = min(nread, self.capacity - r0)
        out[:first] = self._buf[r0 : r0 + first]
        remain = nread - first
        if remain > 0:
            out[first : first + remain] = self._buf[0:remain]

        self._r += nread
        return nread


@dataclass
class MonitorConfig:
    """
    Настройки monitor output (audible playback).

    buffer_ms:
      размер ring-buffer (в миллисекундах). Должен давать запас по времени,
      чтобы UI/WS/аналитика не влияли на аудио.
    """
    buffer_ms: int = 500  # ~0.5s запас


class OutputMonitor:
    """
    Драйвер monitor output через sounddevice.OutputStream + ring buffer.

    Гарантии:
    - write() не блокирует: если буфер заполнен, данные дропаются
    - callback не блокирует: при underrun выдаёт нули
    """

    def __init__(self, cfg: Optional[MonitorConfig] = None) -> None:
        self.cfg = cfg or MonitorConfig()

        self._stream: Optional["sd.OutputStream"] = None
        self._rb: Optional[_MonitorRingBuffer] = None

        self._sr: Optional[int] = None
        self._ch: Optional[int] = None

        self._closed = False

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @property
    def current_format(self) -> Optional[Tuple[int, int]]:
        if self._sr is None or self._ch is None:
            return None
        return (self._sr, self._ch)

    def ensure(self, samplerate: int, channels: int) -> None:
        """
        Открывает OutputStream если не открыт, либо если формат поменялся.
        """
        if sd is None:
            raise MonitorError("sounddevice is not available")

        if self._closed:
            raise MonitorError("OutputMonitor is closed")

        sr = int(samplerate)
        ch = int(channels)
        if sr <= 0 or ch <= 0:
            raise ValueError("samplerate/channels must be positive")

        if self._stream is not None and self._sr == sr and self._ch == ch:
            return

        # если формат изменился — пересоздаём
        self.close()

        cap_frames = max(2048, int(sr * (self.cfg.buffer_ms / 1000.0)))
        self._rb = _MonitorRingBuffer(channels=ch, capacity_frames=cap_frames)

        def _cb(outdata, frames, time_info, status) -> None:  # sounddevice callback
            # callback должен быть максимально лёгким и НЕ блокировать.
            try:
                rb = self._rb
                if rb is None:
                    outdata.fill(0)
                    return

                tmp = np.empty((frames, ch), dtype=np.float32)
                nread = rb.read_into(tmp)
                if nread < frames:
                    # underrun: заполняем хвост нулями
                    tmp[nread:].fill(0)
                outdata[:] = tmp
            except Exception:
                # Никогда не роняем callback
                outdata.fill(0)

        self._stream = sd.OutputStream(
            samplerate=sr,
            channels=ch,
            dtype="float32",
            callback=_cb,
            blocksize=0, # пусть sounddevice выберет оптимально
            latency=high,
        )
        self._stream.start()
        self._sr = sr
        self._ch = ch

    def write(self, x: np.ndarray, samplerate: int, channels: int) -> int:
        """
        Пишет данные в ring-buffer.

        IMPORTANT:
        - non-blocking
        - если stream не открыт — откроет (ensure)
        - если формат отличается — пересоздаст (ensure)
        """
        if self._closed:
            return 0

        self.ensure(samplerate=samplerate, channels=channels)

        rb = self._rb
        if rb is None:
            return 0

        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        # ожидаем (frames, ch)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError(f"expected 1D/2D float32 array, got shape {x.shape}")

        if x.shape[1] != channels:
            raise ValueError(f"expected {channels} channels, got {x.shape[1]}")

        return rb.write(x)

    def close(self) -> None:
        """
        Закрывает stream и освобождает буфер.
        """
        st = self._stream
        self._stream = None
        self._rb = None
        self._sr = None
        self._ch = None

        if st is not None:
            try:
                st.stop()
            except Exception:
                pass
            try:
                st.close()
            except Exception:
                pass

    def shutdown(self) -> None:
        """
        Одноразовое закрытие навсегда (используется при stop/close движка).
        """
        if self._closed:
            return
        self.close()
        self._closed = True
