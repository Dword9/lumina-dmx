# v3/lumina_dsp/audio/device.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore


class InputDeviceError(RuntimeError):
    pass


@dataclass
class InputDeviceConfig:
    """
    Настройки входного потока.

    NOTE:
    - blocksize можно оставить 0 (пусть sounddevice решает),
      но для DSP удобнее фиксировать блок.
    """
    channels: int = 1
    samplerate: int = 48000
    blocksize: int = 0  # 0 = auto
    dtype: str = "float32"


class InputDevice:
    """
    Драйвер input audio через sounddevice.InputStream.

    enqueue_fn(chunk: np.ndarray) -> None
      будет вызван в event loop (через call_soon_threadsafe),
      должен быть максимально быстрым (обычно put_nowait в asyncio.Queue).
    """

    def __init__(
        self,
        loop: Any,
        enqueue_fn: Callable[[np.ndarray], None],
        cfg: Optional[InputDeviceConfig] = None,
    ) -> None:
        if sd is None:
            raise InputDeviceError("sounddevice is not available")

        self.loop = loop
        self.enqueue_fn = enqueue_fn
        self.cfg = cfg or InputDeviceConfig()

        self._stream: Optional["sd.InputStream"] = None
        self._device: Optional[int] = None
        self._closed = False

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @property
    def device_index(self) -> Optional[int]:
        return self._device

    def open(self, device: Optional[int] = None) -> None:
        """
        Открывает InputStream на указанном device.

        Важно:
        - callback НЕ блокирует
        - chunk отправляется в asyncio loop через call_soon_threadsafe
        """
        if self._closed:
            raise InputDeviceError("InputDevice is closed")

        if self._stream is not None:
            # уже открыт
            return

        dev = device
        self._device = dev

        ch = int(self.cfg.channels)
        sr = int(self.cfg.samplerate)
        bs = int(self.cfg.blocksize)

        if ch <= 0 or sr <= 0:
            raise ValueError("channels/samplerate must be positive")

        def _cb(indata, frames, time_info, status) -> None:  # sounddevice callback
            try:
                # sounddevice даёт numpy view (float32)
                x = np.asarray(indata, dtype=np.float32)

                # приводим форму к (frames, channels)
                if x.ndim == 1:
                    x = x[:, None]

                # НЕ делаем тяжёлых вещей в callback.
                # Только enqueue в event loop.
                self.loop.call_soon_threadsafe(self.enqueue_fn, x)
            except Exception:
                # Никогда не роняем callback
                return

        self._stream = sd.InputStream(
            device=dev,
            channels=ch,
            samplerate=sr,
            dtype=self.cfg.dtype,
            callback=_cb,
            blocksize=bs,
        )
        self._stream.start()

    def close(self) -> None:
        """
        Закрывает input stream.
        """
        st = self._stream
        self._stream = None
        self._device = None

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
