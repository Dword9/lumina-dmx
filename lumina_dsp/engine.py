# FILE: v4/lumina_dsp/engine.py
from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import platform
import time
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
import sounddevice as sd

from lumina_dsp.config import DSPConfig
from lumina_dsp.dsp.core import DSPCore, DSPParams
from lumina_dsp.audio.file_player import FilePlayer, FileRegistry

from .state import RuntimeState, EventCallback
from .telemetry import emit as telemetry_emit, emit_file_status as telemetry_emit_file_status
from .control import handle_control as handle_control_router

from .ml.instrument_classifier import InstrumentClassifier
from .ml.instrument_classifier_stub import MLClassifierConfig




class AudioEngine:
    """
    v4 Engine (refactor-ready):
    - НЕ поднимает WS/HTTP
    - handle_control(msg) -> response msg для UI (контракт в control.py)
    - telemetry наружу через set_event_callback(cb) (контракт fixed)
    - realtime: никаких блокировок в audio callback
    """

    def __init__(self, dsp: DSPConfig | None = None, ml_cfg: MLClassifierConfig | None = None):
        self.dsp_cfg = dsp or DSPConfig()

        self.state = RuntimeState(
            running=False,
            source_id="device:default",
        )
        # back-compat если state.py ещё не расширен
        if not hasattr(self.state, "sr"):
            setattr(self.state, "sr", 48000)
        if not hasattr(self.state, "channels"):
            setattr(self.state, "channels", 1)

        self.dsp_core = DSPCore(
            DSPParams(
                fps=self.dsp_cfg.fps,
                block=self.dsp_cfg.block,
                fft_size=self.dsp_cfg.fft_size,
                num_bands=self.dsp_cfg.num_bands,
                mock_events=self.dsp_cfg.mock_events,
            )
        )

        self.registry = FileRegistry()
        self.file_player = FilePlayer(self.registry, block=self.dsp_cfg.block)

        self._q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Input stream (device:* / loopback:*)
        self._in_stream: Optional[sd.InputStream] = None

        # Output stream (monitor for file playback)
        self._out_stream: Optional[sd.OutputStream] = None
        self._monitor_enabled: bool = True
        # Чуть больше headroom для аудиомонитора: на коротких лагax ОС буфер
        # опустевал и в слышимом сигнале появлялись щелчки. 1.8s даёт запас
        # без изменения контрактов или задержки UI.
        self._monitor_headroom_sec: float = 1.8

        # Realtime monitor ring buffer
        self._mon_rb: Optional[_MonitorRingBuffer] = None
        self._mon_sr: int = 0
        self._mon_ch: int = 0

        # Analysis-only knobs (НЕ влияют на audible output)
        self._analysis_gain: float = 200.0
        self._gate_rms: float = 0.0

        self._event_cb: Optional[EventCallback] = None

        self._dsp_task: Optional[asyncio.Task] = None
        self._closing = False

        # loopback cache (optional)
        self._loopback_cached: Optional[Tuple[str, int, int]] = None  # (name, sr, ch)

        # --- ML stub cfg ---
        self._ml_cfg = ml_cfg or MLClassifierConfig(enabled=True)

        # --- ML facade (внутри пока stub) ---
        self._ml = InstrumentClassifier(publish_fn=self._emit_sync, cfg=self._ml_cfg)
        self._ml.start()

        # One-time best-effort boost: приоритет процесса повыше, чтобы frontend
        # или сторонние heavy-потоки не отбирали квант у аудио-потоков PortAudio.
        self._priority_boosted: bool = False


    # -------------------- lifecycle --------------------

    def set_event_callback(self, cb: EventCallback | None) -> None:
        self._event_cb = cb

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._boost_process_priority()
        if self._dsp_task is None or self._dsp_task.done():
            self._dsp_task = asyncio.create_task(self._dsp_loop(), name="lumina.dsp_loop")

    async def close(self) -> None:
        self._closing = True
        self.state.running = False

        try:
            await self.file_player.stop()
        except Exception:
            pass

        self._close_input_stream()
        self._close_output_stream()

        try:
            self._ml.shutdown()
        except Exception:
            pass

        if self._dsp_task is not None:
            self._dsp_task.cancel()
            try:
                await self._dsp_task
            except Exception:
                pass
            self._dsp_task = None

    # -------------------- telemetry helpers --------------------
    # ВАЖНО: оставляем как в рабочей версии, иначе UI теряет telemetry.

    def _emit_sync(self, msg: Dict[str, Any]) -> None:
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(asyncio.create_task, telemetry_emit(self._event_cb, msg))
        except Exception:
            pass

    async def emit(self, msg: Dict[str, Any]) -> None:
        await telemetry_emit(self._event_cb, msg)

    async def emit_file_status(self, req_id: Optional[str] = None) -> None:
        await telemetry_emit_file_status(self._event_cb, self.file_player, req_id=req_id)

    # -------------------- control entrypoint --------------------

    async def handle_control(self, msg: Dict[str, Any]) -> Dict[str, Any] | None:
        return await handle_control_router(self, msg)

    # -------------------- queue helpers --------------------

    def _q_put_nowait(self, x: np.ndarray) -> None:
        try:
            self._q.put_nowait(x)
        except asyncio.QueueFull:
            pass

    def _drain_queue(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except asyncio.QueueEmpty:
            pass

    def _boost_process_priority(self) -> None:
        """
        Best-effort: поднять приоритет процесса, чтобы проигрыватель/PortAudio
        реже вытеснялись тяжёлыми потоками (например, от UI).
        Делаем один раз и без исключений — стабильность аудио важнее.
        """

        if self._priority_boosted:
            return

        self._priority_boosted = True

        try:
            # Windows: HIGH_PRIORITY_CLASS (0x00000080)
            if platform.system().lower().startswith("win"):
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                ctypes.windll.kernel32.SetPriorityClass(handle, 0x00000080)
                return

            # POSIX: снизим nice (более высокий приоритет -> меньше значение)
            if hasattr(os, "nice"):
                try:
                    current = os.nice(0)
                except Exception:
                    current = None
                try:
                    if current is None or current > -5:
                        os.nice(-5)
                except Exception:
                    pass
        except Exception:
            # Не ломаем поток аудио — если не удалось, просто продолжаем.
            pass

    # -------------------- input device I/O --------------------

    def _device_callback(self, indata, frames, time_info, status) -> None:
        try:
            x = np.asarray(indata, dtype=np.float32)
            if x.ndim == 1:
                x = x[:, None]
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._q_put_nowait, x)
        except Exception:
            return

    def _open_input_stream(self, source_id: str) -> None:
        self._close_input_stream()

        idx_s = source_id.split(":", 1)[1]
        device = None if idx_s == "default" else int(idx_s)

        if device is not None:
            info = sd.query_devices(device)
            sr = int(info.get("default_samplerate", 48000) or 48000)
            ch = int(info.get("max_input_channels", 1) or 1)
        else:
            sr = 48000
            ch = 1

        self.state.sr = int(sr)
        self.state.channels = int(max(1, min(2, ch)))

        self._in_stream = sd.InputStream(
            device=device,
            channels=self.state.channels,
            samplerate=self.state.sr,
            dtype="float32",
            callback=self._device_callback,
            blocksize=0,
        )
        self._in_stream.start()

    def _resolve_loopback_wasapi(self) -> Tuple[int, str, int, int, Any]:
        """
        Windows Desktop Audio:
        Используем WASAPI loopback через sd.WasapiSettings(loopback=True),
        открывая InputStream на default OUTPUT device.

        Возвращает:
          (device_index, name, sr, ch, extra_settings)
        """
        # sounddevice должен иметь WasapiSettings (иначе loopback не поддержан этим билдом)
        WasapiSettings = getattr(sd, "WasapiSettings", None)
        if WasapiSettings is None:
            raise RuntimeError("sounddevice.WasapiSettings is not available (WASAPI loopback not supported)")

        # default output device index
        out_dev = None
        try:
            out_dev = sd.default.device[1]
        except Exception:
            out_dev = None

        if out_dev is None or int(out_dev) < 0:
            raise RuntimeError("No default output device (needed for WASAPI loopback)")

        out_dev = int(out_dev)
        out_info = sd.query_devices(out_dev)

        name = str(out_info.get("name", "Desktop Audio") or "Desktop Audio")
        sr = int(out_info.get("default_samplerate", 48000) or 48000)

        # Для loopback обычно хочется 2 канала
        max_out_ch = int(out_info.get("max_output_channels", 2) or 2)
        ch = int(max(1, min(2, max_out_ch)))

        extra = WasapiSettings(loopback=True)
        return out_dev, f"{name} (WASAPI loopback)", sr, ch, extra

    def _resolve_loopback_fallback_input(self) -> Tuple[Optional[int], str, int, int]:
        """
        Без WASAPI пробуем использовать дефолтный input (например, виртуальный кабель).
        Не поднимаем исключения, чтобы loopback не рушил остальной поток.
        """
        dev = None
        name = "Desktop Audio (virtual input)"
        sr = 48000
        ch = 2

        try:
            try:
                dev = sd.default.device[0]
            except Exception:
                dev = None

            if dev is not None and int(dev) >= 0:
                dev = int(dev)
                info = sd.query_devices(dev)
                name = str(info.get("name", name) or name)
                sr = int(info.get("default_samplerate", sr) or sr)
                ch = max(1, min(2, int(info.get("max_input_channels", ch) or ch)))
        except Exception:
            dev = None

        return dev, name, sr, ch

    def _resolve_loopback(self) -> Tuple[Optional[int], str, int, int, Optional[Any]]:
        try:
            return self._resolve_loopback_wasapi()
        except Exception:
            dev, name, sr, ch = self._resolve_loopback_fallback_input()
            return dev, name, sr, ch, None

    def _open_loopback_stream(self) -> None:
        """
        SourceId: loopback:default (контракт с UI фиксированный!)
        """
        self._close_input_stream()

        dev, name, sr, ch, extra = self._resolve_loopback()
        self._loopback_cached = (name, sr, ch)

        self.state.sr = int(sr)
        self.state.channels = int(ch)

        kwargs: Dict[str, Any] = {
            "device": int(dev) if dev is not None else None,
            "channels": self.state.channels,
            "samplerate": self.state.sr,
            "dtype": "float32",
            "callback": self._device_callback,
            "blocksize": 0,
        }
        if extra is not None:
            kwargs["extra_settings"] = extra

        self._in_stream = sd.InputStream(**kwargs)
        self._in_stream.start()

    def _close_input_stream(self) -> None:
        st = self._in_stream
        self._in_stream = None
        if st is not None:
            try:
                st.stop()
            except Exception:
                pass
            try:
                st.close()
            except Exception:
                pass

    # -------------------- output monitor I/O --------------------

    def _monitor_callback(self, outdata, frames, time_info, status) -> None:
        # audio callback: NEVER block, NEVER allocate heavy stuff
        try:
            rb = self._mon_rb
            if rb is None:
                outdata.fill(0)
                return
            rb.read_into(outdata)  # fills with zeros on underrun
        except Exception:
            outdata.fill(0)

    def _ensure_output_stream(self, sr: int, ch: int) -> None:
        if not self._monitor_enabled:
            return

        sr = int(sr)
        ch = int(ch)
        if sr <= 0 or ch <= 0:
            return

        if self._out_stream is not None and self._mon_sr == sr and self._mon_ch == ch:
            return

        self._close_output_stream()

        self._mon_sr = sr
        self._mon_ch = ch

        # IMPORTANT: provide headroom. A bit more than a second reduces underruns.
        cap_frames = max(2048, int(sr * self._monitor_headroom_sec))
        self._mon_rb = _MonitorRingBuffer(channels=ch, capacity_frames=cap_frames)

        self._out_stream = sd.OutputStream(
            samplerate=sr,
            channels=ch,
            dtype="float32",
            callback=self._monitor_callback,
            blocksize=0,
        )
        self._out_stream.start()

    def _close_output_stream(self) -> None:
        st = self._out_stream
        self._out_stream = None
        self._mon_rb = None
        self._mon_sr = 0
        self._mon_ch = 0

        if st is not None:
            try:
                st.stop()
            except Exception:
                pass
            try:
                st.close()
            except Exception:
                pass

    def _monitor_push(self, chunk: np.ndarray, sr: int, ch: int) -> None:
        """
        Writer: called from asyncio context (file feeder).
        MUST be non-blocking, and MUST NOT depend on event loop scheduling.
        """
        if not self._monitor_enabled:
            return
        try:
            self._ensure_output_stream(sr, ch)
            rb = self._mon_rb
            if rb is None:
                return

            x = np.asarray(chunk, dtype=np.float32)
            if x.ndim == 1:
                x = x[:, None]
            if x.ndim != 2 or x.shape[1] != ch:
                return

            rb.write(x)  # non-blocking; drops if full
        except Exception:
            # reset and try again next time
            self._close_output_stream()

    # -------------------- sources --------------------

    def list_sources_payload(self) -> Dict[str, Any]:
        devs = sd.query_devices()
        items = []
        for i, d in enumerate(devs):
            if int(d.get("max_input_channels", 0) or 0) > 0:
                name = d.get("name", f"Device {i}") or f"Device {i}"
                if "ndi" in str(name).lower():
                    continue
                items.append(
                    {
                        "id": f"device:{i}",
                        "kind": "device",
                        "name": name,
                        "channels": int(d.get("max_input_channels", 1) or 1),
                        "sr": int(d.get("default_samplerate", 48000) or 48000),
                    }
                )

        # loopback:default (контракт фиксирован)
        lb_name = "Desktop Audio (loopback)"
        lb_sr = 48000
        lb_ch = 2
        try:
            # обновим кэш, если можем
            dev, name, sr, ch, extra = self._resolve_loopback()
            self._loopback_cached = (name, sr, ch)
            lb_name, lb_sr, lb_ch = name, sr, ch
        except Exception:
            cached = self._loopback_cached
            if cached is not None:
                lb_name, lb_sr, lb_ch = cached

        items.append(
            {
                "id": "loopback:default",
                "kind": "loopback",
                "name": lb_name,
                "channels": int(lb_ch),
                "sr": int(lb_sr),
            }
        )

        for fid, info in self.registry.items():
            items.append(
                {
                    "id": f"file:{fid}",
                    "kind": "file",
                    "name": info.name,
                    "channels": info.channels,
                    "sr": info.sr,
                    "duration": info.duration,
                }
            )

        return {"default": "device:default", "items": items}

    async def set_source(self, source_id: str) -> None:
        self.state.running = False

        self._close_input_stream()
        self._close_output_stream()
        await self.file_player.stop()
        self._drain_queue()

        self.state.source_id = source_id

        if source_id.startswith("device:"):
            self._open_input_stream(source_id)
            return

        if source_id.startswith("file:"):
            fid = source_id.split(":", 1)[1]
            await self.file_player.set_active(fid)
            info = self.registry.get(fid)
            self.state.sr = int(info.sr)
            self.state.channels = max(1, min(2, int(info.channels)))
            return

        if source_id.startswith("loopback:"):
            # contract: loopback:default
            self._open_loopback_stream()
            return

        raise ValueError(f"Unknown sourceId: {source_id}")

    # -------------------- DSP loop --------------------

    async def _dsp_loop(self) -> None:
        last_send = 0.0
        ml_log_ts = 0.0
        while not self._closing:
            if not self.state.running:
                await asyncio.sleep(0.02)
                continue

            if str(self.state.source_id).startswith("file:"):

                async def _on_chunk(data: np.ndarray, sr: int, ch: int) -> None:
                    # 1) monitor push (audible) — independent from dsp/telemetry
                    self._monitor_push(data, sr=sr, ch=ch)
                    # 2) dsp queue (analysis)
                    self._q_put_nowait(data)

                await self.file_player.ensure_task(_on_chunk)

            try:
                x = await asyncio.wait_for(self._q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.clip(x, -1.0, 1.0)

            if self._gate_rms > 0.0:
                rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
                if rms < self._gate_rms:
                    x = np.zeros_like(x)

            x_ana = np.clip(x * float(self._analysis_gain), -1.0, 1.0)
            meter = self.dsp_core.compute_meter(x_ana)

            now = time.time()
            if (now - last_send) >= (1.0 / max(10, int(self.dsp_cfg.fps))):
                await self.emit(
                    {
                        "type": "audio_meter",
                        "payload": {
                            "rms": float(meter.get("rms", 0.0) or 0.0),
                            "peak": float(meter.get("peak", 0.0) or 0.0),
                            "bands": meter.get("bands", []) or [],
                        },
                        "ts": now,
                    }
                )

                bands = meter.get("bands", []) or []
                low = float(bands[0]) if len(bands) > 0 else 0.0
                mid = float(bands[len(bands) // 2]) if len(bands) > 1 else 0.0
                high = float(bands[-1]) if len(bands) > 2 else 0.0
                await self.emit(
                    {
                        "type": "dsp_event",
                        "payload": {
                            "kick": float(min(1.0, max(0.0, low * 1.2))),
                            "snare": float(min(1.0, max(0.0, mid * 1.1))),
                            "hihat": float(min(1.0, max(0.0, high * 1.0))),
                            "energy": float(meter.get("rms", 0.0) or 0.0),
                            "bpm": 0.0,
                        },
                        "ts": now,
                    }
                )

                last_send = now

            # ML stub: enqueue (non-blocking drop)
            try:
                mono = x_ana.mean(axis=1) if x_ana.ndim == 2 else x_ana.reshape(-1)
                self._ml.enqueue_pcm(mono.astype(np.float32, copy=False))
            except Exception:
                pass

            if self._ml.enabled and (time.time() - ml_log_ts) >= 5.0:
                try:
                    logging.debug("[ML] snapshot %s", self._ml.debug_snapshot())
                except Exception:
                    pass
                ml_log_ts = time.time()


class _MonitorRingBuffer:
    """
    Fast ring buffer for monitor output.

    Writer: asyncio thread (file feeder) calls write(x) — non-blocking, may drop.
    Reader: sounddevice callback calls read_into(outdata) — non-blocking, fills zeros on underrun.

    CRITICAL: no asyncio primitives here, no create_task, no await.
    """

    def __init__(self, channels: int, capacity_frames: int):
        self.channels = int(channels)
        self.capacity = int(capacity_frames)
        self._buf = np.zeros((self.capacity, self.channels), dtype=np.float32)

        self._w = 0
        self._r = 0
        self._lock = threading.Lock()
        self._last = np.zeros((1, self.channels), dtype=np.float32)

    def _available_to_read(self) -> int:
        return max(0, self._w - self._r)

    def _available_to_write(self) -> int:
        # keep 1 frame gap
        return max(0, self.capacity - self._available_to_read() - 1)

    def write(self, x: np.ndarray) -> int:
        """
        Non-blocking write:
        - if lock is busy -> drop
        - if buffer is full -> drop excess
        """
        if x.ndim != 2 or x.shape[1] != self.channels:
            return 0

        if not self._lock.acquire(blocking=False):
            return 0
        try:
            n = int(x.shape[0])
            can = self._available_to_write()
            if can <= 0 or n <= 0:
                return 0
            nwrite = min(n, can)

            w0 = self._w % self.capacity
            first = min(nwrite, self.capacity - w0)
            self._buf[w0 : w0 + first] = x[:first]
            remain = nwrite - first
            if remain > 0:
                self._buf[0:remain] = x[first : first + remain]

            self._w += nwrite
            return nwrite
        finally:
            self._lock.release()

    def read_into(self, out: np.ndarray) -> None:
        """
        Fill out with available frames, rest zeros.
        Must be fast for audio callback.
        """
        try:
            out.fill(0)
        except Exception:
            return

        can = self._available_to_read()
        need = int(out.shape[0])
        nread = min(need, can)
        if nread <= 0:
            # nothing available: keep last sample to avoid hard steps
            if need > 0:
                out[:] = self._last
            return

        r0 = self._r % self.capacity
        first = min(nread, self.capacity - r0)
        out[:first] = self._buf[r0 : r0 + first]
        remain = nread - first
        if remain > 0:
            out[first : first + remain] = self._buf[0:remain]

        self._r += nread

        # remember tail to soften underruns; fade remaining zeros toward silence
        try:
            self._last[:] = out[nread - 1 : nread]
            if nread < need:
                tail = need - nread
                fade = np.linspace(1.0, 0.0, num=tail, endpoint=False, dtype=np.float32)[
                    :, None
                ]
                out[nread:] = self._last * fade
        except Exception:
            pass
