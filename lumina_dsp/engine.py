# FILE: v4/lumina_dsp/engine.py
from __future__ import annotations

import asyncio
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

# ML stub (по умолчанию выключен и не меняет поведение)
from .ml.instrument_classifier_stub import InstrumentClassifierStub, MLClassifierConfig


class AudioEngine:
    """
    v4 Engine (refactor-ready):
    - НЕ поднимает WS/HTTP
    - handle_control(msg) -> response msg для UI (контракт в control.py)
    - telemetry наружу через set_event_callback(cb) (контракт fixed)
    - realtime: никаких блокировок в audio callback
    """

    def __init__(self, dsp: DSPConfig | None = None):
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
        self._monitor_headroom_sec: float = 1.2

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

        # --- ML stub ---
        self._ml_cfg = MLClassifierConfig(enabled=False)
        self._ml = InstrumentClassifierStub(publish_fn=self._emit_sync, cfg=self._ml_cfg)
        self._ml.start()

    # -------------------- lifecycle --------------------

    def set_event_callback(self, cb: EventCallback | None) -> None:
        self._event_cb = cb

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
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

    # -------------------- input device I/O --------------------

    def _device_callback(self, indata, frames, time_info, status) -> None:
        try:
            x = np.asarray(indata, dtype=np.float32)
            if x.ndim == 1:
                x = x[:, None]
            # enforce (frames, ch<=2)
            ch = min(2, max(1, x.shape[1]))
            if x.shape[1] > ch:
                x = x[:, :ch]
            elif x.shape[1] < ch:
                pad = np.zeros((x.shape[0], ch), dtype=np.float32)
                pad[:, : x.shape[1]] = x
                x = pad

            # Analysis-only gain/gate (must NOT affect audible output)
            g = float(self._analysis_gain)
            if g != 1.0:
                x = x * g

            gate = float(self._gate_rms)
            if gate > 0.0:
                rms = float(np.sqrt(np.mean(np.square(x))))
                if rms < gate:
                    # below gate -> push zeros to analysis queue
                    x = np.zeros_like(x)

            self._q_put_nowait(x)
            # ML non-blocking enqueue
            try:
                self._ml.enqueue_audio(x, sr=int(self.state.sr), ch=int(self.state.channels))
            except Exception:
                pass
        except Exception:
            return

    def _open_input_stream(self, source_id: str) -> None:
        """
        SourceId: device:default OR device:<index> OR device:<name>
        """
        self._close_input_stream()

        dev = None
        if source_id == "device:default":
            dev = None
        else:
            tok = source_id.split(":", 1)[1]
            try:
                dev = int(tok)
            except Exception:
                # sounddevice accepts name fragments too
                dev = tok

        self._in_stream = sd.InputStream(
            device=dev,
            channels=int(getattr(self.state, "channels", 1)),
            samplerate=int(getattr(self.state, "sr", 48000)),
            dtype="float32",
            callback=self._device_callback,
            blocksize=0,
        )
        self._in_stream.start()

    # -------------------- loopback (Windows WASAPI) --------------------

    def _resolve_loopback_wasapi(self) -> Tuple[int, str, int, int, Any]:
        """
        Tries to find default output device and open WASAPI loopback on it.
        Returns: (device_index, name, sr, ch, extra_settings)
        """
        WasapiSettings = getattr(sd, "WasapiSettings", None)
        if WasapiSettings is None:
            raise RuntimeError("WASAPI is not available in sounddevice build")

        out_dev = sd.default.device[1]  # (in, out)
        if out_dev is None:
            raise RuntimeError("No default output device for loopback")

        info = sd.query_devices(out_dev, "output")
        name = str(info.get("name", "output"))
        sr = int(info.get("default_samplerate", 48000))
        max_out_ch = int(info.get("max_output_channels", 2))
        ch = int(min(2, max(1, max_out_ch)))

        extra = WasapiSettings(loopback=True)
        return out_dev, f"{name} (WASAPI loopback)", sr, ch, extra

    def _open_loopback_stream(self) -> None:
        """
        SourceId: loopback:default (контракт с UI фиксированный!)
        """
        self._close_input_stream()

        dev, name, sr, ch, extra = self._resolve_loopback_wasapi()
        self._loopback_cached = (name, sr, ch)

        self.state.sr = int(sr)
        self.state.channels = int(ch)

        self._in_stream = sd.InputStream(
            device=int(dev),
            channels=self.state.channels,
            samplerate=self.state.sr,
            dtype="float32",
            callback=self._device_callback,
            blocksize=0,
            extra_settings=extra,
        )
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

    def _monitor_push(self, data: np.ndarray, sr: int, ch: int) -> None:
        """
        Push audio to monitor ring buffer (audible).
        MUST be non-blocking and safe to call from asyncio loop.
        """
        if not self._monitor_enabled:
            return

        try:
            self._ensure_output_stream(sr=sr, ch=ch)
            rb = self._mon_rb
            if rb is None:
                return
            rb.write(data)
        except Exception:
            return

    # -------------------- DSP / sources --------------------

    def set_dsp_params(
        self,
        bands: int | None = None,
        fft_size: int | None = None,
        block: int | None = None,
        mock_events: bool | None = None,
        monitor: bool | None = None,
        analysis_gain: float | None = None,
        gate_rms: float | None = None,
    ) -> None:
        if bands is not None:
            self.dsp_core.params.num_bands = int(bands)
        if fft_size is not None:
            self.dsp_core.params.fft_size = int(fft_size)
        if block is not None:
            self.dsp_core.params.block = int(block)
        if mock_events is not None:
            self.dsp_core.params.mock_events = bool(mock_events)

        if monitor is not None:
            self._monitor_enabled = bool(monitor)
            if not self._monitor_enabled:
                self._close_output_stream()

        if analysis_gain is not None:
            self._analysis_gain = float(analysis_gain)

        if gate_rms is not None:
            self._gate_rms = float(gate_rms)

    def list_sources(self) -> Dict[str, object]:
        items = []

        # devices
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                # input capable?
                max_in = int(d.get("max_input_channels", 0))
                if max_in <= 0:
                    continue
                name = str(d.get("name", f"device {i}"))
                sr = int(d.get("default_samplerate", 48000))
                ch = int(min(2, max(1, max_in)))
                items.append(
                    {
                        "id": f"device:{i}",
                        "kind": "device",
                        "name": name,
                        "channels": ch,
                        "sr": sr,
                    }
                )
        except Exception:
            pass

        # loopback default (optional, Windows)
        lb_name = "[NOT FOUND]"
        lb_sr = 48000
        lb_ch = 2
        try:
            _, name, sr, ch, _ = self._resolve_loopback_wasapi()
            lb_name = name
            lb_sr = sr
            lb_ch = ch
        except Exception:
            # остаётся [NOT FOUND] — UI увидит это и будет понятно почему не работает
            pass

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
        # DEBUG: ML internal counters (not in audio callback)
        next_ml_dbg = time.time() + 5.0
        last_send = 0.0
        while not self._closing:
            now = time.time()
            if now >= next_ml_dbg:
                try:
                    print(f"ML DEBUG {time.strftime('%H:%M:%S')} | {self._ml.debug_snapshot()}")
                except Exception:
                    pass
                next_ml_dbg = now + 5.0
            if not self.state.running:
                await asyncio.sleep(0.02)
                continue

            if str(self.state.source_id).startswith("file:"):

                async def _on_chunk(data: np.ndarray, sr: int, ch: int) -> None:
                    # 1) monitor push (audible) — independent from dsp/telemetry
                    self._monitor_push(data, sr=sr, ch=ch)
                    # 2) dsp queue (analysis)
                    self._q_put_nowait(data)

                await self.file_player.ensure_task(_on_chunk, status_hz=15.0)

                if self.file_player.status_should_emit():
                    await self.emit_file_status()

            # DSP tick
            try:
                x = self._q.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0)
                continue

            meter = self.dsp_core.process(x, sr=int(self.state.sr), ch=int(self.state.channels))

            # emit audio meter @ fps
            now = time.time()
            if now - last_send >= (1.0 / max(1.0, float(self.dsp_core.params.fps))):
                last_send = now
                await self.emit({"type": "audio_meter", "payload": meter})

                if self.dsp_core.params.mock_events:
                    ev = self.dsp_core.mock_event(meter)
                    await self.emit({"type": "dsp_event", "payload": ev})

            # ML enqueue (non-blocking, never affects audio callback)
            try:
                self._ml.enqueue_audio(x, sr=int(self.state.sr), ch=int(self.state.channels))
            except Exception:
                pass

    # -------------------- control helpers (called from control.py) --------------------

    async def audio_start(self, fps: int | None = None) -> Dict[str, Any]:
        if fps is not None:
            try:
                self.dsp_core.params.fps = int(fps)
            except Exception:
                pass
        self.state.running = True
        return {"type": "audio_status", "payload": {"state": "running", "sourceId": self.state.source_id}}

    async def audio_stop(self) -> Dict[str, Any]:
        self.state.running = False
        return {"type": "audio_status", "payload": {"state": "stopped", "sourceId": self.state.source_id}}


# =========================
# Minimal ring buffer (monitor)
# =========================

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

        # Индексы в "frames"
        self._w = 0  # write index (monotonic)
        self._r = 0  # read index (monotonic)

        # Лок для write (read в callback без lock)
        self._lock = threading.Lock()

    def available_to_read(self) -> int:
        # frames available
        return max(0, int(self._w - self._r))

    def write(self, data: np.ndarray) -> None:
        """
        Writes (frames, ch) float32 into ring buffer.
        Non-blocking: if overflow, drop oldest data.
        """
        if data is None:
            return
        x = np.asarray(data, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != self.channels:
            # reshape/pad/truncate to channels
            ch = self.channels
            if x.shape[1] > ch:
                x = x[:, :ch]
            else:
                pad = np.zeros((x.shape[0], ch), dtype=np.float32)
                pad[:, : x.shape[1]] = x
                x = pad

        n = int(x.shape[0])
        if n <= 0:
            return

        with self._lock:
            # If overflow, advance read pointer to keep last capacity frames
            avail = self.available_to_read()
            if avail + n > self.capacity:
                drop = (avail + n) - self.capacity
                self._r += int(drop)

            w0 = self._w % self.capacity
            end = w0 + n
            if end <= self.capacity:
                self._buf[w0:end, :] = x
            else:
                n1 = self.capacity - w0
                self._buf[w0:self.capacity, :] = x[:n1, :]
                n2 = n - n1
                self._buf[0:n2, :] = x[n1:n1 + n2, :]

            self._w += n

    def read_into(self, out: np.ndarray) -> None:
        """
        Reads into out (frames, ch). If underrun, fills zeros.
        MUST be fast and safe to call from audio callback.
        """
        if out is None:
            return

        outdata = out
        outdata.fill(0)

        n = int(outdata.shape[0])
        if n <= 0:
            return

        # Optimistic read without lock: tolerate minor races; correctness not perfect but safe.
        avail = self.available_to_read()
        if avail <= 0:
            return

        nread = min(n, avail)
        r0 = self._r % self.capacity
        end = r0 + nread
        if end <= self.capacity:
            outdata[:nread, :] = self._buf[r0:end, :]
        else:
            n1 = self.capacity - r0
            outdata[:n1, :] = self._buf[r0:self.capacity, :]
            remain = nread - n1
            if remain > 0:
                outdata[n1:n1 + remain, :] = self._buf[0:remain, :]

        self._r += nread
