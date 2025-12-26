# lumina_dsp/audio/file_player.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf


@dataclass
class FileInfo:
    file_id: str
    path: str
    name: str
    sr: int
    channels: int
    frames: int
    duration: float


class FileRegistry:
    """Хранит загруженные файлы (fileId -> FileInfo)."""

    def __init__(self):
        self._items: Dict[str, FileInfo] = {}

    def add(self, info: FileInfo) -> None:
        self._items[str(info.file_id)] = info

    def get(self, file_id: str) -> FileInfo:
        if file_id not in self._items:
            raise KeyError(f"Unknown file_id: {file_id}")
        return self._items[file_id]

    def items(self):
        return self._items.items()


class FilePlayer:
    """
    Async file playback feeder:
    - читает chunks из файла
    - вызывает on_chunk(data, sr, ch)
    - позиция/seek/pause/play/stop
    ВАЖНО: звук реально играет через sounddevice.OutputStream callback в AudioAnalyzer,
    поэтому здесь нельзя кормить "впритык" -> нужен prebuffer.
    """

    def __init__(self, registry: FileRegistry, block: int = 1024):
        self.registry = registry
        self.block = int(block)

        self._active_id: Optional[str] = None
        self._state: str = "stopped"  # stopped|playing|paused
        self._pos_sec: float = 0.0

        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._last_status_ts: float = 0.0

        # Сколько запаса держим (сек) относительно realtime.
        # Чуть больше секунды помогает избежать заиканий при нагрузке ОС.
        self._prebuffer_sec: float = 1.0

    async def set_active(self, file_id: str) -> None:
        async with self._lock:
            self._active_id = str(file_id)
            self._pos_sec = 0.0
            self._state = "stopped"

    async def play(self) -> None:
        async with self._lock:
            if self._active_id is None:
                raise ValueError("No active file")
            self._state = "playing"

    async def pause(self) -> None:
        async with self._lock:
            if self._state == "playing":
                self._state = "paused"

    async def stop(self) -> None:
        async with self._lock:
            self._state = "stopped"
            self._pos_sec = 0.0
        # task сам завершится на следующей итерации, но можно отменить принудительно
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        self._task = None

    async def seek(self, seconds: float) -> None:
        async with self._lock:
            if self._active_id is None:
                return
            info = self.registry.get(self._active_id)
            sec = float(seconds)
            if sec < 0:
                sec = 0.0
            if info.duration > 0:
                sec = min(sec, float(info.duration))
            self._pos_sec = sec

    async def get_status_payload(self) -> Dict[str, object]:
        async with self._lock:
            fid = self._active_id
            state = self._state
            pos = float(self._pos_sec)

        if fid is None:
            return {"state": "stopped", "fileId": None, "duration": 0.0, "position": 0.0}

        info = self.registry.get(fid)
        return {
            "state": state,
            "fileId": fid,
            "name": info.name,
            "duration": float(info.duration),
            "position": float(pos),
            "sr": int(info.sr),
            "channels": int(info.channels),
        }

    async def _snapshot(self) -> Tuple[Optional[str], str, float]:
        async with self._lock:
            return self._active_id, self._state, self._pos_sec

    def status_should_emit(self, hz: float = 15.0) -> bool:
        now = time.time()
        if now - self._last_status_ts >= (1.0 / max(1.0, hz)):
            self._last_status_ts = now
            return True
        return False

    async def ensure_task(self, on_chunk, status_hz: float = 15.0) -> None:
        """
        Гарантирует, что loop запущен, если состояние playing.
        on_chunk(data: np.ndarray, sr: int, channels: int) -> None/awaitable
        """
        fid, state, _ = await self._snapshot()
        if fid is None or state != "playing":
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(on_chunk, status_hz=status_hz))

    async def _loop(self, on_chunk, status_hz: float = 15.0) -> None:
        fid, _, _ = await self._snapshot()
        if fid is None:
            return

        info = self.registry.get(fid)
        frames_per_block = int(self.block)

        # PREBUFFER pacing:
        # мы хотим отправлять данные чуть "вперёд" realtime на self._prebuffer_sec
        # чтобы ring-buffer у OutputStream был всегда заполнен и не ловил микролаги.
        prebuffer_frames = int(max(0.0, self._prebuffer_sec) * float(info.sr))

        # Виртуальная "точка старта" для расчёта дедлайнов
        # frames_sent = сколько кадров мы уже отдали в on_chunk с начала текущего файла/позиции
        frames_sent = 0
        t0 = time.perf_counter()

        try:
            with sf.SoundFile(info.path, mode="r") as f:
                while True:
                    fid_now, state, pos = await self._snapshot()

                    if state == "stopped" or fid_now is None:
                        break

                    if state == "paused":
                        # при паузе не продвигаем realtime-час
                        await asyncio.sleep(0.02)
                        # чтобы после паузы не пытаться "догнать", пересинхроним t0:
                        # оставим frames_sent как есть, но t0 сдвинем к текущему времени
                        t0 = time.perf_counter() - max(0.0, (frames_sent - prebuffer_frames) / float(info.sr))
                        continue

                    # seek
                    target_frame = int(pos * info.sr)
                    target_frame = max(0, min(target_frame, info.frames))
                    if f.tell() != target_frame:
                        f.seek(target_frame)
                        frames_sent = target_frame
                        # пересинхронизация дедлайна после seek
                        t0 = time.perf_counter() - max(0.0, (frames_sent - prebuffer_frames) / float(info.sr))

                    # read (ВАЖНО: сразу float32, чтобы не было случайных CPU-спайков на astype)
                    # Disk I/O может блокировать event loop, поэтому читаем в thread-пуле.
                    data = await asyncio.to_thread(
                        f.read, frames_per_block, dtype="float32", always_2d=True
                    )
                    if data.size == 0 or len(data) == 0:
                        # EOF -> stop
                        async with self._lock:
                            self._state = "stopped"
                        break

                    # normalize to max 2 channels (DSP side обычно 1-2)
                    ch = min(2, info.channels)
                    if data.shape[1] > ch:
                        data = data[:, :ch]
                    elif data.shape[1] < ch:
                        pad = np.zeros((data.shape[0], ch), dtype=np.float32)
                        pad[:, : data.shape[1]] = data
                        data = pad

                    # callback
                    out = on_chunk(data, info.sr, ch)
                    if asyncio.iscoroutine(out):
                        await out

                    # update position
                    new_pos = f.tell() / float(info.sr)
                    async with self._lock:
                        self._pos_sec = new_pos

                    # --- PREBUFFER REALTIME PACING ---
                    # Мы считаем, что "аудио реально звучит" с задержкой prebuffer_sec,
                    # поэтому разрешаем себе быть впереди на prebuffer_frames.
                    frames_sent += int(len(data))

                    desired_play_time_sec = max(
                        0.0, (frames_sent - prebuffer_frames) / float(info.sr)
                    )
                    deadline = t0 + desired_play_time_sec
                    now = time.perf_counter()
                    sleep_s = deadline - now
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                    else:
                        # если мы отстали (например, ОС лагнула), не делаем "догонялки" через busy loop
                        await asyncio.sleep(0)

        except asyncio.CancelledError:
            raise
        except Exception:
            raise
