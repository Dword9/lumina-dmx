# dsp_client.py
# websockets==15.0.1
#
# ✅ Принимает бинарные аудио-чанки от React по ws://localhost:8765/audio
# ✅ Логирует, что реально приходят данные (размер, скорость, RMS-оценка)
# ✅ Делает mock-анализ (kick/snare/hihat)
# ✅ Отправляет dsp_event обратно в React по тому же WebSocket
# ✅ (опционально) вызывает send_func (DMX/ArtNet), sync или async

import asyncio
import json
import random
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import websockets
from websockets.server import ServerConnection

SendFunc = Optional[Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]]


def _get_path(conn: Any) -> str:
    # websockets 15: conn.request.path
    req = getattr(conn, "request", None)
    if req is not None:
        p = getattr(req, "path", "")
        if isinstance(p, str):
            return p
    p = getattr(conn, "path", "")
    return p if isinstance(p, str) else ""


async def _maybe_await(x: Any) -> None:
    if asyncio.iscoroutine(x) or isinstance(x, Awaitable):
        await x  # type: ignore[misc]


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _estimate_rms_int16le(buf: bytes) -> float:
    """
    Очень грубая оценка "есть звук/нет" по RMS для PCM int16 little-endian.
    Если данные не PCM — всё равно даст какую-то цифру, но для диагностики хватит.
    Возвращает RMS в диапазоне примерно 0..32767.
    """
    n = len(buf) // 2
    if n <= 0:
        return 0.0
    s = 0
    # берём не весь буфер, а хвост до 4096 сэмплов, чтобы не тормозить
    tail = buf[-min(len(buf), 4096 * 2) :]
    n = len(tail) // 2
    if n <= 0:
        return 0.0
    for i in range(0, n * 2, 2):
        v = int.from_bytes(tail[i : i + 2], "little", signed=True)
        s += v * v
    return (s / n) ** 0.5


class AudioAnalyzer:
    def __init__(
        self,
        send_func: SendFunc = None,
        port: int = 8765,
        host: str = "0.0.0.0",
        endpoint: str = "/audio",
        analyze_every_n_chunks: int = 5,   # чтобы не спамить событиями/логом
        log_every_n_chunks: int = 10,      # как часто писать "звук приходит"
    ):
        self.send_func = send_func
        self.port = int(port)
        self.host = host
        self.endpoint = endpoint
        self.analyze_every_n_chunks = max(1, int(analyze_every_n_chunks))
        self.log_every_n_chunks = max(1, int(log_every_n_chunks))

        self.audio_buffer = bytearray()
        self._chunk_counter = 0

        # статистика скорости
        self._bytes_since = 0
        self._t0 = time.time()

    def start(self) -> None:
        thread = threading.Thread(target=self._run_server, daemon=True)
        thread.start()

    def _run_server(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async def handler(conn: ServerConnection) -> None:
            path = _get_path(conn)
            if not (
                path == self.endpoint
                or path.startswith(self.endpoint + "?")
                or path.startswith(self.endpoint + "/")
            ):
                print(f"{_now()} | DSP | ❌ invalid path: {path!r} from {getattr(conn,'remote_address',None)}")
                await conn.close(code=1008, reason="Invalid path")
                return

            print(f"{_now()} | DSP | ✅ connected: {conn.remote_address} path={path}")
            await conn.send(json.dumps({
              "type": "dsp_event",
              "payload": {"kick": 0.9, "snare": 0.1, "hihat": 0.5},
              "ts": time.time()
                    }))
            print("DSP | sent test dsp_event on connect")

            try:
                async for message in conn:
                    if isinstance(message, (bytes, bytearray)):
                        b = bytes(message)
                        self.audio_buffer.extend(b)

                        self._chunk_counter += 1
                        self._bytes_since += len(b)

                        # ---- ЛОГ: подтверждаем, что звук реально приходит ----
                        if (self._chunk_counter % self.log_every_n_chunks) == 0:
                            dt = max(1e-3, time.time() - self._t0)
                            kbps = (self._bytes_since / 1024.0) / dt
                            rms = _estimate_rms_int16le(b)
                            print(
                                f"{_now()} | DSP | 🎧 audio in: chunk#{self._chunk_counter} "
                                f"size={len(b)}B  rate~{kbps:.1f}KB/s  rms~{rms:.0f}"
                            )
                            # сбрасываем окно скорости
                            self._t0 = time.time()
                            self._bytes_since = 0

                        # ---- анализ и обратный ответ ----
                        if (self._chunk_counter % self.analyze_every_n_chunks) == 0:
                            await self._analyze_and_respond(conn)

                    else:
                        # текстовые сообщения (если вдруг) — логируем один раз
                        t = str(message)
                        if t.strip():
                            print(f"{_now()} | DSP | ℹ️ text (ignored): {t[:120]}")

            except websockets.ConnectionClosed:
                pass
            except Exception as e:
                print(f"{_now()} | DSP | ❌ handler error: {repr(e)}")
            finally:
                print(f"{_now()} | DSP | 👋 disconnected: {conn.remote_address}")

        async with websockets.serve(
            handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
            max_size=16 * 1024 * 1024,
        ):
            print(f"{_now()} | DSP | 🎧 WS listening on ws://{self.host}:{self.port}{self.endpoint}")
            await asyncio.Future()

    async def _analyze_and_respond(self, conn: ServerConnection) -> None:
        # ---- MOCK DSP ----
        result = {
            "kick": round(random.uniform(0.0, 1.0), 2),
            "snare": round(random.uniform(0.0, 1.0), 2),
            "hihat": round(random.uniform(0.0, 1.0), 2),
        }

        msg: Dict[str, Any] = {
            "type": "dsp_event",
            "payload": result,
            "ts": time.time(),
        }

        # ✅ обратная связь в React
        try:
            await conn.send(json.dumps(msg, ensure_ascii=False))
        except Exception as e:
            print(f"{_now()} | DSP | ⚠️ send dsp_event failed: {repr(e)}")

        # ✅ твой DMX/ArtNet callback остаётся
        if self.send_func is not None:
            try:
                maybe = self.send_func(msg)
                await _maybe_await(maybe)
            except Exception as e:
                print(f"{_now()} | DSP | ⚠️ send_func failed: {repr(e)}")

        # для диагностики/мока чистим буфер, чтобы не рос
        self.audio_buffer.clear()


if __name__ == "__main__":
    analyzer = AudioAnalyzer(send_func=None, host="0.0.0.0", port=8765, endpoint="/audio")
    analyzer.start()
    while True:
        time.sleep(1)
