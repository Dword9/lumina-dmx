# v3/lumina_dsp/telemetry.py
from __future__ import annotations

import inspect
import time
from typing import Any, Dict, Optional, Callable, Awaitable, Union

from .state import EventCallback


MaybeAwaitable = Union[None, Awaitable[None]]


async def _maybe_await(x: Any) -> None:
    """
    Позволяет event callback быть sync или async.

    Важно: мы НЕ хотим тут блокировок.
    """
    if x is None:
        return
    if inspect.isawaitable(x):
        await x


async def emit(event_cb: Optional[EventCallback], msg: Dict[str, Any]) -> None:
    """
    Безопасная публикация telemetry наружу (в WS слой).

    КРИТИЧНО ДЛЯ КОНТРАКТА:
    - msg это готовый JSON envelope {type, payload, reqId?, ts?}
    - Любая ошибка эмиттера НЕ должна убивать DSP loop / аудио
    """
    if event_cb is None:
        return
    try:
        maybe = event_cb(msg)
        await _maybe_await(maybe)
    except Exception:
        # Don't kill dsp/audio because UI/WS emitter failed
        pass


async def emit_file_status(
    event_cb: Optional[EventCallback],
    file_player: Any,
    req_id: Optional[str] = None,
) -> None:
    """
    Контракт с UI (НЕ ЛОМАТЬ):
      type: "audio_file_status"
      payload: {
        state: "playing" | "paused" | "stopped",
        fileId: string,
        position: number,
        duration: number
      }

    Мы не формируем payload руками — берем из FilePlayer.get_status_payload(),
    чтобы сохранить существующее поведение 1-в-1.
    """
    msg: Dict[str, Any] = {
        "type": "audio_file_status",
        "payload": await file_player.get_status_payload(),
        "ts": time.time(),
    }
    if req_id is not None:
        msg["reqId"] = req_id
    await emit(event_cb, msg)
