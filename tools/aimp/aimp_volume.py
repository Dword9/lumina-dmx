# -*- coding: utf-8 -*-
"""AIMP: внутренняя громкость плеера через штатный плагин Remote Control.

Плагин (установлен в AIMP) поднимает локальный HTTP+JSON-RPC на :3333:
  POST http://127.0.0.1:3333/RPC_JSON
  {"jsonrpc":"2.0","id":1,"method":"GetPlayerControlPanelState","params":{}}
      -> {"result":{"volume":0..100,"playback_state":"...","mute_mode_on":bool,...}}
  {"jsonrpc":"2.0","id":2,"method":"VolumeLevel","params":{"level":0..100}}
      -> {"result":{"volume":0..100}}          (это ВНУТРЕННЯЯ громкость AIMP,
                                                видна в самом плеере)
  {"jsonrpc":"2.0","id":3,"method":"VolumeLevel","params":{}}
      -> чтение текущей громкости

Почему RPC, а не микшер Windows: микшер меняет канал приложения (ползунок в
AIMP при этом не двигается), а юзеру нужно видеть громкость в плеере — это
родная громкость AIMP. Бонус: RPC доступен по TCP из любой сессии (главная
грабля session 0 у Core Audio здесь неактуальна), и работает даже когда AIMP
ничего не играет.

Никаких внешних зависимостей: urllib/json из stdlib.
"""

import itertools
import json
import os
import urllib.request

RPC_URL = os.environ.get("AIMP_RPC_URL", "http://127.0.0.1:3333/RPC_JSON")
_TIMEOUT = 2.5
_ids = itertools.count(1)


def _call(method: str, params: dict = None):
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": next(_ids),
        "method": method,
        "params": params or {},
    }).encode("utf-8")
    req = urllib.request.Request(RPC_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _state() -> dict:
    r = _call("GetPlayerControlPanelState")
    if "error" in r:
        raise RuntimeError(str(r["error"]))
    return r.get("result", {})


def get_status() -> dict:
    """found — AIMP отвечает по RPC (плеер запущен и плагин работает)."""
    try:
        st = _state()
    except Exception as e:
        return {"found": False, "volume": None, "muted": None,
                "info": f"AIMP недоступен ({e})"}
    vol = st.get("volume")
    play = st.get("playback_state", "?")
    word = {"playing": "играет", "paused": "пауза", "stopped": "стоп"}.get(play, play)
    return {
        "found": True,
        "volume": (vol / 100.0) if isinstance(vol, (int, float)) else None,
        "muted": bool(st.get("mute_mode_on")),
        "playback": play,
        "info": f"AIMP: {vol}% · {word}" + (" · MUTE" if st.get("mute_mode_on") else ""),
    }


def set_volume(value: float) -> dict:
    """value 0..1 -> внутренняя громкость AIMP 0..100."""
    level = max(0, min(100, int(round(float(value) * 100))))
    try:
        r = _call("VolumeLevel", {"level": level})
    except Exception as e:
        return {"ok": False, "found": False, "volume": value, "error": str(e)}
    if "error" in r:
        return {"ok": False, "found": True, "volume": value, "error": str(r["error"])}
    got = r.get("result", {}).get("volume", level)
    return {"ok": True, "found": True, "volume": got / 100.0, "level": got}
