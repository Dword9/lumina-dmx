# -*- coding: utf-8 -*-
"""
x32_osc.py — OSC-мост к Behringer X32 Producer для ноды X32 (09.09).

Контракт: N:\\python_ide\\X-32\\X32_CONTRACT.md (IP 192.168.0.113, udp/10023,
firmware 4.06 с граблями /node). Группы управляются ЧЕРЕЗ DCA (звукач уже
настроил: DCA1=MIC ch1-8, DCA2=AIMP ch19-20, DCA3=PC ch17-18, DCA4=VIDEO
ch21-22), MASTER = /main/st/mix/fader. DCA сохраняет индивидуальный баланс
каналов; моторизованные фейдеры звукача физически едут.

Потокобезопасность: один UDP-сокет + lock (писать и читать могут из разных
задач aiohttp). Ответы пульта парсятся минимальным OSC-парсером без внешних
зависимостей (python-osc не ставим — рунг 3 лесенки).
"""
import logging
import re
import socket
import struct
import threading
import time

logger = logging.getLogger("x32")

DEFAULT_HOST = "192.168.0.113"
DEFAULT_PORT = 10023

# Группы ноды: id -> (osc-адрес фейдера, osc-адрес mute). Имена подтягиваются
# с пульта живьём (/dca/N/config/name), здесь только адреса.
GROUPS = {
    "mic":    ("/dca/1/fader", "/dca/1/on"),
    "aimp":   ("/dca/2/fader", "/dca/2/on"),
    "pc":     ("/dca/3/fader", "/dca/3/on"),
    "video":  ("/dca/4/fader", "/dca/4/on"),
    "master": ("/main/st/mix/fader", "/main/st/mix/on"),
}
# Отображение группы -> число DCA для стягивания живого имени (None у master)
_GROUP_DCA = {"mic": 1, "aimp": 2, "pc": 3, "video": 4, "master": None}

_FLOAT_MIN = 0.0
_FLOAT_MAX = 1.0


def _pad4(b: bytes) -> bytes:
    n = (4 - len(b) % 4) % 4
    return b + b"\x00" * n if n else b + b"\x00" * 4


def _osc_pack(addr: str, *args) -> bytes:
    """Минимальный OSC-пакер: float и int-аргументы (для наших нужд хватает)."""
    types = ","
    blob = b""
    for a in args:
        if isinstance(a, float):
            types += "f"
            blob += struct.pack(">f", a)
        elif isinstance(a, bool):
            types += "i"
            blob += struct.pack(">i", 1 if a else 0)
        elif isinstance(a, int):
            types += "i"
            blob += struct.pack(">i", a)
        elif isinstance(a, str):
            types += "s"
            blob += _pad4(a.encode("utf-8"))
    return _pad4(addr.encode()) + _pad4(types.encode()) + blob


def _osc_unpack(data: bytes):
    """Минимальный OSC-анпакер: возвращает (addr, [values])."""
    def _str(buf: bytes, off: int):
        end = buf.index(b"\x00", off)
        s = buf[off:end].decode("utf-8", errors="replace")
        return s, (end + 4) & ~3

    if not data:
        return None, []
    addr, off = _str(data, 0)
    if off >= len(data) or data[off : off + 1] != b",":
        return addr, []
    types, off = _str(data, off)
    vals = []
    for t in types[1:]:
        if t == "f":
            vals.append(struct.unpack(">f", data[off : off + 4])[0]); off += 4
        elif t == "i":
            vals.append(struct.unpack(">i", data[off : off + 4])[0]); off += 4
        elif t == "s":
            v, off = _str(data, off)
            vals.append(v)
    return addr, vals


class X32Client:
    """UDP-клиент X32: ask (запрос-ответ), set (без ответа), poll-кэш групп."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 0.8):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout)
        # кэш живых имён групп {id: {"name": str, "fader": float, "on": int}}
        self._names: dict = {}
        self._last_seen: float = 0.0

    # ---------- низкий уровень ----------

    def _ask_raw(self, addr: str, retries: int = 1):
        """Отправить read-запрос, вернуть (addr, values) или (None, []).
        retries=1 (было 2): при отвале пульта группа из 5 вопросов × 2 DCA-
        адреса не должна висеть по 8+ секунд — нода показывает красную
        метку сразу (10.09, кейс: линк до X32 отвалился, нода «висела»)."""
        pkt = _osc_pack(addr)
        with self._lock:
            for _ in range(retries + 1):
                try:
                    self._sock.sendto(pkt, (self.host, self.port))
                    data, _ = self._sock.recvfrom(2048)
                    return _osc_unpack(data)
                except socket.timeout:
                    continue
                except OSError as e:
                    logger.warning("x32 ask %s: %s", addr, e)
                    return None, []
        return None, []

    def _set(self, addr: str, value) -> None:
        """Отправить SET (X32 не подтверждает SET — читаем назад только для
        тестов, в бою не дублируем трафик)."""
        pkt = _osc_pack(addr, value)
        with self._lock:
            try:
                self._sock.sendto(pkt, (self.host, self.port))
            except OSError as e:
                logger.warning("x32 set %s: %s", addr, e)

    # ---------- публичное API ----------

    def ping(self) -> dict:
        """/info — статус связи. Возвращает {"ok": bool, "info": str}."""
        addr, vals = self._ask_raw("/info")
        if not addr or not vals:
            return {"ok": False, "info": "нет ответа"}
        # /info отвечает 4 строками: version, name, model, firmware
        return {"ok": True, "info": " ".join(str(v) for v in vals[:4])}

    def _read_group(self, gid: str) -> dict | None:
        fader_addr, on_addr = GROUPS[gid]
        _, fv = self._ask_raw(fader_addr)
        _, ov = self._ask_raw(on_addr)
        if not fv and not ov:
            return None
        return {
            "id": gid,
            "fader": _clamp_fader(fv[0]) if fv else None,
            "on": int(ov[0]) if ov else None,
        }

    def groups(self) -> list:
        """Живые имена и состояние всех групп. None-элементы = группа молчит."""
        out = []
        for gid in GROUPS:
            g = self._read_group(gid)
            if g is None:
                out.append({"id": gid, "fader": None, "on": None,
                            "name": self._names.get(gid) or GROUP_LABELS.get(gid, gid)})
                continue
            g["name"] = self._live_name(gid) or self._names.get(gid) or GROUP_LABELS.get(gid, gid)
            if g["name"]:
                self._names[gid] = g["name"]  # кэш: при отвале покажем последнее живое имя
            out.append(g)
        return out

    def _live_name(self, gid: str) -> str | None:
        dca = _GROUP_DCA[gid]
        if dca is None:
            return "MAIN LR"
        _, vals = self._ask_raw(f"/dca/{dca}/config/name")
        if vals and isinstance(vals[0], str):
            return vals[0].strip() or None
        return None

    def set_fader(self, gid: str, value: float) -> float:
        addr, _ = GROUPS[gid]
        v = _clamp_fader(value)
        self._set(addr, v)
        return v

    def set_on(self, gid: str, on: bool) -> bool:
        _, addr = GROUPS[gid]
        self._set(addr, 1 if on else 0)
        return on


# Человекочитаемые метки по умолчанию (пока имена не стянуты с пульта)
GROUP_LABELS = {
    "mic": "MIC (DCA1)",
    "aimp": "AIMP (DCA2)",
    "pc": "PC (DCA3)",
    "video": "VIDEO (DCA4)",
    "master": "MAIN LR",
}


def _clamp_fader(v: float) -> float:
    return max(_FLOAT_MIN, min(_FLOAT_MAX, float(v)))
