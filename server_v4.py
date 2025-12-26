# server_v3.py
import asyncio
import json
import logging
import os
import socket
import struct
import time
import uuid
import hashlib
import shutil
import subprocess
from typing import Any, Dict, Optional, Set

import soundfile as sf
from aiohttp import web, WSMsgType

from lumina_dsp.app import AudioAnalyzer
from lumina_dsp.audio.file_player import FileInfo


# ================== CONFIG ==================
WS_HOST = "0.0.0.0"
PORT = 8000

UNIVERSE = 0
DMX_CHANNELS = 512
FPS = 40
TARGET_IP = "192.168.0.255"

UPLOAD_DIR = "./_uploads"
INDEX_FILE = os.path.join(UPLOAD_DIR, "index.json")

LOG_EVERY = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

msg_count = 0
last_log_time = time.time()


# ================== CORS ==================
@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        resp = web.Response(status=200)
    else:
        resp = await handler(request)

    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp


# ================== HELPERS ==================
def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract_audio_with_ffmpeg(src_path: str, dst_path: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        src_path,
        "-vn",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-f",
        "wav",
        dst_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="ignore")[:500])


def load_index() -> Dict[str, str]:
    if not os.path.exists(INDEX_FILE):
        return {}
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_index(index: Dict[str, str]) -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def rebuild_registry_from_uploads(analyzer: AudioAnalyzer, index: Dict[str, str]) -> None:
    """
    Восстанавливает analyzer.registry из папки _uploads и index.json.

    ВАЖНО:
    - audio_path может быть extracted.wav, но name для UI берём из оригинального файла digest__<name>.
    - чистим протухшие записи index, если файлов на диске уже нет.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    def find_orig_name_and_path(digest: str) -> tuple[Optional[str], Optional[str]]:
        """
        Возвращает (orig_name, orig_path) для файла вида "{digest}__<orig_name>".
        """
        prefix = f"{digest}__"
        for fn in os.listdir(UPLOAD_DIR):
            if fn.endswith(".json"):
                continue
            if fn == f"{digest}__extracted.wav":
                continue
            if fn.startswith(prefix):
                orig_name = fn[len(prefix):]  # то, что после digest__
                return orig_name, os.path.join(UPLOAD_DIR, fn)
        return None, None

    def resolve_audio_path(digest: str) -> Optional[str]:
        """
        Если есть extracted.wav — используем его (для видео/контейнеров).
        Иначе используем оригинальный файл.
        """
        extracted = os.path.join(UPLOAD_DIR, f"{digest}__extracted.wav")
        if os.path.exists(extracted):
            return extracted

        # fallback: оригинал
        _, orig_path = find_orig_name_and_path(digest)
        return orig_path

    for digest, file_id in list(index.items()):
        audio_path = resolve_audio_path(digest)
        orig_name, _orig_path = find_orig_name_and_path(digest)

        if not audio_path or not os.path.exists(audio_path):
            index.pop(digest, None)
            continue

        # уже есть в registry? пропускаем
        try:
            analyzer.registry.get(file_id)
            continue
        except Exception:
            pass

        # читаем метаданные
        try:
            with sf.SoundFile(audio_path) as snd:
                sr = int(snd.samplerate)
                ch = int(snd.channels)
                frames = int(len(snd))
                dur = float(frames) / float(sr) if sr else 0.0
        except Exception:
            index.pop(digest, None)
            continue

        # UI-friendly name: если нашли оригинал — показываем его, иначе basename(audio_path)
        display_name = orig_name or os.path.basename(audio_path)

        analyzer.registry.add(
            FileInfo(
                file_id=file_id,
                path=audio_path,
                name=display_name,
                sr=sr,
                channels=ch,
                frames=frames,
                duration=dur,
            )
        )

# ================== ARTNET ==================
class ArtNetSender:
    def __init__(self, target_ip: str, universe: int, dmx_len: int):
        self.target_ip = target_ip
        self.universe = universe
        self.dmx_len = dmx_len

        self.dmx_data = bytearray([0] * dmx_len)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("", 0))
        self.sock.setblocking(False)

        self.packet = self._build_packet()
        self.dirty = False

    def _build_packet(self) -> bytearray:
        header = b"Art-Net\x00"
        opcode = struct.pack("<H", 0x5000)
        proto_ver = struct.pack(">H", 14)
        sequence = b"\x00"
        physical = b"\x00"
        universe = struct.pack("<H", self.universe)
        length = struct.pack(">H", self.dmx_len)
        return bytearray(header + opcode + proto_ver + sequence + physical + universe + length + self.dmx_data)

    def set_channel(self, ch_index: int, value: int) -> None:
        value = max(0, min(255, int(value)))
        if 0 <= ch_index < self.dmx_len and self.dmx_data[ch_index] != value:
            self.dmx_data[ch_index] = value
            self.packet[18 + ch_index] = value
            self.dirty = True

    def send_if_dirty(self) -> None:
        if not self.dirty:
            return
        try:
            self.sock.sendto(self.packet, (self.target_ip, 6454))
        except Exception as e:
            logging.error("ArtNet send failed: %r", e)
        self.dirty = False

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


async def sender_loop(app: web.Application):
    artnet: ArtNetSender = app["artnet"]
    interval = 1.0 / FPS
    while True:
        artnet.send_if_dirty()
        await asyncio.sleep(interval)


# ================== WS SUBS ==================
UI_CLIENTS: Set[web.WebSocketResponse] = set()
SUBS: Dict[web.WebSocketResponse, Set[str]] = {}


def is_subscribed(ws: web.WebSocketResponse, stream: str) -> bool:
    return stream in SUBS.get(ws, set())


async def broadcast(app: web.Application, stream: str, msg: Dict[str, Any]) -> None:
    raw = json.dumps(msg, ensure_ascii=False)
    dead = []
    for ws in UI_CLIENTS:
        if not is_subscribed(ws, stream):
            continue
        try:
            await ws.send_str(raw)
        except Exception:
            dead.append(ws)
    for ws in dead:
        UI_CLIENTS.discard(ws)
        SUBS.pop(ws, None)


def analyzer_event_cb(app: web.Application):
    async def _cb(msg: Dict[str, Any]) -> None:
        t = msg.get("type")
        if t:
            await broadcast(app, t, msg)
    return _cb


# ================== HTTP ==================
async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def upload(request: web.Request) -> web.Response:
    analyzer: AudioAnalyzer = request.app["analyzer"]
    index: Dict[str, str] = request.app["upload_index"]

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    reader = await request.multipart()
    part = await reader.next()
    if part is None or part.name != "file":
        return web.json_response({"ok": False, "error": "file field missing"}, status=400)

    filename = part.filename or "upload.bin"
    tmp_path = os.path.join(UPLOAD_DIR, f"__tmp__{uuid.uuid4().hex}")

    with open(tmp_path, "wb") as f:
        while True:
            chunk = await part.read_chunk()
            if not chunk:
                break
            f.write(chunk)

    digest = sha256_file(tmp_path)

    # -------- DEDUP path --------
    if digest in index:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        fid = index[digest]

        # IMPORTANT FIX: ensure registry has this file (after restart it may be empty)
        try:
            analyzer.registry.get(fid)
        except Exception:
            rebuild_registry_from_uploads(analyzer, index)
            save_index(index)

        return web.json_response({"ok": True, "fileId": fid, "sourceId": f"file:{fid}", "dedup": True})

    safe_name = filename.replace("/", "_").replace("\\", "_")
    orig_path = os.path.join(UPLOAD_DIR, f"{digest}__{safe_name}")
    os.replace(tmp_path, orig_path)

    audio_path = orig_path
    from_video = False

    # if it's not an audio file, extract audio via ffmpeg
    try:
        with sf.SoundFile(audio_path) as snd:
            sr = int(snd.samplerate)
            ch = int(snd.channels)
            frames = int(len(snd))
            dur = frames / sr if sr else 0.0
    except Exception:
        wav_path = os.path.join(UPLOAD_DIR, f"{digest}__extracted.wav")
        extract_audio_with_ffmpeg(orig_path, wav_path)
        audio_path = wav_path
        from_video = True
        with sf.SoundFile(audio_path) as snd:
            sr = int(snd.samplerate)
            ch = int(snd.channels)
            frames = int(len(snd))
            dur = frames / sr if sr else 0.0

    file_id = digest[:24]
    index[digest] = file_id
    save_index(index)

    analyzer.registry.add(
        FileInfo(
            file_id=file_id,
            path=audio_path,
            name=safe_name,
            sr=sr,
            channels=ch,
            frames=frames,
            duration=dur,
        )
    )

    return web.json_response(
        {
            "ok": True,
            "fileId": file_id,
            "sourceId": f"file:{file_id}",
            "name": safe_name,
            "sr": sr,
            "channels": ch,
            "duration": dur,
            "fromVideo": from_video,
            "dedup": False,
        }
    )


# ================== WS ==================
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    global msg_count, last_log_time
    app = request.app
    analyzer: AudioAnalyzer = app["analyzer"]
    artnet: ArtNetSender = app["artnet"]

    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)

    UI_CLIENTS.add(ws)
    SUBS[ws] = set()

    await ws.send_str(json.dumps({"type": "hello_ack", "payload": {"server": "lumina-v3"}}, ensure_ascii=False))

    try:
        async for m in ws:
            if m.type != WSMsgType.TEXT:
                continue

            msg_count += 1
            now = time.time()
            if now - last_log_time > LOG_EVERY:
                logging.info("WS msg rate: ~%d msgs/%0.1fs", msg_count, now - last_log_time)
                msg_count = 0
                last_log_time = now

            data = json.loads(m.data)

            # DMX legacy: list of {ch,val}
            if isinstance(data, list):
                for it in data:
                    if "ch" in it and "val" in it:
                        artnet.set_channel(int(it["ch"]) - 1, int(it["val"]))
                continue

            # DMX legacy: single {ch,val}
            if "ch" in data and "val" in data:
                artnet.set_channel(int(data["ch"]) - 1, int(data["val"]))
                continue

            # subscriptions
            if data.get("type") == "subscribe":
                for s in data.get("payload", {}).get("streams", []):
                    SUBS[ws].add(s)
                await ws.send_str(json.dumps({"type": "ack", "reqId": data.get("reqId")}, ensure_ascii=False))
                continue

            if data.get("type") == "unsubscribe":
                for s in data.get("payload", {}).get("streams", []):
                    SUBS[ws].discard(s)
                await ws.send_str(json.dumps({"type": "ack", "reqId": data.get("reqId")}, ensure_ascii=False))
                continue

            # DSP control
            resp = await analyzer.handle_control(data)
            if resp:
                await ws.send_str(json.dumps(resp, ensure_ascii=False))

    finally:
        UI_CLIENTS.discard(ws)
        SUBS.pop(ws, None)

    return ws


# ================== APP ==================
async def on_startup(app: web.Application):
    analyzer: AudioAnalyzer = app["analyzer"]
    await analyzer.start()
    analyzer.set_event_callback(analyzer_event_cb(app))
    app["sender_task"] = asyncio.create_task(sender_loop(app))


async def on_cleanup(app: web.Application):
    for ws in list(UI_CLIENTS):
        await ws.close()
    UI_CLIENTS.clear()
    SUBS.clear()

    app["sender_task"].cancel()
    try:
        await app["sender_task"]
    except Exception:
        pass

    await app["analyzer"].close()
    app["artnet"].close()
    save_index(app["upload_index"])


def create_app() -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["analyzer"] = AudioAnalyzer()
    app["artnet"] = ArtNetSender(TARGET_IP, UNIVERSE, DMX_CHANNELS)

    # load index + restore registry from disk
    app["upload_index"] = load_index()
    rebuild_registry_from_uploads(app["analyzer"], app["upload_index"])
    save_index(app["upload_index"])  # remove stale entries if any

    app.router.add_get("/health", health)
    app.router.add_post("/upload", upload)
    app.router.add_get("/ws", ws_handler)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host=WS_HOST, port=PORT)
