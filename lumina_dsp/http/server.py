# lumina_dsp/http/server.py
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Optional

import soundfile as sf
from aiohttp import web

from lumina_dsp.audio.file_player import FileInfo, FileRegistry


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """
    CORS нужен, потому что Google AI Studio Preview = другой origin,
    и браузер делает preflight OPTIONS перед POST /upload.
    """
    if request.method == "OPTIONS":
        resp = web.Response(status=200)
    else:
        resp = await handler(request)

    # Для разработки ок ставить "*"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp


@dataclass
class HttpServer:
    host: str
    port: int
    upload_dir: str
    registry: FileRegistry

    _runner: Optional[web.AppRunner] = None

    async def start(self) -> web.AppRunner:
        os.makedirs(self.upload_dir, exist_ok=True)

        app = web.Application(middlewares=[cors_middleware])
        app.router.add_post("/upload", self.upload)
        app.router.add_options("/upload", self.options_ok)
        app.router.add_get("/health", self.health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        self._runner = runner
        print(f"HTTP ✅ listening on http://{self.host}:{self.port} (/upload)")
        return runner

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def options_ok(self, request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "upload": "/upload"})

    async def upload(self, request: web.Request) -> web.Response:
        """
        POST /upload
        multipart/form-data:
          field "file": binary

        Response:
          { ok, fileId, sourceId, name, sr, channels, duration }
        """
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "file":
            return web.json_response({"ok": False, "error": "missing form field 'file'"}, status=400)

        filename = part.filename or "upload.bin"
        file_id = uuid.uuid4().hex
        safe_name = filename.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(self.upload_dir, f"{file_id}__{safe_name}")

        size = 0
        with open(out_path, "wb") as f:
            while True:
                chunk = await part.read_chunk()
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)

        # validate as audio
        try:
            with sf.SoundFile(out_path, mode="r") as snd:
                sr = int(snd.samplerate)
                ch = int(snd.channels)
                frames = int(len(snd))
                dur = float(frames) / float(sr) if sr > 0 else 0.0
        except Exception as e:
            try:
                os.remove(out_path)
            except Exception:
                pass
            return web.json_response({"ok": False, "error": f"not an audio file: {e}"}, status=400)

        info = FileInfo(
            file_id=file_id,
            path=out_path,
            name=safe_name,
            sr=sr,
            channels=ch,
            frames=frames,
            duration=dur,
        )
        self.registry.add(info)

        print(f"HTTP 📥 uploaded {safe_name} -> file:{file_id} dur={dur:.2f}s sr={sr} ch={ch}")

        return web.json_response(
            {
                "ok": True,
                "fileId": file_id,
                "sourceId": f"file:{file_id}",
                "name": safe_name,
                "sr": sr,
                "channels": ch,
                "duration": dur,
                "bytes": size,
            }
        )
