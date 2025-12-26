# v3/lumina_dsp/control.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional


def _ack(req_id: Optional[str], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"type": "ack", "reqId": req_id, "payload": payload or {"ok": True}}


def _err(req_id: Optional[str], code: str, message: str) -> Dict[str, Any]:
    return {"type": "error", "reqId": req_id, "payload": {"code": code, "message": message}}


async def handle_control(analyzer: Any, msg: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    UI -> Backend control router.

    Вход:  { "type": "...", "payload": {...}, "reqId": "..." }
    Выход: response envelope (ack/error/audio_sources/audio_status/audio_file_status)

    ВАЖНО:
    - Никакого inference / тяжелых операций.
    - Никаких блокировок аудио/DSР: только await на методы analyzer/file_player.
    - Telemetry push (audio_meter/dsp_event/ai_classifier_event/audio_file_status) не отсюда,
      а из analyzer через event callback.
    """
    t = msg.get("type")
    req = msg.get("reqId")
    payload = msg.get("payload", {}) or {}

    # -------------------- sources --------------------
    if t == "audio_list_sources":
        return {"type": "audio_sources", "payload": analyzer.list_sources_payload(), "reqId": req}

    if t == "audio_set_source":
        sid = str(payload.get("sourceId", "device:default"))
        try:
            await analyzer.set_source(sid)
            return {"type": "ack", "reqId": req, "payload": {"ok": True, "sourceId": sid}}
        except Exception as e:
            return _err(req, "AUDIO_SET_SOURCE_FAILED", str(e))

    # -------------------- engine start/stop --------------------
    if t == "audio_start":
        # совместимое расширение: если UI прислал fps, попробуем применить
        if "fps" in payload:
            try:
                analyzer.dsp_core.set_params(fps=int(payload["fps"]))
            except Exception:
                pass

        analyzer.state.running = True
        return {
            "type": "audio_status",
            "payload": {"state": "running", "sourceId": analyzer.state.source_id},
            "reqId": req,
        }

    if t == "audio_stop":
        analyzer.state.running = False
        try:
            # как было: при стопе файла можно закрыть output stream
            if str(analyzer.state.source_id).startswith("file:"):
                analyzer._close_output_stream()
        except Exception:
            pass

        return {
            "type": "audio_status",
            "payload": {"state": "stopped", "sourceId": analyzer.state.source_id},
            "reqId": req,
        }

    # -------------------- dsp params --------------------
    if t == "dsp_set_params":
        params: Dict[str, Any] = {}

        if "bands" in payload:
            params["num_bands"] = int(payload["bands"])
        if "fftSize" in payload:
            params["fft_size"] = int(payload["fftSize"])

        if "block" in payload:
            b = int(payload["block"])
            analyzer.dsp_cfg.block = b
            analyzer.file_player.block = b
            params["block"] = b

        if "mockEvents" in payload:
            params["mock_events"] = bool(payload["mockEvents"])

        # расширение без ломки: monitor/analysisGain/gateRms (UI может игнорировать)
        if "monitor" in payload:
            try:
                analyzer._monitor_enabled = bool(payload["monitor"])
                if not analyzer._monitor_enabled:
                    analyzer._close_output_stream()
            except Exception:
                pass

        if "analysisGain" in payload:
            try:
                analyzer._analysis_gain = max(0.1, float(payload["analysisGain"]))
            except Exception:
                pass

        if "gateRms" in payload:
            try:
                analyzer._gate_rms = max(0.0, float(payload["gateRms"]))
            except Exception:
                pass

        try:
            analyzer.dsp_core.set_params(**params)
        except Exception:
            pass

        return _ack(req, {"ok": True})

    # -------------------- file controls --------------------
    if t == "audio_file_play":
        try:
            await analyzer.file_player.play()
            await analyzer.emit_file_status()
            return _ack(req, {"ok": True})
        except Exception as e:
            return _err(req, "FILE_PLAY_FAILED", str(e))

    if t == "audio_file_pause":
        await analyzer.file_player.pause()
        await analyzer.emit_file_status()
        return _ack(req, {"ok": True})

    if t == "audio_file_stop":
        await analyzer.file_player.stop()
        try:
            analyzer._close_output_stream()
        except Exception:
            pass
        await analyzer.emit_file_status()
        return _ack(req, {"ok": True})

    if t == "audio_file_seek":
        try:
            sec = float(payload.get("seconds", 0.0))
            await analyzer.file_player.seek(sec)
            await analyzer.emit_file_status()
            return _ack(req, {"ok": True})
        except Exception as e:
            return _err(req, "FILE_SEEK_FAILED", str(e))

    if t == "audio_file_status_get":
        return {
            "type": "audio_file_status",
            "payload": await analyzer.file_player.get_status_payload(),
            "reqId": req,
            "ts": time.time(),
        }

    # subscribe/unsubscribe обычно обрабатываются в ws сервере,
    # но для полной совместимости отвечаем ack и здесь.
    if t in ("subscribe", "unsubscribe"):
        return _ack(req, {"ok": True})

    # -------------------- unknown --------------------
    return _err(req, "UNKNOWN_MSG", str(t))
