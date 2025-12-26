# lumina_dsp/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSPConfig:
    # DSP
    fps: int = 60
    block: int = 1024
    fft_size: int = 2048
    num_bands: int = 8
    mock_events: bool = True


@dataclass
class ServerConfig:
    # WS (telemetry/control)
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    ws_endpoint: str = "/audio"

    # HTTP (file upload / health)
    http_host: str = "0.0.0.0"
    http_port: int = 8766

    # storage
    upload_dir: str = "./_uploads"
