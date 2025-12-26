# v3/lumina_dsp/state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any


# Callback для телеметрии:
# AudioAnalyzer будет дергать его для отправки событий наружу (WS)
EventCallback = Callable[[Dict[str, Any]], None]


@dataclass
class RuntimeState:
    """
    Runtime state аудио/DSP пайплайна.

    ВАЖНО:
    - Это НЕ UI-state
    - Это НЕ конфигурация ML
    - Только текущее состояние движка
    """

    # === Audio source ===
    source_id: Optional[str] = None   # "device:0", "file:<id>", etc.

    # === DSP running state ===
    running: bool = False

    # === DSP params (контракт с UI, НЕ ЛОМАТЬ) ===
    fps: int = 40

    bands: int = 32
    fft_size: int = 2048
    block_size: int = 1024

    analysis_gain: float = 200.0
    gate_rms: float = 0.0

    mock_events: bool = False

    # === Monitor ===
    monitor_enabled: bool = True

    # === File playback state (используется FilePlayer) ===
    file_id: Optional[str] = None

    # === Internal flags ===
    shutting_down: bool = False
