# v3/lumina_dsp/app.py
"""
v3 unified: NO websockets / NO http server.

ВАЖНО:
- Этот модуль теперь фасад (compat layer).
- Вся логика аудио/DSP/ML живёт в engine.py
- Внешний контракт импорта сохраняем: AudioAnalyzer остаётся доступным из lumina_dsp.app
"""

from __future__ import annotations

from .state import RuntimeState, EventCallback

# Основная реализация переехала в engine.py.
# Сохраняем имя AudioAnalyzer для совместимости с существующим кодом.
from .engine import AudioEngine as AudioAnalyzer  # noqa: F401

__all__ = [
    "AudioAnalyzer",
    "RuntimeState",
    "EventCallback",
]
