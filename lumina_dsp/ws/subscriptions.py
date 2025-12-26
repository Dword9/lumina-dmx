# lumina_dsp/ws/subscriptions.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set

from websockets.server import ServerConnection


@dataclass
class Subscriptions:
    """
    Хранит подписки WS-клиентов на стримы данных.

    stream_name -> set(ws_connections)

    Примеры stream:
      - "audio_meter"
      - "dsp_event"
      - "audio_file_status"
    """

    by_stream: Dict[str, Set[ServerConnection]] = field(default_factory=dict)

    def add(self, stream: str, ws: ServerConnection) -> None:
        self.by_stream.setdefault(stream, set()).add(ws)

    def remove_ws(self, ws: ServerConnection) -> None:
        for stream in list(self.by_stream.keys()):
            self.by_stream[stream].discard(ws)
            if not self.by_stream[stream]:
                del self.by_stream[stream]

    def get(self, stream: str) -> Set[ServerConnection]:
        return self.by_stream.get(stream, set())
