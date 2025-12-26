# lumina_dsp/dsp/core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class DSPParams:
    fps: int = 60
    block: int = 1024
    fft_size: int = 2048
    num_bands: int = 8
    mock_events: bool = True

    # --- Meter scaling (UI-friendly) ---
    # Convert FFT magnitude to dBFS-ish scale and map to 0..1:
    #   db_min..db_max -> 0..1
    db_min: float = -80.0
    db_max: float = 0.0

    # Optional gentle compression for visuals (0 = off)
    # Applied after 0..1 mapping: y = y ** gamma (gamma < 1 boosts quiet signals)
    band_gamma: float = 0.65

    # Optional temporal smoothing for bands (0..1): 0=no smoothing, closer to 1=more smoothing
    band_smoothing: float = 0.35


class DSPCore:
    """
    Чистая DSP-логика:
      - meter: rms/peak + FFT bands (0..1)
      - mock event: kick/snare/hihat/energy/bpm (пока)

    ВАЖНО (исправление "analysis_gain не влияет"):
    Раньше спектр нормализовался на max(spec) внутри каждого кадра:
        spec = spec / max(spec)
    Это делало bands практически инвариантными к усилению (gain).
    Теперь bands считаются в dB-шкале и маппятся в 0..1 без per-frame max-normalize,
    поэтому analysis_gain реально влияет на визуализацию (bands), не меняя протокол.
    """

    def __init__(self, params: DSPParams | None = None):
        self.params = params or DSPParams()

        # state for smoothing
        self._prev_bands: np.ndarray | None = None

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)

    def compute_meter(self, x: np.ndarray) -> Dict[str, object]:
        """
        Возвращает:
          { "rms": float, "peak": float, "bands": list[float] }

        x: numpy float32 array shape (frames, channels) или (frames,)
        Ожидается диапазон примерно [-1..1] (после sanitize в AudioAnalyzer).
        """
        # --- mono ---
        mono = x.mean(axis=1) if x.ndim == 2 else x
        mono = mono.astype(np.float32, copy=False)

        # --- time-domain meters ---
        # +eps to avoid exact zeros (UI/logic stability)
        rms = float(np.sqrt(np.mean(mono * mono)) + 1e-12) if mono.size else 1e-12
        peak = float(np.max(np.abs(mono)) + 1e-12) if mono.size else 1e-12

        # --- FFT prep ---
        n = int(self.params.fft_size)
        if n <= 0:
            return {"rms": rms, "peak": peak, "bands": [0.0] * max(1, int(self.params.num_bands))}

        if mono.size < n:
            mono2 = np.zeros(n, dtype=np.float32)
            mono2[: mono.size] = mono
        else:
            mono2 = mono[-n:]

        win = np.hanning(n).astype(np.float32)
        spec = np.abs(np.fft.rfft(mono2 * win)).astype(np.float32, copy=False)

        # --- Convert to dB scale (UI-friendly, gain-sensitive) ---
        # Rough amplitude normalization to make magnitude less dependent on FFT size
        # (Not a strict dBFS reference, but stable enough for visualization.)
        spec = spec * (2.0 / float(n))

        eps = 1e-12
        db = 20.0 * np.log10(spec + eps)  # negative values for typical audio

        # Map db_min..db_max -> 0..1
        db_min = float(self.params.db_min)
        db_max = float(self.params.db_max)
        if db_max <= db_min:
            db_min, db_max = -80.0, 0.0

        spec01 = (db - db_min) / (db_max - db_min)
        spec01 = np.clip(spec01, 0.0, 1.0)

        # Optional gamma to boost quiet content visually
        gamma = float(self.params.band_gamma)
        if gamma > 0.0 and gamma != 1.0:
            spec01 = np.power(spec01, gamma)

        # --- banding ---
        bins = int(spec01.size)
        nb = max(1, int(self.params.num_bands))
        edges = np.linspace(0, bins - 1, nb + 1).astype(int)

        bands = np.zeros(nb, dtype=np.float32)
        for i in range(nb):
            a = int(edges[i])
            b = int(max(edges[i + 1], edges[i] + 1))
            # mean energy in this band
            bands[i] = float(np.mean(spec01[a:b])) if b > a else 0.0

        # --- temporal smoothing (optional) ---
        s = float(self.params.band_smoothing)
        if 0.0 < s < 1.0:
            if self._prev_bands is None or self._prev_bands.shape != bands.shape:
                self._prev_bands = bands.copy()
            else:
                self._prev_bands = (s * self._prev_bands) + ((1.0 - s) * bands)
            bands_out = self._prev_bands
        else:
            bands_out = bands

        return {"rms": rms, "peak": peak, "bands": [float(v) for v in bands_out.tolist()]}

    def mock_event(self, meter: Dict[str, object]) -> Dict[str, float]:
        """
        Супер-простой mock, чтобы UI жил.
        """
        rms = float(meter.get("rms", 0.0))
        peak = float(meter.get("peak", 0.0))
        bands = meter.get("bands", [])
        if not isinstance(bands, list) or len(bands) == 0:
            bands = [0.0] * max(1, int(self.params.num_bands))

        e = rms
        kick = float(np.clip((e - 0.05) * 10.0, 0.0, 1.0))
        snare = float(np.clip((peak - 0.10) * 5.0, 0.0, 1.0))
        hihat = float(np.clip(float(np.mean(bands[-2:])) * 2.0, 0.0, 1.0))
        bpm = 128.0

        return {
            "kick": kick,
            "snare": snare,
            "hihat": hihat,
            "energy": float(np.clip(e * 3.0, 0.0, 1.0)),
            "bpm": bpm,
        }
