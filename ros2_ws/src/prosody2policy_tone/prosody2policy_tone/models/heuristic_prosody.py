"""
Heuristic prosody backend.

This is a strong "engineering baseline":
- no ML dependencies
- runs everywhere
- provides stable outputs (with smoothing hooks)
- good for plumbing, integration tests, and demos

What it does:
- Arousal correlates with energy (RMS in dB) and pitch variability proxy.
- Valence is a placeholder (0.0) by default; you can map spectral tilt, etc.
- Confidence decreases when audio is silent or pitch estimate is unreliable.

NOTE: This is *not* state-of-the-art prosody detection; it's a robust placeholder
that demonstrates correct software architecture.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from .base import ToneEstimate, ProsodyBackend, clamp, safe_prob


def rms_db(audio_f32: np.ndarray, eps: float = 1e-12) -> float:
    audio_f32 = np.asarray(audio_f32, dtype=np.float32)
    r = float(np.sqrt(np.mean(audio_f32 * audio_f32) + eps))
    return 20.0 * math.log10(max(r, eps))


def autocorr_pitch_hz(audio_f32: np.ndarray, sr: int,
                      fmin: float = 70.0, fmax: float = 300.0) -> Tuple[Optional[float], float]:
    """
    Very lightweight pitch estimate via autocorrelation.

    Returns: (f0_hz or None, quality in [0,1]).
    """
    x = np.asarray(audio_f32, dtype=np.float32)
    if x.size < int(sr * 0.02):
        return None, 0.0

    # Remove DC and apply a mild window
    x = x - float(np.mean(x))
    x = x * np.hanning(x.size).astype(np.float32)

    # Autocorrelation
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    if not np.isfinite(ac[0]) or ac[0] <= 1e-8:
        return None, 0.0

    # Search lag range for plausible pitch
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    lag_max = min(lag_max, ac.size - 1)
    if lag_max <= lag_min + 2:
        return None, 0.0

    seg = ac[lag_min:lag_max]
    peak_i = int(np.argmax(seg)) + lag_min
    peak_v = float(ac[peak_i])

    # Quality: normalized peak height vs zero-lag energy
    q = clamp(peak_v / float(ac[0]), 0.0, 1.0)
    if q < 0.15:
        return None, q

    f0 = float(sr / peak_i) if peak_i > 0 else None
    return f0, q


@dataclass
class HeuristicProsodyConfig:
    # RMS → arousal mapping
    rms_db_silence: float = -60.0
    rms_db_loud: float = -15.0

    # Arousal scaling weights
    w_rms: float = 0.85
    w_pitch: float = 0.15

    # Confidence shaping
    conf_silence_db: float = -50.0
    conf_full_db: float = -25.0


class HeuristicProsodyBackend(ProsodyBackend):
    def __init__(self, cfg: HeuristicProsodyConfig | None = None):
        self.cfg = cfg or HeuristicProsodyConfig()

    def process_audio(self, audio_f32, sample_rate: int) -> ToneEstimate:
        x = np.asarray(audio_f32, dtype=np.float32)

        db = rms_db(x)
        f0, q = autocorr_pitch_hz(x, sample_rate)

        # Normalize RMS into [0,1]
        db01 = (db - self.cfg.rms_db_silence) / (self.cfg.rms_db_loud - self.cfg.rms_db_silence)
        db01 = clamp(db01, 0.0, 1.0)

        # Pitch proxy into [0,1] using quality only (simple, robust)
        pitch01 = clamp(q, 0.0, 1.0)

        # Arousal in [-1,1]
        a01 = self.cfg.w_rms * db01 + self.cfg.w_pitch * pitch01
        arousal = 2.0 * clamp(a01, 0.0, 1.0) - 1.0

        # Valence placeholder: neutral.
        # You can extend this with spectral tilt, formant spread, etc.
        valence = 0.0

        # Confidence: depends on signal energy + pitch quality
        c_db01 = (db - self.cfg.conf_silence_db) / (self.cfg.conf_full_db - self.cfg.conf_silence_db)
        c_db01 = clamp(c_db01, 0.0, 1.0)
        confidence = safe_prob(0.7 * c_db01 + 0.3 * pitch01)

        # Speech probability: rough proxy using RMS
        speech_prob = safe_prob(db01)

        return ToneEstimate(
            arousal=float(arousal),
            valence=float(valence),
            confidence=float(confidence),
            speech_prob=float(speech_prob),
            rms_db=float(db),
            f0_hz=float(f0) if f0 is not None else None,
        )
