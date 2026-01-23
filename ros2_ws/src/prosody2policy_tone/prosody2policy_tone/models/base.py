"""
Prosody model backends (pluggable).

The idea:
- Keep tone inference independent from ROS node concerns.
- Provide a small, stable interface so you can swap:
  - a simple heuristic baseline
  - a learned model (HF wav2vec2/hubert/etc.)
  - a custom on-device model

This module is pure Python and can be tested without ROS2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Tuple


@dataclass(frozen=True)
class ToneEstimate:
    """A minimal, calibrated tone estimate."""
    arousal: float      # typically normalized to [-1, 1]
    valence: float      # typically normalized to [-1, 1]
    confidence: float   # [0, 1], calibrated
    # Optional diagnostic fields
    speech_prob: Optional[float] = None
    rms_db: Optional[float] = None
    f0_hz: Optional[float] = None


class ProsodyBackend(Protocol):
    """
    Prosody backend protocol.

    Implementations should be:
      - streaming-friendly
      - low-latency
      - robust to silence/noise

    `process_audio` takes a short audio window and returns an estimate.
    """
    def process_audio(self, audio_f32, sample_rate: int) -> ToneEstimate:
        ...


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def safe_prob(x: float) -> float:
    """Clamp to a safe [0,1] range."""
    return clamp(float(x), 0.0, 1.0)
