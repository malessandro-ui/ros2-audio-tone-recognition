from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import numpy as np

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    block_ms: int = 20
    channels: int = 1

def stream_audio(cfg: AudioConfig) -> Iterator[np.ndarray]:
    """
    Stub interface: yields float32 mono blocks of shape (N,).
    Implement with sounddevice/portaudio on Linux/macOS, or read from WAV for offline tests.
    """
    raise NotImplementedError("Audio streaming not implemented in scaffold. Use dummy mode or implement sounddevice backend.")
