from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class VadConfig:
    aggressiveness: int = 2
    sample_rate: int = 16000

class Vad:
    """
    Stub VAD. Replace with webrtcvad (recommended) in Linux runtime.
    """
    def __init__(self, cfg: VadConfig):
        self.cfg = cfg

    def speech_prob(self, block: np.ndarray) -> float:
        # placeholder: assume speech present if RMS above threshold
        rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2) + 1e-12))
        return 1.0 if rms > 0.01 else 0.0
