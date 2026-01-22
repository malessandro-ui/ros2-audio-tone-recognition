from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class StreamConfig:
    sample_rate: int = 16000
    window_s: float = 1.2
    smoothing: float = 0.75
    silence_conf: float = 0.05
    emb_dim: int = 64

class StreamingEstimator:
    """
    Holds a rolling audio window and produces (embedding, arousal, valence, confidence).
    In scaffold mode, generates deterministic signals derived from audio energy.
    """
    def __init__(self, cfg: StreamConfig):
        self.cfg = cfg
        self.win = int(cfg.window_s * cfg.sample_rate)
        self.buf = deque(maxlen=self.win)

        self.emb = np.zeros((cfg.emb_dim,), dtype=np.float32)
        self.arousal = 0.0
        self.valence = 0.0
        self.conf = 0.0

    def _ema(self, new_emb, new_ar, new_va, new_cf):
        a = self.cfg.smoothing
        self.emb = a * self.emb + (1.0 - a) * new_emb
        self.arousal = a * self.arousal + (1.0 - a) * new_ar
        self.valence = a * self.valence + (1.0 - a) * new_va
        self.conf = a * self.conf + (1.0 - a) * new_cf

    def push(self, block: np.ndarray, speech_prob: float) -> float | None:
        for s in block.tolist():
            self.buf.append(float(s))
        if len(self.buf) < self.win:
            return None

        wav = np.array(self.buf, dtype=np.float32)
        rms = float(np.sqrt(np.mean(wav**2) + 1e-12))

        if speech_prob < 0.2 or rms < 0.005:
            self._ema(np.zeros_like(self.emb), 0.0, 0.0, self.cfg.silence_conf)
            return rms

        # Scaffold “inference”: energy -> arousal, sign-of-mean -> valence
        ar = float(np.clip((rms - 0.01) * 20.0, -1.0, 1.0))
        va = float(np.clip(np.mean(wav) * 10.0, -1.0, 1.0))
        cf = float(np.clip(0.5 + 0.5 * speech_prob, 0.0, 1.0))

        emb = np.zeros_like(self.emb)
        emb[0], emb[1] = ar, va
        self._ema(emb, ar, va, cf)
        return rms
