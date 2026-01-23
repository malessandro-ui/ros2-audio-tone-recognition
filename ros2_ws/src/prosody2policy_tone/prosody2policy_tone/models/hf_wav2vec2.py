"""
HuggingFace wav2vec2 prosody backend (optional).

This is intentionally optional so the repo stays lightweight by default.
If you want to use it:
  pip install torch transformers

This backend demonstrates how you would:
- extract embeddings from a pretrained speech encoder
- run a tiny head to predict arousal/valence/confidence

NOTE: The head here is a placeholder. For a serious implementation:
- train a lightweight regressor/classifier head
- calibrate confidence (see tools/calibrate_confidence.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ToneEstimate, ProsodyBackend, clamp, safe_prob


@dataclass
class HFWav2Vec2Config:
    model_name: str = "facebook/wav2vec2-base-960h"
    device: str = "cpu"  # "cuda" if available on Linux GPU
    # Placeholder scaling
    arousal_scale: float = 1.0
    valence_scale: float = 1.0


class HFWav2Vec2Backend(ProsodyBackend):
    def __init__(self, cfg: HFWav2Vec2Config | None = None):
        self.cfg = cfg or HFWav2Vec2Config()
        try:
            import torch  # type: ignore
            from transformers import Wav2Vec2Model, Wav2Vec2Processor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "HFWav2Vec2Backend requires `torch` and `transformers`.\n"
                "Install with: pip install torch transformers\n"
                f"Import error: {e}"
            )

        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(self.cfg.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.cfg.model_name).to(self.cfg.device)
        self.model.eval()

        # Placeholder linear head weights (random) — replace with trained weights.
        # This exists to show the software wiring.
        self.head_w = None
        self.head_b = None

    def _predict_from_embedding(self, emb: np.ndarray) -> tuple[float, float, float]:
        # Placeholder: map embedding mean to stable-ish outputs
        m = float(np.mean(emb))
        arousal = clamp(m * 0.5 * self.cfg.arousal_scale, -1.0, 1.0)
        valence = clamp(-m * 0.3 * self.cfg.valence_scale, -1.0, 1.0)
        confidence = safe_prob(0.5 + 0.4 * abs(m))
        return arousal, valence, confidence

    def process_audio(self, audio_f32, sample_rate: int) -> ToneEstimate:
        x = np.asarray(audio_f32, dtype=np.float32)

        inputs = self.processor(x, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            out = self.model(**inputs)
            # last_hidden_state: [B, T, C]
            h = out.last_hidden_state[0].detach().cpu().numpy()
            emb = np.mean(h, axis=0)  # [C]

        arousal, valence, confidence = self._predict_from_embedding(emb)

        return ToneEstimate(
            arousal=float(arousal),
            valence=float(valence),
            confidence=float(confidence),
            speech_prob=None,
            rms_db=None,
            f0_hz=None,
        )
