#!/usr/bin/env python3
"""
tools/sim_adapter.py

Pure-Python simulator for "audio tone -> style params" adapter behavior.  
- Tbh, this is designed to mirror the behavior of the ROS2 adapter node at a high level on native OS:
  - confidence gating
  - stale timeout fallback to neutral
  - hysteresis thresholds to avoid flicker
  - minimum mode hold time
Usage:
  python tools/sim_adapter.py --scenario scripted
  python tools/sim_adapter.py --scenario random --steps 40 --hz 10
  python tools/sim_adapter.py --scenario smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADAPTER_PKG_PATH = os.path.join(
    REPO_ROOT, "ros2_ws", "src", "prosody2policy_adapter"
)
if ADAPTER_PKG_PATH not in sys.path:
    sys.path.insert(0, ADAPTER_PKG_PATH)

try:
    from prosody2policy_adapter.mode_logic import (  # type: ignore
        Thresholds,
        StyleMap,
        compute_mode,
        params_for_mode,
    )
except Exception as e:
    print(
        "ERROR: Failed to import adapter logic.\n"
        "Expected file: ros2_ws/src/prosody2policy_adapter/prosody2policy_adapter/mode_logic.py\n"
        f"Import error: {e}",
        file=sys.stderr,
    )
    sys.exit(2)


@dataclass
class ToneSignal:
    """Minimal stand-in for prosody2policy_msgs/ToneSignal."""
    stamp_s: float
    arousal: float      # e.g., [-1, 1]
    valence: float      # e.g., [-1, 1]
    confidence: float   # [0, 1]


@dataclass
class StyleParams:
    """Minimal stand-in for prosody2policy_msgs/StyleParams."""
    stamp_s: float
    mode: str
    velocity_scale: float
    accel_scale: float
    approach_offset_m: float
    stiffness_scale: float
    grip_force_cap: float


@dataclass
class AdapterConfig:
    # Robustness gates
    min_conf: float = 0.25
    stale_s: float = 1.0
    min_mode_hold_s: float = 1.5

    # Hysteresis thresholds
    urgent_enter: float = 0.40
    urgent_exit: float = 0.25
    negative_enter: float = -0.30
    negative_exit: float = -0.15
    calm_arousal: float = -0.15

    # Style mapping (defaults; your StyleMap may override/define these)
    vel_neutral: float = 1.0
    vel_urgent: float = 1.4
    vel_backoff: float = 0.7
    acc_neutral: float = 1.0
    acc_urgent: float = 1.3
    acc_backoff: float = 0.8
    offset_neutral_m: float = 0.0
    offset_backoff_m: float = 0.10
    stiff_neutral: float = 1.0
    stiff_calm: float = 0.85
    stiff_urgent: float = 1.10

    @staticmethod
    def from_json(path: str) -> "AdapterConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return AdapterConfig(**d)


def _try_load_yaml(path: str) -> Optional[Dict]:
    """Optional YAML loader; avoids hard dependency if not installed."""
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_adapter_config(path: Optional[str]) -> AdapterConfig:
    """
    Supports:
      - JSON file: {"min_conf": ..., ...}
      - YAML file shaped like your ROS params, e.g.:
          tone_to_style_node:
            ros__parameters:
              min_conf: 0.25
              ...
    """
    if not path:
        return AdapterConfig()

    if path.endswith(".json"):
        return AdapterConfig.from_json(path)

    if path.endswith(".yaml") or path.endswith(".yml"):
        data = _try_load_yaml(path)
        if data is None:
            print(
                f"NOTE: PyYAML not installed; ignoring YAML config at {path}. "
                "Run: pip install pyyaml  (or let CI install it).",
                file=sys.stderr,
            )
            return AdapterConfig()

        # Support ROS2 param file layout
        params = data
        if isinstance(data, dict):
            node = data.get("tone_to_style_node")
            if isinstance(node, dict):
                ros_params = node.get("ros__parameters")
                if isinstance(ros_params, dict):
                    params = ros_params

        if not isinstance(params, dict):
            print(f"WARNING: Unexpected YAML structure in {path}. Using defaults.", file=sys.stderr)
            return AdapterConfig()
        cfg_kwargs = {k: v for k, v in params.items() if hasattr(AdapterConfig(), k)}
        return AdapterConfig(**cfg_kwargs)

    print(f"WARNING: Unknown config format for {path}. Using defaults.", file=sys.stderr)
    return AdapterConfig()


class AdapterState:
    def __init__(self, cfg: AdapterConfig):
        self.cfg = cfg
        self.mode: str = "neutral"
        self.last_mode_change_s: float = 0.0
        self.last_msg_s: Optional[float] = None

        self.thresholds = Thresholds(
            urgent_enter=cfg.urgent_enter,
            urgent_exit=cfg.urgent_exit,
            negative_enter=cfg.negative_enter,
            negative_exit=cfg.negative_exit,
            calm_arousal=cfg.calm_arousal,
        )

        self.style_map = StyleMap()

    def _hold_ok(self, now_s: float) -> bool:
        return (now_s - self.last_mode_change_s) >= self.cfg.min_mode_hold_s

    def _neutral_params(self, now_s: float) -> StyleParams:
        v, a, off, stiff, grip = params_for_mode("neutral", self.style_map)
        return StyleParams(
            stamp_s=now_s,
            mode="neutral",
            velocity_scale=float(v),
            accel_scale=float(a),
            approach_offset_m=float(off),
            stiffness_scale=float(stiff),
            grip_force_cap=float(grip),
        )

    def step(self, msg: Optional[ToneSignal], now_s: float) -> StyleParams:
        """
        msg=None simulates no incoming tone message at this timestep.
        """

        # Stale handling: if we haven't seen a message in stale_s -> neutral
        if msg is None:
            if self.last_msg_s is None:
                return self._neutral_params(now_s)
            if (now_s - self.last_msg_s) > self.cfg.stale_s:
                if self.mode != "neutral":
                    self.mode = "neutral"
                    self.last_mode_change_s = now_s
                return self._neutral_params(now_s)
            # Not stale yet: keep previous mode's params
            v, a, off, stiff, grip = params_for_mode(self.mode, self.style_map)
            return StyleParams(now_s, self.mode, float(v), float(a), float(off), float(stiff), float(grip))

        # Got a message
        self.last_msg_s = msg.stamp_s

        # Confidence gate
        if msg.confidence < self.cfg.min_conf:
            if self.mode != "neutral" and self._hold_ok(now_s):
                self.mode = "neutral"
                self.last_mode_change_s = now_s
            return self._neutral_params(now_s)

        # Compute proposed mode using hysteresis
        proposed = compute_mode(self.mode, arousal=msg.arousal, valence=msg.valence, th=self.thresholds)

        # Minimum hold time prevents rapid flips
        if proposed != self.mode and self._hold_ok(now_s):
            self.mode = proposed
            self.last_mode_change_s = now_s

        v, a, off, stiff, grip = params_for_mode(self.mode, self.style_map)
        return StyleParams(
            stamp_s=now_s,
            mode=self.mode,
            velocity_scale=float(v),
            accel_scale=float(a),
            approach_offset_m=float(off),
            stiffness_scale=float(stiff),
            grip_force_cap=float(grip),
        )


def scripted_scenario(steps: int, hz: float, start_s: float) -> List[Optional[ToneSignal]]:
    """
    A deterministic scenario:
      - neutral
      - urgent ramp (arousal high)
      - negative/backoff (valence low)
      - calm (arousal low)
      - stale gap
    """
    dt = 1.0 / hz
    out: List[Optional[ToneSignal]] = []

    def mk(i: int, a: float, v: float, c: float) -> ToneSignal:
        t = start_s + i * dt
        return ToneSignal(stamp_s=t, arousal=a, valence=v, confidence=c)

    # 0..9 neutral
    for i in range(min(10, steps)):
        out.append(mk(i, a=0.0, v=0.0, c=0.9))

    idx = len(out)

    # urgent burst 10 frames
    for j in range(10):
        if idx >= steps:
            break
        out.append(mk(idx, a=0.6, v=0.1, c=0.95))
        idx += 1

    # negative/backoff 10 frames
    for j in range(10):
        if idx >= steps:
            break
        out.append(mk(idx, a=0.1, v=-0.5, c=0.95))
        idx += 1

    # calm 10 frames
    for j in range(10):
        if idx >= steps:
            break
        out.append(mk(idx, a=-0.4, v=0.2, c=0.95))
        idx += 1

    # stale gap (no messages)
    for j in range(10):
        if idx >= steps:
            break
        out.append(None)
        idx += 1

    # finish neutral
    while idx < steps:
        out.append(mk(idx, a=0.0, v=0.0, c=0.9))
        idx += 1

    return out


def random_scenario(steps: int, hz: float, start_s: float) -> List[Optional[ToneSignal]]:
    dt = 1.0 / hz
    out: List[Optional[ToneSignal]] = []
    for i in range(steps):
        t = start_s + i * dt

        # Occasionally drop messages to test stale behavior and everything gucci
        if random.random() < 0.08:
            out.append(None)
            continue

        # Biased random for more interesting transitions
        a = max(-1.0, min(1.0, random.gauss(0.0, 0.35)))
        v = max(-1.0, min(1.0, random.gauss(0.0, 0.35)))

        # Sometimes spike urgent/negative
        if random.random() < 0.07:
            a = 0.7
        if random.random() < 0.07:
            v = -0.7

        c = max(0.0, min(1.0, random.gauss(0.9, 0.15)))
        out.append(ToneSignal(stamp_s=t, arousal=a, valence=v, confidence=c))
    return out


def print_row(i: int, msg: Optional[ToneSignal], sp: StyleParams) -> None:
    if msg is None:
        a = v = c = float("nan")
        mtag = "None"
    else:
        a, v, c = msg.arousal, msg.valence, msg.confidence
        mtag = "Tone"

    print(
        f"{i:03d} | {mtag:4s} | a={a:+.2f} v={v:+.2f} c={c:.2f} "
        f"-> mode={sp.mode:8s} vel={sp.velocity_scale:.2f} acc={sp.accel_scale:.2f} "
        f"off={sp.approach_offset_m:.3f} stiff={sp.stiffness_scale:.2f} grip={sp.grip_force_cap:.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["scripted", "random", "smoke"], default="scripted")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=str, default=None, help="Optional path to adapter.yaml or adapter.json")
    ap.add_argument("--sleep", action="store_true", help="Sleep in real time (for live-ish prints)")
    args = ap.parse_args()

    random.seed(args.seed)

    cfg = load_adapter_config(args.config)
    state = AdapterState(cfg)

    start_s = time.time()
    if args.scenario == "scripted":
        seq = scripted_scenario(args.steps, args.hz, start_s)
    elif args.scenario == "random":
        seq = random_scenario(args.steps, args.hz, start_s)
    else:  # smoke
        seq = scripted_scenario(12, args.hz, start_s)

    dt = 1.0 / args.hz
    now_s = start_s

    print("idx | src  | input(a,v,c) -> output(mode + style params)")
    print("-" * 110)

    for i, msg in enumerate(seq):
        # For msg, keep stamp aligned to now_s if provided
        if msg is not None:
            msg = ToneSignal(stamp_s=now_s, arousal=msg.arousal, valence=msg.valence, confidence=msg.confidence)

        sp = state.step(msg, now_s)
        print_row(i, msg, sp)

        now_s += dt
        if args.sleep:
            time.sleep(dt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# sim_adapter.py
