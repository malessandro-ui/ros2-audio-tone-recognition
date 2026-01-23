#!/usr/bin/env python3
"""
tools/replay_tone_csv.py

Replay ToneSignal-like CSV through the adapter logic (no ROS2).
Writes StyleParams-like CSV.

Inputs:
- CSV columns: stamp_s, arousal, valence, confidence

Outputs:
- CSV columns: stamp_s, mode, velocity_scale, accel_scale, approach_offset_m, stiffness_scale, grip_force_cap

Usage:
  python tools/replay_tone_csv.py --in tone.csv --out style.csv
  python tools/replay_tone_csv.py --in tone.csv --out style.csv --hz 10 --config ros2_ws/src/prosody2policy_bringup/config/adapter.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from typing import List, Optional

# Reuse simulator adapter state and config loader (no ROS2)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.sim_adapter import ToneSignal, AdapterState, load_adapter_config  # type: ignore


OUT_FIELDS = [
    "stamp_s",
    "mode",
    "velocity_scale",
    "accel_scale",
    "approach_offset_m",
    "stiffness_scale",
    "grip_force_cap",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input tone CSV")
    ap.add_argument("--out", dest="out", required=True, help="Output style CSV")
    ap.add_argument("--hz", type=float, default=10.0, help="Replay rate used to compute timesteps")
    ap.add_argument("--config", default=None, help="Optional adapter yaml/json config (same as sim)")
    args = ap.parse_args()

    cfg = load_adapter_config(args.config)
    st = AdapterState(cfg)

    # Read input
    rows = []
    with open(args.inp, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        print("No rows found.", file=sys.stderr)
        return 2

    # Replay
    dt = 1.0 / args.hz
    now_s = time.time()

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()

        for i, row in enumerate(rows):
            a = float(row["arousal"])
            v = float(row["valence"])
            c = float(row["confidence"])
            msg = ToneSignal(stamp_s=now_s, arousal=a, valence=v, confidence=c)

            sp = st.step(msg, now_s)
            w.writerow({
                "stamp_s": f"{sp.stamp_s:.6f}",
                "mode": sp.mode,
                "velocity_scale": f"{sp.velocity_scale:.6f}",
                "accel_scale": f"{sp.accel_scale:.6f}",
                "approach_offset_m": f"{sp.approach_offset_m:.6f}",
                "stiffness_scale": f"{sp.stiffness_scale:.6f}",
                "grip_force_cap": f"{sp.grip_force_cap:.6f}",
            })
            now_s += dt

    print(f"Wrote style replay -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
