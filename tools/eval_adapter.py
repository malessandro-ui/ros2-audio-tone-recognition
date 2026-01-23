#!/usr/bin/env python3
"""
tools/eval_adapter.py

Compute quantitative metrics from:
- Tone CSV (inputs)
- Style CSV (adapter outputs), e.g. produced by tools/replay_tone_csv.py

Metrics:
- mode_flip_rate (flips per second)
- time_in_mode fraction
- neutral_due_to_low_conf (if available; otherwise not computed)
- basic responsiveness: time to first switch away from neutral

Optional plotting:
- If matplotlib is installed, save:
    --plot out.png

Usage:
  python tools/replay_tone_csv.py --in tone.csv --out style.csv
  python tools/eval_adapter.py --tone tone.csv --style style.csv
  python tools/eval_adapter.py --tone tone.csv --style style.csv --plot report.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def try_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def compute_flip_rate(stamps: List[float], modes: List[str]) -> float:
    if len(stamps) < 2:
        return 0.0
    flips = 0
    for i in range(1, len(modes)):
        if modes[i] != modes[i - 1]:
            flips += 1
    duration = max(1e-6, stamps[-1] - stamps[0])
    return flips / duration


def fraction_time_in_modes(modes: List[str]) -> Dict[str, float]:
    if not modes:
        return {}
    c = Counter(modes)
    n = len(modes)
    return {k: v / n for k, v in sorted(c.items())}


def time_to_leave_neutral(stamps: List[float], modes: List[str]) -> Optional[float]:
    if not modes:
        return None
    if modes[0] != "neutral":
        return 0.0
    for i in range(1, len(modes)):
        if modes[i] != "neutral":
            return stamps[i] - stamps[0]
    return None


def maybe_plot(stamps: List[float], arousal: List[float], valence: List[float], modes: List[str], out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not installed; skipping plot. Install with: pip install matplotlib")
        return

    # Map modes to integers for a simple timeline chart
    uniq = sorted(set(modes))
    mode_to_i = {m: i for i, m in enumerate(uniq)}
    mode_i = [mode_to_i[m] for m in modes]

    t0 = stamps[0] if stamps else 0.0
    t = [s - t0 for s in stamps]

    plt.figure()
    plt.plot(t, arousal, label="arousal")
    plt.plot(t, valence, label="valence")
    plt.plot(t, mode_i, label="mode_index")
    plt.yticks(list(range(len(uniq))), uniq)
    plt.xlabel("time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"Wrote plot -> {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tone", required=True, help="Tone CSV (stamp_s, arousal, valence, confidence)")
    ap.add_argument("--style", required=True, help="Style CSV from replay (stamp_s, mode, ...)")
    ap.add_argument("--plot", default=None, help="Optional output PNG path")
    args = ap.parse_args()

    tone_rows = read_csv(args.tone)
    style_rows = read_csv(args.style)

    if not tone_rows or not style_rows:
        print("Missing rows in input.", file=sys.stderr)
        return 2

    # Align by index (replay preserves ordering / count)
    n = min(len(tone_rows), len(style_rows))

    stamps = []
    arousal = []
    valence = []
    conf = []
    modes = []

    for i in range(n):
        ts = try_float(style_rows[i].get("stamp_s"))
        if ts is None:
            continue
        stamps.append(ts)
        arousal.append(float(tone_rows[i]["arousal"]))
        valence.append(float(tone_rows[i]["valence"]))
        conf.append(float(tone_rows[i]["confidence"]))
        modes.append(style_rows[i]["mode"])

    flip_rate = compute_flip_rate(stamps, modes)
    frac = fraction_time_in_modes(modes)
    t_leave = time_to_leave_neutral(stamps, modes)

    print("=== Adapter Evaluation ===")
    print(f"samples: {len(modes)}")
    print(f"mode_flip_rate: {flip_rate:.4f} flips/sec")
    print("time_in_mode_fraction:")
    for k, v in frac.items():
        print(f"  - {k}: {v:.3f}")
    if t_leave is None:
        print("time_to_leave_neutral: (never left neutral)")
    else:
        print(f"time_to_leave_neutral: {t_leave:.3f} s")

    if args.plot:
        maybe_plot(stamps, arousal, valence, modes, args.plot)

    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
