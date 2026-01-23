#!/usr/bin/env python3
"""
tools/calibrate_confidence.py

Calibrate "confidence" values to be better probabilistic estimates.

Why:
- gating / safety decisions depend on confidence thresholds
- raw model scores are often miscalibrated
- calibration improves reliability and interpretability

Input:
- a CSV with at least:
    confidence, label
  where:
    confidence in [0,1]
    label in {0,1} meaning "prediction was correct / trustworthy"

Method:
- Temperature scaling in logit space:
    p' = sigmoid(logit(p) / T)
  Fit T to minimize negative log-likelihood.

Output:
- JSON: {"temperature": T}

Usage:
  python tools/calibrate_confidence.py --in data/conf.csv --out calibration.json
  python tools/calibrate_confidence.py --apply calibration.json --in data/conf.csv --out calibrated.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from typing import List, Tuple


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def logit(p: float, eps: float = 1e-6) -> float:
    p = clamp(p, eps, 1.0 - eps)
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def temp_scale(p: float, T: float) -> float:
    return sigmoid(logit(p) / max(T, 1e-6))


def nll(conf: List[float], y: List[int], T: float) -> float:
    eps = 1e-6
    s = 0.0
    for p, yi in zip(conf, y):
        q = clamp(temp_scale(p, T), eps, 1.0 - eps)
        s += -(yi * math.log(q) + (1 - yi) * math.log(1 - q))
    return s / max(1, len(conf))


def fit_temperature(conf: List[float], y: List[int]) -> float:
    """
    Simple 1D search (log-space) for temperature minimizing NLL.
    Robust and dependency-free.
    """
    # Search T in [0.25, 10] in log space with progressive refinement
    lo, hi = math.log(0.25), math.log(10.0)

    best_T = 1.0
    best = float("inf")

    for _ in range(5):  # 5 refinement rounds
        # sample grid
        for i in range(41):
            t = lo + (hi - lo) * (i / 40.0)
            T = math.exp(t)
            v = nll(conf, y, T)
            if v < best:
                best, best_T = v, T
        # narrow around best_T
        center = math.log(best_T)
        span = (hi - lo) * 0.25
        lo, hi = center - span, center + span

    return float(best_T)


def read_csv(path: str) -> Tuple[List[float], List[int], List[dict]]:
    conf, y, rows = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "confidence" not in r.fieldnames or "label" not in r.fieldnames:
            raise ValueError("CSV must contain columns: confidence, label")
        for row in r:
            c = float(row["confidence"])
            lab = int(row["label"])
            conf.append(clamp(c, 0.0, 1.0))
            y.append(1 if lab != 0 else 0)
            rows.append(row)
    return conf, y, rows


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV with confidence,label")
    ap.add_argument("--out", dest="out", required=True, help="Output file (.json for fit, .csv for apply)")
    ap.add_argument("--apply", dest="apply", default=None, help="Calibration JSON to apply to input CSV")
    args = ap.parse_args()

    conf, y, rows = read_csv(args.inp)

    if args.apply:
        cal = json.loads(open(args.apply, "r", encoding="utf-8").read())
        T = float(cal.get("temperature", 1.0))
        for row in rows:
            p = float(row["confidence"])
            row["confidence_calibrated"] = f"{temp_scale(p, T):.6f}"
        fieldnames = list(rows[0].keys())
        write_csv(args.out, rows, fieldnames)
        print(f"Wrote calibrated CSV -> {args.out}")
        return 0

    T = fit_temperature(conf, y)
    out = {"temperature": T}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Fit temperature={T:.4f} (saved to {args.out})")
    print("Tip: apply with --apply calibration.json --in data.csv --out calibrated.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
