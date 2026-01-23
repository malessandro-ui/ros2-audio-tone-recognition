#!/usr/bin/env python3
"""
tools/log_tone_csv.py

Utility to log ToneSignal-like records to CSV.

This is intentionally ROS-agnostic. You can:
- pipe JSON lines from a ROS2 topic echo/logger into this tool
- or use it in a future ROS node wrapper
- or log from simulator outputs

Input mode (stdin):
- expects JSON lines with keys: stamp_s, arousal, valence, confidence
- writes CSV with a header

Example (later on Ubuntu with ROS2):
  ros2 topic echo /tone/signal --once   # (or a JSON logger)
For now you can simulate:
  python tools/sim_adapter.py --scenario scripted --steps 10 > /tmp/out.txt

Usage:
  python tools/log_tone_csv.py --out tone.csv
  cat tone.jsonl | python tools/log_tone_csv.py --out tone.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from typing import Dict, Iterable, List


FIELDS = ["stamp_s", "arousal", "valence", "confidence"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV file")
    ap.add_argument("--append", action="store_true", help="Append to existing CSV (no header rewrite)")
    args = ap.parse_args()

    mode = "a" if args.append else "w"
    write_header = not args.append

    with open(args.out, mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()

        n = 0
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            row = {k: obj.get(k) for k in FIELDS}
            # best effort
            if row["stamp_s"] is None:
                row["stamp_s"] = f"{time.time():.6f}"
            w.writerow(row)
            n += 1

    print(f"Wrote {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
