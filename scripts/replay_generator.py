#!/usr/bin/env python3
"""The other end of the spectrum from a from-scratch generator: instead of
LEARNING the human distribution (motor_synergy / GMM / flow all plateau at
~0.82-0.86 vs a strong detector), just REPLAY real human trajectories with a
small per-sample jitter. Since human-vs-human is 0.500 (see
results/human_floor.md), a replay that stays close to the real manifold
should also land near 0.50 - the empirical proof that 50% is reachable, and
the honest ceiling of "how good could any generator get."

The `--jitter` knob traces the whole spectrum: jitter=0 is pure replay
(~0.50, but memorization), larger jitter walks away from the real manifold
toward the detectable region. Writes replay_bot_movements.jsonl for
validate_flow_bot_strong_detector.py to score with the identical protocol.

Threat-model note: replay of recorded human movement is a REAL bot-detection
bypass, so this isn't a cheat - it's the correct answer to "what actually
defeats the detector," and it frames exactly how much a true generator
(which must generalize, not memorize) is leaving on the table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_gmm_ceiling import load_human_pool_raw_points

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUT_PATH = DATA_DIR / "replay_bot_movements.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jitter", type=float, default=1.0,
                    help="stdev of per-point positional gaussian jitter, px")
    ap.add_argument("--t_jitter", type=float, default=0.0,
                    help="stdev of per-point timestamp jitter, ms")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--pool_start", type=int, default=8000,
                    help="draw replayed trajectories from pool[start:] so they are "
                         "DISJOINT from the pool[:8000] the flow trained on")
    args = ap.parse_args()

    pool = load_human_pool_raw_points(seed=0)
    src = pool[args.pool_start:]
    print(f"[replay] {len(src)} source human trajectories (pool[{args.pool_start}:]), "
          f"jitter={args.jitter}px t_jitter={args.t_jitter}ms")

    rng = np.random.default_rng(1)
    pick = rng.integers(0, len(src), size=args.n)
    out = []
    for i in pick:
        pts = np.asarray(src[i], dtype=float)
        xy = pts[:, :2] + rng.normal(0.0, args.jitter, size=pts[:, :2].shape)
        t = pts[:, 2].copy()
        if args.t_jitter > 0:
            t = t + rng.normal(0.0, args.t_jitter, size=t.shape)
            t = np.maximum.accumulate(t)  # keep timestamps monotone
        points = list(zip(xy[:, 0].tolist(), xy[:, 1].tolist(), t.tolist()))
        out.append(points)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for pts in out:
            f.write(json.dumps({"points": pts}) + "\n")
    print(f"[replay] wrote {len(out)} replayed movements to {OUT_PATH}")


if __name__ == "__main__":
    main()
