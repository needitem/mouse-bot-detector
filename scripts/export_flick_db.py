#!/usr/bin/env python3
"""Export a warped-replay trajectory database for needaimbot's flick generator.

Keeps only STRAIGHT, low-lateral-deviation human strokes (an aimbot flick
should go roughly straight to the target), resamples each to a FIXED 48 points
(so the C++ generator can add a human-variability perturbation = the difference
between two other real strokes, which requires a common dimension), and stores
a UNIT canonical shape (origin -> (1,0)) plus its real distance and its real
(point-index-resampled, so still irregular) timestamps.

At runtime warped_replay.hpp picks a distance-matched stroke, adds
`mag * (shape_a - shape_b)` for two random strokes (moves it along the manifold
humans genuinely vary along, so it evades both the single-movement and the
near-duplicate/reuse detector - see mouse-bot-detector's attack_sweet_spot.py:
mag~0.07 gives single-move ~0.54 and reuse-detection ~0.00), then rotates and
scales it onto the aim vector.

Filters: path_efficiency >= MIN_EFFICIENCY, max lateral / distance <= MAX_LATERAL.
Output: flick_trajectories.json  { "unit":"px", "n_pts":48, "traj":[ {"d":dist,
        "s":[[ux,uy],...48], "t":[t_ms,...48]}, ... ] } sorted by distance.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR.parent / "data" / "processed" / "human_movements.jsonl"
OUT = SCRIPT_DIR.parent / os.environ.get("OUTNAME","flick_trajectories.json")

MAX_PTS        = 64      # cap for the C++ fixed array; native strokes are ~11-40 pts
MIN_EFFICIENCY = float(os.environ.get("EFF","0.75"))
MAX_LATERAL    = float(os.environ.get("LAT","0.15"))
MIN_DIST       = 30.0
MIN_DUR, MAX_DUR = 120.0, 1600.0
MAX_KEEP       = 6000


def canonical_native(pts):
    """Keep the stroke's NATIVE points (no resampling - resampling to a fixed
    grid changes the fine kinematics and is itself a ~0.2 detector tell). Only
    rotate the endpoint onto +x and scale to unit distance so it can be warped
    onto any target; the native point count and irregular timing are preserved."""
    p = np.asarray(pts, dtype=float)
    if len(p) < 5 or len(p) > MAX_PTS:
        return None
    x, y, t = p[:, 0] - p[0, 0], p[:, 1] - p[0, 1], p[:, 2] - p[0, 2]
    dx, dy = x[-1], y[-1]
    dist = math.hypot(dx, dy)
    dur = float(t[-1])
    if dist < MIN_DIST or not (MIN_DUR <= dur <= MAX_DUR):
        return None
    seg = np.hypot(np.diff(x), np.diff(y))
    path_len = float(seg.sum())
    if path_len <= 1e-6 or dist / path_len < MIN_EFFICIENCY:
        return None
    phi = math.atan2(dy, dx)
    c, s = math.cos(-phi), math.sin(-phi)
    ux = (x * c - y * s) / dist                     # unit canonical: endpoint at (1, 0)
    uy = (x * s + y * c) / dist
    if np.max(np.abs(uy)) > MAX_LATERAL:
        return None
    t = np.maximum.accumulate(t)
    return {"d": round(dist, 2),
            "s": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(ux, uy)],
            "t": [round(float(tt), 1) for tt in t]}


def main():
    kept, n_read = [], 0
    with open(DATA) as f:
        for line in f:
            n_read += 1
            rec = canonical_native(json.loads(line)["points"])
            if rec is not None:
                kept.append(rec)
            if len(kept) >= MAX_KEEP:
                break
    kept.sort(key=lambda r: r["d"])
    OUT.write_text(json.dumps({"unit": "px", "n_pts": 0, "traj": kept}))  # n_pts=0 -> variable
    ds = [r["d"] for r in kept]
    npts = [len(r["s"]) for r in kept]
    print(f"[export] read {n_read}, kept {len(kept)} straight native strokes "
          f"(eff>={MIN_EFFICIENCY}, lateral<={MAX_LATERAL}); "
          f"points/stroke median {sorted(npts)[len(npts)//2]}, max {max(npts)}")
    if ds:
        print(f"[export] distance range {min(ds):.0f}-{max(ds):.0f}px, median {sorted(ds)[len(ds)//2]:.0f}px")
        print(f"[export] wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
