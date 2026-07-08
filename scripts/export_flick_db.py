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
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR.parent / "data" / "processed" / "human_movements.jsonl"
OUT = SCRIPT_DIR.parent / "flick_trajectories.json"

N_PTS          = 48
MIN_EFFICIENCY = 0.90
MAX_LATERAL    = 0.10
MIN_DIST       = 30.0
MIN_DUR, MAX_DUR = 120.0, 1600.0
MAX_KEEP       = 6000


def canonical48(pts):
    p = np.asarray(pts, dtype=float)
    if len(p) < 5:
        return None
    N = len(p)
    xi = np.linspace(0, N - 1, N_PTS)              # resample by point index (keeps irregular timing)
    ar = np.arange(N)
    x = np.interp(xi, ar, p[:, 0]);  y = np.interp(xi, ar, p[:, 1]);  t = np.interp(xi, ar, p[:, 2])
    x, y, t = x - x[0], y - y[0], t - t[0]
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
            rec = canonical48(json.loads(line)["points"])
            if rec is not None:
                kept.append(rec)
            if len(kept) >= MAX_KEEP:
                break
    kept.sort(key=lambda r: r["d"])
    OUT.write_text(json.dumps({"unit": "px", "n_pts": N_PTS, "traj": kept}))
    ds = [r["d"] for r in kept]
    print(f"[export] read {n_read}, kept {len(kept)} straight 48-pt strokes "
          f"(eff>={MIN_EFFICIENCY}, lateral<={MAX_LATERAL})")
    if ds:
        print(f"[export] distance range {min(ds):.0f}-{max(ds):.0f}px, median {sorted(ds)[len(ds)//2]:.0f}px")
        print(f"[export] wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
