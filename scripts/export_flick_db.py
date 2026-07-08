#!/usr/bin/env python3
"""Export a warped-replay trajectory database for needaimbot's flick generator.

Keeps only STRAIGHT, low-lateral-deviation human strokes (an aimbot flick
should go roughly straight to the target, not wander up/down), canonicalizes
each to start at the origin with its endpoint on the +x axis, and stores it
with its real distance + real (irregular) timestamps. At runtime the C++
warped-replay generator picks a stroke whose distance is close to the required
reach, then rotates it to the aim direction and scales it to the exact reach.

Filters:
  - path_efficiency = straight_distance / path_length >= MIN_EFFICIENCY
  - max lateral deviation / distance <= MAX_LATERAL   (no up/down wandering)
  - sane distance and duration
Output: flick_trajectories.json  { "unit":"px", "traj":[ {"d":dist,
        "p":[[cx,cy,t_ms],...]}, ... ] }  sorted by distance for binary search.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR.parent / "data" / "processed" / "human_movements.jsonl"
OUT = SCRIPT_DIR.parent / "flick_trajectories.json"

MIN_EFFICIENCY = 0.90   # straight-distance / path-length
MAX_LATERAL    = 0.10   # max |lateral| as fraction of reach distance
MIN_DIST       = 30.0
MIN_DUR, MAX_DUR = 120.0, 1600.0
MAX_KEEP       = 6000


def canonical(pts):
    p = np.asarray(pts, dtype=float)
    if len(p) < 5:
        return None
    rel = p[:, :2] - p[0, :2]
    t = p[:, 2] - p[0, 2]
    dx, dy = rel[-1, 0], rel[-1, 1]
    dist = math.hypot(dx, dy)
    dur = float(t[-1])
    if dist < MIN_DIST or not (MIN_DUR <= dur <= MAX_DUR):
        return None
    seg = np.hypot(np.diff(rel[:, 0]), np.diff(rel[:, 1]))
    path_len = float(seg.sum())
    if path_len <= 1e-6:
        return None
    eff = dist / path_len
    if eff < MIN_EFFICIENCY:
        return None
    phi = math.atan2(dy, dx)
    c, s = math.cos(-phi), math.sin(-phi)
    cx = rel[:, 0] * c - rel[:, 1] * s
    cy = rel[:, 0] * s + rel[:, 1] * c
    if np.max(np.abs(cy)) / dist > MAX_LATERAL:
        return None
    # enforce monotone non-decreasing timestamps
    t = np.maximum.accumulate(t)
    p_out = [[round(float(a), 2), round(float(b), 2), round(float(tt), 1)]
             for a, b, tt in zip(cx, cy, t)]
    return {"d": round(dist, 2), "p": p_out}


def main():
    kept = []
    n_read = 0
    with open(DATA) as f:
        for line in f:
            n_read += 1
            rec = canonical(json.loads(line)["points"])
            if rec is not None:
                kept.append(rec)
            if len(kept) >= MAX_KEEP:
                break
    kept.sort(key=lambda r: r["d"])
    OUT.write_text(json.dumps({"unit": "px", "traj": kept}))
    ds = [r["d"] for r in kept]
    print(f"[export] read {n_read}, kept {len(kept)} straight strokes "
          f"(eff>={MIN_EFFICIENCY}, lateral<={MAX_LATERAL})")
    if ds:
        print(f"[export] distance range {min(ds):.0f}-{max(ds):.0f}px, "
              f"median {sorted(ds)[len(ds)//2]:.0f}px")
        print(f"[export] wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
