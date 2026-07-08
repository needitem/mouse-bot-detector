#!/usr/bin/env python3
"""Applied replay: the practical answer to "can I replay real human motion but
aimed at an ARBITRARY target?" - which plain replay can't (it only reaches the
distance/direction that was recorded).

For each synthetic movement we pick a target (distance D, direction theta),
select a REAL human trajectory whose recorded distance is close to D
(distance-matched, so the scale warp is ~1 and doesn't distort speed/tremor),
translate it to the origin, rotate to theta, scale to exactly D, and keep its
REAL (irregular) timestamps. The result is real human movement dynamics warped
onto any start->target pair - exactly what an aimbot needs, and (because it IS
real motion, distance-matched) it should stay near the human-vs-human 0.50
floor, unlike motor_synergy's formula (0.99) or the learned generators (~0.85).

--dist_tol controls the distance-match window: tight = minimal scale warp
(stays near 0.50), loose = pick any trajectory and stretch it hard (walks
toward detectable, quantifying the speed-distortion cost of warping). Writes
warped_replay_bot_movements.jsonl for validate_flow_bot_strong_detector.py.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_gmm_ceiling import load_human_pool_raw_points

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUT_PATH = DATA_DIR / "warped_replay_bot_movements.jsonl"


def traj_distance(pts):
    p = np.asarray(pts, dtype=float)
    return math.hypot(p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--pool_start", type=int, default=8000,
                    help="source trajectories from pool[start:] (disjoint from flow's train)")
    ap.add_argument("--dist_tol", type=float, default=0.10,
                    help="relative distance-match window; source dist within +-tol of target D")
    ap.add_argument("--jitter", type=float, default=1.0, help="per-point positional jitter, px")
    args = ap.parse_args()

    pool = load_human_pool_raw_points(seed=0)
    src = [p for p in pool[args.pool_start:] if traj_distance(p) >= 5.0]
    dists = np.array([traj_distance(p) for p in src])
    order = np.argsort(dists)
    dists_sorted = dists[order]
    print(f"[warp] {len(src)} source human trajectories, "
          f"distance range {dists.min():.0f}-{dists.max():.0f}px, tol=+-{args.dist_tol:.0%}")

    rng = np.random.default_rng(1)
    # targets: draw target distances from the REAL distance distribution itself
    target_D = dists[rng.integers(0, len(src), size=args.n)]

    out = []
    n_exact = 0
    for D in target_D:
        lo = np.searchsorted(dists_sorted, D * (1 - args.dist_tol), side="left")
        hi = np.searchsorted(dists_sorted, D * (1 + args.dist_tol), side="right")
        if hi > lo:
            j = order[rng.integers(lo, hi)]
            n_exact += 1
        else:                       # no in-window match: take nearest by distance
            j = order[min(np.searchsorted(dists_sorted, D), len(src) - 1)]
        pts = np.asarray(src[j], dtype=float)

        rel = pts[:, :2] - pts[0, :2]
        orig_dist = math.hypot(rel[-1, 0], rel[-1, 1])
        if orig_dist < 1e-6:
            continue
        scale = D / orig_dist
        phi = math.atan2(rel[-1, 1], rel[-1, 0])          # recorded direction
        theta = rng.uniform(0.0, 2.0 * math.pi)           # desired direction
        a = theta - phi
        c, s = math.cos(a), math.sin(a)
        rx = (rel[:, 0] * c - rel[:, 1] * s) * scale
        ry = (rel[:, 0] * s + rel[:, 1] * c) * scale
        rx = rx + rng.normal(0.0, args.jitter, size=rx.shape)
        ry = ry + rng.normal(0.0, args.jitter, size=ry.shape)
        t = (pts[:, 2] - pts[0, 2]).tolist()              # keep REAL irregular timing
        out.append(list(zip(rx.tolist(), ry.tolist(), t)))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for pts in out:
            f.write(json.dumps({"points": pts}) + "\n")
    print(f"[warp] wrote {len(out)} warped movements ({n_exact}/{args.n} had an in-window "
          f"distance match, scale~1) to {OUT_PATH}")


if __name__ == "__main__":
    main()
