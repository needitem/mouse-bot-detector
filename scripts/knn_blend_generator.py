#!/usr/bin/env python3
"""Can the 16k real-trajectory pool be COMBINED into genuinely NEW trajectories
that (a) stay on the real manifold so the detector can't tell (like replay,
~0.50) yet (b) are not near-duplicates of any single source (unlike replay, so
the DB/reuse defenses fail too)?

Approach: for each sample, pick a random anchor trajectory, find its k nearest
neighbors IN CANONICAL SHAPE SPACE (so we only ever blend trajectories that are
already similar - minimizing the smoothing/averaging that kills jerk when you
mix dissimilar shapes), and take a Dirichlet-weighted convex blend of the
shape + (distance,duration). `--alpha` sets the blend: low = anchor-dominated
(close to a real trajectory, ~0.50 but near-duplicate), high = evenly mixed
(more novel, but averaging flattens jerk toward detectable).

Reports BOTH numbers that matter: this writes knn_blend_bot_movements.jsonl for
the strong detector (does it stay ~0.50?), and prints each blend's nearest-
neighbor distance to the real pool vs a real held-out sample's (is it actually
novel, or a disguised near-duplicate?). The tension between the two is the
whole point.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_generator import build_vectors, N_SHAPE_POINTS, MIN_MOVEMENT_TIME
from trajectory_gmm_ceiling import load_human_pool_raw_points

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUT_PATH = DATA_DIR / "knn_blend_bot_movements.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--k", type=int, default=6, help="neighbors to blend among")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Dirichlet concentration; low=anchor-dominated, high=even mix")
    ap.add_argument("--jitter", type=float, default=1.0)
    args = ap.parse_args()

    pool = load_human_pool_raw_points(seed=0)
    X = build_vectors(pool)
    shapes = X[:, : N_SHAPE_POINTS * 2]
    scal = X[:, N_SHAPE_POINTS * 2:]           # distance, duration
    print(f"[blend] {len(X)} real trajectories; k={args.k} alpha={args.alpha}")

    scaler = StandardScaler().fit(shapes)
    Sn = scaler.transform(shapes)
    nn = NearestNeighbors(n_neighbors=args.k).fit(Sn)

    rng = np.random.default_rng(1)
    anchors = rng.integers(0, len(X), size=args.n)
    _, neigh_idx = nn.kneighbors(Sn[anchors])   # (n, k)

    out, nn_dists = [], []
    # a reusable NN over the pool to measure novelty (distance to nearest real)
    novelty_nn = NearestNeighbors(n_neighbors=1).fit(Sn)
    for row in neigh_idx:
        w = rng.dirichlet(np.full(args.k, args.alpha))
        blended_shape = w @ shapes[row]
        blended_scal = w @ scal[row]
        # novelty: distance from blended shape to its nearest REAL shape
        d, _ = novelty_nn.kneighbors(scaler.transform(blended_shape[None]))
        nn_dists.append(float(d[0, 0]))

        shape = blended_shape.reshape(N_SHAPE_POINTS, 2)
        distance = max(float(blended_scal[0]), 5.0)
        movement_time = max(float(blended_scal[1]), MIN_MOVEMENT_TIME)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        c, s = math.cos(theta), math.sin(theta)
        xs, ys = shape[:, 0] * distance, shape[:, 1] * distance
        rx = xs * c - ys * s + rng.normal(0.0, args.jitter, size=xs.shape)
        ry = xs * s + ys * c + rng.normal(0.0, args.jitter, size=ys.shape)
        t = np.linspace(0.0, movement_time, N_SHAPE_POINTS)
        out.append(list(zip(rx.tolist(), ry.tolist(), t.tolist())))

    # novelty baseline: nearest-real distance for genuine held-out real shapes
    holdout = Sn[rng.integers(0, len(X), size=1000)]
    d_real, _ = NearestNeighbors(n_neighbors=2).fit(Sn).kneighbors(holdout)
    real_nn = d_real[:, 1].mean()   # 2nd neighbor = nearest OTHER real
    print(f"[blend] novelty (nearest-real shape dist): blend mean={np.mean(nn_dists):.3f} "
          f"| real-to-real mean={real_nn:.3f}  (blend << real-to-real => near-duplicate)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for pts in out:
            f.write(json.dumps({"points": pts}) + "\n")
    print(f"[blend] wrote {len(out)} blended movements to {OUT_PATH}")


if __name__ == "__main__":
    main()
