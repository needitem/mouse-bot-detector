#!/usr/bin/env python3
"""Defender side: a SET-LEVEL detector for warped replay.

Warped replay is undetectable from a single point-to-point trajectory (it IS a
real human stroke, rotated/scaled - the strong single-movement detector sits at
~0.50 on it, see validate_flow_bot_strong_detector.py on warped_replay). But it
draws from a FINITE pool of recorded strokes, so across many flicks the SAME
source stroke reappears. Canonicalize each flick (remove the per-flick rotation
and scale) and those repeats collapse onto near-identical shapes - a
near-duplicate a real player, whose every stroke is unique, never produces.

This detector observes a SESSION of N flicks, canonicalizes them, and flags the
session as replay if it contains near-duplicate pairs (min pairwise shape
distance below a threshold calibrated from the human-vs-human gap). It then
sweeps pool size K and session length N to map exactly how many flicks a
defender must observe to catch a warped-replay attacker drawing from a pool of
size K - the birthday-bound the attacker is fighting.

Run: python scripts/detect_replay_reuse.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_noise_search import to_canonical_at, N_SHAPE_POINTS
from trajectory_gmm_ceiling import load_human_pool_raw_points

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

JITTER_PX = 1.0          # matches warped_replay_generator / warped_replay.hpp default


def canonical_shapes(points_list):
    """Rotation/scale-invariant unit shape vectors (endpoint on +x, distance 1)."""
    shapes, dists = [], []
    for pts in points_list:
        c = to_canonical_at(pts, N_SHAPE_POINTS)
        if c is None:
            continue
        shape_xy, distance, _ = c
        shapes.append(shape_xy.ravel())
        dists.append(distance)
    return np.asarray(shapes), np.asarray(dists)


def min_pair_dist(vecs):
    """Smallest pairwise Euclidean distance in a small set (brute force)."""
    n = len(vecs)
    if n < 2:
        return np.inf
    best = np.inf
    for i in range(n):
        d = np.sqrt(((vecs[i + 1:] - vecs[i]) ** 2).sum(axis=1))
        if d.size:
            best = min(best, d.min())
    return best


def main():
    rng = np.random.default_rng(0)
    print("[reuse] loading human pool + canonicalizing...")
    pool = load_human_pool_raw_points(seed=0)
    shapes, dists = canonical_shapes(pool)
    print(f"[reuse] {len(shapes)} canonical human shapes, dim={shapes.shape[1]}")

    # --- calibrate the near-duplicate threshold ---
    # same-source pair: one shape vs itself + canonical-space jitter (jitter_px
    # scaled by the reach distance). different-source pair: two real strokes.
    idx = rng.choice(len(shapes), size=2000, replace=False)
    S = shapes[idx]
    D = dists[idx]
    # nearest OTHER real shape (distinct sources) -> the "human floor" spacing
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(shapes)
    dd, _ = nn.kneighbors(S)
    diff_source = dd[:, 1]                        # nearest distinct real stroke
    # same source under jitter: canonical jitter std = jitter_px / reach
    jit = np.array([np.linalg.norm(rng.normal(0, JITTER_PX / max(d, 1e-6),
                    size=(N_SHAPE_POINTS, 2))) for d in D])
    print(f"[reuse] same-source(jitter) dist: median={np.median(jit):.4f} "
          f"p95={np.percentile(jit,95):.4f}")
    print(f"[reuse] diff-source(real) dist:   median={np.median(diff_source):.4f} "
          f"p5={np.percentile(diff_source,5):.4f}")
    # threshold between the two populations
    eps = float(np.percentile(jit, 99) * 3)
    eps = min(eps, float(np.percentile(diff_source, 1)) * 0.5)
    print(f"[reuse] near-duplicate threshold eps={eps:.4f}")

    def bot_session(K, N):
        src = rng.integers(0, K, size=N)          # draw N flicks from pool[:K]
        base = shapes[src]
        jits = np.array([rng.normal(0, JITTER_PX / max(dists[s], 1e-6),
                                    size=base.shape[1]) for s in src])
        return base + jits

    def human_session(N):
        pick = rng.choice(len(shapes), size=N, replace=False)  # all distinct
        return shapes[pick]

    TRIALS = 60
    Ks = [1000, 3000, 6000, 16000]
    Ns = [10, 25, 50, 100, 250, 500, 1000]
    print("\n[reuse] detection rate = P(session flagged as replay), by pool K x session N")
    print("        (human sessions flag at ~0.00 = false-positive rate)\n")
    header = "  K \\ N |" + "".join(f"{n:>7}" for n in Ns)
    print(header)
    print("  " + "-" * (len(header) - 2))

    grid = {}
    for K in Ks:
        if K > len(shapes):
            continue
        row = []
        for N in Ns:
            if N > K or N > len(shapes):
                row.append(None); continue
            hits = 0
            for _ in range(TRIALS):
                hits += (min_pair_dist(bot_session(K, N)) < eps)
            rate = hits / TRIALS
            grid[(K, N)] = rate
            row.append(rate)
        print(f"{K:>7} |" + "".join(f"{(f'{r:.2f}' if r is not None else '  -'):>7}" for r in row))

    # human false-positive check (largest N)
    fp = sum(min_pair_dist(human_session(min(1000, len(shapes)))) < eps for _ in range(TRIALS)) / TRIALS
    print(f"\n[reuse] human-session false-positive rate (N=1000): {fp:.3f}")

    # birthday-bound reference: P(>=1 collision) ~ 1 - exp(-N(N-1)/2K)
    print("\n[reuse] theory (birthday) P(collision) for reference:")
    for K in Ks:
        for N in [50, 100, 500]:
            p = 1 - math.exp(-N * (N - 1) / (2 * K))
            print(f"        K={K:>6} N={N:>4}: {p:.2f}", end="   ")
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "replay_reuse_detection.json").write_text(json.dumps(
        {"eps": eps, "false_positive_N1000": fp,
         "grid": {f"K{k}_N{n}": v for (k, n), v in grid.items()}}, indent=2))
    print(f"\n[reuse] wrote {RESULTS_DIR / 'replay_reuse_detection.json'}")


if __name__ == "__main__":
    main()
