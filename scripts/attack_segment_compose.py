#!/usr/bin/env python3
"""Attacker: beat the finite-pool problem by COMPOSING strokes from segments.

A finite pool of K strokes gives only K distinct replays, so a defender who
observes > K flicks catches reuse (detect_longhorizon_crossaccount.py). But if
each flick is stitched from segments of SEVERAL real strokes (first part of A,
second part of B, ...), the number of distinct flicks explodes combinatorially
(~K^segments) while every segment is still real motion. This is the "composed"
in SCRAP - and unlike kNN blending (which AVERAGES and kills jerk), stitching
keeps each segment intact; only the SEAM between them is synthetic.

Question: does composition beat BOTH detectors at once?
  - near-duplicate / long-horizon: composed flicks are ~all distinct even from a
    tiny pool -> reuse should vanish.
  - single-move: is the velocity/direction discontinuity at the seam a tell?
We stitch in the canonical unit-shape space with C1 blending over a short seam
window to soften the discontinuity, then measure the strong single-move
detector and the near-duplicate rate, sweeping pool size K and seam softness.
Run: python -u scripts/attack_segment_compose.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_noise_search import to_canonical_at, N_SHAPE_POINTS
from trajectory_gmm_ceiling import load_human_pool_raw_points
from train_detector import SHAPE_ONLY_FEATURES, tune_best_model, TEST_SIZE, RANDOM_STATE
from features import extract_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EPS = 0.0594
N_EVAL = 2000
NP = N_SHAPE_POINTS


def shape_to_raw(shape, dist, rng):
    theta = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(theta), math.sin(theta)
    xs, ys = shape[:, 0] * dist, shape[:, 1] * dist
    rx, ry = xs * c - ys * s, xs * s + ys * c
    t = np.linspace(0.0, dist * 1.5 + 200.0, len(shape))
    return list(zip(rx.tolist(), ry.tolist(), t.tolist()))


def compose(shapes, dists, K, n_seg, seam, rng):
    """Stitch a flick from n_seg segments of distinct pool strokes, blending
    over a +-seam window at each cut for C1 continuity. Returns unit shape."""
    srcs = rng.integers(0, K, size=n_seg)
    cuts = np.sort(rng.choice(np.arange(4, NP - 4), size=n_seg - 1, replace=False)) if n_seg > 1 else []
    out = np.zeros((NP, 2))
    bounds = [0, *cuts, NP]
    offset = np.zeros(2)
    prev_end = None
    for si, a, b in zip(srcs, bounds[:-1], bounds[1:]):
        seg = shapes[si][a:b].copy()
        # translate so this segment starts where the previous ended (position continuity)
        if prev_end is not None:
            offset = prev_end - seg[0]
        seg = seg + offset
        out[a:b] = seg
        prev_end = seg[-1]
    # soften each seam with a short local moving-average (C1 continuity)
    if seam > 0 and len(cuts):
        for cut in cuts:
            lo, hi = max(1, cut - seam), min(NP - 1, cut + seam)
            out[lo:hi] = 0.5 * out[lo:hi] + 0.25 * out[lo - 1:hi - 1] + 0.25 * out[lo + 1:hi + 1]
    # renormalize to unit canonical: start at origin, endpoint on +x at distance 1
    out = out - out[0]
    d = math.hypot(out[-1, 0], out[-1, 1])
    if d < 1e-6:
        return None
    phi = math.atan2(out[-1, 1], out[-1, 0])
    c, s = math.cos(-phi), math.sin(-phi)
    ux = (out[:, 0] * c - out[:, 1] * s) / d
    uy = (out[:, 0] * s + out[:, 1] * c) / d
    return np.stack([ux, uy], axis=1)


def main():
    rng = np.random.default_rng(0)
    pool = load_human_pool_raw_points(seed=0)
    shapes, dists = [], []
    for p in pool:
        c = to_canonical_at(p, NP)
        if c is not None:
            shapes.append(c[0]); dists.append(c[1])
    shapes = np.asarray(shapes); dists = np.asarray(dists)
    hu, hu_d = shapes[:8000], dists[:8000]
    print(f"[compose] {len(shapes)} shapes; human split=8000", flush=True)

    hidx = rng.choice(len(hu), size=N_EVAL, replace=False)
    human_rows = [extract_features(shape_to_raw(hu[i], hu_d[i], rng)) for i in hidx]

    def eval_config(K, n_seg, seam):
        bot_rows, comp = [], []
        for _ in range(N_EVAL):
            sh = compose(shapes, dists, K, n_seg, seam, rng)
            if sh is None:
                continue
            comp.append(sh.ravel())
            di = dists[rng.integers(0, K)]
            bot_rows.append(extract_features(shape_to_raw(sh, di, rng)))
        n = min(len(human_rows), len(bot_rows))
        df = pd.DataFrame(human_rows[:n] + bot_rows[:n])
        y = np.array([0] * n + [1] * n)
        X = df[SHAPE_ONLY_FEATURES].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
        best, _ = tune_best_model(Xtr, ytr)
        acc = accuracy_score(yte, best["estimator"].predict(Xte))
        C = np.asarray(comp)
        d, _ = NearestNeighbors(n_neighbors=2).fit(C).kneighbors(C)
        dup = (d[:, 1] < EPS).mean()          # near-duplicate rate among composed flicks
        return acc, dup

    print("\n[compose] K | n_seg | seam | single-move acc | near-dup rate", flush=True)
    print("        " + "-" * 56, flush=True)
    out = []
    for K, n_seg, seam in [(6000, 1, 0), (200, 2, 0), (200, 2, 2), (200, 3, 2), (50, 3, 2), (50, 4, 2)]:
        acc, dup = eval_config(K, n_seg, seam)
        out.append({"K": K, "n_seg": n_seg, "seam": seam, "single_acc": acc, "near_dup": dup})
        print(f"     {K:>5} | {n_seg:>5} | {seam:>4} | {acc:>15.3f} | {dup:>12.3f}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "attack_segment_compose.json").write_text(json.dumps({"sweep": out}, indent=2))
    print(f"\n[compose] wrote {RESULTS_DIR / 'attack_segment_compose.json'}", flush=True)


if __name__ == "__main__":
    main()
