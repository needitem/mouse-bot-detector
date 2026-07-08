#!/usr/bin/env python3
"""The attacker's dilemma: perturbation jitter can't escape both detectors.

detect_replay_reuse.py catches warped replay by NEAR-DUPLICATE reuse (a finite
pool repeats). The obvious counter is to add larger per-flick perturbation so
repeats no longer look identical - but a real human stroke perturbed enough to
break near-duplicate matching has been pushed off the human manifold, so the
SINGLE-movement strong detector starts catching it again.

This sweeps the perturbation magnitude and measures BOTH detectors at each
level, on the SAME perturbed strokes:
  1. near-duplicate: min canonical pair distance in a session vs the eps from
     detect_replay_reuse.py (does reuse still collapse to a duplicate?)
  2. single-movement: train_detector.py's strong detector, human vs perturbed
     replay (has the stroke left the human distribution?)
The combined detector fires if EITHER catches it - showing there's no jitter
that evades both at once.

Run: python scripts/detect_replay_dilemma.py
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

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

POOL_K = 6000            # attacker's recorded-stroke pool size
SESSION_N = 100          # flicks the defender observes for the reuse check
EPS = 0.0594             # near-duplicate threshold from detect_replay_reuse.py
N_EVAL = 3000            # per-class movements for the single-movement detector


def canonical_shape(pts):
    c = to_canonical_at(pts, N_SHAPE_POINTS)
    return None if c is None else (c[0], c[1])   # (shape (N,2), distance)


def perturb_replay(pts, jitter_px, rng):
    """Rigid-transform a real stroke to a random target and add per-point
    Gaussian jitter of the given magnitude (px). Returns raw (x,y,t)."""
    p = np.asarray(pts, dtype=float)
    rel = p[:, :2] - p[0, :2]
    d = math.hypot(rel[-1, 0], rel[-1, 1])
    if d < 5.0:
        return None
    theta = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(theta), math.sin(theta)
    rx = rel[:, 0] * c - rel[:, 1] * s + rng.normal(0, jitter_px, len(rel))
    ry = rel[:, 0] * s + rel[:, 1] * c + rng.normal(0, jitter_px, len(rel))
    t = p[:, 2] - p[0, 2]
    return list(zip(rx.tolist(), ry.tolist(), t.tolist()))


def main():
    rng = np.random.default_rng(0)
    pool = load_human_pool_raw_points(seed=0)
    print(f"[dilemma] {len(pool)} human strokes; K={POOL_K}, session N={SESSION_N}, eps={EPS}")

    # precompute canonical shapes/dists for reuse-detector sessions
    shapes, dists = [], []
    for pts in pool:
        cs = canonical_shape(pts)
        if cs is not None:
            shapes.append(cs[0].ravel()); dists.append(cs[1])
    shapes = np.asarray(shapes); dists = np.asarray(dists)

    # human feature rows (fixed) for the single-movement detector
    human_idx = rng.choice(len(pool), size=N_EVAL, replace=False)
    human_rows = [extract_features(pool[i]) for i in human_idx]

    def reuse_detect_rate(jitter_px, trials=40):
        """P(session flagged) = P(a near-duplicate survives this jitter)."""
        hits = 0
        for _ in range(trials):
            src = rng.integers(0, POOL_K, size=SESSION_N)
            base = shapes[src]
            # jitter in canonical space = jitter_px / reach distance
            jit = np.array([rng.normal(0, jitter_px / max(dists[s], 1e-6), base.shape[1])
                            for s in src])
            V = base + jit
            # min pairwise distance
            best = np.inf
            for i in range(len(V)):
                dd = np.sqrt(((V[i + 1:] - V[i]) ** 2).sum(axis=1))
                if dd.size:
                    best = min(best, dd.min())
            hits += best < EPS
        return hits / trials

    def single_detect_acc(jitter_px):
        """Strong single-movement detector accuracy, human vs perturbed replay."""
        bot_rows = []
        srcs = rng.integers(0, len(pool), size=N_EVAL)
        for i in srcs:
            pr = perturb_replay(pool[i], jitter_px, rng)
            if pr is not None and len(pr) >= 4:
                bot_rows.append(extract_features(pr))
        n = min(len(human_rows), len(bot_rows))
        df = pd.DataFrame(human_rows[:n] + bot_rows[:n])
        y = np.array([0] * n + [1] * n)
        X = df[SHAPE_ONLY_FEATURES].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                              stratify=y, random_state=RANDOM_STATE)
        best, _ = tune_best_model(Xtr, ytr)
        return accuracy_score(yte, best["estimator"].predict(Xte))

    jitters = [1.0, 3.0, 6.0, 10.0, 16.0, 25.0, 40.0]
    print("\n[dilemma] jitter(px) | reuse-detect P | single-move acc | combined (either)")
    print("          " + "-" * 58)
    out = []
    for j in jitters:
        r = reuse_detect_rate(j)
        a = single_detect_acc(j)
        combined = max(r, a)      # defender runs both; catches if either fires
        out.append({"jitter_px": j, "reuse_detect": r, "single_acc": a, "combined": combined})
        print(f"          {j:>8.1f} | {r:>13.2f} | {a:>14.3f} | {combined:>15.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "replay_dilemma.json").write_text(json.dumps(
        {"pool_K": POOL_K, "session_N": SESSION_N, "eps": EPS, "sweep": out}, indent=2))
    print(f"\n[dilemma] wrote {RESULTS_DIR / 'replay_dilemma.json'}")
    best_evade = min(out, key=lambda o: o["combined"])
    print(f"[dilemma] attacker's best jitter = {best_evade['jitter_px']}px, "
          f"but combined detector still at {best_evade['combined']:.2f}")


if __name__ == "__main__":
    main()
