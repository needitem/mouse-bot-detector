#!/usr/bin/env python3
"""Attacker sweet-spot search: is there a human-variability perturbation size
that escapes BOTH detectors at once?

attack_manifold_perturb.py showed human-variability-direction perturbation
(s0 + mag*(s_a - s_b)) stays far more human-like than gaussian at the same
shift. But at the sizes tested it was still caught single-move (~0.62). This
sweeps SMALL mags and measures, on the SAME perturbed strokes:
  - single-move accuracy (human vs perturbed; 0.5 = evades)
  - reuse-detection rate: simulate a K=6000, N=100 session where each flick is
    a pooled source + an INDEPENDENT human-var perturbation, and check whether
    same-source repeats still fall below eps (near-duplicate). Low rate = evades.

A mag where single-move ~0.5 AND reuse-rate ~0 is a genuine escape from both.
Run: python -u scripts/attack_sweet_spot.py
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EPS = 0.0594
N_EVAL = 2000
POOL_K = 6000
SESSION_N = 100


def shape_to_raw(shape, dist, rng):
    theta = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(theta), math.sin(theta)
    xs, ys = shape[:, 0] * dist, shape[:, 1] * dist
    rx, ry = xs * c - ys * s, xs * s + ys * c
    t = np.linspace(0.0, dist * 1.5 + 200.0, len(shape))
    return list(zip(rx.tolist(), ry.tolist(), t.tolist()))


def main():
    rng = np.random.default_rng(0)
    pool = load_human_pool_raw_points(seed=0)
    shapes, dists = [], []
    for pts in pool:
        c = to_canonical_at(pts, N_SHAPE_POINTS)
        if c is not None:
            shapes.append(c[0]); dists.append(c[1])
    shapes = np.asarray(shapes); dists = np.asarray(dists)
    print(f"[sweet] {len(shapes)} shapes, eps={EPS}, K={POOL_K}, N={SESSION_N}", flush=True)

    hu = shapes[:8000]; hu_d = dists[:8000]
    bo = shapes[8000:]; bo_d = dists[8000:]

    hidx = rng.choice(len(hu), size=N_EVAL, replace=False)
    human_rows = [extract_features(shape_to_raw(hu[i], hu_d[i], rng)) for i in hidx]

    def humanvar_pert(mag):
        a, b = rng.integers(0, len(bo), size=2)
        return mag * (bo[a] - bo[b])

    def single_acc(mag):
        bot_rows, shifts = [], []
        idx = rng.integers(0, len(bo), size=N_EVAL)
        for i in idx:
            p = humanvar_pert(mag)
            shifts.append(float(np.linalg.norm(p)))
            bot_rows.append(extract_features(shape_to_raw(bo[i] + p, bo_d[i], rng)))
        n = min(len(human_rows), len(bot_rows))
        df = pd.DataFrame(human_rows[:n] + bot_rows[:n])
        y = np.array([0] * n + [1] * n)
        X = df[SHAPE_ONLY_FEATURES].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
        best, _ = tune_best_model(Xtr, ytr)
        return float(np.mean(shifts)), accuracy_score(yte, best["estimator"].predict(Xte))

    def reuse_rate(mag, trials=40):
        """P(session flagged) - each flick = pooled source + independent
        human-var perturbation; same-source repeats near-duplicate?"""
        boflat = bo.reshape(len(bo), -1)
        hits = 0
        for _ in range(trials):
            src = rng.integers(0, POOL_K, size=SESSION_N)
            V = np.empty((SESSION_N, boflat.shape[1]))
            for k, si in enumerate(src):
                V[k] = boflat[si] + humanvar_pert(mag).ravel()
            best = np.inf
            for i in range(SESSION_N):
                dd = np.sqrt(((V[i + 1:] - V[i]) ** 2).sum(axis=1))
                if dd.size:
                    best = min(best, dd.min())
            hits += best < EPS
        return hits / trials

    print("\n[sweet] mag   |  δ (shift) | single-acc | reuse-detect | escapes both?", flush=True)
    print("        " + "-" * 60, flush=True)
    out = []
    for mag in [0.03, 0.05, 0.07, 0.10, 0.14]:
        delta, acc = single_acc(mag)
        rr = reuse_rate(mag)
        escapes = (acc < 0.60) and (rr < 0.10)
        out.append({"mag": mag, "delta": delta, "single_acc": acc, "reuse_detect": rr,
                    "escapes_both": escapes})
        print(f"        {mag:<5} | {delta:>9.4f} | {acc:>10.3f} | {rr:>12.2f} | "
              f"{'YES' if escapes else 'no':>12}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "attack_sweet_spot.json").write_text(json.dumps(
        {"eps": EPS, "sweep": out}, indent=2))
    print(f"\n[sweet] wrote {RESULTS_DIR / 'attack_sweet_spot.json'}", flush=True)


if __name__ == "__main__":
    main()
