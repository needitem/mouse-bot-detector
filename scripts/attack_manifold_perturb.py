#!/usr/bin/env python3
"""Attacker escalation: escape BOTH detectors with manifold-aware perturbation.

The defender's two axes pull opposite ways under gaussian jitter (see
detect_replay_dilemma.py): small jitter -> near-duplicate reuse is caught;
large jitter -> the stroke leaves the human manifold and the single-movement
detector catches it. The idea here: perturb a real stroke NOT with spatial
gaussian noise but along the direction real humans actually vary
(motor variability), i.e. toward other real strokes. That should move the
stroke far enough to break near-duplicate matching while staying ON the human
manifold, so single-move can't tell.

For each perturbation size we compare two schemes at the SAME canonical shift δ:
  - gaussian:   s0 + N(0, σ)                      (off-manifold, isotropic)
  - human-var:  s0 + γ (s_a - s_b)                (real difference vector = a
                 direction humans genuinely differ along)
and report, for each: the achieved δ, the single-move detector accuracy
(human vs perturbed - lower = better evasion), and whether δ clears the reuse
threshold eps=0.0594 (so repeats no longer near-duplicate).

If human-var reaches δ > eps with single-move still ~0.5, the attacker escapes
both at once. Run: python -u scripts/attack_manifold_perturb.py
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


def shape_to_raw(shape, dist, rng):
    """Canonical (N,2) unit shape -> raw (x,y,t) at a random direction/scale."""
    theta = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(theta), math.sin(theta)
    xs, ys = shape[:, 0] * dist, shape[:, 1] * dist
    rx, ry = xs * c - ys * s, xs * s + ys * c
    t = np.linspace(0.0, dist * 1.5 + 200.0, len(shape))   # plausible monotone timing
    return list(zip(rx.tolist(), ry.tolist(), t.tolist()))


def main():
    rng = np.random.default_rng(0)
    pool = load_human_pool_raw_points(seed=0)
    shapes, dists = [], []
    for pts in pool:
        c = to_canonical_at(pts, N_SHAPE_POINTS)
        if c is not None:
            shapes.append(c[0]); dists.append(c[1])
    shapes = np.asarray(shapes); dists = np.asarray(dists)   # (M, N, 2), (M,)
    flat = shapes.reshape(len(shapes), -1)
    print(f"[attack] {len(shapes)} canonical shapes, dim={flat.shape[1]}, eps={EPS}", flush=True)

    # disjoint splits so the detector can't just memorize the source pool
    hu = shapes[:8000]; hu_d = dists[:8000]
    bo = shapes[8000:]; bo_d = dists[8000:]; bo_flat = flat[8000:]

    # human class: real strokes, reconstructed through the SAME raw pipeline
    hidx = rng.choice(len(hu), size=N_EVAL, replace=False)
    human_rows = [extract_features(shape_to_raw(hu[i], hu_d[i], rng)) for i in hidx]

    def eval_scheme(kind, mag):
        shifts, bot_rows = [], []
        idx = rng.integers(0, len(bo), size=N_EVAL)
        for i in idx:
            s0 = bo[i]
            if kind == "gaussian":
                pert = rng.normal(0, mag, s0.shape)
            else:  # human-var: real difference vector between two random strokes
                a, b = rng.integers(0, len(bo), size=2)
                pert = mag * (bo[a] - bo[b])
            s1 = s0 + pert
            shifts.append(float(np.linalg.norm(s1 - s0)))
            bot_rows.append(extract_features(shape_to_raw(s1, bo_d[i], rng)))
        delta = float(np.mean(shifts))
        n = min(len(human_rows), len(bot_rows))
        df = pd.DataFrame(human_rows[:n] + bot_rows[:n])
        y = np.array([0] * n + [1] * n)
        X = df[SHAPE_ONLY_FEATURES].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
        best, _ = tune_best_model(Xtr, ytr)
        acc = accuracy_score(yte, best["estimator"].predict(Xte))
        return delta, acc

    print("\n[attack] scheme     | mag   |  δ (shift) | single-acc | δ>eps (reuse-safe)", flush=True)
    print("         " + "-" * 62, flush=True)
    out = []
    for kind, mags in [("gaussian", [0.02, 0.05, 0.10, 0.18]),
                       ("human-var", [0.15, 0.30, 0.50, 0.80])]:
        for m in mags:
            delta, acc = eval_scheme(kind, m)
            safe = "YES" if delta > EPS else "no"
            out.append({"kind": kind, "mag": m, "delta": delta, "single_acc": acc,
                        "reuse_safe": delta > EPS})
            print(f"         {kind:<10} | {m:<5} | {delta:>9.4f} | {acc:>10.3f} | {safe:>10}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "attack_manifold_perturb.json").write_text(json.dumps(
        {"eps": EPS, "sweep": out}, indent=2))
    # verdict: among reuse-safe (δ>eps) points, the lowest single-move accuracy
    safe = [o for o in out if o["reuse_safe"]]
    if safe:
        best = min(safe, key=lambda o: o["single_acc"])
        print(f"\n[attack] best reuse-safe evasion: {best['kind']} mag={best['mag']} "
              f"-> δ={best['delta']:.3f}, single-move acc={best['single_acc']:.3f}", flush=True)
    print(f"[attack] wrote {RESULTS_DIR / 'attack_manifold_perturb.json'}", flush=True)


if __name__ == "__main__":
    main()
