#!/usr/bin/env python3
"""Honesty control for the fair scorer: split REAL human strokes into two
disjoint halves, push BOTH through the identical to_canonical->decode pipeline,
label them 0/1, and run the same strong detector. Two samples from the same
distribution MUST score ~0.5. If this returns ~0.5, the scorer is honest and the
anchor s0.0 = 0.82 is a real property of the anchor strokes. If it returns ~0.8,
the scorer manufactures a spurious tell and every anchor number is suspect.

Also scores a "reals-with-replacement + rotation" set (mimicking anchor s0.0's
construction, but decoding the reals directly with NO flow) vs the fair baseline,
to isolate whether the flow round-trip itself is what adds the tell.
"""
import json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_detector import SHAPE_ONLY_FEATURES, tune_best_model, TEST_SIZE, RANDOM_STATE, MAX_PER_CLASS
from features import extract_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
N = 48; DIM = N * 2 + 2; MIN_MT = 40.0
DROP = [0, 1, N * 2 - 2, N * 2 - 1]
KEEP = [i for i in range(DIM) if i not in DROP]


def to_canonical(points):
    pts = np.asarray(points, float)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    dist = math.hypot(dx, dy); mt = t[-1] - t[0]
    if dist < 5.0 or mt < MIN_MT:
        return None
    ang = math.atan2(dy, dx); c, s = math.cos(-ang), math.sin(-ang)
    rx = (x - x[0]) * c - (y - y[0]) * s
    ry = (x - x[0]) * s + (y - y[0]) * c
    rx, ry = rx / dist, ry / dist
    tg = np.linspace(t[0], t[-1], N)
    sx = np.interp(tg, t, rx); sy = np.interp(tg, t, ry)
    return np.concatenate([np.stack([sx, sy], 1).ravel(), [math.log(dist), math.log(mt)]])


def load_kept(path, cap=14000):
    Vk = []
    for i, line in enumerate(open(path)):
        if i >= cap: break
        pts = json.loads(line)["points"]
        if len(pts) < 4: continue
        v = to_canonical(pts)
        if v is not None and np.all(np.isfinite(v)):
            Vk.append(v[KEEP])
    return np.asarray(Vk, np.float64)


def decode_rows(Vk, seed):
    """Vk (already un-normalized kept dims) -> decode pipeline -> feature rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for vk in Vk:
        v = np.zeros(DIM); v[KEEP] = vk; v[N*2 - 2] = 1.0
        shape = v[:N*2].reshape(N, 2)
        dist = math.exp(min(v[N*2], 12)); mt = max(math.exp(min(v[N*2+1], 8)), MIN_MT)
        if not (np.isfinite(dist) and dist >= 5.0): continue
        xs = shape[:, 0] * dist; ys = shape[:, 1] * dist
        ang = rng.uniform(0, 2*math.pi); c, s = math.cos(ang), math.sin(ang)
        rx = xs*c - ys*s; ry = xs*s + ys*c
        t = np.linspace(0, mt, N)
        rows.append(extract_features([[float(a), float(b), float(tt)]
                                      for a, b, tt in zip(rx, ry, t)]))
    return rows


def acc(rows0, rows1):
    n = min(MAX_PER_CLASS, len(rows0), len(rows1))
    rng = np.random.default_rng(0)
    a = [rows0[i] for i in rng.choice(len(rows0), size=n, replace=False)]
    b = [rows1[i] for i in rng.choice(len(rows1), size=n, replace=False)]
    df = pd.DataFrame(a + b); y = np.array([0]*n + [1]*n)
    X = df[SHAPE_ONLY_FEATURES].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                          random_state=RANDOM_STATE)
    best, _ = tune_best_model(Xtr, ytr)
    return accuracy_score(yte, best["estimator"].predict(Xte)), n


def main():
    Xk = load_kept(DATA_DIR / "human_movements.jsonl")
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(Xk))
    half = len(Xk) // 2
    A, B = Xk[perm[:half]], Xk[perm[half:]]

    print("[control] disjoint-half human vs human (both decoded, DIFFERENT rotation seeds)...", flush=True)
    rowsA = decode_rows(A, seed=11)
    rowsB = decode_rows(B, seed=22)
    a1, n1 = acc(rowsA, rowsB)
    print(f"[control] human(A) vs human(B): acc={a1:.3f} (n={n1})  <-- must be ~0.5", flush=True)

    # reals-with-replacement + rotation (anchor s0.0 construction, NO flow) vs fair baseline
    idx = rng.integers(0, len(Xk), 4000)
    rowsRep = decode_rows(Xk[idx], seed=33)
    rowsFair = decode_rows(Xk, seed=7)
    a2, n2 = acc(rowsFair, rowsRep)
    print(f"[control] fair-human vs reals-w-replacement(4000)+rot: acc={a2:.3f} (n={n2})  <-- ~0.5 if honest", flush=True)


if __name__ == "__main__":
    main()
