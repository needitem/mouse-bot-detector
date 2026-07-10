#!/usr/bin/env python3
"""Does the sigma=0 floor vanish under a TRULY fair baseline?

score_anchor_fair.py pushes human through to_canonical->decode but NOT through
the anchor path's normalize->clamp(+/-8)->unnormalize round-trip. The anchor
strokes DO go through that clamp (Xs = normalize(x).clamp(-8,8) before encode).
Since the flow round-trip is numerically exact (recon MSE ~6e-12), anchor s0.0
is essentially clamp(normalize(x)) unnormalized - so the only thing separating it
from a naive human baseline is the input clamp, an information-destroying step on
heavy-tailed dims, NOT flow error.

This builds the human baseline through the IDENTICAL normalize->clamp->unnormalize
(mean/std recomputed exactly as the trainer does: first 14000 canonical human
strokes, kept dims), so both classes share the clamp distortion. If s0.0 now
drops toward 0.5, the floor is fully an instrument artifact.
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
N = 48
DIM = N * 2 + 2
MIN_MT = 40.0
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
    return np.concatenate([np.stack([sx, sy], 1).ravel(),
                           [math.log(dist), math.log(mt)]])


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


def decode_write(Vk, path, seed):
    rng = np.random.default_rng(seed)
    V = np.zeros((Vk.shape[0], DIM), np.float64)
    V[:, KEEP] = Vk
    V[:, N*2 - 2] = 1.0
    out = []
    for v in V:
        shape = v[:N*2].reshape(N, 2)
        dist = math.exp(min(v[N*2], 12)); mt = max(math.exp(min(v[N*2+1], 8)), MIN_MT)
        if not (np.isfinite(dist) and dist >= 5.0): continue
        xs = shape[:, 0] * dist; ys = shape[:, 1] * dist
        ang = rng.uniform(0, 2*math.pi); c, s = math.cos(ang), math.sin(ang)
        rx = xs*c - ys*s; ry = xs*s + ys*c
        t = np.linspace(0, mt, N)
        pts = [[float(a_), float(b_), float(t_)] for a_, b_, t_ in zip(rx, ry, t)]
        if all(np.isfinite([p for r in pts for p in r])): out.append(pts)
    with open(path, "w") as f:
        for pts in out:
            f.write(json.dumps({"points": pts}) + "\n")
    return out


def load_points(path):
    return [rec["points"] for rec in (json.loads(l) for l in open(path))
            if len(rec["points"]) >= 4]


def strong_acc(human_rows, bot_points):
    bot_rows = [extract_features(p) for p in bot_points]
    n = min(MAX_PER_CLASS, len(human_rows), len(bot_rows))
    rng = np.random.default_rng(0)
    h = [human_rows[i] for i in rng.choice(len(human_rows), size=n, replace=False)]
    b = [bot_rows[i] for i in rng.choice(len(bot_rows), size=n, replace=False)]
    df = pd.DataFrame(h + b)
    y = np.array([0] * n + [1] * n)
    X = df[SHAPE_ONLY_FEATURES].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                          random_state=RANDOM_STATE)
    best, _ = tune_best_model(Xtr, ytr)
    return accuracy_score(yte, best["estimator"].predict(Xte)), n


def main():
    files = sys.argv[1:] or ["anchorinv_s0.0_bot_movements.jsonl",
                             "anchorinv_s0.1_bot_movements.jsonl"]
    # exact same mean/std the trainer used
    Xk = load_kept(DATA_DIR / "human_movements.jsonl")
    mean = Xk.mean(0); std = np.maximum(Xk.std(0), 1e-3)
    # truly-fair human: normalize -> clamp(+/-8) -> unnormalize (the anchor path,
    # minus the flow round-trip which is numerically exact)
    clamped = np.clip((Xk - mean) / std, -8, 8) * std + mean
    n_clip = int(np.any(np.abs((Xk - mean) / std) > 8, axis=1).sum())
    print(f"[clampfair] {len(Xk)} human strokes, {n_clip} have >=1 dim clipped at +/-8", flush=True)

    naive = DATA_DIR / "human_fair_canonical.jsonl"          # from score_anchor_fair (no clamp)
    if not naive.exists():
        decode_write(Xk, naive, seed=7)
    cfair = DATA_DIR / "human_clampfair_canonical.jsonl"
    decode_write(clamped, cfair, seed=7)

    naive_rows = [extract_features(p) for p in load_points(naive)]
    cfair_rows = [extract_features(p) for p in load_points(cfair)]
    print(f"[clampfair] naive baseline n={len(naive_rows)}  clamp-fair baseline n={len(cfair_rows)}", flush=True)

    print(f"\n[clampfair] {'file':38s} | vs-naive | vs-clamp-fair", flush=True)
    print("        " + "-" * 62, flush=True)
    for f in files:
        p = DATA_DIR / f
        if not p.exists():
            print(f"        {f:38s} | MISSING"); continue
        bot = load_points(p)
        a_naive, _ = strong_acc(naive_rows, bot)
        a_cfair, _ = strong_acc(cfair_rows, bot)
        print(f"        {p.name:38s} | {a_naive:8.3f} | {a_cfair:12.3f}", flush=True)


if __name__ == "__main__":
    main()
