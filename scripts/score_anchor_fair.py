#!/usr/bin/env python3
"""Fair-baseline scorer for latent-anchored replay files.

Reproduces the RESEARCH.md sigma-sweep table (0.608 floor .. 0.677 at sig~0.1):
  * FAIR baseline = human strokes pushed through the SAME 48-point canonical ->
    decode pipeline as the generated strokes, so the resampling loss cancels and
    the only thing the detector can latch onto is flow-induced distortion.
  * strong detector = train_detector's full tune_best_model over shape_only.
  * near-dup fraction = fraction of generated strokes whose nearest OTHER
    generated stroke (canonical 48-pt shape) is within eps=0.0594.

Usage: python score_anchor_fair.py <anchor_file1> [<anchor_file2> ...]
Anchor files are paths relative to data/processed/ (or absolute).
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
EPS = 0.0594
N = 48
DIM = N * 2 + 2
MIN_MT = 40.0
DROP = [0, 1, N * 2 - 2, N * 2 - 1]
KEEP = [i for i in range(DIM) if i not in DROP]


# to_canonical / _decode_write copied verbatim from latent_anchor_invertible.py
# (pure numpy; kept here so the scorer has no torch dependency).
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


def _decode_write(Vk, std, mean, keep, path, rng, tag):
    Vk = Vk.astype(np.float64) * std + mean
    V = np.zeros((Vk.shape[0], DIM), np.float64)
    V[:, keep] = Vk
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
    print(f"[fair:{tag}] wrote {len(out)} -> {path}", flush=True)


def build_fair_human(path, cap=14000):
    """Human raw points -> canonical -> SAME decode as the generator. The kept
    dims are already un-normalized, so std=1, mean=0 in _decode_write."""
    Vk = []
    for i, line in enumerate(open(path)):
        if i >= cap: break
        pts = json.loads(line)["points"]
        if len(pts) < 4: continue
        v = to_canonical(pts)
        if v is not None and np.all(np.isfinite(v)):
            Vk.append(v[KEEP])
    Vk = np.asarray(Vk, np.float64)
    out = DATA_DIR / "human_fair_canonical.jsonl"
    rng = np.random.default_rng(7)
    _decode_write(Vk, np.ones(len(KEEP)), np.zeros(len(KEEP)), KEEP, str(out), rng, "human-fair")
    return out


def load_points(path):
    P = []
    for line in open(path):
        rec = json.loads(line)
        if len(rec["points"]) >= 4:
            P.append(rec["points"])
    return P


def canonical_shapes(points_list):
    S = []
    for pts in points_list:
        v = to_canonical(pts)
        if v is not None and np.all(np.isfinite(v)):
            S.append(v[:N * 2])          # 48-pt shape, ravel'd (rotation/scale canonical)
    return np.asarray(S)


def near_dup_fraction(points_list):
    S = canonical_shapes(points_list)
    if len(S) < 2:
        return float("nan")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(S)
    dd, _ = nn.kneighbors(S)
    nearest_other = dd[:, 1]
    return float(np.mean(nearest_other < EPS))


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
    files = sys.argv[1:]
    if not files:
        print("usage: score_anchor_fair.py <anchor_file> ..."); return
    print("[score] building fair human baseline (canonical->decode)...", flush=True)
    fair = build_fair_human(DATA_DIR / "human_movements.jsonl")
    human_rows = [extract_features(p) for p in load_points(fair)]
    print(f"[score] fair human strokes: {len(human_rows)}", flush=True)

    print(f"\n[score] {'file':40s} | strong-acc | near-dup", flush=True)
    print("        " + "-" * 66, flush=True)
    for f in files:
        p = Path(f) if Path(f).is_absolute() else DATA_DIR / f
        if not p.exists():
            print(f"        {f:40s} | MISSING"); continue
        bot = load_points(p)
        acc, n = strong_acc(human_rows, bot)
        nd = near_dup_fraction(bot)
        print(f"        {p.name:40s} | {acc:10.3f} | {nd:8.3f}   (n={n})", flush=True)


if __name__ == "__main__":
    main()
