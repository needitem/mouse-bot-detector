#!/usr/bin/env python3
"""Honest raw-baseline scorer for minimal-perturbation replay (jitter/warp).

Unlike score_anchor_fair.py (whose human baseline is decoded through the 48-pt
canonical pipeline -- the right control for flow anchors, which also go through
that decode), minimal-replay bots are raw human strokes + a raw-pixel
perturbation. The matched control is therefore RAW human strokes (no decode),
exactly detect_replay_dilemma.py's protocol. Both sides: extract_features on raw
points, strong detector (tune_best_model). near-dup = fraction of the generated
set whose nearest OTHER stroke (canonical 48-pt shape) is within eps=0.0594.

Usage: python score_raw.py <bot_file1> [<bot_file2> ...]
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
N = 48; MIN_MT = 40.0


def load_points(path, cap=None):
    P = []
    for i, line in enumerate(open(path)):
        if cap and i >= cap: break
        rec = json.loads(line)
        if len(rec["points"]) >= 4:
            P.append(rec["points"])
    return P


def to_shape(points):
    pts = np.asarray(points, float)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    dx, dy = x[-1]-x[0], y[-1]-y[0]; dist = math.hypot(dx, dy); mt = t[-1]-t[0]
    if dist < 5.0 or mt < MIN_MT: return None
    ang = math.atan2(dy, dx); c, s = math.cos(-ang), math.sin(-ang)
    rx = (x-x[0])*c - (y-y[0])*s; ry = (x-x[0])*s + (y-y[0])*c
    rx, ry = rx/dist, ry/dist
    tg = np.linspace(t[0], t[-1], N)
    return np.concatenate([np.interp(tg, t, rx), np.interp(tg, t, ry)])


def near_dup(points_list):
    S = [s for s in (to_shape(p) for p in points_list) if s is not None]
    S = np.asarray(S)
    if len(S) < 2: return float("nan")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(S)
    dd, _ = nn.kneighbors(S)
    return float(np.mean(dd[:, 1] < EPS))


def acc(human_rows, bot_points):
    bot = [extract_features(p) for p in bot_points]
    n = min(MAX_PER_CLASS, len(human_rows), len(bot))
    rng = np.random.default_rng(0)
    h = [human_rows[i] for i in rng.choice(len(human_rows), n, replace=False)]
    b = [bot[i] for i in rng.choice(len(bot), n, replace=False)]
    df = pd.DataFrame(h + b); y = np.array([0]*n + [1]*n)
    X = df[SHAPE_ONLY_FEATURES].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    best, _ = tune_best_model(Xtr, ytr)
    return accuracy_score(yte, best["estimator"].predict(Xte)), n


def main():
    files = sys.argv[1:]
    print("[raw] loading raw human baseline...", flush=True)
    human = load_points(DATA_DIR / "human_movements.jsonl")
    human_rows = [extract_features(p) for p in human]
    print(f"[raw] raw human strokes: {len(human_rows)}", flush=True)
    print(f"\n[raw] {'file':34s} | strong-acc | near-dup", flush=True)
    print("      " + "-" * 60, flush=True)
    for f in files:
        p = Path(f) if Path(f).is_absolute() else DATA_DIR / f
        if not p.exists():
            print(f"      {f:34s} | MISSING"); continue
        bot = load_points(p)
        a, n = acc(human_rows, bot)
        nd = near_dup(bot)
        print(f"      {p.name:34s} | {a:10.3f} | {nd:8.3f}   (n={n})", flush=True)


if __name__ == "__main__":
    main()
