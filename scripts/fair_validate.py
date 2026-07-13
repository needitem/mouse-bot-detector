#!/usr/bin/env python3
"""Fair-reference validation. The default detector compares aimbot flicks to ALL
human movements - but aim flicks are a distinct subset (fast, straight), so that
tells them apart on distribution alone. A real anti-cheat that models aim motion
compares aim-context to aim-context. This builds the human AIM-FLICK subset (real
strokes passing the same aim filter, at their original distance/direction) and
scores: (floor) aim-flick vs aim-flick, and (test) needaimbot flick vs aim-flick.
"""
import json, math, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features
from train_detector import SHAPE_ONLY_FEATURES, SEARCH_SPACES, tune_best_model, TEST_SIZE, RANDOM_STATE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

def aim_flicks(path, eff=0.75, lat=0.15, dmin=30, dmax=2000, tmin=120, tmax=1600, cap=8000):
    out = []
    for l in open(path):
        p = np.asarray(json.loads(l)["points"], float)
        if len(p) < 5: continue
        x, y, t = p[:,0]-p[0,0], p[:,1]-p[0,1], p[:,2]-p[0,2]
        d = math.hypot(x[-1], y[-1]); dur = t[-1]
        if d < dmin or d > dmax or dur < tmin or dur > tmax: continue
        pl = np.hypot(np.diff(x), np.diff(y)).sum()
        if pl < 1e-6 or d/pl < eff: continue
        phi = math.atan2(y[-1], x[-1]); c, s = math.cos(-phi), math.sin(-phi)
        uy = (x*s + y*c)/d
        if np.max(np.abs(uy)) > lat: continue
        out.append([[float(a), float(b), float(u)] for a, b, u in zip(x, y, t)])
        if len(out) >= cap: break
    return out

def feats(mvs):
    F = []
    for pts in mvs:
        d = extract_features(pts)
        if isinstance(d, dict):
            v = [d[k] for k in SHAPE_ONLY_FEATURES]
            if np.all(np.isfinite(v)): F.append(v)
    return np.array(F)

def strong_acc(F0, F1):
    n = min(len(F0), len(F1))
    X = np.vstack([F0[:n], F1[:n]]); y = np.r_[np.zeros(n), np.ones(n)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    best, _ = tune_best_model(Xtr, ytr)
    return accuracy_score(yte, best["estimator"].predict(Xte))

def main():
    bot_file = sys.argv[1]
    human = aim_flicks(str(DATA/"human_movements.jsonl"))
    print(f"[fair] human aim-flicks: {len(human)}")
    Fh = feats(human)
    rng = np.random.default_rng(0); idx = rng.permutation(len(Fh))
    Fa, Fb = Fh[idx[:len(Fh)//2]], Fh[idx[len(Fh)//2:]]
    print(f"[fair] FLOOR (aim-flick vs aim-flick): {strong_acc(Fa, Fb):.3f}")
    bot = [json.loads(l)["points"] for l in open(DATA/bot_file)]
    Fbot = feats(bot)
    print(f"[fair] needaimbot vs human aim-flicks: {strong_acc(Fh, Fbot):.3f}")

if __name__ == "__main__":
    main()
