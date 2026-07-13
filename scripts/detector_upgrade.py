#!/usr/bin/env python3
"""Detector upgrade vs elastic warped replay. elastic = a real stroke + a SMOOTH
low-frequency PERPENDICULAR bend. So relative to its nearest real-DB neighbor, an
elastic flick differs by a low-frequency, mostly-perpendicular residual, whereas a
real flick differs by high-frequency jerk/tremor. New per-flick features capture
that residual structure (robust to the deformation basis, unlike matching the
exact sine modes). Compares a baseline shape_only detector to shape_only + the
new residual features on real-vs-bot.
"""
import json, math, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features, FEATURE_NAMES
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
NP = 48
SHAPE_ONLY = [f for f in FEATURE_NAMES if f not in {"sample_interval_mean", "sample_interval_cv"}]

def canon(pts):
    p = np.asarray(pts, float); x, y, t = p[:,0], p[:,1], p[:,2]
    dx, dy = x[-1]-x[0], y[-1]-y[0]; d = math.hypot(dx, dy)
    if d < 5 or t[-1]-t[0] < 40: return None
    a = math.atan2(dy, dx); c, s = math.cos(-a), math.sin(-a)
    rx = ((x-x[0])*c - (y-y[0])*s)/d; ry = ((x-x[0])*s + (y-y[0])*c)/d
    tg = np.linspace(t[0], t[-1], NP)
    return np.concatenate([np.interp(tg, t, rx), np.interp(tg, t, ry)])   # (2*NP,)

def load(path, cap=8000, shuffle=True):
    lines = open(DATA/path).readlines() if isinstance(path,str) else path
    if shuffle:
        import random; random.Random(0).shuffle(lines)
    S, F = [], []
    for l in lines:
        pts = json.loads(l)["points"]
        sh = canon(pts); d = extract_features(pts)
        if sh is not None and isinstance(d, dict):
            v = np.array([d[k] for k in SHAPE_ONLY], float)
            if np.all(np.isfinite(v)): S.append(sh); F.append(v)
        if len(S) >= cap: break
    return np.array(S), np.array(F)

def residual_feats(S, db_shapes, db_nn, self_db=False):
    """For each shape, residual vs nearest real-DB neighbor -> spectral features."""
    k = 2 if self_db else 1
    dist, idx = db_nn.kneighbors(S, n_neighbors=k)
    out = []
    for i in range(len(S)):
        j = idx[i][k-1]; nn_d = dist[i][k-1]
        res = S[i] - db_shapes[j]
        rx, ry = res[:NP], res[NP:]
        Fx = np.abs(np.fft.rfft(rx)); Fy = np.abs(np.fft.rfft(ry))
        tot = Fx.sum() + Fy.sum() + 1e-9
        lowfreq = (Fx[1:4].sum() + Fy[1:4].sum()) / tot     # smooth-bend band (modes 1-3)
        highfreq = (Fx[8:].sum() + Fy[8:].sum()) / tot      # jerk/tremor band
        # tangential vs perpendicular energy of the residual along the stroke
        tang = np.gradient(S[i][:NP]); perp = np.gradient(S[i][NP:])
        tl = np.hypot(tang, perp) + 1e-9
        rperp = np.abs(res[:NP]*(-perp/tl) + res[NP:]*(tang/tl)).sum()
        rtang = np.abs(res[:NP]*(tang/tl) + res[NP:]*(perp/tl)).sum()
        perp_frac = rperp / (rperp + rtang + 1e-9)
        out.append([lowfreq, highfreq, lowfreq/(highfreq+1e-9), np.linalg.norm(res), nn_d, perp_frac])
    return np.array(out)

def evaluate(Xr_tr, Xr_te, Xb, tag):
    nb = min(len(Xb), len(Xr_tr))
    Xtr = np.vstack([Xr_tr[:nb], Xb[:nb]]); ytr = np.r_[np.zeros(nb), np.ones(nb)]
    clf = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    ng = min(len(Xr_te), len(Xb)-nb) if len(Xb) > nb else min(len(Xr_te), len(Xb))
    Xte = np.vstack([Xr_te[:ng], Xb[nb:nb+ng] if len(Xb)>nb else Xb[:ng]])
    yte = np.r_[np.zeros(ng), np.ones(ng)]
    print(f"  {tag:28s} acc={accuracy_score(yte, clf.predict(Xte)):.3f}", flush=True)

def main():
    bot_file = sys.argv[1]
    Sr, Fr = load("human_movements.jsonl", cap=9000)
    Sb, Fb = load(bot_file, cap=4000)
    # split real: DB (reference) / train / test  (disjoint)
    n = len(Sr); db_n = n//3
    Sdb = Sr[:db_n]
    db_nn = NearestNeighbors().fit(Sdb)
    tr = slice(db_n, db_n + (n-db_n)//2); te = slice(db_n + (n-db_n)//2, n)
    print(f"[det] real DB={db_n} train={Sr[tr].shape[0]} test={Sr[te].shape[0]} bot={len(Sb)}")

    # baseline: shape_only features
    print("BASELINE (shape_only):")
    evaluate(Fr[tr], Fr[te], Fb, "shape_only")
    # upgraded: shape_only + residual-vs-nearest-real features
    Rr_tr = residual_feats(Sr[tr], Sdb, db_nn); Rr_te = residual_feats(Sr[te], Sdb, db_nn)
    Rb = residual_feats(Sb, Sdb, db_nn)
    Ur_tr = np.hstack([Fr[tr], Rr_tr]); Ur_te = np.hstack([Fr[te], Rr_te]); Ub = np.hstack([Fb, Rb])
    print("UPGRADED (shape_only + residual-spectrum):")
    evaluate(Ur_tr, Ur_te, Ub, "shape_only+residual")
    print("RESIDUAL FEATURES ALONE:")
    evaluate(Rr_tr, Rr_te, Rb, "residual_only")

if __name__ == "__main__":
    main()
