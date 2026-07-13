#!/usr/bin/env python3
"""Session-distribution gate for elastic warped replay. The session detector
(detect_replay_cluster.py) catches a finite-pool attacker because its flicks
cluster around K sources -> the within-session nearest-neighbor DISTANCE
DISTRIBUTION differs from a real player's all-independent strokes. It caught the
old canonical-space humanvar perturbation. Question: does the raw-space smooth
elastic bend also leave that cluster trace, or does it spread samples enough to
pass? Runs both attackers vs real, sweeping session length N. Positive control:
humanvar SHOULD be caught (detector works); the elastic column is the answer.
"""
import argparse, json, math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

NP = 48
def canon_shape(pts):
    p = np.asarray(pts, float); x, y, t = p[:,0], p[:,1], p[:,2]
    dx, dy = x[-1]-x[0], y[-1]-y[0]; d = math.hypot(dx, dy)
    if d < 5 or t[-1]-t[0] < 40: return None
    a = math.atan2(dy, dx); c, s = math.cos(-a), math.sin(-a)
    rx = (x-x[0])*c - (y-y[0])*s; ry = (x-x[0])*s + (y-y[0])*c
    tg = np.linspace(t[0], t[-1], NP)
    return np.concatenate([np.interp(tg, t, rx/d), np.interp(tg, t, ry/d)])

def elastic_deform(pts, amp, modes, rng):
    x, y, t = pts[:,0].copy(), pts[:,1].copy(), pts[:,2]
    x -= x[0]; y -= y[0]; dist = math.hypot(x[-1], y[-1]) + 1e-9
    K = len(pts); u = np.linspace(0, 1, K); disp = np.zeros(K)
    for k in range(1, modes+1):
        disp += rng.normal(0, amp*dist/k) * np.sin(k*math.pi*u)
    dx = np.gradient(x); dy = np.gradient(y); seg = np.hypot(dx, dy)+1e-9
    x = x + disp*(-dy/seg); y = y + disp*(dx/seg)
    return np.stack([x, y, t-t[0]], 1)

def humanvar_shape(shapes, mag, rng):
    i, a, b = rng.integers(len(shapes)), rng.integers(len(shapes)), rng.integers(len(shapes))
    return shapes[i] + mag*(shapes[a]-shapes[b])

def session_features(V):
    nn = NearestNeighbors(n_neighbors=2).fit(V); d,_ = nn.kneighbors(V)
    nnd = d[:,1]
    pcts = np.percentile(nnd, [1,5,10,25,50,75,90])
    bins = [np.mean(nnd < th) for th in (0.03,0.06,0.1,0.15,0.2)]
    return np.concatenate([pcts, bins, [nnd.mean(), nnd.std()]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--K", type=int, default=6000)
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--mag", type=float, default=0.07)
    ap.add_argument("--sess", type=int, default=120)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    raw = []
    for l in open(args.data):
        p = np.asarray(json.loads(l)["points"], float)
        if len(p) >= 4 and math.hypot(p[-1,0]-p[0,0], p[-1,1]-p[0,1]) >= 5 and p[-1,2]-p[0,2] >= 40:
            raw.append(p)
    shapes = np.array([s for s in (canon_shape(p) for p in raw) if s is not None])
    pool_idx = rng.choice(len(raw), args.K, replace=False)
    pool_raw = [raw[i] for i in pool_idx]
    pool_shapes = shapes[pool_idx]
    print(f"[session] {len(raw)} real strokes, pool K={args.K}, elastic amp={args.amp}, humanvar mag={args.mag}")

    def real_session(N):
        idx = rng.choice(len(shapes), N, replace=False)
        return shapes[idx]
    def elastic_session(N):
        idx = rng.integers(0, args.K, N)          # finite pool, reuse allowed
        out = []
        for i in idx:
            sh = canon_shape(elastic_deform(pool_raw[i], args.amp, args.modes, rng))
            if sh is not None: out.append(sh)
        return np.array(out)
    def humanvar_session(N):
        return np.array([humanvar_shape(pool_shapes, args.mag, rng) for _ in range(N)])

    print(f"\n{'N':>6} | {'elastic_acc':>11} | {'humanvar_acc(ctrl)':>18}")
    for N in [50, 100, 200, 500, 1000]:
        for label, gen in [("elastic", elastic_session), ("humanvar", humanvar_session)]:
            X, y = [], []
            for _ in range(args.sess):
                X.append(session_features(gen(N))); y.append(1)
                X.append(session_features(real_session(N))); y.append(0)
            acc = cross_val_score(RandomForestClassifier(n_estimators=120, random_state=0),
                                  np.array(X), np.array(y), cv=5).mean()
            if label == "elastic": ea = acc
            else: ha = acc
        print(f"{N:>6} | {ea:>11.3f} | {ha:>18.3f}", flush=True)

if __name__ == "__main__":
    main()
