#!/usr/bin/env python3
"""Attacker counter: don't reuse sources.

The session-distribution detector (detect_replay_cluster.py) catches the
mag~0.07 attacker via the CLUSTER structure that comes from drawing N flicks
from K sources WITH replacement - repeats bunch up. The counter is trivial:
draw sources WITHOUT replacement (each flick a distinct source) as long as the
pool is at least as large as the session, so there are no repeats and no
clusters. This tests whether that evades the session-distribution detector,
and also checks the one signal that survives no-repeat: if the DEFENDER holds
the attacker's (public) stroke pool, every attack flick sits ~mag from some
pool stroke (a nearest-DB distance a genuinely new player wouldn't have).

Run: python -u scripts/attack_norepeat.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_noise_search import to_canonical_at, N_SHAPE_POINTS
from trajectory_gmm_ceiling import load_human_pool_raw_points
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
POOL_K = 6000
MAG = 0.07
SESS_PER_CLASS = 150


def session_features(V):
    nn = NearestNeighbors(n_neighbors=2).fit(V)
    d, _ = nn.kneighbors(V)
    nnd = d[:, 1]
    qs = np.percentile(nnd, [1, 5, 10, 25, 50])
    frac = [(nnd < thr).mean() for thr in (0.10, 0.15, 0.20, 0.30)]
    return list(qs) + frac + [nnd.mean(), nnd.std()]


def main():
    rng = np.random.default_rng(0)
    pool = load_human_pool_raw_points(seed=0)
    shapes = []
    for pts in pool:
        c = to_canonical_at(pts, N_SHAPE_POINTS)
        if c is not None:
            shapes.append(c[0].ravel())
    shapes = np.asarray(shapes)
    # split: attacker's pool (public), defender-held for nearest-DB test, and a
    # disjoint "genuinely new players" set to stand in for real sessions.
    attacker_pool = shapes[:POOL_K]
    real_new = shapes[POOL_K:]
    print(f"[norepeat] attacker pool K={len(attacker_pool)}, real-new={len(real_new)}, mag={MAG}", flush=True)

    def humanvar():
        a, b = rng.integers(0, len(attacker_pool), size=2)
        return MAG * (attacker_pool[a] - attacker_pool[b])

    def attack_session(N, replace):
        if replace:
            src = rng.integers(0, len(attacker_pool), size=N)
        else:
            src = rng.choice(len(attacker_pool), size=min(N, len(attacker_pool)), replace=False)
        return np.array([attacker_pool[si] + humanvar() for si in src])

    def real_session(N):
        pick = rng.choice(len(real_new), size=N, replace=False)
        return real_new[pick]

    # --- Test 1: session-distribution detector, with vs without source reuse ---
    print("\n[norepeat] session-distribution detector accuracy (attacker evades if ~0.5):", flush=True)
    print("        N | with-reuse | NO-reuse (attacker's counter)", flush=True)
    print("        " + "-" * 52, flush=True)
    t1 = []
    for N in [100, 250, 500, 1000]:
        accs = {}
        for tag, rep in [("reuse", True), ("norepeat", False)]:
            X, y = [], []
            for _ in range(SESS_PER_CLASS):
                X.append(session_features(attack_session(N, rep))); y.append(1)
                X.append(session_features(real_session(N))); y.append(0)
            clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
            accs[tag] = cross_val_score(clf, np.asarray(X), np.asarray(y), cv=5).mean()
        t1.append({"N": N, **accs})
        print(f"     {N:>5} | {accs['reuse']:>10.3f} | {accs['norepeat']:>10.3f}", flush=True)

    # --- Test 2: nearest-DB detector (defender HOLDS the attacker's public pool) ---
    # each flick sits ~mag from some pool stroke; a real new player does not.
    print("\n[norepeat] nearest-DB detector (defender has the attacker's public pool):", flush=True)
    nn_db = NearestNeighbors(n_neighbors=1).fit(attacker_pool)
    atk = np.array([attacker_pool[i] + humanvar() for i in rng.integers(0, len(attacker_pool), 2000)])
    d_atk, _ = nn_db.kneighbors(atk)
    d_real, _ = nn_db.kneighbors(real_new[rng.choice(len(real_new), 2000, replace=False)])
    print(f"        attack flick nearest-DB dist: median={np.median(d_atk):.3f}", flush=True)
    print(f"        real   flick nearest-DB dist: median={np.median(d_real):.3f}", flush=True)
    thr = float(np.percentile(d_atk, 95))
    acc_db = 0.5 * ((d_atk[:, 0] <= thr).mean() + (d_real[:, 0] > thr).mean())
    print(f"        per-flick nearest-DB detector balanced-acc: {acc_db:.3f}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "attack_norepeat.json").write_text(json.dumps(
        {"session_detector": t1,
         "nearest_db_acc": acc_db,
         "atk_db_median": float(np.median(d_atk)),
         "real_db_median": float(np.median(d_real))}, indent=2))
    print(f"\n[norepeat] wrote {RESULTS_DIR / 'attack_norepeat.json'}", flush=True)


if __name__ == "__main__":
    main()
