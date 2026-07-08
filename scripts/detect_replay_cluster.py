#!/usr/bin/env python3
"""Defender improvement: catch the mag~0.07 human-variability attacker that
evades both the single-movement and the min-distance near-duplicate detector.

attack_sweet_spot.py found mag~0.07 gives single-move ~0.54 and reuse-detect
~0.00 - it escapes both. But near-duplicate detection only looked at the
MINIMUM pairwise distance in a session. The attacker still draws from a finite
pool of K sources, so its flicks cluster around those sources (each source +
an independent small perturbation), whereas a real player's strokes are all
independent. That shows up in the whole nearest-neighbor DISTANCE DISTRIBUTION
of a session, not just its minimum.

This builds a session-level classifier: featurize each session by the
distribution of within-session nearest-neighbor distances (percentiles + how
many fall in low-distance bins), train real-session vs attacker-session, and
sweep session length N to see how many flicks the defender must observe to
catch the mag~0.07 attacker (K=6000). Run: python -u scripts/detect_replay_cluster.py
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
MAG = 0.07          # attacker's human-variability perturbation (the sweet spot)
SESS_PER_CLASS = 150


def session_features(V):
    """V: (N, dim) session shapes -> distribution features of within-session
    nearest-neighbor distances."""
    nn = NearestNeighbors(n_neighbors=2).fit(V)
    d, _ = nn.kneighbors(V)
    nnd = d[:, 1]                      # each flick's nearest OTHER flick in the session
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
    print(f"[cluster] {len(shapes)} canonical shapes; K={POOL_K}, attacker mag={MAG}", flush=True)

    def humanvar(mag):
        a, b = rng.integers(0, len(shapes), size=2)
        return mag * (shapes[a] - shapes[b])

    def attacker_session(N):
        src = rng.integers(0, POOL_K, size=N)
        return np.array([shapes[si] + humanvar(MAG) for si in src])

    def real_session(N):
        pick = rng.choice(len(shapes), size=N, replace=False)
        return shapes[pick]

    print("\n[cluster]   N | session-detector accuracy (5-fold) | vs old near-dup(min)", flush=True)
    print("        " + "-" * 60, flush=True)
    out = []
    for N in [50, 100, 250, 500, 1000]:
        X, y = [], []
        for _ in range(SESS_PER_CLASS):
            X.append(session_features(attacker_session(N))); y.append(1)
            X.append(session_features(real_session(N))); y.append(0)
        X = np.asarray(X); y = np.asarray(y)
        clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        acc = cross_val_score(clf, X, y, cv=5).mean()
        out.append({"N": N, "session_acc": float(acc)})
        print(f"        {N:>5} | {acc:>34.3f} | (min-dist detector was evaded)", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "replay_cluster_detection.json").write_text(json.dumps(
        {"pool_K": POOL_K, "attacker_mag": MAG, "sweep": out}, indent=2))
    print(f"\n[cluster] wrote {RESULTS_DIR / 'replay_cluster_detection.json'}", flush=True)


if __name__ == "__main__":
    main()
