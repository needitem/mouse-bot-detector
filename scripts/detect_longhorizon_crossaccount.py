#!/usr/bin/env python3
"""Defender endgame: long-horizon reuse + cross-account linking.

The mag-0 no-repeat attacker (attack_norepeat.py) evades every within-session
statistic as long as the session stays under the pool size K. Two aggregate
signals close that off, because the attacker's recorded pool is finite:

  1. LONG-HORIZON: observe one account past K flicks -> reuse is forced ->
     near-duplicates appear. Detected once N > K.
  2. CROSS-ACCOUNT: even if each account stays under K (evading per-account),
     rotating accounts that share one pool produces the SAME stroke across
     accounts -> pooling accounts reveals cross-account near-duplicates that
     genuinely different players never produce.

Both use the same canonical near-duplicate threshold eps from
detect_replay_reuse.py. Run: python -u scripts/detect_longhorizon_crossaccount.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_noise_search import to_canonical_at, N_SHAPE_POINTS
from trajectory_gmm_ceiling import load_human_pool_raw_points
from sklearn.neighbors import NearestNeighbors

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EPS = 0.0594


def main():
    rng = np.random.default_rng(0)
    pool = load_human_pool_raw_points(seed=0)
    sh = []
    for p in pool:
        c = to_canonical_at(p, N_SHAPE_POINTS)
        if c is not None:
            sh.append(c[0].ravel())
    sh = np.asarray(sh)
    K = 2000
    atk = sh[:K]
    realpool = sh[K:]
    print(f"[endgame] attacker pool K={K}, real pool={len(realpool)}, eps={EPS}", flush=True)

    def has_dup(V):
        d, _ = NearestNeighbors(n_neighbors=2).fit(V).kneighbors(V)
        return d[:, 1].min() < EPS

    def dup_rate(V):
        d, _ = NearestNeighbors(n_neighbors=2).fit(V).kneighbors(V)
        return (d[:, 1] < EPS).mean()

    # 1) long-horizon: one account, no-repeat until pool exhausts
    print("\n[endgame] LONG-HORIZON (one account, pure replay no-repeat):", flush=True)
    print("        N (obs) | N/K | reuse-detected", flush=True)
    lh = []
    for N in [500, 1000, 2000, 3000, 5000, 8000]:
        hits, trials = 0, 30
        for _ in range(trials):
            if N <= K:
                src = rng.choice(K, N, replace=False)
            else:
                src = np.concatenate([rng.permutation(K), rng.integers(0, K, N - K)])
            hits += has_dup(atk[src])
        lh.append({"N": N, "rate": hits / trials})
        print(f"        {N:>7} | {N/K:>3.1f} | {hits/trials:.2f}", flush=True)

    # 2) cross-account: each account stays under K (evades alone), shared pool
    print("\n[endgame] CROSS-ACCOUNT (rotate accounts, shared pool, N=200/account):", flush=True)
    print("        M accts | total | attacker cross-dup | real cross-dup", flush=True)
    N = 200
    ca = []
    for M in [2, 5, 10, 15, 20]:
        a = r = 0.0
        trials = 20
        for _ in range(trials):
            V = np.vstack([atk[rng.choice(K, N, replace=False)] for _ in range(M)])
            a += dup_rate(V)
            Vr = realpool[rng.choice(len(realpool), M * N, replace=False)]
            r += dup_rate(Vr)
        ca.append({"M": M, "attacker": a / trials, "real": r / trials})
        print(f"        {M:>7} | {M*N:>5} | {a/trials:>18.3f} | {r/trials:.3f}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "longhorizon_crossaccount.json").write_text(json.dumps(
        {"K": K, "eps": EPS, "long_horizon": lh, "cross_account": ca}, indent=2))
    print(f"\n[endgame] wrote {RESULTS_DIR / 'longhorizon_crossaccount.json'}", flush=True)


if __name__ == "__main__":
    main()
