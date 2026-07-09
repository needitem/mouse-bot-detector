#!/usr/bin/env python3
"""Synthesis variants on top of the cached sigma-lognormal fits (sl_fits.npz).
Fast to iterate - no refitting.

Modes:
  pure       : sample SL impulse params from a per-K Gaussian, resynthesize
               (reproduces sigma_lognormal_generator.py, the 0.883 baseline).
  residual   : SL macro + a REAL fit-residual (real neuromuscular fine-structure)
               pasted on and rescaled to the synthetic stroke's peak speed. The
               macro is a plausible novel neuromuscular command; the micro-texture
               (jerk/tremor - the detector's #1 tell) is real, not synthesized.
  interp     : interpolate SL params between two real fits (on-manifold blend in
               the neuromuscular parameter space, not position space) + real
               residual. Novel macro that still can't be a near-duplicate.

    python scripts/sl_synth.py residual --n 4000 --out sl_residual_bot_movements.jsonl
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import cholesky

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigma_lognormal_generator import (
    sl_velocity, GRID, MAX_K, MIN_MT_S, MIN_DISTANCE, _to_latent, _from_latent,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CACHE = DATA_DIR / "sl_fits.npz"


def load_cache():
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


def build_k_models(c):
    """Per-K full-covariance Gaussian over the latent impulse params."""
    models, counts = {}, {}
    by_k = {}
    for i in range(len(c["k"])):
        k = int(c["k"][i])
        theta = c["theta"][i][: 6 * k]
        if not np.all(np.isfinite(theta)):
            continue
        by_k.setdefault(k, []).append(_to_latent(theta, k, c["mt"][i], c["dist"][i]))
    for k, rows in by_k.items():
        counts[k] = len(rows)
        if len(rows) < 20:
            continue
        A = np.array(rows)
        cov = np.cov(A.T) + 1e-4 * np.eye(A.shape[1])
        try:
            ch = cholesky(cov)
        except Exception:
            ch = np.diag(np.sqrt(np.diag(cov)))
        models[k] = (A.mean(0), ch)
    tot = sum(counts[k] for k in models)
    return models, {k: counts[k] / tot for k in models}


def integrate(vx, vy, mt_s):
    t = np.linspace(0.0, mt_s, GRID)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * np.diff(t))])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * np.diff(t))])
    return x, y, t * 1000.0


def finalize(x, y, t_ms, dist, rng):
    net = math.hypot(x[-1], y[-1])
    if not np.isfinite(net) or net < 1e-6:
        return None
    s = dist / net
    x, y = x * s, y * s
    a = rng.uniform(0, 2 * math.pi)
    c, sn = math.cos(a), math.sin(a)
    xr, yr = x * c - y * sn, x * sn + y * c
    pts = [[float(px), float(py), float(tt)] for px, py, tt in zip(xr, yr, t_ms)]
    flat = [v for row in pts for v in row]
    return pts if all(np.isfinite(flat)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pure", "residual", "interp"])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="sl_residual_bot_movements.jsonl")
    ap.add_argument("--res_gain", type=float, default=1.0,
                    help="scale on the pasted real residual")
    args = ap.parse_args()

    c = load_cache()
    models, kprob = build_k_models(c)
    print(f"[sl-synth:{args.mode}] cache={len(c['k'])} fits, "
          f"K={ {k: round(v,3) for k,v in kprob.items()} }", flush=True)
    res_pool, peak_pool = c["res"], c["peak"]
    ks = list(kprob); probs = np.array([kprob[k] for k in ks])
    rng = np.random.default_rng(1)

    out, attempts = [], 0
    while len(out) < args.n and attempts < args.n * 6:
        attempts += 1
        k = int(rng.choice(ks, p=probs))
        mean, ch = models[k]
        if args.mode == "interp":
            # two independent draws, blended in latent (neuromuscular) space
            z1 = mean + ch @ rng.standard_normal(mean.shape[0])
            z2 = mean + ch @ rng.standard_normal(mean.shape[0])
            a = rng.uniform(0.25, 0.75)
            z = a * z1 + (1 - a) * z2
        else:
            z = mean + ch @ rng.standard_normal(mean.shape[0])
        theta, mt, dist = _from_latent(z, k)
        mt = max(mt, MIN_MT_S); dist = max(dist, MIN_DISTANCE)
        t = np.linspace(0.0, mt, GRID)
        vx, vy = sl_velocity(t, theta)
        if args.mode in ("residual", "interp"):
            j = rng.integers(len(res_pool))
            peak_syn = float(np.max(np.hypot(vx, vy)) + 1e-9)
            vx = vx + args.res_gain * res_pool[j][:, 0] * peak_syn
            vy = vy + args.res_gain * res_pool[j][:, 1] * peak_syn
        x, y, t_ms = integrate(vx, vy, mt)
        pts = finalize(x, y, t_ms, dist, rng)
        if pts is not None:
            out.append(pts)

    path = DATA_DIR / args.out
    with open(path, "w") as f:
        for pts in out:
            f.write(json.dumps({"points": pts}) + "\n")
    print(f"[sl-synth:{args.mode}] wrote {len(out)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
