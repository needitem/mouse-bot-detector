#!/usr/bin/env python3
"""Fit sigma-lognormal to the human pool ONCE, in parallel, and cache the fits
+ residuals so synthesis variants (pure SL, SL+real-residual, param
interpolation, ...) can be iterated without the ~25-min refit.

Cache (data/processed/sl_fits.npz):
  theta   : (N, 6*MAX_K) padded impulse params (NaN past k)
  k       : (N,) components used
  mt      : (N,) movement time (s)
  dist    : (N,) net displacement (px)
  res     : (N, GRID, 2) fit residual (real - SL) velocity, normalized by the
            stroke's PEAK speed -> dimensionless, transferable across strokes
  peak    : (N,) peak speed (px/s) of the real stroke (to rescale residuals)
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigma_lognormal_generator import (
    fit_stroke, _resample_velocity, sl_velocity, GRID, MAX_K,
)
from trajectory_gmm_ceiling import load_human_pool_raw_points

OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "sl_fits.npz"


def _one(points):
    r = fit_stroke(points)
    if r is None:
        return None
    theta, k, mt, dist = r
    rv = _resample_velocity(points)
    if rv is None:
        return None
    vx, vy, _, _ = rv
    t = np.linspace(0.0, mt, GRID)
    mx, my = sl_velocity(t, theta)
    peak = float(np.max(np.hypot(vx, vy)) + 1e-9)
    res = np.stack([(vx - mx) / peak, (vy - my) / peak], axis=1)  # (GRID,2)
    tpad = np.full(6 * MAX_K, np.nan)
    tpad[: len(theta)] = theta
    return tpad, k, mt, dist, res, peak


def main():
    n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    pool = load_human_pool_raw_points(seed=0)[:n_fit]
    print(f"[sl-fit] fitting {len(pool)} strokes across workers...", flush=True)
    theta, ks, mt, dist, res, peak = [], [], [], [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=10) as ex:
        for out in ex.map(_one, pool, chunksize=8):
            done += 1
            if done % 300 == 0:
                print(f"  {done}/{len(pool)} (kept {len(ks)})", flush=True)
            if out is None:
                continue
            tp, k, m, d, r, pk = out
            theta.append(tp); ks.append(k); mt.append(m); dist.append(d)
            res.append(r); peak.append(pk)
    np.savez_compressed(
        OUT, theta=np.array(theta), k=np.array(ks), mt=np.array(mt),
        dist=np.array(dist), res=np.array(res), peak=np.array(peak),
    )
    print(f"[sl-fit] cached {len(ks)} fits to {OUT}", flush=True)


if __name__ == "__main__":
    main()
