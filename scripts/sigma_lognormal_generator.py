#!/usr/bin/env python3
"""Sigma-lognormal (Kinematic Theory of Rapid Movements, Plamondon) trajectory
generator - the one mechanistic, non-data-driven escalation this project
scope-cut at the start.

The premise (see README "The core finding"): every DATA-DRIVEN generator - high-k
GMM, RealNVP flow, DDPM diffusion, even DMTG on 1M trajectories - plateaus at
~0.85 against a strong detector, because learning a distribution from finite data
leaves a smoothed off-manifold gap the detector exploits. A MECHANISTIC model
escapes that trap by construction: it doesn't LEARN the human-motion manifold, it
IMPLEMENTS the neuromuscular process that generates it. Finite data is then only
used to CALIBRATE a handful of physically-meaningful parameters, not to estimate
the whole distribution - so the generalization gap that caps every learned model
mostly goes away.

Model. A rapid aimed movement's velocity is the vector sum of K lognormal
impulse responses of the neuromuscular system (Plamondon's Kinematic Theory):

    v(t) = sum_i  D_i * Lambda(t; t0_i, mu_i, sig_i) * [cos phi_i(t), sin phi_i(t)]
    Lambda(t) = lognormal PDF with onset t0, log-time-delay mu, log-response sig
    phi_i(t)  = theta_s_i + (theta_e_i - theta_s_i) * LognormalCDF(t; t0_i,mu_i,sig_i)

Each stroke = 1 primary impulse + 0..2 corrective impulses. We FIT this to each
real human stroke (nonlinear least squares on the velocity profile), learn the
joint distribution of the fitted impulse parameters across the pool, sample new
parameter sets, and re-synthesize (x, y, t). Because ANY parameter draw yields a
physically-valid neuromuscular trajectory, the samples stay on-manifold (unlike
blend/perturb) while being unlimited-diverse (unlike replay).

Output schema matches every other generator (data/processed/*_movements.jsonl,
one {"points": [[x,y,t],...]} per line), so validate_flow_bot_strong_detector.py
scores it identically:

    python -u scripts/sigma_lognormal_generator.py --n 4000
    python -u scripts/validate_flow_bot_strong_detector.py sigma_lognormal_bot_movements.jsonl
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_gmm_ceiling import load_human_pool_raw_points

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_PATH = DATA_DIR / "sigma_lognormal_bot_movements.jsonl"

# Fit/synthesis grid. Time is handled in SECONDS internally (nicer conditioning
# for the lognormal), converted back to ms on output.
GRID = 64                 # uniform-time samples used for fitting the velocity profile
MIN_DISTANCE = 5.0
MIN_MT_S = 0.040          # 40 ms, matches MIN_MOVEMENT_TIME elsewhere
MAX_K = 3                 # primary + up to 2 corrections
EPS = 1e-9


# ---------------------------------------------------------------------------
# Sigma-lognormal primitives
# ---------------------------------------------------------------------------
def _lognormal_pdf(t, t0, mu, sig):
    """Lambda(t): lognormal PDF (integrates to 1), zero for t <= t0."""
    out = np.zeros_like(t)
    m = t > t0 + EPS
    tt = t[m] - t0
    out[m] = (1.0 / (sig * math.sqrt(2 * math.pi) * tt)) * np.exp(
        -((np.log(tt) - mu) ** 2) / (2 * sig * sig)
    )
    return out


def _lognormal_cdf(t, t0, mu, sig):
    """Phi(t): fraction of the impulse delivered by time t, in [0,1]."""
    out = np.zeros_like(t)
    m = t > t0 + EPS
    tt = t[m] - t0
    out[m] = 0.5 * (1.0 + erf((np.log(tt) - mu) / (sig * math.sqrt(2))))
    return out


def sl_velocity(t, theta):
    """Velocity (vx, vy) of a K-component sigma-lognormal. theta is a flat array
    of K*6 params: [D, t0, mu, sig, ths, the] per component."""
    vx = np.zeros_like(t)
    vy = np.zeros_like(t)
    for i in range(0, len(theta), 6):
        D, t0, mu, sig, ths, the = theta[i : i + 6]
        lam = _lognormal_pdf(t, t0, mu, sig)
        phi = ths + (the - ths) * _lognormal_cdf(t, t0, mu, sig)
        vx += D * lam * np.cos(phi)
        vy += D * lam * np.sin(phi)
    return vx, vy


def sl_position(theta, mt_s, n_out):
    """Integrate the SL velocity to a position trajectory on n_out points over
    [0, mt_s] seconds. Returns (x, y, t_ms)."""
    t = np.linspace(0.0, mt_s, n_out)
    vx, vy = sl_velocity(t, theta)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * np.diff(t))])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * np.diff(t))])
    return x, y, t * 1000.0


# ---------------------------------------------------------------------------
# Fitting a real stroke
# ---------------------------------------------------------------------------
def _resample_velocity(points):
    """Real stroke -> uniform-time velocity profile (vx, vy) in units/s on GRID
    points, plus movement_time (s). Returns None if degenerate."""
    pts = np.asarray(points, dtype=float)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2] / 1000.0  # t -> seconds
    mt = t[-1] - t[0]
    dist = math.hypot(x[-1] - x[0], y[-1] - y[0])
    if mt < MIN_MT_S or dist < MIN_DISTANCE:
        return None
    tg = np.linspace(t[0], t[-1], GRID)
    xg = np.interp(tg, t, x)
    yg = np.interp(tg, t, y)
    dt = (tg[-1] - tg[0]) / (GRID - 1)
    vx = np.gradient(xg, dt)
    vy = np.gradient(yg, dt)
    return vx, vy, mt, dist


def _init_theta(vx, vy, mt, k):
    """Physically-motivated initialization: place k impulses at the k biggest
    speed peaks, size each from the local speed, aim each along the local
    velocity direction."""
    t = np.linspace(0.0, mt, GRID)
    speed = np.hypot(vx, vy)
    # candidate peak times: interior local maxima, else evenly spaced
    peaks = [i for i in range(1, GRID - 1) if speed[i] >= speed[i - 1] and speed[i] > speed[i + 1]]
    peaks.sort(key=lambda i: speed[i], reverse=True)
    if len(peaks) < k:
        extra = list(np.linspace(GRID * 0.25, GRID * 0.75, k).astype(int))
        peaks = (peaks + extra)[:k]
    peaks = sorted(peaks[:k])
    total_disp = np.array([np.trapz(vx, t), np.trapz(vy, t)])
    theta = []
    for j, pk in enumerate(peaks):
        tp = max(t[pk], 1e-3)
        sig = 0.3
        t0 = max(0.0, tp * 0.4)                    # onset before the peak
        mu = math.log(max(tp - t0, 1e-3)) + sig * sig  # mode of lognormal lands at tp
        D = max(np.hypot(*total_disp) / k, MIN_DISTANCE)
        ang = math.atan2(vy[pk], vx[pk]) if speed[pk] > EPS else math.atan2(total_disp[1], total_disp[0])
        theta += [D, t0, mu, sig, ang, ang]
    return np.array(theta, dtype=float)


def _bounds(mt, k):
    lo, hi = [], []
    for _ in range(k):
        lo += [1.0, 0.0, math.log(1e-3), 0.05, -2 * math.pi, -2 * math.pi]
        hi += [1e5, mt, math.log(mt + 0.5), 1.2, 2 * math.pi, 2 * math.pi]
    return np.array(lo), np.array(hi)


def fit_stroke(points):
    """Fit sigma-lognormal to one real stroke; try K=1..MAX_K, keep the best by
    AIC-like residual+penalty. Returns (theta, k, mt_s, dist) or None."""
    rv = _resample_velocity(points)
    if rv is None:
        return None
    vx, vy, mt, dist = rv
    t = np.linspace(0.0, mt, GRID)
    target = np.concatenate([vx, vy])

    def make_resid(k):
        def resid(theta):
            mx, my = sl_velocity(t, theta)
            return np.concatenate([mx, my]) - target
        return resid

    best = None
    for k in range(1, MAX_K + 1):
        theta0 = _init_theta(vx, vy, mt, k)
        lo, hi = _bounds(mt, k)
        theta0 = np.clip(theta0, lo + EPS, hi - EPS)
        try:
            sol = least_squares(make_resid(k), theta0, bounds=(lo, hi),
                                max_nfev=400 * k, method="trf")
        except Exception:
            continue
        rss = float(np.sum(sol.fun ** 2))
        aic = 2 * len(sol.x) + GRID * 2 * math.log(rss / (GRID * 2) + EPS)
        if best is None or aic < best[0]:
            best = (aic, sol.x, k, rss)
    if best is None:
        return None
    _, theta, k, _ = best
    return theta, k, mt, dist


# ---------------------------------------------------------------------------
# Learning the impulse-parameter distribution + sampling
# ---------------------------------------------------------------------------
def _to_latent(theta, k, mt, dist):
    """Map a fit to an unconstrained latent vector for distribution modeling.
    Positives -> log; angles kept raw; times normalized by mt. Layout per
    component: [logD, t0/mt, mu, log sig, ths, the]; then [log mt, log dist]."""
    z = []
    for i in range(0, 6 * k, 6):
        D, t0, mu, sig, ths, the = theta[i : i + 6]
        z += [math.log(max(D, EPS)), t0 / mt, mu, math.log(max(sig, 1e-3)), ths, the]
    z += [math.log(max(mt, EPS)), math.log(max(dist, EPS))]
    return np.array(z, dtype=float)


def _from_latent(z, k):
    mt = math.exp(z[-2])
    dist = math.exp(z[-1])
    theta = []
    for j in range(k):
        logD, t0n, mu, logsig, ths, the = z[6 * j : 6 * j + 6]
        theta += [math.exp(logD), max(0.0, t0n) * mt, mu, math.exp(logsig), ths, the]
    return np.array(theta, dtype=float), mt, dist


def build_models(pool, max_fit, seed=0):
    """Fit the whole pool, group by K, fit a full-covariance Gaussian per K in
    latent space. Returns {k: (mean, chol, count)} and the empirical K prob."""
    from numpy.linalg import cholesky
    latents = {k: [] for k in range(1, MAX_K + 1)}
    n_fit = 0
    for pts in pool:
        if n_fit >= max_fit:
            break
        r = fit_stroke(pts)
        n_fit += 1
        if r is None:
            continue
        theta, k, mt, dist = r
        latents[k].append(_to_latent(theta, k, mt, dist))
    models = {}
    counts = {}
    for k, rows in latents.items():
        counts[k] = len(rows)
        if len(rows) < 20:
            continue
        A = np.array(rows)
        mean = A.mean(0)
        cov = np.cov(A.T) + 1e-4 * np.eye(A.shape[1])  # ridge for PD
        try:
            chol = cholesky(cov)
        except Exception:
            chol = np.diag(np.sqrt(np.diag(cov)))
        models[k] = (mean, chol)
    total = sum(counts[k] for k in models)
    kprob = {k: counts[k] / total for k in models}
    return models, kprob, n_fit


def sample_movements(models, kprob, n, seed=1):
    rng = np.random.default_rng(seed)
    ks = list(kprob)
    probs = np.array([kprob[k] for k in ks])
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 5:
        attempts += 1
        k = int(rng.choice(ks, p=probs))
        mean, chol = models[k]
        z = mean + chol @ rng.standard_normal(mean.shape[0])
        theta, mt, dist = _from_latent(z, k)
        mt = max(mt, MIN_MT_S)
        dist = max(dist, MIN_DISTANCE)
        x, y, t_ms = sl_position(theta, mt, GRID)
        # rescale so net displacement magnitude equals the sampled distance, then
        # rotate to a uniformly random direction (same convention as the GMM/flow
        # un-canonicalization: shape is learned, absolute direction is not).
        net = math.hypot(x[-1], y[-1])
        if not np.isfinite(net) or net < EPS:
            continue
        s = dist / net
        x, y = x * s, y * s
        ang = rng.uniform(0, 2 * math.pi)
        c, sn = math.cos(ang), math.sin(ang)
        xr = x * c - y * sn
        yr = x * sn + y * c
        pts = [[float(a), float(b), float(tt)] for a, b, tt in zip(xr, yr, t_ms)]
        if all(np.isfinite([p for row in pts for p in row])):
            out.append(pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="movements to generate")
    ap.add_argument("--max_fit", type=int, default=3000, help="human strokes to fit")
    args = ap.parse_args()

    print("[sl] loading human pool...", flush=True)
    pool = load_human_pool_raw_points(seed=0)
    print(f"[sl] fitting sigma-lognormal to up to {args.max_fit} strokes "
          f"(K=1..{MAX_K})...", flush=True)
    models, kprob, n_fit = build_models(pool, args.max_fit)
    print(f"[sl] fitted {n_fit} strokes; K distribution: "
          f"{ {k: round(v,3) for k,v in kprob.items()} }", flush=True)
    for k, (mean, _) in models.items():
        print(f"    K={k}: {mean.shape[0]}-dim latent model", flush=True)

    print(f"[sl] sampling {args.n} synthetic movements...", flush=True)
    movements = sample_movements(models, kprob, args.n)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for pts in movements:
            f.write(json.dumps({"points": pts}) + "\n")
    print(f"[sl] wrote {len(movements)} movements to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
