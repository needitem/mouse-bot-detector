#!/usr/bin/env python3
"""Hybrid generator: a data-driven GMM for the SMOOTH macro-shape of a
movement (learned directly from real human trajectories - captures whatever
joint distance/duration/curvature/timing structure motor_synergy's formula
gets slightly wrong) plus motor_synergy's already-VALIDATED noise injection
(OU-jump, tremor, SDN) for the fine-timescale detail the smooth shape can't
carry on its own.

Motivation: trajectory_gmm_ceiling.py tried two representations -
24 fixed points (v1: worst=0.828, decent) and 64 points + PCA (v2:
worst=0.889, WORSE) - increasing resolution and reducing dimensionality via
PCA made things worse, not better. Diagnosis: PCA finds the directions of
maximum POSITION variance, which is dominated by the smooth bulk path shape;
it discards exactly the small-amplitude, high-frequency jitter/tremor detail
that jerk_rms (this project's #1 permutation-importance discriminator all
session) depends on - so PCA-reconstructed trajectories come out too smooth,
an easy tell.

Rather than trying to make one representation capture BOTH the smooth macro
shape AND the fine noise, this splits the problem the way motor_synergy
already does internally: sample the smooth path from wherever it's best
learned (data, via GMM, at a modest 24-point resolution - already validated
as reasonable in v1) then evaluate it on a FINE raw-sample-rate grid (via
spline interpolation) and add the SAME noise terms motor_synergy_generate
uses (with its own evolved parameter values - see NOISE_CFG below), instead
of motor_synergy's own lognormal-submovement construction for the smooth
part.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adversarial_loop import train_detector_ensemble, ensemble_accuracy, FEATURE_NAMES
from features import extract_features
from trajectory_gmm_ceiling import load_human_pool_raw_points

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

N_SHAPE_POINTS = 24  # v1's resolution for the smooth-shape GMM - kept low deliberately
HUMAN_SAMPLE_SIZE = 1200
FINAL_VALIDATION_SAMPLES = 800
MIN_MOVEMENT_TIME = 40.0

# The evolved_motor_synergy_config.json noise-relevant values in place when
# this was written (a recent seed-sweep result) - reused as-is rather than
# re-tuned, since the point of this test is "do PROVEN noise terms rescue a
# GMM shape," not to re-run the whole evolutionary search on this new base.
NOISE_CFG = json.loads((RESULTS_DIR / "evolved_motor_synergy_config.json").read_text())


def fit_shape_gmm(train_points):
    vecs, kept = [], []
    for pts in train_points:
        c = to_canonical_at(pts, N_SHAPE_POINTS)
        if c is None:
            continue
        shape_xy, distance, movement_time = c
        vecs.append(np.concatenate([shape_xy.ravel(), [distance, movement_time]]))
        kept.append(pts)
    X = np.array(vecs)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    best_bic, best_gmm = None, None
    for k in (5, 10, 15, 20, 30):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0, max_iter=300, reg_covar=1e-5)
        gmm.fit(Xs)
        bic = gmm.bic(Xs)
        print(f"[hybrid]   n_components={k}: BIC={bic:.1f}")
        if best_bic is None or bic < best_bic:
            best_bic, best_gmm = bic, gmm
    return best_gmm, scaler, kept


def to_canonical_at(points, n_points):
    pts = np.asarray(points, dtype=float)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    x0, y0 = x[0], y[0]
    dx, dy = x[-1] - x0, y[-1] - y0
    distance = math.hypot(dx, dy)
    movement_time = t[-1] - t[0]
    if distance < 5.0 or movement_time < MIN_MOVEMENT_TIME:
        return None
    direction = math.atan2(dy, dx)
    c, s = math.cos(-direction), math.sin(-direction)
    rel_x, rel_y = x - x0, y - y0
    rot_x = (rel_x * c - rel_y * s) / distance
    rot_y = (rel_x * s + rel_y * c) / distance
    t_grid = np.linspace(t[0], t[-1], n_points)
    shape_x = np.interp(t_grid, t, rot_x)
    shape_y = np.interp(t_grid, t, rot_y)
    return np.stack([shape_x, shape_y], axis=1), distance, movement_time


def sample_hybrid_trajectory(vec, py_rng):
    """`vec` must come from a BATCH gmm.sample(n) call - see
    hybrid_noise_search.py's sample_hybrid_trajectory docstring: sklearn's
    GaussianMixture.sample() re-derives its RNG from self.random_state on
    every call, so calling .sample(1) repeatedly with a fixed random_state
    returns the IDENTICAL sample every time - collapses movement-to-movement
    variance to ~0 (an even bigger tell than the CubicSpline bug this file
    was originally fixed for)."""
    shape = vec[: N_SHAPE_POINTS * 2].reshape(N_SHAPE_POINTS, 2)
    distance = max(vec[N_SHAPE_POINTS * 2], 5.0)
    movement_time = max(vec[N_SHAPE_POINTS * 2 + 1], MIN_MOVEMENT_TIME)
    direction = py_rng.uniform(0.0, 2.0 * math.pi)
    c, s = math.cos(direction), math.sin(direction)
    xs, ys = shape[:, 0] * distance, shape[:, 1] * distance
    rot_x = xs * c - ys * s
    rot_y = xs * s + ys * c
    ctrl_t = np.linspace(0.0, movement_time, N_SHAPE_POINTS)
    # Linear interpolation, NOT CubicSpline: a cubic spline through only 24
    # sparse control points rings/overshoots between them when evaluated at
    # a much finer grid than it was fit on - found via hybrid_noise_search.py
    # (which used the same CubicSpline approach) producing a "zero-noise"
    # baseline that was perfectly distinguishable (worst=1.000), when the
    # validated GMM-shape-alone result (trajectory_gmm_ceiling.py v1, same
    # 24-point shape, piecewise-linear reconstruction) was 0.828. This
    # script's original 0.993 "catastrophic" result was likely inflated by
    # this same artifact, not purely noise miscalibration as first assumed.
    spline_x = lambda t: np.interp(t, ctrl_t, rot_x)  # noqa: E731
    spline_y = lambda t: np.interp(t, ctrl_t, rot_y)  # noqa: E731

    cfg = NOISE_CFG
    tx, ty = math.cos(direction), math.sin(direction)
    nx, ny = -ty, tx

    tremor_freq = py_rng.uniform(cfg["tremor_freq_min"], cfg["tremor_freq_max"])
    tremor_amp = py_rng.uniform(cfg["tremor_amp_min"], cfg["tremor_amp_max"])
    tremor_phase_x = py_rng.uniform(0.0, 2.0 * math.pi)
    tremor_phase_y = py_rng.uniform(0.0, 2.0 * math.pi)

    g_scale = cfg["sample_dt_mean"] / cfg["gamma_shape"]
    times = [0.0]
    t = 0.0
    while t < movement_time and len(times) < 512:
        dt = min(max(py_rng.gammavariate(cfg["gamma_shape"], g_scale), 2.0), 25.0)
        t += dt
        # Clamp the stored timestamp itself, not just the position lookup
        # below - the old version let it overshoot movement_time by up to
        # 15ms while position was clamped, freezing the cursor for the last
        # sample(s) while the timestamp kept climbing (see
        # hybrid_noise_search.py for the same bug/fix).
        times.append(min(t, movement_time))

    has_jump = py_rng.random() < cfg["ou_jump_prob"]
    jump_idx = py_rng.randrange(1, len(times)) if has_jump and len(times) > 1 else None
    jump_x_val = jump_y_val = 0.0
    if has_jump:
        jump_dir = py_rng.uniform(0.0, 2.0 * math.pi)
        jump_mag = py_rng.gauss(0.0, cfg["ou_jump_scale"])
        jump_x_val, jump_y_val = jump_mag * math.cos(jump_dir), jump_mag * math.sin(jump_dir)

    result = []
    ou_x = ou_y = 0.0
    for i, ti in enumerate(times):
        ti_c = min(ti, movement_time)
        bx = spline_x(ti_c)
        by = spline_y(ti_c)
        dt_ms = (ti - times[i - 1]) if i > 0 else cfg["sample_dt_mean"]
        dt_s = dt_ms / 1000.0
        jump_x_i, jump_y_i = (jump_x_val, jump_y_val) if i == jump_idx else (0.0, 0.0)
        ou_x += -cfg["ou_theta"] * ou_x * dt_s + cfg["ou_sigma"] * math.sqrt(dt_s) * py_rng.gauss(0.0, 1.0) + jump_x_i
        ou_y += -cfg["ou_theta"] * ou_y * dt_s + cfg["ou_sigma"] * math.sqrt(dt_s) * py_rng.gauss(0.0, 1.0) + jump_y_i
        t_s = ti / 1000.0
        speed_est = 1.0  # tremor modulation only needs an order-of-magnitude speed proxy
        trem_mod = 1.0 / (1.0 + speed_est * 0.3)
        tr_x = tremor_amp * trem_mod * math.sin(2.0 * math.pi * tremor_freq * t_s + tremor_phase_x)
        tr_y = tremor_amp * trem_mod * math.sin(2.0 * math.pi * tremor_freq * t_s + tremor_phase_y)
        sdn_x = cfg["sdn_k"] * 3.0 * py_rng.gauss(0.0, 1.0)
        sdn_y = cfg["sdn_k"] * 3.0 * py_rng.gauss(0.0, 1.0)
        result.append((bx + ou_x + tr_x + sdn_x, by + ou_y + tr_y + sdn_y, ti))
    return result


def main():
    print("[hybrid] loading human pool (raw points)...")
    pool_points = load_human_pool_raw_points(seed=0)
    a = HUMAN_SAMPLE_SIZE
    b = a + FINAL_VALIDATION_SAMPLES
    train_points = pool_points[:a]
    final_points = pool_points[a:b]

    print(f"[hybrid] fitting shape GMM ({N_SHAPE_POINTS} control points, no PCA)...")
    gmm, scaler, kept_train = fit_shape_gmm(train_points)
    print(f"[hybrid] {len(kept_train)}/{len(train_points)} movements kept")

    human_rows = [extract_features(pts) for pts in kept_train]

    def sample_bot_rows(n, seed):
        py_rng = __import__("random").Random(seed)
        vecs = scaler.inverse_transform(gmm.sample(n)[0])  # ONE batch call - see sample_hybrid_trajectory docstring
        rows = []
        for vec in vecs:
            pts = sample_hybrid_trajectory(vec, py_rng)
            if len(pts) >= 4:
                rows.append(extract_features(pts))
        return rows

    print(f"[hybrid] sampling {HUMAN_SAMPLE_SIZE} hybrid trajectories...")
    bot_rows = sample_bot_rows(HUMAN_SAMPLE_SIZE, seed=1)

    print("[hybrid] training initial ensemble...")
    ensemble = train_detector_ensemble(human_rows, bot_rows, seed_base=0)
    acc = ensemble_accuracy(ensemble, human_rows, bot_rows)
    acc_worst = ensemble_accuracy(ensemble, human_rows, bot_rows, reduce="worst")
    print(f"[hybrid] same-split ensemble accuracy: mean={acc:.3f} worst={acc_worst:.3f}")

    print("[hybrid] independent final validation...")
    final_human_rows = [extract_features(pts) for pts in final_points]
    final_bot_rows = sample_bot_rows(FINAL_VALIDATION_SAMPLES, seed=2)
    half = len(final_human_rows) // 2
    fresh_ensemble = train_detector_ensemble(
        final_human_rows[:half], final_bot_rows[: len(final_bot_rows) // 2], seed_base=9000
    )
    acc_final = ensemble_accuracy(fresh_ensemble, final_human_rows[half:], final_bot_rows[len(final_bot_rows) // 2:])
    acc_final_worst = ensemble_accuracy(
        fresh_ensemble, final_human_rows[half:], final_bot_rows[len(final_bot_rows) // 2:], reduce="worst"
    )
    print(f"[hybrid] independent final validation accuracy: mean={acc_final:.3f} worst={acc_final_worst:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "n_shape_points": N_SHAPE_POINTS,
        "same_split_acc_mean": acc,
        "same_split_acc_worst": acc_worst,
        "final_validation_acc_mean": acc_final,
        "final_validation_acc_worst": acc_final_worst,
    }
    (RESULTS_DIR / "trajectory_hybrid_ceiling.json").write_text(json.dumps(report, indent=2))
    print(f"[hybrid] wrote {RESULTS_DIR / 'trajectory_hybrid_ceiling.json'}")


if __name__ == "__main__":
    main()
