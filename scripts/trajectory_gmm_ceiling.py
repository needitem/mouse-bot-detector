#!/usr/bin/env python3
"""Trajectory-space generator: learns the actual (x, y, t) shape of human
movements directly from data (a fixed-length, canonical-frame resampling +
Gaussian Mixture Model), instead of motor_synergy's hand-derived lognormal-
submovement formula. Produces genuine synthetic trajectories - not just
feature vectors like feature_space_ceiling.py - so this is a real answer to
"how much better could an unconstrained-form generator do," not just an
idealized upper bound.

Representation: each movement is rotated/scaled into a canonical frame
(start at origin, end at (1, 0)) and resampled to a FIXED number of points
on a duration-fraction grid (distinct from features.py's COMMON_DT_MS grid,
which is for feature extraction on the raw/absolute time axis - this one is
purely for making movements the same fixed dimensionality for the generative
model). The per-movement vector is
[x_1..x_N, y_1..y_N, distance, movement_time] (canonical shape + the two
scalars needed to place it back in real space) - a GMM with full covariance
fit on this jointly captures shape-distance-duration correlations (e.g.
Fitts' law: longer distance -> longer duration -> different curvature/
timing), the same kind of joint structure the adversarial_loop.py fitness
function has been chasing with hand-added covariance/tail-quantile terms.

Sampling: draw a vector from the fitted GMM, un-normalize (scale canonical
x/y by the sampled distance), rotate to a random direction, assign absolute
timestamps via linspace(0, sampled movement_time, N), then run the SAME
extract_features() used everywhere else in this project - direct,
apples-to-apples comparison against motor_synergy's evolved config and
feature_space_ceiling.py's idealized ceiling.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adversarial_loop import train_detector_ensemble, ensemble_accuracy, FEATURE_NAMES
from features import extract_features

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


def load_human_pool_raw_points(seed=0):
    """All valid human movements' raw (x, y, t) point lists, shuffled once -
    same shuffle/slicing convention as adversarial_loop.load_human_pool, but
    keeping the raw points instead of pre-extracting features (this script
    needs the actual trajectory shape, not just the feature summary)."""
    import random
    path = DATA_DIR / "human_movements.jsonl"
    with open(path) as f:
        lines = f.readlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    points_list = []
    for line in lines:
        rec = json.loads(line)
        pts = rec["points"]
        if len(pts) >= 4:
            points_list.append(pts)
    return points_list

# v1 used 24 points (48+2=50 dims) and only got worst-case to 0.828 - barely
# better than motor_synergy's 0.854, well short of feature_space_ceiling's
# idealized 0.694. Suspected cause: 24 points is too coarse to preserve the
# fine-timescale jerk/tremor detail that jerk_rms (this whole project's #1
# permutation-importance discriminator, repeatedly) actually depends on.
# Raised to 64 points for much finer shape fidelity - but a 64*2+2=130-dim
# vector with only 1200 training samples makes a full-covariance GMM in the
# RAW space badly underdetermined (a single component's covariance matrix
# alone needs ~8500 parameters). Fixed via PCA: fit the GMM in a reduced
# ~20-dim PCA space (standard "point distribution model" approach for
# exactly this shape-modeling problem), inverse-transform back to full
# resolution when sampling.
N_POINTS = 64
PCA_COMPONENTS = 20
HUMAN_SAMPLE_SIZE = 1200
FINAL_VALIDATION_SAMPLES = 800
MIN_MOVEMENT_TIME = 40.0


def to_canonical(points):
    """Rotate+scale a movement so it starts at origin and ends at (1, 0),
    resampled to N_POINTS on a duration-fraction grid. Returns
    (shape_xy: (N_POINTS, 2) array, distance, movement_time) or None if the
    movement is degenerate (near-zero distance/duration)."""
    pts = np.asarray(points, dtype=float)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    x0, y0 = x[0], y[0]
    dx, dy = x[-1] - x0, y[-1] - y0
    distance = math.hypot(dx, dy)
    movement_time = t[-1] - t[0]
    if distance < 5.0 or movement_time < MIN_MOVEMENT_TIME:
        return None
    direction = math.atan2(dy, dx)
    cx, sx = math.cos(-direction), math.sin(-direction)
    rel_x, rel_y = x - x0, y - y0
    rot_x = (rel_x * cx - rel_y * sx) / distance
    rot_y = (rel_x * sx + rel_y * cx) / distance
    t_grid = np.linspace(t[0], t[-1], N_POINTS)
    shape_x = np.interp(t_grid, t, rot_x)
    shape_y = np.interp(t_grid, t, rot_y)
    return np.stack([shape_x, shape_y], axis=1), distance, movement_time


def vectorize(points_list):
    vecs, kept = [], []
    for pts in points_list:
        c = to_canonical(pts)
        if c is None:
            continue
        shape_xy, distance, movement_time = c
        vecs.append(np.concatenate([shape_xy.ravel(), [distance, movement_time]]))
        kept.append(pts)
    return np.array(vecs), kept


def unvectorize_and_extract(vec, rng):
    shape = vec[: N_POINTS * 2].reshape(N_POINTS, 2)
    distance, movement_time = vec[N_POINTS * 2], vec[N_POINTS * 2 + 1]
    distance = max(distance, 5.0)
    movement_time = max(movement_time, MIN_MOVEMENT_TIME)
    direction = rng.uniform(0.0, 2.0 * math.pi)
    c, s = math.cos(direction), math.sin(direction)
    xs = shape[:, 0] * distance
    ys = shape[:, 1] * distance
    rot_x = xs * c - ys * s
    rot_y = xs * s + ys * c
    t = np.linspace(0.0, movement_time, N_POINTS)
    points = list(zip(rot_x.tolist(), rot_y.tolist(), t.tolist()))
    return extract_features(points)


def main():
    print("[traj-gmm] loading human pool (raw points)...")
    pool_points = load_human_pool_raw_points(seed=0)
    a = HUMAN_SAMPLE_SIZE
    b = a + FINAL_VALIDATION_SAMPLES
    train_points = pool_points[:a]
    final_points = pool_points[a:b]

    print(f"[traj-gmm] canonicalizing + resampling to {N_POINTS} points...")
    X_train, kept_train = vectorize(train_points)
    print(f"[traj-gmm] {len(X_train)}/{len(train_points)} movements kept (rest too short/degenerate)")

    print("[traj-gmm] computing human feature rows (for the detector's human class)...")
    human_rows = [extract_features(pts) for pts in kept_train]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=0)
    Xp = pca.fit_transform(Xs)
    print(
        f"[traj-gmm] PCA to {PCA_COMPONENTS} components explains "
        f"{pca.explained_variance_ratio_.sum():.1%} of variance"
    )

    print("[traj-gmm] fitting GMMs in PCA space (selecting n_components by BIC)...")
    best_bic, best_k, best_gmm = None, None, None
    for k in (5, 10, 15, 20, 30):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0, max_iter=300, reg_covar=1e-5)
        gmm.fit(Xp)
        bic = gmm.bic(Xp)
        print(f"[traj-gmm]   n_components={k}: BIC={bic:.1f}")
        if best_bic is None or bic < best_bic:
            best_bic, best_k, best_gmm = bic, k, gmm
    print(f"[traj-gmm] selected n_components={best_k}")

    def sample_bot_rows(n, seed):
        rng = np.random.default_rng(seed)
        py_rng = __import__("random").Random(seed)
        Xg, _ = best_gmm.sample(n)
        Xg = Xg[rng.permutation(n)]
        Xg = scaler.inverse_transform(pca.inverse_transform(Xg))
        return [unvectorize_and_extract(v, py_rng) for v in Xg]

    print(f"[traj-gmm] sampling {HUMAN_SAMPLE_SIZE} synthetic trajectories from the fitted GMM...")
    bot_rows = sample_bot_rows(HUMAN_SAMPLE_SIZE, seed=1)

    print("[traj-gmm] training initial ensemble (human vs GMM-sampled trajectories)...")
    ensemble = train_detector_ensemble(human_rows, bot_rows, seed_base=0)
    acc = ensemble_accuracy(ensemble, human_rows, bot_rows)
    acc_worst = ensemble_accuracy(ensemble, human_rows, bot_rows, reduce="worst")
    print(f"[traj-gmm] same-split ensemble accuracy: mean={acc:.3f} worst={acc_worst:.3f}")

    print("[traj-gmm] independent final validation (fresh ensemble, held-out human split)...")
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
    print(f"[traj-gmm] independent final validation accuracy: mean={acc_final:.3f} worst={acc_final_worst:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "n_components": best_k,
        "n_points": N_POINTS,
        "same_split_acc_mean": acc,
        "same_split_acc_worst": acc_worst,
        "final_validation_acc_mean": acc_final,
        "final_validation_acc_worst": acc_final_worst,
    }
    (RESULTS_DIR / "trajectory_gmm_ceiling.json").write_text(json.dumps(report, indent=2))
    print(f"[traj-gmm] wrote {RESULTS_DIR / 'trajectory_gmm_ceiling.json'}")


if __name__ == "__main__":
    main()
