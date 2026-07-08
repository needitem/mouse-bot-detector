#!/usr/bin/env python3
"""Evolves the NOISE parameters (OU drift/jump, tremor, SDN, sample timing)
for trajectory_hybrid_ceiling.py's GMM-shape base, instead of reusing
motor_synergy's own evolved noise values.

Why this exists: trajectory_hybrid_ceiling.py's first attempt (GMM smooth
shape + motor_synergy's full noise config) came back catastrophically WORSE
(worst-case 0.828 -> 0.993) than the GMM shape alone. Diagnosis: those noise
parameters were evolved to compensate for motor_synergy's OWN smooth-shape
construction being too regular - the GMM shape, being fit directly on real
trajectories, likely already carries natural-looking variability, so adding
motor_synergy's full-strength noise on top double-counts it. This runs a
proper (mu+lambda) evolutionary search over the noise parameters ONLY (the
GMM shape model is fixed, fit once), reusing the exact same distribution/
variance/covariance/tail-quantile fitness machinery as adversarial_loop.py,
starting the population from near-zero noise (not motor_synergy's evolved
values) since the shape may need little to none.
"""
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __import__("multiprocessing").current_process().name != "MainProcess":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adversarial_loop import (
    FEATURE_NAMES, DIST_MATCH_FEATURES, human_moment_stats, human_corr_matrix, human_tail_stats,
    distribution_penalty, variance_penalty, covariance_penalty, tail_penalty,
    train_detector_ensemble, ensemble_accuracy, ensemble_proba,
    DIST_MATCH_WEIGHT, VARIANCE_MATCH_WEIGHT, COV_MATCH_WEIGHT, TAIL_MATCH_WEIGHT,
)
from features import extract_features
from trajectory_gmm_ceiling import load_human_pool_raw_points
import math

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Structural change (see results/gmm_bot_tell_diagnosis.md): 3 rounds of full
# alternating co-evolution (generator AND detector hyperparameters both
# re-tuned each round) plateaued at ~0.86-0.89 held-out accuracy, and
# permutation_importance found jerk_rms/jerk_max the dominant remaining
# tell. Empirically, no amount of additive noise (OU sigma up to 4x its
# search bound, tremor_amp up to 3x) can close that gap without visibly
# degrading path_efficiency (already near-perfectly matched) - because
# features.py resamples every movement to a 40ms/25Hz grid before computing
# jerk, so noise finer than that gets smoothed away, and noise coarse enough
# to survive necessarily also lengthens the path. The 24-point linearly-
# interpolated shape itself is too smooth at the resolvable scale; doubling
# the control-point density lets natural inter-point curvature vary more
# without needing extra noise to fake it.
N_SHAPE_POINTS = 48
HUMAN_SAMPLE_SIZE = 1200
FINAL_VALIDATION_SAMPLES = 800
MIN_MOVEMENT_TIME = 40.0
# Bumped alongside adding shape_smooth_window (9 -> 10 dims) - cheap to test
# now that this runs fast+parallel on 20 cores.
SAMPLES_PER_CANDIDATE = 150
POP_SIZE = 18
ELITE_K = 5
GENERATIONS_PER_EPOCH = 8
EPOCHS = 5

# Deliberately narrower and lower than motor_synergy's own CONFIG_BOUNDS -
# the working hypothesis (from the failed full-strength-noise hybrid test)
# is that the GMM shape needs much LESS extra noise than motor_synergy's
# own smooth construction did, so the search starts near zero and is only
# allowed to grow into a modest range, not motor_synergy's full range.
NOISE_BOUNDS = {
    "ou_theta": (0.5, 8.0),
    "ou_sigma": (0.0, 4.0),
    # Widened from 0.2: round 3's 48-point evolved config pinned
    # ou_jump_prob near its old upper bound, and local testing found raising
    # it further closes the jerk_rms gap (0.475->0.496 at 0.3, vs human's
    # 0.499) with a much smaller path_efficiency cost than widening
    # tremor_amp for the same jerk gain.
    "ou_jump_prob": (0.0, 0.4),
    "ou_jump_scale": (0.0, 60.0),
    "tremor_amp_min": (0.0, 0.4),
    "tremor_amp_max": (0.0, 0.6),
    "sdn_k": (0.0, 0.08),
    "sample_dt_mean": (5.0, 20.0),
    "gamma_shape": (1.0, 8.0),
    # 3 rounds of alternating co-evolution (see co_evolution_loop.py /
    # results/coevolution_progress.json) plateaued at ~0.86-0.89 against a
    # genuinely re-tuned detector - permutation_importance on that plateau
    # found jerk_rms/jerk_max are the dominant remaining tell (bot ~24-28%
    # lower than human) despite both already being DIST_MATCH_FEATURES
    # targets. OU drift accumulates smoothly (mean-reverting), tremor is a
    # fixed low-frequency sinusoid, and SDN is scaled by speed_est (already
    # matched via sdn_correlation) - none of them are pure, independent,
    # per-sample high-frequency jitter, which is what jerk (a 3rd
    # derivative, extremely sensitive to uncorrelated noise) actually needs.
    # This term adds exactly that, decoupled from every other already-tuned
    # noise axis so raising it shouldn't disturb path_efficiency/curvature.
    "hf_jitter_sigma": (0.0, 2.0),
}
FIXED = {"tremor_freq_min": 8.0, "tremor_freq_max": 12.0}

# A fresh permutation-importance check on the batch-sampling-bug-fixed
# pipeline found path_efficiency is the #1 discriminator (~0.24, vs
# jerk_rms's ~0.08-0.10): human median 0.928 vs bot's ~0.79 - the raw
# GMM-sampled 24-point shape wanders more than real human paths do. A
# moving-average smoothing test confirmed this (window=7 nearly matches
# human path_efficiency) but crushes jerk_rms back down - the same
# smooth-macro-shape-vs-fine-jerk tradeoff this hybrid architecture exists
# to decouple. First made this a SEARCHED dimension - the evolutionary
# search kept converging back to ~1 (no smoothing) regardless, which given
# this project's now-confirmed high run-to-run search variance is hard to
# trust as a real "smoothing doesn't help" signal from a single search run.
# Pulled out as a FIXED (not evolved) env var instead, so it can be A/B
# tested properly across multiple seeds without the search's own noise
# muddying the comparison - see README for the controlled comparison.
FIXED_SMOOTH_WINDOW = float(os.environ.get("FIXED_SMOOTH_WINDOW", "1"))


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


# BIC-based component selection (5/10/15/20/30, always picked k=5) was
# actively working AGAINST this project's actual goal. BIC optimizes for
# "best-generalizing density estimate," penalizing complexity to avoid
# overfitting - the RIGHT criterion for typical density estimation, but this
# isn't that: the actual goal is "fool a downstream classifier," and a highly
# complex GMM (approaching one component per training point, each with a
# small covariance "jitter ball") stops being a smoothed few-cluster
# approximation and starts closely tracking the REAL empirical joint
# distribution's fine structure - exactly the joint structure (path
# efficiency, jerk, covariance across all 14 features) that low-k GMMs were
# losing. Verified directly, INDEPENDENT validation (never-seen-during-fit
# human split), across multiple seeds: k=5 -> worst=0.736, k=200 -> 0.614,
# k=900 -> 0.568, k=1100 -> ~0.52. Monotonically better all the way up to
# just under the training sample count - so "more components" wins outright
# here, not a BIC-style bias-variance tradeoff.
N_GMM_COMPONENTS_FRACTION = 0.98  # of the training sample count


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
    k = max(5, int(len(kept) * N_GMM_COMPONENTS_FRACTION))
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0, max_iter=200, reg_covar=1e-4)
    gmm.fit(Xs)
    print(f"[hybrid-search] GMM: n_components={k} (of {len(kept)} training points), converged={gmm.converged_}")
    return gmm, scaler, kept


def sample_hybrid_trajectory(vec, cfg, py_rng):
    """`vec` must come from a BATCH gmm.sample(n) call, never gmm.sample(1)
    called repeatedly - found the hard way: sklearn's GaussianMixture.sample()
    re-derives its RNG from self.random_state on every call, so with a fixed
    random_state (0, for reproducibility) calling .sample(1) fresh each time
    returns the IDENTICAL sample every single time. Every "different"
    zero-noise movement was actually the same shape (only the random
    direction rotation differed) - rotation-invariant features
    (path_efficiency, curvature_rms, time_to_peak_ratio, ...) came out with
    ~zero movement-to-movement variance (std 0.000-0.03 vs human's
    0.1-0.45), an even easier tell than the CubicSpline/clamping bugs found
    right before this one. Callers must sample all needed vecs via ONE
    gmm.sample(n) call and pass each row in here individually."""
    shape = vec[: N_SHAPE_POINTS * 2].reshape(N_SHAPE_POINTS, 2)
    window = int(round(FIXED_SMOOTH_WINDOW))
    if window % 2 == 0:
        window += 1
    if window > 1:
        kernel = np.ones(window) / window
        pad = window // 2
        sx = np.convolve(np.pad(shape[:, 0], (pad, pad), mode="edge"), kernel, mode="valid")[:N_SHAPE_POINTS]
        sy = np.convolve(np.pad(shape[:, 1], (pad, pad), mode="edge"), kernel, mode="valid")[:N_SHAPE_POINTS]
        shape = np.stack([sx, sy], axis=1)
    distance = max(vec[N_SHAPE_POINTS * 2], 5.0)
    movement_time = max(vec[N_SHAPE_POINTS * 2 + 1], MIN_MOVEMENT_TIME)
    direction = py_rng.uniform(0.0, 2.0 * math.pi)
    c, s = math.cos(direction), math.sin(direction)
    xs, ys = shape[:, 0] * distance, shape[:, 1] * distance
    rot_x = xs * c - ys * s
    rot_y = xs * s + ys * c
    ctrl_t = np.linspace(0.0, movement_time, N_SHAPE_POINTS)
    # Linear interpolation, NOT CubicSpline: cubic splines through only 24
    # sparse control points ring/overshoot between them (natural boundary
    # conditions especially near the endpoints), and evaluating that at a
    # much finer time grid than the GMM was fit on inflated jerk_rms into an
    # obvious artificial tell - the "zero-noise" baseline came back at
    # worst=1.000 (perfectly distinguishable!) with CubicSpline, vs the
    # validated 0.828 from trajectory_gmm_ceiling.py v1, which used exactly
    # this piecewise-linear reconstruction (np.interp) instead.
    spline_x = lambda t: np.interp(t, ctrl_t, rot_x)  # noqa: E731
    spline_y = lambda t: np.interp(t, ctrl_t, rot_y)  # noqa: E731

    tremor_freq = py_rng.uniform(FIXED["tremor_freq_min"], FIXED["tremor_freq_max"])
    tremor_amp = py_rng.uniform(cfg["tremor_amp_min"], cfg["tremor_amp_max"])
    tremor_phase_x = py_rng.uniform(0.0, 2.0 * math.pi)
    tremor_phase_y = py_rng.uniform(0.0, 2.0 * math.pi)

    g_scale = cfg["sample_dt_mean"] / cfg["gamma_shape"]
    times = [0.0]
    t = 0.0
    while t < movement_time and len(times) < 512:
        dt = min(max(py_rng.gammavariate(cfg["gamma_shape"], g_scale), 2.0), 25.0)
        t += dt
        times.append(min(t, movement_time))
    # Real bug found here: the old version let `t` overshoot movement_time by
    # up to 15ms and stored the RAW (unclamped) timestamp while evaluating
    # position at a CLAMPED time - the cursor's position froze at the
    # movement_time-endpoint while the recorded timestamp kept climbing for
    # the last sample or two, an artificial "stopped but time still passing"
    # tell no real trajectory has. Clamping the stored timestamp itself
    # (not just the position lookup) removes it entirely.

    has_jump = py_rng.random() < cfg["ou_jump_prob"]
    jump_idx = py_rng.randrange(1, len(times)) if has_jump and len(times) > 1 else None
    jump_x_val = jump_y_val = 0.0
    if has_jump:
        jump_dir = py_rng.uniform(0.0, 2.0 * math.pi)
        jump_mag = py_rng.gauss(0.0, cfg["ou_jump_scale"])
        jump_x_val, jump_y_val = jump_mag * math.cos(jump_dir), jump_mag * math.sin(jump_dir)

    result = []
    ou_x = ou_y = 0.0
    prev_x = prev_y = None
    for i, ti in enumerate(times):
        ti_c = min(ti, movement_time)
        bx, by = float(spline_x(ti_c)), float(spline_y(ti_c))
        dt_ms = (ti - times[i - 1]) if i > 0 else cfg["sample_dt_mean"]
        dt_s = max(dt_ms, 1.0) / 1000.0
        jump_x_i, jump_y_i = (jump_x_val, jump_y_val) if i == jump_idx else (0.0, 0.0)
        ou_x += -cfg["ou_theta"] * ou_x * dt_s + cfg["ou_sigma"] * math.sqrt(dt_s) * py_rng.gauss(0.0, 1.0) + jump_x_i
        ou_y += -cfg["ou_theta"] * ou_y * dt_s + cfg["ou_sigma"] * math.sqrt(dt_s) * py_rng.gauss(0.0, 1.0) + jump_y_i
        speed_est = (math.hypot(bx - prev_x, by - prev_y) / dt_ms) if prev_x is not None and dt_ms > 0 else 1.0
        prev_x, prev_y = bx, by
        t_s = ti / 1000.0
        trem_mod = 1.0 / (1.0 + speed_est * 0.3)
        tr_x = tremor_amp * trem_mod * math.sin(2.0 * math.pi * tremor_freq * t_s + tremor_phase_x)
        tr_y = tremor_amp * trem_mod * math.sin(2.0 * math.pi * tremor_freq * t_s + tremor_phase_y)
        sdn_x = cfg["sdn_k"] * speed_est * py_rng.gauss(0.0, 1.0)
        sdn_y = cfg["sdn_k"] * speed_est * py_rng.gauss(0.0, 1.0)
        # Pure IID per-sample jitter - independent of speed (unlike SDN) and
        # non-accumulating (unlike OU) - targets jerk_rms/jerk_max
        # specifically without touching the noise axes that already match
        # human statistics (see NOISE_BOUNDS comment).
        hf_x = py_rng.gauss(0.0, cfg["hf_jitter_sigma"])
        hf_y = py_rng.gauss(0.0, cfg["hf_jitter_sigma"])
        result.append((bx + ou_x + tr_x + sdn_x + hf_x, by + ou_y + tr_y + sdn_y + hf_y, ti))
    return result


# The GMM is fit on data where every movement travels at a decent clip (this
# project only keeps the FAST tier of Balabit movements - see
# parse_balabit.py's speed-percentile filter, human implied speed
# distance/movement_time has a 1st percentile of ~404 px/s), but being a
# smooth density the GMM doesn't itself respect that floor and can sample
# distance/movement_time combinations implying near-zero speed (found via
# direct check: ~2.5% of raw samples implied < 100 px/s, some as low as
# ~16 px/s) - mean_speed's permutation importance flagged exactly this (bot's
# 1st percentile mean_speed was 19 px/s vs human's 411). First tried
# filtering on distance alone - that shifted the KEPT sample's mean_speed
# distribution too far the other way (removing short-but-plausible
# movements skews toward the remaining longer/faster ones); filtering
# directly on the implied speed itself is the correct, targeted fix.
# Tried 200 px/s: fixed the near-zero-speed tail (1st percentile 19->309
# px/s) but the independent-validation result got WORSE overall (worst-case
# 0.720->0.833) - filtering on speed post-hoc, even correctly shuffled,
# distorted the (distance, movement_time) joint relationship enough to
# inflate mean_speed's UPPER tail instead (99th percentile 2504->6364
# px/s). Tried a gentler 50 px/s threshold too - same problem, barely
# changed. Disabled (0 = no filtering): mean_speed's near-zero tail is a
# real, smaller gap, but "fix" it with rejection sampling made the whole
# picture worse both times tried - not every diagnosed gap has a
# rejection-sampling-shaped fix. Revisit with something that respects the
# joint density (e.g. importance reweighting) rather than post-hoc filtering
# if this is worth chasing further.
MIN_SAMPLE_SPEED = 0.0


def _sample_valid_vecs(gmm, scaler, n, seed=None):
    """CRITICAL: gmm.sample(k) returns samples GROUPED BY COMPONENT (sklearn's
    own documented behavior), not shuffled - trajectory_gmm_ceiling.py's
    correct sample_bot_rows always shuffled after sampling for exactly this
    reason. Missing that shuffle here caused a real, silent bias: filtering
    a block-ordered batch down to the first `n` valid rows can (and did)
    consume the entire quota from just the first few components before ever
    reaching the rest of the mixture, if those components alone pass the
    speed filter often enough - systematically dropping whatever the later
    components represent. Confirmed by outcome: filtering without shuffling
    made mean_speed AND path_efficiency both get WORSE, not better."""
    rng = np.random.default_rng(seed)
    vecs = []
    attempts = 0
    while sum(len(v) for v in vecs) < n and attempts < 10:
        batch = scaler.inverse_transform(gmm.sample(max(n * 2, 50))[0])
        distance_col = batch[:, N_SHAPE_POINTS * 2]
        mtime_col = np.clip(batch[:, N_SHAPE_POINTS * 2 + 1], MIN_MOVEMENT_TIME, None)
        implied_speed = distance_col / mtime_col * 1000.0
        valid = batch[implied_speed >= MIN_SAMPLE_SPEED]
        vecs.append(valid)
        attempts += 1
    if not vecs:
        return np.empty((0, N_SHAPE_POINTS * 2 + 2))
    pool = np.concatenate(vecs, axis=0)
    return pool[rng.permutation(len(pool))][:n]


def sample_candidate_movements(gmm, scaler, cfg_overrides, n, py_rng):
    cfg = {**{k: (NOISE_BOUNDS[k][0] + NOISE_BOUNDS[k][1]) / 2 for k in NOISE_BOUNDS}, **cfg_overrides}
    vecs = _sample_valid_vecs(gmm, scaler, n)
    rows = []
    for vec in vecs:
        pts = sample_hybrid_trajectory(vec, cfg, py_rng)
        if len(pts) >= 4:
            rows.append(extract_features(pts))
    return rows


def clip_bounds(cfg):
    return {k: float(np.clip(v, *NOISE_BOUNDS[k])) for k, v in cfg.items()}


def zero_ish_config():
    return {
        "ou_theta": 4.0, "ou_sigma": 0.0, "ou_jump_prob": 0.0, "ou_jump_scale": 0.0,
        "tremor_amp_min": 0.0, "tremor_amp_max": 0.0, "sdn_k": 0.0,
        "sample_dt_mean": 10.0, "gamma_shape": 4.0, "hf_jitter_sigma": 0.0,
    }


def random_config(rng):
    return clip_bounds({k: rng.uniform(lo, hi) for k, (lo, hi) in NOISE_BOUNDS.items()})


def mutate(cfg, rng, scale):
    child = {}
    for k, v in cfg.items():
        lo, hi = NOISE_BOUNDS[k]
        child[k] = v + rng.gauss(0.0, scale * (hi - lo))
    return clip_bounds(child)


def fitness_of(models, rows, human_stats, human_corr, human_tail):
    if not rows:
        return 1.0 + DIST_MATCH_WEIGHT * 4.0 + VARIANCE_MATCH_WEIGHT * 4.0 + COV_MATCH_WEIGHT * 1.0 + TAIL_MATCH_WEIGHT * 4.0
    detector_term = float(np.mean(ensemble_proba(models, rows)))
    dist_term = distribution_penalty(rows, human_stats)
    var_term = variance_penalty(rows, human_stats)
    cov_term = covariance_penalty(rows, human_corr)
    tail_term = tail_penalty(rows, human_tail)
    return (
        detector_term + DIST_MATCH_WEIGHT * dist_term + VARIANCE_MATCH_WEIGHT * var_term
        + COV_MATCH_WEIGHT * cov_term + TAIL_MATCH_WEIGHT * tail_term
    )


# Same fork-deadlock/BLAS-oversubscription lessons as adversarial_loop.py:
# spawn context (mandatory on Windows anyway, and safe on Linux too), one
# persistent pool for the whole run, worker processes limited to 1 BLAS
# thread each (see the os.environ block near the top of this file).
N_WORKERS = max(1, min(int(os.environ.get("MAX_WORKERS", "15")), (os.cpu_count() or 4) - 1))
_MP_CTX = __import__("multiprocessing").get_context("spawn")
_worker_state = {}


def _init_worker(gmm, scaler, human_stats, human_corr, human_tail):
    _worker_state["gmm"] = gmm
    _worker_state["scaler"] = scaler
    _worker_state["human_stats"] = human_stats
    _worker_state["human_corr"] = human_corr
    _worker_state["human_tail"] = human_tail


def _score_candidate_worker(args):
    cfg, seed, n, ensemble = args
    py_rng = random.Random(seed)
    vecs = _sample_valid_vecs(_worker_state["gmm"], _worker_state["scaler"], n)
    rows = []
    for vec in vecs:
        pts = sample_hybrid_trajectory(vec, cfg, py_rng)
        if len(pts) >= 4:
            rows.append(extract_features(pts))
    fit = fitness_of(
        ensemble, rows, _worker_state["human_stats"], _worker_state["human_corr"], _worker_state["human_tail"]
    )
    return fit, rows


def score_population(pool, population, rng, n, ensemble):
    tasks = [(cfg, rng.randrange(2**31), n, ensemble) for cfg in population]
    results = list(pool.map(_score_candidate_worker, tasks))
    return [(fit, cfg, rows) for (fit, rows), cfg in zip(results, population)]


def main():
    print("[hybrid-search] loading human pool + fitting shape GMM...")
    pool_points = load_human_pool_raw_points(seed=0)
    a = HUMAN_SAMPLE_SIZE
    b = a + FINAL_VALIDATION_SAMPLES
    train_points, final_points = pool_points[:a], pool_points[a:b]
    gmm, scaler, kept_train = fit_shape_gmm(train_points)
    human_rows = [extract_features(pts) for pts in kept_train]
    human_stats = human_moment_stats(human_rows)
    human_corr = human_corr_matrix(human_rows)
    human_tail = human_tail_stats(human_rows)

    # Same lesson as adversarial_loop.py's SEARCH_SEED: this evolutionary
    # search shows real run-to-run variance in the FINAL VALIDATION result
    # (0.720-0.856 worst-case seen across otherwise-identical runs) - a
    # single run isn't reliable evidence for/against any one hypothesis.
    search_seed = int(os.environ.get("SEARCH_SEED", "42"))
    print(f"[hybrid-search] search_seed={search_seed}")
    rng = random.Random(search_seed)
    print("[hybrid-search] generating baseline (zero-noise) + default-noise samples for initial ensemble...")
    zero_rows = sample_candidate_movements(gmm, scaler, zero_ish_config(), HUMAN_SAMPLE_SIZE, rng)
    ensemble = train_detector_ensemble(human_rows, zero_rows, seed_base=0)
    baseline_acc = ensemble_accuracy(ensemble, human_rows, zero_rows)
    baseline_acc_worst = ensemble_accuracy(ensemble, human_rows, zero_rows, reduce="worst")
    print(f"[hybrid-search] zero-noise baseline: acc mean={baseline_acc:.3f} worst={baseline_acc_worst:.3f}")

    population = [zero_ish_config()] + [random_config(rng) for _ in range(POP_SIZE - 1)]
    best_cfg_overall, best_fitness_overall = zero_ish_config(), float("inf")

    print(f"[hybrid-search] parallel pool: {N_WORKERS} workers")
    pool = ProcessPoolExecutor(
        max_workers=N_WORKERS, mp_context=_MP_CTX, initializer=_init_worker,
        initargs=(gmm, scaler, human_stats, human_corr, human_tail),
    )
    for epoch in range(EPOCHS):
        mutation_scale = max(0.3 * (0.85 ** epoch), 0.05)
        for gen in range(GENERATIONS_PER_EPOCH):
            scored = score_population(pool, population, rng, SAMPLES_PER_CANDIDATE, ensemble)
            scored.sort(key=lambda item: item[0])
            best_fit, best_cfg, best_rows = scored[0]
            acc = ensemble_accuracy(ensemble, human_rows[:len(best_rows)], best_rows)
            print(f"[hybrid-search] epoch {epoch} gen {gen}: fitness={best_fit:.3f} ensemble_acc={acc:.3f}", flush=True)
            if best_fit < best_fitness_overall:
                best_fitness_overall, best_cfg_overall = best_fit, best_cfg
            elite = [c for _, c, _ in scored[:ELITE_K]]
            children = []
            while len(elite) + len(children) < POP_SIZE:
                children.append(mutate(rng.choice(elite), rng, mutation_scale))
            population = elite + children

        best_rows = sample_candidate_movements(gmm, scaler, best_cfg_overall, SAMPLES_PER_CANDIDATE, rng)
        ensemble = train_detector_ensemble(human_rows, best_rows, seed_base=(epoch + 1) * 100)
    pool.shutdown(wait=True)

    print(f"[hybrid-search] best noise config found (fitness={best_fitness_overall:.3f}):")
    print(json.dumps(best_cfg_overall, indent=2))

    print("[hybrid-search] independent final validation...")
    final_human_rows = [extract_features(pts) for pts in final_points]
    default_final = sample_candidate_movements(gmm, scaler, zero_ish_config(), FINAL_VALIDATION_SAMPLES, random.Random(1))
    evolved_final = sample_candidate_movements(gmm, scaler, best_cfg_overall, FINAL_VALIDATION_SAMPLES, random.Random(2))
    half = len(final_human_rows) // 2
    fresh_default = train_detector_ensemble(final_human_rows[:half], default_final[:len(default_final)//2], seed_base=9000)
    acc_default = ensemble_accuracy(fresh_default, final_human_rows[half:], default_final[len(default_final)//2:])
    acc_default_worst = ensemble_accuracy(fresh_default, final_human_rows[half:], default_final[len(default_final)//2:], reduce="worst")
    fresh_evolved = train_detector_ensemble(final_human_rows[:half], evolved_final[:len(evolved_final)//2], seed_base=9100)
    acc_evolved = ensemble_accuracy(fresh_evolved, final_human_rows[half:], evolved_final[len(evolved_final)//2:])
    acc_evolved_worst = ensemble_accuracy(fresh_evolved, final_human_rows[half:], evolved_final[len(evolved_final)//2:], reduce="worst")
    print(f"[hybrid-search] final validation - zero-noise: mean={acc_default:.3f} worst={acc_default_worst:.3f}")
    print(f"[hybrid-search] final validation - evolved-noise: mean={acc_evolved:.3f} worst={acc_evolved_worst:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "hybrid_noise_evolved_config.json").write_text(json.dumps(best_cfg_overall, indent=2))
    report = {
        "zero_noise_acc_mean": acc_default, "zero_noise_acc_worst": acc_default_worst,
        "evolved_noise_acc_mean": acc_evolved, "evolved_noise_acc_worst": acc_evolved_worst,
    }
    (RESULTS_DIR / "hybrid_noise_search_report.json").write_text(json.dumps(report, indent=2))
    print(f"[hybrid-search] wrote results")


if __name__ == "__main__":
    main()
