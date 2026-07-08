#!/usr/bin/env python3
"""Adversarial loop: evolves needaimbot's motor_synergy config against the
trained detector's own feedback, instead of hand-tuning it - a genuine
generator-vs-discriminator dynamic, just with a classical-ML discriminator
and a gradient-free (evolutionary) generator update instead of backprop
(motor_synergy's generation involves discrete stochastic sampling, so it
isn't end-to-end differentiable the way a neural GAN generator would be).

Uses the `shape_only` feature set (see train_detector.py's "Known confound"),
not `all`: sample_interval_mean/cv deliberately measure raw (pre-resample)
capture cadence - a capture-pipeline artifact no amount of tuning
motor_synergy's config can legitimately fix, so optimizing against them would
just chase an unwinnable, irrelevant signal.

REVISION HISTORY (see README.md "Adversarial loop" for the full story):
- v1 (single detector, retrained each epoch): search reliably drove whichever
  detector instance it saw to chance level, but the "win" never survived a
  fresh, independently-retrained detector - the classic adversarial-example-
  doesn't-transfer pattern. Replicated at 3 different detector strengths.
- v2 (4-model detector ensemble + a light 3-feature distribution-matching
  term): first real, if modest, independently-validated improvement
  (accuracy 0.998->0.991, distribution distance 0.205->0.183).
- v3 (this version): distribution matching is now the DOMINANT term (higher
  weight, 9 features instead of 3, covering speed/curvature/jerk/efficiency/
  skew/timing jointly instead of just 3 axes), and the whole thing runs as a
  proper loop-until-converged: keep adding epochs, checking the CURRENT best
  config's distribution distance to human against a cheap (no training
  needed) per-epoch check, stopping when it's close enough (CONVERGENCE_Z2)
  or has plateaued for PATIENCE epochs running (MAX_EPOCHS is a safety cap,
  not the real stopping condition).

- v4 (mean + variance matching, after fixing a real data bug - see README.md
  "v4"): the first genuine, repeatable improvement (accuracy ~0.989->~0.959
  mean / ~0.985->~0.944 worst), via marginal per-feature mean AND variance
  matching, plus decoupling mean/spread control in generate_synthetic.py
  where a single Gaussian scale factor was conflating the two. Three
  iterations of "fix the worst axis" each found the next-worst axis had the
  identical bug, with shrinking returns each time (curvature: real gain;
  timing: no net gain even with a bigger search budget) - a whack-a-mole
  plateau consistent with approaching this generative family's real ceiling
  for MARGINAL matching, not a search-power problem.
- v5 (this version): matching marginal mean+variance per feature says
  nothing about how features relate to EACH OTHER - added a covariance/
  correlation-matching term so the search is pushed toward the same joint
  structure as human data (e.g. does high curvature co-occur with high jerk
  the same way in both), not just the same 9 independent marginals.

Two-level loop, per epoch:
- Within an epoch, a small population of candidate configs is evaluated
  against the SAME fixed detector ensemble across a few generations (cheap:
  only feature extraction + predict_proba, no retraining) and evolved via a
  simple (mu+lambda) strategy - elitism + Gaussian mutation, with the
  mutation scale annealing both within an epoch and slowly across epochs.
- Between epochs, the whole ensemble is retrained (fresh bootstrap resamples)
  on human vs the epoch's best candidate's output, and the running best
  config is checked against the (cheap, training-free) distribution-distance
  convergence criterion.

At the end, an INDEPENDENT fresh ensemble (never seen during the search) is
trained to compare default vs evolved config - both mean accuracy AND
worst-case (max over the ensemble) accuracy are reported, since "genuinely
generalizes" should mean fooling essentially all of the fresh ensemble, not
just its average - plus a per-feature distribution breakdown.

Outputs: results/adversarial_history.png (fitness/accuracy/convergence per
round), results/evolved_motor_synergy_config.json, results/adversarial_report.md.
"""
import json
import math
import multiprocessing
import os
import random

# MUST be set before numpy/sklearn are imported (BLAS reads these once at
# import/first-use time) - and ONLY in worker processes, not the main one.
# Without this, each of the N_WORKERS worker PROCESSES also lets numpy/
# OpenBLAS spin up its own multi-threaded pool internally - 11 processes x
# ~12 BLAS threads each = ~130 threads fighting over 12 real cores, thrashing
# so badly that "parallel" scoring measured SLOWER than sequential (10.5s/
# round parallel vs 11.8s/round sequential for 18 candidates - barely any
# speedup despite 11 worker processes). Each worker here only ever needs ONE
# thread: motor_synergy_generate is pure Python, and per-candidate
# parallelism (across processes) is already how this workload is
# parallelized. The MAIN process is excluded because it's the one that calls
# train_detector_ensemble, which genuinely benefits from BLAS/sklearn's own
# multi-threading (that step isn't parallelized across processes).
if multiprocessing.current_process().name != "MainProcess":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features import extract_features  # noqa: E402
from train_detector import SHAPE_ONLY_FEATURES  # noqa: E402
from generate_synthetic import (  # noqa: E402
    MOTOR_SYNERGY_DEFAULTS,
    motor_synergy_generate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

FEATURE_NAMES = SHAPE_ONLY_FEATURES

# Parameters actually evolved (the behaviorally meaningful subset - the ones
# our detector's feature importances point at: curvature/OU drift/tremor/SDN
# control jerk & curvature_rms, timing params control submovement/Fitts
# shape). Bounds keep values physically sensible around the hand-tuned
# defaults; most allow shrinking toward 0 since the initial report found
# motor_synergy_bot's jerk/curvature running HIGHER than human.
CONFIG_BOUNDS = {
    # curvature_scale/ou_sigma widened past their original hand-tuned range
    # (0.0-0.05 / 0.0-3.0): the v3 run's per-feature breakdown showed
    # curvature_rms stuck at z=-0.78 vs human (~10x too low) while every
    # other axis was already within 0.1 std - the original ceiling was the
    # bottleneck, not a local optimum the search failed to find.
    "curvature_scale": (0.0, 0.3),
    "ou_theta": (0.5, 8.0),
    # ou_sigma pinned at the old ceiling (8.0) repeatedly (v5/v6/v7: 8.0,
    # 7.65, 7.85) so it was re-tested wider (0, 16) TWICE now - both times
    # (once under the 9-feature fitness function, once under the 13-feature
    # one) the wider bound let the search wander to a worse, more extreme
    # corner (ou_sigma pinned at the NEW ceiling 16.0 the second time, paired
    # with several other params also at their extremes - gamma_shape=8.0,
    # sample_dt_mean=3.0 floor, tremor_amp_max=0.0 - an unstable transient,
    # not a real improvement: worst-case accuracy 0.892->0.916). Reverted for
    # good - pinning at THIS particular bound is apparently a stable local
    # optimum for the rest of the config, not evidence the bound is wrong.
    "ou_sigma": (0.0, 8.0),
    "tremor_amp_min": (0.0, 1.0),
    "tremor_amp_max": (0.0, 1.5),
    # Tried exposing tremor_freq_min/max (was hardcoded 8-12Hz) since
    # tremor_band_energy_ratio kept showing up as the worst-matched spread/
    # shape axis - regressed twice (v20 at v16's budget: 0.854->0.906; v21 at
    # a bumped budget: ->0.871), both worse than not touching it. Reverted;
    # this specific feature may just be inherently hard to match given the
    # FFT computation's own admitted coarseness near Nyquist at the 40ms
    # resampling grid (see features.py's docstring) - not every gap has a
    # config-level fix.
    "sdn_k": (0.0, 0.15),
    # Per-movement OU jump - a standalone test (not the full search) found
    # ou_jump_prob=0.1/ou_jump_scale=100 alone almost exactly reproduces
    # human jerk_rms's skew (3.85 vs 3.83) and kurtosis (24.1 vs 23.4),
    # something the earlier per-sample sdn_burst_prob/scale (also exposed,
    # bounds below) and a direct skew-matching fitness penalty (v10/v11,
    # disabled) both failed to do - a rare per-MOVEMENT impulse on the OU
    # drift survives features.py's 40ms resampling grid; per-sample noise on
    # SDN does not.
    # Pinned at its ceiling (0.35) in every run through v16 - widened to 0.6
    # as a test (v17): it pinned at the new ceiling again (0.595) but overall
    # accuracy came back slightly WORSE (worst-case 0.854->0.875) than v16.
    # Unlike v3's curvature_scale widening (which helped) or v5/v8's
    # ou_sigma/primary_sigma_max widening (which clearly regressed), this one
    # is closer to a wash/slight-regression - reverted to the confirmed-good
    # v16 bound rather than chase a parameter whose own ceiling-pinning
    # signal turned out not to be reliable here.
    "ou_jump_prob": (0.0, 0.35),
    "ou_jump_scale": (0.0, 150.0),
    "sdn_burst_prob": (0.0, 0.3),
    "sdn_burst_scale": (1.0, 8.0),
    "sample_dt_mean": (3.0, 20.0),
    "gamma_shape": (1.0, 8.0),
    "primary_sigma_min": (0.05, 0.5),
    # primary_sigma_max: same story - widening to 0.9 regressed the result
    # (see ou_sigma above), reverted to the pre-widening value.
    "primary_sigma_max": (0.05, 0.5),
    "overshoot_prob": (0.0, 0.5),
    "target_width": (5.0, 60.0),
    "fitts_a": (10.0, 150.0),
    "fitts_b": (50.0, 300.0),
    # Added after v3 plateaued on velocity_skewness while curvature was being
    # fixed: peak_time_ratio directly controls WHERE in the movement the
    # velocity peak lands, which is the main lever for the profile's
    # asymmetry (skewness) - it was never exposed to the search before this,
    # fixed at the hand-tuned default (0.35).
    # v14: direct quantile inspection of time_to_peak_ratio found human's
    # 5th/10th/25th percentile is 0.042/0.062/0.16 (a lot of movements peak
    # VERY early - fast ballistic initial acceleration) vs the bot's
    # 0.333/0.375/0.417 (a single-stump classifier on this ONE feature alone
    # already hits 77% accuracy). peak_time_ratio's old floor (0.15) caps
    # peak_t/total_t around 0.13 even at the extreme - below human's 25th
    # percentile already, nowhere near its 5th/10th. Lowered the floor so
    # early-peaking movements are reachable at all.
    "peak_time_ratio": (0.05, 0.55),
    # v3.1 achieved near-perfect MEAN matching (mean sq. z-score 0.017) with
    # essentially no gain in detector accuracy - a correlation/variance audit
    # (see README.md "v4") found motor_synergy_bot's movement-to-movement
    # VARIANCE was only 2-10% of human's for curvature/jerk/path_efficiency:
    # every generated movement looks nearly identical, regardless of how well
    # the average matches. These two (pulled out of previously-hardcoded
    # constants in generate_synthetic.py) directly control that spread.
    # Pinned at its ceiling in both v7 (0.5) and the wider test (0.9->0.32
    # the second time, so NOT actually pinned there) - widening this one
    # alone didn't clearly help or hurt in isolation, but it moved together
    # with the other two below in a run that regressed overall, so reverted
    # with them for a clean, isolated re-test next time.
    "mt_noise_sigma": (0.05, 0.5),
    "curvature_noise_sigma": (0.5, 4.0),
    # curvature_rms's spread stayed the single worst-matched axis even after
    # curvature_noise_sigma became tunable - the search settled at ~63% of
    # that range, not the ceiling, meaning it's a genuine trade-off against
    # other features rather than a bound blocking it. A single Gaussian
    # scale can't set "typical curvature" and "how much that varies between
    # movements" independently since both ride on the same sigma; this
    # per-movement lognormal "curviness style" multiplier is a second,
    # separate knob for exactly the variance half of that.
    "curvature_style_sigma": (0.0, 1.2),
    # Same fix, applied to time_to_peak_ratio's spread (stuck at log-ratio
    # ~-1.0 to -1.3 the entire previous run): the per-movement jitter window
    # around peak_time_ratio was hardcoded to +-0.03. v5 additionally found
    # this was a DISTRIBUTION-SHAPE ceiling, not just a bound: a uniform
    # jitter can't produce human-like outlier peak timings at any width, so
    # generate_synthetic.py switched it to a Gaussian offset (clipped to
    # [0.05, 0.95] of mt for physical sanity, so it no longer needs to stay
    # below peak_time_ratio's own minimum). That alone cut worst-case
    # accuracy from 0.949 to 0.905 (this run's best result). The search
    # Pinned at 0.10 (the ceiling) in EVERY run so far (v5, v6, v7) while
    # time_to_peak_ratio's variance never closed past log-ratio ~-0.8 - yet
    # widening it (twice now, alongside ou_sigma/primary_sigma_max the first
    # time, alongside ou_sigma/mt_noise_sigma the second) has regressed
    # overall accuracy both times despite the individual param never itself
    # being the one that visibly misbehaved. Reading this honestly: whatever
    # is capping time_to_peak_ratio's variance near log-ratio -0.8 is NOT
    # this bound - it's something else entirely (see README "Suggested next
    # steps"). Reverted for good; stop re-testing this specific lever.
    "peak_time_jitter": (0.01, 0.10),
    # v5 audit: peak_time_jitter's uniform->Gaussian fix was the single
    # biggest accuracy gain of the project. generate_synthetic.py applies the
    # identical fix to every other hardcoded uniform window in generate()
    # (reach fraction, correction timing/amount) - these three scale factors
    # expose that same spread to the search. 1.0 reproduces the original
    # windows' approximate spread; allowed to shrink toward 0 (tighter than
    # original) or grow well past it.
    # Both pinned at their bound in the first run with these exposed
    # (reach_jitter at its floor 0.1 - wants LESS spread than the original
    # window; correction_timing_jitter at its ceiling 3.0 - wants MORE) -
    # widened both further in the direction each was pushing.
    "reach_jitter": (0.0, 3.0),
    "correction_timing_jitter": (0.1, 6.0),
    "correction_amount_jitter": (0.1, 3.0),
}

# v11 tried 20/5/6 (up from 14/4/5) specifically to rescue v10's skew term
# with more budget - it still didn't beat v7 (worst-case 0.912 vs v7's
# 0.892), a SECOND confirmation (v10 at the old budget, v11 at 3x the
# effective search cost) that matching skew directly isn't paying off despite
# being permutation-importance-motivated. Reverted to v7's budget - see
# SKEW_MATCH_WEIGHT below for why the term itself is disabled, not deleted.
# Tried bumping to 28/7/8 (v19) on the theory that with every single feature
# now only weakly discriminative alone (0.52-0.62), the remaining gap is a
# harder BALANCING problem needing more search - it wasn't: v19 came back
# WORSE than v16 (worst-case 0.854->0.885) despite ~2x the per-generation
# cost. Reverted; the next gains are coming from specific structural fixes
# (tremor_band_energy_ratio next), not from raw budget.
# v20 added tremor_freq_min/max (2 new dims) at the v16 budget and
# regressed (0.854->0.906); v21 bumped budget too (22/6/7) to compensate -
# still worse (0.871). Both tremor_freq and the budget bump reverted; v16's
# 18/5/6 remains the best confirmed setting.
POP_SIZE = 18
ELITE_K = 5
GENERATIONS_PER_EPOCH = 6
# Tried 250 pre-v14 (v9) hoping less noisy MOMENT estimates would help -
# came back within noise of 150, not worth the cost, reverted. Different
# story now that fitness includes 99th-percentile QUANTILE matching (v14/15)
# - a 99th percentile from n=150 is essentially just the top ~1.5 samples,
# inherently noisy regardless of how stable quantiles are in general vs
# moments. Raised for the extreme-quantile terms specifically, not because
# the earlier moment-based test was wrong.
SAMPLES_PER_CANDIDATE = 300
HUMAN_SAMPLE_SIZE = 1200
FINAL_VALIDATION_SAMPLES = 800

N_ENSEMBLE = 6

# v3: distribution matching is the DOMINANT signal now, over a much broader
# set of behavioral axes than v2's 3 - covers speed, curvature, jerk (2
# ways), path efficiency, velocity-profile shape, and timing jointly, so
# fixing one axis can't be "free" at the expense of the others.
DIST_MATCH_FEATURES = [
    "mean_speed", "peak_speed", "curvature_rms", "jerk_rms", "jerk_max",
    "path_efficiency", "velocity_skewness", "time_to_peak_ratio", "movement_time",
    # v6: the previous run converged its own tracked z2/var2/cov2 metrics
    # BETTER than v5 yet scored WORSE against a fresh independent ensemble
    # (0.929 vs 0.905 worst-case) - a sign the search was pushing on the 9
    # tracked features while these 4 (part of `shape_only`, the detector's
    # actual training features, but never in the fitness target) drifted
    # further from human. Not "distance": that one's matched by
    # construction (bot samples draw distance from the human empirical
    # distribution directly).
    "num_submovements", "velocity_kurtosis", "tremor_band_energy_ratio", "sdn_correlation",
]
DIST_MATCH_WEIGHT = 2.0

# v4: mean-matching alone (weighted 2.0 above) got every feature's MEAN
# within 0.2 std of human and barely moved detector accuracy - a variance
# audit found motor_synergy_bot's per-feature STD was only 2-10% of human's.
# This term is weighted equally: matching the average without matching the
# spread isn't "half a solution," it's close to no solution at all.
VARIANCE_MATCH_WEIGHT = 2.0
CONVERGENCE_VAR_MAX_ABS_LOGRATIO = 0.3

# v5: mean+variance matching plateaued at ~95-96% accuracy across three
# whack-a-mole iterations even with a doubled search budget - marginal
# matching alone (9 independent means, 9 independent variances) says nothing
# about how the features relate to EACH OTHER. A tree-based detector can
# easily split on "high curvature AND low jerk together" even when both
# marginals match perfectly. This term matches the correlation matrix
# instead (mean squared difference over the 36 off-diagonal feature pairs).
COV_MATCH_WEIGHT = 2.0
CONVERGENCE_COV_MAX_ABS_DIFF = 0.35

# v10: a permutation-importance check on the v7-evolved config's own trained
# ensemble found `jerk_rms` is BY FAR the single most discriminative feature
# (0.18, next is time_to_peak_ratio at 0.11) - despite its mean/variance
# z-score and log-ratio both already looking "close" (within ~0.1-0.3).
# Comparing the raw distributions directly explained the gap: human jerk_rms
# has skew=3.83 (heavy right tail - most movements have small jerk, a few
# have huge jerk) vs the bot's 2.92, and the MEDIAN differs 2x (0.4 vs 0.2)
# even though the MEAN happens to match (a few large bot outliers pull the
# mean up while the bulk sits well below human's bulk) - mean+variance are
# only the first two moments; this is a real distribution-SHAPE gap neither
# term can see. Same pattern visible in peak_speed (skew 5.78 vs 2.23) and
# path_efficiency (skew -1.75 vs -0.99). Adds the third moment to the
# fitness alongside mean/variance/covariance.
#
# Was DISABLED (weight 0): a raw skew penalty on top of the OLD generator
# failed twice (v10 at normal budget: 0.892->0.940; v11 at ~3x budget:
# ->0.912) - the generator had no actual mechanism to produce a heavy tail,
# so the penalty term just fought an impossible constraint. Re-enabled now
# that `generate_synthetic.py` has one (`ou_jump_prob`/`ou_jump_scale` - a
# rare per-movement impulse that alone reproduced human jerk_rms's skew/
# kurtosis almost exactly in a standalone test) - this time the penalty has
# something real to push toward instead of pressure with no lever behind it.
# v12 (weight 1.0) overshot badly: jerk_rms skew went from -2.51 (too low)
# to +9.53 (way too high) vs human. v13 (weight 3.0) pulled the batch-level
# skew estimate back to +0.66 - looked fixed - but accuracy didn't improve
# (0.892->0.907, still worse than v7) and permutation_importance still
# showed jerk_rms as the #1 discriminator (0.107, down from 0.18 but still
# clearly dominant).
#
# ROOT CAUSE, found by directly measuring it: raw skewness is a HIGH-NOISE
# statistic for a distribution this heavy-tailed. Five batches of the SAME
# 150-sample-per-candidate config gave skew estimates of 3.58, 5.02, 6.14,
# 6.37, 8.49 - over 2x spread for identical underlying settings, because a
# handful of rare ou_jump events can swing the empirical 3rd-moment
# estimate enormously. The search wasn't converging on a bad target, it was
# chasing a fitness signal with more noise than gradient. DISABLED for good.
SKEW_MATCH_WEIGHT = 0.0

# v14: quantiles of the same heavy-tailed feature are far more stable than
# its raw moments (the same 5-batch check: p95 varied 0.82-1.25, ~40%
# relative spread, vs skew's >100%) - a well-known property (order
# statistics are robust to exactly the rare extreme values that make raw
# skew/kurtosis noisy). Matches quantiles directly instead of standardized
# moments - same "what shape does this feature's distribution have" question,
# asked in a way the fitness signal can actually act on.
#
# First run (upper tail only, 75/90/95/99) fixed jerk_rms's shape almost
# exactly (tail-diff +3.43 -> +0.09) but accuracy didn't improve - a fresh
# permutation-importance check showed `time_to_peak_ratio` had become the
# new #1 discriminator (0.164). Direct quantile inspection explained why:
# human's 5th/10th percentile is 0.042/0.062 (many movements peak VERY
# early) vs the bot's 0.333/0.375 - a single feature alone gets 77% accuracy
# on this gap. That's a LOWER-tail problem, invisible to an upper-tail-only
# term. Extended to cover both tails (paired with widening peak_time_ratio's
# floor in CONFIG_BOUNDS, which was the other half of the actual bug - the
# floor made low time_to_peak_ratio values structurally unreachable).
# Tried adding 50 (median) after noticing jerk_rms's MEAN matches human
# almost exactly (0.510 vs 0.508) while its MEDIAN is still off (0.328 vs
# 0.389) - with skew this extreme, a matched mean says nothing about the
# median. Regressed slightly instead (v18: worst-case 0.854->0.875) -
# reverted. The median gap is real but this wasn't the fix for it.
TAIL_QUANTILES = (5, 10, 25, 75, 90, 95, 99)
# v15 beat v7 (0.892->0.881 worst-case) but jerk_rms/peak_speed both still
# show real quantile gaps in a fresh diagnostic (peak_speed skew 5.78 vs
# bot's 1.65) - raised alongside the SAMPLES_PER_CANDIDATE bump above so the
# term both sees the gap more clearly (less noisy estimates) and is weighted
# to actually close it.
TAIL_MATCH_WEIGHT = 3.0
CONVERGENCE_TAIL_MAX_ABS_Z = 0.5

# Loop-until-converged controls. MAX_EPOCHS is a safety cap, not the intended
# stopping condition - the loop is meant to stop via CONVERGENCE_Z2 or PATIENCE.
MAX_EPOCHS = 40
PATIENCE = 6
CONVERGENCE_Z2 = 0.05
# Converging on the MEAN squared z-score isn't enough by itself - v3's first
# run under widened curvature bounds hit mean z2 < 0.05 in one epoch by
# fixing curvature_rms, but left velocity_skewness/movement_time pushed to
# z=-0.50/+0.39 in the process (fixing one axis, unbalancing others). Require
# every individual feature within this bound too, not just the average.
CONVERGENCE_MAX_ABS_Z = 0.3
MIN_IMPROVEMENT = 0.002

# The real target ("can the detector even tell?"), not just the distribution
# proxy: every CHECK_ACC_INTERVAL epochs, train a small ensemble on a genuine
# HOLDOUT split (never touched by the search's own ensemble/fitness) and
# measure accuracy directly. 0.5 = true chance - that's the actual goal here;
# PATIENCE is what stops the loop honestly if that turns out to be
# unreachable under the current parametrization, rather than an early
# artificial ceiling.
ACCURACY_TARGET = 0.55
CHECK_ACC_INTERVAL = 3
HOLDOUT_SAMPLE_SIZE = 500


def load_human_pool(seed=0):
    """All valid human movements, shuffled once. Callers slice disjoint
    chunks off the front (search / holdout-for-accuracy-checks / final-
    validation) so those splits never overlap - a real train/holdout
    boundary, not three independently-reshuffled draws from the same pool
    that could coincidentally share rows."""
    path = DATA_DIR / "human_movements.jsonl"
    with open(path) as f:
        lines = f.readlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    rows = []
    distances = []
    for line in lines:
        rec = json.loads(line)
        pts = rec["points"]
        if len(pts) < 4:
            continue
        rows.append(extract_features(pts))
        x0, y0, _ = pts[0]
        x1, y1, _ = pts[-1]
        distances.append(math.hypot(x1 - x0, y1 - y0))
    return rows, distances


def load_human_sample(n, seed=0):
    rows, distances = load_human_pool(seed=seed)
    return rows[:n], distances[:n]


def sample_candidate_movements(cfg_overrides, distances, n, rng):
    rows = []
    for _ in range(n):
        distance = rng.choice(distances)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        pts = motor_synergy_generate(
            0.0, 0.0, distance * math.cos(angle), distance * math.sin(angle),
            cfg=cfg_overrides, rng=rng,
        )
        if len(pts) < 4:
            continue
        rows.append(extract_features(pts))
    return rows


def clip_bounds(cfg):
    return {k: float(np.clip(v, *CONFIG_BOUNDS[k])) for k, v in cfg.items()}


def random_config(rng):
    return clip_bounds({k: rng.uniform(lo, hi) for k, (lo, hi) in CONFIG_BOUNDS.items()})


def mutate(cfg, rng, scale):
    child = {}
    for k, v in cfg.items():
        lo, hi = CONFIG_BOUNDS[k]
        child[k] = v + rng.gauss(0.0, scale * (hi - lo))
    return clip_bounds(child)


_DEFAULT_ENSEMBLE_HYPERPARAMS = {
    "GradientBoosting": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.2, "subsample": 1.0},
    "HistGradientBoosting": {"max_iter": 100, "max_depth": 4, "learning_rate": 0.05, "l2_regularization": 1.0},
    "RandomForest": {"n_estimators": 200, "min_samples_leaf": 8, "max_features": "sqrt", "max_depth": 8},
}

_ENSEMBLE_HYPERPARAMS_PATH = RESULTS_DIR / "ensemble_hyperparams.json"

_ENSEMBLE_CLASSES = {
    "GradientBoosting": GradientBoostingClassifier,
    "HistGradientBoosting": HistGradientBoostingClassifier,
    "RandomForest": RandomForestClassifier,
}


def _ensemble_model_factories():
    """3 independently-tuned model families. Hyperparameters are loaded from
    results/ensemble_hyperparams.json if present (written by
    tune_ensemble_hyperparams.py after a fresh RandomizedSearchCV against the
    CURRENT generator's output - this is what makes co_evolution_loop.py a
    genuine alternating co-evolution: detector hyperparameters get re-tuned
    against the generator's latest output every round, not fixed once).
    Falls back to _DEFAULT_ENSEMBLE_HYPERPARAMS (round 2's values, themselves
    taken from results/strong_detector_validation.md) if the file is absent -
    round 1's ensemble was N bootstraps of a single, never-retuned
    HistGradientBoostingClassifier, which a fresh independent search showed
    was significantly weaker than what a properly-tuned detector finds on the
    SAME shape_only features (0.598 -> 0.855 held-out accuracy)."""
    params = dict(_DEFAULT_ENSEMBLE_HYPERPARAMS)
    if _ENSEMBLE_HYPERPARAMS_PATH.exists():
        loaded = json.loads(_ENSEMBLE_HYPERPARAMS_PATH.read_text())
        params.update(loaded)
    return [
        (lambda seed, name=name, kwargs=kwargs: _ENSEMBLE_CLASSES[name](random_state=seed, **kwargs))
        for name, kwargs in params.items()
    ]


def train_detector_ensemble(human_rows, bot_rows, n=N_ENSEMBLE, seed_base=0):
    """N independently-bootstrapped models, cycling through 3 differently-tuned
    model families (see _ensemble_model_factories). A config must fool several
    differently-trained, differently-architected models at once to score well."""
    df = pd.DataFrame(human_rows + bot_rows)
    y = np.array([0] * len(human_rows) + [1] * len(bot_rows))
    n_total = len(df)
    factories = _ensemble_model_factories()
    models = []
    for i in range(n):
        rng_np = np.random.default_rng(seed_base + i)
        idx = rng_np.integers(0, n_total, size=n_total)  # bootstrap resample
        X = df.iloc[idx][FEATURE_NAMES].to_numpy()
        y_boot = y[idx]
        model = factories[i % len(factories)](seed_base + i)
        model.fit(X, y_boot)
        models.append(model)
    return models


def ensemble_proba(models, rows):
    X = pd.DataFrame(rows)[FEATURE_NAMES].to_numpy()
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def ensemble_accuracy(models, human_rows, bot_rows, reduce="mean"):
    """reduce='mean': accuracy of the ensemble-averaged probability (soft
    vote). reduce='worst': the SINGLE worst (lowest) per-member accuracy -
    "genuinely generalizes" should mean fooling ~all members, not just the
    average of a few."""
    rows = human_rows + bot_rows
    y = np.array([0] * len(human_rows) + [1] * len(bot_rows))
    X = pd.DataFrame(rows)[FEATURE_NAMES].to_numpy()
    per_member = [accuracy_score(y, m.predict(X)) for m in models]
    if reduce == "worst":
        return float(min(per_member))
    proba = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return accuracy_score(y, (proba >= 0.5).astype(int))


def human_moment_stats(human_rows):
    df = pd.DataFrame(human_rows)
    return {f: (float(df[f].mean()), float(df[f].std()) or 1.0) for f in DIST_MATCH_FEATURES}


def distribution_penalty(rows, human_stats, per_feature=False):
    """Mean squared z-score distance of this candidate batch's own feature
    MEANS from the human sample's mean/std - pushes the search toward
    genuinely resembling the human distribution instead of only toward
    whatever the detector ensemble's blind spots are. per_feature=True also
    returns the per-feature breakdown for diagnostics.

    Deliberately blind to spread: v3.1 matched every one of these means to
    within 0.2 std and accuracy barely moved, because mean-matching says
    nothing about variance - see variance_penalty."""
    if not rows:
        return (4.0, {}) if per_feature else 4.0
    df = pd.DataFrame(rows)
    zs = {}
    for f in DIST_MATCH_FEATURES:
        mean_h, std_h = human_stats[f]
        zs[f] = (float(df[f].mean()) - mean_h) / std_h
    mean_sq = sum(z * z for z in zs.values()) / len(zs)
    return (mean_sq, zs) if per_feature else mean_sq


def variance_penalty(rows, human_stats, per_feature=False):
    """Mean squared log-variance-ratio: log(candidate_std / human_std),
    squared, averaged over features - symmetric in over- vs under-dispersion
    (a bot twice as spread as human and a bot half as spread score the same).
    Added after diagnosing that motor_synergy_bot's per-feature std was only
    2-10% of human's for curvature/jerk/path_efficiency even once every mean
    matched near-perfectly - a detector can trivially tell "these samples are
    all nearly identical" apart from "these vary widely" regardless of where
    the average sits."""
    if len(rows) < 8:
        return (4.0, {}) if per_feature else 4.0
    df = pd.DataFrame(rows)
    log_ratios = {}
    for f in DIST_MATCH_FEATURES:
        _, std_h = human_stats[f]
        std_c = float(df[f].std())
        if std_c <= 1e-9 or std_h <= 1e-9:
            log_ratios[f] = -3.0  # candidate has ~zero spread - heavily penalize
            continue
        log_ratios[f] = math.log(std_c / std_h)
    mean_sq = sum(r * r for r in log_ratios.values()) / len(log_ratios)
    return (mean_sq, log_ratios) if per_feature else mean_sq


def human_tail_stats(human_rows):
    df = pd.DataFrame(human_rows)
    stats = {}
    for f in DIST_MATCH_FEATURES:
        s = df[f]
        std = float(s.std()) or 1.0
        stats[f] = {"std": std, "q": {q: float(np.percentile(s, q)) for q in TAIL_QUANTILES}}
    return stats


def tail_penalty(rows, human_tail, per_feature=False):
    """Mean squared (std-normalized) difference in upper-tail quantiles
    (75/90/95/99th percentile) between the candidate batch and human data -
    a robust alternative to skewness_penalty's raw 3rd moment. Order
    statistics barely move when a handful of rare extreme samples change,
    where raw skew/kurtosis swing wildly (measured directly: 5 batches of
    the identical config gave skew 3.58-8.49 but p95 only 0.82-1.25) - same
    "heavy right tail or not" question, asked in a way search can act on."""
    if len(rows) < 20:
        return (4.0, {}) if per_feature else 4.0
    df = pd.DataFrame(rows)
    diffs = {}
    for f in DIST_MATCH_FEATURES:
        s = df[f]
        std_h = human_tail[f]["std"]
        sq_sum = 0.0
        for q in TAIL_QUANTILES:
            c_val = float(np.percentile(s, q))
            sq_sum += ((c_val - human_tail[f]["q"][q]) / std_h) ** 2
        diffs[f] = sq_sum / len(TAIL_QUANTILES)
    mean_sq = sum(diffs.values()) / len(diffs)
    return (mean_sq, diffs) if per_feature else mean_sq


def human_corr_matrix(human_rows):
    return pd.DataFrame(human_rows)[DIST_MATCH_FEATURES].corr()


def covariance_penalty(rows, human_corr, per_pair=False):
    """Mean squared difference between the candidate batch's own correlation
    matrix and the human one, over the off-diagonal feature pairs (36 for 9
    features) - matching 9 independent means and 9 independent variances
    says nothing about whether e.g. curvature and jerk co-occur the same way
    in both; this is the term that actually looks at that."""
    if len(rows) < 8:
        return (1.0, {}) if per_pair else 1.0
    df = pd.DataFrame(rows)[DIST_MATCH_FEATURES]
    cand_corr = df.corr()
    diffs = {}
    for i, f1 in enumerate(DIST_MATCH_FEATURES):
        for f2 in DIST_MATCH_FEATURES[i + 1:]:
            h = human_corr.loc[f1, f2]
            c = cand_corr.loc[f1, f2]
            c = 0.0 if pd.isna(c) else c
            diffs[(f1, f2)] = float(h - c)
    mean_sq = sum(d * d for d in diffs.values()) / len(diffs)
    return (mean_sq, diffs) if per_pair else mean_sq


def fitness_of(models, rows, human_stats, human_corr, human_tail):
    if not rows:
        return (
            1.0 + DIST_MATCH_WEIGHT * 4.0 + VARIANCE_MATCH_WEIGHT * 4.0
            + COV_MATCH_WEIGHT * 1.0 + TAIL_MATCH_WEIGHT * 4.0
        )
    detector_term = float(np.mean(ensemble_proba(models, rows)))
    dist_term = distribution_penalty(rows, human_stats)
    var_term = variance_penalty(rows, human_stats)
    cov_term = covariance_penalty(rows, human_corr)
    tail_term = tail_penalty(rows, human_tail)
    return (
        detector_term + DIST_MATCH_WEIGHT * dist_term
        + VARIANCE_MATCH_WEIGHT * var_term + COV_MATCH_WEIGHT * cov_term
        + TAIL_MATCH_WEIGHT * tail_term
    )


# Parallel candidate scoring. The dominant per-generation cost is scoring
# POP_SIZE candidates (each: SAMPLES_PER_CANDIDATE calls to
# motor_synergy_generate + extract_features, pure Python/numpy, no GIL
# release) - completely independent per candidate given a fixed ensemble, so
# this is embarrassingly parallel. On a 12-core machine this was running
# single-threaded with only sklearn's internal tree-building using a few
# cores (~330% CPU total, most cores idle) - worth fixing regardless of
# whether better hardware is ever available, since the idle cores are
# already sitting right here.
# Cap configurable via env var (default 11, tuned for the 12-core Jetson
# this was originally written on) - a 20-core desktop (p22507) has real
# spare capacity above that cap, set MAX_WORKERS higher there.
N_WORKERS = max(1, min(int(os.environ.get("MAX_WORKERS", "11")), (os.cpu_count() or 4) - 1))
# CRITICAL: must use "spawn", not the Linux default "fork". sklearn/numpy
# (via HistGradientBoostingClassifier training and BLAS) leave background
# threads alive; fork()-ing while those threads exist can copy a held lock
# into the child with no thread left to release it - a classic
# fork-after-threading deadlock. Hit this immediately: all 11 workers plus
# the main process sat at ~0% CPU, blocked in futex_wait_queue_me,
# indefinitely.
#
# Because of that, the pool is created ONCE for the whole run, not per
# epoch: "spawn" starts each worker as a genuinely fresh interpreter that has
# to re-import numpy/pandas/sklearn from scratch, which turned out to cost
# MORE than it saved when a fresh pool (11 fresh interpreters) was spun up
# every epoch (measured: 26.7s to score 18 candidates in parallel vs ~22s
# sequentially - the import overhead alone ate the entire parallel speedup).
# So `_init_worker` only receives what's truly constant for the whole run
# (human targets, distances); the ensemble - which DOES change every epoch -
# is passed as part of each task's arguments instead of via the initializer.
_MP_CTX = multiprocessing.get_context("spawn")
_worker_state = {}


def _init_worker(human_stats, human_corr, human_tail, distances):
    _worker_state["human_stats"] = human_stats
    _worker_state["human_corr"] = human_corr
    _worker_state["human_tail"] = human_tail
    _worker_state["distances"] = distances


def _score_candidate_worker(args):
    cfg, seed, n, ensemble = args
    rows = sample_candidate_movements(cfg, _worker_state["distances"], n, random.Random(seed))
    fit = fitness_of(
        ensemble, rows, _worker_state["human_stats"],
        _worker_state["human_corr"], _worker_state["human_tail"],
    )
    return fit, rows


def score_population(pool, population, rng, n, ensemble):
    """Scores each config in `population` in parallel via the persistent
    `pool`. Each candidate gets an independent seed drawn from the driving
    `rng` so the overall search stays a deterministic function of the
    top-level seed regardless of how many workers happen to run it. `ensemble`
    (the current epoch's trained models) rides along in each task's argument
    tuple rather than the pool's fixed initializer state, since it changes
    every epoch but the pool itself is only created once."""
    tasks = [(cfg, rng.randrange(2**31), n, ensemble) for cfg in population]
    results = list(pool.map(_score_candidate_worker, tasks))
    return [(fit, cfg, rows) for (fit, rows), cfg in zip(results, population)]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Multi-start: the search's own exploration path (population init,
    # mutation, per-candidate seeds) is stochastic and shows real run-to-run
    # variance even at identical settings (v16/v19/v20/v21 all differ by
    # several points of worst-case accuracy) - varying just this seed (NOT
    # load_human_pool's seed=0, which stays fixed so every run compares
    # against the identical human data split) is a legitimate, cheap way to
    # sample multiple local optima now that a full run takes ~5 min instead
    # of 15-90 thanks to the parallelization fix. Sourced from an env var so
    # a sweep script can launch several runs without editing this file.
    search_seed = int(os.environ.get("SEARCH_SEED", "42"))
    print(f"[adversarial_loop] search_seed={search_seed}")
    rng = random.Random(search_seed)

    print("[adversarial_loop] loading human pool + extracting features once...")
    # One shuffle, sliced into disjoint chunks - search / periodic-holdout /
    # final-validation never share a row (three independent re-shuffles of
    # the same ~16.5k-movement pool could coincidentally overlap; slicing
    # one shuffle guarantees they don't).
    pool_rows, pool_distances = load_human_pool(seed=0)
    a, b = HUMAN_SAMPLE_SIZE, HUMAN_SAMPLE_SIZE + HOLDOUT_SAMPLE_SIZE
    c = b + FINAL_VALIDATION_SAMPLES
    human_rows, distances = pool_rows[:a], pool_distances[:a]
    holdout_human_rows, holdout_distances = pool_rows[a:b], pool_distances[a:b]
    final_human_rows, final_distances = pool_rows[b:c], pool_distances[b:c]
    print(f"[adversarial_loop] human sample: {len(human_rows)} movements")
    human_stats = human_moment_stats(human_rows)
    human_corr = human_corr_matrix(human_rows)
    human_tail = human_tail_stats(human_rows)
    print(f"[adversarial_loop] human distribution targets ({len(DIST_MATCH_FEATURES)} features): {human_stats}")
    print(f"[adversarial_loop] holdout sample (periodic accuracy checks): {len(holdout_human_rows)} movements")
    print(f"[adversarial_loop] final-validation sample (never touched during search): {len(final_human_rows)} movements")

    print(f"[adversarial_loop] generating default-config bot sample for the initial {N_ENSEMBLE}-model ensemble...")
    default_bot_rows = sample_candidate_movements({}, distances, HUMAN_SAMPLE_SIZE, rng)
    ensemble = train_detector_ensemble(human_rows, default_bot_rows, seed_base=0)
    baseline_acc = ensemble_accuracy(ensemble, human_rows, default_bot_rows)
    baseline_acc_worst = ensemble_accuracy(ensemble, human_rows, default_bot_rows, reduce="worst")
    baseline_z2, baseline_zs = distribution_penalty(default_bot_rows, human_stats, per_feature=True)
    baseline_max_abs_z = max(abs(z) for z in baseline_zs.values())
    baseline_var2, baseline_logratios = variance_penalty(default_bot_rows, human_stats, per_feature=True)
    baseline_max_abs_logratio = max(abs(r) for r in baseline_logratios.values())
    baseline_cov2, baseline_diffs = covariance_penalty(default_bot_rows, human_corr, per_pair=True)
    baseline_max_abs_diff = max(abs(d) for d in baseline_diffs.values())
    baseline_tail2, baseline_tail_diffs = tail_penalty(default_bot_rows, human_tail, per_feature=True)
    baseline_max_abs_tail_diff = max(abs(d) for d in baseline_tail_diffs.values())
    print(
        f"[adversarial_loop] initial (default config): ensemble_acc mean={baseline_acc:.3f} "
        f"worst_member={baseline_acc_worst:.3f} dist_z2={baseline_z2:.3f} max_abs_z={baseline_max_abs_z:.3f} "
        f"var2={baseline_var2:.3f} max_abs_logratio={baseline_max_abs_logratio:.3f} "
        f"cov2={baseline_cov2:.3f} max_abs_diff={baseline_max_abs_diff:.3f} "
        f"tail2={baseline_tail2:.3f} max_abs_tail_diff={baseline_max_abs_tail_diff:.3f}"
    )

    population = [dict() for _ in range(1)] + [random_config(rng) for _ in range(POP_SIZE - 1)]
    history = []
    convergence_history = []
    holdout_acc_history = []
    round_idx = 0
    best_cfg_overall = {}
    best_fitness_overall = (
        1.0 + DIST_MATCH_WEIGHT * 4.0 + VARIANCE_MATCH_WEIGHT * 4.0
        + COV_MATCH_WEIGHT * 1.0 + SKEW_MATCH_WEIGHT * 9.0
    )
    best_z2_seen = max(
        baseline_max_abs_z / CONVERGENCE_MAX_ABS_Z,
        baseline_max_abs_logratio / CONVERGENCE_VAR_MAX_ABS_LOGRATIO,
        baseline_max_abs_diff / CONVERGENCE_COV_MAX_ABS_DIFF,
        baseline_max_abs_tail_diff / CONVERGENCE_TAIL_MAX_ABS_Z,
    )
    best_holdout_acc_seen = 1.0
    no_improve_epochs = 0
    stop_reason = f"reached MAX_EPOCHS={MAX_EPOCHS}"

    # Created ONCE for the entire run (not per epoch - see the N_WORKERS
    # comment above for why): `ensemble` changes every epoch and rides along
    # in each task's arguments instead of the pool's fixed initializer state.
    pool = ProcessPoolExecutor(
        max_workers=N_WORKERS, mp_context=_MP_CTX, initializer=_init_worker,
        initargs=(human_stats, human_corr, human_tail, distances),
    )

    epoch = 0
    while epoch < MAX_EPOCHS:
        mutation_base = max(0.3 * (0.93 ** epoch), 0.05)
        mutation_scale = mutation_base
        for gen in range(GENERATIONS_PER_EPOCH):
            round_idx += 1
            scored = score_population(pool, population, rng, SAMPLES_PER_CANDIDATE, ensemble)
            scored.sort(key=lambda item: item[0])

            best_fit, best_cfg, best_rows = scored[0]
            measured_acc = ensemble_accuracy(ensemble, human_rows[:len(best_rows)], best_rows)
            measured_z2 = distribution_penalty(best_rows, human_stats)
            history.append({
                "round": round_idx, "epoch": epoch, "gen": gen,
                "best_fitness": best_fit, "measured_accuracy": measured_acc, "measured_z2": measured_z2,
            })
            print(
                f"[adversarial_loop] epoch {epoch} gen {gen}: "
                f"fitness={best_fit:.3f} ensemble_acc={measured_acc:.3f} dist_z2={measured_z2:.3f}"
            )

            if best_fit < best_fitness_overall:
                best_fitness_overall = best_fit
                best_cfg_overall = best_cfg

            elite = [cfg for _, cfg, _ in scored[:ELITE_K]]
            children = []
            while len(elite) + len(children) < POP_SIZE:
                parent = rng.choice(elite)
                children.append(mutate(parent, rng, mutation_scale))
            population = elite + children
            mutation_scale *= 0.75

        # End of epoch: retrain the WHOLE ensemble (fresh bootstrap
        # resamples, new seeds) against the current best candidate's output.
        best_fit, best_cfg, best_rows = min(
            score_population(pool, population, rng, SAMPLES_PER_CANDIDATE, ensemble),
            key=lambda item: item[0],
        )
        ensemble = train_detector_ensemble(human_rows, best_rows, seed_base=(epoch + 1) * 100)

        # Cheap (no training needed), running convergence check: how close is
        # the OVERALL best config found so far to the human distribution?
        # Track mean AND variance match, each by their own worst single
        # feature - converging on the average alone can hide one feature
        # still way off, and (per v3.1/v4) mean-matching alone says nothing
        # about whether the spread matches at all.
        check_rows = sample_candidate_movements(best_cfg_overall, distances, FINAL_VALIDATION_SAMPLES // 2, rng)
        check_z2, check_zs = distribution_penalty(check_rows, human_stats, per_feature=True)
        check_max_abs_z = max(abs(z) for z in check_zs.values())
        worst_feature = max(check_zs, key=lambda f: abs(check_zs[f]))
        check_var2, check_logratios = variance_penalty(check_rows, human_stats, per_feature=True)
        check_max_abs_logratio = max(abs(r) for r in check_logratios.values())
        worst_var_feature = max(check_logratios, key=lambda f: abs(check_logratios[f]))
        check_cov2, check_diffs = covariance_penalty(check_rows, human_corr, per_pair=True)
        check_max_abs_diff = max(abs(d) for d in check_diffs.values())
        worst_cov_pair = max(check_diffs, key=lambda p: abs(check_diffs[p]))
        check_tail2, check_tail_diffs = tail_penalty(check_rows, human_tail, per_feature=True)
        check_max_abs_tail_diff = max(abs(d) for d in check_tail_diffs.values())
        worst_tail_feature = max(check_tail_diffs, key=lambda f: abs(check_tail_diffs[f]))
        convergence_history.append({
            "epoch": epoch, "z2": check_z2, "max_abs_z": check_max_abs_z,
            "var2": check_var2, "max_abs_logratio": check_max_abs_logratio,
            "cov2": check_cov2, "max_abs_diff": check_max_abs_diff,
            "tail2": check_tail2, "max_abs_tail_diff": check_max_abs_tail_diff,
        })
        print(
            f"[adversarial_loop] --- epoch {epoch} done: running-best config dist_z2={check_z2:.4f} "
            f"(target < {CONVERGENCE_Z2}), worst mean = {worst_feature} (z={check_zs[worst_feature]:+.2f}); "
            f"var2={check_var2:.4f}, worst spread = {worst_var_feature} "
            f"(log-ratio={check_logratios[worst_var_feature]:+.2f}, target |.| < {CONVERGENCE_VAR_MAX_ABS_LOGRATIO}); "
            f"cov2={check_cov2:.4f}, worst pair = {worst_cov_pair} "
            f"(diff={check_diffs[worst_cov_pair]:+.2f}, target |.| < {CONVERGENCE_COV_MAX_ABS_DIFF}); "
            f"tail2={check_tail2:.4f}, worst shape = {worst_tail_feature} "
            f"(diff={check_tail_diffs[worst_tail_feature]:+.2f}, target |.| < {CONVERGENCE_TAIL_MAX_ABS_Z}) ---"
        )

        # Track whichever constraint is currently furthest from its own
        # target, normalized so the four different units (z-score vs
        # log-ratio vs correlation difference vs skew difference) are
        # comparable - "improved" means that worst-offending axis got
        # better, whichever one it currently is.
        combined_worst = max(
            check_max_abs_z / CONVERGENCE_MAX_ABS_Z,
            check_max_abs_logratio / CONVERGENCE_VAR_MAX_ABS_LOGRATIO,
            check_max_abs_diff / CONVERGENCE_COV_MAX_ABS_DIFF,
            check_max_abs_tail_diff / CONVERGENCE_TAIL_MAX_ABS_Z,
        )
        z2_improved = combined_worst < best_z2_seen - MIN_IMPROVEMENT
        if z2_improved:
            best_z2_seen = combined_worst

        # Every CHECK_ACC_INTERVAL epochs, measure the REAL target directly:
        # a fresh small ensemble on the untouched holdout split. This is the
        # actual "can the detector even tell?" number, not the z2 proxy.
        holdout_acc = None
        if (epoch + 1) % CHECK_ACC_INTERVAL == 0:
            holdout_bot_rows = sample_candidate_movements(
                best_cfg_overall, holdout_distances, len(holdout_human_rows), rng
            )
            half = len(holdout_human_rows) // 2
            holdout_ensemble = train_detector_ensemble(
                holdout_human_rows[:half], holdout_bot_rows[: len(holdout_bot_rows) // 2],
                seed_base=(epoch + 1) * 7000,
            )
            holdout_acc = ensemble_accuracy(
                holdout_ensemble, holdout_human_rows[half:], holdout_bot_rows[len(holdout_bot_rows) // 2:]
            )
            holdout_acc_history.append({"epoch": epoch, "acc": holdout_acc})
            print(f"[adversarial_loop]     holdout accuracy check: {holdout_acc:.3f} (target < {ACCURACY_TARGET})")
            if holdout_acc < best_holdout_acc_seen - MIN_IMPROVEMENT:
                best_holdout_acc_seen = holdout_acc
                no_improve_epochs = 0
            elif not z2_improved:
                no_improve_epochs += 1
        elif z2_improved:
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if holdout_acc is not None and holdout_acc < ACCURACY_TARGET:
            stop_reason = (
                f"converged: holdout accuracy={holdout_acc:.3f} < {ACCURACY_TARGET} "
                f"(dist_z2={check_z2:.4f}, var2={check_var2:.4f}, cov2={check_cov2:.4f}, tail2={check_tail2:.4f})"
            )
            epoch += 1
            break
        if (
            check_z2 < CONVERGENCE_Z2 and check_max_abs_z < CONVERGENCE_MAX_ABS_Z
            and check_max_abs_logratio < CONVERGENCE_VAR_MAX_ABS_LOGRATIO
            and check_max_abs_diff < CONVERGENCE_COV_MAX_ABS_DIFF
            and check_max_abs_tail_diff < CONVERGENCE_TAIL_MAX_ABS_Z
            and holdout_acc is not None and holdout_acc < 0.8
        ):
            stop_reason = (
                f"converged: dist_z2={check_z2:.4f} < {CONVERGENCE_Z2}, worst mean |z|={check_max_abs_z:.2f} < "
                f"{CONVERGENCE_MAX_ABS_Z}, worst spread |log-ratio|={check_max_abs_logratio:.2f} < "
                f"{CONVERGENCE_VAR_MAX_ABS_LOGRATIO}, worst cov |diff|={check_max_abs_diff:.2f} < "
                f"{CONVERGENCE_COV_MAX_ABS_DIFF}, worst tail |diff|={check_max_abs_tail_diff:.2f} < "
                f"{CONVERGENCE_TAIL_MAX_ABS_Z}, holdout accuracy={holdout_acc:.3f}"
            )
            epoch += 1
            break
        if no_improve_epochs >= PATIENCE:
            best_acc_str = f"{best_holdout_acc_seen:.3f}" if best_holdout_acc_seen < 1.0 else "n/a"
            stop_reason = (
                f"plateaued: no improvement in worst mean/spread constraint or holdout accuracy for {PATIENCE} epochs "
                f"(best combined constraint={best_z2_seen:.4f}, best holdout accuracy={best_acc_str})"
            )
            epoch += 1
            break
        epoch += 1

    pool.shutdown(wait=True)

    n_epochs_run = epoch
    print(f"[adversarial_loop] STOPPED after {n_epochs_run} epochs: {stop_reason}")
    print(f"[adversarial_loop] best config found (fitness={best_fitness_overall:.3f}):")
    print(json.dumps(best_cfg_overall, indent=2))

    # Independent final validation - a FRESH ensemble that never saw the
    # search's adaptively-retrained ensembles, comparing default vs evolved.
    print("[adversarial_loop] independent final validation...")
    human_rows_final, distances_final = final_human_rows, final_distances
    default_final = sample_candidate_movements({}, distances_final, FINAL_VALIDATION_SAMPLES, random.Random(1))
    evolved_final = sample_candidate_movements(best_cfg_overall, distances_final, FINAL_VALIDATION_SAMPLES, random.Random(2))

    half_h = len(human_rows_final) // 2
    fresh_ensemble_default = train_detector_ensemble(
        human_rows_final[:half_h], default_final[: len(default_final) // 2], seed_base=9000
    )
    acc_default = ensemble_accuracy(fresh_ensemble_default, human_rows_final[half_h:], default_final[len(default_final) // 2:])
    acc_default_worst = ensemble_accuracy(
        fresh_ensemble_default, human_rows_final[half_h:], default_final[len(default_final) // 2:], reduce="worst"
    )

    fresh_ensemble_evolved = train_detector_ensemble(
        human_rows_final[:half_h], evolved_final[: len(evolved_final) // 2], seed_base=9100
    )
    acc_evolved = ensemble_accuracy(fresh_ensemble_evolved, human_rows_final[half_h:], evolved_final[len(evolved_final) // 2:])
    acc_evolved_worst = ensemble_accuracy(
        fresh_ensemble_evolved, human_rows_final[half_h:], evolved_final[len(evolved_final) // 2:], reduce="worst"
    )
    print(f"[adversarial_loop] independent ensemble accuracy - default config: mean={acc_default:.3f} worst={acc_default_worst:.3f}")
    print(f"[adversarial_loop] independent ensemble accuracy - evolved config: mean={acc_evolved:.3f} worst={acc_evolved_worst:.3f}")

    dist_default, z_default = distribution_penalty(default_final, human_stats, per_feature=True)
    dist_evolved, z_evolved = distribution_penalty(evolved_final, human_stats, per_feature=True)
    print(f"[adversarial_loop] mean squared z-score vs human - default: {dist_default:.3f} evolved: {dist_evolved:.3f}")
    for f in DIST_MATCH_FEATURES:
        print(f"    {f}: default z={z_default[f]:+.2f}  evolved z={z_evolved[f]:+.2f}")

    var_default, lr_default = variance_penalty(default_final, human_stats, per_feature=True)
    var_evolved, lr_evolved = variance_penalty(evolved_final, human_stats, per_feature=True)
    print(f"[adversarial_loop] mean squared log-variance-ratio vs human - default: {var_default:.3f} evolved: {var_evolved:.3f}")
    for f in DIST_MATCH_FEATURES:
        print(f"    {f}: default log-ratio={lr_default[f]:+.2f}  evolved log-ratio={lr_evolved[f]:+.2f}")

    cov_default, diffs_default = covariance_penalty(default_final, human_corr, per_pair=True)
    cov_evolved, diffs_evolved = covariance_penalty(evolved_final, human_corr, per_pair=True)
    print(f"[adversarial_loop] mean squared correlation-matrix diff vs human - default: {cov_default:.3f} evolved: {cov_evolved:.3f}")
    for f1, f2 in diffs_default:
        print(f"    {f1}/{f2}: default diff={diffs_default[(f1, f2)]:+.2f}  evolved diff={diffs_evolved[(f1, f2)]:+.2f}")

    tail_default, taild_default = tail_penalty(default_final, human_tail, per_feature=True)
    tail_evolved, taild_evolved = tail_penalty(evolved_final, human_tail, per_feature=True)
    print(f"[adversarial_loop] mean squared tail-quantile diff vs human - default: {tail_default:.3f} evolved: {tail_evolved:.3f}")
    for f in DIST_MATCH_FEATURES:
        print(f"    {f}: default tail-diff={taild_default[f]:+.2f}  evolved tail-diff={taild_evolved[f]:+.2f}")

    # Plot history: fitness/accuracy per generation, and the per-epoch
    # convergence trend (the actual stopping criterion) side by side.
    rounds = [h["round"] for h in history]
    fitnesses = [h["best_fitness"] for h in history]
    accs = [h["measured_accuracy"] for h in history]
    conv_epochs = [c["epoch"] for c in convergence_history]
    conv_z2 = [c["z2"] for c in convergence_history]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(rounds, fitnesses, "o-", color="tab:blue", label="fitness (lower=better)")
    ax1.set_xlabel("round (generation)")
    ax1.set_ylabel("fitness", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(rounds, accs, "s--", color="tab:red", label="ensemble accuracy")
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("ensemble accuracy", color="tab:red")
    ax1.set_title("Per-generation search trace")

    ax3.plot(conv_epochs, conv_z2, "d-", color="tab:green", label="dist_z2 (proxy)")
    ax3.axhline(CONVERGENCE_Z2, color="tab:green", linestyle=":", linewidth=1)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("running-best dist_z2 vs human", color="tab:green")
    ax4 = ax3.twinx()
    if holdout_acc_history:
        hac_epochs = [h["epoch"] for h in holdout_acc_history]
        hac_vals = [h["acc"] for h in holdout_acc_history]
        ax4.plot(hac_epochs, hac_vals, "^-", color="tab:purple", label="holdout accuracy (real target)")
    ax4.axhline(ACCURACY_TARGET, color="tab:purple", linestyle=":", linewidth=1)
    ax4.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax4.set_ylabel("holdout accuracy", color="tab:purple")
    ax3.set_title(f"Convergence trend ({stop_reason})", fontsize=9)

    fig.suptitle("Adversarial loop v14: ensemble + mean/variance/covariance/tail-quantile matching, loop-until-converged")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "adversarial_history.png", dpi=120)
    plt.close(fig)

    evolved_full_config = {**MOTOR_SYNERGY_DEFAULTS, **best_cfg_overall}
    (RESULTS_DIR / "evolved_motor_synergy_config.json").write_text(json.dumps(evolved_full_config, indent=2))

    report = [
        "# Adversarial loop v14 results (ensemble + mean, variance, covariance, AND tail-quantile matching, loop-until-converged)",
        "",
        f"Feature set used for the search: `shape_only` ({len(FEATURE_NAMES)} features). "
        f"Distribution-matching target features ({len(DIST_MATCH_FEATURES)}): {', '.join(DIST_MATCH_FEATURES)}.",
        "",
        f"Ran {n_epochs_run} epochs ({n_epochs_run * GENERATIONS_PER_EPOCH} generations). "
        f"**Stop reason: {stop_reason}**",
        "",
        f"Initial (default config): ensemble accuracy mean={baseline_acc:.3f}, "
        f"worst member={baseline_acc_worst:.3f}, dist_z2={baseline_z2:.3f}, var2={baseline_var2:.3f}, "
        f"cov2={baseline_cov2:.3f}, tail2={baseline_tail2:.3f}",
        f"Best fitness found during search: {best_fitness_overall:.3f}",
        "",
        "## Independent final validation (fresh 4-model ensemble, never used during search)",
        "",
        "| config | accuracy (ensemble mean) | accuracy (worst member) | mean sq. z-score (means) | mean sq. log-ratio (spread) | mean sq. corr diff (joint) | mean sq. tail diff (shape) |",
        "|---|---|---|---|---|---|---|",
        f"| default | {acc_default:.3f} | {acc_default_worst:.3f} | {dist_default:.3f} | {var_default:.3f} | {cov_default:.3f} | {tail_default:.3f} |",
        f"| evolved | {acc_evolved:.3f} | {acc_evolved_worst:.3f} | {dist_evolved:.3f} | {var_evolved:.3f} | {cov_evolved:.3f} | {tail_evolved:.3f} |",
        "",
        "### Per-feature mean z-score (default vs evolved, vs human mean/std)",
        "",
        "| feature | default z | evolved z |",
        "|---|---|---|",
    ]
    for f in DIST_MATCH_FEATURES:
        report.append(f"| {f} | {z_default[f]:+.2f} | {z_evolved[f]:+.2f} |")
    report += [
        "",
        "### Per-feature spread log-ratio (log(candidate_std / human_std) - 0 = matched, "
        "negative = too tight/uniform, positive = too spread out)",
        "",
        "| feature | default log-ratio | evolved log-ratio |",
        "|---|---|---|",
    ]
    for f in DIST_MATCH_FEATURES:
        report.append(f"| {f} | {lr_default[f]:+.2f} | {lr_evolved[f]:+.2f} |")
    report += [
        "",
        "### Per-pair correlation difference (human_corr - candidate_corr; 0 = matched)",
        "",
        "| feature pair | default diff | evolved diff |",
        "|---|---|---|",
    ]
    for f1, f2 in diffs_default:
        report.append(f"| {f1}/{f2} | {diffs_default[(f1, f2)]:+.2f} | {diffs_evolved[(f1, f2)]:+.2f} |")
    report += [
        "",
        "### Per-feature tail-quantile difference (mean sq. normalized diff over p75/90/95/99; 0 = matched)",
        "",
        "| feature | default diff | evolved diff |",
        "|---|---|---|",
    ]
    for f in DIST_MATCH_FEATURES:
        report.append(f"| {f} | {taild_default[f]:+.2f} | {taild_evolved[f]:+.2f} |")
    report += [
        "",
        "## Evolved config (deltas from default)",
        "",
        "| param | default | evolved |",
        "|---|---|---|",
    ]
    for k in CONFIG_BOUNDS:
        report.append(f"| {k} | {MOTOR_SYNERGY_DEFAULTS[k]:.4f} | {evolved_full_config[k]:.4f} |")
    report += [
        "",
        "Full evolved config: `evolved_motor_synergy_config.json` (JSON keys match "
        "needaimbot/mouse/motor_synergy.hpp's `struct config` field names 1:1 - can be "
        "translated directly to `flick_*` JSON keys in simple_config.json).",
        "",
        "![history](adversarial_history.png)",
    ]
    (RESULTS_DIR / "adversarial_report.md").write_text("\n".join(report))
    print(f"[adversarial_loop] wrote {RESULTS_DIR / 'adversarial_report.md'}")


if __name__ == "__main__":
    main()
