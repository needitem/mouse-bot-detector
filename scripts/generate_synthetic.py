#!/usr/bin/env python3
"""Generates the two synthetic "bot" classes for the detector, matched to the
REAL distance distribution observed in the parsed human movements (so the
classifier can't just learn "distance" as a shortcut instead of movement
shape - the same methodological point BeCAPTCHA-Mouse makes about
synthesis-for-evaluation).

- naive_bot: constant-velocity straight line + small per-step Gaussian
  jitter. No Fitts timing, no submovements, no curvature - a stand-in for a
  naive aimbot/macro. Gives the detector (and the report) a sanity-check
  baseline: "can it even tell this obviously fake movement apart?"

- motor_synergy_bot: a faithful pure-Python port of
  needaimbot/mouse/motor_synergy.hpp's generate() - Fitts-timed primary
  submovement + 0-2 corrections via lognormal CDFs, direction-dependent
  curvature, Ornstein-Uhlenbeck drift, velocity-modulated tremor,
  signal-dependent noise. Same formulas/parameter names/defaults as the C++
  version; ported to pure Python/numpy since offline training-data
  generation doesn't need the GPU split that matters for real-time use.
  Two fields (`mt_noise_sigma`, `curvature_noise_sigma`) are NOT in the C++
  header yet - they pull out constants that used to be hardcoded inside
  generate(), added after diagnosing that motor_synergy_bot's movement-to-
  movement VARIANCE (not just its mean) was far too low vs human data. See
  results/adversarial_report.md and README.md "Adversarial loop".
"""
import argparse
import json
import math
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
HUMAN_PATHS = [
    DATA_DIR / "processed" / "human_movements.jsonl",
    DATA_DIR / "processed" / "human_movements_web.jsonl",
]
OUT_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# motor_synergy config defaults - mirrors needaimbot/mouse/motor_synergy.hpp
# `struct config` field-for-field.
# ---------------------------------------------------------------------------
MOTOR_SYNERGY_DEFAULTS = dict(
    fitts_a=50.0,
    fitts_b=150.0,
    target_width=20.0,
    undershoot_min=0.92,
    undershoot_max=0.97,
    peak_time_ratio=0.35,
    primary_sigma_min=0.18,
    primary_sigma_max=0.28,
    overshoot_prob=0.15,
    overshoot_min=1.02,
    overshoot_max=1.08,
    correction_sigma_min=0.12,
    correction_sigma_max=0.20,
    second_correction_prob=0.25,
    curvature_scale=0.025,
    ou_theta=3.5,
    ou_sigma=1.2,
    tremor_freq_min=8.0,
    tremor_freq_max=12.0,
    tremor_amp_min=0.15,
    tremor_amp_max=0.55,
    sdn_k=0.04,
    sample_dt_mean=7.8,
    gamma_shape=3.5,
    # --- Not in needaimbot/mouse/motor_synergy.hpp (yet) ---
    # Diagnosed via a correlation/variance comparison against human data:
    # movement-to-movement VARIANCE of curvature_rms/jerk/path_efficiency in
    # motor_synergy_bot output was only 2-10% of human's (every generated
    # movement looks nearly identical), even after mean-matching every
    # feature near-perfectly - fooling a detector needs matching spread, not
    # just the average. These were hardcoded constants in generate(); pulled
    # out here so the adversarial search can widen them. If this helps, port
    # them back into the C++ struct config the same way.
    mt_noise_sigma=0.08,          # was hardcoded in exp(normal(0, 0.08)) for movement time
    curvature_noise_sigma=1.0,    # was hardcoded in curv_amp's normal(0, 1.0) multiplier
    # curvature_rms's variance stayed the single worst-matched axis even
    # after curvature_noise_sigma became tunable (search settled at ~63% of
    # its allowed range, not the ceiling - genuinely balancing against other
    # features, not blocked by a bound). Root cause: a single Gaussian scale
    # factor can't set "how curvy movements are on average" and "how much
    # that varies flick-to-flick" independently - both come from the same
    # sigma. This adds a SEPARATE per-movement multiplicative "curviness
    # style" factor (lognormal, so it's positive and right-skewed like real
    # inter-trial motor variability) - 0 keeps today's behavior exactly
    # (multiplier's variance -> 0 as this -> 0).
    curvature_style_sigma=0.0,
    # Same bug, different axis: time_to_peak_ratio's spread was stuck at
    # log-ratio ~-1.0 to -1.3 (bot's variance ~30% of human's) across an
    # entire run that otherwise moved everything else - the per-movement
    # jitter window around peak_time_ratio was hardcoded to +-0.03
    # regardless of what peak_time_ratio itself got tuned to, capping how
    # much any single movement's peak timing can differ from the mean. Now
    # the std of a Gaussian offset (see motor_synergy_generate) rather than
    # the half-width of a uniform one, after v5's search found the plateau
    # persisted regardless of this value - a uniform jitter structurally
    # cannot produce human-like outlier peak timings no matter how wide.
    peak_time_jitter=0.03,
    # v5-audit: the same bounded-uniform-can't-reach-the-tail bug found in
    # peak_time_jitter almost certainly also caps the OTHER hardcoded windows
    # in generate() - the reach fraction (overshoot/undershoot) and every
    # correction-submovement timing/amount draw were all fixed uniform
    # ranges with no exposed scale at all. Each is now a Gaussian offset
    # around the same original center, scaled by one of these three factors
    # (1.0 = same std as the original uniform window's half-width, so
    # defaults are a close behavioral match, not a free reset).
    reach_jitter=1.0,
    correction_timing_jitter=1.0,
    correction_amount_jitter=1.0,
    # Gaussian-scale-mixture SDN noise (see motor_synergy_generate) - 0/1.0
    # exactly reproduces the old plain-Gaussian behavior. Empirically this
    # one barely moves jerk_rms (per-sample noise washes out under 40ms
    # resampling before it's even measured) - kept since it's harmless at its
    # default, but ou_jump_prob/scale below is the one that actually works.
    sdn_burst_prob=0.0,
    sdn_burst_scale=1.0,
    # Per-movement OU jump (see motor_synergy_generate) - 0 reproduces the
    # old behavior exactly. A quick standalone test (not the full detector
    # loop) found ou_jump_prob=0.1/ou_jump_scale=100 alone reproduces human
    # jerk_rms's skew (3.85 vs human's 3.83) and kurtosis (24.1 vs 23.4)
    # almost exactly - defaults left at 0 so the adversarial search finds its
    # own value rather than starting pre-tuned.
    ou_jump_prob=0.0,
    ou_jump_scale=0.0,
)


def _direction_factor(angle):
    # vertical movements produce more curvature due to wrist/forearm geometry
    sa = abs(math.sin(angle))
    ca = abs(math.cos(angle))
    return 0.5 + 0.8 * sa - 0.15 * ca


def _lognormal_cdf(t, t0, mu, sigma):
    if t <= t0:
        return 0.0
    z = (math.log(t - t0) - mu) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _lognormal_pdf(t, t0, mu, sigma):
    if t <= t0:
        return 0.0
    dt = t - t0
    z = (math.log(dt) - mu) / sigma
    return 1.0 / (sigma * math.sqrt(2.0 * math.pi) * dt) * math.exp(-0.5 * z * z)


def _jittered(rng, center, half_width, scale, lo, hi):
    """Gaussian offset around `center` (std = half_width * scale), clipped to
    [lo, hi] for physical sanity only - not to bound the search the way the
    original hardcoded `rng.uniform(center-half_width, center+half_width)`
    windows did. Same fix as peak_time_jitter (see motor_synergy_generate):
    a bounded uniform draw has a hard cutoff no matter how wide it's tuned,
    so it can't produce the occasional real outlier a Gaussian tail can."""
    return min(max(center + rng.gauss(0.0, half_width * scale), lo), hi)


def motor_synergy_generate(x0, y0, x1, y1, cfg=None, rng=None):
    """Direct port of motor_synergy::generate(). Returns [(x, y, t_ms), ...]."""
    cfg = {**MOTOR_SYNERGY_DEFAULTS, **(cfg or {})}
    rng = rng or random.Random()

    dx, dy = x1 - x0, y1 - y0
    distance = math.hypot(dx, dy)
    direction = math.atan2(dy, dx)

    if distance < 1.0:
        return [(x0, y0, 0.0), (x1, y1, 50.0)]

    tx, ty = dx / distance, dy / distance
    nx, ny = -ty, tx

    idx = math.log2(distance / cfg["target_width"] + 1.0)
    mt = (cfg["fitts_a"] + cfg["fitts_b"] * idx) * math.exp(rng.gauss(0.0, cfg["mt_noise_sigma"]))
    mt = max(mt, 80.0)

    overshoot = rng.uniform(0.0, 1.0) < cfg["overshoot_prob"]
    if overshoot:
        reach_center = (cfg["overshoot_min"] + cfg["overshoot_max"]) / 2.0
        reach_half = (cfg["overshoot_max"] - cfg["overshoot_min"]) / 2.0
    else:
        reach_center = (cfg["undershoot_min"] + cfg["undershoot_max"]) / 2.0
        reach_half = (cfg["undershoot_max"] - cfg["undershoot_min"]) / 2.0
    reach = _jittered(rng, reach_center, reach_half, cfg["reach_jitter"], 0.5, 1.5)

    primary_D = distance * reach
    primary_sigma = rng.uniform(cfg["primary_sigma_min"], cfg["primary_sigma_max"])
    # Gaussian, not uniform: a bounded uniform jitter has a hard cutoff at
    # +-peak_time_jitter, so no matter how wide it's tuned it can never
    # produce the occasional far-outlier peak timing real human submovements
    # show (v5's adversarial search found time_to_peak_ratio's variance stuck
    # at ~30% of human's regardless of jitter value - a distribution-shape
    # ceiling, not a bound-too-narrow problem). A Gaussian has the same
    # "typical" spread for the same sigma value but an unbounded tail, so
    # occasional movements land far from peak_time_ratio - clipped only for
    # physical sanity (peak has to land strictly inside the movement).
    peak_frac = min(max(cfg["peak_time_ratio"] + rng.gauss(0.0, cfg["peak_time_jitter"]), 0.05), 0.95)
    peak_t = mt * peak_frac
    primary_mu = math.log(peak_t) + primary_sigma * primary_sigma

    corrections = []
    remaining = distance - primary_D
    if abs(remaining) > 0.5:
        direction_sign = 1.0 if remaining > 0.0 else -1.0
        cD = abs(remaining) * _jittered(rng, 0.95, 0.07, cfg["correction_amount_jitter"], 0.5, 1.5)
        cS = rng.uniform(cfg["correction_sigma_min"], cfg["correction_sigma_max"])
        cPeak = mt * _jittered(rng, 0.15, 0.03, cfg["correction_timing_jitter"], 0.05, 0.4)
        corrections.append(dict(
            D=cD, t0=mt * _jittered(rng, 0.615, 0.065, cfg["correction_timing_jitter"], 0.3, 0.95),
            mu=math.log(cPeak) + cS * cS, sigma=cS,
            dir_x=tx * direction_sign, dir_y=ty * direction_sign,
        ))

        left = remaining - cD * direction_sign
        if abs(left) > 0.3 and rng.uniform(0.0, 1.0) < cfg["second_correction_prob"]:
            d2 = 1.0 if left > 0.0 else -1.0
            cD2 = abs(left) * _jittered(rng, 0.95, 0.10, cfg["correction_amount_jitter"], 0.5, 1.5)
            cS2 = rng.uniform(0.10, 0.16)
            cP2 = mt * _jittered(rng, 0.10, 0.02, cfg["correction_timing_jitter"], 0.03, 0.3)
            corrections.append(dict(
                D=cD2, t0=mt * _jittered(rng, 0.83, 0.05, cfg["correction_timing_jitter"], 0.5, 0.98),
                mu=math.log(cP2) + cS2 * cS2, sigma=cS2,
                dir_x=tx * d2, dir_y=ty * d2,
            ))

    curvature_style = math.exp(rng.gauss(0.0, cfg["curvature_style_sigma"])) if cfg["curvature_style_sigma"] > 0 else 1.0
    curv_amp = (
        distance * cfg["curvature_scale"] * curvature_style
        * _direction_factor(direction) * rng.gauss(0.0, cfg["curvature_noise_sigma"])
    )

    tremor_freq = rng.uniform(cfg["tremor_freq_min"], cfg["tremor_freq_max"])
    tremor_amp = rng.uniform(cfg["tremor_amp_min"], cfg["tremor_amp_max"])
    tremor_phase_x = rng.uniform(0.0, 2.0 * math.pi)
    tremor_phase_y = rng.uniform(0.0, 2.0 * math.pi)

    total_t = mt * 1.15
    g_scale = cfg["sample_dt_mean"] / cfg["gamma_shape"]

    times = [0.0]
    t = 0.0
    while t < total_t and len(times) < 512:
        dt = min(max(rng.gammavariate(cfg["gamma_shape"], g_scale), 2.0), 25.0)
        t += dt
        if t <= total_t + 15.0:
            times.append(t)

    # OU jump (per-MOVEMENT, not per-sample): a permutation-importance check
    # found jerk_rms is the single most detector-relied-on feature, and human
    # jerk_rms is heavy-tailed (skew 3.83, kurtosis 23.4) - most movements
    # have small jerk, a rare few have much larger jerk. A per-SAMPLE noise
    # boost (tried first, on sdn_x/y) can't produce that: with dozens of
    # samples per movement it averages out across the movement (the RMS
    # feature converges to a similar typical value every time) AND gets
    # smoothed away by features.py's 40ms resampling grid entirely, since raw
    # per-sample noise lives at a much finer timescale. A rare per-movement
    # event added to the OU drift (which persists across many samples and
    # DOES survive resampling) is what actually reproduces the shape: with
    # probability `ou_jump_prob`, exactly one randomly-timed, randomly-
    # directed impulse of magnitude ~N(0, ou_jump_scale) perturbs the OU
    # state once - most movements get none (0 by default reproduces the old
    # behavior exactly), a few get one, giving genuine between-movement
    # heavy-tailedness instead of within-movement noise that washes out.
    has_jump = rng.random() < cfg["ou_jump_prob"]
    jump_idx = rng.randrange(1, len(times)) if has_jump and len(times) > 1 else None
    if has_jump:
        jump_dir = rng.uniform(0.0, 2.0 * math.pi)
        jump_mag = rng.gauss(0.0, cfg["ou_jump_scale"])
        jump_x_val, jump_y_val = jump_mag * math.cos(jump_dir), jump_mag * math.sin(jump_dir)

    result = []
    ou_x = ou_y = 0.0
    for i, ti in enumerate(times):
        dt_ms = (ti - times[i - 1]) if i > 0 else cfg["sample_dt_mean"]
        dt_s = dt_ms / 1000.0

        s = _lognormal_cdf(ti, 0.0, primary_mu, primary_sigma)
        bx = x0 + tx * primary_D * s
        by = y0 + ty * primary_D * s
        curvature = 0.0
        if 0.0 < s < 1.0:
            v = s * s * (1.0 - s) * (1.0 - s) * (1.0 - s)
            curvature = v / (0.4 * 0.4 * 0.6 * 0.6 * 0.6)
        bx += nx * curv_amp * curvature
        by += ny * curv_amp * curvature

        speed = primary_D * _lognormal_pdf(ti, 0.0, primary_mu, primary_sigma)
        for c in corrections:
            cs = _lognormal_cdf(ti, c["t0"], c["mu"], c["sigma"])
            bx += c["dir_x"] * c["D"] * cs
            by += c["dir_y"] * c["D"] * cs
            speed += c["D"] * _lognormal_pdf(ti, c["t0"], c["mu"], c["sigma"])

        # OU drift (Euler-Maruyama) - sequential recurrence, same as the C++ host side.
        jump_x_i, jump_y_i = (jump_x_val, jump_y_val) if i == jump_idx else (0.0, 0.0)
        ou_x += -cfg["ou_theta"] * ou_x * dt_s + cfg["ou_sigma"] * math.sqrt(dt_s) * rng.gauss(0.0, 1.0) + jump_x_i
        ou_y += -cfg["ou_theta"] * ou_y * dt_s + cfg["ou_sigma"] * math.sqrt(dt_s) * rng.gauss(0.0, 1.0) + jump_y_i

        t_s = ti / 1000.0
        trem_mod = 1.0 / (1.0 + speed * 0.3)
        tr_x = tremor_amp * trem_mod * math.sin(2.0 * math.pi * tremor_freq * t_s + tremor_phase_x)
        tr_y = tremor_amp * trem_mod * math.sin(2.0 * math.pi * tremor_freq * t_s + tremor_phase_y)

        # Gaussian-scale-mixture ("contaminated normal"), not plain i.i.d.
        # Gaussian: a permutation-importance check found jerk_rms is the
        # single most detector-relied-on feature, and human jerk_rms is
        # heavy-tailed (skew 3.83, kurtosis 23.4) - most movements have small
        # jerk, a few have much larger jerk. Plain per-sample Gaussian noise
        # can't produce that shape (adding a skew-matching FITNESS penalty on
        # top of it was tried twice and failed - see README "v10/v11" - the
        # generator itself needed the actual capacity, not just pressure
        # toward one). With small probability `sdn_burst_prob` each sample is
        # a "burst" (a brief, larger involuntary micro-correction) scaled up
        # by `sdn_burst_scale`; both default to reproduce the old plain-
        # Gaussian behavior exactly (prob=0 -> never bursts).
        is_burst = rng.random() < cfg["sdn_burst_prob"]
        sdn_scale = cfg["sdn_burst_scale"] if is_burst else 1.0
        sdn_x = cfg["sdn_k"] * speed * sdn_scale * rng.gauss(0.0, 1.0)
        sdn_y = cfg["sdn_k"] * speed * sdn_scale * rng.gauss(0.0, 1.0)

        result.append((bx + ou_x + tr_x + sdn_x, by + ou_y + tr_y + sdn_y, ti))

    return result


def naive_bot_generate(x0, y0, x1, y1, sample_dt_mean=7.8, jitter_px=0.6, rng=None):
    """Constant-velocity straight line + small per-step Gaussian jitter only.
    No Fitts timing, no submovements, no curvature - the "obviously fake"
    baseline naive_bot class."""
    rng = rng or random.Random()
    dx, dy = x1 - x0, y1 - y0
    distance = math.hypot(dx, dy)
    if distance < 1.0:
        return [(x0, y0, 0.0), (x1, y1, 50.0)]

    speed_px_per_ms = rng.uniform(1.2, 2.2)  # roughly human-comparable transit speed
    total_t = distance / speed_px_per_ms

    times = [0.0]
    t = 0.0
    while t < total_t and len(times) < 512:
        t += max(rng.gauss(sample_dt_mean, sample_dt_mean * 0.1), 1.0)
        times.append(min(t, total_t))

    result = []
    for ti in times:
        frac = ti / total_t if total_t > 0 else 1.0
        bx = x0 + dx * frac + rng.gauss(0.0, jitter_px)
        by = y0 + dy * frac + rng.gauss(0.0, jitter_px)
        result.append((bx, by, ti))
    return result


def load_human_distances():
    distances = []
    for path in HUMAN_PATHS:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                pts = rec["points"]
                x0, y0, _ = pts[0]
                x1, y1, _ = pts[-1]
                distances.append(math.hypot(x1 - x0, y1 - y0))
    return distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--count", type=int, default=None,
                         help="Movements per synthetic class (default: match human count)")
    args = parser.parse_args()

    distances = load_human_distances()
    if not distances:
        print("[generate_synthetic] No parsed human movements found - run parse_balabit.py first.")
        return
    n = args.count or len(distances)

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, gen_fn in (("naive_bot", naive_bot_generate), ("motor_synergy_bot", motor_synergy_generate)):
        out_path = OUT_DIR / f"{name}_movements.jsonl"
        with open(out_path, "w") as out:
            for i in range(n):
                distance = rng.choice(distances)
                angle = rng.uniform(0.0, 2.0 * math.pi)
                x0, y0 = 0.0, 0.0
                x1, y1 = distance * math.cos(angle), distance * math.sin(angle)
                points = gen_fn(x0, y0, x1, y1, rng=rng)
                out.write(json.dumps({
                    "user": name,
                    "session": f"{name}_{i}",
                    "points": [[p[0], p[1], p[2]] for p in points],
                }) + "\n")
        print(f"[generate_synthetic] wrote {n} {name} movements -> {out_path}")


if __name__ == "__main__":
    main()
