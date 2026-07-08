#!/usr/bin/env python3
"""Feature extraction shared by every class (human / naive_bot /
motor_synergy_bot). Features are inspired by the papers cited in README.md
(BeCAPTCHA-Mouse's sigma-lognormal-derived human motor signature, Fitts'
Law, Harris & Wolpert signal-dependent noise) rather than a full iterative
sigma-lognormal decomposition (that fit itself - XZERO/iDeLog - is a
research project of its own; out of scope here, see README).

Every function takes a single movement as a list of (x, y, t_ms) points
(same shape as motor_synergy::trajectory_point) and returns plain floats -
no dependency on where the movement came from.

SAMPLING-RATE CONFOUND FIX (see README.md "Known confound"): Balabit's human
data is RDP-captured at a much coarser, more irregular rate (median ~109ms
between samples) than either synthetic bot class (~7-8ms, a native ~125Hz
mouse polling rate). Any feature built on numerical differentiation
amplifies noise by roughly 1/dt per derivative order - jerk is a THIRD
derivative, so a 1/dt^3 term - which made jerk/curvature/tremor massively
overstated for the densely-sampled synthetic classes purely as a capture-
pipeline artifact, not a genuine movement-quality difference (confirmed
empirically: an adversarial search against these features found configs
that fooled one detector instance but didn't generalize to a fresh one -
see results/adversarial_report.md). Every feature below except the
`sample_interval_*` pair (which deliberately measures the RAW, pre-resample
capture cadence) is now computed on a COMMON time grid every class is
resampled to first, so nobody's jerk/curvature/tremor is an artifact of
"how densely was this sampled."
"""
import math

import numpy as np

FEATURE_NAMES = [
    "distance",
    "movement_time",
    "mean_speed",
    "peak_speed",
    "time_to_peak_ratio",
    "num_submovements",
    "path_efficiency",
    "curvature_rms",
    "velocity_skewness",
    "velocity_kurtosis",
    "jerk_rms",
    "jerk_max",
    "tremor_band_energy_ratio",
    "sample_interval_mean",
    "sample_interval_cv",
    "sdn_correlation",
]

# Common resampling grid - a deliberate compromise, not a perfect fix: going
# all the way to Balabit's own ~109ms median cadence would leave most
# sub-300ms flick movements with only 2-3 points (no shape signal at all).
# 40ms (25Hz) removes the vast majority of the differentiation-noise
# disparity (every class is evaluated at the IDENTICAL rate, so there's no
# disparity left at all) while still giving a ~150ms movement ~4-5 samples.
COMMON_DT_MS = 40.0
MIN_RESAMPLED_POINTS = 4


def _arrays(points):
    pts = np.asarray(points, dtype=float)
    return pts[:, 0], pts[:, 1], pts[:, 2]


def _resample_to_grid(x, y, t, dt_ms=COMMON_DT_MS):
    duration = t[-1] - t[0]
    if duration <= 0:
        return x, y, t
    n = max(int(duration / dt_ms) + 1, 2)
    t_grid = np.linspace(t[0], t[-1], n)
    x_grid = np.interp(t_grid, t, x)
    y_grid = np.interp(t_grid, t, y)
    return x_grid, y_grid, t_grid


def extract_features(points):
    """points: list of (x, y, t_ms), at least 4 samples (caller should have
    already applied the same MIN_POINTS/MIN_DISTANCE filter used when the
    movement was parsed/generated)."""
    x_raw, y_raw, t_raw = _arrays(points)

    # Raw capture-cadence stats - deliberately NOT resampled. This is the one
    # pair of features meant to measure the capture pipeline's own fingerprint
    # (bots are often suspiciously uniform; a human/gamma dt has real spread).
    dt_raw = np.diff(t_raw)
    valid_dt = dt_raw[dt_raw > 0]
    sample_interval_mean = float(np.mean(valid_dt)) if len(valid_dt) else 0.0
    sample_interval_cv = (
        float(np.std(valid_dt) / np.mean(valid_dt)) if len(valid_dt) and np.mean(valid_dt) > 0 else 0.0
    )

    distance = math.hypot(x_raw[-1] - x_raw[0], y_raw[-1] - y_raw[0])
    movement_time = t_raw[-1] - t_raw[0]

    # Everything else: resampled to the common grid first (see module
    # docstring) so jerk/curvature/tremor/speed are comparable across classes.
    x, y, t = _resample_to_grid(x_raw, y_raw, t_raw)

    dt = np.diff(t)
    dt_safe = np.where(dt > 0, dt, np.nan)
    dxy = np.hypot(np.diff(x), np.diff(y))
    speed = dxy / dt_safe * 1000.0  # px/s; nan where dt==0
    speed = np.nan_to_num(speed, nan=0.0)

    path_length = float(np.sum(dxy))

    mean_speed = float(np.mean(speed)) if len(speed) else 0.0
    peak_idx = int(np.argmax(speed)) if len(speed) else 0
    peak_speed = float(speed[peak_idx]) if len(speed) else 0.0
    time_to_peak_ratio = float(t[peak_idx + 1] / movement_time) if movement_time > 0 and len(speed) else 0.0

    # Submovement count: peaks above 15% of max in the speed signal - same
    # logic as motor_synergy::compute_metrics.
    threshold = peak_speed * 0.15
    peaks = 0
    for i in range(1, len(speed) - 1):
        if speed[i] > threshold and speed[i] > speed[i - 1] and speed[i] > speed[i + 1]:
            peaks += 1
    num_submovements = max(peaks, 1)

    # Mathematically bounded at 1.0 (a straight line is always the shortest
    # path between two points) - clipped defensively since a resampling/
    # timestamp edge case (e.g. a garbage sentinel coordinate desyncing
    # `distance`'s raw endpoints from `path_length`'s resampled ones) could
    # otherwise blow this up arbitrarily and corrupt any std/variance target
    # built on it.
    path_efficiency = min(distance / path_length, 1.0) if path_length > 0 else 1.0

    # Curvature: perpendicular deviation from the straight line start->end,
    # RMS over all samples (normalized by distance so it's scale-free).
    if distance > 1e-6:
        ux, uy = (x[-1] - x[0]) / distance, (y[-1] - y[0]) / distance
        rel_x, rel_y = x - x[0], y - y[0]
        perp = rel_x * (-uy) + rel_y * ux
        curvature_rms = float(np.sqrt(np.mean(perp ** 2)) / distance)
    else:
        curvature_rms = 0.0

    # Velocity-profile shape: human movements are right-skewed (fast rise,
    # slow decay), matching the lognormal primary submovement.
    if len(speed) >= 3 and np.std(speed) > 1e-9:
        s_mean, s_std = np.mean(speed), np.std(speed)
        velocity_skewness = float(np.mean(((speed - s_mean) / s_std) ** 3))
        velocity_kurtosis = float(np.mean(((speed - s_mean) / s_std) ** 4) - 3.0)
    else:
        velocity_skewness = 0.0
        velocity_kurtosis = 0.0

    # Jerk (rate of change of acceleration) - humans show smoother jerk than
    # a naive constant-velocity-plus-jitter bot. Computed on the common grid
    # (see module docstring) so this isn't just measuring sample density.
    if len(speed) >= 3:
        accel = np.diff(speed) / dt_safe[:-1]
        accel = np.nan_to_num(accel, nan=0.0)
        if len(accel) >= 2:
            jerk = np.diff(accel) / dt_safe[:-2]
            jerk = np.nan_to_num(jerk, nan=0.0)
            jerk_rms = float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) else 0.0
            jerk_max = float(np.max(np.abs(jerk))) if len(jerk) else 0.0
        else:
            jerk_rms = jerk_max = 0.0
    else:
        jerk_rms = jerk_max = 0.0

    # Tremor: FFT of the path detrended against the straight-line fit,
    # energy fraction in the 4-12 Hz physiological tremor band (Harris &
    # Wolpert; also the band motor_synergy's own tremor term samples from).
    # NOTE: 4-12Hz needs samples well above Nyquist for 12Hz (>=24Hz); our
    # 25Hz common grid is right at that edge, so this feature is necessarily
    # coarse - kept for completeness/comparability, not precision.
    tremor_band_energy_ratio = _tremor_band_energy_ratio(x, y, t)

    # Signal-dependent-noise check: correlation between local speed and the
    # magnitude of the perpendicular jitter residual - Harris & Wolpert's
    # noise-scales-with-command-magnitude principle.
    sdn_correlation = _sdn_correlation(x, y, speed)

    return {
        "distance": distance,
        "movement_time": movement_time,
        "mean_speed": mean_speed,
        "peak_speed": peak_speed,
        "time_to_peak_ratio": time_to_peak_ratio,
        "num_submovements": float(num_submovements),
        "path_efficiency": path_efficiency,
        "curvature_rms": curvature_rms,
        "velocity_skewness": velocity_skewness,
        "velocity_kurtosis": velocity_kurtosis,
        "jerk_rms": jerk_rms,
        "jerk_max": jerk_max,
        "tremor_band_energy_ratio": tremor_band_energy_ratio,
        "sample_interval_mean": sample_interval_mean,
        "sample_interval_cv": sample_interval_cv,
        "sdn_correlation": sdn_correlation,
    }


def _tremor_band_energy_ratio(x, y, t, band=(4.0, 12.0)):
    n = len(t)
    if n < MIN_RESAMPLED_POINTS:
        return 0.0
    duration_s = (t[-1] - t[0]) / 1000.0
    if duration_s <= 0:
        return 0.0
    fs = 1000.0 / COMMON_DT_MS  # already on the common uniform grid

    # Detrend against the straight-line path so only the residual (curvature
    # + tremor + noise) contributes to the spectrum, not the bulk motion.
    x_trend = np.linspace(x[0], x[-1], n)
    y_trend = np.linspace(y[0], y[-1], n)
    resid = np.hypot(x - x_trend, y - y_trend)
    if np.allclose(resid, 0.0):
        return 0.0

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(resid - np.mean(resid))) ** 2
    total = float(np.sum(power[1:]))  # drop DC
    if total <= 1e-12:
        return 0.0
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = float(np.sum(power[band_mask]))
    return band_power / total


def _sdn_correlation(x, y, speed):
    if len(speed) < 4:
        return 0.0
    # Residual = deviation from a smoothed (moving-average) path - the
    # "noise" component riding on top of the intended motion.
    window = min(5, len(x) - (len(x) % 2 == 0))
    window = max(window, 3)
    if window >= len(x):
        return 0.0
    kernel = np.ones(window) / window
    x_smooth = np.convolve(x, kernel, mode="valid")
    y_smooth = np.convolve(y, kernel, mode="valid")
    offset = window // 2
    x_trim = x[offset: offset + len(x_smooth)]
    y_trim = y[offset: offset + len(y_smooth)]
    residual = np.hypot(x_trim - x_smooth, y_trim - y_smooth)

    speed_trim = speed[: len(residual)] if len(speed) >= len(residual) else np.pad(speed, (0, len(residual) - len(speed)))
    if len(residual) < 3 or np.std(residual) < 1e-9 or np.std(speed_trim) < 1e-9:
        return 0.0
    corr = float(np.corrcoef(speed_trim, residual)[0, 1])
    return 0.0 if math.isnan(corr) else corr
