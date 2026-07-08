#!/usr/bin/env python3
"""Feature-space ceiling check: how low could detector accuracy go if a
generator could sample the human feature JOINT distribution directly,
unconstrained by motor_synergy's parametric form?

This does NOT generate trajectories. It fits a Gaussian Mixture Model to the
real human feature vectors (the same `shape_only` 14 features the detector
trains on) and samples synthetic "feature vectors" straight from that learned
density, then evaluates the same ensemble-accuracy pipeline used everywhere
else in this project. If a generator that can match the joint feature
distribution as well as a GMM can still gets caught at meaningfully above
chance, that's strong evidence the ~85-92% floor motor_synergy's evolutionary
search has been hitting is a property of this detector/dataset/feature set,
not a limitation of motor_synergy's specific formula - i.e. no generator
(however sophisticated) is likely to reach much lower without a different
feature set or different training data. If the GMM ceiling comes in well
below what motor_synergy has achieved, that's evidence there's real headroom
being left on the table by the current parametric family.

Deliberately NOT the same question as "generate realistic trajectories" -
some GMM-sampled feature combinations may not correspond to any physically
achievable trajectory at all. That's fine for THIS purpose: it's a ceiling
on what feature-level matching alone could achieve, an upper bound, not a
proposal for how to actually generate movements.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adversarial_loop import load_human_pool, train_detector_ensemble, ensemble_accuracy, FEATURE_NAMES

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

HUMAN_SAMPLE_SIZE = 1200
FINAL_VALIDATION_SAMPLES = 800
N_ENSEMBLE = 4

# Physical bounds each feature can't cross, applied to GMM samples before
# they're used - a GMM is a smooth density and WILL occasionally propose
# values a real feature can never take (e.g. path_efficiency > 1 is
# mathematically impossible; num_submovements must be a positive integer).
FEATURE_CLIPS = {
    "distance": (1.0, None),
    "movement_time": (10.0, None),
    "mean_speed": (0.0, None),
    "peak_speed": (0.0, None),
    "time_to_peak_ratio": (0.0, 1.0),
    "num_submovements": (1.0, None),  # rounded separately below
    "path_efficiency": (0.0, 1.0),
    "curvature_rms": (0.0, None),
    "jerk_rms": (0.0, None),
    "jerk_max": (0.0, None),
    "tremor_band_energy_ratio": (0.0, 1.0),
    "sdn_correlation": (-1.0, 1.0),
}


def fit_gmm(human_df, n_components):
    scaler = StandardScaler()
    X = scaler.fit_transform(human_df[FEATURE_NAMES].to_numpy())
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0, max_iter=300)
    gmm.fit(X)
    return gmm, scaler


def sample_rows(gmm, scaler, n, seed):
    X, _ = gmm.sample(n)
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(n)]  # gmm.sample returns samples grouped by component - shuffle
    X = scaler.inverse_transform(X)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    for f, (lo, hi) in FEATURE_CLIPS.items():
        if lo is not None:
            df[f] = df[f].clip(lower=lo)
        if hi is not None:
            df[f] = df[f].clip(upper=hi)
    df["num_submovements"] = df["num_submovements"].round().clip(lower=1)
    return df.to_dict("records")


def main():
    print("[ceiling] loading human pool...")
    pool_rows, _ = load_human_pool(seed=0)
    a = HUMAN_SAMPLE_SIZE
    b = a + FINAL_VALIDATION_SAMPLES
    human_rows = pool_rows[:a]
    final_human_rows = pool_rows[a:b]
    human_df = pd.DataFrame(human_rows)

    # Model-selection: try a few component counts, pick by BIC (standard
    # practice - more components fit the training data better but risk
    # overfitting to it, BIC penalizes that).
    print("[ceiling] fitting GMMs (selecting n_components by BIC)...")
    X_scaled = StandardScaler().fit_transform(human_df[FEATURE_NAMES].to_numpy())
    best_bic, best_k = None, None
    for k in (5, 10, 15, 20, 30):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0, max_iter=300)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        print(f"[ceiling]   n_components={k}: BIC={bic:.1f}")
        if best_bic is None or bic < best_bic:
            best_bic, best_k = bic, k
    print(f"[ceiling] selected n_components={best_k}")

    gmm, scaler = fit_gmm(human_df, best_k)

    print(f"[ceiling] sampling {HUMAN_SAMPLE_SIZE} synthetic feature vectors from the fitted GMM...")
    bot_rows = sample_rows(gmm, scaler, HUMAN_SAMPLE_SIZE, seed=1)

    print("[ceiling] training initial ensemble (human vs GMM-sampled)...")
    ensemble = train_detector_ensemble(human_rows, bot_rows, seed_base=0)
    acc = ensemble_accuracy(ensemble, human_rows, bot_rows)
    acc_worst = ensemble_accuracy(ensemble, human_rows, bot_rows, reduce="worst")
    print(f"[ceiling] same-split ensemble accuracy: mean={acc:.3f} worst={acc_worst:.3f}")

    print("[ceiling] independent final validation (fresh ensemble, held-out human split)...")
    final_bot_rows = sample_rows(gmm, scaler, FINAL_VALIDATION_SAMPLES, seed=2)
    half = len(final_human_rows) // 2
    fresh_ensemble = train_detector_ensemble(
        final_human_rows[:half], final_bot_rows[:len(final_bot_rows) // 2], seed_base=9000
    )
    acc_final = ensemble_accuracy(fresh_ensemble, final_human_rows[half:], final_bot_rows[len(final_bot_rows) // 2:])
    acc_final_worst = ensemble_accuracy(
        fresh_ensemble, final_human_rows[half:], final_bot_rows[len(final_bot_rows) // 2:], reduce="worst"
    )
    print(f"[ceiling] independent final validation accuracy: mean={acc_final:.3f} worst={acc_final_worst:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "n_components": best_k,
        "same_split_acc_mean": acc,
        "same_split_acc_worst": acc_worst,
        "final_validation_acc_mean": acc_final,
        "final_validation_acc_worst": acc_final_worst,
    }
    (RESULTS_DIR / "feature_space_ceiling.json").write_text(json.dumps(report, indent=2))
    print(f"[ceiling] wrote {RESULTS_DIR / 'feature_space_ceiling.json'}")


if __name__ == "__main__":
    main()
