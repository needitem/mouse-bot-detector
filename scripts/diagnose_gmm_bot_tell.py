#!/usr/bin/env python3
"""Diagnoses what's actually driving the ~0.86-0.89 ceiling found by 3
rounds of alternating co-evolution (see co_evolution_loop.py /
results/coevolution_progress.json). Runs permutation_importance using the
final round's freshly-tuned detector ensemble hyperparameters
(results/ensemble_hyperparams.json) against the final round's generator
output (data/processed/gmm_hybrid_bot_coevo_movements.jsonl) - which
shape_only feature(s) is the generator still failing to match, and by how
much (human vs bot mean/std z-gap)? Answers whether the gap is one or two
fixable tells (a specific noise/shape mismatch) or spread thin across many
features (a structural ceiling of the GMM-shape + simple-noise generator
family itself).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_detector import SHAPE_ONLY_FEATURES, MAX_PER_CLASS, RANDOM_STATE
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from features import extract_features

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

ENSEMBLE_CLASSES = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "HistGradientBoosting": HistGradientBoostingClassifier,
}


def load_class(filenames):
    rows = []
    for filename in filenames:
        path = DATA_DIR / filename
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if len(rec["points"]) < 4:
                    continue
                rows.append(extract_features(rec["points"]))
    return rows


def main():
    print("[diagnose] loading human + gmm_hybrid_bot_coevo (round 3 final generator)...")
    human_rows = load_class(["human_movements.jsonl", "human_movements_web.jsonl"])
    bot_rows = load_class(["gmm_hybrid_bot_coevo_movements.jsonl"])
    n = min(MAX_PER_CLASS, len(human_rows), len(bot_rows))
    rng = np.random.default_rng(0)
    human_rows = [human_rows[i] for i in rng.choice(len(human_rows), size=n, replace=False)]
    bot_rows = [bot_rows[i] for i in rng.choice(len(bot_rows), size=n, replace=False)]
    df = pd.DataFrame(human_rows + bot_rows)
    y = np.array([0] * n + [1] * n)
    X = df[SHAPE_ONLY_FEATURES].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    ensemble_params = json.loads((RESULTS_DIR / "ensemble_hyperparams.json").read_text())
    print(f"[diagnose] using round-3 tuned hyperparams: {ensemble_params}")

    importances_per_model = {}
    for name, kwargs in ensemble_params.items():
        model = ENSEMBLE_CLASSES[name](random_state=RANDOM_STATE, **kwargs)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
        importances_per_model[name] = perm.importances_mean
        print(f"[diagnose] {name}: held-out acc={acc:.3f}")

    avg_importance = np.mean(list(importances_per_model.values()), axis=0)
    order = np.argsort(avg_importance)[::-1]

    human_df = df.iloc[:n]
    bot_df = df.iloc[n:]

    print("\n[diagnose] top 15 features by avg permutation importance:")
    report_lines = [
        "# What drives the ~0.86-0.89 detection ceiling (round 3 generator vs round 3 tuned detector)",
        "",
        "| feature | avg importance | human mean (std) | bot mean (std) | z-gap |",
        "|---|---|---|---|---|",
    ]
    for idx in order[:15]:
        feat = SHAPE_ONLY_FEATURES[idx]
        h_mean, h_std = human_df[feat].mean(), human_df[feat].std()
        b_mean, b_std = bot_df[feat].mean(), bot_df[feat].std()
        z_gap = abs(h_mean - b_mean) / (h_std or 1.0)
        print(f"  {feat}: importance={avg_importance[idx]:.4f}  human={h_mean:.3f}+-{h_std:.3f}  "
              f"bot={b_mean:.3f}+-{b_std:.3f}  z-gap={z_gap:.2f}")
        report_lines.append(
            f"| {feat} | {avg_importance[idx]:.4f} | {h_mean:.3f} ({h_std:.3f}) | "
            f"{b_mean:.3f} ({b_std:.3f}) | {z_gap:.2f} |"
        )

    (RESULTS_DIR / "gmm_bot_tell_diagnosis.md").write_text("\n".join(report_lines))
    print(f"\n[diagnose] wrote {RESULTS_DIR / 'gmm_bot_tell_diagnosis.md'}")


if __name__ == "__main__":
    main()
