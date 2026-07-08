#!/usr/bin/env python3
"""Round 2 of the strong-detector check: after strengthening the in-loop
fitness ensemble to 3 diverse, properly-tuned model families (see
adversarial_loop.py's _ensemble_model_factories) and re-running
hybrid_noise_search.py's evolutionary search against it (worst=0.583 in-loop),
this re-runs train_detector.py's OWN fresh RandomizedSearchCV hyperparameter
search (RandomForest, GradientBoosting, HistGradientBoosting, SVM-RBF) against
the round-2 generator's output (data/processed/gmm_hybrid_bot_round2_movements.jsonl,
produced by generate_gmm_bot_round2_file.py) - checking whether the in-loop
0.583 holds up against a genuinely independent hyperparameter search, or
whether the same one-sided-arms-race gap (0.598 -> 0.855 in round 1)
reappears at this new level.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_detector import (
    SEARCH_SPACES, SHAPE_ONLY_FEATURES, FEATURE_NAMES, MAX_PER_CLASS,
    tune_best_model, TEST_SIZE, RANDOM_STATE,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from features import extract_features

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


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
    print("[validate-strong-r2] loading human + gmm_hybrid_bot_round2 classes...")
    human_rows = load_class(["human_movements.jsonl", "human_movements_web.jsonl"])
    bot_rows = load_class(["gmm_hybrid_bot_round2_movements.jsonl"])
    print(f"[validate-strong-r2] human={len(human_rows)} gmm_hybrid_bot_round2={len(bot_rows)}")

    n = min(MAX_PER_CLASS, len(human_rows), len(bot_rows))
    rng = np.random.default_rng(0)
    human_rows = [human_rows[i] for i in rng.choice(len(human_rows), size=n, replace=False)]
    bot_rows = [bot_rows[i] for i in rng.choice(len(bot_rows), size=n, replace=False)]
    df = pd.DataFrame(human_rows + bot_rows)
    y_all = np.array([0] * n + [1] * n)

    report = ["# Round 2 strong-detector validation: human vs gmm_hybrid_bot_round2", "",
              f"n={n} per class. Full RandomizedSearchCV across {len(SEARCH_SPACES)} "
              f"model families ({', '.join(SEARCH_SPACES)}) - re-run specifically against "
              "the round-2 generator (evolved against a diverse 3-model-family fitness "
              "ensemble, in-loop worst=0.583) to check whether it holds up against a "
              "genuinely independent hyperparameter search.",
              ""]

    for fs_name, feature_set in [("shape_only", SHAPE_ONLY_FEATURES), ("all", FEATURE_NAMES)]:
        print(f"\n[validate-strong-r2] === feature set: {fs_name} ===")
        X = df[feature_set].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
        )
        best, search_summary = tune_best_model(X_train, y_train)
        model = best["estimator"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        print(f"[validate-strong-r2] BEST model: {best['model']} (cv_acc={best['cv_accuracy']:.3f})")
        print(f"[validate-strong-r2] held-out test: accuracy={acc:.3f} f1={f1:.3f} roc_auc={auc:.3f}")
        for s in search_summary:
            print(f"[validate-strong-r2]   {s['model']}: cv_acc={s['cv_accuracy']:.3f} params={s['params']}")

        report += [
            f"## Feature set: `{fs_name}`",
            "",
            "| model family | CV accuracy |",
            "|---|---|",
        ]
        for s in search_summary:
            report.append(f"| {s['model']} | {s['cv_accuracy']:.3f} |")
        report += [
            "",
            f"**Best: {best['model']}** - held-out test accuracy={acc:.3f}, f1={f1:.3f}, roc_auc={auc:.3f}",
            "",
        ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "strong_detector_validation_round2.md").write_text("\n".join(report))
    print(f"\n[validate-strong-r2] wrote {RESULTS_DIR / 'strong_detector_validation_round2.md'}")


if __name__ == "__main__":
    main()
