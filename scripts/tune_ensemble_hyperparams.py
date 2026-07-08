#!/usr/bin/env python3
"""One half of genuine alternating co-evolution (see co_evolution_loop.py):
runs a fresh RandomizedSearchCV (train_detector.py's own SEARCH_SPACES) for
RandomForest/GradientBoosting/HistGradientBoosting against the CURRENT
round's generator output, and writes the best hyperparameters found for each
family to results/ensemble_hyperparams.json - which adversarial_loop.py's
_ensemble_model_factories() reads on its next run. This is what makes the
detector side of the arms race actually move: previously the in-loop fitness
ensemble's hyperparameters were fixed once (round 1: 1 hand-picked family,
round 2: 3 hand-picked families) and never revisited even as the generator
kept evolving against it - a fresh independent search always found ~0.86
regardless, meaning the fixed-ensemble evolution wasn't transferring to any
genuinely re-tuned detector. SVM_RBF is deliberately excluded here (kept out
of the in-loop ensemble too) - CalibratedClassifierCV-style probability
support is too slow to retrain every epoch inside the evolutionary search.

Usage: tune_ensemble_hyperparams.py <bot_movements_filename> [<class_name>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_detector import SEARCH_SPACES, SHAPE_ONLY_FEATURES, MAX_PER_CLASS, SEARCH_CV, SEARCH_ITERS, RANDOM_STATE
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from features import extract_features

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Kept in sync with adversarial_loop.py's _ENSEMBLE_CLASSES - the 3 families
# actually used in the in-loop fitness ensemble (SVM excluded, see docstring).
ENSEMBLE_FAMILIES = ["RandomForest", "GradientBoosting", "HistGradientBoosting"]


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
    bot_filename = sys.argv[1]
    print(f"[tune-ensemble] loading human + {bot_filename}...")
    human_rows = load_class(["human_movements.jsonl", "human_movements_web.jsonl"])
    bot_rows = load_class([bot_filename])
    print(f"[tune-ensemble] human={len(human_rows)} bot={len(bot_rows)}")

    n = min(MAX_PER_CLASS, len(human_rows), len(bot_rows))
    rng = np.random.default_rng(0)
    human_rows = [human_rows[i] for i in rng.choice(len(human_rows), size=n, replace=False)]
    bot_rows = [bot_rows[i] for i in rng.choice(len(bot_rows), size=n, replace=False)]
    df = pd.DataFrame(human_rows + bot_rows)
    y_all = np.array([0] * n + [1] * n)
    X = df[SHAPE_ONLY_FEATURES].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=SEARCH_CV, shuffle=True, random_state=RANDOM_STATE)
    new_params = {}
    best_overall = None
    for family in ENSEMBLE_FAMILIES:
        estimator, param_dist = SEARCH_SPACES[family]
        search = RandomizedSearchCV(
            estimator, param_dist, n_iter=SEARCH_ITERS, cv=cv,
            scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, search.best_estimator_.predict(X_test))
        print(f"[tune-ensemble] {family}: cv_acc={search.best_score_:.3f} "
              f"held_out_acc={test_acc:.3f} params={search.best_params_}")
        new_params[family] = search.best_params_
        if best_overall is None or test_acc > best_overall:
            best_overall = test_acc

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ensemble_hyperparams.json").write_text(json.dumps(new_params, indent=2))
    print(f"[tune-ensemble] wrote {RESULTS_DIR / 'ensemble_hyperparams.json'}")
    print(f"[tune-ensemble] ROUND_FRESH_ACCURACY={best_overall:.4f}")


if __name__ == "__main__":
    main()
