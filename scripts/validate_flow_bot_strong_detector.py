#!/usr/bin/env python3
"""Strong-detector validation for the neural-flow generator - identical
protocol to validate_gmm_bot_strong_detector.py (train_detector.py's OWN full
RandomizedSearchCV across RandomForest / GradientBoosting /
HistGradientBoosting / SVM-RBF), pointed at data/processed/flow_bot_movements.jsonl
instead of the GMM output. This is the honest, apples-to-apples number: the
GMM's best strong-detector accuracy was ~0.855, so anything meaningfully below
that is a real win for the flow family.
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

BOT_FILE = sys.argv[1] if len(sys.argv) > 1 else "flow_bot_movements.jsonl"


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
    print(f"[validate-flow] loading human + flow bot ({BOT_FILE})...")
    human_rows = load_class(["human_movements.jsonl", "human_movements_web.jsonl"])
    bot_rows = load_class([BOT_FILE])
    print(f"[validate-flow] human={len(human_rows)} flow_bot={len(bot_rows)}")

    n = min(MAX_PER_CLASS, len(human_rows), len(bot_rows))
    rng = np.random.default_rng(0)
    human_rows = [human_rows[i] for i in rng.choice(len(human_rows), size=n, replace=False)]
    bot_rows = [bot_rows[i] for i in rng.choice(len(bot_rows), size=n, replace=False)]
    df = pd.DataFrame(human_rows + bot_rows)
    y_all = np.array([0] * n + [1] * n)

    report = ["# Strong-detector validation: human vs flow_bot", "",
              f"n={n} per class. Full RandomizedSearchCV across {len(SEARCH_SPACES)} "
              f"model families ({', '.join(SEARCH_SPACES)}) - same protocol as "
              "validate_gmm_bot_strong_detector.py (GMM best was ~0.855), pointed at "
              f"the neural-flow generator's output (`{BOT_FILE}`).", ""]

    results = {}
    for fs_name, feature_set in [("shape_only", SHAPE_ONLY_FEATURES), ("all", FEATURE_NAMES)]:
        print(f"\n[validate-flow] === feature set: {fs_name} ===")
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
        results[fs_name] = acc
        print(f"[validate-flow] BEST model: {best['model']} (cv_acc={best['cv_accuracy']:.3f})")
        print(f"[validate-flow] held-out test: accuracy={acc:.3f} f1={f1:.3f} roc_auc={auc:.3f}")
        for s in search_summary:
            print(f"[validate-flow]   {s['model']}: cv_acc={s['cv_accuracy']:.3f}")

        report += [f"## Feature set: `{fs_name}`", "", "| model family | CV accuracy |", "|---|---|"]
        for s in search_summary:
            report.append(f"| {s['model']} | {s['cv_accuracy']:.3f} |")
        report += ["", f"**Best: {best['model']}** - held-out test accuracy={acc:.3f}, "
                   f"f1={f1:.3f}, roc_auc={auc:.3f}", ""]

    report += ["## Headline", "",
               f"- flow shape_only: **{results['shape_only']:.3f}**  |  all: **{results['all']:.3f}**",
               "- GMM (prior best strong-detector): ~0.855", ""]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "flow_detector_validation.md").write_text("\n".join(report))
    print(f"\n[validate-flow] wrote {RESULTS_DIR / 'flow_detector_validation.md'}")
    print(f"[validate-flow] HEADLINE shape_only={results['shape_only']:.3f} all={results['all']:.3f} (GMM was ~0.855)")


if __name__ == "__main__":
    main()
