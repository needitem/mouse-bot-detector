#!/usr/bin/env python3
"""Builds the labeled feature dataset (human / naive_bot / motor_synergy_bot),
"extreme"-tunes classical classifiers (RandomForest, GradientBoosting,
HistGradientBoosting, SVM-RBF) via randomized hyperparameter search with a
genuine held-out test split (search never touches the test fold), evaluates
two pairings - human vs naive_bot (sanity check: should be near-ceiling) and
human vs motor_synergy_bot (the actual question) - plus a 3-way classifier,
and writes results (metrics, confusion matrices, feature importances) to
results/.

IMPORTANT CAVEAT (see README.md "Known confound" / features.py docstring):
Balabit's human data is RDP-captured with a much coarser/irregular sample
rate than either synthetic bot class's native rate. features.py now
resamples every movement to a common 40ms grid before any derivative-based
feature, which removes MOST (not all - see features.py) of that artifact.
Results are still reported for both the full feature set (`all`) and a
`shape_only` set that additionally drops the two features that deliberately
measure RAW (pre-resample) capture cadence (`sample_interval_mean/cv`) -
`shape_only` is the more trustworthy answer to "does motor_synergy look more
human than a naive bot."
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features import FEATURE_NAMES, extract_features  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

CLASS_FILES = {
    "human": ["human_movements.jsonl", "human_movements_web.jsonl"],
    "naive_bot": ["naive_bot_movements.jsonl"],
    "motor_synergy_bot": ["motor_synergy_bot_movements.jsonl"],
}

MAX_PER_CLASS = 4000

# Only the raw-capture-cadence features remain excluded from `shape_only` -
# see features.py's module docstring for why everything else is now fair game.
SAMPLING_SENSITIVE = {"sample_interval_mean", "sample_interval_cv"}
SHAPE_ONLY_FEATURES = [f for f in FEATURE_NAMES if f not in SAMPLING_SENSITIVE]

RANDOM_STATE = 0
TEST_SIZE = 0.2
SEARCH_ITERS = 15
SEARCH_CV = 3

SEARCH_SPACES = {
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [4, 6, 8, 12, None],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
        },
    ),
    "HistGradientBoosting": (
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "max_iter": [100, 200, 300],
            "max_depth": [None, 4, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
    ),
    "SVM_RBF": (
        Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
        {
            "svm__C": [0.1, 1.0, 3.0, 10.0, 30.0],
            "svm__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        },
    ),
}


def load_class(name):
    rows = []
    for filename in CLASS_FILES[name]:
        path = DATA_DIR / filename
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if len(rec["points"]) < 4:
                    continue
                feats = extract_features(rec["points"])
                feats["label"] = name
                rows.append(feats)
    return rows


def build_dataset():
    raw = {name: load_class(name) for name in CLASS_FILES}
    n_cap = min(MAX_PER_CLASS, *(len(v) for v in raw.values() if v))
    counts = {}
    rows = []
    for name, class_rows in raw.items():
        if not class_rows:
            continue
        if len(class_rows) > n_cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(class_rows), size=n_cap, replace=False)
            class_rows = [class_rows[i] for i in idx]
        counts[name] = len(class_rows)
        rows.extend(class_rows)
    return pd.DataFrame(rows), counts


def tune_best_model(X_train, y_train):
    """Randomized-searches every model family, returns the best-scoring
    fitted estimator (refit on all of X_train/y_train) and a summary dict."""
    best = None
    summary = []
    cv = StratifiedKFold(n_splits=SEARCH_CV, shuffle=True, random_state=RANDOM_STATE)
    for model_name, (estimator, param_dist) in SEARCH_SPACES.items():
        # Param names differ for the SVM pipeline (prefixed `svm__`).
        search = RandomizedSearchCV(
            estimator, param_dist, n_iter=SEARCH_ITERS, cv=cv,
            scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        summary.append({"model": model_name, "cv_accuracy": search.best_score_, "params": search.best_params_})
        if best is None or search.best_score_ > best["cv_accuracy"]:
            best = {"model": model_name, "cv_accuracy": search.best_score_,
                    "params": search.best_params_, "estimator": search.best_estimator_}
    summary.sort(key=lambda s: s["cv_accuracy"], reverse=True)
    return best, summary


def evaluate_pair(df, class_a, class_b, feature_set, feature_set_name, label_tag):
    subset = df[df["label"].isin([class_a, class_b])].reset_index(drop=True)
    if subset.empty or subset["label"].nunique() < 2:
        return None
    X = subset[feature_set].to_numpy()
    y = (subset["label"] == class_b).astype(int).to_numpy()  # class_b = positive ("bot")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    best, search_summary = tune_best_model(X_train, y_train)
    model = best["estimator"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    cm_path = RESULTS_DIR / f"confusion_{label_tag}_{feature_set_name}.png"
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=[class_a, class_b], normalize="true", cmap="Blues",
    )
    disp.ax_.set_title(f"{class_a} vs {class_b} ({feature_set_name}, {best['model']})")
    disp.figure_.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(disp.figure_)

    importances = None
    if best["model"] in ("RandomForest", "GradientBoosting"):
        importances = sorted(zip(feature_set, model.feature_importances_), key=lambda kv: kv[1], reverse=True)
    elif best["model"] == "HistGradientBoosting":
        # No built-in importances; use a quick permutation importance instead.
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        importances = sorted(zip(feature_set, perm.importances_mean), key=lambda kv: kv[1], reverse=True)

    return {
        "test_metrics": metrics, "best": best, "search_summary": search_summary,
        "importances": importances, "n_train": len(X_train), "n_test": len(X_test),
    }


def plot_importances(importances, title, out_path):
    if not importances:
        return
    names = [n for n, _ in importances]
    values = [v for _, v in importances]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(names))))
    ax.barh(names[::-1], values[::-1])
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def three_way(df, feature_set, feature_set_name):
    X = df[feature_set].to_numpy()
    y = df["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    cm_path = RESULTS_DIR / f"confusion_3way_{feature_set_name}.png"
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true", cmap="Blues")
    disp.ax_.set_title(f"3-way ({feature_set_name})")
    disp.figure_.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(disp.figure_)
    return acc


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, counts = build_dataset()
    if df.empty:
        print("[train_detector] No data found - run parse_balabit.py and generate_synthetic.py first.")
        return

    print(f"[train_detector] class counts (balanced): {counts}")

    report_lines = [
        "# Bot detector results",
        "",
        f"Balanced dataset: {counts} (held out {TEST_SIZE:.0%} as a genuine test "
        f"split never touched during hyperparameter search)",
        "",
        "## Known confound (partially fixed)",
        "",
        "Balabit's human data is RDP-captured with a much coarser/irregular sample "
        "rate than either synthetic bot class. features.py now resamples every "
        "movement to a common 40ms grid before computing any derivative/spectral "
        "feature, which removes most of the resulting jerk/curvature amplification "
        "artifact (see features.py's docstring for the full explanation and its "
        "limits). The remaining `sample_interval_mean/cv` pair deliberately measures "
        "RAW (pre-resample) capture cadence and is excluded from `shape_only`.",
        "",
        "## Model selection",
        "",
        f"Each pairing independently random-searches {len(SEARCH_SPACES)} model "
        f"families ({', '.join(SEARCH_SPACES)}) with {SEARCH_ITERS}-iteration "
        f"randomized search, {SEARCH_CV}-fold CV - the winning family/hyperparameters "
        "are reported per pairing, evaluated on the held-out test split.",
        "",
    ]

    feature_sets = {"all": FEATURE_NAMES, "shape_only": SHAPE_ONLY_FEATURES}
    pairings = [("human", "naive_bot"), ("human", "motor_synergy_bot")]

    for fs_name, fs in feature_sets.items():
        report_lines.append(f"## Feature set: `{fs_name}` ({len(fs)} features)")
        report_lines.append("")
        for class_a, class_b in pairings:
            outcome = evaluate_pair(df, class_a, class_b, fs, fs_name, f"{class_a}_vs_{class_b}")
            if outcome is None:
                continue
            m = outcome["test_metrics"]
            best = outcome["best"]
            report_lines.append(
                f"### {class_a} vs {class_b} (train={outcome['n_train']}, test={outcome['n_test']})"
            )
            report_lines.append("")
            report_lines.append(f"**Best model: {best['model']}** (CV accuracy during search: {best['cv_accuracy']:.3f})")
            report_lines.append(f"Params: `{best['params']}`")
            report_lines.append("")
            report_lines.append("| metric (held-out test) | value |")
            report_lines.append("|---|---|")
            report_lines.append(f"| accuracy | {m['accuracy']:.3f} |")
            report_lines.append(f"| f1 | {m['f1']:.3f} |")
            report_lines.append(f"| roc_auc | {m['roc_auc']:.3f} |")
            report_lines.append("")
            report_lines.append("All model families tried (sorted by CV accuracy):")
            report_lines.append("")
            report_lines.append("| model | cv_accuracy |")
            report_lines.append("|---|---|")
            for s in outcome["search_summary"]:
                report_lines.append(f"| {s['model']} | {s['cv_accuracy']:.3f} |")
            report_lines.append("")

            if outcome["importances"]:
                imp_path = RESULTS_DIR / f"importance_{class_a}_vs_{class_b}_{fs_name}.png"
                plot_importances(outcome["importances"], f"{class_a} vs {class_b} ({fs_name})", imp_path)
                top5 = ", ".join(f"{n} ({v:.3f})" for n, v in outcome["importances"][:5])
                report_lines.append(f"Top features: {top5}")
                report_lines.append("")
                report_lines.append(f"![importance](importance_{class_a}_vs_{class_b}_{fs_name}.png)")
            report_lines.append(f"![confusion](confusion_{class_a}_vs_{class_b}_{fs_name}.png)")
            report_lines.append("")

        acc_3way = three_way(df, fs, fs_name)
        report_lines.append(f"### 3-way classification ({fs_name}, RandomForest): accuracy={acc_3way:.3f}")
        report_lines.append("")
        report_lines.append(f"![confusion](confusion_3way_{fs_name}.png)")
        report_lines.append("")

    out_path = RESULTS_DIR / "report.md"
    out_path.write_text("\n".join(report_lines))
    print(f"[train_detector] wrote {out_path}")


if __name__ == "__main__":
    main()
