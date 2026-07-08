#!/usr/bin/env python3
"""THE decisive control experiment: is ~0.50 even reachable with this
feature set + detector, or is ~0.85 a property of the detector/data rather
than of any generator?

Takes the REAL human movements, splits them into two random halves drawn
from the IDENTICAL distribution, labels one half "human" and the other
"bot", and runs train_detector.py's OWN full RandomizedSearchCV strong
detector on that. Two same-distribution samples SHOULD be indistinguishable
(accuracy ~0.50). Whatever this returns is the true floor:

  - ~0.50  -> the pipeline is honest; a good-enough generator really could
              reach 50%, so pushing the flow/generator further is worthwhile.
  - >>0.50 -> the strong detector is reading a split/collection artifact (or
              overfitting n=4000 with 13 features); no generator can beat
              that floor, and the ~0.85 "ceiling" is partly the detector's,
              not the generator's - target must be re-set accordingly.
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
from sklearn.metrics import accuracy_score, roc_auc_score
from features import extract_features

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


def load_human():
    rows = []
    for filename in ["human_movements.jsonl", "human_movements_web.jsonl"]:
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
    print("[human-floor] loading human movements...")
    human = load_human()
    print(f"[human-floor] {len(human)} human movements")

    # need 2n total: n labelled "human", n labelled "bot" - both from human
    n = min(MAX_PER_CLASS, len(human) // 2)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(human))[: 2 * n]
    group_a = [human[i] for i in idx[:n]]      # label 0
    group_b = [human[i] for i in idx[n:2 * n]]  # label 1 (still real humans!)
    print(f"[human-floor] two same-distribution groups of {n} each "
          "(both are REAL humans; one is mislabelled 'bot')")

    df = pd.DataFrame(group_a + group_b)
    y_all = np.array([0] * n + [1] * n)

    report = ["# Human-vs-human control (the true floor)", "",
              f"n={n} per group, both drawn from the same real-human distribution. "
              "A perfectly honest detector should score ~0.50 (chance). Same "
              "RandomizedSearchCV strong detector used for every generator eval.", ""]

    for fs_name, feature_set in [("shape_only", SHAPE_ONLY_FEATURES), ("all", FEATURE_NAMES)]:
        print(f"\n[human-floor] === feature set: {fs_name} ===")
        X = df[feature_set].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
        )
        best, summary = tune_best_model(Xtr, ytr)
        model = best["estimator"]
        acc = accuracy_score(yte, model.predict(Xte))
        auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
        print(f"[human-floor] {fs_name}: best={best['model']} test_acc={acc:.3f} auc={auc:.3f}")
        report += [f"## `{fs_name}`", "", f"- best model: {best['model']}",
                   f"- **test accuracy = {acc:.3f}** (chance = 0.500), roc_auc = {auc:.3f}", ""]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "human_floor.md").write_text("\n".join(report))
    print(f"\n[human-floor] wrote {RESULTS_DIR / 'human_floor.md'}")


if __name__ == "__main__":
    main()
