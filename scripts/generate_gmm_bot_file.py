#!/usr/bin/env python3
"""Generates data/processed/gmm_hybrid_bot_movements.jsonl using the final
high-component-count GMM shape generator (see hybrid_noise_search.py) - the
approach that got independent-validation worst-case accuracy down to ~0.60
against the FIXED HistGradientBoostingClassifier ensemble this whole project
has used throughout.

Written specifically to feed train_detector.py's own hyperparameter search
(RandomForest/GradientBoosting/HistGradientBoosting/SVM-RBF via
RandomizedSearchCV) - the one honest way to check whether that 0.60 number
reflects the generator actually closing the gap, or just this project's
one detector architecture/hyperparameters never having been re-tuned against
it (a one-sided arms race: the generator evolved, the detector's own
architecture never did).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_noise_search import fit_shape_gmm, sample_candidate_movements, zero_ish_config
from trajectory_gmm_ceiling import load_human_pool_raw_points

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR.parent / "data" / "processed" / "gmm_hybrid_bot_movements.jsonl"

N_MOVEMENTS = 4000  # matches train_detector.py's MAX_PER_CLASS


def main():
    print("[generate_gmm_bot_file] loading human pool, fitting shape GMM...")
    pool_points = load_human_pool_raw_points(seed=0)
    gmm, scaler, kept = fit_shape_gmm(pool_points[:1200])

    print(f"[generate_gmm_bot_file] sampling {N_MOVEMENTS} movements...")
    rng = random.Random(777)
    rows = []
    # sample_candidate_movements returns extracted FEATURES, not raw points -
    # we need raw (x, y, t) here, so inline the same logic minus feature extraction.
    from hybrid_noise_search import _sample_valid_vecs, sample_hybrid_trajectory
    cfg = zero_ish_config()  # noise evolution didn't help - GMM shape alone is the validated config
    vecs = _sample_valid_vecs(gmm, scaler, N_MOVEMENTS)
    for vec in vecs:
        pts = sample_hybrid_trajectory(vec, cfg, rng)
        if len(pts) >= 4:
            rows.append(pts)

    print(f"[generate_gmm_bot_file] writing {len(rows)} movements to {OUT_PATH}...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for i, pts in enumerate(rows):
            rec = {
                "user": "gmm_hybrid_bot",
                "session": f"gmm_hybrid_bot_{i}",
                "points": [[float(x), float(y), float(t)] for x, y, t in pts],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[generate_gmm_bot_file] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
