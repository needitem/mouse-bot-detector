#!/usr/bin/env python3
"""Round 2: generates data/processed/gmm_hybrid_bot_round2_movements.jsonl
using the evolved-noise config from results/hybrid_noise_evolved_config.json
- the config hybrid_noise_search.py found when evolving the generator against
the STRENGTHENED, diverse 3-model-family detector ensemble (see
adversarial_loop.py's _ensemble_model_factories), not the single never-retuned
HistGradientBoostingClassifier round 1 evolved against.

Same purpose as generate_gmm_bot_file.py: feed train_detector.py's own fresh
RandomizedSearchCV hyperparameter search to check whether round 2's in-loop
result (worst=0.583 against the fixed 3-model ensemble) holds up against an
INDEPENDENTLY re-tuned detector too, or whether the same one-sided-arms-race
gap reappears at this new level.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_noise_search import fit_shape_gmm, _sample_valid_vecs, sample_hybrid_trajectory
from trajectory_gmm_ceiling import load_human_pool_raw_points

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "results" / "hybrid_noise_evolved_config.json"
OUT_PATH = SCRIPT_DIR.parent / "data" / "processed" / "gmm_hybrid_bot_round2_movements.jsonl"

N_MOVEMENTS = 4000  # matches train_detector.py's MAX_PER_CLASS


def main():
    print("[generate_gmm_bot_round2_file] loading human pool, fitting shape GMM...")
    pool_points = load_human_pool_raw_points(seed=0)
    gmm, scaler, kept = fit_shape_gmm(pool_points[:1200])

    cfg = json.loads(CONFIG_PATH.read_text())
    print(f"[generate_gmm_bot_round2_file] using evolved config: {cfg}")

    print(f"[generate_gmm_bot_round2_file] sampling {N_MOVEMENTS} movements...")
    rng = random.Random(777)
    rows = []
    vecs = _sample_valid_vecs(gmm, scaler, N_MOVEMENTS)
    for vec in vecs:
        pts = sample_hybrid_trajectory(vec, cfg, rng)
        if len(pts) >= 4:
            rows.append(pts)

    print(f"[generate_gmm_bot_round2_file] writing {len(rows)} movements to {OUT_PATH}...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for i, pts in enumerate(rows):
            rec = {
                "user": "gmm_hybrid_bot_round2",
                "session": f"gmm_hybrid_bot_round2_{i}",
                "points": [[float(x), float(y), float(t)] for x, y, t in pts],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[generate_gmm_bot_round2_file] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
