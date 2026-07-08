#!/usr/bin/env python3
"""Generates data/processed/gmm_hybrid_bot_coevo_movements.jsonl from
whatever config currently sits at results/hybrid_noise_evolved_config.json -
used every round of co_evolution_loop.py, so the filename stays fixed across
rounds (each round's tune_ensemble_hyperparams.py call reads the SAME
filename, always representing the latest generator)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_noise_search import fit_shape_gmm, _sample_valid_vecs, sample_hybrid_trajectory
from trajectory_gmm_ceiling import load_human_pool_raw_points

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "results" / "hybrid_noise_evolved_config.json"
OUT_PATH = SCRIPT_DIR.parent / "data" / "processed" / "gmm_hybrid_bot_coevo_movements.jsonl"

N_MOVEMENTS = 4000  # matches train_detector.py's MAX_PER_CLASS


def main():
    print("[generate_gmm_bot_coevo_file] loading human pool, fitting shape GMM...")
    pool_points = load_human_pool_raw_points(seed=0)
    gmm, scaler, kept = fit_shape_gmm(pool_points[:1200])

    cfg = json.loads(CONFIG_PATH.read_text())
    print(f"[generate_gmm_bot_coevo_file] using evolved config: {cfg}")

    print(f"[generate_gmm_bot_coevo_file] sampling {N_MOVEMENTS} movements...")
    rng = random.Random(777)
    rows = []
    vecs = _sample_valid_vecs(gmm, scaler, N_MOVEMENTS)
    for vec in vecs:
        pts = sample_hybrid_trajectory(vec, cfg, rng)
        if len(pts) >= 4:
            rows.append(pts)

    print(f"[generate_gmm_bot_coevo_file] writing {len(rows)} movements to {OUT_PATH}...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for i, pts in enumerate(rows):
            rec = {
                "user": "gmm_hybrid_bot_coevo",
                "session": f"gmm_hybrid_bot_coevo_{i}",
                "points": [[float(x), float(y), float(t)] for x, y, t in pts],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[generate_gmm_bot_coevo_file] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
