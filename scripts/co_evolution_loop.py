#!/usr/bin/env python3
"""Orchestrates genuine alternating adversarial co-evolution. Each round:

  1. hybrid_noise_search.py evolves the GENERATOR against the CURRENT
     detector ensemble (adversarial_loop.py's _ensemble_model_factories,
     which reads results/ensemble_hyperparams.json if present).
  2. generate_gmm_bot_coevo_file.py produces fresh bot movements from the
     round's winning generator config.
  3. tune_ensemble_hyperparams.py runs a genuine RandomizedSearchCV to
     re-tune the DETECTOR's hyperparameters against THAT generator's output,
     overwriting results/ensemble_hyperparams.json for round N+1's generator
     to fight against.

This is the fix for what round 1 (fixed single-detector) and round 2 (fixed
3-family ensemble) both got wrong: the detector side never moved in response
to the generator's own evolution, so a fresh independent search always found
~0.86 regardless of how the generator was pressured. Tracks each round's
in-loop worst-case accuracy AND independently re-tuned fresh accuracy in
results/coevolution_progress.json to see whether alternating actually
converges anywhere or just oscillates.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
PYTHON = sys.executable

N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def run(cmd, log_name):
    log_path = RESULTS_DIR / log_name
    print(f"[co-evolution] running: {' '.join(str(c) for c in cmd)} -> {log_path}", flush=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    text = log_path.read_text(errors="replace")
    if proc.returncode != 0:
        print(text[-4000:], flush=True)
        raise RuntimeError(f"{cmd} failed with code {proc.returncode}, see {log_path}")
    return text


def parse_evolved_worst(stdout):
    m = re.search(r"final validation - evolved-noise: mean=([\d.]+) worst=([\d.]+)", stdout)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def parse_fresh_accuracy(stdout):
    m = re.search(r"ROUND_FRESH_ACCURACY=([\d.]+)", stdout)
    return float(m.group(1)) if m else None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    for round_i in range(1, N_ROUNDS + 1):
        print(f"\n[co-evolution] ===== ROUND {round_i}/{N_ROUNDS} =====", flush=True)

        out = run([PYTHON, str(SCRIPT_DIR / "hybrid_noise_search.py")],
                   f"coevo_round{round_i}_search.log")
        in_loop_mean, in_loop_worst = parse_evolved_worst(out)
        print(f"[co-evolution] round {round_i}: in-loop evolved mean={in_loop_mean} worst={in_loop_worst}", flush=True)

        run([PYTHON, str(SCRIPT_DIR / "generate_gmm_bot_coevo_file.py")],
            f"coevo_round{round_i}_generate.log")

        out = run([PYTHON, str(SCRIPT_DIR / "tune_ensemble_hyperparams.py"),
                   "gmm_hybrid_bot_coevo_movements.jsonl"],
                  f"coevo_round{round_i}_tune.log")
        fresh_acc = parse_fresh_accuracy(out)
        print(f"[co-evolution] round {round_i}: fresh independent accuracy={fresh_acc}", flush=True)

        history.append({
            "round": round_i,
            "in_loop_evolved_mean": in_loop_mean,
            "in_loop_evolved_worst": in_loop_worst,
            "fresh_detector_accuracy": fresh_acc,
        })
        (RESULTS_DIR / "coevolution_progress.json").write_text(json.dumps(history, indent=2))
        print(f"[co-evolution] progress so far:\n{json.dumps(history, indent=2)}", flush=True)

    print("\n[co-evolution] ===== DONE =====", flush=True)
    print(json.dumps(history, indent=2), flush=True)


if __name__ == "__main__":
    main()
