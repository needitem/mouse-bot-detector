#!/usr/bin/env bash
# Fetches the Balabit Mouse Dynamics Challenge dataset (real human RDP mouse
# sessions, 10 users) - used as this project's "human" ground-truth class.
# Fülöp, Kovács, Kurics, Windhager-Pokol, "Balabit Mouse Dynamics Challenge
# Data Set" (2016). Freely git-cloneable, no license gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data/raw"
REPO_URL="https://github.com/balabit/Mouse-Dynamics-Challenge"

mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/Mouse-Dynamics-Challenge/.git" ]; then
    echo "[fetch_dataset] Already present at $DATA_DIR/Mouse-Dynamics-Challenge, skipping clone."
else
    echo "[fetch_dataset] Cloning $REPO_URL ..."
    git clone --depth 1 "$REPO_URL" "$DATA_DIR/Mouse-Dynamics-Challenge"
fi

echo "[fetch_dataset] Done. Training sessions:"
find "$DATA_DIR/Mouse-Dynamics-Challenge/training_files" -type f | wc -l
