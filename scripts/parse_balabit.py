#!/usr/bin/env python3
"""Segments Balabit Mouse Dynamics Challenge session logs into individual
mouse movements - the "human" ground-truth class for the bot detector.

Fülöp, Kovács, Kurics, Windhager-Pokol, "Balabit Mouse Dynamics Challenge
Data Set" (2016). Each session file is a CSV of raw mouse events
(record timestamp, client timestamp, button, state, x, y); this script keeps
only the cursor-position samples (Move/Drag), splits them into discrete
movements on time gaps or clicks (the same "did the aim point just jump to
somewhere new" boundary needaimbot's motor_synergy flick fires on), and drops
degenerate segments.

Also filters to only the FAST tier of movements (--speed-percentile, default
75th): Balabit is general RDP desktop use (slow document scrolling, careful
dragging, etc.), most of which looks nothing like a deliberate game flick.
Keeping only the fastest movements (by mean px/s over the segment) is a
data-driven stand-in for "moved the mouse urgently, like acquiring a target"
- a fixed absolute px/s cutoff would be arbitrary given Balabit's unknown
per-user screen resolution/sensitivity, so the threshold is computed from
this dataset's own speed distribution instead.

Output: one JSON-lines file, data/processed/human_movements.jsonl, one
movement per line: {"user": ..., "session": ..., "points": [[x, y, t_ms], ...]}
- the same (x, y, t) shape as motor_synergy::trajectory_point.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "Mouse-Dynamics-Challenge" / "training_files"
OUT_PATH = DATA_DIR / "processed" / "human_movements.jsonl"

# A movement ends if the gap since the last position sample exceeds this, or
# a mouse button is pressed/released (a deliberate act, same boundary as a
# real aim-then-click cycle).
MAX_GAP_MS = 300.0
MIN_POINTS = 8
MIN_DISTANCE_PX = 30.0
DEFAULT_SPEED_PERCENTILE = 75.0

# A handful of Balabit rows (80 out of 2.25M) carry x/y == 65535 (0xFFFF) -
# a "no cursor position recorded yet" sentinel, not a real coordinate. Only
# ~80 rows, but even one in a movement's point list is enough to blow up any
# downstream statistic that depends on distances (path_efficiency, speed,
# curvature) - e.g. computing "distance" from a garbage first point while
# "path length" is computed from the real points produced path_efficiency
# values up to 64x, corrupting the whole feature's variance target.
SENTINEL_COORD = 65535.0


def iter_session_files():
    for user_dir in sorted(RAW_DIR.iterdir()):
        if not user_dir.is_dir():
            continue
        for session_file in sorted(user_dir.iterdir()):
            if session_file.is_file():
                yield user_dir.name, session_file.name, session_file


def parse_session(path):
    """Yields raw (t_ms, x, y, is_boundary) tuples in file order.

    is_boundary marks a Pressed/Released row (not a position sample itself,
    but a segmentation cut point).
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t_ms = float(row["client timestamp"]) * 1000.0
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, ValueError):
                continue
            if x >= SENTINEL_COORD or y >= SENTINEL_COORD or x < 0 or y < 0:
                continue  # garbage sentinel row, not a real position sample
            state = row.get("state", "")
            is_boundary = state in ("Pressed", "Released")
            is_position = state in ("Move", "Drag")
            if is_position:
                yield t_ms, x, y, False
            elif is_boundary:
                yield t_ms, x, y, True


def segment_movements(events):
    """Splits a session's (t_ms, x, y, is_boundary) stream into movements."""
    current = []
    last_t = None
    for t_ms, x, y, is_boundary in events:
        if is_boundary:
            if current:
                yield current
                current = []
            last_t = t_ms
            continue
        if last_t is not None and (t_ms - last_t) > MAX_GAP_MS:
            if current:
                yield current
                current = []
        current.append((x, y, t_ms))
        last_t = t_ms
    if current:
        yield current


def is_valid_movement(points):
    if len(points) < MIN_POINTS:
        return False
    x0, y0, _ = points[0]
    x1, y1, _ = points[-1]
    dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return dist >= MIN_DISTANCE_PX


def normalize(points):
    t0 = points[0][2]
    return [[x, y, t - t0] for x, y, t in points]


def mean_speed_px_s(points):
    x0, y0, _ = points[0]
    x1, y1, t1 = points[-1]
    duration_ms = t1 - points[0][2]
    if duration_ms <= 0:
        return 0.0
    dist = math.hypot(x1 - x0, y1 - y0)
    return dist / duration_ms * 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speed-percentile", type=float, default=DEFAULT_SPEED_PERCENTILE,
        help="Keep only movements at/above this percentile of mean speed "
             "(px/s) within this dataset - the 'urgent, game-like flick' "
             "filter. 0 disables the speed filter entirely.",
    )
    args = parser.parse_args()

    n_sessions = 0
    n_dropped_shape = 0
    candidates = []  # (user, session, points, speed)
    for user, session, path in iter_session_files():
        n_sessions += 1
        events = parse_session(path)
        for points in segment_movements(events):
            if not is_valid_movement(points):
                n_dropped_shape += 1
                continue
            candidates.append((user, session, points, mean_speed_px_s(points)))

    speeds = np.array([c[3] for c in candidates])
    if args.speed_percentile > 0 and len(speeds):
        threshold = float(np.percentile(speeds, args.speed_percentile))
    else:
        threshold = 0.0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_movements = 0
    n_dropped_slow = 0
    with open(OUT_PATH, "w") as out:
        for user, session, points, speed in candidates:
            if speed < threshold:
                n_dropped_slow += 1
                continue
            record = {
                "user": user,
                "session": session,
                "points": normalize(points),
            }
            out.write(json.dumps(record) + "\n")
            n_movements += 1

    print(f"[parse_balabit] sessions parsed: {n_sessions}")
    print(f"[parse_balabit] candidate movements (shape-valid): {len(candidates)}")
    print(f"[parse_balabit] dropped (too short/small): {n_dropped_shape}")
    print(
        f"[parse_balabit] speed filter: p{args.speed_percentile:.0f} "
        f"threshold={threshold:.1f} px/s, dropped (too slow/leisurely): {n_dropped_slow}"
    )
    print(f"[parse_balabit] movements kept: {n_movements}")
    print(f"[parse_balabit] wrote {OUT_PATH}")


if __name__ == "__main__":
    if not RAW_DIR.exists():
        print(f"[parse_balabit] {RAW_DIR} not found - run fetch_dataset.sh first.", file=sys.stderr)
        sys.exit(1)
    main()
