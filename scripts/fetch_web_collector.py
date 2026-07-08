#!/usr/bin/env python3
"""Pulls real human aiming sessions from the deployed aim-rl-web-collector
Cloudflare Worker (github.com/needitem/aim-rl-web-collector) and converts
them into this project's canonical movement schema - a second, FPS-aim-style
"human" data source alongside Balabit's office/RDP data (parse_balabit.py).

The collector's web game has players continuously track a moving on-screen
target (unlike Balabit's free-form desktop use), which is a much closer match
to what needaimbot's motor_synergy/pd_controller actually have to look like -
both the initial acquisition flick and the sustained tracking that follows.

No changes to the deployed Worker are needed: it already exposes
GET /api/sessions (list) and GET /api/sessions/{id}/jsonl (raw trace), which
is everything this script needs.

Output: data/processed/human_movements_web.jsonl, same
{"user", "session", "points": [[x, y, t_ms], ...]} schema parse_balabit.py
produces, so train_detector.py can combine both sources transparently.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUT_PATH = DATA_DIR / "processed" / "human_movements_web.jsonl"

DEFAULT_BASE_URL = "https://aim-rl-web-collector.th07290828.workers.dev"

# web/src/main.ts renders the arena on a 960x960 canvas and maps cursor
# position 1:1 to it; world coords are normalized to [-1, 1] (2 units wide),
# so 1 world unit = 960/2 = 480 px. Only relative distances/speeds matter for
# our features, so an exact canvas-origin match isn't needed - just a
# consistent linear scale.
WORLD_TO_PX = 480.0

# Same movement-segmentation policy as parse_balabit.py, applied within each
# recorded episode (in case of dropped frames/pauses mid-episode).
MAX_GAP_MS = 300.0
MIN_POINTS = 8
MIN_DISTANCE_PX = 30.0


def fetch_json(url):
    # Cloudflare's bot protection 403s the default Python urllib User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mouse-bot-detector fetch script)"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_sessions(base_url, source_filter):
    data = fetch_json(f"{base_url}/api/sessions")
    sessions = data.get("sessions", [])
    if source_filter:
        sessions = [s for s in sessions if s.get("source") == source_filter]
    return sessions


def fetch_trace(base_url, session_id):
    data = fetch_json(f"{base_url}/api/sessions/{session_id}/jsonl")
    content = data.get("content", "")
    frames = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(json.loads(line))
    return frames


def group_by_episode(frames):
    by_episode = {}
    for frame in frames:
        by_episode.setdefault(frame["episode"], []).append(frame)
    for episode_frames in by_episode.values():
        episode_frames.sort(key=lambda f: f["step"])
    return by_episode


def segment_movements(points):
    """points: list of (x, y, t_ms) already sorted by time."""
    current = []
    last_t = None
    for x, y, t_ms in points:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--source", default="human-web",
        help="Only pull sessions with this `source` tag (default: human-web, "
             "the tag the browser game records under). Pass '' to pull all sources.",
    )
    args = parser.parse_args()

    try:
        sessions = fetch_sessions(args.base_url, args.source or None)
    except Exception as exc:  # noqa: BLE001 - surfacing network errors as-is
        print(f"[fetch_web_collector] Could not reach {args.base_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not sessions:
        print(f"[fetch_web_collector] No sessions found with source={args.source!r} at {args.base_url}.")
        print("[fetch_web_collector] Nobody has played the collector yet (or wrong --source) - nothing to write.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_movements = 0
    n_dropped = 0
    with open(OUT_PATH, "w") as out:
        for session in sessions:
            session_id = session["session_id"]
            player = session.get("player_name", "anonymous")
            frames = fetch_trace(args.base_url, session_id)
            for episode, episode_frames in group_by_episode(frames).items():
                points = [
                    (f["cursor_x"] * WORLD_TO_PX, f["cursor_y"] * WORLD_TO_PX, f["t"] * 1000.0)
                    for f in episode_frames
                ]
                for segment in segment_movements(points):
                    if not is_valid_movement(segment):
                        n_dropped += 1
                        continue
                    record = {
                        "user": player,
                        "session": f"{session_id}#ep{episode}",
                        "points": normalize(segment),
                    }
                    out.write(json.dumps(record) + "\n")
                    n_movements += 1

    print(f"[fetch_web_collector] sessions pulled: {len(sessions)}")
    print(f"[fetch_web_collector] movements kept:  {n_movements}")
    print(f"[fetch_web_collector] movements dropped (too short/small): {n_dropped}")
    print(f"[fetch_web_collector] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
