#!/usr/bin/env python3
"""Minimal-perturbation replay: the smallest per-flick displacement that breaks
near-duplicates, keeping single-move detection as low as possible.

The whole project shows the tradeoff is diversity (break near-dup) vs on-manifold
(low single-move). The existing detect_replay_dilemma.py found per-point 6px white
jitter lands at strong-detector 0.64 with near-dup broken. Two levers to do
better:

  * mode=jitter : per-point Gaussian jitter (px) -- the white-noise baseline.
  * mode=warp   : a SMOOTH low-frequency displacement field (sum of the first
    `modes` half-sine modes along the path, random small amplitude). This breaks
    the macro shape (distinct -> near-dup safe) WITHOUT injecting high-frequency
    jerk, so it should sit below white jitter at equal diversity.

Perturbs real strokes in raw pixel space, then the scorer canonicalizes both
sides identically. No flow, no training. Writes <out>.<tag> per scale.
"""
import argparse, json, math
import numpy as np

MIN_MT = 40.0


def load_points(path, cap=14000):
    P = []
    for i, line in enumerate(open(path)):
        if i >= cap: break
        pts = json.loads(line)["points"]
        if len(pts) >= 4:
            P.append(np.asarray(pts, float))
    return P


def jitter_stroke(p, scale, rng):
    rel = p[:, :2] - p[0, :2]
    d = math.hypot(rel[-1, 0], rel[-1, 1])
    if d < 5.0: return None
    theta = rng.uniform(0, 2*math.pi); c, s = math.cos(theta), math.sin(theta)
    rx = rel[:, 0]*c - rel[:, 1]*s + rng.normal(0, scale, len(rel))
    ry = rel[:, 0]*s + rel[:, 1]*c + rng.normal(0, scale, len(rel))
    t = p[:, 2] - p[0, 2]
    return [[float(a), float(b), float(tt)] for a, b, tt in zip(rx, ry, t)]


def warp_stroke(p, scale, modes, rng):
    """Smooth low-frequency displacement: sum of half-sine modes along arc index,
    amplitude ~ scale px. Zero at both endpoints (keeps start/end anchored)."""
    rel = p[:, :2] - p[0, :2]
    d = math.hypot(rel[-1, 0], rel[-1, 1])
    if d < 5.0: return None
    theta = rng.uniform(0, 2*math.pi); c, s = math.cos(theta), math.sin(theta)
    rx = rel[:, 0]*c - rel[:, 1]*s
    ry = rel[:, 0]*s + rel[:, 1]*c
    n = len(rel); u = np.linspace(0, 1, n)
    dx = np.zeros(n); dy = np.zeros(n)
    for k in range(1, modes + 1):
        basis = np.sin(k * math.pi * u)              # 0 at both ends, low freq
        dx += rng.normal(0, scale) * basis
        dy += rng.normal(0, scale) * basis
    rx = rx + dx; ry = ry + dy
    t = p[:, 2] - p[0, 2]
    return [[float(a), float(b), float(tt)] for a, b, tt in zip(rx, ry, t)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["jitter", "warp"], required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--modes", type=int, default=3, help="warp: number of low-freq modes")
    ap.add_argument("--scales", required=True, help="comma list of px scales")
    args = ap.parse_args()

    P = load_points(args.data)
    n = len(P)
    print(f"[minrep] {n} reals, mode={args.mode}", flush=True)
    rng = np.random.default_rng(1)
    for scale in [float(x) for x in args.scales.split(",")]:
        idx = rng.integers(0, n, args.n)
        out = []
        for i in idx:
            if args.mode == "jitter":
                st = jitter_stroke(P[i], scale, rng)
            else:
                st = warp_stroke(P[i], scale, args.modes, rng)
            if st is not None:
                out.append(st)
        path = f"{args.out}.{args.mode}{scale}"
        with open(path, "w") as f:
            for st in out:
                f.write(json.dumps({"points": st}) + "\n")
        print(f"[minrep:{args.mode} scale={scale}] wrote {len(out)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
