#!/usr/bin/env python3
"""Elastic warped replay: generate diversity from a finite real pool WITHOUT
re-representing the stroke. Take a REAL stroke's ORIGINAL points (native
resolution + native irregular timing - the thing that lets warped replay reach
0.506), apply a rigid rotation, then a SMOOTH low-frequency lateral deformation
that bends the path. The smooth deformation changes the canonical shape (breaks
finite-pool near-duplicates) while preserving the fine local kinematics (so
detection stays near replay, not the 0.85 generation wall). Unlike latent-anchor,
there is no canonical/resample bottleneck (~0.16 tell) - we keep the raw points.

Sweep the deformation amplitude; amp=0 is pure warped replay.
"""
import argparse, json, math
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

def load_real(path, cap=14000):
    out = []
    for i, l in enumerate(open(path)):
        if i >= cap: break
        p = np.asarray(json.loads(l)["points"], float)
        if len(p) >= 4 and math.hypot(p[-1,0]-p[0,0], p[-1,1]-p[0,1]) >= 5 and p[-1,2]-p[0,2] >= 40:
            out.append(p)
    return out

def deform(pts, amp, n_modes, rng):
    """Rigid random rotation + smooth low-frequency lateral (and mild tangential)
    deformation of the ORIGINAL points. Displacement is a sum of a few sine modes
    along the stroke, scaled by the stroke's own size, applied perpendicular to
    the local movement direction. Smoothness (few low modes) keeps jerk realistic."""
    x, y, t = pts[:,0].copy(), pts[:,1].copy(), pts[:,2].copy()
    x0, y0 = x[0], y[0]
    x -= x0; y -= y0
    dist = math.hypot(x[-1], y[-1]) + 1e-9
    K = len(pts)
    u = np.linspace(0.0, 1.0, K)                    # index fraction along stroke
    # smooth perpendicular displacement d(u) = sum_k a_k sin(k*pi*u) (0 at ends)
    disp = np.zeros(K)
    for k in range(1, n_modes+1):
        a = rng.normal(0, amp * dist / k)           # lower amplitude for higher modes
        disp += a * np.sin(k * math.pi * u)
    # local movement direction -> perpendicular
    dx = np.gradient(x); dy = np.gradient(y)
    seg = np.hypot(dx, dy) + 1e-9
    nx, ny = -dy/seg, dx/seg                         # unit normal
    x = x + disp * nx; y = y + disp * ny
    # endpoints must stay pinned (disp is 0 there already) -> keep target intact
    # random rigid rotation
    ang = rng.uniform(0, 2*math.pi); c, s = math.cos(ang), math.sin(ang)
    xr = x*c - y*s; yr = x*s + y*c
    return [[float(xr[i]+x0), float(yr[i]+y0), float(t[i]-t[0])] for i in range(K)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", default="0.0,0.02,0.05,0.1,0.15")
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    pool = load_real(str(DATA/"human_movements.jsonl"))
    print(f"[elastic] pool={len(pool)} real strokes, modes={args.modes}")
    for amp in [float(a) for a in args.amps.split(",")]:
        out = []
        for _ in range(args.n):
            p = pool[rng.integers(len(pool))]
            out.append(deform(p, amp, args.modes, rng))
        path = DATA/f"elastic_a{amp}_bot_movements.jsonl"
        with open(path, "w") as f:
            for pts in out:
                f.write(json.dumps({"points": pts}) + "\n")
        print(f"[elastic] amp={amp}: wrote {len(out)} -> {path.name}", flush=True)

if __name__ == "__main__":
    main()
