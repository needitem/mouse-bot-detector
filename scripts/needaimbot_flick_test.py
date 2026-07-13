#!/usr/bin/env python3
"""Faithful Python port of needaimbot's warped_replay.hpp generate() - used to
measure the strong-detector score of the ACTUAL aimbot output, with and without
the new elastic deformation. Loads the same flick_trajectories.json (48-pt unit
canonical strokes + real irregular timestamps), samples target reaches from the
DB's own distance distribution, warps a distance-matched no-repeat stroke onto
each target, optionally applies the elastic bend, and writes {"points":[[x,y,t]]}.
"""
import argparse, json, math
import numpy as np

def load_db(path):
    d = json.load(open(path))
    traj = d["traj"]
    strokes = [(t["d"], np.array(t["s"], float), np.array(t["t"], float)) for t in traj]
    strokes.sort(key=lambda z: z[0])
    return strokes  # sorted by distance

def gen(strokes, n, amp, modes, tol, seed=1):
    rng = np.random.default_rng(seed)
    dists = np.array([s[0] for s in strokes])
    N = len(strokes[0][1])
    used = set()
    out = []
    for _ in range(n):
        D = float(rng.choice(dists))                 # target reach ~ DB distribution
        phi = rng.uniform(0, 2*math.pi)
        lo = np.searchsorted(dists, D*(1-tol)); hi = np.searchsorted(dists, D*(1+tol))
        avail = [i for i in range(lo, hi) if i not in used]
        if not avail:
            for i in range(lo, hi): used.discard(i)
            avail = list(range(lo, hi)) or [int(np.searchsorted(dists, D))]
            avail = [min(max(i,0),len(strokes)-1) for i in avail]
        i = int(rng.choice(avail)); used.add(i)
        _, s, t = strokes[i]
        # elastic coefficients (once per flick)
        EM = min(max(modes,1),8) if amp > 0 else 0
        ec = [rng.normal(0, amp/(j+1)) for j in range(EM)]
        c, sn = math.cos(phi), math.sin(phi)
        pts = []
        for k in range(N):
            bx, by = s[k,0], s[k,1]
            if EM > 0:
                u = k/(N-1); disp = sum(ec[j]*math.sin((j+1)*math.pi*u) for j in range(EM))
                kp, km = min(k+1,N-1), max(k-1,0)
                tx, ty = s[kp,0]-s[km,0], s[kp,1]-s[km,1]; tl = math.hypot(tx,ty)
                if tl > 1e-9: bx += disp*(-ty/tl); by += disp*(tx/tl)
            ux, uy = bx*D, by*D
            rx, ry = ux*c - uy*sn, ux*sn + uy*c
            pts.append([float(rx), float(ry), float(t[k])])
        pts[-1][0] = D*c; pts[-1][1] = D*sn              # land on target
        for j in range(1,len(pts)):
            if pts[j][2] < pts[j-1][2]: pts[j][2] = pts[j-1][2]
        out.append(pts)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--amp", type=float, default=0.0)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    strokes = load_db(args.db)
    print(f"[flick] DB {len(strokes)} strokes, amp={args.amp}")
    out = gen(strokes, args.n, args.amp, args.modes, args.tol)
    with open(args.out, "w") as f:
        for pts in out: f.write(json.dumps({"points": pts}) + "\n")
    print(f"[flick] wrote {len(out)} -> {args.out}")

if __name__ == "__main__":
    main()
