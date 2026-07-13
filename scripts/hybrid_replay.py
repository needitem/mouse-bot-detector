#!/usr/bin/env python3
"""Hybrid stealth flick: the best practical low-detection bot given the findings.
Per flick, warp a real (native-resolution) stroke onto the target and apply a
RANDOMIZED mix of two perturbations:
  - elastic bend  : smooth low-freq perpendicular sine modes (breaks near-dup)
  - variability   : a step along a NATURAL human-variation direction,
                    mag*(shape_A - shape_B) for two random pool strokes (makes the
                    residual look like natural variation, not a synthetic bend)
Both amplitudes are randomized per flick, so the residual to the nearest real
stroke has no consistent signature for a residual-spectrum detector to learn,
while single-move stays near replay (native points) and near-dup is broken.
Session-distribution (finite pool clustering) is the residual limit no
perturbation removes.
"""
import argparse, json, math
import numpy as np

def load_db(path):
    d = json.load(open(path))["traj"]
    return [(t["d"], np.array(t["s"], float), np.array(t["t"], float)) for t in sorted(d, key=lambda z: z["d"])]

def resample_shape(s, N):
    """Resample a (M,2) unit shape to N points by index."""
    M = len(s); xi = np.linspace(0, M-1, N); ar = np.arange(M)
    return np.stack([np.interp(xi, ar, s[:,0]), np.interp(xi, ar, s[:,1])], 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--elastic_max", type=float, default=0.035)
    ap.add_argument("--var_max", type=float, default=0.06)
    ap.add_argument("--tol", type=float, default=0.12)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    db = load_db(args.db); dists = np.array([z[0] for z in db]); K = len(db)
    used = set(); out = []
    for _ in range(args.n):
        D = float(rng.choice(dists)); phi = rng.uniform(0, 2*math.pi)
        lo = np.searchsorted(dists, D*(1-args.tol)); hi = np.searchsorted(dists, D*(1+args.tol))
        avail = [i for i in range(lo, hi) if i not in used] or list(range(lo, hi)) or [min(np.searchsorted(dists, D), K-1)]
        if all(i in used for i in avail): used -= set(avail)
        i = int(rng.choice(avail)); used.add(i)
        _, s, t = db[i]; N = len(s)
        # per-flick randomized perturbation amplitudes
        amp_e = rng.uniform(0, args.elastic_max)          # elastic bend
        mag_v = rng.uniform(0, args.var_max)              # natural-variation step
        ec = [rng.normal(0, amp_e/(k+1)) for k in range(3)]
        # natural-variation direction from two random pool strokes, resampled to N
        A = resample_shape(db[rng.integers(K)][1], N); B = resample_shape(db[rng.integers(K)][1], N)
        vdir = A - B
        c, sn = math.cos(phi), math.sin(phi); pts = []
        for k in range(N):
            bx, by = s[k,0], s[k,1]
            if N > 1:
                u = k/(N-1); disp = sum(ec[j]*math.sin((j+1)*math.pi*u) for j in range(3))
                kp, km = min(k+1,N-1), max(k-1,0)
                tx, ty = s[kp,0]-s[km,0], s[kp,1]-s[km,1]; tl = math.hypot(tx,ty)+1e-9
                bx += disp*(-ty/tl); by += disp*(tx/tl)     # elastic perpendicular
            bx += mag_v*vdir[k,0]; by += mag_v*vdir[k,1]     # variability step
            ux, uy = bx*D, by*D
            pts.append([float(ux*c-uy*sn), float(ux*sn+uy*c), float(t[k])])
        pts[-1][0] = D*c; pts[-1][1] = D*sn
        for j in range(1,len(pts)):
            if pts[j][2] < pts[j-1][2]: pts[j][2] = pts[j-1][2]
        out.append(pts)
    open(args.out, "w").write("\n".join(json.dumps({"points": p}) for p in out))
    print(f"[hybrid] wrote {len(out)} flicks (elastic<={args.elastic_max}, var<={args.var_max})")

if __name__ == "__main__":
    main()
