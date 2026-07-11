#!/usr/bin/env python3
"""Local-tangent replay: manufacture diversity from the human data's OWN local
geometry, with no generative map and thus no map fingerprint.

The flow-displacement experiment proved two things: (1) emitting x+displacement
instead of decode(z) cancels the reconstruction fingerprint (sigma=0 -> 0.50),
but (2) the flow's latent-perturbation displacement is off-manifold at ANY scale
(sigma=0.02 -> 0.88), because the flow prior itself is only 0.86. So the flow
cannot supply on-manifold diversity.

This supplies diversity from the data instead. For each real anchor x (in the
canonical, standardized kept-dim space), take its k nearest REAL neighbors, fit a
local PCA (the tangent plane of the human manifold at x), and perturb x along it:

    output = x + V @ (alpha * s * noise)

V, s = top-m local principal directions / singular values of the neighborhood.
noise ~ N(0, I_m). alpha scales the step relative to how much real strokes
actually vary in each local direction. Every perturbation is a genuine human
variation direction, so it stays on-manifold; and output != x, so near-duplicates
of a finite pool break. No torch, no training -- pure local geometry.

Writes <out>.a<alpha> for each alpha, schema identical to every other generator.
"""
import argparse, json, math
import numpy as np
from sklearn.neighbors import NearestNeighbors

N = 48
DIM = N * 2 + 2
MIN_MT = 40.0
DROP = [0, 1, N * 2 - 2, N * 2 - 1]
KEEP = [i for i in range(DIM) if i not in DROP]


def to_canonical(points):
    pts = np.asarray(points, float)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    dist = math.hypot(dx, dy); mt = t[-1] - t[0]
    if dist < 5.0 or mt < MIN_MT:
        return None
    ang = math.atan2(dy, dx); c, s = math.cos(-ang), math.sin(-ang)
    rx = (x - x[0]) * c - (y - y[0]) * s
    ry = (x - x[0]) * s + (y - y[0]) * c
    rx, ry = rx / dist, ry / dist
    tg = np.linspace(t[0], t[-1], N)
    sx = np.interp(tg, t, rx); sy = np.interp(tg, t, ry)
    return np.concatenate([np.stack([sx, sy], 1).ravel(),
                           [math.log(dist), math.log(mt)]])


def load_kept(path, cap=14000):
    Vk = []
    for i, line in enumerate(open(path)):
        if i >= cap: break
        pts = json.loads(line)["points"]
        if len(pts) < 4: continue
        v = to_canonical(pts)
        if v is not None and np.all(np.isfinite(v)):
            Vk.append(v[KEEP])
    return np.asarray(Vk, np.float64)


def decode_write(Vk, std, mean, path, rng, tag):
    Vk = Vk * std + mean
    V = np.zeros((Vk.shape[0], DIM))
    V[:, KEEP] = Vk
    V[:, N*2 - 2] = 1.0
    out = []
    for v in V:
        shape = v[:N*2].reshape(N, 2)
        dist = math.exp(min(v[N*2], 12)); mt = max(math.exp(min(v[N*2+1], 8)), MIN_MT)
        if not (np.isfinite(dist) and dist >= 5.0): continue
        xs = shape[:, 0] * dist; ys = shape[:, 1] * dist
        ang = rng.uniform(0, 2*math.pi); c, s = math.cos(ang), math.sin(ang)
        rx = xs*c - ys*s; ry = xs*s + ys*c
        t = np.linspace(0, mt, N)
        pts = [[float(a_), float(b_), float(t_)] for a_, b_, t_ in zip(rx, ry, t)]
        if all(np.isfinite([p for r in pts for p in r])): out.append(pts)
    with open(path, "w") as f:
        for pts in out:
            f.write(json.dumps({"points": pts}) + "\n")
    print(f"[tangent:{tag}] wrote {len(out)} -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--k", type=int, default=16, help="neighbors for local PCA")
    ap.add_argument("--m", type=int, default=5, help="tangent dims")
    ap.add_argument("--alphas", default="0.3,0.5,0.7,1.0,1.3")
    args = ap.parse_args()

    Xk = load_kept(args.data)
    mean = Xk.mean(0); std = np.maximum(Xk.std(0), 1e-3)
    B = (Xk - mean) / std                         # standardized kept space
    n = B.shape[0]
    print(f"[tangent] {n} reals, dim={B.shape[1]}, k={args.k}, m={args.m}", flush=True)

    nn = NearestNeighbors(n_neighbors=args.k + 1).fit(B)
    _, nbr = nn.kneighbors(B)                      # (n, k+1); col 0 is self

    rng = np.random.default_rng(1)
    for alpha in [float(a) for a in args.alphas.split(",")]:
        idx = rng.integers(0, n, args.n)
        out = np.empty((args.n, B.shape[1]))
        for r, i in enumerate(idx):
            nbrs = B[nbr[i, 1:]]                    # k real neighbors (exclude self)
            c = nbrs - nbrs.mean(0)
            # local PCA via SVD; Vt rows are principal directions, sv singular values
            _, sv, Vt = np.linalg.svd(c, full_matrices=False)
            m = min(args.m, Vt.shape[0])
            scale = sv[:m] / math.sqrt(max(len(nbrs) - 1, 1))   # per-dir std of real variation
            beta = alpha * scale * rng.standard_normal(m)
            out[r] = B[i] + beta @ Vt[:m]
        decode_write(out, std, mean, f"{args.out}.a{alpha}", rng, f"a={alpha}")


if __name__ == "__main__":
    main()
