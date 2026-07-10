#!/usr/bin/env python3
"""Latent-space interpolation generator (RealNVP flow) - the last qualitatively
distinct idea for beating the ~0.85 trajectory-realism wall.

Every prior generator either sampled a learned prior (flow/GMM/diffusion -> 0.85)
or blended in POSITION space (kNN blend -> 0.889, averaging kills the jerk). This
one blends in the flow's LEARNED LATENT space: encode two REAL strokes to their
exact latent codes z1=f(x1), z2=f(x2), interpolate z=a*z1+(1-a)*z2, decode
x=f^-1(z). The hypothesis: the flow "unfolds" the curved data manifold into a
Gaussian latent, so a straight line in z stays ON the manifold when decoded -
unlike a straight line in x, which cuts through off-manifold space. If true, we
get novel strokes (not near-duplicates) that stay human (unlike blend).

Modes: prior (sample N(0,I) - reproduces flow_generator ~0.86 baseline) and
interp (the new idea). Output schema matches every other generator, so the
Jetson-side validate_flow_bot_strong_detector.py scores it identically.
"""
import argparse, json, math, sys
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 48                      # shape points
DIM = N * 2 + 2            # 48*2 + [log distance, log movement_time]
MIN_MT = 40.0


# ---------- canonicalize ----------
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


def load_vectors(path, cap=14000):
    V = []
    for i, line in enumerate(open(path)):
        if i >= cap: break
        pts = json.loads(line)["points"]
        if len(pts) < 4: continue
        v = to_canonical(pts)
        if v is not None and np.all(np.isfinite(v)):
            V.append(v)
    return np.array(V, np.float32)


# ---------- RealNVP ----------
class ActNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(d)); self.log_s = nn.Parameter(torch.zeros(d))
        self.inited = False
    def forward(self, x):
        if not self.inited:
            with torch.no_grad():
                self.loc.data = -x.mean(0)
                # floor the std: canonicalization pins point0=(0,0) and
                # point_last=(1,0), so those 4 dims are constant (std=0) and
                # -log(std) would explode to NaN. Clamp keeps it finite.
                self.log_s.data = -torch.log(x.std(0).clamp(min=1e-2))
                self.inited = True
        z = (x + self.loc) * torch.exp(self.log_s.clamp(-6, 6))
        return z, self.log_s.clamp(-6, 6).sum().expand(x.shape[0])
    def inverse(self, z):
        return z * torch.exp(-self.log_s.clamp(-6, 6)) - self.loc


class Coupling(nn.Module):
    def __init__(self, d, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d * 2))
        self.net[-1].weight.data.zero_(); self.net[-1].bias.data.zero_()
    def forward(self, x):
        xm = x * self.mask
        h = self.net(xm); s, t = h.chunk(2, 1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        z = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        return z, s.sum(1)
    def inverse(self, z):
        zm = z * self.mask
        h = self.net(zm); s, t = h.chunk(2, 1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return zm + (1 - self.mask) * ((z - t) * torch.exp(-s))


class Flow(nn.Module):
    def __init__(self, d, hidden=256, layers=16):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(layers):
            mask = torch.zeros(d); mask[torch.arange(d) % 2 == (i % 2)] = 1
            self.blocks.append(nn.ModuleList([ActNorm(d), Coupling(d, hidden, mask)]))
    def forward(self, x):
        ld = torch.zeros(x.shape[0], device=x.device)
        for act, cp in self.blocks:
            x, d1 = act(x); x, d2 = cp(x); ld = ld + d1 + d2
        return x, ld
    def inverse(self, z):
        for act, cp in reversed(self.blocks):
            z = cp.inverse(z); z = act.inverse(z)
        return z


def _decode_write(Vk, std, mean, keep, path, rng, tag):
    Vk = Vk * std + mean
    V = np.zeros((Vk.shape[0], DIM), np.float32)   # reinsert dropped constant dims
    V[:, keep] = Vk
    V[:, N*2 - 2] = 1.0  # point_last x = 1; point0/point_last y stay 0
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
    print(f"[latent:{tag}] wrote {len(out)} -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="interp output; prior goes to <out>.prior")
    ap.add_argument("--mode", choices=["prior", "interp", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--alpha_lo", type=float, default=0.3)
    ap.add_argument("--alpha_hi", type=float, default=0.7)
    args = ap.parse_args()

    print(f"[latent] device={DEVICE}", flush=True)
    X = load_vectors(args.data)
    # Drop the dims pinned constant by canonicalization: point0=(0,0) -> dims 0,1
    # and point_last=(1,0) -> dims N*2-2, N*2-1. Modeling std-0 dims wrecks
    # ActNorm's exp() conditioning (grad blows up ~1e10). Reinserted at decode.
    DROP = [0, 1, N*2 - 2, N*2 - 1]
    keep = [i for i in range(DIM) if i not in DROP]
    Xk = X[:, keep]
    D = len(keep)
    mean = Xk.mean(0); std = np.maximum(Xk.std(0), 1e-3)
    Xs = torch.tensor((Xk - mean) / std, device=DEVICE).clamp(-8, 8)
    print(f"[latent] {Xs.shape[0]} vectors, modeled dim={D} (dropped {len(DROP)} constant)", flush=True)

    flow = Flow(D, hidden=args.hidden, layers=args.layers).to(DEVICE)
    print(f"[latent] flow: {args.layers} layers, hidden {args.hidden}, "
          f"{sum(p.numel() for p in flow.parameters())/1e6:.2f}M params", flush=True)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = Xs.shape[0]; bs = args.batch
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=DEVICE); tot = 0.0
        for j in range(0, n, bs):
            xb = Xs[perm[j:j+bs]]
            z, ld = flow(xb)
            nll = 0.5 * (z ** 2).sum(1) - ld + 0.5 * DIM * math.log(2 * math.pi)
            loss = nll.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step(); tot += loss.item() * xb.shape[0]
        sched.step()
        if (ep + 1) % 200 == 0 or ep == 0:
            print(f"[latent] ep {ep+1}/{args.epochs} nll={tot/n:.3f}", flush=True)

    flow.eval()
    rng = np.random.default_rng(1)
    with torch.no_grad():
        if args.mode in ("interp", "both"):
            Z = flow(Xs)[0]
            i1 = rng.integers(0, n, args.n); i2 = rng.integers(0, n, args.n)
            a = torch.tensor(rng.uniform(args.alpha_lo, args.alpha_hi, (args.n, 1)),
                             dtype=torch.float32, device=DEVICE)
            zi = a * Z[i1] + (1 - a) * Z[i2]
            _decode_write(flow.inverse(zi).cpu().numpy(), std, mean, keep,
                          args.out, rng, "interp")
        if args.mode in ("prior", "both"):
            z = torch.randn(args.n, D, device=DEVICE)
            _decode_write(flow.inverse(z).cpu().numpy(), std, mean, keep,
                          args.out + ".prior", rng, "prior")


if __name__ == "__main__":
    main()
