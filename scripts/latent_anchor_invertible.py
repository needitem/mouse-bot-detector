#!/usr/bin/env python3
"""Latent-anchored replay with EXACT (float64) invertibility - closing the last
open thread of the trajectory-realism study.

Prior finding (commit e7b50a9): encode a real stroke z=f(x_real), add a small
per-dim Gaussian latent perturbation of scale sigma, decode. At sigma~0.1 this
scores ~0.68 detection with near-duplicates fully broken (0.1%) - the only method
that dents the ~0.85 wall WHILE keeping diversity. But the sigma=0 reconstruction
floor was 0.608, not 0.5, even though a normalizing flow is exactly invertible in
principle (decode(encode(x))=x -> 0.5).

The gap is NUMERICAL, not information-theoretic: the ActNorm log-scale clamp
[-6,6] and coupling exp(s) terms are applied identically in forward and inverse,
so they are algebraically exact - but in float32, exp(a)*exp(-a) != 1 to ~1e-6
per op, and that error accumulates over 24 layers x 2 sublayers on a deep
(nll -395) flow, pushing the reconstruction off-manifold enough to cost ~0.1 of
detection accuracy. This script does the anchor encode/decode in FLOAT64, which
should collapse the reconstruction error to ~1e-13 and drop the sig=0 floor
toward 0.5 - and, with it, the sigma~0.1 point toward "replay-grade AND diverse".

Changes vs latent_interp_generator.py:
  * saves the trained flow checkpoint (state_dict + norm stats) so this never has
    to be retrained again;
  * anchor generation runs the flow in double precision (exact inverse);
  * prints the sig=0 reconstruction MSE in float32 vs float64 as direct evidence;
  * anchor sigmas default to include 0.0 (the reconstruction floor).
Training is unchanged (float32), so the flow itself is identical to the -395 run.
"""
import argparse, json, math
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


# ---------- RealNVP (dtype-agnostic: works in float32 or float64) ----------
class ActNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(d)); self.log_s = nn.Parameter(torch.zeros(d))
        self.inited = False
    def forward(self, x):
        if not self.inited:
            with torch.no_grad():
                self.loc.data = -x.mean(0)
                self.log_s.data = -torch.log(x.std(0).clamp(min=1e-2))
                self.inited = True
        ls = self.log_s.clamp(-6, 6)
        z = (x + self.loc) * torch.exp(ls)
        return z, ls.sum().expand(x.shape[0])
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
        ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for act, cp in self.blocks:
            x, d1 = act(x); x, d2 = cp(x); ld = ld + d1 + d2
        return x, ld
    def inverse(self, z):
        for act, cp in reversed(self.blocks):
            z = cp.inverse(z); z = act.inverse(z)
        return z


def _decode_write(Vk, std, mean, keep, path, rng, tag):
    Vk = Vk.astype(np.float64) * std + mean
    V = np.zeros((Vk.shape[0], DIM), np.float64)   # reinsert dropped constant dims
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="save trained flow here (default <out>.ckpt)")
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--anchor_sigmas", default="0.0,0.05,0.1,0.15,0.2")
    args = ap.parse_args()
    ckpt_path = args.ckpt or (args.out + ".ckpt")

    print(f"[latent] device={DEVICE}", flush=True)
    X = load_vectors(args.data)
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
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
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
    torch.save({"state_dict": flow.state_dict(), "mean": mean, "std": std,
                "keep": keep, "D": D, "hidden": args.hidden, "layers": args.layers},
               ckpt_path)
    print(f"[latent] saved checkpoint -> {ckpt_path}", flush=True)

    rng = np.random.default_rng(1)

    # --- direct evidence: reconstruction error, float32 vs float64 ---
    with torch.no_grad():
        z32 = flow(Xs)[0]
        rec32 = flow.inverse(z32)
        mse32 = ((rec32 - Xs) ** 2).mean().item()

        flow_d = Flow(D, hidden=args.hidden, layers=args.layers).to(DEVICE).double()
        flow_d.load_state_dict(flow.state_dict())
        flow_d.eval()
        for b in flow_d.blocks:
            b[0].inited = True     # skip data-dependent re-init on the double copy
        Xd = Xs.double()
        z64 = flow_d(Xd)[0]
        rec64 = flow_d.inverse(z64)
        mse64 = ((rec64 - Xd) ** 2).mean().item()
    print(f"[latent] reconstruction MSE  float32={mse32:.3e}  float64={mse64:.3e}", flush=True)

    # --- latent-anchored replay in float64 (exact inverse) ---
    with torch.no_grad():
        Z = flow_d(Xd)[0]                      # (n, D) real encodings, double
        zstd = Z.std(0, keepdim=True)
        for sig in [float(s) for s in args.anchor_sigmas.split(",")]:
            idx = rng.integers(0, n, args.n)
            noise = torch.tensor(
                rng.standard_normal((args.n, D)), dtype=torch.float64, device=DEVICE)
            za = Z[idx] + sig * zstd * noise
            dec = flow_d.inverse(za).cpu().numpy()
            _decode_write(dec, std, mean, keep,
                          f"{args.out}.s{sig}", rng, f"anchor64(sig={sig})")


if __name__ == "__main__":
    main()
