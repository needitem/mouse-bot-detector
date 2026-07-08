#!/usr/bin/env python3
"""Diffusion (DDPM) trajectory generator - the stronger-model escalation past
the RealNVP flow (which plateaued at ~0.85 strong-detector shape_only).

A flow maps noise->data in one invertible pass; a diffusion model learns the
data manifold through many small denoising steps, which empirically captures
fine structure (the high-frequency jerk/tremor the flow kept missing) far
better. Same representation as flow_generator.py (canonical 48-pt unit shape +
log distance/duration, standardized), same output schema, so
validate_flow_bot_strong_detector.py scores it identically. If DDPM's
single-move accuracy beats the flow's 0.85, a generative attacker gets
UNLIMITED diversity (defeating every reuse/aggregate detector) at a lower
single-move cost - the one thing that could reopen the attacker's side.

Run on the GPU box:  python -u scripts/diffusion_generator.py --epochs 3000
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_generator import (
    DIM, DEVICE, build_vectors, fit_standardizer, standardize, sample_movements,
    N_SHAPE_POINTS, HUMAN_SAMPLE_SIZE, N_MOVEMENTS,
)
from trajectory_gmm_ceiling import load_human_pool_raw_points

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_PATH = DATA_DIR / "flow_bot_movements.jsonl"   # same file the validators read


class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):                      # t: (B,) in [0,1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * freqs[None] * 1000.0
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class Denoiser(nn.Module):
    """MLP eps-predictor with a residual trunk and time conditioning."""
    def __init__(self, dim, hidden=512, temb=128, depth=4):
        super().__init__()
        self.temb = nn.Sequential(SinusoidalEmb(temb), nn.Linear(temb, hidden), nn.SiLU())
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden)) for _ in range(depth)])
        self.out = nn.Linear(hidden, dim)
        self.out.weight.data.zero_(); self.out.bias.data.zero_()

    def forward(self, x, t):
        h = self.inp(x) + self.temb(t)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def make_schedule(T, device):
    # cosine schedule (Nichol & Dhariwal)
    s = 0.008
    ts = torch.linspace(0, T, T + 1, device=device) / T
    ac = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(1e-5, 0.999)
    alphas = 1 - betas
    acp = torch.cumprod(alphas, 0)
    return betas, alphas, acp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n", type=int, default=N_MOVEMENTS)
    args = ap.parse_args()

    print(f"[ddpm] device={DEVICE}", flush=True)
    pool = load_human_pool_raw_points(seed=0)
    X = build_vectors(pool[:args.train_n])
    mean, std = fit_standardizer(X)
    Xs = torch.tensor(standardize(X, mean, std), dtype=torch.float32, device=DEVICE)
    print(f"[ddpm] {Xs.shape[0]} vectors, dim={DIM}, T={args.T}", flush=True)

    betas, alphas, acp = make_schedule(args.T, DEVICE)
    sqrt_acp = acp.sqrt(); sqrt_1macp = (1 - acp).sqrt()

    model = Denoiser(DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = Xs.shape[0]

    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for j in range(0, n, args.batch):
            xb = Xs[idx[j:j + args.batch]]
            b = xb.shape[0]
            t = torch.randint(0, args.T, (b,), device=DEVICE)
            noise = torch.randn_like(xb)
            xt = sqrt_acp[t][:, None] * xb + sqrt_1macp[t][:, None] * noise
            pred = model(xt, t.float() / args.T)
            loss = ((pred - noise) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * b
        sched.step()
        if (ep + 1) % 100 == 0 or ep == 0:
            print(f"[ddpm] epoch {ep+1}/{args.epochs}  loss={tot/n:.4f}", flush=True)

    # DDPM ancestral sampling
    @torch.no_grad()
    def sample(nn_):
        x = torch.randn(nn_, DIM, device=DEVICE)
        for i in reversed(range(args.T)):
            t = torch.full((nn_,), i, device=DEVICE)
            pred = model(x, t.float() / args.T)
            a, ac = alphas[i], acp[i]
            mean_x = (x - (1 - a) / (1 - ac).sqrt() * pred) / a.sqrt()
            if i > 0:
                x = mean_x + betas[i].sqrt() * torch.randn_like(x)
            else:
                x = mean_x
        return x

    print(f"[ddpm] sampling {args.n} movements...", flush=True)
    # wrap into a tiny object exposing .sample(n, seed) for sample_movements()
    class _Gen:
        def sample(self, k, seed=0):
            torch.manual_seed(seed)
            # clamp in standardized space: ancestral sampling can produce rare
            # far-tail draws that blow up after de-standardize + exp on the two
            # log-scalars (distance/duration). +-4 sigma keeps them physical.
            return sample(k).clamp(-4.0, 4.0)
        def parameters(self):
            return model.parameters()
    movements = sample_movements(_Gen(), mean, std, args.n, seed=1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for pts in movements:
            f.write(json.dumps({"points": pts}) + "\n")
    print(f"[ddpm] wrote {len(movements)} movements to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
