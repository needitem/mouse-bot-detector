#!/usr/bin/env python3
"""flow-GAN hybrid co-evolution: the last untried card for pushing a genuine
(generalizing) generator below the ~0.85 strong-detector wall that MLE-trained
GMM and flow both hit.

Every prior generator here was trained by MAXIMUM LIKELIHOOD - "look like the
human distribution on average" - which never directly optimizes the specific
discriminative direction a detector reads. This fine-tunes the MLE-pretrained
flow with an ADVERSARIAL term: a neural discriminator D learns human-vs-flow on
the canonical trajectory vector (98-dim, strictly more information than the
14 extracted features), and the flow is pushed to fool it - while D is
periodically RE-INITIALISED (co-evolution) so the flow can't just overfit one
frozen D (the transfer-failure mode this project hit in v1-v3).

Objective is a hybrid so it can't mode-collapse into unrealistic-but-
undetectable garbage: loss_G = lam_mle * NLL(real) + lam_adv * fool(D). The
flow's exact NLL (unavailable to a plain GAN) anchors realism while the
adversarial term chases the residual joint-structure tell.

Honest expectation: 0.50 needs the real distribution matched EXACTLY, which a
generalizing model can't (human-vs-human=0.50 only because it's literally the
same distribution; replay=0.52). This measures how far below 0.85 a true
generator can be dragged. Writes flow_adv_bot_movements.jsonl for
validate_flow_bot_strong_detector.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_generator import (
    RealNVP, DIM, DEVICE, build_vectors, fit_standardizer, standardize,
    train_flow, sample_movements,
)
from trajectory_gmm_ceiling import load_human_pool_raw_points

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUT_PATH = DATA_DIR / "flow_adv_bot_movements.jsonl"


class Disc(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def new_disc(seed):
    torch.manual_seed(seed)
    d = Disc(DIM).to(DEVICE)
    opt = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return d, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--pretrain_epochs", type=int, default=800)
    ap.add_argument("--adv_steps", type=int, default=3000)
    ap.add_argument("--lam_adv", type=float, default=0.3)
    ap.add_argument("--lam_mle", type=float, default=1.0)
    ap.add_argument("--d_reset", type=int, default=250,
                    help="re-initialise the discriminator every N steps (co-evolution)")
    ap.add_argument("--d_steps", type=int, default=2, help="D updates per G update")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    print(f"[adv] device={DEVICE}")
    pool = load_human_pool_raw_points(seed=0)
    X = build_vectors(pool[:args.train_n])
    mean, std = fit_standardizer(X)
    Xs = standardize(X, mean, std)
    print(f"[adv] {len(Xs)} canonical vectors, dim={DIM}")

    print(f"[adv] MLE pretraining flow ({args.pretrain_epochs} epochs)...")
    flow = train_flow(Xs, hidden=256, n_layers=16, epochs=args.pretrain_epochs,
                      batch=256, lr=2e-4, patience=30, verbose=False).to(DEVICE)
    with torch.no_grad():
        Xt = torch.tensor(Xs, dtype=torch.float32, device=DEVICE)
        print(f"[adv] pretrain done, real NLL={-flow.log_prob(Xt).mean().item():.2f}")

    optG = torch.optim.Adam(flow.parameters(), lr=1e-4, betas=(0.5, 0.999))
    D, optD = new_disc(seed=0)
    bce = nn.BCEWithLogitsLoss()
    ones = torch.ones(args.batch, 1, device=DEVICE)
    zeros = torch.zeros(args.batch, 1, device=DEVICE)
    n = Xt.shape[0]

    for step in range(args.adv_steps):
        if args.d_reset > 0 and step > 0 and step % args.d_reset == 0:
            D, optD = new_disc(seed=step)

        # ---- D updates ----
        for _ in range(args.d_steps):
            xr = Xt[torch.randint(0, n, (args.batch,), device=DEVICE)]
            with torch.no_grad():
                xf = flow.rsample(args.batch)
            lossD = bce(D(xr), ones) + bce(D(xf), zeros)
            optD.zero_grad(); lossD.backward(); optD.step()

        # ---- G update: fool D + stay realistic (MLE anchor) ----
        xf = flow.rsample(args.batch)
        adv = bce(D(xf), ones)
        xr = Xt[torch.randint(0, n, (args.batch,), device=DEVICE)]
        nll = -flow.log_prob(xr).mean()
        lossG = args.lam_adv * adv + args.lam_mle * nll
        optG.zero_grad(); lossG.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        optG.step()

        if step % 100 == 0 or step == args.adv_steps - 1:
            with torch.no_grad():
                xf = flow.rsample(1024); xr = Xt[torch.randint(0, n, (1024,), device=DEVICE)]
                d_real = (D(xr) > 0).float().mean().item()
                d_fake = (D(xf) > 0).float().mean().item()
                d_acc = 0.5 * (d_real + (1 - d_fake))
                real_nll = -flow.log_prob(xr).mean().item()
            print(f"[adv] step {step}/{args.adv_steps}  D_acc={d_acc:.3f} "
                  f"(real->{d_real:.2f} fake->{d_fake:.2f})  realNLL={real_nll:.1f}", flush=True)

    print(f"[adv] sampling {args.n} movements -> {OUT_PATH.name}...")
    movements = sample_movements(flow, mean, std, args.n, seed=1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for pts in movements:
            f.write(json.dumps({"points": pts}) + "\n")
    print(f"[adv] wrote {len(movements)} movements to {OUT_PATH}")


if __name__ == "__main__":
    main()
