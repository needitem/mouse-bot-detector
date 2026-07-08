#!/usr/bin/env python3
"""Neural normalizing-flow trajectory generator - the escalation past the
high-component-GMM generator (hybrid_noise_search.py / co_evolution_loop.py),
which plateaued at ~0.83-0.85 worst-case against a freshly, independently
hyperparameter-tuned detector (results/gmm_bot_tell_diagnosis.md showed the
gap is spread thin across many features + joint structure, with all marginal
z-gaps < 0.35 - a generator-FAMILY ceiling, not a single fixable tell).

Why a flow, per README "Suggested next steps" #4: the high-k GMM approaches
one full-covariance component per training point ("one Gaussian per point"),
which memorizes the TRAINING joint distribution but generalizes poorly to a
held-out human split - exactly why a fresh detector recovers ~0.83. A
RealNVP flow with bounded capacity learns a single smooth invertible map, so
it can represent the joint distribution + tails without that per-point
degeneracy, and its own sampling naturally carries the fine high-frequency
detail (jerk/tremor) that the smooth GMM control points lost.

Representation is IDENTICAL to trajectory_gmm_ceiling.py / hybrid_noise_search.py
(canonical frame, N_SHAPE_POINTS-point duration-fraction resample, vector =
[x_1..x_N, y_1..y_N, distance, movement_time]) so results are directly
comparable. distance/movement_time are log-transformed before standardizing
(both positive and heavy-tailed). Sampling un-normalizes with a random
direction and runs the SAME extract_features() as everywhere else.

Output: data/processed/flow_bot_movements.jsonl, in the same schema as
generate_gmm_bot_file.py - so validate_flow_bot_strong_detector.py can score
it with train_detector.py's OWN full RandomizedSearchCV, the honest
strong-detector protocol the GMM's 0.855 came from.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Small tensors (batch x 98) over a coupling stack: torch's default all-core
# threading THRASHES on ops this small (measured 12s/epoch at 8 threads vs the
# model's trivial FLOP count). Few threads is dramatically faster here.
torch.set_num_threads(int(os.environ.get("FLOW_THREADS", "4")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_noise_search import to_canonical_at, N_SHAPE_POINTS, MIN_MOVEMENT_TIME
from trajectory_gmm_ceiling import load_human_pool_raw_points
from features import extract_features

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
OUT_PATH = DATA_DIR / "flow_bot_movements.jsonl"

DIM = N_SHAPE_POINTS * 2 + 2          # 48*2 + [distance, movement_time] = 98
HUMAN_SAMPLE_SIZE = 1200              # same train slice as the GMM ceiling scripts
N_MOVEMENTS = 4000                    # matches train_detector.py MAX_PER_CLASS
SCALAR_IDX = [DIM - 2, DIM - 1]       # distance, movement_time (log-transformed)


# ----------------------------- data -----------------------------------------
def build_vectors(points_list):
    vecs = []
    for pts in points_list:
        c = to_canonical_at(pts, N_SHAPE_POINTS)
        if c is None:
            continue
        shape_xy, distance, movement_time = c
        vecs.append(np.concatenate([shape_xy.ravel(), [distance, movement_time]]))
    return np.asarray(vecs, dtype=np.float64)


def fit_standardizer(X):
    """log-transform the two positive scalars, then z-score every dim."""
    Xt = X.copy()
    Xt[:, SCALAR_IDX] = np.log(np.clip(Xt[:, SCALAR_IDX], 1e-6, None))
    mean = Xt.mean(axis=0)
    std = Xt.std(axis=0) + 1e-8
    return mean, std


def standardize(X, mean, std):
    Xt = X.copy()
    Xt[:, SCALAR_IDX] = np.log(np.clip(Xt[:, SCALAR_IDX], 1e-6, None))
    return (Xt - mean) / std


def destandardize(Z, mean, std):
    Xt = Z * std + mean
    Xt[:, SCALAR_IDX] = np.exp(Xt[:, SCALAR_IDX])
    return Xt


# ----------------------------- flow -----------------------------------------
class ActNorm(nn.Module):
    """Per-dim affine with data-dependent init (Glow) - normalizes each
    coupling's input to ~zero-mean/unit-var on the first training batch, which
    is what keeps a deep affine-coupling stack from diverging on CPU."""
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("inited", torch.tensor(False))

    def _init(self, x):
        with torch.no_grad():
            m = x.mean(dim=0)
            s = x.std(dim=0) + 1e-6
            self.log_scale.data = -torch.log(s)
            self.bias.data = -m / s
            self.inited.fill_(True)

    def forward(self, x):                        # data -> latent
        if not bool(self.inited) and self.training:
            self._init(x)
        z = x * torch.exp(self.log_scale) + self.bias
        logdet = self.log_scale.sum().expand(x.shape[0])
        return z, logdet

    def inverse(self, z):                        # latent -> data
        return (z - self.bias) * torch.exp(-self.log_scale)


class Coupling(nn.Module):
    """Affine coupling: the `mask==1` dims pass through and condition an MLP
    that scales/shifts the `mask==0` dims. tanh-bounded log-scale for stable
    CPU training."""
    def __init__(self, dim, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )
        # start as identity
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def _st(self, x):
        h = self.net(x * self.mask)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        return s, t

    def forward(self, x):                       # data -> latent
        s, t = self._st(x)
        z = x * torch.exp(s) + t
        return z, s.sum(dim=1)

    def inverse(self, z):                        # latent -> data
        s, t = self._st(z)
        return (z - t) * torch.exp(-s)


class RealNVP(nn.Module):
    def __init__(self, dim, hidden=256, n_layers=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        actnorms, couplings, perms, inv_perms = [], [], [], []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2::2] = 1.0                 # alternating parity, flipped each layer
            actnorms.append(ActNorm(dim))
            couplings.append(Coupling(dim, hidden, mask))
            p = torch.randperm(dim, generator=g)
            perms.append(p)
            inv_perms.append(torch.argsort(p))
        self.actnorms = nn.ModuleList(actnorms)
        self.couplings = nn.ModuleList(couplings)
        self.perms = perms
        self.inv_perms = inv_perms

    def forward(self, x):                        # data -> latent, returns z, logdet
        logdet = torch.zeros(x.shape[0], device=x.device)
        for act, coup, p in zip(self.actnorms, self.couplings, self.perms):
            x, ld_a = act(x)
            x = x[:, p.to(x.device)]
            x, ld_c = coup(x)
            logdet = logdet + ld_a + ld_c
        return x, logdet

    def log_prob(self, x):
        z, logdet = self.forward(x)
        base = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return base + logdet

    @torch.no_grad()
    def sample(self, n, seed=0):
        device = next(self.parameters()).device
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, DIM, generator=g).to(device)
        for act, coup, ip in zip(reversed(self.actnorms), reversed(self.couplings),
                                 reversed(self.inv_perms)):
            z = coup.inverse(z)
            z = z[:, ip.to(device)]
            z = act.inverse(z)
        return z

    def rsample(self, n):
        """Differentiable sampling (reparameterized) - keeps the graph so an
        adversarial loss on the samples can backprop into the flow. Same map
        as sample(), just without the no_grad/seeded generator."""
        device = next(self.parameters()).device
        z = torch.randn(n, DIM, device=device)
        for act, coup, ip in zip(reversed(self.actnorms), reversed(self.couplings),
                                 reversed(self.inv_perms)):
            z = coup.inverse(z)
            z = z[:, ip.to(device)]
            z = act.inverse(z)
        return z


def train_flow(X_std, hidden=256, n_layers=16, epochs=2500, lr=2e-4,
               batch=1024, val_frac=0.1, seed=0, verbose=True, patience=30):
    torch.manual_seed(seed)
    X = torch.tensor(X_std, dtype=torch.float32)
    n = X.shape[0]
    n_val = max(32, int(n * val_frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Xval = X[tr_idx].to(DEVICE), X[val_idx].to(DEVICE)

    model = RealNVP(DIM, hidden=hidden, n_layers=n_layers, seed=seed).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(Xtr.shape[0])
        for j in range(0, Xtr.shape[0], batch):
            xb = Xtr[idx[j:j + batch]]
            loss = -model.log_prob(xb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                vnll = -model.log_prob(Xval).mean().item()
            if vnll < best_val - 1e-3:
                best_val, best_state, bad = vnll, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
            if verbose:
                print(f"[flow] epoch {ep+1}/{epochs}  val_nll={vnll:.3f}  best={best_val:.3f}  bad={bad}")
            if bad >= patience:
                print(f"[flow] early stop at epoch {ep+1} (val_nll plateaued)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# --------------------------- sample -> movements ----------------------------
def sample_movements(model, mean, std, n, seed):
    Zstd = model.sample(n, seed=seed).cpu().numpy().astype(np.float64)
    X = destandardize(Zstd, mean, std)
    rng = np.random.default_rng(seed)
    out = []
    for v in X:
        shape = v[: N_SHAPE_POINTS * 2].reshape(N_SHAPE_POINTS, 2)
        distance = max(float(v[N_SHAPE_POINTS * 2]), 5.0)
        movement_time = max(float(v[N_SHAPE_POINTS * 2 + 1]), MIN_MOVEMENT_TIME)
        direction = rng.uniform(0.0, 2.0 * math.pi)
        c, s = math.cos(direction), math.sin(direction)
        xs, ys = shape[:, 0] * distance, shape[:, 1] * distance
        rot_x = xs * c - ys * s
        rot_y = xs * s + ys * c
        t = np.linspace(0.0, movement_time, N_SHAPE_POINTS)
        points = list(zip(rot_x.tolist(), rot_y.tolist(), t.tolist()))
        out.append(points)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--train_n", type=int, default=8000,
                    help="how many human movements to fit the flow on (flows benefit "
                         "from far more data than the memorizing high-k GMM's 1200)")
    ap.add_argument("--n", type=int, default=N_MOVEMENTS)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=30,
                    help="early-stop patience in eval-intervals (each = 10 epochs)")
    ap.add_argument("--quicktest", action="store_true",
                    help="also print a fast fresh-ensemble accuracy sanity check")
    args = ap.parse_args()

    print(f"[flow] device={DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print("[flow] loading human pool + canonicalizing...")
    pool = load_human_pool_raw_points(seed=0)
    train_points = pool[:args.train_n]
    X = build_vectors(train_points)
    print(f"[flow] {len(X)}/{len(train_points)} movements kept, dim={X.shape[1]}")

    mean, std = fit_standardizer(X)
    X_std = standardize(X, mean, std)

    print(f"[flow] training RealNVP ({args.layers} layers, hidden={args.hidden}, "
          f"batch={args.batch}, lr={args.lr})...")
    model = train_flow(X_std, hidden=args.hidden, n_layers=args.layers,
                       epochs=args.epochs, batch=args.batch, lr=args.lr,
                       patience=args.patience)

    print(f"[flow] sampling {args.n} synthetic movements -> {OUT_PATH.name}...")
    movements = sample_movements(model, mean, std, args.n, seed=1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for pts in movements:
            f.write(json.dumps({"points": pts}) + "\n")
    print(f"[flow] wrote {len(movements)} movements to {OUT_PATH}")

    if args.quicktest:
        from adversarial_loop import train_detector_ensemble, ensemble_accuracy
        print("[flow] quick fresh-ensemble sanity (held-out human vs fresh flow sample)...")
        val_points = pool[args.train_n:args.train_n + 800]
        human_rows = [extract_features(p) for p in val_points]
        bot_pts = sample_movements(model, mean, std, len(human_rows), seed=2)
        bot_rows = [extract_features(p) for p in bot_pts]
        half = len(human_rows) // 2
        ens = train_detector_ensemble(human_rows[:half], bot_rows[:half], seed_base=9000)
        acc = ensemble_accuracy(ens, human_rows[half:], bot_rows[half:])
        accw = ensemble_accuracy(ens, human_rows[half:], bot_rows[half:], reduce="worst")
        print(f"[flow] quick fresh-ensemble: mean={acc:.3f} worst={accw:.3f} "
              "(NOTE: default ensemble, not the strong RandomizedSearchCV detector)")


if __name__ == "__main__":
    main()
