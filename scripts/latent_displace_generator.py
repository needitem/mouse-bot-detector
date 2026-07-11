#!/usr/bin/env python3
"""Displacement replay: manufacture diversity from the flow WITHOUT paying its
round-trip fingerprint.

Diagnosis (RESEARCH.md, the negative-result section): routing a real stroke
through the flow's encode->decode imprints a structured joint fingerprint that a
strong detector reads at ~0.82, even at zero perturbation. The perturbation was
never the problem; emitting decode(...) is. Reals decoded WITHOUT the flow score
0.47.

Fix: never emit decode(z). Emit the CLEAN real plus only the flow-induced
DISPLACEMENT of the perturbation:

    z = encode(x)
    output = x + ( decode(z + sigma*zstd*noise) - decode(z) )

At sigma=0 this is exactly x (-> ~0.47). At sigma>0 the reconstruction bias
delta = decode(z) - x cancels in the subtraction (it is a smooth function of the
latent, so delta(z+noise) ~ delta(z)), leaving x nudged along the flow decoder's
local Jacobian -- i.e. on-manifold diversity applied to a clean real, with no
fingerprint. Loads the saved checkpoint; NO retraining.

Writes <out>.s<sigma> for each sigma, schema identical to every other generator.
"""
import argparse, json, math
import numpy as np
import torch

from latent_anchor_invertible import (
    Flow, to_canonical, _decode_write, N, DIM, MIN_MT, DEVICE)


def load_kept(path, keep, cap=14000):
    Vk = []
    for i, line in enumerate(open(path)):
        if i >= cap: break
        pts = json.loads(line)["points"]
        if len(pts) < 4: continue
        v = to_canonical(pts)
        if v is not None and np.all(np.isfinite(v)):
            Vk.append(v[keep])
    return np.asarray(Vk, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--sigmas", default="0.0,0.1,0.2,0.35,0.5,0.75,1.0")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE)
    mean = np.asarray(ck["mean"], np.float64); std = np.asarray(ck["std"], np.float64)
    keep = list(ck["keep"]); D = ck["D"]
    flow = Flow(D, hidden=ck["hidden"], layers=ck["layers"]).to(DEVICE).double()
    flow.load_state_dict(ck["state_dict"]); flow.eval()
    for b in flow.blocks:
        b[0].inited = True
    print(f"[disp] loaded ckpt: D={D} layers={ck['layers']} hidden={ck['hidden']}", flush=True)

    Xk = load_kept(args.data, keep)
    base = (Xk - mean) / std                      # clean normalized reals (UNclamped)
    Xd = torch.tensor(base, dtype=torch.float64, device=DEVICE).clamp(-8, 8)  # for stable encode
    with torch.no_grad():
        Z = flow(Xd)[0]
        dec_z = flow.inverse(Z)                   # decode(encode(x)) == reconstruction
        zstd = Z.std(0, keepdim=True)
    base_t = torch.tensor(base, dtype=torch.float64, device=DEVICE)
    n = base.shape[0]
    rng = np.random.default_rng(1)

    for sig in [float(s) for s in args.sigmas.split(",")]:
        idx = rng.integers(0, n, args.n)
        noise = torch.tensor(rng.standard_normal((args.n, D)), dtype=torch.float64, device=DEVICE)
        with torch.no_grad():
            zp = Z[idx] + sig * zstd * noise
            disp = flow.inverse(zp) - dec_z[idx]  # flow displacement, fingerprint-cancelled
            out = base_t[idx] + disp              # clean real + displacement
        _decode_write(out.cpu().numpy(), std, mean, keep,
                      f"{args.out}.s{sig}", rng, f"disp(sig={sig})")


if __name__ == "__main__":
    main()
