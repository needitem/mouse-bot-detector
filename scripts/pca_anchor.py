#!/usr/bin/env python3
"""PCA latent-anchored replay: fast, reliable probe of the anchor hypothesis
without a neural flow. Take a REAL stroke's canonical shape, project to a PCA
latent, add small per-component Gaussian noise, reconstruct. Unlike kNN blend
(interpolate BETWEEN two reals -> off-manifold, 0.889), anchor perturbs AROUND
one real -> should stay near replay (0.5) at small sigma while breaking
near-duplicates. Sweep sigma; sigma=0 is the pure-reconstruction floor.
"""
import argparse, json, math, sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
N = 48; DIM = N*2 + 2; MIN_MT = 40.0

def to_canonical(points):
    pts = np.asarray(points, float)
    x, y, t = pts[:,0], pts[:,1], pts[:,2]
    dx, dy = x[-1]-x[0], y[-1]-y[0]
    dist = math.hypot(dx, dy); mt = t[-1]-t[0]
    if dist < 5.0 or mt < MIN_MT: return None
    ang = math.atan2(dy, dx); c, s = math.cos(-ang), math.sin(-ang)
    rx = (x-x[0])*c - (y-y[0])*s; ry = (x-x[0])*s + (y-y[0])*c
    rx, ry = rx/dist, ry/dist
    tg = np.linspace(t[0], t[-1], N)
    sx = np.interp(tg, t, rx); sy = np.interp(tg, t, ry)
    return np.concatenate([np.stack([sx,sy],1).ravel(), [math.log(dist), math.log(mt)]])

def load(path, cap=14000):
    V=[]
    for i,l in enumerate(open(path)):
        if i>=cap: break
        pts=json.loads(l)["points"]
        if len(pts)>=4:
            v=to_canonical(pts)
            if v is not None and np.all(np.isfinite(v)): V.append(v)
    return np.array(V,np.float32)

def decode_write(V, path, rng):
    out=[]
    for v in V:
        shape=v[:N*2].reshape(N,2)
        dist=math.exp(min(v[N*2],12)); mt=max(math.exp(min(v[N*2+1],8)),40.0)
        if not (np.isfinite(dist) and dist>=5.0): continue
        xs=shape[:,0]*dist; ys=shape[:,1]*dist
        ang=rng.uniform(0,2*math.pi); c,s=math.cos(ang),math.sin(ang)
        rx=xs*c-ys*s; ry=xs*s+ys*c
        t=np.linspace(0,mt,N)
        pts=[[float(a),float(b),float(u)] for a,b,u in zip(rx,ry,t)]
        if all(np.isfinite([p for r in pts for p in r])): out.append(pts)
    with open(path,"w") as f:
        for pts in out: f.write(json.dumps({"points":pts})+"\n")
    return len(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30, help="PCA components")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--sigmas", default="0.0,0.1,0.2,0.35,0.5,0.75")
    args=ap.parse_args()
    rng=np.random.default_rng(1)
    X=load(str(DATA/"human_movements.jsonl"))
    # drop the 4 constant canonicalization dims from PCA (start/end pinned)
    DROP=[0,1,N*2-2,N*2-1]; keep=[i for i in range(DIM) if i not in DROP]
    Xk=X[:,keep]
    mean=Xk.mean(0); Xc=Xk-mean
    pca=PCA(n_components=args.k).fit(Xc)
    Z=pca.transform(Xc)                      # (m, k) real latent coords
    zstd=Z.std(0, keepdims=True)
    print(f"[pca-anchor] {len(X)} human strokes, PCA k={args.k}, "
          f"explained var={pca.explained_variance_ratio_.sum():.3f}")
    for sig in [float(s) for s in args.sigmas.split(",")]:
        idx=rng.integers(0,len(Z),args.n)
        za=Z[idx] + sig*zstd*rng.standard_normal((args.n,args.k))
        Vk=pca.inverse_transform(za)+mean    # back to kept dims
        V=np.zeros((args.n,DIM),np.float32); V[:,keep]=Vk; V[:,N*2-2]=1.0
        path=DATA/f"pca_anchor_s{sig}_bot_movements.jsonl"
        m=decode_write(V,path,rng)
        print(f"[pca-anchor] sig={sig}: wrote {m} -> {path.name}", flush=True)

if __name__=="__main__": main()
