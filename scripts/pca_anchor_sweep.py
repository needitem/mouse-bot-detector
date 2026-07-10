#!/usr/bin/env python3
"""The decisive anchor test: from a SMALL finite pool (K real strokes, the
attacker's limited recordings), PCA-latent-anchored replay generates N>>K
samples at each sigma. Measures the two things that must BOTH hold to beat the
replay dilemma:
  - single-move detection acc (are the samples still human?  ~0.5 = yes)
  - near-duplicate fraction  (did sigma break the finite-pool repeats?  ~0 = yes)
Pure replay from K strokes (sigma=0-ish) is human (0.5) but near-dup-caught;
generation is diverse but 0.85. The question: is there a sigma with BOTH?
"""
import json, math, sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features, FEATURE_NAMES
SHAPE=[f for f in FEATURE_NAMES if f not in {"sample_interval_mean","sample_interval_cv"}]
DATA=Path(__file__).resolve().parent.parent/"data"/"processed"
N=48; DIM=N*2+2; MIN_MT=40.0; EPS=0.0594   # project's near-dup canonical-distance threshold

def to_canonical(points):
    pts=np.asarray(points,float); x,y,t=pts[:,0],pts[:,1],pts[:,2]
    dx,dy=x[-1]-x[0],y[-1]-y[0]; dist=math.hypot(dx,dy); mt=t[-1]-t[0]
    if dist<5.0 or mt<MIN_MT: return None
    ang=math.atan2(dy,dx); c,s=math.cos(-ang),math.sin(-ang)
    rx=(x-x[0])*c-(y-y[0])*s; ry=(x-x[0])*s+(y-y[0])*c; rx,ry=rx/dist,ry/dist
    tg=np.linspace(t[0],t[-1],N); sx=np.interp(tg,t,rx); sy=np.interp(tg,t,ry)
    return np.concatenate([np.stack([sx,sy],1).ravel(),[math.log(dist),math.log(mt)]])

def load(path,cap=14000):
    V=[]
    for i,l in enumerate(open(path)):
        if i>=cap: break
        pts=json.loads(l)["points"]
        if len(pts)>=4:
            v=to_canonical(pts)
            if v is not None and np.all(np.isfinite(v)): V.append(v)
    return np.array(V,np.float32)

def to_points(v,rng):
    shape=v[:N*2].reshape(N,2)
    dist=math.exp(min(v[N*2],12)); mt=max(math.exp(min(v[N*2+1],8)),MIN_MT)
    xs=shape[:,0]*dist; ys=shape[:,1]*dist
    ang=rng.uniform(0,2*math.pi); c,s=math.cos(ang),math.sin(ang)
    rx=xs*c-ys*s; ry=xs*s+ys*c; t=np.linspace(0,mt,N)
    return [[float(a),float(b),float(u)] for a,b,u in zip(rx,ry,t)]

def feats(vlist,rng):
    F=[]
    for v in vlist:
        d=extract_features(to_points(v,rng))
        if isinstance(d,dict):
            f=np.array([d[k] for k in SHAPE],float)
            if np.all(np.isfinite(f)): F.append(f)
    return np.array(F)

def main():
    K=50; Ngen=3000
    rng=np.random.default_rng(1)
    X=load(str(DATA/"human_movements.jsonl"))
    DROP=[0,1,N*2-2,N*2-1]; keep=[i for i in range(DIM) if i not in DROP]
    Xk=X[:,keep]; mean=Xk.mean(0); Xc=Xk-mean
    pca=PCA(n_components=30).fit(Xc); Z=pca.transform(Xc); zstd=Z.std(0,keepdims=True)

    # detector: train on a human split; test human = disjoint split
    htr,hte=train_test_split(np.arange(len(X)),test_size=0.35,random_state=0)
    Fhtr=feats(X[htr][:3000],rng); Fhte=feats(X[hte][:2000],rng)

    pool=rng.choice(len(Z),K,replace=False)      # attacker's finite K recordings
    print(f"pool K={K}, generate N={Ngen}/sigma; near-dup eps={EPS}")
    print(f"{'sigma':>6} {'detect_acc':>11} {'nearN_frac':>11} {'median_NN':>10}")
    # reference: pure replay from the K pool (sample with repeats -> near-dups)
    for label, sig in [("replay",None),(0.05,0.05),(0.1,0.1),(0.2,0.2),(0.35,0.35),(0.5,0.5),(0.75,0.75)]:
        idx=rng.integers(0,K,Ngen); anchors=Z[pool[idx]]
        if sig is None:
            za=anchors                      # pure replay of pooled strokes (repeats)
        else:
            za=anchors + sig*zstd*rng.standard_normal((Ngen,30))
        Vk=pca.inverse_transform(za)+mean
        V=np.zeros((Ngen,DIM),np.float32); V[:,keep]=Vk; V[:,N*2-2]=1.0
        # near-dup on canonical SHAPE (drop constant dims already excluded in Vk)
        shp=Vk[:, :N*2-2]                   # shape part minus the 2 pinned end dims proxy
        nn=NearestNeighbors(n_neighbors=2).fit(shp); d,_=nn.kneighbors(shp)
        nnd=d[:,1]                          # distance to nearest OTHER sample
        neardup=float(np.mean(nnd<EPS))
        # detection
        Fg=feats(list(V),rng)
        Xtr=np.vstack([Fhtr,Fg[:len(Fhtr)]]); ytr=np.r_[np.zeros(len(Fhtr)),np.ones(min(len(Fg),len(Fhtr)))]
        clf=HistGradientBoostingClassifier(random_state=0).fit(Xtr,ytr)
        # balanced test: human holdout vs fresh generated (use remaining Fg)
        ng=min(len(Fhte),len(Fg))
        Xte=np.vstack([Fhte[:ng],Fg[:ng]]); yte=np.r_[np.zeros(ng),np.ones(ng)]
        acc=clf.score(Xte,yte)
        print(f"{str(label):>6} {acc:>11.3f} {neardup:>11.3f} {np.median(nnd):>10.4f}", flush=True)

if __name__=="__main__": main()
