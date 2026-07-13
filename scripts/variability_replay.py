#!/usr/bin/env python3
"""Attacker counter to the residual detector: instead of a synthetic sinusoid
bend (elastic), perturb toward ANOTHER real stroke - mag*(shape_A - shape_B) in
canonical shape space. The residual to the nearest real neighbor then has the
NATURAL human variation spectrum, not a low-freq artifact, so it should evade the
residual detector. (The tradeoff, tested separately: it clusters harder -> the
session-distribution detector catches it more.)
"""
import json, math, sys
import numpy as np
DATA = __import__("pathlib").Path("data/processed")
NP=48
def canon(pts):
    p=np.asarray(pts,float);x,y,t=p[:,0],p[:,1],p[:,2]
    dx,dy=x[-1]-x[0],y[-1]-y[0];d=math.hypot(dx,dy)
    if d<5 or t[-1]-t[0]<40: return None,None
    a=math.atan2(dy,dx);c,s=math.cos(-a),math.sin(-a)
    rx=((x-x[0])*c-(y-y[0])*s)/d; ry=((x-x[0])*s+(y-y[0])*c)/d
    tg=np.linspace(t[0],t[-1],NP)
    return np.concatenate([np.interp(tg,t,rx),np.interp(tg,t,ry)]), (d, tg-t[0])
def main():
    mag=float(sys.argv[2]) if len(sys.argv)>2 else 0.07
    rng=np.random.default_rng(1)
    shapes=[];meta=[]
    for l in open(DATA/"human_movements.jsonl"):
        sh,m=canon(json.loads(l)["points"])
        if sh is not None: shapes.append(sh); meta.append(m)
        if len(shapes)>=6000: break
    shapes=np.array(shapes)
    out=[]
    for _ in range(4000):
        i,a,b=rng.integers(len(shapes)),rng.integers(len(shapes)),rng.integers(len(shapes))
        sh=shapes[i]+mag*(shapes[a]-shapes[b])       # perturb toward real variation
        d,t=meta[i]; phi=rng.uniform(0,2*math.pi); c,s=math.cos(phi),math.sin(phi)
        xs=sh[:NP]*d; ys=sh[NP:]*d
        rx=xs*c-ys*s; ry=xs*s+ys*c
        pts=[[float(rx[k]),float(ry[k]),float(t[k])] for k in range(NP)]
        out.append(pts)
    open(DATA/sys.argv[1],"w").write("\n".join(json.dumps({"points":p}) for p in out))
    print(f"[var] wrote {len(out)} variability(mag={mag}) flicks")
main()
