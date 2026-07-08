#!/usr/bin/env python3
"""Defender: catch a multi-person replay pool via per-account style consistency.

The attacker beats every trajectory-statistics detector with pure replay from a
large no-repeat pool (attack_norepeat.py). To make the pool large they need many
recorded strokes - realistically from MANY people. But a real player is ONE
person: their strokes share a consistent motor style (speed/curvature/timing
habits). A session that replays many different people's strokes has a higher
WITHIN-SESSION style variance than any single real player.

This (1) checks that people are even distinguishable by style
(user-classification baseline), then (2) builds a session detector: real session
= N strokes from ONE user, attack session = N strokes from the multi-user pool;
featurize by within-session per-feature std; classify. Also sweeps how few
distinct people the attacker can pool before it looks single-person again.
Run: python -u scripts/detect_style_consistency.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features
from train_detector import SHAPE_ONLY_FEATURES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "human_movements.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SESS_PER_CLASS = 150


def main():
    rng = np.random.default_rng(0)
    by_user = {}
    with open(DATA) as f:
        for line in f:
            r = json.loads(line)
            if len(r["points"]) < 5:
                continue
            by_user.setdefault(r["user"], []).append(extract_features(r["points"]))
    users = sorted(by_user, key=lambda u: -len(by_user[u]))
    print(f"[style] {len(users)} users; strokes/user: "
          + ", ".join(f"{u}={len(by_user[u])}" for u in users[:12]), flush=True)

    # (1) baseline: are users distinguishable by style at all?
    Xall, yall = [], []
    for u in users:
        for f in by_user[u]:
            Xall.append(f); yall.append(u)
    dfall = pd.DataFrame(Xall)
    Xu = dfall[SHAPE_ONLY_FEATURES].to_numpy()
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    base = cross_val_score(clf, Xu, np.array(yall), cv=5).mean()
    print(f"[style] user-classification accuracy (chance={1/len(users):.2f}): {base:.3f}", flush=True)

    feat_arr = {u: pd.DataFrame(by_user[u])[SHAPE_ONLY_FEATURES].to_numpy() for u in users}
    big_users = [u for u in users if len(by_user[u]) >= 300]

    def real_session(N):
        u = rng.choice(big_users)
        A = feat_arr[u]
        idx = rng.choice(len(A), size=N, replace=len(A) < N)
        return A[idx]

    def attack_session(N, n_people):
        chosen = rng.choice(big_users, size=n_people, replace=False)
        rows = np.vstack([feat_arr[u] for u in chosen])
        idx = rng.choice(len(rows), size=N, replace=False)
        return rows[idx]

    def sess_feature(X):
        return np.concatenate([X.std(axis=0), X.mean(axis=0)])

    # (2) session detector: real(1 person) vs attack(pool of n_people)
    print("\n[style] session detector accuracy vs attacker pool diversity (N=200 strokes/session):", flush=True)
    print("        n_people in pool | detector acc (0.5 = attacker looks single-person)", flush=True)
    print("        " + "-" * 60, flush=True)
    out = []
    N = 200
    for n_people in [len(big_users), 5, 3, 2, 1]:
        if n_people > len(big_users):
            continue
        X, y = [], []
        for _ in range(SESS_PER_CLASS):
            X.append(sess_feature(attack_session(N, n_people))); y.append(1)
            X.append(sess_feature(real_session(N))); y.append(0)
        acc = cross_val_score(RandomForestClassifier(200, random_state=0, n_jobs=-1),
                              np.asarray(X), np.asarray(y), cv=5).mean()
        out.append({"n_people": int(n_people), "acc": float(acc)})
        tag = "  <- attacker's counter: pool few people" if n_people <= 2 else ""
        print(f"        {n_people:>16} | {acc:.3f}{tag}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "style_consistency.json").write_text(json.dumps(
        {"user_clf_acc": float(base), "n_users": len(users),
         "session_sweep": out}, indent=2))
    print(f"\n[style] wrote {RESULTS_DIR / 'style_consistency.json'}", flush=True)


if __name__ == "__main__":
    main()
