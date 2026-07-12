"""Export per-game JSON for the interactive "adjust & score" explorer (docs/play.html).

100% data-grounded, NO retrain: reuse the cached supervised skill axis z1
(`phase2._z1_all`, per-game cache) and the same tercile advice direction as
`phase2._card`. The browser does standardize -> kNN(k) -> score gauge + advice,
so all we ship is: real players (standardized features + z1 percentile + skill),
feature meta (name/min/max/step/is_lever), standardization mu/sd, the top-third
target (standardized) and the tercile improve direction.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import games
import phase2

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")
APP = os.path.join(DOCS, "app_data")
MAX_PLAYERS = 800  # subsample big games (chess 7712 / hanabi 1858) for a light payload

# axis strength (from the page results) — shown honestly so weak-axis games say so
AUROC = {"mspacman": 0.796, "spaceinvaders": 0.661, "chess": 0.615,
         "go": 0.546, "overcooked": 0.629, "hanabi": 0.826}


def _round(a, nd=4):
    return [round(float(v), nd) for v in np.asarray(a).ravel()]


def build(key, name, mode, label, X, S):
    z1 = phase2._z1_all(X, S, key=key)
    pz = phase2._pct(z1)          # skill-axis score (percentile of z1)
    ps = phase2._pct(S)           # true-skill percentile
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xz = (X - mu) / sd
    top, bot = ps >= 67, ps <= 33
    improve = Xz[top].mean(0) - Xz[bot].mean(0)   # robust "do more/less" direction
    target = Xz[top].mean(0)                       # strong-player norm (standardized)

    names = phase2.NAMES[key]
    levers = set(phase2.LEVERS[key])
    feats = []
    for j, nm in enumerate(names):
        col = X[:, j]
        lo, hi = float(col.min()), float(col.max())
        step = round((hi - lo) / 100.0, 6) if hi > lo else 0.01
        feats.append(dict(name=nm, idx=j, min=round(lo, 4), max=round(hi, 4),
                          med=round(float(np.median(col)), 4), step=step or 0.01,
                          is_lever=(j in levers)))

    # subsample players for payload (keep skill spread by stratified pick)
    idx = np.arange(len(X))
    if len(X) > MAX_PLAYERS:
        order = np.argsort(ps)
        idx = order[np.linspace(0, len(X) - 1, MAX_PLAYERS).astype(int)]
    players = [dict(fs=_round(Xz[i], 3), z1p=round(float(pz[i]), 1),
                    skill=round(float(S[i]), 2), sp=round(float(ps[i]), 1))
               for i in idx]

    return dict(key=key, name=name, mode=mode, label=label, n=int(len(X)),
                auroc=AUROC.get(key), skill_min=round(float(S.min()), 2),
                skill_max=round(float(S.max()), 2),
                mu=_round(mu), sd=_round(sd),
                improve=_round(improve, 3), target=_round(target, 3),
                feats=feats, players=players)


def run():
    os.makedirs(APP, exist_ok=True)
    index = []
    for key, name, mode, label, loader in phase2.GAMES:
        X, S = loader()
        if X is None or len(X) < 50:
            print(f"skip {key} (no data)")
            continue
        d = build(key, name, mode, label, X, S)
        with open(os.path.join(APP, f"{key}.json"), "w") as f:
            json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
        index.append(dict(key=key, name=name, mode=mode, label=label,
                          n=d["n"], auroc=d["auroc"]))
        print(f"{key:14s} n={d['n']:5d} feats={len(d['feats'])} players={len(d['players'])}")
    with open(os.path.join(APP, "index.json"), "w") as f:
        json.dump(index, f, ensure_ascii=False)
    print("wrote", APP)


if __name__ == "__main__":
    run()
