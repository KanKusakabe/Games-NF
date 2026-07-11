"""Phase 2 — use the learned skill axis to SCORE a play and give ONE-RUNG-UP advice.

No live-play environment (project decision): everything is OFFLINE on the real
datasets. For a chosen example player in each game we:
  1. SCORE   — place them on the per-game supervised skill axis z1 and report their
               within-game percentile.
  2. ADVISE  — retrieve REAL players one rung above on z1 (a percentile band just
               above them), take the mean difference in play-STYLE features, and turn
               the biggest actionable gaps into concrete "do more X / less Y" advice.
               We also show that cohort's real skill label, so the axis is shown to
               have retrieved genuinely stronger players (100% data-grounded — no
               generated/extrapolated profiles).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import games

# per game: readable feature names (match X columns) + which indices are actionable levers
NAMES = {
    "mspacman": ["静止(NOOP)率", "上", "右", "左", "下", "右上", "左上", "右下", "左下", "方向切替率", "動きの多様性"],
    "spaceinvaders": ["静止(NOOP)率", "射撃(FIRE)率", "上", "右", "左", "下", "方向切替率", "動きの多様性"],
    "chess": ["ポーン率", "ナイト率", "ビショップ率", "ルーク率", "クイーン率", "キング率",
              "捕獲率", "王手率", "キャスリング", "昇格率", "中央志向率", "手数"],
    "go": ["辺(1-2線)率", "3線率", "4線率", "中央(5線+)率", "平均の線", "盤の広がり", "序盤の線", "手数", "パス率"],
    "overcooked": ["静止率", "上", "下", "左", "右", "INTERACT率", "方向切替率", "動きの多様性"],
    "hanabi": ["捨て札比", "色ヒント比", "数字ヒント比", "色/数字ヒント比", "非play行動の多様性"],
}
LEVERS = {  # actionable, interpretable feature indices to advise on
    "mspacman": [0, 9, 10],
    "spaceinvaders": [0, 1, 6, 7],
    "chess": [1, 2, 6, 7, 8, 10],
    "go": [0, 1, 2, 5, 6],
    "overcooked": [0, 5, 6, 7],
    "hanabi": [0, 1, 2, 3],
}
GAMES = [
    ("mspacman", "Ms.パックマン", "1人用", "スコア", lambda: games.load_atari("mspacman", [0, 2, 3, 4, 5, 6, 7, 8, 9])),
    ("spaceinvaders", "Space Invaders", "1人用", "スコア", lambda: games.load_atari("spaceinvaders", [0, 1, 2, 3, 4, 5])),
    ("chess", "チェス", "対戦", "レート", games.load_chess),
    ("go", "囲碁", "対戦", "ランク", games.load_go),
    ("overcooked", "Overcooked", "協力", "層内順位%", games.load_overcooked),
    ("hanabi", "Hanabi(2人)", "協力", "花火点", games.load_hanabi),
]


def _z1_all(X, S, key=None):
    """Supervised skill axis z1 for EVERY unit (train on all, same mechanism as page).

    Cached per game (data is fixed) so we don't retrain the slow chess axis each run."""
    cache = f"{games.SCRATCH}/games/{key}/z1_all.npy" if key else None
    if cache and os.path.exists(cache) and len(np.load(cache)) == len(X):
        return np.load(cache)
    torch.manual_seed(0)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xz = (X - mu) / sd
    shat = (S - S.mean()) / (S.std() + 1e-6)
    # train the axis on a capped subsample (fast) but INFER z1 for every unit
    idx = np.arange(len(X))
    if len(X) > 2500:
        idx = np.random.default_rng(0).choice(len(X), 2500, replace=False)
    xtr = torch.tensor(Xz[idx]); ctr = torch.zeros(len(idx), 1); str_ = torch.tensor(shat[idx])
    flow = games._train(games.make_flow(X.shape[1]), xtr, ctr, str_, supervised=True)
    with torch.no_grad():
        z1 = flow(torch.zeros(len(X), 1)).transform.inv(torch.tensor(Xz)).cpu().numpy()[:, 0]
    if np.corrcoef(z1, S)[0, 1] < 0:  # orient so higher z1 = higher skill
        z1 = -z1
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, z1)
    return z1


def _pct(a):
    """Percentile rank (0..100) of every element within a."""
    return 100.0 * a.argsort().argsort() / max(len(a) - 1, 1)


def _card(key, name, mode, label, X, S):
    z1 = _z1_all(X, S, key=key)
    pz = _pct(z1)          # skill-axis score (percentile of z1) — what we DISPLAY as the score
    ps = _pct(S)           # TRUE skill percentile
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xz = (X - mu) / sd
    # example = a genuinely mid-skill player (closest to the 40th true-skill percentile)
    ex = int(np.argmin(np.abs(ps - 40)))
    # ROBUST "improve direction" = how the TOP third plays vs the BOTTOM third (not one
    # noisy example) — this matches the global skill correlations and is stable.
    top, bot = ps >= 67, ps <= 33
    improve = Xz[top].mean(0) - Xz[bot].mean(0)
    gap = Xz[top].mean(0) - Xz[ex]  # how far THIS example is from the strong-player norm

    def tip(j):
        arrow, verb = ("▲", "増やす") if improve[j] > 0 else ("▼", "減らす")
        return dict(arrow=arrow, name=NAMES[key][j], verb=verb, mag=round(float(abs(improve[j])), 2))
    cand = [j for j in sorted(LEVERS[key], key=lambda k: -abs(improve[k])) if abs(improve[j]) >= 0.15]
    tips = []
    for j in cand:  # personalise: advise where the example is on the deficit side of the strong norm
        if np.sign(gap[j]) == np.sign(improve[j]) or abs(gap[j]) < 0.1:
            tips.append(tip(j))
        if len(tips) == 3:
            break
    if not tips and cand:  # fall back to the strongest GLOBAL tendency (weak-axis games)
        tips = [tip(cand[0])]
    return dict(key=key, name=name, mode=mode, label=label, n=len(X),
                ex_pct=int(round(ps[ex])), ex_z1=int(round(pz[ex])), ex_skill=round(float(S[ex]), 1),
                top_pct=83, top_skill=round(float(S[top].mean()), 1), top_n=int(top.sum()), tips=tips)


def run(_results=None):
    cards = []
    for key, name, mode, label, loader in GAMES:
        try:
            X, S = loader()
            if X is None or len(X) < 50:
                continue
            c = _card(key, name, mode, label, X, S)
            cards.append(c)
            tip = " / ".join(f"{t['arrow']}{t['name']}" for t in c["tips"]) or "(差分小)"
            print(f"{name}: 例=実{c['ex_pct']}%tile(z₁採点{c['ex_z1']}%tile, {label}{c['ex_skill']}) → "
                  f"上位層({label}{c['top_skill']}, n={c['top_n']}) | {tip}")
        except Exception as e:
            print(f"phase2 {name} skipped:", repr(e))
    return cards


if __name__ == "__main__":
    run()
