"""Cross-game UNIFIED skill axis — one flow, conditioned on game ID.

The per-game page trains a SEPARATE flow per game, so each z1 lives in its own
model: a chess-z1 and a Hanabi-z1 are not the same coordinate. Here we train ONE
conditional Normalizing Flow over a GAME-AGNOSTIC style descriptor
    P = [entropy, top1-share, switch-rate, length]
(computed the same way for every game; see games._profile), with the game ID as
context (one-hot). Skill is standardized WITHIN each game and pooled, and the
supervised axis z1 ≈ within-game-standardized-skill is applied across ALL games.

Result: a SINGLE latent whose z1 means the same thing everywhere — "how far above
/ below your own game's average you play" — so every game's players share one axis.
Unifying costs sharpness (we drop game-specific features), so we report the single
-flow AUROC next to each game's separate-flow AUROC honestly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(__file__))
import metrics
import games

GAMES = [  # key, name, mode, loader (returns X, S, P with return_prof=True)
    ("mspacman", "Ms.パックマン", "1人用", lambda: games.load_atari("mspacman", [0, 2, 3, 4, 5, 6, 7, 8, 9], return_prof=True)),
    ("spaceinvaders", "Space Invaders", "1人用", lambda: games.load_atari("spaceinvaders", [0, 1, 2, 3, 4, 5], return_prof=True)),
    ("chess", "チェス", "対戦", lambda: games.load_chess(return_prof=True)),
    ("go", "囲碁", "対戦", lambda: games.load_go(return_prof=True)),
    ("overcooked", "Overcooked", "協力", lambda: games.load_overcooked(return_prof=True)),
    ("hanabi", "Hanabi(2人)", "協力", lambda: games.load_hanabi(return_prof=True)),
]
PROF_NAMES = ["行動エントロピー", "最頻行動シェア", "切替率", "前後半のブレ(drift)"]


def _zscore(a, axis=0):
    a = np.asarray(a, float)
    return (a - a.mean(axis)) / (a.std(axis) + 1e-6)


def run(sep_results=None):
    """Train the unified game-conditioned flow; write figure; return metrics dict."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    sep = {r["key"]: r for r in (sep_results or [])}

    P_all, sz_all, sraw_all, gid_all, val_all = [], [], [], [], []
    meta = []  # (key, name, mode)
    for gi, (key, name, mode, loader) in enumerate(GAMES):
        X, S, P = loader()
        if P is None or len(P) < 50:
            print(f"crossgame: {name} skipped (no profile)"); continue
        Pz = _zscore(P)                       # within-game standardized style
        sz = _zscore(S)                       # within-game standardized skill (shared target)
        n = len(P)
        val = np.zeros(n, bool)
        val[rng.choice(n, int(n * 0.25), replace=False)] = True
        P_all.append(Pz); sz_all.append(sz); sraw_all.append(np.asarray(S, float))
        gid_all.append(np.full(n, len(meta))); val_all.append(val)
        meta.append((key, name, mode))

    ng = len(meta)
    P_all = np.concatenate(P_all).astype(np.float32)
    sz_all = np.concatenate(sz_all).astype(np.float32)
    sraw_all = np.concatenate(sraw_all)
    gid_all = np.concatenate(gid_all).astype(int)
    val_all = np.concatenate(val_all)
    onehot = np.eye(ng, dtype=np.float32)[gid_all]  # game-ID context

    tr = ~val_all
    xt = torch.tensor(P_all[tr]); ct = torch.tensor(onehot[tr]); st = torch.tensor(sz_all[tr])
    flow = games.make_flow(P_all.shape[1], context=ng)
    games._train(flow, xt, ct, st, supervised=True, epochs=500, lam=2.0)

    with torch.no_grad():
        z = flow(torch.tensor(onehot)).transform.inv(torch.tensor(P_all)).cpu().numpy()
    z1 = z[:, 0]

    # per-game metrics under the ONE flow (evaluated on that game's held-out set)
    per = []
    pooled_pred, pooled_lab = [], []
    for gi in range(ng):
        key, name, mode = meta[gi]
        m = val_all & (gid_all == gi)
        pred, sv = z1[m], sraw_all[m]
        lo, hi = np.percentile(sv, 33), np.percentile(sv, 67)
        keep = (sv <= lo) | (sv >= hi)
        lab = (sv[keep] >= hi).astype(int)
        auc = metrics.auc(pred[keep], lab)
        corr = metrics.pearson(pred, sv)
        pooled_pred.append(pred[keep]); pooled_lab.append(lab)
        per.append(dict(key=key, name=name, mode=mode, n=int((gid_all == gi).sum()),
                        auc_uni=round(float(auc), 3), corr_uni=round(float(corr), 3),
                        auc_sep=sep.get(key, {}).get("auc")))
    overall_auc = round(float(metrics.auc(np.concatenate(pooled_pred), np.concatenate(pooled_lab))), 3)
    overall_corr = round(float(metrics.pearson(z1[val_all], sz_all[val_all])), 3)

    _figure(z, gid_all, sz_all, val_all, meta, overall_corr)
    print(f"crossgame: unified flow over {ng} games, "
          f"overall AUROC={overall_auc} corr(z1,skill-z)={overall_corr}")
    for p in per:
        print(f"  {p['name']}: uni AUROC={p['auc_uni']} (sep {p['auc_sep']}) corr={p['corr_uni']}")
    return dict(overall_auc=overall_auc, overall_corr=overall_corr, per_game=per,
                n_games=ng, n_total=int(len(P_all)), lam=2.0, features=PROF_NAMES)


def _figure(z, gid, sz, val, meta, overall_corr):
    ng = len(meta)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    v = val
    for gi in range(ng):
        m = v & (gid == gi)
        ax[0].scatter(z[m, 0], z[m, 1], s=14, alpha=0.6, color=cmap(gi), label=meta[gi][1])
        ax[1].scatter(z[m, 0], sz[m], s=14, alpha=0.6, color=cmap(gi))
    ax[0].set_xlabel("z₁ ＝ 共通の技量軸（右＝上手）"); ax[0].set_ylabel("z₂")
    ax[0].set_title("全ゲームを1つの潜在に（単一フロー・ゲームIDで条件づけ）")
    ax[0].legend(fontsize=7, loc="best", framealpha=0.6)
    xs = z[v, 0]
    a, b = np.polyfit(xs, sz[v], 1)
    xl = np.array([xs.min(), xs.max()])
    ax[1].plot(xl, a * xl + b, color="#333", lw=1.2, ls="--")
    ax[1].set_xlabel("z₁（推定技量・全ゲーム共通）"); ax[1].set_ylabel("ゲーム内で標準化した技量")
    ax[1].set_title(f"z₁ vs 技量（ゲーム内z）  全体 r={overall_corr:.2f}")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "crossgame_axis.png")
    fig.savefig(out, dpi=110); plt.close(fig)


if __name__ == "__main__":
    run()
