"""Phase 0 — decide the base / skill-axis mechanism on Ms. Pac-Man.

Goal: a latent where RIGHT = skilled, LEFT = novice, learned from real human play
(Atari Grand Challenge) + the episode's final score (= skill label).

We model a per-episode PLAY-STYLE feature vector x (action-direction histogram +
switch-rate + entropy; NO survival/score leak) and compare two ways to obtain a
readable skill axis:
  U) unsupervised flow p(x) → find the skill direction AFTER, by regressing score
     on the latent (= "train, then use score" without retraining the flow).
  S) supervised skill axis: train p(x) with an extra loss z1 ≈ standardized-score,
     so latent coordinate z1 IS the skill axis (right=good/left=bad).

Both use a Gaussian base on a LINEAR skill axis (skill is monotonic, never circular).
Winner (cleaner skill readout at comparable NLL) is the base/mechanism for all games.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(__file__))
import metrics
import pages

SCRATCH = sorted(glob.glob("/private/tmp/claude-501/*rosbag/*/scratchpad"))[0]
DATA = os.environ.get("AGC", f"{SCRATCH}/games/agc/atari_v1/trajectories/mspacman")
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")
FIG = os.path.join(DOCS, "figures")
os.makedirs(FIG, exist_ok=True)

# Ms. Pac-Man action set (no fire): NOOP + 4 dirs + 4 diagonals
ACTS = [0, 2, 3, 4, 5, 6, 7, 8, 9]
FEAT_NAMES = ["noop", "up", "right", "left", "down", "upR", "upL", "dnR", "dnL",
              "switch_rate", "entropy"]


def episode_features(path):
    acts, final_score = [], 0
    with open(path) as f:
        f.readline()  # 'db traj id : N'
        f.readline()  # header
        for line in f:
            p = line.split(",")
            if len(p) < 5:
                continue
            try:
                final_score = int(p[2])
                acts.append(int(p[4]))
            except ValueError:
                continue
    if len(acts) < 50:
        return None
    a = np.array(acts)
    hist = np.array([(a == k).mean() for k in ACTS], dtype=np.float64)
    switch = float((a[1:] != a[:-1]).mean())
    p = hist[hist > 0]
    ent = float(-(p * np.log(p)).sum() / np.log(len(ACTS)))
    x = np.concatenate([hist, [switch, ent]])
    return x, final_score


def load():
    X, S = [], []
    for f in sorted(glob.glob(os.path.join(DATA, "*.txt"))):
        r = episode_features(f)
        if r is None:
            continue
        X.append(r[0]); S.append(r[1])
    X = np.array(X, dtype=np.float32); S = np.array(S, dtype=np.float32)
    return X, S


def make_flow(dim):
    return zuko.flows.NSF(features=dim, context=1, transforms=3, hidden_features=(64, 64))


def train(flow, x, c, shat, supervised, epochs=400, lam=1.0):
    opt = torch.optim.Adam(flow.parameters(), lr=2e-3)
    for ep in range(epochs):
        opt.zero_grad()
        d = flow(c)
        nll = -d.log_prob(x).mean()
        loss = nll
        if supervised:
            z = d.transform.inv(x)
            loss = nll + lam * ((z[:, 0] - shat) ** 2).mean()
        loss.backward()
        opt.step()
    return flow


def latent(flow, x, c):
    with torch.no_grad():
        return flow(c).transform.inv(x).cpu().numpy()


def val_nll(flow, x, c):
    with torch.no_grad():
        return float(-flow(c).log_prob(x).mean())


def terc_auc(pred, score):
    lo, hi = np.percentile(score, 33), np.percentile(score, 67)
    m = (score <= lo) | (score >= hi)
    lab = (score[m] >= hi).astype(int)   # expert=1
    return metrics.auc(pred[m], lab)


def main():
    X, S = load()
    print(f"loaded {len(X)} Ms.Pacman episodes; score range {int(S.min())}..{int(S.max())}")
    # standardise features + skill
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xz = (X - mu) / sd
    shat = (S - S.mean()) / (S.std() + 1e-6)
    n = len(X)
    rng = np.random.default_rng(0)
    val = np.zeros(n, bool); val[rng.choice(n, int(n * 0.25), replace=False)] = True
    xt = torch.tensor(Xz[~val]); xv = torch.tensor(Xz[val])
    st = torch.tensor(shat[~val]); ct = torch.zeros(len(xt), 1); cv = torch.zeros(len(xv), 1)
    sv_raw, st_raw = S[val], S[~val]

    # ---- U: unsupervised + post-hoc skill direction ----
    fU = train(make_flow(X.shape[1]), xt, ct, st, supervised=False)
    zt, zv = latent(fU, xt, ct), latent(fU, xv, cv)
    # least-squares skill direction on TRAIN latents
    A = np.column_stack([zt, np.ones(len(zt))])
    w, *_ = np.linalg.lstsq(A, shat[~val], rcond=None)
    predU = np.column_stack([zv, np.ones(len(zv))]) @ w
    aucU = terc_auc(predU, sv_raw); rU = metrics.pearson(predU, sv_raw); nllU = val_nll(fU, xv, cv)

    # ---- S: supervised skill axis (z1 ≈ score) ----
    fS = train(make_flow(X.shape[1]), xt, ct, st, supervised=True, lam=1.0)
    zSt, zSv = latent(fS, xt, ct), latent(fS, xv, cv)
    predS = zSv[:, 0]
    aucS = terc_auc(predS, sv_raw); rS = metrics.pearson(predS, sv_raw); nllS = val_nll(fS, xv, cv)

    print(f"U(unsup+posthoc): AUROC {aucU:.3f} corr {rU:.3f} NLL {nllU:.3f}")
    print(f"S(supervised axis): AUROC {aucS:.3f} corr {rS:.3f} NLL {nllS:.3f}")

    # ---- figure 1: the skill axis (supervised) — z1 vs z2 coloured by score ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    sc = ax[0].scatter(zSv[:, 0], zSv[:, 1], c=sv_raw, cmap="viridis", s=45, edgecolor="k", lw=0.3)
    ax[0].set_xlabel("z₁ = 技量軸（右＝高スコア）"); ax[0].set_ylabel("z₂")
    ax[0].set_title("技量軸（教師つき）：右＝上手・左＝下手")
    plt.colorbar(sc, ax=ax[0], label="エピソード最終スコア")
    ax[1].scatter(predS, sv_raw, s=40, c="#c2410c", alpha=0.7)
    ax[1].set_xlabel("z₁（推定技量）"); ax[1].set_ylabel("実スコア")
    ax[1].set_title(f"z₁ は実スコアと相関 r={rS:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "skill_axis.png"), dpi=110); plt.close(fig)

    # ---- figure 2: bakeoff bars (skill readout + NLL) ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    b = ax[0].bar(["U 後付け", "S 教師つき軸"], [aucU, aucS], color=["#8b93a1", "#d97757"])
    ax[0].axhline(0.5, ls="--", c="gray"); ax[0].set_ylim(0.4, 1.0)
    ax[0].set_ylabel("初心者vs熟練 AUROC"); ax[0].set_title("技量の読み取り（0.5=勘）")
    for bar, v in zip(b, [aucU, aucS]):
        ax[0].annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom")
    b2 = ax[1].bar(["U", "S"], [nllU, nllS], color=["#8b93a1", "#d97757"])
    ax[1].set_ylabel("held-out NLL（低いほど密度が良い）"); ax[1].set_title("密度の当てはまり")
    for bar, v in zip(b2, [nllU, nllS]):
        ax[1].annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "bakeoff.png"), dpi=110); plt.close(fig)

    winner = "S（教師つき技量軸）" if aucS >= aucU else "U（教師なし＋後付け方向）"
    var = dict(
        id="phase0", title="Phase 0：技量軸の基底をMs.パックマンで決める",
        tagline="実人間プレイ（Grand Challenge）＋最終スコアで、潜在に『右＝上手・左＝下手』の軸を通す。基底はガウス＋直線技量軸。",
        status="done",
        metrics={"エピソード数": int(n), "スコア範囲": f"{int(S.min())}–{int(S.max())}",
                 "AUROC(教師つき)": round(aucS, 3), "AUROC(後付け)": round(aucU, 3),
                 "採用": winner},
        data="<b>Atari Grand Challenge</b> の <b>Ms.パックマン 実人間プレイ</b>（"
             f"{int(n)}エピソード）。1エピソード＝行動列＋最終スコア。<b>技量ラベル＝最終スコア</b>"
             f"（{int(S.min())}〜{int(S.max())}と実際に大きくばらつく＝本物の初心者〜上級）。"
             "各エピソードから<b>プレイスタイル特徴</b>（行動方向の分布9＋切替率＋エントロピー）を作る"
             "（生存時間やスコアは特徴に入れない＝リークなし）。",
        method="条件付き Neural Spline Flow で <code>p(スタイル特徴)</code> を学習し、潜在 z を得る。"
               "<b>技量は必ず直線軸</b>（単調・順序尺度なので円は不適）。2方式を比較：<br>"
               "<b>U 教師なし＋後付け</b>：普通に学習→潜在でスコアを線形回帰＝技量方向を後から取り出す"
               "（再学習なし）。<br><b>S 教師つき技量軸</b>：学習時に <code>z₁≈標準化スコア</code> の項を足し、"
               "<b>z₁ そのものを技量軸</b>にする（右＝上手／左＝下手）。",
        results=f"技量の読み取り（初心者vs熟練 AUROC）は <b>教師つき {aucS:.2f} / 後付け {aucU:.2f}</b>、"
                f"z₁と実スコアの相関 r={rS:.2f}。密度 held-out NLL は U {nllU:.2f} / S {nllS:.2f}。"
                f"→ <b>採用＝{winner}</b>。全ゲームでこの基底（ガウス＋直線技量軸）を固定する。",
        figures=[("skill_axis.png", "左:潜在 z₁-z₂ を実スコアで色分け（右ほど高スコア＝上手）。右:z₁ vs 実スコア。"),
                 ("bakeoff.png", "左:技量読み取りAUROC（U後付け vs S教師つき）。右:held-out NLL。")],
        howto="<b>左上図</b>：点=1エピソード。色=最終スコア。<b>右に行くほど高スコア（上手）</b>に"
              "並んでいれば技量軸が通っている。<br><b>右上図</b>：z₁（推定技量）と実スコアが右上がり＝"
              "プレイ内容だけから巧さを読めている。<br><b>下段</b>：AUROCが高い方式を採用、NLLが極端に"
              "悪化していないかも確認。",
        interpretation="<b>示すこと</b>：実人間プレイの<b>スタイルだけ</b>から技量が（ある程度）読め、"
                       "潜在に『右＝上手・左＝下手』の軸を構成できる。<b>基底＝ガウス＋直線技量軸</b>が妥当"
                       "（円は移動方向用で、技量には使わない）。<br><b>なぜNFか</b>：密度として学ぶので、"
                       "技量軸＝生成の技量ノブ（初心者風〜熟練風を作る）にそのまま繋がる。<br>"
                       "<b>正直な限界</b>：粗い行動スタイルだけなので技量軸は<b>ソフト</b>（AUROCは中程度の想定）。"
                       "状態条件つき特徴（次フェーズ）で鋭くなる。多次元技量は z₂ 以降で。")
    pages.write_all(DOCS, "Games-NF", REPO_DESC, [var], RAW_INTRO, OUTLOOK)
    print("wrote docs/ pages. winner:", winner)


REPO_DESC = ("ゲームの実人間プレイを条件付き Normalizing Flow で学習し、潜在に「右＝上手・左＝下手」の"
             "技量軸を通す試み。1人用/協力/対戦の複数ゲームを同じ基底で扱う（Phase 0＝基底の決定）。")
RAW_INTRO = (
    "<b>Phase 0 の生データ</b>＝Atari Grand Challenge の Ms.パックマン実人間プレイ（"
    "<code>atari_v1/trajectories/mspacman/*.txt</code>）。各ファイル＝1エピソードで、列は "
    "<code>frame, reward, score, terminal, action</code>。<b>最終スコア＝技量ラベル</b>。<br>"
    "<b>1レコードの実例</b>：あるエピソードは 2,270フレーム・最終スコア1,970、行動は NOOP/上/右/左/下…の系列。"
    "全体でスコアは数十〜1万超まで分布＝<b>実在の初心者〜上級</b>。")
OUTLOOK = (
    "<p>ここで決めた基底（ガウス＋直線技量軸）を、次からこう広げる：</p><ul>"
    "<li><b>＋状態条件つき特徴</b>（画面/位置）→ 技量軸が鋭くなる（今はスタイルだけでソフト）。</li>"
    "<li><b>＋ゲームIDを条件にした単一フロー</b>→ Space Invaders/Overcooked/Hanabi/チェス/囲碁を"
    "同じ技量軸で横断（方向の意味も共通化）。</li>"
    "<li><b>＋技量ノブで生成</b>→ 初心者風〜熟練風のプレイを作り、環境で実際にプレイ。</li>"
    "<li><b>＋z₂ 以降</b>→ 多次元技量（エイム/回避/管理…）。</li></ul>")


if __name__ == "__main__":
    main()
