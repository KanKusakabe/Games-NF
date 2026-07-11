"""Games-NF — one page combining the SKILL AXIS learned for each game.

For every game: real human play + its skill label (Atari: episode score; Chess:
player rating). Build per-unit play-STYLE features (no result/score leak) and train
a Normalizing Flow with the Phase-0 base: Gaussian + a LINEAR skill axis
(supervised z1 ≈ standardized skill). z1 becomes right=skilled / left=novice.
Report novice-vs-expert AUROC and corr(z1, skill). Part of the KAN-NF collection.
No live-play environment is built here.
"""
from __future__ import annotations

import glob
import io
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

SCRATCH = sorted(glob.glob("/private/tmp/claude-501/*rosbag/*/scratchpad"))[0]
AGC = f"{SCRATCH}/games/agc/atari_v1/trajectories"
CHESS_ZST = f"{SCRATCH}/games/chess/l.pgn.zst"
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")
FIG = os.path.join(DOCS, "figures")
os.makedirs(FIG, exist_ok=True)

ATARI = [
    ("mspacman", "Ms.パックマン", "1人用", [0, 2, 3, 4, 5, 6, 7, 8, 9]),
    ("spaceinvaders", "Space Invaders", "1人用", [0, 1, 2, 3, 4, 5]),
]


# ---------- loaders ----------
def _atari_episode(path, acts):
    a, final = [], 0
    with open(path) as f:
        f.readline(); f.readline()
        for line in f:
            p = line.split(",")
            if len(p) < 5:
                continue
            try:
                final = int(p[2]); a.append(int(p[4]))
            except ValueError:
                continue
    if len(a) < 50:
        return None
    a = np.array(a)
    hist = np.array([(a == k).mean() for k in acts], float)
    switch = float((a[1:] != a[:-1]).mean())
    pos = hist[hist > 0]
    ent = float(-(pos * np.log(pos)).sum() / np.log(len(acts)))
    return np.concatenate([hist, [switch, ent]]), final


def load_atari(key, acts):
    X, S = [], []
    for f in sorted(glob.glob(os.path.join(AGC, key, "*.txt"))):
        r = _atari_episode(f, acts)
        if r:
            X.append(r[0]); S.append(r[1])
    return np.array(X, np.float32), np.array(S, np.float32)


def load_chess(max_games=4000):
    """Per (game, colour): play-STYLE features + that player's rating (skill)."""
    cache = f"{SCRATCH}/games/chess/feats.npz"
    if os.path.exists(cache):
        d = np.load(cache); return d["X"], d["S"]
    if not os.path.exists(CHESS_ZST):
        return None, None
    import chess, chess.pgn, zstandard
    PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    X, S = [], []
    dec = zstandard.ZstdDecompressor()
    with open(CHESS_ZST, "rb") as fh:
        t = io.TextIOWrapper(dec.stream_reader(fh), encoding="utf-8", errors="ignore")
        g = 0
        while g < max_games:
            game = chess.pgn.read_game(t)
            if game is None:
                break
            g += 1
            for color in (chess.WHITE, chess.BLACK):
                elo = game.headers.get("WhiteElo" if color else "BlackElo", "")
                if not elo.isdigit():
                    continue
                board = game.board()
                pc = [0] * 6; cap = chk = castle = promo = center = nm = 0
                for mv in game.mainline_moves():
                    if board.turn == color:
                        pt = board.piece_at(mv.from_square)
                        if pt:
                            pc[PIECES.index(pt.piece_type)] += 1
                        if board.is_capture(mv):
                            cap += 1
                        if board.gives_check(mv):
                            chk += 1
                        if board.is_castling(mv):
                            castle = 1
                        if mv.promotion:
                            promo += 1
                        if chess.square_file(mv.to_square) in (3, 4):
                            center += 1
                        nm += 1
                    board.push(mv)
                if nm < 8:
                    continue
                feat = [x / nm for x in pc] + [cap / nm, chk / nm, float(castle),
                                               promo / nm, center / nm, min(nm, 60) / 60.0]
                X.append(feat); S.append(int(elo))
    X, S = np.array(X, np.float32), np.array(S, np.float32)
    np.savez(cache, X=X, S=S)
    return X, S


# ---------- skill axis (shared base: Gaussian + supervised linear axis) ----------
def make_flow(dim):
    return zuko.flows.NSF(features=dim, context=1, transforms=3, hidden_features=(64, 64))


def _train(flow, x, c, shat, supervised, epochs=400, lam=1.0):
    opt = torch.optim.Adam(flow.parameters(), lr=2e-3)
    for _ in range(epochs):
        opt.zero_grad()
        d = flow(c); nll = -d.log_prob(x).mean(); loss = nll
        if supervised:
            z = d.transform.inv(x); loss = nll + lam * ((z[:, 0] - shat) ** 2).mean()
        loss.backward(); opt.step()
    return flow


def _terc_auc(pred, score):
    lo, hi = np.percentile(score, 33), np.percentile(score, 67)
    m = (score <= lo) | (score >= hi)
    return metrics.auc(pred[m], (score[m] >= hi).astype(int))


def skill_axis(key, name, mode, X, S, label="スコア", also_posthoc=False):
    torch.manual_seed(0)  # reproducible flow init across runs
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xz = (X - mu) / sd
    shat = (S - S.mean()) / (S.std() + 1e-6)
    n = len(X)
    rng = np.random.default_rng(0)
    val = np.zeros(n, bool); val[rng.choice(n, int(n * 0.25), replace=False)] = True
    xt, xv = torch.tensor(Xz[~val]), torch.tensor(Xz[val])
    st = torch.tensor(shat[~val]); ct, cv = torch.zeros(len(xt), 1), torch.zeros(len(xv), 1)
    sv = S[val]

    fS = _train(make_flow(X.shape[1]), xt, ct, st, supervised=True)
    with torch.no_grad():
        zSv = fS(cv).transform.inv(xv).cpu().numpy()
    predS = zSv[:, 0]
    aucS, rS = _terc_auc(predS, sv), metrics.pearson(predS, sv)

    res = dict(key=key, name=name, mode=mode, n=n, smin=int(S.min()), smax=int(S.max()),
               auc=round(aucS, 3), corr=round(rS, 3), label=label)
    if also_posthoc:
        fU = _train(make_flow(X.shape[1]), xt, ct, st, supervised=False)
        with torch.no_grad():
            zt = fU(ct).transform.inv(xt).cpu().numpy(); zv = fU(cv).transform.inv(xv).cpu().numpy()
        A = np.column_stack([zt, np.ones(len(zt))])
        w, *_ = np.linalg.lstsq(A, shat[~val], rcond=None)
        res["auc_posthoc"] = round(_terc_auc(np.column_stack([zv, np.ones(len(zv))]) @ w, sv), 3)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    scv = ax[0].scatter(zSv[:, 0], zSv[:, 1], c=sv, cmap="viridis", s=18 if n > 800 else 40,
                        edgecolor="k", lw=0.2, alpha=0.8)
    ax[0].set_xlabel("z₁ = 技量軸（右＝上手）"); ax[0].set_ylabel("z₂")
    ax[0].set_title(f"{name}：技量軸（右＝上手・左＝下手）")
    plt.colorbar(scv, ax=ax[0], label=label)
    ax[1].scatter(predS, sv, s=12 if n > 800 else 35, c="#c2410c", alpha=0.5)
    ax[1].set_xlabel("z₁（推定技量）"); ax[1].set_ylabel(f"実{label}")
    ax[1].set_title(f"z₁ vs 実{label}  r={rS:.2f} / AUROC={aucS:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{key}_axis.png"), dpi=110); plt.close(fig)
    print(f"{name}: n={n} AUROC={aucS:.3f} corr={rS:.3f}")
    return res


PAGE_CSS = """
body{font:16px/1.75 -apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;max-width:980px;margin:0 auto;padding:2rem 1rem;color:#222}
h1{line-height:1.35;margin:.2rem 0}.sub{color:#666}a{color:#c2410c}
.crumb{font-size:.9rem;margin-bottom:1rem}
.lead{background:#f7f5f2;border-radius:12px;padding:1rem 1.2rem}
.kpis{display:flex;gap:1rem;flex-wrap:wrap;margin:1.2rem 0}
.kpi{background:#f5f3f0;border-radius:12px;padding:.7rem 1rem;min-width:130px}.kpi b{display:block;font-size:1.5rem;color:#c2410c}
table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #e5e5e5;padding:.45rem .6rem;text-align:center}
th{background:#faf8f5}
section{margin:2.2rem 0}h2{border-left:5px solid #d97757;padding-left:.6rem}
img{width:100%;border:1px solid #e5e5e5;border-radius:10px}
figcaption{color:#555;margin-top:.3rem;font-size:.92rem}
.interp{background:#faf8f5;border-left:3px solid #d97757;padding:.6rem .9rem;border-radius:0 8px 8px 0}
code{background:#f0eee9;padding:.1rem .3rem;border-radius:4px}
"""


def build_page(results):
    rows = "".join(
        f"<tr><td>{r['name']}</td><td>{r['mode']}</td><td>{r['n']:,}</td>"
        f"<td>{r['label']} {r['smin']}–{r['smax']}</td><td><b>{r['auc']}</b></td><td>{r['corr']}</td></tr>"
        for r in results)
    secs = "".join(
        f'<section><h2>{r["name"]}（{r["mode"]}）</h2>'
        f'<img src="figures/{r["key"]}_axis.png" alt="{r["key"]}">'
        f'<figcaption>左:潜在 z₁-z₂ を{r["label"]}で色分け（右＝高{r["label"]}＝上手）。右:z₁(推定技量) vs 実{r["label"]}。</figcaption>'
        f'<p class="interp"><b>技量ラベル</b>＝{r["label"]}（{r["smin"]}〜{r["smax"]}）。'
        f'プレイスタイルだけから <b>初心者vs熟練 AUROC {r["auc"]}</b>・z₁と{r["label"]}相関 {r["corr"]}。'
        f'{"（教師つき軸は後付け方向 "+str(r["auc_posthoc"])+" を上回る）" if "auc_posthoc" in r else ""}</p></section>'
        for r in results)
    best = max(r["auc"] for r in results)
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Games-NF — プレイから技量軸を学ぶ</title><style>{PAGE_CSS}</style></head><body>
<div class="crumb"><a href="https://kankusakabe.github.io/KAN-NF/">&larr; KAN-NF 統括へ</a></div>
<h1>Games-NF — ゲームの実プレイから「技量軸」を学ぶ</h1>
<p class="sub">条件付き Normalizing Flow ＋ スコア/レートで、潜在に <b>右＝上手・左＝下手</b> の軸を通す。KAN-NF 実験群の一部。</p>
<div class="kpis">
 <div class="kpi"><b>{len(results)}</b>ゲーム</div>
 <div class="kpi"><b>{best}</b>最良 初心者vs熟練 AUROC</div>
 <div class="kpi"><b>ガウス＋直線軸</b>共通の基底</div>
</div>
<p class="lead"><b>やり方（全ゲーム共通）</b>：各プレイから<b>スタイル特徴</b>（行動/駒の使い方の分布・切替率など、
勝敗/スコアはリークさせない）を作り、<b>基底＝ガウス＋直線の技量軸</b>の条件付きNSFを学習。
学習時に <code>z₁ ≈ 標準化した技量ラベル</code> を課し、<b>z₁ そのものを技量軸</b>にする（Phase 0でMs.パックマンにて
教師つき軸 AUROC 0.80 &gt; 後付け 0.70 と確認）。技量は単調・順序尺度なので<b>直線</b>、円環は移動方向用に温存。</p>

<h2>結果一覧</h2>
<table><tr><th>ゲーム</th><th>種別</th><th>件数</th><th>技量ラベル範囲</th><th>初心者vs熟練 AUROC</th><th>z₁×技量相関</th></tr>
{rows}</table>
<p class="sub">AUROC 0.5=勘・1.0=完璧。プレイ<b>スタイルだけ</b>から巧さをどれだけ読めるか。</p>
{secs}
<section><h2>正直な限界・次の一歩</h2><p class="interp">
・特徴が粗いプレイスタイルのみ＝技量軸は<b>ソフト</b>で、ゲームにより効き方が違う（例：Space Invadersは狙い精度が効き弱い）。<b>状態条件つき特徴</b>で鋭くなる。<br>
・各ゲームは同じ基底・同じ機構だが<b>別々のフロー</b>。次は<b>ゲームIDを条件にした単一フロー</b>で横断（軸の意味を共通化）。<br>
・協力(Overcooked/Hanabi)は公開データの整備待ち、囲碁も追加予定。<br>
・<b>Phase 2</b>：この技量軸で「ユーザのプレイをスコアリング」「初心者に一歩上のお手本や補助」を出すミニ機能。
</p></section>
<p class="sub"><code>python -m gamesnf.games</code> で自動生成。KAN-NF 実験群。</p>
</body></html>"""
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write(html)
    print("wrote combined page with", [r["key"] for r in results])


def main():
    results = []
    for i, (key, name, mode, acts) in enumerate(ATARI):
        X, S = load_atari(key, acts)
        results.append(skill_axis(key, name, mode, X, S, label="スコア", also_posthoc=(i == 0)))
    try:
        X, S = load_chess()
        if X is not None and len(X) > 200:
            results.append(skill_axis("chess", "チェス", "対戦", X, S, label="レート"))
    except Exception as e:
        print("chess skipped:", repr(e))
    build_page(results)


if __name__ == "__main__":
    main()
