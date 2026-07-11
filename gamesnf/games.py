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

def _find_scratch():
    """Pick the scratchpad that actually holds the data.

    Several sessions each create a `.../<uuid>/scratchpad`; a new (empty) one can
    sort before the one with our `games/` data, so never blindly take sorted[0].
    Prefer $GAMESNF_DATA, then any scratchpad containing `games/`, else sorted[0].
    """
    env = os.environ.get("GAMESNF_DATA")
    if env and os.path.isdir(env):
        return env
    cands = sorted(glob.glob("/private/tmp/claude-501/*rosbag/*/scratchpad"))
    for d in cands:
        if os.path.isdir(os.path.join(d, "games")):
            return d
    return cands[0] if cands else "."


SCRATCH = _find_scratch()
AGC = f"{SCRATCH}/games/agc/atari_v1/trajectories"
CHESS_ZST = f"{SCRATCH}/games/chess/l.pgn.zst"
GO_JSON = f"{SCRATCH}/games/go/sample-1k.json.gz"
OVERCOOKED_PKL = f"{SCRATCH}/games/overcooked/2019_hh_trials_all.pickle"
HANABI_DIR = f"{SCRATCH}/games/hanabi"
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
    return np.concatenate([hist, [switch, ent]]), final, _profile(hist, a)


def load_atari(key, acts, return_prof=False):
    X, S, P = [], [], []
    for f in sorted(glob.glob(os.path.join(AGC, key, "*.txt"))):
        r = _atari_episode(f, acts)
        if r:
            X.append(r[0]); S.append(r[1]); P.append(r[2])
    X, S, P = np.array(X, np.float32), np.array(S, np.float32), np.array(P, np.float32)
    return (X, S, P) if return_prof else (X, S)


def load_chess(max_games=4000, return_prof=False):
    """Per (game, colour): play-STYLE features + that player's rating (skill)."""
    cache = f"{SCRATCH}/games/chess/feats.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        if not return_prof:
            return d["X"], d["S"]
        if "P" in d:
            return d["X"], d["S"], d["P"]  # else fall through and recompute to add P
    if not os.path.exists(CHESS_ZST):
        return (None, None, None) if return_prof else (None, None)
    import chess, chess.pgn, zstandard
    PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    X, S, P = [], [], []
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
                pc = [0] * 6; cap = chk = castle = promo = center = nm = 0; pcseq = []
                for mv in game.mainline_moves():
                    if board.turn == color:
                        pt = board.piece_at(mv.from_square)
                        if pt:
                            pc[PIECES.index(pt.piece_type)] += 1
                            pcseq.append(PIECES.index(pt.piece_type))
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
                X.append(feat); S.append(int(elo)); P.append(_profile(pc, pcseq))
    X, S, P = np.array(X, np.float32), np.array(S, np.float32), np.array(P, np.float32)
    np.savez(cache, X=X, S=S, P=P)
    return (X, S, P) if return_prof else (X, S)


def _entropy(p):
    """Normalised Shannon entropy of a distribution (0..1)."""
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def _profile(dist, seq):
    """Game-AGNOSTIC style descriptor [entropy, top1-share, switch-rate, drift].

    Same construction for every game (how varied / concentrated / switchy the play
    is, and how much the action mix DRIFTS between the first and second half), so it
    can feed ONE cross-game flow whose z1 has a shared meaning. Deliberately NO
    length: episode length = survival time in Atari, which would leak the score.
    """
    dist = np.asarray(dist, float); s = dist.sum()
    d = dist / s if s > 0 else dist
    seq = np.asarray(seq)
    switch = float((seq[1:] != seq[:-1]).mean()) if seq.size > 1 else 0.0
    if seq.size >= 4:  # non-stationarity: TV distance between 1st- and 2nd-half mix
        h = seq.size // 2; cats = np.unique(seq)
        p1 = np.array([(seq[:h] == c).mean() for c in cats])
        p2 = np.array([(seq[h:] == c).mean() for c in cats])
        drift = float(0.5 * np.abs(p1 - p2).sum())
    else:
        drift = 0.0
    return [_entropy(d), float(d.max()) if d.size else 0.0, switch, drift]


def load_go(return_prof=False):
    """Go (19x19, versus). Per (game, colour): board-geometry play STYLE + OGS rank.

    Real human OGS games (za3k dump). Features = where the player puts stones
    (edge / 3rd / 4th / centre line rates, opening line, spread, pass, length) —
    no winner/outcome leak. Skill = that colour's OGS rank (higher = stronger).
    """
    cache = f"{SCRATCH}/games/go/feats.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        if not return_prof:
            return d["X"], d["S"]
        if "P" in d:
            return d["X"], d["S"], d["P"]
    if not os.path.exists(GO_JSON):
        return (None, None, None) if return_prof else (None, None)
    import gzip
    X, S, P = [], [], []
    with gzip.open(GO_JSON, "rt", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = __import__("json").loads(line)
            except Exception:
                continue
            if g.get("width") != 19 or g.get("height") != 19 or g.get("handicap", 0):
                continue
            moves = g.get("moves") or []
            for start, col in ((0, "black"), (1, "white")):
                rank = g["players"][col].get("rank")
                if rank is None:
                    continue
                mv = moves[start::2]
                xs, ys, npass = [], [], 0
                for m in mv:
                    x, y = m[0], m[1]
                    if x is None or y is None or x < 0 or y < 0 or x > 18 or y > 18:
                        npass += 1; continue
                    xs.append(x); ys.append(y)
                if len(xs) < 15:
                    continue
                xs = np.array(xs); ys = np.array(ys)
                line = np.minimum(np.minimum(xs, 18 - xs), np.minimum(ys, 18 - ys)) + 1
                bucket = np.clip(line, 1, 5) - 1  # line-region sequence: 0=edge..4=centre+
                dist = np.array([np.mean(line <= 2), np.mean(line == 3),
                                 np.mean(line == 4), np.mean(line >= 5)], float)
                feat = [dist[0], dist[1], dist[2], dist[3],
                        float(line.mean()) / 8.0,
                        float(np.sqrt(xs.var() + ys.var())) / 9.0,
                        float(line[:6].mean()) / 8.0,
                        min(len(mv), 300) / 300.0, npass / max(len(mv), 1)]
                X.append(feat); S.append(float(rank))
                P.append(_profile(dist, bucket))
    X, S, P = np.array(X, np.float32), np.array(S, np.float32), np.array(P, np.float32)
    np.savez(cache, X=X, S=S, P=P)
    return (X, S, P) if return_prof else (X, S)


_OC_DIR = {"[0, 0]": 0, "[0, -1]": 1, "[0, 1]": 2, "[-1, 0]": 3, "[1, 0]": 4}  # stay,U,D,L,R


def load_overcooked(return_prof=False):
    """Overcooked (co-op). Per (trial, player): action-mix STYLE + team score.

    Real human-human trials (HumanCompatibleAI). Features = that player's action
    distribution (stay/up/down/left/right/interact), switch rate, entropy — no
    score leak. Skill = team score, ranked WITHIN layout (percentile) so layouts
    of different difficulty are comparable.
    """
    cache = f"{SCRATCH}/games/overcooked/feats.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        if not return_prof:
            return d["X"], d["S"]
        if "P" in d:
            return d["X"], d["S"], d["P"]
    if not os.path.exists(OVERCOOKED_PKL):
        return (None, None, None) if return_prof else (None, None)
    import json as _json
    import pandas as pd
    df = pd.read_pickle(OVERCOOKED_PKL).sort_values(["trial_id", "cur_gameloop"])
    fin = df.groupby("trial_id")["score_total"].max()
    lay = df.groupby("trial_id")["layout_name"].first()
    X, raw, lays, PROF = [], [], [], []
    for tid, sub in df.groupby("trial_id"):
        seqs = [[], []]
        for ja in sub["joint_action"]:
            try:
                pa = _json.loads(ja.replace("'", '"'))
            except Exception:
                try:
                    pa = eval(ja)
                except Exception:
                    continue
            for p in (0, 1):
                a = pa[p]
                seqs[p].append(5 if str(a).upper() == "INTERACT" else _OC_DIR.get(str(a), 0))
        for p in (0, 1):
            s = np.array(seqs[p])
            if len(s) < 50:
                continue
            hist = np.array([(s == k).mean() for k in range(6)], float)
            switch = float((s[1:] != s[:-1]).mean())
            X.append(list(hist) + [switch, _entropy(hist)])
            raw.append(float(fin[tid])); lays.append(str(lay[tid]))
            PROF.append(_profile(hist, s))
    X = np.array(X, np.float32); raw = np.array(raw, np.float32); lays = np.array(lays)
    PROF = np.array(PROF, np.float32)
    pct = np.zeros(len(raw), np.float32)  # within-layout percentile skill
    for L in np.unique(lays):
        m = lays == L; v = raw[m]
        pct[m] = 100.0 * v.argsort().argsort() / max(len(v) - 1, 1)
    np.savez(cache, X=X, S=pct, P=PROF)
    return (X, pct, PROF) if return_prof else (X, pct)


def load_hanabi(return_prof=False):
    """Hanabi 2-player (co-op). Per game: NON-PLAY action-style + fireworks score.

    Real human games from Hanab Live (AH2AC2). Actions 0-4 discard / 5-9 play /
    10-14 colour-hint / 15-19 rank-hint (20 = no-op). We deliberately DROP the
    play/discard *counts* (score = successful plays, so raw play-rate leaks the
    label) and use the composition AMONG non-play actions + colour-vs-rank hint
    preference. Skill = final fireworks score (13-25).
    """
    from safetensors.numpy import load_file
    A, SC, NA = [], [], []
    for f in ("2_player_games_train_1k.safetensors", "2_player_games_val.safetensors"):
        path = os.path.join(HANABI_DIR, f)
        if not os.path.exists(path):
            continue
        d = load_file(path)
        A.append(d["actions"]); SC.append(d["scores"]); NA.append(d["num_actions"])
    if not A:
        return (None, None, None) if return_prof else (None, None)
    A = np.concatenate(A); SC = np.concatenate(SC); NA = np.concatenate(NA)
    X, S, P = [], [], []
    for i in range(len(A)):
        na = int(NA[i])
        flat = A[i, :na].reshape(-1)
        flat = flat[flat != 20]
        typ = np.where(flat <= 4, 0, np.where(flat <= 9, 1, np.where(flat <= 14, 2, 3)))
        keep = typ != 1  # non-play type sequence: 0 discard / 2 colour-hint / 3 rank-hint
        seq = typ[keep]
        disc = int(np.sum(flat <= 4))
        colh = int(np.sum((flat >= 10) & (flat <= 14)))
        rnkh = int(np.sum((flat >= 15) & (flat <= 19)))
        nonplay = disc + colh + rnkh
        if nonplay < 5:
            continue
        ds, cs, rs = disc / nonplay, colh / nonplay, rnkh / nonplay
        X.append([ds, cs, rs, colh / (colh + rnkh + 1e-9), _entropy([ds, cs, rs])])
        S.append(float(SC[i]))
        P.append(_profile([ds, cs, rs], seq))
    X, S, P = np.array(X, np.float32), np.array(S, np.float32), np.array(P, np.float32)
    return (X, S, P) if return_prof else (X, S)


# ---------- skill axis (shared base: Gaussian + supervised linear axis) ----------
def make_flow(dim, context=1):
    return zuko.flows.NSF(features=dim, context=context, transforms=3, hidden_features=(64, 64))


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


def _cross_section(cg):
    """HTML for the cross-game UNIFIED-axis section (one flow, game-ID context).

    Two ways to unify: (A) 4 universal style features; (B) per-game encoders that
    keep each game's own features. Shown side by side, honestly."""
    if not cg:
        return ""
    has_enc = cg.get("overall_auc_enc") is not None
    def mark(v, s):
        return "" if (v is None or s is None) else (" ▲" if v >= s else " ▽")
    def cell(p):
        s, u, e = p.get("auc_sep"), p["auc_uni"], p.get("auc_enc")
        ecol = f'<td><b>{e}</b>{mark(e, s)}</td>' if has_enc else ""
        return (f'<tr><td>{p["name"]}</td><td>{p["mode"]}</td>'
                f'<td>{"—" if s is None else s}</td><td>{u}{mark(u, s)}</td>{ecol}</tr>')
    rows = "".join(cell(p) for p in cg["per_game"])
    feats = "・".join(cg["features"])
    ehead = '<th>単一フロー・B<br>固有エンコーダ</th>' if has_enc else ""
    erow = f'<td><b>{cg["overall_auc_enc"]}</b></td>' if has_enc else ""
    encfig = (f'<img src="figures/crossgame_enc_axis.png" alt="encoder unified axis">'
              f'<figcaption>方式B：各ゲームの<b>全特徴</b>を小さな線形エンコーダで共通{cg.get("enc_dim","")}次元へ写し、'
              f'同じ 1 フローに通す（全体 AUROC {cg["overall_auc_enc"]} / r={cg["overall_corr_enc"]}）。</figcaption>') if has_enc else ""
    enc = {p["key"]: p.get("auc_enc") for p in cg["per_game"]}
    enc_note = (f"<br>・<b>方式B（固有エンコーダ）</b>は各ゲームの全特徴を活かすので、"
                f"<b>チェス（{enc.get('chess')}）と Overcooked（{enc.get('overcooked')}）</b>が回復・向上。"
                f"ただし<b>ただ乗りではない</b>：プール学習は個別フローより難しく、Hanabi/SI は逆に少し落ちる。"
                f"全体はA・Bとも <b>約0.635</b>＝<b>与える情報量に依らず共通軸は同じ水準</b>に着地する（＝軸の意味がゲームを跨いで安定）。") if has_enc else ""
    return f"""
<section><h2>横断：単一フローで技量軸の意味を共通化</h2>
<p>各ゲーム別々のフローだと、チェスの z₁ と Hanabi の z₁ は別モデルの別座標で、直接は比べられない。そこで
<b>ゲームIDを条件（context）にした 1 つの条件付きフロー</b>を学習し、技量は<b>ゲーム内で標準化</b>してプール、
<code>z₁ ≈ 自分のゲーム内での相対技量</code>を全ゲームに課す。→ z₁ が<b>1 本の共通座標</b>になり、
チェスの z₁＝+1 と Hanabi の z₁＝+1 が「どちらも自分のゲームの平均より約1σ上手」という<b>同じ意味</b>になる。
入力の作り方を 2 通り比べた：<b>A</b>＝全ゲーム共通のスタイル記述子 <code>{feats}</code>（同じ計算・同じ意味）／
<b>B</b>＝各ゲームの<b>固有特徴を小エンコーダで共通空間へ</b>（固有情報を捨てない）。</p>
<img src="figures/crossgame_axis.png" alt="cross-game unified axis">
<figcaption>方式A：全 {cg["n_games"]} ゲーム{cg["n_total"]:,}人ぶんを 1 つの潜在へ（色＝ゲーム）。右：共通 z₁ vs ゲーム内で標準化した技量（全体 r={cg["overall_corr"]}）。</figcaption>
{encfig}
<table><tr><th>ゲーム</th><th>種別</th><th>個別フロー</th><th>単一フロー・A<br>汎用4特徴</th>{ehead}</tr>
{rows}
<tr><td colspan="3" style="text-align:right"><b>共通軸・全体 AUROC</b></td><td><b>{cg["overall_auc"]}</b></td>{erow}</tr></table>
<p class="interp">たった<b>4 個の汎用スタイル特徴</b>・1 本の共通 z₁ で、全体 AUROC <b>{cg["overall_auc"]}</b>。
「巧さは<b>行動の散らし方・切替・打ち筋のブレ</b>に共通して滲む」＝技量軸が<b>ゲームを跨いで意味を持つ</b>。{enc_note}
<br><b>▲</b>＝個別フロー以上。</p>
</section>"""


def build_page(results, cg=None):
    rows = "".join(
        f"<tr><td>{r['name']}</td><td>{r['mode']}</td><td>{r['n']:,}</td>"
        f"<td>{r['label']} {r['smin']}–{r['smax']}</td><td><b>{r['auc']}</b></td><td>{r['corr']}</td></tr>"
        for r in results)
    cross = _cross_section(cg)
    secs = "".join(
        f'<section><h2>{r["name"]}（{r["mode"]}）</h2>'
        f'<img src="figures/{r["key"]}_axis.png" alt="{r["key"]}">'
        f'<figcaption>左:潜在 z₁-z₂ を{r["label"]}で色分け（右＝高{r["label"]}＝上手）。右:z₁(推定技量) vs 実{r["label"]}。</figcaption>'
        f'<p class="interp"><b>技量ラベル</b>＝{r["label"]}（{r["smin"]}〜{r["smax"]}）。'
        f'プレイスタイルだけから <b>初心者vs熟練 AUROC {r["auc"]}</b>・z₁と{r["label"]}相関 {r["corr"]}。'
        f'{"（教師つき軸は後付け方向 "+str(r["auc_posthoc"])+" を上回る）" if "auc_posthoc" in r else ""}</p></section>'
        for r in results)
    best = max(r["auc"] for r in results)
    n_modes = len({r["mode"] for r in results})
    cross_kpi = (f'<div class="kpi"><b>{cg["overall_auc"]}</b>共通軸・全体 AUROC'
                 f'<br><span class="sub">単一フロー×{cg["n_games"]}ゲーム</span></div>') if cg else ""
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Games-NF — プレイから技量軸を学ぶ</title><style>{PAGE_CSS}</style></head><body>
<div class="crumb"><a href="https://kankusakabe.github.io/KAN-NF/">&larr; KAN-NF 統括へ</a></div>
<h1>Games-NF — ゲームの実プレイから「技量軸」を学ぶ</h1>
<p class="sub">条件付き Normalizing Flow ＋ スコア/レートで、潜在に <b>右＝上手・左＝下手</b> の軸を通す。KAN-NF 実験群の一部。</p>
<div class="kpis">
 <div class="kpi"><b>{len(results)}</b>ゲーム</div>
 <div class="kpi"><b>{n_modes}</b>種の関わり方<br><span class="sub">1人用・対戦・協力</span></div>
 <div class="kpi"><b>{best}</b>最良 初心者vs熟練 AUROC</div>
 {cross_kpi}
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
{cross}
{secs}
<section><h2>正直な限界・次の一歩</h2><p class="interp">
・<b>3種の関わり方すべて</b>を同じ機構で通した：1人用（スコア）・対戦（レート/ランク）・協力（チーム点）。技量ラベルの正体はゲームごとに違うが、軸の作り方は共通。<br>
・技量軸は<b>ソフト</b>で、ゲームにより効き方が大きく違う。<b>Hanabi(0.83)とMs.パックマン(0.80)が最も鋭く</b>、<b>囲碁(0.55)が最も弱い</b>（盤の粗い配石＝線の位置分布だけでは段級位を読みにくい。定石・形・状態条件つき特徴が要る）。同じ協力でも Hanabi は鋭く、Overcooked は中程度（実人間trialが n=172 と小さく荒い）。<br>
・<b>スコア漏れに注意した</b>：Hanabi は「成功プレイ数＝点数」なので play率をそのまま特徴にすると点数がそのまま漏れる。あえて play/捨て札の量を捨て、プレイ<b>以外</b>の行動（色ヒント/数字ヒントの構成比）だけで軸を作った。それでも AUROC0.83 ＝<b>ヒント主体の協調スタイル</b>そのものが強さと結びつく、という中身のある結果。Overcooked はレイアウトごとに点の桁が違うので<b>レイアウト内の順位（％）</b>を技量ラベルにした。<br>
・<b>軸の共通化は達成</b>（上の横断セクション）：ゲームIDを条件にした単一フローで、z₁ が全ゲーム共通の相対技量座標に。入力を汎用4特徴（A）にしても、各ゲーム固有特徴を小エンコーダで共通空間へ写して（B）も、全体は約0.635で同水準＝<b>軸の意味は与える情報量に依らず安定</b>。固有エンコーダはチェス・Overcookedを回復させる一方、プール学習が難しくHanabi/SIは少し落ちる（ただ乗りではない）。<br>
・<b>Phase 2（次）</b>：この共通技量軸で「ユーザのプレイをスコアリング」「初心者に一歩上のお手本や補助」を出すミニ機能。
</p></section>
<p class="sub"><code>python -m gamesnf.games</code> で自動生成。KAN-NF 実験群。</p>
</body></html>"""
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write(html)
    import json as _json
    with open(os.path.join(DOCS, "results.json"), "w") as f:  # rebuild page w/o retraining
        _json.dump(results, f, ensure_ascii=False, indent=1)
    if cg:
        with open(os.path.join(DOCS, "crossgame.json"), "w") as f:
            _json.dump(cg, f, ensure_ascii=False, indent=1)
    print("wrote combined page with", [r["key"] for r in results], "cross=" + str(bool(cg)))


def _add(results, key, name, mode, loader, label, min_n=100):
    """Run one game's loader + skill axis, appending to results (skip on failure)."""
    try:
        X, S = loader()
        if X is None or len(X) < min_n:
            print(f"{name} skipped: n={0 if X is None else len(X)} (<{min_n})"); return
        results.append(skill_axis(key, name, mode, X, S, label=label))
    except Exception as e:
        print(f"{name} skipped:", repr(e))


def main():
    import json as _json
    if len(sys.argv) > 1 and sys.argv[1] == "page":  # rebuild HTML from cache, no retrain
        with open(os.path.join(DOCS, "results.json")) as f:
            results = _json.load(f)
        cgp = os.path.join(DOCS, "crossgame.json")
        cg = _json.load(open(cgp)) if os.path.exists(cgp) else None
        build_page(results, cg)
        return
    results = []
    for i, (key, name, mode, acts) in enumerate(ATARI):  # 1p, score = skill
        X, S = load_atari(key, acts)
        results.append(skill_axis(key, name, mode, X, S, label="スコア", also_posthoc=(i == 0)))
    _add(results, "chess", "チェス", "対戦", load_chess, "レート", min_n=200)      # versus
    _add(results, "go", "囲碁", "対戦", load_go, "ランク", min_n=200)             # versus
    _add(results, "overcooked", "Overcooked", "協力", load_overcooked, "層内順位%")  # co-op
    _add(results, "hanabi", "Hanabi(2人)", "協力", load_hanabi, "花火点")          # co-op
    cg = None
    try:
        import crossgame
        cg = crossgame.run_all(results)  # ONE flow, game-ID context, shared z1 (A: 4 feats, B: encoders)
    except Exception as e:
        print("crossgame skipped:", repr(e))
    build_page(results, cg)


if __name__ == "__main__":
    main()
