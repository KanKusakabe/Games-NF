# Games-NF

ゲームの**実人間プレイ**を条件付き **Normalizing Flow** で学習し、潜在に
**「右＝上手・左＝下手」の技量軸**を通す。1人用/協力/対戦の複数ゲームを**同じ基底**で扱い、
**結果を1枚のPages**にまとめる（KAN-NF 実験群の一部）。

**▶ 結果ページ（GitHub Pages）**: https://kankusakabe.github.io/Games-NF/
**▶ 統括**: https://kankusakabe.github.io/KAN-NF/

## 基底の決定（Phase 0）
Ms.パックマン（Atari Grand Challenge・実人間）で、**プレイスタイルのみ**（行動方向分布＋切替率＋エントロピー・
生存/スコアはリークなし）から技量軸を学習。**教師つき技量軸（z₁≈スコア）AUROC 0.80 > 後付け 0.70**。
→ **基底＝ガウス＋直線の技量軸**を採用（技量は単調・順序なので円は不使用。円は移動方向用）。

## 現在の結果（1人用2種）
| ゲーム | 初心者vs熟練 AUROC | z₁×スコア相関 |
|---|---|---|
| Ms.パックマン | **0.83** | 0.50 |
| Space Invaders | 0.56 | 0.24 |

→ 粗いプレイスタイルに技量が出るゲーム（パックマン）と、出にくいゲーム（Space Invaders＝狙い/回避精度が効く）が
あるという正直な結果。次は**状態条件つき特徴**で鋭くし、協力/対戦へ拡張。

## 設計方針（全ゲーム共通）
基底＝ガウス＋直線技量軸／技量ラベル＝1人用スコア・協力チームスコア・対戦レート（各ゲーム内で標準化）／
次段：ゲームID条件の**単一フロー**で横断。**Phase 2**：技量軸でユーザのプレイをスコアリング・補助する小機能。

## 触って体験（Phase 2 アプリ）
**[▶ play.html](https://kankusakabe.github.io/Games-NF/play.html)**：スライダーで打ち方を動かすと、
**あなたに最も近い実在プレイヤー**（標準化特徴空間の kNN、k=25）を探して全6ゲームをその場で採点＋一歩上の助言。
さらに**全6ゲームとも実際に遊んで採点**できる（Ms.パックマン/Space Invaders/Overcooked＝実時間、Hanabi/チェス/囲碁＝ターン制でAI相手）。
学習データと同じ特徴を生成し、プレイ中は**ライブ・コーチ**（データ由来）＋**参考手・経路ヒント**（参考ヒューリスティック）付き。モデル推論なし・**100%実データ由来**。

**[▶ bridge.html](https://kankusakabe.github.io/Games-NF/bridge.html)**：**潜在でゲームを繋ぐ**実験（パックマン↔インベーダー）。片方を遊ぶと、
共通スタイル潜在を通じてもう片方が同じ打ち回しで**エコー自動プレイ**する（双方向）。

各ページ相互リンク：[結果](https://kankusakabe.github.io/Games-NF/) ／ [触って体験](https://kankusakabe.github.io/Games-NF/play.html) ／ [潜在で繋ぐ](https://kankusakabe.github.io/Games-NF/bridge.html)

## 再現
```bash
python -m gamesnf.games       # 各ゲームの技量軸を学習→図→1枚のdocs/index.html
python -m gamesnf.export_app  # 体験アプリ用の docs/app_data/*.json を生成（再学習なし・キャッシュz₁を再利用）
```
_条件付きNFで「プレイから技量」を測るシリーズ。実プレイ環境は作らず、学習と技量軸の可視化に集中。_
