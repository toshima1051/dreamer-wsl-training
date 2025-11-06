## Dreamer: 学習の全体像（日本語まとめ・数式入り）

このドキュメントは、あなたの実行ログ（例: `train/loss/*`、`episode/score`、`replay/replay_ratio` など）に対応する、Dreamer 系（V3 系を含む）の学習モジュールと損失関数、最適化の流れを日本語で整理したものです。環境や実装差により細部は異なり得ますが、基本構造は以下の通りです。

### 実行コマンド（WSL・最小デバッグ）

CPU 実行（安定・まずは動作確認）

```bash
source /home/ruku/source/dreamer/.venv/bin/activate
cd /home/ruku/source/dreamer/external/dreamerv3
JAX_PLATFORMS=cpu python -u dreamerv3/main.py \
  --logdir "$HOME/logdir/dreamer/$(date +%Y%m%d-%H%M%S)" \
  --configs debug \
  --jax.platform cpu
```

GPU 実行（調整中・CUDA を明示）

```bash
source /home/ruku/source/dreamer/.venv/bin/activate
cd /home/ruku/source/dreamer/external/dreamerv3
export JAX_PLATFORM_NAME=cuda
export JAX_PLATFORMS=cuda
python -u dreamerv3/main.py \
  --logdir "$HOME/logdir/dreamer/$(date +%Y%m%d-%H%M%S)" \
  --configs debug \
  --jax.platform gpu
```

ログは `~/logdir/dreamer/<timestamp>` に保存されます。

---

### 概要
- 学習するものは大きく分けて 3 つ：
  - 世界モデル（表現・ダイナミクス・デコーダ・報酬モデル）
  - 価値モデル（Critic）
  - 方策モデル（Actor）
- 世界モデルは観測履歴から潜在状態を推定・予測し、デコーダで観測を再構成、さらに報酬も予測します。
- 方策と価値は、この潜在空間上での「想像ロールアウト（imagination）」を用いて学習されます。

---

## 記法
- 観測: \(o_t\)（画像やベクトル、離散/連続トークンを含む）
- 行動: \(a_t\)
- 報酬: \(r_t\)
- 潜在状態（表現）: \(z_t\)
- ダイナミクス状態（予測に用いる潜在）: \(h_t\)（表現と同一視されることもあります）
- モデル・分布: \(p_\theta(\cdot), q_\phi(\cdot)\)
- 価値: \(V_\psi(\cdot)\)
- 方策: \(\pi_\eta(a\mid h)\)

---

## 世界モデル（Representation / Dynamics / Decoder / Reward）

### 1) 表現学習（Representation）とダイナミクス（Dynamics）
エンコーダで観測 \(o_t\) を潜在表現に写像し（`enc/cnn*`, `enc/mlp*`）、再帰的/予測的なモデルで状態遷移を学習します。一般形は以下です。

- 事前ダイナミクス（prior）: \(p_\theta(h_t \mid h_{t-1}, a_{t-1})\)
- 事後表現（posterior）: \(q_\phi(h_t \mid h_{t-1}, a_{t-1}, o_t)\)

表現とダイナミクスの整合は KL 正則化で測ります。

\[\
\mathcal{L}_{\text{rep}} = \mathrm{KL}\big(q_\phi(h_t\mid h_{t-1}, a_{t-1}, o_t)\,\Vert\, p_\theta(h_t\mid h_{t-1}, a_{t-1})\big)
\]

ログでは `train/loss/rep` や `train/loss/dyn` として出る実装もあります（しばしば等価または分割表示）。

### 2) 再構成損失（Decoder）
潜在 \(h_t\) から観測を復元します。観測のタイプごとに損失が分かれ、ログ上は次のような名前に対応します。
- `train/loss/image`: 画像再構成（例: 平均化されたピクセル単位の負対数尤度、MS-SSIM 等）
- `train/loss/vector`: 連続ベクトル観測の回帰誤差（L2/NLL）
- `train/loss/int2d`, `train/loss/float2d`, `train/loss/token`: 離散/連続トークンやマップ状観測の NLL/回帰誤差

一般に、再構成の負対数尤度を用い、合計損失は

\[\
\mathcal{L}_{\text{recon}} = -\mathbb{E}_{q_\phi}\big[\log p_\theta(o_t\mid h_t)\big]
\]

全体としての世界モデル損失は重み付き和:

\[\
\mathcal{L}_{\text{world}} = \lambda_{\text{rep}}\, \mathcal{L}_{\text{rep}} + \sum_k \lambda_{k}\, \mathcal{L}_{\text{recon},k}
\]

ここで \(k\) は各観測チャネル（image, vector, token など）。ログの `train/loss/image` が時間とともに下がるのは、再構成が改善していることを意味します。

### 3) 報酬モデル
\(h_t\) から報酬 \(r_t\) を予測します。

\[\
\mathcal{L}_{\text{rew}} = -\mathbb{E}_{q_\phi}\big[\log p_\theta(r_t\mid h_t)\big]\quad\text{（回帰なら MSE 相当）}
\]

ログでは `train/loss/rew`。

---

## 価値モデル（Critic）
潜在状態 \(h_t\) に対して将来報酬の期待値（割引和）を推定します。

\[\
V_\psi(h_t) \approx \mathbb{E}\Big[\sum_{k=0}^{\infty} \gamma^k r_{t+k}\Big]
\]

学習はターゲット（例: TD(\(\lambda\)) もしくは \(n\)-step など）との回帰で、典型的には：

\[\
\mathcal{L}_{\text{value}} = \mathbb{E}\big[\big(V_\psi(h_t) - G_t\big)^2\big]
\]

ここで \(G_t\) はモデル上の想像ロールアウトから計算されるターゲット（例: TD(\(\lambda\)) での \(\lambda\)-return）。ログでは `train/loss/value`。

---

## 方策モデル（Actor）
潜在 \(h_t\) 上で方策 \(\pi_\eta(a\mid h)\) を更新します。Dreamer では「世界モデル内での想像ロールアウト」を使って将来の価値を最大化します。

代表的な目的（符号は実装依存。ここでは最小化問題として - を付ける形で記載）:

\[\
\mathcal{L}_{\text{policy}} = -\mathbb{E}_{a_t\sim\pi_\eta,\ h_{t+1}\sim p_\theta}\big[ V_\psi(h_{t+1}) \big]
\]

エントロピー正則化（探索促進）や KL バランシングを加えることが多い：

\[\
\mathcal{L}_{\text{policy}} \leftarrow \mathcal{L}_{\text{policy}} - \alpha\,\mathbb{E}[\mathcal{H}(\pi_\eta(\cdot\mid h))] + \beta\,\mathrm{KL}\big(\pi_\eta\,\Vert\,\pi_{\text{target}}\big)
\]

ログでは `train/loss/policy` がこの項に対応します。連続/離散アクション両対応のとき、パラメータとして `pol/head/act_cont/*`, `pol/head/act_disc/*` が現れます。

---

## 全体の最適化（例）
実装によって重みやスケジューラは異なりますが、よくある合成損失は：

\[\
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{world}} + \lambda_{\text{rew}}\,\mathcal{L}_{\text{rew}} + \lambda_{\text{value}}\,\mathcal{L}_{\text{value}} + \lambda_{\text{policy}}\,\mathcal{L}_{\text{policy}}
\]

勾配は各モジュールのパラメータ（`enc/*`, `rew/*`, `val/*`, `pol/*` など）へ流れます。

---

## ログの見方（あなたのランの対応）
- `episode/score`, `episode/length`: エージェントの性能。あなたのログでは `score=100` に張り付いており、環境上限に安定到達。
- `train/loss/image`: 観測再構成（画像）誤差。実行中に 3829 → 1895 近辺まで低下＝世界モデルの復元能力が向上。
- `train/loss/rep`, `train/loss/dyn`: 表現/ダイナミクス整合（KL を含む）。\(\sim\)1 前後で安定。
- `train/loss/rew`: 報酬予測誤差。安定推移。
- `train/loss/value`: 価値回帰誤差。0.31–0.38 付近で推移。
- `train/loss/policy`: 方策目的（エントロピー・KL 付き）。小さな ± の揺れは通常挙動。
- `replay/replay_ratio`: サンプル再利用比。\(\sim\)8–9 で安定。
- `fps/policy`, `fps/train`: ポリシー更新・全体学習速度。ハード・設定による。
- 先頭のパラメータ一覧（例: `enc/cnn1/kernel`, `pol/mlp/linear0/bias` など）は各層の重み。`*norm/scale` は正規化層のスケール。角括弧の数値はシャーディング/レプリカや形状要約の表示に相当。

---

## 想像ロールアウトと TD(\(\lambda\)) の一例
世界モデルの prior を使い、潜在でロールアウト：

\[\
h_{t+1} \sim p_\theta(h_{t+1}\mid h_t, a_t),\quad a_t \sim \pi_\eta(\cdot\mid h_t),\quad r_t \sim p_\theta(r_t\mid h_t)
\]

TD(\(\lambda\)) のターゲットは（時間幅 \(H\) で打ち切る一例）：

\[\
G_t^{(\lambda)} = (1-\lambda) \sum_{n=1}^{H} \lambda^{n-1} G_t^{(n)},\quad G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\psi(h_{t+n})
\]

価値損失は \( (V_\psi(h_t) - G_t^{(\lambda)})^2 \) を平均。方策は \(V\) を最大化する方向（最小化なら負号）に更新されます。

---

## コストとプロファイリング
- ログの「Train/Report cost analysis」は FLOPS と一時メモリの概算。バッチ・シーケンス長・ロールアウト長で変化。
- `Start JAX profiler` は JAX 側のプロファイラ開始を示し、`Can't import tensorflow.python.profiler.trace` の警告は TF のフックが無いだけで通常無害です。

---

## よくある調整ポイント（ヒント）
- KL バランス（`\lambda_{\text{rep}}`）の強弱で世界モデルのシャープさ/多様性を調整。
- 再構成重みの配分（画像 vs ベクトル/トークン）
- エントロピー係数 \(\alpha\) や \(\beta\)（KL 正則）のスケジューリング
- 想像ホライズン \(H\)、割引率 \(\gamma\)、TD(\(\lambda\)) の \(\lambda\)
- リプレイ比やバッチサイズ、学習率

---

## 参考：パラメータ名の読み方（例）
- `enc/cnn1/kernel`, `enc/cnn1/bias`: 観測エンコーダの CNN 層
- `enc/mlp0/kernel`, `enc/mlp0norm/scale`: ベクトル/トークン用の MLP + 正規化
- `pol/mlp/linear0/*`, `pol/head/act_cont/*`, `pol/head/act_disc/*`: 方策 MLP とアクションヘッド（連続/離散）
- `val/mlp/*`, `val/head/logits/*`: 価値ネット
- `rew/mlp/*`, `rew/head/logits/*`: 報酬予測ネット

---

## まとめ
- 世界モデルが観測を圧縮・予測（再構成/報酬）し、その潜在空間上で価値・方策を学習します。
- ログ上の各 `train/loss/*` は、上述の各損失に対応します。
- あなたのランではスコアが上限に到達しており、世界モデルの再構成誤差も着実に低下。学習は健全に進行しています。


