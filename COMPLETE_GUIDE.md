# DreamerV3 実装・学習プロジェクト完全ガイド

## 目次
1. [プロジェクト概要](#プロジェクト概要)
2. [環境構築の全過程](#環境構築の全過程)
3. [実装手順の詳細](#実装手順の詳細)
4. [発生した問題と解決策](#発生した問題と解決策)
5. [学習の実行と確認方法](#学習の実行と確認方法)
6. [DreamerV3とDreamer4の比較](#dreamerv3とdreamer4の比較)
7. [専門用語集](#専門用語集)
8. [参考文献一覧](#参考文献一覧)
9. [プロジェクトの構成](#プロジェクトの構成)

---

## プロジェクト概要

### 目的
- DreamerV3をWSL環境でGPU実行可能にする
- CartPole Swingupタスクで学習を実行
- 学習前後の動画を比較して性能向上を可視化

### 使用環境
- **OS**: WSL2 (Windows Subsystem for Linux)
- **GPU**: NVIDIA GPU (CUDA 12.x)
- **Python**: 3.11.9
- **主要ライブラリ**: JAX 0.4.33, NumPy 1.26.4

---

## 環境構築の全過程

### 1. 初期セットアップ

#### 1.1 GPU確認
```bash
nvidia-smi
```
- CUDAバージョンとGPUの動作を確認

#### 1.2 Python環境の準備
```bash
# Python 3.11のインストール（pyenv使用）
pyenv install 3.11.9
pyenv local 3.11.9

# 仮想環境の作成
python3 -m venv .venv311
source .venv311/bin/activate
pip install -U pip setuptools wheel
```

### 2. 依存関係のインストール

#### 2.1 JAXとNumPyの固定インストール
```bash
# JAX (CUDA12対応) とNumPyを固定バージョンでインストール
pip install "jax[cuda12_pip]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "numpy==1.26.4"
```

**重要**: JAX 0.4.33とNumPy 1.26.4の組み合わせがDreamerV3で安定動作します。

#### 2.2 danijar本家リポジトリのインストール
```bash
# 依存関係なしでインストール（衝突防止）
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/elements.git@main"
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/ninjax.git@main"
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/portal.git@main"
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/granular.git@main"
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/embodied.git@main"
```

#### 2.3 DreamerV3リポジトリのクローンと依存関係のインストール
```bash
mkdir -p external && cd external
git clone https://github.com/danijar/dreamerv3.git
cd dreamerv3
pip install -U -r requirements.txt
```

#### 2.4 追加依存関係
```bash
pip install --no-deps "optax==0.1.7" "ale-py==0.9.0"
pip install mujoco dm_control imageio-ffmpeg
```

#### 2.5 システム依存関係（MuJoCo用）
```bash
sudo apt-get update -y
sudo apt-get install -y libgl1 libglew2.2 libosmesa6 libglfw3
```

### 3. GPU動作確認
```bash
python - << 'PY'
import jax
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
print("Backend:", jax.default_backend())
PY
```

期待される出力:
```
JAX: 0.4.33
Devices: [cuda:0]
Backend: cuda
```

---

## 実装手順の詳細

### 1. 基本的な学習実行スクリプト

#### 1.1 CPUデバッグ実行（安定）
`scripts/train_cpu_debug.sh`:
```bash
#!/usr/bin/env bash
source /home/ruku/source/dreamer/.venv311/bin/activate
cd /home/ruku/source/dreamer/external/dreamerv3
JAX_PLATFORMS=cpu python -u dreamerv3/main.py \
  --logdir "$HOME/logdir/dreamer/$(date +%Y%m%d-%H%M%S)" \
  --configs debug \
  --jax.platform cpu
```

#### 1.2 GPU実行（オフスクリーン描画）
`scripts/train_cartpole_gpu.sh`:
```bash
#!/usr/bin/env bash
source /home/ruku/source/dreamer/.venv311/bin/activate
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=cuda
export HIP_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa  # オフスクリーン描画

cd /home/ruku/source/dreamer/external/dreamerv3
python -u dreamerv3/main.py \
  --logdir "$HOME/logdir/dreamer/$(date +%Y%m%d-%H%M%S)" \
  --configs debug \
  --jax.platform gpu \
  --task dmc_cartpole_swingup
```

### 2. GUI描画対応スクリプト

#### 2.1 X11転送の設定
Windows側でVcXsrvを起動:
1. VcXsrvをダウンロード・インストール
2. XLaunchを起動
3. 設定:
   - Display number: 0
   - Client startup: Start no client
   - Extra settings: ✅ Disable access control

WSL側でDISPLAY設定:
```bash
export DISPLAY=:0
```

#### 2.2 GUI描画で学習実行（最新版・推奨）
`scripts/train_gui_memory_optimized.sh`:

このスクリプトは、通常設定（フルサイズモデル）を使用しつつ、メモリ使用量を最適化した設定で学習を実行します。

**主な特徴**:
- **通常設定を使用**: debug設定ではなく、フルサイズのモデルを使用（`units: 1024`, `layers: 3`など）
- **メモリ最適化**: バッチサイズ、環境数、リプレイバッファサイズを調整してメモリ使用量を削減
- **チェックポイントからの再開**: 既存のログディレクトリを指定することで学習を再開可能
- **ROCmエラー回避**: CUDAバックエンドを強制し、ROCmの誤検出を防止

**使用方法**:

新しい学習を開始:
```bash
export DISPLAY=:0
bash /home/ruku/source/dreamer/scripts/train_gui_memory_optimized.sh
```

既存のログから再開:
```bash
bash /home/ruku/source/dreamer/scripts/train_gui_memory_optimized.sh \
  "$HOME/logdir/dreamer" \
  "dmc_cartpole_swingup" \
  "$HOME/logdir/dreamer/20251107-003058"
```

**現在の設定**（メモリ最適化済み）:
- `batch_size`: 8（デフォルト16から削減）
- `batch_length`: 32（デフォルト64から削減）
- `run.envs`: 4（デフォルト16から削減）
- `run.train_ratio`: 32（デフォルト維持）
- `replay.size`: 200000（デフォルト5,000,000から削減）

**重要なポイント**:
- 環境変数をPythonスクリプト内で設定（JAXインポート前）
- `jax.config.update('jax_platforms', 'cuda')`で明示的にCUDAを指定
- メモリ不足が発生した場合は、さらに`batch_size`や`run.envs`を削減

### 3. 可視化ツール

#### 3.1 Scopeビューアの起動
`scripts/start_scope_viewer.sh`:
```bash
#!/usr/bin/env bash
source /home/ruku/source/dreamer/.venv311/bin/activate
BASEDIR="${1:-$HOME/logdir}"
PORT="${2:-8000}"
python -m scope.viewer --basedir "${BASEDIR}" --port "${PORT}"
```

使用方法:
```bash
bash /home/ruku/source/dreamer/scripts/start_scope_viewer.sh $HOME/logdir 8000
# ブラウザで http://localhost:8000 を開く
```

#### 3.2 メトリクスの可視化
`scripts/plot_metrics.py`:
- `metrics.jsonl`から学習曲線を生成
- スコアと損失の推移をグラフ化

使用方法:
```bash
python /home/ruku/source/dreamer/scripts/plot_metrics.py \
  $HOME/logdir/dreamer/<timestamp> \
  --output learning_curve.png
```

---

## 発生した問題と解決策

### 問題1: ROCmバックエンドの誤検出

**症状**:
```
RuntimeError: Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
```

**原因**:
- JAXがROCm（AMD GPU）バックエンドを検出しようとしている
- CUDAバックエンドが正しく設定されていない

**解決策**:
1. 環境変数をPythonスクリプト内で設定（JAXインポート前）
2. `HIP_VISIBLE_DEVICES=""`でROCmデバイスを無効化
3. `jax.config.update('jax_platforms', 'cuda')`で明示的にCUDAを指定

### 問題2: メモリ不足（OOM）

**症状**:
```
Killed (プロセスが強制終了)
```

**原因**:
- GUI描画（glfw）がメモリを多く消費
- 通常設定（フルサイズモデル）を使用している場合、メモリ使用量が大きい
- バッチサイズ、環境数、リプレイバッファサイズが大きすぎる

**解決策**:
1. **バッチサイズを削減**: 12 → 8 → 6（必要に応じて）
2. **環境数を削減**: 8 → 4 → 2（必要に応じて）
3. **リプレイバッファサイズを削減**: 500000 → 200000 → 100000（必要に応じて）
4. **batch_lengthを削減**: 48 → 32 → 16（必要に応じて）
5. **debug設定に戻す**: メモリが非常に限られている場合（モデルサイズが小さくなる）

**メモリ使用量の目安**:
- 通常設定（フルサイズ）: 約2-4GB（GPUメモリ）
- debug設定: 約500MB-1GB（GPUメモリ）
- GUI描画（glfw）: 追加で約500MB-1GB（システムメモリ）

### 問題3: 動画が真っ白

**症状**:
- Scopeビューアで動画が真っ白に表示される

**原因**:
- オフスクリーン描画（osmesa/egl）がWSL環境で正しく動作しない

**解決策**:
- GUI描画（glfw）を使用
- X11転送を設定してWindows側で表示

### 問題4: メトリクスが表示されない

**症状**:
- Scopeビューアでログディレクトリを選択してもメトリクスが表示されない

**原因**:
- ログディレクトリのパスが間違っている
- メトリクスファイルの形式が正しくない

**解決策**:
- 正しいログディレクトリを選択
- `metrics.jsonl`と`scores.jsonl`が存在することを確認

---

## 学習の実行と確認方法

### 1. 学習の開始

#### GUI描画で学習（推奨）
```bash
export DISPLAY=:0
bash /home/ruku/source/dreamer/scripts/train_gui_memory_optimized.sh
```

#### オフスクリーン描画で学習（高速）
```bash
bash /home/ruku/source/dreamer/scripts/train_cartpole_gpu.sh
```

### 2. 学習の確認

#### Scopeビューアで確認
```bash
bash /home/ruku/source/dreamer/scripts/start_scope_viewer.sh $HOME/logdir 8000
```

確認項目:
- `episode/score`: エピソードスコア（高いほど良い）
- `episode/length`: エピソード長
- `train/loss/image`: 画像再構成損失（低いほど良い）
- `train/loss/policy`: 方策損失
- `epstats-policy_image.mp4`: エピソード動画

#### メトリクスファイルから確認
```bash
tail -f $HOME/logdir/dreamer/<timestamp>/metrics.jsonl
```

### 3. CartPole Swingupタスクの目標

**タスクの説明**:
- **Swingup**: ポールを下向きから上向きに振り上げる
- **Balance**: 上向きの状態を維持する（倒さない）

**スコアの意味**:
- スコアが高い = ポールを上向きにして長時間バランスを保てている
- スコア100 = エピソードの最大長（約1000ステップ）まで上向きを維持できている
- 学習が進むと、スコアが15 → 100 → 200以上と向上していく

**学習の進捗確認**:
- `episode/score`: エピソードスコア（高いほど良い）
- `train/loss/image`: 画像再構成損失（低いほど良い、通常10以下）
- `train/loss/policy`: 方策損失（低いほど良い、通常0.1以下）
- `replay/replay_ratio`: リプレイバッファの使用率（30以上が理想的）

### 4. チェックポイントからの学習再開

学習を中断した場合や、設定を変更して再開したい場合、チェックポイントから学習を再開できます。

**再開方法**:
```bash
bash /home/ruku/source/dreamer/scripts/train_gui_memory_optimized.sh \
  "$HOME/logdir/dreamer" \
  "dmc_cartpole_swingup" \
  "$HOME/logdir/dreamer/YYYYMMDD-HHMMSS"
```

**注意事項**:
- チェックポイントは`<logdir>/ckpt/`ディレクトリに保存されます
- 設定を変更した場合（`batch_size`、`run.envs`など）、チェックポイントと互換性がない可能性があります
- その場合は、新しいログディレクトリで学習を開始するか、設定を元の値に戻してください

---

## DreamerV3とDreamer4の比較

### DreamerV3の特徴

#### アーキテクチャ
- **世界モデル**: RSSM（Recurrent State Space Model）を使用
- **表現**: カテゴリカル表現を使用
- **予測**: 将来の表現と報酬を予測
- **方策学習**: 想像ロールアウト（imagination）を使用

#### 主な特徴
1. **固定ハイパーパラメータ**: 多様なタスクで同じ設定を使用可能
2. **スケーラビリティ**: モデルサイズを大きくすると性能とデータ効率が向上
3. **汎用性**: Atari、DeepMind Control Suite、Minecraftなど多様な環境に対応

#### 技術的詳細
- **エンコーダ**: CNNベースの画像エンコーダ
- **ダイナミクス**: RSSM（決定論的状態 + 確率的状態）
- **デコーダ**: 画像とベクトル観測を再構成
- **報酬モデル**: 報酬を予測
- **方策**: Actor-Criticアーキテクチャ
- **価値関数**: 価値を予測

### Dreamer4について

**発表日**: 2025年9月29日  
**論文**: [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527)  
**プロジェクトサイト**: [danijar.com/project/dreamer4/](https://danijar.com/project/dreamer4/)

#### 主な革新点

1. **正確なオブジェクト相互作用の予測**
   - 複雑な環境でのオブジェクト相互作用を正確に予測
   - 以前の世界モデル（Lucid、Oasis）を大幅に上回る性能

2. **リアルタイムインタラクティブ推論**
   - 単一GPUでリアルタイム推論が可能
   - **Shortcut forcing objective**: 新しい目的関数により高速化
   - **効率的なTransformerアーキテクチャ**: 計算効率を向上

3. **オフライン学習の実現**
   - 環境との相互作用なしで学習可能
   - 少量のデータから一般的なアクション条件付けを学習
   - 多様なラベルなし動画から知識を抽出

4. **Minecraftでの画期的な成果**
   - **初の達成**: オフラインデータのみでMinecraftでダイヤモンドを取得
   - 20,000以上のマウス・キーボードアクションのシーケンスを選択
   - OpenAIのVPTオフラインエージェントを上回り、100倍少ないデータで学習

5. **想像学習（Imagination Training）**
   - 世界モデル内で強化学習により行動を学習
   - 多様なMinecraftシナリオの分布を学習
   - 複雑で長期的なタスクを想像空間で学習可能

#### 技術的詳細

- **アーキテクチャ**: 効率的なTransformerベースの世界モデル
- **学習方法**: Shortcut forcing objectiveによる高速化
- **データ効率**: 少量のラベル付きデータ + 大量のラベルなし動画
- **推論速度**: 単一GPUでリアルタイムインタラクティブ推論
- **応用**: ロボティクスなど、オンライン相互作用が困難な分野に対応

#### DreamerV3とDreamer4の比較表

| 項目 | DreamerV3 | Dreamer4 |
|------|-----------|----------|
| **世界モデル** | RSSM（Recurrent State Space Model） | Transformerベース（効率的） |
| **表現** | カテゴリカル表現 | カテゴリカル表現（改良） |
| **オブジェクト相互作用** | 限定的 | 正確な予測が可能 |
| **推論速度** | 高速 | リアルタイムインタラクティブ（単一GPU） |
| **オフライン学習** | 部分的対応 | 完全対応（環境相互作用なし） |
| **データ効率** | 高い | 非常に高い（100倍少ないデータ） |
| **複雑なタスク** | 対応可能 | より複雑なタスクに対応（Minecraftダイヤモンド取得） |
| **ハイパーパラメータ** | 固定 | 固定 |
| **スケーラビリティ** | 中程度（軽量だが限定的） | 非常に高い（多様なデータセット・環境に対応） |
| **主な応用** | Atari、DMC、Minecraft | Minecraft、ロボティクス |

#### 主な違いの詳細

### 1. アーキテクチャの根本的な違い

#### DreamerV3のアーキテクチャ
- **世界モデル**: RSSM（Recurrent State Space Model）
  - 決定論的状態（deterministic state）と確率的状態（stochastic state）の組み合わせ
  - CNNベースのエンコーダとデコーダ
  - リカレントニューラルネットワーク（RNN）ベースのダイナミクスモデル
  - **特徴**: より軽量だが、Dreamer4と比較するとスケーラビリティが低い（RNNベースの制約）
  - **目的関数**: 変分目的関数（variational objective）を使用
  - **注意**: DreamerV3論文ではスケーラビリティが高いと評価されているが、これはDreamer4と比較した場合の相対的な評価
  - **参考文献**: 
    - [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 3.1 "World Model"
    - [Danijar HafnerのX投稿](https://x.com/danijarh) (2025年10月1日): "Dreamer 3, which is based on a more lightweight but less scalable RNN with variational objective"
- **表現**: カテゴリカル表現（categorical representations）
  - 離散的な潜在変数を使用
  - シンボリックな表現を学習
  - **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 3.2 "Categorical Representations"

#### Dreamer4のアーキテクチャ
- **世界モデル**: Transformerベースの効率的なアーキテクチャ
  - より長期的な依存関係を捉える能力
  - 効率的な注意機構により計算コストを削減
  - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract, Section 3 "Architecture"
- **表現**: カテゴリカル表現（改良版）
  - DreamerV3の表現を継承しつつ、より高精度な予測が可能
  - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Section 3
- **Shortcut Forcing Objective**: 新しい目的関数
  - リアルタイム推論を可能にする重要な革新
  - 中間表現を直接学習することで推論速度を向上
  - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "through a shortcut forcing objective and an efficient transformer architecture"
- **DreamerV3との比較**: 
  - DreamerV3はより軽量だがスケーラビリティが低いRNNベースで変分目的関数を使用
  - Dreamer4はより多様なデータセットと環境にスケール可能
  - 軽量なアプローチは簡単なタスクには適しているが、Dreamer4はより複雑なタスクにスケール可能
  - **参考文献**: 
    - [Danijar HafnerのX投稿](https://x.com/danijarh) (2025年10月1日): "We have come a long way since Dreamer 3, which is based on a more lightweight but less scalable RNN with variational objective"
    - [Danijar HafnerのX投稿](https://x.com/danijarh) (2025年10月1日): "While the lightweight approach still makes sense for easier tasks, Dreamer 4 allows scaling to much more diverse datasets and environments 🚀"

### 2. オブジェクト相互作用の予測精度

#### DreamerV3の限界
- **基本的な予測**: 単純な環境での予測は可能
- **複雑な相互作用**: 複数のオブジェクトが関与する相互作用の予測が困難
- **例**: Minecraftでの複雑なクラフトや、オブジェクト間の物理的相互作用
- **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "However, previous world models have been unable to accurately predict object interactions in complex environments."

#### Dreamer4の革新
- **正確な予測**: 複雑な環境でのオブジェクト相互作用を正確に予測
- **実証例**: 
  - Minecraftでのダイヤモンド採掘（20,000以上のアクションシーケンス）
  - 複雑なクラフトタスク
  - 物理的なオブジェクト相互作用
- **以前の世界モデルとの比較**: Lucid、Oasisなどの既存の世界モデルを大幅に上回る性能
- **参考文献**: 
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "In the complex video game Minecraft, the world model accurately predicts object interactions and game mechanics, outperforming previous world models by a large margin."
  - [Dreamer4プロジェクトサイト](https://danijar.com/project/dreamer4/): "Human Interaction"セクションでLucid、Oasisとの比較デモを公開

### 3. 推論速度とリアルタイム性能

#### DreamerV3
- **推論速度**: 高速だが、リアルタイムインタラクティブには限界
- **用途**: 主に学習時のシミュレーションに使用
- **制約**: リアルタイムでの人間との相互作用には不向き
- **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) - リアルタイム推論に関する明示的な言及なし

#### Dreamer4
- **リアルタイム推論**: 単一GPUでリアルタイムインタラクティブ推論が可能
- **Shortcut Forcing Objective**: 
  - 中間表現を直接学習することで推論パスを短縮
  - 計算コストを大幅に削減
- **実証**: プロジェクトサイトで人間が世界モデル内で様々なタスクを実行するデモを公開
- **応用**: リアルタイムでの対話的なシミュレーションが可能
- **参考文献**: 
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "The world model achieves real-time interactive inference on a single GPU through a shortcut forcing objective and an efficient transformer architecture."
  - [Dreamer4プロジェクトサイト](https://danijar.com/project/dreamer4/): "Human Interaction"セクションでリアルタイムインタラクティブ推論のデモを公開

### 4. 学習方法の違い

#### DreamerV3の学習方法
- **オンライン学習**: 環境との相互作用が必要
  - エージェントが環境で行動し、その経験から学習
  - 環境との相互作用が安全で高速な場合に適している
  - **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 2 "Method" - 環境との相互作用から学習
- **想像学習（Imagination Training）**: 
  - 世界モデル内で想像ロールアウトを実行
  - しかし、世界モデル自体は環境との相互作用から学習
  - **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 2.3 "Actor-Critic Learning"

#### Dreamer4の学習方法
- **完全オフライン学習**: 環境との相互作用なしで学習可能
  - **少量のラベル付きデータ**: アクション条件付けを学習
  - **大量のラベルなし動画**: 多様な動画から知識を抽出
  - **想像学習（Imagination Training）**: 
    - 世界モデル内で強化学習により行動を学習
    - 環境との相互作用なしで複雑なタスクを学習可能
    - **想像学習の利点**: 方策をより堅牢で効率的にする。ダイヤモンドへのマイルストーンをより早く達成
- **実証**: Minecraftでダイヤモンドを取得（初の達成）
- **参考文献**: 
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "Moreover, the world model learns general action conditioning from only a small amount of data, allowing it to extract the majority of its knowledge from diverse unlabeled videos."
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "By learning behaviors in imagination, Dreamer 4 is the first agent to obtain diamonds in Minecraft purely from offline data, without environment interaction."
  - [Dreamer4プロジェクトサイト](https://danijar.com/project/dreamer4/): "Diamonds from Offline Experience"セクション
  - [Danijar HafnerのX投稿](https://x.com/danijarh) (2025年10月1日): "We find that imagination training not only makes policies more robust but also more efficient, so they achieve milestones towards the diamond faster"

### 5. データ効率の比較

#### DreamerV3
- **データ効率**: 高い
  - 固定ハイパーパラメータで多様なタスクに対応
  - モデルサイズを大きくするとデータ効率が向上
- **データ要件**: 環境との相互作用から十分な経験を収集する必要がある
- **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 4 "Results" - スケーラビリティとデータ効率について

#### Dreamer4
- **データ効率**: 非常に高い（100倍少ないデータ）
- **実証**: OpenAIのVPTオフラインエージェントを上回り、100倍少ないデータで学習
- **データソース**: 
  - 少量のラベル付きデータ（アクション条件付け）
  - 大量のラベルなし動画（多様なシナリオ）
- **利点**: 実世界の応用（ロボティクスなど）で、安全で効率的な学習が可能
- **参考文献**: 
  - [Dreamer4プロジェクトサイト](https://danijar.com/project/dreamer4/): "Dreamer 4 significantly outperforms OpenAI's VPT offline agent, while using 100 times less data."
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "Moreover, the world model learns general action conditioning from only a small amount of data"

### 6. 複雑なタスクへの対応

#### DreamerV3の対応範囲
- **多様なタスク**: Atari、DeepMind Control Suite、Minecraftなど
- **タスクの複雑さ**: 中程度の複雑さまで対応可能
- **長期的タスク**: 限定的な対応
- **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 4 "Results" - 多様なタスクでの実験結果

#### Dreamer4の対応範囲
- **より複雑なタスク**: Minecraftでのダイヤモンド取得など
- **長期的タスク**: 20,000以上のアクションシーケンスを選択
- **多様なシナリオ**: 世界モデルが多様なMinecraftシナリオの分布を学習
- **実証**: 
  - 木を集める
  - 石を採掘する
  - ダイヤモンドを取得する
  - 複雑なクラフトタスク
- **参考文献**: 
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "This task requires choosing sequences of over 20,000 mouse and keyboard actions from raw pixels."
  - [Dreamer4プロジェクトサイト](https://danijar.com/project/dreamer4/): "Imagination Training"セクションで様々なタスクのデモを公開

### 7. 実世界への応用

#### DreamerV3
- **主な応用**: シミュレーション環境（Atari、DMC、Minecraft）
- **実世界応用**: 限定的（環境との相互作用が必要なため）
- **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) - 主にシミュレーション環境での実験

#### Dreamer4
- **実世界応用**: ロボティクスデータセットでテスト
- **物理的相互作用**: 物理的なオブジェクト相互作用をシミュレート可能
- **オフライン学習**: オンライン相互作用が困難な実世界の応用に対応
- **安全性**: 危険な環境での学習が可能（環境との相互作用なし）
- **参考文献**: 
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "aligning with practical applications such as robotics where learning from environment interaction can be unsafe and slow."
  - [Dreamer4プロジェクトサイト](https://danijar.com/project/dreamer4/): "Real World Video"セクションでロボティクスデータセットでのテスト結果を公開

### 8. 技術的な革新点の詳細

#### DreamerV3の技術的特徴
- **RSSM**: 再帰的な状態空間モデル
- **カテゴリカル表現**: 離散的な潜在変数
- **固定ハイパーパラメータ**: 多様なタスクで同じ設定を使用
- **スケーラビリティ**: モデルサイズを大きくすると性能が向上（ただし、Dreamer4と比較するとRNNベースの制約によりスケーラビリティが限定的）
- **参考文献**: 
  - [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 3 "Method"
  - [Danijar HafnerのX投稿](https://x.com/danijarh) (2025年10月1日): "Dreamer 3, which is based on a more lightweight but less scalable RNN"

#### Dreamer4の技術的革新
- **Transformerアーキテクチャ**: より長期的な依存関係を捉える
- **Shortcut Forcing Objective**: リアルタイム推論を可能にする
- **効率的な注意機構**: 計算コストを削減
- **オフライン学習**: 環境との相互作用なしで学習
- **多様なデータソース**: ラベル付きデータとラベルなし動画の組み合わせ
- **世界モデル表現の優位性**: 
  - 世界モデルの表現を行動クローニング（behavioral cloning）に使用すると、Gemma 3の一般的な表現よりも優れた性能を示す
  - 世界モデルが環境の深い理解を学習し、意思決定に有用な形式で表現していることを示す
- **参考文献**: 
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "through a shortcut forcing objective and an efficient transformer architecture"
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "the world model learns general action conditioning from only a small amount of data, allowing it to extract the majority of its knowledge from diverse unlabeled videos"
  - [Danijar HafnerのX投稿](https://x.com/danijarh) (2025年10月1日): "Moreover, using the WM representations for behavioral cloning outperforms using the general representations of Gemma 3"

### 9. 性能比較の具体例

#### Minecraftタスクでの比較
- **DreamerV3**: Minecraftで基本的なタスクに対応可能
  - **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) - Minecraftでの実験結果
- **Dreamer4**: 
  - オフラインデータのみでダイヤモンドを取得（初の達成）
  - VPTオフラインエージェントを上回る性能
  - 100倍少ないデータで学習
  - 20,000以上のアクションシーケンスを選択
  - **参考文献**: 
    - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "By learning behaviors in imagination, Dreamer 4 is the first agent to obtain diamonds in Minecraft purely from offline data, without environment interaction."
    - [Dreamer4プロジェクトサイト](https://danijar.com/project/dreamer4/): "Diamonds from Offline Experience"セクション

#### 世界モデルの精度比較
- **DreamerV3**: 基本的な世界モデルで多様なタスクに対応
  - **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 4 "Results"
- **Dreamer4**: 
  - 以前の世界モデル（Lucid、Oasis）を大幅に上回る性能
  - 複雑なオブジェクト相互作用を正確に予測
  - リアルタイムインタラクティブ推論が可能
  - **参考文献**: 
    - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "outperforming previous world models by a large margin"
    - [Dreamer4プロジェクトサイト](https://danijar.com/project/dreamer4/): "Human Interaction"セクションでLucid、Oasisとの比較デモ

### 10. 実用性の違い

#### DreamerV3の実用性
- **強み**: 
  - 多様なタスクで固定ハイパーパラメータを使用可能
  - モデルサイズを大きくすると性能が向上（ただし、Dreamer4と比較するとRNNベースの制約により限定的）
  - オンライン学習が可能な環境で効果的
  - 軽量で簡単なタスクに適している
- **制約**: 
  - 環境との相互作用が必要
  - リアルタイムインタラクティブには限界
  - Dreamer4と比較するとスケーラビリティが限定的（RNNベースの制約）
- **参考文献**: 
  - [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 1 "Introduction" - 固定ハイパーパラメータとスケーラビリティについて
  - [Danijar HafnerのX投稿](https://x.com/danijarh) (2025年10月1日): "While the lightweight approach still makes sense for easier tasks"

#### Dreamer4の実用性
- **強み**: 
  - 完全オフライン学習が可能
  - リアルタイムインタラクティブ推論が可能
  - より複雑なタスクに対応
  - 実世界の応用（ロボティクス）に対応
- **利点**: 
  - 安全で効率的な学習が可能
  - 環境との相互作用が困難な場合でも学習可能
  - より少ないデータで学習可能
- **参考文献**: 
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "aligning with practical applications such as robotics where learning from environment interaction can be unsafe and slow."
  - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "Our work provides a scalable recipe for imagination training, marking a step towards intelligent agents."

### 11. まとめ：DreamerV3からDreamer4への進化

Dreamer4は、DreamerV3の優れた特徴（固定ハイパーパラメータ、汎用性）を継承しつつ、スケーラビリティを大幅に向上させ、以下の重要な革新を実現しました：

1. **アーキテクチャの進化**: RSSMからTransformerベースへ
   - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract, Section 3

2. **予測精度の向上**: 複雑なオブジェクト相互作用の正確な予測
   - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "outperforming previous world models by a large margin"

3. **推論速度の向上**: リアルタイムインタラクティブ推論の実現
   - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "achieves real-time interactive inference on a single GPU"

4. **学習方法の革新**: 完全オフライン学習の実現
   - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "purely from offline data, without environment interaction"

5. **データ効率の向上**: 100倍少ないデータで学習可能
   - **参考文献**: [Dreamer4プロジェクトサイト](https://danijar.com/project/dreamer4/): "using 100 times less data"

6. **複雑なタスクへの対応**: より長期的で複雑なタスクに対応
   - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "sequences of over 20,000 mouse and keyboard actions"

7. **実世界への応用**: ロボティクスなど実世界の応用に対応
   - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "aligning with practical applications such as robotics"

8. **スケーラビリティの向上**: DreamerV3のRNNベースの制約を克服し、より多様なデータセットと環境にスケール可能
   - **参考文献**: [Danijar HafnerのX投稿](https://x.com/danijarh) (2025年10月1日): "Dreamer 4 allows scaling to much more diverse datasets and environments 🚀"

これらの革新により、Dreamer4は実世界の応用（特にロボティクスなど、オンライン相互作用が困難な分野）において重要な進歩を実現しています。

**参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract: "Our work provides a scalable recipe for imagination training, marking a step towards intelligent agents."

---

## 専門用語集

### 強化学習関連

- **エピソード（Episode）**: エージェントが環境と相互作用する一連のステップ。通常、終了条件（タスク完了、失敗など）に達するまで続く
  - **参考文献**: [Sutton & Barto (2018) Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) Chapter 3

- **リプレイバッファ（Replay Buffer）**: 過去の経験（観測、行動、報酬）を保存するメモリ。学習時にランダムにサンプリングして使用する
  - **サイズ**: 大きいほど多様な経験を保持できるが、メモリ使用量が増える
  - **参考文献**: [Mnih et al. (2015) Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

- **バッチサイズ（Batch Size）**: 一度に処理するサンプル数。大きいほど学習が安定するが、メモリ使用量が増える
  - **参考文献**: [Goodfellow et al. (2016) Deep Learning](https://www.deeplearningbook.org/) Chapter 8

- **バッチ長（Batch Length）**: 時系列データの長さ。長いほど長期的な依存関係を学習できるが、メモリ使用量が増える
  - **参考文献**: [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 3

### DreamerV3関連

- **RSSM（Recurrent State Space Model）**: 再帰的な状態空間モデル。決定論的状態と確率的状態の組み合わせで環境のダイナミクスをモデル化
  - **決定論的状態（Deterministic State）**: 過去の情報を集約した状態
  - **確率的状態（Stochastic State）**: 不確実性を表現する状態
  - **参考文献**: 
    - [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 3.1
    - [Hafner et al. (2019) Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)

- **カテゴリカル表現（Categorical Representations）**: 離散的な潜在変数を使用した表現。連続値ではなく、離散的なカテゴリで状態を表現
  - **利点**: シンボリックな表現を学習でき、解釈しやすい
  - **参考文献**: 
    - [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 3.2
    - [Van den Oord et al. (2017) Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

- **想像ロールアウト（Imagination Rollout）**: 世界モデル内で将来の状態を予測し、その予測された軌道で方策を学習する手法
  - **利点**: 実際の環境との相互作用なしで学習できるため、データ効率が高い
  - **参考文献**: 
    - [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 2.3
    - [Hafner et al. (2020) Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)

- **Actor-Critic**: 方策（Actor）と価値関数（Critic）を同時に学習する強化学習アルゴリズム
  - **Actor**: 行動を選択する方策
  - **Critic**: 状態や行動の価値を評価する関数
  - **参考文献**: 
    - [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html) Chapter 13
    - [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 2.3

- **変分目的関数（Variational Objective）**: 変分推論に基づく目的関数。潜在変数の分布を学習するために使用
  - **参考文献**: 
    - [Kingma & Welling (2014) Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
    - [DreamerV3論文](https://arxiv.org/pdf/2301.04104) Section 3

### Dreamer4関連

- **Transformer**: 注意機構（Attention Mechanism）を使用したニューラルネットワークアーキテクチャ。長期的な依存関係を捉える能力が高い
  - **参考文献**: 
    - [Vaswani et al. (2017) Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Section 3

- **Shortcut Forcing Objective**: 中間表現を直接学習することで推論パスを短縮し、リアルタイム推論を可能にする目的関数
  - **参考文献**: [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract

- **オフライン学習（Offline Learning）**: 環境との相互作用なしで、事前に収集されたデータから学習する手法
  - **利点**: 安全で効率的な学習が可能。実世界の応用（ロボティクスなど）に適している
  - **参考文献**: 
    - [Levine et al. (2020) Offline Reinforcement Learning: Tutorial, Review, and Perspectives](https://arxiv.org/abs/2005.01643)
    - [Dreamer4論文](https://arxiv.org/abs/2509.24527) Abstract

### 技術用語

- **JAX**: Googleが開発した機械学習フレームワーク。NumPyライクなAPIを持ち、自動微分とGPU/TPU対応が特徴
  - **参考文献**: [JAX公式ドキュメント](https://jax.readthedocs.io/)

- **CUDA**: NVIDIAが開発したGPU向けの並列計算プラットフォーム
  - **参考文献**: [CUDA公式ドキュメント](https://docs.nvidia.com/cuda/)

- **ROCm**: AMDが開発したGPU向けの並列計算プラットフォーム。CUDAの代替
  - **問題**: JAXがROCmを誤検出することがあるため、CUDAを強制する必要がある
  - **参考文献**: [ROCm公式ドキュメント](https://rocm.docs.amd.com/)

- **X11転送**: LinuxのX Window Systemをリモートで使用する技術。WSLからWindows側のXサーバーにGUIを表示できる
  - **参考文献**: [X11公式ドキュメント](https://www.x.org/)

- **MuJoCo**: 物理シミュレーションエンジン。ロボティクスや強化学習の研究で広く使用される
  - **参考文献**: [MuJoCo公式ドキュメント](https://mujoco.readthedocs.io/)

---

## 参考文献一覧

### DreamerV3の参考文献

#### 論文
- **Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models.**
  - **URL**: https://arxiv.org/pdf/2301.04104
  - **主要セクション**:
    - Section 1 "Introduction": 固定ハイパーパラメータとスケーラビリティについて
    - Section 2 "Method": 学習方法（環境との相互作用から学習）
    - Section 2.3 "Actor-Critic Learning": 想像学習（Imagination Training）
    - Section 3 "Method": アーキテクチャの詳細
    - Section 3.1 "World Model": RSSMアーキテクチャ
    - Section 3.2 "Categorical Representations": カテゴリカル表現
    - Section 4 "Results": 実験結果とスケーラビリティ

#### 公式リソース
- **GitHubリポジトリ**: https://github.com/danijar/dreamerv3
- **プロジェクトサイト**: https://danijar.com/dreamerv3

### Dreamer4の参考文献

#### 論文
- **Hafner, D., Yan, W., & Lillicrap, T. (2025). Training Agents Inside of Scalable World Models.**
  - **URL**: https://arxiv.org/abs/2509.24527
  - **発表日**: 2025年9月29日
  - **主要セクション**:
    - **Abstract**: 
      - "However, previous world models have been unable to accurately predict object interactions in complex environments."
      - "In the complex video game Minecraft, the world model accurately predicts object interactions and game mechanics, outperforming previous world models by a large margin."
      - "The world model achieves real-time interactive inference on a single GPU through a shortcut forcing objective and an efficient transformer architecture."
      - "Moreover, the world model learns general action conditioning from only a small amount of data, allowing it to extract the majority of its knowledge from diverse unlabeled videos."
      - "This task requires choosing sequences of over 20,000 mouse and keyboard actions from raw pixels."
      - "By learning behaviors in imagination, Dreamer 4 is the first agent to obtain diamonds in Minecraft purely from offline data, without environment interaction."
      - "aligning with practical applications such as robotics where learning from environment interaction can be unsafe and slow."
      - "Our work provides a scalable recipe for imagination training, marking a step towards intelligent agents."
    - **Section 3 "Architecture"**: Transformerベースのアーキテクチャの詳細

#### プロジェクトサイト
- **URL**: https://danijar.com/project/dreamer4/
- **主要セクション**:
  - **"Diamonds from Offline Experience"**: 
    - "Dreamer 4 significantly outperforms OpenAI's VPT offline agent, while using 100 times less data."
    - Minecraftでのダイヤモンド取得の実証
  - **"Imagination Training"**: 想像学習のデモ（木を集める、石を採掘するなど）
  - **"Human Interaction"**: リアルタイムインタラクティブ推論のデモ（Lucid、Oasisとの比較）
  - **"Real World Video"**: ロボティクスデータセットでのテスト結果

#### X（旧Twitter）投稿
- **Danijar Hafner (@danijarh) のX投稿** (2025年10月1日)
  - **URL**: https://x.com/danijarh
  - **主要な投稿内容**:
    - "We have come a long way since Dreamer 3, which is based on a more lightweight but less scalable RNN with variational objective"
    - "While the lightweight approach still makes sense for easier tasks, Dreamer 4 allows scaling to much more diverse datasets and environments 🚀"
    - "We find that imagination training not only makes policies more robust but also more efficient, so they achieve milestones towards the diamond faster"
    - "Moreover, using the WM representations for behavioral cloning outperforms using the general representations of Gemma 3"

### 関連ドキュメント
- `SETUP_DREAMERV3_WSL.md`: WSL環境でのセットアップ手順
- `TRAINING_OVERVIEW_JA.md`: 学習の全体像と損失関数の説明
- `X11_SETUP.md`: X11転送の設定手順

### スクリプト一覧
- `scripts/train_cpu_debug.sh`: CPUデバッグ実行
- `scripts/train_cartpole_gpu.sh`: GPU実行（オフスクリーン描画）
- `scripts/train_gui_memory_optimized.sh`: GUI描画で学習（メモリ節約）
- `scripts/start_scope_viewer.sh`: Scopeビューアの起動
- `scripts/plot_metrics.py`: メトリクスの可視化
- `scripts/generate_comparison_videos.py`: 比較動画の生成

---

## まとめ

このプロジェクトでは、DreamerV3をWSL環境でGPU実行可能にし、CartPole Swingupタスクで学習を実行しました。主な成果:

1. **環境構築**: Python 3.11、JAX 0.4.33、NumPy 1.26.4の組み合わせで安定動作
2. **GPU実行**: ROCmバックエンドの誤検出を回避し、CUDAで正常動作
3. **GUI描画**: X11転送を設定し、動画を正しく表示
4. **可視化**: Scopeビューアとメトリクス可視化ツールで学習を監視
5. **メモリ最適化**: 通常設定（フルサイズモデル）を使用しつつ、メモリ使用量を最適化
6. **チェックポイント機能**: 学習を中断・再開できる機能を実装
7. **GitHub公開**: プロジェクトをGitHubで公開し、サンプルログと動画を含める

### 学習結果の例

実際の学習では、以下のような進捗が確認できました：
- **初期**: `episode/score: 15.03`（ランダム行動に近い）
- **学習中**: `episode/score: 106.39 → 109.55 → 214.22`（徐々に向上）
- **損失**: `train/loss/image: 223.57 → 3.87`（画像再構成が改善）
- **リプレイバッファ**: `replay/replay_ratio: 27.16 → 32.7`（十分な経験を収集）

### 今後の改善点

- **Dreamer4への移行**: Dreamer4が公開されたら、オフライン学習やより複雑なタスクへの対応を検討
- より複雑なタスク（Minecraft、Atari等）での学習
- ハイパーパラメータの最適化
- オフライン学習の実装（Dreamer4の手法を参考）
- より大きなモデルサイズでの実験（メモリに余裕がある場合）

---

**最終更新**: 2025年11月7日
**プロジェクトディレクトリ**: `/home/ruku/source/dreamer`
**GitHubリポジトリ**: https://github.com/toshima1051/dreamer-wsl-training

## プロジェクトの構成

### ディレクトリ構造
```
dreamer/
├── scripts/                    # 学習・可視化スクリプト
│   ├── train_gui_memory_optimized.sh  # GUI描画でメモリ最適化された学習（推奨）
│   ├── train_with_gui.sh      # GUI描画での学習（基本版）
│   ├── train_memory_efficient.sh  # オフスクリーン描画での学習
│   ├── plot_metrics.py         # メトリクスの可視化
│   └── render_episodes.py     # エピソードのレンダリング
├── logs/sample/               # サンプルログと動画
│   ├── metrics.jsonl          # 学習メトリクス
│   ├── scores.jsonl           # エピソードスコア
│   ├── config.yaml            # 学習設定
│   └── videos/                # エピソード動画
├── external/dreamerv3/        # DreamerV3リポジトリ（サブモジュール）
├── COMPLETE_GUIDE.md          # この完全ガイド
├── SETUP_DREAMERV3_WSL.md     # WSL環境でのセットアップ手順
├── X11_SETUP.md               # X11転送の設定方法
└── README.md                  # プロジェクト概要
```

### サンプルログと動画

プロジェクトには、実際の学習結果のサンプルが含まれています：
- `logs/sample/metrics.jsonl`: 学習メトリクスの時系列データ
- `logs/sample/scores.jsonl`: エピソードスコアと長さ
- `logs/sample/videos/epstats-policy_image/`: エピソードごとのポリシー画像動画

これらのサンプルは、学習の進捗を理解するのに役立ちます。

