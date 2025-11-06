## DreamerV3 セットアップ手順（WSL + NVIDIA GPU）

このドキュメントは、WSL 環境で DreamerV3 をGPU実行するための最小構成セットアップと、デバッグ用の短時間学習ジョブ実行までの流れをまとめたものです。

### 前提条件
- Windows 側に NVIDIA ドライバ（WSL対応）が導入済み
- `nvidia-smi` が WSL 内で動作する（CUDA バージョン 12.x 目安）
- 本プロジェクトルート: `/home/ruku/source/dreamer`
- Python 3.11〜3.12（本手順は 3.12 で確認）

### 実施内容（これまで）
1. GPU の確認（WSL 内）
   - `nvidia-smi` で GPU と CUDA バージョンを確認
2. 仮想環境の作成
   - `python3 -m venv .venv` で venv 作成し、`source .venv/bin/activate` で有効化
   - `pip/setuptools/wheel` を最新化

### これから実施すること
3. JAX / NumPy の固定インストール（DreamerV3 が安定して動く既知の組み合わせ）
   - `jax[cuda12_pip]==0.4.33`（JAX 公式 CUDA12 wheel）
   - `numpy==1.26.4`（NumPy<2 系に固定）
4. danijar 本家リポジトリ群を依存なしで導入（混在回避のため）
   - `elements`, `ninjax`, `portal`, `granular`, `embodied`, `dreamerv3`
5. 最小追加依存の導入
   - `optax==0.1.7`（JAX 0.4 系と相性の良い版）
   - `ale-py==0.9.0`（Atari を使う場合に要求される版）
6. JAX が GPU を認識できているか検証
7. DreamerV3 のデバッグ学習を短時間実行

### コマンド（プロジェクトルートで順に実行）

```bash
cd /home/ruku/source/dreamer

# 1) GPU 確認
nvidia-smi

# 2) venv 作成・有効化と基盤更新
python3 -m venv .venv
source .venv/bin/activate
python -V
pip install -U pip setuptools wheel

# 3) JAX / NumPy を固定（CUDA12 wheel を使用）
pip install "jax[cuda12_pip]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "numpy==1.26.4"

# 4) danijar 本家のみ依存なし（--no-deps）で導入（衝突防止）
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/elements.git@main"
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/ninjax.git@main"
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/portal.git@main"
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/granular.git@main"
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/embodied.git@main"
pip install --no-deps --no-cache-dir "git+https://github.com/danijar/dreamerv3.git@main"

# 5) 最小追加依存
pip install --no-deps "optax==0.1.7"
pip install "ale-py==0.9.0"

# 6) JAX が GPU を認識するか確認（Python ワンライナー）
python - << 'PY'
import jax
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
print("Backend:", jax.default_backend())
PY

# 7) デバッグ学習を短時間実行（性能ではなく動作確認目的）
python -m dreamerv3.main \
  --logdir "$HOME/logdir/dreamer/$(date +%Y%m%d-%H%M%S)" \
  --configs debug \
  --jax.platform gpu \
  --run.train_ratio 8 \
  --batch_size 8 \
  --train.batch_size 8 \
  --replay.batch 8 \
  --run.log_every 1000 \
  --run.eval_every 0 \
  --run.save_every 5000
```

### トラブルシューティング
- `ImportError: embodied ... envs` のようなエラーが出る場合:
  - 既に入っている別版 `embodied` / `dreamer`（PyPI の別パッケージ）が干渉している可能性が高いです。
  - その場合は `pip uninstall -y embodied dreamer dreamerv3` で削除し、本手順の「4) danijar 本家の導入」をやり直してください。
- NumPy が 2.x に上がってしまう衝突例があるため、`numpy==1.26.4` を最後に再固定すると安定します。

### 可視化と動画表示

#### Scopeビューアでメトリクスを確認
```bash
bash /home/ruku/source/dreamer/scripts/start_scope_viewer.sh $HOME/logdir 8000
# ブラウザで http://localhost:8000 を開く
```

#### メトリクスをグラフで可視化
```bash
source /home/ruku/source/dreamer/.venv311/bin/activate
python /home/ruku/source/dreamer/scripts/plot_metrics.py \
  $HOME/logdir/<timestamp> \
  --output learning_curve.png
```

#### GUI描画で動画を表示（X11転送が必要）
1. Windows側でXサーバー（VcXsrv等）を起動
2. X11転送をテスト:
   ```bash
   bash /home/ruku/source/dreamer/scripts/test_x11.sh
   ```
3. GUI描画で学習を実行:
   ```bash
   bash /home/ruku/source/dreamer/scripts/train_with_gui.sh
   ```

詳細は `X11_SETUP.md` を参照してください。

### 参考
- 公式実装（README・使い方）: [dreamerv3 GitHub](https://github.com/danijar/dreamerv3)


