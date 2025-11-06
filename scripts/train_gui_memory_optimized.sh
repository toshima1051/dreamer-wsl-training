#!/usr/bin/env bash
set -euo pipefail

# GUI描画で動画を表示しつつ、メモリ使用量を最小化
# train_cartpole_gpu.shと同じ確実な方法を使用

LOGDIR_BASE="${1:-$HOME/logdir/dreamer}"
TASK="${2:-dmc_cartpole_swingup}"
EXISTING_LOGDIR="${3:-}"  # 第3引数で既存のログディレクトリを指定可能

# 既存のログディレクトリが指定されている場合はそれを使用、そうでなければ新しいディレクトリを作成
if [ -n "${EXISTING_LOGDIR}" ] && [ -d "${EXISTING_LOGDIR}" ]; then
    LOGDIR="${EXISTING_LOGDIR}"
    echo "Resuming training from existing logdir: ${LOGDIR}"
else
    TS="$(date +%Y%m%d-%H%M%S)"
    LOGDIR="${LOGDIR_BASE}/${TS}"
    echo "Starting new training run: ${LOGDIR}"
fi

# DISPLAY環境変数を設定（X11転送用）
if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=:0
fi

source "/home/ruku/source/dreamer/.venv311/bin/activate"

# ROCmを無効化してCUDAを強制
# これらの環境変数はPythonスクリプト内でも設定されるが、
# シェルレベルでも設定して確実にROCmを無効化
export HIP_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=cuda
export MUJOCO_GL=glfw  # GUI描画を使用（動画が正しく表示される）
# ROCmを完全に無効化する追加設定
export ROCM_PATH=""
export HSA_OVERRIDE_GFX_VERSION=""

cd /home/ruku/source/dreamer/external/dreamerv3

echo "Starting training with GUI rendering (normal config, memory optimized)..."
echo "  Logdir: ${LOGDIR}"
echo "  Task: ${TASK}"
echo "  DISPLAY: ${DISPLAY}"
echo "  MUJOCO_GL: ${MUJOCO_GL}"
echo "  Config: Normal (full model size, reduced batch/envs for memory)"
if [ -n "${EXISTING_LOGDIR}" ] && [ -d "${EXISTING_LOGDIR}" ]; then
    echo "  Mode: Resuming from checkpoint"
else
    echo "  Mode: New training run"
fi
echo ""

# CUDAを強制（ROCmエラーを回避）
# 環境変数をPythonスクリプトの最初で設定（JAXインポート前に必須）
python -c "
import os
import sys

# 最初に環境変数を設定（JAXがインポートされる前に必須）
# setdefaultではなく明示的に設定して確実にROCmを無効化
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['JAX_PLATFORM_NAME'] = 'cuda'
os.environ['MUJOCO_GL'] = 'glfw'
# ROCmを完全に無効化する追加設定
os.environ['ROCM_PATH'] = ''
os.environ['HSA_OVERRIDE_GFX_VERSION'] = ''

# JAXの設定を確認（インポート前に設定されていることを確認）
print(f'JAX_PLATFORMS={os.environ.get(\"JAX_PLATFORMS\")}')
print(f'HIP_VISIBLE_DEVICES={os.environ.get(\"HIP_VISIBLE_DEVICES\")}')

# JAXをインポートして設定を確認（dreamerv3をインポートする前に）
import jax
# JAXの設定を明示的に更新
jax.config.update('jax_platforms', 'cuda')
print(f'JAX devices after config: {jax.devices()}')

# dreamerv3をインポート（この時点でJAXは既に設定済み）
from dreamerv3 import main as dv3_main

sys.argv = [
    'dreamerv3/main.py',
    '--logdir', '${LOGDIR}',
    # debug設定を削除して通常設定を使用（モデルサイズが大きくなる）
    '--jax.platform', 'gpu',
    '--task', '${TASK}',
    # メモリ不足対策：より小さな値に設定
    '--batch_size', '8',  # 12から8に削減（メモリ節約）
    '--batch_length', '32',  # 48から32に削減（メモリ節約）
    '--run.envs', '4',  # 8から4に削減（メモリ節約）
    '--run.train_ratio', '32',  # デフォルト32を維持
    '--replay.size', '200000',  # 500000から200000に削減（メモリ節約）
]
dv3_main.main()
"

