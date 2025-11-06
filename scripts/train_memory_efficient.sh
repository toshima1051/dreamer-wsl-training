#!/usr/bin/env bash
set -euo pipefail

# メモリ節約版：オフスクリーン描画 + 小さいバッチサイズ

LOGDIR_BASE="${1:-$HOME/logdir/dreamer}"
TASK="${2:-dmc_cartpole_swingup}"

TS="$(date +%Y%m%d-%H%M%S)"
LOGDIR="${LOGDIR_BASE}/${TS}"

source /home/ruku/source/dreamer/.venv311/bin/activate

# CUDA固定（ROCm回避）
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=cuda
export HIP_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa  # オフスクリーン描画（メモリ節約）

echo "Starting training with memory-efficient settings..."
echo "  Logdir: ${LOGDIR}"
echo "  Task: ${TASK}"
echo "  MUJOCO_GL: ${MUJOCO_GL} (off-screen rendering)"
echo ""

cd /home/ruku/source/dreamer/external/dreamerv3

python - << PY
import os
os.environ["JAX_PLATFORMS"]="cuda"
os.environ["JAX_PLATFORM_NAME"]="cuda"
os.environ["HIP_VISIBLE_DEVICES"]=""
os.environ["MUJOCO_GL"]="osmesa"  # オフスクリーン描画（メモリ節約）
from dreamerv3 import main as dv3_main
dv3_main.main()
PY

