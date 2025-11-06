#!/usr/bin/env bash
set -euo pipefail

# WSLでX11転送を使って動画を表示する設定
# Windows側でXサーバー（VcXsrv等）を起動してから実行してください

LOGDIR_BASE="${1:-$HOME/logdir/dreamer}"
TASK="${2:-dmc_cartpole_swingup}"

TS="$(date +%Y%m%d-%H%M%S)"
LOGDIR="${LOGDIR_BASE}/${TS}"

# DISPLAY環境変数を設定（X11転送用）
if [ -z "${DISPLAY:-}" ]; then
    # WSLgまたはローカルXサーバーを使用
    export DISPLAY=:0
fi

export MUJOCO_GL=glfw  # GUI描画を使用（動画が正しく表示される）

source /home/ruku/source/dreamer/.venv311/bin/activate

echo "Starting training with GUI rendering..."
echo "  Logdir: ${LOGDIR}"
echo "  Task: ${TASK}"
echo "  DISPLAY: ${DISPLAY}"
echo "  MUJOCO_GL: ${MUJOCO_GL}"
echo ""
echo "Note: Make sure X server (VcXsrv) is running on Windows side!"
echo ""

cd /home/ruku/source/dreamer/external/dreamerv3

python - << PY
import os
os.environ["JAX_PLATFORMS"]="cuda"
os.environ["JAX_PLATFORM_NAME"]="cuda"
os.environ["HIP_VISIBLE_DEVICES"]=""
os.environ["MUJOCO_GL"]="glfw"  # GUI描画
from dreamerv3 import main as dv3_main
dv3_main.main()
PY

