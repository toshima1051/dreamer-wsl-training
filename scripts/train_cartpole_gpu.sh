#!/usr/bin/env bash
set -euo pipefail

LOGDIR_BASE="${1:-$HOME/logdir/dreamer}"
TASK="${2:-gym_cartpole}"

TS="$(date +%Y%m%d-%H%M%S)"
LOGDIR="${LOGDIR_BASE}/${TS}"

source "/home/ruku/source/dreamer/.venv311/bin/activate"

# Force CUDA-only JAX
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=cuda
export HIP_VISIBLE_DEVICES=

cd /home/ruku/source/dreamer/external/dreamerv3

python -u dreamerv3/main.py \
  --logdir "${LOGDIR}" \
  --configs debug \
  --jax.platform gpu \
  --task "${TASK}" || {
  echo "\nHint: If task '${TASK}' is unknown, try one of:\n  --task gym_cartpole\n  --task dmc_cartpole_balance\n" >&2;
  exit 1;
}


