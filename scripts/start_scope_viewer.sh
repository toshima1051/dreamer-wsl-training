#!/usr/bin/env bash
set -euo pipefail

BASEDIR="${1:-$HOME/logdir}"
PORT="${2:-8000}"

source "/home/ruku/source/dreamer/.venv311/bin/activate" 2>/dev/null || true

echo "Starting Scope viewer..."
echo "  Base directory: ${BASEDIR}"
echo "  Port: ${PORT}"
echo "  URL: http://localhost:${PORT}"
echo ""
echo "Make sure to select the correct log directory in the web interface."
echo "Example: ${BASEDIR}/dreamer/20251106T193622 or ${BASEDIR}/20251106T193622"
echo ""

exec python -m scope.viewer --basedir "${BASEDIR}" --port "${PORT}"


