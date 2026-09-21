#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="memp_minimax_script_test134"
LOG_DIR="${ROOT}/.runtime/memp/minimax_m27_script_query_cosine/test134"
LOG_FILE="${LOG_DIR}/run.log"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already running: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && bash scripts/memp/run_minimax_script_cosine_test134.sh 2>&1 | tee -a '${LOG_FILE}'"

echo "Started tmux session: ${SESSION}"
echo "Log: ${LOG_FILE}"
echo "Inspect: tmux capture-pane -pt ${SESSION} -S -120"
