#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="memp_gpt4o_release_corrected_top3"
OUTPUT="${ROOT}/.runtime/memp/gpt4o_script_release_corrected/top3_test134"
LOG_FILE="${OUTPUT}/run.log"

mkdir -p "${OUTPUT}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && bash scripts/memp/run_gpt4o_release_corrected_top3_test134.sh 2>&1 | tee -a '${LOG_FILE}'"

echo "Started tmux session: ${SESSION}"
echo "Log: ${LOG_FILE}"
echo "Results: ${OUTPUT}"
