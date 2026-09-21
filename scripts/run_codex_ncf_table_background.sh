#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="codex_ncf_table"
RUN_ROOT="${ROOT}/.runtime/codex_ncf_table"
LOG="${RUN_ROOT}/manager.log"
RUNNER="${ROOT}/scripts/run_codex_ncf_table_background.sh"
MODEL="openai/gpt-5.2-codex"

run_all() {
  mkdir -p "${RUN_ROOT}"

  echo "[1/2] SkillsBench V16 search-NCF, 87 tasks, attempts=1"
  SKILLBENCH_V16_NCF_SESSION=skillbench_v16_codex_yunwu_once \
  SKILLBENCH_V16_NCF_RUN_ROOT="${ROOT}/.runtime/skillbench_ncf/eval/v16_skillonly_codex_yunwu_once" \
  SKILLBENCH_V16_NCF_JOB_NAME_OVERRIDE=skillsbench-v16-skillonly-codex-yunwu-once \
  SKILLBENCH_V16_NCF_AGENT_MODEL="${MODEL}" \
  SKILLBENCH_V16_NCF_AGENT_LABEL=gpt-5.2-codex \
  SKILLBENCH_V16_NCF_REASONING_EFFORT=high \
  SKILLBENCH_V16_NCF_WORKERS=5 \
    bash "${ROOT}/scripts/skillbench_ncf/run_v16_skillonly_once_background.sh" run

  echo "[2/2] ALFWorld phase NCF + every-search NCF, 140 tasks, max_steps=30"
  SKILLDAG_AGENT_MODEL_OVERRIDE="${MODEL}" \
  MAX_GAMES=140 MAX_STEPS=30 MAX_WORKERS=3 \
  EXP_NAME=skilldag_ncf_v3_hybrid_codex_yunwu_once \
    bash "${ROOT}/scripts/run_paper_main_alfworld_once.sh" skilldag-ncf-v3-hybrid yunwu

  echo "[done] both Codex table runs completed"
}

status() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "running: tmux=${SESSION}"
  elif [ -f "${LOG}" ] && rg -q "\[done\] (all three|both)" "${LOG}"; then
    echo "completed: log=${LOG}"
  elif [ -f "${LOG}" ]; then
    echo "stopped/incomplete: log=${LOG}"
  else
    echo "not started"
  fi
}

case "${1:-start}" in
  start)
    mkdir -p "${RUN_ROOT}"
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      echo "ERROR: tmux session already exists: ${SESSION}" >&2
      exit 3
    fi
    tmux new-session -d -s "${SESSION}" "bash '${RUNNER}' run 2>&1 | tee '${LOG}'"
    echo "started: ${SESSION}"
    ;;
  run) run_all ;;
  status) status ;;
  logs)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      tmux capture-pane -pt "${SESSION}" -S -160
    elif [ -f "${LOG}" ]; then
      tail -n 160 "${LOG}"
    else
      status
    fi
    ;;
  *) echo "usage: $0 {start|run|status|logs}" >&2; exit 2 ;;
esac
