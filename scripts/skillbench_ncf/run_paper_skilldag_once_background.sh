#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="${ROOT}/scripts/skillbench_ncf/run_paper_skilldag_once_background.sh"
SESSION=skillbench_paper_skilldag_once
RUN_ROOT="${ROOT}/.runtime/skillbench_ncf/eval/paper_skilldag_once"
CANONICAL_TASKS="${ROOT}/.runtime/skillbench_ncf/official/skillsbench-v1.0/tasks"
TASKS_DIR="${RUN_ROOT}/tasks"
JOBS_DIR="${RUN_ROOT}/results"
JOB_NAME=skillsbench-paper-skilldag-once
CONFIG_TEMPLATE="${ROOT}/configs/skillbench_ncf/paper_skilldag_remaining_once.yaml"
CONFIG="${RUN_ROOT}/config.yaml"
LOG="${RUN_ROOT}/background.log"
BOOT_LOG="${RUN_ROOT}/launcher.log"
SCORE_LOG="${RUN_ROOT}/score.log"
MANIFEST="${RUN_ROOT}/reuse_manifest.json"
GRAPH="${ROOT}/.runtime/skillbench_ncf/sources/skillgraph_1000.json"
SKILLS="${ROOT}/.runtime/skillbench_ncf/sources/skills_1000"
PACKAGE="${ROOT}/src/SkillDAG_NCF"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"

setup_runtime() {
  export PATH="${ROOT}/.envs/skillsbench-harbor/bin:${ROOT}/.runtime/skillbench_env/docker-bin:${PATH}"
  export DOCKER_HOST="unix://${ROOT}/.runtime/skillbench_env/docker.sock"
  export XDG_RUNTIME_DIR="${ROOT}/.runtime/skillbench_env/xdg"
  export DOCKER_CONFIG="${ROOT}/.runtime/skillbench_env/docker-config"
  export SKILLDAG_DOCKER_CPUS=0
  export SKILLDAG_BUILD_NETWORK=host
  export SKILLDAG_NETWORK_MODE=host
  export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"

  set -a
  source "${ROOT}/.runtime/skilldag/skilldag.env"
  set +a
  export OPENAI_API_KEY="${YUNWU_API_KEY}"
  export OPENAI_BASE_URL="${YUNWU_BASE_URL}"
  export SKILLDAG_EMBEDDING_API_KEY="${YUNWU_API_KEY}"
  export SKILLDAG_EMBEDDING_BASE="${YUNWU_BASE_URL}"
  export SKILLDAG_EMBEDDING_MODEL=text-embedding-3-large
  # A pure SkillDAG baseline must never inherit NCF from the caller's shell.
  unset SKILLDAG_NCF_MODEL SKILLDAG_NCF_PLAN SKILLDAG_NCF_ALPHA || true
  unset SKILLDAG_NCF_CANDIDATE_K SKILLDAG_NCF_EXPECTED_POOL_SIZE || true
}

build_manifest() {
  "${PYTHON}" "${ROOT}/scripts/skillbench_ncf/build_paper_baseline_reuse_manifest.py"
}

preflight() {
  for path in "${CANONICAL_TASKS}" "${SKILLS}" "${GRAPH}" "${PACKAGE}" "${MANIFEST}"; do
    if [ ! -e "${path}" ]; then
      echo "ERROR: missing required input: ${path}" >&2
      exit 2
    fi
  done
  "${PYTHON}" - "${MANIFEST}" "${CANONICAL_TASKS}" "${GRAPH}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tasks = sorted(path.name for path in Path(sys.argv[2]).iterdir() if path.is_dir())
graph = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
assert len(tasks) == 87
assert "mhc-layer-impl" not in tasks
assert len(graph.get("nodes") or {}) == 1000
assert manifest["paper_snapshot_tasks"] == 87
assert manifest["attempts_per_task"] == 1
# Nine existing pure-SkillDAG trials used this same task tree and runtime.
assert manifest["accepted_count"] == 9
assert manifest["remaining_count"] == 78
print(
    f"[preflight] paper snapshot=87, reusable={manifest['accepted_count']}, "
    f"new={manifest['remaining_count']}, skills=1000"
)
PY
  docker info >/dev/null
  command -v harbor >/dev/null
  command -v envsubst >/dev/null
}

prepare_tasks() {
  mapfile -t tasks < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path
x = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("\n".join(x["remaining_task_ids"]))
PY
  )
  local args=()
  local task
  for task in "${tasks[@]}"; do
    args+=(--task "${task}")
  done
  PYTHONPATH="${PACKAGE}:${PACKAGE}/src" "${PYTHON}" \
    "${PACKAGE}/benchmarks/skillsbench/skilldag_benchmark.py" \
    --tasks-root "${CANONICAL_TASKS}" \
    --skills-root "${SKILLS}" \
    --skillgraph-path "${GRAPH}" \
    --skilldag-package-root "${PACKAGE}" \
    --output-root "${TASKS_DIR}" \
    --clean \
    "${args[@]}"

  local generated
  generated="$(find "${TASKS_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  if [ "${generated}" -ne "${#tasks[@]}" ]; then
    echo "ERROR: generated ${generated} tasks, expected ${#tasks[@]}" >&2
    exit 4
  fi
  if find "${TASKS_DIR}" \( -name ncf_bundle.json -o -name ncf_plan.json -o -name gold_skills.json \) -print -quit | grep -q .; then
    echo "ERROR: NCF or container-visible gold leaked into pure SkillDAG tasks" >&2
    exit 4
  fi
  echo "[preflight] pure SkillDAG tasks generated; NCF=false; gold-visible=false"
}

run_once() {
  setup_runtime
  mkdir -p "${RUN_ROOT}"
  build_manifest
  preflight
  if [ -e "${JOBS_DIR}/${JOB_NAME}/result.json" ]; then
    echo "ERROR: completed result exists; refusing to spend API again" >&2
    exit 3
  fi
  if [ -d "${JOBS_DIR}/${JOB_NAME}" ] && [ -n "$(find "${JOBS_DIR}/${JOB_NAME}" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "ERROR: partial result directory exists; inspect before rerun" >&2
    exit 3
  fi

  prepare_tasks
  export SKILLBENCH_PAPER_BASELINE_JOB_NAME="${JOB_NAME}"
  export SKILLBENCH_PAPER_BASELINE_JOBS_DIR="${JOBS_DIR}"
  export SKILLBENCH_PAPER_BASELINE_TASKS_DIR="${TASKS_DIR}"
  export SKILLBENCH_PAPER_BASELINE_WORKERS="${SKILLBENCH_PAPER_BASELINE_WORKERS:-5}"
  envsubst < "${CONFIG_TEMPLATE}" > "${CONFIG}"
  echo "[run] 78 remaining tasks, one attempt, ${SKILLBENCH_PAPER_BASELINE_WORKERS} workers"
  harbor run -c "${CONFIG}" 2>&1 | tee "${LOG}"
  "${PYTHON}" "${PACKAGE}/analysis/score_skillsbench_gos.py" \
    "${JOBS_DIR}" --job-name "${JOB_NAME}" --total 78 | tee "${SCORE_LOG}"
}

status() {
  local trial_count=0
  if [ -d "${JOBS_DIR}/${JOB_NAME}" ]; then
    trial_count="$(find "${JOBS_DIR}/${JOB_NAME}" -mindepth 2 -maxdepth 2 -name result.json | wc -l | tr -d ' ')"
  fi
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "running: ${trial_count}/78 new + 9 reused; tmux=${SESSION}"
  elif [ -f "${JOBS_DIR}/${JOB_NAME}/result.json" ]; then
    echo "completed: ${trial_count}/78 new + 9 reused; result=${JOBS_DIR}/${JOB_NAME}/result.json"
  elif [ "${trial_count}" -gt 0 ]; then
    echo "stopped/incomplete: ${trial_count}/78 new + 9 reused; inspect ${JOBS_DIR}/${JOB_NAME}"
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
    if [ -e "${JOBS_DIR}/${JOB_NAME}/result.json" ]; then
      echo "ERROR: completed result exists; refusing to rerun" >&2
      exit 3
    fi
    tmux new-session -d -s "${SESSION}" \
      "bash '${RUNNER}' run 2>&1 | tee '${BOOT_LOG}'"
    echo "started: ${SESSION}"
    ;;
  run) run_once ;;
  status) status ;;
  logs)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      tmux capture-pane -pt "${SESSION}" -S -120
    elif [ -f "${LOG}" ]; then
      tail -n 120 "${LOG}"
    elif [ -f "${BOOT_LOG}" ]; then
      tail -n 120 "${BOOT_LOG}"
    else
      status
    fi
    ;;
  *) echo "usage: $0 {start|run|status|logs}" >&2; exit 2 ;;
esac
