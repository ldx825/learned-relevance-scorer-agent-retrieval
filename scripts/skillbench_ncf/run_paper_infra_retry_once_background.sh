#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="${ROOT}/scripts/skillbench_ncf/run_paper_infra_retry_once_background.sh"
SESSION=skillbench_paper_infra_retry_once
RUN_ROOT="${ROOT}/.runtime/skillbench_ncf/eval/paper_skilldag_infra_retry_once"
CANONICAL_TASKS="${ROOT}/.runtime/skillbench_ncf/official/skillsbench-v1.0/tasks"
TASKS_DIR="${RUN_ROOT}/tasks"
JOBS_DIR="${RUN_ROOT}/results"
JOB_NAME=skillsbench-paper-skilldag-infra-retry-once
CONFIG_TEMPLATE="${ROOT}/configs/skillbench_ncf/paper_skilldag_infra_retry_once.yaml"
CONFIG="${RUN_ROOT}/config.yaml"
LOG="${RUN_ROOT}/background.log"
BOOT_LOG="${RUN_ROOT}/launcher.log"
SCORE_LOG="${RUN_ROOT}/score.log"
MANIFEST="${RUN_ROOT}/retry_manifest.json"
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
  unset SKILLDAG_NCF_MODEL SKILLDAG_NCF_PLAN SKILLDAG_NCF_ALPHA || true
  unset SKILLDAG_NCF_CANDIDATE_K SKILLDAG_NCF_EXPECTED_POOL_SIZE || true
}

prepare_tasks() {
  mapfile -t tasks < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import json,sys
print("\n".join(json.load(open(sys.argv[1]))["task_ids"]))
PY
  )
  local args=()
  local task
  for task in "${tasks[@]}"; do args+=(--task "${task}"); done
  PYTHONPATH="${PACKAGE}:${PACKAGE}/src" "${PYTHON}" \
    "${PACKAGE}/benchmarks/skillsbench/skilldag_benchmark.py" \
    --tasks-root "${CANONICAL_TASKS}" \
    --skills-root "${SKILLS}" \
    --skillgraph-path "${GRAPH}" \
    --skilldag-package-root "${PACKAGE}" \
    --output-root "${TASKS_DIR}" \
    --clean "${args[@]}"
  local generated
  generated="$(find "${TASKS_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  [[ "${generated}" -eq 11 ]] || { echo "Expected 11 tasks, got ${generated}" >&2; exit 4; }

  # These are precisely the tasks whose heavy Docker environments exceeded
  # Harbor's default 600-second build limit in the paper run.  Give builds a
  # full hour; this changes infrastructure setup only, not agent execution or
  # verifier budgets.
  local build_timeout="${SKILLBENCH_INFRA_BUILD_TIMEOUT_SEC:-3600}"
  find "${TASKS_DIR}" -mindepth 2 -maxdepth 2 -name task.toml -exec \
    sed -i -E "s/^build_timeout_sec = [0-9.]+$/build_timeout_sec = ${build_timeout}.0/" {} +
  local timeout_count
  timeout_count="$(rg -l "^build_timeout_sec = ${build_timeout}\\.0$" "${TASKS_DIR}" -g task.toml | wc -l | tr -d ' ')"
  [[ "${timeout_count}" -eq 11 ]] || {
    echo "Expected 11 patched build timeouts, got ${timeout_count}" >&2
    exit 4
  }
  if find "${TASKS_DIR}" \( -name ncf_bundle.json -o -name ncf_plan.json -o -name gold_skills.json \) -print -quit | grep -q .; then
    echo "ERROR: NCF or gold leaked into retry tasks" >&2
    exit 4
  fi
}

run_once() {
  setup_runtime
  mkdir -p "${RUN_ROOT}"
  "${PYTHON}" "${ROOT}/scripts/skillbench_ncf/build_paper_infra_retry_manifest.py"
  [[ ! -e "${JOBS_DIR}/${JOB_NAME}/result.json" ]] || { echo "Completed retry already exists" >&2; exit 3; }
  prepare_tasks
  export SKILLBENCH_PAPER_BASELINE_JOB_NAME="${JOB_NAME}"
  export SKILLBENCH_PAPER_BASELINE_JOBS_DIR="${JOBS_DIR}"
  export SKILLBENCH_PAPER_BASELINE_TASKS_DIR="${TASKS_DIR}"
  export SKILLBENCH_PAPER_BASELINE_WORKERS="${SKILLBENCH_PAPER_BASELINE_WORKERS:-3}"
  envsubst < "${CONFIG_TEMPLATE}" > "${CONFIG}"
  harbor run -c "${CONFIG}" 2>&1 | tee "${LOG}"
  "${PYTHON}" "${PACKAGE}/analysis/score_skillsbench_gos.py" \
    "${JOBS_DIR}" --job-name "${JOB_NAME}" --total 11 | tee "${SCORE_LOG}"
}

status() {
  local count=0
  if [[ -d "${JOBS_DIR}/${JOB_NAME}" ]]; then
    count="$(find "${JOBS_DIR}/${JOB_NAME}" -mindepth 2 -maxdepth 2 -name result.json | wc -l | tr -d ' ')"
  fi
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "running: ${count}/11; tmux=${SESSION}"
  elif [[ -f "${JOBS_DIR}/${JOB_NAME}/result.json" ]]; then
    echo "completed: ${count}/11"
  elif [[ "${count}" -gt 0 ]]; then
    echo "stopped/incomplete: ${count}/11"
  else
    echo "not started"
  fi
}

case "${1:-start}" in
  start)
    mkdir -p "${RUN_ROOT}"
    tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Session exists" >&2; exit 3; }
    tmux new-session -d -s "${SESSION}" "bash '${RUNNER}' run 2>&1 | tee '${BOOT_LOG}'"
    echo "started: ${SESSION}"
    ;;
  run) run_once ;;
  status) status ;;
  logs)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then tmux capture-pane -pt "${SESSION}" -S -120;
    elif [[ -f "${LOG}" ]]; then tail -n 120 "${LOG}"; else status; fi
    ;;
  *) echo "usage: $0 {start|run|status|logs}" >&2; exit 2 ;;
esac
