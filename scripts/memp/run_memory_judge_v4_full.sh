#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
CANDIDATES="${ROOT}/.runtime/memp_ncf/alfworld_script_candidates_v4/judge_candidates.jsonl"
OUTPUT_DIR="${ROOT}/.runtime/memp_ncf/judge_full945_v4_failure_targeted"

set -a
export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
source "${ENV_FILE}"
set +a
export SKILLDAG_LLM_API_KEY="${YUNWU_API_KEY:?YUNWU_API_KEY is required}"
export SKILLDAG_LLM_BASE="${YUNWU_BASE_URL:-https://yunwu.ai/v1}"

exec "${PYTHON}" "${ROOT}/scripts/memp/run_memory_task_judge.py" \
  --candidates "${CANDIDATES}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-workers 12 \
  --max-attempts 3 \
  --model MiniMax-M2.7
