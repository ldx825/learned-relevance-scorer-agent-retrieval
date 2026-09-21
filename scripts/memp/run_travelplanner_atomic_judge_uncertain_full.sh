#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
CANDIDATES="${ROOT}/data/memp_ncf/travelplanner_atomic_memory_candidates_v4/judge_uncertain_train.jsonl"
OUTPUT="${ROOT}/data/memp_ncf/travelplanner_atomic_memory_judge_uncertain_v1"

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
export SKILLDAG_LLM_API_KEY="${YUNWU_API_KEY:?YUNWU_API_KEY is required}"
export SKILLDAG_LLM_BASE="${YUNWU_BASE_URL:-https://yunwu.ai/v1}"
export PYTHONPATH="${ROOT}/scripts/memp${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -u "${ROOT}/scripts/memp/run_travelplanner_atomic_operation_judge.py" \
  --candidates "${CANDIDATES}" \
  --output-dir "${OUTPUT}" \
  --model gpt-4o \
  --max-workers 12 \
  --max-attempts 3 \
  --timeout 240 \
  --max-tokens 1800

