#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
PROFILES="${ROOT}/.runtime/memp/travelplanner/ncf_v2/memory_profiles/memory_capability_profiles.json"
OUTPUT="${ROOT}/data/memp_ncf/travelplanner_atomic_memory_gpt4o_v1"

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
export SKILLDAG_LLM_API_KEY="${YUNWU_API_KEY:?YUNWU_API_KEY is required}"
export SKILLDAG_LLM_BASE="${YUNWU_BASE_URL:-https://yunwu.ai/v1}"
export PYTHONPATH="${ROOT}/scripts/memp${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -u "${ROOT}/scripts/memp/extract_travelplanner_atomic_memory_operations.py" \
  --memory-profiles "${PROFILES}" \
  --output-dir "${OUTPUT}" \
  --model gpt-4o \
  --max-workers 8 \
  --max-attempts 3 \
  "$@"
