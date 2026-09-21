#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"

if [ ! -f "${ENV_FILE}" ] || [ ! -x "${PYTHON}" ]; then
  echo "ERROR: private SkillDAG environment is incomplete." >&2
  exit 2
fi

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
export SKILLDAG_PROFILE="yunwu"
set -a
source "${ENV_FILE}"
set +a

if [ -z "${SKILLDAG_LLM_API_KEY:-}" ]; then
  echo "ERROR: YUNWU/MiniMax Judge API key is empty." >&2
  exit 2
fi

cd "${ROOT}"
exec "${PYTHON}" scripts/skillbench_ncf/judge_graded_candidates_v4.py "$@"
