#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-yunwu}"
if [ "$#" -gt 0 ]; then shift; fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
export SKILLDAG_PROFILE="${PROFILE}"
set -a
source "${ENV_FILE}"
set +a

if [ -z "${SKILLDAG_LLM_API_KEY:-}" ]; then
  echo "ERROR: Judge API key is empty for profile ${PROFILE}." >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/src/SkillDAG_NCF/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
exec "${PYTHON}" scripts/ncf_v3/run_phase_template_judge.py "$@"
