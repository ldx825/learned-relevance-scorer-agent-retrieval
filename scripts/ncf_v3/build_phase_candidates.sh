#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-yunwu}"
if [ "$#" -gt 0 ]; then shift; fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"

if [ ! -f "${ENV_FILE}" ] || [ ! -x "${PYTHON}" ]; then
  echo "ERROR: private SkillDAG environment is incomplete." >&2
  exit 2
fi

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
export SKILLDAG_PROFILE="${PROFILE}"
set -a
source "${ENV_FILE}"
set +a

if [ -z "${SKILLDAG_EMBEDDING_API_KEY:-}" ]; then
  echo "ERROR: embedding API key is empty for profile ${PROFILE}." >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/src/SkillDAG_NCF/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
exec "${PYTHON}" scripts/ncf_v3/build_phase_candidates.py "$@"
