#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-yunwu}"
if [ "$#" -gt 0 ]; then shift; fi
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
PROJECT="${ROOT}/src/SkillDAG_NCF"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"

if [ ! -f "${ENV_FILE}" ] || [ ! -x "${PYTHON}" ]; then
  echo "ERROR: SkillDAG private environment is incomplete." >&2
  exit 2
fi
export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
export SKILLDAG_PROFILE="${PROFILE}"
set -a
source "${ENV_FILE}"
set +a
if [ -z "${SKILLDAG_EMBEDDING_API_KEY:-}" ]; then
  echo "ERROR: embedding API key is empty for profile ${SKILLDAG_PROFILE}." >&2
  exit 2
fi
export PYTHONPATH="${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT}"
exec "${PYTHON}" scripts/data/04_build_train_task_candidates.py "$@"
