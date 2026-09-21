#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
OUTPUT="${ROOT}/.runtime/memp/gpt4o_script_query_cosine/memory"
TRAJECTORIES="${ROOT}/src/MemP_NCF/ProcedureMem/Alfworld/alfworld_format_traj.json"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing private API environment: ${ENV_FILE}" >&2
  exit 1
fi
if [[ ! -f "${TRAJECTORIES}" ]]; then
  echo "Missing public trajectory source: ${TRAJECTORIES}" >&2
  echo "Run: bash scripts/memp/prepare_public_sources.sh" >&2
  exit 1
fi

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
source "${ENV_FILE}"
set +a

export MEMP_API_KEY="${YUNWU_API_KEY:?YUNWU_API_KEY is required}"
export MEMP_API_BASE="${YUNWU_BASE_URL:-https://yunwu.ai/v1}"
export MEMP_CHAT_MODEL="gpt-4o"
export MEMP_EMBEDDING_MODEL="text-embedding-3-small"

mkdir -p "${OUTPUT}"

exec "${PYTHON}" "${ROOT}/src/MemP_NCF/ProcedureMem/build_script_memory.py" \
  --output-dir "${OUTPUT}" \
  --memory-size 300 \
  --workers 16 \
  --prompt-mode alfworld-faithful
