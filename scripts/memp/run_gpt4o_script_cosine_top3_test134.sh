#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
MEMORY="${ROOT}/.runtime/memp/gpt4o_script_query_cosine/memory"
RUNTIME="${ROOT}/.runtime/memp/gpt4o_script_query_cosine_top3"

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
source "${ENV_FILE}"
set +a

export MEMP_API_KEY="${YUNWU_API_KEY:?YUNWU_API_KEY is required}"
export MEMP_API_BASE="${YUNWU_BASE_URL:-https://yunwu.ai/v1}"
export MEMP_CHAT_MODEL="gpt-4o"
export MEMP_EMBEDDING_MODEL="text-embedding-3-small"
export ALFWORLD_DATA="${ROOT}/.runtime/skilldag/data/alfworld"

if [[ ! -f "${MEMORY}/script_bank.json" ]]; then
  echo "Missing frozen Script memory bank; run scripts/memp/build_gpt4o_script_bank.sh first." >&2
  exit 2
fi

mkdir -p "${RUNTIME}/test134"
exec "${PYTHON}" "${ROOT}/src/MemP_NCF/ProcedureMem/eval_script_query_cosine.py" \
  --split test \
  --memory-dir "${MEMORY}" \
  --output-dir "${RUNTIME}/test134" \
  --batch-size 10 \
  --limit 134 \
  --max-steps 30 \
  --top-k 3 \
  --faiss-l2-threshold 0.5
