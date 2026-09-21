#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
# Reuse the exact frozen Script bank and query embeddings from the GPT-4o run.
# Only the acting backbone changes, which makes this a paired protocol rerun.
MEMORY="${ROOT}/.runtime/memp/gpt4o_script_query_cosine/memory"
RUNTIME="${ROOT}/.runtime/memp/minimax_m27_script_query_cosine"

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
source "${ENV_FILE}"
set +a

export MEMP_API_KEY="${YUNWU_API_KEY:?YUNWU_API_KEY is required}"
export MEMP_API_BASE="${YUNWU_BASE_URL:-https://yunwu.ai/v1}"
export MEMP_CHAT_MODEL="MiniMax-M2.7"
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
  --top-k 10 \
  --faiss-l2-threshold 0.5
