#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
MEMORY="${ROOT}/.runtime/memp/gpt4o_script_release_corrected/memory"
SMOKE="${ROOT}/.runtime/memp/gpt4o_script_release_corrected/top3_smoke10_test"
OUTPUT="${ROOT}/.runtime/memp/gpt4o_script_release_corrected/top3_test134"

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
  echo "Missing release-corrected Script bank." >&2
  exit 2
fi

mkdir -p "${OUTPUT}"
if [[ ! -f "${OUTPUT}/idx_000.json" && -f "${SMOKE}/idx_000.json" ]]; then
  for index in $(seq -f '%03g' 0 9); do
    cp "${SMOKE}/idx_${index}.json" "${OUTPUT}/idx_${index}.json"
  done
fi

exec "${PYTHON}" "${ROOT}/src/MemP_NCF/ProcedureMem/eval_script_query_cosine.py" \
  --split test \
  --memory-dir "${MEMORY}" \
  --output-dir "${OUTPUT}" \
  --batch-size 10 \
  --limit 134 \
  --max-steps 30 \
  --top-k 3 \
  --faiss-l2-threshold 0.5
