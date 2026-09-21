#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"
MEMORY_DIR="${ROOT}/.runtime/memp/gpt4o_script_query_cosine/memory"
CHECKPOINT="${ROOT}/.runtime/memp_ncf/models/query_only_neumf_v17_balanced_dual_coverage/neumf.pt"
MEMORY_EMBEDDINGS="${MEMORY_DIR}/query_embeddings.json"

SPLIT="${1:-dev}"
LIMIT="${2:-10}"
OUTPUT_DIR="${3:-${ROOT}/.runtime/memp_ncf/eval/gpt4o_script_query_cosine_ncf_${SPLIT}_${LIMIT}}"

if [[ "${SPLIT}" != "dev" && "${SPLIT}" != "test" ]]; then
  echo "Usage: $0 [dev|test] [limit] [output_dir]" >&2
  exit 2
fi

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
source "${ENV_FILE}"
set +a

export MEMP_API_KEY="${YUNWU_API_KEY:?YUNWU_API_KEY is required}"
export MEMP_API_BASE="${YUNWU_BASE_URL:-https://yunwu.ai/v1}"
export MEMP_CHAT_MODEL="gpt-4o"
export MEMP_EMBEDDING_MODEL="text-embedding-3-small"
export ALFWORLD_DATA="${ROOT}/.runtime/skilldag/data/alfworld"
# TextWorld copies a 34 MB Fast Downward library per environment process.
# Keep those ephemeral copies off the nearly-full system /tmp filesystem.
export TMPDIR="${MEMP_TMPDIR:-/dev/shm/agentskillevolution_memp_v17}"
mkdir -p "${TMPDIR}"

for required in "${MEMORY_DIR}/script_bank.json" "${MEMORY_DIR}/query_embeddings.json" "${CHECKPOINT}" "${MEMORY_EMBEDDINGS}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
exec "${PYTHON}" "${ROOT}/src/MemP_NCF/ProcedureMem/eval_script_query_cosine.py" \
  --split "${SPLIT}" \
  --memory-dir "${MEMORY_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 10 \
  --limit "${LIMIT}" \
  --max-steps 30 \
  --top-k 10 \
  --faiss-l2-threshold 0.5 \
  --retriever query_cosine_ncf \
  --ncf-checkpoint "${CHECKPOINT}" \
  --ncf-memory-embeddings "${MEMORY_EMBEDDINGS}" \
  --ncf-candidate-k 50 \
  --ncf-candidate-threshold 0.5 \
  --ncf-embedding-model text-embedding-3-small
