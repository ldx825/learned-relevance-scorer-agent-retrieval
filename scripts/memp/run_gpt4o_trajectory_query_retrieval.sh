#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"
MEMORY_DIR="${ROOT}/.runtime/memp/gpt4o_trajectory_query_cosine/memory"
CHECKPOINT="${ROOT}/.runtime/memp_ncf/models/query_only_neumf_v17_balanced_dual_coverage/neumf.pt"
MEMORY_EMBEDDINGS="${MEMORY_DIR}/query_embeddings.json"

RETRIEVER="${1:-cosine}"
SPLIT="${2:-dev}"
LIMIT="${3:-10}"
OUTPUT_DIR="${4:-${ROOT}/.runtime/memp_ncf/eval/gpt4o_trajectory_${RETRIEVER}_${SPLIT}_${LIMIT}}"

if [[ "${RETRIEVER}" != "cosine" && "${RETRIEVER}" != "ncf" ]]; then
  echo "Usage: $0 [cosine|ncf] [dev|test] [limit] [output_dir]" >&2
  exit 2
fi
if [[ "${SPLIT}" != "dev" && "${SPLIT}" != "test" ]]; then
  echo "Usage: $0 [cosine|ncf] [dev|test] [limit] [output_dir]" >&2
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
export TMPDIR="${MEMP_TMPDIR:-/dev/shm/agentskillevolution_memp_trajectory}"
mkdir -p "${TMPDIR}"

for required in "${MEMORY_DIR}/trajectory_bank.json" "${MEMORY_DIR}/query_embeddings.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
COMMON_ARGS=(
  --split "${SPLIT}"
  --memory-dir "${MEMORY_DIR}"
  --memory-format trajectory
  --output-dir "${OUTPUT_DIR}"
  --batch-size 10
  --limit "${LIMIT}"
  --max-steps 30
  --top-k 10
  --faiss-l2-threshold 0.5
)

if [[ "${RETRIEVER}" == "cosine" ]]; then
  exec "${PYTHON}" "${ROOT}/src/MemP_NCF/ProcedureMem/eval_script_query_cosine.py" \
    "${COMMON_ARGS[@]}" \
    --retriever query_cosine
fi

for required in "${CHECKPOINT}" "${MEMORY_EMBEDDINGS}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done
exec "${PYTHON}" "${ROOT}/src/MemP_NCF/ProcedureMem/eval_script_query_cosine.py" \
  "${COMMON_ARGS[@]}" \
  --retriever query_cosine_ncf \
  --ncf-checkpoint "${CHECKPOINT}" \
  --ncf-memory-embeddings "${MEMORY_EMBEDDINGS}" \
  --ncf-candidate-k 50 \
  --ncf-candidate-threshold 0.5 \
  --ncf-embedding-model text-embedding-3-small
