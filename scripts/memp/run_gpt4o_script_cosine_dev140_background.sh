#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${ROOT}/.runtime/memp/gpt4o_script_query_cosine"
SOURCE="${RUNTIME}/smoke10_dev"
OUTPUT="${RUNTIME}/dev140"
LOG="${RUNTIME}/dev140.log"
SESSION="memp_gpt4o_script_cosine_dev140"

if [[ ! -f "${RUNTIME}/memory/script_bank.json" ]]; then
  echo "Missing frozen Script bank: ${RUNTIME}/memory/script_bank.json" >&2
  exit 2
fi

mkdir -p "${OUTPUT}"

# The existing ten-task pilot used the exact same bank, model, retrieval and
# environment settings.  Reuse it as the first batch and let the resumable
# runner execute only positions 10..139.
for index in $(seq 0 9); do
  printf -v padded "%03d" "${index}"
  source_path="${SOURCE}/idx_${padded}.json"
  target_path="${OUTPUT}/idx_${padded}.json"
  if [[ -f "${source_path}" && ! -f "${target_path}" ]]; then
    cp -a "${source_path}" "${target_path}"
  fi
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session already running: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && exec bash -lc '
    set -euo pipefail
    export AGENT_SKILL_EVOLUTION_ROOT=${ROOT}
    set -a
    source .runtime/skilldag/skilldag.env
    set +a
    export MEMP_API_KEY=\${YUNWU_API_KEY:?YUNWU_API_KEY is required}
    export MEMP_API_BASE=\${YUNWU_BASE_URL:-https://yunwu.ai/v1}
    export MEMP_CHAT_MODEL=gpt-4o
    export MEMP_EMBEDDING_MODEL=text-embedding-3-small
    export ALFWORLD_DATA=${ROOT}/.runtime/skilldag/data/alfworld
    exec ${ROOT}/.envs/skilldag-debug/bin/python \
      ${ROOT}/src/MemP_NCF/ProcedureMem/eval_script_query_cosine.py \
      --split dev \
      --memory-dir ${RUNTIME}/memory \
      --output-dir ${OUTPUT} \
      --batch-size 10 \
      --limit 140 \
      --max-steps 30 \
      --top-k 10 \
      --faiss-l2-threshold 0.5
  ' >> '${LOG}' 2>&1"

echo "Started ${SESSION}"
echo "Log: ${LOG}"
echo "Output: ${OUTPUT}"
