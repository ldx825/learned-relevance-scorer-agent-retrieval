#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
PILOT="${ROOT}/.runtime/memp/gpt4o_script_prompt_pilot30/release-corrected"
OUTPUT="${ROOT}/.runtime/memp/gpt4o_script_release_corrected/memory"
DOCUMENTS="${OUTPUT}/documents.json"
RELEASE_DOCUMENTS="${ROOT}/src/MemP_NCF/ProcedureMem/memory/alfworld/direct/documents.json"

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
source "${ENV_FILE}"
set +a

export MEMP_API_KEY="${YUNWU_API_KEY:?YUNWU_API_KEY is required}"
export MEMP_API_BASE="${YUNWU_BASE_URL:-https://yunwu.ai/v1}"
export MEMP_CHAT_MODEL="gpt-4o"
export MEMP_EMBEDDING_MODEL="text-embedding-3-small"

mkdir -p "${OUTPUT}"
if [[ ! -f "${OUTPUT}/script_bank.json" && -f "${PILOT}/script_bank.json" ]]; then
  cp "${PILOT}/script_bank.json" "${OUTPUT}/script_bank.json"
fi

"${PYTHON}" "${ROOT}/src/MemP_NCF/ProcedureMem/build_script_memory.py" \
  --output-dir "${OUTPUT}" \
  --memory-size 300 \
  --workers 16 \
  --prompt-mode release-corrected

"${PYTHON}" "${ROOT}/src/MemP_NCF/ProcedureMem/export_script_documents.py" \
  --bank "${OUTPUT}/script_bank.json" \
  --output "${DOCUMENTS}"

"${PYTHON}" "${ROOT}/src/MemP_NCF/ProcedureMem/export_script_documents.py" \
  --bank "${OUTPUT}/script_bank.json" \
  --output "${RELEASE_DOCUMENTS}"
