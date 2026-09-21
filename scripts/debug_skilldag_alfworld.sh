#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [codex|minimax|yunwu]" >&2
  exit 2
fi
export SKILLDAG_PROFILE="${1:-${SKILLDAG_PROFILE:-auto}}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${ROOT}/references/SkillDAG"
RUNTIME="${ROOT}/.runtime/skilldag"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
ENV_FILE="${RUNTIME}/skilldag.env"
SOURCE_GRAPH="${RUNTIME}/data/skilldag/skilldag_graphs/skillgraph_alfworld.json"
SOURCE_CACHE="${RUNTIME}/data/skilldag/skilldag_graphs/skillgraph_alfworld.embeddings.json"
SKILLS_DIR="${RUNTIME}/data/skilldag/alfworld_skills"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${SKILLDAG_DEBUG_EXP_NAME:-debug_single_${STAMP}}"
WORK_DIR="${RUNTIME}/debug/alfworld/${EXP_NAME}"
WORK_GRAPH="${WORK_DIR}/skillgraph.json"

for required in "${PYTHON}" "${ENV_FILE}" "${SOURCE_GRAPH}" "${SOURCE_CACHE}"; do
  if [ ! -e "${required}" ]; then
    echo "ERROR: missing ${required}; run scripts/setup_skilldag_debug.sh first." >&2
    exit 2
  fi
done

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
source "${ENV_FILE}"
set +a
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${ROOT}/.envs/skilldag-debug/bin:${PATH}"
# Keep the whole one-task learning run in this debugpy process and surface the
# original API exception instead of converting it into a zero-token result.
export SKILLDAG_DEBUG_INLINE="${SKILLDAG_DEBUG_INLINE:-1}"
export SKILLDAG_DEBUG_RERAISE="${SKILLDAG_DEBUG_RERAISE:-1}"
export SKILLDAG_DEBUG_NO_RETRY="${SKILLDAG_DEBUG_NO_RETRY:-1}"

if [ -z "${SKILLDAG_API_KEY:-}" ]; then
  echo "ERROR: ${SKILLDAG_PROFILE} API key is empty in ${ENV_FILE}." >&2
  echo "Fill YUNWU_API_KEY for yunwu, OPENAI_API_KEY for codex, or OPENROUTER_API_KEY for minimax." >&2
  exit 2
fi

if [ -z "${SKILLDAG_EMBEDDING_API_KEY:-}" ]; then
  echo "ERROR: ${SKILLDAG_PROFILE} embedding API key is empty in ${ENV_FILE}." >&2
  echo "Fill the selected profile's API key; Yunwu uses YUNWU_API_KEY for both chat and embeddings." >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"
cp "${SOURCE_GRAPH}" "${WORK_GRAPH}"
cp "${SOURCE_CACHE}" "${WORK_DIR}/skillgraph.embeddings.json"

echo "debugpy is waiting on port 2607"
echo "In VS Code, start: SkillDAG: Attach debugpy 2607"
echo "Experiment: ${EXP_NAME}"
echo "Profile: ${SKILLDAG_PROFILE} (${SKILLDAG_MODEL})"
echo "Inline debugging: ${SKILLDAG_DEBUG_INLINE}; re-raise errors: ${SKILLDAG_DEBUG_RERAISE}"
echo "API retry disabled for debugging: ${SKILLDAG_DEBUG_NO_RETRY}"

cd "${PROJECT}"
exec "${PYTHON}" -Xfrozen_modules=off -m debugpy --listen 2607 --wait-for-client \
  --configure-subProcess True \
  benchmarks/alfworld/run_alfworld.py \
  --model "${SKILLDAG_MODEL}" \
  --skilldag_api_base "${SKILLDAG_API_BASE}" \
  --skills_dir "${SKILLS_DIR}" \
  --skilldag_graph "${WORK_GRAPH}" \
  --split dev \
  --max_games 1 \
  --max_workers 1 \
  --max_steps "${SKILLDAG_DEBUG_MAX_STEPS:-12}" \
  --task_indices "${SKILLDAG_DEBUG_TASK_INDEX:-0}" \
  --exp_name "${EXP_NAME}"
