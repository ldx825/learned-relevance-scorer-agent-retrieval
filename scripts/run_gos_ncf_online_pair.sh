#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${ROOT}/src/GraphOfSkills_NCF"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
SKILLS_DIR="${ROOT}/.runtime/skilldag/data/skilldag/alfworld_skills"
GRAPH="${ROOT}/.runtime/skilldag/data/skilldag/skilldag_graphs/skillgraph_alfworld.json"
EMBEDDINGS="${ROOT}/.runtime/skilldag/data/skilldag/skilldag_graphs/skillgraph_alfworld.embeddings.json"
MODEL="${ROOT}/.runtime/gos_ncf/models/neumf_graded_v2_clean/neumf_graded.pt"
CACHE="${ROOT}/.runtime/gos_ncf/cache/online_task_embeddings.json"
STAMP="${GOS_NCF_ONLINE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TASK_INDEX="${GOS_NCF_ONLINE_TASK_INDEX:-0}"
MAX_STEPS="${GOS_NCF_ONLINE_MAX_STEPS:-12}"

for required in "${PYTHON}" "${ENV_FILE}" "${SKILLS_DIR}" "${GRAPH}" "${EMBEDDINGS}" "${MODEL}"; do
  if [ ! -e "${required}" ]; then
    echo "ERROR: missing ${required}" >&2
    exit 2
  fi
done

export SKILLDAG_PROFILE="${SKILLDAG_PROFILE:-yunwu}"
export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
source "${ENV_FILE}"
set +a

export API_KEY="${SKILLDAG_LLM_API_KEY}"
export BASE_URL="${SKILLDAG_LLM_BASE}"
export GOS37_EMBEDDING_API_KEY="${SKILLDAG_EMBEDDING_API_KEY}"
export GOS37_EMBEDDING_BASE="${SKILLDAG_EMBEDDING_BASE}"
export GOS37_EMBEDDING_MODEL="${SKILLDAG_EMBEDDING_MODEL}"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export LLM_REQUEST_TIMEOUT_SECS="${LLM_REQUEST_TIMEOUT_SECS:-90}"

cd "${PROJECT}"

for mode in graph37 graph37_ncf; do
  exp_name="paired_${STAMP}_task${TASK_INDEX}"
  echo "Running ${mode}: task=${TASK_INDEX}, max_steps=${MAX_STEPS}, model=${SKILLDAG_LLM_MODEL}"
  "${PYTHON}" evaluation/alfworld_run.py \
    --model "${SKILLDAG_LLM_MODEL}" \
    --split dev \
    --max_workers 1 \
    --max_steps "${MAX_STEPS}" \
    --exp_name "${exp_name}" \
    --use_skill \
    --mode "${mode}" \
    --skills_dir "${SKILLS_DIR}" \
    --skill_graph "${GRAPH}" \
    --skill_embeddings "${EMBEDDINGS}" \
    --ncf_model "${MODEL}" \
    --embedding_cache "${CACHE}" \
    --task_indices "${TASK_INDEX}" \
    --max_games 1
done
