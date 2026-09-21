#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  codex|minimax|openai|openrouter|yunwu)
    export SKILLDAG_PROFILE="$1"
    shift
    ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${ROOT}/references/SkillDAG"
RUNTIME="${ROOT}/.runtime/skilldag"
PYTHON="${ROOT}/.envs/skilldag-debug/bin/python"
ENV_FILE="${RUNTIME}/skilldag.env"
SOURCE_GRAPH="${RUNTIME}/data/skilldag/skilldag_graphs/skillgraph_200.json"
SOURCE_CACHE="${RUNTIME}/data/skilldag/skilldag_graphs/skillgraph_200.embeddings.json"
SKILLS_DIR="${RUNTIME}/data/skilldag/skillsets/skills_200"
WORK_DIR="${RUNTIME}/debug/cli"
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

if [ -z "${SKILLDAG_EMBEDDING_API_KEY:-}" ]; then
  echo "ERROR: ${SKILLDAG_PROFILE} embedding API key is empty in ${ENV_FILE}." >&2
  echo "Fill YUNWU_API_KEY for yunwu, OPENAI_API_KEY for codex, or OPENROUTER_API_KEY for minimax." >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"
if [ ! -f "${WORK_GRAPH}" ] || [ "${SKILLDAG_RESET_DEBUG_GRAPH:-0}" = "1" ]; then
  cp "${SOURCE_GRAPH}" "${WORK_GRAPH}"
  cp "${SOURCE_CACHE}" "${WORK_DIR}/skillgraph.embeddings.json"
fi

if [ "$#" -eq 0 ]; then
  set -- search "put a clean apple in the refrigerator" --top-k 5 --depth 2
fi

cd "${PROJECT}"
echo "Profile: ${SKILLDAG_PROFILE} (${SKILLDAG_MODEL})"
exec "${PYTHON}" -Xfrozen_modules=off -m debugpy --listen 2607 --wait-for-client \
  -m skilldag graph \
  --graph-path "${WORK_GRAPH}" \
  --skills-dir "${SKILLS_DIR}" \
  "$@"
