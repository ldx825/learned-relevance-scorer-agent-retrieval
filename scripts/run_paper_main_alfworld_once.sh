#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <gos|gos-ncf|skilldag|skilldag-ncf|skilldag-ncf-v3|skilldag-ncf-v3-search|skilldag-ncf-v3-hybrid> [yunwu|minimax|codex]" >&2
  exit 2
fi

METHOD="$1"
export SKILLDAG_PROFILE="${2:-${SKILLDAG_PROFILE:-yunwu}}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/.runtime/skilldag/skilldag.env"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"
GOS_PROJECT="${ROOT}/src/GraphOfSkills_NCF"
SKILLDAG_PROJECT="${ROOT}/src/SkillDAG_NCF"
RUNTIME="${ROOT}/.runtime/paper_main_alfworld"
ALFWORLD_DATA="${ROOT}/.runtime/skilldag/data/alfworld"
SKILLS_DIR="${ROOT}/.runtime/skilldag/data/skilldag/alfworld_skills"
SOURCE_GRAPH="${ROOT}/.runtime/skilldag/data/skilldag/skilldag_graphs/skillgraph_alfworld.json"
SOURCE_GRAPH_CACHE="${ROOT}/.runtime/skilldag/data/skilldag/skilldag_graphs/skillgraph_alfworld.embeddings.json"
GOS_WORKSPACE="${ROOT}/.runtime/gos_ncf/gos_workspace/skills_1000_v1"
SKILL_EMBEDDINGS="${ROOT}/data/alfworld_task_skill/shared/embeddings/skill_embeddings_37.json"
NCF_MODEL="${ROOT}/.runtime/gos_ncf/models/neumf_graded_v2_clean/neumf_graded.pt"
NCF_V3_MODEL="${SKILLDAG_NCF_V3_MODEL_OVERRIDE:-${ROOT}/.runtime/skilldag_ncf_v3/models/content_neumf_v1/neumf.pt}"
NCF_V3_CALIBRATION="${SKILLDAG_NCF_V3_CALIBRATION_OVERRIDE:-${ROOT}/.runtime/skilldag_ncf_v3/models/content_neumf_v1/online_calibration.json}"
TASK_EMBEDDING_CACHE="${ROOT}/.runtime/gos_ncf/cache/paper_main_task_embeddings.json"

MAX_GAMES="${MAX_GAMES:-140}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_WORKERS="${MAX_WORKERS:-}"
TASK_INDICES="${TASK_INDICES:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${METHOD//-/_}_${SKILLDAG_PROFILE}_once_${STAMP}}"
RUN_DIR="${RUNTIME}/${EXP_NAME}"

# A queued experiment can be cancelled before it starts without interrupting
# an earlier experiment in the same background manager.
if [ -e "${RUNTIME}/${EXP_NAME}.skip" ]; then
  echo "[skip] ${EXP_NAME}: ${RUNTIME}/${EXP_NAME}.skip exists"
  exit 0
fi

for required in \
  "${ENV_FILE}" "${PYTHON}" "${ALFWORLD_DATA}" "${SKILLS_DIR}" \
  "${SOURCE_GRAPH}" "${SOURCE_GRAPH_CACHE}"; do
  if [ ! -e "${required}" ]; then
    echo "ERROR: missing ${required}" >&2
    exit 2
  fi
done

export AGENT_SKILL_EVOLUTION_ROOT="${ROOT}"
set -a
source "${ENV_FILE}"
set +a

# Keep the selected provider credentials/base URL while allowing a paper
# backbone to be routed through that provider (for example Codex via Yunwu).
if [ -n "${SKILLDAG_AGENT_MODEL_OVERRIDE:-}" ]; then
  export SKILLDAG_MODEL="${SKILLDAG_AGENT_MODEL_OVERRIDE}"
  export SKILLDAG_LLM_MODEL="${SKILLDAG_AGENT_MODEL_OVERRIDE}"
fi

if [ -z "${SKILLDAG_API_KEY:-}" ] || [ -z "${SKILLDAG_EMBEDDING_API_KEY:-}" ]; then
  echo "ERROR: selected profile has an empty chat or embedding API key." >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"

case "${METHOD}" in
  gos|gos-ncf)
    MAX_WORKERS="${MAX_WORKERS:-5}"
    for required in "${GOS_WORKSPACE}" "${SKILL_EMBEDDINGS}"; do
      if [ ! -e "${required}" ]; then
        echo "ERROR: missing ${required}" >&2
        exit 2
      fi
    done
    if [ "${METHOD}" = "gos-ncf" ] && [ ! -e "${NCF_MODEL}" ]; then
      echo "ERROR: missing ${NCF_MODEL}" >&2
      exit 2
    fi

    export API_KEY="${SKILLDAG_API_KEY}"
    export BASE_URL="${SKILLDAG_API_BASE}"
    export OPENAI_API_KEY="${SKILLDAG_EMBEDDING_API_KEY}"
    export OPENAI_BASE_URL="${SKILLDAG_EMBEDDING_BASE}"
    export GOS_EMBEDDING_MODEL="openai/text-embedding-3-large"
    export GOS_EMBEDDING_DIM=3072
    case "${SKILLDAG_PROFILE}" in
      yunwu) AGENT_MODEL="MiniMax-M2.7" ;;
      minimax) AGENT_MODEL="minimax/minimax-m2.7" ;;
      codex) AGENT_MODEL="gpt-5.2-codex" ;;
      *) echo "ERROR: unsupported profile ${SKILLDAG_PROFILE}" >&2; exit 2 ;;
    esac

    GOS_MODE="paper_gos"
    NCF_ARGS=()
    if [ "${METHOD}" = "gos-ncf" ]; then
      GOS_MODE="paper_gos_ncf"
      NCF_ARGS=(
        --skill_graph "${SOURCE_GRAPH}"
        --skill_embeddings "${SKILL_EMBEDDINGS}"
        --ncf_model "${NCF_MODEL}"
        --embedding_cache "${TASK_EMBEDDING_CACHE}"
      )
    fi

    cd "${GOS_PROJECT}"
    export PYTHONPATH="${GOS_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${PYTHON}" evaluation/alfworld_run.py \
      --model "${AGENT_MODEL}" \
      --split dev \
      --use_skill \
      --mode "${GOS_MODE}" \
      --gos_workspace "${GOS_WORKSPACE}" \
      --skills_dir "${SKILLS_DIR}" \
      --max_games "${MAX_GAMES}" \
      --max_workers "${MAX_WORKERS}" \
      --max_steps "${MAX_STEPS}" \
      --exp_name "${EXP_NAME}" \
      "${NCF_ARGS[@]}"
    ;;

  skilldag|skilldag-ncf|skilldag-ncf-v3|skilldag-ncf-v3-search|skilldag-ncf-v3-hybrid)
    MAX_WORKERS="${MAX_WORKERS:-3}"
    WORK_GRAPH="${RUN_DIR}/skillgraph.json"
    # Preserve the evolved graph and embedding cache when resuming an
    # interrupted experiment with the same EXP_NAME. New experiment names
    # still start from the exact frozen source snapshot.
    if [ ! -e "${WORK_GRAPH}" ]; then
      cp "${SOURCE_GRAPH}" "${WORK_GRAPH}"
    fi
    if [ ! -e "${RUN_DIR}/skillgraph.embeddings.json" ]; then
      cp "${SOURCE_GRAPH_CACHE}" "${RUN_DIR}/skillgraph.embeddings.json"
    fi
    if [ "${METHOD}" = "skilldag-ncf" ]; then
      if [ ! -e "${NCF_MODEL}" ]; then
        echo "ERROR: missing ${NCF_MODEL}" >&2
        exit 2
      fi
      export SKILLDAG_NCF_MODEL="${NCF_MODEL}"
      export SKILLDAG_NCF_CANDIDATE_K="${SKILLDAG_NCF_CANDIDATE_K:-12}"
      unset SKILLDAG_NCF_PHASE_MODEL SKILLDAG_NCF_PHASE_CALIBRATION || true
    elif [ "${METHOD}" = "skilldag-ncf-v3" ]; then
      for required in "${NCF_V3_MODEL}" "${NCF_V3_CALIBRATION}"; do
        if [ ! -e "${required}" ]; then
          echo "ERROR: missing ${required}" >&2
          exit 2
        fi
      done
      unset SKILLDAG_NCF_MODEL SKILLDAG_NCF_CANDIDATE_K || true
      export SKILLDAG_NCF_PHASE_MODEL="${NCF_V3_MODEL}"
      export SKILLDAG_NCF_PHASE_CALIBRATION="${NCF_V3_CALIBRATION}"
      export SKILLDAG_FROZEN_MODE=1
    elif [ "${METHOD}" = "skilldag-ncf-v3-search" ]; then
      if [ ! -e "${NCF_V3_MODEL}" ]; then
        echo "ERROR: missing ${NCF_V3_MODEL}" >&2
        exit 2
      fi
      # Clean ablation: keep the original mutable SkillDAG runtime and only
      # plug the V3-trained content NeuMF into every search/search_batch.
      export SKILLDAG_NCF_MODEL="${NCF_V3_MODEL}"
      export SKILLDAG_NCF_CANDIDATE_K="${SKILLDAG_NCF_CANDIDATE_K:-12}"
      unset SKILLDAG_NCF_PHASE_MODEL SKILLDAG_NCF_PHASE_CALIBRATION || true
      unset SKILLDAG_FROZEN_MODE || true
    elif [ "${METHOD}" = "skilldag-ncf-v3-hybrid" ]; then
      for required in "${NCF_V3_MODEL}" "${NCF_V3_CALIBRATION}"; do
        if [ ! -e "${required}" ]; then
          echo "ERROR: missing ${required}" >&2
          exit 2
        fi
      done
      # Hybrid ablation: retain V3 proactive phase planning and also reuse
      # the same content NeuMF for every online search/search_batch. Keep the
      # original SkillDAG graph mutable.
      export SKILLDAG_NCF_MODEL="${NCF_V3_MODEL}"
      export SKILLDAG_NCF_CANDIDATE_K="${SKILLDAG_NCF_CANDIDATE_K:-12}"
      export SKILLDAG_NCF_PHASE_MODEL="${NCF_V3_MODEL}"
      export SKILLDAG_NCF_PHASE_CALIBRATION="${NCF_V3_CALIBRATION}"
      unset SKILLDAG_FROZEN_MODE || true
    else
      unset SKILLDAG_NCF_MODEL SKILLDAG_NCF_CANDIDATE_K || true
      unset SKILLDAG_NCF_PHASE_MODEL SKILLDAG_NCF_PHASE_CALIBRATION || true
      unset SKILLDAG_FROZEN_MODE || true
    fi
    export ALFWORLD_DATA
    export PYTHONPATH="${SKILLDAG_PROJECT}/src:${SKILLDAG_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

    cd "${SKILLDAG_PROJECT}"
    TASK_INDEX_ARGS=()
    if [ -n "${TASK_INDICES}" ]; then
      IFS=',' read -r -a TASK_INDEX_VALUES <<< "${TASK_INDICES}"
      TASK_INDEX_ARGS=(--task_indices "${TASK_INDEX_VALUES[@]}")
    fi
    exec "${PYTHON}" benchmarks/alfworld/run_alfworld.py \
      --model "${SKILLDAG_MODEL}" \
      --skilldag_api_base "${SKILLDAG_API_BASE}" \
      --skills_dir "${SKILLS_DIR}" \
      --skilldag_graph "${WORK_GRAPH}" \
      --split dev \
      --max_games "${MAX_GAMES}" \
      --max_workers "${MAX_WORKERS}" \
      --max_steps "${MAX_STEPS}" \
      --exp_name "${EXP_NAME}" \
      "${TASK_INDEX_ARGS[@]}"
    ;;

  *)
    echo "ERROR: unsupported METHOD: ${METHOD}." >&2
    exit 2
    ;;
esac
