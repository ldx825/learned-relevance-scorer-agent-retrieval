#!/usr/bin/env bash
# run_alfworld.sh — reproduce the ALFWorld arm of the SkillDAG paper.
#
# Wraps the bundled ALFWorld SkillDAG runner, pulling secrets from .env and
# using sensible defaults for the headline 140-task split.
#
# Override defaults via env: MAX_GAMES=10 bash scripts/run_alfworld.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

# ─── Load secrets ───
if [ ! -f "${ENV_FILE}" ]; then
  echo "ERROR: ${ENV_FILE} missing. Run scripts/prepare_env.sh, then fill in API keys." >&2
  exit 2
fi
set -a; source "${ENV_FILE}"; set +a
# ─── Backbone routing ─────────────────────────────────────────────────────
case "${SKILLDAG_BACKBONE:-openai}" in
  openrouter)
    SKILLDAG_API_KEY="${OPENROUTER_API_KEY}"
    SKILLDAG_API_BASE="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    SKILLDAG_EMBEDDING_API_KEY="${OPENROUTER_API_KEY}"
    SKILLDAG_EMBEDDING_BASE="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    SKILLDAG_EMBEDDING_MODEL="${OPENROUTER_EMBEDDING_MODEL:-openai/text-embedding-3-large}"
    echo "[backbone] OpenRouter: api=${SKILLDAG_API_BASE} emb=${SKILLDAG_EMBEDDING_BASE}"
    ;;
  openai|*)
    SKILLDAG_API_KEY="${OPENAI_API_KEY}"
    ;;
esac
export SKILLDAG_EMBEDDING_API_KEY SKILLDAG_EMBEDDING_BASE SKILLDAG_EMBEDDING_MODEL


# ─── Required state from setup.sh ───
PYTHON_BIN="${PYTHON:-python3.11}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: python not on PATH (or PYTHON=${PYTHON_BIN} is invalid)." >&2
  echo "       Run scripts/prepare_env.sh to detect python3.11/PATH." >&2
  exit 2
fi
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "ERROR: PYTHON=${PYTHON_BIN} must be Python >=3.10; python3.11 is recommended." >&2
  exit 2
fi
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
missing = [name for name in ("alfworld", "litellm", "retry", "yaml") if importlib.util.find_spec(name) is None]
if missing:
    print("missing: " + ", ".join(missing))
    raise SystemExit(1)
PY
then
  echo "ERROR: ALFWorld dependencies missing. Run: ${PYTHON_BIN} -m pip install -e '.[repro,alfworld]'" >&2
  exit 2
fi

# ─── Defaults (override via env) ───
SKILLS_DIR="${SKILLS_DIR:-${REPO_ROOT}/data/alfworld_skills}"
GRAPH_PATH="${GRAPH_PATH:-${REPO_ROOT}/data/skilldag_graphs/skillgraph_alfworld.json}"
ALFWORLD_DATA="${ALFWORLD_DATA:-${REPO_ROOT}/data/alfworld/data}"
export ALFWORLD_DATA
MAX_GAMES="${MAX_GAMES:-140}"
MAX_WORKERS="${SKILLDAG_WORKERS:-3}"
MAX_STEPS="${MAX_STEPS:-30}"
SPLIT="${SPLIT:-dev}"

# ─── Sanity checks ───
for path in "${SKILLS_DIR}" "${GRAPH_PATH}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required input missing: ${path}" >&2
    echo "       (re-run scripts/setup.sh, or initialize a graph via 'skilldag initialize-graph')" >&2
    exit 2
  fi
done
if [ ! -d "${ALFWORLD_DATA}" ]; then
  echo "ERROR: ALFWORLD_DATA directory missing: ${ALFWORLD_DATA}" >&2
  echo "       Install/download ALFWorld data, or update ALFWORLD_DATA in .env." >&2
  exit 2
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/validate_graph_inputs.py" \
  --skills-dir "${SKILLS_DIR}" \
  --graph-path "${GRAPH_PATH}" \
  --embedding-model "${SKILLDAG_EMBEDDING_MODEL:-}" \
  --require-cache \
  --strict-cache-model

EXP_NAME="${EXP_NAME:-skilldag_alfworld_$(date +%Y%m%d_%H%M%S)}"

echo "=== ALFWorld SkillDAG run ==="
echo "  exp_name   : ${EXP_NAME}"
echo "  backbone   : ${SKILLDAG_BACKBONE:-openai}"
echo "  model      : ${SKILLDAG_MODEL}"
echo "  api_base   : ${SKILLDAG_API_BASE}"
echo "  data       : ${ALFWORLD_DATA}"
echo "  skills_dir : ${SKILLS_DIR}"
echo "  graph      : ${GRAPH_PATH}"
echo "  max_games  : ${MAX_GAMES}    workers: ${MAX_WORKERS}    steps: ${MAX_STEPS}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" benchmarks/alfworld/run_alfworld.py \
  --model "${SKILLDAG_MODEL}" \
  --skilldag_api_base "${SKILLDAG_API_BASE}" \
  --skills_dir "${SKILLS_DIR}" \
  --skilldag_graph "${GRAPH_PATH}" \
  --split "${SPLIT}" \
  --max_games "${MAX_GAMES}" \
  --max_workers "${MAX_WORKERS}" \
  --max_steps "${MAX_STEPS}" \
  --exp_name "${EXP_NAME}"
