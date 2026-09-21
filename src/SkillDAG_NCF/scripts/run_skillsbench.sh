#!/usr/bin/env bash
# run_skillsbench.sh — reproduce the SkillsBench arm of the SkillDAG paper.
#
# Pipeline:
#   1. (optional) Generate tasks_skilldag_full_<SCALE>/ via the bundled
#      SkillsBench generator if it doesn't already exist. Each per-task
#      directory ships a bind-mounted skillgraph.json under environment/skilldag/.
#   2. Render configs/skillsbench/skilldag.yaml via envsubst.
#   3. Invoke `harbor run -c <rendered yaml>`.
#   4. Score via analysis/score_skillsbench_gos.py.
#
# Override scale/workers/etc. through .env; see .env.example.
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
    CODEX_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    SKILLDAG_EMBEDDING_API_KEY="${OPENROUTER_API_KEY}"
    SKILLDAG_EMBEDDING_BASE="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    SKILLDAG_EMBEDDING_MODEL="${OPENROUTER_EMBEDDING_MODEL:-openai/text-embedding-3-large}"
    echo "[backbone] OpenRouter: codex=${CODEX_BASE_URL} emb=${SKILLDAG_EMBEDDING_BASE}"
    ;;
  openai|*)
    # Codex CLI default: use built-in OpenAI provider (CODEX_BASE_URL empty)
    ;;
esac
export SKILLDAG_EMBEDDING_API_KEY SKILLDAG_EMBEDDING_BASE SKILLDAG_EMBEDDING_MODEL


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

if ! command -v harbor >/dev/null 2>&1; then
  echo "ERROR: 'harbor' binary not on PATH. Install the SkillsBench framework first" >&2
  echo "       (see docs/reproducing.md §Harbor install), then rerun scripts/prepare_env.sh." >&2
  exit 2
fi
if ! command -v envsubst >/dev/null 2>&1; then
  echo "ERROR: envsubst missing (install gettext)." >&2
  exit 2
fi
# ─── Defaults (overridable via .env or shell env) ───
export SKILLDAG_SCALE="${SKILLDAG_SCALE:-200}"
export SKILLDAG_METHOD="${SKILLDAG_METHOD:-skilldag}"
case "${SKILLDAG_METHOD}" in
  skilldag|skilldag-ncf-hybrid) ;;
  *) echo "ERROR: SKILLDAG_METHOD must be skilldag or skilldag-ncf-hybrid" >&2; exit 2 ;;
esac
export SKILLDAG_WORKERS="${SKILLDAG_WORKERS:-3}"
export SKILLDAG_CODEX_MODEL="${SKILLDAG_CODEX_MODEL:-gpt-5.2-codex}"
export SKILLDAG_CODEX_REASONING_EFFORT="${SKILLDAG_CODEX_REASONING_EFFORT:-high}"
export SKILLDAG_RUN_STAMP="${SKILLDAG_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SKILLDAG_JOB_NAME="${SKILLDAG_JOB_NAME:-${SKILLDAG_METHOD}-${SKILLDAG_SCALE}-w${SKILLDAG_WORKERS}-${SKILLDAG_RUN_STAMP}}"
export SKILLDAG_JOBS_DIR="${SKILLDAG_JOBS_DIR:-${REPO_ROOT}/results/skillsbench/${SKILLDAG_JOB_NAME}}"
export SKILLDAG_TASKS_DIR="${SKILLDAG_TASKS_DIR:-${REPO_ROOT}/results/skillsbench_tasks/tasks_${SKILLDAG_METHOD}_full_${SKILLDAG_SCALE}}"

NCF_GENERATOR_ARGS=()
if [ "${SKILLDAG_METHOD}" = "skilldag-ncf-hybrid" ]; then
  : "${SKILLDAG_NCF_BUNDLE:?SKILLDAG_NCF_BUNDLE is required for hybrid mode}"
  : "${SKILLDAG_NCF_PLAN_MANIFEST:?SKILLDAG_NCF_PLAN_MANIFEST is required for hybrid mode}"
  [ -f "${SKILLDAG_NCF_BUNDLE}" ] || { echo "ERROR: missing ${SKILLDAG_NCF_BUNDLE}" >&2; exit 2; }
  [ -f "${SKILLDAG_NCF_PLAN_MANIFEST}" ] || { echo "ERROR: missing ${SKILLDAG_NCF_PLAN_MANIFEST}" >&2; exit 2; }
  export SKILLDAG_NCF_MODEL="/var/lib/skilldag/runtime/ncf_bundle.json"
  # ALFWorld has only 37 skills and uses Top-12 recall.  SkillsBench-1000
  # needs a wider shallow-retrieval boundary before the same NeuMF reranker.
  export SKILLDAG_NCF_CANDIDATE_K="${SKILLDAG_NCF_CANDIDATE_K:-100}"
  export SKILLDAG_NCF_EXPECTED_POOL_SIZE="${SKILLDAG_SCALE}"
  NCF_GENERATOR_ARGS=(
    --ncf-bundle "${SKILLDAG_NCF_BUNDLE}"
    --ncf-plan-manifest "${SKILLDAG_NCF_PLAN_MANIFEST}"
  )
else
  unset SKILLDAG_NCF_MODEL SKILLDAG_NCF_CANDIDATE_K SKILLDAG_NCF_EXPECTED_POOL_SIZE || true
fi
export SKILLDAG_NCF_MODEL SKILLDAG_NCF_CANDIDATE_K SKILLDAG_NCF_EXPECTED_POOL_SIZE

mkdir -p "${SKILLDAG_JOBS_DIR}"

# ─── 0. Validate reproducibility inputs ───
SKILLS_DIR_HOST="${SKILLS_DIR_HOST:-${REPO_ROOT}/data/skillsets/skills_${SKILLDAG_SCALE}}"
if [ -z "$(find "${SKILLS_DIR_HOST}" -mindepth 2 -maxdepth 2 -name SKILL.md -print -quit 2>/dev/null)" ] \
   && [ -d "${SKILLS_DIR_HOST}/skills_${SKILLDAG_SCALE}" ]; then
  SKILLS_DIR_HOST="${SKILLS_DIR_HOST}/skills_${SKILLDAG_SCALE}"
fi
GRAPH_PATH_HOST="${REPO_ROOT}/data/skilldag_graphs/skillgraph_${SKILLDAG_SCALE}.json"
[ -d "${SKILLS_DIR_HOST}" ] || { echo "ERROR: ${SKILLS_DIR_HOST} missing (run scripts/setup.sh)." >&2; exit 2; }
[ -f "${GRAPH_PATH_HOST}" ] || { echo "ERROR: ${GRAPH_PATH_HOST} missing (initialize graph or fetch from HF)." >&2; exit 2; }
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/validate_graph_inputs.py" \
  --skills-dir "${SKILLS_DIR_HOST}" \
  --graph-path "${GRAPH_PATH_HOST}" \
  --embedding-model "${SKILLDAG_EMBEDDING_MODEL:-}" \
  --require-cache \
  --strict-cache-model

# ─── 1. Generate SkillDAG-wrapped task variants if missing ───
if [ "${SKILLDAG_METHOD}" = "skilldag-ncf-hybrid" ] \
   || [ ! -d "${SKILLDAG_TASKS_DIR}" ] \
   || [ -z "$(ls -A "${SKILLDAG_TASKS_DIR}" 2>/dev/null)" ]; then
  echo "[run] generating ${SKILLDAG_TASKS_DIR} …"
  SKILLS_ROOT="${SKILLS_DIR_HOST}"
  GRAPH_PATH="${REPO_ROOT}/data/skilldag_graphs/skillgraph_${SKILLDAG_SCALE}.json"
  TASKS_ROOT="${SKILLSBENCH_TASKS_ROOT:-${REPO_ROOT}/data/tasks/tasks}"
  for path in "${SKILLS_ROOT}" "${GRAPH_PATH}"; do
    if [ ! -e "${path}" ]; then
      echo "ERROR: required input missing: ${path}" >&2
      echo "       (re-run scripts/setup.sh, or initialize via 'skilldag initialize-graph')" >&2
      exit 2
    fi
  done
  [ -d "${TASKS_ROOT}" ] || { echo "ERROR: ${TASKS_ROOT} missing (run scripts/setup.sh)." >&2; exit 2; }
  ( cd "${REPO_ROOT}" && "${PYTHON_BIN}" benchmarks/skillsbench/skilldag_benchmark.py \
      --tasks-root "${TASKS_ROOT}" \
      --skills-root "${SKILLS_ROOT}" \
      --skillgraph-path "${GRAPH_PATH}" \
      --skilldag-package-root "${REPO_ROOT}" \
      --output-root "${SKILLDAG_TASKS_DIR}" \
      "${NCF_GENERATOR_ARGS[@]}" )
fi

# ─── 2. Render YAML config from template ───
RENDERED_YAML="${SKILLDAG_JOBS_DIR}/config.yaml"
envsubst < "${REPO_ROOT}/configs/skillsbench/skilldag.yaml" > "${RENDERED_YAML}"
echo "[run] rendered config → ${RENDERED_YAML}"

# ─── 3. Run via Harbor ───
LOG="${SKILLDAG_JOBS_DIR}/run.log"
echo "[run] harbor run -c ${RENDERED_YAML}  (log: ${LOG})"
harbor run -c "${RENDERED_YAML}" 2>&1 | tee "${LOG}"

# ─── 4. Score ───
TOTAL=$(ls -d "${SKILLDAG_TASKS_DIR}"/*/ 2>/dev/null | wc -l | tr -d ' ')
"${PYTHON_BIN}" "${REPO_ROOT}/analysis/score_skillsbench_gos.py" \
  "${SKILLDAG_JOBS_DIR}" --job-name "${SKILLDAG_JOB_NAME}" \
  --total "${TOTAL}" | tee -a "${LOG}"
