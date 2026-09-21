#!/usr/bin/env bash
# run_alfworld_traintest.sh — 3-phase ALFWorld protocol used in the paper.
#
#   Stage 1 (TRAIN)         : run on train split with online graph edits enabled
#                              → produces "edited graph"
#   Stage 2 (TEST_INITIAL)  : run on dev split with a *fresh copy* of the clean
#                              starting graph (baseline; isolates graph effect)
#   Stage 3 (TEST_EDITED)   : run on dev split with a copy of the train-edited
#                              graph (the actual contribution)
#
# Graph file flow:
#
#   CLEAN ──► train_working  ── (agent edits in-place) ──► graph_after_train
#                                                                │
#                                                                ▼
#                                            test_edited_working (copy)
#
#   CLEAN ──► test_initial_working (fresh copy)
#
# Usage:
#   bash scripts/run_alfworld_traintest.sh                       # full 420/140/140
#   TRAIN_MAX=3 TEST_INIT_MAX=2 TEST_EDITED_MAX=2 \
#     bash scripts/run_alfworld_traintest.sh                     # smoke
#
# Env (in addition to .env):
#   CLEAN_GRAPH      starting graph file (defaults to data/skilldag_graphs/skillgraph_alfworld.json)
#   TRAIN_MAX        # train tasks (default 420)
#   TEST_INIT_MAX    # test_initial tasks (default 140)
#   TEST_EDITED_MAX  # test_edited tasks (default 140)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
[ -f "${ENV_FILE}" ] || { echo "ERROR: ${ENV_FILE} missing. Run scripts/prepare_env.sh, then fill in API keys." >&2; exit 2; }
set -a; source "${ENV_FILE}"; set +a
# ─── Backbone routing ─────────────────────────────────────────────────────
case "${SKILLDAG_BACKBONE:-openai}" in
  openrouter)
    SKILLDAG_API_KEY="${OPENROUTER_API_KEY}"
    SKILLDAG_API_BASE="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    SKILLDAG_EMBEDDING_API_KEY="${OPENROUTER_API_KEY}"
    SKILLDAG_EMBEDDING_BASE="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    SKILLDAG_EMBEDDING_MODEL="${OPENROUTER_EMBEDDING_MODEL:-openai/text-embedding-3-large}"
    ;;
  openai|*)
    SKILLDAG_API_KEY="${OPENAI_API_KEY}"
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

CLEAN_GRAPH="${CLEAN_GRAPH:-${REPO_ROOT}/data/skilldag_graphs/skillgraph_alfworld.json}"
[ -f "${CLEAN_GRAPH}" ] || { echo "ERROR: clean graph missing: ${CLEAN_GRAPH}" >&2; exit 2; }

TRAIN_MAX="${TRAIN_MAX:-420}"
TEST_INIT_MAX="${TEST_INIT_MAX:-140}"
TEST_EDITED_MAX="${TEST_EDITED_MAX:-140}"

TS="$(date +%Y%m%d_%H%M%S)"
WORK="${REPO_ROOT}/results/alfworld/traintest_${TS}"
mkdir -p "${WORK}"

CLEAN="${WORK}/clean_graph.json"
TRAIN_GRAPH="${WORK}/train_working.json"
GRAPH_AFTER_TRAIN="${WORK}/graph_after_train.json"
TEST_INIT_GRAPH="${WORK}/test_initial_working.json"
TEST_EDITED_GRAPH="${WORK}/test_edited_working.json"
SUMMARY="${WORK}/summary.txt"

cp -p "${CLEAN_GRAPH}" "${CLEAN}"
cp -p "${CLEAN_GRAPH}" "${TRAIN_GRAPH}"

# ─── Helpers ───────────────────────────────────────────────────────
count_edges() { "${PYTHON_BIN}" -c "import json; print(len(json.load(open('$1')).get('edges',[])))"; }
count_online() { "${PYTHON_BIN}" -c "import json; print(sum(1 for h in json.load(open('$1')).get('history',[]) if h.get('online')))"; }
tally() {
  local exp="$1" rd="${REPO_ROOT}/results/alfworld/$1"
  local p=0 f=0
  if [ -d "${rd}" ]; then
    for fp in "${rd}"/idx_*.json; do
      [ -f "${fp}" ] || continue
      local pass_v=$("${PYTHON_BIN}" -c "import json; print(int(json.load(open('${fp}')).get('reward')==True))" 2>/dev/null || echo 0)
      [ "${pass_v}" = "1" ] && p=$((p+1)) || f=$((f+1))
    done
  fi
  echo "pass=${p} fail=${f}"
}

run_phase() {
  local stage="$1" exp="$2" graph="$3" split="$4" max_games="$5" logf="$6"
  echo "=== [$(date)] ${stage} START  exp=${exp}  split=${split}  max_games=${max_games}  graph=${graph} ===" | tee "${logf}"
  echo "  pre-edges=$(count_edges "${graph}")  pre-online=$(count_online "${graph}")" | tee -a "${logf}"

  local frozen_flag=""
  # test_initial and test_edited use frozen mode (disable online graph edits per paper protocol)
  if [ "${stage}" = "TEST_INITIAL" ] || [ "${stage}" = "TEST_EDITED" ]; then
    frozen_flag="SKILLDAG_FROZEN_MODE=1"
  fi

  ${frozen_flag} EXP_NAME="${exp}" SPLIT="${split}" MAX_GAMES="${max_games}" \
    GRAPH_PATH="${graph}" \
    bash "${REPO_ROOT}/scripts/run_alfworld.sh" 2>&1 | tee -a "${logf}"
  local rc=${PIPESTATUS[0]}

  echo "=== [$(date)] ${stage} DONE rc=${rc}  $(tally "${exp}")  post-edges=$(count_edges "${graph}")  post-online=$(count_online "${graph}") ===" | tee -a "${logf}"
  return ${rc}
}

# ─── Stage 1: TRAIN ─────────────────────────────────────────────────
EXP_TRAIN="traintest_${TS}_train"
run_phase "TRAIN" "${EXP_TRAIN}" "${TRAIN_GRAPH}" "train" "${TRAIN_MAX}" "${WORK}/train.log"
cp -p "${TRAIN_GRAPH}" "${GRAPH_AFTER_TRAIN}"

# ─── Stage 2: TEST_INITIAL (baseline graph) ────────────────────────
EXP_TEST_INIT="traintest_${TS}_test_initial"
cp -p "${CLEAN}" "${TEST_INIT_GRAPH}"
run_phase "TEST_INITIAL" "${EXP_TEST_INIT}" "${TEST_INIT_GRAPH}" "dev" "${TEST_INIT_MAX}" "${WORK}/test_initial.log"

# ─── Stage 3: TEST_EDITED (train-edited graph) ─────────────────────
EXP_TEST_EDITED="traintest_${TS}_test_edited"
cp -p "${GRAPH_AFTER_TRAIN}" "${TEST_EDITED_GRAPH}"
run_phase "TEST_EDITED" "${EXP_TEST_EDITED}" "${TEST_EDITED_GRAPH}" "dev" "${TEST_EDITED_MAX}" "${WORK}/test_edited.log"

# ─── Summary ────────────────────────────────────────────────────────
{
  echo "=================================================================="
  echo " ALFWorld 3-phase run — ${TS}"
  echo "=================================================================="
  echo " Initial graph: ${CLEAN_GRAPH}"
  echo "                edges=$(count_edges "${CLEAN}")  online=$(count_online "${CLEAN}")"
  echo
  echo " Stage 1 TRAIN          (${TRAIN_MAX} task)  : $(tally "${EXP_TRAIN}")"
  echo "   graph: $(count_edges "${CLEAN}") → $(count_edges "${GRAPH_AFTER_TRAIN}") edges  (online: 0 → $(count_online "${GRAPH_AFTER_TRAIN}"))"
  echo " Stage 2 TEST_INITIAL   (${TEST_INIT_MAX} task)   : $(tally "${EXP_TEST_INIT}")"
  echo " Stage 3 TEST_EDITED    (${TEST_EDITED_MAX} task) : $(tally "${EXP_TEST_EDITED}")"
  echo
  echo " Logs in ${WORK}"
  echo "=================================================================="
} | tee "${SUMMARY}"
