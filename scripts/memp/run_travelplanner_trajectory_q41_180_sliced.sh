#!/usr/bin/env bash
# Resume the 180-task Trajectory paired run (41-doc quality pool) with 6
# parallel worker processes (2 arms x 3 slices).  Each worker shares the
# results directory; the runner skips saved tasks, so parallelism never
# re-pays finished work.  Pilot tasks 10..19 can be re-run inside the same
# directory for a uniform 180-task protocol.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENT_SKILL_EVOLUTION_ROOT="$ROOT"
set -a
# shellcheck disable=SC1090
source "$ROOT/.runtime/skilldag/skilldag.env"
set +a
export MEMP_API_KEY="${MEMP_API_KEY:-${SKILLDAG_LLM_API_KEY}}"
export MEMP_API_BASE="${MEMP_API_BASE:-${SKILLDAG_LLM_BASE}}"
export MEMP_CHAT_MODEL="gpt-4o"

PYTHON="$ROOT/.envs/memp-travelplanner/bin/python"
BASE="$ROOT/private/memp/travelplanner/v25_trajectory_q41_validation180_paired_run"
SLICES="$ROOT/private/memp/travelplanner/v25_trajectory_q41_180_retrieval_slices"

PIDS=()
for arm in cosine neumf; do
  for part in a b c; do
    "$PYTHON" -u "$ROOT/scripts/memp/run_travelplanner_script_query_cosine_test.py" \
      --split validation --start 0 --count 180 \
      --retrieval "$SLICES/${arm}_${part}.json" \
      --output-dir "$BASE/$arm/results" --postprocess \
      >"$BASE/$arm/slice_${part}.log" 2>&1 &
    PIDS+=("$!")
  done
done

STATUS=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || STATUS=1
done
exit "$STATUS"
