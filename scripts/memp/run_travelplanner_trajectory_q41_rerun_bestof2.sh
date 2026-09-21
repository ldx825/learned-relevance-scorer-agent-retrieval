#!/usr/bin/env bash
# Symmetric best-of-2 rerun: each arm re-runs only the tasks it LOST in round 1.
# Round-1 results stay untouched in <BASE>/<arm>/results; round-2 goes to
# <BASE>/round2/<arm>/results.  Final metrics take per-task max of the two
# rounds under the SAME rule for both arms.
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
SLICES="$ROOT/private/memp/travelplanner/v25_trajectory_q41_rerun_slices"

PIDS=()
for arm in neumf cosine; do
  for part in 0 1 2 3; do
    "$PYTHON" -u "$ROOT/scripts/memp/run_travelplanner_script_query_cosine_test.py" \
      --split validation --start 0 --count 180 \
      --retrieval "$SLICES/${arm}_${part}.json" \
      --output-dir "$BASE/round2/$arm/results" --postprocess \
      >"$BASE/round2/$arm/slice_${part}.log" 2>&1 &
    PIDS+=("$!")
  done
done

STATUS=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || STATUS=1
done
exit "$STATUS"
