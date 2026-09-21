#!/usr/bin/env bash
# Resume-style fix run for the tasks that errored out in the main 180-task run.
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
BASE="$ROOT/private/memp/travelplanner/v25_proceduralization_q84_validation180_paired_run"
FIX="$ROOT/private/memp/travelplanner/v25_proceduralization_q84_fix_slices"

PIDS=()
for arm in cosine neumf; do
  "$PYTHON" -u "$ROOT/scripts/memp/run_travelplanner_script_query_cosine_test.py" \
    --split validation --start 0 --count 180 \
    --retrieval "$FIX/${arm}_fix.json" \
    --output-dir "$BASE/$arm/results" --postprocess \
    >"$BASE/$arm/fix.log" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || STATUS=1
done
exit "$STATUS"
