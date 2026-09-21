#!/usr/bin/env bash
# Symmetric best-of-2 rerun for the 84-pool proceduralization run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENT_SKILL_EVOLUTION_ROOT="$ROOT"
set -a
source "$ROOT/.runtime/skilldag/skilldag.env"
set +a
export MEMP_API_KEY="${MEMP_API_KEY:-${SKILLDAG_LLM_API_KEY}}"
export MEMP_API_BASE="${MEMP_API_BASE:-${SKILLDAG_LLM_BASE}}"
export MEMP_CHAT_MODEL="gpt-4o"
PYTHON="$ROOT/.envs/memp-travelplanner/bin/python"
BASE="$ROOT/private/memp/travelplanner/v25_proceduralization_q84_validation180_paired_run"
RERUN="$ROOT/private/memp/travelplanner/v25_proceduralization_q84_rerun_slices"
mkdir -p "$BASE/round2/cosine/results" "$BASE/round2/neumf/results"
PIDS=()
for arm in cosine neumf; do
  for part in 0 1 2 3; do
    [ -f "$RERUN/${arm}_${part}.json" ] || continue
    "$PYTHON" -u "$ROOT/scripts/memp/run_travelplanner_script_query_cosine_test.py" \
      --split validation --start 0 --count 180 \
      --retrieval "$RERUN/${arm}_${part}.json" \
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
