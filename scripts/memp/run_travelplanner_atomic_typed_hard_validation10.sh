#!/usr/bin/env bash
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
RESULTS="$ROOT/.runtime/memp/travelplanner/atomic_typed_hard_validation10/results"
RETRIEVAL="$ROOT/data/memp_ncf/travelplanner_atomic_dynamic_validation10_v2_typed_hard/retrieval.json"

"$PYTHON" -u "$ROOT/scripts/memp/run_travelplanner_script_query_cosine_test.py" \
  --split validation --start 0 --count 10 \
  --retrieval "$RETRIEVAL" --output-dir "$RESULTS" --postprocess

"$PYTHON" "$ROOT/scripts/memp/evaluate_travelplanner_validation_slice.py" \
  --limit 10 --results-dir "$RESULTS" \
  --output "$ROOT/.runtime/memp/travelplanner/atomic_typed_hard_validation10/evaluation.json"
