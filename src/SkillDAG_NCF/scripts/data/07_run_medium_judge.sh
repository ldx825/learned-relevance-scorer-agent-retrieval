#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA="${ROOT}/.runtime/skilldag_ncf/data"
PROJECT="${ROOT}/src/SkillDAG_NCF"
export PYTHONPATH="${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${ROOT}/.envs/skilldag-debug/bin/python" \
  "${ROOT}/src/SkillDAG_NCF/scripts/data/07_prepare_medium_judge_selection.py"

exec "${ROOT}/src/SkillDAG_NCF/scripts/data/05_run_judge_pilot.sh" yunwu \
  --candidates-path "${DATA}/intermediate/train_combined_candidates.jsonl" \
  --selection-path "${DATA}/reports/judge_medium_v1_selection.json" \
  --collection-name medium_v1 \
  --fallback-cache-dir "${DATA}/judge/pilot/responses" \
  "$@"
