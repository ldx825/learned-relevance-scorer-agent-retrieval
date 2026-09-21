#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="$ROOT/.runtime/skilldag_ncf_v3/models/content_neumf_v1/mlp.pt"
CALIBRATION="$ROOT/.runtime/ablations/alfworld_ncf_v1/deployed_head_replay/mlp_calibration.json"
EXP_NAME="${EXP_NAME:-skilldag_ncf_v3_hybrid_mlp_openlux_once_20260825}"

# Agent evaluation and the lightweight scorer are CPU-only in this ablation.
export CUDA_VISIBLE_DEVICES=""
export SKILLDAG_NCF_V3_MODEL_OVERRIDE="$MODEL"
export SKILLDAG_NCF_V3_CALIBRATION_OVERRIDE="$CALIBRATION"
export MAX_GAMES="${MAX_GAMES:-140}"
export MAX_STEPS="${MAX_STEPS:-30}"
export MAX_WORKERS="${MAX_WORKERS:-3}"
export EXP_NAME

exec bash "$ROOT/scripts/run_paper_main_alfworld_once.sh" \
  skilldag-ncf-v3-hybrid yunwu
