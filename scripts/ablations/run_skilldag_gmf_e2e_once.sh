#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="$ROOT/.runtime/skilldag_ncf_v3/models/content_neumf_v1/gmf.pt"
CALIBRATION="$ROOT/.runtime/ablations/alfworld_ncf_v1/deployed_head_replay/gmf_calibration.json"
EXP_NAME="${EXP_NAME:-skilldag_ncf_v3_hybrid_gmf_openlux_once_20260826}"

# Keep the end-to-end protocol identical to the MLP-only ablation and change
# only the deployed scoring head and its matched calibration parameters.
export CUDA_VISIBLE_DEVICES=""
export SKILLDAG_NCF_V3_MODEL_OVERRIDE="$MODEL"
export SKILLDAG_NCF_V3_CALIBRATION_OVERRIDE="$CALIBRATION"
export MAX_GAMES="${MAX_GAMES:-140}"
export MAX_STEPS="${MAX_STEPS:-30}"
export MAX_WORKERS="${MAX_WORKERS:-1}"
export EXP_NAME

exec bash "$ROOT/scripts/run_paper_main_alfworld_once.sh" \
  skilldag-ncf-v3-hybrid yunwu
