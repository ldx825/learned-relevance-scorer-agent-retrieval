#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.envs/gos-ncf-eval/bin/python"
TRAIN="$ROOT/scripts/ablations/train_alfworld_ncf_ablation.py"
RUNTIME="$ROOT/.runtime/ablations/alfworld_ncf_v1"
SKILL_ARRAYS="$ROOT/data/alfworld_task_skill/task_skill_v3_phase/model_data/content_neumf_v1/arrays.npz"
MEMORY_ARRAYS="$ROOT/.runtime/memp_ncf/model_data/query_only_neumf_v17_balanced_dual_coverage/arrays.npz"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_arm() {
  local domain="$1"
  local variant="$2"
  local arrays="$3"
  local output="$RUNTIME/training/$domain/2026/$variant"
  "$PYTHON" "$TRAIN" \
    --domain "$domain" \
    --variant "$variant" \
    --arrays "$arrays" \
    --output "$output" \
    --seed 2026 \
    --pretrain-epochs 20 \
    --finetune-epochs 20 \
    --patience 5 \
    --device cpu \
    --resume
}

cd "$ROOT"

run_arm skill linear "$SKILL_ARRAYS"
run_arm skill binary "$RUNTIME/derived_arrays/skill/2026/binary/arrays.npz"
run_arm skill no_pairwise "$SKILL_ARRAYS"
run_arm skill random_negative "$RUNTIME/derived_arrays/skill/2026/random_negative/arrays.npz"

run_arm memory linear "$MEMORY_ARRAYS"
run_arm memory binary "$RUNTIME/derived_arrays/memory/2026/binary/arrays.npz"
run_arm memory no_pairwise "$MEMORY_ARRAYS"
run_arm memory random_negative "$RUNTIME/derived_arrays/memory/2026/random_negative/arrays.npz"
