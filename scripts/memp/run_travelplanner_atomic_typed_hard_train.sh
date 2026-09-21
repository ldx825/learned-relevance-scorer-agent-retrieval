#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EMBED_PYTHON="$ROOT/.envs/memp-travelplanner/bin/python"
TRAIN_PYTHON="$ROOT/.envs/gos-ncf-eval/bin/python"
DATA="$ROOT/data/memp_ncf"
SOURCE_DATA="$DATA/travelplanner_atomic_operation_neumf_data_v1"
TARGET_DATA="$DATA/travelplanner_atomic_operation_neumf_data_v2_typed_hard"
TARGET_MODEL="$DATA/travelplanner_atomic_operation_neumf_model_v2_typed_hard"

export AGENT_SKILL_EVOLUTION_ROOT="$ROOT"

# The text set is unchanged, so reuse the audited embedding cache.  This makes
# rebuilding the typed-hard dataset entirely offline and free of API cost.
mkdir -p "$TARGET_DATA/embedding_cache"
cp -a "$SOURCE_DATA/embedding_cache/." "$TARGET_DATA/embedding_cache/"

"$EMBED_PYTHON" "$ROOT/scripts/memp/prepare_travelplanner_atomic_operation_neumf_data.py" \
  --subgoals "$DATA/travelplanner_atomic_memory_candidates_v4/task_subgoals.jsonl" \
  --operations "$DATA/travelplanner_atomic_memory_candidates_v4/memory_operations.jsonl" \
  --train-pairs "$DATA/travelplanner_atomic_typed_hard_labels_v2/train_pairs_resolved.jsonl" \
  --dev-pairs "$DATA/travelplanner_atomic_typed_hard_labels_v2/internal_dev_pairs_resolved.jsonl" \
  --output-dir "$TARGET_DATA" \
  --model text-embedding-3-small \
  --batch-size 64

"$TRAIN_PYTHON" "$ROOT/scripts/memp/train_memory_neumf.py" \
  --arrays "$TARGET_DATA/arrays.npz" \
  --output "$TARGET_MODEL" \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --patience 5 \
  --device cpu

"$TRAIN_PYTHON" "$ROOT/scripts/memp/evaluate_travelplanner_atomic_operation_neumf.py" \
  --arrays "$TARGET_DATA/arrays.npz" \
  --checkpoint "$TARGET_MODEL/neumf.pt" \
  --output "$TARGET_MODEL/paired_eval.json"
