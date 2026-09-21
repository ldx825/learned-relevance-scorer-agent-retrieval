#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT/.runtime/skilldag/skilldag.env"
EMBED_PYTHON="$ROOT/.envs/memp-travelplanner/bin/python"
TRAIN_PYTHON="$ROOT/.envs/gos-ncf-eval/bin/python"
DATA="$ROOT/data/memp_ncf"
SOURCE_DATA="$DATA/travelplanner_atomic_operation_neumf_data_v1"
TARGET_DATA="$DATA/travelplanner_atomic_phase_neumf_data_v1"
TARGET_MODEL="$DATA/travelplanner_atomic_phase_neumf_model_v1"

export AGENT_SKILL_EVOLUTION_ROOT="$ROOT"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Reuse all519 operation embeddings.  Only the1,980 new train-derived phase
# queries can trigger embedding API requests, and their cache is resumable.
mkdir -p "$TARGET_DATA/embedding_cache"
cp -a "$SOURCE_DATA/embedding_cache/operations.json" "$TARGET_DATA/embedding_cache/operations.json"

"$EMBED_PYTHON" "$ROOT/scripts/memp/prepare_travelplanner_atomic_operation_neumf_data.py" \
  --subgoals "$DATA/travelplanner_atomic_phase_pairs_v1/phase_queries.jsonl" \
  --operations "$DATA/travelplanner_atomic_memory_candidates_v4/memory_operations.jsonl" \
  --train-pairs "$DATA/travelplanner_atomic_phase_pairs_v1/train_pairs_resolved.jsonl" \
  --dev-pairs "$DATA/travelplanner_atomic_phase_pairs_v1/internal_dev_pairs_resolved.jsonl" \
  --output-dir "$TARGET_DATA" \
  --model text-embedding-3-small \
  --batch-size 64

"$TRAIN_PYTHON" "$ROOT/scripts/memp/train_memory_neumf.py" \
  --arrays "$TARGET_DATA/arrays.npz" \
  --output "$TARGET_MODEL" \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --patience 5 \
  --resume \
  --device cpu

"$TRAIN_PYTHON" "$ROOT/scripts/memp/evaluate_travelplanner_atomic_operation_neumf.py" \
  --arrays "$TARGET_DATA/arrays.npz" \
  --checkpoint "$TARGET_MODEL/neumf.pt" \
  --output "$TARGET_MODEL/paired_eval.json"
