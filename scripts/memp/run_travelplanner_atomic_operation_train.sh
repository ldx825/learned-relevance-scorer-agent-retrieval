#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT/.runtime/skilldag/skilldag.env"
EMBED_PYTHON="$ROOT/.envs/memp-travelplanner/bin/python"
TRAIN_PYTHON="$ROOT/.envs/gos-ncf-eval/bin/python"
DATA="$ROOT/data/memp_ncf"

export AGENT_SKILL_EVOLUTION_ROOT="$ROOT"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

"$EMBED_PYTHON" "$ROOT/scripts/memp/prepare_travelplanner_atomic_operation_neumf_data.py" \
  --subgoals "$DATA/travelplanner_atomic_memory_candidates_v4/task_subgoals.jsonl" \
  --operations "$DATA/travelplanner_atomic_memory_candidates_v4/memory_operations.jsonl" \
  --train-pairs "$DATA/travelplanner_atomic_memory_labels_v1/train_pairs_resolved.jsonl" \
  --dev-pairs "$DATA/travelplanner_atomic_memory_labels_v1/internal_dev_pairs_resolved.jsonl" \
  --output-dir "$DATA/travelplanner_atomic_operation_neumf_data_v1" \
  --model text-embedding-3-small \
  --batch-size 64

"$TRAIN_PYTHON" "$ROOT/scripts/memp/train_memory_neumf.py" \
  --arrays "$DATA/travelplanner_atomic_operation_neumf_data_v1/arrays.npz" \
  --output "$DATA/travelplanner_atomic_operation_neumf_model_v1" \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --patience 5 \
  --device cpu
