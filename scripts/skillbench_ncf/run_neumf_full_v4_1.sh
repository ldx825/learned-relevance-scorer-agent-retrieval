#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"
ARRAYS="${ROOT}/data/skillbench_ncf/model_data/neumf_full_v4_1/arrays.npz"
MODEL_DIR="${ROOT}/.runtime/skillbench_ncf/models/neumf_full_v4_1"
HYBRID_SKILLS="${ROOT}/data/skillbench_ncf/enriched_skill_embeddings_v3/hybrid_skill_embeddings.json"
GRAPH="${ROOT}/.runtime/skillbench_ncf/sources/skillgraph_1000.json"

export PYTHONPATH="${ROOT}/src/SkillDAG_NCF/src:${ROOT}/src/GraphOfSkills_NCF:${ROOT}/scripts/ncf_v3:${ROOT}/scripts/skillbench_ncf${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PYTHON}" scripts/skillbench_ncf/reweight_neumf_full_v4_1.py
"${PYTHON}" scripts/skillbench_ncf/train_neumf_full_v4.py \
  --arrays "${ARRAYS}" \
  --output "${MODEL_DIR}" \
  --public-report artifacts/skillbench_ncf/manifests/neumf_full_v4_1_training_report.json \
  --pretrain-epochs 50 \
  --finetune-epochs 30 \
  --patience 6 \
  --sgd-lr 0.001 \
  --pretrain-alpha 0.8
"${PYTHON}" scripts/skillbench_ncf/calibrate_neumf_full_v4.py \
  --arrays "${ARRAYS}" \
  --model-dir "${MODEL_DIR}" \
  --alpha-grid 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00
"${PYTHON}" src/SkillDAG_NCF/scripts/export_portable_ncf.py \
  --checkpoint "${MODEL_DIR}/neumf.pt" \
  --embeddings "${HYBRID_SKILLS}" \
  --graph "${GRAPH}" \
  --calibration "${MODEL_DIR}/online_calibration.json" \
  --output "${MODEL_DIR}/ncf_bundle.json"
