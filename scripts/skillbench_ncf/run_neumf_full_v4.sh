#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"
MODEL_DIR="${ROOT}/.runtime/skillbench_ncf/models/neumf_full_v4"
ARRAYS="${ROOT}/data/skillbench_ncf/model_data/neumf_full_v4/arrays.npz"
HYBRID_SKILLS="${ROOT}/data/skillbench_ncf/enriched_skill_embeddings_v3/hybrid_skill_embeddings.json"
GRAPH="${ROOT}/.runtime/skillbench_ncf/sources/skillgraph_1000.json"

if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: missing environment: ${PYTHON}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/src/SkillDAG_NCF/src:${ROOT}/src/GraphOfSkills_NCF:${ROOT}/scripts/ncf_v3${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PYTHON}" scripts/skillbench_ncf/preflight_neumf_full_v4.py
"${PYTHON}" scripts/skillbench_ncf/train_neumf_full_v4.py
"${PYTHON}" scripts/skillbench_ncf/calibrate_neumf_full_v4.py \
  --arrays "${ARRAYS}" \
  --model-dir "${MODEL_DIR}"
"${PYTHON}" src/SkillDAG_NCF/scripts/export_portable_ncf.py \
  --checkpoint "${MODEL_DIR}/neumf.pt" \
  --embeddings "${HYBRID_SKILLS}" \
  --graph "${GRAPH}" \
  --calibration "${MODEL_DIR}/online_calibration.json" \
  --output "${MODEL_DIR}/ncf_bundle.json"
