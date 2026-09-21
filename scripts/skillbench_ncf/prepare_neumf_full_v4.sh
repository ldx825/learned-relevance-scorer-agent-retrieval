#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${ROOT}/.envs/gos-ncf-eval/bin/python"

if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: missing environment: ${PYTHON}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/scripts/skillbench_ncf${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PYTHON}" scripts/skillbench_ncf/calibrate_graded_labels_v4.py
"${PYTHON}" scripts/skillbench_ncf/check_no_eval_leakage_v4.py
"${PYTHON}" scripts/skillbench_ncf/prepare_neumf_full_v4.py
"${PYTHON}" scripts/skillbench_ncf/preflight_neumf_full_v4.py
