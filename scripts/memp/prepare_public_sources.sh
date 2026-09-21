#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM="${ROOT}/.runtime/memp/upstream/MemP"
TARGET="${ROOT}/src/MemP_NCF/ProcedureMem/Alfworld/alfworld_format_traj.json"
COMMIT="3066a1b280a39c7433ae37d9a745861903a5d3c4"

if [[ ! -d "${UPSTREAM}/.git" ]]; then
  mkdir -p "$(dirname "${UPSTREAM}")"
  git clone https://github.com/zjunlp/MemP.git "${UPSTREAM}"
fi

git -C "${UPSTREAM}" fetch origin
git -C "${UPSTREAM}" checkout --detach "${COMMIT}"
mkdir -p "$(dirname "${TARGET}")"
cp "${UPSTREAM}/ProcedureMem/Alfworld/alfworld_format_traj.json" "${TARGET}"

echo "Restored public MemP ALFWorld trajectories at ${TARGET}"
