#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${ROOT}/references/SkillDAG"
ENV_DIR="${ROOT}/.envs/skilldag-debug"
RUNTIME="${ROOT}/.runtime/skilldag"
DATA_ROOT="${RUNTIME}/data/skilldag"
ALFWORLD_DATA_DIR="${RUNTIME}/data/alfworld"
DOWNLOADS="${RUNTIME}/downloads"
PYTHON="${ENV_DIR}/bin/python"
CONDA="/data/miniconda3/bin/conda"
HF_BASE="https://huggingface.co/datasets/Eric068/SkillDAG/resolve/main"
SKILLDAG_PATCH="${ROOT}/patches/skilldag-runtime-debug-yunwu.patch"

if [ ! -f "${PROJECT}/benchmarks/alfworld/skilldag_runtime.py" ]; then
  echo "ERROR: missing SkillDAG checkout at ${PROJECT}." >&2
  exit 2
fi
if [ ! -f "${SKILLDAG_PATCH}" ]; then
  echo "ERROR: missing local SkillDAG patch at ${SKILLDAG_PATCH}." >&2
  exit 2
fi

# references/SkillDAG stays an independently managed, Git-ignored checkout.
# Persist the local debug and Yunwu integration as a tracked, idempotent patch.
if git -C "${ROOT}" apply --unidiff-zero --reverse --check \
  --directory=references/SkillDAG "${SKILLDAG_PATCH}" 2>/dev/null; then
  echo "[skip] SkillDAG debug/Yunwu patch already applied"
elif git -C "${ROOT}" apply --unidiff-zero --check \
  --directory=references/SkillDAG "${SKILLDAG_PATCH}"; then
  git -C "${ROOT}" apply --unidiff-zero \
    --directory=references/SkillDAG "${SKILLDAG_PATCH}"
  echo "[patch] applied SkillDAG debug/Yunwu integration"
else
  echo "ERROR: SkillDAG source does not match the expected patch base." >&2
  echo "Expected upstream commit: 5ade1cb097287ac624dee445fc44c8d463ccb6a6" >&2
  exit 2
fi

mkdir -p "${RUNTIME}/cache/conda-pkgs" "${RUNTIME}/cache/pip" \
  "${DATA_ROOT}/skilldag_graphs" "${DATA_ROOT}/skillsets" "${DOWNLOADS}"

if [ ! -x "${PYTHON}" ]; then
  CONDA_PKGS_DIRS="${RUNTIME}/cache/conda-pkgs" \
    "${CONDA}" create -p "${ENV_DIR}" python=3.11 pip -y
fi

PIP_CACHE_DIR="${RUNTIME}/cache/pip" \
  "${PYTHON}" -m pip install --no-input \
  -e "${PROJECT}[repro,alfworld]" debugpy

download() {
  local url="$1" destination="$2" partial
  if [ -s "${destination}" ]; then
    echo "[skip] ${destination}"
    return
  fi
  mkdir -p "$(dirname "${destination}")"
  partial="${destination}.part"
  curl -fL --retry 3 --progress-bar -o "${partial}" "${url}"
  mv "${partial}" "${destination}"
}

for scale in 200 500 1000 2000 alfworld; do
  for suffix in json embeddings.json; do
    file="skillgraph_${scale}.${suffix}"
    download "${HF_BASE}/data/skilldag_graphs/${file}" \
      "${DATA_ROOT}/skilldag_graphs/${file}"
  done
done

for scale in 200 500 1000 2000; do
  archive="${DOWNLOADS}/skills_${scale}.tar.gz"
  skills_dir="${DATA_ROOT}/skillsets/skills_${scale}"
  download "${HF_BASE}/data/skillsets_archives/skills_${scale}.tar.gz" "${archive}"
  count=0
  if [ -d "${skills_dir}" ]; then
    count="$(find "${skills_dir}" -maxdepth 2 -name SKILL.md | wc -l)"
  fi
  if [ "${count}" -ne "${scale}" ]; then
    mkdir -p "${skills_dir}"
    tar -xzf "${archive}" -C "${skills_dir}" --strip-components=1
  fi
done

alfworld_archive="${DOWNLOADS}/alfworld_skills.tar.gz"
download "${HF_BASE}/alfworld_skills.tar.gz" "${alfworld_archive}"
alfworld_skill_count=0
if [ -d "${DATA_ROOT}/alfworld_skills" ]; then
  alfworld_skill_count="$(find "${DATA_ROOT}/alfworld_skills" -maxdepth 2 -name SKILL.md | wc -l)"
fi
if [ "${alfworld_skill_count}" -ne 37 ]; then
  tar -xzf "${alfworld_archive}" -C "${DATA_ROOT}"
fi

if [ ! -d "${ALFWORLD_DATA_DIR}/json_2.1.1/valid_seen" ]; then
  "${ENV_DIR}/bin/alfworld-download" --data-dir "${ALFWORLD_DATA_DIR}"
fi

if [ ! -f "${RUNTIME}/skilldag.env" ]; then
  cp "${ROOT}/config/skilldag.env.example" "${RUNTIME}/skilldag.env"
  chmod 600 "${RUNTIME}/skilldag.env"
  echo "[config] created ${RUNTIME}/skilldag.env; fill the provider keys needed before an API run"
fi

echo "[done] Python: ${PYTHON}"
echo "[done] SkillDAG data: ${DATA_ROOT}"
echo "[done] ALFWorld data: ${ALFWORLD_DATA_DIR}"
