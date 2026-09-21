#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REVISION="46fbb5121d06ab5c6c3e1713f47ac62c8e57b75a"
BASE="https://huggingface.co/datasets/Eric068/SkillDAG/resolve/${REVISION}"
RUNTIME="${ROOT}/.runtime/skillbench_ncf/sources"
ARCHIVE="${RUNTIME}/skills_1000.tar.gz"
GRAPH="${RUNTIME}/skillgraph_1000.json"
EMBEDDINGS="${RUNTIME}/skillgraph_1000.embeddings.json"
SKILLS_DIR="${RUNTIME}/skills_1000"
ARCHIVE_SHA="48cf777043cea7547afa1f0082337428add0a4543c73c56a863c03dc1908faba"
GRAPH_SHA="92bf8e8fa6f1e45efe90e1f5f303dca294efe5d4c7986b5dfc71a66825183651"
EMBEDDINGS_SHA="7382a7e3e2613dfd3f8915552243b9cb6ff6ad9c6c473e239693cbec99a8e02f"

download_locked() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local actual_sha

  if [ -f "${destination}" ]; then
    actual_sha="$(sha256sum "${destination}" | awk '{print $1}')"
    if [ "${actual_sha}" != "${expected_sha}" ]; then
      echo "ERROR: existing file has the wrong SHA256: ${destination}" >&2
      echo "expected=${expected_sha}" >&2
      echo "actual=${actual_sha}" >&2
      exit 2
    fi
    echo "[verified] ${destination}"
    return
  fi

  mkdir -p "$(dirname "${destination}")"
  local temporary
  temporary="$(mktemp "${destination}.download.XXXXXX")"
  if ! curl -fL --retry 3 --progress-bar -o "${temporary}" "${url}"; then
    rm -f "${temporary}"
    exit 2
  fi
  actual_sha="$(sha256sum "${temporary}" | awk '{print $1}')"
  if [ "${actual_sha}" != "${expected_sha}" ]; then
    echo "ERROR: downloaded SHA256 mismatch: ${url}" >&2
    echo "expected=${expected_sha}" >&2
    echo "actual=${actual_sha}" >&2
    rm -f "${temporary}"
    exit 2
  fi
  mv "${temporary}" "${destination}"
  echo "[downloaded+verified] ${destination}"
}

download_locked \
  "${BASE}/data/skillsets_archives/skills_1000.tar.gz" \
  "${ARCHIVE}" \
  "${ARCHIVE_SHA}"
download_locked \
  "${BASE}/data/skilldag_graphs/skillgraph_1000.json" \
  "${GRAPH}" \
  "${GRAPH_SHA}"
download_locked \
  "${BASE}/data/skilldag_graphs/skillgraph_1000.embeddings.json" \
  "${EMBEDDINGS}" \
  "${EMBEDDINGS_SHA}"

if [ ! -d "${SKILLS_DIR}" ]; then
  temporary_dir="$(mktemp -d "${RUNTIME}/skills_1000.extract.XXXXXX")"
  tar -xzf "${ARCHIVE}" -C "${temporary_dir}" --strip-components=1
  mv "${temporary_dir}" "${SKILLS_DIR}"
  echo "[extracted] ${SKILLS_DIR}"
fi

python3 "${ROOT}/scripts/skillbench_ncf/build_source_manifest.py" \
  --skills-dir "${SKILLS_DIR}" \
  --archive "${ARCHIVE}" \
  --graph "${GRAPH}"
