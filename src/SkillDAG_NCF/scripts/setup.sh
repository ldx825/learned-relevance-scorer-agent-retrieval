#!/usr/bin/env bash
# setup.sh — one-shot environment setup for SkillDAG reproduction.
#
# Does:
#   1. Creates/refreshes .env with local Python/PATH defaults
#   2. Installs this package in editable mode with reproduction dependencies
#   3. Downloads graphs, skill libraries, SkillsBench tasks, and optional prebuilt workspaces
#      into ./data using the bundled downloader
#   4. Builds data/alfworld_skills from the downloaded skill libraries when needed
#   5. Verifies the expected SkillDAG graph artifacts are present
#
# Does NOT:
#   - Install the SkillsBench/Harbor framework (separate repo + heavy deps).
#     See docs/reproducing.md for Harbor install once setup.sh completes.
#   - Install ALFWorld task data; run the upstream downloader if
#     data/alfworld/data is absent.
#
# Usage:
#   bash scripts/setup.sh                 # full setup
#   bash scripts/setup.sh --skip-install  # only data/setup checks
#   bash scripts/setup.sh --skip-data     # skip external data downloads
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
DATA_DOWNLOAD_SCRIPT="${REPO_ROOT}/scripts/download_data.sh"
ENV_PREP_SCRIPT="${REPO_ROOT}/scripts/prepare_env.sh"
ENV_FILE="${REPO_ROOT}/.env"

SKIP_INSTALL=0
SKIP_DATA=0
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-data) SKIP_DATA=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

set_env_line() {
  local key="$1" value="$2" tmp
  tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
  awk -v key="${key}" -v value="${value}" '
    BEGIN { done = 0 }
    $0 ~ "^" key "=" {
      if (!done) {
        print key "=" value
        done = 1
      }
      next
    }
    { print }
    END {
      if (!done) {
        print key "=" value
      }
    }
  ' "${ENV_FILE}" > "${tmp}"
  mv "${tmp}" "${ENV_FILE}"
}

prepend_env_path() {
  local dir="$1" current
  current="$(grep -E '^PATH=' "${ENV_FILE}" | tail -n 1 | cut -d= -f2- || true)"
  [ -n "${current}" ] || current="\${PATH}"
  case ":${current}:" in
    *":${dir}:"*) return 0 ;;
  esac
  set_env_line "PATH" "${dir}:${current}"
}

echo "[setup] repo root: ${REPO_ROOT}"
mkdir -p "${DATA_DIR}"

bash "${ENV_PREP_SCRIPT}"
set -a; source "${ENV_FILE}"; set +a

PYTHON_BIN="${PYTHON:-python3.11}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: PYTHON=${PYTHON_BIN} is not on PATH. Run scripts/prepare_env.sh again." >&2
  exit 2
fi
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "ERROR: PYTHON=${PYTHON_BIN} must be Python >=3.10; python3.11 is recommended." >&2
  exit 2
fi

if [ "${SKIP_INSTALL}" = "0" ]; then
  if [ "${SKILLDAG_USE_VENV:-1}" != "0" ]; then
    VENV_DIR="${SKILLDAG_VENV:-${REPO_ROOT}/.venv}"
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
      echo "[setup] creating virtual environment: ${VENV_DIR}"
      "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
    PYTHON_BIN="${VENV_DIR}/bin/python"
    export PYTHON="${PYTHON_BIN}"
    export PATH="${VENV_DIR}/bin:${PATH}"
    set_env_line "PYTHON" "${PYTHON_BIN}"
    prepend_env_path "${VENV_DIR}/bin"
  fi
  INSTALL_TARGET="${SKILLDAG_INSTALL_TARGET:-.[repro,alfworld]}"
  echo "[setup] installing SkillDAG editable package: pip install -e ${INSTALL_TARGET}"
  ( cd "${REPO_ROOT}" && "${PYTHON_BIN}" -m pip install -e "${INSTALL_TARGET}" )
else
  echo "[setup] --skip-install set; skipping pip install -e."
fi

# Remove stale symlinks from older prerelease setups; keep real directories.
for name in skillsets gos_workspace tasks; do
  dst="${DATA_DIR}/${name}"
  if [ -L "${dst}" ]; then
    rm -f "${dst}"
    echo "[setup] removed stale symlink data/${name}"
  fi
done

# ─── 1. Download benchmark data (skill libraries + tasks + workspaces) ───
if [ "${SKIP_DATA}" = "0" ]; then
  echo "[setup] downloading benchmark data into ${DATA_DIR} ..."
  bash "${DATA_DOWNLOAD_SCRIPT}"
else
  echo "[setup] --skip-data set; skipping HuggingFace downloads."
fi
mkdir -p "${DATA_DIR}/skilldag_graphs"

# Build a narrow ALFWorld skill pool for the ALFWorld graph. The skillset
# archives contain these skills inside the larger skill libraries; the
# ALFWorld runner must not point at the full pool or it will mutate the graph
# with unrelated nodes on load.
ALFWORLD_SKILLS_DIR="${DATA_DIR}/alfworld_skills"
if [ ! -d "${ALFWORLD_SKILLS_DIR}" ] || [ -z "$(find "${ALFWORLD_SKILLS_DIR}" -mindepth 2 -maxdepth 2 -name SKILL.md -print -quit 2>/dev/null)" ]; then
  rm -rf "${ALFWORLD_SKILLS_DIR}"
  for candidate in \
    "${DATA_DIR}/skillsets/skills_1000" \
    "${DATA_DIR}/skillsets/skills_1000/skills_1000" \
    "${DATA_DIR}/skillsets/skills_2000" \
    "${DATA_DIR}/skillsets/skills_2000/skills_2000" \
    "${DATA_DIR}/skillsets/skills_500" \
    "${DATA_DIR}/skillsets/skills_500/skills_500"; do
    if [ -d "${candidate}" ] && [ -n "$(find "${candidate}" -maxdepth 1 -type d -name 'alfworld-*' -print -quit)" ]; then
      mkdir -p "${ALFWORLD_SKILLS_DIR}"
      rel_prefix="../${candidate#"${DATA_DIR}/"}"
      for skill_dir in "${candidate}"/alfworld-*; do
        [ -d "${skill_dir}" ] || continue
        skill_id="$(basename "${skill_dir}")"
        ln -s "${rel_prefix}/${skill_id}" "${ALFWORLD_SKILLS_DIR}/${skill_id}"
      done
      echo "[setup] symlinked data/alfworld_skills from ${candidate}"
      break
    fi
  done
fi
if [ ! -e "${ALFWORLD_SKILLS_DIR}" ]; then
  echo "[setup] ALFWorld skill pool not found yet; run setup without --skip-data or set SKILLS_DIR manually."
fi

# ─── 2. Verify SkillDAG graph artifacts (downloaded by scripts/download_data.sh) ───
echo
missing_graphs=0
missing_embeddings=0
for scale in 200 500 1000 2000 alfworld; do
  graph_file="${DATA_DIR}/skilldag_graphs/skillgraph_${scale}.json"
  emb_file="${DATA_DIR}/skilldag_graphs/skillgraph_${scale}.embeddings.json"
  if [ ! -f "${graph_file}" ]; then
    missing_graphs=1
    echo "  missing graph: skillgraph_${scale}.json"
  fi
  if [ ! -f "${emb_file}" ]; then
    missing_embeddings=1
    echo "  missing embedding cache: skillgraph_${scale}.embeddings.json"
  fi
done

if [ "${missing_graphs}" = "0" ] && [ "${missing_embeddings}" = "0" ]; then
  echo "[setup] found published SkillDAG graph artifacts and embeddings under data/skilldag_graphs/"
elif [ "${missing_graphs}" != "0" ]; then
  echo "[setup] graph artifacts are missing; rerun: bash scripts/download_data.sh --graphs"
  echo "        Or initialize graphs locally, for example:"
  echo "          skilldag initialize-graph \\"
  echo "            --skills-dir data/skillsets/skills_200/skills_200 \\"
  echo "            --graph-path data/skilldag_graphs/skillgraph_200.json"
  echo "        If your downloaded archive is flat, use data/skillsets/skills_200 instead."
  echo "        This requires SKILLDAG_EMBEDDING_API_KEY + SKILLDAG_LLM_API_KEY in .env."
else
  echo "[setup] embedding caches are missing; rerun: bash scripts/download_data.sh --graphs"
fi

echo
echo "[setup] done. Next steps:"
echo "  1. fill API keys in .env"
echo "  2. install the SkillsBench/Harbor framework if you plan to run SkillsBench"
echo "  3. run alfworld-download if ALFWORLD_DATA does not exist"
echo "  4. bash scripts/run_skillsbench.sh   # or scripts/run_alfworld.sh"
