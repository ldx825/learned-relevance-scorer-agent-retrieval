#!/usr/bin/env bash
# Create or refresh local .env defaults for SkillDAG reproduction.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
TEMPLATE_FILE="${REPO_ROOT}/.env.example"

python_is_compatible() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

detect_python() {
  local candidate path
  for candidate in \
    "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/.venv/bin/python3" \
    python3.11 python3.12 python3.10 python3 python \
    "${HOME}/.local/bin/python3.11" \
    "${HOME}/.local/bin/python3.12" \
    /opt/homebrew/bin/python3.11 \
    /usr/local/bin/python3.11; do
    if [[ "${candidate}" == */* ]]; then
      [ -x "${candidate}" ] || continue
      path="${candidate}"
    else
      path="$(command -v "${candidate}" 2>/dev/null || true)"
    fi
    if [ -n "${path}" ] && python_is_compatible "${path}"; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

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

add_path_dir() {
  local dir="$1"
  [ -n "${dir}" ] || return 0
  [ -d "${dir}" ] || return 0
  case ":${PATH_DIRS_JOINED:-}:" in
    *":${dir}:"*) return 0 ;;
  esac
  PATH_DIRS+=("${dir}")
  PATH_DIRS_JOINED="${PATH_DIRS_JOINED:+${PATH_DIRS_JOINED}:}${dir}"
}

add_tool_dir() {
  local tool="$1" tool_path
  tool_path="$(command -v "${tool}" 2>/dev/null || true)"
  if [ -n "${tool_path}" ]; then
    add_path_dir "$(dirname "${tool_path}")"
  fi
  return 0
}

if [ ! -f "${ENV_FILE}" ]; then
  cp "${TEMPLATE_FILE}" "${ENV_FILE}"
  echo "[env] created ${ENV_FILE} from .env.example"
else
  echo "[env] updating existing ${ENV_FILE}"
fi

PYTHON_BIN="$(detect_python || true)"
if [ -z "${PYTHON_BIN}" ]; then
  echo "ERROR: Python >=3.10 not found. Install python3.11, then rerun this script." >&2
  exit 2
fi
set_env_line "PYTHON" "${PYTHON_BIN}"

PATH_DIRS=()
PATH_DIRS_JOINED=""
add_tool_dir harbor
add_tool_dir envsubst
add_tool_dir codex
add_tool_dir docker
add_path_dir "${REPO_ROOT}/.venv/bin"
add_path_dir "/opt/homebrew/bin"
add_path_dir "/usr/local/bin"
add_path_dir "${HOME}/.local/bin"

if [ -n "${PATH_DIRS_JOINED}" ]; then
  set_env_line "PATH" "${PATH_DIRS_JOINED}:\${PATH}"
fi
set_env_line "ALFWORLD_DATA" "${REPO_ROOT}/data/alfworld/data"

echo "[env] PYTHON=${PYTHON_BIN}"
echo "[env] PATH prefixes=${PATH_DIRS_JOINED:-<none detected>}"
echo "[env] done. Fill API keys in ${ENV_FILE} before running benchmarks."
