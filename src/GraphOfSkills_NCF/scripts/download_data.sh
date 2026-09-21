#!/usr/bin/env bash
set -euo pipefail

# Download benchmark data assets for Graph of Skills evaluation.
#
# Usage:
#   ./scripts/download_data.sh                # download all assets
#   ./scripts/download_data.sh --skillsets    # download skill sets only
#   ./scripts/download_data.sh --tasks        # clone SkillsBench tasks only
#   ./scripts/download_data.sh --workspace    # download prebuilt workspace only

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
SKILLSBENCH_DIR="${REPO_ROOT}/evaluation/skillsbench"

# ─── Sources ─────────────────────────────────────────────────────────────────
# Tasks: upstream SkillsBench repo (sparse checkout, tasks/ only)
SKILLSBENCH_REPO="https://github.com/benchflow-ai/skillsbench.git"

# Skill sets & workspace: HuggingFace Hub
HF_REPO="${GOS_HF_REPO:-DLPenn/graph-of-skills-data}"
HF_BASE="https://huggingface.co/datasets/${HF_REPO}/resolve/main"

download_and_extract() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  [skip] ${name} already exists at ${dest}"
        return
    fi

    mkdir -p "$dest"
    echo "  [download] ${name} ..."

    local curl_opts=(-fSL)
    if [ -n "${HF_TOKEN:-}" ]; then
        curl_opts+=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi

    if command -v curl &>/dev/null; then
        curl "${curl_opts[@]}" "$url" | tar -xz -C "$dest" --strip-components=1
    elif command -v wget &>/dev/null; then
        local wget_opts=(-qO-)
        if [ -n "${HF_TOKEN:-}" ]; then
            wget_opts+=(--header="Authorization: Bearer ${HF_TOKEN}")
        fi
        wget "${wget_opts[@]}" "$url" | tar -xz -C "$dest" --strip-components=1
    else
        echo "  [error] Neither curl nor wget found." >&2
        exit 1
    fi
    echo "  [done] ${name}"
}

# Download one skill-set tarball into data/skillsets/<name>/.
# Skips if the directory already exists and is non-empty.
# If the archive is missing on the Hub (404 / network failure), prints [skip] and continues
# so optional skill sets (e.g. skills_500) do not abort the whole run.
download_skillset_archive() {
    local name="$1"
    local url="${HF_BASE}/${name}.tar.gz"
    local dest="${DATA_DIR}/skillsets/${name}"

    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  [skip] ${name} already exists at ${dest}"
        return 0
    fi

    mkdir -p "${DATA_DIR}/skillsets"
    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/gos-skillset.XXXXXX.tar.gz")"

    echo "  [download] ${name} ..."

    local ok=0
    if command -v curl &>/dev/null; then
        local curl_opts=(-fSL)
        if [ -n "${HF_TOKEN:-}" ]; then
            curl_opts+=(-H "Authorization: Bearer ${HF_TOKEN}")
        fi
        if curl "${curl_opts[@]}" -o "$tmp" "$url"; then
            ok=1
        fi
    elif command -v wget &>/dev/null; then
        if [ -n "${HF_TOKEN:-}" ]; then
            if wget -q --header="Authorization: Bearer ${HF_TOKEN}" -O "$tmp" "$url"; then
                ok=1
            fi
        else
            if wget -q -O "$tmp" "$url"; then
                ok=1
            fi
        fi
    else
        rm -f "$tmp"
        echo "  [error] Neither curl nor wget found." >&2
        exit 1
    fi

    if [ "$ok" -ne 1 ]; then
        rm -f "$tmp"
        echo "  [skip] ${name} — archive not on Hub or download failed: ${url}" >&2
        return 0
    fi

    mkdir -p "$dest"
    if ! tar -xzf "$tmp" -C "$dest" --strip-components=1; then
        rm -f "$tmp"
        echo "  [error] ${name} — failed to extract archive" >&2
        exit 1
    fi
    rm -f "$tmp"
    echo "  [done] ${name}"
}

# Prebuilt graph workspace: gos_workspace_skills_200_v1.tar.gz -> data/gos_workspace/skills_200_v1/
download_prebuilt_workspace_archive() {
    local suffix="$1"
    local archive="gos_workspace_${suffix}.tar.gz"
    local url="${HF_BASE}/${archive}"
    local dest="${DATA_DIR}/gos_workspace/${suffix}"

    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  [skip] ${suffix} already exists at ${dest}"
        return 0
    fi

    mkdir -p "${DATA_DIR}/gos_workspace"
    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/gos-workspace.XXXXXX.tar.gz")"

    echo "  [download] ${archive} ..."

    local ok=0
    if command -v curl &>/dev/null; then
        local curl_opts=(-fSL)
        if [ -n "${HF_TOKEN:-}" ]; then
            curl_opts+=(-H "Authorization: Bearer ${HF_TOKEN}")
        fi
        if curl "${curl_opts[@]}" -o "$tmp" "$url"; then
            ok=1
        fi
    elif command -v wget &>/dev/null; then
        if [ -n "${HF_TOKEN:-}" ]; then
            if wget -q --header="Authorization: Bearer ${HF_TOKEN}" -O "$tmp" "$url"; then
                ok=1
            fi
        else
            if wget -q -O "$tmp" "$url"; then
                ok=1
            fi
        fi
    else
        rm -f "$tmp"
        echo "  [error] Neither curl nor wget found." >&2
        exit 1
    fi

    if [ "$ok" -ne 1 ]; then
        rm -f "$tmp"
        echo "  [skip] ${archive} — not on Hub or download failed: ${url}" >&2
        return 0
    fi

    mkdir -p "$dest"
    if ! tar -xzf "$tmp" -C "$dest" --strip-components=1; then
        rm -f "$tmp"
        echo "  [error] ${suffix} — failed to extract archive" >&2
        exit 1
    fi
    rm -f "$tmp"
    echo "  [done] ${suffix}"
}

download_skillsets() {
    echo "Downloading skill sets from HuggingFace ..."
    for name in skills_200 skills_500 skills_1000 skills_2000; do
        download_skillset_archive "$name"
    done
}

download_tasks() {
    local dest="${SKILLSBENCH_DIR}/tasks"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  [skip] tasks already exists at ${dest}"
        return
    fi

    echo "Cloning SkillsBench tasks from GitHub (sparse checkout) ..."
    local tmpdir
    tmpdir="$(mktemp -d)"
    git clone --depth 1 --filter=blob:none --sparse \
        "${SKILLSBENCH_REPO}" "$tmpdir" 2>&1 | sed 's/^/  /'
    (cd "$tmpdir" && git sparse-checkout set tasks) 2>&1 | sed 's/^/  /'

    mkdir -p "$(dirname "$dest")"
    mv "$tmpdir/tasks" "$dest"
    rm -rf "$tmpdir"
    echo "  [done] tasks ($(ls "$dest" | wc -l | tr -d ' ') tasks from benchflow-ai/skillsbench)"
}

download_workspace() {
    echo "Downloading prebuilt GoS workspaces from HuggingFace ..."
    for suffix in skills_200_v1 skills_500_v1 skills_1000_v1 skills_2000_v1; do
        download_prebuilt_workspace_archive "$suffix"
    done
}

show_help() {
    echo "Usage: $0 [--skillsets] [--tasks] [--workspace] [--all] [--help]"
    echo ""
    echo "Downloads benchmark data assets for Graph of Skills evaluation."
    echo ""
    echo "  --skillsets   Skill libraries from HuggingFace -> data/skillsets/"
    echo "                (skills_200, skills_500, skills_1000, skills_2000)"
    echo "                Existing non-empty dirs are skipped; missing Hub archives log [skip] and continue."
    echo "  --tasks       87 benchmark tasks from SkillsBench GitHub -> evaluation/skillsbench/tasks/"
    echo "  --workspace   Prebuilt workspaces (gos_workspace_skills_*_v1.tar.gz) -> data/gos_workspace/"
    echo "                Tries 200/500/1000/2000; skips existing dirs; missing archives log [skip]."
    echo "  --all         Download everything (default when no flags given)"
    echo ""
    echo "Environment variables:"
    echo "  HF_TOKEN        HuggingFace token (for private/gated repos)"
    echo "  GOS_HF_REPO     Override HuggingFace dataset repo (default: DLPenn/graph-of-skills-data)"
}

if [ $# -eq 0 ]; then
    download_skillsets
    download_tasks
    download_workspace
    echo ""
    echo "Data assets downloaded:"
    echo "  Skill sets  -> ${DATA_DIR}/skillsets/"
    echo "  Tasks       -> ${SKILLSBENCH_DIR}/tasks/"
    echo "  Workspace   -> ${DATA_DIR}/gos_workspace/"
    exit 0
fi

for arg in "$@"; do
    case "$arg" in
        --skillsets)  download_skillsets  ;;
        --tasks)      download_tasks      ;;
        --workspace)  download_workspace  ;;
        --all)
            download_skillsets
            download_tasks
            download_workspace
            echo ""
            echo "All data assets downloaded."
            ;;
        --help|-h)    show_help; exit 0   ;;
        *)
            echo "Unknown option: $arg" >&2
            show_help
            exit 1
            ;;
    esac
done
