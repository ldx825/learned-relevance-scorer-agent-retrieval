#!/usr/bin/env bash
set -euo pipefail

# Download benchmark data assets used by the bundled SkillDAG evaluations.
#
# Usage:
#   ./scripts/download_data.sh                # download all assets
#   ./scripts/download_data.sh --skillsets    # download skill sets only
#   ./scripts/download_data.sh --tasks        # clone SkillsBench tasks only
#   ./scripts/download_data.sh --workspace    # download prebuilt workspace only

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"

# ─── Sources ─────────────────────────────────────────────────────────────────
# Tasks: upstream SkillsBench repo (latest default branch, sparse checkout of tasks/ only).
SKILLSBENCH_REPO="https://github.com/benchflow-ai/skillsbench.git"

# Upstream GoS HuggingFace Hub (skill sets + GoS HNSW workspaces)
HF_GOS_REPO="${GOS_HF_REPO:-DLPenn/graph-of-skills-data}"
HF_GOS_BASE="https://huggingface.co/datasets/${HF_GOS_REPO}/resolve/main"

# SkillDAG paper artifacts HuggingFace Hub (skill graphs + embeddings + skillsets)
# These are the authoritative artifacts used in the paper experiments.
HF_SD_REPO="${SKILLDAG_HF_REPO:-Eric068/SkillDAG}"
HF_SD_BASE="https://huggingface.co/datasets/${HF_SD_REPO}/resolve/main"

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
    local url="${HF_SD_BASE}/${name}.tar.gz"
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
    local url="${HF_GOS_BASE}/${archive}"
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

# ─── SkillDAG graph artifacts ────────────────────────────────────────────────
# Download skill graphs + embeddings from our HF repo (Eric068/SkillDAG).
# These are the authoritative artifacts from the paper experiments.
download_hf_file() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [ -f "$dest" ]; then
        echo "  [skip] ${name} already exists at ${dest}"
        return
    fi

    mkdir -p "$(dirname "$dest")"
    echo "  [download] ${name} ..."
    local curl_opts=(-fSL)
    if [ -n "${HF_TOKEN:-}" ]; then
        curl_opts+=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    if command -v curl &>/dev/null; then
        if curl "${curl_opts[@]}" -o "$dest" "$url"; then
            echo "  [done] ${name}"
        else
            rm -f "$dest"
            echo "  [skip] ${name} — download failed: ${url}" >&2
        fi
    elif command -v wget &>/dev/null; then
        local wget_opts=(-qO "$dest")
        if [ -n "${HF_TOKEN:-}" ]; then
            wget_opts+=(--header="Authorization: Bearer ${HF_TOKEN}")
        fi
        if wget "${wget_opts[@]}" "$url"; then
            echo "  [done] ${name}"
        else
            rm -f "$dest"
            echo "  [skip] ${name} — download failed: ${url}" >&2
        fi
    else
        echo "  [error] Neither curl nor wget found." >&2
        exit 1
    fi
}

download_graphs() {
    echo "Downloading SkillDAG graph artifacts from HuggingFace ..."
    mkdir -p "${DATA_DIR}/skilldag_graphs"

    local base="${HF_SD_BASE}"

    # Skill graphs + embeddings
    for scale in 200 500 1000 2000; do
        download_hf_file "${base}/data/skilldag_graphs/skillgraph_${scale}.json" \
            "${DATA_DIR}/skilldag_graphs/skillgraph_${scale}.json" \
            "skillgraph_${scale}.json"
        download_hf_file "${base}/data/skilldag_graphs/skillgraph_${scale}.embeddings.json" \
            "${DATA_DIR}/skilldag_graphs/skillgraph_${scale}.embeddings.json" \
            "skillgraph_${scale}.embeddings.json"
    done

    # ALFWorld graph
    download_hf_file "${base}/data/skilldag_graphs/skillgraph_alfworld.json" \
        "${DATA_DIR}/skilldag_graphs/skillgraph_alfworld.json" \
        "skillgraph_alfworld.json"
    download_hf_file "${base}/data/skilldag_graphs/skillgraph_alfworld.embeddings.json" \
        "${DATA_DIR}/skilldag_graphs/skillgraph_alfworld.embeddings.json" \
        "skillgraph_alfworld.embeddings.json"

    echo "  Graph artifacts -> ${DATA_DIR}/skilldag_graphs/"
}

download_alfworld_skills() {
    echo "Downloading ALFWorld skill library from HuggingFace ..."
    mkdir -p "${DATA_DIR}/alfworld_skills"
    local tmp="${TMPDIR:-/tmp}/alfworld_skills.tar.gz"
    local url="${HF_SD_BASE}/alfworld_skills.tar.gz"
    local curl_opts=(-fSL)
    if [ -n "${HF_TOKEN:-}" ]; then
        curl_opts+=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    echo "  [download] alfworld_skills.tar.gz ..."
    if command -v curl &>/dev/null; then
        curl "${curl_opts[@]}" -o "$tmp" "$url"
    elif command -v wget &>/dev/null; then
        wget -qO "$tmp" "$url"
    fi
    if tar -xzf "$tmp" -C "${DATA_DIR}" && [ -d "${DATA_DIR}/alfworld_skills" ]; then
        echo "  [done] alfworld_skills"
    else
        echo "  [skip] alfworld_skills — download failed" >&2
    fi
    rm -f "$tmp"
}

download_tasks() {
    local dest="${DATA_DIR}/tasks/tasks"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  [skip] tasks already exists at ${dest}"
        return
    fi
    if [ -d "$dest" ]; then
        rm -rf "$dest"
    fi

    echo "Cloning SkillsBench tasks from GitHub (latest sparse checkout) ..."
    local tmpdir
    tmpdir="$(mktemp -d)"
    git clone --depth 1 --filter=blob:none --sparse \
        "${SKILLSBENCH_REPO}" "$tmpdir" 2>&1 | sed 's/^/  /'
    (cd "$tmpdir" && git sparse-checkout set tasks) 2>&1 | sed 's/^/  /'
    local commit_hash
    commit_hash="$(cd "$tmpdir" && git rev-parse HEAD)"
    echo "    fetched commit: ${commit_hash}"

    mkdir -p "$(dirname "$dest")"
    mv "$tmpdir/tasks" "$dest"
    rm -rf "$tmpdir"
    echo "  [done] tasks ($(find "$dest" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') task dirs from benchflow-ai/skillsbench@${commit_hash})"
}

download_workspace() {
    echo "Downloading prebuilt workspaces from HuggingFace ..."
    for suffix in skills_200_v1 skills_500_v1 skills_1000_v1 skills_2000_v1; do
        download_prebuilt_workspace_archive "$suffix"
    done
}

show_help() {
    echo "Usage: $0 [--graphs] [--skillsets] [--tasks] [--workspace] [--all] [--help]"
    echo ""
    echo "Downloads benchmark data assets for the bundled SkillDAG evaluations."
    echo ""
    echo "  --graphs        SkillDAG graph artifacts from HuggingFace (Eric068/SkillDAG)"
    echo "                  -> data/skilldag_graphs/ (skillgraph_*.json + .embeddings.json)"
    echo "                  This is the PRIMARY download for reproducible runs."
    echo "  --skillsets      Skill libraries from HuggingFace -> data/skillsets/"
    echo "                  (skills_200, skills_500, skills_1000, skills_2000)"
    echo "                  Existing non-empty dirs are skipped."
    echo "  --alfworld-skills  ALFWorld skill library -> data/alfworld_skills/"
    echo "  --tasks          SkillsBench benchmark tasks from GitHub -> data/tasks/tasks/"
    echo "  --workspace      Prebuilt workspaces (GoS HNSW, from DLPenn/graph-of-skills-data)"
    echo "                  -> data/gos_workspace/  (optional, not needed for reproduction)"
    echo "  --all            Download everything (default when no flags given)"
    echo ""
    echo "Environment variables:"
    echo "  HF_TOKEN          HuggingFace token (for private/gated repos)"
    echo "  SKILLDAG_HF_REPO  Override SkillDAG artifacts repo (default: Eric068/SkillDAG)"
    echo "  GOS_HF_REPO       Override GoS dataset repo (default: DLPenn/graph-of-skills-data)"
}

if [ $# -eq 0 ]; then
    download_graphs
    download_skillsets
    download_tasks
    download_alfworld_skills
    echo ""
    echo "Data assets downloaded:"
    echo "  SkillDAG graphs -> ${DATA_DIR}/skilldag_graphs/"
    echo "  Skill sets      -> ${DATA_DIR}/skillsets/"
    echo "  Tasks           -> ${DATA_DIR}/tasks/tasks/"
    echo "  ALFWorld skills -> ${DATA_DIR}/alfworld_skills/"
    exit 0
fi

for arg in "$@"; do
    case "$arg" in
        --graphs)         download_graphs         ;;
        --skillsets)      download_skillsets      ;;
        --alfworld-skills) download_alfworld_skills ;;
        --tasks)          download_tasks          ;;
        --workspace)      download_workspace      ;;
        --all)
            download_graphs
            download_skillsets
            download_tasks
            download_alfworld_skills
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
