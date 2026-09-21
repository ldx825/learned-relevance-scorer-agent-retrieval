# Evaluation

Benchmark runners and shared experiment code for **Graph of Skills** (GoS).

## Related documentation

| Doc | Use it for |
|-----|------------|
| [README.md](../README.md) | Install, Quick Start, workspace layout, **minimal verification** (retrieval + one SkillsBench task) |
| [DATA.md](../DATA.md) | Downloading skill sets, tasks, prebuilt workspaces; `scripts/download_data.sh` flags |
| [`.env.example`](../.env.example) | `GOS_*` embedding variables (must match the indexed workspace) |
| [skillsbench/README.md](skillsbench/README.md) | Harbor, Docker, `graphskills_benchmark.py`, batch YAML, agents |

---

## Before you run anything

1. **Repository root** — run paths below from the repo root (`graph-of-skills/`), unless noted.
2. **Data** — skill libraries and (for SkillsBench) task definitions:

   ```bash
   ./scripts/download_data.sh
   ```

   Selective options: [DATA.md](../DATA.md).

3. **GoS workspace** — for `gos` / `vector` modes you need an indexed directory under `data/gos_workspace/` whose **embedding** matches your `.env` (see [Building or fetching a workspace](#building-or-fetching-a-workspace)).

4. **Agent LLM (ALFWorld)** — the ALFWorld runner expects **`API_KEY`** and **`BASE_URL`** for an OpenAI-compatible chat API (separate from `GOS_*` embedding vars). We recommend [OpenRouter](https://openrouter.ai/) for evaluation (see [README.md](../README.md#evaluation)).

---

## Overview

| Track | What it is | Scale | Entry point |
|-------|------------|-------|-------------|
| [ALFWorld](#alfworld) | Text-based household tasks | 134 games (splits configurable) | `evaluation/alfworld_run.py` |
| [SkillsBench](#skillsbench) | Dockerized coding tasks | 87 tasks | Harbor + [skillsbench/](skillsbench/) |

### Retrieval modes (ALFWorld / shared `SkillModule`)

All modes below are implemented in [`skill.py`](skill.py) and exposed via ALFWorld’s `--mode`.

| Mode | Description |
|------|-------------|
| `gos` | Graph + vector seeds, graph reranking (default for ALFWorld) |
| `vector` | Vector similarity only (no graph rerank) |
| `all_full` | Entire skill library injected into context |
| `none` | No skills |

SkillsBench uses separate **conditions** (`graphskills`, `vectorskills`, `allskills`, `without`); see [skillsbench/README.md](skillsbench/README.md).

---

## ALFWorld

Interactive text games with GoS (or baselines) wired into the loop.

### Prerequisites

```bash
uv sync --extra alfworld
uv run alfworld-download   # game assets (~300 MB)
```

Ensure `data/skillsets/skills_200/` (or your chosen set) and a matching workspace exist — see [Workspace and skills directory pairing](#workspace-and-skills-directory-pairing).

### Environment

- **`API_KEY`** / **`BASE_URL`** — required; OpenAI-compatible chat completions (e.g. OpenRouter: `BASE_URL=https://openrouter.ai/api/v1`, `API_KEY=<openrouter key>`).
- **Embedding** — for `gos` / `vector`, set `GEMINI_*` / `OPENAI_*` / etc. in `.env` or the shell so `SkillModule` can call GoS; must match the workspace build.

### Workspace and skills directory pairing

The runner checks that the **last path component** of `--gos_workspace` is consistent with `--skills_dir` (e.g. `skills_200` ↔ `skills_200_v1` or `skills_200_*`). Recommended layout:

| `--skills_dir` | `--gos_workspace` |
|----------------|-------------------|
| `data/skillsets/skills_200` | `data/gos_workspace/skills_200_v1` |
| `data/skillsets/skills_500` | `data/gos_workspace/skills_500_v1` |
| `data/skillsets/skills_1000` | `data/gos_workspace/skills_1000_v1` |
| `data/skillsets/skills_2000` | `data/gos_workspace/skills_2000_v1` |

Defaults in `alfworld_run.py`: `skills_dir=data/skillsets/skills_200`, and if `--gos_workspace` is omitted in `gos`/`vector` mode, `data/gos_workspace/skills_200_v1`.

### Example: single game (GoS, eval OOD split)

From repo root:

```bash
API_KEY=<key> BASE_URL=https://openrouter.ai/api/v1 \
uv run python evaluation/alfworld_run.py \
  --model openai/gpt-4o-mini \
  --split eval_out_of_distribution \
  --use_skill \
  --mode gos \
  --gos_workspace data/gos_workspace/skills_200_v1 \
  --skills_dir data/skillsets/skills_200 \
  --max_workers 1 \
  --max_steps 30 \
  --max_games 1 \
  --exp_name my_smoke_test
```

Omit `--use_skill` for a no-skill baseline; set `--mode vector`, `all_full`, or `none` as needed.

---

## SkillsBench

Harbor-based benchmark: each task runs in Docker. GoS is exercised under the **graphskills** condition (prebuilt workspace + `graphskills-query`).

**Full guide:** [skillsbench/README.md](skillsbench/README.md) (install Harbor, generate task packs with `graphskills_benchmark.py`, batch configs).

**Minimal one-task flow** (copy-paste friendly): [README.md § Minimal verification](../README.md#minimal-verification).

Typical steps:

1. Generate task variants (example: one task, graph-skills only):

   ```bash
   uv run python evaluation/skillsbench/graphskills_benchmark.py \
     --skillset-name skills_200 \
     --task dialogue-parser \
     --skip-allskills --skip-vectorskills \
     --output-root evaluation/skillsbench/generated_my_run
   ```

2. Run Harbor from `evaluation/skillsbench/`:

   ```bash
   cd evaluation/skillsbench
   set -a && source ../../.env && set +a
   harbor run --agent gemini-cli \
     --model gemini/gemini-3-flash-preview \
     --force-build \
     -p generated_my_run/tasks_graph_skills/dialogue-parser \
     -o jobs/my-test-run
   ```

Adjust `--output-root`, `-p`, and agent/model to your setup.

---

## Embedding environment (all tracks using GoS)

Set variables so they match **how the workspace was indexed** (model + dimension):

```bash
# Example: Google Gemini embeddings
export GEMINI_API_KEY=<your-key>
export GOS_EMBEDDING_MODEL=gemini/gemini-embedding-001
export GOS_EMBEDDING_DIM=3072
```

```bash
# Example: OpenRouter embeddings
export OPENROUTER_API_KEY=<openrouter-key>
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export GOS_EMBEDDING_MODEL=openrouter/openai/text-embedding-3-large
export GOS_EMBEDDING_DIM=3072
```

```bash
# Example: OpenAI API (official, no custom base URL)
export OPENAI_API_KEY=<your-key>
export GOS_EMBEDDING_MODEL=openai/text-embedding-3-large
export GOS_EMBEDDING_DIM=3072
```

On **Azure**, use `openai/<deployment-name>` and the correct dimension; see [`.env.example`](../.env.example).

---

## Building or fetching a workspace

**Build locally:**

```bash
uv run gos index data/skillsets/skills_1000 \
  --workspace data/gos_workspace/skills_1000_v1 \
  --clear
```

**Or download prebuilt archives** (when published on the Hub):

```bash
./scripts/download_data.sh --workspace
```

Details and maintainer notes: [DATA.md](../DATA.md).

Verify:

```bash
uv run gos status --workspace data/gos_workspace/skills_1000_v1
```

---

## Result format and reporting

- **ALFWorld** — per-game JSON under your experiment directory; includes `reward`, steps, and usage fields from [`token_usage.py`](token_usage.py).
- **SkillsBench / Harbor** — `result.json` under `evaluation/skillsbench/jobs/<job>/<timestamp>/`.

Common concepts:

| Concept | Notes |
|---------|--------|
| `reward` | Task success in \([0,1]\) (definition varies by benchmark) |
| `retrieval_status` | `SKILL_HIT` / `NO_SKILL_HIT` where applicable (GoS / vector modes) |
| Timing | When reporting agent time, use **in-episode** intervals; exclude Docker image build and one-off setup unless stated |

---

## File reference

| Path | Role |
|------|------|
| `alfworld_run.py` | ALFWorld driver (multiprocessing, `SkillModule`) |
| `skill.py` | Unified retrieval adapter: GoS, vector, full-library, none; ALFWorld structured queries |
| `prompt_generator.py` | Prompt helpers |
| `token_usage.py` | Token accounting |
| `utils.py` | Shared helpers |
| `skills_ref/` | Skill markdown parsing / validation CLI |
| `alfworld/` | ALFWorld prompt templates |
| `skillsbench/` | SkillsBench + Harbor integration ([README](skillsbench/README.md)) |
