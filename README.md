# Learned Task-Relevance Scoring for Agent Skills and Memories

**Code release accompanying an anonymous submission.**

This repository contains the implementation and experiment pipeline for a
lightweight, plug-and-play scorer that learns task–candidate relevance for
LLM-agent skill retrieval and procedural-memory retrieval. The scorer replaces
the fixed cosine (or graph-based) scoring stage of an existing retriever while
leaving the upstream retriever, the frozen content representations, and the
downstream agent unchanged.

---

## 1. Method at a glance

- **Content-based interaction scorer.** A GMF/MLP/NeuMF architecture
  (He et al., 2017) operating on *frozen* text embeddings instead of ID
  embeddings, so it can score unseen tasks and candidates.
- **Graded supervision distillation.** Training tasks and candidate documents
  are converted into candidate pools (semantic + structural + hard negatives),
  annotated with three executability grades (required / supportive /
  irrelevant-or-harmful) by an LLM judge, and distilled into the scorer with a
  pointwise BCE term plus a pairwise ranking term.
- **Two integration modes.** Cascade reranking (rescore a cosine shortlist) and
  calibrated fusion (z-score combination with the base retriever score).
- **Tiny footprint.** ~0.30M parameters for 3072-d embeddings (~0.15M for
  1536-d); CPU training; no online judge or extra LLM call at inference time.

## 2. Repository layout

```
src/GraphOfSkills_NCF/gos/ncf/    # scorer implementation (GMF / MLP / NeuMF), losses, training loop
src/GraphOfSkills_NCF/gos/core/   # personalized-PageRank skill-graph retrieval used by the GoS baseline
src/SkillDAG_NCF/src/skilldag/    # integration into the typed skill graph (reranker, calibration, fusion)
src/MemP_NCF/ProcedureMem/        # procedural-memory pipeline (script/trajectory/proceduralization) + reranker

scripts/ncf_v3/                   # shared training/evaluation pipeline for the phase-based supervision
scripts/skillbench_ncf/           # SkillsBench-1000 skill-retrieval experiments
scripts/memp/                     # ALFWorld and TravelPlanner procedural-memory experiments
scripts/ablations/                # architecture and training ablations
configs/                          # experiment configuration files
```

## 3. Environment

The scorer trains comfortably on CPU. Per-module dependency files are provided:

```bash
# scorer + skill-graph retrieval
pip install -r src/GraphOfSkills_NCF/pyproject.toml    # or: pip install -e src/GraphOfSkills_NCF

# skill-graph host system
pip install -e src/SkillDAG_NCF

# procedural-memory host system
pip install -r src/MemP_NCF/requirements.txt
```

Core requirements: Python 3.10+, PyTorch (CPU build is sufficient for scorer
training), NumPy. LLM agents (for end-to-end evaluation) additionally require
the API clients listed in the respective module dependency files.

## 4. Data and external API usage

Two kinds of assets are consumed:

1. **Public source data** (benchmarks, skill documents, task suites) — restored
   from pinned upstream commits by the preparation scripts
   (e.g. `scripts/memp/prepare_public_sources.sh` for the memory side).
2. **Embeddings and judge annotations** — produced once via API calls and cached
   locally. Cache directories are intentionally excluded from this release;
   they can be regenerated with the provided builders if API access is
   available:

   - embedding builders: `scripts/memp/build_memory_embedding_candidates.py`,
     `scripts/skillbench_ncf/prepare_neumf_full_v4.py`
   - judge/label builders: `scripts/ncf_v3/run_phase_template_judge.py`,
     `scripts/skillbench_ncf/build_graded_judge_full_v4.py`

All modeling scripts read only **train-derived** supervision; test labels and
execution outcomes are reserved for final evaluation and are never used for
model selection.

## 5. Reproducing the results

### 5.0 Result-to-code map

| Paper artifact | Entry points |
|---|---|
| Table 1, SkillsBench-1000 (intrinsic skill retrieval) | `scripts/skillbench_ncf/` (V4 pipeline: `build_graded_judge_full_v4.py` → `prepare_neumf_full_v4.py` → `train_neumf_full_v4.py`) |
| Table 1, ALFWorld-37 (intrinsic skill retrieval) | `scripts/ncf_v3/` (`build_phase_data.py` → `build_phase_candidates.py` → `prepare_phase_model_data.py` → `train_phase_neumf.py` → `evaluate_phase_neumf_test.py`) |
| Table 2 (intrinsic memory retrieval, full pool + cascade) | `scripts/memp/evaluate_memory_retriever_table.py` |
| Table 3 (end-to-end skill selection) | `scripts/skillbench_ncf/run_paper_skilldag_once_background.sh`, `scripts/run_paper_main_alfworld_once.sh` |
| Table 4, ALFWorld memory (Script / Trajectory / Proceduralization) | `scripts/memp/run_gpt4o_script_cosine_*.sh`, `run_gpt4o_trajectory_query_retrieval.sh`, `run_gpt4o_script_query_cosine_ncf.sh` |
| Table 4, TravelPlanner (Trajectory / Proceduralization, 180 validation tasks) | `scripts/memp/build_travelplanner_trajectory_quality_bank.py` → `build_travelplanner_trajectory_neumf_data.py` → `prepare_travelplanner_trajectory_retrieval.py` → `run_travelplanner_{trajectory,proceduralization}_q{41,84}_*.sh` → `evaluate_travelplanner_validation_slice.py` |
| Table 5 and Appendix D.2 (architectural ablation) | `scripts/ablations/` |
| Online TravelPlanner operation-level retrieval | `scripts/memp/travelplanner_atomic_memory.py` and the `*_travelplanner_atomic_*.py` builders |

### 5.1 Skill retrieval (ALFWorld-37 and SkillsBench-1000)

```bash
# supervision and arrays
python scripts/ncf_v3/build_phase_data.py ...          # frozen phase/task views
python scripts/ncf_v3/build_phase_candidates.py ...    # candidate pools
python scripts/ncf_v3/prepare_phase_model_data.py ...  # training arrays

# scorer training (GMF -> MLP -> NeuMF)
python scripts/ncf_v3/train_phase_neumf.py --arrays <arrays.npz> --output <dir>

# frozen-checkpoint evaluation
python scripts/ncf_v3/evaluate_phase_neumf_test.py --arrays <arrays.npz> --checkpoint <dir>/neumf.pt
```

SkillsBench-1000 uses the same recipe with the 1000-skill catalog
(`scripts/skillbench_ncf/train_neumf_full_v4.py`).

End-to-end ALFWorld runs (agent + retrieval):

```bash
bash scripts/run_paper_main_alfworld_once.sh
```

### 5.2 Procedural-memory retrieval (ALFWorld and TravelPlanner)

```bash
# train the memory scorer on frozen supervision
python scripts/memp/train_memory_neumf.py --arrays <arrays.npz> --output <dir>

# intrinsic retrieval tables
python scripts/memp/evaluate_memory_retriever_table.py ...

# TravelPlanner end-to-end: dynamic retrieval -> agent -> official constraint check
bash scripts/memp/run_travelplanner_atomic_typed_hard_validation10.sh
```

## 6. Hardware and runtime

Scorer training completes in seconds to a few minutes on a single CPU
(the reported recipe: 20 GMF epochs + 20 MLP epochs + early-stopped NeuMF
fine-tuning). Full-pool inference over the 300-memory ALFWorld bank costs
~0.25 ms per query after representations are cached. End-to-end agent runs
require an LLM API and, for SkillsBench, a Docker-capable environment.

## 7. Notes

- Candidate scoring is measured within the candidate pool produced by the host
  retriever; the release contains the exact candidate-construction scripts so
  the pools can be regenerated.
- Supervision labels are LLM-judge proxies for executability, not causal effect
  estimates.
- Regenerating embeddings and judge labels requires paid API access; the
  released code paths, prompts, and splits are sufficient to reproduce the
  pipeline end to end.

---

*This repository is anonymized for double-blind review. No author, institution,
or affiliation information is included.*
