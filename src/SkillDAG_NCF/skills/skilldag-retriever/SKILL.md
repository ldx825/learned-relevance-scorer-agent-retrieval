---
name: skilldag-retriever
description: Retrieve a bounded bundle of relevant external skills from a prebuilt SkillDAG workspace. Use when the task may need specialized skills, scripts, or references that are not already obvious from current context, especially in containerized eval environments that mount a prebuilt SkillDAG graph.
inputs:
  - task description or subproblem summary
outputs:
  - a ranked list of skill ids, and full SKILL.md bodies on demand
compatibility:
  - claude-code
  - codex
  - gemini-cli
allowed-tools:
  - shell
---

# Purpose

Use this skill instead of manually browsing a large skill library.

It assumes the environment already provides:

- the `skilldag` CLI on PATH
- a prebuilt SkillDAG workspace reachable through the `skilldag` CLI

If that wiring is missing, read `references/container-layout.md`.

# Retrieve Relevant Skills

Retrieval is two steps: rank by query, then read the bodies you care about.

Construct the query yourself. Do not rely on the retrieval system to infer missing task structure for you.

A good query should usually include only the retrieval-critical fields that are actually known:

- the concrete goal
- the main artifact or file format
- the key operation or algorithm
- the required library, API, protocol, or tool name if known
- the verifier-critical constraint or invariant
- the task object being edited, parsed, generated, optimized, or validated

Keep it short, but make it specific. Prefer a compact noun/verb phrase over a long paragraph.

Good patterns:

```text
update embedded xlsx in pptx and preserve formulas
parallel tfidf indexing with processpoolexecutor deterministic ranking
civ6 district adjacency exact calculator for verifier
parse branching dialogue script into graph export
```

Bad patterns:

```text
please solve this task for me
I need help with a benchmark task
fix the project and make everything work
```

Run:

```bash
skilldag graph search "short specific query with goal + artifact + operation + constraint" --top-k 5
```

Then read each skill you want to use:

```bash
skilldag show <skill_id>
```

Related exploration commands:

```bash
skilldag graph get-dependencies <skill_id>     # depends_on / composes_with neighbors
skilldag graph get-alternatives <skill_id>     # similar_to / specializes neighbors
skilldag graph get-conflicts <skill_id>        # conflicts_with neighbors
```

# How To Use The Results

1. Start with a short task-level query via `skilldag graph search`.
2. Read the ranked list. If it is empty, explicitly state that no relevant skill was retrieved and continue on a no-skill path. Do not imply that you used a retrieved skill.
3. For each skill that looks relevant, run `skilldag show <skill_id>` to read the full SKILL.md body.
4. Inspect the task requirements, tests, and verifier first. Write down the minimum acceptance requirements before implementing.
5. Do not scan the filesystem for skill directories; retrieve skill bodies only through `skilldag show <id>`.
6. Follow the retrieved skill instructions when they directly help.
7. Use the retrieved skills to narrow the solution space. Prefer the shortest path to verifier pass, and prefer adapting an existing script or interface over inventing a broader replacement.
8. Re-query with a narrower subproblem if the task shifts.

# Guidance

- Prefer 1-2 targeted retrieval calls over scanning the whole library.
- Keep the query focused on the current task or subproblem, not the whole conversation history.
- Query content priority: `goal > artifact/format > operation/API > verifier constraint`.
- Include filenames, formats, protocols, or library names when they are part of the task signal.
- Include exact invariants when they matter, e.g. `preserve formulas`, `deterministic ranking`, `exact total`, `match verifier`.
- Do not include benchmark names, generic filler, or conversation meta-text unless they are truly task-relevant.
- If the result is too broad, narrow the query and reduce `--top-k`.
- If the result is empty, retry with simpler keywords before giving up.
- If no skill is retrieved after retrying, say so explicitly and solve without pretending a skill was used.
- After a skill hit, take the shortest path to verifier pass and satisfy only the verifier's minimum requirement first.
- Do not scan the filesystem for skill directories; use the `skilldag` CLI as the only skill retrieval surface.
- Do not add extra features, side outputs, UI panels, or refactors unless the task explicitly requires them.
- Treat retrieved skills as a constraint on implementation choices, not permission to explore more branches.

# Online Graph Edit

If you observe a repeated failure that clearly exposes a reusable structural error between two skills (not a one-off local state mistake), mutations follow a two-step flow.

**Step 1 — propose (dry-run; never mutates).** The response returns `{would, related}`. `related` lists every existing edge and recent history entry whose endpoints match the target pair — each with its reason — so you can read prior evidence before acting.

```bash
skilldag graph propose-edge    <source> <target> <type>                  --reason "<task evidence>"
skilldag graph propose-remove  <source> <target> <type>                  --reason "<task evidence>"
skilldag graph propose-retype  <source> <target> --from <old> --to <new> --reason "<task evidence>"
```

**Step 2 — commit (only mutation path).** `--reason` is required; `--task-id` is optional.

```bash
skilldag graph edit-edge add    <source> <target> <type>                   --reason "<task evidence>" --task-id "<task-id>"
skilldag graph edit-edge remove <source> <target> <type>                   --reason "<task evidence>" --task-id "<task-id>"
skilldag graph edit-edge retype <source> <target> --from <old> --to <new>  --reason "<task evidence>" --task-id "<task-id>"
```

Rules:

- Always propose first. Read `related` — prior reasons on the same pair tell you whether your new evidence is consistent, contradictory, or redundant.
- If new evidence contradicts a prior edge's reason, use `edit-edge retype` (or `edit-edge remove` first). Do not stack an opposing edge type on the same pair.
- Do not mutate after one weak failure.
- Only mutate when the relation is reusable — likely to recur in another task using the same two skills.
- Keep the reason short and grounded in observed failure from this task.
