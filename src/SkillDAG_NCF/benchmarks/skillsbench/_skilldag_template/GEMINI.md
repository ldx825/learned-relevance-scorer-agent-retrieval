# Task Environment

This environment contains a prebuilt **SkillDAG** typed-skill graph workspace.

## Required First Step

Before writing any code, retrieve relevant skills with two shell commands:

```bash
skilldag graph search "short description of what you need to do" --top-k 5
skilldag show <skill_id>
```

Use `run_shell_command` to execute these. Both are on PATH inside the container. The first returns a ranked list of skill ids; the second prints the SKILL.md body. Pass multiple ids — `skilldag show A B C` — to read several skills in one call. Retrieval is free and interruptible: you may search or show more skills later if the task changes shape.

## Reading the Output

`skilldag show <skill_id>` is the only supported way to read skill bodies. Do not scan the filesystem for skill directories or infer private SkillDAG storage paths.

## Failure Reflection

SkillDAG failure-reflection protocol:

1. When an action or command fails, infer the missing precondition or wrong assumption before doing anything else.
2. Ask whether the failure is reusable across tasks:
   - Reusable structural error between two skills → mutate the graph.
   - One-off world-state confusion or local mistake → do not mutate; inspect state or try a different action/command.
3. Repeating the same failed action or near-identical search query is not progress.
4. Retrieval is free and interruptible: `skilldag show` / `search` / `get-*` calls do NOT consume your env-action budget, so you may consult skills at any point during the task — including mid-execution after an unexpected observation.
5. Mutation reasons must cite concrete evidence from THIS task in ≤30 words.

## When to Mutate

- `depends_on`: skill A keeps failing until skill B's setup/output should happen first.
- `composes_with`: A and B should be chained, but the current graph does not expose that composition.
- `conflicts_with`: two skills suggest incompatible or redundant procedures and jointly mislead execution.
- `edit-edge remove` / `edit-edge retype`: an existing relation is clearly contradicted by repeated failure evidence. Prefer `retype` over adding a second contradictory edge on the same pair.

CRITICAL — `depends_on` direction:

- `edit-edge add A B depends_on` = A requires B first. Source = A (dependent / failing skill); Target = B (prerequisite).
- Chronology may be B → A, but edge is A → B.
- If evidence says "A failed because B was missing", write `add A B depends_on`.
- Pre-commit check: read "A requires B first"; if backward, swap A/B.

## Online Graph Edit

Mutation is a two-step flow — propose (dry-run) first, commit second.

**Step 1 — propose.** Returns `{would, related}`; never mutates. `related` lists every existing edge and recent history entry whose endpoints match the target pair (ignoring direction), each with its reason so you can read prior evidence before acting. `--reason` is required as a forcing function: state the evidence in ≤30 words BEFORE you see `related`. The same string can be reused at commit.

```bash
skilldag graph propose-edge    <src> <tgt> <type>                --reason "<evidence>"
skilldag graph propose-remove  <src> <tgt> <type>                --reason "<evidence>"
skilldag graph propose-retype  <src> <tgt> --from <t1> --to <t2> --reason "<evidence>"
```

**Step 2 — commit.** Only path that writes. `--reason` is required.

```bash
skilldag graph edit-edge add    <src> <tgt> <type>                   --reason "<evidence>"
skilldag graph edit-edge remove <src> <tgt> <type>                   --reason "<evidence>"
skilldag graph edit-edge retype <src> <tgt> --from <t1> --to <t2>    --reason "<evidence>"
```

Edge types: `depends_on | composes_with | similar_to | conflicts_with | specializes`.

Rules:

- Propose whenever you have thought about it and the relation looks reasonable. `propose-*` is a pure dry-run — it never writes, so the bar is your own judgement, not a quota of failures.
- `--reason` is required on propose. State the evidence (≤30 words) BEFORE you see `related`; this forces you to articulate the case rather than rationalize after the fact.
- After propose, read the `related` list. If a prior edge or history entry on the same pair exists, its reason tells you whether your new evidence is consistent, contradictory, or redundant.
- If your new evidence contradicts a prior edge's reason, use `edit-edge retype` (or `edit-edge remove` first) — do NOT stack an opposing edge type on the same pair.
- Do not keep issuing near-duplicate retrieval queries without explaining why the previous retrieval was insufficient.
- Keep the reason short and grounded in observed evidence from this task.

## Few-Shot

Example 1, do NOT mutate:

- Observation: a single command returned a simple error you have not yet inspected.
- You tried an action once and have not yet checked state, logs, or input assumptions.
- Next step: inspect state before editing the graph.

Example 2, DO mutate — propose first, then commit:

- Observation: a handoff between two loaded skills failed in a way you find reasonable to encode (e.g. one skill plainly needs the other's setup output).
- Step A (propose; never mutates — `--reason` is the evidence you stake before seeing prior history):

  ```bash
  skilldag graph propose-edge <skill_a> <skill_b> depends_on --reason "<concise evidence from this task>"
  ```

- If the next response shows `{"result": {"would": {..., "reason": "<your reason>"}, "related": []}}` — empty related, safe to commit.
- Step B (commit; the same reason can be reused):

  ```bash
  skilldag graph edit-edge add <skill_a> <skill_b> depends_on --reason "<concise evidence from this task>"
  ```

Example 3, propose surfaces a contradiction — retype instead of add:

- Observation from `propose-edge`: `related` contains a prior `depends_on(A, B)` with reason "earlier reasoning" written by another task.
- Your new evidence says A+B actually conflict. Adding a second, contradictory edge is wrong; retype the existing one.

  ```bash
  skilldag graph edit-edge retype <skill_a> <skill_b> --from depends_on --to conflicts_with --reason "<current evidence contradicting earlier>"
  ```

## Path Notes

- Task data is available at both `/data/` (shell access) and `/root/data/` (file tool access)
- Write output files to `/root/output/<filename>` or use `run_shell_command` to write to `/output/<filename>`

## Workflow

1. Run `skilldag graph search "<task description>" --top-k 5` via `run_shell_command`
2. For each relevant id, run `skilldag show <id>` to read the full body (use multi-id form when several look relevant)
3. Inspect the task tests/verifier and identify the minimum acceptance requirements
4. Prefer the shortest path to verifier pass; do not open extra implementation branches unless the verifier forces them
5. Use retrieved interfaces only when they directly help satisfy the minimum requirements
