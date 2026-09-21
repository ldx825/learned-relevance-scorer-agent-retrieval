"""Centralized SkillDAG agent prompt text.

Two callers share this module so ALFWorld and SkillsBench stay byte-identical on:
  - how to call the skilldag CLI
  - when/why to reflect on failures
  - few-shot examples showing when to emit graph mutations

Callers:
  - ``benchmarks/alfworld/skilldag_runtime.py``
      ALFWORLD_SYSTEM_PROMPT uses ``cli_reference()`` + ``mutation_rules()``
      ``_skilldag_task_header()`` uses ``failure_reflection_protocol()`` +
      ``when_to_mutate()`` + ``few_shot()``
  - ``benchmarks/skillsbench/skilldag_benchmark.py``
      ``inject_skilldag_instruction_protocol()`` uses ``full_protocol()``
      to inline every section into per-task ``instruction.md``.

Mutation model (2026-05-04): two-step flow. ``propose-*`` is dry-run and
returns ``{would, related}`` — never mutates; ``--reason`` is required
as a forcing function. ``edit-edge`` is the only path that writes;
``--reason`` is required and ``--task-id`` is optional.
"""
from __future__ import annotations


CLI_REFERENCE = """Useful SkillDAG commands (stdout/stderr becomes your next observation):

  Read-only:
    skilldag graph search "<query>" [more queries...] --top-k 5
        → ranked list of skill ids + scores + 1-line descriptions.
          Multiple positional queries run as ONE batched embedding call and
          return {batch: true, queries: [...], results: [{query, matches, ...}, ...]};
          a single query returns the original {query, matches, ...} schema.
    skilldag show <skill_id> [<skill_id> ...]
        → full SKILL.md body for one or more skills (multiple ids in a single call returns all bodies)
    skilldag graph get-skill <skill_id>
        → node metadata (name, description, domain)
    skilldag graph get-dependencies <skill_id>
        → depends_on / composes_with neighbors
    skilldag graph get-alternatives <skill_id>
        → similar_to / specializes neighbors
    skilldag graph get-conflicts <skill_id>
        → conflicts_with neighbors

  Mutation is a two-step flow — propose (dry-run) first, commit second.

    Step 1 — propose. Returns {would, related}; never mutates. `related`
    lists every existing edge and recent history entry whose endpoints match
    the target pair (ignoring direction), each with its reason so you can
    read prior evidence before acting.
    --reason is required as a forcing function: state the evidence in <=30
    words before you see `related`. The same string can be reused at commit.

      skilldag graph propose-edge    <src> <tgt> <type>                --reason "<ev>"
      skilldag graph propose-remove  <src> <tgt> <type>                --reason "<ev>"
      skilldag graph propose-retype  <src> <tgt> --from <t1> --to <t2> --reason "<ev>"

    Step 2 — commit. Only path that writes. --reason is required.

      skilldag graph edit-edge add    <src> <tgt> <type>                   --reason "<ev>"
      skilldag graph edit-edge remove <src> <tgt> <type>                   --reason "<ev>"
      skilldag graph edit-edge retype <src> <tgt> --from <t1> --to <t2>    --reason "<ev>"

  Edge types: depends_on | composes_with | similar_to | conflicts_with | specializes"""


FAILURE_REFLECTION_PROTOCOL = """SkillDAG failure-reflection protocol:
1. When an action or command fails, infer the missing precondition or wrong assumption before doing anything else.
2. Ask whether the failure is reusable across tasks:
   - Reusable structural error between two skills -> mutate the graph.
   - One-off world-state confusion or local mistake -> do not mutate; inspect state or try a different action/command.
3. Repeating the same failed action or near-identical search query is not progress.
4. Retrieval is free and interruptible: `skilldag show` / `search` / `get-*` calls do NOT consume your env-action budget, so you may consult skills at any point during the task — including mid-execution after an unexpected observation. Use `skilldag show A B C` to read several skills in one call when convenient.
5. Mutation reasons must cite concrete evidence from THIS task in <=30 words."""


WHEN_TO_MUTATE = """When to mutate:
- `depends_on`: skill A keeps failing until skill B's setup/output should happen first.
- `composes_with`: A and B should be chained, but the current graph does not expose that composition.
- `conflicts_with`: two skills suggest incompatible or redundant procedures and jointly mislead execution.
- `edit-edge remove` / `edit-edge retype`: an existing relation is clearly contradicted by repeated failure evidence.
  Prefer `retype` over adding a second contradictory edge on the same pair.

CRITICAL — depends_on direction:
  `edit-edge add A B depends_on` = A requires B first.
  Source=A (dependent / failing skill); Target=B (prerequisite).
  Chronology may be B→A, but edge is A→B.
  Do NOT copy failure narrative order.
  If evidence says "A failed because B was missing", write `add A B depends_on`.
  Pre-commit check: read "A requires B first"; if backward, swap A/B."""


FEW_SHOT = """Few-shot guidance:

Example 1, do NOT mutate:
  Observation: Nothing happens (or: command returned a simple error you have not inspected).
  You tried an action once and have not yet checked state/logs/input assumptions.
  Good next step:
    {"thought": "The failure may be a local state mismatch, so I should inspect state before editing the graph.", "action": "go to countertop"}

Example 2, DO mutate — propose first, then commit:
  Observation: A handoff between two loaded skills failed in a way you find reasonable to encode (e.g. one skill plainly needs the other's setup output).
  Step A (propose; never mutates — `--reason` is the evidence you stake before seeing prior history):
    {"thought": "Looks reasonable to encode this as depends_on; I'll propose with my evidence first and then read any prior reasons.", "command": "skilldag graph propose-edge <skill_a> <skill_b> depends_on --reason \\"<concise evidence from this task>\\""}
  Next observation contains {"result": {"would": {..., "reason": "<your reason>"}, "related": []}} — empty related, safe to commit.
  Step B (commit; the same reason can be reused):
    {"command": "skilldag graph edit-edge add <skill_a> <skill_b> depends_on --reason \\"<concise evidence from this task>\\""}

Example 3, propose surfaces a contradiction — retype instead of add:
  Observation from `propose-edge`: `related` contains a prior depends_on(A, B) with reason "earlier reasoning" written by another task.
  Your new evidence says A+B actually conflict. Adding a second, contradictory edge is wrong; retype the existing one.
    {"command": "skilldag graph edit-edge retype <skill_a> <skill_b> --from depends_on --to conflicts_with --reason \\"<current evidence contradicting earlier>\\""}"""


SUCCESS_PATH_DISCOVERY = """SkillDAG success-path structural discovery:

Graph mutation is NOT gated on failure. When a successful task reveals a
structural relation between two skills that the graph does not currently
encode, propose the missing edge — the graph should reflect what worked,
not only what broke.

After finishing a task (success OR failure), perform this check:

  1. Identify the skills you actually used in sequence: S1 → S2 → ... → Sn.
  2. For each adjacent or causally-linked pair (Si, Sj) where Sj materially
     contributed to Si's progress — Sj prepared a precondition, located a
     target, validated a step, or supplied a prior that Si consumed —
     verify whether the relation is already in the graph:
         skilldag graph get-dependencies <Si>     # for depends_on
         skilldag graph get-alternatives <Si>     # for similar_to / specializes
  3. If the relation is absent, propose it with a one-sentence reason
     pointing at the concrete step in THIS trajectory that exposed it.

Direction reminder (same rule as failure-driven mutation, see `When to mutate`):
  `propose-edge Si Sj depends_on` ⇔ "Si needs Sj first".
  Source = the consumer; Target = the provider.

Heuristic: a proposal triggered by SUCCESS carries the same evidentiary
weight as one triggered by FAILURE — both are observations about how skills
collaborate. The threshold is identical: concrete evidence, ≤30 words,
correct direction.

Skip when no novel relation surfaced (every collaborator was already wired
in). Do NOT invent edges to satisfy a quota."""


SEARCH_PROTOCOL = """SkillDAG search protocol — decompose composite tasks:

For multi-phase tasks ("clean X and put in Y", "heat X and put in Y", "cool X
and put in Y", "look at X under Y"), issue ONE focused search per phase
instead of a single combined query.

`skilldag graph search` accepts MULTIPLE queries in ONE call (positional args,
each separately quoted). Multiple queries run as a batch — one embedding API
roundtrip — and return a list aligned with input order. Prefer batch over
sequential single-query calls whenever the phases are known up-front.

  Recommended batch form (one call, three phases):
      skilldag graph search "find <object>" "<verb> <object>" "put <object> in <receptacle>" --top-k 5

  Per-phase intent:
    - LOCATE phase  ("find <object>") → surfaces locator / finder / searcher
      skills carrying object-location priors. Composite queries like
      "clean pan stoveburner" miss these because top-K gets dominated by
      heater/cleaner skills.
    - TRANSFORM phase ("<verb> <object>") → clean / heat / cool / wash;
      surfaces state-modifier skills.
    - PLACE phase ("put <object> in <receptacle>") → surfaces
      transporter / storer / placer skills.

Output schema:
  - 1 query  → original {result: {query, matches, neighbors, conflicts}}.
  - >1 query → batch  {result: {batch: true, queries: [...],
                                results: [{query, matches, neighbors, conflicts}, ...]}}.

Rule: if your first env action would be `go to <recep>` to find an object,
you should run a LOCATE-phase search first (or a batch including LOCATE).
Do NOT enter physical DFS over cabinets/drawers without checking whether a
locator skill exists in the graph."""


MUTATION_RULES = """Rules:
- Propose whenever you have thought about it and the relation looks reasonable. `propose-*` is a pure dry-run — it never writes, so the bar is your own judgement, not a quota of failures.
- `--reason` is required on propose. State the evidence (<=30 words) BEFORE you see `related`; this forces you to articulate the case rather than rationalize after the fact.
- After propose, read the `related` list. If a prior edge or history entry on the same pair exists, its reason tells you whether your new evidence is consistent, contradictory, or redundant.
- If your new evidence contradicts a prior edge's reason, use `edit-edge retype` (or `edit-edge remove` first) — do NOT stack an opposing edge type on the same pair.
- Do not keep issuing near-duplicate retrieval queries without explaining why the previous retrieval was insufficient.
- Keep the reason short and grounded in observed evidence from this task."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def cli_reference() -> str:
    return CLI_REFERENCE


def failure_reflection_protocol() -> str:
    return FAILURE_REFLECTION_PROTOCOL


def when_to_mutate() -> str:
    return WHEN_TO_MUTATE


def few_shot() -> str:
    return FEW_SHOT


def mutation_rules() -> str:
    return MUTATION_RULES


def search_protocol() -> str:
    return SEARCH_PROTOCOL


def success_path_discovery() -> str:
    return SUCCESS_PATH_DISCOVERY


def full_protocol() -> str:
    """Concatenate every shared section in canonical order. Used directly by
    SkillsBench's ``inject_skilldag_instruction_protocol`` to inline the same
    content that ALFWorld's runner assembles piecewise.
    """
    return "\n\n".join([
        cli_reference(),
        search_protocol(),
        failure_reflection_protocol(),
        success_path_discovery(),
        when_to_mutate(),
        few_shot(),
        mutation_rules(),
    ])
