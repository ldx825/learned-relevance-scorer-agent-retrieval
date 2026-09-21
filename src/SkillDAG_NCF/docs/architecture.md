# SkillDAG — architecture overview

## What it is

SkillDAG is a typed, dependency-aware skill graph layered on top of a large
skill corpus (e.g. the 200 / 500 / 1000 / 2000 skill libraries from
[`DLPenn/graph-of-skills-data`](https://huggingface.co/datasets/DLPenn/graph-of-skills-data)).

At inference time, an agent calls `skilldag graph search "<task description>"`
to receive a small, dependency-aware bundle of relevant skills. Beyond
retrieval, the agent can **edit the graph online** through a propose/commit
protocol — every executed task can leave evidence behind for the next.

## Single source of truth: `skillgraph.json`

```json
{
  "nodes": { "<skill_id>": { ... metadata ... } },
  "edges": [
    { "source": "...", "target": "...", "type": "depends_on",     "reason": "...", "evidence": [...] },
    { "source": "...", "target": "...", "type": "composes_with",  "reason": "...", "evidence": [...] },
    { "source": "...", "target": "...", "type": "similar_to",     "reason": "..." },
    { "source": "...", "target": "...", "type": "specializes",    "reason": "..." },
    { "source": "...", "target": "...", "type": "conflicts_with", "reason": "...", "subtype": "..." }
  ]
}
```

Edge taxonomy:

| Type | Meaning | Where it comes from |
|---|---|---|
| `similar_to` | A and B are functionally redundant; using both wastes context. **Symmetric.** | Initialization / online edits |
| `specializes` | A is a refinement / specialization of B. | Initialization / online edits |
| `depends_on` | A requires B as a precondition (true ordering constraint). | Initialization / online edits |
| `composes_with` | A and B work well together (positive co-use). | Initialization / online edits |
| `conflicts_with` | A + B together causes predictable failure. **5 sub-types**: `resource_exclusion`, `state_contamination`, `protocol_conflict`, `artifact_clobber`, `condition_triggered`. | Online (only inferable from execution evidence) |

The line between `similar_to` and `conflicts_with`: *does co-use cause
predictable success degradation?* If yes → `conflicts_with`; if it's only
wasteful (no harm) → `similar_to`.

## Two phases

### Cold start — `skilldag initialize-graph`

```
SKILL.md files
  → text-embedding-3-large  → embedding cache
  → LLM pair classification → typed positive edges
  → skillgraph.json
```

Implementation: [`src/skilldag/initialize.py`](../src/skilldag/initialize.py).

### Online — agent-driven edit protocol

The agent calls `skilldag graph propose-edge` (dry run, returns `{would, related}`)
then `skilldag graph edit-edge` (commits). Both require `--reason` as a
forcing function.

```
agent run
  ├── skilldag graph search "<query>"  →  {matches, neighbors, conflicts}
  ├── ... agent uses skills, observes outcomes ...
  └── skilldag graph edit-edge add <src> <tgt> <type> --reason "..." --task-id "..."
```

Three edit ops: `add`, `remove`, `retype`. Single-evidence commit (no
threshold) — the propose/commit split is itself the quality gate.

Implementation: [`src/skilldag/graph.py`](../src/skilldag/graph.py).

## Search semantics

`skilldag graph search <query> --top-k K --depth D` returns three
non-overlapping sections:

```json
{
  "matches":   [...],   // top-K by cosine on text-embedding-3-large
  "neighbors": [...],   // BFS depth-D from matches over WALKABLE edges
  "conflicts": [...]    // 1-hop conflicts_with edges incident to the union (pruning signal)
}
```

Walkable edge types: `similar_to`, `specializes`, `composes_with`, `depends_on`.
`conflicts_with` is **never traversed** — it only signals "do not co-use",
surfaced separately in the `conflicts` section.

## CLI surface

```
skilldag show <skill_id>           # print SKILL.md (with graph context header)
skilldag add  <skill_id>... --to D # copy skill dirs into a workspace
skilldag graph search   <query>
skilldag graph get-{skill,dependencies,conflicts,alternatives}
skilldag graph {check,expand,repair}-set <skill_ids...>
skilldag graph propose-{edge,remove,retype}    # dry run
skilldag graph edit-edge {add,remove,retype}   # commit
skilldag initialize-graph                       # build skillgraph.json from a skills dir
```

`skilldag help` prints the full reference.
