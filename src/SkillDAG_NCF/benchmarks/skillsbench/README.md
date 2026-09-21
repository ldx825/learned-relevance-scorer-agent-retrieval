# `benchmarks/skillsbench/`

The SkillDAG SkillsBench arm has two halves:

1. **Task variant generator** — wraps the upstream 87 SkillsBench tasks with
   a SkillDAG-aware Docker template. Lives at
   `benchmarks/skillsbench/skilldag_benchmark.py`.
   Each generated task ships a per-scale `skillgraph.json` mounted into
   `/var/lib/skilldag/runtime/` inside the container.

2. **Harbor runner config** — [`configs/skillsbench/skilldag.yaml`](../../configs/skillsbench/skilldag.yaml).
   This is an `envsubst` template; `scripts/run_skillsbench.sh` renders it
   with the env vars from `.env` before passing to the `harbor` binary.

## Reproduce

```bash
SKILLDAG_SCALE=200  bash scripts/run_skillsbench.sh    # default
SKILLDAG_SCALE=1000 SKILLDAG_WORKERS=5 bash scripts/run_skillsbench.sh
```

`run_skillsbench.sh` will (a) generate `results/skillsbench_tasks/tasks_skilldag_full_<SCALE>/` if
missing, (b) render the YAML, (c) call `harbor run`, (d) score via
`analysis/score_skillsbench_gos.py`.

## Prerequisites

The Harbor framework binary must be on `PATH`. See
[`docs/reproducing.md`](../../docs/reproducing.md) §Harbor install.
