# SkillDAG SkillsBench Template

This template is overlaid onto each canonical SkillsBench task by
`benchmarks/skillsbench/skilldag_benchmark.py`.

The generated task mounts three host paths into the container:

- SkillDAG package source at `/opt/skilldag/package`
- Per-task mutable graph workspace at `/var/lib/skilldag/runtime`
- Skill library bodies at `/var/lib/skilldag/bodies`

The Dockerfile patch writes `/usr/local/bin/skilldag`, so agents can call:

```bash
skilldag graph search "task intent" --top-k 5
skilldag show <skill_id>
skilldag graph propose-edge <src> <tgt> depends_on --reason "<evidence>"
skilldag graph edit-edge add <src> <tgt> depends_on --reason "<evidence>"
```

Generate and run through the top-level script:

```bash
SKILLDAG_SCALE=200 bash scripts/run_skillsbench.sh
```

Direct generator invocation:

```bash
python benchmarks/skillsbench/skilldag_benchmark.py \
  --tasks-root data/tasks/tasks \
  --skills-root data/skillsets/skills_200/skills_200 \
  --skillgraph-path data/skilldag_graphs/skillgraph_200.json \
  --skilldag-package-root . \
  --output-root results/skillsbench_tasks/tasks_skilldag_full_200
```
