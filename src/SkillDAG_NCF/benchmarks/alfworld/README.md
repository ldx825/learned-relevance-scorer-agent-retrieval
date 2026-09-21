# `benchmarks/alfworld/`

The SkillDAG ALFWorld arm is bundled here and invoked by
`scripts/run_alfworld.sh`:

| What | Where |
|---|---|
| Eval entry point | `benchmarks/alfworld/run_alfworld.py` |
| Runtime loop | `benchmarks/alfworld/skilldag_runtime.py` |
| Prompt templates | `benchmarks/shared/skilldag_prompt.py` |
| ALFWorld task data | expected under `data/alfworld/data/` or `ALFWORLD_DATA` |

## Reproduce

`scripts/setup.sh` installs the Python prerequisites (`alfworld`, `litellm`,
`pyyaml`, and `retry`). Set `ALFWORLD_DATA` if the upstream ALFWorld data is not
under `data/alfworld/data/`.

```bash
bash scripts/run_alfworld.sh                 # full 140-task dev split
MAX_GAMES=10 bash scripts/run_alfworld.sh    # smoke test
```

See [`scripts/run_alfworld.sh`](../../scripts/run_alfworld.sh) for the exact
command-line flags and [`docs/reproducing.md`](../../docs/reproducing.md) for
the full walkthrough including the initialization step.
