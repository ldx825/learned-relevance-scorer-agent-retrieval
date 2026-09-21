# `data/`

This repository does **not** commit benchmark data or large paper artifacts.

After a fresh clone, populate `data/` with the bundled scripts:

```bash
bash scripts/download_data.sh --graphs --skillsets --tasks --alfworld-skills
# or simply:
bash scripts/setup.sh
```

## Expected layout after download/setup

```text
data/
├── skillsets/                  # skill libraries (200/500/1000/2000)
│   └── skills_<N>/...
├── alfworld_skills/            # 37-skill ALFWorld pool
├── gos_workspace/              # optional upstream GoS workspaces
│   └── skills_<N>_v1/...
├── tasks/                      # latest downloaded SkillsBench task set
│   └── tasks/<task>/...
└── skilldag_graphs/            # downloaded SkillDAG graph artifacts
    ├── skillgraph_<N>.json
    ├── skillgraph_<N>.embeddings.json
    ├── skillgraph_alfworld.json
    └── skillgraph_alfworld.embeddings.json
```

## Sources

| Subdir | Source | Notes |
|---|---|---|
| `skilldag_graphs/` | HuggingFace `Eric068/SkillDAG` | authoritative paper graph artifacts |
| `skillsets/` | HuggingFace `Eric068/SkillDAG` | packaged skill libraries used by this repo |
| `tasks/` | latest `benchflow-ai/skillsbench` tasks/ tree | fetched by `scripts/download_data.sh --tasks` |
| `gos_workspace/` | HuggingFace `DLPenn/graph-of-skills-data` | optional upstream GoS workspaces |
| `alfworld_skills/` | HuggingFace `Eric068/SkillDAG` | packaged ALFWorld skill pool |
