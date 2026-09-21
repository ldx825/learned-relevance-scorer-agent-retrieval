# SkillDAG graph artifacts

This directory is intentionally kept lightweight in Git.

The actual graph artifacts used for reproduction are downloaded on demand:

```bash
bash scripts/download_data.sh --graphs
```

That command fetches the authoritative paper artifacts from HuggingFace dataset `Eric068/SkillDAG`:

- `skillgraph_200.json` + embeddings
- `skillgraph_500.json` + embeddings
- `skillgraph_1000.json` + embeddings
- `skillgraph_2000.json` + embeddings
- `skillgraph_alfworld.json` + embeddings

If you want to regenerate graphs locally instead of downloading the published ones,
see `docs/reproducing.md` and `docs/paper_reproduction.md`.
