#!/usr/bin/env python3
import json
import sys
from pathlib import Path
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project))
from gos.ncf.data_v2 import build_gos_v2_candidates
from gos.ncf.alfworld_policy import (
    CONTRASTIVE_SKILL_GROUPS,
    CONTRASTIVE_TRIGGER_PATTERNS,
)

root = project.parents[1]
shared = root / ".runtime/skilldag_ncf/data"
report = build_gos_v2_candidates(
    shared / "raw/tasks.jsonl", shared / "raw/skills.jsonl",
    shared / "cache/task_embeddings/train",
    root / ".runtime/skilldag/data/skilldag/skilldag_graphs/skillgraph_alfworld.embeddings.json",
    root / ".runtime/gos_ncf",
    contrastive_skill_groups=CONTRASTIVE_SKILL_GROUPS,
    contrastive_trigger_patterns=CONTRASTIVE_TRIGGER_PATTERNS,
)
print(json.dumps(report, ensure_ascii=False, indent=2))
