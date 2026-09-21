#!/usr/bin/env python3
"""Clean the formal GoS V2 labels and prepare local NeuMF arrays."""

from __future__ import annotations

import json
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[2]
root = project.parents[1]
sys.path.insert(0, str(project))

from gos.ncf.audit_v2 import audit_gos_v2_labels
from gos.ncf.dataset_v2 import clean_gos_v2_labels, prepare_gos_v2_arrays


raw = root / ".runtime/skilldag_ncf/data"
gos = root / ".runtime/gos_ncf/data"
labels = gos / "labels/task_skill_v2_gos_generic_evidence_700.jsonl"
cleaned = gos / "labels/task_skill_v2_gos_generic_evidence_700_clean.jsonl"
cleaning_report = gos / "reports/task_skill_v2_gos_cleaning_report.json"

clean_report = clean_gos_v2_labels(
    raw / "raw/tasks.jsonl",
    labels,
    gos / "intermediate/task_skill_v2_gos_candidates.jsonl",
    cleaned,
    cleaning_report,
)
audit_report = audit_gos_v2_labels(
    raw / "raw/tasks.jsonl",
    cleaned,
    gos / "reports/task_skill_v2_gos_clean_semantic_audit.json",
)
arrays_report = prepare_gos_v2_arrays(
    raw / "raw/tasks.jsonl",
    raw / "raw/skills.jsonl",
    cleaned,
    raw / "datasets/task_skill_v1/splits.json",
    raw / "cache/task_embeddings/train",
    root
    / ".runtime/skilldag/data/skilldag/skilldag_graphs/skillgraph_alfworld.embeddings.json",
    gos / "model_data/task_skill_v2_gos_clean",
)
print(
    json.dumps(
        {
            "cleaning": {
                key: value
                for key, value in clean_report.items()
                if key not in {"changes", "outputs"}
            },
            "audit": {
                key: value
                for key, value in audit_report.items()
                if key not in {"violations", "primary_skill_frequency"}
            },
            "arrays": arrays_report,
        },
        ensure_ascii=False,
        indent=2,
    )
)
