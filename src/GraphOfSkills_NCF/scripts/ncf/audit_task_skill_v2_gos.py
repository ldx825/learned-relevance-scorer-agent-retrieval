#!/usr/bin/env python3
import json
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project))
from gos.ncf.audit_v2 import audit_gos_v2_labels

root = project.parents[1]
report = audit_gos_v2_labels(
    root / ".runtime/skilldag_ncf/data/raw/tasks.jsonl",
    root / ".runtime/gos_ncf/data/labels/task_skill_v2_gos_signature_pilot.jsonl",
    root / ".runtime/gos_ncf/data/reports/task_skill_v2_gos_semantic_audit.json",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
