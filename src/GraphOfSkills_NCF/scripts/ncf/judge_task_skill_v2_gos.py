#!/usr/bin/env python3
"""Run the resumable one-call Judge pilot or formal 700-task collection."""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project))
from gos.ncf.judge_v2 import run_gos_v2_judge

root = project.parents[1]
shared = root / ".runtime/skilldag_ncf/data"
out = root / ".runtime/gos_ncf"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--selection-path",
    type=Path,
    default=out / "data/reports/task_skill_v2_gos_pilot_selection.json",
)
parser.add_argument("--collection-name", default="generic_evidence_pilot")
parser.add_argument("--max-workers", type=int, default=4)
parser.add_argument("--max-attempts", type=int, default=3)
parser.add_argument("--limit", type=int)
parser.add_argument("--dry-run", action="store_true", help="Validate selection/candidates without API calls.")
parser.add_argument(
    "--formal-700",
    action="store_true",
    help="Use the existing stratified 100-per-task-type (700 task) selection.",
)
args = parser.parse_args()
if args.formal_700:
    args.selection_path = shared / "reports/judge_medium_v1_selection.json"
    if args.collection_name == "generic_evidence_pilot":
        args.collection_name = "generic_evidence_700"
if args.dry_run:
    selected = json.loads(args.selection_path.read_text())["record_ids"]
    candidate_task_ids = {
        json.loads(line)["task_record_id"]
        for line in (out / "data/intermediate/task_skill_v2_gos_candidates.jsonl").open()
        if line.strip()
    }
    missing = sorted(set(selected) - candidate_task_ids)
    print(json.dumps({
        "collection_name": args.collection_name,
        "selected_task_count": len(selected),
        "candidate_task_count": len(candidate_task_ids),
        "missing_selected_tasks": missing,
        "api_calls_made": 0,
        "ready": not missing,
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not missing else 2)

key = os.environ.get("SKILLDAG_LLM_API_KEY", "")
base = os.environ.get("SKILLDAG_LLM_BASE", "https://yunwu.ai/v1").rstrip("/")
model = os.environ.get("SKILLDAG_LLM_MODEL", "MiniMax-M2.7")
if not key:
    raise SystemExit("SKILLDAG_LLM_API_KEY is empty")


def chat(payload):
    request = urllib.request.Request(
        base + "/chat/completions",
        json.dumps(payload).encode(),
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


report = run_gos_v2_judge(
    shared / "raw/tasks.jsonl",
    shared / "raw/skills.jsonl",
    out / "data/intermediate/task_skill_v2_gos_candidates.jsonl",
    args.selection_path,
    out,
    model,
    chat,
    collection_name=args.collection_name,
    max_workers=args.max_workers,
    max_attempts=args.max_attempts,
    limit=args.limit,
)
print(json.dumps(report, ensure_ascii=False, indent=2))
