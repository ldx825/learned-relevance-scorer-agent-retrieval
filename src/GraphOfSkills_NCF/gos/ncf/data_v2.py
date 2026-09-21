"""Leakage-free candidate construction for the GoS task-skill V2 dataset."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np

DATASET_VERSION = "task_skill_v2_gos"
STOP = {"the", "and", "for", "from", "with", "into", "some", "object"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows))
    tmp.replace(path)


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def _tokens(text: str) -> set[str]:
    return {x.rstrip("s") for x in re.findall(r"[a-z0-9]+", text.lower()) if len(x) >= 3 and x not in STOP}


def _lexical(q: set[str], fields: tuple[set[str], set[str], set[str]]) -> float:
    if not q:
        return 0.0
    name, desc, body = fields
    return (1.25*len(q & name) + .9*len(q & desc) + .2*len(q & body)) / len(q)


def _task_vectors(shards: Path, model: str) -> dict[str, list[float]]:
    result = {}
    for path in sorted(shards.glob("*.json")):
        for key, entry in json.loads(path.read_text()).items():
            if entry["model"] != model:
                raise ValueError(f"embedding model mismatch: {path}")
            result[key] = entry["embedding"]
    return result


def build_gos_v2_candidates(
    tasks_path: Path | str, skills_path: Path | str, task_embedding_shards: Path | str,
    skill_embeddings_path: Path | str, output_root: Path | str, *,
    embedding_model: str = "text-embedding-3-large", semantic_top_k: int = 12,
    lexical_top_k: int = 12, pilot_samples_per_task_type: int = 2,
    contrastive_skill_groups: dict[str, tuple[str, ...]] | None = None,
    contrastive_trigger_patterns: dict[str, str] | None = None,
) -> dict[str, Any]:
    tasks = sorted((x for x in load_jsonl(Path(tasks_path)) if x.get("split") == "train"),
                   key=lambda x: (x.get("task_type", ""), x["record_id"]))
    skills = {x["skill_id"]: x for x in load_jsonl(Path(skills_path))}
    task_vecs = _task_vectors(Path(task_embedding_shards), embedding_model)
    raw = json.loads(Path(skill_embeddings_path).read_text())
    skill_vecs = {k: v["embedding"] for k, v in raw.items() if v["model"] == embedding_model}
    if set(skills) != set(skill_vecs):
        raise ValueError("skill table and vectors differ")
    skill_order = sorted(skill_vecs)
    skill_matrix = np.asarray([skill_vecs[x] for x in skill_order], dtype=np.float32)
    skill_matrix /= np.maximum(np.linalg.norm(skill_matrix, axis=1, keepdims=True), 1e-12)
    lexical_fields = {
        sid: (_tokens(sid + " " + skill.get("name", "")),
              _tokens(skill.get("description", "")), _tokens(skill.get("skill_body", "")))
        for sid, skill in skills.items()
    }
    rows, counts, sources = [], [], Counter()
    contrastive_skill_groups = contrastive_skill_groups or {}
    contrastive_trigger_patterns = contrastive_trigger_patterns or {}
    for task in tasks:
        tid, text = task["record_id"], task["task_text"]
        task_vector = np.asarray(task_vecs[tid], dtype=np.float32)
        task_vector /= max(float(np.linalg.norm(task_vector)), 1e-12)
        scores = skill_matrix @ task_vector
        sem = sorted(zip(skill_order, (float(x) for x in scores)),
                     key=lambda x: (-x[1], x[0]))[:semantic_top_k]
        query_tokens = _tokens(text)
        lex = sorted(((sid, _lexical(query_tokens, lexical_fields[sid])) for sid in skills),
                     key=lambda x: (-x[1], x[0]))
        lex = [x for x in lex if x[1] > 0][:lexical_top_k]
        sr, ss = {x:i+1 for i,(x,_) in enumerate(sem)}, dict(sem)
        lr, ls = {x:i+1 for i,(x,_) in enumerate(lex)}, dict(lex)
        hard = set()
        for trigger, ids in contrastive_skill_groups.items():
            pattern = contrastive_trigger_patterns.get(trigger)
            if pattern and re.search(pattern, text.lower()):
                hard.update(x for x in ids if x in skills)
        ids = set(sr) | set(lr) | hard
        counts.append(len(ids))
        for sid in sorted(ids, key=lambda x: (sr.get(x, 999), lr.get(x, 999), x)):
            src = (["gos_semantic_seed_candidate"] if sid in sr else []) + \
                  (["gos_lexical_seed_candidate"] if sid in lr else []) + \
                  (["counterfactual_hard_negative_candidate"] if sid in hard else [])
            sources.update(src)
            rows.append({"dataset_version": DATASET_VERSION, "task_record_id": tid,
                         "task_type": task.get("task_type", ""), "task_text": text,
                         "skill_id": sid, "candidate_sources": src,
                         "semantic_rank": sr.get(sid), "semantic_cosine": ss.get(sid),
                         "lexical_rank": lr.get(sid), "lexical_score": ls.get(sid),
                         "is_counterfactual_candidate": sid in hard,
                         "expert_plan_used": False, "candidate_stage": "pre_ppr_seed_pool"})
    out = Path(output_root) / "data"
    candidate_path = out / "intermediate/task_skill_v2_gos_candidates.jsonl"
    atomic_jsonl(candidate_path, rows)
    by_type = defaultdict(list)
    for task in tasks:
        by_type[task.get("task_type", "")].append(task["record_id"])
    selected = []
    for kind, ids in sorted(by_type.items()):
        ids.sort(key=lambda x: hashlib.sha256(f"gos-v2:{kind}:{x}".encode()).hexdigest())
        selected.extend(ids[:pilot_samples_per_task_type])
    selection_path = out / "reports/task_skill_v2_gos_pilot_selection.json"
    atomic_json(selection_path, {"dataset_version": DATASET_VERSION, "record_ids": selected,
                                 "samples_per_task_type": pilot_samples_per_task_type})
    report = {"dataset_version": DATASET_VERSION, "task_count": len(tasks), "skill_count": len(skills),
              "pair_count": len(rows), "pilot_task_count": len(selected),
              "candidate_count": {"min": min(counts), "max": max(counts), "mean": sum(counts)/len(counts)},
              "candidate_source_counts": dict(sources), "expert_plan_used": False, "ppr_applied": False,
              "outputs": {"candidates": str(candidate_path), "pilot_selection": str(selection_path)}}
    atomic_json(out / "reports/task_skill_v2_gos_candidates_report.json", report)
    return report
