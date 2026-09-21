"""Build the versioned, auditable task-skill training dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_candidates import _atomic_json, _atomic_jsonl, _load_jsonl, _sha256_bytes, _sha256_text
from .reranking_benchmark import _clean_exclusions


DATASET_VERSION = "task_skill_v1"
GRADE = {"required": 2, "helpful": 1, "irrelevant": 0, "harmful": 0}
POSITIVE_LABELS = {"required", "helpful"}
NEGATIVE_LABELS = {"irrelevant", "harmful"}
SPLIT_SEED = "skilldag-ncf-task-skill-v1-splits"
REDUNDANCY_REASON = re.compile(r"\b(redundant|alternative|equally valid|same .* capability)\b", re.I)


def _group_key(task: dict[str, Any]) -> str:
    value = str(task.get("task_text_hash", "")).strip()
    if value:
        return value
    normalized = re.sub(r"\s+", " ", str(task.get("task_text", "")).lower()).strip()
    return "sha256:" + _sha256_text(normalized)


def _choose_groups(
    groups: list[tuple[str, list[dict[str, Any]]]], target: int, salt: str
) -> set[str]:
    """Choose deterministic whole text-groups with total size nearest to target."""
    ranked = sorted(
        groups,
        key=lambda item: hashlib.sha256(f"{SPLIT_SEED}:{salt}:{item[0]}".encode()).hexdigest(),
    )
    possibilities: dict[int, tuple[str, ...]] = {0: ()}
    for key, members in ranked:
        size = len(members)
        for total, chosen in sorted(list(possibilities.items()), reverse=True):
            new_total = total + size
            if new_total <= target and new_total not in possibilities:
                possibilities[new_total] = chosen + (key,)
    best = max(possibilities)
    return set(possibilities[best])


def _assign_splits(
    tasks: list[dict[str, Any]], pilot_ids: set[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_type[str(task.get("task_type", ""))].append(task)
    assignments: dict[str, str] = {}
    per_type: dict[str, dict[str, int]] = {}
    for task_type, rows in sorted(by_type.items()):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_group_key(row)].append(row)
        forced_train = {
            key for key, members in grouped.items() if any(row["record_id"] in pilot_ids for row in members)
        }
        eligible = [(key, members) for key, members in grouped.items() if key not in forced_train]
        eval_target = round(len(rows) * 0.10)
        dev_groups = _choose_groups(eligible, eval_target, f"{task_type}:dev")
        remaining = [(key, members) for key, members in eligible if key not in dev_groups]
        test_groups = _choose_groups(remaining, eval_target, f"{task_type}:test")
        counts: Counter[str] = Counter()
        for key, members in grouped.items():
            split = "dev" if key in dev_groups else "test" if key in test_groups else "train"
            for row in members:
                assignments[row["record_id"]] = split
                counts[split] += 1
        per_type[task_type] = {name: counts[name] for name in ("train", "dev", "test")}
    return assignments, {"seed": SPLIT_SEED, "counts_by_task_type": per_type}


def _relations_by_skill(graph: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in graph.get("edges", []):
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        edge_type = str(edge.get("type", ""))
        if source and target and edge_type:
            result[source].append({"skill_id": target, "type": edge_type, "direction": "outgoing"})
            result[target].append({"skill_id": source, "type": edge_type, "direction": "incoming"})
    return result


def _class_weights(counts: Counter[int]) -> dict[str, float]:
    total = sum(counts.values())
    classes = len(counts)
    return {str(label): total / (classes * count) for label, count in sorted(counts.items()) if count}


def build_task_skill_dataset(
    tasks_path: Path | str,
    skills_path: Path | str,
    candidates_path: Path | str,
    evidence_path: Path | str,
    graph_path: Path | str,
    pilot_selection_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Fuse Judge evidence and retrieval metadata without any network access."""
    paths = [
        Path(value).expanduser().resolve()
        for value in (tasks_path, skills_path, candidates_path, evidence_path, graph_path, pilot_selection_path)
    ]
    tasks_path, skills_path, candidates_path, evidence_path, graph_path, pilot_selection_path = paths
    output_root = Path(output_root).expanduser().resolve()

    all_tasks = {row["record_id"]: row for row in _load_jsonl(tasks_path)}
    skills = {row["skill_id"]: row for row in _load_jsonl(skills_path)}
    all_candidates = _load_jsonl(candidates_path)
    evidence_rows = _load_jsonl(evidence_path)
    evidence = {(row["task_record_id"], row["skill_id"]): row for row in evidence_rows}
    if len(evidence) != len(evidence_rows):
        raise ValueError("duplicate task-skill evidence rows")
    evidence_task_ids = {task_id for task_id, _ in evidence}
    candidates = [
        row for row in all_candidates if row.get("task_record_id") in evidence_task_ids
    ]
    candidate_keys = {(row["task_record_id"], row["skill_id"]) for row in candidates}
    if len(candidate_keys) != len(candidates):
        raise ValueError("duplicate task-skill candidate rows")
    if candidate_keys != set(evidence):
        raise ValueError("candidate and evidence task-skill pairs do not match")
    selected_ids = sorted(evidence_task_ids)
    missing_tasks = set(selected_ids) - set(all_tasks)
    if missing_tasks:
        raise ValueError(f"dataset tasks missing from task table: {sorted(missing_tasks)[:5]}")
    missing_skills = {skill_id for _, skill_id in candidate_keys} - set(skills)
    if missing_skills:
        raise ValueError(f"dataset skills missing from skill table: {sorted(missing_skills)}")

    selected_tasks = [all_tasks[task_id] for task_id in selected_ids]
    pilot_ids = set(json.loads(pilot_selection_path.read_text(encoding="utf-8"))["record_ids"])
    if not pilot_ids <= set(selected_ids):
        raise ValueError("pilot selection is not a subset of dataset tasks")
    split_by_task, split_report = _assign_splits(selected_tasks, pilot_ids)

    graph_raw = graph_path.read_bytes()
    graph = json.loads(graph_raw)
    graph_snapshot_id = f"sha256:{_sha256_bytes(graph_raw)}"
    relations = _relations_by_skill(graph)
    evidence_by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    candidates_by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in evidence_rows:
        evidence_by_task[row["task_record_id"]][row["skill_id"]] = row
    for row in candidates:
        candidates_by_task[row["task_record_id"]][row["skill_id"]] = row
    exclusion_reasons = _clean_exclusions(all_tasks, evidence_by_task)

    task_rows = []
    split_task_counts: Counter[str] = Counter()
    for task in selected_tasks:
        row = dict(task)
        row.update(
            {
                "dataset_version": DATASET_VERSION,
                "dataset_split": split_by_task[task["record_id"]],
                "is_pilot_task": task["record_id"] in pilot_ids,
                "text_group_id": _group_key(task),
            }
        )
        split_task_counts[row["dataset_split"]] += 1
        task_rows.append(row)

    pair_rows = []
    label_counts: Counter[str] = Counter()
    split_pair_counts: Counter[str] = Counter()
    split_grade_counts: dict[str, Counter[int]] = defaultdict(Counter)
    hard_negative_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    for task_id in selected_ids:
        task_evidence = evidence_by_task[task_id]
        candidate_ids = set(candidates_by_task[task_id])
        positive_ids = {
            skill_id for skill_id, item in task_evidence.items() if item["raw_label"] in POSITIVE_LABELS
        }
        for skill_id in sorted(candidate_ids):
            candidate = candidates_by_task[task_id][skill_id]
            item = task_evidence[skill_id]
            raw_label = item["raw_label"]
            grade = GRADE.get(raw_label)
            quality_flags = [
                reason
                for reason in exclusion_reasons.get(task_id, [])
                if reason != "required_reason_indicates_redundancy"
            ]
            if raw_label == "required" and REDUNDANCY_REASON.search(str(item.get("reason", ""))):
                quality_flags.append("required_reason_indicates_redundancy")
            candidate_relations = sorted(
                (relation for relation in relations.get(skill_id, []) if relation["skill_id"] in candidate_ids),
                key=lambda value: (value["type"], value["skill_id"], value["direction"]),
            )
            similar_positive_peers = sorted(
                {
                    relation["skill_id"]
                    for relation in candidate_relations
                    if relation["type"] == "similar_to" and relation["skill_id"] in positive_ids
                }
            )
            if raw_label == "required" and similar_positive_peers:
                quality_flags.append("required_has_similar_positive_peer")
            quality_flags = sorted(set(quality_flags))
            for flag in quality_flags:
                quality_counts[flag] += 1

            negative_types: list[str] = []
            if raw_label in NEGATIVE_LABELS:
                negative_types.append("judge_negative")
                if candidate.get("task_semantic_rank") is not None:
                    negative_types.append("task_semantic_hard_negative")
                if candidate.get("action_semantic_rank") is not None:
                    negative_types.append("action_semantic_hard_negative")
                if similar_positive_peers:
                    negative_types.append("graph_similar_hard_negative")
                if raw_label == "harmful":
                    negative_types.append("harmful_negative")
            for value in negative_types:
                hard_negative_counts[value] += 1

            confidence = float(item["confidence"])
            weight = 0.0 if grade is None else 0.6 * confidence
            if "all_candidates_positive" in quality_flags:
                weight *= 0.75
            if "required_reason_indicates_redundancy" in quality_flags and raw_label == "required":
                weight *= 0.5
            if any(flag.startswith("task_text_conflicts_") for flag in quality_flags):
                weight *= 0.5
            split = split_by_task[task_id]
            label_counts[raw_label] += 1
            split_pair_counts[split] += 1
            if grade is not None:
                split_grade_counts[split][grade] += 1
            pair_rows.append(
                {
                    "pair_id": "pair:" + _sha256_text(f"{DATASET_VERSION}:{task_id}:{skill_id}")[:24],
                    "dataset_version": DATASET_VERSION,
                    "dataset_split": split,
                    "task_record_id": task_id,
                    "skill_id": skill_id,
                    "raw_label": raw_label,
                    "target_grade": grade,
                    "target_relevance": float(item["score"]),
                    "judge_confidence": confidence,
                    "sample_weight": round(weight, 6),
                    "trainable": grade is not None,
                    "is_harmful": raw_label == "harmful",
                    "is_hard_negative": bool(negative_types),
                    "negative_types": negative_types,
                    "quality_flags": quality_flags,
                    "evidence_ids": [item["evidence_id"]],
                    "judge_reason": item.get("reason", ""),
                    "candidate_sources": candidate.get("candidate_sources", []),
                    "task_cosine_score": candidate.get("task_cosine_score"),
                    "task_semantic_rank": candidate.get("task_semantic_rank"),
                    "action_cosine_score": candidate.get("action_cosine_score"),
                    "action_semantic_rank": candidate.get("action_semantic_rank"),
                    "matched_actions": candidate.get("matched_actions", []),
                    "graph_snapshot_id": candidate.get("graph_snapshot_id", graph_snapshot_id),
                    "graph_relations_to_candidates": candidate_relations,
                    "similar_positive_peer_ids": similar_positive_peers,
                    "label_source": item.get("label_source", "llm_judge"),
                    "source_model": item.get("source_model", ""),
                    "source_prompt_version": item.get("source_prompt_version", ""),
                    "expert_plan_visible": bool(item.get("expert_plan_visible")),
                }
            )

    dataset_dir = output_root / "datasets" / DATASET_VERSION
    report_dir = output_root / "reports"
    tasks_output = dataset_dir / "tasks.jsonl"
    skills_output = dataset_dir / "skills.jsonl"
    pairs_output = dataset_dir / "pairs.jsonl"
    split_output = dataset_dir / "splits.json"
    card_output = dataset_dir / "DATASET_CARD.md"
    _atomic_jsonl(tasks_output, task_rows)
    _atomic_jsonl(skills_output, [skills[key] for key in sorted(skills)])
    _atomic_jsonl(pairs_output, pair_rows)
    split_payload = {
        "dataset_version": DATASET_VERSION,
        **split_report,
        "record_ids": {
            name: sorted(task_id for task_id, split in split_by_task.items() if split == name)
            for name in ("train", "dev", "test")
        },
    }
    _atomic_json(split_output, split_payload)

    report = {
        "schema_version": "skilldag_ncf.task_skill_dataset_report.v1",
        "dataset_version": DATASET_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_made": 0,
        "judge_evidence_is_gold": False,
        "task_count": len(task_rows),
        "skill_count": len(skills),
        "pair_count": len(pair_rows),
        "trainable_pair_count": sum(row["trainable"] for row in pair_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "task_counts_by_dataset_split": dict(sorted(split_task_counts.items())),
        "pair_counts_by_dataset_split": dict(sorted(split_pair_counts.items())),
        "grade_counts_by_dataset_split": {
            split: {str(grade): count for grade, count in sorted(counts.items())}
            for split, counts in sorted(split_grade_counts.items())
        },
        "suggested_balanced_grade_weights_train": _class_weights(split_grade_counts["train"]),
        "hard_negative_counts": dict(sorted(hard_negative_counts.items())),
        "quality_flag_pair_counts": dict(sorted(quality_counts.items())),
        "sample_weight_policy": {
            "base": "0.6 * judge_confidence",
            "uncertain": 0.0,
            "all_candidates_positive_multiplier": 0.75,
            "redundant_required_multiplier": 0.5,
            "task_text_plan_conflict_multiplier": 0.5,
            "class_balance": "reported separately; not baked into sample_weight",
        },
        "split_policy": {
            "unit": "task_text_hash group",
            "pilot_tasks": "forced to train",
            "target_per_task_type": "80% train / 10% dev / 10% test",
            "seed": SPLIT_SEED,
        },
        "limitations": [
            "Labels are MiniMax-M2.7 weak labels, not gold labels.",
            "The Judge saw expert plans and action candidates derive from expert plans.",
            "Only retrieved candidates are labeled; unobserved skills are not negatives.",
            "Required/helpful labels are strongly overrepresented.",
        ],
        "provenance": {
            "tasks_path": str(tasks_path),
            "skills_path": str(skills_path),
            "candidates_path": str(candidates_path),
            "evidence_path": str(evidence_path),
            "graph_path": str(graph_path),
            "pilot_selection_path": str(pilot_selection_path),
            "input_sha256": {
                path.name: f"sha256:{_sha256_bytes(path.read_bytes())}" for path in paths
            },
            "graph_snapshot_id": graph_snapshot_id,
        },
        "outputs": {
            "tasks": str(tasks_output),
            "skills": str(skills_output),
            "pairs": str(pairs_output),
            "splits": str(split_output),
            "dataset_card": str(card_output),
        },
    }
    report_path = report_dir / f"{DATASET_VERSION}_report.json"
    report["outputs"]["report"] = str(report_path)
    card = f"""# {DATASET_VERSION}

这是由 ALFWorld 专家计划候选和 MiniMax-M2.7 Judge 弱标签构造的 task–skill 检索数据集。

## 规模

- task：{len(task_rows)}
- skill：{len(skills)}
- candidate pair：{len(pair_rows)}
- 可训练 pair：{sum(row['trainable'] for row in pair_rows)}
- split：train {split_task_counts['train']} / dev {split_task_counts['dev']} / test {split_task_counts['test']}

## 标签

`required=2`、`helpful=1`、`irrelevant/harmful=0`，`uncertain` 的 `target_grade` 为 null，默认不训练。`harmful` 通过 `is_harmful` 单独保留。

`sample_weight` 是透明的初始建议，不是已经校准的概率。类别平衡权重只写入 report，不与弱标签可信度混在一起。

## 防泄漏

原 70-task pilot 固定进入 train；相同 `task_text_hash` 的任务作为一个整体分配，不会跨 train/dev/test。

## 重要限制

- Judge 标签不是人工 gold。
- Judge 看过 expert plan，action cosine 也来自 expert plan。
- 数据只覆盖检索候选，未被检索到的 skill 不能直接视为负例。
- 该版本适合训练和比较重排器，不足以单独证明 Agent 最终成功率提升。
"""
    card_output.parent.mkdir(parents=True, exist_ok=True)
    card_output.write_text(card, encoding="utf-8")
    _atomic_json(report_path, report)
    return report
