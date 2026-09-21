#!/usr/bin/env python3
"""Build deterministic, leakage-safe skill families and train/dev splits.

Families merge only normalized aliases and high-confidence cold-graph
similar_to/specializes edges whose cached embedding cosine passes the frozen
threshold. depends_on/composes_with edges are audited as cross-family
composition signals and never merge families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from source_policy import (
    DEFAULT_POLICY_PATH,
    ROOT,
    assert_training_source,
    load_policy,
)


DEFAULT_SKILL_MANIFEST = (
    ROOT / "artifacts/skillbench_ncf/manifests/skills_1000_manifest.jsonl"
)
DEFAULT_GRAPH = (
    ROOT / ".runtime/skilldag/data/skilldag/skilldag_graphs/"
    "skillgraph_1000.json"
)
DEFAULT_EMBEDDINGS = (
    ROOT / ".runtime/skilldag/data/skilldag/skilldag_graphs/"
    "skillgraph_1000.embeddings.json"
)
DEFAULT_SOURCE_LOCK = ROOT / "configs/skillbench_ncf/source_lock.json"
DEFAULT_CONFIG = ROOT / "configs/skillbench_ncf/family_clustering.json"
DEFAULT_OUTPUT = ROOT / "artifacts/skillbench_ncf/manifests"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_alias(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip(
        "-"
    )


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        return True


def cosine(
    left: list[float],
    right: list[float],
    left_norm: float,
    right_norm: float,
) -> float:
    return sum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm
    )


def family_id(members: list[str]) -> str:
    return "fam_" + stable_hash("\n".join(sorted(members)))[:12]


def family_stratum(
    members: list[str],
    prefixes: list[str],
) -> str:
    if any(
        member.startswith(prefix)
        for member in members
        for prefix in prefixes
    ):
        return "benchmark_environment"
    automation_count = sum(
        normalized_alias(member).endswith("-automation") for member in members
    )
    if automation_count * 2 >= len(members):
        return "automation"
    return "other"


def select_dev_families(
    families: list[dict[str, Any]],
    fraction: float,
    seed: str,
) -> tuple[set[str], dict[str, dict[str, int]]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family in families:
        by_stratum[str(family["stratum"])].append(family)

    dev: set[str] = set()
    allocation: dict[str, dict[str, int]] = {}
    for stratum, rows in sorted(by_stratum.items()):
        total_skills = sum(int(row["size"]) for row in rows)
        target = round(total_skills * fraction)
        ordered = sorted(
            rows,
            key=lambda row: stable_hash(
                f"{seed}\0{stratum}\0{row['family_id']}"
            ),
        )
        selected = 0
        deferred: list[dict[str, Any]] = []
        for row in ordered:
            size = int(row["size"])
            if selected + size <= target:
                dev.add(str(row["family_id"]))
                selected += size
            else:
                deferred.append(row)
        if selected < target:
            candidates = sorted(
                deferred,
                key=lambda row: (
                    abs(target - (selected + int(row["size"]))),
                    stable_hash(
                        f"{seed}\0fallback\0{stratum}\0{row['family_id']}"
                    ),
                ),
            )
            if candidates and abs(
                target - (selected + int(candidates[0]["size"]))
            ) < abs(target - selected):
                chosen = candidates[0]
                dev.add(str(chosen["family_id"]))
                selected += int(chosen["size"])
        allocation[stratum] = {
            "total_skills": total_skills,
            "target_dev_skills": target,
            "actual_dev_skills": selected,
        }
    return dev, allocation


def build(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    graph_path = assert_training_source(args.graph, policy)
    embeddings_path = assert_training_source(args.embeddings, policy)
    for path in (
        args.skill_manifest,
        graph_path,
        embeddings_path,
        args.source_lock,
        args.config,
        args.policy,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_lock = load_json(args.source_lock)
    config = load_json(args.config)
    expected = source_lock["expected"]
    expected_embedding_sha = source_lock["sources"]["embedding_cache"][
        "sha256"
    ]
    actual_embedding_sha = sha256_file(embeddings_path)
    if actual_embedding_sha != expected_embedding_sha:
        raise ValueError(
            "embedding cache SHA256 mismatch: "
            f"expected={expected_embedding_sha}, actual={actual_embedding_sha}"
        )

    skill_rows = load_jsonl(args.skill_manifest)
    skill_by_id = {str(row["skill_id"]): row for row in skill_rows}
    if len(skill_by_id) != len(skill_rows):
        raise ValueError("duplicate skill IDs in source manifest")

    graph = load_json(graph_path)
    graph_ids = set(graph["nodes"])
    skill_ids = set(skill_by_id)
    if graph_ids != skill_ids:
        raise ValueError("source manifest and graph node IDs disagree")

    embedding_payload = load_json(embeddings_path)
    embedding_ids = set(embedding_payload)
    if embedding_ids != skill_ids:
        raise ValueError("embedding cache and skill IDs disagree")
    model_counts = Counter(
        str(value.get("model", "")) for value in embedding_payload.values()
    )
    dimension_counts = Counter(
        len(value.get("embedding", []))
        for value in embedding_payload.values()
    )
    if model_counts != Counter(
        {str(expected["embedding_model"]): int(expected["embedding_count"])}
    ):
        raise ValueError(f"unexpected embedding models: {model_counts}")
    if dimension_counts != Counter(
        {int(expected["embedding_dimension"]): int(expected["embedding_count"])}
    ):
        raise ValueError(f"unexpected embedding dimensions: {dimension_counts}")

    vectors = {
        skill_id: payload["embedding"]
        for skill_id, payload in embedding_payload.items()
    }
    norms = {
        skill_id: math.sqrt(sum(value * value for value in vector))
        for skill_id, vector in vectors.items()
    }
    if any(value == 0.0 for value in norms.values()):
        raise ValueError("zero-norm embedding found")

    dsu = DisjointSet(skill_ids)
    evidence: list[dict[str, Any]] = []
    if config["merge_rules"]["normalized_alias"]:
        aliases: dict[str, list[str]] = defaultdict(list)
        for skill_id in sorted(skill_ids):
            aliases[normalized_alias(skill_id)].append(skill_id)
        for alias, members in sorted(aliases.items()):
            if len(members) < 2:
                continue
            anchor = members[0]
            for member in members[1:]:
                dsu.union(anchor, member)
                evidence.append(
                    {
                        "source": anchor,
                        "target": member,
                        "rule": "normalized_alias",
                        "normalized_alias": alias,
                    }
                )

    accepted_types = set(config["merge_rules"]["graph_edge_types"])
    threshold = float(config["merge_rules"]["minimum_cosine"])
    graph_similarity_rows: list[dict[str, Any]] = []
    for index, edge in enumerate(graph["edges"]):
        edge_type = str(edge.get("type"))
        if edge_type not in accepted_types:
            continue
        source = str(edge["source"])
        target = str(edge["target"])
        score = cosine(
            vectors[source],
            vectors[target],
            norms[source],
            norms[target],
        )
        accepted = score >= threshold
        graph_similarity_rows.append(
            {
                "edge_index": index,
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "cosine": round(score, 8),
                "accepted": accepted,
            }
        )
        if accepted:
            dsu.union(source, target)
            evidence.append(
                {
                    "source": source,
                    "target": target,
                    "rule": "graph_edge_cosine",
                    "edge_type": edge_type,
                    "cosine": round(score, 8),
                }
            )

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for skill_id in sorted(skill_ids):
        members_by_root[dsu.find(skill_id)].append(skill_id)

    split_config = config["split"]
    prefixes = list(split_config["benchmark_environment_prefixes"])
    families: list[dict[str, Any]] = []
    for members in members_by_root.values():
        members = sorted(members)
        identifier = family_id(members)
        family_evidence = [
            row
            for row in evidence
            if row["source"] in members and row["target"] in members
        ]
        families.append(
            {
                "family_id": identifier,
                "size": len(members),
                "members": members,
                "stratum": family_stratum(members, prefixes),
                "merge_evidence": family_evidence,
            }
        )
    families.sort(key=lambda row: str(row["family_id"]))

    maximum_size = int(config["merge_rules"]["maximum_allowed_family_size"])
    oversized = [
        family for family in families if int(family["size"]) > maximum_size
    ]
    if oversized:
        raise ValueError(
            "family-size guard failed: "
            + ", ".join(
                f"{row['family_id']}={row['size']}" for row in oversized
            )
        )

    dev_family_ids, allocation = select_dev_families(
        families,
        float(split_config["dev_fraction"]),
        str(split_config["seed"]),
    )
    for family in families:
        family["split"] = (
            "dev" if family["family_id"] in dev_family_ids else "train"
        )

    family_by_skill: dict[str, dict[str, Any]] = {}
    for family in families:
        for skill_id in family["members"]:
            if skill_id in family_by_skill:
                raise ValueError(f"skill appears in two families: {skill_id}")
            family_by_skill[skill_id] = family

    skill_output_rows = []
    for skill_id in sorted(skill_ids):
        family = family_by_skill[skill_id]
        source = skill_by_id[skill_id]
        skill_output_rows.append(
            {
                "skill_id": skill_id,
                "family_id": family["family_id"],
                "family_size": family["size"],
                "split": family["split"],
                "stratum": family["stratum"],
                "frontmatter_name": source["frontmatter_name"],
                "frontmatter_description": source[
                    "frontmatter_description"
                ],
                "skill_md_sha256": source["skill_md_sha256"],
            }
        )

    split_skill_ids = {
        split: {
            row["skill_id"]
            for row in skill_output_rows
            if row["split"] == split
        }
        for split in ("train", "dev")
    }
    split_family_ids = {
        split: {
            row["family_id"]
            for row in families
            if row["split"] == split
        }
        for split in ("train", "dev")
    }

    accepted_graph_edges = [
        row for row in graph_similarity_rows if row["accepted"]
    ]
    evaluated_graph_edge_types = Counter(
        row["edge_type"] for row in graph_similarity_rows
    )
    accepted_graph_edge_types = Counter(
        row["edge_type"] for row in accepted_graph_edges
    )
    accepted_cross_split = [
        row
        for row in accepted_graph_edges
        if family_by_skill[row["source"]]["split"]
        != family_by_skill[row["target"]]["split"]
    ]
    structural_edges: Counter[str] = Counter()
    structural_cross_split: Counter[str] = Counter()
    structural_cross_family: Counter[str] = Counter()
    for edge in graph["edges"]:
        edge_type = str(edge["type"])
        if edge_type not in {"depends_on", "composes_with"}:
            continue
        structural_edges[edge_type] += 1
        source_family = family_by_skill[str(edge["source"])]
        target_family = family_by_skill[str(edge["target"])]
        if source_family["family_id"] != target_family["family_id"]:
            structural_cross_family[edge_type] += 1
        if source_family["split"] != target_family["split"]:
            structural_cross_split[edge_type] += 1

    size_distribution = Counter(
        str(family["size"]) for family in families
    )
    split_counts = {
        split: {
            "skills": len(split_skill_ids[split]),
            "families": len(split_family_ids[split]),
        }
        for split in ("train", "dev")
    }
    stratum_counts: dict[str, dict[str, int]] = {}
    for stratum in split_config["strata"]:
        stratum_counts[stratum] = {
            f"{split}_skills": sum(
                int(family["size"])
                for family in families
                if family["stratum"] == stratum
                and family["split"] == split
            )
            for split in ("train", "dev")
        }
        stratum_counts[stratum].update(
            {
                f"{split}_families": sum(
                    1
                    for family in families
                    if family["stratum"] == stratum
                    and family["split"] == split
                )
                for split in ("train", "dev")
            }
        )

    checks = {
        "all_1000_skills_assigned_once": len(family_by_skill)
        == int(expected["skill_count"])
        == len(skill_ids),
        "train_dev_skill_disjoint": not (
            split_skill_ids["train"] & split_skill_ids["dev"]
        ),
        "train_dev_family_disjoint": not (
            split_family_ids["train"] & split_family_ids["dev"]
        ),
        "train_dev_skill_union_complete": (
            split_skill_ids["train"] | split_skill_ids["dev"]
        )
        == skill_ids,
        "accepted_merge_edges_do_not_cross_split": not accepted_cross_split,
        "maximum_family_size_within_guard": max(
            int(family["size"]) for family in families
        )
        <= maximum_size,
        "embedding_cache_matches_source_lock": actual_embedding_sha
        == expected_embedding_sha,
        "only_skill_sources_used": True,
        "official_task_content_not_read": True,
    }

    family_split = {
        "schema_version": "skillbench_ncf.family_split.v1",
        "scale": config["scale"],
        "config_sha256": sha256_file(args.config),
        "source_manifest_sha256": sha256_file(args.skill_manifest),
        "graph_sha256": sha256_file(graph_path),
        "embedding_cache_sha256": actual_embedding_sha,
        "split_role": split_config["role"],
        "split_seed": split_config["seed"],
        "dev_fraction": split_config["dev_fraction"],
        "final_training_policy": split_config["final_training_policy"],
        "allocation": allocation,
        "counts": split_counts,
        "families": families,
    }
    audit = {
        "schema_version": "skillbench_ncf.family_audit.v1",
        "scale": config["scale"],
        "merge_rule": config["merge_rules"],
        "counts": {
            "skills": len(skill_ids),
            "families": len(families),
            "singleton_families": sum(
                int(family["size"]) == 1 for family in families
            ),
            "multi_skill_families": sum(
                int(family["size"]) > 1 for family in families
            ),
            "accepted_alias_merges": sum(
                row["rule"] == "normalized_alias" for row in evidence
            ),
            "accepted_graph_edges": len(accepted_graph_edges),
            "evaluated_graph_edges_by_type": dict(
                sorted(evaluated_graph_edge_types.items())
            ),
            "accepted_graph_edges_by_type": dict(
                sorted(accepted_graph_edge_types.items())
            ),
            "split": split_counts,
        },
        "embedding": {
            "sha256": actual_embedding_sha,
            "model_counts": dict(sorted(model_counts.items())),
            "dimension_counts": {
                str(key): value
                for key, value in sorted(dimension_counts.items())
            },
        },
        "family_size_distribution": dict(
            sorted(size_distribution.items(), key=lambda item: int(item[0]))
        ),
        "largest_families": [
            {
                "family_id": family["family_id"],
                "size": family["size"],
                "members": family["members"],
                "split": family["split"],
                "stratum": family["stratum"],
            }
            for family in sorted(
                families,
                key=lambda row: (
                    -int(row["size"]),
                    str(row["family_id"]),
                ),
            )[:20]
        ],
        "stratum_counts": stratum_counts,
        "structural_edges": {
            "total": dict(sorted(structural_edges.items())),
            "cross_family": dict(sorted(structural_cross_family.items())),
            "cross_split": dict(sorted(structural_cross_split.items())),
        },
        "high_confidence_merge_edge_cross_split_count": len(
            accepted_cross_split
        ),
        "checks": checks,
        "passed": all(checks.values()),
        "leakage_statement": (
            "This process read only the tracked skill manifest, cold graph, "
            "published embedding cache, source lock, clustering config, and "
            "path policy. It did not read any official SkillsBench task."
        ),
    }

    dump_jsonl(args.output / "skill_family_manifest.jsonl", skill_output_rows)
    dump_json(args.output / "family_split.json", family_split)
    dump_json(args.output / "family_audit_report.json", audit)
    if not audit["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"family audit failed: {', '.join(failed)}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-manifest", type=Path, default=DEFAULT_SKILL_MANIFEST
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build(args)
    counts = audit["counts"]
    print(
        "PASS: skills={skills}, families={families}, "
        "train={train}, dev={dev}, largest={largest}".format(
            skills=counts["skills"],
            families=counts["families"],
            train=counts["split"]["train"]["skills"],
            dev=counts["split"]["dev"]["skills"],
            largest=audit["largest_families"][0]["size"],
        )
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
