"""Shared conservative NCF + cosine + graph selection primitives."""

from __future__ import annotations

import math
from typing import Any, Iterable


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.0] * len(values)
    return [(value - mean) / std for value in values]


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / max(denominator, 1e-12)


def edge_lookup(edges: Iterable[dict[str, Any]]) -> dict[frozenset[str], set[str]]:
    lookup: dict[frozenset[str], set[str]] = {}
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        edge_type = edge.get("type")
        if source and target and edge_type:
            lookup.setdefault(frozenset((str(source), str(target))), set()).add(
                str(edge_type)
            )
    return lookup


def select_diverse(
    rows: list[dict[str, Any]],
    *,
    edges: Iterable[dict[str, Any]],
    cosine_safe_ids: set[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Select a graph-aware set while retaining one cosine safety result."""
    relationships = edge_lookup(edges)
    selected: list[dict[str, Any]] = []
    remaining = list(rows)
    while remaining and len(selected) < top_k:
        choices = []
        for row in remaining:
            gain = float(row["base_score"])
            blocked = False
            for prior in selected:
                types = relationships.get(
                    frozenset((row["skill_id"], prior["skill_id"])), set()
                )
                if "conflicts_with" in types:
                    blocked = True
                    break
                if "similar_to" in types:
                    gain -= 0.35
                if "depends_on" in types or "composes_with" in types:
                    gain += 0.08
                if "specializes" in types:
                    gain += 0.05
            if not blocked:
                choices.append((gain, row["skill_id"], row))
        if not choices:
            break
        chosen = max(choices, key=lambda item: (item[0], item[1]))[2]
        selected.append(chosen)
        remaining.remove(chosen)

    if selected and not any(row["skill_id"] in cosine_safe_ids for row in selected):
        fallback = next(
            (row for row in rows if row["skill_id"] in cosine_safe_ids), None
        )
        if fallback is not None:
            selected[-1] = fallback
    return selected
