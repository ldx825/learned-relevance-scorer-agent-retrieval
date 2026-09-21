"""Online V3 phase recommendations using the train-frozen content NeuMF."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from benchmarks.alfworld.phase_context import (
    build_phase_specs,
    build_task_structure,
    format_phase_context,
)
from skilldag.ncf_reranker import NeuMFReranker
from skilldag.hybrid_selection import (
    cosine as _cosine,
    normalize_scores as _normalize,
    select_diverse,
)


STATE_ACTION_CONTRACTS = {
    "clean": {
        "precondition": "hold the target object and stand at a sinkbasin",
        "action_template": "clean <object> with <sinkbasin>",
        "success_evidence": "the observation explicitly says the object was cleaned",
        "non_example": "moving an object into a sinkbasin does not clean it",
    },
    "cool": {
        "precondition": "hold the target object and stand at a fridge",
        "action_template": "cool <object> with <fridge>",
        "success_evidence": "the observation explicitly says the object was cooled",
        "non_example": "moving an object into a fridge does not cool it",
    },
    "heat": {
        "precondition": "hold the target object and stand at a microwave",
        "action_template": "heat <object> with <microwave>",
        "success_evidence": "the observation explicitly says the object was heated",
        "non_example": "moving an object into a microwave does not heat it",
    },
}


def _execution_contract(structure, phase) -> dict[str, str] | None:
    if phase.phase_name != "transform_object":
        return None
    contract = STATE_ACTION_CONTRACTS.get(structure.required_state)
    return dict(contract) if contract else None


def _select_diverse(
    rows: list[dict[str, Any]], graph, cosine_top3: set[str], top_k: int = 3
) -> list[dict[str, Any]]:
    return select_diverse(
        rows,
        edges=graph.data.get("edges", []),
        cosine_safe_ids=cosine_top3,
        top_k=top_k,
    )


def build_phase_recommendations(
    *, graph, raw_full_task: str, task_type: str, pddl_params: dict[str, Any]
) -> list[dict[str, Any]]:
    model_path = os.environ.get("SKILLDAG_NCF_PHASE_MODEL", "").strip()
    if not model_path:
        return []
    calibration_path = os.environ.get("SKILLDAG_NCF_PHASE_CALIBRATION", "").strip()
    calibration = (
        json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        if calibration_path
        else {"alpha": 0.25, "skill_logit_prior": {}}
    )
    alpha = float(calibration.get("alpha", 0.25))
    priors = calibration.get("skill_logit_prior", {})
    structure = build_task_structure(task_type, pddl_params)
    phases = build_phase_specs(structure)
    contexts = [
        format_phase_context(raw_full_task=raw_full_task, structure=structure, phase=phase)
        for phase in phases
    ]
    # One embedding request contains both local-query and full-context views.
    vectors = graph._embed_queries_batch(
        [phase.phase_query for phase in phases] + contexts
    )
    query_vectors = vectors[: len(phases)]
    context_vectors = vectors[len(phases) :]
    skill_vectors = graph._load_or_build_node_embeddings()
    if len(skill_vectors) != 37:
        raise ValueError(f"V3 expects 37 ALFWorld skills, got {len(skill_vectors)}")
    reranker = NeuMFReranker(model_path, input_dim=len(context_vectors[0]))
    skill_ids = sorted(skill_vectors)
    output = []
    for phase, query_vector, context_vector in zip(phases, query_vectors, context_vectors):
        cosine_all = {
            skill_id: _cosine(query_vector, skill_vectors[skill_id])
            for skill_id in skill_ids
        }
        candidate_ids = sorted(skill_ids, key=lambda sid: (-cosine_all[sid], sid))[:12]
        logits = reranker.score(
            context_vector, [skill_vectors[skill_id] for skill_id in candidate_ids]
        )
        specific = [
            score - float(priors.get(skill_id, 0.0))
            for skill_id, score in zip(candidate_ids, logits)
        ]
        cosine_z = _normalize([cosine_all[skill_id] for skill_id in candidate_ids])
        ncf_z = _normalize(specific)
        rows = []
        for skill_id, cosine_score, ncf_score, cz, nz in zip(
            candidate_ids,
            [cosine_all[sid] for sid in candidate_ids],
            logits,
            cosine_z,
            ncf_z,
        ):
            rows.append(
                {
                    "skill_id": skill_id,
                    "description": graph.data["nodes"][skill_id].get("description", ""),
                    "cosine_score": cosine_score,
                    "ncf_logit": ncf_score,
                    "base_score": alpha * cz + (1.0 - alpha) * nz,
                }
            )
        rows.sort(key=lambda row: (-row["base_score"], row["skill_id"]))
        cosine_top3 = set(
            sorted(candidate_ids, key=lambda sid: (-cosine_all[sid], sid))[:3]
        )
        selected = _select_diverse(rows, graph, cosine_top3, top_k=3)
        output.append(
            {
                "phase": phase.phase_name,
                "phase_group": phase.phase_group,
                "query": phase.phase_query,
                "execution_contract": _execution_contract(structure, phase),
                "matches": selected,
            }
        )
    return output


def render_phase_recommendations(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return ""
    lines = [
        "V3 phase-aware recommendations (read-only hints; use `skilldag show <id>` for exact procedures):"
    ]
    for item in recommendations:
        lines.append(f"- {item['phase']} | {item['query']}")
        contract = item.get("execution_contract")
        if contract:
            lines.extend(
                [
                    f"  REQUIRED precondition: {contract['precondition']}",
                    f"  REQUIRED environment action: {contract['action_template']}",
                    f"  Completion evidence: {contract['success_evidence']}",
                    f"  Do not substitute: {contract['non_example']}",
                ]
            )
        for rank, match in enumerate(item["matches"], 1):
            description = " ".join(str(match["description"]).split())[:180]
            lines.append(f"  {rank}. {match['skill_id']}: {description}")
    return "\n".join(lines)
