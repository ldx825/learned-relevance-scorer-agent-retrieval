#!/usr/bin/env python3
"""Resumably extract grounded atomic operations from45 train-only Memories."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src/SkillDAG_NCF"
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from skilldag.initialize import _http_post_json  # noqa: E402
from travelplanner_atomic_memory import OPERATION_SPECS  # noqa: E402


PROMPT_VERSION = "travelplanner_atomic_memory_extraction_v4_semantic_gate"
PARSER_VERSION = "grounded_operation_parser_v2"
ALLOWED = {row["key"]: row for row in OPERATION_SPECS}
SYSTEM_PROMPT = """You exhaustively extract ALL atomic procedural operations from one TravelPlanner Script Memory.

Rules:
1. Use only the supplied parent Memory workflow and visible source-task constraints.
2. Choose operation_key only from the supplied ontology.
3. Emit an operation only when the workflow contains explicit evidence for it.
4. workflow_evidence must be one short verbatim contiguous quote from the workflow.
5. Do not convert generic planning language into an unsupported specialized check.
6. Keep EACH operation atomic, but return EVERY independently supported operation in the entire workflow. Do not stop after the first operation. A typical workflow contains8-15 operations.
7. Conditional operations (transport restriction, room type, house rule, cuisine) must be supported by both the source task and workflow.
8. Do not mention validation/test data or infer any hidden evaluation requirement.
9. Inspect every workflow sentence. If the workflow explicitly contains AccommodationSearch, RestaurantSearch, AttractionSearch, Notebook recording, budget accounting, verification, or repair, the corresponding ontology operation must be returned.
10. Returning only route_search is invalid when later workflow clauses explicitly describe other operations.
11. The same operation_key may appear more than once when it is a genuinely different workflow step (for example, recording transport and recording lodging in Notebook). Keep those as separate operation instances.
12. Prefer copying one complete workflow sentence verbatim as workflow_evidence.

Return exactly one JSON object without markdown:
{"operations":[{"operation_key":"route_search","goal":"...","precondition":"...","procedure":"...","success_evidence":"...","failure_mode":"...","workflow_evidence":"exact workflow quote"}]}
"""


def normalize(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def align_evidence_to_workflow(quote: str, workflow: str) -> str:
    """Map a lightly compressed quote back to one exact parent-workflow sentence."""
    normalized_quote = normalize(quote).strip('"“”\'')
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", workflow) if part.strip()]
    direct = [sentence for sentence in sentences if normalized_quote in normalize(sentence)]
    if direct:
        return min(direct, key=len)
    quote_tokens = set(re.findall(r"[a-z0-9-]+", normalized_quote))
    best_sentence, best_score = "", 0.0
    for sentence in sentences:
        normalized_sentence = normalize(sentence)
        sentence_tokens = set(re.findall(r"[a-z0-9-]+", normalized_sentence))
        union = quote_tokens | sentence_tokens
        overlap = len(quote_tokens & sentence_tokens) / max(len(union), 1)
        sequence = difflib.SequenceMatcher(None, normalized_quote, normalized_sentence).ratio()
        score = 0.65 * sequence + 0.35 * overlap
        if score > best_score:
            best_sentence, best_score = sentence, score
    if best_score < 0.68:
        raise ValueError(f"evidence cannot be aligned to workflow sentence (score={best_score:.3f})")
    return best_sentence


EVIDENCE_TERMS: dict[str, tuple[str, ...]] = {
    "route_search": ("flightsearch", "googledistancematrix", "transportation", "route"),
    "route_closure": ("route closure", "route is closed", "return transportation", "every leg"),
    "transport_restriction": ("no flight", "non-flight", "no self-driving", "non-self-driving"),
    "notebook_grounding": ("notebook", "record the", "log the", "store the"),
    "lodging_search": ("accommodation", "lodging"),
    "minimum_nights": ("minimum-night", "minimum night"),
    "party_capacity": ("party size", "capacity", "all travelers"),
    "room_type": ("room type", "entire room", "private room"),
    "house_rule": ("house rule", "smoking", "pets", "parties", "visitors", "children under 10"),
    "dining_search": ("restaurant", "meal", "breakfast", "lunch", "dinner"),
    "cuisine_coverage": ("cuisine",),
    "restaurant_uniqueness": ("restaurant is repeated", "non-repeated", "non-repeating", "repeated restaurants"),
    "attraction_search": ("attraction", "citysearch"),
    "attraction_uniqueness": ("unique attractions", "repeated attractions", "attraction is repeated", "no duplicates"),
    "full_party_budget": ("total cost", "party size", "full party", "within budget", "total budget"),
    "final_constraint_audit": ("before running planner", "before invoking planner", "verify all", "validate all", "confirm transportation"),
    "repair_before_planner": ("re-search", "search for alternatives", "rather than invent", "rather than fabricate"),
}


def evidence_supports_operation(operation_key: str, evidence: str) -> bool:
    normalized = normalize(evidence)
    return any(term in normalized for term in EVIDENCE_TERMS[operation_key])


def prompt_for(profile: dict[str, Any]) -> str:
    ontology = [
        {
            "operation_key": row["key"],
            "family": row["family"],
            "definition": row["goal"],
            "conditional_field": row.get("conditional_field"),
        }
        for row in OPERATION_SPECS
    ]
    payload = {
        "memory_id": int(profile["memory_id"]),
        "source_task_query": profile["memory_query"],
        "source_task_signature": profile["source_signature"],
        "parent_workflow": profile["workflow"],
        "allowed_operation_ontology": ontology,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def extract_content(response: dict[str, Any]) -> str:
    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def parse_response(content: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response contains no JSON object")
        payload = json.loads(text[start : end + 1])
    operations = payload.get("operations") if isinstance(payload, dict) else None
    if not isinstance(operations, list):
        raise ValueError("response has no operations list")
    workflow = normalize(profile["workflow"])
    key_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for raw in operations:
        key = str(raw.get("operation_key", "")).strip()
        if key not in ALLOWED:
            raise ValueError(f"invalid operation_key: {key!r}")
        fields = {
            name: " ".join(str(raw.get(name, "")).split())
            for name in (
                "goal", "precondition", "procedure", "success_evidence",
                "failure_mode", "workflow_evidence",
            )
        }
        if any(len(value) < 8 for value in fields.values()):
            raise ValueError(f"operation {key} contains an empty/weak field")
        try:
            fields["workflow_evidence"] = align_evidence_to_workflow(
                fields["workflow_evidence"], profile["workflow"]
            )
        except ValueError:
            # Reject only the unsupported operation instance.  Other grounded
            # operations from the same parent Memory remain valuable.
            continue
        if not evidence_supports_operation(key, fields["workflow_evidence"]):
            continue
        spec = ALLOWED[key]
        conditional = spec.get("conditional_field")
        if conditional and not profile["source_signature"].get(conditional):
            continue
        instance_index = key_counts[key]
        key_counts[key] += 1
        rows.append(
            {
                "operation_id": f"memory_{int(profile['memory_id']):03d}:{key}:{instance_index:02d}",
                "operation_instance_index": instance_index,
                "parent_memory_id": int(profile["memory_id"]),
                "parent_source": profile["source"],
                "parent_memory_query": profile["memory_query"],
                "source_signature": profile["source_signature"],
                "operation_key": key,
                "operation_family": spec["family"],
                "order": int(spec["order"]),
                "applicability_field": conditional,
                "applicability_value": (
                    profile["source_signature"].get(conditional) if conditional else None
                ),
                **fields,
                "operation_text": "\n".join(
                    [
                        f"Operation: {key}", f"Goal: {fields['goal']}",
                        f"Precondition: {fields['precondition']}",
                        f"Procedure: {fields['procedure']}",
                        f"Success evidence: {fields['success_evidence']}",
                        f"Failure mode: {fields['failure_mode']}",
                        f"Grounding from parent Memory: {fields['workflow_evidence']}",
                    ]
                ),
                "extraction": f"gpt4o_grounded:{PROMPT_VERSION}",
                "validation_or_test_used": False,
            }
        )
    returned = {row["operation_key"] for row in rows}
    workflow_terms = normalize(profile["workflow"])
    required_when_visible = {
        "flightsearch": "route_search",
        "search for flights": "route_search",
        "route closure": "route_closure",
        "accommodationsearch": "lodging_search",
        "restaurantsearch": "dining_search",
        "attractionsearch": "attraction_search",
        "notebook": "notebook_grounding",
        "minimum-night": "minimum_nights",
        "non-repeated dining": "restaurant_uniqueness",
        "no restaurant is repeated": "restaurant_uniqueness",
        "non-repeating breakfast": "restaurant_uniqueness",
        "unique attractions": "attraction_uniqueness",
        "no duplicates": "attraction_uniqueness",
        "total cost": "full_party_budget",
        "within budget": "full_party_budget",
        "before running planner": "final_constraint_audit",
        "verify the sandbox": "final_constraint_audit",
        "re-search": "repair_before_planner",
    }
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", profile["workflow"]) if part.strip()]
    for phrase, operation_key in required_when_visible.items():
        if phrase not in workflow_terms or operation_key in returned:
            continue
        evidence = next(sentence for sentence in sentences if phrase in normalize(sentence))
        spec = ALLOWED[operation_key]
        index = key_counts[operation_key]
        key_counts[operation_key] += 1
        rows.append(
            {
                "operation_id": f"memory_{int(profile['memory_id']):03d}:{operation_key}:{index:02d}",
                "operation_instance_index": index,
                "parent_memory_id": int(profile["memory_id"]),
                "parent_source": profile["source"],
                "parent_memory_query": profile["memory_query"],
                "source_signature": profile["source_signature"],
                "operation_key": operation_key,
                "operation_family": spec["family"],
                "order": int(spec["order"]),
                "applicability_field": spec.get("conditional_field"),
                "applicability_value": (
                    profile["source_signature"].get(spec["conditional_field"])
                    if spec.get("conditional_field") else None
                ),
                "goal": spec["goal"],
                "precondition": spec["precondition"],
                "procedure": spec["procedure"],
                "success_evidence": spec["success_evidence"],
                "failure_mode": spec["failure_mode"],
                "workflow_evidence": evidence,
                "operation_text": "\n".join(
                    [
                        f"Operation: {operation_key}", f"Goal: {spec['goal']}",
                        f"Precondition: {spec['precondition']}",
                        f"Procedure: {spec['procedure']}",
                        f"Success evidence: {spec['success_evidence']}",
                        f"Failure mode: {spec['failure_mode']}",
                        f"Grounding from parent Memory: {evidence}",
                    ]
                ),
                "extraction": f"deterministic_visible_clause_completion:{PROMPT_VERSION}",
                "validation_or_test_used": False,
            }
        )
        returned.add(operation_key)
    if not rows:
        raise ValueError("no grounded operation returned")
    return sorted(rows, key=lambda row: (row["order"], row["operation_key"]))


def extract_one(profile: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    prompt = prompt_for(profile)
    prompt_hash = "sha256:" + hashlib.sha256(
        (PROMPT_VERSION + "\n" + SYSTEM_PROMPT + "\n" + prompt).encode()
    ).hexdigest()
    cache = args.output_dir / "responses" / f"memory_{int(profile['memory_id']):03d}.json"
    if cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
        if result.get("status") == "success" and result.get("prompt_hash") == prompt_hash:
            raw_responses = [
                attempt.get("response") for attempt in result.get("attempts", [])
                if attempt.get("response")
            ]
            if raw_responses:
                result["operations"] = parse_response(
                    extract_content(raw_responses[-1]), profile
                )
                result["parser_version"] = PARSER_VERSION
                cache.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            return result, True
    attempts = []
    result = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            request = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": args.max_tokens,
            }
            status, body = _http_post_json(
                args.api_base.rstrip("/") + "/chat/completions",
                {"Authorization": f"Bearer {args.api_key}"}, request, args.timeout,
            )
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {body[:500]}")
            response = json.loads(body)
            operations = parse_response(extract_content(response), profile)
            attempts.append({"attempt": attempt, "response": response, "error": None})
            result = {
                "status": "success", "memory_id": int(profile["memory_id"]),
                "model": args.model, "prompt_version": PROMPT_VERSION,
                "parser_version": PARSER_VERSION,
                "prompt_hash": prompt_hash, "operations": operations,
                "attempts": attempts, "created_at": datetime.now(timezone.utc).isoformat(),
            }
            break
        except Exception as exc:
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
    if result is None:
        result = {
            "status": "failed", "memory_id": int(profile["memory_id"]),
            "model": args.model, "prompt_version": PROMPT_VERSION,
            "parser_version": PARSER_VERSION,
            "prompt_hash": prompt_hash, "attempts": attempts,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-profiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api-base", default=os.environ.get("SKILLDAG_LLM_BASE", "https://yunwu.ai/v1"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.memory_profiles.read_text(encoding="utf-8"))
    profiles = payload["profiles"] if isinstance(payload, dict) else payload
    if any(row.get("validation_or_test_used") is not False for row in profiles):
        raise ValueError("validation/test-derived Memory detected")
    if args.limit is not None:
        profiles = profiles[: args.limit]
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run", "memory_count": len(profiles), "model": args.model,
            "prompt_chars": sum(len(SYSTEM_PROMPT) + len(prompt_for(row)) for row in profiles),
            "validation_or_test_used": False,
        }, ensure_ascii=False, indent=2))
        return 0
    args.api_key = os.environ.get("SKILLDAG_LLM_API_KEY", "")
    if not args.api_key:
        raise SystemExit("SKILLDAG_LLM_API_KEY is empty")
    results: dict[int, dict[str, Any]] = {}
    cache_hits = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(extract_one, row, args): int(row["memory_id"]) for row in profiles}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result, hit = future.result()
            results[int(result["memory_id"])] = result
            cache_hits += int(hit)
            print(f"[atomic-memory] {index}/{len(profiles)} {result['status']} cache={int(hit)} memory={result['memory_id']}", flush=True)
    operations, failures = [], []
    usage: Counter[str] = Counter()
    for profile in profiles:
        result = results[int(profile["memory_id"])]
        if result["status"] != "success":
            failures.append(result)
            continue
        operations.extend(result["operations"])
        for attempt in result.get("attempts", []):
            response = attempt.get("response") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += int((response.get("usage") or {}).get(key, 0) or 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "memory_operations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in operations), encoding="utf-8"
    )
    (args.output_dir / "failures.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in failures), encoding="utf-8"
    )
    report = {
        "schema_version": "memp.travelplanner.atomic_memory_extraction.v1",
        "model": args.model, "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "memory_count": len(profiles), "successful_memory_count": len(profiles) - len(failures),
        "failed_memory_count": len(failures), "operation_count": len(operations),
        "operation_counts_by_key": dict(Counter(row["operation_key"] for row in operations)),
        "empty_evidence_count": sum(not row["workflow_evidence"] for row in operations),
        "cache_hit_count": cache_hits, "usage": dict(usage),
        "validation_or_test_used": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
