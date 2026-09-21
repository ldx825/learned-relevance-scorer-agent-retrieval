#!/usr/bin/env python3
"""Run a resumable TravelPlanner test/validation slice with Script Query retrieval."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_TEST_SHA256 = "d1c8918683455ee9986e2e61a1aa31cf8f7a493a775b6d7ff59f26a421b89cc1"
EXPECTED_VALIDATION_SHA256 = "0e54b26b13c0b6d50e8683765e930bffca949488c44634e530da899914d40e80"
SPLIT_SPECS = {
    "test": (1000, EXPECTED_TEST_SHA256),
    "validation": (180, EXPECTED_VALIDATION_SHA256),
}
VALID_TOOL_NAMES = (
    "FlightSearch",
    "AttractionSearch",
    "GoogleDistanceMatrix",
    "AccommodationSearch",
    "RestaurantSearch",
    "Planner",
    "NotebookWrite",
    "CitySearch",
)
EVIDENCE_PRODUCING_TOOLS = (
    "FlightSearch",
    "AttractionSearch",
    "GoogleDistanceMatrix",
    "AccommodationSearch",
    "RestaurantSearch",
    "CitySearch",
)


def _successful_action(record: dict[str, Any]) -> str | None:
    if str(record.get("state")) != "Successful":
        return None
    return normalize_tool_action_response(str(record.get("action", "")).strip())


def required_return_evidence_action(
    task: dict[str, Any], action_log: list[dict[str, Any]]
) -> str | None:
    """Return the next evidence action required before Planner, if any.

    This uses only visible task fields and successful tool actions. It is
    intentionally limited to direct one-city flight itineraries; other route
    topologies retain the released executor behavior.
    """

    if int(task.get("visiting_city_number") or 0) != 1:
        return None
    try:
        constraints = ast.literal_eval(str(task.get("local_constraint") or "{}"))
    except (SyntaxError, ValueError):
        constraints = {}
    transportation = str(constraints.get("transportation") or "").lower()
    if "no flight" in transportation or "without flight" in transportation:
        return None
    try:
        dates = ast.literal_eval(str(task["date"]))
    except (SyntaxError, ValueError, KeyError):
        return None
    if not isinstance(dates, list) or not dates:
        return None
    origin, destination = str(task["org"]), str(task["dest"])
    actions = [_successful_action(record) for record in action_log]
    outbound = f"FlightSearch[{origin}, {destination}, {dates[0]}]"
    if outbound not in actions:
        return None
    returning = f"FlightSearch[{destination}, {origin}, {dates[-1]}]"
    return_positions = [index for index, action in enumerate(actions) if action == returning]
    if not return_positions:
        return returning
    if not any(
        action is not None and action.startswith("NotebookWrite[")
        for action in actions[return_positions[-1] + 1 :]
    ):
        return (
            f"NotebookWrite[Verified return flight from {destination} to {origin} "
            f"on {dates[-1]}]"
        )
    return None


def required_notebook_write_action(action_log: list[dict[str, Any]]) -> str | None:
    """Force immediate persistence of the preceding successful search result."""
    if not action_log:
        return None
    action = _successful_action(action_log[-1])
    if action is None or not action.startswith(tuple(f"{name}[" for name in EVIDENCE_PRODUCING_TOOLS)):
        return None
    tool = action.split("[", 1)[0]
    return f"NotebookWrite[Preserve the exact result from the preceding {tool} action]"


def visible_transport_violation(task: dict[str, Any], action: str) -> str | None:
    """Reject tool calls that conflict with an inference-visible mode rule."""
    try:
        constraints = ast.literal_eval(str(task.get("local_constraint") or "{}"))
    except (SyntaxError, ValueError):
        constraints = {}
    transportation = str(constraints.get("transportation") or "").lower()
    if "no flight" in transportation and action.startswith("FlightSearch["):
        return "The visible task forbids flights; choose an allowed ground mode."
    if (
        ("no self-driving" in transportation or "no self driving" in transportation)
        and action.startswith("GoogleDistanceMatrix[")
        and re.search(r",\s*self[- ]driving\s*\]$", action, flags=re.IGNORECASE)
    ):
        return "The visible task forbids self-driving; use an allowed alternative such as taxi."
    return None


def normalize_tool_action_response(response: str) -> str | None:
    """Extract one allow-listed bare tool call from provider formatting noise."""

    text = response.strip()
    if not text:
        return None
    names = "|".join(map(re.escape, VALID_TOOL_NAMES))
    match = re.search(rf"(?<!\w)({names})\[([^\n\r]*)\]", text)
    if not match:
        return None
    argument = match.group(2).strip()
    if not argument:
        return None
    return f"{match.group(1)}[{argument}]"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_private_runtime(root: Path) -> tuple[str, str]:
    key = os.environ.get("MEMP_API_KEY") or os.environ.get("YUNWU_API_KEY")
    base = os.environ.get("MEMP_API_BASE") or os.environ.get("YUNWU_BASE_URL")
    if not key or not base:
        raise RuntimeError("Set MEMP_API_KEY/MEMP_API_BASE or YUNWU_API_KEY/YUNWU_BASE_URL")
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_API_BASE"] = base.rstrip("/")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(root / ".runtime/memp/travelplanner/tiktoken"))
    return base.rstrip("/"), os.environ.get("MEMP_CHAT_MODEL", "gpt-4o")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(SPLIT_SPECS), default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-api-attempts", type=int, default=3)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument(
        "--route-evidence-guard",
        action="store_true",
        help="Require visible-task return-route evidence before Planner (shared ablation).",
    )
    parser.add_argument(
        "--operation-contract-guard",
        action="store_true",
        help=(
            "Enforce immediate Notebook writes and visible transportation constraints "
            "for instantiated-operation ablations."
        ),
    )
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument(
        "--memory",
        type=Path,
        default=Path(".runtime/memp/travelplanner/memory_full_v1/script/documents.json"),
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=Path(".runtime/memp/travelplanner/script_query_cosine_test100/retrieval.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runtime/memp/travelplanner/script_query_cosine_test100/results"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    tp_root = root / "references/TravelPlanner"
    default_csv = Path(f".runtime/memp/travelplanner/hf_dataset_repo/{args.split}.csv")
    csv_arg = args.data_csv or args.test_csv or default_csv
    test_csv = csv_arg if csv_arg.is_absolute() else root / csv_arg
    memory_path = args.memory if args.memory.is_absolute() else root / args.memory
    retrieval_path = args.retrieval if args.retrieval.is_absolute() else root / args.retrieval
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    expected_count, expected_sha256 = SPLIT_SPECS[args.split]
    if sha256(test_csv) != expected_sha256:
        raise ValueError(f"TravelPlanner {args.split}.csv does not match the pinned official LFS object")
    with test_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    memories = json.loads(memory_path.read_text(encoding="utf-8"))
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    retrieval_by_index = {int(item["test_index"]): item for item in retrieval["retrievals"]}
    retrieval_slice = retrieval.get("task_slice") or retrieval.get("test_slice")
    if not retrieval_slice or len(retrieval_slice) != 2:
        raise ValueError("retrieval artifact must declare task_slice or test_slice")
    if retrieval.get("selected_indices") is not None:
        selected = [int(value) for value in retrieval["selected_indices"]]
        if len(selected) != len(set(selected)) or any(index not in retrieval_by_index for index in selected):
            raise ValueError("retrieval selected_indices are invalid or missing retrieval rows")
    else:
        retrieval_limit = int(retrieval_slice[1])
        end = min(args.start + args.count, retrieval_limit, len(rows))
        selected = list(range(args.start, end))
    if not selected:
        raise ValueError(f"No {args.split} query selected in the predeclared retrieval slice")

    api_base, model = configure_private_runtime(root)
    agents_dir = tp_root / "agents"
    os.chdir(agents_dir)
    sys.path.insert(0, str(agents_dir))
    sys.path.insert(0, str(tp_root))
    import tool_agents
    from langchain.schema import HumanMessage
    from langchain_community.callbacks import get_openai_callback

    class BoundedReactAgent(tool_agents.ReactAgent):
        evidence_guard_task: dict[str, Any] | None = None

        def prompt_agent(self) -> str:
            error: Exception | None = None
            action_phase = self.scratchpad.rstrip().endswith(f"Action {self.step_n}:")
            guard_action: str | None = None
            guard_feedback: str | None = None
            if args.operation_contract_guard and action_phase:
                notebook_action = required_notebook_write_action(self.json_log[:-1])
                if notebook_action is not None:
                    return notebook_action
            for attempt in range(args.max_api_attempts):
                try:
                    prompt = self._build_agent_prompt()
                    if guard_action is not None:
                        prompt += (
                            "\nEvidence gate: Planner cannot run yet because the visible task's "
                            "return-route evidence is incomplete. Output exactly this next tool call "
                            f"and nothing else: {guard_action}"
                        )
                    if guard_feedback is not None:
                        prompt += (
                            "\nVisible-constraint gate rejected the previous proposed action: "
                            f"{guard_feedback} Output one allowed tool call instead."
                        )
                    if action_phase and attempt:
                        prompt += (
                            "\nFormat correction: output exactly one bare tool call and nothing else. "
                            "Do not use Markdown fences and do not prefix it with Thought, Action, "
                            "Observation, or a step number."
                        )
                    response = self.llm([HumanMessage(content=prompt)]).content
                    if not action_phase:
                        return tool_agents.format_step(response)
                    normalized = normalize_tool_action_response(response)
                    if normalized is not None:
                        if (
                            args.operation_contract_guard
                            and self.evidence_guard_task is not None
                        ):
                            guard_feedback = visible_transport_violation(
                                self.evidence_guard_task, normalized
                            )
                            if guard_feedback is not None:
                                error = ValueError(guard_feedback)
                                continue
                        if (
                            args.route_evidence_guard
                            and normalized.startswith("Planner[")
                            and self.evidence_guard_task is not None
                        ):
                            guard_action = required_return_evidence_action(
                                self.evidence_guard_task, self.json_log[:-1]
                            )
                            if guard_action is not None:
                                error = ValueError(
                                    f"Planner blocked pending route evidence: {guard_action}"
                                )
                                continue
                        return normalized
                    error = ValueError(
                        f"provider returned no parseable allow-listed tool call: {response!r}"
                    )
                except Exception as exc:
                    error = exc
                if attempt + 1 < args.max_api_attempts:
                    time.sleep(min(15, 2**attempt))
            raise RuntimeError(
                f"tool-agent response failed API/format validation after "
                f"{args.max_api_attempts} attempts: {error}"
            )

    tools = [
        "notebook",
        "flights",
        "attractions",
        "accommodations",
        "restaurants",
        "googleDistanceMatrix",
        "planner",
        "cities",
    ]
    agent = BoundedReactAgent(
        None,
        tools=tools,
        max_steps=30,
        max_retries=3,
        illegal_early_stop_patience=3,
        react_llm_name=model,
        planner_llm_name=model,
    )

    for test_index in selected:
        output_path = output_dir / f"test_{test_index:04d}.json"
        if output_path.is_file():
            print(f"[resume] keeping {output_path}", flush=True)
            continue
        row = rows[test_index]
        agent.evidence_guard_task = (
            row if args.route_evidence_guard or args.operation_contract_guard else None
        )
        original_query = row["query"]
        retrieval_item = retrieval_by_index[test_index]
        hits = retrieval_item.get("hits", [])
        if "dynamic_memory_text" in retrieval_item:
            memory_payload = [{
                "task_name": "Task-specific procedural memory",
                "guidelines": retrieval_item["dynamic_memory_text"],
            }]
        else:
            memory_payload = [
                {
                    "task_name": memories[int(hit["memory_index"])] ["metadata"]["query"],
                    "guidelines": memories[int(hit["memory_index"])] ["metadata"]["workflow"],
                }
                for hit in hits
            ]
        memory_text = json.dumps(memory_payload, indent=2, ensure_ascii=False)
        agent_query = (
            original_query
            + "\n\nHere are some guidelines of how to solve similar tasks:\n"
            + memory_text
        )
        started = time.time()
        with get_openai_callback() as callback:
            final_plan, scratchpad, action_log = agent.run(agent_query)
        record = {
            "split": args.split,
            "test_index": test_index,
            "query": original_query,
            "agent_query": agent_query,
            "level": row["level"],
            "source": str(test_csv),
            "source_sha256": expected_sha256,
            "model": model,
            "api_base_host": api_base.split("://", 1)[-1].split("/", 1)[0],
            "settings": {
                "mode": "two-stage",
                "temperature": 0,
                "max_steps": 30,
                "max_tool_retries": 3,
                "tool_call_max_tokens": 256,
                "planner_max_tokens": 4096,
                "memory": True,
                "memory_format": "script",
                "retriever": retrieval.get("retriever", retrieval.get("reranker", "unknown")),
                "retrieve_num": retrieval.get("top_k", 1),
                "score_threshold": retrieval.get("faiss_l2_threshold"),
                "ncf": "dynamic_memory_text" in retrieval_item,
                "allowlisted_action_normalization": True,
                "route_evidence_guard": bool(args.route_evidence_guard),
                "operation_contract_guard": bool(args.operation_contract_guard),
            },
            "retrieval": retrieval_item,
            "retrieved_memories": memory_payload,
            "final_plan_text": final_plan,
            "scratchpad": scratchpad,
            "trajectory": action_log,
            "executed_steps": len(action_log),
            "elapsed_seconds": round(time.time() - started, 3),
            "callback_usage": {
                "prompt_tokens": int(callback.prompt_tokens),
                "completion_tokens": int(callback.completion_tokens),
                "total_tokens": int(callback.total_tokens),
                "total_cost": float(callback.total_cost),
            },
            "test_gold_used": False,
            "allowed_use": (
                "validation evaluation only; forbidden as NCF training data"
                if args.split == "validation"
                else "test smoke/error analysis only; forbidden as NCF training data"
            ),
        }
        atomic_json(output_path, record)
        if args.postprocess:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/memp/postprocess_travelplanner_pilot.py"),
                    str(output_path),
                ],
                cwd=root,
                check=False,
            )
            print(
                f"[postprocess] test {test_index:04d} exit={completed.returncode}",
                flush=True,
            )
        print(
            json.dumps(
                {
                    "saved": str(output_path),
                    "test_index": test_index,
                    "retrieved": len(hits),
                    "steps": len(action_log),
                    "tokens": int(callback.total_tokens),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
