#!/usr/bin/env python3
"""Run auditable MemP procedural-memory ALFWorld evaluations."""

from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from alfworld.agents.environment import get_environment

from Alfworld.prompts import alfworld_system_prompt
from memp_repro import (
    ApiConfig,
    CachedOpenAI,
    atomic_json,
    ensure_bank_embeddings,
    extract_task_query,
    rank_memories,
    strip_welcome,
    sum_usage,
)
from ncf_retriever import MemoryNCFReranker


HERE = Path(__file__).resolve().parent
PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def task_name_from_gamefile(gamefile: str) -> str:
    path = Path(gamefile)
    return f"{path.parent.parent.name}/{path.parent.name}"


def example_for(name: str, examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    for prefix, task in PREFIXES.items():
        if name.startswith(prefix):
            for example in examples:
                if example["task"] == task:
                    return copy.deepcopy(example["example"])
    raise ValueError(f"No few-shot example for {name}")


def process_observation(text: str) -> str:
    if text.startswith("You arrive at loc "):
        return text[text.find(". ") + 2 :]
    return text


def invalid_observation(text: str) -> bool:
    lower = text.lower()
    markers = ("nothing happened", "nothing happens", "can't", "cannot", "not a valid", "invalid")
    return any(marker in lower for marker in markers)


def parse_indices(value: str | None) -> set[int] | None:
    """Parse a comma-separated, zero-based evaluation subset."""
    if value is None:
        return None
    indices = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not indices or min(indices) < 0:
        raise ValueError("--indices must contain non-negative integer positions")
    return indices


def make_messages(
    observation: str,
    name: str,
    examples: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    memory_format: str = "script",
) -> list[dict[str, str]]:
    if memory_format == "script":
        memory_view = [
            {"task_name": item["query"], "guidelines": item["workflow"]}
            for item in memories
        ]
        heading = "Here are some guidelines of how to solve similar tasks:"
    elif memory_format == "trajectory":
        memory_view = [
            {"task_name": item["query"], "successful_trajectory": item["trajectory"]}
            for item in memories
        ]
        heading = "Here are successful trajectories from similar tasks:"
    elif memory_format == "proceduralization":
        memory_view = [
            {
                "task_name": item["query"],
                "guidelines": item["workflow"],
                "successful_trajectory": item["trajectory"],
            }
            for item in memories
        ]
        heading = (
            "Here are guidelines and their supporting successful trajectories "
            "from similar tasks:"
        )
    else:
        raise ValueError(f"unsupported memory format: {memory_format}")
    augmented = observation + f"\n\n{heading}\n" + json.dumps(
        memory_view, indent=2, ensure_ascii=False
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": alfworld_system_prompt}]
    shot = example_for(name, examples)
    shot[0]["content"] = "Here is an example of how to solve the task:\nExample:\n" + shot[0]["content"]
    messages.extend(shot)
    messages.append({"role": "user", "content": "Now it's your turn.\n" + augmented})
    return messages


def run_batch(
    env: Any,
    raw_observations: list[str],
    gamefiles: list[str],
    bank: list[dict[str, Any]],
    bank_embeddings: np.ndarray,
    api: CachedOpenAI,
    examples: list[dict[str, Any]],
    max_steps: int,
    top_k: int,
    threshold: float | None,
    target_count: int | None = None,
    environment_batch_size: int | None = None,
    environment_indices: list[int] | None = None,
    ncf_reranker: MemoryNCFReranker | None = None,
    ncf_candidate_k: int = 24,
    ncf_candidate_threshold: float | None = None,
    memory_format: str = "script",
) -> list[dict[str, Any]]:
    if environment_indices is None:
        if target_count is None:
            target_count = len(raw_observations)
        if not 0 < target_count <= len(raw_observations):
            raise ValueError(
                f"target_count must be in [1, {len(raw_observations)}], got {target_count}"
            )
        environment_indices = list(range(target_count))
    elif not environment_indices:
        raise ValueError("environment_indices must not be empty")
    if len(set(environment_indices)) != len(environment_indices):
        raise ValueError("environment_indices must be unique")
    action_slots = environment_batch_size or len(raw_observations)
    if min(environment_indices) < 0 or max(environment_indices) >= action_slots:
        raise ValueError(
            f"environment_indices must be within [0, {action_slots}), got {environment_indices}"
        )

    # ALFWorld's final reset may expose a full fixed-size environment batch even
    # when only a smaller evaluation tail remains.  Query the API only for the
    # real tail tasks; the unused environment slots receive harmless padding
    # actions below so TextWorld still gets exactly one action per environment.
    raw_observations = [raw_observations[index] for index in environment_indices]
    gamefiles = [gamefiles[index] for index in environment_indices]
    queries = [extract_task_query(observation) for observation in raw_observations]
    query_embeddings = api.embeddings(queries)
    names = [task_name_from_gamefile(gamefile) for gamefile in gamefiles]
    candidate_k = ncf_candidate_k if ncf_reranker is not None else top_k
    candidate_threshold = ncf_candidate_threshold if ncf_reranker is not None else threshold
    retrieval_candidates = [
        rank_memories(
            vector,
            bank_embeddings,
            top_k=candidate_k,
            faiss_l2_threshold=candidate_threshold,
        )
        for vector in query_embeddings
    ]
    retrievals = retrieval_candidates
    if ncf_reranker is not None:
        ncf_query_embeddings = api.embeddings(queries, model=ncf_reranker.embedding_model)
        retrievals = [
            ncf_reranker.rerank_query(query, vector, hits, top_k=top_k)
            for query, vector, hits in zip(
                queries, ncf_query_embeddings, retrieval_candidates, strict=True
            )
        ]
    memory_lists = [
        [dict(bank[int(hit["memory_index"])], retrieval=hit) for hit in hits]
        for hits in retrievals
    ]
    observations = [strip_welcome(text) for text in raw_observations]
    messages = [
        make_messages(observation, name, examples, memories, memory_format)
        for observation, name, memories in zip(observations, names, memory_lists)
    ]
    active = list(range(len(observations)))
    usages: list[list[dict[str, int]]] = [[] for _ in observations]
    errors: list[list[str]] = [[] for _ in observations]
    invalid_counts = [0 for _ in observations]
    steps = [0 for _ in observations]
    rewards = [0 for _ in observations]
    final_done = [False for _ in observations]

    for _ in range(max_steps):
        if not active:
            break
        responses: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(active)) as pool:
            futures = {
                pool.submit(api.chat, messages[index], api.config.agent_temperature): index
                for index in active
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    response, usage = future.result()
                    responses[index] = response
                    usages[index].append(usage)
                except Exception as exc:
                    errors[index].append(str(exc))

        actions = ["look" for _ in range(action_slots)]
        for index in active:
            response = responses.get(index, "")
            messages[index].append({"role": "assistant", "content": response})
            if "Action: " in response:
                actions[environment_indices[index]] = (
                    response.rsplit("Action: ", 1)[-1].strip().splitlines()[0]
                )
            else:
                invalid_counts[index] += 1
                errors[index].append("missing Action: marker")
            steps[index] += 1

        next_observation, _, done, info = env.step(actions)
        won = info["won"]
        new_active: list[int] = []
        for index in active:
            slot = environment_indices[index]
            observation = process_observation(next_observation[slot])
            if invalid_observation(observation):
                invalid_counts[index] += 1
            messages[index].append({"role": "user", "content": f"Observation: {observation}"})
            rewards[index] = int(bool(won[slot]))
            final_done[index] = bool(done[slot])
            if not done[slot]:
                new_active.append(index)
        active = new_active

    results = []
    for index in range(len(observations)):
        results.append({
            "task_id": names[index],
            "gamefile": gamefiles[index],
            "query": queries[index],
            "reward": rewards[index],
            "steps": steps[index],
            "done": final_done[index],
            "invalid_actions": invalid_counts[index],
            "usage": sum_usage(usages[index]),
            "retrieval": retrievals[index],
            "retrieval_candidates": retrieval_candidates[index],
            "retriever": "query_cosine_plus_content_neumf" if ncf_reranker else "query_cosine",
            "memory_format": memory_format,
            "retrieved_memories": [
                {
                    "source_index": item["source_index"],
                    "query": item["query"],
                    **(
                        {"workflow": item["workflow"]}
                        if memory_format == "script"
                        else (
                            {"trajectory": item["trajectory"]}
                            if memory_format == "trajectory"
                            else {
                                "workflow": item["workflow"],
                                "trajectory": item["trajectory"],
                            }
                        )
                    ),
                    "retrieval": item["retrieval"],
                }
                for item in memory_lists[index]
            ],
            "messages": messages[index],
            "errors": errors[index],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--config", type=Path, default=HERE / "Alfworld/base_config.yaml")
    parser.add_argument("--examples", type=Path, default=HERE / "Alfworld/alfworld_examples.json")
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument(
        "--memory-format",
        choices=["script", "trajectory", "proceduralization"],
        default="script",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--indices",
        help="Optional comma-separated zero-based task positions to evaluate within --limit.",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--faiss-l2-threshold", type=float, default=0.5)
    parser.add_argument(
        "--retriever",
        choices=["query_cosine", "query_cosine_ncf"],
        default="query_cosine",
    )
    parser.add_argument("--ncf-checkpoint", type=Path)
    parser.add_argument("--ncf-memory-embeddings", type=Path)
    parser.add_argument("--ncf-candidate-k", type=int, default=24)
    parser.add_argument(
        "--ncf-candidate-threshold",
        type=float,
        default=None,
        help="Optional squared-L2 threshold for NCF candidate recall; default keeps the full Top-C.",
    )
    parser.add_argument("--ncf-embedding-model", default="text-embedding-3-large")
    args = parser.parse_args()
    try:
        selected_indices = parse_indices(args.indices)
    except ValueError as exc:
        parser.error(str(exc))

    api = CachedOpenAI(ApiConfig.from_env())
    # The trajectory bank deliberately carries both the raw trajectory and its
    # paired Script workflow.  Proceduralization therefore preserves a single
    # 300-item memory identity space instead of concatenating two 300-item banks.
    bank_filename = (
        "trajectory_bank.json"
        if args.memory_format == "proceduralization"
        else f"{args.memory_format}_bank.json"
    )
    bank_path = args.memory_dir / bank_filename
    embedding_path = args.memory_dir / "query_embeddings.json"
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank = bank_payload["records"]
    bank_embeddings = ensure_bank_embeddings(bank_path, embedding_path, api)
    ncf_reranker = None
    if args.retriever == "query_cosine_ncf":
        if args.ncf_checkpoint is None or args.ncf_memory_embeddings is None:
            parser.error("query_cosine_ncf requires --ncf-checkpoint and --ncf-memory-embeddings")
        if args.ncf_candidate_k < args.top_k:
            parser.error("--ncf-candidate-k must be >= --top-k")
        ncf_reranker = MemoryNCFReranker(
            args.ncf_checkpoint,
            args.ncf_memory_embeddings,
            bank,
            embedding_model=args.ncf_embedding_model,
        )

    config = yaml.safe_load(os.path.expandvars(args.config.read_text(encoding="utf-8")))
    split_name = "eval_in_distribution" if args.split == "dev" else "eval_out_of_distribution"
    env = get_environment(config["env"]["type"])(config, train_eval=split_name)
    env = env.init_env(batch_size=args.batch_size)
    total = min(len(env.gamefiles), args.limit if args.limit > 0 else len(env.gamefiles))
    if selected_indices is not None:
        out_of_range = sorted(index for index in selected_indices if index >= total)
        if out_of_range:
            parser.error(f"--indices outside evaluated range [0, {total}): {out_of_range}")
    examples = json.loads(args.examples.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    batch_index = 0
    while completed < total:
        raw_observations, info = env.reset()
        gamefiles = list(info["extra.gamefile"])
        take = min(len(raw_observations), total - completed)
        offsets = [
            offset
            for offset in range(take)
            if selected_indices is None or completed + offset in selected_indices
        ]
        if not offsets:
            completed += take
            continue
        expected_paths = [
            args.output_dir / f"idx_{completed + offset:03d}.json" for offset in offsets
        ]
        if expected_paths and all(path.exists() for path in expected_paths):
            completed += take
            print(json.dumps({"resume_skipped": len(offsets), "completed_position": completed}))
            continue
        pending_offsets = [
            offset
            for offset, path in zip(offsets, expected_paths, strict=True)
            if not path.exists()
        ]
        results = run_batch(
            env=env,
            raw_observations=raw_observations,
            gamefiles=gamefiles,
            bank=bank,
            bank_embeddings=bank_embeddings,
            api=api,
            examples=examples,
            max_steps=args.max_steps,
            top_k=args.top_k,
            threshold=args.faiss_l2_threshold,
            environment_indices=pending_offsets,
            environment_batch_size=args.batch_size,
            ncf_reranker=ncf_reranker,
            ncf_candidate_k=args.ncf_candidate_k,
            ncf_candidate_threshold=args.ncf_candidate_threshold,
            memory_format=args.memory_format,
        )
        for offset, result in zip(pending_offsets, results, strict=True):
            result["global_index"] = completed + offset
            result["split"] = args.split
            atomic_json(args.output_dir / f"idx_{completed + offset:03d}.json", result)
        completed += take
        batch_index += 1
        successes = 0
        all_results = []
        for path in sorted(args.output_dir.glob("idx_*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            all_results.append(item)
            successes += int(item["reward"])
        summary = {
            "split": args.split,
            "completed": len(all_results),
            "successes": successes,
            "success_rate": successes / len(all_results) if all_results else 0.0,
            "average_steps": sum(item["steps"] for item in all_results) / len(all_results),
            "average_invalid_actions": sum(item["invalid_actions"] for item in all_results) / len(all_results),
            "usage": sum_usage(item["usage"] for item in all_results),
            "retriever": args.retriever,
            "memory_format": args.memory_format,
            "top_k": args.top_k,
            "ncf_candidate_k": args.ncf_candidate_k if ncf_reranker else None,
            "selected_indices": sorted(selected_indices) if selected_indices is not None else None,
        }
        atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
