"""Auditable MemP ALFWorld Script baseline utilities.

This module keeps the released MemP algorithmic choices (300 train
trajectories, query-key retrieval, Top-10 injection) while making every paid
artifact resumable and every retrieval decision inspectable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from openai import OpenAI


TASK_RE = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_task_query(observation: str) -> str:
    match = TASK_RE.search(observation)
    if not match:
        raise ValueError("ALFWorld observation has no 'Your task is to:' goal")
    return match.group(1).strip().rstrip(".")


def strip_welcome(observation: str) -> str:
    parts = observation.split("\n\n")
    return "\n\n".join(parts[1:]) if len(parts) > 1 else observation


def alfworld_script_messages(query: str, trajectory: Any) -> list[dict[str, str]]:
    """Faithful reconstruction of the paper-described ALFWorld Script prompt."""
    system = "You distill successful ALFWorld trajectories into reusable procedural memory."
    user = f"""You are given an ALFWorld household task and a successful interaction trajectory.
Extract only the critical, reusable procedure that would help an agent solve similar tasks.

Requirements:
- Preserve the required operation and its order, such as locate, open, take, clean, heat, cool, use, and place.
- State important preconditions, for example holding the object or using the correct appliance.
- Generalize away instance numbers and accidental exploration specific to this room.
- Do not invent objects, locations, actions, or observations absent from the trajectory.
- Write one concise natural-language paragraph, not a numbered list.

Task:
{query}

Successful trajectory:
{json.dumps(trajectory, ensure_ascii=False)}

Reusable workflow:"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def paper_minimal_script_messages(query: str, trajectory: Any) -> list[dict[str, str]]:
    """Minimal reconstruction of the paper-described Script builder.

    Unlike ``alfworld_script_messages``, this prompt does not inject an
    ALFWorld-specific action ontology, appliance hints, or explicit
    precondition rules.  It only asks the model to abstract the successful
    evidence already present in the trajectory.  This makes it suitable for
    diagnosing how much of the downstream gain comes from prompt-engineered
    procedural knowledge rather than memory retrieval itself.
    """
    system = "You are a helpful assistant."
    user = f"""You are given a task query and a successful trajectory.

Summarize the critical successful steps into a reusable high-level workflow that may help solve similar tasks in the future.

Write one concise and coherent paragraph. Generalize away object instance numbers and irrelevant exploration. Only use information supported by the trajectory. Do not provide explanations outside the workflow.

Query:
{query}

Trajectory:
{json.dumps(trajectory, ensure_ascii=False)}

Workflow:"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def paper_minimal_v2_script_messages(query: str, trajectory: Any) -> list[dict[str, str]]:
    """Compact evidence-only reconstruction matching the paper's case-study style."""
    system = "You are a helpful assistant."
    user = f"""You are given a task query and a successful trajectory.

Turn the trajectory into a compact reusable script. Retain only the actions that directly contributed to success, in the order they were performed. Replace instance numbers with generic object and receptacle names.

Do not add advice, checks, alternatives, likely locations, preconditions, tools, or actions that are not explicitly supported by the trajectory. Write one plain paragraph of 2 to 4 sentences and no more than 60 English words. Do not use bullets or numbered steps. Output only the script.

Query:
{query}

Trajectory:
{json.dumps(trajectory, ensure_ascii=False)}

Script:"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def paper_minimal_v3_script_messages(query: str, trajectory: Any) -> list[dict[str, str]]:
    """Balanced evidence-only Script reconstruction for the full-bank candidate."""
    system = "You are a helpful assistant."
    user = f"""You are given a task query and a successful trajectory.

Write a compact reusable script that preserves every goal-critical action and their successful order. Keep the target object, target receptacle, required transformation, and any tool or appliance actually used by the successful trajectory. Remove instance numbers, failed exploration, repeated navigation, and incidental room details.

Do not add likely locations, alternative tools, advice, checks, preconditions, or actions not supported by the trajectory. Write one plain paragraph of 30 to 60 English words, with no bullets or numbered steps. Output only the script.

Query:
{query}

Trajectory:
{json.dumps(trajectory, ensure_ascii=False)}

Script:"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def release_exact_messages(query: str, trajectory: Any) -> list[dict[str, str]]:
    """The released prompt, retained only to diagnose the public-code gap."""
    system = "You are a helpful assistant."
    user = f"""You are provided with a query and a trajectory taken to solve the query. The trajectory consists of multiple steps of thought, action and observation.
Your task is to generate a workflow based on critical steps to help solve similar queries in the future.
A critical step is one that have a significant impact on fulfilling the query, the step action belongs to the set [FlightSearch, GoogleDistanceMatrix, AccommodationSearch, RestaurantSearch, AttractionSearch, CitySearch, NotebookWrite, Planner], and the action's outcome is successful and contributes positively to achieving the query.
Notice: Write the workflow as a natural, coherent paragraph (not as a bullet list or numbered steps). Use clear, concise language to describe what actions should be taken and in what general order.
Query:
{query}
Trajectory:
{trajectory}
Output the workflow without any explanation or context:"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def release_corrected_messages(query: str, trajectory: Any) -> list[dict[str, str]]:
    """Released direct-build prompt with only cross-benchmark defects removed."""
    system = "You are a helpful assistant."
    user = f"""You are provided with a query and a trajectory taken to solve the query. The trajectory consists of multiple steps of thought, action and observation.
Your task is to generate a workflow based on critical steps to help solve similar queries in the future.
A critical step is one that have a significant impact on fulfilling the query, the step action belongs to the set [go, take, put, open, close, toggle, clean, heat, cool], and the action's outcome is successful and contributes positively to achieving the query.
Notice: Write the workflow as a natural, coherent paragraph (not as a bullet list or numbered steps). Use clear, concise language to describe what actions should be taken and in what general order.
Query:
{query}
Trajectory:
{trajectory}
Output the workflow without any explanation or context:"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class ApiConfig:
    api_key: str
    base_url: str
    chat_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    build_temperature: float = 0.2
    agent_temperature: float = 1.0
    retries: int = 10

    @classmethod
    def from_env(cls) -> "ApiConfig":
        key = os.environ.get("MEMP_API_KEY") or os.environ.get("YUNWU_API_KEY")
        base = os.environ.get("MEMP_API_BASE") or os.environ.get("YUNWU_BASE_URL")
        if not key or not base:
            raise RuntimeError("Set MEMP_API_KEY/MEMP_API_BASE or YUNWU_API_KEY/YUNWU_BASE_URL")
        return cls(
            api_key=key,
            base_url=base,
            chat_model=os.environ.get("MEMP_CHAT_MODEL", "gpt-4o"),
            embedding_model=os.environ.get("MEMP_EMBEDDING_MODEL", "text-embedding-3-small"),
        )


class CachedOpenAI:
    def __init__(self, config: ApiConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(self, messages: list[dict[str, str]], temperature: float) -> tuple[str, dict[str, int]]:
        error: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.chat_model,
                    messages=messages,
                    temperature=temperature,
                )
                usage = response.usage
                return response.choices[0].message.content or "", {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
            except Exception as exc:  # paid calls must be resumable
                error = exc
                if attempt + 1 < self.config.retries:
                    time.sleep(min(30, 2**attempt))
        raise RuntimeError(f"chat request failed after {self.config.retries} attempts: {error}")

    def embeddings(
        self,
        texts: list[str],
        batch_size: int = 64,
        model: str | None = None,
    ) -> list[list[float]]:
        values: list[list[float]] = []
        embedding_model = model or self.config.embedding_model
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            error: Exception | None = None
            for attempt in range(self.config.retries):
                try:
                    response = self.client.embeddings.create(
                        model=embedding_model,
                        input=batch,
                    )
                    ordered = sorted(response.data, key=lambda item: item.index)
                    values.extend([item.embedding for item in ordered])
                    break
                except Exception as exc:
                    error = exc
                    if attempt + 1 < self.config.retries:
                        time.sleep(min(30, 2**attempt))
            else:
                raise RuntimeError(f"embedding request failed: {error}")
        return values


def build_script_bank(
    trajectory_path: Path,
    output_path: Path,
    api: CachedOpenAI,
    memory_size: int = 300,
    workers: int = 16,
    prompt_mode: str = "alfworld-faithful",
) -> dict[str, Any]:
    trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))[:memory_size]
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        payload = {"records": []}
    cached = {int(item["source_index"]): item for item in payload.get("records", [])}
    prompt_fns = {
        "alfworld-faithful": alfworld_script_messages,
        "paper-minimal": paper_minimal_script_messages,
        "paper-minimal-v2": paper_minimal_v2_script_messages,
        "paper-minimal-v3": paper_minimal_v3_script_messages,
        "release-exact": release_exact_messages,
        "release-corrected": release_corrected_messages,
    }
    try:
        prompt_fn = prompt_fns[prompt_mode]
    except KeyError as exc:
        raise ValueError(f"unknown prompt mode: {prompt_mode}") from exc

    def build_one(index: int) -> dict[str, Any]:
        row = trajectories[index]
        query = row["query"].split("\n\n")[0].strip()
        messages = prompt_fn(query, row["trajectory"])
        workflow, usage = api.chat(messages, api.config.build_temperature)
        return {
            "source_index": index,
            "source": row.get("source"),
            "query": query,
            "workflow": workflow.strip(),
            "facts": row.get("facts"),
            "usage": usage,
        }

    pending = [index for index in range(len(trajectories)) if index not in cached]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_one, index): index for index in pending}
        for future in as_completed(futures):
            item = future.result()
            cached[item["source_index"]] = item
            payload = {
                "schema_version": 1,
                "trajectory_path": str(trajectory_path),
                "trajectory_sha256": sha256_file(trajectory_path),
                "memory_size": memory_size,
                "build_model": api.config.chat_model,
                "build_temperature": api.config.build_temperature,
                "prompt_mode": prompt_mode,
                "records": [cached[i] for i in sorted(cached)],
            }
            atomic_json(output_path, payload)
    return payload


def ensure_bank_embeddings(bank_path: Path, embedding_path: Path, api: CachedOpenAI) -> np.ndarray:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    queries = [item["query"] for item in bank["records"]]
    if embedding_path.exists():
        payload = json.loads(embedding_path.read_text(encoding="utf-8"))
        if payload.get("queries") == queries and payload.get("model") == api.config.embedding_model:
            return np.asarray(payload["embeddings"], dtype=np.float32)
    embeddings = api.embeddings(queries)
    atomic_json(embedding_path, {"model": api.config.embedding_model, "queries": queries, "embeddings": embeddings})
    return np.asarray(embeddings, dtype=np.float32)


def rank_memories(
    query_embedding: Iterable[float],
    bank_embeddings: np.ndarray,
    top_k: int = 10,
    faiss_l2_threshold: float | None = 0.5,
) -> list[dict[str, float | int]]:
    query = np.asarray(query_embedding, dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    bank = bank_embeddings / np.maximum(np.linalg.norm(bank_embeddings, axis=1, keepdims=True), 1e-12)
    cosine = bank @ query
    squared_l2 = 2.0 - 2.0 * cosine
    order = np.argsort(-cosine, kind="stable")
    results: list[dict[str, float | int]] = []
    for index in order:
        if faiss_l2_threshold is not None and float(squared_l2[index]) > faiss_l2_threshold:
            continue
        results.append({
            "memory_index": int(index),
            "cosine": float(cosine[index]),
            "squared_l2": float(squared_l2[index]),
        })
        if len(results) == top_k:
            break
    return results


def sum_usage(items: Iterable[dict[str, int]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for item in items:
        for key in totals:
            totals[key] += int(item.get(key, 0) or 0)
    return totals


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
