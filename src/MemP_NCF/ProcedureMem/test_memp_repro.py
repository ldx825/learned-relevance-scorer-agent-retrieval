from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memp_repro import (
    extract_task_query,
    paper_minimal_script_messages,
    paper_minimal_v2_script_messages,
    paper_minimal_v3_script_messages,
    rank_memories,
    release_corrected_messages,
    strip_welcome,
)
from build_trajectory_memory import interaction_trajectory
from eval_script_query_cosine import make_messages
from eval_script_query_cosine import parse_indices


def test_extract_task_query_uses_goal_not_room_description():
    observation = "Welcome\n\nRoom has a microwave.\n\nYour task is to: put a clean fork in countertop."
    assert extract_task_query(observation) == "put a clean fork in countertop"
    assert strip_welcome(observation).startswith("Room has")


def test_query_cosine_ranking_and_release_threshold():
    bank = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    ranked = rank_memories([1.0, 0.0], bank, top_k=10, faiss_l2_threshold=0.5)
    assert [item["memory_index"] for item in ranked] == [0, 1]
    assert ranked[0]["cosine"] > ranked[1]["cosine"]


def test_paper_minimal_prompt_does_not_inject_domain_hints():
    messages = paper_minimal_script_messages(
        "heat an egg",
        [{"from": "gpt", "value": "Action: use appliance 1"}],
    )
    prompt = messages[1]["content"].lower()
    assert "successful trajectory" in prompt
    assert "microwave" not in prompt
    assert "precondition" not in prompt
    assert "flightsearch" not in prompt


def test_paper_minimal_v2_prompt_is_compact_and_evidence_only():
    messages = paper_minimal_v2_script_messages(
        "heat an egg",
        [{"from": "gpt", "value": "Action: use appliance 1"}],
    )
    prompt = messages[1]["content"].lower()
    assert "no more than 60 english words" in prompt
    assert "not explicitly supported by the trajectory" in prompt
    assert "microwave" not in prompt
    assert "flightsearch" not in prompt


def test_paper_minimal_v3_prompt_preserves_goal_chain_without_domain_hints():
    messages = paper_minimal_v3_script_messages(
        "heat an egg",
        [{"from": "gpt", "value": "Action: use appliance 1"}],
    )
    prompt = messages[1]["content"].lower()
    assert "every goal-critical action" in prompt
    assert "30 to 60 english words" in prompt
    assert "not supported by the trajectory" in prompt
    assert "microwave" not in prompt
    assert "flightsearch" not in prompt


def test_release_corrected_prompt_only_contains_alfworld_domain():
    messages = release_corrected_messages(
        "heat an egg",
        [{"from": "gpt", "value": "Action: heat egg with microwave"}],
    )
    prompt = messages[1]["content"].lower()
    assert "[go, take, put, open, close, toggle, clean, heat, cool]" in prompt
    assert "natural, coherent paragraph" in prompt
    assert "flightsearch" not in prompt
    assert "accommodation" not in prompt
    assert "planner" not in prompt


def test_trajectory_memory_drops_only_generic_bootstrap():
    messages = [
        {"from": "human", "value": "generic instruction"},
        {"from": "gpt", "value": "OK"},
        {"from": "human", "value": "room and task"},
        {"from": "gpt", "value": "Thought: plan\nAction: go to sink 1"},
    ]
    assert interaction_trajectory(messages) == [
        {"role": "environment", "content": "room and task"},
        {"role": "assistant", "content": "Thought: plan\nAction: go to sink 1"},
    ]


def test_proceduralization_injects_paired_script_and_trajectory():
    examples = [
        {
            "task": "put",
            "example": [
                {"role": "user", "content": "example task"},
                {"role": "assistant", "content": "Action: look"},
            ],
        }
    ]
    messages = make_messages(
        "Your task is to: put an apple on a table.",
        "pick_and_place-simple/trial",
        examples,
        [
            {
                "query": "put an apple on a table",
                "workflow": "Find, take, and place the apple.",
                "trajectory": [{"role": "assistant", "content": "Action: take apple 1"}],
            }
        ],
        "proceduralization",
    )
    prompt = messages[-1]["content"]
    assert '"guidelines"' in prompt
    assert '"successful_trajectory"' in prompt
    assert "Find, take, and place the apple." in prompt
    assert "Action: take apple 1" in prompt


def test_parse_indices_is_deduplicated_and_zero_based():
    assert parse_indices(None) is None
    assert parse_indices("7, 2,7") == {2, 7}
