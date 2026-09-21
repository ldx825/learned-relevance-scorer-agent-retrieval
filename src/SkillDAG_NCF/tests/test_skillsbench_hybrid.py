import json
from pathlib import Path

import pytest

from benchmarks.skillsbench.skilldag_benchmark import (
    NCF_PLAN_MARKER_START,
    NCF_PLAN_SCHEMA,
    build_skilldag_docker_block,
    inject_skilldag_instruction_protocol,
    load_ncf_plan_manifest,
)


def _manifest():
    return {
        "schema_version": NCF_PLAN_SCHEMA,
        "tasks": {
            "demo-task": {
                "full_task": "Build a deterministic dialogue parser.",
                "views": [
                    {
                        "name": "overall",
                        "query": "parse dialogue deterministically",
                        "matches": [
                            {
                                "skill_id": "dialogue-parser",
                                "description": "Parse dialogue into structured records.",
                            },
                            {
                                "skill_id": "text-normalizer",
                                "description": "Normalize text without losing identifiers.",
                            },
                        ],
                    }
                ]
            }
        },
    }


def test_plan_manifest_keeps_only_prompt_safe_fields(tmp_path):
    path = tmp_path / "plans.json"
    payload = _manifest()
    payload["tasks"]["demo-task"]["views"][0]["matches"][0]["score"] = 99.0
    payload["tasks"]["demo-task"]["views"][0]["matches"][0][
        "description"
    ] = "Use a verifier-aligned parser."
    path.write_text(json.dumps(payload), encoding="utf-8")

    plans = load_ncf_plan_manifest(path)

    assert plans["demo-task"]["views"][0]["matches"][0] == {
        "skill_id": "dialogue-parser",
        "description": "Use a verifier-aligned parser.",
    }


def test_plan_manifest_rejects_evaluation_only_keys(tmp_path):
    path = tmp_path / "plans.json"
    payload = _manifest()
    payload["tasks"]["demo-task"]["gold_skills"] = ["dialogue-parser"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation-only"):
        load_ncf_plan_manifest(path)


def test_plan_manifest_allows_context_only_fallback(tmp_path):
    payload = _manifest()
    payload["tasks"]["demo-task"]["views"] = []

    plans = load_ncf_plan_manifest(
        _write_json(tmp_path / "plans.json", payload)
    )

    assert plans["demo-task"] == {
        "full_task": "Build a deterministic dialogue parser.",
        "views": [],
    }


def test_instruction_injection_adds_proactive_plan_and_keeps_online_search(tmp_path):
    instruction = tmp_path / "instruction.md"
    instruction.write_text("Build a deterministic dialogue parser.\n", encoding="utf-8")
    plan = load_ncf_plan_manifest(
        _write_json(tmp_path / "plans.json", _manifest())
    )["demo-task"]

    inject_skilldag_instruction_protocol(instruction, ncf_plan=plan)
    text = instruction.read_text(encoding="utf-8")

    assert "Build a deterministic dialogue parser." in text
    assert NCF_PLAN_MARKER_START in text
    assert "1. dialogue-parser" in text
    assert "does not replace online retrieval" in text
    assert "MANDATORY first step" in text


def test_docker_block_writes_nonempty_cli_without_heredoc():
    block = build_skilldag_docker_block()

    # Classic Docker builders may accept a heredoc-looking RUN instruction
    # while creating an empty file. Keep the wrapper portable and assert the
    # complete executable payload is part of one printf command.
    assert "<<'EOF'" not in block
    assert "RUN printf '%s\\n'" in block
    assert '#!/usr/bin/env sh' in block
    assert 'exec python3 -m skilldag "$@"' in block
    assert "> /usr/local/bin/skilldag" in block
    assert "chmod +x /usr/local/bin/skilldag" in block


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
