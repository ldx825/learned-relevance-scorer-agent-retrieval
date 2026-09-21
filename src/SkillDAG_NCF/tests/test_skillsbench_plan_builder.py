from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_skillsbench_ncf_plans.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_skillsbench_ncf_plans", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_view_builder_uses_full_task_and_explicit_steps():
    builder = _load_builder()
    views = builder.build_task_views(
        """# Task
Create a report and upload it.

1. Parse the source records into JSON.
2. Validate every required field.
3. Upload the resulting artifact.
"""
    )
    assert [view["name"] for view in views] == [
        "full_task",
        "stage_1",
        "stage_2",
        "stage_3",
    ]
    assert "Create a report" in views[0]["query"]
    assert views[1]["query"] == "Parse the source records into JSON."


def test_view_builder_removes_previously_injected_protocol():
    builder = _load_builder()
    views = builder.build_task_views(
        """Do the visible task.
<!-- BEGIN SKILLDAG ONLINE PROTOCOL -->
Ignore this generated retrieval text.
<!-- END SKILLDAG ONLINE PROTOCOL -->
"""
    )
    assert len(views) == 1
    assert views[0]["query"] == "Do the visible task."


def test_skill_vector_loader_accepts_wrapped_enriched_cache(tmp_path: Path):
    builder = _load_builder()
    cache = tmp_path / "embeddings.json"
    cache.write_text(
        json.dumps(
            {
                "model": "test-embedding",
                "items": {
                    "alpha": {
                        "model": "test-embedding",
                        "embedding": [1.0, 0.5],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    vectors, model = builder._load_skill_vectors(cache)

    assert vectors == {"alpha": [1.0, 0.5]}
    assert model == "test-embedding"
