import json

import numpy as np

from gos.ncf.retrieval37 import ALFWorld37Retriever


def test_retriever_keeps_raw_embeddings_for_neumf(tmp_path):
    skill_ids = [f"alfworld-skill-{index:02d}" for index in range(37)]
    graph = {
        "nodes": {skill_id: {} for skill_id in skill_ids},
        "edges": [],
    }
    embeddings = {
        skill_id: {"embedding": [float(index + 1), 2.0]}
        for index, skill_id in enumerate(skill_ids)
    }
    graph_path = tmp_path / "graph.json"
    embeddings_path = tmp_path / "embeddings.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    embeddings_path.write_text(json.dumps(embeddings), encoding="utf-8")

    retriever = ALFWorld37Retriever(
        graph_path=graph_path,
        embeddings_path=embeddings_path,
    )

    assert np.allclose(retriever.skill_embeddings[0], [1.0, 2.0])
    assert np.isclose(
        np.linalg.norm(retriever.skill_embeddings_normalized[0]),
        1.0,
    )
    assert not np.allclose(
        retriever.skill_embeddings[0],
        retriever.skill_embeddings_normalized[0],
    )
