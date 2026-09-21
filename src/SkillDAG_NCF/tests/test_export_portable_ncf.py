from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from skilldag.ncf_reranker import NeuMFReranker
from skilldag.portable_ncf import PortableNeuMFReranker


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_portable_ncf.py"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_portable_ncf", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exported_bundle_matches_torch_logits(tmp_path: Path):
    torch = pytest.importorskip("torch")
    exporter = _load_exporter()
    from skilldag.ncf_reranker import NCFConfig, _build_model

    torch.manual_seed(7)
    config = NCFConfig(input_dim=4, gmf_latent_dim=2, mlp_layer_sizes=(4, 3, 2))
    model = _build_model(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "input_dim": 4,
                "gmf_latent_dim": 2,
                "mlp_layer_sizes": [4, 3, 2],
            },
        },
        checkpoint,
    )
    embeddings = {
        "alpha": {"model": "test-embedding", "embedding": [1.0, 0.0, 0.5, -0.5]},
        "beta": {"model": "test-embedding", "embedding": [0.0, 1.0, -0.5, 0.5]},
    }
    embedding_path = tmp_path / "graph.embeddings.json"
    embedding_path.write_text(json.dumps(embeddings), encoding="utf-8")
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps({"nodes": {"alpha": {}, "beta": {}}}), encoding="utf-8"
    )
    bundle_path = tmp_path / "bundle.json"

    exporter.export_bundle(
        checkpoint_path=checkpoint,
        embeddings_path=embedding_path,
        graph_path=graph_path,
        output_path=bundle_path,
    )
    task = [0.25, -0.5, 0.75, 1.0]
    expected = NeuMFReranker(checkpoint, input_dim=4).score(
        task, [embeddings["alpha"]["embedding"], embeddings["beta"]["embedding"]]
    )
    actual = PortableNeuMFReranker(bundle_path, input_dim=4).score_by_id(
        task, ["alpha", "beta"]
    )
    assert actual == pytest.approx(expected, abs=1e-6)
