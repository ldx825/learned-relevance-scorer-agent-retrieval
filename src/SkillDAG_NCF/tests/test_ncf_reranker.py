import pytest

torch = pytest.importorskip("torch")

from skilldag.ncf_reranker import NCFConfig, NeuMFReranker, _build_model, load_reranker
from skilldag.portable_ncf import SCHEMA_VERSION


def test_neumf_reranker_loads_training_compatible_checkpoint(tmp_path):
    model = _build_model(NCFConfig(input_dim=4))
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)

    reranker = NeuMFReranker(checkpoint, input_dim=4)
    scores = reranker.score(
        [1.0, 2.0, 3.0, 4.0],
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
    )

    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)


def test_portable_neumf_matches_expected_algebra(tmp_path):
    bundle = tmp_path / "model.json"
    bundle.write_text(
        __import__("json").dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "model_config": {"input_dim": 2},
                "task_projections": {
                    "gmf_weight": [[1.0, 0.0]],
                    "mlp_weight": [[0.0, 1.0]],
                },
                "skill_features": {
                    "skill-a": {"gmf": [2.0], "mlp": [3.0]},
                    "skill-b": {"gmf": [-1.0], "mlp": [1.0]},
                },
                "mlp_layers": [
                    {
                        "weight": [[1.0, 1.0]],
                        "bias": [0.0],
                    }
                ],
                "output": {
                    "weight": [1.0, 1.0],
                    "bias": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    reranker = load_reranker(bundle, input_dim=2)
    assert reranker.score_by_id([2.0, 4.0], ["skill-a", "skill-b"]) == [
        11.0,
        3.0,
    ]
