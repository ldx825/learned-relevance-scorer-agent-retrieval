"""Content-based NeuMF inference for the optional SkillDAG search reranker.

The checkpoint is produced by ``GraphOfSkills_NCF``.  This small inference-only
copy keeps ``SkillDAG_NCF`` runnable as an independent project while preserving
the exact trained network shapes and raw-embedding input convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NCFConfig:
    input_dim: int = 3072
    gmf_latent_dim: int = 16
    mlp_layer_sizes: tuple[int, ...] = (64, 32, 16)

    @property
    def mlp_projection_dim(self) -> int:
        return self.mlp_layer_sizes[0] // 2


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "SkillDAG NCF mode requires PyTorch; install the project with "
            "`pip install -e '.[ncf]'`."
        ) from exc
    return torch


def _build_model(config: NCFConfig, model_type: str = "neumf"):
    torch = _torch()
    nn = torch.nn

    class GMF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.task_projection = nn.Linear(
                config.input_dim, config.gmf_latent_dim, bias=False
            )
            self.skill_projection = nn.Linear(
                config.input_dim, config.gmf_latent_dim, bias=False
            )
            self.output = nn.Linear(config.gmf_latent_dim, 1)

        def forward(self, task_embedding, skill_embedding):
            interaction = self.task_projection(task_embedding)
            interaction = interaction * self.skill_projection(skill_embedding)
            return self.output(interaction).squeeze(-1)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.task_projection = nn.Linear(
                config.input_dim, config.mlp_projection_dim, bias=False
            )
            self.skill_projection = nn.Linear(
                config.input_dim, config.mlp_projection_dim, bias=False
            )
            layers = []
            for left, right in zip(
                config.mlp_layer_sizes, config.mlp_layer_sizes[1:]
            ):
                layers.extend((nn.Linear(left, right), nn.ReLU()))
            self.tower = nn.Sequential(*layers)
            self.output = nn.Linear(config.mlp_layer_sizes[-1], 1)

        def forward(self, task_embedding, skill_embedding):
            interaction = self.tower(
                torch.cat(
                    (
                        self.task_projection(task_embedding),
                        self.skill_projection(skill_embedding),
                    ),
                    dim=-1,
                )
            )
            return self.output(interaction).squeeze(-1)

    class NeuMF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gmf_task_projection = nn.Linear(
                config.input_dim, config.gmf_latent_dim, bias=False
            )
            self.gmf_skill_projection = nn.Linear(
                config.input_dim, config.gmf_latent_dim, bias=False
            )
            self.mlp_task_projection = nn.Linear(
                config.input_dim, config.mlp_projection_dim, bias=False
            )
            self.mlp_skill_projection = nn.Linear(
                config.input_dim, config.mlp_projection_dim, bias=False
            )
            layers = []
            for left, right in zip(
                config.mlp_layer_sizes, config.mlp_layer_sizes[1:]
            ):
                layers.extend((nn.Linear(left, right), nn.ReLU()))
            self.mlp_tower = nn.Sequential(*layers)
            self.output = nn.Linear(
                config.gmf_latent_dim + config.mlp_layer_sizes[-1], 1
            )

        def forward(self, task_embedding, skill_embedding):
            gmf = self.gmf_task_projection(task_embedding)
            gmf = gmf * self.gmf_skill_projection(skill_embedding)
            mlp = self.mlp_tower(
                torch.cat(
                    (
                        self.mlp_task_projection(task_embedding),
                        self.mlp_skill_projection(skill_embedding),
                    ),
                    dim=-1,
                )
            )
            return self.output(torch.cat((gmf, mlp), dim=-1)).squeeze(-1)

    classes = {"gmf": GMF, "mlp": MLP, "neumf": NeuMF}
    if model_type not in classes:
        raise ValueError(f"unsupported NCF checkpoint type: {model_type}")
    return classes[model_type]()


class NeuMFReranker:
    """Load one graded NeuMF checkpoint and score raw task/skill embeddings."""

    def __init__(self, model_path: str | Path, *, input_dim: int = 3072) -> None:
        torch = _torch()
        self.torch = torch
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"NeuMF checkpoint not found: {self.model_path}")
        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            raw_config = state.get("model_config", {})
            config = NCFConfig(
                input_dim=int(raw_config.get("input_dim", input_dim)),
                gmf_latent_dim=int(raw_config.get("gmf_latent_dim", 16)),
                mlp_layer_sizes=tuple(raw_config.get("mlp_layer_sizes", (64, 32, 16))),
            )
            state = state["state_dict"]
        else:
            config = NCFConfig(input_dim=input_dim)
        if config.input_dim != input_dim:
            raise ValueError(
                f"checkpoint expects embedding dimension {config.input_dim}, got {input_dim}"
            )
        keys = set(state)
        if "gmf_task_projection.weight" in keys:
            model_type = "neumf"
        elif "tower.0.weight" in keys:
            model_type = "mlp"
        elif "task_projection.weight" in keys:
            model_type = "gmf"
        else:
            raise ValueError(
                f"cannot infer NCF checkpoint type from {self.model_path}"
            )
        self.model_type = model_type
        self.model = _build_model(config, model_type)
        self.model.load_state_dict(state)
        self.model.eval()

    def score(
        self,
        task_embedding: list[float],
        skill_embeddings: list[list[float]],
    ) -> list[float]:
        if not skill_embeddings:
            return []
        torch = self.torch
        task = torch.tensor(task_embedding, dtype=torch.float32)
        task = task.unsqueeze(0).repeat(len(skill_embeddings), 1)
        skills = torch.tensor(skill_embeddings, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(task, skills)
        return [float(value) for value in logits.cpu().tolist()]


def load_reranker(model_path: str | Path, *, input_dim: int):
    """Load a host PyTorch checkpoint or a dependency-free JSON bundle."""
    path = Path(model_path).expanduser().resolve()
    if path.suffix.lower() == ".json":
        from .portable_ncf import PortableNeuMFReranker

        return PortableNeuMFReranker(path, input_dim=input_dim)
    return NeuMFReranker(path, input_dim=input_dim)
