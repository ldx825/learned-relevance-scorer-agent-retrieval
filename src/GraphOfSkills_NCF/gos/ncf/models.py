"""GMF, MLP, and NeuMF adapted from He et al. (WWW 2017).

The paper uses one-hot user/item IDs followed by embedding lookup tables.  Our
task-skill setting replaces only that lookup operation with independent linear
projections of frozen text embeddings.  The interaction functions remain the
paper's GMF, MLP, and NeuMF architectures.  This content-based input is what
allows the scorer to process a task or skill that was not present in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


ModelName = Literal["gmf", "mlp", "neumf"]


@dataclass(frozen=True)
class NCFConfig:
    """Shape and regularization settings shared by all NCF variants.

    ``mlp_layer_sizes`` follows the convention used in the original NCF code:
    its first value is the width after concatenating task and skill latent
    vectors, and every later value is a hidden-layer width.  Consequently, the
    task and skill MLP projections each have width ``mlp_layer_sizes[0] / 2``.
    A tower such as ``(128, 64, 32, 16)`` halves its width at each layer.
    """

    input_dim: int = 3072
    gmf_latent_dim: int = 16
    mlp_layer_sizes: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.0
    init_std: float = 0.01

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.gmf_latent_dim < 1:
            raise ValueError("gmf_latent_dim must be positive")
        if len(self.mlp_layer_sizes) < 2:
            raise ValueError("mlp_layer_sizes must contain an input and hidden width")
        if any(width < 1 for width in self.mlp_layer_sizes):
            raise ValueError("all MLP layer sizes must be positive")
        if self.mlp_layer_sizes[0] % 2:
            raise ValueError("the first MLP layer size must be even")
        if any(
            right > left
            for left, right in zip(self.mlp_layer_sizes, self.mlp_layer_sizes[1:])
        ):
            raise ValueError("mlp_layer_sizes must form a non-increasing tower")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_std <= 0:
            raise ValueError("init_std must be positive")

    @property
    def mlp_projection_dim(self) -> int:
        return self.mlp_layer_sizes[0] // 2

    @property
    def mlp_output_dim(self) -> int:
        return self.mlp_layer_sizes[-1]


class _ContentNCFBase(nn.Module):
    config: NCFConfig

    def _validate_inputs(self, task_embedding: Tensor, skill_embedding: Tensor) -> None:
        if task_embedding.shape != skill_embedding.shape:
            raise ValueError(
                "task_embedding and skill_embedding must have identical shapes; "
                f"got {tuple(task_embedding.shape)} and {tuple(skill_embedding.shape)}"
            )
        if task_embedding.ndim < 1 or task_embedding.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected embeddings with final dimension {self.config.input_dim}; "
                f"got {tuple(task_embedding.shape)}"
            )

    def _reset_linear_parameters(self) -> None:
        # The paper initializes parameters from N(0, 0.01^2).
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def predict_proba_from_logits(logits: Tensor) -> Tensor:
        return torch.sigmoid(logits)


def _make_mlp_tower(config: NCFConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_width, output_width in zip(
        config.mlp_layer_sizes, config.mlp_layer_sizes[1:]
    ):
        layers.append(nn.Linear(input_width, output_width))
        layers.append(nn.ReLU())
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
    return nn.Sequential(*layers)


class GMF(_ContentNCFBase):
    """Generalized Matrix Factorization, paper equations (8) and (9)."""

    def __init__(self, config: NCFConfig | None = None) -> None:
        super().__init__()
        self.config = config or NCFConfig()
        self.task_projection = nn.Linear(
            self.config.input_dim, self.config.gmf_latent_dim, bias=False
        )
        self.skill_projection = nn.Linear(
            self.config.input_dim, self.config.gmf_latent_dim, bias=False
        )
        self.output = nn.Linear(self.config.gmf_latent_dim, 1)
        self._reset_linear_parameters()

    def interaction(self, task_embedding: Tensor, skill_embedding: Tensor) -> Tensor:
        self._validate_inputs(task_embedding, skill_embedding)
        task_latent = self.task_projection(task_embedding)
        skill_latent = self.skill_projection(skill_embedding)
        return task_latent * skill_latent

    def forward(self, task_embedding: Tensor, skill_embedding: Tensor) -> Tensor:
        return self.output(self.interaction(task_embedding, skill_embedding)).squeeze(-1)

    def predict_proba(self, task_embedding: Tensor, skill_embedding: Tensor) -> Tensor:
        return self.predict_proba_from_logits(self(task_embedding, skill_embedding))


class MLP(_ContentNCFBase):
    """The non-linear NCF MLP branch, paper equation (10)."""

    def __init__(self, config: NCFConfig | None = None) -> None:
        super().__init__()
        self.config = config or NCFConfig()
        projection_dim = self.config.mlp_projection_dim
        self.task_projection = nn.Linear(
            self.config.input_dim, projection_dim, bias=False
        )
        self.skill_projection = nn.Linear(
            self.config.input_dim, projection_dim, bias=False
        )
        self.tower = _make_mlp_tower(self.config)
        self.output = nn.Linear(self.config.mlp_output_dim, 1)
        self._reset_linear_parameters()

    def interaction(self, task_embedding: Tensor, skill_embedding: Tensor) -> Tensor:
        self._validate_inputs(task_embedding, skill_embedding)
        task_latent = self.task_projection(task_embedding)
        skill_latent = self.skill_projection(skill_embedding)
        return self.tower(torch.cat((task_latent, skill_latent), dim=-1))

    def forward(self, task_embedding: Tensor, skill_embedding: Tensor) -> Tensor:
        return self.output(self.interaction(task_embedding, skill_embedding)).squeeze(-1)

    def predict_proba(self, task_embedding: Tensor, skill_embedding: Tensor) -> Tensor:
        return self.predict_proba_from_logits(self(task_embedding, skill_embedding))


class NeuMF(_ContentNCFBase):
    """Neural Matrix Factorization with separate GMF and MLP embeddings.

    This is equation (12), not the weight-sharing shortcut in equation (11).
    The two branches have independent task and skill projections and are fused
    only immediately before the final prediction layer.
    """

    def __init__(self, config: NCFConfig | None = None) -> None:
        super().__init__()
        self.config = config or NCFConfig()
        projection_dim = self.config.mlp_projection_dim

        self.gmf_task_projection = nn.Linear(
            self.config.input_dim, self.config.gmf_latent_dim, bias=False
        )
        self.gmf_skill_projection = nn.Linear(
            self.config.input_dim, self.config.gmf_latent_dim, bias=False
        )
        self.mlp_task_projection = nn.Linear(
            self.config.input_dim, projection_dim, bias=False
        )
        self.mlp_skill_projection = nn.Linear(
            self.config.input_dim, projection_dim, bias=False
        )
        self.mlp_tower = _make_mlp_tower(self.config)
        self.output = nn.Linear(
            self.config.gmf_latent_dim + self.config.mlp_output_dim, 1
        )
        self._reset_linear_parameters()

    def interaction_components(
        self, task_embedding: Tensor, skill_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        self._validate_inputs(task_embedding, skill_embedding)
        gmf_vector = self.gmf_task_projection(task_embedding) * self.gmf_skill_projection(
            skill_embedding
        )
        mlp_input = torch.cat(
            (
                self.mlp_task_projection(task_embedding),
                self.mlp_skill_projection(skill_embedding),
            ),
            dim=-1,
        )
        return gmf_vector, self.mlp_tower(mlp_input)

    def interaction(self, task_embedding: Tensor, skill_embedding: Tensor) -> Tensor:
        gmf_vector, mlp_vector = self.interaction_components(
            task_embedding, skill_embedding
        )
        return torch.cat((gmf_vector, mlp_vector), dim=-1)

    def forward(self, task_embedding: Tensor, skill_embedding: Tensor) -> Tensor:
        return self.output(self.interaction(task_embedding, skill_embedding)).squeeze(-1)

    def predict_proba(self, task_embedding: Tensor, skill_embedding: Tensor) -> Tensor:
        return self.predict_proba_from_logits(self(task_embedding, skill_embedding))

    def load_pretrained(self, gmf: GMF, mlp: MLP, *, alpha: float = 0.5) -> None:
        """Initialize NeuMF from separately trained GMF and MLP models.

        The output weights follow paper equation (13): ``alpha * h_GMF`` and
        ``(1-alpha) * h_MLP``.  Biases are combined with the same convex
        weighting so that the initialized NeuMF logit exactly equals the
        weighted sum of the two pretrained logits when dropout is disabled.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if gmf.config != self.config or mlp.config != self.config:
            raise ValueError("GMF, MLP, and NeuMF must use identical NCFConfig values")

        self.gmf_task_projection.load_state_dict(gmf.task_projection.state_dict())
        self.gmf_skill_projection.load_state_dict(gmf.skill_projection.state_dict())
        self.mlp_task_projection.load_state_dict(mlp.task_projection.state_dict())
        self.mlp_skill_projection.load_state_dict(mlp.skill_projection.state_dict())
        self.mlp_tower.load_state_dict(mlp.tower.state_dict())
        with torch.no_grad():
            self.output.weight[:, : self.config.gmf_latent_dim].copy_(
                alpha * gmf.output.weight
            )
            self.output.weight[:, self.config.gmf_latent_dim :].copy_(
                (1.0 - alpha) * mlp.output.weight
            )
            self.output.bias.copy_(
                alpha * gmf.output.bias + (1.0 - alpha) * mlp.output.bias
            )


def build_ncf_model(name: ModelName, config: NCFConfig | None = None) -> nn.Module:
    """Construct one of the three NCF variants using a stable public API."""
    models: dict[str, type[GMF] | type[MLP] | type[NeuMF]] = {
        "gmf": GMF,
        "mlp": MLP,
        "neumf": NeuMF,
    }
    try:
        model_class = models[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown NCF model {name!r}; expected one of {sorted(models)}") from exc
    return model_class(config)
