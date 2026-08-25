from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn


SAFETY_CLASS_NAMES = ("safe", "boundary", "risk")


@dataclass(frozen=True)
class RiskHeadConfig:
    video_dim: int
    action_dim: int
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    max_action_steps: int = 16
    num_risk_types: int = 0
    class_names: tuple[str, str, str] = SAFETY_CLASS_NAMES

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.video_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Feature dimensions must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.max_action_steps < 1:
            raise ValueError("max_action_steps must be at least 1")
        if tuple(self.class_names) != SAFETY_CLASS_NAMES:
            raise ValueError(
                f"Safety classes must use the fixed order {SAFETY_CLASS_NAMES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ActionQueriesToVideoBlock(nn.Module):
    """Let state/action queries read Efficient-WAM condition and future tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.video_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        normalized_video = self.video_norm(video)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_video,
            normalized_video,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class SafetyRiskHead(nn.Module):
    """Classify a candidate action by querying Efficient-WAM visual dynamics.

    Efficient-WAM first produces condition/future video tokens and aligned
    state/action tokens. This head keeps the spatial video tokens intact,
    treats the state and candidate-action features as queries, and predicts the
    mutually exclusive classes ``safe``, ``boundary``, and ``risk``.
    """

    def __init__(self, config: RiskHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.video_norm = nn.LayerNorm(config.video_dim)
        self.state_norm = nn.LayerNorm(config.action_dim)
        self.action_norm = nn.LayerNorm(config.action_dim)
        self.video_projection = nn.Linear(config.video_dim, config.hidden_dim)
        self.state_projection = nn.Linear(config.action_dim, config.hidden_dim)
        self.action_projection = nn.Linear(config.action_dim, config.hidden_dim)

        # condition, future, state, and action token identities
        self.token_types = nn.Embedding(4, config.hidden_dim)
        self.action_positions = nn.Parameter(
            torch.empty(1, config.max_action_steps, config.hidden_dim)
        )
        self.query_blocks = nn.ModuleList(
            [
                _ActionQueriesToVideoBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.chunk_query = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        self.chunk_query_norm = nn.LayerNorm(config.hidden_dim)
        self.chunk_memory_norm = nn.LayerNorm(config.hidden_dim)
        self.chunk_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.chunk_output_norm = nn.LayerNorm(config.hidden_dim)
        self.step_output_norm = nn.LayerNorm(config.hidden_dim)
        self.chunk_head = nn.Linear(config.hidden_dim, len(config.class_names))
        self.step_head = nn.Linear(config.hidden_dim, len(config.class_names))
        self.type_head = (
            nn.Linear(config.hidden_dim, config.num_risk_types)
            if config.num_risk_types > 0
            else None
        )
        self.register_buffer(
            "severity_values",
            torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.action_positions, std=0.02)
        nn.init.normal_(self.chunk_query, std=0.02)
        nn.init.xavier_uniform_(self.chunk_head.weight)
        nn.init.zeros_(self.chunk_head.bias)
        nn.init.xavier_uniform_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias)
        if self.type_head is not None:
            nn.init.xavier_uniform_(self.type_head.weight)
            nn.init.zeros_(self.type_head.bias)

    @staticmethod
    def _validate_feature(
        name: str,
        value: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> None:
        if value.ndim != 3:
            raise ValueError(f"{name} must be [B,T,D], got {tuple(value.shape)}")
        if batch_size is not None and value.shape[0] != batch_size:
            raise ValueError(f"{name} batch size differs from the other features")

    def forward(
        self,
        condition_features: torch.Tensor,
        future_features: torch.Tensor,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_feature("condition_features", condition_features)
        batch = int(condition_features.shape[0])
        self._validate_feature("future_features", future_features, batch_size=batch)
        self._validate_feature("state_features", state_features, batch_size=batch)
        self._validate_feature("action_features", action_features, batch_size=batch)
        if state_features.shape[1] != 1:
            raise ValueError("state_features must contain exactly one current-state token")
        action_steps = int(action_features.shape[1])
        if action_steps > self.config.max_action_steps:
            raise ValueError(
                f"Action feature length {action_steps} exceeds "
                f"{self.config.max_action_steps}"
            )

        device = condition_features.device
        video_dtype = self.video_projection.weight.dtype
        action_dtype = self.action_projection.weight.dtype
        condition = self.video_projection(
            self.video_norm(condition_features.to(dtype=video_dtype))
        )
        future = self.video_projection(
            self.video_norm(future_features.to(dtype=video_dtype))
        )
        condition = condition + self.token_types(
            torch.tensor(0, device=device)
        ).view(1, 1, -1)
        future = future + self.token_types(torch.tensor(1, device=device)).view(
            1, 1, -1
        )
        video = torch.cat([condition, future], dim=1)

        state = self.state_projection(
            self.state_norm(state_features.to(dtype=action_dtype))
        )
        action = self.action_projection(
            self.action_norm(action_features.to(dtype=action_dtype))
        )
        state = state + self.token_types(torch.tensor(2, device=device)).view(
            1, 1, -1
        )
        action = (
            action
            + self.action_positions[:, :action_steps]
            + self.token_types(torch.tensor(3, device=device)).view(1, 1, -1)
        )
        queries = torch.cat([state, action], dim=1)
        for block in self.query_blocks:
            queries = block(queries, video)

        chunk_query = self.chunk_query.expand(batch, -1, -1)
        chunk_memory = self.chunk_memory_norm(queries)
        pooled, _ = self.chunk_attention(
            self.chunk_query_norm(chunk_query),
            chunk_memory,
            chunk_memory,
            need_weights=False,
        )
        chunk_feature = self.chunk_output_norm(chunk_query + pooled)[:, 0]
        action_consequences = self.step_output_norm(queries[:, 1:])
        outputs = {
            "class_logits": self.chunk_head(chunk_feature),
            "step_class_logits": self.step_head(action_consequences),
        }
        if self.type_head is not None:
            outputs["risk_type_logits"] = self.type_head(chunk_feature)
        return outputs


class SafetyVerifyWAM(nn.Module):
    """Three-view observation + state + candidate action + text -> safety class."""

    def __init__(self, backbone: nn.Module, risk_head: SafetyRiskHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.risk_head = risk_head

    @property
    def class_names(self) -> tuple[str, str, str]:
        return tuple(self.risk_head.config.class_names)

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        text_embeddings: torch.Tensor | Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image [B,3,H,W], got {tuple(image.shape)}")
        if state.ndim != 2:
            raise ValueError(f"Expected state [B,S], got {tuple(state.shape)}")
        if action.ndim != 3:
            raise ValueError(f"Expected action [B,T,A], got {tuple(action.shape)}")
        if not (image.shape[0] == state.shape[0] == action.shape[0]):
            raise ValueError("Image, state, and action batch sizes differ")
        features = self.backbone.imagine(
            image,
            state,
            action,
            text_embeddings,
        )
        outputs = self.risk_head(
            features["condition_features"],
            features["future_features"],
            features["state_features"],
            features["action_features"],
        )
        probabilities = torch.softmax(outputs["class_logits"].float(), dim=-1)
        outputs["class_probabilities"] = probabilities
        outputs["predicted_class"] = probabilities.argmax(dim=-1)
        outputs["severity_score"] = (
            probabilities
            * self.risk_head.severity_values.to(
                device=probabilities.device, dtype=probabilities.dtype
            )
        ).sum(dim=-1)
        return outputs

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        text_embeddings: torch.Tensor | Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        outputs = self(image, state, action, text_embeddings)
        outputs["requires_intervention"] = outputs["predicted_class"] != 0
        if was_training:
            self.train()
        return outputs


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    state = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state.items()
        if name in trainable_names
    }


def load_delta_state_dict(model: nn.Module, delta: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    unknown = sorted(set(delta) - set(model_state))
    if unknown:
        raise RuntimeError(f"Safety checkpoint has unknown tensors: {unknown[:10]}")
    mismatched = [
        name
        for name, tensor in delta.items()
        if tuple(tensor.shape) != tuple(model_state[name].shape)
    ]
    if mismatched:
        raise RuntimeError(
            f"Safety checkpoint tensor shapes do not match: {mismatched[:10]}"
        )
    model.load_state_dict(delta, strict=False)
