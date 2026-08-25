from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn


SAFETY_CLASS_NAMES = ("safe", "boundary", "risk")


@dataclass(frozen=True)
class RiskHeadConfig:
    action_dim: int
    rank: int = 16
    alpha: float = 16.0
    num_taps: int = 2
    dropout: float = 0.1
    max_action_steps: int = 16
    num_risk_types: int = 0
    class_names: tuple[str, str, str] = SAFETY_CLASS_NAMES

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.rank < 1 or self.rank > self.action_dim:
            raise ValueError("rank must be in [1, action_dim]")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.num_taps < 1:
            raise ValueError("num_taps must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_action_steps < 1:
            raise ValueError("max_action_steps must be at least 1")
        if tuple(self.class_names) != SAFETY_CLASS_NAMES:
            raise ValueError(
                f"Safety classes must use the fixed order {SAFETY_CLASS_NAMES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _LowRankResidualAdapter(nn.Module):
    """Read a frozen ActionExpert residual stream through a low-rank delta."""

    def __init__(
        self,
        dim: int,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / float(rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        # A zero output projection makes the initial adapter an identity map.
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.down(self.norm(tokens))) * self.scale
        return tokens + self.dropout(delta)


class SafetyRiskHead(nn.Module):
    """Decode risk from frozen, future-conditioned ActionExpert tokens.

    Efficient-WAM has already let action queries attend video keys and values
    in every joint block. The head therefore reads late state/action/register
    residual streams directly. Each tapped layer receives only a low-rank
    residual delta; the original Efficient-WAM parameters and imagined future
    remain unchanged.
    """

    def __init__(self, config: RiskHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.tap_adapters = nn.ModuleList(
            [
                _LowRankResidualAdapter(
                    config.action_dim,
                    config.rank,
                    config.alpha,
                    config.dropout,
                )
                for _ in range(config.num_taps)
            ]
        )
        self.tap_logits = nn.Parameter(torch.zeros(config.num_taps))
        self.pool_norm = nn.LayerNorm(config.action_dim)
        self.pool_score = nn.Linear(config.action_dim, 1, bias=False)
        self.chunk_output_norm = nn.LayerNorm(config.action_dim)
        self.step_output_norm = nn.LayerNorm(config.action_dim)
        self.chunk_head = nn.Linear(config.action_dim, len(config.class_names))
        self.step_head = nn.Linear(config.action_dim, len(config.class_names))
        self.type_head = (
            nn.Linear(config.action_dim, config.num_risk_types)
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
        nn.init.zeros_(self.tap_logits)
        nn.init.xavier_uniform_(self.pool_score.weight)
        nn.init.xavier_uniform_(self.chunk_head.weight)
        nn.init.zeros_(self.chunk_head.bias)
        nn.init.xavier_uniform_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias)
        if self.type_head is not None:
            nn.init.xavier_uniform_(self.type_head.weight)
            nn.init.zeros_(self.type_head.bias)

    def _validate_taps(
        self,
        name: str,
        value: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> None:
        if value.ndim != 4:
            raise ValueError(f"{name} must be [B,K,T,D], got {tuple(value.shape)}")
        if value.shape[1] != self.config.num_taps:
            raise ValueError(
                f"{name} has {value.shape[1]} taps, expected {self.config.num_taps}"
            )
        if value.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"{name} feature dim is {value.shape[-1]}, "
                f"expected {self.config.action_dim}"
            )
        if batch_size is not None and value.shape[0] != batch_size:
            raise ValueError(f"{name} batch size differs from the other features")

    def forward(
        self,
        state_feature_taps: torch.Tensor,
        action_feature_taps: torch.Tensor,
        register_feature_taps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_taps("state_feature_taps", state_feature_taps)
        batch = int(state_feature_taps.shape[0])
        self._validate_taps(
            "action_feature_taps", action_feature_taps, batch_size=batch
        )
        self._validate_taps(
            "register_feature_taps", register_feature_taps, batch_size=batch
        )
        if state_feature_taps.shape[2] != 1:
            raise ValueError("state_feature_taps must contain one state token per tap")
        action_steps = int(action_feature_taps.shape[2])
        if action_steps > self.config.max_action_steps:
            raise ValueError(
                f"Action feature length {action_steps} exceeds "
                f"{self.config.max_action_steps}"
            )

        adapter_dtype = self.tap_adapters[0].down.weight.dtype
        joint_taps = torch.cat(
            [state_feature_taps, action_feature_taps, register_feature_taps],
            dim=2,
        ).to(dtype=adapter_dtype)
        adapted_taps = torch.stack(
            [
                adapter(joint_taps[:, tap_index])
                for tap_index, adapter in enumerate(self.tap_adapters)
            ],
            dim=1,
        )
        tap_weights = torch.softmax(self.tap_logits.float(), dim=0).to(
            dtype=adapted_taps.dtype
        )
        joint_tokens = torch.sum(
            adapted_taps * tap_weights.view(1, -1, 1, 1),
            dim=1,
        )

        action_tokens = joint_tokens[:, 1 : 1 + action_steps]
        pool_logits = self.pool_score(self.pool_norm(joint_tokens)).squeeze(-1)
        pool_weights = torch.softmax(pool_logits.float(), dim=1).to(
            dtype=joint_tokens.dtype
        )
        chunk_feature = torch.sum(joint_tokens * pool_weights.unsqueeze(-1), dim=1)
        chunk_feature = self.chunk_output_norm(chunk_feature)
        step_features = self.step_output_norm(action_tokens)
        outputs = {
            "class_logits": self.chunk_head(chunk_feature),
            "step_class_logits": self.step_head(step_features),
            "risk_features": chunk_feature,
            "step_risk_features": step_features,
            "safety_pool_weights": pool_weights,
            "safety_tap_weights": tap_weights,
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
            features["state_feature_taps"],
            features["action_feature_taps"],
            features["register_feature_taps"],
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
