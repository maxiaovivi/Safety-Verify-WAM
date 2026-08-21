from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RiskHeadConfig:
    video_dim: int
    action_dim: int
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    max_future_steps: int = 8
    max_action_steps: int = 16
    num_risk_types: int = 0
    unsafe_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 < self.unsafe_threshold < 1.0:
            raise ValueError("unsafe_threshold must be in (0, 1)")
        if self.video_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Feature dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetyRiskHead(nn.Module):
    """Fuse imagined future tokens and aligned action tokens into risk logits."""

    def __init__(self, config: RiskHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.video_norm = nn.LayerNorm(config.video_dim)
        self.action_norm = nn.LayerNorm(config.action_dim)
        self.video_projection = nn.Linear(config.video_dim, config.hidden_dim)
        self.action_projection = nn.Linear(config.action_dim, config.hidden_dim)
        self.risk_token = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        self.future_positions = nn.Parameter(
            torch.empty(1, config.max_future_steps, config.hidden_dim)
        )
        self.action_positions = nn.Parameter(
            torch.empty(1, config.max_action_steps, config.hidden_dim)
        )
        self.token_types = nn.Embedding(3, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.chunk_head = nn.Linear(config.hidden_dim, 1)
        self.step_head = nn.Linear(config.hidden_dim, 1)
        self.type_head = (
            nn.Linear(config.hidden_dim, config.num_risk_types)
            if config.num_risk_types > 0
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.risk_token, std=0.02)
        nn.init.normal_(self.future_positions, std=0.02)
        nn.init.normal_(self.action_positions, std=0.02)
        nn.init.xavier_uniform_(self.chunk_head.weight)
        nn.init.zeros_(self.chunk_head.bias)
        nn.init.xavier_uniform_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias)

    def forward(
        self,
        future_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if future_features.ndim != 3 or action_features.ndim != 3:
            raise ValueError("Expected future/action features shaped [B, T, D]")
        if future_features.shape[0] != action_features.shape[0]:
            raise ValueError("Future and action batch sizes differ")
        future_steps = int(future_features.shape[1])
        action_steps = int(action_features.shape[1])
        if future_steps > self.config.max_future_steps:
            raise ValueError(
                f"Future feature length {future_steps} exceeds {self.config.max_future_steps}"
            )
        if action_steps > self.config.max_action_steps:
            raise ValueError(
                f"Action feature length {action_steps} exceeds {self.config.max_action_steps}"
            )

        batch = future_features.shape[0]
        device = future_features.device
        risk = self.risk_token.expand(batch, -1, -1)
        head_dtype = self.video_projection.weight.dtype
        future = self.video_projection(
            self.video_norm(future_features.to(dtype=head_dtype))
        )
        action = self.action_projection(
            self.action_norm(action_features.to(dtype=self.action_projection.weight.dtype))
        )
        future = future + self.future_positions[:, :future_steps]
        action = action + self.action_positions[:, :action_steps]
        risk = risk + self.token_types(torch.zeros(1, dtype=torch.long, device=device)).view(1, 1, -1)
        future = future + self.token_types(torch.ones(1, dtype=torch.long, device=device)).view(1, 1, -1)
        action = action + self.token_types(torch.full((1,), 2, dtype=torch.long, device=device)).view(1, 1, -1)

        fused = self.output_norm(self.fusion(torch.cat([risk, future, action], dim=1)))
        risk_feature = fused[:, 0]
        action_feature = fused[:, 1 + future_steps :]
        outputs = {
            "unsafe_logit": self.chunk_head(risk_feature),
            "step_logits": self.step_head(action_feature).squeeze(-1),
        }
        if self.type_head is not None:
            outputs["risk_type_logits"] = self.type_head(risk_feature)
        return outputs


class SafetyVerifyWAM(nn.Module):
    """Network contract: current RGB image + action chunk -> safety decision."""

    def __init__(self, backbone: nn.Module, risk_head: SafetyRiskHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.risk_head = risk_head

    @property
    def unsafe_threshold(self) -> float:
        return float(self.risk_head.config.unsafe_threshold)

    def forward(self, image: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image [B, 3, H, W], got {tuple(image.shape)}")
        if action.ndim != 3:
            raise ValueError(f"Expected action [B, T, A], got {tuple(action.shape)}")
        if image.shape[0] != action.shape[0]:
            raise ValueError("Image and action batch sizes differ")
        features = self.backbone.imagine(image, action)
        outputs = self.risk_head(features["future_features"], features["action_features"])
        outputs["unsafe_probability"] = torch.sigmoid(outputs["unsafe_logit"])
        return outputs

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        outputs = self(image, action)
        probability = outputs["unsafe_probability"]
        outputs["is_unsafe"] = probability >= self.unsafe_threshold
        if was_training:
            self.train()
        return outputs


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
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
        raise RuntimeError(f"Safety checkpoint tensor shapes do not match: {mismatched[:10]}")
    model.load_state_dict(delta, strict=False)
