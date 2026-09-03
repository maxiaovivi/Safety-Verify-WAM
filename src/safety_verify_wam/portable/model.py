from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from .contracts import (
    PORTABLE_CLASS_NAMES,
    RISK_CLASS_INDEX,
    RobotProfile,
    SafetyBatch,
)


@dataclass(frozen=True)
class PortableSafetyConfig:
    state_dim: int
    action_dim: int
    model_dim: int = 128
    vision_channels: tuple[int, ...] = (32, 64, 128)
    transformer_layers: int = 3
    attention_heads: int = 4
    feedforward_multiplier: int = 4
    dropout: float = 0.1
    max_views: int = 8
    time_frequencies: int = 8
    motion_features: bool = False

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("state_dim and action_dim must be positive")
        if self.model_dim < 16:
            raise ValueError("model_dim must be at least 16")
        if self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.transformer_layers < 1 or self.attention_heads < 1:
            raise ValueError("transformer_layers and attention_heads must be positive")
        if self.feedforward_multiplier < 1:
            raise ValueError("feedforward_multiplier must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        channels = tuple(int(value) for value in self.vision_channels)
        if not channels or any(value < 4 for value in channels):
            raise ValueError("vision_channels must contain values of at least 4")
        if self.max_views < 1:
            raise ValueError("max_views must be positive")
        if self.time_frequencies < 1:
            raise ValueError("time_frequencies must be positive")
        if not isinstance(self.motion_features, bool):
            raise TypeError("motion_features must be bool")
        object.__setattr__(self, "vision_channels", channels)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["vision_channels"] = list(self.vision_channels)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortableSafetyConfig":
        values = dict(payload)
        if "vision_channels" in values:
            values["vision_channels"] = tuple(values["vision_channels"])
        return cls(**values)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class FrameEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                    nn.GELU(),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(channels[-1], output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.output(self.pool(features).flatten(1))


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, frequencies: int, output_dim: int) -> None:
        super().__init__()
        values = torch.logspace(-1, 2, steps=frequencies)
        self.register_buffer("frequencies", values, persistent=True)
        self.projection = nn.Sequential(
            nn.Linear(1 + 2 * frequencies, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        angles = timestamps.unsqueeze(-1) * self.frequencies.to(
            device=timestamps.device,
            dtype=timestamps.dtype,
        )
        encoded = torch.cat(
            [timestamps.unsqueeze(-1), angles.sin(), angles.cos()], dim=-1
        )
        return self.projection(encoded)


class PortableSafetyCore(nn.Module):
    """Policy-independent video/state/action binary safety classifier.

    The core deliberately has no reference to a policy backbone or its hidden
    width. A 16-step Efficient-WAM candidate and a 32-step Fast-WAM candidate
    use the same weights because action positions are represented by continuous
    timestamps rather than a fixed learned horizon table.
    """

    def __init__(self, config: PortableSafetyConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.frame_encoder = FrameEncoder(config.vision_channels, dim)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.motion_encoder = (
            nn.Sequential(
                nn.LayerNorm(2 * config.action_dim),
                nn.Linear(2 * config.action_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            if config.motion_features
            else None
        )
        self.time_encoder = ContinuousTimeEncoding(config.time_frequencies, dim)
        self.view_embedding = nn.Embedding(config.max_views, dim)
        self.modality_embedding = nn.Parameter(torch.zeros(3, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.attention_heads,
            dim_feedforward=dim * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )
        self.chunk_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, len(PORTABLE_CLASS_NAMES))
        )
        self.step_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, len(PORTABLE_CLASS_NAMES))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.view_embedding.weight, std=0.02)
        for head in (self.chunk_head, self.step_head):
            nn.init.xavier_uniform_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @property
    def class_names(self) -> tuple[str, str]:
        return PORTABLE_CLASS_NAMES

    def _validate_dimensions(self, batch: SafetyBatch) -> None:
        if batch.state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state dim is {batch.state.shape[-1]}, expected "
                f"{self.config.state_dim}"
            )
        if batch.action.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"action dim is {batch.action.shape[-1]}, expected "
                f"{self.config.action_dim}"
            )
        if batch.video.shape[2] > self.config.max_views:
            raise ValueError(
                f"video has {batch.video.shape[2]} views, maximum is "
                f"{self.config.max_views}"
            )
        if self.motion_encoder is not None and batch.state.shape[-1] < self.config.action_dim:
            raise ValueError(
                "motion_features requires state_dim to cover every action dimension"
            )

    def _motion_tokens(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        state_times: torch.Tensor,
        action_times: torch.Tensor,
        batch: SafetyBatch,
    ) -> torch.Tensor:
        if self.motion_encoder is None:
            return action.new_zeros((*action.shape[:2], self.config.model_dim))
        state_indices = batch.state_mask.long().sum(dim=1) - 1
        state_gather = state_indices.view(-1, 1, 1).expand(
            -1, 1, state.shape[-1]
        )
        current = state.gather(1, state_gather).squeeze(1)[
            :, : self.config.action_dim
        ]
        previous_targets = torch.cat(
            [current.unsqueeze(1), action[:, :-1]], dim=1
        )
        state_time = state_times.gather(1, state_indices.unsqueeze(1)).squeeze(1)
        previous_times = torch.cat(
            [state_time.unsqueeze(1), action_times[:, :-1]], dim=1
        )
        step_dt = (action_times - previous_times).clamp_min(1e-4)
        displacement = action - previous_targets
        velocity = displacement / step_dt.unsqueeze(-1)
        motion = torch.cat([displacement, velocity], dim=-1)
        return self.motion_encoder(motion)

    def forward(self, batch: SafetyBatch) -> dict[str, torch.Tensor]:
        self._validate_dimensions(batch)
        parameter = self.cls_token
        if batch.video.device != parameter.device:
            raise RuntimeError(
                f"Safety batch is on {batch.video.device}, model is on "
                f"{parameter.device}"
            )
        dtype = parameter.dtype
        video = batch.video.to(dtype=dtype)
        state = batch.state.to(dtype=dtype)
        action = batch.action.to(dtype=dtype)
        video_times = batch.video_timestamps.to(dtype=dtype)
        state_times = batch.state_timestamps.to(dtype=dtype)
        action_times = batch.action_timestamps.to(dtype=dtype)

        batch_size, frames, views = video.shape[:3]
        flat_video = video.reshape(-1, *video.shape[3:])
        video_tokens = self.frame_encoder(flat_video).reshape(
            batch_size, frames, views, -1
        )
        video_time_tokens = self.time_encoder(video_times).unsqueeze(2)
        view_indices = torch.arange(views, device=video.device)
        video_tokens = (
            video_tokens
            + video_time_tokens
            + self.view_embedding(view_indices).view(1, 1, views, -1)
            + self.modality_embedding[0].view(1, 1, 1, -1)
        ).reshape(batch_size, frames * views, -1)
        state_tokens = (
            self.state_encoder(state)
            + self.time_encoder(state_times)
            + self.modality_embedding[1].view(1, 1, -1)
        )
        action_tokens = (
            self.action_encoder(action)
            + self._motion_tokens(
                state,
                action,
                state_times,
                action_times,
                batch,
            )
            + self.time_encoder(action_times)
            + self.modality_embedding[2].view(1, 1, -1)
        )
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, video_tokens, state_tokens, action_tokens], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=video.device
                ),
                ~batch.video_mask.reshape(batch_size, frames * views),
                ~batch.state_mask,
                ~batch.action_mask,
            ],
            dim=1,
        )
        fused = self.fusion(tokens, src_key_padding_mask=padding_mask)
        action_start = 1 + frames * views + state.shape[1]
        fused_actions = fused[:, action_start : action_start + action.shape[1]]
        class_logits = self.chunk_head(fused[:, 0])
        step_class_logits = self.step_head(fused_actions)
        step_class_logits = step_class_logits.masked_fill(
            ~batch.action_mask.unsqueeze(-1), 0.0
        )
        probabilities = torch.softmax(class_logits.float(), dim=-1)
        step_probabilities = torch.softmax(step_class_logits.float(), dim=-1)
        return {
            "class_logits": class_logits,
            "class_probabilities": probabilities,
            "risk_probability": probabilities[:, RISK_CLASS_INDEX],
            "step_class_logits": step_class_logits,
            "step_class_probabilities": step_probabilities,
            "step_risk_probability": step_probabilities[:, :, RISK_CLASS_INDEX],
            "action_mask": batch.action_mask,
            "risk_features": fused[:, 0],
            "step_risk_features": fused_actions,
        }

    @classmethod
    def for_profile(
        cls,
        profile: RobotProfile,
        **overrides: Any,
    ) -> "PortableSafetyCore":
        return cls(
            PortableSafetyConfig(
                state_dim=profile.state_dim,
                action_dim=profile.action_dim,
                **overrides,
            )
        )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
