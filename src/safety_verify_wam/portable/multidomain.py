from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import (
    PORTABLE_CLASS_NAMES,
    RISK_CLASS_INDEX,
    RobotProfile,
    SafetyBatch,
)
from .model import ContinuousTimeEncoding, FrameEncoder, PortableSafetyCore
from .runtime import SafetyThresholds


MULTIDOMAIN_CHECKPOINT_SCHEMA = "portable_safety_multidomain_checkpoint/v1"
MOTION_MODES = ("position_target", "velocity_command", "none")


@dataclass(frozen=True)
class ProfileAdapterConfig:
    key: str
    state_dim: int
    action_dim: int
    motion_mode: str

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("Profile adapter key cannot be empty")
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("Profile adapter dimensions must be positive")
        if self.motion_mode not in MOTION_MODES:
            raise ValueError(
                f"Unsupported motion mode {self.motion_mode}; expected {MOTION_MODES}"
            )
        if self.motion_mode == "position_target" and self.state_dim < self.action_dim:
            raise ValueError("position_target requires state_dim >= action_dim")


@dataclass(frozen=True)
class MultiProfileSafetyConfig:
    profiles: tuple[ProfileAdapterConfig, ...]
    model_dim: int = 128
    vision_channels: tuple[int, ...] = (32, 64, 128)
    transformer_layers: int = 3
    attention_heads: int = 4
    feedforward_multiplier: int = 4
    dropout: float = 0.1
    max_views: int = 8
    time_frequencies: int = 8
    profile_specific_heads: bool = False
    coarse_visual_grid: int = 0
    chunk_visual_context: bool = False

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if not profiles:
            raise ValueError("At least one profile adapter is required")
        keys = [profile.key for profile in profiles]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate profile adapter keys: {keys}")
        if self.model_dim < 16 or self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim must be >=16 and divisible by attention_heads")
        if self.transformer_layers < 1 or self.attention_heads < 1:
            raise ValueError("Transformer layer/head counts must be positive")
        if self.feedforward_multiplier < 1:
            raise ValueError("feedforward_multiplier must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        channels = tuple(int(value) for value in self.vision_channels)
        if not channels or any(value < 4 for value in channels):
            raise ValueError("vision_channels must contain values >=4")
        if self.max_views < 1 or self.time_frequencies < 1:
            raise ValueError("max_views and time_frequencies must be positive")
        if not isinstance(self.profile_specific_heads, bool):
            raise TypeError("profile_specific_heads must be bool")
        if not isinstance(self.chunk_visual_context, bool):
            raise TypeError("chunk_visual_context must be bool")
        if self.coarse_visual_grid < 0:
            raise ValueError("coarse_visual_grid cannot be negative")
        if self.chunk_visual_context and self.coarse_visual_grid < 1:
            raise ValueError(
                "chunk_visual_context requires a positive coarse_visual_grid"
            )
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "vision_channels", channels)

    @property
    def profile_map(self) -> dict[str, ProfileAdapterConfig]:
        return {profile.key: profile for profile in self.profiles}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["profiles"] = [asdict(profile) for profile in self.profiles]
        payload["vision_channels"] = list(self.vision_channels)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MultiProfileSafetyConfig":
        values = dict(payload)
        values["profiles"] = tuple(
            ProfileAdapterConfig(**profile) for profile in values["profiles"]
        )
        if "vision_channels" in values:
            values["vision_channels"] = tuple(values["vision_channels"])
        return cls(**values)


class ProfileInputAdapter(nn.Module):
    def __init__(
        self,
        config: ProfileAdapterConfig,
        model_dim: int,
        *,
        coarse_visual_grid: int = 0,
        max_views: int = 1,
        chunk_visual_context: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_dim = int(model_dim)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.motion_encoder = (
            nn.Sequential(
                nn.LayerNorm(2 * config.action_dim),
                nn.Linear(2 * config.action_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
            if config.motion_mode != "none"
            else None
        )
        self.profile_embedding = nn.Parameter(torch.zeros(model_dim))
        self.coarse_visual_grid = int(coarse_visual_grid)
        self.max_views = int(max_views)
        self.visual_encoder = (
            nn.Sequential(
                nn.LayerNorm(3 * self.coarse_visual_grid**2),
                nn.Linear(3 * self.coarse_visual_grid**2, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
            if self.coarse_visual_grid > 0
            else None
        )
        if self.visual_encoder is not None:
            # A zero residual keeps migration from the original one-profile
            # checkpoint bit-exact while allowing a new robot profile to learn
            # low-frequency scene/contact cues.
            nn.init.zeros_(self.visual_encoder[-1].weight)
            nn.init.zeros_(self.visual_encoder[-1].bias)
        coarse_dim = 3 * self.coarse_visual_grid**2
        self.chunk_visual_encoder = (
            nn.Sequential(
                nn.LayerNorm(4 * self.max_views * coarse_dim),
                nn.Linear(4 * self.max_views * coarse_dim, model_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim // 2, model_dim),
            )
            if chunk_visual_context
            else None
        )
        if self.chunk_visual_encoder is not None:
            nn.init.zeros_(self.chunk_visual_encoder[-1].weight)
            nn.init.zeros_(self.chunk_visual_encoder[-1].bias)

    def validate(self, batch: SafetyBatch) -> None:
        if batch.state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"{self.config.key} state dim is {batch.state.shape[-1]}, "
                f"expected {self.config.state_dim}"
            )
        if batch.action.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"{self.config.key} action dim is {batch.action.shape[-1]}, "
                f"expected {self.config.action_dim}"
            )

    def motion_tokens(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        state_times: torch.Tensor,
        action_times: torch.Tensor,
        batch: SafetyBatch,
    ) -> torch.Tensor:
        if self.motion_encoder is None:
            return action.new_zeros((*action.shape[:2], self.model_dim))
        if self.config.motion_mode == "position_target":
            state_indices = batch.state_mask.long().sum(dim=1) - 1
            gather = state_indices.view(-1, 1, 1).expand(-1, 1, state.shape[-1])
            current = state.gather(1, gather).squeeze(1)[
                :, : self.config.action_dim
            ]
            previous = torch.cat([current.unsqueeze(1), action[:, :-1]], dim=1)
            state_time = state_times.gather(
                1, state_indices.unsqueeze(1)
            ).squeeze(1)
            previous_times = torch.cat(
                [state_time.unsqueeze(1), action_times[:, :-1]], dim=1
            )
            dt = (action_times - previous_times).clamp_min(1e-4)
            displacement = action - previous
            velocity = displacement / dt.unsqueeze(-1)
            features = torch.cat([displacement, velocity], dim=-1)
        elif self.config.motion_mode == "velocity_command":
            delta = torch.cat(
                [torch.zeros_like(action[:, :1]), action[:, 1:] - action[:, :-1]],
                dim=1,
            )
            dt = torch.cat(
                [
                    torch.ones_like(action_times[:, :1]),
                    (action_times[:, 1:] - action_times[:, :-1]).clamp_min(1e-4),
                ],
                dim=1,
            )
            acceleration = delta / dt.unsqueeze(-1)
            features = torch.cat([action, acceleration], dim=-1)
        else:
            raise AssertionError(self.config.motion_mode)
        return self.motion_encoder(features)

    def coarse_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.visual_encoder is None:
            return images.new_zeros((images.shape[0], 0))
        return F.adaptive_avg_pool2d(
            images, (self.coarse_visual_grid, self.coarse_visual_grid)
        ).flatten(1)

    def visual_tokens(
        self, images: torch.Tensor, coarse: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.visual_encoder is None:
            return images.new_zeros((images.shape[0], self.model_dim))
        if coarse is None:
            coarse = self.coarse_visual_features(images)
        return self.visual_encoder(coarse)

    def chunk_visual_tokens(
        self,
        coarse: torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        views: int,
    ) -> torch.Tensor:
        if self.chunk_visual_encoder is None:
            return coarse.new_zeros((batch_size, self.model_dim))
        if views > self.max_views:
            raise ValueError("Visual context view count exceeds adapter maximum")
        sequence = coarse.reshape(batch_size, frames, views, -1)
        if views < self.max_views:
            sequence = F.pad(sequence, (0, 0, 0, self.max_views - views))
        current = sequence[:, -1]
        mean = sequence.mean(dim=1)
        delta = sequence[:, -1] - sequence[:, 0]
        if frames > 1:
            maximum_delta = (sequence[:, 1:] - sequence[:, :-1]).abs().amax(dim=1)
        else:
            maximum_delta = torch.zeros_like(current)
        summary = torch.cat([current, mean, delta, maximum_delta], dim=-1)
        return self.chunk_visual_encoder(summary.flatten(1))


class MultiProfilePortableSafetyCore(nn.Module):
    """Shared safety core with explicit adapters for different robot semantics."""

    def __init__(self, config: MultiProfileSafetyConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.frame_encoder = FrameEncoder(config.vision_channels, dim)
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
        self.profile_adapters = nn.ModuleDict(
            {
                profile.key: ProfileInputAdapter(
                    profile,
                    dim,
                    coarse_visual_grid=config.coarse_visual_grid,
                    max_views=config.max_views,
                    chunk_visual_context=config.chunk_visual_context,
                    dropout=config.dropout,
                )
                for profile in config.profiles
            }
        )
        self.profile_chunk_heads = nn.ModuleDict(
            {
                profile.key: nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, len(PORTABLE_CLASS_NAMES))
                )
                for profile in config.profiles
            }
            if config.profile_specific_heads
            else {}
        )
        self.profile_step_heads = nn.ModuleDict(
            {
                profile.key: nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, len(PORTABLE_CLASS_NAMES))
                )
                for profile in config.profiles
            }
            if config.profile_specific_heads
            else {}
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.view_embedding.weight, std=0.02)
        for adapter in self.profile_adapters.values():
            nn.init.zeros_(adapter.profile_embedding)
        heads = [self.chunk_head, self.step_head]
        heads.extend(self.profile_chunk_heads.values())
        heads.extend(self.profile_step_heads.values())
        for head in heads:
            nn.init.xavier_uniform_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @property
    def profile_keys(self) -> tuple[str, ...]:
        return tuple(self.profile_adapters.keys())

    def forward(
        self, profile_key: str, batch: SafetyBatch
    ) -> dict[str, torch.Tensor]:
        if profile_key not in self.profile_adapters:
            raise KeyError(
                f"Unknown profile {profile_key}; available={self.profile_keys}"
            )
        adapter = self.profile_adapters[profile_key]
        adapter.validate(batch)
        if batch.video.shape[2] > self.config.max_views:
            raise ValueError("Video view count exceeds configured maximum")
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
        coarse_video = adapter.coarse_visual_features(flat_video)
        video_tokens = (
            self.frame_encoder(flat_video)
            + adapter.visual_tokens(flat_video, coarse=coarse_video)
        ).reshape(
            batch_size, frames, views, -1
        )
        profile_embedding = adapter.profile_embedding.view(1, 1, -1)
        view_indices = torch.arange(views, device=video.device)
        video_tokens = (
            video_tokens
            + self.time_encoder(video_times).unsqueeze(2)
            + self.view_embedding(view_indices).view(1, 1, views, -1)
            + self.modality_embedding[0].view(1, 1, 1, -1)
            + profile_embedding.view(1, 1, 1, -1)
        ).reshape(batch_size, frames * views, -1)
        state_tokens = (
            adapter.state_encoder(state)
            + self.time_encoder(state_times)
            + self.modality_embedding[1].view(1, 1, -1)
            + profile_embedding
        )
        action_tokens = (
            adapter.action_encoder(action)
            + adapter.motion_tokens(
                state, action, state_times, action_times, batch
            )
            + self.time_encoder(action_times)
            + self.modality_embedding[2].view(1, 1, -1)
            + profile_embedding
        )
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, video_tokens, state_tokens, action_tokens], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=video.device),
                ~batch.video_mask.reshape(batch_size, frames * views),
                ~batch.state_mask,
                ~batch.action_mask,
            ],
            dim=1,
        )
        fused = self.fusion(tokens, src_key_padding_mask=padding_mask)
        action_start = 1 + frames * views + state.shape[1]
        fused_actions = fused[:, action_start : action_start + action.shape[1]]
        chunk_features = fused[:, 0] + adapter.chunk_visual_tokens(
            coarse_video,
            batch_size=batch_size,
            frames=frames,
            views=views,
        )
        chunk_head = (
            self.profile_chunk_heads[profile_key]
            if self.config.profile_specific_heads
            else self.chunk_head
        )
        step_head = (
            self.profile_step_heads[profile_key]
            if self.config.profile_specific_heads
            else self.step_head
        )
        class_logits = chunk_head(chunk_features)
        step_class_logits = step_head(fused_actions).masked_fill(
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
            "risk_features": chunk_features,
            "step_risk_features": fused_actions,
        }


def initialize_from_single_profile(
    target: MultiProfilePortableSafetyCore,
    source: PortableSafetyCore,
    *,
    profile_key: str,
) -> None:
    if profile_key not in target.profile_adapters:
        raise KeyError(profile_key)
    shared_names = (
        "frame_encoder",
        "time_encoder",
        "view_embedding",
        "fusion",
        "chunk_head",
        "step_head",
    )
    for name in shared_names:
        getattr(target, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
    if target.config.profile_specific_heads:
        target.profile_chunk_heads[profile_key].load_state_dict(
            source.chunk_head.state_dict(), strict=True
        )
        target.profile_step_heads[profile_key].load_state_dict(
            source.step_head.state_dict(), strict=True
        )
    with torch.no_grad():
        target.modality_embedding.copy_(source.modality_embedding)
        target.cls_token.copy_(source.cls_token)
    adapter = target.profile_adapters[profile_key]
    adapter.state_encoder.load_state_dict(source.state_encoder.state_dict(), strict=True)
    adapter.action_encoder.load_state_dict(source.action_encoder.state_dict(), strict=True)
    if source.motion_encoder is None or adapter.motion_encoder is None:
        if source.motion_encoder is not None or adapter.motion_encoder is not None:
            raise RuntimeError("Source and target motion encoders differ")
    else:
        adapter.motion_encoder.load_state_dict(source.motion_encoder.state_dict(), strict=True)
    with torch.no_grad():
        adapter.profile_embedding.zero_()


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedMultiProfileCheckpoint:
    model: MultiProfilePortableSafetyCore
    profiles: dict[str, RobotProfile]
    thresholds: SafetyThresholds
    profile_thresholds: dict[str, SafetyThresholds]
    metadata: dict[str, Any]


def save_multidomain_checkpoint(
    path: str | Path,
    model: MultiProfilePortableSafetyCore,
    profiles: Mapping[str, RobotProfile],
    *,
    thresholds: SafetyThresholds | None = None,
    profile_thresholds: Mapping[str, SafetyThresholds] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    expected = set(model.profile_keys)
    if set(profiles) != expected:
        raise ValueError(
            f"Checkpoint profiles {sorted(profiles)} differ from model {sorted(expected)}"
        )
    for key, adapter_config in model.config.profile_map.items():
        profile = profiles[key]
        if (
            profile.state_dim != adapter_config.state_dim
            or profile.action_dim != adapter_config.action_dim
        ):
            raise ValueError(f"Profile dimensions differ for {key}")
    common_thresholds = thresholds or SafetyThresholds()
    bound_thresholds = (
        {key: common_thresholds for key in expected}
        if profile_thresholds is None
        else dict(profile_thresholds)
    )
    if set(bound_thresholds) != expected:
        raise ValueError(
            "Profile threshold keys differ from checkpoint profile keys"
        )
    payload = {
        "checkpoint_schema": MULTIDOMAIN_CHECKPOINT_SCHEMA,
        "input_schema": "portable_safety_input/v1",
        "class_names": list(PORTABLE_CLASS_NAMES),
        "model_config": model.config.to_dict(),
        "profiles": {key: value.to_dict() for key, value in profiles.items()},
        "profile_fingerprints": {
            key: value.fingerprint for key, value in profiles.items()
        },
        "thresholds": asdict(common_thresholds),
        "profile_thresholds": {
            key: asdict(value) for key, value in bound_thresholds.items()
        },
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(
        checkpoint_path.suffix + f".tmp.{os.getpid()}"
    )
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)
    return checkpoint_path


def load_multidomain_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_profiles: Mapping[str, RobotProfile] | None = None,
) -> LoadedMultiProfileCheckpoint:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if payload.get("checkpoint_schema") != MULTIDOMAIN_CHECKPOINT_SCHEMA:
        raise RuntimeError(f"Unsupported checkpoint schema: {checkpoint_path}")
    if tuple(payload.get("class_names", ())) != PORTABLE_CLASS_NAMES:
        raise RuntimeError("Checkpoint class order differs from runtime")
    config = MultiProfileSafetyConfig.from_dict(payload["model_config"])
    profiles = {
        key: RobotProfile.from_dict(value)
        for key, value in payload["profiles"].items()
    }
    if expected_profiles is not None:
        if set(expected_profiles) != set(profiles):
            raise RuntimeError("Checkpoint profile key set differs")
        for key, expected in expected_profiles.items():
            if profiles[key].fingerprint != expected.fingerprint:
                raise RuntimeError(f"Checkpoint profile differs for {key}")
    model = MultiProfilePortableSafetyCore(config)
    model.load_state_dict(payload["model_state"], strict=True)
    common_thresholds = SafetyThresholds(**payload["thresholds"])
    profile_thresholds = {
        key: SafetyThresholds(**value)
        for key, value in payload.get("profile_thresholds", {}).items()
    }
    if not profile_thresholds:
        profile_thresholds = {key: common_thresholds for key in profiles}
    if set(profile_thresholds) != set(profiles):
        raise RuntimeError("Checkpoint profile threshold key set differs")
    return LoadedMultiProfileCheckpoint(
        model=model,
        profiles=profiles,
        thresholds=common_thresholds,
        profile_thresholds=profile_thresholds,
        metadata=dict(payload.get("metadata", {})),
    )


def config_fingerprint(config: MultiProfileSafetyConfig) -> str:
    value = json.dumps(
        config.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()
