from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .ovcr_s import OVCRSActionGenerator, OVCRSConfig


@dataclass(frozen=True)
class MeanStdStatistics:
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError("Normalization mean/std must be one-dimensional")
        if self.mean.shape != self.std.shape or self.mean.numel() == 0:
            raise ValueError(
                "Normalization mean/std shapes differ: "
                f"{tuple(self.mean.shape)} vs {tuple(self.std.shape)}"
            )
        if (self.std <= 0).any():
            raise ValueError("Normalization std must be positive in every dimension")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        mean_key: str,
        std_key: str,
        source: str,
    ) -> "MeanStdStatistics":
        if mean_key not in payload or std_key not in payload:
            raise ValueError(
                f"{source} is missing {mean_key!r}/{std_key!r}"
            )
        return cls(
            mean=torch.as_tensor(payload[mean_key], dtype=torch.float32).reshape(-1),
            std=torch.as_tensor(payload[std_key], dtype=torch.float32).reshape(-1),
        )

    def _parameters_for(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.shape[-1] != self.mean.numel():
            raise ValueError(
                f"Tensor dim {tensor.shape[-1]} differs from stats dim {self.mean.numel()}"
            )
        return (
            self.mean.to(device=tensor.device, dtype=tensor.dtype),
            self.std.to(device=tensor.device, dtype=tensor.dtype),
        )

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self._parameters_for(tensor)
        return (tensor - mean) / std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self._parameters_for(tensor)
        return tensor * std + mean


@dataclass(frozen=True)
class AHAEfficientNormalizerBridge:
    """Map AHA-normalized state/actions into Efficient-WAM's qpos space."""

    aha_action: MeanStdStatistics
    aha_state: MeanStdStatistics
    efficient_qpos: MeanStdStatistics

    @classmethod
    def from_dataset_stats(
        cls,
        aha_stats_path: str | Path,
        efficient_stats_path: str | Path,
        *,
        aha_action_key: str = "default",
        aha_state_key: str = "default",
        efficient_key: str = "robotwin_qpos",
    ) -> "AHAEfficientNormalizerBridge":
        aha_path = Path(aha_stats_path).expanduser().resolve()
        efficient_path = Path(efficient_stats_path).expanduser().resolve()
        aha_payload = json.loads(aha_path.read_text(encoding="utf-8"))
        efficient_payload = json.loads(efficient_path.read_text(encoding="utf-8"))

        def aha_stats(kind: str, key: str) -> MeanStdStatistics:
            kind_payload = aha_payload.get(kind)
            if not isinstance(kind_payload, Mapping):
                raise ValueError(f"AHA stats have no {kind!r} mapping: {aha_path}")
            selected = kind_payload.get(key)
            if not isinstance(selected, Mapping):
                raise ValueError(
                    f"AHA stats have no {kind}.{key} mapping: {aha_path}"
                )
            return MeanStdStatistics.from_mapping(
                selected,
                mean_key="global_mean",
                std_key="global_std",
                source=f"AHA {kind}.{key}",
            )

        selected_efficient = efficient_payload.get(efficient_key)
        if not isinstance(selected_efficient, Mapping):
            raise ValueError(
                f"Efficient stats have no {efficient_key!r} mapping: {efficient_path}"
            )
        efficient_qpos = MeanStdStatistics.from_mapping(
            selected_efficient,
            mean_key="mean",
            std_key="std",
            source=f"Efficient {efficient_key}",
        )
        aha_action = aha_stats("action", aha_action_key)
        aha_state = aha_stats("state", aha_state_key)
        dimensions = {
            aha_action.mean.numel(),
            aha_state.mean.numel(),
            efficient_qpos.mean.numel(),
        }
        if len(dimensions) != 1:
            raise ValueError(
                "AHA action/state and Efficient qpos dimensions differ: "
                f"{sorted(dimensions)}"
            )
        return cls(
            aha_action=aha_action,
            aha_state=aha_state,
            efficient_qpos=efficient_qpos,
        )

    def action_aha_to_efficient(self, action: torch.Tensor) -> torch.Tensor:
        physical = self.aha_action.denormalize(action)
        return self.efficient_qpos.normalize(physical)

    def state_aha_to_efficient(self, state: torch.Tensor) -> torch.Tensor:
        physical = self.aha_state.denormalize(state)
        return self.efficient_qpos.normalize(physical)

    def action_efficient_to_aha(self, action: torch.Tensor) -> torch.Tensor:
        physical = self.efficient_qpos.denormalize(action)
        return self.aha_action.normalize(physical)


def _as_int(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one value, got shape {tuple(value.shape)}")
        value = value.item()
    return int(value)


def observation_tokens_from_condition_latent(
    condition_latent: torch.Tensor,
    *,
    downsample_factor: int = 1,
) -> torch.Tensor:
    """Convert the current-frame VAE latent into OVCR-S observation tokens."""

    if condition_latent.ndim != 5:
        raise ValueError(
            "condition_latent must be [B,C,1,H,W], "
            f"got {tuple(condition_latent.shape)}"
        )
    if condition_latent.shape[2] != 1:
        raise ValueError(
            "condition_latent must contain one current frame, "
            f"got time dimension {condition_latent.shape[2]}"
        )
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be positive")

    spatial = condition_latent[:, :, 0]
    if downsample_factor > 1:
        spatial = F.avg_pool2d(
            spatial,
            kernel_size=downsample_factor,
            stride=downsample_factor,
        )
    return spatial.flatten(2).transpose(1, 2).contiguous()


def _condition_token_count(video_cache: Mapping[str, Any]) -> int:
    grid_sizes = video_cache.get("grid_sizes")
    if isinstance(grid_sizes, Mapping):
        if "condition_seq_len" not in grid_sizes:
            raise ValueError("Multiscale video cache is missing condition_seq_len")
        return _as_int(grid_sizes["condition_seq_len"], name="condition_seq_len")

    if isinstance(grid_sizes, torch.Tensor):
        if grid_sizes.ndim == 2:
            if grid_sizes.shape[0] != 1 or grid_sizes.shape[1] < 3:
                raise ValueError(
                    "Single-frame cache expects grid_sizes [1,3], "
                    f"got {tuple(grid_sizes.shape)}"
                )
            return int(grid_sizes[0, 1].item()) * int(grid_sizes[0, 2].item())
        if grid_sizes.ndim == 1 and grid_sizes.numel() >= 3:
            return int(grid_sizes[1].item()) * int(grid_sizes[2].item())

    raise ValueError("Cannot determine current-frame token count from video cache")


def compact_video_cache(
    video_cache: Mapping[str, Any],
    *,
    expected_layers: int,
    expected_dim: int,
    condition_only: bool = False,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Convert Efficient-WAM K/V to OVCR-S [B,S,D] layout.

    The full cache is the default because the Efficient action expert was
    trained to attend to both the condition frame and imagined future tokens.
    ``condition_only`` remains available for the old first-frame ablation.
    """

    keys = video_cache.get("video_k")
    values = video_cache.get("video_v")
    if not isinstance(keys, Sequence) or not isinstance(values, Sequence):
        raise TypeError("video_cache must contain video_k/video_v layer sequences")
    if len(keys) != expected_layers or len(values) != expected_layers:
        raise ValueError(
            "Efficient-WAM and OVCR-S layer counts differ: "
            f"k={len(keys)}, v={len(values)}, expected={expected_layers}"
        )

    token_count = _condition_token_count(video_cache) if condition_only else None
    compact: list[dict[str, torch.Tensor]] = []
    for layer_index, (layer_k, layer_v) in enumerate(zip(keys, values), start=1):
        if not isinstance(layer_k, torch.Tensor) or not isinstance(layer_v, torch.Tensor):
            raise TypeError(f"Video cache layer {layer_index} is not a tensor pair")
        if layer_k.shape != layer_v.shape:
            raise ValueError(
                f"Video cache layer {layer_index} K/V shapes differ: "
                f"{tuple(layer_k.shape)} vs {tuple(layer_v.shape)}"
            )
        if layer_k.ndim == 4:
            layer_k = layer_k.flatten(2)
            layer_v = layer_v.flatten(2)
        if layer_k.ndim != 3:
            raise ValueError(
                f"Video cache layer {layer_index} must be [B,S,D] or [B,S,H,Dh], "
                f"got {tuple(layer_k.shape)}"
            )
        if token_count is not None and layer_k.shape[1] < token_count:
            raise ValueError(
                f"Video cache layer {layer_index} has {layer_k.shape[1]} tokens, "
                f"but current frame needs {token_count}"
            )
        if layer_k.shape[-1] != expected_dim:
            raise ValueError(
                f"Video cache layer {layer_index} dim is {layer_k.shape[-1]}, "
                f"expected {expected_dim}"
            )
        selected_tokens = layer_k.shape[1] if token_count is None else token_count
        compact.append(
            {
                "k": layer_k[:, :selected_tokens].contiguous(),
                "v": layer_v[:, :selected_tokens].contiguous(),
            }
        )
    return tuple(compact)


def compact_first_frame_video_cache(
    video_cache: Mapping[str, Any],
    *,
    expected_layers: int,
    expected_dim: int,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Compatibility wrapper for the first-frame-only ablation."""

    return compact_video_cache(
        video_cache,
        expected_layers=expected_layers,
        expected_dim=expected_dim,
        condition_only=True,
    )


def prepare_ovcrs_conditioning(
    student: OVCRSActionGenerator,
    *,
    condition_latent: torch.Tensor,
    video_cache: Mapping[str, Any],
    observation_downsample_factor: int = 1,
    condition_only: bool = False,
) -> dict[str, Any]:
    parameter = next(student.parameters())
    observation_tokens = observation_tokens_from_condition_latent(
        condition_latent,
        downsample_factor=observation_downsample_factor,
    ).to(device=parameter.device, dtype=parameter.dtype)
    if observation_tokens.shape[-1] != student.config.observation_dim:
        raise ValueError(
            "Efficient-WAM VAE latent channels differ from OVCR-S observation_dim: "
            f"{observation_tokens.shape[-1]} vs {student.config.observation_dim}"
        )
    compact_cache = compact_video_cache(
        video_cache,
        expected_layers=student.config.num_layers,
        expected_dim=student.config.video_dim,
        condition_only=condition_only,
    )
    compact_cache = tuple(
        {
            "k": layer["k"].to(device=parameter.device, dtype=parameter.dtype),
            "v": layer["v"].to(device=parameter.device, dtype=parameter.dtype),
        }
        for layer in compact_cache
    )
    observation_mask = torch.ones(
        observation_tokens.shape[:2],
        device=observation_tokens.device,
        dtype=torch.bool,
    )
    return student.prepare_conditioning(
        observation_tokens,
        compact_cache,
        observation_mask,
        return_trace=False,
    )


def load_ovcrs_student(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[OVCRSActionGenerator, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "ovcr_s_stage1":
        raise TypeError(f"Invalid OVCR-S Stage 1 checkpoint: {path}")
    student_state = payload.get("student")
    student_config = payload.get("student_config")
    if not isinstance(student_state, Mapping) or not isinstance(student_config, Mapping):
        raise TypeError(f"OVCR-S checkpoint is missing student state/config: {path}")

    student = OVCRSActionGenerator(OVCRSConfig(**dict(student_config)))
    student.load_state_dict(student_state, strict=True)
    student.to(device=torch.device(device), dtype=dtype).eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student, payload


@dataclass(frozen=True)
class AHAActionDenormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_dataset_stats(
        cls,
        stats_path: str | Path,
        *,
        action_key: str | None = None,
    ) -> "AHAActionDenormalizer":
        path = Path(stats_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        action_stats = payload.get("action")
        if not isinstance(action_stats, Mapping) or not action_stats:
            raise ValueError(f"AHA dataset stats have no action mapping: {path}")
        if action_key is None:
            if len(action_stats) != 1:
                raise ValueError(
                    f"AHA dataset stats contain multiple action keys: {sorted(action_stats)}"
                )
            action_key = str(next(iter(action_stats)))
        selected = action_stats.get(action_key)
        if not isinstance(selected, Mapping):
            raise ValueError(f"AHA dataset stats have no action key {action_key!r}: {path}")
        if "global_mean" not in selected or "global_std" not in selected:
            raise ValueError(f"AHA action stats are missing global_mean/global_std: {path}")
        mean = torch.as_tensor(selected["global_mean"], dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(selected["global_std"], dtype=torch.float32).reshape(-1)
        if mean.shape != std.shape or mean.numel() == 0:
            raise ValueError(
                f"AHA action mean/std shapes differ: {tuple(mean.shape)} vs {tuple(std.shape)}"
            )
        if (std <= 0).any():
            raise ValueError("AHA action std must be positive in every dimension")
        return cls(mean=mean, std=std)

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.mean.numel():
            raise ValueError(
                f"Action dim {action.shape[-1]} differs from stats dim {self.mean.numel()}"
            )
        mean = self.mean.to(device=action.device, dtype=torch.float32)
        std = self.std.to(device=action.device, dtype=torch.float32)
        return action.float() * std + mean
