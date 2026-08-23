from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .ovcr_s import OVCRSActionGenerator, OVCRSConfig


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


def compact_first_frame_video_cache(
    video_cache: Mapping[str, Any],
    *,
    expected_layers: int,
    expected_dim: int,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Extract first-frame Efficient-WAM K/V in OVCR-S [B,S,D] layout."""

    keys = video_cache.get("video_k")
    values = video_cache.get("video_v")
    if not isinstance(keys, Sequence) or not isinstance(values, Sequence):
        raise TypeError("video_cache must contain video_k/video_v layer sequences")
    if len(keys) != expected_layers or len(values) != expected_layers:
        raise ValueError(
            "Efficient-WAM and OVCR-S layer counts differ: "
            f"k={len(keys)}, v={len(values)}, expected={expected_layers}"
        )

    token_count = _condition_token_count(video_cache)
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
        if layer_k.shape[1] < token_count:
            raise ValueError(
                f"Video cache layer {layer_index} has {layer_k.shape[1]} tokens, "
                f"but current frame needs {token_count}"
            )
        if layer_k.shape[-1] != expected_dim:
            raise ValueError(
                f"Video cache layer {layer_index} dim is {layer_k.shape[-1]}, "
                f"expected {expected_dim}"
            )
        compact.append(
            {
                "k": layer_k[:, :token_count].contiguous(),
                "v": layer_v[:, :token_count].contiguous(),
            }
        )
    return tuple(compact)


def prepare_ovcrs_conditioning(
    student: OVCRSActionGenerator,
    *,
    condition_latent: torch.Tensor,
    video_cache: Mapping[str, Any],
    observation_downsample_factor: int = 1,
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
    compact_cache = compact_first_frame_video_cache(
        video_cache,
        expected_layers=student.config.num_layers,
        expected_dim=student.config.video_dim,
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
