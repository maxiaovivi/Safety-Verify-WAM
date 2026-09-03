from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch


PORTABLE_INPUT_SCHEMA = "portable_safety_input/v1"
PORTABLE_CLASS_NAMES = ("safe", "risk")
SAFE_CLASS_INDEX = 0
RISK_CLASS_INDEX = 1


def _numeric_tuple(
    value: tuple[float, ...] | list[float] | None,
    *,
    expected: int,
    name: str,
    strictly_positive: bool = False,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    result = tuple(float(item) for item in value)
    if len(result) != expected:
        raise ValueError(f"{name} has {len(result)} values, expected {expected}")
    tensor = torch.tensor(result, dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if strictly_positive and bool((tensor <= 0).any()):
        raise ValueError(f"{name} must contain positive values")
    return result


@dataclass(frozen=True)
class RobotProfile:
    """Canonical robot/action contract shared by otherwise unrelated policies."""

    name: str
    state_dim: int
    action_dim: int
    control_dt: float
    state_mean: tuple[float, ...] | None = None
    state_std: tuple[float, ...] | None = None
    action_mean: tuple[float, ...] | None = None
    action_std: tuple[float, ...] | None = None
    joint_names: tuple[str, ...] = ()
    camera_names: tuple[str, ...] = ()
    image_size: tuple[int, int] = (128, 128)
    pixel_semantics: str = "rgb"
    state_semantics: str = "joint_position"
    action_semantics: str = "joint_position_target"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Robot profile name cannot be empty")
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("state_dim and action_dim must be positive")
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive")
        state_mean = _numeric_tuple(
            self.state_mean,
            expected=self.state_dim,
            name="state_mean",
        )
        state_std = _numeric_tuple(
            self.state_std,
            expected=self.state_dim,
            name="state_std",
            strictly_positive=True,
        )
        action_mean = _numeric_tuple(
            self.action_mean,
            expected=self.action_dim,
            name="action_mean",
        )
        action_std = _numeric_tuple(
            self.action_std,
            expected=self.action_dim,
            name="action_std",
            strictly_positive=True,
        )
        if (state_mean is None) != (state_std is None):
            raise ValueError("state_mean and state_std must be provided together")
        if (action_mean is None) != (action_std is None):
            raise ValueError("action_mean and action_std must be provided together")
        joint_names = tuple(str(item) for item in self.joint_names)
        if joint_names and len(joint_names) != self.action_dim:
            raise ValueError(
                f"joint_names has {len(joint_names)} values, expected {self.action_dim}"
            )
        camera_names = tuple(str(item) for item in self.camera_names)
        if any(not item.strip() for item in camera_names):
            raise ValueError("camera_names cannot contain empty values")
        if len(set(camera_names)) != len(camera_names):
            raise ValueError("camera_names cannot contain duplicates")
        image_size = tuple(int(item) for item in self.image_size)
        if len(image_size) != 2 or any(item < 16 for item in image_size):
            raise ValueError("image_size must contain height and width of at least 16")
        if self.pixel_semantics.lower() != "rgb":
            raise ValueError("Only RGB pixel semantics are supported")
        object.__setattr__(self, "state_mean", state_mean)
        object.__setattr__(self, "state_std", state_std)
        object.__setattr__(self, "action_mean", action_mean)
        object.__setattr__(self, "action_std", action_std)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "camera_names", camera_names)
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "pixel_semantics", self.pixel_semantics.lower())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RobotProfile":
        values = dict(payload)
        for key in (
            "state_mean",
            "state_std",
            "action_mean",
            "action_std",
            "joint_names",
            "camera_names",
            "image_size",
        ):
            if values.get(key) is not None:
                values[key] = tuple(values[key])
        return cls(**values)

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _normalize(
        self,
        value: torch.Tensor,
        mean: tuple[float, ...] | None,
        std: tuple[float, ...] | None,
    ) -> torch.Tensor:
        if mean is None or std is None:
            return value
        mean_tensor = value.new_tensor(mean)
        std_tensor = value.new_tensor(std)
        return (value - mean_tensor) / std_tensor

    def normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return self._normalize(value, self.state_mean, self.state_std)

    def normalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return self._normalize(value, self.action_mean, self.action_std)


def _require_prefix_mask(mask: torch.Tensor, name: str) -> None:
    if mask.shape[-1] > 1 and bool((mask[..., 1:] & ~mask[..., :-1]).any()):
        raise ValueError(f"{name} valid values must form a contiguous prefix")


def _validate_timestamps(
    timestamps: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> None:
    if tuple(timestamps.shape) != tuple(mask.shape):
        raise ValueError(
            f"{name} shape {tuple(timestamps.shape)} differs from mask "
            f"{tuple(mask.shape)}"
        )
    if not torch.isfinite(timestamps).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if timestamps.shape[1] > 1:
        comparable = mask[:, 1:] & mask[:, :-1]
        decreasing = timestamps[:, 1:] < timestamps[:, :-1]
        if bool((decreasing & comparable).any()):
            raise ValueError(f"{name} must be nondecreasing over valid values")


@dataclass(frozen=True)
class SafetyBatch:
    """Host-neutral, normalized tensors consumed by the portable safety core.

    Video uses ``[B,F,V,3,H,W]`` so frame history and multiple camera views stay
    explicit. State and candidate action use ``[B,T,D]``. Timestamps are in
    seconds relative to the decision instant; negative history and positive
    future action times are recommended but not required.
    """

    video: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    video_timestamps: torch.Tensor
    state_timestamps: torch.Tensor
    action_timestamps: torch.Tensor
    video_mask: torch.Tensor
    state_mask: torch.Tensor
    action_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.action.shape[0])

    def validate(self, profile: RobotProfile) -> None:
        if self.video.ndim != 6 or self.video.shape[3] != 3:
            raise ValueError(
                "video must be [B,F,V,3,H,W], got " f"{tuple(self.video.shape)}"
            )
        if self.state.ndim != 3:
            raise ValueError(f"state must be [B,T,S], got {tuple(self.state.shape)}")
        if self.action.ndim != 3:
            raise ValueError(
                f"action must be [B,T,A], got {tuple(self.action.shape)}"
            )
        batch = int(self.video.shape[0])
        if self.state.shape[0] != batch or self.action.shape[0] != batch:
            raise ValueError("video, state, and action batch sizes differ")
        if self.state.shape[-1] != profile.state_dim:
            raise ValueError(
                f"state dim is {self.state.shape[-1]}, expected {profile.state_dim}"
            )
        if self.action.shape[-1] != profile.action_dim:
            raise ValueError(
                f"action dim is {self.action.shape[-1]}, expected {profile.action_dim}"
            )
        if profile.camera_names and self.video.shape[2] != len(profile.camera_names):
            raise ValueError(
                f"video has {self.video.shape[2]} views, robot profile defines "
                f"{len(profile.camera_names)}"
            )
        expected_masks = {
            "video_mask": (batch, self.video.shape[1], self.video.shape[2]),
            "state_mask": (batch, self.state.shape[1]),
            "action_mask": (batch, self.action.shape[1]),
        }
        for name, expected in expected_masks.items():
            value = getattr(self, name)
            if tuple(value.shape) != tuple(expected):
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, expected {tuple(expected)}"
                )
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must use torch.bool")
            if not bool(value.reshape(batch, -1).any(dim=1).all()):
                raise ValueError(f"Every batch item needs at least one valid {name}")
        _require_prefix_mask(self.state_mask, "state_mask")
        _require_prefix_mask(self.action_mask, "action_mask")
        _validate_timestamps(
            self.video_timestamps,
            self.video_mask.any(dim=2),
            "video_timestamps",
        )
        _validate_timestamps(
            self.state_timestamps,
            self.state_mask,
            "state_timestamps",
        )
        _validate_timestamps(
            self.action_timestamps,
            self.action_mask,
            "action_timestamps",
        )
        for name in ("video", "state", "action"):
            value = getattr(self, name)
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
        if bool((self.video < -1e-6).any()) or bool((self.video > 1.0 + 1e-6).any()):
            raise ValueError("video pixels must be in [0, 1]")

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "SafetyBatch":
        floating = {
            name: getattr(self, name).to(
                device=device,
                dtype=dtype or getattr(self, name).dtype,
                non_blocking=True,
            )
            for name in (
                "video",
                "state",
                "action",
                "video_timestamps",
                "state_timestamps",
                "action_timestamps",
            )
        }
        masks = {
            name: getattr(self, name).to(device=device, non_blocking=True)
            for name in ("video_mask", "state_mask", "action_mask")
        }
        return replace(self, **floating, **masks)


@dataclass(frozen=True)
class AdaptedSafetyInput:
    """Normalized model input plus optional raw values for physical checks."""

    batch: SafetyBatch
    profile_fingerprint: str
    raw_state: torch.Tensor | None = None
    raw_action: torch.Tensor | None = None
    host_name: str = "generic"

    def validate(self, profile: RobotProfile) -> None:
        if self.profile_fingerprint != profile.fingerprint:
            raise RuntimeError(
                "Robot profile mismatch between adapter and safety module: "
                f"adapter={self.profile_fingerprint}, model={profile.fingerprint}"
            )
        self.batch.validate(profile)
        if self.raw_state is not None:
            if tuple(self.raw_state.shape) != tuple(self.batch.state.shape):
                raise ValueError("raw_state shape differs from normalized state")
            if not torch.isfinite(self.raw_state).all():
                raise ValueError("raw_state contains NaN or infinity")
        if self.raw_action is not None:
            if tuple(self.raw_action.shape) != tuple(self.batch.action.shape):
                raise ValueError("raw_action shape differs from normalized action")
            if not torch.isfinite(self.raw_action).all():
                raise ValueError("raw_action contains NaN or infinity")

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "AdaptedSafetyInput":
        return replace(
            self,
            batch=self.batch.to(device, dtype=dtype),
            raw_state=None
            if self.raw_state is None
            else self.raw_state.to(device=device, dtype=torch.float32),
            raw_action=None
            if self.raw_action is None
            else self.raw_action.to(
                device=device, dtype=torch.float32
            ),
        )
