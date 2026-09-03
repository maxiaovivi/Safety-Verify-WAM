from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .contracts import AdaptedSafetyInput, RobotProfile, SafetyBatch


TensorLike = torch.Tensor | np.ndarray | Sequence[float]


def _tensor_copy(value: TensorLike) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return torch.as_tensor(np.asarray(value).copy())


def _as_float_tensor(value: TensorLike, name: str) -> torch.Tensor:
    try:
        tensor = _tensor_copy(value)
    except Exception as exc:
        raise TypeError(f"Cannot convert {name} to a tensor") from exc
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.contiguous()


def _default_timestamps(
    batch: int,
    steps: int,
    dt: float,
    *,
    future: bool,
    device: torch.device,
) -> torch.Tensor:
    if future:
        values = torch.arange(1, steps + 1, dtype=torch.float32, device=device) * dt
    else:
        values = torch.arange(1 - steps, 1, dtype=torch.float32, device=device) * dt
    return values.unsqueeze(0).expand(batch, -1).clone()


def _canonical_video(video: TensorLike, pixel_range: str) -> torch.Tensor:
    source = _tensor_copy(video)
    if source.ndim == 3:
        source = source.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif source.ndim == 4:
        source = source.unsqueeze(1).unsqueeze(2)
    elif source.ndim == 5:
        source = source.unsqueeze(2)
    elif source.ndim != 6:
        raise ValueError(
            "video must be CHW, BCHW, BTFCHW, or BTFVCHW-compatible; "
            f"got {tuple(source.shape)}"
        )
    if source.shape[3] != 3:
        raise ValueError(
            "Canonical video expects channel-first RGB with three channels, got "
            f"{tuple(source.shape)}"
        )
    if pixel_range == "uint8":
        if source.dtype != torch.uint8:
            raise TypeError("pixel_range='uint8' requires uint8 video")
        result = source.float() / 255.0
    elif pixel_range == "zero_one":
        result = source.float()
    elif pixel_range == "minus_one_one":
        result = (source.float() + 1.0) * 0.5
    else:
        raise ValueError(
            "pixel_range must be 'uint8', 'zero_one', or 'minus_one_one'"
        )
    return result.contiguous()


def _canonical_state(state: TensorLike) -> torch.Tensor:
    result = _as_float_tensor(state, "state")
    if result.ndim == 1:
        result = result.unsqueeze(0).unsqueeze(0)
    elif result.ndim == 2:
        result = result.unsqueeze(1)
    elif result.ndim != 3:
        raise ValueError(f"state must be [S], [B,S], or [B,T,S], got {result.shape}")
    return result


def _canonical_action(action: TensorLike) -> torch.Tensor:
    result = _as_float_tensor(action, "action")
    if result.ndim == 2:
        result = result.unsqueeze(0)
    elif result.ndim != 3:
        raise ValueError(f"action must be [T,A] or [B,T,A], got {result.shape}")
    return result


def _mask(
    value: TensorLike | None,
    shape: tuple[int, ...],
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    result = _tensor_copy(value).to(device=device)
    if tuple(result.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(result.shape)}, expected {shape}")
    return result.to(dtype=torch.bool)


def _timestamps(
    value: TensorLike | None,
    *,
    batch: int,
    steps: int,
    dt: float,
    future: bool,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return _default_timestamps(batch, steps, dt, future=future, device=device)
    result = _as_float_tensor(value, name).to(device=device)
    if result.ndim == 1:
        result = result.unsqueeze(0)
    if result.shape[0] == 1 and batch > 1:
        result = result.expand(batch, -1).clone()
    if tuple(result.shape) != (batch, steps):
        raise ValueError(
            f"{name} has shape {tuple(result.shape)}, expected {(batch, steps)}"
        )
    return result


class CanonicalSafetyAdapter:
    """Convert policy-facing tensors into the host-neutral safety contract."""

    def __init__(
        self,
        profile: RobotProfile,
        *,
        image_size: tuple[int, int] | None = None,
        host_name: str = "generic",
    ) -> None:
        self.profile = profile
        self.image_size = profile.image_size if image_size is None else image_size
        self.host_name = str(host_name)
        if tuple(self.image_size) != profile.image_size:
            raise ValueError(
                f"Adapter image_size {tuple(self.image_size)} differs from robot "
                f"profile {profile.image_size}"
            )

    def from_tensors(
        self,
        *,
        video: TensorLike,
        state: TensorLike,
        action: TensorLike,
        pixel_range: str,
        values_are_normalized: bool = False,
        raw_state: TensorLike | None = None,
        raw_action: TensorLike | None = None,
        video_timestamps: TensorLike | None = None,
        state_timestamps: TensorLike | None = None,
        action_timestamps: TensorLike | None = None,
        video_mask: TensorLike | None = None,
        state_mask: TensorLike | None = None,
        action_mask: TensorLike | None = None,
    ) -> AdaptedSafetyInput:
        video_tensor = _canonical_video(video, pixel_range)
        state_tensor = _canonical_state(state)
        action_tensor = _canonical_action(action)
        batch = int(video_tensor.shape[0])
        if state_tensor.shape[0] != batch or action_tensor.shape[0] != batch:
            raise ValueError("video, state, and action batch sizes differ")
        if self.image_size is not None and tuple(video_tensor.shape[-2:]) != tuple(
            self.image_size
        ):
            original = video_tensor.shape
            flattened = video_tensor.reshape(-1, 3, *original[-2:])
            flattened = F.interpolate(
                flattened,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
            video_tensor = flattened.reshape(*original[:-2], *self.image_size)

        raw_state_tensor: torch.Tensor | None
        raw_action_tensor: torch.Tensor | None
        if values_are_normalized:
            normalized_state = state_tensor
            normalized_action = action_tensor
            raw_state_tensor = (
                None if raw_state is None else _canonical_state(raw_state)
            )
            raw_action_tensor = (
                None if raw_action is None else _canonical_action(raw_action)
            )
        else:
            raw_state_tensor = state_tensor
            raw_action_tensor = action_tensor
            normalized_state = self.profile.normalize_state(state_tensor)
            normalized_action = self.profile.normalize_action(action_tensor)

        video_frames = int(video_tensor.shape[1])
        views = int(video_tensor.shape[2])
        state_steps = int(state_tensor.shape[1])
        action_steps = int(action_tensor.shape[1])
        device = video_tensor.device
        batch_value = SafetyBatch(
            video=video_tensor,
            state=normalized_state,
            action=normalized_action,
            video_timestamps=_timestamps(
                video_timestamps,
                batch=batch,
                steps=video_frames,
                dt=self.profile.control_dt,
                future=False,
                name="video_timestamps",
                device=device,
            ),
            state_timestamps=_timestamps(
                state_timestamps,
                batch=batch,
                steps=state_steps,
                dt=self.profile.control_dt,
                future=False,
                name="state_timestamps",
                device=device,
            ),
            action_timestamps=_timestamps(
                action_timestamps,
                batch=batch,
                steps=action_steps,
                dt=self.profile.control_dt,
                future=True,
                name="action_timestamps",
                device=device,
            ),
            video_mask=_mask(
                video_mask,
                (batch, video_frames, views),
                "video_mask",
                device,
            ),
            state_mask=_mask(
                state_mask,
                (batch, state_steps),
                "state_mask",
                device,
            ),
            action_mask=_mask(
                action_mask,
                (batch, action_steps),
                "action_mask",
                device,
            ),
        )
        adapted = AdaptedSafetyInput(
            batch=batch_value,
            profile_fingerprint=self.profile.fingerprint,
            raw_state=raw_state_tensor,
            raw_action=raw_action_tensor,
            host_name=self.host_name,
        )
        adapted.validate(self.profile)
        return adapted


class RoboTwinPolicyAdapter(CanonicalSafetyAdapter):
    """Adapter for RoboTwin observations and Fast-WAM-style candidate dicts.

    It reads only public observation pixels, ``joint_action.vector``, and the
    denormalized candidate action. No Fast-WAM or Efficient-WAM hidden feature
    is imported, so the same adapter works with either policy.
    """

    def __init__(
        self,
        profile: RobotProfile,
        *,
        image_size: tuple[int, int] | None = None,
        camera_keys: tuple[str, ...] | None = None,
        host_name: str = "robotwin-policy",
    ) -> None:
        super().__init__(profile, image_size=image_size, host_name=host_name)
        if camera_keys is None:
            camera_keys = profile.camera_names or (
                "head_camera",
                "left_camera",
                "right_camera",
            )
        if not camera_keys:
            raise ValueError("camera_keys cannot be empty")
        self.camera_keys = tuple(camera_keys)
        if profile.camera_names and self.camera_keys != profile.camera_names:
            raise ValueError(
                f"camera_keys {self.camera_keys} differ from robot profile "
                f"{profile.camera_names}"
            )

    def _observation_frame(self, observation: Mapping[str, Any]) -> torch.Tensor:
        try:
            camera_data = observation["observation"]
        except KeyError as exc:
            raise KeyError(
                "RoboTwin observation has no 'observation' camera mapping"
            ) from exc
        views: list[torch.Tensor] = []
        for camera_key in self.camera_keys:
            try:
                array = np.asarray(camera_data[camera_key]["rgb"])
            except KeyError as exc:
                raise KeyError(f"RoboTwin observation has no {camera_key}.rgb") from exc
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(
                    f"{camera_key}.rgb must be HWC RGB, got {tuple(array.shape)}"
                )
            view = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
            view = F.interpolate(
                view.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            views.append(view)
        return torch.stack(views, dim=0)

    def _history_tensors(
        self,
        observation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observations = (
            [observation]
            if isinstance(observation, Mapping)
            else list(observation)
        )
        if not observations:
            raise ValueError("observation history cannot be empty")
        frames = torch.stack(
            [self._observation_frame(item) for item in observations], dim=0
        ).unsqueeze(0)
        states = torch.stack(
            [
                torch.as_tensor(
                    np.asarray(item["joint_action"]["vector"]).copy(),
                    dtype=torch.float32,
                )
                for item in observations
            ],
            dim=0,
        ).unsqueeze(0)
        return frames, states

    def from_policy_candidates(
        self,
        observation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any] | TensorLike],
        *,
        video_timestamps: TensorLike | None = None,
        state_timestamps: TensorLike | None = None,
        action_timestamps: TensorLike | None = None,
        action_mask: TensorLike | None = None,
    ) -> AdaptedSafetyInput:
        candidate_values = list(candidates)
        if not candidate_values:
            raise ValueError("candidates cannot be empty")
        frames, states = self._history_tensors(observation)
        actions = []
        for candidate in candidate_values:
            value = candidate["action"] if isinstance(candidate, Mapping) else candidate
            canonical = _canonical_action(value)
            if canonical.shape[0] != 1:
                raise ValueError("Each policy candidate must contain one action chunk")
            actions.append(canonical.squeeze(0))
        try:
            action_batch = torch.stack(actions, dim=0)
        except RuntimeError as exc:
            raise ValueError(
                "All policy candidates must have the same action shape"
            ) from exc
        candidate_count = len(actions)
        frames = frames.expand(candidate_count, *frames.shape[1:]).clone()
        states = states.expand(candidate_count, *states.shape[1:]).clone()
        return self.from_tensors(
            video=frames,
            state=states,
            action=action_batch,
            pixel_range="zero_one",
            values_are_normalized=False,
            video_timestamps=video_timestamps,
            state_timestamps=state_timestamps,
            action_timestamps=action_timestamps,
            action_mask=action_mask,
        )

    def from_policy_candidate(
        self,
        observation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        candidate: Mapping[str, Any] | TensorLike,
        *,
        video_timestamps: TensorLike | None = None,
        state_timestamps: TensorLike | None = None,
        action_timestamps: TensorLike | None = None,
        action_mask: TensorLike | None = None,
    ) -> AdaptedSafetyInput:
        return self.from_policy_candidates(
            observation,
            [candidate],
            video_timestamps=video_timestamps,
            state_timestamps=state_timestamps,
            action_timestamps=action_timestamps,
            action_mask=action_mask,
        )
