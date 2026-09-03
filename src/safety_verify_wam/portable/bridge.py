from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .adapters import RoboTwinPolicyAdapter, TensorLike
from .checkpoint import LoadedPortableSafety, load_portable_checkpoint
from .runtime import SafetyAssessment


class RoboTwinSafetySidecar:
    """Ready-to-call bridge for Fast-WAM or any RoboTwin policy candidate."""

    def __init__(
        self,
        loaded: LoadedPortableSafety,
        *,
        image_size: tuple[int, int] | None = None,
        camera_keys: tuple[str, ...] | None = None,
        host_name: str = "fast-wam",
    ) -> None:
        self.loaded = loaded
        self.adapter = RoboTwinPolicyAdapter(
            loaded.profile,
            image_size=image_size,
            camera_keys=camera_keys,
            host_name=host_name,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        image_size: tuple[int, int] | None = None,
        camera_keys: tuple[str, ...] | None = None,
        host_name: str = "fast-wam",
    ) -> "RoboTwinSafetySidecar":
        loaded = load_portable_checkpoint(checkpoint, map_location=device)
        return cls(
            loaded,
            image_size=image_size,
            camera_keys=camera_keys,
            host_name=host_name,
        )

    def assess_candidate(
        self,
        observation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        candidate: Mapping[str, Any] | TensorLike,
        **timestamp_values: Any,
    ) -> SafetyAssessment:
        adapted = self.adapter.from_policy_candidate(
            observation, candidate, **timestamp_values
        )
        return self.loaded.runtime.assess(adapted)

    def assess_candidates(
        self,
        observation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any] | TensorLike],
        **timestamp_values: Any,
    ) -> SafetyAssessment:
        adapted = self.adapter.from_policy_candidates(
            observation, candidates, **timestamp_values
        )
        return self.loaded.runtime.assess(adapted)
