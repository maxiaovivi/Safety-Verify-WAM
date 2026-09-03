from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar

import torch

from .contracts import AdaptedSafetyInput, RobotProfile
from .guard import DeterministicActionGuard, GuardResult
from .model import PortableSafetyCore


@dataclass(frozen=True)
class SafetyThresholds:
    chunk_risk: float = 0.5
    step_risk: float = 0.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"{name} threshold must be in (0, 1)")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SafetyThresholds":
        return cls(**{key: float(value) for key, value in payload.items()})


@dataclass(frozen=True)
class SafetyAssessment:
    class_logits: torch.Tensor
    class_probabilities: torch.Tensor
    learned_risk_probability: torch.Tensor
    step_class_logits: torch.Tensor
    step_class_probabilities: torch.Tensor
    learned_step_risk_probability: torch.Tensor
    learned_requires_intervention: torch.Tensor
    guard_requires_intervention: torch.Tensor
    requires_intervention: torch.Tensor
    step_requires_intervention: torch.Tensor
    first_risk_step: torch.Tensor
    guard_reason_masks: dict[str, torch.Tensor]
    host_name: str


class PortableSafetyRuntime:
    """Run a learned safety core and explicit physical checks as one sidecar."""

    def __init__(
        self,
        model: PortableSafetyCore,
        profile: RobotProfile,
        *,
        thresholds: SafetyThresholds | None = None,
        guard: DeterministicActionGuard | None = None,
    ) -> None:
        if model.config.state_dim != profile.state_dim:
            raise ValueError("Model state_dim differs from robot profile")
        if model.config.action_dim != profile.action_dim:
            raise ValueError("Model action_dim differs from robot profile")
        if len(profile.camera_names) > model.config.max_views:
            raise ValueError("Robot profile has more cameras than model max_views")
        if guard is not None:
            if guard.config.action_dim != profile.action_dim:
                raise ValueError("Guard action_dim differs from robot profile")
            if abs(guard.config.control_dt - profile.control_dt) > 1e-12:
                raise ValueError("Guard control_dt differs from robot profile")
        self.model = model
        self.profile = profile
        self.thresholds = thresholds or SafetyThresholds()
        self.guard = guard

    @torch.inference_mode()
    def assess(self, value: AdaptedSafetyInput) -> SafetyAssessment:
        value.validate(self.profile)
        parameter = next(self.model.parameters())
        runtime_value = value.to(parameter.device, dtype=parameter.dtype)
        was_training = self.model.training
        self.model.eval()
        outputs = self.model(runtime_value.batch)
        if was_training:
            self.model.train()

        learned_chunk = (
            outputs["risk_probability"] >= self.thresholds.chunk_risk
        )
        learned_step = (
            outputs["step_risk_probability"] >= self.thresholds.step_risk
        ) & runtime_value.batch.action_mask
        learned_requires = learned_chunk | learned_step.any(dim=1)

        if self.guard is None:
            guard_result = GuardResult(
                chunk_risk=torch.zeros_like(learned_requires),
                step_risk=torch.zeros_like(learned_step),
                reason_masks={},
            )
        else:
            if runtime_value.raw_state is None or runtime_value.raw_action is None:
                raise RuntimeError(
                    "Physical guard is enabled but the adapter did not provide raw "
                    "state and action values"
                )
            guard_result = self.guard.check(
                raw_state=runtime_value.raw_state,
                raw_action=runtime_value.raw_action,
                state_mask=runtime_value.batch.state_mask,
                action_mask=runtime_value.batch.action_mask,
                state_timestamps=runtime_value.batch.state_timestamps.float(),
                action_timestamps=runtime_value.batch.action_timestamps.float(),
            )
        step_requires = learned_step | guard_result.step_risk
        requires = learned_requires | guard_result.chunk_risk
        first_risk_step = torch.full(
            (requires.shape[0],),
            -1,
            dtype=torch.long,
            device=requires.device,
        )
        has_risk_step = step_requires.any(dim=1)
        if bool(has_risk_step.any()):
            first_risk_step[has_risk_step] = step_requires[
                has_risk_step
            ].to(torch.int64).argmax(dim=1)
        return SafetyAssessment(
            class_logits=outputs["class_logits"],
            class_probabilities=outputs["class_probabilities"],
            learned_risk_probability=outputs["risk_probability"],
            step_class_logits=outputs["step_class_logits"],
            step_class_probabilities=outputs["step_class_probabilities"],
            learned_step_risk_probability=outputs["step_risk_probability"],
            learned_requires_intervention=learned_requires,
            guard_requires_intervention=guard_result.chunk_risk,
            requires_intervention=requires,
            step_requires_intervention=step_requires,
            first_risk_step=first_risk_step,
            guard_reason_masks=guard_result.reason_masks,
            host_name=value.host_name,
        )


ActionT = TypeVar("ActionT")


def choose_action(
    candidate_action: ActionT,
    assessment: SafetyAssessment,
    fallback: ActionT | Callable[[], ActionT],
    *,
    batch_index: int = 0,
) -> ActionT:
    """Return the candidate object unchanged when the sidecar accepts it."""

    if assessment.requires_intervention.ndim != 1:
        raise ValueError("requires_intervention must be a batch vector")
    if not 0 <= batch_index < assessment.requires_intervention.shape[0]:
        raise IndexError("batch_index is outside the assessment batch")
    if not bool(assessment.requires_intervention[batch_index].item()):
        return candidate_action
    return fallback() if callable(fallback) else fallback
