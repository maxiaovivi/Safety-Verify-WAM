from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


def _limit_tuple(
    value: tuple[float, ...] | list[float] | None,
    *,
    action_dim: int,
    name: str,
    positive: bool = False,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    result = tuple(float(item) for item in value)
    if len(result) != action_dim:
        raise ValueError(f"{name} has {len(result)} values, expected {action_dim}")
    tensor = torch.tensor(result, dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if positive and bool((tensor <= 0).any()):
        raise ValueError(f"{name} values must be positive")
    return result


@dataclass(frozen=True)
class ActionGuardConfig:
    action_dim: int
    control_dt: float
    joint_lower: tuple[float, ...] | None = None
    joint_upper: tuple[float, ...] | None = None
    max_velocity: tuple[float, ...] | None = None
    max_acceleration: tuple[float, ...] | None = None
    joint_limit_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive")
        if self.joint_limit_margin < 0:
            raise ValueError("joint_limit_margin cannot be negative")
        for name, positive in (
            ("joint_lower", False),
            ("joint_upper", False),
            ("max_velocity", True),
            ("max_acceleration", True),
        ):
            object.__setattr__(
                self,
                name,
                _limit_tuple(
                    getattr(self, name),
                    action_dim=self.action_dim,
                    name=name,
                    positive=positive,
                ),
            )
        if self.joint_lower is not None and self.joint_upper is not None:
            lower = torch.tensor(self.joint_lower)
            upper = torch.tensor(self.joint_upper)
            if bool((lower >= upper).any()):
                raise ValueError("Every joint_lower value must be below joint_upper")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionGuardConfig":
        values = dict(payload)
        for key in (
            "joint_lower",
            "joint_upper",
            "max_velocity",
            "max_acceleration",
        ):
            if values.get(key) is not None:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class GuardResult:
    chunk_risk: torch.Tensor
    step_risk: torch.Tensor
    reason_masks: dict[str, torch.Tensor]


class DeterministicActionGuard:
    """Check explicit qpos limits, target velocity, and target acceleration."""

    def __init__(self, config: ActionGuardConfig) -> None:
        self.config = config

    @staticmethod
    def _last_state(state: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
        indices = state_mask.long().sum(dim=1) - 1
        if bool((indices < 0).any()):
            raise ValueError("Every sample requires one valid raw state")
        gather = indices.view(-1, 1, 1).expand(-1, 1, state.shape[-1])
        return state.gather(1, gather).squeeze(1)

    @staticmethod
    def _last_timestamp(
        timestamps: torch.Tensor,
        state_mask: torch.Tensor,
    ) -> torch.Tensor:
        indices = state_mask.long().sum(dim=1) - 1
        return timestamps.gather(1, indices.unsqueeze(1)).squeeze(1)

    @staticmethod
    def _joint_limit(
        value: tuple[float, ...] | None,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        return None if value is None else reference.new_tensor(value)

    def check(
        self,
        *,
        raw_state: torch.Tensor,
        raw_action: torch.Tensor,
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        state_timestamps: torch.Tensor | None = None,
        action_timestamps: torch.Tensor | None = None,
    ) -> GuardResult:
        if raw_state.ndim != 3 or raw_action.ndim != 3:
            raise ValueError("raw_state and raw_action must be [B,T,D]")
        if raw_state.shape[0] != raw_action.shape[0]:
            raise ValueError("raw_state and raw_action batch sizes differ")
        if raw_action.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"raw_action dim is {raw_action.shape[-1]}, "
                f"expected {self.config.action_dim}"
            )
        if raw_state.shape[-1] < self.config.action_dim:
            raise ValueError("raw_state has fewer qpos dimensions than raw_action")
        current = self._last_state(raw_state, state_mask)[
            :, : self.config.action_dim
        ]
        previous_targets = torch.cat(
            [current.unsqueeze(1), raw_action[:, :-1]], dim=1
        )
        if state_timestamps is None or action_timestamps is None:
            step_dt = raw_action.new_full(
                raw_action.shape[:2], self.config.control_dt
            )
        else:
            current_time = self._last_timestamp(state_timestamps, state_mask)
            previous_times = torch.cat(
                [current_time.unsqueeze(1), action_timestamps[:, :-1]], dim=1
            )
            step_dt = action_timestamps - previous_times
            if bool(((step_dt <= 0) & action_mask).any()):
                raise ValueError(
                    "Valid action timestamps must follow the latest state time"
                )
            step_dt = torch.where(
                action_mask,
                step_dt,
                step_dt.new_full((), self.config.control_dt),
            )
        velocity = (raw_action - previous_targets) / step_dt.unsqueeze(-1)
        acceleration = torch.zeros_like(velocity)
        if raw_action.shape[1] > 1:
            acceleration[:, 1:] = (
                velocity[:, 1:] - velocity[:, :-1]
            ) / step_dt[:, 1:].unsqueeze(-1)

        reason_masks: dict[str, torch.Tensor] = {}
        lower = self._joint_limit(self.config.joint_lower, raw_action)
        if lower is not None:
            reason_masks["joint_lower"] = (
                raw_action < lower + self.config.joint_limit_margin
            ).any(dim=-1)
        upper = self._joint_limit(self.config.joint_upper, raw_action)
        if upper is not None:
            reason_masks["joint_upper"] = (
                raw_action > upper - self.config.joint_limit_margin
            ).any(dim=-1)
        max_velocity = self._joint_limit(self.config.max_velocity, raw_action)
        if max_velocity is not None:
            reason_masks["velocity"] = (velocity.abs() > max_velocity).any(dim=-1)
        max_acceleration = self._joint_limit(
            self.config.max_acceleration, raw_action
        )
        if max_acceleration is not None:
            reason_masks["acceleration"] = (
                acceleration.abs() > max_acceleration
            ).any(dim=-1)

        for name, mask in tuple(reason_masks.items()):
            reason_masks[name] = mask & action_mask
        step_risk = torch.zeros_like(action_mask)
        for mask in reason_masks.values():
            step_risk |= mask
        return GuardResult(
            chunk_risk=step_risk.any(dim=1),
            step_risk=step_risk,
            reason_masks=reason_masks,
        )
