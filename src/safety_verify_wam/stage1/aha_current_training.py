from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from .aha_teacher import AHAOVCRSTeacherBatch
from .ovcr_s import OVCRSActionGenerator


class AHACurrentKVTrainingAdapter(nn.Module):
    """Use dataset action flow while preserving AHA current-frame conditioning."""

    def __init__(
        self,
        *,
        student: OVCRSActionGenerator,
        action_sigma_shift: float = 5.0,
        action_noise_sampling: str = "uniform_shifted",
        action_flow_target: str = "ground_truth",
        freeze_action_expert: bool = True,
    ) -> None:
        super().__init__()
        self.student_config = student.config
        self.conditioning_source = "aha_current_kv"
        self.action_sigma_shift = float(action_sigma_shift)
        self.action_noise_sampling = str(action_noise_sampling).strip().lower()
        self.action_flow_target = str(action_flow_target).strip().lower()
        self.freeze_action_expert = bool(freeze_action_expert)
        if self.action_sigma_shift <= 0:
            raise ValueError("action_sigma_shift must be positive")
        if self.action_noise_sampling != "uniform_shifted":
            raise ValueError(
                "AHA current-KV training requires action_noise_sampling='uniform_shifted'"
            )
        if self.action_flow_target != "ground_truth":
            raise ValueError(
                "AHA current-KV training requires action_flow_target='ground_truth'"
            )
        if self.freeze_action_expert:
            for parameter in student.action_expert.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "AHACurrentKVTrainingAdapter":
        self.training = False
        return self

    def _sample_sigma(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_timesteps = int(self.student_config.num_train_timesteps)
        timestep_id = torch.randint(0, num_timesteps, (batch_size,), device=device)
        linear_sigma = 1.0 - timestep_id.to(dtype=torch.float32) / float(num_timesteps)
        shifted_sigma = (
            self.action_sigma_shift
            * linear_sigma
            / (1.0 + (self.action_sigma_shift - 1.0) * linear_sigma)
        )
        sigma = shifted_sigma.to(dtype=dtype)
        action_t = sigma * float(num_timesteps)
        return sigma, action_t

    @torch.no_grad()
    def prepare_batch(
        self,
        sample: dict[str, Any],
        targets: AHAOVCRSTeacherBatch,
    ) -> AHAOVCRSTeacherBatch:
        del sample
        # The adapter intentionally owns no parameters. Use the target device;
        # the trainer's autocast handles the frozen action expert dtype.
        device = targets.ground_truth_action.device
        dtype = targets.ground_truth_action.dtype

        ground_truth_action = targets.ground_truth_action.to(device=device, dtype=dtype)
        sigma, action_t = self._sample_sigma(
            int(ground_truth_action.shape[0]), device=device, dtype=dtype
        )
        noise = torch.randn_like(ground_truth_action)
        noisy_action = (
            ground_truth_action * (1.0 - sigma[:, None, None])
            + noise * sigma[:, None, None]
        )
        target_velocity = noise - ground_truth_action
        return replace(
            targets,
            noisy_action=noisy_action,
            action_t=action_t,
            sigma=sigma,
            teacher_velocity=target_velocity,
            reference_velocity=None,
        )
