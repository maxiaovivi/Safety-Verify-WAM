from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .aha_teacher import AHAOVCRSTeacherBatch
from .efficient_adapter import (
    AHAEfficientNormalizerBridge,
    compact_video_cache,
    observation_tokens_from_condition_latent,
    share_efficient_action_expert,
)
from .ovcr_s import OVCRSActionGenerator


def _shared_batch_timestep(timestep: torch.Tensor) -> torch.Tensor:
    """Return the scalar timestep used by a scheduler for a shared batch step."""

    if timestep.numel() == 0:
        raise ValueError("A video timestep batch cannot be empty")
    flattened = timestep.reshape(-1)
    first = flattened[0]
    if not torch.equal(flattened, first.expand_as(flattened)):
        raise ValueError("Efficient video denoising requires one shared batch timestep")
    return first


class EfficientStudentTrainingAdapter(nn.Module):
    """Build deployment-matched Efficient K/V and Efficient-space flow targets.

    AHA supplies only its final action sample. The frozen Efficient-WAM model
    supplies the condition latent, imagined-future video cache, and an
    original-Efficient action prediction used for preservation regularization.
    """

    def __init__(
        self,
        *,
        student: OVCRSActionGenerator,
        deploy_config_path: str | Path,
        aha_dataset_stats_path: str | Path,
        efficient_dataset_stats_path: str | Path,
        device: str | torch.device = "cuda",
        student_dtype: torch.dtype = torch.bfloat16,
        efficient_python_root: str | Path | None = None,
        num_video_steps: int = 2,
        num_video_frames: int = 8,
        action_sigma_shift: float = 5.0,
        video_sigma_shift: float = 5.0,
    ) -> None:
        super().__init__()
        student_config = student.config
        self.student_config = student_config
        self.device_name = str(device)
        self.student_dtype = student_dtype
        self.num_video_steps = int(num_video_steps)
        self.num_video_frames = int(num_video_frames)
        if self.num_video_steps <= 0:
            raise ValueError("num_video_steps must be positive")
        if self.num_video_frames <= 0 or self.num_video_frames % 4:
            raise ValueError("num_video_frames must be a positive multiple of four")

        if efficient_python_root not in (None, "", "null"):
            python_root = str(Path(efficient_python_root).expanduser().resolve())
            if python_root not in sys.path:
                sys.path.insert(0, python_root)

        model_loader = importlib.import_module("EfficientWAM.model_loader")
        scheduler_module = importlib.import_module("EfficientWAM.third_party.wan.utils.fm")
        deploy_config = model_loader.load_deploy_config(
            str(Path(deploy_config_path).expanduser().resolve())
        )
        model_cfg = deploy_config.get("model", {})
        action_cfg = model_cfg.get("action_expert", {})
        compact_cfg = model_cfg.get("compact_wan", {})
        expected = {
            "action dim": (int(action_cfg.get("action_dim", -1)), student_config.action_dim),
            "state dim": (int(action_cfg.get("state_dim", -1)), student_config.state_dim),
            "chunk size": (
                int(action_cfg.get("chunk_size", -1)),
                student_config.action_chunk_size,
            ),
            "action hidden dim": (
                int(action_cfg.get("dim", -1)),
                student_config.action_hidden_dim,
            ),
            "action FFN dim": (
                int(action_cfg.get("ffn_dim", -1)),
                student_config.action_ffn_dim,
            ),
            "action layers": (
                int(action_cfg.get("num_layers", -1)),
                student_config.num_layers,
            ),
            "video dim": (int(compact_cfg.get("dim", -1)), student_config.video_dim),
            "video heads": (
                int(compact_cfg.get("num_heads", -1)),
                student_config.num_heads,
            ),
            "video layers": (
                int(compact_cfg.get("num_layers", -1)),
                student_config.num_layers,
            ),
        }
        mismatches = {
            name: values for name, values in expected.items() if values[0] != values[1]
        }
        if mismatches:
            raise ValueError(
                "Efficient deploy config differs from the fixed student architecture: "
                f"{mismatches}"
            )

        self.efficient_model = model_loader.build_model_from_config(
            deploy_config, device=self.device_name
        )
        self.efficient_model.eval()
        for parameter in self.efficient_model.parameters():
            parameter.requires_grad_(False)
        share_efficient_action_expert(self.efficient_model, student)

        flow_scheduler = scheduler_module.FlowMatchScheduler
        self.video_scheduler = flow_scheduler(
            shift=float(video_sigma_shift), sigma_min=0.0, extra_one_step=True
        )
        self.video_scheduler.set_timesteps(
            num_inference_steps=self.num_video_steps, training=False
        )
        self.action_scheduler = flow_scheduler(
            shift=float(action_sigma_shift), sigma_min=0.0, extra_one_step=True
        )
        self.normalizer = AHAEfficientNormalizerBridge.from_dataset_stats(
            aha_dataset_stats_path,
            efficient_dataset_stats_path,
        )

    def train(self, mode: bool = True) -> "EfficientStudentTrainingAdapter":
        super().train(False)
        self.efficient_model.eval()
        return self

    def _select_chunk_frame(
        self,
        video: torch.Tensor,
        chunk_index: torch.Tensor,
        action_horizon: int,
    ) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(
                "sample['video'] must be [B,3,T,H,W], "
                f"got {tuple(video.shape)}"
            )
        if video.shape[0] != chunk_index.shape[0] or video.shape[2] <= 1:
            raise ValueError("Video batch/time dimensions do not match chunk indices")
        chunk_starts = chunk_index * self.student_config.action_chunk_size
        frame_index = torch.floor(
            chunk_starts.float()
            * float(video.shape[2] - 1)
            / float(action_horizon)
        ).long()
        frame_index = frame_index.clamp(max=video.shape[2] - 1)
        gather_index = frame_index.view(-1, 1, 1, 1, 1).expand(
            -1, video.shape[1], 1, video.shape[3], video.shape[4]
        )
        return torch.gather(video, dim=2, index=gather_index)

    def _initialize_video_latent(
        self, current_frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_model = self.efficient_model.compact_wan.video_model
        video_dtype = video_model.precision
        current_frame = current_frame.to(
            device=self.device_name, dtype=video_dtype
        )
        condition_latent = self.efficient_model.compact_wan.encode_video(current_frame)
        batch, channels, _, condition_height, condition_width = condition_latent.shape
        if self.efficient_model.compact_wan.is_multiscale:
            future_size = self.efficient_model.compact_wan.config.future_video_size
            if future_size is None:
                raise RuntimeError("Multiscale Efficient-WAM has no future_video_size")
            future_height, future_width = (int(value) // 16 for value in future_size)
            future_latent = torch.randn(
                (
                    batch,
                    channels,
                    self.num_video_frames // 4,
                    future_height,
                    future_width,
                ),
                device=self.device_name,
                dtype=video_dtype,
            )
            return future_latent, condition_latent

        video_latent = torch.randn(
            (
                batch,
                channels,
                1 + self.num_video_frames // 4,
                condition_height,
                condition_width,
            ),
            device=self.device_name,
            dtype=video_dtype,
        )
        video_latent[:, :, :1] = condition_latent
        return video_latent, condition_latent

    @torch.no_grad()
    def _build_video_cache(
        self,
        *,
        current_frame: torch.Tensor,
        text_context: torch.Tensor,
        noisy_action: torch.Tensor,
        action_t: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[dict[str, torch.Tensor], ...],
        torch.Tensor,
    ]:
        # ``student.train()`` also changes the shared action expert's mode.
        # Keep the frozen video branch in evaluation mode without resetting
        # that shared module immediately before the student forward pass.
        self.efficient_model.compact_wan.eval()
        video_latent, condition_latent = self._initialize_video_latent(current_frame)
        batch_size = int(current_frame.shape[0])
        if text_context.ndim != 3 or text_context.shape[0] != batch_size:
            raise ValueError(
                "sample['context'] must be [B,L,D], "
                f"got {tuple(text_context.shape)}"
            )
        text_embeddings = [text_context[index] for index in range(batch_size)]
        efficient_action_parameter = next(
            self.efficient_model.action_expert.parameters()
        )
        efficient_action_device = efficient_action_parameter.device
        efficient_action_dtype = efficient_action_parameter.dtype
        noisy_action = noisy_action.to(
            device=efficient_action_device, dtype=efficient_action_dtype
        )
        action_t = action_t.to(
            device=efficient_action_device, dtype=efficient_action_dtype
        )
        initial_state = initial_state.to(
            device=efficient_action_device, dtype=efficient_action_dtype
        )
        video_timesteps = self.video_scheduler.timesteps.to(
            device=self.device_name,
            dtype=self.efficient_model.compact_wan.video_model.precision,
        )
        video_cache: Mapping[str, Any] | None = None
        original_velocity: torch.Tensor | None = None
        for video_step in range(self.num_video_steps):
            current_video_t = video_timesteps[video_step].expand(batch_size)
            batch: dict[str, Any] = {
                "video_t": current_video_t,
                "initial_state": initial_state,
                "noisy_actions": noisy_action,
                "action_t": action_t,
                "text_embeddings": text_embeddings,
                "return_video_cache": True,
            }
            if self.efficient_model.compact_wan.is_multiscale:
                batch["condition_latent"] = condition_latent
                batch["future_latent"] = video_latent
            else:
                batch["video_latent"] = video_latent
            outputs = self.efficient_model(batch)
            video_cache = outputs.get("video_cache")
            original_velocity = outputs.get("action_pred")
            if video_cache is None or original_velocity is None:
                raise RuntimeError("Efficient-WAM did not return video cache/action prediction")
            if video_step + 1 < self.num_video_steps:
                video_latent = self.video_scheduler.step(
                    outputs["video_pred"],
                    _shared_batch_timestep(current_video_t),
                    video_latent,
                )
                if not self.efficient_model.compact_wan.is_multiscale:
                    video_latent[:, :, :1] = condition_latent

        assert video_cache is not None and original_velocity is not None
        compact_cache = compact_video_cache(
            video_cache,
            expected_layers=self.student_config.num_layers,
            expected_dim=self.student_config.video_dim,
            condition_only=False,
        )
        observation_tokens = observation_tokens_from_condition_latent(
            condition_latent
        )
        return observation_tokens, compact_cache, original_velocity

    @torch.no_grad()
    def prepare_batch(
        self,
        sample: dict[str, Any],
        targets: AHAOVCRSTeacherBatch,
    ) -> AHAOVCRSTeacherBatch:
        teacher_action = self.normalizer.action_aha_to_efficient(
            targets.teacher_action.float()
        )
        ground_truth_action = self.normalizer.action_aha_to_efficient(
            targets.ground_truth_action.float()
        )
        initial_state = self.normalizer.state_aha_to_efficient(
            targets.initial_state.float()
        )
        target_device = next(self.efficient_model.action_expert.parameters()).device
        target_dtype = self.student_dtype
        teacher_action = teacher_action.to(device=target_device, dtype=target_dtype)
        ground_truth_action = ground_truth_action.to(
            device=target_device, dtype=target_dtype
        )
        initial_state = initial_state.to(device=target_device, dtype=target_dtype)
        sigma = targets.sigma.to(device=target_device, dtype=target_dtype).clamp(0, 1)
        noise = torch.randn_like(teacher_action)
        noisy_action = (
            teacher_action * (1 - sigma[:, None, None])
            + noise * sigma[:, None, None]
        )
        target_velocity = noise - teacher_action
        action_t = sigma * float(self.student_config.num_train_timesteps)

        sample_action = sample.get("action")
        video = sample.get("video")
        context = sample.get("context")
        if not isinstance(sample_action, torch.Tensor) or sample_action.ndim != 3:
            raise ValueError("sample['action'] must be [B,T,D]")
        if not isinstance(video, torch.Tensor) or not isinstance(context, torch.Tensor):
            raise ValueError("Efficient training needs sample video and text context tensors")
        current_frame = self._select_chunk_frame(
            video,
            targets.chunk_index.to(device=video.device),
            action_horizon=int(sample_action.shape[1]),
        )
        observation_tokens, video_kv_cache, reference_velocity = (
            self._build_video_cache(
                current_frame=current_frame,
                text_context=context,
                noisy_action=noisy_action,
                action_t=action_t,
                initial_state=initial_state,
            )
        )
        observation_tokens = observation_tokens.to(
            device=target_device, dtype=target_dtype
        )
        video_kv_cache = tuple(
            {
                "k": layer["k"].to(device=target_device, dtype=target_dtype),
                "v": layer["v"].to(device=target_device, dtype=target_dtype),
            }
            for layer in video_kv_cache
        )
        observation_mask = torch.ones(
            observation_tokens.shape[:2],
            device=target_device,
            dtype=torch.bool,
        )
        return replace(
            targets,
            noisy_action=noisy_action,
            action_t=action_t,
            sigma=sigma,
            teacher_velocity=target_velocity,
            teacher_action=teacher_action,
            ground_truth_action=ground_truth_action,
            initial_state=initial_state,
            observation_tokens=observation_tokens,
            observation_mask=observation_mask,
            video_kv_cache=video_kv_cache,
            reference_velocity=reference_velocity.to(
                device=target_device, dtype=target_dtype
            ),
        )
