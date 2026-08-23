from __future__ import annotations

import logging
import time
from typing import Any, Dict, Mapping

import numpy as np
import torch

from EfficientWAM.model_loader import build_runtime_from_config, normalize_deploy_config
from EfficientWAM.preprocess import preprocess_robotwin_observation
from EfficientWAM.runner import EfficientWAMRunner, _new_forward_timer

from safety_verify_wam.stage1.efficient_adapter import (
    load_ovcrs_student,
    prepare_ovcrs_conditioning,
    share_efficient_action_expert,
)

logger = logging.getLogger(__name__)


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported OVCR-S precision: {name}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


class OVCRSEfficientWAMRunner(EfficientWAMRunner):
    """Use Efficient-WAM video K/V and the distilled OVCR-S action generator."""

    def __init__(
        self,
        *,
        runtime: Any,
        student: torch.nn.Module,
        observation_downsample_factor: int = 1,
        condition_only_video_cache: bool = False,
    ) -> None:
        self.ovcrs_student = student
        self.observation_downsample_factor = int(observation_downsample_factor)
        self.condition_only_video_cache = bool(condition_only_video_cache)
        super().__init__(runtime=runtime)
        if self.observation_downsample_factor <= 0:
            raise ValueError("observation_downsample_factor must be positive")
        if int(self.runtime.chunk_size) != int(self.ovcrs_student.config.action_chunk_size):
            raise ValueError(
                "Efficient-WAM and OVCR-S chunk sizes differ: "
                f"{self.runtime.chunk_size} vs {self.ovcrs_student.config.action_chunk_size}"
            )
        if int(self.model.config.action_dim) != int(self.ovcrs_student.config.action_dim):
            raise ValueError(
                "Efficient-WAM and OVCR-S action dims differ: "
                f"{self.model.config.action_dim} vs {self.ovcrs_student.config.action_dim}"
            )

    def _sample_action_chunk(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        bool,
        float,
        float,
        float,
        int,
        Dict[str, float],
        Dict[str, int],
        list[Dict[str, Any]],
    ]:
        video_dtype = self.model.compact_wan.video_model.precision
        student_parameter = next(self.ovcrs_student.parameters())
        action_dtype = student_parameter.dtype
        first_frame = first_frame.to(self.device, dtype=video_dtype)

        efficient_action_dtype = next(self.model.action_expert.parameters()).dtype
        normalized_state = self.runtime.action_normalizer.normalize(
            state.to(self.device, dtype=efficient_action_dtype)
        )
        text_embeddings, t5_encode_ms = self._get_text_embeddings()
        video_latent, condition_latent = self._initialize_video_latent(first_frame)
        noisy_actions = torch.randn(
            (1, self.runtime.chunk_size, self.ovcrs_student.config.action_dim),
            device=student_parameter.device,
            dtype=action_dtype,
        )

        action_timesteps = self.action_scheduler.timesteps.to(
            device=student_parameter.device,
            dtype=action_dtype,
        )
        video_timesteps = self.video_scheduler.timesteps.to(
            device=self.device,
            dtype=video_dtype,
        )
        video_refresh_schedule = self._video_refresh_schedule()
        if hasattr(self.model, "reset_teacache"):
            self.model.reset_teacache(num_steps=max(1, len(video_refresh_schedule)))

        video_cache: Mapping[str, Any] | None = None
        student_conditioning: Mapping[str, Any] | None = None
        last_video_step_idx = 0
        previous_video_pred_for_control: torch.Tensor | None = None
        video_branch_stopped = False
        similarity_events: list[Dict[str, Any]] = [{"kind": "chunk"}]
        vgm_forward_timer = _new_forward_timer()

        self._cuda_synchronize()
        world_start = time.perf_counter()
        with torch.inference_mode():
            for step_idx in range(self.runtime.num_inference_steps):
                current_action_t = action_timesteps[step_idx].expand(1)
                video_step_idx = video_refresh_schedule.get(step_idx)
                do_video_refresh = video_step_idx is not None and not video_branch_stopped

                if do_video_refresh or video_cache is None:
                    if video_step_idx is None:
                        video_step_idx = last_video_step_idx
                    last_video_step_idx = int(video_step_idx)
                    current_video_t = video_timesteps[video_step_idx].expand(1)
                    batch: Dict[str, Any] = {
                        "video_t": current_video_t,
                        "initial_state": normalized_state,
                        "noisy_actions": noisy_actions.to(dtype=efficient_action_dtype),
                        "action_t": current_action_t.to(dtype=efficient_action_dtype),
                        "text_embeddings": text_embeddings,
                        "return_video_cache": True,
                    }
                    if self.model.compact_wan.is_multiscale:
                        batch["condition_latent"] = condition_latent
                        batch["future_latent"] = video_latent
                    else:
                        batch["video_latent"] = video_latent

                    outputs = self._timed_vgm_forward(batch, vgm_forward_timer)
                    video_cache = outputs.get("video_cache")
                    if not isinstance(video_cache, Mapping):
                        raise RuntimeError("Efficient-WAM did not return the required video K/V cache")
                    student_conditioning = prepare_ovcrs_conditioning(
                        self.ovcrs_student,
                        condition_latent=condition_latent,
                        video_cache=video_cache,
                        observation_downsample_factor=self.observation_downsample_factor,
                        condition_only=self.condition_only_video_cache,
                    )

                    current_video_pred = self._video_velocity_for_similarity(
                        outputs["video_pred"]
                    )
                    if (
                        self.runtime.video_stop_cosine_threshold is not None
                        and previous_video_pred_for_control is not None
                    ):
                        video_cos = self._velocity_cosine(
                            current_video_pred,
                            previous_video_pred_for_control,
                        )
                        if video_cos >= self.runtime.video_stop_cosine_threshold:
                            video_branch_stopped = True
                    previous_video_pred_for_control = current_video_pred.detach()
                    similarity_events.append(
                        {
                            "kind": "video",
                            "pred": outputs["video_pred"].detach(),
                            "action_step": step_idx,
                        }
                    )
                    video_latent = self.video_scheduler.step(
                        outputs["video_pred"],
                        current_video_t,
                        video_latent,
                    )
                    if not self.model.compact_wan.is_multiscale:
                        video_latent[:, :, 0:1] = condition_latent

                if student_conditioning is None:
                    raise RuntimeError("OVCR-S conditioning was not initialized")
                action_pred = self.ovcrs_student.predict_velocity(
                    noisy_actions,
                    current_action_t,
                    normalized_state.to(
                        device=student_parameter.device,
                        dtype=student_parameter.dtype,
                    ),
                    student_conditioning,
                    return_trace=False,
                )["action_velocity"]
                similarity_events.append(
                    {
                        "kind": "action",
                        "pred": action_pred.detach(),
                        "action_step": step_idx,
                        "is_cache_step": not do_video_refresh,
                        "cache_age": step_idx - last_video_step_idx,
                    }
                )
                noisy_actions = self.action_scheduler.step(
                    action_pred,
                    current_action_t,
                    noisy_actions,
                )

        self._cuda_synchronize()
        world_model_ms = (time.perf_counter() - world_start) * 1000.0
        (
            vgm_forward_ms_sum,
            vgm_forward_calls,
            profile_ms_sums,
            profile_calls,
        ) = self._finish_forward_timer(vgm_forward_timer)
        # RoboTwin converts the returned tensor to a NumPy array. NumPy has no
        # bfloat16 dtype, so expose physical actions in float32 at this API edge.
        actions = self.runtime.action_normalizer.denormalize(noisy_actions).float()
        predicted_video_has_condition_frame = not self.model.compact_wan.is_multiscale
        return (
            actions,
            video_latent,
            predicted_video_has_condition_frame,
            t5_encode_ms,
            world_model_ms,
            vgm_forward_ms_sum,
            vgm_forward_calls,
            profile_ms_sums,
            profile_calls,
            similarity_events,
        )


def get_model(usr_args: Dict[str, Any]) -> OVCRSEfficientWAMRunner:
    device = str(usr_args.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OVCR-S RoboTwin task evaluation")

    seed = int(usr_args.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = normalize_deploy_config(dict(usr_args))
    runtime = build_runtime_from_config(config, device=device)
    checkpoint_path = usr_args.get("ovcrs_checkpoint")
    if not checkpoint_path:
        raise ValueError("ovcrs_checkpoint is required")
    student, payload = load_ovcrs_student(
        checkpoint_path,
        device=device,
        dtype=_dtype_from_name(str(usr_args.get("ovcrs_precision", "bf16"))),
    )
    share_efficient_action_expert(runtime.model, student)
    runner = OVCRSEfficientWAMRunner(
        runtime=runtime,
        student=student,
        observation_downsample_factor=int(
            usr_args.get("observation_downsample_factor", 1)
        ),
        condition_only_video_cache=_as_bool(
            usr_args.get("condition_only_video_cache", False)
        ),
    )
    logger.info(
        "Loaded OVCR-S task policy | checkpoint=%s | step=%s | efficient_checkpoint=%s",
        checkpoint_path,
        payload.get("step"),
        config.get("checkpoint_path"),
    )
    return runner


def eval(task_env: Any, model: OVCRSEfficientWAMRunner, observation: Dict[str, Any]) -> None:
    if observation is None:
        raise ValueError("OVCR-S policy requires a fresh observation for each action chunk")
    processed = preprocess_robotwin_observation(
        observation,
        target_size=model.runtime.video_size,
    )
    model.set_instruction(task_env.get_instruction())
    model.set_prediction_video_context(
        root=getattr(task_env, "eval_video_path", None),
        episode_idx=getattr(task_env, "test_num", None),
    )
    actions = model.step(processed)
    for action in actions:
        task_env.take_action(action, action_type="qpos")


def reset_model(model: OVCRSEfficientWAMRunner) -> None:
    model.print_episode_latency_summary(episode_idx=model.predicted_video_episode_idx)
    model.print_episode_similarity_summary()
    model.reset()
