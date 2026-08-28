"""RoboTwin evaluation using AHA video prefill and the compact aligned action path."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any, Mapping, Sequence

import torch

logger = logging.getLogger(__name__)


def _required_path(args: Mapping[str, Any], key: str) -> Path:
    value = args.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"`{key}` is required")
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"`{key}` does not exist: {path}")
    return path


def _load_base_policy(aha_root: Path) -> ModuleType:
    policy_path = (
        aha_root / "experiments" / "robotwin" / "ahawam_policy" / "deploy_policy.py"
    )
    if not policy_path.is_file():
        raise FileNotFoundError(f"AHA deploy policy not found: {policy_path}")
    module_name = "_aha_ovcrs_official_base_policy"
    spec = importlib.util.spec_from_file_location(module_name, policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import AHA deploy policy: {policy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _compact_current_frame_cache(
    *,
    model: torch.nn.Module,
    student: torch.nn.Module,
    inference_state: Mapping[str, Any],
    teacher_layer_mapping: Sequence[int],
) -> tuple[dict[str, torch.Tensor], ...]:
    from safety_verify_wam.stage1.aha_teacher import _select_heads

    video_cache = inference_state["video_kv_cache"]
    token_count = int(inference_state["video_tokens_per_frame"])
    editor = model.mot.chunk_kv_cache_editor
    config = student.config
    if len(teacher_layer_mapping) != int(config.num_layers):
        raise ValueError("Checkpoint layer mapping differs from the compact student")
    parameter = next(student.parameters())
    compact: list[dict[str, torch.Tensor]] = []
    for teacher_layer in teacher_layer_mapping:
        layer_cache = video_cache[int(teacher_layer) - 1]
        compact.append(
            {
                "k": _select_heads(
                    layer_cache["k"][:, :token_count],
                    teacher_heads=int(editor.num_heads),
                    student_heads=int(config.num_heads),
                    head_dim=int(editor.head_dim),
                )
                .to(device=parameter.device, dtype=parameter.dtype)
                .contiguous(),
                "v": _select_heads(
                    layer_cache["v"][:, :token_count],
                    teacher_heads=int(editor.num_heads),
                    student_heads=int(config.num_heads),
                    head_dim=int(editor.head_dim),
                )
                .to(device=parameter.device, dtype=parameter.dtype)
                .contiguous(),
            }
        )
    return tuple(compact)


@torch.inference_mode()
def _hybrid_infer_action_chunk(
    model: torch.nn.Module,
    *,
    inference_state: dict[str, Any],
    chunk_obs_image: torch.Tensor,
    chunk_proprio: torch.Tensor | None = None,
    chunk_index: int,
    num_inference_steps: int = 10,
    sigma_shift: float | None = None,
    tiled: bool = False,
) -> dict[str, Any]:
    student = model._ovcrs_student
    config = student.config
    if config.action_architecture != "aha_aligned":
        raise ValueError("AHA hybrid evaluation requires an aha_aligned checkpoint")
    if chunk_proprio is None:
        raise ValueError("Current normalized robot state is required")

    chunk_start = int(chunk_index) * int(config.action_chunk_size)
    chunk_end = chunk_start + int(config.action_chunk_size)
    current_latents = inference_state["start_latents"]
    if chunk_end > current_latents.shape[1]:
        raise IndexError(f"Chunk {chunk_index} exceeds the action horizon")

    captured_observation: list[torch.Tensor] = []

    def capture_pre_projection(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured_observation.append(inputs[0].detach())

    handle = model.action_obs_visual_proj.register_forward_pre_hook(
        capture_pre_projection
    )
    try:
        model._prepare_inference_chunk_conditioning(
            chunk_obs_image=chunk_obs_image,
            chunk_proprio=chunk_proprio,
            chunk_index=int(chunk_index),
            inference_state=inference_state,
            tiled=bool(tiled),
        )
    finally:
        handle.remove()
    if len(captured_observation) != 1:
        raise RuntimeError("AHA did not expose exactly one current observation")

    parameter = next(student.parameters())
    observation_tokens = captured_observation[0].to(
        device=parameter.device, dtype=parameter.dtype
    )
    observation_mask = torch.ones(
        observation_tokens.shape[:2],
        device=observation_tokens.device,
        dtype=torch.bool,
    )
    video_cache = _compact_current_frame_cache(
        model=model,
        student=student,
        inference_state=inference_state,
        teacher_layer_mapping=model._ovcrs_teacher_layer_mapping,
    )
    initial_state = chunk_proprio.to(
        device=parameter.device, dtype=parameter.dtype
    )
    action_context = inference_state.get("context")
    action_context_mask = inference_state.get("context_mask")
    if not isinstance(action_context, torch.Tensor):
        raise RuntimeError("AHA inference state has no task-text context")
    action_context = action_context.to(
        device=parameter.device, dtype=parameter.dtype
    )
    if isinstance(action_context_mask, torch.Tensor):
        action_context_mask = action_context_mask.to(
            device=parameter.device, dtype=torch.bool
        )
    initial_noise = current_latents[:, chunk_start:chunk_end].to(
        device=parameter.device, dtype=parameter.dtype
    )
    resolved_shift = sigma_shift
    if resolved_shift is None:
        resolved_shift = getattr(model.infer_action_scheduler, "shift", 5.0)
    action_chunk = student.generate(
        observation_tokens=observation_tokens,
        video_kv_cache=video_cache,
        initial_state=initial_state,
        observation_mask=observation_mask,
        action_context=action_context,
        action_context_mask=action_context_mask,
        num_steps=int(num_inference_steps),
        flow_shift=float(resolved_shift),
        initial_noise=initial_noise,
    )
    if not torch.isfinite(action_chunk).all():
        raise FloatingPointError("Compact aligned action expert produced NaN or Inf")
    current_latents[:, chunk_start:chunk_end] = action_chunk.to(
        device=current_latents.device, dtype=current_latents.dtype
    )
    inference_state["start_latents"] = current_latents
    return {
        "action_chunk": action_chunk[0].detach().cpu().float(),
        "final_latents_chunk": action_chunk.detach().clone(),
        "chunk_index": int(chunk_index),
        "inference_state": inference_state,
    }


def get_model(usr_args: dict[str, Any]):
    aha_root = _required_path(usr_args, "aha_wam_root")
    safety_root = _required_path(usr_args, "safety_verify_wam_root")
    student_checkpoint = _required_path(usr_args, "ovcrs_checkpoint")
    safety_src = str(safety_root / "src")
    if safety_src not in sys.path:
        sys.path.insert(0, safety_src)

    base = _load_base_policy(aha_root)
    policy = base.get_model(usr_args)
    from safety_verify_wam.stage1.efficient_adapter import load_ovcrs_student

    student, payload = load_ovcrs_student(
        student_checkpoint,
        device=policy.model.device,
        dtype=policy.model.torch_dtype,
    )
    if student.config.action_architecture != "aha_aligned":
        raise ValueError("Checkpoint does not contain the aligned AHA action structure")
    mapping = tuple(int(layer) for layer in payload["teacher_layer_mapping"])
    policy.model._ovcrs_student = student
    policy.model._ovcrs_teacher_layer_mapping = mapping
    policy.model.infer_action_chunk = MethodType(
        _hybrid_infer_action_chunk, policy.model
    )
    logger.info(
        "Installed aligned compact AHA action path | checkpoint=%s | step=%s",
        student_checkpoint,
        payload.get("step"),
    )
    return policy


def encode_obs(observation: Any) -> Any:
    return observation


def eval(task_env: Any, model: Any, observation: Any) -> None:
    model.step(task_env, observation)


def reset_model(model: Any) -> None:
    model.reset()
