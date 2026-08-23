from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .ovcr_s import OVCRSConfig


@dataclass(frozen=True)
class AHAOVCRSTeacherBatch:
    """One compact action chunk and its frozen AHA distillation targets."""

    noisy_action: torch.Tensor
    action_t: torch.Tensor
    sigma: torch.Tensor
    teacher_velocity: torch.Tensor
    teacher_action: torch.Tensor
    ground_truth_action: torch.Tensor
    observation_tokens: torch.Tensor
    observation_mask: torch.Tensor
    video_kv_cache: tuple[dict[str, torch.Tensor], ...]
    teacher_queries: torch.Tensor
    teacher_editor_trace: dict[int, dict[str, torch.Tensor]]
    teacher_action_responses: dict[int, torch.Tensor]
    action_is_pad: torch.Tensor | None
    chunk_index: torch.Tensor
    anchor_step: torch.Tensor


def _evenly_spaced_indices(total: int, keep: int, device: torch.device) -> torch.Tensor:
    if keep > total:
        raise ValueError(f"Cannot select {keep} heads from {total}")
    if keep == total:
        return torch.arange(total, device=device, dtype=torch.long)
    indices = torch.round(torch.linspace(0, total - 1, keep, device=device)).long()
    if torch.unique(indices).numel() != keep:
        raise RuntimeError("Even head selection produced duplicate indices")
    return indices


def _select_heads(
    tensor: torch.Tensor,
    *,
    teacher_heads: int,
    student_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if tensor.shape[-1] != teacher_heads * head_dim:
        raise ValueError(
            f"Teacher K/V dim {tensor.shape[-1]} does not match "
            f"{teacher_heads}x{head_dim}"
        )
    head_indices = _evenly_spaced_indices(
        teacher_heads, student_heads, tensor.device
    )
    selected = tensor.reshape(*tensor.shape[:-1], teacher_heads, head_dim)
    selected = selected.index_select(-2, head_indices)
    return selected.flatten(-2)


def _gather_chunk_tokens(
    tensor: torch.Tensor,
    chunk_index: torch.Tensor,
) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError("Chunk tensor must include batch and chunk dimensions")
    if tensor.shape[0] != chunk_index.shape[0]:
        raise ValueError("Chunk index batch size differs from tensor batch size")
    view_shape = (tensor.shape[0], 1) + (1,) * (tensor.ndim - 2)
    gather_index = chunk_index.view(view_shape).expand(
        tensor.shape[0], 1, *tensor.shape[2:]
    )
    return torch.gather(tensor, 1, gather_index)


def _gather_action_chunk(
    tensor: torch.Tensor,
    chunk_index: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[1] % chunk_size:
        raise ValueError("Action tensor horizon must be divisible by chunk_size")
    chunks = tensor.reshape(
        tensor.shape[0], tensor.shape[1] // chunk_size, chunk_size, *tensor.shape[2:]
    )
    return _gather_chunk_tokens(chunks, chunk_index)[:, 0]


class AHAOVCRTeacherAdapter(nn.Module):
    """Run AHA online and expose compact, structured-sliced OVCR targets.

    The adapter relies on the released AHA module boundaries instead of copying
    teacher weights. It temporarily observes the query encoder and K/V editor
    during the frozen rollout, then removes all hooks before returning.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student_config: OVCRSConfig,
        *,
        teacher_layer_mapping: Sequence[int] = (
            1,
            2,
            4,
            6,
            8,
            11,
            14,
            17,
            20,
            23,
            26,
            30,
        ),
        rollout_steps: int = 16,
        capture_steps: Sequence[int] = (0, 1, 2, 4, 8, 12, 16),
        sigma_shift: float | None = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student_config = student_config
        self.teacher_layer_mapping = tuple(int(layer) for layer in teacher_layer_mapping)
        self.rollout_steps = int(rollout_steps)
        self.capture_steps = tuple(sorted({int(step) for step in capture_steps}))
        self.sigma_shift = sigma_shift
        if len(self.teacher_layer_mapping) != student_config.num_layers:
            raise ValueError(
                "teacher_layer_mapping must contain one teacher layer per student layer"
            )
        if self.capture_steps[0] != 0 or self.capture_steps[-1] != self.rollout_steps:
            raise ValueError("capture_steps must start at 0 and end at rollout_steps")
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        self._validate_teacher_contract()

    @property
    def raw_model(self) -> nn.Module:
        teacher: Any = self.teacher
        while hasattr(teacher, "module"):
            teacher = teacher.module
        model = getattr(teacher, "model", teacher)
        while hasattr(model, "module"):
            model = model.module
        return model

    def train(self, mode: bool = True) -> "AHAOVCRTeacherAdapter":
        super().train(False)
        self.teacher.eval()
        return self

    def _validate_teacher_contract(self) -> None:
        model = self.raw_model
        required = (
            "action_obs_visual_proj",
            "chunk_obs_query_encoder",
            "mot",
            "action_chunk_size",
        )
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise TypeError(f"AHA teacher is missing required attributes: {missing}")
        editor = getattr(model.mot, "chunk_kv_cache_editor", None)
        if editor is None:
            raise ValueError("AHA teacher has no configured chunk K/V cache editor")
        if int(model.action_chunk_size) != self.student_config.action_chunk_size:
            raise ValueError(
                "AHA and OVCR-S action chunk sizes differ: "
                f"{model.action_chunk_size} vs {self.student_config.action_chunk_size}"
            )
        if int(editor.head_dim) != self.student_config.head_dim:
            raise ValueError(
                "AHA and OVCR-S attention head dimensions must match for structured slicing"
            )
        if max(self.teacher_layer_mapping) > int(editor.num_layers):
            raise ValueError("teacher_layer_mapping exceeds the AHA layer count")

    def _rollout(self, sample: dict[str, Any], tiled: bool) -> dict[str, Any]:
        teacher: Any = self.teacher
        while hasattr(teacher, "module"):
            teacher = teacher.module
        if hasattr(teacher, "rollout_action_latent_states"):
            return teacher.rollout_action_latent_states(
                sample=sample,
                num_inference_steps=self.rollout_steps,
                sigma_shift=self.sigma_shift,
                capture_step_indices=self.capture_steps,
                tiled=tiled,
            )
        return self.raw_model.rollout_action_prior_only(
            sample=sample,
            num_steps=self.rollout_steps,
            capture_indices=self.capture_steps,
            sigma_shift=self.sigma_shift,
            tiled=tiled,
        )

    @torch.no_grad()
    def _rollout_with_ovcr_trace(
        self,
        sample: dict[str, Any],
        tiled: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model = self.raw_model
        editor = model.mot.chunk_kv_cache_editor
        teacher_heads = int(editor.num_heads)
        head_dim = int(editor.head_dim)
        mapped_zero_based = {
            teacher_layer - 1: student_layer
            for student_layer, teacher_layer in enumerate(
                self.teacher_layer_mapping, start=1
            )
        }
        trace_layers = set(self.student_config.distill_layers)
        captured: dict[str, Any] = {
            "observation_inputs": [],
            "queries": [],
            "cache": {},
            "editor_trace": {},
        }

        def observation_pre_hook(
            _module: nn.Module, args: tuple[torch.Tensor, ...]
        ) -> None:
            if args and isinstance(args[0], torch.Tensor):
                captured["observation_inputs"].append(args[0].detach())

        def query_hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            captured["queries"].append(output.detach())

        observation_handle = model.action_obs_visual_proj.register_forward_pre_hook(
            observation_pre_hook
        )
        query_handle = model.chunk_obs_query_encoder.register_forward_hook(query_hook)

        original_bound_method = editor.build_layer_updated_cache
        had_instance_override = "build_layer_updated_cache" in editor.__dict__
        previous_override = editor.__dict__.get("build_layer_updated_cache")

        def wrapped_build_layer_updated_cache(
            _editor: nn.Module,
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, torch.Tensor]:
            result = original_bound_method(*args, **kwargs)
            names = (
                "layer_idx",
                "chunk_queries",
                "first_frame_keys",
                "first_frame_values",
            )
            values = dict(kwargs)
            for name, value in zip(names, args):
                values.setdefault(name, value)
            teacher_layer_index = int(values["layer_idx"])
            student_layer = mapped_zero_based.get(teacher_layer_index)
            if student_layer is None:
                return result

            first_keys = values["first_frame_keys"]
            first_values = values["first_frame_values"]
            compact_keys = _select_heads(
                first_keys,
                teacher_heads=teacher_heads,
                student_heads=self.student_config.num_heads,
                head_dim=head_dim,
            ).detach()
            compact_values = _select_heads(
                first_values,
                teacher_heads=teacher_heads,
                student_heads=self.student_config.num_heads,
                head_dim=head_dim,
            ).detach()
            captured["cache"][student_layer] = {
                "k": compact_keys,
                "v": compact_values,
            }

            if student_layer in trace_layers:
                chunk_queries = values["chunk_queries"]
                batch, chunks, query_count, _ = chunk_queries.shape
                token_count = int(first_keys.shape[1])
                projected_queries = editor.layer_query_proj[teacher_layer_index](
                    chunk_queries
                ).view(batch, chunks, query_count, teacher_heads, head_dim)
                key_heads = first_keys.view(batch, token_count, teacher_heads, head_dim)
                route_scores = torch.einsum(
                    "bnqhd,bshd->bhnqs", projected_queries, key_heads
                ) / math.sqrt(head_dim)
                route_probabilities = torch.softmax(route_scores, dim=-1)
                head_indices = _evenly_spaced_indices(
                    teacher_heads,
                    self.student_config.num_heads,
                    route_probabilities.device,
                )
                route_probabilities = route_probabilities.index_select(
                    1, head_indices
                ).mean(dim=1)
                base_keys = first_keys.unsqueeze(1).expand(-1, chunks, -1, -1)
                base_values = first_values.unsqueeze(1).expand(-1, chunks, -1, -1)
                effective_delta_k = _select_heads(
                    result["k"] - base_keys,
                    teacher_heads=teacher_heads,
                    student_heads=self.student_config.num_heads,
                    head_dim=head_dim,
                )
                effective_delta_v = _select_heads(
                    result["v"] - base_values,
                    teacher_heads=teacher_heads,
                    student_heads=self.student_config.num_heads,
                    head_dim=head_dim,
                )
                captured["editor_trace"][student_layer] = {
                    "route_probabilities": route_probabilities.detach(),
                    "effective_delta_k": effective_delta_k.detach(),
                    "effective_delta_v": effective_delta_v.detach(),
                }
            return result

        editor.build_layer_updated_cache = MethodType(  # type: ignore[method-assign]
            wrapped_build_layer_updated_cache, editor
        )
        try:
            rollout = self._rollout(sample, tiled)
        finally:
            observation_handle.remove()
            query_handle.remove()
            if had_instance_override:
                editor.build_layer_updated_cache = previous_override
            else:
                delattr(editor, "build_layer_updated_cache")
        return rollout, captured

    def _choose_chunk_indices(
        self,
        sample: Mapping[str, Any],
        *,
        batch_size: int,
        num_chunks: int,
        device: torch.device,
    ) -> torch.Tensor:
        requested = sample.get("stage1_chunk_index")
        if requested is None:
            action_is_pad = sample.get("action_is_pad")
            if isinstance(action_is_pad, torch.Tensor) and tuple(
                action_is_pad.shape[:2]
            ) == (batch_size, num_chunks * self.student_config.action_chunk_size):
                valid_chunks = (~action_is_pad.to(device=device, dtype=torch.bool)).view(
                    batch_size, num_chunks, self.student_config.action_chunk_size
                ).any(dim=-1)
                if not valid_chunks.any(dim=-1).all():
                    raise ValueError("Every sample needs at least one non-padding action chunk")
                return torch.multinomial(valid_chunks.float(), 1).squeeze(1)
            return torch.randint(num_chunks, (batch_size,), device=device)
        indices = torch.as_tensor(requested, device=device, dtype=torch.long)
        if indices.ndim == 0:
            indices = indices.expand(batch_size)
        if tuple(indices.shape) != (batch_size,):
            raise ValueError("stage1_chunk_index must be scalar or [B]")
        if (indices < 0).any() or (indices >= num_chunks).any():
            raise ValueError("stage1_chunk_index is outside the action horizon")
        return indices

    @torch.no_grad()
    def _predict_velocity_with_response_trace(
        self,
        *,
        noisy_action: torch.Tensor,
        timestep_action: torch.Tensor,
        video_state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        model = self.raw_model
        mot = model.mot
        original_bound_method = mot._forward_chunk_routed_prior_only_attention
        had_instance_override = (
            "_forward_chunk_routed_prior_only_attention" in mot.__dict__
        )
        previous_override = mot.__dict__.get(
            "_forward_chunk_routed_prior_only_attention"
        )
        mapped_zero_based = {
            teacher_layer - 1: student_layer
            for student_layer, teacher_layer in enumerate(
                self.teacher_layer_mapping, start=1
            )
        }
        trace_layers = set(self.student_config.distill_layers)
        response_trace: dict[int, torch.Tensor] = {}
        call_index = 0

        def wrapped_attention(
            _mot: nn.Module,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            nonlocal call_index
            response = original_bound_method(*args, **kwargs)
            teacher_layer_index = call_index
            call_index += 1
            student_layer = mapped_zero_based.get(teacher_layer_index)
            if student_layer in trace_layers:
                editor = model.mot.chunk_kv_cache_editor
                response_trace[student_layer] = _select_heads(
                    response,
                    teacher_heads=int(editor.num_heads),
                    student_heads=self.student_config.num_heads,
                    head_dim=int(editor.head_dim),
                ).detach()
            return response

        mot._forward_chunk_routed_prior_only_attention = MethodType(  # type: ignore[method-assign]
            wrapped_attention, mot
        )
        try:
            velocity = model._predict_action_flow_with_video_state(
                noisy_action=noisy_action,
                timestep_action=timestep_action,
                video_state=video_state,
            ).detach()
        finally:
            if had_instance_override:
                mot._forward_chunk_routed_prior_only_attention = previous_override
            else:
                delattr(mot, "_forward_chunk_routed_prior_only_attention")
        if call_index != int(model.mot.num_layers):
            raise RuntimeError(
                "AHA action response trace saw an unexpected layer count: "
                f"{call_index} vs {model.mot.num_layers}"
            )
        return velocity, response_trace

    @torch.no_grad()
    def prepare_batch(
        self,
        sample: dict[str, Any],
        *,
        tiled: bool = False,
    ) -> AHAOVCRSTeacherBatch:
        self.teacher.eval()
        rollout, captured = self._rollout_with_ovcr_trace(sample, tiled)
        video_state = rollout["video_state"]
        ground_truth_action = video_state["action"].detach()
        batch_size, action_horizon, action_dim = ground_truth_action.shape
        chunk_size = self.student_config.action_chunk_size
        if action_dim != self.student_config.action_dim:
            raise ValueError(
                f"AHA action dim {action_dim} differs from OVCR-S "
                f"{self.student_config.action_dim}"
            )
        if action_horizon % chunk_size:
            raise ValueError("AHA action horizon is not divisible by the student chunk size")
        num_chunks = action_horizon // chunk_size

        available_anchor_steps = tuple(
            int(step)
            for step in rollout["capture_step_indices"]
            if int(step) < self.rollout_steps
        )
        state_stack = torch.stack(
            [rollout["captured_states"][step] for step in available_anchor_steps],
            dim=1,
        ).to(device=ground_truth_action.device, dtype=ground_truth_action.dtype)
        state_stack = state_stack.view(
            batch_size,
            len(available_anchor_steps),
            num_chunks,
            chunk_size,
            action_dim,
        ).permute(0, 2, 1, 3, 4)
        anchor_choice = torch.randint(
            len(available_anchor_steps),
            (batch_size, num_chunks),
            device=ground_truth_action.device,
        )
        noisy_chunks = torch.gather(
            state_stack,
            dim=2,
            index=anchor_choice.view(batch_size, num_chunks, 1, 1, 1).expand(
                -1, -1, 1, chunk_size, action_dim
            ),
        ).squeeze(2)
        noisy_action_full = noisy_chunks.reshape(batch_size, action_horizon, action_dim)
        anchor_values = torch.tensor(
            available_anchor_steps,
            device=ground_truth_action.device,
            dtype=torch.long,
        )[anchor_choice]
        timesteps = rollout["timesteps"].to(
            device=ground_truth_action.device, dtype=ground_truth_action.dtype
        )
        timestep_full = timesteps[anchor_values]
        teacher_velocity_full, teacher_action_responses_full = (
            self._predict_velocity_with_response_trace(
                noisy_action=noisy_action_full,
                timestep_action=timestep_full,
                video_state=video_state,
            )
        )

        chunk_index = self._choose_chunk_indices(
            sample,
            batch_size=batch_size,
            num_chunks=num_chunks,
            device=ground_truth_action.device,
        )
        noisy_action = _gather_action_chunk(noisy_action_full, chunk_index, chunk_size)
        teacher_velocity = _gather_action_chunk(
            teacher_velocity_full, chunk_index, chunk_size
        )
        teacher_action = _gather_action_chunk(
            rollout["final_latents"].detach(), chunk_index, chunk_size
        )
        ground_truth_chunk = _gather_action_chunk(
            ground_truth_action, chunk_index, chunk_size
        )
        action_t = _gather_chunk_tokens(timestep_full, chunk_index)[:, 0]
        anchor_step = _gather_chunk_tokens(anchor_values, chunk_index)[:, 0]

        if not captured["queries"] or not captured["observation_inputs"]:
            raise RuntimeError("AHA OVCR hooks did not observe query/observation tensors")
        teacher_queries_full = captured["queries"][-1]
        if teacher_queries_full.shape[:2] != (batch_size, num_chunks):
            raise RuntimeError(
                "Captured AHA queries do not match the action chunk layout: "
                f"{tuple(teacher_queries_full.shape)}"
            )
        observation_flat = captured["observation_inputs"][-1]
        if observation_flat.ndim != 3 or observation_flat.shape[0] != batch_size * num_chunks:
            raise RuntimeError(
                "Captured pre-projection observation tokens must be [B*N,S,C], "
                f"got {tuple(observation_flat.shape)}"
            )
        observation_tokens_full = observation_flat.reshape(
            batch_size,
            num_chunks,
            observation_flat.shape[1],
            observation_flat.shape[2],
        )
        if observation_tokens_full.shape[-1] != self.student_config.observation_dim:
            raise ValueError(
                "OVCR-S observation_dim does not match AHA VAE latent channels: "
                f"{self.student_config.observation_dim} vs "
                f"{observation_tokens_full.shape[-1]}"
            )
        observation_tokens = _gather_chunk_tokens(
            observation_tokens_full, chunk_index
        )
        observation_mask = torch.ones(
            observation_tokens.shape[:-1],
            dtype=torch.bool,
            device=observation_tokens.device,
        )
        teacher_queries = _gather_chunk_tokens(teacher_queries_full, chunk_index)

        missing_cache = [
            layer
            for layer in range(1, self.student_config.num_layers + 1)
            if layer not in captured["cache"]
        ]
        if missing_cache:
            raise RuntimeError(f"AHA trace missed mapped cache layers: {missing_cache}")
        video_kv_cache = tuple(
            captured["cache"][layer]
            for layer in range(1, self.student_config.num_layers + 1)
        )
        teacher_editor_trace: dict[int, dict[str, torch.Tensor]] = {}
        for layer in self.student_config.distill_layers:
            if layer not in captured["editor_trace"]:
                raise RuntimeError(f"AHA trace missed student distill layer {layer}")
            trace = captured["editor_trace"][layer]
            teacher_editor_trace[layer] = {
                "route_probabilities": _gather_chunk_tokens(
                    trace["route_probabilities"], chunk_index
                ),
                "effective_delta_k": _gather_chunk_tokens(
                    trace["effective_delta_k"], chunk_index
                ),
                "effective_delta_v": _gather_chunk_tokens(
                    trace["effective_delta_v"], chunk_index
                ),
            }
        teacher_action_responses = {
            layer: _gather_action_chunk(response, chunk_index, chunk_size)
            for layer, response in teacher_action_responses_full.items()
        }
        missing_responses = sorted(
            set(self.student_config.distill_layers) - set(teacher_action_responses)
        )
        if missing_responses:
            raise RuntimeError(
                f"AHA trace missed mapped action responses: {missing_responses}"
            )

        action_is_pad = sample.get("action_is_pad")
        if isinstance(action_is_pad, torch.Tensor) and tuple(action_is_pad.shape[:2]) == (
            batch_size,
            action_horizon,
        ):
            action_is_pad = _gather_action_chunk(
                action_is_pad.to(device=ground_truth_action.device, dtype=torch.bool),
                chunk_index,
                chunk_size,
            )
        else:
            action_is_pad = None
        scheduler = self.raw_model.infer_action_scheduler
        train_timesteps = float(
            getattr(scheduler, "num_train_timesteps", self.student_config.num_train_timesteps)
        )
        sigma = (action_t.float() / train_timesteps).clamp(0.0, 1.0)

        return AHAOVCRSTeacherBatch(
            noisy_action=noisy_action,
            action_t=action_t,
            sigma=sigma,
            teacher_velocity=teacher_velocity,
            teacher_action=teacher_action,
            ground_truth_action=ground_truth_chunk,
            observation_tokens=observation_tokens,
            observation_mask=observation_mask,
            video_kv_cache=video_kv_cache,
            teacher_queries=teacher_queries,
            teacher_editor_trace=teacher_editor_trace,
            teacher_action_responses=teacher_action_responses,
            action_is_pad=action_is_pad,
            chunk_index=chunk_index,
            anchor_step=anchor_step,
        )
