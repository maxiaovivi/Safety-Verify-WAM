from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aha_teacher import (
    AHAOVCRSTeacherBatch,
    AHAOVCRTeacherAdapter,
    GroundTruthTargetAdapter,
)
from .ovcr_s import OVCRSActionGenerator, OVCRSConfig


def _resolve_parameter_dtype(
    value: str | torch.dtype | None,
    *,
    fallback: torch.dtype,
) -> torch.dtype:
    if value is None:
        return fallback
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(value).strip().lower()
    if key not in aliases:
        raise ValueError(
            "student_parameter_dtype must be one of: bf16, bfloat16, fp32, float32"
        )
    return aliases[key]


@dataclass(frozen=True)
class Stage1LossConfig:
    velocity_weight: float = 1.0
    teacher_action_weight: float = 1.0
    ground_truth_action_weight: float = 0.25
    preservation_weight: float = 0.0
    query_weight: float = 0.05
    route_weight: float = 0.10
    delta_weight: float = 0.10
    response_weight: float = 0.25
    response_projection_dim: int | None = None
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        weights = (
            self.velocity_weight,
            self.teacher_action_weight,
            self.ground_truth_action_weight,
            self.preservation_weight,
            self.query_weight,
            self.route_weight,
            self.delta_weight,
            self.response_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Stage 1 loss weights must be non-negative")
        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one Stage 1 loss term must be enabled")
        if self.response_projection_dim is not None and self.response_projection_dim <= 0:
            raise ValueError("response_projection_dim must be positive when set")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _masked_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: torch.Tensor | None,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Action prediction/target shapes differ: {prediction.shape} vs {target.shape}"
        )
    per_token = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(
        dim=-1
    )
    if action_is_pad is None:
        return per_token.mean()
    if tuple(action_is_pad.shape) != tuple(per_token.shape):
        raise ValueError("action_is_pad must be [B,T]")
    valid = (~action_is_pad).to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


def _resize_last_dimension(tensor: torch.Tensor, output_dim: int) -> torch.Tensor:
    if tensor.shape[-1] == output_dim:
        return tensor
    flattened = tensor.float().reshape(-1, 1, tensor.shape[-1])
    resized = F.adaptive_avg_pool1d(flattened, output_dim)
    return resized.reshape(*tensor.shape[:-1], output_dim).to(dtype=tensor.dtype)


def _cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Cosine tensors differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    return (1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).mean()


def _nonzero_target_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Delta tensors differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    prediction_float = prediction.float()
    target_float = target.float()
    valid = target_float.norm(dim=-1) > eps
    if not valid.any():
        return prediction_float.sum() * 0.0
    cosine = F.cosine_similarity(prediction_float, target_float, dim=-1)
    return (1.0 - cosine[valid]).mean()


def stage1_distillation_loss(
    outputs: Mapping[str, Any],
    targets: AHAOVCRSTeacherBatch,
    config: Stage1LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = outputs["action_velocity"]
    zero = prediction.sum() * 0.0
    velocity_loss = _masked_action_mse(
        prediction, targets.teacher_velocity, targets.action_is_pad
    )
    sigma = targets.sigma.to(device=prediction.device, dtype=prediction.dtype)
    denoised_action = targets.noisy_action - sigma[:, None, None] * prediction
    teacher_action_loss = _masked_action_mse(
        denoised_action, targets.teacher_action, targets.action_is_pad
    )
    ground_truth_action_loss = _masked_action_mse(
        denoised_action, targets.ground_truth_action, targets.action_is_pad
    )

    if config.preservation_weight > 0:
        if targets.reference_velocity is None:
            raise ValueError(
                "preservation_weight is enabled but no Efficient reference velocity was provided"
            )
        preservation_loss = _masked_action_mse(
            prediction, targets.reference_velocity, targets.action_is_pad
        )
    else:
        preservation_loss = zero

    if config.query_weight > 0:
        student_queries = outputs["queries"]
        teacher_queries = _resize_last_dimension(
            targets.teacher_queries, student_queries.shape[-1]
        )
        query_loss = _cosine_loss(student_queries, teacher_queries)
    else:
        query_loss = zero

    editor_trace = outputs.get("editor_trace", {})
    route_losses: list[torch.Tensor] = []
    delta_losses: list[torch.Tensor] = []
    if config.route_weight > 0 or config.delta_weight > 0:
        for layer, teacher_trace in targets.teacher_editor_trace.items():
            if layer not in editor_trace:
                raise ValueError(f"Student did not return editor trace for layer {layer}")
            student_trace = editor_trace[layer]
            if config.route_weight > 0:
                teacher_route = teacher_trace["route_probabilities"].float().clamp_min(
                    config.eps
                )
                student_route = student_trace["route_probabilities"].float().clamp_min(
                    config.eps
                )
                if teacher_route.shape != student_route.shape:
                    raise ValueError(
                        f"Layer {layer} route shapes differ: "
                        f"{teacher_route.shape} vs {student_route.shape}"
                    )
                teacher_route = teacher_route / teacher_route.sum(
                    dim=-1, keepdim=True
                )
                student_route = student_route / student_route.sum(
                    dim=-1, keepdim=True
                )
                route_losses.append(
                    (teacher_route * (teacher_route.log() - student_route.log()))
                    .sum(dim=-1)
                    .mean()
                )
            if config.delta_weight > 0:
                delta_losses.extend(
                    [
                        _nonzero_target_cosine_loss(
                            student_trace["effective_delta_k"],
                            teacher_trace["effective_delta_k"],
                            config.eps,
                        ),
                        _nonzero_target_cosine_loss(
                            student_trace["effective_delta_v"],
                            teacher_trace["effective_delta_v"],
                            config.eps,
                        ),
                    ]
                )
    route_loss = torch.stack(route_losses).mean() if route_losses else zero
    delta_loss = torch.stack(delta_losses).mean() if delta_losses else zero
    response_losses: list[torch.Tensor] = []
    if config.response_weight > 0:
        student_responses = outputs.get("action_responses", {})
        if not targets.teacher_action_responses:
            raise ValueError(
                "response_weight is enabled but no AHA action responses were provided"
            )
        for layer, teacher_response in targets.teacher_action_responses.items():
            if layer not in student_responses:
                raise ValueError(
                    f"Student did not return action response for layer {layer}"
                )
            student_response = student_responses[layer]
            if config.response_projection_dim is not None:
                student_response = _resize_last_dimension(
                    student_response, config.response_projection_dim
                )
                teacher_response = _resize_last_dimension(
                    teacher_response, config.response_projection_dim
                )
            response_losses.append(_cosine_loss(student_response, teacher_response))
    response_loss = torch.stack(response_losses).mean() if response_losses else zero

    total = (
        config.velocity_weight * velocity_loss
        + config.teacher_action_weight * teacher_action_loss
        + config.ground_truth_action_weight * ground_truth_action_loss
        + config.preservation_weight * preservation_loss
        + config.query_weight * query_loss
        + config.route_weight * route_loss
        + config.delta_weight * delta_loss
        + config.response_weight * response_loss
    )
    return total, {
        "loss": total.detach(),
        "loss_velocity": velocity_loss.detach(),
        "loss_teacher_action": teacher_action_loss.detach(),
        "loss_ground_truth_action": ground_truth_action_loss.detach(),
        "loss_preservation": preservation_loss.detach(),
        "loss_query": query_loss.detach(),
        "loss_route": route_loss.detach(),
        "loss_delta": delta_loss.detach(),
        "loss_response": response_loss.detach(),
    }


class _OVCREditorTrainableView(nn.Module):
    """Expose only OVCR conditioning modules to AHA's DiT-only trainer."""

    def __init__(self, student: OVCRSActionGenerator) -> None:
        super().__init__()
        # Keep this as a non-registered reference. The student remains the one
        # owner of checkpoint keys while this view controls optimizer selection.
        object.__setattr__(self, "_student", student)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        student = object.__getattribute__(self, "_student")
        yield from student.query_encoder.parameters(recurse=recurse)
        yield from student.cache_editor.parameters(recurse=recurse)

    def train(self, mode: bool = True) -> "_OVCREditorTrainableView":
        self.training = mode
        student = object.__getattribute__(self, "_student")
        student.query_encoder.train(mode)
        student.cache_editor.train(mode)
        return self


class AHAOVCRSStage1Program(nn.Module):
    """AHA-compatible Stage 1 training model.

    AHA stays frozen. Only ``student`` is exposed as ``dit`` so the released
    AHA trainer can build its optimizer without adding teacher parameters.
    """

    def __init__(
        self,
        teacher_adapter: nn.Module,
        student: OVCRSActionGenerator,
        loss_config: Stage1LossConfig,
        efficient_training_adapter: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.teacher_adapter = teacher_adapter
        self.student = student
        self.loss_config = loss_config
        self.efficient_training_adapter = efficient_training_adapter
        self._freeze_action_expert = bool(
            getattr(efficient_training_adapter, "freeze_action_expert", False)
        )
        object.__setattr__(
            self,
            "_editor_trainable_view",
            _OVCREditorTrainableView(student),
        )
        if teacher_adapter.student_config != student.config:
            raise ValueError("Teacher adapter and student use different OVCR-S configs")

    @property
    def dit(self) -> nn.Module:
        if self._freeze_action_expert:
            return object.__getattribute__(self, "_editor_trainable_view")
        return self.student

    @property
    def device(self) -> torch.device:
        return next(self.student.parameters()).device

    @property
    def torch_dtype(self) -> torch.dtype:
        return next(self.student.parameters()).dtype

    def train(self, mode: bool = True) -> "AHAOVCRSStage1Program":
        super().train(mode)
        self.teacher_adapter.eval()
        if self.efficient_training_adapter is not None:
            self.efficient_training_adapter.eval()
        self.student.train(mode)
        return self

    def get_additional_trainable_modules(self) -> dict[str, nn.Module]:
        return {}

    def training_loss(
        self,
        sample: dict[str, Any],
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        targets = self.teacher_adapter.prepare_batch(sample, tiled=tiled)
        if self.efficient_training_adapter is not None:
            prepare_batch = getattr(self.efficient_training_adapter, "prepare_batch", None)
            if not callable(prepare_batch):
                raise TypeError("Efficient training adapter has no prepare_batch method")
            targets = prepare_batch(sample, targets)
        if self.loss_config.response_weight > 0 and not targets.teacher_action_responses:
            if self.efficient_training_adapter is None:
                raise ValueError("AHA response targets were not prepared")
            convert_action = getattr(
                self.efficient_training_adapter, "action_efficient_to_aha", None
            )
            attach_responses = getattr(
                self.teacher_adapter, "attach_action_response_targets", None
            )
            if not callable(convert_action) or not callable(attach_responses):
                raise TypeError(
                    "Efficient/AHA adapters do not support matched response targets"
                )
            noisy_action_aha = convert_action(targets.noisy_action)
            targets = attach_responses(targets, noisy_action_aha)
        needs_editor_trace = (
            self.loss_config.route_weight > 0
            or self.loss_config.delta_weight > 0
            or self.loss_config.response_weight > 0
        )
        outputs = self.student(
            noisy_action=targets.noisy_action,
            action_t=targets.action_t,
            initial_state=targets.initial_state,
            observation_tokens=targets.observation_tokens,
            observation_mask=targets.observation_mask,
            video_kv_cache=targets.video_kv_cache,
            return_trace=needs_editor_trace,
        )
        loss, tensor_terms = stage1_distillation_loss(
            outputs, targets, self.loss_config
        )
        terms = {key: float(value.item()) for key, value in tensor_terms.items()}
        terms["mean_chunk_index"] = float(targets.chunk_index.float().mean().item())
        terms["mean_anchor_step"] = float(targets.anchor_step.float().mean().item())
        terms["mean_action_sigma"] = float(targets.sigma.float().mean().item())
        return loss, terms

    def forward(
        self,
        sample: dict[str, Any],
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self.training_loss(sample, tiled=tiled)

    @torch.no_grad()
    def evaluate_validation(
        self,
        sample: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, float]:
        was_training = self.student.training
        self.eval()
        _, terms = self.training_loss(sample)
        if was_training:
            self.train()
        return {f"val_{key}": value for key, value in terms.items()}

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        step: int | None = None,
    ) -> None:
        checkpoint_path = Path(path).expanduser()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format": "ovcr_s_stage1",
            "student": {
                key: value.detach().cpu() for key, value in self.student.state_dict().items()
            },
            "student_config": self.student.config.to_dict(),
            "loss_config": self.loss_config.to_dict(),
            "teacher_layer_mapping": list(
                self.teacher_adapter.teacher_layer_mapping
            ),
            "target_source": getattr(
                self.teacher_adapter, "target_source", "aha_teacher"
            ),
            "step": step,
        }
        if self.efficient_training_adapter is not None:
            payload["efficient_training"] = {
                "action_noise_sampling": getattr(
                    self.efficient_training_adapter,
                    "action_noise_sampling",
                    None,
                ),
                "action_flow_target": getattr(
                    self.efficient_training_adapter,
                    "action_flow_target",
                    None,
                ),
                "freeze_action_expert": self._freeze_action_expert,
            }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, checkpoint_path)

    def load_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        checkpoint_path = Path(path).expanduser()
        try:
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("student"), dict):
            raise TypeError(f"Invalid OVCR-S Stage 1 checkpoint: {checkpoint_path}")
        saved_config = payload.get("student_config")
        if saved_config != self.student.config.to_dict():
            raise RuntimeError("OVCR-S checkpoint architecture differs from this student")
        self.student.load_state_dict(payload["student"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


def create_aha_ovcr_s_stage1(
    teacher: nn.Module,
    *,
    student_config: Mapping[str, Any] | OVCRSConfig | None = None,
    loss_config: Mapping[str, Any] | Stage1LossConfig | None = None,
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
    action_horizon: int | None = None,
    action_chunk_size: int | None = None,
    action_train_timestep_mode: str | None = None,
    mot_checkpoint_mixed_attn: bool | None = None,
    num_history_frames: int | None = None,
    efficient_action_checkpoint: str | Path | None = None,
    strict_action_init: bool = True,
    efficient_conditioning: Mapping[str, Any] | None = None,
    student_parameter_dtype: str | torch.dtype | None = None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> AHAOVCRSStage1Program:
    if student_config is None:
        resolved_student_config = OVCRSConfig()
    elif isinstance(student_config, OVCRSConfig):
        resolved_student_config = student_config
    else:
        resolved_student_config = OVCRSConfig(**dict(student_config))
    if loss_config is None:
        resolved_loss_config = Stage1LossConfig()
    elif isinstance(loss_config, Stage1LossConfig):
        resolved_loss_config = loss_config
    else:
        resolved_loss_config = Stage1LossConfig(**dict(loss_config))

    student = OVCRSActionGenerator(resolved_student_config)
    if efficient_action_checkpoint not in (None, "", "null"):
        checkpoint_path = Path(efficient_action_checkpoint).expanduser()
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"Efficient action checkpoint must contain a mapping: {checkpoint_path}"
            )
        student.load_efficient_action_expert(
            checkpoint, strict=bool(strict_action_init)
        )
    resolved_parameter_dtype = _resolve_parameter_dtype(
        student_parameter_dtype,
        fallback=model_dtype,
    )
    student.to(device=torch.device(device), dtype=resolved_parameter_dtype)
    efficient_training_adapter: nn.Module | None = None
    if efficient_conditioning is not None:
        structural_weights = {
            "query_weight": resolved_loss_config.query_weight,
            "route_weight": resolved_loss_config.route_weight,
            "delta_weight": resolved_loss_config.delta_weight,
        }
        enabled_structural = {
            name: weight for name, weight in structural_weights.items() if weight > 0
        }
        if enabled_structural:
            raise ValueError(
                "AHA structural losses cannot supervise Efficient-generated K/V; "
                f"set these weights to zero: {enabled_structural}"
            )
        if resolved_loss_config.response_weight > 0:
            conditioning = dict(efficient_conditioning)
            if conditioning.get("action_flow_target") != "ground_truth":
                raise ValueError(
                    "AHA response KD with Efficient K/V requires "
                    "action_flow_target='ground_truth'"
                )
            if conditioning.get("action_noise_sampling") != "uniform_shifted":
                raise ValueError(
                    "AHA response KD with Efficient K/V requires "
                    "action_noise_sampling='uniform_shifted'"
                )
            if resolved_loss_config.teacher_action_weight > 0:
                raise ValueError(
                    "AHA response KD with Efficient K/V requires "
                    "teacher_action_weight=0"
                )
        from .efficient_training import EfficientStudentTrainingAdapter

        efficient_training_adapter = EfficientStudentTrainingAdapter(
            student=student,
            device=device,
            student_dtype=model_dtype,
            **dict(efficient_conditioning),
        )
    elif resolved_loss_config.preservation_weight > 0:
        raise ValueError(
            "preservation_weight requires efficient_conditioning"
        )

    adapter = AHAOVCRTeacherAdapter(
        teacher,
        resolved_student_config,
        teacher_layer_mapping=teacher_layer_mapping,
        rollout_steps=rollout_steps,
        capture_steps=capture_steps,
        sigma_shift=sigma_shift,
        capture_structural_targets=efficient_training_adapter is None,
        capture_action_response_targets=(
            efficient_training_adapter is not None
            and resolved_loss_config.response_weight > 0
        ),
    )
    teacher_model = adapter.raw_model
    compatibility_values = {
        "action_horizon": action_horizon,
        "action_chunk_size": action_chunk_size,
        "action_train_timestep_mode": action_train_timestep_mode,
    }
    for name, requested in compatibility_values.items():
        if requested is None:
            continue
        actual = getattr(teacher_model, name, None)
        if str(actual) != str(requested):
            raise ValueError(
                f"Stage 1 {name}={requested!r} differs from AHA teacher {actual!r}"
            )
    if mot_checkpoint_mixed_attn is not None:
        actual_mixed_attn = getattr(
            teacher_model.mot, "mot_checkpoint_mixed_attn", None
        )
        if bool(actual_mixed_attn) != bool(mot_checkpoint_mixed_attn):
            raise ValueError(
                "Stage 1 mot_checkpoint_mixed_attn differs from the AHA teacher"
            )
    if num_history_frames is not None:
        history_getter = getattr(
            teacher_model, "_configured_num_history_frames", None
        )
        actual_history = (
            int(history_getter())
            if callable(history_getter)
            else int(getattr(teacher_model, "num_history_frames", 0))
        )
        if actual_history != int(num_history_frames):
            raise ValueError(
                "Stage 1 num_history_frames differs from the AHA teacher: "
                f"{num_history_frames} vs {actual_history}"
            )
    return AHAOVCRSStage1Program(
        adapter,
        student,
        resolved_loss_config,
        efficient_training_adapter=efficient_training_adapter,
    )


def create_ground_truth_ovcr_s_stage1(
    *,
    student_config: Mapping[str, Any] | OVCRSConfig | None = None,
    loss_config: Mapping[str, Any] | Stage1LossConfig | None = None,
    action_horizon: int = 64,
    sigma_shift: float | None = None,
    efficient_action_checkpoint: str | Path | None = None,
    strict_action_init: bool = True,
    efficient_conditioning: Mapping[str, Any] | None = None,
    student_parameter_dtype: str | torch.dtype | None = None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> AHAOVCRSStage1Program:
    """Create dataset-flow training without instantiating an AHA teacher."""

    if sigma_shift is not None:
        raise ValueError(
            "Ground-truth-only Stage 1 takes action_sigma_shift from "
            "efficient_conditioning"
        )
    if efficient_conditioning is None:
        raise ValueError("Ground-truth-only Stage 1 requires efficient_conditioning")
    conditioning = dict(efficient_conditioning)
    if conditioning.get("action_flow_target") != "ground_truth":
        raise ValueError(
            "Ground-truth-only Stage 1 requires action_flow_target='ground_truth'"
        )
    if conditioning.get("action_noise_sampling") != "uniform_shifted":
        raise ValueError(
            "Ground-truth-only Stage 1 requires action_noise_sampling='uniform_shifted'"
        )

    if student_config is None:
        resolved_student_config = OVCRSConfig()
    elif isinstance(student_config, OVCRSConfig):
        resolved_student_config = student_config
    else:
        resolved_student_config = OVCRSConfig(**dict(student_config))
    if loss_config is None:
        resolved_loss_config = Stage1LossConfig()
    elif isinstance(loss_config, Stage1LossConfig):
        resolved_loss_config = loss_config
    else:
        resolved_loss_config = Stage1LossConfig(**dict(loss_config))
    unsupported_weights = {
        "teacher_action_weight": resolved_loss_config.teacher_action_weight,
        "query_weight": resolved_loss_config.query_weight,
        "route_weight": resolved_loss_config.route_weight,
        "delta_weight": resolved_loss_config.delta_weight,
        "response_weight": resolved_loss_config.response_weight,
    }
    enabled_unsupported = {
        name: weight for name, weight in unsupported_weights.items() if weight > 0
    }
    if enabled_unsupported:
        raise ValueError(
            "Ground-truth-only Stage 1 cannot compute AHA teacher losses; "
            f"set these weights to zero: {enabled_unsupported}"
        )

    student = OVCRSActionGenerator(resolved_student_config)
    if efficient_action_checkpoint not in (None, "", "null"):
        checkpoint_path = Path(efficient_action_checkpoint).expanduser()
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"Efficient action checkpoint must contain a mapping: {checkpoint_path}"
            )
        student.load_efficient_action_expert(
            checkpoint, strict=bool(strict_action_init)
        )
    resolved_parameter_dtype = _resolve_parameter_dtype(
        student_parameter_dtype,
        fallback=model_dtype,
    )
    student.to(device=torch.device(device), dtype=resolved_parameter_dtype)

    from .efficient_training import EfficientStudentTrainingAdapter

    efficient_training_adapter = EfficientStudentTrainingAdapter(
        student=student,
        device=device,
        student_dtype=model_dtype,
        **conditioning,
    )
    adapter = GroundTruthTargetAdapter(
        resolved_student_config,
        action_horizon=action_horizon,
        device=device,
    )
    return AHAOVCRSStage1Program(
        adapter,
        student,
        resolved_loss_config,
        efficient_training_adapter=efficient_training_adapter,
    )
