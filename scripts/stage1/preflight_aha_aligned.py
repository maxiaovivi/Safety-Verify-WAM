#!/usr/bin/env python3
"""Audit one real AHA-aligned Stage 1 batch before starting a long run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ahawam.runtime import build_datasets
from ahawam.utils import misc
from safety_verify_wam.stage1.distill import stage1_distillation_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-index", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_info(path: str) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", path, *args], text=True, stderr=subprocess.STDOUT
        ).strip()

    return {
        "path": path,
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
        "upstream": run("remote", "get-url", "upstream"),
    }


def tensor_shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, Mapping):
        return {str(key): tensor_shape(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_shape(item) for item in value]
    return type(value).__name__


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "abs_max": float(value.abs().max().item()),
    }


def module_grad_report(module: torch.nn.Module) -> dict[str, Any]:
    squared_norm = 0.0
    tensor_count = 0
    finite = True
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        tensor_count += 1
        finite = finite and bool(torch.isfinite(gradient).all().item())
        squared_norm += float(gradient.square().sum().item())
    return {
        "norm": squared_norm**0.5,
        "gradient_tensor_count": tensor_count,
        "finite": finite,
        "nonzero": squared_norm > 0.0,
    }


def parameter_grad_report(parameter: torch.nn.Parameter) -> dict[str, Any]:
    if parameter.grad is None:
        return {
            "norm": 0.0,
            "gradient_tensor_count": 0,
            "finite": True,
            "nonzero": False,
        }
    gradient = parameter.grad.detach().float()
    norm = float(gradient.norm().item())
    return {
        "norm": norm,
        "gradient_tensor_count": 1,
        "finite": bool(torch.isfinite(gradient).all().item()),
        "nonzero": norm > 0.0,
    }


def _student_forward(
    model: torch.nn.Module,
    targets: Any,
    *,
    initial_state: torch.Tensor | None = None,
    action_context: torch.Tensor | None = None,
) -> dict[str, Any]:
    conditioning = model.student.prepare_conditioning(
        targets.observation_tokens,
        targets.video_kv_cache,
        targets.observation_mask,
        return_trace=True,
    )
    outputs = model.student.predict_velocity(
        targets.noisy_action,
        targets.action_t,
        targets.initial_state if initial_state is None else initial_state,
        conditioning,
        action_context=(
            targets.action_context if action_context is None else action_context
        ),
        action_context_mask=targets.action_context_mask,
        return_trace=True,
    )
    outputs["queries"] = conditioning["queries"]
    outputs["editor_trace"] = conditioning["editor_trace"]
    return outputs


def run_preflight(args: argparse.Namespace, report: dict[str, Any]) -> None:
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    predictions_path = Path(args.predictions).resolve()
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU")
    if str(cfg.model.student_config.action_architecture) != "aha_aligned":
        raise ValueError("This preflight requires action_architecture=aha_aligned")

    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    report.update(
        {
            "schema_version": 1,
            "purpose": "real-data AHA-aligned Stage 1 launch audit",
            "status": "initializing",
            "seed": seed,
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "sources": {
                "aha_wam": git_info(os.environ["AHA_WAM_ROOT"]),
                "safety_verify_wam": git_info(
                    os.environ["SAFETY_VERIFY_WAM_ROOT"]
                ),
                "efficient_wam": git_info(os.environ["EFFICIENT_WAM_SOURCE"]),
            },
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "total_memory_mib": torch.cuda.get_device_properties(0).total_memory
                // (1024 * 1024),
            },
        }
    )
    atomic_json(output_path, report)

    misc.register_work_dir(str(output_path.parent / ".preflight-work"))
    started = time.perf_counter()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
    torch.cuda.synchronize(0)
    report["timing_seconds"] = {"model_init": time.perf_counter() - started}
    report["memory_mib"] = {
        "model_init_peak_allocated": torch.cuda.max_memory_allocated(0) / 1024**2,
        "model_init_peak_reserved": torch.cuda.max_memory_reserved(0) / 1024**2,
    }

    student = model.student
    teacher_adapter = model.teacher_adapter
    teacher = teacher_adapter.raw_model
    initialization = getattr(student, "initialization_metadata", {})
    layer_mapping = [int(value) for value in teacher_adapter.teacher_layer_mapping]
    expected_mapping = [int(value) for value in cfg.model.teacher_layer_mapping]
    model_contract = {
        "action_architecture": student.config.action_architecture,
        "action_tokens": student.config.action_chunk_size,
        "action_dim": student.config.action_dim,
        "action_hidden_dim": student.config.action_hidden_dim,
        "action_ffn_dim": student.config.action_ffn_dim,
        "action_heads": student.config.num_heads,
        "action_head_dim": student.config.head_dim,
        "action_layers": student.config.num_layers,
        "text_context_dim": student.config.text_context_dim,
        "state_dim": student.config.state_dim,
        "num_registers": student.config.num_registers,
        "has_proprio_encoder": student.proprio_encoder is not None,
        "has_action_branch_embedding": student.action_branch_embedding is not None,
        "all_blocks_have_cross_attention": all(
            hasattr(block, "cross_attn") and hasattr(block, "norm3")
            for block in student.action_expert.blocks
        ),
        "teacher_layer_mapping": layer_mapping,
        "teacher_action_layers": len(teacher.action_expert.blocks),
        "teacher_action_hidden_dim": int(teacher.action_expert.hidden_dim),
        "teacher_action_heads": int(teacher.action_expert.num_heads),
        "teacher_action_head_dim": int(teacher.action_expert.attn_head_dim),
    }
    contract_pass = bool(
        model_contract["action_tokens"] == 16
        and model_contract["action_dim"] == 14
        and model_contract["action_hidden_dim"] == 768
        and model_contract["action_ffn_dim"] == 3072
        and model_contract["action_heads"] == 16
        and model_contract["action_head_dim"] == 128
        and model_contract["action_layers"] == 12
        and model_contract["text_context_dim"] == 4096
        and model_contract["state_dim"] == 14
        and model_contract["num_registers"] == 0
        and model_contract["has_proprio_encoder"]
        and model_contract["has_action_branch_embedding"]
        and model_contract["all_blocks_have_cross_attention"]
        and layer_mapping == expected_mapping
        and layer_mapping == sorted(set(layer_mapping))
        and layer_mapping[0] == 1
        and layer_mapping[-1] == len(teacher.action_expert.blocks)
    )
    report["model_contract"] = {**model_contract, "passed": contract_pass}

    old_init = dict(initialization)
    slice_init = old_init.get("aha_action_structured_slice", {})
    init_pass = bool(
        old_init.get("scope") == "conditioning_only"
        and int(old_init.get("source_step", -1)) == 36000
        and not old_init.get("missing_target_keys")
        and not old_init.get("shape_mismatches")
        and old_init.get("effective_action_initialization")
        == "aha_structured_slice"
        and list(slice_init.get("teacher_layers", [])) == expected_mapping
        and old_init.get("aha_proprio_encoder", {}).get("source")
        == "frozen_teacher.proprio_encoder"
    )
    report["initialization"] = {**old_init, "passed": init_pass}

    dataset_started = time.perf_counter()
    train_dataset, val_dataset = build_datasets(cfg.data, cfg.model)
    if not 0 <= args.val_index < len(val_dataset):
        raise IndexError(f"Validation index {args.val_index} is outside the dataset")
    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    iterator = iter(loader)
    sample = next(iterator)
    for _ in range(args.val_index):
        sample = next(iterator)
    sample["stage1_chunk_index"] = torch.zeros(1, dtype=torch.long)
    report["data"] = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_is_full_scale": len(train_dataset) >= 1_000_000,
        "dataset_dirs": [str(value) for value in cfg.data.train.dataset_dirs],
        "num_frames": int(cfg.data.train.num_frames),
        "action_video_freq_ratio": int(cfg.data.train.action_video_freq_ratio),
    }
    report["timing_seconds"]["dataset_and_sample_load"] = (
        time.perf_counter() - dataset_started
    )

    model.train(True)
    student.zero_grad(set_to_none=True)
    teacher_parameters = list(teacher_adapter.parameters())
    teacher_frozen_before = all(
        not parameter.requires_grad for parameter in teacher_parameters
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(0)
    batch_started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        targets = teacher_adapter.prepare_batch(sample, tiled=False)
        training_adapter = model.efficient_training_adapter
        if training_adapter is None:
            raise RuntimeError("AHA current-conditioning training adapter is absent")
        targets = training_adapter.prepare_batch(sample, targets)
        outputs = _student_forward(model, targets)
        loss, tensor_terms = stage1_distillation_loss(
            outputs, targets, model.loss_config
        )
    torch.cuda.synchronize(0)
    forward_seconds = time.perf_counter() - batch_started
    loss.backward()
    torch.cuda.synchronize(0)
    backward_seconds = time.perf_counter() - batch_started - forward_seconds

    sigma = targets.sigma.to(
        device=targets.noisy_action.device, dtype=targets.noisy_action.dtype
    )
    reconstructed_noisy = (
        targets.ground_truth_action
        + sigma[:, None, None] * targets.teacher_velocity
    )
    perfect_denoised = (
        targets.noisy_action
        - sigma[:, None, None] * targets.teacher_velocity
    )
    noise_formula_error = float(
        (targets.noisy_action - reconstructed_noisy).float().abs().max().item()
    )
    flow_reconstruction_error = float(
        (perfect_denoised - targets.ground_truth_action)
        .float()
        .abs()
        .max()
        .item()
    )
    timestep_error = float(
        (
            targets.action_t.float()
            - targets.sigma.float() * float(student.config.num_train_timesteps)
        )
        .abs()
        .max()
        .item()
    )
    expected_state = sample["proprio"][:, 0].to(
        device=targets.initial_state.device, dtype=targets.initial_state.dtype
    )
    state_alignment_error = float(
        (targets.initial_state - expected_state).float().abs().max().item()
    )

    modules = {
        "query_encoder": student.query_encoder,
        "kv_editor": student.cache_editor,
        "action_encoder": student.action_expert.action_encoder,
        "text_embedding": student.action_expert.text_embedding,
        "proprio_encoder": student.proprio_encoder,
        "block_1_self_attention": student.action_expert.blocks[0].self_attn,
        "block_1_cross_attention": student.action_expert.blocks[0].cross_attn,
        "block_6": student.action_expert.blocks[5],
        "block_12": student.action_expert.blocks[11],
        "action_head": student.action_expert.head,
    }
    gradients = {
        name: module_grad_report(module) for name, module in modules.items()
    }
    gradients["action_branch_embedding"] = parameter_grad_report(
        student.action_branch_embedding
    )
    gradients_pass = all(
        item["finite"] and item["nonzero"] for item in gradients.values()
    )
    teacher_grad_none = all(
        parameter.grad is None for parameter in teacher_parameters
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_velocity = outputs["action_velocity"].detach()
        state_delta = torch.full_like(targets.initial_state, 0.05)
        state_velocity = _student_forward(
            model, targets, initial_state=targets.initial_state + state_delta
        )["action_velocity"]
        changed_context = targets.action_context.clone()
        changed_context[:, 0] = changed_context[:, 0] + 0.05
        context_velocity = _student_forward(
            model, targets, action_context=changed_context
        )["action_velocity"]
    state_sensitivity = float(
        (state_velocity - reference_velocity).float().abs().max().item()
    )
    context_sensitivity = float(
        (context_velocity - reference_velocity).float().abs().max().item()
    )

    terms = {
        key: float(value.detach().float().item())
        for key, value in tensor_terms.items()
    }
    losses_finite = bool(torch.isfinite(loss.detach()).item()) and all(
        np.isfinite(value) for value in terms.values()
    )
    dataflow_pass = bool(
        tuple(targets.noisy_action.shape) == (1, 16, 14)
        and tuple(targets.initial_state.shape) == (1, 14)
        and isinstance(targets.action_context, torch.Tensor)
        and targets.action_context.ndim == 3
        and targets.action_context.shape[-1] == 4096
        and isinstance(targets.action_context_mask, torch.Tensor)
        and noise_formula_error <= 2e-2
        and flow_reconstruction_error <= 2e-2
        # BF16 stores the shifted [0, 1000] timestep at roughly 2-4 point
        # intervals in the upper range. This tolerance only covers that
        # representational rounding; the sigma/noise identities are checked
        # independently above.
        and timestep_error <= 4.0
        and state_alignment_error <= 1e-6
        and state_sensitivity > 0.0
        and context_sensitivity > 0.0
    )
    report["dataflow"] = {
        "shapes": {
            "sample_action": tensor_shape(sample.get("action")),
            "sample_proprio": tensor_shape(sample.get("proprio")),
            "observation_tokens": tensor_shape(targets.observation_tokens),
            "video_kv_cache": tensor_shape(targets.video_kv_cache),
            "action_context": tensor_shape(targets.action_context),
            "action_context_mask": tensor_shape(targets.action_context_mask),
            "noisy_action": tensor_shape(targets.noisy_action),
            "initial_state": tensor_shape(targets.initial_state),
            "student_velocity": tensor_shape(outputs["action_velocity"]),
        },
        "chunk_index": targets.chunk_index.detach().cpu().tolist(),
        "noise_formula_max_abs_error": noise_formula_error,
        "perfect_flow_reconstruction_max_abs_error": flow_reconstruction_error,
        "shifted_timestep_max_abs_error": timestep_error,
        "current_state_alignment_max_abs_error": state_alignment_error,
        "state_input_sensitivity_max_abs": state_sensitivity,
        "task_context_sensitivity_max_abs": context_sensitivity,
        "passed": dataflow_pass,
    }
    report["loss"] = {"total": float(loss.detach().float().item()), **terms}
    report["predictions"] = {
        "velocity": tensor_summary(outputs["action_velocity"]),
        "target_velocity": tensor_summary(targets.teacher_velocity),
        "ground_truth_action": tensor_summary(targets.ground_truth_action),
    }
    report["gradients"] = gradients
    report["teacher_frozen"] = {
        "all_requires_grad_false": teacher_frozen_before,
        "all_grad_none_after_backward": teacher_grad_none,
    }
    report["timing_seconds"].update(
        {"forward": forward_seconds, "backward": backward_seconds}
    )
    report["memory_mib"].update(
        {
            "batch_peak_allocated": torch.cuda.max_memory_allocated(0) / 1024**2,
            "batch_peak_reserved": torch.cuda.max_memory_reserved(0) / 1024**2,
        }
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        noisy_action=targets.noisy_action.detach().float().cpu().numpy(),
        ground_truth_action=targets.ground_truth_action.detach().float().cpu().numpy(),
        target_velocity=targets.teacher_velocity.detach().float().cpu().numpy(),
        student_velocity=outputs["action_velocity"].detach().float().cpu().numpy(),
        sigma=targets.sigma.detach().float().cpu().numpy(),
        initial_state=targets.initial_state.detach().float().cpu().numpy(),
        chunk_index=targets.chunk_index.detach().cpu().numpy(),
    )
    report["prediction_artifact"] = {
        "path": str(predictions_path),
        "sha256": sha256(predictions_path),
    }

    passed = bool(
        contract_pass
        and init_pass
        and report["data"]["train_is_full_scale"]
        and dataflow_pass
        and losses_finite
        and gradients_pass
        and teacher_frozen_before
        and teacher_grad_none
    )
    report["checks"] = {
        "model_contract": contract_pass,
        "initialization": init_pass,
        "full_scale_data": report["data"]["train_is_full_scale"],
        "dataflow": dataflow_pass,
        "finite_losses": losses_finite,
        "finite_nonzero_required_gradients": gradients_pass,
        "teacher_frozen": teacher_frozen_before and teacher_grad_none,
    }
    report["passed"] = passed
    report["status"] = "passed" if passed else "failed"
    atomic_json(output_path, report)
    if not passed:
        raise RuntimeError("AHA-aligned Stage 1 launch audit failed")


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()
    report: dict[str, Any] = {}
    try:
        run_preflight(args, report)
    except Exception as error:
        report["passed"] = False
        report["status"] = "failed"
        report["fatal_error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        atomic_json(output_path, report)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
