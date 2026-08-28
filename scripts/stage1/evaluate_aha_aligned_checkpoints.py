#!/usr/bin/env python3
"""Compare aligned-AHA checkpoints on fixed actions, context, and noise."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ahawam.runtime import build_datasets
from ahawam.trainer import Wan22Trainer
from ahawam.utils import misc


@dataclass(frozen=True)
class FixedTarget:
    index: int
    seed: int
    observation_tokens: torch.Tensor
    video_kv_cache: tuple[dict[str, torch.Tensor], ...]
    observation_mask: torch.Tensor
    initial_state: torch.Tensor
    action_context: torch.Tensor
    action_context_mask: torch.Tensor
    ground_truth_action: torch.Tensor
    initial_noise: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--indices", default="0,2321,2195,58053,43702,48640")
    parser.add_argument("--denoise-steps", default="1,2,5,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flow-shift", type=float, default=5.0)
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


def git_info() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    return {
        "root": str(root),
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def action_metrics(
    prediction: torch.Tensor, ground_truth: torch.Tensor
) -> dict[str, float]:
    prediction = prediction.detach().float()
    ground_truth = ground_truth.detach().float()
    temporal_delta = prediction[:, 1:] - prediction[:, :-1]
    return {
        "ground_truth_mse": float(torch.mean((prediction - ground_truth) ** 2).item()),
        "prediction_abs_mean": float(prediction.abs().mean().item()),
        "prediction_abs_max": float(prediction.abs().max().item()),
        "prediction_std": float(prediction.std(unbiased=False).item()),
        "temporal_delta_rms": float(
            torch.sqrt(torch.mean(temporal_delta.square())).item()
        ),
    }


def rms_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((left.float() - right.float()).square())).item())


def aggregate(rows: Sequence[Mapping[str, Any]], denoise_steps: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for denoise_step in denoise_steps:
        step_key = str(denoise_step)
        metric_names = sorted(rows[0]["denoise"][step_key])
        result[step_key] = {}
        for metric_name in metric_names:
            values = [float(row["denoise"][step_key][metric_name]) for row in rows]
            result[step_key][metric_name] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    for metric_name in ("state_swap_rms", "task_context_swap_rms"):
        values = [float(row[metric_name]) for row in rows]
        result[metric_name] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
    return result


@torch.inference_mode()
def prepare_targets(
    model: torch.nn.Module,
    val_dataset: Any,
    indices: Sequence[int],
    seed: int,
) -> list[FixedTarget]:
    targets: list[FixedTarget] = []
    parameter = next(model.student.parameters())
    for position, index in enumerate(indices):
        sample_seed = seed + position
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        np.random.seed(sample_seed)
        sample = Wan22Trainer._to_batched_eval_sample(val_dataset[index])
        sample["stage1_chunk_index"] = torch.zeros(1, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prepared = model.teacher_adapter.prepare_batch(sample, tiled=False)
        if not isinstance(prepared.action_context, torch.Tensor):
            raise RuntimeError("Aligned evaluation requires task context")
        if not isinstance(prepared.action_context_mask, torch.Tensor):
            raise RuntimeError("Aligned evaluation requires a task context mask")
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(seed * 1000 + position)
        noise = torch.randn(
            prepared.ground_truth_action.shape,
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        targets.append(
            FixedTarget(
                index=int(index),
                seed=sample_seed,
                observation_tokens=prepared.observation_tokens,
                video_kv_cache=tuple(prepared.video_kv_cache),
                observation_mask=prepared.observation_mask,
                initial_state=prepared.initial_state,
                action_context=prepared.action_context,
                action_context_mask=prepared.action_context_mask,
                ground_truth_action=prepared.ground_truth_action,
                initial_noise=noise,
            )
        )
    return targets


@torch.inference_mode()
def generate(
    student: torch.nn.Module,
    target: FixedTarget,
    *,
    denoise_steps: int,
    flow_shift: float,
    initial_state: torch.Tensor | None = None,
    action_context: torch.Tensor | None = None,
    action_context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = student.generate(
            observation_tokens=target.observation_tokens,
            video_kv_cache=target.video_kv_cache,
            initial_state=(
                target.initial_state if initial_state is None else initial_state
            ),
            observation_mask=target.observation_mask,
            action_context=(
                target.action_context if action_context is None else action_context
            ),
            action_context_mask=(
                target.action_context_mask
                if action_context_mask is None
                else action_context_mask
            ),
            num_steps=int(denoise_steps),
            flow_shift=float(flow_shift),
            initial_noise=target.initial_noise,
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError(
            f"Non-finite action for index={target.index}, steps={denoise_steps}"
        )
    return prediction.detach().float()


@torch.inference_mode()
def evaluate_variant(
    student: torch.nn.Module,
    targets: Sequence[FixedTarget],
    denoise_steps: Sequence[int],
    flow_shift: float,
) -> tuple[dict[str, Any], np.ndarray]:
    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    diagnostic_steps = max(denoise_steps)
    for position, target in enumerate(targets):
        step_predictions: list[np.ndarray] = []
        step_metrics: dict[str, Any] = {}
        diagnostic_prediction: torch.Tensor | None = None
        for step_count in denoise_steps:
            prediction = generate(
                student,
                target,
                denoise_steps=step_count,
                flow_shift=flow_shift,
            )
            step_metrics[str(step_count)] = action_metrics(
                prediction, target.ground_truth_action
            )
            step_predictions.append(prediction.cpu().numpy()[0])
            if step_count == diagnostic_steps:
                diagnostic_prediction = prediction
        assert diagnostic_prediction is not None

        swapped = targets[(position + 1) % len(targets)]
        state_swapped = generate(
            student,
            target,
            denoise_steps=diagnostic_steps,
            flow_shift=flow_shift,
            initial_state=swapped.initial_state,
        )
        context_swapped = generate(
            student,
            target,
            denoise_steps=diagnostic_steps,
            flow_shift=flow_shift,
            action_context=swapped.action_context,
            action_context_mask=swapped.action_context_mask,
        )
        rows.append(
            {
                "index": target.index,
                "seed": target.seed,
                "denoise": step_metrics,
                "state_swap_index": swapped.index,
                "state_swap_rms": rms_difference(
                    diagnostic_prediction, state_swapped
                ),
                "task_context_swap_index": swapped.index,
                "task_context_swap_rms": rms_difference(
                    diagnostic_prediction, context_swapped
                ),
            }
        )
        predictions.append(np.stack(step_predictions))
    return {
        "samples": rows,
        "aggregate": aggregate(rows, denoise_steps),
    }, np.stack(predictions)


def compare(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    denoise_steps: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step_count in denoise_steps:
        step_key = str(step_count)
        result[step_key] = {}
        for metric_name in reference["aggregate"][step_key]:
            before = float(reference["aggregate"][step_key][metric_name]["mean"])
            after = float(candidate["aggregate"][step_key][metric_name]["mean"])
            result[step_key][metric_name] = {
                "initialized_mean": before,
                "candidate_mean": after,
                "absolute_change": after - before,
                "relative_change": (after - before) / before if before else None,
                "improved_samples": sum(
                    int(
                        float(candidate_row["denoise"][step_key][metric_name])
                        < float(reference_row["denoise"][step_key][metric_name])
                    )
                    for reference_row, candidate_row in zip(
                        reference["samples"], candidate["samples"], strict=True
                    )
                ),
                "sample_count": len(reference["samples"]),
            }
    return result


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoints = [Path(value).resolve() for value in args.checkpoints]
    output_path = Path(args.output).resolve()
    predictions_path = Path(args.predictions).resolve()
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    denoise_steps = [
        int(value) for value in args.denoise_steps.split(",") if value.strip()
    ]
    if len(indices) < 2:
        raise ValueError("At least two fixed-panel samples are required")
    if not denoise_steps or any(value <= 0 for value in denoise_steps):
        raise ValueError("Positive denoise steps are required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    if str(cfg.model.student_config.action_architecture) != "aha_aligned":
        raise ValueError("This evaluator requires action_architecture=aha_aligned")
    misc.register_work_dir(str(output_path.parent / ".aligned-panel-work"))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
    _, val_dataset = build_datasets(cfg.data, cfg.model)
    if any(index < 0 or index >= len(val_dataset) for index in indices):
        raise IndexError(f"Fixed panel exceeds validation length {len(val_dataset)}")
    model.eval()
    targets = prepare_targets(model, val_dataset, indices, args.seed)

    variant_names = ["initialized"]
    variant_steps: list[int | None] = [None]
    reports: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    initialized, initialized_predictions = evaluate_variant(
        model.student, targets, denoise_steps, args.flow_shift
    )
    reports.append(initialized)
    arrays.append(initialized_predictions)

    checkpoint_records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        payload = model.load_checkpoint(str(checkpoint))
        config = payload.get("student_config", {})
        if config.get("action_architecture") != "aha_aligned":
            raise ValueError(f"Checkpoint is not AHA-aligned: {checkpoint}")
        variant, prediction_array = evaluate_variant(
            model.student, targets, denoise_steps, args.flow_shift
        )
        variant_names.append(checkpoint.stem)
        variant_steps.append(int(payload["step"]))
        reports.append(variant)
        arrays.append(prediction_array)
        checkpoint_records.append(
            {
                "name": checkpoint.stem,
                "path": str(checkpoint),
                "sha256": sha256(checkpoint),
                "step": int(payload["step"]),
            }
        )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        predictions=np.stack(arrays),
        ground_truth=np.stack(
            [target.ground_truth_action.float().cpu().numpy()[0] for target in targets]
        ),
        initial_noise=np.stack(
            [target.initial_noise.float().cpu().numpy()[0] for target in targets]
        ),
        variant_names=np.asarray(variant_names, dtype="U32"),
        variant_steps=np.asarray(
            [-1 if value is None else value for value in variant_steps], dtype=np.int64
        ),
        denoise_steps=np.asarray(denoise_steps, dtype=np.int64),
        validation_indices=np.asarray(indices, dtype=np.int64),
    )
    variants = [
        {
            "name": name,
            "checkpoint_step": step,
            "metrics": report,
            "vs_initialized": compare(initialized, report, denoise_steps),
        }
        for name, step, report in zip(
            variant_names[1:], variant_steps[1:], reports[1:], strict=True
        )
    ]
    report = {
        "schema_version": 1,
        "purpose": "fixed-input aligned-AHA checkpoint direction check",
        "source": git_info(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "validation_indices": indices,
        "seed": args.seed,
        "noise_seed_rule": "seed * 1000 + zero-based panel position",
        "chunk_index": 0,
        "flow_shift": args.flow_shift,
        "denoise_steps": denoise_steps,
        "checkpoints": checkpoint_records,
        "initialized": initialized,
        "variants": variants,
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256(predictions_path),
        },
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
        "duration_seconds": time.perf_counter() - started,
        "finite": bool(np.isfinite(np.stack(arrays)).all()),
    }
    atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "predictions": str(predictions_path),
                "finite": report["finite"],
                "variants": variant_names,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
