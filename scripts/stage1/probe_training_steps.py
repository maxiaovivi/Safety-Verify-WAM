#!/usr/bin/env python3
"""Run a few real Stage 1 optimizer steps and record module-level gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from ahawam.runtime import build_datasets
from ahawam.utils import misc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def module_grad_report(module: torch.nn.Module) -> dict[str, Any]:
    squared_norm = 0.0
    tensor_count = 0
    finite = True
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        squared_norm += float(gradient.square().sum().item())
        tensor_count += 1
        finite = finite and bool(torch.isfinite(gradient).all().item())
    return {
        "norm": squared_norm**0.5,
        "gradient_tensor_count": tensor_count,
        "finite": finite,
        "nonzero": squared_norm > 0.0,
    }


def module_parameter_report(module: torch.nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "trainable_tensors": sum(parameter.requires_grad for parameter in parameters),
        "parameter_tensors": len(parameters),
    }


def git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("batch size and steps must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU")

    config_path = Path(args.config).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_samples = args.batch_size * args.steps
    indices = [int(value) for value in manifest["train_sample_indices"]]
    if len(indices) < required_samples:
        raise ValueError(
            f"Manifest has {len(indices)} train samples; {required_samples} are required"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "real-data multi-step gradient and optimizer probe",
        "status": "initializing",
        "source_commit": git_commit(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "prompt_manifest": str(manifest_path),
        "prompt_manifest_sha256": sha256(manifest_path),
        "batch_size": args.batch_size,
        "requested_steps": args.steps,
        "seed": args.seed,
        "sample_indices": indices[:required_samples],
        "gpu": torch.cuda.get_device_name(0),
        "rows": [],
    }
    atomic_json(output_path, report)

    misc.register_work_dir(str(output_path.parent / ".optimizer-probe-work"))
    initialized = time.perf_counter()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
    efficient_adapter = model.efficient_training_adapter
    shared_action_expert = bool(
        efficient_adapter is not None
        and efficient_adapter.efficient_model.action_expert
        is model.student.action_expert
    )
    report["shared_action_expert"] = shared_action_expert
    if efficient_adapter is not None and not shared_action_expert:
        raise RuntimeError(
            "Efficient-WAM and OVCR-S do not reference the same action expert"
        )
    train_dataset, _ = build_datasets(cfg.data, cfg.model)
    subset = Subset(train_dataset, indices[:required_samples])
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model.train(True)
    parameters = [parameter for parameter in model.student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )
    modules = {
        "query_encoder": model.student.query_encoder,
        "kv_editor": model.student.cache_editor,
        "kv_delta_up": model.student.cache_editor.delta_up,
        "state_encoder": model.student.action_expert.input_encoder.state_encoder,
        "action_block_1": model.student.action_expert.blocks[0],
        "action_block_6": model.student.action_expert.blocks[5],
        "action_block_12": model.student.action_expert.blocks[11],
        "action_decoder": model.student.action_expert.decoder,
    }
    report["module_parameters"] = {
        name: module_parameter_report(module) for name, module in modules.items()
    }
    report["frozen_student_parameter_names"] = [
        name
        for name, parameter in model.student.named_parameters()
        if not parameter.requires_grad
    ]
    report["model_and_dataset_init_seconds"] = time.perf_counter() - initialized
    report["trainable_parameters"] = sum(parameter.numel() for parameter in parameters)
    report["student_parameter_dtypes"] = sorted(
        {str(parameter.dtype) for parameter in parameters}
    )
    report["status"] = "running"
    atomic_json(output_path, report)

    try:
        for step, sample in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, metrics = model.training_loss(sample)
            torch.cuda.synchronize(0)
            forward_seconds = time.perf_counter() - started
            loss.backward()
            torch.cuda.synchronize(0)
            backward_seconds = time.perf_counter() - started - forward_seconds
            gradients = {
                name: module_grad_report(module) for name, module in modules.items()
            }
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize(0)
            total_seconds = time.perf_counter() - started
            optimizer_state_dtypes = sorted(
                {
                    str(value.dtype)
                    for state in optimizer.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                }
            )
            row = {
                "step": step,
                "loss": float(loss.detach().float().item()),
                "metrics": {key: float(value) for key, value in metrics.items()},
                "grad_norm_before_clip": float(grad_norm.detach().float().item()),
                "gradients": gradients,
                "optimizer_state_dtypes": optimizer_state_dtypes,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": total_seconds - forward_seconds - backward_seconds,
                "total_seconds": total_seconds,
                "samples_per_second": args.batch_size / total_seconds,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / (1024 * 1024),
                "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / (1024 * 1024),
            }
            row["finite"] = bool(np.isfinite(row["loss"])) and all(
                np.isfinite(value) for value in row["metrics"].values()
            ) and all(item["finite"] for item in gradients.values())
            report["rows"].append(row)
            atomic_json(output_path, report)
            print(json.dumps(row), flush=True)
            if step >= args.steps:
                break
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        atomic_json(output_path, report)
        raise

    report["status"] = "ok"
    report["all_finite"] = all(row["finite"] for row in report["rows"])
    report["all_required_modules_nonzero_after_step_one"] = all(
        row["gradients"][name]["nonzero"]
        for row in report["rows"][1:]
        for name in modules
    )
    atomic_json(output_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
