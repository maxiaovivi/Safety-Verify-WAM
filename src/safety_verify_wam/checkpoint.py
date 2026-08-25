from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import public_config, resolve_project_path
from .models.safety_verifier import (
    SafetyVerifyWAM,
    load_delta_state_dict,
    trainable_state_dict,
)


SCHEMA_VERSION = 3


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = resolve_project_path(path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported safety checkpoint: {checkpoint_path}")
    if not isinstance(payload.get("delta_state"), dict):
        raise RuntimeError(f"Safety checkpoint has no delta_state: {checkpoint_path}")
    return payload


def save_checkpoint(
    path: str | Path,
    model: SafetyVerifyWAM,
    config: dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    checkpoint_path = resolve_project_path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics),
        "config": public_config(config),
        "risk_head_config": model.risk_head.config.to_dict(),
        "delta_state": trainable_state_dict(model),
        "trainable_parameters": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)
    return checkpoint_path


def restore_model(model: SafetyVerifyWAM, payload: dict[str, Any]) -> None:
    expected = model.risk_head.config.to_dict()
    actual = payload.get("risk_head_config")
    if actual != expected:
        raise RuntimeError(
            f"Risk head configuration mismatch: expected {expected}, checkpoint has {actual}"
        )
    load_delta_state_dict(model, payload["delta_state"])


def inference_config(
    runtime_config: dict[str, Any], checkpoint_payload: dict[str, Any]
) -> dict[str, Any]:
    trained = checkpoint_payload.get("config")
    if not isinstance(trained, dict):
        raise RuntimeError("Safety checkpoint is missing its training configuration")
    merged = dict(trained)
    merged["device"] = runtime_config.get("device", trained.get("device", "cuda"))
    merged["model"] = dict(trained.get("model", {}))
    merged_backbone = dict(trained.get("backbone", {}))
    for key in ("source_root", "base_checkpoint", "wan_root", "vae_path", "precision"):
        if key in runtime_config.get("backbone", {}):
            merged_backbone[key] = runtime_config["backbone"][key]
    merged_backbone["randomize_training_noise"] = False
    merged["backbone"] = merged_backbone
    return merged
