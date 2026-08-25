from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


CLASS_NAMES = ("safe", "boundary", "risk")


def _class_weights(
    value: Any,
    device: torch.device | str,
) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, dict):
        unknown = sorted(set(value) - set(CLASS_NAMES))
        if unknown:
            raise ValueError(f"Unknown class-weight keys: {unknown}")
        weights = [float(value.get(name, 1.0)) for name in CLASS_NAMES]
    elif isinstance(value, (list, tuple)) and len(value) == len(CLASS_NAMES):
        weights = [float(item) for item in value]
    else:
        raise ValueError(
            "class_weights must be [safe,boundary,risk] or a mapping with those keys"
        )
    tensor = torch.tensor(weights, device=device, dtype=torch.float32)
    if not torch.isfinite(tensor).all() or (tensor <= 0).any():
        raise ValueError("class_weights must contain finite positive values")
    return tensor


def safety_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Three-class chunk loss with optional per-step and event-type supervision."""

    chunk_weight = float(config.get("chunk_weight", 1.0))
    step_weight = float(config.get("step_weight", 0.0))
    type_weight = float(config.get("type_weight", 0.0))
    class_weights = _class_weights(config.get("class_weights"), device)
    target = batch["risk"].to(device, non_blocking=True).long()
    chunk_loss = F.cross_entropy(
        outputs["class_logits"].float(),
        target,
        weight=class_weights,
    )
    zero = outputs["class_logits"].sum() * 0.0

    step_loss = zero
    if step_weight > 0:
        step_target = batch["risk_steps"].to(device, non_blocking=True).long()
        valid = step_target != -100
        if valid.any():
            step_loss = F.cross_entropy(
                outputs["step_class_logits"][valid].float(),
                step_target[valid],
                weight=class_weights,
            )

    type_loss = zero
    if type_weight > 0:
        if "risk_type_logits" not in outputs:
            raise ValueError("type_weight > 0 but model.num_risk_types is zero")
        type_target = batch["risk_type"].to(device, non_blocking=True)
        available = type_target != -100
        if available.any():
            type_loss = F.cross_entropy(
                outputs["risk_type_logits"][available].float(),
                type_target[available],
            )

    total = chunk_weight * chunk_loss + step_weight * step_loss + type_weight * type_loss
    return total, {
        "loss": total.detach(),
        "chunk_class_loss": chunk_loss.detach(),
        "step_class_loss": step_loss.detach(),
        "type_loss": type_loss.detach(),
    }
