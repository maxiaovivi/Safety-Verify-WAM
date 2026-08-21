from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def safety_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    chunk_weight = float(config.get("chunk_weight", 1.0))
    step_weight = float(config.get("step_weight", 0.0))
    type_weight = float(config.get("type_weight", 0.0))
    target = batch["risk"].to(device, non_blocking=True)
    positive_weight = config.get("positive_weight")
    pos_weight = (
        torch.tensor([float(positive_weight)], device=device)
        if positive_weight is not None
        else None
    )
    chunk_loss = F.binary_cross_entropy_with_logits(
        outputs["unsafe_logit"].float(), target.float(), pos_weight=pos_weight
    )
    zero = outputs["unsafe_logit"].sum() * 0.0
    step_loss = zero
    if step_weight > 0:
        available = batch["risk_steps_available"].to(device, non_blocking=True)
        if available.any():
            step_target = batch["risk_steps"].to(device, non_blocking=True)
            step_loss = F.binary_cross_entropy_with_logits(
                outputs["step_logits"][available].float(),
                step_target[available].float(),
            )
    type_loss = zero
    if type_weight > 0:
        if "risk_type_logits" not in outputs:
            raise ValueError("type_weight > 0 but model.num_risk_types is zero")
        type_target = batch["risk_type"].to(device, non_blocking=True)
        available = type_target != -100
        if available.any():
            type_loss = F.cross_entropy(
                outputs["risk_type_logits"][available].float(), type_target[available]
            )
    total = chunk_weight * chunk_loss + step_weight * step_loss + type_weight * type_loss
    return total, {
        "loss": total.detach(),
        "chunk_loss": chunk_loss.detach(),
        "step_loss": step_loss.detach(),
        "type_loss": type_loss.detach(),
    }
