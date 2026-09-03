from __future__ import annotations

import torch
import torch.nn.functional as F


def portable_safety_loss(
    outputs: dict[str, torch.Tensor],
    chunk_target: torch.Tensor,
    *,
    step_target: torch.Tensor | None = None,
    chunk_weight: float = 1.0,
    step_weight: float = 1.0,
    class_weights: torch.Tensor | None = None,
    step_class_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Two-class loss for the host-neutral safety core."""

    if chunk_weight < 0 or step_weight < 0 or chunk_weight + step_weight <= 0:
        raise ValueError("Loss weights must be nonnegative and not both zero")
    chunk_target = chunk_target.to(
        device=outputs["class_logits"].device, dtype=torch.long
    )
    chunk_loss = F.cross_entropy(
        outputs["class_logits"].float(),
        chunk_target,
        weight=class_weights,
    )
    step_loss = outputs["class_logits"].sum() * 0.0
    if step_target is not None and step_weight > 0:
        step_target = step_target.to(
            device=outputs["step_class_logits"].device, dtype=torch.long
        )
        valid = (step_target != ignore_index) & outputs["action_mask"]
        if bool(valid.any()):
            step_loss = F.cross_entropy(
                outputs["step_class_logits"][valid].float(),
                step_target[valid],
                weight=(
                    class_weights
                    if step_class_weights is None
                    else step_class_weights
                ),
            )
    total = chunk_weight * chunk_loss + step_weight * step_loss
    return total, {
        "loss": total.detach(),
        "chunk_class_loss": chunk_loss.detach(),
        "step_class_loss": step_loss.detach(),
    }
