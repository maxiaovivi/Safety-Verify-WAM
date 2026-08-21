from __future__ import annotations

import argparse
import json
import logging
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..checkpoint import load_checkpoint, restore_model, save_checkpoint
from ..config import apply_overrides, load_config, resolve_project_path
from ..models.efficient_wam import build_model
from .data import SafetyManifestDataset, model_inputs_from_batch, safety_collate_fn
from .losses import safety_loss


LOGGER = logging.getLogger("safety_verify_wam.training.train")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp(config: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype, Any]:
    name = str(config["training"].get("amp", "bfloat16")).lower()
    if device.type != "cuda" or name in {"none", "float32"}:
        return False, torch.float32, torch.amp.GradScaler("cuda", enabled=False)
    if name == "bfloat16":
        return True, torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False)
    if name == "float16":
        return True, torch.float16, torch.amp.GradScaler("cuda", enabled=True)
    raise ValueError(f"Unsupported amp mode: {name}")


def train(config: dict[str, Any]) -> None:
    _seed_everything(int(config.get("seed", 7)))
    device = torch.device(config.get("device", "cuda"))
    dataset = SafetyManifestDataset.from_config(config["dataset"], "train")
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config["dataset"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        collate_fn=safety_collate_fn,
    )
    model = build_model(config, device=device)
    resume_payload = None
    if config.get("resume_from"):
        resume_payload = load_checkpoint(config["resume_from"])
        restore_model(model, resume_payload)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-3)),
    )
    if resume_payload and "optimizer" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])
    amp_enabled, amp_dtype, scaler = _amp(config, device)
    start_epoch = int(resume_payload.get("epoch", -1)) + 1 if resume_payload else 0
    global_step = int(resume_payload.get("global_step", 0)) if resume_payload else 0
    output_dir = resolve_project_path(config.get("output_dir", "checkpoints/safety_verify_wam"))
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(config["training"].get("epochs", 1))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    log_every = max(1, int(config["training"].get("log_every_steps", 10)))
    save_every = max(1, int(config["training"].get("save_every_epochs", 1)))

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            inputs = model_inputs_from_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                outputs = model(**inputs)
                loss, terms = safety_loss(outputs, batch, config.get("loss", {}), device)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            running_loss += float(loss.detach().item())
            if global_step % log_every == 0:
                LOGGER.info(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            **{key: float(value.item()) for key, value in terms.items()},
                        }
                    )
                )
        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            save_checkpoint(
                output_dir / "latest.pt",
                model,
                config,
                epoch=epoch,
                global_step=global_step,
                metrics={"train_loss": running_loss / max(1, len(loader))},
                optimizer=optimizer,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Safety-Verify-WAM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(apply_overrides(load_config(args.config), args.overrides))


if __name__ == "__main__":
    main()
