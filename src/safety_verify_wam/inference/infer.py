from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import inference_config, load_checkpoint, restore_model
from ..config import apply_overrides, load_config
from ..models.efficient_wam import build_model
from ..models.safety_verifier import SafetyVerifyWAM
from ..training.data import load_action_tensor, load_image_tensor, validate_action_tensor


class SafetyVerifierRuntime:
    """Reusable inference API. It intentionally accepts no task text or robot state."""

    def __init__(self, model: SafetyVerifyWAM, device: torch.device) -> None:
        self.model = model.eval()
        self.device = device

    @classmethod
    def from_config(cls, runtime_config: dict[str, Any]) -> "SafetyVerifierRuntime":
        payload = load_checkpoint(runtime_config["checkpoint"])
        model_config = inference_config(runtime_config, payload)
        device = torch.device(model_config.get("device", "cuda"))
        model = build_model(model_config, device=device)
        restore_model(model, payload)
        return cls(model, device)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, action: torch.Tensor) -> dict[str, Any]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        outputs = self.model.predict(
            image.to(self.device, non_blocking=True),
            action.to(self.device, non_blocking=True),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        probability = float(outputs["unsafe_probability"][0, 0].cpu().item())
        is_unsafe = bool(outputs["is_unsafe"][0, 0].cpu().item())
        result: dict[str, Any] = {
            "unsafe_probability": probability,
            "is_unsafe": is_unsafe,
            "decision": "unsafe" if is_unsafe else "safe",
            "threshold": self.model.unsafe_threshold,
            "step_probabilities": torch.sigmoid(outputs["step_logits"])[0]
            .float()
            .cpu()
            .tolist(),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        if "risk_type_logits" in outputs:
            result["risk_type_probabilities"] = (
                torch.softmax(outputs["risk_type_logits"], dim=-1)[0]
                .float()
                .cpu()
                .tolist()
            )
        return result


def _load_cli_inputs(
    image_path: str,
    action_path: str,
    input_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    image = load_image_tensor(
        Path(image_path).expanduser().resolve(),
        input_config.get("image_size", (384, 320)),
        input_config.get("image_size_policy", "error"),
    )
    action_file = Path(action_path).expanduser().resolve()
    action = load_action_tensor(str(action_file), action_file.parent)
    validate_action_tensor(
        action,
        input_config.get("action_shape", (16, 14)),
        bool(input_config.get("require_normalized_actions", True)),
        float(input_config.get("normalized_action_limit", 1.05)),
    )
    return image, action


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one image-action candidate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--action", required=True, help="Normalized [16,14] .npy file")
    parser.add_argument("--output", default=None, help="Optional result JSON path")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    image, action = _load_cli_inputs(args.image, args.action, config.get("input", {}))
    runtime = SafetyVerifierRuntime.from_config(config)
    result = runtime.predict(image, action)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
