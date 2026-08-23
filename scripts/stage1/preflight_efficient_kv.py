from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from safety_verify_wam.stage1.aha_teacher import AHAOVCRSTeacherBatch
from safety_verify_wam.stage1.efficient_training import EfficientStudentTrainingAdapter
from safety_verify_wam.stage1.ovcr_s import OVCRSActionGenerator, OVCRSConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that neutral OVCR-S reproduces the Efficient action expert."
    )
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aha-stats", required=True)
    parser.add_argument("--efficient-stats", required=True)
    parser.add_argument("--efficient-python-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-video-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Efficient checkpoint is not a mapping: {path}")
    return payload


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    config = OVCRSConfig()
    student = OVCRSActionGenerator(config)
    missing, mismatched = student.load_efficient_action_expert(
        load_checkpoint(args.checkpoint), strict=True
    )
    student.to(device=args.device, dtype=torch.bfloat16).eval()
    provider = EfficientStudentTrainingAdapter(
        student_config=config,
        deploy_config_path=args.deploy_config,
        aha_dataset_stats_path=args.aha_stats,
        efficient_dataset_stats_path=args.efficient_stats,
        device=args.device,
        efficient_python_root=args.efficient_python_root,
        num_video_steps=args.num_video_steps,
    )

    batch_size = 1
    teacher_action = torch.randn(
        batch_size, config.action_chunk_size, config.action_dim
    )
    targets = AHAOVCRSTeacherBatch(
        noisy_action=teacher_action.clone(),
        action_t=torch.full((batch_size,), 500.0),
        sigma=torch.full((batch_size,), 0.5),
        teacher_velocity=torch.zeros_like(teacher_action),
        teacher_action=teacher_action,
        ground_truth_action=teacher_action.clone(),
        initial_state=torch.randn(batch_size, config.state_dim),
        reference_velocity=None,
        observation_tokens=torch.zeros(
            batch_size, 1, 1, config.observation_dim
        ),
        observation_mask=torch.ones(batch_size, 1, 1, dtype=torch.bool),
        video_kv_cache=tuple(),
        teacher_queries=torch.zeros(
            batch_size, 1, config.num_queries, config.query_dim
        ),
        teacher_editor_trace={},
        teacher_action_responses={},
        action_is_pad=None,
        chunk_index=torch.zeros(batch_size, dtype=torch.long),
        anchor_step=torch.zeros(batch_size, dtype=torch.long),
    )
    sample = {
        "video": torch.rand(batch_size, 3, 9, 384, 320) * 2 - 1,
        "action": torch.zeros(batch_size, 64, config.action_dim),
        "context": torch.zeros(batch_size, 128, 4096, dtype=torch.bfloat16),
    }
    prepared = provider.prepare_batch(sample, targets)
    with torch.inference_mode():
        outputs = student(
            noisy_action=prepared.noisy_action,
            action_t=prepared.action_t,
            initial_state=prepared.initial_state,
            observation_tokens=prepared.observation_tokens,
            observation_mask=prepared.observation_mask,
            video_kv_cache=prepared.video_kv_cache,
            return_trace=False,
        )
    prediction = outputs["action_velocity"].float()
    reference = prepared.reference_velocity
    if reference is None:
        raise RuntimeError("Efficient provider returned no reference velocity")
    reference = reference.float()
    error = prediction - reference
    reference_rms = reference.square().mean().sqrt()
    report = {
        "missing_action_tensors": missing,
        "mismatched_action_tensors": mismatched,
        "prediction_shape": list(prediction.shape),
        "observation_shape": list(prepared.observation_tokens.shape),
        "cache_layers": len(prepared.video_kv_cache),
        "cache_tokens": int(prepared.video_kv_cache[0]["k"].shape[1]),
        "max_abs_error": float(error.abs().max()),
        "rmse": float(error.square().mean().sqrt()),
        "relative_rmse": float(error.square().mean().sqrt() / reference_rms.clamp_min(1e-8)),
        "cosine": float(
            F.cosine_similarity(prediction.flatten(1), reference.flatten(1)).mean()
        ),
        "finite": bool(torch.isfinite(prediction).all()),
        "peak_allocated_gib": (
            float(torch.cuda.max_memory_allocated() / 1024**3)
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
