from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive an Efficient-K/V Stage 1 config from a resolved AHA config."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--aha-stats", required=True)
    parser.add_argument("--efficient-stats", required=True)
    parser.add_argument("--efficient-python-root", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument(
        "--student-parameter-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-video-steps", type=int, default=2)
    parser.add_argument(
        "--action-noise-sampling",
        choices=("aha_anchors", "uniform_shifted"),
        default="aha_anchors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError(f"Resolved base config has no model mapping: {base_path}")

    payload.update(
        {
            "output_dir": str(Path(args.run_output).expanduser()),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "learning_rate": float(args.learning_rate),
            "max_steps": int(args.max_steps),
            "log_every": 1,
            "save_every": max(1, int(args.max_steps)),
            "eval_every": max(1, int(args.max_steps)),
            "seed": int(args.seed),
            "resume": None,
            "init_checkpoint": None,
        }
    )
    wandb = payload.setdefault("wandb", {})
    if isinstance(wandb, dict):
        wandb["enabled"] = False
        wandb["name"] = Path(args.run_output).name

    model = payload["model"]
    student = model.setdefault("student_config", {})
    student["state_dim"] = 14
    model["student_parameter_dtype"] = args.student_parameter_dtype
    model["loss_config"] = {
        "velocity_weight": 1.0,
        "teacher_action_weight": 1.0,
        "ground_truth_action_weight": 0.25,
        "preservation_weight": 0.25,
        "query_weight": 0.0,
        "route_weight": 0.0,
        "delta_weight": 0.0,
        "response_weight": 0.0,
        "eps": 1.0e-6,
    }
    model["efficient_conditioning"] = {
        "deploy_config_path": str(Path(args.deploy_config).expanduser()),
        "aha_dataset_stats_path": str(Path(args.aha_stats).expanduser()),
        "efficient_dataset_stats_path": str(
            Path(args.efficient_stats).expanduser()
        ),
        "efficient_python_root": str(
            Path(args.efficient_python_root).expanduser()
        ),
        "num_video_steps": int(args.num_video_steps),
        "num_video_frames": 8,
        "action_sigma_shift": 5.0,
        "video_sigma_shift": 5.0,
        "action_noise_sampling": args.action_noise_sampling,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
