#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from safety_verify_wam.portable import (
    SafetyBatch,
    extract_future_value_tokens,
    load_multidomain_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _required_tensor(payload: Any, key: str) -> torch.Tensor:
    if not isinstance(payload, dict) or key not in payload:
        available = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise KeyError(f"{key} missing from trajectory payload; available={available}")
    value = torch.as_tensor(payload[key], dtype=torch.float32)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(f"{key} must be [T,14], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def _language_tensor(payload: Any) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif (
        isinstance(payload, (list, tuple))
        and len(payload) == 1
        and isinstance(payload[0], torch.Tensor)
    ):
        tensor = payload[0]
    elif isinstance(payload, dict):
        for key in ("text_embeddings", "embedding", "embeddings", "context"):
            if isinstance(payload.get(key), torch.Tensor):
                tensor = payload[key]
                break
        else:
            raise KeyError(f"No text tensor in language payload: {sorted(payload)}")
    else:
        raise TypeError(f"Unsupported language payload: {type(payload).__name__}")
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Text embedding must be [L,D], got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _read_rgb_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _portable_views(frame: np.ndarray, size: int = 128) -> torch.Tensor:
    height, width = frame.shape[:2]
    head_height = (height * 2) // 3
    half_width = width // 2
    views = (
        frame[:head_height],
        frame[head_height:, :half_width],
        frame[head_height:, half_width:],
    )
    resized = [
        cv2.resize(view, (size, size), interpolation=cv2.INTER_AREA)
        for view in views
    ]
    return torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).contiguous()


def _scalar_timestep(timestep: torch.Tensor) -> torch.Tensor:
    values = timestep.reshape(-1)
    if not torch.equal(values, values[:1].expand_as(values)):
        raise ValueError("Video scheduler requires one shared batch timestep")
    return values[0]


def _integer_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [int(item) for item in value]


def _initialize_video_latent(
    model: torch.nn.Module,
    current_frame: torch.Tensor,
    *,
    num_video_frames: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_model = model.compact_wan.video_model
    dtype = video_model.precision
    current_frame = current_frame.to(device=next(model.parameters()).device, dtype=dtype)
    condition_latent = model.compact_wan.encode_video(current_frame)
    batch, channels = condition_latent.shape[:2]
    if not model.compact_wan.is_multiscale:
        raise RuntimeError("This experiment requires Efficient-WAM multiscale futures")
    future_size = model.compact_wan.config.future_video_size
    if future_size is None:
        raise RuntimeError("Efficient-WAM config has no future_video_size")
    future_height, future_width = (int(value) // 16 for value in future_size)
    future_latent = torch.randn(
        (
            batch,
            channels,
            num_video_frames // 4,
            future_height,
            future_width,
        ),
        device=condition_latent.device,
        dtype=dtype,
        generator=generator,
    )
    return condition_latent, future_latent


@torch.inference_mode()
def _future_tokens(
    *,
    model: torch.nn.Module,
    scheduler: Any,
    current_frame: torch.Tensor,
    text: torch.Tensor,
    state: torch.Tensor,
    action: torch.Tensor,
    num_video_frames: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    condition_latent, future_latent = _initialize_video_latent(
        model,
        current_frame,
        num_video_frames=num_video_frames,
        generator=generator,
    )
    parameter = next(model.action_expert.parameters())
    state = state.to(device=parameter.device, dtype=parameter.dtype)
    action = action.to(device=parameter.device, dtype=parameter.dtype)
    action_t = torch.zeros((action.shape[0],), device=parameter.device, dtype=parameter.dtype)
    timesteps = scheduler.timesteps.to(
        device=parameter.device,
        dtype=model.compact_wan.video_model.precision,
    )
    cache: dict[str, Any] | None = None
    for step, raw_timestep in enumerate(timesteps):
        video_t = raw_timestep.expand(action.shape[0])
        output = model(
            {
                "video_t": video_t,
                "initial_state": state,
                "noisy_actions": action,
                "action_t": action_t,
                "text_embeddings": [text.to(parameter.device)],
                "condition_latent": condition_latent,
                "future_latent": future_latent,
                "return_video_cache": True,
            }
        )
        cache = output.get("video_cache")
        if cache is None:
            raise RuntimeError("Efficient-WAM did not return its video cache")
        if step + 1 < len(timesteps):
            future_latent = scheduler.step(
                output["video_pred"], _scalar_timestep(video_t), future_latent
            )
    assert cache is not None
    tokens = extract_future_value_tokens(cache, layer_index=-1)
    grid = cache["grid_sizes"]
    metadata = {
        "layer_index": -1,
        "token_shape": list(tokens.shape),
        "condition_seq_len": int(grid["condition_seq_len"]),
        "future_seq_len": int(grid["future_seq_len"]),
        "condition_grid_shape": _integer_list(grid["condition_grid_shape"]),
        "future_grid_shape": _integer_list(grid["future_grid_shape"]),
    }
    return tokens.cpu().to(torch.bfloat16), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--efficient-python-root", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument("--portable-checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-video-steps", type=int, default=2)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    slice_root = args.slice_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / "records"
    records_dir.mkdir(exist_ok=True)
    progress_path = output / "features.jsonl"
    completed = {
        row["window_id"]: row for row in _read_jsonl(progress_path)
    } if progress_path.exists() else {}

    sys.path.insert(0, str(args.efficient_python_root.expanduser().resolve()))
    model_loader = importlib.import_module("EfficientWAM.model_loader")
    scheduler_module = importlib.import_module("EfficientWAM.third_party.wan.utils.fm")
    image_module = importlib.import_module("EfficientWAM.utils.image_utils")
    config = model_loader.load_deploy_config(str(args.deploy_config.expanduser().resolve()))
    model = model_loader.build_model_from_config(config, device=args.device)
    model.eval()
    scheduler = scheduler_module.FlowMatchScheduler(
        shift=5.0, sigma_min=0.0, extra_one_step=True
    )
    scheduler.set_timesteps(num_inference_steps=args.num_video_steps, training=False)
    portable = load_multidomain_checkpoint(args.portable_checkpoint, map_location=args.device)
    portable.model.eval()
    profile = portable.profiles["bimanual_qpos14"]

    stats_payload = json.loads(args.stats.read_text())["robotwin_qpos"]
    efficient_mean = torch.tensor(stats_payload["mean"], dtype=torch.float32)
    efficient_std = torch.tensor(stats_payload["std"], dtype=torch.float32)
    if profile.state_mean is None or profile.action_mean is None:
        raise RuntimeError("Portable checkpoint has no state/action normalization")
    normalization_delta = max(
        float((efficient_mean - torch.tensor(profile.state_mean)).abs().max()),
        float((efficient_std - torch.tensor(profile.state_std)).abs().max()),
        float((efficient_mean - torch.tensor(profile.action_mean)).abs().max()),
        float((efficient_std - torch.tensor(profile.action_std)).abs().max()),
    )

    windows = _read_jsonl(slice_root / "windows.jsonl")
    started = time.monotonic()
    for index, window in enumerate(windows):
        window_id = str(window["window_id"])
        if window_id in completed and (output / completed[window_id]["record_path"]).is_file():
            continue
        paths = {key: slice_root / value for key, value in window["paths"].items()}
        trajectory = _torch_load(paths["trajectory_path"])
        all_states = _required_tensor(trajectory, "robot_state_qpos")
        all_actions = _required_tensor(trajectory, "action_target_qpos")
        condition = int(window["condition_frame_idx"])
        action_indices = torch.tensor(window["action_indices"], dtype=torch.long)
        state_physical = all_states[condition]
        action_physical = all_actions.index_select(0, action_indices)
        state_efficient = (state_physical - efficient_mean) / efficient_std
        action_efficient = (action_physical - efficient_mean) / efficient_std

        frame = _read_rgb_frame(paths["video_path"], condition)
        efficient_frame = image_module.resize_with_padding(frame, (384, 320))
        efficient_frame = torch.from_numpy(efficient_frame.astype(np.float32) / 255.0)
        efficient_frame = efficient_frame.mul(2.0).sub(1.0)
        efficient_frame = efficient_frame.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        views = _portable_views(frame)
        text = _language_tensor(_torch_load(paths["language_path"]))
        generator = torch.Generator(device=args.device).manual_seed(
            args.seed
            + int(
                hashlib.sha256(str(window["sample_id"]).encode()).hexdigest()[:8],
                16,
            )
        )
        future, future_metadata = _future_tokens(
            model=model,
            scheduler=scheduler,
            current_frame=efficient_frame,
            text=text,
            state=state_efficient.unsqueeze(0),
            action=action_efficient.unsqueeze(0),
            num_video_frames=args.num_video_frames,
            generator=generator,
        )

        state_portable = profile.normalize_state(state_physical).view(1, 1, -1).to(args.device)
        action_portable = profile.normalize_action(action_physical).unsqueeze(0).to(args.device)
        dt = 3.0 / float(json.loads(paths["label_path"].read_text())["temporal_safety"]["capture"]["physical_fps"])
        safety_batch = SafetyBatch(
            video=views.unsqueeze(0).unsqueeze(0).to(args.device).float().div(255.0),
            state=state_portable,
            action=action_portable,
            video_timestamps=torch.zeros((1, 1), device=args.device),
            state_timestamps=torch.zeros((1, 1), device=args.device),
            action_timestamps=(torch.arange(1, 17, device=args.device).float() * dt).unsqueeze(0),
            video_mask=torch.ones((1, 1, 3), dtype=torch.bool, device=args.device),
            state_mask=torch.ones((1, 1), dtype=torch.bool, device=args.device),
            action_mask=torch.ones((1, 16), dtype=torch.bool, device=args.device),
        )
        base_output = portable.model("bimanual_qpos14", safety_batch)
        record_path = records_dir / f"{hashlib.sha256(window_id.encode()).hexdigest()}.pt"
        _atomic_torch_save(
            record_path,
            {
                "schema_version": 1,
                "window": window,
                "video": views,
                "state": state_portable.cpu(),
                "action": action_portable.cpu(),
                "action_dt": dt,
                "future_tokens": future[0],
                "future_metadata": future_metadata,
                "base_class_logits": base_output["class_logits"][0].float().cpu(),
                "base_step_class_logits": base_output["step_class_logits"][0].float().cpu(),
            },
        )
        row = {
            "window_id": window_id,
            "record_path": str(record_path.relative_to(output)),
            "record_sha256": _sha256(record_path),
            "split": window["split"],
            "task": window["task"],
            "chunk_target": window["chunk_target"],
            "future_token_shape": future_metadata["token_shape"][1:],
            "elapsed_seconds": time.monotonic() - started,
        }
        _append_jsonl(progress_path, row)
        print(json.dumps({"completed": index + 1, **row}), flush=True)

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slice_manifest_sha256": _sha256(slice_root / "MANIFEST.json"),
        "efficient_source_commit": "2bd75a8c56acfcd5754b98c7ed313176911ccae0",
        "deploy_config": str(args.deploy_config.resolve()),
        "deploy_config_sha256": _sha256(args.deploy_config),
        "efficient_checkpoint": str(config["checkpoint_path"]),
        "efficient_checkpoint_sha256": _sha256(Path(config["checkpoint_path"])),
        "portable_checkpoint": str(args.portable_checkpoint.resolve()),
        "portable_checkpoint_sha256": _sha256(args.portable_checkpoint),
        "normalization_max_abs_delta": normalization_delta,
        "num_video_steps": args.num_video_steps,
        "num_video_frames": args.num_video_frames,
        "feature_layer": -1,
        "feature_kind": "future_video_value_tokens_spatially_preserved",
        "candidate_conditioning": "clean_candidate_action_at_action_t_zero",
        "records": len(_read_jsonl(progress_path)),
    }
    temporary = output / f"SUMMARY.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "SUMMARY.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
