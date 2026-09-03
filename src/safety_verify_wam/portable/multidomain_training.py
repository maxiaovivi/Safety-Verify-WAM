from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import apply_overrides, load_config, public_config, resolve_project_path
from .checkpoint import load_portable_checkpoint
from .contracts import PORTABLE_CLASS_NAMES, RobotProfile, SafetyBatch
from .data import (
    IGNORE_STEP_LABEL,
    PortableSafetyManifestDataset,
    file_sha256,
    portable_safety_collate,
)
from .losses import portable_safety_loss
from .multidomain import (
    MultiProfilePortableSafetyCore,
    MultiProfileSafetyConfig,
    ProfileAdapterConfig,
    config_fingerprint,
    initialize_from_single_profile,
    load_multidomain_checkpoint,
    save_multidomain_checkpoint,
    sha256_file,
    trainable_parameter_count,
)
from .runtime import SafetyThresholds
from .training import binary_metrics, select_deployment_threshold


LOGGER = logging.getLogger("safety_verify_wam.portable.multidomain_training")
TRAINING_STATE_SCHEMA = "portable_safety_multidomain_training_state/v1"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        public_config(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def command_output(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_metadata(root: Path) -> dict[str, Any]:
    status = command_output(["git", "status", "--porcelain=v1"], root)
    diff = command_output(["git", "diff", "--binary", "HEAD"], root)
    return {
        "root": str(root),
        "branch": command_output(["git", "branch", "--show-current"], root),
        "commit": command_output(["git", "rev-parse", "HEAD"], root),
        "dirty": bool(status),
        "status_sha256": None
        if status is None
        else hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": None
        if diff is None
        else hashlib.sha256(diff.encode()).hexdigest(),
    }


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "contracts.py",
        "data.py",
        "losses.py",
        "model.py",
        "multidomain.py",
        "multidomain_training.py",
        "training.py",
    )
    return {name: file_sha256(root / name) for name in names}


def dataset_from_config(
    dataset_config: dict[str, Any], split: str
) -> PortableSafetyManifestDataset:
    root = Path(dataset_config["manifest_root"]).expanduser().resolve()
    limit = dataset_config.get(f"max_{split}_samples")
    return PortableSafetyManifestDataset(
        root / "manifests" / f"{split}.jsonl",
        image_size=tuple(dataset_config.get("image_size", (128, 128))),
        camera_fields=tuple(
            dataset_config.get(
                "camera_fields",
                ("head_rgb_path", "left_rgb_path", "right_rgb_path"),
            )
        ),
        sample_limit=None if limit is None else int(limit),
        seed=int(dataset_config.get("sample_seed", 17)),
    )


def make_loader(
    dataset: PortableSafetyManifestDataset,
    dataset_config: dict[str, Any],
    *,
    training: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    workers = int(dataset_config.get("num_workers", 4))
    sampler = None
    if training:
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            dataset.stratum_weights(),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(dataset_config.get("batch_size", 32)),
        "shuffle": False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": portable_safety_collate,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(dataset_config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def to_safety_batch(batch: dict[str, Any], device: torch.device) -> SafetyBatch:
    return SafetyBatch(
        video=batch["video"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ).div_(255.0),
        state=batch["state"].to(device=device, dtype=torch.float32, non_blocking=True),
        action=batch["action"].to(device=device, dtype=torch.float32, non_blocking=True),
        video_timestamps=batch["video_timestamps"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        state_timestamps=batch["state_timestamps"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        action_timestamps=batch["action_timestamps"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        video_mask=batch["video_mask"].to(device=device, non_blocking=True),
        state_mask=batch["state_mask"].to(device=device, non_blocking=True),
        action_mask=batch["action_mask"].to(device=device, non_blocking=True),
    )


def real_profile(path: str | Path) -> tuple[RobotProfile, dict[str, Any]]:
    profile_path = Path(path).expanduser().resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    state = payload["state_normalization"]
    action = payload["action_normalization"]
    profile = RobotProfile(
        name=str(payload["name"]),
        state_dim=int(payload["state_dim"]),
        action_dim=int(payload["action_dim"]),
        control_dt=1.0 / 15.0,
        state_mean=tuple(float(value) for value in state["mean"]),
        state_std=tuple(float(value) for value in state["std"]),
        action_mean=tuple(float(value) for value in action["mean"]),
        action_std=tuple(float(value) for value in action["std"]),
        joint_names=(
            "speedl_vx",
            "speedl_vy",
            "speedl_vz",
            "speedl_wx",
            "speedl_wy",
            "speedl_wz",
            "gripper_command",
        ),
        camera_names=("external", "hand", "hand2"),
        image_size=tuple(int(value) for value in payload["image_size"]),
        state_semantics=str(payload["state_semantics"]),
        action_semantics=str(payload["action_semantics"]),
    )
    return profile, {
        "path": str(profile_path),
        "sha256": file_sha256(profile_path),
        "payload": payload,
    }


def build_model_config(
    config: dict[str, Any], sim_profile: RobotProfile, ur5_profile: RobotProfile
) -> MultiProfileSafetyConfig:
    model = dict(config["model"])
    if "vision_channels" in model:
        model["vision_channels"] = tuple(int(value) for value in model["vision_channels"])
    return MultiProfileSafetyConfig(
        profiles=(
            ProfileAdapterConfig(
                key="bimanual_qpos14",
                state_dim=sim_profile.state_dim,
                action_dim=sim_profile.action_dim,
                motion_mode="position_target",
            ),
            ProfileAdapterConfig(
                key="ur5_speedl7",
                state_dim=ur5_profile.state_dim,
                action_dim=ur5_profile.action_dim,
                motion_mode="velocity_command",
            ),
        ),
        **model,
    )


def amp_settings(
    name: str, device: torch.device
) -> tuple[bool, torch.dtype, torch.amp.GradScaler]:
    normalized = name.lower()
    if device.type != "cuda" or normalized in {"none", "float32"}:
        return False, torch.float32, torch.amp.GradScaler("cuda", enabled=False)
    if normalized == "bfloat16":
        return True, torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False)
    if normalized == "float16":
        return True, torch.float16, torch.amp.GradScaler("cuda", enabled=True)
    raise ValueError(f"Unsupported AMP mode: {name}")


@torch.inference_mode()
def evaluate_domains(
    model: MultiProfilePortableSafetyCore,
    evaluations: list[dict[str, Any]],
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    threshold_config: dict[str, Any],
    fixed_threshold: float | None = None,
    fixed_profile_thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    targets: list[int] = []
    scores: list[float] = []
    chunk_scores: list[float] = []
    domains: list[str] = []
    profile_keys: list[str] = []
    records: list[dict[str, Any]] = []
    step_targets: list[int] = []
    step_scores: list[float] = []
    for item in evaluations:
        profile_key = item["profile_key"]
        profile = item["profile"]
        policy = item["deployment_score_policy"]
        metadata_by_id = {
            str(row["sample_id"]): row for row in item["dataset"].rows
        }
        for batch in item["loader"]:
            value = to_safety_batch(batch, device)
            value.validate(profile)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                outputs = model(profile_key, value)
            chunk = outputs["risk_probability"].float()
            maximum_step = outputs["step_risk_probability"].float().masked_fill(
                ~value.action_mask, -1.0
            ).max(dim=1).values
            if policy == "chunk_or_step":
                deployment = torch.maximum(chunk, maximum_step)
            elif policy == "chunk_only":
                deployment = chunk
            else:
                raise ValueError(f"Unknown deployment score policy: {policy}")
            batch_targets = batch["chunk_target"].tolist()
            targets.extend(batch_targets)
            scores.extend(deployment.cpu().tolist())
            chunk_scores.extend(chunk.cpu().tolist())
            domains.extend(batch["simulator_key"])
            profile_keys.extend([profile_key] * len(batch_targets))
            for index, sample_id in enumerate(batch["sample_id"]):
                source = metadata_by_id[str(sample_id)]
                records.append(
                    {
                        "sample_id": str(sample_id),
                        "observation_group_id": str(batch["observation_group_id"][index]),
                        "profile_key": profile_key,
                        "simulator_key": str(batch["simulator_key"][index]),
                        "target": int(batch_targets[index]),
                        "target_name": PORTABLE_CLASS_NAMES[int(batch_targets[index])],
                        "chunk_risk_probability": float(chunk[index].item()),
                        "deployment_risk_probability": float(deployment[index].item()),
                        "deployment_score_policy": policy,
                        "episode_id": source.get("episode_id"),
                        "interaction_phase": source.get("interaction_phase"),
                        "condition_sync_valid": source.get("condition_sync_valid"),
                    }
                )
            target_steps = batch["step_target"].to(device=device)
            valid_steps = (target_steps != IGNORE_STEP_LABEL) & value.action_mask
            if bool(valid_steps.any()):
                step_targets.extend(target_steps[valid_steps].cpu().tolist())
                step_scores.extend(
                    outputs["step_risk_probability"][valid_steps]
                    .float()
                    .cpu()
                    .tolist()
                )
    target_array = np.asarray(targets, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    chunk_array = np.asarray(chunk_scores, dtype=np.float64)
    domain_array = np.asarray(domains, dtype=object)
    profile_array = np.asarray(profile_keys, dtype=object)
    unique_profiles = sorted(set(profile_keys))
    profile_specific = bool(threshold_config.get("profile_specific", False))
    if fixed_profile_thresholds is not None:
        profile_thresholds = {
            str(key): float(value) for key, value in fixed_profile_thresholds.items()
        }
        if set(profile_thresholds) != set(unique_profiles):
            raise ValueError("Fixed profile threshold keys differ from evaluation")
        selections = {
            key: {
                "threshold": value,
                "objective_name": "fixed",
                "constraints_satisfied": None,
            }
            for key, value in profile_thresholds.items()
        }
    elif fixed_threshold is None and profile_specific:
        profile_thresholds = {}
        selections = {}
        for profile_key in unique_profiles:
            mask = profile_array == profile_key
            profile_search = {
                **threshold_config,
                **dict(
                    threshold_config.get("profile_overrides", {}).get(
                        profile_key, {}
                    )
                ),
            }
            selected = select_deployment_threshold(
                target_array[mask],
                score_array[mask],
                domain_array[mask].tolist(),
                profile_search,
            )
            profile_thresholds[profile_key] = float(selected["threshold"])
            selections[profile_key] = selected
    elif fixed_threshold is None:
        selected = select_deployment_threshold(
            target_array, score_array, domains, threshold_config
        )
        profile_thresholds = {
            profile_key: float(selected["threshold"])
            for profile_key in unique_profiles
        }
        selections = {profile_key: selected for profile_key in unique_profiles}
    else:
        profile_thresholds = {
            profile_key: float(fixed_threshold) for profile_key in unique_profiles
        }
        selections = {
            profile_key: {
                "threshold": float(fixed_threshold),
                "objective_name": "fixed",
                "constraints_satisfied": None,
            }
            for profile_key in unique_profiles
        }
    threshold_array = np.asarray(
        [profile_thresholds[key] for key in profile_keys], dtype=np.float64
    )
    calibrated_scores = score_array - threshold_array + 0.5
    by_domain = {
        domain: binary_metrics(
            target_array[domain_array == domain], score_array[domain_array == domain],
            profile_thresholds[str(profile_array[domain_array == domain][0])],
        )
        for domain in sorted(set(domains))
    }
    sim_mask = domain_array != "ur5_real"
    real_mask = domain_array == "ur5_real"
    metrics: dict[str, Any] = {
        "samples": len(targets),
        "threshold": (
            next(iter(profile_thresholds.values()))
            if len(set(profile_thresholds.values())) == 1
            else None
        ),
        "profile_thresholds": profile_thresholds,
        "deployment": binary_metrics(target_array, calibrated_scores, 0.5),
        "chunk_at_0_5": binary_metrics(target_array, chunk_array, 0.5),
        "by_domain": by_domain,
        "sim_pooled": binary_metrics(
            target_array[sim_mask],
            score_array[sim_mask],
            profile_thresholds["bimanual_qpos14"],
        ),
        "ur5_development_holdout": binary_metrics(
            target_array[real_mask],
            score_array[real_mask],
            profile_thresholds["ur5_speedl7"],
        ),
        "threshold_selection": {
            "profile_specific": profile_specific,
            "by_profile": {
                profile_key: {
                    key: value
                    for key, value in selection.items()
                    if key not in {"metrics", "by_simulator"}
                }
                for profile_key, selection in selections.items()
            },
        },
    }
    if step_targets:
        metrics["sim_step"] = binary_metrics(
            step_targets,
            step_scores,
            profile_thresholds["bimanual_qpos14"],
        )
    ur5_records = [row for row in records if row["simulator_key"] == "ur5_real"]
    sync_metrics: dict[str, Any] = {}
    for name, flag in (("sync_valid", True), ("sync_warning", False)):
        subset = [row for row in ur5_records if row["condition_sync_valid"] is flag]
        if subset and {row["target"] for row in subset} == {0, 1}:
            sync_metrics[name] = binary_metrics(
                [row["target"] for row in subset],
                [row["deployment_risk_probability"] for row in subset],
                profile_thresholds["ur5_speedl7"],
            )
        else:
            sync_metrics[name] = {"samples": len(subset), "status": "insufficient_classes"}
    metrics["ur5_by_condition_sync"] = sync_metrics
    for row in records:
        threshold = profile_thresholds[str(row["profile_key"])]
        predicted = int(row["deployment_risk_probability"] >= threshold)
        row["threshold"] = threshold
        row["predicted"] = predicted
        row["predicted_name"] = PORTABLE_CLASS_NAMES[predicted]
        row["correct"] = predicted == row["target"]
    return metrics, records


def acceptance_checks(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    required = config["acceptance"]
    domains = metrics["by_domain"]
    return {
        "sim_pooled_balanced_accuracy": metrics["sim_pooled"]["balanced_accuracy"]
        >= float(required["minimum_sim_pooled_balanced_accuracy"]),
        "maniskill_balanced_accuracy": domains["maniskill"]["balanced_accuracy"]
        >= float(required["minimum_each_simulator_balanced_accuracy"]),
        "robotwin_balanced_accuracy": domains["robotwin"]["balanced_accuracy"]
        >= float(required["minimum_each_simulator_balanced_accuracy"]),
        "ur5_balanced_accuracy": domains["ur5_real"]["balanced_accuracy"]
        >= float(required["minimum_ur5_balanced_accuracy"]),
        "ur5_risk_recall": domains["ur5_real"]["risk_recall"]
        >= float(required["minimum_ur5_risk_recall"]),
        "sim_false_positive_rate": metrics["sim_pooled"]["false_positive_rate"]
        <= float(required["maximum_sim_false_positive_rate"]),
        "sim_false_negative_rate": metrics["sim_pooled"]["false_negative_rate"]
        <= float(required["maximum_sim_false_negative_rate"]),
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    seed = int(config.get("seed", 17))
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    requested = str(config.get("device", "cuda"))
    if requested == "cuda":
        requested = "cuda:0"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        memory_fraction = float(config.get("cuda_memory_fraction", 1.0))
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_hash = config_sha256(config)
    atomic_json(output_dir / "config.resolved.json", public_config(config))

    pretrained_path = resolve_project_path(config["pretrained_checkpoint"])
    pretrained = load_portable_checkpoint(pretrained_path, map_location=device)
    ur5_profile, ur5_profile_metadata = real_profile(config["ur5_profile"])
    profiles = {
        "bimanual_qpos14": pretrained.profile,
        "ur5_speedl7": ur5_profile,
    }
    model_config = build_model_config(config, pretrained.profile, ur5_profile)
    model = MultiProfilePortableSafetyCore(model_config).to(device)
    initialize_from_single_profile(
        model, pretrained.model, profile_key="bimanual_qpos14"
    )

    datasets: dict[str, dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []
    for index, key in enumerate(("simulation", "ur5")):
        dataset_config = config["datasets"][key]
        train_dataset = dataset_from_config(dataset_config, "train")
        val_dataset = dataset_from_config(dataset_config, "val")
        profile_key = str(dataset_config["profile_key"])
        profile = profiles[profile_key]
        train_loader = make_loader(
            train_dataset,
            dataset_config,
            training=True,
            device=device,
            seed=seed + index,
        )
        val_loader = make_loader(
            val_dataset,
            dataset_config,
            training=False,
            device=device,
            seed=seed + 100 + index,
        )
        datasets[key] = {
            "profile_key": profile_key,
            "profile": profile,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "deployment_score_policy": str(
                dataset_config["deployment_score_policy"]
            ),
        }
        evaluations.append(
            {
                "profile_key": profile_key,
                "profile": profile,
                "dataset": val_dataset,
                "loader": val_loader,
                "deployment_score_policy": str(
                    dataset_config["deployment_score_policy"]
                ),
            }
        )
    dataset_summary = {
        key: {
            "profile_key": value["profile_key"],
            "train": value["train_dataset"].summary(),
            "val": value["val_dataset"].summary(),
            "deployment_score_policy": value["deployment_score_policy"],
        }
        for key, value in datasets.items()
    }
    dataset_summary["ur5_profile"] = ur5_profile_metadata
    atomic_json(output_dir / "dataset_summary.json", dataset_summary)

    sim_probe = next(iter(datasets["simulation"]["val_loader"]))
    sim_probe_action = sim_probe["action"].clone()
    sim_probe_batch = to_safety_batch(sim_probe, device)
    sim_probe_batch.validate(pretrained.profile)
    model.eval()
    pretrained.model.eval()
    with torch.inference_mode():
        source_prob = pretrained.model(sim_probe_batch)["risk_probability"].float()
        migrated_prob = model(
            "bimanual_qpos14", sim_probe_batch
        )["risk_probability"].float()
    migration_max_abs_error = float((source_prob - migrated_prob).abs().max().item())

    training = config["training"]
    shared_lr = float(training.get("shared_learning_rate", 1e-4))
    ur5_lr = float(training.get("ur5_adapter_learning_rate", 3e-4))
    ur5_modules: list[torch.nn.Module] = [
        model.profile_adapters["ur5_speedl7"]
    ]
    if model.config.profile_specific_heads:
        ur5_modules.extend(
            [
                model.profile_chunk_heads["ur5_speedl7"],
                model.profile_step_heads["ur5_speedl7"],
            ]
        )
    freeze_pretrained_path = bool(training.get("freeze_pretrained_path", False))
    if freeze_pretrained_path:
        if not model.config.profile_specific_heads:
            raise ValueError(
                "freeze_pretrained_path requires profile_specific_heads"
            )
        model.requires_grad_(False)
        for module in ur5_modules:
            module.requires_grad_(True)
    ur5_parameters = [
        parameter
        for module in ur5_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    ur5_ids = {id(parameter) for parameter in ur5_parameters}
    shared_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in ur5_ids
    ]
    parameter_groups: list[dict[str, Any]] = []
    if shared_parameters:
        parameter_groups.append({"params": shared_parameters, "lr": shared_lr})
    if ur5_parameters:
        parameter_groups.append({"params": ur5_parameters, "lr": ur5_lr})
    if not parameter_groups:
        raise RuntimeError("Training configuration left no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(training.get("weight_decay", 1e-3)),
    )
    amp_enabled, amp_dtype, scaler = amp_settings(
        str(training.get("amp", "bfloat16")), device
    )
    sim_step_class_weights = datasets["simulation"]["train_dataset"].class_weights(
        step=True,
        maximum_ratio=float(config["loss"].get("maximum_step_class_weight_ratio", 6.0)),
    ).to(device)
    chunk_class_weights = torch.ones(2, dtype=torch.float32, device=device)

    initial_metrics, initial_predictions = evaluate_domains(
        model,
        evaluations,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        threshold_config=config["validation"]["threshold_search"],
        fixed_threshold=float(pretrained.thresholds.chunk_risk),
    )
    atomic_json(output_dir / "initial_val_metrics.json", initial_metrics)
    atomic_jsonl(output_dir / "initial_val_predictions.jsonl", initial_predictions)

    epochs = int(training.get("epochs", 30))
    steps_per_epoch = int(training.get("steps_per_epoch", 36))
    sim_steps = int(training.get("simulation_steps_per_epoch", 24))
    ur5_steps = steps_per_epoch - sim_steps
    if (
        epochs < 1
        or steps_per_epoch < 1
        or sim_steps < 0
        or ur5_steps < 1
        or (not freeze_pretrained_path and sim_steps < 1)
    ):
        raise ValueError("Invalid epoch/domain schedule")
    max_steps_raw = training.get("max_steps")
    max_steps = None if max_steps_raw is None else int(max_steps_raw)
    patience = int(training.get("early_stopping_patience", 7))
    min_delta = float(training.get("early_stopping_min_delta", 1e-4))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    step_weight = float(config["loss"].get("step_weight", 0.35))
    global_step = 0
    start_epoch = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    best_threshold: float | None = None
    best_profile_thresholds: dict[str, float] | None = None
    stale_epochs = 0
    state_path = output_dir / "training_state.pt"
    if bool(training.get("resume", True)) and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != TRAINING_STATE_SCHEMA:
            raise RuntimeError("Unsupported multidomain training state")
        if state["config_sha256"] != resolved_hash:
            raise RuntimeError("Training state config hash differs")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["next_epoch"])
        global_step = int(state["global_step"])
        best_metric = state["best_metric"]
        best_epoch = state["best_epoch"]
        best_threshold = state["best_threshold"]
        best_profile_thresholds = state.get("best_profile_thresholds")
        stale_epochs = int(state["stale_epochs"])

    project_root = Path(__file__).resolve().parents[3]
    runtime = {
        "started_at": now(),
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_memory_fraction": config.get("cuda_memory_fraction"),
        "trainable_parameters": trainable_parameter_count(model),
        "model_config_fingerprint": config_fingerprint(model_config),
        "profile_fingerprints": {
            key: value.fingerprint for key, value in profiles.items()
        },
        "pretrained_checkpoint": str(pretrained_path),
        "pretrained_checkpoint_sha256": pretrained.checkpoint_sha256,
        "migration_max_abs_error": migration_max_abs_error,
        "freeze_pretrained_path": freeze_pretrained_path,
        "git": git_metadata(project_root),
        "source_hashes": source_hashes(),
    }
    atomic_json(output_dir / "runtime.json", runtime)
    metrics_path = output_dir / "metrics.jsonl"
    if start_epoch == 0 and metrics_path.exists():
        metrics_path.unlink()

    stopped_reason = "epochs_completed"
    last_train_loss = float("nan")
    for epoch in range(start_epoch, epochs):
        model.train()
        schedule = ["simulation"] * sim_steps + ["ur5"] * ur5_steps
        random.Random(seed + epoch).shuffle(schedule)
        iterators = {
            key: iter(datasets[key]["train_loader"]) for key in datasets
        }
        running_loss = 0.0
        domain_losses: Counter[str] = Counter()
        domain_steps: Counter[str] = Counter()
        for dataset_key in schedule:
            if max_steps is not None and global_step >= max_steps:
                break
            try:
                batch = next(iterators[dataset_key])
            except StopIteration:
                iterators[dataset_key] = iter(datasets[dataset_key]["train_loader"])
                batch = next(iterators[dataset_key])
            profile_key = datasets[dataset_key]["profile_key"]
            value = to_safety_batch(batch, device)
            value.validate(datasets[dataset_key]["profile"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                outputs = model(profile_key, value)
                loss, _ = portable_safety_loss(
                    outputs,
                    batch["chunk_target"],
                    step_target=batch["step_target"],
                    chunk_weight=float(config["loss"].get("chunk_weight", 1.0)),
                    step_weight=step_weight,
                    class_weights=chunk_class_weights,
                    step_class_weights=sim_step_class_weights,
                    ignore_index=IGNORE_STEP_LABEL,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss at step {global_step}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            value_loss = float(loss.item())
            running_loss += value_loss
            domain_losses[dataset_key] += value_loss
            domain_steps[dataset_key] += 1
            global_step += 1
        last_train_loss = running_loss / max(1, sum(domain_steps.values()))
        validation_metrics, predictions = evaluate_domains(
            model,
            evaluations,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            threshold_config=config["validation"]["threshold_search"],
        )
        objective = min(
            value["balanced_accuracy"]
            for value in validation_metrics["by_domain"].values()
        )
        improved = best_metric is None or objective > best_metric + min_delta
        if improved:
            best_metric = objective
            best_epoch = epoch
            best_profile_thresholds = {
                key: float(value)
                for key, value in validation_metrics["profile_thresholds"].items()
            }
            best_threshold = best_profile_thresholds["bimanual_qpos14"]
            stale_epochs = 0
            save_multidomain_checkpoint(
                output_dir / "best.pt",
                model,
                profiles,
                thresholds=SafetyThresholds(
                    chunk_risk=best_threshold, step_risk=best_threshold
                ),
                profile_thresholds={
                    key: SafetyThresholds(chunk_risk=value, step_risk=value)
                    for key, value in best_profile_thresholds.items()
                },
                metadata={
                    **runtime,
                    "training_source": (
                        "frozen_maniskill_robotwin_v2+ur5_command_bouts_v1"
                        if freeze_pretrained_path
                        else "maniskill+robotwin+ur5_command_bouts_v1"
                    ),
                    "epoch": epoch,
                    "global_step": global_step,
                    "validation": validation_metrics,
                    "dataset_summary": dataset_summary,
                    "config_sha256": resolved_hash,
                    "host_policy_parameters_updated": False,
                    "policy_internal_features_required": False,
                },
            )
            atomic_jsonl(output_dir / "best_val_predictions.jsonl", predictions)
        else:
            stale_epochs += 1
        save_multidomain_checkpoint(
            output_dir / "latest.pt",
            model,
            profiles,
            thresholds=SafetyThresholds(
                chunk_risk=float(
                    validation_metrics["profile_thresholds"]["bimanual_qpos14"]
                ),
                step_risk=float(
                    validation_metrics["profile_thresholds"]["bimanual_qpos14"]
                ),
            ),
            profile_thresholds={
                key: SafetyThresholds(
                    chunk_risk=float(value), step_risk=float(value)
                )
                for key, value in validation_metrics["profile_thresholds"].items()
            },
            metadata={
                **runtime,
                "epoch": epoch,
                "global_step": global_step,
                "validation": validation_metrics,
                "config_sha256": resolved_hash,
            },
        )
        atomic_torch_save(
            state_path,
            {
                "schema_version": TRAINING_STATE_SCHEMA,
                "config_sha256": resolved_hash,
                "next_epoch": epoch + 1,
                "global_step": global_step,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "best_threshold": best_threshold,
                "best_profile_thresholds": best_profile_thresholds,
                "stale_epochs": stale_epochs,
                "model_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
            },
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": last_train_loss,
            "domain_train_loss": {
                key: domain_losses[key] / domain_steps[key]
                for key in sorted(domain_steps)
            },
            "validation": validation_metrics,
            "objective": objective,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "timestamp": now(),
        }
        append_jsonl(metrics_path, record)
        LOGGER.info(json.dumps(record, ensure_ascii=False, sort_keys=True))
        if max_steps is not None and global_step >= max_steps:
            stopped_reason = "max_steps"
            break
        if patience >= 0 and stale_epochs > patience:
            stopped_reason = "early_stopping"
            break

    best_path = output_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("Training completed without a best checkpoint")
    loaded = load_multidomain_checkpoint(
        best_path, map_location=device, expected_profiles=profiles
    )
    loaded.model.to(device).eval()
    final_metrics, final_predictions = evaluate_domains(
        loaded.model,
        evaluations,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        threshold_config=config["validation"]["threshold_search"],
        fixed_profile_thresholds={
            key: value.chunk_risk
            for key, value in loaded.profile_thresholds.items()
        },
    )
    atomic_json(output_dir / "val_metrics.json", final_metrics)
    atomic_jsonl(output_dir / "val_predictions.jsonl", final_predictions)

    reloaded = load_multidomain_checkpoint(
        best_path, map_location=device, expected_profiles=profiles
    )
    reloaded.model.to(device).eval()
    reload_difference = 0.0
    action_unchanged = True
    for item in evaluations:
        probe = next(iter(item["loader"]))
        original_action = probe["action"].clone()
        value = to_safety_batch(probe, device)
        with torch.inference_mode():
            first = loaded.model(item["profile_key"], value)["risk_probability"]
            second = reloaded.model(item["profile_key"], value)["risk_probability"]
        reload_difference = max(
            reload_difference, float((first - second).abs().max().item())
        )
        action_unchanged = action_unchanged and bool(
            torch.equal(original_action, probe["action"])
        )
    checks = acceptance_checks(final_metrics, config)
    accepted = all(checks.values())
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        **runtime,
        "finished_at": now(),
        "elapsed_seconds": time.monotonic() - started,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_threshold": best_threshold,
        "best_profile_thresholds": best_profile_thresholds,
        "stopped_reason": stopped_reason,
        "last_train_loss": last_train_loss,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "validation": final_metrics,
        "acceptance_checks": checks,
        "accepted": accepted,
        "migration_max_abs_error": migration_max_abs_error,
        "checkpoint_reload_max_abs_error": reload_difference,
        "action_chunk_unchanged": action_unchanged
        and bool(torch.equal(sim_probe_action, sim_probe["action"])),
        "peak_allocated_mib": peak_memory,
        "config_sha256": resolved_hash,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "checkpoint_manifest.json",
        {
            "schema_version": 1,
            "recommended": accepted,
            "checkpoint": str(best_path),
            "checkpoint_sha256": sha256_file(best_path),
            "pretrained_checkpoint": str(pretrained_path),
            "pretrained_checkpoint_sha256": pretrained.checkpoint_sha256,
            "config_sha256": resolved_hash,
            "model_config_fingerprint": config_fingerprint(model_config),
            "profile_fingerprints": runtime["profile_fingerprints"],
            "profile_thresholds": best_profile_thresholds,
            "source_hashes": runtime["source_hashes"],
        },
    )
    if bool(config.get("sanity", {}).get("enabled", False)):
        sanity_checks = {
            "finite_loss": bool(np.isfinite(last_train_loss)),
            "single_profile_migration_exact": migration_max_abs_error <= 1e-6,
            "checkpoint_reload_exact": reload_difference == 0.0,
            "action_chunk_unchanged": summary["action_chunk_unchanged"],
            "all_three_domains_evaluated": set(final_metrics["by_domain"])
            == {"maniskill", "robotwin", "ur5_real"},
            "parameter_count_bounded": trainable_parameter_count(model) < 1_100_000,
        }
        sanity = {
            "status": "PASS" if all(sanity_checks.values()) else "FAIL",
            "checks": sanity_checks,
            "global_step": global_step,
            "trainable_parameters": trainable_parameter_count(model),
            "checkpoint": str(best_path),
            "checkpoint_sha256": sha256_file(best_path),
            "validation": final_metrics,
        }
        atomic_json(output_dir / "SANITY_RESULT.json", sanity)
        if sanity["status"] != "PASS":
            raise RuntimeError(f"Multidomain sanity failed: {sanity_checks}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jointly train portable safety adapters across robot profiles"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    train(apply_overrides(load_config(args.config), args.overrides))


if __name__ == "__main__":
    main()
