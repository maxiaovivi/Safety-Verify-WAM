from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import apply_overrides, load_config, public_config, resolve_project_path
from .checkpoint import (
    load_portable_checkpoint,
    save_portable_checkpoint,
    sha256_file,
)
from .contracts import PORTABLE_CLASS_NAMES, RobotProfile, SafetyBatch
from .data import (
    IGNORE_STEP_LABEL,
    PortableSafetyManifestDataset,
    file_sha256,
    portable_safety_collate,
)
from .losses import portable_safety_loss
from .model import PortableSafetyConfig, PortableSafetyCore, trainable_parameter_count
from .runtime import SafetyThresholds


LOGGER = logging.getLogger("safety_verify_wam.portable.training")
TRAINING_STATE_SCHEMA = "portable_safety_training_state/v1"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        public_config(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _command_output(command: list[str], *, cwd: Path) -> str | None:
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


def _git_metadata(root: Path) -> dict[str, Any]:
    commit = _command_output(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _command_output(["git", "branch", "--show-current"], cwd=root)
    status = _command_output(["git", "status", "--porcelain=v1"], cwd=root)
    diff = _command_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    return {
        "root": str(root),
        "branch": branch,
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": (
            None if status is None else hashlib.sha256(status.encode()).hexdigest()
        ),
        "tracked_diff_sha256": (
            None if diff is None else hashlib.sha256(diff.encode()).hexdigest()
        ),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "__init__.py",
        "checkpoint.py",
        "contracts.py",
        "data.py",
        "losses.py",
        "model.py",
        "runtime.py",
        "training.py",
    )
    return {name: file_sha256(root / name) for name in names}


def _load_stats(path: str | Path, dimension: int) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, Any]]:
    stats_path = Path(path).expanduser().resolve()
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    mean = tuple(float(value) for value in payload["mean"])
    std = tuple(float(value) for value in payload["std"])
    if len(mean) != dimension or len(std) != dimension:
        raise ValueError(
            f"Normalization statistics have {len(mean)}/{len(std)} values, "
            f"expected {dimension}"
        )
    return mean, std, {
        "path": str(stats_path),
        "sha256": file_sha256(stats_path),
        "declared_asset_sha256": payload.get("asset_file_sha256"),
        "type": payload.get("type"),
    }


def _joint_names(dimension: int) -> tuple[str, ...]:
    if dimension == 14:
        return tuple(
            [f"left_joint_{index}" for index in range(1, 7)]
            + ["left_gripper_openness"]
            + [f"right_joint_{index}" for index in range(1, 7)]
            + ["right_gripper_openness"]
        )
    return tuple(f"action_{index}" for index in range(dimension))


def _build_profile(config: dict[str, Any]) -> tuple[RobotProfile, dict[str, Any]]:
    profile_config = config["profile"]
    dimension = int(profile_config.get("dimension", 14))
    mean, std, stats_metadata = _load_stats(
        config["dataset"]["normalization_stats"], dimension
    )
    image_size = tuple(int(value) for value in config["dataset"]["image_size"])
    camera_names = tuple(
        str(value)
        for value in profile_config.get(
            "camera_names", ("head_rgb", "left_rgb", "right_rgb")
        )
    )
    profile = RobotProfile(
        name=str(profile_config.get("name", "crosssim-bimanual-qpos14-v1")),
        state_dim=dimension,
        action_dim=dimension,
        control_dt=float(profile_config.get("default_control_dt", 0.1)),
        state_mean=mean,
        state_std=std,
        action_mean=mean,
        action_std=std,
        joint_names=_joint_names(dimension),
        camera_names=camera_names,
        image_size=image_size,
        state_semantics=str(
            profile_config.get("state_semantics", "dual_arm_qpos_at_observation_time")
        ),
        action_semantics=str(
            profile_config.get("action_semantics", "dual_arm_qpos_target_chunk")
        ),
    )
    return profile, stats_metadata


def _build_model(config: dict[str, Any], profile: RobotProfile) -> PortableSafetyCore:
    values = dict(config.get("model", {}))
    for key in ("vision_channels",):
        if key in values:
            values[key] = tuple(int(value) for value in values[key])
    return PortableSafetyCore.for_profile(profile, **values)


def _amp_settings(
    config: dict[str, Any], device: torch.device
) -> tuple[bool, torch.dtype, torch.amp.GradScaler]:
    name = str(config["training"].get("amp", "bfloat16")).lower()
    if device.type != "cuda" or name in {"none", "float32"}:
        return False, torch.float32, torch.amp.GradScaler("cuda", enabled=False)
    if name == "bfloat16":
        return True, torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False)
    if name == "float16":
        return True, torch.float16, torch.amp.GradScaler("cuda", enabled=True)
    raise ValueError(f"Unsupported AMP mode: {name}")


def _loader(
    dataset: PortableSafetyManifestDataset,
    config: dict[str, Any],
    device: torch.device,
    *,
    training: bool,
) -> DataLoader:
    workers = int(config["dataset"].get("num_workers", 4))
    sampler = None
    shuffle = False
    if training:
        if bool(config["dataset"].get("balance_strata", True)):
            generator = torch.Generator().manual_seed(int(config.get("seed", 7)))
            sampler = WeightedRandomSampler(
                dataset.stratum_weights(),
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
        else:
            shuffle = True
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["training"].get("batch_size", 32)),
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": portable_safety_collate,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(config["dataset"].get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def _to_safety_batch(batch: dict[str, Any], device: torch.device) -> SafetyBatch:
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


def binary_metrics(
    targets: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    threshold: float,
) -> dict[str, Any]:
    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    probability = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != probability.shape or truth.size == 0:
        raise ValueError("targets and scores must be non-empty vectors of equal size")
    if not np.isfinite(probability).all():
        raise ValueError("scores contain NaN or infinity")
    predicted = probability >= float(threshold)
    positive = truth == 1
    negative = ~positive
    tp = int(np.logical_and(predicted, positive).sum())
    tn = int(np.logical_and(~predicted, negative).sum())
    fp = int(np.logical_and(predicted, negative).sum())
    fn = int(np.logical_and(~predicted, positive).sum())
    safe_recall = tn / (tn + fp) if tn + fp else 0.0
    risk_recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / truth.size
    f1 = (
        2.0 * precision * risk_recall / (precision + risk_recall)
        if precision + risk_recall
        else 0.0
    )
    return {
        "samples": int(truth.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float((safe_recall + risk_recall) / 2.0),
        "risk_precision": float(precision),
        "risk_recall": float(risk_recall),
        "safe_recall": float(safe_recall),
        "f1": float(f1),
        "false_positive_rate": float(1.0 - safe_recall),
        "false_negative_rate": float(1.0 - risk_recall),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "class_order": list(PORTABLE_CLASS_NAMES),
    }


def _per_simulator_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    simulators: list[str],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    simulator_array = np.asarray(simulators, dtype=object)
    for simulator in sorted(set(simulators)):
        mask = simulator_array == simulator
        result[simulator] = binary_metrics(targets[mask], scores[mask], threshold)
    return result


def select_deployment_threshold(
    targets: np.ndarray,
    scores: np.ndarray,
    simulators: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    minimum = float(config.get("minimum", 0.05))
    maximum = float(config.get("maximum", 0.95))
    steps = int(config.get("steps", 181))
    if not 0.0 < minimum < maximum < 1.0 or steps < 2:
        raise ValueError("Invalid threshold search range")
    max_fpr = float(config.get("max_false_positive_rate", 0.15))
    max_fnr = float(config.get("max_false_negative_rate", 0.15))
    objective_name = str(
        config.get("objective", "worst_simulator_balanced_accuracy")
    )
    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(minimum, maximum, num=steps):
        total = binary_metrics(targets, scores, float(threshold))
        by_simulator = _per_simulator_metrics(
            targets, scores, simulators, float(threshold)
        )
        if objective_name == "worst_simulator_balanced_accuracy":
            objective = min(
                metrics["balanced_accuracy"] for metrics in by_simulator.values()
            )
        elif objective_name == "balanced_accuracy":
            objective = total["balanced_accuracy"]
        else:
            raise ValueError(f"Unknown threshold objective: {objective_name}")
        feasible = (
            total["false_positive_rate"] <= max_fpr
            and total["false_negative_rate"] <= max_fnr
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "objective": float(objective),
                "total": total,
                "by_simulator": by_simulator,
                "feasible": bool(feasible),
            }
        )
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    recall_fallback = (
        [
            candidate
            for candidate in candidates
            if candidate["total"]["false_negative_rate"] <= max_fnr
        ]
        if not feasible and bool(config.get("recall_first_fallback", False))
        else []
    )
    pool = feasible or recall_fallback or candidates

    def rank(candidate: dict[str, Any]) -> tuple[float, ...]:
        total = candidate["total"]
        largest_error = max(
            total["false_positive_rate"], total["false_negative_rate"]
        )
        return (
            candidate["objective"],
            total["balanced_accuracy"],
            -largest_error,
            total["risk_recall"],
            -abs(candidate["threshold"] - 0.5),
        )

    selected = max(pool, key=rank)
    return {
        "threshold": selected["threshold"],
        "objective_name": objective_name,
        "objective_value": selected["objective"],
        "constraints_satisfied": bool(feasible),
        "recall_fallback_satisfied": bool(recall_fallback),
        "constraints": {
            "max_false_positive_rate": max_fpr,
            "max_false_negative_rate": max_fnr,
        },
        "feasible_candidate_count": len(feasible),
        "candidate_count": len(candidates),
        "metrics": selected["total"],
        "by_simulator": selected["by_simulator"],
    }


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]


@torch.inference_mode()
def evaluate(
    model: PortableSafetyCore,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    chunk_class_weights: torch.Tensor,
    step_class_weights: torch.Tensor,
    loss_config: dict[str, Any],
    threshold_config: dict[str, Any],
    fixed_threshold: float | None = None,
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    chunk_targets: list[int] = []
    chunk_scores: list[float] = []
    deployment_scores: list[float] = []
    simulators: list[str] = []
    sample_ids: list[str] = []
    group_ids: list[str] = []
    step_targets: list[int] = []
    step_scores: list[float] = []
    for batch in loader:
        value = _to_safety_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            outputs = model(value)
            loss, _ = portable_safety_loss(
                outputs,
                batch["chunk_target"],
                step_target=batch["step_target"],
                chunk_weight=float(loss_config.get("chunk_weight", 1.0)),
                step_weight=float(loss_config.get("step_weight", 0.35)),
                class_weights=chunk_class_weights,
                step_class_weights=step_class_weights,
                ignore_index=IGNORE_STEP_LABEL,
            )
        batch_size = len(batch["sample_id"])
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        chunk_probability = outputs["risk_probability"].float()
        masked_steps = outputs["step_risk_probability"].float().masked_fill(
            ~value.action_mask, -1.0
        )
        maximum_step = masked_steps.max(dim=1).values
        deployment_probability = torch.maximum(chunk_probability, maximum_step)
        chunk_targets.extend(batch["chunk_target"].tolist())
        chunk_scores.extend(chunk_probability.cpu().tolist())
        deployment_scores.extend(deployment_probability.cpu().tolist())
        simulators.extend(batch["simulator_key"])
        sample_ids.extend(batch["sample_id"])
        group_ids.extend(batch["observation_group_id"])
        target_steps = batch["step_target"].to(device=device)
        valid_steps = (target_steps != IGNORE_STEP_LABEL) & value.action_mask
        step_targets.extend(target_steps[valid_steps].cpu().tolist())
        step_scores.extend(
            outputs["step_risk_probability"][valid_steps].float().cpu().tolist()
        )
    if sample_count == 0:
        raise RuntimeError("Evaluation loader produced no samples")
    targets = np.asarray(chunk_targets, dtype=np.int64)
    chunk = np.asarray(chunk_scores, dtype=np.float64)
    deployment = np.asarray(deployment_scores, dtype=np.float64)
    if fixed_threshold is None:
        threshold_selection = select_deployment_threshold(
            targets, deployment, simulators, threshold_config
        )
        threshold = float(threshold_selection["threshold"])
    else:
        threshold = float(fixed_threshold)
        total = binary_metrics(targets, deployment, threshold)
        by_simulator = _per_simulator_metrics(
            targets, deployment, simulators, threshold
        )
        threshold_selection = {
            "threshold": threshold,
            "objective_name": "fixed",
            "objective_value": min(
                value["balanced_accuracy"] for value in by_simulator.values()
            ),
            "constraints_satisfied": None,
            "metrics": total,
            "by_simulator": by_simulator,
        }
    step_metric = binary_metrics(step_targets, step_scores, threshold)
    predictions = [
        {
            "sample_id": sample_id,
            "observation_group_id": group_id,
            "simulator_key": simulator,
            "target": int(target),
            "target_name": PORTABLE_CLASS_NAMES[int(target)],
            "chunk_risk_probability": float(chunk_probability),
            "deployment_risk_probability": float(deployment_probability),
            "threshold": threshold,
            "predicted": int(deployment_probability >= threshold),
            "predicted_name": PORTABLE_CLASS_NAMES[
                int(deployment_probability >= threshold)
            ],
            "correct": bool((deployment_probability >= threshold) == target),
        }
        for sample_id, group_id, simulator, target, chunk_probability, deployment_probability in zip(
            sample_ids,
            group_ids,
            simulators,
            targets.tolist(),
            chunk.tolist(),
            deployment.tolist(),
        )
    ]
    metrics = {
        "loss": total_loss / sample_count,
        "samples": sample_count,
        "chunk_at_0_5": binary_metrics(targets, chunk, 0.5),
        "deployment": threshold_selection["metrics"],
        "by_simulator": threshold_selection["by_simulator"],
        "step": step_metric,
        "threshold_selection": {
            key: value
            for key, value in threshold_selection.items()
            if key not in {"metrics", "by_simulator"}
        },
    }
    return EvaluationResult(metrics=metrics, predictions=predictions)


def _training_state(
    *,
    model: PortableSafetyCore,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    best_epoch: int | None,
    best_threshold: float | None,
    stale_epochs: int,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_STATE_SCHEMA,
        "saved_at": _now(),
        "next_epoch": int(epoch + 1),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "stale_epochs": int(stale_epochs),
        "config_sha256": config_sha256,
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
    }


def _load_training_state(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != TRAINING_STATE_SCHEMA:
        raise RuntimeError(f"Unsupported training state: {path}")
    return payload


def _checkpoint_metadata(
    *,
    config_sha256: str,
    git_metadata: dict[str, Any],
    source_hashes: dict[str, str],
    stats_metadata: dict[str, Any],
    train_summary: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "training_source": "mixed-robotwin-maniskill-raw-input-v2",
        "config_sha256": config_sha256,
        "git": git_metadata,
        "source_hashes": source_hashes,
        "normalization": stats_metadata,
        "train_dataset": train_summary,
        "validation": validation,
        "policy_internal_features_required": False,
        "host_policy_parameters_updated": False,
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    seed = int(config.get("seed", 7))
    _seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    requested_device = str(config.get("device", "cuda"))
    if requested_device == "cuda":
        requested_device = "cuda:0"
    device = torch.device(requested_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        fraction = float(config.get("cuda_memory_fraction", 1.0))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("cuda_memory_fraction must be in (0,1]")
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _config_sha256(config)
    _atomic_json(output_dir / "config.resolved.json", public_config(config))

    profile, stats_metadata = _build_profile(config)
    train_dataset = PortableSafetyManifestDataset.from_config(config["dataset"], "train")
    val_dataset = PortableSafetyManifestDataset.from_config(config["dataset"], "val")
    train_loader = _loader(train_dataset, config, device, training=True)
    val_loader = _loader(val_dataset, config, device, training=False)
    data_summary = {
        "train": train_dataset.summary(),
        "val": val_dataset.summary(),
        "normalization": stats_metadata,
    }
    _atomic_json(output_dir / "dataset_summary.json", data_summary)

    model = _build_model(config, profile).to(device)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"].get("learning_rate", 3e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-3)),
    )
    amp_enabled, amp_dtype, scaler = _amp_settings(config, device)
    chunk_class_weights = train_dataset.class_weights(step=False).to(device)
    step_class_weights = train_dataset.class_weights(
        step=True,
        maximum_ratio=float(config["loss"].get("maximum_step_class_weight_ratio", 6.0)),
    ).to(device)
    training_config = config["training"]
    epochs = int(training_config.get("epochs", 20))
    max_steps_raw = training_config.get("max_steps")
    max_steps = None if max_steps_raw is None else int(max_steps_raw)
    if epochs < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("epochs and max_steps must be positive")
    max_grad_norm = float(training_config.get("max_grad_norm", 1.0))
    patience = int(training_config.get("early_stopping_patience", 6))
    min_delta = float(training_config.get("early_stopping_min_delta", 1e-4))
    log_every = max(1, int(training_config.get("log_every_steps", 10)))

    start_epoch = 0
    global_step = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    best_threshold: float | None = None
    stale_epochs = 0
    training_state_path = output_dir / "training_state.pt"
    if bool(training_config.get("resume", True)) and training_state_path.is_file():
        state = _load_training_state(training_state_path)
        if state["config_sha256"] != config_hash:
            raise RuntimeError("Existing training state uses a different configuration")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["next_epoch"])
        global_step = int(state["global_step"])
        best_metric = state.get("best_metric")
        best_epoch = state.get("best_epoch")
        best_threshold = state.get("best_threshold")
        stale_epochs = int(state.get("stale_epochs", 0))
        LOGGER.info("Resuming at epoch=%s global_step=%s", start_epoch, global_step)

    project_root = Path(__file__).resolve().parents[3]
    git_metadata = _git_metadata(project_root)
    source_hashes = _source_hashes()
    runtime = {
        "started_at": _now(),
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_memory_fraction": config.get("cuda_memory_fraction"),
        "trainable_parameters": trainable_parameter_count(model),
        "class_weights": chunk_class_weights.detach().cpu().tolist(),
        "step_class_weights": step_class_weights.detach().cpu().tolist(),
        "profile": profile.to_dict(),
        "profile_fingerprint": profile.fingerprint,
        "git": git_metadata,
        "source_hashes": source_hashes,
    }
    _atomic_json(output_dir / "runtime.json", runtime)

    metrics_path = output_dir / "metrics.jsonl"
    if start_epoch == 0 and metrics_path.exists():
        metrics_path.unlink()
    stopped_reason = "epochs_completed"
    last_train_loss = float("nan")
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_steps = 0
        for batch in train_loader:
            value = _to_safety_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                outputs = model(value)
                loss, terms = portable_safety_loss(
                    outputs,
                    batch["chunk_target"],
                    step_target=batch["step_target"],
                    chunk_weight=float(config["loss"].get("chunk_weight", 1.0)),
                    step_weight=float(config["loss"].get("step_weight", 0.35)),
                    class_weights=chunk_class_weights,
                    step_class_weights=step_class_weights,
                    ignore_index=IGNORE_STEP_LABEL,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={global_step}")
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(batch["sample_id"])
            epoch_loss += float(loss.detach().item()) * batch_size
            epoch_samples += batch_size
            epoch_steps += 1
            global_step += 1
            if global_step % log_every == 0:
                LOGGER.info(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            **{
                                key: float(value.item())
                                for key, value in terms.items()
                            },
                        },
                        sort_keys=True,
                    )
                )
            if max_steps is not None and global_step >= max_steps:
                stopped_reason = "max_steps"
                break
        if epoch_steps == 0:
            raise RuntimeError("Training loader produced no batches")
        last_train_loss = epoch_loss / epoch_samples
        validation = evaluate(
            model,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            chunk_class_weights=chunk_class_weights,
            step_class_weights=step_class_weights,
            loss_config=config["loss"],
            threshold_config=config["validation"]["threshold_search"],
        )
        candidate = float(
            validation.metrics["threshold_selection"]["objective_value"]
        )
        improved = best_metric is None or candidate > best_metric + min_delta
        if improved:
            best_metric = candidate
            best_epoch = epoch
            best_threshold = float(
                validation.metrics["threshold_selection"]["threshold"]
            )
            stale_epochs = 0
            metadata = _checkpoint_metadata(
                config_sha256=config_hash,
                git_metadata=git_metadata,
                source_hashes=source_hashes,
                stats_metadata=stats_metadata,
                train_summary=data_summary["train"],
                validation=validation.metrics,
            )
            metadata.update({"epoch": epoch, "global_step": global_step})
            save_portable_checkpoint(
                output_dir / "best.pt",
                model,
                profile,
                thresholds=SafetyThresholds(
                    chunk_risk=best_threshold, step_risk=best_threshold
                ),
                metadata=metadata,
            )
            _atomic_jsonl(output_dir / "best_val_predictions.jsonl", validation.predictions)
        else:
            stale_epochs += 1

        current_threshold = float(
            validation.metrics["threshold_selection"]["threshold"]
        )
        save_portable_checkpoint(
            output_dir / "latest.pt",
            model,
            profile,
            thresholds=SafetyThresholds(
                chunk_risk=current_threshold, step_risk=current_threshold
            ),
            metadata=_checkpoint_metadata(
                config_sha256=config_hash,
                git_metadata=git_metadata,
                source_hashes=source_hashes,
                stats_metadata=stats_metadata,
                train_summary=data_summary["train"],
                validation=validation.metrics,
            ),
        )
        _atomic_torch_save(
            training_state_path,
            _training_state(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                best_epoch=best_epoch,
                best_threshold=best_threshold,
                stale_epochs=stale_epochs,
                config_sha256=config_hash,
            ),
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": last_train_loss,
            "train_samples": epoch_samples,
            "train_steps": epoch_steps,
            "validation": validation.metrics,
            "selection_value": candidate,
            "is_best": improved,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "stale_epochs": stale_epochs,
            "timestamp": _now(),
        }
        _append_jsonl(metrics_path, record)
        LOGGER.info(json.dumps(record, ensure_ascii=False, sort_keys=True))
        if max_steps is not None and global_step >= max_steps:
            break
        if patience >= 0 and stale_epochs > patience:
            stopped_reason = "early_stopping"
            break

    best_path = output_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("Training completed without a best checkpoint")
    loaded = load_portable_checkpoint(best_path, map_location=device, expected_profile=profile)
    fixed_threshold = float(loaded.thresholds.chunk_risk)
    final_validation = evaluate(
        loaded.model,
        val_loader,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        chunk_class_weights=chunk_class_weights,
        step_class_weights=step_class_weights,
        loss_config=config["loss"],
        threshold_config=config["validation"]["threshold_search"],
        fixed_threshold=fixed_threshold,
    )
    _atomic_json(output_dir / "val_metrics.json", final_validation.metrics)
    _atomic_jsonl(output_dir / "val_predictions.jsonl", final_validation.predictions)

    probe_batch = next(iter(val_loader))
    original_action = probe_batch["action"].clone()
    probe_value = _to_safety_batch(probe_batch, device)
    probe_value.validate(profile)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        first_probability = loaded.model(probe_value)["risk_probability"].float()
    reloaded = load_portable_checkpoint(best_path, map_location=device, expected_profile=profile)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        second_probability = reloaded.model(probe_value)["risk_probability"].float()
    reload_difference = float(
        (first_probability - second_probability).abs().max().item()
    )
    action_unchanged = bool(torch.equal(original_action, probe_batch["action"]))
    peak_memory_mib = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "cuda"
        else 0.0
    )
    elapsed = time.monotonic() - started
    summary = {
        **runtime,
        "finished_at": _now(),
        "elapsed_seconds": elapsed,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_threshold": best_threshold,
        "stopped_reason": stopped_reason,
        "last_train_loss": last_train_loss,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "validation": final_validation.metrics,
        "checkpoint_reload_max_abs_error": reload_difference,
        "action_chunk_unchanged": action_unchanged,
        "peak_allocated_mib": peak_memory_mib,
        "config_sha256": config_hash,
    }
    _atomic_json(output_dir / "summary.json", summary)

    if bool(config.get("sanity", {}).get("enabled", False)):
        expected_strata = set(
            str(value)
            for value in config["sanity"].get(
                "expected_train_strata",
                ("maniskill/risk", "maniskill/safe", "robotwin/risk", "robotwin/safe"),
            )
        )
        actual_strata = set(data_summary["train"]["strata"])
        checks = {
            "finite_loss": bool(np.isfinite(last_train_loss)),
            "checkpoint_reload_exact": reload_difference == 0.0,
            "action_chunk_unchanged": action_unchanged,
            "all_expected_strata_present": expected_strata <= actual_strata,
            "explicit_mixed_action_dt": len(set(probe_batch["action_dt"].tolist())) > 1,
        }
        sanity = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "training_steps": global_step,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_strata": data_summary["train"]["strata"],
            "loss": last_train_loss,
            "checkpoint_path": str(best_path),
            "checkpoint_sha256": sha256_file(best_path),
            "checkpoint_reload_max_abs_error": reload_difference,
            "peak_allocated_mib": peak_memory_mib,
            "profile_fingerprint": profile.fingerprint,
            "config_sha256": config_hash,
        }
        _atomic_json(output_dir / "SANITY_RESULT.json", sanity)
        if sanity["status"] != "PASS":
            raise RuntimeError(f"Portable safety sanity checks failed: {checks}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the portable safety sidecar")
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
