from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .contracts import PORTABLE_CLASS_NAMES, PORTABLE_INPUT_SCHEMA, RobotProfile
from .guard import ActionGuardConfig, DeterministicActionGuard
from .model import PortableSafetyConfig, PortableSafetyCore
from .runtime import PortableSafetyRuntime, SafetyThresholds


PORTABLE_CHECKPOINT_SCHEMA = "portable_safety_sidecar/v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass(frozen=True)
class LoadedPortableSafety:
    model: PortableSafetyCore
    profile: RobotProfile
    thresholds: SafetyThresholds
    guard: DeterministicActionGuard | None
    runtime: PortableSafetyRuntime
    metadata: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str


def save_portable_checkpoint(
    path: str | Path,
    model: PortableSafetyCore,
    profile: RobotProfile,
    *,
    thresholds: SafetyThresholds | None = None,
    guard_config: ActionGuardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    if model.config.state_dim != profile.state_dim:
        raise ValueError("Model state_dim differs from robot profile")
    if model.config.action_dim != profile.action_dim:
        raise ValueError("Model action_dim differs from robot profile")
    if len(profile.camera_names) > model.config.max_views:
        raise ValueError("Robot profile has more cameras than model max_views")
    if guard_config is not None:
        if guard_config.action_dim != profile.action_dim:
            raise ValueError("Guard action_dim differs from robot profile")
        if abs(guard_config.control_dt - profile.control_dt) > 1e-12:
            raise ValueError("Guard control_dt differs from robot profile")
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PORTABLE_CHECKPOINT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_schema": PORTABLE_INPUT_SCHEMA,
        "class_names": list(PORTABLE_CLASS_NAMES),
        "policy_internal_features_required": False,
        "model_config": model.config.to_dict(),
        "robot_profile": profile.to_dict(),
        "robot_profile_fingerprint": profile.fingerprint,
        "thresholds": (thresholds or SafetyThresholds()).to_dict(),
        "guard_config": None if guard_config is None else guard_config.to_dict(),
        "model_state": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)
    return checkpoint_path


def load_portable_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_profile: RobotProfile | None = None,
) -> LoadedPortableSafety:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = _trusted_load(checkpoint_path, map_location)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Portable safety checkpoint is not a mapping: {checkpoint_path}"
        )
    if payload.get("schema_version") != PORTABLE_CHECKPOINT_SCHEMA:
        raise RuntimeError(
            f"Unsupported portable safety checkpoint schema at {checkpoint_path}"
        )
    if payload.get("input_schema") != PORTABLE_INPUT_SCHEMA:
        raise RuntimeError("Portable safety input schema mismatch")
    if tuple(payload.get("class_names", ())) != PORTABLE_CLASS_NAMES:
        raise RuntimeError("Portable safety class order mismatch")
    if payload.get("policy_internal_features_required") is not False:
        raise RuntimeError("Checkpoint unexpectedly depends on policy hidden features")

    profile = RobotProfile.from_dict(payload["robot_profile"])
    if payload.get("robot_profile_fingerprint") != profile.fingerprint:
        raise RuntimeError("Robot profile fingerprint in checkpoint is invalid")
    if (
        expected_profile is not None
        and expected_profile.fingerprint != profile.fingerprint
    ):
        raise RuntimeError(
            "Requested robot profile differs from checkpoint profile: "
            f"requested={expected_profile.fingerprint}, "
            f"checkpoint={profile.fingerprint}"
        )
    model = PortableSafetyCore(
        PortableSafetyConfig.from_dict(payload["model_config"])
    )
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("Portable safety checkpoint has no model_state")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("Portable safety model tensors do not match config") from exc
    model.to(map_location)
    model.eval()
    thresholds = SafetyThresholds.from_dict(payload["thresholds"])
    raw_guard = payload.get("guard_config")
    guard = (
        None
        if raw_guard is None
        else DeterministicActionGuard(ActionGuardConfig.from_dict(raw_guard))
    )
    runtime = PortableSafetyRuntime(
        model,
        profile,
        thresholds=thresholds,
        guard=guard,
    )
    return LoadedPortableSafety(
        model=model,
        profile=profile,
        thresholds=thresholds,
        guard=guard,
        runtime=runtime,
        metadata=dict(payload.get("metadata", {})),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
    )
