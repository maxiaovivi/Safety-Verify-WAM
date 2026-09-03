from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CHUNK_LABELS = {"safe": 0, "risk": 1}
STEP_LABELS = {"safe": 0, "risk": 1}
IGNORE_STEP_LABEL = -100
DEFAULT_CAMERA_FIELDS = ("head_rgb_path", "left_rgb_path", "right_rgb_path")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row is not a mapping at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest contains no samples: {path}")
    return rows


def _stable_rank(row: dict[str, Any], seed: int) -> str:
    value = f"{seed}:{row.get('sample_id', '')}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _stratified_limit(
    rows: list[dict[str, Any]],
    limit: int | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise ValueError("Sample limit must be positive or null")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row.get("simulator_key")), str(row.get("risk")))].append(row)
    keys = sorted(strata)
    if limit < len(keys):
        raise ValueError(
            f"Sample limit {limit} is smaller than the {len(keys)} input strata"
        )
    for values in strata.values():
        values.sort(key=lambda item: _stable_rank(item, seed))
    quotas = {key: limit // len(keys) for key in keys}
    for key in keys[: limit % len(keys)]:
        quotas[key] += 1
    selected: list[dict[str, Any]] = []
    spare = 0
    for key in keys:
        take = min(quotas[key], len(strata[key]))
        selected.extend(strata[key][:take])
        spare += quotas[key] - take
    if spare:
        leftovers = [
            row
            for key in keys
            for row in strata[key][quotas[key] :]
        ]
        leftovers.sort(key=lambda item: _stable_rank(item, seed + 1))
        selected.extend(leftovers[:spare])
    selected.sort(key=lambda item: _stable_rank(item, seed + 2))
    if len(selected) != limit:
        raise RuntimeError(f"Selected {len(selected)} samples, expected {limit}")
    return selected


def _timestamp_values(
    row: dict[str, Any],
    key: str,
    *,
    steps: int,
    dt: float,
    future: bool,
) -> np.ndarray:
    explicit = row.get(key)
    if explicit is not None:
        values = np.asarray(explicit, dtype=np.float32).reshape(-1)
        if values.shape != (steps,):
            raise ValueError(
                f"{row.get('sample_id')} {key} has shape {values.shape}, "
                f"expected {(steps,)}"
            )
        return values
    if future:
        return np.arange(1, steps + 1, dtype=np.float32) * np.float32(dt)
    return np.arange(1 - steps, 1, dtype=np.float32) * np.float32(dt)


class PortableSafetyManifestDataset(Dataset[dict[str, Any]]):
    """Load the raw RGB/state/action contract used by the portable sidecar."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: tuple[int, int] = (128, 128),
        camera_fields: tuple[str, ...] = DEFAULT_CAMERA_FIELDS,
        sample_limit: int | None = None,
        seed: int = 7,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.image_size = tuple(int(value) for value in image_size)
        if len(self.image_size) != 2 or any(value < 16 for value in self.image_size):
            raise ValueError("image_size must contain height and width of at least 16")
        self.camera_fields = tuple(str(value) for value in camera_fields)
        if not self.camera_fields:
            raise ValueError("camera_fields cannot be empty")
        rows = _read_jsonl(self.manifest_path)
        self.rows = _stratified_limit(rows, sample_limit, seed=seed)
        self._validate_rows()

    def _validate_rows(self) -> None:
        seen: set[str] = set()
        required = {
            "sample_id",
            "simulator_key",
            "risk",
            "state_path",
            "action_path",
            "action_dt",
            "risk_steps",
            *self.camera_fields,
        }
        for index, row in enumerate(self.rows):
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"Manifest row {index} is missing fields: {missing}")
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
            seen.add(sample_id)
            if row["risk"] not in CHUNK_LABELS:
                raise ValueError(f"Unsupported chunk label for {sample_id}: {row['risk']}")
            if float(row["action_dt"]) <= 0:
                raise ValueError(f"Non-positive action_dt for {sample_id}")
            if not isinstance(row["risk_steps"], list):
                raise ValueError(f"risk_steps must be a list for {sample_id}")

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        split: str,
    ) -> "PortableSafetyManifestDataset":
        manifest_root = Path(config["manifest_root"]).expanduser().resolve()
        manifest_path = manifest_root / "manifests" / f"{split}.jsonl"
        limit = config.get(f"max_{split}_samples")
        return cls(
            manifest_path,
            image_size=tuple(config.get("image_size", (128, 128))),
            camera_fields=tuple(config.get("camera_fields", DEFAULT_CAMERA_FIELDS)),
            sample_limit=None if limit is None else int(limit),
            seed=int(config.get("sample_seed", 7)),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_asset_path(self, path: str | Path) -> Path:
        asset_path = Path(path).expanduser()
        if not asset_path.is_absolute():
            asset_path = self.manifest_path.parent.parent / asset_path
        return asset_path.resolve()

    def _load_image(self, path: str | Path) -> torch.Tensor:
        image_path = self._resolve_asset_path(path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.Resampling.BILINEAR,
            )
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _load_array(self, path: str | Path, *, name: str) -> np.ndarray:
        array_path = self._resolve_asset_path(path)
        if not array_path.is_file():
            raise FileNotFoundError(array_path)
        array = np.asarray(np.load(array_path, allow_pickle=False), dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity: {array_path}")
        return array

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = str(row["sample_id"])
        camera_paths = [
            value if isinstance(value, list) else [value]
            for value in (row[field] for field in self.camera_fields)
        ]
        frame_counts = {len(paths) for paths in camera_paths}
        if len(frame_counts) != 1 or not frame_counts or next(iter(frame_counts)) < 1:
            raise ValueError(
                f"{sample_id} camera fields must have the same positive frame count"
            )
        frame_count = next(iter(frame_counts))
        views = torch.stack(
            [
                torch.stack(
                    [self._load_image(paths[frame]) for paths in camera_paths],
                    dim=0,
                )
                for frame in range(frame_count)
            ],
            dim=0,
        )
        state = self._load_array(row["state_path"], name="state")
        action = self._load_array(row["action_path"], name="action")
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim != 2:
            raise ValueError(f"{sample_id} state must be [T,S], got {state.shape}")
        if action.ndim != 2:
            raise ValueError(f"{sample_id} action must be [T,A], got {action.shape}")
        if state.shape[-1] != action.shape[-1]:
            raise ValueError(
                f"{sample_id} state/action dimensions differ: "
                f"{state.shape[-1]} vs {action.shape[-1]}"
            )
        raw_step_labels = list(row["risk_steps"])
        if len(raw_step_labels) != action.shape[0]:
            raise ValueError(
                f"{sample_id} has {len(raw_step_labels)} step labels for "
                f"{action.shape[0]} actions"
            )
        step_target = np.asarray(
            [STEP_LABELS.get(str(value), IGNORE_STEP_LABEL) for value in raw_step_labels],
            dtype=np.int64,
        )
        dt = float(row["action_dt"])
        return {
            "sample_id": sample_id,
            "simulator_key": str(row["simulator_key"]),
            "observation_group_id": str(
                row.get("observation_group_id", sample_id)
            ),
            "chunk_target": CHUNK_LABELS[str(row["risk"])],
            "step_target": torch.from_numpy(step_target),
            "video": views,
            "state": torch.from_numpy(state.copy()),
            "action": torch.from_numpy(action.copy()),
            "video_timestamps": torch.from_numpy(
                _timestamp_values(
                    row,
                    "video_timestamps",
                    steps=frame_count,
                    dt=dt,
                    future=False,
                )
            ),
            "state_timestamps": torch.from_numpy(
                _timestamp_values(
                    row,
                    "state_timestamps",
                    steps=state.shape[0],
                    dt=dt,
                    future=False,
                )
            ),
            "action_timestamps": torch.from_numpy(
                _timestamp_values(
                    row,
                    "action_timestamps",
                    steps=action.shape[0],
                    dt=dt,
                    future=True,
                )
            ),
            "action_dt": dt,
        }

    def summary(self) -> dict[str, Any]:
        strata = Counter(
            (str(row["simulator_key"]), str(row["risk"])) for row in self.rows
        )
        step_labels = Counter(
            str(value) for row in self.rows for value in row["risk_steps"]
        )
        video_frame_counts = Counter(
            len(row[self.camera_fields[0]])
            if isinstance(row[self.camera_fields[0]], list)
            else 1
            for row in self.rows
        )
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": file_sha256(self.manifest_path),
            "samples": len(self.rows),
            "groups": len(
                {
                    str(row.get("observation_group_id", row["sample_id"]))
                    for row in self.rows
                }
            ),
            "strata": {
                f"{simulator}/{label}": count
                for (simulator, label), count in sorted(strata.items())
            },
            "step_labels": dict(sorted(step_labels.items())),
            "video_frame_counts": {
                str(frames): count
                for frames, count in sorted(video_frame_counts.items())
            },
            "image_size": list(self.image_size),
            "camera_fields": list(self.camera_fields),
        }

    def stratum_weights(self) -> torch.Tensor:
        counts = Counter(
            (str(row["simulator_key"]), str(row["risk"])) for row in self.rows
        )
        return torch.tensor(
            [
                1.0 / counts[(str(row["simulator_key"]), str(row["risk"]))]
                for row in self.rows
            ],
            dtype=torch.double,
        )

    def class_weights(self, *, step: bool, maximum_ratio: float = 6.0) -> torch.Tensor:
        if maximum_ratio < 1.0:
            raise ValueError("maximum_ratio must be at least 1")
        if step:
            counts = Counter(
                STEP_LABELS[str(value)]
                for row in self.rows
                for value in row["risk_steps"]
                if str(value) in STEP_LABELS
            )
        else:
            counts = Counter(CHUNK_LABELS[str(row["risk"])] for row in self.rows)
        if set(counts) != {0, 1}:
            raise ValueError(f"Both safe and risk labels are required, got {counts}")
        inverse = torch.tensor(
            [sum(counts.values()) / (2.0 * counts[index]) for index in (0, 1)],
            dtype=torch.float32,
        )
        lower = float(inverse.min().item())
        inverse = inverse.clamp(max=lower * maximum_ratio)
        return inverse / inverse.mean()


def _pad_sequence(
    values: list[torch.Tensor],
    *,
    fill_value: float | int,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(int(value.shape[0]) for value in values)
    shape = (len(values), maximum, *values[0].shape[1:])
    output = values[0].new_full(shape, fill_value)
    mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, value in enumerate(values):
        length = int(value.shape[0])
        if value.shape[1:] != values[0].shape[1:]:
            raise ValueError("Cannot collate tensors with different feature dimensions")
        output[index, :length] = value
        mask[index, :length] = True
    return output, mask


def portable_safety_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    video, video_frame_mask = _pad_sequence(
        [sample["video"] for sample in samples], fill_value=0
    )
    state, state_mask = _pad_sequence(
        [sample["state"] for sample in samples], fill_value=0.0
    )
    action, action_mask = _pad_sequence(
        [sample["action"] for sample in samples], fill_value=0.0
    )
    video_timestamps, _ = _pad_sequence(
        [sample["video_timestamps"] for sample in samples], fill_value=0.0
    )
    state_timestamps, _ = _pad_sequence(
        [sample["state_timestamps"] for sample in samples], fill_value=0.0
    )
    action_timestamps, _ = _pad_sequence(
        [sample["action_timestamps"] for sample in samples], fill_value=0.0
    )
    step_target, _ = _pad_sequence(
        [sample["step_target"] for sample in samples],
        fill_value=IGNORE_STEP_LABEL,
    )
    views = int(video.shape[2])
    video_mask = video_frame_mask.unsqueeze(-1).expand(-1, -1, views).clone()
    return {
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "simulator_key": [str(sample["simulator_key"]) for sample in samples],
        "observation_group_id": [
            str(sample["observation_group_id"]) for sample in samples
        ],
        "chunk_target": torch.tensor(
            [int(sample["chunk_target"]) for sample in samples], dtype=torch.long
        ),
        "step_target": step_target,
        "video": video,
        "state": state,
        "action": action,
        "video_timestamps": video_timestamps,
        "state_timestamps": state_timestamps,
        "action_timestamps": action_timestamps,
        "video_mask": video_mask,
        "state_mask": state_mask,
        "action_mask": action_mask,
        "action_dt": torch.tensor(
            [float(sample["action_dt"]) for sample in samples], dtype=torch.float32
        ),
    }
