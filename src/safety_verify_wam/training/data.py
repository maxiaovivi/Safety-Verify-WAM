from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import resolve_project_path


MODEL_INPUT_KEYS = ("image", "state", "action", "text_embeddings")
RISK_CLASS_TO_INDEX = {"safe": 0, "boundary": 1, "risk": 2}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            record["_line_number"] = line_number
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def _resolve_data_path(root: Path, value: str, field_name: str) -> Path:
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def load_image_tensor(
    path: str | Path,
    image_size: Sequence[int],
    size_policy: str = "error",
) -> torch.Tensor:
    expected_h, expected_w = (int(value) for value in image_size)
    image_path = Path(path)
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        actual_w, actual_h = image.size
        if (actual_h, actual_w) != (expected_h, expected_w):
            if size_policy == "resize":
                image = image.resize((expected_w, expected_h), Image.Resampling.BILINEAR)
            elif size_policy == "error":
                raise ValueError(
                    f"Image {image_path} has size {(actual_h, actual_w)}, "
                    f"expected {(expected_h, expected_w)}"
                )
            else:
                raise ValueError(f"Unsupported image_size_policy: {size_policy!r}")
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_numeric_tensor(value: Any, root: Path, field_name: str) -> torch.Tensor:
    if isinstance(value, str):
        array_path = _resolve_data_path(root, value, field_name)
        if array_path.suffix.lower() != ".npy":
            raise ValueError(f"{field_name} files must use .npy: {array_path}")
        array = np.load(array_path, allow_pickle=False)
    elif isinstance(value, list):
        array = np.asarray(value, dtype=np.float32)
    else:
        raise TypeError(f"Each record needs {field_name} or an inline numeric array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{field_name} array must be numeric")
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def load_action_tensor(value: Any, root: Path) -> torch.Tensor:
    return _load_numeric_tensor(value, root, "action_path")


def load_state_tensor(value: Any, root: Path) -> torch.Tensor:
    return _load_numeric_tensor(value, root, "state_path")


def load_text_embedding_tensor(value: Any, root: Path) -> torch.Tensor:
    return _load_numeric_tensor(value, root, "text_embedding_path")


def validate_action_tensor(
    action: torch.Tensor,
    action_shape: Sequence[int],
    require_normalized: bool,
    normalized_limit: float,
) -> None:
    expected = tuple(int(value) for value in action_shape)
    if tuple(action.shape) != expected:
        raise ValueError(f"Action has shape {tuple(action.shape)}, expected {expected}")
    if not torch.isfinite(action).all():
        raise ValueError("Action contains NaN or infinity")
    if require_normalized and action.abs().max().item() > float(normalized_limit):
        raise ValueError(
            f"Normalized action exceeds ±{normalized_limit}; normalize with the same "
            "statistics used by the Efficient-WAM checkpoint"
        )


def validate_state_tensor(
    state: torch.Tensor,
    state_shape: Sequence[int],
    require_normalized: bool,
    normalized_limit: float,
) -> None:
    expected = tuple(int(value) for value in state_shape)
    if tuple(state.shape) != expected:
        raise ValueError(f"State has shape {tuple(state.shape)}, expected {expected}")
    if not torch.isfinite(state).all():
        raise ValueError("State contains NaN or infinity")
    if require_normalized and state.abs().max().item() > float(normalized_limit):
        raise ValueError(
            f"Normalized state exceeds ±{normalized_limit}; normalize with the same "
            "statistics used by the Efficient-WAM checkpoint"
        )


def validate_text_embedding_tensor(
    text_embeddings: torch.Tensor,
    embedding_dim: int,
) -> None:
    if text_embeddings.ndim != 2 or text_embeddings.shape[-1] != int(embedding_dim):
        raise ValueError(
            f"Text embedding must be [L,{embedding_dim}], "
            f"got {tuple(text_embeddings.shape)}"
        )
    if text_embeddings.shape[0] < 1:
        raise ValueError("Text embedding sequence cannot be empty")
    if not torch.isfinite(text_embeddings).all():
        raise ValueError("Text embedding contains NaN or infinity")


def parse_risk_class(value: Any) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in RISK_CLASS_TO_INDEX:
            raise ValueError(
                f"Unknown risk class {value!r}; expected one of "
                f"{tuple(RISK_CLASS_TO_INDEX)}"
            )
        return RISK_CLASS_TO_INDEX[key]
    if (
        isinstance(value, (int, float, np.integer, np.floating))
        and float(value).is_integer()
        and int(value) in {0, 1, 2}
    ):
        return int(value)
    raise ValueError("risk must be safe/boundary/risk or integer 0/1/2")


class SafetyManifestDataset(Dataset[dict[str, Any]]):
    """JSONL safety data matching the original Efficient-WAM conditions."""

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        image_size: Sequence[int] = (384, 320),
        image_size_policy: str = "error",
        state_shape: Sequence[int] = (14,),
        action_shape: Sequence[int] = (16, 14),
        text_embedding_dim: int = 4096,
        require_normalized_actions: bool = True,
        normalized_action_limit: float = 1.05,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = Path(manifest).expanduser()
        self.manifest_path = (
            manifest_path.resolve() if manifest_path.is_absolute() else (self.root / manifest_path).resolve()
        )
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.manifest_path}")
        self.records = _read_jsonl(self.manifest_path)
        self.image_size = tuple(int(value) for value in image_size)
        self.image_size_policy = str(image_size_policy)
        self.state_shape = tuple(int(value) for value in state_shape)
        self.action_shape = tuple(int(value) for value in action_shape)
        self.text_embedding_dim = int(text_embedding_dim)
        self.require_normalized_actions = bool(require_normalized_actions)
        self.normalized_action_limit = float(normalized_action_limit)

    @classmethod
    def from_config(cls, dataset_config: dict[str, Any], split: str) -> "SafetyManifestDataset":
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split!r}")
        root = resolve_project_path(dataset_config["root"])
        return cls(
            root=root,
            manifest=dataset_config[f"{split}_manifest"],
            image_size=dataset_config.get("image_size", (384, 320)),
            image_size_policy=dataset_config.get("image_size_policy", "error"),
            state_shape=dataset_config.get("state_shape", (14,)),
            action_shape=dataset_config.get("action_shape", (16, 14)),
            text_embedding_dim=int(dataset_config.get("text_embedding_dim", 4096)),
            require_normalized_actions=dataset_config.get("require_normalized_actions", True),
            normalized_action_limit=dataset_config.get("normalized_action_limit", 1.05),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        line = int(record["_line_number"])
        sample_id = str(record.get("sample_id", f"line-{line}"))
        try:
            image_path = _resolve_data_path(self.root, str(record["image_path"]), "image_path")
            state_value = record.get("state_path", record.get("state"))
            action_value = record.get("action_path", record.get("action"))
            text_value = record.get(
                "text_embedding_path", record.get("text_embeddings")
            )
            image = load_image_tensor(image_path, self.image_size, self.image_size_policy)
            state = load_state_tensor(state_value, self.root)
            action = load_action_tensor(action_value, self.root)
            text_embeddings = load_text_embedding_tensor(text_value, self.root)
            validate_state_tensor(
                state,
                self.state_shape,
                self.require_normalized_actions,
                self.normalized_action_limit,
            )
            validate_action_tensor(
                action,
                self.action_shape,
                self.require_normalized_actions,
                self.normalized_action_limit,
            )
            validate_text_embedding_tensor(
                text_embeddings,
                self.text_embedding_dim,
            )
            risk = parse_risk_class(record["risk"])
            item: dict[str, Any] = {
                "sample_id": sample_id,
                "image": image,
                "state": state,
                "action": action,
                "text_embeddings": text_embeddings,
                "risk": torch.tensor(risk, dtype=torch.long),
            }
            if "risk_steps" in record:
                risk_steps = torch.tensor(
                    [parse_risk_class(value) for value in record["risk_steps"]],
                    dtype=torch.long,
                )
                if tuple(risk_steps.shape) != (self.action_shape[0],):
                    raise ValueError(
                        f"risk_steps has shape {tuple(risk_steps.shape)}, "
                        f"expected {(self.action_shape[0],)}"
                    )
                item["risk_steps"] = risk_steps
            if "risk_type" in record:
                item["risk_type"] = int(record["risk_type"])
            return item
        except Exception as exc:
            raise ValueError(
                f"{self.manifest_path}:{line} sample={sample_id}: {exc}"
            ) from exc


def safety_collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch: dict[str, Any] = {
        "sample_id": [item["sample_id"] for item in items],
        "image": torch.stack([item["image"] for item in items]),
        "state": torch.stack([item["state"] for item in items]),
        "action": torch.stack([item["action"] for item in items]),
        "text_embeddings": [item["text_embeddings"] for item in items],
        "risk": torch.stack([item["risk"] for item in items]),
    }
    action_steps = int(items[0]["action"].shape[0])
    step_available = torch.tensor(["risk_steps" in item for item in items], dtype=torch.bool)
    batch["risk_steps_available"] = step_available
    batch["risk_steps"] = torch.stack(
        [
            item.get(
                "risk_steps",
                torch.full((action_steps,), -100, dtype=torch.long),
            )
            for item in items
        ]
    )
    batch["risk_type"] = torch.tensor(
        [int(item.get("risk_type", -100)) for item in items], dtype=torch.long
    )
    return batch


def model_inputs_from_batch(
    batch: dict[str, Any],
    device: torch.device | str,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """The only transfer point from dataset records into the network."""

    return {
        "image": batch["image"].to(device, non_blocking=True),
        "state": batch["state"].to(device, non_blocking=True),
        "action": batch["action"].to(device, non_blocking=True),
        "text_embeddings": [
            item.to(device, non_blocking=True) for item in batch["text_embeddings"]
        ],
    }


def iter_manifest_errors(dataset: SafetyManifestDataset) -> Iterable[str]:
    for index in range(len(dataset)):
        try:
            dataset[index]
        except Exception as exc:
            yield str(exc)
