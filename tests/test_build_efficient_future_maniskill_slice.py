from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_efficient_future_maniskill_slice.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_efficient_future_maniskill_slice", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sample(root: Path, *, label: str) -> dict[str, object]:
    group = "scene-001"
    task = "TwoRobotComplexFruit-v1"
    common = root / group
    common.mkdir(parents=True, exist_ok=True)
    np.save(common / "state.npy", np.zeros(14, dtype=np.float32))
    np.save(common / f"action-{label}.npy", np.zeros((16, 14), dtype=np.float32))
    np.save(common / "text.npy", np.zeros((4, 4096), dtype=np.float32))
    (common / "image.png").write_bytes(b"fixture")
    return {
        "sample_id": f"{group}-{label}",
        "scene_group_id": group,
        "split": "train",
        "risk": label,
        "risk_steps": [label] * 16,
        "task_id": task,
        "scene_seed": 1,
        "scene_variant": "randomized",
        "action_dt": 0.1,
        "image_path": f"{group}/image.png",
        "state_raw_path": f"{group}/state.npy",
        "action_raw_path": f"{group}/action-{label}.npy",
        "text_embedding_path": f"{group}/text.npy",
        "snapshot_path": f"{group}/snapshot.pt",
        "snapshot_sha256": "same",
        "family": "left_object_impact",
        "object_superclass": "fruit",
    }


def test_builds_matched_maniskill_pair_without_copying_source(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "manifests").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "dataset.json").write_text(json.dumps({"status": "complete"}))
    (root / "reports" / "full-audit.json").write_text(
        json.dumps({"status": "PASS"})
    )
    rows = [_sample(root, label="safe"), _sample(root, label="risk")]
    (root / "manifests" / "all.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    output = tmp_path / "slice"
    result = MODULE.build(
        argparse.Namespace(
            source_root=root,
            output=output,
            max_train_groups=0,
            max_val_groups=0,
            max_test_groups=0,
            seed=7,
        )
    )

    assert result["groups"] == 1
    assert result["windows"] == 2
    windows = [json.loads(line) for line in (output / "windows.jsonl").read_text().splitlines()]
    assert {row["chunk_target"] for row in windows} == {0, 1}
    assert {row["split"] for row in windows} == {"train"}
    assert all(row["path_root"] == str(root.resolve()) for row in windows)
    assert not (output / "scene-001").exists()
