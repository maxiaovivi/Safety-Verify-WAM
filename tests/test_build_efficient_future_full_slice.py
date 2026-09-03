from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_efficient_future_full_slice.py"
SPEC = importlib.util.spec_from_file_location("build_efficient_future_full_slice", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_missing_manifest_trajectory_uses_complete_marker(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    sample_id = "sample-001"
    relative_paths = {
        "video_path": "data/efficient_wam/clean/task/videos/sample-001.mp4",
        "qpos_path": "data/efficient_wam/clean/task/qpos/sample-001.pt",
        "trajectory_path": "data/efficient_wam/clean/task/trajectories/sample-001.pt",
        "label_path": "data/efficient_wam/clean/task/safety_labels/sample-001.json",
        "language_path": "data/efficient_wam/clean/task/umt5_wan/sample-001.pt",
        "meta_path": "data/efficient_wam/clean/task/metas/sample-001.txt",
    }
    manifest_exports = dict(relative_paths)
    manifest_exports.pop("trajectory_path")
    row = {
        "sample_id": sample_id,
        "quality_accepted": True,
        "dataset_partition": "train",
        "task": "task",
        "setting": "clean",
        "severity": "danger",
        "collection_source": "counterfactual",
        "scene_group_id": "scene-001",
        "efficient_wam": manifest_exports,
    }
    manifest = root / "manifests" / "samples.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(row) + "\n")

    labels = [0] * 57
    labels[56] = 1
    for field, relative in relative_paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if field == "label_path":
            path.write_text(json.dumps({"temporal_safety": {"frame_labels": labels}}))
        else:
            path.write_bytes(field.encode())

    complete = root / "data" / "samples" / sample_id / "complete.json"
    complete.parent.mkdir(parents=True)
    complete.write_text(json.dumps({"sample_id": sample_id, "efficient_wam": relative_paths}))

    output = tmp_path / "slice"
    result = MODULE.build(
        argparse.Namespace(
            source_root=root,
            fallback_root=None,
            output=output,
            tail_rows=100,
            train_episodes=1,
            eval_episodes=0,
            action_count=16,
            action_stride=3,
            condition_indices=(7, 8),
            seed=7,
        )
    )

    assert result["episodes"] == 1
    assert result["windows"] == 2
    episode = json.loads((output / "episodes.jsonl").read_text().strip())
    resolution = episode["path_resolution"]["trajectory_path"]
    assert resolution["metadata_source"] == "complete_marker"
    assert resolution["file_source"] == "source_root"
    assert resolution["relative_path"] == relative_paths["trajectory_path"]
