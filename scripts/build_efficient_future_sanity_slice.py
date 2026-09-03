#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_EXPORTS = (
    "video_path",
    "qpos_path",
    "trajectory_path",
    "label_path",
    "language_path",
    "meta_path",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _accepted(row: dict[str, Any]) -> bool:
    assessment = row.get("quality_assessment")
    if isinstance(assessment, dict) and assessment.get("accepted") is False:
        return False
    return row.get("quality_accepted", True) is not False


def _paired_conditions(
    frame_labels: list[int], *, action_count: int, stride: int
) -> tuple[int, int, list[int], list[int]] | None:
    horizon = action_count * stride
    candidates: list[tuple[int, list[int], list[int]]] = []
    for condition in range(max(0, len(frame_labels) - horizon)):
        action_indices = [condition + (index + 1) * stride for index in range(action_count)]
        labels = [int(frame_labels[index]) for index in action_indices]
        candidates.append((condition, action_indices, labels))
    risky = [item for item in candidates if frame_labels[item[0]] == 0 and max(item[2]) > 0]
    safe = [item for item in candidates if frame_labels[item[0]] == 0 and max(item[2]) == 0]
    if not risky or not safe:
        return None
    # The latest still-safe condition before an event gives the future branch a
    # meaningful prediction problem.  Match the safe control in time as closely
    # as the episode permits to reduce phase-of-task shortcuts.
    risk = max(risky, key=lambda item: item[0])
    control = min(safe, key=lambda item: (abs(item[0] - risk[0]), item[0]))
    return control[0], risk[0], control[1], risk[1]


def _select_balanced(
    rows: list[dict[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda row: _stable_key(seed, str(row["sample_id"])))
    tasks = sorted(by_task)
    selected: list[dict[str, Any]] = []
    quota = math.ceil(limit / max(1, len(tasks)))
    for task in tasks:
        selected.extend(by_task[task][:quota])
    selected.sort(key=lambda row: _stable_key(seed + 1, str(row["sample_id"])))
    if len(selected) < limit:
        chosen = {str(row["sample_id"]) for row in selected}
        remainder = [row for row in rows if str(row["sample_id"]) not in chosen]
        remainder.sort(key=lambda row: _stable_key(seed + 2, str(row["sample_id"])))
        selected.extend(remainder[: limit - len(selected)])
    return selected[:limit]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve()
    source_manifest = source_root / "manifests" / "samples.jsonl"
    all_lines = [line for line in source_manifest.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in all_lines[-args.tail_rows :]]
    eligible: list[dict[str, Any]] = []
    for row in rows:
        exports = row.get("efficient_wam")
        if not _accepted(row) or not isinstance(exports, dict):
            continue
        if row.get("dataset_partition") not in {"train", "calibration", "challenge"}:
            continue
        source_paths = {name: source_root / str(exports.get(name, "")) for name in REQUIRED_EXPORTS}
        if not all(path.is_file() for path in source_paths.values()):
            continue
        sidecar = json.loads(source_paths["label_path"].read_text(encoding="utf-8"))
        temporal = sidecar.get("temporal_safety", {})
        labels = [int(value) for value in temporal.get("frame_labels", [])]
        if len(labels) < args.action_count * args.action_stride + 1:
            continue
        paired = _paired_conditions(
            labels, action_count=args.action_count, stride=args.action_stride
        )
        if paired is None or any(
            condition < 0
            or condition + args.action_count * args.action_stride >= len(labels)
            for condition in args.condition_indices
        ):
            continue
        enriched = dict(row)
        enriched["_source_paths"] = {key: str(value) for key, value in source_paths.items()}
        enriched["_paired"] = paired
        enriched["_frame_labels"] = labels
        eligible.append(enriched)

    train_candidates = [row for row in eligible if row["dataset_partition"] == "train"]
    eval_candidates = [row for row in eligible if row["dataset_partition"] != "train"]
    train = _select_balanced(train_candidates, limit=args.train_episodes, seed=args.seed)
    evaluation = _select_balanced(eval_candidates, limit=args.eval_episodes, seed=args.seed + 10)
    if len(train) != args.train_episodes or len(evaluation) != args.eval_episodes:
        raise RuntimeError(
            f"Not enough paired local episodes: train={len(train)}/{args.train_episodes}, "
            f"eval={len(evaluation)}/{args.eval_episodes}"
        )

    output = args.output.expanduser().resolve()
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    if output.exists() or temporary.exists():
        raise FileExistsError(output if output.exists() else temporary)
    temporary.mkdir(parents=True)
    episode_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    try:
        for split, selected in (("train", train), ("eval", evaluation)):
            for row in selected:
                sample_id = str(row["sample_id"])
                sample_dir = temporary / "episodes" / sample_id
                sample_dir.mkdir(parents=True)
                copied: dict[str, str] = {}
                hashes: dict[str, str] = {}
                for field, raw_source in row.pop("_source_paths").items():
                    source = Path(raw_source)
                    suffix = "".join(source.suffixes)
                    destination = sample_dir / f"{field.removesuffix('_path')}{suffix}"
                    shutil.copy2(source, destination)
                    copied[field] = str(destination.relative_to(temporary))
                    hashes[field] = _sha256(destination)
                row.pop("_paired")
                frame_labels = row.pop("_frame_labels")
                source_record = dict(row)
                episode_record = {
                    "sample_id": sample_id,
                    "split": split,
                    "source_partition": row["dataset_partition"],
                    "task": row["task"],
                    "setting": row["setting"],
                    "episode_severity": row["severity"],
                    "collection_source": row["collection_source"],
                    "scene_group_id": row["scene_group_id"],
                    "paths": copied,
                    "sha256": hashes,
                    "source_record": source_record,
                }
                episode_rows.append(episode_record)
                for condition in args.condition_indices:
                    action_indices = [
                        condition + (index + 1) * args.action_stride
                        for index in range(args.action_count)
                    ]
                    step_targets = [
                        int(frame_labels[index] > 0) for index in action_indices
                    ]
                    label = "risk" if max(step_targets) else "safe"
                    window_rows.append(
                        {
                            "window_id": f"{sample_id}__{label}__c{condition:03d}",
                            "sample_id": sample_id,
                            "split": split,
                            "task": row["task"],
                            "setting": row["setting"],
                            "risk": label,
                            "chunk_target": int(label == "risk"),
                            "condition_frame_idx": condition,
                            "action_indices": action_indices,
                            "step_targets": step_targets,
                            "paths": copied,
                        }
                    )
        _write_jsonl(temporary / "episodes.jsonl", episode_rows)
        _write_jsonl(temporary / "windows.jsonl", window_rows)
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256(source_manifest),
            "selection": {
                "tail_rows": args.tail_rows,
                "seed": args.seed,
                "train_episodes": args.train_episodes,
                "eval_episodes": args.eval_episodes,
                "action_count": args.action_count,
                "action_stride": args.action_stride,
                "condition_indices": list(args.condition_indices),
            },
            "episodes": len(episode_rows),
            "windows": len(window_rows),
            "episode_split_counts": dict(Counter(row["split"] for row in episode_rows)),
            "window_label_counts": dict(Counter(row["risk"] for row in window_rows)),
            "task_counts": dict(Counter(row["task"] for row in episode_rows)),
            "source_commits": sorted(
                {
                    json.dumps(row["source_record"].get("source_commits", {}), sort_keys=True)
                    for row in episode_rows
                }
            ),
        }
        _write_json(temporary / "MANIFEST.json", manifest)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-rows", type=int, default=256)
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--action-count", type=int, default=16)
    parser.add_argument("--action-stride", type=int, default=3)
    parser.add_argument(
        "--condition-indices",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        default=(7, 8),
    )
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
