#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_PATH_FIELDS = (
    "image_path",
    "state_raw_path",
    "action_raw_path",
    "text_embedding_path",
)
SPLIT_MAP = {"train": "train", "val": "eval", "test": "test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _risk_steps(row: dict[str, Any]) -> list[int]:
    values = row.get("risk_steps")
    if not isinstance(values, list) or len(values) != 16:
        raise ValueError(f"{row.get('sample_id')} must contain 16 risk_steps")
    result = [int(str(value).lower() == "risk") for value in values]
    expected = int(str(row.get("risk")).lower() == "risk")
    if int(any(result)) != expected:
        raise ValueError(
            f"{row.get('sample_id')} chunk label disagrees with risk_steps"
        )
    return result


def _validated_groups(
    rows: list[dict[str, Any]], source_root: Path
) -> dict[str, list[list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split"))
        if split not in SPLIT_MAP:
            raise ValueError(f"Unknown split {split!r}")
        for field in REQUIRED_PATH_FIELDS:
            path = source_root / str(row.get(field, ""))
            if not path.is_file():
                raise FileNotFoundError(f"{field} missing for {row.get('sample_id')}: {path}")
        _risk_steps(row)
        grouped[str(row["scene_group_id"])].append(row)

    by_split: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group_id, group in grouped.items():
        if len(group) != 2 or {str(row["risk"]) for row in group} != {"safe", "risk"}:
            raise ValueError(f"{group_id} is not one safe/risk pair")
        invariant_fields = (
            "split",
            "task_id",
            "scene_seed",
            "image_path",
            "state_raw_path",
            "snapshot_path",
        )
        for field in invariant_fields:
            if len({json.dumps(row.get(field), sort_keys=True) for row in group}) != 1:
                raise ValueError(f"{group_id} does not share {field}")
        group.sort(key=lambda row: str(row["risk"]))
        by_split[str(group[0]["split"])].append(group)
    return by_split


def _select_groups(
    groups: list[list[dict[str, Any]]], *, limit: int, seed: int
) -> list[list[dict[str, Any]]]:
    if limit <= 0 or limit >= len(groups):
        return groups
    selected = list(groups)
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve()
    dataset_path = source_root / "dataset.json"
    audit_path = source_root / "reports" / "full-audit.json"
    source_manifest = source_root / "manifests" / "all.jsonl"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if dataset.get("status") != "complete" or audit.get("status") != "PASS":
        raise RuntimeError("ManiSkill source dataset is not complete and audited")
    rows = _read_jsonl(source_manifest)
    groups = _validated_groups(rows, source_root)
    limits = {
        "train": int(args.max_train_groups),
        "val": int(args.max_val_groups),
        "test": int(args.max_test_groups),
    }

    output = args.output.expanduser().resolve()
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    if output.exists() or temporary.exists():
        raise FileExistsError(output if output.exists() else temporary)
    temporary.mkdir(parents=True)
    window_rows: list[dict[str, Any]] = []
    selected_group_counts: dict[str, int] = {}
    try:
        for split_index, source_split in enumerate(("train", "val", "test")):
            selected = _select_groups(
                groups.get(source_split, []),
                limit=limits[source_split],
                seed=int(args.seed) + split_index,
            )
            selected_group_counts[source_split] = len(selected)
            for group in selected:
                for row in group:
                    step_targets = _risk_steps(row)
                    window_rows.append(
                        {
                            "input_schema": "maniskill_aloha_pair_v1",
                            "window_id": str(row["sample_id"]),
                            "sample_id": str(row["sample_id"]),
                            "scene_group_id": str(row["scene_group_id"]),
                            "split": SPLIT_MAP[source_split],
                            "source_split": source_split,
                            "task": str(row["task_id"]),
                            "setting": str(row.get("scene_variant", "randomized")),
                            "risk": str(row["risk"]),
                            "chunk_target": int(any(step_targets)),
                            "step_targets": step_targets,
                            "action_dt": float(row["action_dt"]),
                            "path_root": str(source_root),
                            "paths": {
                                "image_path": str(row["image_path"]),
                                "state_path": str(row["state_raw_path"]),
                                "action_path": str(row["action_raw_path"]),
                                "language_path": str(row["text_embedding_path"]),
                            },
                            "source": {
                                "episode_id": row.get("episode_id"),
                                "family": row.get("family"),
                                "object_superclass": row.get("object_superclass"),
                                "scene_seed": row.get("scene_seed"),
                                "snapshot_sha256": row.get("snapshot_sha256"),
                            },
                        }
                    )
        window_rows.sort(key=lambda row: str(row["window_id"]))
        _write_jsonl(temporary / "windows.jsonl", window_rows)
        manifest = {
            "schema_version": 1,
            "input_schema": "maniskill_aloha_pair_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256(source_manifest),
            "source_dataset_sha256": _sha256(dataset_path),
            "source_audit_sha256": _sha256(audit_path),
            "source_audit_status": audit["status"],
            "selection": {
                "seed": int(args.seed),
                "max_train_groups": limits["train"],
                "max_val_groups": limits["val"],
                "max_test_groups": limits["test"],
            },
            "groups": sum(selected_group_counts.values()),
            "group_source_split_counts": selected_group_counts,
            "windows": len(window_rows),
            "window_split_counts": dict(Counter(row["split"] for row in window_rows)),
            "window_label_counts": dict(Counter(row["risk"] for row in window_rows)),
            "task_counts": dict(Counter(row["task"] for row in window_rows)),
            "family_counts": dict(
                Counter(str(row["source"]["family"]) for row in window_rows)
            ),
        }
        _write_json(temporary / "MANIFEST.json", manifest)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if temporary.exists():
            temporary.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-groups", type=int, default=0)
    parser.add_argument("--max-val-groups", type=int, default=0)
    parser.add_argument("--max-test-groups", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
