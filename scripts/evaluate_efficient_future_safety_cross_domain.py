#!/usr/bin/env python3
"""Evaluate the trained Efficient-future safety adapters on every compatible domain.

The script keeps deployment thresholds fixed to the thresholds selected while
training the ManiSkill adapters.  RoboTwin uses cached Efficient-WAM future
tokens from an episode-disjoint evaluation split.  UR5 is evaluated through the
explicit no-future path because there is no compatible Efficient-WAM future
cache for the 7-D velocity-control profile.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_efficient_future_safety_controls import (  # noqa: E402
    _complete_metrics,
    _load_adapter,
)
from scripts.train_efficient_future_full_data import (  # noqa: E402
    _batch,
    _read_jsonl,
    _sha256,
    _torch_load,
)
from safety_verify_wam.portable import load_multidomain_checkpoint  # noqa: E402
from safety_verify_wam.portable.multidomain_training import (  # noqa: E402
    dataset_from_config,
    make_loader,
    to_safety_batch,
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *command],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _classification_summary(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    targets = targets.astype(np.int64)
    predictions = predictions.astype(np.int64)
    tn = int(((targets == 0) & (predictions == 0)).sum())
    fp = int(((targets == 0) & (predictions == 1)).sum())
    fn = int(((targets == 1) & (predictions == 0)).sum())
    tp = int(((targets == 1) & (predictions == 1)).sum())
    return {
        "samples": int(len(targets)),
        "correct": tn + tp,
        "accuracy": float((tn + tp) / max(1, len(targets))),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "risk_recall": float(tp / max(1, tp + fn)),
        "safe_recall": float(tn / max(1, tn + fp)),
        "balanced_accuracy": float(
            (tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2
        ),
    }


def _task_metrics(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task") or row.get("simulator_key") or "unknown")].append(
            row
        )
    result: dict[str, Any] = {}
    for key, values in sorted(grouped.items()):
        target = np.asarray([value["target"] for value in values], dtype=np.int64)
        score = np.asarray([value[score_key] for value in values], dtype=np.float64)
        result[key] = _complete_metrics(target, score, threshold=threshold)
    return result


def _seed_aggregate(seed_rows: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    seeds = sorted(seed_rows)
    if not seeds:
        raise ValueError("No seed predictions were supplied")
    reference_ids = [row["sample_id"] for row in seed_rows[seeds[0]]]
    targets = np.asarray(
        [row["target"] for row in seed_rows[seeds[0]]], dtype=np.int64
    )
    votes: list[np.ndarray] = []
    repeated_targets: list[np.ndarray] = []
    repeated_predictions: list[np.ndarray] = []
    for seed in seeds:
        rows = seed_rows[seed]
        if [row["sample_id"] for row in rows] != reference_ids:
            raise RuntimeError(f"Sample order changed for seed {seed}")
        current_targets = np.asarray([row["target"] for row in rows], dtype=np.int64)
        if not np.array_equal(current_targets, targets):
            raise RuntimeError(f"Targets changed for seed {seed}")
        current_predictions = np.asarray(
            [row["prediction"] for row in rows], dtype=np.int64
        )
        votes.append(current_predictions)
        repeated_targets.append(current_targets)
        repeated_predictions.append(current_predictions)
    vote_array = np.stack(votes)
    majority = (vote_array.sum(axis=0) >= (len(seeds) // 2 + 1)).astype(np.int64)
    return {
        "seed_count": len(seeds),
        "repeated_decisions": _classification_summary(
            np.concatenate(repeated_targets), np.concatenate(repeated_predictions)
        ),
        "majority_vote": _classification_summary(targets, majority),
        "seed_prediction_disagreement_rate": float(
            np.mean(vote_array.min(axis=0) != vote_array.max(axis=0))
        ),
    }


def _maniskill_results(config: dict[str, Any], output: Path) -> dict[str, Any]:
    source = Path(config["source_artifact"]).resolve()
    expected = int(config["expected"]["maniskill_test_records"])
    seeds = [int(value) for value in config["seeds"]]
    seed_rows: dict[int, list[dict[str, Any]]] = {}
    runs: dict[str, Any] = {}
    for seed in seeds:
        run_dir = source / f"full-seed{seed}"
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        threshold = float(result["selected_threshold"])
        source_rows = _read_jsonl(run_dir / "test_predictions.jsonl")
        if len(source_rows) != expected:
            raise RuntimeError(
                f"Expected {expected} ManiSkill records for seed {seed}, "
                f"found {len(source_rows)}"
            )
        rows = [
            {
                "sample_id": str(row["window_id"]),
                "scene_group_id": str(row.get("scene_group_id", row["window_id"])),
                "task": str(row["task"]),
                "target": int(row["target"]),
                "score": float(row["score"]),
                "threshold": threshold,
                "prediction": int(float(row["score"]) >= threshold),
            }
            for row in source_rows
        ]
        target = np.asarray([row["target"] for row in rows], dtype=np.int64)
        score = np.asarray([row["score"] for row in rows], dtype=np.float64)
        computed = _complete_metrics(target, score, threshold=threshold)
        reported = result["test"]
        if computed["confusion_matrix"] != reported["confusion_matrix"]:
            raise RuntimeError(f"ManiSkill source metrics disagree for seed {seed}")
        seed_rows[seed] = rows
        runs[str(seed)] = {
            "threshold": threshold,
            "metrics": computed,
            "per_task": _task_metrics(rows, score_key="score", threshold=threshold),
            "adapter": str((run_dir / "adapter.pt").resolve()),
            "adapter_sha256": _sha256(run_dir / "adapter.pt"),
            "raw_predictions": str((run_dir / "test_predictions.jsonl").resolve()),
            "raw_predictions_sha256": _sha256(run_dir / "test_predictions.jsonl"),
        }
    result = {
        "status": "EVALUATED",
        "evaluation_kind": "offline_fixed_candidate_window_classification",
        "split": "test",
        "runs": runs,
        "aggregate": _seed_aggregate(seed_rows),
        "controls_artifact": str(Path(config["maniskill_controls_artifact"]).resolve()),
    }
    _atomic_json(output / "maniskill.json", result)
    return result


def _robotwin_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["robotwin_feature_cache"]).resolve()
    index = _read_jsonl(root / "features.jsonl")
    selected = [row for row in index if str(row["split"]) == "eval"]
    expected = int(config["expected"]["robotwin_eval_records"])
    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} RoboTwin eval records, found {len(selected)}"
        )
    records = [_torch_load(root / row["record_path"]) for row in selected]
    if any(str(row["window"]["split"]) != "eval" for row in records):
        raise RuntimeError("RoboTwin cache contains a non-eval record")
    return records


@torch.inference_mode()
def _predict_robotwin(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    mode: str,
    batch_size: int,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    model.eval()
    selected = records
    if mode == "shuffled":
        rng = np.random.default_rng(shuffle_seed)
        order = np.roll(rng.permutation(len(records)), 1)
        selected = []
        for index, row in enumerate(records):
            transformed = dict(row)
            transformed["future_tokens"] = records[int(order[index])]["future_tokens"]
            selected.append(transformed)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(selected), batch_size):
        current = selected[start : start + batch_size]
        safety, future, target, _ = _batch(current, device)
        output = model(
            "bimanual_qpos14",
            safety,
            None if mode == "none" else future,
            future_mode="none" if mode == "none" else "full",
        )
        chunk = output["risk_probability"].float()
        maximum_step = output["step_risk_probability"].float().masked_fill(
            ~safety.action_mask, -1.0
        ).max(dim=1).values
        deployment = torch.maximum(chunk, maximum_step)
        for offset, record in enumerate(current):
            window = record["window"]
            rows.append(
                {
                    "sample_id": str(window["window_id"]),
                    "scene_group_id": str(window.get("scene_group_id", window["sample_id"])),
                    "task": str(window["task"]),
                    "setting": str(window.get("setting", "unknown")),
                    "target": int(target[offset].item()),
                    "chunk_score": float(chunk[offset].item()),
                    "deployment_score": float(deployment[offset].item()),
                }
            )
    return rows


def _robotwin_results(
    config: dict[str, Any],
    output: Path,
    base: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    records = _robotwin_records(config)
    source = Path(config["source_artifact"]).resolve()
    seeds = [int(value) for value in config["seeds"]]
    batch_size = int(config["batch_size"])
    modes = ("full", "none", "shuffled")
    policies = {"chunk": "chunk_score", "chunk_or_step": "deployment_score"}
    runs: dict[str, Any] = {}
    aggregate_inputs: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = {
        (mode, policy): {} for mode in modes for policy in policies
    }
    for seed in seeds:
        source_dir = source / f"full-seed{seed}"
        source_result = json.loads(
            (source_dir / "result.json").read_text(encoding="utf-8")
        )
        threshold = float(source_result["selected_threshold"])
        model = _load_adapter(
            base=base, checkpoint=source_dir / "adapter.pt", device=device
        )
        seed_result: dict[str, Any] = {
            "threshold": threshold,
            "threshold_source": "maniskill_training_split",
            "modes": {},
        }
        for mode in modes:
            raw_rows = _predict_robotwin(
                model,
                records,
                device,
                mode=mode,
                batch_size=batch_size,
                shuffle_seed=seed + 1000,
            )
            mode_result: dict[str, Any] = {}
            for policy, score_key in policies.items():
                policy_rows = []
                for row in raw_rows:
                    value = dict(row)
                    value["score"] = float(value[score_key])
                    value["threshold"] = threshold
                    value["prediction"] = int(value["score"] >= threshold)
                    value["mode"] = mode
                    value["score_policy"] = policy
                    policy_rows.append(value)
                target = np.asarray(
                    [row["target"] for row in policy_rows], dtype=np.int64
                )
                score = np.asarray(
                    [row["score"] for row in policy_rows], dtype=np.float64
                )
                raw_path = output / "robotwin" / f"seed{seed}-{mode}-{policy}.jsonl"
                _atomic_jsonl(raw_path, policy_rows)
                mode_result[policy] = {
                    "metrics": _complete_metrics(target, score, threshold=threshold),
                    "per_task": _task_metrics(
                        policy_rows, score_key="score", threshold=threshold
                    ),
                    "raw_predictions": str(raw_path.resolve()),
                    "raw_predictions_sha256": _sha256(raw_path),
                }
                aggregate_inputs[(mode, policy)][seed] = policy_rows
            seed_result["modes"][mode] = mode_result
        runs[str(seed)] = seed_result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"phase": "robotwin", "seed": seed}), flush=True)
    aggregates = {
        mode: {
            policy: _seed_aggregate(aggregate_inputs[(mode, policy)])
            for policy in policies
        }
        for mode in modes
    }
    result = {
        "status": "EVALUATED",
        "evaluation_kind": "offline_fixed_candidate_window_classification",
        "split": "eval_episode_disjoint",
        "future_source": "cached_efficient_wam_video_value_tokens",
        "threshold_transfer": "fixed_from_maniskill_training_no_robotwin_tuning",
        "records": len(records),
        "runs": runs,
        "aggregate": aggregates,
    }
    _atomic_json(output / "robotwin.json", result)
    return result


@torch.inference_mode()
def _predict_ur5(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    dataset: Any,
    profile: Any,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    model.eval()
    metadata = {str(row["sample_id"]): row for row in dataset.rows}
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        safety = to_safety_batch(raw_batch, device)
        safety.validate(profile)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            output = model("ur5_speedl7", safety, future_mode="none")
        score = output["risk_probability"].float().cpu()
        for index, sample_id in enumerate(raw_batch["sample_id"]):
            source = metadata[str(sample_id)]
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "task": str(source.get("task") or "ur5_real"),
                    "simulator_key": str(raw_batch["simulator_key"][index]),
                    "episode_id": source.get("episode_id"),
                    "interaction_phase": source.get("interaction_phase"),
                    "target": int(raw_batch["chunk_target"][index].item()),
                    "score": float(score[index].item()),
                }
            )
    return rows


def _ur5_results(
    config: dict[str, Any],
    output: Path,
    base: torch.nn.Module,
    loaded: Any,
    device: torch.device,
) -> dict[str, Any]:
    training_config = json.loads(
        Path(config["portable_training_config"]).read_text(encoding="utf-8")
    )
    dataset_config = dict(training_config["datasets"]["ur5"])
    dataset_config["num_workers"] = 0
    dataset = dataset_from_config(dataset_config, "val")
    expected = int(config["expected"]["ur5_val_records"])
    if len(dataset) != expected:
        raise RuntimeError(f"Expected {expected} UR5 records, found {len(dataset)}")
    profile = loaded.profiles["ur5_speedl7"]
    threshold = float(loaded.profile_thresholds["ur5_speedl7"].chunk_risk)
    amp_name = str(training_config["training"].get("amp", "float32")).lower()
    amp_enabled = device.type == "cuda" and amp_name not in {"none", "float32"}
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    source = Path(config["source_artifact"]).resolve()
    seeds = [int(value) for value in config["seeds"]]
    runs: dict[str, Any] = {}
    seed_rows: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        loader = make_loader(
            dataset, dataset_config, training=False, device=device, seed=seed
        )
        model = _load_adapter(
            base=base,
            checkpoint=source / f"full-seed{seed}" / "adapter.pt",
            device=device,
        )
        rows = _predict_ur5(
            model,
            loader,
            dataset,
            profile,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        for row in rows:
            row["threshold"] = threshold
            row["prediction"] = int(row["score"] >= threshold)
            row["future_mode"] = "none"
        target = np.asarray([row["target"] for row in rows], dtype=np.int64)
        score = np.asarray([row["score"] for row in rows], dtype=np.float64)
        raw_path = output / "ur5" / f"seed{seed}-none.jsonl"
        _atomic_jsonl(raw_path, rows)
        seed_rows[seed] = rows
        runs[str(seed)] = {
            "threshold": threshold,
            "metrics": _complete_metrics(target, score, threshold=threshold),
            "per_task": _task_metrics(rows, score_key="score", threshold=threshold),
            "raw_predictions": str(raw_path.resolve()),
            "raw_predictions_sha256": _sha256(raw_path),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"phase": "ur5", "seed": seed}), flush=True)

    reference = _read_jsonl(Path(config["portable_reference_predictions"]))
    reference_scores = {
        str(row["sample_id"]): float(row["deployment_risk_probability"])
        for row in reference
        if str(row["profile_key"]) == "ur5_speedl7"
    }
    deltas = [
        abs(float(row["score"]) - reference_scores[str(row["sample_id"])])
        for row in seed_rows[seeds[0]]
    ]
    seed_score_delta = 0.0
    first = {row["sample_id"]: row["score"] for row in seed_rows[seeds[0]]}
    for seed in seeds[1:]:
        seed_score_delta = max(
            seed_score_delta,
            max(
                abs(float(row["score"]) - float(first[row["sample_id"]]))
                for row in seed_rows[seed]
            ),
        )
    result = {
        "status": "EVALUATED_NO_FUTURE_PASSTHROUGH",
        "evaluation_kind": "offline_fixed_candidate_window_classification",
        "split": "validation_two_episodes_single_scene_group",
        "future_source": None,
        "inference_precision": amp_name,
        "future_limitation": (
            "The current Efficient-WAM future cache is bimanual qpos14; it is "
            "not compatible with the UR5 speedl7 profile. This run verifies the "
            "trained adapter's explicit no-future path, which equals the frozen base."
        ),
        "reference_recompute_max_abs_score_delta": float(max(deltas, default=0.0)),
        "cross_seed_max_abs_score_delta": float(seed_score_delta),
        "runs": runs,
        "aggregate": _seed_aggregate(seed_rows),
    }
    _atomic_json(output / "ur5.json", result)
    return result


def _libero_result(config: dict[str, Any], loaded: Any, output: Path) -> dict[str, Any]:
    libero = config["libero"]
    manifest_root = libero.get("manifest_root")
    required_profile = libero.get("required_profile")
    reasons: list[str] = []
    if not manifest_root:
        reasons.append("no_project_owned_safety_manifest")
    elif not Path(manifest_root).exists():
        reasons.append("configured_manifest_does_not_exist")
    if not required_profile:
        reasons.append("no_validated_libero_robot_profile")
    elif required_profile not in loaded.profiles:
        reasons.append("checkpoint_does_not_contain_required_profile")
    reasons.append("no_compatible_efficient_future_feature_cache")
    result = {
        "status": "NOT_EVALUATED_INCOMPATIBLE_INPUT_CONTRACT",
        "accuracy": None,
        "supported_checkpoint_profiles": sorted(loaded.profiles),
        "reasons": reasons,
        "note": libero.get("note"),
        "claim_boundary": "No LIBERO detection or task-success number is claimed.",
    }
    _atomic_json(output / "libero_compatibility.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "config.resolved.json", config)

    device = torch.device(str(config["device"]))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            float(config["cuda_memory_fraction"]), device=device
        )
    portable_checkpoint = Path(config["portable_checkpoint"]).resolve()
    loaded = load_multidomain_checkpoint(portable_checkpoint, map_location="cpu")
    base = loaded.model.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    started_at = _now()
    maniskill = _maniskill_results(config, output)
    robotwin = _robotwin_results(config, output, base, device)
    ur5 = _ur5_results(config, output, base, loaded, device)
    libero = _libero_result(config, loaded, output)
    summary = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "started_at": started_at,
        "completed_at": _now(),
        "status": "COMPLETE_WITH_LIBERO_INCOMPATIBLE",
        "evaluation_scope": (
            "Offline classification of fixed candidate action windows. These are "
            "detection accuracies, not closed-loop robot task success rates."
        ),
        "git": {
            "branch": _git(["branch", "--show-current"]),
            "commit": _git(["rev-parse", "HEAD"]),
            "status": _git(["status", "--porcelain=v1"]),
        },
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "portable_checkpoint": str(portable_checkpoint),
        "portable_checkpoint_sha256": _sha256(portable_checkpoint),
        "domains": {
            "maniskill": maniskill,
            "robotwin": robotwin,
            "ur5_real": ur5,
            "libero": libero,
        },
    }
    _atomic_json(output / "SUMMARY.json", summary)
    print(json.dumps({"phase": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
