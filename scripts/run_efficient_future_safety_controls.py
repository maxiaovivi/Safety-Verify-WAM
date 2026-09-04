#!/usr/bin/env python3
"""Run equal-capacity and paired counterfactual controls for the future safety head."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.train_efficient_future_full_data import (  # noqa: E402
    _balanced_indices,
    _batch,
    _metrics,
    _read_jsonl,
    _seed,
    _select_threshold,
    _sha256,
    _torch_load,
)
from safety_verify_wam.portable import (  # noqa: E402
    EfficientFutureSafetyConfig,
    EfficientFutureSafetySidecar,
    load_multidomain_checkpoint,
)


VARIANTS = (
    "true_future",
    "paired_future_swap",
    "constant_future",
    "zero_future",
    "paired_action_swap",
    "paired_action_and_future_swap",
)


def _git(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", *command], cwd=cwd, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _complete_metrics(
    target: np.ndarray, score: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    result = _metrics(target, score, threshold=threshold)
    (tn, fp), (fn, tp) = result["confusion_matrix"]
    total = tn + fp + fn + tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result.update(
        {
            "correct": int(tn + tp),
            "accuracy": float((tn + tp) / max(1, total)),
            "risk_precision": float(precision),
            "f1": float(2 * precision * recall / max(1e-12, precision + recall)),
            "false_positive_rate": float(fp / max(1, fp + tn)),
            "false_negative_rate": float(fn / max(1, fn + tp)),
        }
    )
    return result


def _pair_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["window"]["scene_group_id"])].append(record)
    pair_by_window: dict[str, dict[str, Any]] = {}
    for group, rows in groups.items():
        labels = {int(row["window"]["chunk_target"]) for row in rows}
        if len(rows) != 2 or labels != {0, 1}:
            raise RuntimeError(
                f"Scene group {group!r} must contain one safe and one risk record"
            )
        left, right = rows
        pair_by_window[str(left["window"]["window_id"])] = right
        pair_by_window[str(right["window"]["window_id"])] = left
    return pair_by_window


def _constant_future(records: list[dict[str, Any]]) -> torch.Tensor:
    if not records:
        raise ValueError("Cannot build a constant future from an empty split")
    first = records[0]["future_tokens"].float()
    total = torch.zeros_like(first)
    for record in records:
        value = record["future_tokens"].float()
        if value.shape != first.shape:
            raise RuntimeError(
                f"Future shape changed from {tuple(first.shape)} to {tuple(value.shape)}"
            )
        total.add_(value)
    return total.div_(len(records))


def _tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _variant_records(
    records: list[dict[str, Any]],
    *,
    variant: str,
    pair_by_window: dict[str, dict[str, Any]],
    constant_future: torch.Tensor,
) -> list[dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    selected: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        window_id = str(record["window"]["window_id"])
        pair = pair_by_window[window_id]
        if variant in {"paired_action_swap", "paired_action_and_future_swap"}:
            row["action"] = pair["action"]
        if variant in {"paired_future_swap", "paired_action_and_future_swap"}:
            row["future_tokens"] = pair["future_tokens"]
        elif variant == "constant_future":
            row["future_tokens"] = constant_future
        elif variant == "zero_future":
            row["future_tokens"] = torch.zeros_like(constant_future)
        selected.append(row)
    return selected


@torch.inference_mode()
def _predict_variant(
    model: EfficientFutureSafetySidecar,
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    variant: str,
    pair_by_window: dict[str, dict[str, Any]],
    constant_future: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    transformed = _variant_records(
        records,
        variant=variant,
        pair_by_window=pair_by_window,
        constant_future=constant_future,
    )
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for start in range(0, len(transformed), batch_size):
        safety, future, target, _ = _batch(
            transformed[start : start + batch_size], device
        )
        output = model(
            "bimanual_qpos14", safety, future, future_mode="full"
        )
        targets.append(target.cpu().numpy())
        scores.append(output["risk_probability"].float().cpu().numpy())
    return np.concatenate(targets), np.concatenate(scores)


def _load_adapter(
    *,
    base: torch.nn.Module,
    checkpoint: Path,
    device: torch.device,
) -> EfficientFutureSafetySidecar:
    payload = _torch_load(checkpoint)
    config = EfficientFutureSafetyConfig(**payload["future_config"])
    model = EfficientFutureSafetySidecar(
        copy.deepcopy(base), config, freeze_base=True
    )
    incompatible = model.load_state_dict(payload["adapter_state"], strict=False)
    bad_missing = [
        key for key in incompatible.missing_keys if not key.startswith("base.")
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Adapter state mismatch: missing={bad_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval()


def _train_constant_control(
    *,
    base: torch.nn.Module,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    constant_future: torch.Tensor,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    output: Path,
) -> dict[str, Any]:
    _seed(seed)
    model = EfficientFutureSafetySidecar(
        copy.deepcopy(base),
        EfficientFutureSafetyConfig(future_dim=2048, attention_heads=4, dropout=0.0),
        freeze_base=True,
    ).to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1e-3
    )
    sampler = random.Random(seed)
    trace: list[dict[str, float]] = []
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, steps + 1):
        indices = _balanced_indices(train_records, batch_size, sampler)
        chosen = [dict(train_records[index]) for index in indices]
        for record in chosen:
            record["future_tokens"] = constant_future
        safety, future, chunk_target, step_target = _batch(chosen, device)
        tensors = model("bimanual_qpos14", safety, future, future_mode="full")
        chunk_loss = F.cross_entropy(tensors["class_logits"], chunk_target)
        step_loss = F.cross_entropy(
            tensors["step_class_logits"].reshape(-1, 2), step_target.reshape(-1)
        )
        loss = chunk_loss + 0.35 * step_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            progress = {
                "step": float(step),
                "loss": float(loss.detach()),
                "chunk_loss": float(chunk_loss.detach()),
                "step_loss": float(step_loss.detach()),
                "elapsed_seconds": float(time.monotonic() - started),
            }
            trace.append(progress)
            print(
                json.dumps(
                    {"phase": "constant_control_training", "seed": seed, **progress}
                ),
                flush=True,
            )

    split_records = {
        "train": train_records,
        "eval": eval_records,
        "test": test_records,
    }
    pairs = {name: _pair_map(rows) for name, rows in split_records.items()}
    targets: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for split, rows in split_records.items():
        targets[split], scores[split] = _predict_variant(
            model,
            rows,
            device,
            variant="constant_future",
            pair_by_window=pairs[split],
            constant_future=constant_future,
            batch_size=eval_batch_size,
        )
    threshold = _select_threshold(targets["train"], scores["train"])
    result: dict[str, Any] = {
        "mode": "trainable_constant_future",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "trainable_parameters": int(sum(value.numel() for value in trainable)),
        "selected_threshold": float(threshold),
        "trace": trace,
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_gpu_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    run_dir = output / f"constant-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_records.items():
        result[split] = _complete_metrics(
            targets[split], scores[split], threshold=threshold
        )
        prediction_rows = [
            {
                "window_id": row["window"]["window_id"],
                "scene_group_id": row["window"]["scene_group_id"],
                "task": row["window"]["task"],
                "target": int(target),
                "score": float(score),
            }
            for row, target, score in zip(rows, targets[split], scores[split])
        ]
        (run_dir / f"{split}_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows)
        )
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("base.")
    }
    torch.save(
        {
            "schema_version": 1,
            "mode": "trainable_constant_future",
            "seed": seed,
            "future_config": model.config.to_dict(),
            "adapter_state": state,
        },
        run_dir / "adapter.pt",
    )
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _probe_full_adapter(
    *,
    base: torch.nn.Module,
    source_artifact: Path,
    test_records: list[dict[str, Any]],
    constant_future: torch.Tensor,
    device: torch.device,
    seed: int,
    eval_batch_size: int,
    output: Path,
) -> dict[str, Any]:
    source_dir = source_artifact / f"full-seed{seed}"
    source_result = json.loads((source_dir / "result.json").read_text())
    threshold = float(source_result["selected_threshold"])
    model = _load_adapter(
        base=base, checkpoint=source_dir / "adapter.pt", device=device
    )
    pair_by_window = _pair_map(test_records)
    target: np.ndarray | None = None
    score_by_variant: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for variant in VARIANTS:
        current_target, current_score = _predict_variant(
            model,
            test_records,
            device,
            variant=variant,
            pair_by_window=pair_by_window,
            constant_future=constant_future,
            batch_size=eval_batch_size,
        )
        if target is None:
            target = current_target
        elif not np.array_equal(target, current_target):
            raise RuntimeError("Targets changed between counterfactual variants")
        score_by_variant[variant] = current_score
        metrics[variant] = _complete_metrics(
            current_target, current_score, threshold=threshold
        )
    assert target is not None

    source_rows = _read_jsonl(source_dir / "test_predictions.jsonl")
    source_scores = {
        str(row["window_id"]): float(row["score"]) for row in source_rows
    }
    recomputed = np.asarray(
        [source_scores[str(row["window"]["window_id"])] for row in test_records]
    )
    source_max_abs_delta = float(
        np.max(np.abs(recomputed - score_by_variant["true_future"]))
    )
    true_prediction = score_by_variant["true_future"] >= threshold
    deltas = {}
    for variant in VARIANTS[1:]:
        deltas[variant] = {
            "mean_abs_score_delta": float(
                np.mean(
                    np.abs(
                        score_by_variant["true_future"] - score_by_variant[variant]
                    )
                )
            ),
            "prediction_flip_rate": float(
                np.mean((score_by_variant[variant] >= threshold) != true_prediction)
            ),
            "ap_delta_from_true": float(
                metrics["true_future"]["average_precision"]
                - metrics[variant]["average_precision"]
            ),
        }
    result = {
        "seed": seed,
        "selected_threshold": threshold,
        "source_adapter": str((source_dir / "adapter.pt").resolve()),
        "source_adapter_sha256": _sha256(source_dir / "adapter.pt"),
        "source_result_sha256": _sha256(source_dir / "result.json"),
        "source_recompute_max_abs_score_delta": source_max_abs_delta,
        "trainable_parameters": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
        "variants": metrics,
        "deltas": deltas,
    }
    run_dir = output / f"full-probe-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for index, row in enumerate(test_records):
        predictions.append(
            {
                "window_id": row["window"]["window_id"],
                "scene_group_id": row["window"]["scene_group_id"],
                "task": row["window"]["task"],
                "target": int(target[index]),
                "scores": {
                    variant: float(score_by_variant[variant][index])
                    for variant in VARIANTS
                },
            }
        )
    (run_dir / "test_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate(runs: Iterable[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for run in runs:
        value: Any = run
        for key in path:
            value = value[key]
        selected.append(value)
    confusion = np.asarray(
        [value["confusion_matrix"] for value in selected], dtype=np.int64
    ).sum(axis=0)
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    total = int(confusion.sum())
    return {
        "seed_count": len(selected),
        "decisions": total,
        "correct": int(tn + tp),
        "accuracy": float((tn + tp) / max(1, total)),
        "confusion_matrix": confusion.tolist(),
        "risk_recall": float(tp / max(1, tp + fn)),
        "safe_recall": float(tn / max(1, tn + fp)),
        "average_precision_mean": float(
            np.mean([value["average_precision"] for value in selected])
        ),
        "average_precision_std": float(
            np.std([value["average_precision"] for value in selected])
        ),
        "accuracy_mean": float(np.mean([value["accuracy"] for value in selected])),
        "accuracy_std": float(np.std([value["accuracy"] for value in selected])),
    }


def _pair_integrity(records: list[dict[str, Any]]) -> dict[str, Any]:
    pair_by_window = _pair_map(records)
    visited: set[str] = set()
    video_equal = state_equal = action_equal = future_equal = 0
    action_rms: list[float] = []
    future_rms: list[float] = []
    groups = 0
    for record in records:
        window_id = str(record["window"]["window_id"])
        if window_id in visited:
            continue
        pair = pair_by_window[window_id]
        pair_id = str(pair["window"]["window_id"])
        visited.update((window_id, pair_id))
        groups += 1
        video_equal += int(torch.equal(record["video"], pair["video"]))
        state_equal += int(torch.equal(record["state"], pair["state"]))
        action_equal += int(torch.equal(record["action"], pair["action"]))
        future_equal += int(
            torch.equal(record["future_tokens"], pair["future_tokens"])
        )
        action_rms.append(
            float(
                (record["action"].float() - pair["action"].float())
                .square()
                .mean()
                .sqrt()
            )
        )
        future_rms.append(
            float(
                (
                    record["future_tokens"].float()
                    - pair["future_tokens"].float()
                )
                .square()
                .mean()
                .sqrt()
            )
        )
    return {
        "groups": groups,
        "video_equal_groups": video_equal,
        "state_equal_groups": state_equal,
        "action_equal_groups": action_equal,
        "future_equal_groups": future_equal,
        "action_pair_rms_mean": float(np.mean(action_rms)),
        "future_pair_rms_mean": float(np.mean(future_rms)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--portable-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--expected-train-records", type=int, default=1250)
    parser.add_argument("--expected-eval-records", type=int, default=250)
    parser.add_argument("--expected-test-records", type=int, default=300)
    parser.add_argument("--minimum-ap-delta", type=float, default=0.05)
    args = parser.parse_args()

    cache_root = args.feature_cache.expanduser().resolve()
    source_artifact = args.source_artifact.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_rows = _read_jsonl(cache_root / "features.jsonl")
    records = [_torch_load(cache_root / row["record_path"]) for row in index_rows]
    split_records = {
        split: [row for row in records if row["window"]["split"] == split]
        for split in ("train", "eval", "test")
    }
    expected = {
        "train": args.expected_train_records,
        "eval": args.expected_eval_records,
        "test": args.expected_test_records,
    }
    actual = {name: len(rows) for name, rows in split_records.items()}
    if actual != expected:
        raise RuntimeError(f"Expected split counts {expected}, got {actual}")

    constant_future = _constant_future(split_records["train"])
    constant_path = output / "train_mean_constant_future.pt"
    temporary_constant = output / f"train_mean_constant_future.pt.tmp.{os.getpid()}"
    torch.save(constant_future, temporary_constant)
    os.replace(temporary_constant, constant_path)
    device = torch.device(args.device)
    loaded = load_multidomain_checkpoint(
        args.portable_checkpoint.expanduser().resolve(), map_location="cpu"
    )
    base = loaded.model.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    full_probes = []
    for seed in seeds:
        result = _probe_full_adapter(
            base=base,
            source_artifact=source_artifact,
            test_records=split_records["test"],
            constant_future=constant_future,
            device=device,
            seed=seed,
            eval_batch_size=args.eval_batch_size,
            output=output,
        )
        full_probes.append(result)
        print(
            json.dumps(
                {
                    "phase": "full_adapter_probe",
                    "seed": seed,
                    "true_accuracy": result["variants"]["true_future"]["accuracy"],
                    "paired_future_accuracy": result["variants"]["paired_future_swap"]["accuracy"],
                    "constant_accuracy": result["variants"]["constant_future"]["accuracy"],
                }
            ),
            flush=True,
        )

    constant_runs = []
    for seed in seeds:
        result = _train_constant_control(
            base=base,
            train_records=split_records["train"],
            eval_records=split_records["eval"],
            test_records=split_records["test"],
            constant_future=constant_future,
            device=device,
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            output=output,
        )
        constant_runs.append(result)
        print(
            json.dumps(
                {
                    "phase": "constant_control_complete",
                    "seed": seed,
                    "test_accuracy": result["test"]["accuracy"],
                    "test_ap": result["test"]["average_precision"],
                }
            ),
            flush=True,
        )

    aggregates = {
        "full_true_future": _aggregate(full_probes, ("variants", "true_future")),
        "full_paired_future_swap": _aggregate(
            full_probes, ("variants", "paired_future_swap")
        ),
        "full_constant_future_at_test": _aggregate(
            full_probes, ("variants", "constant_future")
        ),
        "full_zero_future_at_test": _aggregate(
            full_probes, ("variants", "zero_future")
        ),
        "full_paired_action_swap": _aggregate(
            full_probes, ("variants", "paired_action_swap")
        ),
        "full_paired_action_and_future_swap": _aggregate(
            full_probes, ("variants", "paired_action_and_future_swap")
        ),
        "trainable_constant_future": _aggregate(constant_runs, ("test",)),
    }
    full_ap = aggregates["full_true_future"]["average_precision_mean"]
    pair_swap_ap = aggregates["full_paired_future_swap"]["average_precision_mean"]
    constant_ap = aggregates["trainable_constant_future"]["average_precision_mean"]
    pair_swap_delta = full_ap - pair_swap_ap
    constant_delta = full_ap - constant_ap
    supports_future_grounding = bool(
        pair_swap_delta >= args.minimum_ap_delta
        and constant_delta >= args.minimum_ap_delta
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Does the safety head need each sample's Efficient-WAM future, "
            "rather than equal-capacity query-side adaptation?"
        ),
        "scope": "held-out fixed candidate-action classification; not closed-loop safety",
        "split_records": actual,
        "seeds": seeds,
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.001,
            "constant_definition": (
                "elementwise mean [N,D] future-token bank over training records"
            ),
        },
        "constant_future": {
            "path": str(constant_path),
            "shape": list(constant_future.shape),
            "tensor_sha256": _tensor_sha256(constant_future),
            "file_sha256": _sha256(constant_path),
        },
        "pair_integrity": {
            name: _pair_integrity(rows) for name, rows in split_records.items()
        },
        "feature_cache": str(cache_root),
        "feature_cache_summary_sha256": _sha256(cache_root / "SUMMARY.json"),
        "source_artifact": str(source_artifact),
        "source_summary_sha256": _sha256(source_artifact / "SUMMARY.json"),
        "portable_checkpoint": str(args.portable_checkpoint.expanduser().resolve()),
        "portable_checkpoint_sha256": _sha256(
            args.portable_checkpoint.expanduser().resolve()
        ),
        "aggregates": aggregates,
        "future_grounding_check": {
            "minimum_ap_delta": args.minimum_ap_delta,
            "full_minus_paired_future_swap_ap": pair_swap_delta,
            "full_minus_trainable_constant_future_ap": constant_delta,
            "supports_sample_specific_future_grounding": supports_future_grounding,
        },
        "git": {
            "branch": _git(["branch", "--show-current"], Path.cwd()),
            "commit": _git(["rev-parse", "HEAD"], Path.cwd()),
            "status": _git(["status", "--porcelain=v1"], Path.cwd()),
        },
        "full_adapter_probes": full_probes,
        "constant_control_runs": constant_runs,
    }
    temporary = output / f"SUMMARY.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "SUMMARY.json")
    print(json.dumps(summary["future_grounding_check"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
