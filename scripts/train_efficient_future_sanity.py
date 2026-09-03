#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from safety_verify_wam.portable import (
    EfficientFutureSafetyConfig,
    EfficientFutureSafetySidecar,
    SafetyBatch,
    load_multidomain_checkpoint,
)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", *command], cwd=cwd, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch(records: list[dict[str, Any]], device: torch.device) -> tuple[SafetyBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    video = torch.stack([record["video"] for record in records]).unsqueeze(1)
    state = torch.cat([record["state"] for record in records], dim=0)
    action = torch.cat([record["action"] for record in records], dim=0)
    future = torch.stack([record["future_tokens"] for record in records])
    chunk = torch.tensor(
        [record["window"]["chunk_target"] for record in records], dtype=torch.long
    )
    steps = torch.tensor(
        [record["window"]["step_targets"] for record in records], dtype=torch.long
    )
    batch_size = len(records)
    dt = torch.tensor([record["action_dt"] for record in records], dtype=torch.float32)
    action_times = torch.arange(1, 17).float().unsqueeze(0) * dt.unsqueeze(1)
    safety = SafetyBatch(
        video=video.to(device=device, dtype=torch.float32).div_(255.0),
        state=state.to(device=device, dtype=torch.float32),
        action=action.to(device=device, dtype=torch.float32),
        video_timestamps=torch.zeros((batch_size, 1), device=device),
        state_timestamps=torch.zeros((batch_size, 1), device=device),
        action_timestamps=action_times.to(device),
        video_mask=torch.ones((batch_size, 1, 3), dtype=torch.bool, device=device),
        state_mask=torch.ones((batch_size, 1), dtype=torch.bool, device=device),
        action_mask=torch.ones((batch_size, 16), dtype=torch.bool, device=device),
    )
    return safety, future.to(device=device, dtype=torch.float32), chunk.to(device), steps.to(device)


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    ranked = target[order]
    positives = int(ranked.sum())
    if positives == 0:
        return 0.0
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def _roc_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = score[target == 1]
    negative = score[target == 0]
    if not len(positive) or not len(negative):
        return 0.0
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def _threshold_metrics(target: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = score >= threshold
    positive = target == 1
    negative = ~positive
    tp = int((prediction & positive).sum())
    tn = int((~prediction & negative).sum())
    fp = int((prediction & negative).sum())
    fn = int((~prediction & positive).sum())
    risk_recall = tp / max(1, tp + fn)
    safe_recall = tn / max(1, tn + fp)
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float((risk_recall + safe_recall) / 2),
        "risk_recall": float(risk_recall),
        "safe_recall": float(safe_recall),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _select_threshold(target: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], score, [1.0])))
    ranked = [
        (_threshold_metrics(target, score, float(value))["balanced_accuracy"], -float(value), float(value))
        for value in candidates
    ]
    return max(ranked)[2]


def _metrics(
    target: np.ndarray,
    score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    result = _threshold_metrics(target, score, threshold)
    result.update(
        {
            "samples": int(len(target)),
            "positives": int(target.sum()),
            "average_precision": _average_precision(target, score),
            "roc_auc": _roc_auc(target, score),
            "score_mean_safe": float(score[target == 0].mean()),
            "score_mean_risk": float(score[target == 1].mean()),
        }
    )
    return result


@torch.inference_mode()
def _predict(
    model: EfficientFutureSafetySidecar,
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    mode: str,
    shuffle_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    safety, future, target, _ = _batch(records, device)
    shuffle_index = None
    if shuffle_seed is not None:
        generator = torch.Generator(device=device).manual_seed(shuffle_seed)
        shuffle_index = torch.randperm(len(records), generator=generator, device=device)
        if bool((shuffle_index == torch.arange(len(records), device=device)).all()):
            shuffle_index = shuffle_index.roll(1)
    output = model(
        "bimanual_qpos14",
        safety,
        None if mode == "none" else future,
        future_mode=mode,
        shuffle_index=shuffle_index,
    )
    return target.cpu().numpy(), output["risk_probability"].float().cpu().numpy()


def _balanced_indices(records: list[dict[str, Any]], batch_size: int, generator: random.Random) -> list[int]:
    positive = [index for index, row in enumerate(records) if row["window"]["chunk_target"] == 1]
    negative = [index for index, row in enumerate(records) if row["window"]["chunk_target"] == 0]
    if not positive or not negative:
        raise RuntimeError("Training cache needs both safety classes")
    left = batch_size // 2
    chosen = [generator.choice(positive) for _ in range(left)]
    chosen.extend(generator.choice(negative) for _ in range(batch_size - left))
    generator.shuffle(chosen)
    return chosen


def _train_one(
    *,
    base: torch.nn.Module,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    device: torch.device,
    mode: str,
    seed: int,
    steps: int,
    batch_size: int,
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
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-3,
    )
    sampler = random.Random(seed)
    trace: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        indices = _balanced_indices(train_records, batch_size, sampler)
        selected = [train_records[index] for index in indices]
        safety, future, chunk_target, step_target = _batch(selected, device)
        output_tensors = model(
            "bimanual_qpos14", safety, future, future_mode=mode
        )
        chunk_loss = F.cross_entropy(output_tensors["class_logits"], chunk_target)
        step_loss = F.cross_entropy(
            output_tensors["step_class_logits"].reshape(-1, 2),
            step_target.reshape(-1),
        )
        loss = chunk_loss + 0.35 * step_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            trace.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "chunk_loss": float(chunk_loss.detach()),
                    "step_loss": float(step_loss.detach()),
                }
            )

    train_target, train_score = _predict(model, train_records, device, mode=mode)
    threshold = _select_threshold(train_target, train_score)
    eval_target, eval_score = _predict(model, eval_records, device, mode=mode)
    _, shuffled_score = _predict(
        model,
        eval_records,
        device,
        mode="shuffled",
        shuffle_seed=seed + 1000,
    )
    result = {
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "selected_threshold": threshold,
        "trace": trace,
        "train": _metrics(train_target, train_score, threshold=threshold),
        "eval": _metrics(eval_target, eval_score, threshold=threshold),
        "eval_shuffled_future": _metrics(
            eval_target, shuffled_score, threshold=threshold
        ),
    }
    run_dir = output / f"{mode}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("base.")
    }
    torch.save(
        {
            "schema_version": 1,
            "mode": mode,
            "seed": seed,
            "future_config": model.config.to_dict(),
            "adapter_state": adapter_state,
        },
        run_dir / "adapter.pt",
    )
    predictions = [
        {
            "window_id": row["window"]["window_id"],
            "target": int(target),
            "score": float(score),
            "shuffled_score": float(shuffled),
            "task": row["window"]["task"],
        }
        for row, target, score, shuffled in zip(
            eval_records, eval_target, eval_score, shuffled_score
        )
    ]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--portable-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", default="7,17,27")
    args = parser.parse_args()

    cache_root = args.feature_cache.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_rows = _read_jsonl(cache_root / "features.jsonl")
    records = [_torch_load(cache_root / row["record_path"]) for row in index_rows]
    train_records = [row for row in records if row["window"]["split"] == "train"]
    eval_records = [row for row in records if row["window"]["split"] == "eval"]
    if len(train_records) != 24 or len(eval_records) != 16:
        raise RuntimeError(
            f"Expected 24 train/16 eval records, got {len(train_records)}/{len(eval_records)}"
        )
    device = torch.device(args.device)
    loaded = load_multidomain_checkpoint(args.portable_checkpoint, map_location="cpu")
    base = loaded.model.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    probe = EfficientFutureSafetySidecar(copy.deepcopy(base)).to(device).eval()
    probe_target, probe_score = _predict(probe, eval_records, device, mode="none")
    safety, future, _, _ = _batch(eval_records, device)
    with torch.inference_mode():
        no_future = probe("bimanual_qpos14", safety, future_mode="none")
        zero_future = probe("bimanual_qpos14", safety, future, future_mode="full")
    zero_init_max_delta = float(
        (no_future["class_logits"] - zero_future["class_logits"]).abs().max()
    )
    if zero_init_max_delta != 0.0:
        raise RuntimeError(f"Future branch changed initial logits by {zero_init_max_delta}")
    fixed_threshold = float(loaded.profile_thresholds["bimanual_qpos14"].chunk_risk)
    reference = _metrics(probe_target, probe_score, threshold=fixed_threshold)
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = []
    seeds = [int(value) for value in args.seeds.split(",")]
    for mode in ("full", "mean"):
        for seed in seeds:
            result = _train_one(
                base=base,
                train_records=train_records,
                eval_records=eval_records,
                device=device,
                mode=mode,
                seed=seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                output=output,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "seed": seed,
                        "eval_ap": result["eval"]["average_precision"],
                        "eval_auc": result["eval"]["roc_auc"],
                        "shuffled_ap": result["eval_shuffled_future"]["average_precision"],
                    }
                ),
                flush=True,
            )

    def aggregate(mode: str, field: str) -> dict[str, float]:
        values = [
            result[field]["average_precision"]
            for result in results
            if result["mode"] == mode
        ]
        return {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": len(values)}

    full_ap = aggregate("full", "eval")
    full_shuffled_ap = aggregate("full", "eval_shuffled_future")
    mean_ap = aggregate("mean", "eval")
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Do spatially preserved Efficient-WAM future tokens improve an independent safety head on time-matched RoboTwin risk windows?",
        "scope": "small exploratory sanity slice; not a deployment or generalization claim",
        "feature_cache": str(cache_root),
        "feature_cache_summary_sha256": _sha256(cache_root / "SUMMARY.json"),
        "portable_checkpoint": str(args.portable_checkpoint.resolve()),
        "portable_checkpoint_sha256": _sha256(args.portable_checkpoint),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "class_counts": dict(
            Counter(record["window"]["risk"] for record in records)
        ),
        "zero_init_max_logit_delta": zero_init_max_delta,
        "no_future_reference": reference,
        "full_eval_ap": full_ap,
        "full_shuffled_eval_ap": full_shuffled_ap,
        "mean_eval_ap": mean_ap,
        "signal_check": {
            "full_minus_no_future_ap": full_ap["mean"] - reference["average_precision"],
            "full_minus_shuffled_ap": full_ap["mean"] - full_shuffled_ap["mean"],
            "full_minus_mean_ap": full_ap["mean"] - mean_ap["mean"],
            "passes_exploratory_check": bool(
                full_ap["mean"] > reference["average_precision"]
                and full_ap["mean"] > full_shuffled_ap["mean"]
            ),
        },
        "git": {
            "commit": _git(["rev-parse", "HEAD"], Path.cwd()),
            "branch": _git(["branch", "--show-current"], Path.cwd()),
            "status": _git(["status", "--porcelain=v1"], Path.cwd()),
        },
        "runs": results,
    }
    temporary = output / f"SUMMARY.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "SUMMARY.json")
    print(json.dumps(summary["signal_check"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
