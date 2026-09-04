#!/usr/bin/env python3
"""Train one future-safety adapter with balanced replay across domains."""

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
    _batch,
    _metrics,
    _read_jsonl,
    _select_threshold,
    _sha256,
    _torch_load,
)
from safety_verify_wam.portable import (  # noqa: E402
    EfficientFutureSafetyConfig,
    EfficientFutureSafetySidecar,
    MultiProfilePortableSafetyCore,
    load_multidomain_checkpoint,
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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


def _atomic_adapter(
    path: Path,
    model: EfficientFutureSafetySidecar,
    *,
    seed: int,
    step: int,
    selection: dict[str, float],
    include_base: bool = False,
    initialization: str = "pretrained",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if include_base or not name.startswith("base.")
    }
    torch.save(
        {
            "schema_version": 1,
            "mode": "full",
            "training": "balanced_multidomain_replay",
            "seed": seed,
            "step": step,
            "selection": selection,
            "initialization": initialization,
            "state_scope": "full_safety_head" if include_base else "future_adapter",
            "future_config": model.config.to_dict(),
            "adapter_state": state,
        },
        temporary,
    )
    os.replace(temporary, path)


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *command],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _initialize_base(
    pretrained: MultiProfilePortableSafetyCore,
    initialization: str,
) -> MultiProfilePortableSafetyCore:
    if initialization == "pretrained":
        return copy.deepcopy(pretrained)
    if initialization == "random":
        return MultiProfilePortableSafetyCore(copy.deepcopy(pretrained.config))
    raise ValueError(
        "base_initialization must be either 'pretrained' or 'random', "
        f"got {initialization!r}"
    )


class TaskBalancedSampler:
    """Sample equal labels while rotating uniformly over tasks."""

    def __init__(self, records: list[dict[str, Any]], seed: int) -> None:
        buckets: dict[int, dict[str, list[int]]] = {
            0: defaultdict(list),
            1: defaultdict(list),
        }
        for index, record in enumerate(records):
            label = int(record["window"]["chunk_target"])
            task = str(record["window"]["task"])
            buckets[label][task].append(index)
        if any(not values for values in buckets.values()):
            raise RuntimeError("Each domain needs both safety classes")
        self.records = records
        self.buckets = buckets
        self.random = random.Random(seed)

    def sample(self, count: int) -> list[dict[str, Any]]:
        if count < 2 or count % 2:
            raise ValueError("Per-domain batch size must be a positive even number")
        chosen: list[dict[str, Any]] = []
        for label in (0, 1):
            tasks = sorted(self.buckets[label])
            for _ in range(count // 2):
                task = self.random.choice(tasks)
                index = self.random.choice(self.buckets[label][task])
                chosen.append(self.records[index])
        self.random.shuffle(chosen)
        return chosen


def _load_domain(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    root = Path(config["feature_cache"]).expanduser().resolve()
    index = _read_jsonl(root / "features.jsonl")
    records = [_torch_load(root / row["record_path"]) for row in index]
    result = {
        split: [row for row in records if str(row["window"]["split"]) == split]
        for split in ("train", "eval", "test")
    }
    actual = {split: len(rows) for split, rows in result.items()}
    expected = {key: int(value) for key, value in config["expected"].items()}
    if actual != expected:
        raise RuntimeError(f"Expected split counts {expected}, got {actual}")
    return result


def _shuffled_records(
    records: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    generator = np.random.default_rng(seed)
    order = np.roll(generator.permutation(len(records)), 1)
    transformed = []
    for index, record in enumerate(records):
        row = dict(record)
        row["future_tokens"] = records[int(order[index])]["future_tokens"]
        transformed.append(row)
    return transformed


@torch.inference_mode()
def _predict(
    model: EfficientFutureSafetySidecar,
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    mode: str,
    batch_size: int,
    shuffle_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    selected = (
        _shuffled_records(records, seed=shuffle_seed)
        if mode == "shuffled"
        else records
    )
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for start in range(0, len(selected), batch_size):
        current = selected[start : start + batch_size]
        safety, future, target, _ = _batch(current, device)
        output = model(
            "bimanual_qpos14",
            safety,
            None if mode == "none" else future,
            future_mode="none" if mode == "none" else "full",
        )
        current_scores = output["risk_probability"].float().cpu().numpy()
        targets.append(target.cpu().numpy())
        scores.append(current_scores)
        metadata.extend(
            {
                "window_id": str(record["window"]["window_id"]),
                "scene_group_id": str(
                    record["window"].get(
                        "scene_group_id", record["window"]["window_id"]
                    )
                ),
                "task": str(record["window"]["task"]),
                "setting": str(record["window"].get("setting", "unknown")),
            }
            for record in current
        )
    return np.concatenate(targets), np.concatenate(scores), metadata


def _evaluate_ap(
    model: EfficientFutureSafetySidecar,
    domains: dict[str, dict[str, list[dict[str, Any]]]],
    device: torch.device,
    *,
    split: str,
    batch_size: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for offset, (name, values) in enumerate(sorted(domains.items())):
        records = values[split]
        target, score, _ = _predict(
            model, records, device, mode="full", batch_size=batch_size
        )
        _, shuffled, _ = _predict(
            model,
            records,
            device,
            mode="shuffled",
            batch_size=batch_size,
            shuffle_seed=seed + offset + 1000,
        )
        result[name] = {
            "full_ap": float(_metrics(target, score, threshold=0.5)["average_precision"]),
            "shuffled_ap": float(
                _metrics(target, shuffled, threshold=0.5)["average_precision"]
            ),
        }
    return result


def _selection_value(evaluation: dict[str, dict[str, float]]) -> dict[str, float]:
    values = [domain["full_ap"] for domain in evaluation.values()]
    return {
        "worst_domain_ap": float(min(values)),
        "mean_domain_ap": float(np.mean(values)),
        "worst_future_delta": float(
            min(
                domain["full_ap"] - domain["shuffled_ap"]
                for domain in evaluation.values()
            )
        ),
    }


def _is_better(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    if incumbent is None:
        return True
    return (
        candidate["worst_domain_ap"], candidate["mean_domain_ap"]
    ) > (incumbent["worst_domain_ap"], incumbent["mean_domain_ap"])


def _final_split(
    model: EfficientFutureSafetySidecar,
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    threshold: float,
    batch_size: int,
    seed: int,
    output: Path,
    domain: str,
    split: str,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    raw_by_mode: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    for offset, mode in enumerate(("full", "none", "shuffled")):
        raw_by_mode[mode] = _predict(
            model,
            records,
            device,
            mode=mode,
            batch_size=batch_size,
            shuffle_seed=seed + 2000 + offset,
        )
        target, score, metadata = raw_by_mode[mode]
        values[mode] = _metrics(target, score, threshold=threshold)
        rows = [
            {
                **meta,
                "target": int(label),
                "score": float(current_score),
                "threshold": float(threshold),
                "prediction": int(current_score >= threshold),
                "mode": mode,
            }
            for meta, label, current_score in zip(metadata, target, score)
        ]
        path = output / "predictions" / f"{domain}-{split}-{mode}.jsonl"
        _atomic_jsonl(path, rows)
        values[mode]["raw_predictions"] = str(path.resolve())
        values[mode]["raw_predictions_sha256"] = _sha256(path)
    values["full_minus_none_ap"] = float(
        values["full"]["average_precision"] - values["none"]["average_precision"]
    )
    values["full_minus_shuffled_ap"] = float(
        values["full"]["average_precision"]
        - values["shuffled"]["average_precision"]
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "config.resolved.json", config)

    seed = int(config["seed"])
    _seed_everything(seed)
    device = torch.device(str(config["device"]))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            float(config["cuda_memory_fraction"]), device=device
        )
        torch.cuda.reset_peak_memory_stats(device)

    domains = {
        name: _load_domain(value) for name, value in config["domains"].items()
    }
    portable_checkpoint = Path(config["portable_checkpoint"]).resolve()
    loaded = load_multidomain_checkpoint(portable_checkpoint, map_location="cpu")
    base_initialization = str(config.get("base_initialization", "pretrained"))
    freeze_base = bool(config.get("freeze_base", True))
    if base_initialization == "random" and freeze_base:
        raise ValueError("Random base initialization requires freeze_base=false")
    base = _initialize_base(loaded.model, base_initialization)
    future_config = EfficientFutureSafetyConfig(**config["future_adapter"])
    model = EfficientFutureSafetySidecar(
        base, future_config, freeze_base=freeze_base
    ).to(device)
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    samplers = {
        name: TaskBalancedSampler(values["train"], seed + offset)
        for offset, (name, values) in enumerate(sorted(domains.items()))
    }
    batch_size = int(config["batch_size"])
    if batch_size % len(domains):
        raise ValueError("Batch size must divide evenly over domains")
    per_domain = batch_size // len(domains)
    eval_batch_size = int(config["eval_batch_size"])
    eval_every = int(config["eval_every"])
    checkpoint_every = int(config["checkpoint_every"])
    steps = int(config["steps"])
    trace: list[dict[str, Any]] = []
    best_selection: dict[str, float] | None = None
    best_step = 0
    stale_evaluations = 0
    early_stopping_patience = int(config.get("early_stopping_patience_evals", 0))
    early_stopping_min_steps = int(config.get("early_stopping_min_steps", 0))
    started = time.monotonic()

    def evaluate(step: int) -> bool:
        nonlocal best_selection, best_step, stale_evaluations
        evaluation = _evaluate_ap(
            model,
            domains,
            device,
            split="eval",
            batch_size=eval_batch_size,
            seed=seed + step,
        )
        selection = _selection_value(evaluation)
        event = {
            "step": step,
            "elapsed_seconds": float(time.monotonic() - started),
            "evaluation": evaluation,
            "selection": selection,
        }
        trace.append(event)
        improved = _is_better(selection, best_selection)
        if improved:
            best_selection = selection
            best_step = step
            stale_evaluations = 0
            _atomic_adapter(
                output / "best.pt",
                model,
                seed=seed,
                step=step,
                selection=selection,
                include_base=not freeze_base,
                initialization=base_initialization,
            )
        elif step >= early_stopping_min_steps:
            stale_evaluations += 1
        _atomic_json(
            output / "progress.json",
            {"best_step": best_step, "best_selection": best_selection, "trace": trace},
        )
        print(json.dumps({"phase": "validation", **event}), flush=True)
        return improved

    evaluate(0)
    model.train()
    stopped_reason = "steps_completed"
    completed_steps = 0
    for step in range(1, steps + 1):
        chosen: list[dict[str, Any]] = []
        domain_index: list[int] = []
        for index, name in enumerate(sorted(domains)):
            selected = samplers[name].sample(per_domain)
            chosen.extend(selected)
            domain_index.extend([index] * len(selected))
        order = list(range(len(chosen)))
        random.shuffle(order)
        chosen = [chosen[index] for index in order]
        domain_tensor = torch.as_tensor(
            [domain_index[index] for index in order], dtype=torch.long, device=device
        )
        safety, future, chunk_target, step_target = _batch(chosen, device)
        tensors = model("bimanual_qpos14", safety, future, future_mode="full")
        chunk_each = F.cross_entropy(
            tensors["class_logits"], chunk_target, reduction="none"
        )
        step_each = F.cross_entropy(
            tensors["step_class_logits"].reshape(-1, 2),
            step_target.reshape(-1),
            reduction="none",
        ).reshape(len(chosen), -1).mean(dim=1)
        each = (
            float(config["chunk_loss_weight"]) * chunk_each
            + float(config["step_loss_weight"]) * step_each
        )
        domain_losses = [
            each[domain_tensor == index].mean() for index in range(len(domains))
        ]
        loss = torch.stack(domain_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        completed_steps = step
        if step == 1 or step % 25 == 0:
            print(
                json.dumps(
                    {
                        "phase": "training",
                        "step": step,
                        "loss": float(loss.detach()),
                        "domain_losses": [
                            float(value.detach()) for value in domain_losses
                        ],
                    }
                ),
                flush=True,
            )
        if step % checkpoint_every == 0:
            assert best_selection is not None
            _atomic_adapter(
                output / "latest.pt",
                model,
                seed=seed,
                step=step,
                selection=best_selection,
                include_base=not freeze_base,
                initialization=base_initialization,
            )
        if step % eval_every == 0 or step == steps:
            evaluate(step)
            model.train()
            if (
                early_stopping_patience > 0
                and step >= early_stopping_min_steps
                and stale_evaluations >= early_stopping_patience
            ):
                stopped_reason = "cross_domain_early_stopping"
                print(
                    json.dumps(
                        {
                            "phase": "early_stopping",
                            "step": step,
                            "best_step": best_step,
                            "stale_evaluations": stale_evaluations,
                        }
                    ),
                    flush=True,
                )
                break

    best_payload = _torch_load(output / "best.pt")
    incompatible = model.load_state_dict(best_payload["adapter_state"], strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("base.")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Best adapter mismatch: missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    results: dict[str, Any] = {}
    for domain, values in sorted(domains.items()):
        train_target, train_score, _ = _predict(
            model,
            values["train"],
            device,
            mode="full",
            batch_size=eval_batch_size,
        )
        threshold = float(_select_threshold(train_target, train_score))
        domain_result: dict[str, Any] = {"selected_threshold": threshold}
        for split in ("eval", "test"):
            if values[split]:
                domain_result[split] = _final_split(
                    model,
                    values[split],
                    device,
                    threshold=threshold,
                    batch_size=eval_batch_size,
                    seed=seed,
                    output=output,
                    domain=domain,
                    split=split,
                )
        results[domain] = domain_result

    acceptance = config["acceptance"]
    maniskill_test = results["maniskill"]["test"]
    robotwin_eval = results["robotwin"]["eval"]
    checks = {
        "maniskill_test_ap": bool(
            maniskill_test["full"]["average_precision"]
            >= float(acceptance["minimum_maniskill_test_ap"])
        ),
        "robotwin_eval_ap": bool(
            robotwin_eval["full"]["average_precision"]
            >= float(acceptance["minimum_robotwin_eval_ap"])
        ),
        "each_domain_future_delta": bool(
            maniskill_test["full_minus_shuffled_ap"]
            >= float(acceptance["minimum_each_domain_full_minus_shuffled_ap"])
            and robotwin_eval["full_minus_shuffled_ap"]
            >= float(acceptance["minimum_each_domain_full_minus_shuffled_ap"])
        ),
    }
    summary = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "created_at": _now(),
        "question": (
            "Does balanced ManiSkill/RoboTwin replay prevent the single-domain "
            "future safety adapter from forgetting RoboTwin?"
        ),
        "scope": "offline fixed candidate-action classification",
        "git": {
            "branch": _git(["branch", "--show-current"]),
            "commit": _git(["rev-parse", "HEAD"]),
            "status": _git(["status", "--porcelain=v1"]),
        },
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "portable_checkpoint": str(portable_checkpoint),
        "portable_checkpoint_sha256": _sha256(portable_checkpoint),
        "feature_caches": {
            name: {
                "path": str(Path(value["feature_cache"]).resolve()),
                "summary_sha256": _sha256(
                    Path(value["feature_cache"]).resolve() / "SUMMARY.json"
                ),
                "counts": {split: len(domains[name][split]) for split in domains[name]},
            }
            for name, value in config["domains"].items()
        },
        "seed": seed,
        "steps": completed_steps,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "stopped_reason": stopped_reason,
        "base_initialization": base_initialization,
        "freeze_base": freeze_base,
        "trainable_parameters": int(sum(value.numel() for value in trainable)),
        "best_step": best_step,
        "best_selection": best_selection,
        "trace": trace,
        "results": results,
        "acceptance_checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "peak_gpu_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _atomic_json(output / "SUMMARY.json", summary)
    print(
        json.dumps(
            {
                "phase": "complete",
                "verdict": summary["verdict"],
                "best_step": best_step,
                "output": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
