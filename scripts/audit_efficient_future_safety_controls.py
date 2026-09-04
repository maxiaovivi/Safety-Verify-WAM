#!/usr/bin/env python3
"""Audit raw evidence from the ManiSkill future-grounding controls."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_efficient_future_safety_controls import (  # noqa: E402
    VARIANTS,
    _aggregate,
    _complete_metrics,
    _constant_future,
    _pair_integrity,
    _tensor_sha256,
)
from scripts.train_efficient_future_full_data import (  # noqa: E402
    _average_precision,
    _read_jsonl,
    _sha256,
    _torch_load,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *command], cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def _same(expected: Any, recomputed: Any) -> bool:
    if isinstance(expected, dict) and isinstance(recomputed, dict):
        return expected.keys() == recomputed.keys() and all(
            _same(expected[key], recomputed[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)) and isinstance(recomputed, (list, tuple)):
        return len(expected) == len(recomputed) and all(
            _same(left, right) for left, right in zip(expected, recomputed)
        )
    if isinstance(expected, (int, float)) and isinstance(recomputed, (int, float)):
        return bool(
            np.isclose(float(expected), float(recomputed), rtol=0.0, atol=1e-12)
        )
    return expected == recomputed


def _wilson(correct: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return {"low": center - radius, "high": center + radius}


def _bootstrap_ap_delta(
    *,
    records: dict[str, dict[str, Any]],
    target: dict[str, int],
    left: dict[str, float],
    right: dict[str, float],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for window_id in sorted(target):
        groups[str(records[window_id]["window"]["scene_group_id"])].append(window_id)
    group_ids = sorted(groups)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = generator.choice(group_ids, size=len(group_ids), replace=True)
        ids = [window_id for group in sampled for window_id in groups[str(group)]]
        labels = np.asarray([target[window_id] for window_id in ids], dtype=np.int64)
        left_score = np.asarray([left[window_id] for window_id in ids])
        right_score = np.asarray([right[window_id] for window_id in ids])
        values.append(
            _average_precision(labels, left_score)
            - _average_precision(labels, right_score)
        )
    array = np.asarray(values, dtype=np.float64)
    return {
        "unit": "matched_scene_group",
        "groups": len(group_ids),
        "replicates": replicates,
        "p2_5": float(np.quantile(array, 0.025)),
        "p50": float(np.quantile(array, 0.5)),
        "p97_5": float(np.quantile(array, 0.975)),
        "probability_positive": float(np.mean(array > 0.0)),
    }


def _confusion(target: np.ndarray, prediction: np.ndarray) -> list[list[int]]:
    positive = target == 1
    negative = ~positive
    return [
        [int((~prediction & negative).sum()), int((prediction & negative).sum())],
        [int((~prediction & positive).sum()), int((prediction & positive).sum())],
    ]


def _count_summary(confusion: list[list[int]]) -> dict[str, Any]:
    (tn, fp), (fn, tp) = confusion
    total = tn + fp + fn + tp
    return {
        "correct": tn + tp,
        "decisions": total,
        "accuracy": (tn + tp) / total,
        "confusion_matrix": confusion,
        "risk_recall": tp / max(1, tp + fn),
        "safe_recall": tn / max(1, tn + fp),
        "wilson_95": _wilson(tn + tp, total),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    artifact = args.artifact.expanduser().resolve()
    source_artifact = args.source_artifact.expanduser().resolve()
    feature_cache = args.feature_cache.expanduser().resolve()
    training_worktree = args.training_worktree.expanduser().resolve()
    audit_worktree = Path.cwd().resolve()
    summary = _read_json(artifact / "SUMMARY.json")
    feature_summary = _read_json(feature_cache / "SUMMARY.json")
    feature_index = _read_jsonl(feature_cache / "features.jsonl")
    checks: list[dict[str, Any]] = []

    def check(claim: str, expected: Any, recomputed: Any, evidence: str) -> None:
        checks.append(
            {
                "claim": claim,
                "expected": expected,
                "recomputed": recomputed,
                "evidence": evidence,
                "status": "exact" if _same(expected, recomputed) else "number-mismatch",
            }
        )

    check("run exit code", 0, int((artifact / "exit.code").read_text()), "exit.code")
    check("run phase", "completed", (artifact / "phase").read_text().strip(), "phase")
    training_commit = str(summary["git"]["commit"])
    check(
        "training commit resolves",
        training_commit,
        _git(["rev-parse", "HEAD"], training_worktree),
        "SUMMARY.json + training worktree",
    )
    check(
        "training worktree clean",
        "",
        _git(["status", "--porcelain"], training_worktree),
        "training worktree",
    )
    check("training summary status", "", summary["git"]["status"], "SUMMARY.json")
    check(
        "configuration snapshot",
        _sha256(
            training_worktree
            / "configs/experiments/efficient_future_safety_maniskill_controls_20260904.yaml"
        ),
        _sha256(artifact / "config.snapshot.yaml"),
        "config.snapshot.yaml",
    )
    check(
        "feature summary hash",
        summary["feature_cache_summary_sha256"],
        _sha256(feature_cache / "SUMMARY.json"),
        "feature-cache/SUMMARY.json",
    )
    check(
        "source experiment summary hash",
        summary["source_summary_sha256"],
        _sha256(source_artifact / "SUMMARY.json"),
        "source artifact/SUMMARY.json",
    )

    record_hash_failures: list[str] = []
    records: list[dict[str, Any]] = []
    record_by_window: dict[str, dict[str, Any]] = {}
    for index_row in feature_index:
        path = feature_cache / str(index_row["record_path"])
        if _sha256(path) != str(index_row["record_sha256"]):
            record_hash_failures.append(str(index_row["window_id"]))
        record = _torch_load(path)
        records.append(record)
        record_by_window[str(record["window"]["window_id"])] = record
    check("feature record hash failures", 0, len(record_hash_failures), "features.jsonl")
    check("feature record count", feature_summary["records"], len(records), "features.jsonl")
    check("feature identity uniqueness", len(records), len(record_by_window), "feature records")
    split_records = {
        split: [record for record in records if record["window"]["split"] == split]
        for split in ("train", "eval", "test")
    }
    check(
        "split counts",
        summary["split_records"],
        {name: len(rows) for name, rows in split_records.items()},
        "feature records",
    )
    recomputed_pairs = {
        name: _pair_integrity(rows) for name, rows in split_records.items()
    }
    check("paired-scene integrity", summary["pair_integrity"], recomputed_pairs, "feature records")

    stored_constant = _torch_load(artifact / "train_mean_constant_future.pt")
    recomputed_constant = _constant_future(split_records["train"])
    check(
        "constant future exact tensor",
        True,
        bool(torch.equal(stored_constant, recomputed_constant)),
        "training feature records + train_mean_constant_future.pt",
    )
    check(
        "constant future tensor hash",
        summary["constant_future"]["tensor_sha256"],
        _tensor_sha256(stored_constant),
        "train_mean_constant_future.pt",
    )
    check(
        "constant future file hash",
        summary["constant_future"]["file_sha256"],
        _sha256(artifact / "train_mean_constant_future.pt"),
        "train_mean_constant_future.pt",
    )

    test_records = split_records["test"]
    expected_test_ids = {str(row["window"]["window_id"]) for row in test_records}
    expected_target = {
        str(row["window"]["window_id"]): int(row["window"]["chunk_target"])
        for row in test_records
    }
    seeds = [int(value) for value in summary["seeds"]]
    full_runs: list[dict[str, Any]] = []
    full_scores: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    full_thresholds: dict[int, float] = {}
    for seed in seeds:
        run_dir = artifact / f"full-probe-seed{seed}"
        result = _read_json(run_dir / "result.json")
        rows = _read_jsonl(run_dir / "test_predictions.jsonl")
        ids = {str(row["window_id"]) for row in rows}
        check(f"full seed {seed} test identities", sorted(expected_test_ids), sorted(ids), "test_predictions.jsonl")
        check(
            f"full seed {seed} targets",
            expected_target,
            {str(row["window_id"]): int(row["target"]) for row in rows},
            "test_predictions.jsonl",
        )
        target = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
        threshold = float(result["selected_threshold"])
        full_thresholds[seed] = threshold
        recomputed_variants = {}
        for variant in VARIANTS:
            score = np.asarray(
                [float(row["scores"][variant]) for row in rows], dtype=np.float32
            )
            metrics = _complete_metrics(target, score, threshold=threshold)
            recomputed_variants[variant] = metrics
            full_scores[seed][variant] = {
                str(row["window_id"]): float(row["scores"][variant]) for row in rows
            }
            check(
                f"full seed {seed} {variant} metrics",
                result["variants"][variant],
                metrics,
                "test_predictions.jsonl",
            )
        check(
            f"full seed {seed} adapter hash",
            result["source_adapter_sha256"],
            _sha256(source_artifact / f"full-seed{seed}/adapter.pt"),
            "source adapter",
        )
        source_scores = {
            str(row["window_id"]): float(row["score"])
            for row in _read_jsonl(
                source_artifact / f"full-seed{seed}/test_predictions.jsonl"
            )
        }
        max_delta = max(
            abs(source_scores[window_id] - full_scores[seed]["true_future"][window_id])
            for window_id in expected_test_ids
        )
        check(
            f"full seed {seed} source score reproduction",
            result["source_recompute_max_abs_score_delta"],
            max_delta,
            "source and control raw predictions",
        )
        full_runs.append({"variants": recomputed_variants})

    constant_runs: list[dict[str, Any]] = []
    constant_scores: dict[int, dict[str, float]] = {}
    constant_thresholds: dict[int, float] = {}
    split_ids = {
        split: {str(row["window"]["window_id"]) for row in rows}
        for split, rows in split_records.items()
    }
    split_targets = {
        split: {
            str(row["window"]["window_id"]): int(row["window"]["chunk_target"])
            for row in rows
        }
        for split, rows in split_records.items()
    }
    for seed in seeds:
        run_dir = artifact / f"constant-seed{seed}"
        result = _read_json(run_dir / "result.json")
        threshold = float(result["selected_threshold"])
        constant_thresholds[seed] = threshold
        recomputed_splits = {}
        for split in ("train", "eval", "test"):
            rows = _read_jsonl(run_dir / f"{split}_predictions.jsonl")
            ids = {str(row["window_id"]) for row in rows}
            check(f"constant seed {seed} {split} identities", sorted(split_ids[split]), sorted(ids), f"{split}_predictions.jsonl")
            check(
                f"constant seed {seed} {split} targets",
                split_targets[split],
                {str(row["window_id"]): int(row["target"]) for row in rows},
                f"{split}_predictions.jsonl",
            )
            target = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
            score = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
            metrics = _complete_metrics(target, score, threshold=threshold)
            recomputed_splits[split] = metrics
            check(
                f"constant seed {seed} {split} metrics",
                result[split],
                metrics,
                f"{split}_predictions.jsonl",
            )
            if split == "test":
                constant_scores[seed] = {
                    str(row["window_id"]): float(row["score"]) for row in rows
                }
        constant_runs.append(recomputed_splits)
        check(
            f"constant seed {seed} adapter exists",
            True,
            (run_dir / "adapter.pt").is_file(),
            "adapter.pt",
        )

    recomputed_aggregates = {
        "full_true_future": _aggregate(full_runs, ("variants", "true_future")),
        "full_paired_future_swap": _aggregate(full_runs, ("variants", "paired_future_swap")),
        "full_constant_future_at_test": _aggregate(full_runs, ("variants", "constant_future")),
        "full_zero_future_at_test": _aggregate(full_runs, ("variants", "zero_future")),
        "full_paired_action_swap": _aggregate(full_runs, ("variants", "paired_action_swap")),
        "full_paired_action_and_future_swap": _aggregate(
            full_runs, ("variants", "paired_action_and_future_swap")
        ),
        "trainable_constant_future": _aggregate(constant_runs, ("test",)),
    }
    check("aggregate metrics", summary["aggregates"], recomputed_aggregates, "all raw prediction files")

    ordered_ids = sorted(expected_test_ids)
    target_array = np.asarray([expected_target[window_id] for window_id in ordered_ids])

    def ensemble_score(score_map: dict[int, dict[str, float]]) -> dict[str, float]:
        return {
            window_id: float(np.mean([score_map[seed][window_id] for seed in seeds]))
            for window_id in ordered_ids
        }

    full_ensemble = ensemble_score(
        {seed: full_scores[seed]["true_future"] for seed in seeds}
    )
    pair_ensemble = ensemble_score(
        {seed: full_scores[seed]["paired_future_swap"] for seed in seeds}
    )
    constant_ensemble = ensemble_score(constant_scores)
    bootstrap = {
        "full_minus_paired_future_swap_ap": _bootstrap_ap_delta(
            records=record_by_window,
            target=expected_target,
            left=full_ensemble,
            right=pair_ensemble,
            replicates=args.bootstrap_replicates,
            seed=20260907,
        ),
        "full_minus_trainable_constant_future_ap": _bootstrap_ap_delta(
            records=record_by_window,
            target=expected_target,
            left=full_ensemble,
            right=constant_ensemble,
            replicates=args.bootstrap_replicates,
            seed=20260908,
        ),
    }

    detection_by_seed = {"full_true_future": {}, "trainable_constant_future": {}}
    for seed, run in zip(seeds, full_runs):
        detection_by_seed["full_true_future"][str(seed)] = run["variants"]["true_future"]
    for seed, run in zip(seeds, constant_runs):
        detection_by_seed["trainable_constant_future"][str(seed)] = run["test"]

    def majority(
        score_map: dict[int, dict[str, float]], thresholds: dict[int, float]
    ) -> dict[str, Any]:
        votes = np.stack(
            [
                np.asarray(
                    [score_map[seed][window_id] >= thresholds[seed] for window_id in ordered_ids]
                )
                for seed in seeds
            ]
        )
        prediction = votes.sum(axis=0) >= (len(seeds) // 2 + 1)
        return _count_summary(_confusion(target_array, prediction))

    majority_vote = {
        "full_true_future": majority(
            {seed: full_scores[seed]["true_future"] for seed in seeds},
            full_thresholds,
        ),
        "trainable_constant_future": majority(constant_scores, constant_thresholds),
    }

    strategies = {
        "full_true_future": {seed: full_scores[seed]["true_future"] for seed in seeds},
        "full_paired_future_swap": {
            seed: full_scores[seed]["paired_future_swap"] for seed in seeds
        },
        "full_paired_action_swap": {
            seed: full_scores[seed]["paired_action_swap"] for seed in seeds
        },
        "trainable_constant_future": constant_scores,
    }
    thresholds = {
        "full_true_future": full_thresholds,
        "full_paired_future_swap": full_thresholds,
        "full_paired_action_swap": full_thresholds,
        "trainable_constant_future": constant_thresholds,
    }
    task_metrics: dict[str, Any] = {}
    tasks = sorted({str(row["window"]["task"]) for row in test_records})
    for task in tasks:
        ids = sorted(
            window_id
            for window_id in ordered_ids
            if str(record_by_window[window_id]["window"]["task"]) == task
        )
        task_target = np.asarray([expected_target[window_id] for window_id in ids])
        task_metrics[task] = {}
        for name, score_map in strategies.items():
            confusion = np.zeros((2, 2), dtype=np.int64)
            for seed in seeds:
                prediction = np.asarray(
                    [score_map[seed][window_id] >= thresholds[name][seed] for window_id in ids]
                )
                confusion += np.asarray(_confusion(task_target, prediction))
            task_metrics[task][name] = _count_summary(confusion.tolist())

    evidence_hashes = {}
    for path in sorted(artifact.rglob("*")):
        if not path.is_file() or path.name in {"run.lock", "AUDIT.json", "AUDIT.md", "EVIDENCE_SHA256.json"}:
            continue
        evidence_hashes[str(path.relative_to(artifact))] = _sha256(path)

    mismatches = [row for row in checks if row["status"] != "exact"]
    fidelity = "PASS" if not mismatches else "FAIL"
    full_detection = recomputed_aggregates["full_true_future"]
    constant_detection = recomputed_aggregates["trainable_constant_future"]
    future_check = summary["future_grounding_check"]
    check(
        "future-grounding decision",
        False,
        bool(future_check["supports_sample_specific_future_grounding"]),
        "predeclared AP checks",
    )
    result = {
        "schema_version": 1,
        "verdict": "WARN" if fidelity == "PASS" else "FAIL",
        "reporting_fidelity": fidelity,
        "scientific_support": {
            "held_out_action_risk_detection": "PASS",
            "sample_specific_future_grounding": "FAIL",
            "closed_loop_robot_safety": "NOT_TESTED",
        },
        "reason": (
            "The raw held-out decisions support high fixed-window action-risk "
            "classification. An equal-capacity head trained with one constant "
            "future reaches the same AP and accuracy, while swapping candidate "
            "actions reverses predictions; sample-specific future grounding is "
            "therefore not supported."
        ),
        "mismatch_count": len(mismatches),
        "checks": checks,
        "feature_record_hash_failures": record_hash_failures,
        "detection_success": {
            "full_true_future_repeated_decisions": {
                **full_detection,
                "wilson_95": _wilson(
                    full_detection["correct"], full_detection["decisions"]
                ),
            },
            "trainable_constant_future_repeated_decisions": {
                **constant_detection,
                "wilson_95": _wilson(
                    constant_detection["correct"], constant_detection["decisions"]
                ),
            },
            "by_seed": detection_by_seed,
            "majority_vote_on_300_unique_test_windows": majority_vote,
            "note": "Three seeds reuse the same 300 held-out windows; 900 decisions are not 900 independent scenes.",
        },
        "counterfactual_aggregates": {
            name: recomputed_aggregates[name]
            for name in (
                "full_paired_future_swap",
                "full_constant_future_at_test",
                "full_zero_future_at_test",
                "full_paired_action_swap",
                "full_paired_action_and_future_swap",
            )
        },
        "bootstrap": bootstrap,
        "task_metrics": task_metrics,
        "pair_integrity": recomputed_pairs,
        "future_grounding_check": future_check,
        "training_git": summary["git"],
        "audit_git": {
            "branch": _git(["branch", "--show-current"], audit_worktree),
            "commit": _git(["rev-parse", "HEAD"], audit_worktree),
            "status": _git(["status", "--porcelain"], audit_worktree),
            "script_sha256": _sha256(
                audit_worktree / "scripts/audit_efficient_future_safety_controls.py"
            ),
        },
        "evidence_file_count": len(evidence_hashes),
        "limitations": [
            "The test classifies fixed candidate-action windows; it is not a closed-loop robot rollout.",
            "All three seeds reuse the same 300 held-out windows.",
            "The dataset pairs make the candidate action itself highly predictive of the label.",
            "No out-of-distribution scene, calibration-shift, or real-robot test is included.",
        ],
    }
    (artifact / "EVIDENCE_SHA256.json").write_text(
        json.dumps(evidence_hashes, indent=2, sort_keys=True) + "\n"
    )
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    full = result["detection_success"]["full_true_future_repeated_decisions"]
    constant = result["detection_success"]["trainable_constant_future_repeated_decisions"]
    by_seed = result["detection_success"]["by_seed"]["full_true_future"]
    future = result["future_grounding_check"]
    action_swap = result["counterfactual_aggregates"]["full_paired_action_swap"]
    lines = [
        "# Efficient-WAM 未来安全头对照审计",
        "",
        f"审计结论：**{result['verdict']}**",
        "",
        f"- 原始文件与汇总一致性：**{result['reporting_fidelity']}**",
        f"- 固定窗口动作危险检测：**{result['scientific_support']['held_out_action_risk_detection']}**",
        f"- 样本专属未来依赖：**{result['scientific_support']['sample_specific_future_grounding']}**",
        f"- 闭环机器人安全：**{result['scientific_support']['closed_loop_robot_safety']}**",
        "",
        "## 检测成功率",
        "",
        f"完整输入三种子合计：{full['correct']}/{full['decisions']} = {100 * full['accuracy']:.2f}%（同一批 300 个测试窗口重复三次）。",
        f"固定未来公平对照：{constant['correct']}/{constant['decisions']} = {100 * constant['accuracy']:.2f}%。",
        "",
        "| 种子 | 正确数/300 | 正确率 | 危险召回率 | 安全召回率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in sorted(by_seed, key=int):
        value = by_seed[seed]
        lines.append(
            f"| {seed} | {value['correct']}/300 | {100 * value['accuracy']:.2f}% | "
            f"{100 * value['risk_recall']:.2f}% | {100 * value['safe_recall']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 未来依赖检验",
            "",
            f"- 完整输入减去同场景未来互换的 AP：{future['full_minus_paired_future_swap_ap']:.6f}。",
            f"- 完整输入减去可训练固定未来对照的 AP：{future['full_minus_trainable_constant_future_ap']:.6f}。",
            f"- 预先规定的最小差值：{future['minimum_ap_delta']:.2f}；检验结果：未通过。",
            f"- 同场景只换动作时，三种子合计仅 {action_swap['correct']}/{action_swap['decisions']} 判断仍与原标签一致；预测主要跟随候选动作。",
            "",
            "## 有限结论",
            "",
            result["reason"],
            "",
            "这里的成功率是离线危险检测正确率，不是机器人闭环任务成功率。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--training-worktree", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result = audit(args)
    artifact = args.artifact.expanduser().resolve()
    temporary = artifact / "AUDIT.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(artifact / "AUDIT.json")
    _write_report(artifact / "AUDIT.md", result)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "reporting_fidelity": result["reporting_fidelity"],
                "scientific_support": result["scientific_support"],
                "mismatch_count": result["mismatch_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
