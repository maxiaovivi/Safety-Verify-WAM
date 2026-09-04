#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    ranked = target[order]
    positives = int(ranked.sum())
    if positives == 0:
        return 0.0
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def _softmax_risk(logits: torch.Tensor) -> float:
    values = logits.detach().float().reshape(-1).numpy()
    values = np.exp(values - values.max())
    return float(values[1] / values.sum())


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _bootstrap_deltas(
    *,
    windows: dict[str, dict[str, Any]],
    target: dict[str, int],
    left: dict[str, float],
    right: dict[str, float],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for window_id in sorted(target):
        groups[str(windows[window_id]["scene_group_id"])].append(window_id)
    group_ids = sorted(groups)
    generator = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = generator.choice(group_ids, size=len(group_ids), replace=True)
        ids = [window_id for group_id in sampled for window_id in groups[str(group_id)]]
        labels = np.asarray([target[window_id] for window_id in ids], dtype=np.int64)
        left_scores = np.asarray([left[window_id] for window_id in ids])
        right_scores = np.asarray([right[window_id] for window_id in ids])
        deltas.append(
            _average_precision(labels, left_scores)
            - _average_precision(labels, right_scores)
        )
    return {
        "unit": "matched_scene_group",
        "groups": len(group_ids),
        "replicates": replicates,
        "p2_5": _percentile(deltas, 0.025),
        "p50": _percentile(deltas, 0.5),
        "p97_5": _percentile(deltas, 0.975),
        "probability_positive": float(np.mean(np.asarray(deltas) > 0.0)),
    }


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    artifact = args.artifact.expanduser().resolve()
    slice_root = args.slice_root.expanduser().resolve()
    worktree = args.worktree.expanduser().resolve()
    summary = _read_json(artifact / "SUMMARY.json")
    slice_manifest = _read_json(slice_root / "MANIFEST.json")
    windows_list = _read_jsonl(slice_root / "windows.jsonl")
    windows = {str(row["window_id"]): row for row in windows_list}
    feature_summary = _read_json(artifact / "feature-cache" / "SUMMARY.json")
    feature_index = _read_jsonl(artifact / "feature-cache" / "features.jsonl")
    checks: list[dict[str, Any]] = []

    def check(claim: str, expected: Any, recomputed: Any, evidence: str) -> None:
        if isinstance(expected, float) or isinstance(recomputed, float):
            exact = bool(np.isclose(expected, recomputed, rtol=0.0, atol=1e-12))
        else:
            exact = expected == recomputed
        checks.append(
            {
                "claim": claim,
                "evidence": evidence,
                "expected": expected,
                "recomputed": recomputed,
                "status": "exact" if exact else "number-mismatch",
            }
        )

    check("run exit code", 0, int((artifact / "exit.code").read_text()), "exit.code")
    check("run phase", "completed", (artifact / "phase").read_text().strip(), "phase")
    training_commit = str(summary["git"]["commit"])
    check(
        "training commit resolves",
        training_commit,
        _git(["rev-parse", training_commit], worktree),
        "SUMMARY.json + git",
    )
    check("training worktree status", "", _git(["status", "--porcelain"], worktree), "git")
    check("slice windows", int(slice_manifest["windows"]), len(windows_list), "windows.jsonl")
    check("feature records", int(feature_summary["records"]), len(feature_index), "features.jsonl")
    check("summary train records", int(summary["train_records"]), sum(row["split"] == "train" for row in windows_list), "windows.jsonl")
    check("summary eval records", int(summary["eval_records"]), sum(row["split"] == "eval" for row in windows_list), "windows.jsonl")
    check("summary test records", int(summary["test_records"]), sum(row["split"] == "test" for row in windows_list), "windows.jsonl")
    check(
        "feature summary hash",
        str(summary["feature_cache_summary_sha256"]),
        _sha256(artifact / "feature-cache" / "SUMMARY.json"),
        "feature-cache/SUMMARY.json",
    )
    check(
        "slice manifest hash",
        str(feature_summary["slice_manifest_sha256"]),
        _sha256(slice_root / "MANIFEST.json"),
        "slice/MANIFEST.json",
    )
    check(
        "config snapshot",
        _sha256(worktree / "configs" / "experiments" / "efficient_future_safety_maniskill_1800_20260904.yaml"),
        _sha256(artifact / "config.snapshot.yaml"),
        "config.snapshot.yaml",
    )

    split_groups: dict[str, set[str]] = defaultdict(set)
    group_labels: dict[str, set[str]] = defaultdict(set)
    for row in windows_list:
        group = str(row["scene_group_id"])
        split_groups[str(row["split"])].add(group)
        group_labels[group].add(str(row["risk"]))
    overlap = sum(
        len(split_groups[left] & split_groups[right])
        for left, right in (("train", "eval"), ("train", "test"), ("eval", "test"))
    )
    malformed_pairs = sum(labels != {"safe", "risk"} for labels in group_labels.values())
    check("cross-split scene-group overlap", 0, overlap, "windows.jsonl")
    check("malformed safe/risk pairs", 0, malformed_pairs, "windows.jsonl")

    record_hash_failures: list[str] = []
    record_by_window: dict[str, dict[str, Any]] = {}
    base_score: dict[str, float] = {}
    for index_row in feature_index:
        window_id = str(index_row["window_id"])
        record_path = artifact / "feature-cache" / str(index_row["record_path"])
        if _sha256(record_path) != index_row["record_sha256"]:
            record_hash_failures.append(window_id)
        payload = _torch_load(record_path)
        record_by_window[window_id] = payload
        base_score[window_id] = _softmax_risk(payload["base_class_logits"])
        check(
            f"feature target/{window_id}",
            int(windows[window_id]["chunk_target"]),
            int(payload["window"]["chunk_target"]),
            str(index_row["record_path"]),
        )
    check("feature record hash failures", 0, len(record_hash_failures), "feature-cache/records")
    check("feature window uniqueness", len(feature_index), len(record_by_window), "features.jsonl")

    modes = ("full", "mean")
    seeds = (7, 17, 27)
    raw_scores: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(dict)
    aggregate_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for mode in modes:
        for seed in seeds:
            run_dir = artifact / f"{mode}-seed{seed}"
            result = _read_json(run_dir / "result.json")
            check(f"{mode}/seed{seed} mode", mode, result["mode"], "result.json")
            check(f"{mode}/seed{seed} seed", seed, int(result["seed"]), "result.json")
            check(f"{mode}/seed{seed} steps", 2000, int(result["steps"]), "result.json")
            raw_scores[mode][seed] = {}
            for split, filename, result_key in (
                ("eval", "predictions.jsonl", "eval"),
                ("test", "test_predictions.jsonl", "test"),
            ):
                rows = _read_jsonl(run_dir / filename)
                expected_ids = {window_id for window_id, row in windows.items() if row["split"] == split}
                actual_ids = {str(row["window_id"]) for row in rows}
                check(f"{mode}/seed{seed}/{split} identities", sorted(expected_ids), sorted(actual_ids), filename)
                labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
                scores = np.asarray([float(row["score"]) for row in rows])
                shuffled = np.asarray([float(row["shuffled_score"]) for row in rows])
                for row in rows:
                    window_id = str(row["window_id"])
                    check(
                        f"{mode}/seed{seed}/{window_id} target",
                        int(windows[window_id]["chunk_target"]),
                        int(row["target"]),
                        filename,
                    )
                ap = _average_precision(labels, scores)
                shuffled_ap = _average_precision(labels, shuffled)
                check(f"{mode}/seed{seed}/{split} AP", float(result[result_key]["average_precision"]), ap, filename)
                check(
                    f"{mode}/seed{seed}/{split} shuffled AP",
                    float(result[f"{result_key}_shuffled_future"]["average_precision"]),
                    shuffled_ap,
                    filename,
                )
                aggregate_values[mode][split].append(ap)
                aggregate_values[mode][f"{split}_shuffled"].append(shuffled_ap)
                if split == "test":
                    raw_scores[mode][seed]["score"] = {
                        str(row["window_id"]): float(row["score"]) for row in rows
                    }
                    raw_scores[mode][seed]["shuffled"] = {
                        str(row["window_id"]): float(row["shuffled_score"])
                        for row in rows
                    }

    def aggregate(mode: str, field: str) -> dict[str, float | int]:
        values = np.asarray(aggregate_values[mode][field], dtype=np.float64)
        return {"mean": float(values.mean()), "std": float(values.std()), "count": len(values)}

    aggregate_map = {
        "full_eval_ap": aggregate("full", "eval"),
        "full_shuffled_eval_ap": aggregate("full", "eval_shuffled"),
        "mean_eval_ap": aggregate("mean", "eval"),
        "full_primary_ap": aggregate("full", "test"),
        "full_shuffled_primary_ap": aggregate("full", "test_shuffled"),
        "mean_primary_ap": aggregate("mean", "test"),
    }
    for name, value in aggregate_map.items():
        for field in ("mean", "std", "count"):
            check(f"{name}.{field}", summary[name][field], value[field], "raw prediction files")

    target = {window_id: int(row["chunk_target"]) for window_id, row in windows.items() if row["split"] == "test"}
    test_ids = sorted(target)
    base_test_ap = _average_precision(
        np.asarray([target[window_id] for window_id in test_ids]),
        np.asarray([base_score[window_id] for window_id in test_ids]),
    )
    check(
        "no-future test AP",
        float(summary["no_future_test_reference"]["average_precision"]),
        base_test_ap,
        "feature-cache base_class_logits",
    )

    def ensemble(mode: str, field: str) -> dict[str, float]:
        return {
            window_id: float(
                np.mean([raw_scores[mode][seed][field][window_id] for seed in seeds])
            )
            for window_id in test_ids
        }

    full = ensemble("full", "score")
    full_shuffled = ensemble("full", "shuffled")
    mean = ensemble("mean", "score")
    bootstrap = {
        "full_minus_no_future_ap": _bootstrap_deltas(
            windows=windows,
            target=target,
            left=full,
            right=base_score,
            replicates=args.bootstrap_replicates,
            seed=20260904,
        ),
        "full_minus_shuffled_ap": _bootstrap_deltas(
            windows=windows,
            target=target,
            left=full,
            right=full_shuffled,
            replicates=args.bootstrap_replicates,
            seed=20260905,
        ),
        "full_minus_mean_ap": _bootstrap_deltas(
            windows=windows,
            target=target,
            left=full,
            right=mean,
            replicates=args.bootstrap_replicates,
            seed=20260906,
        ),
    }

    task_metrics: dict[str, Any] = {}
    for task in sorted({str(windows[window_id]["task"]) for window_id in test_ids}):
        ids = [window_id for window_id in test_ids if windows[window_id]["task"] == task]
        labels = np.asarray([target[window_id] for window_id in ids])
        task_metrics[task] = {
            "samples": len(ids),
            "risk": int(labels.sum()),
            "no_future_ap": _average_precision(labels, np.asarray([base_score[item] for item in ids])),
            "full_ap": _average_precision(labels, np.asarray([full[item] for item in ids])),
            "full_shuffled_ap": _average_precision(labels, np.asarray([full_shuffled[item] for item in ids])),
            "mean_ap": _average_precision(labels, np.asarray([mean[item] for item in ids])),
        }

    mismatches = [row for row in checks if row["status"] != "exact"]
    reporting = "PASS" if not mismatches else "FAIL"
    result = {
        "schema_version": 1,
        "training_git": summary["git"],
        "audit_git": {
            "branch": _git(["branch", "--show-current"], worktree),
            "commit": _git(["rev-parse", "HEAD"], worktree),
            "script_sha256": _sha256(worktree / "scripts" / "audit_efficient_future_maniskill.py"),
        },
        "verdict": "WARN" if reporting == "PASS" else "FAIL",
        "reporting_fidelity": reporting,
        "scientific_support": "WARN",
        "reason": (
            "Raw predictions, splits, and hashes are internally consistent. "
            "The held-out classifier is strong, but full futures are nearly "
            "unchanged by shuffling and there is no equally trained no-future "
            "capacity control or closed-loop evaluation."
        ),
        "mismatch_count": len(mismatches),
        "record_hash_failures": record_hash_failures,
        "checks": checks,
        "bootstrap": bootstrap,
        "task_metrics": task_metrics,
        "limitations": [
            "Fixed candidate-action classification is not closed-loop deployment safety.",
            "The frozen no-future checkpoint is not an equal-capacity trained control.",
            "Near-perfect full-mode AP after future shuffling does not support a sample-specific future-grounding claim.",
            "No wall-clock latency or memory boundary was recorded for deployment.",
        ],
    }
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    checks = result["checks"]
    bootstrap = result["bootstrap"]
    task_metrics = result["task_metrics"]
    lines = [
        "# Efficient future safety ManiSkill audit",
        "",
        f"Verdict: **{result['verdict']}**",
        "",
        f"- Reporting fidelity: **{result['reporting_fidelity']}**",
        f"- Scientific support: **{result['scientific_support']}**",
        f"- Exact checks: {sum(row['status'] == 'exact' for row in checks)}/{len(checks)}",
        f"- Feature-record hash failures: {len(result['record_hash_failures'])}/1800",
        "",
        "## Matched-scene bootstrap",
        "",
        "```json",
        json.dumps(bootstrap, indent=2, sort_keys=True),
        "```",
        "",
        "## Task-level test AP",
        "",
        "```json",
        json.dumps(task_metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Bounded interpretation",
        "",
        result["reason"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--slice-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
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
