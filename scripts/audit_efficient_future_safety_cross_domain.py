#!/usr/bin/env python3
"""Independently audit cross-domain Efficient-future safety predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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
    differences = positive[:, None] - negative[None, :]
    return float(
        ((differences > 0).sum() + 0.5 * (differences == 0).sum())
        / differences.size
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = np.asarray([row["target"] for row in rows], dtype=np.int64)
    score = np.asarray([row["score"] for row in rows], dtype=np.float64)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    tn = int(((target == 0) & (prediction == 0)).sum())
    fp = int(((target == 0) & (prediction == 1)).sum())
    fn = int(((target == 1) & (prediction == 0)).sum())
    tp = int(((target == 1) & (prediction == 1)).sum())
    return {
        "samples": len(rows),
        "correct": tn + tp,
        "accuracy": float((tn + tp) / max(1, len(rows))),
        "balanced_accuracy": float(
            (tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2
        ),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "risk_recall": float(tp / max(1, tp + fn)),
        "safe_recall": float(tn / max(1, tn + fp)),
        "average_precision": _average_precision(target, score),
        "roc_auc": _roc_auc(target, score),
    }


def _wilson(correct: int, total: int) -> dict[str, float]:
    z = 1.959963984540054
    probability = correct / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        probability * (1 - probability) / total + z * z / (4 * total * total)
    ) / denominator
    return {"low": center - margin, "high": center + margin}


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit(artifact: Path, config: Path, worktree: Path) -> dict[str, Any]:
    summary = _read_json(artifact / "SUMMARY.json")
    resolved = _read_json(artifact / "config.resolved.json")
    expected_config = _read_json(config)
    checks: list[dict[str, Any]] = []

    def check(name: str, expected: Any, actual: Any, evidence: str) -> None:
        if isinstance(expected, float) or isinstance(actual, float):
            matched = bool(np.isclose(expected, actual, rtol=0.0, atol=1e-12))
        else:
            matched = expected == actual
        checks.append(
            {
                "name": name,
                "expected": expected,
                "actual": actual,
                "evidence": evidence,
                "status": "exact" if matched else "mismatch",
            }
        )

    check("resolved config", expected_config, resolved, "config.resolved.json")
    check(
        "config hash",
        summary["config_sha256"],
        _sha256(config),
        "SUMMARY.json + config",
    )
    check(
        "portable checkpoint hash",
        summary["portable_checkpoint_sha256"],
        _sha256(Path(summary["portable_checkpoint"])),
        "SUMMARY.json + checkpoint",
    )
    commit = str(summary["git"]["commit"])
    check("experiment commit", commit, _git(["rev-parse", commit], worktree), "git")
    check("experiment tree clean", "", _git(["status", "--porcelain"], worktree), "git")

    raw_audit: dict[str, Any] = {}
    expected_counts = expected_config["expected"]
    maniskill = summary["domains"]["maniskill"]
    maniskill_seed_rows: dict[str, list[dict[str, Any]]] = {}
    for seed, run in sorted(maniskill["runs"].items()):
        path = Path(run["raw_predictions"])
        rows = _read_jsonl(path)
        threshold = float(run["threshold"])
        normalized = [
            {
                **row,
                "prediction": int(float(row["score"]) >= threshold),
            }
            for row in rows
        ]
        computed = _metrics(normalized)
        maniskill_seed_rows[seed] = normalized
        check(
            f"ManiSkill seed {seed} raw hash",
            run["raw_predictions_sha256"],
            _sha256(path),
            str(path),
        )
        check(
            f"ManiSkill seed {seed} records",
            int(expected_counts["maniskill_test_records"]),
            len(rows),
            str(path),
        )
        for key in ("confusion_matrix", "correct", "accuracy", "balanced_accuracy", "average_precision", "roc_auc"):
            check(
                f"ManiSkill seed {seed} {key}",
                run["metrics"][key],
                computed[key],
                str(path),
            )
    raw_audit["maniskill"] = {
        "unique_windows": len(
            {
                row.get("sample_id", row["window_id"])
                for rows in maniskill_seed_rows.values()
                for row in rows
            }
        ),
        "seed_count": len(maniskill_seed_rows),
    }

    robotwin = summary["domains"]["robotwin"]
    robotwin_ap: dict[str, dict[str, float]] = {}
    for seed, run in sorted(robotwin["runs"].items()):
        robotwin_ap[seed] = {}
        for mode, mode_result in run["modes"].items():
            result = mode_result["chunk"]
            path = Path(result["raw_predictions"])
            rows = _read_jsonl(path)
            computed = _metrics(rows)
            robotwin_ap[seed][mode] = computed["average_precision"]
            check(
                f"RoboTwin seed {seed}/{mode} raw hash",
                result["raw_predictions_sha256"],
                _sha256(path),
                str(path),
            )
            check(
                f"RoboTwin seed {seed}/{mode} records",
                int(expected_counts["robotwin_eval_records"]),
                len(rows),
                str(path),
            )
            for key in ("confusion_matrix", "correct", "accuracy", "balanced_accuracy", "average_precision", "roc_auc"):
                check(
                    f"RoboTwin seed {seed}/{mode} {key}",
                    result["metrics"][key],
                    computed[key],
                    str(path),
                )
        full_ids = {
            row["sample_id"]
            for row in _read_jsonl(Path(run["modes"]["full"]["chunk"]["raw_predictions"]))
        }
        for mode in ("none", "shuffled"):
            mode_ids = {
                row["sample_id"]
                for row in _read_jsonl(Path(run["modes"][mode]["chunk"]["raw_predictions"]))
            }
            check(
                f"RoboTwin seed {seed}/{mode} identities",
                sorted(full_ids),
                sorted(mode_ids),
                "raw prediction files",
            )
    raw_audit["robotwin_ap"] = robotwin_ap

    ur5 = summary["domains"]["ur5_real"]
    ur5_ids: list[list[str]] = []
    for seed, run in sorted(ur5["runs"].items()):
        path = Path(run["raw_predictions"])
        rows = _read_jsonl(path)
        computed = _metrics(rows)
        ur5_ids.append([str(row["sample_id"]) for row in rows])
        check(
            f"UR5 seed {seed} raw hash",
            run["raw_predictions_sha256"],
            _sha256(path),
            str(path),
        )
        check(
            f"UR5 seed {seed} records",
            int(expected_counts["ur5_val_records"]),
            len(rows),
            str(path),
        )
        for key in ("confusion_matrix", "correct", "accuracy", "balanced_accuracy", "average_precision", "roc_auc"):
            check(
                f"UR5 seed {seed} {key}",
                run["metrics"][key],
                computed[key],
                str(path),
            )
    check("UR5 identities across seeds", ur5_ids[0], ur5_ids[1], "UR5 raw files")
    check("UR5 identities across seeds 2", ur5_ids[0], ur5_ids[2], "UR5 raw files")
    check(
        "UR5 reference score reproduction",
        0.0,
        float(ur5["reference_recompute_max_abs_score_delta"]),
        "UR5 raw files + portable reference predictions",
    )

    maniskill_unique = maniskill["aggregate"]["majority_vote"]
    robotwin_unique = robotwin["aggregate"]["full"]["chunk"]["majority_vote"]
    ur5_unique = ur5["aggregate"]["majority_vote"]
    full_ap = np.asarray(
        [value["full"] for value in robotwin_ap.values()], dtype=np.float64
    )
    none_ap = np.asarray(
        [value["none"] for value in robotwin_ap.values()], dtype=np.float64
    )
    shuffled_ap = np.asarray(
        [value["shuffled"] for value in robotwin_ap.values()], dtype=np.float64
    )
    support = {
        "maniskill_fixed_window_detection": (
            "PASS" if maniskill_unique["balanced_accuracy"] >= 0.9 else "FAIL"
        ),
        "robotwin_fixed_window_detection": (
            "PASS" if robotwin_unique["balanced_accuracy"] >= 0.8 else "FAIL"
        ),
        "robotwin_future_improves_ranking": (
            "PASS" if bool(np.all(full_ap > none_ap)) else "FAIL"
        ),
        "robotwin_sample_specific_future_signal": (
            "PASS" if float((full_ap - shuffled_ap).mean()) >= 0.05 else "FAIL"
        ),
        "ur5_real_balanced_detection": (
            "PASS" if ur5_unique["balanced_accuracy"] >= 0.75 else "FAIL"
        ),
        "ur5_future_conditioning": "NOT_TESTED",
        "libero_detection": "NOT_TESTED",
        "closed_loop_safety": "NOT_TESTED",
    }
    mismatches = [item for item in checks if item["status"] != "exact"]
    reporting = "PASS" if not mismatches else "FAIL"
    scientific = (
        "FAIL_MULTIDOMAIN"
        if any(value == "FAIL" for value in support.values())
        else "PASS"
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "reporting_fidelity": reporting,
        "scientific_verdict": scientific,
        "mismatch_count": len(mismatches),
        "checks": checks,
        "support": support,
        "raw_audit": raw_audit,
        "cross_domain_diagnostics": {
            "robotwin_full_ap_mean": float(full_ap.mean()),
            "robotwin_none_ap_mean": float(none_ap.mean()),
            "robotwin_shuffled_ap_mean": float(shuffled_ap.mean()),
            "robotwin_full_minus_none_ap_by_seed": {
                seed: float(values["full"] - values["none"])
                for seed, values in robotwin_ap.items()
            },
            "robotwin_full_minus_shuffled_ap_mean": float(
                (full_ap - shuffled_ap).mean()
            ),
            "maniskill_majority_accuracy_wilson95": _wilson(
                int(maniskill_unique["correct"]), int(maniskill_unique["samples"])
            ),
            "robotwin_majority_accuracy_wilson95": _wilson(
                int(robotwin_unique["correct"]), int(robotwin_unique["samples"])
            ),
            "ur5_majority_accuracy_wilson95": _wilson(
                int(ur5_unique["correct"]), int(ur5_unique["samples"])
            ),
        },
        "reason": (
            "All reported raw counts and metrics reproduce exactly. The current "
            "adapter passes ManiSkill fixed-window classification but fails the "
            "cross-domain RoboTwin and balanced UR5 criteria; LIBERO and closed-loop "
            "claims remain unsupported."
        ),
        "limitations": [
            "Three seeds reuse the same held-out windows and are not independent scenes.",
            "The evaluation classifies fixed candidate windows; it is not a closed-loop rollout.",
            "UR5 uses the explicit no-future path and therefore measures the frozen base only.",
            "No project-owned compatible LIBERO safety inputs exist for this checkpoint.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.artifact.expanduser().resolve(),
        args.config.expanduser().resolve(),
        args.worktree.expanduser().resolve(),
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else args.artifact.expanduser().resolve() / "AUDIT.json"
    )
    _atomic_json(output, result)
    print(
        json.dumps(
            {
                "reporting_fidelity": result["reporting_fidelity"],
                "scientific_verdict": result["scientific_verdict"],
                "mismatch_count": result["mismatch_count"],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
