from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_efficient_future_safety_multidomain_replay.py"
)
SPEC = importlib.util.spec_from_file_location("multidomain_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _records() -> list[dict]:
    return [
        {"window": {"chunk_target": label, "task": task, "window_id": f"{label}-{task}-{index}"}}
        for label in (0, 1)
        for task in ("a", "b", "c")
        for index in range(3)
    ]


def test_sampler_returns_balanced_labels() -> None:
    sampler = MODULE.TaskBalancedSampler(_records(), seed=7)
    sample = sampler.sample(8)
    labels = Counter(row["window"]["chunk_target"] for row in sample)
    assert labels == {0: 4, 1: 4}


def test_selection_prioritizes_worst_domain() -> None:
    stronger_average = {"worst_domain_ap": 0.6, "mean_domain_ap": 0.9}
    stronger_worst = {"worst_domain_ap": 0.7, "mean_domain_ap": 0.75}
    assert MODULE._is_better(stronger_worst, stronger_average)
    assert not MODULE._is_better(stronger_average, stronger_worst)


def test_domain_batch_counts_accept_explicit_ratio() -> None:
    counts = MODULE._domain_batch_counts(
        {
            "batch_size": 8,
            "domain_batch_counts": {"maniskill": 2, "robotwin": 6},
        },
        ["maniskill", "robotwin"],
    )
    assert counts == {"maniskill": 2, "robotwin": 6}


def test_domain_batch_counts_preserve_equal_default() -> None:
    counts = MODULE._domain_batch_counts(
        {"batch_size": 8}, ["maniskill", "robotwin"]
    )
    assert counts == {"maniskill": 4, "robotwin": 4}


def test_domain_loss_weights_are_normalized() -> None:
    weights = MODULE._domain_loss_weights(
        {
            "domain_loss_weights": {"maniskill": 1.0, "robotwin": 3.0},
        },
        ["maniskill", "robotwin"],
    )
    assert weights == {"maniskill": 0.25, "robotwin": 0.75}
