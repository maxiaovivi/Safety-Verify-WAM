from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import torch


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


def test_pair_map_uses_opposite_label_in_same_scene() -> None:
    records = [
        {
            "window": {
                "chunk_target": label,
                "task": "a",
                "scene_group_id": "scene-1",
                "window_id": f"scene-1-{label}",
            }
        }
        for label in (0, 1)
    ]
    pairs, methods = MODULE._pair_map(records)
    assert pairs["scene-1-0"]["window"]["chunk_target"] == 1
    assert pairs["scene-1-1"]["window"]["chunk_target"] == 0
    assert methods == {"exact_scene": 2}


def test_pair_map_falls_back_to_narrow_task_stratum() -> None:
    records = [
        {
            "window": {
                "chunk_target": label,
                "task": "a",
                "setting": "randomized",
                "condition_frame_idx": 7,
                "window_id": f"independent-{label}",
            }
        }
        for label in (0, 1)
    ]
    pairs, methods = MODULE._pair_map(records)
    assert pairs["independent-0"]["window"]["chunk_target"] == 1
    assert pairs["independent-1"]["window"]["chunk_target"] == 0
    assert methods == {"matched_task_setting_condition_frame_idx": 2}


def test_paired_future_margin_loss_rewards_label_direction() -> None:
    target = torch.tensor([0, 1])
    true_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    paired_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    loss = MODULE._paired_future_margin_loss(
        true_logits, paired_logits, target, margin=0.1
    )
    assert torch.equal(loss, torch.zeros_like(loss))


def test_paired_future_margin_loss_penalizes_reversed_direction() -> None:
    target = torch.tensor([0, 1])
    true_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    paired_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    loss = MODULE._paired_future_margin_loss(
        true_logits, paired_logits, target, margin=0.1
    )
    assert bool((loss > 0).all())
