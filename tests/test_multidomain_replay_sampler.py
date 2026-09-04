from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import torch

from safety_verify_wam.portable import (
    MultiProfilePortableSafetyCore,
    MultiProfileSafetyConfig,
    ProfileAdapterConfig,
)


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


def _base_model() -> MultiProfilePortableSafetyCore:
    return MultiProfilePortableSafetyCore(
        MultiProfileSafetyConfig(
            profiles=(
                ProfileAdapterConfig(
                    key="bimanual_qpos14",
                    state_dim=14,
                    action_dim=14,
                    motion_mode="position_target",
                ),
            ),
            model_dim=32,
            vision_channels=(8, 16),
            transformer_layers=1,
            attention_heads=4,
            max_views=3,
            profile_specific_heads=True,
        )
    )


def test_random_initialization_keeps_architecture_without_copying_weights() -> None:
    torch.manual_seed(3)
    pretrained = _base_model()
    with torch.no_grad():
        pretrained.cls_token.fill_(7.0)
    torch.manual_seed(5)
    random_base = MODULE._initialize_base(pretrained, "random")
    assert random_base.config == pretrained.config
    assert not torch.equal(random_base.cls_token, pretrained.cls_token)


def test_pretrained_initialization_is_an_independent_exact_copy() -> None:
    pretrained = _base_model()
    copied = MODULE._initialize_base(pretrained, "pretrained")
    assert all(
        torch.equal(copied.state_dict()[name], value)
        for name, value in pretrained.state_dict().items()
    )
    assert all(
        copied.state_dict()[name].data_ptr() != value.data_ptr()
        for name, value in pretrained.state_dict().items()
    )
