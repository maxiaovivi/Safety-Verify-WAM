from __future__ import annotations

import numpy as np
import torch

from scripts.run_efficient_future_safety_controls import (
    _complete_metrics,
    _constant_future,
    _pair_map,
    _variant_records,
)


def _record(group: str, label: int, value: float) -> dict:
    return {
        "video": torch.full((1,), value),
        "state": torch.full((1, 1), value),
        "action": torch.full((1, 2), value),
        "future_tokens": torch.full((3, 4), value),
        "window": {
            "window_id": f"{group}-{label}",
            "scene_group_id": group,
            "chunk_target": label,
        },
    }


def test_constant_future_is_position_preserving_training_mean() -> None:
    records = [_record("a", 0, 1.0), _record("a", 1, 3.0)]
    constant = _constant_future(records)
    assert constant.shape == (3, 4)
    assert torch.equal(constant, torch.full((3, 4), 2.0))


def test_paired_counterfactuals_change_only_requested_inputs() -> None:
    records = [_record("a", 0, 1.0), _record("a", 1, 3.0)]
    pairs = _pair_map(records)
    constant = _constant_future(records)

    future_swap = _variant_records(
        records,
        variant="paired_future_swap",
        pair_by_window=pairs,
        constant_future=constant,
    )
    assert torch.equal(future_swap[0]["future_tokens"], records[1]["future_tokens"])
    assert torch.equal(future_swap[0]["action"], records[0]["action"])
    assert future_swap[0]["window"] is records[0]["window"]

    action_swap = _variant_records(
        records,
        variant="paired_action_swap",
        pair_by_window=pairs,
        constant_future=constant,
    )
    assert torch.equal(action_swap[0]["action"], records[1]["action"])
    assert torch.equal(action_swap[0]["future_tokens"], records[0]["future_tokens"])


def test_detection_metrics_expose_exact_success_counts() -> None:
    target = np.asarray([0, 0, 1, 1], dtype=np.int64)
    score = np.asarray([0.1, 0.8, 0.9, 0.7], dtype=np.float64)
    result = _complete_metrics(target, score, threshold=0.75)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["correct"] == 2
    assert result["accuracy"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["false_negative_rate"] == 0.5
