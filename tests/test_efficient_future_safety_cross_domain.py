from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_efficient_future_safety_cross_domain.py"
)
SPEC = importlib.util.spec_from_file_location("cross_domain_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_classification_summary_uses_risk_as_positive_class() -> None:
    result = MODULE._classification_summary(
        np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 1])
    )
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == 0.5
    assert result["risk_recall"] == 0.5
    assert result["safe_recall"] == 0.5


def test_seed_aggregate_reports_repeated_and_majority_decisions() -> None:
    targets = [0, 1, 1]
    predictions = {
        7: [0, 1, 0],
        17: [0, 1, 1],
        27: [1, 1, 1],
    }
    rows = {
        seed: [
            {"sample_id": str(index), "target": target, "prediction": prediction}
            for index, (target, prediction) in enumerate(zip(targets, values))
        ]
        for seed, values in predictions.items()
    }
    result = MODULE._seed_aggregate(rows)
    assert result["repeated_decisions"]["correct"] == 7
    assert result["repeated_decisions"]["samples"] == 9
    assert result["majority_vote"]["correct"] == 3
    assert result["seed_prediction_disagreement_rate"] == 2 / 3
