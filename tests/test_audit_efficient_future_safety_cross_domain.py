from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_efficient_future_safety_cross_domain.py"
)
SPEC = importlib.util.spec_from_file_location("cross_domain_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_metrics_recomputes_binary_counts_and_ranking() -> None:
    rows = [
        {"target": 0, "score": 0.1, "prediction": 0},
        {"target": 0, "score": 0.8, "prediction": 1},
        {"target": 1, "score": 0.2, "prediction": 0},
        {"target": 1, "score": 0.9, "prediction": 1},
    ]
    result = MODULE._metrics(rows)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert np.isclose(result["average_precision"], (1 + 2 / 3) / 2)


def test_wilson_interval_contains_observed_fraction() -> None:
    interval = MODULE._wilson(43, 62)
    assert interval["low"] < 43 / 62 < interval["high"]
