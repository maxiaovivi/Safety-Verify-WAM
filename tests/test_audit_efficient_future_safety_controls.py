from __future__ import annotations

from scripts.audit_efficient_future_safety_controls import (
    _count_summary,
    _same,
    _wilson,
)


def test_recursive_audit_comparison_accepts_equal_float_structures() -> None:
    assert _same({"value": [1.0, 2]}, {"value": [1.0, 2]})
    assert not _same({"value": 1.0}, {"value": 1.01})


def test_count_summary_reports_exact_detection_rate() -> None:
    result = _count_summary([[150, 0], [1, 149]])
    assert result["correct"] == 299
    assert result["decisions"] == 300
    assert result["accuracy"] == 299 / 300
    assert result["risk_recall"] == 149 / 150
    assert result["safe_recall"] == 1.0


def test_wilson_interval_contains_observed_fraction() -> None:
    interval = _wilson(899, 900)
    assert interval["low"] < 899 / 900 < interval["high"]
