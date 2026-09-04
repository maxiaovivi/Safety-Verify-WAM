from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_efficient_future_maniskill.py"
SPEC = importlib.util.spec_from_file_location("audit_efficient_future_maniskill", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_average_precision_perfect_and_reversed() -> None:
    target = np.asarray([0, 1, 0, 1])
    assert MODULE._average_precision(target, np.asarray([0.1, 0.9, 0.2, 0.8])) == 1.0
    assert np.isclose(
        MODULE._average_precision(target, np.asarray([0.9, 0.2, 0.8, 0.1])),
        (1 / 3 + 2 / 4) / 2,
    )
