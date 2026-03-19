from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def uniform_cc_benchmark() -> dict[str, float | str]:
    return {
        "profile_name": "uniform",
        "gamma_s": 0.06,
        "gamma_b": 0.28,
        "lc": 1.111,
        "sqrt_Rc": 14.035,
    }
