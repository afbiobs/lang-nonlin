from __future__ import annotations

import math

import pytest

from langmuir import RobinBoundaryConditions, get_profile, solve_linear


def test_uniform_cc_benchmark(uniform_cc_benchmark: dict[str, float | str]) -> None:
    result = solve_linear(
        get_profile(str(uniform_cc_benchmark["profile_name"])),
        RobinBoundaryConditions(
            gamma_s=float(uniform_cc_benchmark["gamma_s"]),
            gamma_b=float(uniform_cc_benchmark["gamma_b"]),
        ),
    )
    assert result.lcL == pytest.approx(float(uniform_cc_benchmark["lc"]), rel=3.0e-2, abs=5.0e-2)
    assert math.sqrt(result.RcL) == pytest.approx(
        float(uniform_cc_benchmark["sqrt_Rc"]),
        rel=4.0e-2,
        abs=6.0e-1,
    )


def test_linear_neutral_curve_explicitly_splits_robin_singularity() -> None:
    bcs = RobinBoundaryConditions(gamma_s=0.06, gamma_b=0.28)
    result = solve_linear(get_profile("uniform"), bcs)
    gamma = bcs.gamma
    l = 0.08

    regular = float(result.R0)
    for order in range(1, len(result.R_coeffs)):
        regular += float(result.R_coeffs[order] + gamma * result.R_tilde_coeffs[order]) * l ** (2 * order)
    expected = gamma * result.R0 / (l * l) + regular

    assert result.neutral_curve(l) == pytest.approx(expected)
    assert result.neutral_curve(0.02) > result.neutral_curve(0.05) > result.neutral_curve(0.2)
