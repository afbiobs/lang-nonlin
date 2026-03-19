from __future__ import annotations

import math

import numpy as np
import pytest

from langmuir.params import LCParams


def test_drag_is_continuous_through_five_ms() -> None:
    low = LCParams(U10=4.9)
    high = LCParams(U10=5.1)
    cd_low = low.hydrodynamic_state.drag_coefficient
    cd_high = high.hydrodynamic_state.drag_coefficient
    assert abs(cd_high - cd_low) / max(cd_low, cd_high) < 0.2


def test_broadband_stokes_drift_exceeds_monochromatic_surface_estimate() -> None:
    params = LCParams(U10=6.0, depth=12.0, fetch=25000.0)
    omega_p = 2.0 * math.pi / params.T_p
    k_p = 2.0 * math.pi / params.lambda_p
    a = 0.5 * params.H_s
    monochromatic = omega_p * k_p * a * a
    assert params.D_max > monochromatic


def test_langmuir_viscosity_matches_enhancement_factor() -> None:
    params = LCParams(U10=7.0, depth=9.0, fetch=20000.0)
    hydro = params.hydrodynamic_state
    expected = hydro.nu_shear * math.sqrt(1.0 + 0.49 / (hydro.La_SL * hydro.La_SL))
    assert hydro.nu_T == pytest.approx(expected)


def test_current_profile_has_zero_net_transport() -> None:
    params = LCParams(U10=5.5, depth=9.0)
    hydro = params.hydrodynamic_state
    transport = np.trapezoid(hydro.current_velocity, hydro.z_physical_m)
    assert abs(float(transport)) < 1.0e-6


def test_hydrodynamic_state_exposes_resolved_lagrangian_profiles() -> None:
    params = LCParams(U10=6.0, depth=9.0, fetch=15000.0)
    hydro = params.hydrodynamic_state
    assert len(hydro.nu_T_profile) == len(hydro.z_physical_m)
    assert np.all(hydro.nu_T_profile > 0.0)
    assert np.allclose(hydro.lagrangian_velocity, hydro.current_velocity + hydro.stokes_drift)
    assert np.allclose(hydro.lagrangian_shear, hydro.current_shear + hydro.stokes_gradient)
