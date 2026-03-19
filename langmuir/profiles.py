"""Shear U'(z) and drift D'(z) profile definitions.

Profiles are polynomial representations on z in [-1, 0] as in
Hayes & Phillips (2017) equation (7).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import nnls

from .utils import poly_eval


class ShearDriftProfile:
    """Polynomial representation of U'(z) and D'(z)."""

    def __init__(
        self,
        a_coeffs: np.ndarray,
        b_coeffs: np.ndarray,
        name: str = "custom",
        *,
        drift_amplitude: float = 1.0,
        shear_amplitude: float = 1.0,
    ):
        self.a = np.array(a_coeffs, dtype=float)
        self.b = np.array(b_coeffs, dtype=float)
        self.M = len(self.a) - 1
        self.N = len(self.b) - 1
        self.name = name
        self.drift_amplitude = float(drift_amplitude)
        self.shear_amplitude = float(shear_amplitude)

    def D_prime(self, z: np.ndarray | float) -> np.ndarray | float:
        return poly_eval(self.a, z)

    def U_prime(self, z: np.ndarray | float) -> np.ndarray | float:
        return poly_eval(self.b, z)

    def check_instability_condition(self, n_pts: int = 100) -> bool:
        z = np.linspace(-1, 0, n_pts)
        product = self.D_prime(z) * self.U_prime(z)
        return bool(np.all(product >= -1e-10))

    def __repr__(self) -> str:
        return f"ShearDriftProfile(name='{self.name}', a={self.a.tolist()}, b={self.b.tolist()})"


PROFILES = {
    "uniform": ShearDriftProfile(a_coeffs=[1.0], b_coeffs=[1.0], name="uniform"),
    "linear_drift": ShearDriftProfile(a_coeffs=[1.0, 1.0], b_coeffs=[1.0], name="linear_drift"),
    "linear_shear": ShearDriftProfile(a_coeffs=[1.0], b_coeffs=[1.0, 1.0], name="linear_shear"),
    "both_linear": ShearDriftProfile(a_coeffs=[1.0, 1.0], b_coeffs=[1.0, 1.0], name="both_linear"),
}


def get_profile(name: str) -> ShearDriftProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}")
    return PROFILES[name]


def _integrate_trapezoid(values: np.ndarray, z: np.ndarray) -> float:
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrator(values, z))


def _fit_positive_profile(z: np.ndarray, values: np.ndarray, *, degree: int) -> np.ndarray:
    safe = np.asarray(values, dtype=float)
    safe = np.maximum(safe, 1.0e-8)
    safe = safe / max(_integrate_trapezoid(safe, z), 1.0e-8)

    x = np.clip(np.asarray(z, dtype=float) + 1.0, 0.0, 1.0)
    basis = np.column_stack(
        [
            math.comb(degree, k) * (x ** k) * ((1.0 - x) ** (degree - k))
            for k in range(degree + 1)
        ]
    )
    bernstein_coeffs, _ = nnls(basis, safe)

    x_poly = Polynomial([1.0, 1.0])      # x = z + 1
    one_minus_x_poly = Polynomial([0.0, -1.0])  # 1 - x = -z
    poly = Polynomial([0.0])
    for k, coeff in enumerate(bernstein_coeffs):
        poly = poly + coeff * math.comb(degree, k) * (x_poly ** k) * (one_minus_x_poly ** (degree - k))

    coeffs = np.asarray(poly.coef, dtype=float)
    scale = max(_integrate_trapezoid(poly(z), z), 1.0e-8)
    return coeffs / scale


def shallow_lake_profile(params) -> ShearDriftProfile:
    """Fit physical shear and drift-gradient profiles into low-order polynomials."""
    hydro = params.hydrodynamic_state
    z = np.asarray(hydro.z_nondim, dtype=float)

    current = np.asarray(hydro.current_velocity, dtype=float)
    stokes = np.asarray(hydro.stokes_drift, dtype=float)

    # The CL theory uses U'(z) and D'(z), not the velocities themselves.
    # The previous implementation fit the velocity profiles directly, which
    # forced the return flow into U' and broke the D'U' >= 0 instability
    # requirement. Use monotone vertical gradients instead.
    current_shear = np.gradient(current, z)
    drift_gradient = np.gradient(stokes, z)

    current_shape = np.maximum(current_shear, 1.0e-6)
    drift_shape = np.maximum(drift_gradient, 1.0e-6)

    shear_amplitude = _integrate_trapezoid(np.abs(current_shear), z)
    drift_amplitude = _integrate_trapezoid(np.abs(drift_gradient), z)

    b_coeffs = _fit_positive_profile(z, current_shape, degree=5)
    a_coeffs = _fit_positive_profile(z, drift_shape, degree=5)

    profile = ShearDriftProfile(
        a_coeffs=a_coeffs,
        b_coeffs=b_coeffs,
        name="shallow_lake",
        drift_amplitude=max(drift_amplitude, 1.0e-8),
        shear_amplitude=max(shear_amplitude, 1.0e-8),
    )
    if not profile.check_instability_condition():
        raise ValueError("Fitted shallow-lake profile violates D'U' >= 0.")
    return profile
