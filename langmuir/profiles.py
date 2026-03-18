"""Shear U'(z) and drift D'(z) profile definitions.

Profiles are polynomial representations on z in [-1, 0] as in
Hayes & Phillips (2017) equation (7).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from .utils import poly_eval


class ShearDriftProfile:
    """Polynomial representation of U'(z) and D'(z).

    D' = sum_{m=0}^{M} a_m z^m
    U' = sum_{n=0}^{N} b_n z^n
    """

    def __init__(self, a_coeffs: np.ndarray, b_coeffs: np.ndarray, name: str = "custom"):
        self.a = np.array(a_coeffs, dtype=float)
        self.b = np.array(b_coeffs, dtype=float)
        self.M = len(self.a) - 1
        self.N = len(self.b) - 1
        self.name = name

    def D_prime(self, z: np.ndarray | float) -> np.ndarray | float:
        return poly_eval(self.a, z)

    def U_prime(self, z: np.ndarray | float) -> np.ndarray | float:
        return poly_eval(self.b, z)

    def check_instability_condition(self, n_pts: int = 100) -> bool:
        """Check D'U' > 0 over [-1, 0] (Leibovich 1983 requirement)."""
        z = np.linspace(-1, 0, n_pts)
        product = self.D_prime(z) * self.U_prime(z)
        return bool(np.all(product >= -1e-10))

    def __repr__(self) -> str:
        return f"ShearDriftProfile(name='{self.name}', a={self.a.tolist()}, b={self.b.tolist()})"


# Predefined profiles from the paper
PROFILES = {
    "uniform": ShearDriftProfile(
        a_coeffs=[1.0], b_coeffs=[1.0], name="uniform"
    ),
    "linear_drift": ShearDriftProfile(
        a_coeffs=[1.0, 1.0], b_coeffs=[1.0], name="linear_drift"
    ),
    "linear_shear": ShearDriftProfile(
        a_coeffs=[1.0], b_coeffs=[1.0, 1.0], name="linear_shear"
    ),
    "both_linear": ShearDriftProfile(
        a_coeffs=[1.0, 1.0], b_coeffs=[1.0, 1.0], name="both_linear"
    ),
}


def get_profile(name: str) -> ShearDriftProfile:
    """Get a predefined profile by name."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}")
    return PROFILES[name]


def shallow_lake_profile(params) -> ShearDriftProfile:
    """Compute realistic U'(z) and D'(z) for a wind-driven shallow lake.

    The raw shear and Stokes-drift derivatives are highly surface intensified.
    A direct low-degree polynomial fit can introduce sign changes in D'U',
    which makes the CL solver unphysical for many observations. Instead we
    build positive polynomial shape functions on x = z + 1 in [0, 1] and let
    their skewness vary with the current forcing state.
    """
    x = Polynomial([1.0, 1.0])  # x = z + 1 maps [-1, 0] -> [0, 1]

    # Broad physical controls for how surface-trapped the profiles are.
    wave_depth_ratio = params.depth / max(params.lambda_p, 1e-6)
    wave_depth_ratio = min(max(wave_depth_ratio, 0.0), 1.0)
    langmuir_intensity = (1.2 - min(params.La_t, 1.2)) / 1.2
    langmuir_intensity = min(max(langmuir_intensity, 0.0), 1.0)

    # Keep some support through the full depth to avoid pathological onset
    # thresholds while still sharpening the profile near the surface when
    # waves are short or Langmuir forcing is intense.
    shear_floor = 0.15
    drift_floor = 0.08
    shear_skew = min(max(0.25 + 0.5 * langmuir_intensity, 0.0), 1.0)
    drift_skew = min(max(0.2 + 0.6 * wave_depth_ratio, 0.0), 1.0)

    shear_shape = Polynomial([shear_floor]) + (1.0 - shear_floor) * (
        (1.0 - shear_skew) * x + shear_skew * (x ** 3)
    )
    drift_shape = Polynomial([drift_floor]) + (1.0 - drift_floor) * (
        (1.0 - drift_skew) * (x ** 2) + drift_skew * (x ** 4)
    )

    profile = ShearDriftProfile(
        a_coeffs=np.asarray(drift_shape.coef, dtype=float),
        b_coeffs=np.asarray(shear_shape.coef, dtype=float),
        name="shallow_lake",
    )
    return profile
