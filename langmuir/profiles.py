"""Shear U'(z) and drift D'(z) profile definitions.

Profiles are polynomial representations on z in [-1, 0] as in
Hayes & Phillips (2017) equation (7).
"""

from __future__ import annotations

import numpy as np
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

    Uses polynomial fits to wind-driven shear and shallow-water Stokes drift,
    following Phillips et al. (2010).
    """
    import math

    depth = params.depth
    u_star = params.u_star
    D_max = params.D_max
    kappa_vk = params.kappa_vk

    # Wind-driven shear profile: log-layer near surface + return current
    # Nondimensional U'(z) on z in [-1, 0]
    # Approximate with degree-3 polynomial fit
    z_pts = np.linspace(-1, 0, 200)

    # Dimensional shear: du/dz = u_star / (kappa * (z_dim + depth + z0))
    # with return current for mass conservation
    z0 = 5e-3 / depth  # nondimensional roughness
    U_surface = params.U_surface

    # Compute dimensional velocity profile
    z_dim = z_pts * depth
    u_profile = np.zeros_like(z_pts)
    for i, zd in enumerate(z_dim):
        dist = (zd + depth) / depth + z0
        u_profile[i] = u_star / kappa_vk * math.log(max(dist / z0, 1.01))

    # Subtract return current for mass conservation
    u_mean = np.trapz(u_profile, z_pts)
    u_profile -= u_mean
    # Normalise so max |U'| = 1
    du = np.gradient(u_profile, z_pts)
    du_max = max(np.max(np.abs(du)), 1e-10)
    du_norm = du / du_max

    # Fit polynomial of degree 3
    b_coeffs = np.polyfit(z_pts, du_norm, 3)[::-1]

    # Shallow-water Stokes drift profile
    omega_p = 2.0 * math.pi / params.T_p
    k_p = omega_p**2 / params.g
    if depth < params.lambda_p / 2.0:
        from scipy.optimize import brentq

        def disp(k):
            return params.g * k * math.tanh(k * depth) - omega_p**2

        k_upper = max(50.0 / max(depth, 0.1), k_p * 20.0, 1.0)
        while disp(k_upper) < 0:
            k_upper *= 2.0
        k_p = brentq(disp, 1e-8, k_upper)

    # Stokes drift derivative d(u_s)/dz (nondimensional)
    a = params.H_s / 2.0
    denom = math.sinh(k_p * depth)**2
    drift_shear = np.zeros_like(z_pts)
    for i, zp in enumerate(z_pts):
        z_d = zp * depth
        drift_shear[i] = (omega_p * k_p**2 * a**2) * np.sinh(2.0 * k_p * (z_d + depth)) / max(denom, 1e-12)

    # Normalise
    ds_max = max(np.max(np.abs(drift_shear)), 1e-10)
    drift_norm = drift_shear / ds_max

    # Fit polynomial of degree 3
    a_coeffs = np.polyfit(z_pts, drift_norm, 3)[::-1]

    profile = ShearDriftProfile(a_coeffs, b_coeffs, name="shallow_lake")

    # Check instability condition
    if not profile.check_instability_condition():
        import warnings
        warnings.warn("D'U' < 0 at some depths — LC may be bottom-limited.")

    return profile
