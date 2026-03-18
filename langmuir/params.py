"""Parameter definitions for the nonlinear CL Langmuir model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LCParams:
    """All parameters needed to specify a shallow-lake LC problem."""

    # --- Environmental forcing (dimensional) ---
    U10: float              # 10-m wind speed (m/s)
    depth: float = 9.0      # Water depth h (m)
    fetch: float = 15000.0  # Wind fetch (m)

    # --- Boundary condition parameters ---
    gamma_s: float = 0.06   # Surface Robin parameter (Cox & Leibovich 1993)
    gamma_b: float = 0.28   # Bottom Robin parameter (rigid bottom)

    # --- Shear/drift profile specification ---
    shear_type: str = "wind_driven"
    drift_type: str = "fetch_limited"

    # --- Colony properties (for secondary biological coupling) ---
    v_float: float = 1e-4       # Colony rise velocity (m/s)
    colony_radius: float = 250e-6  # Colony radius (m)
    rho_colony: float = 990.0   # Colony density (kg/m^3)

    # --- Physical constants ---
    g: float = 9.81
    rho_w: float = 998.2
    rho_air: float = 1.225
    kappa_vk: float = 0.41  # von Karman constant

    # --- Derived quantities (computed in __post_init__) ---
    u_star: float = field(init=False)
    U_surface: float = field(init=False)
    D_max: float = field(init=False)
    nu_T: float = field(init=False)
    Ra: float = field(init=False)
    H_s: float = field(init=False)
    T_p: float = field(init=False)
    lambda_p: float = field(init=False)
    La_t: float = field(init=False)

    def __post_init__(self) -> None:
        self._compute_derived()

    def _compute_derived(self) -> None:
        g = self.g
        U10 = max(self.U10, 0.01)
        depth = self.depth
        fetch = self.fetch

        # 1. Friction velocity with Charnock-like drag
        if U10 < 5.0:
            C_d = 1.0e-3
        else:
            C_d = (0.8 + 0.065 * U10) * 1e-3
        tau = self.rho_air * C_d * U10**2
        self.u_star = math.sqrt(tau / self.rho_w)

        # 2. Surface velocity (3% rule for wind-driven surface currents)
        self.U_surface = 0.03 * U10

        # 3. Fetch-limited wave parameters (JONSWAP parameterisation)
        x_hat = max(g * fetch / max(U10**2, 1e-12), 1.0)
        f_p = 3.5 * (g / U10) * x_hat**(-0.33)
        self.T_p = 1.0 / f_p
        self.H_s = 0.0016 * (U10**2 / g) * x_hat**0.5

        # Peak wavelength (deep-water dispersion)
        self.lambda_p = g * self.T_p**2 / (2.0 * math.pi)

        # Stokes drift at surface for fetch-limited waves
        omega_p = 2.0 * math.pi * f_p
        k_p = omega_p**2 / g  # deep-water approximation
        # Shallow-water correction
        if depth < self.lambda_p / 2.0:
            from scipy.optimize import brentq

            def disp(k):
                return g * k * math.tanh(k * depth) - omega_p**2

            k_upper = max(50.0 / max(depth, 0.1), k_p * 20.0, 1.0)
            while disp(k_upper) < 0:
                k_upper *= 2.0
            k_p = brentq(disp, 1e-8, k_upper)

        a = self.H_s / 2.0
        denom = 2.0 * math.sinh(k_p * depth) ** 2
        self.D_max = (omega_p * k_p * a**2) * math.cosh(2.0 * k_p * depth) / max(denom, 1e-12)

        # 4. Eddy viscosity (depth-averaged parabolic profile)
        self.nu_T = self.kappa_vk * self.u_star * depth / 6.0

        # 5. Rayleigh number
        self.Ra = self.U_surface * self.D_max * depth**2 / max(self.nu_T**2, 1e-30)

        # Turbulent Langmuir number
        self.La_t = math.sqrt(self.u_star / max(self.D_max, 1e-12))

    @property
    def gamma(self) -> float:
        return self.gamma_s + self.gamma_b
