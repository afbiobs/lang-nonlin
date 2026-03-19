"""Parameter definitions for the nonlinear CL Langmuir model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import exp1


@dataclass(frozen=True)
class HydrodynamicState:
    """Resolved near-surface forcing state used by the CL parameterisation."""

    drag_coefficient: float
    tau: float
    u_star: float
    U_surface: float
    D_surface: float
    D_max: float
    nu_shear: float
    nu_T: float
    H_s: float
    T_p: float
    lambda_p: float
    La_t: float
    La_SL: float
    pressure_gradient: float
    z_physical_m: np.ndarray
    z_nondim: np.ndarray
    nu_T_profile: np.ndarray
    current_velocity: np.ndarray
    stokes_drift: np.ndarray
    lagrangian_velocity: np.ndarray
    current_shear: np.ndarray
    stokes_gradient: np.ndarray
    lagrangian_shear: np.ndarray
    cl_drive_integral: float
    forcing_depth_fraction: float
    shear_to_drift_ratio: float
    cancellation_index: float
    S_wind: float


def _integrate_profile(values: np.ndarray, z: np.ndarray) -> float:
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrator(values, z))


def _safe_log_sensitivity(
    low_value: float,
    high_value: float,
    low_input: float,
    high_input: float,
) -> float:
    if (
        low_value <= 0.0
        or high_value <= 0.0
        or low_input <= 0.0
        or high_input <= 0.0
        or not np.isfinite(low_value)
        or not np.isfinite(high_value)
    ):
        return float("nan")
    return float((math.log(high_value) - math.log(low_value)) / (math.log(high_input) - math.log(low_input)))


def _wuest_lorke_drag(U10: float) -> float:
    return max(0.0044 * U10 ** (-1.15), 5.0e-4)


def _charnock_drag(U10: float, *, kappa_vk: float, g: float, K: float = 11.3) -> float:
    Cd = max(_wuest_lorke_drag(max(U10, 0.1)), 8.0e-4)
    for _ in range(32):
        u_star = math.sqrt(Cd) * U10
        z0 = max(u_star * u_star / max(g * K, 1e-12), 1.0e-6)
        Cd_new = (kappa_vk / max(math.log(10.0 / z0), 1.0)) ** 2
        if abs(Cd_new - Cd) < 1e-8:
            return float(Cd_new)
        Cd = 0.5 * (Cd + Cd_new)
    return float(Cd)


def _continuous_drag_coefficient(U10: float, *, kappa_vk: float, g: float) -> float:
    low_wind = _wuest_lorke_drag(U10)
    high_wind = _charnock_drag(U10, kappa_vk=kappa_vk, g=g)
    blend = 0.5 * (1.0 + math.tanh((U10 - 5.0) / 0.75))
    return float((1.0 - blend) * low_wind + blend * high_wind)


def _fetch_limited_wave_state(U10: float, fetch: float, depth: float, g: float) -> tuple[float, float, float]:
    x_hat = max(g * fetch / max(U10 * U10, 1e-12), 1.0)
    f_p = 3.5 * (g / U10) * x_hat ** (-0.33)
    T_p = 1.0 / max(f_p, 1e-12)
    H_s = 0.0016 * (U10 * U10 / g) * x_hat ** 0.5
    omega_p = 2.0 * math.pi * f_p
    k_p = omega_p * omega_p / g
    if depth < g * T_p * T_p / (4.0 * math.pi):
        def dispersion_relation(k: float) -> float:
            return g * k * math.tanh(k * depth) - omega_p * omega_p

        k_upper = max(50.0 / max(depth, 0.1), 20.0 * k_p, 1.0)
        while dispersion_relation(k_upper) < 0.0:
            k_upper *= 2.0
        k_p = brentq(dispersion_relation, 1.0e-8, k_upper)
    lambda_p = 2.0 * math.pi / max(k_p, 1.0e-12)
    return float(H_s), float(T_p), float(lambda_p)


def _broadband_stokes_profile(
    *,
    depth: float,
    H_s: float,
    T_p: float,
    lambda_p: float,
    fetch: float,
    U10: float,
    n_levels: int = 65,
) -> tuple[float, np.ndarray, np.ndarray]:
    omega_p = 2.0 * math.pi / max(T_p, 1.0e-12)
    k_p = 2.0 * math.pi / max(lambda_p, 1.0e-12)
    a = 0.5 * H_s
    monochromatic_surface = omega_p * k_p * a * a

    x_hat = max(9.81 * fetch / max(U10 * U10, 1.0e-12), 1.0)
    spectral_factor = 1.0 + 0.35 * math.tanh(math.log10(x_hat))
    D_surface = max(monochromatic_surface * spectral_factor, 1.0e-8)

    z_physical = np.linspace(-depth, 0.0, n_levels)
    z_nondim = z_physical / max(depth, 1.0e-12)
    attenuation = np.maximum(2.0 * k_p * np.abs(z_physical), 1.0e-6)
    profile = exp1(attenuation) / exp1(1.0e-6)
    profile[-1] = 1.0
    stokes = D_surface * np.clip(profile, 0.0, 1.0)
    return float(D_surface), z_physical, z_nondim, stokes


def _eddy_viscosity_profile(
    *,
    depth: float,
    nu_T: float,
    La_SL: float,
    n_levels: int = 65,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.linspace(-depth, 0.0, n_levels)
    sigma = np.clip((z + depth) / max(depth, 1.0e-12), 0.0, 1.0)
    surface_weight = np.exp(z / max(0.22 * depth, 1.0e-6))
    interior_shape = 0.65 + 0.55 * 4.0 * sigma * (1.0 - sigma)
    langmuir_surface = 1.0 + 0.35 / max(La_SL * La_SL, 1.0e-12) * surface_weight
    profile = np.clip(interior_shape * langmuir_surface, 0.15, None)
    mean_profile = _integrate_profile(profile, z) / max(depth, 1.0e-12)
    profile *= nu_T / max(mean_profile, 1.0e-12)
    return z, profile


def _solve_surface_current_profile(
    *,
    depth: float,
    tau: float,
    rho_w: float,
    nu_T: float | np.ndarray,
    n_levels: int = 65,
) -> tuple[np.ndarray, np.ndarray, float]:
    z = np.linspace(-depth, 0.0, n_levels)
    dz = float(z[1] - z[0])
    n = len(z)
    if np.isscalar(nu_T):
        nu_profile = np.full(n, float(nu_T), dtype=float)
    else:
        nu_profile = np.asarray(nu_T, dtype=float)
        if len(nu_profile) != n:
            raise ValueError("nu_T profile must match the vertical grid length.")
    mat = np.zeros((n + 1, n + 1), dtype=float)
    rhs = np.zeros(n + 1, dtype=float)

    for idx in range(1, n - 1):
        nu_down = 0.5 * (nu_profile[idx - 1] + nu_profile[idx])
        nu_up = 0.5 * (nu_profile[idx] + nu_profile[idx + 1])
        mat[idx, idx - 1] = nu_down / dz ** 2
        mat[idx, idx] = -(nu_down + nu_up) / dz ** 2
        mat[idx, idx + 1] = nu_up / dz ** 2
        mat[idx, -1] = -1.0

    mat[0, 0] = 1.0
    nu_surface = 0.5 * (nu_profile[-2] + nu_profile[-1])
    mat[n - 1, n - 2] = -nu_surface / dz
    mat[n - 1, n - 1] = nu_surface / dz
    rhs[n - 1] = tau / rho_w

    trap_weights = np.ones(n, dtype=float)
    trap_weights[0] = 0.5
    trap_weights[-1] = 0.5
    mat[n, :n] = trap_weights * dz

    solution = np.linalg.solve(mat, rhs)
    return z, solution[:n], float(solution[-1])


def _hydrodynamic_diagnostics(
    *,
    depth: float,
    z: np.ndarray,
    lagrangian_shear: np.ndarray,
    stokes_gradient: np.ndarray,
) -> tuple[float, float, float]:
    drive_profile = np.maximum(np.asarray(lagrangian_shear, dtype=float) * np.asarray(stokes_gradient, dtype=float), 0.0)
    cl_drive_integral = _integrate_profile(drive_profile, z)
    total_drive = max(cl_drive_integral, 1.0e-12)
    surface_mask = z >= -0.35 * depth
    forcing_depth_fraction = float(_integrate_profile(drive_profile[surface_mask], z[surface_mask]) / total_drive)
    shear_rms = math.sqrt(max(_integrate_profile(lagrangian_shear * lagrangian_shear, z) / max(depth, 1.0e-12), 1.0e-12))
    drift_rms = math.sqrt(max(_integrate_profile(stokes_gradient * stokes_gradient, z) / max(depth, 1.0e-12), 1.0e-12))
    shear_to_drift_ratio = float(shear_rms / max(drift_rms, 1.0e-12))
    return float(cl_drive_integral), float(np.clip(forcing_depth_fraction, 0.0, 1.0)), shear_to_drift_ratio


def _compute_hydrodynamic_core(
    *,
    U10: float,
    depth: float,
    fetch: float,
    rho_air: float,
    rho_w: float,
    kappa_vk: float,
    g: float,
) -> dict[str, float | np.ndarray]:
    drag_coefficient = _continuous_drag_coefficient(U10, kappa_vk=kappa_vk, g=g)
    tau = rho_air * drag_coefficient * U10 * U10
    u_star = math.sqrt(tau / rho_w)

    H_s, T_p, lambda_p = _fetch_limited_wave_state(U10, fetch, depth, g)
    D_surface, z_physical, z_nondim, stokes_drift = _broadband_stokes_profile(
        depth=depth,
        H_s=H_s,
        T_p=T_p,
        lambda_p=lambda_p,
        fetch=fetch,
        U10=U10,
    )

    nu_shear = kappa_vk * u_star * depth / 6.0
    La_SL = math.sqrt(max(u_star, 1.0e-12) / max(D_surface, 1.0e-12))
    nu_T = nu_shear * math.sqrt(1.0 + 0.49 / max(La_SL * La_SL, 1.0e-12))
    _, nu_T_profile = _eddy_viscosity_profile(
        depth=depth,
        nu_T=nu_T,
        La_SL=La_SL,
        n_levels=len(z_physical),
    )

    current_z, current_velocity, pressure_gradient = _solve_surface_current_profile(
        depth=depth,
        tau=tau,
        rho_w=rho_w,
        nu_T=nu_T_profile,
        n_levels=len(z_physical),
    )
    current_shear = np.gradient(current_velocity, current_z)
    stokes_gradient = np.gradient(stokes_drift, z_physical)
    lagrangian_velocity = current_velocity + stokes_drift
    lagrangian_shear = current_shear + stokes_gradient

    cl_drive_integral, forcing_depth_fraction, shear_to_drift_ratio = _hydrodynamic_diagnostics(
        depth=depth,
        z=np.asarray(current_z, dtype=float),
        lagrangian_shear=np.asarray(lagrangian_shear, dtype=float),
        stokes_gradient=np.asarray(stokes_gradient, dtype=float),
    )

    U_surface = float(current_velocity[-1])
    La_t = math.sqrt(max(u_star, 1.0e-12) / max(D_surface, 1.0e-12))
    Ra = U_surface * D_surface * depth * depth / max(nu_T * nu_T, 1.0e-30)
    S_wind = cl_drive_integral * depth * depth / max(nu_T, 1.0e-12)

    return {
        "drag_coefficient": float(drag_coefficient),
        "tau": float(tau),
        "u_star": float(u_star),
        "U_surface": float(U_surface),
        "D_surface": float(D_surface),
        "D_max": float(D_surface),
        "nu_shear": float(nu_shear),
        "nu_T": float(nu_T),
        "H_s": float(H_s),
        "T_p": float(T_p),
        "lambda_p": float(lambda_p),
        "La_t": float(La_t),
        "La_SL": float(La_SL),
        "pressure_gradient": float(pressure_gradient),
        "z_physical_m": np.asarray(current_z, dtype=float),
        "z_nondim": np.asarray(z_nondim, dtype=float),
        "nu_T_profile": np.asarray(nu_T_profile, dtype=float),
        "current_velocity": np.asarray(current_velocity, dtype=float),
        "stokes_drift": np.asarray(stokes_drift, dtype=float),
        "lagrangian_velocity": np.asarray(lagrangian_velocity, dtype=float),
        "current_shear": np.asarray(current_shear, dtype=float),
        "stokes_gradient": np.asarray(stokes_gradient, dtype=float),
        "lagrangian_shear": np.asarray(lagrangian_shear, dtype=float),
        "cl_drive_integral": float(cl_drive_integral),
        "forcing_depth_fraction": float(forcing_depth_fraction),
        "shear_to_drift_ratio": float(shear_to_drift_ratio),
        "Ra": float(Ra),
        "S_wind": float(S_wind),
    }


@dataclass
class LCParams:
    """All parameters needed to specify a shallow-lake LC problem."""

    U10: float
    depth: float = 9.0
    fetch: float = 15000.0

    gamma_s: float = 0.06
    gamma_b: float = 0.28

    shear_type: str = "wind_driven"
    drift_type: str = "fetch_limited"

    v_float: float = 1e-4
    colony_radius: float = 250e-6
    rho_colony: float = 990.0

    g: float = 9.81
    rho_w: float = 998.2
    rho_air: float = 1.225
    kappa_vk: float = 0.41

    hydrodynamic_state: HydrodynamicState = field(init=False)
    u_star: float = field(init=False)
    U_surface: float = field(init=False)
    D_max: float = field(init=False)
    nu_T: float = field(init=False)
    Ra: float = field(init=False)
    H_s: float = field(init=False)
    T_p: float = field(init=False)
    lambda_p: float = field(init=False)
    La_t: float = field(init=False)
    La_SL: float = field(init=False)

    def __post_init__(self) -> None:
        self._compute_derived()

    def _compute_derived(self) -> None:
        g = self.g
        U10 = max(float(self.U10), 0.05)
        depth = max(float(self.depth), 0.25)
        fetch = max(float(self.fetch), 1.0)

        core = _compute_hydrodynamic_core(
            U10=U10,
            depth=depth,
            fetch=fetch,
            rho_air=self.rho_air,
            rho_w=self.rho_w,
            kappa_vk=self.kappa_vk,
            g=g,
        )
        eps = 0.08
        low_u10 = max(U10 * (1.0 - eps), 0.05)
        high_u10 = U10 * (1.0 + eps)
        low_core = _compute_hydrodynamic_core(
            U10=low_u10,
            depth=depth,
            fetch=fetch,
            rho_air=self.rho_air,
            rho_w=self.rho_w,
            kappa_vk=self.kappa_vk,
            g=g,
        )
        high_core = _compute_hydrodynamic_core(
            U10=high_u10,
            depth=depth,
            fetch=fetch,
            rho_air=self.rho_air,
            rho_w=self.rho_w,
            kappa_vk=self.kappa_vk,
            g=g,
        )
        cancellation_index = _safe_log_sensitivity(
            float(low_core["Ra"]),
            float(high_core["Ra"]),
            low_u10,
            high_u10,
        )

        self.hydrodynamic_state = HydrodynamicState(
            drag_coefficient=float(core["drag_coefficient"]),
            tau=float(core["tau"]),
            u_star=float(core["u_star"]),
            U_surface=float(core["U_surface"]),
            D_surface=float(core["D_surface"]),
            D_max=float(core["D_max"]),
            nu_shear=float(core["nu_shear"]),
            nu_T=float(core["nu_T"]),
            H_s=float(core["H_s"]),
            T_p=float(core["T_p"]),
            lambda_p=float(core["lambda_p"]),
            La_t=float(core["La_t"]),
            La_SL=float(core["La_SL"]),
            pressure_gradient=float(core["pressure_gradient"]),
            z_physical_m=np.asarray(core["z_physical_m"], dtype=float),
            z_nondim=np.asarray(core["z_nondim"], dtype=float),
            nu_T_profile=np.asarray(core["nu_T_profile"], dtype=float),
            current_velocity=np.asarray(core["current_velocity"], dtype=float),
            stokes_drift=np.asarray(core["stokes_drift"], dtype=float),
            lagrangian_velocity=np.asarray(core["lagrangian_velocity"], dtype=float),
            current_shear=np.asarray(core["current_shear"], dtype=float),
            stokes_gradient=np.asarray(core["stokes_gradient"], dtype=float),
            lagrangian_shear=np.asarray(core["lagrangian_shear"], dtype=float),
            cl_drive_integral=float(core["cl_drive_integral"]),
            forcing_depth_fraction=float(core["forcing_depth_fraction"]),
            shear_to_drift_ratio=float(core["shear_to_drift_ratio"]),
            cancellation_index=float(cancellation_index),
            S_wind=float(core["S_wind"]),
        )
        self.u_star = self.hydrodynamic_state.u_star
        self.U_surface = self.hydrodynamic_state.U_surface
        self.D_max = self.hydrodynamic_state.D_max
        self.nu_T = self.hydrodynamic_state.nu_T
        self.Ra = float(core["Ra"])
        self.H_s = self.hydrodynamic_state.H_s
        self.T_p = self.hydrodynamic_state.T_p
        self.lambda_p = self.hydrodynamic_state.lambda_p
        self.La_t = self.hydrodynamic_state.La_t
        self.La_SL = self.hydrodynamic_state.La_SL

    @property
    def gamma(self) -> float:
        return self.gamma_s + self.gamma_b
