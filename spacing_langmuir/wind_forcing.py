from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def drag_coefficient(U10: float) -> float:
    return (0.8 + 0.065 * U10) * 1e-3


def water_friction_velocity(U10: float, rho_air: float, rho_w: float) -> tuple[float, float, float]:
    cd = drag_coefficient(U10)
    tau = rho_air * cd * U10**2
    return math.sqrt(tau / rho_w), cd, tau


def fetch_limited_wave(U10: float, fetch: float, depth: float, g: float) -> tuple[float, float, float, float]:
    x_hat = max(g * fetch / max(U10**2, 1e-12), 1.0)
    f_p = 3.5 * (g / U10) * x_hat ** (-0.33)
    H_s = 0.0016 * (U10**2 / g) * x_hat**0.5
    omega = 2.0 * math.pi * f_p

    def residual(k: float) -> float:
        return g * k * math.tanh(k * depth) - omega**2

    k_lower = 1.0e-8
    k_upper = max(50.0 / max(depth, 1e-6), omega**2 / g * 20.0, 1.0)
    while residual(k_upper) < 0.0:
        k_upper *= 2.0
    k_p = brentq(residual, k_lower, k_upper, maxiter=200)
    return f_p, H_s, k_p, omega


def stokes_drift_surface(omega: float, k: float, H_s: float, depth: float) -> float:
    a = H_s / 2.0
    denom = 2.0 * math.sinh(k * depth) ** 2
    return (omega * k * a**2) * math.cosh(2.0 * k * depth) / max(denom, 1e-12)


def stokes_drift_shear(z_array: np.ndarray, params) -> np.ndarray:
    z = np.asarray(z_array, dtype=float)
    k = params.peak_wavenumber
    H = params.depth
    a = params.significant_wave_height / 2.0
    omega = params.peak_angular_frequency
    denom = np.sinh(k * H) ** 2
    return (omega * k**2 * a**2) * np.sinh(2.0 * k * (z + H)) / np.maximum(denom, 1e-12)


def wind_current_shear(z_array: np.ndarray, params) -> np.ndarray:
    z = np.asarray(z_array, dtype=float)
    return params.u_star / (params.kappa * np.maximum(z + params.depth + params.z0, 1e-6))


def populate_forcing(params) -> None:
    u_star, cd, tau = water_friction_velocity(params.U10, params.rho_air, params.rho_w)
    f_p, H_s, k_p, omega = fetch_limited_wave(params.U10, params.fetch, params.depth, params.g)
    u_stokes0 = stokes_drift_surface(omega, k_p, H_s, params.depth)
    params.u_star = u_star
    params.drag_coefficient = cd
    params.wind_stress = tau
    params.peak_frequency = f_p
    params.significant_wave_height = H_s
    params.peak_wavenumber = k_p
    params.peak_angular_frequency = omega
    params.u_stokes_surface = u_stokes0
    params.langmuir_number = math.sqrt(params.u_star / max(params.u_stokes_surface, 1e-12))
