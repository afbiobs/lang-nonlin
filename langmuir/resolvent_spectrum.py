"""Coarse resolvent-inspired response spectrum for visible Langmuir spacing.

This module does not attempt a full Orr-Sommerfeld/Squire resolvent solve.
Instead it provides a cheap hybrid spectrum built from:

1. The CL finite-amplitude growth curve on the unstable band.
2. A broadband response proxy based on the resolved mean Lagrangian state.
3. A visibility filter that maps energetic motions to observable surface spacing.

The purpose is to separate CL onset dynamics from the visible spacing selected
by the developed turbulent mean state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .nonlinear_solver import NonlinearResult
from .params import LCParams
from .utils import poly_derivative, poly_eval_at


def _integrate_profile(values: np.ndarray, z: np.ndarray) -> float:
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrator(values, z))


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    peak = float(np.max(arr)) if len(arr) > 0 else 0.0
    if peak <= 0.0 or not np.isfinite(peak):
        return np.zeros_like(arr)
    return arr / peak


def _weighted_log_target(l_values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    l = np.asarray(l_values, dtype=float)
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(np.sum(w))
    if total <= 0.0:
        return float("nan"), float("nan")
    log_l = np.log(np.clip(l, 1.0e-6, None))
    mean_log = float(np.sum(w * log_l) / total)
    variance = float(np.sum(w * (log_l - mean_log) ** 2) / total)
    return float(math.exp(mean_log)), float(math.sqrt(max(variance, 0.0)))


def _surface_weight(z: np.ndarray, depth: float) -> np.ndarray:
    weight = np.exp(z / max(0.22 * depth, 1.0e-6))
    integral = _integrate_profile(weight, z)
    return weight / max(integral, 1.0e-12)


def _cl_growth_spectrum(
    Ra: float,
    neutral_curve_NL,
    l_values: np.ndarray,
) -> np.ndarray:
    neutral = np.array([neutral_curve_NL(float(li)) for li in l_values], dtype=float)
    return np.clip(Ra / np.maximum(neutral, 1.0e-12) - 1.0, 0.0, None)


def _response_energy_proxy(
    params: LCParams,
    *,
    l_values: np.ndarray,
) -> np.ndarray:
    hydro = params.hydrodynamic_state
    z = np.asarray(hydro.z_physical_m, dtype=float)
    weight = _surface_weight(z, params.depth)
    lagrangian_shear = np.asarray(hydro.lagrangian_shear, dtype=float)
    stokes_gradient = np.asarray(hydro.stokes_gradient, dtype=float)
    nu_profile = np.asarray(hydro.nu_T_profile, dtype=float)

    lagrangian_curvature = np.gradient(lagrangian_shear, z)
    shear_rms = math.sqrt(max(_integrate_profile(weight * lagrangian_shear ** 2, z), 1.0e-12))
    stokes_rms = math.sqrt(max(_integrate_profile(weight * stokes_gradient ** 2, z), 1.0e-12))
    curvature_rms = math.sqrt(max(_integrate_profile(weight * lagrangian_curvature ** 2, z), 1.0e-12))
    nu_surface = max(_integrate_profile(weight * nu_profile, z), 1.0e-12)

    roll_strength = (1.15 * stokes_rms + 0.35 * shear_rms) * math.sqrt(1.0 + 0.49 / max(params.La_SL ** 2, 1.0e-12))
    vortex_strength = 0.85 * shear_rms + 0.45 * stokes_rms + 0.25 * curvature_rms
    roll_strength /= max(nu_surface, 1.0e-12)
    vortex_strength /= max(nu_surface, 1.0e-12)

    beta_roll = float(np.clip(0.70 + 0.45 * params.La_t, 0.55, 1.25))
    beta_vortex = float(np.clip(1.55 + 0.65 * params.La_t, 1.35, 2.65))
    surface_peak = float(np.clip(0.95 + 0.45 * params.La_t, 0.85, 2.40))
    surface_sigma = 0.42

    kz = np.asarray(l_values, dtype=float)
    roll_gain = roll_strength * np.sqrt(kz) / np.maximum((kz ** 2 + beta_roll ** 2) ** 2, 1.0e-12)
    surface_envelope = np.exp(-0.5 * ((np.log(kz) - math.log(surface_peak)) / surface_sigma) ** 2)

    kx_values = np.array([0.25, 0.50, 1.00, 1.50, 2.50], dtype=float)
    forcing_weights = np.array([1.00, 0.90, 0.75, 0.50, 0.28], dtype=float)
    streamwise_response = np.zeros_like(kz)
    for kx, forcing_weight in zip(kx_values, forcing_weights, strict=True):
        q2 = kx * kx + kz * kz + beta_vortex * beta_vortex
        tilt_factor = (kx * kx) / (kx * kx + kz * kz + 0.08)
        streamwise_response += forcing_weight * vortex_strength * tilt_factor / np.maximum(q2 * q2, 1.0e-12)

    return np.clip(roll_gain + 0.75 * surface_envelope * streamwise_response, 0.0, None)


def _visibility_filter(
    *,
    l_values: np.ndarray,
    params: LCParams,
    Ra: float,
    RcNL: float,
    psi_tilde_1_coeffs: np.ndarray,
    v_float: float,
    reference_l: float,
) -> np.ndarray:
    if Ra <= RcNL:
        return np.zeros_like(l_values)

    psi_prime = poly_derivative(psi_tilde_1_coeffs)
    surface_shape = abs(poly_eval_at(psi_prime, -1.0e-4))
    u_converge_base = (
        params.nu_T / max(params.depth, 1.0e-12)
        * math.sqrt(max(Ra - RcNL, 0.0))
        * max(surface_shape, 0.05)
    )
    resurfacing_time = params.depth / max(v_float, 1.0e-12)
    kz = np.asarray(l_values, dtype=float)
    half_cell = math.pi * params.depth / np.maximum(kz, 1.0e-6)
    tau_acc = half_cell / max(u_converge_base, 1.0e-12)
    timescale_ratio = resurfacing_time / np.maximum(tau_acc, 1.0e-12)
    line_density = 1.0 + 0.65 * np.sqrt(np.maximum(kz / max(reference_l, 0.08), 0.05))
    visibility = (timescale_ratio / (1.0 + timescale_ratio)) * np.clip((line_density - 1.0) / 0.65, 0.0, None)
    return _normalize(visibility)


@dataclass(frozen=True)
class HybridSpacingSpectrum:
    l_values: np.ndarray
    cl_growth: np.ndarray
    response_energy: np.ndarray
    visibility_weight: np.ndarray
    hybrid_energy: np.ndarray
    cl_target_l: float
    response_target_l: float
    visible_target_l: float
    response_bandwidth: float
    response_mix: float
    large_scale_fraction: float
    response_peak_gain: float
    hybrid_peak_gain: float


def build_hybrid_spacing_spectrum(
    *,
    params: LCParams,
    nl_result: NonlinearResult,
    coherence: float,
    setup_index: float,
    development_index: float,
    coherent_run_hours: float,
    v_float: float,
    n_scan: int = 64,
) -> HybridSpacingSpectrum:
    l_values = np.geomspace(0.18, 3.60, n_scan)
    cl_growth = _cl_growth_spectrum(params.Ra, nl_result.neutral_curve_NL, l_values)
    response_energy = _response_energy_proxy(params, l_values=l_values)
    visibility_weight = _visibility_filter(
        l_values=l_values,
        params=params,
        Ra=params.Ra,
        RcNL=nl_result.RcNL,
        psi_tilde_1_coeffs=nl_result.linear_result.psi_tilde_1_coeffs,
        v_float=v_float,
        reference_l=max(nl_result.lcNL, 0.1),
    )

    cl_norm = _normalize(cl_growth)
    response_norm = _normalize(response_energy)
    supercriticality = max((params.Ra - nl_result.RcNL) / max(nl_result.RcNL, 1.0e-12), 0.0)
    supercritical_drive = supercriticality / (0.45 + supercriticality)
    persistent_drive = 1.0 - math.exp(-coherent_run_hours / 6.0)
    response_mix = min(
        max(
            0.12
            + 0.32 * float(np.clip(coherence, 0.0, 1.0))
            + 0.18 * float(np.clip(setup_index, 0.0, 1.0))
            + 0.18 * float(np.clip(development_index, 0.0, 1.0))
            + 0.12 * supercritical_drive
            + 0.08 * persistent_drive,
            0.0,
        ),
        0.95,
    )

    hybrid_energy = visibility_weight * ((1.0 - response_mix) * cl_norm + response_mix * response_norm)
    cl_target_l, _ = _weighted_log_target(l_values, np.maximum(cl_norm, 1.0e-12))
    response_target_l, response_bandwidth = _weighted_log_target(l_values, np.maximum(response_norm, 1.0e-12))
    visible_target_l, _ = _weighted_log_target(l_values, np.maximum(hybrid_energy, 1.0e-12))

    if not np.isfinite(cl_target_l):
        cl_target_l = float(nl_result.lcNL)
    if not np.isfinite(response_target_l):
        response_target_l = float(nl_result.lcNL)
    if not np.isfinite(visible_target_l):
        visible_target_l = float(response_target_l)

    low_k_mask = l_values < max(nl_result.linear_result.lcL, 1.0e-6)
    large_scale_fraction = (
        float(np.sum(hybrid_energy[low_k_mask]) / np.sum(hybrid_energy))
        if np.sum(hybrid_energy) > 0.0
        else 0.0
    )

    return HybridSpacingSpectrum(
        l_values=l_values,
        cl_growth=cl_norm,
        response_energy=response_norm,
        visibility_weight=visibility_weight,
        hybrid_energy=_normalize(hybrid_energy),
        cl_target_l=float(cl_target_l),
        response_target_l=float(response_target_l),
        visible_target_l=float(visible_target_l),
        response_bandwidth=float(response_bandwidth if np.isfinite(response_bandwidth) else 0.0),
        response_mix=float(response_mix),
        large_scale_fraction=float(large_scale_fraction),
        response_peak_gain=float(np.max(response_norm) if len(response_norm) else 0.0),
        hybrid_peak_gain=float(np.max(hybrid_energy) if len(hybrid_energy) else 0.0),
    )
