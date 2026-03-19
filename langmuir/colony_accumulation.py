"""Colony accumulation model with buoyancy and hourly forcing memory."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np

from .params import LCParams
from .profiles import get_profile, shallow_lake_profile
from .robin_bc import RobinBoundaryConditions
from .nonlinear_solver import SubcriticalBifurcationError, solve_nonlinear
from .rayleigh_mapping import classify_regime, unstable_band
from .resolvent_spectrum import build_hybrid_spacing_spectrum
from .utils import poly_derivative, poly_eval, poly_eval_at

LOGGER = logging.getLogger(__name__)


@dataclass
class LangmuirModeState:
    """Finite-time mode state used to evolve spacing through changing forcing."""

    selected_l: float | None = None
    target_l: float | None = None
    response_target_l: float | None = None
    cl_selected_l: float | None = None
    cl_target_l: float | None = None
    amplitude_index: float = 0.0
    development_index: float = 0.0
    regime: str = "subcritical"
    hydrodynamic_regime: str = "subcritical"
    merging_age_hours: float = 0.0
    setup_index: float = 0.0
    coherent_run_hours: float = 0.0
    response_bandwidth: float = 0.0
    response_mix: float = 0.0
    large_scale_fraction: float = 0.0
    visible_fraction: float = 0.0
    coarsening_index: float = 0.0
    buoyancy_Q: float = 0.15
    rho_cell: float = 990.0
    v_float: float = 1.0e-4


@dataclass(frozen=True)
class LangmuirDynamicsConfig:
    """Reduced, physically anchored controls for finite-time Langmuir evolution."""

    tau_relax_alpha: float = 0.08
    tau_decay_alpha: float = 0.14
    tau_coherence_alpha: float = 0.06
    merge_min_age_alpha: float = 0.04
    merge_supercriticality_threshold: float = 0.15
    merge_step_factor: float = 0.85
    coherence_threshold: float = 0.6

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_DYNAMICS = LangmuirDynamicsConfig()


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _spacing_from_l(l_value: float, depth: float) -> float:
    return 2.0 * math.pi / l_value * depth if l_value > 0.0 else float("nan")


def _l_from_spacing(spacing: float, depth: float) -> float:
    return 2.0 * math.pi * depth / spacing if spacing > 0.0 else float("nan")


def _diffusive_hours(params: LCParams) -> float:
    return max(params.depth * params.depth / max(params.nu_T, 1.0e-12) / 3600.0, 0.25)


def _relaxation_hours(params: LCParams, dynamics: LangmuirDynamicsConfig) -> float:
    return max(dynamics.tau_relax_alpha * _diffusive_hours(params), 0.25)


def _decay_hours(params: LCParams, dynamics: LangmuirDynamicsConfig) -> float:
    return max(dynamics.tau_decay_alpha * _diffusive_hours(params), 0.25)


def _coherence_decay_hours(params: LCParams, dynamics: LangmuirDynamicsConfig) -> float:
    return max(dynamics.tau_coherence_alpha * _diffusive_hours(params), 0.25)


def _merge_min_age_hours(params: LCParams, dynamics: LangmuirDynamicsConfig) -> float:
    return max(dynamics.merge_min_age_alpha * _diffusive_hours(params), 0.25)


def _formation_hours(
    params: LCParams,
    dynamics: LangmuirDynamicsConfig,
    *,
    supercriticality: float,
    coherence: float,
) -> float:
    base = 0.35 * _relaxation_hours(params, dynamics)
    speedup = 0.55 + 1.40 * max(supercriticality, 0.0) + 0.75 * _clip01(coherence)
    return float(min(max(base / max(speedup, 1.0e-6), 0.12), 0.75))


def _coarsening_hours(
    params: LCParams,
    dynamics: LangmuirDynamicsConfig,
    *,
    response_mix: float,
    persistent_drive: float,
) -> float:
    base = _relaxation_hours(params, dynamics)
    speedup = 0.65 + 0.90 * _clip01(response_mix) + 0.65 * _clip01(persistent_drive)
    return float(min(max(base / max(speedup, 1.0e-6), 0.30), 6.0))


def coherent_run_drive(
    params: LCParams,
    *,
    supercriticality: float,
    coherence: float,
    dynamics: LangmuirDynamicsConfig,
) -> float:
    if coherence <= dynamics.coherence_threshold:
        return 0.0
    coherence_drive = (coherence - dynamics.coherence_threshold) / max(1.0 - dynamics.coherence_threshold, 1.0e-6)
    wind_drive = 1.0 - math.exp(-params.u_star / max(0.5 * params.u_star + 1.0e-3, 1.0e-6))
    supercritical_drive = supercriticality / max(supercriticality + 0.25, 1.0e-6)
    return _clip01(wind_drive * supercritical_drive * coherence_drive)


def onset_mode_wavenumber(
    target_l: float,
    lcL: float,
    l_min: float,
    l_max: float,
) -> float:
    onset_l = lcL if lcL > 0 else target_l
    if math.isnan(l_min) or math.isnan(l_max):
        return float(max(onset_l, target_l))
    return float(min(max(onset_l, l_min), l_max))


def _advance_buoyancy(
    state: LangmuirModeState,
    params: LCParams,
    *,
    dt_hours: float,
    shortwave_radiation: float | None,
    temperature_c: float | None,
) -> tuple[float, float, float]:
    shortwave = max(float(shortwave_radiation or 0.0), 0.0)
    temperature = 20.0 if temperature_c is None or math.isnan(float(temperature_c)) else float(temperature_c)

    photo_rate = 0.08 * (1.0 - math.exp(-shortwave / 120.0)) * (1.07 ** ((temperature - 20.0) / 10.0))
    resp_rate = 0.018 * (1.08 ** ((temperature - 20.0) / 10.0))
    Q = float(np.clip(state.buoyancy_Q + dt_hours * (photo_rate - resp_rate), -0.25, 1.5))

    rho_min = params.rho_colony - 8.0 + 0.2 * (temperature - 20.0)
    c_t = 3.5 * (1.015 ** ((temperature - 20.0) / 10.0))
    rho_cell = float(rho_min + c_t * Q)

    mu = 1.0e-3 * math.exp(-0.033 * (temperature - 20.0))
    delta_rho = max(params.rho_w - rho_cell, 0.05)
    v_float = 2.0 * params.colony_radius ** 2 * params.g * delta_rho / max(9.0 * mu, 1.0e-12)
    return Q, rho_cell, float(max(v_float, 1.0e-8))


def supercritical_mode_spectrum(
    Ra: float,
    neutral_curve_NL,
    lcNL: float,
    RcNL: float = 0.0,
    *,
    n_scan: int = 256,
    spectrum_power: float = 2.0,
) -> dict:
    """Build a finite-width supercritical spectrum and derive a target mode."""
    if lcNL <= 0.0:
        return {
            "l_min": float("nan"),
            "l_max": float("nan"),
            "l_scan": np.array([], dtype=float),
            "growth_proxy": np.array([], dtype=float),
            "target_l": float("nan"),
            "peak_growth_proxy": 0.0,
        }

    l_scan_full = np.linspace(0.15 * lcNL, 4.0 * lcNL, n_scan)
    l_min, l_max = unstable_band(Ra, neutral_curve_NL, l_scan_full)
    if math.isnan(l_min) or math.isnan(l_max):
        return {
            "l_min": float("nan"),
            "l_max": float("nan"),
            "l_scan": np.array([], dtype=float),
            "growth_proxy": np.array([], dtype=float),
            "target_l": float("nan"),
            "peak_growth_proxy": 0.0,
        }

    l_scan = np.linspace(l_min, l_max, n_scan)
    neutral_vals = np.array([neutral_curve_NL(li) for li in l_scan])
    growth_proxy = np.clip(Ra / np.maximum(neutral_vals, 1.0e-12) - 1.0, 0.0, None)
    if not np.any(growth_proxy > 0.0):
        return {
            "l_min": float(l_min),
            "l_max": float(l_max),
            "l_scan": l_scan,
            "growth_proxy": growth_proxy,
            "target_l": float(lcNL),
            "peak_growth_proxy": 0.0,
        }

    weights = growth_proxy ** spectrum_power
    target_l = float(np.sum(l_scan * weights) / np.sum(weights))
    return {
        "l_min": float(l_min),
        "l_max": float(l_max),
        "l_scan": l_scan,
        "growth_proxy": growth_proxy,
        "target_l": target_l,
        "peak_growth_proxy": float(np.max(growth_proxy)),
    }


def surface_accumulation_index(
    *,
    l: float,
    Ra: float,
    RcNL: float,
    psi_tilde_1_coeffs: np.ndarray,
    params: LCParams,
    v_float: float,
) -> dict:
    """Estimate visible surface accumulation from the CL streamfunction."""
    if Ra <= RcNL or l <= 0.0:
        return {
            "tau_acc_s": float("inf"),
            "resurfacing_time_s": float("inf"),
            "u_converge": 0.0,
            "line_density_gain": 1.0,
            "is_visible": False,
            "accumulation_factor": 0.0,
            "w_down_max": 0.0,
        }

    z_test = np.linspace(-1.0, 0.0, 256)
    psi_vals = np.abs(poly_eval(psi_tilde_1_coeffs, z_test))
    psi_max = float(np.max(psi_vals))
    w_down_max = params.nu_T / max(params.depth, 1.0e-12) * math.sqrt(max(Ra - RcNL, 0.0)) * l * psi_max

    psi_prime = poly_derivative(psi_tilde_1_coeffs)
    surface_convergence_shape = abs(poly_eval_at(psi_prime, -1.0e-4))
    u_converge = params.nu_T / max(params.depth, 1.0e-12) * math.sqrt(max(Ra - RcNL, 0.0)) * max(
        surface_convergence_shape,
        0.05,
    )

    spacing_m = 2.0 * math.pi / max(l, 1.0e-12) * params.depth
    half_cell = 0.5 * spacing_m
    tau_acc = half_cell / max(u_converge, 1.0e-12)
    resurfacing_time = params.depth / max(v_float, 1.0e-12)

    y = np.linspace(-half_cell, half_cell, 64)
    y0_std = float(np.std(y))
    dt = min(tau_acc, resurfacing_time) / 48.0
    for _ in range(48):
        y = y - dt * u_converge * np.sin(np.pi * y / max(half_cell, 1.0e-12))
        y = np.clip(y, -half_cell, half_cell)
    y_std = max(float(np.std(y)), 1.0e-9)
    line_density_gain = y0_std / y_std

    timescale_ratio = resurfacing_time / max(tau_acc, 1.0e-12)
    accumulation_factor = _clip01((line_density_gain - 1.0) / 3.0) * _clip01(timescale_ratio)
    is_visible = accumulation_factor > 0.15 and tau_acc < 4.0 * resurfacing_time

    return {
        "tau_acc_s": float(tau_acc),
        "resurfacing_time_s": float(resurfacing_time),
        "u_converge": float(u_converge),
        "line_density_gain": float(line_density_gain),
        "is_visible": bool(is_visible),
        "accumulation_factor": float(accumulation_factor),
        "w_down_max": float(w_down_max),
    }


def bloom_feedback_potential(
    accumulation_factor: float,
    surface_residence_time: float,
    light_enhancement: float = 1.5,
) -> float:
    if accumulation_factor <= 0.0 or surface_residence_time <= 0.0:
        return 0.0
    rt_factor = 1.0 - math.exp(-surface_residence_time / 3600.0)
    feedback = accumulation_factor * rt_factor * (light_enhancement - 1.0)
    return float(min(max(feedback, 0.0), 1.0))


def advance_langmuir_state(
    params: LCParams,
    *,
    profile_name: str = "uniform",
    use_lake_profile: bool = True,
    previous_state: LangmuirModeState | None = None,
    dt_hours: float | None = None,
    relaxation_hours: float = 12.0,
    decay_hours: float = 18.0,
    forcing_coherence: float = 1.0,
    dynamics: LangmuirDynamicsConfig | None = None,
    shortwave_radiation: float | None = None,
    temperature_c: float | None = None,
) -> dict:
    """Advance the Langmuir mode through one forcing step."""
    del relaxation_hours, decay_hours
    dynamics = dynamics or DEFAULT_DYNAMICS
    previous_state = previous_state or LangmuirModeState()
    dt_eff = max(float(dt_hours or 1.0), 1.0e-6)

    buoyancy_Q, rho_cell, v_float = _advance_buoyancy(
        previous_state,
        params,
        dt_hours=dt_eff,
        shortwave_radiation=shortwave_radiation,
        temperature_c=temperature_c,
    )

    fallback_triggered = False
    fallback_reason = ""
    physics_status = "complete"

    if use_lake_profile:
        try:
            profile = shallow_lake_profile(params)
        except Exception as exc:
            LOGGER.warning("Shallow-lake profile fit failed; falling back to %s: %s", profile_name, exc)
            fallback_triggered = True
            fallback_reason = f"profile_fallback:{exc}"
            physics_status = "profile_fallback"
            profile = get_profile(profile_name)
    else:
        profile = get_profile(profile_name)

    bcs = RobinBoundaryConditions(params.gamma_s, params.gamma_b)
    try:
        nl_result = solve_nonlinear(profile, bcs, max_order=8)
    except SubcriticalBifurcationError as exc:
        LOGGER.warning("Subcritical safeguard triggered: %s", exc)
        mode_state = LangmuirModeState(
            rho_cell=rho_cell,
            v_float=v_float,
            buoyancy_Q=buoyancy_Q,
        )
        return {
            "spacing_nonlinear": float("nan"),
            "spacing_cl": float("nan"),
            "spacing_response": float("nan"),
            "spacing_visible": float("nan"),
            "spacing_linear": float("nan"),
            "selected_l": float("nan"),
            "target_l": float("nan"),
            "response_target_l": float("nan"),
            "selected_l_cl": float("nan"),
            "target_l_cl": float("nan"),
            "unstable_l_min": float("nan"),
            "unstable_l_max": float("nan"),
            "peak_growth_proxy": 0.0,
            "amplitude_index": 0.0,
            "development_index": 0.0,
            "setup_index": 0.0,
            "coherent_run_hours": 0.0,
            "response_bandwidth": 0.0,
            "response_mix": 0.0,
            "large_scale_fraction": 0.0,
            "mode_state": mode_state,
            "kappa": float("nan"),
            "wavenumber_ratio": float("nan"),
            "regime": "subcritical",
            "hydrodynamic_regime": "subcritical",
            "is_visible": False,
            "accumulation_factor": 0.0,
            "bloom_feedback": 0.0,
            "w_down_max": 0.0,
            "tau_acc_s": float("inf"),
            "resurfacing_time_s": params.depth / max(v_float, 1.0e-12),
            "u_converge": 0.0,
            "line_density_gain": 1.0,
            "Ra": params.Ra,
            "R0": float("nan"),
            "RcNL": float("nan"),
            "lcNL": float("nan"),
            "lcL": float("nan"),
            "aspect_ratio": float("nan"),
            "rho_cell": rho_cell,
            "v_float": v_float,
            "buoyancy_Q": buoyancy_Q,
            "fallback_triggered": True,
            "fallback_reason": str(exc),
            "physics_status": "non_subcritical",
            "error": str(exc),
        }
    except Exception as exc:
        LOGGER.warning("Nonlinear solve failed: %s", exc)
        mode_state = LangmuirModeState(
            rho_cell=rho_cell,
            v_float=v_float,
            buoyancy_Q=buoyancy_Q,
        )
        return {
            "spacing_nonlinear": float("nan"),
            "spacing_cl": float("nan"),
            "spacing_response": float("nan"),
            "spacing_visible": float("nan"),
            "spacing_linear": float("nan"),
            "selected_l": float("nan"),
            "target_l": float("nan"),
            "response_target_l": float("nan"),
            "selected_l_cl": float("nan"),
            "target_l_cl": float("nan"),
            "unstable_l_min": float("nan"),
            "unstable_l_max": float("nan"),
            "peak_growth_proxy": 0.0,
            "amplitude_index": 0.0,
            "development_index": 0.0,
            "setup_index": 0.0,
            "coherent_run_hours": 0.0,
            "response_bandwidth": 0.0,
            "response_mix": 0.0,
            "large_scale_fraction": 0.0,
            "mode_state": mode_state,
            "kappa": float("nan"),
            "wavenumber_ratio": float("nan"),
            "regime": "subcritical",
            "hydrodynamic_regime": "subcritical",
            "is_visible": False,
            "accumulation_factor": 0.0,
            "bloom_feedback": 0.0,
            "w_down_max": 0.0,
            "tau_acc_s": float("inf"),
            "resurfacing_time_s": params.depth / max(v_float, 1.0e-12),
            "u_converge": 0.0,
            "line_density_gain": 1.0,
            "Ra": params.Ra,
            "R0": float("nan"),
            "RcNL": float("nan"),
            "lcNL": float("nan"),
            "lcL": float("nan"),
            "aspect_ratio": float("nan"),
            "rho_cell": rho_cell,
            "v_float": v_float,
            "buoyancy_Q": buoyancy_Q,
            "fallback_triggered": True,
            "fallback_reason": str(exc),
            "physics_status": "solver_failure",
            "error": str(exc),
        }

    Ra = params.Ra
    R0 = nl_result.R0
    RcNL = nl_result.RcNL
    lcNL = nl_result.lcNL
    lcL = nl_result.linear_result.lcL
    hydrodynamic_regime = classify_regime(Ra, R0, RcNL)
    coherence = _clip01(float(forcing_coherence))

    spacing_l = _spacing_from_l(lcL, params.depth)
    relax_hours = _relaxation_hours(params, dynamics)
    decay_hours_eff = _decay_hours(params, dynamics)
    coherence_decay_hours = _coherence_decay_hours(params, dynamics)
    merge_min_age_hours = _merge_min_age_hours(params, dynamics)

    if hydrodynamic_regime == "subcritical" or lcNL <= 0.0:
        regime = "subcritical"
        alpha = 1.0 - math.exp(-dt_eff / max(decay_hours_eff, 1.0e-6))
        amplitude_index = previous_state.amplitude_index + alpha * (0.0 - previous_state.amplitude_index)
        setup_index = previous_state.setup_index + alpha * (0.0 - previous_state.setup_index)
        coherent_run_hours = previous_state.coherent_run_hours * math.exp(-dt_eff / max(coherence_decay_hours, 1.0e-6))
        visible_fraction = previous_state.visible_fraction * math.exp(-dt_eff / max(decay_hours_eff, 1.0e-6))
        coarsening_index = previous_state.coarsening_index * math.exp(
            -dt_eff / max(1.6 * decay_hours_eff, 1.0e-6)
        )
        development_index = visible_fraction * (0.25 + 0.75 * coarsening_index)
        mode_state = LangmuirModeState(
            selected_l=previous_state.selected_l,
            target_l=previous_state.target_l,
            response_target_l=previous_state.response_target_l,
            cl_selected_l=previous_state.cl_selected_l,
            cl_target_l=previous_state.cl_target_l,
            amplitude_index=float(amplitude_index),
            development_index=float(development_index),
            regime=regime,
            hydrodynamic_regime=hydrodynamic_regime,
            merging_age_hours=0.0,
            setup_index=float(setup_index),
            coherent_run_hours=float(coherent_run_hours),
            response_bandwidth=float(previous_state.response_bandwidth * math.exp(-dt_eff / max(decay_hours_eff, 1.0e-6))),
            response_mix=float(previous_state.response_mix * math.exp(-dt_eff / max(decay_hours_eff, 1.0e-6))),
            large_scale_fraction=float(previous_state.large_scale_fraction * math.exp(-dt_eff / max(decay_hours_eff, 1.0e-6))),
            visible_fraction=float(visible_fraction),
            coarsening_index=float(coarsening_index),
            buoyancy_Q=buoyancy_Q,
            rho_cell=rho_cell,
            v_float=v_float,
        )
        selected_l = float(mode_state.selected_l) if mode_state.selected_l is not None else float("nan")
        target_l = float(mode_state.target_l) if mode_state.target_l is not None else float("nan")
        response_target_l = (
            float(mode_state.response_target_l)
            if mode_state.response_target_l is not None
            else float("nan")
        )
        selected_l_cl = (
            float(mode_state.cl_selected_l)
            if mode_state.cl_selected_l is not None
            else float("nan")
        )
        target_l_cl = float(mode_state.cl_target_l) if mode_state.cl_target_l is not None else float("nan")
        spacing_cl = _spacing_from_l(selected_l_cl, params.depth)
        spacing_response = _spacing_from_l(response_target_l, params.depth)
        core_spacing = _spacing_from_l(selected_l, params.depth)
        spacing_visible = core_spacing
        spacing_observable = float(mode_state.visible_fraction * core_spacing) if math.isfinite(core_spacing) else 0.0
        spacing_nl = spacing_visible
        response_bandwidth = float(mode_state.response_bandwidth)
        response_mix = float(mode_state.response_mix)
        large_scale_fraction = float(mode_state.large_scale_fraction)
        spectrum = {"l_min": float("nan"), "l_max": float("nan"), "peak_growth_proxy": 0.0}
    else:
        spectrum = supercritical_mode_spectrum(Ra, nl_result.neutral_curve_NL, lcNL, RcNL)
        target_l_cl = spectrum["target_l"]
        supercriticality = max((Ra - RcNL) / max(RcNL, 1.0e-12), 0.0)
        run_drive = coherent_run_drive(
            params,
            supercriticality=supercriticality,
            coherence=coherence,
            dynamics=dynamics,
        )
        max_coherent_hours = 4.0 * relax_hours
        coherent_run_hours = (
            previous_state.coherent_run_hours + dt_eff * run_drive
            if run_drive > 0.0
            else previous_state.coherent_run_hours * math.exp(-dt_eff / max(coherence_decay_hours, 1.0e-6))
        )
        coherent_run_hours = min(coherent_run_hours, max_coherent_hours)
        setup_index = _clip01(1.0 - math.exp(-coherent_run_hours / max(relax_hours, 1.0e-6)))
        persistent_drive = 1.0 - math.exp(-coherent_run_hours / max(merge_min_age_hours, 1.0e-6))

        onset_l = onset_mode_wavenumber(target_l_cl, lcL, spectrum["l_min"], spectrum["l_max"])
        cl_response_l = target_l_cl + (onset_l - target_l_cl) * (1.0 - setup_index)
        activity_target = run_drive * (0.35 + 0.65 * setup_index)

        alpha_amp = 1.0 - math.exp(-dt_eff / max(relax_hours / max(0.5 + activity_target, 1.0e-6), 1.0e-6))
        amplitude_index = previous_state.amplitude_index + alpha_amp * (activity_target - previous_state.amplitude_index)

        if previous_state.cl_selected_l is None or math.isnan(previous_state.cl_selected_l):
            selected_l_cl = cl_response_l
            cl_merging_age_hours = 0.0
        else:
            selected_l_cl = previous_state.cl_selected_l + (
                1.0 - math.exp(-dt_eff / max(relax_hours, 1.0e-6))
            ) * (cl_response_l - previous_state.cl_selected_l)
            cl_merging_age_hours = previous_state.merging_age_hours + dt_eff
            if (
                supercriticality > dynamics.merge_supercriticality_threshold
                and coherent_run_hours >= merge_min_age_hours
                and cl_merging_age_hours >= merge_min_age_hours
                and selected_l_cl > 0.0
            ):
                subharmonic_l = 0.5 * selected_l_cl
                if subharmonic_l > 0.05 and Ra > nl_result.neutral_curve_NL(subharmonic_l):
                    selected_l_cl = max(selected_l_cl * dynamics.merge_step_factor, subharmonic_l)
                    cl_merging_age_hours = 0.0

        hybrid = build_hybrid_spacing_spectrum(
            params=params,
            nl_result=nl_result,
            coherence=coherence,
            setup_index=setup_index,
            development_index=previous_state.development_index,
            coherent_run_hours=coherent_run_hours,
            v_float=v_float,
        )

        response_target_l = hybrid.response_target_l
        target_l = hybrid.visible_target_l
        response_bandwidth = hybrid.response_bandwidth
        response_mix = hybrid.response_mix
        large_scale_fraction = hybrid.large_scale_fraction

        spacing_cl = _spacing_from_l(selected_l_cl, params.depth)
        spacing_response = _spacing_from_l(response_target_l, params.depth)
        if not math.isfinite(spacing_cl):
            spacing_cl = _spacing_from_l(cl_response_l, params.depth)
        if not math.isfinite(spacing_response):
            spacing_response = spacing_cl

        supercritical_drive = supercriticality / (supercriticality + 0.45)
        coarsening_target = _clip01(
            (
                0.10
                + 0.30 * response_mix
                + 0.20 * large_scale_fraction
                + 0.10 * persistent_drive
                + 0.30 * supercritical_drive
            )
            * (0.25 + 0.75 * setup_index)
        )
        coarsen_hours = _coarsening_hours(
            params,
            dynamics,
            response_mix=response_mix,
            persistent_drive=persistent_drive,
        )
        alpha_coarsen = 1.0 - math.exp(-dt_eff / max(coarsen_hours, 1.0e-6))
        coarsening_index = previous_state.coarsening_index + alpha_coarsen * (
            coarsening_target - previous_state.coarsening_index
        )
        if supercriticality < 0.3:
            onset_decay = (1.0 - supercriticality / 0.3) * 0.15
            coarsening_index *= (1.0 - onset_decay * min(dt_eff, 1.0))
        coarsening_index = _clip01(coarsening_index)

        core_spacing_target = spacing_cl + coarsening_index * max(spacing_response - spacing_cl, 0.0)
        spectrum_spacing = _spacing_from_l(hybrid.visible_target_l, params.depth)
        if math.isfinite(spectrum_spacing) and spectrum_spacing > 0.0:
            spectrum_weight = 0.35 * response_mix * setup_index
            blended_spacing = (1.0 - spectrum_weight) * core_spacing_target + spectrum_weight * spectrum_spacing
        else:
            blended_spacing = core_spacing_target
        target_l = _l_from_spacing(blended_spacing, params.depth)

        visible_relax_hours = max(0.65 * coarsen_hours, 0.20)
        alpha_visible_scale = 1.0 - math.exp(-dt_eff / max(visible_relax_hours, 1.0e-6))
        if previous_state.selected_l is None or math.isnan(previous_state.selected_l):
            selected_l = selected_l_cl
        else:
            selected_l = previous_state.selected_l + alpha_visible_scale * (target_l - previous_state.selected_l)

        core_spacing = _spacing_from_l(selected_l, params.depth)
        accum_probe = surface_accumulation_index(
            l=selected_l,
            Ra=Ra,
            RcNL=RcNL,
            psi_tilde_1_coeffs=nl_result.linear_result.psi_tilde_1_coeffs,
            params=params,
            v_float=v_float,
        )
        visibility_target = _clip01(
            activity_target
            * (0.65 + 0.35 * coherence)
            * (0.75 + 0.25 * max(accum_probe["accumulation_factor"], coarsening_index))
        )
        formation_hours = _formation_hours(
            params,
            dynamics,
            supercriticality=supercriticality,
            coherence=coherence,
        )
        alpha_form = 1.0 - math.exp(-dt_eff / max(formation_hours, 1.0e-6))
        visible_fraction = previous_state.visible_fraction + alpha_form * (
            visibility_target - previous_state.visible_fraction
        )
        visible_fraction = _clip01(visible_fraction)

        development_target = visible_fraction * (0.25 + 0.75 * coarsening_index)
        alpha_dev = 1.0 - math.exp(-dt_eff / max(0.75 * relax_hours, 1.0e-6))
        development_index = previous_state.development_index + alpha_dev * (
            development_target - previous_state.development_index
        )

        regime = "near_onset" if setup_index < 0.35 or coherent_run_hours < 0.5 * relax_hours else "supercritical"
        mode_state = LangmuirModeState(
            selected_l=float(selected_l),
            target_l=float(target_l),
            response_target_l=float(response_target_l),
            cl_selected_l=float(selected_l_cl),
            cl_target_l=float(target_l_cl),
            amplitude_index=float(amplitude_index),
            development_index=float(development_index),
            regime=regime,
            hydrodynamic_regime=hydrodynamic_regime,
            merging_age_hours=float(cl_merging_age_hours),
            setup_index=float(setup_index),
            coherent_run_hours=float(coherent_run_hours),
            response_bandwidth=float(response_bandwidth),
            response_mix=float(response_mix),
            large_scale_fraction=float(large_scale_fraction),
            visible_fraction=float(visible_fraction),
            coarsening_index=float(coarsening_index),
            buoyancy_Q=buoyancy_Q,
            rho_cell=rho_cell,
            v_float=v_float,
        )
        spacing_visible = core_spacing
        spacing_observable = visible_fraction * core_spacing
        spacing_nl = spacing_visible

    accum_l = (
        mode_state.selected_l
        if mode_state.selected_l is not None and not math.isnan(mode_state.selected_l)
        else (lcNL if lcNL > 0 else 0.1)
    )
    accum = surface_accumulation_index(
        l=accum_l,
        Ra=Ra,
        RcNL=RcNL,
        psi_tilde_1_coeffs=nl_result.linear_result.psi_tilde_1_coeffs,
        params=params,
        v_float=v_float,
    )
    effective_accumulation = accum["accumulation_factor"] * (0.35 + 0.65 * mode_state.visible_fraction)
    feedback = bloom_feedback_potential(effective_accumulation, min(accum["resurfacing_time_s"], 1.0e6))
    is_visible = bool(mode_state.visible_fraction > 0.15 or accum["is_visible"])

    return {
        "spacing_nonlinear": spacing_nl,
        "spacing_cl": spacing_cl,
        "spacing_response": spacing_response,
        "spacing_visible": spacing_visible,
        "spacing_observable": spacing_observable,
        "spacing_linear": spacing_l,
        "selected_l": float(mode_state.selected_l) if mode_state.selected_l is not None else float("nan"),
        "target_l": float(mode_state.target_l) if mode_state.target_l is not None else float("nan"),
        "response_target_l": float(mode_state.response_target_l)
        if mode_state.response_target_l is not None
        else float("nan"),
        "selected_l_cl": float(mode_state.cl_selected_l)
        if mode_state.cl_selected_l is not None
        else float("nan"),
        "target_l_cl": float(mode_state.cl_target_l)
        if mode_state.cl_target_l is not None
        else float("nan"),
        "unstable_l_min": float(spectrum["l_min"]),
        "unstable_l_max": float(spectrum["l_max"]),
        "peak_growth_proxy": float(spectrum["peak_growth_proxy"]),
        "amplitude_index": float(mode_state.amplitude_index),
        "development_index": float(mode_state.development_index),
        "setup_index": float(mode_state.setup_index),
        "coherent_run_hours": float(mode_state.coherent_run_hours),
        "response_bandwidth": float(mode_state.response_bandwidth),
        "response_mix": float(mode_state.response_mix),
        "large_scale_fraction": float(mode_state.large_scale_fraction),
        "visible_fraction": float(mode_state.visible_fraction),
        "coarsening_index": float(mode_state.coarsening_index),
        "mode_state": mode_state,
        "kappa": float(nl_result.kappa),
        "wavenumber_ratio": float(nl_result.wavenumber_ratio),
        "regime": regime,
        "hydrodynamic_regime": hydrodynamic_regime,
        "is_visible": is_visible,
        "accumulation_factor": effective_accumulation,
        "bloom_feedback": feedback,
        "w_down_max": accum["w_down_max"],
        "tau_acc_s": accum["tau_acc_s"],
        "resurfacing_time_s": accum["resurfacing_time_s"],
        "u_converge": accum["u_converge"],
        "line_density_gain": accum["line_density_gain"],
        "Ra": Ra,
        "R0": R0,
        "RcNL": RcNL,
        "lcNL": lcNL,
        "lcL": lcL,
        "aspect_ratio": nl_result.aspect_ratio,
        "rho_cell": rho_cell,
        "v_float": v_float,
        "buoyancy_Q": buoyancy_Q,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "physics_status": physics_status,
    }


def predict_spacing_evolution(
    params_series: list[LCParams],
    *,
    times: list | None = None,
    profile_name: str = "uniform",
    use_lake_profile: bool = True,
    initial_state: LangmuirModeState | None = None,
    default_dt_hours: float = 1.0,
    relaxation_hours: float = 12.0,
    decay_hours: float = 18.0,
    forcing_coherence: list[float] | None = None,
    dynamics: LangmuirDynamicsConfig | None = None,
) -> list[dict]:
    rows: list[dict] = []
    state = initial_state

    for idx, params in enumerate(params_series):
        dt_hours = default_dt_hours
        if times is not None and idx > 0:
            delta = times[idx] - times[idx - 1]
            if hasattr(delta, "total_seconds"):
                dt_hours = max(delta.total_seconds() / 3600.0, 1.0e-6)

        result = advance_langmuir_state(
            params,
            profile_name=profile_name,
            use_lake_profile=use_lake_profile,
            previous_state=state,
            dt_hours=dt_hours,
            relaxation_hours=relaxation_hours,
            decay_hours=decay_hours,
            forcing_coherence=forcing_coherence[idx] if forcing_coherence is not None else 1.0,
            dynamics=dynamics,
        )
        row = dict(result)
        row.pop("mode_state", None)
        row["step_index"] = idx
        if times is not None:
            row["time"] = times[idx]
            row["dt_hours"] = dt_hours
        rows.append(row)
        state = result["mode_state"]

    return rows


def predict_spacing_and_visibility(
    params: LCParams,
    profile_name: str = "uniform",
    use_lake_profile: bool = True,
    dynamics: LangmuirDynamicsConfig | None = None,
) -> dict:
    result = advance_langmuir_state(
        params,
        profile_name=profile_name,
        use_lake_profile=use_lake_profile,
        dynamics=dynamics,
    )
    result.pop("mode_state", None)
    return result
