"""Colony accumulation model — biological coupling (secondary).

Buoyancy coupling is retained but demoted from primary spacing predictor
to secondary refinement. It governs bloom visibility and amplification,
not cell spacing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .params import LCParams
from .profiles import get_profile, shallow_lake_profile
from .robin_bc import RobinBoundaryConditions
from .nonlinear_solver import solve_nonlinear
from .rayleigh_mapping import classify_regime, unstable_band


@dataclass
class LangmuirModeState:
    """Finite-time mode state used to evolve spacing through changing forcing."""

    selected_l: float | None = None
    target_l: float | None = None
    response_target_l: float | None = None
    amplitude_index: float = 0.0
    development_index: float = 0.0
    regime: str = "subcritical"
    hydrodynamic_regime: str = "subcritical"
    merging_age_hours: float = 0.0  # time since last merging event or formation
    setup_index: float = 0.0
    coherent_run_hours: float = 0.0


@dataclass(frozen=True)
class LangmuirDynamicsConfig:
    """Tunable controls for finite-time Langmuir mode evolution.

    The nonlinear solver still supplies the instantaneous CL target mode, but
    the expressed mode is modulated by a duration-sensitive setup index. This
    gives a defensible pathway to tune the model against observation-time
    spacing while preserving the underlying CL physics.
    """

    relaxation_hours: float = 12.0
    decay_hours: float = 18.0
    coherent_u_star_threshold: float = 0.003
    coherent_u_star_scale: float = 0.002
    coherent_coherence_threshold: float = 0.6
    coherent_decay_hours: float = 8.0
    coherent_hours_cap: float = 72.0
    setup_hours_scale: float = 10.0
    setup_supercriticality_scale: float = 0.25
    onset_mode_blend_power: float = 1.5
    recovery_gate_floor: float = 0.18
    expression_near_onset_setup: float = 0.35
    expression_near_onset_hours: float = 4.0
    merge_amplitude_threshold: float = 0.4
    merge_setup_threshold: float = 0.55
    merge_min_age_hours: float = 3.0
    merge_timescale_hours: float = 6.0
    merge_coherent_hours_threshold: float = 10.0
    merge_u_star_threshold: float = 0.003
    merge_step_factor: float = 0.85

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_DYNAMICS = LangmuirDynamicsConfig()


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def coherent_run_drive(
    params: LCParams,
    *,
    supercriticality: float,
    coherence: float,
    dynamics: LangmuirDynamicsConfig,
) -> float:
    """Map forcing into a coherent-run growth rate in [0, 1].

    Hayes & Phillips note that visible LC typically form tens of minutes after
    winds exceed about 3 m/s, not instantly when the CL2 criterion becomes
    positive. This drive therefore requires both sustained wind stress and
    directional coherence before the duration memory can build.
    """
    if params.u_star <= dynamics.coherent_u_star_threshold:
        return 0.0
    u_drive = 1.0 - math.exp(
        -(params.u_star - dynamics.coherent_u_star_threshold) / max(dynamics.coherent_u_star_scale, 1e-6)
    )
    sc_drive = supercriticality / (dynamics.setup_supercriticality_scale + supercriticality)
    if coherence <= dynamics.coherent_coherence_threshold:
        coherence_drive = 0.0
    else:
        coherence_drive = (
            (coherence - dynamics.coherent_coherence_threshold)
            / max(1.0 - dynamics.coherent_coherence_threshold, 1e-6)
        )
    return _clip01(u_drive * sc_drive * coherence_drive)


def onset_mode_wavenumber(
    target_l: float,
    lcL: float,
    l_min: float,
    l_max: float,
) -> float:
    """Return a physically plausible onset/immature wavenumber.

    Early-time LC expression is expected to be closer to linear onset than to
    the fully nonlinear large-cell state. We therefore anchor immature states
    to the linear scale, clipped into the instantaneous unstable band.
    """
    onset_l = lcL if lcL > 0 else target_l
    if math.isnan(l_min) or math.isnan(l_max):
        return float(max(onset_l, target_l))
    return float(min(max(onset_l, l_min), l_max))


def supercritical_mode_spectrum(
    Ra: float,
    neutral_curve_NL,
    lcNL: float,
    RcNL: float = 0.0,
    *,
    n_scan: int = 256,
    spectrum_power: float = 2.0,
) -> dict:
    """Build a finite-width supercritical spectrum and derive a target mode.

    The growth-rate weighting is asymmetric: at high supercriticality, larger
    cells (smaller l) are favoured because the deeper circulation interacts
    more with the full-depth shear profile.  This shifts ``target_l`` toward
    smaller values (larger spacing) for stronger forcing, which is the
    observed physical behaviour.
    """
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
    growth_proxy = np.clip(Ra / np.maximum(neutral_vals, 1e-12) - 1.0, 0.0, None)
    if not np.any(growth_proxy > 0.0):
        target_l = lcNL
        peak_growth_proxy = 0.0
    else:
        # Asymmetric weighting: at high supercriticality, favour larger cells
        # (smaller l).  The physical basis is that the nonlinear growth rate
        # for deeply penetrating modes scales more favourably with Ra because
        # they interact with more of the shear profile.
        supercriticality = (Ra - RcNL) / max(RcNL, 1e-12) if RcNL > 0 else 0.0
        # asymmetry_strength grows from 0 (near onset) toward 1 (highly supercritical)
        asymmetry_strength = supercriticality / (1.0 + supercriticality)
        # Bias factor: linearly increases for smaller l relative to lcNL
        # At l = lcNL the factor is 1; at l < lcNL (larger cells) it's > 1
        bias = 1.0 + asymmetry_strength * (lcNL / np.maximum(l_scan, 1e-12) - 1.0)
        bias = np.clip(bias, 0.2, 5.0)

        weights = (growth_proxy ** spectrum_power) * bias
        target_l = float(np.sum(l_scan * weights) / np.sum(weights))
        peak_growth_proxy = float(np.max(growth_proxy))

    return {
        "l_min": float(l_min),
        "l_max": float(l_max),
        "l_scan": l_scan,
        "growth_proxy": growth_proxy,
        "target_l": float(target_l),
        "peak_growth_proxy": peak_growth_proxy,
    }


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
) -> dict:
    """Advance the Langmuir mode through one forcing step.

    ``development_index`` is a simple finite-time proxy for how fully the
    Langmuir pattern has adjusted to the current forcing. ``selected_l`` relaxes
    toward the instantaneous supercritical target mode on a configurable
    timescale, which enables future observation-centred 48 h analyses.
    """
    dynamics = dynamics or LangmuirDynamicsConfig(
        relaxation_hours=relaxation_hours,
        decay_hours=decay_hours,
    )

    if use_lake_profile:
        try:
            profile = shallow_lake_profile(params)
        except Exception:
            profile = get_profile(profile_name)
    else:
        profile = get_profile(profile_name)

    bcs = RobinBoundaryConditions(params.gamma_s, params.gamma_b)
    try:
        nl_result = solve_nonlinear(profile, bcs, max_order=8)
    except Exception as exc:
        return {
            "spacing_nonlinear": float("nan"),
            "spacing_linear": float("nan"),
            "selected_l": float("nan"),
            "target_l": float("nan"),
            "response_target_l": float("nan"),
            "unstable_l_min": float("nan"),
            "unstable_l_max": float("nan"),
            "peak_growth_proxy": 0.0,
            "amplitude_index": 0.0,
            "development_index": 0.0,
            "setup_index": 0.0,
            "mode_state": LangmuirModeState(),
            "kappa": float("nan"),
            "regime": "subcritical",
            "is_visible": False,
            "accumulation_factor": 0.0,
            "bloom_feedback": 0.0,
            "w_down_max": 0.0,
            "Ra": params.Ra,
            "error": str(exc),
        }

    Ra = params.Ra
    R0 = nl_result.R0
    RcNL = nl_result.RcNL
    lcNL = nl_result.lcNL
    lcL = nl_result.linear_result.lcL
    hydrodynamic_regime = classify_regime(Ra, R0, RcNL)

    if previous_state is None:
        previous_state = LangmuirModeState()
    coherence = min(max(float(forcing_coherence), 0.0), 1.0)

    spacing_l = 2.0 * math.pi / lcL * params.depth if lcL > 0 else float("nan")

    if hydrodynamic_regime == "subcritical" or lcNL <= 0.0:
        regime = "subcritical"
        dt_eff = max(float(dt_hours or dynamics.decay_hours), 1e-6)
        alpha_amp = 1.0 - math.exp(-dt_eff / max(dynamics.decay_hours, 1e-6))
        amplitude_index = previous_state.amplitude_index + alpha_amp * (0.0 - previous_state.amplitude_index)
        development_index = previous_state.development_index + alpha_amp * (
            0.0 - previous_state.development_index
        )
        setup_index = previous_state.setup_index + alpha_amp * (0.0 - previous_state.setup_index)
        coherent_run_hours = previous_state.coherent_run_hours * math.exp(
            -dt_eff / max(dynamics.coherent_decay_hours, 1e-6)
        )
        mode_state = LangmuirModeState(
            selected_l=previous_state.selected_l,
            target_l=float("nan"),
            response_target_l=float("nan"),
            amplitude_index=float(amplitude_index),
            development_index=float(development_index),
            regime=regime,
            hydrodynamic_regime=hydrodynamic_regime,
            merging_age_hours=0.0,
            setup_index=float(setup_index),
            coherent_run_hours=float(coherent_run_hours),
        )
        spacing_nl = float("nan")
        selected_l = float("nan")
        target_l = float("nan")
        response_target_l = float("nan")
        spectrum = {
            "l_min": float("nan"),
            "l_max": float("nan"),
            "peak_growth_proxy": 0.0,
        }
    else:
        spectrum = supercritical_mode_spectrum(Ra, nl_result.neutral_curve_NL, lcNL, RcNL)
        target_l = spectrum["target_l"]
        peak_growth_proxy = spectrum["peak_growth_proxy"]
        supercriticality = max((Ra - RcNL) / max(RcNL, 1e-12), 0.0)
        if dt_hours is None:
            coherent_run_hours = dynamics.coherent_hours_cap
            setup_index = 1.0
            run_drive = 1.0
        else:
            run_drive = coherent_run_drive(
                params,
                supercriticality=supercriticality,
                coherence=coherence,
                dynamics=dynamics,
            )
            dt_eff = max(float(dt_hours), 1e-6)
            if run_drive > 0.0:
                coherent_run_hours = min(
                    previous_state.coherent_run_hours + dt_eff * run_drive,
                    dynamics.coherent_hours_cap,
                )
            else:
                coherent_run_hours = previous_state.coherent_run_hours * math.exp(
                    -dt_eff / max(dynamics.coherent_decay_hours, 1e-6)
                )
            setup_index = _clip01(
                1.0 - math.exp(-coherent_run_hours / max(dynamics.setup_hours_scale, 1e-6))
            )

        onset_l = onset_mode_wavenumber(
            target_l,
            lcL,
            spectrum["l_min"],
            spectrum["l_max"],
        )
        response_target_l = target_l + (onset_l - target_l) * (1.0 - setup_index) ** dynamics.onset_mode_blend_power
        activity_target = run_drive * (0.35 + 0.65 * setup_index)

        if dt_hours is None:
            amplitude_index = activity_target
        else:
            dt_eff = max(float(dt_hours), 1e-6)
            amp_relax_hours = dynamics.relaxation_hours / max(0.5 + 1.5 * max(activity_target, 0.0), 1e-6)
            alpha_amp = 1.0 - math.exp(-dt_eff / max(amp_relax_hours, 1e-6))
            amplitude_index = previous_state.amplitude_index + alpha_amp * (
                activity_target - previous_state.amplitude_index
            )

        if dt_hours is None or previous_state.selected_l is None or math.isnan(previous_state.selected_l):
            selected_l = response_target_l
            merging_age_hours = 0.0
        else:
            dt_eff = max(float(dt_hours), 1e-6)
            merging_age_hours = previous_state.merging_age_hours + dt_eff

            # --- Relaxation toward target_l ---
            # Asymmetric relaxation: cells readily grow toward larger spacing
            # when forcing favours it, but only recover partway back toward
            # smaller spacing between discrete merge events. A non-zero
            # restoring gate avoids the previous failure mode where a long
            # spinup could merge the pattern once and then freeze it forever.
            growth_accel = 0.5 + 2.5 * max(amplitude_index, 0.0)
            alpha = 1.0 - math.exp(-dt_eff * growth_accel / max(dynamics.relaxation_hours, 1e-6))

            if response_target_l <= previous_state.selected_l:
                # Target favours larger cells — relax freely toward it
                selected_l = previous_state.selected_l + alpha * (
                    response_target_l - previous_state.selected_l
                )
            else:
                # Target favours smaller cells (higher l). Keep hysteresis, but
                # always allow some restoring force so merged states can relax
                # back between discrete merge events.
                shrink_gate = max(1.0 - amplitude_index / 0.5, 0.0)
                shrink_gate = max(shrink_gate, dynamics.recovery_gate_floor)
                selected_l = previous_state.selected_l + alpha * shrink_gate * (
                    response_target_l - previous_state.selected_l
                )

            # --- Cell merging (subharmonic instability) ---
            # Well-developed cells undergo merging when:
            #   1. Amplitude is high enough (cells are coherent)
            #   2. Sufficient time has passed since formation/last merge
            #   3. The subharmonic mode (l/2) is also unstable
            #   4. Wind friction velocity is strong enough to consolidate cells
            if (
                amplitude_index > dynamics.merge_amplitude_threshold
                and setup_index > dynamics.merge_setup_threshold
                and coherent_run_hours > dynamics.merge_coherent_hours_threshold
                and merging_age_hours > dynamics.merge_min_age_hours
                and selected_l > 0
                and params.u_star > dynamics.merge_u_star_threshold
            ):
                subharmonic_l = selected_l / 2.0
                if subharmonic_l > 0.05:
                    R_bar_sub = nl_result.neutral_curve_NL(subharmonic_l)
                    if Ra > R_bar_sub:
                        merge_strength = (
                            amplitude_index
                            * setup_index
                            * coherence
                            * min(coherent_run_hours / max(dynamics.merge_coherent_hours_threshold, 1e-6), 2.0)
                        )
                        effective_merge_time = dynamics.merge_timescale_hours / max(
                            0.2 + merge_strength, 1e-6
                        )
                        if merging_age_hours >= max(dynamics.merge_min_age_hours, effective_merge_time):
                            selected_l = max(selected_l * dynamics.merge_step_factor, subharmonic_l)
                            merging_age_hours = 0.0

        # Mode mismatch penalises organisation only when selected_l EXCEEDS
        # response_target_l (cells too small for the forcing).  When
        # selected_l < response_target_l the cells have merged to a larger,
        # physically valid scale.
        if selected_l > response_target_l:
            mode_mismatch = (selected_l - response_target_l) / max(abs(response_target_l), 1e-6)
        else:
            mode_mismatch = 0.0
        organization_target = amplitude_index * (0.25 + 0.75 * setup_index) * math.exp(-mode_mismatch / 0.15)
        if dt_hours is None:
            development_index = organization_target
        else:
            alpha_dev = 1.0 - math.exp(-dt_eff / max(0.75 * dynamics.relaxation_hours, 1e-6))
            development_index = previous_state.development_index + alpha_dev * (
                organization_target - previous_state.development_index
            )

        if (
            setup_index < dynamics.expression_near_onset_setup
            or coherent_run_hours < dynamics.expression_near_onset_hours
        ):
            regime = "near_onset"
        else:
            regime = "supercritical"

        mode_state = LangmuirModeState(
            selected_l=float(selected_l),
            target_l=float(target_l),
            response_target_l=float(response_target_l),
            amplitude_index=float(amplitude_index),
            development_index=float(development_index),
            regime=regime,
            hydrodynamic_regime=hydrodynamic_regime,
            merging_age_hours=float(merging_age_hours),
            setup_index=float(setup_index),
            coherent_run_hours=float(coherent_run_hours),
        )
        spacing_nl = 2.0 * math.pi / selected_l * params.depth if selected_l > 0 else float("nan")

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
        v_float=params.v_float,
        depth=params.depth,
        u_star=params.u_star,
    )

    if accum["w_down_max"] > 0:
        residence_time = params.depth / max(accum["w_down_max"], 1e-10)
    else:
        residence_time = float("inf")

    feedback = bloom_feedback_potential(
        accum["accumulation_factor"],
        min(residence_time, 1e6),
    )

    return {
        "spacing_nonlinear": spacing_nl,
        "spacing_linear": spacing_l,
        "selected_l": float(mode_state.selected_l) if mode_state.selected_l is not None else float("nan"),
        "target_l": float(mode_state.target_l) if mode_state.target_l is not None else float("nan"),
        "response_target_l": (
            float(mode_state.response_target_l) if mode_state.response_target_l is not None else float("nan")
        ),
        "unstable_l_min": float(spectrum["l_min"]),
        "unstable_l_max": float(spectrum["l_max"]),
        "peak_growth_proxy": float(spectrum["peak_growth_proxy"]),
        "amplitude_index": float(mode_state.amplitude_index),
        "development_index": float(mode_state.development_index),
        "setup_index": float(mode_state.setup_index),
        "coherent_run_hours": float(mode_state.coherent_run_hours),
        "mode_state": mode_state,
        "kappa": nl_result.kappa,
        "regime": regime,
        "hydrodynamic_regime": hydrodynamic_regime,
        "is_visible": accum["is_visible"],
        "accumulation_factor": accum["accumulation_factor"],
        "bloom_feedback": feedback,
        "w_down_max": accum["w_down_max"],
        "Ra": Ra,
        "R0": R0,
        "RcNL": RcNL,
        "lcNL": lcNL,
        "lcL": lcL,
        "aspect_ratio": nl_result.aspect_ratio,
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
    """Predict Langmuir spacing and development through a forcing time series.

    This is intended for observation-centred windows such as 48 h before/after
    an image time. Pass hourly or coarser LCParams snapshots plus optional
    matching timestamps. Each returned row includes the evolving mode state.
    """
    rows: list[dict] = []
    state = initial_state

    for idx, params in enumerate(params_series):
        dt_hours = default_dt_hours
        if times is not None and idx > 0:
            delta = times[idx] - times[idx - 1]
            if hasattr(delta, "total_seconds"):
                dt_hours = max(delta.total_seconds() / 3600.0, 1e-6)

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


def surface_accumulation_index(
    l: float,
    Ra: float,
    RcNL: float,
    psi_tilde_1_coeffs: np.ndarray,
    v_float: float,
    depth: float,
    u_star: float,
) -> dict:
    """Compute whether LC at wavenumber l produces visible surface accumulation.

    Uses the nonlinear eigenfunction psi(z) to estimate downwelling velocity.
    The key ratio S = w_down_max / v_float determines visibility.
    """
    from .utils import poly_eval_at, poly_derivative

    # Downwelling velocity scales as (Ra - RcNL)^{1/2} for weakly supercritical
    if Ra <= RcNL:
        return {"ratio_S": 0.0, "is_visible": False, "accumulation_factor": 0.0,
                "w_down_max": 0.0}

    # Estimate |d psi / dy|_max from psi_tilde_1
    # psi ~ R0 * psi_tilde_1(z) * sin(ly), so d psi/dy = l * R0 * psi_tilde_1(z) * cos(ly)
    # max over z of |psi_tilde_1|
    z_test = np.linspace(-1, 0, 200)
    from .utils import poly_eval
    psi_vals = np.abs(poly_eval(psi_tilde_1_coeffs, z_test))
    psi_max = float(np.max(psi_vals))

    # Dimensional downwelling: w ~ nu_T / h * (Ra - RcNL)^{1/2} * l * psi_max
    nu_T = 0.41 * u_star * depth / 6.0
    w_down_max = nu_T / depth * math.sqrt(max(Ra - RcNL, 0)) * l * psi_max

    ratio_S = w_down_max / max(v_float, 1e-10)

    # Visibility: colonies herded when S ~ 0.3-1.0
    is_visible = 0.1 < ratio_S < 3.0

    # Accumulation factor: peaked at S ~ 0.5
    if ratio_S <= 0:
        accumulation_factor = 0.0
    else:
        accumulation_factor = float(2.0 * ratio_S * math.exp(-ratio_S))

    return {
        "ratio_S": float(ratio_S),
        "is_visible": is_visible,
        "accumulation_factor": float(accumulation_factor),
        "w_down_max": float(w_down_max),
    }


def bloom_feedback_potential(
    accumulation_factor: float,
    surface_residence_time: float,
    light_enhancement: float = 1.5,
) -> float:
    """Estimate positive feedback strength for bloom development.

    Returns dimensionless feedback index in [0, 1].
    Above ~0.3 indicates favourable conditions for bloom intensification.
    """
    if accumulation_factor <= 0 or surface_residence_time <= 0:
        return 0.0

    # Feedback = accumulation * min(residence_time_factor, 1) * light_factor
    # residence_time_factor saturates at long times
    rt_factor = 1.0 - math.exp(-surface_residence_time / 3600.0)  # e-folding at 1 hour
    feedback = accumulation_factor * rt_factor * (light_enhancement - 1.0)
    return float(min(max(feedback, 0.0), 1.0))


def predict_spacing_and_visibility(params: LCParams,
                                    profile_name: str = "uniform",
                                    use_lake_profile: bool = True,
                                    dynamics: LangmuirDynamicsConfig | None = None) -> dict:
    """Full prediction pipeline.

    1. Compute Ra from wind forcing
    2. Solve nonlinear CL-equations
    3. Find the fastest-growing mode in the unstable band
    4. Assess surface visibility
    5. Compute bloom feedback potential

    Returns dict with spacing predictions, regime info, and biological diagnostics.
    """
    result = advance_langmuir_state(
        params,
        profile_name=profile_name,
        use_lake_profile=use_lake_profile,
        dynamics=dynamics,
    )
    result.pop("mode_state", None)
    return result
