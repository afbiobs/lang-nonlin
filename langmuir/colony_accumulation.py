"""Colony accumulation model — biological coupling (secondary).

Buoyancy coupling is retained but demoted from primary spacing predictor
to secondary refinement. It governs bloom visibility and amplification,
not cell spacing.
"""

from __future__ import annotations

import math

import numpy as np

from .params import LCParams
from .profiles import ShearDriftProfile, get_profile, shallow_lake_profile
from .robin_bc import RobinBoundaryConditions
from .nonlinear_solver import NonlinearResult, solve_nonlinear
from .rayleigh_mapping import wind_to_rayleigh, classify_regime, unstable_band, fastest_growing_mode


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
                                    use_lake_profile: bool = False) -> dict:
    """Full prediction pipeline.

    1. Compute Ra from wind forcing
    2. Solve nonlinear CL-equations
    3. Find the fastest-growing mode in the unstable band
    4. Assess surface visibility
    5. Compute bloom feedback potential

    Returns dict with spacing predictions, regime info, and biological diagnostics.
    """
    # Get profile
    if use_lake_profile:
        try:
            profile = shallow_lake_profile(params)
        except Exception:
            profile = get_profile(profile_name)
    else:
        profile = get_profile(profile_name)

    bcs = RobinBoundaryConditions(params.gamma_s, params.gamma_b)

    # Solve nonlinear problem
    try:
        nl_result = solve_nonlinear(profile, bcs, max_order=8)
    except Exception as e:
        # Fallback: return subcritical result
        return {
            "spacing_nonlinear": float("nan"),
            "spacing_linear": float("nan"),
            "kappa": float("nan"),
            "regime": "subcritical",
            "is_visible": False,
            "accumulation_factor": 0.0,
            "bloom_feedback": 0.0,
            "w_down_max": 0.0,
            "Ra": params.Ra,
            "error": str(e),
        }

    Ra = params.Ra
    R0 = nl_result.R0
    RcNL = nl_result.RcNL
    lcNL = nl_result.lcNL
    lcL = nl_result.linear_result.lcL

    regime = classify_regime(Ra, R0, RcNL)

    # Dimensional spacing = 2*pi / l_c * depth
    if lcNL > 0:
        spacing_nl = 2.0 * math.pi / lcNL * params.depth
    else:
        spacing_nl = float("nan")

    if lcL > 0:
        spacing_l = 2.0 * math.pi / lcL * params.depth
    else:
        spacing_l = float("nan")

    # For supercritical regimes, find the fastest-growing mode
    if regime != "subcritical" and lcNL > 0:
        l_scan = np.linspace(0.01 * lcNL, 3.0 * lcNL, 100)
        l_fastest = fastest_growing_mode(Ra, nl_result.neutral_curve_NL, l_scan)
        if not math.isnan(l_fastest):
            spacing_nl = 2.0 * math.pi / l_fastest * params.depth
    elif regime == "subcritical":
        spacing_nl = float("nan")
        spacing_l = float("nan")

    # Surface visibility
    accum = surface_accumulation_index(
        l=lcNL if lcNL > 0 else 0.1,
        Ra=Ra,
        RcNL=RcNL,
        psi_tilde_1_coeffs=nl_result.linear_result.psi_tilde_1_coeffs,
        v_float=params.v_float,
        depth=params.depth,
        u_star=params.u_star,
    )

    # Bloom feedback
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
        "kappa": nl_result.kappa,
        "regime": regime,
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
