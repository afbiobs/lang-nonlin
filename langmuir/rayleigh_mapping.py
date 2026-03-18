"""Wind speed to Rayleigh number mapping for shallow lakes.

Maps environmental forcing (U10, h, fetch) to Ra and determines
the instability regime.
"""

from __future__ import annotations

import math

import numpy as np

from .params import LCParams


def wind_to_rayleigh(U10: float, depth: float, fetch: float,
                     drag_model: str = "lake_low") -> dict:
    """Compute Ra and all intermediate quantities from wind forcing.

    Returns dict with Ra, u_star, U_surface, D_max, nu_T, wave params, La_t.
    """
    params = LCParams(U10=U10, depth=depth, fetch=fetch)
    return {
        "Ra": params.Ra,
        "u_star": params.u_star,
        "U_surface": params.U_surface,
        "D_max": params.D_max,
        "nu_T": params.nu_T,
        "H_s": params.H_s,
        "T_p": params.T_p,
        "lambda_p": params.lambda_p,
        "La_t": params.La_t,
    }


def classify_regime(Ra: float, R0: float, RcNL: float) -> str:
    """Classify the LC regime based on Rayleigh number.

    Returns "subcritical", "near_onset", "moderate", or "supercritical".
    """
    if Ra < R0:
        return "subcritical"
    if Ra < 1.5 * RcNL:
        return "near_onset"
    if Ra < 5.0 * RcNL:
        return "moderate"
    return "supercritical"


def unstable_band(Ra: float, neutral_curve_NL, l_array: np.ndarray) -> tuple[float, float]:
    """Find [l_min, l_max] where Ra > R_bar(l).

    Returns (l_min, l_max). If subcritical, returns (nan, nan).
    """
    R_vals = np.array([neutral_curve_NL(li) for li in l_array])
    unstable = l_array[Ra > R_vals]

    if len(unstable) == 0:
        return (float("nan"), float("nan"))

    return (float(unstable[0]), float(unstable[-1]))


def fastest_growing_mode(Ra: float, neutral_curve_NL, l_array: np.ndarray) -> float:
    """Find the wavenumber with maximum supercriticality (Ra - R_bar(l)).

    At onset, this is lcNL. Above onset, it shifts toward higher l.
    """
    R_vals = np.array([neutral_curve_NL(li) for li in l_array])
    margin = Ra - R_vals
    unstable = margin > 0

    if not np.any(unstable):
        return float("nan")

    # Find the maximum margin
    idx = np.argmax(margin)
    return float(l_array[idx])
