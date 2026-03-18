"""
Nonlinear Craik-Leibovich Langmuir Circulation Model for Shallow Lakes.

Based on Hayes and Phillips (2017), Geophys. Astrophys. Fluid Dyn., 111(1), 65-90.
"""

from .params import LCParams
from .profiles import ShearDriftProfile, get_profile
from .robin_bc import RobinBoundaryConditions
from .linear_solver import LinearResult, solve_linear
from .nonlinear_solver import NonlinearResult, solve_nonlinear
from .rayleigh_mapping import wind_to_rayleigh, classify_regime
from .colony_accumulation import (
    LangmuirModeState,
    advance_langmuir_state,
    predict_spacing_and_visibility,
    predict_spacing_evolution,
    supercritical_mode_spectrum,
)
from .era5 import build_era5_url, fetch_all_observations
from .timeline_analysis import fit_interpretable_diagnostic_model, predict_observation_timeline
from .weather import extract_model_forcing, classify_wind_regime

__all__ = [
    "LCParams",
    "ShearDriftProfile",
    "get_profile",
    "RobinBoundaryConditions",
    "LinearResult",
    "solve_linear",
    "NonlinearResult",
    "solve_nonlinear",
    "wind_to_rayleigh",
    "classify_regime",
    "LangmuirModeState",
    "advance_langmuir_state",
    "predict_spacing_and_visibility",
    "predict_spacing_evolution",
    "supercritical_mode_spectrum",
    "build_era5_url",
    "fetch_all_observations",
    "predict_observation_timeline",
    "fit_interpretable_diagnostic_model",
    "extract_model_forcing",
    "classify_wind_regime",
]
