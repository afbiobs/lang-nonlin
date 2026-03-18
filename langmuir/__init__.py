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
from .colony_accumulation import predict_spacing_and_visibility

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
    "predict_spacing_and_visibility",
]
