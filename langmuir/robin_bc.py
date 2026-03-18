"""Robin boundary condition implementation for the CL-equations.

Hayes & Phillips (2017) equations (2)-(3) and scaling (6).
Robin BCs are essential for selecting a finite onset wavenumber.
"""

from __future__ import annotations

import numpy as np


class RobinBoundaryConditions:
    """Robin boundary conditions for shallow-layer CL-equations.

    The paper shows Robin BCs are essential for finite onset wavenumber.
    With Neumann BCs (gamma=0), lc=0 regardless of nonlinearities
    (Chapman & Proctor 1980).
    """

    def __init__(self, gamma_s: float = 0.06, gamma_b: float = 0.28):
        self.gamma_s = gamma_s
        self.gamma_b = gamma_b
        self.gamma = gamma_s + gamma_b

    def gamma_tilde_s(self, l: float) -> float:
        """Rescaled surface parameter: gamma_tilde_s = gamma_s / l^4."""
        if abs(l) < 1e-15:
            return np.inf
        return self.gamma_s / l**4

    def gamma_tilde_b(self, l: float) -> float:
        """Rescaled bottom parameter: gamma_tilde_b = gamma_b / l^4."""
        if abs(l) < 1e-15:
            return np.inf
        return self.gamma_b / l**4

    def gamma_tilde(self, l: float) -> float:
        """Rescaled total: gamma_tilde = gamma / l^4."""
        if abs(l) < 1e-15:
            return np.inf
        return self.gamma / l**4

    def __repr__(self) -> str:
        return f"RobinBoundaryConditions(gamma_s={self.gamma_s}, gamma_b={self.gamma_b})"
