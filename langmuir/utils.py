"""Shared utilities: polynomial operations, integration helpers."""

from __future__ import annotations

import numpy as np
from numpy.polynomial import polynomial as nppoly


def poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two polynomials represented in ascending-power coefficients."""
    n = max(len(a), len(b))
    result = np.zeros(n, dtype=float)
    result[:len(a)] += np.asarray(a, dtype=float)
    result[:len(b)] += np.asarray(b, dtype=float)
    return result


def poly_integrate(coeffs: np.ndarray, n_times: int = 1) -> np.ndarray:
    """Integrate polynomial sum(c_k z^k) n_times. Constants of integration are zero."""
    c = np.array(coeffs, dtype=float)
    for _ in range(n_times):
        new = np.zeros(len(c) + 1)
        for k, ck in enumerate(c):
            new[k + 1] = ck / (k + 1)
        c = new
    return c


def poly_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two polynomials represented as coefficient arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.array([0.0])
    return np.convolve(a, b)


def poly_eval(coeffs: np.ndarray, z: np.ndarray | float) -> np.ndarray | float:
    """Evaluate polynomial sum(c_k z^k) at z."""
    z_array = np.asarray(z, dtype=float)
    result = nppoly.polyval(z_array, np.asarray(coeffs, dtype=float))
    if np.isscalar(z):
        return float(result)
    return result


def poly_eval_at(coeffs: np.ndarray, z0: float) -> float:
    """Evaluate polynomial at a single point."""
    return float(nppoly.polyval(float(z0), np.asarray(coeffs, dtype=float)))


def poly_definite_integral(coeffs: np.ndarray, a: float = -1.0, b: float = 0.0) -> float:
    """Compute definite integral of polynomial from a to b."""
    anti = poly_integrate(coeffs)
    return poly_eval_at(anti, b) - poly_eval_at(anti, a)


def poly_derivative(coeffs: np.ndarray) -> np.ndarray:
    """Differentiate polynomial."""
    if len(coeffs) <= 1:
        return np.array([0.0])
    return np.array([k * coeffs[k] for k in range(1, len(coeffs))])


def iterated_integral(f_values: np.ndarray, z_grid: np.ndarray, n_times: int) -> np.ndarray:
    """Compute n-fold cumulative trapezoidal integration on z_grid."""
    result = f_values.copy()
    for _ in range(n_times):
        result = np.cumsum(
            np.concatenate([[0.0], 0.5 * (result[:-1] + result[1:]) * np.diff(z_grid)])
        )
    return result
