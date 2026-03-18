"""Galerkin spectral method infrastructure.

Shifted Legendre basis on z in [-1, 0], inner products, quadrature,
and trigonometric product rules for nonlinear terms.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre


def shifted_legendre(n: int, z: np.ndarray) -> np.ndarray:
    """Shifted Legendre polynomial P_n on [-1, 0].

    Maps z in [-1, 0] to xi in [-1, 1] via xi = 2z + 1.
    """
    xi = 2.0 * z + 1.0
    return eval_legendre(n, xi)


def shifted_legendre_derivatives(n: int, z: np.ndarray, max_deriv: int = 4) -> list[np.ndarray]:
    """Return P_n and its derivatives up to max_deriv-th at z.

    Uses the relation: dP_n/dz = 2 * dP_n/dxi (chain rule for xi = 2z + 1).
    Derivatives computed via the recursion for Legendre polynomial derivatives.
    """
    xi = 2.0 * z + 1.0
    npts = len(z)

    # Build Legendre values for all orders 0..n at all points
    # Then use derivative recursion
    results = [shifted_legendre(n, z)]

    if max_deriv >= 1 and n >= 1:
        # d/dxi P_n = n * (xi * P_n - P_{n-1}) / (xi^2 - 1) with L'Hopital at endpoints
        # Better: use the recursion (1-xi^2) P'_n = -n xi P_n + n P_{n-1}
        # => P'_n = n (P_{n-1} - xi P_n) / (1 - xi^2)  for |xi| != 1
        # At endpoints use the identity P'_n(1) = n(n+1)/2, P'_n(-1) = (-1)^{n+1} n(n+1)/2
        # More robust: use the matrix approach or numerical differentiation
        # We'll use high-order finite differences on a fine grid for robustness
        pass

    # More robust approach: evaluate on grid and use spectral differentiation
    # For the Galerkin method, we pre-compute on quadrature points
    # Use automatic differentiation via polynomial coefficients

    # Actually, let's compute Legendre polynomial coefficients and differentiate analytically
    # P_n(xi) coefficients via recursion
    if n == 0:
        poly_coeffs = np.array([1.0])
    elif n == 1:
        poly_coeffs = np.array([0.0, 1.0])  # xi
    else:
        p_prev = np.zeros(n + 1)
        p_prev[0] = 1.0
        p_curr = np.zeros(n + 1)
        p_curr[1] = 1.0
        for k in range(2, n + 1):
            p_next = np.zeros(n + 1)
            # (k) P_k = (2k-1) xi P_{k-1} - (k-1) P_{k-2}
            for j in range(n + 1):
                if j >= 1:
                    p_next[j] += (2 * k - 1) * p_curr[j - 1]
                p_next[j] -= (k - 1) * p_prev[j]
            p_next /= k
            p_prev = p_curr
            p_curr = p_next
        poly_coeffs = p_curr

    # Now poly_coeffs are coefficients of P_n(xi) = sum c_k xi^k
    # For shifted: xi = 2z + 1, so d/dz = 2 d/dxi
    results = []
    current = poly_coeffs.copy()
    # Evaluate P_n(xi(z))
    results.append(np.polyval(current[::-1], xi))

    for d in range(1, max_deriv + 1):
        # Differentiate w.r.t. xi
        if len(current) <= 1:
            current = np.array([0.0])
        else:
            current = np.array([k * current[k] for k in range(1, len(current))])
        # Multiply by 2^d for chain rule (each d/dz = 2 d/dxi)
        results.append(np.polyval(current[::-1], xi) * (2.0**d))

    return results


def gauss_legendre_quadrature(n_points: int = 32, a: float = -1.0, b: float = 0.0):
    """Gauss-Legendre quadrature points and weights on [a, b]."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    # Map from [-1, 1] to [a, b]
    z = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    w = 0.5 * (b - a) * weights
    return z, w


def legendre_basis_and_derivatives(J: int, z: np.ndarray, max_deriv: int = 4):
    """Compute shifted Legendre basis P_0..P_J and derivatives at z points.

    Returns list of arrays, each of shape (J+1, len(z)).
    Index 0 = values, 1 = first deriv, etc.
    """
    all_derivs = [np.zeros((J + 1, len(z))) for _ in range(max_deriv + 1)]

    for n in range(J + 1):
        derivs = shifted_legendre_derivatives(n, z, max_deriv)
        for d in range(min(len(derivs), max_deriv + 1)):
            all_derivs[d][n, :] = derivs[d]

    return all_derivs


def inner_product(f_values: np.ndarray, g_values: np.ndarray, weights: np.ndarray) -> float:
    """Integral of f*g over [-1, 0] via quadrature."""
    return float(np.sum(weights * f_values * g_values))


def mass_matrix(J: int, z_quad: np.ndarray, w_quad: np.ndarray) -> np.ndarray:
    """M_{ij} = integral P_i P_j dz."""
    M = np.zeros((J + 1, J + 1))
    basis = legendre_basis_and_derivatives(J, z_quad, 0)[0]
    for i in range(J + 1):
        for j in range(i, J + 1):
            val = inner_product(basis[i], basis[j], w_quad)
            M[i, j] = val
            M[j, i] = val
    return M


def stiffness_matrix(J: int, z_quad: np.ndarray, w_quad: np.ndarray) -> np.ndarray:
    """K_{ij} = integral P'_i P'_j dz."""
    K = np.zeros((J + 1, J + 1))
    derivs = legendre_basis_and_derivatives(J, z_quad, 1)
    dbasis = derivs[1]
    for i in range(J + 1):
        for j in range(i, J + 1):
            val = inner_product(dbasis[i], dbasis[j], w_quad)
            K[i, j] = val
            K[j, i] = val
    return K


def trig_product_cos_cos(k1: int, k2: int, I_max: int) -> list[tuple[int, float]]:
    """cos(k1*Y) * cos(k2*Y) = 0.5[cos((k1-k2)*Y) + cos((k1+k2)*Y)].

    Returns list of (harmonic, coefficient) pairs, discarding harmonics > I_max.
    """
    result = []
    km = abs(k1 - k2)
    kp = k1 + k2
    if km <= I_max:
        result.append((km, 0.5))
    if kp <= I_max:
        result.append((kp, 0.5))
    return result


def trig_product_sin_sin(k1: int, k2: int, I_max: int) -> list[tuple[int, float]]:
    """sin(k1*Y) * sin(k2*Y) = 0.5[cos((k1-k2)*Y) - cos((k1+k2)*Y)]."""
    result = []
    km = abs(k1 - k2)
    kp = k1 + k2
    if km <= I_max:
        result.append((km, 0.5))
    if kp <= I_max:
        result.append((kp, -0.5))
    return result


def trig_product_sin_cos(k1: int, k2: int, I_max: int) -> list[tuple[int, float]]:
    """sin(k1*Y) * cos(k2*Y) = 0.5[sin((k1+k2)*Y) + sin((k1-k2)*Y)]."""
    result = []
    kp = k1 + k2
    km = k1 - k2
    if abs(kp) <= I_max and kp >= 0:
        result.append((kp, 0.5))
    if abs(km) <= I_max:
        if km >= 0:
            result.append((km, 0.5))
        else:
            result.append((-km, -0.5))  # sin(-x) = -sin(x)
    return result
