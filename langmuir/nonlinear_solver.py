"""Nonlinear steady-state solver (CORE MODULE).

Implements two approaches from Hayes & Phillips (2017):
A) Small-l asymptotic expansion (sections 4-5) with nonlinearities
B) Galerkin numerical solver (section 6) for validation

Key equations: (29)-(68).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import root

from .profiles import ShearDriftProfile
from .robin_bc import RobinBoundaryConditions
from .linear_solver import LinearResult, solve_linear
from .utils import (
    poly_add,
    poly_multiply,
    poly_integrate,
    poly_eval_at,
    poly_definite_integral,
    poly_derivative,
)
from .galerkin import (
    gauss_legendre_quadrature, legendre_basis_and_derivatives,
    shifted_legendre, inner_product,
)

LOGGER = logging.getLogger(__name__)


class ConvergenceError(Exception):
    """Raised when a nonlinear solve does not converge."""


class SubcriticalBifurcationError(Exception):
    """Raised when the nonlinear branch no longer satisfies the subcritical hypothesis."""


@dataclass
class ContinuationPoint:
    l: float
    Ra: float
    state: np.ndarray
    tangent: np.ndarray | None = None


@dataclass
class NonlinearResult:
    R0: float
    R_bar_coeffs: np.ndarray   # [R0, R_bar_2, ...] nonlinear expansion coeffs
    R_star_2: float             # Neumann part of nonlinear R_2
    R_tilde_2: float            # Robin correction part
    lcNL: float                 # Critical nonlinear wavenumber
    RcNL: float                 # Critical nonlinear Rayleigh number
    kappa: float                # Amplitude-squared coefficient R*_2_NL / R*_2_linear
    wavenumber_ratio: float     # Ratio lcL / lcNL
    aspect_ratio: float         # 2*pi / lcNL (nondimensional)
    linear_result: LinearResult
    neutral_curve_NL: Callable[[float], float]
    neutral_curve_NL_numeric: Callable[[float], float] | None


def _neutral_curve_from_series(
    l: float,
    regular_coeffs: np.ndarray,
    *,
    singular_coeff: float,
) -> float:
    """Evaluate singular Robin contribution plus regular even-power series."""
    l_abs = abs(float(l))
    if l_abs < 1e-15:
        return float(np.inf)
    l2 = l_abs * l_abs
    value = float(regular_coeffs[0])
    l2k = 1.0
    for coeff in regular_coeffs[1:]:
        l2k *= l2
        value += float(coeff) * l2k
    return value + singular_coeff / l2


# ======================================================================
# Approach A: Small-l asymptotic expansion (sections 4-5)
# ======================================================================

def _solve_nonlinear_asymptotic(profile: ShearDriftProfile,
                                 bcs: RobinBoundaryConditions,
                                 max_order: int = 8) -> NonlinearResult:
    """Nonlinear perturbation solution following sections 4-5.

    The key difference from linear: at O(l^2), the nonlinear terms
    J(psi, u) and J(psi, nabla^2 psi) contribute. The expansion
    uses harmonics in Y = ly.

    For the nonlinear steady state, u_0 = cos(Y) (single harmonic, h_1=1).
    """
    a = profile.a  # D' coefficients
    b = profile.b  # U' coefficients

    # First solve the linear problem to get psi_tilde_1, R0, etc.
    linear = solve_linear(profile, bcs, max_order)
    R0 = linear.R0
    psi_t1 = linear.psi_tilde_1_coeffs  # psi_tilde_1(z)

    # For the nonlinear steady state with single harmonic:
    # u_0 = cos(Y) => h_0 = 0, h_1 = 1 in eq. (52)
    # Psi_0 = R_0 * psi_tilde_1 * sin(Y) (from eq. 53, m=1)

    # We need to compute the nonlinear R*_2 (Neumann part) from eq. (63)
    # and R_tilde_2 (Robin part) from eq. (64).

    # At the nonlinear steady state with u_0 = cos(Y):
    # u_2 has harmonics m=0 and m=2 from the nonlinear terms.
    # The m=0 harmonic is the base-flow modification.

    # === Compute u_2 from equation (47)/(54) at steady state ===
    # u_2'' = -Psi_0_Y * U' + Psi_0_z * u_0_Y  (nonlinear terms)
    #       + (u_0_T - u_0_YY) [= 0 at steady state with sigma_2=0]
    #       - Psi_0_Y * (du_0/dz) [= 0 since u_0 independent of z]

    # Psi_0 = R_0 * psi_tilde_1(z) * sin(Y)
    # Psi_0_Y = R_0 * psi_tilde_1(z) * cos(Y)
    # Psi_0_z = R_0 * psi_tilde_1'(z) * sin(Y)
    # u_0 = cos(Y), u_0_Y = -sin(Y), u_0_z = 0

    psi_t1_deriv = poly_derivative(psi_t1)

    # u_2'' = -R_0 * psi_tilde_1 * U' * cos(Y)  [from -Psi_0_Y * U']
    #        + R_0 * psi_tilde_1' * sin(Y) * (-sin(Y))  [from Psi_0_z * u_0_Y]
    #
    # The sin(Y)*(-sin(Y)) = -sin^2(Y) = -1/2 + 1/2*cos(2Y)
    # The cos(Y) term stays as cos(Y)

    # So u_2'' has harmonics m=0 and m=1 and m=2:
    # m=0: R_0 * psi_tilde_1'(z) * (-1/2)
    # m=1: -R_0 * psi_tilde_1(z) * U'(z) * cos(Y)  [this is the linear part]
    # m=2: R_0 * psi_tilde_1'(z) * (1/2) * cos(2Y)

    # Actually, at steady state u_0_T = 0 and u_0_YY = -cos(Y).
    # From eq (51) at onset sigma_2 = 0: u_0_T = 0 trivially.
    # But the linear contribution -u_0_YY + Psi_0_Y * U' gives:
    # cos(Y) - R_0 * psi_tilde_1 * U' * cos(Y) = cos(Y)(1 - R_0 * int psi_t1 U' dz)
    # Wait, u_2 depends on z and Y.

    # Let me be more careful. From eq (47):
    # u_2'' = u_0_T - u_0_YY - Psi_0_Y * U' + Psi_0_z * u_0_Y - Psi_0_Y * u_0_z
    # At steady state: u_0_T = 0, u_0 = cos(Y) (z-indep), so u_0_z = 0, u_0_YY = -cos(Y)
    # u_2'' = cos(Y) - R_0 * psi_t1(z) * U'(z) * cos(Y)
    #        + R_0 * psi_t1'(z) * sin(Y) * (-sin(Y))
    # = cos(Y) * [1 - R_0 * psi_t1(z) * U'(z)]
    #   - R_0 * psi_t1'(z) * sin^2(Y)
    # = cos(Y) * [1 - R_0 * psi_t1(z) * U'(z)]
    #   - R_0 * psi_t1'(z) * [1/2 - 1/2 cos(2Y)]

    # So u_2(Y,z) = u_{2,0}(z) + u_{2,1}(z) cos(Y) + u_{2,2}(z) cos(2Y)

    # m=0 component: u_{2,0}'' = -R_0/2 * psi_t1'(z)
    rhs_u20 = -R0 / 2.0 * psi_t1_deriv

    # m=1 component: u_{2,1}'' = 1 - R_0 * psi_t1(z) * U'(z)
    psi_t1_U = poly_multiply(psi_t1, b)
    rhs_u21_poly = np.array([1.0])  # constant 1
    rhs_u21 = poly_add(rhs_u21_poly, -R0 * psi_t1_U)

    # I=1 harmonic truncation (Appendix C theorem): keep only m=0,1;
    # discard m=2 harmonics as in the paper's asymptotic and numerical (I=1) solutions.
    # The m=2 component u_{2,2} is set to zero under this truncation.

    # Solve each with Neumann BCs (Robin only enters at O(l^4))
    u20 = _solve_u_neumann_integral(rhs_u20, 0.0)  # int u_{2,0} dz = 0
    u21 = _solve_u_neumann_integral(rhs_u21, 0.0)  # int u_{2,1} dz = 0

    # === Compute Psi_2 from equation (55) at steady state ===
    # With I=1 truncation, only the m=1 (sin(Y)) component of Psi_2 is kept.
    # The m=2 (sin(2Y)) component is discarded along with u_{2,2}.

    psi_t1_dd = poly_derivative(psi_t1_deriv)

    # m=1 (sin(Y)) component of Psi_2'''':
    # hat part: 2*R_0*psi_t1'' + R_0*D'*u_{2,1}
    # tilde part: D' (from R_2 * D')
    rhs_psi21_hat = poly_add(
        2.0 * R0 * psi_t1_dd,
        R0 * poly_multiply(a, u21)
    )
    rhs_psi21_tilde = a.copy()  # D'

    # Solve 4th-order BVPs with psi=0, psi''=0 at z=0,-1
    from .linear_solver import _solve_psi_fourth_order

    psi21_hat = _solve_psi_fourth_order(rhs_psi21_hat)
    psi21_tilde = _solve_psi_fourth_order(rhs_psi21_tilde)

    # === Compute R_2 from equation (63) at nonlinear steady state ===
    # Solvability condition from eq (62) projected onto cos(Y).
    # With I=1 truncation (m<=1), u_{2,2}=0 and Psi_{2,2}=0, so:
    # - The only NL Jacobian contribution at m=1 is from Psi_0_Y * u_{2,0}_z
    # - Cross-terms involving m=2 harmonics vanish under truncation.

    # Denominator: int U' psi_tilde_{2,1} dz = 1/R0
    psi_tilde_21_U = poly_multiply(b, psi21_tilde)
    denom_integral = poly_definite_integral(psi_tilde_21_U, -1.0, 0.0)

    # NL Jacobian at cos(Y): int R_0 psi_t1 u'_{2,0} dz
    # (from Psi_0_Y * u_2_z: R_0 psi_t1 cos(Y) * u'_{2,0} → cos(Y) component)
    u20_deriv = poly_derivative(u20)
    nl_correction = R0 * poly_definite_integral(
        poly_multiply(psi_t1, u20_deriv), -1.0, 0.0
    )

    # Linear terms: int U' psi_hat_{2,1} dz
    psi_hat_21_U = poly_multiply(b, psi21_hat)
    lin_num = poly_definite_integral(psi_hat_21_U, -1.0, 0.0)

    # Solvability: 0 = -gamma_tilde + lin_num + R_2*denom + nl_correction
    # R*_2 = (-lin_num - nl_correction) / denom
    if abs(denom_integral) < 1e-20:
        R_star_2_NL = 0.0
        R_bar_2 = 0.0
    else:
        R_star_2_NL = (-lin_num - nl_correction) / denom_integral
        R_bar_2 = R_star_2_NL

    R_tilde_2_NL = R0  # = 1/denom = R0, same as linear case

    # Build nonlinear R_bar coefficients
    R_bar_coeffs = np.zeros(max_order)
    R_bar_coeffs[0] = R0
    R_bar_coeffs[1] = R_bar_2

    # For higher orders, use linear approximation (the O(l^4) terms give the
    # dominant contribution; higher-order NL effects are small)
    for k in range(2, max_order):
        R_bar_coeffs[k] = linear.R_coeffs[k]

    gamma = bcs.gamma

    # Critical nonlinear wavenumber from eq (66):
    # lcNL = (gamma * R_tilde_2 / R*_2)^{1/4}
    amplitude_kappa = (
        float(R_star_2_NL / linear.R_coeffs[1])
        if len(linear.R_coeffs) > 1 and abs(linear.R_coeffs[1]) > 1e-20
        else float("inf")
    )
    # lc ~ (gamma * R_tilde_2 / R_star_2)^(1/4), so subcritical widening
    # requires R_star_2_NL > R_star_2_linear, i.e. amplitude_kappa > 1.
    if not np.isfinite(amplitude_kappa) or amplitude_kappa <= 1.0:
        raise SubcriticalBifurcationError(
            "Nonlinear amplitude-squared coefficient does not widen the branch: "
            f"kappa={amplitude_kappa:.6f}"
        )

    if gamma > 0 and R_star_2_NL > 0 and R_tilde_2_NL > 0:
        lcNL = (gamma * R_tilde_2_NL / R_star_2_NL) ** 0.25
        RcNL = R0 + 2.0 * (gamma * R_tilde_2_NL * R_star_2_NL) ** 0.5
    else:
        lcNL = 0.0
        RcNL = R0

    # Wavenumber ratio from eq. 68 retained for diagnostics.
    lcL = linear.lcL
    if lcNL > 0:
        wavenumber_ratio = lcL / lcNL
    else:
        wavenumber_ratio = float("nan")

    aspect_ratio = 2.0 * np.pi / lcNL if lcNL > 0 else np.inf

    def neutral_curve_NL(l: float) -> float:
        return _neutral_curve_from_series(
            l,
            R_bar_coeffs,
            singular_coeff=gamma * R_tilde_2_NL,
        )

    return NonlinearResult(
        R0=R0,
        R_bar_coeffs=R_bar_coeffs,
        R_star_2=R_star_2_NL,
        R_tilde_2=R_tilde_2_NL,
        lcNL=lcNL,
        RcNL=RcNL,
        kappa=amplitude_kappa,
        wavenumber_ratio=wavenumber_ratio,
        aspect_ratio=aspect_ratio,
        linear_result=linear,
        neutral_curve_NL=neutral_curve_NL,
        neutral_curve_NL_numeric=None,  # filled by Galerkin solver
    )


def _solve_u_neumann_integral(rhs: np.ndarray, integral_constraint: float) -> np.ndarray:
    """Solve u'' = rhs with Neumann BCs u'(0)=0, u'(-1)=0 and integral constraint."""
    u_prime = poly_integrate(rhs, 1)
    # u'(0) = 0: c0 = -u_prime(0)
    c0 = -poly_eval_at(u_prime, 0.0)
    u_prime_full = u_prime.copy()
    if len(u_prime_full) == 0:
        u_prime_full = np.array([c0])
    else:
        u_prime_full[0] += c0

    u = poly_integrate(u_prime_full, 1)
    int_val = poly_definite_integral(u, -1.0, 0.0)
    c1 = integral_constraint - int_val
    u_final = u.copy()
    if len(u_final) == 0:
        u_final = np.array([c1])
    else:
        u_final[0] += c1
    return u_final


# ======================================================================
# Approach B: Galerkin numerical solver (section 6)
# ======================================================================

def _galerkin_residual(x: np.ndarray, l: float, Ra: float,
                       profile: ShearDriftProfile,
                       bcs: RobinBoundaryConditions,
                       I: int, J: int,
                       z_quad: np.ndarray, w_quad: np.ndarray,
                       basis_derivs: list) -> np.ndarray:
    """Route all numeric Galerkin solves through the collocation residual."""
    return _galerkin_collocation_residual(x, l, Ra, profile, bcs, I, J,
                                          z_quad, w_quad, basis_derivs)


def _galerkin_collocation_residual(x: np.ndarray, l: float, Ra: float,
                                    profile: ShearDriftProfile,
                                    bcs: RobinBoundaryConditions,
                                    I: int, J: int,
                                    z_quad: np.ndarray, w_quad: np.ndarray,
                                    basis_derivs: list) -> np.ndarray:
    """Collocation-based residual for steady nonlinear CL-equations.

    For I=1 (fundamental + zeroth harmonic), the system is manageable.
    u = sum_m [A_{m,0} P_m(z) + A_{m,1} P_m(z) cos(ly)]
    psi = sum_m [B_{m,0} P_m(z) * 0 + B_{m,1} P_m(z) sin(ly)]
    (B_{m,0} = 0 since sin(0*ly) = 0, but we keep them for generality)
    """
    n_u_modes = J - 1  # Legendre modes for u
    n_psi_modes = J + 1  # Legendre modes for psi

    n_A = n_u_modes * (I + 1)
    n_B = n_psi_modes * (I + 1)

    A = x[:n_A].reshape(n_u_modes, I + 1)
    B = x[n_A:n_A + n_B].reshape(n_psi_modes, I + 1)

    P = basis_derivs[0]    # (max_n+1, nq)
    dP = basis_derivs[1]
    d2P = basis_derivs[2]
    d3P = basis_derivs[3]
    d4P = basis_derivs[4]

    nq = len(z_quad)
    Dp = profile.D_prime(z_quad)
    Up = profile.U_prime(z_quad)

    # Reconstruct fields for each harmonic k
    # u_k(z) = sum_m A_{m,k} P_m(z)  (coefficient of cos(kly))
    # psi_k(z) = sum_m B_{m,k} P_m(z)  (coefficient of sin(kly))

    u_k = np.zeros((I + 1, nq))
    u_k_zz = np.zeros((I + 1, nq))
    u_k_z = np.zeros((I + 1, nq))
    psi_k = np.zeros((I + 1, nq))
    psi_k_zz = np.zeros((I + 1, nq))
    psi_k_zzzz = np.zeros((I + 1, nq))
    psi_k_z = np.zeros((I + 1, nq))
    psi_k_zzz = np.zeros((I + 1, nq))

    for k in range(I + 1):
        for m in range(n_u_modes):
            u_k[k] += A[m, k] * P[m]
            u_k_z[k] += A[m, k] * dP[m]
            u_k_zz[k] += A[m, k] * d2P[m]
        for m in range(n_psi_modes):
            psi_k[k] += B[m, k] * P[m]
            psi_k_z[k] += B[m, k] * dP[m]
            psi_k_zz[k] += B[m, k] * d2P[m]
            psi_k_zzz[k] += B[m, k] * d3P[m]
            psi_k_zzzz[k] += B[m, k] * d4P[m]

    residuals = []

    # --- u equation (1b) at steady state for each harmonic k ---
    # -u_zz + l^2 k^2 u = U' * (k*l) * psi_k + nonlinear terms
    # Nonlinear: J(psi, u) = psi_y u_z - psi_z u_y
    # For I=1: nonlinear coupling between k=0 and k=1

    for k in range(I + 1):
        # Linear part
        res_u = -u_k_zz[k] + (l * k)**2 * u_k[k]

        # Forcing: -U' * k*l * psi_k (from U' psi_y, where psi_y extracts k*l*cos(kly))
        if k > 0:
            res_u -= Up * (k * l) * psi_k[k]

        # Nonlinear Jacobian terms (for I=1):
        # J(psi, u) projected onto cos(kly):
        if I >= 1:
            # psi_y = l * psi_1 cos(ly), psi_z = psi_1_z sin(ly)
            # u_y = -l * u_1 sin(ly), u_z = u_0_z + u_1_z cos(ly)
            # J = psi_y * u_z - psi_z * u_y
            #   = l psi_1 cos(ly) * [u_0_z + u_1_z cos(ly)]
            #     - psi_1_z sin(ly) * [-l u_1 sin(ly)]
            #   = l psi_1 u_0_z cos(ly) + l psi_1 u_1_z cos^2(ly)
            #     + l psi_1_z u_1 sin^2(ly)
            #   = l psi_1 u_0_z cos(ly)
            #     + l/2 (psi_1 u_1_z + psi_1_z u_1) [from cos^2 + sin^2 = 1, m=0]
            #     + l/2 (psi_1 u_1_z - psi_1_z u_1) cos(2ly) [m=2, discarded if I=1]
            if k == 0:
                # m=0 component: l/2 (psi_1 u_1_z + psi_1_z u_1)
                res_u -= l / 2.0 * (psi_k[1] * u_k_z[1] + psi_k_z[1] * u_k[1])
            elif k == 1:
                # m=1 component: l psi_1 u_0_z
                res_u -= l * psi_k[1] * u_k_z[0]

        # Project onto Legendre basis (Galerkin)
        for j_test in range(max(n_u_modes - 2, 1)):
            val = inner_product(res_u, P[j_test], w_quad)
            residuals.append(val)

    # --- psi equation (1c) at steady state for each harmonic k ---
    # -(nabla^2)^2 psi = Ra D' u_y + J(psi, nabla^2 psi)
    # nabla^2 psi_k = psi_k_zz - l^2 k^2 psi_k
    # (nabla^2)^2 psi_k = psi_k_zzzz - 2 l^2 k^2 psi_k_zz + l^4 k^4 psi_k

    for k in range(I + 1):
        if k == 0:
            # sin(0) = 0, so psi_0 equation is trivial if B_{m,0} = 0
            # Force B_{m,0} = 0
            for m in range(n_psi_modes):
                residuals.append(B[m, 0])
            continue

        # Biharmonic
        biharm = psi_k_zzzz[k] - 2 * (l * k)**2 * psi_k_zz[k] + (l * k)**4 * psi_k[k]

        # Forcing: Ra D' * k*l * u_k (from Ra D' u_y = Ra D' * (-k*l) sin(kly) × u_k cos(kly) → extracts sin(kly))
        # Actually u_y for cos(kly) = -k*l*sin(kly), so Ra*D'*u_y extracts -k*l*Ra*D'*u_k for sin(kly)
        res_psi = -biharm - Ra * Dp * (k * l) * u_k[k]

        # Nonlinear terms for I=1, k=1:
        if I >= 1 and k == 1:
            # J(psi, nabla^2 psi) = psi_y (nabla^2 psi)_z - psi_z (nabla^2 psi)_y
            # For I=1: only k=1 mode exists for psi
            # (nabla^2 psi)_1 = psi_1_zz - l^2 psi_1
            lap_psi_1 = psi_k_zz[1] - l**2 * psi_k[1]
            lap_psi_1_z = psi_k_zzz[1] - l**2 * psi_k_z[1]

            # psi_y = l psi_1 cos(ly), psi_z = psi_1_z sin(ly)
            # (nabla^2 psi)_y = -l * lap_psi_1 * cos(ly) ... wait
            # nabla^2 psi = (psi_zz - l^2 psi) sin(ly) = lap_psi_1 sin(ly)
            # (nabla^2 psi)_z = lap_psi_1_z sin(ly)
            # (nabla^2 psi)_y = l * lap_psi_1 cos(ly)

            # J = l*psi_1*cos(ly) * lap_psi_1_z*sin(ly)
            #   - psi_1_z*sin(ly) * l*lap_psi_1*cos(ly)
            # = l*cos(ly)*sin(ly) * [psi_1 * lap_psi_1_z - psi_1_z * lap_psi_1]
            # = l/2 * sin(2ly) * [psi_1 * lap_psi_1_z - psi_1_z * lap_psi_1]
            # This is a sin(2ly) term → k=2, discarded for I=1

            # So nonlinear Jacobian in psi equation is zero for I=1!
            # The nonlinear effect enters only through the u equation coupling.

            # But there IS a nonlinear contribution from Ra*D'*u_y:
            # u has k=0 component from nonlinear coupling
            # Ra*D'*u_y for u = u_0 + u_1 cos(ly):
            # u_y = -l u_1 sin(ly), so D'*u_y = -l*D'*u_1*sin(ly)
            # This is already included above. The u_0 term has u_0_y = 0, no contribution.
        for j_test in range(max(n_psi_modes - 4, 1)):
            val = inner_product(res_psi, P[j_test], w_quad)
            residuals.append(val)

    # --- Boundary conditions ---
    # u: u_z + gamma_s u = 0 at z=0, -u_z + gamma_b u = 0 at z=-1
    # psi: psi_zz + gamma_s/2 psi_z = 0 at z=0, psi(-1) = 0, psi(0) = 0
    #      -psi_zz + gamma_b psi_z = 0 at z=-1

    z_top = np.array([0.0])
    z_bot = np.array([-1.0])
    P_top = np.array([shifted_legendre(m, z_top)[0] for m in range(max(n_u_modes, n_psi_modes))])
    P_bot = np.array([shifted_legendre(m, z_bot)[0] for m in range(max(n_u_modes, n_psi_modes))])

    dP_top = np.zeros(max(n_u_modes, n_psi_modes))
    dP_bot = np.zeros(max(n_u_modes, n_psi_modes))
    d2P_top = np.zeros(max(n_u_modes, n_psi_modes))
    d2P_bot = np.zeros(max(n_u_modes, n_psi_modes))

    from .galerkin import shifted_legendre_derivatives
    for m in range(max(n_u_modes, n_psi_modes)):
        derivs_t = shifted_legendre_derivatives(m, z_top, 2)
        derivs_b = shifted_legendre_derivatives(m, z_bot, 2)
        dP_top[m] = derivs_t[1][0]
        dP_bot[m] = derivs_b[1][0]
        d2P_top[m] = derivs_t[2][0]
        d2P_bot[m] = derivs_b[2][0]

    for k in range(I + 1):
        # u BCs
        # z=0: u_z + gamma_s u = 0
        uz_top = sum(A[m, k] * dP_top[m] for m in range(n_u_modes))
        u_top = sum(A[m, k] * P_top[m] for m in range(n_u_modes))
        residuals.append(uz_top + bcs.gamma_s * u_top)

        # z=-1: -u_z + gamma_b u = 0
        uz_bot = sum(A[m, k] * dP_bot[m] for m in range(n_u_modes))
        u_bot = sum(A[m, k] * P_bot[m] for m in range(n_u_modes))
        residuals.append(-uz_bot + bcs.gamma_b * u_bot)

    for k in range(1, I + 1):  # skip k=0 for psi (already forced to 0)
        # psi BCs
        # psi(0) = 0
        psi_top = sum(B[m, k] * P_top[m] for m in range(n_psi_modes))
        residuals.append(psi_top)

        # psi(-1) = 0
        psi_bot = sum(B[m, k] * P_bot[m] for m in range(n_psi_modes))
        residuals.append(psi_bot)

        # psi_zz + gamma_s/2 psi_z = 0 at z=0
        psizz_top = sum(B[m, k] * d2P_top[m] for m in range(n_psi_modes))
        psiz_top = sum(B[m, k] * dP_top[m] for m in range(n_psi_modes))
        residuals.append(psizz_top + bcs.gamma_s / 2.0 * psiz_top)

        # -psi_zz + gamma_b psi_z = 0 at z=-1
        psizz_bot = sum(B[m, k] * d2P_bot[m] for m in range(n_psi_modes))
        psiz_bot = sum(B[m, k] * dP_bot[m] for m in range(n_psi_modes))
        residuals.append(-psizz_bot + bcs.gamma_b * psiz_bot)

    # Normalisation: coefficient of cos(ly) in u at z=0 is 1
    # A_{0,1} such that u_1(0) = 1
    u1_at_0 = sum(A[m, 1] * P_top[m] for m in range(n_u_modes)) if I >= 1 else 0.0
    residuals.append(u1_at_0 - 1.0)

    return np.array(residuals, dtype=float)


def _resize_state_vector(x: np.ndarray | None, n_total: int) -> np.ndarray:
    if x is None:
        x = np.zeros(n_total, dtype=float)
        if n_total > 1:
            x[1] = 1.0
        return x
    x_arr = np.asarray(x, dtype=float)
    if len(x_arr) == n_total:
        return x_arr
    out = np.zeros(n_total, dtype=float)
    out[:min(len(x_arr), n_total)] = x_arr[:min(len(x_arr), n_total)]
    return out


def _bootstrap_continuation_point(
    l: float,
    profile: ShearDriftProfile,
    bcs: RobinBoundaryConditions,
    linear_neutral: Callable[[float], float],
    *,
    I: int,
    J: int,
    x0: np.ndarray | None = None,
) -> ContinuationPoint:
    ra_guess = max(float(linear_neutral(l)) * 1.05, 1e-8)
    state = solve_galerkin_steady_state(l, ra_guess, profile, bcs, I=I, J=J, x0=x0)
    return ContinuationPoint(l=float(l), Ra=float(ra_guess), state=state)


def _continuation_corrector(
    l: float,
    predicted: np.ndarray,
    tangent: np.ndarray,
    profile: ShearDriftProfile,
    bcs: RobinBoundaryConditions,
    *,
    I: int,
    J: int,
) -> ContinuationPoint:
    z_quad, w_quad = gauss_legendre_quadrature(n_points=max(32, 2 * J))
    basis_derivs = legendre_basis_and_derivatives(max(J, J + 1), z_quad, max_deriv=4)
    n_u_modes = J - 1
    n_psi_modes = J + 1
    n_total = (n_u_modes + n_psi_modes) * (I + 1)

    def augmented_residual(v: np.ndarray) -> np.ndarray:
        state = _resize_state_vector(v[:-1], n_total)
        Ra = float(v[-1])
        residual = _galerkin_collocation_residual(
            state,
            l,
            Ra,
            profile,
            bcs,
            I,
            J,
            z_quad,
            w_quad,
            basis_derivs,
        )
        orthogonality = float(np.dot(v - predicted, tangent))
        return np.concatenate([residual, np.array([orthogonality])])

    result = root(
        augmented_residual,
        predicted,
        method="lm",
        options={"maxiter": 5000, "ftol": 1e-12},
    )
    if not result.success:
        raise ConvergenceError(
            f"Pseudo-arclength corrector failed at l={l}: {result.message}"
        )
    corrected = np.asarray(result.x, dtype=float)
    tangent_out = corrected - predicted
    tangent_norm = np.linalg.norm(tangent_out)
    if tangent_norm > 0:
        tangent_out = tangent_out / tangent_norm
    else:
        tangent_out = tangent
    return ContinuationPoint(
        l=float(l),
        Ra=float(corrected[-1]),
        state=_resize_state_vector(corrected[:-1], n_total),
        tangent=tangent_out,
    )


def _continue_direction(
    indices: list[int],
    l_values: np.ndarray,
    points: list[ContinuationPoint],
    R_bar: np.ndarray,
    output_indices: np.ndarray,
    profile: ShearDriftProfile,
    bcs: RobinBoundaryConditions,
    linear_neutral: Callable[[float], float],
    *,
    I: int,
    J: int,
) -> None:
    for idx in indices:
        p_prev, p_curr = points[-2], points[-1]
        tangent = np.concatenate([p_curr.state, np.array([p_curr.Ra])]) - np.concatenate(
            [p_prev.state, np.array([p_prev.Ra])]
        )
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0.0:
            tangent[-1] = 1.0
            tangent_norm = np.linalg.norm(tangent)
        tangent = tangent / tangent_norm
        predicted = np.concatenate([p_curr.state, np.array([p_curr.Ra])]) + tangent * tangent_norm
        try:
            point = _continuation_corrector(
                float(l_values[idx]),
                predicted,
                tangent,
                profile,
                bcs,
                I=I,
                J=J,
            )
        except ConvergenceError as exc:
            LOGGER.warning(
                "Pseudo-arclength continuation failed at l=%s; retrying from linear bootstrap. %s",
                l_values[idx],
                exc,
            )
            point = _bootstrap_continuation_point(
                float(l_values[idx]),
                profile,
                bcs,
                linear_neutral,
                I=I,
                J=J,
                x0=p_curr.state,
            )
        R_bar[int(output_indices[idx])] = point.Ra
        points.append(point)


def solve_galerkin_steady_state(l: float, Ra: float,
                                 profile: ShearDriftProfile,
                                 bcs: RobinBoundaryConditions,
                                 I: int = 2, J: int = 13,
                                 x0: np.ndarray | None = None) -> np.ndarray:
    """Solve for nonlinear steady state at given (l, Ra) using Galerkin method."""
    z_quad, w_quad = gauss_legendre_quadrature(n_points=max(32, 2 * J))
    basis_derivs = legendre_basis_and_derivatives(max(J, J + 1), z_quad, max_deriv=4)

    n_u_modes = J - 1
    n_psi_modes = J + 1
    n_A = n_u_modes * (I + 1)
    n_B = n_psi_modes * (I + 1)
    n_total = n_A + n_B

    x0 = _resize_state_vector(x0, n_total)

    def residual_func(x):
        return _galerkin_collocation_residual(x, l, Ra, profile, bcs, I, J,
                                              z_quad, w_quad, basis_derivs)

    # Check residual dimensions match
    r0 = residual_func(x0)
    if len(r0) != len(x0):
        # Adjust - truncate residual or pad unknowns
        # Use least-squares for overdetermined systems
        result = root(residual_func, x0, method='lm',
                     options={'maxiter': 5000, 'ftol': 1e-12})
    else:
        result = root(residual_func, x0, method='hybr',
                     options={'maxfev': 10000})

    if not result.success:
        raise ConvergenceError(f"Newton failed at l={l}, Ra={Ra}: {result.message}")

    return result.x


def nonlinear_neutral_curve_numeric(l_array: np.ndarray,
                                     profile: ShearDriftProfile,
                                     bcs: RobinBoundaryConditions,
                                     linear_neutral: Callable,
                                     I: int = 2, J: int = 13) -> np.ndarray:
    """Compute a numerical nonlinear branch using pseudo-arclength continuation."""
    R_bar = np.full_like(l_array, np.nan)
    finite_mask = np.abs(l_array) >= 1e-6
    R_bar[~finite_mask] = np.inf
    if not np.any(finite_mask):
        return R_bar

    finite_indices = np.where(finite_mask)[0]
    l_finite = l_array[finite_indices]
    linear_vals = np.array([linear_neutral(float(l_val)) for l_val in l_finite], dtype=float)
    seed_pos = int(np.nanargmin(linear_vals))

    if len(finite_indices) == 1:
        point = _bootstrap_continuation_point(
            float(l_finite[0]),
            profile,
            bcs,
            linear_neutral,
            I=I,
            J=J,
        )
        R_bar[finite_indices[0]] = point.Ra
        return R_bar

    if seed_pos == len(finite_indices) - 1:
        second_pos = seed_pos - 1
    else:
        second_pos = seed_pos + 1

    p0 = _bootstrap_continuation_point(
        float(l_finite[seed_pos]),
        profile,
        bcs,
        linear_neutral,
        I=I,
        J=J,
    )
    p1 = _bootstrap_continuation_point(
        float(l_finite[second_pos]),
        profile,
        bcs,
        linear_neutral,
        I=I,
        J=J,
        x0=p0.state,
    )
    R_bar[finite_indices[seed_pos]] = p0.Ra
    R_bar[finite_indices[second_pos]] = p1.Ra

    lower_seed, upper_seed = sorted([seed_pos, second_pos])
    forward_points = [p0, p1] if seed_pos < second_pos else [p1, p0]
    _continue_direction(
        list(range(upper_seed + 1, len(finite_indices))),
        l_finite,
        forward_points,
        R_bar,
        finite_indices,
        profile,
        bcs,
        linear_neutral,
        I=I,
        J=J,
    )

    backward_points = [p1, p0] if seed_pos < second_pos else [p0, p1]
    _continue_direction(
        list(range(lower_seed - 1, -1, -1)),
        l_finite,
        backward_points,
        R_bar,
        finite_indices,
        profile,
        bcs,
        linear_neutral,
        I=I,
        J=J,
    )

    return R_bar


# ======================================================================
# Main entry point
# ======================================================================

def solve_nonlinear(profile: ShearDriftProfile,
                    bcs: RobinBoundaryConditions,
                    max_order: int = 8,
                    run_galerkin: bool = False) -> NonlinearResult:
    """Solve the full nonlinear CL problem.

    Parameters
    ----------
    profile : ShearDriftProfile
    bcs : RobinBoundaryConditions
    max_order : int
        Expansion order (8 → O(l^16))
    run_galerkin : bool
        If True, also run the Galerkin numerical solver for validation.
    """
    result = _solve_nonlinear_asymptotic(profile, bcs, max_order)

    if run_galerkin and result.lcNL > 0:
        try:
            l_array = np.linspace(0.01, 2.0 * result.lcNL, 20)
            R_bar_num = nonlinear_neutral_curve_numeric(
                l_array, profile, bcs, result.linear_result.neutral_curve, I=2
            )

            def neutral_numeric(l_val):
                return float(np.interp(l_val, l_array, R_bar_num))

            result.neutral_curve_NL_numeric = neutral_numeric
        except Exception as exc:
            LOGGER.warning("Galerkin continuation validation skipped: %s", exc)

    return result
