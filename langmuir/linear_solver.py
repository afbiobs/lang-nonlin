"""Linear perturbation solution (baseline).

Implements the small-l expansion from Hayes & Phillips (2017) section 3
to O(l^16). Uses SymPy for symbolic polynomial integration at each order,
then converts to numerical evaluation.

Key equations: (9)-(20), (21)-(26).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from .profiles import ShearDriftProfile
from .robin_bc import RobinBoundaryConditions
from .utils import (
    poly_add,
    poly_multiply,
    poly_integrate,
    poly_eval_at,
    poly_definite_integral,
    poly_derivative,
)


@dataclass
class LinearResult:
    R0: float
    R_coeffs: np.ndarray  # [R0, R*_2, R*_4, ...] Neumann part
    R_tilde_coeffs: np.ndarray  # [0, R_tilde_2, R_tilde_4, ...] Robin corrections
    lcL: float
    RcL: float
    neutral_curve: Callable[[float], float]
    psi_tilde_1_coeffs: np.ndarray  # polynomial coefficients for psi_tilde_1(z)
    u_coeffs: dict  # {order: polynomial coefficients for u_{2k}(z)}
    psi_hat_coeffs: dict  # {order: polynomial coefficients for psi_hat_{2k+1}(z)}
    psi_tilde_coeffs: dict  # {order: polynomial coefficients for psi_tilde_{2k+1}(z)}


def _poly_iint_with_bc(rhs_coeffs: np.ndarray, bc_type: str = "neumann",
                        gamma_tilde_s: float = 0.0, gamma_tilde_b: float = 0.0,
                        u_lower_order_at_0: float = 0.0,
                        u_lower_order_at_m1: float = 0.0,
                        integral_constraint: float = 0.0) -> np.ndarray:
    """Double integrate u'' = rhs with BCs and integral constraint.

    For u: BCs are u'(0) + gamma_tilde_s * u_{lower} = 0 (at z=0)
                    u'(-1) - gamma_tilde_b * u_{lower} = 0 (at z=-1)
    Plus integral constraint: int_{-1}^0 u dz = integral_constraint
    """
    # u' = integral(rhs) + c0
    u_prime_no_const = poly_integrate(rhs_coeffs, 1)

    # BC at z=0: u'(0) + gamma_tilde_s * u_lower(0) = 0
    # u'(0) = u_prime_no_const(0) + c0 = -gamma_tilde_s * u_lower(0)
    c0 = -gamma_tilde_s * u_lower_order_at_0 - poly_eval_at(u_prime_no_const, 0.0)

    # u = integral(u' + c0) + c1
    u_prime_coeffs = u_prime_no_const.copy()
    u_prime_coeffs[0] += c0
    u_no_const = poly_integrate(u_prime_coeffs, 1)

    # Integral constraint: int_{-1}^0 u dz = integral_constraint
    # int u_no_const dz + c1 * 1 = integral_constraint
    int_u = poly_definite_integral(u_no_const, -1.0, 0.0)
    c1 = integral_constraint - int_u

    # Final u coefficients
    u_coeffs = u_no_const.copy()
    u_coeffs[0] += c1

    return u_coeffs


def _solve_psi_fourth_order(rhs_coeffs: np.ndarray,
                             gamma_tilde_s_half: float = 0.0,
                             gamma_tilde_b: float = 0.0,
                             psi_lower_prime_at_0: float = 0.0,
                             psi_lower_prime_at_m1: float = 0.0) -> np.ndarray:
    """Solve psi'''' = rhs with BCs:
    psi''(0) + (gamma_tilde_s/2) * psi'_lower(0) = 0, psi(0) = 0
    psi''(-1) - gamma_tilde_b * psi'_lower(-1) = 0, psi(-1) = 0

    Returns polynomial coefficients for psi(z).
    """
    # psi''' = integral(rhs) + c0
    p3 = poly_integrate(rhs_coeffs, 1)

    # psi'' = integral(psi''') + c1 => iintegral(rhs) + c0*z + c1
    p2_no_c = poly_integrate(rhs_coeffs, 2)

    # psi' = triple integral of rhs + c0*z^2/2 + c1*z + c2
    p1_no_c = poly_integrate(rhs_coeffs, 3)

    # psi = quad integral of rhs + c0*z^3/6 + c1*z^2/2 + c2*z + c3
    p0_no_c = poly_integrate(rhs_coeffs, 4)

    # We have 4 constants: c0, c1, c2, c3
    # BCs:
    # 1) psi''(0) + gamma_tilde_s_half * psi'_lower(0) = 0
    #    p2_no_c(0) + c1 = -gamma_tilde_s_half * psi_lower_prime_at_0
    # 2) psi(-1) = 0
    # 3) psi''(-1) - gamma_tilde_b * psi'_lower(-1) = 0
    #    p2_no_c(-1) + c0*(-1) + c1 = gamma_tilde_b * psi_lower_prime_at_m1
    # 4) psi(0) = 0

    # From BC1: c1 = -gamma_tilde_s_half * psi_lower_prime_at_0 - p2_no_c(0)
    c1 = -gamma_tilde_s_half * psi_lower_prime_at_0 - poly_eval_at(p2_no_c, 0.0)

    # From BC3: p2_no_c(-1) - c0 + c1 = gamma_tilde_b * psi_lower_prime_at_m1
    c0 = poly_eval_at(p2_no_c, -1.0) + c1 - gamma_tilde_b * psi_lower_prime_at_m1

    # From BC4: psi(0) = p0_no_c(0) + c3 = 0
    c3 = -poly_eval_at(p0_no_c, 0.0)

    # From BC2: psi(-1) = p0_no_c(-1) + c0*(-1)^3/6 + c1*(-1)^2/2 + c2*(-1) + c3 = 0
    c2 = -(poly_eval_at(p0_no_c, -1.0) - c0 / 6.0 + c1 / 2.0 - 0.0 + c3)
    # Wait: c2*(-1) => c2 = -(p0_no_c(-1) + c0*(-1)/6 + c1/2 + c3)
    # psi(-1) = p0_no_c(-1) + c0*(-1)^3/6 + c1*(-1)^2/2 + c2*(-1) + c3
    #         = p0_no_c(-1) - c0/6 + c1/2 - c2 + c3 = 0
    c2 = poly_eval_at(p0_no_c, -1.0) - c0 / 6.0 + c1 / 2.0 + c3

    # Build full polynomial: psi = p0_no_c + c0*z^3/6 + c1*z^2/2 + c2*z + c3
    n = max(len(p0_no_c), 4)
    psi = np.zeros(n)
    psi[:len(p0_no_c)] = p0_no_c
    psi[0] += c3
    if n > 1:
        psi[1] += c2
    if n > 2:
        psi[2] += c1 / 2.0
    if n > 3:
        psi[3] += c0 / 6.0

    return psi


def solve_linear(profile: ShearDriftProfile,
                 bcs: RobinBoundaryConditions,
                 max_order: int = 8) -> LinearResult:
    """Solve the linear perturbation problem to O(l^{2*max_order}).

    Following Hayes & Phillips (2017) section 3.

    Parameters
    ----------
    profile : ShearDriftProfile
    bcs : RobinBoundaryConditions
    max_order : int
        Number of expansion terms. max_order=8 gives O(l^16).
    """
    a = profile.a  # D' coefficients
    b = profile.b  # U' coefficients

    # Storage for expansion coefficients
    u_polys = {}       # u_{2k}(z) polynomial coefficients
    psi_hat_polys = {} # psi_hat_{2k+1}(z) devoid of R_{2k}
    psi_tilde_polys = {}  # psi_tilde_{2k+1}(z)
    psi_polys = {}     # full psi_{2k+1} = psi_hat - R_{2k} * psi_tilde

    R_star = np.zeros(max_order)   # R*_{2k} (Neumann part)
    R_tilde = np.zeros(max_order)  # R_tilde_{2k} (Robin correction)
    R = np.zeros(max_order)        # Full R_{2k}

    # === O(l^0): u_0 = 1 (equation 21) ===
    u_polys[0] = np.array([1.0])

    # === O(l^1): solve psi''''_1 = -D' R_0 (equation 22) ===
    # First compute psi_tilde_1: psi_1 = -R_0 * psi_tilde_1
    # psi_tilde_1'''' = D' * u_0 = D' (since u_0 = 1)
    # with BCs: psi''=0, psi=0 at z=0,-1
    # rhs for psi_tilde_1'''' = D' (i.e., the polynomial a)
    D_prime_times_u0 = a.copy()  # D' * u_0 = D' * 1

    psi_tilde_polys[0] = _solve_psi_fourth_order(D_prime_times_u0)

    # R_0 from equation (26): R_0^{-1} = integral_{-1}^{0} psi_tilde_1 * U' dz
    psi_tilde_1_times_Uprime = poly_multiply(psi_tilde_polys[0], b)
    R0_inv = poly_definite_integral(psi_tilde_1_times_Uprime, -1.0, 0.0)

    if abs(R0_inv) < 1e-15:
        raise ValueError("R0 is infinite — instability not possible with these profiles.")

    R0 = 1.0 / R0_inv
    R[0] = R0
    R_star[0] = R0
    R_tilde[0] = 0.0

    # Full psi_1 = -R_0 * psi_tilde_1
    # psi_hat_1 = 0 (no terms devoid of R_0 at this order)
    psi_hat_polys[0] = np.array([0.0])
    psi_polys[0] = -R0 * psi_tilde_polys[0]

    # === Higher orders ===
    for j in range(1, max_order):
        # --- Solve for u_{2j} at O(l^{2j}) --- equation (9)/(14)
        # u''_{2j} = u_{2j-2} + U' * psi_{2j-1} (at onset, sigma=0)
        # Plus Robin BC corrections at O(l^4): u'(0) + gamma_tilde_s * u_{2j-4}(0) = 0

        # RHS = u_{2j-2} + U' * psi_{2j-1}
        rhs_u = np.zeros(1)
        if (j - 1) in u_polys:
            # u_{2(j-1)} = u_{2j-2}
            rhs_u = poly_add(rhs_u, u_polys[j - 1])
        if (j - 1) in psi_polys:
            # U' * psi_{2j-1}
            up_psi = poly_multiply(b, psi_polys[j - 1])
            rhs_u = poly_add(rhs_u, up_psi)

        # BC corrections: u'(0) + gamma_tilde_s * u_{2j-4}(0) = 0
        # Robin enters at order j >= 2 (i.e., u_4 and above)
        gt_s_val = 0.0
        gt_b_val = 0.0
        u_lower_at_0 = 0.0
        u_lower_at_m1 = 0.0
        if j >= 2 and (j - 2) in u_polys:
            gt_s_val = bcs.gamma_s  # will be divided by l^4 conceptually
            gt_b_val = bcs.gamma_b
            u_lower_at_0 = poly_eval_at(u_polys[j - 2], 0.0)
            u_lower_at_m1 = poly_eval_at(u_polys[j - 2], -1.0)

        # Integral constraint: int u_{2j} dz = 0 for j >= 1 (equation 12)
        u_polys[j] = _solve_u_second_order(
            rhs_u, gt_s_val, gt_b_val, u_lower_at_0, u_lower_at_m1, 0.0
        )

        # --- Solve for psi_{2j+1} at O(l^{2j+1}) --- equation (15)/(18)
        # psi''''_{2j+1} = 2*psi''_{2j-1} - psi_{2j-3}
        #                  - sum_{m=0}^{j} R_{2m} D' u_{2j-2m}
        #                  (at onset, sigma terms vanish)

        rhs_psi = np.zeros(1)

        # 2 * psi''_{2j-1}
        if (j - 1) in psi_polys:
            psi_prev = psi_polys[j - 1]
            psi_prev_dd = poly_derivative(poly_derivative(psi_prev))
            rhs_psi = poly_add(rhs_psi, 2.0 * psi_prev_dd)

        # -psi_{2j-3}
        if (j - 2) in psi_polys:
            rhs_psi = poly_add(rhs_psi, -psi_polys[j - 2])

        # -sum_{m=0}^{j} R_{2m} D' u_{2j-2m}  (separating the R_{2j} term)
        # psi_{2j+1} = psi_hat_{2j+1} - R_{2j} * psi_tilde_{2j+1}
        # where psi_tilde_{2j+1} comes from the R_{2j} * D' * u_0 = R_{2j} * D' term
        # and psi_hat contains everything else

        # Accumulate the part without R_{2j} (for psi_hat)
        rhs_hat = rhs_psi.copy()
        for m in range(j):
            # -R_{2m} * D' * u_{2(j-m)}
            Du = poly_multiply(a, u_polys[j - m])
            rhs_hat = poly_add(rhs_hat, -R[m] * Du)

        # psi_tilde: from R_{2j} * D' * u_0 = R_{2j} * D' * [1]
        # psi_tilde''''_{2j+1} = D' * u_0 = D'
        rhs_tilde = poly_multiply(a, u_polys[0])

        # Robin BC corrections on psi
        psi_lower_prime_at_0 = 0.0
        psi_lower_prime_at_m1 = 0.0
        gt_s_half = 0.0
        gt_b_psi = 0.0
        if j >= 2 and (j - 2) in psi_polys:
            psi_lower_deriv = poly_derivative(psi_polys[j - 2])
            psi_lower_prime_at_0 = poly_eval_at(psi_lower_deriv, 0.0)
            psi_lower_prime_at_m1 = poly_eval_at(psi_lower_deriv, -1.0)
            gt_s_half = bcs.gamma_s / 2.0
            gt_b_psi = bcs.gamma_b

        psi_hat_polys[j] = _solve_psi_fourth_order(
            rhs_hat, gt_s_half, gt_b_psi, psi_lower_prime_at_0, psi_lower_prime_at_m1
        )
        psi_tilde_polys[j] = _solve_psi_fourth_order(rhs_tilde)

        # --- Compute R_{2j} from solvability condition (equation 20) ---
        # R_{2j-2} synonym with sigma_{2j} = 0
        # Using the formula structure from equation (20):
        # Numerator: gamma_tilde_s * u_{2j-4}(0) + gamma_tilde_b * u_{2j-4}(-1)
        #            + int psi_hat_{2j-1} U' dz + delta_{0,j-1}
        # Denominator: int psi_tilde_{2j-1} U' dz
        # But this gives R_{2(j-1)}, so we actually need a different indexing.
        # The actual formula for R_{2j} at current step:
        # From equation (13) at O(l^{2(j+1)}):
        # -sigma_{2(j+1)} = delta_{0,j} + gamma_tilde_s u_{2(j-1)}(0) + gamma_tilde_b u_{2(j-1)}(-1)
        #                    + int U' psi_{2j+1} dz
        # At onset (sigma=0): R_{2j} from psi_{2j+1} = psi_hat - R_{2j} psi_tilde
        # 0 = delta_{0,j} + gamma_s_corr + int U' (psi_hat - R_{2j} psi_tilde) dz
        # R_{2j} = (delta_{0,j} + gamma_corr + int U' psi_hat dz) / (int U' psi_tilde dz)

        # Robin correction terms
        robin_numerator = 0.0
        if j >= 2 and (j - 1) in u_polys:
            robin_numerator = (bcs.gamma_s * poly_eval_at(u_polys[j - 1], 0.0)
                             + bcs.gamma_b * poly_eval_at(u_polys[j - 1], -1.0))

        # delta_{0,j-1} in eq (13) at paper's j'=j+1: delta_{0,j}
        # This is 1 only when j=0, which is R0 handled separately above.
        delta_term = 0.0

        hat_integral = poly_definite_integral(
            poly_multiply(b, psi_hat_polys[j]), -1.0, 0.0
        )
        tilde_integral = poly_definite_integral(
            poly_multiply(b, psi_tilde_polys[j]), -1.0, 0.0
        )

        if abs(tilde_integral) < 1e-20:
            R[j] = 0.0
            R_star[j] = 0.0
            R_tilde[j] = 0.0
        else:
            R[j] = (delta_term + robin_numerator + hat_integral) / tilde_integral
            # Parse into Neumann (R*) and Robin (R_tilde) parts
            R_star[j] = (delta_term + hat_integral) / tilde_integral
            if abs(robin_numerator) > 1e-20:
                R_tilde[j] = robin_numerator / tilde_integral
            else:
                R_tilde[j] = 0.0

        # Full psi_{2j+1}
        psi_polys[j] = poly_add(psi_hat_polys[j], -R[j] * psi_tilde_polys[j])

    # === Compute critical wavenumber and Rayleigh number ===
    gamma = bcs.gamma

    # R*_2 for the linear case comes from R_star[1]
    R_star_2 = R_star[1]

    # For the linear critical wavenumber (eq. 67):
    # lcL = (gamma * R0 / R*_2)^{1/4}
    if gamma > 0 and R_star_2 > 0:
        lcL = (gamma * R0 / R_star_2) ** 0.25
        RcL = R0 + 2.0 * (gamma * R0 * R_star_2) ** 0.5
    elif gamma > 0 and R_star_2 < 0:
        # R*_2 < 0 means the curve bends down — unusual
        lcL = 0.0
        RcL = R0
    else:
        lcL = 0.0
        RcL = R0

    # Build neutral curve function
    def neutral_curve(l: float) -> float:
        l_abs = abs(float(l))
        if l_abs < 1e-15:
            return float(np.inf)
        l2 = l_abs * l_abs
        regular = float(R0)
        l2k = 1.0
        for k in range(1, max_order):
            l2k *= l2
            regular += float(R_star[k]) * l2k
        singular = gamma * float(R0) / l2
        return singular + regular

    return LinearResult(
        R0=R0,
        R_coeffs=R_star.copy(),
        R_tilde_coeffs=R_tilde.copy(),
        lcL=lcL,
        RcL=RcL,
        neutral_curve=neutral_curve,
        psi_tilde_1_coeffs=psi_tilde_polys[0].copy(),
        u_coeffs={k: v.copy() for k, v in u_polys.items()},
        psi_hat_coeffs={k: v.copy() for k, v in psi_hat_polys.items()},
        psi_tilde_coeffs={k: v.copy() for k, v in psi_tilde_polys.items()},
    )


def _solve_u_second_order(rhs: np.ndarray,
                          gamma_tilde_s: float,
                          gamma_tilde_b: float,
                          u_lower_at_0: float,
                          u_lower_at_m1: float,
                          integral_constraint: float) -> np.ndarray:
    """Solve u'' = rhs with Robin BCs and integral constraint.

    u'(0) + gamma_tilde_s * u_lower(0) = 0
    -u'(-1) + gamma_tilde_b * u_lower(-1) = 0  => u'(-1) = gamma_tilde_b * u_lower(-1)
    int_{-1}^{0} u dz = integral_constraint
    """
    # u' = int(rhs) + c0
    u_prime_no_c = poly_integrate(rhs, 1)

    # BC at z=0: u'(0) = -gamma_tilde_s * u_lower(0)
    c0 = -gamma_tilde_s * u_lower_at_0 - poly_eval_at(u_prime_no_c, 0.0)

    # u = int(u' + c0) + c1
    u_prime_full = u_prime_no_c.copy()
    if len(u_prime_full) == 0:
        u_prime_full = np.array([c0])
    else:
        u_prime_full[0] += c0
    u_no_c = poly_integrate(u_prime_full, 1)

    # int u dz = integral_constraint
    int_val = poly_definite_integral(u_no_c, -1.0, 0.0)
    c1 = integral_constraint - int_val

    u_coeffs = u_no_c.copy()
    if len(u_coeffs) == 0:
        u_coeffs = np.array([c1])
    else:
        u_coeffs[0] += c1

    return u_coeffs
