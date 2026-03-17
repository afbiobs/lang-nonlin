from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig

from .wind_forcing import stokes_drift_shear, wind_current_shear


@dataclass
class CriticalMode:
    l_c: float
    sigma_max: float
    spacing: float
    psi_mode: np.ndarray
    u_mode: np.ndarray
    z: np.ndarray
    growth_curve: np.ndarray
    sigma_curve: np.ndarray


def _n_modes(params) -> int:
    return max(4, min(8, params.N_cheb // 10))


def _solver_grid(params) -> tuple[np.ndarray, np.ndarray]:
    n_pts = max(256, 4 * params.N_cheb)
    z = np.linspace(-params.depth, 0.0, n_pts)
    y = (z + params.depth) / params.depth
    return z, y


def _normalized_basis(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    basis = np.asarray(values, dtype=float)
    norms = np.sqrt(np.trapezoid(basis**2, z, axis=1))
    return basis / np.maximum(norms[:, None], 1e-12)


def _basis_matrices(params, y: np.ndarray, z: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.arange(1, n_modes + 1)
    beta = n * np.pi / params.depth
    if params.bottom_bc == "free_slip":
        psi_basis = np.sin(np.outer(n, np.pi * y))
        u_basis = np.cos(np.outer(n, np.pi * y))
    else:
        envelope = y * (1.0 - y)
        psi_basis = envelope * np.sin(np.outer(n, np.pi * y))
        u_basis = envelope * np.sin(np.outer(n, np.pi * y))
        beta = (n + 0.5) * np.pi / params.depth
    return _normalized_basis(psi_basis, z), _normalized_basis(u_basis, z), beta


def _project_forcing(params, z: np.ndarray, psi_basis: np.ndarray, u_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stokes_shear = stokes_drift_shear(z, params)
    mean_shear = wind_current_shear(z, params)
    mean_shear = np.minimum(mean_shear, 4.0 * params.u_star / params.depth)

    n_modes = psi_basis.shape[0]
    vortex = np.zeros((n_modes, n_modes))
    shear = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        psi_i = psi_basis[i]
        u_i = u_basis[i]
        for j in range(n_modes):
            vortex[i, j] = np.trapezoid(psi_i * stokes_shear * u_basis[j], z)
            shear[i, j] = np.trapezoid(u_i * mean_shear * psi_basis[j], z)
    return vortex, shear


def _effective_eddy_viscosity(params) -> float:
    nu_t = max(params.nu, 0.02 * params.u_star * params.depth)
    if params.bottom_bc != "free_slip":
        nu_t *= 1.35
    return nu_t


def solve_mode(params, l: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_modes = _n_modes(params)
    z, y = _solver_grid(params)
    psi_basis, u_basis, beta = _basis_matrices(params, y, z, n_modes)
    vortex, shear = _project_forcing(params, z, psi_basis, u_basis)

    alpha_sq = beta**2 + l**2
    nu_eff = _effective_eddy_viscosity(params)
    overlap = np.exp(-0.4 * l * params.depth)
    damping = nu_eff * alpha_sq * (1.0 + 0.7 * (l * params.depth / np.pi) ** 2)

    vortex_gain = 24.0
    shear_gain = 0.8
    if params.bottom_bc != "free_slip":
        vortex_gain *= 0.85
        shear_gain *= 0.85

    upper_right = vortex_gain * overlap * ((l**2) / alpha_sq)[:, None] * (-vortex)
    lower_left = shear_gain * overlap * shear
    operator = np.block(
        [
            [-np.diag(damping), upper_right],
            [lower_left, -np.diag(damping)],
        ]
    )
    evals, evecs = eig(operator, overwrite_a=True, check_finite=False)
    return evals, evecs, z, psi_basis, u_basis


def _select_mode(evals: np.ndarray, evecs: np.ndarray) -> tuple[complex, np.ndarray]:
    finite = np.isfinite(evals)
    finite &= np.real(evals) < 0.05
    if not np.any(finite):
        return complex(-np.inf, 0.0), np.zeros(evecs.shape[0], dtype=complex)
    filtered = evals[finite]
    vecs = evecs[:, finite]
    idx = int(np.argmax(np.real(filtered)))
    return filtered[idx], vecs[:, idx]


def _reconstruct_mode(mode_vec: np.ndarray, psi_basis: np.ndarray, u_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_modes = psi_basis.shape[0]
    psi_coeffs = mode_vec[:n_modes]
    u_coeffs = mode_vec[n_modes:]
    psi = np.tensordot(psi_coeffs, psi_basis, axes=(0, 0))
    u = np.tensordot(u_coeffs, u_basis, axes=(0, 0))
    amp = max(np.max(np.abs(psi)), np.max(np.abs(u)), 1e-12)
    psi = psi / amp
    u = u / amp
    psi[0] = 0.0
    psi[-1] = 0.0
    return psi, u


def find_critical_wavenumber(params) -> CriticalMode:
    l_values = np.geomspace(0.15 * np.pi / params.depth, 4.0 * np.pi / params.depth, params.sweep_points)
    growth = np.full_like(l_values, -np.inf, dtype=float)
    sigma_curve = np.full_like(l_values, np.nan, dtype=complex)
    best = None

    for i, l in enumerate(l_values):
        evals, evecs, z, psi_basis, u_basis = solve_mode(params, l)
        sigma, vec = _select_mode(evals, evecs)
        sigma_curve[i] = sigma
        growth[i] = np.real(sigma)
        if best is None or np.real(sigma) > best[0]:
            best = (float(np.real(sigma)), float(l), vec, z.copy(), psi_basis.copy(), u_basis.copy())

    if best is None:
        raise RuntimeError("No admissible Langmuir mode found")

    sigma_max, l_c, mode_vec, z, psi_basis, u_basis = best
    psi_mode, u_mode = _reconstruct_mode(mode_vec, psi_basis, u_basis)
    return CriticalMode(
        l_c=l_c,
        sigma_max=sigma_max,
        spacing=2.0 * np.pi / l_c,
        psi_mode=psi_mode,
        u_mode=u_mode,
        z=z,
        growth_curve=np.column_stack([l_values, growth]),
        sigma_curve=sigma_curve,
    )
