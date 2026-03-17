# Implementation Spec: Nonlinear Craik–Leibovich Langmuir Circulation Model for Shallow Lakes

**Replaces:** `langmuir_inverse_trend_addendum.md` and the linear eigenvalue solver + buoyancy visibility filter architecture it describes.

**Purpose:** The existing approach uses a linearised eigenvalue solver to predict LC cell spacing and then applies a post-hoc buoyancy visibility filter to explain the observed inverse relationship between wind speed and spacing in shallow lakes. This architecture is physically incorrect for shallow water. Hayes and Phillips (2017, *Geophys. Astrophys. Fluid Dyn.*, 111(1), 65–90) show that nonlinear steady states of the Craik–Leibovich equations predict critical wavenumbers 50–70% smaller (i.e. cells 1.4–1.9× wider) than linear theory, through a subcritical bifurcation caused by symmetry breaking. This nonlinear effect alone accounts for the observed aspect ratios (5–11) in shallow coastal waters. The buoyancy filter in the existing code is compensating for the wrong underlying physics.

This spec instructs you to replace the linear solver with a nonlinear CL-equation solver using Robin boundary conditions, realistic shear/drift profiles, and a wind-to-Rayleigh-number mapping for shallow lakes. The buoyancy coupling is retained but repositioned as a secondary refinement governing bloom visibility and amplification, not cell spacing.

**Key reference:** The attached PDF `GAFD2play.pdf` (Hayes and Phillips, 2017) is the primary theoretical source. All equation numbers, figure references and section references below refer to this paper. Read it carefully before coding.

---

## 0. Critical Context for the Coding Agent

### 0.1 Why the existing approach fails

The existing code solves the *linearised* CL-equations to get a growth rate curve σ_r(l), finds the fastest-growing wavenumber, and then applies a log-normal buoyancy visibility weight to shift the "observed" peak to lower wavenumbers at low wind. Three problems:

1. **The linear solver underestimates cell width by up to 2×.** Hayes and Phillips show that the nonlinear critical wavenumber lcNL is related to the linear lcL by a ratio κ = lcL/lcNL ∈ [1.4, 1.9], depending on the shear and drift profiles. The buoyancy filter is trying to recover this missing factor, but for the wrong reason.

2. **The eigenfunction amplitudes are wrong.** The existing code estimates downwelling as `w_down ∝ C_amp × u* × |ψ_max| × l × H` using linear eigenfunctions. But the nonlinear eigenfunctions include a spanwise-independent base-flow modification (a k=0 mode, see paper figure 6a curve i) that changes the velocity field structure. The linear `ψ_max` does not capture this.

3. **Robin boundary conditions are essential.** Without them, the critical wavenumber at onset is zero — there is no preferred spacing. The existing code may use Neumann conditions, which means it relies entirely on the buoyancy filter for wavenumber selection. Robin conditions (equations 2–3 of the paper) naturally select a finite lcNL > 0.

### 0.2 What replaces what

| Existing component | Replacement | Reason |
|---|---|---|
| Linear eigenvalue solver in `stability.py` | Nonlinear steady-state solver (Galerkin + Newton) | Correct physics for shallow water |
| Neumann boundary conditions | Robin boundary conditions with γ_s, γ_b | Required for finite onset wavenumber |
| Buoyancy visibility filter (`buoyancy_visibility_weight`) | Colony accumulation model (secondary) | Buoyancy governs visibility, not spacing |
| `find_visible_wavenumber` | `find_nonlinear_critical_wavenumber` | Spacing comes from nonlinear CL-equations |
| Fixed `C_amp = 0.01` amplitude scaling | Amplitudes from nonlinear eigenfunctions | Physically consistent |
| `spacing_visible` as primary prediction | `spacing_nonlinear` as primary, `spacing_visible` as secondary diagnostic | Physics carries the load |

### 0.3 Implementation language and dependencies

- Python 3.10+
- NumPy, SciPy (core numerics)
- SymPy (for the small-l asymptotic expansion, computer algebra steps)
- Matplotlib (diagnostics and validation plots)
- No new external dependencies required

---

## 1. Module Structure

Create the following module structure. Each file is described in its own section below.

```
langmuir/
├── __init__.py
├── profiles.py              # Shear U'(z) and drift D'(z) profile definitions
├── robin_bc.py              # Robin boundary condition implementation
├── linear_solver.py         # Linear perturbation solution (retained as baseline)
├── nonlinear_solver.py      # Nonlinear steady-state solver (NEW — core module)
├── galerkin.py              # Galerkin spectral method infrastructure
├── rayleigh_mapping.py      # Wind speed → Ra mapping for shallow lakes
├── colony_accumulation.py   # Buoyancy-mediated visibility (demoted from primary)
├── validation.py            # Validation pipeline
├── params.py                # Parameter classes
└── utils.py                 # Shared utilities (integration helpers, polynomial ops)
```

---

## 2. `params.py` — Parameter Definitions

### 2.1 Physical parameters

Define a dataclass `LCParams` with the following fields. Note the paper nondimensionalises by depth h, surface velocity U, eddy viscosity ν_T and maximum drift D, so these must all be computed from the physical forcing.

```python
@dataclass
class LCParams:
    """All parameters needed to specify a shallow-lake LC problem."""
    
    # --- Environmental forcing (dimensional) ---
    U10: float              # 10-m wind speed (m/s)
    depth: float            # Water depth h (m)
    fetch: float            # Wind fetch (m)
    
    # --- Derived nondimensional parameter ---
    Ra: float = field(init=False)  # Rayleigh number = U * D * h^2 / nu_T^2
    
    # --- Intermediate dimensional quantities (computed) ---
    u_star: float = field(init=False)       # Friction velocity (m/s)
    U_surface: float = field(init=False)    # Mean surface velocity (m/s)
    D_max: float = field(init=False)        # Maximum drift magnitude (m/s)
    nu_T: float = field(init=False)         # Representative eddy viscosity (m^2/s)
    
    # --- Boundary condition parameters ---
    gamma_s: float = 0.06      # Surface Robin parameter (Cox & Leibovich 1993)
    gamma_b: float = 0.28      # Bottom Robin parameter (rigid bottom)
    
    # --- Shear/drift profile specification ---
    shear_type: str = "wind_driven"    # or "tidal", "custom"
    drift_type: str = "fetch_limited"  # or "stokes_deep", "custom"
    
    # --- Colony properties (for secondary biological coupling) ---
    v_float: float = 1e-4       # Colony rise velocity (m/s)
    colony_radius: float = 250e-6  # Colony radius (m)
    rho_colony: float = 990.0   # Colony density (kg/m^3)
```

### 2.2 Computing Ra from wind speed

Implement `compute_derived_params(self)` as a `__post_init__` method. The chain is:

1. **Friction velocity:** `u_star = C_d^{1/2} × U10` where C_d ≈ 1.2 × 10^{-3} for sheltered lakes (low drag regime). For Lough Neagh specifically, use C_d = 1.0 × 10^{-3} for U10 < 5 m/s and the Charnock relation for higher winds.

2. **Surface velocity:** `U_surface ≈ 0.03 × U10` (3% rule for wind-driven surface currents in lakes; refine with a log-layer profile if desired).

3. **Drift magnitude:** For fetch-limited waves, use the Jonswap spectrum parameterisation. The significant wave height H_s and peak period T_p are functions of U10 and fetch. The Stokes drift at the surface for fetch-limited waves is:
   ```
   u_s(0) ≈ (2π / T_p) × (π × H_s / λ_p)
   ```
   where λ_p = g × T_p^2 / (2π) for deep-water dispersion. In shallow water (h < λ_p/2), apply the depth correction from Phillips et al. (2010). Set `D_max = u_s(0)`.

4. **Eddy viscosity:** Use a depth-averaged parabolic profile: `nu_T = κ × u_star × h / 6` where κ = 0.41 is von Kármán's constant. This gives a representative scalar value; the actual z-dependent profile enters the shear/drift computations.

5. **Rayleigh number:** `Ra = U_surface × D_max × depth^2 / nu_T^2`

**Validation check:** For Lough Neagh (h ≈ 9 m, fetch ≈ 15 km) at U10 = 3 m/s, Ra should be O(100–200), i.e. near the critical value R0 ≈ 120 for uniform shear/drift. At U10 = 8 m/s, Ra should be O(1000+), well into the supercritical regime.

---

## 3. `profiles.py` — Shear and Drift Profiles

### 3.1 Core interface

The paper expresses U'(z) and D'(z) as polynomial approximations (equation 7):

```
D' = Σ_{m=0}^{M} a_m z^m
U' = Σ_{n=0}^{N} b_n z^n
```

where z ∈ [-1, 0] (nondimensional, z=0 at surface, z=-1 at bottom). Implement a class:

```python
class ShearDriftProfile:
    """Polynomial representation of U'(z) and D'(z)."""
    
    def __init__(self, a_coeffs: np.ndarray, b_coeffs: np.ndarray):
        """
        a_coeffs: array of length M+1, coefficients for D'(z) = Σ a_m z^m
        b_coeffs: array of length N+1, coefficients for U'(z) = Σ b_n z^n
        """
        self.a = np.array(a_coeffs, dtype=float)
        self.b = np.array(b_coeffs, dtype=float)
        self.M = len(a_coeffs) - 1
        self.N = len(b_coeffs) - 1
    
    def D_prime(self, z): ...  # Evaluate D'(z)
    def U_prime(self, z): ...  # Evaluate U'(z)
```

### 3.2 Predefined profiles

Implement the following named profiles, all from the paper:

| Name | D' coefficients | U' coefficients | Use case |
|---|---|---|---|
| `uniform` | [1] | [1] | Baseline (Cox & Leibovich 1993) |
| `linear_drift` | [1, 1] | [1] | D' = 1+z, U' = 1 |
| `linear_shear` | [1] | [1, 1] | D' = 1, U' = 1+z |
| `both_linear` | [1, 1] | [1, 1] | D' = U' = 1+z |

Additionally implement a factory function `shallow_lake_profile(params: LCParams)` that computes realistic U'(z) and D'(z) for a wind-driven shallow lake, following Phillips et al. (2010). The procedure is:

1. For the shear profile: assume a wind-stress-driven flow with a log-layer near the surface and a return current. The nondimensional U'(z) for a wind-driven lake has its maximum near the surface (z=0) and becomes negative near the bottom (return flow). Approximate with a polynomial fit to degree N=3 or N=4.

2. For the drift profile: use the shallow-water Stokes drift from Phillips et al. (2010), which differs substantially from the classical exponential Stokes drift. In shallow water the drift is affected by the bottom boundary and can be nearly uniform over depth. Compute from the wave parameters in `LCParams` and fit a polynomial of degree M=3 or M=4.

3. **Critical constraint:** The instability requires D'U' > 0 (Leibovich 1983). Verify this condition holds over the domain. If it fails at some depths (e.g. near the bottom where the shear reverses), flag a warning and truncate the profile or note that LC is bottom-limited.

### 3.3 Polynomial integration helpers

The asymptotic expansion in sections 3–4 of the paper requires repeated integration of polynomial expressions. Implement in `utils.py`:

```python
def poly_integrate(coeffs, z, n_times=1):
    """
    Integrate a polynomial Σ c_k z^k with respect to z, n_times.
    Returns coefficients of the resulting polynomial.
    Constants of integration are set to zero (they are determined
    separately by boundary conditions in the solver).
    """
```

Also implement the iterated integral notation from the paper:

```python
def iterated_integral(f_z, z_grid, n_times, z_lower=-1, z_upper=0):
    """
    Compute ∫∫...∫ f(z) dz^n numerically on the grid z_grid.
    Uses cumulative trapezoidal integration.
    """
```

---

## 4. `robin_bc.py` — Robin Boundary Conditions

### 4.1 Theory

The boundary conditions from the paper (equations 2–3) are mixed (Robin) type:

**At z = 0 (free surface):**
```
u_z + γ_s u = 0                    (for the streamwise perturbation)
ψ_zz + (γ_s / 2) ψ_z = 0          (for the streamfunction)
ψ = 0
```

**At z = -1 (rigid bottom):**
```
-u_z + γ_b u = 0
-ψ_zz + γ_b ψ_z = 0
ψ = 0
```

The critical scaling from the paper (equation 6) is:

```
(γ_s, γ_b) = l^4 (γ̃_s, γ̃_b)     with γ = γ_s + γ_b
```

This means the Robin parameters enter at O(l^4) in the small-l expansion, and the boundary conditions reduce to Neumann at lower orders.

### 4.2 Implementation

```python
class RobinBoundaryConditions:
    """
    Robin boundary conditions for the CL-equations in a shallow layer.
    
    The paper shows that Robin BCs are essential for selecting a finite
    onset wavenumber. With Neumann BCs (gamma=0), lc=0 regardless
    of nonlinearities (Chapman & Proctor 1980).
    """
    
    def __init__(self, gamma_s: float = 0.06, gamma_b: float = 0.28):
        self.gamma_s = gamma_s
        self.gamma_b = gamma_b
        self.gamma = gamma_s + gamma_b
    
    def gamma_tilde_s(self, l: float) -> float:
        """Rescaled surface parameter: γ̃_s = γ_s / l^4"""
        if l == 0:
            return np.inf  # handled separately in the expansion
        return self.gamma_s / l**4
    
    def gamma_tilde_b(self, l: float) -> float:
        """Rescaled bottom parameter: γ̃_b = γ_b / l^4"""
        if l == 0:
            return np.inf
        return self.gamma_b / l**4
    
    def apply_u_surface(self, u, u_z, l, order):
        """Apply BC: u_z + γ̃_s u_{order-4} = 0 at z=0"""
        ...
    
    def apply_u_bottom(self, u, u_z, l, order):
        """Apply BC: u_z - γ̃_b u_{order-4} = 0 at z=-1"""
        ...
    
    def apply_psi_surface(self, psi, psi_z, psi_zz, l, order):
        """Apply BC: ψ_zz + (γ̃_s/2) ψ_{order-4,z} = 0, ψ = 0 at z=0"""
        ...
    
    def apply_psi_bottom(self, psi, psi_z, psi_zz, l, order):
        """Apply BC: -ψ_zz + γ̃_b ψ_{order-4,z} = 0, ψ = 0 at z=-1"""
        ...
```

---

## 5. `linear_solver.py` — Linear Perturbation Solution (Baseline)

### 5.1 Purpose

Retain the linear solution as a baseline for comparison. Implement it following the paper's algorithm (section 3) so that results are exactly comparable with the nonlinear solver. This is **not** the existing linear eigenvalue solver — it must be reimplemented to use the same framework (small-l expansion, Robin BCs, polynomial profiles).

### 5.2 Algorithm

Follow the paper's section 3.1 exactly. The algorithm proceeds order by order in l:

**At O(l^{2j}) for j ≥ 0:** Solve equation (9) for u_{2j} with BCs (10–11). The solvability condition (equation 13) gives σ_{2j}. At onset (σ_{2j} = 0), equation (14) gives u_{2j}.

**At O(l^{2j+1}):** Solve equation (15) for ψ_{2j+1} with BCs (16–17). Decompose as ψ_{2j+1} = ψ̂_{2j+1} - R_{2j} ψ̃_{2j+1} (equation 19).

**Neutral curve:** The expansion coefficients R_{2k} follow from equation (20). The neutral curve is R(l) = Σ l^{2k} R_{2k} (equation 5b).

**Critical wavenumber:** From equations (67) and (70):
```
lcL = (γ R0 / R*_2)^{1/4}
RcL = R0 + 2 (γ R0 R*_2)^{1/2}
```

### 5.3 Specific leading-order results (section 3.2)

These are needed by the nonlinear solver. At O(l^0):

```
u_0 = 1                                           (equation 21)
```

At O(l):

```
ψ''''_1 = -D' R_0                                 (equation 22)
ψ_1 = -R_0 ψ̃_1                                   (equation 24, with BCs 23)
```

The critical R_0 is determined by equation (26):

```
R_0^{-1} = ∫_{-1}^{0} ψ̃_1 U' dz
```

**Implement to O(l^16)** as stated in the paper (section 7, first paragraph). Use SymPy for the symbolic computation of the polynomial integrals at each order, then convert to numerical evaluation.

### 5.4 Outputs

```python
@dataclass
class LinearResult:
    R0: float                          # Leading-order Rayleigh number
    R_coeffs: np.ndarray               # Array [R0, R2, R4, ...] to O(l^16)
    lcL: float                         # Critical linear wavenumber
    RcL: float                         # Critical linear Rayleigh number
    neutral_curve: Callable[[float], float]  # R(l) from the expansion
    psi_tilde_1: np.ndarray            # ψ̃_1(z) on a grid — needed by nonlinear solver
    u_eigenfunctions: dict             # {order: u_{2k}(z)} on a grid
    psi_eigenfunctions: dict           # {order: ψ_{2k+1}(z)} on a grid
```

---

## 6. `nonlinear_solver.py` — Nonlinear Steady-State Solver (CORE MODULE)

This is the central new module. It implements two complementary approaches from the paper.

### 6.1 Approach A: Small-l asymptotic expansion (sections 4–5)

This follows the paper's nonlinear perturbation solution. It extends the linear expansion to include the Jacobian nonlinearities J(ψ, u) and J(ψ, ∇²ψ) in equations (1b,c).

#### 6.1.1 Rescaled equations

The rescaling Y = ly, T = l²t (equation 27) converts the CL-equations to (29–30). The expansion is:

```
ũ = Σ_{k=0}^∞ l^{2k} u_{2k}(Y, z, T)
Ψ̃ = Σ_{k=0}^∞ l^{2k} Ψ_{2k}(Y, z, T)
Ra = R = Σ_{k=0}^∞ l^{2k} R_{2k}
```

Note the change of notation from the linear case: the nonlinear u_{2k} and Ψ_{2k} are functions of Y and T, not just z.

#### 6.1.2 Order-by-order solution

**O(l^0):** Equation (41–42) gives u_0 = u_0(Y,T) — a function independent of z. This is equation (43).

**O(l^1):** Equation (44–45) gives Ψ_0 = -R_0 ψ̃_1 ∂u_0/∂Y. This is equation (46).

**O(l^2) — first nonlinear order:** Equation (47) with BCs (48). Integration over z (applying BCs to eliminate u_2) gives the evolution equation for u_0:

```
∂u_0/∂T + σ_2 ∂²u_0/∂Y² = 0                     (equation 51)
```

where σ_2 is the linear growth rate at O(l²). The nonlinear steady states at this order coincide with linear neutral stability.

**Solution for u_0:** By separation of variables (equation 52):
```
u_0 = Σ_{m=0}^∞ h_m e^{σ_2 m² T} cos(mY)
```

**O(l^2) continued — solving for u_2:** Equation (54) gives u_2 in terms of u_0 and Ψ_0.

**O(l^3):** Equation (55–56) gives Ψ_2 in terms of lower-order quantities. Decompose as Ψ_2 = Ψ̂_2 - R_2 Ψ̃_2 (equation 58).

#### 6.1.3 Nonlinear steady states and R_2

The key result is equation (63), which determines R_2 at the nonlinear steady state (∂/∂T = 0). Parse R_2 into:

```
R_2 = R*_2 + γ̃ R̃_2                               (equation 64)
```

where R*_2 is the γ=0 (Neumann) part and γ̃ R̃_2 is the Robin correction.

**This is the central calculation.** R*_2 differs from its linear counterpart because the nonlinear eigenfunctions contribute. Equation (63) contains four integrals involving the nonlinear Ψ̂_2, u_2, Ψ̃_2 and Ψ_0 — all of which must be computed carefully.

#### 6.1.4 Critical nonlinear wavenumber

From equations (65–66):

```
R̄(l) = R_0 + l² R*_2 + (γ / l²) R̃_2 + ...      (equation 65)

lcNL = (γ R̃_2 / R*_2)^{1/4}                       (equation 66)
```

The ratio:
```
κ = lcL / lcNL = (R_0 R*_2 / (R*_2 R̃_2))^{1/4}   (equation 68)
```

is **independent of γ**. This is the key result: κ depends only on the shear/drift profiles.

#### 6.1.5 Spectral decomposition

Each u_{2k} and Ψ_{2k} contains harmonics in Y (equations 59–60):

```
u_{2k}(Y,z,T) = Σ_m u_{2k,m}(z,T) cos(mY)
Ψ_{2k}(Y,z,T) = Σ_m Ψ_{2k,m}(z,T) sin(mY)
```

Nonlinear terms produce higher harmonics. Apply the theorem in Appendix C of the paper: retain only harmonics up to the order of the expansion, discard higher ones.

#### 6.1.6 Implementation notes

- Use SymPy to symbolically compute the polynomial integrals at each order, then lambdify for numerical evaluation.
- Carry the expansion to O(l^16) to match the linear solver.
- At each order, the constants of integration are determined by BCs and the normalisation u_{2k,m}|_{z=0} = δ_{2k,0} δ_{m,1}.
- Test by reproducing Table-equivalent values from the paper: for D'=U'=1, κ ≈ 1.425; for D'=U'=1+z, κ ≈ 1.944.

### 6.2 Approach B: Galerkin numerical solver (section 6)

This is the complementary numerical method that validates and extends the asymptotic expansion. It solves the full nonlinear CL-equations directly.

#### 6.2.1 Spectral expansion

Expand u and ψ in shifted Legendre polynomials P_m(z) on z ∈ [-1, 0] (equations 71–72):

```python
# u expansion: J-2 Legendre modes × I+1 Fourier harmonics
u = Σ_{m=0}^{J-2} Σ_{k=0}^{I} A_{m,k}(t) P_m(z) cos(k l y)

# ψ expansion: J Legendre modes × I+1 Fourier harmonics
ψ = Σ_{m=0}^{J} Σ_{k=0}^{I} B_{m,k}(t) P_m(z) sin(k l y)
```

The paper uses I=1 (fundamental + first harmonic) and J=13.

#### 6.2.2 Galerkin projection

1. Substitute the expansions into the CL-equations (1b,c).
2. Collect trigonometric terms (cos(kly), sin(kly)).
3. Discard harmonics above order I (per Appendix C theorem).
4. Form residuals r_{1,i}(z,t) and r_{2,i}(z,t).
5. Project onto the Legendre basis via inner products (equation 73):
   ```
   ∫_{-1}^{0} r_{1,i} P_j dz = 0    for j = 0,...,J-4
   ∫_{-1}^{0} r_{2,i} P_j dz = 0    for j = 0,...,J-4
   ```
6. The remaining equations come from substituting the expansion into the BCs (2–3).

#### 6.2.3 Newton solver for steady states

For steady states (∂/∂t = 0), the ODE system becomes algebraic:

```python
def galerkin_residual(x, l, Ra, profile, bcs):
    """
    Compute the residual F(x) = 0 for the nonlinear steady state.
    
    x: flattened array of [A_{m,k}, B_{m,k}] coefficients
    l: spanwise wavenumber
    Ra: Rayleigh number
    profile: ShearDriftProfile
    bcs: RobinBoundaryConditions
    
    Returns: residual vector, same length as x
    """
    # Unpack x into A and B coefficient matrices
    A, B = unpack_coefficients(x, I, J)
    
    # Evaluate Legendre basis and derivatives on quadrature grid
    z_quad, w_quad = gauss_legendre_quadrature(n_points=32, a=-1, b=0)
    P, dP, d2P, d3P, d4P = legendre_basis_and_derivatives(J, z_quad)
    
    # Reconstruct u, ψ and their derivatives on (z_quad, k) grid
    # Compute nonlinear terms (Jacobians)
    # Form residuals for each (i, j) pair
    # Include BC equations
    
    return residual_vector


def find_nonlinear_steady_state(l, Ra, profile, bcs, x0=None):
    """
    Solve for the nonlinear steady state at given (l, Ra).
    
    Uses adaptive Newton's method with line search.
    If x0 is None, initialise from the linear eigenfunction.
    """
    if x0 is None:
        x0 = initialise_from_linear(l, Ra, profile, bcs)
    
    result = scipy.optimize.root(
        galerkin_residual,
        x0,
        args=(l, Ra, profile, bcs),
        method='hybr',  # Powell's hybrid (equivalent to Newton with dogleg)
        options={'maxfev': 5000}
    )
    
    if not result.success:
        raise ConvergenceError(f"Newton failed at l={l}, Ra={Ra}: {result.message}")
    
    return result.x
```

#### 6.2.4 Computing the nonlinear neutral curve R̄(l)

For each l, find the minimum Ra at which a nontrivial steady state exists:

```python
def nonlinear_neutral_curve(l_array, profile, bcs, Ra_linear_curve):
    """
    Compute R̄(l) — the nonlinear neutral curve.
    
    Strategy: for each l, start at Ra slightly above the linear neutral curve
    and binary-search downward to find the minimum Ra that supports a
    nontrivial steady state.
    
    The linear neutral curve Ra_linear_curve provides the starting point.
    R̄(l) ≥ R(l) for all l (supercritical stability).
    """
    R_bar = np.zeros_like(l_array)
    
    for i, l in enumerate(l_array):
        R_linear = Ra_linear_curve(l)
        
        # Start above linear curve and search for onset
        Ra_hi = R_linear * 1.5
        Ra_lo = R_linear * 0.99  # Just below linear
        
        # Binary search for the minimum Ra with nontrivial solution
        for _ in range(50):  # sufficient iterations for convergence
            Ra_mid = (Ra_hi + Ra_lo) / 2
            try:
                x = find_nonlinear_steady_state(l, Ra_mid, profile, bcs)
                amplitude = np.max(np.abs(x))
                if amplitude > 1e-8:
                    Ra_hi = Ra_mid  # nontrivial solution exists, try lower
                else:
                    Ra_lo = Ra_mid  # trivial solution, try higher
            except ConvergenceError:
                Ra_lo = Ra_mid  # no solution, try higher
        
        R_bar[i] = Ra_hi
    
    return R_bar
```

#### 6.2.5 Normalisation

For consistency with the asymptotic expansion, specify A_{0,1} such that the coefficient of cos(ly) in u|_{z=0} is unity. See paper section 6 last paragraph.

### 6.3 Outputs

```python
@dataclass
class NonlinearResult:
    R0: float                          # Leading-order Rayleigh number (same as linear)
    R_bar_coeffs: np.ndarray           # [R0, R̄_2, R̄_4, ...] — nonlinear expansion coeffs
    lcNL: float                        # Critical nonlinear wavenumber
    RcNL: float                        # Critical nonlinear Rayleigh number
    kappa: float                       # Ratio κ = lcL / lcNL
    aspect_ratio: float                # 2π / lcNL (nondimensional)
    neutral_curve_NL: Callable         # R̄(l) from asymptotic expansion
    neutral_curve_NL_numeric: Callable # R̄(l) from Galerkin solver
    eigenfunctions_NL: dict            # Nonlinear eigenmodes u^k_e(z), Ψ^k_e(z)
    base_flow_modification: np.ndarray # k=0 mode u^0_e(z) — the nonlinear correction
```

---

## 7. `galerkin.py` — Spectral Method Infrastructure

### 7.1 Shifted Legendre basis

Implement shifted Legendre polynomials on z ∈ [-1, 0]:

```python
def shifted_legendre(n, z):
    """
    Shifted Legendre polynomial P_n on [-1, 0].
    Use the standard recursion relation with the affine map z → 2z + 1.
    """

def shifted_legendre_derivatives(n, z, max_deriv=4):
    """
    Return P_n and its first through max_deriv-th derivatives at z.
    Computed analytically from the recursion.
    """
```

### 7.2 Inner products and quadrature

```python
def inner_product(f_values, g_values, weights):
    """∫_{-1}^{0} f(z) g(z) dz via Gauss-Legendre quadrature."""
    return np.sum(weights * f_values * g_values)

def mass_matrix(J, z_quad, w_quad):
    """M_{ij} = ∫ P_i P_j dz — precompute for efficiency."""

def stiffness_matrix(J, z_quad, w_quad):
    """K_{ij} = ∫ P'_i P'_j dz"""
```

### 7.3 Trigonometric product rules

Nonlinear terms in the CL-equations produce products of cos(kly) and sin(kly). Implement the standard identities:

```python
def trig_product_rules():
    """
    cos(a) cos(b) = 0.5 [cos(a-b) + cos(a+b)]
    sin(a) sin(b) = 0.5 [cos(a-b) - cos(a+b)]
    sin(a) cos(b) = 0.5 [sin(a+b) + sin(a-b)]
    
    For the Jacobian terms J(ψ, u) and J(ψ, ∇²ψ), the products of
    harmonics k1 and k2 produce harmonics |k1-k2| and k1+k2.
    Discard k1+k2 if > I (Appendix C theorem).
    """
```

---

## 8. `rayleigh_mapping.py` — Wind Speed to Rayleigh Number

### 8.1 Purpose

Map environmental forcing (U10, h, fetch) to the nondimensional Rayleigh number and determine the instability regime.

### 8.2 Core function

```python
def wind_to_rayleigh(U10: float, depth: float, fetch: float,
                     drag_model: str = "lake_low") -> dict:
    """
    Compute Ra and all intermediate quantities from wind forcing.
    
    Returns dict with:
        Ra: float — Rayleigh number
        u_star: float — friction velocity (m/s)
        U_surface: float — surface current (m/s)
        D_max: float — peak drift (m/s)
        nu_T: float — eddy viscosity (m^2/s)
        H_s: float — significant wave height (m)
        T_p: float — peak wave period (s)
        lambda_p: float — peak wavelength (m)
        La_t: float — turbulent Langmuir number = sqrt(u_star / u_s(0))
    """
```

### 8.3 Regime classification

```python
def classify_regime(Ra: float, R0: float, RcNL: float) -> str:
    """
    Classify the LC regime based on the Rayleigh number.
    
    Returns:
        "subcritical"  — Ra < R0, no LC possible
        "near_onset"   — R0 ≤ Ra < 1.5 × RcNL, narrow band of unstable modes
        "moderate"     — 1.5 × RcNL ≤ Ra < 5 × RcNL
        "supercritical" — Ra ≥ 5 × RcNL, broad unstable spectrum
    
    The "near_onset" regime is where the linear/nonlinear divergence
    is greatest and where the low-wind bloom conditions arise.
    """
```

### 8.4 Unstable wavenumber band

```python
def unstable_band(Ra: float, neutral_curve_NL: Callable,
                  l_array: np.ndarray) -> tuple[float, float]:
    """
    Find the range [l_min, l_max] where Ra > R̄(l).
    These are the wavenumbers that are supercritically unstable.
    
    At low wind (Ra near RcNL), this band is narrow and centred on lcNL.
    At high wind (Ra >> RcNL), this band is broad.
    
    Returns (l_min, l_max). If Ra < min(R̄), returns (nan, nan).
    """
```

---

## 9. `colony_accumulation.py` — Biological Coupling (Secondary)

### 9.1 Repositioned role

The buoyancy coupling is retained but demoted from primary spacing predictor to secondary refinement. It now answers two questions:

1. **Is the LC pattern visible from satellite?** (Given the nonlinear cell width and amplitude, does surface accumulation occur?)
2. **Does surface accumulation trigger bloom amplification?** (Positive feedback through increased light/warmth.)

It does NOT determine cell spacing. Spacing comes from the nonlinear CL solver.

### 9.2 Surface accumulation diagnostic

```python
def surface_accumulation_index(
    l: float,
    Ra: float,
    eigenfunctions: dict,
    v_float: float,
    depth: float,
    u_star: float,
) -> dict:
    """
    Compute whether LC at wavenumber l produces visible surface accumulation.
    
    Uses the nonlinear eigenfunction Ψ(z) to estimate downwelling velocity
    at the convergence zone:
        w_down_max = (Ra - RcNL)^{1/2} × |dΨ/dy|_max
    (finite-amplitude scaling from weakly nonlinear theory).
    
    The key ratio is:
        S = w_down_max / v_float
    
    - S >> 1: colonies subducted, weak surface signal
    - S << 1: colonies uniformly spread, no convergence signal
    - S ~ 0.3–1.0: colonies float but are herded → visible windrows
    
    Returns:
        ratio_S: float
        is_visible: bool (True if 0.1 < S < 3.0)
        accumulation_factor: float (peaked at S ≈ 0.5)
    """
```

### 9.3 Bloom amplification feedback

```python
def bloom_feedback_potential(
    accumulation_factor: float,
    surface_residence_time: float,
    light_enhancement: float = 1.5,
) -> float:
    """
    Estimate the positive feedback strength for bloom development.
    
    When LC cells are wide and weak (near onset, low wind):
    - Surface residence time is long (colonies not subducted)
    - Light exposure increases → growth rate increases
    - Colony size increases → v_float increases → harder to subduct
    - Positive feedback loop
    
    Returns a dimensionless feedback index ∈ [0, 1].
    Above ~0.3 indicates conditions favourable for bloom intensification.
    """
```

### 9.4 Integration with main pipeline

```python
def predict_spacing_and_visibility(params: LCParams) -> dict:
    """
    Full prediction pipeline:
    
    1. Compute Ra from wind forcing (rayleigh_mapping)
    2. Solve nonlinear CL-equations (nonlinear_solver)
    3. Find the fastest-growing mode in the unstable band
    4. Assess surface visibility (colony_accumulation)
    5. Compute bloom feedback potential
    
    Returns:
        spacing_nonlinear: float — primary prediction from nonlinear CL physics (m)
        spacing_linear: float — baseline linear prediction for comparison (m)
        kappa: float — ratio lcL / lcNL
        regime: str — "subcritical", "near_onset", etc.
        is_visible: bool — whether the pattern is satellite-observable
        accumulation_factor: float — strength of surface accumulation
        bloom_feedback: float — positive feedback index
        w_down_max: float — peak downwelling velocity (m/s)
        Ra: float — Rayleigh number
    """
```

---

## 10. `validation.py` — Validation Pipeline

### 10.1 Retain from existing code

Keep the following checks from the addendum — they are well-designed and independent of the underlying model:

- **Check 1 (confounders):** `check_wind_spacing_confounders` — partial correlation controlling for depth, within-location analysis. No changes needed.
- **Check 2 (observation quality):** `check_observation_quality_bias` — spacing spread by wind bin, geographic concentration. No changes needed.

### 10.2 Replace the theoretical consistency envelope

Replace Check 3 with a nonlinear version:

```python
def nonlinear_consistency_envelope(results_df, depths=[5, 9, 15]):
    """
    For each depth, compute the NONLINEAR predicted spacing across a
    wind speed range. The primary prediction is spacing_nonlinear from
    the CL-equation solver, not the buoyancy-filtered version.
    
    Also compute the biological visibility mask to indicate which
    wind speeds produce satellite-observable patterns.
    
    Overlay all observations on the theoretical envelope.
    
    Expected behaviour:
    - At low wind: narrow unstable band near lcNL → wide cells → large spacing
    - At high wind: broad unstable band → fastest mode dominates → moderate spacing
    - The inverse spacing-wind relationship emerges from the hydrodynamics
    """
    wind_range = np.linspace(1.5, 15.0, 50)
    
    envelopes = {}
    for depth in depths:
        spacings_NL = []
        spacings_L = []
        is_visible = []
        Ra_values = []
        regimes = []
        
        for U10 in wind_range:
            params = LCParams(U10=U10, depth=depth, fetch=15000.0)
            result = predict_spacing_and_visibility(params)
            
            spacings_NL.append(result["spacing_nonlinear"])
            spacings_L.append(result["spacing_linear"])
            is_visible.append(result["is_visible"])
            Ra_values.append(result["Ra"])
            regimes.append(result["regime"])
        
        envelopes[depth] = {
            "wind": wind_range,
            "spacing_NL": np.array(spacings_NL),
            "spacing_L": np.array(spacings_L),
            "is_visible": np.array(is_visible),
            "Ra": np.array(Ra_values),
            "regimes": regimes,
        }
    
    return envelopes
```

### 10.3 New diagnostic: nonlinear vs linear divergence

```python
def kappa_diagnostic(results_df):
    """
    For each observation, compute κ = lcL / lcNL and plot against wind speed.
    
    κ should be approximately constant (depends on profiles, not wind),
    but the *dimensional* spacing divergence grows at low wind because
    both lcL and lcNL scale as γ^{1/4}, and the absolute spacing is
    2π h / lc (dimensional = nondimensional × depth).
    
    Key plot: observed_spacing / predicted_spacing_NL vs U10.
    If the nonlinear physics is correct, this ratio should be ~1 at all
    wind speeds. Any residual trend is attributable to biology.
    """
```

### 10.4 Updated cumulative improvement table

```
| Configuration                                | RMSE (m) | R²   | Capture Rate | Notes                     |
|----------------------------------------------|----------|------|--------------|---------------------------|
| Old baseline (linear solver, constant ν_T)   | ...      | ...  | ...          | Previous architecture     |
| Old + buoyancy visibility filter             | ...      | ...  | ...          | Previous best             |
| Nonlinear CL solver, uniform profiles        | ...      | ...  | ...          | New baseline              |
| + Robin BCs (γ_s=0.06, γ_b=0.28)            | ...      | ...  | ...          | Essential for lc > 0      |
| + Realistic shear/drift profiles             | ...      | ...  | ...          | Lake-specific physics     |
| + Parabolic ν_T                              | ...      | ...  | ...          |                           |
| + Spectral Stokes drift                      | ...      | ...  | ...          |                           |
| + Colony visibility diagnostic               | ...      | ...  | ...          | Secondary refinement      |
| + Depth ensemble (capture rate)              | —        | —    | ...          |                           |
```

### 10.5 Validation figures

```
outputs/validation/figures/
├── neutral_curves_linear_vs_nonlinear.png     # R(l) and R̄(l) for each profile
├── kappa_vs_profiles.png                       # κ for each U'/D' combination
├── Ra_trajectory_vs_wind.png                   # Ra(U10) overlaid on R̄(l) contours
├── unstable_band_vs_wind.png                   # [l_min, l_max] as function of U10
├── spacing_vs_wind_NL.png                      # Primary validation: NL prediction vs obs
├── spacing_vs_wind_residual.png                # Observed/predicted ratio vs U10
├── eigenfunctions_linear_vs_nonlinear.png      # u^k_e(z), Ψ^k_e(z) comparison (paper fig 6)
├── base_flow_modification.png                  # k=0 mode (nonlinear correction)
├── confounder_scatter.png                       # Retained from addendum
├── partial_correlation.png                      # Retained from addendum
├── within_location_slopes.png                   # Retained from addendum
├── wind_bin_summary.png                         # Retained from addendum
├── visibility_mask_vs_wind.png                  # Where biology says pattern is observable
├── bloom_feedback_map.png                       # Feedback index as f(U10, depth)
```

---

## 11. Numerical Verification Targets

Before running validation against observations, verify the solver reproduces the paper's results. These are pass/fail tests.

### 11.1 Leading-order Rayleigh number

For all profiles with Robin BCs (γ_s = 0.0001, γ_b = 0), the paper gives R0 from equation (26). For D' = U' = 1:

```
R0 = 1 / ∫_{-1}^{0} ψ̃_1 dz
```

For uniform D' = U' = 1, R0 ≈ 120 (implied by paper figure 6 caption: RcL ≈ 121.068). **Test:** Compute R0 and verify it matches to 4 significant figures.

### 11.2 Critical wavenumbers (figure 6 caption)

For D' = U' = 1, γ_s = 0.0001, γ_b = 0:

```
lcL ≈ 0.150
RcL ≈ 121.068
lcNL ≈ 0.105
RcNL ≈ 122.194
```

**Test:** Reproduce all four values. The ratio κ = 0.150/0.105 ≈ 1.429, consistent with the paper's κ ≈ 1.425 for this profile.

### 11.3 κ values (section 7.2)

For (D', U') = (1,1), (1+z, 1), (1, 1+z), (1+z, 1+z):

```
κ ≈ (1.425, 1.427, 1.919, 1.944)
```

**Test:** Reproduce all four to 3 significant figures.

### 11.4 Neutral curve shape

**Test:** Reproduce figures 1 and 2 of the paper. Specifically:

- The nonlinear neutral curve R̄(l) lies above R(l) at all l > 0.
- R̄(l) → R(l) as l → 0 for Neumann BCs (figure 1).
- R̄(l) > R(l) at all l (including l → 0) for Robin BCs (figure 2).
- The gap Δ = R̄ - R increases with l (figures 3, 4).

### 11.5 Aspect ratio range

With γ_s = 0.06, γ_b = 0.28 (Cox & Leibovich values):

```
lcNL ∈ [0.57, 1.24] depending on profiles
Aspect ratio L = 2π/lcNL ∈ [5, 11]
```

**Test:** Verify this range is recovered and matches the paper's section 8 discussion.

### 11.6 Asymptotic vs numerical agreement

**Test:** The O(l^16) asymptotic expansion and the Galerkin numerical solver must produce indistinguishable results over a region well in excess of lcNL (paper section 7, first paragraph).

---

## 12. Implementation Order

Follow this sequence. Each step must pass its verification tests before proceeding.

### Phase 1: Core infrastructure
1. `params.py` — parameter classes
2. `utils.py` — polynomial integration helpers
3. `profiles.py` — shear/drift profile definitions (uniform, linear variants)
4. `robin_bc.py` — Robin boundary condition machinery
5. `galerkin.py` — Legendre basis, inner products, trig product rules

### Phase 2: Linear baseline
6. `linear_solver.py` — implement the small-l expansion to O(l^16)
7. **Verify:** R0, lcL, RcL match section 11.1–11.2 targets

### Phase 3: Nonlinear solver
8. `nonlinear_solver.py` Approach A — small-l asymptotic expansion with nonlinearities
9. **Verify:** κ values match section 11.3 targets
10. **Verify:** neutral curve shape matches section 11.4

### Phase 4: Numerical validation of asymptotics
11. `nonlinear_solver.py` Approach B — Galerkin + Newton solver
12. **Verify:** asymptotic and numerical results agree (section 11.6)

### Phase 5: Environmental mapping
13. `rayleigh_mapping.py` — wind-to-Ra chain
14. **Verify:** Ra values are physically sensible (section 2.2 of this spec)

### Phase 6: Biological coupling
15. `colony_accumulation.py` — surface accumulation diagnostic
16. **Verify:** visibility diagnostic makes physical sense (S ratio behaviour)

### Phase 7: Validation pipeline
17. `validation.py` — confounder checks (retained), nonlinear envelope, κ diagnostic
18. Run against observation dataset
19. Produce all validation figures
20. Populate cumulative improvement table

### Phase 8: Lake-specific profiles
21. Implement `shallow_lake_profile` in `profiles.py` using Phillips et al. (2010) drift/shear
22. Re-run validation with realistic profiles
23. Compare with uniform-profile results

---

## 13. Files to Delete

Once the new pipeline is validated, remove these files or functions from the existing codebase:

- `buoyancy_visibility_weight()` in `stability.py`
- `find_visible_wavenumber()` in `stability.py`
- `spacing_visible` as primary prediction (retain as diagnostic only via `colony_accumulation.py`)
- The La_t correction interaction analysis from section 6 of the addendum (La_t enters naturally through the Rayleigh number mapping; no separate empirical correction is needed)
- `BUOYANCY_SCENARIOS` config (colony properties now enter through the accumulation diagnostic, not the spacing prediction)

Do NOT delete:
- `check_wind_spacing_confounders()` — keep
- `check_observation_quality_bias()` — keep
- The validation data files and observation annotations

---

## 14. Key Equations Quick Reference

For convenience when implementing. All from Hayes and Phillips (2017).

| Equation | Number | Purpose |
|---|---|---|
| CL-equations (perturbation form) | (1a–c) | Governing equations |
| Robin BCs (surface) | (2a–d) | Free surface conditions |
| Robin BCs (bottom) | (3a–d) | Rigid bottom conditions |
| γ scaling | (6) | (γ_s, γ_b) = l^4 (γ̃_s, γ̃_b) |
| D', U' polynomial form | (7a,b) | Profile specification |
| Cauchy product formula | (8) | For equating powers of l |
| Solvability condition (linear) | (13) | Gives σ_{2j} |
| R_{2j} formula (linear) | (20) | Neutral curve coefficients |
| R0 condition | (26) | R0^{-1} = ∫ ψ̃_1 U' dz |
| Nonlinear u_0 evolution | (51) | ∂u_0/∂T + σ_2 ∂²u_0/∂Y² = 0 |
| u_0 solution | (52) | Fourier series in Y |
| Ψ_0 solution | (53) | In terms of u_0 |
| u_2 solution | (54) | First nonlinear correction |
| Ψ_2 equation | (55) | With BCs (56) |
| Nonlinear R_2 | (63) | Central nonlinear result |
| R_2 parsing | (64) | R_2 = R*_2 + γ̃ R̃_2 |
| Nonlinear neutral curve | (65) | R̄ = R_0 + l² R*_2 + (γ/l²) R̃_2 |
| lcNL | (66) | (γ R̃_2 / R*_2)^{1/4} |
| lcL | (67) | (γ R_0 / R*_2)^{1/4} |
| κ ratio | (68) | (R_0 R*_2 / R*_2 R̃_2)^{1/4} |
| RcNL | (69) | R_0 + 2(γ R̃_2 R*_2)^{1/2} |
| RcL | (70) | R_0 + 2(γ R_0 R*_2)^{1/2} |
| Symmetry breaking | (74) | Shows u_0 → -u_0 invariance broken |

---

## 15. Acceptance Criteria

1. **All verification targets in section 11 pass** — the solver reproduces the paper's results.
2. **The nonlinear neutral curve R̄(l) > R(l)** at all l > 0 for Robin BCs — supercritical stability confirmed.
3. **κ ∈ [1.4, 1.9]** for the tested profiles — nonlinear spacing is 1.4–1.9× the linear spacing.
4. **Aspect ratios 5–11 recovered** with Cox & Leibovich γ values — matches observation.
5. **Confounder analysis complete** before model validation — retained from addendum.
6. **The observed/predicted spacing ratio is approximately constant** across wind speeds when using the nonlinear prediction — the inverse trend is explained by the hydrodynamics, not the biology.
7. **Colony visibility diagnostic is operational** but secondary — it flags whether patterns are satellite-observable, it does not determine spacing.
8. **Cumulative improvement table shows RMSE reduction** relative to both the old linear baseline and the old linear+buoyancy-filter approach.
9. **Asymptotic and Galerkin numerical solutions agree** — mutual validation of the two approaches.
