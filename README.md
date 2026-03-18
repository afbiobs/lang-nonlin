# Langmuir Circulation — Nonlinear CL Solver

Nonlinear Craik-Leibovich (CL) equation solver for predicting Langmuir cell
spacing in shallow lakes. Based on Hayes & Phillips (2017), *Geophys. Astrophys.
Fluid Dyn.*, 111(1), 65–90.

## Why nonlinear?

Linear stability theory underestimates observed cell spacing in shallow water
by roughly 2×. The nonlinear solver accounts for the subcritical bifurcation
that shifts the critical wavenumber, producing aspect ratios (spacing/depth) of
5–11× consistent with field observations rather than the 2–3× from linear
theory.

## Install

```bash
pip install -e .
```

Dependencies: `numpy>=1.26`, `pandas>=2.2`, `scipy>=1.12`.

## Module layout

```
langmuir/
├── params.py            # LCParams: wind → physical parameters (Ra, u*, D_max, …)
├── profiles.py          # Shear U'(z) and Stokes drift D'(z) profiles
├── robin_bc.py          # Robin boundary conditions (γ_s, γ_b)
├── linear_solver.py     # Linear perturbation baseline
├── nonlinear_solver.py  # Nonlinear asymptotic + Galerkin solver (CORE)
├── galerkin.py          # Legendre spectral infrastructure
├── rayleigh_mapping.py  # Wind speed → Rayleigh number
├── colony_accumulation.py  # Biological visibility diagnostic (secondary)
├── validation.py        # Validation pipeline
└── utils.py             # Polynomial helpers
```

## Quick start — single prediction

```python
from langmuir import LCParams, predict_spacing_and_visibility

params = LCParams(U10=6.0, depth=9.0, fetch=15000.0)
result = predict_spacing_and_visibility(params)

print(f"Predicted spacing (nonlinear): {result['spacing_nonlinear']:.1f} m")
print(f"Predicted spacing (linear):    {result['spacing_linear']:.1f} m")
print(f"Regime: {result['regime']}")
print(f"Rayleigh number: {result['Ra']:.1f}")
print(f"kappa = lcL/lcNL: {result['kappa']:.3f}")
```

Key fields returned by `predict_spacing_and_visibility`:

| Field | Description |
|---|---|
| `spacing_nonlinear` | Predicted cell spacing (m) from nonlinear solver |
| `spacing_linear` | Baseline spacing from linear theory (m) |
| `kappa` | lcL / lcNL — ratio of linear to nonlinear wavenumber |
| `regime` | `subcritical` / `near_onset` / `moderate` / `supercritical` |
| `Ra` | Rayleigh number for this wind/depth/fetch combination |
| `RcNL` | Critical Rayleigh number (nonlinear) |
| `is_visible` | Whether surface accumulation is detectable |
| `accumulation_factor` | Biological herding strength (0–1) |

## Validating against windrow spacing observations

The dataset `data/observations_minimal.csv` contains 67 field observations.
The column `manual_spacing_m` holds manually measured windrow spacing in metres.

### Run validation from Python

```python
from langmuir.validation import validate_nonlinear

result = validate_nonlinear(
    dataset_path="data/observations_minimal.csv",
    spacing_column="manual_spacing_m",
    output_dir="outputs/manual_validation",
    default_depth=9.0,       # fallback depth (m) when depth_m column is absent
    default_fetch=15000.0,   # fallback fetch (m)
    profile_name="uniform",  # shear/drift profile: "uniform" | "linear_drift" | …
)

print(result.metrics)
```

### Run validation from the command line

```bash
python - <<'EOF'
from langmuir.validation import validate_nonlinear
r = validate_nonlinear(
    "data/observations_minimal.csv",
    spacing_column="manual_spacing_m",
    output_dir="outputs/manual_validation",
)
import json, sys
print(json.dumps(r.metrics, indent=2, default=str))
EOF
```

### Output files

After running, `outputs/manual_validation/` contains:

| File | Contents |
|---|---|
| `results.csv` | Per-observation predictions vs. observations |
| `metrics.json` | Aggregate statistics (RMSE, R², bias, capture rate) |
| `confounder_check.json` | Wind–spacing partial correlations |

### Key metrics

| Metric | Meaning |
|---|---|
| `rmse_NL_m` | Root-mean-square error of nonlinear predictions (m) |
| `bias_NL_m` | Mean signed error — positive = model overpredicts |
| `r_squared_NL` | R² of nonlinear predictions vs. observed spacing |
| `capture_rate_2x_NL` | Fraction of observations within factor 2× of prediction |
| `kappa` | Nonlinear wavenumber ratio (profile-dependent constant) |
| `rmse_L_m` | RMSE for linear baseline — compare to `rmse_NL_m` |

### Checking the `results.csv` output

```python
import pandas as pd

df = pd.read_csv("outputs/manual_validation/results.csv")
print(df[["observed_spacing_m", "predicted_spacing_NL_m", "error_NL_m",
          "regime", "Ra", "depth_m"]].to_string())
```

## LCParams reference

```python
LCParams(
    U10    = 6.0,       # 10-m wind speed (m/s)  — required
    depth  = 9.0,       # Water depth (m)
    fetch  = 15000.0,   # Wind fetch (m)
    gamma_s = 0.06,     # Surface Robin parameter
    gamma_b = 0.28,     # Bottom Robin parameter
)
```

Derived quantities computed automatically: `u_star`, `Ra`, `D_max`, `nu_T`,
`La_t`, `H_s`, `T_p`.

## Profiles

| Name | Description |
|---|---|
| `uniform` | Uniform shear, fetch-limited drift (default) |
| `linear_drift` | Linear Stokes drift profile |
| `linear_shear` | Linear wind-driven shear |
| `both_linear` | Both linear |

## Dataset columns (`observations_minimal.csv`)

| Column | Description |
|---|---|
| `observation_id` | Unique ID (obs_001 … obs_067) |
| `image_date` | Date of satellite/aerial image |
| `authoritative_lat` | Latitude |
| `authoritative_lng` | Longitude |
| `manual_spacing_m` | Windrow spacing measured manually (m) — primary validation target |
| `wiggle_spacing_m` | Spacing from automated wiggle detection (m) |
| `depth_m` | Water depth (m) — optional, defaults to 9.0 m |
| `fetch_m` | Wind fetch (m) — optional, defaults to 15 000 m |

## Physical background

Langmuir circulation consists of counter-rotating vortex pairs aligned with
the wind. Their cross-wind spacing is the primary observable. In shallow lakes
the nonlinear term J(ψ, ∇²ψ) shifts the marginal stability boundary to smaller
wavenumbers (wider cells) relative to the linear prediction. The ratio
κ = l_cL / l_cNL is a profile-dependent constant ≈ 0.55–0.75 for typical
shallow-lake conditions.
