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
├── colony_accumulation.py  # Spacing, visibility, and finite-time mode evolution
├── era5.py              # ERA5/Open-Meteo download + cache helpers
├── weather.py           # Weather-window summaries and persistence diagnostics
├── timeline_analysis.py # Observation-centred hourly timelines + diagnostics
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
    cache_dir="data/era5_cache",
    output_dir="outputs/manual_validation",
    default_depth=9.0,       # fallback depth (m) when depth_m column is absent
    default_fetch=15000.0,   # fallback fetch (m)
    profile_name="uniform",  # shear/drift profile: "uniform" | "linear_drift" | …
    skip_download=False,     # use cached ERA5 if present, otherwise download from Open-Meteo
    use_lake_profile=True,   # dynamic shallow-lake profile built from forcing
    timeline_hours_before=48,
    timeline_hours_after=48,
    observation_hour_utc=12, # date-only observations are centred on noon UTC
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
    cache_dir="data/era5_cache",
    output_dir="outputs/manual_validation",
    use_lake_profile=True,
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
| `weather_summary.csv` | ERA5/Open-Meteo forcing summaries for each successful observation |
| `observation_diagnostics.csv` | One-row-per-observation summary of 96 h timeline diagnostics |
| `diagnostic_model.json` | Simple interpretable ridge-model comparison of diagnostic feature families |
| `timelines/*.csv` | Per-observation hourly timelines from 48 h before to 48 h after image time |
| `metrics.json` | Aggregate statistics (RMSE, R², bias, capture rate) |
| `confounder_check.json` | Wind–spacing partial correlations |

### Observation-centred timeline workflow

The validation pipeline now uses two linked forcing products for each
observation:

1. A 10-day spinup window used to compute a representative wind forcing.
2. A 96-hour hourly timeline centred on the observation date for finite-time
   state evolution and diagnostics.

Because the source dataset stores dates rather than timestamps, timelines are
centred on `12:00 UTC` by default. Each hourly step evolves:

- `selected_l`: the current Langmuir mode actually expressed by the system
- `target_l`: the instantaneous preferred mode from the current unstable band
- `amplitude_index`: a simple activity/growth proxy
- `development_index`: a simple organization/spin-up proxy

This makes it possible to inspect whether observed spacing is more consistent
with recent wind history, wave-history proxies, directional persistence, or
limitations in the current reduced-order model structure.

### Timeline diagnostics and simple interpretable model

For each observation, `timeline_analysis.py` computes summary diagnostics such
as:

- 48 h mean/max/std wind speed
- directional steadiness and cumulative turning angle
- mean `H_s`, `lambda_p`, `La_t`, and `D_max`
- hours supercritical and integrated supercriticality
- spacing, amplitude, and development at image time

These diagnostics are compared against observed spacing using simple
family-grouped ridge regressions:

- `wind_history`
- `wave_history_proxy`
- `direction_persistence`
- `model_structure`
- `combined`

The goal is diagnostic, not production prediction: to reveal which forcing
histories carry real explanatory signal and where the present CL-based reduced
model is still too insensitive.

### Key metrics

| Metric | Meaning |
|---|---|
| `rmse_NL_m` | Root-mean-square error of nonlinear predictions (m) |
| `bias_NL_m` | Mean signed error — positive = model overpredicts |
| `r_squared_NL` | R² of nonlinear predictions vs. observed spacing |
| `capture_rate_2x_NL` | Fraction of observations within factor 2× of prediction |
| `kappa` | Nonlinear wavenumber ratio (profile-dependent constant) |
| `rmse_L_m` | RMSE for linear baseline — compare to `rmse_NL_m` |
| `diagnostic_best_family` | Best-performing diagnostic feature family by LOOCV R² |
| `diagnostic_best_family_loocv_r2` | Cross-validated R² for that family |

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

When `use_lake_profile=True`, validation and timeline workflows instead use a
dynamic shallow-lake profile derived from the local forcing (`LCParams`) at
each step.

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
