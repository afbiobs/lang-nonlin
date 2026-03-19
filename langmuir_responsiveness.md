# Langmuir Responsiveness Review

## Scope
This note reviews the last 6 commits on `main` as a first-principles record of what was changed in the core physics, what problem each change targeted, and why the model still struggles to produce observation-scale temporal responsiveness.

Reviewed commits:

| Commit | Role in review |
| --- | --- |
| `abd1a73` | First major attempt to add duration-aware temporal response on top of CL spacing |
| `f075619` | Major hydrodynamic/solver/lifecycle refactor; introduced the hybrid CL/response model |
| `710056e` | Added literature note on resolvent analysis; no code change |
| `0fb71b4` | Placeholder only; no physics change |
| `2e7319c` | Targeted fix for the narrow-spacing-band failure mode |
| `f00ae1c` | Merge commit for `2e7319c`; no additional model logic beyond that patch |

Only `abd1a73`, `f075619`, and `2e7319c` materially changed the modeled physics.

## Executive Summary
The reviewed changes progressively moved the code from:

1. a duration-tuned finite-time CL expression model with many free parameters,
2. to a more physically anchored hybrid model with improved hydrodynamics, solver robustness, buoyancy, and a CL-vs-response decomposition,
3. to a direct-forcing patch intended to recover wind sensitivity after the hybrid model collapsed into an almost constant spacing band.

The central responsiveness problem was not primarily a Newton/Galerkin failure. The important structural issue is that the model chain increasingly discarded amplitude information before scale selection:

- `shallow_lake_profile()` fit **normalized positive polynomials** for `D'(z)` and `U'(z)`, so the CL solver mostly saw shape, not magnitude.
- `Ra = U_surface D_surface H^2 / nu_T^2` became only weakly wind-sensitive under the new hydrodynamic closure because `U_surface`, `D_surface`, and `nu_T` all increased with wind and partially cancelled.
- `build_hybrid_spacing_spectrum()` normalized the CL and response spectra before mixing them, so absolute response gain was removed and only target-scale shape remained.
- lifecycle states (`coherent_run_hours`, `setup_index`, `coarsening_index`, `response_mix`) then saturated under long spin-up windows, compressing the final spacing range even further.

The result is that later commits improved physical plausibility and robustness, but also made the observable spacing less responsive because the chain became dominated by normalized, slowly varying, or saturated quantities.

## Model Workflow At The End Of The Reviewed Window
The effective modeling workflow after `f00ae1c` was:

1. Build hourly forcing windows around each observation.
2. For each hour, construct `LCParams(U10, depth, fetch)`.
3. In `LCParams._compute_derived()`:
   - compute continuous lake drag from Wuest-Lorke at low wind and Charnock at higher wind,
   - compute fetch-limited wave state `(H_s, T_p, lambda_p)`,
   - compute broadband Stokes drift profile using an `exp1` vertical profile,
   - compute Langmuir-enhanced viscosity `nu_T = nu_shear * sqrt(1 + 0.49 / La_SL^2)`,
   - solve a 1D variable-viscosity current profile with zero depth-integrated transport,
   - assemble `Ra = U_surface D_surface H^2 / nu_T^2`.
4. In `shallow_lake_profile()`:
   - take gradients of the resolved current and Stokes profiles,
   - clip them positive,
   - fit low-order positive Bernstein polynomials,
   - normalize those profiles before passing them to the CL solver.
5. In `solve_linear()` / `solve_nonlinear()`:
   - solve the asymptotic CL problem,
   - explicitly retain the Robin singular term in the neutral curve,
   - enforce the nonlinear subcritical safeguard `kappa = R^*_{2,NL} / R^*_{2,L} > 1`,
   - optionally validate with a pseudo-arclength Galerkin continuation at `I=2`.
6. In `supercritical_mode_spectrum()`:
   - find the unstable band of the nonlinear neutral curve,
   - compute a growth-weighted target `l`.
7. In `build_hybrid_spacing_spectrum()`:
   - compute a normalized CL growth spectrum,
   - compute a normalized coarse “resolvent-inspired” response spectrum,
   - compute a normalized visibility filter,
   - mix them with a scalar `response_mix`.
8. In `advance_langmuir_state()`:
   - evolve finite-time memory states such as `coherent_run_hours`, `setup_index`, `coarsening_index`, `visible_fraction`,
   - relax `selected_l` toward a target derived from CL scale and response scale,
   - convert `selected_l` to spacing via `lambda = 2 pi H / l`.
9. In `validate_nonlinear()`:
   - compare the observation-time predicted spacing against the observed windrow spacing,
   - store hourly timeline diagnostics and correlation summaries.

That workflow is important because the sensitivity bottleneck can occur at any one of steps 3, 4, 7, or 8.

## Commit-By-Commit Physics Review

### `abd1a73` — Duration-Aware Langmuir Tuning Workflow
Primary idea:

- Treat visible spacing as the nonlinear CL target scale filtered through finite-time development.
- Add a large number of duration, coherence, onset, and merge controls.
- Tune these controls against observation-time diagnostics.

Core assumptions introduced:

- The main missing physics was **finite-time adjustment**, not hydrodynamic scale-selection.
- A scalar duration memory could bridge the gap between instantaneous CL spacing and observed windrow spacing.
- Larger cells could be promoted heuristically by asymmetrically biasing the supercritical spectrum toward smaller `l` at higher supercriticality.

Important logic:

- `coherent_run_drive()` required sufficient stress, supercriticality, and directional coherence before the setup memory could grow.
- `setup_index = 1 - exp(-coherent_run_hours / setup_hours_scale)` controlled expression of large cells.
- The selected mode relaxed toward the instantaneous supercritical target on configurable timescales.
- A heuristic asymmetric bias in `supercritical_mode_spectrum()` explicitly favored larger cells at higher forcing.

What this achieved:

- It produced large temporal variability in the archived `v1` manual timeline outputs:
  `timeline_spacing_q10 ≈ 57.6 m`, `timeline_spacing_q90 ≈ 197.8 m`.
- It improved wiggle-vs-linear RMSE relative to the earlier baseline in the committed `v1` metrics.

Why this was not a satisfactory fix:

- Variability came largely from **tunable memory and bias logic**, not from physically derived hydrodynamic sensitivity.
- The parameter count was very high and the parameters were not strongly identifiable.
- The asymmetric bias in the supercritical spectrum was heuristic and not derived from the CL equations.
- The model still treated windrow spacing as one scalar “expressed mode” and did not separate cell scale from visible tracer scale.
- This stage likely overfit lifecycle behavior without resolving whether the CL driver itself had the right sensitivity.

Bottom line:

- `abd1a73` showed that a temporal-memory wrapper can generate variability, but it did not prove that the variability was physically trustworthy.

### `f075619` — Hybrid Langmuir Lifecycle Model
This was the largest physics refactor in the reviewed window.

#### 1. Hydrodynamics
Changes:

- Replaced the drag step function with a continuous low-wind/high-wind blend.
- Replaced monochromatic Stokes drift with a broadband fetch-limited profile.
- Replaced constant eddy viscosity with a Langmuir-enhanced profile.
- Replaced the 3% surface current rule with a 1D vertical momentum balance and return flow.

Net effect:

- The hydrodynamic closure became more defensible.
- The nondimensional forcing regime changed drastically:
  committed `v1` had `Ra_mean ~ 3.8e4`,
  committed `v5` had `Ra_mean ~ 4.8e2`.

Important limitation:

- The closure made `Ra` only weakly wind-sensitive because the numerator and denominator co-varied with wind.
- This matters because the later scale-selection stages still relied heavily on `Ra` and functions of `Ra`.

#### 2. Solver Robustness
Changes:

- Linear neutral curve explicitly separated the Robin singular term from the regular series.
- Nonlinear neutral curve used the same singular-plus-regular structure.
- Added `SubcriticalBifurcationError` with `kappa > 1` as the subcritical widening condition.
- Deleted the abandoned partial Galerkin path and switched numeric validation to collocation plus pseudo-arclength continuation.

Net effect:

- Numerical artifacts were less likely to masquerade as physics.
- Remaining spacing problems are therefore more likely to be structural than purely numerical.

This part should mostly be retained.

#### 3. Profile Reduction
Changes:

- `shallow_lake_profile()` stopped fitting velocities directly and instead fit gradients of current and Stokes drift.
- The fitted `D'(z)` and `U'(z)` were forced positive using Bernstein/NNLS fits.

Important limitation:

- The profiles were normalized before fitting and renormalized to unit integral.
- That removed most amplitude information from the CL solver input.
- The CL problem therefore responded mostly to **profile shape**, while the forcing magnitude was pushed into the scalar `Ra`.

This is a major structural candidate for the observed loss of wind sensitivity.

#### 4. Hybrid Response Model
Changes:

- Introduced `resolvent_spectrum.py`, but not a true resolvent solve.
- Response energy was approximated from near-surface `U^L'(z)`, `U_s'(z)`, curvature, and `nu_T(z)`.
- `build_hybrid_spacing_spectrum()` mixed:
  normalized CL growth,
  normalized response energy,
  normalized visibility weights.

Important limitation:

- All three spectra were normalized before mixing.
- This discarded absolute gain and therefore much of the information about how strongly the flow should respond at a given hour.
- `response_mix` was a scalar heuristic depending on coherence, setup, development, supercriticality, and persistence.
- Because those state variables were themselves slowly varying, the mixed target scale was also slowly varying.

This stage added conceptual decomposition, but the implementation remained a low-order surrogate rather than a physics-based forced-response operator.

#### 5. Buoyancy / Visibility / Lagrangian Accumulation
Changes:

- Added a buoyancy state variable `Q`, temperature/light dependence, and a dynamic `v_float`.
- Replaced the old ad hoc accumulation factor with a convergence-timescale and line-density estimate derived from the CL streamfunction.
- Added `visible_fraction`, `coarsening_index`, `response_mix`, `large_scale_fraction`.

Important limitation:

- At this stage, `spacing_nonlinear = spacing_visible = core_spacing`.
- A separate `spacing_observable = visible_fraction * core_spacing` was computed, but it was diagnostic, not the main validated spacing.
- The model therefore still conflated:
  roll/core scale,
  visible streak scale,
  and observation-time measurable spacing.

#### 6. Observed behavior after `f075619`
Committed `v5` outputs show the main failure clearly:

- Manual: `timeline_spacing_q10 ≈ 102.4 m`, `timeline_spacing_q90 ≈ 104.5 m`
- Wiggle: `timeline_spacing_q10 ≈ 102.2 m`, `timeline_spacing_q90 ≈ 103.9 m`

So the hybrid model produced a nearly constant spacing in both datasets despite hourly forcing. This is the key collapse that the next patch tried to repair.

### `2e7319c` / `f00ae1c` — Direct Forcing Sensitivity Patch
Primary idea:

- Keep the hybrid model, but make the finite-time wrapper more responsive to instantaneous forcing so that the 96 h timelines do not collapse into a near-constant spacing.

Changes:

- Added `supercritical_drive = supercriticality / (supercriticality + 0.45)` to `coarsening_target`.
- Blended the spectrum-derived `visible_target_l` back into the target spacing, instead of fully overriding it with the coarsening interpolation.
- Capped `coherent_run_hours` at `4 * relax_hours` to reduce permanent saturation of `setup_index`.
- Added an onset-decay term when forcing was weak but still supercritical.

What problem it correctly identified:

- The previous collapse was partly due to state saturation and fully overridden spectrum targets.

Why it still could not solve the root issue:

- `supercriticality` still depended on `Ra`, which was already only weakly wind-sensitive under the new hydrodynamic closure.
- The spectrum target was still generated from normalized spectra with weakly varying mean-state shapes.
- The fix changed the **wrapper around scale selection**, not the deeper hydrodynamic or spectral sensitivity.

This patch was therefore plausible but necessarily partial.

## What Likely Caused The Responsiveness Failure

### 1. Hydrodynamic cancellation
The post-`f075619` closure computes

`Ra = U_surface D_surface H^2 / nu_T^2`

with all of `U_surface`, `D_surface`, and `nu_T` wind-dependent.

That makes wind sensitivity nontrivial and can easily compress `d log Ra / d log U10`. If the scalar control parameter is nearly flat, downstream CL thresholds and target scales will also be flat.

### 2. Profile-amplitude loss
`shallow_lake_profile()` fits normalized, positive `D'(z)` and `U'(z)` profiles. This preserves sign and broad shape but discards amplitude. The CL solver then sees only shape variation plus the scalar `Ra`. If shape is approximately self-similar across wind states, spacing becomes nearly invariant.

### 3. Normalized spectral mixing
`build_hybrid_spacing_spectrum()` uses normalized CL growth and normalized response energy, then normalizes the hybrid energy again. This makes the selected `l` depend mostly on spectral shape, not absolute amplification. If the shapes move little, the selected scale moves little.

### 4. Saturating lifecycle states
`coherent_run_hours`, `setup_index`, `response_mix`, and `coarsening_index` all had strong tendencies to saturate under long spin-up windows. Once saturated, they cease to transmit additional forcing information to spacing.

### 5. Observable conflation
The reviewed commits never cleanly separated:

- circulation/core scale,
- visible tracer/streak expression,
- observed image-time spacing.

That made it hard to distinguish “no cells”, “cells exist but are not visible”, and “cells exist but have the wrong scale”.

## Successful Changes Worth Retaining
- Explicit Robin singular handling in the linear and nonlinear neutral curves.
- Subcritical safeguard via `kappa`.
- Pseudo-arclength continuation for numeric branch tracing.
- Continuous drag, broadband Stokes drift, Langmuir-modified viscosity, and return-flow current solve.
- The conceptual separation of `CL scale`, `response scale`, and `visible/observed scale`, even though the implementation remained incomplete.
- Hourly observation-centred validation rather than scalar representative-wind validation.

## Failed Or Only Partially Successful Fixes
- Large parameterized duration-memory model in `abd1a73`: produced variability, but not from clearly defensible physics.
- Asymmetric supercritical spectrum bias in `abd1a73`: heuristic, not first-principles.
- Hybrid response proxy in `f075619`: useful diagnostic concept, but not a real forced-response calculation and too normalized to carry amplitude sensitivity.
- Direct-forcing wrapper patch in `2e7319c`: sensible local repair, but downstream of the main cancellation mechanisms.

## Questions An Expert Reviewer Should Answer
1. Is `Ra = U_surface D_surface H^2 / nu_T^2` the right scalar control for spacing under the new shallow-lake closure, or is the relevant scale controlled by a different combination of stress, wave forcing depth, and vertical structure?
2. Does normalizing `D'(z)` and `U'(z)` before the CL solve remove exactly the amplitude dependence needed for wind responsiveness?
3. Is a scalar CL-based spacing target fundamentally insufficient once the flow is in a turbulence-maintained, forced-response regime?
4. Should visible windrow spacing be modeled as a separate coarsening/aggregation observable rather than as the same quantity as roll spacing?
5. If a resolvent framework is to be used, should the code move from the current heuristic response proxy to an actual linear operator with `U^L(z)`, `U_s'(z)`, and `nu_T(z)`?
6. Are the current timescale states (`setup`, `coarsening`, `visibility`) physically necessary, or are they compensating for missing hydrodynamic sensitivity upstream?

## Bottom Line
The reviewed commit window shows a clear progression:

- early variability came from a heavily parameterized temporal wrapper,
- later physics refactors improved realism and robustness,
- but those same refactors also pushed the model toward normalized, slowly varying quantities,
- so the final spacing predictor became too insensitive to wind.

The strongest candidate root causes are:

- weak wind sensitivity of the scalar forcing closure,
- profile normalization that removes amplitude from the CL solve,
- normalized response-spectrum mixing,
- and incomplete separation between cell scale and observed streak scale.

An external first-principles review should therefore focus less on numerical continuation and more on the dimensional scaling, profile reduction, scale-selection observable, and whether the current hybrid “resolvent-inspired” surrogate can ever carry the required temporal responsiveness without a more explicit forced-response model.
