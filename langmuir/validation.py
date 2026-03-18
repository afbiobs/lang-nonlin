"""Validation pipeline for the nonlinear CL model.

Retains confounder and quality checks from the existing code,
adds nonlinear consistency envelope and kappa diagnostic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .params import LCParams
from .profiles import get_profile, ShearDriftProfile
from .robin_bc import RobinBoundaryConditions
from .nonlinear_solver import solve_nonlinear
from .colony_accumulation import predict_spacing_and_visibility
from .rayleigh_mapping import wind_to_rayleigh


# ======================================================================
# Confounder checks (retained from existing code)
# ======================================================================

def check_wind_spacing_confounders(results_df: pd.DataFrame) -> dict:
    """Check 1: Partial correlation controlling for depth, within-location analysis."""
    report = {}

    if "U10" not in results_df.columns or "observed_spacing_m" not in results_df.columns:
        return {"status": "skipped", "reason": "missing columns"}

    valid = results_df.dropna(subset=["U10", "observed_spacing_m"])
    if len(valid) < 5:
        return {"status": "skipped", "reason": "insufficient data"}

    # Simple correlation
    r_wind_spacing = float(np.corrcoef(valid["U10"], valid["observed_spacing_m"])[0, 1])
    report["r_wind_spacing"] = r_wind_spacing

    # Partial correlation controlling for depth
    if "depth_m" in valid.columns:
        from numpy.linalg import lstsq
        X = valid[["U10", "depth_m"]].values
        y = valid["observed_spacing_m"].values
        # Residualize both U10 and spacing on depth
        depth = valid["depth_m"].values.reshape(-1, 1)
        u10_res = X[:, 0] - depth.flatten() * (lstsq(depth, X[:, 0], rcond=None)[0][0])
        sp_res = y - depth.flatten() * (lstsq(depth, y, rcond=None)[0][0])
        if np.std(u10_res) > 1e-10 and np.std(sp_res) > 1e-10:
            r_partial = float(np.corrcoef(u10_res, sp_res)[0, 1])
            report["r_partial_depth_controlled"] = r_partial

    # Within-location analysis
    if "lat" in valid.columns and "lon" in valid.columns:
        valid = valid.copy()
        valid["location_key"] = (
            valid["lat"].round(1).astype(str) + "_" + valid["lon"].round(1).astype(str)
        )
        loc_counts = valid["location_key"].value_counts()
        multi_obs = loc_counts[loc_counts >= 3].index
        slopes = []
        for loc in multi_obs:
            subset = valid[valid["location_key"] == loc]
            if subset["U10"].std() > 0.5:
                coeffs = np.polyfit(subset["U10"], subset["observed_spacing_m"], 1)
                slopes.append(coeffs[0])
        if slopes:
            report["within_location_slopes"] = slopes
            report["mean_within_slope"] = float(np.mean(slopes))

    report["status"] = "complete"
    return report


def check_observation_quality_bias(results_df: pd.DataFrame) -> dict:
    """Check 2: Spacing spread by wind bin, geographic concentration."""
    report = {}

    if "U10" not in results_df.columns or "observed_spacing_m" not in results_df.columns:
        return {"status": "skipped"}

    valid = results_df.dropna(subset=["U10", "observed_spacing_m"])
    if len(valid) < 5:
        return {"status": "skipped", "reason": "insufficient data"}

    # Wind bin analysis
    bins = [0, 3, 5, 7, 10, 15, 25]
    valid = valid.copy()
    valid["wind_bin"] = pd.cut(valid["U10"], bins=bins)
    bin_stats = valid.groupby("wind_bin", observed=False)["observed_spacing_m"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    report["wind_bin_stats"] = bin_stats.to_dict()

    # Geographic concentration
    if "lat" in valid.columns and "lon" in valid.columns:
        lat_range = valid["lat"].max() - valid["lat"].min()
        lon_range = valid["lon"].max() - valid["lon"].min()
        report["geographic_extent_lat"] = float(lat_range)
        report["geographic_extent_lon"] = float(lon_range)
        n_unique_locs = len(valid.groupby([valid["lat"].round(1), valid["lon"].round(1)]))
        report["n_unique_locations"] = int(n_unique_locs)

    report["status"] = "complete"
    return report


# ======================================================================
# Nonlinear consistency envelope (NEW - replaces Check 3)
# ======================================================================

def nonlinear_consistency_envelope(results_df: pd.DataFrame | None = None,
                                    depths: list[float] | None = None,
                                    fetch: float = 15000.0,
                                    profile_name: str = "uniform") -> dict:
    """Compute nonlinear predicted spacing across wind speed range.

    For each depth, compute spacing_nonlinear and spacing_linear,
    plus biological visibility mask.
    """
    if depths is None:
        depths = [5.0, 9.0, 15.0]

    wind_range = np.linspace(1.5, 15.0, 50)
    envelopes = {}

    for depth in depths:
        spacings_NL = []
        spacings_L = []
        is_visible = []
        Ra_values = []
        regimes = []

        for U10 in wind_range:
            params = LCParams(U10=float(U10), depth=depth, fetch=fetch)
            result = predict_spacing_and_visibility(params, profile_name=profile_name)

            spacings_NL.append(result["spacing_nonlinear"])
            spacings_L.append(result["spacing_linear"])
            is_visible.append(result["is_visible"])
            Ra_values.append(result["Ra"])
            regimes.append(result["regime"])

        envelopes[depth] = {
            "wind": wind_range.tolist(),
            "spacing_NL": spacings_NL,
            "spacing_L": spacings_L,
            "is_visible": is_visible,
            "Ra": Ra_values,
            "regimes": regimes,
        }

    return envelopes


# ======================================================================
# Kappa diagnostic (NEW)
# ======================================================================

def kappa_diagnostic(results_df: pd.DataFrame, profile_name: str = "uniform") -> dict:
    """For each observation, compute kappa = lcL / lcNL.

    kappa should be approximately constant (depends on profiles, not wind).
    """
    kappas = []
    obs_wind = []
    obs_ratio = []

    profile = get_profile(profile_name)
    bcs = RobinBoundaryConditions()

    # Solve once for kappa (profile-dependent, not wind-dependent)
    nl_result = solve_nonlinear(profile, bcs)
    kappa_theory = nl_result.kappa

    for _, row in results_df.iterrows():
        if pd.isna(row.get("observed_spacing_m")) or pd.isna(row.get("predicted_spacing_NL_m")):
            continue

        ratio = row["observed_spacing_m"] / row["predicted_spacing_NL_m"]
        obs_ratio.append(ratio)
        obs_wind.append(row.get("U10", float("nan")))
        kappas.append(kappa_theory)

    return {
        "kappa_theory": kappa_theory,
        "obs_to_pred_ratio": obs_ratio,
        "wind_speeds": obs_wind,
        "mean_ratio": float(np.nanmean(obs_ratio)) if obs_ratio else float("nan"),
        "std_ratio": float(np.nanstd(obs_ratio)) if obs_ratio else float("nan"),
    }


# ======================================================================
# Full validation pipeline
# ======================================================================

@dataclass
class NonlinearValidationResult:
    results: pd.DataFrame
    metrics: dict
    envelopes: dict
    confounder_check: dict
    quality_check: dict
    kappa_check: dict | None
    output_dir: Path | None = None


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def validate_nonlinear(
    dataset_path: str,
    spacing_column: str = "manual_spacing_m",
    cache_dir: str = "data/era5_cache",
    output_dir: str | None = None,
    default_depth: float = 9.0,
    default_fetch: float = 15000.0,
    profile_name: str = "uniform",
    skip_weather: bool = True,
) -> NonlinearValidationResult:
    """Run full nonlinear validation pipeline.

    1. Load observations
    2. Run confounder checks
    3. For each observation, predict spacing using nonlinear model
    4. Compute metrics (RMSE, R^2, capture rate)
    5. Generate consistency envelope
    6. Run kappa diagnostic
    """
    # Load observations
    df = pd.read_csv(dataset_path)
    required = {"image_date", "authoritative_lat", "authoritative_lng", spacing_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    obs = pd.DataFrame({
        "image_date": pd.to_datetime(df["image_date"], errors="coerce").dt.date,
        "lat": pd.to_numeric(df["authoritative_lat"], errors="coerce"),
        "lon": pd.to_numeric(df["authoritative_lng"], errors="coerce"),
        "observed_spacing_m": pd.to_numeric(df[spacing_column], errors="coerce"),
    })
    if "depth_m" in df.columns:
        obs["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce").fillna(default_depth)
    else:
        obs["depth_m"] = default_depth
    if "fetch_m" in df.columns:
        obs["fetch_m"] = pd.to_numeric(df["fetch_m"], errors="coerce").fillna(default_fetch)
    else:
        obs["fetch_m"] = default_fetch

    obs = obs.dropna(subset=["image_date", "lat", "lon", "observed_spacing_m"])
    obs = obs[(obs["observed_spacing_m"] > 1.0) & (obs["observed_spacing_m"] < 2000.0)]
    obs = obs.reset_index(drop=True)

    # Try to load weather data if available
    weather_data = {}
    cache_path = Path(cache_dir)
    if not skip_weather and cache_path.exists():
        try:
            from spacing_langmuir.era5 import fetch_all_observations
            from spacing_langmuir.weather import extract_model_forcing
            # Would load weather here
        except ImportError:
            pass

    # Predict spacing for each observation
    result_rows = []
    # Use representative wind speeds if no weather data
    # For validation without weather, use a range of typical wind speeds
    profile = get_profile(profile_name)
    bcs = RobinBoundaryConditions()
    nl_result = solve_nonlinear(profile, bcs)

    for idx, row in obs.iterrows():
        depth = float(row["depth_m"])
        fetch = float(row["fetch_m"])

        # If we have cached weather, use it; otherwise use heuristic U10
        U10 = weather_data.get(idx, {}).get("U10_representative", 5.0)

        params = LCParams(U10=U10, depth=depth, fetch=fetch)
        pred = predict_spacing_and_visibility(params, profile_name=profile_name)

        result_rows.append({
            "obs_index": idx,
            "image_date": str(row["image_date"]),
            "lat": row["lat"],
            "lon": row["lon"],
            "observed_spacing_m": row["observed_spacing_m"],
            "predicted_spacing_NL_m": pred["spacing_nonlinear"],
            "predicted_spacing_L_m": pred["spacing_linear"],
            "error_NL_m": pred["spacing_nonlinear"] - row["observed_spacing_m"]
                          if not math.isnan(pred["spacing_nonlinear"]) else float("nan"),
            "error_L_m": pred["spacing_linear"] - row["observed_spacing_m"]
                         if not math.isnan(pred["spacing_linear"]) else float("nan"),
            "depth_m": depth,
            "fetch_m": fetch,
            "U10": U10,
            "Ra": pred["Ra"],
            "regime": pred["regime"],
            "kappa": pred["kappa"],
            "is_visible": pred["is_visible"],
            "accumulation_factor": pred["accumulation_factor"],
            "w_down_max": pred["w_down_max"],
        })

    results = pd.DataFrame(result_rows)

    # Compute metrics
    valid_nl = results.dropna(subset=["predicted_spacing_NL_m"])
    valid_l = results.dropna(subset=["predicted_spacing_L_m"])

    metrics = {
        "spacing_column": spacing_column,
        "profile": profile_name,
        "n_observations": int(len(results)),
        "n_valid_NL": int(len(valid_nl)),
        "n_valid_L": int(len(valid_l)),
    }

    if len(valid_nl) > 0:
        metrics["rmse_NL_m"] = float(np.sqrt(np.mean(valid_nl["error_NL_m"] ** 2)))
        metrics["bias_NL_m"] = float(valid_nl["error_NL_m"].mean())
        metrics["r_squared_NL"] = _r_squared(
            valid_nl["observed_spacing_m"].values,
            valid_nl["predicted_spacing_NL_m"].values,
        )

    if len(valid_l) > 0:
        metrics["rmse_L_m"] = float(np.sqrt(np.mean(valid_l["error_L_m"] ** 2)))
        metrics["bias_L_m"] = float(valid_l["error_L_m"].mean())
        metrics["r_squared_L"] = _r_squared(
            valid_l["observed_spacing_m"].values,
            valid_l["predicted_spacing_L_m"].values,
        )

    # Capture rate: fraction within 2x of observed
    if len(valid_nl) > 0:
        ratio = valid_nl["predicted_spacing_NL_m"] / valid_nl["observed_spacing_m"]
        metrics["capture_rate_2x_NL"] = float(((ratio > 0.5) & (ratio < 2.0)).mean())

    # R0 and kappa from the model
    metrics["R0"] = float(nl_result.R0)
    metrics["kappa"] = float(nl_result.kappa)
    metrics["lcNL"] = float(nl_result.lcNL)
    metrics["lcL"] = float(nl_result.linear_result.lcL)
    metrics["RcNL"] = float(nl_result.RcNL)
    metrics["RcL"] = float(nl_result.linear_result.RcL)

    # Confounder checks
    confounder = check_wind_spacing_confounders(results)
    quality = check_observation_quality_bias(results)

    # Envelope
    envelopes = nonlinear_consistency_envelope(
        results, depths=[5.0, 9.0, 15.0], profile_name=profile_name
    )

    # Kappa diagnostic
    kappa_check = None
    if len(valid_nl) > 0:
        kappa_check = kappa_diagnostic(results, profile_name=profile_name)

    result = NonlinearValidationResult(
        results=results,
        metrics=metrics,
        envelopes=envelopes,
        confounder_check=confounder,
        quality_check=quality,
        kappa_check=kappa_check,
    )

    # Write outputs
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results.to_csv(out / "results.csv", index=False)
        with (out / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2, default=str)
        with (out / "confounder_check.json").open("w") as f:
            json.dump(confounder, f, indent=2, default=str)
        result.output_dir = out

    return result
