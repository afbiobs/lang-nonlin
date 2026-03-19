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

from .open_meteo_client import fetch_all_observations
from .params import LCParams
from .profiles import get_profile
from .robin_bc import RobinBoundaryConditions
from .nonlinear_solver import solve_nonlinear
from .colony_accumulation import LangmuirDynamicsConfig, predict_spacing_and_visibility
from .timeline_analysis import (
    fit_interpretable_diagnostic_model,
    predict_observation_timeline,
    write_observation_timelines,
)
from .weather import classify_wind_regime, summarise_context_window, summarise_spinup_forcing


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
    if np.std(valid["U10"]) <= 1e-10 or np.std(valid["observed_spacing_m"]) <= 1e-10:
        return {"status": "skipped", "reason": "zero variance"}

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
                                    profile_name: str = "uniform",
                                    use_lake_profile: bool = False) -> dict:
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
        spacings_core = []
        spacings_CL = []
        spacings_response = []
        spacings_L = []
        visibility_index = []
        is_visible = []
        Ra_values = []
        regimes = []

        for U10 in wind_range:
            params = LCParams(U10=float(U10), depth=depth, fetch=fetch)
            result = predict_spacing_and_visibility(
                params,
                profile_name=profile_name,
                use_lake_profile=use_lake_profile,
            )

            spacings_NL.append(result["spacing_nonlinear"])
            spacings_core.append(result.get("spacing_core", float("nan")))
            spacings_CL.append(result.get("spacing_cl", float("nan")))
            spacings_response.append(result.get("spacing_response", float("nan")))
            spacings_L.append(result["spacing_linear"])
            visibility_index.append(result.get("visible_fraction", float("nan")))
            is_visible.append(result["is_visible"])
            Ra_values.append(result["Ra"])
            regimes.append(result["regime"])

        envelopes[depth] = {
            "wind": wind_range.tolist(),
            "spacing_NL": spacings_NL,
            "spacing_core": spacings_core,
            "spacing_CL": spacings_CL,
            "spacing_response": spacings_response,
            "spacing_L": spacings_L,
            "visibility_index": visibility_index,
            "is_visible": is_visible,
            "Ra": Ra_values,
            "regimes": regimes,
        }

    return envelopes


# ======================================================================
# Kappa diagnostic (NEW)
# ======================================================================

def kappa_diagnostic(
    results_df: pd.DataFrame,
    profile_name: str = "uniform",
    use_lake_profile: bool = False,
) -> dict:
    """Summarise the nonlinear amplitude coefficient diagnostic."""
    kappas = []
    obs_wind = []
    obs_ratio = []

    if use_lake_profile:
        kappa_theory = float(results_df["kappa"].dropna().mean()) if "kappa" in results_df else float("nan")
    else:
        profile = get_profile(profile_name)
        bcs = RobinBoundaryConditions()
        nl_result = solve_nonlinear(profile, bcs)
        kappa_theory = nl_result.kappa

    for _, row in results_df.iterrows():
        if pd.isna(row.get("observed_spacing_m")) or pd.isna(row.get("predicted_spacing_NL_m")):
            continue

        ratio = row["observed_spacing_m"] / row["predicted_spacing_NL_m"]
        obs_ratio.append(ratio)
        obs_wind.append(row.get("U10", float("nan")))
        kappas.append(row.get("kappa", kappa_theory))

    return {
        "kappa_theory": kappa_theory,
        "kappa_values": kappas,
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
    weather_summary: pd.DataFrame
    observation_diagnostics: pd.DataFrame | None
    metrics: dict
    envelopes: dict
    confounder_check: dict
    quality_check: dict
    kappa_check: dict | None
    diagnostic_model: dict | None = None
    output_dir: Path | None = None


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def filter_degraded_results(results: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    """Exclude NaN and degraded-physics rows from metric calculations."""
    valid = results[results[prediction_col].notna()].copy()
    if "fallback_triggered" in valid.columns:
        valid = valid[~valid["fallback_triggered"].fillna(False).astype(bool)]
    return valid


def load_observations(
    dataset_path: str,
    spacing_column: str = "manual_spacing_m",
    default_depth: float = 9.0,
    default_fetch: float = 15000.0,
) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    required = {"image_date", "authoritative_lat", "authoritative_lng", spacing_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    observations = pd.DataFrame(
        {
            "image_date": pd.to_datetime(df["image_date"], errors="coerce").dt.date,
            "authoritative_lat": pd.to_numeric(df["authoritative_lat"], errors="coerce"),
            "authoritative_lng": pd.to_numeric(df["authoritative_lng"], errors="coerce"),
            "observed_spacing_m": pd.to_numeric(df[spacing_column], errors="coerce"),
        }
    )
    if "observation_id" in df.columns:
        observations["observation_id"] = df["observation_id"]
    if "source_row" in df.columns:
        observations["source_row"] = df["source_row"]
    if "depth_m" in df.columns:
        observations["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce").fillna(default_depth)
    else:
        observations["depth_m"] = default_depth
    if "fetch_m" in df.columns:
        observations["fetch_m"] = pd.to_numeric(df["fetch_m"], errors="coerce").fillna(default_fetch)
    else:
        observations["fetch_m"] = default_fetch

    observations = observations.dropna(
        subset=["image_date", "authoritative_lat", "authoritative_lng", "observed_spacing_m"]
    )
    observations = observations[
        (observations["observed_spacing_m"] > 1.0) & (observations["observed_spacing_m"] < 2000.0)
    ]
    return observations.sort_values("image_date").reset_index(drop=True)


def validate_nonlinear(
    dataset_path: str,
    spacing_column: str = "manual_spacing_m",
    cache_dir: str = "data/era5_cache",
    output_dir: str | None = None,
    default_depth: float = 9.0,
    default_fetch: float = 15000.0,
    profile_name: str = "uniform",
    skip_download: bool = False,
    use_lake_profile: bool = False,
    timeline_hours_before: int = 48,
    timeline_hours_after: int = 48,
    observation_hour_utc: int = 12,
    dynamics: LangmuirDynamicsConfig | None = None,
) -> NonlinearValidationResult:
    """Run full nonlinear validation pipeline.

    1. Load observations
    2. Run confounder checks
    3. For each observation, predict spacing using nonlinear model
    4. Compute metrics (RMSE, R^2, capture rate)
    5. Generate consistency envelope
    6. Run kappa diagnostic
    """
    observations = load_observations(
        dataset_path=dataset_path,
        spacing_column=spacing_column,
        default_depth=default_depth,
        default_fetch=default_fetch,
    )
    weather_data = fetch_all_observations(observations, Path(cache_dir), skip_download=skip_download)

    # Predict spacing for each observation
    result_rows = []
    weather_rows = []
    diagnostics_rows = []
    timeline_frames: dict[str, pd.DataFrame] = {}
    nl_result = None
    if not use_lake_profile:
        profile = get_profile(profile_name)
        bcs = RobinBoundaryConditions()
        nl_result = solve_nonlinear(profile, bcs)

    for idx, row in observations.iterrows():
        depth = float(row["depth_m"])
        fetch = float(row["fetch_m"])
        lat = float(row["authoritative_lat"])
        lon = float(row["authoritative_lng"])
        image_date = row["image_date"].isoformat()
        obs_weather = weather_data.get(idx, {})

        if "error" in obs_weather:
            result_rows.append(
                {
                    "obs_index": idx,
                    "image_date": image_date,
                    "lat": lat,
                    "lon": lon,
                    "observed_spacing_m": row["observed_spacing_m"],
                    "predicted_spacing_NL_m": float("nan"),
                    "predicted_core_spacing_m": float("nan"),
                    "predicted_spacing_CL_m": float("nan"),
                    "predicted_spacing_response_m": float("nan"),
                    "predicted_spacing_visible_m": float("nan"),
                    "predicted_spacing_observable_m": float("nan"),
                    "predicted_visibility_index": float("nan"),
                    "predicted_spacing_L_m": float("nan"),
                    "error_NL_m": float("nan"),
                    "error_L_m": float("nan"),
                    "depth_m": depth,
                    "fetch_m": fetch,
                    "U10": float("nan"),
                    "U10_representative": float("nan"),
                    "U10_final_24h_mean": float("nan"),
                    "U10_final_6h_mean": float("nan"),
                    "U10_10day_mean": float("nan"),
                    "U10_10day_std": float("nan"),
                    "wind_dir_dominant": float("nan"),
                    "wind_steadiness": float("nan"),
                    "wind_regime": "",
                    "Ra": float("nan"),
                    "regime": "weather_error",
                    "kappa": float("nan"),
                    "selected_l": float("nan"),
                    "selected_l_cl": float("nan"),
                    "target_l": float("nan"),
                    "target_l_cl": float("nan"),
                    "unstable_l_min": float("nan"),
                    "unstable_l_max": float("nan"),
                    "peak_growth_proxy": float("nan"),
                    "amplitude_index": float("nan"),
                    "development_index": float("nan"),
                    "response_bandwidth": float("nan"),
                    "response_mix": float("nan"),
                    "large_scale_fraction": float("nan"),
                    "visible_fraction": float("nan"),
                    "coarsening_index": float("nan"),
                    "drive_fast": float("nan"),
                    "drive_slow": float("nan"),
                    "direction_persistence": float("nan"),
                    "decay_memory": float("nan"),
                    "event_age_hours": float("nan"),
                    "decay_age_hours": float("nan"),
                    "response_separation": float("nan"),
                    "cl_drive_integral": float("nan"),
                    "forcing_depth_fraction": float("nan"),
                    "shear_to_drift_ratio": float("nan"),
                    "cancellation_index": float("nan"),
                    "state_saturation_flag": False,
                    "mismatch_class": "weather_error",
                    "is_visible": False,
                    "accumulation_factor": float("nan"),
                    "w_down_max": float("nan"),
                    "download_error": obs_weather["error"],
                    "fallback_triggered": True,
                    "fallback_reason": obs_weather["error"],
                    "physics_status": "weather_error",
                    "timeline_status": "weather_error",
                }
            )
            continue

        forcing = summarise_spinup_forcing(obs_weather["spinup"])
        pre_context = summarise_context_window(obs_weather["pre_context"], "pre_context")
        post_context = summarise_context_window(obs_weather["post_context"], "post_context")
        wind_regime = classify_wind_regime(forcing, pre_context, post_context)
        U10 = float(forcing["U10_representative"])
        timeline_df, diagnostics = predict_observation_timeline(
            row,
            obs_weather,
            profile_name=profile_name,
            use_lake_profile=use_lake_profile,
            hours_before=timeline_hours_before,
            hours_after=timeline_hours_after,
            observation_hour_utc=observation_hour_utc,
            dynamics=dynamics,
        )
        timeline_complete = len(timeline_df) > 0 and diagnostics.get("status") == "complete"
        if timeline_complete:
            spacing_nl = float(diagnostics["spacing_at_obs_m"])
            spacing_core = float(diagnostics.get("spacing_core_at_obs_m", float("nan")))
            spacing_cl = float(diagnostics.get("spacing_cl_at_obs_m", float("nan")))
            spacing_response = float(diagnostics.get("spacing_response_at_obs_m", float("nan")))
            spacing_l = float(diagnostics.get("spacing_linear_at_obs_m", float("nan")))
            Ra_at_obs = float(diagnostics.get("Ra_at_obs", float("nan")))
            kappa_at_obs = float(diagnostics.get("kappa_at_obs", float("nan")))
            wavenumber_ratio = float(diagnostics.get("wavenumber_ratio_at_obs", float("nan")))
            fallback_triggered = bool(diagnostics.get("fallback_at_obs", False))
            fallback_reason = diagnostics.get("physics_status_at_obs", "") if fallback_triggered else ""
            physics_status = diagnostics.get("physics_status_at_obs", "complete")
            is_visible = bool(
                timeline_df.loc[np.argmin(np.abs(timeline_df["hours_from_observation"])), "is_visible"]
            )
            accumulation_factor = float(
                timeline_df.loc[np.argmin(np.abs(timeline_df["hours_from_observation"])), "accumulation_factor"]
            )
            w_down_max = float(
                timeline_df.loc[np.argmin(np.abs(timeline_df["hours_from_observation"])), "w_down_max"]
            )
        else:
            spacing_nl = float("nan")
            spacing_core = float("nan")
            spacing_cl = float("nan")
            spacing_response = float("nan")
            spacing_l = float("nan")
            Ra_at_obs = float("nan")
            kappa_at_obs = float("nan")
            wavenumber_ratio = float("nan")
            fallback_triggered = True
            fallback_reason = diagnostics.get("reason", "timeline_incomplete")
            physics_status = diagnostics.get("status", "timeline_incomplete")
            is_visible = False
            accumulation_factor = float("nan")
            w_down_max = float("nan")

        result_rows.append(
            {
                "obs_index": idx,
                "image_date": image_date,
                "lat": lat,
                "lon": lon,
                "observed_spacing_m": row["observed_spacing_m"],
                "predicted_spacing_NL_m": spacing_nl,
                "predicted_core_spacing_m": spacing_core,
                "predicted_spacing_CL_m": spacing_cl,
                "predicted_spacing_response_m": spacing_response,
                "predicted_spacing_visible_m": float(diagnostics.get("spacing_visible_at_obs_m", float("nan"))),
                "predicted_spacing_observable_m": float(diagnostics.get("spacing_observable_at_obs_m", float("nan"))),
                "predicted_visibility_index": float(diagnostics.get("visibility_at_obs", float("nan"))),
                "predicted_spacing_L_m": spacing_l,
                "error_NL_m": spacing_nl - row["observed_spacing_m"]
                if not math.isnan(spacing_nl)
                else float("nan"),
                "error_L_m": spacing_l - row["observed_spacing_m"]
                if not math.isnan(spacing_l)
                else float("nan"),
                "depth_m": depth,
                "fetch_m": fetch,
                "U10": U10,
                "U10_representative": U10,
                "U10_final_24h_mean": forcing["U10_final_24h_mean"],
                "U10_final_6h_mean": forcing["U10_final_6h_mean"],
                "U10_10day_mean": forcing["U10_10day_mean"],
                "U10_10day_std": forcing["U10_10day_std"],
                "wind_dir_dominant": forcing["wind_dir_dominant"],
                "wind_steadiness": forcing["wind_steadiness"],
                "wind_regime": wind_regime,
                "Ra": Ra_at_obs,
                "regime": diagnostics.get("regime_at_obs", "") if timeline_complete else "timeline_incomplete",
                "hydrodynamic_regime": diagnostics.get("hydrodynamic_regime_at_obs", "") if timeline_complete else "",
                "kappa": kappa_at_obs,
                "wavenumber_ratio": wavenumber_ratio,
                "selected_l": float("nan") if not timeline_complete else float(diagnostics.get("selected_l_at_obs", float("nan"))),
                "selected_l_cl": float("nan") if not timeline_complete else float(diagnostics.get("selected_l_cl_at_obs", float("nan"))),
                "target_l": float("nan"),
                "target_l_cl": float("nan") if not timeline_complete else float(diagnostics.get("target_l_cl_at_obs", float("nan"))),
                "unstable_l_min": float("nan"),
                "unstable_l_max": float("nan"),
                "peak_growth_proxy": float("nan"),
                "amplitude_index": float("nan") if not timeline_complete else float(diagnostics.get("amplitude_at_obs", float("nan"))),
                "development_index": float("nan") if not timeline_complete else float(diagnostics.get("development_at_obs", float("nan"))),
                "response_bandwidth": float("nan") if not timeline_complete else float(diagnostics.get("response_bandwidth_at_obs", float("nan"))),
                "response_mix": float("nan") if not timeline_complete else float(diagnostics.get("response_mix_at_obs", float("nan"))),
                "large_scale_fraction": float("nan") if not timeline_complete else float(diagnostics.get("large_scale_fraction_at_obs", float("nan"))),
                "visible_fraction": float("nan") if not timeline_complete else float(diagnostics.get("visible_fraction_at_obs", float("nan"))),
                "coarsening_index": float("nan") if not timeline_complete else float(diagnostics.get("coarsening_index_at_obs", float("nan"))),
                "drive_fast": float("nan") if not timeline_complete else float(diagnostics.get("drive_fast_at_obs", float("nan"))),
                "drive_slow": float("nan") if not timeline_complete else float(diagnostics.get("drive_slow_at_obs", float("nan"))),
                "direction_persistence": float("nan") if not timeline_complete else float(diagnostics.get("direction_persistence_at_obs", float("nan"))),
                "decay_memory": float("nan") if not timeline_complete else float(diagnostics.get("decay_memory_at_obs", float("nan"))),
                "event_age_hours": float("nan") if not timeline_complete else float(diagnostics.get("event_age_at_obs_h", float("nan"))),
                "decay_age_hours": float("nan") if not timeline_complete else float(diagnostics.get("decay_age_at_obs_h", float("nan"))),
                "response_separation": float("nan") if not timeline_complete else float(diagnostics.get("response_separation_at_obs", float("nan"))),
                "cl_drive_integral": float("nan") if not timeline_complete else float(diagnostics.get("cl_drive_integral_at_obs", float("nan"))),
                "forcing_depth_fraction": float("nan") if not timeline_complete else float(diagnostics.get("forcing_depth_fraction_at_obs", float("nan"))),
                "shear_to_drift_ratio": float("nan") if not timeline_complete else float(diagnostics.get("shear_to_drift_ratio_at_obs", float("nan"))),
                "cancellation_index": float("nan") if not timeline_complete else float(diagnostics.get("cancellation_index_at_obs", float("nan"))),
                "state_saturation_flag": False if not timeline_complete else bool(diagnostics.get("state_saturated_at_obs", False)),
                "mismatch_class": "timeline_incomplete" if not timeline_complete else str(diagnostics.get("mismatch_class", "")),
                "is_visible": is_visible,
                "accumulation_factor": accumulation_factor,
                "w_down_max": w_down_max,
                "download_error": "",
                "fallback_triggered": fallback_triggered,
                "fallback_reason": fallback_reason,
                "physics_status": physics_status,
                "timeline_status": diagnostics.get("status", "skipped"),
            }
        )
        weather_rows.append(
            {
                "obs_index": idx,
                "image_date": image_date,
                **forcing,
                "pre_wind_mean": pre_context["wind_mean"],
                "post_wind_mean": post_context["wind_mean"],
                "pre_temp_mean": pre_context["temp_mean"],
                "post_temp_mean": post_context["temp_mean"],
                "wind_regime": wind_regime,
            }
        )
        if len(timeline_df) > 0 and diagnostics.get("status") == "complete":
            obs_key = str(row.get("observation_id", f"obs_{idx:03d}"))
            timeline_frames[obs_key] = timeline_df
            diagnostics_rows.append(diagnostics)

    results = pd.DataFrame(result_rows)
    weather_summary = pd.DataFrame(weather_rows)
    observation_diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostic_model = (
        fit_interpretable_diagnostic_model(observation_diagnostics)
        if len(observation_diagnostics) > 0
        else None
    )

    # Compute metrics
    valid_nl = filter_degraded_results(results, "predicted_spacing_NL_m")
    valid_l = filter_degraded_results(results, "predicted_spacing_L_m")

    metrics = {
        "spacing_column": spacing_column,
        "profile": "shallow_lake_dynamic" if use_lake_profile else profile_name,
        "n_observations": int(len(results)),
        "n_valid_NL": int(len(valid_nl)),
        "n_valid_L": int(len(valid_l)),
        "n_fallback": int(results.get("fallback_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
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

    timeline_valid = observation_diagnostics.dropna(subset=["observed_spacing_m", "spacing_at_obs_m"])
    if len(timeline_valid) > 0:
        timeline_error = timeline_valid["spacing_at_obs_m"] - timeline_valid["observed_spacing_m"]
        metrics["rmse_timeline_obs_m"] = float(np.sqrt(np.mean(timeline_error ** 2)))
        metrics["bias_timeline_obs_m"] = float(timeline_error.mean())
        metrics["r_squared_timeline_obs"] = _r_squared(
            timeline_valid["observed_spacing_m"].to_numpy(dtype=float),
            timeline_valid["spacing_at_obs_m"].to_numpy(dtype=float),
        )
        ratio_timeline = timeline_valid["spacing_at_obs_m"] / timeline_valid["observed_spacing_m"]
        metrics["capture_rate_timeline_obs_2x"] = float(((ratio_timeline > 0.5) & (ratio_timeline < 2.0)).mean())
        metrics["timeline_spacing_q10_m"] = float(timeline_valid["spacing_at_obs_m"].quantile(0.10))
        metrics["timeline_spacing_q90_m"] = float(timeline_valid["spacing_at_obs_m"].quantile(0.90))
        metrics["timeline_spacing_std_mean_m"] = float(timeline_valid["spacing_std_prev_48h_m"].mean())
        if "spacing_core_at_obs_m" in timeline_valid.columns:
            metrics["timeline_core_spacing_q10_m"] = float(timeline_valid["spacing_core_at_obs_m"].quantile(0.10))
            metrics["timeline_core_spacing_q90_m"] = float(timeline_valid["spacing_core_at_obs_m"].quantile(0.90))
            core_error = timeline_valid["spacing_core_at_obs_m"] - timeline_valid["observed_spacing_m"]
            metrics["rmse_timeline_core_obs_m"] = float(np.sqrt(np.mean(core_error ** 2)))
        if "visibility_at_obs" in timeline_valid.columns:
            metrics["timeline_visibility_q10"] = float(timeline_valid["visibility_at_obs"].quantile(0.10))
            metrics["timeline_visibility_q90"] = float(timeline_valid["visibility_at_obs"].quantile(0.90))
        if "core_spacing_range_prev_48h_m" in timeline_valid.columns:
            metrics["timeline_core_spacing_range_mean_m"] = float(timeline_valid["core_spacing_range_prev_48h_m"].mean())
        if "state_saturated_fraction_prev_48h" in timeline_valid.columns:
            metrics["timeline_state_saturated_fraction"] = float(timeline_valid["state_saturated_fraction_prev_48h"].mean())
        if "integrated_supercriticality_prev_48h" in timeline_valid.columns:
            metrics["timeline_spacing_vs_integrated_supercriticality_corr"] = float(
                timeline_valid["spacing_at_obs_m"].corr(timeline_valid["integrated_supercriticality_prev_48h"])
            )
        if "wind_mean_prev_48h" in timeline_valid.columns:
            metrics["timeline_spacing_vs_wind_mean_corr"] = float(
                timeline_valid["spacing_at_obs_m"].corr(timeline_valid["wind_mean_prev_48h"])
            )
        if "spacing_core_at_obs_m" in timeline_valid.columns and "wind_mean_prev_48h" in timeline_valid.columns:
            metrics["timeline_core_spacing_vs_wind_mean_corr"] = float(
                timeline_valid["spacing_core_at_obs_m"].corr(timeline_valid["wind_mean_prev_48h"])
            )
        if "spacing_cl_at_obs_m" in timeline_valid.columns:
            cl_valid = timeline_valid.dropna(subset=["spacing_cl_at_obs_m"])
            if len(cl_valid) > 0:
                cl_error = cl_valid["spacing_cl_at_obs_m"] - cl_valid["observed_spacing_m"]
                metrics["rmse_timeline_cl_obs_m"] = float(np.sqrt(np.mean(cl_error ** 2)))
                metrics["bias_timeline_cl_obs_m"] = float(cl_error.mean())
        if "spacing_response_at_obs_m" in timeline_valid.columns:
            response_valid = timeline_valid.dropna(subset=["spacing_response_at_obs_m"])
            if len(response_valid) > 0:
                response_error = response_valid["spacing_response_at_obs_m"] - response_valid["observed_spacing_m"]
                metrics["rmse_timeline_response_obs_m"] = float(np.sqrt(np.mean(response_error ** 2)))
                metrics["bias_timeline_response_obs_m"] = float(response_error.mean())
        if "coherent_run_hours_at_obs" in timeline_valid.columns:
            metrics["timeline_spacing_vs_coherent_run_corr"] = float(
                timeline_valid["spacing_at_obs_m"].corr(timeline_valid["coherent_run_hours_at_obs"])
            )
        if "coherent_run_hours_mean_prev_48h" in timeline_valid.columns:
            metrics["timeline_spacing_vs_coherent_run_mean_corr"] = float(
                timeline_valid["spacing_at_obs_m"].corr(timeline_valid["coherent_run_hours_mean_prev_48h"])
            )
        if "setup_at_obs" in timeline_valid.columns:
            metrics["timeline_spacing_vs_setup_corr"] = float(
                timeline_valid["spacing_at_obs_m"].corr(timeline_valid["setup_at_obs"])
            )
        if "cancellation_index_at_obs" in timeline_valid.columns:
            metrics["timeline_spacing_vs_cancellation_corr"] = float(
                timeline_valid["spacing_at_obs_m"].corr(timeline_valid["cancellation_index_at_obs"])
            )
        if "mismatch_class" in timeline_valid.columns:
            mismatch_counts = timeline_valid["mismatch_class"].value_counts(dropna=False).to_dict()
            metrics["mismatch_class_counts"] = {str(key): int(value) for key, value in mismatch_counts.items()}

    # R0 and kappa from the model
    if nl_result is not None:
        metrics["R0"] = float(nl_result.R0)
        metrics["kappa"] = float(nl_result.kappa)
        metrics["wavenumber_ratio"] = float(nl_result.wavenumber_ratio)
        metrics["lcNL"] = float(nl_result.lcNL)
        metrics["lcL"] = float(nl_result.linear_result.lcL)
        metrics["RcNL"] = float(nl_result.RcNL)
        metrics["RcL"] = float(nl_result.linear_result.RcL)
    elif len(valid_nl) > 0:
        metrics["kappa_mean"] = float(valid_nl["kappa"].mean())
        metrics["kappa_min"] = float(valid_nl["kappa"].min())
        metrics["kappa_max"] = float(valid_nl["kappa"].max())
        if "wavenumber_ratio" in valid_nl:
            metrics["wavenumber_ratio_mean"] = float(valid_nl["wavenumber_ratio"].mean())
        metrics["Ra_mean"] = float(valid_nl["Ra"].mean())
        metrics["Ra_min"] = float(valid_nl["Ra"].min())
        metrics["Ra_max"] = float(valid_nl["Ra"].max())

    if diagnostic_model is not None:
        metrics["diagnostic_best_family"] = diagnostic_model.get("best_family_by_loocv_r2")
        metrics["diagnostic_best_family_loocv_r2"] = diagnostic_model.get("best_family_loocv_r2")

    # Confounder checks
    confounder = check_wind_spacing_confounders(results)
    quality = check_observation_quality_bias(results)

    # Envelope
    envelopes = nonlinear_consistency_envelope(
        results,
        depths=[5.0, 9.0, 15.0],
        profile_name=profile_name,
        use_lake_profile=use_lake_profile,
    )

    # Kappa diagnostic
    kappa_check = None
    if len(valid_nl) > 0:
        kappa_check = kappa_diagnostic(
            results,
            profile_name=profile_name,
            use_lake_profile=use_lake_profile,
        )

    result = NonlinearValidationResult(
        results=results,
        weather_summary=weather_summary,
        observation_diagnostics=observation_diagnostics,
        metrics=metrics,
        envelopes=envelopes,
        confounder_check=confounder,
        quality_check=quality,
        kappa_check=kappa_check,
        diagnostic_model=diagnostic_model,
    )

    # Write outputs
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results.to_csv(out / "results.csv", index=False)
        weather_summary.to_csv(out / "weather_summary.csv", index=False)
        if len(observation_diagnostics) > 0:
            write_observation_timelines(observation_diagnostics, timeline_frames, out)
        if diagnostic_model is not None:
            with (out / "diagnostic_model.json").open("w") as f:
                json.dump(diagnostic_model, f, indent=2, default=str)
        with (out / "dynamics_config.json").open("w") as f:
            json.dump((dynamics or LangmuirDynamicsConfig()).to_dict(), f, indent=2, default=str)
        with (out / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2, default=str)
        with (out / "confounder_check.json").open("w") as f:
            json.dump(confounder, f, indent=2, default=str)
        result.output_dir = out

    return result
