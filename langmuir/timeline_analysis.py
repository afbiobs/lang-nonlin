"""Observation-centred hourly timeline analysis for Langmuir validation."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .colony_accumulation import LangmuirModeState, advance_langmuir_state
from .params import LCParams
from .weather import (
    classify_wind_regime,
    directional_steadiness,
    direction_change_series,
    forcing_coherence_series,
    summarise_context_window,
)

DEFAULT_OBSERVATION_HOUR_UTC = 12
DEFAULT_TIMELINE_HOURS_BEFORE = 48
DEFAULT_TIMELINE_HOURS_AFTER = 48


def observation_timestamp(
    image_date: date,
    observation_hour_utc: int = DEFAULT_OBSERVATION_HOUR_UTC,
) -> pd.Timestamp:
    """Map a date-only observation to a UTC timestamp.

    The source dataset only contains dates, so timelines are centred on noon UTC
    by default to avoid bias toward either edge of the day.
    """
    return pd.Timestamp(image_date).tz_localize("UTC") + pd.Timedelta(hours=observation_hour_utc)


def combine_weather_windows(obs_weather: dict) -> pd.DataFrame:
    """Combine cached observation weather windows into a single hourly frame."""
    frames = []
    for key in ("spinup", "timeline", "post_context"):
        frame = obs_weather.get(key)
        if isinstance(frame, pd.DataFrame) and len(frame) > 0:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def extract_observation_timeline_weather(
    obs_weather: dict,
    image_date: date,
    *,
    hours_before: int = DEFAULT_TIMELINE_HOURS_BEFORE,
    hours_after: int = DEFAULT_TIMELINE_HOURS_AFTER,
    observation_hour_utc: int = DEFAULT_OBSERVATION_HOUR_UTC,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Extract the hourly observation-centred forcing window."""
    image_time = observation_timestamp(image_date, observation_hour_utc=observation_hour_utc)
    start = image_time - pd.Timedelta(hours=hours_before)
    end = image_time + pd.Timedelta(hours=hours_after)

    if isinstance(obs_weather.get("timeline"), pd.DataFrame):
        timeline = obs_weather["timeline"].sort_index()
    else:
        timeline = combine_weather_windows(obs_weather)
    if len(timeline) == 0:
        return pd.DataFrame(), image_time

    timeline = timeline.loc[(timeline.index >= start) & (timeline.index <= end)].copy()
    return timeline, image_time


def _window_slice(
    df: pd.DataFrame,
    end_time: pd.Timestamp,
    *,
    hours: int,
    include_end: bool = True,
) -> pd.DataFrame:
    start_time = end_time - pd.Timedelta(hours=hours)
    if include_end:
        return df.loc[(df.index >= start_time) & (df.index <= end_time)]
    return df.loc[(df.index >= start_time) & (df.index < end_time)]


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else float("nan")


def _safe_std(series: pd.Series) -> float:
    return float(series.std()) if len(series) > 1 else 0.0


def _safe_max(series: pd.Series) -> float:
    return float(series.max()) if len(series) > 0 else float("nan")


def _safe_min(series: pd.Series) -> float:
    return float(series.min()) if len(series) > 0 else float("nan")


def _safe_sum(series: pd.Series) -> float:
    return float(series.sum()) if len(series) > 0 else 0.0


def predict_observation_timeline(
    observation_row: pd.Series,
    obs_weather: dict,
    *,
    profile_name: str = "uniform",
    use_lake_profile: bool = False,
    hours_before: int = DEFAULT_TIMELINE_HOURS_BEFORE,
    hours_after: int = DEFAULT_TIMELINE_HOURS_AFTER,
    observation_hour_utc: int = DEFAULT_OBSERVATION_HOUR_UTC,
    relaxation_hours: float = 12.0,
    decay_hours: float = 18.0,
) -> tuple[pd.DataFrame, dict]:
    """Predict hourly LC evolution around a single observation."""
    image_date = observation_row["image_date"]
    timeline_weather, image_time = extract_observation_timeline_weather(
        obs_weather,
        image_date,
        hours_before=hours_before,
        hours_after=hours_after,
        observation_hour_utc=observation_hour_utc,
    )
    forcing_weather = combine_weather_windows(obs_weather)
    if len(forcing_weather) == 0 or len(timeline_weather) == 0:
        return pd.DataFrame(), {
            "status": "skipped",
            "reason": "missing weather timeline",
            "image_time": str(image_time),
        }
    forcing_weather = forcing_weather.loc[
        (forcing_weather.index >= forcing_weather.index.min()) &
        (forcing_weather.index <= timeline_weather.index.max())
    ].copy()

    coherence = forcing_coherence_series(forcing_weather["wind_direction_10m"])
    turn_angle = direction_change_series(forcing_weather["wind_direction_10m"])
    depth = float(observation_row["depth_m"])
    fetch = float(observation_row["fetch_m"])

    state = LangmuirModeState()
    rows: list[dict] = []
    previous_time: pd.Timestamp | None = None
    for timestamp, weather_row in forcing_weather.iterrows():
        if previous_time is None:
            dt_hours = 1.0
        else:
            dt_hours = max((timestamp - previous_time).total_seconds() / 3600.0, 1e-6)
        previous_time = timestamp

        params = LCParams(
            U10=float(weather_row["wind_speed_10m"]),
            depth=depth,
            fetch=fetch,
        )
        result = advance_langmuir_state(
            params,
            profile_name=profile_name,
            use_lake_profile=use_lake_profile,
            previous_state=state,
            dt_hours=dt_hours,
            relaxation_hours=relaxation_hours,
            decay_hours=decay_hours,
            forcing_coherence=float(coherence.loc[timestamp]),
        )
        state = result["mode_state"]

        if timestamp < timeline_weather.index.min() or timestamp > timeline_weather.index.max():
            continue

        row = {
            "time": timestamp,
            "hours_from_observation": float((timestamp - image_time).total_seconds() / 3600.0),
            "U10": float(weather_row["wind_speed_10m"]),
            "wind_direction_10m": float(weather_row["wind_direction_10m"]),
            "wind_gusts_10m": float(weather_row.get("wind_gusts_10m", float("nan"))),
            "temperature_2m": float(weather_row.get("temperature_2m", float("nan"))),
            "shortwave_radiation": float(weather_row.get("shortwave_radiation", float("nan"))),
            "cloud_cover": float(weather_row.get("cloud_cover", float("nan"))),
            "precipitation": float(weather_row.get("precipitation", float("nan"))),
            "forcing_coherence": float(coherence.loc[timestamp]),
            "wind_turn_angle_deg": float(turn_angle.loc[timestamp]),
            "u_star": params.u_star,
            "H_s": params.H_s,
            "T_p": params.T_p,
            "lambda_p": params.lambda_p,
            "La_t": params.La_t,
            "D_max": params.D_max,
            "Ra": result["Ra"],
            "RcNL": result.get("RcNL", float("nan")),
            "supercriticality_ratio": max(
                (result["Ra"] - result.get("RcNL", float("nan"))) / max(result.get("RcNL", 1.0), 1e-12),
                0.0,
            ) if not math.isnan(result.get("RcNL", float("nan"))) else float("nan"),
            "predicted_spacing_NL_m": result["spacing_nonlinear"],
            "predicted_spacing_L_m": result["spacing_linear"],
            "selected_l": result["selected_l"],
            "target_l": result["target_l"],
            "unstable_l_min": result["unstable_l_min"],
            "unstable_l_max": result["unstable_l_max"],
            "unstable_band_width": (
                result["unstable_l_max"] - result["unstable_l_min"]
                if not math.isnan(result["unstable_l_min"]) and not math.isnan(result["unstable_l_max"])
                else float("nan")
            ),
            "peak_growth_proxy": result["peak_growth_proxy"],
            "amplitude_index": result["amplitude_index"],
            "development_index": result["development_index"],
            "regime": result["regime"],
            "is_visible": result["is_visible"],
            "accumulation_factor": result["accumulation_factor"],
            "bloom_feedback": result["bloom_feedback"],
            "w_down_max": result["w_down_max"],
            "kappa": result["kappa"],
        }
        rows.append(row)

    timeline_df = pd.DataFrame(rows)
    if len(timeline_df) == 0:
        return timeline_df, {
            "status": "skipped",
            "reason": "empty hourly timeline",
            "image_time": str(image_time),
        }

    timeline_df = timeline_df.sort_values("time").reset_index(drop=True)
    return timeline_df, summarise_observation_timeline(
        timeline_df,
        observation_row=observation_row,
        obs_weather=obs_weather,
        image_time=image_time,
        hours_before=hours_before,
    )


def summarise_observation_timeline(
    timeline_df: pd.DataFrame,
    *,
    observation_row: pd.Series,
    obs_weather: dict,
    image_time: pd.Timestamp,
    hours_before: int,
) -> dict:
    """Summarise the dynamics around one observation into interpretable diagnostics."""
    df = timeline_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    pre_df = _window_slice(df, image_time, hours=hours_before, include_end=True)
    post_df = df.loc[df.index >= image_time]
    obs_row = df.loc[[image_time]].iloc[0] if image_time in df.index else df.iloc[np.argmin(np.abs(df.index - image_time))]

    if isinstance(obs_weather.get("pre_context"), pd.DataFrame):
        pre_context = summarise_context_window(obs_weather["pre_context"], "pre_context")
    else:
        pre_context = {}
    if isinstance(obs_weather.get("post_context"), pd.DataFrame):
        post_context = summarise_context_window(obs_weather["post_context"], "post_context")
    else:
        post_context = {}

    summary = {
        "status": "complete",
        "observation_id": observation_row.get("observation_id", ""),
        "image_date": str(observation_row["image_date"]),
        "image_time_utc": str(image_time),
        "observed_spacing_m": float(observation_row["observed_spacing_m"]),
        "depth_m": float(observation_row["depth_m"]),
        "fetch_m": float(observation_row["fetch_m"]),
        "timeline_hours": int(len(df)),
        "timeline_start_utc": str(df.index.min()),
        "timeline_end_utc": str(df.index.max()),
        "spacing_at_obs_m": float(obs_row["predicted_spacing_NL_m"]),
        "spacing_linear_at_obs_m": float(obs_row["predicted_spacing_L_m"]),
        "spacing_mean_prev_48h_m": _safe_mean(pre_df["predicted_spacing_NL_m"]),
        "spacing_std_prev_48h_m": _safe_std(pre_df["predicted_spacing_NL_m"]),
        "spacing_range_prev_48h_m": _safe_max(pre_df["predicted_spacing_NL_m"]) - _safe_min(pre_df["predicted_spacing_NL_m"]),
        "spacing_mean_post_48h_m": _safe_mean(post_df["predicted_spacing_NL_m"]),
        "wind_mean_prev_48h": _safe_mean(pre_df["U10"]),
        "wind_max_prev_48h": _safe_max(pre_df["U10"]),
        "wind_std_prev_48h": _safe_std(pre_df["U10"]),
        "gust_mean_prev_48h": _safe_mean(pre_df["wind_gusts_10m"]),
        "wind_steadiness_prev_48h": directional_steadiness(pre_df["U10"], pre_df["wind_direction_10m"]),
        "turning_mean_prev_48h_deg": _safe_mean(pre_df["wind_turn_angle_deg"]),
        "turning_total_prev_48h_deg": _safe_sum(pre_df["wind_turn_angle_deg"]),
        "coherence_mean_prev_48h": _safe_mean(pre_df["forcing_coherence"]),
        "Hs_mean_prev_48h": _safe_mean(pre_df["H_s"]),
        "lambda_p_mean_prev_48h": _safe_mean(pre_df["lambda_p"]),
        "La_t_mean_prev_48h": _safe_mean(pre_df["La_t"]),
        "D_max_mean_prev_48h": _safe_mean(pre_df["D_max"]),
        "hours_supercritical_prev_48h": float((pre_df["regime"] != "subcritical").sum()),
        "integrated_supercriticality_prev_48h": _safe_sum(pre_df["supercriticality_ratio"]),
        "amplitude_at_obs": float(obs_row["amplitude_index"]),
        "amplitude_mean_prev_48h": _safe_mean(pre_df["amplitude_index"]),
        "amplitude_max_prev_48h": _safe_max(pre_df["amplitude_index"]),
        "development_at_obs": float(obs_row["development_index"]),
        "development_mean_prev_48h": _safe_mean(pre_df["development_index"]),
        "development_max_prev_48h": _safe_max(pre_df["development_index"]),
        "selected_l_at_obs": float(obs_row["selected_l"]),
        "selected_l_std_prev_48h": _safe_std(pre_df["selected_l"]),
        "unstable_band_width_mean_prev_48h": _safe_mean(pre_df["unstable_band_width"]),
        "peak_growth_mean_prev_48h": _safe_mean(pre_df["peak_growth_proxy"]),
        "wind_regime": classify_wind_regime(
            {
                "U10_10day_mean": _safe_mean(pre_df["U10"]),
            },
            pre_context if pre_context else {"wind_mean": float("nan"), "wind_std": float("nan")},
            post_context if post_context else {"wind_mean": float("nan"), "wind_std": float("nan")},
        ) if pre_context and post_context else "",
    }
    return summary


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
) -> dict:
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    Xs = (X - x_mean) / x_std
    y_mean = float(np.mean(y))
    ys = y - y_mean
    beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ ys)
    pred = y_mean + Xs @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "beta": beta,
        "x_mean": x_mean,
        "x_std": x_std,
        "intercept": y_mean,
        "pred": pred,
        "r2": r2,
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }


def _loocv_r2(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> float:
    if len(y) < 3:
        return float("nan")
    preds = []
    for idx in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[idx] = False
        fit = _fit_ridge(X[mask], y[mask], alpha=alpha)
        x_row = (X[idx] - fit["x_mean"]) / fit["x_std"]
        preds.append(float(fit["intercept"] + x_row @ fit["beta"]))
    preds_arr = np.asarray(preds)
    ss_res = float(np.sum((y - preds_arr) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_interpretable_diagnostic_model(
    diagnostics_df: pd.DataFrame,
    *,
    target_col: str = "observed_spacing_m",
    alpha: float = 1.0,
) -> dict:
    """Fit simple interpretable ridge models to grouped diagnostic features."""
    feature_families = {
        "wind_history": [
            "wind_mean_prev_48h",
            "wind_max_prev_48h",
            "wind_std_prev_48h",
            "integrated_supercriticality_prev_48h",
        ],
        "wave_history_proxy": [
            "Hs_mean_prev_48h",
            "lambda_p_mean_prev_48h",
            "La_t_mean_prev_48h",
            "D_max_mean_prev_48h",
        ],
        "direction_persistence": [
            "wind_steadiness_prev_48h",
            "turning_mean_prev_48h_deg",
            "turning_total_prev_48h_deg",
            "coherence_mean_prev_48h",
        ],
        "model_structure": [
            "spacing_at_obs_m",
            "spacing_mean_prev_48h_m",
            "amplitude_at_obs",
            "development_at_obs",
            "unstable_band_width_mean_prev_48h",
        ],
    }
    feature_families["combined"] = sorted(
        {feature for features in feature_families.values() for feature in features}
    )

    output: dict[str, dict] = {}
    y_all = diagnostics_df[target_col].to_numpy(dtype=float)
    for family, features in feature_families.items():
        subset = diagnostics_df[features + [target_col]].dropna()
        if len(subset) < max(6, len(features) + 2):
            output[family] = {
                "status": "skipped",
                "reason": "insufficient complete cases",
                "n_samples": int(len(subset)),
                "features": features,
            }
            continue

        X = subset[features].to_numpy(dtype=float)
        y = subset[target_col].to_numpy(dtype=float)
        fit = _fit_ridge(X, y, alpha=alpha)
        coef_by_feature = {
            feature: float(coef) for feature, coef in zip(features, fit["beta"])
        }
        correlations = {
            feature: float(np.corrcoef(subset[feature], y)[0, 1])
            if np.std(subset[feature]) > 1e-12 and np.std(y) > 1e-12
            else float("nan")
            for feature in features
        }
        output[family] = {
            "status": "complete",
            "n_samples": int(len(subset)),
            "features": features,
            "r2_in_sample": float(fit["r2"]),
            "r2_loocv": float(_loocv_r2(X, y, alpha=alpha)),
            "rmse_in_sample_m": float(fit["rmse"]),
            "coefficients_standardized": coef_by_feature,
            "feature_correlations": correlations,
        }

    best_family = None
    best_score = -float("inf")
    for family, report in output.items():
        score = report.get("r2_loocv")
        if report.get("status") == "complete" and score is not None and not math.isnan(score) and score > best_score:
            best_score = score
            best_family = family
    output["best_family_by_loocv_r2"] = best_family
    output["best_family_loocv_r2"] = float(best_score) if best_family is not None else float("nan")
    return output


def write_observation_timelines(
    diagnostics_df: pd.DataFrame,
    timeline_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write per-observation timelines and the diagnostics summary."""
    timelines_dir = output_dir / "timelines"
    timelines_dir.mkdir(parents=True, exist_ok=True)
    for obs_key, frame in timeline_frames.items():
        frame.to_csv(timelines_dir / f"{obs_key}_timeline.csv", index=False)
    diagnostics_df.to_csv(output_dir / "observation_diagnostics.csv", index=False)
