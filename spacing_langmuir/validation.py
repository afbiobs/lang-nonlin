from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LCParams
from .era5 import fetch_all_observations
from .stability import find_critical_wavenumber
from .weather import classify_wind_regime, extract_model_forcing, summarise_context_window


@dataclass
class ValidationResult:
    results: pd.DataFrame
    weather_summary: pd.DataFrame
    metrics: dict
    output_dir: Path | None = None


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
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "image_date": pd.to_datetime(df["image_date"], errors="coerce").dt.date,
            "authoritative_lat": pd.to_numeric(df["authoritative_lat"], errors="coerce"),
            "authoritative_lng": pd.to_numeric(df["authoritative_lng"], errors="coerce"),
            "observed_spacing_m": pd.to_numeric(df[spacing_column], errors="coerce"),
        }
    )
    if "observation_id" in df.columns:
        out["observation_id"] = df["observation_id"]
    if "source_row" in df.columns:
        out["source_row"] = df["source_row"]
    if "depth_m" in df.columns:
        out["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce").fillna(default_depth)
    else:
        out["depth_m"] = default_depth
    if "fetch_m" in df.columns:
        out["fetch_m"] = pd.to_numeric(df["fetch_m"], errors="coerce").fillna(default_fetch)
    else:
        out["fetch_m"] = default_fetch

    out = out.dropna(subset=["image_date", "authoritative_lat", "authoritative_lng", "observed_spacing_m"])
    out = out[(out["observed_spacing_m"] > 1.0) & (out["observed_spacing_m"] < 2000.0)]
    return out.sort_values("image_date").reset_index(drop=True)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _write_outputs(result: ValidationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.results.to_csv(output_dir / "results.csv", index=False)
    result.weather_summary.to_csv(output_dir / "weather_summary.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="ascii") as fh:
        json.dump(result.metrics, fh, indent=2)


def validate_observations(
    dataset_path: str,
    spacing_column: str = "manual_spacing_m",
    cache_dir: str = "data/era5_cache",
    output_dir: str | None = None,
    default_depth: float = 9.0,
    default_fetch: float = 15000.0,
    skip_download: bool = False,
) -> ValidationResult:
    observations = load_observations(
        dataset_path=dataset_path,
        spacing_column=spacing_column,
        default_depth=default_depth,
        default_fetch=default_fetch,
    )
    era5_data = fetch_all_observations(observations, Path(cache_dir), skip_download=skip_download)

    result_rows = []
    weather_rows = []
    for idx, row in observations.iterrows():
        obs = era5_data.get(idx, {})
        if "error" in obs:
            result_rows.append(
                {
                    "obs_index": idx,
                    "image_date": row["image_date"].isoformat(),
                    "lat": row["authoritative_lat"],
                    "lon": row["authoritative_lng"],
                    "observed_spacing_m": row["observed_spacing_m"],
                    "predicted_spacing_m": float("nan"),
                    "error_m": float("nan"),
                    "depth_m": row["depth_m"],
                    "fetch_m": row["fetch_m"],
                    "download_error": obs["error"],
                }
            )
            continue

        forcing = extract_model_forcing(obs["spinup"])
        pre_context = summarise_context_window(obs["pre_context"], "pre_context")
        post_context = summarise_context_window(obs["post_context"], "post_context")
        regime = classify_wind_regime(forcing, pre_context, post_context)

        mode = find_critical_wavenumber(
            LCParams(U10=forcing["U10_representative"], depth=float(row["depth_m"]), fetch=float(row["fetch_m"]))
        )
        predicted = float(mode.spacing)

        result_rows.append(
            {
                "obs_index": idx,
                "image_date": row["image_date"].isoformat(),
                "lat": row["authoritative_lat"],
                "lon": row["authoritative_lng"],
                "observed_spacing_m": row["observed_spacing_m"],
                "predicted_spacing_m": predicted,
                "error_m": predicted - row["observed_spacing_m"],
                "depth_m": row["depth_m"],
                "fetch_m": row["fetch_m"],
                "U10_representative": forcing["U10_representative"],
                "U10_final_24h_mean": forcing["U10_final_24h_mean"],
                "U10_10day_mean": forcing["U10_10day_mean"],
                "wind_dir_dominant": forcing["wind_dir_dominant"],
                "wind_steadiness": forcing["wind_steadiness"],
                "wind_regime": regime,
                "download_error": "",
            }
        )
        weather_rows.append(
            {
                "obs_index": idx,
                "image_date": row["image_date"].isoformat(),
                **forcing,
                "pre_wind_mean": pre_context["wind_mean"],
                "post_wind_mean": post_context["wind_mean"],
                "pre_temp_mean": pre_context["temp_mean"],
                "post_temp_mean": post_context["temp_mean"],
                "wind_regime": regime,
            }
        )

    results = pd.DataFrame(result_rows)
    weather_summary = pd.DataFrame(weather_rows)
    valid = results.dropna(subset=["predicted_spacing_m"])
    metrics = {
        "spacing_column": spacing_column,
        "n_observations": int(len(results)),
        "n_valid_predictions": int(len(valid)),
        "rmse_m": float(np.sqrt(np.mean(valid["error_m"] ** 2))) if len(valid) else float("nan"),
        "bias_m": float(valid["error_m"].mean()) if len(valid) else float("nan"),
        "r_squared": _r_squared(
            valid["observed_spacing_m"].to_numpy(),
            valid["predicted_spacing_m"].to_numpy(),
        ) if len(valid) else float("nan"),
        "mean_observed_spacing_m": float(results["observed_spacing_m"].mean()) if len(results) else float("nan"),
        "mean_predicted_spacing_m": float(valid["predicted_spacing_m"].mean()) if len(valid) else float("nan"),
    }

    result = ValidationResult(results=results, weather_summary=weather_summary, metrics=metrics)
    if output_dir is not None:
        out_path = Path(output_dir)
        _write_outputs(result, out_path)
        result.output_dir = out_path
    return result
