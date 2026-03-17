from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"
ERA5_HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "temperature_2m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
]


def build_era5_url(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables or ERA5_HOURLY_VARS),
        "models": "era5",
        "timezone": "GMT",
        "timeformat": "iso8601",
    }
    return f"{OPEN_METEO_BASE}?{urlencode(params)}"


def _json_to_frame(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index().ffill()


def download_era5(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    output_path: Path,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    url = build_era5_url(lat, lon, start_date, end_date, variables=variables)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"ERA5 download failed for ({lat}, {lon}) {start_date} to {end_date}: {exc}"
        ) from exc

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return _json_to_frame(payload)


def compute_time_windows(image_date: date) -> dict[str, dict[str, str]]:
    spinup_start = image_date - timedelta(days=10)
    spinup_end = image_date
    pre_start = spinup_start - timedelta(days=30)
    pre_end = spinup_start - timedelta(days=1)
    post_start = image_date + timedelta(days=1)
    post_end = image_date + timedelta(days=30)
    return {
        "spinup": {"start": spinup_start.isoformat(), "end": spinup_end.isoformat()},
        "pre_context": {"start": pre_start.isoformat(), "end": pre_end.isoformat()},
        "post_context": {"start": post_start.isoformat(), "end": post_end.isoformat()},
    }


def _cache_key(lat: float, lon: float, start_date: str, end_date: str) -> str:
    return f"{lat:.3f}_{lon:.3f}_{start_date}_{end_date}".replace("-", "").replace(".", "p")


def load_cached_era5(cache_file: Path) -> pd.DataFrame:
    with cache_file.open(encoding="utf-8") as fh:
        return _json_to_frame(json.load(fh))


def _cache_has_variables(cache_file: Path, variables: list[str]) -> bool:
    if not cache_file.exists():
        return False
    try:
        df = load_cached_era5(cache_file)
    except Exception:
        return False
    return all(variable in df.columns for variable in variables)


def fetch_all_observations(
    obs_df: pd.DataFrame,
    cache_dir: Path,
    skip_download: bool = False,
    variables: list[str] | None = None,
) -> dict[int, dict]:
    chosen_vars = variables or ERA5_HOURLY_VARS
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict] = {}

    for idx, row in obs_df.iterrows():
        lat = float(row["authoritative_lat"])
        lon = float(row["authoritative_lng"])
        image_date = row["image_date"]
        windows = compute_time_windows(image_date)
        obs_data: dict = {
            "meta": {
                "lat": lat,
                "lon": lon,
                "image_date": image_date.isoformat(),
                "observed_spacing": float(row["observed_spacing_m"]),
            }
        }
        try:
            for window_name, dates in windows.items():
                cache_file = cache_dir / (_cache_key(lat, lon, dates["start"], dates["end"]) + ".json")
                if _cache_has_variables(cache_file, chosen_vars):
                    obs_data[window_name] = load_cached_era5(cache_file)
                else:
                    if skip_download:
                        raise FileNotFoundError(f"Cache miss for {cache_file}")
                    obs_data[window_name] = download_era5(
                        lat,
                        lon,
                        dates["start"],
                        dates["end"],
                        cache_file,
                        variables=chosen_vars,
                    )
                    time.sleep(0.5)
            results[idx] = obs_data
        except Exception as exc:
            obs_data["error"] = str(exc)
            results[idx] = obs_data
    return results
