from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import LCParams
from .era5 import fetch_all_observations
from .stability import find_critical_wavenumber
from .validation import load_observations, validate_observations


def predict_spacing(U10: float, depth: float = 9.0, fetch: float = 15000.0) -> dict:
    params = LCParams(U10=U10, depth=depth, fetch=fetch)
    mode = find_critical_wavenumber(params)
    return {
        "U10": U10,
        "depth": depth,
        "fetch": fetch,
        "spacing_m": float(mode.spacing),
        "growth_rate_s": float(mode.sigma_max),
        "langmuir_number": float(params.langmuir_number),
        "u_star": float(params.u_star),
        "u_stokes_surface": float(params.u_stokes_surface),
    }


def download_weather_for_dataset(
    dataset_path: str,
    spacing_column: str = "manual_spacing_m",
    cache_dir: str = "data/era5_cache",
    default_depth: float = 9.0,
    default_fetch: float = 15000.0,
    skip_download: bool = False,
) -> dict:
    observations = load_observations(
        dataset_path=dataset_path,
        spacing_column=spacing_column,
        default_depth=default_depth,
        default_fetch=default_fetch,
    )
    era5_data = fetch_all_observations(
        observations,
        Path(cache_dir),
        skip_download=skip_download,
    )
    failures = sum(1 for item in era5_data.values() if "error" in item)
    return {
        "dataset_path": dataset_path,
        "cache_dir": cache_dir,
        "spacing_column": spacing_column,
        "n_observations": int(len(observations)),
        "n_failures": int(failures),
        "n_cached_or_downloaded": int(len(observations) - failures),
    }


def validate_dataset(
    dataset_path: str,
    spacing_column: str = "manual_spacing_m",
    cache_dir: str = "data/era5_cache",
    output_dir: str = "outputs/validation",
    default_depth: float = 9.0,
    default_fetch: float = 15000.0,
    skip_download: bool = False,
) -> dict:
    result = validate_observations(
        dataset_path=dataset_path,
        spacing_column=spacing_column,
        cache_dir=cache_dir,
        output_dir=output_dir,
        default_depth=default_depth,
        default_fetch=default_fetch,
        skip_download=skip_download,
    )
    return result.metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Langmuir spacing prediction workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict Langmuir spacing for a single forcing state.")
    predict_parser.add_argument("--u10", type=float, required=True)
    predict_parser.add_argument("--depth", type=float, default=9.0)
    predict_parser.add_argument("--fetch", type=float, default=15000.0)

    download_parser = subparsers.add_parser("download-weather", help="Cache ERA5/Open-Meteo weather for the dataset.")
    download_parser.add_argument("--dataset", type=str, default="data/observations_minimal.csv")
    download_parser.add_argument("--spacing-column", type=str, default="manual_spacing_m")
    download_parser.add_argument("--cache-dir", type=str, default="data/era5_cache")
    download_parser.add_argument("--default-depth", type=float, default=9.0)
    download_parser.add_argument("--default-fetch", type=float, default=15000.0)
    download_parser.add_argument("--skip-download", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate the model against the observation dataset.")
    validate_parser.add_argument("--dataset", type=str, default="data/observations_minimal.csv")
    validate_parser.add_argument("--spacing-column", type=str, default="manual_spacing_m")
    validate_parser.add_argument("--cache-dir", type=str, default="data/era5_cache")
    validate_parser.add_argument("--output-dir", type=str, default="outputs/validation")
    validate_parser.add_argument("--default-depth", type=float, default=9.0)
    validate_parser.add_argument("--default-fetch", type=float, default=15000.0)
    validate_parser.add_argument("--skip-download", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "predict":
        payload = predict_spacing(args.u10, depth=args.depth, fetch=args.fetch)
    elif args.command == "download-weather":
        payload = download_weather_for_dataset(
            dataset_path=args.dataset,
            spacing_column=args.spacing_column,
            cache_dir=args.cache_dir,
            default_depth=args.default_depth,
            default_fetch=args.default_fetch,
            skip_download=args.skip_download,
        )
    else:
        payload = validate_dataset(
            dataset_path=args.dataset,
            spacing_column=args.spacing_column,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            default_depth=args.default_depth,
            default_fetch=args.default_fetch,
            skip_download=args.skip_download,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
