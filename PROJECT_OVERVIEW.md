# SPACING_LANGMUIR

Minimal standalone project for two tasks:

1. Predict Langmuir spacing from wind speed, depth, and fetch.
2. Download ERA5/Open-Meteo weather data and validate predictions against a small observation CSV.

## Files

- `spacing_langmuir/config.py`: parameter container and derived forcing state.
- `spacing_langmuir/wind_forcing.py`: wind stress, fetch-limited wave state, and Stokes drift forcing.
- `spacing_langmuir/stability.py`: reduced Langmuir instability solver that returns the critical spacing.
- `spacing_langmuir/era5.py`: Open-Meteo ERA5 download and local JSON cache helpers.
- `spacing_langmuir/weather.py`: converts hourly weather into representative forcing summaries.
- `spacing_langmuir/validation.py`: dataset loading, weather-backed prediction loop, and validation outputs.
- `spacing_langmuir/run.py`: simple CLI entry points.
- `data/observations_minimal.csv`: cleaned validation dataset with only the columns used here.

## Dataset columns

- `observation_id`
- `source_row`
- `image_date`
- `authoritative_lat`
- `authoritative_lng`
- `manual_spacing_m`
- `wiggle_spacing_m`

`depth_m` and `fetch_m` are optional. If absent, validation defaults to `9.0 m` depth and `15000 m` fetch.

## Typical commands

```bash
python -m spacing_langmuir.run predict --u10 6 --depth 8.9 --fetch 15000
python -m spacing_langmuir.run download-weather --dataset data/observations_minimal.csv --cache-dir data/era5_cache
python -m spacing_langmuir.run validate --dataset data/observations_minimal.csv --spacing-column manual_spacing_m --cache-dir data/era5_cache --output-dir outputs/manual_validation
python -m spacing_langmuir.run validate --dataset data/observations_minimal.csv --spacing-column wiggle_spacing_m --cache-dir data/era5_cache --output-dir outputs/wiggle_validation
```

## Notes

- This project intentionally excludes plotting, ensemble models, particle tracking, and exploratory analysis.
- It is meant to be a clean base for replacing the current linear stability model with newer nonlinear approaches.
