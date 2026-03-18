"""Weather summarisation helpers for validation workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def angle_difference_deg(angle_a: float, angle_b: float) -> float:
    """Smallest signed angular difference angle_a - angle_b in degrees."""
    return float(((angle_a - angle_b + 180.0) % 360.0) - 180.0)


def directional_steadiness(speed: pd.Series, direction_deg: pd.Series) -> float:
    """Return vector-to-scalar mean wind ratio on [0, 1]."""
    ws = speed.astype(float).to_numpy()
    wd = direction_deg.astype(float).to_numpy()
    if len(ws) == 0:
        return float("nan")
    mean_speed = float(np.mean(ws))
    if mean_speed <= 0.0:
        return 0.0
    u_comp = -ws * np.sin(np.deg2rad(wd))
    v_comp = -ws * np.cos(np.deg2rad(wd))
    return float(np.hypot(np.mean(u_comp), np.mean(v_comp)) / mean_speed)


def direction_change_series(direction_deg: pd.Series) -> pd.Series:
    """Absolute hour-to-hour turning angle in degrees."""
    direction = direction_deg.astype(float).reset_index(drop=True)
    out = [0.0]
    for idx in range(1, len(direction)):
        out.append(abs(angle_difference_deg(direction.iloc[idx], direction.iloc[idx - 1])))
    return pd.Series(out, index=direction_deg.index, dtype=float)


def forcing_coherence_series(direction_deg: pd.Series) -> pd.Series:
    """Directional coherence factor on [0, 1] for successive wind steps."""
    turns = direction_change_series(direction_deg).astype(float)
    coherence = 0.5 * (1.0 + np.cos(np.deg2rad(turns)))
    if len(coherence) > 0:
        coherence.iloc[0] = 1.0
    return coherence


def extract_model_forcing(spinup_df: pd.DataFrame) -> dict:
    """Summarise a 10-day spinup window into a representative 10 m wind forcing."""
    ws = spinup_df["wind_speed_10m"].astype(float)
    wd = spinup_df["wind_direction_10m"].astype(float)
    hours_before = ((spinup_df.index[-1] - spinup_df.index).total_seconds() / 3600.0).to_numpy()
    weights = np.exp(-hours_before / 12.0)
    weights = weights / weights.sum()

    u10_rep = float(np.average(ws.to_numpy(), weights=weights))
    last_24h = ws.iloc[-24:] if len(ws) >= 24 else ws
    last_6h = ws.iloc[-6:] if len(ws) >= 6 else ws

    u_comp = -ws * np.sin(np.deg2rad(wd))
    v_comp = -ws * np.cos(np.deg2rad(wd))
    mean_u = float(np.average(u_comp.to_numpy(), weights=weights))
    mean_v = float(np.average(v_comp.to_numpy(), weights=weights))
    wind_dir = float((np.rad2deg(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)
    steadiness = float(np.hypot(mean_u, mean_v) / u10_rep) if u10_rep > 0 else 0.0

    return {
        "U10_representative": u10_rep,
        "U10_final_24h_mean": float(last_24h.mean()),
        "U10_final_6h_mean": float(last_6h.mean()),
        "U10_10day_mean": float(ws.mean()),
        "U10_10day_std": float(ws.std()),
        "wind_dir_dominant": wind_dir,
        "wind_steadiness": steadiness,
    }


def summarise_context_window(df: pd.DataFrame, label: str) -> dict:
    ws = df["wind_speed_10m"].astype(float)
    return {
        "label": label,
        "wind_mean": float(ws.mean()),
        "wind_std": float(ws.std()),
        "temp_mean": float(df["temperature_2m"].astype(float).mean()),
        "radiation_mean": float(df["shortwave_radiation"].astype(float).mean()),
        "precip_total": float(df["precipitation"].astype(float).sum()),
    }


def classify_wind_regime(spinup_forcing: dict, pre_context: dict, post_context: dict) -> str:
    spinup_mean = spinup_forcing["U10_10day_mean"]
    context_mean = 0.5 * (pre_context["wind_mean"] + post_context["wind_mean"])
    context_std = 0.5 * (pre_context["wind_std"] + post_context["wind_std"])
    if spinup_mean < 2.0:
        return "calm_spell"
    if spinup_mean > context_mean + context_std:
        return "above_normal"
    if spinup_mean < context_mean - context_std:
        return "below_normal"
    return "typical"
