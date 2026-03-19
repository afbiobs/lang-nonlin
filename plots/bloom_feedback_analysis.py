#!/usr/bin/env python3
"""Analyse temporal bloom-feedback dynamics in validation timelines."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from output_locator import latest_timeline_output


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"
MANUAL_DIR = latest_timeline_output("manual")
DEPTH_FALLBACK_M = 9.0
DPI = 150
LIGHT_GAIN = 0.5  # bloom_feedback uses (light_enhancement - 1.0) with default 1.5


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.9)


def save_figure(fig: plt.Figure, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=DPI)
    plt.close(fig)


def load_timeline_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_csv(MANUAL_DIR / "observation_diagnostics.csv")
    diagnostics = diagnostics[["observation_id", "depth_m", "observed_spacing_m"]].copy()

    frames = []
    for path in sorted((MANUAL_DIR / "timelines").glob("*_timeline.csv")):
        df = pd.read_csv(path)
        df["observation_id"] = path.stem.replace("_timeline", "")
        frames.append(df)
    timelines = pd.concat(frames, ignore_index=True)
    timelines = timelines.merge(diagnostics, on="observation_id", how="left")
    timelines["depth_m"] = timelines["depth_m"].fillna(DEPTH_FALLBACK_M)
    timelines["surface_residence_time_s"] = np.where(
        timelines["w_down_max"] > 0.0,
        timelines["depth_m"] / timelines["w_down_max"],
        np.nan,
    )
    timelines["residence_time_hours"] = timelines["surface_residence_time_s"] / 3600.0
    timelines["residence_time_factor"] = np.where(
        timelines["surface_residence_time_s"] > 0.0,
        1.0 - np.exp(-timelines["surface_residence_time_s"] / 3600.0),
        0.0,
    )
    timelines["feedback_from_formula"] = (
        timelines["accumulation_factor"] * timelines["residence_time_factor"] * LIGHT_GAIN
    )
    return timelines, diagnostics


def theory_curve(accumulation_factor: np.ndarray, residence_time_hours: float) -> np.ndarray:
    rt_factor = 1.0 - np.exp(-(residence_time_hours * 3600.0) / 3600.0)
    return accumulation_factor * rt_factor * LIGHT_GAIN


def choose_examples(timelines: pd.DataFrame) -> list[str]:
    per_obs = (
        timelines.groupby("observation_id", as_index=False)
        .agg(
            mean_bloom_feedback=("bloom_feedback", "mean"),
            max_bloom_feedback=("bloom_feedback", "max"),
            observed_spacing_m=("observed_spacing_m", "first"),
        )
        .sort_values("mean_bloom_feedback")
        .reset_index(drop=True)
    )
    indices = [0, len(per_obs) // 2, len(per_obs) - 1]
    return [str(per_obs.iloc[idx]["observation_id"]) for idx in indices]


def annotate_corr(ax: plt.Axes, x: pd.Series, y: pd.Series, label: str) -> None:
    corr = float(x.corr(y))
    ax.text(
        0.04,
        0.96,
        f"{label} r = {corr:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
    )


def plot_relationships(timelines: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    scatter = axes[0, 0].scatter(
        timelines["development_index"],
        timelines["bloom_feedback"],
        c=timelines["accumulation_factor"],
        cmap="viridis",
        s=22,
        alpha=0.55,
        edgecolor="none",
    )
    axes[0, 0].set_xlabel("Development index")
    axes[0, 0].set_ylabel("Bloom feedback potential")
    axes[0, 0].set_title("Bloom feedback vs cell development")
    annotate_corr(axes[0, 0], timelines["development_index"], timelines["bloom_feedback"], "Development")
    cbar = fig.colorbar(scatter, ax=axes[0, 0], shrink=0.88)
    cbar.set_label("Accumulation factor")

    accumulation_grid = np.linspace(0.0, max(0.8, float(timelines["accumulation_factor"].max()) + 0.02), 300)
    for residence_hours, color in [(0.25, "#9c6644"), (1.0, "#386641"), (6.0, "#1d3557"), (24.0, "#6a4c93")]:
        axes[0, 1].plot(
            accumulation_grid,
            theory_curve(accumulation_grid, residence_hours),
            linewidth=2.0,
            color=color,
            label=f"Residence time = {residence_hours:g} h",
        )
    axes[0, 1].scatter(
        timelines["accumulation_factor"],
        timelines["bloom_feedback"],
        c=timelines["residence_time_hours"].clip(upper=24.0),
        cmap="magma",
        s=18,
        alpha=0.45,
        edgecolor="none",
    )
    axes[0, 1].set_xlabel("Accumulation factor")
    axes[0, 1].set_ylabel("Bloom feedback potential")
    axes[0, 1].set_title("Theoretical mapping in bloom_feedback_potential()")
    annotate_corr(axes[0, 1], timelines["accumulation_factor"], timelines["bloom_feedback"], "Accumulation")
    axes[0, 1].legend(loc="upper left", fontsize=9)

    sns.histplot(
        timelines["residence_time_factor"],
        bins=30,
        ax=axes[1, 0],
        color="#457b9d",
        edgecolor="white",
    )
    axes[1, 0].axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[1, 0].set_xlabel("Residence-time factor")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Residence-time term is near saturation")

    scatter_2 = axes[1, 1].scatter(
        timelines["w_down_max"] * 1e4,
        timelines["bloom_feedback"],
        c=timelines["development_index"],
        cmap="cividis",
        s=22,
        alpha=0.55,
        edgecolor="none",
    )
    axes[1, 1].set_xlabel(r"Maximum downwelling $w_{down,max}$ ($10^{-4}$ m/s)")
    axes[1, 1].set_ylabel("Bloom feedback potential")
    axes[1, 1].set_title("Weak direct dependence on downwelling")
    annotate_corr(axes[1, 1], timelines["w_down_max"], timelines["bloom_feedback"], r"$w_{down,max}$")
    cbar2 = fig.colorbar(scatter_2, ax=axes[1, 1], shrink=0.88)
    cbar2.set_label("Development index")

    fig.suptitle("Bloom feedback is dominated by accumulation, not directly by development", fontsize=16)
    save_figure(fig, "plot08_bloom_feedback_relationships.png")


def plot_temporal_examples(timelines: pd.DataFrame) -> None:
    example_ids = choose_examples(timelines)
    fig, axes = plt.subplots(len(example_ids), 1, figsize=(14, 13), constrained_layout=True, sharex=True)
    if len(example_ids) == 1:
        axes = [axes]

    for ax, obs_id in zip(axes, example_ids, strict=True):
        df = timelines[timelines["observation_id"] == obs_id].sort_values("hours_from_observation")
        twin = ax.twinx()

        ax.plot(
            df["hours_from_observation"],
            df["bloom_feedback"],
            color="#d97706",
            linewidth=2.4,
            label="Bloom feedback",
        )
        ax.plot(
            df["hours_from_observation"],
            df["accumulation_factor"] * LIGHT_GAIN,
            color="#9c6644",
            linewidth=1.8,
            linestyle="--",
            label="0.5 x accumulation factor",
        )
        twin.plot(
            df["hours_from_observation"],
            df["development_index"],
            color="#1d3557",
            linewidth=2.0,
            label="Development index",
        )
        twin.plot(
            df["hours_from_observation"],
            df["amplitude_index"],
            color="#2a9d8f",
            linewidth=1.5,
            alpha=0.85,
            label="Amplitude index",
        )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_ylabel("Bloom feedback")
        twin.set_ylabel("Development / amplitude")
        ax.set_ylim(0.0, max(0.42, float(df["bloom_feedback"].max()) * 1.08))
        twin.set_ylim(0.0, 1.02)
        ax.set_title(
            f"{obs_id}: observed spacing = {float(df['observed_spacing_m'].iloc[0]):.1f} m, "
            f"mean feedback = {float(df['bloom_feedback'].mean()):.3f}"
        )

        handles_left, labels_left = ax.get_legend_handles_labels()
        handles_right, labels_right = twin.get_legend_handles_labels()
        ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", fontsize=9, ncol=2)

    axes[-1].set_xlabel("Hours from observation")
    save_figure(fig, "plot09_bloom_feedback_temporal_examples.png")


def print_summary(timelines: pd.DataFrame) -> None:
    summary = {
        "corr_bloom_vs_development": float(timelines["bloom_feedback"].corr(timelines["development_index"])),
        "corr_bloom_vs_amplitude": float(timelines["bloom_feedback"].corr(timelines["amplitude_index"])),
        "corr_bloom_vs_accumulation": float(timelines["bloom_feedback"].corr(timelines["accumulation_factor"])),
        "corr_bloom_vs_w_down_max": float(timelines["bloom_feedback"].corr(timelines["w_down_max"])),
        "mean_residence_factor": float(timelines["residence_time_factor"].mean()),
        "median_residence_factor": float(timelines["residence_time_factor"].median()),
        "max_bloom_feedback": float(timelines["bloom_feedback"].max()),
        "max_formula_feedback": float(timelines["feedback_from_formula"].max()),
    }
    print("Bloom feedback summary:")
    print(json.dumps(summary, indent=2))


def main() -> None:
    setup_style()
    timelines, _ = load_timeline_frame()
    plot_relationships(timelines)
    plot_temporal_examples(timelines)
    print_summary(timelines)


if __name__ == "__main__":
    main()
