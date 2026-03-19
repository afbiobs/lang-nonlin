#!/usr/bin/env python3
"""Plot temporal-responsiveness diagnostics for the latest timeline outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from output_locator import latest_timeline_output


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"
DPI = 150
DATASETS = {
    "manual": latest_timeline_output("manual"),
    "wiggle": latest_timeline_output("wiggle"),
}
MISMATCH_ORDER = [
    "envelope_match",
    "visibility-limited",
    "under-responsive-hydro",
    "saturated-lifecycle",
    "response-overreach",
    "observation-outside-envelope",
    "physics_failure",
]
MISMATCH_COLORS = {
    "envelope_match": "#2a9d8f",
    "visibility-limited": "#d97706",
    "under-responsive-hydro": "#577590",
    "saturated-lifecycle": "#6d597a",
    "response-overreach": "#e76f51",
    "observation-outside-envelope": "#b5179e",
    "physics_failure": "#6b7280",
}


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    base_dir: Path
    diagnostics: pd.DataFrame
    metrics: dict[str, float]


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.9)


def save_figure(fig: plt.Figure, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=DPI)
    plt.close(fig)


def load_bundle(name: str, base_dir: Path) -> DatasetBundle:
    diagnostics = pd.read_csv(base_dir / "observation_diagnostics.csv")
    metrics = json.loads((base_dir / "metrics.json").read_text())
    return DatasetBundle(name=name, base_dir=base_dir, diagnostics=diagnostics, metrics=metrics)


def plot_responsiveness_vs_wind(bundles: list[DatasetBundle]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), constrained_layout=True, sharey="row")
    for row_idx, bundle in enumerate(bundles):
        df = bundle.diagnostics.copy()
        sns.scatterplot(
            data=df,
            x="wind_mean_prev_48h",
            y="spacing_at_obs_m",
            hue="visibility_at_obs",
            palette="viridis",
            ax=axes[row_idx, 0],
            s=70,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[row_idx, 0].set_title(f"{bundle.name.title()}: core spacing vs 48 h wind")
        axes[row_idx, 0].set_xlabel("48 h mean wind (m/s)")
        axes[row_idx, 0].set_ylabel("Core spacing at observation (m)")

        sns.scatterplot(
            data=df,
            x="integrated_supercriticality_prev_48h",
            y="spacing_at_obs_m",
            hue="mismatch_class",
            hue_order=MISMATCH_ORDER,
            palette=MISMATCH_COLORS,
            ax=axes[row_idx, 1],
            s=70,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[row_idx, 1].set_title(f"{bundle.name.title()}: core spacing vs integrated supercriticality")
        axes[row_idx, 1].set_xlabel("Integrated supercriticality (48 h)")
        axes[row_idx, 1].set_ylabel("Core spacing at observation (m)")

    for ax in axes.ravel():
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(legend.get_title().get_text())
            for text in legend.get_texts():
                text.set_fontsize(8)
    save_figure(fig, "plot25_responsiveness_vs_wind.png")


def plot_mismatch_breakdown(bundles: list[DatasetBundle]) -> None:
    rows = []
    for bundle in bundles:
        counts = bundle.diagnostics["mismatch_class"].value_counts()
        total = max(int(counts.sum()), 1)
        for mismatch in MISMATCH_ORDER:
            rows.append(
                {
                    "dataset": bundle.name,
                    "mismatch_class": mismatch,
                    "fraction": counts.get(mismatch, 0) / total,
                }
            )
    mismatch_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)
    sns.barplot(
        data=mismatch_df,
        x="mismatch_class",
        y="fraction",
        hue="dataset",
        palette={"manual": "#1d3557", "wiggle": "#d97706"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of observations")
    ax.set_title("Mismatch classes show whether the miss is visibility-limited or outside the model envelope")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(loc="best", fontsize=9)
    save_figure(fig, "plot26_mismatch_class_breakdown.png")


def plot_visibility_gap(bundles: list[DatasetBundle]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True, sharey=True)
    for ax, bundle in zip(axes, bundles, strict=True):
        df = bundle.diagnostics.copy()
        df["observed_to_core_ratio"] = df["observed_spacing_m"] / df["spacing_at_obs_m"]
        sns.scatterplot(
            data=df,
            x="visibility_at_obs",
            y="observed_to_core_ratio",
            hue="mismatch_class",
            hue_order=MISMATCH_ORDER,
            palette=MISMATCH_COLORS,
            ax=ax,
            s=70,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
        ax.set_title(f"{bundle.name.title()} dataset")
        ax.set_xlabel("Visibility index at observation")
        ax.set_ylabel("Observed spacing / predicted core spacing")
        ax.legend(loc="best", fontsize=8)
    save_figure(fig, "plot27_visibility_vs_scale_gap.png")


def build_summary(bundles: list[DatasetBundle]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for bundle in bundles:
        df = bundle.diagnostics
        summary[bundle.name] = {
            "source_dir": str(bundle.base_dir),
            "core_spacing_q10_m": float(df["spacing_at_obs_m"].quantile(0.10)),
            "core_spacing_q90_m": float(df["spacing_at_obs_m"].quantile(0.90)),
            "core_spacing_range_mean_m": float(df["core_spacing_range_prev_48h_m"].mean()),
            "visibility_q10": float(df["visibility_at_obs"].quantile(0.10)),
            "visibility_q90": float(df["visibility_at_obs"].quantile(0.90)),
            "mismatch_counts": {str(key): int(value) for key, value in df["mismatch_class"].value_counts().to_dict().items()},
        }
    return summary


def main() -> None:
    setup_style()
    bundles = [load_bundle(name, path) for name, path in DATASETS.items()]
    plot_responsiveness_vs_wind(bundles)
    plot_mismatch_breakdown(bundles)
    plot_visibility_gap(bundles)
    summary = build_summary(bundles)
    (PLOTS_DIR / "responsiveness_diagnostics_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
