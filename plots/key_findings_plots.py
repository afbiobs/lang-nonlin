#!/usr/bin/env python3
"""Generate focused figures that summarize the main post-fix findings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from output_locator import latest_timeline_output


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"
OUTPUTS_DIR = ROOT / "outputs"
DPI = 150
DATASETS = {
    "manual": latest_timeline_output("manual"),
    "wiggle": latest_timeline_output("wiggle"),
}
COLORS = {
    "manual": "#1d3557",
    "wiggle": "#d97706",
    "observed": "#111827",
    "linear": "#6b7280",
    "visible": "#2a9d8f",
}


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    diagnostics: pd.DataFrame
    metrics: dict[str, float]


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.9)
    sns.set_palette("deep")


def save_figure(fig: plt.Figure, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=DPI)
    plt.close(fig)


def cohen_d(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x_arr) < 2 or len(y_arr) < 2:
        return float("nan")
    pooled_var = (
        ((len(x_arr) - 1) * np.var(x_arr, ddof=1) + (len(y_arr) - 1) * np.var(y_arr, ddof=1))
        / (len(x_arr) + len(y_arr) - 2)
    )
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return float("nan")
    return float((np.mean(x_arr) - np.mean(y_arr)) / np.sqrt(pooled_var))


def lowess_like(
    x: pd.Series,
    y: pd.Series,
    frac: float = 0.5,
    points: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    mask = x.notna() & y.notna()
    xs = x[mask].to_numpy(dtype=float)
    ys = y[mask].to_numpy(dtype=float)
    if len(xs) < 4:
        order = np.argsort(xs)
        return xs[order], ys[order]

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    span = max(4, int(np.ceil(frac * len(xs))))
    x_eval = np.linspace(xs.min(), xs.max(), min(points, max(60, len(xs))))
    y_eval = np.empty_like(x_eval)

    for idx, x0 in enumerate(x_eval):
        distance = np.abs(xs - x0)
        nearest = np.argpartition(distance, span - 1)[:span]
        x_local = xs[nearest]
        y_local = ys[nearest]
        d_max = distance[nearest].max()
        if d_max <= 0:
            y_eval[idx] = np.mean(y_local)
            continue
        weights = (1.0 - (distance[nearest] / d_max) ** 3) ** 3
        design = np.column_stack([np.ones(len(x_local)), x_local - x0])
        weighted_design = design.T * weights
        beta = np.linalg.pinv(weighted_design @ design) @ (weighted_design @ y_local)
        y_eval[idx] = beta[0]
    return x_eval, y_eval


def resolve_observed_column(df: pd.DataFrame) -> str:
    for column in ("observed_spacing_m", "manual_spacing_m", "wiggle_spacing_m"):
        if column in df.columns:
            return column
    raise KeyError("Missing observed spacing column")


def load_bundle(name: str, base_dir: Path) -> DatasetBundle:
    diagnostics = pd.read_csv(base_dir / "observation_diagnostics.csv")
    metrics = json.loads((base_dir / "metrics.json").read_text())
    observed_column = resolve_observed_column(diagnostics)
    diagnostics = diagnostics.rename(columns={observed_column: "observed_spacing_m"}).copy()
    diagnostics["dataset"] = name
    diagnostics["gap_ratio_nl"] = diagnostics["observed_spacing_m"] / diagnostics["spacing_at_obs_m"]
    diagnostics["gap_ratio_l"] = diagnostics["observed_spacing_m"] / diagnostics["spacing_linear_at_obs_m"]
    diagnostics["nl_over_l_spacing_ratio"] = diagnostics["spacing_at_obs_m"] / diagnostics["spacing_linear_at_obs_m"]
    diagnostics["gap_log10"] = np.log10(diagnostics["gap_ratio_nl"])
    diagnostics["observation_rank"] = diagnostics["observed_spacing_m"].rank(method="first")
    return DatasetBundle(name=name, diagnostics=diagnostics, metrics=metrics)


def plot_scale_gap_ranked(bundles: Iterable[DatasetBundle]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), constrained_layout=True, sharey=True)

    for ax, bundle in zip(axes, bundles, strict=True):
        ordered = bundle.diagnostics.sort_values("observed_spacing_m").reset_index(drop=True)
        x = np.arange(1, len(ordered) + 1, dtype=float)
        ax.plot(
            x,
            ordered["observed_spacing_m"],
            color=COLORS["observed"],
            linewidth=2.6,
            marker="o",
            markersize=3.5,
            label="Observed spacing",
        )
        ax.plot(
            x,
            ordered["spacing_linear_at_obs_m"],
            color=COLORS["linear"],
            linewidth=2.0,
            linestyle="--",
            label="Linear prediction",
        )
        ax.plot(
            x,
            ordered["spacing_at_obs_m"],
            color=COLORS["visible"],
            linewidth=2.2,
            label="Visible prediction",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Observation rank (sorted by observed spacing)")
        ax.set_ylabel("Spacing (m)")
        ax.set_title(f"{bundle.name.title()} dataset")
        median_gap = ordered["gap_ratio_nl"].median()
        median_nl_over_l = ordered["nl_over_l_spacing_ratio"].median()
        ax.text(
            0.03,
            0.97,
            (
                f"N = {len(ordered)}\n"
                f"Observed median = {ordered['observed_spacing_m'].median():.1f} m\n"
                f"Visible median = {ordered['spacing_at_obs_m'].median():.1f} m\n"
                f"Linear median = {ordered['spacing_linear_at_obs_m'].median():.1f} m\n"
                f"Observed / visible median = {median_gap:.2f}x\n"
                f"Visible / linear median = {median_nl_over_l:.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.96},
        )
        ax.legend(loc="lower right", fontsize=9)

    save_figure(fig, "plot17_scale_gap_ranked.png")


def plot_ratio_summary(combined: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    order = ["manual", "wiggle"]

    sns.boxplot(
        data=combined,
        x="dataset",
        y="nl_over_l_spacing_ratio",
        order=order,
        hue="dataset",
        palette=[COLORS["manual"], COLORS["wiggle"]],
        ax=axes[0],
        fliersize=3,
        legend=False,
    )
    sns.stripplot(
        data=combined,
        x="dataset",
        y="nl_over_l_spacing_ratio",
        order=order,
        color="white",
        edgecolor="black",
        linewidth=0.5,
        size=4,
        ax=axes[0],
        jitter=0.14,
    )
    axes[0].axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Visible spacing / linear spacing")
    axes[0].set_title("Visible branch remains below the linear onset scale")

    sns.boxplot(
        data=combined,
        x="dataset",
        y="gap_ratio_nl",
        order=order,
        hue="dataset",
        palette=[COLORS["manual"], COLORS["wiggle"]],
        ax=axes[1],
        fliersize=3,
        legend=False,
    )
    sns.stripplot(
        data=combined,
        x="dataset",
        y="gap_ratio_nl",
        order=order,
        color="white",
        edgecolor="black",
        linewidth=0.5,
        size=4,
        ax=axes[1],
        jitter=0.14,
    )
    axes[1].axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Observed spacing / visible prediction")
    axes[1].set_title("Large unresolved spacing gap remains")

    for ax, column in zip(axes, ["nl_over_l_spacing_ratio", "gap_ratio_nl"], strict=True):
        summary = combined.groupby("dataset")[column].median()
        ax.text(
            0.03,
            0.97,
            "\n".join(f"{name.title()} median = {value:.2f}" for name, value in summary.items()),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.96},
        )

    save_figure(fig, "plot18_ratio_summary.png")


def plot_gap_vs_memory_controls(combined: pd.DataFrame) -> None:
    variables = [
        ("integrated_supercriticality_prev_48h", "Integrated supercriticality\nprevious 48 h"),
        ("coherent_run_hours_at_obs", "Coherent run hours\nat observation"),
        ("development_at_obs", "Development index\nat observation"),
        ("amplitude_at_obs", "Amplitude index\nat observation"),
    ]
    markers = {"manual": "o", "wiggle": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)

    for ax, (column, label) in zip(axes.ravel(), variables, strict=True):
        for dataset in ("manual", "wiggle"):
            subset = combined[combined["dataset"] == dataset].copy()
            ax.scatter(
                subset[column],
                subset["gap_ratio_nl"],
                s=58,
                alpha=0.82,
                marker=markers[dataset],
                color=COLORS[dataset],
                edgecolor="white",
                linewidth=0.4,
                label=dataset.title(),
            )
            x_smooth, y_smooth = lowess_like(subset[column], subset["gap_ratio_nl"])
            if len(x_smooth) > 1:
                ax.plot(x_smooth, y_smooth, color=COLORS[dataset], linewidth=2.0)
        correlations = []
        for dataset in ("manual", "wiggle"):
            subset = combined[combined["dataset"] == dataset]
            r = subset[[column, "gap_ratio_nl"]].corr(numeric_only=True).iloc[0, 1]
            correlations.append(f"{dataset.title()} r = {r:.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Observed spacing / visible prediction")
        ax.set_yscale("log")
        ax.text(
            0.03,
            0.97,
            "\n".join(correlations),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.96},
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles[:2], labels[:2], loc="lower right", fontsize=9)
    save_figure(fig, "plot19_gap_vs_memory_controls.png")


def plot_dataset_contrast_effect_sizes(combined: pd.DataFrame) -> None:
    variables = [
        ("observed_spacing_m", "Observed spacing"),
        ("spacing_at_obs_m", "Visible prediction"),
        ("spacing_linear_at_obs_m", "Linear prediction"),
        ("gap_ratio_nl", "Observed / visible"),
        ("development_at_obs", "Development index"),
        ("amplitude_at_obs", "Amplitude index"),
        ("integrated_supercriticality_prev_48h", "Integrated supercriticality"),
        ("coherent_run_hours_at_obs", "Coherent run hours"),
        ("wind_mean_prev_48h", "Mean wind prev 48 h"),
        ("kappa_at_obs", "Kappa at observation"),
        ("wavenumber_ratio_at_obs", "Wavenumber ratio at observation"),
    ]
    manual = combined[combined["dataset"] == "manual"]
    wiggle = combined[combined["dataset"] == "wiggle"]
    effect_rows = []
    for column, label in variables:
        effect_rows.append(
            {
                "label": label,
                "effect_size": cohen_d(wiggle[column], manual[column]),
                "category": (
                    "Observed"
                    if column == "observed_spacing_m"
                    else "Prediction"
                    if "spacing" in column or column in {"kappa_at_obs", "wavenumber_ratio_at_obs", "gap_ratio_nl"}
                    else "State / forcing"
                ),
            }
        )
    effects = pd.DataFrame(effect_rows).sort_values("effect_size")

    palette = {"Observed": "#111827", "Prediction": "#2a9d8f", "State / forcing": "#8c5e34"}
    fig, ax = plt.subplots(figsize=(10.5, 7.5), constrained_layout=True)
    bar_colors = effects["category"].map(palette)
    ax.barh(effects["label"], effects["effect_size"], color=bar_colors)
    ax.axvline(0.0, color="black", linewidth=1.2)
    for _, row in effects.iterrows():
        x = row["effect_size"]
        ha = "left" if x >= 0 else "right"
        offset = 0.03 if x >= 0 else -0.03
        ax.text(x + offset, row["label"], f"{x:.2f}", va="center", ha=ha, fontsize=10)
    ax.set_xlabel("Cohen's d (wiggle minus manual)")
    ax.set_ylabel("")
    ax.set_title("Dataset contrast is dominated by observed scale, not the visible-spacing state")
    legend_handles = [
        plt.Line2D([0], [0], color=color, linewidth=8, label=label) for label, color in palette.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    save_figure(fig, "plot20_dataset_contrast_effect_sizes.png")


def build_summary(bundles: Iterable[DatasetBundle], combined: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    development_summary_path = PLOTS_DIR / "development_before_after_summary.json"
    development_summary = {}
    if development_summary_path.exists():
        development_summary = json.loads(development_summary_path.read_text())

    for bundle in bundles:
        diagnostics = bundle.diagnostics
        gap_correlations = (
            diagnostics[
                [
                    "gap_ratio_nl",
                    "integrated_supercriticality_prev_48h",
                    "coherent_run_hours_at_obs",
                    "development_at_obs",
                    "amplitude_at_obs",
                    "wind_mean_prev_48h",
                ]
            ]
            .corr(numeric_only=True)["gap_ratio_nl"]
            .drop("gap_ratio_nl")
            .dropna()
        )
        strongest = gap_correlations.reindex(gap_correlations.abs().sort_values(ascending=False).index).iloc[0]
        strongest_name = gap_correlations.abs().sort_values(ascending=False).index[0]
        dev_delta = float("nan")
        if bundle.name in development_summary:
            tests = development_summary[bundle.name].get("tests", {})
            dev_delta = float(tests.get("obs_after_q90_mean", np.nan) - tests.get("obs_before_q90_mean", np.nan))
        summary[bundle.name] = {
            "n_observations": int(len(diagnostics)),
            "median_observed_spacing_m": float(diagnostics["observed_spacing_m"].median()),
            "median_visible_spacing_m": float(diagnostics["spacing_at_obs_m"].median()),
            "median_linear_spacing_m": float(diagnostics["spacing_linear_at_obs_m"].median()),
            "median_observed_over_visible": float(diagnostics["gap_ratio_nl"].median()),
            "median_observed_over_linear": float(diagnostics["gap_ratio_l"].median()),
            "median_visible_over_linear": float(diagnostics["nl_over_l_spacing_ratio"].median()),
            "rmse_visible_m": float(bundle.metrics["rmse_NL_m"]),
            "rmse_linear_m": float(bundle.metrics["rmse_L_m"]),
            "r_squared_visible": float(bundle.metrics["r_squared_NL"]),
            "r_squared_linear": float(bundle.metrics["r_squared_L"]),
            "development_q90_after_minus_before": dev_delta,
            "strongest_gap_correlation": {
                "variable": strongest_name,
                "pearson_r": float(strongest),
            },
        }

    manual = combined[combined["dataset"] == "manual"]
    wiggle = combined[combined["dataset"] == "wiggle"]
    summary["dataset_contrast_effect_sizes"] = {
        "observed_spacing_m": cohen_d(wiggle["observed_spacing_m"], manual["observed_spacing_m"]),
        "spacing_at_obs_m": cohen_d(wiggle["spacing_at_obs_m"], manual["spacing_at_obs_m"]),
        "spacing_linear_at_obs_m": cohen_d(wiggle["spacing_linear_at_obs_m"], manual["spacing_linear_at_obs_m"]),
        "gap_ratio_nl": cohen_d(wiggle["gap_ratio_nl"], manual["gap_ratio_nl"]),
        "kappa_at_obs": cohen_d(wiggle["kappa_at_obs"], manual["kappa_at_obs"]),
        "wavenumber_ratio_at_obs": cohen_d(wiggle["wavenumber_ratio_at_obs"], manual["wavenumber_ratio_at_obs"]),
    }
    return summary


def main() -> None:
    setup_style()
    bundles = [load_bundle(name, path) for name, path in DATASETS.items()]
    combined = pd.concat([bundle.diagnostics for bundle in bundles], ignore_index=True)

    plot_scale_gap_ranked(bundles)
    plot_ratio_summary(combined)
    plot_gap_vs_memory_controls(combined)
    plot_dataset_contrast_effect_sizes(combined)

    summary = build_summary(bundles, combined)
    (PLOTS_DIR / "key_findings_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
