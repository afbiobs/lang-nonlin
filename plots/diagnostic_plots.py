#!/usr/bin/env python3
"""Generate diagnostic plots for timeline-based Langmuir validation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import patches

try:
    import cmasher as cmr
except ImportError:  # pragma: no cover - optional dependency
    cmr = None


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"
MANUAL_DIR = ROOT / "outputs" / "2026-03-18_manual_validation_timeline_v1"
WIGGLE_DIR = ROOT / "outputs" / "2026-03-18_wiggle_validation_timeline_v1"
DPI = 150
DEFAULT_DEPTH_M = 9.0
REGIME_COLORS = {
    "supercritical": "#9fd8a3",
    "near_onset": "#f2df74",
    "subcritical": "#e59b8f",
}


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    results: pd.DataFrame
    diagnostics: pd.DataFrame
    timeline_dir: Path


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.9)
    sns.set_palette("deep")


def get_cmap():
    if cmr is not None:
        return cmr.ember
    return plt.get_cmap("viridis")


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def load_bundle(name: str, base_dir: Path) -> DatasetBundle:
    results = pd.read_csv(base_dir / "results.csv")
    diagnostics = pd.read_csv(base_dir / "observation_diagnostics.csv")
    observed_results_col = resolve_column(results, ["observed_spacing_m", "manual_spacing_m", "wiggle_spacing_m"])
    observed_diag_col = resolve_column(diagnostics, ["observed_spacing_m", "manual_spacing_m", "wiggle_spacing_m"])
    results = results.rename(columns={observed_results_col: "observed_spacing_m"}).copy()
    diagnostics = diagnostics.rename(columns={observed_diag_col: "observed_spacing_m"}).copy()
    results["dataset"] = name
    diagnostics["dataset"] = name
    return DatasetBundle(
        name=name,
        results=results,
        diagnostics=diagnostics,
        timeline_dir=base_dir / "timelines",
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="constrained_layout not applied because axes sizes collapsed to zero.*",
            category=UserWarning,
        )
        fig.savefig(PLOTS_DIR / filename, dpi=DPI)
    plt.close(fig)


def compute_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    mask = observed.notna() & predicted.notna() & (observed > 0)
    obs = observed[mask].to_numpy(dtype=float)
    pred = predicted[mask].to_numpy(dtype=float)
    if len(obs) == 0:
        return {"rmse": np.nan, "r2": np.nan, "bias": np.nan, "capture_rate": np.nan}
    residual = pred - obs
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else np.nan
    ratio = pred / obs
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(r2),
        "bias": float(np.mean(residual)),
        "capture_rate": float(((ratio >= 0.5) & (ratio <= 2.0)).mean()),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return "\n".join(
        [
            f"RMSE = {metrics['rmse']:.1f} m",
            f"R^2 = {metrics['r2']:.2f}",
            f"Bias = {metrics['bias']:+.1f} m",
            f"Within 2x = {metrics['capture_rate']:.0%}",
        ]
    )


def lowess_like(x: pd.Series, y: pd.Series, frac: float = 0.45, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    mask = x.notna() & y.notna()
    xs = x[mask].to_numpy(dtype=float)
    ys = y[mask].to_numpy(dtype=float)
    if len(xs) < 3:
        order = np.argsort(xs)
        return xs[order], ys[order]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    span = max(3, int(np.ceil(frac * len(xs))))
    x_eval = np.linspace(xs.min(), xs.max(), min(points, max(50, len(xs))))
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


def add_reference_lines(ax: plt.Axes) -> None:
    x_line = np.linspace(0.0, 400.0, 300)
    ax.plot(x_line, x_line, color="black", linestyle="--", linewidth=1.2, label="1:1")
    ax.plot(x_line, 2.0 * x_line, color="0.5", linestyle=":", linewidth=1.0, label="2:1")
    ax.plot(x_line, 0.5 * x_line, color="0.5", linestyle=":", linewidth=1.0, label="1:2")


def theoretical_spacing(selected_l: pd.Series | np.ndarray, depth_m: float) -> np.ndarray:
    selected = np.asarray(selected_l, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        spacing = 2.0 * np.pi * depth_m / selected
    spacing[~np.isfinite(spacing)] = np.nan
    return spacing


def plot_predicted_vs_observed(manual: DatasetBundle, wiggle: DatasetBundle) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True, sharex=True, sharey=True)
    combined_wind = pd.concat(
        [
            manual.diagnostics["wind_mean_prev_48h"],
            wiggle.diagnostics["wind_mean_prev_48h"],
        ],
        ignore_index=True,
    )
    norm = mcolors.Normalize(float(combined_wind.min()), float(combined_wind.max()))
    cmap = get_cmap()

    for ax, bundle, title in zip(
        axes,
        [manual, wiggle],
        ["Manual observations", "Wiggle observations"],
        strict=True,
    ):
        x = bundle.diagnostics["observed_spacing_m"]
        y = bundle.diagnostics["spacing_at_obs_m"]
        scatter = ax.scatter(
            x,
            y,
            c=bundle.diagnostics["wind_mean_prev_48h"],
            cmap=cmap,
            norm=norm,
            s=65,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )
        add_reference_lines(ax)
        metrics = compute_metrics(x, y)
        ax.text(
            0.04,
            0.96,
            format_metrics(metrics),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
        )
        ax.set_title(title)
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed spacing (m)")
        ax.set_ylabel("Predicted spacing at observation (m)")

    cbar = fig.colorbar(scatter, ax=axes, shrink=0.92)
    cbar.set_label("Mean wind over previous 48 h (m/s)")
    save_figure(fig, "plot01_predicted_vs_observed.png")


def plot_residual_analysis(manual: DatasetBundle) -> None:
    diagnostics = manual.diagnostics.copy()
    diagnostics["residual_m"] = diagnostics["spacing_at_obs_m"] - diagnostics["observed_spacing_m"]
    variables = [
        ("wind_mean_prev_48h", "Mean wind over previous 48 h (m/s)"),
        ("Hs_mean_prev_48h", "Mean significant wave height over previous 48 h (m)"),
        ("La_t_mean_prev_48h", "Mean turbulent Langmuir number over previous 48 h"),
        ("integrated_supercriticality_prev_48h", "Integrated supercriticality over previous 48 h"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, (column, xlabel) in zip(axes.ravel(), variables, strict=True):
        ax.scatter(
            diagnostics[column],
            diagnostics["residual_m"],
            s=55,
            color="#3b7ddd",
            alpha=0.8,
            edgecolor="white",
            linewidth=0.4,
        )
        x_fit, y_fit = lowess_like(diagnostics[column], diagnostics["residual_m"])
        if len(x_fit) > 0:
            ax.plot(x_fit, y_fit, color="#c4512d", linewidth=2.0, label="LOWESS-like trend")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residual: predicted - observed (m)")
        ax.legend(loc="best", fontsize=9)

    save_figure(fig, "plot02_residual_analysis_manual.png")


def choose_example_ids(diagnostics: pd.DataFrame) -> list[str]:
    targets = [40.0, 80.0, 220.0]
    available = diagnostics[["observation_id", "observed_spacing_m"]].dropna().copy()
    selected: list[str] = []
    used: set[str] = set()
    for target in targets:
        ranked = available.assign(delta=(available["observed_spacing_m"] - target).abs()).sort_values("delta")
        for _, row in ranked.iterrows():
            obs_id = str(row["observation_id"])
            if obs_id not in used:
                selected.append(obs_id)
                used.add(obs_id)
                break
    return selected


def add_regime_bands(ax: plt.Axes, timeline: pd.DataFrame) -> None:
    for hour, regime in zip(timeline["hours_from_observation"], timeline["regime"], strict=True):
        color = REGIME_COLORS.get(str(regime), "#d9d9d9")
        ax.axvspan(hour - 0.5, hour + 0.5, color=color, alpha=0.15, linewidth=0)


def plot_example_timeline_evolution(manual: DatasetBundle) -> None:
    diagnostics = manual.diagnostics.set_index("observation_id")
    example_ids = choose_example_ids(manual.diagnostics)
    fig = plt.figure(figsize=(15, 20), constrained_layout=True)
    subfigs = fig.subfigures(len(example_ids), 1)
    if len(example_ids) == 1:
        subfigs = [subfigs]

    for subfig, obs_id in zip(subfigs, example_ids, strict=True):
        timeline_path = manual.timeline_dir / f"{obs_id}_timeline.csv"
        timeline = pd.read_csv(timeline_path).sort_values("hours_from_observation")
        observed_spacing = float(diagnostics.loc[obs_id, "observed_spacing_m"])
        depth_m = float(diagnostics.loc[obs_id, "depth_m"])
        target_spacing = theoretical_spacing(timeline["target_l"], depth_m)
        axes = subfig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 1.1]})

        axes[0].plot(timeline["hours_from_observation"], timeline["U10"], color="#1f77b4", linewidth=2.0, label="U10")
        gust_lower = timeline["U10"].to_numpy(dtype=float)
        gust_upper = np.maximum(timeline["wind_gusts_10m"].to_numpy(dtype=float), gust_lower)
        axes[0].fill_between(
            timeline["hours_from_observation"],
            gust_lower,
            gust_upper,
            color="#8ecae6",
            alpha=0.35,
            label="Wind gust envelope",
        )
        axes[0].set_ylabel("Wind speed (m/s)")
        axes[0].legend(loc="upper left", fontsize=9)

        axes[1].plot(
            timeline["hours_from_observation"],
            timeline["predicted_spacing_NL_m"],
            color="#d97706",
            linewidth=2.2,
            label="Predicted NL spacing",
        )
        axes[1].plot(
            timeline["hours_from_observation"],
            target_spacing,
            color="0.45",
            linewidth=1.2,
            label="Target spacing from target_l",
        )
        axes[1].axhline(observed_spacing, color="#b22222", linestyle="--", linewidth=1.5, label="Observed spacing")
        axes[1].set_ylabel("Spacing (m)")
        axes[1].legend(loc="upper left", fontsize=9)

        add_regime_bands(axes[2], timeline)
        axes[2].plot(
            timeline["hours_from_observation"],
            timeline["amplitude_index"],
            color="#2a9d8f",
            linewidth=2.0,
            label="Amplitude index",
        )
        twin = axes[2].twinx()
        twin.plot(
            timeline["hours_from_observation"],
            timeline["development_index"],
            color="#6a4c93",
            linewidth=2.0,
            label="Development index",
        )
        axes[2].set_ylabel("Amplitude index")
        twin.set_ylabel("Development index")

        for axis in (axes[0], axes[1], axes[2], twin):
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.1)

        regime_handles = [
            patches.Patch(color=color, alpha=0.4, label=label.replace("_", " "))
            for label, color in REGIME_COLORS.items()
        ]
        line_handles, line_labels = axes[2].get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        axes[2].legend(
            line_handles + twin_handles + regime_handles,
            line_labels + twin_labels + [handle.get_label() for handle in regime_handles],
            loc="upper left",
            fontsize=8,
            ncol=3,
        )

        for axis in axes:
            axis.set_xlim(-48, 48)
        axes[2].set_xlabel("Hours from observation")
        subfig.suptitle(f"{obs_id}: observed spacing = {observed_spacing:.1f} m", fontsize=15, y=1.02)

    save_figure(fig, "plot03_example_timeline_evolution.png")


def plot_spacing_distribution_comparison(manual: DatasetBundle, wiggle: DatasetBundle) -> None:
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    distributions = [
        ("Manual observed", manual.results["observed_spacing_m"], "#1f77b4"),
        ("Manual predicted", manual.results["predicted_spacing_NL_m"], "#ff7f0e"),
        ("Wiggle observed", wiggle.results["observed_spacing_m"], "#2a9d8f"),
        ("Wiggle predicted", wiggle.results["predicted_spacing_NL_m"], "#c4512d"),
    ]
    bins = np.linspace(0, 400, 22)
    for label, values, color in distributions:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        ax.hist(clean, bins=bins, density=True, alpha=0.28, color=color, label=f"{label} histogram")
        sns.kdeplot(clean, ax=ax, color=color, linewidth=2.0, label=f"{label} KDE", clip=(0, 400))
        ax.axvline(clean.median(), color=color, linestyle="--", linewidth=1.3)

    ax.set_xlim(0, 400)
    ax.set_xlabel("Spacing (m)")
    ax.set_ylabel("Density")
    ax.set_title("Observed and predicted spacing distributions across manual and wiggle datasets")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    save_figure(fig, "plot04_spacing_distribution_comparison.png")


def summarise_timelines(bundle: DatasetBundle) -> pd.DataFrame:
    summaries: list[dict[str, float | str]] = []
    for path in sorted(bundle.timeline_dir.glob("*_timeline.csv")):
        timeline = pd.read_csv(path).sort_values("hours_from_observation")
        if len(timeline) == 0:
            continue
        obs_idx = int(np.argmin(np.abs(timeline["hours_from_observation"].to_numpy(dtype=float))))
        obs_row = timeline.iloc[obs_idx]
        pre = timeline[(timeline["hours_from_observation"] >= -48) & (timeline["hours_from_observation"] <= 0)].copy()
        summaries.append(
            {
                "dataset": bundle.name,
                "observation_id": path.name.replace("_timeline.csv", ""),
                "spacing_at_obs_m": float(obs_row["predicted_spacing_NL_m"]),
                "hours_supercritical_prev_48h": float((pre["regime"] == "supercritical").sum()),
                "wind_mean_prev_48h": float(pre["U10"].mean()),
                "selected_l_at_obs": float(obs_row["selected_l"]),
            }
        )
    return pd.DataFrame(summaries)


def plot_merging_dynamics(manual: DatasetBundle, wiggle: DatasetBundle) -> None:
    timeline_summary = pd.concat([summarise_timelines(manual), summarise_timelines(wiggle)], ignore_index=True)
    diagnostics = pd.concat(
        [
            manual.diagnostics[["dataset", "observation_id", "observed_spacing_m"]],
            wiggle.diagnostics[["dataset", "observation_id", "observed_spacing_m"]],
        ],
        ignore_index=True,
    )
    merged = timeline_summary.merge(diagnostics, on=["dataset", "observation_id"], how="left")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    cmap = get_cmap()
    norm = mcolors.Normalize(float(merged["wind_mean_prev_48h"].min()), float(merged["wind_mean_prev_48h"].max()))

    markers = {"manual": "o", "wiggle": "s"}
    for dataset_name, group in merged.groupby("dataset", sort=False):
        scatter = axes[0].scatter(
            group["hours_supercritical_prev_48h"],
            group["spacing_at_obs_m"],
            c=group["wind_mean_prev_48h"],
            cmap=cmap,
            norm=norm,
            marker=markers.get(dataset_name, "o"),
            s=60,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            label=f"{dataset_name.title()} timelines",
        )
        axes[1].scatter(
            group["selected_l_at_obs"],
            group["observed_spacing_m"],
            marker=markers.get(dataset_name, "o"),
            s=60,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            label=f"{dataset_name.title()} timelines",
        )

    selected_range = np.linspace(
        float(np.nanmin(merged["selected_l_at_obs"])),
        float(np.nanmax(merged["selected_l_at_obs"])),
        300,
    )
    axes[1].plot(
        selected_range,
        theoretical_spacing(selected_range, DEFAULT_DEPTH_M),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Theory: 2pi x {DEFAULT_DEPTH_M:.1f} / selected_l",
    )

    axes[0].set_xlabel("Hours supercritical over previous 48 h")
    axes[0].set_ylabel("Predicted spacing at observation (m)")
    axes[0].legend(loc="best", fontsize=9)
    cbar = fig.colorbar(scatter, ax=axes[0], shrink=0.9)
    cbar.set_label("Mean wind over previous 48 h (m/s)")

    axes[1].set_xlabel("selected_l at observation")
    axes[1].set_ylabel("Observed spacing (m)")
    axes[1].legend(loc="best", fontsize=9)

    save_figure(fig, "plot05_merging_dynamics_exploration.png")


def plot_feature_correlation_heatmap(manual: DatasetBundle) -> None:
    diagnostics = manual.diagnostics.copy()
    numeric = diagnostics.select_dtypes(include=[np.number])
    correlations = numeric.corr(numeric_only=True)["observed_spacing_m"].drop(labels=["observed_spacing_m"]).dropna()
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    heatmap_data = pd.DataFrame([correlations.values], columns=correlations.index, index=["Observed spacing"])

    fig_width = max(12, 0.33 * len(correlations))
    fig, ax = plt.subplots(figsize=(fig_width, 2.8), constrained_layout=True)
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Pearson correlation"},
    )
    ax.set_title("Correlation of diagnostic features with observed spacing")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, "plot06_feature_correlation_heatmap.png")


def plot_wind_history_signatures(manual: DatasetBundle) -> None:
    diagnostics = manual.diagnostics.copy()
    diagnostics["spacing_tercile"] = pd.qcut(
        diagnostics["observed_spacing_m"],
        q=3,
        labels=["Small", "Medium", "Large"],
        duplicates="drop",
    )
    variables = [
        ("wind_mean_prev_48h", "Mean wind over previous 48 h (m/s)"),
        ("coherence_mean_prev_48h", "Mean coherence over previous 48 h"),
        ("integrated_supercriticality_prev_48h", "Integrated supercriticality over previous 48 h"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    for ax, (column, ylabel) in zip(axes, variables, strict=True):
        sns.boxplot(
            data=diagnostics,
            x="spacing_tercile",
            y=column,
            ax=ax,
            color="#8ecae6",
            fliersize=3,
        )
        sns.stripplot(
            data=diagnostics,
            x="spacing_tercile",
            y=column,
            ax=ax,
            color="#1d3557",
            size=4,
            alpha=0.6,
            jitter=0.18,
        )
        ax.set_xlabel("Observed spacing tercile")
        ax.set_ylabel(ylabel)

    save_figure(fig, "plot07_wind_history_signatures.png")


def main() -> None:
    setup_style()
    manual = load_bundle("manual", MANUAL_DIR)
    wiggle = load_bundle("wiggle", WIGGLE_DIR)

    plot_predicted_vs_observed(manual, wiggle)
    plot_residual_analysis(manual)
    plot_example_timeline_evolution(manual)
    plot_spacing_distribution_comparison(manual, wiggle)
    plot_merging_dynamics(manual, wiggle)
    plot_feature_correlation_heatmap(manual)
    plot_wind_history_signatures(manual)

    generated = sorted(path.name for path in PLOTS_DIR.glob("plot*.png"))
    print("Generated figures:")
    for name in generated:
        print(f"- {name}")


if __name__ == "__main__":
    main()
