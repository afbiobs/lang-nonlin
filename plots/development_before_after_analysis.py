#!/usr/bin/env python3
"""Analyse before-versus-after development-index dynamics across timelines."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from output_locator import latest_timeline_output


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"
OUTPUTS_DIR = ROOT / "outputs"
DPI = 150
DATASETS = {
    "manual": latest_timeline_output("manual") / "timelines",
    "wiggle": latest_timeline_output("wiggle") / "timelines",
}


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.9)


def save_figure(fig: plt.Figure, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=DPI)
    plt.close(fig)


def load_hourly_development() -> pd.DataFrame:
    frames = []
    for dataset_name, timeline_dir in DATASETS.items():
        for path in sorted(timeline_dir.glob("*_timeline.csv")):
            df = pd.read_csv(path, usecols=["hours_from_observation", "development_index"])
            df["dataset"] = dataset_name
            df["observation_id"] = path.stem.replace("_timeline", "")
            df["period"] = np.where(df["hours_from_observation"] < 0.0, "before", "after")
            frames.append(df)
    hourly = pd.concat(frames, ignore_index=True)
    hourly["period"] = pd.Categorical(hourly["period"], categories=["before", "after"], ordered=True)
    return hourly


def summarise_by_observation(hourly: pd.DataFrame) -> pd.DataFrame:
    summary = (
        hourly.groupby(["dataset", "observation_id", "period"], as_index=False, observed=False)["development_index"]
        .agg(
            development_mean="mean",
            development_q90=lambda s: float(s.quantile(0.9)),
        )
    )
    pivot = (
        summary.pivot_table(
            index=["dataset", "observation_id"],
            columns="period",
            values=["development_mean", "development_q90"],
            aggfunc="first",
            observed=False,
        )
        .reset_index()
    )
    pivot.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in pivot.columns
    ]
    return pivot


def regression_report(x: pd.Series, y: pd.Series) -> dict[str, float]:
    mask = x.notna() & y.notna()
    x_valid = x[mask].to_numpy(dtype=float)
    y_valid = y[mask].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    pred = slope * x_valid + intercept
    ss_res = float(np.sum((y_valid - pred) ** 2))
    ss_tot = float(np.sum((y_valid - y_valid.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(x_valid, y_valid)[0, 1]) if len(x_valid) > 1 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "r": r}


def statistical_tests(hourly: pd.DataFrame, obs_summary: pd.DataFrame) -> dict[str, float]:
    before_hourly = hourly.loc[hourly["period"] == "before", "development_index"].to_numpy(dtype=float)
    after_hourly = hourly.loc[hourly["period"] == "after", "development_index"].to_numpy(dtype=float)
    before_q90 = obs_summary["development_q90_before"].to_numpy(dtype=float)
    after_q90 = obs_summary["development_q90_after"].to_numpy(dtype=float)

    ks = stats.ks_2samp(before_hourly, after_hourly)
    mann_whitney = stats.mannwhitneyu(before_hourly, after_hourly, alternative="two-sided")
    wilcoxon = stats.wilcoxon(before_q90, after_q90, alternative="two-sided", zero_method="wilcox")
    paired_t = stats.ttest_rel(before_q90, after_q90)

    return {
        "hourly_before_mean": float(np.mean(before_hourly)),
        "hourly_after_mean": float(np.mean(after_hourly)),
        "obs_before_q90_mean": float(np.mean(before_q90)),
        "obs_after_q90_mean": float(np.mean(after_q90)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "mannwhitney_u": float(mann_whitney.statistic),
        "mannwhitney_pvalue": float(mann_whitney.pvalue),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_pvalue": float(wilcoxon.pvalue),
        "paired_t_statistic": float(paired_t.statistic),
        "paired_t_pvalue": float(paired_t.pvalue),
    }


def plot_obs_mean_scatter_by_dataset(obs_summary: pd.DataFrame, reports: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True, sharex=True, sharey=True)
    dataset_order = ["manual", "wiggle"]
    colors = {"manual": "#1d3557", "wiggle": "#d97706"}
    all_min = float(min(obs_summary["development_q90_before"].min(), obs_summary["development_q90_after"].min()))
    all_max = float(max(obs_summary["development_q90_before"].max(), obs_summary["development_q90_after"].max()))

    for ax, dataset in zip(axes, dataset_order, strict=True):
        subset = obs_summary[obs_summary["dataset"] == dataset].copy()
        fit = reports[dataset]["regression"]
        x_line = np.linspace(all_min, all_max, 200)
        ax.scatter(
            subset["development_q90_before"],
            subset["development_q90_after"],
            s=75,
            color=colors[dataset],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.plot(x_line, x_line, color="0.35", linestyle="--", linewidth=1.2)
        ax.plot(x_line, fit["slope"] * x_line + fit["intercept"], color="#c4512d", linewidth=2.0)
        ax.set_title(f"{dataset.title()} observations")
        ax.set_xlabel("Q90 development index before (-48 to <0 h)")
        ax.set_ylabel("Q90 development index after (0 to +48 h)")
        ax.text(
            0.04,
            0.96,
            f"after = {fit['slope']:.3f} x before + {fit['intercept']:.3f}\nR^2 = {fit['r2']:.3f}\nr = {fit['r']:.3f}\nN = {len(subset)} obs",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
        )
    save_figure(fig, "plot10_development_q90_before_after_scatter_by_dataset.png")


def plot_distributions_by_dataset(hourly: pd.DataFrame, obs_summary: pd.DataFrame, reports: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    dataset_order = ["manual", "wiggle"]

    obs_long = obs_summary.melt(
        id_vars=["dataset", "observation_id"],
        value_vars=["development_q90_before", "development_q90_after"],
        var_name="window",
        value_name="development_q90",
    )
    obs_long["window"] = obs_long["window"].map(
        {"development_q90_before": "before", "development_q90_after": "after"}
    )

    for row_idx, dataset in enumerate(dataset_order):
        hourly_subset = hourly[hourly["dataset"] == dataset].copy()
        obs_subset = obs_long[obs_long["dataset"] == dataset].copy()
        tests = reports[dataset]["tests"]

        sns.violinplot(
            data=hourly_subset,
            x="period",
            y="development_index",
            inner="quartile",
            cut=0,
            hue="period",
            palette=["#8ecae6", "#f4a261"],
            ax=axes[row_idx, 0],
            legend=False,
        )
        axes[row_idx, 0].set_xlabel("")
        axes[row_idx, 0].set_ylabel("Hourly development index")
        axes[row_idx, 0].set_title(f"{dataset.title()}: pooled hourly values")
        axes[row_idx, 0].text(
            0.04,
            0.96,
            f"KS p = {tests['ks_pvalue']:.3g}\nMann-Whitney p = {tests['mannwhitney_pvalue']:.3g}",
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
        )

        sns.boxplot(
            data=obs_subset,
            x="window",
            y="development_q90",
            hue="window",
            palette=["#8ecae6", "#f4a261"],
            ax=axes[row_idx, 1],
            fliersize=3,
            legend=False,
        )
        sns.stripplot(
            data=obs_subset,
            x="window",
            y="development_q90",
            color="#1d3557" if dataset == "manual" else "#d97706",
            alpha=0.55,
            size=4,
            ax=axes[row_idx, 1],
        )
        axes[row_idx, 1].set_xlabel("")
        axes[row_idx, 1].set_ylabel("Per-observation Q90 development index")
        axes[row_idx, 1].set_title(f"{dataset.title()}: per-observation Q90 values")
        axes[row_idx, 1].text(
            0.04,
            0.96,
            f"Wilcoxon p = {tests['wilcoxon_pvalue']:.3g}\nPaired t-test p = {tests['paired_t_pvalue']:.3g}\nQ90 means: {tests['obs_before_q90_mean']:.3f} vs {tests['obs_after_q90_mean']:.3f}",
            transform=axes[row_idx, 1].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
        )

    save_figure(fig, "plot11_development_q90_before_after_distributions_by_dataset.png")


def plot_stacked_timeseries(hourly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, sharex=True, sharey=True)
    colors = {"manual": "#1d3557", "wiggle": "#d97706"}
    for ax, dataset in zip(axes, ["manual", "wiggle"], strict=True):
        subset = hourly[hourly["dataset"] == dataset].copy()
        for _, group in subset.groupby("observation_id", sort=False):
            ax.plot(
                group["hours_from_observation"],
                group["development_index"],
                color=colors[dataset],
                alpha=0.18,
                linewidth=1.0,
            )
        median_curve = (
            subset.groupby("hours_from_observation", as_index=False)["development_index"]
            .median()
            .sort_values("hours_from_observation")
        )
        ax.plot(
            median_curve["hours_from_observation"],
            median_curve["development_index"],
            color="black",
            linewidth=2.2,
            label="Median across observations",
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{dataset.title()} timelines")
        ax.set_xlabel("Hours from observation")
        ax.set_ylabel("Development index")
        ax.legend(loc="lower right", fontsize=9)
    save_figure(fig, "plot12_development_timeseries_stacked_by_dataset.png")


def build_reports(hourly: pd.DataFrame, obs_summary: pd.DataFrame) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for dataset in ["manual", "wiggle"]:
        hourly_subset = hourly[hourly["dataset"] == dataset].copy()
        obs_subset = obs_summary[obs_summary["dataset"] == dataset].copy()
        reports[dataset] = {
            "n_observations": int(len(obs_subset)),
            "regression": regression_report(
                obs_subset["development_q90_before"],
                obs_subset["development_q90_after"],
            ),
            "tests": statistical_tests(hourly_subset, obs_subset),
        }
    return reports


def write_summary(reports: dict[str, dict]) -> None:
    summary = reports
    out_path = PLOTS_DIR / "development_before_after_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    setup_style()
    hourly = load_hourly_development()
    obs_summary = summarise_by_observation(hourly)
    reports = build_reports(hourly, obs_summary)
    plot_obs_mean_scatter_by_dataset(obs_summary, reports)
    plot_distributions_by_dataset(hourly, obs_summary, reports)
    plot_stacked_timeseries(hourly)
    write_summary(reports)


if __name__ == "__main__":
    main()
