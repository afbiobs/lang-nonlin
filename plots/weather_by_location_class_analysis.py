#!/usr/bin/env python3
"""Compare manual versus wiggle weather histories by location."""

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
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
DPI = 150

DATASETS = {
    "manual": latest_timeline_output("manual") / "observation_diagnostics.csv",
    "wiggle": latest_timeline_output("wiggle") / "observation_diagnostics.csv",
}

LOCATION_ORDER = ["China", "Ireland", "America"]
DATASET_ORDER = ["manual", "wiggle"]
DATASET_COLORS = {"manual": "#1d3557", "wiggle": "#d97706"}
WEATHER_LABELS = {
    "wind_mean_prev_48h": "Mean wind\nprevious 48 h (m/s)",
    "wind_max_prev_48h": "Max wind\nprevious 48 h (m/s)",
    "wind_std_prev_48h": "Wind variability\nprevious 48 h (m/s)",
    "gust_mean_prev_48h": "Mean gust\nprevious 48 h (m/s)",
    "wind_steadiness_prev_48h": "Wind steadiness\nprevious 48 h",
    "turning_mean_prev_48h_deg": "Mean turning\nprevious 48 h (deg/h)",
    "turning_total_prev_48h_deg": "Total turning\nprevious 48 h (deg)",
    "coherence_mean_prev_48h": "Coherence mean\nprevious 48 h",
    "Hs_mean_prev_48h": "Mean significant\nwave height (m)",
    "lambda_p_mean_prev_48h": "Mean peak wavelength\nprevious 48 h (m)",
    "La_t_mean_prev_48h": "Mean turbulent\nLangmuir number",
    "D_max_mean_prev_48h": "Mean max divergence\nprevious 48 h",
    "hours_supercritical_prev_48h": "Supercritical hours\nprevious 48 h",
    "integrated_supercriticality_prev_48h": "Integrated supercriticality\nprevious 48 h",
}
KEY_WEATHER_VARS = [
    "wind_mean_prev_48h",
    "wind_std_prev_48h",
    "wind_steadiness_prev_48h",
    "hours_supercritical_prev_48h",
]


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.88)


def save_figure(fig: plt.Figure, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=DPI)
    plt.close(fig)


def assign_location(longitude: float) -> str:
    if longitude > 100.0:
        return "China"
    if longitude > -20.0:
        return "Ireland"
    return "America"


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(dtype=float)
    n = len(p)
    if n == 0:
        return pd.Series(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        running = min(running, ranked[idx] * n / rank)
        adjusted[idx] = running
    q = np.empty(n, dtype=float)
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(q, index=pvalues.index)


def cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    greater = sum((value > b).sum() for value in a)
    lower = sum((value < b).sum() for value in a)
    return float((greater - lower) / (len(a) * len(b)))


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def regression_fit(x: pd.Series, y: pd.Series) -> dict[str, float]:
    mask = x.notna() & y.notna()
    x_vals = x[mask].to_numpy(dtype=float)
    y_vals = y[mask].to_numpy(dtype=float)
    if len(x_vals) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "r": float("nan")}
    if np.isclose(np.std(x_vals), 0.0):
        return {"slope": 0.0, "intercept": float(np.mean(y_vals)), "r2": 0.0, "r": float("nan")}
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    pred = slope * x_vals + intercept
    ss_res = float(np.sum((y_vals - pred) ** 2))
    ss_tot = float(np.sum((y_vals - y_vals.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(x_vals, y_vals)[0, 1]) if len(x_vals) > 1 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "r": r}


def load_data() -> pd.DataFrame:
    observations = pd.read_csv(DATA_DIR / "observations_minimal.csv")
    observations["location"] = observations["authoritative_lng"].map(assign_location)
    frames = []
    for dataset, path in DATASETS.items():
        diagnostics = pd.read_csv(path)
        merged = diagnostics.merge(
            observations[["observation_id", "authoritative_lat", "authoritative_lng", "location"]],
            on="observation_id",
            how="left",
        )
        merged["dataset"] = dataset
        frames.append(merged)
    combined = pd.concat(frames, ignore_index=True)
    combined["dataset"] = pd.Categorical(combined["dataset"], categories=DATASET_ORDER, ordered=True)
    combined["location"] = pd.Categorical(combined["location"], categories=LOCATION_ORDER, ordered=True)
    return combined


def compute_development_summary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    tests = []
    for location in LOCATION_ORDER:
        subset = data[data["location"] == location]
        manual = subset.loc[subset["dataset"] == "manual", "development_at_obs"].dropna().to_numpy(dtype=float)
        wiggle = subset.loc[subset["dataset"] == "wiggle", "development_at_obs"].dropna().to_numpy(dtype=float)
        if len(manual) == 0 or len(wiggle) == 0:
            continue
        mw = stats.mannwhitneyu(wiggle, manual, alternative="two-sided")
        ks = stats.ks_2samp(wiggle, manual)
        delta = cliffs_delta(wiggle, manual)
        d_value = cohens_d(wiggle, manual)
        auc = float(mw.statistic / (len(wiggle) * len(manual)))
        records.append(
            {
                "location": location,
                "manual_n": int(len(manual)),
                "wiggle_n": int(len(wiggle)),
                "manual_mean": float(np.mean(manual)),
                "wiggle_mean": float(np.mean(wiggle)),
                "manual_median": float(np.median(manual)),
                "wiggle_median": float(np.median(wiggle)),
                "mean_diff_wiggle_minus_manual": float(np.mean(wiggle) - np.mean(manual)),
                "median_diff_wiggle_minus_manual": float(np.median(wiggle) - np.median(manual)),
                "cliffs_delta_wiggle_vs_manual": delta,
                "cohens_d_wiggle_vs_manual": d_value,
                "auc_wiggle_gt_manual": auc,
                "mannwhitney_pvalue": float(mw.pvalue),
                "ks_pvalue": float(ks.pvalue),
            }
        )
        tests.append({"location": location, "pvalue": float(mw.pvalue)})
    summary = pd.DataFrame.from_records(records)
    if not summary.empty:
        q_values = benjamini_hochberg(summary["mannwhitney_pvalue"])
        summary["mannwhitney_qvalue"] = q_values
    return summary, pd.DataFrame.from_records(tests)


def compute_weather_summary(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for location in LOCATION_ORDER:
        subset = data[data["location"] == location]
        for variable, label in WEATHER_LABELS.items():
            manual = subset.loc[subset["dataset"] == "manual", variable].dropna().to_numpy(dtype=float)
            wiggle = subset.loc[subset["dataset"] == "wiggle", variable].dropna().to_numpy(dtype=float)
            if len(manual) < 2 or len(wiggle) < 2:
                continue
            mw = stats.mannwhitneyu(wiggle, manual, alternative="two-sided")
            ks = stats.ks_2samp(wiggle, manual)
            records.append(
                {
                    "location": location,
                    "variable": variable,
                    "label": label,
                    "manual_n": int(len(manual)),
                    "wiggle_n": int(len(wiggle)),
                    "manual_mean": float(np.mean(manual)),
                    "wiggle_mean": float(np.mean(wiggle)),
                    "manual_median": float(np.median(manual)),
                    "wiggle_median": float(np.median(wiggle)),
                    "manual_std": float(np.std(manual, ddof=1)),
                    "wiggle_std": float(np.std(wiggle, ddof=1)),
                    "mean_diff_wiggle_minus_manual": float(np.mean(wiggle) - np.mean(manual)),
                    "median_diff_wiggle_minus_manual": float(np.median(wiggle) - np.median(manual)),
                    "cohens_d_wiggle_vs_manual": cohens_d(wiggle, manual),
                    "cliffs_delta_wiggle_vs_manual": cliffs_delta(wiggle, manual),
                    "auc_wiggle_gt_manual": float(mw.statistic / (len(wiggle) * len(manual))),
                    "mannwhitney_pvalue": float(mw.pvalue),
                    "ks_pvalue": float(ks.pvalue),
                }
            )
    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary
    summary["mannwhitney_qvalue"] = np.nan
    for location in LOCATION_ORDER:
        mask = summary["location"] == location
        summary.loc[mask, "mannwhitney_qvalue"] = benjamini_hochberg(summary.loc[mask, "mannwhitney_pvalue"])
    summary["abs_cliffs_delta"] = summary["cliffs_delta_wiggle_vs_manual"].abs()
    summary["abs_cohens_d"] = summary["cohens_d_wiggle_vs_manual"].abs()
    return summary.sort_values(["location", "abs_cliffs_delta"], ascending=[True, False]).reset_index(drop=True)


def plot_development_gap(data: pd.DataFrame, development_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True, sharey=True)
    for ax, location in zip(axes, LOCATION_ORDER, strict=True):
        subset = data[data["location"] == location].copy()
        sns.boxplot(
            data=subset,
            x="dataset",
            y="development_at_obs",
            hue="dataset",
            order=DATASET_ORDER,
            palette=DATASET_COLORS,
            fliersize=0,
            width=0.55,
            dodge=False,
            ax=ax,
            legend=False,
        )
        sns.stripplot(
            data=subset,
            x="dataset",
            y="development_at_obs",
            order=DATASET_ORDER,
            hue="dataset",
            palette=DATASET_COLORS,
            dodge=False,
            alpha=0.7,
            size=5,
            linewidth=0.35,
            edgecolor="white",
            ax=ax,
            legend=False,
        )
        row = development_summary[development_summary["location"] == location].iloc[0]
        ax.set_title(f"{location} (manual n={int(row['manual_n'])}, wiggle n={int(row['wiggle_n'])})")
        ax.set_xlabel("")
        ax.set_ylabel("Development index at observation")
        ax.set_ylim(0.0, 1.02)
        ax.text(
            0.04,
            0.96,
            (
                f"Wiggle-manual mean = {row['mean_diff_wiggle_minus_manual']:.3f}\n"
                f"Cliff's delta = {row['cliffs_delta_wiggle_vs_manual']:.2f}\n"
                f"MW q = {row['mannwhitney_qvalue']:.3g}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
        )
    fig.suptitle("Development-index separation is present in every location")
    save_figure(fig, "plot13_development_gap_by_location.png")


def plot_weather_heatmap(weather_summary: pd.DataFrame) -> None:
    heatmap = (
        weather_summary.pivot(index="label", columns="location", values="cliffs_delta_wiggle_vs_manual")
        .reindex(index=[WEATHER_LABELS[var] for var in WEATHER_LABELS], columns=LOCATION_ORDER)
    )
    qvalues = (
        weather_summary.pivot(index="label", columns="location", values="mannwhitney_qvalue")
        .reindex(index=heatmap.index, columns=heatmap.columns)
    )
    annotations = heatmap.copy().astype(object)
    for row_label in heatmap.index:
        for column in heatmap.columns:
            value = heatmap.loc[row_label, column]
            qvalue = qvalues.loc[row_label, column]
            if pd.isna(value):
                annotations.loc[row_label, column] = ""
                continue
            marker = "*" if pd.notna(qvalue) and qvalue < 0.10 else ""
            annotations.loc[row_label, column] = f"{value:.2f}{marker}"

    fig, ax = plt.subplots(figsize=(8.4, 10.5), constrained_layout=True)
    sns.heatmap(
        heatmap,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.8,
        linecolor="white",
        annot=annotations,
        fmt="",
        cbar_kws={"label": "Cliff's delta (wiggle higher -> positive)"},
        ax=ax,
    )
    ax.set_title("Weather-history differences between wiggle and manual by location\n* marks Benjamini-Hochberg q < 0.10 within a location")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "plot14_weather_effect_size_heatmap_by_location.png")


def plot_key_weather_distributions(data: pd.DataFrame, weather_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(KEY_WEATHER_VARS), len(LOCATION_ORDER), figsize=(15, 14), constrained_layout=True)
    for row_idx, variable in enumerate(KEY_WEATHER_VARS):
        for col_idx, location in enumerate(LOCATION_ORDER):
            ax = axes[row_idx, col_idx]
            subset = data[data["location"] == location].copy()
            sns.boxplot(
                data=subset,
                x="dataset",
                y=variable,
                hue="dataset",
                order=DATASET_ORDER,
                palette=DATASET_COLORS,
                width=0.55,
                fliersize=0,
                dodge=False,
                ax=ax,
                legend=False,
            )
            sns.stripplot(
                data=subset,
                x="dataset",
                y=variable,
                order=DATASET_ORDER,
                hue="dataset",
                palette=DATASET_COLORS,
                dodge=False,
                alpha=0.65,
                size=4.5,
                linewidth=0.3,
                edgecolor="white",
                ax=ax,
                legend=False,
            )
            summary_row = weather_summary[
                (weather_summary["location"] == location) & (weather_summary["variable"] == variable)
            ].iloc[0]
            if row_idx == 0:
                ax.set_title(f"{location}\nmanual n={int(summary_row['manual_n'])}, wiggle n={int(summary_row['wiggle_n'])}")
            ax.set_xlabel("")
            ax.set_ylabel(WEATHER_LABELS[variable])
            ax.text(
                0.04,
                0.96,
                (
                    f"Delta = {summary_row['mean_diff_wiggle_minus_manual']:.2f}\n"
                    f"Cliff's delta = {summary_row['cliffs_delta_wiggle_vs_manual']:.2f}\n"
                    f"MW q = {summary_row['mannwhitney_qvalue']:.3g}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.93},
            )
    fig.suptitle("Key weather-history variables most relevant to wind intensity, steadiness, and persistence")
    save_figure(fig, "plot15_key_weather_distributions_by_location.png")


def plot_development_vs_weather(data: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    regression_summary: dict[str, dict[str, dict[str, float]]] = {}
    fig, axes = plt.subplots(len(LOCATION_ORDER), len(KEY_WEATHER_VARS), figsize=(18, 12.5), constrained_layout=True)
    for row_idx, location in enumerate(LOCATION_ORDER):
        regression_summary[location] = {}
        for col_idx, variable in enumerate(KEY_WEATHER_VARS):
            ax = axes[row_idx, col_idx]
            subset = data[data["location"] == location].copy()
            for dataset in DATASET_ORDER:
                group = subset[subset["dataset"] == dataset].copy()
                ax.scatter(
                    group[variable],
                    group["development_at_obs"],
                    s=55,
                    alpha=0.8,
                    color=DATASET_COLORS[dataset],
                    edgecolor="white",
                    linewidth=0.35,
                    label=dataset if row_idx == 0 and col_idx == 0 else None,
                )
                fit = regression_fit(group[variable], group["development_at_obs"])
                regression_summary[location][f"{dataset}:{variable}"] = fit
                valid = group[[variable, "development_at_obs"]].dropna()
                if len(valid) >= 2:
                    x_line = np.linspace(valid[variable].min(), valid[variable].max(), 100)
                    ax.plot(
                        x_line,
                        fit["slope"] * x_line + fit["intercept"],
                        color=DATASET_COLORS[dataset],
                        linewidth=1.6,
                        alpha=0.9,
                    )
            if row_idx == 0:
                ax.set_title(WEATHER_LABELS[variable])
            if col_idx == 0:
                ax.set_ylabel(f"{location}\nDevelopment index at observation")
            else:
                ax.set_ylabel("")
            if row_idx == len(LOCATION_ORDER) - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("")
            ax.set_ylim(0.0, 1.02)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DATASET_COLORS[name], markeredgecolor="white",
                   markersize=8, label=name.title())
        for name in DATASET_ORDER
    ]
    fig.legend(handles=handles, labels=[name.title() for name in DATASET_ORDER], loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Development index versus key wind-history variables by location")
    save_figure(fig, "plot16_development_vs_key_weather_by_location.png")
    return regression_summary


def build_summary_json(
    data: pd.DataFrame,
    development_summary: pd.DataFrame,
    weather_summary: pd.DataFrame,
    regression_summary: dict[str, dict[str, dict[str, float]]],
) -> dict[str, object]:
    counts = (
        data.groupby(["location", "dataset"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=LOCATION_ORDER, columns=DATASET_ORDER)
    )
    top_signals = {}
    for location in LOCATION_ORDER:
        subset = weather_summary[weather_summary["location"] == location].copy()
        top_rows = subset.sort_values("abs_cliffs_delta", ascending=False).head(5)
        top_signals[location] = [
            {
                "variable": row["variable"],
                "label": row["label"],
                "mean_diff_wiggle_minus_manual": float(row["mean_diff_wiggle_minus_manual"]),
                "cliffs_delta_wiggle_vs_manual": float(row["cliffs_delta_wiggle_vs_manual"]),
                "mannwhitney_pvalue": float(row["mannwhitney_pvalue"]),
                "mannwhitney_qvalue": float(row["mannwhitney_qvalue"]),
            }
            for _, row in top_rows.iterrows()
        ]
    return {
        "counts_by_location_dataset": counts.astype(int).to_dict(),
        "development_summary": development_summary.to_dict(orient="records"),
        "top_weather_signals_by_location": top_signals,
        "regression_summary": regression_summary,
    }


def main() -> None:
    setup_style()
    data = load_data()
    development_summary, _ = compute_development_summary(data)
    weather_summary = compute_weather_summary(data)
    regression_summary = plot_development_vs_weather(data)

    plot_development_gap(data, development_summary)
    plot_weather_heatmap(weather_summary)
    plot_key_weather_distributions(data, weather_summary)

    weather_summary.to_csv(PLOTS_DIR / "weather_by_location_class_summary.csv", index=False)
    summary = build_summary_json(data, development_summary, weather_summary, regression_summary)
    (PLOTS_DIR / "weather_by_location_class_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
