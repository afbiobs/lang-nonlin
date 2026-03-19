#!/usr/bin/env python3
"""Plot the visible / CL / response spacing decomposition."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
DPI = 150
DATASETS = {
    "manual": latest_timeline_output("manual"),
    "wiggle": latest_timeline_output("wiggle"),
}
COMPONENTS = {
    "visible": {
        "column": "spacing_visible_at_obs_m",
        "timeline_column": "predicted_spacing_visible_m",
        "color": "#d97706",
        "label": "Visible",
    },
    "cl": {
        "column": "spacing_cl_at_obs_m",
        "timeline_column": "predicted_spacing_CL_m",
        "color": "#1d3557",
        "label": "CL",
    },
    "response": {
        "column": "spacing_response_at_obs_m",
        "timeline_column": "predicted_spacing_response_m",
        "color": "#2a9d8f",
        "label": "Response",
    },
    "linear": {
        "column": "spacing_linear_at_obs_m",
        "timeline_column": "predicted_spacing_L_m",
        "color": "#6b7280",
        "label": "Linear",
    },
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


def component_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    mask = observed.notna() & predicted.notna() & (observed > 0.0) & (predicted > 0.0)
    obs = observed[mask].to_numpy(dtype=float)
    pred = predicted[mask].to_numpy(dtype=float)
    if len(obs) == 0:
        return {"rmse": float("nan"), "bias": float("nan"), "capture_rate": float("nan"), "median_ratio": float("nan")}
    residual = pred - obs
    ratio = obs / pred
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "capture_rate": float((((pred / obs) >= 0.5) & ((pred / obs) <= 2.0)).mean()),
        "median_ratio": float(np.median(ratio)),
    }


def build_skill_table(bundles: list[DatasetBundle]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for bundle in bundles:
        observed = bundle.diagnostics["observed_spacing_m"]
        for component_name, spec in COMPONENTS.items():
            stats = component_metrics(observed, bundle.diagnostics.get(spec["column"], pd.Series(dtype=float)))
            rows.append(
                {
                    "dataset": bundle.name,
                    "component": spec["label"],
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def plot_ranked_components(bundles: list[DatasetBundle]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), constrained_layout=True, sharey=True)
    for ax, bundle in zip(axes, bundles, strict=True):
        ordered = bundle.diagnostics.sort_values("observed_spacing_m").reset_index(drop=True)
        x = np.arange(1, len(ordered) + 1)
        ax.plot(
            x,
            ordered["observed_spacing_m"],
            color="#111827",
            linewidth=2.8,
            marker="o",
            markersize=3,
            label="Observed",
        )
        for component_name in ("visible", "cl", "response", "linear"):
            spec = COMPONENTS[component_name]
            if spec["column"] not in ordered.columns:
                continue
            ax.plot(
                x,
                ordered[spec["column"]],
                color=spec["color"],
                linewidth=2.0 if component_name == "visible" else 1.6,
                linestyle="-" if component_name == "visible" else "--",
                label=spec["label"],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Observation rank (sorted by observed spacing)")
        ax.set_ylabel("Spacing (m)")
        ax.set_title(f"{bundle.name.title()} dataset")
        ax.legend(loc="lower right", fontsize=9, ncol=2)
    save_figure(fig, "plot21_hybrid_spacing_components_ranked.png")


def plot_skill_comparison(skill: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
    order = ["Visible", "CL", "Response", "Linear"]
    palette = {"manual": "#1d3557", "wiggle": "#d97706"}

    sns.barplot(
        data=skill,
        x="component",
        y="rmse",
        hue="dataset",
        order=order,
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE against observed spacing (m)")
    axes[0].set_title("Skill by spacing component")

    sns.barplot(
        data=skill,
        x="component",
        y="capture_rate",
        hue="dataset",
        order=order,
        palette=palette,
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Capture rate within 2x")
    axes[1].set_title("Observed capture improves only if the visible mapping works")
    for ax in axes:
        ax.legend(loc="best", fontsize=9)
    save_figure(fig, "plot22_hybrid_component_skill.png")


def plot_ratio_distributions(bundles: list[DatasetBundle]) -> None:
    records: list[dict[str, float | str]] = []
    for bundle in bundles:
        observed = bundle.diagnostics["observed_spacing_m"]
        for component_name, spec in COMPONENTS.items():
            if spec["column"] not in bundle.diagnostics.columns:
                continue
            predicted = bundle.diagnostics[spec["column"]]
            mask = observed.notna() & predicted.notna() & (observed > 0.0) & (predicted > 0.0)
            for ratio in (observed[mask] / predicted[mask]).to_numpy(dtype=float):
                records.append({"dataset": bundle.name, "component": spec["label"], "ratio": float(ratio)})
    ratios = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True, sharey=True)
    for ax, dataset in zip(axes, ("manual", "wiggle"), strict=True):
        subset = ratios[ratios["dataset"] == dataset]
        sns.boxplot(
            data=subset,
            x="component",
            y="ratio",
            order=["Visible", "CL", "Response", "Linear"],
            hue="component",
            palette={spec["label"]: spec["color"] for spec in COMPONENTS.values()},
            ax=ax,
            fliersize=3,
            legend=False,
        )
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("Observed spacing / predicted spacing")
        ax.set_title(f"{dataset.title()} dataset")
    save_figure(fig, "plot23_hybrid_component_ratios.png")


def choose_example_ids(bundle: DatasetBundle) -> list[str]:
    ranked = bundle.diagnostics[["observation_id", "observed_spacing_m"]].dropna().sort_values("observed_spacing_m")
    if len(ranked) == 0:
        return []
    indices = sorted({0, len(ranked) // 2, len(ranked) - 1})
    return [str(ranked.iloc[idx]["observation_id"]) for idx in indices]


def plot_example_timelines(bundle: DatasetBundle) -> None:
    example_ids = choose_example_ids(bundle)
    if not example_ids:
        return
    fig, axes = plt.subplots(len(example_ids), 1, figsize=(14.5, 12.5), constrained_layout=True, sharex=True)
    if len(example_ids) == 1:
        axes = [axes]

    for ax, observation_id in zip(axes, example_ids, strict=True):
        timeline_path = bundle.base_dir / "timelines" / f"{observation_id}_timeline.csv"
        timeline = pd.read_csv(timeline_path).sort_values("hours_from_observation")
        observed_spacing = float(
            bundle.diagnostics.loc[bundle.diagnostics["observation_id"] == observation_id, "observed_spacing_m"].iloc[0]
        )
        ax.plot(
            timeline["hours_from_observation"],
            timeline["predicted_spacing_NL_m"],
            color=COMPONENTS["visible"]["color"],
            linewidth=2.4,
            label="Visible",
        )
        for component_name in ("cl", "response", "linear"):
            spec = COMPONENTS[component_name]
            column = spec["timeline_column"]
            if column not in timeline.columns:
                continue
            ax.plot(
                timeline["hours_from_observation"],
                timeline[column],
                color=spec["color"],
                linewidth=1.8,
                linestyle="--" if component_name != "response" else ":",
                label=spec["label"],
            )
        ax.axhline(observed_spacing, color="#111827", linestyle="-.", linewidth=1.6, label="Observed")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_ylabel("Spacing (m)")
        ax.set_title(f"{observation_id}: observed = {observed_spacing:.1f} m")
        ax.legend(loc="upper left", fontsize=9, ncol=5)

    axes[-1].set_xlabel("Hours from observation")
    save_figure(fig, "plot24_hybrid_component_examples.png")


def build_summary(bundles: list[DatasetBundle], skill: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for bundle in bundles:
        dataset_skill = skill[skill["dataset"] == bundle.name].set_index("component")
        summary[bundle.name] = {
            "source_dir": str(bundle.base_dir),
            "n_observations": int(len(bundle.diagnostics)),
            "median_observed_spacing_m": float(bundle.diagnostics["observed_spacing_m"].median()),
            "median_visible_spacing_m": float(bundle.diagnostics["spacing_visible_at_obs_m"].median()),
            "median_cl_spacing_m": float(bundle.diagnostics["spacing_cl_at_obs_m"].median()),
            "median_response_spacing_m": float(bundle.diagnostics["spacing_response_at_obs_m"].median()),
            "median_linear_spacing_m": float(bundle.diagnostics["spacing_linear_at_obs_m"].median()),
            "component_skill": dataset_skill.to_dict(orient="index"),
        }
    return summary


def main() -> None:
    setup_style()
    bundles = [load_bundle(name, path) for name, path in DATASETS.items()]
    skill = build_skill_table(bundles)

    plot_ranked_components(bundles)
    plot_skill_comparison(skill)
    plot_ratio_distributions(bundles)
    plot_example_timelines(bundles[0])

    summary = build_summary(bundles, skill)
    (PLOTS_DIR / "hybrid_decomposition_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
