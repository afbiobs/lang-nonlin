"""Calibration utilities for duration-aware Langmuir timeline dynamics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .colony_accumulation import LangmuirDynamicsConfig
from .validation import validate_nonlinear


@dataclass(frozen=True)
class TuningObjectiveWeights:
    """Weights and floors for the manual-observation tuning objective."""

    rmse_weight: float = 1.0
    bias_weight: float = 0.25
    quantile_range_weight: float = 0.35
    variability_weight: float = 3.0
    duration_sensitivity_weight: float = 35.0
    setup_sensitivity_weight: float = 20.0
    low_quantile_weight: float = 0.4
    high_quantile_weight: float = 0.4
    min_timeline_std_m: float = 12.0
    min_duration_corr: float = 0.20
    min_setup_corr: float = 0.20

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return float("nan")
    return float(x[valid].corr(y[valid]))


def evaluate_manual_config(
    dynamics: LangmuirDynamicsConfig,
    *,
    dataset_path: str = "data/observations_minimal.csv",
    cache_dir: str = "data/era5_cache",
    weights: TuningObjectiveWeights | None = None,
) -> dict:
    """Evaluate one dynamics configuration against manual observations."""
    weights = weights or TuningObjectiveWeights()
    result = validate_nonlinear(
        dataset_path=dataset_path,
        spacing_column="manual_spacing_m",
        cache_dir=cache_dir,
        output_dir=None,
        use_lake_profile=True,
        skip_download=True,
        timeline_hours_before=48,
        timeline_hours_after=48,
        observation_hour_utc=12,
        dynamics=dynamics,
    )
    diagnostics = result.observation_diagnostics.copy()
    valid = diagnostics.dropna(subset=["observed_spacing_m", "spacing_at_obs_m"]).copy()
    if len(valid) == 0:
        raise ValueError("No valid observation diagnostics were produced for tuning.")

    observed = valid["observed_spacing_m"]
    predicted = valid["spacing_at_obs_m"]
    obs_q10 = float(observed.quantile(0.10))
    obs_q90 = float(observed.quantile(0.90))
    pred_q10 = float(predicted.quantile(0.10))
    pred_q90 = float(predicted.quantile(0.90))
    mean_timeline_std = float(valid["spacing_std_prev_48h_m"].mean())
    duration_corr = (
        _safe_corr(predicted, valid["coherent_run_hours_at_obs"])
        if "coherent_run_hours_at_obs" in valid.columns
        else float("nan")
    )
    duration_mean_corr = (
        _safe_corr(predicted, valid["coherent_run_hours_mean_prev_48h"])
        if "coherent_run_hours_mean_prev_48h" in valid.columns
        else float("nan")
    )
    setup_corr = _safe_corr(predicted, valid["setup_at_obs"]) if "setup_at_obs" in valid.columns else float("nan")

    score_terms = {
        "rmse": weights.rmse_weight * float(result.metrics.get("rmse_timeline_obs_m", float("nan"))),
        "bias": weights.bias_weight * abs(float(result.metrics.get("bias_timeline_obs_m", 0.0))),
        "range": weights.quantile_range_weight * (abs(pred_q10 - obs_q10) + abs(pred_q90 - obs_q90)),
        "low_quantile": weights.low_quantile_weight * abs(pred_q10 - obs_q10),
        "high_quantile": weights.high_quantile_weight * abs(pred_q90 - obs_q90),
        "variability": weights.variability_weight * max(weights.min_timeline_std_m - mean_timeline_std, 0.0),
        "duration": weights.duration_sensitivity_weight * max(weights.min_duration_corr - duration_corr, 0.0),
        "setup": weights.setup_sensitivity_weight * max(weights.min_setup_corr - setup_corr, 0.0),
    }
    score = float(sum(score_terms.values()))

    return {
        "score": score,
        "score_terms": score_terms,
        "metrics": {
            **result.metrics,
            "observed_q10_m": obs_q10,
            "observed_q90_m": obs_q90,
            "predicted_q10_m": pred_q10,
            "predicted_q90_m": pred_q90,
            "mean_timeline_std_m": mean_timeline_std,
            "duration_corr_predicted_vs_coherent_run_at_obs": duration_corr,
            "duration_corr_predicted_vs_coherent_run_mean_prev_48h": duration_mean_corr,
            "setup_corr_predicted_vs_setup_at_obs": setup_corr,
        },
        "dynamics": dynamics.to_dict(),
    }


def sample_dynamics_config(rng: np.random.Generator) -> LangmuirDynamicsConfig:
    """Sample a plausible duration-aware dynamics configuration."""
    return LangmuirDynamicsConfig(
        tau_relax_alpha=float(rng.uniform(0.03, 0.18)),
        tau_decay_alpha=float(rng.uniform(0.05, 0.30)),
        tau_coherence_alpha=float(rng.uniform(0.02, 0.15)),
        merge_min_age_alpha=float(rng.uniform(0.01, 0.10)),
        merge_supercriticality_threshold=float(rng.uniform(0.05, 0.6)),
        merge_step_factor=float(rng.uniform(0.72, 0.96)),
        coherence_threshold=float(rng.uniform(0.35, 0.9)),
    )


def search_manual_dynamics(
    *,
    dataset_path: str = "data/observations_minimal.csv",
    cache_dir: str = "data/era5_cache",
    n_trials: int = 24,
    seed: int = 0,
    weights: TuningObjectiveWeights | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run a random-search calibration of duration-aware dynamics."""
    weights = weights or TuningObjectiveWeights()
    rng = np.random.default_rng(seed)

    reports: list[dict] = []
    trial_specs = [("default", LangmuirDynamicsConfig())]
    trial_specs.extend((f"sample_{idx:03d}", sample_dynamics_config(rng)) for idx in range(n_trials))

    for trial_name, dynamics in trial_specs:
        report = evaluate_manual_config(
            dynamics,
            dataset_path=dataset_path,
            cache_dir=cache_dir,
            weights=weights,
        )
        report["trial_name"] = trial_name
        reports.append(report)
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            partial_rows = []
            for partial in reports:
                partial_rows.append(
                    {
                        "trial_name": partial["trial_name"],
                        "score": partial["score"],
                        **{f"term_{k}": v for k, v in partial["score_terms"].items()},
                        **{f"metric_{k}": v for k, v in partial["metrics"].items()},
                        **{f"cfg_{k}": v for k, v in partial["dynamics"].items()},
                    }
                )
            pd.DataFrame(partial_rows).sort_values("score").to_csv(out / "leaderboard.partial.csv", index=False)

    leaderboard_rows = []
    for report in reports:
        row = {
            "trial_name": report["trial_name"],
            "score": report["score"],
            **{f"term_{k}": v for k, v in report["score_terms"].items()},
            **{f"metric_{k}": v for k, v in report["metrics"].items()},
            **{f"cfg_{k}": v for k, v in report["dynamics"].items()},
        }
        leaderboard_rows.append(row)
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("score").reset_index(drop=True)
    best_trial_name = str(leaderboard.iloc[0]["trial_name"])
    best_report = next(report for report in reports if report["trial_name"] == best_trial_name)

    payload = {
        "best": best_report,
        "leaderboard": leaderboard,
        "weights": weights.to_dict(),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(out / "leaderboard.csv", index=False)
        with (out / "best_config.json").open("w") as fh:
            json.dump(best_report, fh, indent=2, default=str)
        with (out / "weights.json").open("w") as fh:
            json.dump(weights.to_dict(), fh, indent=2, default=str)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune duration-aware Langmuir dynamics against manual observations.")
    parser.add_argument("--dataset-path", default="data/observations_minimal.csv")
    parser.add_argument("--cache-dir", default="data/era5_cache")
    parser.add_argument("--trials", type=int, default=8, help="Number of random samples in addition to the default config.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/manual_tuning_search")
    args = parser.parse_args()

    summary = search_manual_dynamics(
        dataset_path=args.dataset_path,
        cache_dir=args.cache_dir,
        n_trials=args.trials,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    best = summary["best"]
    print(json.dumps(
        {
            "trial_name": best["trial_name"],
            "score": best["score"],
            "score_terms": best["score_terms"],
            "metrics": best["metrics"],
            "dynamics": best["dynamics"],
        },
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
