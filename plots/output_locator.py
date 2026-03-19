"""Locate the most recent timeline validation outputs for each dataset."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"


def latest_timeline_output(dataset: str) -> Path:
    candidates = sorted(
        path
        for path in OUTPUTS_DIR.glob(f"*_{dataset}_validation_timeline_v*")
        if path.is_dir() and (path / "observation_diagnostics.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No timeline validation output found for dataset={dataset!r}")
    return candidates[-1]
