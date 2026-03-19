from __future__ import annotations

import pandas as pd

from langmuir.open_meteo_client import build_era5_url, build_open_meteo_url
from langmuir.validation import filter_degraded_results, nonlinear_consistency_envelope


def test_filter_degraded_results_excludes_fallback_rows() -> None:
    results = pd.DataFrame(
        {
            "predicted_spacing_NL_m": [10.0, 20.0, float("nan"), 30.0],
            "fallback_triggered": [False, True, False, False],
        }
    )
    filtered = filter_degraded_results(results, "predicted_spacing_NL_m")
    assert list(filtered["predicted_spacing_NL_m"]) == [10.0, 30.0]


def test_open_meteo_client_keeps_legacy_url_builder() -> None:
    new_url = build_open_meteo_url(1.0, 2.0, "2024-01-01", "2024-01-02")
    legacy_url = build_era5_url(1.0, 2.0, "2024-01-01", "2024-01-02")
    assert legacy_url == new_url


def test_consistency_envelope_exposes_cl_and_response_spacing() -> None:
    envelope = nonlinear_consistency_envelope(depths=[9.0], use_lake_profile=True)
    depth_report = envelope[9.0]
    assert "spacing_CL" in depth_report
    assert "spacing_core" in depth_report
    assert "spacing_response" in depth_report
    assert "visibility_index" in depth_report
    assert len(depth_report["spacing_NL"]) == len(depth_report["spacing_response"])
