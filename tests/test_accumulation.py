from __future__ import annotations

import math

from langmuir.colony_accumulation import LangmuirModeState, advance_langmuir_state
from langmuir.params import LCParams


def test_buoyancy_state_evolves_with_hourly_forcing() -> None:
    params = LCParams(U10=6.0, depth=9.0, fetch=15000.0)
    dark = advance_langmuir_state(
        params,
        profile_name="uniform",
        use_lake_profile=False,
        previous_state=LangmuirModeState(),
        dt_hours=1.0,
        shortwave_radiation=0.0,
        temperature_c=18.0,
    )
    bright = advance_langmuir_state(
        params,
        profile_name="uniform",
        use_lake_profile=False,
        previous_state=dark["mode_state"],
        dt_hours=1.0,
        shortwave_radiation=500.0,
        temperature_c=24.0,
    )

    assert bright["buoyancy_Q"] != dark["buoyancy_Q"]
    assert bright["rho_cell"] != dark["rho_cell"]
    assert bright["v_float"] != dark["v_float"]
    assert bright["tau_acc_s"] >= 0.0


def test_hybrid_visible_spacing_sits_between_cl_and_response_scales() -> None:
    state = LangmuirModeState()
    result = {}
    for _ in range(12):
        result = advance_langmuir_state(
            LCParams(U10=6.0, depth=9.0, fetch=15000.0),
            previous_state=state,
            dt_hours=1.0,
            forcing_coherence=0.95,
            shortwave_radiation=350.0,
            temperature_c=22.0,
        )
        state = result["mode_state"]

    core_spacing = 2.0 * math.pi * 9.0 / result["selected_l"]
    assert result["spacing_response"] > core_spacing > result["spacing_cl"]
    assert abs(result["spacing_visible"] - core_spacing) < 1.0e-10
    assert abs(result["spacing_observable"] - result["visible_fraction"] * core_spacing) < 1.0e-10
    assert abs(result["spacing_nonlinear"] - result["spacing_visible"]) < 1.0e-10
    assert 0.0 < result["response_mix"] < 1.0
    assert result["response_bandwidth"] > 0.0
    assert 0.0 <= result["large_scale_fraction"] <= 1.0


def test_spacing_lifecycle_grows_then_decays_with_forcing() -> None:
    state = LangmuirModeState()
    growth = []
    for _ in range(6):
        result = advance_langmuir_state(
            LCParams(U10=6.0, depth=9.0, fetch=15000.0),
            previous_state=state,
            dt_hours=0.25,
            forcing_coherence=0.95,
            shortwave_radiation=350.0,
            temperature_c=22.0,
        )
        growth.append(result)
        state = result["mode_state"]

    assert growth[0]["visible_fraction"] < growth[-1]["visible_fraction"]
    assert growth[0]["coarsening_index"] < growth[-1]["coarsening_index"]
    assert growth[0]["spacing_observable"] < growth[-1]["spacing_observable"]

    decay = []
    for _ in range(8):
        result = advance_langmuir_state(
            LCParams(U10=1.0, depth=9.0, fetch=15000.0),
            previous_state=state,
            dt_hours=0.25,
            forcing_coherence=0.10,
            shortwave_radiation=0.0,
            temperature_c=18.0,
        )
        decay.append(result)
        state = result["mode_state"]

    assert math.isfinite(decay[0]["spacing_nonlinear"])
    assert decay[0]["spacing_observable"] > decay[-1]["spacing_observable"]
    assert decay[0]["visible_fraction"] > decay[-1]["visible_fraction"]
