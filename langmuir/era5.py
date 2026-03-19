"""Backward-compatible shim for the renamed Open-Meteo client."""

from __future__ import annotations

import warnings

from .open_meteo_client import *  # noqa: F401,F403

warnings.warn(
    "langmuir.era5 is deprecated; import langmuir.open_meteo_client instead.",
    DeprecationWarning,
    stacklevel=2,
)
