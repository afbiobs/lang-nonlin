from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class LCParams:
    U10: float
    depth: float = 9.0
    fetch: float = 15000.0
    rho_w: float = 998.2
    nu: float = 1.0e-6
    rho_air: float = 1.225
    g: float = 9.81
    kappa: float = 0.41
    z0: float = 5.0e-3
    N_cheb: int = 64
    sweep_points: int = 120
    bottom_bc: str = "free_slip"
    u_star: float = field(init=False)
    u_stokes_surface: float = field(init=False)
    langmuir_number: float = field(init=False)
    drag_coefficient: float = field(init=False)
    wind_stress: float = field(init=False)
    peak_frequency: float = field(init=False)
    significant_wave_height: float = field(init=False)
    peak_wavenumber: float = field(init=False)
    peak_angular_frequency: float = field(init=False)

    def __post_init__(self) -> None:
        from .wind_forcing import populate_forcing

        populate_forcing(self)

    def to_dict(self) -> dict:
        return asdict(self)
