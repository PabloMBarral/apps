"""Lo que comparten el doble tubo y el casco y tubos (Fase 12.1).

- **Las corrientes**: un fluido de CoolProp (propiedades a la temperatura media de la
  corriente, que se itera) o propiedades dadas, como los libros con los aceites
  (Incropera, tabla A.5; Çengel y Ghajar, tabla A-13): ρ, c_p, μ, k y, si se quiere la
  corrección de Sieder y Tate, μ a la temperatura de la pared.
- **El lado de los tubos**: las correlaciones de la Fase 8.1 (Incropera, §8.4 y §8.5;
  Çengel y Ghajar, §8-5 y §8-6) con Re = ρ·u·D/μ, y el factor de fricción de Darcy
  (64/Re laminar; Petukhov, turbulento).
- **El U por metro de tubo**: 1/(U·A) = 1/(h_i·A_i) + R''_f,i/A_i + ln(d_o/d_i)/(2π·k·L) +
  R''_f,o/A_o + 1/(h_o·A_o) (Çengel y Ghajar, §11-2), con L = 1 m; si d_o = d_i la pared
  es delgada y no cuenta (como en los ejemplos de los libros).

Todo en SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from core.fluids import fluid_limits, fluid_with_article, saturation_at_pressure
from core.heat_transfer.convection import (
    RE_LAMINAR_TUBE,
    RE_TURBULENT_TUBE,
    fluid_properties,
    friction_petukhov,
    nu_dittus_boelter,
    nu_gnielinski,
    nu_hausen,
    nu_sieder_tate,
    nu_tube_laminar,
)
from core.heat_transfer.exchangers import OverallUInputs, OverallUResult, UResistance

__all__ = [
    "TUBE_SIDE_CORRELATIONS",
    "DesignStream",
    "FluidProperties",
    "Side",
    "StreamProps",
    "TubeFlow",
    "check_single_phase",
    "darcy_f",
    "overall_u_per_length",
    "stream_name",
    "stream_props",
    "tube_flow",
    "wall_viscosity",
]

Side = Literal["hot", "cold"]

#: Las correlaciones del lado de los tubos (las de la 8.1) y "auto".
TUBE_SIDE_CORRELATIONS: dict[str, str] = {
    "auto": "Automática (laminar 3,66; turbulento, Gnielinski)",
    "tube_laminar": "Laminar desarrollada (Nu = 3,66)",
    "hausen": "Hausen (laminar con entrada térmica)",
    "dittus_boelter": "Dittus y Boelter",
    "gnielinski": "Gnielinski",
    "sieder_tate": "Sieder y Tate",
}


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


@dataclass(frozen=True)
class FluidProperties:
    """Propiedades dadas de un fluido (las de un enunciado), constantes."""

    rho_kg_per_m3: float
    cp_J_per_kgK: float
    mu_Pa_s: float
    k_W_per_mK: float
    mu_wall_Pa_s: float | None = None
    name: str = "fluido a elección"


@dataclass(frozen=True)
class DesignStream:
    """Una corriente: un fluido de CoolProp (``fluid``) o propiedades dadas."""

    T_in_K: float
    m_dot_kg_s: float
    fluid: str | None = None
    p_Pa: float = 101_325.0
    properties: FluidProperties | None = None
    R_f_m2K_per_W: float = 0.0


@dataclass(frozen=True)
class StreamProps:
    """Las propiedades de una corriente a una temperatura."""

    T_K: float
    rho_kg_per_m3: float
    cp_J_per_kgK: float
    mu_Pa_s: float
    k_W_per_mK: float

    @property
    def Pr(self) -> float:
        return self.mu_Pa_s * self.cp_J_per_kgK / self.k_W_per_mK


def stream_name(s: DesignStream) -> str:
    """«agua», «aceite»…: el nombre para los rótulos."""
    if s.properties is not None:
        return s.properties.name
    return fluid_with_article(s.fluid or "")[3:]


def check_stream(s: DesignStream, label: str) -> None:
    """Los datos de una corriente tienen sentido."""
    if s.T_in_K <= 0.0:
        raise ValueError(f"La temperatura del fluido {label} tiene que ser absoluta positiva.")
    if s.m_dot_kg_s <= 0.0:
        raise ValueError(f"El caudal del fluido {label} tiene que ser positivo.")
    if s.properties is None and s.fluid is None:
        raise ValueError(f"Al fluido {label} le faltan las propiedades (o elegí uno de CoolProp).")
    g = s.properties
    if g is not None and min(g.rho_kg_per_m3, g.cp_J_per_kgK, g.mu_Pa_s, g.k_W_per_mK) <= 0.0:
        raise ValueError(
            f"Las propiedades del fluido {label} (ρ, c_p, μ y k) tienen que ser positivas."
        )
    if s.R_f_m2K_per_W < 0.0:
        raise ValueError(f"El factor de ensuciamiento del fluido {label} no puede ser negativo.")


def stream_props(s: DesignStream, T_K: float, label: str) -> StreamProps:
    """Las propiedades de la corriente a T (CoolProp) o las dadas."""
    if s.properties is not None:
        g = s.properties
        return StreamProps(T_K, g.rho_kg_per_m3, g.cp_J_per_kgK, g.mu_Pa_s, g.k_W_per_mK)
    assert s.fluid is not None
    try:
        p = fluid_properties(s.fluid, T_K, s.p_Pa)
    except ValueError as exc:
        raise ValueError(f"El fluido {label} a {_degC(T_K)}: {exc}") from exc
    return StreamProps(T_K, p.rho_kg_per_m3, p.cp_J_per_kgK, p.mu_Pa_s, p.k_W_per_mK)


def wall_viscosity(s: DesignStream, T_wall_K: float) -> float | None:
    """μ a la temperatura de la pared (Sieder y Tate, Kern); None si no se conoce."""
    if s.properties is not None:
        return s.properties.mu_wall_Pa_s
    assert s.fluid is not None
    try:
        return fluid_properties(s.fluid, T_wall_K, s.p_Pa).mu_Pa_s
    except ValueError:
        return None


def check_single_phase(s: DesignStream, T_out_K: float, label: str) -> None:
    """El fluido no puede cruzar la saturación entre la entrada y la salida."""
    if s.fluid is None:
        return
    limits = fluid_limits(s.fluid)
    if not (limits.P_triple_Pa <= s.p_Pa < limits.P_crit_Pa):
        return
    T_sat = saturation_at_pressure(s.fluid, s.p_Pa).T_sat_K
    lo, hi = sorted((s.T_in_K, T_out_K))
    if lo <= T_sat <= hi:
        boils = s.T_in_K < T_out_K
        raise ValueError(
            f"{fluid_with_article(s.fluid).capitalize()} (fluido {label}) va de "
            f"{_degC(s.T_in_K)} a {_degC(T_out_K)}, y a {_num(s.p_Pa / 1e5)} bar "
            f"{'hierve' if boils else 'condensa'} a {_degC(T_sat)}: cambia de fase en el "
            f"intercambiador, y estas correlaciones son de una sola fase. "
            f"{'Subí' if boils else 'Bajá'} la presión."
        )


def darcy_f(Re: float) -> float:
    """Factor de fricción de Darcy de un tubo liso: 64/Re laminar y Petukhov turbulento
    (Incropera, §8.1); en la transición, Petukhov con Re ≥ 3000."""
    if Re < RE_LAMINAR_TUBE:
        return 64.0 / Re
    return friction_petukhov(max(Re, 3000.0))


@dataclass(frozen=True)
class TubeFlow:
    """El flujo por un conducto: Re, el régimen, Nu, h y el factor de fricción."""

    D_h_m: float
    area_m2: float
    velocity_m_s: float
    Re: float
    regime: str
    correlation: str
    Nu: float
    h_W_per_m2K: float
    f: float
    mu_ratio: float
    warnings: tuple[str, ...]


def regime(Re: float) -> str:
    if Re < RE_LAMINAR_TUBE:
        return "laminar"
    if Re < RE_TURBULENT_TUBE:
        return "de transición"
    return "turbulento"


def tube_flow(
    props: StreamProps,
    m_dot_kg_s: float,
    D_m: float,
    L_m: float,
    heating: bool,
    mu_wall_Pa_s: float | None,
    correlation: str = "auto",
) -> TubeFlow:
    """h dentro de un tubo circular con las correlaciones de la 8.1 (Incropera, §8.4 y
    §8.5): laminar desarrollada (3,66), Hausen, Dittus y Boelter (n = 0,4 si el fluido se
    calienta), Gnielinski o Sieder y Tate (con μ/μ_p). ``auto``: 3,66 en laminar (el
    intercambiador no tiene ni T ni flujo constantes) y Gnielinski en turbulento."""
    if correlation not in TUBE_SIDE_CORRELATIONS:
        raise ValueError(f"Correlación de tubo desconocida: {correlation!r}.")
    A = math.pi * D_m**2 / 4.0
    u = m_dot_kg_s / (props.rho_kg_per_m3 * A)
    Re = props.rho_kg_per_m3 * u * D_m / props.mu_Pa_s
    Pr = props.Pr
    mu_ratio = props.mu_Pa_s / mu_wall_Pa_s if mu_wall_Pa_s else 1.0
    key = correlation
    if key == "auto":
        key = "tube_laminar" if Re < RE_LAMINAR_TUBE else "gnielinski"
    warnings: list[str] = []
    if key == "tube_laminar":
        Nu = nu_tube_laminar(True)
    elif key == "hausen":
        Nu = nu_hausen(Re, Pr, D_m / L_m)
    elif key == "dittus_boelter":
        Nu = nu_dittus_boelter(Re, Pr, heating)
    elif key == "gnielinski":
        Nu = nu_gnielinski(max(Re, 3000.0), Pr)
    else:
        Nu = nu_sieder_tate(Re, Pr, mu_ratio)
    laminar_key = key in ("tube_laminar", "hausen")
    if laminar_key and Re >= RE_LAMINAR_TUBE:
        warnings.append(
            f"Re = {_num(Re)}: el flujo no es laminar y la correlación laminar no se aplica."
        )
    if not laminar_key and Re < RE_LAMINAR_TUBE:
        warnings.append(
            f"Re = {_num(Re)}: el flujo es laminar; la correlación turbulenta no se aplica."
        )
    elif key in ("dittus_boelter", "sieder_tate") and Re < RE_TURBULENT_TUBE:
        warnings.append(
            f"Re = {_num(Re)}: Dittus y Boelter y Sieder y Tate piden Re ≥ 10⁴; en la "
            "transición, Gnielinski es la mejor estimación."
        )
    elif key == "gnielinski" and Re < 3000.0:
        warnings.append(f"Re = {_num(Re)}: Gnielinski pide Re ≥ 3000 (se evalúa con 3000).")
    h = Nu * props.k_W_per_mK / D_m
    return TubeFlow(D_m, A, u, Re, regime(Re), key, Nu, h, darcy_f(Re), mu_ratio, tuple(warnings))


def overall_u_per_length(
    h_i: float,
    h_o: float,
    d_i: float,
    d_o: float,
    k_wall: float,
    R_f_i: float,
    R_f_o: float,
    inner: str = "adentro del tubo",
    outer: str = "afuera del tubo",
) -> OverallUResult:
    """Las resistencias de un metro de tubo (Çengel y Ghajar, §11-2). Con d_o = d_i la
    pared es delgada: no cuenta (A_i = A_o)."""
    A_i, A_o = math.pi * d_i, math.pi * d_o
    rows = [UResistance(f"Convección {inner}", 1.0 / (h_i * A_i), "convección")]
    if R_f_i > 0.0:
        rows.append(UResistance(f"Ensuciamiento {inner}", R_f_i / A_i, "ensuciamiento"))
    if d_o > d_i:
        rows.append(UResistance("Pared", math.log(d_o / d_i) / (2.0 * math.pi * k_wall), "pared"))
    if R_f_o > 0.0:
        rows.append(UResistance(f"Ensuciamiento {outer}", R_f_o / A_o, "ensuciamiento"))
    rows.append(UResistance(f"Convección {outer}", 1.0 / (h_o * A_o), "convección"))
    inputs = OverallUInputs(
        geometry="tube",
        h_i_W_per_m2K=h_i,
        h_o_W_per_m2K=h_o,
        k_wall_W_per_mK=k_wall,
        D_i_m=d_i,
        D_o_m=d_o,
        L_m=1.0,
        R_f_i_m2K_per_W=R_f_i,
        R_f_o_m2K_per_W=R_f_o,
    )
    return OverallUResult(inputs, tuple(rows), A_i, A_o)
