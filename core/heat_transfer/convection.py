"""Convección: correlaciones del número de Nusselt (Fase 8.1).

Cengel y Ghajar (2015), caps. 7 a 9; Incropera et al. (2007), caps. 7 a 9.

Tres situaciones:

- **forzada externa**: placa plana (laminar, mixta o turbulenta), cilindro en
  flujo cruzado (Churchill y Bernstein, 1977; Hilpert) y esfera (Whitaker,
  1972; Ranz y Marshall, 1952);
- **forzada interna** en un tubo circular: laminar desarrollada (3,66 o 4,36),
  con entrada térmica (Hausen, 1943) y turbulenta (Dittus y Boelter, 1930;
  Gnielinski, 1976; Sieder y Tate, 1936), con T de pared o flujo de calor
  constante; la temperatura media del fluido se itera;
- **natural**: placa vertical (Churchill y Chu, 1975a; McAdams), placa
  horizontal (Lloyd y Moran, 1974; McAdams, 1954), cilindro horizontal
  (Churchill y Chu, 1975b; Morgan, 1975) y esfera (Churchill, 1983).

Cada correlación es una función con su cita y su rango en el docstring, y
``CORRELATIONS`` las junta con el LaTeX (la misma regla que las del PCS de la
Fase 6). Las propiedades salen de CoolProp (Bell et al., 2014) a la
temperatura de película T_f = (T_s + T∞)/2, salvo en la esfera de Whitaker
(a T∞) y en un tubo (a la temperatura media del fluido). β también sale de
CoolProp (≈ 1/T en un gas).

Todo en SI. Q̇ = h·A·(T_s − T∞): positivo si la superficie pierde calor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from CoolProp.CoolProp import PropsSI

from core.fluids import (
    FLUID_NAMES_ES,
    SUPPORTED_FLUIDS,
    fluid_state_from_pair,
    fluid_with_article,
    saturation_at_pressure,
)
from core.units_system import UnitSystem, format_quantity

__all__ = [
    "CORRELATIONS",
    "EXTERNAL_EXAMPLES",
    "EXTERNAL_GEOMETRIES",
    "G_STANDARD",
    "INTERNAL_CONDITIONS",
    "INTERNAL_EXAMPLES",
    "NATURAL_EXAMPLES",
    "NATURAL_GEOMETRIES",
    "RANGE_TOLERANCE",
    "RE_BLUFF_TURBULENT",
    "RE_CRITICAL_PLATE",
    "RE_LAMINAR_TUBE",
    "RE_TURBULENT_TUBE",
    "TUBE_CORRELATIONS",
    "Alternative",
    "ConvectionExample",
    "ConvectionResult",
    "Correlation",
    "ExternalFlowInputs",
    "ExternalGeometry",
    "InternalCondition",
    "InternalFlowInputs",
    "InternalFlowResult",
    "NaturalConvectionInputs",
    "NaturalGeometry",
    "Properties",
    "convection_notes",
    "convection_fluids",
    "convection_to_dict",
    "dimensionless_groups",
    "fluid_properties",
    "friction_petukhov",
    "internal_notes",
    "internal_to_dict",
    "nu_churchill_bernstein",
    "nu_curve",
    "nu_dittus_boelter",
    "nu_gnielinski",
    "nu_hausen",
    "nu_hilpert",
    "nu_horizontal_cylinder_churchill_chu",
    "nu_horizontal_lower",
    "nu_horizontal_upper",
    "nu_morgan",
    "nu_plate_laminar",
    "nu_plate_mixed",
    "nu_plate_turbulent",
    "nu_ranz_marshall",
    "nu_sieder_tate",
    "nu_sphere_churchill",
    "nu_tube_laminar",
    "nu_vertical_churchill_chu",
    "nu_vertical_mcadams",
    "nu_whitaker",
    "solve_external",
    "solve_internal",
    "solve_natural",
    "tube_profile",
]

#: Aceleración de la gravedad estándar (m/s²).
G_STANDARD = 9.80665
#: Reynolds de la transición en una placa plana (Incropera, §7.2).
RE_CRITICAL_PLATE = 5.0e5
#: Reynolds de la transición en un tubo (Incropera, §8.1).
RE_LAMINAR_TUBE = 2300.0
#: Reynolds a partir del cual el flujo en un tubo es plenamente turbulento.
RE_TURBULENT_TUBE = 1.0e4

ExternalGeometry = Literal["plate", "cylinder", "sphere"]
NaturalGeometry = Literal[
    "vertical_plate",
    "horizontal_plate_up",
    "horizontal_plate_down",
    "horizontal_cylinder",
    "sphere",
]
InternalCondition = Literal["constant_T", "constant_flux"]

EXTERNAL_GEOMETRIES: dict[ExternalGeometry, str] = {
    "plate": "Placa plana (flujo paralelo)",
    "cylinder": "Cilindro (flujo cruzado)",
    "sphere": "Esfera",
}
NATURAL_GEOMETRIES: dict[NaturalGeometry, str] = {
    "vertical_plate": "Placa vertical",
    "horizontal_plate_up": "Placa horizontal: la cara de arriba",
    "horizontal_plate_down": "Placa horizontal: la cara de abajo",
    "horizontal_cylinder": "Cilindro horizontal",
    "sphere": "Esfera",
}
INTERNAL_CONDITIONS: dict[InternalCondition, str] = {
    "constant_T": "Temperatura de pared constante",
    "constant_flux": "Flujo de calor constante",
}


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _pct(x: float) -> str:
    """Un porcentaje legible: 1,3 o 41 (sin notación científica)."""
    return (f"{x:.0f}" if abs(x) >= 10.0 else f"{x:.1f}").replace(".", ",")


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


def _el(fluid: str) -> str:
    """«el agua», «el R-134a»."""
    return fluid_with_article(fluid)


def _del(fluid: str) -> str:
    """«del agua», «del R-134a»."""
    return "del " + fluid_with_article(fluid)[3:]


# ---------------------------------------------------------------------
# Propiedades
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Properties:
    """Las propiedades del fluido a una temperatura (CoolProp)."""

    fluid: str
    T_K: float
    p_Pa: float
    rho_kg_per_m3: float
    mu_Pa_s: float
    k_W_per_mK: float
    cp_J_per_kgK: float
    beta_per_K: float
    is_liquid: bool

    @property
    def Pr(self) -> float:
        return self.mu_Pa_s * self.cp_J_per_kgK / self.k_W_per_mK

    @property
    def nu_m2_per_s(self) -> float:
        return self.mu_Pa_s / self.rho_kg_per_m3

    @property
    def alpha_m2_per_s(self) -> float:
        return self.k_W_per_mK / (self.rho_kg_per_m3 * self.cp_J_per_kgK)


def fluid_properties(fluid: str, T_K: float, p_Pa: float) -> Properties:
    """ρ, μ, k, c_p y β de CoolProp a (T, p).

    Raises
    ------
    ValueError
        Si CoolProp no tiene las propiedades de transporte del fluido o el
        estado cae en la campana.
    """
    state = fluid_state_from_pair(fluid, "TP", t=T_K, p=p_Pa)
    if state.region == "saturated_mixture":
        raise ValueError(
            f"A {_num(p_Pa / 1e5)} bar y {_degC(T_K)} {_el(fluid)} está cambiando "
            "de fase: estas correlaciones son de una sola fase."
        )
    mu, k, cp = state.viscosity_Pa_s, state.conductivity_W_per_m_K, state.cp_J_per_kg_K
    if mu is None or k is None or cp is None:
        raise ValueError(
            f"CoolProp no tiene la viscosidad o la conductividad {_del(fluid)}: elegí otro fluido."
        )
    beta = float(PropsSI("isobaric_expansion_coefficient", "T", T_K, "P", p_Pa, fluid))
    liquid = state.region in ("compressed_liquid", "saturated_liquid")
    return Properties(fluid, T_K, p_Pa, state.rho_kg_per_m3, mu, k, cp, beta, liquid)


@lru_cache(maxsize=1)
def convection_fluids() -> tuple[str, ...]:
    """Los fluidos del proyecto con viscosidad y conductividad en CoolProp."""
    out = []
    for fluid in SUPPORTED_FLUIDS:
        try:
            PropsSI("V", "T", 300.0, "P", 101_325.0, fluid)
            PropsSI("L", "T", 300.0, "P", 101_325.0, fluid)
        except ValueError:
            continue
        out.append(fluid)
    return tuple(out)


def _check_wall_phase(fluid: str, T_s: float, T_bulk: float, p: float) -> str | None:
    """Aviso si la pared está del otro lado de la saturación que el fluido (ebullición o
    condensación en la pared)."""
    try:
        sat = saturation_at_pressure(fluid, p)
    except ValueError:
        return None  # supercrítico o fuera de la tabla de saturación: no hay cambio de fase
    T_sat = sat.T_sat_K
    if T_bulk < T_sat < T_s:
        return (
            f"La pared está a {_degC(T_s)} y a esa presión {_el(fluid)} hierve a "
            f"{_degC(T_sat)}: puede haber ebullición en la pared, que estas correlaciones (de "
            "una sola fase) no tienen en cuenta."
        )
    if T_s < T_sat < T_bulk:
        return (
            f"La pared está a {_degC(T_s)}, por debajo de la saturación {_del(fluid)} a esa "
            f"presión ({_degC(T_sat)}): puede condensar en la pared."
        )
    return None


# ---------------------------------------------------------------------
# Correlaciones (una función por correlación)
# ---------------------------------------------------------------------


def nu_plate_laminar(Re: float, Pr: float) -> float:
    """Placa plana, laminar: Nu = 0,664·Re^½·Pr^⅓ (Pohlhausen; Incropera, §7.2.1).

    Rango: Re_L < 5·10⁵ y Pr ≥ 0,6. Propiedades a T_f.
    """
    return 0.664 * Re**0.5 * Pr ** (1.0 / 3.0)


def nu_plate_mixed(Re: float, Pr: float) -> float:
    """Placa plana, laminar y después turbulenta (Re_cr = 5·10⁵):
    Nu = (0,037·Re^0,8 − 871)·Pr^⅓ (Incropera, §7.2.3; Cengel y Ghajar, §7-2).

    Rango: 5·10⁵ ≤ Re_L ≤ 10⁸ y 0,6 ≤ Pr ≤ 60.
    """
    return (0.037 * Re**0.8 - 871.0) * Pr ** (1.0 / 3.0)


def nu_plate_turbulent(Re: float, Pr: float) -> float:
    """Placa plana turbulenta desde el borde (con un alambre o rugosidad que la dispara):
    Nu = 0,037·Re^0,8·Pr^⅓ (Incropera, §7.2.3; Cengel y Ghajar, §7-2).

    Rango: 5·10⁵ ≤ Re_L ≤ 10⁷ y 0,6 ≤ Pr ≤ 60.
    """
    return 0.037 * Re**0.8 * Pr ** (1.0 / 3.0)


def nu_churchill_bernstein(Re: float, Pr: float) -> float:
    """Cilindro en flujo cruzado (Churchill y Bernstein, 1977, J. Heat Transfer 99:300):
    Nu = 0,3 + 0,62·Re^½·Pr^⅓/[1 + (0,4/Pr)^⅔]^¼·[1 + (Re/282 000)^⅝]^⅘.

    Rango: Re·Pr > 0,2; propiedades a T_f (Incropera, §7.4).
    """
    return 0.3 + (
        0.62
        * Re**0.5
        * Pr ** (1.0 / 3.0)
        / (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25
        * (1.0 + (Re / 282_000.0) ** 0.625) ** 0.8
    )


#: Constantes de Hilpert (1933) para el cilindro: (Re mínimo, Re máximo, C, m)
#: (Incropera, tabla 7.2).
_HILPERT = (
    (0.4, 4.0, 0.989, 0.330),
    (4.0, 40.0, 0.911, 0.385),
    (40.0, 4000.0, 0.683, 0.466),
    (4000.0, 40_000.0, 0.193, 0.618),
    (40_000.0, 400_000.0, 0.027, 0.805),
)


def nu_hilpert(Re: float, Pr: float) -> float:
    """Cilindro en flujo cruzado: Nu = C·Re^m·Pr^⅓ con C y m de Hilpert (1933) por tramo de
    Re (Incropera, §7.4 y tabla 7.2).

    Rango: 0,4 ≤ Re ≤ 4·10⁵ y Pr ≥ 0,7.
    """
    C, m = _HILPERT[-1][2:]
    for _lo, hi, C_k, m_k in _HILPERT:
        if Re < hi:
            C, m = C_k, m_k
            break
    return C * Re**m * Pr ** (1.0 / 3.0)


def nu_whitaker(Re: float, Pr: float, mu_ratio: float) -> float:
    """Esfera (Whitaker, 1972, AIChE J. 18:361):
    Nu = 2 + (0,4·Re^½ + 0,06·Re^⅔)·Pr^0,4·(μ∞/μ_s)^¼.

    Rango: 3,5 ≤ Re ≤ 7,6·10⁴, 0,71 ≤ Pr ≤ 380 y 1 ≤ μ∞/μ_s ≤ 3,2; propiedades a
    T∞ y μ_s a T_s (Incropera, §7.5). En un gas que se calienta μ∞/μ_s < 1 y se usa
    igual (Incropera, ejemplo 7.7).
    """
    return 2.0 + (0.4 * Re**0.5 + 0.06 * Re ** (2.0 / 3.0)) * Pr**0.4 * mu_ratio**0.25


def nu_ranz_marshall(Re: float, Pr: float) -> float:
    """Esfera, gotas (Ranz y Marshall, 1952, Chem. Eng. Prog. 48:141):
    Nu = 2 + 0,6·Re^½·Pr^⅓ (Incropera, §7.5). Propiedades a T_f.
    """
    return 2.0 + 0.6 * Re**0.5 * Pr ** (1.0 / 3.0)


def nu_tube_laminar(constant_T: bool) -> float:
    """Tubo, laminar y desarrollado: Nu = 3,66 (T de pared constante) o 4,36 (flujo de
    calor constante) (Incropera, §8.4.1)."""
    return 3.66 if constant_T else 48.0 / 11.0


def nu_hausen(Re: float, Pr: float, D_over_L: float) -> float:
    """Tubo, laminar con entrada térmica (Hausen, 1943; Incropera, §8.4.2):
    Nu = 3,66 + 0,0668·Gz/(1 + 0,04·Gz^⅔), Gz = (D/L)·Re·Pr.

    Rango: Re < 2300, T de pared constante; perfil de velocidad ya desarrollado.
    """
    Gz = D_over_L * Re * Pr
    return 3.66 + 0.0668 * Gz / (1.0 + 0.04 * Gz ** (2.0 / 3.0))


def nu_dittus_boelter(Re: float, Pr: float, heating: bool) -> float:
    """Tubo, turbulento (Dittus y Boelter, 1930; Incropera, §8.5):
    Nu = 0,023·Re^0,8·Pr^n, con n = 0,4 si el fluido se calienta y 0,3 si se enfría.

    Rango: Re ≥ 10⁴, 0,6 ≤ Pr ≤ 160 y L/D ≥ 10. Error típico ±25 %.
    """
    return 0.023 * Re**0.8 * Pr ** (0.4 if heating else 0.3)


def friction_petukhov(Re: float) -> float:
    """Factor de fricción de Darcy de un tubo liso (Petukhov, 1970; Incropera, §8.1):
    f = (0,790·ln Re − 1,64)⁻², para 3000 ≤ Re ≤ 5·10⁶."""
    return (0.790 * math.log(Re) - 1.64) ** -2


def nu_gnielinski(Re: float, Pr: float) -> float:
    """Tubo, turbulento y de transición (Gnielinski, 1976, Int. Chem. Eng. 16:359):
    Nu = (f/8)·(Re − 1000)·Pr / [1 + 12,7·(f/8)^½·(Pr^⅔ − 1)], con f de Petukhov.

    Rango: 3000 ≤ Re ≤ 5·10⁶ y 0,5 ≤ Pr ≤ 2000. Error típico ±10 % (Incropera, §8.5).
    """
    f = friction_petukhov(Re)
    return (
        (f / 8.0) * (Re - 1000.0) * Pr / (1.0 + 12.7 * (f / 8.0) ** 0.5 * (Pr ** (2.0 / 3.0) - 1.0))
    )


def nu_sieder_tate(Re: float, Pr: float, mu_ratio: float) -> float:
    """Tubo, turbulento con propiedades muy variables (Sieder y Tate, 1936):
    Nu = 0,027·Re^0,8·Pr^⅓·(μ/μ_s)^0,14 (Incropera, §8.5).

    Rango: Re ≥ 10⁴, 0,7 ≤ Pr ≤ 16 700 y L/D ≥ 10; μ_s a la temperatura de la pared.
    """
    return 0.027 * Re**0.8 * Pr ** (1.0 / 3.0) * mu_ratio**0.14


def nu_vertical_churchill_chu(Ra: float, Pr: float) -> float:
    """Placa vertical, todo Ra (Churchill y Chu, 1975a, Int. J. Heat Mass Transfer 18:1323):
    Nu = {0,825 + 0,387·Ra^⅙/[1 + (0,492/Pr)^{9/16}]^{8/27}}² (Incropera, §9.6.1)."""
    return (
        0.825 + 0.387 * Ra ** (1.0 / 6.0) / (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    ) ** 2


def nu_vertical_mcadams(Ra: float) -> float:
    """Placa vertical, la forma simple Nu = C·Ra^n (McAdams, 1954; Cengel y Ghajar, tabla 9-1):
    0,59·Ra^¼ si 10⁴ ≤ Ra ≤ 10⁹ y 0,10·Ra^⅓ si 10⁹ < Ra ≤ 10¹³."""
    return 0.59 * Ra**0.25 if Ra <= 1.0e9 else 0.10 * Ra ** (1.0 / 3.0)


def nu_horizontal_upper(Ra: float) -> float:
    """Cara de arriba de una placa caliente (o de abajo de una fría), L = A/P (Lloyd y Moran,
    1974; Incropera, §9.6.2): 0,54·Ra^¼ si 10⁴ ≤ Ra ≤ 10⁷ y 0,15·Ra^⅓ si
    10⁷ ≤ Ra ≤ 10¹¹."""
    return 0.54 * Ra**0.25 if Ra <= 1.0e7 else 0.15 * Ra ** (1.0 / 3.0)


def nu_horizontal_lower(Ra: float) -> float:
    """Cara de abajo de una placa caliente (o de arriba de una fría), L = A/P (McAdams, 1954;
    Cengel y Ghajar, tabla 9-1): Nu = 0,27·Ra^¼ si 10⁵ ≤ Ra ≤ 10¹¹."""
    return 0.27 * Ra**0.25


def nu_horizontal_cylinder_churchill_chu(Ra: float, Pr: float) -> float:
    """Cilindro horizontal (Churchill y Chu, 1975b, Int. J. Heat Mass Transfer 18:1049):
    Nu = {0,60 + 0,387·Ra^⅙/[1 + (0,559/Pr)^{9/16}]^{8/27}}², Ra ≤ 10¹² (Incropera, §9.6.3)."""
    return (
        0.60 + 0.387 * Ra ** (1.0 / 6.0) / (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    ) ** 2


#: Constantes de Morgan (1975) para el cilindro horizontal: (Ra máximo, C, n)
#: (Incropera, tabla 9.1).
_MORGAN = (
    (1.0e-2, 0.675, 0.058),
    (1.0e2, 1.02, 0.148),
    (1.0e4, 0.850, 0.188),
    (1.0e7, 0.480, 0.250),
    (1.0e12, 0.125, 0.333),
)


def nu_morgan(Ra: float) -> float:
    """Cilindro horizontal: Nu = C·Ra^n con C y n de Morgan (1975) por tramo de Ra
    (Incropera, §9.6.3 y tabla 9.1). Rango: 10⁻¹⁰ ≤ Ra ≤ 10¹²."""
    C, n = _MORGAN[-1][1:]
    for hi, C_k, n_k in _MORGAN:
        if Ra <= hi:
            C, n = C_k, n_k
            break
    return C * Ra**n


def nu_sphere_churchill(Ra: float, Pr: float) -> float:
    """Esfera en convección natural (Churchill, 1983; Incropera, §9.6.4):
    Nu = 2 + 0,589·Ra^¼/[1 + (0,469/Pr)^{9/16}]^{4/9}. Rango: Ra ≤ 10¹¹ y Pr ≥ 0,7."""
    return 2.0 + 0.589 * Ra**0.25 / (1.0 + (0.469 / Pr) ** (9.0 / 16.0)) ** (4.0 / 9.0)


@dataclass(frozen=True)
class Correlation:
    """Una correlación: cita, LaTeX, rango y cómo se evalúa con los adimensionales."""

    key: str
    name: str
    latex: str
    validity: str
    func: Callable[[dict[str, float]], float]
    in_range: Callable[[dict[str, float]], bool]


#: Tolerancia de los rangos de validez: las propiedades de CoolProp difieren de las
#: tablas en algunos % (el Pr del aire a 20 °C es 0,708 contra 0,7309 de la A-15).
RANGE_TOLERANCE = 0.01


def _between(x: float, lo: float, hi: float) -> bool:
    return lo * (1.0 - RANGE_TOLERANCE) <= x <= hi * (1.0 + RANGE_TOLERANCE)


CORRELATIONS: dict[str, Correlation] = {
    c.key: c
    for c in (
        Correlation(
            "plate_laminar",
            "Laminar (Pohlhausen)",
            r"Nu = 0.664\,Re^{1/2}\,Pr^{1/3}",
            "Re < 5·10⁵; Pr ≥ 0,6",
            lambda d: nu_plate_laminar(d["Re"], d["Pr"]),
            lambda d: d["Re"] < RE_CRITICAL_PLATE and _between(d["Pr"], 0.6, math.inf),
        ),
        Correlation(
            "plate_mixed",
            "Laminar y turbulenta (mixta)",
            r"Nu = \left(0.037\,Re^{0.8} - 871\right) Pr^{1/3}",
            "5·10⁵ ≤ Re ≤ 10⁸; 0,6 ≤ Pr ≤ 60",
            lambda d: nu_plate_mixed(d["Re"], d["Pr"]),
            lambda d: _between(d["Re"], RE_CRITICAL_PLATE, 1e8) and _between(d["Pr"], 0.6, 60),
        ),
        Correlation(
            "plate_turbulent",
            "Turbulenta desde el borde",
            r"Nu = 0.037\,Re^{0.8}\,Pr^{1/3}",
            "5·10⁵ ≤ Re ≤ 10⁷; 0,6 ≤ Pr ≤ 60",
            lambda d: nu_plate_turbulent(d["Re"], d["Pr"]),
            lambda d: _between(d["Re"], RE_CRITICAL_PLATE, 1e7) and _between(d["Pr"], 0.6, 60),
        ),
        Correlation(
            "churchill_bernstein",
            "Churchill y Bernstein (1977)",
            r"\begin{aligned}Nu &= 0.3 + \frac{0.62\,Re^{1/2}Pr^{1/3}}"
            r"{\left[1 + (0.4/Pr)^{2/3}\right]^{1/4}} \\ &\quad \cdot "
            r"\left[1 + \left(\frac{Re}{282\,000}\right)^{5/8}\right]^{4/5}\end{aligned}",
            "Re·Pr > 0,2",
            lambda d: nu_churchill_bernstein(d["Re"], d["Pr"]),
            lambda d: d["Re"] * d["Pr"] > 0.2,
        ),
        Correlation(
            "hilpert",
            "Hilpert (1933)",
            r"Nu = C\,Re^{m}\,Pr^{1/3}",
            "0,4 ≤ Re ≤ 4·10⁵; Pr ≥ 0,7",
            lambda d: nu_hilpert(d["Re"], d["Pr"]),
            lambda d: _between(d["Re"], 0.4, 4e5) and _between(d["Pr"], 0.7, math.inf),
        ),
        Correlation(
            "whitaker",
            "Whitaker (1972)",
            r"\begin{aligned}Nu &= 2 + \left(0.4\,Re^{1/2} + 0.06\,Re^{2/3}\right) \\ &\quad "
            r"\cdot Pr^{0.4}\left(\frac{\mu_\infty}{\mu_s}\right)^{1/4}\end{aligned}",
            "3,5 ≤ Re ≤ 7,6·10⁴; 0,71 ≤ Pr ≤ 380",
            lambda d: nu_whitaker(d["Re"], d["Pr"], d["mu_ratio"]),
            lambda d: _between(d["Re"], 3.5, 7.6e4) and _between(d["Pr"], 0.71, 380),
        ),
        Correlation(
            "ranz_marshall",
            "Ranz y Marshall (1952)",
            r"Nu = 2 + 0.6\,Re^{1/2}\,Pr^{1/3}",
            "gotas y esferas; Re < 10⁵",
            lambda d: nu_ranz_marshall(d["Re"], d["Pr"]),
            lambda d: d["Re"] < 1e5,
        ),
        Correlation(
            "tube_laminar",
            "Laminar desarrollada",
            r"\begin{aligned}Nu &= 3.66 \;\;(T_s\ \text{cte.}) \\ "
            r"Nu &= 4.36 \;\;(q''_s\ \text{cte.})\end{aligned}",
            "Re < 2300; lejos de la entrada",
            lambda d: nu_tube_laminar(d["constant_T"] > 0.5),
            lambda d: d["Re"] < RE_LAMINAR_TUBE,
        ),
        Correlation(
            "hausen",
            "Hausen (1943): entrada térmica",
            r"Nu = 3.66 + \frac{0.0668\,Gz}{1 + 0.04\,Gz^{2/3}}",
            "Re < 2300; T de pared constante",
            lambda d: nu_hausen(d["Re"], d["Pr"], d["D_over_L"]),
            lambda d: d["Re"] < RE_LAMINAR_TUBE and d["constant_T"] > 0.5,
        ),
        Correlation(
            "dittus_boelter",
            "Dittus y Boelter (1930)",
            r"Nu = 0.023\,Re^{0.8}\,Pr^{n}",
            "Re ≥ 10⁴; 0,6 ≤ Pr ≤ 160; L/D ≥ 10",
            lambda d: nu_dittus_boelter(d["Re"], d["Pr"], d["heating"] > 0.5),
            lambda d: d["Re"] >= RE_TURBULENT_TUBE and _between(d["Pr"], 0.6, 160),
        ),
        Correlation(
            "gnielinski",
            "Gnielinski (1976)",
            r"Nu = \frac{(f/8)\,(Re - 1000)\,Pr}{1 + 12.7\,(f/8)^{1/2}\left(Pr^{2/3} - 1\right)}",
            "3000 ≤ Re ≤ 5·10⁶; 0,5 ≤ Pr ≤ 2000",
            lambda d: nu_gnielinski(d["Re"], d["Pr"]),
            lambda d: _between(d["Re"], 3000, 5e6) and _between(d["Pr"], 0.5, 2000),
        ),
        Correlation(
            "sieder_tate",
            "Sieder y Tate (1936)",
            r"Nu = 0.027\,Re^{0.8}\,Pr^{1/3}\left(\frac{\mu}{\mu_s}\right)^{0.14}",
            "Re ≥ 10⁴; 0,7 ≤ Pr ≤ 16 700",
            lambda d: nu_sieder_tate(d["Re"], d["Pr"], d["mu_ratio"]),
            lambda d: d["Re"] >= RE_TURBULENT_TUBE and _between(d["Pr"], 0.7, 16700),
        ),
        Correlation(
            "vertical_churchill_chu",
            "Churchill y Chu (1975)",
            r"\begin{aligned}Nu &= \left(0.825 + \frac{0.387\,Ra^{1/6}}{\psi}\right)^{2} \\ "
            r"\psi &= \left[1 + (0.492/Pr)^{9/16}\right]^{8/27}\end{aligned}",
            "todo Ra",
            lambda d: nu_vertical_churchill_chu(d["Ra"], d["Pr"]),
            lambda d: d["Ra"] <= 1e13,
        ),
        Correlation(
            "vertical_mcadams",
            "McAdams (1954)",
            r"Nu = 0.59\,Ra^{1/4} \;\;\text{o}\;\; 0.10\,Ra^{1/3}",
            "10⁴ ≤ Ra ≤ 10¹³",
            lambda d: nu_vertical_mcadams(d["Ra"]),
            lambda d: _between(d["Ra"], 1e4, 1e13),
        ),
        Correlation(
            "horizontal_upper",
            "Lloyd y Moran (1974)",
            r"Nu = 0.54\,Ra^{1/4} \;\;\text{o}\;\; 0.15\,Ra^{1/3}",
            "10⁴ ≤ Ra ≤ 10¹¹; L = A/P",
            lambda d: nu_horizontal_upper(d["Ra"]),
            lambda d: _between(d["Ra"], 1e4, 1e11),
        ),
        Correlation(
            "horizontal_lower",
            "McAdams (1954)",
            r"Nu = 0.27\,Ra^{1/4}",
            "10⁵ ≤ Ra ≤ 10¹¹; L = A/P",
            lambda d: nu_horizontal_lower(d["Ra"]),
            lambda d: _between(d["Ra"], 1e5, 1e11),
        ),
        Correlation(
            "horizontal_cylinder_churchill_chu",
            "Churchill y Chu (1975)",
            r"\begin{aligned}Nu &= \left(0.60 + \frac{0.387\,Ra^{1/6}}{\psi}\right)^{2} \\ "
            r"\psi &= \left[1 + (0.559/Pr)^{9/16}\right]^{8/27}\end{aligned}",
            "Ra ≤ 10¹²",
            lambda d: nu_horizontal_cylinder_churchill_chu(d["Ra"], d["Pr"]),
            lambda d: d["Ra"] <= 1e12,
        ),
        Correlation(
            "morgan",
            "Morgan (1975)",
            r"Nu = C\,Ra^{n}",
            "10⁻¹⁰ ≤ Ra ≤ 10¹²",
            lambda d: nu_morgan(d["Ra"]),
            lambda d: _between(d["Ra"], 1e-10, 1e12),
        ),
        Correlation(
            "sphere_churchill",
            "Churchill (1983)",
            r"Nu = 2 + \frac{0.589\,Ra^{1/4}}{\left[1 + (0.469/Pr)^{9/16}\right]^{4/9}}",
            "Ra ≤ 10¹¹; Pr ≥ 0,7",
            lambda d: nu_sphere_churchill(d["Ra"], d["Pr"]),
            lambda d: d["Ra"] <= 1e11 and _between(d["Pr"], 0.7, math.inf),
        ),
    )
}


# ---------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Alternative:
    """Otra correlación aplicada a los mismos datos, para comparar."""

    key: str
    name: str
    Nu: float
    h_W_per_m2K: float
    in_range: bool


@dataclass(frozen=True)
class ExternalFlowInputs:
    """Flujo forzado sobre una placa, un cilindro o una esfera.

    ``length_m`` es el largo de la placa en el sentido del flujo o el diámetro
    del cilindro o de la esfera; ``width_m`` el ancho de la placa o el largo
    del cilindro (para el área).
    """

    fluid: str
    geometry: ExternalGeometry
    V_m_per_s: float
    T_s_K: float
    T_inf_K: float
    length_m: float
    width_m: float = 1.0
    p_Pa: float = 101_325.0
    tripped: bool = False


@dataclass(frozen=True)
class NaturalConvectionInputs:
    """Convección natural sobre una placa, un cilindro horizontal o una esfera.

    ``length_m`` es el alto de la placa vertical, el largo de la horizontal o
    el diámetro; ``width_m`` el ancho de la placa o el largo del cilindro.
    """

    fluid: str
    geometry: NaturalGeometry
    T_s_K: float
    T_inf_K: float
    length_m: float
    width_m: float = 1.0
    p_Pa: float = 101_325.0


@dataclass(frozen=True)
class ConvectionResult:
    """El resultado de una convección externa o natural."""

    inputs: ExternalFlowInputs | NaturalConvectionInputs
    props: Properties
    L_char_m: float
    Re: float | None
    Gr: float | None
    Ra: float | None
    regime: str
    correlation: str
    Nu: float
    h_W_per_m2K: float
    area_m2: float
    alternatives: tuple[Alternative, ...]
    warnings: tuple[str, ...]
    mu_s_Pa_s: float | None = None

    @property
    def Pr(self) -> float:
        return self.props.Pr

    @property
    def is_natural(self) -> bool:
        return isinstance(self.inputs, NaturalConvectionInputs)

    @property
    def Q_W(self) -> float:
        i = self.inputs
        return self.h_W_per_m2K * self.area_m2 * (i.T_s_K - i.T_inf_K)

    @property
    def q_W_per_m2(self) -> float:
        i = self.inputs
        return self.h_W_per_m2K * (i.T_s_K - i.T_inf_K)

    @property
    def T_film_K(self) -> float:
        return 0.5 * (self.inputs.T_s_K + self.inputs.T_inf_K)


def _validate_common(T_s: float, T_inf: float, p: float, lengths: dict[str, float]) -> None:
    if not (T_s > 0.0 and T_inf > 0.0):
        raise ValueError("Las temperaturas tienen que ser absolutas positivas.")
    if not p > 0.0:
        raise ValueError("La presión tiene que ser positiva.")
    for name, value in lengths.items():
        if not value > 0.0:
            raise ValueError(f"{name} tiene que ser positivo.")


def _alternatives(
    keys: list[str], d: dict[str, float], k: float, L: float
) -> tuple[Alternative, ...]:
    out = []
    for key in keys:
        c = CORRELATIONS[key]
        Nu = c.func(d)
        if Nu > 0.0:  # la mixta de la placa da Nu < 0 con Re chico: no tiene sentido mostrarla
            out.append(Alternative(key, c.name, Nu, Nu * k / L, bool(c.in_range(d))))
    return tuple(out)


#: Re a partir del cual la capa límite de un cilindro o una esfera se vuelve turbulenta
#: antes de separarse (Incropera, §7.4).
RE_BLUFF_TURBULENT = 2.0e5


def _bluff_regime(Re: float) -> str:
    if Re < RE_BLUFF_TURBULENT:
        return "capa límite laminar (se separa a ~80° del frente)"
    return "capa límite turbulenta (se separa más atrás, a ~140°)"


def solve_external(inputs: ExternalFlowInputs) -> ConvectionResult:
    """Convección forzada externa (Cengel y Ghajar, cap. 7; Incropera, cap. 7).

    Raises
    ------
    ValueError
        Con un mensaje para el alumno (velocidad, largo o temperaturas no
        positivos, fluido sin propiedades de transporte o cambiando de fase).
    """
    i = inputs
    if i.geometry not in EXTERNAL_GEOMETRIES:
        raise ValueError(f"Geometría desconocida: {i.geometry!r}.")
    L_name = "El largo de la placa" if i.geometry == "plate" else "El diámetro"
    _validate_common(i.T_s_K, i.T_inf_K, i.p_Pa, {L_name: i.length_m, "El ancho": i.width_m})
    if not i.V_m_per_s > 0.0:
        raise ValueError("La velocidad del fluido tiene que ser positiva.")
    T_f = 0.5 * (i.T_s_K + i.T_inf_K)
    warnings: list[str] = []
    wall = _check_wall_phase(i.fluid, i.T_s_K, i.T_inf_K, i.p_Pa)
    if wall:
        warnings.append(wall)
    L = i.length_m
    mu_s: float | None = None
    if i.geometry == "sphere":
        # Whitaker: propiedades a T∞ y μ_s a T_s
        props = fluid_properties(i.fluid, i.T_inf_K, i.p_Pa)
        mu_s = fluid_properties(i.fluid, i.T_s_K, i.p_Pa).mu_Pa_s
        film = fluid_properties(i.fluid, T_f, i.p_Pa)
        Re = i.V_m_per_s * L / props.nu_m2_per_s
        d = {"Re": Re, "Pr": props.Pr, "mu_ratio": props.mu_Pa_s / mu_s}
        main = "whitaker"
        Nu = CORRELATIONS[main].func(d)
        h = Nu * props.k_W_per_mK / L
        d_film = {"Re": i.V_m_per_s * L / film.nu_m2_per_s, "Pr": film.Pr}
        alts = (
            Alternative(
                "whitaker",
                CORRELATIONS["whitaker"].name,
                Nu,
                h,
                CORRELATIONS["whitaker"].in_range(d),
            ),
            *_alternatives(["ranz_marshall"], d_film, film.k_W_per_mK, L),
        )
        area = math.pi * L * L
        regime = _bluff_regime(Re)
    else:
        props = fluid_properties(i.fluid, T_f, i.p_Pa)
        Re = i.V_m_per_s * L / props.nu_m2_per_s
        d = {"Re": Re, "Pr": props.Pr}
        if i.geometry == "plate":
            if i.tripped:
                main, regime = "plate_turbulent", "turbulento desde el borde de ataque"
            elif Re < RE_CRITICAL_PLATE:
                main, regime = "plate_laminar", "laminar"
            else:
                main, regime = "plate_mixed", "laminar y después turbulento"
            keys = ["plate_laminar", "plate_mixed", "plate_turbulent"]
            area = L * i.width_m
        else:
            main, regime = "churchill_bernstein", _bluff_regime(Re)
            keys = ["churchill_bernstein", "hilpert"]
            area = math.pi * L * i.width_m
        Nu = CORRELATIONS[main].func(d)
        h = Nu * props.k_W_per_mK / L
        alts = _alternatives(keys, d, props.k_W_per_mK, L)
    c = CORRELATIONS[main]
    if not c.in_range(d):
        warnings.append(f"{c.name}: fuera de su rango ({c.validity}).")
    return ConvectionResult(
        i, props, L, Re, None, None, regime, main, Nu, h, area, alts, tuple(warnings), mu_s
    )


def solve_natural(inputs: NaturalConvectionInputs) -> ConvectionResult:
    """Convección natural (Cengel y Ghajar, cap. 9; Incropera, cap. 9).

    Ra = g·β·|T_s − T∞|·L³/(ν·α), con las propiedades a T_f. En las placas
    horizontales L = A/P (Goldstein et al., 1973).
    """
    i = inputs
    if i.geometry not in NATURAL_GEOMETRIES:
        raise ValueError(f"Geometría desconocida: {i.geometry!r}.")
    _validate_common(i.T_s_K, i.T_inf_K, i.p_Pa, {"El largo": i.length_m, "El ancho": i.width_m})
    if i.T_s_K == i.T_inf_K:
        raise ValueError(
            "La superficie y el fluido están a la misma temperatura: sin diferencia de "
            "temperatura no hay empuje ni convección natural."
        )
    T_f = 0.5 * (i.T_s_K + i.T_inf_K)
    props = fluid_properties(i.fluid, T_f, i.p_Pa)
    warnings: list[str] = []
    wall = _check_wall_phase(i.fluid, i.T_s_K, i.T_inf_K, i.p_Pa)
    if wall:
        warnings.append(wall)
    if props.beta_per_K <= 0.0:
        raise ValueError(
            f"A {_degC(T_f)} {_el(i.fluid)} no se dilata al calentarse (β ≤ 0; el "
            "agua entre 0 y 4 °C): el empuje se invierte y estas correlaciones no sirven."
        )
    hot = i.T_s_K > i.T_inf_K
    if i.geometry in ("horizontal_plate_up", "horizontal_plate_down"):
        L = i.length_m * i.width_m / (2.0 * (i.length_m + i.width_m))
        area = i.length_m * i.width_m
    elif i.geometry == "vertical_plate":
        L = i.length_m
        area = i.length_m * i.width_m
    elif i.geometry == "horizontal_cylinder":
        L = i.length_m
        area = math.pi * L * i.width_m
    else:
        L = i.length_m
        area = math.pi * L * L
    dT = abs(i.T_s_K - i.T_inf_K)
    Gr = G_STANDARD * props.beta_per_K * dT * L**3 / props.nu_m2_per_s**2
    Ra = Gr * props.Pr
    d = {"Ra": Ra, "Pr": props.Pr}
    if i.geometry == "vertical_plate":
        main, keys = "vertical_churchill_chu", ["vertical_churchill_chu", "vertical_mcadams"]
        regime = "laminar" if Ra <= 1e9 else "turbulento"
    elif i.geometry in ("horizontal_plate_up", "horizontal_plate_down"):
        # la cara de arriba de una placa caliente se comporta como la de abajo de una fría
        up = i.geometry == "horizontal_plate_up"
        main = "horizontal_upper" if up == hot else "horizontal_lower"
        keys = [main]
        regime = "turbulento" if main == "horizontal_upper" and Ra > 1e7 else "laminar"
    elif i.geometry == "horizontal_cylinder":
        main = "horizontal_cylinder_churchill_chu"
        keys = ["horizontal_cylinder_churchill_chu", "morgan"]
        regime = "laminar" if Ra <= 1e9 else "turbulento"
    else:
        main, keys = "sphere_churchill", ["sphere_churchill"]
        regime = "laminar" if Ra <= 1e9 else "turbulento"
    Nu = CORRELATIONS[main].func(d)
    h = Nu * props.k_W_per_mK / L
    c = CORRELATIONS[main]
    if not c.in_range(d):
        warnings.append(f"{c.name}: fuera de su rango ({c.validity}).")
    alts = _alternatives(keys, d, props.k_W_per_mK, L)
    return ConvectionResult(
        i, props, L, None, Gr, Ra, regime, main, Nu, h, area, alts, tuple(warnings)
    )


# ---------------------------------------------------------------------
# Flujo interno
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class InternalFlowInputs:
    """Un fluido que circula por un tubo circular de diámetro D y largo L.

    Con ``condition="constant_T"`` la pared está a ``T_s_K``; con
    ``"constant_flux"`` entra ``q_W_per_m2`` por la pared. ``correlation`` es
    ``"auto"`` o la key de una correlación de tubo.
    """

    fluid: str
    D_m: float
    L_m: float
    m_dot_kg_s: float
    T_in_K: float
    condition: InternalCondition
    T_s_K: float | None = None
    q_W_per_m2: float | None = None
    p_Pa: float = 101_325.0
    correlation: str = "auto"


TUBE_CORRELATIONS = ("tube_laminar", "hausen", "dittus_boelter", "gnielinski", "sieder_tate")


@dataclass(frozen=True)
class InternalFlowResult:
    """El tubo: la temperatura de salida, h, Q̇ y la caída de presión."""

    inputs: InternalFlowInputs
    props: Properties
    T_out_K: float
    Re: float
    regime: str
    correlation: str
    Nu: float
    h_W_per_m2K: float
    f: float
    alternatives: tuple[Alternative, ...]
    warnings: tuple[str, ...]
    iterations: int
    mu_s_Pa_s: float | None = None

    @property
    def T_mean_K(self) -> float:
        return 0.5 * (self.inputs.T_in_K + self.T_out_K)

    @property
    def heating(self) -> bool:
        """True si el fluido se calienta (n = 0,4 en Dittus y Boelter)."""
        return self.T_out_K > self.inputs.T_in_K

    @property
    def area_m2(self) -> float:
        return math.pi * self.inputs.D_m * self.inputs.L_m

    @property
    def Q_W(self) -> float:
        """Q̇ = ṁ·c_p·(T_sal − T_ent): positivo si el fluido se calienta."""
        return (
            self.inputs.m_dot_kg_s * self.props.cp_J_per_kgK * (self.T_out_K - self.inputs.T_in_K)
        )

    @property
    def dT_lm_K(self) -> float | None:
        """ΔT_ml = (ΔT_ent − ΔT_sal)/ln(ΔT_ent/ΔT_sal) con T de pared constante."""
        if self.inputs.condition != "constant_T":
            return None
        Ts = self.inputs.T_s_K or 0.0
        dTi, dTo = Ts - self.inputs.T_in_K, Ts - self.T_out_K
        if abs(dTi - dTo) < 1e-12:
            return dTi
        return (dTi - dTo) / math.log(dTi / dTo)

    @property
    def T_s_out_K(self) -> float | None:
        """La T de la pared a la salida con flujo constante: T_s = T_m + q''/h."""
        if self.inputs.condition != "constant_flux":
            return None
        return self.T_out_K + (self.inputs.q_W_per_m2 or 0.0) / self.h_W_per_m2K

    @property
    def V_m_per_s(self) -> float:
        i = self.inputs
        return i.m_dot_kg_s / (self.props.rho_kg_per_m3 * math.pi * i.D_m**2 / 4.0)

    @property
    def dp_Pa(self) -> float:
        """Δp = f·(L/D)·ρ·V²/2 (Darcy–Weisbach; Incropera, §8.1)."""
        i = self.inputs
        return self.f * (i.L_m / i.D_m) * self.props.rho_kg_per_m3 * self.V_m_per_s**2 / 2.0

    @property
    def entry_lengths_m(self) -> tuple[float, float]:
        """(hidrodinámica, térmica): 0,05·Re·D y 0,05·Re·Pr·D en laminar; ≈ 10·D en
        turbulento (Incropera, §8.1 y §8.2)."""
        D = self.inputs.D_m
        if self.Re < RE_LAMINAR_TUBE:
            return 0.05 * self.Re * D, 0.05 * self.Re * self.props.Pr * D
        return 10.0 * D, 10.0 * D


def tube_profile(
    result: InternalFlowResult, n: int = 61
) -> tuple[list[float], list[float], list[float]]:
    """(x en m, T_m(x), T_s(x)) a lo largo del tubo, con el h promedio (Incropera, §8.3).

    Con T de pared constante, T_s − T_m cae exponencialmente; con flujo
    constante, T_m sube lineal y la pared queda q''/h por encima.
    """
    i = result.inputs
    C = i.m_dot_kg_s * result.props.cp_J_per_kgK
    P = math.pi * i.D_m
    xs = [i.L_m * j / (n - 1) for j in range(n)]
    if i.condition == "constant_T":
        Ts = i.T_s_K or 0.0
        Tm = [Ts - (Ts - i.T_in_K) * math.exp(-result.h_W_per_m2K * P * x / C) for x in xs]
        return xs, Tm, [Ts] * n
    q = i.q_W_per_m2 or 0.0
    Tm = [i.T_in_K + q * P * x / C for x in xs]
    return xs, Tm, [T + q / result.h_W_per_m2K for T in Tm]


def _saturation_T(fluid: str, p: float) -> float | None:
    """T de saturación a p, o None si no hay (supercrítico o fuera de la tabla)."""
    try:
        return saturation_at_pressure(fluid, p).T_sat_K
    except ValueError:
        return None


def solve_internal(inputs: InternalFlowInputs) -> InternalFlowResult:
    """Flujo en un tubo (Cengel y Ghajar, cap. 8; Incropera, cap. 8).

    La temperatura media T_m = (T_ent + T_sal)/2 fija las propiedades y la de
    salida depende de h: se itera hasta 10⁻⁹ K. Con T de pared constante,
    T_sal = T_s − (T_s − T_ent)·exp(−h·A/(ṁ·c_p)); con flujo constante,
    T_sal = T_ent + q''·A/(ṁ·c_p) (Incropera, §8.3).

    Raises
    ------
    ValueError
        Con un mensaje para el alumno (datos no positivos, falta la pared,
        el fluido cambia de fase dentro del tubo).
    """
    i = inputs
    if i.condition not in INTERNAL_CONDITIONS:
        raise ValueError(f"Condición de pared desconocida: {i.condition!r}.")
    if not (i.D_m > 0.0 and i.L_m > 0.0):
        raise ValueError("El diámetro y el largo del tubo tienen que ser positivos.")
    if not i.m_dot_kg_s > 0.0:
        raise ValueError("El caudal másico tiene que ser positivo.")
    if not (i.T_in_K > 0.0 and i.p_Pa > 0.0):
        raise ValueError("La temperatura de entrada y la presión tienen que ser positivas.")
    constant_T = i.condition == "constant_T"
    T_s = i.T_s_K or 0.0
    q = i.q_W_per_m2 or 0.0
    if constant_T:
        if not T_s > 0.0:
            raise ValueError("Falta la temperatura de la pared.")
        if T_s == i.T_in_K:
            raise ValueError(
                "La pared y el fluido entran a la misma temperatura: no hay transferencia de calor."
            )
    elif q == 0.0:
        raise ValueError("Falta el flujo de calor por la pared (distinto de cero).")
    if i.correlation != "auto" and i.correlation not in TUBE_CORRELATIONS:
        raise ValueError(f"Correlación de tubo desconocida: {i.correlation!r}.")
    A = math.pi * i.D_m * i.L_m
    T_sat = _saturation_T(i.fluid, i.p_Pa)
    props = fluid_properties(i.fluid, i.T_in_K, i.p_Pa)
    # primera vuelta con las propiedades de la entrada (T_m = T_ent)
    T_out = i.T_in_K
    main = i.correlation
    h = 0.0
    mu_s = props.mu_Pa_s
    d: dict[str, float] = {}
    for iterations in range(1, 201):  # noqa: B007 — se informa cuántas hicieron falta
        T_m = 0.5 * (i.T_in_K + T_out)
        props = _tube_properties(i, T_m, "la temperatura media")
        Re = 4.0 * i.m_dot_kg_s / (math.pi * i.D_m * props.mu_Pa_s)
        # la pared: la dada, o la media con flujo constante (T_m + q''/h)
        T_wall = T_s if constant_T else T_m + q / (h if h > 0.0 else 1000.0)
        mu_s = _tube_properties(i, T_wall, "la pared").mu_Pa_s
        d = {
            "Re": Re,
            "Pr": props.Pr,
            "D_over_L": i.D_m / i.L_m,
            "mu_ratio": props.mu_Pa_s / mu_s,
            "heating": 1.0 if T_wall > T_m else 0.0,
            "constant_T": 1.0 if constant_T else 0.0,
        }
        if i.correlation == "auto":
            if Re < RE_LAMINAR_TUBE:
                main = "hausen" if constant_T else "tube_laminar"
            else:
                main = "gnielinski"
        Nu = CORRELATIONS[main].func(d)
        h = Nu * props.k_W_per_mK / i.D_m
        if constant_T:
            NTU = h * A / (i.m_dot_kg_s * props.cp_J_per_kgK)
            T_new = T_s - (T_s - i.T_in_K) * math.exp(-NTU)
        else:
            T_new = i.T_in_K + q * A / (i.m_dot_kg_s * props.cp_J_per_kgK)
            if T_new <= 0.0:
                raise ValueError(
                    f"Con ese flujo de calor {_el(i.fluid)} tendría que salir a {_num(T_new)} K, "
                    "por debajo del cero absoluto: el tubo le saca más calor del que puede "
                    "entregar. Revisá el signo y el valor de q''."
                )
        _check_no_phase_change(i, T_new, T_sat)
        converged = abs(T_new - T_out) < 1e-9
        T_out = T_new
        if converged:
            break
    else:  # pragma: no cover — converge en decenas de iteraciones
        raise ValueError("La temperatura media del fluido no convergió.")
    _tube_properties(i, T_out, "la salida")  # dentro del rango del fluido
    Re = d["Re"]
    if Re < RE_LAMINAR_TUBE:
        regime, f = "laminar", 64.0 / Re
    elif Re < RE_TURBULENT_TUBE:
        regime, f = "de transición", friction_petukhov(max(Re, 3000.0))
    else:
        regime, f = "turbulento", friction_petukhov(Re)
    warnings: list[str] = []
    c = CORRELATIONS[main]
    if not c.in_range(d):
        warnings.append(f"{c.name}: fuera de su rango ({c.validity}).")
    if regime == "de transición":
        warnings.append(
            f"Re = {_num(Re)}: el flujo está en transición (2300 a 10⁴); el h es incierto y "
            "Gnielinski es la mejor estimación."
        )
    if main in ("dittus_boelter", "sieder_tate") and i.L_m / i.D_m < 10.0:
        warnings.append(
            "L/D < 10: el tubo es corto y la entrada aumenta h por encima de la correlación."
        )
    # la pared más alejada del fluido: la dada, o la de la salida con flujo constante
    T_wall_max = T_s if constant_T else T_out + q / h
    wall = _check_wall_phase(i.fluid, T_wall_max, 0.5 * (i.T_in_K + T_out), i.p_Pa)
    if wall:
        warnings.append(wall)
    if Re < RE_LAMINAR_TUBE:
        keys = ["tube_laminar", "hausen"] if constant_T else ["tube_laminar"]
    else:
        keys = ["dittus_boelter", "gnielinski", "sieder_tate"]
    alts = _alternatives(keys, d, props.k_W_per_mK, i.D_m)
    return InternalFlowResult(
        i, props, T_out, Re, regime, main, Nu, h, f, alts, tuple(warnings), iterations, mu_s
    )


def _tube_properties(i: InternalFlowInputs, T_K: float, where: str) -> Properties:
    """Las propiedades en el tubo; si salen del rango del fluido, el error dice dónde."""
    try:
        return fluid_properties(i.fluid, T_K, i.p_Pa)
    except ValueError as exc:
        raise ValueError(
            f"Con estos datos {where} del tubo llegaría a {_degC(T_K)}. {exc}"
        ) from exc


def _check_no_phase_change(i: InternalFlowInputs, T_out: float, T_sat: float | None) -> None:
    """Error si el fluido llega a la saturación entre la entrada y la salida."""
    if T_sat is None or (i.T_in_K - T_sat) * (T_out - T_sat) > 0.0:
        return
    boils = T_out > i.T_in_K
    raise ValueError(
        f"{_el(i.fluid).capitalize()} saldría a {_degC(T_out)} y a {_num(i.p_Pa / 1e5)} bar "
        f"{'hierve' if boils else 'condensa'} a {_degC(T_sat)}: cambia de fase dentro del "
        "tubo, y estas correlaciones son de una sola fase. Achicá el largo, subí el caudal o "
        + ("subí" if boils else "bajá")
        + " la presión."
    )


# ---------------------------------------------------------------------
# Curva de Nu
# ---------------------------------------------------------------------


def dimensionless_groups(result: ConvectionResult | InternalFlowResult) -> dict[str, float]:
    """Los números con los que se evalúan las correlaciones (Re o Ra, Pr, μ/μ_s…)."""
    mu_ratio = result.props.mu_Pa_s / (result.mu_s_Pa_s or result.props.mu_Pa_s)
    if isinstance(result, InternalFlowResult):
        return {
            "Re": result.Re,
            "Pr": result.props.Pr,
            "D_over_L": result.inputs.D_m / result.inputs.L_m,
            "mu_ratio": mu_ratio,
            "heating": 1.0 if result.heating else 0.0,
            "constant_T": 1.0 if result.inputs.condition == "constant_T" else 0.0,
        }
    d = {"Pr": result.Pr, "mu_ratio": mu_ratio}
    if result.Ra is not None:
        d["Ra"] = result.Ra
    if result.Re is not None:
        d["Re"] = result.Re
    return d


def nu_curve(
    result: ConvectionResult | InternalFlowResult, n: int = 80
) -> tuple[str, list[float], list[float]]:
    """La correlación usada como curva Nu(Re) o Nu(Ra), con el Pr de los datos.

    Devuelve (variable, valores, Nu): Re de 1/30 a 30 veces el de los datos (o
    Ra de 10⁻³ a 10³ veces), recortado al rango de la correlación pero siempre
    con el punto de los datos adentro (es uno de los valores).
    """
    c = CORRELATIONS[result.correlation]
    base = dimensionless_groups(result)
    if isinstance(result, InternalFlowResult):
        var, x0, span = "Re", result.Re, 30.0
        if result.correlation in ("hausen", "tube_laminar"):
            lo, hi = 10.0, RE_LAMINAR_TUBE
        elif result.correlation == "gnielinski":
            lo, hi = 3000.0, max(x0 * span, 1e5)
        else:
            lo, hi = 1e4, max(x0 * span, 1e5)
    elif result.Ra is not None:
        var, x0 = "Ra", result.Ra
        lo, hi = x0 / 1e3, x0 * 1e3
    else:
        var, x0 = "Re", result.Re or 1.0
        lo, hi = x0 / 30.0, x0 * 30.0
        if result.correlation == "plate_laminar":
            hi = min(hi, RE_CRITICAL_PLATE)
        if result.correlation in ("plate_mixed", "plate_turbulent"):
            lo = max(lo, RE_CRITICAL_PLATE)
    lo, hi = min(lo, x0 / 2.0), max(hi, 2.0 * x0)
    xs = sorted({lo * (hi / lo) ** (j / (n - 1)) for j in range(n)} | {x0})
    nus = [c.func({**base, var: x}) for x in xs]
    return var, xs, nus


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def convection_notes(result: ConvectionResult) -> list[str]:
    """Interpretación física (markdown)."""
    notes: list[str] = list(result.warnings)
    props = result.props
    if result.is_natural:
        i_n = result.inputs
        assert isinstance(i_n, NaturalConvectionInputs)
        if i_n.geometry in ("horizontal_plate_up", "horizontal_plate_down"):
            hot = i_n.T_s_K > i_n.T_inf_K
            if result.correlation == "horizontal_upper":
                notes.append(
                    "El fluido "
                    + ("calentado sube" if hot else "enfriado baja")
                    + " libre desde la cara y lo reemplaza fluido del ambiente: es la cara "
                    "que mejor transfiere."
                )
            else:
                notes.append(
                    "El fluido "
                    + ("calentado queda atrapado debajo" if hot else "enfriado queda arriba")
                    + " de la placa y tiene que escaparse por los bordes: h es menor que en "
                    "la otra cara."
                )
            notes.append(
                f"En una placa horizontal la longitud característica es L = A/P = "
                f"{_num(100.0 * result.L_char_m)} cm (Goldstein et al., 1973)."
            )
        elif i_n.geometry == "vertical_plate" or i_n.geometry == "horizontal_cylinder":
            notes.append(
                f"Ra = Gr·Pr = {_num(result.Ra or 0.0)}: "
                + (
                    "flujo laminar en la capa límite (Ra ≤ 10⁹)."
                    if (result.Ra or 0.0) <= 1e9
                    else "la capa límite es turbulenta (Ra > 10⁹)."
                )
            )
    else:
        i = result.inputs
        assert isinstance(i, ExternalFlowInputs)
        if i.geometry == "plate" and not i.tripped and (result.Re or 0.0) >= RE_CRITICAL_PLATE:
            x_cr = RE_CRITICAL_PLATE * props.nu_m2_per_s / i.V_m_per_s
            notes.append(
                f"La capa límite pasa a turbulenta a x_cr = Re_cr·ν/V = {_num(100.0 * x_cr)} cm "
                f"del borde de ataque ({_num(100.0 * x_cr / i.length_m)} % de la placa)."
            )
    if props.is_liquid:
        notes.append(
            f"Pr = {_num(props.Pr)}: en un líquido la capa límite térmica es más delgada que la "
            "de velocidad, y h es mucho mayor que en un gas a la misma velocidad."
        )
    if not result.is_natural and result.inputs.geometry == "plate":
        notes.extend(_plate_comparison(result))
    else:
        notes.extend(_spread_note(result.alternatives, CORRELATIONS[result.correlation].name))
    if props.fluid == "Air":
        notes.append(
            "Las propiedades del aire salen de CoolProp: k es ~2,7 % mayor y Pr ~3 % menor que en "
            "la tabla A-15 de Cengel y Ghajar, así que h queda 1,5 a 2 % por encima del libro."
        )
    return notes


def _spread_note(alternatives: tuple[Alternative, ...], main_name: str) -> list[str]:
    """Cuánto difieren las correlaciones que valen para los datos."""
    valid = [a for a in alternatives if a.in_range]
    if len(valid) < 2:
        return []
    hs = [a.h_W_per_m2K for a in valid]
    spread = 100.0 * (max(hs) - min(hs)) / min(hs)
    if spread <= 25.0:
        return [
            f"Las correlaciones que valen para estos datos difieren en {_pct(spread)} % entre "
            "sí: es la incertidumbre típica de una correlación (±10 a 25 %)."
        ]
    return [
        f"Las correlaciones que valen para estos datos difieren en {_pct(spread)} % entre sí, "
        f"más que la incertidumbre típica (±10 a 25 %): se usa {main_name}, la más general."
    ]


def _plate_comparison(result: ConvectionResult) -> list[str]:
    """La placa: qué pasaría con la capa límite turbulenta desde el borde (o sin alambre)."""
    alts = {a.key: a for a in result.alternatives}
    h = result.h_W_per_m2K
    if result.correlation != "plate_turbulent":
        turb = alts.get("plate_turbulent")
        if turb is None or not turb.in_range:
            return []
        return [
            f"Si la capa límite fuera turbulenta desde el borde de ataque (con un alambre o "
            f"una rugosidad que la dispare), h sería {_num(turb.h_W_per_m2K)} W/(m²·K): "
            f"{_num(turb.h_W_per_m2K / h, 2)} veces más."
        ]
    natural = alts.get("plate_mixed") or alts.get("plate_laminar")
    if natural is None:
        return []
    return [
        f"Sin el alambre, con la transición natural a Re = 5·10⁵, h sería "
        f"{_num(natural.h_W_per_m2K)} W/(m²·K): disparar la turbulencia lo multiplica por "
        f"{_num(h / natural.h_W_per_m2K, 2)}."
    ]


def internal_notes(result: InternalFlowResult) -> list[str]:
    notes: list[str] = list(result.warnings)
    L_h, L_t = result.entry_lengths_m
    i = result.inputs
    if result.Re < RE_LAMINAR_TUBE:
        notes.append(
            f"Flujo laminar: el perfil de velocidad se desarrolla en {_num(L_h)} m y el de "
            f"temperatura en {_num(L_t)} m (el tubo mide {_num(i.L_m)} m)."
        )
    else:
        notes.append(
            f"Flujo turbulento: el perfil se desarrolla en ~10·D = {_num(L_h)} m; el h promedio "
            "del tubo es casi el desarrollado."
        )
    alts = {a.key: a for a in result.alternatives}
    if "hausen" in alts and "tube_laminar" in alts:
        gain = 100.0 * (alts["hausen"].Nu / alts["tube_laminar"].Nu - 1.0)
        notes.append(
            f"Cerca de la entrada la capa límite térmica es delgada y h es mayor: Hausen da "
            f"Nu = {_num(alts['hausen'].Nu)}, {_pct(gain)} % más que el desarrollado (3,66)."
        )
    elif result.Re >= RE_LAMINAR_TUBE:
        notes.extend(_spread_note(result.alternatives, CORRELATIONS[result.correlation].name))
        if "gnielinski" in alts and "dittus_boelter" in alts:
            notes.append(
                "Gnielinski (±10 %) es más precisa que Dittus y Boelter (±25 %), que es la más "
                "simple; Sieder y Tate corrige por la viscosidad en la pared."
            )
    notes.append(f"La temperatura media del fluido se iteró: {result.iterations} vueltas.")
    return notes


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------

C = 273.15


@dataclass(frozen=True)
class ConvectionExample:
    inputs: ExternalFlowInputs | NaturalConvectionInputs | InternalFlowInputs
    note: str = ""


EXTERNAL_EXAMPLES: dict[str, ConvectionExample] = {
    "Caño de vapor con viento (Cengel y Ghajar: D = 10 cm a 110 °C, aire a 10 °C y 8 m/s)": (
        ConvectionExample(
            ExternalFlowInputs("Air", "cylinder", 8.0, 110.0 + C, 10.0 + C, 0.10, 1.0),
            "El libro (con la tabla A-15): Nu = 124, h = 34,8 W/(m²·K) y 1093 W por metro.",
        )
    ),
    "Placa al viento (1 m a 60 °C; aire a 20 °C y 10 m/s)": ConvectionExample(
        ExternalFlowInputs("Air", "plate", 10.0, 60.0 + C, 20.0 + C, 1.0, 1.0),
        "Re apenas pasa 5·10⁵: la mayor parte de la placa tiene capa límite laminar.",
    ),
    "Placa con capa límite laminar (0,5 m a 75 °C; aire a 25 °C y 2 m/s)": ConvectionExample(
        ExternalFlowInputs("Air", "plate", 2.0, 75.0 + C, 25.0 + C, 0.5, 0.5),
    ),
    "Placa turbulenta desde el borde (con un alambre que dispara la turbulencia)": (
        ConvectionExample(
            ExternalFlowInputs("Air", "plate", 10.0, 60.0 + C, 20.0 + C, 1.0, 1.0, tripped=True),
            "Los mismos datos que la placa al viento, pero turbulenta desde el borde: h sube.",
        )
    ),
    "Esfera de 25 mm en aire (75 °C; aire a 20 °C y 15 m/s)": ConvectionExample(
        ExternalFlowInputs("Air", "sphere", 15.0, 75.0 + C, 20.0 + C, 0.025),
    ),
    "Caño de 2 cm en agua (40 °C; agua a 20 °C y 0,5 m/s)": ConvectionExample(
        ExternalFlowInputs("Water", "cylinder", 0.5, 40.0 + C, 20.0 + C, 0.02, 1.0),
        "El mismo tipo de flujo con agua: h es dos órdenes de magnitud mayor que con aire.",
    ),
}

NATURAL_EXAMPLES: dict[str, ConvectionExample] = {
    "Caño horizontal de agua caliente (Cengel y Ghajar: D = 8 cm, 6 m, 70 °C; aire a 20 °C)": (
        ConvectionExample(
            NaturalConvectionInputs("Air", "horizontal_cylinder", 70.0 + C, 20.0 + C, 0.08, 6.0),
            "El libro (con la tabla A-15): Ra = 1,867·10⁶, Nu = 17,40 y 443 W.",
        )
    ),
    "Placa vertical caliente (0,6 × 0,6 m a 90 °C; aire a 30 °C)": ConvectionExample(
        NaturalConvectionInputs("Air", "vertical_plate", 90.0 + C, 30.0 + C, 0.6, 0.6),
        "Basado en Cengel y Ghajar: la misma placa en tres posiciones (también las dos "
        "horizontales).",
    ),
    "Placa horizontal caliente, la cara de arriba (0,6 × 0,6 m a 90 °C)": ConvectionExample(
        NaturalConvectionInputs("Air", "horizontal_plate_up", 90.0 + C, 30.0 + C, 0.6, 0.6),
    ),
    "Placa horizontal caliente, la cara de abajo (0,6 × 0,6 m a 90 °C)": ConvectionExample(
        NaturalConvectionInputs("Air", "horizontal_plate_down", 90.0 + C, 30.0 + C, 0.6, 0.6),
        "Abajo el aire caliente queda atrapado contra la placa: h es la mitad que arriba.",
    ),
    "Esfera caliente en aire quieto (D = 10 cm a 80 °C; aire a 20 °C)": ConvectionExample(
        NaturalConvectionInputs("Air", "sphere", 80.0 + C, 20.0 + C, 0.10),
    ),
    "Placa vertical en agua (0,3 m a 50 °C; agua a 20 °C)": ConvectionExample(
        NaturalConvectionInputs("Water", "vertical_plate", 50.0 + C, 20.0 + C, 0.3, 0.3),
    ),
}

INTERNAL_EXAMPLES: dict[str, ConvectionExample] = {
    "Agua calentada por vapor en un tubo (D = 2,5 cm, 0,3 kg/s, 15 °C, pared a 120 °C)": (
        ConvectionExample(
            InternalFlowInputs(
                "Water", 0.025, 5.0, 0.3, 15.0 + C, "constant_T", T_s_K=120.0 + C, p_Pa=3.0e5
            ),
            "Basado en Cengel y Ghajar (vapor que condensa afuera a 120 °C). A 3 bar el agua "
            "no hierve en la pared.",
        )
    ),
    "Agua calentada con resistencias (Cengel y Ghajar: D = 3 cm, 5 m, 10 L/min de 15 a 65 °C)": (
        ConvectionExample(
            InternalFlowInputs(
                "Water",
                0.03,
                5.0,
                0.1654,
                15.0 + C,
                "constant_flux",
                q_W_per_m2=73_400.0,
                p_Pa=2.0e5,
                correlation="dittus_boelter",
            ),
            "El libro (Dittus y Boelter, propiedades a 40 °C): Re = 10 750, Nu = 69,4, "
            "h = 1460 W/(m²·K) y la pared a 115 °C a la salida. A 1 atm esa pared pasaría los "
            "100 °C: el ejemplo usa 2 bar, como el agua de una instalación.",
        )
    ),
    "Aire calentado por resistencias en un ducto (D = 15 cm, flujo de calor constante)": (
        ConvectionExample(
            InternalFlowInputs(
                "Air", 0.15, 7.0, 0.065, 20.0 + C, "constant_flux", q_W_per_m2=300.0
            ),
        )
    ),
    "Agua fría en un caño fino: laminar con entrada térmica (D = 6 mm, 3 m)": ConvectionExample(
        InternalFlowInputs("Water", 0.006, 3.0, 0.004, 10.0 + C, "constant_T", T_s_K=40.0 + C),
        "Re < 2300: Hausen tiene en cuenta que cerca de la entrada h es mayor.",
    ),
    "Agua enfriada en un tubo (Dittus y Boelter con n = 0,3)": ConvectionExample(
        InternalFlowInputs(
            "Water",
            0.02,
            4.0,
            0.2,
            80.0 + C,
            "constant_T",
            T_s_K=20.0 + C,
            correlation="dittus_boelter",
        ),
        "Si el fluido se enfría, Dittus y Boelter usa Pr^0,3 en vez de Pr^0,4.",
    ),
}


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _props_dict(p: Properties, system: UnitSystem) -> dict[str, Any]:
    return {
        "T": format_quantity(p.T_K, "temperature", system),
        "rho": format_quantity(p.rho_kg_per_m3, "density", system),
        "mu": format_quantity(p.mu_Pa_s, "dynamic_viscosity", system),
        "k": format_quantity(p.k_W_per_mK, "thermal_conductivity", system),
        "cp": format_quantity(p.cp_J_per_kgK, "specific_heat", system),
        "Pr": p.Pr,
        "nu": format_quantity(p.nu_m2_per_s, "diffusivity", system),
        "beta_1_por_K": p.beta_per_K,
    }


def convection_to_dict(result: ConvectionResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado (externa o natural) → dict serializable (export CSV/JSON)."""
    i = result.inputs
    geo = NATURAL_GEOMETRIES[i.geometry] if result.is_natural else EXTERNAL_GEOMETRIES[i.geometry]  # type: ignore[index]
    data: dict[str, Any] = {
        "fluido": FLUID_NAMES_ES.get(i.fluid, i.fluid),
        "geometria": geo,
        "T_superficie": format_quantity(i.T_s_K, "temperature", system),
        "T_fluido": format_quantity(i.T_inf_K, "temperature", system),
        "propiedades": _props_dict(result.props, system),
        "L_caracteristica": format_quantity(result.L_char_m, "length", system),
        "correlacion": CORRELATIONS[result.correlation].name,
        "regimen": result.regime,
        "Nu": result.Nu,
        "h": format_quantity(result.h_W_per_m2K, "heat_transfer_coefficient", system),
        "area": format_quantity(result.area_m2, "area", system),
        "Q": format_quantity(result.Q_W, "heat_rate", system),
        "comparacion": {a.name: a.Nu for a in result.alternatives},
    }
    if result.Re is not None:
        data["Re"] = result.Re
    if result.Ra is not None:
        data["Gr"] = result.Gr
        data["Ra"] = result.Ra
    return data


def internal_to_dict(result: InternalFlowResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado del tubo → dict serializable (export CSV/JSON)."""
    i = result.inputs
    data: dict[str, Any] = {
        "fluido": FLUID_NAMES_ES.get(i.fluid, i.fluid),
        "condicion": INTERNAL_CONDITIONS[i.condition],
        "D": format_quantity(i.D_m, "small_length", system),
        "L": format_quantity(i.L_m, "length", system),
        "caudal": format_quantity(i.m_dot_kg_s, "mass_flow", system),
        "T_entrada": format_quantity(i.T_in_K, "temperature", system),
        "T_salida": format_quantity(result.T_out_K, "temperature", system),
        "propiedades": _props_dict(result.props, system),
        "Re": result.Re,
        "regimen": result.regime,
        "correlacion": CORRELATIONS[result.correlation].name,
        "Nu": result.Nu,
        "h": format_quantity(result.h_W_per_m2K, "heat_transfer_coefficient", system),
        "Q": format_quantity(result.Q_W, "heat_rate", system),
        "f": result.f,
        "dp": format_quantity(result.dp_Pa, "pressure", system),
        "comparacion": {a.name: a.Nu for a in result.alternatives},
    }
    if result.dT_lm_K is not None:
        data["dT_ml"] = format_quantity(result.dT_lm_K, "temperature_difference", system)
    if result.T_s_out_K is not None:
        data["T_pared_salida"] = format_quantity(result.T_s_out_K, "temperature", system)
    return data
