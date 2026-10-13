"""Condensación en película (Fase 11.2).

Vapor saturado (o sobrecalentado) a la presión p que condensa sobre una pared más fría,
T_s < T_sat (Çengel y Ghajar, 2015, §10-4 a §10-7; Incropera et al., 2007, §10.6 a §10.11):

- **placa o tubo vertical** con los tres regímenes de la película según su número de
  Reynolds, Re = 4·ṁ/(p·μ_ℓ): laminar (Nusselt, 1916) con Re < 30, laminar con ondas
  (Kutateladze, 1963) hasta 1800 y turbulenta (Labuntsov, 1957);
- **placa inclinada**: la vertical por (cos θ)^¼;
- **tubo horizontal** (0,729), **banco de N tubos** en columna (h_N = h_1·N^−¼) y **esfera**
  (0,815), siempre laminares;
- **adentro de un tubo horizontal**, con poco vapor (Chato, 1962).

El calor latente corregido h*_fg = h_fg + 0,68·c_pℓ·(T_sat − T_s) suma el subenfriamiento del
condensado (Rohsenow, 1956) y, con vapor sobrecalentado, c_pv·(T_v − T_sat). Las propiedades
del líquido van a T_f = (T_sat + T_s)/2; h_fg y ρ_v, a T_sat.

Todo en SI: temperaturas en K, presiones en Pa, longitudes en m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

from CoolProp.CoolProp import PropsSI

from core.heat_transfer.phase_change import (
    PHASE_CHANGE_FLUIDS,
    G,
    PhaseProperties,
    SaturationState,
    fluid_name,
    phase_properties,
    saturation_state,
    with_article,
)
from core.units_system import UnitSystem, convert_from_si, unit_label

__all__ = [
    "CONDENSATION_EXAMPLES",
    "GEOMETRIES",
    "FILM_REGIMES",
    "RE_LAMINAR",
    "RE_TURBULENT",
    "RE_VAPOR_CHATO",
    "ComparisonRow",
    "CondensationCurve",
    "CondensationExample",
    "CondensationInputs",
    "CondensationResult",
    "FilmRegime",
    "Geometry",
    "condensation_curve",
    "condensation_to_dict",
    "dropwise_h",
    "film_thickness",
    "geometry_comparison",
    "h_chato",
    "h_horizontal_tube",
    "h_laminar_vertical",
    "h_sphere",
    "h_star",
    "h_turbulent",
    "h_wavy",
    "re_turbulent",
    "re_wavy",
    "solve_condensation",
]

Geometry = Literal[
    "vertical_plate",
    "vertical_tube",
    "inclined_plate",
    "horizontal_tube",
    "tube_bank",
    "sphere",
    "inside_tube",
]
FilmRegime = Literal["laminar", "wavy", "turbulent"]

GEOMETRIES: dict[Geometry, str] = {
    "vertical_plate": "Placa vertical",
    "vertical_tube": "Tubo vertical (por fuera)",
    "inclined_plate": "Placa inclinada",
    "horizontal_tube": "Tubo horizontal (por fuera)",
    "tube_bank": "Banco de tubos horizontales",
    "sphere": "Esfera",
    "inside_tube": "Adentro de un tubo horizontal",
}
FILM_REGIMES: dict[FilmRegime, str] = {
    "laminar": "laminar",
    "wavy": "laminar con ondas",
    "turbulent": "turbulenta",
}

#: Los límites de los regímenes de la película sobre una pared vertical (Çengel y Ghajar §10-5).
RE_LAMINAR = 30.0
RE_TURBULENT = 1800.0
#: Chato (1962) vale con Re_v = ρ_v·V_v·D/μ_v < 35 000 a la entrada (Çengel y Ghajar §10-6).
RE_VAPOR_CHATO = 35_000.0

#: Las geometrías que dependen del alto (o del largo en la pendiente) y tienen regímenes.
_TALL: tuple[Geometry, ...] = ("vertical_plate", "vertical_tube", "inclined_plate")


# ---------------------------------------------------------------------
# Las correlaciones (con las propiedades explícitas, para reproducir los libros)
# ---------------------------------------------------------------------


def h_star(h_fg: float, cp_l: float, dT: float, cp_v: float = 0.0, superheat: float = 0.0) -> float:
    """El calor latente corregido (Rohsenow, 1956; Çengel y Ghajar §10-5):
    h*_fg = h_fg + 0,68·c_pℓ·(T_sat − T_s) + c_pv·(T_v − T_sat). El segundo término es el
    subenfriamiento del condensado y el tercero, el sobrecalentamiento del vapor."""
    return h_fg + 0.68 * cp_l * dT + cp_v * superheat


def h_laminar_vertical(
    rho_l: float,
    rho_v: float,
    h_fg: float,
    k_l: float,
    mu_l: float,
    dT: float,
    L: float,
    g: float = G,
) -> float:
    """Película laminar sobre una placa vertical (Nusselt, 1916, Z. VDI 60:541; Çengel y Ghajar
    §10-5): h = 0,943·[g·ρ_ℓ·(ρ_ℓ − ρ_v)·h*_fg·k_ℓ³/(μ_ℓ·(T_sat − T_s)·L)]^¼, con Re < 30."""
    return 0.943 * (g * rho_l * (rho_l - rho_v) * h_fg * k_l**3 / (mu_l * dT * L)) ** 0.25


def _x_group(
    k_l: float, mu_l: float, nu_l: float, h_fg: float, dT: float, L: float, g: float = G
) -> float:
    """El grupo de las formas explícitas: L·k_ℓ·(T_sat − T_s)/(μ_ℓ·h*_fg)·(g/ν_ℓ²)^⅓."""
    return L * k_l * dT / (mu_l * h_fg) * (g / nu_l**2) ** (1.0 / 3.0)


def re_wavy(
    k_l: float, mu_l: float, nu_l: float, h_fg: float, dT: float, L: float, g: float = G
) -> float:
    """El Reynolds de una película con ondas (Çengel y Ghajar §10-5):
    Re = [4,81 + 3,70·L·k_ℓ·(T_sat − T_s)/(μ_ℓ·h*_fg)·(g/ν_ℓ²)^⅓]^0,820."""
    return (4.81 + 3.70 * _x_group(k_l, mu_l, nu_l, h_fg, dT, L, g)) ** 0.820


def h_wavy(Re: float, k_l: float, nu_l: float, g: float = G) -> float:
    """Película laminar con ondas, 30 < Re < 1800 (Kutateladze, 1963, Fundamentals of Heat
    Transfer; Çengel y Ghajar §10-5): h = Re·k_ℓ/(1,08·Re^1,22 − 5,2)·(g/ν_ℓ²)^⅓."""
    return Re * k_l / (1.08 * Re**1.22 - 5.2) * (g / nu_l**2) ** (1.0 / 3.0)


def re_turbulent(
    k_l: float,
    mu_l: float,
    nu_l: float,
    Pr_l: float,
    h_fg: float,
    dT: float,
    L: float,
    g: float = G,
) -> float:
    """El Reynolds de una película turbulenta (Çengel y Ghajar §10-5):
    Re = [0,0690·L·k_ℓ·Pr^0,5·(T_sat − T_s)/(μ_ℓ·h*_fg)·(g/ν_ℓ²)^⅓ − 151·Pr^0,5 + 253]^{4/3}."""
    x = _x_group(k_l, mu_l, nu_l, h_fg, dT, L, g)
    return (0.0690 * x * Pr_l**0.5 - 151.0 * Pr_l**0.5 + 253.0) ** (4.0 / 3.0)


def h_turbulent(Re: float, k_l: float, nu_l: float, Pr_l: float, g: float = G) -> float:
    """Película turbulenta, Re > 1800 (Labuntsov, 1957, Teploenergetika 4(7):72; Çengel y
    Ghajar §10-5): h = Re·k_ℓ/[8750 + 58·Pr^−0,5·(Re^0,75 − 253)]·(g/ν_ℓ²)^⅓."""
    return (
        Re * k_l / (8750.0 + 58.0 * Pr_l**-0.5 * (Re**0.75 - 253.0)) * (g / nu_l**2) ** (1.0 / 3.0)
    )


def h_horizontal_tube(
    rho_l: float,
    rho_v: float,
    h_fg: float,
    k_l: float,
    mu_l: float,
    dT: float,
    D: float,
    g: float = G,
) -> float:
    """Por fuera de un tubo horizontal (Nusselt, 1916; Çengel y Ghajar §10-5):
    h = 0,729·[g·ρ_ℓ·(ρ_ℓ − ρ_v)·h*_fg·k_ℓ³/(μ_ℓ·(T_sat − T_s)·D)]^¼."""
    return 0.729 * (g * rho_l * (rho_l - rho_v) * h_fg * k_l**3 / (mu_l * dT * D)) ** 0.25


def h_sphere(
    rho_l: float,
    rho_v: float,
    h_fg: float,
    k_l: float,
    mu_l: float,
    dT: float,
    D: float,
    g: float = G,
) -> float:
    """Sobre una esfera (Çengel y Ghajar §10-5): h = 0,815·[g·ρ_ℓ·(ρ_ℓ − ρ_v)·h*_fg·k_ℓ³/
    (μ_ℓ·(T_sat − T_s)·D)]^¼."""
    return 0.815 * (g * rho_l * (rho_l - rho_v) * h_fg * k_l**3 / (mu_l * dT * D)) ** 0.25


def h_chato(
    rho_l: float,
    rho_v: float,
    h_fg: float,
    cp_l: float,
    k_l: float,
    mu_l: float,
    dT: float,
    D: float,
    g: float = G,
) -> float:
    """Adentro de un tubo horizontal con poco vapor (Chato, 1962, ASHRAE J. 4:52; Çengel y
    Ghajar §10-6): h = 0,555·[g·ρ_ℓ·(ρ_ℓ − ρ_v)·k_ℓ³/(μ_ℓ·(T_sat − T_s)·D)·
    (h_fg + ⅜·c_pℓ·(T_sat − T_s))]^¼, con Re_v < 35 000."""
    h_mod = h_fg + 3.0 / 8.0 * cp_l * dT
    return 0.555 * (g * rho_l * (rho_l - rho_v) * k_l**3 / (mu_l * dT * D) * h_mod) ** 0.25


def film_thickness(
    rho_l: float,
    rho_v: float,
    h_fg: float,
    k_l: float,
    mu_l: float,
    dT: float,
    x: float,
    g: float = G,
) -> float:
    """El espesor de la película laminar a la distancia x del borde de arriba (Nusselt, 1916;
    Çengel y Ghajar §10-5): δ = [4·μ_ℓ·k_ℓ·(T_sat − T_s)·x/(g·ρ_ℓ·(ρ_ℓ − ρ_v)·h*_fg)]^¼."""
    return (4.0 * mu_l * k_l * dT * x / (g * rho_l * (rho_l - rho_v) * h_fg)) ** 0.25


def dropwise_h(T_sat_K: float) -> float | None:
    """La condensación en gotas del vapor de agua (Griffith, 1983, en Çengel y Ghajar §10-7):
    h = 51 104 + 2044·T_sat [°C] W/(m²·K) entre 22 y 100 °C y 255 510 W/(m²·K) arriba.
    ``None`` bajo 22 °C, fuera del ajuste."""
    t = T_sat_K - 273.15
    if t < 22.0:
        return None
    if t <= 100.0:
        return 51_104.0 + 2044.0 * t
    return 255_510.0


# ---------------------------------------------------------------------
# Datos y resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CondensationInputs:
    """Un vapor que condensa sobre una pared a T_s.

    ``L_m`` es el alto de la placa o del tubo vertical, el largo de la placa inclinada a lo largo
    de la pendiente o el largo de los tubos horizontales; ``W_m``, el ancho de la placa;
    ``D_m``, el diámetro del tubo o de la esfera. ``angle_deg`` es la inclinación de la placa
    respecto de la vertical; ``rows`` y ``columns``, los tubos del banco (``rows`` en cada
    columna vertical). ``T_v_K`` es la del vapor sobrecalentado (``None``: saturado) y
    ``V_v_m_s``, la velocidad del vapor a la entrada del tubo (para el rango de Chato).
    """

    fluid: str
    p_Pa: float
    geometry: Geometry
    T_s_K: float
    L_m: float = 1.0
    W_m: float = 1.0
    D_m: float = 0.025
    angle_deg: float = 0.0
    rows: int = 1
    columns: int = 1
    T_v_K: float | None = None
    V_v_m_s: float | None = None


@dataclass(frozen=True)
class CondensationResult:
    inputs: CondensationInputs
    sat: SaturationState
    liquid: PhaseProperties
    vapor: PhaseProperties | None
    dT_K: float
    superheat_K: float
    h_fg_star: float
    regime: FilmRegime
    Re: float | None
    h: float
    h_single: float | None
    area_m2: float
    delta_m: float | None
    Re_vapor: float | None
    h_dropwise: float | None
    notes: tuple[str, ...]

    @property
    def q(self) -> float:
        """El flujo de calor medio, h·(T_sat − T_s)."""
        return self.h * self.dT_K

    @property
    def Q_W(self) -> float:
        return self.q * self.area_m2

    @property
    def m_dot(self) -> float:
        """El caudal que condensa, Q̇/h*_fg."""
        return self.Q_W / self.h_fg_star

    @property
    def tubes(self) -> int:
        i = self.inputs
        return i.rows * i.columns if i.geometry == "tube_bank" else 1


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _area(i: CondensationInputs) -> float:
    if i.geometry in ("vertical_plate", "inclined_plate"):
        return i.L_m * i.W_m
    if i.geometry == "sphere":
        return math.pi * i.D_m**2
    tubes = i.rows * i.columns if i.geometry == "tube_bank" else 1
    return tubes * math.pi * i.D_m * i.L_m


def _validate(i: CondensationInputs) -> None:
    if i.fluid not in PHASE_CHANGE_FLUIDS:
        raise ValueError(f"El fluido {i.fluid!r} no está en la lista de la página.")
    if i.geometry not in GEOMETRIES:
        raise ValueError(f"La geometría tiene que ser una de {', '.join(GEOMETRIES)}.")
    if i.geometry in ("vertical_plate", "inclined_plate"):
        if not (i.L_m > 0.0 and i.W_m > 0.0):
            raise ValueError(
                "El alto (o el largo) y el ancho de la placa tienen que ser positivos."
            )
    else:
        if not i.D_m > 0.0:
            raise ValueError("El diámetro tiene que ser positivo.")
        if i.geometry != "sphere" and not i.L_m > 0.0:
            raise ValueError("El largo del tubo tiene que ser positivo.")
    if i.geometry == "inclined_plate" and not 0.0 <= i.angle_deg < 90.0:
        raise ValueError(
            "La inclinación se mide desde la vertical y va de 0° a menos de 90°: con 90° la "
            "placa queda horizontal y el condensado no escurre (no hay correlación de película)."
        )
    if i.geometry == "tube_bank" and not (i.rows >= 1 and i.columns >= 1):
        raise ValueError("El banco tiene que tener al menos un tubo por fila y por columna.")
    if i.V_v_m_s is not None and i.V_v_m_s < 0.0:
        raise ValueError("La velocidad del vapor no puede ser negativa.")


def _check_temperatures(i: CondensationInputs, sat: SaturationState) -> None:
    if not i.T_s_K < sat.T_sat_K:
        raise ValueError(
            f"La pared ({i.T_s_K - 273.15:.5g} °C) no está más fría que la saturación "
            f"{_del(i.fluid)} a esa presión ({sat.T_sat_K - 273.15:.5g} °C): el vapor no "
            "condensa. Bajá T_s o subí la presión."
        )
    T_tr = PropsSI("Ttriple", i.fluid)
    if (sat.T_sat_K + i.T_s_K) / 2.0 <= T_tr or i.T_s_K <= T_tr:
        raise ValueError(
            f"La pared ({i.T_s_K - 273.15:.5g} °C) está por debajo del punto triple "
            f"{_del(i.fluid)} ({T_tr - 273.15:.4g} °C): el condensado se congelaría (escarcha) "
            "y no hay película de líquido. Subí T_s."
        )
    if i.T_v_K is not None and i.T_v_K < sat.T_sat_K - 1e-9:
        raise ValueError(
            f"El vapor ({i.T_v_K - 273.15:.5g} °C) no puede estar más frío que la saturación "
            f"({sat.T_sat_K - 273.15:.5g} °C): a esa presión sería líquido."
        )


def _vertical(
    sat: SaturationState, liq: PhaseProperties, h_fg: float, dT: float, L: float
) -> tuple[FilmRegime, float, float]:
    """Una pared vertical de alto L: (régimen, Re, h) con el procedimiento de Çengel y Ghajar:
    se supone la película con ondas y, si el Re sale de 30 a 1800, se pasa a la laminar o a la
    turbulenta."""
    Re = re_wavy(liq.k, liq.mu, liq.nu, h_fg, dT, L)
    if Re < RE_LAMINAR:
        h = h_laminar_vertical(liq.rho, sat.rho_v, h_fg, liq.k, liq.mu, dT, L)
        return "laminar", 4.0 * L * h * dT / (liq.mu * h_fg), h
    if Re <= RE_TURBULENT:
        return "wavy", Re, h_wavy(Re, liq.k, liq.nu)
    Re = re_turbulent(liq.k, liq.mu, liq.nu, liq.Pr, h_fg, dT, L)
    return "turbulent", Re, h_turbulent(Re, liq.k, liq.nu, liq.Pr)


def solve_condensation(inputs: CondensationInputs) -> CondensationResult:
    """El coeficiente de condensación en película, el calor y el caudal que condensa
    (Çengel y Ghajar §10-5 a §10-7; Incropera §10.7 a §10.10)."""
    i = inputs
    _validate(i)
    sat = saturation_state(i.fluid, i.p_Pa)
    _check_temperatures(i, sat)
    dT = sat.T_sat_K - i.T_s_K
    liq = phase_properties(i.fluid, (sat.T_sat_K + i.T_s_K) / 2.0, sat.p_Pa, "liquid")
    superheat = 0.0 if i.T_v_K is None else max(i.T_v_K - sat.T_sat_K, 0.0)
    vapor = (
        phase_properties(i.fluid, sat.T_sat_K + superheat / 2.0, sat.p_Pa, "vapor")
        if superheat > 0.0
        else None
    )
    cp_v = vapor.cp if vapor is not None else 0.0
    h_fg = h_star(sat.h_fg, liq.cp, dT, cp_v, superheat)

    regime: FilmRegime = "laminar"
    Re: float | None = None
    h_single: float | None = None
    delta: float | None = None
    Re_vapor: float | None = None
    g = i.geometry
    if g in _TALL:
        regime, Re, h = _vertical(sat, liq, h_fg, dT, i.L_m)
        g_eff = G
        if g == "inclined_plate":
            factor = math.cos(math.radians(i.angle_deg)) ** 0.25
            h, Re = h * factor, Re * factor
            g_eff = G * math.cos(math.radians(i.angle_deg))
        delta = film_thickness(liq.rho, sat.rho_v, h_fg, liq.k, liq.mu, dT, i.L_m, g_eff)
    elif g in ("horizontal_tube", "tube_bank"):
        h = h_horizontal_tube(liq.rho, sat.rho_v, h_fg, liq.k, liq.mu, dT, i.D_m)
        if g == "tube_bank":
            h_single, h = h, h * i.rows**-0.25
    elif g == "sphere":
        h = h_sphere(liq.rho, sat.rho_v, h_fg, liq.k, liq.mu, dT, i.D_m)
    else:
        # Chato usa su propio calor latente: h_fg + ⅜·c_pℓ·ΔT (+ el sobrecalentamiento).
        h = h_chato(
            liq.rho, sat.rho_v, sat.h_fg + cp_v * superheat, liq.cp, liq.k, liq.mu, dT, i.D_m
        )
        if i.V_v_m_s is not None:
            mu_v = PropsSI("V", "P", sat.p_Pa, "Q", 1.0, i.fluid)
            Re_vapor = sat.rho_v * i.V_v_m_s * i.D_m / mu_v

    result = CondensationResult(
        inputs=i,
        sat=sat,
        liquid=liq,
        vapor=vapor,
        dT_K=dT,
        superheat_K=superheat,
        h_fg_star=h_fg,
        regime=regime,
        Re=Re,
        h=h,
        h_single=h_single,
        area_m2=_area(i),
        delta_m=delta,
        Re_vapor=Re_vapor,
        h_dropwise=dropwise_h(sat.T_sat_K) if i.fluid == "Water" else None,
        notes=(),
    )
    return replace(result, notes=_notes(result))


def _del(fluid: str) -> str:
    return "del " + with_article(fluid)[3:]


_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _fmt(x: float, sig: int = 3) -> str:
    """Un número para las notas: 8,94, 1930 o 1,26·10⁶."""
    if x == 0.0 or not math.isfinite(x):
        return f"{x:g}"
    e = math.floor(math.log10(abs(x)))
    if -3 <= e < 5:
        text = f"{round(x, sig - 1 - e):.{max(sig - 1 - e, 0)}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
    else:
        m = round(x / 10.0**e, sig - 1)
        if abs(m) >= 10.0:
            m, e = m / 10.0, e + 1
        text = f"{m:.{sig - 1}f}".rstrip("0").rstrip(".") + "·10" + str(e).translate(_SUPERSCRIPTS)
    return text.replace(".", ",").replace("-", "−")


def _notes(r: CondensationResult) -> tuple[str, ...]:
    i, notes = r.inputs, []
    g = i.geometry
    if g in _TALL:
        assert r.Re is not None
        regime = FILM_REGIMES[r.regime]
        if r.regime == "laminar":
            notes.append(
                f"Re = {_fmt(r.Re, 4)} < 30: película laminar y lisa (Nusselt, 1916). Con ondas "
                "(Re > 30) el coeficiente real sube."
            )
        elif r.regime == "wavy":
            notes.append(
                f"Re = {_fmt(r.Re, 4)}: película {regime} (30 < Re < 1800, Kutateladze). Las "
                "ondas adelgazan la película y suben h respecto de Nusselt."
            )
        else:
            notes.append(f"Re = {_fmt(r.Re, 4)} > 1800: película turbulenta (Labuntsov).")
            if r.Re < RE_TURBULENT:
                notes.append(
                    "La forma con ondas daba Re > 1800 pero la turbulenta da algo menos: el "
                    "límite entre las dos depende de la correlación (con Pr chico pasa)."
                )
        if g == "inclined_plate":
            notes.append(
                f"Inclinada {_fmt(i.angle_deg)}° de la vertical: la gravedad que hace escurrir "
                "la película es g·cos θ, así que h baja como (cos θ)^¼. Çengel y Ghajar lo "
                "deducen para la película laminar y lo usan también con ondas."
            )
        if g == "vertical_tube":
            assert r.delta_m is not None
            ratio = i.D_m / r.delta_m
            notes.append(
                "Un tubo vertical se trata como una placa de ancho π·D: vale si la película es "
                f"mucho más fina que el diámetro (D/δ ≈ {_fmt(ratio)})."
            )
        if r.regime != "laminar" and r.delta_m is not None:
            notes.append(
                "El espesor δ es el de la película laminar de Nusselt: con ondas o turbulencia "
                "es una estimación."
            )
    if g == "tube_bank":
        notes.append(
            f"En el banco el condensado de cada tubo cae sobre el de abajo: con {i.rows} tubos "
            f"por columna, h_N = h_1·N^−¼ = {_fmt(i.rows**-0.25)}·h_1 (Çengel y Ghajar §10-5)."
        )
    if g == "inside_tube":
        if r.Re_vapor is None:
            notes.append(
                "Chato vale con poco vapor: Re_v = ρ_v·V_v·D/μ_v < 35 000 a la entrada del "
                "tubo. Cargá la velocidad del vapor para verificarlo."
            )
        elif r.Re_vapor > RE_VAPOR_CHATO:
            notes.append(
                f"Re_v = {_fmt(r.Re_vapor)} > 35 000: con tanto vapor el condensado forma una "
                "película anular y Chato no vale (Çengel y Ghajar §10-6). El h es una "
                "estimación (por lo bajo)."
            )
        else:
            notes.append(f"Re_v = {_fmt(r.Re_vapor)} < 35 000: dentro del rango de Chato.")
    if r.superheat_K > 0.0:
        assert r.vapor is not None
        notes.append(
            f"El vapor llega {_fmt(r.superheat_K)} K sobrecalentado: h*_fg suma "
            f"c_pv·(T_v − T_sat) = {_fmt(r.vapor.cp * r.superheat_K / 1e3)} kJ/kg y la fuerza "
            "impulsora sigue siendo T_sat − T_s (Çengel y Ghajar §10-5)."
        )
    if r.h_dropwise is not None:
        notes.append(
            f"Con condensación en gotas (una superficie que el agua no moja), Griffith da "
            f"h ≈ {_fmt(r.h_dropwise)} W/(m²·K), {_fmt(r.h_dropwise / r.h, 2)} veces la "
            "película. No dura: los promotores se gastan y el diseño supone película "
            "(Çengel y Ghajar §10-7)."
        )
    if i.fluid != "Water":
        T_rel = r.sat.T_sat_K / PropsSI("Tcrit", i.fluid)
        if T_rel > 0.9:
            notes.append(
                f"T_sat/T_cr = {_fmt(T_rel)}: cerca del punto crítico ρ_v deja de ser chica "
                "frente a ρ_ℓ y las formas explícitas (que la desprecian) pierden precisión."
            )
    return tuple(notes)


# ---------------------------------------------------------------------
# Comparación de geometrías y curvas
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRow:
    """Una geometría con las mismas temperaturas: h, el flujo y el régimen."""

    key: str
    name: str
    h: float
    q: float
    regime: FilmRegime
    Re: float | None
    current: bool


def _comparison_set(i: CondensationInputs) -> list[tuple[str, str, CondensationInputs]]:
    g = i.geometry
    rows: list[tuple[str, str, CondensationInputs]] = []
    if g in ("vertical_plate", "inclined_plate"):
        L = _fmt(i.L_m)
        rows.append(
            ("vertical", f"Vertical ({L} m de alto)", replace(i, geometry="vertical_plate"))
        )
        angles = sorted({30.0, 60.0} | ({i.angle_deg} if g == "inclined_plate" else set()))
        for a in angles:
            rows.append(
                (
                    f"inclined_{a:g}",
                    f"Inclinada {_fmt(a)}°",
                    replace(i, geometry="inclined_plate", angle_deg=a),
                )
            )
        return rows
    if g == "sphere":
        D = _fmt(i.D_m * 1000.0)
        return [
            ("sphere", f"Esfera de {D} mm", i),
            (
                "horizontal_tube",
                f"Tubo horizontal de {D} mm",
                replace(i, geometry="horizontal_tube"),
            ),
        ]
    D, L = _fmt(i.D_m * 1000.0), _fmt(i.L_m)
    rows = [
        ("vertical_tube", f"Tubo vertical ({L} m de alto)", replace(i, geometry="vertical_tube")),
        (
            "horizontal_tube",
            f"Tubo horizontal (D = {D} mm)",
            replace(i, geometry="horizontal_tube"),
        ),
    ]
    if g == "tube_bank" and i.rows > 1:
        rows.append(("tube_bank", f"Banco, {i.rows} tubos por columna", i))
    rows.append(("inside_tube", "Adentro del tubo (Chato)", replace(i, geometry="inside_tube")))
    return rows


def geometry_comparison(r: CondensationResult) -> tuple[ComparisonRow, ...]:
    """El mismo vapor y la misma pared en las otras geometrías de su familia: la placa vertical
    contra la inclinada, o el tubo vertical contra el horizontal, el banco y por dentro. Con la
    película laminar, h_vertical/h_horizontal = 1,29·(D/L)^¼ (Çengel y Ghajar §10-5)."""
    out = []
    for key, name, inputs in _comparison_set(r.inputs):
        rr = r if inputs == r.inputs else solve_condensation(inputs)
        out.append(ComparisonRow(key, name, rr.h, rr.q, rr.regime, rr.Re, inputs == r.inputs))
    if not any(row.current for row in out):
        raise AssertionError("la geometría de los datos tiene que estar en la comparación")
    return tuple(out)


@dataclass(frozen=True)
class CondensationCurve:
    """h (y Re en las paredes verticales) contra el alto L o contra el diámetro D."""

    variable: Literal["L", "D"]
    x: tuple[float, ...]
    h: tuple[float, ...]
    Re: tuple[float, ...] | None
    regime: tuple[FilmRegime, ...]


def condensation_curve(r: CondensationResult, points: int = 120) -> CondensationCurve:
    """La placa o el tubo vertical con otros altos (de 1 cm a 10 veces el dato, o hasta pasar
    a la turbulenta) o el tubo y la esfera con otros diámetros (de 2 mm a 10 veces el dato)."""
    i = r.inputs
    tall = i.geometry in _TALL
    if tall:
        lo, hi = 0.01, max(10.0 * i.L_m, 1.0)
    else:
        lo, hi = 0.002, max(10.0 * i.D_m, 0.1)
    xs: list[float] = []
    hs: list[float] = []
    res: list[float] = []
    regimes: list[FilmRegime] = []
    for k in range(points):
        x = lo * (hi / lo) ** (k / (points - 1))
        rr = solve_condensation(replace(i, L_m=x) if tall else replace(i, D_m=x))
        xs.append(x)
        hs.append(rr.h)
        regimes.append(rr.regime)
        if tall:
            assert rr.Re is not None
            res.append(rr.Re)
    return CondensationCurve(
        "L" if tall else "D", tuple(xs), tuple(hs), tuple(res) if tall else None, tuple(regimes)
    )


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CondensationExample:
    description: str
    inputs: CondensationInputs


_ATM = 101_325.0
_C = 273.15

CONDENSATION_EXAMPLES: dict[str, CondensationExample] = {
    "Placa vertical en vapor a 1 atm (Çengel y Ghajar §10-5)": CondensationExample(
        "Vapor de agua saturado a 1 atm condensa sobre una placa vertical de 2 m de alto y 3 m "
        "de ancho a 80 °C. El libro: Re = 1287 (con ondas), h = 5848 W/(m²·K), 702 kW y "
        "0,303 kg/s.",
        CondensationInputs("Water", _ATM, "vertical_plate", 80.0 + _C, L_m=2.0, W_m=3.0),
    ),
    "La misma placa inclinada 30° (Çengel y Ghajar §10-5)": CondensationExample(
        "La placa anterior inclinada 30° respecto de la vertical: h baja como (cos 30°)^¼. El "
        "libro: 677 kW (3,6 % menos).",
        CondensationInputs(
            "Water", _ATM, "inclined_plate", 80.0 + _C, L_m=2.0, W_m=3.0, angle_deg=30.0
        ),
    ),
    "Tubo horizontal del condensador de una central (Çengel y Ghajar §10-5)": CondensationExample(
        "El condensador trabaja a 7,38 kPa (T_sat = 40 °C) y el vapor condensa sobre tubos "
        "horizontales de 3 cm a 30 °C. El libro, por metro de tubo: h = 9292 W/(m²·K), "
        "8,76 kW y 3,6 g/s.",
        CondensationInputs("Water", 7380.0, "horizontal_tube", 30.0 + _C, L_m=1.0, D_m=0.03),
    ),
    "Banco de 3 × 4 tubos (Çengel y Ghajar §10-5)": CondensationExample(
        "Los tubos del condensador anterior en un arreglo de 3 de alto y 4 de ancho: el "
        "condensado de cada tubo cae sobre el de abajo. El libro: h = 7060 W/(m²·K) y "
        "79,8 kW por metro.",
        CondensationInputs(
            "Water", 7380.0, "tube_bank", 30.0 + _C, L_m=1.0, D_m=0.03, rows=3, columns=4
        ),
    ),
    "R-134a adentro de un tubo (ejemplo propio)": CondensationExample(
        "R-134a a 10 bar (T_sat = 39,4 °C) condensa adentro de un tubo horizontal de 1 cm y "
        "2 m de largo con la pared a 30 °C y el vapor entrando a 0,5 m/s.",
        CondensationInputs(
            "R134a", 10.0e5, "inside_tube", 30.0 + _C, L_m=2.0, D_m=0.01, V_v_m_s=0.5
        ),
    ),
    "Vapor sobrecalentado sobre un tubo vertical (ejemplo propio)": CondensationExample(
        "Vapor de agua a 1 atm y 150 °C (50 K de sobrecalentamiento) sobre un tubo vertical de "
        "5 cm y 1,5 m de alto a 90 °C.",
        CondensationInputs(
            "Water", _ATM, "vertical_tube", 90.0 + _C, L_m=1.5, D_m=0.05, T_v_K=150.0 + _C
        ),
    ),
}


# ---------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------


def _q(value: float, kind: Any, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value, kind, system), "unidad": unit_label(kind, system)}


def condensation_to_dict(r: CondensationResult, system: UnitSystem) -> dict[str, Any]:
    """El resultado de la condensación, serializable a JSON y CSV."""
    i, sat, liq = r.inputs, r.sat, r.liquid
    data: dict[str, Any] = {
        "fluido": fluid_name(i.fluid),
        "presion": _q(i.p_Pa, "pressure", system),
        "T_sat": _q(sat.T_sat_K, "temperature", system),
        "geometria": GEOMETRIES[i.geometry],
        "T_pared": _q(i.T_s_K, "temperature", system),
        "Delta_T": _q(r.dT_K, "temperature_difference", system),
        "sobrecalentamiento": _q(r.superheat_K, "temperature_difference", system),
        "h_fg": _q(sat.h_fg, "specific_enthalpy", system),
        "h_fg_corregido": _q(r.h_fg_star, "specific_enthalpy", system),
        "regimen": FILM_REGIMES[r.regime],
        "Re": r.Re,
        "h": _q(r.h, "heat_transfer_coefficient", system),
        "q": _q(r.q, "heat_flux", system),
        "area": _q(r.area_m2, "area", system),
        "Q": _q(r.Q_W, "heat_rate", system),
        "m_condensada": _q(r.m_dot, "mass_flow", system),
        "propiedades_liquido": {
            "T_pelicula": _q(liq.T_K, "temperature", system),
            "rho_l": _q(liq.rho, "density", system),
            "mu_l": _q(liq.mu, "dynamic_viscosity", system),
            "cp_l": _q(liq.cp, "specific_heat", system),
            "k_l": _q(liq.k, "thermal_conductivity", system),
            "Pr_l": liq.Pr,
            "rho_v": _q(sat.rho_v, "density", system),
        },
        "notas": list(r.notes),
    }
    if i.geometry in ("vertical_plate", "inclined_plate"):
        data["L"] = _q(i.L_m, "length", system)
        data["ancho"] = _q(i.W_m, "length", system)
    else:
        data["D"] = _q(i.D_m, "small_length", system)
        if i.geometry != "sphere":
            data["L"] = _q(i.L_m, "length", system)
    if i.geometry == "inclined_plate":
        data["inclinacion_grados"] = i.angle_deg
    if i.geometry == "tube_bank":
        data["tubos_por_columna"] = i.rows
        data["columnas"] = i.columns
        assert r.h_single is not None
        data["h_un_tubo"] = _q(r.h_single, "heat_transfer_coefficient", system)
    if r.delta_m is not None:
        data["espesor_pelicula"] = _q(r.delta_m, "small_length", system)
    if r.Re_vapor is not None:
        data["Re_vapor"] = r.Re_vapor
    if r.h_dropwise is not None:
        data["h_gotas_Griffith"] = _q(r.h_dropwise, "heat_transfer_coefficient", system)
    return data
