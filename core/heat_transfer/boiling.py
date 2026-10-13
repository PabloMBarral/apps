"""Ebullición en pileta (Fase 11.2).

La curva de ebullición (Nukiyama, 1934) con las correlaciones de Çengel y Ghajar (2015) §10-2
e Incropera et al. (2007) §10.4, para un calentador sumergido en líquido saturado a la
presión p (T_sat = T_sat(p)), según el exceso de temperatura ΔT = T_s − T_sat:

- **convección natural**: con ΔT chico no se forman burbujas y el calor sale como en la
  Fase 8.1, con el líquido saturado (la curva sigue a la mayor de la convección natural y
  de Rohsenow);
- **nucleada** (Rohsenow, 1952), con C_sf y n del par fluido–superficie (Vachon, Nix y
  Tanger, 1968; Incropera, tabla 10.1);
- **flujo crítico** (Zuber, 1958), con el C_cr de Lienhard y Dhir (1973) según la geometría
  y su tamaño adimensional L* = L·√(g·(ρ_ℓ − ρ_v)/σ);
- **transición**: no tiene una correlación confiable; la app la dibuja como una recta en
  log–log entre el flujo crítico y el mínimo;
- **flujo mínimo**, el punto de Leidenfrost (Zuber, 1958, con el 0,09 de Berenson, 1961);
- **película**: Bromley (1950) en cilindros y esferas y Berenson (1961) en placas, con el
  vapor a T_f = (T_s + T_sat)/2 y la radiación: q̇ = q̇_película + ¾·q̇_rad.

Todo en SI: temperaturas en K, presiones en Pa, longitudes en m, flujos en W/m².
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from core.heat_transfer.convection import (
    nu_horizontal_cylinder_churchill_chu,
    nu_horizontal_upper,
    nu_sphere_churchill,
)
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
from core.heat_transfer.radiation import SIGMA
from core.units_system import UnitSystem, convert_from_si, unit_label

__all__ = [
    "BOILING_EXAMPLES",
    "BOILING_MODES",
    "HEATERS",
    "REGIMES",
    "SURFACES",
    "BoilingCurve",
    "BoilingExample",
    "BoilingMode",
    "FilmBoiling",
    "Heater",
    "MinimumCase",
    "NaturalConvection",
    "NucleateSurface",
    "PoolBoilingInputs",
    "PoolBoilingResult",
    "Regime",
    "boiling_curve",
    "c_critical",
    "dT_rohsenow",
    "film_coefficient",
    "film_radiation_combined",
    "length_ratio",
    "pool_boiling_to_dict",
    "q_critical",
    "q_film",
    "q_minimum",
    "q_radiation",
    "q_rohsenow",
    "solve_pool_boiling",
    "surfaces_for",
]

Heater = Literal["plate", "cylinder", "sphere"]
BoilingMode = Literal["temperature", "flux", "critical"]
Regime = Literal["natural", "nucleate", "transition", "film"]
#: Cómo sale el punto de Leidenfrost: donde la rama de película baja al flujo mínimo
#: («crossing»), en el flujo crítico si la película ya lo supera ahí («no_transition») o al
#: final del rango si no llega a bajar («beyond_range»).
MinimumCase = Literal["crossing", "no_transition", "beyond_range"]

HEATERS: dict[Heater, str] = {
    "plate": "Placa horizontal (disco) mirando hacia arriba",
    "cylinder": "Cilindro horizontal",
    "sphere": "Esfera",
}
BOILING_MODES: dict[BoilingMode, str] = {
    "temperature": "La temperatura de la superficie (sale q̇)",
    "flux": "El flujo de calor (sale T_s)",
    "critical": "El flujo crítico (sale T_s)",
}
REGIMES: dict[Regime, str] = {
    "natural": "convección natural",
    "nucleate": "ebullición nucleada",
    "transition": "ebullición de transición",
    "film": "ebullición en película",
}

#: Exceso de temperatura más grande que recorre la curva (y que busca la rama de película):
#: con el agua a 1 atm, un flujo apenas mayor que el crítico lleva la superficie a casi
#: 2000 °C en la rama de película.
DT_MAX_SEARCH = 2500.0
#: Exceso de temperatura más chico de la curva.
DT_MIN_CURVE = 0.1


# ---------------------------------------------------------------------
# El par fluido–superficie de Rohsenow
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class NucleateSurface:
    """C_sf y n de Rohsenow para un par fluido–superficie."""

    name: str
    fluid: str
    C_sf: float
    n: float


#: Incropera et al. (2007), tabla 10.1, de Vachon, Nix y Tanger (1968). Çengel y Ghajar
#: (2015), tabla 10-3, redondean algunos (0,0130 en el acero inoxidable pulido
#: mecánicamente, que es el de su ejemplo de la olla).
SURFACES: dict[str, NucleateSurface] = {
    "water_copper_scored": NucleateSurface("Agua – cobre rayado", "Water", 0.0068, 1.0),
    "water_copper_polished": NucleateSurface("Agua – cobre pulido", "Water", 0.0128, 1.0),
    "water_ss_etched": NucleateSurface(
        "Agua – acero inoxidable atacado químicamente", "Water", 0.0133, 1.0
    ),
    "water_ss_mech": NucleateSurface(
        "Agua – acero inoxidable pulido mecánicamente", "Water", 0.0132, 1.0
    ),
    "water_ss_ground": NucleateSurface(
        "Agua – acero inoxidable rectificado y pulido", "Water", 0.0080, 1.0
    ),
    "water_brass": NucleateSurface("Agua – latón", "Water", 0.0060, 1.0),
    "water_nickel": NucleateSurface("Agua – níquel", "Water", 0.0060, 1.0),
    "water_platinum": NucleateSurface("Agua – platino", "Water", 0.0130, 1.0),
    "pentane_copper_polished": NucleateSurface(
        "n-Pentano – cobre pulido", "n-Pentane", 0.0154, 1.7
    ),
    "pentane_copper_lapped": NucleateSurface("n-Pentano – cobre lapeado", "n-Pentane", 0.0049, 1.7),
    "benzene_chromium": NucleateSurface("Benceno – cromo", "Benzene", 0.0101, 1.7),
    "ethanol_chromium": NucleateSurface("Etanol – cromo", "Ethanol", 0.0027, 1.7),
}


def surfaces_for(fluid: str) -> dict[str, NucleateSurface]:
    """Las superficies de la tabla para un fluido (vacío: C_sf y n a elección)."""
    return {key: s for key, s in SURFACES.items() if s.fluid == fluid}


# ---------------------------------------------------------------------
# Las correlaciones (con las propiedades explícitas, para reproducir los libros)
# ---------------------------------------------------------------------


def q_rohsenow(
    mu_l: float,
    h_fg: float,
    rho_l: float,
    rho_v: float,
    sigma: float,
    cp_l: float,
    Pr_l: float,
    C_sf: float,
    n: float,
    dT: float,
    g: float = G,
) -> float:
    """Ebullición nucleada (Rohsenow, 1952, Trans. ASME 74:969; Çengel y Ghajar §10-2):
    q̇ = μ_ℓ·h_fg·[g·(ρ_ℓ − ρ_v)/σ]^½·[c_pℓ·ΔT/(C_sf·h_fg·Pr_ℓ^n)]³, con las propiedades del
    líquido a T_sat. Error típico: ±100 % en q̇ con ΔT dado y ±30 % en ΔT con q̇ dado
    (Incropera §10.4)."""
    return (
        mu_l
        * h_fg
        * math.sqrt(g * (rho_l - rho_v) / sigma)
        * (cp_l * dT / (C_sf * h_fg * Pr_l**n)) ** 3
    )


def dT_rohsenow(
    mu_l: float,
    h_fg: float,
    rho_l: float,
    rho_v: float,
    sigma: float,
    cp_l: float,
    Pr_l: float,
    C_sf: float,
    n: float,
    q: float,
    g: float = G,
) -> float:
    """Rohsenow despejado: el exceso de temperatura para un flujo q̇ dado,
    ΔT = (C_sf·h_fg·Pr_ℓ^n/c_pℓ)·{q̇/[μ_ℓ·h_fg·√(g·(ρ_ℓ − ρ_v)/σ)]}^⅓."""
    return (C_sf * h_fg * Pr_l**n / cp_l) * (
        q / (mu_l * h_fg * math.sqrt(g * (rho_l - rho_v) / sigma))
    ) ** (1.0 / 3.0)


def length_ratio(L: float, rho_l: float, rho_v: float, sigma: float, g: float = G) -> float:
    """El tamaño adimensional del calentador, L* = L·√(g·(ρ_ℓ − ρ_v)/σ): L sobre la longitud
    capilar (Lienhard y Dhir, 1973)."""
    return L * math.sqrt(g * (rho_l - rho_v) / sigma)


#: Los casos de la tabla de Lienhard y Dhir (1973) (Çengel y Ghajar, tabla 10-4): el texto
#: de C_cr y el rango de L*.
CRITICAL_CASES: dict[str, tuple[str, str]] = {
    "plate": ("Placa horizontal grande", "L = el diámetro; L* > 27"),
    "cylinder_large": ("Cilindro horizontal grande", "L = el radio; L* > 1,2"),
    "cylinder_small": ("Cilindro horizontal chico", "L = el radio; 0,15 < L* < 1,2"),
    "sphere_large": ("Esfera grande", "L = el radio; L* > 4,26"),
    "sphere_small": ("Esfera chica", "L = el radio; 0,15 < L* < 4,26"),
}


def c_critical(heater: Heater, L_star: float) -> tuple[float, str]:
    """C_cr del flujo crítico según la geometría (Lienhard y Dhir, 1973, J. Heat Transfer
    95:152; Çengel y Ghajar, tabla 10-4): 0,149 en una placa grande; 0,12 en un cilindro
    grande y 0,12·L*^−¼ en uno chico; 0,11 en una esfera grande y 0,227·L*^−½ en una
    chica. Devuelve (C_cr, el caso)."""
    if heater == "plate":
        return 0.149, "plate"
    if heater == "cylinder":
        if L_star >= 1.2:
            return 0.12, "cylinder_large"
        return 0.12 * L_star**-0.25, "cylinder_small"
    if L_star >= 4.26:
        return 0.11, "sphere_large"
    return 0.227 * L_star**-0.5, "sphere_small"


def q_critical(
    C_cr: float, h_fg: float, rho_l: float, rho_v: float, sigma: float, g: float = G
) -> float:
    """El flujo crítico (Zuber, 1958, AECU-4439; Çengel y Ghajar §10-2):
    q̇_máx = C_cr·h_fg·[σ·g·ρ_v²·(ρ_ℓ − ρ_v)]^¼."""
    return C_cr * h_fg * (sigma * g * rho_v**2 * (rho_l - rho_v)) ** 0.25


def q_minimum(h_fg: float, rho_l: float, rho_v: float, sigma: float, g: float = G) -> float:
    """El flujo mínimo, el punto de Leidenfrost, en una placa horizontal grande (Zuber, 1958,
    con la constante 0,09 de Berenson, 1961; Çengel y Ghajar §10-2):
    q̇_mín = 0,09·ρ_v·h_fg·[σ·g·(ρ_ℓ − ρ_v)/(ρ_ℓ + ρ_v)²]^¼."""
    return 0.09 * rho_v * h_fg * (sigma * g * (rho_l - rho_v) / (rho_l + rho_v) ** 2) ** 0.25


def film_coefficient(heater: Heater) -> float:
    """La constante de la película: 0,62 en un cilindro horizontal y 0,67 en una esfera
    (Bromley, 1950, Chem. Eng. Prog. 46:221; Çengel y Ghajar §10-2) y 0,425 en una placa
    horizontal, con la longitud capilar en vez del diámetro (Berenson, 1961, J. Heat Transfer
    83:351)."""
    return {"plate": 0.425, "cylinder": 0.62, "sphere": 0.67}[heater]


def q_film(
    C_film: float,
    k_v: float,
    rho_v: float,
    rho_l: float,
    h_fg: float,
    cp_v: float,
    mu_v: float,
    D: float,
    dT: float,
    g: float = G,
) -> float:
    """Ebullición en película (Bromley, 1950):
    q̇ = C·[g·k_v³·ρ_v·(ρ_ℓ − ρ_v)·(h_fg + 0,4·c_pv·ΔT)/(μ_v·D·ΔT)]^¼·ΔT, con el vapor a la
    temperatura de la película y el líquido y h_fg a T_sat."""
    return (
        C_film
        * (g * k_v**3 * rho_v * (rho_l - rho_v) * (h_fg + 0.4 * cp_v * dT) / (mu_v * D * dT))
        ** 0.25
        * dT
    )


def film_radiation_combined(q_film: float, q_rad: float) -> float:
    """La película con la radiación en la forma completa de Bromley (1950; Incropera §10.4):
    h^{4/3} = h_película^{4/3} + h_rad·h^{1/3}, con h = q̇/ΔT (el ΔT se simplifica, así que
    vale con los flujos). Con q̇_rad < q̇_película da casi q̇_película + ¾·q̇_rad."""
    if q_rad <= 0.0:
        return q_film

    def f(q: float) -> float:
        return q ** (4.0 / 3.0) - q_film ** (4.0 / 3.0) - q_rad * q ** (1.0 / 3.0)

    return float(brentq(f, q_film, q_film + q_rad, xtol=1e-9 * (q_film + q_rad)))


def q_radiation(emissivity: float, T_s: float, T_sat: float) -> float:
    """La radiación de la superficie al líquido a través del vapor: q̇_rad = ε·σ·(T_s⁴ − T_sat⁴)
    (Çengel y Ghajar §10-2)."""
    return emissivity * SIGMA * (T_s**4 - T_sat**4)


# ---------------------------------------------------------------------
# Datos y resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PoolBoilingInputs:
    """Un calentador sumergido en un líquido saturado a la presión p.

    ``D_m`` es el diámetro (del disco, del cilindro o de la esfera); ``L_m``, el largo del
    cilindro. ``mode`` dice qué se da: la temperatura de la superficie, el flujo o el flujo
    crítico. C_sf y n son los de Rohsenow (los de la tabla o a elección); ``surface`` es la
    clave de :data:`SURFACES` si salen de la tabla (solo para los rótulos).
    """

    fluid: str
    p_Pa: float
    heater: Heater
    D_m: float
    C_sf: float
    n: float
    mode: BoilingMode
    T_s_K: float | None = None
    q_W_per_m2: float | None = None
    L_m: float = 1.0
    emissivity: float = 0.0
    surface: str | None = None


@dataclass(frozen=True)
class NaturalConvection:
    """La convección natural del líquido saturado (antes de que haya burbujas)."""

    L_m: float
    Ra: float
    Nu: float
    h: float
    q: float
    correlation: str


@dataclass(frozen=True)
class FilmBoiling:
    """La ebullición en película: el vapor de la película y los dos aportes."""

    C_film: float
    length_m: float
    vapor: PhaseProperties
    q_film: float
    q_rad: float

    @property
    def q_total(self) -> float:
        return self.q_film + 0.75 * self.q_rad


@dataclass(frozen=True)
class PoolBoilingResult:
    inputs: PoolBoilingInputs
    sat: SaturationState
    area_m2: float
    regime: Regime
    dT_K: float
    q: float
    L_star: float
    C_cr: float
    critical_case: str
    q_max: float
    dT_max: float
    q_min: float
    dT_min: float
    minimum_case: MinimumCase
    dT_onset: float
    nucleate_q: float
    natural: NaturalConvection
    film: FilmBoiling | None
    burnout: bool
    notes: tuple[str, ...]

    @property
    def T_s_K(self) -> float:
        return self.sat.T_sat_K + self.dT_K

    @property
    def h(self) -> float:
        return self.q / self.dT_K

    @property
    def Q_W(self) -> float:
        return self.q * self.area_m2

    @property
    def m_dot(self) -> float:
        """El caudal de líquido que se evapora, Q̇/h_fg."""
        return self.Q_W / self.sat.h_fg

    @property
    def fraction_of_critical(self) -> float:
        return self.q / self.q_max


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _area(heater: Heater, D: float, L: float) -> float:
    if heater == "plate":
        return math.pi * D**2 / 4.0
    if heater == "cylinder":
        return math.pi * D * L
    return math.pi * D**2


def _critical_length(heater: Heater, D: float) -> float:
    """La longitud de L*: el diámetro de la placa o el radio del cilindro y de la esfera."""
    return D if heater == "plate" else D / 2.0


def _natural_convection(
    sat: SaturationState, heater: Heater, D: float, dT: float, g: float = G
) -> NaturalConvection:
    """La convección natural del líquido saturado: la placa con L = A/P = D/4 (Lloyd y Moran),
    el cilindro horizontal (Churchill y Chu) y la esfera (Churchill) de la Fase 8.1."""
    L = D / 4.0 if heater == "plate" else D
    Ra = g * sat.beta_l * dT * L**3 / (sat.nu_l * sat.alpha_l)
    if heater == "plate":
        Nu, name = nu_horizontal_upper(Ra), "Lloyd y Moran (placa caliente hacia arriba)"
    elif heater == "cylinder":
        Nu, name = nu_horizontal_cylinder_churchill_chu(Ra, sat.Pr_l), "Churchill y Chu"
    else:
        Nu, name = nu_sphere_churchill(Ra, sat.Pr_l), "Churchill (esfera)"
    h = Nu * sat.k_l / L
    return NaturalConvection(L, Ra, Nu, h, h * dT, name)


def _film(
    sat: SaturationState, heater: Heater, D: float, dT: float, emissivity: float, g: float = G
) -> FilmBoiling:
    vapor = phase_properties(sat.fluid, sat.T_sat_K + dT / 2.0, sat.p_Pa, "vapor")
    C = film_coefficient(heater)
    length = math.sqrt(sat.sigma / (g * sat.delta_rho)) if heater == "plate" else D
    qf = q_film(C, vapor.k, vapor.rho, sat.rho_l, sat.h_fg, vapor.cp, vapor.mu, length, dT, g)
    qr = q_radiation(emissivity, sat.T_sat_K + dT, sat.T_sat_K)
    return FilmBoiling(C, length, vapor, qf, qr)


def _q_nucleate(sat: SaturationState, C_sf: float, n: float, dT: float) -> float:
    return q_rohsenow(
        sat.mu_l, sat.h_fg, sat.rho_l, sat.rho_v, sat.sigma, sat.cp_l, sat.Pr_l, C_sf, n, dT
    )


def _dT_nucleate(sat: SaturationState, C_sf: float, n: float, q: float) -> float:
    return dT_rohsenow(
        sat.mu_l, sat.h_fg, sat.rho_l, sat.rho_v, sat.sigma, sat.cp_l, sat.Pr_l, C_sf, n, q
    )


def _dT_eos(sat: SaturationState) -> float:
    """El exceso de temperatura más grande con el vapor de la película dentro de la ecuación
    de estado (T_f = T_sat + ΔT/2 ≤ T_máx, con 2 % de margen)."""
    T_max = PropsSI("Tmax", sat.fluid)
    return 2.0 * (T_max - sat.T_sat_K) * 0.98


def _dT_cap(sat: SaturationState) -> float:
    """El exceso de temperatura más grande que acepta la app."""
    return min(DT_MAX_SEARCH, _dT_eos(sat))


def _cap_reason(sat: SaturationState) -> str:
    """Por qué no se puede pasar de :func:`_dT_cap`."""
    if _dT_eos(sat) < DT_MAX_SEARCH:
        return (
            f"el vapor de la película quedaría fuera de la ecuación de estado {_del(sat.fluid)} "
            f"(hasta {PropsSI('Tmax', sat.fluid) - 273.15:.0f} °C)"
        )
    return f"la app llega hasta ΔT = {DT_MAX_SEARCH:.0f} K"


def _validate(i: PoolBoilingInputs) -> None:
    if i.fluid not in PHASE_CHANGE_FLUIDS:
        raise ValueError(f"El fluido {i.fluid!r} no está en la lista de la página.")
    if i.heater not in HEATERS:
        raise ValueError(f"El calentador tiene que ser uno de {', '.join(HEATERS)}.")
    if not i.D_m > 0.0:
        raise ValueError("El diámetro del calentador tiene que ser positivo.")
    if i.heater == "cylinder" and not i.L_m > 0.0:
        raise ValueError("El largo del cilindro tiene que ser positivo.")
    if not (i.C_sf > 0.0 and i.n > 0.0):
        raise ValueError(
            "C_sf y n de Rohsenow tienen que ser positivos: salen de ensayos con cada par "
            "fluido–superficie (si no está en la tabla, cargá valores medidos)."
        )
    if not 0.0 <= i.emissivity <= 1.0:
        raise ValueError("La emisividad de la superficie va de 0 a 1.")
    if i.mode not in BOILING_MODES:
        raise ValueError(f"El dato tiene que ser uno de {', '.join(BOILING_MODES)}.")
    if i.mode == "temperature" and i.T_s_K is None:
        raise ValueError("Falta la temperatura de la superficie.")
    if i.mode == "flux" and (i.q_W_per_m2 is None or not i.q_W_per_m2 > 0.0):
        raise ValueError("El flujo de calor tiene que ser positivo.")


def _leidenfrost(
    sat: SaturationState, heater: Heater, D: float, emissivity: float, q_min: float, lo: float
) -> tuple[float, MinimumCase]:
    """El exceso de temperatura del flujo mínimo: donde la rama de película (que crece con
    ΔT) baja a q̇_mín. Devuelve (ΔT_mín, el caso)."""
    hi = _dT_cap(sat)

    def f(dT: float) -> float:
        return _film(sat, heater, D, dT, emissivity).q_total - q_min

    lo = max(lo, DT_MIN_CURVE)
    if lo >= hi:
        return lo, "beyond_range"
    if f(lo) >= 0.0:
        return lo, "no_transition"
    if f(hi) <= 0.0:
        return hi, "beyond_range"
    return float(brentq(f, lo, hi, xtol=1e-6)), "crossing"


def solve_pool_boiling(inputs: PoolBoilingInputs) -> PoolBoilingResult:
    """El punto de operación de un calentador en ebullición en pileta y los puntos notables
    de su curva (Çengel y Ghajar §10-2; Incropera §10.4)."""
    i = inputs
    _validate(i)
    sat = saturation_state(i.fluid, i.p_Pa)
    area = _area(i.heater, i.D_m, i.L_m)
    L_star = length_ratio(_critical_length(i.heater, i.D_m), sat.rho_l, sat.rho_v, sat.sigma)
    C_cr, case = c_critical(i.heater, L_star)
    q_max = q_critical(C_cr, sat.h_fg, sat.rho_l, sat.rho_v, sat.sigma)
    dT_max = _dT_nucleate(sat, i.C_sf, i.n, q_max)
    q_min = q_minimum(sat.h_fg, sat.rho_l, sat.rho_v, sat.sigma)
    dT_min, minimum_case = _leidenfrost(sat, i.heater, i.D_m, i.emissivity, q_min, dT_max)

    def excess(dT: float) -> float:
        return _q_nucleate(sat, i.C_sf, i.n, dT) - _natural_convection(sat, i.heater, i.D_m, dT).q

    dT_onset = float(brentq(excess, 1e-6, dT_max)) if excess(dT_max) > 0.0 else dT_max

    burnout = False
    if i.mode == "temperature":
        assert i.T_s_K is not None
        dT = i.T_s_K - sat.T_sat_K
        if not dT > 0.0:
            raise ValueError(
                f"La superficie ({i.T_s_K - 273.15:.5g} °C) no está más caliente que la "
                f"saturación {_del(i.fluid)} a esa presión ({sat.T_sat_K - 273.15:.5g} °C): no "
                "hierve. Subí T_s o bajá la presión."
            )
        if dT > _dT_cap(sat):
            raise ValueError(
                f"ΔT = {dT:.5g} K es demasiado: {_cap_reason(sat)}. Bajá la temperatura de la "
                "superficie."
            )
    elif i.mode == "critical":
        dT = dT_max
    else:
        assert i.q_W_per_m2 is not None
        q_target = i.q_W_per_m2
        if q_target <= q_max:

            def rising(dT: float) -> float:
                return (
                    max(
                        _q_nucleate(sat, i.C_sf, i.n, dT),
                        _natural_convection(sat, i.heater, i.D_m, dT).q,
                    )
                    - q_target
                )

            dT = float(brentq(rising, 1e-9, dT_max * (1.0 + 1e-12)))
        else:
            burnout = True
            cap = _dT_cap(sat)

            def falling(dT: float) -> float:
                return _film(sat, i.heater, i.D_m, dT, i.emissivity).q_total - q_target

            if falling(cap) < 0.0:
                raise ValueError(
                    f"Con q̇ = {_fmt(q_target)} W/m², más que el flujo crítico "
                    f"({_fmt(q_max)} W/m²), la superficie salta a la ebullición en película y "
                    f"tendría que pasar de {sat.T_sat_K + cap - 273.15:.0f} °C para entregar ese "
                    f"flujo: se quemaría antes ({_cap_reason(sat)}). Bajá el flujo o subí la "
                    "emisividad."
                )
            dT = float(brentq(falling, max(dT_min, DT_MIN_CURVE), cap, xtol=1e-9))

    natural = _natural_convection(sat, i.heater, i.D_m, dT)
    q_nb = _q_nucleate(sat, i.C_sf, i.n, dT)
    film: FilmBoiling | None = None
    if burnout or (dT >= dT_min and dT > dT_max and minimum_case != "beyond_range"):
        regime: Regime = "film"
        film = _film(sat, i.heater, i.D_m, dT, i.emissivity)
        q = film.q_total
    elif dT > dT_max:
        regime = "transition"
        q = _transition(dT, dT_max, q_max, dT_min, q_min)
    elif q_nb >= natural.q:
        regime = "nucleate"
        q = q_max if i.mode == "critical" else q_nb
    else:
        regime = "natural"
        q = natural.q
    if i.mode == "flux" and not burnout:
        assert i.q_W_per_m2 is not None
        q = i.q_W_per_m2

    result = PoolBoilingResult(
        inputs=i,
        sat=sat,
        area_m2=area,
        regime=regime,
        dT_K=dT,
        q=q,
        L_star=L_star,
        C_cr=C_cr,
        critical_case=case,
        q_max=q_max,
        dT_max=dT_max,
        q_min=q_min,
        dT_min=dT_min,
        minimum_case=minimum_case,
        dT_onset=dT_onset,
        nucleate_q=q_nb,
        natural=natural,
        film=film,
        burnout=burnout,
        notes=(),
    )
    return _with_notes(result)


def _transition(dT: float, dT_max: float, q_max: float, dT_min: float, q_min: float) -> float:
    """La transición como una recta en log–log entre el flujo crítico y el mínimo."""
    t = math.log(dT / dT_max) / math.log(dT_min / dT_max)
    return math.exp(math.log(q_max) + t * (math.log(q_min) - math.log(q_max)))


def _del(fluid: str) -> str:
    return "del " + with_article(fluid)[3:]


_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _fmt(x: float, sig: int = 3) -> str:
    """Un número para las notas: 8,94, 1930 o 1,26·10⁶."""
    if x == 0.0 or not math.isfinite(x):
        return f"{x:g}"
    e = math.floor(math.log10(abs(x)))
    if -3 <= e < 4:
        text = f"{round(x, sig - 1 - e):.{max(sig - 1 - e, 0)}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
    else:
        m = round(x / 10.0**e, sig - 1)
        if abs(m) >= 10.0:
            m, e = m / 10.0, e + 1
        text = f"{m:.{sig - 1}f}".rstrip("0").rstrip(".") + "·10" + str(e).translate(_SUPERSCRIPTS)
    return text.replace(".", ",").replace("-", "−")


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


#: Temperaturas de fusión de los materiales de las resistencias, en K (Incropera, tabla A.1),
#: para la nota del «burnout».
MELTING_POINTS_K: dict[str, float] = {
    "el cobre": 1358.0,
    "el acero inoxidable AISI 304": 1670.0,
    "el nicromo": 1672.0,
    "el níquel": 1728.0,
    "el platino": 2045.0,
}


def _with_notes(r: PoolBoilingResult) -> PoolBoilingResult:
    i, notes = r.inputs, []
    if r.regime == "natural":
        notes.append(
            f"Con ΔT = {_fmt(r.dT_K)} K todavía no se forman burbujas: el calor sale por "
            f"convección natural del líquido (Rohsenow daría {_fmt(r.nucleate_q)} W/m², menos "
            f"que la convección natural). La ebullición nucleada arranca hacia "
            f"ΔT ≈ {_fmt(r.dT_onset)} K."
        )
        if not 1.0e4 <= r.natural.Ra <= 1.0e11 and i.heater == "plate":
            notes.append(
                f"Ra = {_fmt(r.natural.Ra)} está fuera del rango de Lloyd y Moran "
                "(10⁴ a 10¹¹): la convección natural es una estimación."
            )
    if r.regime == "nucleate":
        if i.mode == "critical":
            notes.append(
                "Este es el máximo de la curva: con un poco más de flujo, la superficie no "
                "puede seguir en ebullición nucleada y salta a la rama de película, mucho más "
                "caliente (el «burnout»)."
            )
        else:
            notes.append(
                f"q̇ es el {r.fraction_of_critical * 100:.0f} % del flujo crítico "
                f"({_fmt(r.q_max)} W/m²)."
            )
        notes.append(
            "Rohsenow tiene errores de hasta ±100 % en q̇ para un ΔT dado (±30 % en ΔT para "
            "un q̇ dado): C_sf depende de la rugosidad y de cómo se terminó la superficie "
            "(Incropera §10.4)."
        )
    if r.regime == "transition":
        notes.append(
            "No hay una correlación confiable para la ebullición de transición (Çengel y "
            "Ghajar §10-2): la app interpola en log–log entre el flujo crítico "
            f"(ΔT = {_fmt(r.dT_max)} K) y el mínimo (ΔT = {_fmt(r.dT_min)} K). Con la "
            "temperatura de la superficie fija es un régimen inestable: el flujo baja cuando "
            "la temperatura sube."
        )
    if r.burnout:
        assert i.q_W_per_m2 is not None
        T_s = r.T_s_K - 273.15
        melts = [name for name, T_m in MELTING_POINTS_K.items() if T_m < r.T_s_K]
        notes.append(
            f"q̇ = {_fmt(i.q_W_per_m2)} W/m² supera el flujo crítico ({_fmt(r.q_max)} W/m²): "
            "la superficie no puede seguir en ebullición nucleada y su temperatura salta a la "
            f"rama de película, de ΔT = {_fmt(r.dT_max)} K a ΔT = {_fmt(r.dT_K)} K "
            f"(T_s ≈ {T_s:.0f} °C): el «burnout»."
            + (
                f" A esa temperatura se funden {_join(melts)}: una resistencia eléctrica se "
                "quemaría."
                if melts
                else ""
            )
        )
    if r.film is not None:
        f = r.film
        if f.q_rad > f.q_film:
            notes.append(
                "La radiación supera a la conducción por la película: sumar ¾ de q̇_rad es la "
                "aproximación de Bromley para h_rad < h_película. Con su forma completa, "
                "h^(4/3) = h_película^(4/3) + h_rad·h^(1/3) (Incropera §10.4), a este ΔT "
                f"saldrían {_fmt(film_radiation_combined(f.q_film, f.q_rad))} W/m²."
            )
        elif f.q_rad > 0.0:
            share = f"{0.75 * f.q_rad / f.q_total * 100:.1f}".replace(".", ",")
            notes.append(f"La radiación aporta el {share} % del flujo.")
        if i.heater == "plate":
            notes.append(
                "En una placa horizontal la película va con Berenson (1961): Bromley con "
                "C = 0,425 y la longitud capilar √(σ/(g·Δρ)) en vez del diámetro."
            )
    if i.heater == "plate" and r.L_star < 27.0:
        notes.append(
            f"L* = {_fmt(r.L_star)} < 27: la placa es chica para la correlación de la placa "
            "grande (C_cr = 0,149); el flujo crítico real depende de los bordes del "
            "calentador."
        )
    if i.heater != "plate" and r.L_star < 0.15:
        notes.append(
            f"L* = {_fmt(r.L_star)} < 0,15: el calentador es más chico que el rango de Lienhard "
            "y Dhir; el flujo crítico es una extrapolación."
        )
    if r.minimum_case == "no_transition":
        notes.append(
            "Con este calentador la rama de película (Bromley) ya supera el flujo mínimo de "
            "Zuber en el flujo crítico: el modelo no tiene ebullición de transición y la curva "
            "cae directo a la película."
        )
    elif r.minimum_case == "beyond_range":
        notes.append(
            f"La rama de película no baja al flujo mínimo de Zuber ({_fmt(r.q_min)} W/m²) "
            f"antes de ΔT = {_fmt(r.dT_min)} K ({_cap_reason(r.sat)}): la app no puede ubicar "
            "el punto de Leidenfrost y dibuja la transición hasta ahí."
        )
    if i.heater != "plate" and r.regime in ("transition", "film"):
        notes.append(
            "El flujo mínimo de Zuber es el de una placa horizontal grande: en cilindros y "
            "esferas, el punto de Leidenfrost es una estimación."
        )
    if i.surface is None and not surfaces_for(i.fluid):
        notes.append(
            f"No hay C_sf medido para {with_article(i.fluid)} en la tabla: con valores "
            "supuestos la ebullición nucleada es una estimación."
        )
    return PoolBoilingResult(**{**r.__dict__, "notes": tuple(notes)})


# ---------------------------------------------------------------------
# La curva de ebullición
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BoilingCurve:
    """La curva q̇(ΔT) por regímenes, con los puntos notables."""

    dT: tuple[float, ...]
    q: tuple[float, ...]
    regime: tuple[Regime, ...]
    onset: tuple[float, float]
    critical: tuple[float, float]
    leidenfrost: tuple[float, float]


def boiling_curve(r: PoolBoilingResult, points: int = 240) -> BoilingCurve:
    """La curva de ebullición del calentador (Nukiyama, 1934), en ΔT logarítmico desde
    0,1 K hasta pasar el punto de operación y el de Leidenfrost, con los puntos notables
    (inicio de la nucleada, flujo crítico y mínimo) exactos."""
    i, sat = r.inputs, r.sat
    hi = min(_dT_cap(sat), max(3.0 * r.dT_min, 1.5 * r.dT_K, 10.0 * r.dT_max))
    lo = DT_MIN_CURVE
    grid = {lo * (hi / lo) ** (k / (points - 1)) for k in range(points)}
    grid |= {x for x in (r.dT_onset, r.dT_max, r.dT_min) if lo < x < hi}
    dTs: list[float] = []
    qs: list[float] = []
    regimes: list[Regime] = []
    for dT in sorted(grid):
        if dT <= r.dT_max:
            q_nb = _q_nucleate(sat, i.C_sf, i.n, dT)
            q_nc = _natural_convection(sat, i.heater, i.D_m, dT).q
            q, reg = (q_nb, "nucleate") if q_nb >= q_nc else (q_nc, "natural")
        elif dT < r.dT_min or r.minimum_case == "beyond_range":
            q, reg = _transition(dT, r.dT_max, r.q_max, r.dT_min, r.q_min), "transition"
        else:
            q, reg = _film(sat, i.heater, i.D_m, dT, i.emissivity).q_total, "film"
        dTs.append(dT)
        qs.append(q)
        regimes.append(reg)  # type: ignore[arg-type]
    onset_q = _q_nucleate(sat, i.C_sf, i.n, r.dT_onset)
    return BoilingCurve(
        tuple(dTs),
        tuple(qs),
        tuple(regimes),
        (r.dT_onset, onset_q),
        (r.dT_max, r.q_max),
        (r.dT_min, r.q_min),
    )


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BoilingExample:
    description: str
    inputs: PoolBoilingInputs


_ATM = 101_325.0
_C = 273.15

BOILING_EXAMPLES: dict[str, BoilingExample] = {
    "Olla de acero inoxidable pulido (Çengel y Ghajar §10-2)": BoilingExample(
        "Agua a 1 atm en una olla de 30 cm de diámetro con el fondo a 108 °C. El libro usa "
        "C_sf = 0,0130 para el acero inoxidable pulido mecánicamente (la tabla 10.1 de "
        "Incropera, de Vachon, Nix y Tanger, da 0,0132): 7,21·10⁴ W/m², 5,1 kW y 2,26 g/s.",
        PoolBoilingInputs(
            "Water", _ATM, "plate", 0.30, 0.0130, 1.0, "temperature", T_s_K=108.0 + _C
        ),
    ),
    "Flujo crítico de un calentador niquelado (Çengel y Ghajar §10-2)": BoilingExample(
        "Agua a 1 atm hirviendo sobre una resistencia de níquel de 1 cm de diámetro: el flujo "
        "máximo con el C_cr de un cilindro grande y la temperatura de la superficie en ese "
        "punto (el libro: 1,02 MW/m² y 109 °C).",
        PoolBoilingInputs(
            "Water", _ATM, "cylinder", 0.01, 0.0060, 1.0, "critical", surface="water_nickel"
        ),
    ),
    "Resistencia en ebullición en película (Çengel y Ghajar §10-2)": BoilingExample(
        "Agua a 1 atm sobre una resistencia de acero inoxidable pulido de 5 mm de diámetro, "
        "a 350 °C y con ε = 0,05: ebullición en película con la radiación, por metro de largo.",
        PoolBoilingInputs(
            "Water",
            _ATM,
            "cylinder",
            0.005,
            0.0132,
            1.0,
            "temperature",
            T_s_K=350.0 + _C,
            emissivity=0.05,
            surface="water_ss_mech",
        ),
    ),
    "Agua a 2 bar sobre cobre pulido (ejemplo propio)": BoilingExample(
        "Un fondo de cobre pulido de 20 cm que entrega 400 kW/m² al agua a 2 bar: la "
        "temperatura de la superficie sale de Rohsenow despejado.",
        PoolBoilingInputs(
            "Water",
            2.0e5,
            "plate",
            0.20,
            0.0128,
            1.0,
            "flux",
            q_W_per_m2=4.0e5,
            surface="water_copper_polished",
        ),
    ),
    "Una resistencia que se quema (ejemplo propio)": BoilingExample(
        "La resistencia de níquel del flujo crítico con 1,1 MW/m², apenas más que el máximo: "
        "salta a la rama de película, donde solo la radiación (ε = 0,9) y la conducción por el "
        "vapor sacan el calor.",
        PoolBoilingInputs(
            "Water",
            _ATM,
            "cylinder",
            0.01,
            0.0060,
            1.0,
            "flux",
            q_W_per_m2=1.1e6,
            emissivity=0.9,
            surface="water_nickel",
        ),
    ),
    "n-Pentano sobre cobre pulido (ejemplo propio)": BoilingExample(
        "n-Pentano a 1 atm (hierve a 36 °C) sobre una placa de cobre pulido de 10 cm a 50 °C: "
        "un fluido con n = 1,7 y un flujo crítico mucho menor que el del agua.",
        PoolBoilingInputs(
            "n-Pentane",
            _ATM,
            "plate",
            0.10,
            0.0154,
            1.7,
            "temperature",
            T_s_K=50.0 + _C,
            surface="pentane_copper_polished",
        ),
    ),
}


# ---------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------


def _q(value: float, kind: Any, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value, kind, system), "unidad": unit_label(kind, system)}


def pool_boiling_to_dict(r: PoolBoilingResult, system: UnitSystem) -> dict[str, Any]:
    """El resultado de la ebullición en pileta, serializable a JSON y CSV."""
    i, sat = r.inputs, r.sat
    data: dict[str, Any] = {
        "fluido": fluid_name(i.fluid),
        "presion": _q(i.p_Pa, "pressure", system),
        "T_sat": _q(sat.T_sat_K, "temperature", system),
        "calentador": HEATERS[i.heater],
        "D": _q(i.D_m, "small_length", system),
        "superficie": SURFACES[i.surface].name if i.surface else "a elección",
        "C_sf": i.C_sf,
        "n": i.n,
        "dato": BOILING_MODES[i.mode],
        "regimen": REGIMES[r.regime],
        "T_s": _q(r.T_s_K, "temperature", system),
        "Delta_T": _q(r.dT_K, "temperature_difference", system),
        "q": _q(r.q, "heat_flux", system),
        "h": _q(r.h, "heat_transfer_coefficient", system),
        "area": _q(r.area_m2, "area", system),
        "Q": _q(r.Q_W, "heat_rate", system),
        "m_evaporada": _q(r.m_dot, "mass_flow", system),
        "L_estrella": r.L_star,
        "C_cr": r.C_cr,
        "q_max": _q(r.q_max, "heat_flux", system),
        "Delta_T_max": _q(r.dT_max, "temperature_difference", system),
        "q_min": _q(r.q_min, "heat_flux", system),
        "Delta_T_min": _q(r.dT_min, "temperature_difference", system),
        "Delta_T_inicio_nucleada": _q(r.dT_onset, "temperature_difference", system),
        "q_sobre_q_max": r.fraction_of_critical,
        "burnout": r.burnout,
        "propiedades_saturacion": {
            "rho_l": _q(sat.rho_l, "density", system),
            "rho_v": _q(sat.rho_v, "density", system),
            "h_fg": _q(sat.h_fg, "specific_enthalpy", system),
            "sigma_N_por_m": sat.sigma,
            "mu_l": _q(sat.mu_l, "dynamic_viscosity", system),
            "cp_l": _q(sat.cp_l, "specific_heat", system),
            "k_l": _q(sat.k_l, "thermal_conductivity", system),
            "Pr_l": sat.Pr_l,
        },
        "notas": list(r.notes),
    }
    if r.film is not None:
        f = r.film
        data["pelicula"] = {
            "C": f.C_film,
            "T_pelicula": _q(f.vapor.T_K, "temperature", system),
            "q_pelicula": _q(f.q_film, "heat_flux", system),
            "q_radiacion": _q(f.q_rad, "heat_flux", system),
        }
    return data
