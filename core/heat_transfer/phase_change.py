"""Lo común de la ebullición y la condensación (Fase 11.2): los fluidos y sus propiedades.

Las correlaciones de Çengel y Ghajar (2015), cap. 10, e Incropera et al. (2007), cap. 10,
piden las propiedades del líquido y del vapor saturados a T_sat (ρ, h_fg, σ, μ, c_p, k) y,
según el caso, las de una película: el vapor de la ebullición en película a
T_f = (T_s + T_sat)/2 (más caliente que T_sat) o el líquido que condensa a
T_f = (T_sat + T_s)/2 (más frío). Todas salen de CoolProp, con la fase indicada para que una
película a pocos kelvin de la saturación no caiga del otro lado.

Todo en SI y sin Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import CoolProp
from CoolProp.CoolProp import PropsSI

#: La gravedad de los ejemplos de Çengel y Ghajar (2015), cap. 10.
G = 9.81

#: Los fluidos puros con tensión superficial y propiedades de transporte en CoolProp que
#: ofrece la página (los pseudo-puros, como el R-410A, cambian de fase con deslizamiento).
PHASE_CHANGE_FLUIDS: dict[str, str] = {
    "Water": "Agua",
    "n-Pentane": "n-Pentano",
    "Benzene": "Benceno",
    "Ethanol": "Etanol",
    "R134a": "R-134a",
    "Ammonia": "Amoníaco (R-717)",
    "Propane": "Propano (R-290)",
    "IsoButane": "Isobutano (R-600a)",
    "Nitrogen": "Nitrógeno",
}

_ARTICLES: dict[str, str] = {
    "Water": "el agua",
    "n-Pentane": "el n-pentano",
    "Benzene": "el benceno",
    "Ethanol": "el etanol",
    "R134a": "el R-134a",
    "Ammonia": "el amoníaco",
    "Propane": "el propano",
    "IsoButane": "el isobutano",
    "Nitrogen": "el nitrógeno",
}

Phase = Literal["liquid", "vapor"]


def fluid_name(fluid: str) -> str:
    """«Agua», «n-Pentano», …"""
    return PHASE_CHANGE_FLUIDS.get(fluid, fluid)


def with_article(fluid: str) -> str:
    """«el agua», «el n-pentano», … para usar dentro de una oración."""
    return _ARTICLES.get(fluid, f"el fluido {fluid}")


def _check_fluid(fluid: str) -> None:
    if fluid not in PHASE_CHANGE_FLUIDS:
        raise ValueError(
            f"El fluido {fluid!r} no está en la lista de la página: hacen falta la tensión "
            f"superficial y las propiedades de transporte de CoolProp. Elegí uno de "
            f"{', '.join(PHASE_CHANGE_FLUIDS.values())}."
        )


def pressure_limits(fluid: str) -> tuple[float, float]:
    """(p del punto triple, p crítica) en Pa: la saturación existe entre las dos."""
    _check_fluid(fluid)
    return PropsSI("ptriple", fluid), PropsSI("pcrit", fluid)


def _check_pressure(fluid: str, p_Pa: float) -> None:
    p_tr, p_cr = pressure_limits(fluid)
    if not p_Pa > 0.0:
        raise ValueError(f"La presión tiene que ser positiva (p = {p_Pa:g} Pa).")
    if p_Pa >= p_cr:
        raise ValueError(
            f"A {p_Pa / 1e5:.4g} bar {with_article(fluid)} está por encima de su presión "
            f"crítica ({p_cr / 1e5:.4g} bar): no hay líquido y vapor que convivan, así que no "
            "hay ebullición ni condensación. Bajá la presión."
        )
    if p_Pa <= p_tr * 1.0001:
        raise ValueError(
            f"A {p_Pa / 1e5:.4g} bar {with_article(fluid)} está por debajo de su punto triple "
            f"({p_tr / 1e5:.4g} bar): no hay líquido. Subí la presión."
        )


def saturation_pressure(fluid: str, T_K: float) -> float:
    """La presión de saturación a T (para cargar T_sat en vez de p)."""
    _check_fluid(fluid)
    T_tr, T_cr = PropsSI("Ttriple", fluid), PropsSI("Tcrit", fluid)
    if not T_tr < T_K < T_cr:
        raise ValueError(
            f"La temperatura de saturación {with_article(fluid)} va de "
            f"{T_tr - 273.15:.4g} °C (punto triple) a {T_cr - 273.15:.4g} °C (punto crítico); "
            f"cargaste {T_K - 273.15:.4g} °C."
        )
    return float(PropsSI("P", "T", T_K, "Q", 0.0, fluid))


@dataclass(frozen=True)
class SaturationState:
    """El líquido y el vapor saturados a p (Çengel y Ghajar evalúan a T_sat las propiedades
    del líquido en la ebullición, y h_fg y ρ_v en la condensación)."""

    fluid: str
    p_Pa: float
    T_sat_K: float
    rho_l: float
    rho_v: float
    h_fg: float
    sigma: float
    mu_l: float
    cp_l: float
    k_l: float
    beta_l: float

    @property
    def Pr_l(self) -> float:
        return self.mu_l * self.cp_l / self.k_l

    @property
    def nu_l(self) -> float:
        return self.mu_l / self.rho_l

    @property
    def alpha_l(self) -> float:
        return self.k_l / (self.rho_l * self.cp_l)

    @property
    def delta_rho(self) -> float:
        return self.rho_l - self.rho_v


def saturation_state(fluid: str, p_Pa: float) -> SaturationState:
    """Las propiedades de saturación a p (CoolProp)."""
    _check_pressure(fluid, p_Pa)
    T = float(PropsSI("T", "P", p_Pa, "Q", 0.0, fluid))

    def liq(key: str) -> float:
        return float(PropsSI(key, "P", p_Pa, "Q", 0.0, fluid))

    h_g = float(PropsSI("H", "P", p_Pa, "Q", 1.0, fluid))
    return SaturationState(
        fluid=fluid,
        p_Pa=p_Pa,
        T_sat_K=T,
        rho_l=liq("D"),
        rho_v=float(PropsSI("D", "P", p_Pa, "Q", 1.0, fluid)),
        h_fg=h_g - liq("H"),
        sigma=liq("I"),
        mu_l=liq("V"),
        cp_l=liq("C"),
        k_l=liq("L"),
        beta_l=liq("isobaric_expansion_coefficient"),
    )


@dataclass(frozen=True)
class PhaseProperties:
    """Una fase a una temperatura y a la presión de saturación: el vapor de la película en
    la ebullición o el líquido que condensa."""

    phase: Phase
    T_K: float
    rho: float
    mu: float
    k: float
    cp: float

    @property
    def Pr(self) -> float:
        return self.mu * self.cp / self.k

    @property
    def nu(self) -> float:
        return self.mu / self.rho


def phase_properties(fluid: str, T_K: float, p_Pa: float, phase: Phase) -> PhaseProperties:
    """Las propiedades de una fase a (T, p), con la fase impuesta: un vapor a T > T_sat o un
    líquido a T < T_sat (CoolProp no adivina de qué lado está a pocos milikelvin)."""
    _check_fluid(fluid)
    state = CoolProp.AbstractState("HEOS", fluid)
    state.specify_phase(CoolProp.iphase_gas if phase == "vapor" else CoolProp.iphase_liquid)
    T_max = PropsSI("Tmax", fluid)
    if T_K > T_max:
        raise ValueError(
            f"La película {'de vapor' if phase == 'vapor' else 'de líquido'} estaría a "
            f"{T_K - 273.15:.5g} °C, más que la temperatura máxima de la ecuación de estado "
            f"{_del(fluid)} ({T_max - 273.15:.5g} °C). Bajá la temperatura de la superficie."
        )
    try:
        state.update(CoolProp.PT_INPUTS, p_Pa, T_K)
        return PhaseProperties(
            phase=phase,
            T_K=T_K,
            rho=float(state.rhomass()),
            mu=float(state.viscosity()),
            k=float(state.conductivity()),
            cp=float(state.cpmass()),
        )
    except ValueError as exc:
        raise ValueError(
            f"CoolProp no tiene las propiedades {_del(fluid)} "
            f"{'vapor' if phase == 'vapor' else 'líquido'} a {T_K - 273.15:.5g} °C y "
            f"{p_Pa / 1e5:.4g} bar. Probá con una temperatura de la superficie más cercana "
            "a la de saturación."
        ) from exc


def _del(fluid: str) -> str:
    """«del agua», «del n-pentano», …"""
    return "del " + with_article(fluid)[3:]
