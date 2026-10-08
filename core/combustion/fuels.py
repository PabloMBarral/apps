"""Combustibles: mezclas de especies o análisis elemental — Fase 5 (vademecum §16.2 y §16.9).

Tres clases de combustible:

- **Gaseoso**: una mezcla de gases con sus fracciones molares yₖ (el gas
  natural, el GLP, el biogás). El vademecum (§16.2) escribe una reacción por
  componente CₙHₘOₚN_qS_r; acá se suman los átomos: n_C = Σ yₖ·nₖ, etc.
- **Líquido puro**: n-octano, metanol, etanol, querosén (Jet-A) con sus
  polinomios NASA, o propano y butano líquidos (el gas de NASA menos el h_fg de
  CoolProp a la temperatura del combustible).
- **Análisis elemental** (sólidos y líquidos pesados: carbón, bagazo, fueloil):
  las fracciones másicas de C, H, O, N y S, la humedad W y las cenizas A, tal
  cual se queman, y el PCS como dato (las correlaciones para estimarlo son la
  Fase 6). Los resultados van por kg de combustible.

Los combustibles con especies dan los resultados por mol (kmol) de combustible,
como Cengel y el vademecum; su PCS y PCI salen de las entalpías de formación
(vademecum §16.9). Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Literal

import CoolProp
from CoolProp.CoolProp import PropsSI

from core.combustion.thermo import (
    ELEMENTS,
    P_STANDARD_PA,
    R_U,
    T_REF_K,
    atomic_mass,
    species,
    water_hfg_J_per_mol,
)

__all__ = [
    "FUELS",
    "FUEL_KINDS",
    "GAS_COMPONENTS",
    "LIQUID_COMPONENTS",
    "Fuel",
    "FuelKind",
    "UltimateAnalysis",
    "component_entropy",
    "component_enthalpy",
    "component_info",
]

FuelKind = Literal["gas", "liquid", "analysis"]
FUEL_KINDS: dict[FuelKind, str] = {
    "gas": "Mezcla gaseosa",
    "liquid": "Líquido puro",
    "analysis": "Análisis elemental (sólidos y líquidos pesados)",
}

#: Componentes de los combustibles gaseosos (clave de la base NASA → nombre).
GAS_COMPONENTS: dict[str, str] = {
    "CH4": "metano (CH₄)",
    "C2H6": "etano (C₂H₆)",
    "C3H8": "propano (C₃H₈)",
    "n-C4H10": "n-butano (C₄H₁₀)",
    "i-C4H10": "isobutano (C₄H₁₀)",
    "n-C5H12": "n-pentano (C₅H₁₂)",
    "C2H4": "etileno (C₂H₄)",
    "C2H2": "acetileno (C₂H₂)",
    "H2": "hidrógeno (H₂)",
    "CO": "monóxido de carbono (CO)",
    "H2S": "sulfuro de hidrógeno (H₂S)",
    "NH3": "amoníaco (NH₃)",
    "CO2": "dióxido de carbono (CO₂)",
    "N2": "nitrógeno (N₂)",
    "O2": "oxígeno (O₂)",
    "H2O": "vapor de agua (H₂O)",
    "Ar": "argón (Ar)",
}

#: Gases que se queman como líquidos (GLP a presión): el gas de NASA menos el h_fg de CoolProp.
_LIQUEFIED: dict[str, tuple[str, str]] = {
    "C3H8(l)": ("C3H8", "Propane"),
    "n-C4H10(l)": ("n-C4H10", "n-Butane"),
}

#: Combustibles líquidos puros.
LIQUID_COMPONENTS: dict[str, str] = {
    "C8H18(l)": "n-octano (C₈H₁₈)",
    "CH3OH(l)": "metanol (CH₃OH)",
    "C2H5OH(l)": "etanol (C₂H₅OH)",
    "C12H23(l)": "querosén Jet-A (C₁₂H₂₃)",
    "C3H8(l)": "propano líquido (C₃H₈)",
    "n-C4H10(l)": "butano líquido (C₄H₁₀)",
}


# ---------------------------------------------------------------------
# Componentes (especies de NASA y gases licuados)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentInfo:
    key: str
    name: str
    atoms: dict[str, float]
    M_kg_per_mol: float
    hf_J_per_mol: float
    is_gas: bool
    T_min_K: float
    T_max_K: float


@cache
def component_info(key: str) -> ComponentInfo:
    """Fórmula, M, h_f y rango de un componente de combustible.

    Raises
    ------
    ValueError
        Si el componente no está en la base.
    """
    if key in _LIQUEFIED:
        gas_key, fluid = _LIQUEFIED[key]
        gas = species(gas_key)
        T_crit = float(PropsSI("Tcrit", fluid))
        return ComponentInfo(
            key,
            LIQUID_COMPONENTS[key],
            gas.formula,
            gas.M_kg_per_mol,
            gas.hf_J_per_mol - _liquefied_hfg(key, T_REF_K),
            False,
            max(float(PropsSI("Ttriple", fluid)) + 1.0, 250.0),
            T_crit - 5.0,
        )
    sp = species(key)
    name = GAS_COMPONENTS.get(key) or LIQUID_COMPONENTS.get(key) or sp.name
    return ComponentInfo(
        key, name, sp.formula, sp.M_kg_per_mol, sp.hf_J_per_mol, sp.is_gas, sp.T_min_K, sp.T_max_K
    )


def _liquefied_hfg(key: str, T_K: float) -> float:
    """h_fg molar del gas licuado a T (CoolProp, en saturación)."""
    _, fluid = _LIQUEFIED[key]
    state = CoolProp.AbstractState("HEOS", fluid)
    state.update(CoolProp.QT_INPUTS, 1.0, T_K)
    h_v = state.hmolar()
    state.update(CoolProp.QT_INPUTS, 0.0, T_K)
    return h_v - state.hmolar()


def _check_component_T(key: str, T_K: float) -> None:
    info = component_info(key)
    if not (math.isfinite(T_K) and info.T_min_K - 3.0 <= T_K <= info.T_max_K):
        raise ValueError(
            f"El {info.name} solo está tabulado entre {info.T_min_K - 273.15:.0f} °C y "
            f"{info.T_max_K - 273.15:.0f} °C; el combustible a {T_K - 273.15:.1f} °C queda "
            "afuera."
        )


def component_enthalpy(key: str, T_K: float) -> float:
    """Entalpía absoluta del componente (J/mol): h_f + Δh (vademecum §16.6)."""
    if key in _LIQUEFIED:
        gas_key, _ = _LIQUEFIED[key]
        return species(gas_key).h(T_K) - _liquefied_hfg(key, T_K)
    return species(key).h(T_K)


def component_entropy(key: str, T_K: float, p_partial_Pa: float) -> float:
    """Entropía absoluta del componente (J/(mol·K)).

    Gas: s° − R·ln(p_i/p°); líquido: s°(T) (la presión casi no cuenta). Para un
    gas licuado: la del vapor saturado como gas ideal menos h_fg/T.
    """
    if key in _LIQUEFIED:
        gas_key, fluid = _LIQUEFIED[key]
        p_sat = float(PropsSI("P", "T", T_K, "Q", 0, fluid))
        return (
            species(gas_key).s0(T_K)
            - R_U * math.log(p_sat / P_STANDARD_PA)
            - _liquefied_hfg(key, T_K) / T_K
        )
    sp = species(key)
    if not sp.is_gas:
        return sp.s0(T_K)
    return sp.s0(T_K) - R_U * math.log(p_partial_Pa / P_STANDARD_PA)


# ---------------------------------------------------------------------
# Análisis elemental
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class UltimateAnalysis:
    """Análisis elemental en masa, tal cual se quema (fracciones, no porcentajes).

    C, H, O, N y S son los de la materia orgánica; la humedad W (agua líquida) y
    las cenizas A (inertes) van aparte. Tienen que sumar 1.
    """

    C: float
    H: float
    O: float = 0.0  # noqa: E741 (el símbolo del oxígeno)
    N: float = 0.0
    S: float = 0.0
    W: float = 0.0
    A: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "C": self.C,
            "H": self.H,
            "O": self.O,
            "N": self.N,
            "S": self.S,
            "W": self.W,
            "A": self.A,
        }

    def validate(self) -> None:
        values = self.as_dict()
        if any(not math.isfinite(v) or v < 0.0 for v in values.values()):
            raise ValueError("Las fracciones del análisis elemental no pueden ser negativas.")
        total = sum(values.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"El análisis elemental suma {100 * total:.2f} %: C + H + O + N + S + humedad + "
                "cenizas tiene que dar 100 %."
            )
        if self.C + self.H <= 0.0:
            raise ValueError("El combustible no tiene carbono ni hidrógeno: no hay qué quemar.")


# ---------------------------------------------------------------------
# Combustible
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Fuel:
    """Combustible: mezcla de especies (por mol) o análisis elemental (por kg)."""

    name: str
    kind: FuelKind
    components: tuple[tuple[str, float], ...] = ()
    analysis: UltimateAnalysis | None = None
    hhv_J_per_kg: float | None = None
    cp_J_per_kg_K: float | None = None

    # --- construcción -------------------------------------------------
    @classmethod
    def mixture(cls, name: str, fractions: Mapping[str, float], *, kind: FuelKind = "gas") -> Fuel:
        """Combustible gaseoso (o líquido puro) con fracciones molares (se normalizan).

        Raises
        ------
        ValueError
            Si un componente no existe o no corresponde a la clase, si hay
            fracciones negativas o si no queda nada que se queme.
        """
        allowed = GAS_COMPONENTS if kind == "gas" else LIQUID_COMPONENTS
        unknown = [k for k in fractions if k not in allowed]
        if unknown:
            raise ValueError(
                f"Componentes que no son de un combustible {FUEL_KINDS[kind].lower()}: {unknown}."
            )
        if any(not math.isfinite(v) or v < 0.0 for v in fractions.values()):
            raise ValueError("Las fracciones del combustible no pueden ser negativas.")
        total = sum(fractions.values())
        if total <= 0.0:
            raise ValueError("La composición del combustible está vacía.")
        comp = tuple((k, v / total) for k, v in fractions.items() if v > 0.0)
        fuel = cls(name, kind, components=comp)
        o2 = fuel.theoretical_oxygen()
        if o2 <= 1e-12:
            raise ValueError(
                "El combustible no tiene nada que se queme (solo inertes como N₂, CO₂ o agua)."
            )
        return fuel

    @classmethod
    def from_analysis(
        cls,
        name: str,
        analysis: UltimateAnalysis,
        hhv_J_per_kg: float,
        cp_J_per_kg_K: float = 1500.0,
    ) -> Fuel:
        """Combustible por su análisis elemental y su PCS tal cual se quema.

        Raises
        ------
        ValueError
            Si el análisis no suma 1 o el PCS no es positivo.
        """
        analysis.validate()
        if not (math.isfinite(hhv_J_per_kg) and hhv_J_per_kg > 0.0):
            raise ValueError("El PCS del combustible tiene que ser positivo.")
        fuel = cls(
            name,
            "analysis",
            analysis=analysis,
            hhv_J_per_kg=hhv_J_per_kg,
            cp_J_per_kg_K=cp_J_per_kg_K,
        )
        if fuel.lhv_per_kg <= 0.0:
            raise ValueError(
                "Con tanta humedad (y ese PCS) el PCI queda negativo: el combustible no alcanza "
                "a evaporar su propia agua."
            )
        return fuel

    # --- base de cálculo ----------------------------------------------
    @property
    def per_mol(self) -> bool:
        """Los resultados van por mol (especies) o por kg (análisis elemental)."""
        return self.kind != "analysis"

    @property
    def M_kg_per_mol(self) -> float | None:
        """Masa molar (aparente) del combustible; ``None`` para un análisis elemental."""
        if not self.per_mol:
            return None
        return sum(y * component_info(k).M_kg_per_mol for k, y in self.components)

    @property
    def mass_per_basis(self) -> float:
        """kg de combustible por unidad de cálculo (M si va por mol, 1 si va por kg)."""
        return self.M_kg_per_mol if self.per_mol else 1.0  # type: ignore[return-value]

    def elements(self) -> dict[str, float]:
        """Moles de átomos de cada elemento por unidad de cálculo (sin la humedad)."""
        out = dict.fromkeys(ELEMENTS, 0.0)
        if self.per_mol:
            for k, y in self.components:
                for el, n in component_info(k).atoms.items():
                    out[el] += y * n
            return out
        a = self.analysis
        assert a is not None
        for el in ("C", "H", "O", "N", "S"):
            out[el] = getattr(a, el) / atomic_mass(el)
        return out

    @property
    def moisture_mol(self) -> float:
        """Agua de la humedad (mol por kg); cero para las especies."""
        if self.per_mol:
            return 0.0
        assert self.analysis is not None
        return self.analysis.W / species("H2O(l)").M_kg_per_mol

    @property
    def ash_kg(self) -> float:
        """Cenizas por unidad de cálculo (kg); cero para las especies."""
        return 0.0 if self.per_mol else self.analysis.A  # type: ignore[union-attr]

    def theoretical_oxygen(self) -> float:
        """O₂ teórico por unidad de cálculo: n_C + n_H/4 + n_S − n_O/2 (vademecum §16.2)."""
        e = self.elements()
        return e["C"] + e["H"] / 4.0 + e["S"] - e["O"] / 2.0

    def water_formed(self) -> float:
        """Agua en los productos (mol por unidad): n_H/2 + la humedad."""
        return self.elements()["H"] / 2.0 + self.moisture_mol

    # --- energía ----------------------------------------------------------
    def _complete_products_hf(self, water: Literal["H2O", "H2O(l)"]) -> float:
        e = self.elements()
        return (
            e["C"] * species("CO2").hf_J_per_mol
            + self.water_formed() * species(water).hf_J_per_mol
            + e["S"] * species("SO2").hf_J_per_mol
        )

    @property
    def hf_per_basis(self) -> float:
        """Entalpía de formación del combustible a 25 °C (J por unidad de cálculo).

        Especies: Σ yₖ·h_f,k. Análisis elemental: sale del PCS dado,
        h_comb = PCS + Σ ν_p·h_f,p (con el agua líquida, vademecum §16.9).
        """
        if self.per_mol:
            return sum(y * component_info(k).hf_J_per_mol for k, y in self.components)
        assert self.hhv_J_per_kg is not None
        return self.hhv_J_per_kg + self._complete_products_hf("H2O(l)")

    def check_T(self, T_K: float) -> None:
        """Error para el alumno si algún componente está fuera de su rango a T."""
        if self.per_mol:
            for k, _ in self.components:
                _check_component_T(k, T_K)
        elif not (math.isfinite(T_K) and 250.0 <= T_K <= 700.0):
            raise ValueError(
                "La temperatura del combustible tiene que estar entre −23 °C y 427 °C (con su "
                "c_p constante)."
            )

    def enthalpy(self, T_K: float) -> float:
        """Entalpía del combustible a T (J por unidad de cálculo): h_f + Δh."""
        if self.per_mol:
            return sum(y * component_enthalpy(k, T_K) for k, y in self.components)
        cp = self.cp_J_per_kg_K or 0.0
        return self.hf_per_basis + cp * (T_K - T_REF_K)

    def entropy(self, T_K: float, p_Pa: float) -> float | None:
        """Entropía del combustible (J/K por mol) o ``None`` para un análisis elemental.

        Un gas: mezcla de gases ideales a p (cada uno a su presión parcial).
        """
        if not self.per_mol:
            return None
        return sum(y * component_entropy(k, T_K, y * p_Pa) for k, y in self.components)

    @property
    def hhv_per_basis(self) -> float:
        """PCS por unidad de cálculo (J): el agua de los productos líquida (vademecum §16.9)."""
        if not self.per_mol:
            return self.hhv_J_per_kg  # type: ignore[return-value]
        return self.hf_per_basis - self._complete_products_hf("H2O(l)")

    @property
    def lhv_per_basis(self) -> float:
        """PCI por unidad de cálculo (J): PCS − (agua formada)·h_fg(25 °C)."""
        return self.hhv_per_basis - self.water_formed() * water_hfg_J_per_mol()

    @property
    def hhv_per_kg(self) -> float:
        return self.hhv_per_basis / self.mass_per_basis

    @property
    def lhv_per_kg(self) -> float:
        return self.lhv_per_basis / self.mass_per_basis

    @property
    def formula_label(self) -> str:
        """Fórmula para mostrar: la del componente si es puro, si no el nombre."""
        if self.per_mol and len(self.components) == 1:
            return component_info(self.components[0][0]).name
        return self.name


# ---------------------------------------------------------------------
# Biblioteca de combustibles
# ---------------------------------------------------------------------

#: Bagazo seco (Hugot, *Handbook of Cane Sugar Engineering*, 3.ª ed., 1986, §32):
#: C 47 %, H 6,5 %, O 44 %, N 0,3 %, cenizas 2,2 %; PCS seco ≈ 19 250 kJ/kg.
_BAGASSE_DRY = {"C": 0.47, "H": 0.065, "O": 0.44, "N": 0.003, "A": 0.022}


def _bagasse(moisture: float) -> UltimateAnalysis:
    dry = 1.0 - moisture
    return UltimateAnalysis(
        C=_BAGASSE_DRY["C"] * dry,
        H=_BAGASSE_DRY["H"] * dry,
        O=_BAGASSE_DRY["O"] * dry,
        N=_BAGASSE_DRY["N"] * dry,
        W=moisture,
        A=_BAGASSE_DRY["A"] * dry,
    )


FUELS: dict[str, Fuel] = {
    "Metano": Fuel.mixture("Metano", {"CH4": 1.0}),
    "Gas natural típico": Fuel.mixture(
        "Gas natural típico",
        {"CH4": 0.91, "C2H6": 0.05, "C3H8": 0.01, "n-C4H10": 0.005, "N2": 0.01, "CO2": 0.015},
    ),
    "Propano": Fuel.mixture("Propano", {"C3H8": 1.0}),
    "GLP (60 % propano y 40 % butano)": Fuel.mixture(
        "GLP (60 % propano y 40 % butano)", {"C3H8": 0.6, "n-C4H10": 0.4}
    ),
    "Hidrógeno": Fuel.mixture("Hidrógeno", {"H2": 1.0}),
    "Biogás (60 % CH₄ y 40 % CO₂)": Fuel.mixture(
        "Biogás (60 % CH₄ y 40 % CO₂)", {"CH4": 0.6, "CO2": 0.4}
    ),
    "n-Octano (líquido)": Fuel.mixture("n-Octano (líquido)", {"C8H18(l)": 1.0}, kind="liquid"),
    "Metanol (líquido)": Fuel.mixture("Metanol (líquido)", {"CH3OH(l)": 1.0}, kind="liquid"),
    "Etanol (líquido)": Fuel.mixture("Etanol (líquido)", {"C2H5OH(l)": 1.0}, kind="liquid"),
    "Querosén Jet-A (líquido)": Fuel.mixture(
        "Querosén Jet-A (líquido)", {"C12H23(l)": 1.0}, kind="liquid"
    ),
    "Propano líquido": Fuel.mixture("Propano líquido", {"C3H8(l)": 1.0}, kind="liquid"),
    # Fueloil pesado: análisis típico; PCS con Channiwala & Parikh (2002), Fuel 81, 1051.
    "Fueloil": Fuel.from_analysis(
        "Fueloil",
        UltimateAnalysis(C=0.856, H=0.105, O=0.006, N=0.005, S=0.025, W=0.002, A=0.001),
        42.4e6,
        cp_J_per_kg_K=2000.0,
    ),
    # Carbón de Pensilvania (análisis de Cengel, cap. 15); PCS ≈ fórmula de Dulong.
    "Carbón (Pensilvania)": Fuel.from_analysis(
        "Carbón (Pensilvania)",
        UltimateAnalysis(C=0.8436, H=0.0189, O=0.0440, N=0.0063, S=0.0089, A=0.0783),
        30.5e6,
        cp_J_per_kg_K=1300.0,
    ),
    "Bagazo de caña (50 % de humedad)": Fuel.from_analysis(
        "Bagazo de caña (50 % de humedad)", _bagasse(0.5), 0.5 * 19.25e6
    ),
}
