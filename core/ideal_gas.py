"""Mezclas de gases ideales: aire y gases de combustión — Fases 3.3 y 3.4.

El aire y los gases de combustión (N₂, O₂, CO₂, H₂O y Ar) se tratan como
**mezclas de gases ideales** (vademecum §5): la entalpía y la entropía de la
mezcla son la suma de las de cada componente pesadas con las fracciones
másicas. La de cada componente sale del gas ideal de CoolProp: cada uno a
1 Pa, donde la ecuación de estado coincide con la de gas ideal (equivale a
los polinomios NASA del vademecum §4.8, que dan c_p, h y s° en función de T).

- **Entalpía** h_g(T): medida desde 25 °C, la temperatura de referencia de
  los poderes caloríficos; así el balance de una cámara de combustión se
  escribe con el PCI.
- **Función s°(T)**: la de las tablas de gas ideal de Cengel (A-17 para el
  aire), medida desde 25 °C. Entre dos estados de la misma mezcla,
  s₂ − s₁ = s°(T₂) − s°(T₁) − R·ln(p₂/p₁): con eso salen las compresiones y
  expansiones isoentrópicas.
- **Entropía absoluta** s(T, p): con las entropías estándar a 25 °C y 1 bar
  de NIST-JANAF (Chase, 1998) y el término de mezcla. Sirve para dibujar en
  un mismo diagrama T–s el aire y los gases de combustión, que tienen
  distinta composición.

Lo usan la HRSG (:mod:`core.cycles.hrsg`) y la turbina de gas
(:mod:`core.cycles.brayton`). Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Literal

import CoolProp
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from core.fluids import fluid_limits

__all__ = [
    "AIR_DRY",
    "AIR_TECHNICAL",
    "GAS_NAMES",
    "GAS_SPECIES",
    "P_STANDARD_PA",
    "R_U",
    "S_STANDARD_J_PER_MOL_K",
    "T_GAS_MAX_K",
    "FlueGas",
    "combustion_products",
    "exhaust_composition",
    "molar_mass",
]

#: Componentes de los gases (fórmula → nombre de CoolProp).
GAS_SPECIES: tuple[str, ...] = ("N2", "O2", "CO2", "H2O", "Ar")
_COOLPROP_NAMES: dict[str, str] = {
    "N2": "Nitrogen",
    "O2": "Oxygen",
    "CO2": "CarbonDioxide",
    "H2O": "Water",
    "Ar": "Argon",
}
#: Fórmulas para mostrar.
GAS_NAMES: dict[str, str] = {"N2": "N₂", "O2": "O₂", "CO2": "CO₂", "H2O": "H₂O", "Ar": "Ar"}

#: Constante universal de los gases, J/(mol·K) (vademecum §4.1).
R_U = 8.314462618

#: Presión del estado estándar de las entropías absolutas (1 bar, NIST-JANAF).
P_STANDARD_PA = 1.0e5
#: Entropía absoluta molar de cada componente a 25 °C y 1 bar, J/(mol·K)
#: (NIST-JANAF Thermochemical Tables, 4.ª ed., Chase, 1998; Cengel A-26 da
#: casi lo mismo a 1 atm).
S_STANDARD_J_PER_MOL_K: dict[str, float] = {
    "N2": 191.609,
    "O2": 205.147,
    "CO2": 213.795,
    "H2O": 188.834,
    "Ar": 154.845,
}

# Límite de gas ideal: cada componente a 1 Pa (desvío de la ecuación de estado < 1e-8).
_P_IDEAL_PA = 1.0
# Las entalpías y la función s° se miden desde 25 °C (cero ahí).
_T_H_REF_K = 298.15
#: Temperatura máxima de los gases que acepta el modelo (1700 °C).
T_GAS_MAX_K = 1973.15
# CoolProp llega hasta 2000 K en los cinco componentes.
_T_SEARCH_MAX_K = T_GAS_MAX_K + 20.0

# Aire técnico simplificado del vademecum (§16.1): 21 % O₂ y 79 % N₂ en moles.
_AIR_O2 = 0.21
_AIR_N2 = 0.79


@cache
def molar_mass(species: str) -> float:
    """Masa molar del componente (kg/mol)."""
    return float(PropsSI("M", _COOLPROP_NAMES[species]))


@cache
def _T_min_species(species: str) -> float:
    """Temperatura mínima de CoolProp para el componente (su punto triple), con margen."""
    return fluid_limits(_COOLPROP_NAMES[species]).T_min_K + 0.01


@cache
def _at_minimum(species: str) -> tuple[float, float, float]:
    """h, s y c_p de gas ideal del componente a su temperatura mínima (1 Pa)."""
    state = CoolProp.AbstractState("HEOS", _COOLPROP_NAMES[species])
    state.update(CoolProp.PT_INPUTS, _P_IDEAL_PA, _T_min_species(species))
    return state.hmass(), state.smass(), state.cpmass()


def _below_minimum(species: str, T: float, what: Literal["h", "cp", "s"]) -> float:
    """Gas ideal por debajo de la temperatura mínima de CoolProp, con c_p constante.

    CoolProp no evalúa el agua por debajo de su punto triple (0,01 °C). Como gas
    ideal, su c_p casi no cambia ahí (1,86 kJ/(kg·K)): h = h_mín + c_p·(T − T_mín)
    y s = s_mín + c_p·ln(T/T_mín). Hace falta para el estado muerto de la
    exergía de los gases con un ambiente bajo cero.
    """
    h_m, s_m, cp_m = _at_minimum(species)
    T_m = _T_min_species(species)
    if what == "h":
        return h_m + cp_m * (T - T_m)
    if what == "s":
        return s_m + cp_m * math.log(T / T_m)
    return cp_m


def _ideal_gas(species: str, temps: Sequence[float], what: Literal["h", "cp", "s"]) -> np.ndarray:
    """h, c_p o s de gas ideal (J/kg, J/(kg·K)) de un componente a varias temperaturas."""
    state = CoolProp.AbstractState("HEOS", _COOLPROP_NAMES[species])
    T_min = _T_min_species(species)
    out = np.empty(len(temps))
    for k, T in enumerate(temps):
        if T < T_min:
            out[k] = _below_minimum(species, float(T), what)
            continue
        state.update(CoolProp.PT_INPUTS, _P_IDEAL_PA, float(T))
        out[k] = state.hmass() if what == "h" else state.cpmass() if what == "cp" else state.smass()
    return out


@cache
def _h_reference(species: str) -> float:
    """Entalpía de gas ideal del componente a 25 °C (la referencia de h_g)."""
    return float(_ideal_gas(species, (_T_H_REF_K,), "h")[0])


@cache
def _s_reference(species: str) -> float:
    """Entropía de gas ideal del componente a 25 °C y 1 Pa (la referencia de s°)."""
    return float(_ideal_gas(species, (_T_H_REF_K,), "s")[0])


@dataclass(frozen=True)
class FlueGas:
    """Mezcla de gases ideales: aire o gases de combustión (vademecum §5).

    ``y`` son las fracciones molares normalizadas, en el orden de
    :data:`GAS_SPECIES`; ``p_Pa`` es la presión de los gases (solo cuenta
    para el punto de rocío del vapor de agua).
    """

    y: tuple[float, ...]
    p_Pa: float = 101_325.0

    @classmethod
    def from_fractions(
        cls,
        fractions: Mapping[str, float],
        *,
        basis: Literal["molar", "mass"] = "molar",
        p_Pa: float = 101_325.0,
    ) -> FlueGas:
        """Mezcla a partir de fracciones molares o másicas (se normalizan).

        Raises
        ------
        ValueError
            Si hay un componente desconocido, una fracción negativa o todas son cero.
        """
        unknown = set(fractions) - set(GAS_SPECIES)
        if unknown:
            raise ValueError(
                f"Componentes desconocidos: {sorted(unknown)}. Los gases pueden tener "
                f"{', '.join(GAS_NAMES[s] for s in GAS_SPECIES)}."
            )
        values = [float(fractions.get(s, 0.0)) for s in GAS_SPECIES]
        if any(v < 0.0 or not math.isfinite(v) for v in values):
            raise ValueError("Las fracciones de los componentes no pueden ser negativas.")
        if basis == "mass":  # nᵢ ∝ wᵢ / Mᵢ
            values = [v / molar_mass(s) for v, s in zip(values, GAS_SPECIES, strict=True)]
        total = sum(values)
        if total <= 0.0:
            raise ValueError(
                "La composición de los gases está vacía: cargá al menos un componente."
            )
        if not (math.isfinite(p_Pa) and p_Pa > 0.0):
            raise ValueError("La presión de los gases tiene que ser positiva.")
        return cls(tuple(v / total for v in values), float(p_Pa))

    @property
    def mole_fractions(self) -> dict[str, float]:
        return dict(zip(GAS_SPECIES, self.y, strict=True))

    @property
    def M_kg_per_mol(self) -> float:
        """Masa molar de la mezcla, M = Σ yᵢ·Mᵢ."""
        return sum(y * molar_mass(s) for s, y in zip(GAS_SPECIES, self.y, strict=True))

    @property
    def mass_fractions(self) -> dict[str, float]:
        """wᵢ = yᵢ·Mᵢ / M."""
        M = self.M_kg_per_mol
        return {s: y * molar_mass(s) / M for s, y in zip(GAS_SPECIES, self.y, strict=True)}

    @property
    def R_J_per_kg_K(self) -> float:
        """Constante particular de la mezcla, R = R_u / M."""
        return R_U / self.M_kg_per_mol

    @property
    def T_min_K(self) -> float:
        """Temperatura mínima del modelo: el punto triple del componente más alto (agua, CO₂)."""
        return max(_T_min_species(s) for s, y in zip(GAS_SPECIES, self.y, strict=True) if y > 0)

    # --- entalpía -------------------------------------------------------
    def h_array(self, temps: Sequence[float]) -> np.ndarray:
        """h_g(T) = Σ wᵢ·[hᵢ(T) − hᵢ(25 °C)] para varias temperaturas (J/kg)."""
        total = np.zeros(len(temps))
        for s, w in self.mass_fractions.items():
            if w > 0.0:
                total += w * (_ideal_gas(s, temps, "h") - _h_reference(s))
        return total

    def h(self, T_K: float) -> float:
        """Entalpía de los gases (J/kg), cero a 25 °C."""
        return float(self.h_array((T_K,))[0])

    def cp(self, T_K: float) -> float:
        """c_p de la mezcla a T (J/(kg·K)): Σ wᵢ·c_p,i(T)."""
        return sum(
            w * float(_ideal_gas(s, (T_K,), "cp")[0])
            for s, w in self.mass_fractions.items()
            if w > 0.0
        )

    def T_from_h(self, h_J_per_kg: float) -> float:
        """La temperatura a la que los gases tienen esa entalpía (raíz de h_g(T) = h)."""
        lo, hi = self.T_min_K, _T_SEARCH_MAX_K
        if not self.h(lo) <= h_J_per_kg <= self.h(hi):
            raise ValueError(
                f"Los gases quedarían fuera del rango del modelo (entre {_degC(lo)} y {_degC(hi)})."
            )
        return float(brentq(lambda T: self.h(T) - h_J_per_kg, lo, hi, xtol=1e-9, rtol=1e-13))

    # --- entropía -------------------------------------------------------
    def s0_array(self, temps: Sequence[float]) -> np.ndarray:
        """Función s°(T) = Σ wᵢ·[sᵢ°(T) − sᵢ°(25 °C)] para varias temperaturas (J/(kg·K)).

        Es la s° de las tablas de gas ideal (Cengel A-17 para el aire), medida
        desde 25 °C: s₂ − s₁ = s°(T₂) − s°(T₁) − R·ln(p₂/p₁) para la misma mezcla.
        """
        total = np.zeros(len(temps))
        for s, w in self.mass_fractions.items():
            if w > 0.0:
                total += w * (_ideal_gas(s, temps, "s") - _s_reference(s))
        return total

    def s0(self, T_K: float) -> float:
        """Función s°(T) de la mezcla (J/(kg·K)), cero a 25 °C."""
        return float(self.s0_array((T_K,))[0])

    def T_from_s0(self, s0_J_per_kg_K: float) -> float:
        """La temperatura a la que la función s° vale ``s0_J_per_kg_K``."""
        lo, hi = self.T_min_K, _T_SEARCH_MAX_K
        if not self.s0(lo) <= s0_J_per_kg_K <= self.s0(hi):
            raise ValueError(
                "La compresión o expansión llevaría los gases fuera del rango del modelo (entre "
                f"{_degC(lo)} y {_degC(hi)})."
            )
        return float(brentq(lambda T: self.s0(T) - s0_J_per_kg_K, lo, hi, xtol=1e-9, rtol=1e-13))

    def T_isentropic(self, T_in_K: float, p_in_Pa: float, p_out_Pa: float) -> float:
        """Temperatura al final de una compresión o expansión isoentrópica.

        s°(T_s) = s°(T_ent) + R·ln(p_sal / p_ent) (Cengel §7-9, gases ideales con
        calores específicos variables).
        """
        return self.T_from_s0(self.s0(T_in_K) + self.R_J_per_kg_K * math.log(p_out_Pa / p_in_Pa))

    def s(self, T_K: float, p_Pa: float) -> float:
        """Entropía absoluta de la mezcla a T y p (J/(kg·K)).

        s = Σ wᵢ·[sᵢ,abs°(25 °C) + s°ᵢ(T) − R_i·ln(yᵢ·p / p°)], con p° = 1 bar:
        comparable entre mezclas de distinta composición (aire y gases).
        """
        M = self.M_kg_per_mol
        mixing = sum(y * math.log(y) for y in self.y if y > 0.0)
        standard = sum(
            y * S_STANDARD_J_PER_MOL_K[s] for s, y in zip(GAS_SPECIES, self.y, strict=True)
        )
        return (
            standard / M
            + self.s0(T_K)
            - self.R_J_per_kg_K * math.log(p_Pa / P_STANDARD_PA)
            - R_U / M * mixing
        )

    # --- vapor de agua ----------------------------------------------------
    @property
    def p_H2O_Pa(self) -> float:
        """Presión parcial del vapor de agua (ley de Dalton, vademecum §5.3)."""
        return self.mole_fractions["H2O"] * self.p_Pa

    @property
    def dew_point_K(self) -> float | None:
        """Punto de rocío del vapor de agua: T_sat(p_H₂O). ``None`` sin agua (o casi)."""
        p_w = self.p_H2O_Pa
        if p_w <= fluid_limits("Water").P_triple_Pa:
            return None
        return float(PropsSI("T", "P", p_w, "Q", 1, "Water"))


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


#: Aire seco (fracciones molares redondeadas: N₂ 78,08 %, O₂ 20,95 %, Ar 0,93 % y
#: CO₂ 0,04 %). Es el de las tablas de gas ideal de Cengel (A-17).
AIR_DRY = FlueGas.from_fractions({"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004})
#: Aire técnico simplificado del vademecum (§16.1): 21 % O₂ y 79 % N₂.
AIR_TECHNICAL = FlueGas.from_fractions({"N2": _AIR_N2, "O2": _AIR_O2})


def combustion_products(
    air: FlueGas, atoms: tuple[float, float, float, float], excess_air: float
) -> tuple[FlueGas, float]:
    """Gases de la combustión completa de un combustible CₐHᵦN_cO_d con aire (vademecum §16.2).

    ``atoms`` son los átomos (C, H, N, O) por mol de combustible. El oxígeno
    teórico es a + b/4 − d/2 moles por mol; con exceso de aire λ entran
    λ·(a + b/4 − d/2)/y_O₂ moles de aire. Salen CO₂ (a más el del aire), H₂O
    (b/2 más la del aire), N₂ (c/2 más el del aire), el Ar del aire y el O₂
    que sobra. Devuelve los gases y los moles de aire por mol de combustible.

    Raises
    ------
    ValueError
        Si λ < 1 (combustión incompleta) o el aire no tiene oxígeno.
    """
    if not (math.isfinite(excess_air) and excess_air >= 1.0):
        raise ValueError(
            f"El exceso de aire λ = {excess_air:g} tiene que ser al menos 1: con menos aire la "
            "combustión es incompleta (vademecum §16.2)."
        )
    y = air.mole_fractions
    if y["O2"] <= 0.0:
        raise ValueError("El aire de combustión tiene que tener oxígeno.")
    a, b, c, d = atoms
    o2_theoretical = a + b / 4.0 - d / 2.0
    n_air = excess_air * o2_theoretical / y["O2"]
    moles = {s: n_air * y[s] for s in GAS_SPECIES}
    moles["CO2"] += a
    moles["H2O"] += b / 2.0
    moles["N2"] += c / 2.0
    moles["O2"] -= o2_theoretical
    total = sum(moles.values())
    gas = FlueGas.from_fractions({s: v / total for s, v in moles.items()}, p_Pa=air.p_Pa)
    return gas, n_air


def exhaust_composition(excess_air: float) -> dict[str, float]:
    """Gases de la combustión completa del metano con exceso de aire λ (fracciones molares).

    Con el aire técnico simplificado del vademecum (§16.1: 21 % O₂ y 79 % N₂):
    CH₄ + λ·a_s·(0,21 O₂ + 0,79 N₂) → CO₂ + 2 H₂O + 0,79·λ·a_s N₂ + (λ − 1)·2 O₂,
    con a_s = 2 / 0,21 (vademecum §16.2). Una turbina de gas trabaja con λ ≈ 3.
    """
    if not (math.isfinite(excess_air) and excess_air >= 1.0):
        raise ValueError(
            f"El exceso de aire λ = {excess_air:g} tiene que ser al menos 1: con menos aire la "
            "combustión es incompleta (vademecum §16.2)."
        )
    a = excess_air * 2.0 / _AIR_O2
    moles = {"N2": _AIR_N2 * a, "O2": _AIR_O2 * a - 2.0, "CO2": 1.0, "H2O": 2.0, "Ar": 0.0}
    total = sum(moles.values())
    return {s: moles[s] / total for s in GAS_SPECIES}
