"""Propiedades de las especies de la combustión: polinomios NASA de 9 coeficientes — Fase 5.

Las entalpías y entropías de los gases (y de los combustibles líquidos) salen
de los polinomios de McBride, Zehe & Gordon (2002), *NASA Glenn Coefficients
for Calculating Thermodynamic Properties of Individual Species*,
NASA/TP-2002-211556: la fuente de la tabla de entalpías de formación y
entropías del vademecum (§16.12, vía Klein & Nellis). Los coeficientes están
en ``data/nasa9_thermo.csv`` (extraídos de ``thermo.inp`` de NASA CEA con
``scripts/extract_nasa9.py``)::

    c_p/R  = a1·T⁻² + a2·T⁻¹ + a3 + a4·T + a5·T² + a6·T³ + a7·T⁴
    h/(RT) = −a1·T⁻² + a2·ln(T)/T + a3 + a4·T/2 + a5·T²/3 + a6·T³/4 + a7·T⁴/5 + b1/T
    s°/R   = −a1·T⁻²/2 − a2·T⁻¹ + a3·ln(T) + a4·T + a5·T²/2 + a6·T³/3 + a7·T⁴/4 + b2

(vademecum §4.8 trae la forma de 7 coeficientes de los mismos datos).

- La entalpía es **absoluta** en la convención de la tabla: los elementos en
  su estado de referencia valen cero a 25 °C, así que h(25 °C) = h_f y
  h(T) = h_f + [h(T) − h(25 °C)] (vademecum §16.6, la entalpía estándar).
- La entropía s°(T) es la del gas ideal a p° = 1 bar (NASA); en una mezcla,
  cada gas va a su presión parcial: s = s° − R·ln(yᵢ·p/p°) (vademecum §16.11).

Todo en SI y **por mol** (J/mol = kJ/kmol). No importa Streamlit.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from core.ideal_gas import R_U

__all__ = [
    "ELEMENTS",
    "P_REF_PA",
    "P_STANDARD_PA",
    "R_U",
    "T_MAX_K",
    "T_REF_K",
    "Species",
    "all_species",
    "atomic_mass",
    "gas_mixture_entropy",
    "mixture_enthalpy",
    "species",
    "water_hfg_J_per_mol",
]

_DATA = Path(__file__).resolve().parents[2] / "data" / "nasa9_thermo.csv"

#: Temperatura de referencia de las entalpías de formación (vademecum §16.6).
T_REF_K = 298.15
#: Presión de referencia de los reactivos y productos de la tabla (1 atm, vademecum §16.6).
P_REF_PA = 101_325.0
#: Presión del estado estándar de las entropías de NASA (1 bar).
P_STANDARD_PA = 1.0e5
#: Temperatura máxima de los polinomios que se usan (todos llegan a 6000 K).
T_MAX_K = 6000.0

#: Elementos que se siguen en los balances.
ELEMENTS: tuple[str, ...] = ("C", "H", "O", "N", "S", "Ar")

#: Masas atómicas (g/mol), IUPAC: las de los polinomios de NASA.
_ATOMIC_MASS_G: dict[str, float] = {
    "C": 12.0107,
    "H": 1.00794,
    "O": 15.9994,
    "N": 14.0067,
    "S": 32.065,
    "Ar": 39.948,
}


def atomic_mass(element: str) -> float:
    """Masa atómica del elemento (kg/mol)."""
    return _ATOMIC_MASS_G[element] / 1000.0


@dataclass(frozen=True)
class _Interval:
    T_lo: float
    T_hi: float
    a: tuple[float, float, float, float, float, float, float]
    b1: float
    b2: float


@dataclass(frozen=True)
class Species:
    """Una especie con sus polinomios NASA-9 (gas ideal o líquido)."""

    key: str
    name: str
    phase: Literal["g", "l"]
    atoms: tuple[tuple[str, float], ...]
    M_kg_per_mol: float
    hf_J_per_mol: float
    intervals: tuple[_Interval, ...]
    ref: str

    @property
    def formula(self) -> dict[str, float]:
        """Átomos de cada elemento por molécula."""
        return dict(self.atoms)

    @property
    def T_min_K(self) -> float:
        return self.intervals[0].T_lo

    @property
    def T_max_K(self) -> float:
        return self.intervals[-1].T_hi

    @property
    def is_gas(self) -> bool:
        return self.phase == "g"

    def _interval(self, T: float) -> _Interval:
        for itv in self.intervals:
            if T <= itv.T_hi:
                return itv
        return self.intervals[-1]

    def check_T(self, T_K: float) -> None:
        """Error para el alumno si T está fuera del rango de los datos (con 3 K de margen)."""
        if not (math.isfinite(T_K) and self.T_min_K - 3.0 <= T_K <= self.T_max_K):
            raise ValueError(
                f"El {self.name} solo está tabulado entre {self.T_min_K - 273.15:.0f} °C y "
                f"{self.T_max_K - 273.15:.0f} °C (polinomios NASA); T = {T_K - 273.15:.1f} °C "
                "queda afuera."
            )

    def cp(self, T_K: float) -> float:
        """c_p molar (J/(mol·K))."""
        a = self._interval(T_K).a
        T = T_K
        return R_U * (
            a[0] / T**2 + a[1] / T + a[2] + a[3] * T + a[4] * T**2 + a[5] * T**3 + a[6] * T**4
        )

    def _h_poly(self, T_K: float) -> float:
        itv = self._interval(T_K)
        a = itv.a
        T = T_K
        return R_U * (
            -a[0] / T
            + a[1] * math.log(T)
            + a[2] * T
            + a[3] * T**2 / 2
            + a[4] * T**3 / 3
            + a[5] * T**4 / 4
            + a[6] * T**5 / 5
            + itv.b1
        )

    def delta_h(self, T_K: float) -> float:
        """Entalpía sensible h(T) − h(25 °C) (J/mol): la de las tablas de Cengel A-18 a A-25."""
        return self._h_poly(T_K) - _h_poly_ref(self.key)

    def h(self, T_K: float) -> float:
        """Entalpía absoluta (J/mol): h = h_f + [h(T) − h(25 °C)] (vademecum §16.6).

        Con el h_f tabulado: el polinomio da h(25 °C) a ±3 J/mol del h_f.
        """
        return self.hf_J_per_mol + self.delta_h(T_K)

    def s0(self, T_K: float) -> float:
        """Entropía absoluta a 1 bar (J/(mol·K))."""
        itv = self._interval(T_K)
        a = itv.a
        T = T_K
        return R_U * (
            -a[0] / T**2 / 2
            - a[1] / T
            + a[2] * math.log(T)
            + a[3] * T
            + a[4] * T**2 / 2
            + a[5] * T**3 / 3
            + a[6] * T**4 / 4
            + itv.b2
        )

    def g0(self, T_K: float) -> float:
        """Energía libre de Gibbs a 1 bar, g° = h − T·s° (J/mol)."""
        return self.h(T_K) - T_K * self.s0(T_K)


@cache
def _load() -> dict[str, Species]:
    rows: dict[str, list[dict[str, str]]] = {}
    with _DATA.open(encoding="utf-8") as fh:
        lines = (line for line in fh if not line.startswith("#"))
        for row in csv.DictReader(lines):
            rows.setdefault(row["key"], []).append(row)
    out: dict[str, Species] = {}
    for key, group in rows.items():
        first = group[0]
        atoms = tuple((el, float(first[el])) for el in ELEMENTS if float(first[el]) != 0.0)
        intervals = tuple(
            _Interval(
                float(r["T_lo_K"]),
                float(r["T_hi_K"]),
                tuple(float(r[f"a{k}"]) for k in range(1, 8)),  # type: ignore[arg-type]
                float(r["b1"]),
                float(r["b2"]),
            )
            for r in group
        )
        out[key] = Species(
            key=key,
            name=first["name"],
            phase="l" if first["phase"] == "l" else "g",
            atoms=atoms,
            M_kg_per_mol=float(first["M_g_per_mol"]) / 1000.0,
            hf_J_per_mol=float(first["hf_J_per_mol"]),
            intervals=intervals,
            ref=first["ref"],
        )
    return out


def species(key: str) -> Species:
    """La especie ``key`` (``"CO2"``, ``"H2O(l)"``…).

    Raises
    ------
    ValueError
        Si no está en la base.
    """
    data = _load()
    if key not in data:
        raise ValueError(f"La especie {key!r} no está en la base (data/nasa9_thermo.csv).")
    return data[key]


def all_species() -> dict[str, Species]:
    """Todas las especies de la base, por clave."""
    return dict(_load())


@cache
def _h_poly_ref(key: str) -> float:
    return species(key)._h_poly(T_REF_K)


@cache
def water_hfg_J_per_mol() -> float:
    """h_fg del agua a 25 °C: h_f(vapor) − h_f(líquida) (vademecum §16.9)."""
    return species("H2O").hf_J_per_mol - species("H2O(l)").hf_J_per_mol


def mixture_enthalpy(moles: Mapping[str, float], T_K: float) -> float:
    """Entalpía absoluta de una mezcla (J): Σ nᵢ·hᵢ(T) (gases ideales, sin calor de mezcla)."""
    return sum(n * species(k).h(T_K) for k, n in moles.items() if n != 0.0)


def gas_mixture_entropy(moles: Mapping[str, float], T_K: float, p_Pa: float) -> float:
    """Entropía de una mezcla de gases ideales (J/K): Σ nᵢ·[s°ᵢ(T) − R·ln(yᵢ·p/p°)].

    Cengel §15-6 y vademecum §16.11: cada gas a su presión parcial.
    """
    total = sum(n for n in moles.values() if n > 0.0)
    if total <= 0.0:
        return 0.0
    return sum(
        n * (species(k).s0(T_K) - R_U * math.log(n / total * p_Pa / P_STANDARD_PA))
        for k, n in moles.items()
        if n > 0.0
    )
