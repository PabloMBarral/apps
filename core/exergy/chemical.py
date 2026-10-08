"""Exergía química (Szargut): sustancias, mezclas de gases y combustibles.

La exergía química es el trabajo máximo que se obtiene al llevar una sustancia,
ya a T₀ y p₀, al equilibrio químico con el ambiente: reaccionando y
mezclándose con sus componentes de referencia (vademecum §11 y §16.13; Kotas,
1985, cap. 2; Bejan, Tsatsaronis y Moran, 1996, §3.4). Su valor depende del
**ambiente de referencia** elegido:

- **Modelo II** (Szargut, Morris y Steward, 1988): un ambiente que imita al
  real (aire con 2,2 kPa de vapor y 33,5 Pa de CO₂, agua de mar, la corteza).
  Es el de las tablas de Kotas, de Bejan et al. y de la A-26 de Moran y
  Shapiro (columna II).
- **Modelo I** (Ahrendts, 1977, 1980): un ambiente en equilibrio químico
  (columna I de la A-26).

Los valores estándar están en ``data/szargut_chemical_exergy.csv`` (a
298,15 K). Para una especie que no está tabulada se usa el **método de
Szargut**: e = Δg_f + Σν·e de los elementos, con Δg_f de los polinomios NASA
del proyecto (``core.combustion.thermo``).

Referencias:
    Szargut, J., Morris, D. R. y Steward, F. R. (1988). *Exergy Analysis of
    Thermal, Chemical, and Metallurgical Processes*. Hemisphere, Nueva York.
    Ahrendts, J. (1980). Reference states. *Energy* 5(8–9), 667–677.
    Szargut, J. y Styrylska, T. (1964). Angenäherte Bestimmung der Exergie von
    Brennstoffen. *Brennstoff-Wärme-Kraft* 16(12), 589–596.
    Kotas, T. J. (1985). *The Exergy Method of Thermal Plant Analysis*.
    Butterworths, Londres.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any, Literal

from core.combustion.heating_value import (
    DryUltimate,
    hydrogen_water_ratio,
    lhv_from_hhv,
    water_hfg_J_per_kg,
)
from core.combustion.thermo import R_U, Species, atomic_mass, species
from core.fluids import saturation_at_temperature
from core.units_system import UnitSystem, format_quantity

__all__ = [
    "MIXTURE_EXAMPLES",
    "MODELS",
    "MODEL_P0",
    "T_STANDARD_K",
    "ChemicalSpecies",
    "FuelExergy",
    "FuelKind",
    "MixtureExergy",
    "MixtureRow",
    "ReferenceModel",
    "SpeciesExergy",
    "SzargutMethod",
    "chemical_species",
    "fuel_chemical_exergy",
    "fuel_exergy_to_dict",
    "fuel_ratio_table",
    "mixture_chemical_exergy",
    "mixture_exergy_to_dict",
    "reference_fraction",
    "species_available",
    "species_exergy",
    "species_exergy_to_dict",
    "standard_chemical_exergy",
    "szargut_method",
]

ReferenceModel = Literal["szargut1988", "ahrendts1980"]
FuelKind = Literal["sólido", "líquido"]

#: Nombre de cada ambiente de referencia, para la página.
MODELS: dict[ReferenceModel, str] = {
    "szargut1988": "Szargut et al. (1988) — modelo II",
    "ahrendts1980": "Ahrendts (1980) — modelo I",
}
#: Presión del ambiente de cada modelo (Moran y Shapiro, tabla A-26).
MODEL_P0: dict[ReferenceModel, float] = {
    "szargut1988": 101_325.0,
    "ahrendts1980": 1.019 * 101_325.0,
}
#: Temperatura de las tablas estándar (25 °C).
T_STANDARD_K = 298.15
P_STANDARD_PA = 101_325.0

_CSV = Path(__file__).resolve().parents[2] / "data" / "szargut_chemical_exergy.csv"

# Masas atómicas (g/mol): las del proyecto (core.combustion.thermo) más los
# gases nobles del aire (IUPAC, 2021).
_NOBLE_GAS_MASS_G: dict[str, float] = {"He": 4.002602, "Ne": 20.1797, "Kr": 83.798, "Xe": 131.293}

# Entropía absoluta a 298,15 K y 1 bar de los elementos sólidos, que no están
# entre los polinomios NASA del proyecto (J/(mol·K)): grafito, 5,74 (NIST-JANAF);
# azufre rómbico, 32,054 (CODATA, Cox et al., 1989).
_SOLID_ELEMENT_S0: dict[str, float] = {"C": 5.74, "S": 32.054}

# Estado de referencia de cada elemento para Δg_f (y su clave en la tabla).
_ELEMENT_REF: dict[str, tuple[str, float]] = {
    "C": ("C(s)", 1.0),
    "S": ("S(s)", 1.0),
    "H": ("H2", 0.5),
    "O": ("O2", 0.5),
    "N": ("N2", 0.5),
    "Ar": ("Ar", 1.0),
}

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _parse_formula(formula: str) -> tuple[tuple[str, float], ...]:
    """``"CH3OH"`` → ``(("C", 1), ("H", 4), ("O", 1))`` (suma los repetidos)."""
    atoms: dict[str, float] = {}
    for symbol, count in _FORMULA_RE.findall(formula):
        atoms[symbol] = atoms.get(symbol, 0.0) + (float(count) if count else 1.0)
    return tuple(atoms.items())


def _atomic_mass_kg(element: str) -> float:
    if element in _NOBLE_GAS_MASS_G:
        return _NOBLE_GAS_MASS_G[element] / 1000.0
    return atomic_mass(element)


# ---------------------------------------------------------------------
# Tabla
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ChemicalSpecies:
    """Una sustancia de la tabla de exergías químicas estándar.

    ``tabulated`` guarda la exergía química molar (J/mol) de cada modelo que
    la tabula; ``nasa_key`` es su especie entre los polinomios NASA (para el
    método de Szargut y los poderes caloríficos), si la hay.
    """

    key: str
    name: str
    formula: tuple[tuple[str, float], ...]
    phase: Literal["g", "l", "s"]
    tabulated: tuple[tuple[ReferenceModel, float], ...]
    group: str
    nasa_key: str | None
    iso6976_name: str | None

    @property
    def atoms(self) -> dict[str, float]:
        return dict(self.formula)

    @property
    def M_kg_per_mol(self) -> float:
        return sum(n * _atomic_mass_kg(el) for el, n in self.formula)

    def tabulated_J_per_mol(self, model: ReferenceModel) -> float | None:
        return dict(self.tabulated).get(model)

    @property
    def phase_label(self) -> str:
        return {"g": "gas", "l": "líquido", "s": "sólido"}[self.phase]

    @property
    def is_fuel(self) -> bool:
        """Tiene C, H o S que se queman (no es un producto ni un gas del aire)."""
        return self.group.startswith("combustible") or self.group == "elemento"


@cache
def chemical_species() -> dict[str, ChemicalSpecies]:
    """Las sustancias de ``data/szargut_chemical_exergy.csv``, por clave."""
    with _CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    out: dict[str, ChemicalSpecies] = {}
    for r in rows:
        tab: list[tuple[ReferenceModel, float]] = []
        for model, col in (("szargut1988", "e_szargut1988"), ("ahrendts1980", "e_ahrendts1980")):
            if r[col].strip():
                tab.append((model, float(r[col]) * 1000.0))  # type: ignore[arg-type]
        out[r["key"]] = ChemicalSpecies(
            key=r["key"],
            name=r["name"],
            formula=_parse_formula(r["formula"]),
            phase=r["phase"],  # type: ignore[arg-type]
            tabulated=tuple(tab),
            group=r["group"],
            nasa_key=r["nasa_key"] or None,
            iso6976_name=r["iso6976_name"] or None,
        )
    return out


def _species(key: str) -> ChemicalSpecies:
    table = chemical_species()
    if key not in table:
        raise ValueError(
            f"«{key}» no está en la tabla de exergías químicas. Opciones: {', '.join(table)}."
        )
    return table[key]


def _nasa(key: str) -> Species | None:
    sp = _species(key)
    return species(sp.nasa_key) if sp.nasa_key else None


# ---------------------------------------------------------------------
# Método de Szargut: e = Δg_f + Σν·e_elementos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SzargutMethod:
    """Exergía química de una especie desde su energía libre de formación.

    e = Δg_f + Σ ν·e_ref (Szargut et al., 1988; Kotas, 1985), con
    Δg_f = Δh_f − T₀·(s° − Σ ν·s°_ref) a 25 °C y los elementos en su estado de
    referencia (grafito, azufre rómbico, H₂, O₂, N₂, Ar). ``elements`` guarda,
    por elemento: (elemento, ν en moles de su sustancia de referencia, clave de
    esa sustancia, su e de la tabla en J/mol y su s° en J/(mol·K)).
    """

    key: str
    model: ReferenceModel
    hf_J_per_mol: float
    s0_J_per_mol_K: float
    elements: tuple[tuple[str, float, str, float, float], ...]

    @property
    def T0_K(self) -> float:
        return T_STANDARD_K

    @property
    def s0_elements_J_per_mol_K(self) -> float:
        """Σ ν·s°_ref."""
        return sum(nu * s0 for _, nu, _, _, s0 in self.elements)

    @property
    def ds_f_J_per_mol_K(self) -> float:
        """Δs_f = s° − Σ ν·s°_ref."""
        return self.s0_J_per_mol_K - self.s0_elements_J_per_mol_K

    @property
    def dg_f_J_per_mol(self) -> float:
        return self.hf_J_per_mol - self.T0_K * self.ds_f_J_per_mol_K

    @property
    def elements_J_per_mol(self) -> float:
        """Σ ν·e_ref."""
        return sum(nu * e for _, nu, _, e, _ in self.elements)

    @property
    def e_J_per_mol(self) -> float:
        return self.dg_f_J_per_mol + self.elements_J_per_mol


def _reference_s0(ref_key: str) -> float:
    """s° a 25 °C de la sustancia de referencia de un elemento (J/(mol·K))."""
    if ref_key == "C(s)":
        return _SOLID_ELEMENT_S0["C"]
    if ref_key == "S(s)":
        return _SOLID_ELEMENT_S0["S"]
    return species(ref_key).s0(T_STANDARD_K)


def szargut_method(key: str, model: ReferenceModel = "szargut1988") -> SzargutMethod:
    """La exergía química por Δg_f y los elementos (necesita la especie de NASA).

    Raises
    ------
    ValueError
        Si la sustancia no está entre los polinomios NASA del proyecto o tiene
        un elemento sin referencia (solo C, H, O, N, S y Ar).
    """
    sp = _species(key)
    nasa = _nasa(key)
    if nasa is None:
        raise ValueError(
            f"El método de Szargut necesita la entalpía y la entropía de formación del "
            f"{sp.name}, que no está entre los polinomios NASA del proyecto."
        )
    elements: list[tuple[str, float, str, float, float]] = []
    for el, n in sp.formula:
        if el not in _ELEMENT_REF:
            raise ValueError(f"El elemento {el} no tiene sustancia de referencia en este modelo.")
        ref, per_atom = _ELEMENT_REF[el]
        e_ref = _species(ref).tabulated_J_per_mol(model)
        if e_ref is None:
            raise ValueError(f"La tabla de {MODELS[model]} no trae la referencia del {el}.")
        elements.append((el, n * per_atom, ref, e_ref, _reference_s0(ref)))
    return SzargutMethod(
        key=key,
        model=model,
        hf_J_per_mol=nasa.hf_J_per_mol,
        s0_J_per_mol_K=nasa.s0(T_STANDARD_K),
        elements=tuple(elements),
    )


# ---------------------------------------------------------------------
# Exergía química de una sustancia
# ---------------------------------------------------------------------


def standard_chemical_exergy(key: str, model: ReferenceModel = "szargut1988") -> float:
    """Exergía química molar estándar (J/mol): la de la tabla o, si el modelo no
    la tabula, la del método de Szargut."""
    tab = _species(key).tabulated_J_per_mol(model)
    if tab is not None:
        return tab
    return szargut_method(key, model).e_J_per_mol


def reference_fraction(key: str, model: ReferenceModel = "szargut1988") -> float | None:
    """Fracción molar del gas en el aire de referencia: x⁰⁰ = exp(−e/R̄T₀).

    La exergía química de un gas del aire es el trabajo de comprimirlo desde su
    presión parcial en la atmósfera de referencia hasta p₀ (Szargut et al.,
    1988): e = −R̄T₀·ln(x⁰⁰). ``None`` si no es un gas del aire.
    """
    sp = _species(key)
    if sp.group != "aire" or sp.phase != "g":
        return None
    return math.exp(-standard_chemical_exergy(key, model) / (R_U * T_STANDARD_K))


def _formation_enthalpy(sp: ChemicalSpecies) -> float | None:
    """h_f (J/mol) de NASA; los elementos de referencia valen 0."""
    if sp.key in ("C(s)", "S(s)"):
        return 0.0
    nasa = species(sp.nasa_key) if sp.nasa_key else None
    return nasa.hf_J_per_mol if nasa is not None else None


def _heating_values(sp: ChemicalSpecies) -> tuple[float, float] | None:
    """(PCS, PCI) molares (J/mol) con combustión completa a CO₂, H₂O y SO₂ (vademecum §16.9)."""
    atoms = sp.atoms
    if not sp.is_fuel or not set(atoms) <= {"C", "H", "O", "N", "S"}:
        return None
    hf = _formation_enthalpy(sp)
    if hf is None:
        return None
    a, b, s = atoms.get("C", 0.0), atoms.get("H", 0.0), atoms.get("S", 0.0)
    products_l = (
        a * species("CO2").hf_J_per_mol
        + b / 2.0 * species("H2O(l)").hf_J_per_mol
        + s * species("SO2").hf_J_per_mol
    )
    products_g = products_l + b / 2.0 * (
        species("H2O").hf_J_per_mol - species("H2O(l)").hf_J_per_mol
    )
    return hf - products_l, hf - products_g


@dataclass(frozen=True)
class SpeciesExergy:
    """Exergía química estándar de una sustancia, con sus datos para la página."""

    species: ChemicalSpecies
    model: ReferenceModel
    e_J_per_mol: float
    source: Literal["tabla", "método de Szargut"]
    method: SzargutMethod | None
    other_model_J_per_mol: float | None
    hhv_J_per_mol: float | None
    lhv_J_per_mol: float | None
    x_reference: float | None

    @property
    def e_J_per_kg(self) -> float:
        return self.e_J_per_mol / self.species.M_kg_per_mol

    @property
    def ratio_lhv(self) -> float | None:
        """φ = e/PCI (Kotas, 1985: ≈ 1,04 para los hidrocarburos gaseosos)."""
        return None if not self.lhv_J_per_mol else self.e_J_per_mol / self.lhv_J_per_mol

    @property
    def ratio_hhv(self) -> float | None:
        return None if not self.hhv_J_per_mol else self.e_J_per_mol / self.hhv_J_per_mol

    @property
    def method_deviation(self) -> float | None:
        """Desvío relativo del método de Szargut contra la tabla (si hay los dos)."""
        if self.method is None or self.source != "tabla":
            return None
        return self.method.e_J_per_mol / self.e_J_per_mol - 1.0


def _has_method(sp: ChemicalSpecies) -> bool:
    """Si el método de Szargut se puede aplicar (datos de NASA y elementos con referencia)."""
    is_reference = sp.key in {ref for ref, _ in _ELEMENT_REF.values()}
    return bool(sp.nasa_key) and set(sp.atoms) <= set(_ELEMENT_REF) and not is_reference


def species_available(key: str, model: ReferenceModel = "szargut1988") -> bool:
    """Si el modelo tabula la sustancia o el método de Szargut la puede calcular.

    El modelo I (Ahrendts, 1980) no incluye los gases nobles menores (He, Ne, Kr,
    Xe) ni varios hidrocarburos que no están entre los polinomios NASA del proyecto.
    """
    sp = _species(key)
    return sp.tabulated_J_per_mol(model) is not None or _has_method(sp)


def species_exergy(key: str, model: ReferenceModel = "szargut1988") -> SpeciesExergy:
    """Exergía química de una sustancia de la tabla (o por el método de Szargut)."""
    sp = _species(key)
    method = szargut_method(key, model) if _has_method(sp) else None
    tab = sp.tabulated_J_per_mol(model)
    if tab is None and method is None:
        other = next(m for m in MODELS if m != model)
        if not sp.nasa_key and len(sp.atoms) == 1:  # un gas de referencia del aire
            reason = f"el ambiente de {MODELS[model]} no incluye el {sp.name}"
        else:
            reason = (
                f"la tabla de {MODELS[model]} no trae el {sp.name} y, sin sus datos de NASA, "
                "no se puede calcular con el método de Szargut"
            )
        raise ValueError(f"{reason[0].upper()}{reason[1:]}: elegí {MODELS[other]}.")
    e = tab if tab is not None else method.e_J_per_mol  # type: ignore[union-attr]
    other: ReferenceModel = "ahrendts1980" if model == "szargut1988" else "szargut1988"
    other_e = sp.tabulated_J_per_mol(other)
    hv = _heating_values(sp)
    return SpeciesExergy(
        species=sp,
        model=model,
        e_J_per_mol=e,
        source="tabla" if tab is not None else "método de Szargut",
        method=method,
        other_model_J_per_mol=other_e,
        hhv_J_per_mol=hv[0] if hv else None,
        lhv_J_per_mol=hv[1] if hv else None,
        x_reference=reference_fraction(key, model),
    )


def fuel_ratio_table(model: ReferenceModel = "szargut1988") -> list[SpeciesExergy]:
    """Los combustibles de la tabla con su φ = e/PCI, de menor a mayor.

    Solo los que el modelo tabula o el método de Szargut puede calcular
    (:func:`species_available`).
    """
    rows = [
        species_exergy(k, model)
        for k, sp in chemical_species().items()
        if sp.is_fuel and species_available(k, model)
    ]
    rows = [r for r in rows if r.ratio_lhv is not None]
    return sorted(rows, key=lambda r: r.ratio_lhv or 0.0)


# ---------------------------------------------------------------------
# Mezclas de gases ideales (con el agua que condensa)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureRow:
    """Un componente de la mezcla en el estado de referencia (25 °C, p₀)."""

    key: str
    x_in: float  # fracción molar en la mezcla dada
    x_gas: float  # fracción molar en la fase gaseosa a 25 °C
    e_J_per_mol: float  # exergía química del componente puro (gas)

    @property
    def x_e_J_per_mol(self) -> float:
        """x·e (por mol de gas)."""
        return self.x_gas * self.e_J_per_mol

    @property
    def mixing_J_per_mol(self) -> float:
        """R̄T₀·x·ln x (por mol de gas): la exergía que se pierde al mezclar."""
        return R_U * T_STANDARD_K * self.x_gas * math.log(self.x_gas) if self.x_gas > 0 else 0.0


@dataclass(frozen=True)
class MixtureExergy:
    """Exergía química de una mezcla de gases ideales (Szargut et al., 1988; Kotas, 1985).

    e_mez = n_g·[Σ x_i·e_i + R̄T₀·Σ x_i·ln x_i] + n_ℓ·e_H₂O(ℓ), por mol de
    mezcla: a 25 °C y p₀ el vapor de agua por encima de su presión de
    saturación condensa (n_ℓ moles de líquido) y el gas queda saturado, como
    calcula TESPy (Witte y Tuschy, 2020) siguiendo a Bejan et al. (1996).
    """

    rows: tuple[MixtureRow, ...]
    model: ReferenceModel
    n_gas: float  # mol de gas por mol de mezcla
    e_liquid_J_per_mol: float  # exergía química del agua líquida
    x_sat: float  # fracción molar del vapor saturado a 25 °C y p₀
    M_kg_per_mol: float

    @property
    def n_liquid(self) -> float:
        return 1.0 - self.n_gas

    @property
    def condenses(self) -> bool:
        return self.n_liquid > 1e-12

    @property
    def sum_x_e_J_per_mol(self) -> float:
        return sum(r.x_e_J_per_mol for r in self.rows)

    @property
    def mixing_J_per_mol(self) -> float:
        return sum(r.mixing_J_per_mol for r in self.rows)

    @property
    def gas_J_per_mol(self) -> float:
        """Por mol de gas: Σx·e + R̄T₀·Σx·ln x."""
        return self.sum_x_e_J_per_mol + self.mixing_J_per_mol

    @property
    def e_J_per_mol(self) -> float:
        """Por mol de mezcla (gas + agua condensada)."""
        return self.n_gas * self.gas_J_per_mol + self.n_liquid * self.e_liquid_J_per_mol

    @property
    def e_J_per_kg(self) -> float:
        return self.e_J_per_mol / self.M_kg_per_mol


def mixture_chemical_exergy(
    fractions: Mapping[str, float],
    model: ReferenceModel = "szargut1988",
    *,
    condense: bool = True,
) -> MixtureExergy:
    """Exergía química de una mezcla de gases dada por fracciones molares (se normalizan).

    Raises
    ------
    ValueError
        Si un componente no es un gas de la tabla, si hay fracciones negativas o
        la mezcla está vacía.
    """
    table = chemical_species()
    bad = [k for k in fractions if k not in table or table[k].phase != "g"]
    if bad:
        raise ValueError(
            f"Componentes que no son gases de la tabla de exergías químicas: {', '.join(bad)}."
        )
    if any(not math.isfinite(x) or x < 0.0 for x in fractions.values()):
        raise ValueError("Las fracciones de la mezcla no pueden ser negativas.")
    total = sum(fractions.values())
    if total <= 0.0:
        raise ValueError("La mezcla está vacía.")
    y = {k: x / total for k, x in fractions.items() if x > 0.0}
    x_sat = saturation_at_temperature("Water", T_STANDARD_K).P_sat_Pa / P_STANDARD_PA
    n_gas = 1.0
    gas = dict(y)
    if condense and y.get("H2O", 0.0) > x_sat:
        n_gas = (1.0 - y["H2O"]) / (1.0 - x_sat)
        gas = {k: (x / n_gas if k != "H2O" else x_sat) for k, x in y.items()}
    rows = tuple(MixtureRow(k, y[k], gas[k], standard_chemical_exergy(k, model)) for k in y)
    M = sum(x * table[k].M_kg_per_mol for k, x in y.items())
    return MixtureExergy(
        rows=rows,
        model=model,
        n_gas=n_gas,
        e_liquid_J_per_mol=standard_chemical_exergy("H2O(l)", model),
        x_sat=x_sat,
        M_kg_per_mol=M,
    )


#: Mezclas de ejemplo (fracciones molares).
MIXTURE_EXAMPLES: dict[str, dict[str, float]] = {
    "Aire seco (N₂, O₂, Ar y CO₂)": {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004},
    "Aire técnico (21 % de O₂ y 79 % de N₂, vademecum §16.1)": {"N2": 0.79, "O2": 0.21},
    "Gas natural típico": {"CH4": 0.90, "C2H6": 0.05, "C3H8": 0.02, "N2": 0.02, "CO2": 0.01},
    "Biogás (60 % de metano)": {"CH4": 0.60, "CO2": 0.38, "H2O": 0.018, "H2S": 0.002},
    "Humos de metano con λ = 1,2 (condensan a 25 °C)": {
        "CO2": 1.0 / 12.424,
        "H2O": 2.0 / 12.424,
        "O2": 0.4 / 12.424,
        "N2": 9.024 / 12.424,
    },
    "Gas de síntesis (H₂ + CO)": {"H2": 0.40, "CO": 0.40, "CO2": 0.10, "N2": 0.10},
}


# ---------------------------------------------------------------------
# Combustibles por análisis elemental: Szargut y Styrylska (1964)
# ---------------------------------------------------------------------

#: o/c (en masa) hasta el que vale la forma para carbones y hasta el que vale la de la madera.
O_C_COAL_MAX = 0.667
O_C_WOOD_MAX = 2.67


@dataclass(frozen=True)
class FuelExergy:
    """Exergía química de un combustible sólido o líquido por su análisis elemental.

    β = e/PCI de la materia orgánica seca (Szargut y Styrylska, 1964, como las
    reproducen Szargut et al., 1988, y Kotas, 1985), con las relaciones en
    masa h/c, o/c, n/c y s/c:

    - carbones, lignitos, coque y turba (o/c ≤ 0,667):
      β = 1,0437 + 0,1882·h/c + 0,0610·o/c + 0,0404·n/c;
    - madera y biomasa (0,667 < o/c ≤ 2,67):
      β = [1,0438 + 0,1882·h/c − 0,2509·o/c·(1 + 0,7256·h/c) + 0,0383·n/c] /
      (1 − 0,3035·o/c);
    - líquidos: β = 1,0401 + 0,1728·h/c + 0,0432·o/c + 0,2169·s/c·(1 − 2,0628·h/c).

    Por kg tal cual (Kotas, 1985): e = β·(PCI + W·h_fg) + (e_S − PCI_S)·S + e_w·W.
    PCI + W·h_fg es el PCI de la materia seca; el azufre de un sólido suma su
    exergía menos el calor que ya contó el PCI, y la humedad (agua líquida), la
    suya. La exergía de las cenizas se desprecia.
    """

    kind: FuelKind
    branch: Literal["carbón", "madera", "líquido"]
    ultimate: DryUltimate
    moisture: float
    hhv_d_J_per_kg: float
    model: ReferenceModel
    warnings: tuple[str, ...] = ()

    # --- relaciones en masa -------------------------------------------------
    @property
    def h_c(self) -> float:
        return self.ultimate.H / self.ultimate.C

    @property
    def o_c(self) -> float:
        return self.ultimate.O / self.ultimate.C

    @property
    def n_c(self) -> float:
        return self.ultimate.N / self.ultimate.C

    @property
    def s_c(self) -> float:
        return self.ultimate.S / self.ultimate.C

    @property
    def beta(self) -> float:
        return _beta(self.branch, self.h_c, self.o_c, self.n_c, self.s_c)

    # --- poderes caloríficos ---------------------------------------------
    @property
    def lhv_d_J_per_kg(self) -> float:
        return lhv_from_hhv(self.hhv_d_J_per_kg, self.ultimate.H, self.moisture)[0]

    @property
    def lhv_ar_J_per_kg(self) -> float:
        return lhv_from_hhv(self.hhv_d_J_per_kg, self.ultimate.H, self.moisture)[1]

    @property
    def hhv_ar_J_per_kg(self) -> float:
        return self.hhv_d_J_per_kg * (1.0 - self.moisture)

    @property
    def h_fg_J_per_kg(self) -> float:
        return water_hfg_J_per_kg()

    # --- términos ---------------------------------------------------------
    @property
    def lhv_plus_water_J_per_kg(self) -> float:
        """PCI + W·h_fg = PCI_s·(1 − W): el PCI de la materia seca, por kg tal cual."""
        return self.lhv_ar_J_per_kg + self.moisture * self.h_fg_J_per_kg

    @property
    def e_organic_J_per_kg(self) -> float:
        return self.beta * self.lhv_plus_water_J_per_kg

    @property
    def S_ar(self) -> float:
        return self.ultimate.S * (1.0 - self.moisture)

    @property
    def e_S_J_per_kg(self) -> float:
        """Exergía química del azufre rómbico por kg (de la tabla del modelo)."""
        sp = _species("S(s)")
        return standard_chemical_exergy("S(s)", self.model) / sp.M_kg_per_mol

    @property
    def lhv_S_J_per_kg(self) -> float:
        """Calor de S → SO₂ por kg de azufre (NASA): ≈ 9,26 MJ/kg."""
        return -species("SO2").hf_J_per_mol / atomic_mass("S")

    @property
    def e_sulfur_J_per_kg(self) -> float:
        if self.kind == "líquido":
            return 0.0  # el azufre de un líquido ya está en β
        return (self.e_S_J_per_kg - self.lhv_S_J_per_kg) * self.S_ar

    @property
    def e_w_J_per_kg(self) -> float:
        sp = _species("H2O(l)")
        return standard_chemical_exergy("H2O(l)", self.model) / sp.M_kg_per_mol

    @property
    def e_water_J_per_kg(self) -> float:
        return self.e_w_J_per_kg * self.moisture

    @property
    def e_J_per_kg(self) -> float:
        """Exergía química por kg tal cual."""
        return self.e_organic_J_per_kg + self.e_sulfur_J_per_kg + self.e_water_J_per_kg

    @property
    def ratio_lhv(self) -> float | None:
        lhv = self.lhv_ar_J_per_kg
        return self.e_J_per_kg / lhv if lhv > 0.0 else None

    @property
    def ratio_hhv(self) -> float:
        return self.e_J_per_kg / self.hhv_ar_J_per_kg


def _beta(branch: str, h_c: float, o_c: float, n_c: float, s_c: float) -> float:
    if branch == "carbón":
        return 1.0437 + 0.1882 * h_c + 0.0610 * o_c + 0.0404 * n_c
    if branch == "madera":
        return (1.0438 + 0.1882 * h_c - 0.2509 * o_c * (1.0 + 0.7256 * h_c) + 0.0383 * n_c) / (
            1.0 - 0.3035 * o_c
        )
    return 1.0401 + 0.1728 * h_c + 0.0432 * o_c + 0.2169 * s_c * (1.0 - 2.0628 * h_c)


def fuel_chemical_exergy(
    ultimate: DryUltimate,
    hhv_d_J_per_kg: float,
    *,
    moisture: float = 0.0,
    kind: FuelKind = "sólido",
    model: ReferenceModel = "szargut1988",
) -> FuelExergy:
    """Exergía química de un combustible por su análisis elemental (Szargut y Styrylska, 1964).

    ``ultimate`` en base seca (fracciones en masa), ``hhv_d_J_per_kg`` el PCS
    seco y ``moisture`` la humedad tal cual. Rango de validez (Kotas, 1985):
    sólidos con o/c ≤ 2,67 en masa (hasta 0,667, la forma de los
    carbones; arriba, la de la madera) y combustibles líquidos con C, H, O y S.

    Raises
    ------
    ValueError
        Si el análisis no suma 100 %, no hay carbono, la humedad no está entre 0
        y 1, el PCS no es positivo o un sólido tiene o/c > 2,67.
    """
    ultimate.validate()
    if ultimate.C <= 0.0:
        raise ValueError(
            "La correlación de Szargut y Styrylska usa las relaciones h/c, o/c y n/c: "
            "el combustible tiene que tener carbono."
        )
    if not (math.isfinite(moisture) and 0.0 <= moisture < 1.0):
        raise ValueError("La humedad tal cual tiene que estar entre 0 y 100 %.")
    if not (math.isfinite(hhv_d_J_per_kg) and hhv_d_J_per_kg > 0.0):
        raise ValueError("El PCS seco tiene que ser positivo.")
    o_c = ultimate.O / ultimate.C
    oc_txt = f"{o_c:.2f}".replace(".", ",")
    warnings: list[str] = []
    branch: Literal["carbón", "madera", "líquido"]
    if kind == "líquido":
        branch = "líquido"
        if o_c > 1.0:
            warnings.append(
                f"o/c = {oc_txt}: la forma de los líquidos se ajustó con combustibles con poco "
                "oxígeno (con el metanol, o/c = 1,33, se pasa +2,7 %)."
            )
        if moisture > 0.0:
            warnings.append(
                "La forma de los líquidos es para combustibles sin humedad: el agua se suma "
                "como agua líquida."
            )
    elif o_c <= O_C_COAL_MAX:
        branch = "carbón"
    elif o_c <= O_C_WOOD_MAX:
        branch = "madera"
        if o_c > 2.0:
            warnings.append(
                f"o/c = {oc_txt}: cerca del límite de la correlación (2,67) β crece rápido "
                "(el denominador 1 − 0,3035·o/c se achica). Revisá el análisis."
            )
    else:
        raise ValueError(
            f"Con o/c = {oc_txt} (en masa) el combustible tiene más oxígeno del que admite la "
            "correlación de Szargut y Styrylska (o/c ≤ 2,67, la relación del CO₂): un sólido "
            "así casi no tiene qué oxidar. Revisá el análisis."
        )
    fe = FuelExergy(
        kind=kind,
        branch=branch,
        ultimate=ultimate,
        moisture=moisture,
        hhv_d_J_per_kg=hhv_d_J_per_kg,
        model=model,
    )
    if fe.lhv_ar_J_per_kg <= 0.0:
        warnings.append(
            "Con esa humedad el PCI tal cual no es positivo: el combustible no se sostiene "
            "solo, aunque su exergía química sigue siendo positiva."
        )
    return replace(fe, warnings=tuple(warnings)) if warnings else fe


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _value(value_si: float, kind: Any, system: UnitSystem) -> dict[str, Any]:
    return {"valor": format_quantity(value_si, kind, system)}


def species_exergy_to_dict(result: SpeciesExergy, system: UnitSystem) -> dict[str, Any]:
    """Resultado de una sustancia → dict serializable (export CSV/JSON)."""
    sp = result.species
    data: dict[str, Any] = {
        "sustancia": sp.name,
        "formula": "".join(f"{el}{'' if n == 1 else int(n)}" for el, n in sp.formula),
        "fase": sp.phase_label,
        "modelo": MODELS[result.model],
        "fuente": result.source,
        "M_kg_kmol": sp.M_kg_per_mol * 1000.0,
        "e_molar": format_quantity(result.e_J_per_mol, "molar_enthalpy", system),
        "e_masica": format_quantity(result.e_J_per_kg, "specific_enthalpy", system),
    }
    if result.method is not None:
        m = result.method
        data["metodo_de_Szargut"] = {
            "dh_f": format_quantity(m.hf_J_per_mol, "molar_enthalpy", system),
            "dg_f": format_quantity(m.dg_f_J_per_mol, "molar_enthalpy", system),
            "elementos": format_quantity(m.elements_J_per_mol, "molar_enthalpy", system),
            "e": format_quantity(m.e_J_per_mol, "molar_enthalpy", system),
        }
    if result.lhv_J_per_mol is not None:
        data["PCI_molar"] = format_quantity(result.lhv_J_per_mol, "molar_enthalpy", system)
        data["PCS_molar"] = format_quantity(result.hhv_J_per_mol or 0.0, "molar_enthalpy", system)
        data["e_sobre_PCI"] = result.ratio_lhv
        data["e_sobre_PCS"] = result.ratio_hhv
    if result.x_reference is not None:
        data["x_referencia"] = result.x_reference
    if result.other_model_J_per_mol is not None:
        data["e_otro_modelo"] = format_quantity(
            result.other_model_J_per_mol, "molar_enthalpy", system
        )
    return data


def mixture_exergy_to_dict(result: MixtureExergy, system: UnitSystem) -> dict[str, Any]:
    """Resultado de una mezcla → dict serializable."""
    table = chemical_species()
    return {
        "modelo": MODELS[result.model],
        "componentes": {
            table[r.key].name: {
                "x": r.x_in,
                "x_gas": r.x_gas,
                "e": format_quantity(r.e_J_per_mol, "molar_enthalpy", system),
            }
            for r in result.rows
        },
        "agua_condensada_mol_por_mol": result.n_liquid,
        "suma_x_e": format_quantity(result.sum_x_e_J_per_mol, "molar_enthalpy", system),
        "mezcla_RT0_x_lnx": format_quantity(result.mixing_J_per_mol, "molar_enthalpy", system),
        "e_molar": format_quantity(result.e_J_per_mol, "molar_enthalpy", system),
        "e_masica": format_quantity(result.e_J_per_kg, "specific_enthalpy", system),
    }


def fuel_exergy_to_dict(result: FuelExergy, system: UnitSystem) -> dict[str, Any]:
    """Resultado de un combustible por análisis elemental → dict serializable."""
    return {
        "tipo": result.kind,
        "forma_de_la_correlacion": result.branch,
        "modelo": MODELS[result.model],
        "analisis_seco": result.ultimate.pct(),
        "humedad_pct": 100.0 * result.moisture,
        "h_c": result.h_c,
        "o_c": result.o_c,
        "n_c": result.n_c,
        "s_c": result.s_c,
        "beta": result.beta,
        "PCS_seco": format_quantity(result.hhv_d_J_per_kg, "specific_enthalpy", system),
        "PCI_tal_cual": format_quantity(result.lhv_ar_J_per_kg, "specific_enthalpy", system),
        "e_materia_organica": format_quantity(
            result.e_organic_J_per_kg, "specific_enthalpy", system
        ),
        "e_azufre": format_quantity(result.e_sulfur_J_per_kg, "specific_enthalpy", system),
        "e_humedad": format_quantity(result.e_water_J_per_kg, "specific_enthalpy", system),
        "e_tal_cual": format_quantity(result.e_J_per_kg, "specific_enthalpy", system),
        "e_sobre_PCI": result.ratio_lhv,
        "e_sobre_PCS": result.ratio_hhv,
        "avisos": list(result.warnings),
        "hidrogeno_a_agua": hydrogen_water_ratio(),
    }
