"""Poder calorífico por correlaciones — Fase 6.

El PCS de una sustancia pura sale exacto de las entalpías de formación
(vademecum §16.9, :mod:`core.combustion.fuels`). Para un carbón o una biomasa
no hay h_f: se mide en una bomba calorimétrica o se **estima** con una
correlación sobre su análisis:

- **elemental** (último): C, H, O, N, S y cenizas → Dulong, Boie (1953) y
  Channiwala y Parikh (2002);
- **inmediato** (próximo): materia volátil MV, carbono fijo CF y cenizas →
  Parikh, Channiwala y Ghosal (2005) y Cordero et al. (2001).

Todas se publicaron en **base seca**, con los porcentajes en masa y el PCS en
MJ/kg; acá cada una es una función que recibe fracciones en base seca y
devuelve J/kg (SI), con la cita, el tipo de combustible y el rango de ajuste
en el docstring. El registro :data:`CORRELATIONS` las junta con sus datos
para la página y el procedimiento.

**Bases** (ASTM D3180-25): *tal cual* (como se recibe, con la humedad
W), *seca* y *seca y sin cenizas*:

    x_s = x_tc / (1 − W)        x_sscz = x_s / (1 − Cz_s)
    PCS_tc = PCS_s·(1 − W)      PCS_sscz = PCS_s / (1 − Cz_s)

El H y el O del análisis elemental son los de la materia seca: la humedad va
aparte (la convención de :class:`core.combustion.fuels.UltimateAnalysis`).

**PCI** (vademecum §16.9): el agua que se forma, (M_H₂O/2M_H)·H ≈ 8,94·H, y
la humedad salen como vapor: PCI = PCS − h_fg·(8,94·H + W), con h_fg a 25 °C
(2442,6 kJ/kg con los datos de NASA).

**Validación**:

- 536 biomasas con el PCS medido (Ghugare et al., 2014,
  ``data/ghugare2014_biomass.csv``);
- cinco carbones del Argonne Premium Coal Sample Program (Vorres, 1990,
  ``data/argonne_premium_coals.csv``);
- el PCS exacto de las sustancias puras (las h_f de NASA de la Fase 5).

Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any, Literal

from core.combustion.fuels import FUELS, Fuel, UltimateAnalysis
from core.combustion.thermo import atomic_mass, species, water_hfg_J_per_mol
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "ANALYSIS_KINDS",
    "BASES",
    "CORRELATIONS",
    "FUEL_TYPES",
    "HEATING_VALUE_EXAMPLES",
    "HEATING_VALUE_EXAMPLE_NOTES",
    "AnalysisKind",
    "ArgonneCoal",
    "BiomassSample",
    "Basis",
    "Correlation",
    "CorrelationEstimate",
    "DatasetFit",
    "DryProximate",
    "DryUltimate",
    "FuelType",
    "HeatingValueInputs",
    "HeatingValueResult",
    "Reference",
    "ReferenceKind",
    "applicable_correlations",
    "argonne_coals",
    "argonne_deviations",
    "basis_value",
    "dataset_fit",
    "default_moisture_values",
    "dry_ultimate_from_analysis",
    "element_heating_values",
    "estimate_hhv_as_fired",
    "ghugare_dataset",
    "heating_value_notes",
    "heating_value_to_dict",
    "hhv_boie",
    "hhv_channiwala_parikh",
    "hhv_cordero",
    "hhv_dulong",
    "hhv_parikh",
    "hydrogen_water_ratio",
    "lhv_from_hhv",
    "moisture_sweep",
    "out_of_range",
    "pure_substance_inputs",
    "solve_heating_value",
    "to_basis",
    "water_hfg_J_per_kg",
    "zero_lhv_moisture",
]

_DATA = Path(__file__).resolve().parents[2] / "data"
_GHUGARE_CSV = _DATA / "ghugare2014_biomass.csv"
_ARGONNE_CSV = _DATA / "argonne_premium_coals.csv"

Basis = Literal["ar", "d", "daf"]
#: Bases del análisis (ASTM D3180-25): tal cual, seca y seca y sin cenizas.
BASES: dict[Basis, str] = {
    "ar": "tal cual (como se recibe)",
    "d": "base seca",
    "daf": "seca y sin cenizas",
}
#: Subíndice de cada base en el procedimiento y las tablas.
_BASIS_SUB: dict[Basis, str] = {"ar": "tc", "d": "s", "daf": "sscz"}
#: «El análisis elemental … suma»: la base dentro de una frase.
_BASIS_PHRASE: dict[Basis, str] = {
    "ar": "tal cual",
    "d": "en base seca",
    "daf": "seco y sin cenizas",
}

AnalysisKind = Literal["ultimate", "proximate"]
ANALYSIS_KINDS: dict[AnalysisKind, str] = {
    "ultimate": "elemental (último)",
    "proximate": "inmediato (próximo)",
}

FuelType = Literal["biomass", "char", "coal", "liquid", "gas", "waste"]
#: Tipos de combustible (el dominio de ajuste de cada correlación).
FUEL_TYPES: dict[FuelType, str] = {
    "biomass": "Biomasa",
    "char": "Carbonizado (carbón vegetal)",
    "coal": "Carbón mineral o lignito",
    "liquid": "Combustible líquido",
    "gas": "Combustible gaseoso",
    "waste": "Residuo (lodos, RSU)",
}

ReferenceKind = Literal["measured", "exact"]

_ULT_KEYS = ("C", "H", "O", "N", "S")
_PROX_KEYS = ("VM", "FC")
#: Tolerancia de las sumas de los análisis (puntos porcentuales: el redondeo de los datos).
_SUM_TOL_PCT = 0.1
_BTU_PER_LB_J_PER_KG = 2326.0


@cache
def water_hfg_J_per_kg() -> float:
    """h_fg del agua a 25 °C por kg (vademecum §16.9): 2442,6 kJ/kg con los datos de NASA."""
    return water_hfg_J_per_mol() / species("H2O(l)").M_kg_per_mol


@cache
def hydrogen_water_ratio() -> float:
    """kg de agua que forma cada kg de H: M_H₂O/(2·M_H) ≈ 8,94 (el «9·H» de los libros)."""
    return species("H2O(l)").M_kg_per_mol / (2.0 * atomic_mass("H"))


# ---------------------------------------------------------------------
# Análisis en base seca
# ---------------------------------------------------------------------


def _check_fractions(values: Mapping[str, float], what: str) -> None:
    if any(not math.isfinite(v) or v < 0.0 for v in values.values()):
        raise ValueError(f"Las fracciones del análisis {what} no pueden ser negativas.")


@dataclass(frozen=True)
class DryUltimate:
    """Análisis elemental en base seca (fracciones en masa; con las cenizas suman 1)."""

    C: float
    H: float
    O: float = 0.0  # noqa: E741 (el símbolo del oxígeno)
    N: float = 0.0
    S: float = 0.0
    A: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"C": self.C, "H": self.H, "O": self.O, "N": self.N, "S": self.S, "A": self.A}

    def pct(self) -> dict[str, float]:
        """Los porcentajes en masa (base seca), como los usan las correlaciones."""
        return {k: 100.0 * v for k, v in self.as_dict().items()}

    def validate(self) -> None:
        values = self.as_dict()
        _check_fractions(values, "elemental")
        total = sum(values.values())
        if abs(total - 1.0) > _SUM_TOL_PCT / 100.0:
            raise ValueError(
                f"El análisis elemental en base seca suma {100 * total:.2f} %: C + H + O + N + S "
                "+ cenizas tiene que dar 100 %."
            )
        if self.C + self.H <= 0.0:
            raise ValueError("El combustible no tiene carbono ni hidrógeno: no hay qué quemar.")


@dataclass(frozen=True)
class DryProximate:
    """Análisis inmediato en base seca: materia volátil, carbono fijo y cenizas (fracciones)."""

    VM: float
    FC: float
    A: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"VM": self.VM, "FC": self.FC, "A": self.A}

    def pct(self) -> dict[str, float]:
        return {k: 100.0 * v for k, v in self.as_dict().items()}

    def validate(self) -> None:
        values = self.as_dict()
        _check_fractions(values, "inmediato")
        total = sum(values.values())
        if abs(total - 1.0) > _SUM_TOL_PCT / 100.0:
            raise ValueError(
                f"El análisis inmediato en base seca suma {100 * total:.2f} %: materia volátil + "
                "carbono fijo + cenizas tiene que dar 100 %."
            )
        if self.VM + self.FC <= 0.0:
            raise ValueError("El combustible es todo cenizas: no hay qué quemar.")


@dataclass(frozen=True)
class Reference:
    """PCS de referencia en base seca: medido en una bomba o exacto (de las h_f)."""

    hhv_d_J_per_kg: float
    kind: ReferenceKind = "measured"
    source: str = ""

    @property
    def label(self) -> str:
        return "PCS exacto" if self.kind == "exact" else "PCS medido"


# ---------------------------------------------------------------------
# Correlaciones
# ---------------------------------------------------------------------


def hhv_dulong(u: DryUltimate) -> float:
    """PCS por la fórmula de Dulong (J/kg, base seca).

    ``PCS = 0,3383·C + 1,443·(H − O/8) + 0,0942·S``  [MJ/kg; C, H, O, S en %
    en masa, base seca]; es la forma clásica en kcal/kg, 8080·C + 34 500·(H −
    O/8) + 2250·S con fracciones en masa.

    Se le atribuye a P. L. Dulong (1785–1838), que midió los calores de
    combustión del carbono y del hidrógeno; no hay un trabajo suyo con la
    fórmula. Tiene base física: cada elemento aporta su propio PCS (C ≈ 33,8
    MJ/kg, H₂ ≈ 144 MJ/kg, S ≈ 9,4 MJ/kg) y el O del combustible se supone ya
    unido al H como agua, así que solo se quema el «hidrógeno disponible»,
    H − O/8. Ignora la entalpía de formación del combustible.

    **Rango de validez** (tipo de combustible): carbones con poco oxígeno,
    O < 10 % en base seca (Perry's Chemical Engineers' Handbook, 9.ª ed.,
    2018, para su forma modificada). Con más O (biomasa) subestima: −10 % en
    promedio en las 536 biomasas de Ghugare et al. (2014).

    **Referencia**: la forma que tabulan Channiwala, S. A. y Parikh, P. P.
    (2002), A unified correlation for estimating HHV of solid, liquid and
    gaseous fuels, *Fuel* 81(8), 1051–1063, doi:10.1016/S0016-2361(01)00131-4.
    """
    c = u.pct()
    return 1e6 * (0.3383 * c["C"] + 1.443 * (c["H"] - c["O"] / 8.0) + 0.0942 * c["S"])


def hhv_boie(u: DryUltimate) -> float:
    """PCS por la correlación de Boie (J/kg, base seca).

    ``PCS = 0,3516·C + 1,16225·H − 0,1109·O + 0,0628·N + 0,10465·S``  [MJ/kg;
    % en masa, base seca]. Es la forma original en kcal/kg, ≈ 84,0·C +
    277,65·H + 25,0·S + 15,0·N − 26,5·O, por 4,1868 kJ/kcal.

    **Rango de validez** (tipo de combustible): combustibles sólidos; Boie la
    ajustó con carbones y lignitos. En la validación de este módulo da bien
    también en biomasa (AAE 5,2 % en Ghugare et al., 2014) y en los
    hidrocarburos y alcoholes puros (±2 %), pero no en el H₂ (−18 %).

    **Referencia**: Boie, W. (1953). Fuel technology calculations.
    *Energietechnik* 3, 309–316; en la forma que tabulan Channiwala y Parikh
    (2002), *Fuel* 81(8), 1051–1063.
    """
    c = u.pct()
    return 1e6 * (
        0.3516 * c["C"] + 1.16225 * c["H"] - 0.1109 * c["O"] + 0.0628 * c["N"] + 0.10465 * c["S"]
    )


def hhv_channiwala_parikh(u: DryUltimate) -> float:
    """PCS por la correlación unificada de Channiwala y Parikh (J/kg, base seca).

    ``PCS = 0,3491·C + 1,1783·H + 0,1005·S − 0,1034·O − 0,0151·N − 0,0211·Cz``
    [MJ/kg; % en masa, base seca].

    **Rango de validez**: combustibles sólidos, líquidos y gaseosos (gases,
    líquidos, carbones, biomasa, carbonizados y residuos), con C 0–92,25 %,
    H 0,43–25,15 %, O 0–50 %, N 0–5,6 %, S 0–94,08 %, cenizas 0–71,4 % y PCS
    4,745–55,345 MJ/kg. Ajustada con 225 datos y validada con otros 50: error
    absoluto medio (AAE) 1,45 % y error medio con signo (ABE) 0,00 %.

    **Referencia**: Channiwala, S. A. y Parikh, P. P. (2002). A unified
    correlation for estimating HHV of solid, liquid and gaseous fuels.
    *Fuel* 81(8), 1051–1063, doi:10.1016/S0016-2361(01)00131-4.
    """
    c = u.pct()
    return 1e6 * (
        0.3491 * c["C"]
        + 1.1783 * c["H"]
        + 0.1005 * c["S"]
        - 0.1034 * c["O"]
        - 0.0151 * c["N"]
        - 0.0211 * c["A"]
    )


def hhv_parikh(p: DryProximate) -> float:
    """PCS por la correlación de Parikh, Channiwala y Ghosal con el análisis inmediato (J/kg).

    ``PCS = 0,3536·CF + 0,1559·MV − 0,0078·Cz``  [MJ/kg; carbono fijo, materia
    volátil y cenizas en % en masa, base seca].

    **Rango de validez**: combustibles sólidos carbonosos (carbones, lignitos,
    biomasa, carbonizados y residuos), con CF 1,0–91,5 %, MV 0,92–90,6 % y
    cenizas 0,12–77,7 %. Ajustada con 450 datos y validada con otros 100: AAE
    3,74 %, ABE 0,12 %. Ojo: en los carbones bituminosos de Argonne (Vorres,
    1990), dentro de esos rangos, subestima 14–23 %.

    **Referencia**: Parikh, J., Channiwala, S. A. y Ghosal, G. K. (2005). A
    correlation for calculating HHV from proximate analysis of solid fuels.
    *Fuel* 84(5), 487–494.
    """
    c = p.pct()
    return 1e6 * (0.3536 * c["FC"] + 0.1559 * c["VM"] - 0.0078 * c["A"])


def hhv_cordero(p: DryProximate) -> float:
    """PCS por la correlación de Cordero et al. con el análisis inmediato (J/kg, base seca).

    ``PCS = 0,3543·CF + 0,1708·MV``  [MJ/kg; % en masa, base seca]: sin término
    de cenizas, que solo diluyen. Algunas tablas secundarias copian el
    coeficiente de la MV como 0,17008.

    **Rango de validez**: materiales lignocelulósicos (biomasa) y los
    carbonizados que se obtienen de ellos (carbón vegetal).

    **Referencia**: Cordero, T., Marquez, F., Rodriguez-Mirasol, J. y
    Rodriguez, J. J. (2001). Predicting heating values of lignocellulosics and
    carbonaceous materials from proximate analysis. *Fuel* 80(11), 1567–1571,
    doi:10.1016/S0016-2361(01)00034-5.
    """
    c = p.pct()
    return 1e6 * (0.3543 * c["FC"] + 0.1708 * c["VM"])


@dataclass(frozen=True)
class Correlation:
    """Una correlación con sus datos: lo que la página y el procedimiento muestran."""

    key: str
    name: str
    analysis: AnalysisKind
    function: Callable[[Any], float]
    #: Coeficientes en su forma publicada (MJ/kg por % en masa, base seca).
    coefficients: tuple[tuple[str, float], ...]
    fuel_types: frozenset[FuelType]
    fuels: str
    #: Rangos de ajuste (variable, mínimo, máximo) en % en base seca.
    ranges: tuple[tuple[str, float, float], ...]
    reference: str
    doi: str | None
    reported: str | None
    #: La fórmula en LaTeX (PCS_s en MJ/kg, variables en % base seca).
    latex: str


#: Correlaciones, en el orden en que se muestran (elementales y después inmediatas).
CORRELATIONS: dict[str, Correlation] = {
    "channiwala_parikh": Correlation(
        key="channiwala_parikh",
        name="Channiwala y Parikh (2002)",
        analysis="ultimate",
        function=hhv_channiwala_parikh,
        coefficients=(
            ("C", 0.3491),
            ("H", 1.1783),
            ("S", 0.1005),
            ("O", -0.1034),
            ("N", -0.0151),
            ("A", -0.0211),
        ),
        fuel_types=frozenset(FUEL_TYPES),
        fuels="sólidos, líquidos y gaseosos",
        ranges=(
            ("C", 0.0, 92.25),
            ("H", 0.43, 25.15),
            ("O", 0.0, 50.0),
            ("N", 0.0, 5.6),
            ("S", 0.0, 94.08),
            ("A", 0.0, 71.4),
        ),
        reference=(
            "Channiwala, S. A. y Parikh, P. P. (2002). A unified correlation for estimating HHV "
            "of solid, liquid and gaseous fuels. Fuel 81(8), 1051–1063."
        ),
        doi="10.1016/S0016-2361(01)00131-4",
        reported="AAE 1,45 % y ABE 0,00 % (225 datos de ajuste y 50 de validación)",
        latex=(
            r"0.3491\,C + 1.1783\,H \\ &\quad + 0.1005\,S - 0.1034\,O"
            r" \\ &\quad - 0.0151\,N - 0.0211\,\mathit{Cz}"
        ),
    ),
    "boie": Correlation(
        key="boie",
        name="Boie (1953)",
        analysis="ultimate",
        function=hhv_boie,
        coefficients=(("C", 0.3516), ("H", 1.16225), ("O", -0.1109), ("N", 0.0628), ("S", 0.10465)),
        fuel_types=frozenset({"coal", "char", "biomass", "waste"}),
        fuels="sólidos (la ajustó con carbones y lignitos)",
        ranges=(),
        reference=(
            "Boie, W. (1953). Fuel technology calculations. Energietechnik 3, 309–316 (forma "
            "tabulada por Channiwala y Parikh, 2002)."
        ),
        doi=None,
        reported=None,
        latex=(
            r"0.3516\,C + 1.16225\,H \\ &\quad - 0.1109\,O + 0.0628\,N"
            r" \\ &\quad + 0.10465\,S"
        ),
    ),
    "dulong": Correlation(
        key="dulong",
        name="Dulong",
        analysis="ultimate",
        function=hhv_dulong,
        coefficients=(("C", 0.3383), ("H", 1.443), ("O", -1.443 / 8.0), ("S", 0.0942)),
        fuel_types=frozenset({"coal", "char"}),
        fuels="carbones con poco oxígeno",
        ranges=(("O", 0.0, 10.0),),
        reference=(
            "Fórmula de Dulong (siglo XIX), en la forma que tabulan Channiwala y Parikh (2002); "
            "rango de Perry's Chemical Engineers' Handbook (9.ª ed., 2018)."
        ),
        doi=None,
        reported=None,
        latex=r"0.3383\,C + 1.443\left(H - \frac{O}{8}\right) \\ &\quad + 0.0942\,S",
    ),
    "parikh": Correlation(
        key="parikh",
        name="Parikh et al. (2005)",
        analysis="proximate",
        function=hhv_parikh,
        coefficients=(("FC", 0.3536), ("VM", 0.1559), ("A", -0.0078)),
        fuel_types=frozenset({"coal", "char", "biomass", "waste"}),
        fuels="sólidos carbonosos (carbones, biomasa, carbonizados y residuos)",
        ranges=(("FC", 1.0, 91.5), ("VM", 0.92, 90.6), ("A", 0.12, 77.7)),
        reference=(
            "Parikh, J., Channiwala, S. A. y Ghosal, G. K. (2005). A correlation for calculating "
            "HHV from proximate analysis of solid fuels. Fuel 84(5), 487–494."
        ),
        doi=None,
        reported="AAE 3,74 % y ABE 0,12 % (450 datos de ajuste y 100 de validación)",
        latex=r"0.3536\,\mathit{CF} + 0.1559\,\mathit{MV} \\ &\quad - 0.0078\,\mathit{Cz}",
    ),
    "cordero": Correlation(
        key="cordero",
        name="Cordero et al. (2001)",
        analysis="proximate",
        function=hhv_cordero,
        coefficients=(("FC", 0.3543), ("VM", 0.1708)),
        fuel_types=frozenset({"biomass", "char"}),
        fuels="biomasa lignocelulósica y sus carbonizados",
        ranges=(),
        reference=(
            "Cordero, T., Marquez, F., Rodriguez-Mirasol, J. y Rodriguez, J. J. (2001). "
            "Predicting heating values of lignocellulosics and carbonaceous materials from "
            "proximate analysis. Fuel 80(11), 1567–1571."
        ),
        doi="10.1016/S0016-2361(01)00034-5",
        reported=None,
        latex=r"0.3543\,\mathit{CF} + 0.1708\,\mathit{MV}",
    ),
}

#: Nombre de cada variable de las correlaciones (para los avisos de rango).
_VAR_NAMES: dict[str, str] = {
    "C": "C",
    "H": "H",
    "O": "O",
    "N": "N",
    "S": "S",
    "A": "cenizas",
    "VM": "materia volátil",
    "FC": "carbono fijo",
}


# ---------------------------------------------------------------------
# Datos del combustible
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HeatingValueInputs:
    """Datos del cálculo: el análisis en base seca, la humedad tal cual y la referencia.

    ``ultimate`` y ``proximate`` pueden ir uno o los dos (con las mismas
    cenizas). ``H_d`` es el H en base seca para el PCI cuando solo hay
    inmediato. ``main`` es la correlación de las métricas y el barrido
    (``None``: Channiwala y Parikh con el elemental, Parikh con el inmediato).
    La página usa :meth:`from_basis`, que acepta los datos en cualquier base.
    """

    name: str = "Combustible"
    fuel_type: FuelType = "biomass"
    ultimate: DryUltimate | None = None
    proximate: DryProximate | None = None
    moisture: float = 0.0
    H_d: float | None = None
    reference: Reference | None = None
    main: str | None = None
    #: Base en que se cargaron los datos (el procedimiento muestra el cambio a seca).
    basis: Basis = "d"

    # --- construcción desde cualquier base -------------------------------
    @classmethod
    def from_basis(
        cls,
        basis: Basis,
        *,
        name: str = "Combustible",
        fuel_type: FuelType = "biomass",
        ultimate_pct: Mapping[str, float] | None = None,
        proximate_pct: Mapping[str, float] | None = None,
        ash_pct: float = 0.0,
        moisture_pct: float = 0.0,
        o_by_difference: bool = False,
        fc_by_difference: bool = False,
        H_d_pct: float | None = None,
        reference_J_per_kg: float | None = None,
        reference_basis: Basis = "d",
        reference_kind: ReferenceKind = "measured",
        reference_source: str = "",
        main: str | None = None,
    ) -> HeatingValueInputs:
        """Datos como los carga el alumno: porcentajes en la base ``basis``.

        - ``"ar"`` (tal cual): C + H + O + N + S + cenizas + humedad = 100 (y
          MV + CF + cenizas + humedad = 100);
        - ``"d"`` (seca): C + H + O + N + S + cenizas = 100;
        - ``"daf"`` (seca y sin cenizas): C + H + O + N + S = 100, y
          ``ash_pct`` son las cenizas **en base seca**.

        La humedad es siempre la tal cual. El O (o el CF) por diferencia
        completa la suma. ``reference_J_per_kg`` es el PCS de referencia en
        ``reference_basis``.

        Raises
        ------
        ValueError
            Con un mensaje para el alumno: sumas que no dan 100, valores
            negativos, la humedad o las cenizas fuera de rango, el O o el CF
            por diferencia negativos.
        """
        values = [ash_pct, moisture_pct]
        for group in (ultimate_pct, proximate_pct):
            if group is not None:
                values += list(group.values())
        if any(not math.isfinite(v) for v in values):
            raise ValueError("Falta un dato del análisis (hay un valor vacío o inválido).")
        if not 0.0 <= moisture_pct < 100.0:
            raise ValueError("La humedad tal cual tiene que estar entre 0 y 100 %.")
        if not 0.0 <= ash_pct < 100.0:
            raise ValueError("Las cenizas tienen que estar entre 0 y 100 %.")
        if ultimate_pct is None and proximate_pct is None:
            raise ValueError("Hace falta el análisis elemental, el inmediato o los dos.")
        W = moisture_pct / 100.0
        # Lo que suma a 100 además del análisis en cada base.
        extra = {"ar": ash_pct + moisture_pct, "d": ash_pct, "daf": 0.0}[basis]
        extra_text = {"ar": " + cenizas + humedad", "d": " + cenizas", "daf": ""}[basis]
        if basis == "daf":
            A_d = ash_pct / 100.0
        elif basis == "ar":
            A_d = ash_pct / (100.0 - moisture_pct)
        else:
            A_d = ash_pct / 100.0

        def to_dry(pct: Mapping[str, float]) -> dict[str, float]:
            if basis == "ar":
                return {k: v / (100.0 - moisture_pct) for k, v in pct.items()}
            if basis == "daf":
                return {k: v / 100.0 * (1.0 - A_d) for k, v in pct.items()}
            return {k: v / 100.0 for k, v in pct.items()}

        def complete(
            pct: Mapping[str, float], keys: tuple[str, ...], diff_key: str | None, what: str
        ) -> dict[str, float]:
            out = {k: float(pct.get(k, 0.0)) for k in keys}
            if any(v < 0.0 for v in out.values()):
                raise ValueError(f"Los porcentajes del análisis {what} no pueden ser negativos.")
            labels = " + ".join(_VAR_SYMBOLS[k] for k in keys)
            if diff_key is not None:
                rest = sum(v for k, v in out.items() if k != diff_key) + extra
                out[diff_key] = 100.0 - rest
                if out[diff_key] < -1e-9:
                    raise ValueError(
                        f"Los demás componentes del análisis {what} {_BASIS_PHRASE[basis]} ya "
                        f"suman {rest:.2f} %: {_VAR_NAMES[diff_key]} por diferencia daría "
                        f"{out[diff_key]:.2f} %."
                    )
                out[diff_key] = max(out[diff_key], 0.0)
            total = sum(out.values()) + extra
            if abs(total - 100.0) > _SUM_TOL_PCT:
                raise ValueError(
                    f"El análisis {what} {_BASIS_PHRASE[basis]} suma {total:.2f} %: {labels}"
                    f"{extra_text} tiene que dar 100 %."
                )
            return out

        ultimate = proximate = None
        if ultimate_pct is not None:
            u = complete(ultimate_pct, _ULT_KEYS, "O" if o_by_difference else None, "elemental")
            ud = to_dry(u)
            ultimate = _normalized_ultimate(ud, A_d)
        if proximate_pct is not None:
            p = complete(proximate_pct, _PROX_KEYS, "FC" if fc_by_difference else None, "inmediato")
            pdry = to_dry(p)
            proximate = _normalized_proximate(pdry, A_d)
        reference = None
        if reference_J_per_kg is not None:
            if not (math.isfinite(reference_J_per_kg) and reference_J_per_kg > 0.0):
                raise ValueError("El PCS de referencia tiene que ser positivo.")
            hhv_d = {
                "ar": reference_J_per_kg / (1.0 - W),
                "d": reference_J_per_kg,
                "daf": reference_J_per_kg * (1.0 - A_d),
            }[reference_basis]
            reference = Reference(hhv_d, reference_kind, reference_source)
        H_d = None
        if ultimate is None and H_d_pct is not None:
            if not (math.isfinite(H_d_pct) and 0.0 <= H_d_pct < 100.0):
                raise ValueError("El H en base seca tiene que estar entre 0 y 100 %.")
            H_d = H_d_pct / 100.0
        inputs = cls(
            name=name,
            fuel_type=fuel_type,
            ultimate=ultimate,
            proximate=proximate,
            moisture=W,
            H_d=H_d,
            reference=reference,
            main=main,
            basis=basis,
        )
        inputs.validate()
        return inputs

    # --- propiedades -----------------------------------------------------
    @property
    def ash_d(self) -> float:
        """Cenizas en base seca (fracción)."""
        if self.ultimate is not None:
            return self.ultimate.A
        assert self.proximate is not None
        return self.proximate.A

    @property
    def hydrogen_d(self) -> float | None:
        """H en base seca (del elemental o el dato aparte); ``None`` si no se conoce."""
        if self.ultimate is not None:
            return self.ultimate.H
        return self.H_d

    @property
    def analyses(self) -> tuple[AnalysisKind, ...]:
        out: list[AnalysisKind] = []
        if self.ultimate is not None:
            out.append("ultimate")
        if self.proximate is not None:
            out.append("proximate")
        return tuple(out)

    def validate(self) -> None:
        """Errores para el alumno (en castellano).

        Raises
        ------
        ValueError
            Sin análisis, sumas que no cierran, cenizas distintas entre los dos
            análisis, humedad fuera de [0, 1), referencia no positiva o una
            correlación principal que no corresponde.
        """
        if self.ultimate is None and self.proximate is None:
            raise ValueError("Hace falta el análisis elemental, el inmediato o los dos.")
        if self.ultimate is not None:
            self.ultimate.validate()
        if self.proximate is not None:
            self.proximate.validate()
        if (
            self.ultimate is not None
            and self.proximate is not None
            and abs(self.ultimate.A - self.proximate.A) > _SUM_TOL_PCT / 100.0
        ):
            raise ValueError(
                f"Las cenizas del análisis elemental ({100 * self.ultimate.A:.2f} %) y del "
                f"inmediato ({100 * self.proximate.A:.2f} %) tienen que ser las mismas."
            )
        if not (math.isfinite(self.moisture) and 0.0 <= self.moisture < 1.0):
            raise ValueError("La humedad tal cual tiene que estar entre 0 y 100 %.")
        if self.ash_d >= 1.0:
            raise ValueError("El combustible es todo cenizas: no hay qué quemar.")
        if self.reference is not None and not (
            math.isfinite(self.reference.hhv_d_J_per_kg) and self.reference.hhv_d_J_per_kg > 0.0
        ):
            raise ValueError("El PCS de referencia tiene que ser positivo.")
        if self.main is not None:
            if self.main not in CORRELATIONS:
                raise ValueError(f"No hay una correlación {self.main!r}.")
            if self.main not in applicable_correlations(self):
                kind = ANALYSIS_KINDS[CORRELATIONS[self.main].analysis]
                raise ValueError(
                    f"{CORRELATIONS[self.main].name} usa el análisis {kind}, que no cargaste."
                )


_VAR_SYMBOLS: dict[str, str] = {
    "C": "C",
    "H": "H",
    "O": "O",
    "N": "N",
    "S": "S",
    "VM": "MV",
    "FC": "CF",
}


def _normalized_ultimate(dry: Mapping[str, float], A_d: float) -> DryUltimate:
    """Base seca con la suma exacta: reparte el redondeo (≤ 0,1 %) entre los componentes."""
    total = sum(dry.values()) + A_d
    return DryUltimate(**{k: dry[k] / total for k in _ULT_KEYS}, A=A_d / total)


def _normalized_proximate(dry: Mapping[str, float], A_d: float) -> DryProximate:
    total = sum(dry.values()) + A_d
    return DryProximate(VM=dry["VM"] / total, FC=dry["FC"] / total, A=A_d / total)


def to_basis(
    dry: Mapping[str, float], basis: Basis, moisture: float, ash_d: float
) -> dict[str, float]:
    """Fracciones en base seca → la base pedida.

    Tal cual suma la humedad ``"W"``; seca y sin cenizas saca las cenizas
    (``"A"``). Las demás claves se escalan.
    """
    if basis == "d":
        return dict(dry)
    if basis == "ar":
        out = {k: v * (1.0 - moisture) for k, v in dry.items()}
        out["W"] = moisture
        return out
    return {k: v / (1.0 - ash_d) for k, v in dry.items() if k != "A"}


def basis_value(value_d: float, basis: Basis, moisture: float, ash_d: float) -> float:
    """Un valor por kg seco (el PCS) en la base pedida: ×(1 − W) o ÷(1 − Cz_s)."""
    if basis == "ar":
        return value_d * (1.0 - moisture)
    if basis == "daf":
        return value_d / (1.0 - ash_d)
    return value_d


def lhv_from_hhv(hhv_d: float, H_d: float, moisture: float) -> tuple[float, float]:
    """PCI en base seca y tal cual desde el PCS seco (vademecum §16.9).

    PCI_s = PCS_s − h_fg·(M_H₂O/2M_H)·H_s y PCI_tc = PCI_s·(1 − W) − h_fg·W:
    el agua que se forma y la humedad salen como vapor (h_fg a 25 °C).
    """
    hfg = water_hfg_J_per_kg()
    lhv_d = hhv_d - hfg * hydrogen_water_ratio() * H_d
    return lhv_d, lhv_d * (1.0 - moisture) - hfg * moisture


def zero_lhv_moisture(lhv_d: float) -> float | None:
    """Humedad tal cual con la que el PCI se anula: W* = PCI_s / (PCI_s + h_fg)."""
    if lhv_d <= 0.0:
        return None
    return lhv_d / (lhv_d + water_hfg_J_per_kg())


def element_heating_values() -> dict[str, float]:
    """PCS de los elementos (J/kg de elemento) desde las h_f de NASA (vademecum §16.9).

    C (grafito) → CO₂, H₂ → H₂O(ℓ), S → SO₂; los elementos valen cero. Son
    la base física de los coeficientes de Dulong (33,8, 144,3 y 9,4 MJ/kg).
    """
    return {
        "C": -species("CO2").hf_J_per_mol / atomic_mass("C"),
        "H": -species("H2O(l)").hf_J_per_mol / (2.0 * atomic_mass("H")),
        "S": -species("SO2").hf_J_per_mol / atomic_mass("S"),
    }


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationEstimate:
    """El PCS de una correlación en las tres bases, su PCI y su desvío."""

    key: str
    name: str
    analysis: AnalysisKind
    hhv_d: float
    hhv_ar: float
    hhv_daf: float
    lhv_d: float | None
    lhv_ar: float | None
    deviation: float | None
    warnings: tuple[str, ...]
    in_type: bool

    @property
    def flagged(self) -> bool:
        """Fuera del rango o del tipo de combustible con que se ajustó."""
        return bool(self.warnings) or not self.in_type


@dataclass(frozen=True)
class HeatingValueResult:
    inputs: HeatingValueInputs
    estimates: tuple[CorrelationEstimate, ...]
    main_key: str
    #: La referencia en las tres bases y su PCI (``None`` sin referencia).
    reference: CorrelationEstimate | None
    #: kg de agua que forma el H por kg seco (``None`` sin H).
    water_formed_d: float | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def main(self) -> CorrelationEstimate:
        return self.estimate(self.main_key)

    def estimate(self, key: str) -> CorrelationEstimate:
        for e in self.estimates:
            if e.key == key:
                return e
        raise ValueError(f"La correlación {key!r} no se calculó (falta su análisis).")

    @property
    def zero_lhv_moisture(self) -> float | None:
        """W* con la correlación principal (``None`` sin H)."""
        lhv = self.main.lhv_d
        return None if lhv is None else zero_lhv_moisture(lhv)


def applicable_correlations(inputs: HeatingValueInputs) -> tuple[str, ...]:
    """Las correlaciones que se pueden calcular con los análisis cargados."""
    kinds = inputs.analyses
    return tuple(k for k, c in CORRELATIONS.items() if c.analysis in kinds)


def _analysis_pct(inputs: HeatingValueInputs, kind: AnalysisKind) -> dict[str, float]:
    a = inputs.ultimate if kind == "ultimate" else inputs.proximate
    assert a is not None
    return a.pct()


def out_of_range(key: str, inputs: HeatingValueInputs) -> tuple[str, ...]:
    """Avisos de las variables fuera del rango con que se ajustó la correlación."""
    corr = CORRELATIONS[key]
    pct = _analysis_pct(inputs, corr.analysis)
    out: list[str] = []
    for var, lo, hi in corr.ranges:
        v = pct[var]
        if v < lo - 1e-9 or v > hi + 1e-9:
            out.append(
                f"{_VAR_NAMES[var]} = {_fmt(v)} % (base seca), fuera del rango de ajuste "
                f"({_fmt(lo)}–{_fmt(hi)} %)"
            )
    return tuple(out)


def _fmt(x: float, sig: int = 3) -> str:
    """Número con coma decimal para los textos (``12,3``)."""
    s = f"{x:.{sig}g}" if abs(x) < 1000 else f"{x:.0f}"
    return s.replace(".", ",")


def _estimate(
    key: str,
    name: str,
    analysis: AnalysisKind,
    hhv_d: float,
    inputs: HeatingValueInputs,
    warnings: tuple[str, ...] = (),
    in_type: bool = True,
) -> CorrelationEstimate:
    W, A_d, H_d = inputs.moisture, inputs.ash_d, inputs.hydrogen_d
    lhv_d = lhv_ar = None
    if H_d is not None:
        lhv_d, lhv_ar = lhv_from_hhv(hhv_d, H_d, W)
    deviation = None
    if inputs.reference is not None and key != "reference":
        deviation = hhv_d / inputs.reference.hhv_d_J_per_kg - 1.0
    return CorrelationEstimate(
        key=key,
        name=name,
        analysis=analysis,
        hhv_d=hhv_d,
        hhv_ar=basis_value(hhv_d, "ar", W, A_d),
        hhv_daf=basis_value(hhv_d, "daf", W, A_d),
        lhv_d=lhv_d,
        lhv_ar=lhv_ar,
        deviation=deviation,
        warnings=warnings,
        in_type=in_type,
    )


def solve_heating_value(inputs: HeatingValueInputs) -> HeatingValueResult:
    """Todas las correlaciones aplicables, la referencia y las notas.

    Raises
    ------
    ValueError
        Si los datos no son válidos (ver :meth:`HeatingValueInputs.validate`)
        o si una correlación da un PCS que no es positivo.
    """
    inputs.validate()
    keys = applicable_correlations(inputs)
    estimates: list[CorrelationEstimate] = []
    for key in keys:
        corr = CORRELATIONS[key]
        analysis = inputs.ultimate if corr.analysis == "ultimate" else inputs.proximate
        hhv_d = corr.function(analysis)
        estimates.append(
            _estimate(
                key,
                corr.name,
                corr.analysis,
                hhv_d,
                inputs,
                out_of_range(key, inputs),
                inputs.fuel_type in corr.fuel_types,
            )
        )
    main_key = inputs.main or ("channiwala_parikh" if "channiwala_parikh" in keys else keys[0])
    if estimates[[e.key for e in estimates].index(main_key)].hhv_d <= 0.0:
        raise ValueError(
            f"Con estos datos {CORRELATIONS[main_key].name} da un PCS negativo: revisá el análisis "
            "(¿tanto oxígeno o tantas cenizas?)."
        )
    reference = None
    if inputs.reference is not None:
        reference = _estimate(
            "reference", inputs.reference.label, "ultimate", inputs.reference.hhv_d_J_per_kg, inputs
        )
    H_d = inputs.hydrogen_d
    water = None if H_d is None else hydrogen_water_ratio() * H_d
    result = HeatingValueResult(
        inputs=inputs,
        estimates=tuple(estimates),
        main_key=main_key,
        reference=reference,
        water_formed_d=water,
    )
    return replace(result, notes=tuple(heating_value_notes(result)))


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def _pct(x: float, sig: int = 3) -> str:
    return f"{_fmt(100.0 * x, sig)} %"


def heating_value_notes(result: HeatingValueResult) -> list[str]:
    """Notas para el alumno: rangos, tipos de combustible, humedad, cenizas, referencia.

    Una nota que explica por qué falla una correlación reemplaza a sus avisos
    genéricos de rango y de tipo (que igual quedan en la tabla).
    """
    inp = result.inputs
    notes: list[str] = []
    explained: set[str] = set()
    keys = [e.key for e in result.estimates]
    if inp.reference is not None and inp.reference.kind == "exact":
        notes.append(
            "Es una sustancia pura: su PCS exacto sale de las entalpías de formación (vademecum "
            "§16.9). Las correlaciones son para mezclas complejas (carbón, biomasa) cuya h_f no "
            "se conoce; acá sirven para ver cuánto se equivocan."
        )
    if inp.ultimate is not None:
        oxygen = inp.ultimate.O
        if oxygen > 0.10 and "dulong" in keys:
            dev = result.estimate("dulong").deviation
            seen = f" Acá da {_signed_pct(dev)}." if dev is not None else ""
            notes.append(
                f"Con O = {_pct(oxygen)} en base seca, Dulong supone que todo ese oxígeno ya está "
                "unido al H como agua (solo quema el «H disponible», H − O/8): en las 536 "
                "biomasas de Ghugare et al. (2014) subestima 10 % en promedio y 18 % con O > "
                f"45 %. Vale para carbones con O < 10 %.{seen}"
            )
            explained.add("dulong")
        hydrocarbon = oxygen < 0.02 and inp.ultimate.H > 0.10
        if hydrocarbon and "dulong" in keys:
            dev = result.estimate("dulong").deviation
            if dev is not None and dev > 0.04:
                notes.append(
                    f"Dulong da {_signed_pct(dev)}: suma el PCS del C y del H como si estuvieran "
                    "sueltos, pero en un hidrocarburo ya están unidos y esa unión (la h_f del "
                    "combustible) no se libera al quemar. Las correlaciones de ajuste lo "
                    "absorben en sus coeficientes."
                )
                explained.add("dulong")
        if inp.ultimate.H > 0.2515:
            notes.append(
                f"Con H = {_pct(inp.ultimate.H)} el combustible está fuera del rango de H de "
                "Channiwala y Parikh (hasta 25,15 %): las correlaciones de ajuste no extrapolan. "
                "Dulong sí da bien con mucho H, porque su coeficiente del H es el PCS del H₂."
            )
            explained.update({"channiwala_parikh", "boie", "dulong"})
    if inp.fuel_type == "coal" and inp.proximate is not None:
        notes.append(
            "Las correlaciones con el análisis inmediato subestiman los carbones de Argonne "
            "(Vorres, 1990) entre 10 y 23 %, aunque sus valores estén dentro de los rangos de "
            "ajuste: la materia volátil de un carbón bituminoso tiene mucho más H (y vale más) "
            "que la de la biomasa, y una correlación con MV y CF no las distingue. Para carbón "
            "conviene el elemental."
        )
        explained.update({"parikh", "cordero"})
    out_type = [e.name for e in result.estimates if not e.in_type and e.key not in explained]
    if out_type:
        notes.append(
            f"{_join(out_type)} no se {'ajustaron' if len(out_type) > 1 else 'ajustó'} con "
            f"{_TYPE_WITH[inp.fuel_type]}: tomalo como una estimación fuera de su dominio."
        )
    for e in result.estimates:
        if e.warnings and e.key not in explained:
            notes.append(f"{e.name}: {'; '.join(e.warnings)}.")
    if inp.hydrogen_d is None:
        notes.append(
            "Con solo el análisis inmediato no se conoce el H, y sin él no hay PCI: cargá el H "
            "en base seca (de un análisis elemental; en la biomasa ronda el 6 %)."
        )
    W = inp.moisture
    main = result.main
    w_star = result.zero_lhv_moisture
    if main.lhv_ar is not None and main.lhv_ar <= 0.0:
        notes.append(
            f"Con {_pct(W)} de humedad el PCI tal cual da negativo: el combustible no alcanza a "
            f"evaporar su propia agua (el PCI se anula con W* = {_pct(w_star or 0.0)})."
        )
    elif W >= 0.3 and main.lhv_ar is not None and main.lhv_d is not None:
        notes.append(
            f"Con {_pct(W)} de humedad el PCI tal cual ({_mj(main.lhv_ar)} MJ/kg) es el "
            f"{_pct(main.lhv_ar / main.lhv_d)} del seco: el agua no aporta y además se lleva su "
            f"calor latente. Con W* = {_pct(w_star or 0.0)} el PCI se anula."
        )
    if inp.ash_d > 0.2:
        notes.append(
            f"Tiene {_pct(inp.ash_d)} de cenizas en base seca: diluyen el combustible (el PCS "
            "seco baja) y en el hogar se llevan calor, ensucian las superficies y pueden "
            "fundirse."
        )
    if inp.reference is not None and inp.reference.kind == "measured":
        devs = [
            e.deviation
            for e in result.estimates
            if e.deviation is not None and e.key not in explained
        ]
        if devs and min(abs(d) for d in devs) > 0.05:
            notes.append(
                "Ninguna correlación queda a menos de 5 % del PCS medido: puede ser un "
                "combustible distinto de los que se usaron para ajustarlas o un error en los "
                "datos (¿la base del análisis?, ¿las cenizas?, ¿el PCS es seco o tal cual?)."
            )
    return notes


#: «no se ajustó con …» para cada tipo de combustible.
_TYPE_WITH: dict[FuelType, str] = {
    "biomass": "biomasa",
    "char": "carbonizados",
    "coal": "carbones",
    "liquid": "combustibles líquidos",
    "gas": "combustibles gaseosos",
    "waste": "residuos",
}


def _signed_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{'+' if x >= 0 else '−'}{_fmt(abs(100.0 * x), 2)} %"


def _mj(value_J_per_kg: float) -> str:
    return _fmt(value_J_per_kg / 1e6, 4)


def _join(names: Sequence[str]) -> str:
    names = list(names)
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " y " + names[-1]


# ---------------------------------------------------------------------
# Datos de validación
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BiomassSample:
    """Una biomasa de Ghugare et al. (2014): análisis elemental seco y PCS medido seco."""

    name: str
    subset: str
    ultimate: DryUltimate
    hhv_d: float


@cache
def ghugare_dataset() -> tuple[BiomassSample, ...]:
    """Las 536 biomasas de Ghugare et al. (2014) (``data/ghugare2014_biomass.csv``).

    Ghugare, S. B., Tiwary, S., Elangovan, V. y Tambe, S. S. (2014).
    Prediction of higher heating value of solid biomass fuels using artificial
    intelligence formalisms. *BioEnergy Research* 7, 681–692 (tabla S.2 del
    material suplementario), tal como las distribuye el paquete de R
    ``modeldata`` (Posit, licencia MIT). Las cenizas son 100 − (C + H + O +
    N + S), cero si la suma pasa de 100 por redondeo.
    """
    out: list[BiomassSample] = []
    with _GHUGARE_CSV.open(encoding="utf-8") as fh:
        lines = (line for line in fh if not line.startswith("#"))
        for row in csv.DictReader(lines):
            pct = {k: float(row[col]) for k, col in zip(_ULT_KEYS, _GHUGARE_COLS, strict=True)}
            ash = max(0.0, 100.0 - sum(pct.values()))
            out.append(
                BiomassSample(
                    name=row["sample"].strip(),
                    subset=row["dataset"],
                    ultimate=DryUltimate(**{k: v / 100.0 for k, v in pct.items()}, A=ash / 100.0),
                    hhv_d=float(row["HHV"]) * 1e6,
                )
            )
    return tuple(out)


_GHUGARE_COLS = ("carbon", "hydrogen", "oxygen", "nitrogen", "sulfur")


@dataclass(frozen=True)
class DatasetFit:
    """Cómo le va a una correlación con las biomasas de Ghugare et al. (2014)."""

    key: str
    n: int
    aae: float
    abe: float
    within_10: float
    measured: tuple[float, ...]
    predicted: tuple[float, ...]
    oxygen: tuple[float, ...]
    names: tuple[str, ...]

    @property
    def errors(self) -> tuple[float, ...]:
        """Error relativo de cada muestra: (estimado − medido)/medido."""
        return tuple(p / m - 1.0 for p, m in zip(self.predicted, self.measured, strict=True))


@cache
def dataset_fit(key: str) -> DatasetFit:
    """AAE, ABE y % dentro de ±10 % de una correlación elemental en las 536 biomasas.

    Las métricas son las de Channiwala y Parikh (2002): AAE = Σ|e|/n y
    ABE = Σe/n, con e = (estimado − medido)/medido.

    Raises
    ------
    ValueError
        Si la correlación usa el análisis inmediato (los datos son elementales).
    """
    corr = CORRELATIONS[key]
    if corr.analysis != "ultimate":
        raise ValueError(
            f"{corr.name} usa el análisis inmediato y los datos de Ghugare et al. son elementales."
        )
    data = ghugare_dataset()
    measured = tuple(s.hhv_d for s in data)
    predicted = tuple(corr.function(s.ultimate) for s in data)
    errors = [p / m - 1.0 for p, m in zip(predicted, measured, strict=True)]
    n = len(errors)
    return DatasetFit(
        key=key,
        n=n,
        aae=sum(abs(e) for e in errors) / n,
        abe=sum(errors) / n,
        within_10=sum(abs(e) <= 0.10 for e in errors) / n,
        measured=measured,
        predicted=predicted,
        oxygen=tuple(s.ultimate.O for s in data),
        names=tuple(s.name for s in data),
    )


@dataclass(frozen=True)
class ArgonneCoal:
    """Un carbón del Argonne Premium Coal Sample Program (Vorres, 1990), en base seca."""

    key: str
    name: str
    rank: str
    region: str
    moisture: float
    proximate: DryProximate
    ultimate: DryUltimate | None
    hhv_d: float

    @property
    def label(self) -> str:
        return f"{self.name} ({self.rank})"


@cache
def argonne_coals() -> tuple[ArgonneCoal, ...]:
    """Los carbones de Argonne (``data/argonne_premium_coals.csv``).

    Vorres, K. S. (1990). The Argonne Premium Coal Sample Program. *Energy &
    Fuels* 4(5), 420–426, doi:10.1021/ef00023a001, y su *Users Handbook*
    (ANL/PCSP-93/1, 1993). Inmediato y PCS tal cual → base seca con la
    humedad; elemental en base seca con el O por diferencia (incluye el Cl,
    0,03–0,11 %); las cenizas del inmediato pasadas a seco coinciden con las
    del elemental a 0,01 %.
    """
    out: list[ArgonneCoal] = []
    with _ARGONNE_CSV.open(encoding="utf-8") as fh:
        lines = (line for line in fh if not line.startswith("#"))
        for row in csv.DictReader(lines):
            W = float(row["moisture_ar"]) / 100.0
            if row["C_d"]:
                el = {k: float(row[f"{k}_d"]) for k in ("C", "H", "N", "S")}
                A_d = float(row["ash_d"])
                oxygen = 100.0 - sum(el.values()) - A_d  # por diferencia, con el Cl
                ultimate = DryUltimate(
                    **{k: v / 100.0 for k, v in el.items()}, O=oxygen / 100.0, A=A_d / 100.0
                )
                A_prox = A_d / 100.0
            else:
                ultimate = None
                A_prox = float(row["ash_ar"]) / 100.0 / (1.0 - W)
            VM = float(row["vm_ar"]) / 100.0 / (1.0 - W)
            out.append(
                ArgonneCoal(
                    key=row["key"],
                    name=row["name"],
                    rank=row["rank"],
                    region=row["region"],
                    moisture=W,
                    proximate=DryProximate(VM=VM, FC=1.0 - VM - A_prox, A=A_prox),
                    ultimate=ultimate,
                    hhv_d=float(row["hhv_ar_btu_lb"]) * _BTU_PER_LB_J_PER_KG / (1.0 - W),
                )
            )
    return tuple(out)


def argonne_deviations() -> list[dict[str, Any]]:
    """Desvío de cada correlación en cada carbón de Argonne (fracción; ``None`` sin datos)."""
    rows: list[dict[str, Any]] = []
    for coal in argonne_coals():
        row: dict[str, Any] = {"coal": coal, "deviations": {}}
        for key, corr in CORRELATIONS.items():
            analysis = coal.ultimate if corr.analysis == "ultimate" else coal.proximate
            row["deviations"][key] = (
                None if analysis is None else corr.function(analysis) / coal.hhv_d - 1.0
            )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------
# Sustancias puras y /Combustion
# ---------------------------------------------------------------------


def _ultimate_from_fuel(fuel: Fuel) -> DryUltimate:
    el = fuel.elements()
    mass = {e: el[e] * atomic_mass(e) for e in _ULT_KEYS}
    total = sum(mass.values())
    return DryUltimate(**{e: m / total for e, m in mass.items()}, A=0.0)


def pure_substance_inputs(
    fuel: Fuel, fuel_type: FuelType, main: str | None = None
) -> HeatingValueInputs:
    """Una sustancia pura (de la Fase 5) con su PCS exacto de las h_f como referencia."""
    if not fuel.per_mol:
        raise ValueError("Solo los combustibles con especies tienen un PCS exacto.")
    return HeatingValueInputs(
        name=fuel.name,
        fuel_type=fuel_type,
        ultimate=_ultimate_from_fuel(fuel),
        reference=Reference(
            fuel.hhv_per_kg, "exact", "entalpías de formación de NASA (vademecum §16.9)"
        ),
        main=main,
    )


def dry_ultimate_from_analysis(analysis: UltimateAnalysis) -> DryUltimate:
    """El análisis tal cual de la Fase 5 (con la humedad aparte) en base seca."""
    dry = 1.0 - analysis.W
    if dry <= 0.0:
        raise ValueError("Con 100 % de humedad no queda combustible.")
    return DryUltimate(
        C=analysis.C / dry,
        H=analysis.H / dry,
        O=analysis.O / dry,
        N=analysis.N / dry,
        S=analysis.S / dry,
        A=analysis.A / dry,
    )


def estimate_hhv_as_fired(analysis: UltimateAnalysis, key: str) -> float:
    """PCS tal cual (J/kg) de un análisis elemental de la Fase 5 con una correlación elemental.

    Pasa a base seca, aplica la correlación y vuelve: PCS_tc = PCS_s·(1 − W).

    Raises
    ------
    ValueError
        Si la correlación no es elemental o el PCS que da no es positivo.
    """
    corr = CORRELATIONS[key]
    if corr.analysis != "ultimate":
        raise ValueError(f"{corr.name} usa el análisis inmediato, no el elemental.")
    hhv_d = corr.function(dry_ultimate_from_analysis(analysis))
    if hhv_d <= 0.0:
        raise ValueError(
            f"Con este análisis {corr.name} da un PCS negativo: no se puede estimar así."
        )
    return hhv_d * (1.0 - analysis.W)


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------


def _ghugare_example(
    name: str, sample: str, subset: str, fuel_type: FuelType, moisture: float = 0.0
) -> HeatingValueInputs:
    for s in ghugare_dataset():
        if s.name == sample and s.subset == subset:
            u = s.ultimate
            total = sum(u.as_dict().values())
            return HeatingValueInputs(
                name=name,
                fuel_type=fuel_type,
                ultimate=DryUltimate(**{k: v / total for k, v in u.as_dict().items()}),
                moisture=moisture,
                reference=Reference(s.hhv_d, "measured", f"Ghugare et al. (2014): «{sample}»"),
            )
    raise ValueError(f"No está la muestra {sample!r} ({subset}).")


def _argonne_example(key: str, with_ultimate: bool = True) -> HeatingValueInputs:
    coal = next(c for c in argonne_coals() if c.key == key)
    return HeatingValueInputs(
        name=f"{coal.name} ({coal.rank})",
        fuel_type="coal",
        ultimate=coal.ultimate if with_ultimate else None,
        proximate=coal.proximate,
        moisture=coal.moisture,
        reference=Reference(coal.hhv_d, "measured", "Vorres (1990)"),
    )


def _build_examples() -> dict[str, HeatingValueInputs]:
    return {
        "Bagazo de caña (50 % de humedad)": _ghugare_example(
            "Bagazo de caña", "Sugarcane Bagasse", "Testing", "biomass", 0.50
        ),
        "Eucalipto": _ghugare_example(
            "Eucalipto", "Eucalyptus Globulus Wood", "Testing", "biomass"
        ),
        "Cáscara de maní": _ghugare_example(
            "Cáscara de maní", "Peanut Hulls", "Training", "biomass"
        ),
        "Cáscara de arroz": _ghugare_example(
            "Cáscara de arroz", "Rice Hulls", "Training", "biomass"
        ),
        "Orujo de uva": _ghugare_example("Orujo de uva", "Grape Pomace", "Training", "biomass"),
        "Paja de trigo": _ghugare_example("Paja de trigo", "Wheat Straw", "Testing", "biomass"),
        "Carbón vegetal": _ghugare_example("Carbón vegetal", "Charcoal", "Training", "char"),
        "Lodo cloacal seco": _ghugare_example(
            "Lodo cloacal seco", "Sewage Sludge, Dried", "Training", "waste"
        ),
        "Carbón bituminoso Pittsburgh N.º 8": _argonne_example("pittsburgh8"),
        "Carbón bituminoso Illinois N.º 6": _argonne_example("illinois6"),
        "Carbón subbituminoso Wyodak-Anderson": _argonne_example("wyodak"),
        "Lignito Beulah-Zap": _argonne_example("beulahzap"),
        "Carbón bituminoso Pocahontas N.º 3 (solo inmediato)": _argonne_example(
            "pocahontas3", with_ultimate=False
        ),
        "Metano (sustancia pura)": pure_substance_inputs(FUELS["Metano"], "gas"),
        "n-Octano (sustancia pura)": pure_substance_inputs(FUELS["n-Octano (líquido)"], "liquid"),
        "Etanol (sustancia pura)": pure_substance_inputs(FUELS["Etanol (líquido)"], "liquid"),
        "Metanol (sustancia pura)": pure_substance_inputs(FUELS["Metanol (líquido)"], "liquid"),
        "Hidrógeno (sustancia pura)": pure_substance_inputs(FUELS["Hidrógeno"], "gas"),
    }


#: Ejemplos: biomasa de Ghugare et al. (2014), carbones de Argonne y sustancias puras.
HEATING_VALUE_EXAMPLES: dict[str, HeatingValueInputs] = _build_examples()

HEATING_VALUE_EXAMPLE_NOTES: dict[str, str] = {
    "Bagazo de caña (50 % de humedad)": (
        "Bagazo de Ghugare et al. (2014), con el PCS medido en base seca. Sale del trapiche con "
        "~50 % de humedad (Hugot, *Handbook of Cane Sugar Engineering*): mirá cuánto cae el PCI "
        "tal cual y probá el barrido de la humedad."
    ),
    "Eucalipto": (
        "Madera de *Eucalyptus globulus* (Ghugare et al., 2014): poca ceniza y O ≈ 44 %, una "
        "biomasa típica. Compará Dulong con las otras dos."
    ),
    "Cáscara de maní": (
        "Residuo de la industria del maní (Ghugare et al., 2014), que en Córdoba se quema en "
        "calderas."
    ),
    "Cáscara de arroz": (
        "Mucha ceniza (~20 %, casi toda sílice): diluye el combustible y baja el PCS (Ghugare "
        "et al., 2014)."
    ),
    "Orujo de uva": (
        "Residuo de la vinificación (Ghugare et al., 2014): más C y menos O que la madera."
    ),
    "Paja de trigo": "Residuo agrícola con ~12 % de cenizas (Ghugare et al., 2014).",
    "Carbón vegetal": (
        "Biomasa carbonizada (Ghugare et al., 2014): casi todo C y O ≈ 3 %, así que Dulong "
        "vuelve a andar bien."
    ),
    "Lodo cloacal seco": (
        "Lodo de depuradora seco (Ghugare et al., 2014), con 32 % de cenizas y 4,5 % de N: las "
        "correlaciones se ajustaron con pocos residuos así y lo sobreestiman."
    ),
    "Carbón bituminoso Pittsburgh N.º 8": (
        "Carbón de Argonne (Vorres, 1990) con los dos análisis: las elementales aciertan a ±2 % "
        "y las del inmediato subestiman ~20 %."
    ),
    "Carbón bituminoso Illinois N.º 6": (
        "Carbón de Argonne (Vorres, 1990) con 4,8 % de S y 15 % de cenizas en base seca."
    ),
    "Carbón subbituminoso Wyodak-Anderson": (
        "Carbón de bajo rango de Argonne (Vorres, 1990): 28 % de humedad tal cual y O ≈ 16 %."
    ),
    "Lignito Beulah-Zap": (
        "Lignito de Argonne (Vorres, 1990): 32 % de humedad tal cual y O ≈ 18 % en base seca."
    ),
    "Carbón bituminoso Pocahontas N.º 3 (solo inmediato)": (
        "Carbón bituminoso bajo en volátiles de Argonne (Vorres, 1990), con solo el análisis "
        "inmediato: sin el H no hay PCI (cargalo aparte)."
    ),
    "Metano (sustancia pura)": (
        "PCS exacto de las entalpías de formación (vademecum §16.9). Dulong da +11 %: ignora "
        "la h_f del combustible."
    ),
    "n-Octano (sustancia pura)": "PCS exacto de las h_f (Cengel A-27: 47 890 kJ/kg).",
    "Etanol (sustancia pura)": "Un combustible con O (35 %): PCS exacto de las h_f.",
    "Metanol (sustancia pura)": "La mitad de su masa es O: PCS exacto de las h_f.",
    "Hidrógeno (sustancia pura)": (
        "H puro: las correlaciones de ajuste no extrapolan hasta ahí; solo Dulong, que tiene "
        "base física, acierta."
    ),
}


# ---------------------------------------------------------------------
# Barrido de la humedad
# ---------------------------------------------------------------------


def default_moisture_values(n: int = 36) -> list[float]:
    """Humedades tal cual de 0 a 70 %."""
    return [0.70 * k / (n - 1) for k in range(n)]


def moisture_sweep(
    result: HeatingValueResult, values: Sequence[float], key: str | None = None
) -> dict[str, list[float | None]]:
    """PCS y PCI tal cual en función de la humedad, con la correlación ``key`` (o la principal).

    El análisis seco no cambia: cada kg tal cual tiene (1 − W) kg secos y W kg
    de agua que se evapora. ``"PCI"`` es ``None`` sin H.
    """
    est = result.estimate(key or result.main_key)
    hhv: list[float | None] = []
    lhv: list[float | None] = []
    for W in values:
        if not 0.0 <= W < 1.0:
            raise ValueError("La humedad del barrido tiene que estar entre 0 y 100 %.")
        hhv.append(est.hhv_d * (1.0 - W))
        if est.lhv_d is None:
            lhv.append(None)
        else:
            lhv.append(est.lhv_d * (1.0 - W) - water_hfg_J_per_kg() * W)
    return {"W": list(values), "PCS": hhv, "PCI": lhv}


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _value(value_si: float | None, kind: QuantityKind, system: UnitSystem) -> Any:
    if value_si is None:
        return None
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def _analysis_dict(
    dry: Mapping[str, float], moisture: float, ash_d: float
) -> dict[str, dict[str, float]]:
    return {
        BASES[b]: {k: 100.0 * v for k, v in to_basis(dry, b, moisture, ash_d).items()}
        for b in ("ar", "d", "daf")
    }


def heating_value_to_dict(result: HeatingValueResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (PCS y PCI en ``system``; análisis en %)."""
    inp = result.inputs
    eh: QuantityKind = "specific_enthalpy"
    out: dict[str, Any] = {
        "modelo": "correlaciones del PCS (base seca); PCI con h_fg a 25 °C (vademecum §16.9)",
        "combustible": inp.name,
        "tipo": FUEL_TYPES[inp.fuel_type],
        "humedad_tal_cual_%": 100.0 * inp.moisture,
        "cenizas_base_seca_%": 100.0 * inp.ash_d,
    }
    if inp.ultimate is not None:
        out["analisis_elemental_%"] = _analysis_dict(
            inp.ultimate.as_dict(), inp.moisture, inp.ash_d
        )
    if inp.proximate is not None:
        out["analisis_inmediato_%"] = _analysis_dict(
            inp.proximate.as_dict(), inp.moisture, inp.ash_d
        )
    if inp.H_d is not None:
        out["H_base_seca_%"] = 100.0 * inp.H_d
    if result.reference is not None:
        ref = result.reference
        out["referencia"] = {
            "tipo": ref.name,
            "fuente": inp.reference.source if inp.reference else "",
            "PCS_seca": _value(ref.hhv_d, eh, system),
            "PCS_tal_cual": _value(ref.hhv_ar, eh, system),
            "PCS_seca_sin_cenizas": _value(ref.hhv_daf, eh, system),
            "PCI_tal_cual": _value(ref.lhv_ar, eh, system),
        }
    out["correlacion_principal"] = result.main.name
    out["correlaciones"] = [
        {
            "nombre": e.name,
            "analisis": ANALYSIS_KINDS[e.analysis],
            "PCS_seca": _value(e.hhv_d, eh, system),
            "PCS_tal_cual": _value(e.hhv_ar, eh, system),
            "PCS_seca_sin_cenizas": _value(e.hhv_daf, eh, system),
            "PCI_seca": _value(e.lhv_d, eh, system),
            "PCI_tal_cual": _value(e.lhv_ar, eh, system),
            "desvio_%": None if e.deviation is None else 100.0 * e.deviation,
            "dentro_de_su_tipo": e.in_type,
            "avisos": list(e.warnings),
            "referencia": CORRELATIONS[e.key].reference,
        }
        for e in result.estimates
    ]
    if result.water_formed_d is not None:
        out["agua_formada_kg_por_kg_seco"] = result.water_formed_d
    w_star = result.zero_lhv_moisture
    if w_star is not None:
        out["humedad_con_PCI_nulo_%"] = 100.0 * w_star
    out["notas"] = list(result.notes)
    return out
