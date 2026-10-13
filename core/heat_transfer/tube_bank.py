"""Banco de tubos en flujo cruzado: Zukauskas (Fase 12.1).

Çengel y Ghajar (2015), §7-4; Incropera et al. (2007), §7.6. Un fluido que llega con
velocidad V y temperatura T_ent cruza un banco de N_L filas (en el sentido del flujo) de
N_T tubos cada una, de diámetro D y largo L, con pasos transversal S_T y longitudinal S_L,
alineados o en tresbolillo; los tubos están a T_s, como en los libros.

- **La velocidad máxima**: alineado, V_máx = S_T/(S_T − D)·V; en tresbolillo pasa por la
  diagonal si S_D = √(S_L² + (S_T/2)²) < (S_T + D)/2, y entonces V_máx = S_T/(2·(S_D − D))·V.
- **Zukauskas** (1972, 1987): Nu = C·Re^m·Pr^n·(Pr/Pr_s)^¼, con Re = ρ·V_máx·D/μ, las
  propiedades a la media de la entrada y la salida (iterada) y Pr_s a T_s. Hay dos versiones
  de las constantes: la de Çengel y Ghajar (tabla 7-2, con la corrección F de la 7-3 con
  menos de 16 filas) y la de Incropera (tabla 7.5, con C₂ de la 7.6 con menos de 20).
- **La salida**: T_s − T_sal = (T_s − T_ent)·exp(−A_s·h/(ṁ·c_p)), con ṁ = ρ_ent·V·N_T·S_T·L y
  A_s = N_L·N_T·π·D·L; Q̇ = h·A_s·ΔT_ml.
- **La caída de presión**: Δp = N_L·χ·f·ρ·V_máx²/2, con f y χ de los gráficos de Zukauskas
  (1972), los de Incropera (figs. 7.13 y 7.14) y Çengel y Ghajar (fig. 7-27). No tienen
  fórmula: se leen de ``data/zukauskas_tube_bank_friction.csv`` (la digitalización de la
  biblioteca ht, licencia MIT), interpolando como se lee un gráfico.

Todo en SI.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from core.fluids import fluid_with_article
from core.heat_transfer.convection import (
    Properties,
    _check_wall_phase,
    _saturation_T,
    convection_fluids,
    fluid_properties,
)
from core.units_system import UnitSystem, convert_from_si, unit_label

__all__ = [
    "ARRANGEMENTS",
    "CONSTANT_SOURCES",
    "FRICTION_DATA_PATH",
    "TUBE_BANK_EXAMPLES",
    "Arrangement",
    "BankProperties",
    "ConstantSource",
    "Curve",
    "FrictionReading",
    "SweepPoint",
    "TubeBankExample",
    "TubeBankInputs",
    "TubeBankResult",
    "ZukauskasFit",
    "compare_arrangements",
    "friction_chart",
    "friction_curves",
    "max_velocity",
    "nu_zukauskas",
    "row_correction",
    "solve_tube_bank",
    "tube_bank_fluids",
    "tube_bank_notes",
    "tube_bank_to_dict",
    "velocity_sweep",
    "zukauskas_fit",
]

Arrangement = Literal["aligned", "staggered"]
ConstantSource = Literal["cengel", "incropera"]

ARRANGEMENTS: dict[Arrangement, str] = {"aligned": "Alineado", "staggered": "Tresbolillo"}
CONSTANT_SOURCES: dict[ConstantSource, str] = {
    "cengel": "Çengel y Ghajar (tablas 7-2 y 7-3)",
    "incropera": "Incropera (tablas 7.5 y 7.6)",
}

FRICTION_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "zukauskas_tube_bank_friction.csv"
)

#: El arreglo de referencia de cada gráfico de χ (χ = 1 ahí): el alineado cuadrado,
#: (P_T − 1)/(P_L − 1) = 1, y el tresbolillo equilátero, P_T/P_L = 2/√3.
_CHI_REFERENCE: dict[Arrangement, float] = {"aligned": 1.0, "staggered": 2.0 / math.sqrt(3.0)}
_RANGE_TOLERANCE = 0.01


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


def _power10(x: float) -> str:
    """10², 10³…: los Re de las curvas del gráfico."""
    exponent = round(math.log10(x))
    sup = str(exponent).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    return f"10{sup}"


# ---------------------------------------------------------------------
# Zukauskas: las constantes
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ZukauskasFit:
    """Un tramo de Zukauskas: Nu = C·(S_T/S_L)^p·Re^m·Pr^n·(Pr/Pr_s)^¼.

    ``single_cylinder``: Incropera aproxima el tramo de 10² a 10³ con un cilindro aislado
    (Zukauskas, tabla 7.4: C = 0,51, m = 0,5; n = 0,37 con Pr ≤ 10 y 0,36 si no).
    """

    C: float
    p: float
    m: float
    n: float
    Re_low: float
    Re_high: float
    single_cylinder: bool = False

    def coefficient(self, ST_over_SL: float) -> float:
        """C·(S_T/S_L)^p."""
        return self.C * ST_over_SL**self.p if self.p else self.C


# Çengel y Ghajar (2015), tabla 7-2 (de Zukauskas, 1987): N_L ≥ 16, 0,7 < Pr < 500,
# 0 < Re < 2·10⁶. El tramo de Re alto del alineado lleva Pr^0,4.
_CENGEL: dict[Arrangement, tuple[ZukauskasFit, ...]] = {
    "aligned": (
        ZukauskasFit(0.9, 0.0, 0.4, 0.36, 0.0, 1.0e2),
        ZukauskasFit(0.52, 0.0, 0.5, 0.36, 1.0e2, 1.0e3),
        ZukauskasFit(0.27, 0.0, 0.63, 0.36, 1.0e3, 2.0e5),
        ZukauskasFit(0.033, 0.0, 0.8, 0.4, 2.0e5, 2.0e6),
    ),
    "staggered": (
        ZukauskasFit(1.04, 0.0, 0.4, 0.36, 0.0, 5.0e2),
        ZukauskasFit(0.71, 0.0, 0.5, 0.36, 5.0e2, 1.0e3),
        ZukauskasFit(0.35, 0.2, 0.6, 0.36, 1.0e3, 2.0e5),
        ZukauskasFit(0.031, 0.2, 0.8, 0.36, 2.0e5, 2.0e6),
    ),
}
# Incropera et al. (2007), tabla 7.5 (de Zukauskas, 1972): N_L ≥ 20, 0,7 ≲ Pr ≲ 500,
# 10 ≲ Re ≲ 2·10⁶. En tresbolillo con S_T/S_L > 2, C = 0,40 sin el factor del paso.
_INCROPERA: dict[Arrangement, tuple[ZukauskasFit, ...]] = {
    "aligned": (
        ZukauskasFit(0.80, 0.0, 0.40, 0.36, 10.0, 1.0e2),
        ZukauskasFit(0.51, 0.0, 0.5, 0.37, 1.0e2, 1.0e3, single_cylinder=True),
        ZukauskasFit(0.27, 0.0, 0.63, 0.36, 1.0e3, 2.0e5),
        ZukauskasFit(0.021, 0.0, 0.84, 0.36, 2.0e5, 2.0e6),
    ),
    "staggered": (
        ZukauskasFit(0.90, 0.0, 0.40, 0.36, 10.0, 1.0e2),
        ZukauskasFit(0.51, 0.0, 0.5, 0.37, 1.0e2, 1.0e3, single_cylinder=True),
        ZukauskasFit(0.35, 0.2, 0.60, 0.36, 1.0e3, 2.0e5),
        ZukauskasFit(0.022, 0.0, 0.84, 0.36, 2.0e5, 2.0e6),
    ),
}

# Corrección por la cantidad de filas: Çengel y Ghajar, tabla 7-3 (F, N_L < 16) e
# Incropera, tabla 7.6 (C₂, N_L < 20), las dos para Re ≳ 10³.
_ROWS_CENGEL = (1, 2, 3, 4, 5, 7, 10, 13, 16)
_F_CENGEL: dict[Arrangement, tuple[float, ...]] = {
    "aligned": (0.70, 0.80, 0.86, 0.90, 0.93, 0.96, 0.98, 0.99, 1.0),
    "staggered": (0.64, 0.76, 0.84, 0.89, 0.93, 0.96, 0.98, 0.99, 1.0),
}
_ROWS_INCROPERA = (1, 2, 3, 4, 5, 7, 10, 13, 16, 20)
_C2_INCROPERA: dict[Arrangement, tuple[float, ...]] = {
    "aligned": (0.70, 0.80, 0.86, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 1.0),
    "staggered": (0.64, 0.76, 0.84, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99, 1.0),
}


def zukauskas_fit(
    Re: float, arrangement: Arrangement, source: ConstantSource, ST_over_SL: float, Pr: float
) -> ZukauskasFit:
    """El tramo de Zukauskas que corresponde a Re (Çengel y Ghajar, tabla 7-2; Incropera,
    tabla 7.5).

    Fuera del rango de Re se usa el tramo del borde (la página lo avisa).
    """
    table = (_CENGEL if source == "cengel" else _INCROPERA)[arrangement]
    fit = next((t for t in table if Re < t.Re_high), table[-1])
    if fit.single_cylinder and Pr > 10.0:
        fit = replace(fit, n=0.36)
    if source == "incropera" and arrangement == "staggered" and fit.p and ST_over_SL > 2.0:
        fit = replace(fit, C=0.40, p=0.0)
    return fit


def nu_zukauskas(
    Re: float,
    Pr: float,
    Pr_s: float,
    arrangement: Arrangement,
    ST_over_SL: float,
    source: ConstantSource = "cengel",
) -> float:
    """Nu de un banco de tubos con 16 filas o más (20 en Incropera), sin la corrección por
    filas: Nu = C·(S_T/S_L)^p·Re^m·Pr^n·(Pr/Pr_s)^¼ (Zukauskas, 1972 y 1987; Çengel y Ghajar,
    tabla 7-2; Incropera, tabla 7.5)."""
    fit = zukauskas_fit(Re, arrangement, source, ST_over_SL, Pr)
    return fit.coefficient(ST_over_SL) * Re**fit.m * Pr**fit.n * (Pr / Pr_s) ** 0.25


def row_correction(N_L: int, arrangement: Arrangement, source: ConstantSource = "cengel") -> float:
    """F (Çengel y Ghajar, tabla 7-3) o C₂ (Incropera, tabla 7.6): Nu_N = F·Nu, interpolado
    entre las filas de la tabla; 1 con 16 filas o más (20 en Incropera)."""
    rows, table = (
        (_ROWS_CENGEL, _F_CENGEL[arrangement])
        if source == "cengel"
        else (_ROWS_INCROPERA, _C2_INCROPERA[arrangement])
    )
    if N_L >= rows[-1]:
        return 1.0
    pairs = list(zip(rows, table, strict=True))
    for (n0, f0), (n1, f1) in zip(pairs, pairs[1:], strict=False):
        if n0 <= N_L <= n1:
            return f0 + (f1 - f0) * (N_L - n0) / (n1 - n0)
    return table[0]


def max_velocity(
    V: float, D: float, S_T: float, S_L: float, arrangement: Arrangement
) -> tuple[float, float | None, bool]:
    """(V_máx, S_D, por la diagonal) (Çengel y Ghajar, §7-4; Incropera, §7.6).

    En tresbolillo el fluido pasa por la diagonal si S_D < (S_T + D)/2 (Çengel y Ghajar):
    entonces V_máx = S_T/(2·(S_D − D))·V; si no, por el plano transversal como el alineado.
    """
    if arrangement == "aligned":
        return S_T / (S_T - D) * V, None, False
    S_D = math.hypot(S_L, S_T / 2.0)
    if S_D < (S_T + D) / 2.0:
        return S_T / (2.0 * (S_D - D)) * V, S_D, True
    return S_T / (S_T - D) * V, S_D, False


# ---------------------------------------------------------------------
# f y χ: los gráficos de Zukauskas
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def _chart_data() -> dict[tuple[str, float], tuple[tuple[float, ...], tuple[float, ...]]]:
    """Las curvas de ``data/zukauskas_tube_bank_friction.csv``: (x, y) por (gráfico, curva)."""
    curves: dict[tuple[str, float], tuple[list[float], list[float]]] = {}
    with FRICTION_DATA_PATH.open(encoding="utf-8") as fh:
        rows = csv.reader(line for line in fh if not line.startswith("#"))
        next(rows)
        for chart, curve, x, y in rows:
            xs, ys = curves.setdefault((chart, float(curve)), ([], []))
            xs.append(float(x))
            ys.append(float(y))
    return {key: (tuple(xs), tuple(ys)) for key, (xs, ys) in curves.items()}


def _curve_values(chart: str) -> tuple[float, ...]:
    return tuple(sorted(c for (name, c) in _chart_data() if name == chart))


def _read_curve(chart: str, curve: float, x: float, extrapolate: bool) -> tuple[float, bool]:
    """y(x) sobre una curva, lineal en log–log; fuera de su rango, recta por el último
    cuarto de década (``extrapolate``) o el valor del borde. Devuelve (y, fuera)."""
    xs, ys = _chart_data()[(chart, curve)]
    lx = math.log(x)
    if xs[0] <= x <= xs[-1]:
        for i in range(1, len(xs)):
            if x <= xs[i]:
                t = (lx - math.log(xs[i - 1])) / (math.log(xs[i]) - math.log(xs[i - 1]))
                return math.exp((1 - t) * math.log(ys[i - 1]) + t * math.log(ys[i])), False
    low = x < xs[0]
    if not extrapolate:
        return (ys[0] if low else ys[-1]), True
    # la pendiente del extremo, entre el borde y el punto a ~¼ de década
    j0 = 0 if low else len(xs) - 1
    step = 1 if low else -1
    j1 = j0
    while 0 <= j1 + step < len(xs) and abs(math.log10(xs[j1] / xs[j0])) < 0.25:
        j1 += step
    slope = math.log(ys[j1] / ys[j0]) / math.log(xs[j1] / xs[j0])
    return ys[j0] * (x / xs[j0]) ** slope, True


@dataclass(frozen=True)
class FrictionReading:
    """f y χ leídos de los gráficos de Zukauskas, con lo que hizo falta para leerlos."""

    f: float
    chi: float
    pitch: float
    ratio: float
    f_curves: tuple[float, float]
    chi_curves: tuple[float, float]
    warnings: tuple[str, ...]


def friction_chart(Re: float, arrangement: Arrangement, P_T: float, P_L: float) -> FrictionReading:
    """f y χ de Zukauskas (1972), como se leen en los gráficos de Incropera (figs. 7.13 y
    7.14) o Çengel y Ghajar (fig. 7-27): Δp = N_L·χ·f·ρ·V_máx²/2.

    - **f**: las curvas son del alineado cuadrado (por P_L = S_L/D) y del tresbolillo
      equilátero (por P_T = S_T/D), de 1,25 a 2,5; se interpola lineal entre las dos curvas
      vecinas y, sobre cada una, en log–log. Fuera de 1,25–2,5 se lee la curva del borde;
      fuera del rango de Re de una curva, se prolonga su tramo final.
    - **χ** corrige otros pasos: contra (P_T − 1)/(P_L − 1) en el alineado y contra P_T/P_L
      en el tresbolillo, con una curva por Re (10³ a 10⁶ y 10² a 10⁵); entre curvas se
      interpola en log Re. χ = 1 en el arreglo de referencia por definición: la
      digitalización se aparta hasta 5 % ahí, así que cada curva se divide por su valor en
      la referencia.
    """
    warnings: list[str] = []
    if arrangement == "aligned":
        f_chart, chi_chart = "f_aligned", "chi_aligned"
    else:
        f_chart, chi_chart = "f_staggered", "chi_staggered"
    name = "P_L" if arrangement == "aligned" else "P_T"
    pitch = P_L if arrangement == "aligned" else P_T
    curves = _curve_values(f_chart)
    used = min(max(pitch, curves[0]), curves[-1])
    if used != pitch:
        warnings.append(
            f"{name} = {_num(pitch)} está fuera del gráfico de f ({_num(curves[0])} a "
            f"{_num(curves[-1])}): se lee la curva de {_num(used)}."
        )
    c0 = max(c for c in curves if c <= used)
    c1 = min(c for c in curves if c >= used)
    f0, out0 = _read_curve(f_chart, c0, Re, extrapolate=True)
    f1, out1 = _read_curve(f_chart, c1, Re, extrapolate=True)
    for c, out in ((c0, out0), ((c1, out1) if c1 != c0 else (c0, False))):
        if out:
            xs = _chart_data()[(f_chart, c)][0]
            warnings.append(
                f"Re = {_num(Re)} está fuera de la curva de {name} = {_num(c)} del gráfico de f "
                f"(Re de {_num(xs[0])} a {_num(xs[-1])}): se prolonga su tramo final."
            )
    f = f0 if c1 == c0 else f0 + (f1 - f0) * (used - c0) / (c1 - c0)

    ref = _CHI_REFERENCE[arrangement]
    if arrangement == "aligned":
        ratio = (P_T - 1.0) / (P_L - 1.0)
        axis = "(P_T − 1)/(P_L − 1)"
    else:
        ratio = P_T / P_L
        axis = "P_T/P_L"
    re_curves = _curve_values(chi_chart)
    Re_used = min(max(Re, re_curves[0]), re_curves[-1])
    if Re_used != Re:
        warnings.append(
            f"El gráfico de χ tiene curvas de Re = {_power10(re_curves[0])} a "
            f"{_power10(re_curves[-1])}: con Re = {_num(Re)} se usa la de "
            f"{_power10(Re_used)}."
        )
    r0 = max(c for c in re_curves if c <= Re_used)
    r1 = min(c for c in re_curves if c >= Re_used)
    chis = []
    out_ratio = False
    for rc in (r0, r1):
        value, out = _read_curve(chi_chart, rc, ratio, extrapolate=False)
        reference, _ = _read_curve(chi_chart, rc, ref, extrapolate=False)
        chis.append(value / reference)
        out_ratio = out_ratio or out
    if out_ratio:
        xs = _chart_data()[(chi_chart, r0)][0]
        warnings.append(
            f"χ: {axis} = {_num(ratio)} está fuera del gráfico ({_num(xs[0])} a "
            f"{_num(xs[-1])}): se usa el valor del borde."
        )
    if r1 == r0:
        chi = chis[0]
    else:
        t = math.log(Re_used / r0) / math.log(r1 / r0)
        chi = chis[0] + (chis[1] - chis[0]) * t
    if abs(ratio - ref) < 1e-9:
        chi = 1.0
    return FrictionReading(f, chi, pitch, ratio, (c0, c1), (r0, r1), tuple(warnings))


Curve = tuple[tuple[float, ...], tuple[float, ...]]


def friction_curves(arrangement: Arrangement) -> dict[float, Curve]:
    """Las curvas de f del gráfico (Re, f) por P_L o P_T, para dibujarlo."""
    chart = "f_aligned" if arrangement == "aligned" else "f_staggered"
    return {c: _chart_data()[(chart, c)] for c in _curve_values(chart)}


# ---------------------------------------------------------------------
# El banco
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BankProperties:
    """Propiedades dadas en vez de las de CoolProp (las de un enunciado): ρ, μ, k, c_p y Pr
    a la temperatura media, Pr_s a la de la pared y ρ a la entrada (para el caudal)."""

    rho_kg_per_m3: float
    mu_Pa_s: float
    k_W_per_mK: float
    cp_J_per_kgK: float
    Pr_s: float
    rho_in_kg_per_m3: float
    Pr: float | None = None


@dataclass(frozen=True)
class TubeBankInputs:
    """Un banco de N_L × N_T tubos de diámetro D y largo L a T_s, que cruza un fluido."""

    fluid: str
    p_Pa: float
    V_m_s: float
    T_in_K: float
    T_s_K: float
    D_m: float
    S_T_m: float
    S_L_m: float
    N_L: int
    N_T: int
    L_m: float = 1.0
    arrangement: Arrangement = "staggered"
    source: ConstantSource = "cengel"
    properties: BankProperties | None = None


@dataclass(frozen=True)
class TubeBankResult:
    """El banco resuelto."""

    inputs: TubeBankInputs
    rho_kg_per_m3: float
    mu_Pa_s: float
    k_W_per_mK: float
    cp_J_per_kgK: float
    Pr: float
    Pr_s: float
    rho_in_kg_per_m3: float
    T_mean_K: float
    V_max_m_s: float
    S_D_m: float | None
    diagonal: bool
    Re: float
    fit: ZukauskasFit
    C: float
    Nu_inf: float
    F: float
    Nu: float
    h_W_per_m2K: float
    T_out_K: float
    friction: FrictionReading
    warnings: tuple[str, ...]
    iterations: int

    @property
    def N(self) -> int:
        return self.inputs.N_L * self.inputs.N_T

    @property
    def area_m2(self) -> float:
        """A_s = N_L·N_T·π·D·L."""
        i = self.inputs
        return self.N * math.pi * i.D_m * i.L_m

    @property
    def m_dot_kg_s(self) -> float:
        """ṁ = ρ_ent·V·N_T·S_T·L (Çengel y Ghajar, §7-4)."""
        i = self.inputs
        return self.rho_in_kg_per_m3 * i.V_m_s * i.N_T * i.S_T_m * i.L_m

    @property
    def dT_lm_K(self) -> float:
        """ΔT_ml = (ΔT_ent − ΔT_sal)/ln(ΔT_ent/ΔT_sal), con ΔT = T_s − T."""
        i = self.inputs
        d_in, d_out = i.T_s_K - i.T_in_K, i.T_s_K - self.T_out_K
        if abs(d_in - d_out) <= 1e-12 * abs(d_in):
            return d_in
        return (d_in - d_out) / math.log(d_in / d_out)

    @property
    def Q_W(self) -> float:
        """Q̇ = h·A_s·ΔT_ml: positivo si el banco le da calor al fluido."""
        return self.h_W_per_m2K * self.area_m2 * self.dT_lm_K

    @property
    def Q_per_length_W_m(self) -> float:
        return self.Q_W / self.inputs.L_m

    @property
    def f(self) -> float:
        return self.friction.f

    @property
    def chi(self) -> float:
        return self.friction.chi

    @property
    def dp_Pa(self) -> float:
        """Δp = N_L·χ·f·ρ·V_máx²/2 (Zukauskas; Incropera, ec. 7.65)."""
        return self.inputs.N_L * self.chi * self.f * self.rho_kg_per_m3 * self.V_max_m_s**2 / 2.0

    @property
    def P_T(self) -> float:
        return self.inputs.S_T_m / self.inputs.D_m

    @property
    def P_L(self) -> float:
        return self.inputs.S_L_m / self.inputs.D_m


def tube_bank_fluids() -> tuple[str, ...]:
    """Los fluidos de CoolProp con propiedades de transporte (los de la convección)."""
    return convection_fluids()


def _validate(i: TubeBankInputs) -> None:
    if i.arrangement not in ARRANGEMENTS:
        raise ValueError(f"Arreglo desconocido: {i.arrangement!r}.")
    if i.source not in CONSTANT_SOURCES:
        raise ValueError(f"Versión de las constantes desconocida: {i.source!r}.")
    if not (i.D_m > 0.0 and i.L_m > 0.0):
        raise ValueError("El diámetro y el largo de los tubos tienen que ser positivos.")
    if not (i.V_m_s > 0.0):
        raise ValueError("La velocidad del fluido tiene que ser positiva.")
    if not (i.T_in_K > 0.0 and i.T_s_K > 0.0 and i.p_Pa > 0.0):
        raise ValueError("Las temperaturas y la presión tienen que ser positivas.")
    if i.N_L < 1 or i.N_T < 1:
        raise ValueError("Hace falta al menos una fila y un tubo por fila.")
    if i.S_T_m <= i.D_m:
        raise ValueError(
            f"El paso transversal S_T = {_num(i.S_T_m * 1e3)} mm no supera al diámetro "
            f"D = {_num(i.D_m * 1e3)} mm: los tubos de una fila se tocan y el fluido no pasa."
        )
    if i.arrangement == "aligned" and i.S_L_m <= i.D_m:
        raise ValueError(
            f"El paso longitudinal S_L = {_num(i.S_L_m * 1e3)} mm no supera al diámetro: "
            "los tubos de dos filas seguidas se tocan."
        )
    if i.arrangement == "staggered" and math.hypot(i.S_L_m, i.S_T_m / 2.0) <= i.D_m:
        raise ValueError(
            "La distancia en diagonal S_D = √(S_L² + (S_T/2)²) no supera al diámetro: los "
            "tubos de dos filas seguidas se superponen. Agrandá S_L o S_T."
        )
    if abs(i.T_s_K - i.T_in_K) < 1e-9:
        raise ValueError(
            "Los tubos y el fluido entran a la misma temperatura: no hay transferencia de calor."
        )


def _properties(i: TubeBankInputs, T_K: float, where: str) -> Properties:
    try:
        return fluid_properties(i.fluid, T_K, i.p_Pa)
    except ValueError as exc:
        raise ValueError(f"Con estos datos {where} llegaría a {_degC(T_K)}. {exc}") from exc


def solve_tube_bank(inputs: TubeBankInputs) -> TubeBankResult:
    """El banco de tubos con Zukauskas (Çengel y Ghajar, §7-4; Incropera, §7.6).

    Con las propiedades de CoolProp, la temperatura media (T_ent + T_sal)/2 fija las
    propiedades y la salida depende de h: se itera hasta 10⁻⁹ K. Con ``properties`` (las de
    un enunciado) no hay nada que iterar.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno (geometría imposible, el fluido cambia de fase).
    """
    i = inputs
    _validate(i)
    ST_SL = i.S_T_m / i.S_L_m
    V_max, S_D, diagonal = max_velocity(i.V_m_s, i.D_m, i.S_T_m, i.S_L_m, i.arrangement)
    F = row_correction(i.N_L, i.arrangement, i.source)
    A_s = i.N_L * i.N_T * math.pi * i.D_m * i.L_m
    T_sat = None if i.properties is not None else _saturation_T(i.fluid, i.p_Pa)
    T_out = i.T_in_K
    iterations = 0
    while True:
        iterations += 1
        T_m = 0.5 * (i.T_in_K + T_out)
        if i.properties is None:
            props = _properties(i, T_m, "la temperatura media del fluido")
            rho, mu, k, cp, Pr = (
                props.rho_kg_per_m3,
                props.mu_Pa_s,
                props.k_W_per_mK,
                props.cp_J_per_kgK,
                props.Pr,
            )
            Pr_s = _properties(i, i.T_s_K, "la pared").Pr
            rho_in = _properties(i, i.T_in_K, "la entrada").rho_kg_per_m3
        else:
            g = i.properties
            rho, mu, k, cp, Pr_s, rho_in = (
                g.rho_kg_per_m3,
                g.mu_Pa_s,
                g.k_W_per_mK,
                g.cp_J_per_kgK,
                g.Pr_s,
                g.rho_in_kg_per_m3,
            )
            Pr = g.Pr if g.Pr is not None else mu * cp / k
        Re = rho * V_max * i.D_m / mu
        fit = zukauskas_fit(Re, i.arrangement, i.source, ST_SL, Pr)
        C = fit.coefficient(ST_SL)
        Nu_inf = C * Re**fit.m * Pr**fit.n * (Pr / Pr_s) ** 0.25
        Nu = F * Nu_inf
        h = Nu * k / i.D_m
        m_dot = rho_in * i.V_m_s * i.N_T * i.S_T_m * i.L_m
        T_new = i.T_s_K - (i.T_s_K - i.T_in_K) * math.exp(-A_s * h / (m_dot * cp))
        if T_sat is not None and (i.T_in_K - T_sat) * (T_new - T_sat) <= 0.0:
            raise ValueError(
                f"{fluid_with_article(i.fluid).capitalize()} saldría a {_degC(T_new)} y a "
                f"{_num(i.p_Pa / 1e5)} bar cambia de fase a {_degC(T_sat)}: Zukauskas es de "
                "una sola fase. Achicá el banco o subí la velocidad."
            )
        converged = i.properties is not None or abs(T_new - T_out) < 1e-9
        T_out = T_new
        if converged:
            break
        if iterations > 200:  # pragma: no cover — converge en decenas de vueltas
            raise ValueError("La temperatura media del fluido no convergió.")
    friction = friction_chart(Re, i.arrangement, i.S_T_m / i.D_m, i.S_L_m / i.D_m)
    warnings = list(_warnings(i, Re, Pr, ST_SL, fit))
    if i.properties is None:
        wall = _check_wall_phase(i.fluid, i.T_s_K, 0.5 * (i.T_in_K + T_out), i.p_Pa)
        if wall:
            warnings.append(wall)
    warnings.extend(friction.warnings)
    return TubeBankResult(
        inputs=i,
        rho_kg_per_m3=rho,
        mu_Pa_s=mu,
        k_W_per_mK=k,
        cp_J_per_kgK=cp,
        Pr=Pr,
        Pr_s=Pr_s,
        rho_in_kg_per_m3=rho_in,
        T_mean_K=0.5 * (i.T_in_K + T_out),
        V_max_m_s=V_max,
        S_D_m=S_D,
        diagonal=diagonal,
        Re=Re,
        fit=fit,
        C=C,
        Nu_inf=Nu_inf,
        F=F,
        Nu=Nu,
        h_W_per_m2K=h,
        T_out_K=T_out,
        friction=friction,
        warnings=tuple(warnings),
        iterations=iterations,
    )


def _warnings(
    i: TubeBankInputs, Re: float, Pr: float, ST_SL: float, fit: ZukauskasFit
) -> list[str]:
    out: list[str] = []
    lo, hi = (0.7, 500.0)
    if not (lo * (1 - _RANGE_TOLERANCE) <= Pr <= hi * (1 + _RANGE_TOLERANCE)):
        out.append(f"Pr = {_num(Pr)} está fuera del rango de Zukauskas (0,7 a 500).")
    Re_lo = 0.0 if i.source == "cengel" else 10.0
    if Re < Re_lo * (1 - _RANGE_TOLERANCE) or Re > 2.0e6 * (1 + _RANGE_TOLERANCE):
        out.append(
            f"Re = {_num(Re)} está fuera del rango de Zukauskas "
            f"({'0' if i.source == 'cengel' else '10'} a 2·10⁶): se usa el tramo del borde."
        )
    rows_limit = 16 if i.source == "cengel" else 20
    if i.N_L < rows_limit and Re < 1.0e3:
        out.append(
            f"La corrección por la cantidad de filas ({i.N_L} < {rows_limit}) es para "
            f"Re ≳ 10³; con Re = {_num(Re)} es aproximada."
        )
    if i.source == "incropera" and i.arrangement == "aligned" and ST_SL < 0.7:
        out.append(
            f"S_T/S_L = {_num(ST_SL)} < 0,7: Incropera advierte que el arreglo alineado así "
            "transfiere mal; conviene el tresbolillo."
        )
    return out


# ---------------------------------------------------------------------
# Comparaciones, barrido y notas
# ---------------------------------------------------------------------


def compare_arrangements(inputs: TubeBankInputs) -> dict[Arrangement, TubeBankResult]:
    """El mismo banco alineado y en tresbolillo (mismos pasos), para comparar h, Q̇ y Δp."""
    out: dict[Arrangement, TubeBankResult] = {}
    for arrangement in ARRANGEMENTS:
        try:
            out[arrangement] = solve_tube_bank(replace(inputs, arrangement=arrangement))
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class SweepPoint:
    V_m_s: float
    h_W_per_m2K: float
    Q_W: float
    dp_Pa: float


def velocity_sweep(inputs: TubeBankInputs, n: int = 25) -> list[SweepPoint]:
    """h, Q̇ y Δp contra la velocidad de llegada, de V/4 a 3·V (escala logarítmica)."""
    points: list[SweepPoint] = []
    lo, hi = math.log(inputs.V_m_s / 4.0), math.log(3.0 * inputs.V_m_s)
    for j in range(n):
        V = math.exp(lo + (hi - lo) * j / (n - 1))
        try:
            r = solve_tube_bank(replace(inputs, V_m_s=V))
        except ValueError:
            continue
        points.append(SweepPoint(V, r.h_W_per_m2K, r.Q_W, r.dp_Pa))
    return points


def tube_bank_notes(result: TubeBankResult) -> list[str]:
    """Interpretación física del resultado (markdown)."""
    r, i = result, result.inputs
    notes: list[str] = []
    if r.diagonal:
        notes.append(
            f"En este tresbolillo la diagonal S_D = {_num(r.S_D_m * 1e3 if r.S_D_m else 0.0)} "
            f"mm es menor que (S_T + D)/2: el paso más angosto es la diagonal y V_máx sale de "
            "ahí."
        )
    other: ConstantSource = "incropera" if i.source == "cengel" else "cengel"
    Nu_other = row_correction(i.N_L, i.arrangement, other) * nu_zukauskas(
        r.Re, r.Pr, r.Pr_s, i.arrangement, i.S_T_m / i.S_L_m, other
    )
    diff = 100.0 * (Nu_other / r.Nu - 1.0)
    if abs(diff) >= 0.5:
        notes.append(
            f"Con las constantes de {CONSTANT_SOURCES[other]} el Nu daría {_num(Nu_other)} "
            f"({'+' if diff > 0 else '−'}{_num(abs(diff), 2)} %): son dos ajustes de los "
            "mismos datos de Zukauskas."
        )
    if r.fit.single_cylinder:
        notes.append(
            "Entre Re = 10² y 10³ Incropera aproxima cada tubo como un cilindro aislado "
            "(Zukauskas: Nu = 0,51·Re^0,5·Pr^n·(Pr/Pr_s)^¼, con la velocidad máxima)."
        )
    rise = abs(r.T_out_K - i.T_in_K)
    notes.append(
        f"El fluido se {'calienta' if r.T_out_K > i.T_in_K else 'enfría'} "
        f"{_num(rise)} K: las propiedades van a la media, {_degC(r.T_mean_K)}, y Pr_s a la "
        "temperatura de los tubos."
        if i.properties is None
        else "Las propiedades son las dadas (las del enunciado): no se iteran."
    )
    return notes


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TubeBankExample:
    description: str
    inputs: TubeBankInputs


_C = 273.15
_ATM = 101_325.0

TUBE_BANK_EXAMPLES: dict[str, TubeBankExample] = {
    "Calentador de aire en tresbolillo (Incropera, ejemplo 7.7)": TubeBankExample(
        "Aire a 15 °C y 6 m/s cruza un banco en tresbolillo de 7 filas de 8 tubos de 16,4 mm "
        "a 70 °C (S_T = 31,3 mm, S_L = 34,3 mm). El libro evalúa las propiedades a la entrada "
        "y redondea C a 0,34: h = 135,6 W/(m²·K), q' = 19,4 kW/m y Δp = 246 Pa (lee f = 0,35 y "
        "χ = 1,04 de los gráficos).",
        TubeBankInputs(
            fluid="Air",
            p_Pa=_ATM,
            V_m_s=6.0,
            T_in_K=15.0 + _C,
            T_s_K=70.0 + _C,
            D_m=0.0164,
            S_T_m=0.0313,
            S_L_m=0.0343,
            N_L=7,
            N_T=8,
            L_m=1.0,
            arrangement="staggered",
            source="incropera",
        ),
    ),
    "Aire precalentado con agua geotérmica (Çengel y Ghajar, ejemplo 7-7)": TubeBankExample(
        "Aire a 20 °C y 4,5 m/s cruza un banco alineado de 6 filas de 10 tubos de 1,5 cm a "
        "120 °C, con S_T = S_L = 5 cm. El libro supone las propiedades a 60 °C: Q̇ = 24,9 kW "
        "por metro de tubo y Δp = 21 Pa (lee f = 0,16, aunque S_L/D = 3,33 cae fuera del "
        "gráfico). Con la temperatura media real, unos 25 °C, h sale ~5 % mayor.",
        TubeBankInputs(
            fluid="Air",
            p_Pa=_ATM,
            V_m_s=4.5,
            T_in_K=20.0 + _C,
            T_s_K=120.0 + _C,
            D_m=0.015,
            S_T_m=0.05,
            S_L_m=0.05,
            N_L=6,
            N_T=10,
            L_m=1.0,
            arrangement="aligned",
            source="cengel",
        ),
    ),
    "Agua sobre un banco de tubos calientes": TubeBankExample(
        "Agua a 15 °C y 0,6 m/s cruza un banco en tresbolillo de 10 filas de 12 tubos de 25 mm "
        "a 60 °C (S_T = 50 mm, S_L = 45 mm), de 0,5 m de largo. Con Pr ≈ 7 el (Pr/Pr_s)^¼ pesa: "
        "el agua junto a la pared es menos viscosa.",
        TubeBankInputs(
            fluid="Water",
            p_Pa=2.0e5,
            V_m_s=0.6,
            T_in_K=15.0 + _C,
            T_s_K=60.0 + _C,
            D_m=0.025,
            S_T_m=0.05,
            S_L_m=0.045,
            N_L=10,
            N_T=12,
            L_m=0.5,
            arrangement="staggered",
            source="cengel",
        ),
    ),
}


def _q(value: float, kind: Any, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value, kind, system), "unidad": unit_label(kind, system)}


def tube_bank_to_dict(r: TubeBankResult, system: UnitSystem) -> dict[str, Any]:
    """El banco resuelto, serializable a JSON y CSV."""
    i = r.inputs
    return {
        "fluido": i.fluid,
        "presion": _q(i.p_Pa, "pressure", system),
        "arreglo": ARRANGEMENTS[i.arrangement],
        "constantes": CONSTANT_SOURCES[i.source],
        "V": _q(i.V_m_s, "speed", system),
        "T_entrada": _q(i.T_in_K, "temperature", system),
        "T_tubos": _q(i.T_s_K, "temperature", system),
        "D": _q(i.D_m, "small_length", system),
        "S_T": _q(i.S_T_m, "small_length", system),
        "S_L": _q(i.S_L_m, "small_length", system),
        "N_L": i.N_L,
        "N_T": i.N_T,
        "L": _q(i.L_m, "length", system),
        "propiedades": "dadas" if i.properties is not None else "CoolProp a la temperatura media",
        "rho": _q(r.rho_kg_per_m3, "density", system),
        "mu": _q(r.mu_Pa_s, "dynamic_viscosity", system),
        "k": _q(r.k_W_per_mK, "thermal_conductivity", system),
        "cp": _q(r.cp_J_per_kgK, "specific_heat", system),
        "Pr": r.Pr,
        "Pr_s": r.Pr_s,
        "V_max": _q(r.V_max_m_s, "speed", system),
        "por_la_diagonal": r.diagonal,
        "Re": r.Re,
        "C": r.C,
        "m": r.fit.m,
        "n": r.fit.n,
        "correccion_filas": r.F,
        "Nu": r.Nu,
        "h": _q(r.h_W_per_m2K, "heat_transfer_coefficient", system),
        "area": _q(r.area_m2, "area", system),
        "m_dot": _q(r.m_dot_kg_s, "mass_flow", system),
        "T_salida": _q(r.T_out_K, "temperature", system),
        "Delta_T_ml": _q(r.dT_lm_K, "temperature_difference", system),
        "Q": _q(r.Q_W, "heat_rate", system),
        "Q_por_metro": _q(r.Q_per_length_W_m, "linear_heat_rate", system),
        "f": r.f,
        "chi": r.chi,
        "Delta_p": _q(r.dp_Pa, "pressure_drop", system),
        "avisos": list(r.warnings),
        "notas": tube_bank_notes(r),
    }
