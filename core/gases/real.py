"""Gases reales: el factor de compresibilidad con seis modelos (vademecum §7 y §8).

**Estados correspondientes** (§7): p_R = p/p_cr, T_R = T/T_cr, v_R = v/v_cr y el
volumen pseudorreducido de Nelson y Obert (1954), v′_R = v·p_cr/(R·T_cr) = Z_cr·v_R
(§7.4); Z = p·v/(R·T) = Z(p_R, T_R) (§7.3).

Los modelos, contra la ecuación de estado de referencia de cada fluido (CoolProp,
Bell et al., 2014):

- **gas ideal**: Z = 1;
- **carta generalizada**: el fluido simple de Lee y Kesler (1975), Z⁰(p_R, T_R),
  la carta de dos parámetros de Nelson y Obert que trae Çengel (§3-7);
- **Lee–Kesler con ω**: Z = Z⁰ + (ω/ω_r)·(Z^r − Z⁰), con el fluido de referencia
  (n-octano, ω_r = 0,3978) y el factor acéntrico ω de cada fluido;
- **Van der Waals** (§8), con a y b del punto crítico;
- **Peng y Robinson (1976)**, la cúbica que se usa hoy: con ω acierta al 1 % donde
  Van der Waals se equivoca 5–7 %.

Con (p, T) el modelo da v; con (T, v), p; con (p, v), T. La fase (líquido o vapor)
la decide el fluido real: con varias raíces, el modelo toma la del vapor (la mayor)
o la del líquido (la menor). Bajo la temperatura crítica, una raíz de vapor tiene
Z > Z_cr y una de líquido, v < v_cr (los dos lados de la campana): si el modelo
solo tiene la raíz de la otra fase, no tiene solución en esa fase.

El punto crítico y ω son los de CoolProp (su ecuación de estado), en SI.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Any, Literal

import CoolProp
import numpy as np
from scipy.optimize import brentq

from core.fluids import fluid_limits
from core.ideal_gas import R_U

__all__ = [
    "LK_OMEGA_REF",
    "MODELS",
    "MODEL_PHRASES",
    "REAL_EXAMPLES",
    "REAL_FLUIDS",
    "CompressibilityInputs",
    "CompressibilityResult",
    "LeeKesler",
    "ModelKind",
    "ModelResult",
    "RealExample",
    "RealFluid",
    "StatePairKind",
    "VADEMECUM_FLUIDS",
    "CUBIC_ZC",
    "compressibility",
    "compressibility_to_dict",
    "cubic_ab",
    "cubic_coefficients",
    "cubic_constants",
    "cubic_p",
    "cubic_z_roots",
    "generalized_chart",
    "lee_kesler",
    "lee_kesler_psat_reduced",
    "lk_coefficients",
    "lk_z_of_vr",
    "pr_kappa",
    "real_fluid",
    "z_isotherm",
]

ModelKind = Literal["ideal", "lk0", "lk", "vdw", "pr", "coolprop"]
MODELS: dict[ModelKind, str] = {
    "ideal": "Gas ideal",
    "lk0": "Carta generalizada (Z⁰)",
    "lk": "Lee–Kesler con ω",
    "vdw": "Van der Waals",
    "pr": "Peng–Robinson",
    "coolprop": "CoolProp (referencia)",
}
#: Cómo se nombra cada modelo dentro de una oración.
MODEL_PHRASES: dict[ModelKind, str] = {
    "ideal": "el gas ideal",
    "lk0": "la carta generalizada (Z⁰)",
    "lk": "Lee–Kesler con ω",
    "vdw": "Van der Waals",
    "pr": "Peng–Robinson",
    "coolprop": "CoolProp",
}
StatePairKind = Literal["pT", "Tv", "pv"]
CubicKind = Literal["vdw", "pr"]
Phase = Literal["vapor", "liquid"]
LKFluid = Literal["simple", "reference"]


class _NoSolution(ValueError):
    """Un modelo sin solución, con el porqué para el alumno."""


def _num(x: float, fmt: str = ".3g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


def _pct(x: float, decimals: int = 1) -> str:
    return f"{100.0 * x:.{decimals}f} %".replace(".", ",")


# ---------------------------------------------------------------------
# Fluidos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RealFluid:
    """Un fluido con su punto crítico y su factor acéntrico (de CoolProp)."""

    key: str  # nombre de CoolProp
    name: str
    formula: str

    @property
    def _c(self) -> dict[str, float]:
        return _constants(self.key)

    @property
    def M(self) -> float:
        """Masa molar (kg/mol)."""
        return self._c["M"]

    @property
    def R(self) -> float:
        """R = R_u/M (J/(kg·K))."""
        return self._c["R"]

    @property
    def T_cr(self) -> float:
        return self._c["T_cr"]

    @property
    def p_cr(self) -> float:
        return self._c["p_cr"]

    @property
    def v_cr(self) -> float:
        """Volumen específico crítico (m³/kg)."""
        return self._c["v_cr"]

    @property
    def Z_cr(self) -> float:
        """Z_cr = p_cr·v_cr/(R·T_cr) (vademecum §7.4)."""
        return self.p_cr * self.v_cr / (self.R * self.T_cr)

    @property
    def omega(self) -> float:
        """Factor acéntrico de Pitzer: ω = −log₁₀(p_sat/p_cr)|_{T_R = 0,7} − 1."""
        return self._c["omega"]

    @property
    def T_min(self) -> float:
        return self._c["T_min"]

    @property
    def T_max(self) -> float:
        return self._c["T_max"]

    @property
    def p_max(self) -> float:
        return self._c["p_max"]

    @property
    def label(self) -> str:
        """El nombre con la fórmula, para los selectores."""
        return self.name if self.formula == "—" else f"{self.name} ({self.formula})"

    @property
    def noun(self) -> str:
        """El nombre dentro de una oración: «del agua», «el R-134a», «del aire»."""
        short = self.name.split(" (")[0]
        return short if short.startswith("R-") else short.lower()


@cache
def _constants(key: str) -> dict[str, float]:
    lim = fluid_limits(key)
    state = CoolProp.AbstractState("HEOS", key)
    return {
        "M": lim.M_kg_per_mol,
        "R": R_U / lim.M_kg_per_mol,
        "T_cr": lim.T_crit_K,
        "p_cr": lim.P_crit_Pa,
        "v_cr": lim.v_crit_m3_per_kg,
        "omega": float(state.acentric_factor()),
        "T_min": lim.T_min_K,
        "T_max": lim.T_max_K,
        "p_max": lim.P_max_Pa,
    }


#: Los nueve de la tabla de propiedades críticas del vademecum §7.5 y los del proyecto.
REAL_FLUIDS: dict[str, RealFluid] = {
    f.key: f
    for f in (
        RealFluid("Air", "Aire (pseudo puro)", "—"),
        RealFluid("Argon", "Argón", "Ar"),
        RealFluid("CarbonDioxide", "Dióxido de carbono", "CO₂"),
        RealFluid("Helium", "Helio", "He"),
        RealFluid("Hydrogen", "Hidrógeno", "H₂"),
        RealFluid("Methane", "Metano", "CH₄"),
        RealFluid("Nitrogen", "Nitrógeno", "N₂"),
        RealFluid("Oxygen", "Oxígeno", "O₂"),
        RealFluid("Water", "Agua", "H₂O"),
        RealFluid("R134a", "R-134a", "CH₂FCF₃"),
        RealFluid("Ammonia", "Amoníaco", "NH₃"),
        RealFluid("n-Propane", "Propano", "C₃H₈"),
        RealFluid("IsoButane", "Isobutano", "C₄H₁₀"),
        RealFluid("R32", "R-32", "CH₂F₂"),
    )
}

#: Los nueve de la tabla del vademecum §7.5.
VADEMECUM_FLUIDS: tuple[str, ...] = (
    "Air",
    "Argon",
    "CarbonDioxide",
    "Helium",
    "Hydrogen",
    "Methane",
    "Nitrogen",
    "Oxygen",
    "Water",
)


def real_fluid(key: str) -> RealFluid:
    if key not in REAL_FLUIDS:
        raise ValueError(f"El fluido {key!r} no está en la lista: {', '.join(REAL_FLUIDS)}.")
    return REAL_FLUIDS[key]


def _pick(
    roots: Sequence[float], phase: Phase, Tr: float, pr: float, Zc: float, what: str
) -> float:
    """La raíz Z de la fase pedida.

    Bajo la temperatura crítica del modelo, una raíz de vapor tiene Z > Z_cr y una
    de líquido, v < v_cr (en Z: Z < Z_cr·p_R/T_R); si la única raíz es de la otra
    fase, el modelo no tiene solución en esa fase.
    """
    if not roots:
        raise _NoSolution(f"{what} no tiene solución a esa presión y temperatura.")
    z = roots[-1] if phase == "vapor" else roots[0]
    if Tr < 1.0:
        if phase == "vapor" and z <= Zc:
            raise _NoSolution(
                f"Para {what} ese estado ya es líquido: a esa presión y temperatura no tiene "
                "raíz de vapor."
            )
        if phase == "liquid" and z >= Zc * pr / Tr:
            raise _NoSolution(
                f"Para {what} ese estado todavía es vapor: a esa presión y temperatura no "
                "tiene raíz de líquido."
            )
    return z


# ---------------------------------------------------------------------
# Lee y Kesler (1975)
# ---------------------------------------------------------------------

#: Lee y Kesler (1975), tabla 1: el fluido simple (Ar, Kr, CH₄) y el de referencia (n-octano).
_LK: dict[str, dict[str, float]] = {
    "simple": {
        "b1": 0.1181193,
        "b2": 0.265728,
        "b3": 0.154790,
        "b4": 0.030323,
        "c1": 0.0236744,
        "c2": 0.0186984,
        "c3": 0.0,
        "c4": 0.042724,
        "d1": 0.155488e-4,
        "d2": 0.623689e-4,
        "beta": 0.65392,
        "gamma": 0.060167,
    },
    "reference": {
        "b1": 0.2026579,
        "b2": 0.331511,
        "b3": 0.027655,
        "b4": 0.203488,
        "c1": 0.0313385,
        "c2": 0.0503618,
        "c3": 0.016901,
        "c4": 0.041577,
        "d1": 0.48736e-4,
        "d2": 0.0740336e-4,
        "beta": 1.226,
        "gamma": 0.03754,
    },
}
#: El factor acéntrico del fluido de referencia (n-octano) de Lee y Kesler.
LK_OMEGA_REF: float = 0.3978
#: Z_cr de cada fluido de Lee y Kesler (la inflexión de la isoterma T_R = 1, calculada;
#: en el punto crítico v_R = Z porque T_R = p_R = 1).
_LK_ZC: dict[str, float] = {"simple": 0.2905, "reference": 0.2560}
_VR_GRID = np.geomspace(0.015, 1.0e4, 3000)


def lk_coefficients(Tr: float, fluid: LKFluid) -> tuple[float, float, float]:
    """B, C y D de Lee y Kesler (1975) a T_R:

    B = b₁ − b₂/T_R − b₃/T_R² − b₄/T_R³, C = c₁ − c₂/T_R + c₃/T_R³ y D = d₁ + d₂/T_R.
    """
    c = _LK[fluid]
    B = c["b1"] - c["b2"] / Tr - c["b3"] / Tr**2 - c["b4"] / Tr**3
    C = c["c1"] - c["c2"] / Tr + c["c3"] / Tr**3
    D = c["d1"] + c["d2"] / Tr
    return B, C, D


def lk_z_of_vr(Tr: float, vr: Any, fluid: LKFluid = "simple") -> Any:
    """Z de la ecuación de Lee y Kesler (1975, ec. 2) a (T_R, v_R):

    Z = 1 + B/v_R + C/v_R² + D/v_R⁵ + c₄/(T_R³·v_R²)·(β + γ/v_R²)·exp(−γ/v_R²),

    con v_R = p_cr·v/(R·T_cr) el volumen «ideal reducido» de cada fluido (para el
    simple, el v′_R de Nelson y Obert).
    """
    c = _LK[fluid]
    B, C, D = lk_coefficients(Tr, fluid)
    vr2 = vr * vr
    return (
        1.0
        + B / vr
        + C / vr2
        + D / vr2**2 / vr
        + c["c4"] / (Tr**3 * vr2) * (c["beta"] + c["gamma"] / vr2) * np.exp(-c["gamma"] / vr2)
    )


def lee_kesler_psat_reduced(Tr: float, omega: float = 0.0) -> float:
    """p_sat/p_cr de Lee y Kesler (1975, ec. 4): ln p_R = f⁰(T_R) + ω·f¹(T_R)."""
    f0 = 5.92714 - 6.09648 / Tr - 1.28862 * math.log(Tr) + 0.169347 * Tr**6
    f1 = 15.2518 - 15.6875 / Tr - 13.4721 * math.log(Tr) + 0.43577 * Tr**6
    return math.exp(f0 + omega * f1)


def _lk_roots(Tr: float, pr: float, fluid: LKFluid) -> list[float]:
    """Todas las raíces v_R de p_R·v_R/T_R = Z(T_R, v_R), de menor a mayor."""
    vals = pr * _VR_GRID / Tr - lk_z_of_vr(Tr, _VR_GRID, fluid)
    roots: list[float] = []
    sign = np.sign(vals)
    for k in np.nonzero(sign[:-1] * sign[1:] < 0)[0]:
        a, b = float(_VR_GRID[k]), float(_VR_GRID[k + 1])
        roots.append(
            brentq(lambda v: pr * v / Tr - float(lk_z_of_vr(Tr, v, fluid)), a, b, xtol=1e-14)
        )
    return roots


def _lk_z(Tr: float, pr: float, fluid: LKFluid, phase: Phase) -> tuple[float, float]:
    """(v_R, Z) de un fluido de Lee y Kesler en la fase pedida."""
    what = "el fluido simple" if fluid == "simple" else "el fluido de referencia"
    z = _pick([pr * v / Tr for v in _lk_roots(Tr, pr, fluid)], phase, Tr, pr, _LK_ZC[fluid], what)
    return z * Tr / pr, z


@dataclass(frozen=True)
class LeeKesler:
    """El resultado de Lee y Kesler en (T_R, p_R)."""

    Tr: float
    pr: float
    omega: float
    vr0: float  # el v_R del fluido simple (= v′_R)
    Z0: float
    vr_ref: float
    Z_ref: float
    Z: float


def lee_kesler(Tr: float, pr: float, omega: float, phase: Phase = "vapor") -> LeeKesler:
    """Z⁰, Z^r y Z = Z⁰ + (ω/ω_r)·(Z^r − Z⁰) de Lee y Kesler (1975) a (T_R, p_R).

    Los dos fluidos se evalúan en la misma fase: la del vapor (la raíz mayor) o la del
    líquido (la menor).

    Raises
    ------
    ValueError
        Si alguno de los dos fluidos no tiene raíz en esa fase a ese T_R y p_R.
    """
    vr0, z0 = _lk_z(Tr, pr, "simple", phase)
    vrr, zr = _lk_z(Tr, pr, "reference", phase)
    return LeeKesler(Tr, pr, omega, vr0, z0, vrr, zr, z0 + omega / LK_OMEGA_REF * (zr - z0))


# ---------------------------------------------------------------------
# Van der Waals y Peng–Robinson
# ---------------------------------------------------------------------

#: Z_cr de cada cúbica: 3/8 para Van der Waals (vademecum §8.3) y 0,3074 para Peng–Robinson.
CUBIC_ZC: dict[CubicKind, float] = {"vdw": 0.375, "pr": 0.3074}


def pr_kappa(omega: float) -> float:
    """κ de Peng y Robinson (1976): 0,37464 + 1,54226·ω − 0,26992·ω²."""
    return 0.37464 + 1.54226 * omega - 0.26992 * omega**2


def cubic_constants(fl: RealFluid, model: CubicKind, T: float) -> tuple[float, float]:
    """(a·α(T), b) por kg: Van der Waals (vademecum §8.2) o Peng y Robinson (1976).

    - Van der Waals: a = 27·R²·T_cr²/(64·p_cr) y b = R·T_cr/(8·p_cr);
    - Peng–Robinson: a = 0,45724·R²·T_cr²/p_cr, b = 0,07780·R·T_cr/p_cr y
      α = [1 + κ·(1 − √T_R)]² con κ = 0,37464 + 1,54226·ω − 0,26992·ω².
    """
    R, Tc, pc = fl.R, fl.T_cr, fl.p_cr
    if model == "vdw":
        return 27.0 * R**2 * Tc**2 / (64.0 * pc), R * Tc / (8.0 * pc)
    kappa = pr_kappa(fl.omega)
    alpha = (1.0 + kappa * (1.0 - math.sqrt(T / Tc))) ** 2
    return 0.45724 * R**2 * Tc**2 / pc * alpha, 0.07780 * R * Tc / pc


def cubic_p(fl: RealFluid, model: CubicKind, T: float, v: float) -> float:
    """p(T, v): R·T/(v − b) − a/v² o R·T/(v − b) − a·α/(v² + 2·b·v − b²)."""
    a, b = cubic_constants(fl, model, T)
    if model == "vdw":
        return fl.R * T / (v - b) - a / v**2
    return fl.R * T / (v - b) - a / (v**2 + 2.0 * b * v - b**2)


def cubic_ab(fl: RealFluid, model: CubicKind, T: float, p: float) -> tuple[float, float]:
    """Los adimensionales A = a·p/(R·T)² y B = b·p/(R·T) de la cúbica en Z."""
    a, b = cubic_constants(fl, model, T)
    RT = fl.R * T
    return a * p / RT**2, b * p / RT


def cubic_coefficients(model: CubicKind, A: float, B: float) -> tuple[float, float, float]:
    """Z³ + c₂·Z² + c₁·Z + c₀ = 0.

    - Van der Waals: Z³ − (1 + B)·Z² + A·Z − A·B = 0;
    - Peng–Robinson: Z³ − (1 − B)·Z² + (A − 3B² − 2B)·Z − (A·B − B² − B³) = 0.
    """
    if model == "vdw":
        return -(1.0 + B), A, -A * B
    return -(1.0 - B), A - 3.0 * B**2 - 2.0 * B, -(A * B - B**2 - B**3)


def cubic_z_roots(model: CubicKind, A: float, B: float) -> tuple[float, ...]:
    """Las raíces reales de la cúbica en Z con v > b (Z > B), de menor a mayor."""
    c2, c1, c0 = cubic_coefficients(model, A, B)
    roots = np.roots([1.0, c2, c1, c0])
    real = sorted(
        float(r.real) for r in roots if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)) and r.real > B
    )
    # dos raíces casi iguales (el borde del lazo) cuentan una vez
    out: list[float] = []
    for r in real:
        if not out or abs(r - out[-1]) > 1e-9:
            out.append(r)
    return tuple(out)


def _cubic_z(
    fl: RealFluid, model: CubicKind, p: float, T: float, phase: Phase
) -> tuple[float, tuple[float, ...]]:
    """La Z de la cúbica en la fase pedida, y todas sus raíces."""
    A, B = cubic_ab(fl, model, T, p)
    roots = cubic_z_roots(model, A, B)
    z = _pick(roots, phase, T / fl.T_cr, p / fl.p_cr, CUBIC_ZC[model], MODEL_PHRASES[model])
    return z, roots


# ---------------------------------------------------------------------
# El factor de compresibilidad
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CompressibilityInputs:
    """El fluido y dos de p, T y v (SI): ``pT`` → v; ``Tv`` → p; ``pv`` → T."""

    fluid_key: str
    pair: StatePairKind
    a: float  # p en pT y pv; T en Tv
    b: float  # T en pT; v en Tv y pv


@dataclass(frozen=True)
class ModelResult:
    """Lo que da un modelo (None si no tiene solución, con el porqué en ``note``)."""

    model: ModelKind
    p_Pa: float | None
    T_K: float | None
    v_m3_per_kg: float | None
    Z: float | None
    error: float | None  # de la incógnita contra CoolProp (relativo)
    roots: tuple[float, ...] = ()  # las raíces Z de una cúbica con (p, T)
    note: str = ""


@dataclass(frozen=True)
class CompressibilityResult:
    inputs: CompressibilityInputs
    fluid: RealFluid
    unknown: Literal["v", "p", "T"]
    p_Pa: float  # el estado real (CoolProp)
    T_K: float
    v_m3_per_kg: float
    Z: float
    phase: Phase
    p_R: float
    T_R: float
    v_R: float
    v_R_pseudo: float  # v′_R de Nelson y Obert
    models: tuple[ModelResult, ...]
    notes: tuple[str, ...]

    def model(self, kind: ModelKind) -> ModelResult:
        return next(m for m in self.models if m.model == kind)

    @property
    def best(self) -> ModelResult | None:
        """El modelo que menos se equivoca (sin contar CoolProp)."""
        ok = [m for m in self.models if m.model != "coolprop" and m.error is not None]
        return min(ok, key=lambda m: abs(m.error or 0.0)) if ok else None


def _phase_of(state: CoolProp.AbstractState) -> Phase:
    if state.phase() in (CoolProp.iphase_liquid, CoolProp.iphase_supercritical_liquid):
        return "liquid"
    return "vapor"


def _real_state(fl: RealFluid, inputs: CompressibilityInputs) -> CoolProp.AbstractState:
    """El estado de CoolProp, con los errores explicados al alumno."""
    state = CoolProp.AbstractState("HEOS", fl.key)
    p = T = v = None
    if inputs.pair == "pT":
        p, T = inputs.a, inputs.b
    elif inputs.pair == "Tv":
        T, v = inputs.a, inputs.b
    else:
        p, v = inputs.a, inputs.b
    for name, value in (
        ("La presión (absoluta) tiene que ser positiva", p),
        ("La temperatura (absoluta) tiene que ser positiva", T),
        ("El volumen específico tiene que ser positivo", v),
    ):
        if value is not None and not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name}.")
    name = fl.noun
    if T is not None and not (fl.T_min <= T <= fl.T_max):
        raise ValueError(
            f"La temperatura ({_num(T, '.4g')} K) está fuera del rango de la ecuación de estado "
            f"del {name}: de {_num(fl.T_min, '.4g')} K a {_num(fl.T_max, '.4g')} K."
        )
    if p is not None and p > fl.p_max:
        raise ValueError(
            f"La presión ({_num(p / 1e6, '.4g')} MPa) supera la máxima de la ecuación de estado "
            f"del {name} ({_num(fl.p_max / 1e6, '.4g')} MPa)."
        )
    try:
        if inputs.pair == "pT":
            state.update(CoolProp.PT_INPUTS, p, T)
        elif inputs.pair == "Tv":
            state.update(CoolProp.DmassT_INPUTS, 1.0 / v, T)  # type: ignore[operator]
        else:
            state.update(CoolProp.DmassP_INPUTS, 1.0 / v, p)  # type: ignore[operator]
    except ValueError as exc:
        raise ValueError(
            f"CoolProp no encuentra ese estado del {name}. Revisá que los datos estén dentro "
            f"del rango de la ecuación de estado ({exc})."
        ) from exc
    if state.phase() == CoolProp.iphase_twophase:
        raise ValueError(
            "Ese estado está dentro de la campana (mezcla de líquido y vapor, título "
            f"x = {_num(state.Q(), '.3g')}): el factor Z de una mezcla no sale de estas "
            "ecuaciones. Mirá la campana en el modo de Van der Waals o calculá la mezcla en "
            "/Propiedades."
        )
    if not (fl.T_min <= state.T() <= fl.T_max) or state.p() > fl.p_max:
        raise ValueError(
            f"Ese estado queda fuera del rango de la ecuación de estado del {name} "
            f"(T = {_num(state.T(), '.4g')} K, p = {_num(state.p() / 1e6, '.4g')} MPa)."
        )
    return state


def _solve_1d(f: Callable[[float], float], lo: float, hi: float, n: int = 60) -> float | None:
    """La raíz de f en [lo, hi] (barrido geométrico y Brent) o None.

    Un cambio de signo que no es raíz (un salto de la función, cuando la raíz del modelo
    cambia de rama) se descarta: el residuo en la raíz tiene que ser chico.
    """
    prev: tuple[float, float] | None = None
    for x in np.geomspace(lo, hi, n):
        x = float(x)
        try:
            fx = f(x)
        except (ValueError, ZeroDivisionError, OverflowError):
            prev = None
            continue
        if not math.isfinite(fx):
            prev = None
            continue
        if fx == 0.0:
            return x
        if prev is not None and prev[1] * fx < 0.0:
            try:
                root = float(brentq(f, prev[0], x, xtol=1e-12 * x, rtol=1e-13))
                if abs(f(root)) < 1e-7:
                    return root
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        prev = (x, fx)
    return None


def _lk_v(fl: RealFluid, p: float, T: float, phase: Phase, with_omega: bool) -> float:
    """v de Lee y Kesler: con ω, o solo con el fluido simple (la carta, Z⁰)."""
    Tr, pr = T / fl.T_cr, p / fl.p_cr
    if with_omega:
        Z = lee_kesler(Tr, pr, fl.omega, phase).Z
    else:
        Z = _lk_z(Tr, pr, "simple", phase)[1]
    return Z * fl.R * T / p


def _covolume_check(fl: RealFluid, model: CubicKind, v: float) -> None:
    _, b = cubic_constants(fl, model, fl.T_cr)
    if v <= b:
        raise _NoSolution(
            f"El volumen dado es menor que el covolumen b de {MODEL_PHRASES[model]} (el "
            "volumen propio de las moléculas): la ecuación no tiene solución. Pasa con los "
            "líquidos."
        )


def _model_state(
    fl: RealFluid,
    kind: ModelKind,
    inputs: CompressibilityInputs,
    ref: dict[str, float],
    phase: Phase,
) -> tuple[float, float, float, tuple[float, ...]]:
    """(p, T, v, raíces) de un modelo con el par de datos del alumno."""
    R = fl.R
    roots: tuple[float, ...] = ()
    if inputs.pair == "pT":
        p, T = inputs.a, inputs.b
        if kind == "ideal":
            v = R * T / p
        elif kind in ("lk0", "lk"):
            v = _lk_v(fl, p, T, phase, kind == "lk")
        else:
            z, roots = _cubic_z(fl, kind, p, T, phase)  # type: ignore[arg-type]
            v = z * R * T / p
        return p, T, v, roots
    if inputs.pair == "Tv":
        T, v = inputs.a, inputs.b
        if kind == "ideal":
            p = R * T / v
        elif kind == "lk0":
            # con (T, v) el fluido simple da Z⁰ directo de v′_R (Çengel §3-7)
            vr = v * fl.p_cr / (R * fl.T_cr)
            p = float(lk_z_of_vr(T / fl.T_cr, vr, "simple")) * R * T / v
            if p <= 0.0:
                raise _NoSolution(
                    "Con ese volumen la ecuación del fluido simple da una presión negativa."
                )
        elif kind == "lk":
            found = _solve_1d(
                lambda pp: _lk_v(fl, pp, T, phase, True) / v - 1.0,
                max(ref["p"] * 0.05, 1.0),
                min(ref["p"] * 20.0, 50.0 * fl.p_cr),
            )
            if found is None:
                raise _NoSolution(
                    "No hay una presión que le dé ese volumen con este modelo (se buscó de "
                    "1/20 a 20 veces la presión real)."
                )
            p = found
        else:
            _covolume_check(fl, kind, v)  # type: ignore[arg-type]
            p = cubic_p(fl, kind, T, v)  # type: ignore[arg-type]
            if p <= 0.0:
                raise _NoSolution(
                    "Da una presión negativa: con ese volumen el término de atracción le gana "
                    "a R·T/(v − b). Pasa con los líquidos a temperatura baja."
                )
        return p, T, v, roots
    p, v = inputs.a, inputs.b  # pv → T
    if kind == "ideal":
        T = p * v / R
    elif kind == "vdw":
        _covolume_check(fl, "vdw", v)
        a, b = cubic_constants(fl, "vdw", fl.T_cr)
        T = (p + a / v**2) * (v - b) / R
    else:
        vr = v * fl.p_cr / (R * fl.T_cr)

        def resid(TT: float) -> float:
            if kind == "pr":
                return cubic_p(fl, "pr", TT, v) / p - 1.0
            if kind == "lk0":  # p_R = Z⁰(T_R, v′_R)·T_R/v′_R, directo en v′_R
                Tr = TT / fl.T_cr
                return float(lk_z_of_vr(Tr, vr, "simple")) * Tr / vr / (p / fl.p_cr) - 1.0
            return _lk_v(fl, p, TT, phase, True) / v - 1.0

        if kind == "pr":
            _covolume_check(fl, "pr", v)
        found = _solve_1d(resid, ref["T"] * 0.3, ref["T"] * 3.0)
        if found is None:
            raise _NoSolution(
                "No hay una temperatura que le dé ese volumen a esa presión con este modelo "
                "(se buscó de 0,3 a 3 veces la temperatura real)."
            )
        T = found
    return p, T, v, roots


def _model_result(
    fl: RealFluid,
    kind: ModelKind,
    inputs: CompressibilityInputs,
    ref: dict[str, float],
    phase: Phase,
) -> ModelResult:
    try:
        p, T, v, roots = _model_state(fl, kind, inputs, ref, phase)
    except _NoSolution as exc:
        return ModelResult(kind, None, None, None, None, None, note=str(exc))
    except (ValueError, ZeroDivisionError, OverflowError):
        return ModelResult(
            kind, None, None, None, None, None, note="Con este modelo ese estado no tiene solución."
        )
    note = ""
    if len(roots) == 3:
        note = (
            "La cúbica tiene tres raíces: la mayor es la del vapor, la menor la del líquido y "
            "la del medio no tiene sentido físico (ahí la presión subiría con el volumen)."
        )
    unknown = {"pT": v, "Tv": p, "pv": T}[inputs.pair]
    error = unknown / {"pT": ref["v"], "Tv": ref["p"], "pv": ref["T"]}[inputs.pair] - 1.0
    return ModelResult(kind, p, T, v, p * v / (fl.R * T), error, roots, note)


def compressibility(inputs: CompressibilityInputs) -> CompressibilityResult:
    """Z y la incógnita (v, p o T) con los seis modelos (vademecum §7 y §8).

    Raises
    ------
    ValueError
        Con un dato que no es positivo, fuera del rango de la ecuación de estado o
        un estado dentro de la campana.
    """
    fl = real_fluid(inputs.fluid_key)
    state = _real_state(fl, inputs)
    p, T, v = float(state.p()), float(state.T()), 1.0 / float(state.rhomass())
    phase = _phase_of(state)
    ref = {"p": p, "T": T, "v": v}
    models = [_model_result(fl, k, inputs, ref, phase) for k in MODELS if k != "coolprop"]
    models.append(ModelResult("coolprop", p, T, v, p * v / (fl.R * T), None))
    unknown: Literal["v", "p", "T"] = {"pT": "v", "Tv": "p", "pv": "T"}[inputs.pair]  # type: ignore[assignment]
    result = CompressibilityResult(
        inputs=inputs,
        fluid=fl,
        unknown=unknown,
        p_Pa=p,
        T_K=T,
        v_m3_per_kg=v,
        Z=p * v / (fl.R * T),
        phase=phase,
        p_R=p / fl.p_cr,
        T_R=T / fl.T_cr,
        v_R=v / fl.v_cr,
        v_R_pseudo=v * fl.p_cr / (fl.R * fl.T_cr),
        models=tuple(models),
        notes=(),
    )
    return replace(result, notes=_notes(result))


def _notes(r: CompressibilityResult) -> tuple[str, ...]:
    notes: list[str] = []
    fl = r.fluid
    ideal = r.model("ideal")
    Z = _num(r.Z, ".4g")
    if r.phase == "liquid":
        notes.append(
            f"Es líquido (Z = {Z}): el gas ideal no sirve. Van der Waals le erra mucho a la "
            "densidad del líquido (su Z_cr = 3/8 es muy grande); Lee–Kesler y Peng–Robinson "
            "andan mejor, pero para un líquido se usan las tablas."
        )
        if r.inputs.pair != "pT":
            notes.append(
                "Con el volumen como dato, un líquido es un problema delicado: casi no se "
                "comprime, así que un error chico del modelo en v se vuelve un error enorme en "
                "p (o en T)."
            )
    elif abs(r.Z - 1.0) <= 0.02:
        if r.p_R < 0.1:
            why = "con p_R ≪ 1 todo gas se comporta como ideal, a cualquier temperatura"
        elif r.T_R >= 2.0:
            why = "con T_R > 2 el gas se comporta como ideal salvo a p_R ≫ 1"
        else:
            why = "acá la atracción entre las moléculas y su volumen propio casi se compensan"
        notes.append(
            f"Z = {Z}: el gas ideal se equivoca {_pct(abs(ideal.error or 0.0))} y alcanza: "
            f"{why} (Çengel §3-7)."
        )
    else:
        if r.Z < 1.0:
            why = (
                "Z < 1: dominan las fuerzas de atracción entre las moléculas (a la misma p y T "
                "el gas es más denso que el ideal)"
            )
        else:
            why = (
                "Z > 1: domina el volumen propio de las moléculas (a la misma p y T el gas es "
                "menos denso que el ideal)"
            )
        notes.append(f"Z = {Z}: el gas ideal se equivoca {_pct(abs(ideal.error or 0.0))}. {why}.")
    if 0.9 <= r.T_R <= 1.2 and 0.5 <= r.p_R <= 2.0:
        notes.append(
            "Cerca del punto crítico Z cambia muy rápido con p y T: es donde todos los modelos "
            "se equivocan más."
        )
    if fl.key in ("Helium", "Hydrogen"):
        notes.append(
            f"El {fl.noun} es un gas cuántico (ω = {_num(fl.omega, '.3f')} < 0): la "
            "carta generalizada se armó con gases normales y no lo representa bien."
        )
    if fl.key in ("Water", "Ammonia", "R134a", "R32"):
        notes.append(
            f"Es un fluido polar (ω = {_num(fl.omega, '.3f')}): el ω corrige buena parte, pero "
            "los estados correspondientes de tres parámetros son para fluidos no polares."
        )
    if not (0.3 <= r.T_R <= 4.0 and r.p_R <= 10.0):
        notes.append(
            "Fuera del rango de ajuste de Lee y Kesler (0,3 ≤ T_R ≤ 4 y p_R ≤ 10): sus valores "
            "son extrapolaciones."
        )
    best = r.best
    if best is not None:
        notes.append(
            f"El modelo que menos se equivoca acá es {MODEL_PHRASES[best.model]}: "
            f"{_pct(abs(best.error or 0.0), 2)}."
        )
    return tuple(notes)


# ---------------------------------------------------------------------
# Gráficos: la carta generalizada y Z(p) a la T del dato
# ---------------------------------------------------------------------


def generalized_chart(
    Tr_values: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0),
    pr_max: float = 10.0,
    points: int = 80,
) -> dict[str, Any]:
    """Z⁰ de Lee y Kesler contra p_R para cada T_R, y la campana (Çengel, figura 3-51).

    Devuelve ``{"isotherms": {T_R: (p_R, Z)}, "dome": (p_R, Z_f, Z_g)}``, con p_R de 0,01
    a ``pr_max`` en escala logarítmica: bajo T_R = 1 cada isoterma sigue el vapor hasta
    p_sat (de la ecuación de Lee y Kesler con ω = 0) y el líquido desde ahí, con un salto
    vertical en p_sat.
    """
    isotherms: dict[float, tuple[list[float], list[float]]] = {}
    for Tr in Tr_values:
        psat = lee_kesler_psat_reduced(Tr) if Tr < 1.0 else math.inf
        if Tr < 1.0:
            grid = [
                *np.geomspace(0.01, psat * (1 - 1e-9), max(points // 3, 12)),
                *np.geomspace(psat * (1 + 1e-9), pr_max, points),
            ]
        else:
            grid = list(np.geomspace(0.01, pr_max, points))
        xs: list[float] = []
        zs: list[float] = []
        for pr in grid:
            phase: Phase = "vapor" if pr < psat else "liquid"
            try:
                z = _lk_z(Tr, float(pr), "simple", phase)[1]
            except ValueError:
                continue
            xs.append(float(pr))
            zs.append(float(z))
        isotherms[Tr] = (xs, zs)
    pr_sat: list[float] = []
    zf: list[float] = []
    zg: list[float] = []
    for Tr in np.linspace(0.6, 0.995, 40):
        ps = lee_kesler_psat_reduced(float(Tr))
        try:
            g = _lk_z(float(Tr), ps, "simple", "vapor")[1]
            f = _lk_z(float(Tr), ps, "simple", "liquid")[1]
        except ValueError:
            continue
        pr_sat.append(ps)
        zg.append(g)
        zf.append(f)
    # la campana se cierra en el punto crítico del fluido simple (p_R = 1, Z = Z_cr)
    pr_sat.append(1.0)
    zf.append(_LK_ZC["simple"])
    zg.append(_LK_ZC["simple"])
    return {"isotherms": isotherms, "dome": (pr_sat, zf, zg)}


def z_isotherm(
    fluid_key: str, T_K: float, p_max_Pa: float | None = None, points: int = 60
) -> dict[str, Any]:
    """Z(p) a la temperatura del dato con cada modelo, en la fase del fluido real.

    Bajo la temperatura crítica, hasta p_sat el vapor y desde ahí el líquido (los
    modelos toman la raíz de esa misma fase), con los dos puntos de la saturación.
    Devuelve ``{"p": [...], "p_sat": p_sat | None, "Z": {modelo: [Z o None]}}``;
    p_max por defecto, el mayor de 2·p_cr y lo que pida el gráfico.
    """
    fl = real_fluid(fluid_key)
    p_hi = min(p_max_Pa or 2.0 * fl.p_cr, fl.p_max)
    state = CoolProp.AbstractState("HEOS", fl.key)
    p_sat: float | None = None
    if T_K < fl.T_cr:
        state.update(CoolProp.QT_INPUTS, 0.0, T_K)
        p_sat = float(state.p())
    grid = [float(p) for p in np.linspace(p_hi / points, p_hi, points)]
    if p_sat is not None and p_sat < p_hi:
        grid = sorted([p for p in grid if abs(p / p_sat - 1.0) > 1e-3] + [p_sat, p_sat])
    out: dict[ModelKind, list[float | None]] = {k: [] for k in MODELS}
    first_sat = True
    for p in grid:
        if p_sat is not None and p == p_sat:
            phase: Phase = "vapor" if first_sat else "liquid"
            state.update(CoolProp.QT_INPUTS, 0.0 if phase == "liquid" else 1.0, T_K)
            first_sat = False
        else:
            phase = "vapor" if p_sat is None or p < p_sat else "liquid"
            try:
                state.update(CoolProp.PT_INPUTS, p, T_K)
            except ValueError:
                for k in MODELS:
                    out[k].append(None)
                continue
        v = 1.0 / float(state.rhomass())
        out["coolprop"].append(float(p * v / (fl.R * T_K)))
        out["ideal"].append(1.0)
        for kind in ("lk0", "lk", "vdw", "pr"):
            try:
                if kind in ("lk0", "lk"):
                    z = _lk_v(fl, p, T_K, phase, kind == "lk") * p / (fl.R * T_K)
                else:
                    z = _cubic_z(fl, kind, p, T_K, phase)[0]  # type: ignore[arg-type]
            except ValueError:
                out[kind].append(None)  # type: ignore[index]
                continue
            out[kind].append(float(z))  # type: ignore[index]
    return {"p": grid, "p_sat": p_sat, "Z": out}


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RealExample:
    inputs: CompressibilityInputs
    note: str = ""


_FT3_LB = 0.3048**3 / 0.45359237
C = 273.15

REAL_EXAMPLES: dict[str, RealExample] = {
    "R-134a a 1 MPa y 50 °C (Çengel §3-7)": RealExample(
        CompressibilityInputs("R134a", "pT", 1e6, 50.0 + C),
        "Çengel: v = 0,021796 m³/kg de las tablas, 0,026325 como gas ideal (20,8 % de error) "
        "y Z = 0,84 de la carta (0,0221 m³/kg, menos de 2 % de error).",
    ),
    "Nitrógeno a 175 K y 0,00375 m³/kg (Çengel §3-8)": RealExample(
        CompressibilityInputs("Nitrogen", "Tv", 175.0, 0.00375),
        "Çengel: 10 000 kPa medidos, 13 851 kPa como gas ideal y 9471 kPa con Van der Waals "
        "(con a y b de su tabla, redondeadas; con el punto crítico de CoolProp sale 9511 kPa).",
    ),
    "Vapor de agua a 600 °F y 0,51431 ft³/lb (Çengel §3-7)": RealExample(
        CompressibilityInputs("Water", "Tv", (600.0 + 459.67) * 5 / 9, 0.51431 * _FT3_LB),
        "Çengel: 1000 psia de las tablas, 1228 psia como gas ideal y 1056 psia con la carta "
        "(leída con v′_R y T_R: p_R ≈ 0,33).",
    ),
    "Metano de un gasoducto a 70 bar y 15 °C": RealExample(
        CompressibilityInputs("Methane", "pT", 70e5, 15.0 + C),
        "Z ≈ 0,87: el gas natural se factura por volumen corregido con Z.",
    ),
    "Dióxido de carbono cerca del punto crítico: 80 bar y 35 °C": RealExample(
        CompressibilityInputs("CarbonDioxide", "pT", 80e5, 35.0 + C),
        "Justo arriba del punto crítico (31 °C y 73,8 bar): Z cambia muy rápido.",
    ),
    "Aire a 1 atm y 25 °C": RealExample(
        CompressibilityInputs("Air", "pT", 101_325.0, 25.0 + C),
        "Casi ideal: Z = 0,9997.",
    ),
    "Hidrógeno a 700 bar y 25 °C (el tanque de un auto)": RealExample(
        CompressibilityInputs("Hydrogen", "pT", 700e5, 25.0 + C),
        "Z ≈ 1,45: a esa presión el hidrógeno ocupa 45 % más que como gas ideal.",
    ),
    "Aire de un tubo a 200 bar: la temperatura con p y v": RealExample(
        CompressibilityInputs("Air", "pv", 200e5, 0.0044438),
        "CoolProp: 300 K. Como gas ideal saldrían 310 K.",
    ),
    "Agua líquida a 1 bar y 20 °C": RealExample(
        CompressibilityInputs("Water", "pT", 1e5, 20.0 + C),
        "Las tablas dan 0,001002 m³/kg: los modelos de gas real no son para líquidos.",
    ),
}


def compressibility_to_dict(r: CompressibilityResult) -> dict[str, Any]:
    """El resultado para exportar (SI)."""
    return {
        "fluido": r.fluid.name,
        "dato": r.inputs.pair,
        "incognita": r.unknown,
        "T_cr_K": r.fluid.T_cr,
        "p_cr_Pa": r.fluid.p_cr,
        "v_cr_m3_per_kg": r.fluid.v_cr,
        "Z_cr": r.fluid.Z_cr,
        "omega": r.fluid.omega,
        "estado_real": {
            "p_Pa": r.p_Pa,
            "T_K": r.T_K,
            "v_m3_per_kg": r.v_m3_per_kg,
            "Z": r.Z,
            "fase": r.phase,
        },
        "p_R": r.p_R,
        "T_R": r.T_R,
        "v_R": r.v_R,
        "v_R_pseudo": r.v_R_pseudo,
        "modelos": {
            m.model: {
                "p_Pa": m.p_Pa,
                "T_K": m.T_K,
                "v_m3_per_kg": m.v_m3_per_kg,
                "Z": m.Z,
                "error": m.error,
                "raices_Z": list(m.roots),
                "nota": m.note,
            }
            for m in r.models
        },
        "notas": list(r.notes),
    }
