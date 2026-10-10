"""Las cúbicas de Van der Waals y Peng–Robinson: raíces, Maxwell y la campana (vademecum §8).

Con (p, T) la ecuación de Van der Waals (§8.1), escrita en Z, es una cúbica:

    Z³ − (1 + B)·Z² + A·Z − A·B = 0,  A = a·p/(R·T)²,  B = b·p/(R·T),

y la de Peng y Robinson (1976), Z³ − (1 − B)·Z² + (A − 3B² − 2B)·Z − (A·B − B² − B³) = 0.
Bajo la temperatura crítica puede tener tres raíces reales: la menor es el líquido, la
mayor el vapor y la del medio no tiene sentido físico (ahí ∂p/∂v > 0). Cuál de las
dos es la estable lo decide la energía libre de Gibbs: la de menor coeficiente de
fugacidad φ.

**Construcción de Maxwell**: la presión de saturación del modelo es la que iguala las
dos áreas que la recta horizontal corta al lazo de la isoterma,

    ∫ p·dv (de v_f a v_g) = p_sat·(v_g − v_f),

lo mismo que g_f = g_g (φ_f = φ_g), que es como se calcula acá. Van der Waals da a
T_R = 0,7 una p_R,sat = 0,200: equivale a un fluido con ω = −0,30 (los reales tienen
ω de 0 a 0,35), y por eso le erra a la presión de vapor; Peng–Robinson ajusta α(T)
con el ω del fluido.

Las fórmulas de ln φ: Van der Waals, ln φ = Z − 1 − ln(Z − B) − A/Z; Peng–Robinson,
ln φ = Z − 1 − ln(Z − B) − A/(2√2·B)·ln[(Z + (1 + √2)·B)/(Z + (1 − √2)·B)]
(Peng y Robinson, 1976, ec. 15).

Todo en SI y por kg (a en Pa·m⁶/kg², b en m³/kg, R = R_u/M); los adimensionales A y
B son los mismos que por mol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import CoolProp
import numpy as np
from scipy.optimize import brentq

from core.gases.real import (
    CUBIC_ZC,
    RealFluid,
    cubic_ab,
    cubic_coefficients,
    cubic_constants,
    cubic_p,
    cubic_z_roots,
    pr_kappa,
    real_fluid,
)

__all__ = [
    "CUBIC_EXAMPLES",
    "CUBIC_MODELS",
    "CubicExample",
    "CubicInputs",
    "CubicResult",
    "CubicRoot",
    "CubicSolution",
    "MaxwellSaturation",
    "cubic_solution",
    "cubic_to_dict",
    "fluid_isotherms",
    "implied_omega",
    "ln_phi",
    "maxwell",
    "saturation_curves",
    "solve_cubic",
    "vdw_reduced_isotherms",
    "vdw_reduced_p",
]

CubicKind = Literal["vdw", "pr"]
RootKind = Literal["liquid", "vapor", "unstable", "supercritical"]
CUBIC_MODELS: dict[CubicKind, str] = {"vdw": "Van der Waals", "pr": "Peng–Robinson"}
ROOT_NAMES: dict[RootKind, str] = {
    "liquid": "líquido",
    "vapor": "vapor",
    "unstable": "sin sentido físico",
    "supercritical": "fluido supercrítico",
}
_SQ2 = math.sqrt(2.0)


def _num(x: float, fmt: str = ".3g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


def _bar(p: float) -> str:
    return f"{_num(p / 1e5, '.4g')} bar"


def ln_phi(model: CubicKind, Z: float, A: float, B: float) -> float:
    """ln φ de una raíz Z (Peng y Robinson, 1976, ec. 15; Van der Waals, la misma idea)."""
    if model == "vdw":
        return Z - 1.0 - math.log(Z - B) - A / Z
    return (
        Z
        - 1.0
        - math.log(Z - B)
        - A / (2.0 * _SQ2 * B) * math.log((Z + (1.0 + _SQ2) * B) / (Z + (1.0 - _SQ2) * B))
    )


# ---------------------------------------------------------------------
# Maxwell: la presión de saturación de cada cúbica
# ---------------------------------------------------------------------


def _spinodals(fl: RealFluid, model: CubicKind, T: float) -> tuple[float, float] | None:
    """Los volúmenes del mínimo y del máximo de la isoterma (∂p/∂v = 0), o None si no hay lazo.

    Van der Waals: R·T·v³ − 2a·(v − b)² = 0; Peng–Robinson:
    R·T·(v² + 2bv − b²)² − 2a·(v + b)·(v − b)² = 0.
    """
    a, b = cubic_constants(fl, model, T)
    RT = fl.R * T
    if model == "vdw":
        poly = np.array([RT, -2.0 * a, 4.0 * a * b, -2.0 * a * b**2])
    else:
        q = np.polynomial.polynomial.Polynomial([-(b**2), 2.0 * b, 1.0])  # v² + 2bv − b²
        r = np.polynomial.polynomial.Polynomial([b, 1.0]) * np.polynomial.polynomial.Polynomial(
            [b**2, -2.0 * b, 1.0]
        )  # (v + b)·(v − b)²
        poly = (RT * q**2 - 2.0 * a * r).coef[::-1]
    vs = sorted(
        float(x.real) for x in np.roots(poly) if abs(x.imag) < 1e-12 * abs(x) and x.real > b
    )
    if len(vs) < 2:
        return None
    return vs[0], vs[-1]


@dataclass(frozen=True)
class MaxwellSaturation:
    """La saturación de una cúbica a T: p_sat y los volúmenes del líquido y del vapor."""

    T_K: float
    p_sat_Pa: float
    v_f: float  # m³/kg
    v_g: float
    Z_f: float
    Z_g: float
    p_spinodal_liquid_Pa: float  # el mínimo del lazo (puede ser negativo)
    p_spinodal_vapor_Pa: float  # el máximo del lazo


def maxwell(fl: RealFluid, model: CubicKind, T: float) -> MaxwellSaturation | None:
    """La construcción de Maxwell (áreas iguales, φ_f = φ_g) a T; None si T ≥ T_cr del modelo."""
    if T >= fl.T_cr * 0.9995:
        return None
    sp = _spinodals(fl, model, T)
    if sp is None:
        return None
    p_min = cubic_p(fl, model, T, sp[0])
    p_max = cubic_p(fl, model, T, sp[1])
    if p_max <= 0.0 or p_max <= p_min:
        return None
    lo = p_min * (1.0 + 1e-9) if p_min > 0.0 else p_max * 1e-12
    hi = p_max * (1.0 - 1e-9)

    def gap(p: float) -> float:
        A, B = cubic_ab(fl, model, T, p)
        roots = cubic_z_roots(model, A, B)
        return ln_phi(model, roots[0], A, B) - ln_phi(model, roots[-1], A, B)

    try:
        p_sat = float(brentq(gap, lo, hi, xtol=1e-12 * hi, rtol=1e-14))
    except ValueError:
        return None
    A, B = cubic_ab(fl, model, T, p_sat)
    roots = cubic_z_roots(model, A, B)
    RT_p = fl.R * T / p_sat
    return MaxwellSaturation(
        T_K=T,
        p_sat_Pa=p_sat,
        v_f=roots[0] * RT_p,
        v_g=roots[-1] * RT_p,
        Z_f=roots[0],
        Z_g=roots[-1],
        p_spinodal_liquid_Pa=p_min,
        p_spinodal_vapor_Pa=p_max,
    )


def implied_omega(fl: RealFluid, model: CubicKind) -> float:
    """El ω que «tiene» la cúbica: −log₁₀(p_sat/p_cr) − 1 a T_R = 0,7 (Van der Waals: −0,30)."""
    sat = maxwell(fl, model, 0.7 * fl.T_cr)
    if sat is None:  # pragma: no cover - a T_R = 0,7 siempre hay lazo
        raise ValueError("Sin saturación a T_R = 0,7.")
    return -math.log10(sat.p_sat_Pa / fl.p_cr) - 1.0


# ---------------------------------------------------------------------
# Las raíces con (p, T)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CubicRoot:
    Z: float
    v_m3_per_kg: float
    kind: RootKind
    ln_phi: float | None  # None en la del medio (no es un estado)
    stable: bool


@dataclass(frozen=True)
class CubicSolution:
    """Lo que dice una cúbica en (p, T)."""

    model: CubicKind
    a: float  # Van der Waals: a; Peng–Robinson: a·α(T) (Pa·m⁶/kg²)
    a_cr: float  # a en el punto crítico (sin α)
    b: float
    kappa: float | None  # Peng–Robinson
    alpha: float  # 1 en Van der Waals
    A: float
    B: float
    coefficients: tuple[float, float, float]  # c₂, c₁, c₀ de Z³ + c₂Z² + c₁Z + c₀
    roots: tuple[CubicRoot, ...]
    saturation: MaxwellSaturation | None
    phase: RootKind  # la fase estable según el modelo (liquid, vapor o supercritical)

    @property
    def stable(self) -> CubicRoot:
        return next(r for r in self.roots if r.stable)

    def root(self, kind: RootKind) -> CubicRoot | None:
        return next((r for r in self.roots if r.kind == kind), None)


def cubic_solution(fl: RealFluid, model: CubicKind, p: float, T: float) -> CubicSolution:
    """Las raíces de la cúbica en (p, T), cuál es cada una y cuál es la estable."""
    a, b = cubic_constants(fl, model, T)
    a_cr, _ = cubic_constants(fl, model, fl.T_cr)
    kappa = pr_kappa(fl.omega) if model == "pr" else None
    alpha = a / a_cr
    A, B = cubic_ab(fl, model, T, p)
    zs = cubic_z_roots(model, A, B)
    if not zs:  # pragma: no cover - una cúbica con B > 0 siempre tiene una raíz Z > B
        raise ValueError(f"{CUBIC_MODELS[model]} no tiene raíz a esa presión y temperatura.")
    Tr, pr = T / fl.T_cr, p / fl.p_cr
    sat = maxwell(fl, model, T)
    kinds: list[RootKind]
    if len(zs) >= 3:
        kinds = ["liquid", *(["unstable"] * (len(zs) - 2)), "vapor"]
    elif Tr >= 1.0 or sat is None:
        kinds = ["supercritical"] * len(zs)
    else:
        z_line = CUBIC_ZC[model] * pr / Tr  # v = v_cr
        kinds = ["vapor" if z > CUBIC_ZC[model] and z > z_line else "liquid" for z in zs]
    lnphis = [
        None if k == "unstable" else ln_phi(model, z, A, B) for z, k in zip(zs, kinds, strict=True)
    ]
    physical = [i for i, k in enumerate(kinds) if k != "unstable"]
    stable_i = min(physical, key=lambda i: lnphis[i])  # type: ignore[arg-type, return-value]
    RT_p = fl.R * T / p
    roots = tuple(
        CubicRoot(z, z * RT_p, k, lp, i == stable_i)
        for i, (z, k, lp) in enumerate(zip(zs, kinds, lnphis, strict=True))
    )
    return CubicSolution(
        model=model,
        a=a,
        a_cr=a_cr,
        b=b,
        kappa=kappa,
        alpha=alpha,
        A=A,
        B=B,
        coefficients=cubic_coefficients(model, A, B),
        roots=roots,
        saturation=sat,
        phase=kinds[stable_i],
    )


@dataclass(frozen=True)
class CubicInputs:
    fluid_key: str
    p_Pa: float
    T_K: float


@dataclass(frozen=True)
class CubicResult:
    inputs: CubicInputs
    fluid: RealFluid
    phase: RootKind  # la del fluido real (CoolProp)
    v_m3_per_kg: float  # la real
    Z: float
    p_sat_Pa: float | None  # la real a T (None sobre la crítica)
    v_f: float | None
    v_g: float | None
    p_R: float
    T_R: float
    vdw: CubicSolution
    pr: CubicSolution
    omega_vdw: float
    omega_pr: float
    notes: tuple[str, ...]

    def solution(self, model: CubicKind) -> CubicSolution:
        return self.vdw if model == "vdw" else self.pr

    def model_v(self, model: CubicKind) -> float | None:
        """El v del modelo en la fase del fluido real (la raíz que se compara)."""
        sol = self.solution(model)
        want: RootKind = "supercritical" if self.phase == "supercritical" else self.phase
        root = sol.root(want) or (sol.roots[0] if len(sol.roots) == 1 else None)
        if root is None or (want != "supercritical" and root.kind not in (want, "supercritical")):
            return None
        return root.v_m3_per_kg


def solve_cubic(inputs: CubicInputs) -> CubicResult:
    """Van der Waals y Peng–Robinson en (p, T), contra el fluido real (vademecum §8).

    Raises
    ------
    ValueError
        Con un dato que no es positivo o fuera del rango de la ecuación de estado.
    """
    fl = real_fluid(inputs.fluid_key)
    p, T = inputs.p_Pa, inputs.T_K
    if not (math.isfinite(p) and p > 0.0):
        raise ValueError("La presión (absoluta) tiene que ser positiva.")
    if not (math.isfinite(T) and T > 0.0):
        raise ValueError("La temperatura (absoluta) tiene que ser positiva.")
    name = fl.noun
    if not (fl.T_min <= T <= fl.T_max):
        raise ValueError(
            f"La temperatura ({_num(T, '.4g')} K) está fuera del rango de la ecuación de estado "
            f"del {name}: de {_num(fl.T_min, '.4g')} K a {_num(fl.T_max, '.4g')} K."
        )
    if p > fl.p_max:
        raise ValueError(
            f"La presión ({_num(p / 1e6, '.4g')} MPa) supera la máxima de la ecuación de estado "
            f"del {name} ({_num(fl.p_max / 1e6, '.4g')} MPa)."
        )
    state = CoolProp.AbstractState("HEOS", fl.key)
    p_sat = v_f = v_g = None
    if T < fl.T_cr:
        state.update(CoolProp.QT_INPUTS, 0.0, T)
        p_sat, v_f = float(state.p()), 1.0 / float(state.rhomass())
        state.update(CoolProp.QT_INPUTS, 1.0, T)
        v_g = 1.0 / float(state.rhomass())
        if abs(p / p_sat - 1.0) < 1e-6:
            raise ValueError(
                "Esa presión es justo la de saturación a esa temperatura: el estado puede ser "
                "líquido, vapor o una mezcla. Corré un poco la presión o la temperatura."
            )
    try:
        state.update(CoolProp.PT_INPUTS, p, T)
    except ValueError as exc:
        raise ValueError(
            f"CoolProp no encuentra ese estado del {name}. Revisá que los datos estén dentro "
            f"del rango de la ecuación de estado ({exc})."
        ) from exc
    v = 1.0 / float(state.rhomass())
    if T >= fl.T_cr:
        phase: RootKind = "supercritical"
    else:
        phase = "liquid" if p > (p_sat or 0.0) else "vapor"
    result = CubicResult(
        inputs=inputs,
        fluid=fl,
        phase=phase,
        v_m3_per_kg=v,
        Z=p * v / (fl.R * T),
        p_sat_Pa=p_sat,
        v_f=v_f,
        v_g=v_g,
        p_R=p / fl.p_cr,
        T_R=T / fl.T_cr,
        vdw=cubic_solution(fl, "vdw", p, T),
        pr=cubic_solution(fl, "pr", p, T),
        omega_vdw=implied_omega(fl, "vdw"),
        omega_pr=implied_omega(fl, "pr"),
        notes=(),
    )
    return _with_notes(result)


def _vs_real(ratio: float) -> str:
    """Cuánto se aparta un valor del real: en % si es menos del doble, si no, cuántas veces."""
    if 0.5 <= ratio <= 2.0:
        return f"{_num(100.0 * (ratio - 1.0), '+.1f')} %"
    return f"{_num(ratio, '.3g')} veces la real"


def _with_notes(r: CubicResult) -> CubicResult:
    notes: list[str] = []
    fl = r.fluid
    phase_name = {"liquid": "líquido", "vapor": "vapor", "supercritical": "supercrítico"}
    for model in ("vdw", "pr"):
        sol = r.solution(model)  # type: ignore[arg-type]
        name = CUBIC_MODELS[model]  # type: ignore[index]
        if len(sol.roots) >= 3 and sol.saturation is not None:
            sign = ">" if sol.phase == "liquid" else "<"
            notes.append(
                f"{name}: tres raíces. La estable es la del {phase_name[sol.phase]}, la de "
                f"menor energía libre de Gibbs (p {sign} p_sat del modelo, "
                f"{_bar(sol.saturation.p_sat_Pa)}); la otra es metaestable y la del medio no "
                "tiene sentido físico."
            )
        if r.phase != "supercritical" and sol.phase != "supercritical" and sol.phase != r.phase:
            notes.append(
                f"Para {name} este estado es {phase_name[sol.phase]}, pero el fluido real es "
                f"{phase_name[r.phase]}: su campana no coincide con la real."
            )
    if r.p_sat_Pa is not None and r.vdw.saturation is not None and r.pr.saturation is not None:
        notes.append(
            f"Presión de saturación a esta T: la real es {_bar(r.p_sat_Pa)}; Van der Waals da "
            f"{_bar(r.vdw.saturation.p_sat_Pa)} "
            f"({_vs_real(r.vdw.saturation.p_sat_Pa / r.p_sat_Pa)}) y Peng–Robinson "
            f"{_bar(r.pr.saturation.p_sat_Pa)} ({_vs_real(r.pr.saturation.p_sat_Pa / r.p_sat_Pa)})."
        )
    notes.append(
        f"Van der Waals equivale a un fluido con ω = {_num(r.omega_vdw, '.2f')} (el {fl.noun} "
        f"tiene {_num(fl.omega, '.3f')}) y Z_cr = 3/8 = 0,375 (el real, {_num(fl.Z_cr, '.3f')}): "
        "por eso le erra a la presión de vapor y a la densidad del líquido. Peng–Robinson usa "
        "el ω del fluido."
    )
    if r.T_R >= 1.0:
        notes.append(
            "Sobre la temperatura crítica la cúbica tiene una sola raíz: no hay líquido ni "
            "vapor sino un fluido supercrítico."
        )
    return replace(r, notes=tuple(notes))


# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------


def vdw_reduced_p(Tr: float, vr: Any) -> Any:
    """La forma reducida de Van der Waals (vademecum §8.4): p_R = 8·T_R/(3·v_R − 1) − 3/v_R²."""
    return 8.0 * Tr / (3.0 * vr - 1.0) - 3.0 / vr**2


def _vdw_reduced_maxwell(Tr: float) -> tuple[float, float, float] | None:
    """(p_R,sat, v_R,f, v_R,g) de Van der Waals reducido; None si T_R ≥ 1."""
    if Tr >= 0.9995:
        return None
    sp = sorted(
        float(x.real)
        for x in np.roots([4.0 * Tr, -9.0, 6.0, -1.0])
        if abs(x.imag) < 1e-12 and x.real > 1.0 / 3.0
    )
    p_min, p_max = vdw_reduced_p(Tr, sp[0]), vdw_reduced_p(Tr, sp[-1])
    lo = p_min * (1.0 + 1e-9) if p_min > 0.0 else 1e-12
    hi = p_max * (1.0 - 1e-9)

    def zs(pr: float) -> tuple[list[float], float, float]:
        A, B = 27.0 / 64.0 * pr / Tr**2, pr / (8.0 * Tr)
        return list(cubic_z_roots("vdw", A, B)), A, B

    def gap(pr: float) -> float:
        roots, A, B = zs(pr)
        return ln_phi("vdw", roots[0], A, B) - ln_phi("vdw", roots[-1], A, B)

    ps = float(brentq(gap, lo, hi, xtol=1e-14))
    roots, _, _ = zs(ps)
    # Z = p·v/(R·T) = (3/8)·p_R·v_R/T_R
    return ps, 8.0 * roots[0] * Tr / (3.0 * ps), 8.0 * roots[-1] * Tr / (3.0 * ps)


def vdw_reduced_isotherms(
    Tr_values: tuple[float, ...] = (0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2),
    points: int = 300,
) -> dict[str, Any]:
    """Las isotermas reducidas de Van der Waals con su lazo y la recta de Maxwell.

    Devuelve ``{"isotherms": {T_R: (v_R, p_R)}, "maxwell": {T_R: (p_R,sat, v_R,f,
    v_R,g)}, "dome": (v_R, p_R)}`` con v_R = v/v_cr y v_cr = 3b; la campana de Van
    der Waals es la curva de los extremos de las rectas de Maxwell.
    """
    vr = np.geomspace(0.4, 20.0, points)
    isotherms = {Tr: (vr.tolist(), vdw_reduced_p(Tr, vr).tolist()) for Tr in Tr_values}
    lines = {Tr: m for Tr in Tr_values if (m := _vdw_reduced_maxwell(Tr)) is not None}
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for Tr in np.linspace(0.45, 0.999, 60):
        m = _vdw_reduced_maxwell(float(Tr))
        if m is not None:
            left.append((m[1], m[0]))
            right.append((m[2], m[0]))
    dome = [*left, (1.0, 1.0), *reversed(right)]
    return {
        "isotherms": isotherms,
        "maxwell": lines,
        "dome": ([v for v, _ in dome], [p for _, p in dome]),
    }


def fluid_isotherms(fluid_key: str, T_K: float, points: int = 400) -> dict[str, Any]:
    """p(v) a T del fluido real (CoolProp, con el tramo horizontal de la campana), de Van der
    Waals y de Peng–Robinson (con su lazo), y las rectas de Maxwell de las dos cúbicas.

    Devuelve ``{"v": [...], "p": {"coolprop" | "vdw" | "pr": [...]}, "maxwell":
    {"vdw" | "pr": MaxwellSaturation | None}, "p_sat": p_sat real | None}``.
    """
    fl = real_fluid(fluid_key)
    state = CoolProp.AbstractState("HEOS", fl.key)
    p_sat = None
    if T_K < fl.T_cr:
        state.update(CoolProp.QT_INPUTS, 0.0, T_K)
        p_sat = float(state.p())
        v_lo = 0.85 / float(state.rhomass())
        state.update(CoolProp.QT_INPUTS, 1.0, T_K)
        v_hi = 8.0 / float(state.rhomass())
    else:
        state.update(CoolProp.PT_INPUTS, min(3.0 * fl.p_cr, fl.p_max), T_K)
        v_lo = 1.0 / float(state.rhomass())
        v_hi = 40.0 * fl.R * T_K / fl.p_cr
    _, b_pr = cubic_constants(fl, "pr", T_K)
    _, b_vdw = cubic_constants(fl, "vdw", T_K)
    v_lo = min(v_lo, 1.05 * b_pr)
    vs = np.geomspace(v_lo, v_hi, points)
    curves: dict[str, list[float | None]] = {"coolprop": [], "vdw": [], "pr": []}
    for v in vs:
        try:
            state.update(CoolProp.DmassT_INPUTS, 1.0 / float(v), T_K)
            curves["coolprop"].append(float(state.p()))
        except ValueError:
            curves["coolprop"].append(None)
        for model, b in (("vdw", b_vdw), ("pr", b_pr)):
            curves[model].append(
                cubic_p(fl, model, T_K, float(v)) if v > 1.01 * b else None  # type: ignore[arg-type]
            )
    return {
        "v": vs.tolist(),
        "p": curves,
        "maxwell": {m: maxwell(fl, m, T_K) for m in ("vdw", "pr")},  # type: ignore[arg-type]
        "p_sat": p_sat,
    }


def saturation_curves(fluid_key: str, points: int = 40) -> dict[str, Any]:
    """p_sat(T) del fluido real, de Van der Waals y de Peng–Robinson (Maxwell), de T_R = 0,5
    (o la mínima del fluido) a la crítica."""
    fl = real_fluid(fluid_key)
    T_lo = max(0.5 * fl.T_cr, fl.T_min * 1.001)
    Ts = np.linspace(T_lo, 0.995 * fl.T_cr, points)
    state = CoolProp.AbstractState("HEOS", fl.key)
    out: dict[str, list[float | None]] = {"coolprop": [], "vdw": [], "pr": []}
    for T in Ts:
        state.update(CoolProp.QT_INPUTS, 0.0, float(T))
        out["coolprop"].append(float(state.p()))
        for m in ("vdw", "pr"):
            sat = maxwell(fl, m, float(T))  # type: ignore[arg-type]
            out[m].append(sat.p_sat_Pa if sat else None)
    return {"T": Ts.tolist(), "p_sat": out, "T_cr": fl.T_cr, "p_cr": fl.p_cr}


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CubicExample:
    inputs: CubicInputs
    note: str = ""


C = 273.15

CUBIC_EXAMPLES: dict[str, CubicExample] = {
    "R-134a a 1 MPa y 50 °C: tres raíces": CubicExample(
        CubicInputs("R134a", 1e6, 50.0 + C),
        "Vapor sobrecalentado: las dos cúbicas tienen tres raíces y la estable es la del vapor.",
    ),
    "R-134a líquido a 10 bar y 20 °C": CubicExample(
        CubicInputs("R134a", 10e5, 20.0 + C),
        "Para Van der Waals es vapor: su presión de vapor a 20 °C es más del doble de la real.",
    ),
    "Nitrógeno a 175 K y 100 bar (Çengel §3-8)": CubicExample(
        CubicInputs("Nitrogen", 100e5, 175.0),
        "Sobre la temperatura crítica (126 K): una sola raíz.",
    ),
    "Agua a 1 bar y 20 °C": CubicExample(
        CubicInputs("Water", 1e5, 20.0 + C),
        "Líquido: la raíz menor da el doble del volumen real con Van der Waals.",
    ),
    "Propano a 10 bar y 20 °C (una garrafa)": CubicExample(
        CubicInputs("n-Propane", 10e5, 20.0 + C),
        "Líquido comprimido: la presión de vapor del propano a 20 °C es 8,4 bar.",
    ),
    "Dióxido de carbono a 60 bar y 15 °C": CubicExample(
        CubicInputs("CarbonDioxide", 60e5, 15.0 + C),
        "Cerca del punto crítico (31 °C, 73,8 bar): las cúbicas sufren.",
    ),
}


def cubic_to_dict(r: CubicResult) -> dict[str, Any]:
    """El resultado para exportar (SI)."""

    def sol(s: CubicSolution) -> dict[str, Any]:
        sat = s.saturation
        return {
            "a": s.a,
            "a_cr": s.a_cr,
            "b": s.b,
            "kappa": s.kappa,
            "alpha": s.alpha,
            "A": s.A,
            "B": s.B,
            "coeficientes": list(s.coefficients),
            "raices": [
                {
                    "Z": root.Z,
                    "v_m3_per_kg": root.v_m3_per_kg,
                    "tipo": root.kind,
                    "ln_phi": root.ln_phi,
                    "estable": root.stable,
                }
                for root in s.roots
            ],
            "fase": s.phase,
            "p_sat_Pa": sat.p_sat_Pa if sat else None,
            "v_f": sat.v_f if sat else None,
            "v_g": sat.v_g if sat else None,
        }

    return {
        "fluido": r.fluid.name,
        "p_Pa": r.inputs.p_Pa,
        "T_K": r.inputs.T_K,
        "fase_real": r.phase,
        "v_real_m3_per_kg": r.v_m3_per_kg,
        "Z_real": r.Z,
        "p_sat_real_Pa": r.p_sat_Pa,
        "p_R": r.p_R,
        "T_R": r.T_R,
        "omega": r.fluid.omega,
        "omega_vdw": r.omega_vdw,
        "omega_pr": r.omega_pr,
        "van_der_waals": sol(r.vdw),
        "peng_robinson": sol(r.pr),
        "notas": list(r.notes),
    }
