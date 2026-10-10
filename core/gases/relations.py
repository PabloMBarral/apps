"""Funciones características y relaciones de Maxwell (vademecum §15).

En un estado de una sola fase, con la ecuación de estado de CoolProp:

- los potenciales (§15.1): u, h = u + p·v, f = u − T·s y g = h − T·s, y sus derivadas
  con las variables naturales (§15.2), que dan T, p, v y s;
- las cuatro **relaciones de Maxwell** (§15.3), cada lado de dos formas: la derivada
  exacta (CoolProp) y una diferencia centrada con pasos ΔT y Δp, como se haría con las
  tablas (Çengel §12-2: el vapor a 250 °C y 300 kPa). Las diferencias usan cuatro
  caminos: la isoterma y la isoentrópica en p ± Δp, la isobara y la isócora en T ± ΔT;
- α = (1/v)·(∂v/∂T)_p, κ_T = −(1/v)·(∂v/∂p)_T y la relación de Mayer generalizada
  c_p − c_v = T·v·α²/κ_T (§15.4 y §15.5), contra el límite del gas ideal (α = 1/T,
  κ_T = 1/p, c_p − c_v = R);
- el coeficiente de Joule–Thomson μ_JT = (∂T/∂p)_h = [T·(∂v/∂T)_p − v]/c_p
  (Çengel §12-5) y su curva de inversión.

Y en la saturación, la ecuación de Clapeyron, h_fg = T·v_fg·(dp/dT)_sat, y la
aproximación de Clausius–Clapeyron (Çengel §12-3).

Todo en SI. Los valores de u, h, s, f y g dependen del estado de referencia de cada
fluido (el de CoolProp); las derivadas y las diferencias, no.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import CoolProp
import numpy as np
from scipy.optimize import brentq

from core.gases.real import RealFluid, real_fluid

__all__ = [
    "CLAPEYRON_EXAMPLES",
    "JT_EXAMPLES",
    "JouleThomsonExample",
    "JouleThomsonInputs",
    "JouleThomsonResult",
    "clausius_curve",
    "joule_thomson",
    "joule_thomson_to_dict",
    "MAXWELL_KEYS",
    "RELATIONS_EXAMPLES",
    "ClapeyronExample",
    "ClapeyronInputs",
    "ClapeyronResult",
    "Derivative",
    "MaxwellRelation",
    "NaturalDerivative",
    "PathState",
    "RelationsExample",
    "RelationsInputs",
    "RelationsResult",
    "clapeyron",
    "clapeyron_to_dict",
    "inversion_curve",
    "isenthalps",
    "relations",
    "relations_to_dict",
]

MaxwellKey = Literal["u", "h", "f", "g"]
MAXWELL_KEYS: tuple[MaxwellKey, ...] = ("u", "h", "f", "g")
PathKind = Literal["isothermal", "isentropic", "isobaric", "isochoric"]
PATH_NAMES: dict[PathKind, str] = {
    "isothermal": "isoterma (T constante, p ± Δp)",
    "isentropic": "isoentrópica (s constante, p ± Δp)",
    "isobaric": "isobara (p constante, T ± ΔT)",
    "isochoric": "isócora (v constante, T ± ΔT)",
}


def _num(x: float, fmt: str = ".3g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


def _pct(x: float, decimals: int = 1) -> str:
    return f"{100.0 * x:.{decimals}f} %".replace(".", ",").replace("-", "−")


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sci(x: float, digits: int = 3) -> str:
    """Un número con 3 cifras; si es muy chico o muy grande, con ·10ⁿ (3,71·10⁻⁷)."""
    if x == 0.0 or 1e-3 <= abs(x) < 1e5:
        return _num(x, f".{digits}g")
    exp = math.floor(math.log10(abs(x)))
    mant = x / 10**exp
    return f"{_num(mant, f'.{digits}g')}·10{str(exp).translate(_SUP)}"


# ---------------------------------------------------------------------
# Estados
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PathState:
    """Un estado de CoolProp (SI)."""

    T: float
    p: float
    v: float
    s: float
    u: float
    h: float

    @property
    def f(self) -> float:
        return self.u - self.T * self.s

    @property
    def g(self) -> float:
        return self.h - self.T * self.s


def _read(state: CoolProp.AbstractState) -> PathState:
    return PathState(
        T=float(state.T()),
        p=float(state.p()),
        v=1.0 / float(state.rhomass()),
        s=float(state.smass()),
        u=float(state.umass()),
        h=float(state.hmass()),
    )


def _update(state: CoolProp.AbstractState, pair: str, x: float, y: float, what: str) -> PathState:
    """Un estado de un camino de diferencias finitas, que tiene que ser de una sola fase."""
    inputs = {
        "PT": CoolProp.PT_INPUTS,
        "PS": CoolProp.PSmass_INPUTS,
        "DT": CoolProp.DmassT_INPUTS,
        "DS": CoolProp.DmassSmass_INPUTS,
        "PH": CoolProp.HmassP_INPUTS,
    }[pair]
    try:
        if pair == "PH":
            state.update(inputs, y, x)  # HmassP_INPUTS: (h, p)
        else:
            state.update(inputs, x, y)
    except ValueError as exc:
        raise ValueError(
            f"CoolProp no encuentra el estado {what} de las diferencias finitas: achicá ΔT o Δp "
            f"({exc})."
        ) from exc
    if state.phase() == CoolProp.iphase_twophase:
        raise ValueError(
            f"Con esos pasos, el estado {what} (T = {_num(state.T(), '.4g')} K, p = "
            f"{_num(state.p() / 1e5, '.4g')} bar) cae dentro de la campana: achicá ΔT o Δp."
        )
    return _read(state)


# ---------------------------------------------------------------------
# Las relaciones de Maxwell
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Derivative:
    """Un lado de una relación de Maxwell: exacto (CoolProp) y por diferencias centradas."""

    key: str  # p. ej. "dT_dv_s"
    exact: float
    finite: float
    path: PathKind  # el camino de las diferencias
    numerator: Literal["T", "p", "v", "s"]
    denominator: Literal["T", "p", "v", "s"]

    @property
    def error(self) -> float:
        """El error relativo de la diferencia finita."""
        return self.finite / self.exact - 1.0 if self.exact else math.nan


@dataclass(frozen=True)
class MaxwellRelation:
    """Una relación de Maxwell: left = sign·right (vademecum §15.3)."""

    potential: MaxwellKey
    left: Derivative
    right: Derivative
    sign: int

    @property
    def gap_exact(self) -> float:
        """left − sign·right relativo con las derivadas exactas (≈ 10⁻¹⁴)."""
        return self.left.exact / (self.sign * self.right.exact) - 1.0

    @property
    def gap_finite(self) -> float:
        """La diferencia relativa entre los dos lados por diferencias finitas."""
        return self.left.finite / (self.sign * self.right.finite) - 1.0


@dataclass(frozen=True)
class NaturalDerivative:
    """Una derivada de un potencial con sus variables naturales (§15.2) y lo que tiene que dar."""

    potential: MaxwellKey
    key: str  # "du_ds_v", …
    value: float  # por diferencias centradas con un paso chico
    expected: float  # T, −p, v o −s
    expected_symbol: str  # "T", "-p", "v", "-s"


@dataclass(frozen=True)
class RelationsInputs:
    """El fluido, el estado (p, T) y los pasos de las diferencias finitas (SI)."""

    fluid_key: str
    p_Pa: float
    T_K: float
    dT_K: float = 10.0
    dp_Pa: float | None = None  # None: 10 % de p


@dataclass(frozen=True)
class RelationsResult:
    inputs: RelationsInputs
    fluid: RealFluid
    phase: Literal["liquid", "vapor", "supercritical"]
    state: PathState
    cp: float
    cv: float
    dT: float
    dp: float
    paths: dict[PathKind, tuple[PathState, PathState]]  # (−, +) de cada camino
    natural: tuple[NaturalDerivative, ...]
    maxwell: tuple[MaxwellRelation, ...]
    convergence: tuple[tuple[float, tuple[float, ...]], ...]  # (fracción del paso, brechas)
    alpha: float
    kappa_T: float
    mayer: float  # T·v·α²/κ_T
    mu_JT: float  # exacto
    mu_JT_formula: float  # [T·(∂v/∂T)_p − v]/c_p
    mu_JT_finite: float  # (T₊ − T₋)/(2Δp) a h constante
    isenthalpic: tuple[PathState, PathState]
    notes: tuple[str, ...]

    def relation(self, potential: MaxwellKey) -> MaxwellRelation:
        return next(m for m in self.maxwell if m.potential == potential)

    @property
    def degenerate(self) -> bool:
        """α ≈ 0 (el agua cerca de 4 °C): los dos lados de las cuatro relaciones valen ≈ 0."""
        return abs(self.alpha * self.state.T) < 1e-3


def _phase(
    state: CoolProp.AbstractState, fl: RealFluid
) -> Literal["liquid", "vapor", "supercritical"]:
    ph = state.phase()
    if ph in (CoolProp.iphase_liquid, CoolProp.iphase_supercritical_liquid):
        return "liquid"
    if state.T() >= fl.T_cr:
        return "supercritical"
    return "vapor"


def _paths(
    state: CoolProp.AbstractState, base: PathState, dT: float, dp: float
) -> dict[PathKind, tuple[PathState, PathState]]:
    """Los ocho estados de las diferencias finitas: (−, +) de cada camino."""
    rho = 1.0 / base.v
    return {
        "isothermal": (
            _update(state, "PT", base.p - dp, base.T, "a p − Δp de la isoterma"),
            _update(state, "PT", base.p + dp, base.T, "a p + Δp de la isoterma"),
        ),
        "isentropic": (
            _update(state, "PS", base.p - dp, base.s, "a p − Δp de la isoentrópica"),
            _update(state, "PS", base.p + dp, base.s, "a p + Δp de la isoentrópica"),
        ),
        "isobaric": (
            _update(state, "PT", base.p, base.T - dT, "a T − ΔT de la isobara"),
            _update(state, "PT", base.p, base.T + dT, "a T + ΔT de la isobara"),
        ),
        "isochoric": (
            _update(state, "DT", rho, base.T - dT, "a T − ΔT de la isócora"),
            _update(state, "DT", rho, base.T + dT, "a T + ΔT de la isócora"),
        ),
    }


def _quotient(path: tuple[PathState, PathState], num: str, den: str) -> float:
    lo, hi = path
    return (getattr(hi, num) - getattr(lo, num)) / (getattr(hi, den) - getattr(lo, den))


#: Cada relación: potencial, (lado izquierdo), (lado derecho), signo. Cada lado es
#: (clave, numerador, denominador, constante, camino).
_RELATIONS: tuple[tuple[MaxwellKey, tuple[str, ...], tuple[str, ...], int], ...] = (
    ("u", ("dT_dv_s", "T", "v", "s", "isentropic"), ("dp_ds_v", "p", "s", "v", "isochoric"), -1),
    ("h", ("dT_dp_s", "T", "p", "s", "isentropic"), ("dv_ds_p", "v", "s", "p", "isobaric"), 1),
    ("f", ("ds_dv_T", "s", "v", "T", "isothermal"), ("dp_dT_v", "p", "T", "v", "isochoric"), 1),
    ("g", ("ds_dp_T", "s", "p", "T", "isothermal"), ("dv_dT_p", "v", "T", "p", "isobaric"), -1),
)
_PARAM = {"T": CoolProp.iT, "p": CoolProp.iP, "s": CoolProp.iSmass, "v": CoolProp.iDmass}


def _exact(state: CoolProp.AbstractState, num: str, den: str, const: str) -> float:
    """(∂num/∂den)_const de CoolProp; v se deriva por la densidad: dv = −dρ/ρ²."""
    d = float(state.first_partial_deriv(_PARAM[num], _PARAM[den], _PARAM[const]))
    rho = float(state.rhomass())
    if num == "v":
        d *= -1.0 / rho**2
    if den == "v":
        d *= -(rho**2)
    return d


def _maxwell(
    state: CoolProp.AbstractState,
    base: PathState,
    paths: dict[PathKind, tuple[PathState, PathState]],
) -> tuple[MaxwellRelation, ...]:
    state.update(CoolProp.PT_INPUTS, base.p, base.T)
    out: list[MaxwellRelation] = []
    for potential, left, right, sign in _RELATIONS:
        sides = []
        for key, num, den, const, path in (left, right):
            sides.append(
                Derivative(
                    key=key,
                    exact=_exact(state, num, den, const),
                    finite=_quotient(paths[path], num, den),  # type: ignore[index]
                    path=path,  # type: ignore[arg-type]
                    numerator=num,  # type: ignore[arg-type]
                    denominator=den,  # type: ignore[arg-type]
                )
            )
        out.append(MaxwellRelation(potential, sides[0], sides[1], sign))
    return tuple(out)


def _natural(
    state: CoolProp.AbstractState, base: PathState, cp: float
) -> tuple[NaturalDerivative, ...]:
    """Las ocho derivadas de §15.2 por diferencias centradas con un paso de 10⁻⁵."""
    eps = 1e-5
    rho = 1.0 / base.v
    dT, dp, dv, ds = eps * base.T, eps * base.p, eps * base.v, eps * cp

    def pair(kind: str, x: float, y: float) -> PathState:
        return _update(state, kind, x, y, "de la derivada")

    # u(s, v): s ± ds a v, y v ± dv a s
    us = (pair("DS", rho, base.s - ds), pair("DS", rho, base.s + ds))
    uv = (pair("DS", 1.0 / (base.v - dv), base.s), pair("DS", 1.0 / (base.v + dv), base.s))
    # h(s, p)
    hs = (pair("PS", base.p, base.s - ds), pair("PS", base.p, base.s + ds))
    hp = (pair("PS", base.p - dp, base.s), pair("PS", base.p + dp, base.s))
    # f(T, v)
    fT = (pair("DT", rho, base.T - dT), pair("DT", rho, base.T + dT))
    fv = (pair("DT", 1.0 / (base.v - dv), base.T), pair("DT", 1.0 / (base.v + dv), base.T))
    # g(T, p)
    gT = (pair("PT", base.p, base.T - dT), pair("PT", base.p, base.T + dT))
    gp = (pair("PT", base.p - dp, base.T), pair("PT", base.p + dp, base.T))
    rows = (
        ("u", "du_ds_v", us, "u", "s", base.T, "T"),
        ("u", "du_dv_s", uv, "u", "v", -base.p, "-p"),
        ("h", "dh_ds_p", hs, "h", "s", base.T, "T"),
        ("h", "dh_dp_s", hp, "h", "p", base.v, "v"),
        ("f", "df_dT_v", fT, "f", "T", -base.s, "-s"),
        ("f", "df_dv_T", fv, "f", "v", -base.p, "-p"),
        ("g", "dg_dT_p", gT, "g", "T", -base.s, "-s"),
        ("g", "dg_dp_T", gp, "g", "p", base.v, "v"),
    )
    return tuple(
        NaturalDerivative(pot, key, _quotient(path, num, den), exp, sym)  # type: ignore[arg-type]
        for pot, key, path, num, den, exp, sym in rows
    )


def relations(inputs: RelationsInputs) -> RelationsResult:
    """Los potenciales, las relaciones de Maxwell, α, κ_T, Mayer y Joule–Thomson en (p, T).

    Raises
    ------
    ValueError
        Con datos no positivos, fuera del rango de la ecuación de estado o un estado (el
        dato o uno de las diferencias finitas) dentro de la campana.
    """
    fl = real_fluid(inputs.fluid_key)
    p, T = inputs.p_Pa, inputs.T_K
    dT = inputs.dT_K
    dp = inputs.dp_Pa if inputs.dp_Pa is not None else 0.1 * p
    for msg, value in (
        ("La presión (absoluta) tiene que ser positiva", p),
        ("La temperatura (absoluta) tiene que ser positiva", T),
        ("El paso ΔT tiene que ser positivo", dT),
        ("El paso Δp tiene que ser positivo", dp),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{msg}.")
    if dp >= p:
        raise ValueError("El paso Δp tiene que ser menor que la presión (p − Δp > 0).")
    if dT >= 0.5 * T:
        raise ValueError("El paso ΔT es demasiado grande: tiene que ser menor que T/2.")
    if not (fl.T_min <= T - dT and T + dT <= fl.T_max):
        raise ValueError(
            f"T ± ΔT tiene que quedar dentro del rango de la ecuación de estado del {fl.noun}: de "
            f"{_num(fl.T_min, '.4g')} K a {_num(fl.T_max, '.4g')} K."
        )
    if p + dp > fl.p_max:
        raise ValueError(
            f"p + Δp supera la presión máxima de la ecuación de estado del {fl.noun} "
            f"({_num(fl.p_max / 1e6, '.4g')} MPa)."
        )
    state = CoolProp.AbstractState("HEOS", fl.key)
    if T < fl.T_cr:
        state.update(CoolProp.QT_INPUTS, 0.0, T)
        if abs(p / state.p() - 1.0) < 1e-6:
            raise ValueError(
                "Esa presión es justo la de saturación a esa temperatura: las relaciones de "
                "Maxwell se aplican en una sola fase. Corré un poco la presión o la temperatura "
                "(en la saturación, mirá Clapeyron)."
            )
    base = _update(state, "PT", p, T, "del dato")
    phase = _phase(state, fl)
    cp, cv = float(state.cpmass()), float(state.cvmass())
    alpha = float(state.isobaric_expansion_coefficient())
    kappa = float(state.isothermal_compressibility())
    mu = float(state.first_partial_deriv(CoolProp.iT, CoolProp.iP, CoolProp.iHmass))
    dv_dT = _exact(state, "v", "T", "p")
    mu_formula = (T * dv_dT - base.v) / cp
    try:
        paths = _paths(state, base, dT, dp)
    except ValueError as exc:
        if phase == "liquid" and "isócora" in str(exc):
            raise ValueError(
                f"{exc} En un líquido, a volumen constante la presión cambia "
                f"α/κ_T = {_num(alpha / kappa / 1e5, '.3g')} bar por kelvin: el paso ΔT tiene "
                "que ser chico."
            ) from exc
        raise
    maxwell = _maxwell(state, base, paths)
    natural = _natural(state, base, cp)
    convergence = [(1.0, tuple(m.gap_finite for m in maxwell))]
    for frac in (0.5, 0.25):
        sub = _paths(state, base, dT * frac, dp * frac)
        convergence.append(
            (frac, tuple(m.gap_finite for m in _maxwell(state, base, sub)))  # type: ignore[arg-type]
        )
    h_lo = _update(state, "PH", base.p - dp, base.h, "a p − Δp de la isoentálpica")
    h_hi = _update(state, "PH", base.p + dp, base.h, "a p + Δp de la isoentálpica")
    result = RelationsResult(
        inputs=inputs,
        fluid=fl,
        phase=phase,
        state=base,
        cp=cp,
        cv=cv,
        dT=dT,
        dp=dp,
        paths=paths,
        natural=natural,
        maxwell=maxwell,
        convergence=tuple(convergence),  # type: ignore[arg-type]
        alpha=alpha,
        kappa_T=kappa,
        mayer=T * base.v * alpha**2 / kappa,
        mu_JT=mu,
        mu_JT_formula=mu_formula,
        mu_JT_finite=(h_hi.T - h_lo.T) / (2.0 * dp),
        isenthalpic=(h_lo, h_hi),
        notes=(),
    )
    return replace(result, notes=_notes(result))


def _notes(r: RelationsResult) -> tuple[str, ...]:
    notes: list[str] = []
    if r.degenerate:
        notes.append(
            "Con α ≈ 0 los dos lados de las cuatro relaciones valen casi cero (todas dependen de "
            "la dilatación térmica): las diferencias relativas no dicen nada acá."
        )
        return _state_notes(r, notes)
    worst = max(abs(m.gap_finite) for m in r.maxwell)
    g_full, g_half = r.convergence[0][1][3], r.convergence[1][1][3]
    ratio = abs(g_full / g_half) if g_half else math.inf
    if 3.5 <= ratio <= 4.5:
        how = (
            f"baja {_num(ratio, '.2g')} veces: con diferencias centradas el error va con Δ², "
            "como en las tablas"
        )
    else:
        how = (
            f"baja {_num(ratio, '.2g')} veces, todavía lejos de las 4 de un error que va con Δ²: "
            "acá las propiedades cambian muy rápido (achicá los pasos)"
        )
    notes.append(
        "Con las derivadas exactas los dos lados de cada relación coinciden: la ecuación de "
        "estado es termodinámicamente consistente. Con diferencias finitas difieren hasta "
        f"{_sci(100.0 * worst, 2)} %; con los pasos a la mitad, la diferencia de (∂s/∂p)_T = "
        "−(∂v/∂T)_p "
        f"{how}."
    )
    return _state_notes(r, notes)


def _state_notes(r: RelationsResult, notes: list[str]) -> tuple[str, ...]:
    s = r.state
    ideal_alpha, ideal_kappa = r.alpha * s.T, r.kappa_T * s.p
    cpcv = r.cp - r.cv
    if r.phase == "liquid":
        notes.append(
            f"Es líquido: α·T = {_sci(ideal_alpha)} y κ_T·p = {_sci(ideal_kappa)} (1 en un gas "
            f"ideal). Como κ_T es tan chico, c_p − c_v = {_sci(cpcv)} J/(kg·K) es apenas "
            f"{_sci(100 * cpcv / r.cp, 2)} % de c_p: por eso en un líquido c_p ≈ c_v ≈ c "
            "(vademecum §13)."
        )
        if abs(r.alpha) < 2e-5:
            notes.append(
                "Con α ≈ 0 (el agua cerca de 4 °C) c_p = c_v: el agua no se dilata al calentarla."
            )
    else:
        R = r.fluid.R
        notes.append(
            f"Contra el gas ideal: α·T = {_num(ideal_alpha, '.4g')}, κ_T·p = "
            f"{_num(ideal_kappa, '.4g')} y (c_p − c_v)/R = {_num(cpcv / R, '.4g')} (los tres "
            "valen 1 en un gas ideal)."
        )
    if r.mu_JT > 0:
        notes.append(
            "μ_JT > 0: al estrangularlo (p baja a h constante) el fluido se enfría. Así se "
            "licúan los gases (Linde) y funciona la válvula de un refrigerador."
        )
    else:
        notes.append(
            "μ_JT < 0: al estrangularlo (p baja a h constante) el fluido se calienta; está "
            "fuera de la curva de inversión."
        )
    notes.append(
        "u, h, s, f y g dependen del estado de referencia del fluido (el de CoolProp); las "
        "derivadas y las diferencias, no."
    )
    return tuple(notes)


# ---------------------------------------------------------------------
# Clapeyron
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ClapeyronInputs:
    """La temperatura de saturación, el paso de la derivada y, para Clausius–Clapeyron, otra T."""

    fluid_key: str
    T_K: float
    dT_K: float = 2.0
    T2_K: float | None = None


@dataclass(frozen=True)
class ClapeyronResult:
    inputs: ClapeyronInputs
    fluid: RealFluid
    T: float
    p_sat: float
    v_f: float
    v_g: float
    h_fg: float  # CoolProp
    s_fg: float
    p_minus: float  # p_sat(T − ΔT)
    p_plus: float  # p_sat(T + ΔT)
    dpdT_exact: float
    dpdT_finite: float
    h_fg_clapeyron: float  # T·v_fg·(dp/dT) por diferencias
    h_fg_cc: float  # R·T²/p·(dp/dT): Clausius–Clapeyron (v_f ≈ 0, vapor ideal)
    T2: float | None
    p_sat2: float | None  # la real a T₂
    p_sat2_cc: float | None  # Clausius–Clapeyron con h_fg constante
    notes: tuple[str, ...]

    @property
    def v_fg(self) -> float:
        return self.v_g - self.v_f


def clapeyron(inputs: ClapeyronInputs) -> ClapeyronResult:
    """La ecuación de Clapeyron y la aproximación de Clausius–Clapeyron (Çengel §12-3).

    Raises
    ------
    ValueError
        Si T ± ΔT (o T₂) no está entre la temperatura mínima y la crítica.
    """
    fl = real_fluid(inputs.fluid_key)
    T, dT = inputs.T_K, inputs.dT_K
    if not (math.isfinite(dT) and dT > 0.0):
        raise ValueError("El paso ΔT tiene que ser positivo.")
    if not (math.isfinite(T) and fl.T_min <= T - dT and T + dT < fl.T_cr):
        raise ValueError(
            f"T ± ΔT tiene que quedar en la campana del {fl.noun}: entre "
            f"{_num(fl.T_min, '.4g')} K y la temperatura crítica ({_num(fl.T_cr, '.4g')} K)."
        )
    T2 = inputs.T2_K
    if T2 is not None and not (math.isfinite(T2) and fl.T_min <= T2 < fl.T_cr):
        raise ValueError(
            f"T₂ tiene que quedar en la campana del {fl.noun}: entre {_num(fl.T_min, '.4g')} K y "
            f"{_num(fl.T_cr, '.4g')} K."
        )
    state = CoolProp.AbstractState("HEOS", fl.key)

    def sat(TT: float, x: float) -> tuple[float, float, float, float]:
        state.update(CoolProp.QT_INPUTS, x, TT)
        return (
            float(state.p()),
            1.0 / float(state.rhomass()),
            float(state.hmass()),
            float(state.smass()),
        )

    p_sat, v_f, h_f, s_f = sat(T, 0.0)
    _, v_g, h_g, s_g = sat(T, 1.0)
    dpdT = float(state.first_saturation_deriv(CoolProp.iP, CoolProp.iT))
    p_minus = sat(T - dT, 0.0)[0]
    p_plus = sat(T + dT, 0.0)[0]
    dpdT_fd = (p_plus - p_minus) / (2.0 * dT)
    h_fg = h_g - h_f
    h_cc = fl.R * T**2 / p_sat * dpdT_fd
    p2 = p2_cc = None
    if T2 is not None:
        p2 = sat(T2, 0.0)[0]
        p2_cc = p_sat * math.exp(h_fg / fl.R * (1.0 / T - 1.0 / T2))
    result = ClapeyronResult(
        inputs=inputs,
        fluid=fl,
        T=T,
        p_sat=p_sat,
        v_f=v_f,
        v_g=v_g,
        h_fg=h_fg,
        s_fg=s_g - s_f,
        p_minus=p_minus,
        p_plus=p_plus,
        dpdT_exact=dpdT,
        dpdT_finite=dpdT_fd,
        h_fg_clapeyron=T * (v_g - v_f) * dpdT_fd,
        h_fg_cc=h_cc,
        T2=T2,
        p_sat2=p2,
        p_sat2_cc=p2_cc,
        notes=(),
    )
    return replace(result, notes=_clapeyron_notes(result))


def _clapeyron_notes(r: ClapeyronResult) -> tuple[str, ...]:
    err = _pct(r.h_fg_clapeyron / r.h_fg - 1, 2)
    notes = [
        f"Clapeyron con la derivada por diferencias da h_fg con {err} de error: la ecuación es "
        "exacta (sale de una relación de Maxwell), el error es solo de la derivada numérica.",
        f"Clausius–Clapeyron (v_f ≪ v_g y el vapor como gas ideal) se equivoca "
        f"{_pct(r.h_fg_cc / r.h_fg - 1)}: anda bien a presiones bajas y empeora cerca del punto "
        "crítico.",
    ]
    if r.p_sat2 is not None and r.p_sat2_cc is not None:
        err2 = _pct(r.p_sat2_cc / r.p_sat2 - 1)
        notes.append(
            f"Con h_fg constante, Clausius–Clapeyron da p_sat a T₂ con {err2} de error: sirve "
            "para extrapolar en intervalos chicos de temperatura."
        )
    return tuple(notes)


# ---------------------------------------------------------------------
# Joule–Thomson: la curva de inversión y las isoentálpicas
# ---------------------------------------------------------------------


def _mu(state: CoolProp.AbstractState, p: float, T: float) -> float:
    state.update(CoolProp.PT_INPUTS, p, T)
    return float(state.first_partial_deriv(CoolProp.iT, CoolProp.iP, CoolProp.iHmass))


def _inversion_temperatures(
    state: CoolProp.AbstractState, fl: RealFluid, p: float, Ts: np.ndarray
) -> tuple[list[float], list[float]]:
    """Las temperaturas de inversión a p: (rama baja, rama alta).

    Busca los cambios de signo de μ_JT(T) en una sola fase (el salto de la saturación no
    cuenta): en la rama baja μ pasa de − a + al subir T, en la alta de + a −.
    """
    T_sat = None
    if p < fl.p_cr:
        state.update(CoolProp.PQ_INPUTS, p, 0.0)
        T_sat = float(state.T())
    values = []
    for T in Ts:
        try:
            values.append(_mu(state, p, float(T)))
        except ValueError:
            values.append(math.nan)
    lower: list[float] = []
    upper: list[float] = []
    for i in range(len(Ts) - 1):
        a, b = values[i], values[i + 1]
        if not (math.isfinite(a) and math.isfinite(b)) or a * b > 0.0:
            continue
        if T_sat is not None and Ts[i] <= T_sat <= Ts[i + 1]:
            continue  # el salto de líquido a vapor no es una inversión
        try:
            T_inv = float(brentq(lambda T: _mu(state, p, T), Ts[i], Ts[i + 1], xtol=1e-6))
        except ValueError:
            continue
        (lower if a < 0.0 else upper).append(T_inv)
    return lower, upper


def _T_grid(fl: RealFluid) -> np.ndarray:
    return np.geomspace(fl.T_min * 1.01, fl.T_max * 0.99, 240)


def inversion_curve(fluid_key: str, points: int = 50) -> dict[str, list[float]]:
    """La curva de inversión de Joule–Thomson (μ_JT = 0) en el plano T–p (Çengel §12-5).

    Devuelve ``{"p": [...], "T": [...]}`` en orden: la rama baja de p chica a la nariz y
    la alta de vuelta.
    """
    fl = real_fluid(fluid_key)
    state = CoolProp.AbstractState("HEOS", fl.key)
    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    Ts = _T_grid(fl)
    for p in np.geomspace(0.01 * fl.p_cr, min(30.0 * fl.p_cr, fl.p_max), points):
        lo, hi = _inversion_temperatures(state, fl, float(p), Ts)
        lower += [(float(p), T) for T in lo]
        upper += [(float(p), T) for T in hi]
    curve = [*lower, *reversed(upper)]
    return {"p": [p for p, _ in curve], "T": [T for _, T in curve]}


def isenthalps(
    fluid_key: str, T_values: tuple[float, ...], p_ref: float, p_max: float, points: int = 60
) -> dict[float, tuple[list[float], list[float]]]:
    """Líneas de h constante en el plano T–p (las de una válvula), cada una con la h del
    estado (p_ref, T) de ``T_values``. Devuelve ``{T: (p, T(p, h))}``."""
    fl = real_fluid(fluid_key)
    state = CoolProp.AbstractState("HEOS", fl.key)
    out: dict[float, tuple[list[float], list[float]]] = {}
    for T0 in T_values:
        state.update(CoolProp.PT_INPUTS, p_ref, T0)
        h = float(state.hmass())
        ps, ts = [], []
        for p in np.linspace(p_ref, p_max, points):
            try:
                state.update(CoolProp.HmassP_INPUTS, h, float(p))
            except ValueError:
                continue
            if fl.T_min <= state.T() <= fl.T_max:
                ps.append(float(p))
                ts.append(float(state.T()))
        out[T0] = (ps, ts)
    return out


@dataclass(frozen=True)
class JouleThomsonInputs:
    """El fluido, el estado (p, T) y el paso Δp de la diferencia a h constante (SI)."""

    fluid_key: str
    p_Pa: float
    T_K: float
    dp_Pa: float | None = None  # None: 10 % de p


@dataclass(frozen=True)
class JouleThomsonResult:
    inputs: JouleThomsonInputs
    fluid: RealFluid
    phase: Literal["liquid", "vapor", "supercritical"]
    state: PathState
    cp: float
    dv_dT: float  # (∂v/∂T)_p
    dp: float
    mu: float  # exacto (CoolProp)
    mu_formula: float  # [T·(∂v/∂T)_p − v]/c_p
    mu_finite: float  # (T₊ − T₋)/(2Δp) a h constante
    isenthalpic: tuple[PathState, PathState]
    T_inversion: tuple[float, ...]  # las temperaturas de inversión a esta presión
    notes: tuple[str, ...]


def joule_thomson(inputs: JouleThomsonInputs) -> JouleThomsonResult:
    """El coeficiente de Joule–Thomson μ_JT = (∂T/∂p)_h = [T·(∂v/∂T)_p − v]/c_p en (p, T)
    (Çengel §12-5), exacto, con la fórmula y por diferencias a h constante, y las
    temperaturas de inversión a esa presión.

    Raises
    ------
    ValueError
        Con datos no positivos, fuera del rango de la ecuación de estado o en la campana.
    """
    fl = real_fluid(inputs.fluid_key)
    p, T = inputs.p_Pa, inputs.T_K
    dp = inputs.dp_Pa if inputs.dp_Pa is not None else 0.1 * p
    for msg, value in (
        ("La presión (absoluta) tiene que ser positiva", p),
        ("La temperatura (absoluta) tiene que ser positiva", T),
        ("El paso Δp tiene que ser positivo", dp),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{msg}.")
    if dp >= p:
        raise ValueError("El paso Δp tiene que ser menor que la presión (p − Δp > 0).")
    if not (fl.T_min <= T <= fl.T_max):
        raise ValueError(
            f"La temperatura está fuera del rango de la ecuación de estado del {fl.noun}: de "
            f"{_num(fl.T_min, '.4g')} K a {_num(fl.T_max, '.4g')} K."
        )
    if p + dp > fl.p_max:
        raise ValueError(
            f"p + Δp supera la presión máxima de la ecuación de estado del {fl.noun} "
            f"({_num(fl.p_max / 1e6, '.4g')} MPa)."
        )
    state = CoolProp.AbstractState("HEOS", fl.key)
    if T < fl.T_cr:
        state.update(CoolProp.QT_INPUTS, 0.0, T)
        if abs(p / state.p() - 1.0) < 1e-6:
            raise ValueError(
                "Esa presión es justo la de saturación a esa temperatura: μ_JT es de una sola "
                "fase. Corré un poco la presión o la temperatura."
            )
    base = _update(state, "PT", p, T, "del dato")
    phase = _phase(state, fl)
    cp = float(state.cpmass())
    mu = float(state.first_partial_deriv(CoolProp.iT, CoolProp.iP, CoolProp.iHmass))
    dv_dT = _exact(state, "v", "T", "p")
    lo = _update(state, "PH", p - dp, base.h, "a p − Δp de la isoentálpica")
    hi = _update(state, "PH", p + dp, base.h, "a p + Δp de la isoentálpica")
    low, up = _inversion_temperatures(state, fl, p, _T_grid(fl))
    result = JouleThomsonResult(
        inputs=inputs,
        fluid=fl,
        phase=phase,
        state=base,
        cp=cp,
        dv_dT=dv_dT,
        dp=dp,
        mu=mu,
        mu_formula=(T * dv_dT - base.v) / cp,
        mu_finite=(hi.T - lo.T) / (2.0 * dp),
        isenthalpic=(lo, hi),
        T_inversion=tuple(sorted(low + up)),
        notes=(),
    )
    return replace(result, notes=_jt_notes(result))


def _jt_notes(r: JouleThomsonResult) -> tuple[str, ...]:
    notes: list[str] = []
    fl = r.fluid
    K_bar = r.mu * 1e5
    if r.mu > 0:
        notes.append(
            f"μ_JT > 0: al estrangularlo (la presión baja a h constante) el {fl.noun} se enfría, "
            f"unos {_num(abs(K_bar), '.2g')} K por bar. Así se licúan los gases (Linde)."
        )
    else:
        notes.append(
            f"μ_JT < 0: al estrangularlo el {fl.noun} se calienta, unos "
            f"{_num(abs(K_bar), '.2g')} K por bar: está fuera de la curva de inversión."
        )
    if len(r.T_inversion) == 2:
        lo, hi = (f"{_num(T - 273.15, '.4g')} °C" for T in r.T_inversion)
        notes.append(
            f"A esta presión el {fl.noun} se enfría al estrangularlo solo entre {lo} y {hi} (las "
            "temperaturas de inversión): adentro de la curva de inversión."
        )
    elif len(r.T_inversion) == 1:
        notes.append(
            f"A esta presión la temperatura de inversión es "
            f"{_num(r.T_inversion[0] - 273.15, '.4g')} °C."
        )
    else:
        notes.append(
            "A esta presión no hay temperatura de inversión en el rango de la ecuación de "
            "estado: está por encima de la nariz de la curva."
        )
    notes.append(
        "Un gas ideal tiene μ_JT = 0: su h depende solo de T, así que una válvula no le cambia "
        "la temperatura. El enfriamiento de un gas real mide cuánto se aparta del ideal."
    )
    return tuple(notes)


def clausius_curve(r: ClapeyronResult, points: int = 60) -> dict[str, list[float]]:
    """p_sat(T) real (CoolProp) y la curva de Clausius–Clapeyron por (T₁, p₁) con h_fg(T₁)
    constante, de la T mínima (o 0,45·T_cr) a la crítica: ``{"T", "real", "cc"}``."""
    fl = r.fluid
    state = CoolProp.AbstractState("HEOS", fl.key)
    Ts = np.linspace(max(fl.T_min * 1.001, 0.45 * fl.T_cr), 0.999 * fl.T_cr, points)
    real, cc = [], []
    for T in Ts:
        state.update(CoolProp.QT_INPUTS, 0.0, float(T))
        real.append(float(state.p()))
        cc.append(r.p_sat * math.exp(r.h_fg / fl.R * (1.0 / r.T - 1.0 / float(T))))
    return {"T": [float(T) for T in Ts], "real": real, "cc": cc}


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class JouleThomsonExample:
    inputs: JouleThomsonInputs
    note: str = ""


@dataclass(frozen=True)
class RelationsExample:
    inputs: RelationsInputs
    note: str = ""


@dataclass(frozen=True)
class ClapeyronExample:
    inputs: ClapeyronInputs
    note: str = ""


C = 273.15

RELATIONS_EXAMPLES: dict[str, RelationsExample] = {
    "Vapor de agua a 250 °C y 300 kPa (Çengel §12-2)": RelationsExample(
        RelationsInputs("Water", 300e3, 250.0 + C, 50.0, 100e3),
        "Çengel, con las tablas: (∂s/∂p)_T ≈ −0,00165 m³/(kg·K) y −(∂v/∂T)_p ≈ −0,00159: "
        "difieren 3,7 % por los pasos grandes (ΔT = 50 °C, Δp = 100 kPa).",
    ),
    "Nitrógeno a 300 K y 1 bar (casi ideal)": RelationsExample(
        RelationsInputs("Nitrogen", 1e5, 300.0, 10.0, 0.1e5),
        "α·T, κ_T·p y (c_p − c_v)/R valen casi 1.",
    ),
    "R-134a a 1 MPa y 50 °C": RelationsExample(
        RelationsInputs("R134a", 1e6, 50.0 + C, 5.0, 0.5e5),
        "Vapor sobrecalentado cerca de la saturación: lejos del gas ideal.",
    ),
    "Dióxido de carbono a 80 bar y 35 °C": RelationsExample(
        RelationsInputs("CarbonDioxide", 80e5, 35.0 + C, 2.0, 2e5),
        "Cerca del punto crítico α y κ_T se disparan.",
    ),
    "Agua líquida a 100 bar y 20 °C": RelationsExample(
        RelationsInputs("Water", 100e5, 20.0 + C, 1.0, 10e5),
        "Un líquido: c_p − c_v es menos del 1 % de c_p. A volumen constante la presión sube "
        "4,9 bar por kelvin: por eso ΔT = 1 °C.",
    ),
}

CLAPEYRON_EXAMPLES: dict[str, ClapeyronExample] = {
    "R-134a a 20 °C (Çengel §12-3)": ClapeyronExample(
        ClapeyronInputs("R134a", 20.0 + C, 4.0, 40.0 + C),
        "Çengel estima la pendiente con las tablas a 16 y 24 °C: h_fg = 182,40 kJ/kg (tabla: "
        "182,27).",
    ),
    "Agua a 100 °C": ClapeyronExample(
        ClapeyronInputs("Water", 100.0 + C, 5.0, 120.0 + C),
        "h_fg = 2256,4 kJ/kg (tablas).",
    ),
    "Nitrógeno a 100 K (cerca del punto crítico, 126 K)": ClapeyronExample(
        ClapeyronInputs("Nitrogen", 100.0, 1.0, 90.0),
        "Clausius–Clapeyron empeora: el vapor ya no es un gas ideal.",
    ),
}


JT_EXAMPLES: dict[str, JouleThomsonExample] = {
    "Nitrógeno a 300 K y 50 bar (se enfría)": JouleThomsonExample(
        JouleThomsonInputs("Nitrogen", 50e5, 300.0, 5e5),
        "Por debajo de su máxima temperatura de inversión (≈ 608 K): se enfría.",
    ),
    "Hidrógeno a 300 K y 50 bar (se calienta)": JouleThomsonExample(
        JouleThomsonInputs("Hydrogen", 50e5, 300.0, 5e5),
        "Su máxima temperatura de inversión ronda los 200 K: a temperatura ambiente el "
        "hidrógeno se calienta en una válvula. Hay que preenfriarlo para licuarlo.",
    ),
    "Aire a 20 °C y 200 bar (un tubo de buceo)": JouleThomsonExample(
        JouleThomsonInputs("Air", 200e5, 20.0 + C, 10e5),
        "El regulador de un tubo se enfría al bajar la presión.",
    ),
    "Dióxido de carbono a 40 °C y 60 bar": JouleThomsonExample(
        JouleThomsonInputs("CarbonDioxide", 60e5, 40.0 + C, 5e5),
        "Cerca del punto crítico μ_JT es grande: el CO₂ de un matafuego sale helado.",
    ),
    "Helio a 300 K y 10 bar": JouleThomsonExample(
        JouleThomsonInputs("Helium", 10e5, 300.0, 1e5),
        "Su máxima temperatura de inversión es ≈ 45 K: a temperatura ambiente se calienta.",
    ),
}


def joule_thomson_to_dict(r: JouleThomsonResult) -> dict[str, Any]:
    return {
        "fluido": r.fluid.name,
        "fase": r.phase,
        "p_Pa": r.state.p,
        "T_K": r.state.T,
        "v": r.state.v,
        "c_p": r.cp,
        "dv_dT_p": r.dv_dT,
        "dp_Pa": r.dp,
        "mu_JT": r.mu,
        "mu_JT_formula": r.mu_formula,
        "mu_JT_diferencias": r.mu_finite,
        "T_inversion_K": list(r.T_inversion),
        "notas": list(r.notes),
    }


def relations_to_dict(r: RelationsResult) -> dict[str, Any]:
    """El resultado para exportar (SI)."""
    s = r.state
    return {
        "fluido": r.fluid.name,
        "fase": r.phase,
        "estado": {
            "T_K": s.T,
            "p_Pa": s.p,
            "v": s.v,
            "u": s.u,
            "h": s.h,
            "s": s.s,
            "f": s.f,
            "g": s.g,
        },
        "c_p": r.cp,
        "c_v": r.cv,
        "dT_K": r.dT,
        "dp_Pa": r.dp,
        "caminos": {
            k: [{"T_K": x.T, "p_Pa": x.p, "v": x.v, "s": x.s} for x in pair]
            for k, pair in r.paths.items()
        },
        "maxwell": {
            m.potential: {
                m.left.key: {"exacta": m.left.exact, "diferencias": m.left.finite},
                m.right.key: {"exacta": m.right.exact, "diferencias": m.right.finite},
                "signo": m.sign,
                "brecha_exacta": m.gap_exact,
                "brecha_diferencias": m.gap_finite,
            }
            for m in r.maxwell
        },
        "derivadas_naturales": {
            n.key: {"valor": n.value, "esperado": n.expected} for n in r.natural
        },
        "alpha": r.alpha,
        "kappa_T": r.kappa_T,
        "c_p_menos_c_v": r.cp - r.cv,
        "mayer": r.mayer,
        "mu_JT": r.mu_JT,
        "mu_JT_formula": r.mu_JT_formula,
        "mu_JT_diferencias": r.mu_JT_finite,
        "notas": list(r.notes),
    }


def clapeyron_to_dict(r: ClapeyronResult) -> dict[str, Any]:
    return {
        "fluido": r.fluid.name,
        "T_K": r.T,
        "p_sat_Pa": r.p_sat,
        "v_f": r.v_f,
        "v_g": r.v_g,
        "h_fg": r.h_fg,
        "s_fg": r.s_fg,
        "dpdT_exacta": r.dpdT_exact,
        "dpdT_diferencias": r.dpdT_finite,
        "h_fg_clapeyron": r.h_fg_clapeyron,
        "h_fg_clausius_clapeyron": r.h_fg_cc,
        "T2_K": r.T2,
        "p_sat2_Pa": r.p_sat2,
        "p_sat2_clausius_clapeyron_Pa": r.p_sat2_cc,
        "notas": list(r.notes),
    }
