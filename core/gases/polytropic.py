"""Transformaciones politrópicas de un gas ideal (vademecum §3.5, §6 y §10.5).

Cinco procesos reversibles desde (p₁, T₁): isócora (n = ±∞), isóbara (n = 0),
isoterma (n = 1), adiabática (n = k con c_p constante) y politrópica p·vⁿ = cte.
El estado final se da con p₂, con la relación de compresión v₁/v₂ o con T₂.

Con la notación del vademecum, w = ∫p dv es el trabajo de expansión (sistema
cerrado) y w_f = −∫v dp el de circulación (sistema abierto en régimen
permanente); los dos son positivos si los hace el gas, así que una compresión
tiene w_f < 0. En la politrópica, T₂/T₁ = (p₂/p₁)^((n−1)/n), w = R·(T₂ − T₁)/(1 − n),
w_f = n·w y q = c·(T₂ − T₁) con c = c_v·(n − k)/(n − 1) (§6.2 a §6.4).

Con c_p variable (polinomios NASA) p·vⁿ = cte sigue fijando T₂, Δu sale del
polinomio y q = Δu + w; la adiabática reversible ya no es p·v^k = cte: T₂ sale de
s°(T₂) = s°(T₁) + R·ln(p₂/p₁), la presión relativa del vademecum §10.5.3.

También: la comparación de los caminos que llegan al mismo dato final, la
compresión en etapas con interenfriamiento (§6.3.1) y el n de dos estados
medidos (§6.1). Todo en SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq

from core.gases.ideal import (
    T_REF_K,
    GasModel,
    GasState,
    IdealGas,
    gas,
    gas_state,
    model_cp,
    model_h,
    model_s,
    model_u,
)

__all__ = [
    "EXPONENT_EXAMPLE",
    "FINAL_SPECS",
    "PROCESS_EXAMPLES",
    "PROCESS_KINDS",
    "ComparisonRow",
    "ExponentResult",
    "FinalSpec",
    "PolytropicInputs",
    "ProcessExample",
    "ProcessKind",
    "ProcessResult",
    "StagedInputs",
    "StagedResult",
    "SystemKind",
    "allowed_finals",
    "exponent_from_states",
    "exponent_to_dict",
    "process_comparison",
    "process_curve",
    "process_to_dict",
    "solve_process",
    "staged_compression",
    "staged_curves",
    "staged_to_dict",
]

ProcessKind = Literal["isochoric", "isobaric", "isothermal", "adiabatic", "polytropic"]
PROCESS_KINDS: dict[ProcessKind, str] = {
    "isochoric": "Isócora (v = cte)",
    "isobaric": "Isóbara (p = cte)",
    "isothermal": "Isoterma (T = cte)",
    "adiabatic": "Adiabática reversible (q = 0)",
    "polytropic": "Politrópica (p·vⁿ = cte)",
}
FinalSpec = Literal["p2", "ratio", "T2"]
FINAL_SPECS: dict[FinalSpec, str] = {
    "p2": "La presión final p₂",
    "ratio": "La relación de compresión v₁/v₂",
    "T2": "La temperatura final T₂",
}
SystemKind = Literal["closed", "open"]

#: Qué dato final sirve en cada proceso (el resto lo fija el proceso).
_ALLOWED: dict[ProcessKind, tuple[FinalSpec, ...]] = {
    "isochoric": ("p2", "T2"),
    "isobaric": ("ratio", "T2"),
    "isothermal": ("p2", "ratio"),
    "adiabatic": ("p2", "ratio", "T2"),
    "polytropic": ("p2", "ratio", "T2"),
}
_WHY_NOT: dict[tuple[ProcessKind, FinalSpec], str] = {
    ("isochoric", "ratio"): "en una isócora v₂ = v₁ (la relación de compresión es 1)",
    ("isobaric", "p2"): "en una isóbara p₂ = p₁",
    ("isothermal", "T2"): "en una isoterma T₂ = T₁",
}
#: El n de la politrópica de la comparación si el proceso del dato es otro (Çengel §7-12).
DEFAULT_N: float = 1.3


def allowed_finals(process: ProcessKind) -> tuple[FinalSpec, ...]:
    """Los datos finales que sirven para un proceso."""
    return _ALLOWED[process]


def _num(x: float, fmt: str = ".3g") -> str:
    return f"{x:{fmt}}".replace(".", ",")


def _pct(x: float, decimals: int = 1) -> str:
    return f"{100.0 * x:.{decimals}f} %".replace(".", ",")


# ---------------------------------------------------------------------
# Un proceso
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PolytropicInputs:
    """El gas, el estado inicial, el proceso y su dato final.

    ``amount`` es la masa (kg) en un sistema cerrado o el caudal (kg/s) en uno
    abierto; ``n`` solo se usa en la politrópica.
    """

    gas_key: str
    p1_Pa: float
    T1_K: float
    process: ProcessKind
    final: FinalSpec
    final_value: float
    n: float = DEFAULT_N
    model: GasModel = "constant"
    system: SystemKind = "closed"
    amount: float = 1.0


@dataclass(frozen=True)
class ProcessResult:
    """El proceso resuelto, por kg y en total."""

    inputs: PolytropicInputs
    gas: IdealGas
    state1: GasState
    state2: GasState
    n: float
    k: float
    c: float | None
    du: float
    dh: float
    ds: float
    w: float
    w_f: float
    q: float
    W_total: float
    Q_total: float
    S_total: float
    other_T2_K: float | None
    other_w: float | None
    other_w_f: float | None
    other_q: float | None
    notes: tuple[str, ...]

    @property
    def work(self) -> float:
        """El trabajo que importa: w (cerrado) o w_f (abierto), por kg."""
        return self.w if self.inputs.system == "closed" else self.w_f


def _check_inputs(inputs: PolytropicInputs) -> IdealGas:
    g = gas(inputs.gas_key)
    if inputs.final not in _ALLOWED[inputs.process]:
        why = _WHY_NOT.get((inputs.process, inputs.final), "")
        options = ", ".join(FINAL_SPECS[f].lower() for f in _ALLOWED[inputs.process])
        raise ValueError(
            f"Con una {PROCESS_KINDS[inputs.process].split(' (')[0].lower()} no se puede dar "
            f"{FINAL_SPECS[inputs.final].lower()} ({why}): dá {options}."
        )
    if not (math.isfinite(inputs.final_value) and inputs.final_value > 0.0):
        raise ValueError(f"{FINAL_SPECS[inputs.final]} tiene que ser positiva.")
    if inputs.process == "polytropic":
        n = inputs.n
        if not math.isfinite(n) or abs(n) > 20.0:
            raise ValueError("El exponente politrópico n tiene que ser un número entre −20 y 20.")
        if abs(n - 1.0) < 1e-6:
            raise ValueError("Con n = 1 es una isoterma: elegí ese proceso.")
        if abs(n) < 1e-6:
            raise ValueError("Con n = 0 es una isóbara: elegí ese proceso.")
    if not (math.isfinite(inputs.amount) and inputs.amount > 0.0):
        unit = "masa" if inputs.system == "closed" else "caudal"
        raise ValueError(f"La {unit} tiene que ser positiva.")
    return g


def _final_state(inputs: PolytropicInputs, g: IdealGas, s1: GasState) -> tuple[float, float]:
    """(p₂, T₂) según el proceso y el dato final."""
    p1, T1, value = s1.p_Pa, s1.T_K, inputs.final_value
    kind, final, model = inputs.process, inputs.final, inputs.model
    if kind == "isochoric":
        return (value, T1 * value / p1) if final == "p2" else (p1 * value / T1, value)
    if kind == "isobaric":
        return (p1, T1 / value) if final == "ratio" else (p1, value)
    if kind == "isothermal":
        return (value, T1) if final == "p2" else (p1 * value, T1)
    if kind == "polytropic" or (kind == "adiabatic" and model == "constant"):
        n = inputs.n if kind == "polytropic" else g.k(T_REF_K)
        if final == "p2":
            return value, T1 * (value / p1) ** ((n - 1.0) / n)
        if final == "ratio":
            return p1 * value**n, T1 * value ** (n - 1.0)
        return p1 * (value / T1) ** (n / (n - 1.0)), value
    # adiabática con c_p variable: s°(T₂) = s°(T₁) + R·ln(p₂/p₁)
    R = g.R
    s01 = g.s0(T1)
    lo, hi = g.T_min_K, g.T_max_K
    if final == "T2":
        return p1 * math.exp((g.s0(value) - s01) / R), value
    if final == "p2":
        target = s01 + R * math.log(value / p1)
        f = lambda T: g.s0(T) - target  # noqa: E731
        if f(lo) * f(hi) > 0:
            raise ValueError(
                "La temperatura final de esa adiabática queda fuera del rango de los polinomios "
                "NASA: probá con una relación de presiones más chica."
            )
        return value, brentq(f, lo, hi, xtol=1e-10)
    # relación de compresión: p₂/p₁ = (v₁/v₂)·(T₂/T₁)
    target = s01 - R * math.log(T1) + R * math.log(value)
    f = lambda T: g.s0(T) - R * math.log(T) - target  # noqa: E731
    if f(lo) * f(hi) > 0:
        raise ValueError(
            "La temperatura final de esa adiabática queda fuera del rango de los polinomios NASA: "
            "probá con una relación de compresión más chica."
        )
    T2 = brentq(f, lo, hi, xtol=1e-10)
    return p1 * value * T2 / T1, T2


def _numbers(inputs: PolytropicInputs, g: IdealGas, s1: GasState, s2: GasState) -> dict[str, float]:
    model = inputs.model
    T1, T2, p1, p2 = s1.T_K, s2.T_K, s1.p_Pa, s2.p_Pa
    R = g.R
    du = model_u(g, T2, model) - model_u(g, T1, model)
    dh = model_h(g, T2, model) - model_h(g, T1, model)
    ds = model_s(g, T2, p2, model) - model_s(g, T1, p1, model)
    kind = inputs.process
    if kind == "isochoric":
        w, w_f = 0.0, s1.v_m3_per_kg * (p1 - p2)
    elif kind == "isobaric":
        w, w_f = R * (T2 - T1), 0.0
    elif kind == "isothermal":
        w = R * T1 * math.log(s2.v_m3_per_kg / s1.v_m3_per_kg)
        w_f = w
    elif kind == "adiabatic":
        w, w_f = -du, -dh
        ds = 0.0  # reversible: lo que da la cuenta es redondeo (~1e-15)
    else:
        n = inputs.n
        w = R * (T2 - T1) / (1.0 - n)
        w_f = n * w
    q = du + w
    if kind == "adiabatic":
        q = 0.0
    return {"du": du, "dh": dh, "ds": ds, "w": w, "w_f": w_f, "q": q}


def solve_process(inputs: PolytropicInputs, *, compare_model: bool = True) -> ProcessResult:
    """Resuelve el proceso (vademecum §6; con c_p variable, §10.5).

    Raises
    ------
    ValueError
        Si el dato final no sirve para ese proceso, si n no tiene sentido, si
        algún valor no es positivo o si T₂ queda fuera del rango de los
        polinomios.
    """
    g = _check_inputs(inputs)
    s1 = gas_state(g, p_Pa=inputs.p1_Pa, T_K=inputs.T1_K, label="1")
    p2, T2 = _final_state(inputs, g, s1)
    s2 = gas_state(g, p_Pa=p2, T_K=T2, label="2")
    nums = _numbers(inputs, g, s1, s2)
    k = g.k(T_REF_K)
    kind = inputs.process
    if kind == "isochoric":
        n = math.inf
    elif kind == "isobaric":
        n = 0.0
    elif kind == "isothermal":
        n = 1.0
    elif kind == "polytropic":
        n = inputs.n
    elif inputs.model == "constant":
        n = k
    else:
        ratio = s1.v_m3_per_kg / s2.v_m3_per_kg
        n = math.log(s2.p_Pa / s1.p_Pa) / math.log(ratio) if abs(ratio - 1.0) > 1e-12 else k
    dT = s2.T_K - s1.T_K
    c = nums["q"] / dT if abs(dT) > 1e-9 else None

    other: ProcessResult | None = None
    if compare_model and not g.monatomic:
        other_model: GasModel = "variable" if inputs.model == "constant" else "constant"
        try:
            other = solve_process(replace(inputs, model=other_model), compare_model=False)
        except ValueError:
            other = None

    notes = _process_notes(inputs, g, s1, s2, nums, n, k, other)
    amount = inputs.amount
    work = nums["w"] if inputs.system == "closed" else nums["w_f"]
    return ProcessResult(
        inputs=inputs,
        gas=g,
        state1=s1,
        state2=s2,
        n=n,
        k=k,
        c=c,
        du=nums["du"],
        dh=nums["dh"],
        ds=nums["ds"],
        w=nums["w"],
        w_f=nums["w_f"],
        q=nums["q"],
        W_total=amount * work,
        Q_total=amount * nums["q"],
        S_total=amount * nums["ds"],
        other_T2_K=other.state2.T_K if other else None,
        other_w=other.w if other else None,
        other_w_f=other.w_f if other else None,
        other_q=other.q if other else None,
        notes=tuple(notes),
    )


def _process_notes(
    inputs: PolytropicInputs,
    g: IdealGas,
    s1: GasState,
    s2: GasState,
    nums: dict[str, float],
    n: float,
    k: float,
    other: ProcessResult | None,
) -> list[str]:
    notes: list[str] = []
    kind = inputs.process
    compressing = s2.p_Pa > s1.p_Pa * (1 + 1e-12)
    if kind == "polytropic" and compressing and 1.0 < n < k:
        notes.append(
            f"Con 1 < n < k ({_num(n)} < {_num(k, '.4f')}) el gas cede calor mientras se comprime "
            "(q < 0): es un compresor refrigerado, a mitad de camino entre la isoterma y la "
            "adiabática."
        )
    if kind == "polytropic" and not compressing and 1.0 < n < k:
        notes.append(
            f"Con 1 < n < k ({_num(n)} < {_num(k, '.4f')}) el gas recibe calor mientras se expande "
            "(q > 0)."
        )
    if kind == "isothermal":
        notes.append(
            "En la isoterma Δu = Δh = 0 (u y h dependen solo de T): todo el trabajo sale o entra "
            "como calor, q = w = w_f."
        )
    if kind == "adiabatic" and inputs.model == "variable" and not g.monatomic:
        notes.append(
            f"Con c_p variable la adiabática no es p·v^k = cte: T₂ sale de s°(T₂) = s°(T₁) + "
            f"R·ln(p₂/p₁) y el exponente equivalente es n = {_num(n, '.4f')} (con c_p a 25 °C, "
            f"k = {_num(k, '.4f')})."
        )
    if other is not None and abs(s2.T_K - s1.T_K) > 150.0:
        label = "c_p variable" if inputs.model == "constant" else "c_p constante a 25 °C"
        mine = nums["w"] if inputs.system == "closed" else nums["w_f"]
        theirs = other.w if inputs.system == "closed" else other.w_f
        if abs(other.state2.T_K - s2.T_K) > 0.05:
            if abs(mine) > 1e-9:
                notes.append(
                    f"La temperatura cambia {abs(s2.T_K - s1.T_K):.0f} K: con {label}, T₂ da "
                    f"{_num(other.state2.T_K, '.1f')} K y el trabajo cambia "
                    f"{_pct(abs(theirs / mine - 1.0))}."
                )
        elif abs(nums["q"]) > 1e-9 and abs(other.q / nums["q"] - 1.0) > 1e-3:
            notes.append(
                f"La temperatura cambia {abs(s2.T_K - s1.T_K):.0f} K. El proceso fija T₂ y el "
                f"trabajo, que no dependen de c_p; con {label} cambia el calor: "
                f"{_pct(abs(other.q / nums['q'] - 1.0))} (porque cambia Δu)."
            )
    return notes


# ---------------------------------------------------------------------
# Los caminos que llegan al mismo dato final
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRow:
    """Un proceso de la comparación: su resultado o por qué no llega."""

    process: ProcessKind
    result: ProcessResult | None
    reason: str
    is_data: bool


def process_comparison(inputs: PolytropicInputs) -> tuple[ComparisonRow, ...]:
    """Los cinco procesos desde el mismo estado inicial hasta el mismo dato final.

    El que no puede llegar a ese dato (la isóbara no cambia p, la isoterma no
    cambia T, la isócora no cambia v) va sin resultado y con el motivo. La
    politrópica usa el n del dato o, si el proceso es otro, n = 1,3.
    """
    rows: list[ComparisonRow] = []
    n = inputs.n if inputs.process == "polytropic" else DEFAULT_N
    for kind in PROCESS_KINDS:
        is_data = kind == inputs.process
        if inputs.final not in _ALLOWED[kind]:
            reason = _WHY_NOT.get((kind, inputs.final), "no llega a ese dato")
            rows.append(ComparisonRow(kind, None, reason[0].upper() + reason[1:] + ".", is_data))
            continue
        try:
            result = solve_process(replace(inputs, process=kind, n=n), compare_model=False)
        except ValueError as exc:
            rows.append(ComparisonRow(kind, None, str(exc), is_data))
            continue
        rows.append(ComparisonRow(kind, result, "", is_data))
    return tuple(rows)


# ---------------------------------------------------------------------
# Curvas para los diagramas p–v y T–s
# ---------------------------------------------------------------------


def process_curve(result: ProcessResult, points: int = 60) -> dict[str, list[float]]:
    """El camino del proceso: v, p, T y s (con el modelo del proceso)."""
    g, model = result.gas, result.inputs.model
    s1, s2 = result.state1, result.state2
    kind = result.inputs.process
    R = g.R
    if kind in ("isochoric", "isobaric") or (kind == "adiabatic" and model == "variable"):
        Ts = np.linspace(s1.T_K, s2.T_K, points)
        if kind == "isochoric":
            vs = np.full(points, s1.v_m3_per_kg)
            ps = R * Ts / vs
        elif kind == "isobaric":
            ps = np.full(points, s1.p_Pa)
            vs = R * Ts / ps
        else:
            s01 = g.s0(s1.T_K)
            ps = np.array([s1.p_Pa * math.exp((g.s0(T) - s01) / R) for T in Ts])
            vs = R * Ts / ps
    else:
        vs = np.geomspace(s1.v_m3_per_kg, s2.v_m3_per_kg, points)
        if kind == "isothermal":
            ps = s1.p_Pa * s1.v_m3_per_kg / vs
        else:
            n = result.n
            ps = s1.p_Pa * (s1.v_m3_per_kg / vs) ** n
        Ts = ps * vs / R
    ss = [model_s(g, float(T), float(p), model) for T, p in zip(Ts, ps, strict=True)]
    return {
        "v": [float(v) for v in vs],
        "p": [float(p) for p in ps],
        "T": [float(T) for T in Ts],
        "s": ss,
    }


# ---------------------------------------------------------------------
# Compresión en etapas y n de dos estados
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class StagedInputs:
    """Una compresión politrópica de p₁ a p₂ en N etapas con interenfriamiento perfecto."""

    gas_key: str
    p1_Pa: float
    T1_K: float
    p2_Pa: float
    n: float
    stages: int = 2


@dataclass(frozen=True)
class StagedResult:
    """La compresión en etapas contra una sola etapa (por kg, c_p constante)."""

    inputs: StagedInputs
    gas: IdealGas
    pressures_Pa: tuple[float, ...]
    stage_ratio: float
    T_out_K: float
    w_f: float
    w_f_single: float
    T_out_single_K: float
    q_intercoolers: float
    saving: float


def staged_compression(inputs: StagedInputs) -> StagedResult:
    """Compresión en N etapas con interenfriamiento perfecto (vademecum §6.3.1).

    La misma relación en cada etapa, r = (p₂/p₁)^(1/N), es la de mínimo trabajo
    (p_x = √(p₁·p₂) con dos etapas) y cada etapa vuelve a T₁:
    w_f = −N·n·R·T₁/(n − 1)·[r^((n−1)/n) − 1].

    Raises
    ------
    ValueError
        Si p₂ ≤ p₁, si N no está entre 1 y 4 o si n < 1.
    """
    g = gas(inputs.gas_key)
    s1 = gas_state(g, p_Pa=inputs.p1_Pa, T_K=inputs.T1_K, label="1")
    if not (math.isfinite(inputs.p2_Pa) and inputs.p2_Pa > s1.p_Pa):
        raise ValueError("Para comprimir en etapas, p₂ tiene que ser mayor que p₁.")
    if not (1 <= inputs.stages <= 4):
        raise ValueError("La cantidad de etapas va de 1 a 4.")
    n = inputs.n
    if not (math.isfinite(n) and 1.0 <= n <= 3.0):
        raise ValueError(
            "El exponente de un compresor va de 1 (isoterma) a algo más que k (adiabática "
            "con fricción): poné un n entre 1 y 3."
        )
    N = inputs.stages
    R, T1, ratio = g.R, s1.T_K, inputs.p2_Pa / s1.p_Pa
    r = ratio ** (1.0 / N)

    def work(rr: float, stages: int) -> float:
        if abs(n - 1.0) < 1e-9:
            return -stages * R * T1 * math.log(rr)
        return -stages * n * R * T1 / (n - 1.0) * (rr ** ((n - 1.0) / n) - 1.0)

    T_out = T1 * r ** ((n - 1.0) / n)
    T_out_single = T1 * ratio ** ((n - 1.0) / n)
    for T in (T_out, T_out_single):
        g.check_T(T, "La temperatura a la salida de la compresión")
    w_f = work(r, N)
    w_single = work(ratio, 1)
    q_ic = -(N - 1) * model_cp(g, T1, "constant") * (T_out - T1)
    return StagedResult(
        inputs=inputs,
        gas=g,
        pressures_Pa=tuple(s1.p_Pa * r**j for j in range(N + 1)),
        stage_ratio=r,
        T_out_K=T_out,
        w_f=w_f,
        w_f_single=w_single,
        T_out_single_K=T_out_single,
        q_intercoolers=q_ic,
        saving=1.0 - w_f / w_single if w_single != 0.0 else 0.0,
    )


def staged_curves(result: StagedResult, points: int = 40) -> dict[str, dict[str, list[float]]]:
    """Los caminos del p–v de la compresión en etapas (Çengel §7-12, figura del interenfriamiento).

    - ``"stages"``: cada etapa (p·vⁿ = cte desde T₁) seguida del interenfriador (a p
      constante, de T_x a T₁), una detrás de la otra;
    - ``"single"``: la misma compresión en una etapa;
    - ``"isothermal"``: la isoterma a T₁ (el mínimo trabajo, el límite con infinitas etapas).

    Cada camino es un dict con ``v`` y ``p`` (SI).
    """
    inp = result.inputs
    R, T1, n = result.gas.R, inp.T1_K, inp.n
    ps = result.pressures_Pa

    def polytropic(pa: float, pb: float) -> tuple[list[float], list[float]]:
        pp = np.geomspace(pa, pb, points)
        va = R * T1 / pa
        vv = va * (pa / pp) ** (1.0 / n)
        return [float(v) for v in vv], [float(p) for p in pp]

    v_st: list[float] = []
    p_st: list[float] = []
    for j in range(len(ps) - 1):
        vv, pp = polytropic(ps[j], ps[j + 1])
        v_st += vv
        p_st += pp
        if j < len(ps) - 2:  # el interenfriador: a p constante hasta T₁
            v_st.append(R * T1 / ps[j + 1])
            p_st.append(ps[j + 1])
    v_one, p_one = polytropic(ps[0], ps[-1])
    pp = np.geomspace(ps[0], ps[-1], points)
    return {
        "stages": {"v": v_st, "p": p_st},
        "single": {"v": v_one, "p": p_one},
        "isothermal": {"v": [float(R * T1 / p) for p in pp], "p": [float(p) for p in pp]},
    }


def staged_to_dict(result: StagedResult) -> dict[str, Any]:
    """La compresión en etapas para exportar (SI)."""
    inp = result.inputs
    return {
        "gas": result.gas.name,
        "p1_Pa": inp.p1_Pa,
        "T1_K": inp.T1_K,
        "p2_Pa": inp.p2_Pa,
        "n": inp.n,
        "etapas": inp.stages,
        "presiones_Pa": list(result.pressures_Pa),
        "relacion_por_etapa": result.stage_ratio,
        "T_salida_etapa_K": result.T_out_K,
        "w_f_J_per_kg": result.w_f,
        "q_interenfriadores_J_per_kg": result.q_intercoolers,
        "w_f_una_etapa_J_per_kg": result.w_f_single,
        "T_salida_una_etapa_K": result.T_out_single_K,
        "ahorro": result.saving,
    }


@dataclass(frozen=True)
class ExponentResult:
    """El exponente politrópico de dos estados medidos."""

    n: float
    k: float
    text: str


def exponent_from_states(
    gas_key: str,
    p1_Pa: float,
    p2_Pa: float,
    *,
    T1_K: float | None = None,
    T2_K: float | None = None,
    v1: float | None = None,
    v2: float | None = None,
) -> ExponentResult:
    """n de dos estados (vademecum §6.1): con p y v, n = ln(p₁/p₂)/ln(v₂/v₁); con p y T,
    (n − 1)/n = ln(T₂/T₁)/ln(p₂/p₁).

    Raises
    ------
    ValueError
        Si faltan datos, si no son positivos o si los dos estados tienen la misma
        presión (el n no queda definido).
    """
    g = gas(gas_key)
    for name, value in (("p₁", p1_Pa), ("p₂", p2_Pa)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"La presión {name} tiene que ser positiva.")
    if abs(math.log(p2_Pa / p1_Pa)) < 1e-9:
        raise ValueError("Con p₂ = p₁ el proceso es una isóbara (n = 0): cambiá alguna presión.")
    if v1 is not None and v2 is not None:
        if not (v1 > 0.0 and v2 > 0.0):
            raise ValueError("Los volúmenes específicos tienen que ser positivos.")
        if abs(math.log(v2 / v1)) < 1e-12:
            raise ValueError("Con v₂ = v₁ es una isócora (n = ±∞).")
        n = math.log(p1_Pa / p2_Pa) / math.log(v2 / v1)
    elif T1_K is not None and T2_K is not None:
        if not (T1_K > 0.0 and T2_K > 0.0):
            raise ValueError("Las temperaturas tienen que ser absolutas y positivas.")
        a = math.log(T2_K / T1_K) / math.log(p2_Pa / p1_Pa)
        if abs(1.0 - a) < 1e-12:
            raise ValueError("Con esos datos el proceso es una isócora (n = ±∞).")
        n = 1.0 / (1.0 - a)
    else:
        raise ValueError("Dá los dos volúmenes específicos o las dos temperaturas.")
    k = g.k(T_REF_K)
    compression = p2_Pa > p1_Pa
    if abs(n - 1.0) < 0.01:
        text = "Prácticamente una isoterma: el gas intercambia todo el trabajo como calor."
    elif 1.0 < n < k:
        text = (
            f"1 < n < k: el gas {'cede' if compression else 'recibe'} calor mientras se "
            f"{'comprime' if compression else 'expande'} (q {'<' if compression else '>'} 0)."
        )
    elif abs(n - k) <= 0.01:
        text = "n ≈ k: prácticamente una adiabática reversible."
    elif n > k:
        text = (
            "n > k: más que una adiabática reversible, como en una compresión adiabática con "
            "fricción o calentada."
            if compression
            else "n > k: el gas cede calor mientras se expande (q < 0)."
        )
    else:
        text = "n < 1: el gas recibe mucho calor; la temperatura sube al expandirse."
    return ExponentResult(n=n, k=k, text=text)


def exponent_to_dict(
    result: ExponentResult,
    gas_key: str,
    p1_Pa: float,
    p2_Pa: float,
    *,
    T1_K: float | None = None,
    T2_K: float | None = None,
    v1: float | None = None,
    v2: float | None = None,
) -> dict[str, Any]:
    """El exponente de dos estados para exportar (SI)."""
    data: dict[str, Any] = {"gas": gas(gas_key).name, "p1_Pa": p1_Pa, "p2_Pa": p2_Pa}
    if v1 is not None and v2 is not None:
        data |= {"v1_m3_per_kg": v1, "v2_m3_per_kg": v2}
    else:
        data |= {"T1_K": T1_K, "T2_K": T2_K}
    return data | {"n": result.n, "k": result.k, "interpretacion": result.text}


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------

C = 273.15
_PSI = 6894.757293168361


@dataclass(frozen=True)
class ProcessExample:
    inputs: PolytropicInputs
    note: str = ""


PROCESS_EXAMPLES: dict[str, ProcessExample] = {
    "Compresor de aire de 100 kPa y 300 K a 900 kPa con n = 1,3 (Çengel §7-12)": ProcessExample(
        PolytropicInputs("air", 100e3, 300.0, "polytropic", "p2", 900e3, n=1.3, system="open"),
        "Çengel: 246,4 kJ/kg (n = 1,3), 263,2 (isoentrópica), 189,2 (isotérmica) y 215,3 en dos "
        "etapas.",
    ),
    "Motor de auto: aire de 95 kPa y 22 °C, adiabática con relación de compresión 8 "
    "(Çengel §7-9)": ProcessExample(
        PolytropicInputs("air", 95e3, 22.0 + C, "adiabatic", "ratio", 8.0, model="variable"),
        "Çengel: 662,7 K con c_p variable (v_r) y 665,2 K con k = 1,391.",
    ),
    "Helio comprimido de 14 psia y 50 °F a 320 °F (Çengel §7-9)": ProcessExample(
        PolytropicInputs(
            "He",
            14.0 * _PSI,
            (50.0 + 459.67) * 5 / 9,
            "adiabatic",
            "T2",
            (320.0 + 459.67) * 5 / 9,
            system="open",
        ),
        "Çengel: p₂ = 40,5 psia.",
    ),
    "Nitrógeno en un cilindro: de 300 kPa y 400 K a 100 kPa con n = 1,3": ProcessExample(
        PolytropicInputs("N2", 300e3, 400.0, "polytropic", "p2", 100e3, n=1.3, amount=0.5),
    ),
    "Aire calentado en un tanque rígido de 100 kPa y 300 K a 600 K": ProcessExample(
        PolytropicInputs("air", 100e3, 300.0, "isochoric", "T2", 600.0, amount=2.0),
    ),
    "Expansión isotérmica de aire de 1 MPa a 100 kPa a 400 K": ProcessExample(
        PolytropicInputs("air", 1e6, 400.0, "isothermal", "p2", 100e3),
    ),
    "CO₂ calentado a presión constante hasta duplicar su volumen": ProcessExample(
        PolytropicInputs("CO2", 200e3, 300.0, "isobaric", "ratio", 0.5, model="variable"),
    ),
}

#: El ensayo de un compresor para «n de dos estados»: 100 kPa y 20 °C a 600 kPa y 200 °C.
EXPONENT_EXAMPLE: dict[str, float] = {
    "p1_Pa": 100e3,
    "T1_K": 20.0 + C,
    "p2_Pa": 600e3,
    "T2_K": 200.0 + C,
}


def process_to_dict(result: ProcessResult) -> dict[str, Any]:
    """El proceso para exportar (SI)."""
    inp = result.inputs
    return {
        "gas": result.gas.name,
        "proceso": PROCESS_KINDS[inp.process],
        "modelo": inp.model,
        "sistema": "cerrado" if inp.system == "closed" else "abierto",
        "cantidad": inp.amount,
        "n": result.n if math.isfinite(result.n) else "infinito",
        "k": result.k,
        "c_J_per_kgK": result.c,
        "estado_1": {
            "p_Pa": result.state1.p_Pa,
            "T_K": result.state1.T_K,
            "v_m3_per_kg": result.state1.v_m3_per_kg,
        },
        "estado_2": {
            "p_Pa": result.state2.p_Pa,
            "T_K": result.state2.T_K,
            "v_m3_per_kg": result.state2.v_m3_per_kg,
        },
        "du_J_per_kg": result.du,
        "dh_J_per_kg": result.dh,
        "ds_J_per_kgK": result.ds,
        "w_J_per_kg": result.w,
        "w_f_J_per_kg": result.w_f,
        "q_J_per_kg": result.q,
        "W_total": result.W_total,
        "Q_total": result.Q_total,
        "notas": list(result.notes),
    }
