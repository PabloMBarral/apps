"""Sistemas cerrados: los balances de energía y de entropía (vademecum §3, §10 y §13).

Primer principio (§3.1): Q − W = ΔU, con W = W_b − W_ent: el trabajo de frontera que
hace el sistema y el que recibe por otra vía (una resistencia eléctrica, una paleta).
El trabajo de frontera, por el camino:

- cuasiestático (§3.5): W_b = m·∫p·dv;
- contra una presión exterior constante (§3.6): W_b = m·p_ext·(v₂ − v₁).

Segundo principio (§10.3.1): S₂ − S₁ = Q/T_b + S_gen, con T_b la temperatura de la
frontera por donde entra el calor (la de la fuente o el ambiente); X_dest = T₀·S_gen
(§11). Con S_gen < 0 los datos violan el segundo principio.

Los procesos (Çengel y Boles §4-1 y §4-2; §7-3 y §7-13 para la entropía):

- volumen constante: W_b = 0;
- presión constante: W_b = m·p·(v₂ − v₁) y Q + W_ent = m·(h₂ − h₁);
- temperatura constante, cuasiestático e internamente reversible: Q = T·m·(s₂ − s₁) y
  W_b = Q − m·(u₂ − u₁) (sin integrar p·dv: vale también dentro de la campana);
- politrópico p·vⁿ = cte: W_b = m·(p₂v₂ − p₁v₁)/(1 − n) (m·p₁v₁·ln(v₂/v₁) con n = 1);
- adiabático reversible: s₂ = s₁ y W_b = −m·(u₂ − u₁);
- contra p_ext constante: termina a p₂ = p_ext, con u₂ + p_ext·v₂ = u₁ + p_ext·v₁ +
  (Q + W_ent)/m (un estado con p y h);
- estado final dado: el calor o el trabajo, y el otro sale del balance.

El equilibrio térmico entre dos incompresibles (o uno con un reservorio) va aparte,
como en Çengel §4-5 y §7-13.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq

from core.balances.substance import (
    PairCode,
    Substance,
    ThermoState,
    fluid_substance,
    ideal_gas_substance,
    incompressible_substance,
    of,
    state,
)

__all__ = [
    "ALLOWED_ENDS",
    "CLOSED_EXAMPLES",
    "END_NAMES",
    "EQUILIBRIUM_EXAMPLES",
    "PROCESSES",
    "Body",
    "ClosedExample",
    "ClosedInputs",
    "ClosedResult",
    "EndKind",
    "EquilibriumExample",
    "EquilibriumInputs",
    "EquilibriumResult",
    "ProcessKind",
    "VERDICTS",
    "Verdict",
    "allowed_ends",
    "allowed_processes",
    "closed_to_dict",
    "equilibrium_to_dict",
    "snap_sgen",
    "solve_closed",
    "solve_equilibrium",
    "verdict",
]

ProcessKind = Literal[
    "v_const", "p_const", "T_const", "polytropic", "isentropic", "p_ext", "general"
]
PROCESSES: dict[ProcessKind, str] = {
    "v_const": "Volumen constante (tanque rígido)",
    "p_const": "Presión constante (pistón libre)",
    "T_const": "Temperatura constante",
    "polytropic": "Politrópico (p·vⁿ = cte)",
    "isentropic": "Adiabático reversible (s = cte)",
    "p_ext": "Contra una presión exterior constante",
    "general": "Estado final dado",
}

EndKind = Literal["T", "p", "v", "V", "x", "Q", "W"]
END_NAMES: dict[EndKind, str] = {
    "T": "la temperatura final T₂",
    "p": "la presión final p₂",
    "v": "el volumen específico final v₂",
    "V": "el volumen final V₂",
    "x": "el título final x₂",
    "Q": "el calor Q",
    "W": "el trabajo W",
}
#: Qué dato final admite cada proceso (con «general», el estado 2 va con su par).
ALLOWED_ENDS: dict[ProcessKind, tuple[EndKind, ...]] = {
    "v_const": ("T", "p", "x", "Q"),
    "p_const": ("T", "v", "V", "x", "Q"),
    "T_const": ("p", "v", "V", "x", "Q"),
    "polytropic": ("p", "v", "V", "T"),
    "isentropic": ("p", "v", "V", "T"),
    "p_ext": ("Q", "T"),
    "general": ("Q", "W"),
}
#: Los procesos con otro trabajo de entrada (resistencia, paleta); los demás son
#: cuasiestáticos e internamente reversibles, sin él.
_ACCEPTS_W_IN: frozenset[ProcessKind] = frozenset({"v_const", "p_const", "p_ext"})

_PATH_POINTS = 41


def allowed_processes(sub: Substance) -> tuple[ProcessKind, ...]:
    """Los procesos que tienen sentido para ``sub``: un incompresible no cambia de volumen."""
    if sub.kind == "incompressible":
        return ("v_const",)
    return tuple(PROCESSES)


def allowed_ends(process: ProcessKind, sub: Substance) -> tuple[EndKind, ...]:
    """Los datos del estado final para ``process`` y ``sub`` (el título, solo un fluido)."""
    if sub.kind == "incompressible":
        return ("T", "Q")
    ends = ALLOWED_ENDS[process]
    return ends if sub.kind == "fluid" else tuple(e for e in ends if e != "x")


Verdict = Literal["reversible", "irreversible", "imposible"]
VERDICTS: dict[Verdict, str] = {
    "reversible": "Reversible (S_gen = 0)",
    "irreversible": "Irreversible (S_gen > 0)",
    "imposible": "Imposible (S_gen < 0)",
}


def snap_sgen(S_gen: float, scale: float) -> float:
    """S_gen = 0 si es ruido de redondeo frente a ``scale`` (los términos del balance)."""
    return 0.0 if abs(S_gen) <= 1e-9 * (abs(scale) + 1e-12) else S_gen


def verdict(S_gen: float) -> Verdict:
    """Vademecum §10.2: S_gen > 0 irreversible, = 0 reversible y < 0 imposible."""
    if S_gen == 0.0:
        return "reversible"
    return "irreversible" if S_gen > 0.0 else "imposible"


def _num(x: float, fmt: str = ".4g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


def _kJ(x: float) -> str:
    return f"{_num(x / 1e3)} kJ"


def _degC(T: float) -> str:
    return f"{_num(T - 273.15)} °C"


# ---------------------------------------------------------------------
# Datos y resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ClosedInputs:
    """Un sistema cerrado (SI). ``end`` dice qué se da del estado final (``end_value``);
    con ``process="general"``, el estado 2 va con ``end_pair`` y ``end_a``, ``end_b``, y
    ``end`` es el calor o el trabajo que se conoce.

    Se da la masa o el volumen inicial. ``W_in_J`` es el trabajo que entra por otra vía
    que la frontera (eléctrico o de paleta). ``T_b_K`` es la temperatura de la frontera
    por donde pasa el calor (``None``: el ambiente, a ``T0_K``).
    """

    substance: Substance
    pair1: PairCode
    a1: float
    b1: float
    mass_kg: float | None = None
    volume_m3: float | None = None
    process: ProcessKind = "p_const"
    end: EndKind = "T"
    end_value: float = math.nan
    end_pair: PairCode | None = None
    end_a: float = math.nan
    end_b: float = math.nan
    n: float = 1.3
    p_ext_Pa: float | None = None
    W_in_J: float = 0.0
    T_b_K: float | None = None
    T0_K: float = 298.15


@dataclass(frozen=True)
class ClosedResult:
    inputs: ClosedInputs
    state1: ThermoState
    state2: ThermoState
    mass: float
    W_b: float  # trabajo de frontera que hace el sistema (J)
    W_in: float  # trabajo que entra por otra vía (J)
    Q: float  # calor que recibe el sistema (J)
    dU: float
    dS: float  # m·(s₂ − s₁) (J/K)
    T_b: float  # temperatura de la frontera del calor (K)
    S_gen: float  # (J/K)
    path: tuple[ThermoState, ...]
    quasi_static: bool
    reversible_W: float | None = None  # contra p_ext adiabático: el de la reversible
    reversible_state: ThermoState | None = None
    violation: str | None = None  # S_gen < 0: por qué el proceso es imposible
    notes: tuple[str, ...] = ()

    @property
    def V1(self) -> float:
        return self.mass * self.state1.v

    @property
    def V2(self) -> float:
        return self.mass * self.state2.v

    @property
    def W(self) -> float:
        """El trabajo neto que hace el sistema: W = W_b − W_ent (J)."""
        return self.W_b - self.W_in

    @property
    def S_heat(self) -> float:
        """La entropía que entra con el calor: Q/T_b (J/K)."""
        return self.Q / self.T_b

    @property
    def X_dest(self) -> float:
        """Exergía destruida: T₀·S_gen (J)."""
        return self.inputs.T0_K * self.S_gen

    @property
    def verdict(self) -> Verdict:
        return verdict(self.S_gen)

    @property
    def energy_residual(self) -> float:
        """Q − W − ΔU, que cierra a cero (control)."""
        return self.Q - self.W - self.dU


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _mass(inputs: ClosedInputs, s1: ThermoState) -> float:
    if (inputs.mass_kg is None) == (inputs.volume_m3 is None):
        raise ValueError("Dá la masa o el volumen inicial del sistema (uno de los dos).")
    if inputs.mass_kg is not None:
        m = inputs.mass_kg
        if not (math.isfinite(m) and m > 0.0):
            raise ValueError("La masa tiene que ser positiva.")
        return m
    V = inputs.volume_m3
    assert V is not None
    if not (math.isfinite(V) and V > 0.0):
        raise ValueError("El volumen tiene que ser positivo.")
    return V / s1.v


def _check_inputs(inputs: ClosedInputs) -> None:
    proc, end = inputs.process, inputs.end
    if proc not in PROCESSES:
        raise ValueError(f"Proceso desconocido: {proc!r}.")
    if end not in ALLOWED_ENDS[proc]:
        names = ", ".join(END_NAMES[e] for e in ALLOWED_ENDS[proc])
        raise ValueError(
            f"En un proceso «{PROCESSES[proc].lower()}» el dato final es uno de: {names}."
        )
    sub = inputs.substance
    if sub.kind == "incompressible" and proc not in ("v_const", "p_const"):
        raise ValueError(
            f"El volumen {of(sub)} no cambia (vademecum §13): no hay trabajo de frontera y "
            "todos los procesos son el mismo. Usá volumen o presión constante, con T₂ o Q."
        )
    if sub.kind == "incompressible" and end not in ("T", "Q"):
        raise ValueError(f"Para {sub.noun} el dato final es la temperatura T₂ o el calor Q.")
    if not math.isfinite(inputs.W_in_J):
        raise ValueError("El trabajo de entrada tiene que ser un número.")
    if inputs.W_in_J != 0.0 and proc not in _ACCEPTS_W_IN:
        raise ValueError(
            "Otro trabajo de entrada (una resistencia o una paleta) hace al proceso "
            "irreversible: va con volumen constante, presión constante o contra p_ext. Los "
            "demás procesos de la lista son cuasiestáticos y reversibles."
        )
    if proc == "general" and (
        inputs.end_pair is None or not (math.isfinite(inputs.end_a) and math.isfinite(inputs.end_b))
    ):
        raise ValueError("Con «estado final dado» hacen falta dos propiedades del estado 2.")
    if proc != "general" and not math.isfinite(inputs.end_value):
        raise ValueError(f"Falta {END_NAMES[end]}.")
    if proc == "p_ext":
        if inputs.p_ext_Pa is None or not (
            math.isfinite(inputs.p_ext_Pa) and inputs.p_ext_Pa > 0.0
        ):
            raise ValueError("La presión exterior p_ext tiene que ser positiva (absoluta).")
    if proc == "polytropic" and not (math.isfinite(inputs.n) and inputs.n > 0.0):
        raise ValueError("El exponente n de la politrópica tiene que ser positivo.")
    if inputs.T_b_K is not None and not (math.isfinite(inputs.T_b_K) and inputs.T_b_K > 0.0):
        raise ValueError("La temperatura de la fuente T_b tiene que ser positiva (absoluta).")
    if not (math.isfinite(inputs.T0_K) and 200.0 <= inputs.T0_K <= 350.0):
        raise ValueError("La temperatura del ambiente T₀ tiene que estar entre −73 y 77 °C.")


def _end_volume(inputs: ClosedInputs, m: float) -> float:
    """v₂ con v₂ o V₂ dados."""
    value = inputs.end_value
    if not value > 0.0:
        raise ValueError("El volumen final tiene que ser positivo.")
    return value / m if inputs.end == "V" else value


def _path(sub: Substance, pair: PairCode, a_values: Any, b_values: Any) -> tuple[ThermoState, ...]:
    out = []
    for a, b in zip(a_values, b_values, strict=True):
        try:
            out.append(state(sub, pair, float(a), float(b)))
        except ValueError:
            continue
    return tuple(out)


def _geom(a: float, b: float) -> np.ndarray:
    return np.geomspace(a, b, _PATH_POINTS)


def _lin(a: float, b: float) -> np.ndarray:
    return np.linspace(a, b, _PATH_POINTS)


def solve_closed(inputs: ClosedInputs) -> ClosedResult:
    """Los balances de un sistema cerrado entre el estado 1 y el 2.

    Una S_gen < 0 no es un error: el balance de energía cierra igual, pero el proceso es
    imposible (``violation`` dice por qué).

    Raises
    ------
    ValueError
        Con datos que no sirven para el proceso o un estado fuera de rango, con un
        mensaje para el alumno.
    """
    _check_inputs(inputs)
    sub, proc, end = inputs.substance, inputs.process, inputs.end
    s1 = state(sub, inputs.pair1, inputs.a1, inputs.b1, "1")
    m = _mass(inputs, s1)
    W_in = inputs.W_in_J
    value = inputs.end_value
    quasi_static = True
    path: tuple[ThermoState, ...] = ()
    reversible_W: float | None = None
    reversible_state: ThermoState | None = None
    Q: float | None = None

    if sub.kind == "incompressible":
        # v = cte: W_b = 0 y Q + W_ent = m·Δu (vademecum §13)
        if end == "T":
            s2 = state(sub, "TP", value, s1.p, "2")
        else:
            s2 = state(sub, "PU", s1.p, s1.u + (value + W_in) / m, "2")
        W_b = 0.0
        path = _path(sub, "TP", _lin(s1.T, s2.T), [s1.p] * _PATH_POINTS)
    elif proc == "v_const":
        if end == "T":
            s2 = state(sub, "TV", value, s1.v, "2")
        elif end == "p":
            s2 = state(sub, "PV", value, s1.v, "2")
        elif end == "x":
            s2 = state(sub, "VX", s1.v, value, "2")
        else:
            s2 = state(sub, "VU", s1.v, s1.u + (value + W_in) / m, "2")
        W_b = 0.0
        path = _path(sub, "VU", [s1.v] * _PATH_POINTS, _lin(s1.u, s2.u))
    elif proc == "p_const":
        if end == "T":
            s2 = state(sub, "TP", value, s1.p, "2")
        elif end in ("v", "V"):
            s2 = state(sub, "PV", s1.p, _end_volume(inputs, m), "2")
        elif end == "x":
            s2 = state(sub, "PX", s1.p, value, "2")
        else:
            s2 = state(sub, "PH", s1.p, s1.h + (value + W_in) / m, "2")
        W_b = m * s1.p * (s2.v - s1.v)
        path = _path(sub, "PH", [s1.p] * _PATH_POINTS, _lin(s1.h, s2.h))
    elif proc == "T_const":
        if end == "p":
            s2 = state(sub, "TP", s1.T, value, "2")
        elif end in ("v", "V"):
            s2 = state(sub, "TV", s1.T, _end_volume(inputs, m), "2")
        elif end == "x":
            s2 = state(sub, "TX", s1.T, value, "2")
        else:
            s2 = state(sub, "TS", s1.T, s1.s + value / (m * s1.T), "2")
        Q = s1.T * m * (s2.s - s1.s)
        W_b = Q - m * (s2.u - s1.u)
        path = _path(sub, "TV", [s1.T] * _PATH_POINTS, _geom(s1.v, s2.v))
    elif proc == "polytropic":
        n = inputs.n
        if end == "p":
            p2 = value
            s2 = state(sub, "PV", p2, s1.v * (s1.p / p2) ** (1.0 / n), "2")
        elif end in ("v", "V"):
            v2 = _end_volume(inputs, m)
            s2 = state(sub, "PV", s1.p * (s1.v / v2) ** n, v2, "2")
        else:
            s2 = _polytropic_to_T(sub, s1, n, value)
        W_b = (
            m * s1.p * s1.v * math.log(s2.v / s1.v)
            if math.isclose(n, 1.0)
            else m * (s2.p * s2.v - s1.p * s1.v) / (1.0 - n)
        )
        ps = _geom(s1.p, s2.p)
        path = _path(sub, "PV", ps, [s1.v * (s1.p / p) ** (1.0 / n) for p in ps])
    elif proc == "isentropic":
        if end == "p":
            s2 = state(sub, "PS", value, s1.s, "2")
        elif end in ("v", "V"):
            s2 = state(sub, "VS", _end_volume(inputs, m), s1.s, "2")
        else:
            s2 = state(sub, "TS", value, s1.s, "2")
        Q = 0.0
        W_b = -m * (s2.u - s1.u)
        path = _path(sub, "PS", _geom(s1.p, s2.p), [s1.s] * _PATH_POINTS)
    elif proc == "p_ext":
        p_ext = inputs.p_ext_Pa
        assert p_ext is not None
        if end == "Q":
            h2 = s1.u + p_ext * s1.v + (value + W_in) / m
            s2 = state(sub, "PH", p_ext, h2, "2")
        else:
            s2 = state(sub, "TP", value, p_ext, "2")
        W_b = m * p_ext * (s2.v - s1.v)
        quasi_static = False
        path = (s1, s2)
        if end == "Q" and value == 0.0 and W_in == 0.0:
            reversible_state = state(sub, "PS", p_ext, s1.s, "2s")
            reversible_W = -m * (reversible_state.u - s1.u)
    else:  # general
        assert inputs.end_pair is not None
        s2 = state(sub, inputs.end_pair, inputs.end_a, inputs.end_b, "2")
        quasi_static = False
        path = (s1, s2)
        if end == "Q":
            Q = value
            W_b = Q - m * (s2.u - s1.u)
        else:
            W_b = value
            Q = W_b + m * (s2.u - s1.u)

    dU = m * (s2.u - s1.u)
    if Q is None:
        Q = dU + W_b - W_in
    T_b = inputs.T_b_K if inputs.T_b_K is not None else inputs.T0_K
    if proc == "T_const" and inputs.T_b_K is None:
        T_b = s1.T  # el calor reversible entra a la temperatura del sistema
    dS = m * (s2.s - s1.s)
    S_gen = snap_sgen(dS - Q / T_b, abs(dS) + abs(Q / T_b))
    violation = _violation(Q, T_b, s1, s2, S_gen) if S_gen < 0.0 else None
    result = ClosedResult(
        inputs=inputs,
        state1=s1,
        state2=s2,
        mass=m,
        W_b=W_b,
        W_in=W_in,
        Q=Q,
        dU=dU,
        dS=dS,
        T_b=T_b,
        S_gen=S_gen,
        path=path if path else (s1, s2),
        quasi_static=quasi_static,
        reversible_W=reversible_W,
        reversible_state=reversible_state,
        violation=violation,
    )
    return replace(result, notes=_notes(result))


def _polytropic_to_T(sub: Substance, s1: ThermoState, n: float, T2: float) -> ThermoState:
    """El estado de la politrópica que llega a T₂ (se busca p₂)."""
    if math.isclose(n, 1.0) and sub.kind == "ideal_gas":
        raise ValueError(
            "Con n = 1 un gas ideal no cambia de temperatura (p·v = R·T = cte): usá el proceso a "
            "temperatura constante y dale p₂ o v₂."
        )

    def T_at(p: float) -> float:
        return state(sub, "PV", p, s1.v * (s1.p / p) ** (1.0 / n)).T - T2

    lo, hi = s1.p / 1e3, s1.p * 1e3
    grid = np.geomspace(lo, hi, 121)
    values = []
    for p in grid:
        try:
            values.append(T_at(float(p)))
        except ValueError:
            values.append(math.nan)
    for i in range(len(grid) - 1):
        a, b = values[i], values[i + 1]
        if math.isfinite(a) and math.isfinite(b) and a * b <= 0.0:
            p2 = float(brentq(T_at, grid[i], grid[i + 1], xtol=1e-9, rtol=1e-12))
            return state(sub, "PV", p2, s1.v * (s1.p / p2) ** (1.0 / n), "2")
    raise ValueError(
        f"La politrópica con n = {_num(n)} no llega a {_degC(T2)} entre p₁/1000 y 1000·p₁. "
        "Revisá n y la temperatura final."
    )


def _violation(Q: float, T_b: float, s1: ThermoState, s2: ThermoState, S_gen: float) -> str:
    base = (
        f"S_gen = {_num(S_gen)} J/K < 0: el proceso es imposible. El balance de energía cierra, "
        "pero viola el segundo principio (vademecum §10.2)."
    )
    if Q > 0 and T_b < max(s1.T, s2.T):
        return (
            f"{base} El calor entra desde una fuente a {_degC(T_b)}, más fría que el sistema "
            f"(llega a {_degC(max(s1.T, s2.T))}): el calor no pasa solo de frío a caliente. "
            "Subí la temperatura de la fuente T_b."
        )
    if Q < 0 and T_b > min(s1.T, s2.T):
        return (
            f"{base} El calor sale hacia un medio a {_degC(T_b)}, más caliente que el sistema "
            f"(baja a {_degC(min(s1.T, s2.T))}). Bajá la temperatura del medio T_b."
        )
    if Q == 0.0:
        return (
            f"{base} Sin calor, la entropía no puede bajar: un sistema aislado (o adiabático) "
            "solo genera entropía."
        )
    return f"{base} Revisá los datos."


def _notes(r: ClosedResult) -> tuple[str, ...]:
    notes: list[str] = []
    inputs, s1, s2 = r.inputs, r.state1, r.state2
    sub = inputs.substance
    if sub.kind == "fluid":
        if s1.region == "compressed_liquid" or s2.region == "compressed_liquid":
            notes.append(
                "Hay un líquido comprimido: las tablas suelen aproximarlo con el líquido "
                "saturado a la misma T (u ≈ u_f(T), Çengel §3-5); acá sale de la ecuación de "
                "estado, así que puede diferir un poco del libro."
            )
        if s2.is_two_phase:
            x2 = _num(s2.x or 0.0, ".3g")
            notes.append(f"El estado final es una mezcla de líquido y vapor (x₂ = {x2}).")
        if inputs.process == "T_const" and s1.is_two_phase and s2.is_two_phase:
            notes.append(
                "Dentro de la campana la isoterma es también isobara: el calor es m·h_fg·Δx."
            )
    if not r.quasi_static and inputs.process == "p_ext":
        if r.reversible_W is not None and r.reversible_state is not None:
            notes.append(
                f"No es cuasiestático: W = p_ext·ΔV = {_kJ(r.W_b)}. Adiabático y reversible hasta "
                f"la misma presión haría {_kJ(r.reversible_W)} y terminaría a "
                f"{_degC(r.reversible_state.T)}: la diferencia es la irreversibilidad "
                f"(S_gen = {_num(r.S_gen)} J/K)."
            )
        else:
            notes.append(
                "No es cuasiestático: el sistema no tiene una presión definida durante el "
                "proceso y el trabajo se calcula con la del exterior (vademecum §3.6)."
            )
    if inputs.process == "general":
        notes.append(
            "Con el estado final dado no hay un camino: W y Q dependen de cómo se llegó; acá "
            "uno es dato y el otro sale del balance. En los diagramas, la recta rayada solo "
            "une los dos estados."
        )
    if r.S_gen == 0.0:
        notes.append("S_gen = 0: el proceso es reversible (internamente y con la fuente).")
    elif r.S_gen > 0.0 and r.W_in > 0.0:
        notes.append(
            "El trabajo de una resistencia o una paleta siempre genera entropía: se disipa "
            "en el sistema."
        )
    return tuple(notes)


# ---------------------------------------------------------------------
# Equilibrio térmico
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Body:
    """Un cuerpo incompresible a T (vademecum §13)."""

    substance: Substance
    mass_kg: float
    T_K: float
    label: str = ""


@dataclass(frozen=True)
class EquilibriumInputs:
    """Dos cuerpos en un recipiente aislado, o uno con un reservorio a ``T_reservoir_K``."""

    body_a: Body
    body_b: Body | None = None
    T_reservoir_K: float | None = None
    T0_K: float = 298.15


@dataclass(frozen=True)
class BodyChange:
    body: Body
    Q: float  # calor que recibe (J)
    dS: float  # (J/K)


@dataclass(frozen=True)
class EquilibriumResult:
    inputs: EquilibriumInputs
    T_final: float
    changes: tuple[BodyChange, ...]
    reservoir_dS: float | None  # el reservorio: −Q/T_res (J/K)
    S_gen: float
    notes: tuple[str, ...] = field(default=())

    @property
    def X_dest(self) -> float:
        return self.inputs.T0_K * self.S_gen


def _check_body(b: Body) -> None:
    where = f" ({b.label})" if b.label else ""
    if b.substance.kind != "incompressible":
        raise ValueError(f"Cada cuerpo{where} tiene que ser un líquido o un sólido incompresible.")
    if not (math.isfinite(b.mass_kg) and b.mass_kg > 0.0):
        raise ValueError(f"La masa{where} tiene que ser positiva.")
    if not (math.isfinite(b.T_K) and b.T_K > 0.0):
        raise ValueError(f"La temperatura{where} tiene que ser positiva (absoluta).")


def solve_equilibrium(inputs: EquilibriumInputs) -> EquilibriumResult:
    """La temperatura final y la entropía generada (Çengel §4-5 y §7-13).

    Dos cuerpos aislados: T_f = Σm·c·T/Σm·c. Con un reservorio: T_f = T_res, y el
    reservorio cambia su entropía en −Q/T_res.

    Raises
    ------
    ValueError
        Si un cuerpo no es incompresible o falta el segundo cuerpo o el reservorio.
    """
    a, b = inputs.body_a, inputs.body_b
    _check_body(a)
    if (b is None) == (inputs.T_reservoir_K is None):
        raise ValueError("Dá el segundo cuerpo o la temperatura del reservorio (uno de los dos).")
    bodies = [a] if b is None else [a, b]
    if b is not None:
        _check_body(b)
    if not (math.isfinite(inputs.T0_K) and 200.0 <= inputs.T0_K <= 350.0):
        raise ValueError("La temperatura del ambiente T₀ tiene que estar entre −73 y 77 °C.")
    mc = [x.mass_kg * x.substance.material.c for x in bodies]
    if b is None:
        T_res = inputs.T_reservoir_K
        assert T_res is not None
        if not (math.isfinite(T_res) and T_res > 0.0):
            raise ValueError("La temperatura del reservorio tiene que ser positiva (absoluta).")
        T_f = T_res
    else:
        T_f = sum(k * x.T_K for k, x in zip(mc, bodies, strict=True)) / sum(mc)
    changes = tuple(
        BodyChange(x, k * (T_f - x.T_K), k * math.log(T_f / x.T_K))
        for k, x in zip(mc, bodies, strict=True)
    )
    reservoir_dS = None if b is not None else -changes[0].Q / T_f
    S_gen = snap_sgen(
        sum(c.dS for c in changes) + (reservoir_dS or 0.0),
        sum(abs(c.dS) for c in changes) + abs(reservoir_dS or 0.0),
    )
    result = EquilibriumResult(inputs, T_f, changes, reservoir_dS, S_gen)
    notes = [
        "El calor que pierde el caliente lo gana el frío (Q_a + Q_b = 0), pero la entropía "
        "que gana el frío es mayor que la que pierde el caliente: recibe el mismo calor a "
        "menor temperatura."
        if b is not None
        else "El reservorio no cambia de temperatura: su ΔS es −Q/T_res. El cuerpo cambia "
        "de entropía en m·c·ln(T_res/T₁), y la suma siempre es positiva."
    ]
    return replace(result, notes=tuple(notes))


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------

C = 273.15
_PSIA = 6894.757293168
_LBM = 0.45359237


def _F(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0 + C


@dataclass(frozen=True)
class ClosedExample:
    inputs: ClosedInputs
    description: str


@dataclass(frozen=True)
class EquilibriumExample:
    inputs: EquilibriumInputs
    description: str


_WATER = fluid_substance("Water")
_AIR = ideal_gas_substance("air", "variable")
_V1_UNRESTRAINED = 1.0 / 997.05  # v del líquido a 200 kPa y 25 °C (≈ 0,001003 m³/kg)
_V1_ENGINE = state(_AIR, "TP", 22.0 + C, 95e3).v

CLOSED_EXAMPLES: dict[str, ClosedExample] = {
    "Vapor de agua calentado a presión constante (Çengel §4-1)": ClosedExample(
        ClosedInputs(
            _WATER,
            "TP",
            _F(320.0),
            60.0 * _PSIA,
            mass_kg=10.0 * _LBM,
            process="p_const",
            end="T",
            end_value=_F(400.0),
            T_b_K=_F(500.0),
            T0_K=_F(77.0),
        ),
        "10 lbm de vapor a 60 psia y 320 °F calentados a 400 °F: el libro da W_b = 96,4 Btu.",
    ),
    "Calentamiento eléctrico a presión constante (Çengel §4-2)": ClosedExample(
        ClosedInputs(
            _WATER,
            "PX",
            300e3,
            1.0,
            mass_kg=0.025,
            process="p_const",
            end="Q",
            end_value=-3.7e3,
            W_in_J=7.2e3,
            T0_K=25.0 + C,
        ),
        "25 g de vapor saturado a 300 kPa con una resistencia de 120 V y 0,2 A durante 5 min "
        "(7,2 kJ) que pierden 3,7 kJ: el libro da T₂ = 200 °C.",
    ),
    "Expansión libre del agua (Çengel §4-2)": ClosedExample(
        ClosedInputs(
            _WATER,
            "TP",
            25.0 + C,
            200e3,
            mass_kg=5.0,
            process="general",
            end="W",
            end_value=0.0,
            end_pair="TV",
            end_a=25.0 + C,
            end_b=2.0 * _V1_UNRESTRAINED,
            T0_K=25.0 + C,
        ),
        "5 kg de agua a 200 kPa y 25 °C llenan un tanque del doble de volumen (sin trabajo) y "
        "vuelven a 25 °C: el libro da p₂ = 3,169 kPa y Q = 0,25 kJ, con u₁ ≈ u_f(25 °C).",
    ),
    "R-134a que se enfría en un tanque rígido (Çengel §7-3)": ClosedExample(
        ClosedInputs(
            fluid_substance("R134a"),
            "TP",
            20.0 + C,
            140e3,
            mass_kg=5.0,
            process="v_const",
            end="p",
            end_value=100e3,
            T_b_K=-30.0 + C,
            T0_K=25.0 + C,
        ),
        "5 kg de R-134a a 20 °C y 140 kPa que se enfrían hasta 100 kPa: el libro da "
        "ΔS = −1,173 kJ/K.",
    ),
    "Compresión isotérmica de aire (Çengel §4-1)": ClosedExample(
        ClosedInputs(
            ideal_gas_substance("air", "constant"),
            "TP",
            80.0 + C,
            100e3,
            volume_m3=0.4,
            process="T_const",
            end="V",
            end_value=0.1,
            T0_K=25.0 + C,
        ),
        "0,4 m³ de aire a 100 kPa y 80 °C comprimidos a 80 °C hasta 0,1 m³: el libro da "
        "W_b = −55,5 kJ.",
    ),
    "Mezcla saturada que cede calor al ambiente (Çengel §7-13)": ClosedExample(
        ClosedInputs(
            _WATER,
            "TX",
            100.0 + C,
            0.5,
            mass_kg=1.0,
            process="p_const",
            end="Q",
            end_value=-600e3,
            T_b_K=25.0 + C,
            T0_K=25.0 + C,
        ),
        "Una mezcla de agua a 100 °C cede 600 kJ al aire a 25 °C: el libro da ΔS = −1,61 kJ/K "
        "y S_gen = 0,40 kJ/K.",
    ),
    "Aire que se expande contra la atmósfera (no cuasiestático)": ClosedExample(
        ClosedInputs(
            ideal_gas_substance("air", "constant"),
            "TP",
            300.0,
            3e5,
            volume_m3=0.01,
            process="p_ext",
            end="Q",
            end_value=0.0,
            p_ext_Pa=1e5,
            T0_K=300.0,
        ),
        "10 litros de aire a 3 bar y 300 K en un cilindro: se suelta el pistón contra 1 bar sin "
        "intercambiar calor. Comparalo con la expansión reversible.",
    ),
    "Vapor de agua que se expande sin calor ni fricción": ClosedExample(
        ClosedInputs(
            _WATER,
            "TP",
            300.0 + C,
            1e6,
            mass_kg=1.0,
            process="isentropic",
            end="p",
            end_value=1e5,
            T0_K=25.0 + C,
        ),
        "1 kg de vapor a 1 MPa y 300 °C que se expande isoentrópicamente hasta 100 kPa: entra "
        "en la campana (x₂ ≈ 0,96).",
    ),
    "Compresión isoentrópica del aire en un motor (Çengel §7-9)": ClosedExample(
        ClosedInputs(
            _AIR,
            "TP",
            22.0 + C,
            95e3,
            mass_kg=1.0,
            process="isentropic",
            end="v",
            end_value=_V1_ENGINE / 8.0,
            T0_K=22.0 + C,
        ),
        "Aire a 22 °C y 95 kPa comprimido en forma reversible y adiabática con una relación de "
        "compresión V₁/V₂ = 8 (v₂ = v₁/8): el libro da T₂ = 662,7 K con la tabla A-17.",
    ),
    "Compresión politrópica de aire (n = 1,3)": ClosedExample(
        ClosedInputs(
            _AIR,
            "TP",
            300.0,
            1e5,
            mass_kg=0.1,
            process="polytropic",
            end="p",
            end_value=8e5,
            n=1.3,
            T0_K=25.0 + C,
        ),
        "100 g de aire a 1 bar y 300 K comprimidos con p·v^1,3 = cte hasta 8 bar: con n < k el "
        "aire cede calor al ambiente mientras se comprime.",
    ),
    "Agua caliente en un termo que se enfría": ClosedExample(
        ClosedInputs(
            incompressible_substance("water"),
            "TP",
            90.0 + C,
            101325.0,
            mass_kg=1.0,
            process="v_const",
            end="T",
            end_value=60.0 + C,
            T0_K=20.0 + C,
        ),
        "1 kg de agua a 90 °C que se enfría a 60 °C cediendo el calor al aire a 20 °C "
        "(incompresible).",
    ),
}

_IRON_CENGEL = incompressible_substance("iron", c=450.0)

EQUILIBRIUM_EXAMPLES: dict[str, EquilibriumExample] = {
    "Bloque de hierro en agua (Çengel §4-5)": EquilibriumExample(
        EquilibriumInputs(
            Body(_IRON_CENGEL, 50.0, 80.0 + C, "hierro"),
            Body(
                incompressible_substance("water", c=4180.0, rho=997.0),
                0.5 * 997.0,
                25.0 + C,
                "agua",
            ),
            T0_K=25.0 + C,
        ),
        "50 kg de hierro a 80 °C (c = 0,45 kJ/(kg·K)) en 0,5 m³ de agua a 25 °C: el libro da "
        "25,6 °C.",
    ),
    "Bloque de hierro en un lago (Çengel §7-13)": EquilibriumExample(
        EquilibriumInputs(
            Body(_IRON_CENGEL, 50.0, 500.0, "hierro"), T_reservoir_K=285.0, T0_K=285.0
        ),
        "50 kg de hierro a 500 K en un lago a 285 K: el libro da S_gen = 4,32 kJ/K.",
    ),
    "Cobre caliente en aceite": EquilibriumExample(
        EquilibriumInputs(
            Body(incompressible_substance("copper"), 2.0, 300.0 + C, "cobre"),
            Body(incompressible_substance("oil"), 5.0, 20.0 + C, "aceite"),
            T0_K=20.0 + C,
        ),
        "Una pieza de cobre de 2 kg a 300 °C templada en 5 kg de aceite a 20 °C.",
    ),
}


def _state_dict(s: ThermoState) -> dict[str, Any]:
    return {
        "p_Pa": s.p,
        "T_K": s.T,
        "v_m3_kg": s.v,
        "u_J_kg": s.u,
        "h_J_kg": s.h,
        "s_J_kgK": s.s,
        "x": s.x,
        "region": s.region_es,
    }


def closed_to_dict(r: ClosedResult) -> dict[str, Any]:
    inp = r.inputs
    return {
        "sustancia": inp.substance.label,
        "proceso": PROCESSES[inp.process],
        "masa_kg": r.mass,
        "estado_1": _state_dict(r.state1),
        "estado_2": _state_dict(r.state2),
        "V1_m3": r.V1,
        "V2_m3": r.V2,
        "W_frontera_J": r.W_b,
        "W_entrada_J": r.W_in,
        "Q_J": r.Q,
        "dU_J": r.dU,
        "dS_J_K": r.dS,
        "T_b_K": r.T_b,
        "Q_sobre_Tb_J_K": r.S_heat,
        "S_gen_J_K": r.S_gen,
        "X_dest_J": r.X_dest,
        "T0_K": inp.T0_K,
        "cuasiestatico": r.quasi_static,
        "notas": list(r.notes),
    }


def equilibrium_to_dict(r: EquilibriumResult) -> dict[str, Any]:
    return {
        "T_final_K": r.T_final,
        "cuerpos": [
            {
                "cuerpo": c.body.label or c.body.substance.label,
                "material": c.body.substance.label,
                "masa_kg": c.body.mass_kg,
                "T1_K": c.body.T_K,
                "Q_J": c.Q,
                "dS_J_K": c.dS,
            }
            for c in r.changes
        ],
        "dS_reservorio_J_K": r.reservoir_dS,
        "S_gen_J_K": r.S_gen,
        "X_dest_J": r.X_dest,
        "notas": list(r.notes),
    }
