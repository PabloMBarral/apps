"""Régimen transitorio: el llenado y el vaciado de un tanque rígido.

Vademecum §3.4 y §10.3.3, sin energía cinética ni potencial y sin trabajo (tanque
rígido):

    Q = m₂·u₂ − m₁·u₁ + Σ_sal m·h − Σ_ent m·h
    S_gen = m₂·s₂ − m₁·s₁ + Σ_sal m·s − Σ_ent m·s − Q/T_b ≥ 0

con el estado del tanque uniforme (Çengel y Boles §5-5).

- **Llenado** desde una línea de estado constante: la entalpía que entra es la de la línea.
  Con la presión final y el calor (0: adiabático), la masa final sale de que el estado
  (v₂ = V/m₂, u₂) tenga esa presión; con p₂ y T₂, el balance da el calor.
- **Vaciado**: la entalpía que sale cambia con el estado del tanque, así que Σ m·h es la
  integral ∫h·dm. Adiabático, lo que queda en el tanque sigue la isoentrópica (d(m·u) =
  h·dm con s constante); isotérmico, la isoterma con el calor que la mantiene. La integral
  se hace con Simpson y se compara con la aproximación de flujo uniforme de Çengel,
  h_sal ≈ (h₁ + h₂)/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq

from core.balances.common import (
    S_FLOOR,
    Verdict,
    deg_c,
    heat_direction_message,
    num,
    snap_sgen,
    verdict,
    violation_message,
)
from core.balances.substance import (
    PairCode,
    Substance,
    ThermoState,
    fluid_substance,
    ideal_gas_substance,
    state,
)

__all__ = [
    "CHARGING_EXAMPLES",
    "DISCHARGE_MODES",
    "DISCHARGING_EXAMPLES",
    "ChargingExample",
    "ChargingInputs",
    "ChargingResult",
    "DischargeMode",
    "DischargingExample",
    "DischargingInputs",
    "DischargingResult",
    "charging_to_dict",
    "discharging_to_dict",
    "solve_charging",
    "solve_discharging",
]

_PATH_POINTS = 41
_INTEGRATION_POINTS = 201  # impar: Simpson


def _kJ(x: float) -> str:
    return f"{num(x / 1e3)} kJ"


def _check_substance(sub: Substance) -> None:
    if sub.kind == "incompressible":
        raise ValueError(
            "Un incompresible no cambia de densidad: un tanque rígido lleno de líquido no se "
            "llena ni se vacía sin que entre otro fluido. Usá un fluido real o un gas ideal."
        )


def _check_common(volume: float, T_b: float | None, T0: float) -> None:
    if not (math.isfinite(volume) and volume > 0.0):
        raise ValueError("El volumen del tanque tiene que ser mayor que cero.")
    if T_b is not None and not (math.isfinite(T_b) and T_b > 0.0):
        raise ValueError("La temperatura de la fuente T_b tiene que ser positiva (absoluta).")
    if not (math.isfinite(T0) and 200.0 <= T0 <= 350.0):
        raise ValueError("La temperatura del ambiente T₀ tiene que estar entre −73 y 77 °C.")


# ---------------------------------------------------------------------
# Llenado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ChargingInputs:
    """Un tanque rígido que se llena desde una línea (SI).

    ``initial_pair = None`` es el tanque vacío. Termina a ``p2_Pa`` con el calor ``Q_J``
    (``end = "Q"``; 0 es adiabático) o con la temperatura ``T2_K`` (``end = "T"``).
    """

    substance: Substance
    volume_m3: float
    line_pair: PairCode
    line_a: float
    line_b: float
    p2_Pa: float
    initial_pair: PairCode | None = None
    initial_a: float = math.nan
    initial_b: float = math.nan
    end: Literal["Q", "T"] = "Q"
    Q_J: float = 0.0
    T2_K: float = math.nan
    T_b_K: float | None = None
    T0_K: float = 298.15


@dataclass(frozen=True)
class ChargingResult:
    inputs: ChargingInputs
    line: ThermoState
    state1: ThermoState | None  # None: vacío
    state2: ThermoState
    m1: float
    m2: float
    Q: float
    T_b: float
    S_gen: float
    path: tuple[tuple[float, ThermoState], ...]  # (masa, estado del tanque)
    violation: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def m_in(self) -> float:
        return self.m2 - self.m1

    @property
    def U1(self) -> float:
        return self.m1 * self.state1.u if self.state1 is not None else 0.0

    @property
    def U2(self) -> float:
        return self.m2 * self.state2.u

    @property
    def H_in(self) -> float:
        return self.m_in * self.line.h

    @property
    def S1(self) -> float:
        return self.m1 * self.state1.s if self.state1 is not None else 0.0

    @property
    def S2(self) -> float:
        return self.m2 * self.state2.s

    @property
    def S_in(self) -> float:
        return self.m_in * self.line.s

    @property
    def X_dest(self) -> float:
        return self.inputs.T0_K * self.S_gen

    @property
    def energy_residual(self) -> float:
        """Q − (U₂ − U₁ − m_ent·h_línea), que cierra a cero (J)."""
        return self.Q - (self.U2 - self.U1 - self.H_in)

    @property
    def verdict(self) -> Verdict:
        return "imposible" if self.violation else verdict(self.S_gen)


def _charging_state(sub: Substance, V: float, m: float, U: float, label: str = "") -> ThermoState:
    return state(sub, "VU", V / m, U / m, label)


def solve_charging(inputs: ChargingInputs) -> ChargingResult:
    """Los balances de un tanque rígido que se llena desde una línea (Çengel §5-5).

    Raises
    ------
    ValueError
        Si la presión final no está entre la inicial y la de la línea, el tanque no llega
        a esa presión con ese calor o un estado queda fuera de rango.
    """
    sub, V = inputs.substance, inputs.volume_m3
    _check_substance(sub)
    _check_common(V, inputs.T_b_K, inputs.T0_K)
    line = state(sub, inputs.line_pair, inputs.line_a, inputs.line_b, "línea")
    p2 = inputs.p2_Pa
    if not (math.isfinite(p2) and p2 > 0.0):
        raise ValueError("La presión final p₂ tiene que ser mayor que cero.")
    if p2 > line.p * (1.0 + 1e-9):
        raise ValueError(
            "El tanque se llena mientras la línea tiene más presión: p₂ no puede superar la "
            "presión de la línea."
        )
    s1: ThermoState | None = None
    m1 = 0.0
    if inputs.initial_pair is not None:
        s1 = state(sub, inputs.initial_pair, inputs.initial_a, inputs.initial_b, "1")
        m1 = V / s1.v
        if p2 <= s1.p:
            raise ValueError(
                "Al llenarse, la presión del tanque sube: p₂ tiene que ser mayor que la inicial."
            )
    U1 = m1 * s1.u if s1 is not None else 0.0
    if inputs.end == "T":
        if not (math.isfinite(inputs.T2_K) and inputs.T2_K > 0.0):
            raise ValueError("La temperatura final T₂ tiene que ser positiva (absoluta).")
        s2 = state(sub, "TP", inputs.T2_K, p2, "2")
        m2 = V / s2.v
        if m2 <= m1:
            raise ValueError(
                f"A {deg_c(inputs.T2_K)} y esa presión cabe menos masa que la inicial "
                f"({num(m2)} kg contra {num(m1)} kg): el tanque no se llenaría."
            )
        Q = m2 * s2.u - U1 - (m2 - m1) * line.h
    else:
        if not math.isfinite(inputs.Q_J):
            raise ValueError("El calor Q tiene que ser un número.")
        Q = inputs.Q_J
        m2 = _charging_mass(sub, V, m1, U1, line, Q, p2)
        s2 = _charging_state(sub, V, m2, U1 + (m2 - m1) * line.h + Q, "2")
    T_b = inputs.T_b_K if inputs.T_b_K is not None else inputs.T0_K
    S1 = m1 * s1.s if s1 is not None else 0.0
    dS = m2 * s2.s - S1 - (m2 - m1) * line.s
    scale = (m2 + m1) * S_FLOOR + abs(m2 * s2.s) + abs(S1) + abs((m2 - m1) * line.s)
    S_gen = snap_sgen(dS - Q / T_b, scale + abs(Q / T_b))
    path = _charging_path(sub, V, m1, U1, line, Q, m2)
    temps = [s2.T] + ([s1.T] if s1 is not None else []) + [s.T for _, s in path]
    violation = heat_direction_message(Q, T_b, min(temps), max(temps), "el tanque") or (
        violation_message(
            f"S_gen = {num(S_gen)} J/K", Q, dS, T_b, min(temps), max(temps), "el tanque"
        )
        if S_gen < 0.0
        else None
    )
    result = ChargingResult(
        inputs=inputs,
        line=line,
        state1=s1,
        state2=s2,
        m1=m1,
        m2=m2,
        Q=Q,
        T_b=T_b,
        S_gen=S_gen,
        path=path,
        violation=violation,
    )
    return replace(result, notes=_charging_notes(result))


def _charging_mass(
    sub: Substance, V: float, m1: float, U1: float, line: ThermoState, Q: float, p2: float
) -> float:
    """La masa final: la del estado (V/m, u) que tiene la presión p₂."""

    def dp(m: float) -> float:
        return _charging_state(sub, V, m, U1 + (m - m1) * line.h + Q).p - p2

    m_ref = V / line.v
    lo = m1 * (1.0 + 1e-9) if m1 > 0.0 else m_ref * 1e-6
    grid = np.geomspace(lo, max(lo, m_ref) * 50.0, 241)
    previous: tuple[float, float] | None = None
    for m in grid:
        try:
            value = dp(float(m))
        except ValueError:
            previous = None
            continue
        if previous is not None and previous[1] < 0.0 <= value:
            return float(brentq(dp, previous[0], float(m), xtol=1e-12, rtol=1e-12))
        previous = (float(m), value)
    raise ValueError(
        f"Con Q = {_kJ(Q)} el tanque no llega a esa presión con un estado posible: revisá "
        "la presión final o el calor."
    )


def _charging_path(
    sub: Substance, V: float, m1: float, U1: float, line: ThermoState, Q: float, m2: float
) -> tuple[tuple[float, ThermoState], ...]:
    """El estado del tanque a medida que entra masa, con el calor proporcional a ella."""
    start = m1 if m1 > 0.0 else m2 / _PATH_POINTS
    out = []
    for m in np.linspace(start, m2, _PATH_POINTS):
        frac = (m - m1) / (m2 - m1)
        try:
            st = _charging_state(sub, V, float(m), U1 + (m - m1) * line.h + Q * frac)
        except ValueError:
            continue
        out.append((float(m), st))
    return tuple(out)


def _charging_notes(r: ChargingResult) -> tuple[str, ...]:
    notes: list[str] = []
    sub = r.inputs.substance
    if r.state1 is None and r.Q == 0.0:
        notes.append(
            "Tanque vacío y adiabático: u₂ = h_línea, así que el tanque termina más caliente "
            "que la línea (el trabajo de flujo p·v de lo que entra se vuelve energía interna)."
        )
        if sub.kind == "ideal_gas" and sub.model == "constant":
            notes.append("Con c_p constante, T₂ = k·T_línea (Çengel §5-5).")
    elif r.Q == 0.0:
        notes.append(
            "Adiabático: lo que entra comprime al gas que ya estaba, y los dos se calientan."
        )
    if (r.inputs.end == "Q" and r.Q != 0.0) or r.inputs.end == "T":
        notes.append(
            "La evolución p(m) y T(m) supone que el calor sale (o entra) en proporción a la masa "
            "que entra; los estados inicial y final no dependen de eso."
        )
    if sub.kind == "fluid" and r.state2.is_two_phase:
        notes.append(f"El tanque termina con una mezcla (x₂ = {num(r.state2.x or 0.0, '.3g')}).")
    return tuple(notes)


# ---------------------------------------------------------------------
# Vaciado
# ---------------------------------------------------------------------

DischargeMode = Literal["adiabatic", "isothermal"]
DISCHARGE_MODES: dict[DischargeMode, str] = {
    "adiabatic": "Adiabático (lo que queda sigue la isoentrópica)",
    "isothermal": "Isotérmico (con el calor que mantiene la temperatura)",
}


@dataclass(frozen=True)
class DischargingInputs:
    """Un tanque rígido que se vacía desde ``pair1`` hasta ``p2_Pa`` (SI).

    Con ``p_out_Pa`` el volumen de control incluye la válvula y lo que sale está a esa
    presión (la entalpía no cambia en la válvula); si no, el volumen de control es solo el
    tanque. ``T_b_K`` es la temperatura de la fuente del calor (isotérmico; ``None``: la
    del tanque).
    """

    substance: Substance
    volume_m3: float
    pair1: PairCode
    a1: float
    b1: float
    p2_Pa: float
    mode: DischargeMode = "adiabatic"
    p_out_Pa: float | None = None
    T_b_K: float | None = None
    T0_K: float = 298.15


@dataclass(frozen=True)
class DischargingResult:
    inputs: DischargingInputs
    state1: ThermoState
    state2: ThermoState
    m1: float
    m2: float
    H_out: float  # ∫h·dm (J)
    S_out: float  # ∫s·dm de lo que sale (J/K), a la presión de salida si hay válvula
    Q: float
    T_b: float
    S_gen: float
    integration_residual: float  # adiabático: m₁u₁ − m₂u₂ − ∫h·dm (control de la integral)
    path: tuple[tuple[float, ThermoState], ...]
    violation: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def m_out(self) -> float:
        return self.m1 - self.m2

    @property
    def h_out_mean(self) -> float:
        """La entalpía media de lo que sale: ∫h·dm / m_sal (J/kg)."""
        return self.H_out / self.m_out

    @property
    def H_out_uniform(self) -> float:
        """La aproximación de flujo uniforme: m_sal·(h₁ + h₂)/2 (Çengel §5-5)."""
        return self.m_out * (self.state1.h + self.state2.h) / 2.0

    @property
    def Q_uniform(self) -> float:
        """El calor que da el balance con la aproximación de flujo uniforme (J)."""
        return self.m2 * self.state2.u - self.m1 * self.state1.u + self.H_out_uniform

    @property
    def X_dest(self) -> float:
        return self.inputs.T0_K * self.S_gen

    @property
    def energy_residual(self) -> float:
        """Q − (m₂u₂ − m₁u₁ + ∫h·dm), que cierra a cero (J)."""
        return self.Q - (self.m2 * self.state2.u - self.m1 * self.state1.u + self.H_out)

    @property
    def verdict(self) -> Verdict:
        return "imposible" if self.violation else verdict(self.S_gen)


def solve_discharging(inputs: DischargingInputs) -> DischargingResult:
    """Los balances de un tanque rígido que se vacía (Çengel §5-5), con ∫h·dm.

    Raises
    ------
    ValueError
        Si la presión final no es menor que la inicial, la de salida es mayor que la final
        o un estado del camino queda fuera de rango.
    """
    sub, V = inputs.substance, inputs.volume_m3
    _check_substance(sub)
    _check_common(V, inputs.T_b_K, inputs.T0_K)
    if inputs.mode not in DISCHARGE_MODES:
        raise ValueError(f"Modo desconocido: {inputs.mode!r}.")
    s1 = state(sub, inputs.pair1, inputs.a1, inputs.b1, "1")
    p2 = inputs.p2_Pa
    if not (math.isfinite(p2) and 0.0 < p2 < s1.p):
        raise ValueError("Al vaciarse, la presión del tanque baja: p₂ tiene que ser menor que p₁.")
    p_out = inputs.p_out_Pa
    if p_out is not None and not (math.isfinite(p_out) and 0.0 < p_out <= p2):
        raise ValueError(
            "Lo que sale pasa por la válvula hacia una presión menor: p_sal tiene que ser "
            "positiva y no mayor que la presión final del tanque."
        )
    pressures = np.geomspace(s1.p, p2, _INTEGRATION_POINTS)
    states = []
    for p in pressures:
        if inputs.mode == "adiabatic":
            states.append(state(sub, "PS", float(p), s1.s))
        else:
            states.append(state(sub, "TP", s1.T, float(p)))
    s2 = states[-1].relabel("2")
    masses = np.array([V / s.v for s in states])
    h = np.array([s.h for s in states])
    m1, m2 = float(masses[0]), float(masses[-1])
    # lo que sale: dm_sal = −dm_tanque; ∫ de m₂ a m₁ (las masas bajan con p)
    H_out = float(simpson(h[::-1], x=masses[::-1]))
    if p_out is None:
        s_exit = np.array([s.s for s in states])
    else:
        s_exit = np.array([state(sub, "PH", p_out, s.h).s for s in states])
    S_out = float(simpson(s_exit[::-1], x=masses[::-1]))
    dU = m2 * s2.u - m1 * s1.u
    if inputs.mode == "adiabatic":
        Q = 0.0
        residual = -dU - H_out
    else:
        Q = dU + H_out
        residual = 0.0
    T_b = inputs.T_b_K if inputs.T_b_K is not None else s1.T
    dS = m2 * s2.s - m1 * s1.s + S_out
    scale = 2.0 * m1 * S_FLOOR + abs(m2 * s2.s) + abs(m1 * s1.s) + abs(S_out)
    S_gen = snap_sgen(dS - Q / T_b, scale + abs(Q / T_b))
    T_lo, T_hi = min(s.T for s in states), max(s.T for s in states)
    violation = heat_direction_message(Q, T_b, T_lo, T_hi, "el tanque") or (
        violation_message(f"S_gen = {num(S_gen)} J/K", Q, dS, T_b, T_lo, T_hi, "el tanque")
        if S_gen < 0.0
        else None
    )
    step = (_INTEGRATION_POINTS - 1) // (_PATH_POINTS - 1)
    path = tuple((float(masses[i]), states[i]) for i in range(0, _INTEGRATION_POINTS, step))
    result = DischargingResult(
        inputs=inputs,
        state1=s1,
        state2=s2,
        m1=m1,
        m2=m2,
        H_out=H_out,
        S_out=S_out,
        Q=Q,
        T_b=T_b,
        S_gen=S_gen,
        integration_residual=residual,
        path=path,
        violation=violation,
    )
    return replace(result, notes=_discharging_notes(result, states))


def _discharging_notes(r: DischargingResult, states: list[ThermoState]) -> tuple[str, ...]:
    notes: list[str] = []
    sub, mode = r.inputs.substance, r.inputs.mode
    error = r.Q_uniform - r.Q
    if mode == "adiabatic":
        notes.append(
            f"Con h_sal ≈ (h₁ + h₂)/2 (flujo uniforme, Çengel §5-5) el balance daría "
            f"Q = {_kJ(r.Q_uniform)} en vez de 0: la entalpía de lo que sale baja durante el "
            f"vaciado y el promedio de los extremos no es su media ({_kJ(error)} de diferencia "
            f"sobre ∫h·dm = {_kJ(r.H_out)})."
        )
    elif abs(error) > 1e-6 * abs(r.Q):
        notes.append(
            f"Con h_sal ≈ (h₁ + h₂)/2 (flujo uniforme) el calor sería {_kJ(r.Q_uniform)}, "
            f"{num(100.0 * error / r.Q, '.2g')} % distinto."
        )
    if sub.kind == "ideal_gas" and mode == "isothermal":
        notes.append(
            "Con un gas ideal a T constante, u y h no cambian: Q = m_sal·R·T = V·(p₁ − p₂)."
        )
    if r.inputs.p_out_Pa is None:
        if mode == "adiabatic":
            notes.append(
                "El volumen de control es el tanque: lo que sale tiene la entropía de lo que "
                "queda y S_gen = 0. La irreversibilidad está en la válvula (sumala con la "
                "presión de salida)."
            )
    else:
        notes.append(
            "El volumen de control incluye la válvula: lo que sale se estrangula hasta p_sal a "
            "h constante, y eso genera entropía."
        )
    if sub.kind == "fluid" and any(s.is_two_phase for s in states):
        notes.append(
            "Lo que queda en el tanque entra en la campana: el modelo supone que sale la "
            "mezcla, pero en un tanque real el líquido se junta abajo y sale vapor."
        )
    return tuple(notes)


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------

C = 273.15


@dataclass(frozen=True)
class ChargingExample:
    inputs: ChargingInputs
    description: str


@dataclass(frozen=True)
class DischargingExample:
    inputs: DischargingInputs
    description: str


_AIR_IDEAL = ideal_gas_substance("air", "variable")

CHARGING_EXAMPLES: dict[str, ChargingExample] = {
    "Tanque vacío que se llena con vapor (Çengel §5-5)": ChargingExample(
        ChargingInputs(
            fluid_substance("Water"),
            1.0,
            "TP",
            300.0 + C,
            1e6,
            p2_Pa=1e6,
            T0_K=25.0 + C,
        ),
        "Un tanque rígido aislado y vacío se llena desde una línea de vapor a 1 MPa y 300 °C "
        "hasta 1 MPa: el libro da T₂ = 456,1 °C (no depende del volumen).",
    ),
    "Tanque con aire que se llena sin calor": ChargingExample(
        ChargingInputs(
            _AIR_IDEAL,
            2.0,
            "TP",
            22.0 + C,
            600e3,
            p2_Pa=600e3,
            initial_pair="TP",
            initial_a=22.0 + C,
            initial_b=100e3,
            T0_K=22.0 + C,
        ),
        "Un tanque de 2 m³ con aire a 100 kPa y 22 °C se llena desde una línea a 600 kPa y "
        "22 °C hasta 600 kPa, aislado.",
    ),
    "Tanque con aire que se llena y se enfría (Çengel §5-5)": ChargingExample(
        ChargingInputs(
            _AIR_IDEAL,
            2.0,
            "TP",
            22.0 + C,
            600e3,
            p2_Pa=600e3,
            initial_pair="TP",
            initial_a=22.0 + C,
            initial_b=100e3,
            end="T",
            T2_K=77.0 + C,
            T0_K=22.0 + C,
        ),
        "El mismo tanque termina a 77 °C: un problema del cap. 5 del libro da 9,58 kg de aire "
        "que entra y 339 kJ de calor que sale (con la tabla A-17).",
    ),
}

DISCHARGING_EXAMPLES: dict[str, DischargingExample] = {
    "Tanque de aire que se vacía sin calor": DischargingExample(
        DischargingInputs(
            fluid_substance("Air"),
            1.0,
            "TP",
            300.0,
            10e5,
            p2_Pa=2e5,
            T0_K=300.0,
        ),
        "1 m³ de aire a 10 bar y 300 K que se vacía hasta 2 bar, aislado: lo que queda se "
        "expande en forma isoentrópica y se enfría (como gas ideal con c_p constante, "
        "300·0,2^(0,4/1,4) = 189,4 K).",
    ),
    "Tanque de aire que se vacía a temperatura constante": DischargingExample(
        DischargingInputs(
            _AIR_IDEAL,
            1.0,
            "TP",
            300.0,
            10e5,
            p2_Pa=2e5,
            mode="isothermal",
            T0_K=300.0,
        ),
        "El mismo tanque, calentado para que se mantenga a 300 K: como gas ideal, "
        "Q = V·(p₁ − p₂) = 800 kJ.",
    ),
    "Vaciado con la válvula en el volumen de control": DischargingExample(
        DischargingInputs(
            fluid_substance("Air"),
            1.0,
            "TP",
            300.0,
            10e5,
            p2_Pa=2e5,
            p_out_Pa=101325.0,
            T0_K=300.0,
        ),
        "El vaciado adiabático, contando la válvula que descarga a la atmósfera: la entropía "
        "se genera en el estrangulamiento.",
    ),
}


def _state_dict(s: ThermoState | None) -> dict[str, Any] | None:
    if s is None:
        return None
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


def charging_to_dict(r: ChargingResult) -> dict[str, Any]:
    return {
        "sustancia": r.inputs.substance.label,
        "volumen_m3": r.inputs.volume_m3,
        "linea": _state_dict(r.line),
        "estado_1": _state_dict(r.state1),
        "estado_2": _state_dict(r.state2),
        "m1_kg": r.m1,
        "m2_kg": r.m2,
        "m_entra_kg": r.m_in,
        "Q_J": r.Q,
        "T_b_K": r.T_b,
        "S_gen_J_K": r.S_gen,
        "X_dest_J": r.X_dest,
        "veredicto": r.verdict,
        "notas": list(r.notes),
    }


def discharging_to_dict(r: DischargingResult) -> dict[str, Any]:
    return {
        "sustancia": r.inputs.substance.label,
        "volumen_m3": r.inputs.volume_m3,
        "modo": DISCHARGE_MODES[r.inputs.mode],
        "estado_1": _state_dict(r.state1),
        "estado_2": _state_dict(r.state2),
        "m1_kg": r.m1,
        "m2_kg": r.m2,
        "m_sale_kg": r.m_out,
        "H_sale_J": r.H_out,
        "H_sale_uniforme_J": r.H_out_uniform,
        "Q_J": r.Q,
        "Q_uniforme_J": r.Q_uniform,
        "p_salida_Pa": r.inputs.p_out_Pa,
        "S_gen_J_K": r.S_gen,
        "X_dest_J": r.X_dest,
        "veredicto": r.verdict,
        "notas": list(r.notes),
    }
