"""Procesos de acondicionamiento de aire y torre de enfriamiento húmeda — Fase 4.

Los procesos del aire húmedo del vademecum §14.12 (Çengel & Boles §14-7), en
régimen permanente y sin energías cinética ni potencial. Todo por kg de aire
seco, que se conserva; el agua que entra o sale lo hace con la entalpía de
las tablas (h_w = h_f(T_w) o h_g(T_w)):

- **calentamiento o enfriamiento sensible** (§14.12.1): ω constante,
  Q̇ = ṁ_a·(h₂ − h₁);
- **calentamiento con humidificación** (§14.12.2): Q̇ + ṁ_w·h_w = ṁ_a·(h₂ − h₁),
  con ṁ_w = ṁ_a·(ω₂ − ω₁);
- **enfriamiento y deshumidificación** (§14.12.3):
  Q̇ = ṁ_a·[(h₂ − h₁) + (ω₁ − ω₂)·h_w], con el condensado a T_w;
- **humidificación adiabática** (§14.12.4): h₂ = h₁ + (ω₂ − ω₁)·h_w;
- **mezcla adiabática** de dos corrientes (§14.12.6);
- **torre de enfriamiento húmeda** (§14.12.7).

Un tren de procesos encadena hasta :data:`MAX_PROCESSES` sobre una corriente
(con mezclas que suman otras); los estados se numeran en el sentido del flujo,
como en los ejemplos de Cengel. La **exergía destruida** en cada proceso sale
del balance de exergía con la ψ del vademecum §14.11 (y la del agua de Wepfer
et al., 1979): X_dest = Σ ṁ·ψ (entra) − Σ ṁ·ψ (sale) + Q̇·(1 − T₀/T_b), con T_b la
temperatura de la fuente (o del serpentín) que entrega o recibe el calor.

Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq

from core.psychrometrics import (
    P_SEA_LEVEL_PA,
    T_MAX_K,
    T_MIN_K,
    T_TRIPLE_K,
    DeadState,
    MoistAirState,
    WaterPhase,
    dry_bulb_from_h_omega,
    humidity_ratio,
    moist_enthalpy,
    pressure_from_altitude,
    saturation_pressure,
    saturation_temperature,
    state_from_T_omega,
    state_from_T_phi,
    vapor_pressure,
    water_enthalpy,
    water_exergy,
)
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "HVAC_EXAMPLES",
    "HVAC_EXAMPLE_NOTES",
    "MAX_PROCESSES",
    "TOWER_EXAMPLES",
    "TOWER_EXAMPLE_NOTES",
    "AdiabaticHumidification",
    "AdiabaticMixing",
    "AirFlow",
    "AirInlet",
    "CoolingDehumidification",
    "CoolingTowerInputs",
    "CoolingTowerResult",
    "HeatingHumidification",
    "HvacInputs",
    "HvacResult",
    "HvacStream",
    "HvacSweepPoint",
    "ProcessResult",
    "ProcessSpec",
    "SensibleProcess",
    "TowerSweepPoint",
    "cooling_tower_notes",
    "cooling_tower_sweep",
    "cooling_tower_to_dict",
    "default_source_temperature",
    "default_tower_sweep_values",
    "hvac_notes",
    "hvac_sweep",
    "hvac_to_dict",
    "solve_cooling_tower",
    "solve_hvac",
]

#: Procesos que admite un tren.
MAX_PROCESSES = 4
#: Fuente de calor por defecto de un calentamiento: agua caliente a 60 °C.
_DEFAULT_HOT_SOURCE_K = 333.15


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _pct(fraction: float) -> str:
    return f"{100 * fraction:.4g} %"


# ---------------------------------------------------------------------
# Datos: corrientes y procesos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AirFlow:
    """Caudal de una corriente: volumétrico (m³/s, en su estado) o de aire seco (kg/s)."""

    value: float
    kind: Literal["volume", "dry_air"] = "volume"

    def __post_init__(self) -> None:
        if not self.value > 0.0:
            raise ValueError("El caudal de aire tiene que ser positivo.")

    def dry_air(self, state: MoistAirState) -> float:
        """Caudal de aire seco, kg/s: ṁ_a = V̇/v (v por kg de aire seco) o el dado."""
        if self.kind == "dry_air":
            return self.value
        return self.value / state.v_m3_per_kg


@dataclass(frozen=True)
class AirInlet:
    """Corriente de aire que entra: T, φ y caudal."""

    T_K: float
    phi: float
    flow: AirFlow


@dataclass(frozen=True)
class SensibleProcess:
    """Calentamiento o enfriamiento sensible, sin cambio de humedad (vademecum §14.12.1).

    ``T_source_K`` es la temperatura de la fuente de calor (calentamiento) o de
    la superficie del serpentín (enfriamiento), para la exergía; ``None`` usa
    :func:`default_source_temperature`.
    """

    T_out_K: float
    T_source_K: float | None = None


@dataclass(frozen=True)
class HeatingHumidification:
    """Calentamiento con humidificación (§14.12.2): la salida (T, φ) y el agua que se agrega.

    El agua entra como vapor saturado o como líquido a ``T_water_K``; el calor
    sale del balance de energía: Q̇ = ṁ_a·(h₂ − h₁) − ṁ_w·h_w.
    """

    T_out_K: float
    phi_out: float
    water: Literal["vapor", "liquid"] = "vapor"
    T_water_K: float = 373.15
    T_source_K: float | None = None


@dataclass(frozen=True)
class CoolingDehumidification:
    """Serpentín de enfriamiento y deshumidificación (§14.12.3).

    El aire sale a (T, φ) y el condensado a ``T_water_K``, la temperatura de
    la superficie fría del serpentín (``None``: la de salida del aire, como
    Cengel).
    """

    T_out_K: float
    phi_out: float = 1.0
    T_water_K: float | None = None


@dataclass(frozen=True)
class AdiabaticHumidification:
    """Humidificación adiabática (§14.12.4): sale con φ dada; la T sale del balance.

    Con agua líquida es el enfriador evaporativo: ``T_water_K`` ``None`` toma el
    agua a la temperatura de bulbo húmedo del aire que entra (el agua que
    recircula llega a esa temperatura). Con vapor, saturado a ``T_water_K``.
    """

    phi_out: float
    water: Literal["liquid", "vapor"] = "liquid"
    T_water_K: float | None = None


@dataclass(frozen=True)
class AdiabaticMixing:
    """Mezcla adiabática con otra corriente (§14.12.6), dada por T, φ y caudal."""

    T_K: float
    phi: float
    flow: AirFlow


ProcessSpec = (
    SensibleProcess
    | HeatingHumidification
    | CoolingDehumidification
    | AdiabaticHumidification
    | AdiabaticMixing
)


@dataclass(frozen=True)
class HvacInputs:
    """Un tren de procesos sobre una corriente de aire.

    ``dead`` es el ambiente para la exergía (``None``: el aire que entra a
    la presión del tren). ``altitude_m`` solo informa de dónde salió la presión.
    """

    p_Pa: float
    inlet: AirInlet
    processes: tuple[ProcessSpec, ...]
    dead: DeadState | None = None
    altitude_m: float | None = None

    @property
    def dead_state(self) -> DeadState:
        """El ambiente de la exergía."""
        if self.dead is not None:
            return self.dead
        return DeadState(self.inlet.T_K, max(self.inlet.phi, 1e-6), self.p_Pa)


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HvacStream:
    """Un estado numerado del tren, con su caudal de aire seco."""

    number: int
    label: str
    state: MoistAirState
    m_dry_air_kg_s: float

    @property
    def V_m3_s(self) -> float:
        """Caudal volumétrico en este estado: V̇ = ṁ_a·v."""
        return self.m_dry_air_kg_s * self.state.v_m3_per_kg

    @property
    def m_vapor_kg_s(self) -> float:
        """Caudal de vapor de agua: ṁ_v = ṁ_a·ω."""
        return self.m_dry_air_kg_s * self.state.omega


ProcessKind = Literal[
    "heating",
    "cooling",
    "heating_humidification",
    "cooling_dehumidification",
    "adiabatic_humidification",
    "mixing",
]

_PROCESS_NAMES: dict[ProcessKind, str] = {
    "heating": "Calentamiento sensible",
    "cooling": "Enfriamiento sensible",
    "heating_humidification": "Calentamiento con humidificación",
    "cooling_dehumidification": "Enfriamiento y deshumidificación",
    "adiabatic_humidification": "Humidificación adiabática",
    "mixing": "Mezcla adiabática",
}

#: Sección del vademecum de cada proceso.
PROCESS_SECTIONS: dict[ProcessKind, str] = {
    "heating": "§14.12.1",
    "cooling": "§14.12.1",
    "heating_humidification": "§14.12.2",
    "cooling_dehumidification": "§14.12.3",
    "adiabatic_humidification": "§14.12.4",
    "mixing": "§14.12.6",
}


@dataclass(frozen=True)
class ProcessResult:
    """Un proceso resuelto: entradas, salida, calor, agua y exergía (todo el caudal)."""

    index: int
    kind: ProcessKind
    spec: ProcessSpec
    inlet: HvacStream
    outlet: HvacStream
    other: HvacStream | None
    Q_W: float
    m_water_kg_s: float
    water_phase: WaterPhase | None
    T_water_K: float | None
    h_water_J_per_kg: float | None
    T_source_K: float | None
    X_heat_W: float
    X_water_W: float
    X_destroyed_W: float

    @property
    def name(self) -> str:
        """Nombre del proceso (con el agua, si es una humidificación adiabática)."""
        base = _PROCESS_NAMES[self.kind]
        if self.kind == "adiabatic_humidification":
            return f"{base} con {'vapor' if self.water_phase == 'vapor' else 'agua'}"
        return base

    @property
    def label(self) -> str:
        """«Proceso k: nombre (a → b)»."""
        inlets = f"{self.inlet.number}"
        if self.other is not None:
            inlets += f" + {self.other.number}"
        return f"{self.name} ({inlets} → {self.outlet.number})"

    @property
    def m_dry_air_kg_s(self) -> float:
        """Caudal de aire seco de la salida."""
        return self.outlet.m_dry_air_kg_s

    @property
    def q_J_per_kg(self) -> float:
        """Calor por kg de aire seco que sale del proceso."""
        return self.Q_W / self.m_dry_air_kg_s


@dataclass(frozen=True)
class HvacResult:
    """El tren resuelto: estados numerados y procesos."""

    inputs: HvacInputs
    streams: tuple[HvacStream, ...]
    processes: tuple[ProcessResult, ...]

    @property
    def inlet(self) -> HvacStream:
        """El estado 1."""
        return self.streams[0]

    @property
    def outlet(self) -> HvacStream:
        """El último estado (sale del tren)."""
        return self.processes[-1].outlet if self.processes else self.streams[0]

    @property
    def heating_W(self) -> float:
        """Calor total entregado al aire."""
        return sum(max(p.Q_W, 0.0) for p in self.processes)

    @property
    def cooling_W(self) -> float:
        """Calor total quitado al aire."""
        return sum(max(-p.Q_W, 0.0) for p in self.processes)

    @property
    def water_added_kg_s(self) -> float:
        """Agua agregada (humidificadores)."""
        return sum(max(p.m_water_kg_s, 0.0) for p in self.processes)

    @property
    def water_removed_kg_s(self) -> float:
        """Agua condensada (serpentines)."""
        return sum(max(-p.m_water_kg_s, 0.0) for p in self.processes)

    @property
    def X_destroyed_W(self) -> float:
        """Exergía destruida en todo el tren."""
        return sum(p.X_destroyed_W for p in self.processes)

    @property
    def dead(self) -> DeadState:
        """El ambiente de la exergía."""
        return self.inputs.dead_state

    def stream(self, number: int) -> HvacStream:
        """El estado con ese número."""
        for s in self.streams:
            if s.number == number:
                return s
        raise KeyError(number)


def default_source_temperature(spec: ProcessSpec, inlet: MoistAirState) -> float | None:
    """Temperatura de la fuente o del serpentín cuando no se da (para la exergía).

    - Calentamiento: agua caliente a 60 °C, o 10 K por encima de la salida si
      el aire sale más caliente.
    - Enfriamiento sensible: un serpentín 5 K más frío que la salida, pero no
      por debajo del punto de rocío (si no, condensaría).
    - Serpentín con deshumidificación: la temperatura del condensado.
    """
    if isinstance(spec, SensibleProcess):
        if spec.T_source_K is not None:
            return spec.T_source_K
        if spec.T_out_K >= inlet.T_K:
            return max(_DEFAULT_HOT_SOURCE_K, spec.T_out_K + 10.0)
        T_dp = inlet.T_dp_K if inlet.T_dp_K is not None else T_MIN_K
        return max(spec.T_out_K - 5.0, T_dp)
    if isinstance(spec, HeatingHumidification):
        if spec.T_source_K is not None:
            return spec.T_source_K
        return max(_DEFAULT_HOT_SOURCE_K, spec.T_out_K + 10.0)
    if isinstance(spec, CoolingDehumidification):
        return spec.T_water_K if spec.T_water_K is not None else spec.T_out_K
    return None


def _prefix(index: int, kind: ProcessKind) -> str:
    return f"Proceso {index} ({_PROCESS_NAMES[kind].lower()}): "


def _check_T(T_K: float, what: str, prefix: str) -> None:
    if not (T_MIN_K <= T_K <= T_MAX_K):
        raise ValueError(f"{prefix}{what} tiene que estar entre −50 y 200 °C ({_degC(T_K)}).")


def _check_phi(phi: float, what: str, prefix: str) -> None:
    if not (0.0 < phi <= 1.0):
        raise ValueError(f"{prefix}{what} tiene que estar entre 0 y 100 % ({_pct(phi)}).")


def _state_T_phi(p: float, T: float, phi: float, prefix: str) -> MoistAirState:
    try:
        return state_from_T_phi(p, T, phi)
    except ValueError as exc:
        raise ValueError(f"{prefix}{exc}") from None


def _heat_exergy(Q: float, T_b: float | None, T0: float) -> float:
    """Exergía del calor que recibe el aire desde una frontera a T_b: Q̇·(1 − T₀/T_b)."""
    if T_b is None or Q == 0.0:
        return 0.0
    return Q * (1.0 - T0 / T_b)


def _water_temperature_check(
    T_w: float, phase: Literal["vapor", "liquid"], p: float, prefix: str
) -> None:
    T_boil = saturation_temperature(p)
    if phase == "liquid":
        if not (T_TRIPLE_K <= T_w <= T_boil):
            raise ValueError(
                f"{prefix}el agua líquida tiene que estar entre 0,01 °C y {_degC(T_boil)} (la de "
                f"ebullición a la presión del aire); pusiste {_degC(T_w)}."
            )
    elif not (T_boil - 1e-6 <= T_w <= T_MAX_K):
        raise ValueError(
            f"{prefix}el vapor tiene que entrar a una presión mayor que la del aire: saturado, "
            f"por lo menos a {_degC(T_boil)} (y hasta 200 °C); pusiste {_degC(T_w)}."
        )


def _sensible(
    index: int, spec: SensibleProcess, inlet: HvacStream, number: int, dead: DeadState
) -> ProcessResult:
    s1 = inlet.state
    kind: ProcessKind = "heating" if spec.T_out_K >= s1.T_K else "cooling"
    prefix = _prefix(index, kind)
    _check_T(spec.T_out_K, "la temperatura de salida", prefix)
    if abs(spec.T_out_K - s1.T_K) < 1e-3:
        raise ValueError(f"{prefix}la temperatura de salida es igual a la de entrada.")
    if kind == "cooling" and s1.T_dp_K is not None and spec.T_out_K < s1.T_dp_K:
        raise ValueError(
            f"{prefix}el aire llegaría a su punto de rocío ({_degC(s1.T_dp_K)}) y empezaría a "
            "condensar: usá «Enfriamiento y deshumidificación»."
        )
    s2 = state_from_T_omega(s1.p_Pa, spec.T_out_K, s1.omega)
    T_b = default_source_temperature(spec, s1)
    assert T_b is not None
    if kind == "heating" and T_b <= spec.T_out_K:
        raise ValueError(
            f"{prefix}la fuente de calor ({_degC(T_b)}) tiene que estar más caliente que el aire "
            f"que sale ({_degC(spec.T_out_K)})."
        )
    if kind == "cooling":
        if T_b >= spec.T_out_K:
            raise ValueError(
                f"{prefix}el serpentín ({_degC(T_b)}) tiene que estar más frío que el aire que "
                f"sale ({_degC(spec.T_out_K)})."
            )
        if s1.T_dp_K is not None and T_b < s1.T_dp_K - 1e-9:
            raise ValueError(
                f"{prefix}con el serpentín a {_degC(T_b)}, por debajo del punto de rocío del aire "
                f"({_degC(s1.T_dp_K)}), el vapor condensaría sobre él: no sería un enfriamiento "
                "sensible."
            )
    m = inlet.m_dry_air_kg_s
    Q = m * (s2.h_J_per_kg - s1.h_J_per_kg)
    out = HvacStream(number, f"salida del {_PROCESS_NAMES[kind].lower()}", s2, m)
    X_Q = _heat_exergy(Q, T_b, dead.T0_K)
    X_dest = m * (s1.psi(dead) - s2.psi(dead)) + X_Q
    return ProcessResult(
        index, kind, spec, inlet, out, None, Q, 0.0, None, None, None, T_b, X_Q, 0.0, X_dest
    )


def _heating_humidification(
    index: int, spec: HeatingHumidification, inlet: HvacStream, number: int, dead: DeadState
) -> ProcessResult:
    s1 = inlet.state
    kind: ProcessKind = "heating_humidification"
    prefix = _prefix(index, kind)
    _check_T(spec.T_out_K, "la temperatura de salida", prefix)
    _check_phi(spec.phi_out, "la humedad relativa de salida", prefix)
    _water_temperature_check(spec.T_water_K, spec.water, s1.p_Pa, prefix)
    s2 = _state_T_phi(s1.p_Pa, spec.T_out_K, spec.phi_out, prefix)
    if s2.omega <= s1.omega:
        raise ValueError(
            f"{prefix}el aire saldría con ω = {s2.omega:.4g} kg/kg, sin más humedad que la que "
            f"trae (ω = {s1.omega:.4g} kg/kg): no hay agua que agregar. Subí φ de salida o usá "
            "otro proceso."
        )
    m = inlet.m_dry_air_kg_s
    m_w = m * (s2.omega - s1.omega)
    h_w = water_enthalpy(spec.T_water_K, spec.water)
    Q = m * (s2.h_J_per_kg - s1.h_J_per_kg) - m_w * h_w
    T_b = default_source_temperature(spec, s1)
    assert T_b is not None
    if Q > 0.0 and T_b <= spec.T_out_K:
        raise ValueError(
            f"{prefix}la fuente de calor ({_degC(T_b)}) tiene que estar más caliente que el aire "
            f"que sale ({_degC(spec.T_out_K)})."
        )
    out = HvacStream(number, "salida del humidificador", s2, m)
    X_Q = _heat_exergy(Q, T_b, dead.T0_K)
    X_w = m_w * water_exergy(spec.T_water_K, spec.water, dead)
    X_dest = m * (s1.psi(dead) - s2.psi(dead)) + X_w + X_Q
    return ProcessResult(
        index,
        kind,
        spec,
        inlet,
        out,
        None,
        Q,
        m_w,
        spec.water,
        spec.T_water_K,
        h_w,
        T_b,
        X_Q,
        X_w,
        X_dest,
    )


def _cooling_dehumidification(
    index: int, spec: CoolingDehumidification, inlet: HvacStream, number: int, dead: DeadState
) -> ProcessResult:
    s1 = inlet.state
    kind: ProcessKind = "cooling_dehumidification"
    prefix = _prefix(index, kind)
    _check_T(spec.T_out_K, "la temperatura de salida", prefix)
    _check_phi(spec.phi_out, "la humedad relativa de salida", prefix)
    if spec.T_out_K >= s1.T_K:
        raise ValueError(
            f"{prefix}el aire tiene que salir más frío de lo que entra ({_degC(s1.T_K)})."
        )
    T_w = spec.T_water_K if spec.T_water_K is not None else spec.T_out_K
    if T_w < T_TRIPLE_K:
        raise ValueError(
            f"{prefix}con la superficie a {_degC(T_w)} el agua se congela sobre el serpentín "
            "(escarcha): el modelo pide el condensado líquido, a 0,01 °C o más."
        )
    if T_w > spec.T_out_K + 1e-9:
        raise ValueError(
            f"{prefix}el condensado sale a la temperatura de la superficie fría ({_degC(T_w)}), "
            f"que no puede ser mayor que la del aire que sale ({_degC(spec.T_out_K)})."
        )
    s2 = _state_T_phi(s1.p_Pa, spec.T_out_K, spec.phi_out, prefix)
    if s2.omega >= s1.omega:
        raise ValueError(
            f"{prefix}con esa salida el aire no pierde humedad (ω pasaría de {s1.omega:.4g} a "
            f"{s2.omega:.4g} kg/kg): no condensa. Es un enfriamiento sensible, o bajá φ o T de "
            "salida."
        )
    m = inlet.m_dry_air_kg_s
    m_w = m * (s1.omega - s2.omega)
    h_w = water_enthalpy(T_w, "liquid")
    Q = m * ((s2.h_J_per_kg - s1.h_J_per_kg) + (s1.omega - s2.omega) * h_w)
    out = HvacStream(number, "salida del serpentín", s2, m)
    X_Q = _heat_exergy(Q, T_w, dead.T0_K)
    X_w = -m_w * water_exergy(T_w, "liquid", dead)
    X_dest = m * (s1.psi(dead) - s2.psi(dead)) + X_w + X_Q
    return ProcessResult(
        index, kind, spec, inlet, out, None, Q, -m_w, "liquid", T_w, h_w, T_w, X_Q, X_w, X_dest
    )


def _adiabatic_humidification(
    index: int, spec: AdiabaticHumidification, inlet: HvacStream, number: int, dead: DeadState
) -> ProcessResult:
    s1 = inlet.state
    kind: ProcessKind = "adiabatic_humidification"
    prefix = _prefix(index, kind)
    _check_phi(spec.phi_out, "la humedad relativa de salida", prefix)
    if spec.phi_out <= s1.phi + 1e-9:
        raise ValueError(
            f"{prefix}el aire ya entra con φ = {_pct(s1.phi)}: para humidificar, la salida "
            "tiene que tener una humedad relativa mayor."
        )
    if spec.water == "liquid":
        T_w = spec.T_water_K if spec.T_water_K is not None else max(s1.T_wb_K, T_TRIPLE_K)
    else:
        T_w = spec.T_water_K if spec.T_water_K is not None else saturation_temperature(s1.p_Pa)
    _water_temperature_check(T_w, spec.water, s1.p_Pa, prefix)
    h_w = water_enthalpy(T_w, spec.water)
    p = s1.p_Pa
    phi2 = spec.phi_out

    def omega_at(T: float) -> float:
        return humidity_ratio(phi2 * saturation_pressure(T), p)

    def residual(T: float) -> float:
        omega2 = omega_at(T)
        return moist_enthalpy(T, omega2) - s1.h_J_per_kg - (omega2 - s1.omega) * h_w

    T_hi = T_MAX_K
    if phi2 * saturation_pressure(T_hi) >= 0.99 * p:
        T_hi = brentq(lambda T: phi2 * saturation_pressure(T) - 0.99 * p, T_MIN_K, T_MAX_K)
    grid = np.linspace(max(T_MIN_K, s1.T_wb_K - 20.0), T_hi, 200)
    values = [residual(float(T)) for T in grid]
    root = None
    for k in range(len(grid) - 1):
        if values[k] == 0.0:
            root = float(grid[k])
            break
        if values[k] * values[k + 1] < 0.0:
            root = float(brentq(residual, float(grid[k]), float(grid[k + 1]), xtol=1e-10))
            break
    if root is None:
        raise ValueError(f"{prefix}no hay una salida con φ = {_pct(phi2)} para esa entrada.")
    s2 = state_from_T_omega(p, root, omega_at(root))
    if s2.omega <= s1.omega:
        raise ValueError(f"{prefix}con esos datos el aire no gana humedad.")
    m = inlet.m_dry_air_kg_s
    m_w = m * (s2.omega - s1.omega)
    out = HvacStream(number, "salida del humidificador", s2, m)
    X_w = m_w * water_exergy(T_w, spec.water, dead)
    X_dest = m * (s1.psi(dead) - s2.psi(dead)) + X_w
    return ProcessResult(
        index, kind, spec, inlet, out, None, 0.0, m_w, spec.water, T_w, h_w, None, 0.0, X_w, X_dest
    )


def _mixing(
    index: int,
    spec: AdiabaticMixing,
    inlet: HvacStream,
    number: int,
    dead: DeadState,
) -> tuple[ProcessResult, HvacStream]:
    kind: ProcessKind = "mixing"
    prefix = _prefix(index, kind)
    _check_T(spec.T_K, "la temperatura de la otra corriente", prefix)
    _check_phi(spec.phi, "la humedad relativa de la otra corriente", prefix)
    s1 = inlet.state
    s2 = _state_T_phi(s1.p_Pa, spec.T_K, spec.phi, prefix)
    m1 = inlet.m_dry_air_kg_s
    m2 = spec.flow.dry_air(s2)
    other = HvacStream(number, "corriente que se mezcla", s2, m2)
    m3 = m1 + m2
    omega3 = (m1 * s1.omega + m2 * s2.omega) / m3
    h3 = (m1 * s1.h_J_per_kg + m2 * s2.h_J_per_kg) / m3
    T3 = dry_bulb_from_h_omega(h3, omega3)
    try:
        s3 = state_from_T_omega(s1.p_Pa, T3, omega3)
    except ValueError:
        phi3 = _phi_of(omega3, T3, s1.p_Pa)
        raise ValueError(
            f"{prefix}la mezcla quedaría a {_degC(T3)} con φ = {_pct(phi3)}: sobresaturada. Se "
            "formaría niebla (el vapor de más condensa en gotitas) y el modelo no la contempla."
        ) from None
    out = HvacStream(number + 1, "mezcla", s3, m3)
    X_dest = m1 * s1.psi(dead) + m2 * s2.psi(dead) - m3 * s3.psi(dead)
    result = ProcessResult(
        index, kind, spec, inlet, out, other, 0.0, 0.0, None, None, None, None, 0.0, 0.0, X_dest
    )
    return result, other


def _phi_of(omega: float, T_K: float, p_Pa: float) -> float:
    """φ que tendría el aire con ω a T (puede pasar de 1: sobresaturado)."""
    return vapor_pressure(omega, p_Pa) / saturation_pressure(T_K)


def solve_hvac(inputs: HvacInputs) -> HvacResult:
    """Resuelve el tren de procesos (vademecum §14.12; Cengel §14-7).

    Numera los estados en el sentido del flujo: 1 es la entrada, cada proceso
    suma su salida y una mezcla numera antes la corriente que se suma (como el
    ejemplo 14-8 de Cengel). Los errores nombran el proceso y explican qué dato
    no tiene sentido físico.
    """
    if not inputs.processes:
        raise ValueError("Agregá al menos un proceso.")
    if len(inputs.processes) > MAX_PROCESSES:
        raise ValueError(f"Como mucho {MAX_PROCESSES} procesos por tren.")
    p = inputs.p_Pa
    inlet = inputs.inlet
    if not (0.0 < inlet.phi <= 1.0):
        raise ValueError("La humedad relativa del aire que entra tiene que estar entre 0 y 100 %.")
    s1 = _state_T_phi(p, inlet.T_K, inlet.phi, "Aire que entra: ")
    dead = inputs.dead_state
    if abs(dead.p0_Pa - p) > 1e-6 * p:
        dead = DeadState(dead.T0_K, dead.phi0, p)
    current = HvacStream(1, "entrada", s1, inlet.flow.dry_air(s1))
    streams: list[HvacStream] = [current]
    results: list[ProcessResult] = []
    number = 2
    for index, spec in enumerate(inputs.processes, start=1):
        if isinstance(spec, AdiabaticMixing):
            result, other = _mixing(index, spec, current, number, dead)
            streams += [other, result.outlet]
            number += 2
        else:
            if isinstance(spec, SensibleProcess):
                result = _sensible(index, spec, current, number, dead)
            elif isinstance(spec, HeatingHumidification):
                result = _heating_humidification(index, spec, current, number, dead)
            elif isinstance(spec, CoolingDehumidification):
                result = _cooling_dehumidification(index, spec, current, number, dead)
            elif isinstance(spec, AdiabaticHumidification):
                result = _adiabatic_humidification(index, spec, current, number, dead)
            else:  # pragma: no cover - tipos cerrados
                raise TypeError(f"Proceso desconocido: {spec!r}")
            streams.append(result.outlet)
            number += 1
        results.append(result)
        current = result.outlet
    return HvacResult(replace(inputs, dead=dead), tuple(streams), tuple(results))


# ---------------------------------------------------------------------
# Notas didácticas
# ---------------------------------------------------------------------


def hvac_notes(result: HvacResult) -> list[str]:
    """Observaciones sobre el tren (markdown)."""
    notes: list[str] = []
    for proc in result.processes:
        s1, s2 = proc.inlet.state, proc.outlet.state
        head = f"**Proceso {proc.index}** ({proc.name.lower()}): "
        if proc.kind == "heating" and s2.phi < 0.5 * s1.phi:
            notes.append(
                head + f"la humedad relativa baja de {_pct(s1.phi)} a {_pct(s2.phi)} aunque ω no "
                "cambia: el aire caliente admite más vapor. Por eso la calefacción reseca el "
                "ambiente."
            )
        if proc.kind == "cooling_dehumidification" and s2.phi >= 0.999:
            notes.append(
                head + "el aire sale saturado (como en Cengel). En un serpentín real sale con "
                "90–95 %: una parte del aire pasa sin tocar los tubos fríos."
            )
        if proc.kind == "cooling_dehumidification":
            latent = (
                proc.m_dry_air_kg_s
                * (s1.omega - s2.omega)
                * (s1.h_vapor - (proc.h_water_J_per_kg or 0.0))
            )
            if proc.Q_W < 0.0:
                share = latent / -proc.Q_W
                notes.append(
                    head + f"cerca del {100 * share:.0f} % del calor que se quita es latente "
                    "(condensar el vapor); el resto baja la temperatura del aire."
                )
        if proc.kind == "heating_humidification" and proc.Q_W > 0.0 and proc.water_phase == "vapor":
            h_needed = (s2.h_J_per_kg - s1.h_J_per_kg) / (s2.omega - s1.omega)
            T_needed = _steam_temperature_for(h_needed, s1.p_Pa)
            extra = (
                f" (vapor sobrecalentado a unos {_degC(T_needed)} a la presión del aire)"
                if T_needed is not None
                else ""
            )
            notes.append(
                head + "el vapor solo no alcanza para llegar a esa temperatura: hace falta además "
                f"calor. Sin calor, el vapor tendría que entrar con h_w = {h_needed / 1e3:.4g} "
                f"kJ/kg{extra}."
            )
        if proc.kind == "adiabatic_humidification" and proc.water_phase == "liquid":
            notes.append(
                head + f"el agua que se evapora enfría el aire de {_degC(s1.T_K)} a "
                f"{_degC(s2.T_K)} casi sobre la línea de bulbo húmedo constante (h casi no "
                f"cambia): como mucho bajaría hasta T_bh = {_degC(s1.T_wb_K)}."
            )
        if proc.kind == "mixing" and s2.phi > 0.9:
            notes.append(
                head + f"la mezcla sale con φ = {_pct(s2.phi)}, cerca de la saturación: con aire "
                "exterior más frío o más húmedo se formaría niebla."
            )
    return notes


def _steam_temperature_for(h: float, p_Pa: float) -> float | None:
    """T del vapor a la presión p con entalpía h (sobrecalentado), o ``None`` si no hay."""
    from CoolProp.CoolProp import PropsSI

    try:
        h_g = water_enthalpy(saturation_temperature(p_Pa), "vapor")
        if h <= h_g:
            return None
        return float(PropsSI("T", "P", p_Pa, "H", h, "Water"))
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------

_PER_MIN = 1.0 / 60.0

#: Ejemplos de procesos (Cengel §14-7 y dos de climatización).
HVAC_EXAMPLES: dict[str, HvacInputs] = {
    "Cengel 14-5: calentamiento y humidificación": HvacInputs(
        100.0e3,
        AirInlet(283.15, 0.30, AirFlow(45.0 * _PER_MIN)),
        (
            SensibleProcess(295.15, T_source_K=333.15),
            HeatingHumidification(298.15, 0.60, "vapor", 373.15, T_source_K=333.15),
        ),
    ),
    "Cengel 14-6: enfriamiento y deshumidificación": HvacInputs(
        P_SEA_LEVEL_PA,
        AirInlet(303.15, 0.80, AirFlow(10.0 * _PER_MIN)),
        (CoolingDehumidification(287.15, 1.0, 287.15),),
    ),
    "Cengel 14-7: enfriador evaporativo": HvacInputs(
        14.7 * 6894.757293168361,
        AirInlet(308.15, 0.20, AirFlow(0.45359237, "dry_air")),
        (AdiabaticHumidification(0.80, "liquid"),),
    ),
    "Cengel 14-8: mezcla con aire exterior": HvacInputs(
        P_SEA_LEVEL_PA,
        AirInlet(287.15, 1.0, AirFlow(50.0 * _PER_MIN)),
        (AdiabaticMixing(305.15, 0.60, AirFlow(20.0 * _PER_MIN)),),
        dead=DeadState(305.15, 0.60),
    ),
    "Verano: mezcla, serpentín y recalentamiento": HvacInputs(
        P_SEA_LEVEL_PA,
        AirInlet(298.15, 0.50, AirFlow(2.0)),
        (
            AdiabaticMixing(308.15, 0.40, AirFlow(0.6)),
            CoolingDehumidification(286.15, 0.95, 284.15),
            SensibleProcess(290.15, T_source_K=323.15),
        ),
        dead=DeadState(308.15, 0.40),
    ),
    "Invierno en Bariloche: calentamiento y enfriador evaporativo": HvacInputs(
        pressure_from_altitude(900.0),
        AirInlet(273.15, 0.80, AirFlow(1.0)),
        (
            SensibleProcess(308.15, T_source_K=343.15),
            AdiabaticHumidification(0.40, "liquid", 288.15),
        ),
        dead=DeadState(273.15, 0.80, pressure_from_altitude(900.0)),
        altitude_m=900.0,
    ),
}

#: Qué muestra cada ejemplo (y lo que da el libro).
HVAC_EXAMPLE_NOTES: dict[str, str] = {
    "Cengel 14-5: calentamiento y humidificación": (
        "Aire exterior a 10 °C y 30 % (45 m³/min, 100 kPa) que se calienta a 22 °C y se "
        "humidifica con vapor hasta 25 °C y 60 %. Cengel da Q̇ = 673 kJ/min (11,2 kW) y "
        "ṁ_w = 0,539 kg/min: redondea las h de la carta. El libro no hace el balance del "
        "humidificador; con vapor saturado a 100 °C hace falta además un poco de calor."
    ),
    "Cengel 14-6: enfriamiento y deshumidificación": (
        "Un aire acondicionado de ventana toma aire a 30 °C y 80 % (10 m³/min) y lo devuelve "
        "saturado a 14 °C; el condensado sale a 14 °C. Cengel da 511 kJ/min (8,5 kW) y "
        "0,131 kg/min de agua."
    ),
    "Cengel 14-7: enfriador evaporativo": (
        "Un enfriador evaporativo con aire a 14,7 psia, 95 °F y 20 % que sale con 80 %. Cengel "
        "da T₂ ≈ 70 °F y la mínima, T_bh = 66 °F. Mirá el proceso en la carta: sigue la línea "
        "de bulbo húmedo."
    ),
    "Cengel 14-8: mezcla con aire exterior": (
        "Aire saturado a 14 °C que sale del serpentín (50 m³/min) se mezcla con aire exterior a "
        "32 °C y 60 % (20 m³/min). Cengel da ω₃ = 0,0122, φ₃ = 89 %, T₃ = 19,0 °C y "
        "V̇₃ = 70,1 m³/min. En la carta, 3 cae sobre la recta 1–2."
    ),
    "Verano: mezcla, serpentín y recalentamiento": (
        "Un equipo central: el aire de retorno (25 °C, 50 %) se mezcla con aire exterior de "
        "verano (35 °C, 40 %), el serpentín lo enfría y lo seca, y una batería lo recalienta "
        "para que no entre al local saturado. Recalentar gasta energía, pero controla la "
        "humedad."
    ),
    "Invierno en Bariloche: calentamiento y enfriador evaporativo": (
        "A 900 m (0,91 bar): aire exterior a 0 °C y 80 % se calienta a 35 °C (su φ cae a "
        "6 %) y un humidificador de agua lo lleva a 40 %, enfriándolo. La exergía muestra "
        "cuánto se pierde al calentar con agua a 70 °C."
    ),
}


# ---------------------------------------------------------------------
# Barrido: el aire que entra
# ---------------------------------------------------------------------

HvacSweepParameter = Literal["T_in", "phi_in"]


@dataclass(frozen=True)
class HvacSweepPoint:
    """Un punto del barrido: el valor (SI) y los totales del tren."""

    value: float
    heating_W: float
    cooling_W: float
    water_added_kg_s: float
    water_removed_kg_s: float
    T_out_K: float
    phi_out: float
    X_destroyed_W: float


def default_hvac_sweep_values(inputs: HvacInputs, parameter: HvacSweepParameter) -> np.ndarray:
    """Valores del barrido alrededor del aire que entra."""
    if parameter == "T_in":
        T = inputs.inlet.T_K
        return np.linspace(max(T_MIN_K, T - 15.0), min(T_MAX_K, T + 15.0), 31)
    return np.linspace(0.05, 1.0, 20)


def hvac_sweep(
    inputs: HvacInputs, parameter: HvacSweepParameter, values: Sequence[float]
) -> list[HvacSweepPoint]:
    """Resuelve el tren para cada valor del aire que entra; omite los que no tienen sentido."""
    points: list[HvacSweepPoint] = []
    for value in values:
        inlet = inputs.inlet
        inlet = (
            replace(inlet, T_K=float(value))
            if parameter == "T_in"
            else replace(inlet, phi=float(value))
        )
        # El ambiente sigue siendo el de los datos (si era el aire que entra, queda fijo).
        trial = replace(inputs, inlet=inlet, dead=inputs.dead_state)
        try:
            r = solve_hvac(trial)
        except ValueError:
            continue
        points.append(
            HvacSweepPoint(
                float(value),
                r.heating_W,
                r.cooling_W,
                r.water_added_kg_s,
                r.water_removed_kg_s,
                r.outlet.state.T_K,
                r.outlet.state.phi,
                r.X_destroyed_W,
            )
        )
    return points


# ---------------------------------------------------------------------
# Torre de enfriamiento húmeda (vademecum §14.12.7)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CoolingTowerInputs:
    """Torre de enfriamiento húmeda: agua caliente que baja y aire que sube.

    El agua entra con ``m_water_in_kg_s`` a ``T_water_in_K`` y sale a
    ``T_water_out_K``; el aire entra a (T, φ) y sale a (T, φ). ``dead`` es el
    ambiente para la exergía (``None``: el aire que entra).
    """

    p_Pa: float
    m_water_in_kg_s: float
    T_water_in_K: float
    T_water_out_K: float
    T_air_in_K: float
    phi_air_in: float
    T_air_out_K: float
    phi_air_out: float = 1.0
    dead: DeadState | None = None
    altitude_m: float | None = None

    @property
    def dead_state(self) -> DeadState:
        """El ambiente de la exergía."""
        if self.dead is not None:
            return self.dead
        return DeadState(self.T_air_in_K, self.phi_air_in, self.p_Pa)


@dataclass(frozen=True)
class CoolingTowerResult:
    """La torre resuelta (balances de masa y energía del vademecum §14.12.7)."""

    inputs: CoolingTowerInputs
    air_in: MoistAirState
    air_out: MoistAirState
    m_dry_air_kg_s: float
    h_water_in: float
    h_water_out: float
    X_destroyed_W: float

    @property
    def makeup_kg_s(self) -> float:
        """Agua evaporada = reposición: ṁ_a·(ω₂ − ω₁)."""
        return self.m_dry_air_kg_s * (self.air_out.omega - self.air_in.omega)

    @property
    def m_water_out_kg_s(self) -> float:
        """Agua que sale por abajo: ṁ_w,1 − agua evaporada."""
        return self.inputs.m_water_in_kg_s - self.makeup_kg_s

    @property
    def V_air_in_m3_s(self) -> float:
        """Caudal volumétrico de aire que entra: ṁ_a·v₁."""
        return self.m_dry_air_kg_s * self.air_in.v_m3_per_kg

    @property
    def Q_W(self) -> float:
        """Calor que el agua le entrega al aire: ṁ_a·(h₂ − h₁)."""
        return self.m_dry_air_kg_s * (self.air_out.h_J_per_kg - self.air_in.h_J_per_kg)

    @property
    def range_K(self) -> float:
        """Rango: cuánto se enfría el agua."""
        return self.inputs.T_water_in_K - self.inputs.T_water_out_K

    @property
    def approach_K(self) -> float:
        """Aproximación: cuánto le falta al agua para llegar al bulbo húmedo del aire."""
        return self.inputs.T_water_out_K - self.air_in.T_wb_K

    @property
    def effectiveness(self) -> float:
        """Efectividad: rango / (T_agua,ent − T_bh,aire), el máximo enfriamiento posible."""
        return self.range_K / (self.inputs.T_water_in_K - self.air_in.T_wb_K)

    @property
    def water_to_air(self) -> float:
        """Relación L/G: kg de agua por kg de aire seco."""
        return self.inputs.m_water_in_kg_s / self.m_dry_air_kg_s

    @property
    def evaporated_fraction(self) -> float:
        """Fracción del agua que se evapora."""
        return self.makeup_kg_s / self.inputs.m_water_in_kg_s

    @property
    def dead(self) -> DeadState:
        """El ambiente de la exergía."""
        return self.inputs.dead_state


def solve_cooling_tower(inputs: CoolingTowerInputs) -> CoolingTowerResult:
    """Torre de enfriamiento húmeda (vademecum §14.12.7; Cengel §14-7, ejemplo 14-9).

    Masa de aire seco y de agua y energía (adiabática hacia el ambiente):
    ṁ_a = ṁ_w,1·(h_w,1 − h_w,2) / [(h₂ − h₁) − (ω₂ − ω₁)·h_w,2].
    """
    p = inputs.p_Pa
    if not inputs.m_water_in_kg_s > 0.0:
        raise ValueError("El caudal de agua caliente tiene que ser positivo.")
    for T, what in (
        (inputs.T_water_in_K, "La temperatura del agua que entra"),
        (inputs.T_water_out_K, "La temperatura del agua que sale"),
    ):
        if not (T_TRIPLE_K <= T <= saturation_temperature(p)):
            raise ValueError(
                f"{what} tiene que estar entre 0,01 °C y la de ebullición a esa presión "
                f"({_degC(saturation_temperature(p))}); pusiste {_degC(T)}."
            )
    if inputs.T_water_out_K >= inputs.T_water_in_K:
        raise ValueError("El agua tiene que salir de la torre más fría de lo que entra.")
    air_in = _state_T_phi(p, inputs.T_air_in_K, inputs.phi_air_in, "Aire que entra: ")
    air_out = _state_T_phi(p, inputs.T_air_out_K, inputs.phi_air_out, "Aire que sale: ")
    if inputs.T_water_out_K <= air_in.T_wb_K:
        raise ValueError(
            f"El agua no puede enfriarse hasta {_degC(inputs.T_water_out_K)}: el límite es la "
            f"temperatura de bulbo húmedo del aire que entra ({_degC(air_in.T_wb_K)}). La "
            "diferencia (la aproximación) tiene que ser positiva."
        )
    if inputs.T_air_out_K > inputs.T_water_in_K:
        raise ValueError(
            f"El aire no puede salir más caliente ({_degC(inputs.T_air_out_K)}) que el agua que "
            f"lo calienta ({_degC(inputs.T_water_in_K)})."
        )
    if air_out.omega <= air_in.omega:
        raise ValueError(
            "El aire tiene que salir con más humedad de la que entra: en la torre el agua se "
            "evapora."
        )
    h_w1 = water_enthalpy(inputs.T_water_in_K, "liquid")
    h_w2 = water_enthalpy(inputs.T_water_out_K, "liquid")
    denominator = (air_out.h_J_per_kg - air_in.h_J_per_kg) - (air_out.omega - air_in.omega) * h_w2
    if denominator <= 0.0:
        raise ValueError(
            "Con esos estados el aire no se lleva calor del agua (su entalpía casi no sube): "
            "subí la temperatura o la humedad del aire que sale."
        )
    m_a = inputs.m_water_in_kg_s * (h_w1 - h_w2) / denominator
    makeup = m_a * (air_out.omega - air_in.omega)
    if makeup >= inputs.m_water_in_kg_s:
        raise ValueError("Con esos datos se evaporaría toda el agua.")
    dead = inputs.dead_state
    if abs(dead.p0_Pa - p) > 1e-6 * p:
        dead = DeadState(dead.T0_K, dead.phi0, p)
    m_w2 = inputs.m_water_in_kg_s - makeup
    X_dest = (
        m_a * (air_in.psi(dead) - air_out.psi(dead))
        + inputs.m_water_in_kg_s * water_exergy(inputs.T_water_in_K, "liquid", dead)
        - m_w2 * water_exergy(inputs.T_water_out_K, "liquid", dead)
    )
    return CoolingTowerResult(replace(inputs, dead=dead), air_in, air_out, m_a, h_w1, h_w2, X_dest)


def cooling_tower_notes(result: CoolingTowerResult) -> list[str]:
    """Observaciones sobre la torre (markdown)."""
    notes = [
        f"Se evapora el {100 * result.evaporated_fraction:.2g} % del agua y eso alcanza para "
        f"enfriarla {result.range_K:.3g} K: casi todo el calor se va como calor latente del "
        "vapor."
    ]
    if result.approach_K < 3.0:
        notes.append(
            f"La aproximación es de solo {result.approach_K:.2g} K: una torre así necesita mucho "
            "relleno (se acerca al límite del bulbo húmedo)."
        )
    if result.air_out.phi < 0.9:
        notes.append(
            "El aire sale lejos de la saturación: en una torre real suele salir casi saturado."
        )
    notes.append(
        "La reposición real es mayor: además del agua evaporada se pierde agua por arrastre de "
        "gotas y por purga (para que no se concentren las sales), que acá no se cuentan."
    )
    return notes


#: Ejemplos de torre.
TOWER_EXAMPLES: dict[str, CoolingTowerInputs] = {
    "Cengel 14-9: torre de una central": CoolingTowerInputs(
        P_SEA_LEVEL_PA, 100.0, 308.15, 295.15, 293.15, 0.60, 303.15, 1.0
    ),
    "Verano en Buenos Aires: condensador de 100 MW": CoolingTowerInputs(
        P_SEA_LEVEL_PA, 2400.0, 311.15, 301.15, 305.15, 0.45, 308.15, 1.0
    ),
}

#: Qué muestra cada ejemplo.
TOWER_EXAMPLE_NOTES: dict[str, str] = {
    "Cengel 14-9: torre de una central": (
        "El agua de enfriamiento del condensador entra a 35 °C (100 kg/s) y sale a 22 °C; el aire "
        "entra a 1 atm, 20 °C y 60 % y sale saturado a 30 °C. Cengel pide el caudal de aire y "
        "el agua de reposición (1,80 kg/s); lee los estados en la carta."
    ),
    "Verano en Buenos Aires: condensador de 100 MW": (
        "El condensador de una central rechaza unos 100 MW: 2400 kg/s de agua de 38 a 28 °C. "
        "El aire de verano (32 °C, 45 %) sale saturado a 35 °C. Probá con el barrido cómo "
        "cambian el aire y la reposición con la aproximación."
    ),
}

TowerSweepParameter = Literal["T_water_out", "T_air_in", "phi_air_in"]


@dataclass(frozen=True)
class TowerSweepPoint:
    """Un punto del barrido de la torre."""

    value: float
    m_dry_air_kg_s: float
    V_air_in_m3_s: float
    makeup_kg_s: float
    approach_K: float
    effectiveness: float


def default_tower_sweep_values(
    inputs: CoolingTowerInputs, parameter: TowerSweepParameter
) -> np.ndarray:
    """Valores del barrido de la torre."""
    if parameter == "T_water_out":
        air_in = state_from_T_phi(inputs.p_Pa, inputs.T_air_in_K, inputs.phi_air_in)
        lo = air_in.T_wb_K + 0.5
        return np.linspace(lo, inputs.T_water_in_K - 1.0, 25)
    if parameter == "T_air_in":
        return np.linspace(inputs.T_air_in_K - 15.0, inputs.T_air_in_K + 10.0, 26)
    return np.linspace(0.1, 1.0, 19)


def cooling_tower_sweep(
    inputs: CoolingTowerInputs, parameter: TowerSweepParameter, values: Sequence[float]
) -> list[TowerSweepPoint]:
    """Resuelve la torre para cada valor; omite los puntos imposibles."""
    points: list[TowerSweepPoint] = []
    for value in values:
        field = {
            "T_water_out": "T_water_out_K",
            "T_air_in": "T_air_in_K",
            "phi_air_in": "phi_air_in",
        }[parameter]
        trial = replace(inputs, **{field: float(value)}, dead=inputs.dead_state)
        try:
            r = solve_cooling_tower(trial)
        except ValueError:
            continue
        points.append(
            TowerSweepPoint(
                float(value),
                r.m_dry_air_kg_s,
                r.V_air_in_m3_s,
                r.makeup_kg_s,
                r.approach_K,
                r.effectiveness,
            )
        )
    return points


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _value(value_si: float | None, kind: QuantityKind, system: UnitSystem) -> Any:
    if value_si is None:
        return None
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def _stream_dict(s: HvacStream, system: UnitSystem, dead: DeadState) -> dict[str, Any]:
    st = s.state
    return {
        "estado": s.number,
        "descripcion": s.label,
        "T": _value(st.T_K, "temperature", system),
        "phi": st.phi,
        "omega": st.omega,
        "h": _value(st.h_J_per_kg, "specific_enthalpy", system),
        "v": _value(st.v_m3_per_kg, "specific_volume", system),
        "T_bulbo_humedo": _value(st.T_wb_K, "temperature", system),
        "T_punto_de_rocio": _value(st.T_dp_K, "temperature", system),
        "psi": _value(st.psi(dead), "specific_enthalpy", system),
        "m_aire_seco": _value(s.m_dry_air_kg_s, "mass_flow", system),
        "V": _value(s.V_m3_s, "volume_flow", system),
    }


def hvac_to_dict(result: HvacResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado del tren serializable a JSON (valores en ``system``)."""
    dead = result.dead
    return {
        "modelo": "gas ideal con c_p constante (vademecum §14)",
        "p": _value(result.inputs.p_Pa, "pressure", system),
        "altura_m": result.inputs.altitude_m,
        "ambiente": {
            "T0": _value(dead.T0_K, "temperature", system),
            "phi0": dead.phi0,
        },
        "estados": [_stream_dict(s, system, dead) for s in result.streams],
        "procesos": [
            {
                "proceso": p.index,
                "nombre": p.name,
                "estados": p.label,
                "Q": _value(p.Q_W, "power", system),
                "agua": _value(p.m_water_kg_s, "mass_flow", system),
                "T_agua": _value(p.T_water_K, "temperature", system),
                "h_agua": _value(p.h_water_J_per_kg, "specific_enthalpy", system),
                "T_fuente": _value(p.T_source_K, "temperature", system),
                "exergia_del_calor": _value(p.X_heat_W, "power", system),
                "exergia_destruida": _value(p.X_destroyed_W, "power", system),
            }
            for p in result.processes
        ],
        "calor_entregado": _value(result.heating_W, "power", system),
        "calor_quitado": _value(result.cooling_W, "power", system),
        "agua_agregada": _value(result.water_added_kg_s, "mass_flow", system),
        "agua_condensada": _value(result.water_removed_kg_s, "mass_flow", system),
        "exergia_destruida": _value(result.X_destroyed_W, "power", system),
    }


def cooling_tower_to_dict(result: CoolingTowerResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado de la torre serializable a JSON (valores en ``system``)."""
    ti = result.inputs
    dead = result.dead
    return {
        "modelo": "gas ideal con c_p constante (vademecum §14.12.7)",
        "p": _value(ti.p_Pa, "pressure", system),
        "agua": {
            "m_entra": _value(ti.m_water_in_kg_s, "mass_flow", system),
            "T_entra": _value(ti.T_water_in_K, "temperature", system),
            "T_sale": _value(ti.T_water_out_K, "temperature", system),
            "m_sale": _value(result.m_water_out_kg_s, "mass_flow", system),
        },
        "aire_entra": {
            "T": _value(result.air_in.T_K, "temperature", system),
            "phi": result.air_in.phi,
            "omega": result.air_in.omega,
            "h": _value(result.air_in.h_J_per_kg, "specific_enthalpy", system),
            "T_bulbo_humedo": _value(result.air_in.T_wb_K, "temperature", system),
        },
        "aire_sale": {
            "T": _value(result.air_out.T_K, "temperature", system),
            "phi": result.air_out.phi,
            "omega": result.air_out.omega,
            "h": _value(result.air_out.h_J_per_kg, "specific_enthalpy", system),
        },
        "m_aire_seco": _value(result.m_dry_air_kg_s, "mass_flow", system),
        "V_aire_entra": _value(result.V_air_in_m3_s, "volume_flow", system),
        "agua_de_reposicion": _value(result.makeup_kg_s, "mass_flow", system),
        "calor": _value(result.Q_W, "power", system),
        "rango": _value(result.range_K, "temperature_difference", system),
        "aproximacion": _value(result.approach_K, "temperature_difference", system),
        "efectividad": result.effectiveness,
        "L_sobre_G": result.water_to_air,
        "ambiente": {"T0": _value(dead.T0_K, "temperature", system), "phi0": dead.phi0},
        "exergia_destruida": _value(result.X_destroyed_W, "power", system),
    }
