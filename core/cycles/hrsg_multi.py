"""Caldera de recuperación (HRSG) de dos y tres presiones — Fase 3.5.

Cada **nivel de presión** tiene su sobrecalentador (opcional), su evaporador
y su economizador. Los gases recorren los niveles de mayor a menor presión
(alta → media → baja) y el agua hace el camino inverso, en **cascada**:

- el economizador de baja calienta toda el agua de alimentación;
- el domo de baja evapora el vapor de baja y entrega el resto, como líquido
  saturado, a la bomba del nivel siguiente, que lo lleva a su economizador;
- y así hasta el domo de alta, que evapora todo lo que le llega.

Es el arreglo en serie más simple (como una caldera con el desaireador
integrado al domo de baja); las calderas reales intercalan secciones de
distintos niveles (economizadores partidos o en paralelo) para acercar
todavía más las curvas (Kehlhofer, Hannemann, Stirnimann y Rukes,
*Combined-Cycle Gas & Steam Turbine Power Plants*, 3.ª ed., PennWell, 2009,
cap. 5).

El diseño de cada nivel se fija con su **pinch** y su **approach**, como en
la caldera de una presión (:mod:`core.cycles.hrsg`). Con eso, los caudales
salen de un balance por nivel, de alta a baja: el tramo de los gases desde
que llegan al nivel hasta su pinch calienta el vapor del nivel (sobre-
calentador y evaporador) y, en el domo, lleva de T_sat − approach a T_sat el
agua que sigue hacia los niveles de más presión. Las bombas entre niveles
quedan fuera de la caldera; se toman isoentrópicas salvo que se dé su
rendimiento (en el ciclo combinado, el de las bombas del ciclo de vapor).

**Exergía** (con T₀ = la temperatura de referencia, 15 °C): de la exergía que
traen los gases, una parte gana el agua, otra se destruye en la transferencia
de calor con diferencia de temperatura (T₀·S_gen de cada sección) y otra se
va por la chimenea. Más niveles acercan la curva del agua a la de los gases:
menos destrucción y una chimenea más fría.

**Recalentador** (Fase 3.6, opcional): el vapor que vuelve de la turbina de
alta se recalienta en un banco **en paralelo con el sobrecalentador de alta**:
los dos ven los gases de entrada y los dejan a una temperatura común (equivale a
repartir los gases entre los dos bancos). Con tres niveles, el vapor de media se
suma al recalentamiento frío antes del recalentador; entonces los caudales de
alta y de media salen de un sistema lineal de 2×2.

Los gases, como mezcla de gases ideales (:mod:`core.ideal_gas`); el agua y el
vapor, con IAPWS-95. Los tests lo comparan con una red de TESPy. Todo en SI.
No importa Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from CoolProp.CoolProp import PropsSI

from core.cycles.hrsg import _T_GAS_MIN_K, HRSGInputs, HRSGSection, Water, _bar, _degC
from core.fluids import FluidState, fluid_limits, fluid_state_from_pair
from core.ideal_gas import GAS_NAMES, T_GAS_MAX_K, FlueGas
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "MAX_LEVELS",
    "MULTI_HRSG_EXAMPLES",
    "MULTI_HRSG_EXAMPLE_NOTES",
    "HRSGExergy",
    "LevelResult",
    "MultiHRSGInputs",
    "MultiHRSGResult",
    "MultiSweepParameter",
    "MultiSweepPoint",
    "MultiTQProfile",
    "PressureLevel",
    "ReheatResult",
    "ReheatSystem",
    "Reheater",
    "TQSegment",
    "default_multi_sweep_values",
    "from_single",
    "hrsg_exergy",
    "level_comparison",
    "level_names",
    "multi_hrsg_notes",
    "multi_hrsg_sweep",
    "multi_hrsg_to_dict",
    "multi_tq_profile",
    "reheat_system",
    "solve_multi_hrsg",
    "validate_multi_hrsg_inputs",
]

MAX_LEVELS = 3
_EH: QuantityKind = "specific_enthalpy"
# Rótulos de los puntos de los gases; sin la «g», que en h_g(T_g) se confundiría con la de
# «gases» (h_g es la entalpía de los gases).
_GAS_LETTERS = "abcdefhijkl"


def level_names(n: int) -> tuple[str, ...]:
    """Nombres de los niveles, de mayor a menor presión."""
    return {1: ("",), 2: ("alta", "baja"), 3: ("alta", "media", "baja")}[n]


def _of(name: str) -> str:
    """«de alta» (vacío con un solo nivel)."""
    return f" de {name}" if name else ""


# ---------------------------------------------------------------------
# Datos y resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PressureLevel:
    """Un nivel de presión: domo a ``p_Pa`` y vapor sobrecalentado o saturado (``None``)."""

    p_Pa: float
    T_steam_K: float | None = None
    pinch_K: float = 10.0
    approach_K: float = 5.0

    @property
    def superheated(self) -> bool:
        return self.T_steam_K is not None


@dataclass(frozen=True)
class Reheater:
    """Recalentador en paralelo con el sobrecalentador de alta, en SI.

    El vapor vuelve de la turbina de alta a ``p_Pa`` con entalpía
    ``h_cold_J_per_kg`` (el recalentamiento frío) y sale a ``T_K``. Con
    ``joins_middle`` (tres niveles, ``p_Pa`` = la presión de media) el vapor de
    media se suma al recalentamiento frío antes de entrar al recalentador.
    """

    p_Pa: float
    T_K: float
    h_cold_J_per_kg: float
    joins_middle: bool = False


@dataclass(frozen=True)
class MultiHRSGInputs:
    """Datos de la HRSG de varias presiones, en SI.

    ``levels`` va de mayor a menor presión (1 a 3 niveles). El agua de
    alimentación entra al economizador del nivel de menor presión. ``T_ref_K``
    es la temperatura hasta la que se cuenta el calor disponible en los gases
    y la del ambiente para la exergía (15 °C). ``reheat`` es el recalentador
    (``None``: sin recalentamiento, como en la Fase 3.5) y ``eta_pump`` el
    rendimiento isoentrópico de las bombas entre niveles (1: isoentrópicas).
    """

    T_gas_in_K: float
    m_gas_kg_s: float
    gas: FlueGas
    levels: tuple[PressureLevel, ...]
    T_feedwater_K: float
    T_ref_K: float = 288.15
    reheat: Reheater | None = None
    eta_pump: float = 1.0

    @property
    def names(self) -> tuple[str, ...]:
        return level_names(len(self.levels))


def from_single(inputs: HRSGInputs) -> MultiHRSGInputs:
    """La caldera de una presión (diseño por pinch) en el modelo de varias presiones."""
    if inputs.by_stack:
        raise ValueError("El modelo de varias presiones se diseña por pinch, no por chimenea.")
    level = PressureLevel(inputs.p_steam_Pa, inputs.T_steam_K, inputs.pinch_K, inputs.approach_K)
    return MultiHRSGInputs(
        inputs.T_gas_in_K,
        inputs.m_gas_kg_s,
        inputs.gas,
        (level,),
        inputs.T_feedwater_K,
        inputs.T_ref_K,
    )


@dataclass(frozen=True)
class LevelResult:
    """Un nivel resuelto (SI).

    ``water``: entrada al economizador (el agua de alimentación en el nivel de
    menor presión; si no, la que sale de la bomba que la trae del domo de abajo),
    salida del economizador, líquido saturado, vapor saturado y, si hay
    sobrecalentador, vapor sobrecalentado. ``m_water_kg_s`` es el caudal que
    pasa por su economizador: el vapor de este nivel más el de los de mayor
    presión. ``T_gas_K``: los gases al llegar al nivel, tras el sobrecalentador
    (si hay), a la salida del evaporador (el pinch) y tras el economizador.
    ``w_pump_J_per_kg`` es el trabajo de la bomba que alimenta este nivel (0 en
    el de menor presión).
    """

    name: str
    level: PressureLevel
    m_steam_kg_s: float
    m_water_kg_s: float
    water: tuple[FluidState, ...]
    T_gas_K: tuple[float, ...]
    h_gas_J_per_kg: tuple[float, ...]
    w_pump_J_per_kg: float

    @property
    def T_sat_K(self) -> float:
        return self.water[2].T_K

    @property
    def h_steam_J_per_kg(self) -> float:
        return self.water[-1].h_J_per_kg

    @property
    def T_pinch_gas_K(self) -> float:
        return self.T_gas_K[-2]

    @property
    def pinch_K(self) -> float:
        return self.T_pinch_gas_K - self.T_sat_K

    @property
    def m_through_kg_s(self) -> float:
        """El agua que pasa por el domo hacia los niveles de mayor presión."""
        return self.m_water_kg_s - self.m_steam_kg_s


@dataclass(frozen=True)
class ReheatResult:
    """El recalentador resuelto (SI).

    ``cold`` es el recalentamiento frío (la salida de la turbina de alta),
    ``inlet`` la entrada al recalentador (la mezcla con el vapor de media, si se
    suma) y ``hot`` el recalentamiento caliente. ``m_kg_s`` es el caudal que lo
    atraviesa y ``m_middle_kg_s`` el vapor de media que se le suma.
    """

    reheater: Reheater
    m_kg_s: float
    m_middle_kg_s: float
    cold: FluidState
    inlet: FluidState
    hot: FluidState

    @property
    def Q_W(self) -> float:
        return self.m_kg_s * (self.hot.h_J_per_kg - self.inlet.h_J_per_kg)


SectionKind = Literal["sobrecalentador", "recalentador", "evaporador", "economizador"]


@dataclass(frozen=True)
class _SectionFlows:
    """Una sección con sus corrientes de agua (caudal, entrada, salida) para los balances.

    ``gas_share`` es la parte de los gases que pasa por la sección: 1, salvo en los
    bancos en paralelo (sobrecalentador de alta y recalentador), donde es su parte
    del calor del tramo (los dos dejan los gases a la misma temperatura).
    """

    section: HRSGSection
    level: int
    kind: SectionKind
    streams: tuple[tuple[float, FluidState, FluidState], ...]
    gas_share: float = 1.0


@dataclass(frozen=True)
class MultiHRSGResult:
    """La HRSG de varias presiones resuelta (SI)."""

    inputs: MultiHRSGInputs
    levels: tuple[LevelResult, ...]
    reheat: ReheatResult | None = None

    @property
    def T_stack_K(self) -> float:
        return self.levels[-1].T_gas_K[-1]

    @property
    def h_stack_J_per_kg(self) -> float:
        return self.levels[-1].h_gas_J_per_kg[-1]

    @property
    def m_steam_kg_s(self) -> float:
        """Vapor total (todos los niveles)."""
        return sum(lv.m_steam_kg_s for lv in self.levels)

    @property
    def gas_points(self) -> tuple[tuple[str, float, float], ...]:
        """Los gases en el sentido del flujo: (rótulo, T, h), con rótulos a, b, c… (sin g)."""
        points: list[tuple[float, float]] = [
            (self.inputs.T_gas_in_K, self.levels[0].h_gas_J_per_kg[0])
        ]
        for lv in self.levels:
            points += list(zip(lv.T_gas_K[1:], lv.h_gas_J_per_kg[1:], strict=True))
        return tuple((_GAS_LETTERS[k], T, h) for k, (T, h) in enumerate(points))

    def _flows(self) -> tuple[_SectionFlows, ...]:
        """Las secciones en el sentido de los gases, con sus corrientes de agua.

        El evaporador incluye el calentamiento en el domo, de la salida del
        economizador a T_sat, del agua del nivel y de la que sigue hacia los de
        mayor presión; sus tubos ven agua a T_sat (su ΔT frío es el pinch). Con
        recalentador, el sobrecalentador de alta y el recalentador son dos bancos
        en paralelo entre los mismos puntos de los gases.
        """
        out: list[_SectionFlows] = []
        rh = self.reheat
        for i, lv in enumerate(self.levels):
            w = lv.water
            of = _of(lv.name)
            T = lv.T_gas_K
            k = 0
            q_sh = 0.0
            if lv.level.superheated:
                q_sh = lv.m_steam_kg_s * (w[4].h_J_per_kg - w[3].h_J_per_kg)
            q_par = q_sh + (rh.Q_W if i == 0 and rh is not None else 0.0)
            if lv.level.superheated:
                sec = HRSGSection(f"sobrecalentador{of}", q_sh, T[0], T[1], w[3].T_K, w[4].T_K)
                out.append(
                    _SectionFlows(
                        sec,
                        i,
                        "sobrecalentador",
                        ((lv.m_steam_kg_s, w[3], w[4]),),
                        q_sh / q_par,
                    )
                )
                k = 1
            if i == 0 and rh is not None:
                sec = HRSGSection("recalentador", rh.Q_W, T[0], T[1], rh.inlet.T_K, rh.hot.T_K)
                out.append(
                    _SectionFlows(
                        sec, i, "recalentador", ((rh.m_kg_s, rh.inlet, rh.hot),), rh.Q_W / q_par
                    )
                )
                k = 1
            streams: tuple[tuple[float, FluidState, FluidState], ...] = (
                (lv.m_steam_kg_s, w[1], w[3]),
            )
            if lv.m_through_kg_s > 0.0:
                streams += ((lv.m_through_kg_s, w[1], w[2]),)
            q = sum(m * (b.h_J_per_kg - a.h_J_per_kg) for m, a, b in streams)
            sec = HRSGSection(f"evaporador{of}", q, T[k], T[k + 1], w[2].T_K, w[3].T_K)
            out.append(_SectionFlows(sec, i, "evaporador", streams))
            q = lv.m_water_kg_s * (w[1].h_J_per_kg - w[0].h_J_per_kg)
            sec = HRSGSection(f"economizador{of}", q, T[k + 1], T[k + 2], w[0].T_K, w[1].T_K)
            out.append(_SectionFlows(sec, i, "economizador", ((lv.m_water_kg_s, w[0], w[1]),)))
        return tuple(out)

    @property
    def sections(self) -> tuple[HRSGSection, ...]:
        """Sobrecalentador, evaporador y economizador de cada nivel, en el sentido de los gases."""
        return tuple(f.section for f in self._flows())

    @property
    def Q_W(self) -> float:
        """Calor que ceden los gases (= el que reciben el agua y el vapor en la caldera)."""
        return self.inputs.m_gas_kg_s * (self.levels[0].h_gas_J_per_kg[0] - self.h_stack_J_per_kg)

    @property
    def Q_available_W(self) -> float:
        """Calor que cederían los gases enfriándose hasta T_ref (15 °C)."""
        inputs = self.inputs
        return inputs.m_gas_kg_s * (self.levels[0].h_gas_J_per_kg[0] - inputs.gas.h(inputs.T_ref_K))

    @property
    def recovery(self) -> float:
        """Aprovechamiento: Q̇ / Q̇ disponible hasta T_ref."""
        return self.Q_W / self.Q_available_W

    @property
    def W_pumps_W(self) -> float:
        """Potencia de las bombas entre niveles (fuera de la caldera)."""
        return sum(lv.m_water_kg_s * lv.w_pump_J_per_kg for lv in self.levels)

    @property
    def dew_point_K(self) -> float | None:
        return self.inputs.gas.dew_point_K


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


def _T_sat(p_Pa: float) -> float:
    return float(PropsSI("T", "P", p_Pa, "Q", 0, Water))


def validate_multi_hrsg_inputs(inputs: MultiHRSGInputs) -> None:
    """Verifica que la caldera tenga sentido físico antes de calcular.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno: qué dato está fuera de rango y por qué.
    """
    if not (math.isfinite(inputs.m_gas_kg_s) and inputs.m_gas_kg_s > 0.0):
        raise ValueError("El caudal de gases tiene que ser positivo.")
    if not _T_GAS_MIN_K < inputs.T_gas_in_K <= T_GAS_MAX_K:
        raise ValueError(
            f"La temperatura de los gases ({_degC(inputs.T_gas_in_K)}) está fuera del rango del "
            f"modelo (hasta {_degC(T_GAS_MAX_K)})."
        )
    n = len(inputs.levels)
    if not 1 <= n <= MAX_LEVELS:
        raise ValueError(f"La caldera tiene que tener entre 1 y {MAX_LEVELS} niveles de presión.")
    if not inputs.T_feedwater_K >= _T_GAS_MIN_K:
        raise ValueError("El agua de alimentación tiene que estar por encima de 0 °C.")
    if not inputs.T_ref_K > 0.0:
        raise ValueError("La temperatura de referencia tiene que ser absoluta positiva.")
    if not 0.0 < inputs.eta_pump <= 1.0:
        raise ValueError("El rendimiento de las bombas tiene que estar entre 0 y 1.")
    lim = fluid_limits(Water)
    names = inputs.names
    for name, lv in zip(names, inputs.levels, strict=True):
        where = f"Nivel{_of(name)}: " if name else ""
        p = lv.p_Pa
        if p <= lim.P_triple_Pa:
            raise ValueError(
                f"{where}la presión ({_bar(p)}) está por debajo de la del punto triple del agua."
            )
        if p >= lim.P_crit_Pa:
            raise ValueError(
                f"{where}la presión ({_bar(p)}) supera la crítica del agua "
                f"({_bar(lim.P_crit_Pa)}): por encima no hay evaporación ni domo."
            )
        if not lv.pinch_K > 0.0:
            raise ValueError(
                f"{where}el pinch tiene que ser positivo: con ΔT = 0 el evaporador necesitaría un "
                "área infinita."
            )
        if not lv.approach_K >= 0.0:
            raise ValueError(
                f"{where}el approach no puede ser negativo: el agua evaporaría dentro del "
                "economizador (con 0 sale como líquido saturado)."
            )
        if lv.T_steam_K is not None:
            T_sat = _T_sat(p)
            if lv.T_steam_K <= T_sat + 0.01:
                raise ValueError(
                    f"{where}el vapor sobrecalentado ({_degC(lv.T_steam_K)}) tiene que salir más "
                    f"caliente que la saturación ({_degC(T_sat)} a {_bar(p)}). Para vapor "
                    "saturado, elegí esa opción."
                )
    for (hi_name, hi), (lo_name, lo) in zip(
        zip(names, inputs.levels, strict=True),
        zip(names[1:], inputs.levels[1:], strict=True),
        strict=False,
    ):
        if not lo.p_Pa < hi.p_Pa:
            raise ValueError(
                f"La presión{_of(lo_name)} ({_bar(lo.p_Pa)}) tiene que ser menor que la"
                f"{_of(hi_name)} ({_bar(hi.p_Pa)}): los niveles van de mayor a menor presión."
            )
        T_eco_hi = _T_sat(hi.p_Pa) - hi.approach_K
        T_lo = _T_sat(lo.p_Pa)
        if T_lo >= T_eco_hi:
            raise ValueError(
                f"Las presiones{_of(hi_name)} y{_of(lo_name)} están muy cerca: el agua que sale "
                f"del domo{_of(lo_name)} ({_degC(T_lo)}) ya está más caliente que la salida del "
                f"economizador{_of(hi_name)}, T_sat − approach = {_degC(T_eco_hi)}. Separá más "
                "las presiones."
            )
    low = inputs.levels[-1]
    T_eco_low = _T_sat(low.p_Pa) - low.approach_K
    if inputs.T_feedwater_K >= T_eco_low:
        raise ValueError(
            f"El agua de alimentación ({_degC(inputs.T_feedwater_K)}) tiene que llegar más fría "
            f"que la salida del economizador{_of(names[-1])}, T_sat − approach = "
            f"{_degC(T_eco_low)}: si no, no hay nada que calentar en ese economizador."
        )
    if inputs.reheat is not None:
        _validate_reheat(inputs)


def _validate_reheat(inputs: MultiHRSGInputs) -> None:
    """El recalentador: presión, temperatura y de dónde viene el vapor."""
    rh = inputs.reheat
    assert rh is not None
    high = inputs.levels[0]
    lim = fluid_limits(Water)
    p = rh.p_Pa
    if not lim.P_triple_Pa < p < high.p_Pa:
        raise ValueError(
            f"Recalentador: la presión de recalentamiento ({_bar(p)}) tiene que estar por debajo "
            f"de la de alta ({_bar(high.p_Pa)}): el vapor vuelve de la turbina de alta, después "
            "de expandirse."
        )
    if rh.joins_middle:
        if len(inputs.levels) != 3:
            raise ValueError(
                "Recalentador: el vapor de media solo se puede sumar al recalentamiento con tres "
                "niveles de presión."
            )
        middle = inputs.levels[1]
        if not math.isclose(p, middle.p_Pa, rel_tol=1e-9):
            raise ValueError(
                f"Recalentador: para sumarle el vapor de media, el recalentamiento tiene que ser "
                f"a la presión de media ({_bar(middle.p_Pa)}), no a {_bar(p)}."
            )
    if not (math.isfinite(rh.h_cold_J_per_kg) and math.isfinite(rh.T_K)):
        raise ValueError("Recalentador: faltan la entalpía de entrada o la temperatura de salida.")
    try:
        cold = fluid_state_from_pair(Water, "PH", p=p, h=rh.h_cold_J_per_kg)
    except ValueError as exc:
        raise ValueError(
            f"Recalentador: el vapor que vuelve de la turbina no tiene un estado válido ({exc})."
        ) from exc
    if rh.T_K >= inputs.T_gas_in_K:
        raise ValueError(
            f"Recalentador: el vapor recalentado ({_degC(rh.T_K)}) no puede salir más caliente "
            f"que los gases que entran a la caldera ({_degC(inputs.T_gas_in_K)})."
        )
    if rh.T_K <= cold.T_K + 0.01:
        raise ValueError(
            f"Recalentador: el vapor vuelve de la turbina de alta a {_degC(cold.T_K)}; para "
            f"recalentarlo, la temperatura de salida ({_degC(rh.T_K)}) tiene que ser mayor."
        )
    if rh.joins_middle:
        middle = inputs.levels[1]
        T_mid = middle.T_steam_K if middle.T_steam_K is not None else _T_sat(middle.p_Pa)
        if rh.T_K <= T_mid + 0.01:
            raise ValueError(
                f"Recalentador: el vapor de media ({_degC(T_mid)}) se suma al recalentamiento; la "
                f"temperatura de salida ({_degC(rh.T_K)}) tiene que ser mayor."
            )


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _level_water(level: PressureLevel, inlet: FluidState) -> tuple[FluidState, ...]:
    """Entrada y salida del economizador, líquido y vapor saturados y (si hay) sobrecalentado."""
    p = level.p_Pa
    liquid = fluid_state_from_pair(Water, "PX", p=p, x=0.0)
    vapor = fluid_state_from_pair(Water, "PX", p=p, x=1.0)
    if level.approach_K == 0.0:
        eco = liquid
    else:
        eco = fluid_state_from_pair(Water, "TP", t=liquid.T_K - level.approach_K, p=p)
    states = [inlet, eco, liquid, vapor]
    if level.T_steam_K is not None:
        states.append(fluid_state_from_pair(Water, "TP", t=level.T_steam_K, p=p))
    return tuple(states)


def solve_multi_hrsg(inputs: MultiHRSGInputs) -> MultiHRSGResult:
    """Balance de energía de la HRSG en cascada, nivel por nivel (de alta a baja).

    Para cada nivel i, con los gases que le llegan (h_g):

    1. Del ingreso al nivel al pinch (T_sat + ΔT_pinch), los gases calientan su
       vapor (sobrecalentador y evaporador) y llevan a T_sat, en el domo, el agua
       que sigue hacia los niveles de mayor presión (Ṁ_arriba):
       ṁ_i·(h_vapor − h_eco) + Ṁ_arriba·(h_f − h_eco) = ṁ_g·[h_g − h_g(T_pinch)].
    2. El economizador calienta ṁ_i + Ṁ_arriba de la entrada a T_sat − approach;
       los gases salen de él hacia el nivel siguiente (o a la chimenea).

    Con recalentador, el tramo del nivel de alta suma su calor: con el vapor de
    media sumado al recalentamiento, los balances de alta y de media forman un
    sistema lineal en (ṁ_A, ṁ_M) (:func:`reheat_system`).

    Raises
    ------
    ValueError
        Si los datos no tienen sentido (:func:`validate_multi_hrsg_inputs`), los
        gases no alcanzan para un nivel, hay un cruce de temperaturas o los gases
        condensarían.
    """
    validate_multi_hrsg_inputs(inputs)
    gas, m_g = inputs.gas, inputs.m_gas_kg_s
    names = inputs.names
    n = len(inputs.levels)
    # Entradas de agua de cada economizador: la alimentación (el de menor presión)
    # o la salida de la bomba que trae el líquido saturado del domo de abajo.
    inlets: list[FluidState] = []
    pumps: list[float] = []
    for i, level in enumerate(inputs.levels):
        if i == n - 1:
            inlets.append(fluid_state_from_pair(Water, "TP", t=inputs.T_feedwater_K, p=level.p_Pa))
            pumps.append(0.0)
        else:
            below = fluid_state_from_pair(Water, "PX", p=inputs.levels[i + 1].p_Pa, x=0.0)
            pumped = fluid_state_from_pair(Water, "PS", p=level.p_Pa, s=below.s_J_per_kg_K)
            if inputs.eta_pump != 1.0:
                h_real = below.h_J_per_kg + (pumped.h_J_per_kg - below.h_J_per_kg) / inputs.eta_pump
                pumped = fluid_state_from_pair(Water, "PH", p=level.p_Pa, h=h_real)
            inlets.append(pumped)
            pumps.append(pumped.h_J_per_kg - below.h_J_per_kg)
    waters = [_level_water(level, inlets[i]) for i, level in enumerate(inputs.levels)]
    reheat = inputs.reheat
    rh_cold = rh_hot = None
    if reheat is not None:
        rh_cold = fluid_state_from_pair(Water, "PH", p=reheat.p_Pa, h=reheat.h_cold_J_per_kg)
        rh_hot = fluid_state_from_pair(Water, "TP", t=reheat.T_K, p=reheat.p_Pa)
    coupled = reheat is not None and reheat.joins_middle
    if coupled:
        system = reheat_system(inputs, waters, rh_cold, rh_hot)
        m_coupled = system.solution
    T_g = inputs.T_gas_in_K
    h_g = gas.h(T_g)
    m_above = 0.0
    m_rh = m_mid = 0.0
    levels: list[LevelResult] = []
    for i, (name, level) in enumerate(zip(names, inputs.levels, strict=True)):
        where = f"Nivel{_of(name)}: " if name else ""
        water = waters[i]
        h = [s.h_J_per_kg for s in water]
        T_sat = water[2].T_K
        T_p = T_sat + level.pinch_K
        after = " después de los niveles de mayor presión" if i else ""
        if T_g <= T_p:
            raise ValueError(
                f"{where}los gases le llegan a {_degC(T_g)}, sin superar T_sat + pinch = "
                f"{_degC(T_p)}: no alcanzan para evaporar a {_bar(level.p_Pa)}{after}. Bajá esa "
                "presión o su pinch."
            )
        if level.T_steam_K is not None and level.T_steam_K >= T_g:
            raise ValueError(
                f"{where}el vapor ({_degC(level.T_steam_K)}) no puede salir más caliente que los "
                f"gases que le llegan ({_degC(T_g)}"
                + (
                    ", después de pasar por los niveles de mayor presión). Bajá la temperatura "
                    "del vapor o elegí vapor saturado."
                    if i
                    else "). En el sobrecalentador el calor va de los gases al vapor."
                )
            )
        h_p = gas.h(T_p)
        if coupled and i < 2:
            m_s = m_coupled[i]
        elif i == 0 and rh_hot is not None:
            # alta con recalentador (sin la media): el RH lleva el mismo vapor
            m_s = m_g * (h_g - h_p) / ((h[-1] - h[1]) + (rh_hot.h_J_per_kg - rh_cold.h_J_per_kg))
        else:
            m_s = (m_g * (h_g - h_p) - m_above * (h[2] - h[1])) / (h[-1] - h[1])
        if not m_s > 0.0:
            raise ValueError(
                f"{where}los gases llegan a {_degC(T_g)}, pero hasta el pinch ({_degC(T_p)}) "
                "solo alcanzan para calentar en el domo el agua que sigue hacia los niveles de "
                f"mayor presión: no queda calor para evaporar a {_bar(level.p_Pa)}. Bajá esa "
                "presión o su pinch."
            )
        temps, enthalpies = [T_g], [h_g]
        q_first = m_s * (h[4] - h[3]) if level.T_steam_K is not None else 0.0
        if i == 0 and rh_hot is not None:
            m_mid = m_coupled[1] if coupled else 0.0
            m_rh = m_s + m_mid
            h_mid = waters[1][-1].h_J_per_kg if coupled else 0.0
            q_first += m_s * (rh_hot.h_J_per_kg - rh_cold.h_J_per_kg) + m_mid * (
                rh_hot.h_J_per_kg - h_mid
            )
        if q_first > 0.0:
            h_b = h_g - q_first / m_g
            T_b = gas.T_from_h(h_b)
            temps.append(T_b)
            enthalpies.append(h_b)
            if i == 0 and rh_hot is not None:
                h_in_rh = (m_s * rh_cold.h_J_per_kg + m_mid * h_mid) / m_rh
                T_in_rh = float(PropsSI("T", "P", reheat.p_Pa, "H", h_in_rh, Water))
                if T_b <= T_in_rh:
                    raise ValueError(
                        f"Recalentador: los gases salen del tramo del sobrecalentador de alta y "
                        f"el recalentador a {_degC(T_b)}, más fríos que el vapor que entra al "
                        f"recalentador ({_degC(T_in_rh)}): hay un cruce de temperaturas. Bajá la "
                        "presión de recalentamiento (el vapor vuelve más frío de la turbina) o "
                        "la temperatura del vapor."
                    )
        temps.append(T_p)
        enthalpies.append(h_p)
        m_water = m_above + m_s
        h_after = h_p - m_water * (h[1] - h[0]) / m_g
        try:
            T_after = gas.T_from_h(h_after)
        except ValueError:
            T_after = -math.inf
        T_in_water = water[0].T_K
        if T_after <= T_in_water:
            entering = "el agua de alimentación" if i == n - 1 else "el agua que le llega"
            raise ValueError(
                f"{where}el economizador tendría un cruce de temperaturas: los gases tendrían que "
                f"salir más fríos que {entering} ({_degC(T_in_water)}) para calentar todo el "
                "caudal. Subí el pinch (menos vapor) o la temperatura del agua."
            )
        temps.append(T_after)
        enthalpies.append(h_after)
        levels.append(
            LevelResult(
                name=name,
                level=level,
                m_steam_kg_s=float(m_s),
                m_water_kg_s=float(m_water),
                water=water,
                T_gas_K=tuple(temps),
                h_gas_J_per_kg=tuple(enthalpies),
                w_pump_J_per_kg=pumps[i],
            )
        )
        T_g, h_g, m_above = T_after, h_after, m_water
    rh_result = None
    if reheat is not None:
        h_in_rh = (
            levels[0].m_steam_kg_s * rh_cold.h_J_per_kg
            + (m_mid * waters[1][-1].h_J_per_kg if coupled else 0.0)
        ) / m_rh
        inlet = fluid_state_from_pair(Water, "PH", p=reheat.p_Pa, h=h_in_rh) if coupled else rh_cold
        rh_result = ReheatResult(reheat, float(m_rh), float(m_mid), rh_cold, inlet, rh_hot)
    result = MultiHRSGResult(inputs=inputs, levels=tuple(levels), reheat=rh_result)
    dew = gas.dew_point_K
    if dew is not None and result.T_stack_K < dew:
        raise ValueError(
            f"Los gases saldrían por la chimenea a {_degC(result.T_stack_K)}, por debajo de su "
            f"punto de rocío ({_degC(dew)}): su vapor de agua condensaría, y este modelo (gases "
            "ideales, sin condensación) no lo contempla. Subí la temperatura del agua de "
            "alimentación o la presión de baja."
        )
    profile = multi_tq_profile(result, n=60)
    if profile.min_dT_K <= 0.0:
        raise ValueError(
            "Hay un cruce de temperaturas dentro de la caldera (los gases quedarían más fríos "
            f"que el agua en Q = {profile.min_dT_Q_W / 1e3:.4g} kW). Subí los pinch o la "
            "temperatura del agua de alimentación."
        )
    return result


@dataclass(frozen=True)
class ReheatSystem:
    """Los balances de alta y de media acoplados por el recalentador (SI).

    a11·ṁ_A + a12·ṁ_M = b1 (de la entrada de los gases al pinch de alta) y
    a21·ṁ_A + a22·ṁ_M = b2 (del pinch de alta al de media, con el economizador
    de alta y el domo de media), con

    - a11 = (h_top,A − h_2,A) + (h_RH − h_frío), a12 = h_RH − h_top,M;
    - a21 = (h_3,M − h_2,M) + (h_2,A − h_1,A), a22 = h_top,M − h_2,M;
    - b1 = ṁ_g·[h_g(T_entrada) − h_g(T_pinch,A)] y b2 = ṁ_g·[h_g(T_pinch,A) − h_g(T_pinch,M)].
    """

    a11: float
    a12: float
    a21: float
    a22: float
    b1: float
    b2: float

    @property
    def det(self) -> float:
        return self.a11 * self.a22 - self.a12 * self.a21

    @property
    def solution(self) -> tuple[float, float]:
        """(ṁ_A, ṁ_M) por la regla de Cramer."""
        d = self.det
        return (
            (self.b1 * self.a22 - self.a12 * self.b2) / d,
            (self.a11 * self.b2 - self.a21 * self.b1) / d,
        )


def reheat_system(
    inputs: MultiHRSGInputs,
    waters: Sequence[Sequence[FluidState]],
    cold: FluidState,
    hot: FluidState,
) -> ReheatSystem:
    """Arma el sistema 2×2 de alta y media con el recalentador (Kehlhofer et al., cap. 5)."""
    gas, m_g = inputs.gas, inputs.m_gas_kg_s
    hA = [s.h_J_per_kg for s in waters[0]]
    hM = [s.h_J_per_kg for s in waters[1]]
    high, middle = inputs.levels[0], inputs.levels[1]
    h_pA = gas.h(waters[0][2].T_K + high.pinch_K)
    h_pM = gas.h(waters[1][2].T_K + middle.pinch_K)
    return ReheatSystem(
        a11=(hA[-1] - hA[1]) + (hot.h_J_per_kg - cold.h_J_per_kg),
        a12=hot.h_J_per_kg - hM[-1],
        a21=(hM[2] - hM[1]) + (hA[1] - hA[0]),
        a22=hM[-1] - hM[1],
        b1=m_g * (gas.h(inputs.T_gas_in_K) - h_pA),
        b2=m_g * (h_pA - h_pM),
    )


# ---------------------------------------------------------------------
# Diagrama T–Q
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TQSegment:
    """Un tramo de la curva del agua: sección de un nivel (índice 0 = mayor presión).

    Los bancos en paralelo (sobrecalentador de alta y recalentador) ocupan los
    dos todo el tramo de Q del paralelo: cada uno ve los gases entre los mismos
    puntos.
    """

    level: int
    kind: SectionKind
    Q_W: tuple[float, ...]
    T_K: tuple[float, ...]


@dataclass(frozen=True)
class MultiTQProfile:
    """Curvas del diagrama T–Q (SI), con el calor acumulado desde la chimenea.

    ``segments`` son los tramos del agua desde la chimenea (economizador de
    baja) hasta la entrada de los gases (sobrecalentador de alta): un
    «serrucho», porque el agua de cada nivel arranca a la temperatura del domo
    de abajo. ``boundaries_W`` son los Q donde cambia de sección.
    """

    Q_gas_W: tuple[float, ...]
    T_gas_K: tuple[float, ...]
    segments: tuple[TQSegment, ...]
    boundaries_W: tuple[float, ...]
    min_dT_K: float
    min_dT_Q_W: float


def _water_T(p_Pa: float, hs: Sequence[float]) -> list[float]:
    return [float(PropsSI("T", "P", p_Pa, "H", hh, Water)) for hh in hs]


def multi_tq_profile(result: MultiHRSGResult, n: int = 40) -> MultiTQProfile:
    """Temperaturas de los gases y del agua contra el calor transferido.

    Los gases: Q(T) = ṁ_g·[h_g(T) − h_g(T_chimenea)]. El agua: cada sección en su
    tramo de Q, con T(p, h) de IAPWS-95; en cada evaporador, el agua sube
    vertical de T_sat − approach a T_sat al entrar (el domo) y sigue a T_sat.
    Con recalentador, el sobrecalentador de alta y el recalentador se dibujan los
    dos a lo largo de todo el tramo en paralelo (cada banco ve los gases de la
    entrada a la salida del tramo, con su parte del caudal).
    """
    inputs = result.inputs
    gas, m_g = inputs.gas, inputs.m_gas_kg_s
    T_stack = result.T_stack_K
    point_T = [T for _, T, _ in result.gas_points]
    temps = np.unique(np.concatenate([np.linspace(T_stack, inputs.T_gas_in_K, 4 * n), point_T]))
    Q_gas = m_g * (gas.h_array(temps) - result.h_stack_J_per_kg)
    segments: list[TQSegment] = []
    boundaries: list[float] = []
    Q0 = 0.0
    group_start: float | None = None
    for flows in reversed(result._flows()):  # desde la chimenea
        lv = result.levels[flows.level]
        w = lv.water
        if flows.gas_share < 1.0:  # banco en paralelo: todos arrancan en el mismo Q
            if group_start is None:
                group_start = Q0
            start = group_start
        else:
            group_start = None
            start = Q0
        span = flows.section.Q_W / flows.gas_share
        if flows.kind == "evaporador":
            Qs = [start, start, start + span]
            Ts = [w[1].T_K, w[2].T_K, w[2].T_K]
        else:
            _, a, b = flows.streams[0]
            hs = np.linspace(a.h_J_per_kg, b.h_J_per_kg, n)
            Ts = _water_T(a.P_Pa, hs)
            Ts[0], Ts[-1] = a.T_K, b.T_K
            dh = b.h_J_per_kg - a.h_J_per_kg
            Qs = [start + span * (hh - a.h_J_per_kg) / dh for hh in hs]
        segments.append(TQSegment(flows.level, flows.kind, tuple(Qs), tuple(Ts)))
        Q0 = start + span
        if not boundaries or boundaries[-1] != Q0:
            boundaries.append(Q0)
    min_dT, min_Q = math.inf, 0.0
    for seg in segments:
        Ts = np.asarray(seg.T_K)
        if seg.kind == "evaporador":  # los tubos ven agua a T_sat
            Ts = np.full_like(Ts, seg.T_K[-1])
        T_gas_at = np.interp(seg.Q_W, Q_gas, temps)
        dT = T_gas_at - Ts
        k = int(np.argmin(dT))
        if dT[k] < min_dT:
            min_dT, min_Q = float(dT[k]), float(seg.Q_W[k])
    return MultiTQProfile(
        Q_gas_W=tuple(float(q) for q in Q_gas),
        T_gas_K=tuple(float(t) for t in temps),
        segments=tuple(segments),
        boundaries_W=tuple(boundaries[:-1]),
        min_dT_K=min_dT,
        min_dT_Q_W=min_Q,
    )


# ---------------------------------------------------------------------
# Exergía
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HRSGExergy:
    """Balance de exergía de la caldera (W), con el ambiente a ``T0_K``.

    Exergía física (térmica) de los gases a su presión, relativa a T₀:
    ``X_gas_in_W`` = X_ganada + X_destruida + X_chimenea. La ganada es la que
    reciben el agua y el vapor en la caldera (las bombas quedan afuera).
    """

    T0_K: float
    X_gas_in_W: float
    X_stack_W: float
    X_water_W: float
    destroyed_W: tuple[tuple[str, float], ...]

    @property
    def X_destroyed_W(self) -> float:
        return sum(x for _, x in self.destroyed_W)

    @property
    def X_given_W(self) -> float:
        """Exergía que ceden los gases (de la entrada a la chimenea)."""
        return self.X_gas_in_W - self.X_stack_W

    @property
    def efficiency(self) -> float:
        """Parte de la exergía de los gases que termina en el agua y el vapor."""
        return self.X_water_W / self.X_gas_in_W


def hrsg_exergy(result: MultiHRSGResult) -> HRSGExergy:
    """Exergía que traen los gases, la que gana el agua, la destruida y la de la chimenea.

    X_dest de cada sección = T₀·S_gen, con S_gen = Σ ṁ_agua·Δs_agua − ṁ_g·Δs_g
    (Cengel cap. 8: exergía destruida = T₀·S_gen). Los gases, a presión constante:
    Δs_g = s°(T_sal) − s°(T_ent); en los bancos en paralelo, con su parte del caudal.
    """
    inputs = result.inputs
    gas, m_g, T0 = inputs.gas, inputs.m_gas_kg_s, inputs.T_ref_K
    h0, s0 = gas.h(T0), gas.s0(T0)

    def x_gas(T: float, h: float) -> float:
        return m_g * ((h - h0) - T0 * (gas.s0(T) - s0))

    X_in = x_gas(inputs.T_gas_in_K, result.levels[0].h_gas_J_per_kg[0])
    X_stack = x_gas(result.T_stack_K, result.h_stack_J_per_kg)
    X_water = 0.0
    destroyed: list[tuple[str, float]] = []
    for flows in result._flows():
        sec = flows.section
        dS_w = sum(m * (b.s_J_per_kg_K - a.s_J_per_kg_K) for m, a, b in flows.streams)
        dH_w = sum(m * (b.h_J_per_kg - a.h_J_per_kg) for m, a, b in flows.streams)
        dS_g = flows.gas_share * m_g * (gas.s0(sec.T_gas_out_K) - gas.s0(sec.T_gas_in_K))
        X_water += dH_w - T0 * dS_w
        destroyed.append((sec.name, T0 * (dS_w + dS_g)))
    return HRSGExergy(T0, X_in, X_stack, X_water, tuple(destroyed))


# ---------------------------------------------------------------------
# Comparación por cantidad de niveles, notas y ejemplos
# ---------------------------------------------------------------------


def level_comparison(
    inputs: MultiHRSGInputs,
) -> list[tuple[str, MultiHRSGResult | None, str]]:
    """La misma caldera con 1, 2 y 3 niveles (los de los datos), para comparar.

    Con 3 niveles: solo alta, alta y baja, y los tres; con 2: solo alta y los
    dos. Cada fila: (rótulo, resultado o ``None``, el error si no tiene sentido).
    """
    levels = inputs.levels
    cases: list[tuple[str, tuple[PressureLevel, ...]]] = [("1 presión (solo alta)", levels[:1])]
    if len(levels) >= 2:
        cases.append(("2 presiones (alta y baja)", (levels[0], levels[-1])))
    if len(levels) == 3:
        cases.append(("3 presiones", levels))
    out: list[tuple[str, MultiHRSGResult | None, str]] = []
    for label, lv in cases:
        reheat = inputs.reheat
        if reheat is not None and reheat.joins_middle and len(lv) < 3:
            reheat = replace(reheat, joins_middle=False)
        try:
            out.append((label, solve_multi_hrsg(replace(inputs, levels=lv, reheat=reheat)), ""))
        except ValueError as exc:
            out.append((label, None, str(exc)))
    return out


def multi_hrsg_notes(result: MultiHRSGResult) -> list[str]:
    """Observaciones didácticas sobre la caldera calculada (markdown)."""
    notes: list[str] = []
    inputs = result.inputs
    flows = {f.section.name: f.section for f in result._flows()}
    for lv in result.levels:
        sh = flows.get(f"sobrecalentador{_of(lv.name)}")
        if sh is not None and sh.dT_hot_K < lv.pinch_K:
            notes.append(
                f"En el extremo caliente del sobrecalentador{_of(lv.name)} los gases quedan solo "
                f"{sh.dT_hot_K:.3g} K por encima del vapor, menos que el pinch "
                f"({lv.pinch_K:.3g} K): ese sobrecalentador va a ser muy grande. En una caldera "
                "en cascada los sobrecalentadores de media y de baja ven los gases que ya pasaron "
                "por los niveles de mayor presión; las calderas reales los intercalan más arriba."
            )
    if len(result.levels) > 1:
        low = result.levels[-1]
        share = low.m_steam_kg_s / result.m_steam_kg_s
        notes.append(
            f"El nivel{_of(low.name)} produce solo el {share * 100:.3g} % del vapor, pero "
            "aprovecha los gases que los niveles de mayor presión ya no pueden enfriar: los "
            f"recibe a {_degC(low.T_gas_K[0])} y los deja en la chimenea a "
            f"{_degC(result.T_stack_K)}. Además calienta en su economizador y en su domo el agua "
            "de todos los niveles."
        )
    dew = result.dew_point_K
    if dew is not None and inputs.T_feedwater_K < dew:
        notes.append(
            f"El agua de alimentación ({_degC(inputs.T_feedwater_K)}) está por debajo del punto "
            f"de rocío de los gases ({_degC(dew)}): su vapor de agua condensaría sobre los tubos "
            "fríos del economizador de baja (con azufre, condensado ácido y corrosión)."
        )
    if any(lv.level.approach_K == 0.0 for lv in result.levels):
        notes.append(
            "Con approach = 0 el economizador entrega líquido saturado: en operación empezaría a "
            "evaporar adentro (*steaming*). Se diseña con unos pocos kelvin de approach."
        )
    return notes


#: Escape típico de una turbina de gas a gas natural (el mismo de la caldera de una presión).
_GT_EXHAUST = FlueGas.from_fractions(
    {"N2": 0.7448, "O2": 0.1225, "CO2": 0.0396, "H2O": 0.0842, "Ar": 0.0089}
)
_C = 273.15
_EX_2P = "Dos presiones: 80 bar y 540 °C + 6 bar y 200 °C (escape a 600 °C)"
_EX_3P = "Tres presiones: 100 bar y 540 °C + 20 bar y 240 °C + 4 bar (saturado)"
_EX_MODERN = "Turbina moderna: escape a 640 °C; 120 bar y 565 °C + 25 bar y 240 °C + 4 bar"
_EX_MODERN_2P = "Turbina moderna con dos presiones: 120 bar y 565 °C + 5 bar (saturado)"

#: Ejemplos precargados (los gases y el agua de alimentación de la caldera de una presión).
MULTI_HRSG_EXAMPLES: dict[str, MultiHRSGInputs] = {
    _EX_2P: MultiHRSGInputs(
        600.0 + _C,
        100.0,
        _GT_EXHAUST,
        (PressureLevel(80e5, 540.0 + _C), PressureLevel(6e5, 200.0 + _C)),
        60.0 + _C,
    ),
    _EX_3P: MultiHRSGInputs(
        600.0 + _C,
        100.0,
        _GT_EXHAUST,
        (
            PressureLevel(100e5, 540.0 + _C),
            PressureLevel(20e5, 240.0 + _C),
            PressureLevel(4e5),
        ),
        60.0 + _C,
    ),
    _EX_MODERN: MultiHRSGInputs(
        640.0 + _C,
        150.0,
        _GT_EXHAUST,
        (
            PressureLevel(120e5, 565.0 + _C, 8.0),
            PressureLevel(25e5, 240.0 + _C, 8.0),
            PressureLevel(4e5, None, 8.0),
        ),
        60.0 + _C,
    ),
    _EX_MODERN_2P: MultiHRSGInputs(
        640.0 + _C,
        150.0,
        _GT_EXHAUST,
        (PressureLevel(120e5, 565.0 + _C, 8.0), PressureLevel(5e5, None, 8.0)),
        60.0 + _C,
    ),
}

MULTI_HRSG_EXAMPLE_NOTES: dict[str, str] = {
    _EX_2P: (
        "El escape de la caldera de una presión (600 °C, 100 kg/s) con un segundo nivel: alta a "
        "80 bar y 540 °C y baja a 6 bar y 200 °C, pinch 10 K y approach 5 K. Compará la "
        "chimenea con la de una presión."
    ),
    _EX_3P: (
        "Los mismos gases con tres niveles: alta a 100 bar y 540 °C, media a 20 bar y 240 °C y "
        "baja a 4 bar (vapor saturado). En cascada, la media no se puede sobrecalentar mucho: "
        "sus gases ya pasaron por el economizador de alta."
    ),
    _EX_MODERN: (
        "Una turbina de gas moderna (escape a 640 °C, 150 kg/s) con tres niveles y pinch 8 K: "
        "alta a 120 bar y 565 °C, media a 25 bar y 240 °C y baja a 4 bar."
    ),
    _EX_MODERN_2P: (
        "La misma turbina moderna con dos niveles: alta a 120 bar y 565 °C y baja a 5 bar, "
        "vapor saturado (el que se usa para desgasificar el agua de alimentación o para "
        "procesos). Compará con la de tres presiones."
    ),
}


# ---------------------------------------------------------------------
# Barridos y export
# ---------------------------------------------------------------------

MultiSweepParameter = Literal["p_low", "p_high", "pinch", "T_feedwater"]


@dataclass(frozen=True)
class MultiSweepPoint:
    """Un punto del barrido: valor del parámetro (SI) y resultados de la caldera."""

    value_si: float
    m_steam_kg_s: tuple[float, ...]
    T_stack_K: float
    Q_W: float
    exergy_efficiency: float


def _with(inputs: MultiHRSGInputs, parameter: MultiSweepParameter, value: float) -> MultiHRSGInputs:
    levels = list(inputs.levels)
    if parameter == "p_low":
        levels[-1] = replace(levels[-1], p_Pa=value)
    elif parameter == "p_high":
        levels[0] = replace(levels[0], p_Pa=value)
    elif parameter == "pinch":
        levels = [replace(lv, pinch_K=value) for lv in levels]
    else:
        return replace(inputs, T_feedwater_K=value)
    return replace(inputs, levels=tuple(levels))


def default_multi_sweep_values(
    inputs: MultiHRSGInputs, parameter: MultiSweepParameter, n: int = 9
) -> list[float]:
    """Valores razonables del parámetro alrededor de la caldera dada (SI)."""
    levels = inputs.levels
    if parameter == "pinch":
        return [float(v) for v in np.linspace(3.0, 30.0, n)]
    if parameter == "T_feedwater":
        low = levels[-1]
        top = _T_sat(low.p_Pa) - low.approach_K - 10.0
        return [float(v) for v in np.linspace(min(303.15, top - 10.0), top, n)]
    if parameter == "p_low":
        low = levels[-1]
        lo_T = inputs.T_feedwater_K + low.approach_K + 15.0
        hi_T = _T_sat(levels[-2].p_Pa) - levels[-2].approach_K - 15.0 if len(levels) > 1 else None
        if low.T_steam_K is not None:  # que el vapor de baja siga sobrecalentado
            hi_T = low.T_steam_K - 10.0 if hi_T is None else min(hi_T, low.T_steam_K - 10.0)
        lo = max(float(PropsSI("P", "T", lo_T, "Q", 0, Water)), 1.2e5)
        hi = float(PropsSI("P", "T", hi_T, "Q", 0, Water)) if hi_T is not None else 30e5
        hi = min(hi, 30e5)
        lo, hi = min(lo, low.p_Pa), max(hi, low.p_Pa)
        return [float(v) for v in np.geomspace(lo, hi, n)]
    if parameter == "p_high":
        high = levels[0]
        below = levels[1] if len(levels) > 1 else None
        lo = 1.5 * below.p_Pa if below is not None else 10e5
        hi_T = inputs.T_gas_in_K - high.pinch_K - 30.0
        if high.T_steam_K is not None:
            hi_T = min(hi_T, high.T_steam_K - 10.0)
        hi_T = min(hi_T, fluid_limits(Water).T_crit_K - 10.0)
        hi = min(float(PropsSI("P", "T", hi_T, "Q", 0, Water)), 160e5)
        lo, hi = min(lo, high.p_Pa), max(hi, high.p_Pa)
        return [float(v) for v in np.geomspace(lo, hi, n)]
    raise ValueError(f"Parámetro de barrido desconocido: {parameter!r}.")


def multi_hrsg_sweep(
    inputs: MultiHRSGInputs, parameter: MultiSweepParameter, values: Sequence[float]
) -> list[MultiSweepPoint]:
    """Resuelve la caldera para cada valor del parámetro; omite los que no tienen sentido."""
    points: list[MultiSweepPoint] = []
    for value in values:
        try:
            result = solve_multi_hrsg(_with(inputs, parameter, float(value)))
        except ValueError:
            continue
        points.append(
            MultiSweepPoint(
                float(value),
                tuple(lv.m_steam_kg_s for lv in result.levels),
                result.T_stack_K,
                result.Q_W,
                hrsg_exergy(result).efficiency,
            )
        )
    return points


def _value(value_si: float, kind: QuantityKind, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


_WATER_LABELS = (
    "entrada al economizador",
    "salida del economizador",
    "líquido saturado",
    "vapor saturado",
    "vapor sobrecalentado",
)


def multi_hrsg_to_dict(result: MultiHRSGResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``)."""
    inputs = result.inputs
    gas = inputs.gas
    exergy = hrsg_exergy(result)
    n = len(result.levels)
    return {
        "equipo": f"caldera de recuperación (HRSG) de {n} presi{'ón' if n == 1 else 'ones'}, "
        "en cascada",
        "sistema_de_unidades": system,
        "datos": {
            "T_gases_entrada": _value(inputs.T_gas_in_K, "temperature", system),
            "caudal_gases": _value(inputs.m_gas_kg_s, "mass_flow", system),
            "composicion_molar": {GAS_NAMES[s]: y for s, y in gas.mole_fractions.items()},
            "T_agua_alimentacion": _value(inputs.T_feedwater_K, "temperature", system),
            "niveles": [
                {
                    "nivel": lv.name or "único",
                    "p": _value(lv.level.p_Pa, "pressure", system),
                    "T_vapor": (
                        None
                        if lv.level.T_steam_K is None
                        else _value(lv.level.T_steam_K, "temperature", system)
                    ),
                    "pinch": _value(lv.level.pinch_K, "temperature_difference", system),
                    "approach": _value(lv.level.approach_K, "temperature_difference", system),
                }
                for lv in result.levels
            ],
        },
        "resultados": {
            "caudal_vapor_total": _value(result.m_steam_kg_s, "mass_flow", system),
            "T_chimenea": _value(result.T_stack_K, "temperature", system),
            "calor_total": _value(result.Q_W, "power", system),
            "aprovechamiento": result.recovery,
            "potencia_bombas": _value(result.W_pumps_W, "power", system),
            "niveles": {
                (lv.name or "único"): {
                    "caudal_vapor": _value(lv.m_steam_kg_s, "mass_flow", system),
                    "caudal_economizador": _value(lv.m_water_kg_s, "mass_flow", system),
                    "T_saturacion": _value(lv.T_sat_K, "temperature", system),
                    "estados": [
                        {
                            "estado": label,
                            "T": _value(s.T_K, "temperature", system),
                            "p": _value(s.P_Pa, "pressure", system),
                            "h": _value(s.h_J_per_kg, _EH, system),
                        }
                        for label, s in zip(_WATER_LABELS, lv.water, strict=False)
                    ],
                }
                for lv in result.levels
            },
            "recalentador": (
                None
                if result.reheat is None
                else {
                    "p": _value(result.reheat.reheater.p_Pa, "pressure", system),
                    "caudal": _value(result.reheat.m_kg_s, "mass_flow", system),
                    "vapor_de_media_sumado": _value(
                        result.reheat.m_middle_kg_s, "mass_flow", system
                    ),
                    "T_recalentamiento_frio": _value(result.reheat.cold.T_K, "temperature", system),
                    "T_entrada": _value(result.reheat.inlet.T_K, "temperature", system),
                    "T_salida": _value(result.reheat.hot.T_K, "temperature", system),
                    "calor": _value(result.reheat.Q_W, "power", system),
                }
            ),
            "gases": {label: _value(T, "temperature", system) for label, T, _ in result.gas_points},
            "secciones": {
                s.name: {
                    "calor": _value(s.Q_W, "power", system),
                    "T_gases_entrada": _value(s.T_gas_in_K, "temperature", system),
                    "T_gases_salida": _value(s.T_gas_out_K, "temperature", system),
                    "T_agua_entrada": _value(s.T_water_in_K, "temperature", system),
                    "T_agua_salida": _value(s.T_water_out_K, "temperature", system),
                }
                for s in result.sections
            },
            "exergia": {
                "T0": _value(exergy.T0_K, "temperature", system),
                "gases_que_entran": _value(exergy.X_gas_in_W, "power", system),
                "ganada_por_el_agua": _value(exergy.X_water_W, "power", system),
                "destruida": _value(exergy.X_destroyed_W, "power", system),
                "chimenea": _value(exergy.X_stack_W, "power", system),
                "rendimiento_exergetico": exergy.efficiency,
                "destruida_por_seccion": {
                    name: _value(x, "power", system) for name, x in exergy.destroyed_W
                },
            },
        },
        "fuente": "Balance de energía y exergía con gases ideales (CoolProp) y agua IAPWS-95",
    }
