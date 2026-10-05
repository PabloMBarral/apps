"""Ciclo de Rankine con TESPy — Fases 3.1a y 3.1b.

Ciclo de vapor de agua simple (bomba, caldera, turbina, condensador), ideal
o real (rendimientos isoentrópicos de turbina y bomba), con recalentamiento
en una etapa y con regeneración (hasta tres calentadores de agua de
alimentación, abiertos o cerrados; Cengel §10-6). Se resuelve con una red de
TESPy armada como los ejemplos oficiales: el tutorial
``tutorial/basics/rankine.py`` de TESPy 0.11 (``CycleCloser`` +
``SimpleHeatExchanger`` + ``Turbine`` + ``Pump``), el de optimización de una
central (calentadores cerrados como ``Condenser``, drenaje bombeado hacia
adelante a un ``Merge``) y el modelo SEGS de sus tests (drenajes en cascada
con ``Valve`` + ``Merge``); el calentador abierto es un ``Merge`` con
líquido saturado a la salida. El condensador se modela, como en Cengel, con
salida en líquido saturado (x = 0) y sin pérdidas de carga.

Numeración de estados de Çengel & Boles, *Termodinámica*, cap. 10 (ver
:mod:`core.cycles.rankine_layout`): 1 entrada a la bomba (líquido
saturado), 2 salida de la bomba, 3 entrada a la turbina, 4 salida de la
turbina. Con recalentamiento: 3 entrada a la turbina de alta, 4 salida de
alta (entrada al recalentador), 5 entrada a la turbina de baja, 6 salida de
baja. Con regeneración, como los ejemplos 10-5 y 10-6.

Todo en SI. No importa Streamlit. Fórmulas del vademecum: §3.3 (balance en
sistema abierto), §9.2 (rendimiento térmico y de Carnot), §10.4
(rendimientos isoentrópicos), §12 (vapor húmedo) y §13 (bomba de líquido).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from CoolProp.CoolProp import PropsSI
from tespy.components import (
    Condenser,
    CycleCloser,
    Merge,
    Pump,
    SimpleHeatExchanger,
    Splitter,
    Turbine,
    Valve,
)
from tespy.connections import Connection

from core.cycles.rankine_layout import (
    MAX_HEATERS,
    CycleComponent,
    FeedwaterHeater,
    PlantLayout,
    Port,
    heater_names,
    plant_layout,
)
from core.cycles.rankine_procedure import _bar, _degC, cooling_water_step, rankine_steps
from core.cycles.tespy_utils import new_network, solve
from core.fluids import FluidState, fluid_limits, fluid_state_from_pair, saturation_at_pressure
from core.state_report import states_table
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

# ``rankine_steps`` y ``cooling_water_step`` viven en
# :mod:`core.cycles.rankine_procedure`; se reexportan para que el import
# público de la Fase 3.1a (``from core.cycles.rankine import rankine_steps``)
# siga funcionando.
__all__ = [
    "FLUID",
    "RANKINE_EXAMPLES",
    "RANKINE_EXAMPLE_NOTES",
    "X_TURBINE_MIN",
    "CoolingWaterResult",
    "CycleComponent",
    "FeedwaterHeater",
    "RankineInputs",
    "RankineResult",
    "Reheat",
    "SweepParameter",
    "SweepPoint",
    "condenser_cooling_water",
    "cooling_water_step",
    "default_extraction_values",
    "default_heater_pressures",
    "default_sweep_values",
    "extraction_pressure_sweep",
    "rankine_isentropic_states",
    "rankine_labeled_states",
    "rankine_notes",
    "rankine_segments",
    "rankine_steps",
    "rankine_sweep",
    "rankine_to_dict",
    "solve_rankine",
    "validate_rankine_inputs",
]

#: Fluido de trabajo (nombre de CoolProp).
FLUID = "Water"

_EH: QuantityKind = "specific_enthalpy"

# Margen mínimo sobre T_sat para la entrada a una turbina (K): más cerca,
# CoolProp no distingue el estado de la campana con T y p.
_T_SAT_TOL = 0.01

# Título de salida de turbina por debajo del cual se avisa (Cengel §10-4:
# más de 10–12 % de humedad erosiona los álabes).
X_TURBINE_MIN = 0.88


# ---------------------------------------------------------------------
# Datos y resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Reheat:
    """Recalentamiento: el vapor que sale de la turbina de alta vuelve a la
    caldera a presión constante ``p_Pa`` y entra a la de baja a ``T_K``."""

    p_Pa: float
    T_K: float


@dataclass(frozen=True)
class RankineInputs:
    """Datos del ciclo, en SI.

    ``T_turbine_in_K=None`` indica vapor saturado seco (x = 1) a la entrada
    de la turbina. Se da el caudal másico **o** la potencia neta; el otro
    se calcula. El rendimiento ``eta_turbine`` vale para cada turbina (la
    de alta y la de baja con recalentamiento) y se mide desde su entrada:
    las extracciones y la salida quedan sobre la misma línea de expansión.

    ``heaters`` son los calentadores de agua de alimentación, de menor a
    mayor presión (Cengel §10-6). ``drain_forward`` bombea hacia adelante
    el drenaje del de mayor presión (que tiene que ser cerrado); los demás
    drenajes van en cascada hacia atrás.
    """

    p_boiler_Pa: float
    T_turbine_in_K: float | None
    p_condenser_Pa: float
    eta_turbine: float = 1.0
    eta_pump: float = 1.0
    reheat: Reheat | None = None
    m_dot_kg_s: float | None = 1.0
    W_net_W: float | None = None
    heaters: tuple[FeedwaterHeater, ...] = ()
    drain_forward: bool = False

    def __post_init__(self) -> None:
        # Se acepta una lista (la página la arma así); se guarda como tupla para
        # que los datos sigan siendo inmutables y sirvan de clave de caché.
        object.__setattr__(self, "heaters", tuple(self.heaters))

    @property
    def is_ideal(self) -> bool:
        """Turbina y bomba isoentrópicas (el ciclo ideal de Rankine)."""
        return self.eta_turbine == 1.0 and self.eta_pump == 1.0

    @property
    def layout(self) -> PlantLayout:
        """Plano de la planta con la numeración de estados (sin resolver)."""
        return plant_layout(
            heaters=self.heaters,
            reheat_p_Pa=None if self.reheat is None else self.reheat.p_Pa,
            drain_forward=self.drain_forward,
        )


@dataclass(frozen=True)
class RankineResult:
    """Estados y balances del ciclo, en SI.

    ``states`` sigue la numeración de Cengel (1 … 4, o 1 … 6 con
    recalentamiento; con regeneración, la de :mod:`core.cycles.rankine_layout`).
    ``pump_out_s`` es el estado 2s y ``turbine_out_s`` tiene la salida
    isoentrópica de cada tramo de turbina, medida desde la entrada de su
    turbina (4s, o 4s y 6s; con extracciones, una por tramo).
    ``tespy_balances`` son los resultados por componente que calcula TESPy
    para 1 kg/s en la caldera (potencia o calor en W, positivos hacia el
    fluido), como control cruzado de los balances con entalpías;
    ``tespy_heaters`` es el calor que cada calentador cerrado pasa de la
    extracción al agua de alimentación (calor interno del ciclo).

    ``components`` son los componentes del ciclo con el caudal de cada
    corriente (ṁ/ṁ_caldera): todas las energías específicas son por
    kilogramo de vapor que pasa por la caldera, como en Cengel.
    """

    inputs: RankineInputs
    states: tuple[FluidState, ...]
    pump_out_s: FluidState
    turbine_out_s: tuple[FluidState, ...]
    m_dot_kg_s: float
    tespy_balances: tuple[tuple[str, float], ...] = ()
    layout: PlantLayout | None = None
    components: tuple[CycleComponent, ...] = ()
    pumps_out_s: tuple[FluidState, ...] = ()
    tespy_heaters: tuple[tuple[str, float], ...] = ()

    @property
    def h_pump_out_s_J_per_kg(self) -> float:
        return self.pump_out_s.h_J_per_kg

    @property
    def h_turbine_out_s_J_per_kg(self) -> tuple[float, ...]:
        return tuple(state.h_J_per_kg for state in self.turbine_out_s)

    # --- componentes e índices de los estados --------------------------
    @property
    def has_reheat(self) -> bool:
        return self.inputs.reheat is not None

    @property
    def has_heaters(self) -> bool:
        return bool(self.inputs.heaters)

    def of_kind(self, kind: str) -> tuple[CycleComponent, ...]:
        """Componentes de un tipo (``"turbine"``, ``"pump"``, ``"closed_heater"``…)."""
        return tuple(c for c in self.components if c.kind == kind)

    @property
    def turbine_pairs(self) -> tuple[tuple[int, int], ...]:
        """(entrada, salida) de cada tramo de turbina, como índices de ``states``."""
        return tuple((c.port("in").state, c.port("out").state) for c in self.of_kind("turbine"))

    @property
    def heating_pairs(self) -> tuple[tuple[int, int], ...]:
        """(entrada, salida) de cada aporte de calor: caldera y recalentador."""
        return tuple(
            (c.port("in").state, c.port("out").state)
            for c in self.components
            if c.kind in ("boiler", "reheater")
        )

    @property
    def exhaust(self) -> int:
        """Índice del escape de la (última) turbina, que entra al condensador."""
        assert self.layout is not None
        return self.layout.exhaust

    @property
    def extraction_fractions(self) -> tuple[float, ...]:
        """Fracción de extracción y = ṁ_ext/ṁ_caldera de cada calentador
        (de menor a mayor presión)."""
        heaters = sorted(
            (c for c in self.components if c.kind in ("open_heater", "closed_heater")),
            key=lambda c: c.heater if c.heater is not None else -1,
        )
        return tuple(c.port("bleed").fraction for c in heaters)

    def _h(self, port: Port) -> float:
        return port.fraction * self.states[port.state].h_J_per_kg

    def _flow_through(self, kind: str) -> float:
        """Σ (y·h)_salida − Σ (y·h)_entrada de los componentes de ese tipo."""
        return sum(
            sum(self._h(p) for p in c.outlets) - sum(self._h(p) for p in c.inlets)
            for c in self.of_kind(kind)
        )

    # --- energías específicas (J/kg de vapor de caldera) -----------------
    @property
    def w_pump_J_per_kg(self) -> float:
        return self._flow_through("pump")

    @property
    def w_turbine_J_per_kg(self) -> float:
        return -self._flow_through("turbine")

    @property
    def q_in_J_per_kg(self) -> float:
        return self._flow_through("boiler") + self._flow_through("reheater")

    @property
    def q_out_J_per_kg(self) -> float:
        return -self._flow_through("condenser")

    @property
    def w_net_J_per_kg(self) -> float:
        return self.w_turbine_J_per_kg - self.w_pump_J_per_kg

    # --- rendimientos ---------------------------------------------------
    @property
    def eta_th(self) -> float:
        """η = w_neto / q_H (vademecum §9.2)."""
        return self.w_net_J_per_kg / self.q_in_J_per_kg

    @property
    def back_work_ratio(self) -> float:
        """Relación de trabajo de retroceso w_B / w_T (Cengel §9-8)."""
        return self.w_pump_J_per_kg / self.w_turbine_J_per_kg

    @property
    def x_turbine_out(self) -> float | None:
        """Título a la salida de la (última) turbina; ``None`` si sale sobrecalentado."""
        return self.states[self.exhaust].x

    @property
    def T_low_K(self) -> float:
        """Temperatura de condensación T_sat(p_cond) (sumidero del ciclo)."""
        return self.states[0].T_K

    @property
    def T_high_K(self) -> float:
        """Temperatura máxima del ciclo (entrada a turbina o recalentamiento)."""
        return max(self.states[o].T_K for _, o in self.heating_pairs)

    @property
    def eta_carnot(self) -> float:
        """Carnot entre T_H = T máx. y T_C = T de condensación (vademecum §9.2)."""
        return 1.0 - self.T_low_K / self.T_high_K

    @property
    def T_mean_in_K(self) -> float:
        """Temperatura media de aporte de calor T̄_H = q_H / Δs (Cengel §10-4)."""
        delta_s = sum(
            c.port("in").fraction
            * (
                self.states[c.port("out").state].s_J_per_kg_K
                - self.states[c.port("in").state].s_J_per_kg_K
            )
            for c in self.components
            if c.kind in ("boiler", "reheater")
        )
        return self.q_in_J_per_kg / delta_s

    @property
    def eta_mean_temperature(self) -> float:
        """1 − T_C / T̄_H: igual a η solo en el ciclo ideal sin regeneración y con
        salida húmeda de la turbina (todo el calor se cede a T_C constante). Con
        regeneración, la mezcla y la transferencia de calor con diferencia de
        temperatura en los calentadores generan entropía y η queda por debajo."""
        return 1.0 - self.T_low_K / self.T_mean_in_K

    # --- potencias (W) --------------------------------------------------
    @property
    def W_net_W(self) -> float:
        return self.m_dot_kg_s * self.w_net_J_per_kg

    @property
    def W_turbine_W(self) -> float:
        return self.m_dot_kg_s * self.w_turbine_J_per_kg

    @property
    def W_pump_W(self) -> float:
        return self.m_dot_kg_s * self.w_pump_J_per_kg

    @property
    def Q_in_W(self) -> float:
        return self.m_dot_kg_s * self.q_in_J_per_kg

    @property
    def Q_out_W(self) -> float:
        return self.m_dot_kg_s * self.q_out_J_per_kg


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


def _T_sat(p_Pa: float) -> float:
    return saturation_at_pressure(FLUID, p_Pa).T_sat_K


def _check_superheated(T_K: float, p_Pa: float, where: str, at_saturation: str) -> None:
    """La entrada a una turbina tiene que ser vapor (sobrecalentado o supercrítico).

    ``at_saturation`` es la sugerencia para el alumno si T es justo la de saturación.
    """
    lim = fluid_limits(FLUID)
    if T_K > lim.T_max_K:
        raise ValueError(
            f"La temperatura de {where} ({_degC(T_K)}) supera el máximo de validez de la "
            f"ecuación de estado del agua ({_degC(lim.T_max_K)})."
        )
    if p_Pa >= lim.P_crit_Pa:
        if T_K <= lim.T_crit_K:
            raise ValueError(
                f"Con presión supercrítica ({_bar(p_Pa)}), la temperatura de {where} tiene que "
                f"superar la crítica ({_degC(lim.T_crit_K)}): si no, a la turbina entraría líquido."
            )
        return
    T_sat = _T_sat(p_Pa)
    if T_K < T_sat - _T_SAT_TOL:
        raise ValueError(
            f"Con T = {_degC(T_K)} y p = {_bar(p_Pa)}, el agua de {where} sería líquida: a esa "
            f"presión hierve a T_sat = {_degC(T_sat)}. A una turbina tiene que entrar vapor."
        )
    if T_K <= T_sat + _T_SAT_TOL:
        raise ValueError(
            f"T = {_degC(T_K)} es justo la de saturación a {_bar(p_Pa)}: con T y p solas no se "
            f"sabe cuánto vapor hay. {at_saturation}"
        )


def validate_rankine_inputs(inputs: RankineInputs) -> None:
    """Verifica que el ciclo tenga sentido físico; si no, explica por qué.

    Raises
    ------
    ValueError
        Con un mensaje en castellano dirigido al alumno.
    """
    lim = fluid_limits(FLUID)
    values = (inputs.p_boiler_Pa, inputs.p_condenser_Pa, inputs.eta_turbine, inputs.eta_pump)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Todos los datos del ciclo tienen que ser números finitos.")
    for name, eta in (("de la turbina", inputs.eta_turbine), ("de la bomba", inputs.eta_pump)):
        if not 0.0 < eta <= 1.0:
            raise ValueError(
                f"El rendimiento isoentrópico {name} tiene que estar entre 0 y 1 (η = {eta:g}); "
                "1 corresponde al proceso ideal (isoentrópico)."
            )

    p_hi, p_lo = inputs.p_boiler_Pa, inputs.p_condenser_Pa
    if not lim.P_triple_Pa <= p_lo < lim.P_crit_Pa:
        raise ValueError(
            f"En el condensador el vapor condensa a T_sat(p): la presión ({_bar(p_lo)}) tiene "
            f"que estar entre la del punto triple ({_bar(lim.P_triple_Pa)}) y la crítica "
            f"({_bar(lim.P_crit_Pa)})."
        )
    if p_lo >= p_hi:
        raise ValueError(
            f"La presión del condensador ({_bar(p_lo)}) tiene que ser menor que la de la caldera "
            f"({_bar(p_hi)}): la bomba sube la presión y la turbina expande el vapor."
        )
    if p_hi > lim.P_max_Pa:
        raise ValueError(
            f"La presión de la caldera ({_bar(p_hi)}) supera el máximo de validez de la "
            f"ecuación de estado del agua ({_bar(lim.P_max_Pa)})."
        )

    if inputs.T_turbine_in_K is None:
        if p_hi >= lim.P_crit_Pa:
            raise ValueError(
                f"Con presión supercrítica ({_bar(p_hi)}) no hay vapor saturado: indicá la "
                "temperatura de entrada a la turbina."
            )
    else:
        if not math.isfinite(inputs.T_turbine_in_K):
            raise ValueError("La temperatura de entrada a la turbina tiene que ser un número.")
        _check_superheated(
            inputs.T_turbine_in_K,
            p_hi,
            "entrada a la turbina",
            "Si entra vapor saturado seco, elegí «Vapor saturado seco (x₃ = 1)».",
        )

    if inputs.reheat is not None:
        rh = inputs.reheat
        if not (math.isfinite(rh.p_Pa) and p_lo < rh.p_Pa < p_hi):
            raise ValueError(
                f"La presión de recalentamiento ({_bar(rh.p_Pa)}) tiene que estar entre la del "
                f"condensador ({_bar(p_lo)}) y la de la caldera ({_bar(p_hi)}): el vapor se "
                "expande en la turbina de alta hasta esa presión, se recalienta y sigue en la de "
                "baja."
            )
        _check_superheated(
            rh.T_K,
            rh.p_Pa,
            "entrada a la turbina de baja",
            "El recalentador tiene que dejar el vapor sobrecalentado: subí su temperatura.",
        )

    given = [v for v in (inputs.m_dot_kg_s, inputs.W_net_W) if v is not None]
    if len(given) != 1:
        raise ValueError("Indicá el caudal másico o la potencia neta (uno de los dos).")
    if not (math.isfinite(given[0]) and given[0] > 0.0):
        raise ValueError("El caudal másico o la potencia neta tiene que ser positivo.")

    _validate_heaters(inputs)


def _pump_outlet_h(p_from_Pa: float, p_to_Pa: float, eta: float) -> float:
    """h a la salida de una bomba que toma líquido saturado a ``p_from_Pa``."""
    inlet = fluid_state_from_pair(FLUID, "PX", p=p_from_Pa, x=0.0)
    h_s = _isentropic_state(p_to_Pa, inlet.s_J_per_kg_K).h_J_per_kg
    return inlet.h_J_per_kg + (h_s - inlet.h_J_per_kg) / eta


def default_heater_pressures(p_boiler_Pa: float, p_condenser_Pa: float, n: int) -> list[float]:
    """Presiones de extracción para ``n`` calentadores (SI), de menor a mayor.

    Reparten por igual el salto de T_sat entre el condensador y la caldera
    (tope: 0,9·p_crit), una regla práctica cercana al óptimo (Cengel §10-6):
    con un calentador en el ejemplo 10-5 da 13,7 bar, y el máximo de η está
    entre 12 y 20 bar.
    """
    lim = fluid_limits(FLUID)
    p_top = min(p_boiler_Pa, 0.9 * lim.P_crit_Pa)
    T_low, T_high = _T_sat(p_condenser_Pa), _T_sat(p_top)
    pressures = []
    for k in range(1, n + 1):
        T = T_low + k * (T_high - T_low) / (n + 1)
        pressures.append(float(PropsSI("P", "T", T, "Q", 0, "Water")))
    return pressures


def _feedwater_inlet_T(inputs: RankineInputs) -> dict[int, float]:
    """T del agua de alimentación que le llega a cada calentador, antes de resolver.

    En cada tramo de la línea una bomba toma líquido saturado (del condensador
    o de un abierto) y lo lleva a la presión del próximo abierto o de la
    caldera; los cerrados de ese tramo, en orden, dejan el agua a
    T_sat(p_ext) − TTD (Cengel §10-6).
    """
    heaters = inputs.heaters
    opens = [i for i, h in enumerate(heaters) if h.kind == "open"]
    starts = [inputs.p_condenser_Pa, *(heaters[i].p_Pa for i in opens)]
    lines = [*(heaters[i].p_Pa for i in opens), inputs.p_boiler_Pa]
    T_in: dict[int, float] = {}
    for stage, (p_from, p_line) in enumerate(zip(starts, lines, strict=True)):
        h = _pump_outlet_h(p_from, p_line, inputs.eta_pump)
        T = fluid_state_from_pair(FLUID, "PH", p=p_line, h=h).T_K
        closed = [
            i
            for i, h_ in enumerate(heaters)
            if h_.kind == "closed" and sum(1 for j in opens if j < i) == stage
        ]
        for i in closed:
            T_in[i] = T
            T = _T_sat(heaters[i].p_Pa) - heaters[i].ttd_K
        if stage < len(opens):
            T_in[opens[stage]] = T
    return T_in


def _validate_heaters(inputs: RankineInputs) -> None:
    """Calentadores de agua de alimentación (Cengel §10-6): presiones, tipos y TTD."""
    heaters = inputs.heaters
    if not heaters:
        return
    if len(heaters) > MAX_HEATERS:
        raise ValueError(
            f"Se admiten hasta {MAX_HEATERS} calentadores de agua de alimentación "
            f"(cargaste {len(heaters)})."
        )
    lim = fluid_limits(FLUID)
    names = heater_names(heaters)
    p_lo, p_hi = inputs.p_condenser_Pa, inputs.p_boiler_Pa
    for name, h in zip(names, heaters, strict=True):
        if h.kind not in ("open", "closed"):
            raise ValueError(f"El {name} tiene que ser abierto o cerrado.")
        if not (math.isfinite(h.p_Pa) and p_lo < h.p_Pa < p_hi):
            raise ValueError(
                f"La presión de extracción del {name} ({_bar(h.p_Pa)}) tiene que estar entre la "
                f"del condensador ({_bar(p_lo)}) y la de la caldera ({_bar(p_hi)}): el vapor se "
                "saca de la turbina en algún punto de la expansión."
            )
        if h.p_Pa >= lim.P_crit_Pa:
            raise ValueError(
                f"La presión de extracción del {name} ({_bar(h.p_Pa)}) tiene que ser menor que "
                f"la crítica ({_bar(lim.P_crit_Pa)}): el calentador condensa la extracción hasta "
                "líquido saturado, y por encima de la crítica no hay saturación."
            )
        if not (math.isfinite(h.ttd_K) and h.ttd_K >= 0.0):
            raise ValueError(
                f"La diferencia terminal (TTD) del {name} no puede ser negativa en este modelo "
                "(sin desrecalentador): el agua de alimentación sale, como mucho, a la "
                "temperatura de saturación de la extracción (TTD = 0, el calentador ideal)."
            )
        if h.kind == "open" and h.ttd_K != 0.0:
            raise ValueError(
                f"El {name} es abierto: el agua sale como líquido saturado a la presión de "
                "extracción, así que no lleva TTD (es solo para los cerrados)."
            )
    for (n1, a), (n2, b) in zip(
        zip(names, heaters, strict=True), zip(names[1:], heaters[1:], strict=True), strict=False
    ):
        if not b.p_Pa > a.p_Pa or math.isclose(a.p_Pa, b.p_Pa, rel_tol=1e-4):
            raise ValueError(
                f"Las presiones de extracción tienen que crecer de un calentador al siguiente: "
                f"el {n1} ({_bar(a.p_Pa)}) tiene que extraer a menos presión que el {n2} "
                f"({_bar(b.p_Pa)}), y no pueden ser iguales."
            )
    if inputs.drain_forward and heaters[-1].kind != "closed":
        raise ValueError(
            "Solo un calentador cerrado puede bombear su drenaje hacia adelante, y el de mayor "
            f"presión ({names[-1]}) es abierto."
        )
    for i, T_in in _feedwater_inlet_T(inputs).items():
        h = heaters[i]
        T_max = _T_sat(h.p_Pa) - h.ttd_K
        if T_max <= T_in + _T_SAT_TOL:
            raise ValueError(
                f"El {names[i]} no puede calentar el agua de alimentación: le llega a "
                f"{_degC(T_in)} y, con la extracción a {_bar(h.p_Pa)}, como mucho la deja a "
                f"T_sat − TTD = {_degC(T_max)}. Subí su presión de extracción"
                + (", bajá el TTD" if h.kind == "closed" else "")
                + " o separala más de la del calentador anterior."
            )


# ---------------------------------------------------------------------
# Cálculo con TESPy
# ---------------------------------------------------------------------


def _state_from_connection(connection: Connection) -> FluidState:
    """Estado completo (región, x, v, u …) a partir de p y h de TESPy (SI)."""
    return fluid_state_from_pair(
        FLUID, "PH", p=float(connection.p.val_SI), h=float(connection.h.val_SI)
    )


def _isentropic_state(p_out_Pa: float, s_in: float) -> FluidState:
    """Estado a la presión de salida con la entropía de entrada (2s, 4s, 6s)."""
    return fluid_state_from_pair(FLUID, "PS", p=p_out_Pa, s=s_in)


def _turbine_inlet_state(inputs: RankineInputs, casing: str) -> FluidState:
    """Estado a la entrada de cada turbina: caldera o recalentador."""
    if casing == "turbina de baja":
        assert inputs.reheat is not None
        return fluid_state_from_pair(FLUID, "TP", t=inputs.reheat.T_K, p=inputs.reheat.p_Pa)
    if inputs.T_turbine_in_K is None:
        return fluid_state_from_pair(FLUID, "PX", p=inputs.p_boiler_Pa, x=1.0)
    return fluid_state_from_pair(FLUID, "TP", t=inputs.T_turbine_in_K, p=inputs.p_boiler_Pa)


def _section_end_pressure(inputs: RankineInputs, layout: PlantLayout, out_state: int) -> float:
    """Presión a la salida de un tramo de turbina (extracción, recalentamiento o escape)."""
    if out_state == layout.exhaust:
        return inputs.p_condenser_Pa
    if out_state == layout.reheat_in:
        assert inputs.reheat is not None
        return inputs.reheat.p_Pa
    return inputs.heaters[layout.extraction_states.index(out_state)].p_Pa


def _section_efficiencies(inputs: RankineInputs, layout: PlantLayout) -> dict[str, float]:
    """``eta_s`` de cada tramo de turbina para TESPy.

    η_T se mide desde la entrada de cada turbina, como en las soluciones de
    Cengel: h_k = h_ent − η_T·(h_ent − h_ks), con ks a la presión de la
    extracción y la entropía de la entrada. El primer tramo lleva η_T; los
    siguientes, el rendimiento local que reproduce esa misma línea de
    expansión (un poco menor, porque los saltos isoentrópicos de los tramos
    suman más que el de la turbina entera).
    """
    eta = inputs.eta_turbine
    sections = layout.of_kind("turbine")
    if eta == 1.0:
        return {c.label: 1.0 for c in sections}
    inlets: dict[str, FluidState] = {}
    previous: dict[str, tuple[float, float]] = {}
    efficiencies: dict[str, float] = {}
    for c in sections:
        inlet = inlets.setdefault(c.casing, _turbine_inlet_state(inputs, c.casing))
        p_end = _section_end_pressure(inputs, layout, c.port("out").state)
        h_s = _isentropic_state(p_end, inlet.s_J_per_kg_K).h_J_per_kg
        h_target = inlet.h_J_per_kg - eta * (inlet.h_J_per_kg - h_s)
        if c.casing not in previous:
            efficiencies[c.label] = eta
        else:
            h_prev, s_prev = previous[c.casing]
            h_s_local = _isentropic_state(p_end, s_prev).h_J_per_kg
            efficiencies[c.label] = (h_prev - h_target) / (h_prev - h_s_local)
        target = fluid_state_from_pair(FLUID, "PH", p=p_end, h=h_target)
        previous[c.casing] = (h_target, target.s_J_per_kg_K)
    return efficiencies


@dataclass
class _Network:
    """Red de TESPy armada desde el layout, con el mapa a los estados lógicos."""

    network: Any
    states: dict[int, Connection]  # conexión principal de cada estado
    ports: dict[tuple[int, int], Connection]  # (componente, conexión) -> conexión de TESPy
    objects: list[Any]  # objeto de TESPy de cada componente lógico


def _build_network(inputs: RankineInputs, layout: PlantLayout) -> _Network:
    """Red de TESPy de la planta (ejemplos oficiales de TESPy 0.11, ver el módulo).

    La presión se fija una sola vez por grupo (entrada de la turbina, salida
    de cada tramo); bombas, válvulas y ``Merge`` la toman de la red. Sin
    calentadores, los componentes, las etiquetas y los datos son los de la
    0.11.0.
    """
    network = new_network()
    closer = CycleCloser("cierre del ciclo")
    eta_sections = _section_efficiencies(inputs, layout)
    objects: list[Any] = []
    inlets: dict[tuple[int, int], tuple[Any, str]] = {}
    outlets: dict[tuple[int, int], tuple[Any, str]] = {}
    aux: list[tuple[Any, str, Any, str, str]] = []

    for ci, comp in enumerate(layout.components):
        roles = [p.role for p in comp.ports]

        def at(role: str, k: int = 0, *, _roles: list[str] = roles) -> int:
            return [i for i, r in enumerate(_roles) if r == role][k]

        if comp.kind == "pump":
            obj: Any = Pump(comp.label)
            obj.set_attr(eta_s=inputs.eta_pump)
        elif comp.kind in ("boiler", "reheater", "condenser"):
            obj = SimpleHeatExchanger(comp.label)
            obj.set_attr(pr=1)  # sin pérdidas de carga, como en Cengel
        elif comp.kind == "turbine":
            obj = Turbine(comp.label)
            obj.set_attr(eta_s=eta_sections[comp.label])
        elif comp.kind == "valve":
            obj = Valve(comp.label)
        elif comp.kind == "open_heater":
            obj = Merge(comp.label, num_in=len(comp.inlets))
        elif comp.kind == "mixer":
            obj = Merge(comp.label, num_in=2)
        else:  # calentador cerrado
            assert comp.heater is not None
            obj = Condenser(comp.label)
            obj.set_attr(pr1=1, pr2=1, ttd_u=inputs.heaters[comp.heater].ttd_K)
        objects.append(obj)

        if comp.kind in ("pump", "boiler", "reheater", "turbine", "valve"):
            inlets[(ci, at("in"))] = (obj, "in1")
            outlets[(ci, at("out"))] = (obj, "out1")
        elif comp.kind == "condenser":
            outlets[(ci, at("out"))] = (obj, "out1")
            drains = comp.ports_with("drain_in")
            if drains:
                merge = Merge("mezcla antes del condensador", num_in=1 + len(drains))
                inlets[(ci, at("in"))] = (merge, "in1")
                for k in range(len(drains)):
                    inlets[(ci, at("drain_in", k))] = (merge, f"in{k + 2}")
                aux.append((merge, "out1", obj, "in1", "entrada al condensador"))
            else:
                inlets[(ci, at("in"))] = (obj, "in1")
        elif comp.kind in ("open_heater", "mixer"):
            for k, (pi, _port) in enumerate(
                (i, p) for i, p in enumerate(comp.ports) if p.role != "out"
            ):
                inlets[(ci, pi)] = (obj, f"in{k + 1}")
            outlets[(ci, at("out"))] = (obj, "out1")
        else:  # calentador cerrado: lado caliente in1/out1, agua de alimentación in2/out2
            inlets[(ci, at("fw_in"))] = (obj, "in2")
            outlets[(ci, at("fw_out"))] = (obj, "out2")
            outlets[(ci, at("drain_out"))] = (obj, "out1")
            drains = comp.ports_with("drain_in")
            if drains:
                merge = Merge(f"mezcla en el {comp.label}", num_in=1 + len(drains))
                inlets[(ci, at("bleed"))] = (merge, "in1")
                for k in range(len(drains)):
                    inlets[(ci, at("drain_in", k))] = (merge, f"in{k + 2}")
                aux.append((merge, "out1", obj, "in1", f"entrada al {comp.label}"))
            else:
                inlets[(ci, at("bleed"))] = (obj, "in1")

    # La caldera entrega al cierre del ciclo, y el cierre a la turbina (tutorial).
    boiler_i = next(i for i, c in enumerate(layout.components) if c.kind == "boiler")
    boiler_out = (boiler_i, [p.role for p in layout.components[boiler_i].ports].index("out"))
    boiler_obj, _ = outlets.pop(boiler_out)
    aux_boiler = Connection(boiler_obj, "out1", closer, "in1", label="0")

    producers: dict[int, tuple[Any, str, tuple[int, int] | None]] = {
        layout.turbine_inlet: (closer, "out1", None)
    }
    for key, (obj, port_id) in outlets.items():
        producers[layout.components[key[0]].ports[key[1]].state] = (obj, port_id, key)
    consumers: dict[int, list[tuple[tuple[int, int], Any, str]]] = {}
    for key, (obj, port_id) in inlets.items():
        state = layout.components[key[0]].ports[key[1]].state
        consumers.setdefault(state, []).append((key, obj, port_id))

    states: dict[int, Connection] = {}
    ports: dict[tuple[int, int], Connection] = {boiler_out: aux_boiler}
    connections: list[Connection] = [aux_boiler]
    for state in range(layout.n_states):
        number = state + 1
        src, src_id, src_key = producers[state]
        targets = consumers[state]
        if len(targets) == 1:
            key, dst, dst_id = targets[0]
            conn = Connection(src, src_id, dst, dst_id, label=str(number))
            ports[key] = conn
        else:
            # Extracción: el vapor sigue (turbina o recalentador) y una parte sale.
            splitter = Splitter(f"extracción en el estado {number}", num_out=2)
            conn = Connection(src, src_id, splitter, "in1", label=str(number))
            ordered = sorted(
                targets, key=lambda t: layout.components[t[0][0]].ports[t[0][1]].role == "bleed"
            )
            for k, (key, dst, dst_id) in enumerate(ordered):
                suffix = "sigue" if k == 0 else "extracción"
                branch = Connection(
                    splitter, f"out{k + 1}", dst, dst_id, label=f"{number} {suffix}"
                )
                ports[key] = branch
                connections.append(branch)
        if src_key is not None:
            ports[src_key] = conn
        states[state] = conn
        connections.append(conn)
    for src, src_id, dst, dst_id, label in aux:
        connections.append(Connection(src, src_id, dst, dst_id, label=label))
    network.add_conns(*connections)

    # Datos (una sola vez cada presión).
    inlet: dict[str, Any] = {"p": inputs.p_boiler_Pa, "m": 1.0, "fluid": {"water": 1}}
    if inputs.T_turbine_in_K is None:
        inlet["x"] = 1.0
    else:
        inlet["T"] = inputs.T_turbine_in_K
    states[layout.turbine_inlet].set_attr(**inlet)
    for comp in layout.of_kind("turbine"):
        out = comp.port("out").state
        states[out].set_attr(p=_section_end_pressure(inputs, layout, out))
    if inputs.reheat is not None and layout.reheat_out is not None:
        states[layout.reheat_out].set_attr(T=inputs.reheat.T_K)
    states[0].set_attr(x=0.0)  # el condensador entrega líquido saturado
    for comp in layout.of_kind("open_heater"):
        states[comp.port("out").state].set_attr(x=0.0)  # Cengel: sale líquido saturado
    return _Network(network=network, states=states, ports=ports, objects=objects)


def _check_heater_flows(components: tuple[CycleComponent, ...]) -> None:
    """Después de resolver: cada extracción y cada tramo de turbina con caudal positivo.

    TESPy no avisa una extracción negativa hacia un calentador abierto (un
    ``Merge`` no tiene límites): puede pasar si el agua ya le llega caliente o
    si un drenaje en cascada le aporta todo el calor.
    """
    for comp in components:
        if comp.kind in ("open_heater", "closed_heater"):
            y = comp.port("bleed").fraction
            if not 0.0 < y < 1.0:
                raise ValueError(
                    f"Con estos datos el {comp.label} necesitaría una extracción imposible "
                    f"(y = {y:.4f}): el agua de alimentación le llega casi tan caliente como la "
                    "puede dejar, o el drenaje que recibe ya le aporta todo el calor. Separá más "
                    "las presiones de extracción."
                )
    for comp in components:
        if comp.kind == "turbine" and comp.port("in").fraction <= 0.0:
            raise ValueError(
                f"Las extracciones se llevan todo el vapor antes de la {comp.label}: separá más "
                "las presiones de extracción."
            )


def solve_rankine(inputs: RankineInputs) -> RankineResult:
    """Resuelve el ciclo con TESPy (Witte & Tuschy, 2020) y arma el resultado.

    La red sigue los ejemplos oficiales de TESPy 0.11 (ver el módulo): un
    ``CycleCloser`` cierra el lazo, la caldera, el recalentador y el
    condensador son ``SimpleHeatExchanger`` sin pérdida de carga (``pr = 1``),
    turbinas y bombas llevan su ``eta_s`` y los calentadores son ``Merge``
    (abiertos) o ``Condenser`` (cerrados). Se resuelve por kilogramo de vapor
    en la caldera (1 kg/s) y las potencias se escalan con el caudal.

    Raises
    ------
    ValueError
        Si los datos no tienen sentido físico (ver
        :func:`validate_rankine_inputs`) o TESPy no converge.
    """
    validate_rankine_inputs(inputs)
    layout = inputs.layout
    built = _build_network(inputs, layout)
    try:
        solve(built.network, what="el ciclo de Rankine")
    except ValueError as exc:
        if not inputs.heaters:
            raise
        raise ValueError(
            f"{exc} Con regeneración suele pasar cuando dos presiones de extracción están muy "
            "cerca o un calentador casi no tiene nada que calentar."
        ) from exc

    # TESPy da p y h; el resto del estado (región, título, v, u, s) se arma
    # con CoolProp: el título de las conexiones de TESPy no es confiable fuera
    # de la campana.
    states = tuple(_state_from_connection(built.states[i]) for i in range(layout.n_states))
    m_boiler = float(built.states[layout.turbine_inlet].m.val_SI)
    components = tuple(
        replace(
            comp,
            ports=tuple(
                replace(port, fraction=float(built.ports[(ci, pi)].m.val_SI) / m_boiler)
                for pi, port in enumerate(comp.ports)
            ),
        )
        for ci, comp in enumerate(layout.components)
    )
    if inputs.heaters:
        _check_heater_flows(components)

    pumps = [c for c in components if c.kind == "pump"]
    pumps_out_s = tuple(
        _isentropic_state(states[c.port("out").state].P_Pa, states[c.port("in").state].s_J_per_kg_K)
        for c in pumps
    )
    casing_inlet: dict[str, int] = {}
    for c in components:
        if c.kind == "turbine":
            casing_inlet.setdefault(c.casing, c.port("in").state)
    turbine_out_s = tuple(
        _isentropic_state(
            states[c.port("out").state].P_Pa, states[casing_inlet[c.casing]].s_J_per_kg_K
        )
        for c in components
        if c.kind == "turbine"
    )
    objects = dict(zip((c.label for c in components), built.objects, strict=True))
    balances = tuple(
        (
            c.label,
            float(
                (objects[c.label].P if c.kind in ("turbine", "pump") else objects[c.label].Q).val_SI
            ),
        )
        for kind in ("turbine", "pump", "boiler", "reheater", "condenser")
        for c in components
        if c.kind == kind
    )
    heater_heat = tuple(
        (c.label, -float(objects[c.label].Q.val_SI))
        for c in components
        if c.kind == "closed_heater"
    )

    if layout.reheat_in is not None and layout.reheat_out is not None:
        if states[layout.reheat_out].h_J_per_kg <= states[layout.reheat_in].h_J_per_kg:
            assert inputs.reheat is not None
            raise ValueError(
                f"La temperatura de recalentamiento ({_degC(inputs.reheat.T_K)}) no supera la de "
                f"salida de la turbina de alta ({_degC(states[layout.reheat_in].T_K)}): el "
                "«recalentador» tendría que enfriar el vapor. Subí la temperatura o la presión de "
                "recalentamiento."
            )
    result = RankineResult(
        inputs=inputs,
        states=states,
        pump_out_s=pumps_out_s[0],
        turbine_out_s=turbine_out_s,
        m_dot_kg_s=1.0,
        tespy_balances=balances,
        layout=layout,
        components=components,
        pumps_out_s=pumps_out_s,
        tespy_heaters=heater_heat,
    )
    w_net = result.w_net_J_per_kg
    if w_net <= 0.0:
        raise ValueError(
            "Con esos datos el ciclo no entrega trabajo neto (la bomba consume más de lo que "
            "da la turbina). Revisá las presiones y los rendimientos."
        )
    m_dot = inputs.m_dot_kg_s if inputs.m_dot_kg_s is not None else inputs.W_net_W / w_net  # type: ignore[operator]
    return replace(result, m_dot_kg_s=float(m_dot))


# ---------------------------------------------------------------------
# Presentación: estados, avisos, ejemplos
# ---------------------------------------------------------------------


def rankine_labeled_states(result: RankineResult) -> list[tuple[str, FluidState]]:
    """``(etiqueta, estado)`` en el orden de la numeración, para la tabla y el diagrama."""
    assert result.layout is not None
    return list(zip(result.layout.labeled(), result.states, strict=True))


def rankine_isentropic_states(result: RankineResult) -> list[tuple[str, FluidState]]:
    """Estados isoentrópicos de referencia del ciclo real: la salida de cada bomba
    (2s…) y de cada tramo de turbina, medida desde la entrada de su turbina (4s…)."""
    pumps = [
        (f"{c.port('out').state + 1}s", state)
        for c, state in zip(result.of_kind("pump"), result.pumps_out_s, strict=True)
    ]
    turbines = [
        (f"{c.port('out').state + 1}s", state)
        for c, state in zip(result.of_kind("turbine"), result.turbine_out_s, strict=True)
    ]
    return pumps + turbines


def rankine_notes(result: RankineResult) -> list[str]:
    """Observaciones didácticas sobre el ciclo calculado (markdown)."""
    notes: list[str] = []
    x = result.x_turbine_out
    if x is not None and x < X_TURBINE_MIN:
        notes.append(
            f"El vapor sale de la turbina con título x = {x:.3f}: más de 10–12 % de humedad "
            "erosiona los álabes de las últimas etapas (Cengel §10-4). Recalentar o "
            "sobrecalentar más lo evita."
        )
    if x is None:
        notes.append(
            "El vapor sale **sobrecalentado** de la turbina: el condensador primero lo enfría "
            "hasta T_sat(p) y recién después lo condensa."
        )
    if result.inputs.p_boiler_Pa >= fluid_limits(FLUID).P_crit_Pa:
        notes.append(
            "Ciclo **supercrítico**: en la caldera el agua pasa de líquido a vapor sin hervir "
            "(por encima de la presión crítica no hay campana)."
        )
    return notes


#: Ejemplos precargados (Çengel & Boles, *Termodinámica*, cap. 10).
RANKINE_EXAMPLES: dict[str, RankineInputs] = {
    "Cengel 10-1 — ideal simple (3 MPa, 350 °C; 75 kPa)": RankineInputs(3.0e6, 623.15, 75.0e3),
    "Cengel 10-3 a) — ideal (3 MPa, 350 °C; 10 kPa)": RankineInputs(3.0e6, 623.15, 10.0e3),
    "Cengel 10-3 b) — más sobrecalentamiento (3 MPa, 600 °C; 10 kPa)": RankineInputs(
        3.0e6, 873.15, 10.0e3
    ),
    "Cengel 10-3 c) — más presión (15 MPa, 600 °C; 10 kPa)": RankineInputs(15.0e6, 873.15, 10.0e3),
    "Cengel 10-4 — con recalentamiento (15 MPa, 600 °C; 4 MPa, 600 °C; 10 kPa)": RankineInputs(
        15.0e6, 873.15, 10.0e3, reheat=Reheat(4.0e6, 873.15)
    ),
    "Basado en Cengel 10-2, sin pérdidas de carga (15 MPa, 600 °C; 10 kPa; "
    "η_T = 0,87, η_B = 0,85)": RankineInputs(
        15.0e6, 873.15, 10.0e3, eta_turbine=0.87, eta_pump=0.85
    ),
    "Cengel 10-5 — regenerativo, un calentador abierto (15 MPa, 600 °C; 1,2 MPa; "
    "10 kPa)": RankineInputs(15.0e6, 873.15, 10.0e3, heaters=(FeedwaterHeater(1.2e6),)),
    "Cengel 10-6 — recalentamiento y regeneración: cerrado a 4 MPa y abierto a 0,5 MPa": (
        RankineInputs(
            15.0e6,
            873.15,
            10.0e3,
            reheat=Reheat(4.0e6, 873.15),
            heaters=(FeedwaterHeater(0.5e6), FeedwaterHeater(4.0e6, "closed")),
            drain_forward=True,
        )
    ),
    "Basado en Cengel 10-5 — un calentador cerrado con drenaje al condensador": RankineInputs(
        15.0e6, 873.15, 10.0e3, heaters=(FeedwaterHeater(1.2e6, "closed"),)
    ),
}

#: Aclaraciones de algunos ejemplos (se muestran junto al resultado).
RANKINE_EXAMPLE_NOTES: dict[str, str] = {
    "Cengel 10-6 — recalentamiento y regeneración: cerrado a 4 MPa y abierto a 0,5 MPa": (
        "El libro informa y = 0,1766 (cerrado), z = 0,1306 (abierto) y η = 49,2 %. Esos "
        "números salen de calcular la bomba II solo hasta 4 MPa (h₄ = 643,9 kJ/kg); como el "
        "agua de alimentación va a 15 MPa, acá h₄ = 655,9 kJ/kg y quedan y ≈ 0,173, "
        "z ≈ 0,131 y η ≈ 49,0 %."
    ),
}


# ---------------------------------------------------------------------
# Agua de enfriamiento del condensador
# ---------------------------------------------------------------------

#: c_p del agua líquida para la cuenta aproximada (Cengel, tabla A-3).
CP_WATER_J_PER_KG_K = 4180.0


@dataclass(frozen=True)
class CoolingWaterResult:
    """Agua de enfriamiento del condensador (balance de energía, vademecum §3.3), en SI.

    ``m_dot_kg_s`` sale de Q̇_C / (h_s − h_e) con la ecuación de estado;
    ``m_dot_cp_kg_s``, de la aproximación Q̇_C / (c_p·ΔT).
    """

    T_in_K: float
    T_out_K: float
    Q_W: float
    dh_J_per_kg: float
    m_dot_kg_s: float
    m_dot_cp_kg_s: float


def condenser_cooling_water(
    result: RankineResult, T_in_K: float, T_out_K: float, p_Pa: float = 101_325.0
) -> CoolingWaterResult:
    """Caudal del agua de enfriamiento que se lleva el calor del condensador.

    Raises
    ------
    ValueError
        Si el agua no se calienta, entra por debajo de 0 °C o saldría más
        caliente que el vapor que condensa (T_sat a la presión del
        condensador).
    """
    if not (math.isfinite(T_in_K) and math.isfinite(T_out_K)):
        raise ValueError("Las temperaturas del agua de enfriamiento tienen que ser números.")
    if T_in_K <= 273.16:
        raise ValueError("El agua de enfriamiento tiene que entrar líquida, por encima de 0 °C.")
    if T_out_K <= T_in_K:
        raise ValueError(
            f"El agua de enfriamiento se calienta en el condensador: la temperatura de salida "
            f"({_degC(T_out_K)}) tiene que ser mayor que la de entrada ({_degC(T_in_K)})."
        )
    T_cond = result.T_low_K
    if T_out_K >= T_cond:
        raise ValueError(
            "El agua de enfriamiento no puede salir más caliente que el vapor que condensa: a "
            f"{_bar(result.states[0].P_Pa)} el vapor condensa a T_sat = {_degC(T_cond)}, y el "
            f"agua saldría a {_degC(T_out_K)}. Bajá la temperatura de salida del agua o subí la "
            "presión del condensador."
        )
    h_in = fluid_state_from_pair(FLUID, "TP", t=T_in_K, p=p_Pa).h_J_per_kg
    h_out = fluid_state_from_pair(FLUID, "TP", t=T_out_K, p=p_Pa).h_J_per_kg
    dh = h_out - h_in
    Q = result.Q_out_W
    return CoolingWaterResult(
        T_in_K=T_in_K,
        T_out_K=T_out_K,
        Q_W=Q,
        dh_J_per_kg=dh,
        m_dot_kg_s=Q / dh,
        m_dot_cp_kg_s=Q / (CP_WATER_J_PER_KG_K * (T_out_K - T_in_K)),
    )


# ---------------------------------------------------------------------
# Barridos: ¿cómo aumentar el rendimiento? (Cengel §10-4)
# ---------------------------------------------------------------------

SweepParameter = Literal["p_boiler", "T_turbine_in", "p_condenser"]


@dataclass(frozen=True)
class SweepPoint:
    """Un punto del barrido: valor del parámetro (SI) y resultado del ciclo."""

    value_si: float
    eta_th: float
    w_net_J_per_kg: float
    x_turbine_out: float | None
    fractions: tuple[float, ...] = ()


_SWEEP_FIELDS: dict[str, str] = {
    "p_boiler": "p_boiler_Pa",
    "T_turbine_in": "T_turbine_in_K",
    "p_condenser": "p_condenser_Pa",
}


def default_sweep_values(
    inputs: RankineInputs, parameter: SweepParameter, n: int = 7
) -> list[float]:
    """Valores razonables del parámetro alrededor del ciclo dado (SI)."""
    lim = fluid_limits(FLUID)
    if parameter == "p_boiler":
        low = max(inputs.p_condenser_Pa * 20.0, 5.0e5)
        if inputs.reheat is not None:
            low = max(low, inputs.reheat.p_Pa * 1.25)
        if inputs.heaters:  # la caldera, por encima del calentador de mayor presión
            low = max(low, inputs.heaters[-1].p_Pa * 1.25)
        high = 0.95 * lim.P_crit_Pa if inputs.T_turbine_in_K is None else 3.0e7
        if inputs.T_turbine_in_K is not None and inputs.T_turbine_in_K <= lim.T_crit_K:
            high = 0.95 * lim.P_crit_Pa
        return [float(v) for v in np.geomspace(low, high, n)]
    if parameter == "T_turbine_in":
        p = inputs.p_boiler_Pa
        T_floor = lim.T_crit_K if p >= lim.P_crit_Pa else _T_sat(p)
        return [float(v) for v in np.linspace(T_floor + 20.0, 923.15, n)]
    high = 1.0e5
    if inputs.reheat is not None:
        high = min(high, 0.5 * inputs.reheat.p_Pa)
    if inputs.heaters:  # el condensador, por debajo del calentador de menor presión
        high = min(high, 0.5 * inputs.heaters[0].p_Pa)
    return [float(v) for v in np.geomspace(4.0e3, high, n)]


def rankine_sweep(
    inputs: RankineInputs, parameter: SweepParameter, values: Sequence[float]
) -> list[SweepPoint]:
    """Resuelve el ciclo para cada valor del parámetro (como la sección 8 del
    tutorial de TESPy). Los valores sin sentido físico se omiten."""
    points: list[SweepPoint] = []
    for value in values:
        trial = replace(inputs, **{_SWEEP_FIELDS[parameter]: float(value)})
        try:
            result = solve_rankine(trial)
        except ValueError:
            continue
        points.append(_sweep_point(float(value), result))
    return points


def _sweep_point(value_si: float, result: RankineResult) -> SweepPoint:
    return SweepPoint(
        value_si,
        result.eta_th,
        result.w_net_J_per_kg,
        result.x_turbine_out,
        result.extraction_fractions,
    )


def default_extraction_values(inputs: RankineInputs, heater: int, n: int = 9) -> list[float]:
    """Presiones de extracción para el barrido del calentador ``heater`` (SI).

    Quedan entre las de los calentadores vecinos (o el condensador y la
    caldera) y no cruzan la presión de recalentamiento: de un lado la
    extracción sale de la turbina de alta y del otro, del vapor recalentado,
    y la curva saltaría.
    """
    heaters = inputs.heaters
    p_k = heaters[heater].p_Pa
    low = heaters[heater - 1].p_Pa * 1.15 if heater > 0 else inputs.p_condenser_Pa * 2.0
    high = (
        heaters[heater + 1].p_Pa / 1.15 if heater + 1 < len(heaters) else inputs.p_boiler_Pa / 1.15
    )
    high = min(high, 0.95 * fluid_limits(FLUID).P_crit_Pa)
    if inputs.reheat is not None:
        p_rh = inputs.reheat.p_Pa
        if p_k >= p_rh * (1.0 - 1.0e-4):
            low = max(low, p_rh)
        else:
            high = min(high, 0.98 * p_rh)
    if high <= low:
        return [p_k]
    return [float(v) for v in np.geomspace(low, high, n)]


def extraction_pressure_sweep(
    inputs: RankineInputs, heater: int, values: Sequence[float]
) -> list[SweepPoint]:
    """η, w_neto, título y fracciones para cada presión de extracción del
    calentador ``heater`` (Cengel §10-6: hay una presión óptima). Los valores
    sin sentido físico se omiten."""
    points: list[SweepPoint] = []
    for value in values:
        heaters = tuple(
            replace(h, p_Pa=float(value)) if i == heater else h
            for i, h in enumerate(inputs.heaters)
        )
        try:
            result = solve_rankine(replace(inputs, heaters=heaters))
        except ValueError:
            continue
        points.append(_sweep_point(float(value), result))
    return points


# ---------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------


def _value(value_si: float, kind: QuantityKind, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def rankine_to_dict(
    result: RankineResult, system: UnitSystem, cooling: CoolingWaterResult | None = None
) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``).

    Con calentadores suma los datos de cada uno y las fracciones de
    extracción; con ``cooling``, el agua de enfriamiento del condensador.
    """
    inputs = result.inputs
    if inputs.heaters:
        cycle = "Rankine regenerativo" + (" con recalentamiento" if result.has_reheat else "")
    else:
        cycle = "Rankine con recalentamiento" if result.has_reheat else "Rankine simple"
    data: dict[str, Any] = {
        "ciclo": cycle,
        "fluido": FLUID,
        "sistema_de_unidades": system,
        "datos": {
            "p_caldera": _value(inputs.p_boiler_Pa, "pressure", system),
            "entrada_turbina": (
                "vapor saturado (x = 1)"
                if inputs.T_turbine_in_K is None
                else _value(inputs.T_turbine_in_K, "temperature", system)
            ),
            "p_condensador": _value(inputs.p_condenser_Pa, "pressure", system),
            "eta_turbina": inputs.eta_turbine,
            "eta_bomba": inputs.eta_pump,
            "recalentamiento": (
                None
                if inputs.reheat is None
                else {
                    "p": _value(inputs.reheat.p_Pa, "pressure", system),
                    "T": _value(inputs.reheat.T_K, "temperature", system),
                }
            ),
        },
        "resultados": {
            "eta_termica": result.eta_th,
            "eta_carnot": result.eta_carnot,
            "T_media_aporte_de_calor": _value(result.T_mean_in_K, "temperature", system),
            "w_turbina": _value(result.w_turbine_J_per_kg, _EH, system),
            "w_bomba": _value(result.w_pump_J_per_kg, _EH, system),
            "w_neto": _value(result.w_net_J_per_kg, _EH, system),
            "q_entregado": _value(result.q_in_J_per_kg, _EH, system),
            "q_cedido": _value(result.q_out_J_per_kg, _EH, system),
            "relacion_de_trabajo_de_retroceso": result.back_work_ratio,
            "titulo_salida_turbina": result.x_turbine_out,
            "caudal": _value(result.m_dot_kg_s, "mass_flow", system),
            "potencia_neta": _value(result.W_net_W, "power", system),
            "calor_entregado": _value(result.Q_in_W, "power", system),
            "calor_cedido": _value(result.Q_out_W, "power", system),
        },
        "estados": states_table(rankine_labeled_states(result), system),
        "fuente": "TESPy 0.11 (red de componentes) + CoolProp (IAPWS-95)",
    }
    if inputs.heaters:
        assert result.layout is not None
        names = result.layout.heater_names
        data["datos"]["calentadores"] = [
            {
                "nombre": name,
                "tipo": "abierto" if h.kind == "open" else "cerrado",
                "p_extraccion": _value(h.p_Pa, "pressure", system),
                "TTD": _value(h.ttd_K, "temperature_difference", system),
            }
            for name, h in zip(names, inputs.heaters, strict=True)
        ]
        data["datos"]["drenaje_bombeado_hacia_adelante"] = result.layout.drain_forward
        data["resultados"]["fracciones_de_extraccion"] = [
            {"calentador": name, "y": y}
            for name, y in zip(names, result.extraction_fractions, strict=True)
        ]
    if cooling is not None:
        data["resultados"]["agua_de_enfriamiento"] = {
            "T_entrada": _value(cooling.T_in_K, "temperature", system),
            "T_salida": _value(cooling.T_out_K, "temperature", system),
            "caudal": _value(cooling.m_dot_kg_s, "mass_flow", system),
            "caudal_con_cp": _value(cooling.m_dot_cp_kg_s, "mass_flow", system),
        }
    return data


def rankine_segments(result: RankineResult) -> list[tuple[int, int]]:
    """Tramos (entrada, salida) de cada proceso, como índices de ``states``,
    para dibujar el ciclo: bombas, caldera, turbinas, condensador y, con
    regeneración, extracciones, drenajes, válvulas y mezclas."""
    pairs = {
        "fw_in": ("fw_out", "out"),
        "in": ("out",),
        "bleed": ("drain_out", "out"),
        "drain_in": ("drain_out", "out"),
    }
    segments: list[tuple[int, int]] = []
    for comp in result.components:
        outs = {p.role: p.state for p in comp.outlets}
        for port in comp.inlets:
            target = next(outs[r] for r in pairs[port.role] if r in outs)
            if (port.state, target) not in segments:
                segments.append((port.state, target))
    return segments
