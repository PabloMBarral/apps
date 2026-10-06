"""Ciclo de Rankine con TESPy — Fases 3.1a, 3.1b y 3.1c.

Ciclo de vapor simple (bomba, caldera, turbina, condensador), ideal o real
(rendimientos isoentrópicos de turbina y bomba), con recalentamiento en una
etapa y con regeneración (hasta tres calentadores de agua de alimentación,
abiertos o cerrados; Cengel §10-6). Se resuelve con una red de TESPy armada
como los ejemplos oficiales: el tutorial ``tutorial/basics/rankine.py`` de
TESPy 0.11 (``CycleCloser`` + ``SimpleHeatExchanger`` + ``Turbine`` +
``Pump``), el de optimización de una central (calentadores cerrados como
``Condenser``, drenaje bombeado hacia adelante a un ``Merge``) y el modelo
SEGS de sus tests (drenajes en cascada con ``Valve`` + ``Merge``); el
calentador abierto es un ``Merge`` con líquido saturado a la salida. Sin
pérdidas, el condensador entrega líquido saturado (x = 0), como en Cengel.

Fase 3.1c:

- **Ciclo real** (Cengel §10-5 y ejemplo 10-2): caídas de presión en la
  caldera, el recalentador, el condensador y los calentadores cerrados,
  subenfriamiento del condensado (``td_bubble``) y cañerías con pérdida de
  carga y de calor (``Pipe``).
- **Calentadores cerrados reales**: subenfriador de drenaje
  (``Condenser`` con ``subcooling`` y ``ttd_l`` = DCA, como el ejemplo de
  subenfriamiento de la documentación de TESPy), desrecalentador
  (``Desuperheater`` en serie) y drenajes bombeados hacia adelante desde
  cualquier cerrado.
- **ORC**: otros fluidos de trabajo y recuperador (``HeatExchanger`` con
  ``eff_max``, la efectividad ε = q/q_máx).

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
    Desuperheater,
    HeatExchanger,
    Merge,
    Pump,
    SimpleHeatExchanger,
    Splitter,
    Turbine,
    Valve,
)
from tespy.components import Pipe as TespyPipe
from tespy.connections import Connection, Ref

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
from core.fluids import (
    FLUID_NAMES_ES,
    FluidState,
    _del,
    _el,
    fluid_limits,
    fluid_state_from_pair,
    saturation_at_pressure,
    saturation_at_temperature,
)
from core.state_report import states_table, textbook_reference_note
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

# ``rankine_steps`` y ``cooling_water_step`` viven en
# :mod:`core.cycles.rankine_procedure`; se reexportan para que el import
# público de la Fase 3.1a (``from core.cycles.rankine import rankine_steps``)
# siga funcionando.
__all__ = [
    "FLUID",
    "RANKINE_EXAMPLES",
    "RANKINE_EXAMPLE_NOTES",
    "RANKINE_FLUIDS",
    "X_TURBINE_MIN",
    "CoolingWaterResult",
    "CycleComponent",
    "FeedwaterHeater",
    "Losses",
    "PipeLoss",
    "RankineInputs",
    "RankineResult",
    "Recuperator",
    "Reheat",
    "SweepParameter",
    "SweepPoint",
    "condenser_cooling_water",
    "cooling_water_step",
    "default_extraction_values",
    "default_heater_pressures",
    "default_sweep_values",
    "extraction_pressure_sweep",
    "FluidBehaviorInfo",
    "fluid_behavior",
    "rankine_isentropic_states",
    "rankine_labeled_states",
    "rankine_notes",
    "rankine_segments",
    "rankine_steps",
    "rankine_sweep",
    "rankine_to_dict",
    "solve_rankine",
    "suggested_rankine_inputs",
    "validate_rankine_inputs",
]

#: Fluido de trabajo por defecto (nombre de CoolProp): el ciclo de vapor de agua.
FLUID = "Water"

#: Fluidos de trabajo del ciclo: el agua y los del ciclo de Rankine orgánico
#: (ORC; Quoilin et al., 2013, *Renewable and Sustainable Energy Reviews* 22,
#: 168–186, doi:10.1016/j.rser.2013.01.028).
RANKINE_FLUIDS: tuple[str, ...] = (
    "Water",
    "R245fa",
    "R1233zd(E)",
    "Isopentane",
    "Toluene",
    "R134a",
    "R1234yf",
    "Ammonia",
)

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
class PipeLoss:
    """Cañería con pérdidas (Cengel §10-5): caída de presión ``dp_Pa`` por la
    fricción y caída de temperatura ``dT_K`` por la pérdida de calor al
    ambiente. Con ``dT_K = 0`` la cañería es adiabática (h constante)."""

    dp_Pa: float = 0.0
    dT_K: float = 0.0


@dataclass(frozen=True)
class Losses:
    """Pérdidas del ciclo real (Cengel §10-5), en SI.

    Los datos del ciclo siguen siendo los de la turbina: la presión y la
    temperatura de entrada, la presión de escape (la del condensador, a la
    entrada) y la de recalentamiento (a la salida de la turbina de alta). La
    bomba tiene que compensar todas las caídas de presión hasta la turbina.

    - ``dp_boiler_Pa``, ``dp_reheater_Pa``, ``dp_condenser_Pa``: caídas de
      presión por fricción en cada intercambiador.
    - ``subcooling_K``: el condensado sale ΔT por debajo de T_sat (se
      subenfría para que la bomba no cavite).
    - ``feed_pipe``: cañería de la línea de agua a la caldera; ``steam_pipe``:
      de la caldera a la turbina (Cengel 10-2). ``None`` = sin cañería.
    """

    dp_boiler_Pa: float = 0.0
    dp_reheater_Pa: float = 0.0
    dp_condenser_Pa: float = 0.0
    subcooling_K: float = 0.0
    feed_pipe: PipeLoss | None = None
    steam_pipe: PipeLoss | None = None

    @property
    def any(self) -> bool:
        """Hay alguna pérdida (si no, es el ciclo de Cengel §10-2 a §10-6)."""
        return (
            any(
                v != 0.0
                for v in (
                    self.dp_boiler_Pa,
                    self.dp_reheater_Pa,
                    self.dp_condenser_Pa,
                    self.subcooling_K,
                )
            )
            or self.feed_pipe is not None
            or self.steam_pipe is not None
        )


@dataclass(frozen=True)
class Recuperator:
    """Recuperador del ORC: el escape de la turbina (vapor sobrecalentado)
    precalienta el líquido que sale de la bomba. ``effectiveness`` es la
    efectividad ε = q / q_máx, con q_máx el calor que daría el lado que puede
    entregar menos (es la efectividad de un intercambiador; Cengel §9-9 usa la
    misma idea para el regenerador del ciclo Brayton)."""

    effectiveness: float = 0.8


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
    el drenaje del de mayor presión (que tiene que ser cerrado); cada cerrado
    puede pedirlo también con :attr:`FeedwaterHeater.drain_forward`. Los
    demás drenajes van en cascada hacia atrás.

    ``losses`` son las pérdidas del ciclo real (Cengel §10-5), ``fluid`` el
    fluido de trabajo (uno de :data:`RANKINE_FLUIDS`; con uno orgánico es un
    ORC) y ``recuperator`` el recuperador del ORC (sin calentadores).
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
    losses: Losses = Losses()
    fluid: str = FLUID
    recuperator: Recuperator | None = None

    def __post_init__(self) -> None:
        # Se acepta una lista (la página la arma así); se guarda como tupla para
        # que los datos sigan siendo inmutables y sirvan de clave de caché.
        object.__setattr__(self, "heaters", tuple(self.heaters))

    @property
    def is_ideal(self) -> bool:
        """Turbina y bomba isoentrópicas (el ciclo ideal de Rankine)."""
        return self.eta_turbine == 1.0 and self.eta_pump == 1.0

    @property
    def boiler_name(self) -> str:
        """Dónde entra el calor: la caldera del ciclo de vapor o el evaporador del ORC."""
        return "caldera" if self.fluid == FLUID else "evaporador"

    @property
    def layout(self) -> PlantLayout:
        """Plano de la planta con la numeración de estados (sin resolver)."""
        return plant_layout(
            heaters=self.heaters,
            reheat_p_Pa=None if self.reheat is None else self.reheat.p_Pa,
            drain_forward=self.drain_forward,
            feed_pipe=self.losses.feed_pipe is not None,
            steam_pipe=self.losses.steam_pipe is not None,
            recuperator=self.recuperator is not None,
            boiler=self.boiler_name,
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
    kilogramo de vapor que pasa por la caldera, como en Cengel. El
    recuperador del ORC también figura en ``tespy_heaters`` (es calor
    interno).
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

    @property
    def has_losses(self) -> bool:
        """Ciclo real con pérdidas de carga o de calor (Cengel §10-5)."""
        return self.inputs.losses.any

    @property
    def has_recuperator(self) -> bool:
        return self.inputs.recuperator is not None

    @property
    def fluid(self) -> str:
        return self.inputs.fluid

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
    def q_loss_J_per_kg(self) -> float:
        """Calor que pierden las cañerías al ambiente (Cengel §10-5); 0 sin cañerías.
        Con él, el primer principio queda w_neto = q_H − q_C − q_pérd."""
        return -self._flow_through("pipe")

    @property
    def q_recuperator_J_per_kg(self) -> float:
        """Calor que el recuperador pasa del escape de la turbina al líquido (interno)."""
        return sum(
            c.port("fw_out").fraction * self.states[c.port("fw_out").state].h_J_per_kg
            - c.port("fw_in").fraction * self.states[c.port("fw_in").state].h_J_per_kg
            for c in self.of_kind("recuperator")
        )

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
        """Temperatura de condensación T_sat(p_cond) (sumidero del ciclo).

        Sin pérdidas es la del líquido saturado que sale del condensador; con
        subenfriamiento o caída de presión, la de saturación a la presión de
        escape de la turbina (donde empieza a condensar)."""
        losses = self.inputs.losses
        if losses.subcooling_K == 0.0 and losses.dp_condenser_Pa == 0.0:
            return self.states[0].T_K
        return _T_sat(self.inputs.p_condenser_Pa, self.inputs.fluid)

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

    @property
    def Q_loss_W(self) -> float:
        return self.m_dot_kg_s * self.q_loss_J_per_kg


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


def _T_sat(p_Pa: float, fluid: str = FLUID) -> float:
    return saturation_at_pressure(fluid, p_Pa).T_sat_K


def _would_be_liquid(fluid: str, where: str) -> str:
    """«el agua de entrada a la turbina sería líquida» (con el género del fluido)."""
    if fluid == FLUID:
        return f"el agua de {where} sería líquida"
    return f"{_el(fluid)} de {where} sería líquido"


def _check_superheated(
    T_K: float, p_Pa: float, where: str, at_saturation: str, fluid: str = FLUID
) -> None:
    """La entrada a una turbina tiene que ser vapor (sobrecalentado o supercrítico).

    ``at_saturation`` es la sugerencia para el alumno si T es justo la de saturación.
    """
    lim = fluid_limits(fluid)
    if T_K > lim.T_max_K:
        raise ValueError(
            f"La temperatura de {where} ({_degC(T_K)}) supera el máximo de validez de la "
            f"ecuación de estado {_del(fluid)} ({_degC(lim.T_max_K)})."
        )
    if p_Pa >= lim.P_crit_Pa:
        if T_K <= lim.T_crit_K:
            raise ValueError(
                f"Con presión supercrítica ({_bar(p_Pa)}), la temperatura de {where} tiene que "
                f"superar la crítica ({_degC(lim.T_crit_K)}): si no, a la turbina entraría líquido."
            )
        return
    T_sat = _T_sat(p_Pa, fluid)
    if T_K < T_sat - _T_SAT_TOL:
        raise ValueError(
            f"Con T = {_degC(T_K)} y p = {_bar(p_Pa)}, {_would_be_liquid(fluid, where)}: a esa "
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
    fluid = inputs.fluid
    if fluid not in RANKINE_FLUIDS:
        raise ValueError(
            f"El fluido «{fluid}» no está disponible para el ciclo de Rankine. Usá uno de: "
            + ", ".join(FLUID_NAMES_ES.get(f, f) for f in RANKINE_FLUIDS)
            + "."
        )
    lim = fluid_limits(fluid)
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
            f"ecuación de estado {_del(fluid)} ({_bar(lim.P_max_Pa)})."
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
            "Si entra vapor saturado seco, elegí «Vapor saturado seco (x = 1)».",
            fluid,
        )

    _validate_losses(inputs)

    if inputs.reheat is not None:
        rh = inputs.reheat
        if not (math.isfinite(rh.p_Pa) and p_lo < rh.p_Pa < p_hi):
            raise ValueError(
                f"La presión de recalentamiento ({_bar(rh.p_Pa)}) tiene que estar entre la del "
                f"condensador ({_bar(p_lo)}) y la de la caldera ({_bar(p_hi)}): el vapor se "
                "expande en la turbina de alta hasta esa presión, se recalienta y sigue en la de "
                "baja."
            )
        p_lp = rh.p_Pa - inputs.losses.dp_reheater_Pa
        if p_lp <= p_lo:
            raise ValueError(
                f"Con la caída de presión del recalentador "
                f"({_bar(inputs.losses.dp_reheater_Pa)}), la turbina de baja arrancaría a "
                f"{_bar(p_lp)}, sin salto hasta el condensador ({_bar(p_lo)}). Achicá la caída "
                "de presión o subí la presión de recalentamiento."
            )
        _check_superheated(
            rh.T_K,
            p_lp,
            "entrada a la turbina de baja",
            "El recalentador tiene que dejar el vapor sobrecalentado: subí su temperatura.",
            fluid,
        )
    elif inputs.losses.dp_reheater_Pa != 0.0:
        raise ValueError(
            "Hay caída de presión en el recalentador, pero el ciclo no recalienta: dejala en 0."
        )

    given = [v for v in (inputs.m_dot_kg_s, inputs.W_net_W) if v is not None]
    if len(given) != 1:
        raise ValueError("Indicá el caudal másico o la potencia neta (uno de los dos).")
    if not (math.isfinite(given[0]) and given[0] > 0.0):
        raise ValueError("El caudal másico o la potencia neta tiene que ser positivo.")

    _validate_heater_data(inputs)
    _validate_line(inputs)
    _validate_heater_operation(inputs)
    _validate_recuperator(inputs)


def _validate_losses(inputs: RankineInputs) -> None:
    """Pérdidas del ciclo real (Cengel §10-5): signos, punto triple y caldera."""
    losses = inputs.losses
    if not losses.any:
        return
    fluid = inputs.fluid
    lim = fluid_limits(fluid)
    drops = {
        "la caldera": losses.dp_boiler_Pa,
        "el recalentador": losses.dp_reheater_Pa,
        "el condensador": losses.dp_condenser_Pa,
    }
    pipes = {
        "cañería de alimentación": losses.feed_pipe,
        "cañería de vapor": losses.steam_pipe,
    }
    for name, pipe in pipes.items():
        if pipe is not None:
            drops[f"la {name}"] = pipe.dp_Pa
    for where, dp in drops.items():
        if not (math.isfinite(dp) and dp >= 0.0):
            raise ValueError(
                f"La caída de presión en {where} no puede ser negativa: la fricción siempre baja "
                "la presión del fluido (0 = sin pérdida de carga)."
            )
    if not (math.isfinite(losses.subcooling_K) and losses.subcooling_K >= 0.0):
        raise ValueError(
            "El subenfriamiento del condensado no puede ser negativo: como mucho, el condensado "
            "sale como líquido saturado (subenfriamiento 0)."
        )
    for name, pipe in pipes.items():
        if pipe is not None and not (math.isfinite(pipe.dT_K) and pipe.dT_K >= 0.0):
            raise ValueError(
                f"La caída de temperatura en la {name} no puede ser negativa: la cañería pierde "
                "calor hacia el ambiente (0 = cañería aislada, sin pérdida de calor)."
            )

    p_out = inputs.p_condenser_Pa - losses.dp_condenser_Pa
    if p_out <= lim.P_triple_Pa:
        raise ValueError(
            f"Con una caída de presión de {_bar(losses.dp_condenser_Pa)} en el condensador, el "
            f"condensado saldría a {_bar(p_out)}, por debajo del punto triple "
            f"({_bar(lim.P_triple_Pa)}). Achicá la caída de presión."
        )
    T_out = _T_sat(p_out, fluid) - losses.subcooling_K
    if T_out <= lim.T_triple_K + 0.5:
        raise ValueError(
            f"Con {losses.subcooling_K:g} K de subenfriamiento el condensado quedaría a "
            f"{_degC(T_out)}, al nivel del punto triple ({_degC(lim.T_triple_K)}): se "
            "congelaría. Achicá el subenfriamiento."
        )

    steam = losses.steam_pipe
    if steam is not None:
        T_turbine = inputs.T_turbine_in_K
        if T_turbine is None:
            T_turbine = _T_sat(inputs.p_boiler_Pa, fluid)
        T_boiler = T_turbine + steam.dT_K
        p_boiler = inputs.p_boiler_Pa + steam.dp_Pa
        if T_boiler > lim.T_max_K:
            raise ValueError(
                f"Para que el vapor llegue a la turbina a {_degC(T_turbine)}, la "
                f"{inputs.boiler_name} lo tendría que entregar a {_degC(T_boiler)}, por encima "
                f"del máximo de validez de la ecuación de estado {_del(fluid)} "
                f"({_degC(lim.T_max_K)}). Achicá la pérdida de calor de la cañería de vapor."
            )
        if steam.dT_K > 0.0 and p_boiler < lim.P_crit_Pa:
            T_sat = _T_sat(p_boiler, fluid)
            if T_boiler <= T_sat + _T_SAT_TOL:
                raise ValueError(
                    f"La cañería de vapor enfría el vapor {steam.dT_K:g} K, así que la "
                    f"{inputs.boiler_name} lo tendría que entregar a {_degC(T_boiler)}; a "
                    f"{_bar(p_boiler)} eso no es vapor (T_sat = {_degC(T_sat)}). Con vapor "
                    "saturado o poco sobrecalentado a la entrada de la turbina, subí la "
                    "temperatura de entrada o achicá la pérdida de calor."
                )


def _validate_line(inputs: RankineInputs) -> None:
    """Línea de agua con pérdidas: presión de la bomba y cañería de alimentación."""
    if not inputs.losses.any and not any(h.dp_Pa for h in inputs.heaters):
        return
    fluid = inputs.fluid
    lim = fluid_limits(fluid)
    p_pump = max(_line_pressures(inputs))
    if p_pump > lim.P_max_Pa:
        raise ValueError(
            f"Con las caídas de presión, la bomba tendría que llegar a {_bar(p_pump)}, por encima "
            f"del máximo de validez de la ecuación de estado {_del(fluid)} "
            f"({_bar(lim.P_max_Pa)}). Achicá las pérdidas de carga."
        )
    feed = inputs.losses.feed_pipe
    if feed is not None and feed.dT_K > 0.0:
        T_in = _feedwater_line(inputs)[1]
        if T_in - feed.dT_K <= lim.T_triple_K + 0.5:
            raise ValueError(
                f"La cañería de alimentación enfriaría el líquido {feed.dT_K:g} K, de "
                f"{_degC(T_in)} a menos que el punto triple ({_degC(lim.T_triple_K)}). Achicá "
                "la pérdida de calor."
            )


def _condensate_state(inputs: RankineInputs) -> FluidState:
    """Salida del condensador antes de resolver: líquido saturado o subenfriado."""
    losses = inputs.losses
    p = inputs.p_condenser_Pa - losses.dp_condenser_Pa
    if losses.subcooling_K > 0.0:
        T = _T_sat(p, inputs.fluid) - losses.subcooling_K
        return fluid_state_from_pair(inputs.fluid, "TP", t=T, p=p)
    return fluid_state_from_pair(inputs.fluid, "PX", p=p, x=0.0)


def _pump_outlet_h(inlet: FluidState, p_to_Pa: float, eta: float) -> float:
    """h a la salida de una bomba (vademecum §10.4): h_e + (h_s − h_e)/η_B."""
    h_s = _isentropic_state(p_to_Pa, inlet.s_J_per_kg_K, inlet.fluid).h_J_per_kg
    return inlet.h_J_per_kg + (h_s - inlet.h_J_per_kg) / eta


def default_heater_pressures(
    p_boiler_Pa: float, p_condenser_Pa: float, n: int, fluid: str = FLUID
) -> list[float]:
    """Presiones de extracción para ``n`` calentadores (SI), de menor a mayor.

    Reparten por igual el salto de T_sat entre el condensador y la caldera
    (tope: 0,9·p_crit), una regla práctica cercana al óptimo (Cengel §10-6):
    con un calentador en el ejemplo 10-5 da 13,7 bar, y el máximo de η está
    entre 12 y 20 bar.
    """
    lim = fluid_limits(fluid)
    p_top = min(p_boiler_Pa, 0.9 * lim.P_crit_Pa)
    T_low, T_high = _T_sat(p_condenser_Pa, fluid), _T_sat(p_top, fluid)
    pressures = []
    for k in range(1, n + 1):
        T = T_low + k * (T_high - T_low) / (n + 1)
        pressures.append(float(PropsSI("P", "T", T, "Q", 0, fluid)))
    return pressures


def _stages(inputs: RankineInputs) -> list[list[int]]:
    """Calentadores cerrados de cada tramo de la línea (entre bombas)."""
    heaters = inputs.heaters
    opens = [i for i, h in enumerate(heaters) if h.kind == "open"]
    stages: list[list[int]] = [[] for _ in range(len(opens) + 1)]
    for i, h in enumerate(heaters):
        if h.kind == "closed":
            stages[sum(1 for j in opens if j < i)].append(i)
    return stages


def _line_pressures(inputs: RankineInputs) -> list[float]:
    """Presión a la salida de cada bomba de la línea, antes de resolver.

    Cada bomba lleva el líquido a la presión del próximo calentador abierto o,
    la última, a la de entrada a la turbina, más las caídas de presión del
    camino: los cerrados de su tramo y, la última, las cañerías y la caldera
    (Cengel §10-5).
    """
    heaters = inputs.heaters
    losses = inputs.losses
    opens = [i for i, h in enumerate(heaters) if h.kind == "open"]
    targets = [heaters[i].p_Pa for i in opens]
    last = inputs.p_boiler_Pa + losses.dp_boiler_Pa
    for pipe in (losses.feed_pipe, losses.steam_pipe):
        if pipe is not None:
            last += pipe.dp_Pa
    targets.append(last)
    return [
        p + sum(heaters[i].dp_Pa for i in closed)
        for p, closed in zip(targets, _stages(inputs), strict=True)
    ]


def _feedwater_line(inputs: RankineInputs) -> tuple[dict[int, float], float]:
    """T del agua que le llega a cada calentador y al final de la línea, antes de resolver.

    En cada tramo una bomba toma líquido saturado (del condensador, quizás
    subenfriado, o de un abierto) y lo lleva a la presión de su tramo; los
    cerrados, en orden, dejan el agua a T_sat(p_ext) − TTD (Cengel §10-6). Una
    mezcla con un drenaje bombeado la calienta algo más: la estimación se queda
    con la salida del cerrado (cota inferior).
    """
    heaters = inputs.heaters
    fluid = inputs.fluid
    opens = [i for i, h in enumerate(heaters) if h.kind == "open"]
    pressures = _line_pressures(inputs)
    T_in: dict[int, float] = {}
    T = 0.0
    for stage, closed in enumerate(_stages(inputs)):
        if stage == 0:
            inlet = _condensate_state(inputs)
        else:
            inlet = fluid_state_from_pair(fluid, "PX", p=heaters[opens[stage - 1]].p_Pa, x=0.0)
        p_line = pressures[stage]
        h = _pump_outlet_h(inlet, p_line, inputs.eta_pump)
        T = fluid_state_from_pair(fluid, "PH", p=p_line, h=h).T_K
        for i in closed:
            T_in[i] = T
            T = _T_sat(heaters[i].p_Pa, fluid) - heaters[i].ttd_K
        if stage < len(opens):
            T_in[opens[stage]] = T
    return T_in, T


def _feedwater_inlet_T(inputs: RankineInputs) -> dict[int, float]:
    """T del agua de alimentación que le llega a cada calentador, antes de resolver."""
    return _feedwater_line(inputs)[0]


def _validate_heater_data(inputs: RankineInputs) -> None:
    """Calentadores de agua de alimentación (Cengel §10-6): presiones, tipos y TTD."""
    heaters = inputs.heaters
    if not heaters:
        return
    if len(heaters) > MAX_HEATERS:
        raise ValueError(
            f"Se admiten hasta {MAX_HEATERS} calentadores de agua de alimentación "
            f"(cargaste {len(heaters)})."
        )
    fluid = inputs.fluid
    lim = fluid_limits(fluid)
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
        if h.kind == "open" and (
            h.dca_K is not None or h.desuperheater or h.drain_forward or h.dp_Pa != 0.0
        ):
            raise ValueError(
                f"El {name} es abierto: el subenfriador de drenaje, el desrecalentador, el "
                "drenaje bombeado y la caída de presión del agua son de los calentadores cerrados."
            )
        if not math.isfinite(h.ttd_K) or (h.ttd_K < 0.0 and not h.desuperheater):
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
        if h.dca_K is not None and not (math.isfinite(h.dca_K) and h.dca_K >= 0.0):
            raise ValueError(
                f"La aproximación del subenfriador de drenaje (DCA) del {name} no puede ser "
                "negativa: el drenaje sale, como mucho, a la temperatura del agua que entra al "
                "calentador (DCA = 0)."
            )
        if not (math.isfinite(h.dp_Pa) and h.dp_Pa >= 0.0):
            raise ValueError(
                f"La caída de presión del agua en el {name} no puede ser negativa (0 = sin "
                "pérdida de carga en los tubos)."
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
    if inputs.reheat is not None and inputs.losses.dp_reheater_Pa > 0.0:
        p_rh = inputs.reheat.p_Pa
        p_lp = p_rh - inputs.losses.dp_reheater_Pa
        for name, h in zip(names, heaters, strict=True):
            if p_lp <= h.p_Pa < p_rh * (1.0 - 1.0e-4):
                raise ValueError(
                    f"La presión de extracción del {name} ({_bar(h.p_Pa)}) cae dentro del "
                    f"recalentador: la turbina de alta descarga a {_bar(p_rh)} y la de baja "
                    f"arranca a {_bar(p_lp)}. Extraé a {_bar(p_rh)} o más (de la turbina de "
                    f"alta) o por debajo de {_bar(p_lp)} (de la de baja)."
                )


def _validate_heater_operation(inputs: RankineInputs) -> None:
    """Que cada calentador pueda calentar, con su subenfriador y su desrecalentador."""
    heaters = inputs.heaters
    if not heaters:
        return
    fluid = inputs.fluid
    names = heater_names(heaters)
    T_inlet = _feedwater_inlet_T(inputs)
    for i, T_in in T_inlet.items():
        h = heaters[i]
        T_max = _T_sat(h.p_Pa, fluid) - h.ttd_K
        if T_max <= T_in + _T_SAT_TOL:
            raise ValueError(
                f"El {names[i]} no puede calentar el agua de alimentación: le llega a "
                f"{_degC(T_in)} y, con la extracción a {_bar(h.p_Pa)}, como mucho la deja a "
                f"T_sat − TTD = {_degC(T_max)}. Subí su presión de extracción"
                + (", bajá el TTD" if h.kind == "closed" else "")
                + " o separala más de la del calentador anterior."
            )
        if h.dca_K is not None:
            T_sat = _T_sat(h.p_Pa, fluid)
            if T_in + h.dca_K >= T_sat - _T_SAT_TOL:
                raise ValueError(
                    f"Con DCA = {h.dca_K:g} K el drenaje del {names[i]} saldría a "
                    f"{_degC(T_in + h.dca_K)}, sin bajar de la saturación de la extracción "
                    f"({_degC(T_sat)}): el subenfriador no tendría nada que enfriar. Bajá el DCA "
                    "o sacá el subenfriador."
                )
    if any(h.desuperheater for h in heaters):
        layout = inputs.layout
        line = _expansion_line(inputs, layout)
        for i, h in enumerate(heaters):
            if not h.desuperheater:
                continue
            bleed = line[layout.extraction_states[i]]
            T_sat = _T_sat(h.p_Pa, fluid)
            if bleed.x is not None or bleed.T_K <= T_sat + _T_SAT_TOL:
                state = "vapor húmedo" if bleed.x is not None else "vapor saturado"
                quality = f" (x = {bleed.x:.3f})" if bleed.x is not None else ""
                raise ValueError(
                    f"El desrecalentador del {names[i]} necesita vapor sobrecalentado, pero la "
                    f"extracción a {_bar(h.p_Pa)} sale como {state}{quality}: no hay "
                    "sobrecalentamiento que aprovechar. Sacá el desrecalentador o usalo en un "
                    "calentador de más presión."
                )
            T_out = T_sat - h.ttd_K
            if T_out >= bleed.T_K - _T_SAT_TOL:
                raise ValueError(
                    f"Con TTD = {h.ttd_K:g} K el agua saldría del {names[i]} a {_degC(T_out)}, "
                    f"más caliente que la extracción que la calienta ({_degC(bleed.T_K)}). Subí "
                    "el TTD."
                )


def _validate_recuperator(inputs: RankineInputs) -> None:
    """Recuperador del ORC: efectividad, sin calentadores y escape sobrecalentado."""
    rec = inputs.recuperator
    if rec is None:
        return
    if not (math.isfinite(rec.effectiveness) and 0.0 < rec.effectiveness < 1.0):
        raise ValueError(
            f"La efectividad del recuperador tiene que estar entre 0 y 1, sin incluirlos "
            f"(ε = {rec.effectiveness:g}): ε = q/q_máx, y q_máx haría falta un intercambiador "
            "infinito."
        )
    if inputs.heaters:
        raise ValueError(
            "El recuperador va en el ciclo sin calentadores de agua de alimentación: en este "
            "modelo no se combinan (los dos precalientan el líquido que va a la caldera)."
        )
    layout = inputs.layout
    exhaust = _expansion_line(inputs, layout)[layout.exhaust]
    if exhaust.x is not None:
        raise ValueError(
            f"Con estos datos la turbina descarga vapor húmedo (x = {exhaust.x:.3f}): el "
            "recuperador no tiene calor sensible que aprovechar. Sirve con fluidos «secos» "
            "(R-245fa, isopentano, tolueno…), que salen sobrecalentados de la turbina; con "
            "agua o con fluidos húmedos, sacalo o sobrecalentá más."
        )
    T_pump = _feedwater_line(inputs)[1]
    if exhaust.T_K <= T_pump + 1.0:
        raise ValueError(
            f"El escape de la turbina ({_degC(exhaust.T_K)}) no está más caliente que el "
            f"líquido que sale de la bomba ({_degC(T_pump)}): el recuperador no tendría nada "
            "para pasarle."
        )


# ---------------------------------------------------------------------
# Cálculo con TESPy
# ---------------------------------------------------------------------


def _state_from_connection(connection: Connection, fluid: str = FLUID) -> FluidState:
    """Estado completo (región, x, v, u …) a partir de p y h de TESPy (SI)."""
    return fluid_state_from_pair(
        fluid, "PH", p=float(connection.p.val_SI), h=float(connection.h.val_SI)
    )


def _isentropic_state(p_out_Pa: float, s_in: float, fluid: str = FLUID) -> FluidState:
    """Estado a la presión de salida con la entropía de entrada (2s, 4s, 6s)."""
    return fluid_state_from_pair(fluid, "PS", p=p_out_Pa, s=s_in)


def _turbine_inlet_state(inputs: RankineInputs, casing: str) -> FluidState:
    """Estado a la entrada de cada turbina: caldera (o cañería) o recalentador."""
    fluid = inputs.fluid
    if casing == "turbina de baja":
        assert inputs.reheat is not None
        p_lp = inputs.reheat.p_Pa - inputs.losses.dp_reheater_Pa
        return fluid_state_from_pair(fluid, "TP", t=inputs.reheat.T_K, p=p_lp)
    if inputs.T_turbine_in_K is None:
        return fluid_state_from_pair(fluid, "PX", p=inputs.p_boiler_Pa, x=1.0)
    return fluid_state_from_pair(fluid, "TP", t=inputs.T_turbine_in_K, p=inputs.p_boiler_Pa)


def _section_end_pressure(inputs: RankineInputs, layout: PlantLayout, out_state: int) -> float:
    """Presión a la salida de un tramo de turbina (extracción, recalentamiento o escape)."""
    if out_state == layout.exhaust:
        return inputs.p_condenser_Pa
    if out_state == layout.reheat_in:
        assert inputs.reheat is not None
        return inputs.reheat.p_Pa
    return inputs.heaters[layout.extraction_states.index(out_state)].p_Pa


def _expansion_targets(
    inputs: RankineInputs, layout: PlantLayout
) -> list[tuple[CycleComponent, float, float, FluidState]]:
    """(tramo, p al final, h al final, estado al final) de cada tramo de turbina."""
    fluid = inputs.fluid
    eta = inputs.eta_turbine
    inlets: dict[str, FluidState] = {}
    targets = []
    for c in layout.of_kind("turbine"):
        inlet = inlets.setdefault(c.casing, _turbine_inlet_state(inputs, c.casing))
        p_end = _section_end_pressure(inputs, layout, c.port("out").state)
        h_s = _isentropic_state(p_end, inlet.s_J_per_kg_K, fluid).h_J_per_kg
        h_target = inlet.h_J_per_kg - eta * (inlet.h_J_per_kg - h_s)
        state = fluid_state_from_pair(fluid, "PH", p=p_end, h=h_target)
        targets.append((c, p_end, h_target, state))
    return targets


def _expansion_line(inputs: RankineInputs, layout: PlantLayout) -> dict[int, FluidState]:
    """Estado al final de cada tramo de turbina, antes de resolver.

    η_T se mide desde la entrada de cada turbina, como en las soluciones de
    Cengel: h_k = h_ent − η_T·(h_ent − h_ks), con ks a la presión de la
    extracción y la entropía de la entrada. Sirve para fijar el ``eta_s`` de
    cada tramo y para saber, antes de resolver, si una extracción o el escape
    salen sobrecalentados (desrecalentador, recuperador).
    """
    return {c.port("out").state: state for c, _, _, state in _expansion_targets(inputs, layout)}


def _section_efficiencies(inputs: RankineInputs, layout: PlantLayout) -> dict[str, float]:
    """``eta_s`` de cada tramo de turbina para TESPy.

    El primer tramo de cada turbina lleva η_T; los siguientes, el rendimiento
    local que reproduce la misma línea de expansión (ver
    :func:`_expansion_line`; un poco menor, porque los saltos isoentrópicos de
    los tramos suman más que el de la turbina entera).
    """
    eta = inputs.eta_turbine
    if eta == 1.0:
        return {c.label: 1.0 for c in layout.of_kind("turbine")}
    previous: dict[str, tuple[float, float]] = {}
    efficiencies: dict[str, float] = {}
    for c, p_end, h_target, target in _expansion_targets(inputs, layout):
        if c.casing not in previous:
            efficiencies[c.label] = eta
        else:
            h_prev, s_prev = previous[c.casing]
            h_s_local = _isentropic_state(p_end, s_prev, inputs.fluid).h_J_per_kg
            efficiencies[c.label] = (h_prev - h_target) / (h_prev - h_s_local)
        previous[c.casing] = (h_target, target.s_J_per_kg_K)
    return efficiencies


def _tespy_fluid(fluid: str) -> str:
    """Nombre del fluido para TESPy: ``water`` como en el tutorial; el resto, el de CoolProp."""
    return "water" if fluid == FLUID else fluid


def _pipe_loss(inputs: RankineInputs, layout: PlantLayout, comp: CycleComponent) -> PipeLoss:
    """Datos de una cañería: la de alimentación termina en la caldera; la de vapor sale de ella."""
    pipe = (
        inputs.losses.feed_pipe
        if comp.port("out").state == layout.boiler_in
        else inputs.losses.steam_pipe
    )
    assert pipe is not None
    return pipe


@dataclass
class _Network:
    """Red de TESPy armada desde el layout, con el mapa a los estados lógicos."""

    network: Any
    states: dict[int, Connection]  # conexión principal de cada estado
    ports: dict[tuple[int, int], Connection]  # (componente, conexión) -> conexión de TESPy
    objects: list[list[Any]]  # objetos de TESPy de cada componente lógico (el principal primero)


def _build_network(inputs: RankineInputs, layout: PlantLayout) -> _Network:
    """Red de TESPy de la planta (ejemplos oficiales de TESPy 0.11, ver el módulo).

    La presión se fija una sola vez por grupo (entrada de la turbina, salida
    de cada tramo); bombas, válvulas y ``Merge`` la toman de la red, y las
    caídas de presión van como ``dp`` en cada componente. Sin pérdidas, sin
    calentadores reales ni recuperador, los componentes, las etiquetas y los
    datos son los de la 0.12.0.
    """
    network = new_network()
    closer = CycleCloser("cierre del ciclo")
    eta_sections = _section_efficiencies(inputs, layout)
    losses = inputs.losses
    objects: list[list[Any]] = []
    inlets: dict[tuple[int, int], tuple[Any, str]] = {}
    outlets: dict[tuple[int, int], tuple[Any, str]] = {}
    aux: list[tuple[Any, str, Any, str, str]] = []

    for ci, comp in enumerate(layout.components):
        roles = [p.role for p in comp.ports]

        def at(role: str, k: int = 0, *, _roles: list[str] = roles) -> int:
            return [i for i, r in enumerate(_roles) if r == role][k]

        extra: list[Any] = []
        if comp.kind == "pump":
            obj: Any = Pump(comp.label)
            obj.set_attr(eta_s=inputs.eta_pump)
        elif comp.kind in ("boiler", "reheater", "condenser"):
            obj = SimpleHeatExchanger(comp.label)
            dp = {
                "boiler": losses.dp_boiler_Pa,
                "reheater": losses.dp_reheater_Pa,
                "condenser": losses.dp_condenser_Pa,
            }[comp.kind]
            if dp > 0.0:
                obj.set_attr(dp=dp)  # pérdida de carga (Cengel §10-5)
            else:
                obj.set_attr(pr=1)  # sin pérdidas de carga, como en Cengel
        elif comp.kind == "pipe":
            pipe = _pipe_loss(inputs, layout, comp)
            obj = TespyPipe(comp.label)
            if pipe.dp_Pa > 0.0:
                obj.set_attr(dp=pipe.dp_Pa)
            else:
                obj.set_attr(pr=1)
            if pipe.dT_K == 0.0:
                obj.set_attr(Q=0)  # cañería aislada: h constante
        elif comp.kind == "turbine":
            obj = Turbine(comp.label)
            obj.set_attr(eta_s=eta_sections[comp.label])
        elif comp.kind == "valve":
            obj = Valve(comp.label)
        elif comp.kind == "open_heater":
            obj = Merge(comp.label, num_in=len(comp.inlets))
        elif comp.kind == "mixer":
            obj = Merge(comp.label, num_in=2)
        elif comp.kind == "recuperator":
            assert inputs.recuperator is not None
            obj = HeatExchanger(comp.label)
            obj.set_attr(pr1=1, pr2=1, eff_max=inputs.recuperator.effectiveness)
        else:  # calentador cerrado
            assert comp.heater is not None
            heater = inputs.heaters[comp.heater]
            obj = Condenser(comp.label)
            obj.set_attr(pr1=1)
            if heater.dp_Pa > 0.0:
                obj.set_attr(dp2=heater.dp_Pa)
            else:
                obj.set_attr(pr2=1)
            if heater.dca_K is not None:
                # Subenfriador de drenaje: el drenaje sale a T_entrada del agua + DCA
                # (como el ejemplo de subcooling de la documentación de TESPy).
                obj.set_attr(subcooling=True, ttd_l=heater.dca_K)
            if heater.desuperheater:
                extra.append(Desuperheater(f"desrecalentador del {comp.label}"))
                extra[0].set_attr(pr1=1, pr2=1)
            else:
                obj.set_attr(ttd_u=heater.ttd_K)
        objects.append([obj, *extra])

        if comp.kind in ("pump", "boiler", "reheater", "turbine", "valve", "pipe"):
            inlets[(ci, at("in"))] = (obj, "in1")
            outlets[(ci, at("out"))] = (obj, "out1")
        elif comp.kind == "recuperator":  # lado caliente in1/out1, líquido in2/out2
            inlets[(ci, at("in"))] = (obj, "in1")
            outlets[(ci, at("out"))] = (obj, "out1")
            inlets[(ci, at("fw_in"))] = (obj, "in2")
            outlets[(ci, at("fw_out"))] = (obj, "out2")
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
            outlets[(ci, at("drain_out"))] = (obj, "out1")
            # La extracción entra a la zona de condensación o, con desrecalentador,
            # primero a él (sale como vapor saturado).
            steam_in: tuple[Any, str]
            if extra:
                desuperheater = extra[0]
                outlets[(ci, at("fw_out"))] = (desuperheater, "out2")
                aux.append(
                    (obj, "out2", desuperheater, "in2", f"agua entre las zonas del {comp.label}")
                )
                inlets[(ci, at("bleed"))] = (desuperheater, "in1")
                steam_in = (desuperheater, "out1")
            else:
                outlets[(ci, at("fw_out"))] = (obj, "out2")
                steam_in = None  # type: ignore[assignment]
            drains = comp.ports_with("drain_in")
            if drains:
                merge = Merge(f"mezcla en el {comp.label}", num_in=1 + len(drains))
                if extra:
                    aux.append((*steam_in, merge, "in1", f"vapor saturado en el {comp.label}"))
                else:
                    inlets[(ci, at("bleed"))] = (merge, "in1")
                for k in range(len(drains)):
                    inlets[(ci, at("drain_in", k))] = (merge, f"in{k + 2}")
                aux.append((merge, "out1", obj, "in1", f"entrada al {comp.label}"))
            elif extra:
                aux.append((*steam_in, obj, "in1", f"vapor saturado en el {comp.label}"))
            else:
                inlets[(ci, at("bleed"))] = (obj, "in1")

    # La caldera entrega al cierre del ciclo, y el cierre a lo que sigue (tutorial).
    boiler_i = next(i for i, c in enumerate(layout.components) if c.kind == "boiler")
    boiler_out = (boiler_i, [p.role for p in layout.components[boiler_i].ports].index("out"))
    boiler_obj, _ = outlets.pop(boiler_out)
    aux_boiler = Connection(boiler_obj, "out1", closer, "in1", label="0")

    producers: dict[int, tuple[Any, str, tuple[int, int] | None]] = {
        layout.boiler_out: (closer, "out1", None)
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
    fluid = inputs.fluid
    inlet: dict[str, Any] = {"p": inputs.p_boiler_Pa, "m": 1.0, "fluid": {_tespy_fluid(fluid): 1}}
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
    if losses.subcooling_K > 0.0:
        states[0].set_attr(td_bubble=losses.subcooling_K)  # condensado subenfriado
    else:
        states[0].set_attr(x=0.0)  # el condensador entrega líquido saturado
    for comp in layout.of_kind("open_heater"):
        states[comp.port("out").state].set_attr(x=0.0)  # Cengel: sale líquido saturado
    if losses.steam_pipe is not None and losses.steam_pipe.dT_K > 0.0:
        T_turbine = inputs.T_turbine_in_K
        if T_turbine is None:
            T_turbine = _T_sat(inputs.p_boiler_Pa, fluid)
        states[layout.boiler_out].set_attr(T=T_turbine + losses.steam_pipe.dT_K)
    if losses.feed_pipe is not None and losses.feed_pipe.dT_K > 0.0:
        (feed,) = (c for c in layout.of_kind("pipe") if c.port("out").state == layout.boiler_in)
        states[layout.boiler_in].set_attr(
            T=Ref(states[feed.port("in").state], 1, -losses.feed_pipe.dT_K)
        )
    for comp in layout.of_kind("closed_heater"):
        assert comp.heater is not None
        heater = inputs.heaters[comp.heater]
        if heater.desuperheater:  # el TTD se mide con el agua que sale del desrecalentador
            states[comp.port("fw_out").state].set_attr(T=_T_sat(heater.p_Pa, fluid) - heater.ttd_K)
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


def _check_drain_coolers(
    inputs: RankineInputs, components: tuple[CycleComponent, ...], states: tuple[FluidState, ...]
) -> None:
    """Subenfriador de drenaje: dentro del calentador el agua no puede pasar T_sat.

    En la zona de subenfriamiento el agua toma del drenaje lo que este pierde
    al pasar de líquido saturado a T_e + DCA; si el drenaje fuera mucho más
    caudal que el agua, el agua quedaría más caliente que la condensación.
    """
    fluid = inputs.fluid
    for comp in components:
        if comp.kind != "closed_heater" or comp.heater is None:
            continue
        heater = inputs.heaters[comp.heater]
        if heater.dca_K is None:
            continue
        fw_in, drain = comp.port("fw_in"), comp.port("drain_out")
        sat = saturation_at_pressure(fluid, states[drain.state].P_Pa)
        water = states[fw_in.state]
        h_b = (
            water.h_J_per_kg
            + drain.fraction
            * (sat.liquid.h_J_per_kg - states[drain.state].h_J_per_kg)
            / fw_in.fraction
        )
        T_b = fluid_state_from_pair(fluid, "PH", p=water.P_Pa, h=h_b).T_K
        if T_b >= sat.T_sat_K - _T_SAT_TOL:
            raise ValueError(
                f"El subenfriador del {comp.label} no puede cumplir DCA = {heater.dca_K:g} K: "
                "el drenaje le pasaría tanto calor al agua que esta superaría la temperatura "
                "de condensación dentro del calentador. Subí el DCA."
            )


def solve_rankine(inputs: RankineInputs) -> RankineResult:
    """Resuelve el ciclo con TESPy (Witte & Tuschy, 2020) y arma el resultado.

    La red sigue los ejemplos oficiales de TESPy 0.11 (ver el módulo): un
    ``CycleCloser`` cierra el lazo, la caldera, el recalentador y el
    condensador son ``SimpleHeatExchanger`` (sin pérdida de carga, ``pr = 1``,
    o con su ``dp``), turbinas y bombas llevan su ``eta_s``, los calentadores
    son ``Merge`` (abiertos) o ``Condenser`` (cerrados, con ``Desuperheater``
    si tienen desrecalentador), las cañerías ``Pipe`` y el recuperador un
    ``HeatExchanger``. Se resuelve por kilogramo de vapor en la caldera
    (1 kg/s) y las potencias se escalan con el caudal.

    Raises
    ------
    ValueError
        Si los datos no tienen sentido físico (ver
        :func:`validate_rankine_inputs`) o TESPy no converge.
    """
    validate_rankine_inputs(inputs)
    fluid = inputs.fluid
    layout = inputs.layout
    built = _build_network(inputs, layout)
    try:
        solve(built.network, what="el ciclo de Rankine")
    except ValueError as exc:
        hints = []
        if inputs.heaters:
            hints.append(
                "Con regeneración suele pasar cuando dos presiones de extracción están muy "
                "cerca o un calentador casi no tiene nada que calentar."
            )
        if any(h.desuperheater for h in inputs.heaters):
            hints.append(
                "Con desrecalentador, también puede ser que la extracción no tenga "
                "sobrecalentamiento suficiente para el TTD pedido: subí el TTD."
            )
        if not hints:
            raise
        raise ValueError(f"{exc} " + " ".join(hints)) from exc

    # TESPy da p y h; el resto del estado (región, título, v, u, s) se arma
    # con CoolProp: el título de las conexiones de TESPy no es confiable fuera
    # de la campana.
    states = tuple(_state_from_connection(built.states[i], fluid) for i in range(layout.n_states))
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
        _check_drain_coolers(inputs, components, states)

    pumps = [c for c in components if c.kind == "pump"]
    pumps_out_s = tuple(
        _isentropic_state(
            states[c.port("out").state].P_Pa, states[c.port("in").state].s_J_per_kg_K, fluid
        )
        for c in pumps
    )
    casing_inlet: dict[str, int] = {}
    for c in components:
        if c.kind == "turbine":
            casing_inlet.setdefault(c.casing, c.port("in").state)
    turbine_out_s = tuple(
        _isentropic_state(
            states[c.port("out").state].P_Pa, states[casing_inlet[c.casing]].s_J_per_kg_K, fluid
        )
        for c in components
        if c.kind == "turbine"
    )
    objects = dict(zip((c.label for c in components), built.objects, strict=True))

    def power_or_heat(comp: CycleComponent) -> float:
        main = objects[comp.label][0]
        return float((main.P if comp.kind in ("turbine", "pump") else main.Q).val_SI)

    balances = tuple(
        (c.label, power_or_heat(c))
        for kind in ("turbine", "pump", "boiler", "reheater", "pipe", "condenser")
        for c in components
        if c.kind == kind
    )
    internal_heat = tuple(
        (c.label, -sum(float(obj.Q.val_SI) for obj in objects[c.label]))
        for c in components
        if c.kind in ("closed_heater", "recuperator")
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
        tespy_heaters=internal_heat,
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


FluidBehavior = Literal["húmedo", "isoentrópico", "seco"]


@dataclass(frozen=True)
class FluidBehaviorInfo:
    """Cómo es la curva de vapor saturado de un fluido en el diagrama T–s.

    ``xi`` es la pendiente ξ = ds_g/dT [J/(kg·K²)] a T = 0,8·T_c y
    ``xi_star = T·ξ/s_fg`` la misma pendiente adimensional, que permite
    comparar fluidos.
    """

    behavior: FluidBehavior
    xi_J_per_kg_K2: float
    xi_star: float
    T_K: float


def fluid_behavior(fluid: str) -> FluidBehaviorInfo:
    """Fluido húmedo, seco o casi isoentrópico (Chen, Goswami y Stefanakos, 2010).

    Según el signo de la pendiente ξ = ds_g/dT de la curva de vapor saturado
    en T–s (Chen, H., Goswami, D. Y. y Stefanakos, E. K. (2010), *Renewable
    and Sustainable Energy Reviews* 14, 3059–3067,
    doi:10.1016/j.rser.2010.07.006): negativa (agua, amoníaco) es un fluido
    **húmedo**, la expansión desde vapor saturado entra en la campana;
    positiva (R-245fa, isopentano, tolueno) es **seco**, termina sobrecalentada;
    cerca de cero (R-134a, R-1234yf), **casi isoentrópico**. La pendiente
    cambia con la temperatura: se evalúa a T = 0,8·T_c y se clasifica con la
    forma adimensional T·ξ/s_fg (|·| < 0,25: casi isoentrópico).
    """
    lim = fluid_limits(fluid)
    T = 0.8 * lim.T_crit_K
    dT = 0.5
    s_minus = saturation_at_temperature(fluid, T - dT).vapor.s_J_per_kg_K
    s_plus = saturation_at_temperature(fluid, T + dT).vapor.s_J_per_kg_K
    xi = (s_plus - s_minus) / (2.0 * dT)
    xi_star = T * xi / saturation_at_temperature(fluid, T).s_fg_J_per_kg_K
    behavior: FluidBehavior
    if xi_star > 0.25:
        behavior = "seco"
    elif xi_star < -0.25:
        behavior = "húmedo"
    else:
        behavior = "isoentrópico"
    return FluidBehaviorInfo(behavior, float(xi), float(xi_star), float(T))


def rankine_notes(result: RankineResult, system: UnitSystem = "Técnico") -> list[str]:
    """Observaciones didácticas sobre el ciclo calculado (markdown), con los
    valores en ``system``."""
    notes: list[str] = []
    fluid = result.inputs.fluid
    x = result.x_turbine_out
    if x is not None and x < X_TURBINE_MIN:
        notes.append(
            f"El vapor sale de la turbina con título x = {x:.3f}: más de 10–12 % de humedad "
            "erosiona los álabes de las últimas etapas (Cengel §10-4). Recalentar o "
            "sobrecalentar más lo evita."
        )
    if x is None:
        if result.has_recuperator:
            notes.append(
                "El vapor sale **sobrecalentado** de la turbina: el recuperador aprovecha parte "
                "de ese calor para precalentar el líquido, y el condensador enfría el resto "
                "hasta T_sat(p) y recién después lo condensa."
            )
        else:
            notes.append(
                "El vapor sale **sobrecalentado** de la turbina: el condensador primero lo enfría "
                "hasta T_sat(p) y recién después lo condensa."
            )
    if result.inputs.p_boiler_Pa >= fluid_limits(fluid).P_crit_Pa:
        if fluid == FLUID:
            notes.append(
                "Ciclo **supercrítico**: en la caldera el agua pasa de líquido a vapor sin hervir "
                "(por encima de la presión crítica no hay campana)."
            )
        else:
            notes.append(
                f"Ciclo **supercrítico**: en el evaporador {_el(fluid)} pasa de líquido a vapor "
                "sin hervir (por encima de la presión crítica no hay campana)."
            )
    if fluid != FLUID:
        info = fluid_behavior(fluid)
        name = FLUID_NAMES_ES.get(fluid, fluid)
        if info.behavior == "seco":
            text = (
                f"El {name} es un **fluido seco**: en el diagrama T–s su curva de vapor saturado "
                "tiene pendiente positiva, así que la expansión termina sobrecalentada aun "
                "entrando saturado a la turbina (no hace falta sobrecalentar y no hay humedad "
                "en los álabes)."
            )
            text += (
                " El recuperador aprovecha ese calor del escape."
                if result.has_recuperator
                else " Ese calor del escape se puede aprovechar con un recuperador."
            )
        elif info.behavior == "húmedo":
            text = (
                f"El {name} es un **fluido húmedo**, como el agua: su curva de vapor saturado "
                "tiene pendiente negativa en T–s y la expansión desde vapor saturado entra en la "
                "campana."
            )
        else:
            text = (
                f"El {name} es un fluido **casi isoentrópico**: su curva de vapor saturado es casi "
                "vertical en T–s y la expansión desde vapor saturado termina cerca de la "
                "saturación."
            )
        notes.append(text + " (Chen, Goswami y Stefanakos, 2010.)")
    reference = textbook_reference_note(fluid, system)
    if reference is not None:  # R-134a: las tablas de Cengel usan otra referencia
        notes.append(reference)
    return notes


def suggested_rankine_inputs(fluid: str) -> RankineInputs:
    """Datos por defecto de un ciclo con ``fluid`` (SI); siempre se pueden resolver.

    Agua: Cengel 10-1. Fluidos de ORC: evaporación unos 25 K por debajo de la
    temperatura crítica (en °C redondos), condensación a 30 °C, η_T = 0,8 y
    η_B = 0,7 (valores típicos de un ORC; Quoilin et al., 2013). Entra vapor
    saturado; un fluido húmedo, 10 K sobrecalentado. Los fluidos secos llevan
    recuperador (ε = 0,8).
    """
    if fluid == FLUID:
        return RankineInputs(3.0e6, 623.15, 75.0e3)
    lim = fluid_limits(fluid)
    T_ev = 5.0 * math.floor((lim.T_crit_K - 25.0 - 273.15) / 5.0) + 273.15
    p_hi = round(saturation_at_temperature(fluid, T_ev).P_sat_Pa / 1.0e4) * 1.0e4
    p_lo = round(saturation_at_temperature(fluid, 303.15).P_sat_Pa / 1.0e3) * 1.0e3
    info = fluid_behavior(fluid)
    T_in = None
    if info.behavior == "húmedo":
        T_in = float(round(_T_sat(p_hi, fluid) + 10.0))
    return RankineInputs(
        p_hi,
        T_in,
        p_lo,
        eta_turbine=0.8,
        eta_pump=0.7,
        fluid=fluid,
        recuperator=Recuperator(0.8) if info.behavior == "seco" else None,
    )


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


def _cengel_10_2() -> RankineInputs:
    """Cengel ej. 10-2 (ciclo real), con los estados de la figura del libro.

    La bomba entrega 16 MPa; el agua llega a la caldera a 15,9 MPa y 35 °C y
    sale a 15,2 MPa y 625 °C; la turbina recibe 15 MPa y 600 °C y descarga a
    10 kPa; el condensado sale a 9 kPa y 38 °C. Se cargan como caídas de
    presión y de temperatura; la de la cañería de alimentación sale de la
    temperatura a la que la bomba (η_B = 0,85) deja el agua.
    """
    p1, T1 = 9.0e3, 311.15
    inlet = fluid_state_from_pair(FLUID, "TP", t=T1, p=p1)
    h2 = _pump_outlet_h(inlet, 16.0e6, 0.85)
    T2 = fluid_state_from_pair(FLUID, "PH", p=16.0e6, h=h2).T_K
    return RankineInputs(
        15.0e6,
        873.15,
        10.0e3,
        0.87,
        0.85,
        m_dot_kg_s=15.0,
        losses=Losses(
            dp_boiler_Pa=0.7e6,
            dp_condenser_Pa=1.0e3,
            subcooling_K=_T_sat(p1) - T1,
            feed_pipe=PipeLoss(0.1e6, T2 - 308.15),
            steam_pipe=PipeLoss(0.2e6, 25.0),
        ),
    )


_CENGEL_10_2 = "Cengel 10-2 — ciclo real, con pérdidas de carga y de calor (η_T = 0,87, η_B = 0,85)"
_REAL_HEATERS = (
    "Calentadores reales: cerrado de baja con drenaje bombeado, desaireador y cerrado de alta "
    "con desrecalentador y subenfriador"
)
RANKINE_EXAMPLES.update(
    {
        _CENGEL_10_2: _cengel_10_2(),
        _REAL_HEATERS: RankineInputs(
            15.0e6,
            873.15,
            10.0e3,
            0.87,
            0.85,
            heaters=(
                FeedwaterHeater(2.0e5, "closed", 2.8, drain_forward=True),
                FeedwaterHeater(12.0e5),
                FeedwaterHeater(40.0e5, "closed", -1.5, dca_K=5.6, desuperheater=True),
            ),
        ),
        "ORC con R-245fa y recuperador (20 bar, vapor saturado; 1,8 bar)": RankineInputs(
            20.0e5,
            None,
            1.8e5,
            0.8,
            0.7,
            m_dot_kg_s=30.0,
            fluid="R245fa",
            recuperator=Recuperator(0.8),
        ),
        "ORC con tolueno y recuperador (25 bar, 300 °C; 0,08 bar)": RankineInputs(
            25.0e5,
            573.15,
            0.08e5,
            0.8,
            0.7,
            m_dot_kg_s=5.0,
            fluid="Toluene",
            recuperator=Recuperator(0.8),
        ),
        "ORC con R-134a, fluido casi isoentrópico (26 bar, vapor saturado; 8 bar)": (
            RankineInputs(26.0e5, None, 8.0e5, 0.8, 0.7, m_dot_kg_s=50.0, fluid="R134a")
        ),
    }
)

#: Aclaraciones de algunos ejemplos (se muestran junto al resultado).
RANKINE_EXAMPLE_NOTES: dict[str, str] = {
    "Cengel 10-6 — recalentamiento y regeneración: cerrado a 4 MPa y abierto a 0,5 MPa": (
        "El libro informa y = 0,1766 (cerrado), z = 0,1306 (abierto) y η = 49,2 %. Esos "
        "números salen de calcular la bomba II solo hasta 4 MPa (h₄ = 643,9 kJ/kg); como el "
        "agua de alimentación va a 15 MPa, acá h₄ = 655,9 kJ/kg y quedan y ≈ 0,173, "
        "z ≈ 0,131 y η ≈ 49,0 %."
    ),
    _CENGEL_10_2: (
        "El libro da η = 36,1 % y Ẇ_neto = 18,9 MW para ṁ = 15 kg/s. Las temperaturas de su "
        "figura se cargan como pérdidas: el agua que la bomba deja a 39,1 °C llega a la "
        "caldera a 35 °C (la cañería pierde 4,13 K), el vapor sale de la caldera a 625 °C y "
        "llega a la turbina a 600 °C (25 K), y el condensado sale a 38 °C, 5,76 K por debajo "
        "de T_sat(9 kPa). El libro calcula la bomba con v·Δp; acá, con la ecuación de estado."
    ),
    _REAL_HEATERS: (
        "TTD = −1,5 K en el cerrado de alta (con desrecalentador el agua sale por encima de "
        "T_sat) y DCA = 5,6 K son valores habituales de diseño. Como el drenaje del cerrado "
        "de baja se bombea a la línea, las fracciones de extracción quedan acopladas y se "
        "resuelven juntas."
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
            f"{_bar(result.inputs.p_condenser_Pa)} el vapor condensa a T_sat = "
            f"{_degC(T_cond)}, y el agua saldría a {_degC(T_out_K)}. Bajá la temperatura de "
            "salida del agua o subí la presión del condensador."
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

SweepParameter = Literal["p_boiler", "T_turbine_in", "p_condenser", "recuperator_eff"]


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
    if parameter == "recuperator_eff":
        return [float(v) for v in np.linspace(0.1, 0.95, n)]
    if inputs.fluid != FLUID:
        return _orc_sweep_values(inputs, parameter, n)
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


def _orc_sweep_values(inputs: RankineInputs, parameter: SweepParameter, n: int) -> list[float]:
    """Como :func:`default_sweep_values`, para un fluido de ORC: de la condensación a
    cerca del punto crítico y hasta la temperatura máxima de su ecuación de estado."""
    fluid = inputs.fluid
    lim = fluid_limits(fluid)
    if parameter == "p_boiler":
        T_cond = _T_sat(inputs.p_condenser_Pa, fluid)
        p_20K = saturation_at_temperature(fluid, T_cond + 20.0).P_sat_Pa
        low = max(2.0 * inputs.p_condenser_Pa, p_20K)
        if inputs.reheat is not None:
            low = max(low, inputs.reheat.p_Pa * 1.25)
        if inputs.heaters:
            low = max(low, inputs.heaters[-1].p_Pa * 1.25)
        high = 0.95 * lim.P_crit_Pa
        if inputs.p_boiler_Pa > high:  # ORC supercrítico: alrededor de la presión dada
            high = min(1.5 * inputs.p_boiler_Pa, lim.P_max_Pa)
        if high <= low:
            return [inputs.p_boiler_Pa]
        return [float(v) for v in np.geomspace(low, high, n)]
    if parameter == "T_turbine_in":
        p = inputs.p_boiler_Pa
        T_floor = lim.T_crit_K if p >= lim.P_crit_Pa else _T_sat(p, fluid)
        T_top = min(lim.T_max_K - 5.0, T_floor + 120.0)
        return [float(v) for v in np.linspace(T_floor + 5.0, T_top, n)]
    low = saturation_at_temperature(fluid, max(lim.T_triple_K + 5.0, 288.15)).P_sat_Pa
    high = saturation_at_temperature(fluid, min(333.15, lim.T_crit_K - 10.0)).P_sat_Pa
    high = min(high, 0.5 * inputs.p_boiler_Pa)
    if inputs.reheat is not None:
        high = min(high, 0.5 * inputs.reheat.p_Pa)
    if inputs.heaters:
        high = min(high, 0.5 * inputs.heaters[0].p_Pa)
    if high <= low:
        return [inputs.p_condenser_Pa]
    return [float(v) for v in np.geomspace(low, high, n)]


def rankine_sweep(
    inputs: RankineInputs, parameter: SweepParameter, values: Sequence[float]
) -> list[SweepPoint]:
    """Resuelve el ciclo para cada valor del parámetro (como la sección 8 del
    tutorial de TESPy). Los valores sin sentido físico se omiten."""
    points: list[SweepPoint] = []
    for value in values:
        if parameter == "recuperator_eff":
            if inputs.recuperator is None:
                return []
            trial = replace(inputs, recuperator=Recuperator(float(value)))
        else:
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
    extracción; con pérdidas, las del ciclo real y el calor perdido; con
    recuperador, su efectividad y su calor; con ``cooling``, el agua de
    enfriamiento del condensador.
    """
    inputs = result.inputs
    if inputs.heaters:
        cycle = "Rankine regenerativo" + (" con recalentamiento" if result.has_reheat else "")
    else:
        cycle = "Rankine con recalentamiento" if result.has_reheat else "Rankine simple"
    if inputs.fluid != FLUID:
        cycle = cycle.replace("Rankine", "Rankine orgánico (ORC)", 1)
    if result.has_recuperator:
        cycle += " con recuperador"
    if result.has_losses:
        cycle += ", ciclo real (con pérdidas)"
    data: dict[str, Any] = {
        "ciclo": cycle,
        "fluido": inputs.fluid,
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
        "fuente": "TESPy 0.11 (red de componentes) + CoolProp ("
        + ("IAPWS-95" if inputs.fluid == FLUID else "ecuación de estado de Helmholtz")
        + ")",
    }
    if inputs.fluid != FLUID:
        data["fluido_es"] = FLUID_NAMES_ES.get(inputs.fluid, inputs.fluid)
    if result.has_losses:
        losses = inputs.losses
        dT: QuantityKind = "temperature_difference"
        data["datos"]["perdidas"] = {
            "dp_caldera": _value(losses.dp_boiler_Pa, "pressure", system),
            "dp_recalentador": _value(losses.dp_reheater_Pa, "pressure", system),
            "dp_condensador": _value(losses.dp_condenser_Pa, "pressure", system),
            "subenfriamiento_condensado": _value(losses.subcooling_K, dT, system),
            "cañeria_de_alimentacion": (
                None
                if losses.feed_pipe is None
                else {
                    "dp": _value(losses.feed_pipe.dp_Pa, "pressure", system),
                    "dT": _value(losses.feed_pipe.dT_K, dT, system),
                }
            ),
            "cañeria_de_vapor": (
                None
                if losses.steam_pipe is None
                else {
                    "dp": _value(losses.steam_pipe.dp_Pa, "pressure", system),
                    "dT": _value(losses.steam_pipe.dT_K, dT, system),
                }
            ),
        }
        data["resultados"]["q_perdido_en_cañerias"] = _value(result.q_loss_J_per_kg, _EH, system)
        data["resultados"]["calor_perdido"] = _value(result.Q_loss_W, "power", system)
    if inputs.recuperator is not None:
        data["datos"]["recuperador"] = {"efectividad": inputs.recuperator.effectiveness}
        data["resultados"]["q_recuperador"] = _value(result.q_recuperator_J_per_kg, _EH, system)
    if inputs.heaters:
        assert result.layout is not None
        names = result.layout.heater_names
        heaters: list[dict[str, Any]] = []
        for name, h in zip(names, inputs.heaters, strict=True):
            entry: dict[str, Any] = {
                "nombre": name,
                "tipo": "abierto" if h.kind == "open" else "cerrado",
                "p_extraccion": _value(h.p_Pa, "pressure", system),
                "TTD": _value(h.ttd_K, "temperature_difference", system),
            }
            # Solo lo que el calentador usa (sin calentadores reales, como en la 0.12.0).
            if h.dca_K is not None:
                entry["DCA"] = _value(h.dca_K, "temperature_difference", system)
            if h.desuperheater:
                entry["desrecalentador"] = True
            if h.drain_forward:
                entry["drenaje_bombeado_hacia_adelante"] = True
            if h.dp_Pa:
                entry["dp_agua"] = _value(h.dp_Pa, "pressure", system)
            heaters.append(entry)
        data["datos"]["calentadores"] = heaters
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
