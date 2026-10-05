"""Ciclo de Rankine con TESPy — Fase 3.1a.

Ciclo de vapor de agua simple (bomba, caldera, turbina, condensador), ideal
o real (rendimientos isoentrópicos de turbina y bomba), y con
recalentamiento en una etapa. Se resuelve con una red de TESPy armada como
el tutorial oficial (``tutorial/basics/rankine.py`` de TESPy 0.11:
``CycleCloser`` + ``SimpleHeatExchanger`` + ``Turbine`` + ``Pump``); el
condensador se modela, como en Cengel, con salida en líquido saturado
(x = 0) y sin pérdidas de carga.

Numeración de estados de Çengel & Boles, *Termodinámica*, cap. 10:
1 entrada a la bomba (líquido saturado), 2 salida de la bomba, 3 entrada a
la turbina, 4 salida de la turbina. Con recalentamiento: 3 entrada a la
turbina de alta, 4 salida de alta (entrada al recalentador), 5 entrada a la
turbina de baja, 6 salida de baja.

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
from tespy.components import CycleCloser, Pump, SimpleHeatExchanger, Turbine
from tespy.connections import Connection

from core.cycles.tespy_utils import new_network, solve
from core.fluids import FluidState, fluid_limits, fluid_state_from_pair, saturation_at_pressure
from core.latex import latex_chain, latex_number, latex_paren, latex_quantity, latex_value
from core.state_report import ProcedureStep, pv_energy_factor, states_table
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

#: Fluido de trabajo (nombre de CoolProp).
FLUID = "Water"

_EH: QuantityKind = "specific_enthalpy"
_ES: QuantityKind = "specific_entropy"

# Margen mínimo sobre T_sat para la entrada a una turbina (K): más cerca,
# CoolProp no distingue el estado de la campana con T y p.
_T_SAT_TOL = 0.01

# Título de salida de turbina por debajo del cual se avisa (Cengel §10-6:
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
    se calcula. El mismo rendimiento ``eta_turbine`` vale para las dos
    turbinas del ciclo con recalentamiento.
    """

    p_boiler_Pa: float
    T_turbine_in_K: float | None
    p_condenser_Pa: float
    eta_turbine: float = 1.0
    eta_pump: float = 1.0
    reheat: Reheat | None = None
    m_dot_kg_s: float | None = 1.0
    W_net_W: float | None = None

    @property
    def is_ideal(self) -> bool:
        """Turbina y bomba isoentrópicas (el ciclo ideal de Rankine)."""
        return self.eta_turbine == 1.0 and self.eta_pump == 1.0


@dataclass(frozen=True)
class RankineResult:
    """Estados y balances del ciclo, en SI.

    ``states`` sigue la numeración de Cengel (1 … 4, o 1 … 6 con
    recalentamiento). ``pump_out_s`` es el estado 2s y ``turbine_out_s``
    tiene la salida isoentrópica de cada turbina (4s, o 4s y 6s).
    ``tespy_balances`` son los resultados por componente que calcula TESPy
    para 1 kg/s (potencia o calor en W, positivos hacia el fluido), como
    control cruzado de los balances con entalpías.
    """

    inputs: RankineInputs
    states: tuple[FluidState, ...]
    pump_out_s: FluidState
    turbine_out_s: tuple[FluidState, ...]
    m_dot_kg_s: float
    tespy_balances: tuple[tuple[str, float], ...] = ()

    @property
    def h_pump_out_s_J_per_kg(self) -> float:
        return self.pump_out_s.h_J_per_kg

    @property
    def h_turbine_out_s_J_per_kg(self) -> tuple[float, ...]:
        return tuple(state.h_J_per_kg for state in self.turbine_out_s)

    # --- índices de los estados en el ciclo ----------------------------
    @property
    def has_reheat(self) -> bool:
        return self.inputs.reheat is not None

    @property
    def turbine_pairs(self) -> tuple[tuple[int, int], ...]:
        """(entrada, salida) de cada turbina, como índices de ``states``."""
        return ((2, 3), (4, 5)) if self.has_reheat else ((2, 3),)

    @property
    def heating_pairs(self) -> tuple[tuple[int, int], ...]:
        """(entrada, salida) de cada aporte de calor: caldera y recalentador."""
        return ((1, 2), (3, 4)) if self.has_reheat else ((1, 2),)

    # --- energías específicas (J/kg) -----------------------------------
    @property
    def w_pump_J_per_kg(self) -> float:
        return self.states[1].h_J_per_kg - self.states[0].h_J_per_kg

    @property
    def w_turbine_J_per_kg(self) -> float:
        return sum(
            self.states[i].h_J_per_kg - self.states[o].h_J_per_kg for i, o in self.turbine_pairs
        )

    @property
    def q_in_J_per_kg(self) -> float:
        return sum(
            self.states[o].h_J_per_kg - self.states[i].h_J_per_kg for i, o in self.heating_pairs
        )

    @property
    def q_out_J_per_kg(self) -> float:
        return self.states[-1].h_J_per_kg - self.states[0].h_J_per_kg

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
        return self.states[-1].x

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
        """Temperatura media de aporte de calor T̄_H = q_H / Δs (Cengel §10-6)."""
        delta_s = sum(
            self.states[o].s_J_per_kg_K - self.states[i].s_J_per_kg_K for i, o in self.heating_pairs
        )
        return self.q_in_J_per_kg / delta_s

    @property
    def eta_mean_temperature(self) -> float:
        """1 − T_C / T̄_H: igual a η en el ciclo ideal con salida húmeda de la turbina
        (todo el calor se cede a T_C constante)."""
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


def _bar(p_Pa: float) -> str:
    return f"{p_Pa / 1e5:.4g} bar"


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


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


def solve_rankine(inputs: RankineInputs) -> RankineResult:
    """Resuelve el ciclo con TESPy (Witte & Tuschy, 2020) y arma el resultado.

    La red sigue el tutorial oficial de TESPy 0.11: un ``CycleCloser``
    cierra el lazo, la caldera, el recalentador y el condensador son
    ``SimpleHeatExchanger`` sin pérdida de carga (``pr = 1``), y turbinas y
    bomba llevan su ``eta_s``. Se resuelve por kilogramo (1 kg/s) y las
    potencias se escalan con el caudal.

    Raises
    ------
    ValueError
        Si los datos no tienen sentido físico (ver
        :func:`validate_rankine_inputs`) o TESPy no converge.
    """
    validate_rankine_inputs(inputs)
    network = new_network()
    closer = CycleCloser("cierre del ciclo")
    boiler = SimpleHeatExchanger("caldera")
    condenser = SimpleHeatExchanger("condensador")
    pump = Pump("bomba")
    turbine_hp = Turbine("turbina de alta" if inputs.reheat else "turbina")

    c3 = Connection(closer, "out1", turbine_hp, "in1", label="3")
    if inputs.reheat is None:
        c4 = Connection(turbine_hp, "out1", condenser, "in1", label="4")
        cycle = [c3, c4]
        heat_exchangers = [boiler, condenser]
        turbines = [turbine_hp]
    else:
        reheater = SimpleHeatExchanger("recalentador")
        turbine_lp = Turbine("turbina de baja")
        c4 = Connection(turbine_hp, "out1", reheater, "in1", label="4")
        c5 = Connection(reheater, "out1", turbine_lp, "in1", label="5")
        c6 = Connection(turbine_lp, "out1", condenser, "in1", label="6")
        cycle = [c3, c4, c5, c6]
        heat_exchangers = [boiler, reheater, condenser]
        turbines = [turbine_hp, turbine_lp]
    c1 = Connection(condenser, "out1", pump, "in1", label="1")
    c2 = Connection(pump, "out1", boiler, "in1", label="2")
    c0 = Connection(boiler, "out1", closer, "in1", label="0")
    network.add_conns(*cycle, c1, c2, c0)

    for hx in heat_exchangers:
        hx.set_attr(pr=1)  # sin pérdidas de carga, como en Cengel
    for turbine in turbines:
        turbine.set_attr(eta_s=inputs.eta_turbine)
    pump.set_attr(eta_s=inputs.eta_pump)

    inlet: dict[str, Any] = {"p": inputs.p_boiler_Pa, "m": 1.0, "fluid": {"water": 1}}
    if inputs.T_turbine_in_K is None:
        inlet["x"] = 1.0
    else:
        inlet["T"] = inputs.T_turbine_in_K
    c3.set_attr(**inlet)
    if inputs.reheat is None:
        c4.set_attr(p=inputs.p_condenser_Pa)
    else:
        c4.set_attr(p=inputs.reheat.p_Pa)
        c5.set_attr(T=inputs.reheat.T_K)
        c6.set_attr(p=inputs.p_condenser_Pa)
    c1.set_attr(x=0.0)  # el condensador entrega líquido saturado

    solve(network, what="el ciclo de Rankine")

    # TESPy da p y h; el resto del estado (región, título, v, u, s) se arma
    # con CoolProp: el título de las conexiones de TESPy no es confiable fuera
    # de la campana.
    states = tuple(_state_from_connection(c) for c in (c1, c2, *cycle))
    pump_out_s = _isentropic_state(states[1].P_Pa, states[0].s_J_per_kg_K)
    pairs = ((2, 3), (4, 5)) if inputs.reheat else ((2, 3),)
    turbine_out_s = tuple(
        _isentropic_state(states[o].P_Pa, states[i].s_J_per_kg_K) for i, o in pairs
    )
    balances = tuple(
        (component.label, float(value.val_SI))
        for component, value in (
            *((t, t.P) for t in turbines),
            (pump, pump.P),
            *((hx, hx.Q) for hx in heat_exchangers),
        )
    )

    if inputs.reheat is not None and states[4].h_J_per_kg <= states[3].h_J_per_kg:
        raise ValueError(
            f"La temperatura de recalentamiento ({_degC(inputs.reheat.T_K)}) no supera la de "
            f"salida de la turbina de alta ({_degC(states[3].T_K)}): el «recalentador» tendría "
            "que enfriar el vapor. Subí la temperatura o la presión de recalentamiento."
        )
    w_net = sum(states[i].h_J_per_kg - states[o].h_J_per_kg for i, o in pairs) - (
        states[1].h_J_per_kg - states[0].h_J_per_kg
    )
    if w_net <= 0.0:
        raise ValueError(
            "Con esos datos el ciclo no entrega trabajo neto (la bomba consume más de lo que "
            "da la turbina). Revisá las presiones y los rendimientos."
        )
    m_dot = inputs.m_dot_kg_s if inputs.m_dot_kg_s is not None else inputs.W_net_W / w_net  # type: ignore[operator]
    return RankineResult(
        inputs=inputs,
        states=states,
        pump_out_s=pump_out_s,
        turbine_out_s=turbine_out_s,
        m_dot_kg_s=float(m_dot),
        tespy_balances=balances,
    )


# ---------------------------------------------------------------------
# Presentación: estados, avisos, ejemplos
# ---------------------------------------------------------------------

_LABELS_SIMPLE = (
    "1 (entrada a la bomba)",
    "2 (salida de la bomba)",
    "3 (entrada a la turbina)",
    "4 (salida de la turbina)",
)
_LABELS_REHEAT = (
    "1 (entrada a la bomba)",
    "2 (salida de la bomba)",
    "3 (entrada a la turbina de alta)",
    "4 (salida de la turbina de alta)",
    "5 (entrada a la turbina de baja)",
    "6 (salida de la turbina de baja)",
)


def rankine_labeled_states(result: RankineResult) -> list[tuple[str, FluidState]]:
    """``(etiqueta, estado)`` en el orden del ciclo, para la tabla y el diagrama."""
    labels = _LABELS_REHEAT if result.has_reheat else _LABELS_SIMPLE
    return list(zip(labels, result.states, strict=True))


def rankine_isentropic_states(result: RankineResult) -> list[tuple[str, FluidState]]:
    """Estados isoentrópicos de referencia (2s, 4s y 6s) del ciclo real."""
    turbine_labels = ("4s", "6s") if result.has_reheat else ("4s",)
    return [("2s", result.pump_out_s), *zip(turbine_labels, result.turbine_out_s, strict=True)]


def rankine_notes(result: RankineResult) -> list[str]:
    """Observaciones didácticas sobre el ciclo calculado (markdown)."""
    notes: list[str] = []
    x = result.x_turbine_out
    if x is not None and x < X_TURBINE_MIN:
        notes.append(
            f"El vapor sale de la turbina con título x = {x:.3f}: más de 10–12 % de humedad "
            "erosiona los álabes de las últimas etapas (Cengel §10-6). Recalentar o "
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
}


# ---------------------------------------------------------------------
# Procedimiento (LaTeX en el sistema de unidades activo)
# ---------------------------------------------------------------------


def _q(value_si: float, kind: QuantityKind, system: UnitSystem) -> str:
    return latex_quantity(value_si, kind, system)


def _n(value_si: float, kind: QuantityKind, system: UnitSystem) -> str:
    return latex_value(value_si, kind, system)


def _diff(a_si: float, b_si: float, kind: QuantityKind, system: UnitSystem) -> str:
    """``a - b`` con los números en ``system`` (``b`` entre paréntesis si es negativo)."""
    return rf"{_n(a_si, kind, system)} - {latex_paren(_n(b_si, kind, system))}"


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _sub(n: int) -> str:
    """Número de estado como subíndice para el texto: ``3`` → ``₃``."""
    return str(n).translate(_SUBSCRIPT_DIGITS)


def _wrap(head: str, op: str, tail: str) -> str:
    r"""``head op tail``; si hay números con ×10ⁿ (SI), ``op tail`` va en otro renglón.

    Pensado para un paso de :func:`~core.latex.latex_chain`: el corte
    ``\\ &\quad`` mantiene la ecuación dentro del ancho de un celular.
    """
    if r"\times" in head + tail:
        return rf"{head} \\ &\quad {op} {tail}"
    return f"{head} {op} {tail}"


def _absolute_T(T_K: float, system: UnitSystem) -> str:
    """Temperatura absoluta para Carnot: K (SI y Técnico) o °R (Inglés)."""
    if system == "Inglés":
        return rf"{latex_number(T_K * 1.8, 5)}\ \mathrm{{R}}"
    return rf"{latex_number(T_K, 5)}\ \mathrm{{K}}"


def _turbine_step(
    result: RankineResult, i: int, o: int, h_out_s: float, system: UnitSystem, title: str
) -> ProcedureStep:
    """Expansión i → o: estado isoentrópico (palanca si cae en la campana) y real."""
    s_in = result.states[i]
    s_out = result.states[o]
    a, b = i + 1, o + 1  # numeración de Cengel
    eta = result.inputs.eta_turbine
    sat = (
        saturation_at_pressure(FLUID, s_out.P_Pa)
        if s_out.P_Pa < fluid_limits(FLUID).P_crit_Pa
        else None
    )
    lines = [rf"s_{{{b}s}} = s_{{{a}}} = {_q(s_in.s_J_per_kg_K, _ES, system)}"]
    wet = sat is not None and sat.liquid.s_J_per_kg_K <= s_in.s_J_per_kg_K <= sat.vapor.s_J_per_kg_K
    if wet:
        assert sat is not None
        x_s = (s_in.s_J_per_kg_K - sat.liquid.s_J_per_kg_K) / sat.s_fg_J_per_kg_K
        lines.append(
            latex_chain(
                rf"x_{{{b}s}}",
                r"\frac{s - s_f}{s_{fg}}",
                rf"\frac{{{_n(s_in.s_J_per_kg_K, _ES, system)} - "
                rf"{latex_paren(_n(sat.liquid.s_J_per_kg_K, _ES, system))}}}"
                rf"{{{_n(sat.s_fg_J_per_kg_K, _ES, system)}}}",
                latex_number(x_s, 5),
            )
        )
        lines.append(
            latex_chain(
                rf"h_{{{b}s}}",
                rf"h_f + x_{{{b}s}}\,h_{{fg}}",
                _wrap(
                    _n(sat.liquid.h_J_per_kg, _EH, system),
                    "+",
                    rf"{latex_number(x_s, 5)}\cdot {_n(sat.h_fg_J_per_kg, _EH, system)}",
                ),
                _q(h_out_s, _EH, system),
            )
        )
        how = (
            f"A p = {_bar(s_out.P_Pa)} la entropía s{_sub(a)} cae dentro de la campana: el estado "
            f"isoentrópico {b}s es vapor húmedo y su título sale de la regla de la palanca, "
            "con s_fg = s_g − s_f y h_fg = h_g − h_f de la tabla de saturación (vademecum §12)."
        )
    else:
        lines.append(
            latex_chain(rf"h_{{{b}s}}", rf"h(p_{{{b}}},\ s_{{{b}s}})", _q(h_out_s, _EH, system))
        )
        how = (
            f"A p = {_bar(s_out.P_Pa)} el estado isoentrópico {b}s sigue sobrecalentado: se lee "
            "en la tabla de vapor sobrecalentado (Cengel A-6), interpolando."
        )
    if eta == 1.0:
        lines.append(rf"h_{{{b}}} = h_{{{b}s}} = {_q(s_out.h_J_per_kg, _EH, system)}")
        real = "Turbina ideal (isoentrópica): el estado real coincide con el isoentrópico."
    else:
        lines.append(
            latex_chain(
                rf"h_{{{b}}}",
                rf"h_{{{a}}} - \eta_T\,(h_{{{a}}} - h_{{{b}s}})",
                rf"{_n(s_in.h_J_per_kg, _EH, system)} \\ &\quad - {latex_number(eta, 4)}\,"
                rf"({_n(s_in.h_J_per_kg, _EH, system)} - "
                rf"{latex_paren(_n(h_out_s, _EH, system))})",
                _q(s_out.h_J_per_kg, _EH, system),
            )
        )
        real = (
            f"Con η_T = {eta:g}, la turbina entrega solo esa fracción del salto isoentrópico "
            "(vademecum §10.4)."
        )
    if s_out.x is not None and sat is not None and 0.0 < s_out.x < 1.0:
        if eta == 1.0:
            lines.append(rf"x_{{{b}}} = x_{{{b}s}} = {latex_number(s_out.x, 5)}")
        else:
            lines.append(
                latex_chain(
                    rf"x_{{{b}}}",
                    r"\frac{h - h_f}{h_{fg}}",
                    rf"\frac{{{_n(s_out.h_J_per_kg, _EH, system)} - "
                    rf"{latex_paren(_n(sat.liquid.h_J_per_kg, _EH, system))}}}"
                    rf"{{{_n(sat.h_fg_J_per_kg, _EH, system)}}}",
                    latex_number(s_out.x, 5),
                )
            )
    lines.append(
        latex_chain(
            rf"w_{{T,{a}{b}}}",
            rf"h_{{{a}}} - h_{{{b}}}",
            _diff(s_in.h_J_per_kg, s_out.h_J_per_kg, _EH, system),
            _q(s_in.h_J_per_kg - s_out.h_J_per_kg, _EH, system),
        )
    )
    return ProcedureStep(title=title, text=f"{how} {real}", latex=tuple(lines))


def rankine_steps(result: RankineResult, system: UnitSystem) -> list[ProcedureStep]:
    """Procedimiento estado por estado, como se resuelve con las tablas (Cengel §10-2)."""
    st = result.states
    s1, s2, s3 = st[0], st[1], st[2]
    inputs = result.inputs
    steps: list[ProcedureStep] = []

    # 1 — salida del condensador.
    steps.append(
        ProcedureStep(
            title="Estado 1: salida del condensador",
            text=(
                f"Sale líquido saturado (x₁ = 0) a la presión del condensador, "
                f"{_bar(s1.P_Pa)}: se lee en la tabla de saturación por presión (Cengel A-5)."
            ),
            latex=(
                rf"T_1 = T_{{\mathrm{{sat}}}}(p_1) = {_q(s1.T_K, 'temperature', system)}",
                rf"h_1 = h_f(p_1) = {_q(s1.h_J_per_kg, _EH, system)}",
                rf"v_1 = v_f(p_1) = {_q(s1.v_m3_per_kg, 'specific_volume', system)}",
                rf"s_1 = s_f(p_1) = {_q(s1.s_J_per_kg_K, _ES, system)}",
            ),
        )
    )

    # 1 → 2 — bomba.
    factor = pv_energy_factor(system)
    p_unit = unit_label("pressure", system)
    dp = _diff(s2.P_Pa, s1.P_Pa, "pressure", system)
    v1 = _n(s1.v_m3_per_kg, "specific_volume", system)
    # El factor de unidades (p. ej. 1 bar·m³/kg = 100 kJ/kg) va en su propio renglón
    # para que la ecuación entre en un celular.
    substitution = rf"{v1}\,({dp})" + (
        "" if math.isclose(factor, 1.0) else rf" \\ &\quad \cdot {latex_number(factor, 5)}"
    )
    w_ps_approx = s1.v_m3_per_kg * (s2.P_Pa - s1.P_Pa)
    w_ps = result.h_pump_out_s_J_per_kg - s1.h_J_per_kg
    pump_lines = [
        latex_chain(
            r"w_{B,s}",
            r"v_1\,(p_2 - p_1)",
            substitution,
            _q(w_ps_approx, _EH, system),
            relation=r"\approx",
        ),
        rf"h_{{2s}} = h(p_2,\ s_1) = {_q(result.h_pump_out_s_J_per_kg, _EH, system)}",
    ]
    if inputs.eta_pump == 1.0:
        pump_lines.append(
            latex_chain(
                "w_B",
                "h_2 - h_1",
                _diff(s2.h_J_per_kg, s1.h_J_per_kg, _EH, system),
                _q(result.w_pump_J_per_kg, _EH, system),
            )
        )
        pump_txt = "Bomba ideal: h₂ = h₂s."
    else:
        pump_lines.append(
            latex_chain(
                "w_B",
                r"\frac{h_{2s} - h_1}{\eta_B}",
                rf"\frac{{{_n(w_ps, _EH, system)}}}{{{latex_number(inputs.eta_pump, 4)}}}",
                _q(result.w_pump_J_per_kg, _EH, system),
            )
        )
        pump_lines.append(
            latex_chain(
                "h_2",
                "h_1 + w_B",
                rf"{_n(s1.h_J_per_kg, _EH, system)} + "
                rf"{latex_paren(_n(result.w_pump_J_per_kg, _EH, system))}",
                _q(s2.h_J_per_kg, _EH, system),
            )
        )
        pump_txt = (
            f"Con η_B = {inputs.eta_pump:g}, la bomba real consume más que la ideal "
            "(vademecum §10.4)."
        )
    units_note = (
        ""
        if math.isclose(factor, 1.0)
        else f" En este sistema, 1 {p_unit}·{unit_label('specific_volume', system)} = "
        f"{latex_number(factor, 5)} {unit_label(_EH, system)}."
    )
    steps.append(
        ProcedureStep(
            title="1 → 2: bomba",
            text=(
                "La bomba lleva el líquido a la presión de la caldera. Como el líquido es casi "
                "incompresible, el trabajo ideal es w = v₁·(p₂ − p₁) (vademecum §13); con la "
                f"ecuación de estado, h₂s = h(p₂, s₁).{units_note} {pump_txt}"
            ),
            latex=tuple(pump_lines),
        )
    )

    # 2 → 3 — caldera.
    inlet_txt = (
        "vapor saturado seco (x₃ = 1)"
        if inputs.T_turbine_in_K is None
        else f"vapor a {_degC(s3.T_K)}"
    )
    steps.append(
        ProcedureStep(
            title="2 → 3: caldera",
            text=(
                f"La caldera calienta el agua a p constante ({_bar(s3.P_Pa)}) hasta {inlet_txt}; "
                "h₃ y s₃ se leen en la tabla de vapor (Cengel A-6). Balance de sistema abierto "
                "sin trabajo (vademecum §3.3):"
            ),
            latex=(
                rf"h_3 = {_q(s3.h_J_per_kg, _EH, system)}",
                rf"s_3 = {_q(s3.s_J_per_kg_K, _ES, system)}",
                latex_chain(
                    r"q_{\mathrm{cald}}",
                    "h_3 - h_2",
                    _diff(s3.h_J_per_kg, s2.h_J_per_kg, _EH, system),
                    _q(s3.h_J_per_kg - s2.h_J_per_kg, _EH, system),
                ),
            ),
        )
    )

    # 3 → 4 (y 4 → 5 → 6) — turbinas y recalentador.
    if result.has_reheat:
        steps.append(
            _turbine_step(
                result, 2, 3, result.h_turbine_out_s_J_per_kg[0], system, "3 → 4: turbina de alta"
            )
        )
        s4, s5 = st[3], st[4]
        steps.append(
            ProcedureStep(
                title="4 → 5: recalentador",
                text=(
                    f"El vapor vuelve a la caldera y se recalienta a p constante "
                    f"({_bar(s4.P_Pa)}) hasta {_degC(s5.T_K)}:"
                ),
                latex=(
                    rf"h_5 = {_q(s5.h_J_per_kg, _EH, system)}",
                    rf"s_5 = {_q(s5.s_J_per_kg_K, _ES, system)}",
                    latex_chain(
                        r"q_{\mathrm{rec}}",
                        "h_5 - h_4",
                        _diff(s5.h_J_per_kg, s4.h_J_per_kg, _EH, system),
                        _q(s5.h_J_per_kg - s4.h_J_per_kg, _EH, system),
                    ),
                ),
            )
        )
        steps.append(
            _turbine_step(
                result, 4, 5, result.h_turbine_out_s_J_per_kg[1], system, "5 → 6: turbina de baja"
            )
        )
    else:
        steps.append(
            _turbine_step(
                result, 2, 3, result.h_turbine_out_s_J_per_kg[0], system, "3 → 4: turbina"
            )
        )

    # Condensador.
    last = len(st)
    steps.append(
        ProcedureStep(
            title=f"{last} → 1: condensador",
            text="El condensador cede calor a p constante hasta líquido saturado:",
            latex=(
                latex_chain(
                    "q_C",
                    rf"h_{{{last}}} - h_1",
                    _diff(st[-1].h_J_per_kg, s1.h_J_per_kg, _EH, system),
                    _q(result.q_out_J_per_kg, _EH, system),
                ),
            ),
        )
    )

    # Balance y rendimiento.
    w_t = " + ".join(rf"w_{{T,{i + 1}{o + 1}}}" for i, o in result.turbine_pairs)
    q_h = r"q_{\mathrm{cald}} + q_{\mathrm{rec}}" if result.has_reheat else r"q_{\mathrm{cald}}"
    steps.append(
        ProcedureStep(
            title="Balance y rendimiento térmico",
            text=(
                "El trabajo neto es lo que da la turbina menos lo que consume la bomba, y "
                "también el calor recibido menos el cedido (primer principio para el ciclo). "
                "El rendimiento es η = W/Q_H = 1 − Q_C/Q_H (vademecum §9.2)."
            ),
            latex=(
                latex_chain(r"q_H", q_h, _q(result.q_in_J_per_kg, _EH, system)),
                latex_chain(r"w_T", w_t, _q(result.w_turbine_J_per_kg, _EH, system)),
                latex_chain(
                    r"w_{\mathrm{neto}}",
                    "w_T - w_B",
                    rf"{_n(result.w_turbine_J_per_kg, _EH, system)} - "
                    rf"{latex_paren(_n(result.w_pump_J_per_kg, _EH, system))}",
                    _q(result.w_net_J_per_kg, _EH, system),
                ),
                latex_chain(
                    r"\eta",
                    r"\frac{w_{\mathrm{neto}}}{q_H}",
                    rf"\frac{{{_n(result.w_net_J_per_kg, _EH, system)}}}"
                    rf"{{{_n(result.q_in_J_per_kg, _EH, system)}}}",
                    rf"{latex_number(result.eta_th * 100, 4)}\,\%",
                ),
                latex_chain(
                    r"\mathrm{BWR}",
                    r"\frac{w_B}{w_T}",
                    rf"{latex_number(result.back_work_ratio * 100, 3)}\,\%",
                ),
            ),
        )
    )

    # Comparación con Carnot.
    T_c = _absolute_T(result.T_low_K, system)
    T_h = _absolute_T(result.T_high_K, system)
    delta_s = " + ".join(rf"(s_{{{o + 1}}} - s_{{{i + 1}}})" for i, o in result.heating_pairs)
    steps.append(
        ProcedureStep(
            title="Comparación con Carnot",
            text=(
                "Un ciclo de Carnot entre la temperatura máxima y la de condensación (en "
                "temperaturas absolutas) es el techo del rendimiento (vademecum §9.2). El "
                "Rankine queda abajo porque recibe el calor a una temperatura media T̄_H menor "
                "que la máxima: subir T̄_H (más presión, más sobrecalentamiento, recalentar) es "
                "lo que mejora el ciclo (Cengel §10-6)."
            ),
            latex=(
                latex_chain(
                    r"\eta_{\mathrm{Carnot}}",
                    r"1 - \frac{T_C}{T_H}",
                    rf"1 - \frac{{{T_c}}}{{{T_h}}}",
                    rf"{latex_number(result.eta_carnot * 100, 4)}\,\%",
                ),
                latex_chain(
                    r"\bar{T}_H",
                    rf"\frac{{q_H}}{{{delta_s}}}",
                    _absolute_T(result.T_mean_in_K, system),
                ),
                latex_chain(
                    r"1 - \frac{T_C}{\bar{T}_H}",
                    rf"{latex_number(result.eta_mean_temperature * 100, 4)}\,\%",
                ),
            ),
        )
    )

    # Caudal y potencias.
    m_line = (
        latex_chain(
            r"\dot{m}",
            r"\frac{\dot{W}_{\mathrm{neto}}}{w_{\mathrm{neto}}}",
            _q(result.m_dot_kg_s, "mass_flow", system),
        )
        if inputs.W_net_W is not None
        else rf"\dot{{m}} = {_q(result.m_dot_kg_s, 'mass_flow', system)}"
    )
    steps.append(
        ProcedureStep(
            title="Caudal y potencias",
            text="Cada energía específica por el caudal másico da la potencia correspondiente.",
            latex=(
                m_line,
                latex_chain(
                    r"\dot{W}_{\mathrm{neto}}",
                    r"\dot{m}\,w_{\mathrm{neto}}",
                    _q(result.W_net_W, "power", system),
                ),
                latex_chain(r"\dot{Q}_H", r"\dot{m}\,q_H", _q(result.Q_in_W, "power", system)),
                latex_chain(r"\dot{Q}_C", r"\dot{m}\,q_C", _q(result.Q_out_W, "power", system)),
            ),
        )
    )
    return steps


# ---------------------------------------------------------------------
# Barridos: ¿cómo aumentar el rendimiento? (Cengel §10-6)
# ---------------------------------------------------------------------

SweepParameter = Literal["p_boiler", "T_turbine_in", "p_condenser"]


@dataclass(frozen=True)
class SweepPoint:
    """Un punto del barrido: valor del parámetro (SI) y resultado del ciclo."""

    value_si: float
    eta_th: float
    w_net_J_per_kg: float
    x_turbine_out: float | None


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
        points.append(
            SweepPoint(float(value), result.eta_th, result.w_net_J_per_kg, result.x_turbine_out)
        )
    return points


# ---------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------


def _value(value_si: float, kind: QuantityKind, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def rankine_to_dict(result: RankineResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``)."""
    inputs = result.inputs
    return {
        "ciclo": "Rankine con recalentamiento" if result.has_reheat else "Rankine simple",
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
