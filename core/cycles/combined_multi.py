"""Ciclo combinado de una, dos o tres presiones, con recalentamiento — Fase 3.6.

La **turbina de gas** (:mod:`core.cycles.brayton`) descarga en una **caldera
de recuperación en cascada** (:mod:`core.cycles.hrsg_multi`, Fase 3.5) con
uno, dos o tres niveles de presión y, opcionalmente, un **recalentador** en
paralelo con el sobrecalentador de alta. El vapor de cada nivel entra a la
**turbina de vapor** a la presión de su domo (*admisiones*): el de alta por la
entrada; el de media y el de baja se mezclan con el que viene de la turbina
(mezclas adiabáticas a presión constante). Con tres niveles y
recalentamiento, el vapor de media se suma al recalentamiento frío antes del
recalentador (Kehlhofer, Hannemann, Stirnimann y Rukes, *Combined-Cycle Gas &
Steam Turbine Power Plants*, 3.ª ed., PennWell, 2009, cap. 5).

El orden es alta → (recalentador) → media → baja → condensador. Cada tramo
entre una admisión (o el recalentador) y la siguiente es una turbina, con el
rendimiento ``eta_turbine`` medido desde su entrada (como el Rankine de las
Fases 3.1). El desaireador (opcional) se alimenta con una extracción, como en
la Fase 3.4; sin él, el domo de baja hace de desaireador y el condensado va
directo al economizador de baja. La **regla de Baumann** (opcional) baja el
rendimiento en la parte húmeda de cada expansión: η_húmedo = η·(1 − α·ȳ), con
ȳ la humedad media de esa parte.

El cálculo es directo, como «a mano»: el agua de alimentación y el
recalentamiento frío no dependen de los caudales; la HRSG da los caudales y
la turbina, las potencias. :func:`combined_multi_tespy` resuelve el lado
agua–vapor con TESPy como control cruzado. Con un nivel y sin recalentamiento
reproduce el ciclo combinado de la Fase 3.4. Todo en SI. No importa
Streamlit.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from core.cycles.brayton import BraytonInputs, BraytonResult, brayton_notes, solve_brayton
from core.cycles.combined import (
    SH_HOT_END_K,
    CombinedInputs,
    SteamCycle,
    gas_turbine_to_dict,
    solve_combined,
)
from core.cycles.hrsg import Water, _bar, _degC
from core.cycles.hrsg_multi import (
    MAX_LEVELS,
    MultiHRSGInputs,
    MultiHRSGResult,
    PressureLevel,
    Reheater,
    hrsg_exergy,
    level_names,
    multi_hrsg_notes,
    multi_hrsg_to_dict,
    solve_multi_hrsg,
)
from core.fluids import FluidState, fluid_limits, fluid_state_from_pair
from core.ideal_gas import AIR_DRY, T_GAS_MAX_K
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "COMBINED_MULTI_EXAMPLES",
    "COMBINED_MULTI_EXAMPLE_NOTES",
    "Admission",
    "BottomingExergy",
    "ConfigurationRow",
    "MultiCombinedInputs",
    "MultiCombinedResult",
    "MultiCombinedSweepParameter",
    "MultiCombinedSweepPoint",
    "MultiCombinedTespy",
    "MultiSteamCycle",
    "ReheatSpec",
    "TurbineSection",
    "WetExpansion",
    "bottoming_exergy",
    "combined_multi_notes",
    "combined_multi_sweep",
    "combined_multi_tespy",
    "combined_multi_to_dict",
    "configuration_comparison",
    "default_multi_combined_sweep_values",
    "from_combined",
    "from_combined_with_pinch",
    "solve_combined_multi",
    "validate_multi_combined_inputs",
]

_EH: QuantityKind = "specific_enthalpy"
#: Título mínimo razonable a la salida de la turbina de vapor (Cengel §10-5; Kehlhofer).
X_EXHAUST_MIN = 0.88


# ---------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ReheatSpec:
    """Recalentamiento: temperatura del vapor recalentado y su presión (SI).

    Con tres niveles la presión es la de media (``p_Pa=None``): el vapor de media
    se suma al recalentamiento frío. Con uno o dos niveles hay que darla.
    """

    T_K: float
    p_Pa: float | None = None


@dataclass(frozen=True)
class MultiCombinedInputs:
    """Datos del ciclo combinado de varias presiones, en SI.

    ``levels`` son los niveles de la HRSG, de mayor a menor presión (1 a 3);
    ``steam`` el condensador, los rendimientos y el desaireador (los de la Fase
    3.4); ``reheat`` el recalentamiento (``None``: sin recalentar). Con
    ``baumann_alpha`` > 0 la humedad baja el rendimiento de la turbina (regla de
    Baumann; α = 1 es la original). Si se da ``W_net_W``, el caudal de aire se
    ajusta para lograr esa potencia neta (todo escala con el caudal).
    """

    gas_turbine: BraytonInputs
    levels: tuple[PressureLevel, ...]
    steam: SteamCycle
    reheat: ReheatSpec | None = None
    baumann_alpha: float = 0.0
    W_net_W: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))

    @property
    def names(self) -> tuple[str, ...]:
        return level_names(len(self.levels))

    @property
    def reheat_joins_middle(self) -> bool:
        """Con tres niveles, el vapor de media se suma al recalentamiento frío."""
        return self.reheat is not None and len(self.levels) == MAX_LEVELS

    @property
    def p_reheat_Pa(self) -> float | None:
        if self.reheat is None:
            return None
        if self.reheat_joins_middle:
            return self.levels[1].p_Pa
        return self.reheat.p_Pa

    @property
    def configuration(self) -> str:
        """«3 presiones con recalentamiento», «1 presión», …"""
        n = len(self.levels)
        text = f"{n} presi{'ón' if n == 1 else 'ones'}"
        return text + (" con recalentamiento" if self.reheat is not None else "")


def from_combined(inputs: CombinedInputs) -> MultiCombinedInputs:
    """El ciclo combinado de la Fase 3.4 (diseño por pinch) en el modelo de varias presiones."""
    if inputs.T_stack_K is not None:
        raise ValueError("El modelo de varias presiones se diseña por pinch, no por chimenea.")
    level = PressureLevel(inputs.p_steam_Pa, inputs.T_steam_K, inputs.pinch_K, inputs.approach_K)
    return MultiCombinedInputs(
        gas_turbine=inputs.gas_turbine,
        levels=(level,),
        steam=inputs.steam,
        W_net_W=inputs.W_net_W,
    )


def from_combined_with_pinch(inputs: CombinedInputs) -> MultiCombinedInputs:
    """Como :func:`from_combined`, pero acepta también el diseño por chimenea.

    Un ciclo diseñado por la temperatura de chimenea (Cengel §10-9, ejemplo 10-9)
    se resuelve una vez con la Fase 3.4 y pasa al modelo de varias presiones con
    el pinch que resulta: es el mismo ciclo (mismo η y misma chimenea, test a
    1e-9). Si el diseño por chimenea no cierra (cruce de temperaturas), el
    ``ValueError`` es el de la Fase 3.4.
    """
    if inputs.T_stack_K is None:
        return from_combined(inputs)
    pinch_K = solve_combined(inputs).hrsg.pinch_K
    return from_combined(replace(inputs, T_stack_K=None, pinch_K=pinch_K))


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class WetExpansion:
    """La parte húmeda de una expansión con la regla de Baumann (SI).

    La línea de expansión (con η_T desde la entrada) cruza la curva de vapor
    saturado en ``p_x_Pa``; de ahí en adelante el rendimiento es
    η_húmedo = η_T·(1 − α·ȳ), con ȳ = (y_entrada + y_salida)/2.
    """

    p_x_Pa: float
    h_x_J_per_kg: float
    y_in: float
    y_out: float
    eta_wet: float


@dataclass(frozen=True)
class TurbineSection:
    """Un tramo de la turbina de vapor, entre dos admisiones (o el recalentador).

    ``inlet`` y ``outlet`` son índices de :attr:`MultiSteamCycle.states`;
    ``outlet_s`` la salida isoentrópica desde la entrada. ``extraction`` es el
    estado de la extracción del desaireador, si cae en este tramo.
    """

    name: str
    inlet: int
    outlet: int
    m_in_kg_s: float
    outlet_s: FluidState
    W_W: float
    extraction: int | None = None
    m_extraction_kg_s: float = 0.0
    wet: WetExpansion | None = None


@dataclass(frozen=True)
class Admission:
    """Una mezcla adiabática a presión constante: el vapor de un nivel se suma al de la turbina.

    ``level`` es el nivel cuyo vapor entra; ``turbine`` el estado que viene de la
    turbina (o el recalentamiento frío), ``steam`` el del vapor del nivel y
    ``outlet`` la mezcla.
    """

    level: int
    turbine: int
    steam: int
    outlet: int
    m_turbine_kg_s: float
    m_steam_kg_s: float

    @property
    def m_out_kg_s(self) -> float:
        return self.m_turbine_kg_s + self.m_steam_kg_s


@dataclass(frozen=True)
class MultiSteamCycle:
    """El lado agua–vapor resuelto (SI), con los estados numerados en el sentido del flujo.

    La numeración arranca en el condensado (1 = índice 0): línea de
    alimentación, vapor de alta, turbinas, admisiones y la salida al
    condensador. Coincide con la de Cengel en un Rankine simple, con
    recalentamiento o con un desaireador.
    """

    states: tuple[FluidState, ...]
    labels: tuple[str, ...]
    turbines: tuple[TurbineSection, ...]
    admissions: tuple[Admission, ...]
    level_states: tuple[int, ...]
    feedwater: int
    condensate_pump_out: int
    deaerator: int | None
    cold_reheat: int | None
    reheat_inlet: int | None
    hot_reheat: int | None
    exhaust: int
    m_total_kg_s: float
    m_extraction_kg_s: float
    W_condensate_pump_W: float
    W_feed_pump_W: float

    @property
    def m_condenser_kg_s(self) -> float:
        return self.m_total_kg_s - self.m_extraction_kg_s

    @property
    def W_turbines_W(self) -> float:
        return sum(t.W_W for t in self.turbines)

    @property
    def W_pumps_W(self) -> float:
        """Bomba de condensado y, con desaireador, la de alimentación (las de la HRSG aparte)."""
        return self.W_condensate_pump_W + self.W_feed_pump_W

    @property
    def Q_condenser_W(self) -> float:
        exh, cond = self.states[self.exhaust], self.states[0]
        return self.m_condenser_kg_s * (exh.h_J_per_kg - cond.h_J_per_kg)

    @property
    def x_exhaust(self) -> float:
        """Título a la salida de la turbina (1 si sale seco)."""
        x = self.states[self.exhaust].x
        return 1.0 if x is None else float(x)

    @property
    def y_deaerator(self) -> float:
        """Fracción del vapor total que se extrae para el desaireador."""
        return self.m_extraction_kg_s / self.m_total_kg_s

    @property
    def extraction(self) -> int | None:
        return next((t.extraction for t in self.turbines if t.extraction is not None), None)


@dataclass(frozen=True)
class MultiCombinedResult:
    """Turbina de gas, HRSG de varias presiones y ciclo de vapor resueltos y acoplados (SI)."""

    inputs: MultiCombinedInputs
    gas_turbine: BraytonResult
    hrsg: MultiHRSGResult
    steam: MultiSteamCycle

    @property
    def W_gas_turbine_W(self) -> float:
        return self.gas_turbine.W_net_W

    @property
    def W_pumps_W(self) -> float:
        """Todas las bombas del ciclo de vapor (condensado, alimentación y entre niveles)."""
        return self.steam.W_pumps_W + self.hrsg.W_pumps_W

    @property
    def W_steam_turbine_W(self) -> float:
        """Potencia neta del ciclo de vapor: turbinas menos todas las bombas."""
        return self.steam.W_turbines_W - self.W_pumps_W

    @property
    def W_net_W(self) -> float:
        return self.W_gas_turbine_W + self.W_steam_turbine_W

    @property
    def Q_fuel_W(self) -> float:
        return self.gas_turbine.Q_in_W

    @property
    def eta_th(self) -> float:
        """η_CC = (Ẇ_TG + Ẇ_TV) / Q̇_comb."""
        return self.W_net_W / self.Q_fuel_W

    @property
    def eta_gas_turbine(self) -> float:
        return self.gas_turbine.eta_th

    @property
    def Q_exhaust_W(self) -> float:
        return self.Q_fuel_W - self.W_gas_turbine_W

    @property
    def Q_hrsg_W(self) -> float:
        return self.hrsg.Q_W

    @property
    def eta_hrsg(self) -> float:
        return self.Q_hrsg_W / self.Q_exhaust_W

    @property
    def eta_steam(self) -> float:
        """Rendimiento del ciclo de vapor sobre el calor de la HRSG."""
        return self.W_steam_turbine_W / self.Q_hrsg_W

    @property
    def Q_condenser_W(self) -> float:
        return self.steam.Q_condenser_W

    @property
    def Q_stack_W(self) -> float:
        return self.Q_exhaust_W - self.Q_hrsg_W

    @property
    def m_steam_kg_s(self) -> float:
        return self.hrsg.m_steam_kg_s

    @property
    def steam_gas_ratio(self) -> float:
        return self.m_steam_kg_s / self.gas_turbine.m_gas_kg_s

    @property
    def heat_rate_kJ_per_kWh(self) -> float:
        return 3600.0 / self.eta_th

    @property
    def energy_balance_W(self) -> dict[str, float]:
        """Q̇_comb = Ẇ_TG + Ẇ_TV + Q̇_cond + Q̇_chim."""
        return {
            "turbina de gas": self.W_gas_turbine_W,
            "turbina de vapor": self.W_steam_turbine_W,
            "condensador": self.Q_condenser_W,
            "chimenea": self.Q_stack_W,
        }


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


def _T_sat(p_Pa: float) -> float:
    return float(PropsSI("T", "P", p_Pa, "Q", 0, Water))


def validate_multi_combined_inputs(inputs: MultiCombinedInputs) -> None:
    """Verifica el ciclo de vapor y el recalentamiento antes de calcular.

    La turbina de gas y la HRSG se validan en sus módulos (con el nombre de la
    parte en el mensaje).

    Raises
    ------
    ValueError
        Con un mensaje para el alumno: qué dato está fuera de rango y por qué.
    """
    if inputs.W_net_W is not None and not (math.isfinite(inputs.W_net_W) and inputs.W_net_W > 0.0):
        raise ValueError("La potencia neta tiene que ser positiva.")
    n = len(inputs.levels)
    if not 1 <= n <= MAX_LEVELS:
        raise ValueError(f"La caldera tiene que tener entre 1 y {MAX_LEVELS} niveles de presión.")
    if not 0.0 <= inputs.baumann_alpha <= 2.0:
        raise ValueError(
            "El factor de Baumann tiene que estar entre 0 (sin pérdida por humedad) y 2 "
            "(típico: 1)."
        )
    steam = inputs.steam
    lim = fluid_limits(Water)
    low, high = inputs.levels[-1], inputs.levels[0]
    if not lim.P_triple_Pa < steam.p_condenser_Pa < low.p_Pa:
        raise ValueError(
            f"Ciclo de vapor: la presión del condensador ({_bar(steam.p_condenser_Pa)}) tiene que "
            f"estar por debajo de la del nivel de menor presión ({_bar(low.p_Pa)}) y por encima "
            "de la del punto triple."
        )
    for name, eta in (("turbina", steam.eta_turbine), ("bomba", steam.eta_pump)):
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"Ciclo de vapor: el rendimiento de la {name} va entre 0 y 1.")
    events = [lv.p_Pa for lv in inputs.levels[1:]]
    if inputs.reheat is not None:
        rh = inputs.reheat
        if n == MAX_LEVELS:
            if rh.p_Pa is not None and not math.isclose(
                rh.p_Pa, inputs.levels[1].p_Pa, rel_tol=1e-9
            ):
                raise ValueError(
                    "Recalentador: con tres niveles el recalentamiento es a la presión de media "
                    f"({_bar(inputs.levels[1].p_Pa)}): el vapor de media se suma al "
                    "recalentamiento frío."
                )
        else:
            if rh.p_Pa is None:
                raise ValueError("Recalentador: falta la presión de recalentamiento.")
            bottom = low.p_Pa if n > 1 else steam.p_condenser_Pa
            where = "la de baja" if n > 1 else "la del condensador"
            if not bottom < rh.p_Pa < high.p_Pa:
                raise ValueError(
                    f"Recalentador: la presión de recalentamiento ({_bar(rh.p_Pa)}) tiene que "
                    f"estar entre {where} ({_bar(bottom)}) y la de alta ({_bar(high.p_Pa)})."
                )
            events.append(rh.p_Pa)
    p_da = steam.deaerator_p_Pa
    if p_da is not None:
        if not steam.p_condenser_Pa < p_da < low.p_Pa:
            raise ValueError(
                f"Ciclo de vapor: el desaireador ({_bar(p_da)}) tiene que trabajar entre la "
                f"presión del condensador y la del nivel de menor presión ({_bar(low.p_Pa)})."
            )
        T_eco = _T_sat(low.p_Pa) - low.approach_K
        if _T_sat(p_da) >= T_eco:
            raise ValueError(
                f"Ciclo de vapor: el desaireador a {_bar(p_da)} entrega el agua a "
                f"{_degC(_T_sat(p_da))}, más caliente que la salida del economizador de "
                f"{'baja' if n > 1 else 'la caldera'} (T_sat − approach = {_degC(T_eco)}): bajá "
                "la presión del desaireador."
            )
        if any(math.isclose(p_da, p, rel_tol=1e-6) for p in events):
            raise ValueError(
                "Ciclo de vapor: el desaireador no puede estar justo a la presión de una "
                "admisión o del recalentamiento; corré un poco su presión."
            )


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _state_ph(p_Pa: float, h: float) -> FluidState:
    return fluid_state_from_pair(Water, "PH", p=p_Pa, h=h)


def _pump(inlet: FluidState, p_out_Pa: float, eta: float) -> FluidState:
    """Bomba con rendimiento η desde ``inlet`` hasta ``p_out_Pa`` (vademecum §13)."""
    out_s = fluid_state_from_pair(Water, "PS", p=p_out_Pa, s=inlet.s_J_per_kg_K)
    if eta == 1.0:
        return out_s
    return _state_ph(p_out_Pa, inlet.h_J_per_kg + (out_s.h_J_per_kg - inlet.h_J_per_kg) / eta)


def _moisture(p_Pa: float, h: float) -> float:
    """Humedad y = 1 − x (0 si el vapor está seco)."""
    h_f = float(PropsSI("H", "P", p_Pa, "Q", 0, Water))
    h_g = float(PropsSI("H", "P", p_Pa, "Q", 1, Water))
    return min(1.0, max(0.0, (h_g - h) / (h_g - h_f)))


def _expand(
    inlet: FluidState, p_out_Pa: float, eta: float, alpha: float
) -> tuple[FluidState, FluidState, WetExpansion | None]:
    """Expansión de ``inlet`` a ``p_out_Pa``: (salida real, isoentrópica, parte húmeda).

    Con la regla de Baumann (α > 0) y una salida húmeda, la línea se parte donde
    cruza x = 1: la parte seca va con η y la húmeda con η·(1 − α·ȳ).
    """
    h_in, s_in = inlet.h_J_per_kg, inlet.s_J_per_kg_K
    out_s = fluid_state_from_pair(Water, "PS", p=p_out_Pa, s=s_in)
    h_out = h_in - eta * (h_in - out_s.h_J_per_kg)
    if alpha == 0.0 or _moisture(p_out_Pa, h_out) == 0.0:
        return _state_ph(p_out_Pa, h_out), out_s, None
    y_in = _moisture(inlet.P_Pa, h_in) if inlet.P_Pa < fluid_limits(Water).P_crit_Pa else 0.0
    if y_in > 0.0:
        p_x, h_x = inlet.P_Pa, h_in
    else:

        def line(p: float) -> float:
            h_s = float(PropsSI("H", "P", p, "S", s_in, Water))
            return h_in - eta * (h_in - h_s) - float(PropsSI("H", "P", p, "Q", 1, Water))

        p_top = min(inlet.P_Pa, fluid_limits(Water).P_crit_Pa * 0.999)
        p_x = float(brentq(line, p_out_Pa, p_top, xtol=1e-6, rtol=1e-12))
        h_x = float(PropsSI("H", "P", p_x, "Q", 1, Water))
    s_x = float(PropsSI("S", "P", p_x, "H", h_x, Water))
    h_s2 = float(PropsSI("H", "P", p_out_Pa, "S", s_x, Water))
    y_out = _moisture(p_out_Pa, h_out)
    eta_w = eta
    for _ in range(100):
        eta_w = eta * (1.0 - alpha * (y_in + y_out) / 2.0)
        h2 = h_x - eta_w * (h_x - h_s2)
        y_new = _moisture(p_out_Pa, h2)
        if abs(y_new - y_out) < 1e-13:
            y_out = y_new
            break
        y_out = y_new
    h2 = h_x - eta_w * (h_x - h_s2)
    wet = WetExpansion(p_x, h_x, y_in, y_out, eta_w)
    return _state_ph(p_out_Pa, h2), out_s, wet


def _on_line(inlet: FluidState, p_Pa: float, eta: float, wet: WetExpansion | None) -> FluidState:
    """Un punto de la línea de expansión de una turbina (una extracción), a ``p_Pa``."""
    if wet is None or p_Pa >= wet.p_x_Pa:
        h_s = float(PropsSI("H", "P", p_Pa, "S", inlet.s_J_per_kg_K, Water))
        return _state_ph(p_Pa, inlet.h_J_per_kg - eta * (inlet.h_J_per_kg - h_s))
    s_x = float(PropsSI("S", "P", wet.p_x_Pa, "H", wet.h_x_J_per_kg, Water))
    h_s = float(PropsSI("H", "P", p_Pa, "S", s_x, Water))
    return _state_ph(p_Pa, wet.h_x_J_per_kg - wet.eta_wet * (wet.h_x_J_per_kg - h_s))


def _staged(stage: str, exc: ValueError) -> ValueError:
    text = str(exc)
    if text.startswith("Recalentador:"):
        return ValueError(text)
    return ValueError(f"{stage}: {text}")


def _turbine_names(count: int) -> list[str]:
    if count == 1:
        return ["turbina"]
    if count == 2:
        return ["turbina de alta", "turbina de baja"]
    return ["turbina de alta", "turbina de media", "turbina de baja"]


def _level_steam(level: PressureLevel) -> FluidState:
    """El vapor que entrega un nivel: sobrecalentado o saturado."""
    if level.T_steam_K is None:
        return fluid_state_from_pair(Water, "PX", p=level.p_Pa, x=1.0)
    return fluid_state_from_pair(Water, "TP", t=level.T_steam_K, p=level.p_Pa)


def solve_combined_multi(inputs: MultiCombinedInputs) -> MultiCombinedResult:
    """Resuelve y acopla la turbina de gas, la HRSG de varias presiones y el ciclo de vapor.

    1. La turbina de gas da el escape.
    2. La línea de alimentación (bombas y desaireador) da el agua que entra a la
       HRSG, y la turbina de alta el recalentamiento frío: no dependen de los
       caudales.
    3. La HRSG da los caudales de cada nivel (con el recalentador).
    4. La turbina de vapor, con las admisiones, da las potencias.

    Raises
    ------
    ValueError
        Con el mensaje de la parte que no tiene sentido físico.
    """
    validate_multi_combined_inputs(inputs)
    if inputs.W_net_W is not None:
        base = solve_combined_multi(replace(inputs, W_net_W=None))
        gt = inputs.gas_turbine
        factor = inputs.W_net_W / base.W_net_W
        scaled = replace(gt, m_air_kg_s=gt.m_air_kg_s * factor)
        return replace(
            solve_combined_multi(replace(inputs, gas_turbine=scaled, W_net_W=None)),
            inputs=inputs,
        )
    try:
        gt = solve_brayton(inputs.gas_turbine)
    except ValueError as exc:
        raise _staged("Turbina de gas", exc) from exc
    steam = inputs.steam
    levels = inputs.levels
    eta_T, eta_P, alpha = steam.eta_turbine, steam.eta_pump, inputs.baumann_alpha
    states: list[FluidState] = []
    labels: list[str] = []

    def add(state: FluidState, label: str) -> int:
        states.append(state)
        labels.append(label)
        return len(states) - 1

    # 1. Línea de alimentación (no depende de los caudales).
    p_low = levels[-1].p_Pa
    try:
        cond = fluid_state_from_pair(Water, "PX", p=steam.p_condenser_Pa, x=0.0)
        add(cond, "salida del condensador (líquido saturado)")
        p_da = steam.deaerator_p_Pa
        deaerator = None
        if p_da is None:
            feedwater = add(
                _pump(cond, p_low, eta_P),
                "salida de la bomba de condensado (agua de alimentación de la HRSG)",
            )
            cp_out = feedwater
        else:
            cp_out = add(_pump(cond, p_da, eta_P), "salida de la bomba de condensado")
            da = fluid_state_from_pair(Water, "PX", p=p_da, x=0.0)
            deaerator = add(da, "salida del desaireador (líquido saturado)")
            feedwater = add(
                _pump(da, p_low, eta_P),
                "salida de la bomba de alimentación (agua de alimentación de la HRSG)",
            )
        # 2. Recalentamiento frío: la turbina de alta hasta la presión de recalentamiento.
        hp_steam = _level_steam(levels[0])
        p_rh = inputs.p_reheat_Pa
        reheater = None
        if p_rh is not None and inputs.reheat is not None:
            cold, _, _ = _expand(hp_steam, p_rh, eta_T, alpha)
            reheater = Reheater(
                p_rh, inputs.reheat.T_K, cold.h_J_per_kg, inputs.reheat_joins_middle
            )
    except ValueError as exc:
        raise _staged("Ciclo de vapor", exc) from exc
    # 3. HRSG.
    hrsg_inputs = MultiHRSGInputs(
        T_gas_in_K=gt.T_exhaust_K,
        m_gas_kg_s=gt.m_gas_kg_s,
        gas=gt.gas,
        levels=levels,
        T_feedwater_K=states[feedwater].T_K,
        T_ref_K=inputs.gas_turbine.T_amb_K,
        reheat=reheater,
        eta_pump=eta_P,
    )
    try:
        hrsg = solve_multi_hrsg(hrsg_inputs)
    except ValueError as exc:
        raise _staged("Caldera de recuperación", exc) from exc
    # 4. Turbina de vapor con las admisiones.
    try:
        steam_cycle = _steam_path(
            inputs, hrsg, states, labels, add, feedwater, cp_out, deaerator, p_rh
        )
    except ValueError as exc:
        raise _staged("Ciclo de vapor", exc) from exc
    return MultiCombinedResult(inputs=inputs, gas_turbine=gt, hrsg=hrsg, steam=steam_cycle)


def _steam_path(  # noqa: PLR0913, PLR0915 — el recorrido del vapor, paso a paso
    inputs: MultiCombinedInputs,
    hrsg: MultiHRSGResult,
    states: list[FluidState],
    labels: list[str],
    add: Any,
    feedwater: int,
    cp_out: int,
    deaerator: int | None,
    p_rh: float | None,
) -> MultiSteamCycle:
    steam = inputs.steam
    levels = inputs.levels
    n = len(levels)
    names = inputs.names
    eta_T, alpha = steam.eta_turbine, inputs.baumann_alpha
    m = [lv.m_steam_kg_s for lv in hrsg.levels]
    m_total = sum(m)
    # Eventos de la turbina, de mayor a menor presión: recalentador, admisiones, condensador.
    events: list[tuple[str, float, int]] = []
    if p_rh is not None:
        events.append(("rh", p_rh, -1))
    for k in range(1, n):
        if not (inputs.reheat_joins_middle and k == 1):
            events.append(("adm", levels[k].p_Pa, k))
    events.append(("cond", steam.p_condenser_Pa, -1))
    events.sort(key=lambda e: -e[1])
    turbine_names = _turbine_names(len(events))
    first = "turbina" if len(events) == 1 else "turbina de alta"
    entering = "entrada a la " + first
    hp_label = f"vapor de la HRSG ({entering})" if n == 1 else f"vapor de alta ({entering})"
    level_states = [add(hrsg.levels[0].water[-1], hp_label)]
    cur, flow = level_states[0], m[0]
    turbines: list[TurbineSection] = []
    admissions: list[Admission] = []
    cold_reheat = reheat_inlet = hot_reheat = None
    p_da = steam.deaerator_p_Pa
    m_ext = 0.0
    for t_index, (kind, p_stop, k) in enumerate(events):
        name = turbine_names[t_index]
        inlet = states[cur]
        out, out_s, wet = _expand(inlet, p_stop, eta_T, alpha)
        extraction = None
        m_ext_here = 0.0
        if p_da is not None and p_stop < p_da < inlet.P_Pa:
            ext = _on_line(inlet, p_da, eta_T, wet)
            da_state, cp_state = states[deaerator], states[cp_out]
            y = (da_state.h_J_per_kg - cp_state.h_J_per_kg) / (ext.h_J_per_kg - cp_state.h_J_per_kg)
            m_ext_here = y * m_total
            if not 0.0 < m_ext_here < flow:
                raise ValueError(
                    f"la {name} no tiene vapor suficiente para el desaireador "
                    f"({m_ext_here:.4g} kg/s de {flow:.4g} kg/s): subí la presión del desaireador."
                )
            extraction = add(ext, f"extracción de la {name} para el desaireador")
            m_ext = m_ext_here
        if kind == "cond":
            out_label = f"salida de la {name} (al condensador)"
        elif kind == "rh":
            out_label = f"salida de la {name} (recalentamiento frío)"
        else:
            out_label = f"salida de la {name}"
        outlet = add(out, out_label)
        if extraction is not None:
            h_in, h_ext, h_out = inlet.h_J_per_kg, states[extraction].h_J_per_kg, out.h_J_per_kg
            W = flow * (h_in - h_ext) + (flow - m_ext_here) * (h_ext - h_out)
        else:
            W = flow * (inlet.h_J_per_kg - out.h_J_per_kg)
        turbines.append(
            TurbineSection(name, cur, outlet, flow, out_s, W, extraction, m_ext_here, wet)
        )
        flow -= m_ext_here
        cur = outlet
        if kind == "rh":
            cold_reheat = outlet
            rh = hrsg.reheat
            assert rh is not None
            if inputs.reheat_joins_middle:
                mid = add(hrsg.levels[1].water[-1], f"vapor de {names[1]}")
                level_states.append(mid)
                reheat_inlet = add(rh.inlet, "mezcla (entrada al recalentador)")
                admissions.append(Admission(1, outlet, mid, reheat_inlet, flow, m[1]))
                flow += m[1]
            else:
                reheat_inlet = outlet
            nxt = turbine_names[t_index + 1]
            hot_reheat = add(rh.hot, f"recalentamiento caliente (entrada a la {nxt})")
            cur = hot_reheat
        elif kind == "adm":
            st = add(hrsg.levels[k].water[-1], f"vapor de {names[k]}")
            level_states.append(st)
            h_mix = (flow * states[outlet].h_J_per_kg + m[k] * states[st].h_J_per_kg) / (
                flow + m[k]
            )
            nxt = turbine_names[t_index + 1]
            mix = add(_state_ph(levels[k].p_Pa, h_mix), f"mezcla (entrada a la {nxt})")
            admissions.append(Admission(k, outlet, st, mix, flow, m[k]))
            flow += m[k]
            cur = mix
    exhaust = cur
    cond = states[0]
    if deaerator is None:
        W_cp = m_total * (states[feedwater].h_J_per_kg - cond.h_J_per_kg)
        W_fp = 0.0
    else:
        W_cp = (m_total - m_ext) * (states[cp_out].h_J_per_kg - cond.h_J_per_kg)
        W_fp = m_total * (states[feedwater].h_J_per_kg - states[deaerator].h_J_per_kg)
    return MultiSteamCycle(
        states=tuple(states),
        labels=tuple(labels),
        turbines=tuple(turbines),
        admissions=tuple(admissions),
        level_states=tuple(level_states),
        feedwater=feedwater,
        condensate_pump_out=cp_out,
        deaerator=deaerator,
        cold_reheat=cold_reheat,
        reheat_inlet=reheat_inlet,
        hot_reheat=hot_reheat,
        exhaust=exhaust,
        m_total_kg_s=float(m_total),
        m_extraction_kg_s=float(m_ext),
        W_condensate_pump_W=float(W_cp),
        W_feed_pump_W=float(W_fp),
    )


# ---------------------------------------------------------------------
# Exergía del ciclo de fondo
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BottomingExergy:
    """Balance de exergía del ciclo de fondo, desde los gases de escape (W).

    Ẋ_gases = Ẇ_TV + Σ Ẋ_dest + Ẋ_cond + Ẋ_chim: la exergía de los gases que
    entran a la HRSG termina como trabajo neto del ciclo de vapor, se destruye
    (T₀·S_gen) en cada sección de la caldera, en las turbinas, en las mezclas,
    en el desaireador y en las bombas, se la lleva el agua de enfriamiento del
    condensador o sale por la chimenea (Cengel cap. 8).
    """

    T0_K: float
    X_gas_in_W: float
    X_stack_W: float
    W_net_W: float
    X_condenser_W: float
    destroyed_W: tuple[tuple[str, float], ...]

    @property
    def X_destroyed_W(self) -> float:
        return sum(x for _, x in self.destroyed_W)

    @property
    def efficiency(self) -> float:
        """Parte de la exergía de los gases de escape que termina como trabajo."""
        return self.W_net_W / self.X_gas_in_W

    @property
    def residual_W(self) -> float:
        """Lo que no cierra del balance (tiene que dar ~0)."""
        return (
            self.X_gas_in_W
            - self.W_net_W
            - self.X_destroyed_W
            - self.X_condenser_W
            - self.X_stack_W
        )


def bottoming_exergy(result: MultiCombinedResult) -> BottomingExergy:
    """Exergía destruida por componente del ciclo de fondo (Kehlhofer et al., cap. 2 y 5)."""
    hx = hrsg_exergy(result.hrsg)
    T0 = hx.T0_K
    cyc = result.steam
    st = cyc.states

    def s(i: int) -> float:
        return st[i].s_J_per_kg_K

    destroyed: list[tuple[str, float]] = list(hx.destroyed_W)
    for t in cyc.turbines:
        if t.extraction is not None:
            S = t.m_extraction_kg_s * s(t.extraction) + (t.m_in_kg_s - t.m_extraction_kg_s) * s(
                t.outlet
            )
        else:
            S = t.m_in_kg_s * s(t.outlet)
        destroyed.append((t.name, T0 * (S - t.m_in_kg_s * s(t.inlet))))
    names = result.inputs.names
    for a in cyc.admissions:
        S = (
            a.m_out_kg_s * s(a.outlet)
            - a.m_turbine_kg_s * s(a.turbine)
            - a.m_steam_kg_s * s(a.steam)
        )
        where = "recalentamiento frío" if cyc.reheat_inlet == a.outlet else "turbina"
        destroyed.append((f"mezcla del vapor de {names[a.level]} ({where})", T0 * S))
    m_cond = cyc.m_condenser_kg_s
    if cyc.deaerator is not None:
        ext = cyc.extraction
        assert ext is not None
        S = (
            cyc.m_total_kg_s * s(cyc.deaerator)
            - cyc.m_extraction_kg_s * s(ext)
            - m_cond * s(cyc.condensate_pump_out)
        )
        destroyed.append(("desaireador", T0 * S))
    # Bombas: condensado, alimentación y las de la HRSG entre niveles.
    S_pumps = m_cond * (s(cyc.condensate_pump_out) - s(0))
    if cyc.deaerator is not None:
        S_pumps += cyc.m_total_kg_s * (s(cyc.feedwater) - s(cyc.deaerator))
    lv = result.hrsg.levels
    for i in range(len(lv) - 1):
        S_pumps += lv[i].m_water_kg_s * (
            lv[i].water[0].s_J_per_kg_K - lv[i + 1].water[2].s_J_per_kg_K
        )
    destroyed.append(("bombas", T0 * S_pumps))
    exh, cond = st[cyc.exhaust], st[0]
    X_cond = m_cond * (
        (exh.h_J_per_kg - cond.h_J_per_kg) - T0 * (exh.s_J_per_kg_K - cond.s_J_per_kg_K)
    )
    return BottomingExergy(
        T0, hx.X_gas_in_W, hx.X_stack_W, result.W_steam_turbine_W, X_cond, tuple(destroyed)
    )


# ---------------------------------------------------------------------
# Comparación de configuraciones
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigurationRow:
    """Una configuración de la comparación: con η_T constante y con la regla de Baumann."""

    label: str
    inputs: MultiCombinedInputs
    result: MultiCombinedResult | None
    result_baumann: MultiCombinedResult | None
    error: str = ""


def configuration_comparison(inputs: MultiCombinedInputs) -> list[ConfigurationRow]:
    """La misma turbina de gas con 1, 2 y 3 presiones, con y sin recalentamiento.

    Usa los niveles de los datos (alta; alta y baja; los tres) y el
    recalentamiento de los datos (con tres niveles, a la presión de media). Si
    los datos no tienen recalentamiento y tienen tres niveles, se recalienta a la
    temperatura del vapor de alta. Cada configuración se resuelve con η_T
    constante (α = 0) y con la regla de Baumann (α = 1).
    """
    levels = inputs.levels
    reheat = inputs.reheat
    if reheat is None and len(levels) == MAX_LEVELS and levels[0].T_steam_K is not None:
        reheat = ReheatSpec(levels[0].T_steam_K)
    subsets: list[tuple[str, tuple[PressureLevel, ...]]] = [("1 presión", levels[:1])]
    if len(levels) >= 2:
        subsets.append(("2 presiones", (levels[0], levels[-1])))
    if len(levels) == MAX_LEVELS:
        subsets.append(("3 presiones", levels))
    rows: list[ConfigurationRow] = []
    for label, lv in subsets:
        options: list[tuple[str, ReheatSpec | None]] = [(label, None)]
        if reheat is not None:
            p = reheat.p_Pa if reheat.p_Pa is not None else levels[1].p_Pa
            spec = ReheatSpec(reheat.T_K, None if len(lv) == MAX_LEVELS else p)
            options.append((f"{label} + RH", spec))
        for text, spec in options:
            case = replace(inputs, levels=lv, reheat=spec, W_net_W=None)
            try:
                res = solve_combined_multi(replace(case, baumann_alpha=0.0))
                res_b = solve_combined_multi(replace(case, baumann_alpha=1.0))
            except ValueError as exc:
                rows.append(ConfigurationRow(text, case, None, None, str(exc)))
                continue
            rows.append(ConfigurationRow(text, case, res, res_b))
    return rows


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def combined_multi_notes(result: MultiCombinedResult) -> list[str]:
    """Observaciones didácticas del ciclo combinado de varias presiones (markdown)."""
    inputs = result.inputs
    cyc = result.steam
    notes: list[str] = []
    gain = result.eta_th - result.eta_gas_turbine
    notes.append(
        f"El ciclo de vapor sube el rendimiento de {result.eta_gas_turbine * 100:.1f} % (la "
        f"turbina de gas sola) a {result.eta_th * 100:.1f} %: {gain * 100:.1f} puntos más con el "
        "mismo combustible (Cengel §10-9)."
    )
    x = cyc.x_exhaust
    if x < X_EXHAUST_MIN:
        fix = (
            "recalentar, subir la temperatura del vapor o bajar la presión de alta"
            if inputs.reheat is None
            else "subir la temperatura de recalentamiento o bajar su presión"
        )
        notes.append(
            f"El vapor sale de la turbina con {100 * (1 - x):.1f} % de humedad (x = {x:.3f}): más "
            "del 10–12 % erosiona los álabes de las últimas etapas y baja su rendimiento. Para "
            f"secarlo: {fix} (Cengel §10-5)."
        )
    if inputs.reheat is not None and len(inputs.levels) == 1:
        notes.append(
            "Con una sola presión el recalentamiento le saca calor de alta temperatura a la "
            "producción de vapor: hay menos vapor y la chimenea sale más caliente, así que el "
            "rendimiento casi no mejora (o empeora). Lo que gana es un vapor más seco a la salida "
            "de la turbina. Con dos o tres presiones los niveles de baja recuperan ese calor."
        )
    if inputs.baumann_alpha > 0.0:
        wet = [t for t in cyc.turbines if t.wet is not None]
        for t in wet:
            assert t.wet is not None
            notes.append(
                f"Regla de Baumann en la {t.name}: desde {_bar(t.wet.p_x_Pa)} la expansión es "
                f"húmeda y su rendimiento baja de {inputs.steam.eta_turbine:.3g} a "
                f"{t.wet.eta_wet:.3g} (humedad media {(t.wet.y_in + t.wet.y_out) / 2 * 100:.2g} %)."
            )
    if cyc.deaerator is not None:
        notes.append(
            f"El desaireador calienta el agua de alimentación a "
            f"{_degC(cyc.states[cyc.deaerator].T_K)} con vapor de la turbina: la HRSG recibe el "
            "agua más caliente y la chimenea sale más caliente. Con varias presiones, el domo de "
            "baja puede hacer de desaireador y el economizador de baja aprovecha ese calor."
        )
    notes += brayton_notes(result.gas_turbine)
    notes += multi_hrsg_notes(result.hrsg)
    return notes


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------

_C = 273.15
_GT_MODERN = BraytonInputs(
    17.0, 1673.15, 0.90, 0.91, dp_combustor=0.03, air=AIR_DRY, m_air_kg_s=650.0
)
_GT_TYPICAL = BraytonInputs(
    15.0, 1523.15, 0.88, 0.90, dp_combustor=0.03, air=AIR_DRY, m_air_kg_s=500.0
)
_EX_3PRH = "Tres presiones con recalentamiento: turbina de gas moderna, 120/25/4 bar y 565/565 °C"
_EX_2P = "Dos presiones: turbina de gas típica, 80 bar y 540 °C + 6 bar"
_EX_2PRH = "Dos presiones con recalentamiento: 100 bar, recalentamiento a 25 bar y 565 °C + 5 bar"
_EX_1PRH = "Una presión con recalentamiento (para ver qué hace el recalentamiento solo)"
_EX_3P = "Tres presiones sin recalentamiento: turbina de gas moderna, 120/25/4 bar y 565 °C"
_EX_2P_MODERN = "Dos presiones: turbina de gas moderna, 100 bar y 565 °C + 5 bar"
_EX_2PRH_TYPICAL = (
    "Dos presiones con recalentamiento: turbina de gas típica, 100 bar, recalentamiento a "
    "25 bar y 540 °C + 6 bar"
)

#: Ejemplos precargados.
COMBINED_MULTI_EXAMPLES: dict[str, MultiCombinedInputs] = {
    _EX_3PRH: MultiCombinedInputs(
        gas_turbine=_GT_MODERN,
        levels=(
            PressureLevel(120e5, 565.0 + _C, 8.0, 5.0),
            PressureLevel(25e5, None, 8.0, 5.0),
            PressureLevel(4e5, None, 8.0, 5.0),
        ),
        steam=SteamCycle(0.06e5, 0.90, 0.80),
        reheat=ReheatSpec(565.0 + _C),
    ),
    _EX_2P: MultiCombinedInputs(
        gas_turbine=_GT_TYPICAL,
        levels=(PressureLevel(80e5, 540.0 + _C, 10.0, 5.0), PressureLevel(6e5, None, 10.0, 5.0)),
        steam=SteamCycle(0.08e5, 0.88, 0.80),
    ),
    _EX_2PRH: MultiCombinedInputs(
        gas_turbine=_GT_MODERN,
        levels=(PressureLevel(100e5, 565.0 + _C, 8.0, 5.0), PressureLevel(5e5, None, 8.0, 5.0)),
        steam=SteamCycle(0.06e5, 0.90, 0.80),
        reheat=ReheatSpec(565.0 + _C, 25e5),
    ),
    _EX_1PRH: MultiCombinedInputs(
        gas_turbine=_GT_TYPICAL,
        levels=(PressureLevel(100e5, 540.0 + _C, 10.0, 5.0),),
        steam=SteamCycle(0.08e5, 0.88, 0.80, 1.5e5),
        reheat=ReheatSpec(540.0 + _C, 25e5),
    ),
    _EX_3P: MultiCombinedInputs(
        gas_turbine=_GT_MODERN,
        levels=(
            PressureLevel(120e5, 565.0 + _C, 8.0, 5.0),
            PressureLevel(25e5, None, 8.0, 5.0),
            PressureLevel(4e5, None, 8.0, 5.0),
        ),
        steam=SteamCycle(0.06e5, 0.90, 0.80),
    ),
    _EX_2P_MODERN: MultiCombinedInputs(
        gas_turbine=_GT_MODERN,
        levels=(PressureLevel(100e5, 565.0 + _C, 8.0, 5.0), PressureLevel(5e5, None, 8.0, 5.0)),
        steam=SteamCycle(0.06e5, 0.90, 0.80),
    ),
    _EX_2PRH_TYPICAL: MultiCombinedInputs(
        gas_turbine=_GT_TYPICAL,
        levels=(PressureLevel(100e5, 540.0 + _C, 10.0, 5.0), PressureLevel(6e5, None, 10.0, 5.0)),
        steam=SteamCycle(0.08e5, 0.88, 0.80),
        reheat=ReheatSpec(540.0 + _C, 25e5),
    ),
}

#: Aclaraciones de cada ejemplo para mostrar en la página.
COMBINED_MULTI_EXAMPLE_NOTES: dict[str, str] = {
    _EX_3PRH: (
        "La turbina de gas moderna de la Fase 3.4 (r_p = 17, TIT = 1400 °C, escape a 675 °C) con "
        "la HRSG de tres presiones y recalentamiento de las centrales actuales: alta a 120 bar y "
        "565 °C, media a 25 bar (su vapor se suma al recalentamiento) y baja a 4 bar, "
        "recalentamiento a 565 °C, pinch 8 K. Sin desaireador: el domo de baja desgasifica el "
        "agua y el condensado entra directo al economizador de baja."
    ),
    _EX_2P: (
        "La turbina de gas típica de la Fase 3.4 (escape a 609 °C) con dos presiones: alta a "
        "80 bar y 540 °C y baja a 6 bar, sin recalentamiento. Compará con la de una presión en "
        "la tabla de configuraciones."
    ),
    _EX_2PRH: (
        "La turbina moderna con dos presiones y recalentamiento: el vapor de alta (100 bar y "
        "565 °C) vuelve de la turbina a 25 bar, se recalienta a 565 °C y el vapor de baja (5 bar) "
        "entra a la turbina de baja."
    ),
    _EX_1PRH: (
        "Una sola presión (100 bar y 540 °C) con recalentamiento a 25 bar y 540 °C y desaireador "
        "a 1,5 bar. Mirá la tabla de configuraciones: el recalentamiento seca el vapor, pero con "
        "una sola presión sube la chimenea y el rendimiento casi no cambia."
    ),
    _EX_3P: (
        "La HRSG de tres presiones de la central moderna, pero sin recalentar: el vapor de "
        "120 bar termina la expansión con mucha humedad (más del 16 %). Prendé el "
        "recalentamiento (o la regla de Baumann) y compará."
    ),
    _EX_2P_MODERN: (
        "La turbina moderna con dos presiones sin recalentamiento: alta a 100 bar y 565 °C y baja "
        "a 5 bar. Con el recalentamiento a 25 bar es el ejemplo de dos presiones con "
        "recalentamiento."
    ),
    _EX_2PRH_TYPICAL: (
        "La turbina típica con dos presiones y recalentamiento: alta a 100 bar y 540 °C, "
        "recalentamiento a 25 bar y 540 °C y baja a 6 bar, pinch 10 K."
    ),
}


# ---------------------------------------------------------------------
# Barridos
# ---------------------------------------------------------------------

MultiCombinedSweepParameter = Literal[
    "p_high",
    "p_middle",
    "p_low",
    "p_reheat",
    "T_reheat",
    "pinch",
    "pressure_ratio",
    "T_turbine_in",
]


@dataclass(frozen=True)
class MultiCombinedSweepPoint:
    """Un punto del barrido: valor del parámetro (SI) y resultados del ciclo combinado."""

    value_si: float
    eta: float
    eta_gas_turbine: float
    W_steam_W: float
    x_exhaust: float
    T_stack_K: float
    m_steam_kg_s: tuple[float, ...]


def _with(
    inputs: MultiCombinedInputs, parameter: MultiCombinedSweepParameter, value: float
) -> MultiCombinedInputs:
    levels = list(inputs.levels)
    gt = inputs.gas_turbine
    if parameter == "p_high":
        levels[0] = replace(levels[0], p_Pa=value)
    elif parameter == "p_middle":
        levels[1] = replace(levels[1], p_Pa=value)
    elif parameter == "p_low":
        levels[-1] = replace(levels[-1], p_Pa=value)
    elif parameter == "pinch":
        levels = [replace(lv, pinch_K=value) for lv in levels]
    elif parameter == "p_reheat":
        assert inputs.reheat is not None
        return replace(inputs, reheat=replace(inputs.reheat, p_Pa=value))
    elif parameter == "T_reheat":
        assert inputs.reheat is not None
        return replace(inputs, reheat=replace(inputs.reheat, T_K=value))
    elif parameter == "pressure_ratio":
        return replace(inputs, gas_turbine=replace(gt, pressure_ratio=value))
    elif parameter == "T_turbine_in":
        return replace(inputs, gas_turbine=replace(gt, T_turbine_in_K=value))
    else:
        raise ValueError(f"Parámetro de barrido desconocido: {parameter!r}.")
    return replace(inputs, levels=tuple(levels))


def _p_sat(T_K: float) -> float:
    return float(PropsSI("P", "T", T_K, "Q", 0, Water))


def default_multi_combined_sweep_values(
    inputs: MultiCombinedInputs, parameter: MultiCombinedSweepParameter, n: int = 9
) -> list[float]:
    """Valores razonables del parámetro alrededor del ciclo dado (SI)."""
    levels = inputs.levels
    gt = inputs.gas_turbine
    if parameter == "p_high":
        below = levels[1].p_Pa if len(levels) > 1 else 2e5
        if inputs.reheat is not None and inputs.reheat.p_Pa is not None:
            below = max(below, inputs.reheat.p_Pa)
        lo = max(1.6 * below, 20e5 if len(levels) > 1 else 10e5)
        hi = 160e5
        T_crit = fluid_limits(Water).T_crit_K
        if levels[0].T_steam_K is not None and levels[0].T_steam_K - 10.0 < T_crit:
            hi = min(hi, _p_sat(levels[0].T_steam_K - 10.0))
        lo, hi = min(lo, levels[0].p_Pa), max(hi, levels[0].p_Pa)
        return [float(v) for v in np.geomspace(lo, hi, n)]
    if parameter == "p_middle":
        lo, hi = 1.6 * levels[-1].p_Pa, levels[0].p_Pa / 1.6
        lo, hi = min(lo, levels[1].p_Pa), max(hi, levels[1].p_Pa)
        return [float(v) for v in np.geomspace(lo, hi, n)]
    if parameter == "p_low":
        lo = max(1.5e5, 3.0 * inputs.steam.p_condenser_Pa)
        if inputs.steam.deaerator_p_Pa is not None:
            lo = max(lo, 1.5 * inputs.steam.deaerator_p_Pa)
        above = levels[-2].p_Pa if len(levels) > 1 else levels[0].p_Pa
        if inputs.reheat is not None and inputs.reheat.p_Pa is not None and len(levels) < 3:
            above = min(above, inputs.reheat.p_Pa)
        hi = min(above / 1.6, 15e5)
        lo, hi = min(lo, levels[-1].p_Pa), max(hi, levels[-1].p_Pa)
        return [float(v) for v in np.geomspace(lo, hi, n)]
    if parameter == "p_reheat":
        assert inputs.reheat is not None and inputs.reheat.p_Pa is not None
        lo = 1.5 * (levels[-1].p_Pa if len(levels) > 1 else 2e5)
        hi = levels[0].p_Pa / 1.5
        p = inputs.reheat.p_Pa
        lo, hi = min(lo, p), max(hi, p)
        return [float(v) for v in np.geomspace(lo, hi, n)]
    if parameter == "T_reheat":
        assert inputs.reheat is not None
        T = inputs.reheat.T_K
        T_max = solve_brayton(gt).T_exhaust_K - SH_HOT_END_K
        return [
            float(v) for v in np.linspace(min(T - 150.0, T_max - 160.0), min(T_max, T + 50.0), n)
        ]
    if parameter == "pinch":
        return [float(v) for v in np.linspace(3.0, 30.0, n)]
    if parameter == "pressure_ratio":
        return [float(v) for v in np.geomspace(8.0, 40.0, n)]
    if parameter == "T_turbine_in":
        hi = min(T_GAS_MAX_K - 50.0, max(gt.T_turbine_in_K + 200.0, 1373.15))
        lo = min(gt.T_turbine_in_K - 200.0, hi - 300.0)
        return [float(v) for v in np.linspace(lo, hi, n)]
    raise ValueError(f"Parámetro de barrido desconocido: {parameter!r}.")


def _limit_steam_temperatures(inputs: MultiCombinedInputs) -> MultiCombinedInputs:
    """Vapor de alta y recalentado como mucho hasta T_escape − 25 K (barridos de la TG)."""
    T_max = solve_brayton(inputs.gas_turbine).T_exhaust_K - SH_HOT_END_K
    levels = list(inputs.levels)
    if levels[0].T_steam_K is not None:
        levels[0] = replace(levels[0], T_steam_K=min(levels[0].T_steam_K, T_max))
    reheat = inputs.reheat
    if reheat is not None:
        reheat = replace(reheat, T_K=min(reheat.T_K, T_max))
    return replace(inputs, levels=tuple(levels), reheat=reheat)


def combined_multi_sweep(
    inputs: MultiCombinedInputs,
    parameter: MultiCombinedSweepParameter,
    values: Sequence[float],
) -> list[MultiCombinedSweepPoint]:
    """Resuelve el ciclo combinado para cada valor; omite los que no tienen sentido físico."""
    points: list[MultiCombinedSweepPoint] = []
    for value in values:
        try:
            case = replace(_with(inputs, parameter, float(value)), W_net_W=None)
            if parameter in ("pressure_ratio", "T_turbine_in"):
                case = _limit_steam_temperatures(case)
            r = solve_combined_multi(case)
        except ValueError:
            continue
        points.append(
            MultiCombinedSweepPoint(
                float(value),
                r.eta_th,
                r.eta_gas_turbine,
                r.W_steam_turbine_W,
                r.steam.x_exhaust,
                r.hrsg.T_stack_K,
                tuple(lv.m_steam_kg_s for lv in r.hrsg.levels),
            )
        )
    return points


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _value(value_si: float, kind: QuantityKind, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def combined_multi_to_dict(result: MultiCombinedResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``)."""
    inputs = result.inputs
    cyc = result.steam
    ex = bottoming_exergy(result)
    steam = inputs.steam
    return {
        "equipo": f"ciclo combinado gas–vapor de {inputs.configuration}",
        "sistema_de_unidades": system,
        "turbina_de_gas": gas_turbine_to_dict(result.gas_turbine, system),
        "hrsg": multi_hrsg_to_dict(result.hrsg, system),
        "ciclo_de_vapor": {
            "p_condensador": _value(steam.p_condenser_Pa, "pressure", system),
            "rendimiento_turbina": steam.eta_turbine,
            "rendimiento_bombas": steam.eta_pump,
            "p_desaireador": (
                None
                if steam.deaerator_p_Pa is None
                else _value(steam.deaerator_p_Pa, "pressure", system)
            ),
            "factor_de_baumann": inputs.baumann_alpha,
            "recalentamiento": (
                None
                if inputs.reheat is None
                else {
                    "p": _value(inputs.p_reheat_Pa or 0.0, "pressure", system),
                    "T": _value(inputs.reheat.T_K, "temperature", system),
                }
            ),
            "estados": [
                {
                    "estado": k + 1,
                    "descripcion": label,
                    "T": _value(s.T_K, "temperature", system),
                    "p": _value(s.P_Pa, "pressure", system),
                    "h": _value(s.h_J_per_kg, _EH, system),
                    "s": _value(s.s_J_per_kg_K, "specific_entropy", system),
                    "x": s.x,
                }
                for k, (label, s) in enumerate(zip(cyc.labels, cyc.states, strict=True))
            ],
            "turbinas": [
                {
                    "tramo": t.name,
                    "de_estado": t.inlet + 1,
                    "a_estado": t.outlet + 1,
                    "caudal_entrada": _value(t.m_in_kg_s, "mass_flow", system),
                    "potencia": _value(t.W_W, "power", system),
                    "rendimiento_humedo": None if t.wet is None else t.wet.eta_wet,
                }
                for t in cyc.turbines
            ],
            "caudal_vapor_total": _value(cyc.m_total_kg_s, "mass_flow", system),
            "caudal_extraccion_desaireador": _value(cyc.m_extraction_kg_s, "mass_flow", system),
            "titulo_salida_turbina": cyc.x_exhaust,
            "potencia_turbinas": _value(cyc.W_turbines_W, "power", system),
            "potencia_bombas": _value(result.W_pumps_W, "power", system),
            "calor_condensador": _value(cyc.Q_condenser_W, "power", system),
        },
        "resultados": {
            "potencia_turbina_de_gas": _value(result.W_gas_turbine_W, "power", system),
            "potencia_ciclo_de_vapor": _value(result.W_steam_turbine_W, "power", system),
            "potencia_neta": _value(result.W_net_W, "power", system),
            "calor_combustible": _value(result.Q_fuel_W, "power", system),
            "rendimiento_ciclo_combinado": result.eta_th,
            "rendimiento_turbina_de_gas": result.eta_gas_turbine,
            "rendimiento_ciclo_de_vapor": result.eta_steam,
            "aprovechamiento_hrsg": result.eta_hrsg,
            "heat_rate_kJ_kWh": result.heat_rate_kJ_per_kWh,
            "balance_de_energia": {
                name: _value(value, "power", system)
                for name, value in result.energy_balance_W.items()
            },
            "exergia_ciclo_de_fondo": {
                "T0": _value(ex.T0_K, "temperature", system),
                "gases_de_escape": _value(ex.X_gas_in_W, "power", system),
                "trabajo_neto_vapor": _value(ex.W_net_W, "power", system),
                "condensador": _value(ex.X_condenser_W, "power", system),
                "chimenea": _value(ex.X_stack_W, "power", system),
                "destruida": {name: _value(x, "power", system) for name, x in ex.destroyed_W},
                "rendimiento_exergetico": ex.efficiency,
            },
        },
        "fuente": (
            "Turbina de gas y HRSG con gases ideales (CoolProp), PCI de ISO 6976:2016 a 25 °C; "
            "agua IAPWS-95; cálculo directo (TESPy como control)"
        ),
    }


# ---------------------------------------------------------------------
# Control con TESPy
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MultiCombinedTespy:
    """El lado agua–vapor resuelto con TESPy, para comparar con el cálculo directo."""

    m_steam_kg_s: tuple[float, ...]
    W_steam_W: float
    T_stack_K: float
    seconds: float


def combined_multi_tespy(result: MultiCombinedResult) -> MultiCombinedTespy:  # noqa: PLR0912, PLR0915
    """Resuelve la misma HRSG y el mismo ciclo de vapor con una red de TESPy.

    Gases: ``HeatExchanger`` en serie; el sobrecalentador de alta y el
    recalentador en paralelo (``Splitter`` y ``Merge`` de gases, con la misma T
    a la salida de los dos bancos). Agua: el condensado (fuente) → bombas y
    desaireador (``Merge`` con x = 0) → economizadores, domos
    (``DropletSeparator``) y bombas entre niveles → sobrecalentadores →
    turbinas (``Turbine``) con las admisiones (``Merge``) y la extracción
    (``Splitter``) → condensador (sumidero). Cada tramo de turbina lleva el
    rendimiento que reproduce la línea de expansión del cálculo directo (con
    Baumann incluida). Como los ejemplos oficiales de TESPy (CCPP, SEGS).

    Raises
    ------
    ValueError
        Si TESPy no converge (con su explicación en castellano).
    """
    from tespy.components import (
        DropletSeparator,
        HeatExchanger,
        Merge,
        Pump,
        Sink,
        Source,
        Splitter,
        Turbine,
    )
    from tespy.connections import Connection, Ref

    from core.cycles.tespy_utils import new_network, solve

    inputs = result.inputs
    hrsg, cyc = result.hrsg, result.steam
    gt = result.gas_turbine
    levels = inputs.levels
    n = len(levels)
    rh = hrsg.reheat
    st = cyc.states
    nw = new_network()
    conns: list[Connection] = []

    def link(a: Any, ap: str, b: Any, bp: str, label: str) -> Connection:
        c = Connection(a, ap, b, bp, label=label)
        conns.append(c)
        return c

    # --- gases
    hx: dict[str, HeatExchanger] = {}
    order: list[str] = []
    for i, lv in enumerate(levels):
        if i == 0 and rh is not None:
            order.append("PAR")
        elif lv.T_steam_K is not None:
            order.append(f"SH{i}")
        order += [f"EV{i}", f"ECO{i}"]
    for name in order:
        if name == "PAR":
            if levels[0].T_steam_K is not None:
                hx["SH0"] = HeatExchanger("SH0")
            hx["RH"] = HeatExchanger("RH")
        else:
            hx[name] = HeatExchanger(name)
    g_in: dict[str, Connection] = {}
    prev: Any = Source("gases")
    port = "out1"
    par_out: tuple[Connection, Connection] | None = None
    for name in order:
        if name == "PAR":
            if "SH0" in hx:
                split = Splitter("reparto de gases", num_out=2)
                merge = Merge("mezcla de gases", num_in=2)
                link(prev, port, split, "in1", "g_par")
                link(split, "out1", hx["SH0"], "in1", "g_SH0")
                link(split, "out2", hx["RH"], "in1", "g_RH")
                par_out = (
                    link(hx["SH0"], "out1", merge, "in1", "g_SH0_out"),
                    link(hx["RH"], "out1", merge, "in2", "g_RH_out"),
                )
                prev, port = merge, "out1"
            else:
                link(prev, port, hx["RH"], "in1", "g_RH")
                prev, port = hx["RH"], "out1"
            continue
        g_in[name] = link(prev, port, hx[name], "in1", f"g_{name}")
        prev, port = hx[name], "out1"
    stack = link(prev, port, Sink("chimenea"), "in1", "g_stack")
    # --- agua: condensado → bombas → HRSG
    cp = Pump("bomba de condensado")
    c_cond = link(Source("condensado"), "out1", cp, "in1", "w_cond")
    w: dict[str, Connection] = {}
    pumps = [cp]
    if cyc.deaerator is None:
        src: Any = cp
    else:
        da, fp = Merge("desaireador", num_in=2), Pump("bomba de alimentación")
        pumps.append(fp)
        w["cp_out"] = link(cp, "out1", da, "in1", "w_cp_out")
        w["da_out"] = link(da, "out1", fp, "in1", "w_da_out")
        src = fp
    sport = "out1"
    steam_src: dict[int, tuple[Any, str]] = {}
    for i in reversed(range(n)):
        lv = levels[i]
        w[f"eco_in{i}"] = link(src, sport, hx[f"ECO{i}"], "in2", f"w_eco_in{i}")
        w[f"eco_out{i}"] = link(hx[f"ECO{i}"], "out2", hx[f"EV{i}"], "in2", f"w_eco_out{i}")
        if i == 0:
            if lv.T_steam_K is not None:
                ev_out0 = link(hx["EV0"], "out2", hx["SH0"], "in2", "w_ev_out0")
                steam_src[0] = (hx["SH0"], "out2")
            else:
                ev_out0 = None
                steam_src[0] = (hx["EV0"], "out2")
        else:
            drum = DropletSeparator(f"domo {i}")
            link(hx[f"EV{i}"], "out2", drum, "in1", f"w_ev_out{i}")
            if lv.T_steam_K is not None:
                link(drum, "out2", hx[f"SH{i}"], "in2", f"w_vap{i}")
                steam_src[i] = (hx[f"SH{i}"], "out2")
            else:
                steam_src[i] = (drum, "out2")
            pmp = Pump(f"bomba {i}")
            pumps.append(pmp)
            link(drum, "out1", pmp, "in1", f"w_liq{i}")
            src, sport = pmp, "out1"
    # --- turbinas
    cur, cport = steam_src[0]
    turbines: list[tuple[Turbine, float]] = []
    level_conn: dict[int, Connection] = {}
    for t_index, t in enumerate(cyc.turbines):
        tin, tout = st[t.inlet], st[t.outlet]
        if t.extraction is not None:
            ext = st[t.extraction]
            s_in = tin.s_J_per_kg_K
            h1s = float(PropsSI("H", "P", ext.P_Pa, "S", s_in, Water))
            h2s = float(PropsSI("H", "P", tout.P_Pa, "S", ext.s_J_per_kg_K, Water))
            eta1 = (tin.h_J_per_kg - ext.h_J_per_kg) / (tin.h_J_per_kg - h1s)
            eta2 = (ext.h_J_per_kg - tout.h_J_per_kg) / (ext.h_J_per_kg - h2s)
            t1, t2 = Turbine(f"T{t_index}a"), Turbine(f"T{t_index}b")
            c_in = link(cur, cport, t1, "in1", f"t_in{t_index}")
            spl = Splitter("extracción", num_out=2)
            link(t1, "out1", spl, "in1", "ext_in")
            w["ext"] = link(spl, "out1", da, "in2", "ext")
            link(spl, "out2", t2, "in1", "t_after_ext")
            turbines += [(t1, eta1), (t2, eta2)]
            cur, cport = t2, "out1"
        else:
            eta = (tin.h_J_per_kg - tout.h_J_per_kg) / (tin.h_J_per_kg - t.outlet_s.h_J_per_kg)
            t1 = Turbine(f"T{t_index}")
            c_in = link(cur, cport, t1, "in1", f"t_in{t_index}")
            turbines.append((t1, eta))
            cur, cport = t1, "out1"
        if t_index == 0:
            level_conn[0] = c_in
        if t.outlet == cyc.cold_reheat:
            assert rh is not None
            if inputs.reheat_joins_middle:
                mx = Merge("mezcla RH", num_in=2)
                link(cur, cport, mx, "in1", "crh")
                s, sp = steam_src[1]
                level_conn[1] = link(s, sp, mx, "in2", "steam1")
                link(mx, "out1", hx["RH"], "in2", "rh_in")
            else:
                link(cur, cport, hx["RH"], "in2", "rh_in")
            cur, cport = hx["RH"], "out2"
        adm = next(
            (a for a in cyc.admissions if a.turbine == t.outlet and a.outlet != cyc.reheat_inlet),
            None,
        )
        if adm is not None:
            mx = Merge(f"admisión {adm.level}", num_in=2)
            link(cur, cport, mx, "in1", f"adm_in{adm.level}")
            s, sp = steam_src[adm.level]
            level_conn[adm.level] = link(s, sp, mx, "in2", f"steam{adm.level}")
            cur, cport = mx, "out1"
    exhaust = link(cur, cport, Sink("condensador"), "in1", "exhaust")
    nw.add_conns(*conns)
    # --- especificaciones
    for name, h in hx.items():
        if name == "RH" and "SH0" in hx:
            h.set_attr(pr2=1)  # la presión de los gases a la salida la fija la mezcla
        else:
            h.set_attr(pr1=1, pr2=1)
    for turb, eta in turbines:
        turb.set_attr(eta_s=eta)
    cp.set_attr(eta_s=inputs.steam.eta_pump)
    for pmp in pumps[1:]:
        pmp.set_attr(eta_s=inputs.steam.eta_pump)
    fluid = {s: x for s, x in gt.gas.mass_fractions.items() if x > 0}
    conns[0].set_attr(T=gt.T_exhaust_K, p=gt.gas.p_Pa, m=gt.m_gas_kg_s, fluid=fluid)
    c_cond.set_attr(p=inputs.steam.p_condenser_Pa, x=0, fluid={"water": 1})
    if cyc.deaerator is not None:
        w["cp_out"].set_attr(p=inputs.steam.deaerator_p_Pa)
        w["da_out"].set_attr(x=0)
    for i, lv in enumerate(levels):
        g_in[f"ECO{i}"].set_attr(T=_T_sat(lv.p_Pa) + lv.pinch_K)
        w[f"eco_out{i}"].set_attr(td_bubble=lv.approach_K, m0=hrsg.levels[i].m_water_kg_s)
        w[f"eco_in{i}"].set_attr(p=lv.p_Pa)
        if lv.T_steam_K is not None:
            level_conn[i].set_attr(T=lv.T_steam_K)
    # el domo de alta evapora todo: x = 1 a la salida del evaporador de alta
    (ev_out0 if ev_out0 is not None else level_conn[0]).set_attr(x=1)
    if rh is not None:
        hot = next(c for c in conns if c.source is hx["RH"] and c.source_id == "out2")
        hot.set_attr(T=rh.hot.T_K)
        if par_out is not None:
            par_out[0].set_attr(T=Ref(par_out[1], 1, 0))
        if not inputs.reheat_joins_middle:
            next(c for c in conns if c.label == "rh_in").set_attr(p=rh.reheater.p_Pa)
    exhaust.set_attr(p=inputs.steam.p_condenser_Pa)
    t0 = time.perf_counter()
    solve(nw, what="el lado agua–vapor del ciclo combinado")
    seconds = time.perf_counter() - t0
    W_T = -sum(turb.P.val_SI for turb, _ in turbines)
    W_P = sum(p.P.val_SI for p in pumps)
    flows = tuple(float(level_conn[i].m.val_SI) for i in range(n))
    return MultiCombinedTespy(flows, float(W_T - W_P), float(stack.T.val_SI), seconds)
