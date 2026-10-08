"""Refrigeración por compresión de vapor con TESPy — Fase 3.2.

Ciclos de Çengel & Boles, *Termodinámica* (8.ª ed.), cap. 11:

- **Simple** (§11-3 ideal y §11-4 real): compresor, condensador, válvula de
  expansión y evaporador. El ciclo real suma el rendimiento isoentrópico del
  compresor, el sobrecalentamiento a la salida del evaporador, el
  subenfriamiento a la salida del condensador y las caídas de presión en los
  dos intercambiadores (ejemplo 11-2).
- **Dos etapas con cámara de evaporación instantánea** (§11-8, ejemplo 11-5;
  el «economizador abierto»): el líquido del condensador se estrangula hasta
  una presión intermedia; el vapor que se forma va directo al compresor de
  alta y solo el líquido sigue hasta el evaporador.
- **Cascada de dos etapas** (§11-8, ejemplo 11-4): el condensador del ciclo
  de baja es el evaporador del de alta; cada ciclo puede tener su
  refrigerante (p. ej. CO₂ abajo y amoníaco arriba).

Sirve como **refrigerador** (efecto útil Q_C, el calor que se saca de la
fuente fría) o como **bomba de calor** (Q_H, el que se entrega a la fuente
caliente; §11-7). Con las temperaturas de las fuentes suma el análisis de
**segundo principio** (§11-5): exergía destruida en cada componente,
trabajo mínimo y rendimiento exergético.

La red sigue el tutorial oficial de bomba de calor de TESPy 0.11
(``tutorial/basics/heat_pump.py``: ``CycleCloser`` + ``SimpleHeatExchanger``
+ ``Compressor`` + ``Valve``). El sobrecalentamiento y el subenfriamiento se
fijan con ``td_dew`` y ``td_bubble``, como en ``tutorial/advanced/stepwise.py``.
La cámara es un ``DropletSeparator`` (separa el líquido y el vapor saturados)
más un ``Merge``; el intercambiador de la cascada, un ``HeatExchanger``.

Notación del vademecum (§9.3): Q_C es el calor que se saca de la fuente fría
(el Q_L de Cengel) y Q_H el que se entrega a la caliente; COP_R = Q_C/W y
COP_B = Q_H/W. Numeración de estados de Cengel (ver
:func:`refrigeration_layout`). Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from tespy.components import (
    Compressor,
    CycleCloser,
    DropletSeparator,
    HeatExchanger,
    Merge,
    SimpleHeatExchanger,
    Valve,
)
from tespy.connections import Connection

from core.cycles.refrigeration_procedure import refrigeration_steps
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

__all__ = [
    "check_reservoirs",
    "default_reservoirs",
    "CYCLE_NAMES",
    "REFRIGERATION_EXAMPLES",
    "REFRIGERATION_EXAMPLE_NOTES",
    "REFRIGERATION_EXAMPLES_BY_TEMPERATURE",
    "REFRIGERATION_FLUIDS",
    "T_DISCHARGE_HIGH_K",
    "TON_OF_REFRIGERATION_W",
    "Application",
    "Cascade",
    "CycleKind",
    "ExergyAnalysis",
    "RefrigerationComponent",
    "RefrigerationInputs",
    "RefrigerationLayout",
    "RefrigerationResult",
    "Reservoirs",
    "SingleStage",
    "SweepParameter",
    "SweepPoint",
    "condensation_pressure",
    "condensation_temperature",
    "default_flash_pressure",
    "default_sweep_values",
    "evaporation_pressure",
    "evaporation_temperature",
    "refrigeration_diagram_groups",
    "refrigeration_isentropic_states",
    "refrigeration_labeled_states",
    "refrigeration_layout",
    "refrigeration_notes",
    "refrigeration_steps",
    "refrigeration_sweep",
    "refrigeration_to_dict",
    "single_stage_equivalent",
    "solve_refrigeration",
    "suggested_refrigeration_inputs",
    "validate_refrigeration_inputs",
]

#: Refrigerantes de la página (Cengel §11-6): HFC (R-134a, R-410A, R-32), HFO
#: (R-1234yf) y naturales (amoníaco, CO₂, propano e isobutano).
REFRIGERATION_FLUIDS: tuple[str, ...] = (
    "R134a",
    "R1234yf",
    "R410A",
    "R32",
    "Ammonia",
    "CarbonDioxide",
    "Propane",
    "IsoButane",
)

CycleKind = Literal["simple", "flash", "cascade"]
Application = Literal["refrigerator", "heat_pump"]
ComponentKind = Literal[
    "compressor", "condenser", "evaporator", "valve", "flash_tank", "mixer", "cascade_hx"
]
SweepParameter = Literal[
    "T_evap", "T_cond", "eta_compressor", "superheat", "subcooling", "p_flash", "T_cascade"
]

#: Cómo se llama cada ciclo (para títulos y el export).
CYCLE_NAMES: dict[str, str] = {
    "simple": "simple",
    "flash": "de dos etapas con cámara de evaporación instantánea",
    "cascade": "en cascada de dos etapas",
}

#: Tonelada de refrigeración: 12 000 Btu/h ≈ 211 kJ/min ≈ 3,517 kW (Cengel §11-1).
TON_OF_REFRIGERATION_W = 12_000.0 * 1055.05585262 / 3600.0

#: Temperatura de descarga del compresor por encima de la cual se avisa (120 °C).
T_DISCHARGE_HIGH_K = 393.15

_EH: QuantityKind = "specific_enthalpy"
_P_ATM = 101_325.0
# Tolerancia de temperatura (K) para comparar con la saturación o con las fuentes.
_T_TOL = 0.01


def _bar(p_Pa: float) -> str:
    return f"{p_Pa / 1e5:.4g} bar"


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


# ---------------------------------------------------------------------
# Temperaturas y presiones de saturación (rocío y burbuja)
# ---------------------------------------------------------------------


def evaporation_temperature(fluid: str, p_Pa: float) -> float:
    """T de evaporación a ``p_Pa``: la de rocío (vapor saturado), donde termina
    de evaporar. En un fluido puro es la T de saturación; el R-410A, que es una
    mezcla, tiene un pequeño deslizamiento entre burbuja y rocío."""
    return saturation_at_pressure(fluid, p_Pa).vapor.T_K


def condensation_temperature(fluid: str, p_Pa: float) -> float:
    """T de condensación a ``p_Pa``: la de burbuja (líquido saturado), donde
    termina de condensar."""
    return saturation_at_pressure(fluid, p_Pa).liquid.T_K


def evaporation_pressure(fluid: str, T_K: float) -> float:
    """Presión a la que ``fluid`` termina de evaporar a ``T_K`` (punto de rocío)."""
    return saturation_at_temperature(fluid, T_K).vapor.P_Pa


def condensation_pressure(fluid: str, T_K: float) -> float:
    """Presión a la que ``fluid`` termina de condensar a ``T_K`` (punto de burbuja)."""
    return saturation_at_temperature(fluid, T_K).liquid.P_Pa


def default_flash_pressure(p_evap_Pa: float, p_cond_Pa: float) -> float:
    """Presión de la cámara por defecto: la media geométrica √(p_evap·p_cond).

    Es la presión intermedia que minimiza el trabajo de una compresión en dos
    etapas de un gas ideal con enfriamiento intermedio (Cengel cap. 7); en un
    ciclo con cámara queda cerca del óptimo (el barrido lo muestra).
    """
    return math.sqrt(p_evap_Pa * p_cond_Pa)


# ---------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Cascade:
    """Ciclo de baja de una cascada de dos etapas (Cengel §11-8), en SI.

    El ciclo de baja condensa a ``p_cond_low_Pa`` en el intercambiador que lo
    une con el de alta, que ahí evapora a ``p_evap_high_Pa``. ``fluid_low`` es
    el refrigerante del ciclo de baja (``None``: el mismo que el de alta). Con
    el mismo fluido y la misma presión es el intercambiador ideal del ejemplo
    11-4; en la práctica el de baja condensa unos grados más caliente.
    """

    p_cond_low_Pa: float
    p_evap_high_Pa: float
    fluid_low: str | None = None


@dataclass(frozen=True)
class Reservoirs:
    """Temperaturas de las fuentes (SI), para el segundo principio (Cengel §11-5).

    ``T_cold_K`` es la fuente fría: el espacio refrigerado o, en una bomba de
    calor, el ambiente exterior. ``T_hot_K`` es la caliente: el ambiente o el
    espacio que se calefacciona. El ambiente es el estado muerto T₀.
    """

    T_cold_K: float
    T_hot_K: float


@dataclass(frozen=True)
class RefrigerationInputs:
    """Datos del ciclo, en SI.

    ``p_evap_Pa`` y ``p_cond_Pa`` son las presiones de aspiración y de
    descarga del compresor (o de los compresores): la salida del evaporador y
    la entrada al condensador. Con caídas de presión, la válvula descarga a
    ``p_evap_Pa + dp_evap_Pa`` y el condensador entrega el líquido a
    ``p_cond_Pa − dp_cond_Pa``. ``superheat_K`` se mide sobre la temperatura
    de rocío a la salida del evaporador y ``subcooling_K`` bajo la de burbuja
    a la salida del condensador.

    Se da el caudal másico **por el condensador** (la base de Cengel: el total
    con cámara, el del ciclo de alta en la cascada) **o** la capacidad (Q̇_C
    del refrigerador o Q̇_H de la bomba de calor); el otro se calcula.
    ``p_flash_Pa`` arma el ciclo de dos etapas con cámara y ``cascade``, la
    cascada; ``reservoirs`` agrega el segundo principio.
    """

    p_evap_Pa: float
    p_cond_Pa: float
    eta_compressor: float = 1.0
    fluid: str = "R134a"
    superheat_K: float = 0.0
    subcooling_K: float = 0.0
    dp_evap_Pa: float = 0.0
    dp_cond_Pa: float = 0.0
    m_dot_kg_s: float | None = 1.0
    capacity_W: float | None = None
    application: Application = "refrigerator"
    p_flash_Pa: float | None = None
    cascade: Cascade | None = None
    reservoirs: Reservoirs | None = None

    @property
    def cycle(self) -> CycleKind:
        if self.cascade is not None:
            return "cascade"
        if self.p_flash_Pa is not None:
            return "flash"
        return "simple"

    @property
    def fluid_low(self) -> str:
        """Refrigerante del evaporador (el del ciclo de baja en una cascada)."""
        if self.cascade is not None and self.cascade.fluid_low is not None:
            return self.cascade.fluid_low
        return self.fluid

    @property
    def has_losses(self) -> bool:
        """Sobrecalentamiento, subenfriamiento o caídas de presión (ciclo real, §11-4)."""
        return any(
            v != 0.0
            for v in (self.superheat_K, self.subcooling_K, self.dp_evap_Pa, self.dp_cond_Pa)
        )

    @property
    def is_ideal(self) -> bool:
        """El ciclo ideal de Cengel §11-3: compresión isoentrópica y nada más."""
        return self.eta_compressor == 1.0 and not self.has_losses

    @property
    def heat_pump(self) -> bool:
        return self.application == "heat_pump"

    @property
    def layout(self) -> RefrigerationLayout:
        """Plano del ciclo con la numeración de estados (sin resolver)."""
        return refrigeration_layout(self.cycle, self.fluid, self.fluid_low)


# ---------------------------------------------------------------------
# Plano del ciclo: componentes y numeración (Cengel cap. 11)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RefrigerationComponent:
    """Componente tal como lo ve el alumno: entradas y salidas como índices de
    estado (número de estado − 1). En el intercambiador de la cascada,
    ``inlets[k]`` sale por ``outlets[k]`` (lado caliente y lado frío); en la
    cámara, la entrada sale por las dos salidas (vapor y líquido); en la
    mezcla, las dos entradas salen juntas."""

    kind: ComponentKind
    label: str
    inlets: tuple[int, ...]
    outlets: tuple[int, ...]

    @property
    def inlet(self) -> int:
        return self.inlets[0]

    @property
    def outlet(self) -> int:
        return self.outlets[0]

    @property
    def paths(self) -> tuple[tuple[int, int], ...]:
        """Tramos (entrada, salida) que recorre el refrigerante en el componente."""
        if self.kind == "cascade_hx":
            return tuple(zip(self.inlets, self.outlets, strict=True))
        if self.kind == "flash_tank":
            return tuple((self.inlet, o) for o in self.outlets)
        if self.kind == "mixer":
            return tuple((i, self.outlet) for i in self.inlets)
        return ((self.inlet, self.outlet),)


@dataclass(frozen=True)
class RefrigerationLayout:
    """Estados (etiqueta y fluido) y componentes de un ciclo, sin resolver."""

    cycle: CycleKind
    labels: tuple[str, ...]
    fluids: tuple[str, ...]
    components: tuple[RefrigerationComponent, ...]

    @property
    def n_states(self) -> int:
        return len(self.labels)

    def labeled(self) -> list[str]:
        """«1 (entrada al compresor …)», … en el orden de la numeración."""
        return [f"{i + 1} ({label})" for i, label in enumerate(self.labels)]

    def of_kind(self, kind: ComponentKind) -> tuple[RefrigerationComponent, ...]:
        return tuple(c for c in self.components if c.kind == kind)

    def one(self, kind: ComponentKind) -> RefrigerationComponent:
        """El único componente de ese tipo (evaporador, condensador, cámara…)."""
        (comp,) = self.of_kind(kind)
        return comp

    @property
    def segments(self) -> tuple[tuple[int, int], ...]:
        """Todos los tramos del ciclo, para el diagrama."""
        return tuple(path for c in self.components for path in c.paths)


def _c(kind: ComponentKind, label: str, ins: Sequence[int], outs: Sequence[int]):
    return RefrigerationComponent(kind, label, tuple(ins), tuple(outs))


def refrigeration_layout(
    cycle: CycleKind, fluid: str = "R134a", fluid_low: str | None = None
) -> RefrigerationLayout:
    """Numeración de estados de Cengel cap. 11 (índice = número de estado − 1).

    - **Simple** (§11-3): 1 entrada al compresor (salida del evaporador),
      2 salida del compresor, 3 salida del condensador, 4 salida de la
      válvula (entrada al evaporador).
    - **Con cámara** (ejemplo 11-5): 1 → 2 compresor de baja; 3 vapor de la
      cámara; 9 mezcla → 4 compresor de alta; 5 salida del condensador;
      6 entrada a la cámara; 7 líquido de la cámara; 8 entrada al evaporador.
    - **Cascada** (ejemplo 11-4): 1 a 4 el ciclo de baja (3 sale del
      intercambiador como líquido) y 5 a 8 el de alta (5 sale del
      intercambiador como vapor).
    """
    low = fluid_low or fluid
    if cycle == "simple":
        labels = (
            "entrada al compresor (salida del evaporador)",
            "salida del compresor",
            "salida del condensador",
            "entrada al evaporador (salida de la válvula)",
        )
        components = (
            _c("compressor", "compresor", [0], [1]),
            _c("condenser", "condensador", [1], [2]),
            _c("valve", "válvula de expansión", [2], [3]),
            _c("evaporator", "evaporador", [3], [0]),
        )
        fluids = (fluid,) * 4
    elif cycle == "flash":
        labels = (
            "entrada al compresor de baja (salida del evaporador)",
            "salida del compresor de baja",
            "vapor saturado de la cámara",
            "salida del compresor de alta",
            "salida del condensador",
            "entrada a la cámara (salida de la válvula de alta)",
            "líquido saturado de la cámara",
            "entrada al evaporador (salida de la válvula de baja)",
            "entrada al compresor de alta (mezcla)",
        )
        components = (
            _c("compressor", "compresor de baja", [0], [1]),
            _c("mixer", "cámara de mezcla", [1, 2], [8]),
            _c("compressor", "compresor de alta", [8], [3]),
            _c("condenser", "condensador", [3], [4]),
            _c("valve", "válvula de alta", [4], [5]),
            _c("flash_tank", "cámara de evaporación instantánea", [5], [2, 6]),
            _c("valve", "válvula de baja", [6], [7]),
            _c("evaporator", "evaporador", [7], [0]),
        )
        fluids = (fluid,) * 9
    elif cycle == "cascade":
        labels = (
            "entrada al compresor de baja (salida del evaporador)",
            "salida del compresor de baja",
            "salida del intercambiador (líquido del ciclo de baja)",
            "entrada al evaporador (salida de la válvula de baja)",
            "entrada al compresor de alta (vapor del intercambiador)",
            "salida del compresor de alta",
            "salida del condensador",
            "entrada al intercambiador (salida de la válvula de alta)",
        )
        components = (
            _c("compressor", "compresor de baja", [0], [1]),
            _c("cascade_hx", "intercambiador de la cascada", [1, 7], [2, 4]),
            _c("valve", "válvula de baja", [2], [3]),
            _c("evaporator", "evaporador", [3], [0]),
            _c("compressor", "compresor de alta", [4], [5]),
            _c("condenser", "condensador", [5], [6]),
            _c("valve", "válvula de alta", [6], [7]),
        )
        fluids = (low,) * 4 + (fluid,) * 4
    else:
        raise ValueError(f"Ciclo {cycle!r} desconocido: simple, flash o cascade.")
    return RefrigerationLayout(cycle, labels, fluids, components)


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ExergyAnalysis:
    """Segundo principio del ciclo (Cengel §11-5; vademecum §11.5, §11.8 y §11.10).

    ``T0_K`` es el ambiente (la fuente caliente del refrigerador, la fría de la
    bomba de calor). ``W_min_W`` es la exergía del efecto útil, el trabajo
    mínimo (el de un ciclo reversible entre las fuentes). ``destroyed`` es la
    exergía destruida en cada componente, X_dest = T₀·S_gen (W). Se cumple
    Ẇ = Ẇ_mín + ΣX_dest.
    """

    T0_K: float
    W_min_W: float
    W_W: float
    destroyed: tuple[tuple[str, float], ...]
    S_gen: tuple[tuple[str, float], ...]

    @property
    def X_destroyed_W(self) -> float:
        return sum(x for _, x in self.destroyed)

    @property
    def eta_ex(self) -> float:
        """Rendimiento exergético Ẇ_mín / Ẇ = COP / COP_rev (vademecum §11.10)."""
        return self.W_min_W / self.W_W


@dataclass(frozen=True)
class SingleStage:
    """Ciclo simple equivalente, para comparar con la cámara o la cascada: el
    refrigerante del condensador entre las mismas temperaturas de evaporación y
    condensación, con el mismo compresor."""

    fluid: str
    COP: float
    T_discharge_K: float
    pressure_ratio: float


@dataclass(frozen=True)
class RefrigerationResult:
    """Estados y balances del ciclo, en SI.

    ``states`` sigue la numeración de Cengel (:func:`refrigeration_layout`) y
    ``m_dot_kg_s`` es el caudal de cada estado. ``compressors_out_s`` es la
    salida isoentrópica de cada compresor (2s, o 2s y 4s / 6s), a la presión
    de descarga con la entropía de la aspiración. ``tespy_balances`` son la
    potencia de cada compresor y el calor de cada intercambiador que calcula
    TESPy (W, positivos hacia el refrigerante), como control cruzado;
    ``tespy_internal`` es el calor que pasa del ciclo de baja al de alta en la
    cascada. ``single_stage`` es el ciclo simple equivalente (para la cámara y
    la cascada; ver :class:`SingleStage`).
    """

    inputs: RefrigerationInputs
    layout: RefrigerationLayout
    states: tuple[FluidState, ...]
    m_dot_kg_s: tuple[float, ...]
    compressors_out_s: tuple[FluidState, ...]
    tespy_balances: tuple[tuple[str, float], ...] = ()
    tespy_internal: tuple[tuple[str, float], ...] = ()
    single_stage: SingleStage | None = None

    @property
    def single_stage_COP(self) -> float | None:
        """COP del ciclo simple equivalente (``None`` si no hay con qué comparar)."""
        return None if self.single_stage is None else self.single_stage.COP

    # --- componentes -----------------------------------------------------
    @property
    def cycle(self) -> CycleKind:
        return self.layout.cycle

    @property
    def components(self) -> tuple[RefrigerationComponent, ...]:
        return self.layout.components

    def of_kind(self, kind: ComponentKind) -> tuple[RefrigerationComponent, ...]:
        return self.layout.of_kind(kind)

    @property
    def evaporator(self) -> RefrigerationComponent:
        return self.layout.one("evaporator")

    @property
    def condenser(self) -> RefrigerationComponent:
        return self.layout.one("condenser")

    @property
    def compressors(self) -> tuple[RefrigerationComponent, ...]:
        return self.layout.of_kind("compressor")

    def energy_out_minus_in_W(self, comp: RefrigerationComponent) -> float:
        """Σ ṁ·h a la salida − Σ ṁ·h a la entrada de un componente (W)."""
        m, st = self.m_dot_kg_s, self.states
        return sum(m[o] * st[o].h_J_per_kg for o in comp.outlets) - sum(
            m[i] * st[i].h_J_per_kg for i in comp.inlets
        )

    def entropy_out_minus_in_W_per_K(self, comp: RefrigerationComponent) -> float:
        """Σ ṁ·s a la salida − Σ ṁ·s a la entrada de un componente (W/K)."""
        m, st = self.m_dot_kg_s, self.states
        return sum(m[o] * st[o].s_J_per_kg_K for o in comp.outlets) - sum(
            m[i] * st[i].s_J_per_kg_K for i in comp.inlets
        )

    # --- calores, trabajo y COP (W) ---------------------------------------
    @property
    def Q_cold_W(self) -> float:
        """Q̇_C: calor que el evaporador saca de la fuente fría (el Q̇_L de Cengel)."""
        return self.energy_out_minus_in_W(self.evaporator)

    @property
    def Q_hot_W(self) -> float:
        """Q̇_H: calor que el condensador entrega a la fuente caliente."""
        return -self.energy_out_minus_in_W(self.condenser)

    @property
    def compressor_powers_W(self) -> tuple[float, ...]:
        return tuple(self.energy_out_minus_in_W(c) for c in self.compressors)

    @property
    def W_W(self) -> float:
        """Ẇ: potencia de los compresores."""
        return sum(self.compressor_powers_W)

    @property
    def Q_cascade_W(self) -> float:
        """Calor que el ciclo de baja le pasa al de alta (0 si no es una cascada)."""
        hx = self.of_kind("cascade_hx")
        if not hx:
            return 0.0
        (comp,) = hx
        m, st = self.m_dot_kg_s, self.states
        hot_in, cold_in = comp.inlets
        hot_out, _ = comp.outlets
        return m[hot_in] * (st[hot_in].h_J_per_kg - st[hot_out].h_J_per_kg)

    @property
    def COP_R(self) -> float:
        """COP_R = Q̇_C / Ẇ (vademecum §9.3)."""
        return self.Q_cold_W / self.W_W

    @property
    def COP_B(self) -> float:
        """COP_B = Q̇_H / Ẇ = COP_R + 1 (vademecum §9.3)."""
        return self.Q_hot_W / self.W_W

    @property
    def COP(self) -> float:
        """El COP del uso elegido: COP_R del refrigerador o COP_B de la bomba de calor."""
        return self.COP_B if self.inputs.heat_pump else self.COP_R

    @property
    def useful_heat_W(self) -> float:
        """Efecto útil: Q̇_C (refrigerador) o Q̇_H (bomba de calor)."""
        return self.Q_hot_W if self.inputs.heat_pump else self.Q_cold_W

    @property
    def tons_of_refrigeration(self) -> float:
        """Q̇_C en toneladas de refrigeración (1 TR ≈ 211 kJ/min, Cengel §11-1)."""
        return self.Q_cold_W / TON_OF_REFRIGERATION_W

    # --- por kilogramo que pasa por el condensador (la base de Cengel) ------
    @property
    def m_cond_kg_s(self) -> float:
        return self.m_dot_kg_s[self.condenser.inlet]

    @property
    def m_evap_kg_s(self) -> float:
        return self.m_dot_kg_s[self.evaporator.inlet]

    @property
    def q_cold_J_per_kg(self) -> float:
        return self.Q_cold_W / self.m_cond_kg_s

    @property
    def q_hot_J_per_kg(self) -> float:
        return self.Q_hot_W / self.m_cond_kg_s

    @property
    def w_J_per_kg(self) -> float:
        return self.W_W / self.m_cond_kg_s

    # --- temperaturas y Carnot ---------------------------------------------
    @property
    def T_evap_K(self) -> float:
        """Temperatura de evaporación: la de rocío a la presión de aspiración."""
        return evaporation_temperature(self.inputs.fluid_low, self.inputs.p_evap_Pa)

    @property
    def T_cond_K(self) -> float:
        """Temperatura de condensación: la de burbuja a la presión de descarga."""
        return condensation_temperature(self.inputs.fluid, self.inputs.p_cond_Pa)

    @property
    def COP_carnot(self) -> float:
        """COP de un ciclo de Carnot invertido entre T_evap y T_cond (vademecum §9.3):
        el máximo posible si el refrigerante evaporara y condensara a esas temperaturas."""
        T_L, T_H = self.T_evap_K, self.T_cond_K
        return (T_H if self.inputs.heat_pump else T_L) / (T_H - T_L)

    @property
    def COP_rev(self) -> float | None:
        """COP reversible entre las fuentes (vademecum §9.3); ``None`` sin fuentes."""
        res = self.inputs.reservoirs
        if res is None:
            return None
        T_L, T_H = res.T_cold_K, res.T_hot_K
        return (T_H if self.inputs.heat_pump else T_L) / (T_H - T_L)

    # --- otros resultados ----------------------------------------------------
    @property
    def T_discharge_K(self) -> float:
        """Temperatura máxima de descarga de los compresores."""
        return max(self.states[c.outlet].T_K for c in self.compressors)

    @property
    def pressure_ratios(self) -> tuple[float, ...]:
        """p_descarga / p_aspiración de cada compresor."""
        st = self.states
        return tuple(st[c.outlet].P_Pa / st[c.inlet].P_Pa for c in self.compressors)

    @property
    def x_evaporator_in(self) -> float | None:
        """Título a la entrada del evaporador (después de la válvula)."""
        return self.states[self.evaporator.inlet].x

    @property
    def flash_fraction(self) -> float | None:
        """Fracción del caudal que se evapora en la cámara: x₆ (ejemplo 11-5)."""
        tanks = self.of_kind("flash_tank")
        if not tanks:
            return None
        (tank,) = tanks
        vapor = tank.outlets[0]
        return self.m_dot_kg_s[vapor] / self.m_dot_kg_s[tank.inlet]

    @property
    def mass_ratio(self) -> float | None:
        """ṁ_baja / ṁ_alta en la cascada (ejemplo 11-4)."""
        if self.cycle != "cascade":
            return None
        return self.m_evap_kg_s / self.m_cond_kg_s

    @property
    def suction_volume_flow_m3_s(self) -> float:
        """Caudal volumétrico a la entrada del (primer) compresor, V̇₁ = ṁ₁·v₁:
        a igual capacidad, el refrigerante que necesita menos pide un
        compresor más chico (Cengel §11-6)."""
        i = self.compressors[0].inlet
        return self.m_dot_kg_s[i] * self.states[i].v_m3_per_kg

    @property
    def wet_compressors(self) -> tuple[tuple[str, float], ...]:
        """Compresores cuya descarga queda dentro de la campana: (nombre, título)."""
        out = []
        for c in self.compressors:
            x = self.states[c.outlet].x
            if x is not None:
                out.append((c.label, float(x)))
        return tuple(out)

    @property
    def exergy(self) -> ExergyAnalysis | None:
        """Segundo principio con las fuentes de ``inputs.reservoirs`` (o ``None``)."""
        return _exergy_analysis(self)


def _exergy_analysis(result: RefrigerationResult) -> ExergyAnalysis | None:
    """X_dest = T₀·S_gen de cada componente; el evaporador recibe Q̇_C de la
    fuente fría y el condensador entrega Q̇_H a la caliente (vademecum §11.8)."""
    res = result.inputs.reservoirs
    if res is None:
        return None
    heat_pump = result.inputs.heat_pump
    T0 = res.T_cold_K if heat_pump else res.T_hot_K
    s_gen: list[tuple[str, float]] = []
    for comp in result.components:
        ds = result.entropy_out_minus_in_W_per_K(comp)
        if comp.kind == "evaporator":
            ds -= result.Q_cold_W / res.T_cold_K
        elif comp.kind == "condenser":
            ds += result.Q_hot_W / res.T_hot_K
        s_gen.append((comp.label, ds))
    if heat_pump:
        W_min = result.Q_hot_W * (1.0 - T0 / res.T_hot_K)
    else:
        W_min = result.Q_cold_W * (T0 / res.T_cold_K - 1.0)
    return ExergyAnalysis(
        T0_K=T0,
        W_min_W=W_min,
        W_W=result.W_W,
        destroyed=tuple((name, T0 * s) for name, s in s_gen),
        S_gen=tuple(s_gen),
    )


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


def _check_fluid(fluid: str, what: str) -> None:
    if fluid not in REFRIGERATION_FLUIDS:
        names = ", ".join(FLUID_NAMES_ES.get(f, f) for f in REFRIGERATION_FLUIDS)
        raise ValueError(
            f"El fluido {fluid!r} no está disponible como {what}. Refrigerantes: {names}."
        )


def _check_evaporation(fluid: str, p_Pa: float, name: str) -> float:
    """Presión de evaporación dentro de la campana; devuelve la T de rocío.

    ``name`` completa «La presión …» (p. ej. ``"del evaporador"``).
    """
    lim = fluid_limits(fluid)
    if p_Pa <= lim.P_triple_Pa:
        raise ValueError(
            f"La presión {name} ({_bar(p_Pa)}) está por debajo de la del punto triple "
            f"{_del(fluid)} ({_bar(lim.P_triple_Pa)}, {_degC(lim.T_triple_K)}): a esa presión el "
            "refrigerante no puede evaporar, sería sólido. Subí la presión."
        )
    if p_Pa >= lim.P_crit_Pa:
        raise ValueError(
            f"La presión {name} ({_bar(p_Pa)}) supera la crítica {_del(fluid)} "
            f"({_bar(lim.P_crit_Pa)}): por encima del punto crítico no hay evaporación."
        )
    return evaporation_temperature(fluid, p_Pa)


def _check_condensation(fluid: str, p_Pa: float, name: str) -> float:
    """Presión de condensación subcrítica; devuelve la T de burbuja.

    ``name`` completa «La presión …» (p. ej. ``"del condensador"``).
    """
    lim = fluid_limits(fluid)
    if p_Pa >= lim.P_crit_Pa:
        message = (
            f"La presión {name} ({_bar(p_Pa)}) supera la crítica {_del(fluid)} "
            f"({_bar(lim.P_crit_Pa)}, {_degC(lim.T_crit_K)}): por encima del punto crítico el "
            "refrigerante no condensa."
        )
        if fluid == "CarbonDioxide":
            message += (
                " Con CO₂ (R-744) pasa por encima de 31 °C: es el ciclo transcrítico, con un "
                "enfriador de gas en vez de condensador, que esta página no resuelve. Usalo con "
                "condensación por debajo de 31 °C, por ejemplo como ciclo de baja de una cascada."
            )
        else:
            message += " Bajá la presión."
        raise ValueError(message)
    if p_Pa <= lim.P_triple_Pa:
        raise ValueError(
            f"La presión {name} ({_bar(p_Pa)}) está por debajo de la del punto triple "
            f"{_del(fluid)} ({_bar(lim.P_triple_Pa)}): el refrigerante no puede condensar."
        )
    return condensation_temperature(fluid, p_Pa)


def _condenser_outlet_T(inputs: RefrigerationInputs) -> float:
    """T a la salida del condensador: burbuja a p_cond − Δp, menos el subenfriamiento."""
    p_out = inputs.p_cond_Pa - inputs.dp_cond_Pa
    return condensation_temperature(inputs.fluid, p_out) - inputs.subcooling_K


def validate_refrigeration_inputs(inputs: RefrigerationInputs) -> None:
    """Verifica que el ciclo tenga sentido físico; si no, explica por qué.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno: qué dato está fuera de rango y por qué.
    """
    fluid, low = inputs.fluid, inputs.fluid_low
    _check_fluid(fluid, "refrigerante")
    if inputs.cascade is not None:
        _check_fluid(low, "refrigerante del ciclo de baja")
    if inputs.application not in ("refrigerator", "heat_pump"):
        raise ValueError("El uso tiene que ser refrigerador o bomba de calor.")
    if inputs.p_flash_Pa is not None and inputs.cascade is not None:
        raise ValueError(
            "Elegí un solo tipo de ciclo: la cámara de evaporación instantánea o la cascada."
        )
    given = [v for v in (inputs.m_dot_kg_s, inputs.capacity_W) if v is not None]
    if len(given) != 1:
        raise ValueError("Dá el caudal másico o la capacidad (uno solo de los dos).")
    if not (math.isfinite(given[0]) and given[0] > 0.0):
        raise ValueError("El caudal y la capacidad tienen que ser positivos.")
    if not 0.0 < inputs.eta_compressor <= 1.0:
        raise ValueError(
            f"El rendimiento isoentrópico del compresor η_C = {inputs.eta_compressor:g} tiene "
            "que estar entre 0 y 1 (1 = compresor ideal; vademecum §10.4)."
        )
    for value, name in (
        (inputs.superheat_K, "El sobrecalentamiento"),
        (inputs.subcooling_K, "El subenfriamiento"),
        (inputs.dp_evap_Pa, "La caída de presión en el evaporador"),
        (inputs.dp_cond_Pa, "La caída de presión en el condensador"),
    ):
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} no puede ser negativo.")
    if inputs.p_evap_Pa <= 0.0 or inputs.p_cond_Pa <= 0.0:
        raise ValueError("Las presiones tienen que ser positivas (absolutas).")
    # En una cascada son de ciclos (y quizá fluidos) distintos: cada ciclo se revisa aparte.
    if inputs.cascade is None and inputs.p_evap_Pa >= inputs.p_cond_Pa:
        raise ValueError(
            f"La presión del evaporador ({_bar(inputs.p_evap_Pa)}) tiene que ser menor que la "
            f"del condensador ({_bar(inputs.p_cond_Pa)}): el compresor sube la presión para que "
            "el refrigerante condense más caliente que lo que evaporó."
        )

    # Evaporador: entre el punto triple y el crítico, y el sobrecalentamiento.
    p_valve_out = inputs.p_evap_Pa + inputs.dp_evap_Pa
    T_evap = _check_evaporation(low, inputs.p_evap_Pa, "del evaporador")
    _check_evaporation(low, p_valve_out, "de entrada al evaporador")
    lim_low = fluid_limits(low)
    T1 = T_evap + inputs.superheat_K
    if T1 >= lim_low.T_max_K:
        raise ValueError(
            f"Con ese sobrecalentamiento el vapor saldría del evaporador a {_degC(T1)}, por encima "
            f"del máximo de la ecuación de estado {_del(low)} ({_degC(lim_low.T_max_K)})."
        )

    # Condensador: subcrítico, y el subenfriamiento no congela el líquido.
    _check_condensation(fluid, inputs.p_cond_Pa, "del condensador")
    p_cond_out = inputs.p_cond_Pa - inputs.dp_cond_Pa
    if p_cond_out <= 0.0:
        raise ValueError(
            f"La caída de presión en el condensador ({_bar(inputs.dp_cond_Pa)}) es mayor que la "
            f"presión de descarga ({_bar(inputs.p_cond_Pa)})."
        )
    _check_condensation(fluid, p_cond_out, "de salida del condensador")
    lim = fluid_limits(fluid)
    T3 = _condenser_outlet_T(inputs)
    if T3 <= lim.T_triple_K + 0.5:
        raise ValueError(
            f"Con ese subenfriamiento el líquido saldría del condensador a {_degC(T3)}, por "
            f"debajo del punto triple {_del(fluid)} ({_degC(lim.T_triple_K)}): se congelaría."
        )

    if inputs.cycle == "simple":
        if p_cond_out <= p_valve_out:
            raise ValueError(
                f"Con esas caídas de presión el líquido llegaría a la válvula a {_bar(p_cond_out)} "
                f"y tendría que salir a {_bar(p_valve_out)}: una válvula de expansión solo puede "
                "bajar la presión. Achicá las caídas de presión o separá más las presiones."
            )
    elif inputs.cycle == "flash":
        _validate_flash(inputs, p_valve_out, p_cond_out, T3)
    else:
        _validate_cascade(inputs, p_valve_out, p_cond_out)

    if inputs.reservoirs is not None:
        _validate_reservoirs(inputs.reservoirs, T1, T3)


def _validate_flash(
    inputs: RefrigerationInputs, p_valve_out: float, p_cond_out: float, T3: float
) -> None:
    """La cámara, entre las dos presiones; el líquido tiene que poder evaporarse ahí."""
    p_flash = inputs.p_flash_Pa
    assert p_flash is not None
    if not p_valve_out < p_flash < p_cond_out:
        raise ValueError(
            f"La presión de la cámara ({_bar(p_flash)}) tiene que quedar entre la de entrada al "
            f"evaporador ({_bar(p_valve_out)}) y la de salida del condensador "
            f"({_bar(p_cond_out)}): la válvula de alta baja la presión hasta la cámara y la de "
            "baja, de la cámara al evaporador."
        )
    T_flash = condensation_temperature(inputs.fluid, p_flash)
    if T3 <= T_flash + _T_TOL:
        raise ValueError(
            f"El líquido sale del condensador a {_degC(T3)}, sin superar la temperatura de "
            f"saturación en la cámara ({_degC(T_flash)}): al bajar la presión no se evaporaría "
            "nada y la cámara no tendría vapor que separar. Bajá la presión de la cámara o el "
            "subenfriamiento."
        )


def _validate_cascade(inputs: RefrigerationInputs, p_valve_out: float, p_cond_out: float) -> None:
    """Cada ciclo en orden y el calor pasando del de baja al de alta."""
    casc = inputs.cascade
    assert casc is not None
    low, high = inputs.fluid_low, inputs.fluid
    T_cond_low = _check_condensation(low, casc.p_cond_low_Pa, "de condensación del ciclo de baja")
    T_evap_high = _check_evaporation(high, casc.p_evap_high_Pa, "de evaporación del ciclo de alta")
    if casc.p_cond_low_Pa <= p_valve_out:
        raise ValueError(
            f"El ciclo de baja condensa a {_bar(casc.p_cond_low_Pa)}, sin superar la presión de "
            f"entrada al evaporador ({_bar(p_valve_out)}): su compresor tiene que subir la "
            "presión. Subí la presión de condensación del ciclo de baja."
        )
    if casc.p_evap_high_Pa >= p_cond_out:
        raise ValueError(
            f"El ciclo de alta evapora a {_bar(casc.p_evap_high_Pa)}, sin quedar por debajo de "
            f"la salida del condensador ({_bar(p_cond_out)}): su válvula tiene que bajar la "
            "presión. Bajá la presión de evaporación del ciclo de alta."
        )
    if T_cond_low < T_evap_high - _T_TOL:
        raise ValueError(
            f"En el intercambiador de la cascada, el ciclo de baja condensa a {_degC(T_cond_low)} "
            f"y el de alta evapora a {_degC(T_evap_high)}: el calor tendría que pasar de lo frío "
            "a lo caliente. El de baja tiene que condensar a una temperatura mayor o igual (igual "
            "en el ejemplo 11-4 de Cengel; en la práctica, unos 5 K más)."
        )


def check_reservoirs(result: RefrigerationResult, reservoirs: Reservoirs) -> None:
    """Valida las fuentes contra el ciclo ya resuelto (los mismos mensajes que antes de resolver).

    Lo usa la página de exergía, que agrega las fuentes a un ciclo que no las traía.
    """
    T1 = result.states[result.evaporator.outlet].T_K
    T3 = result.states[result.condenser.outlet].T_K
    _validate_reservoirs(reservoirs, T1, T3)


def default_reservoirs(result: RefrigerationResult) -> Reservoirs:
    """Fuentes por defecto: 5 K arriba de la salida del evaporador y 5 K abajo de la del
    condensador, redondeadas al grado (la regla de la página de refrigeración)."""
    if result.inputs.reservoirs is not None:
        return result.inputs.reservoirs
    T1 = result.states[result.evaporator.outlet].T_K
    T3 = result.states[result.condenser.outlet].T_K
    return Reservoirs(round(T1 - 273.15 + 5.0) + 273.15, round(T3 - 273.15 - 5.0) + 273.15)


def _validate_reservoirs(res: Reservoirs, T1: float, T3: float) -> None:
    """El refrigerante, más frío que la fuente fría en el evaporador y más caliente
    que la fuente caliente en el condensador (si no, el calor iría al revés)."""
    if res.T_cold_K <= 0.0 or res.T_hot_K <= 0.0:
        raise ValueError("Las temperaturas de las fuentes tienen que ser absolutas positivas.")
    if res.T_cold_K >= res.T_hot_K:
        raise ValueError(
            f"La fuente fría ({_degC(res.T_cold_K)}) tiene que estar más fría que la caliente "
            f"({_degC(res.T_hot_K)})."
        )
    if T1 > res.T_cold_K + _T_TOL:
        raise ValueError(
            f"El refrigerante sale del evaporador a {_degC(T1)}, más caliente que la fuente fría "
            f"({_degC(res.T_cold_K)}): para que el calor pase de la fuente al refrigerante, el "
            "refrigerante tiene que estar más frío en todo el evaporador. Bajá la presión del "
            "evaporador o el sobrecalentamiento."
        )
    if T3 < res.T_hot_K - _T_TOL:
        raise ValueError(
            f"El refrigerante sale del condensador a {_degC(T3)}, más frío que la fuente caliente "
            f"({_degC(res.T_hot_K)}): para que el calor pase del refrigerante a la fuente, el "
            "refrigerante tiene que estar más caliente en todo el condensador. Subí la presión "
            "del condensador o bajá el subenfriamiento."
        )


# ---------------------------------------------------------------------
# Red de TESPy
# ---------------------------------------------------------------------


@dataclass
class _Network:
    network: Any
    states: list[Connection]
    objects: dict[str, Any]


def _heat_exchanger_pressure(obj: Any, dp_Pa: float) -> None:
    """Caída de presión de un intercambiador: ``dp`` o, sin pérdidas, ``pr = 1``."""
    if dp_Pa > 0.0:
        obj.set_attr(dp=dp_Pa)
    else:
        obj.set_attr(pr=1.0)


def _evaporator_outlet(conn: Connection, inputs: RefrigerationInputs) -> None:
    """Salida del evaporador: p de aspiración y vapor saturado o sobrecalentado."""
    conn.set_attr(p=inputs.p_evap_Pa, fluid={inputs.fluid_low: 1.0})
    if inputs.superheat_K > 0.0:
        conn.set_attr(td_dew=inputs.superheat_K)  # sobrecalentamiento sobre el rocío
    else:
        conn.set_attr(x=1.0)


def _condenser_outlet(conn: Connection, inputs: RefrigerationInputs) -> None:
    """Salida del condensador: líquido saturado o subenfriado; 1 kg/s de base."""
    conn.set_attr(m=1.0)
    if inputs.subcooling_K > 0.0:
        conn.set_attr(td_bubble=inputs.subcooling_K)  # subenfriamiento bajo la burbuja
    else:
        conn.set_attr(x=0.0)


def _build_network(inputs: RefrigerationInputs) -> _Network:
    """Red de TESPy con 1 kg/s por el condensador (los estados no dependen del caudal)."""
    network = new_network()
    objects: dict[str, Any] = {}

    def make(cls: Any, label: str) -> Any:
        obj = cls(label)
        objects[label] = obj
        return obj

    eta = inputs.eta_compressor
    if inputs.cycle == "simple":
        cp = make(Compressor, "compresor")
        co = make(SimpleHeatExchanger, "condensador")
        va = make(Valve, "válvula de expansión")
        ev = make(SimpleHeatExchanger, "evaporador")
        closer = CycleCloser("cierre del ciclo")
        states = [
            Connection(ev, "out1", cp, "in1", label="1"),
            Connection(cp, "out1", co, "in1", label="2"),
            Connection(co, "out1", va, "in1", label="3"),
            Connection(va, "out1", closer, "in1", label="4"),
        ]
        extra = [Connection(closer, "out1", ev, "in1", label="4 (cierre)")]
        cp.set_attr(eta_s=eta)
        states[1].set_attr(p=inputs.p_cond_Pa)
        cond_out = states[2]
    elif inputs.cycle == "flash":
        cp_lo = make(Compressor, "compresor de baja")
        mixer = make(Merge, "cámara de mezcla")
        cp_hi = make(Compressor, "compresor de alta")
        co = make(SimpleHeatExchanger, "condensador")
        va_hi = make(Valve, "válvula de alta")
        tank = make(DropletSeparator, "cámara de evaporación instantánea")
        va_lo = make(Valve, "válvula de baja")
        ev = make(SimpleHeatExchanger, "evaporador")
        closer = CycleCloser("cierre del ciclo")
        states = [
            Connection(ev, "out1", cp_lo, "in1", label="1"),
            Connection(cp_lo, "out1", mixer, "in1", label="2"),
            Connection(tank, "out2", mixer, "in2", label="3"),  # vapor saturado
            Connection(cp_hi, "out1", co, "in1", label="4"),
            Connection(co, "out1", va_hi, "in1", label="5"),
            Connection(va_hi, "out1", tank, "in1", label="6"),
            Connection(tank, "out1", va_lo, "in1", label="7"),  # líquido saturado
            Connection(va_lo, "out1", closer, "in1", label="8"),
            Connection(mixer, "out1", cp_hi, "in1", label="9"),
        ]
        extra = [Connection(closer, "out1", ev, "in1", label="8 (cierre)")]
        cp_lo.set_attr(eta_s=eta)
        cp_hi.set_attr(eta_s=eta)
        states[1].set_attr(p=inputs.p_flash_Pa)  # la cámara y la mezcla la toman de acá
        states[3].set_attr(p=inputs.p_cond_Pa)
        cond_out = states[4]
    else:
        casc = inputs.cascade
        assert casc is not None
        cp_lo = make(Compressor, "compresor de baja")
        hx = make(HeatExchanger, "intercambiador de la cascada")
        va_lo = make(Valve, "válvula de baja")
        ev = make(SimpleHeatExchanger, "evaporador")
        cp_hi = make(Compressor, "compresor de alta")
        co = make(SimpleHeatExchanger, "condensador")
        va_hi = make(Valve, "válvula de alta")
        closer_lo = CycleCloser("cierre del ciclo de baja")
        closer_hi = CycleCloser("cierre del ciclo de alta")
        states = [
            Connection(ev, "out1", cp_lo, "in1", label="1"),
            Connection(cp_lo, "out1", hx, "in1", label="2"),
            Connection(hx, "out1", va_lo, "in1", label="3"),
            Connection(va_lo, "out1", closer_lo, "in1", label="4"),
            Connection(hx, "out2", cp_hi, "in1", label="5"),
            Connection(cp_hi, "out1", co, "in1", label="6"),
            Connection(co, "out1", va_hi, "in1", label="7"),
            Connection(va_hi, "out1", closer_hi, "in1", label="8"),
        ]
        extra = [
            Connection(closer_lo, "out1", ev, "in1", label="4 (cierre)"),
            Connection(closer_hi, "out1", hx, "in2", label="8 (cierre)"),
        ]
        cp_lo.set_attr(eta_s=eta)
        cp_hi.set_attr(eta_s=eta)
        hx.set_attr(pr1=1.0, pr2=1.0)
        states[1].set_attr(p=casc.p_cond_low_Pa)
        states[2].set_attr(x=0.0)  # el ciclo de baja sale del intercambiador como líquido saturado
        states[4].set_attr(p=casc.p_evap_high_Pa, x=1.0, fluid={inputs.fluid: 1.0})
        states[5].set_attr(p=inputs.p_cond_Pa)
        cond_out = states[6]
    network.add_conns(*states, *extra)
    _heat_exchanger_pressure(ev, inputs.dp_evap_Pa)
    _heat_exchanger_pressure(co, inputs.dp_cond_Pa)
    _evaporator_outlet(states[0], inputs)
    _condenser_outlet(cond_out, inputs)
    return _Network(network=network, states=states, objects=objects)


def _hints(inputs: RefrigerationInputs) -> list[str]:
    if inputs.cycle == "flash":
        return ["Con cámara, revisá que su presión quede bien separada de las otras dos."]
    if inputs.cycle == "cascade":
        return ["En la cascada, revisá las presiones del intercambiador que une los dos ciclos."]
    return []


def solve_refrigeration(
    inputs: RefrigerationInputs, *, compare: bool = True
) -> RefrigerationResult:
    """Resuelve el ciclo con TESPy (Witte & Tuschy, 2020) y arma el resultado.

    Se resuelve con 1 kg/s por el condensador y después se escala al caudal o
    a la capacidad pedidos. Con ``compare`` (y cámara o cascada) también se
    resuelve el ciclo simple equivalente, para comparar el COP.

    Raises
    ------
    ValueError
        Si los datos no tienen sentido físico (ver
        :func:`validate_refrigeration_inputs`) o TESPy no converge.
    """
    validate_refrigeration_inputs(inputs)
    layout = inputs.layout
    built = _build_network(inputs)
    try:
        solve(built.network, what="el ciclo de refrigeración")
    except ValueError as exc:
        hints = _hints(inputs)
        if not hints:
            raise
        raise ValueError(f"{exc} " + " ".join(hints)) from exc

    # De TESPy solo p y h: el estado completo (región, título, v, s) sale de CoolProp.
    states = tuple(
        fluid_state_from_pair(
            layout.fluids[i], "PH", p=float(conn.p.val_SI), h=float(conn.h.val_SI)
        )
        for i, conn in enumerate(built.states)
    )
    flows = tuple(float(conn.m.val_SI) for conn in built.states)
    if any(m <= 0.0 for m in flows):
        raise ValueError(
            "Con esos datos algún tramo del ciclo quedaría con caudal nulo o negativo. Revisá "
            "las presiones."
        )
    compressors_out_s = tuple(
        fluid_state_from_pair(
            layout.fluids[c.outlet],
            "PS",
            p=states[c.outlet].P_Pa,
            s=states[c.inlet].s_J_per_kg_K,
        )
        for c in layout.of_kind("compressor")
    )
    balances: list[tuple[str, float]] = []
    internal: list[tuple[str, float]] = []
    for comp in layout.components:
        obj = built.objects[comp.label]
        if comp.kind == "compressor":
            balances.append((comp.label, float(obj.P.val_SI)))
        elif comp.kind in ("condenser", "evaporator"):
            balances.append((comp.label, float(obj.Q.val_SI)))
        elif comp.kind == "cascade_hx":
            internal.append((comp.label, -float(obj.Q.val_SI)))
    basis = RefrigerationResult(
        inputs=inputs,
        layout=layout,
        states=states,
        m_dot_kg_s=flows,
        compressors_out_s=compressors_out_s,
    )
    if basis.W_W <= 0.0 or basis.Q_cold_W <= 0.0:
        raise ValueError(
            "Con esos datos el ciclo no refrigera (el evaporador no recibe calor). Revisá las "
            "presiones y el subenfriamiento."
        )
    if inputs.m_dot_kg_s is not None:
        scale = inputs.m_dot_kg_s
    else:
        assert inputs.capacity_W is not None
        scale = inputs.capacity_W / basis.useful_heat_W
    single = None
    if compare and inputs.cycle != "simple":
        equivalent = single_stage_equivalent(inputs)
        if equivalent is not None:
            try:
                simple = solve_refrigeration(equivalent, compare=False)
            except ValueError:
                simple = None
            if simple is not None:
                single = SingleStage(
                    fluid=equivalent.fluid,
                    COP=simple.COP,
                    T_discharge_K=simple.T_discharge_K,
                    pressure_ratio=simple.pressure_ratios[0],
                )
    return replace(
        basis,
        m_dot_kg_s=tuple(m * scale for m in flows),
        tespy_balances=tuple((name, value * scale) for name, value in balances),
        tespy_internal=tuple((name, value * scale) for name, value in internal),
        single_stage=single,
    )


def single_stage_equivalent(inputs: RefrigerationInputs) -> RefrigerationInputs | None:
    """Ciclo simple entre las mismas temperaturas de evaporación y condensación.

    Usa el refrigerante del condensador (el del ciclo de alta en la cascada) y
    los mismos rendimiento, sobrecalentamiento, subenfriamiento y caídas de
    presión. ``None`` si el ciclo ya es simple o ese refrigerante no puede
    evaporar a esa temperatura.
    """
    if inputs.cycle == "simple":
        return None
    fluid = inputs.fluid
    p_evap = inputs.p_evap_Pa
    if inputs.fluid_low != fluid:
        T_evap = evaporation_temperature(inputs.fluid_low, inputs.p_evap_Pa)
        lim = fluid_limits(fluid)
        if T_evap <= lim.T_triple_K + 1.0:
            return None
        try:
            p_evap = evaporation_pressure(fluid, T_evap)
        except ValueError:
            return None
    return RefrigerationInputs(
        p_evap,
        inputs.p_cond_Pa,
        inputs.eta_compressor,
        fluid,
        inputs.superheat_K,
        inputs.subcooling_K,
        inputs.dp_evap_Pa,
        inputs.dp_cond_Pa,
        m_dot_kg_s=1.0,
        application=inputs.application,
    )


# ---------------------------------------------------------------------
# Presentación: estados, diagrama y notas
# ---------------------------------------------------------------------


def refrigeration_labeled_states(result: RefrigerationResult) -> list[tuple[str, FluidState]]:
    """``(etiqueta, estado)`` en el orden de la numeración, para la tabla y el export."""
    return list(zip(result.layout.labeled(), result.states, strict=True))


def refrigeration_isentropic_states(
    result: RefrigerationResult,
) -> list[tuple[str, FluidState]]:
    """Salidas isoentrópicas de los compresores con su rótulo («2s», «4s»…)."""
    return [
        (f"{c.outlet + 1}s", state)
        for c, state in zip(result.compressors, result.compressors_out_s, strict=True)
    ]


def refrigeration_diagram_groups(
    result: RefrigerationResult,
) -> list[tuple[str, tuple[int, ...], tuple[tuple[int, int], ...]]]:
    """Qué estados y tramos van en cada diagrama: ``(fluido, estados, tramos)``.

    Un diagrama por refrigerante: la cascada con dos fluidos distintos da dos
    (ciclo de baja y de alta); si no, uno con todo el ciclo.
    """
    groups: list[tuple[str, tuple[int, ...], tuple[tuple[int, int], ...]]] = []
    fluids = result.layout.fluids
    for fluid in dict.fromkeys(fluids):
        indices = tuple(i for i, f in enumerate(fluids) if f == fluid)
        segments = tuple(
            (a, b) for a, b in result.layout.segments if fluids[a] == fluid and fluids[b] == fluid
        )
        groups.append((fluid, indices, segments))
    return groups


def refrigeration_notes(result: RefrigerationResult, system: UnitSystem) -> list[str]:
    """Observaciones didácticas sobre el ciclo calculado (markdown)."""
    notes: list[str] = []
    inputs = result.inputs
    for fluid in dict.fromkeys(result.layout.fluids):
        note = textbook_reference_note(fluid, system)
        if note:
            notes.append(note)
    for label, x in result.wet_compressors:
        fluid = result.layout.fluids[result.layout.one("condenser").inlet]
        notes.append(
            f"La descarga del {label} queda **dentro de la campana** (x = {x:.3f}): la "
            "compresión isoentrópica desde vapor saturado no alcanza a sobrecalentar el vapor "
            "porque la entropía del vapor saturado aumenta con la temperatura (pasa con "
            f"fluidos «secos» como el isobutano; acá, {_el(fluid)}). Los compresores no toleran "
            "gotas de líquido: en la práctica se sobrecalienta el vapor que entra al compresor."
        )
    if result.T_discharge_K > T_DISCHARGE_HIGH_K:
        notes.append(
            f"La temperatura de descarga del compresor es alta ({_degC(result.T_discharge_K)}): "
            "degrada el aceite lubricante y el compresor. Con relaciones de presión grandes se "
            "comprime en dos etapas (cámara de evaporación instantánea o cascada, Cengel §11-8)."
        )
    if inputs.p_evap_Pa < _P_ATM:
        notes.append(
            "El evaporador trabaja por debajo de la presión atmosférica "
            f"({_bar(inputs.p_evap_Pa)}): si hay una fuga, entran aire y humedad al circuito. "
            "Por eso conviene un refrigerante que evapore por encima de 1 atm a la temperatura "
            "buscada (Cengel §11-6)."
        )
    x6 = result.flash_fraction
    if x6 is not None:
        notes.append(
            f"En la cámara se evapora el {x6 * 100:.1f} % del refrigerante (x₆ = {x6:.4f}): ese "
            "vapor va directo al compresor de alta, así que por el evaporador y el compresor de "
            "baja pasa solo 1 − x₆. Además, la mezcla enfría el vapor que sale del compresor de "
            "baja antes de la segunda compresión."
        )
    ratio = result.mass_ratio
    if ratio is not None:
        notes.append(
            f"Por el ciclo de baja circula {ratio:.4g} kg por cada kg del ciclo de alta: lo fija "
            "el balance de energía del intercambiador que los une."
        )
    single = result.single_stage
    if single is not None:
        notes.append(_comparison_note(result, single))
    return notes


def _comparison_note(result: RefrigerationResult, single: SingleStage) -> str:
    """Cámara o cascada contra el ciclo simple equivalente: COP, relación de
    presiones y temperatura de descarga."""
    gain = (result.COP / single.COP - 1.0) * 100.0
    text = (
        f"Un ciclo simple con {_el(single.fluid)} entre las mismas temperaturas de "
        f"evaporación y condensación tendría COP = {single.COP:.3f} (este da "
        f"{result.COP:.3f}, {gain:+.1f} %), con una relación de presiones de "
        f"{single.pressure_ratio:.3g} y la descarga a {_degC(single.T_discharge_K)}; acá cada "
        f"compresor trabaja con {max(result.pressure_ratios):.3g} como máximo y la descarga "
        f"más caliente está a {_degC(result.T_discharge_K)}."
    )
    if gain > 0.5:
        return text + (
            " Comprimir en dos etapas ahorra trabajo cuando la diferencia de temperaturas es "
            "grande (Cengel §11-8)."
        )
    if result.cycle == "cascade":
        return text + (
            " El COP casi no mejora porque la diferencia de temperatura del intercambiador de "
            "la cascada se come la ganancia; las ventajas son otras: relaciones de presión y "
            "temperaturas de descarga razonables y, con dos refrigerantes, cada uno en el "
            "rango de temperaturas donde trabaja mejor."
        )
    return text + " El COP casi no mejora con estas presiones."


# ---------------------------------------------------------------------
# Datos sugeridos y ejemplos
# ---------------------------------------------------------------------


def suggested_refrigeration_inputs(fluid: str) -> RefrigerationInputs:
    """Datos por defecto de un ciclo simple real con ``fluid`` (SI); siempre se resuelven.

    Evaporación a −10 °C y condensación a 40 °C (con el CO₂, que no condensa por
    encima de 31 °C, −35 °C y 0 °C: el ciclo de baja de una cascada), η_C = 0,8,
    5 K de sobrecalentamiento y 3 K de subenfriamiento, para 10 kW de
    refrigeración.
    """
    T_evap, T_cond = (238.15, 273.15) if fluid == "CarbonDioxide" else (263.15, 313.15)
    return RefrigerationInputs(
        round(evaporation_pressure(fluid, T_evap) / 1.0e3) * 1.0e3,
        round(condensation_pressure(fluid, T_cond) / 1.0e3) * 1.0e3,
        eta_compressor=0.8,
        fluid=fluid,
        superheat_K=5.0,
        subcooling_K=3.0,
        m_dot_kg_s=None,
        capacity_W=10.0e3,
    )


def _cengel_11_2() -> RefrigerationInputs:
    """Cengel ej. 11-2 (ciclo real), con los estados del libro.

    1: 0,14 MPa y −10 °C; 2: 0,8 MPa y 50 °C; 3: 0,72 MPa y 26 °C; 4: 0,15 MPa.
    El sobrecalentamiento, el subenfriamiento, las caídas de presión y η_C
    (0,939 en el libro, con sus tablas) salen de esos estados.
    """
    fluid = "R134a"
    p1, p2, p3, p4 = 0.14e6, 0.8e6, 0.72e6, 0.15e6
    T1, T2, T3 = 263.15, 323.15, 299.15
    inlet = fluid_state_from_pair(fluid, "TP", t=T1, p=p1)
    h2s = fluid_state_from_pair(fluid, "PS", p=p2, s=inlet.s_J_per_kg_K).h_J_per_kg
    h2 = fluid_state_from_pair(fluid, "TP", t=T2, p=p2).h_J_per_kg
    return RefrigerationInputs(
        p1,
        p2,
        eta_compressor=(h2s - inlet.h_J_per_kg) / (h2 - inlet.h_J_per_kg),
        fluid=fluid,
        superheat_K=T1 - evaporation_temperature(fluid, p1),
        subcooling_K=condensation_temperature(fluid, p3) - T3,
        dp_evap_Pa=p4 - p1,
        dp_cond_Pa=p2 - p3,
        m_dot_kg_s=0.05,
    )


def _by_temperature(
    fluid: str, T_evap_K: float, T_cond_K: float, fluid_low: str | None = None
) -> tuple[float, float]:
    """Presiones de evaporación (rocío) y condensación (burbuja) para esas T."""
    return evaporation_pressure(fluid_low or fluid, T_evap_K), condensation_pressure(
        fluid, T_cond_K
    )


def _heat_pump_example() -> RefrigerationInputs:
    p_ev, p_co = _by_temperature("R410A", 273.15, 313.15)
    return RefrigerationInputs(
        p_ev,
        p_co,
        eta_compressor=0.7,
        fluid="R410A",
        superheat_K=5.0,
        subcooling_K=3.0,
        m_dot_kg_s=None,
        capacity_W=10.0e3,
        application="heat_pump",
        reservoirs=Reservoirs(280.15, 308.15),
    )


def _co2_nh3_cascade() -> RefrigerationInputs:
    p_ev, p_co = _by_temperature("Ammonia", 228.15, 308.15, fluid_low="CarbonDioxide")
    return RefrigerationInputs(
        p_ev,
        p_co,
        eta_compressor=0.75,
        fluid="Ammonia",
        superheat_K=5.0,
        subcooling_K=2.0,
        m_dot_kg_s=None,
        capacity_W=100.0e3,
        cascade=Cascade(
            condensation_pressure("CarbonDioxide", 268.15),
            evaporation_pressure("Ammonia", 263.15),
            fluid_low="CarbonDioxide",
        ),
        reservoirs=Reservoirs(238.15, 303.15),
    )


def _ammonia_flash() -> RefrigerationInputs:
    p_ev, p_co = _by_temperature("Ammonia", 243.15, 308.15)
    return RefrigerationInputs(
        p_ev,
        p_co,
        eta_compressor=0.75,
        fluid="Ammonia",
        superheat_K=5.0,
        m_dot_kg_s=None,
        capacity_W=100.0e3,
        p_flash_Pa=default_flash_pressure(p_ev, p_co),
    )


def _isobutane_fridge() -> RefrigerationInputs:
    p_ev, p_co = _by_temperature("IsoButane", 248.15, 313.15)
    return RefrigerationInputs(
        p_ev,
        p_co,
        eta_compressor=0.6,
        fluid="IsoButane",
        superheat_K=5.0,
        m_dot_kg_s=None,
        capacity_W=100.0,
        reservoirs=Reservoirs(255.15, 298.15),
    )


_EX_11_1 = "Cengel 11-1 — ideal (R-134a; 0,14 y 0,8 MPa; 0,05 kg/s)"
_EX_11_2 = "Cengel 11-2 — real (R-134a; sobrecalentamiento, subenfriamiento y caídas de presión)"
_EX_11_3 = "Cengel 11-3 — segundo principio (R-134a; espacio a −13 °C, ambiente a 27 °C)"
_EX_11_4 = "Cengel 11-4 — cascada de dos etapas (R-134a; 0,14, 0,32 y 0,8 MPa)"
_EX_11_5 = "Cengel 11-5 — dos etapas con cámara de evaporación instantánea (R-134a)"
_EX_HP = "Bomba de calor aire–agua (R-410A; evapora a 0 °C y condensa a 40 °C)"
_EX_CASCADE = "Cascada CO₂ / amoníaco para congelados (−45 °C)"
_EX_FLASH = "Amoníaco en dos etapas con cámara (−30 °C y 35 °C)"
_EX_FRIDGE = "Heladera con isobutano (R-600a; −25 °C y 40 °C)"

#: Ejemplos precargados (Çengel & Boles, *Termodinámica*, 8.ª ed., cap. 11, y propios).
REFRIGERATION_EXAMPLES: dict[str, RefrigerationInputs] = {
    _EX_11_1: RefrigerationInputs(0.14e6, 0.8e6, m_dot_kg_s=0.05),
    _EX_11_2: _cengel_11_2(),
    _EX_11_3: RefrigerationInputs(
        1.0e5,
        1.0e6,
        eta_compressor=0.85,
        superheat_K=6.4,
        m_dot_kg_s=0.05,
        reservoirs=Reservoirs(260.15, 300.15),
    ),
    _EX_11_4: RefrigerationInputs(0.14e6, 0.8e6, m_dot_kg_s=0.05, cascade=Cascade(0.32e6, 0.32e6)),
    _EX_11_5: RefrigerationInputs(0.14e6, 0.8e6, m_dot_kg_s=1.0, p_flash_Pa=0.32e6),
    _EX_HP: _heat_pump_example(),
    _EX_CASCADE: _co2_nh3_cascade(),
    _EX_FLASH: _ammonia_flash(),
    _EX_FRIDGE: _isobutane_fridge(),
}

#: Ejemplos cuyos niveles se plantean con temperaturas de saturación (los
#: de Cengel, con presiones).
REFRIGERATION_EXAMPLES_BY_TEMPERATURE: frozenset[str] = frozenset(
    {_EX_HP, _EX_CASCADE, _EX_FLASH, _EX_FRIDGE}
)

#: Aclaraciones de cada ejemplo para mostrar en la página.
REFRIGERATION_EXAMPLE_NOTES: dict[str, str] = {
    _EX_11_1: (
        "Q̇_C, Ẇ y el COP coinciden con el libro (7,18 kW, 1,81 kW y 3,97); las h y s difieren "
        "en una constante porque CoolProp usa otra referencia (ver la nota del resultado)."
    ),
    _EX_11_2: (
        "El sobrecalentamiento, el subenfriamiento, las caídas de presión y η_C salen de los "
        "estados del libro: 1 a 0,14 MPa y −10 °C; 2 a 0,8 MPa y 50 °C; 3 a 0,72 MPa y 26 °C; "
        "4 a 0,15 MPa. El libro da η_C = 0,939 con sus tablas; con la ecuación de estado, esos "
        "estados dan 0,937."
    ),
    _EX_11_3: (
        "Ejemplo 11-3 de la 8.ª edición (no está en la 7.ª): el R-134a entra al compresor a "
        "100 kPa sobrecalentado 6,4 °C, η_C = 0,85 y sale del condensador líquido saturado a "
        "39,4 °C (1 MPa). Espacio refrigerado a −13 °C y ambiente a 27 °C."
    ),
    _EX_11_4: (
        "Ejemplo 11-4 de la 8.ª edición (11-3 en la 7.ª). Los dos ciclos intercambian calor a "
        "0,32 MPa: el de baja condensa a la misma temperatura a la que evapora el de alta "
        "(intercambiador ideal). El caudal dado es el del ciclo de alta, 0,05 kg/s. El libro "
        "da COP = 4,46 con sus tablas."
    ),
    _EX_11_5: (
        "Ejemplo 11-5 de la 8.ª edición (11-4 en la 7.ª), con la cámara a 0,32 MPa. Cengel lo "
        "resuelve por kilogramo de refrigerante que pasa por el condensador: acá, ṁ = 1 kg/s."
    ),
    _EX_HP: (
        "Condición de ensayo A7/W35: aire exterior a 7 °C y agua de calefacción a 35 °C. "
        "Compresor con η_C = 0,7, 5 K de sobrecalentamiento y 3 K de subenfriamiento."
    ),
    _EX_CASCADE: (
        "El CO₂ evapora a −45 °C y condensa a −5 °C en el intercambiador, donde el amoníaco "
        "evapora a −10 °C (ΔT = 5 K); el amoníaco condensa a 35 °C. Con un solo ciclo, el "
        "CO₂ no podría condensar a 35 °C (es supercrítico arriba de 31 °C)."
    ),
    _EX_FLASH: (
        "Presión de la cámara en la media geométrica de las otras dos; el barrido de esa "
        "presión muestra dónde está el óptimo."
    ),
    _EX_FRIDGE: (
        "Compresor hermético chico (η_C = 0,6) con 100 W de refrigeración; congelador a −18 °C "
        "y cocina a 25 °C."
    ),
}


# ---------------------------------------------------------------------
# Barridos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    """Un punto del barrido: valor del parámetro (SI) y resultado del ciclo."""

    value_si: float
    COP: float
    COP_carnot: float
    T_discharge_K: float


def _span(lo: float, hi: float, n: int, current: float) -> list[float]:
    if hi <= lo:
        return [current]
    return [float(v) for v in np.linspace(lo, hi, n)]


def default_sweep_values(
    inputs: RefrigerationInputs, parameter: SweepParameter, n: int = 9
) -> list[float]:
    """Valores razonables del parámetro alrededor del ciclo dado (SI).

    Las temperaturas de evaporación y condensación se barren como
    temperaturas de saturación (la presión sale de ellas). En la cascada,
    ``T_cascade`` es la temperatura de condensación del ciclo de baja, con el
    ΔT del intercambiador fijo.
    """
    fluid, low = inputs.fluid, inputs.fluid_low
    T_ev = evaporation_temperature(low, inputs.p_evap_Pa)
    T_co = condensation_temperature(fluid, inputs.p_cond_Pa)
    if parameter == "eta_compressor":
        return [float(v) for v in np.linspace(0.5, 1.0, n)]
    if parameter == "superheat":
        return [float(v) for v in np.linspace(0.0, 20.0, n)]
    if parameter == "subcooling":
        return [float(v) for v in np.linspace(0.0, 15.0, n)]
    lim, lim_low = fluid_limits(fluid), fluid_limits(low)
    if parameter == "T_evap":
        top = T_co - 10.0
        if inputs.p_flash_Pa is not None:
            top = min(top, condensation_temperature(fluid, inputs.p_flash_Pa) - 5.0)
        if inputs.cascade is not None:
            top = min(top, condensation_temperature(low, inputs.cascade.p_cond_low_Pa) - 5.0)
        lo = max(T_ev - 25.0, lim_low.T_triple_K + 5.0)
        return _span(lo, min(T_ev + 15.0, top), n, T_ev)
    if parameter == "T_cond":
        bottom = T_ev + 10.0
        if inputs.p_flash_Pa is not None:
            bottom = max(bottom, condensation_temperature(fluid, inputs.p_flash_Pa) + 5.0)
        if inputs.cascade is not None:
            bottom = max(
                bottom, evaporation_temperature(fluid, inputs.cascade.p_evap_high_Pa) + 5.0
            )
        return _span(max(T_co - 15.0, bottom), min(T_co + 20.0, lim.T_crit_K - 3.0), n, T_co)
    if parameter == "p_flash":
        lo = inputs.p_evap_Pa + inputs.dp_evap_Pa
        hi = inputs.p_cond_Pa - inputs.dp_cond_Pa
        hi = min(hi, condensation_pressure(fluid, _condenser_outlet_T(inputs)))
        if hi <= lo * 1.3:
            return [inputs.p_flash_Pa or default_flash_pressure(lo, hi)]
        return [float(v) for v in np.geomspace(lo * 1.15, hi / 1.15, n)]
    if parameter == "T_cascade":
        casc = inputs.cascade
        if casc is None:
            return []
        dT = condensation_temperature(low, casc.p_cond_low_Pa) - evaporation_temperature(
            fluid, casc.p_evap_high_Pa
        )
        lo = max(T_ev + 8.0, lim.T_triple_K + 5.0 + dT)
        hi = min(T_co - 8.0 + dT, lim_low.T_crit_K - 3.0)
        current = condensation_temperature(low, casc.p_cond_low_Pa)
        return _span(lo, hi, n, current)
    raise ValueError(f"Parámetro de barrido desconocido: {parameter!r}.")


def _sweep_inputs(
    inputs: RefrigerationInputs, parameter: SweepParameter, value: float
) -> RefrigerationInputs:
    fluid, low = inputs.fluid, inputs.fluid_low
    if parameter == "T_evap":
        return replace(inputs, p_evap_Pa=evaporation_pressure(low, value))
    if parameter == "T_cond":
        return replace(inputs, p_cond_Pa=condensation_pressure(fluid, value))
    if parameter == "eta_compressor":
        return replace(inputs, eta_compressor=value)
    if parameter == "superheat":
        return replace(inputs, superheat_K=value)
    if parameter == "subcooling":
        return replace(inputs, subcooling_K=value)
    if parameter == "p_flash":
        return replace(inputs, p_flash_Pa=value)
    casc = inputs.cascade
    assert casc is not None
    dT = condensation_temperature(low, casc.p_cond_low_Pa) - evaporation_temperature(
        fluid, casc.p_evap_high_Pa
    )
    return replace(
        inputs,
        cascade=replace(
            casc,
            p_cond_low_Pa=condensation_pressure(low, value),
            p_evap_high_Pa=evaporation_pressure(fluid, value - dT),
        ),
    )


def refrigeration_sweep(
    inputs: RefrigerationInputs, parameter: SweepParameter, values: Sequence[float]
) -> list[SweepPoint]:
    """Resuelve el ciclo para cada valor del parámetro (como la sección 10 del
    tutorial de bomba de calor de TESPy). Sin las fuentes (el barrido muestra
    el COP); los valores sin sentido físico se omiten."""
    base = replace(inputs, reservoirs=None)
    if parameter == "p_flash" and base.p_flash_Pa is None:
        return []
    if parameter == "T_cascade" and base.cascade is None:
        return []
    points: list[SweepPoint] = []
    for value in values:
        try:
            result = solve_refrigeration(
                _sweep_inputs(base, parameter, float(value)), compare=False
            )
        except ValueError:
            continue
        points.append(SweepPoint(float(value), result.COP, result.COP_carnot, result.T_discharge_K))
    return points


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _value(value_si: float, kind: QuantityKind, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def refrigeration_to_dict(result: RefrigerationResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``)."""
    inputs = result.inputs
    use = "bomba de calor" if inputs.heat_pump else "refrigerador"
    T3 = _condenser_outlet_T(inputs)
    data: dict[str, Any] = {
        "ciclo": f"refrigeración por compresión de vapor {CYCLE_NAMES[inputs.cycle]}"
        + ("" if inputs.is_ideal else ", real"),
        "uso": use,
        "fluido": inputs.fluid,
        "fluido_es": FLUID_NAMES_ES.get(inputs.fluid, inputs.fluid),
        "sistema_de_unidades": system,
        "datos": {
            "p_evaporador": _value(inputs.p_evap_Pa, "pressure", system),
            "T_evaporacion": _value(result.T_evap_K, "temperature", system),
            "p_condensador": _value(inputs.p_cond_Pa, "pressure", system),
            "T_condensacion": _value(result.T_cond_K, "temperature", system),
            "eta_compresor": inputs.eta_compressor,
            "sobrecalentamiento": _value(inputs.superheat_K, "temperature_difference", system),
            "subenfriamiento": _value(inputs.subcooling_K, "temperature_difference", system),
            "dp_evaporador": _value(inputs.dp_evap_Pa, "pressure", system),
            "dp_condensador": _value(inputs.dp_cond_Pa, "pressure", system),
            "T_salida_condensador": _value(T3, "temperature", system),
            "p_camara": (
                None if inputs.p_flash_Pa is None else _value(inputs.p_flash_Pa, "pressure", system)
            ),
            "cascada": (
                None
                if inputs.cascade is None
                else {
                    "fluido_ciclo_de_baja": inputs.fluid_low,
                    "p_condensacion_ciclo_de_baja": _value(
                        inputs.cascade.p_cond_low_Pa, "pressure", system
                    ),
                    "p_evaporacion_ciclo_de_alta": _value(
                        inputs.cascade.p_evap_high_Pa, "pressure", system
                    ),
                }
            ),
            "fuentes": (
                None
                if inputs.reservoirs is None
                else {
                    "T_fria": _value(inputs.reservoirs.T_cold_K, "temperature", system),
                    "T_caliente": _value(inputs.reservoirs.T_hot_K, "temperature", system),
                }
            ),
            "caudal_condensador": (
                None
                if inputs.m_dot_kg_s is None
                else _value(inputs.m_dot_kg_s, "mass_flow", system)
            ),
            "capacidad": (
                None if inputs.capacity_W is None else _value(inputs.capacity_W, "power", system)
            ),
        },
        "resultados": {
            "COP": result.COP,
            "COP_R": result.COP_R,
            "COP_B": result.COP_B,
            "COP_carnot": result.COP_carnot,
            "Q_C": _value(result.Q_cold_W, "power", system),
            "Q_H": _value(result.Q_hot_W, "power", system),
            "W": _value(result.W_W, "power", system),
            "potencia_de_cada_compresor": {
                c.label: _value(w, "power", system)
                for c, w in zip(result.compressors, result.compressor_powers_W, strict=True)
            },
            "q_C": _value(result.q_cold_J_per_kg, _EH, system),
            "q_H": _value(result.q_hot_J_per_kg, _EH, system),
            "w": _value(result.w_J_per_kg, _EH, system),
            "caudal_condensador": _value(result.m_cond_kg_s, "mass_flow", system),
            "caudal_evaporador": _value(result.m_evap_kg_s, "mass_flow", system),
            "toneladas_de_refrigeracion": result.tons_of_refrigeration,
            "T_descarga": _value(result.T_discharge_K, "temperature", system),
            "relaciones_de_presion": list(result.pressure_ratios),
            "titulo_entrada_evaporador": result.x_evaporator_in,
            "caudal_volumetrico_aspiracion": _value(
                result.suction_volume_flow_m3_s, "volume_flow", system
            ),
            "fraccion_evaporada_en_la_camara": result.flash_fraction,
            "relacion_de_caudales_baja_alta": result.mass_ratio,
            "calor_del_intercambiador_de_la_cascada": (
                _value(result.Q_cascade_W, "power", system) if result.cycle == "cascade" else None
            ),
            "COP_ciclo_simple_equivalente": result.single_stage_COP,
        },
        "segundo_principio": None,
        "estados": states_table(refrigeration_labeled_states(result), system),
        "fuente": "TESPy 0.11 (red de componentes) + CoolProp (ecuación de estado de Helmholtz)",
    }
    exergy = result.exergy
    if exergy is not None:
        data["segundo_principio"] = {
            "T0": _value(exergy.T0_K, "temperature", system),
            "COP_reversible": result.COP_rev,
            "W_minimo": _value(exergy.W_min_W, "power", system),
            "exergia_destruida": {name: _value(x, "power", system) for name, x in exergy.destroyed},
            "exergia_destruida_total": _value(exergy.X_destroyed_W, "power", system),
            "rendimiento_exergetico": exergy.eta_ex,
        }
    return data
