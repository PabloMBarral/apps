"""Turbina de gas con interenfriamiento, recalentamiento y regenerador — Fase 3.7.

Generaliza la turbina de gas de la Fase 3.4 (:mod:`core.cycles.brayton`, que
sigue siendo la del ciclo combinado) con las tres mejoras del ciclo Brayton
(Çengel & Boles, *Termodinámica*, §9-9 y §9-10):

- **Compresión en etapas con interenfriamiento**: entre una etapa y la
  siguiente, el aire se enfría a presión constante (hasta T₁, o hasta otra
  temperatura). Con relaciones de presión iguales en cada etapa el trabajo es
  mínimo: con dos etapas, p_x = √(p₁·p₂) (vademecum §6.3).
- **Expansión en etapas con recalentamiento**: entre una etapa y la siguiente
  los gases se vuelven a calentar hasta la TIT (o hasta otra temperatura). Con
  combustible, el recalentamiento es una **segunda cámara de combustión** que
  quema en los gases que salen de la primera turbina, que todavía tienen
  oxígeno (la *combustión secuencial* de las GT24/GT26): la composición de los
  gases cambia en cada cámara. Con aire estándar es un intercambiador.
- **Regenerador**: los gases que salen de la última turbina calientan el aire
  que sale del último compresor. Su efectividad es la de Cengel §9-9,
  ε = (h₅ − h₂)/(h_a(T₄) − h₂), con h_a la entalpía del aire: el aire se
  podría calentar, como mucho, hasta la temperatura de los gases que entran.

El aire y los gases son mezclas de gases ideales (:mod:`core.ideal_gas`), con
las entalpías desde 25 °C; cada cámara cumple
(1 + F_ant)·h_ent + Δf·PCI = (1 + F)·h_g(T_sal), con F la relación
combustible/aire acumulada. Con combustión y regenerador, f depende de T₅ y
T₄ de f: se itera hasta que la temperatura del aire que sale del regenerador
no cambia.

Los estados se numeran como Cengel: 1–4 en el ciclo simple, 1–6 con
regenerador (5 y 6 son las salidas del regenerador, figura 9-38) y en el
sentido del flujo cuando hay etapas (1–10 en el ejemplo 9-8, figura 9-43). Los
isoentrópicos llevan una «s» (2s, 4s…).

También: el balance de exergía por componente (vademecum §11; la exergía del
combustible es ≈ PCI, §16.13), la comparación de las mejoras, notas,
ejemplos, barridos, export y :func:`gas_turbine_tespy` (la misma turbina con
TESPy, como control cruzado). Con una etapa y sin regenerador reproduce
:func:`core.cycles.brayton.solve_brayton`. Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq

from core.cycles.brayton import (
    _TESPY_FUEL_NAMES,
    METHANE,
    T_FUEL_K,
    BraytonInputs,
    Fuel,
    GasState,
    _iso_tables,
    _stoichiometric_fuel_air_ratio,
    validate_brayton_inputs,
)
from core.ideal_gas import AIR_DRY, GAS_NAMES, T_GAS_MAX_K, FlueGas, combustion_products
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "DEFAULT_EFFECTIVENESS",
    "GAS_TURBINE_EXAMPLES",
    "GAS_TURBINE_EXAMPLE_NOTES",
    "GAS_TURBINE_SWEEP_PARAMETERS",
    "MAX_STAGES",
    "Combustor",
    "CycleLine",
    "CycleState",
    "ExergyItem",
    "GasTurbineExergy",
    "GasTurbineInputs",
    "GasTurbineResult",
    "GasTurbineSweepPoint",
    "GasTurbineTespy",
    "ImprovementRow",
    "Intercooler",
    "Regenerator",
    "Stage",
    "default_gas_turbine_sweep_values",
    "from_brayton",
    "gas_turbine_exergy",
    "gas_turbine_lines",
    "gas_turbine_notes",
    "gas_turbine_sweep",
    "gas_turbine_tespy",
    "gas_turbine_to_dict",
    "improvement_comparison",
    "of_component",
    "solve_gas_turbine",
    "stage_names",
    "validate_gas_turbine_inputs",
]

#: Cantidad máxima de etapas de compresión o de expansión del modelo.
MAX_STAGES = 8
#: Efectividad del regenerador en la comparación si los datos no tienen (Cengel 9-7).
DEFAULT_EFFECTIVENESS = 0.80
#: TIT por encima de la cual las máquinas reales refrigeran los álabes (nota).
_T_BLADE_COOLING_K = 1373.15
_TOL_T_K = 1e-9


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _bar(p_Pa: float) -> str:
    return f"{p_Pa / 1e5:.4g} bar"


# ---------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class GasTurbineInputs:
    """Datos de la turbina de gas con etapas y regenerador, en SI.

    Los primeros campos son los de :class:`core.cycles.brayton.BraytonInputs`.
    ``pressure_ratio`` es la del compresor completo (descarga/entrada) y los
    rendimientos isoentrópicos son los de cada etapa. ``dp_combustor`` vale
    para cada cámara (y cada recalentador); ``dp_intercooler`` para cada
    interenfriador y ``dp_regenerator`` para cada lado del regenerador, todas
    como fracción de la presión de entrada.

    - ``T_intercool_K``: temperatura del aire a la salida de cada
      interenfriador (``None``: la del ambiente, interenfriamiento perfecto).
    - ``T_reheat_K``: la de los gases a la salida de cada recalentamiento
      (``None``: la TIT).
    - ``regenerator``: efectividad ε (``None``: sin regenerador).
    - ``p_intercool_Pa`` / ``p_reheat_Pa``: las presiones de los
      interenfriadores (salida de cada etapa de compresión menos la última) y
      de los recalentamientos (salida de cada etapa de la turbina menos la
      última). Vacías: relaciones de presión iguales en cada etapa.
    - ``W_net_W``: si se da, el caudal de aire sale de la potencia neta.
    """

    pressure_ratio: float
    T_turbine_in_K: float
    eta_compressor: float = 1.0
    eta_turbine: float = 1.0
    T_amb_K: float = 288.15
    p_amb_Pa: float = 101_325.0
    dp_combustor: float = 0.0
    air: FlueGas = AIR_DRY
    fuel: Fuel | None = METHANE
    m_air_kg_s: float = 1.0
    compressor_stages: int = 1
    turbine_stages: int = 1
    T_intercool_K: float | None = None
    T_reheat_K: float | None = None
    dp_intercooler: float = 0.0
    regenerator: float | None = None
    dp_regenerator: float = 0.0
    p_intercool_Pa: tuple[float, ...] = ()
    p_reheat_Pa: tuple[float, ...] = ()
    W_net_W: float | None = None

    @property
    def air_standard(self) -> bool:
        return self.fuel is None

    @property
    def intercooled(self) -> bool:
        return self.compressor_stages > 1

    @property
    def reheated(self) -> bool:
        return self.turbine_stages > 1

    @property
    def regenerated(self) -> bool:
        return self.regenerator is not None

    @property
    def T_intercool(self) -> float:
        """Temperatura a la salida de los interenfriadores (K)."""
        return self.T_amb_K if self.T_intercool_K is None else self.T_intercool_K

    @property
    def T_reheat(self) -> float:
        """Temperatura a la salida de los recalentamientos (K)."""
        return self.T_turbine_in_K if self.T_reheat_K is None else self.T_reheat_K

    @property
    def is_ideal(self) -> bool:
        return (
            self.eta_compressor == 1.0
            and self.eta_turbine == 1.0
            and self.dp_combustor == 0.0
            and self.dp_intercooler == 0.0
            and self.dp_regenerator == 0.0
        )

    @property
    def configuration(self) -> str:
        """Nombre de la configuración: «simple», «con regenerador», «con interenfriamiento»…"""
        parts = []
        if self.intercooled:
            parts.append("interenfriamiento")
        if self.reheated:
            parts.append("recalentamiento")
        if self.regenerated:
            parts.append("regenerador")
        if not parts:
            return "simple"
        if len(parts) == 1:
            return f"con {parts[0]}"
        return "con " + ", ".join(parts[:-1]) + " y " + parts[-1]

    def brayton(self) -> BraytonInputs:
        """Los datos que tiene en común con la turbina de gas de la Fase 3.4."""
        return BraytonInputs(
            pressure_ratio=self.pressure_ratio,
            T_turbine_in_K=self.T_turbine_in_K,
            eta_compressor=self.eta_compressor,
            eta_turbine=self.eta_turbine,
            T_amb_K=self.T_amb_K,
            p_amb_Pa=self.p_amb_Pa,
            dp_combustor=self.dp_combustor,
            air=self.air,
            fuel=self.fuel,
            m_air_kg_s=self.m_air_kg_s,
        )


def from_brayton(inputs: BraytonInputs) -> GasTurbineInputs:
    """La turbina de gas de la Fase 3.4 en el modelo nuevo (una etapa, sin regenerador)."""
    return GasTurbineInputs(
        pressure_ratio=inputs.pressure_ratio,
        T_turbine_in_K=inputs.T_turbine_in_K,
        eta_compressor=inputs.eta_compressor,
        eta_turbine=inputs.eta_turbine,
        T_amb_K=inputs.T_amb_K,
        p_amb_Pa=inputs.p_amb_Pa,
        dp_combustor=inputs.dp_combustor,
        air=inputs.air,
        fuel=inputs.fuel,
        m_air_kg_s=inputs.m_air_kg_s,
    )


def of_component(name: str) -> str:
    """«del compresor», «de la cámara de combustión»: el nombre con su artículo."""
    return f"de la {name}" if name.startswith(("cámara", "turbina")) else f"del {name}"


def stage_names(kind: Literal["compresor", "turbina"], count: int) -> list[str]:
    """«compresor», o «compresor de baja», «… de media», «… de alta» (turbina: de alta a baja)."""
    if count == 1:
        return [kind]
    if count == 2:
        levels = ["baja", "alta"]
    elif count == 3:
        levels = ["baja", "media", "alta"]
    else:
        return [f"{kind} {k + 1}" for k in range(count)]
    if kind == "turbina":
        levels = levels[::-1]
    return [f"{kind} de {lv}" for lv in levels]


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CycleState(GasState):
    """Un estado del ciclo: el de la turbina de gas, con su mezcla y una descripción."""

    mix: FlueGas
    description: str

    def psi(self, T0_K: float, p0_Pa: float) -> float:
        """Exergía de flujo física (J/kg de la corriente), contra la misma mezcla a T₀ y p₀.

        ψ = (h − h₀) − T₀·(s − s₀) (vademecum §11.3).
        """
        mix = self.mix
        return (self.h_J_per_kg - mix.h(T0_K)) - T0_K * (self.s_J_per_kg_K - mix.s(T0_K, p0_Pa))


@dataclass(frozen=True)
class Stage:
    """Una etapa de compresión o de expansión (índices en ``GasTurbineResult.states``).

    ``m`` son los kg que pasan por kg de aire: 1 en el compresor y 1 + F en
    cada turbina (F, el combustible quemado hasta ahí).
    """

    kind: Literal["compresor", "turbina"]
    name: str
    inlet: int
    outlet_s: int
    outlet: int
    m: float


@dataclass(frozen=True)
class Intercooler:
    """Un interenfriador: enfría el aire a presión (casi) constante entre dos etapas."""

    name: str
    inlet: int
    outlet: int


@dataclass(frozen=True)
class Combustor:
    """La cámara de combustión o un recalentamiento (con aire estándar, un calentador).

    ``fuel`` es el combustible que se quema acá por kg de aire; ``F_in`` y
    ``F_out``, el acumulado antes y después; ``m_in``, los kg que entran por kg
    de aire (1 + F_in). ``q`` es el calor que entra por kg de aire (Δf·PCI, o
    h_sal − h_ent con aire estándar).
    """

    name: str
    inlet: int
    outlet: int
    fuel: float
    F_in: float
    F_out: float
    q: float

    @property
    def m_in(self) -> float:
        return 1.0 + self.F_in

    @property
    def m_out(self) -> float:
        return 1.0 + self.F_out


@dataclass(frozen=True)
class Regenerator:
    """El regenerador: el aire (2 → 5) se calienta con los gases (4 → 6), por kg de aire."""

    effectiveness: float
    air_in: int
    air_out: int
    gas_in: int
    gas_out: int
    m_gas: float
    q: float
    q_max: float


@dataclass(frozen=True)
class GasTurbineResult:
    """Estados, componentes y balances de la turbina de gas, en SI.

    Las energías específicas van por kilogramo de **aire** (como en la Fase
    3.4); las potencias se escalan con el caudal de aire.
    """

    inputs: GasTurbineInputs
    states: tuple[CycleState, ...]
    compressors: tuple[Stage, ...]
    intercoolers: tuple[Intercooler, ...]
    combustors: tuple[Combustor, ...]
    turbines: tuple[Stage, ...]
    regenerator: Regenerator | None
    exhaust: int
    gas: FlueGas
    fuel_air_ratio: float
    excess_air: float | None

    def state(self, label: str) -> CycleState:
        return next(s for s in self.states if s.label == label)

    def h(self, i: int) -> float:
        return self.states[i].h_J_per_kg

    # --- trabajos y calores, por kg de aire --------------------------------
    def stage_work(self, stage: Stage) -> float:
        """Trabajo de una etapa por kg de aire (positivo): m·|h_sal − h_ent|."""
        return stage.m * abs(self.h(stage.outlet) - self.h(stage.inlet))

    def intercooler_heat(self, ic: Intercooler) -> float:
        """Calor que saca un interenfriador por kg de aire."""
        return self.h(ic.inlet) - self.h(ic.outlet)

    @property
    def w_compressor_J_per_kg(self) -> float:
        return sum(self.stage_work(c) for c in self.compressors)

    @property
    def w_turbine_J_per_kg(self) -> float:
        return sum(self.stage_work(t) for t in self.turbines)

    @property
    def w_net_J_per_kg(self) -> float:
        return self.w_turbine_J_per_kg - self.w_compressor_J_per_kg

    @property
    def q_in_J_per_kg(self) -> float:
        """Calor que entra por kg de aire: F·PCI (combustión) o la suma de los calentadores."""
        return sum(c.q for c in self.combustors)

    @property
    def q_intercoolers_J_per_kg(self) -> float:
        return sum(self.intercooler_heat(ic) for ic in self.intercoolers)

    @property
    def q_exhaust_J_per_kg(self) -> float:
        """Lo que se lleva el escape, contra el aire a T₁: (1 + F)·h_g(T_esc) − h_a(T₁)."""
        exh = self.states[self.exhaust]
        return (1.0 + self.fuel_air_ratio) * exh.h_J_per_kg - self.states[0].h_J_per_kg

    @property
    def q_regenerator_J_per_kg(self) -> float:
        return 0.0 if self.regenerator is None else self.regenerator.q

    @property
    def eta_th(self) -> float:
        return self.w_net_J_per_kg / self.q_in_J_per_kg

    @property
    def back_work_ratio(self) -> float:
        """Relación de trabajo de retroceso w_C / w_T (Cengel §9-8)."""
        return self.w_compressor_J_per_kg / self.w_turbine_J_per_kg

    @property
    def heat_rate_kJ_per_kWh(self) -> float:
        """Consumo específico de calor: 3600 / η (kJ por kWh producido)."""
        return 3600.0 / self.eta_th

    @property
    def T_exhaust_K(self) -> float:
        return self.states[self.exhaust].T_K

    @property
    def T_min_K(self) -> float:
        return min(s.T_K for s in self.states)

    @property
    def T_max_K(self) -> float:
        return max(s.T_K for s in self.states)

    @property
    def eta_carnot(self) -> float:
        """Carnot entre la menor y la mayor temperatura del ciclo (Cengel §9-10, Ericsson)."""
        return 1.0 - self.T_min_K / self.T_max_K

    @property
    def energy_residual_J_per_kg(self) -> float:
        """q_ent − w_neto − q_interenfriadores − q_escape (tiene que dar ~0)."""
        return (
            self.q_in_J_per_kg
            - self.w_net_J_per_kg
            - self.q_intercoolers_J_per_kg
            - self.q_exhaust_J_per_kg
        )

    # --- caudales y potencias (W) ---------------------------------------
    @property
    def m_air_kg_s(self) -> float:
        return self.inputs.m_air_kg_s

    @property
    def m_fuel_kg_s(self) -> float:
        return self.m_air_kg_s * self.fuel_air_ratio

    @property
    def m_gas_kg_s(self) -> float:
        return self.m_air_kg_s + self.m_fuel_kg_s

    @property
    def W_compressor_W(self) -> float:
        return self.m_air_kg_s * self.w_compressor_J_per_kg

    @property
    def W_turbine_W(self) -> float:
        return self.m_air_kg_s * self.w_turbine_J_per_kg

    @property
    def W_net_W(self) -> float:
        return self.m_air_kg_s * self.w_net_J_per_kg

    @property
    def Q_in_W(self) -> float:
        return self.m_air_kg_s * self.q_in_J_per_kg

    @property
    def Q_intercoolers_W(self) -> float:
        return self.m_air_kg_s * self.q_intercoolers_J_per_kg

    @property
    def Q_exhaust_W(self) -> float:
        return self.m_air_kg_s * self.q_exhaust_J_per_kg

    @property
    def Q_regenerator_W(self) -> float:
        return self.m_air_kg_s * self.q_regenerator_J_per_kg


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


def _check_stages(count: int, what: str) -> None:
    if not (isinstance(count, int) and 1 <= count <= MAX_STAGES):
        raise ValueError(
            f"La cantidad de etapas de {what} ({count}) tiene que ser un entero entre 1 y "
            f"{MAX_STAGES}."
        )


def validate_gas_turbine_inputs(inputs: GasTurbineInputs) -> None:
    """Verifica que la turbina de gas tenga sentido físico antes de calcular.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno: qué dato está fuera de rango y por qué.
    """
    validate_brayton_inputs(inputs.brayton())
    _check_stages(inputs.compressor_stages, "compresión")
    _check_stages(inputs.turbine_stages, "expansión")
    for name, dp in (
        ("en cada interenfriador", inputs.dp_intercooler),
        ("en cada lado del regenerador", inputs.dp_regenerator),
    ):
        if not (math.isfinite(dp) and 0.0 <= dp < 0.2):
            raise ValueError(
                f"La caída de presión {name} tiene que estar entre 0 y 20 % (típico: 1 a 4 %)."
            )
    if inputs.intercooled:
        T_ic = inputs.T_intercool
        if not (math.isfinite(T_ic) and 223.15 <= T_ic <= 773.15):
            raise ValueError(
                f"La temperatura del aire después de interenfriar ({_degC(T_ic)}) tiene que "
                "estar entre −50 y 500 °C (lo habitual: la del ambiente, o unos grados más)."
            )
    if inputs.reheated:
        T_rh = inputs.T_reheat
        if not (math.isfinite(T_rh) and T_rh <= T_GAS_MAX_K):
            raise ValueError(
                f"La temperatura de recalentamiento ({_degC(T_rh)}) supera el rango del modelo "
                f"({_degC(T_GAS_MAX_K)})."
            )
    if inputs.regenerator is not None:
        eps = inputs.regenerator
        if not (math.isfinite(eps) and 0.0 < eps <= 1.0):
            raise ValueError(
                f"La efectividad del regenerador ({eps:g}) tiene que estar entre 0 y 1 (1 = "
                "regenerador ideal; los reales andan entre 0,6 y 0,9, Cengel §9-9)."
            )
    if inputs.W_net_W is not None and not (math.isfinite(inputs.W_net_W) and inputs.W_net_W > 0):
        raise ValueError("La potencia neta tiene que ser positiva.")
    _compressor_pressures(inputs)
    _turbine_pressures(inputs)


def _compressor_pressures(inputs: GasTurbineInputs) -> list[tuple[float, float]]:
    """(p_entrada, p_salida) de cada etapa de compresión.

    Sin presiones dadas, la relación de cada etapa es la misma:
    r_e = [r_p / (1 − Δp_IC)^(N−1)]^(1/N), así la descarga queda en r_p·p₁.
    """
    n = inputs.compressor_stages
    p1 = inputs.p_amb_Pa
    p2 = p1 * inputs.pressure_ratio
    k_ic = 1.0 - inputs.dp_intercooler
    given = inputs.p_intercool_Pa
    if given:
        if len(given) != n - 1:
            raise ValueError(
                f"Con {n} etapas de compresión hacen falta {n - 1} presiones de "
                f"interenfriamiento (se dieron {len(given)})."
            )
        outs = [float(p) for p in given] + [p2]
    else:
        r = (inputs.pressure_ratio / k_ic ** (n - 1)) ** (1.0 / n)
        outs = []
        p = p1
        for _ in range(n):
            p *= r
            outs.append(p)
            p *= k_ic
        outs[-1] = p2
    pairs = []
    p_in = p1
    for k, p_out in enumerate(outs):
        if not (math.isfinite(p_out) and p_out > p_in * (1.0 + 1e-9)):
            raise ValueError(
                f"Las presiones de interenfriamiento tienen que ir creciendo entre p₁ "
                f"({_bar(p1)}) y la descarga ({_bar(p2)}): la etapa {k + 1} del compresor "
                f"entraría a {_bar(p_in)} y saldría a {_bar(p_out)}."
            )
        pairs.append((p_in, p_out))
        p_in = p_out * k_ic
    return pairs


def _turbine_pressures(inputs: GasTurbineInputs) -> list[tuple[float, float]]:
    """(p_entrada, p_salida) de cada etapa de la turbina.

    La primera entra a p₂·(1 − Δp_reg)·(1 − Δp_cám) y la última sale a
    p₁/(1 − Δp_reg) (el escape todavía cruza el regenerador). Sin presiones
    dadas, la relación de expansión de cada etapa es la misma.
    """
    n = inputs.turbine_stages
    p2 = inputs.p_amb_Pa * inputs.pressure_ratio
    k_cc = 1.0 - inputs.dp_combustor
    k_rg = 1.0 - inputs.dp_regenerator if inputs.regenerated else 1.0
    p_in = p2 * k_rg * k_cc
    p_exh = inputs.p_amb_Pa / k_rg
    given = inputs.p_reheat_Pa
    if given:
        if len(given) != n - 1:
            raise ValueError(
                f"Con {n} etapas de expansión hacen falta {n - 1} presiones de "
                f"recalentamiento (se dieron {len(given)})."
            )
        outs = [float(p) for p in given] + [p_exh]
    else:
        r = (p_in * k_cc ** (n - 1) / p_exh) ** (1.0 / n)
        outs = []
        p = p_in
        for _ in range(n):
            p /= r
            outs.append(p)
            p *= k_cc
        outs[-1] = p_exh
    pairs = []
    p_first = p_in
    for k, p_out in enumerate(outs):
        if not (math.isfinite(p_out) and p_out < p_in * (1.0 - 1e-9)):
            raise ValueError(
                f"Las presiones de recalentamiento tienen que ir bajando entre la entrada a la "
                f"turbina ({_bar(p_first)}) y el escape ({_bar(p_exh)}): la etapa {k + 1} de la "
                f"turbina entraría a {_bar(p_in)} y saldría a {_bar(p_out)}."
            )
        pairs.append((p_in, p_out))
        p_in = p_out * k_cc
    return pairs


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _gas_for(inputs: GasTurbineInputs, F: float) -> FlueGas:
    """Los gases después de quemar F kg de combustible por kg de aire (aire si F = 0)."""
    fuel = inputs.fuel
    if fuel is None or F <= 0.0:
        return inputs.air
    f_st = _stoichiometric_fuel_air_ratio(inputs.brayton())
    gas, _ = combustion_products(inputs.air, fuel.atoms, max(1.0, f_st / F))
    return FlueGas(gas.y, inputs.p_amb_Pa)


def _burn(inputs: GasTurbineInputs, F_in: float, h_in: float, T_out: float, where: str) -> float:
    """Combustible que hay que quemar (por kg de aire) para llevar 1 + F_in kg a T_out.

    (1 + F_in)·h_ent + Δf·PCI = (1 + F_in + Δf)·h_g(T_sal), con las entalpías
    desde 25 °C (vademecum §16.7). La composición de los gases depende de F,
    así que h_g también: se busca Δf con ``brentq``.
    """
    fuel = inputs.fuel
    assert fuel is not None
    lhv = fuel.lhv_J_per_kg
    f_st = _stoichiometric_fuel_air_ratio(inputs.brayton())

    def residual(df: float) -> float:
        F = F_in + df
        return (1.0 + F_in) * h_in + df * lhv - (1.0 + F) * _gas_for(inputs, F).h(T_out)

    if F_in >= f_st or residual(f_st - F_in) <= 0.0:
        raise ValueError(
            f"{where}: para llegar a {_degC(T_out)} haría falta quemar más combustible que el "
            "que admite el oxígeno que queda (λ < 1, combustión incompleta). Bajá esa "
            "temperatura."
        )
    return float(brentq(residual, 1e-12 * f_st, f_st - F_in, xtol=1e-15, rtol=1e-14))


@dataclass
class _Hot:
    """El lado caliente para una temperatura del aire que entra a la cámara."""

    combustors: list[tuple[str, float, float, float, float, float]] = field(default_factory=list)
    turbines: list[dict[str, Any]] = field(default_factory=list)
    F: float = 0.0


def _hot_side(inputs: GasTurbineInputs, T_cc_in: float, p_cc_in: float) -> _Hot:
    """Cámara, etapas de la turbina y recalentamientos desde el aire a T_cc_in.

    Devuelve, por cámara, (nombre, T_ent, p_ent, h_ent, Δf, F_ent) y, por etapa,
    las temperaturas, presiones y entalpías de entrada, isoentrópica y salida.
    """
    air = inputs.air
    TIT = inputs.T_turbine_in_K
    if TIT <= T_cc_in + 1.0:
        raise ValueError(
            f"La temperatura de entrada a la turbina ({_degC(TIT)}) tiene que superar la del "
            f"aire que llega a la cámara ({_degC(T_cc_in)}): la cámara calienta el aire. Subí "
            "la TIT o bajá la relación de presiones."
        )
    pressures = _turbine_pressures(inputs)
    names = stage_names("turbina", inputs.turbine_stages)
    hot = _Hot()
    F = 0.0
    T_in, p_in, h_in = T_cc_in, p_cc_in, air.h(T_cc_in)
    T_target = TIT
    for j, (p_t_in, p_t_out) in enumerate(pressures):
        where = "Cámara de combustión" if j == 0 else f"Recalentamiento antes de la {names[j]}"
        if j > 0 and T_target <= T_in + 1.0:
            raise ValueError(
                f"El recalentamiento hasta {_degC(T_target)} no calienta: los gases salen de la "
                f"{names[j - 1]} a {_degC(T_in)}. Subí la temperatura de recalentamiento o bajá "
                "la presión a la que se recalienta."
            )
        if inputs.fuel is None:
            df = 0.0
        else:
            df = _burn(inputs, F, h_in, T_target, where)
        hot.combustors.append((where, T_in, p_in, h_in, df, F))
        F += df
        gas = _gas_for(inputs, F)
        h_t_in = gas.h(T_target)
        T_s = gas.T_isentropic(T_target, p_t_in, p_t_out)
        h_s = gas.h(T_s)
        h_out = h_t_in - inputs.eta_turbine * (h_t_in - h_s)
        T_out = gas.T_from_h(h_out)
        hot.turbines.append(
            {
                "T_in": T_target,
                "p_in": p_t_in,
                "h_in": h_t_in,
                "T_s": T_s,
                "h_s": h_s,
                "T_out": T_out,
                "p_out": p_t_out,
                "h_out": h_out,
                "F": F,
                "gas": gas,
            }
        )
        T_in, p_in, h_in = T_out, p_t_out, h_out
        T_target = inputs.T_reheat
    hot.F = F
    return hot


def _numbering(n_c: int, n_t: int, regenerated: bool) -> dict[tuple[str, int], int]:
    """Número de Cengel de cada estado real: (rol, índice) → número.

    Roles: ("c_in", k), ("c_out", k), ("t_in", j), ("t_out", j), ("r_air", 0),
    ("r_gas", 0). Con una etapa y regenerador, 5 y 6 son las salidas del
    regenerador (Cengel figura 9-38); si no, en el sentido del flujo (figura 9-43).
    """
    if n_c == 1 and n_t == 1 and regenerated:
        return {
            ("c_in", 0): 1,
            ("c_out", 0): 2,
            ("t_in", 0): 3,
            ("t_out", 0): 4,
            ("r_air", 0): 5,
            ("r_gas", 0): 6,
        }
    order: list[tuple[str, int]] = []
    for k in range(n_c):
        order += [("c_in", k), ("c_out", k)]
    if regenerated:
        order.append(("r_air", 0))
    for j in range(n_t):
        order += [("t_in", j), ("t_out", j)]
    if regenerated:
        order.append(("r_gas", 0))
    return {role: i + 1 for i, role in enumerate(order)}


def solve_gas_turbine(inputs: GasTurbineInputs) -> GasTurbineResult:  # noqa: PLR0915
    """Turbina de gas con interenfriamiento, recalentamiento y regenerador (Cengel §9-8 a §9-10).

    Raises
    ------
    ValueError
        Si los datos no tienen sentido (:func:`validate_gas_turbine_inputs`), un
        interenfriador no enfría, un recalentamiento no calienta, una cámara
        pediría λ < 1 o el regenerador no sirve (T₄ ≤ T₂).
    """
    validate_gas_turbine_inputs(inputs)
    air = inputs.air
    T1 = inputs.T_amb_K
    T_ic = inputs.T_intercool
    k_rg = 1.0 - inputs.dp_regenerator if inputs.regenerated else 1.0
    c_names = stage_names("compresor", inputs.compressor_stages)
    t_names = stage_names("turbina", inputs.turbine_stages)

    # --- compresión (s°(T₂s) = s°(T₁) + R·ln(p₂/p₁), vademecum §10.5) -------
    comp: list[dict[str, float]] = []
    T_in = T1
    for k, (p_in, p_out) in enumerate(_compressor_pressures(inputs)):
        h_in = air.h(T_in)
        T_s = air.T_isentropic(T_in, p_in, p_out)
        h_s = air.h(T_s)
        h_out = h_in + (h_s - h_in) / inputs.eta_compressor
        T_out = air.T_from_h(h_out)
        comp.append(
            {"T_in": T_in, "p_in": p_in, "h_in": h_in, "T_s": T_s, "h_s": h_s, "T_out": T_out}
            | {"p_out": p_out, "h_out": h_out}
        )
        if k < inputs.compressor_stages - 1:
            if T_ic >= T_out - 1.0:
                raise ValueError(
                    f"El interenfriamiento hasta {_degC(T_ic)} no enfría: el aire sale del "
                    f"{c_names[k]} a {_degC(T_out)}. Bajá la temperatura después de interenfriar "
                    "o subí la presión intermedia."
                )
            T_in = T_ic
    c_last = comp[-1]
    T2, p2, h2 = c_last["T_out"], c_last["p_out"], c_last["h_out"]

    # --- lado caliente, con el regenerador iterado ------------------------
    eps = inputs.regenerator
    p_cc_in = p2 * k_rg

    def regenerated_T(hot: _Hot) -> float:
        T4 = hot.turbines[-1]["T_out"]
        if T4 <= T2 + 1e-6:
            raise ValueError(
                f"El regenerador no sirve: los gases salen de la turbina a {_degC(T4)} y el aire "
                f"sale del compresor a {_degC(T2)}, más caliente. El regenerador solo calienta el "
                "aire si T₄ > T₂ (Cengel §9-9): bajá la relación de presiones o sacá el "
                "regenerador."
            )
        assert eps is not None
        return air.T_from_h(h2 + eps * (air.h(T4) - h2))

    if eps is None:
        T_cc_in = T2
        hot = _hot_side(inputs, T_cc_in, p_cc_in)
    else:
        # Punto fijo T₅ = g(T₅), acelerado con la secante: con aire estándar
        # T₄ no depende de T₅ y converge en una vuelta.
        x0 = T2
        hot = _hot_side(inputs, x0, p_cc_in)
        g0 = regenerated_T(hot)
        x1 = g0
        T_cc_in = x1
        for _ in range(60):
            hot = _hot_side(inputs, x1, p_cc_in)
            g1 = regenerated_T(hot)
            if abs(g1 - x1) < _TOL_T_K:
                T_cc_in = x1
                break
            d0, d1 = g0 - x0, g1 - x1
            x_new = x1 - d1 * (x1 - x0) / (d1 - d0) if d1 != d0 else g1
            x0, g0, x1 = x1, g1, x_new
        else:  # pragma: no cover - no pasa con datos válidos
            raise ValueError("El regenerador no converge: revisá los datos.")
        hot = _hot_side(inputs, T_cc_in, p_cc_in)
    F = hot.F
    gas = hot.turbines[-1]["gas"]

    # --- estados numerados como Cengel ---------------------------------
    number = _numbering(inputs.compressor_stages, inputs.turbine_stages, eps is not None)
    entries: list[tuple[str, float, float, float, FlueGas, str]] = []  # label, T, p, h, mix, desc

    def add(label: str, T: float, p: float, h: float, mix: FlueGas, desc: str) -> None:
        entries.append((label, T, p, h, mix, desc))

    compressors_idx: list[tuple[str, str, str]] = []
    for k, c in enumerate(comp):
        n_in, n_out = number[("c_in", k)], number[("c_out", k)]
        desc_in = (
            "aire ambiente (entrada del compresor)"
            if k == 0 and inputs.compressor_stages == 1
            else "aire ambiente (entrada del compresor de baja)"
            if k == 0
            else f"salida del interenfriador (entrada del {c_names[k]})"
        )
        add(str(n_in), c["T_in"], c["p_in"], c["h_in"], air, desc_in)
        desc_s = f"salida isoentrópica del {c_names[k]}"
        add(f"{n_out}s", c["T_s"], c["p_out"], c["h_s"], air, desc_s)
        add(str(n_out), c["T_out"], c["p_out"], c["h_out"], air, f"salida del {c_names[k]}")
        compressors_idx.append((str(n_in), f"{n_out}s", str(n_out)))
    regen_labels: tuple[str, str] | None = None
    if eps is not None:
        n5 = number[("r_air", 0)]
        add(str(n5), T_cc_in, p_cc_in, air.h(T_cc_in), air, "aire que sale del regenerador")
    turbines_idx: list[tuple[str, str, str]] = []
    heat_name = "de la cámara de combustión" if inputs.fuel is not None else "del calentador"
    for j, t in enumerate(hot.turbines):
        n_in, n_out = number[("t_in", j)], number[("t_out", j)]
        mix = t["gas"]
        if j == 0:
            desc_in = f"salida {heat_name} (entrada a la {t_names[0]})"
        elif inputs.fuel is not None:
            desc_in = f"salida del recalentamiento (entrada a la {t_names[j]})"
        else:
            desc_in = f"salida del recalentador (entrada a la {t_names[j]})"
        add(str(n_in), t["T_in"], t["p_in"], t["h_in"], mix, desc_in)
        desc_s = f"salida isoentrópica de la {t_names[j]}"
        add(f"{n_out}s", t["T_s"], t["p_out"], t["h_s"], mix, desc_s)
        add(str(n_out), t["T_out"], t["p_out"], t["h_out"], mix, f"salida de la {t_names[j]}")
        turbines_idx.append((str(n_in), f"{n_out}s", str(n_out)))
    regen: tuple[float, float, float] | None = None
    if eps is not None:
        n6 = number[("r_gas", 0)]
        t_last = hot.turbines[-1]
        h5 = air.h(T_cc_in)
        h6 = t_last["h_out"] - (h5 - h2) / (1.0 + F)
        T6 = gas.T_from_h(h6)
        if T6 < T2 - 1e-6:  # pragma: no cover - con ε ≤ 1 no pasa
            raise ValueError("El regenerador cruzaría las temperaturas: revisá la efectividad.")
        add(str(n6), T6, inputs.p_amb_Pa, h6, gas, "escape (sale del regenerador)")
        regen = (h5, h6, air.h(t_last["T_out"]))
        regen_labels = (str(n5), str(n6))

    def sort_key(entry: tuple[str, float, float, float, FlueGas, str]) -> tuple[int, int]:
        label = entry[0]
        return (int(label.rstrip("s")), 0 if label.endswith("s") else 1)

    entries.sort(key=sort_key)
    states = tuple(
        CycleState(
            label,
            T,
            p,
            h,
            mix.s(T, p),
            "aire" if mix is air else "gases",
            mix,
            desc,
        )
        for label, T, p, h, mix, desc in entries
    )
    index = {s.label: i for i, s in enumerate(states)}
    compressors = tuple(
        Stage("compresor", c_names[k], index[a], index[b], index[c], 1.0)
        for k, (a, b, c) in enumerate(compressors_idx)
    )
    intercoolers = tuple(
        Intercooler(
            "interenfriador" if inputs.compressor_stages == 2 else f"interenfriador {k + 1}",
            compressors[k].outlet,
            compressors[k + 1].inlet,
        )
        for k in range(inputs.compressor_stages - 1)
    )
    turbines = tuple(
        Stage("turbina", t_names[j], index[a], index[b], index[c], 1.0 + hot.turbines[j]["F"])
        for j, (a, b, c) in enumerate(turbines_idx)
    )
    combustors: list[Combustor] = []
    for j, (where, _T, _p, h_in, df, F_in) in enumerate(hot.combustors):
        inlet = (
            (index[regen_labels[0]] if regen_labels else compressors[-1].outlet)
            if j == 0
            else turbines[j - 1].outlet
        )
        outlet = turbines[j].inlet
        if inputs.fuel is None:
            q = states[outlet].h_J_per_kg - h_in
            name = (
                "calentador"
                if j == 0
                else ("recalentador" if inputs.turbine_stages == 2 else f"recalentador {j}")
            )
        else:
            q = df * inputs.fuel.lhv_J_per_kg
            name = (
                "cámara de combustión"
                if j == 0
                else (
                    "cámara de recalentamiento"
                    if inputs.turbine_stages == 2
                    else f"cámara de recalentamiento {j}"
                )
            )
        del where
        combustors.append(Combustor(name, inlet, outlet, df, F_in, F_in + df, q))
    regenerator = None
    if regen is not None and regen_labels is not None and eps is not None:
        h5, h6, h_a4 = regen
        regenerator = Regenerator(
            effectiveness=eps,
            air_in=compressors[-1].outlet,
            air_out=index[regen_labels[0]],
            gas_in=turbines[-1].outlet,
            gas_out=index[regen_labels[1]],
            m_gas=1.0 + F,
            q=h5 - h2,
            q_max=h_a4 - h2,
        )
    exhaust = index[regen_labels[1]] if regen_labels else turbines[-1].outlet
    excess = None
    if inputs.fuel is not None:
        excess = _stoichiometric_fuel_air_ratio(inputs.brayton()) / F
    result = GasTurbineResult(
        inputs=inputs,
        states=states,
        compressors=compressors,
        intercoolers=intercoolers,
        combustors=tuple(combustors),
        turbines=turbines,
        regenerator=regenerator,
        exhaust=exhaust,
        gas=gas,
        fuel_air_ratio=F,
        excess_air=excess,
    )
    if inputs.W_net_W is not None:
        m_air = inputs.W_net_W / result.w_net_J_per_kg
        if not (math.isfinite(m_air) and m_air > 0.0):
            raise ValueError(
                "Con estos datos la turbina de gas no entrega trabajo neto: no se puede "
                "dimensionar por potencia."
            )
        result = replace(result, inputs=replace(inputs, m_air_kg_s=m_air))
    return result


# ---------------------------------------------------------------------
# Exergía
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ExergyItem:
    """Una parte del balance de exergía, por kg de aire (J/kg)."""

    name: str
    kind: Literal["útil", "destruida", "perdida"]
    x_J_per_kg: float


@dataclass(frozen=True)
class GasTurbineExergy:
    """Balance de exergía de la turbina de gas (vademecum §11.9), por kg de aire.

    Ẋ_ent = Ẇ_neto + Σ Ẋ_dest + Ẋ_perdida. Con combustión, la exergía que entra
    es la del combustible, ≈ PCI (vademecum §16.13); con aire estándar, la del
    calor que entra en cada calentador, Q·(1 − T₀/T) con T su temperatura de
    salida (§11.5). El estado muerto es el ambiente (T₀, p₀).
    """

    T0_K: float
    p0_Pa: float
    x_in_J_per_kg: float
    items: tuple[ExergyItem, ...]
    m_air_kg_s: float

    def total(self, kind: Literal["útil", "destruida", "perdida"]) -> float:
        return sum(i.x_J_per_kg for i in self.items if i.kind == kind)

    @property
    def efficiency(self) -> float:
        """Rendimiento exergético: Ẇ_neto / Ẋ_ent (vademecum §11.10)."""
        return self.total("útil") / self.x_in_J_per_kg

    @property
    def residual_J_per_kg(self) -> float:
        """Lo que no cierra del balance (tiene que dar ~0)."""
        return self.x_in_J_per_kg - sum(i.x_J_per_kg for i in self.items)

    @property
    def X_in_W(self) -> float:
        return self.m_air_kg_s * self.x_in_J_per_kg


def _zero_if_noise(x: float, scale: float) -> float:
    """Un compresor o turbina isoentrópicos dan T₀·Δs ~ 1e-10: es 0."""
    return 0.0 if abs(x) < 1e-9 * max(scale, 1.0) else x


def gas_turbine_exergy(result: GasTurbineResult) -> GasTurbineExergy:
    """Exergía destruida y perdida en cada componente (Cengel §9-12; vademecum §11).

    - Compresores y turbinas (adiabáticos): Ẋ_dest = T₀·ṁ·(s_sal − s_ent).
    - Regenerador (adiabático): T₀·[ṁ_a·(s₅ − s₂) + ṁ_g·(s₆ − s₄)].
    - Cámaras: lo que no sale de su balance, ṁ_ent·ψ_ent + Ẋ_comb − ṁ_sal·ψ_sal.
    - Interenfriadores: el calor va al agua de enfriamiento (o al ambiente) a
      T₀; se pierde la exergía que pierde el aire, ψ_ent − ψ_sal.
    - Escape: se pierde su exergía física, (1 + F)·ψ_esc.
    """
    inputs = result.inputs
    T0, p0 = inputs.T_amb_K, inputs.p_amb_Pa
    st = result.states
    scale = abs(result.w_turbine_J_per_kg)

    def psi(i: int) -> float:
        return st[i].psi(T0, p0)

    def ds(i: int, j: int) -> float:
        return st[j].s_J_per_kg_K - st[i].s_J_per_kg_K

    items: list[ExergyItem] = [ExergyItem("trabajo neto", "útil", result.w_net_J_per_kg)]
    for c in result.compressors:
        items.append(
            ExergyItem(c.name, "destruida", _zero_if_noise(T0 * ds(c.inlet, c.outlet), scale))
        )
    for ic in result.intercoolers:
        items.append(ExergyItem(ic.name, "perdida", psi(ic.inlet) - psi(ic.outlet)))
    rg = result.regenerator
    if rg is not None:
        x = T0 * (ds(rg.air_in, rg.air_out) + rg.m_gas * ds(rg.gas_in, rg.gas_out))
        items.append(ExergyItem("regenerador", "destruida", _zero_if_noise(x, scale)))
    x_in = 0.0
    for cc, t in zip(result.combustors, result.turbines, strict=True):
        if inputs.fuel is None:
            x_heat = cc.q * (1.0 - T0 / st[cc.outlet].T_K)
        else:
            x_heat = cc.fuel * inputs.fuel.lhv_J_per_kg
        x_in += x_heat
        x_cc = cc.m_in * psi(cc.inlet) + x_heat - cc.m_out * psi(cc.outlet)
        items.append(ExergyItem(cc.name, "destruida", x_cc))
        items.append(
            ExergyItem(t.name, "destruida", _zero_if_noise(T0 * t.m * ds(t.inlet, t.outlet), scale))
        )
    x_exhaust = (1.0 + result.fuel_air_ratio) * psi(result.exhaust)
    items.append(ExergyItem("escape", "perdida", x_exhaust))
    return GasTurbineExergy(T0, p0, x_in, tuple(items), result.m_air_kg_s)


# ---------------------------------------------------------------------
# Comparación de las mejoras
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ImprovementRow:
    """Una configuración de la comparación: el resultado o por qué no se puede."""

    label: str
    inputs: GasTurbineInputs
    result: GasTurbineResult | None
    error: str = ""
    current: bool = False


def improvement_comparison(inputs: GasTurbineInputs) -> list[ImprovementRow]:
    """La misma turbina de gas simple, con regenerador, interenfriamiento y recalentamiento.

    Usa los rendimientos, la TIT y las pérdidas de los datos. Las etapas son
    las de los datos (o dos si los datos tienen una) y la efectividad del
    regenerador, la de los datos (o 0,80, la del ejemplo 9-7 de Cengel). Las
    presiones intermedias dadas se usan solo con la misma cantidad de etapas.
    Marca con ``current`` la configuración de los datos.
    """
    n_c = inputs.compressor_stages if inputs.intercooled else 2
    n_t = inputs.turbine_stages if inputs.reheated else 2
    eps = inputs.regenerator if inputs.regenerated else DEFAULT_EFFECTIVENESS
    options = [
        ("Simple", 1, 1, None),
        ("Con regenerador", 1, 1, eps),
        (f"Con interenfriamiento ({n_c} etapas)", n_c, 1, None),
        (f"Con recalentamiento ({n_t} etapas)", 1, n_t, None),
        ("Interenfriamiento y recalentamiento", n_c, n_t, None),
        ("Interenfriamiento, recalentamiento y regenerador", n_c, n_t, eps),
    ]
    rows: list[ImprovementRow] = []
    for label, c, t, e in options:
        case = replace(
            inputs,
            compressor_stages=c,
            turbine_stages=t,
            regenerator=e,
            p_intercool_Pa=inputs.p_intercool_Pa if c == inputs.compressor_stages else (),
            p_reheat_Pa=inputs.p_reheat_Pa if t == inputs.turbine_stages else (),
            W_net_W=None,
        )
        current = (
            c == inputs.compressor_stages and t == inputs.turbine_stages and e == inputs.regenerator
        )
        try:
            rows.append(ImprovementRow(label, case, solve_gas_turbine(case), current=current))
        except ValueError as exc:
            rows.append(ImprovementRow(label, case, None, str(exc), current))
    return rows


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def gas_turbine_notes(result: GasTurbineResult) -> list[str]:
    """Observaciones didácticas sobre la turbina de gas calculada (markdown)."""
    inputs = result.inputs
    notes: list[str] = []
    rbw = result.back_work_ratio
    if rbw > 0.5:
        notes.append(
            f"El compresor se lleva el {rbw * 100:.0f} % del trabajo de la turbina (relación de "
            "trabajo de retroceso): por eso pesan tanto los rendimientos del compresor y de la "
            "turbina (Cengel §9-8). Interenfriar baja ese trabajo."
        )
    if (inputs.intercooled or inputs.reheated) and not inputs.regenerated:
        notes.append(
            "Interenfriar o recalentar sin regenerador sube el trabajo neto, pero casi no mejora "
            "el rendimiento (o lo empeora): el aire sale más frío del compresor y la cámara tiene "
            "que calentarlo desde más abajo, y los gases salen más calientes de la turbina. Con "
            "un regenerador, ese escape caliente precalienta el aire (Cengel §9-10)."
        )
    rg = result.regenerator
    if rg is not None:
        T2 = result.states[rg.air_in].T_K
        T4 = result.states[rg.gas_in].T_K
        notes.append(
            f"El regenerador calienta el aire de {_degC(T2)} a "
            f"{_degC(result.states[rg.air_out].T_K)} con los gases que salen de la turbina a "
            f"{_degC(T4)}: le ahorra a la cámara {rg.q / 1e3:.4g} kJ por kg de aire. Sirve "
            f"mientras T₄ > T₂ (acá, {T4 - T2:.0f} K de diferencia); con más relación de presiones "
            "esa diferencia se achica y el regenerador deja de servir (Cengel §9-9)."
        )
        if rg.effectiveness == 1.0:
            notes.append(
                "Un regenerador ideal (ε = 1) calienta el aire hasta la temperatura de los gases "
                "que salen de la turbina: necesitaría un área infinita. Los reales tienen ε de "
                "0,6 a 0,9; más ε es más área, más costo y más pérdida de carga."
            )
    if (
        inputs.intercooled
        and inputs.reheated
        and inputs.regenerated
        and min(inputs.compressor_stages, inputs.turbine_stages) >= 2
    ):
        notes.append(
            f"Con muchas etapas de interenfriamiento y de recalentamiento y un regenerador ideal, "
            "el ciclo se acerca al de Ericsson, que tiene el rendimiento de Carnot entre la "
            f"menor y la mayor temperatura (Cengel §9-10). Acá η = {result.eta_th * 100:.1f} % y "
            f"Carnot = {result.eta_carnot * 100:.1f} %."
        )
    if inputs.fuel is not None and result.excess_air is not None:
        o2 = result.gas.mole_fractions["O2"]
        text = (
            f"La turbina quema con un exceso de aire λ = {result.excess_air:.3g}: el aire de más "
            "baja la temperatura de los gases hasta la que soportan los álabes. Por eso el escape "
            f"todavía tiene {o2 * 100:.3g} % de O₂."
        )
        if inputs.reheated:
            text += (
                " Cada recalentamiento es una cámara que quema más combustible en esos gases "
                "(combustión secuencial, como las GT24/GT26)."
            )
        notes.append(text)
        if inputs.T_turbine_in_K > _T_BLADE_COOLING_K:
            notes.append(
                f"Con la TIT de {_degC(inputs.T_turbine_in_K)}, las turbinas reales refrigeran los "
                "álabes con aire que sacan del compresor, y tienen pérdidas mecánicas y en el "
                "generador. El modelo no las tiene: los rendimientos dan algunos puntos más altos "
                "que los de catálogo."
            )
        dew = result.gas.dew_point_K
        if dew is not None and result.T_exhaust_K < dew + 10.0:
            notes.append(
                f"El escape sale a {_degC(result.T_exhaust_K)}, cerca del punto de rocío del agua "
                f"de los gases ({_degC(dew)}): podría condensar en el regenerador."
            )
    else:
        notes.append(
            "Modelo de aire estándar (Cengel §9-3): la combustión se reemplaza por un "
            "intercambiador que le pasa calor al aire, y por la turbina pasa aire. Es una "
            "simplificación: no hay combustible ni cambia la composición."
        )
    return notes


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------

_C = 273.15
_CENGEL_9_5 = GasTurbineInputs(8.0, 1300.0, T_amb_K=300.0, p_amb_Pa=100e3, fuel=None)
_CENGEL_9_6 = replace(_CENGEL_9_5, eta_compressor=0.80, eta_turbine=0.85)
_CENGEL_9_8 = replace(_CENGEL_9_5, compressor_stages=2, turbine_stages=2)

_EX_9_5 = "Cengel 9-5 — Brayton ideal, aire estándar (r_p 8, 1300 K)"
_EX_9_6 = "Cengel 9-6 — real: η_C = 0,80 y η_T = 0,85"
_EX_9_7 = "Cengel 9-7 — real con regenerador (ε = 0,80)"
_EX_9_8A = "Cengel 9-8 — 2 + 2 etapas, interenfriamiento y recalentamiento"
_EX_9_8B = "Cengel 9-8 — 2 + 2 etapas y regenerador ideal"
_EX_INDUSTRIAL = "Industrial a metano (r_p 15, TIT 1250 °C)"
_EX_MICRO = "Microturbina con regenerador (r_p 4, TIT 950 °C)"
_EX_AERO = "Aeroderivada con interenfriamiento (r_p 40, TIT 1380 °C)"
_EX_SEQUENTIAL = "Combustión secuencial (r_p 33, recalienta a 18 bar)"

#: Ejemplos de la página (el primero es el que aparece al entrar).
GAS_TURBINE_EXAMPLES: dict[str, GasTurbineInputs] = {
    _EX_9_5: _CENGEL_9_5,
    _EX_9_6: _CENGEL_9_6,
    _EX_9_7: replace(_CENGEL_9_6, regenerator=0.80),
    _EX_9_8A: _CENGEL_9_8,
    _EX_9_8B: replace(_CENGEL_9_8, regenerator=1.0),
    _EX_INDUSTRIAL: GasTurbineInputs(
        15.0, 1250 + _C, 0.88, 0.90, dp_combustor=0.03, m_air_kg_s=500.0
    ),
    _EX_MICRO: GasTurbineInputs(
        4.0,
        950 + _C,
        0.78,
        0.83,
        dp_combustor=0.03,
        regenerator=0.87,
        dp_regenerator=0.03,
        m_air_kg_s=0.5,
    ),
    _EX_AERO: GasTurbineInputs(
        40.0,
        1380 + _C,
        0.88,
        0.90,
        dp_combustor=0.03,
        compressor_stages=2,
        T_intercool_K=30 + _C,
        dp_intercooler=0.03,
        p_intercool_Pa=(3.5e5,),
        m_air_kg_s=200.0,
    ),
    _EX_SEQUENTIAL: GasTurbineInputs(
        33.0,
        1250 + _C,
        0.88,
        0.90,
        dp_combustor=0.04,
        turbine_stages=2,
        p_reheat_Pa=(18e5,),
        m_air_kg_s=600.0,
    ),
}

#: Qué muestra cada ejemplo (se ve debajo del selector).
GAS_TURBINE_EXAMPLE_NOTES: dict[str, str] = {
    _EX_9_5: (
        "Cengel & Boles, ejemplo 9-5: el libro da T₂ = 540 K, T₄ = 770 K, r_bw = 0,403 y "
        "η = 42,6 %, con la tabla A-17 del aire."
    ),
    _EX_9_6: (
        "Ejemplo 9-6: el mismo ciclo con un compresor de 80 % y una turbina de 85 %. El libro "
        "da r_bw = 0,592 y η = 26,6 %: las irreversibilidades pesan mucho en una turbina de gas."
    ),
    _EX_9_7: (
        "Ejemplo 9-7: al ciclo del 9-6 se le suma un regenerador de 80 % de efectividad. El "
        "libro da η = 36,9 %: el trabajo neto no cambia, pero la cámara calienta menos."
    ),
    _EX_9_8A: (
        "Ejemplo 9-8 a): dos etapas de compresión con interenfriamiento a 300 K y dos de "
        "expansión con recalentamiento a 1300 K, con r_p = 8. El libro da r_bw = 0,304 y "
        "η = 35,8 %: más trabajo neto que el 9-5, pero menos rendimiento."
    ),
    _EX_9_8B: (
        "Ejemplo 9-8 b): lo mismo con un regenerador ideal (ε = 1). El libro da η = 69,6 %: "
        "con el regenerador, el interenfriamiento y el recalentamiento sí suben el rendimiento."
    ),
    _EX_INDUSTRIAL: (
        "Una turbina de gas industrial a gas natural (metano), de ciclo simple, como la del ciclo "
        "combinado: 500 kg/s de aire."
    ),
    _EX_MICRO: (
        "Las microturbinas (30 a 300 kW) tienen compresor y turbina radiales, de menor "
        "rendimiento, y relaciones de presión bajas: sin el regenerador rendirían menos del 20 %."
    ),
    _EX_AERO: (
        "Con el aire enfriado a 3,5 bar entre el compresor de baja y el de alta (como la LMS100 "
        "de GE), el compresor de alta trabaja con aire frío: sube mucho el trabajo neto por kg "
        "de aire y el rendimiento casi no cambia."
    ),
    _EX_SEQUENTIAL: (
        "Combustión secuencial (como las GT24/GT26 de Alstom, hoy Ansaldo): una primera turbina "
        "de una sola etapa baja la presión a 18 bar y una segunda cámara vuelve a quemar en los "
        "gases, que todavía tienen oxígeno. Recalentar a presión alta deja el escape a una "
        "temperatura buena para un ciclo combinado."
    ),
}


# ---------------------------------------------------------------------
# Barridos
# ---------------------------------------------------------------------

SweepParameter = Literal[
    "pressure_ratio",
    "T_turbine_in",
    "regenerator",
    "T_amb",
    "p_intercool",
    "p_reheat",
    "stages",
]

#: Parámetros de barrido disponibles.
GAS_TURBINE_SWEEP_PARAMETERS: tuple[SweepParameter, ...] = (
    "pressure_ratio",
    "T_turbine_in",
    "regenerator",
    "T_amb",
    "p_intercool",
    "p_reheat",
    "stages",
)


@dataclass(frozen=True)
class GasTurbineSweepPoint:
    """Un punto del barrido: el valor del parámetro (SI, o etapas) y los resultados."""

    value: float
    eta: float
    w_net_J_per_kg: float
    w_compressor_J_per_kg: float
    w_turbine_J_per_kg: float
    back_work_ratio: float
    T_exhaust_K: float
    eta_carnot: float


def _with(inputs: GasTurbineInputs, parameter: SweepParameter, value: float) -> GasTurbineInputs:
    if parameter == "pressure_ratio":
        return replace(inputs, pressure_ratio=value, p_intercool_Pa=(), p_reheat_Pa=())
    if parameter == "T_turbine_in":
        T_rh = inputs.T_reheat_K
        if T_rh is not None and math.isclose(T_rh, inputs.T_turbine_in_K):
            T_rh = value
        return replace(inputs, T_turbine_in_K=value, T_reheat_K=T_rh)
    if parameter == "regenerator":
        return replace(inputs, regenerator=value)
    if parameter == "T_amb":
        return replace(inputs, T_amb_K=value)
    if parameter == "p_intercool":
        return replace(inputs, p_intercool_Pa=(value,))
    if parameter == "p_reheat":
        return replace(inputs, p_reheat_Pa=(value,))
    if parameter == "stages":
        n = round(value)
        return replace(
            inputs, compressor_stages=n, turbine_stages=n, p_intercool_Pa=(), p_reheat_Pa=()
        )
    raise ValueError(f"Parámetro de barrido desconocido: {parameter!r}.")


def default_gas_turbine_sweep_values(
    inputs: GasTurbineInputs, parameter: SweepParameter, n: int = 25
) -> list[float]:
    """Valores razonables del parámetro alrededor de la turbina dada (SI, o etapas).

    Raises
    ------
    ValueError
        Si el parámetro no aplica a la configuración (p. ej., presión de
        interenfriamiento sin dos etapas de compresión).
    """
    if parameter == "pressure_ratio":
        hi = 40.0 if not inputs.regenerated else 20.0
        lo = 2.0
        return [float(v) for v in np.geomspace(lo, max(hi, inputs.pressure_ratio), n)]
    if parameter == "T_turbine_in":
        lo = min(inputs.T_turbine_in_K - 300.0, 1073.15)
        hi = min(T_GAS_MAX_K - 50.0, max(inputs.T_turbine_in_K + 200.0, 1573.15))
        return [float(v) for v in np.linspace(lo, hi, n)]
    if parameter == "regenerator":
        return [float(v) for v in np.linspace(0.5, 1.0, n)]
    if parameter == "T_amb":
        return [float(v) for v in np.linspace(-20 + _C, 45 + _C, n)]
    if parameter in ("p_intercool", "p_reheat"):
        if parameter == "p_intercool" and inputs.compressor_stages != 2:
            raise ValueError("El barrido de la presión intermedia pide dos etapas de compresión.")
        if parameter == "p_reheat" and inputs.turbine_stages != 2:
            raise ValueError("El barrido de la presión intermedia pide dos etapas de expansión.")
        p1 = inputs.p_amb_Pa
        p2 = p1 * inputs.pressure_ratio
        return [float(v) for v in np.geomspace(p1 * 1.15, p2 / 1.15, n)]
    if parameter == "stages":
        return [float(k) for k in range(1, MAX_STAGES + 1)]
    raise ValueError(f"Parámetro de barrido desconocido: {parameter!r}.")


def gas_turbine_sweep(
    inputs: GasTurbineInputs, parameter: SweepParameter, values: Sequence[float]
) -> list[GasTurbineSweepPoint]:
    """Resuelve la turbina para cada valor; omite los que no tienen sentido físico."""
    points: list[GasTurbineSweepPoint] = []
    for value in values:
        try:
            r = solve_gas_turbine(replace(_with(inputs, parameter, float(value)), W_net_W=None))
        except ValueError:
            continue
        points.append(
            GasTurbineSweepPoint(
                float(value),
                r.eta_th,
                r.w_net_J_per_kg,
                r.w_compressor_J_per_kg,
                r.w_turbine_J_per_kg,
                r.back_work_ratio,
                r.T_exhaust_K,
                r.eta_carnot,
            )
        )
    return points


# ---------------------------------------------------------------------
# Diagrama T–s
# ---------------------------------------------------------------------

LineKind = Literal[
    "isobar", "heat", "cooling", "regenerator", "isentropic", "actual", "composition", "exhaust"
]


@dataclass(frozen=True)
class CycleLine:
    """Una línea del T–s de la turbina de gas (SI).

    ``kind``: ``"isobar"`` (referencia), ``"heat"`` (cámara o calentador),
    ``"cooling"`` (interenfriador), ``"regenerator"`` (los dos lados del
    regenerador), ``"isentropic"`` (2s, 4s…), ``"actual"`` (compresión o
    expansión real, solo de referencia), ``"composition"`` (una cámara cambia la
    composición a la misma T) o ``"exhaust"`` (el escape se enfría en la
    atmósfera hasta T₁, lo que cierra el ciclo).
    """

    kind: LineKind
    medium: Literal["aire", "gases"]
    T_K: tuple[float, ...]
    s_J_per_kg_K: tuple[float, ...]
    p_Pa: tuple[float, ...]


def _heat_line(kind: LineKind, a: CycleState, b: CycleState, mix: FlueGas, n: int) -> CycleLine:
    """De a a b con la mezcla ``mix``: p lineal en h si hay pérdida de carga (Fase 3.1c)."""
    temps = [a.T_K + (b.T_K - a.T_K) * k / (n - 1) for k in range(n)]
    h_a, h_b = mix.h(a.T_K), mix.h(b.T_K)
    pressures = []
    for T in temps:
        frac = 0.0 if h_b == h_a else (mix.h(T) - h_a) / (h_b - h_a)
        pressures.append(a.P_Pa + (b.P_Pa - a.P_Pa) * frac)
    return CycleLine(
        kind,
        b.medium,
        tuple(temps),
        tuple(mix.s(T, p) for T, p in zip(temps, pressures, strict=True)),
        tuple(pressures),
    )


def _segment(kind: LineKind, a: CycleState, b: CycleState) -> CycleLine:
    return CycleLine(
        kind, a.medium, (a.T_K, b.T_K), (a.s_J_per_kg_K, b.s_J_per_kg_K), (a.P_Pa, b.P_Pa)
    )


def gas_turbine_lines(result: GasTurbineResult, n: int = 30) -> list[CycleLine]:
    """Procesos de la turbina de gas para dibujarla en un T–s (entropías absolutas).

    Con entropías absolutas (:meth:`FlueGas.s`), el aire y los gases quedan en la
    misma escala aunque cambie la composición. Las isobaras de referencia son
    la del ambiente y la de la descarga del compresor.
    """
    st = result.states
    inputs = result.inputs
    air = inputs.air
    lines: list[CycleLine] = []
    T_lo = min(s.T_K for s in st) - 20.0
    T_hi = max(s.T_K for s in st)
    p_top = st[result.compressors[-1].outlet].P_Pa
    for p in (inputs.p_amb_Pa, p_top):
        temps = [
            max(T_lo, air.T_min_K + 1.0) + (T_hi - max(T_lo, air.T_min_K + 1.0)) * k / (n - 1)
            for k in range(n)
        ]
        lines.append(
            CycleLine(
                "isobar",
                "aire",
                tuple(temps),
                tuple(air.s(T, p) for T in temps),
                tuple(p for _ in temps),
            )
        )
    for stage in (*result.compressors, *result.turbines):
        a, s_, b = st[stage.inlet], st[stage.outlet_s], st[stage.outlet]
        lines.append(_segment("isentropic", a, s_))
        if stage.outlet_s != stage.outlet and not math.isclose(s_.T_K, b.T_K, abs_tol=1e-6):
            lines.append(_segment("actual", a, b))
    for ic in result.intercoolers:
        lines.append(_heat_line("cooling", st[ic.inlet], st[ic.outlet], air, n))
    rg = result.regenerator
    if rg is not None:
        lines.append(_heat_line("regenerator", st[rg.air_in], st[rg.air_out], air, n))
        lines.append(_heat_line("regenerator", st[rg.gas_in], st[rg.gas_out], result.gas, n))
    for cc in result.combustors:
        a, b = st[cc.inlet], st[cc.outlet]
        mix = b.mix
        if mix is not a.mix:
            s_new = mix.s(a.T_K, a.P_Pa)
            lines.append(
                CycleLine(
                    "composition",
                    b.medium,
                    (a.T_K, a.T_K),
                    (a.s_J_per_kg_K, s_new),
                    (a.P_Pa, a.P_Pa),
                )
            )
            start = CycleState(
                a.label, a.T_K, a.P_Pa, mix.h(a.T_K), s_new, b.medium, mix, a.description
            )
        else:
            start = a
        lines.append(_heat_line("heat", start, b, mix, n))
    exh = st[result.exhaust]
    if exh.T_K > inputs.T_amb_K + 1.0:
        mix, T = exh.mix, inputs.T_amb_K
        end = CycleState(
            "amb", T, exh.P_Pa, mix.h(T), mix.s(T, exh.P_Pa), exh.medium, mix, "ambiente"
        )
        lines.append(_heat_line("exhaust", exh, end, mix, n))
    return lines


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _value(value_si: float, kind: QuantityKind, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def gas_turbine_to_dict(result: GasTurbineResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``)."""
    gi = result.inputs
    fuel = gi.fuel
    eh: QuantityKind = "specific_enthalpy"
    exergy = gas_turbine_exergy(result)
    return {
        "modelo": "aire estándar" if fuel is None else f"combustión de {fuel.name}",
        "configuracion": gi.configuration,
        "relacion_de_presiones": gi.pressure_ratio,
        "etapas_de_compresion": gi.compressor_stages,
        "etapas_de_expansion": gi.turbine_stages,
        "rendimiento_compresor": gi.eta_compressor,
        "rendimiento_turbina": gi.eta_turbine,
        "efectividad_regenerador": gi.regenerator,
        "caida_de_presion_camara": gi.dp_combustor,
        "caida_de_presion_interenfriador": gi.dp_intercooler,
        "caida_de_presion_regenerador": gi.dp_regenerator,
        "T_ambiente": _value(gi.T_amb_K, "temperature", system),
        "p_ambiente": _value(gi.p_amb_Pa, "pressure", system),
        "TIT": _value(gi.T_turbine_in_K, "temperature", system),
        "T_interenfriamiento": (
            _value(gi.T_intercool, "temperature", system) if gi.intercooled else None
        ),
        "T_recalentamiento": _value(gi.T_reheat, "temperature", system) if gi.reheated else None,
        "composicion_aire": {GAS_NAMES[s]: y for s, y in gi.air.mole_fractions.items()},
        "PCI": None if fuel is None else _value(fuel.lhv_J_per_kg, eh, system),
        "estados": [
            {
                "estado": s.label,
                "descripcion": s.description,
                "medio": s.medium,
                "T": _value(s.T_K, "temperature", system),
                "p": _value(s.P_Pa, "pressure", system),
                "h_desde_25C": _value(s.h_J_per_kg, eh, system),
                "s": _value(s.s_J_per_kg_K, "specific_entropy", system),
            }
            for s in result.states
        ],
        "compresores": [
            {"nombre": c.name, "w": _value(result.stage_work(c), eh, system)}
            for c in result.compressors
        ],
        "interenfriadores": [
            {"nombre": ic.name, "q": _value(result.intercooler_heat(ic), eh, system)}
            for ic in result.intercoolers
        ],
        "camaras": [
            {
                "nombre": cc.name,
                "combustible_por_kg_de_aire": cc.fuel,
                "q": _value(cc.q, eh, system),
            }
            for cc in result.combustors
        ],
        "turbinas": [
            {"nombre": t.name, "w": _value(result.stage_work(t), eh, system)}
            for t in result.turbines
        ],
        "regenerador_q": (
            None if result.regenerator is None else _value(result.regenerator.q, eh, system)
        ),
        "relacion_combustible_aire": result.fuel_air_ratio,
        "exceso_de_aire": result.excess_air,
        "composicion_escape": {GAS_NAMES[s]: y for s, y in result.gas.mole_fractions.items()},
        "caudal_aire": _value(result.m_air_kg_s, "mass_flow", system),
        "caudal_combustible": _value(result.m_fuel_kg_s, "mass_flow", system),
        "w_compresor": _value(result.w_compressor_J_per_kg, eh, system),
        "w_turbina": _value(result.w_turbine_J_per_kg, eh, system),
        "w_neto": _value(result.w_net_J_per_kg, eh, system),
        "q_entrada": _value(result.q_in_J_per_kg, eh, system),
        "potencia_neta": _value(result.W_net_W, "power", system),
        "calor_entrada": _value(result.Q_in_W, "power", system),
        "rendimiento": result.eta_th,
        "relacion_trabajo_retroceso": result.back_work_ratio,
        "consumo_especifico_kJ_por_kWh": result.heat_rate_kJ_per_kWh,
        "rendimiento_carnot": result.eta_carnot,
        "T_escape": _value(result.T_exhaust_K, "temperature", system),
        "exergia": {
            "T0": _value(exergy.T0_K, "temperature", system),
            "entrada": _value(exergy.x_in_J_per_kg, eh, system),
            "rendimiento_exergetico": exergy.efficiency,
            "partes": [
                {"nombre": i.name, "tipo": i.kind, "x": _value(i.x_J_per_kg, eh, system)}
                for i in exergy.items
            ],
        },
    }


# ---------------------------------------------------------------------
# Control con TESPy
# ---------------------------------------------------------------------


#: Efectividad máxima del regenerador que acepta el control con TESPy.
_TESPY_MAX_EFFECTIVENESS = 0.995


@dataclass(frozen=True)
class GasTurbineTespy:
    """La misma turbina de gas resuelta con TESPy (por kg de aire, SI)."""

    w_compressor_J_per_kg: float
    w_turbine_J_per_kg: float
    q_in_J_per_kg: float
    fuel_air_ratio: float
    T_exhaust_K: float
    T_regenerator_air_K: float | None
    seconds: float

    @property
    def w_net_J_per_kg(self) -> float:
        return self.w_turbine_J_per_kg - self.w_compressor_J_per_kg

    @property
    def eta_th(self) -> float:
        return self.w_net_J_per_kg / self.q_in_J_per_kg


def gas_turbine_tespy(result: GasTurbineResult) -> GasTurbineTespy:  # noqa: PLR0912, PLR0915
    """Resuelve la misma turbina de gas con TESPy, como control cruzado.

    Un ``Compressor`` por etapa, ``SimpleHeatExchanger`` en los
    interenfriadores, un ``HeatExchanger`` con ``eff_cold`` = ε en el
    regenerador (la efectividad de Cengel), ``DiabaticCombustionChamber`` en la
    cámara y en cada recalentamiento (como el tutorial de turbina de gas de
    TESPy 0.11; con aire estándar, ``SimpleHeatExchanger``) y una ``Turbine``
    por etapa. TESPy evalúa los gases con la ecuación de estado real, así que
    difiere unas décimas de % del cálculo con gases ideales. Arranca de los
    valores del cálculo directo.

    Raises
    ------
    ValueError
        Si el combustible tiene componentes que TESPy no quema o el cálculo no
        converge.
    """
    from tespy.components import (
        Compressor,
        DiabaticCombustionChamber,
        HeatExchanger,
        SimpleHeatExchanger,
        Sink,
        Source,
        Turbine,
    )
    from tespy.connections import Connection, Ref

    from core.cycles.tespy_utils import new_network, solve

    inputs = result.inputs
    fuel = inputs.fuel
    st = result.states
    if (
        result.regenerator is not None
        and result.regenerator.effectiveness > _TESPY_MAX_EFFECTIVENESS
    ):
        raise ValueError(
            "TESPy no resuelve un regenerador ideal (ε = 1): necesitaría un área infinita y la "
            "red queda en el borde de lo físico. El control se puede hacer con ε = 0,99."
        )
    if fuel is not None:
        missing = [n for n, _ in fuel.composition if n not in _TESPY_FUEL_NAMES]
        if missing:
            raise ValueError(f"TESPy no tiene en su modelo de combustión: {missing}.")
    nw = new_network()
    conns: list[Connection] = []
    prev, port = Source("aire"), "out1"
    first: Connection | None = None
    compressors: list[Compressor] = []
    for k, stage in enumerate(result.compressors):
        cp = Compressor(stage.name)
        c_in = Connection(prev, port, cp, "in1", label=f"{st[stage.inlet].label}")
        conns.append(c_in)
        if first is None:
            first = c_in
        else:
            c_in.set_attr(T=st[stage.inlet].T_K)
        cp.set_attr(eta_s=inputs.eta_compressor, pr=st[stage.outlet].P_Pa / st[stage.inlet].P_Pa)
        compressors.append(cp)
        prev, port = cp, "out1"
        if k < len(result.intercoolers):
            ic = SimpleHeatExchanger(result.intercoolers[k].name)
            c_out = Connection(prev, port, ic, "in1", label=st[stage.outlet].label)
            c_out.set_attr(T0=st[stage.outlet].T_K)
            ic.set_attr(pr=1.0 - inputs.dp_intercooler)
            conns.append(c_out)
            prev, port = ic, "out1"
    rg = result.regenerator
    regen: HeatExchanger | None = None
    c_regen_air: Connection | None = None
    if rg is not None:
        regen = HeatExchanger("regenerador")
        c2 = Connection(prev, port, regen, "in2", label=st[rg.air_in].label)
        c2.set_attr(T0=st[rg.air_in].T_K)
        k_rg = 1.0 - inputs.dp_regenerator
        regen.set_attr(eff_cold=rg.effectiveness, pr1=k_rg, pr2=k_rg)
        conns.append(c2)
        prev, port = regen, "out2"
    fuel_conns: list[Connection] = []
    heaters: list[SimpleHeatExchanger] = []
    turbines: list[Turbine] = []
    if fuel is not None:
        rows = _iso_tables().components
        M = fuel.M_kg_per_mol * 1e3
        w_fuel = {_TESPY_FUEL_NAMES[n]: x * rows[n].M_kg_per_kmol / M for n, x in fuel.composition}
    for j, (cc, stage) in enumerate(zip(result.combustors, result.turbines, strict=True)):
        label_in = st[cc.inlet].label
        if fuel is None:
            heater = SimpleHeatExchanger(cc.name)
            c_in = Connection(prev, port, heater, "in1", label=f"{label_in}→{cc.name}")
            heater.set_attr(pr=1.0 - inputs.dp_combustor)
            heaters.append(heater)
            conns.append(c_in)
        else:
            chamber = DiabaticCombustionChamber(cc.name)
            c_in = Connection(prev, port, chamber, "in1", label=f"{label_in}→{cc.name}")
            c_fuel = Connection(
                Source(f"combustible {j + 1}"), "out1", chamber, "in2", label=f"combustible {j + 1}"
            )
            c_fuel.set_attr(T=T_FUEL_K, fluid=w_fuel, p=Ref(c_in, 1.05, 0), m0=cc.fuel)
            chamber.set_attr(pr=1.0 - inputs.dp_combustor, eta=1.0)
            fuel_conns.append(c_fuel)
            conns += [c_in, c_fuel]
            heater = chamber
        c_in.set_attr(T0=st[cc.inlet].T_K, p0=st[cc.inlet].P_Pa)
        if j > 0:
            c_in.set_attr(p=st[cc.inlet].P_Pa)
        elif rg is not None:
            c_regen_air = c_in
        tb = Turbine(stage.name)
        c_t = Connection(heater, "out1", tb, "in1", label=st[stage.inlet].label)
        c_t.set_attr(T=st[stage.inlet].T_K)
        tb.set_attr(eta_s=inputs.eta_turbine)
        conns.append(c_t)
        turbines.append(tb)
        prev, port = tb, "out1"
    exhaust = Sink("escape")
    if regen is not None and rg is not None:
        c_t_out = Connection(prev, port, regen, "in1", label=st[rg.gas_in].label)
        c_t_out.set_attr(T0=st[rg.gas_in].T_K)
        c_exh = Connection(regen, "out1", exhaust, "in1", label=st[rg.gas_out].label)
        conns += [c_t_out, c_exh]
    else:
        c_exh = Connection(prev, port, exhaust, "in1", label=f"{st[result.exhaust].label} escape")
        conns.append(c_exh)
    c_exh.set_attr(p=inputs.p_amb_Pa, T0=result.T_exhaust_K)
    nw.add_conns(*conns)
    assert first is not None
    first.set_attr(
        T=inputs.T_amb_K,
        p=inputs.p_amb_Pa,
        m=1.0,
        fluid={s: w for s, w in inputs.air.mass_fractions.items() if w > 0.0},
    )
    t0 = time.perf_counter()
    solve(nw, what="la turbina de gas con TESPy")
    seconds = time.perf_counter() - t0
    w_c = sum(cp.P.val_SI for cp in compressors)
    w_t = -sum(tb.P.val_SI for tb in turbines)
    if fuel is None:
        f = 0.0
        q_in = sum(h.Q.val_SI for h in heaters)
    else:
        f = sum(c.m.val_SI for c in fuel_conns)
        q_in = f * fuel.lhv_J_per_kg
    return GasTurbineTespy(
        w_compressor_J_per_kg=w_c,
        w_turbine_J_per_kg=w_t,
        q_in_J_per_kg=q_in,
        fuel_air_ratio=f,
        T_exhaust_K=c_exh.T.val_SI,
        T_regenerator_air_K=None if c_regen_air is None else c_regen_air.T.val_SI,
        seconds=seconds,
    )
