"""Caldera de recuperación (HRSG) de una presión: diagrama T–Q — Fase 3.3.

Los gases calientes (el escape de una turbina de gas) pasan por el
**sobrecalentador**, el **evaporador** y el **economizador**; el agua de
alimentación recorre el camino inverso y sale como vapor sobrecalentado o,
sin sobrecalentador, como vapor saturado. El diseño se fija con dos
diferencias de temperatura (Kehlhofer, Hannemann, Stirnimann y Rukes,
*Combined-Cycle Gas & Steam Turbine Power Plants*, 3.ª ed., PennWell, 2009):

- **pinch**: los gases salen del evaporador a T_sat + ΔT_pinch;
- **approach**: el agua sale del economizador a T_sat − ΔT_approach (así no
  evapora en el economizador).

Con eso, el balance de energía sección por sección da el caudal de vapor,
la temperatura de los gases entre secciones y la de chimenea, y el
diagrama T–Q (temperatura de cada corriente contra el calor transferido).

Los gases son una **mezcla de gases ideales** (vademecum §5): h_g = Σ wᵢ·hᵢ(T),
con cada hᵢ del gas ideal de CoolProp (cada componente a 1 Pa, donde la
ecuación de estado coincide con la de gas ideal; equivale a los polinomios
NASA del vademecum §4.8). El agua y el vapor, con IAPWS-95. El cálculo es el
mismo que se hace a mano; los tests lo comparan con una red de TESPy
(``HeatExchanger`` en serie, como su tutorial de turbina de gas).

Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from CoolProp.CoolProp import PropsSI

from core.fluids import FluidState, fluid_limits, fluid_state_from_pair

# El modelo de gases se mudó a core.ideal_gas (Fase 3.4); se reexporta acá para
# que la API de la Fase 3.3 no cambie.
from core.ideal_gas import (
    GAS_NAMES,
    GAS_SPECIES,
    R_U,
    T_GAS_MAX_K,
    FlueGas,
    exhaust_composition,
    molar_mass,
)
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "GAS_NAMES",
    "GAS_SPECIES",
    "HRSG_EXAMPLES",
    "HRSG_EXAMPLES_EXCESS_AIR",
    "HRSG_EXAMPLE_NOTES",
    "R_U",
    "T_GAS_MAX_K",
    "FlueGas",
    "HRSGInputs",
    "HRSGResult",
    "HRSGSection",
    "SweepParameter",
    "SweepPoint",
    "TQProfile",
    "default_steam_temperature",
    "default_sweep_values",
    "exhaust_composition",
    "hrsg_notes",
    "hrsg_sweep",
    "hrsg_to_dict",
    "hrsg_tq_profile",
    "molar_mass",
    "other_steam_option",
    "solve_hrsg",
    "validate_hrsg_inputs",
]

# Rango del modelo: el agua de alimentación no baja del punto triple del agua.
_T_GAS_MIN_K = 273.17

Water = "Water"
_EH: QuantityKind = "specific_enthalpy"


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _bar(p_Pa: float) -> str:
    return f"{p_Pa / 1e5:.4g} bar"


# ---------------------------------------------------------------------
# Datos y resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HRSGInputs:
    """Datos de la HRSG de una presión, en SI.

    ``T_steam_K=None`` es la caldera de vapor saturado (sin sobrecalentador).
    El agua y el vapor se toman a la presión de evaporación ``p_steam_Pa``
    (sin pérdidas de carga). ``T_ref_K`` es la temperatura hasta la que se
    cuenta el calor disponible en los gases (15 °C, la de las normas ISO).

    Con ``T_stack_K`` la caldera se diseña por la **temperatura de
    chimenea** (como el intercambiador de Cengel §10-9) en vez de por el
    pinch: ``pinch_K`` no se usa y el pinch pasa a ser un resultado.
    """

    T_gas_in_K: float
    m_gas_kg_s: float
    gas: FlueGas
    p_steam_Pa: float
    T_feedwater_K: float
    T_steam_K: float | None = None
    pinch_K: float = 10.0
    approach_K: float = 5.0
    T_ref_K: float = 288.15
    T_stack_K: float | None = None

    @property
    def superheated(self) -> bool:
        return self.T_steam_K is not None

    @property
    def by_stack(self) -> bool:
        """Diseño por temperatura de chimenea (el pinch es un resultado)."""
        return self.T_stack_K is not None


@dataclass(frozen=True)
class HRSGSection:
    """Una sección de la caldera: sus temperaturas de entrada y salida y su calor."""

    name: str
    Q_W: float
    T_gas_in_K: float
    T_gas_out_K: float
    T_water_in_K: float
    T_water_out_K: float

    @property
    def dT_hot_K(self) -> float:
        """Diferencia de temperatura en el extremo caliente (entrada de gases)."""
        return self.T_gas_in_K - self.T_water_out_K

    @property
    def dT_cold_K(self) -> float:
        """Diferencia de temperatura en el extremo frío (salida de gases)."""
        return self.T_gas_out_K - self.T_water_in_K


@dataclass(frozen=True)
class HRSGResult:
    """Estados, caudal de vapor y calores de la HRSG, en SI.

    ``water`` son los estados del agua: 1 alimentación, 2 salida del
    economizador, 3 líquido saturado, 4 vapor saturado y, con sobrecalentador,
    5 vapor sobrecalentado. ``T_gas_K`` y ``h_gas_J_per_kg`` son los puntos de
    los gases: a entrada, b salida del sobrecalentador (solo si lo hay), c
    salida del evaporador (el pinch) y d chimenea.
    """

    inputs: HRSGInputs
    water: tuple[FluidState, ...]
    T_gas_K: tuple[float, ...]
    h_gas_J_per_kg: tuple[float, ...]
    m_steam_kg_s: float

    @property
    def T_sat_K(self) -> float:
        return self.water[2].T_K

    @property
    def gas_labels(self) -> tuple[str, ...]:
        if self.inputs.superheated:
            return ("a", "b", "c", "d")
        return ("a", "c", "d")

    @property
    def T_stack_K(self) -> float:
        return self.T_gas_K[-1]

    @property
    def T_pinch_gas_K(self) -> float:
        """Temperatura de los gases a la salida del evaporador: T_sat + pinch."""
        return self.T_gas_K[-2]

    @property
    def pinch_K(self) -> float:
        """Pinch: los gases a la salida del evaporador menos T_sat (dato o resultado)."""
        return self.T_pinch_gas_K - self.T_sat_K

    @property
    def T_after_superheater_K(self) -> float | None:
        return self.T_gas_K[1] if self.inputs.superheated else None

    @property
    def h_steam_J_per_kg(self) -> float:
        return self.water[-1].h_J_per_kg

    @property
    def sections(self) -> tuple[HRSGSection, ...]:
        """Sobrecalentador (si hay), evaporador y economizador, en el sentido de los gases.

        El calor del evaporador incluye el calentamiento de 2 a 3 en el domo, pero
        sus tubos ven agua a T_sat en los dos extremos: su ΔT frío es el pinch.
        """
        w = self.water
        m = self.m_steam_kg_s
        T = self.T_gas_K
        out: list[HRSGSection] = []
        if self.inputs.superheated:
            out.append(
                HRSGSection(
                    "sobrecalentador",
                    m * (w[4].h_J_per_kg - w[3].h_J_per_kg),
                    T[0],
                    T[1],
                    w[3].T_K,
                    w[4].T_K,
                )
            )
        out.append(  # los tubos ven agua a T_sat: la del economizador se mezcla en el domo
            HRSGSection(
                "evaporador",
                m * (w[3].h_J_per_kg - w[1].h_J_per_kg),
                T[-3],
                T[-2],
                w[2].T_K,
                w[3].T_K,
            )
        )
        out.append(
            HRSGSection(
                "economizador",
                m * (w[1].h_J_per_kg - w[0].h_J_per_kg),
                T[-2],
                T[-1],
                w[0].T_K,
                w[1].T_K,
            )
        )
        return tuple(out)

    @property
    def Q_W(self) -> float:
        """Calor total que reciben el agua y el vapor."""
        return self.m_steam_kg_s * (self.h_steam_J_per_kg - self.water[0].h_J_per_kg)

    @property
    def Q_gas_W(self) -> float:
        """Calor que ceden los gases (igual a :attr:`Q_W`: sin pérdidas al ambiente)."""
        return self.inputs.m_gas_kg_s * (self.h_gas_J_per_kg[0] - self.h_gas_J_per_kg[-1])

    @property
    def Q_available_W(self) -> float:
        """Calor que cederían los gases enfriándose hasta T_ref (15 °C)."""
        gas = self.inputs.gas
        return self.inputs.m_gas_kg_s * (self.h_gas_J_per_kg[0] - gas.h(self.inputs.T_ref_K))

    @property
    def recovery(self) -> float:
        """Aprovechamiento: Q̇ / Q̇ disponible hasta T_ref."""
        return self.Q_W / self.Q_available_W

    @property
    def cp_gas_mean_J_per_kg_K(self) -> float:
        """c_p medio de los gases entre la entrada y la chimenea."""
        return (self.h_gas_J_per_kg[0] - self.h_gas_J_per_kg[-1]) / (
            self.T_gas_K[0] - self.T_gas_K[-1]
        )

    @property
    def dew_point_K(self) -> float | None:
        return self.inputs.gas.dew_point_K


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


def validate_hrsg_inputs(inputs: HRSGInputs) -> None:
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
    lim = fluid_limits(Water)
    p = inputs.p_steam_Pa
    if p <= lim.P_triple_Pa:
        raise ValueError(
            f"La presión de evaporación ({_bar(p)}) está por debajo de la del punto triple del "
            "agua."
        )
    if p >= lim.P_crit_Pa:
        raise ValueError(
            f"La presión de evaporación ({_bar(p)}) supera la crítica del agua "
            f"({_bar(lim.P_crit_Pa)}): por encima no hay evaporación ni domo. Esta página "
            "resuelve calderas subcríticas."
        )
    if not inputs.by_stack and not inputs.pinch_K > 0.0:
        raise ValueError(
            "El pinch tiene que ser positivo: con ΔT = 0 el evaporador necesitaría un área "
            "infinita (los gases y el agua llegarían a la misma temperatura)."
        )
    if not inputs.approach_K >= 0.0:
        raise ValueError(
            "El approach no puede ser negativo: el agua evaporaría dentro del economizador "
            "(con 0 sale como líquido saturado)."
        )
    T_sat = float(PropsSI("T", "P", p, "Q", 0, Water))
    T_eco = T_sat - inputs.approach_K
    if not _T_GAS_MIN_K <= inputs.T_feedwater_K:
        raise ValueError("El agua de alimentación tiene que estar por encima de 0 °C.")
    if inputs.T_feedwater_K >= T_eco:
        raise ValueError(
            f"El agua de alimentación ({_degC(inputs.T_feedwater_K)}) tiene que llegar más fría "
            f"que la salida del economizador, T_sat − approach = {_degC(T_eco)} (T_sat = "
            f"{_degC(T_sat)} a {_bar(p)}): si no, no hay nada que calentar en el economizador."
        )
    if inputs.T_stack_K is not None:
        if not inputs.T_feedwater_K < inputs.T_stack_K < inputs.T_gas_in_K:
            raise ValueError(
                f"La chimenea ({_degC(inputs.T_stack_K)}) tiene que quedar entre el agua de "
                f"alimentación ({_degC(inputs.T_feedwater_K)}) y los gases que entran "
                f"({_degC(inputs.T_gas_in_K)}): los gases se enfrían, pero no por debajo del "
                "agua que calientan."
            )
        if inputs.T_gas_in_K <= T_sat:
            raise ValueError(
                f"Los gases entran a {_degC(inputs.T_gas_in_K)}, sin superar T_sat = "
                f"{_degC(T_sat)}: no alcanzan para evaporar a {_bar(p)}. Bajá la presión de "
                "evaporación."
            )
    else:
        T_pinch = T_sat + inputs.pinch_K
        if inputs.T_gas_in_K <= T_pinch:
            raise ValueError(
                f"Los gases entran a {_degC(inputs.T_gas_in_K)}, sin superar T_sat + pinch = "
                f"{_degC(T_pinch)}: no alcanzan para evaporar a {_bar(p)}. Bajá la presión de "
                "evaporación o el pinch."
            )
    if inputs.T_steam_K is not None:
        if inputs.T_steam_K <= T_sat + 0.01:
            raise ValueError(
                f"El vapor sobrecalentado ({_degC(inputs.T_steam_K)}) tiene que salir más caliente "
                f"que la saturación ({_degC(T_sat)} a {_bar(p)}). Para vapor saturado, elegí "
                "esa opción."
            )
        if inputs.T_steam_K >= inputs.T_gas_in_K:
            raise ValueError(
                f"El vapor ({_degC(inputs.T_steam_K)}) no puede salir más caliente que los gases "
                f"que entran ({_degC(inputs.T_gas_in_K)}): en el sobrecalentador el calor va de "
                "los gases al vapor."
            )
    if not inputs.T_ref_K > 0.0:
        raise ValueError("La temperatura de referencia tiene que ser absoluta positiva.")


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _water_states(inputs: HRSGInputs) -> tuple[FluidState, ...]:
    """1 alimentación, 2 salida del economizador, 3 y 4 saturados, 5 sobrecalentado."""
    p = inputs.p_steam_Pa
    feed = fluid_state_from_pair(Water, "TP", t=inputs.T_feedwater_K, p=p)
    liquid = fluid_state_from_pair(Water, "PX", p=p, x=0.0)
    vapor = fluid_state_from_pair(Water, "PX", p=p, x=1.0)
    if inputs.approach_K == 0.0:
        eco = liquid
    else:
        eco = fluid_state_from_pair(Water, "TP", t=liquid.T_K - inputs.approach_K, p=p)
    states = [feed, eco, liquid, vapor]
    if inputs.T_steam_K is not None:
        states.append(fluid_state_from_pair(Water, "TP", t=inputs.T_steam_K, p=p))
    return tuple(states)


def solve_hrsg(inputs: HRSGInputs) -> HRSGResult:
    """Balance de energía de la HRSG, sección por sección.

    Diseño por pinch (Kehlhofer et al., 2009):

    1. El sobrecalentador y el evaporador juntos (de la entrada de los gases
       al pinch) dan el caudal de vapor:
       ṁ_v·(h_vapor − h_2) = ṁ_g·[h_g(T_a) − h_g(T_c)], con T_c = T_sat + pinch.
    2. El sobrecalentador da la temperatura de los gases entre secciones.
    3. El economizador da la de chimenea.

    Diseño por temperatura de chimenea (Cengel §10-9): el balance de toda la
    caldera da el caudal, ṁ_v·(h_vapor − h_1) = ṁ_g·[h_g(T_a) − h_g(T_d)]; el
    economizador da T_c y el pinch, T_c − T_sat, es un resultado.

    Raises
    ------
    ValueError
        Si los datos no tienen sentido (:func:`validate_hrsg_inputs`), hay un
        cruce de temperaturas o los gases condensarían.
    """
    validate_hrsg_inputs(inputs)
    gas, m_g = inputs.gas, inputs.m_gas_kg_s
    water = _water_states(inputs)
    h = [s.h_J_per_kg for s in water]
    T_sat = water[2].T_K
    T_a = inputs.T_gas_in_K
    h_a = gas.h(T_a)
    if inputs.T_stack_K is None:
        T_c = T_sat + inputs.pinch_K
        h_c = gas.h(T_c)
        m_s = m_g * (h_a - h_c) / (h[-1] - h[1])
        h_d = h_c - m_s * (h[1] - h[0]) / m_g
        try:
            T_d = gas.T_from_h(h_d)
        except ValueError:
            T_d = -math.inf
        if T_d <= inputs.T_feedwater_K:
            raise ValueError(
                "Con ese pinch el economizador tendría un cruce de temperaturas: los gases "
                "tendrían que salir más fríos que el agua de alimentación "
                f"({_degC(inputs.T_feedwater_K)}) para calentar todo el vapor que se produce. "
                "Subí el pinch (menos vapor) o la temperatura del agua de alimentación."
            )
    else:
        T_d = inputs.T_stack_K
        h_d = gas.h(T_d)
        m_s = m_g * (h_a - h_d) / (h[-1] - h[0])
        h_c = h_d + m_s * (h[1] - h[0]) / m_g
        T_c = gas.T_from_h(h_c)
        if T_c <= T_sat:
            raise ValueError(
                f"Con la chimenea a {_degC(T_d)} hay un cruce de temperaturas: el agua tendría "
                f"que empezar a hervir ({_degC(T_sat)}) con los gases a {_degC(T_c)}, más fríos "
                "que ella. Subí la temperatura de chimenea (menos vapor) o bajá la presión de "
                "evaporación."
            )
    temps = [T_a]
    enthalpies = [h_a]
    if inputs.superheated:
        h_b = h_a - m_s * (h[4] - h[3]) / m_g
        temps.append(gas.T_from_h(h_b))
        enthalpies.append(h_b)
    temps += [T_c, T_d]
    enthalpies += [h_c, h_d]
    result = HRSGResult(
        inputs=inputs,
        water=water,
        T_gas_K=tuple(temps),
        h_gas_J_per_kg=tuple(enthalpies),
        m_steam_kg_s=float(m_s),
    )
    dew = gas.dew_point_K
    if dew is not None and T_d < dew:
        raise ValueError(
            f"Los gases saldrían por la chimenea a {_degC(T_d)}, por debajo de su punto de rocío "
            f"({_degC(dew)}): su vapor de agua condensaría, y este modelo (gases ideales, sin "
            "condensación) no lo contempla. Subí la temperatura del agua de alimentación o el "
            "pinch."
        )
    profile = hrsg_tq_profile(result, n=80)
    if profile.min_dT_K <= 0.0:
        raise ValueError(
            "Hay un cruce de temperaturas dentro de la caldera (los gases quedarían más fríos "
            f"que el agua en Q = {profile.min_dT_Q_W / 1e3:.4g} kW). Subí el pinch o la "
            "temperatura del agua de alimentación."
        )
    return result


# ---------------------------------------------------------------------
# Diagrama T–Q
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TQProfile:
    """Curvas del diagrama T–Q (SI): Q acumulado desde la chimenea.

    ``boundaries_W`` son los Q donde cambia de sección (economizador →
    evaporador → sobrecalentador). ``min_dT_K`` es la menor diferencia entre
    los gases y el agua a lo largo de la caldera y ``min_dT_Q_W`` dónde está.
    """

    Q_gas_W: tuple[float, ...]
    T_gas_K: tuple[float, ...]
    Q_water_W: tuple[float, ...]
    T_water_K: tuple[float, ...]
    boundaries_W: tuple[float, ...]
    min_dT_K: float
    min_dT_Q_W: float


def hrsg_tq_profile(result: HRSGResult, n: int = 60) -> TQProfile:
    """Temperaturas de los gases y del agua contra el calor transferido.

    Los gases: Q(T) = ṁ_g·[h_g(T) − h_g(T_chimenea)], que no es una recta
    porque c_p cambia con T. El agua: Q(h) = ṁ_v·(h − h_1), con T(p, h) de
    IAPWS-95 en el economizador y el sobrecalentador. El agua que sale del
    economizador se mezcla en el domo con el agua saturada que circula por el
    evaporador: en el diagrama sube vertical de T_sat − approach a T_sat al
    entrar al evaporador, que es la meseta a T_sat. Así el pinch es la menor
    distancia entre las curvas en la salida de gases del evaporador.
    """
    inputs = result.inputs
    gas, m_g, m_s = inputs.gas, inputs.m_gas_kg_s, result.m_steam_kg_s
    T_d = result.T_stack_K
    temps = np.unique(np.concatenate([np.linspace(T_d, inputs.T_gas_in_K, n), result.T_gas_K]))
    Q_gas = m_g * (gas.h_array(temps) - result.h_gas_J_per_kg[-1])
    w = result.water
    p = inputs.p_steam_Pa
    h1, h2, h4 = w[0].h_J_per_kg, w[1].h_J_per_kg, w[3].h_J_per_kg
    T_sat = result.T_sat_K
    hs = np.linspace(h1, h2, n)
    T_water = [float(PropsSI("T", "P", p, "H", hh, Water)) for hh in hs]
    T_water[-1] = w[1].T_K
    Q_water = [m_s * (hh - h1) for hh in hs]
    for hh in np.linspace(h2, h4, n):  # domo y evaporador: a T_sat
        Q_water.append(m_s * (hh - h1))
        T_water.append(T_sat)
    boundaries = [m_s * (h2 - h1)]
    if inputs.superheated:
        h5 = w[4].h_J_per_kg
        boundaries.append(m_s * (h4 - h1))
        for hh in np.linspace(h4, h5, n)[1:]:
            Q_water.append(m_s * (hh - h1))
            T_water.append(float(PropsSI("T", "P", p, "H", hh, Water)))
        T_water[-1] = w[4].T_K
    T_gas_at = np.interp(Q_water, Q_gas, temps)
    dT = np.asarray(T_gas_at) - np.asarray(T_water)
    k = int(np.argmin(dT))
    return TQProfile(
        Q_gas_W=tuple(float(q) for q in Q_gas),
        T_gas_K=tuple(float(t) for t in temps),
        Q_water_W=tuple(float(q) for q in Q_water),
        T_water_K=tuple(float(t) for t in T_water),
        boundaries_W=tuple(boundaries),
        min_dT_K=float(dT[k]),
        min_dT_Q_W=float(Q_water[k]),
    )


# ---------------------------------------------------------------------
# Notas, ejemplos, barridos y export
# ---------------------------------------------------------------------


def hrsg_notes(result: HRSGResult) -> list[str]:
    """Observaciones didácticas sobre la caldera calculada (markdown)."""
    notes: list[str] = []
    inputs = result.inputs
    pinch = result.pinch_K
    sections = {s.name: s for s in result.sections}
    if inputs.by_stack and pinch < 5.0:
        notes.append(
            f"Con la chimenea a {_degC(result.T_stack_K)} el pinch queda en {pinch:.3g} K: los "
            "gases salen del evaporador casi a la temperatura del agua que hierve, y el "
            "evaporador necesitaría muchísima área. Se diseña con unos 5 a 15 K."
        )
    eco = sections["economizador"]
    if eco.dT_cold_K < pinch:
        notes.append(
            f"En la chimenea los gases quedan solo {eco.dT_cold_K:.3g} K por encima del agua de "
            f"alimentación, menos que el pinch ({pinch:.3g} K): ahí está la menor "
            "diferencia de temperatura de toda la caldera, y el economizador necesita mucha "
            "área."
        )
    sh = sections.get("sobrecalentador")
    if sh is not None and sh.dT_hot_K < pinch:
        notes.append(
            f"En el extremo caliente del sobrecalentador los gases quedan solo {sh.dT_hot_K:.3g} K "
            f"por encima del vapor, menos que el pinch ({pinch:.3g} K): bajá la "
            "temperatura del vapor o el sobrecalentador va a ser muy grande."
        )
    dew = result.dew_point_K
    if dew is not None and inputs.T_feedwater_K < dew:
        notes.append(
            f"El agua de alimentación ({_degC(inputs.T_feedwater_K)}) está por debajo del punto "
            f"de rocío de los gases ({_degC(dew)}): aunque los gases salgan más calientes, su "
            "vapor de agua condensaría sobre los tubos fríos del economizador (con azufre en el "
            "combustible, condensado ácido y corrosión). Por eso se precalienta el agua de "
            "alimentación."
        )
    if inputs.approach_K == 0.0:
        notes.append(
            "Con approach = 0 el economizador entrega líquido saturado: en operación, con "
            "cualquier perturbación empezaría a evaporar adentro (*steaming*). Se diseña con "
            "unos pocos kelvin de approach."
        )
    return notes


#: Escape típico de una turbina de gas a gas natural (aire húmedo), en fracción molar.
_GT_EXHAUST = {"N2": 0.7448, "O2": 0.1225, "CO2": 0.0396, "H2O": 0.0842, "Ar": 0.0089}

_EX_TG = "Turbina de gas: escape a 600 °C y 100 kg/s; vapor a 60 bar y 540 °C"
_EX_SAT = "Vapor saturado para proceso: gases de metano (λ = 3) a 500 °C; 10 bar"
_EX_LOW = "Pinch chico (5 K) y approach 3 K: más vapor, más área"

#: Ejemplos precargados.
HRSG_EXAMPLES: dict[str, HRSGInputs] = {
    _EX_TG: HRSGInputs(
        873.15, 100.0, FlueGas.from_fractions(_GT_EXHAUST), 60.0e5, 333.15, 813.15, 10.0, 5.0
    ),
    _EX_SAT: HRSGInputs(
        773.15, 50.0, FlueGas.from_fractions(exhaust_composition(3.0)), 10.0e5, 378.15
    ),
    _EX_LOW: HRSGInputs(
        873.15, 100.0, FlueGas.from_fractions(_GT_EXHAUST), 60.0e5, 333.15, 813.15, 5.0, 3.0
    ),
}

#: Ejemplos cuyos gases salen de :func:`exhaust_composition` (ejemplo → λ); el resto
#: trae la composición cargada en fracción molar.
HRSG_EXAMPLES_EXCESS_AIR: dict[str, float] = {_EX_SAT: 3.0}

#: Aclaraciones de cada ejemplo para mostrar en la página.
HRSG_EXAMPLE_NOTES: dict[str, str] = {
    _EX_TG: (
        "Escape típico de una turbina de gas a gas natural (en % en volumen: N₂ 74,48; O₂ "
        "12,25; CO₂ 3,96; H₂O 8,42; Ar 0,89). Agua de alimentación a 60 °C, pinch 10 K y "
        "approach 5 K."
    ),
    _EX_SAT: (
        "Caldera de recuperación para vapor de proceso: gases de la combustión del metano con "
        "λ = 3 (aire técnico del vademecum §16) a 500 °C y 50 kg/s, agua de alimentación a "
        "105 °C (después del desaireador) y sin sobrecalentador."
    ),
    _EX_LOW: (
        "El mismo caso de la turbina de gas con pinch 5 K: más vapor y la chimenea más fría, a "
        "cambio de un evaporador y un economizador más grandes."
    ),
}

# El vapor sobrecalentado sale unos 25 K por debajo de los gases (Kehlhofer et al.) y
# no pasa de 565 °C, el límite habitual de los aceros de los sobrecalentadores.
_SH_HOT_END_K = 25.0
_T_STEAM_MAX_K = 838.15


def default_steam_temperature(inputs: HRSGInputs) -> float | None:
    """Temperatura del vapor sobrecalentado sugerida para esos gases (K, redondeada).

    Unos 25 K por debajo de los gases que entran, hasta 565 °C. ``None`` si
    así no queda al menos 10 K por encima de la saturación.
    """
    T_sat = float(PropsSI("T", "P", inputs.p_steam_Pa, "Q", 0, Water))
    T = min(inputs.T_gas_in_K - _SH_HOT_END_K, _T_STEAM_MAX_K)
    T = round(T - 273.15) + 273.15
    return T if T >= T_sat + 10.0 else None


def other_steam_option(inputs: HRSGInputs) -> HRSGInputs | None:
    """La misma caldera con la otra opción de vapor, para comparar.

    Sin sobrecalentador si la dada lo tiene; si no, con vapor a
    :func:`default_steam_temperature`. ``None`` si no hay a qué temperatura
    sobrecalentar.
    """
    if inputs.superheated:
        return replace(inputs, T_steam_K=None)
    T_steam = default_steam_temperature(inputs)
    return None if T_steam is None else replace(inputs, T_steam_K=T_steam)


SweepParameter = Literal["pinch", "approach", "p_steam", "T_steam", "T_feedwater", "T_gas_in"]

_SWEEP_FIELDS: dict[str, str] = {
    "pinch": "pinch_K",
    "approach": "approach_K",
    "p_steam": "p_steam_Pa",
    "T_steam": "T_steam_K",
    "T_feedwater": "T_feedwater_K",
    "T_gas_in": "T_gas_in_K",
}


@dataclass(frozen=True)
class SweepPoint:
    """Un punto del barrido: valor del parámetro (SI) y resultados de la caldera."""

    value_si: float
    m_steam_kg_s: float
    T_stack_K: float
    Q_W: float


def default_sweep_values(inputs: HRSGInputs, parameter: SweepParameter, n: int = 9) -> list[float]:
    """Valores razonables del parámetro alrededor de la caldera dada (SI)."""
    T_sat = float(PropsSI("T", "P", inputs.p_steam_Pa, "Q", 0, Water))
    if parameter == "pinch":
        return [float(v) for v in np.linspace(2.0, 40.0, n)]
    if parameter == "approach":
        return [float(v) for v in np.linspace(0.0, 20.0, n)]
    if parameter == "p_steam":
        # Entre 1 y 160 bar (los domos de una HRSG de una presión), sin acercarse
        # a la T del vapor, al pinch ni al agua de alimentación; siempre incluye
        # la presión dada.
        hi_T = inputs.T_gas_in_K - inputs.pinch_K - 30.0
        if inputs.T_steam_K is not None:
            hi_T = min(hi_T, inputs.T_steam_K - 10.0)
        hi_T = min(hi_T, fluid_limits(Water).T_crit_K - 10.0)
        lo_T = inputs.T_feedwater_K + inputs.approach_K + 20.0
        if hi_T <= lo_T:
            return [inputs.p_steam_Pa]
        lo = max(float(PropsSI("P", "T", lo_T, "Q", 0, Water)), 1.0e5)
        hi = min(float(PropsSI("P", "T", hi_T, "Q", 0, Water)), 160.0e5)
        lo, hi = min(lo, inputs.p_steam_Pa), max(hi, inputs.p_steam_Pa)
        return [float(v) for v in np.geomspace(lo, hi, n)]
    if parameter == "T_steam":
        if inputs.T_steam_K is None:
            return []
        return [float(v) for v in np.linspace(T_sat + 10.0, inputs.T_gas_in_K - 10.0, n)]
    if parameter == "T_feedwater":
        top = T_sat - inputs.approach_K - 10.0
        return [float(v) for v in np.linspace(min(303.15, top - 10.0), top, n)]
    if parameter == "T_gas_in":
        lo = max(T_sat + inputs.pinch_K + 30.0, (inputs.T_steam_K or 0.0) + 20.0)
        hi = min(max(lo + 50.0, inputs.T_gas_in_K + 150.0), T_GAS_MAX_K)
        return [float(v) for v in np.linspace(lo, hi, n)]
    raise ValueError(f"Parámetro de barrido desconocido: {parameter!r}.")


def hrsg_sweep(
    inputs: HRSGInputs, parameter: SweepParameter, values: Sequence[float]
) -> list[SweepPoint]:
    """Resuelve la caldera para cada valor del parámetro; omite los que no tienen sentido."""
    if parameter == "T_steam" and inputs.T_steam_K is None:
        return []
    points: list[SweepPoint] = []
    for value in values:
        try:
            result = solve_hrsg(replace(inputs, **{_SWEEP_FIELDS[parameter]: float(value)}))
        except ValueError:
            continue
        points.append(SweepPoint(float(value), result.m_steam_kg_s, result.T_stack_K, result.Q_W))
    return points


def _value(value_si: float, kind: QuantityKind, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def hrsg_to_dict(result: HRSGResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``)."""
    inputs = result.inputs
    gas = inputs.gas
    water_labels = (
        "alimentación",
        "salida del economizador",
        "líquido saturado",
        "vapor saturado",
        "vapor sobrecalentado",
    )
    return {
        "equipo": "caldera de recuperación (HRSG) de una presión, "
        + ("vapor sobrecalentado" if inputs.superheated else "vapor saturado"),
        "sistema_de_unidades": system,
        "datos": {
            "T_gases_entrada": _value(inputs.T_gas_in_K, "temperature", system),
            "caudal_gases": _value(inputs.m_gas_kg_s, "mass_flow", system),
            "composicion_molar": {GAS_NAMES[s]: y for s, y in gas.mole_fractions.items()},
            "p_gases": _value(gas.p_Pa, "pressure", system),
            "p_evaporacion": _value(inputs.p_steam_Pa, "pressure", system),
            "T_agua_alimentacion": _value(inputs.T_feedwater_K, "temperature", system),
            "T_vapor": (
                None
                if inputs.T_steam_K is None
                else _value(inputs.T_steam_K, "temperature", system)
            ),
            "diseno": "temperatura de chimenea" if inputs.by_stack else "pinch",
            "pinch": (
                None
                if inputs.by_stack
                else _value(inputs.pinch_K, "temperature_difference", system)
            ),
            "T_chimenea": (
                None
                if inputs.T_stack_K is None
                else _value(inputs.T_stack_K, "temperature", system)
            ),
            "approach": _value(inputs.approach_K, "temperature_difference", system),
        },
        "gases": {
            "masa_molar_kg_mol": gas.M_kg_per_mol,
            "fracciones_masicas": {GAS_NAMES[s]: w for s, w in gas.mass_fractions.items()},
            "cp_medio": _value(result.cp_gas_mean_J_per_kg_K, "specific_heat", system),
            "punto_de_rocio": (
                None
                if result.dew_point_K is None
                else _value(result.dew_point_K, "temperature", system)
            ),
            "temperaturas": {
                label: _value(T, "temperature", system)
                for label, T in zip(result.gas_labels, result.T_gas_K, strict=True)
            },
        },
        "resultados": {
            "caudal_vapor": _value(result.m_steam_kg_s, "mass_flow", system),
            "T_saturacion": _value(result.T_sat_K, "temperature", system),
            "pinch": _value(result.pinch_K, "temperature_difference", system),
            "T_chimenea": _value(result.T_stack_K, "temperature", system),
            "calor_total": _value(result.Q_W, "power", system),
            "aprovechamiento": result.recovery,
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
        },
        "agua": [
            {
                "estado": f"{i + 1} ({label})",
                "T": _value(s.T_K, "temperature", system),
                "h": _value(s.h_J_per_kg, _EH, system),
            }
            for i, (label, s) in enumerate(zip(water_labels, result.water, strict=False))
        ],
        "fuente": "Balance de energía con gases ideales (CoolProp) y agua IAPWS-95 (CoolProp)",
    }
