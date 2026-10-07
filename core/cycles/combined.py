"""Ciclo combinado gas–vapor de una presión — Fase 3.4.

Una **turbina de gas** (:mod:`core.cycles.brayton`) descarga sus gases
calientes en una **caldera de recuperación de una presión**
(:mod:`core.cycles.hrsg`), que produce el vapor de un **ciclo de Rankine**
(:mod:`core.cycles.rankine`): turbina de vapor, condensador, bomba y,
opcionalmente, un desaireador (calentador abierto alimentado con una
extracción). Cengel & Boles §10-9; Kehlhofer, Hannemann, Stirnimann y Rukes,
*Combined-Cycle Gas & Steam Turbine Power Plants* (3.ª ed., PennWell, 2009).

El acople:

1. La turbina de gas da la temperatura, el caudal y la composición del escape.
2. El ciclo de vapor (por kg de vapor) da el estado del agua que la bomba
   entrega a la caldera: es el agua de alimentación de la HRSG.
3. La HRSG (por pinch o por temperatura de chimenea) da el caudal de vapor,
   que entra a la turbina de vapor a la presión y la temperatura de la caldera.

Rendimiento: η_CC = (Ẇ_TG + Ẇ_TV) / Q̇_comb. Con el calor del escape
Q̇_esc = Q̇_comb − Ẇ_TG y η_HRSG = Q̇_HRSG / Q̇_esc vale exacta la relación de
Kehlhofer η_CC = η_TG + η_HRSG·η_TV·(1 − η_TG). Todo en SI. No importa
Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from core.cycles.brayton import BraytonInputs, BraytonResult, brayton_notes, solve_brayton
from core.cycles.hrsg import HRSGInputs, HRSGResult, hrsg_notes, hrsg_to_dict, solve_hrsg
from core.cycles.rankine import (
    FeedwaterHeater,
    RankineInputs,
    RankineResult,
    rankine_notes,
    rankine_to_dict,
    solve_rankine,
)
from core.ideal_gas import AIR_DRY, GAS_NAMES, T_GAS_MAX_K
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "SH_HOT_END_K",
    "COMBINED_EXAMPLES",
    "COMBINED_EXAMPLE_NOTES",
    "CombinedInputs",
    "CombinedResult",
    "SteamCycle",
    "SweepParameter",
    "SweepPoint",
    "combined_notes",
    "combined_sweep",
    "combined_to_dict",
    "default_sweep_values",
    "gas_turbine_to_dict",
    "solve_combined",
]


@dataclass(frozen=True)
class SteamCycle:
    """Datos del ciclo de vapor (el Rankine de las Fases 3.1), en SI.

    ``deaerator_p_Pa`` es la presión del desaireador: un calentador abierto
    alimentado con una extracción de la turbina (Cengel §10-6). ``None`` =
    sin desaireador: el condensado va directo de la bomba a la caldera.
    """

    p_condenser_Pa: float
    eta_turbine: float = 1.0
    eta_pump: float = 1.0
    deaerator_p_Pa: float | None = None


@dataclass(frozen=True)
class CombinedInputs:
    """Datos del ciclo combinado de una presión, en SI.

    ``T_steam_K=None`` es vapor saturado (HRSG sin sobrecalentador). Con
    ``T_stack_K`` la HRSG se diseña por la temperatura de chimenea (Cengel
    §10-9) y ``pinch_K`` no se usa. Si se da ``W_net_W`` (potencia neta del
    ciclo combinado), el caudal de aire de ``gas_turbine`` se ajusta para
    lograrla: todo escala con el caudal.
    """

    gas_turbine: BraytonInputs
    p_steam_Pa: float
    T_steam_K: float | None
    steam: SteamCycle
    pinch_K: float = 10.0
    approach_K: float = 5.0
    T_stack_K: float | None = None
    W_net_W: float | None = None


@dataclass(frozen=True)
class CombinedResult:
    """Turbina de gas, HRSG y ciclo de vapor resueltos y acoplados (SI)."""

    inputs: CombinedInputs
    gas_turbine: BraytonResult
    hrsg: HRSGResult
    steam: RankineResult

    @property
    def W_gas_turbine_W(self) -> float:
        return self.gas_turbine.W_net_W

    @property
    def W_steam_turbine_W(self) -> float:
        """Potencia neta del ciclo de vapor (turbina menos bombas)."""
        return self.steam.W_net_W

    @property
    def W_net_W(self) -> float:
        return self.W_gas_turbine_W + self.W_steam_turbine_W

    @property
    def Q_fuel_W(self) -> float:
        """Calor del combustible (ṁ_f·PCI), o el que entra al aire estándar."""
        return self.gas_turbine.Q_in_W

    @property
    def eta_th(self) -> float:
        """Rendimiento del ciclo combinado, η_CC = (Ẇ_TG + Ẇ_TV) / Q̇_comb."""
        return self.W_net_W / self.Q_fuel_W

    @property
    def eta_gas_turbine(self) -> float:
        return self.gas_turbine.eta_th

    @property
    def eta_steam(self) -> float:
        return self.steam.eta_th

    @property
    def Q_exhaust_W(self) -> float:
        """Calor del escape: lo que la turbina de gas no convierte, Q̇_comb − Ẇ_TG."""
        return self.Q_fuel_W - self.W_gas_turbine_W

    @property
    def Q_hrsg_W(self) -> float:
        return self.hrsg.Q_W

    @property
    def eta_hrsg(self) -> float:
        """Aprovechamiento del escape en la HRSG: Q̇_HRSG / (Q̇_comb − Ẇ_TG)."""
        return self.Q_hrsg_W / self.Q_exhaust_W

    @property
    def Q_condenser_W(self) -> float:
        return self.steam.Q_out_W

    @property
    def Q_stack_W(self) -> float:
        """Lo que se va por la chimenea (respecto del aire que entra)."""
        return self.Q_exhaust_W - self.Q_hrsg_W

    @property
    def steam_gas_ratio(self) -> float:
        """ṁ_v / ṁ_g (la y de Cengel 10-9)."""
        return self.hrsg.m_steam_kg_s / self.gas_turbine.m_gas_kg_s

    @property
    def heat_rate_kJ_per_kWh(self) -> float:
        return 3600.0 / self.eta_th

    @property
    def energy_balance_W(self) -> dict[str, float]:
        """Destino del calor del combustible: Q̇_comb = Ẇ_TG + Ẇ_TV + Q̇_cond + Q̇_chim."""
        return {
            "turbina de gas": self.W_gas_turbine_W,
            "turbina de vapor": self.W_steam_turbine_W,
            "condensador": self.Q_condenser_W,
            "chimenea": self.Q_stack_W,
        }


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _staged(stage: str, exc: ValueError) -> ValueError:
    return ValueError(f"{stage}: {exc}")


def _steam_inputs(inputs: CombinedInputs, m_dot: float) -> RankineInputs:
    steam = inputs.steam
    heaters = () if steam.deaerator_p_Pa is None else (FeedwaterHeater(steam.deaerator_p_Pa),)
    return RankineInputs(
        inputs.p_steam_Pa,
        inputs.T_steam_K,
        steam.p_condenser_Pa,
        steam.eta_turbine,
        steam.eta_pump,
        m_dot_kg_s=m_dot,
        heaters=heaters,
    )


def _scaled_steam(steam: RankineResult, m_dot: float) -> RankineResult:
    """El mismo ciclo de vapor con otro caudal (todo escala con ṁ)."""
    return replace(steam, inputs=replace(steam.inputs, m_dot_kg_s=m_dot), m_dot_kg_s=m_dot)


def _scaled(result: CombinedResult, factor: float) -> CombinedResult:
    """El mismo ciclo combinado con todos los caudales multiplicados por ``factor``."""
    gt = result.gas_turbine
    gt = replace(gt, inputs=replace(gt.inputs, m_air_kg_s=gt.inputs.m_air_kg_s * factor))
    hrsg = result.hrsg
    hrsg = replace(
        hrsg,
        inputs=replace(hrsg.inputs, m_gas_kg_s=hrsg.inputs.m_gas_kg_s * factor),
        m_steam_kg_s=hrsg.m_steam_kg_s * factor,
    )
    steam = _scaled_steam(result.steam, hrsg.m_steam_kg_s)
    return replace(result, gas_turbine=gt, hrsg=hrsg, steam=steam)


def solve_combined(inputs: CombinedInputs) -> CombinedResult:
    """Resuelve y acopla la turbina de gas, la HRSG y el ciclo de vapor.

    Raises
    ------
    ValueError
        Con el mensaje de la parte que no tiene sentido físico (turbina de gas,
        caldera de recuperación o ciclo de vapor).
    """
    if inputs.W_net_W is not None and not (math.isfinite(inputs.W_net_W) and inputs.W_net_W > 0.0):
        raise ValueError("La potencia neta tiene que ser positiva.")
    steam = inputs.steam
    if steam.deaerator_p_Pa is not None and not (
        steam.p_condenser_Pa < steam.deaerator_p_Pa < inputs.p_steam_Pa
    ):
        raise ValueError(
            "Ciclo de vapor: el desaireador tiene que trabajar entre la presión del "
            "condensador y la de la caldera (típico: 1,2 a 2 bar, con el agua a 105–120 °C)."
        )
    try:
        gt = solve_brayton(inputs.gas_turbine)
    except ValueError as exc:
        raise _staged("Turbina de gas", exc) from exc
    try:
        rankine = solve_rankine(_steam_inputs(inputs, 1.0))
    except ValueError as exc:
        raise _staged("Ciclo de vapor", exc) from exc
    boiler = rankine.of_kind("boiler")[0]
    feedwater = rankine.states[boiler.port("in").state]
    hrsg_inputs = HRSGInputs(
        T_gas_in_K=gt.T_exhaust_K,
        m_gas_kg_s=gt.m_gas_kg_s,
        gas=gt.gas,
        p_steam_Pa=inputs.p_steam_Pa,
        T_feedwater_K=feedwater.T_K,
        T_steam_K=inputs.T_steam_K,
        pinch_K=inputs.pinch_K,
        approach_K=inputs.approach_K,
        T_ref_K=inputs.gas_turbine.T_amb_K,
        T_stack_K=inputs.T_stack_K,
    )
    try:
        hrsg = solve_hrsg(hrsg_inputs)
    except ValueError as exc:
        raise _staged("Caldera de recuperación", exc) from exc
    result = CombinedResult(
        inputs=inputs,
        gas_turbine=gt,
        hrsg=hrsg,
        steam=_scaled_steam(rankine, hrsg.m_steam_kg_s),
    )
    if inputs.W_net_W is not None:
        result = _scaled(result, inputs.W_net_W / result.W_net_W)
    return result


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def combined_notes(result: CombinedResult, system: UnitSystem = "Técnico") -> list[str]:
    """Observaciones didácticas del ciclo combinado (markdown): las de cada parte y el acople."""
    notes: list[str] = []
    gain = result.eta_th - result.eta_gas_turbine
    notes.append(
        f"El ciclo de vapor sube el rendimiento de {result.eta_gas_turbine * 100:.1f} % (la "
        f"turbina de gas sola) a {result.eta_th * 100:.1f} %: {gain * 100:.1f} puntos más con el "
        "mismo combustible, porque aprovecha el calor que la turbina de gas tiraría por el "
        "escape (Cengel §10-9)."
    )
    notes += brayton_notes(result.gas_turbine)
    notes += hrsg_notes(result.hrsg)
    notes += rankine_notes(result.steam, system)
    return notes


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------

_EX_TYPICAL = (
    "Ciclo combinado de una presión: turbina de gas a metano (r_p 15, TIT 1250 °C) y vapor a "
    "60 bar y 540 °C"
)
_EX_CENGEL = "Cengel 10-9 — aire estándar; los gases salen del intercambiador a 450 K"
_EX_MODERN = "Turbina de gas moderna (r_p 17, TIT 1400 °C) y vapor a 100 bar y 565 °C"

#: Ejemplos precargados.
COMBINED_EXAMPLES: dict[str, CombinedInputs] = {
    _EX_TYPICAL: CombinedInputs(
        gas_turbine=BraytonInputs(
            15.0, 1523.15, 0.88, 0.90, dp_combustor=0.03, air=AIR_DRY, m_air_kg_s=500.0
        ),
        p_steam_Pa=60e5,
        T_steam_K=813.15,
        steam=SteamCycle(0.08e5, 0.88, 0.80, 1.5e5),
        pinch_K=10.0,
        approach_K=5.0,
    ),
    _EX_CENGEL: CombinedInputs(
        gas_turbine=BraytonInputs(
            8.0, 1300.0, 0.80, 0.85, T_amb_K=300.0, p_amb_Pa=100e3, fuel=None, m_air_kg_s=1.0
        ),
        p_steam_Pa=7e6,
        T_steam_K=773.15,
        steam=SteamCycle(5e3),
        approach_K=0.0,
        T_stack_K=450.0,
    ),
    _EX_MODERN: CombinedInputs(
        gas_turbine=BraytonInputs(
            17.0, 1673.15, 0.90, 0.91, dp_combustor=0.03, air=AIR_DRY, m_air_kg_s=650.0
        ),
        p_steam_Pa=100e5,
        T_steam_K=838.15,
        steam=SteamCycle(0.06e5, 0.90, 0.80, 1.5e5),
        pinch_K=8.0,
        approach_K=5.0,
    ),
}

#: Aclaraciones de cada ejemplo para mostrar en la página.
COMBINED_EXAMPLE_NOTES: dict[str, str] = {
    _EX_TYPICAL: (
        "Una turbina de gas de potencia media (500 kg/s de aire a 15 °C, η_C = 0,88, η_T = 0,90, "
        "3 % de pérdida en la cámara) y una HRSG de una presión con pinch 10 K y approach 5 K. "
        "El ciclo de vapor condensa a 0,08 bar y tiene un desaireador a 1,5 bar."
    ),
    _EX_CENGEL: (
        "Cengel & Boles, ejemplo 10-9 (8.ª ed.): turbina de gas de aire estándar con r_p = 8, "
        "aire a 300 K y 1300 K a la entrada de la turbina (η_C = 0,80, η_T = 0,85); Rankine "
        "ideal entre 7 MPa y 5 kPa con vapor a 500 °C. El intercambiador se diseña por la "
        "temperatura de salida de los gases (450 K). El libro da ṁ_v/ṁ_g = 0,131 y η = 48,7 %."
    ),
    _EX_MODERN: (
        "Una turbina de gas más moderna (r_p = 17, TIT = 1400 °C, η_C = 0,90, η_T = 0,91) con "
        "vapor a 100 bar y 565 °C, pinch 8 K, condensador a 0,06 bar y desaireador a 1,5 bar. "
        "Las centrales reales de este tipo usan HRSG de tres presiones con recalentamiento."
    ),
}


# ---------------------------------------------------------------------
# Barridos
# ---------------------------------------------------------------------

SweepParameter = Literal["pressure_ratio", "T_turbine_in", "p_steam", "pinch", "T_stack"]


@dataclass(frozen=True)
class SweepPoint:
    """Un punto del barrido: valor del parámetro (SI) y resultados del ciclo combinado.

    ``T_steam_K`` es la temperatura del vapor que se usó: en los barridos de la
    turbina de gas se limita a 25 K por debajo del escape.
    """

    value_si: float
    eta: float
    eta_gas_turbine: float
    eta_steam: float
    w_net_J_per_kg_air: float
    T_exhaust_K: float
    T_steam_K: float | None


#: En los barridos de la turbina de gas, el vapor se sobrecalienta como mucho hasta
#: 25 K por debajo de los gases de escape (Kehlhofer et al.).
SH_HOT_END_K = 25.0


def _with(inputs: CombinedInputs, parameter: SweepParameter, value: float) -> CombinedInputs:
    gt = inputs.gas_turbine
    if parameter == "pressure_ratio":
        return replace(inputs, gas_turbine=replace(gt, pressure_ratio=value))
    if parameter == "T_turbine_in":
        return replace(inputs, gas_turbine=replace(gt, T_turbine_in_K=value))
    if parameter == "p_steam":
        return replace(inputs, p_steam_Pa=value)
    if parameter == "pinch":
        return replace(inputs, pinch_K=value)
    if parameter == "T_stack":
        return replace(inputs, T_stack_K=value)
    raise ValueError(f"Parámetro de barrido desconocido: {parameter!r}.")


def default_sweep_values(
    inputs: CombinedInputs, parameter: SweepParameter, n: int = 9
) -> list[float]:
    """Valores razonables del parámetro alrededor del ciclo dado (SI)."""
    gt = inputs.gas_turbine
    if parameter == "pressure_ratio":
        return [float(v) for v in np.geomspace(6.0, 40.0, n)]
    if parameter == "T_turbine_in":
        hi = min(T_GAS_MAX_K - 50.0, max(gt.T_turbine_in_K + 200.0, 1373.15))
        lo = min(gt.T_turbine_in_K - 200.0, hi - 300.0)
        return [float(v) for v in np.linspace(lo, hi, n)]
    if parameter == "p_steam":
        lo = 5e5
        if inputs.steam.deaerator_p_Pa is not None:
            lo = max(lo, 2.0 * inputs.steam.deaerator_p_Pa)
        return [float(v) for v in np.geomspace(lo, 160e5, n)]
    if parameter == "pinch":
        return [float(v) for v in np.linspace(2.0, 40.0, n)]
    if parameter == "T_stack":
        base = inputs.T_stack_K or 423.15
        return [float(v) for v in np.linspace(base - 60.0, base + 100.0, n)]
    raise ValueError(f"Parámetro de barrido desconocido: {parameter!r}.")


def _limit_steam_temperature(inputs: CombinedInputs) -> CombinedInputs:
    """El vapor no puede salir más caliente que T_escape − 25 K (barridos de la TG)."""
    if inputs.T_steam_K is None:
        return inputs
    T_max = solve_brayton(inputs.gas_turbine).T_exhaust_K - SH_HOT_END_K
    return replace(inputs, T_steam_K=min(inputs.T_steam_K, T_max))


def combined_sweep(
    inputs: CombinedInputs, parameter: SweepParameter, values: Sequence[float]
) -> list[SweepPoint]:
    """Resuelve el ciclo combinado para cada valor; omite los que no tienen sentido físico.

    Al barrer la relación de presiones o la TIT cambia la temperatura del
    escape: el vapor se sobrecalienta hasta la pedida o, si los gases salen más
    fríos, hasta 25 K por debajo de ellos.
    """
    points: list[SweepPoint] = []
    for value in values:
        try:
            case = replace(_with(inputs, parameter, float(value)), W_net_W=None)
            if parameter in ("pressure_ratio", "T_turbine_in"):
                case = _limit_steam_temperature(case)
            r = solve_combined(case)
        except ValueError:
            continue
        points.append(
            SweepPoint(
                float(value),
                r.eta_th,
                r.eta_gas_turbine,
                r.eta_steam,
                r.W_net_W / r.gas_turbine.m_air_kg_s,
                r.gas_turbine.T_exhaust_K,
                r.inputs.T_steam_K,
            )
        )
    return points


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _value(value_si: float, kind: QuantityKind, system: UnitSystem) -> dict[str, Any]:
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def gas_turbine_to_dict(gt: BraytonResult, system: UnitSystem) -> dict[str, Any]:
    """La turbina de gas de un resultado, serializable a JSON (valores en ``system``)."""
    gi = gt.inputs
    fuel = gi.fuel
    return {
        "modelo": (
            "aire estándar"
            if fuel is None
            else f"combustión de {'metano' if fuel.name == 'methane' else fuel.name}"
        ),
        "relacion_de_presiones": gi.pressure_ratio,
        "rendimiento_compresor": gi.eta_compressor,
        "rendimiento_turbina": gi.eta_turbine,
        "caida_de_presion_camara": gi.dp_combustor,
        "T_ambiente": _value(gi.T_amb_K, "temperature", system),
        "p_ambiente": _value(gi.p_amb_Pa, "pressure", system),
        "TIT": _value(gi.T_turbine_in_K, "temperature", system),
        "composicion_aire": {GAS_NAMES[s]: y for s, y in gi.air.mole_fractions.items()},
        "PCI": None if fuel is None else _value(fuel.lhv_J_per_kg, "specific_enthalpy", system),
        "estados": [
            {
                "estado": s.label,
                "medio": s.medium,
                "T": _value(s.T_K, "temperature", system),
                "p": _value(s.P_Pa, "pressure", system),
                "h_desde_25C": _value(s.h_J_per_kg, "specific_enthalpy", system),
                "s": _value(s.s_J_per_kg_K, "specific_entropy", system),
            }
            for s in gt.states
        ],
        "relacion_combustible_aire": gt.fuel_air_ratio,
        "exceso_de_aire": gt.excess_air,
        "composicion_escape": {GAS_NAMES[s]: y for s, y in gt.gas.mole_fractions.items()},
        "caudal_aire": _value(gt.m_air_kg_s, "mass_flow", system),
        "caudal_combustible": _value(gt.m_fuel_kg_s, "mass_flow", system),
        "potencia_compresor": _value(gt.W_compressor_W, "power", system),
        "potencia_turbina": _value(gt.W_turbine_W, "power", system),
        "potencia_neta": _value(gt.W_net_W, "power", system),
        "rendimiento": gt.eta_th,
        "relacion_trabajo_retroceso": gt.back_work_ratio,
    }


def combined_to_dict(result: CombinedResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``)."""
    gt = result.gas_turbine
    return {
        "equipo": "ciclo combinado gas–vapor de una presión",
        "sistema_de_unidades": system,
        "turbina_de_gas": gas_turbine_to_dict(gt, system),
        "hrsg": hrsg_to_dict(result.hrsg, system),
        "ciclo_de_vapor": rankine_to_dict(result.steam, system),
        "resultados": {
            "potencia_turbina_de_gas": _value(result.W_gas_turbine_W, "power", system),
            "potencia_ciclo_de_vapor": _value(result.W_steam_turbine_W, "power", system),
            "potencia_neta": _value(result.W_net_W, "power", system),
            "calor_combustible": _value(result.Q_fuel_W, "power", system),
            "rendimiento_ciclo_combinado": result.eta_th,
            "rendimiento_turbina_de_gas": result.eta_gas_turbine,
            "rendimiento_ciclo_de_vapor": result.eta_steam,
            "aprovechamiento_hrsg": result.eta_hrsg,
            "relacion_vapor_gases": result.steam_gas_ratio,
            "heat_rate_kJ_kWh": result.heat_rate_kJ_per_kWh,
            "balance_de_energia": {
                name: _value(value, "power", system)
                for name, value in result.energy_balance_W.items()
            },
        },
        "fuente": (
            "Turbina de gas y HRSG con gases ideales (CoolProp), PCI de ISO 6976:2016 a 25 °C; "
            "ciclo de vapor con TESPy y agua IAPWS-95"
        ),
    }
