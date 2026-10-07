"""Procedimiento del ciclo combinado — Fase 3.4.

Junta los procedimientos de las tres partes (turbina de gas, caldera de
recuperación y ciclo de vapor) y suma el acople y el rendimiento del ciclo
combinado: η_CC = (Ẇ_TG + Ẇ_TV) / Q̇_comb y la relación de Kehlhofer
η_CC = η_TG + η_HRSG·η_TV·(1 − η_TG), con el balance de energía de toda la
planta. Recibe un resultado ya calculado: no importa Streamlit.
"""

from __future__ import annotations

from core.cycles.brayton_procedure import brayton_steps
from core.cycles.combined import CombinedResult
from core.cycles.hrsg_procedure import hrsg_steps
from core.cycles.rankine_procedure import _ix, _n, _q, rankine_steps
from core.latex import latex_chain, latex_number
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem

__all__ = ["combined_sections"]

_EH: QuantityKind = "specific_enthalpy"
_P: QuantityKind = "power"


def _pct(value: float) -> str:
    return rf"{latex_number(value * 100, 4)}\,\%"


def _coupling_step(result: CombinedResult, system: UnitSystem) -> ProcedureStep:
    steam = result.steam
    boiler = steam.of_kind("boiler")[0]
    k = boiler.port("in").state + 1
    j = boiler.port("out").state + 1
    top = 5 if result.hrsg.inputs.superheated else 4
    feed = steam.states[k - 1]
    return ProcedureStep(
        title="Acople: HRSG y ciclo de vapor",
        text=(
            f"La bomba del ciclo de vapor entrega el agua a la caldera de recuperación: el "
            f"estado {k} del ciclo de vapor es el agua de alimentación (1) de la HRSG. El vapor "
            f"que sale de la HRSG ({top}) entra a la turbina de vapor (estado {j} del ciclo). La "
            "HRSG fija el caudal de vapor, y el ciclo de vapor da el trabajo por kilogramo."
        ),
        latex=(
            rf"T_1^{{\mathrm{{HRSG}}}} = T_{_ix(k)} = {_q(feed.T_K, 'temperature', system)}",
            rf"\dot{{m}}_v = {_q(result.hrsg.m_steam_kg_s, 'mass_flow', system)}",
            latex_chain(
                r"\dot{W}_{TV}",
                r"\dot{m}_v\,w_{\mathrm{neto},TV}",
                rf"{_n(result.hrsg.m_steam_kg_s, 'mass_flow', system)}\cdot "
                rf"{_n(steam.w_net_J_per_kg, _EH, system)}",
                _q(result.W_steam_turbine_W, _P, system),
            ),
        ),
    )


def _efficiency_step(result: CombinedResult, system: UnitSystem) -> ProcedureStep:
    W_gt, W_st, W = result.W_gas_turbine_W, result.W_steam_turbine_W, result.W_net_W
    eta_gt, eta_st, eta_h = result.eta_gas_turbine, result.eta_steam, result.eta_hrsg
    fuel = "del combustible" if not result.gas_turbine.inputs.air_standard else "que entra"
    return ProcedureStep(
        title="Rendimiento del ciclo combinado",
        text=(
            f"El calor {fuel} entra una sola vez, en la turbina de gas; el ciclo de vapor "
            "trabaja con el calor del escape. Si η_HRSG es la parte del calor del escape "
            "(Q̇_comb − Ẇ_TG) que recupera la caldera, el rendimiento del ciclo combinado "
            "cumple la relación de Kehlhofer et al. (2009)."
        ),
        latex=(
            latex_chain(
                r"\dot{W}",
                r"\dot{W}_{TG} + \dot{W}_{TV}",
                rf"{_n(W_gt, _P, system)} + {_n(W_st, _P, system)}",
                _q(W, _P, system),
            ),
            latex_chain(
                r"\eta_{CC}",
                r"\frac{\dot{W}}{\dot{Q}_{\mathrm{comb}}}",
                rf"\frac{{{_n(W, _P, system)}}}{{{_n(result.Q_fuel_W, _P, system)}}}",
                _pct(result.eta_th),
            ),
            latex_chain(
                r"\eta_{\mathrm{HRSG}}",
                r"\frac{\dot{Q}_{\mathrm{HRSG}}}{\dot{Q}_{\mathrm{comb}} - \dot{W}_{TG}}",
                rf"\frac{{{_n(result.Q_hrsg_W, _P, system)}}}"
                rf"{{{_n(result.Q_exhaust_W, _P, system)}}}",
                _pct(eta_h),
            ),
            latex_chain(
                r"\eta_{CC}",
                r"\eta_{TG} + \eta_{\mathrm{HRSG}}\,\eta_{TV}\,(1 - \eta_{TG})",
                rf"{latex_number(eta_gt, 4)} + {latex_number(eta_h, 4)}\cdot "
                rf"{latex_number(eta_st, 4)} \\ &\quad \cdot (1 - {latex_number(eta_gt, 4)})",
                _pct(result.eta_th),
            ),
        ),
    )


def _balance_step(result: CombinedResult, system: UnitSystem) -> ProcedureStep:
    Q = result.Q_fuel_W
    flows = result.energy_balance_W
    parts = ", ".join(f"{name} {value / Q * 100:.1f} %" for name, value in flows.items())
    numbers = [_n(v, _P, system) for v in flows.values()]
    rows = [
        r"\dot{Q}_{\mathrm{comb}} &= \dot{W}_{TG} + \dot{W}_{TV}"
        r" \\ &\quad + \dot{Q}_{\mathrm{cond}} + \dot{Q}_{\mathrm{chim}}",
        rf"&= {numbers[0]}",
        *(rf"&\quad + {v}" for v in numbers[1:]),
        rf"&= {_q(sum(flows.values()), _P, system)}",
    ]
    return ProcedureStep(
        title="Balance de energía de la planta",
        text=(
            "Todo el calor que entra sale como trabajo de las dos turbinas, como calor en el "
            "condensador del ciclo de vapor o por la chimenea (los gases salen más calientes que "
            f"el aire que entra). De cada 100 unidades: {parts}."
        ),
        latex=(r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}",),
    )


def combined_sections(
    result: CombinedResult, system: UnitSystem
) -> list[tuple[str, list[ProcedureStep]]]:
    """Procedimiento por partes: (título de la parte, pasos)."""
    return [
        ("Turbina de gas", brayton_steps(result.gas_turbine, system)),
        # La composición de los gases ya está en la turbina de gas.
        ("Caldera de recuperación", hrsg_steps(result.hrsg, system)[1:]),
        ("Ciclo de vapor", rankine_steps(result.steam, system)),
        (
            "Ciclo combinado",
            [
                _coupling_step(result, system),
                _efficiency_step(result, system),
                _balance_step(result, system),
            ],
        ),
    ]
