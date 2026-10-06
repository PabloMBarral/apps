"""Procedimiento del ciclo de Rankine «como con las tablas» — Fases 3.1a, 3.1b y 3.1c.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo (vademecum §3.3, §9.2, §10.4, §12 y §13; Çengel & Boles,
*Termodinámica*, cap. 10). Recibe un resultado ya calculado
(:class:`core.cycles.rankine.RankineResult`): no importa TESPy ni Streamlit.

Sin calentadores, pérdidas ni recuperador sigue el orden de la Fase 3.1a:
bomba, caldera, turbinas, condensador y balance. Si no, sigue el orden de
Cengel §10-3 y §10-6: primero los estados que se leen en las tablas (con las
caídas de presión y de temperatura del ciclo real y el recuperador), después
los balances de los calentadores —de mayor a menor presión— para despejar
las fracciones de extracción (o, si un drenaje bombeado las acopla, el
sistema resuelto y su verificación) y al final los calores y trabajos por
kilogramo de vapor que pasa por la caldera.

Las ecuaciones se arman con :func:`core.latex.latex_chain` (una igualdad por
renglón) para que entren en el ancho de un celular.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.cycles.rankine_layout import Flow, port_flows
from core.fluids import (
    _el,
    fluid_limits,
    fluid_state_from_pair,
    saturation_at_pressure,
    saturation_at_temperature,
)
from core.latex import latex_chain, latex_number, latex_paren, latex_quantity, latex_value
from core.state_report import ProcedureStep, pv_energy_factor
from core.units_system import QuantityKind, UnitSystem, unit_label

if TYPE_CHECKING:
    from core.cycles.rankine import CoolingWaterResult, RankineResult
    from core.cycles.rankine_layout import CycleComponent

_EH: QuantityKind = "specific_enthalpy"
_ES: QuantityKind = "specific_entropy"


# ---------------------------------------------------------------------
# Helpers de texto y LaTeX
# ---------------------------------------------------------------------


def _bar(p_Pa: float) -> str:
    return f"{p_Pa / 1e5:.4g} bar"


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


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


def _ix(n: int) -> str:
    """Subíndice LaTeX de un número de estado: ``h_3`` o ``h_{13}``."""
    return str(n) if n < 10 else f"{{{n}}}"


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


# ---------------------------------------------------------------------
# Textos según el fluido (agua o ORC) y el equipo donde entra el calor
# ---------------------------------------------------------------------

_WATER = "Water"
_MASCULINE = frozenset({"evaporador"})


def _boiler(result: RankineResult) -> str:
    """«caldera» (ciclo de vapor) o «evaporador» (ORC)."""
    return result.inputs.boiler_name


def _the(name: str) -> str:
    return f"el {name}" if name in _MASCULINE else f"la {name}"


def _of(name: str) -> str:
    return f"del {name}" if name in _MASCULINE else f"de la {name}"


def _to(name: str) -> str:
    return f"al {name}" if name in _MASCULINE else f"a la {name}"


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _q_boiler(result: RankineResult) -> str:
    """Símbolo del calor de la caldera (q_cald) o del evaporador (q_evap)."""
    return r"q_{\mathrm{cald}}" if _boiler(result) == "caldera" else r"q_{\mathrm{evap}}"


def _fluid(result: RankineResult) -> str:
    """«el agua», «el R-245fa», «el tolueno»… para usar dentro de una oración."""
    return _el(result.inputs.fluid)


def _read(result: RankineResult, table: str) -> str:
    """Dónde se lee una propiedad: la tabla de Cengel que corresponde o la ecuación de estado.

    ``table``: ``"sat_p"`` (saturación por presión), ``"sat"`` (saturación),
    ``"vapor"`` (vapor), ``"superheated"`` (vapor sobrecalentado) o
    ``"compressed"`` (líquido comprimido).
    """
    fluid = result.inputs.fluid
    if fluid == _WATER:
        return {
            "sat_p": "la tabla de saturación por presión (Cengel A-5)",
            "sat": "la tabla de saturación",
            "vapor": "la tabla de vapor (Cengel A-6)",
            "superheated": "la tabla de vapor sobrecalentado (Cengel A-6)",
            "compressed": "la tabla de líquido comprimido (Cengel A-7)",
        }[table]
    if fluid == "R134a" and table != "compressed":
        return {
            "sat_p": "la tabla de saturación del R-134a por presión (Cengel A-12)",
            "sat": "la tabla de saturación del R-134a (Cengel A-12)",
            "vapor": "la tabla de vapor sobrecalentado del R-134a (Cengel A-13)",
            "superheated": "la tabla de vapor sobrecalentado del R-134a (Cengel A-13)",
        }[table]
    return f"la ecuación de estado (CoolProp; Cengel no tabula {_el(fluid)})"


def _cite(text: str, ref: str) -> str:
    """Agrega una cita: «… (Cengel A-12; vademecum §12)» o «… (vademecum §12)»."""
    if text.endswith(")"):
        return f"{text[:-1]}; {ref})"
    return f"{text} ({ref})"


def _tabulated(result: RankineResult) -> bool:
    return result.inputs.fluid in (_WATER, "R134a")


def _lookup(
    result: RankineResult, table: str, *, interpolate: bool = False, plural: bool = False
) -> str:
    """«se lee en la tabla …» o, si Cengel no tabula el fluido, «sale de la ecuación de estado»."""
    if result.inputs.fluid == "R134a" and table == "compressed":
        table = "sat"
    if _tabulated(result):
        verb = "se leen en" if plural else "se lee en"
        return f"{verb} {_read(result, table)}" + (", interpolando" if interpolate else "")
    verb = "salen de" if plural else "sale de"
    return f"{verb} {_read(result, table)}"


# ---------------------------------------------------------------------
# Pasos que comparten el ciclo simple y el regenerativo
# ---------------------------------------------------------------------


def _condenser_outlet_step(result: RankineResult, system: UnitSystem) -> ProcedureStep:
    s1 = result.states[0]
    losses = result.inputs.losses
    saturated = (
        rf"T_1 = T_{{\mathrm{{sat}}}}(p_1) = {_q(s1.T_K, 'temperature', system)}",
        rf"h_1 = h_f(p_1) = {_q(s1.h_J_per_kg, _EH, system)}",
        rf"v_1 = v_f(p_1) = {_q(s1.v_m3_per_kg, 'specific_volume', system)}",
        rf"s_1 = s_f(p_1) = {_q(s1.s_J_per_kg_K, _ES, system)}",
    )
    if losses.subcooling_K == 0.0 and losses.dp_condenser_Pa == 0.0:
        return ProcedureStep(
            title="Estado 1: salida del condensador",
            text=(
                f"Sale líquido saturado (x₁ = 0) a la presión del condensador, "
                f"{_bar(s1.P_Pa)}: {_lookup(result, 'sat_p')}."
            ),
            latex=saturated,
        )
    # Ciclo real (Cengel §10-3): caída de presión y subenfriamiento del condensado.
    lines: list[str] = []
    clauses: list[str] = []
    if losses.dp_condenser_Pa > 0.0:
        p_exh = result.inputs.p_condenser_Pa
        lines.append(
            latex_chain(
                "p_1",
                r"p_{\mathrm{esc}} - \Delta p_{\mathrm{cond}}",
                _diff(p_exh, losses.dp_condenser_Pa, "pressure", system),
                _q(s1.P_Pa, "pressure", system),
            )
        )
        clauses.append(
            f"por la fricción, el condensado sale a p₁ = {_bar(s1.P_Pa)}, "
            f"{_bar(losses.dp_condenser_Pa)} menos que el escape de la turbina"
        )
    if losses.subcooling_K == 0.0:
        return ProcedureStep(
            title="Estado 1: salida del condensador",
            text=(
                f"{_cap(clauses[0])} (Cengel §10-3). Sale líquido saturado (x₁ = 0): "
                f"{_lookup(result, 'sat_p')}."
            ),
            latex=(*lines, *saturated),
        )
    sat = saturation_at_pressure(s1.fluid, s1.P_Pa)
    liquid = saturation_at_temperature(s1.fluid, s1.T_K).liquid
    sub = losses.subcooling_K
    lines += [
        latex_chain(
            "T_1",
            r"T_{\mathrm{sat}}(p_1) - \Delta T_{\mathrm{sub}}",
            rf"{_n(sat.T_sat_K, 'temperature', system)} - "
            rf"{_n(sub, 'temperature_difference', system)}",
            _q(s1.T_K, "temperature", system),
        ),
        rf"h_1 = h(p_1,\ T_1) = {_q(s1.h_J_per_kg, _EH, system)}",
        rf"h_f(T_1) = {_q(liquid.h_J_per_kg, _EH, system)}",
        rf"v_1 \approx v_f(T_1) = {_q(liquid.v_m3_per_kg, 'specific_volume', system)}",
        rf"s_1 = s(p_1,\ T_1) = {_q(s1.s_J_per_kg_K, _ES, system)}",
    ]
    clauses.append(
        f"para que la bomba no cavite, el condensado sale subenfriado {sub:.3g} K por debajo de "
        "la temperatura de saturación"
    )
    text = (
        f"{_cap(' y, '.join(clauses))} (Cengel §10-3). Es líquido comprimido: h, v y s casi no "
        "dependen de la presión y se aproximan con los del líquido saturado a la misma "
        "temperatura (vademecum §13); con la ecuación de estado, h(p₁, T₁) da casi lo mismo."
    )
    return ProcedureStep(title="Estado 1: salida del condensador", text=text, latex=tuple(lines))


def _pump_step(
    result: RankineResult,
    i: int,
    o: int,
    h_out_s: float,
    system: UnitSystem,
    *,
    title: str,
    label: str,
    destination: str,
    pressure: str | None = None,
) -> ProcedureStep:
    """Bomba i → o: w ≈ v·Δp (líquido incompresible), h_s = h(p, s) y el estado real.

    ``pressure`` es el renglón que explica la presión de salida cuando la bomba
    tiene que compensar caídas de presión (ciclo real, Cengel §10-3).
    """
    s_in, s_out = result.states[i], result.states[o]
    a, b = i + 1, o + 1
    eta = result.inputs.eta_pump
    factor = pv_energy_factor(system)
    p_unit = unit_label("pressure", system)
    dp = _diff(s_out.P_Pa, s_in.P_Pa, "pressure", system)
    v_in = _n(s_in.v_m3_per_kg, "specific_volume", system)
    # El factor de unidades (p. ej. 1 bar·m³/kg = 100 kJ/kg) va en su propio renglón
    # para que la ecuación entre en un celular.
    substitution = rf"{v_in}\,({dp})" + (
        "" if math.isclose(factor, 1.0) else rf" \\ &\quad \cdot {latex_number(factor, 5)}"
    )
    w_ps_approx = s_in.v_m3_per_kg * (s_out.P_Pa - s_in.P_Pa)
    w_ps = h_out_s - s_in.h_J_per_kg
    w = s_out.h_J_per_kg - s_in.h_J_per_kg
    lines = [
        *(() if pressure is None else (pressure,)),
        latex_chain(
            r"w_{B,s}",
            rf"v_{_ix(a)}\,(p_{_ix(b)} - p_{_ix(a)})",
            substitution,
            _q(w_ps_approx, _EH, system),
            relation=r"\approx",
        ),
        rf"h_{{{b}s}} = h(p_{_ix(b)},\ s_{_ix(a)}) = {_q(h_out_s, _EH, system)}",
    ]
    if eta == 1.0:
        lines.append(
            latex_chain(
                "w_B",
                rf"h_{_ix(b)} - h_{_ix(a)}",
                _diff(s_out.h_J_per_kg, s_in.h_J_per_kg, _EH, system),
                _q(w, _EH, system),
            )
        )
        pump_txt = f"Bomba ideal: h{_sub(b)} = h{_sub(b)}s."
    else:
        lines.append(
            latex_chain(
                "w_B",
                rf"\frac{{h_{{{b}s}} - h_{_ix(a)}}}{{\eta_B}}",
                rf"\frac{{{_n(w_ps, _EH, system)}}}{{{latex_number(eta, 4)}}}",
                _q(w, _EH, system),
            )
        )
        lines.append(
            latex_chain(
                f"h_{_ix(b)}",
                f"h_{_ix(a)} + w_B",
                rf"{_n(s_in.h_J_per_kg, _EH, system)} + {latex_paren(_n(w, _EH, system))}",
                _q(s_out.h_J_per_kg, _EH, system),
            )
        )
        pump_txt = f"Con η_B = {eta:g}, la bomba real consume más que la ideal (vademecum §10.4)."
    units_note = (
        ""
        if math.isclose(factor, 1.0)
        else f" En este sistema, 1 {p_unit}·{unit_label('specific_volume', system)} = "
        f"{latex_number(factor, 5)} {unit_label(_EH, system)}."
    )
    return ProcedureStep(
        title=title,
        text=(
            f"La {label} lleva el líquido a {destination}. Como el líquido es casi "
            f"incompresible, el trabajo ideal es w = v{_sub(a)}·(p{_sub(b)} − p{_sub(a)}) "
            f"(vademecum §13); con la ecuación de estado, h{_sub(b)}s = "
            f"h(p{_sub(b)}, s{_sub(a)}).{units_note} {pump_txt}"
        ),
        latex=tuple(lines),
    )


def _turbine_step(
    result: RankineResult,
    i: int,
    o: int,
    h_out_s: float,
    system: UnitSystem,
    title: str,
    *,
    work_line: bool = True,
    note: str = "",
) -> ProcedureStep:
    """Expansión i → o: estado isoentrópico (palanca si cae en la campana) y real.

    ``i`` es la entrada de la turbina: con extracciones, todas se miden desde
    ahí (η_T de la turbina entera, como en las soluciones de Cengel).
    """
    s_in = result.states[i]
    s_out = result.states[o]
    a, b = i + 1, o + 1  # numeración de Cengel
    eta = result.inputs.eta_turbine
    fluid = s_in.fluid
    sat = (
        saturation_at_pressure(fluid, s_out.P_Pa)
        if s_out.P_Pa < fluid_limits(fluid).P_crit_Pa
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
            "con s_fg = s_g − s_f y h_fg = h_g − h_f de "
            f"{_cite(_read(result, 'sat'), 'vademecum §12')}."
        )
    else:
        lines.append(
            latex_chain(rf"h_{{{b}s}}", rf"h(p_{{{b}}},\ s_{{{b}s}})", _q(h_out_s, _EH, system))
        )
        how = (
            f"A p = {_bar(s_out.P_Pa)} el estado isoentrópico {b}s sigue sobrecalentado: "
            f"{_lookup(result, 'superheated', interpolate=True)}."
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
    if work_line:
        lines.append(
            latex_chain(
                rf"w_{{T,{a}{b}}}",
                rf"h_{{{a}}} - h_{{{b}}}",
                _diff(s_in.h_J_per_kg, s_out.h_J_per_kg, _EH, system),
                _q(s_in.h_J_per_kg - s_out.h_J_per_kg, _EH, system),
            )
        )
    text = f"{how} {real}" + (f" {note}" if note else "")
    return ProcedureStep(title=title, text=text, latex=tuple(lines))


def _power_step(result: RankineResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    m_line = (
        latex_chain(
            r"\dot{m}",
            r"\frac{\dot{W}_{\mathrm{neto}}}{w_{\mathrm{neto}}}",
            _q(result.m_dot_kg_s, "mass_flow", system),
        )
        if inputs.W_net_W is not None
        else rf"\dot{{m}} = {_q(result.m_dot_kg_s, 'mass_flow', system)}"
    )
    lines = [
        m_line,
        latex_chain(
            r"\dot{W}_{\mathrm{neto}}",
            r"\dot{m}\,w_{\mathrm{neto}}",
            _q(result.W_net_W, "power", system),
        ),
        latex_chain(r"\dot{Q}_H", r"\dot{m}\,q_H", _q(result.Q_in_W, "power", system)),
        latex_chain(r"\dot{Q}_C", r"\dot{m}\,q_C", _q(result.Q_out_W, "power", system)),
    ]
    text = "Cada energía específica por el caudal másico da la potencia correspondiente."
    if result.has_heaters:
        for k, y in enumerate(result.extraction_fractions):
            lines.append(
                latex_chain(
                    rf"\dot{{m}}_{{\mathrm{{ext}},{k + 1}}}",
                    rf"{_y(result, k)}\,\dot{{m}}",
                    _q(y * result.m_dot_kg_s, "mass_flow", system),
                )
            )
        boiler = _of(_boiler(result))
        text += (
            f" Las energías son por kilogramo de vapor que pasa por {_the(_boiler(result))}, así "
            f"que se multiplican por el caudal {boiler}; cada extracción es y·ṁ."
        )
    if result.q_loss_J_per_kg > 0.0:
        lines.append(
            latex_chain(
                r"\dot{Q}_{\text{pérd}}",
                r"\dot{m}\,q_{\text{pérd}}",
                _q(result.Q_loss_W, "power", system),
            )
        )
    return ProcedureStep(title="Caudal y potencias", text=text, latex=tuple(lines))


# ---------------------------------------------------------------------
# Ciclo simple y con recalentamiento (Fase 3.1a)
# ---------------------------------------------------------------------


def _simple_steps(result: RankineResult, system: UnitSystem) -> list[ProcedureStep]:
    """Procedimiento estado por estado, como se resuelve con las tablas (Cengel §10-2)."""
    st = result.states
    s1, s2, s3 = st[0], st[1], st[2]
    inputs = result.inputs
    steps: list[ProcedureStep] = [_condenser_outlet_step(result, system)]

    # 1 → 2 — bomba.
    steps.append(
        _pump_step(
            result,
            0,
            1,
            result.h_pump_out_s_J_per_kg,
            system,
            title="1 → 2: bomba",
            label="bomba",
            destination=f"la presión {_of(_boiler(result))}",
        )
    )

    # 2 → 3 — caldera (o evaporador).
    inlet_txt = (
        "vapor saturado seco (x₃ = 1)"
        if inputs.T_turbine_in_K is None
        else f"vapor a {_degC(s3.T_K)}"
    )
    boiler = _boiler(result)
    steps.append(
        ProcedureStep(
            title=f"2 → 3: {boiler}",
            text=(
                f"{_cap(_the(boiler))} calienta {_fluid(result)} a p constante ({_bar(s3.P_Pa)}) "
                f"hasta {inlet_txt}; h₃ y s₃ {_lookup(result, 'vapor', plural=True)}. Balance de "
                "sistema abierto sin trabajo (vademecum §3.3):"
            ),
            latex=(
                rf"h_3 = {_q(s3.h_J_per_kg, _EH, system)}",
                rf"s_3 = {_q(s3.s_J_per_kg_K, _ES, system)}",
                latex_chain(
                    _q_boiler(result),
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
                    f"El vapor vuelve {_to(boiler)} y se recalienta a p constante "
                    f"({_bar(s4.P_Pa)}) hasta {_degC(s5.T_K)} (Cengel §10-5):"
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
    q_h = rf"{_q_boiler(result)} + q_{{\mathrm{{rec}}}}" if result.has_reheat else _q_boiler(result)
    steps.append(
        _balance_step(
            result,
            system,
            (
                latex_chain(r"q_H", q_h, _q(result.q_in_J_per_kg, _EH, system)),
                latex_chain(r"w_T", w_t, _q(result.w_turbine_J_per_kg, _EH, system)),
            ),
            text=(
                "El trabajo neto es lo que da la turbina menos lo que consume la bomba, y "
                "también el calor recibido menos el cedido (primer principio para el ciclo). "
                "El rendimiento es η = W/Q_H = 1 − Q_C/Q_H (vademecum §9.2)."
            ),
        )
    )
    steps.append(
        _carnot_step(
            result,
            system,
            " + ".join(rf"(s_{{{o + 1}}} - s_{{{i + 1}}})" for i, o in result.heating_pairs),
        )
    )
    steps.append(_power_step(result, system))
    return steps


def _balance_step(
    result: RankineResult, system: UnitSystem, head: Sequence[str], *, text: str
) -> ProcedureStep:
    """w_neto, η y BWR (vademecum §9.2), después de las líneas propias de cada ciclo."""
    return ProcedureStep(
        title="Balance y rendimiento térmico",
        text=text,
        latex=(
            *head,
            latex_chain(
                r"w_{\mathrm{neto}}",
                "w_T - w_B",
                rf"{_n(result.w_turbine_J_per_kg, _EH, system)} - "
                rf"{latex_paren(_n(result.w_pump_J_per_kg, _EH, system))}",
                _q(result.w_net_J_per_kg, _EH, system),
            ),
            *_first_law_lines(result, system),
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


def _first_law_lines(result: RankineResult, system: UnitSystem) -> tuple[str, ...]:
    """Con cañerías, el primer principio incluye el calor que pierden (Cengel §10-3)."""
    if result.q_loss_J_per_kg <= 0.0:
        return ()
    return (
        latex_chain(
            r"w_{\mathrm{neto}}",
            r"q_H - q_C - q_{\text{pérd}}",
            _wrap(
                rf"{_n(result.q_in_J_per_kg, _EH, system)} - "
                rf"{latex_paren(_n(result.q_out_J_per_kg, _EH, system))}",
                "-",
                latex_paren(_n(result.q_loss_J_per_kg, _EH, system)),
            ),
            _q(result.w_net_J_per_kg, _EH, system),
        ),
    )


def _carnot_step(
    result: RankineResult, system: UnitSystem, delta_s: str | Sequence[str]
) -> ProcedureStep:
    T_c = _absolute_T(result.T_low_K, system)
    T_h = _absolute_T(result.T_high_K, system)
    text = (
        "Un ciclo de Carnot entre la temperatura máxima y la de condensación (en "
        "temperaturas absolutas) es el techo del rendimiento (vademecum §9.2). El "
        "Rankine queda abajo porque recibe el calor a una temperatura media T̄_H menor "
        "que la máxima: subir T̄_H (más presión, más sobrecalentamiento, recalentar) es "
        "lo que mejora el ciclo (Cengel §10-4)."
    )
    if result.has_heaters:
        arrives = (
            "el agua llega más caliente a la caldera"
            if result.inputs.fluid == _WATER
            else f"el líquido llega más caliente {_to(_boiler(result))}"
        )
        text += (
            f" La regeneración sube T̄_H porque {arrives} "
            "(Cengel §10-6). Pero η queda por debajo de 1 − T_C/T̄_H: en los calentadores se "
            "mezclan o intercambian calor corrientes a distinta temperatura, y eso genera "
            "entropía."
        )
    elif result.has_recuperator:
        text += (
            " El recuperador sube T̄_H (el líquido llega precalentado al evaporador) sin sacar "
            "vapor de la turbina."
        )
    if not isinstance(delta_s, str):
        terms = list(delta_s)
        rows = [rf"\Delta s_H &= {terms[0]}", *(rf"&\quad + {t}" for t in terms[1:])]
        rows.append(rf"&= {_q(result.q_in_J_per_kg / result.T_mean_in_K, _ES, system)}")
        mean = (
            r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}",
            latex_chain(
                r"\bar{T}_H",
                r"\frac{q_H}{\Delta s_H}",
                _absolute_T(result.T_mean_in_K, system),
            ),
        )
    else:
        mean = (
            latex_chain(
                r"\bar{T}_H",
                rf"\frac{{q_H}}{{{delta_s}}}",
                _absolute_T(result.T_mean_in_K, system),
            ),
        )
    return ProcedureStep(
        title="Comparación con Carnot",
        text=text,
        latex=(
            latex_chain(
                r"\eta_{\mathrm{Carnot}}",
                r"1 - \frac{T_C}{T_H}",
                rf"1 - \frac{{{T_c}}}{{{T_h}}}",
                rf"{latex_number(result.eta_carnot * 100, 4)}\,\%",
            ),
            *mean,
            latex_chain(
                r"1 - \frac{T_C}{\bar{T}_H}",
                rf"{latex_number(result.eta_mean_temperature * 100, 4)}\,\%",
            ),
        ),
    )


# ---------------------------------------------------------------------
# Ciclo regenerativo (Fase 3.1b, Cengel §10-6)
# ---------------------------------------------------------------------


def _y(result: RankineResult, k: int) -> str:
    """Símbolo de la fracción de extracción del calentador k (0 = menor presión)."""
    return "y" if len(result.inputs.heaters) == 1 else f"y_{k + 1}"


def _flow_tex(result: RankineResult, flow: Flow) -> str:
    """``1 - y_1 - y_2``, ``y_2`` o ``1``."""
    parts: list[str] = []
    if None in flow:
        parts.append("1" if flow[None] == 1 else f"{flow[None]}")
    for key in sorted(k for k in flow if k is not None):
        coef = flow[key]
        sign = "-" if coef < 0 else "+"
        term = _y(result, key) if abs(coef) == 1 else f"{abs(coef)}\\,{_y(result, key)}"
        parts.append(f"{sign} {term}" if parts else (f"-{term}" if coef < 0 else term))
    return " ".join(parts) if parts else "0"


def _y_text(result: RankineResult, k: int) -> str:
    """La fracción para el texto (no LaTeX): ``y`` o ``y₂``."""
    return "y" if len(result.inputs.heaters) == 1 else f"y{_sub(k + 1)}"


def _flow_text(result: RankineResult, flow: Flow) -> str:
    """El caudal para el texto: ``1 − y₁ − y₂``."""
    tex = _flow_tex(result, flow)
    for k in range(len(result.inputs.heaters)):
        tex = tex.replace(_y(result, k), _y_text(result, k))
    return tex.replace(" - ", " − ").replace("-", "−")


def _flow_value(result: RankineResult, flow: Flow) -> float:
    ys = result.extraction_fractions
    return sum(coef * (1.0 if key is None else ys[key]) for key, coef in flow.items())


def _times(result: RankineResult, flow: Flow, expr: str) -> str:
    """``(1 - y)\\,(h_6 - h_7)``; con caudal 1, solo ``(h_6 - h_7)``."""
    if flow == {None: 1}:
        return expr
    tex = _flow_tex(result, flow)
    coef = tex if len(flow) == 1 and None not in flow else f"({tex})"
    return rf"{coef}\,{expr}"


def _times_n(result: RankineResult, flow: Flow, expr: str) -> str:
    """Igual que :func:`_times`, con el caudal ya en número (``·`` si ``expr`` es un número)."""
    if flow == {None: 1}:
        return expr
    op = r"\," if expr.startswith("(") else r"\cdot "
    return rf"{latex_number(_flow_value(result, flow), 5)}{op}{expr}"


def _rows(lhs: str, terms: Sequence[str], value: str) -> str:
    """``lhs = t₁ + t₂ …`` con un término (ya con números) por renglón y el valor."""
    rows = [rf"{lhs} &= {terms[0]}"]
    rows += [rf"&\quad + {t}" for t in terms[1:]]
    rows.append(rf"&= {value}")
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def _sum_lines(lhs: str, terms: Sequence[str], numbers: Sequence[str], value: str) -> str:
    """``lhs = t₁ + t₂ …`` con un término por renglón, después los números y el valor."""
    rows = [rf"{lhs} &= {terms[0]}"]
    rows += [rf"&\quad + {t}" for t in terms[1:]]
    rows.append(rf"&= {numbers[0]}")
    rows += [rf"&\quad + {t}" for t in numbers[1:]]
    rows.append(rf"&= {value}")
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def _state_lines(result: RankineResult, i: int, system: UnitSystem) -> tuple[str, ...]:
    s = result.states[i]
    k = _ix(i + 1)
    return (
        rf"h_{k} = {_q(s.h_J_per_kg, _EH, system)}",
        rf"s_{k} = {_q(s.s_J_per_kg_K, _ES, system)}",
    )


def _line_path(result: RankineResult, state: int) -> tuple[CycleComponent, list[tuple[str, float]]]:
    """Sigue la línea de agua desde ``state`` hasta el próximo abierto o la caldera.

    Devuelve ese componente y las caídas de presión del camino (nombre, Δp):
    los cerrados por los que pasa el agua, la cañería de alimentación, la
    caldera y la cañería de vapor (Cengel §10-3).
    """
    inputs = result.inputs
    line_kinds = ("open_heater", "closed_heater", "mixer", "recuperator", "pipe", "boiler")
    consumers = {
        p.state: c
        for c in result.components
        if c.kind in line_kinds
        for p in c.ports
        if p.role in ("in", "fw_in")
    }
    drops: list[tuple[str, float]] = []
    while True:
        comp = consumers[state]
        if comp.kind == "open_heater":
            return comp, drops
        if comp.kind == "boiler":
            drops.append((_the(comp.label), inputs.losses.dp_boiler_Pa))
            if inputs.losses.steam_pipe is not None:
                drops.append(("la cañería de vapor", inputs.losses.steam_pipe.dp_Pa))
            return comp, drops
        if comp.kind == "closed_heater":
            assert comp.heater is not None
            drops.append((f"el {comp.label}", inputs.heaters[comp.heater].dp_Pa))
            state = comp.port("fw_out").state
        elif comp.kind == "pipe":
            pipe = inputs.losses.feed_pipe
            drops.append(("la cañería de alimentación", 0.0 if pipe is None else pipe.dp_Pa))
            state = comp.port("out").state
        elif comp.kind == "recuperator":
            state = comp.port("fw_out").state
        else:  # cámara de mezcla
            state = comp.port("out").state


def _destination(
    result: RankineResult, comp: CycleComponent, system: UnitSystem
) -> tuple[str, str | None]:
    """A qué presión lleva el líquido cada bomba (para el texto) y, si tiene que
    compensar caídas de presión, el renglón p_sal = p_destino + ΣΔp."""
    out_i = comp.port("out").state
    out = result.states[out_i]
    if comp.heater is not None:
        return f"la presión de la línea de agua de alimentación ({_bar(out.P_Pa)})", None
    target, path = _line_path(result, out_i)
    drops = [(name, dp) for name, dp in path if dp > 0.0]
    assert result.layout is not None
    if target.kind == "open_heater":
        ref = target.port("out").state
        base = f"la presión del {target.label}"
    else:
        ref = result.layout.turbine_inlet
        base = "la presión de entrada a la turbina" if drops else f"la presión {_of(target.label)}"
    if not drops:
        return f"{base} ({_bar(out.P_Pa)})", None
    names = [name for name, _ in drops]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " y " + names[-1]
    total = sum(dp for _, dp in drops)
    line = latex_chain(
        f"p_{_ix(out_i + 1)}",
        rf"p_{_ix(ref + 1)} + \sum \Delta p",
        rf"{_n(result.states[ref].P_Pa, 'pressure', system)} + {_n(total, 'pressure', system)}",
        _q(out.P_Pa, "pressure", system),
    )
    return f"{base} más las caídas de presión en {listed} ({_bar(out.P_Pa)})", line


def _open_heater_state(
    result: RankineResult, comp: CycleComponent, system: UnitSystem
) -> ProcedureStep:
    i = comp.port("out").state
    s = result.states[i]
    k = _ix(i + 1)
    return ProcedureStep(
        title=f"Estado {i + 1}: salida del {comp.label}",
        text=(
            f"Del calentador abierto sale líquido saturado a la presión de extracción, "
            f"{_bar(s.P_Pa)} (Cengel §10-6): {_lookup(result, 'sat')}."
        ),
        latex=(
            rf"T_{k} = T_{{\mathrm{{sat}}}}(p_{k}) = {_q(s.T_K, 'temperature', system)}",
            rf"h_{k} = h_f(p_{k}) = {_q(s.h_J_per_kg, _EH, system)}",
            rf"v_{k} = v_f(p_{k}) = {_q(s.v_m3_per_kg, 'specific_volume', system)}",
            rf"s_{k} = s_f(p_{k}) = {_q(s.s_J_per_kg_K, _ES, system)}",
        ),
    )


def _closed_heater_states(
    result: RankineResult, comp: CycleComponent, system: UnitSystem
) -> ProcedureStep:
    """Salida del agua (líquido comprimido a T_sat − TTD) y drenaje (líquido saturado,
    o subenfriado a T_entrada + DCA si tiene subenfriador)."""
    assert comp.heater is not None
    heater = result.inputs.heaters[comp.heater]
    ttd = heater.ttd_K
    in_i, out_i = comp.port("fw_in").state, comp.port("fw_out").state
    drain_i, bleed_i = comp.port("drain_out").state, comp.port("bleed").state
    water_in, out, drain = result.states[in_i], result.states[out_i], result.states[drain_i]
    i_, o, d, b = _ix(in_i + 1), _ix(out_i + 1), _ix(drain_i + 1), _ix(bleed_i + 1)
    h_f_T = saturation_at_temperature(out.fluid, out.T_K).liquid.h_J_per_kg
    ttd_tex = "" if ttd == 0.0 else r" - \mathrm{TTD}"
    lines: list[str] = []
    if heater.dp_Pa > 0.0:
        lines.append(
            latex_chain(
                f"p_{o}",
                rf"p_{i_} - \Delta p",
                _diff(water_in.P_Pa, heater.dp_Pa, "pressure", system),
                _q(out.P_Pa, "pressure", system),
            )
        )
    lines += [
        latex_chain(
            f"T_{o}",
            rf"T_{{\mathrm{{sat}}}}(p_{b}){ttd_tex}",
            _q(out.T_K, "temperature", system),
        ),
        rf"h_{o} = h(p_{o},\ T_{o}) = {_q(out.h_J_per_kg, _EH, system)}",
    ]
    lines.append(rf"h_f(T_{o}) = {_q(h_f_T, _EH, system)}")
    if heater.dca_K is None:
        lines.append(rf"h_{d} = h_f(p_{b}) = {_q(drain.h_J_per_kg, _EH, system)}")
    else:
        lines.append(
            latex_chain(
                f"T_{d}",
                rf"T_{i_} + \mathrm{{DCA}}",
                rf"{_n(water_in.T_K, 'temperature', system)} + "
                rf"{_n(heater.dca_K, 'temperature_difference', system)}",
                _q(drain.T_K, "temperature", system),
            )
        )
        lines.append(rf"h_{d} = h(p_{d},\ T_{d}) = {_q(drain.h_J_per_kg, _EH, system)}")
    for valve in result.components:
        if valve.kind == "valve" and valve.heater == comp.heater:
            v_i = valve.port("out").state
            after = result.states[v_i]
            lines.append(rf"h_{_ix(v_i + 1)} = h_{d} = {_q(after.h_J_per_kg, _EH, system)}")
            drain_txt = (
                f" El drenaje pasa por una válvula (trampa) a {_bar(after.P_Pa)}: se estrangula "
                f"a h constante, h{_sub(v_i + 1)} = h{_sub(drain_i + 1)}."
            )
            break
    else:
        drain_txt = " El drenaje se bombea hacia adelante, a la línea de agua de alimentación."
    water = result.inputs.fluid == _WATER
    approx = (
        f" Como {'el agua' if water else 'el líquido'} está a más presión, es líquido "
        "comprimido: h sale de "
        + ("la tabla de líquido comprimido (Cengel A-7)" if water else "la ecuación de estado")
        + " y, aproximando, h ≈ h_f(T) (vademecum §13)."
    )
    ttd_txt = (
        f"a la temperatura de saturación de la extracción (T_sat a {_bar(drain.P_Pa)})"
        if ttd == 0.0
        else f"a T_sat − TTD, con TTD = {ttd:g} K".replace("-", "−")
    )
    if heater.dca_K is None:
        condenses = (
            f"y la extracción condensa y sale como líquido saturado a {_bar(drain.P_Pa)} (el "
            "drenaje; Cengel §10-6)."
        )
    else:
        condenses = (
            "y la extracción condensa; con subenfriador de drenaje, el condensado se sigue "
            "enfriando contra el agua que entra y sale a la temperatura de esa agua más el DCA "
            f"(DCA = {heater.dca_K:g} K), como líquido comprimido a {_bar(drain.P_Pa)}."
        )
    extra = ""
    if heater.dp_Pa > 0.0:
        extra += f" En los tubos el agua pierde {_bar(heater.dp_Pa)} por fricción (Cengel §10-3)."
    if heater.desuperheater:
        extra += (
            " Tiene desrecalentador: la extracción, sobrecalentada, se enfría primero hasta "
            "vapor saturado calentando el agua que ya salió de la zona de condensación; por "
            "eso el agua puede salir por encima de T_sat (TTD < 0)."
        )
    return ProcedureStep(
        title=f"{comp.label[0].upper()}{comp.label[1:]}: estados {out_i + 1} y {drain_i + 1}",
        text=(
            f"En el calentador cerrado el agua de alimentación sale {ttd_txt}, {condenses}"
            f"{extra}{approx}{drain_txt}"
        ),
        latex=tuple(lines),
    )


def _pipe_step(result: RankineResult, comp: CycleComponent, system: UnitSystem) -> ProcedureStep:
    """Cañería con pérdida de carga y de calor (Cengel §10-3 y ejemplo 10-2)."""
    assert result.layout is not None
    i, o = comp.port("in").state, comp.port("out").state
    s_in, s_out = result.states[i], result.states[o]
    a, c = _ix(i + 1), _ix(o + 1)
    feed = o == result.layout.boiler_in
    pipe = result.inputs.losses.feed_pipe if feed else result.inputs.losses.steam_pipe
    assert pipe is not None
    p_kind: QuantityKind = "pressure"
    dT: QuantityKind = "temperature_difference"
    lines: list[str] = []
    if feed:  # se conoce la entrada (sale de la bomba o del último calentador)
        if pipe.dp_Pa > 0.0:
            lines.append(
                latex_chain(
                    f"p_{c}",
                    rf"p_{a} - \Delta p",
                    _diff(s_in.P_Pa, pipe.dp_Pa, p_kind, system),
                    _q(s_out.P_Pa, p_kind, system),
                )
            )
        if pipe.dT_K > 0.0:
            lines.append(
                latex_chain(
                    f"T_{c}",
                    rf"T_{a} - \Delta T",
                    rf"{_n(s_in.T_K, 'temperature', system)} - {_n(pipe.dT_K, dT, system)}",
                    _q(s_out.T_K, "temperature", system),
                )
            )
            lines.append(rf"h_{c} = h(p_{c},\ T_{c}) = {_q(s_out.h_J_per_kg, _EH, system)}")
        else:
            lines.append(rf"h_{c} = h_{a} = {_q(s_out.h_J_per_kg, _EH, system)}")
        lines.append(rf"s_{c} = {_q(s_out.s_J_per_kg_K, _ES, system)}")
        known, unknown = a, c
        what = "el líquido"
        where = f"hasta {_the(_boiler(result))}"
    else:  # se conoce la salida (la entrada a la turbina)
        if pipe.dp_Pa > 0.0:
            lines.append(
                latex_chain(
                    f"p_{a}",
                    rf"p_{c} + \Delta p",
                    rf"{_n(s_out.P_Pa, p_kind, system)} + {_n(pipe.dp_Pa, p_kind, system)}",
                    _q(s_in.P_Pa, p_kind, system),
                )
            )
        if pipe.dT_K > 0.0:
            lines.append(
                latex_chain(
                    f"T_{a}",
                    rf"T_{c} + \Delta T",
                    rf"{_n(s_out.T_K, 'temperature', system)} + {_n(pipe.dT_K, dT, system)}",
                    _q(s_in.T_K, "temperature", system),
                )
            )
            lines.append(rf"h_{a} = h(p_{a},\ T_{a}) = {_q(s_in.h_J_per_kg, _EH, system)}")
        else:
            lines.append(rf"h_{a} = h_{c} = {_q(s_in.h_J_per_kg, _EH, system)}")
        lines.append(rf"s_{a} = {_q(s_in.s_J_per_kg_K, _ES, system)}")
        known, unknown = c, a
        what = "el vapor"
        where = f"desde {_the(_boiler(result))} hasta la turbina"
    del known, unknown
    q_loss = s_in.h_J_per_kg - s_out.h_J_per_kg
    lines.append(
        latex_chain(
            r"q_{\text{pérd}}",
            rf"h_{a} - h_{c}",
            _diff(s_in.h_J_per_kg, s_out.h_J_per_kg, _EH, system),
            _q(q_loss, _EH, system),
        )
    )
    parts = []
    if pipe.dp_Pa > 0.0:
        parts.append(f"pierde {_bar(pipe.dp_Pa)} de presión por la fricción")
    if pipe.dT_K > 0.0:
        parts.append(
            f"se enfría {pipe.dT_K:.3g} K porque pierde calor hacia el ambiente, ya que la "
            "aislación no es perfecta"
        )
    else:
        parts.append("no pierde calor (cañería aislada: h constante)")
    told = " y ".join(parts)
    if not feed:
        told += (
            f". Por eso {_the(_boiler(result))} tiene que entregar el vapor más caliente y a más "
            "presión que lo que pide la turbina"
        )
    return ProcedureStep(
        title=f"{i + 1} → {o + 1}: {comp.label}",
        text=f"En la cañería {where}, {what} {told} (Cengel §10-3):",
        latex=tuple(lines),
    )


def _recuperator_step(
    result: RankineResult, ci: int, flows: dict[tuple[int, int], Flow], system: UnitSystem
) -> ProcedureStep:
    """Recuperador del ORC: efectividad ε = q/q_máx (Cengel §9-9) y los dos estados."""
    comp = result.components[ci]
    assert result.inputs.recuperator is not None
    eps = result.inputs.recuperator.effectiveness
    cold_in, cold_out = comp.port("fw_in").state, comp.port("fw_out").state
    hot_in, hot_out = comp.port("in").state, comp.port("out").state
    st = result.states
    fluid = result.inputs.fluid
    h_hot_min = fluid_state_from_pair(fluid, "TP", t=st[cold_in].T_K, p=st[hot_out].P_Pa).h_J_per_kg
    h_cold_max = fluid_state_from_pair(
        fluid, "TP", t=st[hot_in].T_K, p=st[cold_out].P_Pa
    ).h_J_per_kg
    q_hot = st[hot_in].h_J_per_kg - h_hot_min
    q_cold = h_cold_max - st[cold_in].h_J_per_kg
    q = st[cold_out].h_J_per_kg - st[cold_in].h_J_per_kg
    ci_, co, hi, ho = (_ix(s + 1) for s in (cold_in, cold_out, hot_in, hot_out))
    hot_limits = q_hot <= q_cold
    lines = [
        latex_chain(
            r"q_{\text{máx},c}",
            rf"h_{hi} - h(p_{ho},\ T_{ci_})",
            _q(q_hot, _EH, system),
        ),
        latex_chain(
            r"q_{\text{máx},f}",
            rf"h(p_{co},\ T_{hi}) - h_{ci_}",
            _q(q_cold, _EH, system),
        ),
        latex_chain(
            r"q",
            r"\varepsilon\,q_{\text{máx}}",
            rf"{latex_number(eps, 4)}\cdot {_n(min(q_hot, q_cold), _EH, system)}",
            _q(q, _EH, system),
        ),
        latex_chain(
            f"h_{co}",
            f"h_{ci_} + q",
            rf"{_n(st[cold_in].h_J_per_kg, _EH, system)} + {_n(q, _EH, system)}",
            _q(st[cold_out].h_J_per_kg, _EH, system),
        ),
        latex_chain(
            f"h_{ho}",
            f"h_{hi} - q",
            _diff(st[hot_in].h_J_per_kg, q, _EH, system),
            _q(st[hot_out].h_J_per_kg, _EH, system),
        ),
        rf"T_{co} = {_q(st[cold_out].T_K, 'temperature', system)}",
        rf"s_{co} = {_q(st[cold_out].s_J_per_kg_K, _ES, system)}",
        rf"T_{ho} = {_q(st[hot_out].T_K, 'temperature', system)}",
    ]
    del flows
    side = "el vapor" if hot_limits else "el líquido"
    return ProcedureStep(
        title=f"Recuperador: estados {cold_out + 1} y {hot_out + 1}",
        text=(
            "El escape de la turbina sale sobrecalentado y, en el recuperador, le pasa calor al "
            f"líquido que va {_to(_boiler(result))}. Lo más que podría pasar, q_máx, es lo que "
            "daría el lado que menos puede entregar si llegara a la temperatura de entrada del "
            f"otro: acá limita {side}. Con la efectividad ε = q/q_máx sale el calor q, y con el "
            "balance (sin calor ni trabajo, el mismo caudal de los dos lados) los dos estados "
            "(Cengel §9-9 usa la misma idea para el regenerador del ciclo Brayton):"
        ),
        latex=tuple(lines),
    )


def _equation(lhs: str, rhs: str, *, split: bool) -> str:
    """``lhs = rhs``; con ``split``, el lado derecho en otro renglón (celular)."""
    if not split:
        return f"{lhs} = {rhs}"
    return rf"\begin{{aligned}}&{lhs} \\ &\quad = {rhs}\end{{aligned}}"


@dataclass
class _HeaterParts:
    """Piezas del balance de un calentador: ecuación, Δh y caudales (Cengel §10-6)."""

    comp: CycleComponent
    y: str
    lhs: str
    rhs: str
    split: bool
    text: str
    water: tuple[float, str, int, int]
    ext: tuple[float, str, int, int]
    dren: tuple[float, str, int, int] | None
    F: Flow
    D: Flow
    forward: bool


def _heater_parts(
    result: RankineResult, ci: int, flows: dict[tuple[int, int], Flow]
) -> _HeaterParts:
    comp = result.components[ci]
    assert comp.heater is not None and result.layout is not None
    k = comp.heater
    y = _y(result, k)
    h = [s.h_J_per_kg for s in result.states]
    roles = [p.role for p in comp.ports]

    def flow(role: str) -> Flow:
        return flows[(ci, roles.index(role))]

    def H(i: int) -> str:
        return f"h_{_ix(i + 1)}"

    bleed = comp.port("bleed").state
    drains = comp.ports_with("drain_in")
    drain_in = drains[0].state if drains else None
    D: Flow = flows[(ci, roles.index("drain_in"))] if drains else {}
    if comp.kind == "open_heater":
        fw, out = comp.port("fw_in").state, comp.port("out").state
        out_flow = flow("out")
        lhs = f"{y}\\,{H(bleed)}"
        if drains:
            lhs += f" + {_times(result, D, H(drain_in))}"  # type: ignore[arg-type]
        lhs += f" + {_times(result, flow('fw_in'), H(fw))}"
        rhs = _times(result, out_flow, H(out))
        split = bool(drains) or out_flow != {None: 1}
        water = (h[out] - h[fw], rf"{H(out)} - {H(fw)}", out, fw)
        ext = (h[bleed] - h[fw], rf"{H(bleed)} - {H(fw)}", bleed, fw)
        dren = (h[drain_in] - h[fw], rf"{H(drain_in)} - {H(fw)}", drain_in, fw) if drains else None
        F = out_flow
        text = (
            f"Balance de energía del {comp.label} (vademecum §3.3, sin calor ni trabajo): la "
            "extracción y el agua de alimentación entran, y sale líquido saturado"
            + (" (también entra el drenaje del calentador de mayor presión)" if drains else "")
            + "."
        )
        if out_flow != {None: 1}:
            text += (
                f" Del {comp.label} sale {_flow_text(result, out_flow)} del caudal de la "
                "caldera: lo extraído para los calentadores de mayor presión se suma más "
                "adelante."
            )
        forward = False
    else:
        fw_in, fw_out = comp.port("fw_in").state, comp.port("fw_out").state
        d_out = comp.port("drain_out").state
        F = flow("fw_in")
        forward = k in result.layout.forward_drains
        water = (h[fw_out] - h[fw_in], rf"{H(fw_out)} - {H(fw_in)}", fw_out, fw_in)
        ext = (h[bleed] - h[d_out], rf"{H(bleed)} - {H(d_out)}", bleed, d_out)
        dren = (
            (h[drain_in] - h[d_out], rf"{H(drain_in)} - {H(d_out)}", drain_in, d_out)  # type: ignore[index]
            if drains
            else None
        )
        lhs = f"{y}\\,({ext[1]})"
        if drains:
            lhs += f" + {_times(result, D, '(' + dren[1] + ')')}"  # type: ignore[index]
        rhs = _times(result, F, "(" + water[1] + ")")
        split = bool(drains) or len(F) > 2
        text = (
            f"Balance de energía del {comp.label}: lo que cede la extracción al condensar"
            + (" (y el drenaje que llega del calentador de mayor presión)" if drains else "")
            + " lo recibe el agua de alimentación, que pasa por los tubos"
            + (
                f" (todo el caudal menos {_y_text(result, k)}, que se bombea aparte)."
                if forward and F == {None: 1, k: -1}
                else "."
            )
        )
    return _HeaterParts(comp, y, lhs, rhs, split, text, water, ext, dren, F, D, forward)


def _delta_lines(parts: _HeaterParts, h: list[float], system: UnitSystem) -> list[str]:
    """Δh del agua, de la extracción y del drenaje que llega."""
    lines = []
    for name, item in (("agua", parts.water), ("ext", parts.ext), ("dren", parts.dren)):
        if item is None:
            continue
        lines.append(
            latex_chain(
                rf"\Delta h_{{\mathrm{{{name}}}}}",
                item[1],
                _diff(h[item[2]], h[item[3]], _EH, system),
                _q(item[0], _EH, system),
            )
        )
    return lines


def _desuperheater_lines(
    result: RankineResult, parts: _HeaterParts, system: UnitSystem
) -> tuple[list[str], str]:
    """Con desrecalentador: a qué temperatura sale el agua de la zona de condensación.

    La zona de desrecalentamiento le da al agua y·(h_ext − h_g): el agua entra
    a esa zona a T_z = T(p, h_sal − y·(h_ext − h_g)/F), que tiene que quedar por
    debajo de T_sat (si no, el calentador no podría funcionar).
    """
    comp = parts.comp
    assert comp.heater is not None
    if comp.kind != "closed_heater" or not result.inputs.heaters[comp.heater].desuperheater:
        return [], ""
    st = result.states
    bleed, out = comp.port("bleed").state, comp.port("fw_out").state
    sat = saturation_at_pressure(st[bleed].fluid, st[bleed].P_Pa)
    y_val = result.extraction_fractions[comp.heater]
    f_val = _flow_value(result, parts.F)
    h_z = st[out].h_J_per_kg - y_val * (st[bleed].h_J_per_kg - sat.vapor.h_J_per_kg) / f_val
    T_z = fluid_state_from_pair(st[out].fluid, "PH", p=st[out].P_Pa, h=h_z).T_K
    b, o = _ix(bleed + 1), _ix(out + 1)
    y = parts.y
    num = rf"{y}\,(h_{b} - h_g)"
    sym = rf"h_{o} - {num}" if parts.F == {None: 1} else rf"h_{o} - \frac{{{num}}}{{F}}"
    lines = [
        rf"h_g(p_{b}) = {_q(sat.vapor.h_J_per_kg, _EH, system)}",
        latex_chain("h_z", sym, _q(h_z, _EH, system)),
        latex_chain("T_z", rf"T(p_{o},\ h_z)", _q(T_z, "temperature", system)),
    ]
    text = (
        " En la zona de desrecalentamiento la extracción baja hasta vapor saturado (h_g) y le "
        f"da al agua {_y_text(result, comp.heater)}·(h_ext − h_g): el agua sale de la zona de "
        "condensación a T_z = "
        f"{_degC(T_z)}, por debajo de T_sat = {_degC(sat.T_sat_K)}, y el desrecalentador la "
        f"termina de calentar hasta {_degC(st[out].T_K)}."
    )
    return lines, text


def _heater_balance(
    result: RankineResult, ci: int, flows: dict[tuple[int, int], Flow], system: UnitSystem
) -> ProcedureStep:
    """Balance de energía de un calentador y su fracción de extracción, despejada."""
    parts = _heater_parts(result, ci, flows)
    comp = parts.comp
    assert comp.heater is not None
    k = comp.heater
    y = parts.y
    h = [s.h_J_per_kg for s in result.states]
    water, ext, dren, F, D = parts.water, parts.ext, parts.dren, parts.F, parts.D
    text = parts.text
    if comp.kind == "open_heater":
        text += " De ahí sale la fracción de extracción:"
    lines: list[str] = [_equation(parts.lhs, parts.rhs, split=parts.split)]
    lines += _delta_lines(parts, h, system)
    value = result.extraction_fractions[k]
    n_water, n_ext = _n(water[0], _EH, system), _n(ext[0], _EH, system)
    if parts.forward:
        lines.append(
            latex_chain(
                y,
                r"\frac{\Delta h_{\mathrm{agua}}}"
                r"{\Delta h_{\mathrm{ext}} + \Delta h_{\mathrm{agua}}}",
                rf"\frac{{{n_water}}}{{{n_ext} + {n_water}}}",
                latex_number(value, 5),
            )
        )
    else:
        f_val = _flow_value(result, F)
        num_sym = (
            r"\Delta h_{\mathrm{agua}}"
            if F == {None: 1}
            else rf"({_flow_tex(result, F)})\,\Delta h_{{\mathrm{{agua}}}}"
        )
        num_n = n_water if F == {None: 1} else rf"{latex_number(f_val, 5)}\cdot {n_water}"
        if dren is not None:
            d_val = _flow_value(result, D)
            num_sym += rf" - {_flow_tex(result, D)}\,\Delta h_{{\mathrm{{dren}}}}"
            num_n = _wrap(num_n, "-", rf"{latex_number(d_val, 5)}\cdot {_n(dren[0], _EH, system)}")
            # Numerador en su propio renglón: con el drenaje la fracción no entra en el celular.
            numerator = f_val * water[0] - d_val * dren[0]
            lines.append(latex_chain("N", num_sym, num_n, _q(numerator, _EH, system)))
            lines.append(
                latex_chain(
                    y,
                    r"\frac{N}{\Delta h_{\mathrm{ext}}}",
                    rf"\frac{{{_n(numerator, _EH, system)}}}{{{n_ext}}}",
                    latex_number(value, 5),
                )
            )
        else:
            lines.append(
                latex_chain(
                    y,
                    rf"\frac{{{num_sym}}}{{\Delta h_{{\mathrm{{ext}}}}}}",
                    rf"\frac{{{num_n}}}{{{n_ext}}}",
                    latex_number(value, 5),
                )
            )
    if F != {None: 1} and not parts.forward and comp.kind == "closed_heater":
        text += (
            f" Por los tubos del {comp.label} pasa {_flow_text(result, F)} del caudal de la "
            "caldera: lo que se extrajo para los calentadores de mayor presión no pasa por acá."
        )
    zone_lines, zone_text = _desuperheater_lines(result, parts, system)
    return ProcedureStep(
        title=f"{comp.label[0].upper()}{comp.label[1:]}: fracción de extracción {y}",
        text=text + zone_text,
        latex=(*lines, *zone_lines),
    )


def _coupled(result: RankineResult) -> bool:
    """Un drenaje bombeado que vuelve a la línea antes de otro calentador acopla las fracciones."""
    assert result.layout is not None
    top = len(result.inputs.heaters) - 1
    return any(i != top for i in result.layout.forward_drains)


def _balance_check(
    result: RankineResult, ci: int, flows: dict[tuple[int, int], Flow], system: UnitSystem
) -> ProcedureStep:
    """Verificación de un balance con las fracciones ya resueltas (sistema acoplado)."""
    parts = _heater_parts(result, ci, flows)
    comp = parts.comp
    h = [s.h_J_per_kg for s in result.states]
    lines = _delta_lines(parts, h, system)
    ys = result.extraction_fractions
    assert comp.heater is not None
    y_val = ys[comp.heater]
    if comp.kind == "open_heater":
        roles = [p.role for p in comp.ports]
        fw, out = comp.port("fw_in").state, comp.port("out").state
        f_in = flows[(ci, roles.index("fw_in"))]
        f_out = flows[(ci, roles.index("out"))]
        bleed = comp.port("bleed").state
        drains = comp.ports_with("drain_in")
        e_in = y_val * h[bleed] + _flow_value(result, f_in) * h[fw]
        in_terms = [rf"{latex_number(y_val, 5)}\cdot {_n(h[bleed], _EH, system)}"]
        if drains:
            d = drains[0].state
            e_in += _flow_value(result, parts.D) * h[d]
            in_terms.append(_times_n(result, parts.D, _n(h[d], _EH, system)))
        in_terms.append(_times_n(result, f_in, _n(h[fw], _EH, system)))
        e_out = _flow_value(result, f_out) * h[out]
        lines += [
            _rows(r"E_{\mathrm{entra}}", in_terms, _q(e_in, _EH, system)),
            latex_chain(
                r"E_{\mathrm{sale}}",
                _times(result, f_out, f"h_{_ix(out + 1)}"),
                *(
                    ()
                    if f_out == {None: 1}
                    else (_times_n(result, f_out, _n(h[out], _EH, system)),)
                ),
                _q(e_out, _EH, system),
            ),
        ]
        text = (
            f"Con las fracciones resueltas, lo que entra al {comp.label} (extracción, agua y "
            "drenajes) es lo que sale: el balance cierra."
        )
    else:
        gives = y_val * parts.ext[0]
        gives_terms = [rf"{latex_number(y_val, 5)}\cdot {_n(parts.ext[0], _EH, system)}"]
        if parts.dren is not None:
            gives += _flow_value(result, parts.D) * parts.dren[0]
            gives_terms.append(_times_n(result, parts.D, _n(parts.dren[0], _EH, system)))
        takes = _flow_value(result, parts.F) * parts.water[0]
        water_sym = _times(result, parts.F, r"\Delta h_{\mathrm{agua}}")
        lines += [
            _rows(r"q_{\mathrm{cede}}", gives_terms, _q(gives, _EH, system)),
            latex_chain(
                r"q_{\mathrm{recibe}}",
                water_sym,
                *(
                    ()
                    if parts.F == {None: 1}
                    else (_times_n(result, parts.F, _n(parts.water[0], _EH, system)),)
                ),
                _q(takes, _EH, system),
            ),
        ]
        text = (
            f"Con las fracciones resueltas, lo que cede la extracción (y el drenaje que llega) "
            f"en el {comp.label} es lo que recibe el agua de alimentación: el balance cierra."
        )
    zone_lines, zone_text = _desuperheater_lines(result, parts, system)
    return ProcedureStep(
        title=f"{comp.label[0].upper()}{comp.label[1:]}: verificación del balance",
        text=text + zone_text,
        latex=(*lines, *zone_lines),
    )


def _coupled_balance_steps(
    result: RankineResult, flows: dict[tuple[int, int], Flow], system: UnitSystem
) -> list[ProcedureStep]:
    """Fracciones acopladas por un drenaje bombeado intermedio: sistema, solución y verificación."""
    comps = result.components
    heaters = sorted(
        (ci for ci, c in enumerate(comps) if c.kind in ("open_heater", "closed_heater")),
        key=lambda ci: -(comps[ci].heater or 0),
    )
    mixers = [cj for cj, c in enumerate(comps) if c.kind == "mixer"]
    equations = []
    for ci in heaters:
        parts = _heater_parts(result, ci, flows)
        equations.append(_equation(parts.lhs, parts.rhs, split=True))
    for cj in mixers:
        comp = comps[cj]
        roles = [p.role for p in comp.ports]
        fw, dr, out = (comp.port(r).state for r in ("fw_in", "drain_in", "out"))
        f_fw, f_dr = flows[(cj, roles.index("fw_in"))], flows[(cj, roles.index("drain_in"))]
        f_out = flows[(cj, roles.index("out"))]
        equations.append(
            _equation(
                _times(result, f_out, f"h_{_ix(out + 1)}"),
                rf"{_times(result, f_fw, f'h_{_ix(fw + 1)}')} + "
                rf"{_times(result, f_dr, f'h_{_ix(dr + 1)}')}",
                split=True,
            )
        )
    names = [comps[cj].label for cj in mixers]
    unknowns = [_y_text(result, k) for k in range(len(result.inputs.heaters))]
    unknowns += [f"h{_sub(comps[cj].port('out').state + 1)}" for cj in mixers]
    unknowns_txt = ", ".join(unknowns[:-1]) + " y " + unknowns[-1]
    steps = [
        ProcedureStep(
            title="Fracciones de extracción: un sistema acoplado",
            text=(
                "Hay un drenaje bombeado que vuelve a la línea antes de otro calentador: la "
                f"entalpía que sale de la {' y de la '.join(names)} depende de la fracción de "
                "extracción de abajo, y entra al calentador de arriba. Por eso las fracciones no "
                "se despejan de a una: los balances de energía de los calentadores y de la "
                f"mezcla forman un sistema con las incógnitas {unknowns_txt}, que se resuelve "
                "todo junto (TESPy lo hace con el método de Newton):"
            ),
            latex=tuple(equations),
        ),
        ProcedureStep(
            title="Solución del sistema",
            text="Las fracciones de extracción que cumplen todos los balances a la vez:",
            latex=tuple(
                rf"{_y(result, k)} = {latex_number(y, 5)}"
                for k, y in reversed(list(enumerate(result.extraction_fractions)))
            ),
        ),
    ]
    for cj in mixers:
        steps.append(_mixer_step(result, cj, flows, system))
    for ci in heaters:
        steps.append(_balance_check(result, ci, flows, system))
    return steps


def _mixer_step(
    result: RankineResult, ci: int, flows: dict[tuple[int, int], Flow], system: UnitSystem
) -> ProcedureStep:
    comp = result.components[ci]
    roles = [p.role for p in comp.ports]
    fw, dr, out = (comp.port(r).state for r in ("fw_in", "drain_in", "out"))
    f_fw, f_dr = flows[(ci, roles.index("fw_in"))], flows[(ci, roles.index("drain_in"))]
    h = [s.h_J_per_kg for s in result.states]
    o = _ix(out + 1)
    return ProcedureStep(
        title=f"Estado {out + 1}: cámara de mezcla",
        text=(
            "El drenaje bombeado se mezcla con el agua de alimentación que sale del calentador "
            "cerrado (balance de energía de la cámara, vademecum §3.3):"
        ),
        latex=(
            latex_chain(
                f"h_{o}",
                rf"{_times(result, f_fw, f'h_{_ix(fw + 1)}')} + "
                rf"{_times(result, f_dr, f'h_{_ix(dr + 1)}')}",
                rf"{_times_n(result, f_fw, _n(h[fw], _EH, system))} \\ &\quad + "
                rf"{_times_n(result, f_dr, _n(h[dr], _EH, system))}",
                _q(h[out], _EH, system),
            ),
        ),
    )


def _heat_and_work_steps(
    result: RankineResult, flows: dict[tuple[int, int], Flow], system: UnitSystem
) -> list[ProcedureStep]:
    """q_H, q_C, q_pérd, w_T y w_B por kilogramo de vapor en la caldera (Cengel §10-6)."""
    h = [s.h_J_per_kg for s in result.states]

    def terms(kinds: tuple[str, ...], sign: int) -> tuple[list[str], list[str]]:
        """Un término y·(Δh) por corriente que entra a cada componente de esos tipos."""
        sym: list[str] = []
        num: list[str] = []
        for ci, comp in enumerate(result.components):
            if comp.kind not in kinds:
                continue
            out = comp.port("out").state
            for pi, port in enumerate(comp.ports):
                if port.role not in ("in", "drain_in"):
                    continue
                a, b = (port.state, out) if sign > 0 else (out, port.state)
                flow = flows[(ci, pi)]
                sym.append(_times(result, flow, f"(h_{_ix(b + 1)} - h_{_ix(a + 1)})"))
                num.append(_times_n(result, flow, f"({_diff(h[b], h[a], _EH, system)})"))
        return sym, num

    steps: list[ProcedureStep] = []
    q_sym, q_num = terms(("boiler", "reheater"), +1)
    c_sym, c_num = terms(("condenser",), -1)
    boiler = _boiler(result)
    if result.has_heaters:
        text = (
            f"Por kilogramo de vapor que pasa por {_the(boiler)}: cada corriente pesa según su "
            "fracción del caudal"
            + (
                " (por el recalentador pasa lo que no se extrajo antes)"
                if result.has_reheat
                else ""
            )
            + ". El condensador recibe el vapor que llegó al final de la turbina"
            + (" y los drenajes en cascada." if len(c_sym) > 1 else ".")
        )
    else:
        text = (
            f"El calor entra en {_the(boiler)}"
            + (" y en el recalentador" if result.has_reheat else "")
            + " y sale en el condensador."
        )
        if result.has_recuperator:
            text += (
                " El recuperador no aparece: su calor es interno (pasa del escape de la turbina "
                "al líquido)."
            )
    latex = [
        _sum_lines("q_H", q_sym, q_num, _q(result.q_in_J_per_kg, _EH, system)),
        _sum_lines("q_C", c_sym, c_num, _q(result.q_out_J_per_kg, _EH, system)),
    ]
    if result.of_kind("pipe"):
        l_sym, l_num = terms(("pipe",), -1)
        latex.append(
            _sum_lines(r"q_{\text{pérd}}", l_sym, l_num, _q(result.q_loss_J_per_kg, _EH, system))
        )
        text += (
            " Las cañerías pierden q_pérd hacia el ambiente: ese calor no llega ni a la turbina "
            "ni al condensador (Cengel §10-3)."
        )
    steps.append(ProcedureStep(title="Calor recibido y cedido", text=text, latex=tuple(latex)))
    t_sym, t_num = terms(("turbine",), -1)
    p_sym, p_num = terms(("pump",), +1)
    losses = result.q_loss_J_per_kg > 0.0
    steps.append(
        _balance_step(
            result,
            system,
            (
                _sum_lines("w_T", t_sym, t_num, _q(result.w_turbine_J_per_kg, _EH, system)),
                _sum_lines("w_B", p_sym, p_num, _q(result.w_pump_J_per_kg, _EH, system)),
            ),
            text=(
                "Cada tramo de turbina y cada bomba pesa según el caudal que pasa por él. El "
                "trabajo neto es lo que dan las turbinas menos lo que consumen las bombas, y "
                "también el calor recibido menos el cedido"
                + (" y el perdido en las cañerías" if losses else "")
                + " (primer principio). El rendimiento es η = W/Q_H"
                + ("" if losses else " = 1 − Q_C/Q_H")
                + " (vademecum §9.2)."
            ),
        )
    )
    return steps


def _cycle_steps(result: RankineResult, system: UnitSystem) -> list[ProcedureStep]:
    """Procedimiento general: regeneración (Cengel §10-6), ciclo real (§10-3) y recuperador."""
    layout = result.layout
    assert layout is not None
    flows = port_flows(layout)
    comps = result.components
    st = result.states
    boiler = _boiler(result)
    steps: list[ProcedureStep] = [_condenser_outlet_step(result, system)]

    # 1. Línea de agua de alimentación: los estados que se leen en las tablas.
    pumps = [c for c in comps if c.kind == "pump"]
    pumps_s = dict(zip((id(c) for c in pumps), result.pumps_out_s, strict=True))
    for comp in comps:
        if comp.kind == "pump":
            i, o = comp.port("in").state, comp.port("out").state
            destination, pressure = _destination(result, comp, system)
            steps.append(
                _pump_step(
                    result,
                    i,
                    o,
                    pumps_s[id(comp)].h_J_per_kg,
                    system,
                    title=f"{i + 1} → {o + 1}: {comp.label}",
                    label=comp.label,
                    destination=destination,
                    pressure=pressure,
                )
            )
        elif comp.kind == "open_heater":
            steps.append(_open_heater_state(result, comp, system))
        elif comp.kind == "closed_heater":
            steps.append(_closed_heater_states(result, comp, system))
        elif comp.kind == "pipe" and comp.port("out").state == layout.boiler_in:
            steps.append(_pipe_step(result, comp, system))

    # 2. Camino del vapor: entrada a la turbina, cañería, tramos y recalentador.
    i_in = layout.turbine_inlet
    s_in = st[i_in]
    inlet_txt = (
        "vapor saturado seco"
        if result.inputs.T_turbine_in_K is None
        else f"vapor a {_degC(s_in.T_K)} y {_bar(s_in.P_Pa)}"
    )
    steam_pipes = [c for c in comps if c.kind == "pipe" and c.port("in").state == layout.boiler_out]
    source = "Llega a la turbina" if steam_pipes else f"Sale {_of(boiler)}"
    steps.append(
        ProcedureStep(
            title=f"Estado {i_in + 1}: {layout.labels[i_in]}",
            text=f"{source} {inlet_txt}: h y s de {_read(result, 'vapor')}.",
            latex=_state_lines(result, i_in, system),
        )
    )
    for pipe in steam_pipes:
        steps.append(_pipe_step(result, pipe, system))
    turbines = [c for c in comps if c.kind == "turbine"]
    casing_in: dict[str, int] = {}
    for c in turbines:
        casing_in.setdefault(c.casing, c.port("in").state)
    sections = {c.casing: sum(1 for t in turbines if t.casing == c.casing) for c in turbines}
    note = (
        "η_T se mide desde la entrada de la turbina: cada extracción y la salida caen sobre "
        "la misma línea de expansión."
        if result.inputs.eta_turbine != 1.0 and len(turbines) > len(sections)
        else ""
    )
    for comp in comps:
        if comp.kind == "reheater":
            i, o = comp.port("in").state, comp.port("out").state
            dp = result.inputs.losses.dp_reheater_Pa
            pressure_txt = (
                f"a p constante ({_bar(st[i].P_Pa)})"
                if dp == 0.0
                else f"perdiendo {_bar(dp)} de presión (de {_bar(st[i].P_Pa)} a {_bar(st[o].P_Pa)})"
            )
            steps.append(
                ProcedureStep(
                    title=f"{i + 1} → {o + 1}: recalentador",
                    text=(
                        f"El vapor que sigue vuelve {_to(boiler)} y se recalienta "
                        f"{pressure_txt} hasta {_degC(st[o].T_K)} (Cengel §10-5):"
                    ),
                    latex=_state_lines(result, o, system),
                )
            )
        elif comp.kind == "turbine":
            k = turbines.index(comp)
            o = comp.port("out").state
            a = casing_in[comp.casing]
            if sections[comp.casing] == 1:
                suffix = ""
            elif o == layout.exhaust or o == layout.reheat_in:
                suffix = " (hasta la salida)"
            else:
                suffix = " (hasta la extracción)"
            steps.append(
                _turbine_step(
                    result,
                    a,
                    o,
                    result.h_turbine_out_s_J_per_kg[k],
                    system,
                    f"{a + 1} → {o + 1}: {comp.casing}{suffix}",
                    work_line=False,
                    note=note,
                )
            )
    for ci, comp in enumerate(comps):
        if comp.kind == "recuperator":
            steps.append(_recuperator_step(result, ci, flows, system))

    # 3. Fracciones de extracción, de mayor a menor presión, y la mezcla.
    if result.has_heaters and _coupled(result):
        steps.extend(_coupled_balance_steps(result, flows, system))
    else:
        heaters = sorted(
            (ci for ci, c in enumerate(comps) if c.kind in ("open_heater", "closed_heater")),
            key=lambda ci: -(comps[ci].heater or 0),
        )
        for ci in heaters:
            steps.append(_heater_balance(result, ci, flows, system))
            for cj, other in enumerate(comps):
                if other.kind == "mixer" and other.heater == comps[ci].heater:
                    steps.append(_mixer_step(result, cj, flows, system))

    # 4. Calores, trabajos y rendimiento.
    steps.extend(_heat_and_work_steps(result, flows, system))
    delta_s_terms = []
    for ci, comp in enumerate(comps):
        if comp.kind in ("boiler", "reheater"):
            i, o = comp.port("in").state, comp.port("out").state
            flow = flows[(ci, 0)]
            delta_s_terms.append(_times(result, flow, f"(s_{_ix(o + 1)} - s_{_ix(i + 1)})"))
    steps.append(_carnot_step(result, system, delta_s_terms))
    steps.append(_power_step(result, system))
    return steps


# ---------------------------------------------------------------------
# Agua de enfriamiento y punto de entrada
# ---------------------------------------------------------------------


def cooling_water_step(cooling: CoolingWaterResult, system: UnitSystem) -> ProcedureStep:
    """ṁ del agua de enfriamiento del condensador (balance del condensador, §3.3)."""
    dT = cooling.T_out_K - cooling.T_in_K
    return ProcedureStep(
        title="Agua de enfriamiento del condensador",
        text=(
            "Todo el calor que cede el vapor se lo lleva el agua de enfriamiento, que se calienta "
            f"de {_degC(cooling.T_in_K)} a {_degC(cooling.T_out_K)} (balance del condensador, "
            "vademecum §3.3). Con c_p ≈ 4,18 kJ/(kg·K) el resultado casi no cambia."
        ),
        latex=(
            latex_chain(
                r"\Delta h_{\mathrm{ae}}",
                r"h(T_s) - h(T_e)",
                _q(cooling.dh_J_per_kg, _EH, system),
            ),
            latex_chain(
                r"\dot{m}_{\mathrm{ae}}",
                r"\frac{\dot{Q}_C}{\Delta h_{\mathrm{ae}}}",
                rf"\frac{{{_n(cooling.Q_W, 'power', system)}}}"
                rf"{{{_n(cooling.dh_J_per_kg, _EH, system)}}}",
                _q(cooling.m_dot_kg_s, "mass_flow", system),
            ),
            latex_chain(
                r"\dot{m}_{\mathrm{ae}}",
                r"\frac{\dot{Q}_C}{c_p\,\Delta T}",
                _q(cooling.m_dot_cp_kg_s, "mass_flow", system),
                relation=r"\approx",
            ),
            rf"\Delta T = {_q(dT, 'temperature_difference', system)}",
        ),
    )


def rankine_steps(
    result: RankineResult, system: UnitSystem, cooling: CoolingWaterResult | None = None
) -> list[ProcedureStep]:
    """Procedimiento estado por estado, como se resuelve con las tablas (Cengel cap. 10).

    Sin calentadores, pérdidas ni recuperador, el de la Fase 3.1a (§10-2 a
    §10-4); si no, el general: regenerativo (§10-6), ciclo real (§10-3) y
    recuperador del ORC. Con ``cooling`` se suma el agua de enfriamiento.
    """
    plain = not (result.has_heaters or result.has_losses or result.has_recuperator)
    steps = _simple_steps(result, system) if plain else _cycle_steps(result, system)
    if cooling is not None:
        steps.append(cooling_water_step(cooling, system))
    return steps
