"""Procedimiento del ciclo de Rankine «como con las tablas» — Fases 3.1a y 3.1b.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo (vademecum §3.3, §9.2, §10.4, §12 y §13; Çengel & Boles,
*Termodinámica*, cap. 10). Recibe un resultado ya calculado
(:class:`core.cycles.rankine.RankineResult`): no importa TESPy ni Streamlit.

Sin calentadores sigue el orden de la Fase 3.1a: bomba, caldera, turbinas,
condensador y balance. Con regeneración sigue el orden de Cengel §10-6:
primero los estados que se leen en las tablas, después los balances de los
calentadores —de mayor a menor presión— para despejar las fracciones de
extracción y al final los calores y trabajos por kilogramo de vapor que pasa
por la caldera.

Las ecuaciones se arman con :func:`core.latex.latex_chain` (una igualdad por
renglón) para que entren en el ancho de un celular.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from core.cycles.rankine_layout import Flow, port_flows
from core.fluids import fluid_limits, saturation_at_pressure, saturation_at_temperature
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
# Pasos que comparten el ciclo simple y el regenerativo
# ---------------------------------------------------------------------


def _condenser_outlet_step(result: RankineResult, system: UnitSystem) -> ProcedureStep:
    s1 = result.states[0]
    return ProcedureStep(
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
) -> ProcedureStep:
    """Bomba i → o: w ≈ v·Δp (líquido incompresible), h_s = h(p, s) y el estado real."""
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
        text += (
            " Las energías son por kilogramo de vapor que pasa por la caldera, así que se "
            "multiplican por el caudal de la caldera; cada extracción es y·ṁ."
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
            destination="la presión de la caldera",
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
        text += (
            " La regeneración sube T̄_H porque el agua llega más caliente a la caldera "
            "(Cengel §10-6). Pero η queda por debajo de 1 − T_C/T̄_H: en los calentadores se "
            "mezclan o intercambian calor corrientes a distinta temperatura, y eso genera "
            "entropía."
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


def _destination(result: RankineResult, comp: CycleComponent) -> str:
    """A qué presión lleva el líquido cada bomba (para el texto)."""
    out = result.states[comp.port("out").state]
    if comp.heater is not None:
        return f"la presión de la línea de agua de alimentación ({_bar(out.P_Pa)})"
    for other in result.components:
        if other.kind == "open_heater" and math.isclose(
            result.states[other.port("out").state].P_Pa, out.P_Pa, rel_tol=1e-9
        ):
            return f"la presión del {other.label} ({_bar(out.P_Pa)})"
    return f"la presión de la caldera ({_bar(out.P_Pa)})"


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
            f"{_bar(s.P_Pa)} (Cengel §10-6): se lee en la tabla de saturación."
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
    """Salida del agua (líquido comprimido a T_sat − TTD) y drenaje (líquido saturado)."""
    assert comp.heater is not None
    ttd = result.inputs.heaters[comp.heater].ttd_K
    out_i, drain_i = comp.port("fw_out").state, comp.port("drain_out").state
    bleed_i = comp.port("bleed").state
    out, drain = result.states[out_i], result.states[drain_i]
    o, d, b = _ix(out_i + 1), _ix(drain_i + 1), _ix(bleed_i + 1)
    h_f_T = saturation_at_temperature(out.fluid, out.T_K).liquid.h_J_per_kg
    ttd_tex = "" if ttd == 0.0 else r" - \mathrm{TTD}"
    lines = [
        latex_chain(
            f"T_{o}",
            rf"T_{{\mathrm{{sat}}}}(p_{b}){ttd_tex}",
            _q(out.T_K, "temperature", system),
        ),
        rf"h_{o} = h(p_{o},\ T_{o}) = {_q(out.h_J_per_kg, _EH, system)}",
    ]
    lines.append(rf"h_f(T_{o}) = {_q(h_f_T, _EH, system)}")
    lines.append(rf"h_{d} = h_f(p_{b}) = {_q(drain.h_J_per_kg, _EH, system)}")
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
    approx = (
        " Como el agua está a más presión, es líquido comprimido: h sale de la tabla de "
        "líquido comprimido (Cengel A-7) y, aproximando, h ≈ h_f(T) (vademecum §13)."
    )
    ttd_txt = (
        f"a la temperatura de saturación de la extracción (T_sat a {_bar(drain.P_Pa)})"
        if ttd == 0.0
        else f"a T_sat − TTD, con TTD = {ttd:g} K"
    )
    return ProcedureStep(
        title=f"{comp.label[0].upper()}{comp.label[1:]}: estados {out_i + 1} y {drain_i + 1}",
        text=(
            f"En el calentador cerrado el agua de alimentación sale {ttd_txt}, y la extracción "
            f"condensa y sale como líquido saturado a {_bar(drain.P_Pa)} (el drenaje; Cengel "
            f"§10-6).{approx}{drain_txt}"
        ),
        latex=tuple(lines),
    )


def _equation(lhs: str, rhs: str, *, split: bool) -> str:
    """``lhs = rhs``; con ``split``, el lado derecho en otro renglón (celular)."""
    if not split:
        return f"{lhs} = {rhs}"
    return rf"\begin{{aligned}}&{lhs} \\ &\quad = {rhs}\end{{aligned}}"


def _heater_balance(
    result: RankineResult, ci: int, flows: dict[tuple[int, int], Flow], system: UnitSystem
) -> ProcedureStep:
    """Balance de energía de un calentador y su fracción de extracción, despejada."""
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
    lines: list[str] = []
    if comp.kind == "open_heater":
        fw, out = comp.port("fw_in").state, comp.port("out").state
        out_flow = flow("out")
        lhs = f"{y}\\,{H(bleed)}"
        if drains:
            lhs += f" + {_times(result, D, H(drain_in))}"  # type: ignore[arg-type]
        lhs += f" + {_times(result, flow('fw_in'), H(fw))}"
        lines.append(
            _equation(
                lhs, _times(result, out_flow, H(out)), split=bool(drains) or out_flow != {None: 1}
            )
        )
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
        text += " De ahí sale la fracción de extracción:"
        forward = False
    else:
        fw_in, fw_out = comp.port("fw_in").state, comp.port("fw_out").state
        d_out = comp.port("drain_out").state
        F = flow("fw_in")
        forward = result.layout.drain_forward and k == len(result.inputs.heaters) - 1
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
        lines.append(
            _equation(
                lhs, _times(result, F, "(" + water[1] + ")"), split=bool(drains) or len(F) > 2
            )
        )
        text = (
            f"Balance de energía del {comp.label}: lo que cede la extracción al condensar"
            + (" (y el drenaje que llega del calentador de mayor presión)" if drains else "")
            + " lo recibe el agua de alimentación, que pasa por los tubos"
            + (
                f" (todo el caudal menos {_y_text(result, k)}, que se bombea aparte)."
                if forward
                else "."
            )
        )
    lines.append(
        latex_chain(
            r"\Delta h_{\mathrm{agua}}",
            water[1],
            _diff(h[water[2]], h[water[3]], _EH, system),
            _q(water[0], _EH, system),
        )
    )
    lines.append(
        latex_chain(
            r"\Delta h_{\mathrm{ext}}",
            ext[1],
            _diff(h[ext[2]], h[ext[3]], _EH, system),
            _q(ext[0], _EH, system),
        )
    )
    if dren is not None:
        lines.append(
            latex_chain(
                r"\Delta h_{\mathrm{dren}}",
                dren[1],
                _diff(h[dren[2]], h[dren[3]], _EH, system),  # type: ignore[index]
                _q(dren[0], _EH, system),
            )
        )
    value = result.extraction_fractions[k]
    n_water, n_ext = _n(water[0], _EH, system), _n(ext[0], _EH, system)
    if forward:
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
    if F != {None: 1} and not forward and comp.kind == "closed_heater":
        text += (
            f" Por los tubos del {comp.label} pasa {_flow_text(result, F)} del caudal de la "
            "caldera: lo que se extrajo para los calentadores de mayor presión no pasa por acá."
        )
    return ProcedureStep(
        title=f"{comp.label[0].upper()}{comp.label[1:]}: fracción de extracción {y}",
        text=text,
        latex=tuple(lines),
    )


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
    """q_H, q_C, w_T y w_B por kilogramo de vapor en la caldera (Cengel §10-6)."""
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
    steps.append(
        ProcedureStep(
            title="Calor recibido y cedido",
            text=(
                "Por kilogramo de vapor que pasa por la caldera: cada corriente pesa según su "
                "fracción del caudal"
                + (
                    " (por el recalentador pasa lo que no se extrajo antes)"
                    if result.has_reheat
                    else ""
                )
                + ". El condensador recibe el vapor que llegó al final de la turbina"
                + (" y los drenajes en cascada." if len(c_sym) > 1 else ".")
            ),
            latex=(
                _sum_lines("q_H", q_sym, q_num, _q(result.q_in_J_per_kg, _EH, system)),
                _sum_lines("q_C", c_sym, c_num, _q(result.q_out_J_per_kg, _EH, system)),
            ),
        )
    )
    t_sym, t_num = terms(("turbine",), -1)
    p_sym, p_num = terms(("pump",), +1)
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
                "también el calor recibido menos el cedido (primer principio). El rendimiento "
                "es η = W/Q_H = 1 − Q_C/Q_H (vademecum §9.2)."
            ),
        )
    )
    return steps


def _regenerative_steps(result: RankineResult, system: UnitSystem) -> list[ProcedureStep]:
    """Procedimiento con calentadores de agua de alimentación (Cengel §10-6)."""
    layout = result.layout
    assert layout is not None
    flows = port_flows(layout)
    comps = result.components
    st = result.states
    steps: list[ProcedureStep] = [_condenser_outlet_step(result, system)]

    # 1. Línea de agua de alimentación: los estados que se leen en las tablas.
    pumps = [c for c in comps if c.kind == "pump"]
    pumps_s = dict(zip((id(c) for c in pumps), result.pumps_out_s, strict=True))
    for comp in comps:
        if comp.kind == "pump":
            i, o = comp.port("in").state, comp.port("out").state
            steps.append(
                _pump_step(
                    result,
                    i,
                    o,
                    pumps_s[id(comp)].h_J_per_kg,
                    system,
                    title=f"{i + 1} → {o + 1}: {comp.label}",
                    label=comp.label,
                    destination=_destination(result, comp),
                )
            )
        elif comp.kind == "open_heater":
            steps.append(_open_heater_state(result, comp, system))
        elif comp.kind == "closed_heater":
            steps.append(_closed_heater_states(result, comp, system))

    # 2. Camino del vapor: entrada a la turbina, tramos y recalentador.
    i_in = layout.turbine_inlet
    s_in = st[i_in]
    inlet_txt = (
        "vapor saturado seco"
        if result.inputs.T_turbine_in_K is None
        else f"vapor a {_degC(s_in.T_K)} y {_bar(s_in.P_Pa)}"
    )
    steps.append(
        ProcedureStep(
            title=f"Estado {i_in + 1}: {layout.labels[i_in]}",
            text=f"Sale de la caldera {inlet_txt}: h y s de la tabla de vapor (Cengel A-6).",
            latex=_state_lines(result, i_in, system),
        )
    )
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
            steps.append(
                ProcedureStep(
                    title=f"{i + 1} → {o + 1}: recalentador",
                    text=(
                        f"El vapor que sigue vuelve a la caldera y se recalienta a p constante "
                        f"({_bar(st[i].P_Pa)}) hasta {_degC(st[o].T_K)}:"
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

    # 3. Fracciones de extracción, de mayor a menor presión, y la mezcla.
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

    Sin calentadores, el de la Fase 3.1a (§10-2 a §10-5); con calentadores,
    el regenerativo (§10-6). Con ``cooling`` se suma el agua de enfriamiento.
    """
    steps = (
        _regenerative_steps(result, system) if result.has_heaters else _simple_steps(result, system)
    )
    if cooling is not None:
        steps.append(cooling_water_step(cooling, system))
    return steps
