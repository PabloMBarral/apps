"""Procedimiento del ciclo combinado de varias presiones «como a mano» — Fase 3.6.

Por partes, como el de la Fase 3.4: la turbina de gas
(:func:`~core.cycles.brayton_procedure.brayton_steps`), la caldera de
recuperación (:func:`~core.cycles.hrsg_multi_procedure.multi_hrsg_steps`, con
el recalentador y el sistema de 2×2 de alta y media), el ciclo de vapor y el
acople (rendimiento de Kehlhofer, balance de energía y exergía del ciclo de
fondo).

El ciclo de vapor sigue el orden de Cengel (cap. 10): los estados de la línea
de alimentación (condensador, bombas y desaireador), el vapor que entrega cada
nivel de la HRSG, cada turbina (desde su entrada, con la palanca si la
expansión termina en la campana y, si se pidió, la regla de Baumann en la parte
húmeda), las mezclas de las admisiones y del recalentamiento, el balance del
desaireador, el condensador y las potencias. Los estados se numeran en el
sentido del flujo (:class:`core.cycles.combined_multi.MultiSteamCycle`); los de
la HRSG siguen con su notación (1–5 por nivel). Recibe un resultado ya
calculado: no importa Streamlit.
"""

from __future__ import annotations

import math

from CoolProp.CoolProp import PropsSI

from core.cycles.brayton_procedure import brayton_steps
from core.cycles.combined_multi import (
    MultiCombinedResult,
    TurbineSection,
    bottoming_exergy,
)
from core.cycles.combined_procedure import _balance_step, _efficiency_step
from core.cycles.hrsg_multi_procedure import multi_hrsg_steps
from core.cycles.rankine_procedure import _bar, _diff, _factor_diff, _ix, _n, _q, _sub, _wrap
from core.fluids import fluid_limits, fluid_state_from_pair, saturation_at_pressure
from core.latex import latex_chain, latex_number, latex_paren
from core.state_report import ProcedureStep, pv_energy_factor
from core.units_system import QuantityKind, UnitSystem, unit_label

__all__ = ["combined_multi_sections", "steam_cycle_steps"]

_EH: QuantityKind = "specific_enthalpy"
_ES: QuantityKind = "specific_entropy"
_P: QuantityKind = "power"
_M: QuantityKind = "mass_flow"
_LETTERS = {"alta": "A", "media": "M", "baja": "B", "": ""}


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _hs(k: int) -> str:
    """``h_{4s}``: el estado isoentrópico k."""
    return f"h_{{{k}s}}"


def _m_level(result: MultiCombinedResult, level: int) -> str:
    letter = _LETTERS[result.inputs.names[level]]
    return rf"\dot{{m}}_{{{letter}}}" if letter else r"\dot{m}_v"


# ---------------------------------------------------------------------
# Línea de alimentación
# ---------------------------------------------------------------------


def _condenser_outlet_step(result: MultiCombinedResult, system: UnitSystem) -> ProcedureStep:
    s1 = result.steam.states[0]
    return ProcedureStep(
        title="Estado 1: salida del condensador",
        text=(
            f"Sale líquido saturado (x₁ = 0) a la presión del condensador, {_bar(s1.P_Pa)}: se "
            "lee en la tabla de saturación por presión (Cengel A-5)."
        ),
        latex=(
            rf"T_1 = T_{{\mathrm{{sat}}}}(p_1) = {_q(s1.T_K, 'temperature', system)}",
            rf"h_1 = h_f(p_1) = {_q(s1.h_J_per_kg, _EH, system)}",
            rf"v_1 = v_f(p_1) = {_q(s1.v_m3_per_kg, 'specific_volume', system)}",
            rf"s_1 = s_f(p_1) = {_q(s1.s_J_per_kg_K, _ES, system)}",
        ),
    )


def _pump_step(
    result: MultiCombinedResult,
    i: int,
    o: int,
    system: UnitSystem,
    *,
    title: str,
    what: str,
) -> ProcedureStep:
    """Bomba i → o: w ≈ v·Δp, h_s = h(p, s) y el estado real (vademecum §13)."""
    st = result.steam.states
    s_in, s_out = st[i], st[o]
    a, b = i + 1, o + 1
    eta = result.inputs.steam.eta_pump
    factor = pv_energy_factor(system)
    v_in = _n(s_in.v_m3_per_kg, "specific_volume", system)
    substitution = _factor_diff(v_in, s_out.P_Pa, s_in.P_Pa, "pressure", system) + (
        "" if math.isclose(factor, 1.0) else rf" \\ &\quad \cdot {latex_number(factor, 5)}"
    )
    h_s = fluid_state_from_pair("Water", "PS", p=s_out.P_Pa, s=s_in.s_J_per_kg_K).h_J_per_kg
    w = s_out.h_J_per_kg - s_in.h_J_per_kg
    lines = [
        latex_chain(
            r"w_{B,s}",
            rf"v_{_ix(a)}\,(p_{_ix(b)} - p_{_ix(a)})",
            substitution,
            _q(s_in.v_m3_per_kg * (s_out.P_Pa - s_in.P_Pa), _EH, system),
            relation=r"\approx",
        ),
        rf"{_hs(b)} = h(p_{_ix(b)},\ s_{_ix(a)}) = {_q(h_s, _EH, system)}",
    ]
    if eta == 1.0:
        lines.append(rf"h_{_ix(b)} = {_hs(b)} = {_q(s_out.h_J_per_kg, _EH, system)}")
        real = f"Bomba ideal: h{_sub(b)} = h{_sub(b)}s."
    else:
        lines += [
            latex_chain(
                "w_B",
                rf"\frac{{{_hs(b)} - h_{_ix(a)}}}{{\eta_B}}",
                rf"\frac{{{_n(h_s - s_in.h_J_per_kg, _EH, system)}}}{{{latex_number(eta, 4)}}}",
                _q(w, _EH, system),
            ),
            latex_chain(
                f"h_{_ix(b)}",
                f"h_{_ix(a)} + w_B",
                rf"{_n(s_in.h_J_per_kg, _EH, system)} + {latex_paren(_n(w, _EH, system))}",
                _q(s_out.h_J_per_kg, _EH, system),
            ),
        ]
        real = f"Con η_B = {eta:g}, la bomba real consume más que la ideal (vademecum §10.4)."
    units = (
        ""
        if math.isclose(factor, 1.0)
        else f" En este sistema, 1 {unit_label('pressure', system)}·"
        f"{unit_label('specific_volume', system)} = {latex_number(factor, 5)} "
        f"{unit_label(_EH, system)}."
    )
    return ProcedureStep(
        title=title,
        text=(
            f"La bomba lleva el líquido {what} ({_bar(s_out.P_Pa)}). Como el líquido es casi "
            f"incompresible, el trabajo ideal es w = v{_sub(a)}·(p{_sub(b)} − p{_sub(a)}); con "
            f"la ecuación de estado, h{_sub(b)}s = h(p{_sub(b)}, s{_sub(a)}).{units} {real}"
        ),
        latex=tuple(lines),
    )


def _feed_line_steps(result: MultiCombinedResult, system: UnitSystem) -> list[ProcedureStep]:
    cyc = result.steam
    n = len(result.inputs.levels)
    target = "al economizador de baja" if n > 1 else "a la caldera de recuperación"
    steps = [_condenser_outlet_step(result, system)]
    if cyc.deaerator is None:
        o = cyc.feedwater
        steps.append(
            _pump_step(
                result,
                0,
                o,
                system,
                title=f"Bomba de condensado (1 → {o + 1})",
                what=f"del condensador {target}",
            )
        )
        return steps
    cp, da, fw = cyc.condensate_pump_out, cyc.deaerator, cyc.feedwater
    steps.append(
        _pump_step(
            result,
            0,
            cp,
            system,
            title=f"Bomba de condensado (1 → {cp + 1})",
            what="del condensador al desaireador",
        )
    )
    s_da = cyc.states[da]
    k = da + 1
    steps.append(
        ProcedureStep(
            title=f"Estado {k}: salida del desaireador",
            text=(
                "El desaireador es un calentador abierto: la extracción de la turbina se mezcla "
                "con el condensado y sale líquido saturado a la presión del desaireador "
                f"({_bar(s_da.P_Pa)}; Cengel §10-6). Su caudal sale del balance, más abajo."
            ),
            latex=(
                rf"T_{_ix(k)} = T_{{\mathrm{{sat}}}}(p_{_ix(k)}) = "
                rf"{_q(s_da.T_K, 'temperature', system)}",
                rf"h_{_ix(k)} = h_f(p_{_ix(k)}) = {_q(s_da.h_J_per_kg, _EH, system)}",
                rf"v_{_ix(k)} = v_f(p_{_ix(k)}) = "
                rf"{_q(s_da.v_m3_per_kg, 'specific_volume', system)}",
            ),
        )
    )
    steps.append(
        _pump_step(
            result,
            da,
            fw,
            system,
            title=f"Bomba de alimentación ({k} → {fw + 1})",
            what=f"del desaireador {target}",
        )
    )
    return steps


# ---------------------------------------------------------------------
# Vapor de la HRSG, turbinas y mezclas
# ---------------------------------------------------------------------


def _hrsg_steam_step(result: MultiCombinedResult, system: UnitSystem) -> ProcedureStep:
    """Qué entrega la HRSG: el vapor de cada nivel (estados del ciclo) y sus caudales."""
    cyc = result.steam
    names = result.inputs.names
    lines: list[str] = []
    for level, k in enumerate(cyc.level_states):
        lv = result.hrsg.levels[level]
        top = 5 if lv.level.superheated else 4
        letter = _LETTERS[names[level]]
        hrsg_h = f"h_{{{top},{letter}}}" if letter else f"h_{top}"
        lines.append(rf"h_{_ix(k + 1)} = {hrsg_h} = {_q(cyc.states[k].h_J_per_kg, _EH, system)}")
        lines.append(rf"{_m_level(result, level)} = {_q(lv.m_steam_kg_s, _M, system)}")
    feed = cyc.feedwater + 1
    low = names[-1]
    where = f"del nivel de {low} (1_{_LETTERS[low]})" if low else "(1)"
    return ProcedureStep(
        title="Vapor que entrega la caldera de recuperación",
        text=(
            f"La bomba entrega el agua a la HRSG: el estado {feed} es el agua de alimentación "
            f"{where}. La HRSG calcula los caudales de cada nivel y entrega su vapor a la presión "
            "de su domo (con la numeración de la caldera, 5 sobrecalentado o 4 saturado). El de "
            "alta entra por la turbina de alta; el de los otros niveles, en las admisiones."
        ),
        latex=tuple(lines),
    )


def _turbine_step(
    result: MultiCombinedResult, t: TurbineSection, system: UnitSystem
) -> ProcedureStep:
    """Una turbina: estado isoentrópico (palanca si cae en la campana), real y la extracción."""
    cyc = result.steam
    st = cyc.states
    s_in, s_out = st[t.inlet], st[t.outlet]
    a, b = t.inlet + 1, t.outlet + 1
    eta = result.inputs.steam.eta_turbine
    h_ks = t.outlet_s.h_J_per_kg
    lines = [rf"s_{{{b}s}} = s_{_ix(a)} = {_q(s_in.s_J_per_kg_K, _ES, system)}"]
    sat = (
        saturation_at_pressure("Water", s_out.P_Pa)
        if s_out.P_Pa < fluid_limits("Water").P_crit_Pa
        else None
    )
    wet_s = (
        sat is not None and sat.liquid.s_J_per_kg_K <= s_in.s_J_per_kg_K <= sat.vapor.s_J_per_kg_K
    )
    if wet_s:
        assert sat is not None
        x_s = (s_in.s_J_per_kg_K - sat.liquid.s_J_per_kg_K) / sat.s_fg_J_per_kg_K
        lines += [
            latex_chain(
                rf"x_{{{b}s}}",
                r"\frac{s - s_f}{s_{fg}}",
                rf"\frac{{{_n(s_in.s_J_per_kg_K, _ES, system)} - "
                rf"{latex_paren(_n(sat.liquid.s_J_per_kg_K, _ES, system))}}}"
                rf"{{{_n(sat.s_fg_J_per_kg_K, _ES, system)}}}",
                latex_number(x_s, 5),
            ),
            latex_chain(
                _hs(b),
                rf"h_f + x_{{{b}s}}\,h_{{fg}}",
                _wrap(
                    _n(sat.liquid.h_J_per_kg, _EH, system),
                    "+",
                    rf"{latex_number(x_s, 5)}\cdot {_n(sat.h_fg_J_per_kg, _EH, system)}",
                ),
                _q(h_ks, _EH, system),
            ),
        ]
        how = (
            f"A {_bar(s_out.P_Pa)} la entropía de la entrada cae dentro de la campana: el "
            f"estado {b}s es vapor húmedo y su título sale de la regla de la palanca (Cengel A-5)."
        )
    else:
        lines.append(latex_chain(_hs(b), rf"h(p_{_ix(b)},\ s_{{{b}s}})", _q(h_ks, _EH, system)))
        how = (
            f"A {_bar(s_out.P_Pa)} el estado isoentrópico {b}s sigue sobrecalentado: se lee "
            "en la tabla de vapor sobrecalentado (Cengel A-6), interpolando."
        )
    texts = [how]
    if t.extraction is not None:  # sobre la misma línea, antes de la salida
        lines += _extraction_lines(result, t, system)
    if t.wet is None:
        lines.append(
            latex_chain(
                f"h_{_ix(b)}",
                rf"h_{_ix(a)} - \eta_T\,(h_{_ix(a)} - {_hs(b)})",
                rf"{_n(s_in.h_J_per_kg, _EH, system)} \\ &\quad - "
                + _factor_diff(
                    latex_number(eta, 4), s_in.h_J_per_kg, h_ks, _EH, system, indent=r"\qquad"
                ),
                _q(s_out.h_J_per_kg, _EH, system),
            )
        )
        texts.append(f"Con η_T = {eta:g}, la turbina entrega esa fracción del salto isoentrópico.")
    else:
        lines += _baumann_lines(result, t, system)
        texts.append(
            "Con la regla de Baumann, la humedad baja el rendimiento: la expansión se parte "
            f"donde la línea de expansión cruza la curva de vapor saturado ({_bar(t.wet.p_x_Pa)}); "
            f"la parte seca va con η_T = {eta:g} y la húmeda con η_T·(1 − α·ȳ), con ȳ la "
            "humedad media de esa parte (se itera porque la de salida depende del resultado)."
        )
    if s_out.x is not None and sat is not None and 0.0 < s_out.x < 1.0:
        lines.append(
            latex_chain(
                f"x_{_ix(b)}",
                r"\frac{h - h_f}{h_{fg}}",
                rf"\frac{{{_n(s_out.h_J_per_kg, _EH, system)} - "
                rf"{latex_paren(_n(sat.liquid.h_J_per_kg, _EH, system))}}}"
                rf"{{{_n(sat.h_fg_J_per_kg, _EH, system)}}}",
                latex_number(s_out.x, 5),
            )
        )
    m_sym = rf"\dot{{m}}_{{{a}}}"
    lines.append(rf"{m_sym} = {_q(t.m_in_kg_s, _M, system)}")
    W_sym = rf"\dot{{W}}_{{T,{a}}}"
    if t.extraction is None:
        lines.append(
            latex_chain(
                W_sym,
                rf"{m_sym}\,(h_{_ix(a)} - h_{_ix(b)})",
                _factor_diff(
                    _n(t.m_in_kg_s, _M, system), s_in.h_J_per_kg, s_out.h_J_per_kg, _EH, system
                ),
                _q(t.W_W, _P, system),
            )
        )
    else:
        e = t.extraction + 1
        ext = st[t.extraction]
        lines.append(
            latex_chain(
                W_sym,
                rf"{m_sym}\,(h_{_ix(a)} - h_{_ix(e)}) \\ &\quad"
                rf" + (\dot{{m}}_{{{a}}} - \dot{{m}}_{{\mathrm{{ext}}}})"
                rf"\,(h_{_ix(e)} - h_{_ix(b)})",
                _q(t.W_W, _P, system),
            )
        )
        texts.append(
            f"La extracción del desaireador ({e}) sale a {_bar(ext.P_Pa)}, sobre la misma línea "
            "de expansión (η_T desde la entrada de la turbina, como en Cengel)."
        )
    return ProcedureStep(
        title=f"{_cap(t.name)} ({a} → {b})", text=" ".join(texts), latex=tuple(lines)
    )


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _baumann_lines(result: MultiCombinedResult, t: TurbineSection, system: UnitSystem) -> list[str]:
    """La parte húmeda de la expansión con la regla de Baumann."""
    assert t.wet is not None
    w = t.wet
    st = result.steam.states
    b = t.outlet + 1
    alpha = result.inputs.baumann_alpha
    eta = result.inputs.steam.eta_turbine
    s_out = st[t.outlet]
    lines = [
        rf"p_x = {_q(w.p_x_Pa, 'pressure', system)}",
        rf"h_x = h_g(p_x) = {_q(w.h_x_J_per_kg, _EH, system)}",
    ]
    s_x = float(PropsSI("S", "P", w.p_x_Pa, "H", w.h_x_J_per_kg, "Water"))
    h_s2 = float(PropsSI("H", "P", s_out.P_Pa, "S", s_x, "Water"))
    lines += [
        rf"h_{{{b}s'}} = h(p_{_ix(b)},\ s_x) = {_q(h_s2, _EH, system)}",
        latex_chain(
            r"\bar{y}",
            r"\frac{y_x + y_{" + str(b) + r"}}{2}",
            rf"\frac{{{latex_number(w.y_in, 4)} + {latex_number(w.y_out, 4)}}}{{2}}",
            latex_number((w.y_in + w.y_out) / 2, 4),
        ),
        latex_chain(
            r"\eta_{\mathrm{h}}",
            r"\eta_T\,(1 - \alpha\,\bar{y})",
            rf"{latex_number(eta, 4)}\,(1 - {latex_number(alpha, 3)}\cdot "
            rf"{latex_number((w.y_in + w.y_out) / 2, 4)})",
            latex_number(w.eta_wet, 4),
        ),
        latex_chain(
            f"h_{_ix(b)}",
            rf"h_x - \eta_{{\mathrm{{h}}}}\,(h_x - h_{{{b}s'}})",
            rf"{_n(w.h_x_J_per_kg, _EH, system)} \\ &\quad - "
            + _factor_diff(
                latex_number(w.eta_wet, 4), w.h_x_J_per_kg, h_s2, _EH, system, indent=r"\qquad"
            ),
            _q(s_out.h_J_per_kg, _EH, system),
        ),
    ]
    return lines


def _extraction_lines(
    result: MultiCombinedResult, t: TurbineSection, system: UnitSystem
) -> list[str]:
    """El estado de la extracción del desaireador, sobre la línea de expansión."""
    assert t.extraction is not None
    st = result.steam.states
    s_in, ext = st[t.inlet], st[t.extraction]
    a, e = t.inlet + 1, t.extraction + 1
    eta = result.inputs.steam.eta_turbine
    if t.wet is None or ext.P_Pa >= t.wet.p_x_Pa:
        h_es = float(PropsSI("H", "P", ext.P_Pa, "S", s_in.s_J_per_kg_K, "Water"))
        return [
            rf"{_hs(e)} = h(p_{_ix(e)},\ s_{_ix(a)}) = {_q(h_es, _EH, system)}",
            latex_chain(
                f"h_{_ix(e)}",
                rf"h_{_ix(a)} - \eta_T\,(h_{_ix(a)} - {_hs(e)})",
                rf"{_n(s_in.h_J_per_kg, _EH, system)} \\ &\quad - "
                + _factor_diff(
                    latex_number(eta, 4), s_in.h_J_per_kg, h_es, _EH, system, indent=r"\qquad"
                ),
                _q(ext.h_J_per_kg, _EH, system),
            ),
        ]
    w = t.wet
    s_x = float(PropsSI("S", "P", w.p_x_Pa, "H", w.h_x_J_per_kg, "Water"))
    h_es = float(PropsSI("H", "P", ext.P_Pa, "S", s_x, "Water"))
    return [
        rf"h_{{{e}s'}} = h(p_{_ix(e)},\ s_x) = {_q(h_es, _EH, system)}",
        latex_chain(
            f"h_{_ix(e)}",
            rf"h_x - \eta_{{\mathrm{{h}}}}\,(h_x - h_{{{e}s'}})",
            _q(ext.h_J_per_kg, _EH, system),
        ),
    ]


def _mixing_step(result: MultiCombinedResult, index: int, system: UnitSystem) -> ProcedureStep:
    """Una admisión: el vapor de un nivel se mezcla con el de la turbina (o el recalentamiento)."""
    cyc = result.steam
    adm = cyc.admissions[index]
    st = cyc.states
    t, s, o = adm.turbine + 1, adm.steam + 1, adm.outlet + 1
    name = result.inputs.names[adm.level]
    into_rh = adm.outlet == cyc.reheat_inlet
    mt, ms = rf"\dot{{m}}_{{{t}}}", _m_level(result, adm.level)
    lines = [
        rf"{mt} = {_q(adm.m_turbine_kg_s, _M, system)}",
        latex_chain(
            f"h_{_ix(o)}",
            rf"\frac{{{mt}\,h_{_ix(t)} + {ms}\,h_{_ix(s)}}}{{{mt} + {ms}}}",
            _q(st[adm.outlet].h_J_per_kg, _EH, system),
        ),
        rf"T_{_ix(o)} = T(p,\ h_{_ix(o)}) = {_q(st[adm.outlet].T_K, 'temperature', system)}",
    ]
    where = (
        "con el recalentamiento frío, antes del recalentador"
        if into_rh
        else "con el vapor que sale de la turbina"
    )
    return ProcedureStep(
        title=f"Mezcla con el vapor de {name} ({t} + {s} → {o})",
        text=(
            f"El vapor de {name} ({s}) entra a {_bar(st[adm.outlet].P_Pa)} y se mezcla {where} "
            f"({t}). Es una mezcla adiabática a presión constante: el balance de energía da la "
            "entalpía de la mezcla (vademecum §3.3) y, con p y h, su temperatura."
        ),
        latex=tuple(lines),
    )


def _reheat_step(result: MultiCombinedResult, system: UnitSystem) -> ProcedureStep:
    cyc = result.steam
    st = cyc.states
    assert cyc.hot_reheat is not None and cyc.reheat_inlet is not None
    i, o = cyc.reheat_inlet + 1, cyc.hot_reheat + 1
    rh = result.hrsg.reheat
    assert rh is not None
    return ProcedureStep(
        title=f"Recalentador ({i} → {o})",
        text=(
            f"El vapor vuelve a la HRSG y se recalienta a presión constante hasta "
            f"{_degC(st[cyc.hot_reheat].T_K)} (el recalentamiento caliente; tabla A-6). Su calor "
            "está en la caldera de recuperación: lo ceden los gases en el tramo en paralelo con "
            "el sobrecalentador de alta (Cengel §10-5)."
        ),
        latex=(
            rf"h_{_ix(o)} = h(p,\ T_{_ix(o)}) = {_q(st[cyc.hot_reheat].h_J_per_kg, _EH, system)}",
            latex_chain(
                r"\dot{Q}_{\mathrm{RH}}",
                rf"\dot{{m}}_{{\mathrm{{RH}}}}\,(h_{_ix(o)} - h_{_ix(i)})",
                _q(rh.Q_W, _P, system),
            ),
        ),
    )


# ---------------------------------------------------------------------
# Desaireador, condensador y potencias
# ---------------------------------------------------------------------


def _deaerator_step(result: MultiCombinedResult, system: UnitSystem) -> ProcedureStep:
    cyc = result.steam
    st = cyc.states
    ext = cyc.extraction
    assert ext is not None and cyc.deaerator is not None
    e, c, d = ext + 1, cyc.condensate_pump_out + 1, cyc.deaerator + 1
    y = cyc.y_deaerator
    h_d, h_c, h_e = (st[k].h_J_per_kg for k in (cyc.deaerator, cyc.condensate_pump_out, ext))
    return ProcedureStep(
        title="Balance del desaireador",
        text=(
            "Por cada kilogramo de vapor que produce la HRSG, una fracción y se extrae de la "
            "turbina y calienta el resto, que llega de la bomba de condensado, hasta líquido "
            "saturado (Cengel §10-6): y·h_ext + (1 − y)·h_bomba = h_f."
        ),
        latex=(
            rf"y\,h_{_ix(e)} + (1 - y)\,h_{_ix(c)} = h_{_ix(d)}",
            latex_chain(
                "y",
                rf"\frac{{h_{_ix(d)} - h_{_ix(c)}}}{{h_{_ix(e)} - h_{_ix(c)}}}",
                rf"\frac{{{_diff(h_d, h_c, _EH, system)}}}{{{_diff(h_e, h_c, _EH, system)}}}",
                latex_number(y, 5),
            ),
            latex_chain(
                r"\dot{m}_{\mathrm{ext}}",
                r"y\,\dot{m}_{v,\mathrm{total}}",
                _q(cyc.m_extraction_kg_s, _M, system),
            ),
        ),
    )


def _condenser_step(result: MultiCombinedResult, system: UnitSystem) -> ProcedureStep:
    cyc = result.steam
    st = cyc.states
    x = cyc.exhaust + 1
    lines = [
        latex_chain(
            r"\dot{m}_{\mathrm{cond}}",
            r"\dot{m}_{v,\mathrm{total}} - \dot{m}_{\mathrm{ext}}"
            if cyc.deaerator is not None
            else r"\dot{m}_{v,\mathrm{total}}",
            _q(cyc.m_condenser_kg_s, _M, system),
        ),
        latex_chain(
            r"\dot{Q}_{\mathrm{cond}}",
            rf"\dot{{m}}_{{\mathrm{{cond}}}}\,(h_{_ix(x)} - h_1)",
            _factor_diff(
                _n(cyc.m_condenser_kg_s, _M, system),
                st[cyc.exhaust].h_J_per_kg,
                st[0].h_J_per_kg,
                _EH,
                system,
            ),
            _q(cyc.Q_condenser_W, _P, system),
        ),
    ]
    return ProcedureStep(
        title=f"Condensador ({x} → 1)",
        text=(
            "Todo el vapor que llega al condensador (el total, menos la extracción si hay "
            "desaireador) se condensa hasta líquido saturado: ese calor se lo lleva el agua de "
            "enfriamiento."
        ),
        latex=tuple(lines),
    )


def _power_step(result: MultiCombinedResult, system: UnitSystem) -> ProcedureStep:
    cyc = result.steam
    terms = [rf"\dot{{W}}_{{T,{t.inlet + 1}}}" for t in cyc.turbines]
    values = [_n(t.W_W, _P, system) for t in cyc.turbines]
    lines = [
        latex_chain(
            r"\dot{W}_T",
            " + ".join(terms),
            " \\\\ &\\quad + ".join(values) if len(values) > 2 else " + ".join(values),
            _q(cyc.W_turbines_W, _P, system),
        ),
        latex_chain(
            r"\dot{W}_{B}",
            r"\dot{W}_{B,\mathrm{ciclo}} + \dot{W}_{B,\mathrm{HRSG}}",
            f"{_n(cyc.W_pumps_W, _P, system)} + {_n(result.hrsg.W_pumps_W, _P, system)}",
            _q(result.W_pumps_W, _P, system),
        ),
        latex_chain(
            r"\dot{W}_{TV}",
            r"\dot{W}_T - \dot{W}_B",
            _diff(cyc.W_turbines_W, result.W_pumps_W, _P, system),
            _q(result.W_steam_turbine_W, _P, system),
        ),
    ]
    return ProcedureStep(
        title="Potencia del ciclo de vapor",
        text=(
            "La potencia de las turbinas es la suma de cada tramo; las bombas son la de "
            "condensado (y la de alimentación, con desaireador) más las que pasan el agua de "
            "un domo al nivel siguiente en la HRSG."
        ),
        latex=tuple(lines),
    )


def steam_cycle_steps(result: MultiCombinedResult, system: UnitSystem) -> list[ProcedureStep]:
    """El ciclo de vapor con admisiones, en el orden de Cengel (cap. 10)."""
    cyc = result.steam
    steps = _feed_line_steps(result, system)
    steps.append(_hrsg_steam_step(result, system))
    admissions = {a.turbine: k for k, a in enumerate(cyc.admissions)}
    for t in cyc.turbines:
        steps.append(_turbine_step(result, t, system))
        if t.outlet in admissions:
            steps.append(_mixing_step(result, admissions[t.outlet], system))
        if t.outlet == cyc.cold_reheat:
            steps.append(_reheat_step(result, system))
    if cyc.deaerator is not None:
        steps.append(_deaerator_step(result, system))
    steps += [_condenser_step(result, system), _power_step(result, system)]
    return steps


# ---------------------------------------------------------------------
# Ciclo combinado: acople, rendimiento, balance y exergía
# ---------------------------------------------------------------------


def _coupling_step(result: MultiCombinedResult, system: UnitSystem) -> ProcedureStep:
    cyc = result.steam
    feed = cyc.states[cyc.feedwater]
    return ProcedureStep(
        title="Acople: turbina de gas, HRSG y ciclo de vapor",
        text=(
            "La turbina de gas fija los gases de escape (temperatura, caudal y composición); el "
            f"ciclo de vapor fija el agua de alimentación (estado {cyc.feedwater + 1}) y, con "
            "recalentamiento, el vapor que vuelve de la turbina de alta. Con eso, la HRSG da "
            "los caudales de cada nivel y la turbina de vapor, las potencias."
        ),
        latex=(
            rf"T_{{\mathrm{{alim}}}} = T_{_ix(cyc.feedwater + 1)} = "
            rf"{_q(feed.T_K, 'temperature', system)}",
            latex_chain(
                r"\dot{m}_{v,\mathrm{total}}",
                " + ".join(_m_level(result, k) for k in range(len(result.hrsg.levels))),
                _q(cyc.m_total_kg_s, _M, system),
            ),
            latex_chain(
                r"\eta_{TV}",
                r"\frac{\dot{W}_{TV}}{\dot{Q}_{\mathrm{HRSG}}}",
                rf"\frac{{{_n(result.W_steam_turbine_W, _P, system)}}}"
                rf"{{{_n(result.Q_hrsg_W, _P, system)}}}",
                rf"{latex_number(result.eta_steam * 100, 4)}\,\%",
            ),
        ),
    )


def _exergy_step(result: MultiCombinedResult, system: UnitSystem) -> ProcedureStep:
    x = bottoming_exergy(result)
    hrsg_names = {s.name for s in result.hrsg.sections}
    groups: dict[str, float] = {"HRSG": 0.0, "turbinas": 0.0, "mezclas": 0.0, "otros": 0.0}
    for name, value in x.destroyed_W:
        if name in hrsg_names:
            groups["HRSG"] += value
        elif name.startswith("turbina"):
            groups["turbinas"] += value
        elif name.startswith("mezcla"):
            groups["mezclas"] += value
        else:
            groups["otros"] += value
    cyc = result.steam
    ex = cyc.exhaust + 1
    lines = [
        rf"\dot{{X}}_{{\mathrm{{gases}}}} = {_q(x.X_gas_in_W, _P, system)}",
        latex_chain(
            r"\dot{X}_{\mathrm{dest,HRSG}}",
            r"\sum T_0\,\dot{S}_{\mathrm{gen}}",
            _q(groups["HRSG"], _P, system),
        ),
        latex_chain(
            r"\dot{X}_{\mathrm{dest},T}",
            r"T_0 \sum \dot{m}\,(s_{\mathrm{sal}} - s_{\mathrm{ent}})",
            _q(groups["turbinas"], _P, system),
        ),
    ]
    if cyc.admissions:
        lines.append(
            latex_chain(
                r"\dot{X}_{\mathrm{dest,mez}}",
                r"T_0\,\bigl(\dot{m}_{\mathrm{sal}}\,s_{\mathrm{sal}} - \sum \dot{m}\,s\bigr)",
                _q(groups["mezclas"], _P, system),
            )
        )
    lines += [
        latex_chain(
            r"\dot{X}_{\mathrm{dest,otros}}",
            r"\dot{X}_{\mathrm{dest},B} + \dot{X}_{\mathrm{dest,DA}}"
            if cyc.deaerator is not None
            else r"\dot{X}_{\mathrm{dest},B}",
            _q(groups["otros"], _P, system),
        ),
        latex_chain(
            r"\dot{X}_{\mathrm{cond}}",
            rf"\dot{{m}}_{{\mathrm{{cond}}}}\,\bigl[(h_{_ix(ex)} - h_1) \\ &\quad"
            rf" - T_0\,(s_{_ix(ex)} - s_1)\bigr]",
            _q(x.X_condenser_W, _P, system),
        ),
        rf"\dot{{X}}_{{\mathrm{{chim}}}} = {_q(x.X_stack_W, _P, system)}",
        latex_chain(
            r"\eta_{\mathrm{II,fondo}}",
            r"\frac{\dot{W}_{TV}}{\dot{X}_{\mathrm{gases}}}",
            rf"{latex_number(x.efficiency * 100, 4)}\,\%",
        ),
    ]
    share = ", ".join(
        f"{name} {value / x.X_gas_in_W * 100:.1f} %"
        for name, value in (
            ("trabajo", x.W_net_W),
            ("HRSG", groups["HRSG"]),
            ("turbinas", groups["turbinas"]),
            ("mezclas", groups["mezclas"]),
            ("bombas y desaireador", groups["otros"]),
            ("condensador", x.X_condenser_W),
            ("chimenea", x.X_stack_W),
        )
        if value > 0.0
    )
    return ProcedureStep(
        title="Exergía del ciclo de fondo",
        text=(
            "La exergía de los gases de escape (lo máximo que podría dar el ciclo de vapor con el "
            "ambiente a T₀) termina como trabajo neto, se destruye en cada equipo (X_dest = "
            "T₀·S_gen: la HRSG por transferir calor con diferencia de temperatura, las turbinas "
            "por fricción, las mezclas por juntar vapores a distinta temperatura), se la lleva "
            "el agua de enfriamiento del condensador o sale por la chimenea (Cengel cap. 8). "
            f"De cada 100 unidades: {share}."
        ),
        latex=tuple(lines),
    )


def combined_multi_sections(
    result: MultiCombinedResult, system: UnitSystem
) -> list[tuple[str, list[ProcedureStep]]]:
    """Procedimiento por partes: (título de la parte, pasos)."""
    return [
        ("Turbina de gas", brayton_steps(result.gas_turbine, system)),
        # La composición de los gases ya está en la turbina de gas.
        ("Caldera de recuperación", multi_hrsg_steps(result.hrsg, system)[1:]),
        ("Ciclo de vapor", steam_cycle_steps(result, system)),
        (
            "Ciclo combinado",
            [
                _coupling_step(result, system),
                _efficiency_step(result, system),  # type: ignore[arg-type]
                _balance_step(result, system),  # type: ignore[arg-type]
                _exergy_step(result, system),
            ],
        ),
    ]
