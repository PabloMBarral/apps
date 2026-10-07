"""Procedimiento de la HRSG de varias presiones «como a mano» — Fase 3.5.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo: la composición y la entalpía de los gases (vademecum §5), y
después cada nivel de presión, de mayor a menor (Kehlhofer et al., 2009,
cap. 5): sus estados del agua (tablas de vapor, Cengel A-4 a A-7), el caudal
de vapor del balance hasta su pinch, el sobrecalentador, el evaporador (con
el agua que sigue hacia los niveles de mayor presión) y el economizador. Al
final, el calor total, el balance de exergía de la caldera (Cengel cap. 8:
X_destruida = T₀·S_gen) y el punto de rocío de los gases.

Los estados del agua de cada nivel se numeran como en la caldera de una
presión: 1 entrada al economizador, 2 salida, 3 líquido saturado, 4 vapor
saturado y 5 vapor sobrecalentado, con el nivel como subíndice (A alta,
M media, B baja). Con recalentador (Fase 3.6): RF es el recalentamiento frío
(el vapor que vuelve de la turbina de alta), RC el caliente y «mez» la mezcla
con el vapor de media, si se suma. Recibe un resultado ya calculado
(:class:`core.cycles.hrsg_multi.MultiHRSGResult`): no importa Streamlit.
"""

from __future__ import annotations

from core.cycles.hrsg_multi import (
    LevelResult,
    MultiHRSGResult,
    hrsg_exergy,
    reheat_system,
)
from core.cycles.hrsg_procedure import (
    dew_point_step,
    gas_composition_step,
    times_diff,
)
from core.cycles.rankine_procedure import _bar, _diff, _n, _q, _wrap
from core.latex import latex_chain, latex_is_wide, latex_number
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem

_EH: QuantityKind = "specific_enthalpy"
_LETTERS = {"alta": "A", "media": "M", "baja": "B", "": ""}


def _T(value_K: float, system: UnitSystem) -> str:
    return _q(value_K, "temperature", system)


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _sub(level: LevelResult) -> str:
    """Subíndice del nivel para los estados: ``,A`` (vacío con un solo nivel)."""
    letter = _LETTERS[level.name]
    return f",{letter}" if letter else ""


def _h(k: int, level: LevelResult) -> str:
    """``h_{4,A}``: el estado k del nivel."""
    return f"h_{{{k}{_sub(level)}}}"


def _m(level: LevelResult) -> str:
    letter = _LETTERS[level.name]
    return rf"\dot{{m}}_{{{letter}}}" if letter else r"\dot{m}_v"


def _title(level: LevelResult) -> str:
    return f"Nivel de {level.name}" if level.name else "Caldera"


def _lookup(lhs: str, how: str, value_tex: str) -> str:
    """``h_{2,M} = h(p, T_{2,M}) = …``; con ×10ⁿ (SI) en dos renglones (celular)."""
    if latex_is_wide(value_tex):
        return latex_chain(lhs, how, value_tex)
    return f"{lhs} = {how} = {value_tex}"


def _gas_labels(result: MultiHRSGResult) -> list[list[str]]:
    """Rótulos de los puntos de los gases de cada nivel (el primero es el de llegada)."""
    labels = [label for label, _, _ in result.gas_points]
    out: list[list[str]] = []
    k = 0
    for lv in result.levels:
        out.append(labels[k : k + len(lv.T_gas_K)])
        k += len(lv.T_gas_K) - 1
    return out


def _gas_enthalpy_step(result: MultiHRSGResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    return ProcedureStep(
        title="Entalpía de los gases",
        text=(
            f"Los gases entran a T_a = {_degC(inputs.T_gas_in_K)} y recorren los niveles de mayor "
            "a menor presión; en cada uno se enfrían hasta su pinch, T_sat + ΔT_pinch, y después "
            "pasan por su economizador. En un gas ideal la entalpía depende solo de T: la de "
            "cada componente sale del gas ideal de CoolProp (como integrar los polinomios NASA, "
            "vademecum §4.8) y la de la mezcla es la suma pesada con las fracciones másicas "
            "(vademecum §5.6), medida desde 25 °C."
        ),
        latex=(
            r"h_g(T) = \sum_i w_i\,\bigl[h_i(T) - h_i(25\ {}^{\circ}\mathrm{C})\bigr]",
            rf"h_g(T_a) = {_q(result.levels[0].h_gas_J_per_kg[0], _EH, system)}",
        ),
    )


def _water_step(result: MultiHRSGResult, i: int, system: UnitSystem) -> ProcedureStep:
    """Estados del agua del nivel i: entrada, economizador, domo y vapor."""
    lv = result.levels[i]
    level = lv.level
    w = lv.water
    last = i == len(result.levels) - 1
    lines = [rf"T_{{\mathrm{{sat}}{_sub(lv)}}} = {_T(lv.T_sat_K, system)}"]
    if last:
        lines.append(
            _lookup(_h(1, lv), r"h(p,\ T_{\mathrm{alim}})", _q(w[0].h_J_per_kg, _EH, system))
        )
        inlet = (
            f"El agua de alimentación entra a {_degC(w[0].T_K)} (líquido comprimido: h(p, T) "
            "de la tabla A-7 o, aproximando, h_f(T) de la A-4)."
        )
    else:
        below = result.levels[i + 1]
        h_f_below = below.water[2].h_J_per_kg
        eta = result.inputs.eta_pump
        if eta == 1.0:
            lines += [
                _lookup(
                    _h(1, lv), rf"h(p,\ s_{{3{_sub(below)}}})", _q(w[0].h_J_per_kg, _EH, system)
                ),
                latex_chain(
                    rf"w_{{B{_sub(lv)}}}",
                    rf"{_h(1, lv)} - {_h(3, below)}",
                    _diff(w[0].h_J_per_kg, h_f_below, _EH, system),
                    _q(lv.w_pump_J_per_kg, _EH, system),
                ),
            ]
            how = "se toma isoentrópica (s igual)"
        else:
            h_s = h_f_below + lv.w_pump_J_per_kg * eta
            sub_s = rf"h_{{1s{_sub(lv)}}}"
            lines += [
                _lookup(sub_s, rf"h(p,\ s_{{3{_sub(below)}}})", _q(h_s, _EH, system)),
                latex_chain(
                    rf"w_{{B{_sub(lv)}}}",
                    rf"\frac{{{sub_s} - {_h(3, below)}}}{{\eta_B}}",
                    rf"\frac{{{_diff(h_s, h_f_below, _EH, system)}}}{{{latex_number(eta, 4)}}}",
                    _q(lv.w_pump_J_per_kg, _EH, system),
                ),
                latex_chain(
                    _h(1, lv),
                    rf"{_h(3, below)} + w_{{B{_sub(lv)}}}",
                    _q(w[0].h_J_per_kg, _EH, system),
                ),
            ]
            how = f"tiene rendimiento η_B = {eta:.3g} (con s igual da h_1s; el real, h_1)"
        inlet = (
            f"Su economizador recibe el líquido saturado del domo de {below.name}, que una bomba "
            f"lleva de {_bar(below.level.p_Pa)} a {_bar(level.p_Pa)}. La bomba {how} y queda "
            "fuera de la caldera: su trabajo es chico (vademecum §13)."
        )
    if level.approach_K > 0.0:
        lines.append(
            latex_chain(
                rf"T_{{2{_sub(lv)}}}",
                rf"T_{{\mathrm{{sat}}{_sub(lv)}}} - \Delta T_{{\mathrm{{approach}}}}",
                rf"{_n(lv.T_sat_K, 'temperature', system)} - "
                rf"{_n(level.approach_K, 'temperature_difference', system)}",
                _T(w[1].T_K, system),
            )
        )
        lines.append(
            _lookup(_h(2, lv), rf"h(p,\ T_{{2{_sub(lv)}}})", _q(w[1].h_J_per_kg, _EH, system))
        )
    else:
        lines.append(rf"{_h(2, lv)} = h_f(p) = {_q(w[1].h_J_per_kg, _EH, system)}")
    lines.append(rf"{_h(3, lv)} = h_f(p) = {_q(w[2].h_J_per_kg, _EH, system)}")
    lines.append(rf"{_h(4, lv)} = h_g(p) = {_q(w[3].h_J_per_kg, _EH, system)}")
    if level.superheated:
        lines.append(
            _lookup(_h(5, lv), rf"h(p,\ T_{{5{_sub(lv)}}})", _q(w[4].h_J_per_kg, _EH, system))
        )
    steam = (
        f"vapor sobrecalentado a {_degC(w[4].T_K)} (tabla A-6)"
        if level.superheated
        else "vapor saturado"
    )
    return ProcedureStep(
        title=f"{_title(lv)}: estados del agua ({_bar(level.p_Pa)})",
        text=(
            f"A {_bar(level.p_Pa)}: T_sat, h_f y h_g de la tabla de saturación por presión "
            f"(Cengel A-5). {inlet} El economizador entrega el agua a T_sat − approach y el "
            f"nivel produce {steam}."
        ),
        latex=tuple(lines),
    )


def _flow_step(result: MultiHRSGResult, i: int, system: UnitSystem) -> ProcedureStep:
    """Caudal de vapor del nivel i: balance de la llegada de los gases a su pinch."""
    inputs = result.inputs
    lv = result.levels[i]
    level = lv.level
    m_g = inputs.m_gas_kg_s
    labels = _gas_labels(result)[i]
    a, p = labels[0], labels[-2]
    h_in, h_p = lv.h_gas_J_per_kg[0], lv.h_gas_J_per_kg[-2]
    Q = m_g * (h_in - h_p)
    top = 5 if level.superheated else 4
    w = lv.water
    h_top, h2, h3 = w[-1].h_J_per_kg, w[1].h_J_per_kg, w[2].h_J_per_kg
    above = result.levels[:i]
    rh = result.reheat if i == 0 else None
    q_sym = _q_top(lv, rh is not None)
    lines = [
        latex_chain(
            f"T_{p}",
            rf"T_{{\mathrm{{sat}}{_sub(lv)}}} + \Delta T_{{\mathrm{{pinch}}}}",
            rf"{_n(lv.T_sat_K, 'temperature', system)} + "
            rf"{_n(level.pinch_K, 'temperature_difference', system)}",
            _T(lv.T_pinch_gas_K, system),
        ),
        rf"h_g(T_{p}) = {_q(h_p, _EH, system)}",
        latex_chain(
            q_sym,
            rf"\dot{{m}}_g\,\bigl[h_g(T_{a}) - h_g(T_{p})\bigr]",
            times_diff(m_g, "mass_flow", h_in, h_p, system),
            _q(Q, "power", system),
        ),
    ]
    devices = (
        "están el sobrecalentador y el evaporador" if level.superheated else "está el evaporador"
    )
    if rh is not None:
        h_rc, h_rf = rh.hot.h_J_per_kg, rh.cold.h_J_per_kg
        per_kg = (h_top - h2) + (h_rc - h_rf)
        lines += [
            latex_chain(
                r"\Delta h_{\mathrm{RH}}",
                r"h_{\mathrm{RC}} - h_{\mathrm{RF}}",
                _diff(h_rc, h_rf, _EH, system),
                _q(h_rc - h_rf, _EH, system),
            ),
            latex_chain(
                _m(lv),
                rf"\frac{{{q_sym}}}{{({_h(top, lv)} - {_h(2, lv)}) + \Delta h_{{\mathrm{{RH}}}}}}",
                rf"\frac{{{_n(Q, 'power', system)}}}{{{_n(per_kg, _EH, system)}}}",
                _q(lv.m_steam_kg_s, "mass_flow", system),
            ),
        ]
        devices = (
            "están el sobrecalentador y el recalentador (en paralelo) y el evaporador"
            if level.superheated
            else "están el recalentador y el evaporador"
        )
        return ProcedureStep(
            title=f"{_title(lv)}: caudal de vapor (gases {a} → {p})",
            text=(
                f"Los gases llegan a T_{a} y salen del evaporador a T_{p} = T_sat + pinch. En "
                f"ese tramo {devices}: su calor es Q̇_tramo. El recalentador lleva el mismo vapor "
                "(vuelve de la turbina de alta), así que cada kilogramo recibe su calor en la "
                f"caldera, de 2 al estado {top}, más el del recalentamiento, de RF a RC:"
            ),
            latex=(
                r"\begin{aligned}"
                rf"&{_m(lv)}\,\bigl[({_h(top, lv)} - {_h(2, lv)})"
                r" + \Delta h_{\mathrm{RH}}\bigr]"
                rf" \\ &= {q_sym}"
                r"\end{aligned}",
                *lines,
            ),
        )
    if not above:
        lines.append(
            latex_chain(
                _m(lv),
                rf"\frac{{{q_sym}}}{{{_h(top, lv)} - {_h(2, lv)}}}",
                rf"\frac{{{_n(Q, 'power', system)}}}{{{_diff(h_top, h2, _EH, system)}}}",
                _q(lv.m_steam_kg_s, "mass_flow", system),
            )
        )
        text = (
            f"Los gases llegan al nivel a T_{a} y salen del evaporador a T_{p} = T_sat + pinch. "
            f"En ese tramo {devices}: todo el calor que ceden los gases lo recibe el agua "
            f"del nivel, desde que sale del economizador (2) hasta el estado {top}. El balance "
            "de energía del tramo (adiabático, sin trabajo; vademecum §3.3) da el caudal:"
        )
        balance = rf"{_m(lv)}\,({_h(top, lv)} - {_h(2, lv)}) = {q_sym}"
        return ProcedureStep(
            title=f"{_title(lv)}: caudal de vapor (gases {a} → {p})",
            text=text,
            latex=(balance, *lines),
        )
    m_up = sum(x.m_steam_kg_s for x in above)
    up_sym = _m(above[0]) if len(above) == 1 else rf"\dot{{m}}_{{\mathrm{{sube}}{_sub(lv)}}}"
    if len(above) > 1:
        lines.append(
            latex_chain(
                up_sym,
                " + ".join(_m(x) for x in above),
                _q(m_up, "mass_flow", system),
            )
        )
    Q_dome = m_up * (h3 - h2)
    dome_sym = rf"\dot{{Q}}_{{\mathrm{{domo}}{_sub(lv)}}}"
    lines += [
        latex_chain(
            dome_sym,
            rf"{up_sym}\,({_h(3, lv)} - {_h(2, lv)})",
            times_diff(m_up, "mass_flow", h3, h2, system),
            _q(Q_dome, "power", system),
        ),
        latex_chain(
            _m(lv),
            rf"\frac{{{q_sym} - {dome_sym}}}{{{_h(top, lv)} - {_h(2, lv)}}}",
            rf"\frac{{{_diff(Q, Q_dome, 'power', system)}}}{{{_diff(h_top, h2, _EH, system)}}}",
            _q(lv.m_steam_kg_s, "mass_flow", system),
        ),
    ]
    balance = (
        r"\begin{aligned}"
        rf"&{_m(lv)}\,({_h(top, lv)} - {_h(2, lv)}) \\ "
        rf"&\quad + {up_sym}\,({_h(3, lv)} - {_h(2, lv)}) \\ "
        rf"&= {q_sym}"
        r"\end{aligned}"
    )
    return ProcedureStep(
        title=f"{_title(lv)}: caudal de vapor (gases {a} → {p})",
        text=(
            f"Los gases llegan a T_{a} (después de los niveles de mayor presión) y salen del "
            f"evaporador a T_{p} = T_sat + pinch. En ese tramo {devices}; además, en el "
            "domo, el agua que sube hacia los niveles de mayor presión (sin evaporar) se "
            "calienta de la salida del economizador (2) a la saturación (3). Restado ese calor, "
            "el resto evapora el vapor del nivel:"
        ),
        latex=(balance, *lines),
    )


def _reheat_states_step(result: MultiHRSGResult, system: UnitSystem) -> ProcedureStep:
    """Estados del recalentador: el frío (lo da la turbina de alta) y el caliente."""
    rh = result.reheat
    assert rh is not None
    p = rh.reheater.p_Pa
    joins = rh.reheater.joins_middle
    lines = [
        rf"T_{{\mathrm{{RF}}}} = {_T(rh.cold.T_K, system)}",
        rf"h_{{\mathrm{{RF}}}} = {_q(rh.cold.h_J_per_kg, _EH, system)}",
        _lookup(
            r"h_{\mathrm{RC}}",
            r"h(p_{\mathrm{RH}},\ T_{\mathrm{RC}})",
            _q(rh.hot.h_J_per_kg, _EH, system),
        ),
    ]
    middle = (
        " Antes de entrar se le suma el vapor de media (a la misma presión): la mezcla depende "
        "de los dos caudales y se calcula después."
        if joins
        else ""
    )
    return ProcedureStep(
        title=f"Recalentador: estados del vapor ({_bar(p)})",
        text=(
            f"El recalentador recibe el vapor que vuelve de la turbina de alta a {_bar(p)} "
            "(recalentamiento frío, RF: su entalpía sale de la expansión en la turbina) y lo "
            f"calienta hasta {_degC(rh.hot.T_K)} (recalentamiento caliente, RC; tabla A-6). Va en "
            "paralelo con el sobrecalentador de alta: los dos bancos ven los gases desde la "
            "entrada y los dejan a la misma temperatura, así que comparten el tramo más "
            f"caliente de la caldera.{middle}"
        ),
        latex=tuple(lines),
    )


def _coupled_flow_step(result: MultiHRSGResult, system: UnitSystem) -> ProcedureStep:
    """Caudales de alta y de media con el vapor de media sumado al recalentamiento (2×2)."""
    inputs = result.inputs
    rh = result.reheat
    assert rh is not None
    hi, mid = result.levels[0], result.levels[1]
    waters = [hi.water, mid.water]
    sys_ = reheat_system(inputs, waters, rh.cold, rh.hot)
    labels = _gas_labels(result)
    a, c = labels[0][0], labels[0][-2]
    e = labels[1][-2]
    ta = 5 if hi.level.superheated else 4
    tm = 5 if mid.level.superheated else 4
    mA, mM = _m(hi), _m(mid)
    P: QuantityKind = "power"

    def coef(sym: str, expr: str, value: float, kind: QuantityKind = _EH) -> str:
        return latex_chain(sym, expr, _q(value, kind, system))

    lines = [
        r"\begin{aligned}"
        rf"&{mA}\,\bigl[({_h(ta, hi)} - {_h(2, hi)}) \\ "
        r"&\quad + (h_{\mathrm{RC}} - h_{\mathrm{RF}})\bigr] \\ "
        rf"&\quad + {mM}\,(h_{{\mathrm{{RC}}}} - {_h(tm, mid)}) \\ "
        rf"&= \dot{{m}}_g\,\bigl[h_g(T_{a}) - h_g(T_{c})\bigr]"
        r"\end{aligned}",
        r"\begin{aligned}"
        rf"&{mA}\,\bigl[({_h(3, mid)} - {_h(2, mid)}) \\ "
        rf"&\quad + ({_h(2, hi)} - {_h(1, hi)})\bigr] \\ "
        rf"&\quad + {mM}\,({_h(tm, mid)} - {_h(2, mid)}) \\ "
        rf"&= \dot{{m}}_g\,\bigl[h_g(T_{c}) - h_g(T_{e})\bigr]"
        r"\end{aligned}",
        rf"a_{{11}}\,{mA} + a_{{12}}\,{mM} = b_1",
        rf"a_{{21}}\,{mA} + a_{{22}}\,{mM} = b_2",
        coef(
            "a_{11}",
            rf"({_h(ta, hi)} - {_h(2, hi)}) + (h_{{\mathrm{{RC}}}} - h_{{\mathrm{{RF}}}})",
            sys_.a11,
        ),
        coef("a_{12}", rf"h_{{\mathrm{{RC}}}} - {_h(tm, mid)}", sys_.a12),
        coef("a_{21}", rf"({_h(3, mid)} - {_h(2, mid)}) + ({_h(2, hi)} - {_h(1, hi)})", sys_.a21),
        coef("a_{22}", rf"{_h(tm, mid)} - {_h(2, mid)}", sys_.a22),
        coef("b_1", rf"\dot{{m}}_g\,\bigl[h_g(T_{a}) - h_g(T_{c})\bigr]", sys_.b1, P),
        coef("b_2", rf"\dot{{m}}_g\,\bigl[h_g(T_{c}) - h_g(T_{e})\bigr]", sys_.b2, P),
        latex_chain(
            mA,
            r"\frac{b_1\,a_{22} - a_{12}\,b_2}{a_{11}\,a_{22} - a_{12}\,a_{21}}",
            _q(hi.m_steam_kg_s, "mass_flow", system),
        ),
        latex_chain(
            mM,
            r"\frac{a_{11}\,b_2 - a_{21}\,b_1}{a_{11}\,a_{22} - a_{12}\,a_{21}}",
            _q(mid.m_steam_kg_s, "mass_flow", system),
        ),
    ]
    return ProcedureStep(
        title=f"Niveles de alta y de media: caudales (gases {a} → {e})",
        text=(
            "El vapor de media se suma al recalentamiento: el recalentador calienta ṁ_A + ṁ_M, "
            "así que el calor del tramo de alta depende también de ṁ_M, y ṁ_M depende de cuánto "
            "enfrió la alta a los gases. Se plantean juntos dos balances: el de alta, de la "
            f"entrada de los gases a su pinch (T_{c}), y el de media, del pinch de alta al de "
            f"media (T_{e}), que incluye el economizador de alta (calienta ṁ_A) y el domo de media "
            "(lleva a T_sat el agua que sube a la alta). Es un sistema lineal en (ṁ_A, ṁ_M); por "
            "la regla de Cramer:"
        ),
        latex=tuple(lines),
    )


def _q_top(level: LevelResult, reheat: bool = False) -> str:
    """Símbolo del calor del tramo hasta el pinch: SH + EV, o EV, del nivel.

    Con recalentador (SH, RH y EV) es el «tramo» del nivel: el símbolo largo no
    entraba en el ancho de un celular.
    """
    if reheat:
        return rf"\dot{{Q}}_{{\mathrm{{tramo}}{_sub(level)}}}"
    what = r"\mathrm{SH+EV}" if level.level.superheated else r"\mathrm{EV}"
    return rf"\dot{{Q}}_{{{what}{_sub(level)}}}"


def _gas_out_lines(
    lv: LevelResult,
    labels: list[str],
    k: int,
    Q_W: float,
    m_g: float,
    symbol: str,
    system: UnitSystem,
) -> list[str]:
    """h_g de los gases a la salida de una sección (punto k del nivel) y su temperatura."""
    a, b = labels[k - 1], labels[k]
    return [
        latex_chain(
            f"h_g(T_{b})",
            rf"h_g(T_{a}) - \frac{{{symbol}}}{{\dot{{m}}_g}}",
            _wrap(
                _n(lv.h_gas_J_per_kg[k - 1], _EH, system),
                "-",
                rf"\frac{{{_n(Q_W, 'power', system)}}}{{{_n(m_g, 'mass_flow', system)}}}",
            ),
            _q(lv.h_gas_J_per_kg[k], _EH, system),
        ),
        rf"T_{b} = {_T(lv.T_gas_K[k], system)}",
    ]


def rh_joined_middle(result: MultiHRSGResult, i: int) -> bool:
    """¿El nivel i es la media que se suma al recalentamiento (sus caudales salen del 2×2)?"""
    rh = result.reheat
    return rh is not None and rh.reheater.joins_middle and i == 1


def _reheat_lines(result: MultiHRSGResult, system: UnitSystem) -> list[str]:
    """Calor del recalentador: la mezcla con la media (si se suma) y Q̇_RH."""
    rh = result.reheat
    assert rh is not None
    hi = result.levels[0]
    lines: list[str] = []
    h_in = rh.cold.h_J_per_kg
    inlet = r"h_{\mathrm{RF}}"
    flow = _m(hi)
    if rh.reheater.joins_middle:
        mid = result.levels[1]
        tm = 5 if mid.level.superheated else 4
        flow = r"\dot{m}_{\mathrm{RH}}"
        inlet = r"h_{\mathrm{mez}}"
        h_in = rh.inlet.h_J_per_kg
        lines += [
            latex_chain(
                flow,
                f"{_m(hi)} + {_m(mid)}",
                _q(rh.m_kg_s, "mass_flow", system),
            ),
            latex_chain(
                inlet,
                rf"\frac{{{_m(hi)}\,h_{{\mathrm{{RF}}}} + {_m(mid)}\,{_h(tm, mid)}}}{{{flow}}}",
                _q(h_in, _EH, system),
            ),
            rf"T_{{\mathrm{{mez}}}} = {_T(rh.inlet.T_K, system)}",
        ]
    lines.append(
        latex_chain(
            r"\dot{Q}_{\mathrm{RH}}",
            rf"{flow}\,(h_{{\mathrm{{RC}}}} - {inlet})",
            times_diff(rh.m_kg_s, "mass_flow", rh.hot.h_J_per_kg, h_in, system),
            _q(rh.Q_W, "power", system),
        )
    )
    return lines


def _sections_step(result: MultiHRSGResult, i: int, system: UnitSystem) -> ProcedureStep:
    """Sobrecalentador (si hay) y economizador del nivel i, con las temperaturas de los gases."""
    lv = result.levels[i]
    level = lv.level
    m_g = result.inputs.m_gas_kg_s
    labels = _gas_labels(result)[i]
    letter = _LETTERS[lv.name]
    sub = f",{letter}" if letter else ""
    w = lv.water
    h = [s.h_J_per_kg for s in w]
    lines: list[str] = []
    parts: list[str] = []
    rh = result.reheat if i == 0 else None
    Q_sh = 0.0
    if level.superheated:
        Q_sh = lv.m_steam_kg_s * (h[4] - h[3])
        symbol = rf"\dot{{Q}}_{{\mathrm{{SH}}{sub}}}"
        lines.append(
            latex_chain(
                symbol,
                rf"{_m(lv)}\,({_h(5, lv)} - {_h(4, lv)})",
                times_diff(lv.m_steam_kg_s, "mass_flow", h[4], h[3], system),
                _q(Q_sh, "power", system),
            )
        )
        if rh is None:
            lines += _gas_out_lines(lv, labels, 1, Q_sh, m_g, symbol, system)
            parts.append(f"el sobrecalentador baja los gases de T_{labels[0]} a T_{labels[1]}")
    Q_first = Q_sh
    if rh is not None:
        lines += _reheat_lines(result, system)
        Q_first = Q_sh + rh.Q_W
        if level.superheated:
            par = r"\dot{Q}_{\mathrm{par}}"
            lines.append(
                latex_chain(
                    par,
                    rf"\dot{{Q}}_{{\mathrm{{SH}}{sub}}} + \dot{{Q}}_{{\mathrm{{RH}}}}",
                    f"{_n(Q_sh, 'power', system)} + {_n(rh.Q_W, 'power', system)}",
                    _q(Q_first, "power", system),
                )
            )
            parts.append(
                f"el sobrecalentador y el recalentador, en paralelo, bajan los gases de "
                f"T_{labels[0]} a T_{labels[1]}"
            )
        else:
            par = r"\dot{Q}_{\mathrm{RH}}"
            parts.append(f"el recalentador baja los gases de T_{labels[0]} a T_{labels[1]}")
        lines += _gas_out_lines(lv, labels, 1, Q_first, m_g, par, system)
    if Q_first > 0.0:  # el evaporador (con el domo) es el resto del tramo hasta el pinch
        Q_top = m_g * (lv.h_gas_J_per_kg[0] - lv.h_gas_J_per_kg[-2])
        first = (
            r"\dot{Q}_{\mathrm{par}}"
            if rh is not None and level.superheated
            else (
                r"\dot{Q}_{\mathrm{RH}}" if rh is not None else rf"\dot{{Q}}_{{\mathrm{{SH}}{sub}}}"
            )
        )
        lines.append(
            latex_chain(
                rf"\dot{{Q}}_{{\mathrm{{EV}}{sub}}}",
                rf"{_q_top(lv, rh is not None)} - {first}",
                _diff(Q_top, Q_first, "power", system),
                _q(Q_top - Q_first, "power", system),
            )
        )
    if Q_first == 0.0 and rh_joined_middle(result, i):
        # con el 2×2 el evaporador de media no tuvo su propio paso: es el tramo hasta su pinch
        Q_top = m_g * (lv.h_gas_J_per_kg[0] - lv.h_gas_J_per_kg[-2])
        lines.append(
            latex_chain(
                rf"\dot{{Q}}_{{\mathrm{{EV}}{sub}}}",
                rf"\dot{{m}}_g\,\bigl[h_g(T_{labels[0]}) - h_g(T_{labels[-2]})\bigr]",
                _q(Q_top, "power", system),
            )
        )
    above = result.levels[:i]
    flow_sym = _m(lv)
    if above:
        flow_sym = rf"\dot{{m}}_{{\mathrm{{ECO}}{sub}}}"
        lines.append(
            latex_chain(
                flow_sym,
                " + ".join(_m(x) for x in (*above, lv)),
                _q(lv.m_water_kg_s, "mass_flow", system),
            )
        )
    Q_eco = lv.m_water_kg_s * (h[1] - h[0])
    symbol = rf"\dot{{Q}}_{{\mathrm{{ECO}}{sub}}}"
    k = len(lv.T_gas_K) - 1
    lines += [
        latex_chain(
            symbol,
            rf"{flow_sym}\,({_h(2, lv)} - {_h(1, lv)})",
            times_diff(lv.m_water_kg_s, "mass_flow", h[1], h[0], system),
            _q(Q_eco, "power", system),
        ),
        *_gas_out_lines(lv, labels, k, Q_eco, m_g, symbol, system),
    ]
    last = i == len(result.levels) - 1
    goes = "por la chimenea" if last else f"hacia el nivel de {result.levels[i + 1].name}"
    whose = "el agua de todos los niveles" if above else "el agua del nivel"
    parts.append(
        f"el economizador calienta {whose} hasta T_sat − approach y los gases salen a "
        f"T_{labels[k]}, {goes}"
    )
    devices = ["sobrecalentador"] if level.superheated else []
    if rh is not None:
        devices.append("recalentador")
    devices.append("economizador")
    return ProcedureStep(
        title=f"{_title(lv)}: "
        + (", ".join(devices[:-1]) + " y " if devices[:-1] else "")
        + devices[-1],
        text=(
            "Con el caudal, cada sección da la temperatura de los gases a su salida (como h_g "
            "crece con T, se busca la T que da esa entalpía): " + "; ".join(parts) + "."
        ),
        latex=tuple(lines),
    )


def _total_step(result: MultiHRSGResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    flows = [_m(lv) for lv in result.levels]
    m_values = [_n(lv.m_steam_kg_s, "mass_flow", system) for lv in result.levels]
    h_ref = inputs.gas.h(inputs.T_ref_K)
    last = result.gas_points[-1][0]
    return ProcedureStep(
        title="Vapor total, calor y aprovechamiento",
        text=(
            "El calor que reciben el agua y el vapor es el que ceden los gases entre la entrada y "
            "la chimenea (sin pérdidas al ambiente). El aprovechamiento lo compara con el que "
            f"cederían si se enfriaran hasta {_degC(inputs.T_ref_K)}: lo que falta se va por la "
            "chimenea. Las bombas entre niveles agregan su trabajo aparte."
        ),
        latex=(
            latex_chain(
                r"\dot{m}_{v,\mathrm{total}}",
                " + ".join(flows),
                " + ".join(m_values),
                _q(result.m_steam_kg_s, "mass_flow", system),
            ),
            latex_chain(
                r"\dot{Q}",
                rf"\dot{{m}}_g\,\bigl[h_g(T_a) - h_g(T_{last})\bigr]",
                _q(result.Q_W, "power", system),
            ),
            rf"h_g(T_{{\mathrm{{ref}}}}) = {_q(h_ref, _EH, system)}",
            latex_chain(
                r"\eta_{\mathrm{rec}}",
                r"\frac{\dot{Q}}{\dot{m}_g\,\bigl[h_g(T_a) - h_g(T_{\mathrm{ref}})\bigr]}",
                rf"{latex_number(result.recovery * 100, 4)}\,\%",
            ),
            latex_chain(r"\dot{W}_{B}", _q(result.W_pumps_W, "power", system)),
        ),
    )


def exergy_step(result: MultiHRSGResult, system: UnitSystem) -> ProcedureStep:
    """Balance de exergía de la caldera: gases, agua, destruida y chimenea (Cengel cap. 8)."""
    x = hrsg_exergy(result)
    T0 = _T(x.T0_K, system)
    last = result.gas_points[-1][0]
    return ProcedureStep(
        title="Exergía: cuánto vale el calor recuperado",
        text=(
            f"Con el ambiente a T₀ = {_degC(x.T0_K)}, la exergía de los gases es lo máximo que "
            "se podría convertir en trabajo al enfriarlos hasta T₀. Una parte la gana el agua "
            "(el vapor que después mueve la turbina), otra se destruye porque el calor pasa con "
            "diferencia de temperatura entre los gases y el agua (X_dest = T₀·S_gen) y otra se "
            "va por la chimenea. Más niveles de presión acercan la curva del agua a la de los "
            "gases: menos exergía destruida y menos por la chimenea."
        ),
        latex=(
            r"\begin{aligned}x_g(T) &= \bigl[h_g(T) - h_g(T_0)\bigr] \\ &\quad"
            r" - T_0\,\bigl[s^{\circ}(T) - s^{\circ}(T_0)\bigr]\end{aligned}",
            rf"T_0 = {T0}",
            latex_chain(
                r"\dot{X}_{\mathrm{gases}}",
                r"\dot{m}_g\,x_g(T_a)",
                _q(x.X_gas_in_W, "power", system),
            ),
            latex_chain(
                r"\dot{X}_{\mathrm{chim}}",
                rf"\dot{{m}}_g\,x_g(T_{last})",
                _q(x.X_stack_W, "power", system),
            ),
            latex_chain(
                r"\dot{X}_{\mathrm{agua}}",
                r"\sum \dot{m}\,\bigl[\Delta h - T_0\,\Delta s\bigr]",
                _q(x.X_water_W, "power", system),
            ),
            latex_chain(
                r"\dot{X}_{\mathrm{dest}}",
                r"T_0\,\dot{S}_{\mathrm{gen}}",
                _q(x.X_destroyed_W, "power", system),
            ),
            latex_chain(
                r"\eta_{\mathrm{II}}",
                r"\frac{\dot{X}_{\mathrm{agua}}}{\dot{X}_{\mathrm{gases}}}",
                rf"{latex_number(x.efficiency * 100, 4)}\,\%",
            ),
        ),
    )


def multi_hrsg_steps(result: MultiHRSGResult, system: UnitSystem) -> list[ProcedureStep]:
    """Procedimiento completo de la HRSG de varias presiones, como se resuelve a mano."""
    steps = [gas_composition_step(result.inputs.gas, system), _gas_enthalpy_step(result, system)]
    rh = result.reheat
    start = 0
    if rh is not None and rh.reheater.joins_middle:
        steps += [
            _water_step(result, 0, system),
            _reheat_states_step(result, system),
            _water_step(result, 1, system),
            _coupled_flow_step(result, system),
            _sections_step(result, 0, system),
            _sections_step(result, 1, system),
        ]
        start = 2
    for i in range(start, len(result.levels)):
        steps.append(_water_step(result, i, system))
        if i == 0 and rh is not None:
            steps.append(_reheat_states_step(result, system))
        steps += [_flow_step(result, i, system), _sections_step(result, i, system)]
    steps += [_total_step(result, system), exergy_step(result, system)]
    dew = dew_point_step(result.inputs.gas, system)
    if dew is not None:
        steps.append(dew)
    return steps
