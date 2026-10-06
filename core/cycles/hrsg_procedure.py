"""Procedimiento de la HRSG de una presión «como a mano» — Fase 3.3.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo: la composición de los gases (vademecum §5), su entalpía como
mezcla de gases ideales, los estados del agua (tablas de vapor, Cengel A-4 a
A-6), el caudal de vapor del balance desde la entrada de los gases hasta el
pinch, cada sección (sobrecalentador, evaporador y economizador), el calor
total y el aprovechamiento, y el punto de rocío de los gases. Recibe un
resultado ya calculado (:class:`core.cycles.hrsg.HRSGResult`): no importa
Streamlit. Las ecuaciones se arman con :func:`core.latex.latex_chain` (una
igualdad por renglón) para que entren en el ancho de un celular.
"""

from __future__ import annotations

from core.cycles.hrsg import R_U, HRSGResult, molar_mass
from core.cycles.rankine_procedure import _bar, _diff, _n, _q, _wrap
from core.fluids import saturation_at_temperature
from core.latex import latex_chain, latex_number, latex_paren
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem

_EH: QuantityKind = "specific_enthalpy"
_TEX_SPECIES = {
    "N2": r"\mathrm{N_2}",
    "O2": r"\mathrm{O_2}",
    "CO2": r"\mathrm{CO_2}",
    "H2O": r"\mathrm{H_2O}",
    "Ar": r"\mathrm{Ar}",
}


def _T(value_K: float, system: UnitSystem) -> str:
    return _q(value_K, "temperature", system)


def _times_diff(
    factor_si: float, factor_kind: QuantityKind, a_si: float, b_si: float, system: UnitSystem
) -> str:
    r"""``f\,(a - b)`` con entalpías; con números ×10ⁿ (SI), el ``- b`` va en otro renglón.

    Pensado para un paso de :func:`~core.latex.latex_chain` (ancho de un celular).
    """
    f = _n(factor_si, factor_kind, system)
    diff = _diff(a_si, b_si, _EH, system)
    if r"\times" not in diff:
        return rf"{f}\,({diff})"
    b = latex_paren(_n(b_si, _EH, system))
    return rf"{f}\,({_n(a_si, _EH, system)} \\ &\quad - {b})"


def _molar_unit(system: UnitSystem) -> str:
    return r"\mathrm{lb/lbmol}" if system == "Inglés" else r"\mathrm{kg/kmol}"


def _composition_step(result: HRSGResult, system: UnitSystem) -> ProcedureStep:
    """Masa molar, fracciones másicas y R de la mezcla (vademecum §5)."""
    gas = result.inputs.gas
    english = system == "Inglés"
    present = [(s, y) for s, y in gas.mole_fractions.items() if y > 0.0]
    M = gas.M_kg_per_mol * 1e3
    rows = [r"M &= \sum_i y_i\,M_i"]
    for k, (s, y) in enumerate(present):
        term = rf"{latex_number(y, 5)}\cdot {latex_number(molar_mass(s) * 1e3, 5)}"
        rows.append(rf"&= {term}" if k == 0 else rf"&\quad + {term}")
    rows.append(rf"&= {latex_number(M, 5)}\ {_molar_unit(system)}")
    lines = [r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"]
    w = gas.mass_fractions
    for s, y in present:
        lines.append(
            latex_chain(
                f"w_{{{_TEX_SPECIES[s]}}}",
                r"\frac{y_i\,M_i}{M}",
                rf"\frac{{{latex_number(y, 5)}\cdot {latex_number(molar_mass(s) * 1e3, 5)}}}"
                rf"{{{latex_number(M, 5)}}}",
                latex_number(w[s], 5),
            )
        )
    R = gas.R_J_per_kg_K
    lines.append(
        latex_chain(
            "R",
            r"\frac{R_u}{M}",
            rf"\frac{{{latex_number(R_U, 5)}}}{{{latex_number(M, 5)}}}",
            _q(R, "specific_heat", system),
        )
    )
    return ProcedureStep(
        title="Composición de los gases",
        text=(
            "Los gases son una mezcla de gases ideales (vademecum §5.1 y §5.2). Con las fracciones "
            "molares yᵢ salen la masa molar de la mezcla, las fracciones másicas wᵢ (las que "
            "pesan en la entalpía por kilogramo) y la constante particular R."
            + (
                " Las masas molares en lb/lbmol son el mismo número que en kg/kmol."
                if english
                else ""
            )
        ),
        latex=tuple(lines),
    )


def _gas_enthalpy_step(result: HRSGResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    labels = result.gas_labels
    lines = [r"h_g(T) = \sum_i w_i\,\bigl[h_i(T) - h_i(25\ {}^{\circ}\mathrm{C})\bigr]"]
    lines.append(rf"h_g(T_a) = {_q(result.h_gas_J_per_kg[0], _EH, system)}")
    c = labels.index("c")
    lines.append(
        latex_chain(
            "T_c",
            r"T_{\mathrm{sat}} + \Delta T_{\mathrm{pinch}}",
            rf"{_n(result.T_sat_K, 'temperature', system)} + "
            rf"{_n(inputs.pinch_K, 'temperature_difference', system)}",
            _T(result.T_gas_K[c], system),
        )
    )
    lines.append(rf"h_g(T_c) = {_q(result.h_gas_J_per_kg[c], _EH, system)}")
    return ProcedureStep(
        title="Entalpía de los gases",
        text=(
            f"Los gases entran a T_a = {_T_text(inputs.T_gas_in_K)} y salen del evaporador a "
            "T_c = T_sat + pinch. En un gas ideal la entalpía depende solo de T: la de cada "
            "componente sale del gas ideal de CoolProp (lo mismo que integrar los polinomios "
            "NASA del vademecum §4.8) y la de la mezcla es la suma pesada con las fracciones "
            "másicas (vademecum §5.6), medida desde 25 °C."
        ),
        latex=tuple(lines),
    )


def _T_text(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _water_step(result: HRSGResult, system: UnitSystem) -> ProcedureStep:
    """Estados del agua: alimentación, salida del economizador, saturados y vapor."""
    inputs = result.inputs
    w = result.water
    h_f1 = saturation_at_temperature("Water", w[0].T_K).liquid.h_J_per_kg
    lines = [
        rf"T_{{\mathrm{{sat}}}} = T_{{\mathrm{{sat}}}}(p) = {_T(result.T_sat_K, system)}",
        rf"h_1 = h(p,\ T_1) = {_q(w[0].h_J_per_kg, _EH, system)}",
        rf"h_f(T_1) = {_q(h_f1, _EH, system)}",
    ]
    if inputs.approach_K > 0.0:
        h_f2 = saturation_at_temperature("Water", w[1].T_K).liquid.h_J_per_kg
        lines.append(
            latex_chain(
                "T_2",
                r"T_{\mathrm{sat}} - \Delta T_{\mathrm{approach}}",
                rf"{_n(result.T_sat_K, 'temperature', system)} - "
                rf"{_n(inputs.approach_K, 'temperature_difference', system)}",
                _T(w[1].T_K, system),
            )
        )
        lines.append(rf"h_2 = h(p,\ T_2) = {_q(w[1].h_J_per_kg, _EH, system)}")
        lines.append(rf"h_f(T_2) = {_q(h_f2, _EH, system)}")
    else:
        lines.append(rf"h_2 = h_f(p) = {_q(w[1].h_J_per_kg, _EH, system)}")
    lines.append(rf"h_3 = h_f(p) = {_q(w[2].h_J_per_kg, _EH, system)}")
    lines.append(rf"h_4 = h_g(p) = {_q(w[3].h_J_per_kg, _EH, system)}")
    if inputs.superheated:
        lines.append(rf"h_5 = h(p,\ T_5) = {_q(w[4].h_J_per_kg, _EH, system)}")
    return ProcedureStep(
        title="Estados del agua y del vapor",
        text=(
            f"Todo a la presión de evaporación, {_bar(inputs.p_steam_Pa)} (sin pérdidas de "
            "carga). T_sat, h_f y h_g se leen en la tabla de saturación por presión (Cengel "
            "A-5). El agua de alimentación y la que sale del economizador son líquido "
            "comprimido: h(p, T) sale de la tabla de líquido comprimido (Cengel A-7) y, "
            "aproximando, h ≈ h_f(T) (A-4, vademecum §13)"
            + ("; el vapor sobrecalentado, de la tabla A-6." if inputs.superheated else ".")
            + " El economizador entrega el agua T_sat − approach por debajo de la saturación, "
            "para que no evapore adentro."
        ),
        latex=tuple(lines),
    )


def _steam_flow_step(result: HRSGResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    top = 5 if inputs.superheated else 4
    w = result.water
    c = result.gas_labels.index("c")
    h_a, h_c = result.h_gas_J_per_kg[0], result.h_gas_J_per_kg[c]
    Q_top = inputs.m_gas_kg_s * (h_a - h_c)
    q_sym = r"\dot{Q}_{\mathrm{SH+EV}}" if inputs.superheated else r"\dot{Q}_{\mathrm{EV}}"
    devices = "el sobrecalentador y el evaporador" if inputs.superheated else "el evaporador"
    return ProcedureStep(
        title="Caudal de vapor",
        text=(
            f"Entre la entrada de los gases (a) y la salida del evaporador (c) están {devices}: "
            "todo el calor que ceden ahí los gases lo recibe el agua, desde que sale del "
            f"economizador (2) hasta el estado {top}. El balance de energía de ese tramo "
            "(adiabático, sin trabajo; vademecum §3.3) da el caudal de vapor:"
        ),
        latex=(
            rf"\dot{{m}}_v\,(h_{top} - h_2) = \dot{{m}}_g\,\bigl[h_g(T_a) - h_g(T_c)\bigr]",
            latex_chain(
                q_sym,
                r"\dot{m}_g\,\bigl[h_g(T_a) - h_g(T_c)\bigr]",
                _times_diff(inputs.m_gas_kg_s, "mass_flow", h_a, h_c, system),
                _q(Q_top, "power", system),
            ),
            latex_chain(
                r"\dot{m}_v",
                rf"\frac{{{q_sym}}}{{h_{top} - h_2}}",
                rf"\frac{{{_n(Q_top, 'power', system)}}}"
                rf"{{{_diff(w[-1].h_J_per_kg, w[1].h_J_per_kg, _EH, system)}}}",
                _q(result.m_steam_kg_s, "mass_flow", system),
            ),
        ),
    )


def _gas_outlet_lines(
    result: HRSGResult, system: UnitSystem, i_in: int, i_out: int, Q_W: float, Q_symbol: str
) -> list[str]:
    """h_g(T_sal) = h_g(T_ent) − Q̇/ṁ_g y la T que le corresponde."""
    m_g = result.inputs.m_gas_kg_s
    a, b = result.gas_labels[i_in], result.gas_labels[i_out]
    return [
        latex_chain(
            f"h_g(T_{b})",
            rf"h_g(T_{a}) - \frac{{{Q_symbol}}}{{\dot{{m}}_g}}",
            _wrap(
                _n(result.h_gas_J_per_kg[i_in], _EH, system),
                "-",
                rf"\frac{{{_n(Q_W, 'power', system)}}}{{{_n(m_g, 'mass_flow', system)}}}",
            ),
            _q(result.h_gas_J_per_kg[i_out], _EH, system),
        ),
        rf"T_{b} = {_T(result.T_gas_K[i_out], system)}",
    ]


def _section_steps(result: HRSGResult, system: UnitSystem) -> list[ProcedureStep]:
    w = result.water
    h = [s.h_J_per_kg for s in w]
    m_s = result.m_steam_kg_s
    sections = {s.name: s for s in result.sections}
    steps: list[ProcedureStep] = []
    if result.inputs.superheated:
        sh = sections["sobrecalentador"]
        steps.append(
            ProcedureStep(
                title="Sobrecalentador (gases a → b; vapor 4 → 5)",
                text=(
                    "El vapor saturado que sale del domo se sobrecalienta; los gases que lo "
                    "calientan bajan de T_a a T_b, que sale de despejar su entalpía (como h_g "
                    "crece con T, se busca la T que la da):"
                ),
                latex=(
                    latex_chain(
                        r"\dot{Q}_{\mathrm{SH}}",
                        r"\dot{m}_v\,(h_5 - h_4)",
                        _times_diff(m_s, "mass_flow", h[4], h[3], system),
                        _q(sh.Q_W, "power", system),
                    ),
                    *_gas_outlet_lines(result, system, 0, 1, sh.Q_W, r"\dot{Q}_{\mathrm{SH}}"),
                ),
            )
        )
    ev = sections["evaporador"]
    c = result.gas_labels.index("c")
    steps.append(
        ProcedureStep(
            title=f"Evaporador (gases {result.gas_labels[c - 1]} → c; agua 2 → 4)",
            text=(
                "El agua del economizador entra al domo, se mezcla con la saturada que circula "
                "por el evaporador y sale como vapor saturado. Los gases se enfrían hasta "
                "T_c = T_sat + pinch: el pinch es la menor diferencia de temperatura de la "
                "caldera."
            ),
            latex=(
                latex_chain(
                    r"\dot{Q}_{\mathrm{EV}}",
                    r"\dot{m}_v\,(h_4 - h_2)",
                    _times_diff(m_s, "mass_flow", h[3], h[1], system),
                    _q(ev.Q_W, "power", system),
                ),
            ),
        )
    )
    eco = sections["economizador"]
    steps.append(
        ProcedureStep(
            title="Economizador (gases c → d; agua 1 → 2)",
            text=(
                "El agua de alimentación se calienta hasta T_sat − approach con lo que les queda "
                "a los gases, que salen por la chimenea a T_d:"
            ),
            latex=(
                latex_chain(
                    r"\dot{Q}_{\mathrm{ECO}}",
                    r"\dot{m}_v\,(h_2 - h_1)",
                    _times_diff(m_s, "mass_flow", h[1], h[0], system),
                    _q(eco.Q_W, "power", system),
                ),
                *_gas_outlet_lines(
                    result, system, c, len(result.T_gas_K) - 1, eco.Q_W, r"\dot{Q}_{\mathrm{ECO}}"
                ),
            ),
        )
    )
    return steps


def _total_step(result: HRSGResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    names = [s.name for s in result.sections]
    symbols = {
        "sobrecalentador": r"\dot{Q}_{\mathrm{SH}}",
        "evaporador": r"\dot{Q}_{\mathrm{EV}}",
        "economizador": r"\dot{Q}_{\mathrm{ECO}}",
    }
    total = " + ".join(symbols[n] for n in names)
    values = [_n(s.Q_W, "power", system) for s in result.sections]
    rows = [rf"\dot{{Q}} &= {total}"]
    if any(r"\times" in v for v in values):  # ×10ⁿ: un sumando por renglón
        rows.append(rf"&= {values[0]}")
        rows += [rf"&\quad + {v}" for v in values[1:]]
    else:
        rows.append(rf"&= {' + '.join(values)}")
    rows.append(rf"&= {_q(result.Q_W, 'power', system)}")
    h_ref = inputs.gas.h(inputs.T_ref_K)
    return ProcedureStep(
        title="Calor total y aprovechamiento",
        text=(
            "El calor que recibe el agua es el que ceden los gases entre la entrada y la "
            f"chimenea (sin pérdidas al ambiente). El aprovechamiento lo compara con el que "
            f"cederían si se enfriaran hasta {_T_text(inputs.T_ref_K)} (la temperatura de "
            "referencia de las normas ISO): lo que falta se va por la chimenea."
        ),
        latex=(
            r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}",
            latex_chain(
                r"\dot{Q}",
                r"\dot{m}_g\,\bigl[h_g(T_a) - h_g(T_d)\bigr]",
                _q(result.Q_gas_W, "power", system),
            ),
            rf"h_g(T_{{\mathrm{{ref}}}}) = {_q(h_ref, _EH, system)}",
            latex_chain(
                r"\eta_{\mathrm{rec}}",
                r"\frac{\dot{Q}}{\dot{m}_g\,\bigl[h_g(T_a) - h_g(T_{\mathrm{ref}})\bigr]}",
                rf"{latex_number(result.recovery * 100, 4)}\,\%",
            ),
        ),
    )


def _dew_point_step(result: HRSGResult, system: UnitSystem) -> ProcedureStep | None:
    gas = result.inputs.gas
    dew = result.dew_point_K
    if dew is None:
        return None
    y_w = gas.mole_fractions["H2O"]
    return ProcedureStep(
        title="Punto de rocío de los gases",
        text=(
            "El vapor de agua de los gases condensa si se enfrían por debajo de la temperatura "
            "de saturación a su presión parcial (ley de Dalton, vademecum §5.3). La chimenea "
            "tiene que quedar por encima, y conviene que el agua de alimentación también, "
            "porque los tubos del economizador están a esa temperatura."
        ),
        latex=(
            latex_chain(
                r"p_{\mathrm{H_2O}}",
                r"y_{\mathrm{H_2O}}\,p",
                rf"{latex_number(y_w, 5)}\cdot {_n(gas.p_Pa, 'pressure', system)}",
                _q(gas.p_H2O_Pa, "pressure", system),
            ),
            latex_chain(
                r"T_{\text{rocío}}",
                r"T_{\mathrm{sat}}(p_{\mathrm{H_2O}})",
                _T(dew, system),
            ),
        ),
    )


def hrsg_steps(result: HRSGResult, system: UnitSystem) -> list[ProcedureStep]:
    """Procedimiento completo de la HRSG, como se resuelve a mano."""
    steps = [
        _composition_step(result, system),
        _gas_enthalpy_step(result, system),
        _water_step(result, system),
        _steam_flow_step(result, system),
        *_section_steps(result, system),
        _total_step(result, system),
    ]
    dew = _dew_point_step(result, system)
    if dew is not None:
        steps.append(dew)
    return steps
