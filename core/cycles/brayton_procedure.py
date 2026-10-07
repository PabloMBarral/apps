"""Procedimiento de la turbina de gas «como con las tablas» — Fase 3.4.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo: la composición del aire (vademecum §5), el compresor con la
función s° de gas ideal (como la tabla A-17 de Cengel) y su rendimiento
isoentrópico (vademecum §10.4), la cámara de combustión (estequiometría,
vademecum §16, y el balance con el PCI de ISO 6976:2016) o el calentamiento
del aire estándar, la turbina, el trabajo neto y el rendimiento (Cengel
§9-8). Recibe un resultado ya calculado
(:class:`core.cycles.brayton.BraytonResult`): no importa Streamlit.
"""

from __future__ import annotations

import math

from core.cycles.brayton import BraytonResult
from core.cycles.hrsg_procedure import factor_times_diff, gas_composition_lines
from core.cycles.rankine_procedure import _diff, _n, _q, _wrap
from core.ideal_gas import GAS_NAMES, GAS_SPECIES
from core.latex import latex_chain, latex_number
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem

_EH: QuantityKind = "specific_enthalpy"
_ES: QuantityKind = "specific_entropy"


def _T(value_K: float, system: UnitSystem) -> str:
    return _q(value_K, "temperature", system)


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _formula(a: float, b: float, c: float, d: float) -> str:
    """Fórmula media del combustible: CH₄, o C₁.₀₅H₄.₁ para una mezcla."""
    parts = []
    for element, n in (("C", a), ("H", b), ("N", c), ("O", d)):
        if n <= 0.0:
            continue
        if math.isclose(n, round(n)):
            count = "" if round(n) == 1 else str(round(n)).translate(_SUBSCRIPTS)
        else:
            count = f"{n:.3g}".translate(_SUBSCRIPTS)
        parts.append(element + count)
    return "".join(parts)


def _composition(result: BraytonResult) -> str:
    y = result.gas.mole_fractions
    return ", ".join(f"{GAS_NAMES[s]} {y[s] * 100:.3g} %" for s in GAS_SPECIES if y[s] > 0.0)


def _air_step(result: BraytonResult, system: UnitSystem) -> ProcedureStep:
    air = result.inputs.air
    return ProcedureStep(
        title="Composición del aire",
        text=(
            "El aire es una mezcla de gases ideales (vademecum §5.1 y §5.2): con las fracciones "
            "molares salen la masa molar, las fracciones másicas y la constante particular R. "
            "Las entalpías se miden desde 25 °C, la referencia del poder calorífico."
        ),
        latex=tuple(gas_composition_lines(air, system)),
    )


def _compressor_step(result: BraytonResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    air = inputs.air
    s1, s2s, s2 = result.state("1"), result.state("2s"), result.state("2")
    R = air.R_J_per_kg_K
    rp = inputs.pressure_ratio
    lines = [
        latex_chain(
            "p_2",
            r"r_p\,p_1",
            rf"{latex_number(rp, 5)}\cdot {_n(s1.P_Pa, 'pressure', system)}",
            _q(s2.P_Pa, "pressure", system),
        ),
        latex_chain(r"s^{\circ}_1", r"s^{\circ}(T_1)", _q(air.s0(s1.T_K), _ES, system)),
        latex_chain(
            r"s^{\circ}_{2s}",
            r"s^{\circ}_1 + R\,\ln\frac{p_2}{p_1}",
            rf"{_n(air.s0(s1.T_K), _ES, system)} + "
            rf"{_n(R, 'specific_heat', system)}\cdot \ln {latex_number(rp, 5)}",
            _q(air.s0(s2s.T_K), _ES, system),
        ),
        rf"T_{{2s}} = {_T(s2s.T_K, system)}",
        rf"h_1 = {_q(s1.h_J_per_kg, _EH, system)}",
        rf"h_{{2s}} = {_q(s2s.h_J_per_kg, _EH, system)}",
    ]
    if inputs.eta_compressor < 1.0:
        lines.append(
            latex_chain(
                "h_2",
                r"h_1 + \frac{h_{2s} - h_1}{\eta_C}",
                _wrap(
                    _n(s1.h_J_per_kg, _EH, system),
                    "+",
                    rf"\frac{{{_diff(s2s.h_J_per_kg, s1.h_J_per_kg, _EH, system)}}}"
                    rf"{{{latex_number(inputs.eta_compressor, 4)}}}",
                ),
                _q(s2.h_J_per_kg, _EH, system),
            )
        )
        lines.append(rf"T_2 = {_T(s2.T_K, system)}")
    lines.append(
        latex_chain(
            "w_C",
            "h_2 - h_1",
            _diff(s2.h_J_per_kg, s1.h_J_per_kg, _EH, system),
            _q(result.w_compressor_J_per_kg, _EH, system),
        )
    )
    ideal = inputs.eta_compressor == 1.0
    return ProcedureStep(
        title="Compresor (1 → 2)",
        text=(
            f"El aire entra a T₁ = {_degC(s1.T_K)} y sale a p₂ = r_p·p₁. Con calores específicos "
            "variables, la compresión isoentrópica cumple s°₂s = s°₁ + R·ln(p₂/p₁), con "
            "s° = s°(T) la función de la temperatura de la columna s° de la tabla A-17 de Cengel "
            "(acá medida desde 25 °C, Cengel §7-9): se busca la temperatura T₂s que da ese s°."
            + (
                " El compresor es ideal: 2 = 2s."
                if ideal
                else " El rendimiento isoentrópico da la salida real (vademecum §10.4)."
            )
        ),
        latex=tuple(lines),
    )


def _combustion_step(result: BraytonResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    fuel = inputs.fuel
    assert fuel is not None
    air = inputs.air
    s2, s3 = result.state("2"), result.state("3")
    a, b, c, d = fuel.atoms
    o2 = a + b / 4.0 - d / 2.0
    y_o2 = air.mole_fractions["O2"]
    n_air_t = o2 / y_o2
    afr_t = n_air_t * air.M_kg_per_mol / fuel.M_kg_per_mol
    f_t = 1.0 / afr_t
    f = result.fuel_air_ratio
    lhv = fuel.lhv_J_per_kg
    hhv_m = fuel.hhv_molar_J_per_mol / 1e3
    lhv_m = fuel.lhv_molar_J_per_mol / 1e3
    L0 = (hhv_m - lhv_m) / (b / 2.0)
    lines = [
        latex_chain(
            r"\mathrm{PCI}_m",
            r"\mathrm{PCS}_m - \tfrac{b}{2}\,L_0",
            rf"{latex_number(hhv_m, 5)} - {latex_number(b / 2.0, 4)}\cdot {latex_number(L0, 5)}",
            rf"{latex_number(lhv_m, 5)}\ \mathrm{{kJ/mol}}",
        ),
        latex_chain(r"\mathrm{PCI}", r"\frac{\mathrm{PCI}_m}{M_f}", _q(lhv, _EH, system)),
        latex_chain(
            r"n_{a,t}",
            r"\frac{a + b/4 - d/2}{y_{\mathrm{O_2}}}",
            rf"\frac{{{latex_number(o2, 5)}}}{{{latex_number(y_o2, 5)}}}",
            rf"{latex_number(n_air_t, 5)}\ \mathrm{{mol/mol}}",
        ),
        latex_chain(
            "f_t",
            r"\frac{M_f}{n_{a,t}\,M_a}",
            rf"\frac{{{latex_number(fuel.M_kg_per_mol * 1e3, 5)}}}"
            rf"{{{latex_number(n_air_t, 5)}\cdot {latex_number(air.M_kg_per_mol * 1e3, 5)}}}",
            latex_number(f_t, 5),
        ),
        r"h_a(T_2) + f\,\mathrm{PCI} = (1 + f)\,h_g(T_3)",
        rf"h_g(T_3) = {_q(s3.h_J_per_kg, _EH, system)}",
        latex_chain(
            "f",
            r"\frac{h_g(T_3) - h_a(T_2)}{\mathrm{PCI} - h_g(T_3)}",
            rf"\frac{{{_diff(s3.h_J_per_kg, s2.h_J_per_kg, _EH, system)}}}"
            rf"{{{_diff(lhv, s3.h_J_per_kg, _EH, system)}}}",
            latex_number(f, 5),
        ),
        latex_chain(
            r"\lambda",
            r"\frac{f_t}{f}",
            rf"\frac{{{latex_number(f_t, 5)}}}{{{latex_number(f, 5)}}}",
            latex_number(result.excess_air or math.nan, 4),
        ),
    ]
    name = "metano" if fuel.name == "methane" else fuel.name
    formula = _formula(a, b, c, d)
    return ProcedureStep(
        title="Cámara de combustión (2 → 3)",
        text=(
            f"El combustible ({name}, {formula} por mol) entra a 25 °C. Su poder calorífico "
            "inferior sale de ISO 6976:2016 a 25 °C: el superior menos el calor latente del agua "
            "que se forma (la misma tabla de la página ISO 6976). La combustión completa pide "
            "a + b/4 − d/2 moles de O₂ por mol de combustible (vademecum §16.2): eso da el aire "
            "teórico y la relación combustible/aire teórica f_t. Con las entalpías medidas desde "
            "25 °C, el balance de la cámara adiabática, por kilogramo de aire, da la relación "
            "combustible/aire f; como h_g(T₃) depende de la composición de los gases, que "
            "depende de f, se resuelve iterando (estos son los valores finales). Los gases "
            f"salen con {_composition(result)}."
        ),
        latex=tuple(lines),
    )


def _heater_step(result: BraytonResult, system: UnitSystem) -> ProcedureStep:
    s2, s3 = result.state("2"), result.state("3")
    return ProcedureStep(
        title="Calentamiento (2 → 3), aire estándar",
        text=(
            "Con el modelo de aire estándar (Cengel §9-3) la cámara de combustión se reemplaza "
            "por un intercambiador: el aire recibe calor a presión constante hasta T₃."
        ),
        latex=(
            rf"h_3 = {_q(s3.h_J_per_kg, _EH, system)}",
            latex_chain(
                r"q_{\mathrm{ent}}",
                "h_3 - h_2",
                _diff(s3.h_J_per_kg, s2.h_J_per_kg, _EH, system),
                _q(result.q_in_J_per_kg, _EH, system),
            ),
        ),
    )


def _gases_step(result: BraytonResult, system: UnitSystem) -> ProcedureStep:
    return ProcedureStep(
        title="Composición de los gases de combustión",
        text=(
            "Los gases que salen de la cámara (y van a la turbina y a la caldera de recuperación) "
            "son otra mezcla de gases ideales: su masa molar y su R cambian respecto del aire."
        ),
        latex=tuple(gas_composition_lines(result.gas, system)),
    )


def _turbine_step(result: BraytonResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    gas = result.gas
    s2, s3, s4s, s4 = result.state("2"), result.state("3"), result.state("4s"), result.state("4")
    R = gas.R_J_per_kg_K
    ratio = s3.P_Pa / s4.P_Pa
    g = "_g" if not inputs.air_standard else ""

    def s0(label: str) -> str:  # s°₃ del aire o s°_g,₃ de los gases
        return rf"s^{{\circ}}_{{g,{label}}}" if g else rf"s^{{\circ}}_{{{label}}}"

    lines = []
    if inputs.dp_combustor > 0.0:
        lines.append(
            latex_chain(
                "p_3",
                r"p_2\,(1 - \Delta p)",
                rf"{_n(s2.P_Pa, 'pressure', system)}\cdot (1 - "
                rf"{latex_number(inputs.dp_combustor, 3)})",
                _q(s3.P_Pa, "pressure", system),
            )
        )
    lines += [
        latex_chain(s0("3"), rf"s^{{\circ}}{g}(T_3)", _q(gas.s0(s3.T_K), _ES, system)),
        latex_chain(
            s0("4s"),
            rf"{s0('3')} - R{g}\,\ln\frac{{p_3}}{{p_4}}",
            rf"{_n(gas.s0(s3.T_K), _ES, system)} - "
            rf"{_n(R, 'specific_heat', system)}\cdot \ln {latex_number(ratio, 5)}",
            _q(gas.s0(s4s.T_K), _ES, system),
        ),
        rf"T_{{4s}} = {_T(s4s.T_K, system)}",
        rf"h_3 = {_q(s3.h_J_per_kg, _EH, system)}",
        rf"h_{{4s}} = {_q(s4s.h_J_per_kg, _EH, system)}",
    ]
    if inputs.eta_turbine < 1.0:
        eta = latex_number(inputs.eta_turbine, 4)
        dh_s = s3.h_J_per_kg - s4s.h_J_per_kg
        diff = _diff(s3.h_J_per_kg, s4s.h_J_per_kg, _EH, system)
        h3 = _n(s3.h_J_per_kg, _EH, system)
        if r"\times" in diff:
            # Con ×10ⁿ (SI) la resta no entra en el renglón: primero el salto isoentrópico.
            lines.append(latex_chain(r"\Delta h_s", r"h_3 - h_{4s}", diff, _q(dh_s, _EH, system)))
            definition = r"h_3 - \eta_T\,\Delta h_s"
            replaced = _wrap(h3, "-", rf"{eta}\cdot {_n(dh_s, _EH, system)}")
        else:
            definition = r"h_3 - \eta_T\,(h_3 - h_{4s})"
            replaced = rf"{h3} - {eta}\,({diff})"
        lines.append(latex_chain("h_4", definition, replaced, _q(s4.h_J_per_kg, _EH, system)))
        lines.append(rf"T_4 = {_T(s4.T_K, system)}")
    f = result.fuel_air_ratio
    if inputs.air_standard:
        lines.append(
            latex_chain(
                "w_T",
                "h_3 - h_4",
                _diff(s3.h_J_per_kg, s4.h_J_per_kg, _EH, system),
                _q(result.w_turbine_J_per_kg, _EH, system),
            )
        )
    else:
        lines.append(
            latex_chain(
                "w_T",
                r"(1 + f)\,(h_3 - h_4)",
                factor_times_diff(latex_number(1.0 + f, 6), s3.h_J_per_kg, s4.h_J_per_kg, system),
                _q(result.w_turbine_J_per_kg, _EH, system),
            )
        )
    return ProcedureStep(
        title="Turbina (3 → 4)",
        text=(
            "Los gases se expanden hasta la presión ambiente (la caldera de recuperación casi no "
            "frena el escape). La expansión isoentrópica sale otra vez de la función s°, ahora "
            "la de los gases"
            + (" (por la turbina pasa aire)." if inputs.air_standard else ".")
            + (
                " La turbina es ideal: 4 = 4s."
                if inputs.eta_turbine == 1.0
                else " El rendimiento isoentrópico da la salida real."
            )
            + (
                ""
                if inputs.air_standard
                else " Por la turbina pasan 1 + f kg de gases por cada kg de aire."
            )
        ),
        latex=tuple(lines),
    )


def _net_step(result: BraytonResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    lines = [
        latex_chain(
            r"w_{\mathrm{neto}}",
            "w_T - w_C",
            _diff(result.w_turbine_J_per_kg, result.w_compressor_J_per_kg, _EH, system),
            _q(result.w_net_J_per_kg, _EH, system),
        ),
    ]
    if not inputs.air_standard:
        fuel = inputs.fuel
        assert fuel is not None
        lines.append(
            latex_chain(
                r"q_{\mathrm{ent}}",
                r"f\,\mathrm{PCI}",
                rf"{latex_number(result.fuel_air_ratio, 5)}\cdot "
                rf"{_n(fuel.lhv_J_per_kg, _EH, system)}",
                _q(result.q_in_J_per_kg, _EH, system),
            )
        )
    lines += [
        latex_chain(
            r"\eta_{TG}",
            r"\frac{w_{\mathrm{neto}}}{q_{\mathrm{ent}}}",
            rf"\frac{{{_n(result.w_net_J_per_kg, _EH, system)}}}"
            rf"{{{_n(result.q_in_J_per_kg, _EH, system)}}}",
            rf"{latex_number(result.eta_th * 100, 4)}\,\%",
        ),
        latex_chain(
            r"r_{bw}",
            r"\frac{w_C}{w_T}",
            rf"\frac{{{_n(result.w_compressor_J_per_kg, _EH, system)}}}"
            rf"{{{_n(result.w_turbine_J_per_kg, _EH, system)}}}",
            latex_number(result.back_work_ratio, 4),
        ),
    ]
    return ProcedureStep(
        title="Trabajo neto y rendimiento",
        text=(
            "Todo por kilogramo de aire. La relación de trabajo de retroceso r_bw es la parte "
            "del trabajo de la turbina que se lleva el compresor (Cengel §9-8)."
        ),
        latex=tuple(lines),
    )


def _power_step(result: BraytonResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    m_a = _n(result.m_air_kg_s, "mass_flow", system)
    lines = []
    if not inputs.air_standard:
        lines.append(
            latex_chain(
                r"\dot{m}_f",
                r"f\,\dot{m}_a",
                rf"{latex_number(result.fuel_air_ratio, 5)}\cdot {m_a}",
                _q(result.m_fuel_kg_s, "mass_flow", system),
            )
        )
    lines += [
        latex_chain(
            r"\dot{W}_{TG}",
            r"\dot{m}_a\,w_{\mathrm{neto}}",
            rf"{m_a}\cdot {_n(result.w_net_J_per_kg, _EH, system)}",
            _q(result.W_net_W, "power", system),
        ),
        latex_chain(
            r"\dot{Q}_{\mathrm{ent}}",
            r"\dot{m}_a\,q_{\mathrm{ent}}",
            rf"{m_a}\cdot {_n(result.q_in_J_per_kg, _EH, system)}",
            _q(result.Q_in_W, "power", system),
        ),
    ]
    return ProcedureStep(
        title="Caudales y potencias",
        text="Con el caudal de aire, todo escala: potencias, calor y caudal de combustible.",
        latex=tuple(lines),
    )


def brayton_steps(result: BraytonResult, system: UnitSystem) -> list[ProcedureStep]:
    """Procedimiento completo de la turbina de gas, como se resuelve a mano."""
    steps = [_air_step(result, system), _compressor_step(result, system)]
    if result.inputs.air_standard:
        steps.append(_heater_step(result, system))
    else:
        steps += [_combustion_step(result, system), _gases_step(result, system)]
    steps += [
        _turbine_step(result, system),
        _net_step(result, system),
        _power_step(result, system),
    ]
    return steps
