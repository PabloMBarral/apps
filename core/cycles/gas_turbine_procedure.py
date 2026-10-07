"""Procedimiento de la turbina de gas con etapas y regenerador «como con las tablas» — Fase 3.7.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo, en el orden de Çengel & Boles (§9-8 a §9-10):

1. la composición del aire (vademecum §5) y las presiones de cada etapa (con
   relaciones iguales, el mínimo trabajo del vademecum §6.3);
2. cada etapa del compresor con la función s° (como la tabla A-17) y su
   rendimiento isoentrópico (vademecum §10.4), y cada interenfriador;
3. con combustión, cada cámara (el balance con el PCI de ISO 6976:2016,
   vademecum §16) seguida de su turbina; con aire estándar, primero las
   turbinas;
4. el regenerador, con la T₄ de la turbina (como el ejemplo 9-7);
5. con aire estándar, el calor que entra en cada calentador;
6. el trabajo neto, el rendimiento y las potencias;
7. la exergía: lo que se destruye y se pierde en cada componente (vademecum
   §11; la del combustible ≈ PCI, §16.13).

Recibe un resultado ya calculado
(:class:`core.cycles.gas_turbine.GasTurbineResult`): no importa Streamlit.
"""

from __future__ import annotations

from core.cycles.brayton_procedure import _formula
from core.cycles.gas_turbine import (
    Combustor,
    CycleState,
    GasTurbineResult,
    Stage,
    gas_turbine_exergy,
    of_component,
)
from core.cycles.hrsg_procedure import factor_times_diff, gas_composition_lines
from core.cycles.rankine_procedure import _diff, _n, _q, _wrap
from core.ideal_gas import GAS_NAMES, GAS_SPECIES, FlueGas
from core.latex import latex_chain, latex_is_wide, latex_number, latex_paren
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem

__all__ = ["gas_turbine_steps"]

_EH: QuantityKind = "specific_enthalpy"
_ES: QuantityKind = "specific_entropy"


def _T(value_K: float, system: UnitSystem) -> str:
    return _q(value_K, "temperature", system)


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _bar(p_Pa: float) -> str:
    return f"{p_Pa / 1e5:.4g} bar"


def _arrow(a: CycleState, b: CycleState) -> str:
    return f"{a.label} → {b.label}"


def _composition(gas: FlueGas) -> str:
    y = gas.mole_fractions
    return ", ".join(f"{GAS_NAMES[s]} {y[s] * 100:.3g} %" for s in GAS_SPECIES if y[s] > 0.0)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _wrap_wide(head: str, op: str, tail: str) -> str:
    r"""Como ``_wrap``, pero corta también con números negativos (entre paréntesis)."""
    if latex_is_wide(head, tail):
        return rf"{head} \\ &\quad {op} {tail}"
    return f"{head} {op} {tail}"


def _p_line(state: CycleState, system: UnitSystem) -> str:
    return rf"p_{{{state.label}}} = {_q(state.P_Pa, 'pressure', system)}"


def _m_sym(result: GasTurbineResult) -> str:
    """El factor de caudal de los gases: (1 + f), o (1 + F) con varias cámaras."""
    return r"(1 + F)\," if len(result.combustors) > 1 else r"(1 + f)\,"


def _s0(gas_side: bool, label: str) -> str:
    """s°₃ del aire o s°_g,₃ de los gases."""
    return rf"s^{{\circ}}_{{g,{label}}}" if gas_side else rf"s^{{\circ}}_{{{label}}}"


# ---------------------------------------------------------------------
# Aire y presiones
# ---------------------------------------------------------------------


def _air_step(result: GasTurbineResult, system: UnitSystem) -> ProcedureStep:
    return ProcedureStep(
        title="Composición del aire",
        text=(
            "El aire es una mezcla de gases ideales (vademecum §5.1 y §5.2): con las fracciones "
            "molares salen la masa molar, las fracciones másicas y la constante particular R. "
            "Las entalpías se miden desde 25 °C, la referencia del poder calorífico."
        ),
        latex=tuple(gas_composition_lines(result.inputs.air, system)),
    )


def _pressures_step(result: GasTurbineResult, system: UnitSystem) -> ProcedureStep | None:
    """Las presiones de cada etapa (solo con etapas o con pérdidas de carga)."""
    inputs = result.inputs
    st = result.states
    n_c, n_t = inputs.compressor_stages, inputs.turbine_stages
    has_dp = (
        inputs.dp_combustor > 0.0
        or (inputs.intercooled and inputs.dp_intercooler > 0.0)
        or (inputs.regenerated and inputs.dp_regenerator > 0.0)
    )
    if n_c == 1 and n_t == 1 and not has_dp:
        return None
    first, last = result.compressors[0], result.compressors[-1]
    p1, p2 = st[first.inlet], st[last.outlet]
    rp = latex_number(inputs.pressure_ratio, 5)
    lines = [
        latex_chain(
            rf"p_{{{p2.label}}}",
            rf"r_p\,p_{{{p1.label}}}",
            rf"{rp}\cdot {_n(p1.P_Pa, 'pressure', system)}",
            _q(p2.P_Pa, "pressure", system),
        )
    ]
    text = [
        f"El compresor sube la presión de p₁ a r_p·p₁. Con {n_c} etapas, cada una lleva la "
        "misma relación de presiones: es la que hace mínimo el trabajo de compresión con el "
        "interenfriamiento hasta la misma temperatura (vademecum §6.3)."
        if n_c > 1 and not inputs.p_intercool_Pa
        else "El compresor sube la presión de p₁ a r_p·p₁."
    ]
    if n_c > 1:
        if inputs.p_intercool_Pa:
            text.append(" Las presiones de los interenfriadores son un dato.")
            for c in result.compressors[:-1]:
                lines.append(
                    rf"p_{{{st[c.outlet].label}}} = {_q(st[c.outlet].P_Pa, 'pressure', system)}"
                )
        else:
            r_c = st[first.outlet].P_Pa / st[first.inlet].P_Pa
            if inputs.dp_intercooler > 0.0:
                dp = latex_number(inputs.dp_intercooler, 3)
                lines.append(
                    latex_chain(
                        "r_C",
                        rf"\left[\frac{{r_p}}{{(1 - \Delta p_{{IC}})^{{{n_c - 1}}}}}\right]"
                        rf"^{{1/{n_c}}}",
                        rf"\left[\frac{{{rp}}}{{(1 - {dp})^{{{n_c - 1}}}}}\right]^{{1/{n_c}}}",
                        latex_number(r_c, 5),
                    )
                )
                text.append(
                    " Cada interenfriador pierde una fracción Δp_IC de la presión: la relación "
                    "de cada etapa la compensa."
                )
            else:
                lines.append(
                    latex_chain(
                        "r_C", rf"r_p^{{1/{n_c}}}", rf"{rp}^{{1/{n_c}}}", latex_number(r_c, 5)
                    )
                )
                if n_c == 2:
                    lines.append(
                        latex_chain(
                            rf"p_{{{st[first.outlet].label}}}",
                            rf"\sqrt{{p_{{{p1.label}}}\,p_{{{p2.label}}}}}",
                            _q(st[first.outlet].P_Pa, "pressure", system),
                        )
                    )
            for c in result.compressors[:-1]:
                if n_c > 2:
                    lines.append(
                        rf"p_{{{st[c.outlet].label}}} = {_q(st[c.outlet].P_Pa, 'pressure', system)}"
                    )
    t_first, t_last = result.turbines[0], result.turbines[-1]
    p_t_in, p_t_out = st[t_first.inlet], st[t_last.outlet]
    if has_dp:
        losses = [r"(1 - \Delta p_{\text{cám}})"]
        values = [rf"(1 - {latex_number(inputs.dp_combustor, 3)})"]
        if inputs.regenerated and inputs.dp_regenerator > 0.0:
            losses.insert(0, r"(1 - \Delta p_{\text{reg}})")
            values.insert(0, rf"(1 - {latex_number(inputs.dp_regenerator, 3)})")
        lines.append(
            latex_chain(
                rf"p_{{{p_t_in.label}}}",
                rf"p_{{{p2.label}}}\," + r"\,".join(losses),
                _wrap(_n(p2.P_Pa, "pressure", system), r"\cdot", r"\cdot ".join(values)),
                _q(p_t_in.P_Pa, "pressure", system),
            )
        )
        text.append(
            " Las cámaras"
            + (", el regenerador" if inputs.regenerated else "")
            + " y los interenfriadores" * inputs.intercooled
            + " pierden presión (Δp, fracción de la de entrada)."
        )
        if inputs.regenerated and inputs.dp_regenerator > 0.0:
            lines.append(
                latex_chain(
                    rf"p_{{{p_t_out.label}}}",
                    rf"\frac{{p_{{{p1.label}}}}}{{1 - \Delta p_{{\text{{reg}}}}}}",
                    _q(p_t_out.P_Pa, "pressure", system),
                )
            )
            text.append(
                " La turbina descarga un poco por encima del ambiente: los gases todavía tienen "
                "que cruzar el regenerador."
            )
    if n_t > 1:
        if inputs.p_reheat_Pa:
            text.append(" Las presiones de recalentamiento son un dato.")
        else:
            r_t = p_t_in.P_Pa / st[t_first.outlet].P_Pa
            text.append(
                f" Las {n_t} etapas de la turbina se reparten la expansión con la misma relación "
                "de presiones (el máximo trabajo con recalentamiento hasta la misma temperatura)."
            )
            lines.append(
                latex_chain(
                    "r_T",
                    rf"\frac{{p_{{{p_t_in.label}}}}}{{p_{{{st[t_first.outlet].label}}}}}",
                    latex_number(r_t, 5),
                )
            )
        for t in result.turbines[:-1]:
            lines.append(
                rf"p_{{{st[t.outlet].label}}} = {_q(st[t.outlet].P_Pa, 'pressure', system)}"
            )
    return ProcedureStep(title="Presiones de cada etapa", text="".join(text), latex=tuple(lines))


# ---------------------------------------------------------------------
# Compresores e interenfriadores
# ---------------------------------------------------------------------


def _compressor_step(
    result: GasTurbineResult, stage: Stage, system: UnitSystem, single: bool
) -> ProcedureStep:
    inputs = result.inputs
    air = inputs.air
    st = result.states
    a, s_, b = st[stage.inlet], st[stage.outlet_s], st[stage.outlet]
    i, o, os_ = a.label, b.label, s_.label
    R = air.R_J_per_kg_K
    ratio = b.P_Pa / a.P_Pa
    lines = []
    if (
        single
        and not inputs.p_intercool_Pa
        and inputs.dp_combustor == 0.0
        and not inputs.regenerated
    ):
        lines.append(
            latex_chain(
                rf"p_{{{o}}}",
                rf"r_p\,p_{{{i}}}",
                rf"{latex_number(inputs.pressure_ratio, 5)}\cdot {_n(a.P_Pa, 'pressure', system)}",
                _q(b.P_Pa, "pressure", system),
            )
        )
    lines += [
        latex_chain(_s0(False, i), rf"s^{{\circ}}(T_{{{i}}})", _q(air.s0(a.T_K), _ES, system)),
        latex_chain(
            _s0(False, os_),
            rf"{_s0(False, i)} + R\,\ln\frac{{p_{{{o}}}}}{{p_{{{i}}}}}",
            _wrap_wide(
                _n(air.s0(a.T_K), _ES, system),
                "+",
                rf"{_n(R, 'specific_heat', system)}\cdot \ln {latex_number(ratio, 5)}",
            ),
            _q(air.s0(s_.T_K), _ES, system),
        ),
        rf"T_{{{os_}}} = {_T(s_.T_K, system)}",
        rf"h_{{{i}}} = {_q(a.h_J_per_kg, _EH, system)}",
        rf"h_{{{os_}}} = {_q(s_.h_J_per_kg, _EH, system)}",
    ]
    if inputs.eta_compressor < 1.0:
        lines.append(
            latex_chain(
                rf"h_{{{o}}}",
                rf"h_{{{i}}} + \frac{{h_{{{os_}}} - h_{{{i}}}}}{{\eta_C}}",
                _wrap(
                    _n(a.h_J_per_kg, _EH, system),
                    "+",
                    rf"\frac{{{_diff(s_.h_J_per_kg, a.h_J_per_kg, _EH, system)}}}"
                    rf"{{{latex_number(inputs.eta_compressor, 4)}}}",
                ),
                _q(b.h_J_per_kg, _EH, system),
            )
        )
        lines.append(rf"T_{{{o}}} = {_T(b.T_K, system)}")
    w = r"w_C" if single else rf"w_{{C,{stage_index(result, stage)}}}"
    lines.append(
        latex_chain(
            w,
            rf"h_{{{o}}} - h_{{{i}}}",
            _diff(b.h_J_per_kg, a.h_J_per_kg, _EH, system),
            _q(result.stage_work(stage), _EH, system),
        )
    )
    ideal = inputs.eta_compressor == 1.0
    where = (
        f"El aire entra a T{_sub(i)} = {_degC(a.T_K)} y sale a {_bar(b.P_Pa)}."
        if single
        else f"Entra a T{_sub(i)} = {_degC(a.T_K)} y {_bar(a.P_Pa)}, y sale a {_bar(b.P_Pa)}."
    )
    return ProcedureStep(
        title=f"{_cap(stage.name)} ({_arrow(a, b)})",
        text=(
            f"{where} Con calores específicos variables, la compresión isoentrópica cumple "
            "s°_sal,s = s°_ent + R·ln(p_sal/p_ent), con s° = s°(T) la columna s° de la tabla "
            "A-17 de Cengel (acá medida desde 25 °C, Cengel §7-9): se busca la temperatura que "
            "da ese s°."
            + (
                f" La etapa es ideal: {o} = {os_}."
                if ideal
                else " El rendimiento isoentrópico da la salida real (vademecum §10.4)."
            )
        ),
        latex=tuple(lines),
    )


def stage_index(result: GasTurbineResult, stage: Stage) -> int:
    """1, 2, … en el orden del flujo (compresores de baja a alta, turbinas de alta a baja)."""
    group = result.compressors if stage.kind == "compresor" else result.turbines
    return group.index(stage) + 1


_SUBSCRIPTS = str.maketrans("0123456789s", "₀₁₂₃₄₅₆₇₈₉ₛ")


def _sub(label: str) -> str:
    return label.translate(_SUBSCRIPTS)


def _intercooler_step(result: GasTurbineResult, k: int, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    ic = result.intercoolers[k]
    st = result.states
    a, b = st[ic.inlet], st[ic.outlet]
    i, o = a.label, b.label
    lines = []
    if inputs.dp_intercooler > 0.0:
        lines.append(
            latex_chain(
                rf"p_{{{o}}}",
                rf"p_{{{i}}}\,(1 - \Delta p_{{IC}})",
                rf"{_n(a.P_Pa, 'pressure', system)}\cdot "
                rf"(1 - {latex_number(inputs.dp_intercooler, 3)})",
                _q(b.P_Pa, "pressure", system),
            )
        )
    lines += [
        rf"h_{{{o}}} = h(T_{{{o}}}) = {_q(b.h_J_per_kg, _EH, system)}",
        latex_chain(
            r"q_{IC}" if len(result.intercoolers) == 1 else rf"q_{{IC,{k + 1}}}",
            rf"h_{{{i}}} - h_{{{o}}}",
            _diff(a.h_J_per_kg, b.h_J_per_kg, _EH, system),
            _q(result.intercooler_heat(ic), _EH, system),
        ),
    ]
    to = "la temperatura ambiente" if inputs.T_intercool_K is None else _degC(b.T_K)
    return ProcedureStep(
        title=f"{_cap(ic.name)} ({_arrow(a, b)})",
        text=(
            f"El aire se enfría a presión constante hasta {to} antes de la etapa siguiente "
            "(Cengel §9-10): la etapa que sigue comprime aire frío, que pide menos trabajo. El "
            "calor sale al agua de enfriamiento."
        ),
        latex=tuple(lines),
    )


# ---------------------------------------------------------------------
# Cámaras y turbinas
# ---------------------------------------------------------------------


def _fuel_lines(result: GasTurbineResult, system: UnitSystem) -> tuple[list[str], float]:
    """PCI, aire teórico y f teórica del combustible (las líneas de la primera cámara)."""
    inputs = result.inputs
    fuel = inputs.fuel
    assert fuel is not None
    air = inputs.air
    a, b, c, d = fuel.atoms
    o2 = a + b / 4.0 - d / 2.0
    y_o2 = air.mole_fractions["O2"]
    n_air_t = o2 / y_o2
    f_t = fuel.M_kg_per_mol / (n_air_t * air.M_kg_per_mol)
    hhv_m = fuel.hhv_molar_J_per_mol / 1e3
    lhv_m = fuel.lhv_molar_J_per_mol / 1e3
    L0 = (hhv_m - lhv_m) / (b / 2.0)
    del c
    lines = [
        latex_chain(
            r"\mathrm{PCI}_m",
            r"\mathrm{PCS}_m - \tfrac{b}{2}\,L_0",
            rf"{latex_number(hhv_m, 5)} - {latex_number(b / 2.0, 4)}\cdot {latex_number(L0, 5)}",
            rf"{latex_number(lhv_m, 5)}\ \mathrm{{kJ/mol}}",
        ),
        latex_chain(
            r"\mathrm{PCI}", r"\frac{\mathrm{PCI}_m}{M_f}", _q(fuel.lhv_J_per_kg, _EH, system)
        ),
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
    ]
    return lines, f_t


def _combustor_step(
    result: GasTurbineResult, j: int, f_t: float, system: UnitSystem
) -> list[ProcedureStep]:
    """La cámara j (combustión): el balance con el PCI y, en la primera, el combustible."""
    inputs = result.inputs
    fuel = inputs.fuel
    assert fuel is not None
    st = result.states
    cc = result.combustors[j]
    a, b = st[cc.inlet], st[cc.outlet]
    i, o = a.label, b.label
    lhv = fuel.lhv_J_per_kg
    single = len(result.combustors) == 1
    rg = result.regenerator
    lines: list[str] = []
    steps: list[ProcedureStep] = []
    if j == 0:
        fuel_lines, _ = _fuel_lines(result, system)
        lines += fuel_lines
        f = "f" if single else "f_1"
        lines += [
            rf"h_a(T_{{{i}}}) + {f}\,\mathrm{{PCI}} = (1 + {f})\,h_g(T_{{{o}}})",
            rf"h_a(T_{{{i}}}) = {_q(a.h_J_per_kg, _EH, system)}",
            rf"h_g(T_{{{o}}}) = {_q(b.h_J_per_kg, _EH, system)}",
            latex_chain(
                f,
                rf"\frac{{h_g(T_{{{o}}}) - h_a(T_{{{i}}})}}{{\mathrm{{PCI}} - h_g(T_{{{o}}})}}",
                rf"\frac{{{_diff(b.h_J_per_kg, a.h_J_per_kg, _EH, system)}}}"
                rf"{{{_diff(lhv, b.h_J_per_kg, _EH, system)}}}",
                latex_number(cc.fuel, 5),
            ),
            latex_chain(
                r"\lambda" if single else r"\lambda_1",
                rf"\frac{{f_t}}{{{f}}}",
                rf"\frac{{{latex_number(f_t, 5)}}}{{{latex_number(cc.fuel, 5)}}}",
                latex_number(f_t / cc.fuel, 4),
            ),
        ]
        name = "metano" if fuel.name == "methane" else fuel.name
        formula = _formula(*fuel.atoms)
        came = (
            f"El aire llega desde el regenerador a T{_sub(i)} = {_degC(a.T_K)}: sale del paso del "
            "regenerador, más abajo, con la T₄ de la turbina; como T₄ depende de f y f de "
            f"T{_sub(i)}, se itera hasta que no cambia (estos son los valores finales). "
            if rg is not None
            else ""
        )
        steps.append(
            ProcedureStep(
                title=f"{_cap(cc.name)} ({_arrow(a, b)})",
                text=(
                    came + f"El combustible ({name}, {formula} por mol) entra a 25 °C. Su poder "
                    "calorífico inferior sale de ISO 6976:2016 a 25 °C: el superior menos el "
                    "calor latente del agua que se forma. La combustión completa pide "
                    "a + b/4 − d/2 moles de O₂ por mol de combustible (vademecum §16.2): eso da el "
                    "aire teórico y la relación combustible/aire teórica f_t. Con las entalpías "
                    "medidas desde 25 °C, el balance de la cámara adiabática, por kilogramo de "
                    "aire, da la relación combustible/aire; como h_g depende de la composición de "
                    "los gases, que depende de f, se resuelve iterando."
                ),
                latex=tuple(lines),
            )
        )
        steps.append(
            ProcedureStep(
                title="Composición de los gases de combustión",
                text=(
                    "Los gases que salen de la cámara son otra mezcla de gases ideales: su masa "
                    f"molar y su R cambian respecto del aire. Salen con {_composition(b.mix)}."
                ),
                latex=tuple(gas_composition_lines(b.mix, system)),
            )
        )
        return steps
    # Recalentamiento: (1 + F_ant)·h_ent + f_j·PCI = (1 + F_j)·h_sal.
    F_in = rf"F_{{{j}}}"
    F_out = rf"F_{{{j + 1}}}"
    fj = rf"f_{{{j + 1}}}"
    m_in = latex_number(cc.m_in, 6)
    lines += [
        rf"(1 + {F_in})\,h_{{{i}}} + {fj}\,\mathrm{{PCI}} = (1 + {F_out})\,h_{{{o}}}",
        rf"h_{{{i}}} = {_q(a.h_J_per_kg, _EH, system)}",
        rf"h_{{{o}}} = h_g(T_{{{o}}}) = {_q(b.h_J_per_kg, _EH, system)}",
        latex_chain(
            r"\Delta h",
            rf"h_{{{o}}} - h_{{{i}}}",
            _diff(b.h_J_per_kg, a.h_J_per_kg, _EH, system),
            _q(b.h_J_per_kg - a.h_J_per_kg, _EH, system),
        ),
        latex_chain(
            fj,
            rf"\frac{{(1 + {F_in})\,\Delta h}}{{\mathrm{{PCI}} - h_{{{o}}}}}",
            rf"\frac{{{m_in}\cdot {_n(b.h_J_per_kg - a.h_J_per_kg, _EH, system)}}}"
            rf"{{{_diff(lhv, b.h_J_per_kg, _EH, system)}}}",
            latex_number(cc.fuel, 5),
        ),
        latex_chain(
            F_out,
            rf"{F_in} + {fj}",
            rf"{latex_number(cc.F_in, 5)} + {latex_number(cc.fuel, 5)}",
            latex_number(cc.F_out, 5),
        ),
        latex_chain(
            rf"\lambda_{{{j + 1}}}",
            rf"\frac{{f_t}}{{{F_out}}}",
            rf"\frac{{{latex_number(f_t, 5)}}}{{{latex_number(cc.F_out, 5)}}}",
            latex_number(f_t / cc.F_out, 4),
        ),
        latex_chain("R_g", _q(b.mix.R_J_per_kg_K, "specific_heat", system)),
    ]
    steps.append(
        ProcedureStep(
            title=f"{_cap(cc.name)} ({_arrow(a, b)})",
            text=(
                f"Los gases que salen de la turbina anterior todavía tienen "
                f"{a.mix.mole_fractions['O2'] * 100:.3g} % de O₂: una segunda cámara quema más "
                f"combustible en ellos hasta {_degC(b.T_K)} (combustión secuencial). Por cada kg "
                f"de aire entran 1 + F{_sub(str(j))} kg de gases, con F la relación "
                "combustible/aire acumulada. El balance es el de la primera cámara, pero ahora "
                f"la composición de los gases cambia otra vez: salen con {_composition(b.mix)}."
            ),
            latex=tuple(lines),
        )
    )
    return steps


def _turbine_step(result: GasTurbineResult, stage: Stage, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    st = result.states
    a, s_, b = st[stage.inlet], st[stage.outlet_s], st[stage.outlet]
    i, o, os_ = a.label, b.label, s_.label
    gas = a.mix
    gas_side = not inputs.air_standard
    g = "_g" if gas_side else ""
    R = gas.R_J_per_kg_K
    ratio = a.P_Pa / b.P_Pa
    single = len(result.turbines) == 1
    lines = [
        latex_chain(
            _s0(gas_side, i), rf"s^{{\circ}}{g}(T_{{{i}}})", _q(gas.s0(a.T_K), _ES, system)
        ),
        latex_chain(
            _s0(gas_side, os_),
            rf"{_s0(gas_side, i)} - R{g}\,\ln\frac{{p_{{{i}}}}}{{p_{{{o}}}}}",
            _wrap_wide(
                _n(gas.s0(a.T_K), _ES, system),
                "-",
                rf"{_n(R, 'specific_heat', system)}\cdot \ln {latex_number(ratio, 5)}",
            ),
            _q(gas.s0(s_.T_K), _ES, system),
        ),
        rf"T_{{{os_}}} = {_T(s_.T_K, system)}",
        rf"h_{{{i}}} = {_q(a.h_J_per_kg, _EH, system)}",
        rf"h_{{{os_}}} = {_q(s_.h_J_per_kg, _EH, system)}",
    ]
    if inputs.eta_turbine < 1.0:
        eta = latex_number(inputs.eta_turbine, 4)
        dh_s = a.h_J_per_kg - s_.h_J_per_kg
        diff = _diff(a.h_J_per_kg, s_.h_J_per_kg, _EH, system)
        h_in = _n(a.h_J_per_kg, _EH, system)
        if r"\times" in diff:
            # Con ×10ⁿ (SI) la resta no entra en el renglón: primero el salto isoentrópico.
            lines.append(
                latex_chain(r"\Delta h_s", rf"h_{{{i}}} - h_{{{os_}}}", diff, _q(dh_s, _EH, system))
            )
            definition = rf"h_{{{i}}} - \eta_T\,\Delta h_s"
            replaced = _wrap(h_in, "-", rf"{eta}\cdot {_n(dh_s, _EH, system)}")
        else:
            definition = rf"h_{{{i}}} - \eta_T\,(h_{{{i}}} - h_{{{os_}}})"
            replaced = rf"{h_in} - {eta}\,({diff})"
        lines.append(
            latex_chain(rf"h_{{{o}}}", definition, replaced, _q(b.h_J_per_kg, _EH, system))
        )
        lines.append(rf"T_{{{o}}} = {_T(b.T_K, system)}")
    w = r"w_T" if single else rf"w_{{T,{stage_index(result, stage)}}}"
    if inputs.air_standard:
        lines.append(
            latex_chain(
                w,
                rf"h_{{{i}}} - h_{{{o}}}",
                _diff(a.h_J_per_kg, b.h_J_per_kg, _EH, system),
                _q(result.stage_work(stage), _EH, system),
            )
        )
        flow = " (por la turbina pasa aire)."
    else:
        F = "f" if len(result.combustors) == 1 else rf"F_{{{stage_index(result, stage)}}}"
        lines.append(
            latex_chain(
                w,
                rf"(1 + {F})\,(h_{{{i}}} - h_{{{o}}})",
                factor_times_diff(latex_number(stage.m, 6), a.h_J_per_kg, b.h_J_per_kg, system),
                _q(result.stage_work(stage), _EH, system),
            )
        )
        flow = f". Por cada kg de aire pasan {latex_number(stage.m, 5)} kg de gases."
    ideal = inputs.eta_turbine == 1.0
    return ProcedureStep(
        title=f"{_cap(stage.name)} ({_arrow(a, b)})",
        text=(
            f"Los gases entran a {_degC(a.T_K)} y {_bar(a.P_Pa)} y se expanden hasta "
            f"{_bar(b.P_Pa)}. La expansión isoentrópica sale otra vez de la función s°, ahora la "
            "de los gases"
            + flow
            + (
                f" La etapa es ideal: {o} = {os_}."
                if ideal
                else " El rendimiento isoentrópico da la salida real."
            )
        ),
        latex=tuple(lines),
    )


def _heater_step(result: GasTurbineResult, cc: Combustor, system: UnitSystem) -> ProcedureStep:
    """Aire estándar: el calor que entra en un calentador (o recalentador)."""
    st = result.states
    a, b = st[cc.inlet], st[cc.outlet]
    i, o = a.label, b.label
    single = len(result.combustors) == 1
    k = result.combustors.index(cc) + 1
    q = r"q_{\mathrm{ent}}" if single else rf"q_{{{k}}}"
    first = k == 1
    text = (
        "Con el modelo de aire estándar (Cengel §9-3) la cámara de combustión se reemplaza por "
        f"un intercambiador: el aire recibe calor a presión constante hasta T{_sub(o)}."
        if first
        else f"El recalentador vuelve a calentar el aire hasta T{_sub(o)} entre las dos etapas de "
        "la turbina (Cengel §9-10)."
    )
    if first and result.regenerator is not None:
        text += f" Llega precalentado del regenerador a T{_sub(i)}."
    return ProcedureStep(
        title=f"{_cap(cc.name)} ({_arrow(a, b)}), aire estándar",
        text=text,
        latex=(
            rf"h_{{{o}}} = {_q(b.h_J_per_kg, _EH, system)}",
            latex_chain(
                q,
                rf"h_{{{o}}} - h_{{{i}}}",
                _diff(b.h_J_per_kg, a.h_J_per_kg, _EH, system),
                _q(cc.q, _EH, system),
            ),
        ),
    )


# ---------------------------------------------------------------------
# Regenerador
# ---------------------------------------------------------------------


def _regenerator_step(result: GasTurbineResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    rg = result.regenerator
    assert rg is not None
    st = result.states
    a2, a5, g4, g6 = st[rg.air_in], st[rg.air_out], st[rg.gas_in], st[rg.gas_out]
    n2, n5, n4, n6 = a2.label, a5.label, g4.label, g6.label
    eps = latex_number(rg.effectiveness, 4)
    h_a4 = a2.h_J_per_kg + rg.q_max
    air_std = inputs.air_standard
    h4_sym = rf"h_{{{n4}}}" if air_std else rf"h_a(T_{{{n4}}})"
    q_max = r"q_{\text{reg,máx}}"
    lines = [
        rf"{h4_sym} = {_q(h_a4, _EH, system)}",
        latex_chain(
            q_max,
            rf"{h4_sym} - h_{{{n2}}}",
            _diff(h_a4, a2.h_J_per_kg, _EH, system),
            _q(rg.q_max, _EH, system),
        ),
        latex_chain(
            r"q_{\mathrm{reg}}",
            rf"\varepsilon\,{q_max}",
            rf"{eps}\cdot {_n(rg.q_max, _EH, system)}",
            _q(rg.q, _EH, system),
        ),
        latex_chain(
            rf"h_{{{n5}}}",
            rf"h_{{{n2}}} + q_{{\mathrm{{reg}}}}",
            _wrap(_n(a2.h_J_per_kg, _EH, system), "+", _n(rg.q, _EH, system)),
            _q(a5.h_J_per_kg, _EH, system),
        ),
        rf"T_{{{n5}}} = {_T(a5.T_K, system)}",
    ]
    if air_std:
        lines.append(
            latex_chain(
                rf"h_{{{n6}}}",
                rf"h_{{{n4}}} - q_{{\mathrm{{reg}}}}",
                _diff(g4.h_J_per_kg, rg.q, _EH, system),
                _q(g6.h_J_per_kg, _EH, system),
            )
        )
    else:
        F = "f" if len(result.combustors) == 1 else "F"
        lines.append(
            latex_chain(
                rf"h_{{{n6}}}",
                rf"h_{{{n4}}} - \frac{{q_{{\mathrm{{reg}}}}}}{{1 + {F}}}",
                _wrap(
                    _n(g4.h_J_per_kg, _EH, system),
                    "-",
                    rf"\frac{{{_n(rg.q, _EH, system)}}}{{{latex_number(rg.m_gas, 6)}}}",
                ),
                _q(g6.h_J_per_kg, _EH, system),
            )
        )
    lines.append(rf"T_{{{n6}}} = {_T(g6.T_K, system)}")
    gases = "el aire" if air_std else "los gases"
    return ProcedureStep(
        title=f"Regenerador ({n2} → {n5} y {n4} → {n6})",
        text=(
            f"{_cap(gases)} que salen de la turbina a T{_sub(n4)} = {_degC(g4.T_K)} calientan el "
            f"aire que sale del compresor a T{_sub(n2)} = {_degC(a2.T_K)}. Como mucho, el aire "
            f"llegaría a T{_sub(n4)}: el calor máximo es q_reg,máx = h(T{_sub(n4)}) − "
            f"h{_sub(n2)}, con la entalpía del aire, y la efectividad ε es la parte de ese máximo "
            "que se logra (Cengel §9-9). "
            + (
                "Lo que gana el aire lo pierden los gases (por kg de aire pasan 1 + f kg de gases)."
                if not air_std
                else "Lo que gana el aire lo pierde el escape."
            )
        ),
        latex=tuple(lines),
    )


# ---------------------------------------------------------------------
# Trabajo, rendimiento, potencias
# ---------------------------------------------------------------------


def _sum_line(symbol: str, parts: list[tuple[str, float]], total: float, system: UnitSystem) -> str:
    """``w_C = w_{C,1} + w_{C,2} = … = total``, un sumando por renglón si son muchos."""
    if len(parts) == 1:
        return ""
    names = " + ".join(name for name, _ in parts)
    numbers = " + ".join(_n(v, _EH, system) for _, v in parts)
    if len(parts) > 2 or r"\times" in numbers:
        rows = [rf"{symbol} &= {names}"]
        for k, (_, v) in enumerate(parts):
            lead = "= " if k == 0 else r"\quad + "
            rows.append(rf"&{lead}{_n(v, _EH, system)}")
        rows.append(rf"&= {_q(total, _EH, system)}")
        return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"
    return latex_chain(symbol, names, numbers, _q(total, _EH, system))


def _net_step(result: GasTurbineResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    lines = []
    comp = [(rf"w_{{C,{k + 1}}}", result.stage_work(c)) for k, c in enumerate(result.compressors)]
    turb = [(rf"w_{{T,{k + 1}}}", result.stage_work(t)) for k, t in enumerate(result.turbines)]
    for line in (
        _sum_line("w_C", comp, result.w_compressor_J_per_kg, system),
        _sum_line("w_T", turb, result.w_turbine_J_per_kg, system),
    ):
        if line:
            lines.append(line)
    lines.append(
        latex_chain(
            r"w_{\mathrm{neto}}",
            "w_T - w_C",
            _diff(result.w_turbine_J_per_kg, result.w_compressor_J_per_kg, _EH, system),
            _q(result.w_net_J_per_kg, _EH, system),
        )
    )
    if inputs.fuel is not None:
        F = "f" if len(result.combustors) == 1 else "F"
        if len(result.combustors) > 1:
            fs = " + ".join(rf"f_{{{k + 1}}}" for k in range(len(result.combustors)))
            numbers = [latex_number(cc.fuel, 5) for cc in result.combustors]
            values = (
                " + ".join(numbers)
                if len(numbers) == 2
                else rf"{numbers[0]} + {numbers[1]} \\ &\quad + " + " + ".join(numbers[2:])
            )
            lines.append(latex_chain("F", fs, values, latex_number(result.fuel_air_ratio, 5)))
        lines.append(
            latex_chain(
                r"q_{\mathrm{ent}}",
                rf"{F}\,\mathrm{{PCI}}",
                rf"{latex_number(result.fuel_air_ratio, 5)}\cdot "
                rf"{_n(inputs.fuel.lhv_J_per_kg, _EH, system)}",
                _q(result.q_in_J_per_kg, _EH, system),
            )
        )
    else:
        heaters = [
            (r"q_{\mathrm{ent}}" if len(result.combustors) == 1 else rf"q_{{{k + 1}}}", cc.q)
            for k, cc in enumerate(result.combustors)
        ]
        line = _sum_line(r"q_{\mathrm{ent}}", heaters, result.q_in_J_per_kg, system)
        if line:
            lines.append(line)
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
    text = (
        "Todo por kilogramo de aire. La relación de trabajo de retroceso r_bw es la parte del "
        "trabajo de la turbina que se lleva el compresor (Cengel §9-8)."
    )
    if inputs.intercooled and inputs.reheated and inputs.regenerated:
        lines.append(
            latex_chain(
                r"\eta_{\mathrm{Carnot}}",
                r"1 - \frac{T_{\min}}{T_{\max}}",
                rf"1 - \frac{{{latex_number(result.T_min_K, 5)}}}"
                rf"{{{latex_number(result.T_max_K, 5)}}}",
                rf"{latex_number(result.eta_carnot * 100, 4)}\,\%",
            )
        )
        text += (
            " Con interenfriamiento, recalentamiento y regenerador, el ciclo se acerca al de "
            "Ericsson; el límite es el rendimiento de Carnot entre la menor y la mayor "
            "temperatura del ciclo (en K, Cengel §9-10)."
        )
    return ProcedureStep(title="Trabajo neto y rendimiento", text=text, latex=tuple(lines))


def _power_step(result: GasTurbineResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    m_a = _n(result.m_air_kg_s, "mass_flow", system)
    lines = []
    if inputs.W_net_W is not None:
        lines.append(
            latex_chain(
                r"\dot{m}_a",
                r"\frac{\dot{W}_{TG}}{w_{\mathrm{neto}}}",
                rf"\frac{{{_n(inputs.W_net_W, 'power', system)}}}"
                rf"{{{_n(result.w_net_J_per_kg, _EH, system)}}}",
                _q(result.m_air_kg_s, "mass_flow", system),
            )
        )
    if not inputs.air_standard:
        lines.append(
            latex_chain(
                r"\dot{m}_f",
                (r"f" if len(result.combustors) == 1 else "F") + r"\,\dot{m}_a",
                rf"{latex_number(result.fuel_air_ratio, 5)}\cdot {m_a}",
                _q(result.m_fuel_kg_s, "mass_flow", system),
            )
        )
    if inputs.W_net_W is None:
        lines.append(
            latex_chain(
                r"\dot{W}_{TG}",
                r"\dot{m}_a\,w_{\mathrm{neto}}",
                rf"{m_a}\cdot {_n(result.w_net_J_per_kg, _EH, system)}",
                _q(result.W_net_W, "power", system),
            )
        )
    lines.append(
        latex_chain(
            r"\dot{Q}_{\mathrm{ent}}",
            r"\dot{m}_a\,q_{\mathrm{ent}}",
            rf"{m_a}\cdot {_n(result.q_in_J_per_kg, _EH, system)}",
            _q(result.Q_in_W, "power", system),
        )
    )
    return ProcedureStep(
        title="Caudales y potencias",
        text=(
            "Con la potencia neta sale el caudal de aire, y con él todo lo demás."
            if inputs.W_net_W is not None
            else "Con el caudal de aire, todo escala: potencias, calor y caudal de combustible."
        ),
        latex=tuple(lines),
    )


# ---------------------------------------------------------------------
# Exergía
# ---------------------------------------------------------------------


def _component_symbol(result: GasTurbineResult, kind: str, k: int) -> str:
    """Subíndice de un componente: C, C1, T2, \text{cám}, \text{RH}1…"""
    if kind == "compresor":
        return "C" if len(result.compressors) == 1 else f"C{k + 1}"
    if kind == "turbina":
        return "T" if len(result.turbines) == 1 else f"T{k + 1}"
    if k == 0:
        return r"\text{cám}" if result.inputs.fuel is not None else r"\text{cal}"
    return r"\text{RH}" if len(result.combustors) == 2 else rf"\text{{RH}}{k}"


def _exergy_steps(result: GasTurbineResult, system: UnitSystem) -> list[ProcedureStep]:
    inputs = result.inputs
    ex = gas_turbine_exergy(result)
    T0 = ex.T0_K
    st = result.states
    t0 = latex_number(T0, 5)
    x = {i.name: i.x_J_per_kg for i in ex.items}
    many = len(result.combustors) > 1

    def ds(i: int, j: int) -> float:
        return st[j].s_J_per_kg_K - st[i].s_J_per_kg_K

    def psi(i: int) -> float:
        return st[i].psi(T0, inputs.p_amb_Pa)

    lines: list[str] = []
    # Lo que entra.
    if inputs.fuel is not None:
        F = "F" if many else "f"
        lines.append(
            latex_chain(
                r"x_{\mathrm{ent}}", rf"{F}\,\mathrm{{PCI}}", _q(ex.x_in_J_per_kg, _EH, system)
            )
        )
    else:
        for k, cc in enumerate(result.combustors):
            o = st[cc.outlet]
            q = r"q_{\mathrm{ent}}" if not many else rf"q_{{{k + 1}}}"
            lines.append(
                latex_chain(
                    r"x_Q" if not many else rf"x_{{Q,{k + 1}}}",
                    rf"{q}\left(1 - \frac{{T_0}}{{T_{{{o.label}}}}}\right)",
                    rf"{_n(cc.q, _EH, system)}"
                    rf"\left(1 - \frac{{{t0}}}{{{latex_number(o.T_K, 5)}}}\right)",
                    _q(cc.q * (1.0 - T0 / o.T_K), _EH, system),
                )
            )
    # Compresores y turbinas: T₀·Δs (por kg de aire).
    for kind, group in (("compresor", result.compressors), ("turbina", result.turbines)):
        for k, stage in enumerate(group):
            a, b = st[stage.inlet], st[stage.outlet]
            if stage.m == 1.0:
                factor, value = "", ""
            else:
                F = rf"F_{{{k + 1}}}" if many else "f"
                factor = rf"(1 + {F})\,"
                value = rf"{latex_number(stage.m, 6)}\cdot "
            lines.append(
                latex_chain(
                    rf"x_{{d,{_component_symbol(result, kind, k)}}}",
                    rf"{factor}T_0\,(s_{{{b.label}}} - s_{{{a.label}}})",
                    rf"{value}{t0}\cdot "
                    rf"{latex_paren(_n(ds(stage.inlet, stage.outlet), _ES, system))}",
                    _q(x[stage.name], _EH, system),
                )
            )
    # Interenfriadores: se pierde lo que pierde el aire.
    for k, ic in enumerate(result.intercoolers):
        a, b = st[ic.inlet], st[ic.outlet]
        lines.append(
            latex_chain(
                r"x_{IC}" if len(result.intercoolers) == 1 else rf"x_{{IC,{k + 1}}}",
                rf"\psi_{{{a.label}}} - \psi_{{{b.label}}}",
                _diff(psi(ic.inlet), psi(ic.outlet), _EH, system),
                _q(x[ic.name], _EH, system),
            )
        )
    # Regenerador.
    rg = result.regenerator
    if rg is not None:
        n2, n5, n4, n6 = (st[i].label for i in (rg.air_in, rg.air_out, rg.gas_in, rg.gas_out))
        if rg.m_gas == 1.0:
            factor, value = "", ""
        else:
            factor = r"(1 + F)\," if many else r"(1 + f)\,"
            value = rf"{latex_number(rg.m_gas, 6)}\cdot "
        lines.append(
            latex_chain(
                r"x_{d,\mathrm{reg}}",
                rf"T_0\,[s_{{{n5}}} - s_{{{n2}}} \\ &\quad + {factor}(s_{{{n6}}} - s_{{{n4}}})]",
                rf"{t0}\,[{_n(ds(rg.air_in, rg.air_out), _ES, system)} \\ &\quad + {value}"
                rf"{latex_paren(_n(ds(rg.gas_in, rg.gas_out), _ES, system))}]",
                _q(x["regenerador"], _EH, system),
            )
        )
    # Cámaras: lo que no sale de su balance (un término por renglón).
    for k, cc in enumerate(result.combustors):
        a, b = st[cc.inlet], st[cc.outlet]
        if inputs.fuel is not None:
            x_heat = cc.fuel * inputs.fuel.lhv_J_per_kg
            heat = (rf"f_{{{k + 1}}}" if many else "f") + r"\,\mathrm{PCI}"
        else:
            x_heat = cc.q * (1.0 - T0 / b.T_K)
            heat = rf"x_{{Q,{k + 1}}}" if many else "x_Q"
        if cc.m_in == 1.0:
            m_in, v_in = "", ""
        else:
            m_in, v_in = rf"(1 + F_{{{k}}})\,", rf"{latex_number(cc.m_in, 6)}\cdot "
        if cc.m_out == 1.0:
            m_out, v_out = "", ""
        else:
            F_out = rf"F_{{{k + 1}}}" if many else "f"
            m_out, v_out = rf"(1 + {F_out})\,", rf"{latex_number(cc.m_out, 6)}\cdot "
        lines.append(
            latex_chain(
                rf"x_{{d,{_component_symbol(result, 'cámara', k)}}}",
                rf"{m_in}\psi_{{{a.label}}} + {heat} \\ &\quad - {m_out}\psi_{{{b.label}}}",
                rf"{v_in}{latex_paren(_n(psi(cc.inlet), _EH, system))} "
                rf"\\ &\quad + {_n(x_heat, _EH, system)} "
                rf"\\ &\quad - {v_out}{latex_paren(_n(psi(cc.outlet), _EH, system))}",
                _q(x[cc.name], _EH, system),
            )
        )
    # Escape.
    exh = st[result.exhaust]
    if result.fuel_air_ratio > 0.0:
        factor = r"(1 + F)\," if many else r"(1 + f)\,"
        value = rf"{latex_number(1.0 + result.fuel_air_ratio, 6)}\cdot "
    else:
        factor, value = "", ""
    lines.append(
        latex_chain(
            r"x_{\mathrm{esc}}",
            rf"{factor}\psi_{{{exh.label}}}",
            rf"{value}{_n(psi(result.exhaust), _EH, system)}",
            _q(x["escape"], _EH, system),
        )
    )
    first = ProcedureStep(
        title="Exergía destruida y perdida en cada componente",
        text=(
            f"Con el estado muerto en el ambiente (T₀ = {_degC(T0)}, "
            f"p₀ = {_bar(inputs.p_amb_Pa)}), cada corriente lleva la exergía de flujo "
            "ψ = (h − h₀) − T₀·(s − s₀) (vademecum §11.3), con h₀ y s₀ de la misma mezcla a T₀ "
            "y p₀. "
            + (
                "La exergía del combustible se toma igual a su PCI (vademecum §16.13). "
                if inputs.fuel is not None
                else "Con aire estándar entra la exergía del calor, Q·(1 − T₀/T), con T la "
                "temperatura a la que sale el aire de cada calentador (vademecum §11.5). "
            )
            + "Compresores, turbinas y regenerador son adiabáticos: destruyen T₀·S_gen "
            "(vademecum §11.8). Cada cámara destruye lo que no sale de su balance. El calor de "
            "los interenfriadores y la exergía del escape se pierden en el ambiente. Todo por "
            "kilogramo de aire."
        ),
        latex=tuple(lines),
    )
    destroyed = ex.total("destruida")
    lost = ex.total("perdida")
    balance = [
        latex_chain(
            r"x_{\mathrm{ent}}",
            r"w_{\mathrm{neto}} + \textstyle\sum x_d + \sum x_{\text{pérd}}",
            _wrap(
                _n(result.w_net_J_per_kg, _EH, system),
                "+",
                rf"{_n(destroyed, _EH, system)} + {_n(lost, _EH, system)}",
            ),
            _q(ex.x_in_J_per_kg, _EH, system),
        ),
        latex_chain(
            r"\eta_{II}",
            r"\frac{w_{\mathrm{neto}}}{x_{\mathrm{ent}}}",
            rf"\frac{{{_n(result.w_net_J_per_kg, _EH, system)}}}"
            rf"{{{_n(ex.x_in_J_per_kg, _EH, system)}}}",
            rf"{latex_number(ex.efficiency * 100, 4)}\,\%",
        ),
    ]
    biggest = max((i for i in ex.items if i.kind != "útil"), key=lambda i: i.x_J_per_kg)
    second = ProcedureStep(
        title="Balance de exergía",
        text=(
            "Lo que entra sale como trabajo, se destruye o se pierde (vademecum §11.9): el "
            "balance cierra. El rendimiento exergético compara el trabajo con la exergía que entra "
            f"(§11.10). La parte más grande es la {of_component(biggest.name)} "
            f"({biggest.x_J_per_kg / ex.x_in_J_per_kg * 100:.3g} % de lo que entra)"
            + (
                ": la combustión es muy irreversible, porque el combustible se quema con un salto "
                "de temperatura enorme (Cengel §9-12)."
                if biggest.name.startswith("cámara")
                else "."
            )
            + (
                " Con la exergía del combustible ≈ PCI, η_II coincide con η."
                if inputs.fuel is not None
                else ""
            )
        ),
        latex=tuple(balance),
    )
    return [first, second]


# ---------------------------------------------------------------------
# Todo
# ---------------------------------------------------------------------


def gas_turbine_steps(result: GasTurbineResult, system: UnitSystem) -> list[ProcedureStep]:
    """Procedimiento completo de la turbina de gas, como se resuelve a mano."""
    inputs = result.inputs
    steps = [_air_step(result, system)]
    pressures = _pressures_step(result, system)
    if pressures is not None:
        steps.append(pressures)
    single = len(result.compressors) == 1
    for k, c in enumerate(result.compressors):
        steps.append(_compressor_step(result, c, system, single))
        if k < len(result.intercoolers):
            steps.append(_intercooler_step(result, k, system))
    if inputs.air_standard:
        steps += [_turbine_step(result, t, system) for t in result.turbines]
        if result.regenerator is not None:
            steps.append(_regenerator_step(result, system))
        steps += [_heater_step(result, cc, system) for cc in result.combustors]
    else:
        _, f_t = _fuel_lines(result, system)
        for j, t in enumerate(result.turbines):
            steps += _combustor_step(result, j, f_t, system)
            steps.append(_turbine_step(result, t, system))
        if result.regenerator is not None:
            steps.append(_regenerator_step(result, system))
    steps += [_net_step(result, system), _power_step(result, system)]
    steps += _exergy_steps(result, system)
    return steps
