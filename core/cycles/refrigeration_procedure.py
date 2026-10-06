"""Procedimiento del ciclo de refrigeración «como con las tablas» — Fase 3.2.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo, en el orden de las soluciones de Çengel & Boles
(*Termodinámica*, 8.ª ed., cap. 11): primero los estados que se leen en las
tablas (salida del evaporador, compresión, salida del condensador,
estrangulamiento; la cámara y la mezcla o el intercambiador de la cascada),
después los calores, el trabajo y el COP por kilogramo que pasa por el
condensador, la comparación con Carnot, las potencias y, si se dieron las
fuentes, la exergía destruida en cada componente.

Fórmulas del vademecum: §3.3 (balance en sistema abierto), §9.3 (coeficientes
de operación), §10.4 (rendimiento isoentrópico del compresor), §11.8 (trabajo
perdido) y §11.10 (rendimiento exergético), §12 (vapor húmedo) y §13
(líquidos). Recibe un resultado ya calculado
(:class:`core.cycles.refrigeration.RefrigerationResult`): no importa TESPy ni
Streamlit. Las ecuaciones se arman con :func:`core.latex.latex_chain` (una
igualdad por renglón) para que entren en el ancho de un celular.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.cycles.rankine_procedure import (
    _absolute_T,
    _bar,
    _diff,
    _ix,
    _n,
    _q,
    _sub,
    _sum_lines,
    _wrap,
)
from core.fluids import (
    FLUID_NAMES_ES,
    _el,
    fluid_limits,
    saturation_at_pressure,
    saturation_at_temperature,
    textbook_reference_offset,
)
from core.latex import latex_chain, latex_number, latex_paren
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

if TYPE_CHECKING:
    from core.cycles.refrigeration import RefrigerationComponent, RefrigerationResult

_EH: QuantityKind = "specific_enthalpy"
_ES: QuantityKind = "specific_entropy"
_TABULATED = "R134a"


# ---------------------------------------------------------------------
# Dónde se lee cada propiedad
# ---------------------------------------------------------------------


def _table(fluid: str, which: str) -> str:
    """La tabla de Cengel que corresponde (solo el R-134a) o la ecuación de estado.

    ``which``: ``"sat_p"`` (saturación por presión), ``"sat_T"`` (saturación
    por temperatura) o ``"superheated"`` (vapor sobrecalentado).
    """
    if fluid == _TABULATED:
        return {
            "sat_p": "la tabla de saturación por presión del R-134a (Cengel A-12)",
            "sat_T": "la tabla de saturación por temperatura del R-134a (Cengel A-11)",
            "superheated": "la tabla de vapor sobrecalentado del R-134a (Cengel A-13)",
        }[which]
    return f"la ecuación de estado (CoolProp; Cengel no tabula {_el(fluid)})"


def _read(fluid: str, which: str, *, plural: bool = False, interpolate: bool = False) -> str:
    """«se lee en la tabla …» o «sale de la ecuación de estado»."""
    if fluid == _TABULATED:
        verb = "se leen en" if plural else "se lee en"
        return f"{verb} {_table(fluid, which)}" + (", interpolando" if interpolate else "")
    verb = "salen de" if plural else "sale de"
    return f"{verb} {_table(fluid, which)}"


def _pure(fluid: str) -> bool:
    return fluid_limits(fluid).is_pure


def _T_sat_symbol(fluid: str, phase: str) -> str:
    """``T_sat`` en un fluido puro; ``T_g`` (rocío) o ``T_f`` (burbuja) en una mezcla."""
    return r"T_{\mathrm{sat}}" if _pure(fluid) else f"T_{phase}"


def _fluid(result: RefrigerationResult, i: int) -> str:
    return result.layout.fluids[i]


def _reference_text(fluid: str, system: UnitSystem) -> str:
    """Aclaración sobre la referencia de h y s de las tablas de Cengel (R-134a)."""
    offset = textbook_reference_offset(fluid)
    if offset is None:
        return ""
    dh, ds = offset
    return (
        " Ojo: CoolProp usa otra referencia que las tablas del libro, así que h y s salen "
        f"{convert_from_si(dh, _EH, system):.5g} {unit_label(_EH, system)} y "
        f"{convert_from_si(ds, _ES, system):.4g} {unit_label(_ES, system)} más altas; las "
        "diferencias (y todo lo que sigue) son iguales."
    )


# ---------------------------------------------------------------------
# Estados que se leen en las tablas
# ---------------------------------------------------------------------


def _evaporator_outlet_step(result: RefrigerationResult, system: UnitSystem) -> ProcedureStep:
    """Estado 1: vapor saturado o sobrecalentado a la presión de aspiración."""
    i = result.evaporator.outlet
    s = result.states[i]
    fluid = _fluid(result, i)
    n = _ix(i + 1)
    sh = result.inputs.superheat_K
    T_sym = _T_sat_symbol(fluid, "g")
    where = f"a la presión del evaporador, {_bar(s.P_Pa)}"
    if sh == 0.0:
        lines = (
            rf"T_{n} = {T_sym}(p_{n}) = {_q(s.T_K, 'temperature', system)}",
            rf"h_{n} = h_g(p_{n}) = {_q(s.h_J_per_kg, _EH, system)}",
            rf"s_{n} = s_g(p_{n}) = {_q(s.s_J_per_kg_K, _ES, system)}",
        )
        text = (
            f"Del evaporador sale vapor saturado (x{_sub(i + 1)} = 1) {where}: T, h y s "
            f"{_read(fluid, 'sat_p', plural=True)}."
        )
    else:
        T_sat = saturation_at_pressure(fluid, s.P_Pa).vapor.T_K
        lines = (
            latex_chain(
                f"T_{n}",
                rf"{T_sym}(p_{n}) + \Delta T_{{\mathrm{{sob}}}}",
                rf"{_n(T_sat, 'temperature', system)} + "
                rf"{_n(sh, 'temperature_difference', system)}",
                _q(s.T_K, "temperature", system),
            ),
            rf"h_{n} = h(p_{n},\ T_{n}) = {_q(s.h_J_per_kg, _EH, system)}",
            rf"s_{n} = s(p_{n},\ T_{n}) = {_q(s.s_J_per_kg_K, _ES, system)}",
        )
        dew = "saturación" if _pure(fluid) else "rocío"
        text = (
            f"El vapor sale del evaporador {where}, sobrecalentado {sh:.3g} K sobre la "
            f"temperatura de {dew} (así no llegan gotas al compresor; Cengel §11-4): h y s "
            f"{_read(fluid, 'superheated', plural=True, interpolate=True)}."
        )
    return ProcedureStep(
        title=f"Estado {i + 1}: {result.layout.labels[i]}",
        text=text + _reference_text(fluid, system),
        latex=lines,
    )


def _saturated_step(
    result: RefrigerationResult, i: int, phase: str, system: UnitSystem, text: str
) -> ProcedureStep:
    """Un estado saturado: líquido (``phase="f"``) o vapor (``"g"``) a su presión."""
    s = result.states[i]
    fluid = _fluid(result, i)
    n = _ix(i + 1)
    T_sym = _T_sat_symbol(fluid, phase)
    return ProcedureStep(
        title=f"Estado {i + 1}: {result.layout.labels[i]}",
        text=f"{text} T, h y s {_read(fluid, 'sat_p', plural=True)}.",
        latex=(
            rf"T_{n} = {T_sym}(p_{n}) = {_q(s.T_K, 'temperature', system)}",
            rf"h_{n} = h_{phase}(p_{n}) = {_q(s.h_J_per_kg, _EH, system)}",
            rf"s_{n} = s_{phase}(p_{n}) = {_q(s.s_J_per_kg_K, _ES, system)}",
        ),
    )


def _condenser_outlet_step(result: RefrigerationResult, system: UnitSystem) -> ProcedureStep:
    """Salida del condensador: líquido saturado o subenfriado, con su caída de presión."""
    comp = result.condenser
    i, j = comp.outlet, comp.inlet
    s = result.states[i]
    fluid = _fluid(result, i)
    n, m = _ix(i + 1), _ix(j + 1)
    inputs = result.inputs
    lines: list[str] = []
    clauses: list[str] = []
    if inputs.dp_cond_Pa > 0.0:
        lines.append(
            latex_chain(
                f"p_{n}",
                rf"p_{m} - \Delta p_{{\mathrm{{cond}}}}",
                _diff(inputs.p_cond_Pa, inputs.dp_cond_Pa, "pressure", system),
                _q(s.P_Pa, "pressure", system),
            )
        )
        clauses.append(
            f"por la fricción, el líquido sale a p{_sub(i + 1)} = {_bar(s.P_Pa)}, "
            f"{_bar(inputs.dp_cond_Pa)} menos que la descarga del compresor"
        )
    title = f"Estado {i + 1}: {result.layout.labels[i]}"
    T_sym = _T_sat_symbol(fluid, "f")
    if inputs.subcooling_K == 0.0:
        lines += [
            rf"T_{n} = {T_sym}(p_{n}) = {_q(s.T_K, 'temperature', system)}",
            rf"h_{n} = h_f(p_{n}) = {_q(s.h_J_per_kg, _EH, system)}",
            rf"s_{n} = s_f(p_{n}) = {_q(s.s_J_per_kg_K, _ES, system)}",
        ]
        head = (f"{clauses[0][0].upper()}{clauses[0][1:]} (Cengel §11-4). " if clauses else "") + (
            f"Del condensador sale líquido saturado (x{_sub(i + 1)} = 0)"
            + ("" if clauses else f" a {_bar(s.P_Pa)}")
        )
        return ProcedureStep(
            title=title,
            text=f"{head}: T, h y s {_read(fluid, 'sat_p', plural=True)}.",
            latex=tuple(lines),
        )
    sat = saturation_at_pressure(fluid, s.P_Pa)
    liquid = saturation_at_temperature(fluid, s.T_K).liquid
    sub = inputs.subcooling_K
    lines += [
        latex_chain(
            f"T_{n}",
            rf"{T_sym}(p_{n}) - \Delta T_{{\mathrm{{sub}}}}",
            rf"{_n(sat.liquid.T_K, 'temperature', system)} - "
            rf"{_n(sub, 'temperature_difference', system)}",
            _q(s.T_K, "temperature", system),
        ),
        rf"h_{n} = h(p_{n},\ T_{n}) = {_q(s.h_J_per_kg, _EH, system)}",
        rf"h_f(T_{n}) = {_q(liquid.h_J_per_kg, _EH, system)}",
        rf"s_{n} = s(p_{n},\ T_{n}) = {_q(s.s_J_per_kg_K, _ES, system)}",
    ]
    clauses.append(
        f"el líquido sale subenfriado {sub:.3g} K por debajo de la temperatura de "
        + ("saturación" if _pure(fluid) else "burbuja")
    )
    text = (
        f"{' y '.join(clauses)[0].upper()}{' y '.join(clauses)[1:]} (Cengel §11-4). Es "
        "líquido comprimido: h casi no depende de la presión y se aproxima con la del "
        f"líquido saturado a la misma temperatura, h ≈ h_f(T), de {_table(fluid, 'sat_T')} "
        "(vademecum §13); con la ecuación de estado, h(p, T) da casi lo mismo."
    )
    return ProcedureStep(title=title, text=text, latex=tuple(lines))


def _compressor_step(result: RefrigerationResult, k: int, system: UnitSystem) -> ProcedureStep:
    """Compresión a → b: estado isoentrópico (palanca si cae en la campana) y real."""
    comp = result.compressors[k]
    a, b = comp.inlet, comp.outlet
    s_in, s_out = result.states[a], result.states[b]
    s_is = result.compressors_out_s[k]
    fluid = _fluid(result, b)
    na, nb = a + 1, b + 1
    eta = result.inputs.eta_compressor
    lines = [rf"s_{{{nb}s}} = s_{_ix(na)} = {_q(s_in.s_J_per_kg_K, _ES, system)}"]
    sat = (
        saturation_at_pressure(fluid, s_out.P_Pa)
        if s_out.P_Pa < fluid_limits(fluid).P_crit_Pa
        else None
    )
    wet = sat is not None and sat.liquid.s_J_per_kg_K <= s_in.s_J_per_kg_K <= sat.vapor.s_J_per_kg_K
    if wet:
        assert sat is not None
        x_s = (s_in.s_J_per_kg_K - sat.liquid.s_J_per_kg_K) / sat.s_fg_J_per_kg_K
        lines.append(
            latex_chain(
                rf"x_{{{nb}s}}",
                r"\frac{s - s_f}{s_{fg}}",
                rf"\frac{{{_n(s_in.s_J_per_kg_K, _ES, system)} - "
                rf"{latex_paren(_n(sat.liquid.s_J_per_kg_K, _ES, system))}}}"
                rf"{{{_n(sat.s_fg_J_per_kg_K, _ES, system)}}}",
                latex_number(x_s, 5),
            )
        )
        lines.append(
            latex_chain(
                rf"h_{{{nb}s}}",
                rf"h_f + x_{{{nb}s}}\,h_{{fg}}",
                _wrap(
                    _n(sat.liquid.h_J_per_kg, _EH, system),
                    "+",
                    rf"{latex_number(x_s, 5)}\cdot {_n(sat.h_fg_J_per_kg, _EH, system)}",
                ),
                _q(s_is.h_J_per_kg, _EH, system),
            )
        )
        how = (
            f"A {_bar(s_out.P_Pa)} la entropía de la aspiración cae dentro de la campana: la "
            f"compresión isoentrópica termina en vapor húmedo, con el título de la regla de la "
            f"palanca (vademecum §12)."
        )
    else:
        lines.append(
            latex_chain(
                rf"h_{{{nb}s}}",
                rf"h(p_{_ix(nb)},\ s_{{{nb}s}})",
                _q(s_is.h_J_per_kg, _EH, system),
            )
        )
        how = (
            f"A {_bar(s_out.P_Pa)} el estado isoentrópico {nb}s es vapor sobrecalentado: "
            f"{_read(fluid, 'superheated', interpolate=True)}."
        )
    if eta == 1.0:
        lines.append(rf"h_{_ix(nb)} = h_{{{nb}s}} = {_q(s_out.h_J_per_kg, _EH, system)}")
        real = "Compresor ideal (isoentrópico): el estado real es el isoentrópico (Cengel §11-3)."
    else:
        lines.append(
            latex_chain(
                rf"h_{_ix(nb)}",
                rf"h_{_ix(na)} + \frac{{h_{{{nb}s}} - h_{_ix(na)}}}{{\eta_C}}",
                _wrap(
                    _n(s_in.h_J_per_kg, _EH, system),
                    "+",
                    rf"\frac{{{_diff(s_is.h_J_per_kg, s_in.h_J_per_kg, _EH, system)}}}"
                    rf"{{{latex_number(eta, 4)}}}",
                ),
                _q(s_out.h_J_per_kg, _EH, system),
            )
        )
        real = (
            f"Con η_C = {eta:.4g}, el compresor real consume más trabajo que el ideal y el vapor "
            "sale más caliente (vademecum §10.4)."
        )
    lines.append(
        rf"T_{_ix(nb)} = T(p_{_ix(nb)},\ h_{_ix(nb)}) = {_q(s_out.T_K, 'temperature', system)}"
    )
    lines.append(
        latex_chain(
            rf"w_{{C,{na}{nb}}}",
            rf"h_{_ix(nb)} - h_{_ix(na)}",
            _diff(s_out.h_J_per_kg, s_in.h_J_per_kg, _EH, system),
            _q(s_out.h_J_per_kg - s_in.h_J_per_kg, _EH, system),
        )
    )
    return ProcedureStep(
        title=f"{na} → {nb}: {comp.label}",
        text=(
            f"El {comp.label} lleva el vapor de {_bar(s_in.P_Pa)} a {_bar(s_out.P_Pa)}. {how} "
            f"{real} Balance de sistema abierto adiabático (vademecum §3.3): w = h_sal − h_ent."
        ),
        latex=tuple(lines),
    )


def _valve_step(
    result: RefrigerationResult, comp: RefrigerationComponent, system: UnitSystem
) -> ProcedureStep:
    """Estrangulamiento a → b: h constante; título por la regla de la palanca."""
    a, b = comp.inlet, comp.outlet
    s_in, s_out = result.states[a], result.states[b]
    fluid = _fluid(result, b)
    na, nb = _ix(a + 1), _ix(b + 1)
    inputs = result.inputs
    lines: list[str] = []
    pressure = f"a {_bar(s_out.P_Pa)}"
    if b == result.evaporator.inlet and inputs.dp_evap_Pa > 0.0:
        j = _ix(result.evaporator.outlet + 1)
        lines.append(
            latex_chain(
                f"p_{nb}",
                rf"p_{j} + \Delta p_{{\mathrm{{evap}}}}",
                rf"{_n(inputs.p_evap_Pa, 'pressure', system)} + "
                rf"{_n(inputs.dp_evap_Pa, 'pressure', system)}",
                _q(s_out.P_Pa, "pressure", system),
            )
        )
        pressure = (
            f"a {_bar(s_out.P_Pa)}: la válvula descarga {_bar(inputs.dp_evap_Pa)} por encima de "
            "la aspiración, porque el refrigerante pierde presión en el evaporador"
        )
    lines.append(rf"h_{nb} = h_{na} = {_q(s_in.h_J_per_kg, _EH, system)}")
    x = s_out.x
    if x is not None:
        sat = saturation_at_pressure(fluid, s_out.P_Pa)
        lines.append(
            latex_chain(
                f"x_{nb}",
                r"\frac{h - h_f}{h_{fg}}",
                rf"\frac{{{_diff(s_out.h_J_per_kg, sat.liquid.h_J_per_kg, _EH, system)}}}"
                rf"{{{_n(sat.h_fg_J_per_kg, _EH, system)}}}",
                latex_number(x, 5),
            )
        )
        if _pure(fluid):
            lines.append(
                rf"T_{nb} = T_{{\mathrm{{sat}}}}(p_{nb}) = {_q(s_out.T_K, 'temperature', system)}"
            )
        else:
            lines.append(rf"T_{nb} = T(p_{nb},\ x_{nb}) = {_q(s_out.T_K, 'temperature', system)}")
        lines.append(
            latex_chain(
                f"s_{nb}",
                rf"s_f + x_{nb}\,s_{{fg}}",
                _q(s_out.s_J_per_kg_K, _ES, system),
            )
        )
        state = (
            f"Sale vapor húmedo: el título sale de la regla de la palanca con h_f y h_fg de "
            f"{_table(fluid, 'sat_p')} (vademecum §12)"
            + ("" if _pure(fluid) else "; en una mezcla, T queda entre las de burbuja y rocío")
            + "."
        )
    else:
        lines.append(rf"T_{nb} = T(p_{nb},\ h_{nb}) = {_q(s_out.T_K, 'temperature', system)}")
        lines.append(rf"s_{nb} = s(p_{nb},\ h_{nb}) = {_q(s_out.s_J_per_kg_K, _ES, system)}")
        state = "El líquido llega tan subenfriado que sigue líquido después de la válvula."
    return ProcedureStep(
        title=f"{a + 1} → {b + 1}: {comp.label}",
        text=(
            f"La {comp.label} estrangula el líquido hasta {pressure}. Sin calor ni trabajo, el "
            "balance de sistema abierto (vademecum §3.3) da h constante: h_sal = h_ent. "
            f"{state}"
        ),
        latex=tuple(lines),
    )


def _flash_tank_step(
    result: RefrigerationResult, comp: RefrigerationComponent, system: UnitSystem
) -> ProcedureStep:
    """Cámara de evaporación instantánea: vapor y líquido saturados a su presión."""
    i = comp.inlet
    vapor, liquid = comp.outlets
    fluid = _fluid(result, i)
    nv, nl = _ix(vapor + 1), _ix(liquid + 1)
    sv, sl = result.states[vapor], result.states[liquid]
    x = result.flash_fraction or 0.0
    return ProcedureStep(
        title=f"Cámara de evaporación instantánea ({i + 1} → {vapor + 1} y {liquid + 1})",
        text=(
            f"En la cámara, a {_bar(sv.P_Pa)}, se separan las fases: por cada kilogramo que "
            f"llega (estado {i + 1}), x{_sub(i + 1)} = {x:.4f} sale como vapor saturado "
            f"({vapor + 1}) hacia la mezcla y 1 − x{_sub(i + 1)} como líquido saturado "
            f"({liquid + 1}) hacia la válvula de baja (Cengel §11-8). Sus propiedades "
            f"{_read(fluid, 'sat_p', plural=True)}."
        ),
        latex=(
            rf"h_{nv} = h_g(p_{nv}) = {_q(sv.h_J_per_kg, _EH, system)}",
            rf"s_{nv} = s_g(p_{nv}) = {_q(sv.s_J_per_kg_K, _ES, system)}",
            rf"h_{nl} = h_f(p_{nl}) = {_q(sl.h_J_per_kg, _EH, system)}",
            rf"s_{nl} = s_f(p_{nl}) = {_q(sl.s_J_per_kg_K, _ES, system)}",
        ),
    )


def _x_flash_tex(result: RefrigerationResult) -> str:
    (tank,) = result.of_kind("flash_tank")
    return f"x_{_ix(tank.inlet + 1)}"


def _mixer_step(
    result: RefrigerationResult, comp: RefrigerationComponent, system: UnitSystem
) -> ProcedureStep:
    """Cámara de mezcla adiabática: h de la mezcla por balance de energía."""
    from_compressor, from_tank = comp.inlets
    o = comp.outlet
    st = result.states
    x = result.flash_fraction or 0.0
    xs = _x_flash_tex(result)
    nc, nt, no = _ix(from_compressor + 1), _ix(from_tank + 1), _ix(o + 1)
    return ProcedureStep(
        title=f"Mezcla ({from_compressor + 1} + {from_tank + 1} → {o + 1})",
        text=(
            "Balance de energía de la cámara de mezcla (adiabática y sin trabajo; vademecum "
            "§3.3), por kilogramo que pasa por el condensador: llegan "
            f"{xs.replace('_', '')} kg de vapor saturado de la cámara ({from_tank + 1}) y "
            f"1 − {xs.replace('_', '')} kg del compresor de baja ({from_compressor + 1}). La "
            "mezcla enfría el vapor que va al compresor de alta."
        ),
        latex=(
            latex_chain(
                f"h_{no}",
                rf"{xs}\,h_{nt} + (1 - {xs})\,h_{nc}",
                _wrap(
                    rf"{latex_number(x, 5)}\cdot {_n(st[from_tank].h_J_per_kg, _EH, system)}",
                    "+",
                    rf"{latex_number(1.0 - x, 5)}\cdot "
                    rf"{_n(st[from_compressor].h_J_per_kg, _EH, system)}",
                ),
                _q(st[o].h_J_per_kg, _EH, system),
            ),
            rf"s_{no} = s(p_{no},\ h_{no}) = {_q(st[o].s_J_per_kg_K, _ES, system)}",
            rf"T_{no} = T(p_{no},\ h_{no}) = {_q(st[o].T_K, 'temperature', system)}",
        ),
    )


def _cascade_hx_step(
    result: RefrigerationResult, comp: RefrigerationComponent, system: UnitSystem
) -> ProcedureStep:
    """Balance de energía del intercambiador: la relación de caudales ṁ_B/ṁ_A."""
    hot_in, cold_in = comp.inlets
    hot_out, cold_out = comp.outlets
    st = result.states
    h = [s.h_J_per_kg for s in st]
    hi, ho, ci, co = (_ix(k + 1) for k in (hot_in, hot_out, cold_in, cold_out))
    ratio = result.mass_ratio or 0.0
    return ProcedureStep(
        title="Intercambiador de la cascada: relación de caudales",
        text=(
            "El intercambiador es adiabático: lo que cede el ciclo de baja (B) al condensar lo "
            "recibe el de alta (A) al evaporar (Cengel §11-8, ejemplo 11-4). Del balance de "
            "energía (vademecum §3.3) sale cuántos kilogramos circulan por el ciclo de baja por "
            "cada kilogramo del de alta:"
        ),
        latex=(
            rf"\dot{{m}}_A\,(h_{co} - h_{ci}) = \dot{{m}}_B\,(h_{hi} - h_{ho})",
            latex_chain(
                r"\frac{\dot{m}_B}{\dot{m}_A}",
                rf"\frac{{h_{co} - h_{ci}}}{{h_{hi} - h_{ho}}}",
                rf"\frac{{{_diff(h[cold_out], h[cold_in], _EH, system)}}}"
                rf"{{{_diff(h[hot_in], h[hot_out], _EH, system)}}}",
                latex_number(ratio, 5),
            ),
        ),
    )


# ---------------------------------------------------------------------
# Calores, trabajo, COP, Carnot y potencias
# ---------------------------------------------------------------------


def _fraction_tex(result: RefrigerationResult, comp: RefrigerationComponent) -> str:
    """Caudal del componente por kg que pasa por el condensador, en símbolos."""
    fraction = result.m_dot_kg_s[comp.inlet] / result.m_cond_kg_s
    if abs(fraction - 1.0) < 1e-12:
        return ""
    if result.cycle == "flash":
        return rf"(1 - {_x_flash_tex(result)})\,"
    return r"\tfrac{\dot{m}_B}{\dot{m}_A}\,"


def _fraction_number(result: RefrigerationResult, comp: RefrigerationComponent) -> str:
    fraction = result.m_dot_kg_s[comp.inlet] / result.m_cond_kg_s
    if abs(fraction - 1.0) < 1e-12:
        return ""
    return rf"{latex_number(fraction, 5)}\,"


def _per_kg_terms(
    result: RefrigerationResult, kind: str, system: UnitSystem, *, sign: int = 1
) -> tuple[list[str], list[str]]:
    """Términos ``y·(h_sal − h_ent)`` (``sign = -1``: ``h_ent − h_sal``) con símbolos y números."""
    h = [s.h_J_per_kg for s in result.states]
    sym: list[str] = []
    num: list[str] = []
    for comp in result.components:
        if comp.kind != kind:
            continue
        i, o = comp.inlet, comp.outlet
        a, b = (o, i) if sign > 0 else (i, o)
        sym.append(f"{_fraction_tex(result, comp)}(h_{_ix(a + 1)} - h_{_ix(b + 1)})")
        num.append(f"{_fraction_number(result, comp)}({_diff(h[a], h[b], _EH, system)})")
    return sym, num


def _sum_or_chain(lhs: str, sym: list[str], num: list[str], value: str) -> str:
    if len(sym) == 1:
        return latex_chain(lhs, sym[0], num[0], value)
    return _sum_lines(lhs, sym, num, value)


def _basis_text(result: RefrigerationResult) -> str:
    if result.cycle == "flash":
        x = _x_flash_tex(result).replace("_", "")
        return (
            f" Por kilogramo que pasa por el condensador: por el evaporador y el compresor de "
            f"baja pasa 1 − {x}."
        )
    if result.cycle == "cascade":
        return (
            " Por kilogramo del ciclo de alta (A), que es el que pasa por el condensador: el "
            "ciclo de baja (B) pesa ṁ_B/ṁ_A."
        )
    return ""


def _heat_and_work_step(result: RefrigerationResult, system: UnitSystem) -> ProcedureStep:
    """q_C, w y q_H por kilogramo que pasa por el condensador; primer principio."""
    c_sym, c_num = _per_kg_terms(result, "evaporator", system)
    w_sym, w_num = _per_kg_terms(result, "compressor", system)
    h_sym, h_num = _per_kg_terms(result, "condenser", system, sign=-1)
    q_c, w, q_h = result.q_cold_J_per_kg, result.w_J_per_kg, result.q_hot_J_per_kg
    return ProcedureStep(
        title="Calores y trabajo",
        text=(
            "El evaporador saca q_C de la fuente fría, los compresores consumen w y el "
            "condensador entrega q_H a la fuente caliente (balances de sistema abierto, "
            "vademecum §3.3; Cengel usa q_L para q_C)."
            + _basis_text(result)
            + " El primer principio para el ciclo: q_H = q_C + w."
        ),
        latex=(
            _sum_or_chain("q_C", c_sym, c_num, _q(q_c, _EH, system)),
            _sum_or_chain("w", w_sym, w_num, _q(w, _EH, system)),
            _sum_or_chain("q_H", h_sym, h_num, _q(q_h, _EH, system)),
            latex_chain(
                "q_C + w",
                rf"{_n(q_c, _EH, system)} + {_n(w, _EH, system)}",
                _q(q_c + w, _EH, system),
            ),
        ),
    )


def _cop_step(result: RefrigerationResult, system: UnitSystem) -> ProcedureStep:
    q_c, w, q_h = result.q_cold_J_per_kg, result.w_J_per_kg, result.q_hot_J_per_kg
    cop_r = latex_chain(
        r"\mathrm{COP}_R",
        r"\frac{q_C}{w}",
        rf"\frac{{{_n(q_c, _EH, system)}}}{{{_n(w, _EH, system)}}}",
        latex_number(result.COP_R, 5),
    )
    cop_b = latex_chain(
        r"\mathrm{COP}_B",
        r"\frac{q_H}{w}",
        rf"\frac{{{_n(q_h, _EH, system)}}}{{{_n(w, _EH, system)}}}",
        latex_number(result.COP_B, 5),
    )
    if result.inputs.heat_pump:
        text = (
            "Una bomba de calor aprovecha el calor que entrega el condensador: COP_B = q_H/w "
            "(vademecum §9.3). Como q_H = q_C + w, COP_B = COP_R + 1 (Cengel §11-1)."
        )
        lines = (
            cop_b,
            cop_r,
            latex_chain(
                r"\mathrm{COP}_B",
                r"\mathrm{COP}_R + 1",
                rf"{latex_number(result.COP_R, 5)} + 1",
                latex_number(result.COP_R + 1.0, 5),
            ),
        )
    else:
        text = (
            "Un refrigerador aprovecha el calor que saca el evaporador: COP_R = q_C/w "
            "(vademecum §9.3). Usado como bomba de calor, el mismo ciclo daría COP_B = q_H/w = "
            "COP_R + 1."
        )
        lines = (cop_r, cop_b)
    return ProcedureStep(title="Coeficiente de operación (COP)", text=text, latex=lines)


def _carnot_step(result: RefrigerationResult, system: UnitSystem) -> ProcedureStep:
    T_l = _absolute_T(result.T_evap_K, system)
    T_h = _absolute_T(result.T_cond_K, system)
    heat_pump = result.inputs.heat_pump
    lhs = r"\mathrm{COP}_{B,\mathrm{Carnot}}" if heat_pump else r"\mathrm{COP}_{R,\mathrm{Carnot}}"
    num_sym = r"T_{\mathrm{cond}}" if heat_pump else r"T_{\mathrm{evap}}"
    num_val = T_h if heat_pump else T_l
    return ProcedureStep(
        title="Comparación con Carnot",
        text=(
            "Un ciclo de Carnot invertido que evaporara a T_evap y condensara a T_cond "
            "(temperaturas absolutas) daría el COP máximo entre esas temperaturas (vademecum "
            "§9.3). El ciclo de compresión de vapor queda abajo porque la válvula estrangula "
            "(irreversible) en vez de expandir en una turbina y porque el vapor sale "
            "sobrecalentado del compresor"
            + ("" if result.inputs.eta_compressor == 1.0 else ", además de la compresión real")
            + " (Cengel §11-3)."
        ),
        latex=(
            latex_chain(
                lhs,
                rf"\frac{{{num_sym}}}{{T_{{\mathrm{{cond}}}} - T_{{\mathrm{{evap}}}}}}",
                rf"\frac{{{num_val}}}{{{T_h} - {T_l}}}",
                latex_number(result.COP_carnot, 5),
            ),
        ),
    )


def _m_symbol(result: RefrigerationResult) -> str:
    return r"\dot{m}_A" if result.cycle == "cascade" else r"\dot{m}"


def _power_step(result: RefrigerationResult, system: UnitSystem) -> ProcedureStep:
    inputs = result.inputs
    m = _m_symbol(result)
    if inputs.capacity_W is not None:
        q_sym = "q_H" if inputs.heat_pump else "q_C"
        Q_sym = r"\dot{Q}_H" if inputs.heat_pump else r"\dot{Q}_C"
        q_val = result.q_hot_J_per_kg if inputs.heat_pump else result.q_cold_J_per_kg
        m_line = latex_chain(
            m,
            rf"\frac{{{Q_sym}}}{{{q_sym}}}",
            rf"\frac{{{_n(inputs.capacity_W, 'power', system)}}}{{{_n(q_val, _EH, system)}}}",
            _q(result.m_cond_kg_s, "mass_flow", system),
        )
        how = "El caudal sale de la capacidad pedida; con él, cada"
    else:
        m_line = rf"{m} = {_q(result.m_cond_kg_s, 'mass_flow', system)}"
        how = "Cada"
    lines = [
        m_line,
        latex_chain(r"\dot{Q}_C", rf"{m}\,q_C", _q(result.Q_cold_W, "power", system)),
        latex_chain(r"\dot{W}", rf"{m}\,w", _q(result.W_W, "power", system)),
        latex_chain(r"\dot{Q}_H", rf"{m}\,q_H", _q(result.Q_hot_W, "power", system)),
    ]
    if result.cycle == "flash":
        x = _x_flash_tex(result)
        lines.append(
            latex_chain(
                r"\dot{m}_{\mathrm{evap}}",
                rf"(1 - {x})\,\dot{{m}}",
                _q(result.m_evap_kg_s, "mass_flow", system),
            )
        )
    elif result.cycle == "cascade":
        lines.append(
            latex_chain(
                r"\dot{m}_B",
                r"\tfrac{\dot{m}_B}{\dot{m}_A}\,\dot{m}_A",
                _q(result.m_evap_kg_s, "mass_flow", system),
            )
        )
    first = result.compressors[0].inlet
    n1 = _ix(first + 1)
    lines.append(
        latex_chain(
            rf"\dot{{V}}_{n1}",
            rf"\dot{{m}}_{n1}\,v_{n1}",
            rf"{_n(result.m_dot_kg_s[first], 'mass_flow', system)}\cdot "
            rf"{_n(result.states[first].v_m3_per_kg, 'specific_volume', system)}",
            _q(result.suction_volume_flow_m3_s, "volume_flow", system),
        )
    )
    tons = (
        ""
        if inputs.heat_pump
        else (
            f" La capacidad de refrigeración equivale a {result.tons_of_refrigeration:.3g} "
            "toneladas de refrigeración (1 TR = 211 kJ/min, Cengel §11-1)."
        )
    )
    return ProcedureStep(
        title="Caudal y potencias",
        text=(
            f"{how} energía específica por el caudal que pasa por el condensador da la potencia."
            f"{tons} V̇{_sub(first + 1)} es el caudal volumétrico que aspira el compresor: a "
            "igual capacidad, un refrigerante que pide menos necesita un compresor más chico "
            "(Cengel §11-6)."
        ),
        latex=tuple(lines),
    )


def _comparison_step(result: RefrigerationResult) -> ProcedureStep | None:
    single = result.single_stage
    if single is None:
        return None
    gain = (result.COP / single.COP - 1.0) * 100.0
    what = "la cámara" if result.cycle == "flash" else "la cascada"
    return ProcedureStep(
        title="Comparación con el ciclo simple",
        text=(
            f"Un ciclo simple con {_el(single.fluid)}, entre las mismas temperaturas de "
            "evaporación y condensación y con el mismo compresor, tendría este COP; "
            f"{what} lo cambia en {gain:+.1f} % (Cengel §11-8). El ciclo simple necesita "
            f"una relación de presiones de {single.pressure_ratio:.3g} en un solo compresor."
        ),
        latex=(
            rf"\mathrm{{COP}}_{{\text{{simple}}}} = {latex_number(single.COP, 5)}"
            rf"\qquad \mathrm{{COP}} = {latex_number(result.COP, 5)}",
        ),
    )


# ---------------------------------------------------------------------
# Segundo principio: exergía destruida (Cengel §11-5)
# ---------------------------------------------------------------------

_INTERPRETATION: dict[str, str] = {
    "compressor": (
        "La compresión real genera entropía por fricción y turbulencia; con η_C = 1 sería cero."
    ),
    "condenser": (
        "El refrigerante cede calor a la fuente caliente con diferencia de temperatura, sobre "
        "todo mientras se enfría el vapor sobrecalentado que sale del compresor: esa "
        "diferencia es irreversible."
    ),
    "valve": (
        "El estrangulamiento baja la presión sin dar trabajo: es una de las mayores "
        "irreversibilidades del ciclo (una turbina en su lugar recuperaría parte; Cengel "
        "§11-3)."
    ),
    "evaporator": (
        "El calor entra desde la fuente fría, que está más caliente que el refrigerante: la "
        "diferencia de temperatura genera entropía."
    ),
    "mixer": "Se mezclan dos vapores a distinta temperatura: la mezcla es irreversible.",
    "flash_tank": (
        "Separar las fases a la misma presión y temperatura no genera entropía: la cámara no "
        "destruye exergía (las pérdidas están en las válvulas)."
    ),
    "cascade_hx": (
        "El calor pasa del ciclo de baja al de alta con diferencia de temperatura (al menos "
        "mientras se enfría el vapor sobrecalentado del compresor de baja)."
    ),
}


def _absolute_T_text(T_K: float, system: UnitSystem) -> str:
    """Temperatura absoluta para el texto: K (SI y Técnico) o °R (Inglés)."""
    if system == "Inglés":
        return f"{T_K * 1.8:.5g} R"
    return f"{T_K:.5g} K"


def _sgen_unit(system: UnitSystem) -> str:
    """Unidad de Ṡ_gen en LaTeX: potencia sobre temperatura absoluta."""
    power = unit_label("power", system)
    absolute = "°R" if system == "Inglés" else "K"
    return r"\mathrm{" + f"{power}/{absolute}".replace("°", r"{}^{\circ}") + "}"


def _sgen_value(value: float, system: UnitSystem) -> str:
    """Ṡ_gen en el sistema (potencia / temperatura absoluta)."""
    factor = convert_from_si(1.0, "power", system) * (1.8 if system == "Inglés" else 1.0)
    return rf"{latex_number(value * factor, 5)}\ {_sgen_unit(system)}"


def _sgen_symbolic(result: RefrigerationResult, comp: RefrigerationComponent) -> str:
    """Ṡ_gen del componente en símbolos (caudales por estado de entrada)."""
    if comp.kind in ("compressor", "valve", "evaporator", "condenser"):
        a, b = comp.inlet, comp.outlet
        core = rf"\dot{{m}}_{_ix(a + 1)}\,(s_{_ix(b + 1)} - s_{_ix(a + 1)})"
        if comp.kind == "evaporator":
            return rf"{core} - \frac{{\dot{{Q}}_C}}{{T_C}}"
        if comp.kind == "condenser":
            return rf"{core} + \frac{{\dot{{Q}}_H}}{{T_H}}"
        return core
    if comp.kind == "cascade_hx":
        return " + ".join(
            rf"\dot{{m}}_{_ix(i + 1)}\,(s_{_ix(o + 1)} - s_{_ix(i + 1)})" for i, o in comp.paths
        )
    outs = " + ".join(rf"\dot{{m}}_{_ix(o + 1)}\,s_{_ix(o + 1)}" for o in comp.outlets)
    ins = " - ".join(rf"\dot{{m}}_{_ix(i + 1)}\,s_{_ix(i + 1)}" for i in comp.inlets)
    return f"{outs} - {ins}"


def _sgen_numbers(
    result: RefrigerationResult, comp: RefrigerationComponent, system: UnitSystem
) -> str | None:
    """Los números de Ṡ_gen, solo para los componentes de una corriente."""
    if comp.kind not in ("compressor", "valve", "evaporator", "condenser"):
        return None
    a, b = comp.inlet, comp.outlet
    st = result.states
    res = result.inputs.reservoirs
    assert res is not None
    head = (
        rf"{_n(result.m_dot_kg_s[a], 'mass_flow', system)}\,"
        rf"({_diff(st[b].s_J_per_kg_K, st[a].s_J_per_kg_K, _ES, system)})"
    )
    if comp.kind == "evaporator":
        return (
            rf"{head} \\ &\quad - \frac{{{_n(result.Q_cold_W, 'power', system)}}}"
            rf"{{{_absolute_T(res.T_cold_K, system)}}}"
        )
    if comp.kind == "condenser":
        return (
            rf"{head} \\ &\quad + \frac{{{_n(result.Q_hot_W, 'power', system)}}}"
            rf"{{{_absolute_T(res.T_hot_K, system)}}}"
        )
    return head


def _exergy_steps(result: RefrigerationResult, system: UnitSystem) -> list[ProcedureStep]:
    exergy = result.exergy
    res = result.inputs.reservoirs
    if exergy is None or res is None:
        return []
    heat_pump = result.inputs.heat_pump
    T0 = _absolute_T(exergy.T0_K, system)
    ambient = "la fuente fría (el aire exterior)" if heat_pump else "la fuente caliente"
    steps = [
        ProcedureStep(
            title="Segundo principio: el ambiente y las fuentes",
            text=(
                f"El ambiente es {ambient}: T₀ = {_absolute_T_text(exergy.T0_K, system)}. En "
                "cada componente, "
                "la exergía destruida es el trabajo perdido X_dest = T₀·Ṡ_gen (vademecum "
                "§11.8), con Ṡ_gen = Σṁ·s a la salida − Σṁ·s a la entrada; el evaporador "
                "recibe Q̇_C de la fuente fría (T_C) y el condensador entrega Q̇_H a la "
                "caliente (T_H), así que su Ṡ_gen incluye la entropía de esas fuentes "
                "(Cengel §11-5)."
            ),
            latex=(
                rf"T_0 = {T0}",
                rf"T_C = {_absolute_T(res.T_cold_K, system)}"
                rf"\qquad T_H = {_absolute_T(res.T_hot_K, system)}",
            ),
        )
    ]
    s_gen = dict(exergy.S_gen)
    for comp, (label, x_dest) in zip(result.components, exergy.destroyed, strict=True):
        numbers = _sgen_numbers(result, comp, system)
        chain = [r"\dot{S}_{\mathrm{gen}}", _sgen_symbolic(result, comp)]
        if numbers is not None:
            chain.append(numbers)
        chain.append(_sgen_value(s_gen[label], system))
        paths = ", ".join(f"{i + 1} → {o + 1}" for i, o in comp.paths)
        steps.append(
            ProcedureStep(
                title=f"Exergía destruida: {label} ({paths})",
                text=_INTERPRETATION[comp.kind],
                latex=(
                    latex_chain(*chain),
                    latex_chain(
                        r"\dot{X}_{\mathrm{dest}}",
                        r"T_0\,\dot{S}_{\mathrm{gen}}",
                        _q(x_dest, "power", system),
                    ),
                ),
            )
        )
    useful = (
        latex_chain(
            r"\dot{W}_{\text{mín}}",
            r"\dot{Q}_H\left(1 - \frac{T_0}{T_H}\right)",
            _q(exergy.W_min_W, "power", system),
        )
        if heat_pump
        else latex_chain(
            r"\dot{W}_{\text{mín}}",
            r"\dot{Q}_C\left(\frac{T_0}{T_C} - 1\right)",
            _q(exergy.W_min_W, "power", system),
        )
    )
    eta_lhs = r"\eta_{\mathrm{ex},B}" if heat_pump else r"\eta_{\mathrm{ex},R}"
    cop_rev = result.COP_rev or 0.0
    steps.append(
        ProcedureStep(
            title="Trabajo mínimo y rendimiento exergético",
            text=(
                "El trabajo mínimo es la exergía del efecto útil: el de un ciclo reversible "
                "entre las fuentes (vademecum §11.5 y §11.10). Lo que el compresor consume de "
                "más es exactamente la exergía destruida: Ẇ = Ẇ_mín + ΣẊ_dest. El rendimiento "
                "exergético también es COP/COP_rev."
            ),
            latex=(
                useful,
                latex_chain(
                    r"\sum \dot{X}_{\mathrm{dest}}",
                    _q(exergy.X_destroyed_W, "power", system),
                ),
                latex_chain(
                    r"\dot{W}_{\text{mín}} + \sum \dot{X}_{\mathrm{dest}}",
                    _q(exergy.W_min_W + exergy.X_destroyed_W, "power", system),
                ),
                latex_chain(
                    eta_lhs,
                    r"\frac{\dot{W}_{\text{mín}}}{\dot{W}}",
                    rf"\frac{{{_n(exergy.W_min_W, 'power', system)}}}"
                    rf"{{{_n(exergy.W_W, 'power', system)}}}",
                    rf"{latex_number(exergy.eta_ex * 100, 4)}\,\%",
                ),
                latex_chain(
                    r"\mathrm{COP}_{\mathrm{rev}}",
                    (r"\frac{T_H}{T_H - T_C}" if heat_pump else r"\frac{T_C}{T_H - T_C}"),
                    latex_number(cop_rev, 5),
                ),
            ),
        )
    )
    return steps


# ---------------------------------------------------------------------
# Procedimiento completo
# ---------------------------------------------------------------------


def refrigeration_steps(result: RefrigerationResult, system: UnitSystem) -> list[ProcedureStep]:
    """Procedimiento estado por estado, como se resuelve con las tablas (Cengel cap. 11)."""
    steps: list[ProcedureStep] = [_evaporator_outlet_step(result, system)]
    layout = result.layout
    if result.cycle == "simple":
        steps.append(_compressor_step(result, 0, system))
        steps.append(_condenser_outlet_step(result, system))
        steps.append(_valve_step(result, layout.one("valve"), system))
    elif result.cycle == "flash":
        valve_hi, valve_lo = layout.of_kind("valve")
        steps.append(_compressor_step(result, 0, system))  # compresor de baja
        steps.append(_condenser_outlet_step(result, system))
        steps.append(_valve_step(result, valve_hi, system))
        steps.append(_flash_tank_step(result, layout.one("flash_tank"), system))
        steps.append(_valve_step(result, valve_lo, system))
        steps.append(_mixer_step(result, layout.one("mixer"), system))
        steps.append(_compressor_step(result, 1, system))  # compresor de alta
    else:
        hx = layout.one("cascade_hx")
        valve_lo, valve_hi = layout.of_kind("valve")
        fluid_low = FLUID_NAMES_ES.get(result.inputs.fluid_low, result.inputs.fluid_low)
        fluid_high = FLUID_NAMES_ES.get(result.inputs.fluid, result.inputs.fluid)
        steps.append(_compressor_step(result, 0, system))  # compresor de baja
        steps.append(
            _saturated_step(
                result,
                hx.outlets[0],
                "f",
                system,
                f"El ciclo de baja ({fluid_low}) sale del intercambiador como líquido saturado, "
                "a la presión a la que condensa.",
            )
        )
        steps.append(_valve_step(result, valve_lo, system))
        steps.append(
            _saturated_step(
                result,
                hx.outlets[1],
                "g",
                system,
                f"El ciclo de alta ({fluid_high}) sale del intercambiador como vapor saturado, "
                "a la presión a la que evapora.",
            )
        )
        steps.append(_compressor_step(result, 1, system))  # compresor de alta
        steps.append(_condenser_outlet_step(result, system))
        steps.append(_valve_step(result, valve_hi, system))
        steps.append(_cascade_hx_step(result, hx, system))
    steps.append(_heat_and_work_step(result, system))
    steps.append(_cop_step(result, system))
    steps.append(_carnot_step(result, system))
    steps.append(_power_step(result, system))
    comparison = _comparison_step(result)
    if comparison is not None:
        steps.append(comparison)
    steps.extend(_exergy_steps(result, system))
    return steps
