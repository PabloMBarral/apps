"""Procedimiento del poder calorífico «como en el pizarrón» — Fase 6.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo:

1. el análisis en base seca (el cambio de base, si los datos venían tal cual o
   secos y sin cenizas);
2. cada correlación en su forma publicada (los % en masa en base seca y el PCS
   en MJ/kg), con el resultado pasado a las unidades del sistema;
3. el PCS tal cual y seco y sin cenizas;
4. el PCI (vademecum §16.9) y la humedad con la que se anula;
5. el desvío de cada correlación respecto del PCS de referencia.

Las sumas largas van de a dos términos por renglón y los números con ×10ⁿ del
SI pasan a un renglón propio, para que entren en un celular
(:func:`core.latex.latex_chain`). No importa Streamlit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.combustion.heating_value import (
    CORRELATIONS,
    CorrelationEstimate,
    HeatingValueInputs,
    HeatingValueResult,
    hydrogen_water_ratio,
    water_hfg_J_per_kg,
)
from core.latex import latex_chain, latex_is_wide, latex_number, latex_quantity
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem, convert_from_si

__all__ = ["correlation_latex", "heating_value_steps"]

_EH: QuantityKind = "specific_enthalpy"

#: Símbolo de cada variable de los análisis.
_SYM: dict[str, str] = {
    "C": "C",
    "H": "H",
    "O": "O",
    "N": "N",
    "S": "S",
    "A": r"\mathit{Cz}",
    "VM": r"\mathit{MV}",
    "FC": r"\mathit{CF}",
}
#: Subíndice corto de cada correlación (para los desvíos).
_SHORT: dict[str, str] = {
    "channiwala_parikh": r"\text{CyP}",
    "boie": r"\text{Boie}",
    "dulong": r"\text{Dulong}",
    "parikh": r"\text{Parikh}",
    "cordero": r"\text{Cordero}",
}
_ORDER_ULT = ("C", "H", "O", "N", "S")
_ORDER_PROX = ("VM", "FC")


def _pc(fraction: float) -> str:
    """Un porcentaje (sin el signo) con 4 cifras."""
    return latex_number(100.0 * fraction, 4)


def _mj(value_J_per_kg: float) -> str:
    return latex_number(value_J_per_kg / 1e6, 4)


def _n(value_J_per_kg: float, system: UnitSystem) -> str:
    """Un poder calorífico (sin unidad) en el sistema."""
    return latex_number(convert_from_si(value_J_per_kg, _EH, system), 5)


def _q(value_J_per_kg: float, system: UnitSystem) -> str:
    return latex_quantity(value_J_per_kg, _EH, system, 5)


def _contributions(
    coefficients: Sequence[tuple[str, float]], pct: Mapping[str, float], total_MJ: float
) -> str:
    r"""Aporte de cada componente (``C:\quad 0.3491 \cdot 48.64 = 16.98``) y su suma, en MJ/kg.

    Los aportes van redondeados al kJ/kg (``0.004`` y no ``0.003768``) y la suma sin la
    unidad (la dice el renglón siguiente): así la tabla entra en un celular.
    """
    lines = []
    for var, coef in coefficients:
        label = r"H - O/8" if var == "H_av" else _SYM[var]
        lines.append(
            rf"{label}:\quad {latex_number(coef, 6)} \cdot {latex_number(pct[var], 4)} "
            rf"&= {latex_number(round(coef * pct[var], 3), 4)}"
        )
    lines.append(rf"PCS_\mathrm{{s}} = \textstyle\sum &= {latex_number(total_MJ, 4)}")
    return r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}"


def correlation_latex(
    key: str, pct: Mapping[str, float], hhv_d: float, system: UnitSystem
) -> list[str]:
    r"""La correlación con sus números, en renglones que entran en un celular.

    1. la fórmula publicada (``PCS_s`` en MJ/kg, los % en masa en base seca);
    2. en Dulong, el «hidrógeno disponible» H − O/8;
    3. el aporte de cada componente y la suma;
    4. el resultado en las unidades del sistema.

    ``pct`` son los % en masa en base seca del análisis que usa.
    """
    corr = CORRELATIONS[key]
    latex = [rf"\begin{{aligned}}PCS_\mathrm{{s}} &= {corr.latex}\end{{aligned}}"]
    coefficients = list(corr.coefficients)
    values = dict(pct)
    if key == "dulong":
        available = pct["H"] - pct["O"] / 8.0
        latex.append(
            latex_chain(
                r"H - \frac{O}{8}",
                rf"{latex_number(pct['H'], 4)} - \frac{{{latex_number(pct['O'], 4)}}}{{8}}",
                rf"{latex_number(available, 4)}\,\%",
            )
        )
        values["H_av"] = available
        coefficients = [("C", 0.3383), ("H_av", 1.443), ("S", 0.0942)]
    latex.append(_contributions(coefficients, values, hhv_d / 1e6))
    latex.append(
        latex_chain(r"PCS_\mathrm{s}", rf"{_mj(hhv_d)}\ \mathrm{{MJ/kg}}", _q(hhv_d, system))
    )
    return latex


# ---------------------------------------------------------------------
# Pasos
# ---------------------------------------------------------------------


def _dry_rows(inp: HeatingValueInputs) -> list[tuple[str, float]]:
    """(variable, fracción seca) de los análisis cargados, con las cenizas al final."""
    rows: list[tuple[str, float]] = []
    if inp.ultimate is not None:
        rows += [(k, getattr(inp.ultimate, k)) for k in _ORDER_ULT]
    if inp.proximate is not None:
        rows += [(k, getattr(inp.proximate, k)) for k in _ORDER_PROX]
    rows.append(("A", inp.ash_d))
    return rows


def _basis_step(inp: HeatingValueInputs) -> ProcedureStep:
    rows = _dry_rows(inp)
    W, A_d = inp.moisture, inp.ash_d
    latex: list[str] = []
    if inp.basis == "d":
        pairs = [rf"{_SYM[k]}_\mathrm{{s}} &= {_pc(v)}\,\%" for k, v in rows]
        lines = [" & ".join(pairs[i : i + 2]) for i in range(0, len(pairs), 2)]
        latex.append(r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}")
        text = (
            "Los datos ya están en base seca, la de las correlaciones (en % en masa). La "
            f"humedad tal cual es W = {_pc(W)} %."
        )
        return ProcedureStep("Análisis en base seca", text, tuple(latex))
    if inp.basis == "ar":
        latex.append(
            latex_chain(
                r"x_\mathrm{s}",
                r"\frac{x_\mathrm{tc}}{1 - W}",
                rf"\frac{{x_\mathrm{{tc}}}}{{1 - {latex_number(W, 4)}}}",
            )
        )
        dry = latex_number(1.0 - W, 4)
        lines = [
            rf"{_SYM[k]}_\mathrm{{s}} &= \frac{{{_pc(v * (1.0 - W))}}}{{{dry}}} = {_pc(v)}\,\%"
            for k, v in rows
        ]
        text = (
            "Las correlaciones van en base seca: cada kg tal cual tiene W kg de agua y (1 − W) "
            "kg secos, así que cada componente se divide por 1 − W."
        )
    else:
        latex.append(
            latex_chain(
                r"x_\mathrm{s}",
                r"x_\mathrm{sscz}\,(1 - \mathit{Cz}_\mathrm{s})",
                rf"x_\mathrm{{sscz}} \cdot {latex_number(1.0 - A_d, 4)}",
            )
        )
        lines = [
            rf"{_SYM[k]}_\mathrm{{s}} &= {_pc(v / (1.0 - A_d))} \cdot "
            rf"{latex_number(1.0 - A_d, 4)} = {_pc(v)}\,\%"
            for k, v in rows
            if k != "A"
        ]
        text = (
            "Las correlaciones van en base seca: el kg seco tiene Cz_s kg de cenizas y "
            "(1 − Cz_s) kg de materia combustible, así que cada componente sin cenizas se "
            f"multiplica por 1 − Cz_s (con Cz_s = {_pc(A_d)} %)."
        )
    latex.append(r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}")
    return ProcedureStep("Análisis en base seca", text, tuple(latex))


def _range_text(key: str) -> str:
    corr = CORRELATIONS[key]
    if not corr.ranges:
        return ""
    parts = [f"{_var_text(v)} {_num(lo)}–{_num(hi)} %" for v, lo, hi in corr.ranges]
    return " Rango de ajuste (base seca): " + ", ".join(parts) + "."


def _var_text(var: str) -> str:
    return {"A": "cenizas", "VM": "MV", "FC": "CF"}.get(var, var)


def _num(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _correlation_step(
    e: CorrelationEstimate, inp: HeatingValueInputs, system: UnitSystem
) -> ProcedureStep:
    corr = CORRELATIONS[e.key]
    analysis = inp.ultimate if corr.analysis == "ultimate" else inp.proximate
    assert analysis is not None
    pct = analysis.pct()
    latex = correlation_latex(e.key, pct, e.hhv_d, system)
    text = f"{corr.reference} Tipo de combustible: {corr.fuels}.{_range_text(e.key)}"
    if corr.reported:
        text += f" Error que informan los autores: {corr.reported}."
    if e.key == "dulong":
        text += (
            " El «hidrógeno disponible» H − O/8 es el que no está ya unido al oxígeno del "
            "combustible: cada elemento aporta su PCS (C ≈ 33,8 MJ/kg, H₂ ≈ 144 MJ/kg, "
            "S ≈ 9,4 MJ/kg)."
        )
    if e.warnings:
        text += " Ojo: " + "; ".join(e.warnings) + "."
    if not e.in_type:
        text += " Este tipo de combustible no está entre los que se usaron para ajustarla."
    return ProcedureStep(corr.name, text, tuple(latex))


def _bases_step(result: HeatingValueResult, system: UnitSystem) -> ProcedureStep:
    inp = result.inputs
    e = result.main
    W, A_d = inp.moisture, inp.ash_d
    hs = _n(e.hhv_d, system)
    latex = [
        latex_chain(
            r"PCS_\mathrm{tc}",
            r"PCS_\mathrm{s}\,(1 - W)",
            rf"{hs} \cdot (1 - {latex_number(W, 4)})",
            _q(e.hhv_ar, system),
        ),
        latex_chain(
            r"PCS_\mathrm{sscz}",
            r"\frac{PCS_\mathrm{s}}{1 - \mathit{Cz}_\mathrm{s}}",
            rf"\frac{{{hs}}}{{1 - {latex_number(A_d, 4)}}}",
            _q(e.hhv_daf, system),
        ),
    ]
    text = (
        f"Con {e.name}: el kg tal cual tiene (1 − W) kg secos y el agua no aporta; el kg seco "
        "y sin cenizas es solo materia combustible (las cenizas tampoco aportan)."
    )
    return ProcedureStep("PCS tal cual y seco sin cenizas", text, tuple(latex))


def _lhv_step(result: HeatingValueResult, system: UnitSystem) -> ProcedureStep:
    inp = result.inputs
    e = result.main
    H_d = inp.hydrogen_d
    if H_d is None or e.lhv_d is None or e.lhv_ar is None:
        return ProcedureStep(
            "Poder calorífico inferior",
            "Sin el H no se sabe cuánta agua forma el combustible, así que no se puede pasar del "
            "PCS al PCI: cargá el H en base seca.",
        )
    W = inp.moisture
    w = latex_number(W, 4)
    ratio = hydrogen_water_ratio()
    m_w = ratio * H_d
    hfg = water_hfg_J_per_kg()
    hs, hfg_n = _n(e.hhv_d, system), _n(hfg, system)
    ls = _n(e.lhv_d, system)
    wide = latex_is_wide(hs, hfg_n)
    br = r" \\ &\quad " if wide else " "
    latex = [
        latex_chain(
            r"m_w",
            r"\frac{M_\mathrm{H_2O}}{2\,M_\mathrm{H}}\,H_\mathrm{s}",
            rf"{latex_number(ratio, 4)} \cdot {latex_number(H_d, 4)}",
            rf"{latex_number(m_w, 4)}\ \mathrm{{kg/kg}}",
        ),
        latex_chain(
            r"PCI_\mathrm{s}",
            r"PCS_\mathrm{s} - h_{fg}\,m_w",
            rf"{hs}{br}- {hfg_n} \cdot {latex_number(m_w, 4)}",
            _q(e.lhv_d, system),
        ),
        latex_chain(
            r"PCI_\mathrm{tc}",
            r"PCI_\mathrm{s}\,(1 - W) - h_{fg}\,W",
            rf"{ls} \cdot (1 - {w}) \\ &\quad - {hfg_n} \cdot {w}",
            _q(e.lhv_ar, system),
        ),
    ]
    w_star = result.zero_lhv_moisture
    if w_star is not None:
        latex.append(
            latex_chain(
                r"W^*",
                r"\frac{PCI_\mathrm{s}}{PCI_\mathrm{s} + h_{fg}}",
                rf"\frac{{{ls}}}{{{ls} + {hfg_n}}}",
                rf"{_pc(w_star)}\,\%",
            )
        )
    text = (
        f"Con {e.name}. El H forma M_H₂O/(2·M_H) ≈ 8,94 kg de agua por kg (el «9·H» de los "
        "libros); en el PCI esa agua y la humedad salen como vapor y se llevan h_fg a 25 °C "
        f"= {_q_text(hfg)} (vademecum §16.9). W* es la humedad con la que el PCI tal cual se "
        "anula: el combustible no alcanza a evaporar su propia agua."
    )
    return ProcedureStep("Poder calorífico inferior", text, tuple(latex))


def _q_text(value_J_per_kg: float) -> str:
    return f"{value_J_per_kg / 1000:.1f} kJ/kg".replace(".", ",")


def _deviation_step(result: HeatingValueResult) -> ProcedureStep | None:
    ref = result.reference
    if ref is None:
        return None
    r_mj = _mj(ref.hhv_d)
    latex = [r"\delta = \frac{PCS_\mathrm{s} - PCS_\mathrm{s,ref}}{PCS_\mathrm{s,ref}}"]
    lines = []
    for e in result.estimates:
        assert e.deviation is not None
        sign = "+" if e.deviation >= 0 else "-"
        lines.append(
            rf"\delta_{{{_SHORT[e.key]}}} &= \frac{{{_mj(e.hhv_d)} - {r_mj}}}{{{r_mj}}}"
            rf" = {sign}{latex_number(abs(100.0 * e.deviation), 3)}\,\%"
        )
    latex.append(r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}")
    what = "exacto (de las h_f)" if ref.name == "PCS exacto" else "medido"
    text = (
        f"Cuánto se aparta cada correlación del PCS {what}, en base seca y en MJ/kg. Para "
        "comparar: en las 536 biomasas de Ghugare et al. (2014), Channiwala y Parikh se "
        "equivoca 4,9 % en promedio, Boie 5,2 % y Dulong 11,5 %."
    )
    return ProcedureStep("Desvío respecto de la referencia", text, tuple(latex))


def heating_value_steps(result: HeatingValueResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos del procedimiento en el sistema de unidades ``system``."""
    inp = result.inputs
    steps = [_basis_step(inp)]
    steps += [_correlation_step(e, inp, system) for e in result.estimates]
    steps.append(_bases_step(result, system))
    steps.append(_lhv_step(result, system))
    deviation = _deviation_step(result)
    if deviation is not None:
        steps.append(deviation)
    return [
        ProcedureStep(f"{k}. {st.title}", st.text, st.latex) for k, st in enumerate(steps, start=1)
    ]
