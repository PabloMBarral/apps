"""Procedimiento didáctico de los gases ideales (LaTeX), en el sistema de unidades activo.

Las fórmulas son las del vademecum: §4 (gas ideal), §5 (mezclas), §6
(politrópicas), §3.5 (trabajo) y §10.5 (entropía). La temperatura de las
fórmulas es la absoluta (K o °R). p·v necesita un factor de unidades en el
Técnico (1 bar·m³/kg = 100 kJ/kg) y en el Inglés (1 psia·ft³/lb = 0,18505 Btu/lb);
el resto de las sustituciones cierra sin factores (``core.units_system``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from core.gases.ideal import R_U, IdealGas, PropertyChange, StateChange, gas
from core.gases.mixture import MixingResult, MixtureResult, StreamResult
from core.gases.polytropic import (
    PROCESS_KINDS,
    ExponentResult,
    ProcessResult,
    StagedResult,
)
from core.heat_transfer.procedure_common import frac, n, numbered, q, sub, sum_rows, times
from core.latex import latex_chain, latex_is_wide, latex_number, latex_paren, latex_unit
from core.state_report import ProcedureStep, pv_energy_factor
from core.units_system import UnitSystem, unit_label

__all__ = [
    "exponent_steps",
    "gas_symbol",
    "mixing_steps",
    "mixture_steps",
    "process_steps",
    "staged_steps",
    "state_change_steps",
]

_P = "pressure"
_T = "absolute_temperature"
_V = "specific_volume"
_E = "specific_enthalpy"
_S = "specific_entropy"
_M = "molar_mass"
_RU = "molar_entropy"

_SYMBOLS: dict[str, str] = {
    "air": r"\text{aire}",
    "Ar": r"\mathrm{Ar}",
    "CO2": r"\mathrm{CO_2}",
    "He": r"\mathrm{He}",
    "H2": r"\mathrm{H_2}",
    "CH4": r"\mathrm{CH_4}",
    "N2": r"\mathrm{N_2}",
    "O2": r"\mathrm{O_2}",
    "H2O": r"\mathrm{H_2O}",
    "CO": r"\mathrm{CO}",
    "NH3": r"\mathrm{NH_3}",
    "SO2": r"\mathrm{SO_2}",
    "C2H6": r"\mathrm{C_2H_6}",
    "C3H8": r"\mathrm{C_3H_8}",
    "n-C4H10": r"\mathrm{C_4H_{10}}",
}


def gas_symbol(key: str) -> str:
    """El símbolo LaTeX de un gas para un subíndice (``\\mathrm{CO_2}``)."""
    return _SYMBOLS.get(key, rf"\text{{{gas(key).name}}}")


def _r(x: float, sig: int = 5) -> str:
    """Un número sin unidad (relaciones, fracciones, exponentes)."""
    return latex_number(x, sig)


def _pv(system: UnitSystem) -> tuple[float, str]:
    """El factor de p·v (1, 100 o 0,18505) y su LaTeX (vacío en el SI)."""
    f = pv_energy_factor(system)
    return f, ("" if math.isclose(f, 1.0) else latex_number(f, 5))


def _pv_text(system: UnitSystem) -> str:
    f, _ = _pv(system)
    if system == "Técnico":
        return " Ojo con las unidades de p·v: 1 bar·m³/kg = 100 kJ/kg."
    if system == "Inglés":
        value = f"{f:.5f}".replace(".", ",")
        return f" Ojo con las unidades de p·v: 1 psia·ft³/lb = {value} Btu/lb."
    return ""


def _with_factor(f_tex: str, tex: str) -> str:
    """``f · tex`` si hay factor (Técnico e Inglés)."""
    return times(f_tex, tex) if f_tex else tex


def _v_from_pT(g: IdealGas, p: float, T: float, system: UnitSystem, label: str) -> str:
    """v = R·T/p con el factor de unidades."""
    _, f = _pv(system)
    den = _with_factor(f, n(p, _P, system))
    return latex_chain(
        f"v_{label}",
        frac(r"R\,T_" + label, "p_" + label),
        frac(times(n(g.R, _S, system), n(T, _T, system)), den),
        q(g.R * T / p, _V, system),
    )


def _T_from_pv(g: IdealGas, p: float, v: float, system: UnitSystem, label: str) -> str:
    _, f = _pv(system)
    num = times(*([f] if f else []), n(p, _P, system), n(v, _V, system))
    return latex_chain(
        f"T_{label}",
        frac(rf"p_{label}\,v_{label}", "R"),
        frac(num, n(g.R, _S, system)),
        q(p * v / g.R, _T, system),
    )


def _p_from_Tv(g: IdealGas, T: float, v: float, system: UnitSystem, label: str) -> str:
    _, f = _pv(system)
    den = _with_factor(f, n(v, _V, system))
    return latex_chain(
        f"p_{label}",
        frac(rf"R\,T_{label}", f"v_{label}"),
        frac(times(n(g.R, _S, system), n(T, _T, system)), den),
        q(g.R * T / v, _P, system),
    )


def _gas_step(g: IdealGas, system: UnitSystem, with_cp: bool = True) -> ProcedureStep:
    lines = [
        latex_chain(
            "R",
            frac("R_u", "M"),
            frac(q(R_U, _RU, system), q(g.M, _M, system)),
            q(g.R, _S, system),
        )
    ]
    text = "La constante del gas es R = R_u/M (vademecum §4.2)."
    if g.key == "air":
        text += " El aire es la mezcla seca de N₂, O₂, Ar y CO₂ (M = Σ y_i·M_i)."
    if with_cp:
        cp, cv = g.cp_ref, g.cp_ref - g.R
        if g.monatomic:
            lines.append(latex_chain("c_p", r"\tfrac{5}{2}\,R", q(cp, _S, system)))
            text += " El helio es monoatómico: c_p = 5/2·R exacto."
        else:
            lines.append(latex_chain(r"c_p", r"c_p(25\,^{\circ}\mathrm{C})", q(cp, _S, system)))
            text += " El c_p constante es el de 25 °C, el de la tabla del vademecum (§4.6)."
        lines.append(
            latex_chain(
                "c_v",
                "c_p - R",
                sub(n(cp, _S, system), n(g.R, _S, system)),
                q(cv, _S, system),
            )
        )
        lines.append(latex_chain("k", frac("c_p", "c_v"), _r(cp / cv, 5)))
        text += " c_v sale de la relación de Mayer, c_p − c_v = R (§4.5)."
    return ProcedureStep("El gas", text, tuple(lines))


# ---------------------------------------------------------------------
# Un gas entre dos estados
# ---------------------------------------------------------------------


def _ds_constant(
    cp: float, R: float, ratios: tuple[float, float], ds: float, system: UnitSystem
) -> str:
    """Δs = c_p·ln(T₂/T₁) − R·ln(p₂/p₁), con el Δs del resultado (``ratios`` = T₂/T₁, p₂/p₁)."""
    T_ratio, p_ratio = ratios
    return latex_chain(
        r"\Delta s",
        r"c_p \ln\frac{T_2}{T_1} - R \ln\frac{p_2}{p_1}",
        rf"{n(cp, _S, system)} \ln {_r(T_ratio)} \\ &\quad - {n(R, _S, system)} \ln {_r(p_ratio)}",
        q(ds, _S, system),
    )


def _model_lines(change: PropertyChange, r: StateChange, system: UnitSystem) -> list[str]:
    """Δh, Δu y Δs con un c_p constante (el de 25 °C o el de la T media)."""
    T1, T2 = r.state1.T_K, r.state2.T_K
    dT_tex = sub(n(T2, _T, system), n(T1, _T, system))
    return [
        latex_chain(
            r"\Delta h",
            r"c_p\,(T_2 - T_1)",
            times(n(change.cp, _S, system), f"({dT_tex})"),
            q(change.dh, _E, system),
        ),
        latex_chain(
            r"\Delta u",
            r"c_v\,(T_2 - T_1)",
            times(n(change.cv, _S, system), f"({dT_tex})"),
            q(change.du, _E, system),
        ),
        _ds_constant(
            change.cp, r.gas.R, (T2 / T1, r.state2.p_Pa / r.state1.p_Pa), change.ds, system
        ),
    ]


def state_change_steps(r: StateChange, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de un gas ideal entre dos estados, con los tres modelos de c_p."""
    g = r.gas
    steps = [_gas_step(g, system)]

    lines = []
    for label, pair, st in (("1", r.inputs.state1, r.state1), ("2", r.inputs.state2, r.state2)):
        if pair.pair == "pT":
            lines.append(_v_from_pT(g, st.p_Pa, st.T_K, system, label))
        elif pair.pair == "pv":
            lines.append(_T_from_pv(g, st.p_Pa, st.v_m3_per_kg, system, label))
        else:
            lines.append(_p_from_Tv(g, st.T_K, st.v_m3_per_kg, system, label))
    steps.append(
        ProcedureStep(
            "Los dos estados",
            "Con dos de p, T y v, el tercero sale de p·v = R·T (vademecum §4.2), con la "
            "temperatura absoluta." + _pv_text(system),
            tuple(lines),
        )
    )

    steps.append(
        ProcedureStep(
            "Con c_p constante a 25 °C",
            "Δh = c_p·ΔT, Δu = c_v·ΔT y Δs = c_p·ln(T₂/T₁) − R·ln(p₂/p₁) (vademecum §4.5 y "
            "§10.5.1).",
            tuple(_model_lines(r.constant, r, system)),
        )
    )
    T_m = r.T_mean_K
    mean_lines = [
        latex_chain(
            "T_m",
            frac("T_1 + T_2", "2"),
            q(T_m, _T, system),
        ),
        latex_chain(r"c_p(T_m)", q(r.mean.cp, _S, system)),
        latex_chain(
            r"c_v(T_m)",
            sub(n(r.mean.cp, _S, system), n(g.R, _S, system)),
            q(r.mean.cv, _S, system),
        ),
        *_model_lines(r.mean, r, system),
    ]
    steps.append(
        ProcedureStep(
            "Con el c_p a la temperatura media",
            "El mismo cálculo con el c_p del polinomio a la temperatura media del proceso "
            "(Çengel §7-9): casi siempre alcanza.",
            tuple(mean_lines),
        )
    )

    T1, T2 = r.state1.T_K, r.state2.T_K
    h1, h2 = g.h(T1), g.h(T2)
    var_lines = [
        latex_chain(r"h(T_1)", q(h1, _E, system)),
        latex_chain(r"h(T_2)", q(h2, _E, system)),
        latex_chain(
            r"\Delta h",
            "h(T_2) - h(T_1)",
            sub(n(h2, _E, system), n(h1, _E, system)),
            q(r.variable.dh, _E, system),
        ),
        latex_chain(
            r"\Delta u",
            r"\Delta h - R\,(T_2 - T_1)",
            rf"{n(r.variable.dh, _E, system)} \\ &\quad - "
            + times(n(g.R, _S, system), f"({sub(n(T2, _T, system), n(T1, _T, system))})"),
            q(r.variable.du, _E, system),
        ),
        latex_chain(r"s^{\circ}(T_1)", q(r.s01, _S, system)),
        latex_chain(r"s^{\circ}(T_2)", q(r.s02, _S, system)),
        latex_chain(
            r"\Delta s",
            r"s^{\circ}_2 - s^{\circ}_1 - R \ln\frac{p_2}{p_1}",
            rf"{sub(n(r.s02, _S, system), n(r.s01, _S, system))} \\ &\quad - "
            rf"{n(g.R, _S, system)} \ln {_r(r.state2.p_Pa / r.state1.p_Pa)}",
            q(r.variable.ds, _S, system),
        ),
    ]
    text = (
        "h(T) y s°(T) salen de integrar el polinomio NASA de 9 coeficientes (vademecum §4.7 y "
        "§10.5.2), con h = s° = 0 a 25 °C; Δu = Δh − R·ΔT porque h = u + R·T."
    )
    if g.monatomic:
        text = "El helio tiene c_p constante exacto: h = c_p·(T − 25 °C) y s° = c_p·ln(T/298,15 K)."
    steps.append(ProcedureStep("Con c_p variable", text, tuple(var_lines)))

    pr_lines = []
    for label, T, s0, pr, vr in (("1", T1, r.s01, r.pr1, r.vr1), ("2", T2, r.s02, r.pr2, r.vr2)):
        pr_lines.append(
            latex_chain(
                f"p_{{r{label}}}",
                rf"e^{{s^{{\circ}}_{label}/R}}",
                rf"e^{{{_r(s0 / g.R, 5)}}}",
                _r(pr, 5),
            )
        )
        pr_lines.append(
            latex_chain(
                f"v_{{r{label}}}",
                frac(f"T_{label}", f"p_{{r{label}}}"),
                frac(n(T, _T, system), _r(pr, 5)),
                q(vr, _T, system),
            )
        )
    steps.append(
        ProcedureStep(
            "Presión y volumen relativos",
            "p_r = exp(s°/R) y v_r = T/p_r (vademecum §10.5.3), con p_r = 1 a 25 °C. En una "
            "isoentrópica, p₂/p₁ = p_r2/p_r1 y v₂/v₁ = v_r2/v_r1. Los valores no son los de la "
            "tabla A-17 de Çengel, que usa otra referencia, pero los cocientes (lo que se usa) "
            "son los mismos.",
            tuple(pr_lines),
        )
    )

    errs = []
    for model, label in (("constant", "con c_p a 25 °C"), ("mean", "con c_p a la T media")):
        e_dh = r.error(model, "dh")  # type: ignore[arg-type]
        e_ds = r.error(model, "ds")  # type: ignore[arg-type]
        parts = []
        if e_dh is not None:
            parts.append(f"Δh {_signed_pct(e_dh)}")
        if e_ds is not None:
            parts.append(f"Δs {_signed_pct(e_ds)}")
        if parts:
            errs.append(f"{label}: {', '.join(parts)}")
    if errs:
        steps.append(
            ProcedureStep(
                "¿Cuánto te equivocás con c_p constante?",
                "Contra c_p variable: " + "; ".join(errs) + ".",
            )
        )
    return numbered(steps)


def _signed_pct(x: float) -> str:
    return f"{'+' if x >= 0 else '−'}{abs(x) * 100:.2f} %".replace(".", ",")


# ---------------------------------------------------------------------
# Mezcla
# ---------------------------------------------------------------------


def mixture_steps(r: MixtureResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de una mezcla de gases ideales (vademecum §5)."""
    steps: list[ProcedureStep] = []
    basis = r.inputs.basis
    comps = r.components
    if basis in ("mass", "moles"):
        lines = []
        for c in comps:
            s = gas_symbol(c.gas.key)
            assert c.m_kg is not None and c.n_mol is not None
            if basis == "mass":
                lines.append(
                    latex_chain(
                        f"N_{{{s}}}",
                        frac(f"m_{{{s}}}", f"M_{{{s}}}"),
                        frac(n(c.m_kg, "mass", system), n(c.gas.M, _M, system)),
                        q(c.n_mol, "amount", system),
                    )
                )
            else:
                lines.append(
                    latex_chain(
                        f"m_{{{s}}}",
                        f"N_{{{s}}}\\,M_{{{s}}}",
                        times(n(c.n_mol, "amount", system), n(c.gas.M, _M, system)),
                        q(c.m_kg, "mass", system),
                    )
                )
        assert r.m_kg is not None and r.n_mol is not None
        lines.append(latex_chain("m_M", r"\sum m_i", q(r.m_kg, "mass", system)))
        lines.append(latex_chain("N_M", r"\sum N_i", q(r.n_mol, "amount", system)))
        steps.append(
            ProcedureStep(
                "Moles y masa de cada gas",
                "N_i = m_i/M_i (vademecum §5.1).",
                tuple(lines),
            )
        )
        frac_lines = []
        for c in comps:
            s = gas_symbol(c.gas.key)
            frac_lines.append(
                latex_chain(
                    f"x_{{{s}}}",
                    frac(n(c.m_kg, "mass", system), n(r.m_kg, "mass", system)),
                    _r(c.x, 4),
                )
            )
            frac_lines.append(
                latex_chain(
                    f"y_{{{s}}}",
                    frac(n(c.n_mol, "amount", system), n(r.n_mol, "amount", system)),
                    _r(c.y, 4),
                )
            )
        steps.append(
            ProcedureStep(
                "Fracciones másica y molar",
                "x_i = m_i/m_M es la fracción másica e y_i = N_i/N_M la molar (vademecum §5.1).",
                tuple(frac_lines),
            )
        )
    elif basis == "mass_fraction":
        den = sum(c.x / c.gas.M for c in comps)
        terms = [frac(_r(c.x, 4), n(c.gas.M, _M, system)) for c in comps]
        lines = [
            latex_chain(r"\sum \frac{x_j}{M_j}", sum_rows(terms, 2), _r(_den_value(den, system), 5))
        ]
        for c in comps:
            s = gas_symbol(c.gas.key)
            lines.append(
                latex_chain(
                    f"y_{{{s}}}",
                    frac(frac(_r(c.x, 4), n(c.gas.M, _M, system)), _r(_den_value(den, system), 5)),
                    _r(c.y, 4),
                )
            )
        steps.append(
            ProcedureStep(
                "De fracciones másicas a molares",
                "y_i = (x_i/M_i)/Σ(x_j/M_j) (vademecum §5.1).",
                tuple(lines),
            )
        )
    else:
        num_terms = [times(_r(c.y, 4), n(c.gas.M, _M, system)) for c in comps]
        lines = [latex_chain(r"\sum y_j M_j", sum_rows(num_terms, 1), q(r.M, _M, system))]
        for c in comps:
            s = gas_symbol(c.gas.key)
            lines.append(
                latex_chain(
                    f"x_{{{s}}}",
                    frac(times(_r(c.y, 4), n(c.gas.M, _M, system)), n(r.M, _M, system)),
                    _r(c.x, 4),
                )
            )
        steps.append(
            ProcedureStep(
                "De fracciones molares a másicas",
                "x_i = y_i·M_i/Σ(y_j·M_j) (vademecum §5.1).",
                tuple(lines),
            )
        )

    m_terms = [times(_r(c.y, 4), n(c.gas.M, _M, system)) for c in comps]
    steps.append(
        ProcedureStep(
            "Masa molar y constante de la mezcla",
            "M_M = Σ y_i·M_i y R_M = R_u/M_M (vademecum §5.2).",
            (
                latex_chain("M_M", r"\sum y_i M_i", sum_rows(m_terms, 1), q(r.M, _M, system)),
                latex_chain(
                    "R_M",
                    frac("R_u", "M_M"),
                    frac(n(R_U, _RU, system), n(r.M, _M, system)),
                    q(r.R, _S, system),
                ),
            ),
        )
    )

    cp_terms = [times(_r(c.x, 4), n(c.cp, _S, system)) for c in comps]
    steps.append(
        ProcedureStep(
            "Calores específicos de la mezcla",
            f"Con el c_p de cada gas a la temperatura de la mezcla ({_temp_txt(r.inputs.T_K)}): "
            "c_p,M = Σ x_i·c_p,i, c_v,M = c_p,M − R_M y k_M = c_p,M/c_v,M (vademecum §5.6).",
            (
                latex_chain(
                    r"c_{p,M}", r"\sum x_i c_{p,i}", sum_rows(cp_terms, 1), q(r.cp, _S, system)
                ),
                latex_chain(
                    r"c_{v,M}", sub(n(r.cp, _S, system), n(r.R, _S, system)), q(r.cv, _S, system)
                ),
                latex_chain("k_M", frac(n(r.cp, _S, system), n(r.cv, _S, system)), _r(r.k, 5)),
            ),
        )
    )

    p = r.inputs.p_Pa
    lines = [
        latex_chain(
            f"p_{{{gas_symbol(c.gas.key)}}}",
            times(_r(c.y, 4), n(p, _P, system)),
            q(c.p_i, _P, system),
        )
        for c in comps
    ]
    text = "La presión parcial de cada gas es p_i = y_i·p (Dalton, vademecum §5.3)."
    if r.V_m3 is not None and r.m_kg is not None:
        _, f = _pv(system)
        lines.append(
            latex_chain(
                "V",
                frac(r"m_M\,R_M\,T", "p"),
                frac(
                    times(
                        n(r.m_kg, "mass", system), n(r.R, _S, system), n(r.inputs.T_K, _T, system)
                    ),
                    _with_factor(f, n(p, _P, system)),
                ),
                q(r.V_m3, "volume", system),
            )
        )
        lines += [
            latex_chain(
                f"V_{{{gas_symbol(c.gas.key)}}}",
                times(_r(c.y, 4), n(r.V_m3, "volume", system)),
                q(c.V_i or 0.0, "volume", system),
            )
            for c in comps
        ]
        text += (
            " El volumen parcial es V_i = y_i·V (Amagat, §5.4): el que ocuparía cada gas solo a "
            "la presión y la temperatura de la mezcla." + _pv_text(system)
        )
    steps.append(ProcedureStep("Dalton y Amagat", text, tuple(lines)))

    s_lines = []
    for c in comps:
        if c.y <= 0.0:
            continue
        s = gas_symbol(c.gas.key)
        s0 = c.gas.s0(r.inputs.T_K)
        s_lines.append(
            latex_chain(
                f"s_{{{s}}}",
                rf"s^{{\circ}} - R \ln\frac{{p_{{{s}}}}}{{p^{{\circ}}}}",
                rf"{n(s0, _S, system)} \\ &\quad - {n(c.gas.R, _S, system)} "
                rf"\ln {_r(c.p_i / 1e5, 5)}",
                q(c.s_i, _S, system),
            )
        )
    s_terms = [times(_r(c.x, 4), n(c.s_i, _S, system)) for c in comps if c.y > 0.0]
    s_lines.append(latex_chain("s_M", r"\sum x_i s_i", sum_rows(s_terms, 1), q(r.s, _S, system)))
    present = [c for c in comps if c.y > 0.0]
    y_ln_y = sum(c.y * math.log(c.y) for c in present)
    s_lines.append(
        latex_chain(
            r"\sum y_i \ln y_i",
            sum_rows([rf"{_r(c.y, 4)} \ln {_r(c.y, 4)}" for c in present], 1),
            _r(y_ln_y, 5),
        )
    )
    s_lines.append(
        latex_chain(
            r"\Delta s_{\text{mez}}",
            r"-R_M \sum y_i \ln y_i",
            times("-" + n(r.R, _S, system), _r(y_ln_y, 5)),
            q(r.s_mixing, _S, system),
        )
    )
    steps.append(
        ProcedureStep(
            "Entropía de la mezcla",
            "Cada gas a su presión parcial: s_M = Σ x_i·s_i(T, p_i) = Σ x_i·[s_i(T, p) − R_i·ln "
            "y_i] (vademecum §5.7), con s = 0 a 25 °C y p° = 1 bar. El término −Σ x_i·R_i·ln y_i "
            "es lo que suma mezclar; como x_i·R_i = y_i·R_M, es −R_M·Σ y_i·ln y_i.",
            tuple(s_lines),
        )
    )
    return numbered(steps)


def _amount_n(n_mol: float, system: UnitSystem) -> str:
    """Moles (o moles por segundo) en el sistema: mol, kmol o lbmol."""
    return n(n_mol, "amount", system)


def _amount_q(n_mol: float, tank: bool, system: UnitSystem) -> str:
    """Moles en un tanque o caudal molar en un flujo, con su unidad."""
    label = unit_label("amount", system) + ("" if tank else "/s")
    return rf"{_amount_n(n_mol, system)}\ {latex_unit(label)}"


def _den_value(den_si: float, system: UnitSystem) -> float:
    """Σ x/M en las unidades del sistema (1/(kg/mol) en el SI, 1/(kg/kmol) si no)."""
    return den_si if system == "SI" else den_si / 1000.0


def _temp_txt(T_K: float) -> str:
    return f"{T_K:.1f} K".replace(".", ",")


# ---------------------------------------------------------------------
# Mezcla adiabática
# ---------------------------------------------------------------------


def mixing_steps(r: MixingResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de la mezcla adiabática (Çengel §13-3; vademecum §5 y §11)."""
    tank = r.inputs.kind == "tank"
    model = r.inputs.model
    amount_kind = "mass" if tank else "mass_flow"
    m = "m" if tank else r"\dot{m}"
    m_txt = "m" if tank else "ṁ"
    steps: list[ProcedureStep] = []
    _, f = _pv(system)
    if tank:
        lines = []
        for j, st in enumerate(r.streams, 1):
            assert st.V_m3 is not None
            lines.append(
                latex_chain(
                    f"V_{j}",
                    frac(rf"m_{j}\,R_{j}\,T_{j}", f"p_{j}"),
                    frac(
                        times(
                            n(st.amount, "mass", system),
                            n(st.gas.R, _S, system),
                            n(st.T_K, _T, system),
                        ),
                        _with_factor(f, n(st.p_Pa, _P, system)),
                    ),
                    q(st.V_m3, "volume", system),
                )
            )
        assert r.V_m3 is not None
        lines.append(latex_chain("V", r"\sum V_j", q(r.V_m3, "volume", system)))
        steps.append(
            ProcedureStep(
                "El volumen del tanque",
                "Cada compartimiento con p·V = m·R·T; al sacar el tabique el gas ocupa todo el "
                "tanque." + _pv_text(system),
                tuple(lines),
            )
        )

    energy = "u" if tank else "h"
    c = "c_v" if tank else "c_p"
    big = "U" if tank else "H"
    cap = "entropy" if tank else "entropy_flow"
    total = "energy" if tank else "power"
    if model == "constant":
        cs = [(st.gas.cp_ref - st.gas.R) if tank else st.gas.cp_ref for st in r.streams]
        mc = [st.amount * cj for st, cj in zip(r.streams, cs, strict=True)]
        num = sum(x * st.T_K for x, st in zip(mc, r.streams, strict=True))
        lines = [
            latex_chain(
                rf"({m}\,{c})_{j}",
                times(n(st.amount, amount_kind, system), n(cj, _S, system)),
                q(x, cap, system),
            )
            for j, (st, cj, x) in enumerate(zip(r.streams, cs, mc, strict=True), 1)
        ]
        lines.append(
            latex_chain(
                rf"\sum ({m}\,{c})_j\,T_j",
                sum_rows(
                    [
                        times(n(x, cap, system), n(st.T_K, _T, system))
                        for x, st in zip(mc, r.streams, strict=True)
                    ],
                    1,
                ),
                q(num, total, system),
            )
        )
        lines.append(
            latex_chain(
                "T",
                frac(rf"\sum ({m}\,{c})_j\,T_j", rf"\sum ({m}\,{c})_j"),
                frac(n(num, total, system), n(sum(mc), cap, system)),
                q(r.T_K, _T, system),
            )
        )
        text = (
            f"{big} no cambia: Σ ({m_txt}·{c})_j·(T − T_j) = 0, así que T es el promedio de las "
            f"T_j pesado con {m_txt}·{c}, con las temperaturas absolutas y c constante (el de "
            "25 °C)."
        )
    else:

        def e(st: StreamResult, T: float) -> float:
            return st.gas.u(T) if tank else st.gas.h(T)

        def terms(T_of: Callable[[StreamResult], float]) -> str:
            return sum_rows(
                [
                    times(n(st.amount, amount_kind, system), n(e(st, T_of(st)), _E, system))
                    for st in r.streams
                ],
                1,
            )

        lines = []
        for j, st in enumerate(r.streams, 1):
            lines.append(latex_chain(rf"{energy}_{j}(T_{j})", q(e(st, st.T_K), _E, system)))
        before = sum(st.amount * e(st, st.T_K) for st in r.streams)
        lines.append(
            latex_chain(
                rf"\sum {m}_j\,{energy}_j(T_j)",
                terms(lambda st: st.T_K),
                q(before, total, system),
            )
        )
        lines.append(latex_chain("T", q(r.T_K, _T, system)))
        for j, st in enumerate(r.streams, 1):
            lines.append(latex_chain(rf"{energy}_{j}(T)", q(e(st, r.T_K), _E, system)))
        after = sum(st.amount * e(st, r.T_K) for st in r.streams)
        lines.append(
            latex_chain(
                rf"\sum {m}_j\,{energy}_j(T)",
                terms(lambda st: r.T_K),
                q(after, total, system),
            )
        )
        text = (
            f"{big} no cambia: T es la que hace Σ {m_txt}_j·{energy}_j(T) igual a lo que entra. "
            f"{energy}(T) sale del polinomio NASA de cada gas, con {energy} = 0 a 25 °C (cada "
            "gas conserva su masa, así que la referencia se cancela), y T se busca con una "
            "iteración: el último renglón lo verifica."
        )
    steps.append(ProcedureStep("La temperatura final", text, tuple(lines)))

    if tank:
        assert r.V_m3 is not None
        mR = sum(st.amount * st.gas.R for st in r.streams)
        steps.append(
            ProcedureStep(
                "La presión final",
                "Con toda la masa en el volumen del tanque: p = (Σ m_j·R_j)·T/V."
                + _pv_text(system),
                (
                    latex_chain(r"\sum m_j R_j", q(mR, "entropy", system)),
                    latex_chain(
                        "p",
                        frac(r"(\sum m_j R_j)\,T", "V"),
                        frac(
                            times(n(mR, "entropy", system), n(r.T_K, _T, system)),
                            _with_factor(f, n(r.V_m3, "volume", system)),
                        ),
                        q(r.p_Pa, _P, system),
                    ),
                ),
            )
        )

    by_gas: dict[str, tuple[StreamResult, float]] = {}
    for st in r.streams:
        first, mass = by_gas.get(st.gas.key, (st, 0.0))
        by_gas[st.gas.key] = (first, mass + st.amount)
    N = "N" if tank else r"\dot{N}"
    moles = {key: mass / st.gas.M for key, (st, mass) in by_gas.items()}
    y_lines = []
    for key, (st, mass) in by_gas.items():
        s = gas_symbol(key)
        y_lines.append(
            latex_chain(
                f"{N}_{{{s}}}",
                frac(f"{m}_{{{s}}}", f"M_{{{s}}}"),
                frac(n(mass, amount_kind, system), n(st.gas.M, _M, system)),
                _amount_q(moles[key], tank, system),
            )
        )
    y_lines.append(latex_chain(N, rf"\sum {N}_i", _amount_q(sum(moles.values()), tank, system)))
    for key, (st, _mass) in by_gas.items():
        s = gas_symbol(key)
        y_lines.append(
            latex_chain(
                f"y_{{{s}}}",
                frac(f"{N}_{{{s}}}", N),
                frac(_amount_n(moles[key], system), _amount_n(sum(moles.values()), system)),
                _r(st.y, 4),
            )
        )
        y_lines.append(
            latex_chain(
                f"p_{{{s}}}",
                times(_r(st.y, 4), n(r.p_Pa, _P, system)),
                q(st.p_final_Pa, _P, system),
            )
        )
    steps.append(
        ProcedureStep(
            "La composición final",
            "La fracción molar de cada gas en la mezcla, y_i = N_i/N con N_i = m_i/M_i, y su "
            "presión parcial, p_i = y_i·p (Dalton, vademecum §5.3). Dos corrientes del mismo gas "
            "suman sus moles.",
            tuple(y_lines),
        )
    )

    s_lines = []
    for j, st in enumerate(r.streams, 1):
        if model == "constant":
            body = (
                rf"{n(st.gas.cp_ref, _S, system)} \ln {_r(r.T_K / st.T_K)} \\ &\quad - "
                rf"{n(st.gas.R, _S, system)} \ln {_r(r.p_Pa / st.p_Pa)}"
            )
            formula = rf"c_{{p,{j}}} \ln\frac{{T}}{{T_{j}}} - R_{j} \ln\frac{{p}}{{p_{j}}}"
        else:
            s0f, s0i = st.gas.s0(r.T_K), st.gas.s0(st.T_K)
            body = (
                rf"{sub(n(s0f, _S, system), n(s0i, _S, system))} \\ &\quad - "
                rf"{n(st.gas.R, _S, system)} \ln {_r(r.p_Pa / st.p_Pa)}"
            )
            formula = rf"s^{{\circ}}(T) - s^{{\circ}}(T_{j}) - R_{j} \ln\frac{{p}}{{p_{j}}}"
        s_lines.append(latex_chain(rf"\Delta s_{{TP,{j}}}", formula, body, q(st.ds_TP, _S, system)))
        s_lines.append(
            latex_chain(
                rf"\Delta s_{{\text{{mez}},{j}}}",
                rf"-R_{j} \ln y_{j}",
                rf"-{n(st.gas.R, _S, system)} \ln {_r(st.y, 4)}",
                q(st.ds_mix, _S, system),
            )
        )
        s_lines.append(
            latex_chain(
                rf"\Delta s_{j}",
                rf"\Delta s_{{TP,{j}}} + \Delta s_{{\text{{mez}},{j}}}",
                rf"{n(st.ds_TP, _S, system)} + {latex_paren(n(st.ds_mix, _S, system))}",
                q(st.ds_TP + st.ds_mix, _S, system),
            )
        )
    s_kind = "entropy" if tank else "entropy_flow"
    terms = [
        times(n(st.amount, amount_kind, system), n(st.ds_TP + st.ds_mix, _S, system))
        for st in r.streams
    ]
    s_lines.append(
        latex_chain(
            r"S_{\text{gen}}" if tank else r"\dot{S}_{\text{gen}}",
            rf"\sum {m}_j\,\Delta s_j",
            sum_rows(terms, 1),
            q(r.S_gen, s_kind, system),
        )
    )
    steps.append(
        ProcedureStep(
            "La entropía generada",
            "Cada corriente va de (T_j, p_j) a la T final y a su presión parcial: Δs_TP la lleva "
            "a la T y la p de la mezcla y Δs_mez = −R_j·ln y_j es lo que suma mezclarse "
            "(vademecum §5.7). La mezcla es adiabática, así que toda la Δs de las corrientes es "
            "entropía generada.",
            tuple(s_lines),
        )
    )
    x_kind = "energy" if tank else "power"
    steps.append(
        ProcedureStep(
            "La exergía destruida",
            "Gouy–Stodola: la exergía destruida es T₀ por la entropía generada (vademecum §11).",
            (
                latex_chain(
                    r"X_{\text{dest}}" if tank else r"\dot{X}_{\text{dest}}",
                    r"T_0\,S_{\text{gen}}" if tank else r"T_0\,\dot{S}_{\text{gen}}",
                    times(n(r.inputs.T0_K, _T, system), n(r.S_gen, s_kind, system)),
                    q(r.X_dest, x_kind, system),
                ),
            ),
        )
    )
    return numbered(steps)


# ---------------------------------------------------------------------
# Transformación politrópica
# ---------------------------------------------------------------------


def _ratio_formula(final: str, exponent: str) -> tuple[str, str]:
    """La relación de T del dato final (``p2`` o ``ratio``) con el exponente dado."""
    if final == "p2":
        return (
            rf"T_1 \left(\frac{{p_2}}{{p_1}}\right)^{{({exponent}-1)/{exponent}}}",
            "p",
        )
    return rf"T_1 \left(\frac{{v_1}}{{v_2}}\right)^{{{exponent}-1}}", "v"


def process_steps(r: ProcessResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de una transformación politrópica (vademecum §3.5, §6 y §10.5)."""
    inp = r.inputs
    g = r.gas
    model = inp.model
    kind = inp.process
    s1, s2 = r.state1, r.state2
    steps = [_gas_step(g, system, with_cp=model == "constant")]
    if model == "variable":
        steps[0] = ProcedureStep(
            steps[0].title,
            steps[0].text + " Con c_p variable, u, h y s° salen del polinomio NASA (§4.7).",
            steps[0].latex,
        )
    steps.append(
        ProcedureStep(
            "El estado 1",
            "v₁ = R·T₁/p₁ (vademecum §4.2), con la temperatura absoluta." + _pv_text(system),
            (_v_from_pT(g, s1.p_Pa, s1.T_K, system, "1"),),
        )
    )

    T1t, T2t = n(s1.T_K, _T, system), n(s2.T_K, _T, system)
    p1t, p2t = n(s1.p_Pa, _P, system), n(s2.p_Pa, _P, system)
    ratio = s1.v_m3_per_kg / s2.v_m3_per_kg
    lines: list[str] = []
    title = f"El estado 2: {PROCESS_KINDS[kind].split(' (')[0].lower()}"
    if kind == "isochoric":
        if inp.final == "p2":
            lines.append(
                latex_chain(
                    "T_2",
                    r"T_1\,\frac{p_2}{p_1}",
                    times(T1t, frac(p2t, p1t)),
                    q(s2.T_K, _T, system),
                )
            )
        else:
            lines.append(
                latex_chain(
                    "p_2",
                    r"p_1\,\frac{T_2}{T_1}",
                    times(p1t, frac(T2t, T1t)),
                    q(s2.p_Pa, _P, system),
                )
            )
        text = "En la isócora v₂ = v₁, así que T₂/T₁ = p₂/p₁ (vademecum §6.5)."
    elif kind == "isobaric":
        if inp.final == "ratio":
            lines.append(
                latex_chain(
                    "T_2", r"\frac{T_1}{v_1/v_2}", frac(T1t, _r(ratio)), q(s2.T_K, _T, system)
                )
            )
        text = "En la isóbara p₂ = p₁, así que T₂/T₁ = v₂/v₁ (vademecum §6.5)."
    elif kind == "isothermal":
        if inp.final == "ratio":
            lines.append(
                latex_chain(
                    "p_2", r"p_1\,\frac{v_1}{v_2}", times(p1t, _r(ratio)), q(s2.p_Pa, _P, system)
                )
            )
        text = "En la isoterma T₂ = T₁, así que p₁·v₁ = p₂·v₂ (vademecum §6.5)."
    elif kind == "adiabatic" and model == "variable":
        R = g.R
        s01, s02 = g.s0(s1.T_K), g.s0(s2.T_K)
        lines.append(latex_chain(r"s^{\circ}(T_1)", q(s01, _S, system)))
        if inp.final == "T2":
            lines.append(latex_chain(r"s^{\circ}(T_2)", q(s02, _S, system)))
            lines.append(
                latex_chain(
                    "p_2",
                    r"p_1\,e^{(s^{\circ}_2 - s^{\circ}_1)/R}",
                    times(p1t, rf"e^{{{_r((s02 - s01) / R, 5)}}}"),
                    q(s2.p_Pa, _P, system),
                )
            )
            text = (
                "Con c_p variable la adiabática reversible cumple Δs = 0: s°(T₂) − s°(T₁) = "
                "R·ln(p₂/p₁) (vademecum §10.5.3), así que p₂ = p₁·exp((s°₂ − s°₁)/R)."
            )
        elif inp.final == "p2":
            lines.append(
                latex_chain(
                    r"s^{\circ}(T_2)",
                    r"s^{\circ}(T_1) + R \ln\frac{p_2}{p_1}",
                    rf"{n(s01, _S, system)} \\ &\quad + {n(R, _S, system)} "
                    rf"\ln {_r(s2.p_Pa / s1.p_Pa)}",
                    q(s02, _S, system),
                )
            )
            lines.append(latex_chain("T_2", q(s2.T_K, _T, system)))
            text = (
                "Con c_p variable la adiabática reversible cumple Δs = 0: s°(T₂) = s°(T₁) + "
                "R·ln(p₂/p₁) (vademecum §10.5.3), y T₂ es la del polinomio que da ese s° (se "
                "busca con una iteración)."
            )
        else:
            pr1, vr1, vr2 = g.pr(s1.T_K), g.vr(s1.T_K), g.vr(s2.T_K)
            lines += [
                latex_chain(
                    "p_{r1}",
                    r"e^{s^{\circ}_1/R}",
                    rf"e^{{{_r(s01 / R, 5)}}}",
                    _r(pr1, 5),
                ),
                latex_chain(
                    "v_{r1}",
                    frac("T_1", "p_{r1}"),
                    frac(T1t, _r(pr1, 5)),
                    q(vr1, _T, system),
                ),
                latex_chain(
                    "v_{r2}",
                    frac("v_{r1}", "v_1/v_2"),
                    frac(n(vr1, _T, system), _r(ratio)),
                    q(vr2, _T, system),
                ),
                latex_chain("T_2", q(s2.T_K, _T, system)),
                latex_chain(
                    "p_2",
                    r"p_1\,\frac{v_1}{v_2}\,\frac{T_2}{T_1}",
                    times(p1t, _r(ratio), frac(T2t, T1t)),
                    q(s2.p_Pa, _P, system),
                ),
            ]
            text = (
                "Con c_p variable la adiabática reversible cumple v₂/v₁ = v_r2/v_r1, con el "
                "volumen relativo v_r = T/p_r y p_r = exp(s°/R) (vademecum §10.5.3; Çengel §7-9): "
                "T₂ es la del polinomio que da ese v_r (se busca con una iteración). p₂ sale de "
                "p·v = R·T."
            )
    else:
        exp_sym = "n" if kind == "polytropic" else "k"
        e_val = r.n
        if inp.final in ("p2", "ratio"):
            formula, _ = _ratio_formula(inp.final, exp_sym)
            base = frac(p2t, p1t) if inp.final == "p2" else _r(ratio)
            power = (e_val - 1.0) / e_val if inp.final == "p2" else e_val - 1.0
            lines.append(
                latex_chain(
                    "T_2",
                    formula,
                    times(T1t, rf"\left({base}\right)^{{{_r(power, 4)}}}"),
                    q(s2.T_K, _T, system),
                )
            )
            if inp.final == "ratio":
                lines.append(
                    latex_chain(
                        "p_2",
                        rf"p_1 \left(\frac{{v_1}}{{v_2}}\right)^{{{exp_sym}}}",
                        times(p1t, rf"{_r(ratio)}^{{{_r(e_val, 4)}}}"),
                        q(s2.p_Pa, _P, system),
                    )
                )
        else:
            lines.append(
                latex_chain(
                    "p_2",
                    rf"p_1 \left(\frac{{T_2}}{{T_1}}\right)^{{{exp_sym}/({exp_sym}-1)}}",
                    times(p1t, rf"\left({frac(T2t, T1t)}\right)^{{{_r(e_val / (e_val - 1), 4)}}}"),
                    q(s2.p_Pa, _P, system),
                )
            )
        if kind == "polytropic":
            text = f"p·vⁿ = cte con n = {_r_txt(e_val)}: T·p^((1−n)/n) = cte (vademecum §6.1)."
        else:
            text = (
                f"Con c_p constante la adiabática reversible es p·v^k = cte, con k = "
                f"{_r_txt(e_val)} (vademecum §6.4)."
            )
    lines.append(_v_from_pT(g, s2.p_Pa, s2.T_K, system, "2"))
    steps.append(ProcedureStep(title, text, tuple(lines)))

    dT_tex = sub(T2t, T1t)
    if model == "constant":
        cp, cv = g.cp_ref, g.cp_ref - g.R
        en_lines = [
            latex_chain(
                r"\Delta u",
                r"c_v\,(T_2 - T_1)",
                times(n(cv, _S, system), f"({dT_tex})"),
                q(r.du, _E, system),
            ),
            latex_chain(
                r"\Delta h",
                r"c_p\,(T_2 - T_1)",
                times(n(cp, _S, system), f"({dT_tex})"),
                q(r.dh, _E, system),
            ),
        ]
        en_text = "Con c_p constante, Δu = c_v·ΔT y Δh = c_p·ΔT (vademecum §6.5)."
    else:
        en_lines = [
            latex_chain(
                r"\Delta u",
                "u(T_2) - u(T_1)",
                sub(n(g.u(s2.T_K), _E, system), n(g.u(s1.T_K), _E, system)),
                q(r.du, _E, system),
            ),
            latex_chain(
                r"\Delta h",
                "h(T_2) - h(T_1)",
                sub(n(g.h(s2.T_K), _E, system), n(g.h(s1.T_K), _E, system)),
                q(r.dh, _E, system),
            ),
        ]
        en_text = "Con c_p variable, u y h salen del polinomio NASA (h = u + R·T)."
    steps.append(ProcedureStep("Energía interna y entalpía", en_text, tuple(en_lines)))

    R = g.R
    _, f = _pv(system)
    w_lines: list[str] = []
    if kind == "isochoric":
        w_lines.append(latex_chain("w", r"\int p\,dv", q(0.0, _E, system)))
        w_lines.append(
            latex_chain(
                "w_f",
                r"v\,(p_1 - p_2)",
                _with_factor(f, n(s1.v_m3_per_kg, _V, system))
                + (r" \\ &\quad" if f or latex_is_wide(p1t, p2t) else "")
                + rf" \cdot ({sub(p1t, p2t)})",
                q(r.w_f, _E, system),
            )
        )
        w_text = (
            "En la isócora no hay trabajo de expansión; el de circulación es w_f = v·(p₁ − p₂)."
        )
        w_text += _pv_text(system)
    elif kind == "isobaric":
        w_lines.append(
            latex_chain(
                "w",
                r"p\,(v_2 - v_1) = R\,(T_2 - T_1)",
                times(n(R, _S, system), f"({dT_tex})"),
                q(r.w, _E, system),
            )
        )
        w_lines.append(latex_chain("w_f", r"-\int v\,dp", q(0.0, _E, system)))
        w_text = "En la isóbara w = p·(v₂ − v₁) = R·(T₂ − T₁) y w_f = 0 (vademecum §6.5)."
    elif kind == "isothermal":
        w_lines.append(
            latex_chain(
                "w",
                r"R\,T \ln\frac{v_2}{v_1}",
                times(n(R, _S, system), T1t, rf"\ln {_r(1.0 / ratio)}"),
                q(r.w, _E, system),
            )
        )
        w_lines.append(latex_chain("w_f", "w", q(r.w_f, _E, system)))
        w_text = "En la isoterma w = w_f = R·T·ln(v₂/v₁) = R·T·ln(p₁/p₂) (vademecum §6.2)."
    elif kind == "adiabatic":
        w_lines.append(latex_chain("w", r"-\Delta u", q(r.w, _E, system)))
        w_lines.append(latex_chain("w_f", r"-\Delta h", q(r.w_f, _E, system)))
        w_text = (
            "Sin calor, el trabajo sale del primer principio: w = −Δu en un sistema cerrado y "
            "w_f = −Δh en uno abierto (vademecum §3)."
        )
    else:
        nn = r.n
        w_lines.append(
            latex_chain(
                "w",
                frac(r"R\,(T_2 - T_1)", "1 - n"),
                frac(times(n(R, _S, system), f"({dT_tex})"), sub("1", _r(nn, 4))),
                q(r.w, _E, system),
            )
        )
        w_lines.append(
            latex_chain("w_f", r"n\,w", times(_r(nn, 4), n(r.w, _E, system)), q(r.w_f, _E, system))
        )
        w_text = "En la politrópica w = R·(T₂ − T₁)/(1 − n) y w_f = n·w (vademecum §6.2 y §6.3)."
    w_text += " Con la convención del vademecum, positivo si lo hace el gas."
    steps.append(ProcedureStep("Trabajo", w_text, tuple(w_lines)))

    q_lines: list[str] = []
    if kind == "adiabatic":
        q_lines.append(latex_chain("q", q(0.0, _E, system)))
        q_text = "Adiabática: q = 0."
    elif kind == "isothermal":
        q_lines.append(latex_chain("q", "w", q(r.q, _E, system)))
        q_text = "En la isoterma Δu = 0: q = w."
    else:
        if model == "constant" and kind == "polytropic":
            cv, k = g.cp_ref - g.R, r.k
            q_lines.append(
                latex_chain(
                    "c",
                    r"c_v\,\frac{n - k}{n - 1}",
                    times(n(cv, _S, system), frac(sub(_r(r.n, 4), _r(k, 5)), sub(_r(r.n, 4), "1"))),
                    q(r.c or 0.0, _S, system),
                )
            )
            q_lines.append(
                latex_chain(
                    "q",
                    r"c\,(T_2 - T_1)",
                    times(n(r.c or 0.0, _S, system), f"({dT_tex})"),
                    q(r.q, _E, system),
                )
            )
            q_text = "El calor de la politrópica es q = c·ΔT con c = c_v·(n − k)/(n − 1) (§6.5)."
        else:
            q_lines.append(
                latex_chain(
                    "q",
                    r"\Delta u + w",
                    rf"{n(r.du, _E, system)} + {_paren(n(r.w, _E, system))}",
                    q(r.q, _E, system),
                )
            )
            q_text = "Primer principio en el sistema cerrado: q = Δu + w (vademecum §3.1)."
    steps.append(ProcedureStep("Calor", q_text, tuple(q_lines)))

    if model == "constant":
        ds_line = _ds_constant(g.cp_ref, R, (s2.T_K / s1.T_K, s2.p_Pa / s1.p_Pa), r.ds, system)
        ds_text = "Δs = c_p·ln(T₂/T₁) − R·ln(p₂/p₁) (vademecum §10.5.1)."
    else:
        s01, s02 = g.s0(s1.T_K), g.s0(s2.T_K)
        ds_line = latex_chain(
            r"\Delta s",
            r"s^{\circ}_2 - s^{\circ}_1 - R \ln\frac{p_2}{p_1}",
            rf"{sub(n(s02, _S, system), n(s01, _S, system))} \\ &\quad - "
            rf"{n(R, _S, system)} \ln {_r(s2.p_Pa / s1.p_Pa)}",
            q(r.ds, _S, system),
        )
        ds_text = "Δs = s°₂ − s°₁ − R·ln(p₂/p₁) (vademecum §10.5.2)."
    if kind == "adiabatic":
        ds_text += " En la adiabática reversible da 0."
    steps.append(ProcedureStep("Entropía", ds_text, (ds_line,)))

    amount_kind = "mass" if inp.system == "closed" else "mass_flow"
    total_kind = "energy" if inp.system == "closed" else "power"
    W = "W" if inp.system == "closed" else r"\dot{W}_f"
    Q = "Q" if inp.system == "closed" else r"\dot{Q}"
    w_each = r.w if inp.system == "closed" else r.w_f
    steps.append(
        ProcedureStep(
            "Para toda la masa" if inp.system == "closed" else "Para todo el caudal",
            "Sistema cerrado: W = m·w y Q = m·q."
            if inp.system == "closed"
            else "Sistema abierto en régimen permanente: Ẇ_f = ṁ·w_f y Q̇ = ṁ·q.",
            (
                latex_chain(
                    W,
                    times(n(inp.amount, amount_kind, system), n(w_each, _E, system)),
                    q(r.W_total, total_kind, system),
                ),
                latex_chain(
                    Q,
                    times(n(inp.amount, amount_kind, system), n(r.q, _E, system)),
                    q(r.Q_total, total_kind, system),
                ),
            ),
        )
    )
    return numbered(steps)


def _paren(tex: str) -> str:
    return f"({tex})" if tex.startswith("-") else tex


def _r_txt(x: float) -> str:
    return f"{x:.4g}".replace(".", ",")


# ---------------------------------------------------------------------
# Etapas y exponente
# ---------------------------------------------------------------------


def staged_steps(r: StagedResult, system: UnitSystem) -> list[ProcedureStep]:
    """Compresión en N etapas con interenfriamiento perfecto (vademecum §6.3.1)."""
    inp = r.inputs
    g = r.gas
    N = inp.stages
    nn = inp.n
    isothermal = abs(nn - 1.0) < 1e-9
    p1t, p2t = n(inp.p1_Pa, _P, system), n(inp.p2_Pa, _P, system)
    T1t = n(inp.T1_K, _T, system)
    exp = (nn - 1.0) / nn
    steps: list[ProcedureStep] = []

    lines = [
        latex_chain(
            "r",
            r"\left(\frac{p_2}{p_1}\right)^{1/N}",
            rf"\left({frac(p2t, p1t)}\right)^{{1/{N}}}",
            _r(r.stage_ratio, 5),
        )
    ]
    for j, px in enumerate(r.pressures_Pa[1:-1], 1):
        lines.append(latex_chain(f"p_{{x,{j}}}", rf"p_1\,r^{{{j}}}", q(px, _P, system)))
    steps.append(
        ProcedureStep(
            "Las presiones intermedias" if N > 1 else "La relación de presiones",
            "La misma relación de presiones en cada etapa, r = (p₂/p₁)^(1/N), da el mínimo "
            "trabajo (con dos etapas, p_x = √(p₁·p₂)) (vademecum §6.3.1).",
            tuple(lines),
        )
    )

    if isothermal:
        lines = [latex_chain("T_x", "T_1", q(r.T_out_K, _T, system))]
    else:
        lines = [
            latex_chain(
                "T_x",
                r"T_1\,r^{(n-1)/n}",
                times(T1t, rf"{_r(r.stage_ratio, 5)}^{{{_r(exp, 4)}}}"),
                q(r.T_out_K, _T, system),
            )
        ]
    if isothermal:
        lines.append(
            latex_chain(
                "w_f",
                r"-N\,R\,T_1 \ln r",
                times(f"-{N}", n(g.R, _S, system), T1t, rf"\ln {_r(r.stage_ratio, 5)}"),
                q(r.w_f, _E, system),
            )
        )
    else:
        coef = N * nn * g.R * inp.T1_K / (nn - 1.0)
        coef_t = n(coef, _E, system)
        lines.append(
            latex_chain(
                "a",
                frac(r"N\,n\,R\,T_1", "n - 1"),
                frac(times(str(N), _r(nn, 4), n(g.R, _S, system), T1t), sub(_r(nn, 4), "1")),
                q(coef, _E, system),
            )
        )
        lines.append(
            latex_chain(
                "w_f",
                r"-a\left[r^{(n-1)/n} - 1\right]",
                rf"-{coef_t}"
                + (r" \\ &\quad" if latex_is_wide(coef_t) else "")
                + rf" \cdot \left[{_r(r.stage_ratio, 5)}^{{{_r(exp, 4)}}} - 1\right]",
                q(r.w_f, _E, system),
            )
        )
    steps.append(
        ProcedureStep(
            "Temperatura y trabajo",
            "Con interenfriamiento perfecto cada etapa arranca a T₁, así que todas salen a la "
            "misma T_x y hacen el mismo trabajo: w_f = −N·n·R·T₁/(n − 1)·[r^((n−1)/n) − 1] "
            "(vademecum §6.3.1), negativo porque lo recibe el gas."
            if not isothermal
            else "Con n = 1 cada etapa es una isoterma a T₁: w_f = −N·R·T₁·ln r = −R·T₁·ln(p₂/p₁), "
            "igual que en una etapa (vademecum §6.3.1).",
            tuple(lines),
        )
    )

    if N > 1 and not isothermal:
        cp = g.cp_ref
        steps.append(
            ProcedureStep(
                "Los interenfriadores",
                ("El interenfriador lleva" if N == 2 else f"Los {N - 1} interenfriadores llevan")
                + " el gas de T_x a T₁ a presión constante: q_int = −(N − 1)·c_p·(T_x − T₁), con "
                "el c_p de 25 °C (sale del gas).",
                (
                    latex_chain(
                        r"q_{\text{int}}",
                        r"-(N - 1)\,c_p\,(T_x - T_1)",
                        times(
                            f"-{N - 1}",
                            n(cp, _S, system),
                            f"({sub(n(r.T_out_K, _T, system), T1t)})",
                        ),
                        q(r.q_intercoolers, _E, system),
                    ),
                ),
            )
        )

    if N > 1:
        single = r"\text{1 etapa}"
        lines = []
        if not isothermal:
            lines.append(
                latex_chain(
                    rf"T_{{2,\,{single}}}",
                    r"T_1 \left(\frac{p_2}{p_1}\right)^{(n-1)/n}",
                    times(T1t, rf"\left({frac(p2t, p1t)}\right)^{{{_r(exp, 4)}}}"),
                    q(r.T_out_single_K, _T, system),
                )
            )
        lines += [
            latex_chain(rf"w_{{f,\,{single}}}", q(r.w_f_single, _E, system)),
            latex_chain(
                r"\text{ahorro}",
                rf"1 - \frac{{w_f}}{{w_{{f,\,{single}}}}}",
                _r(r.saving * 100.0, 4) + r"\ \%",
            ),
        ]
        steps.append(
            ProcedureStep(
                "Contra una sola etapa",
                "La misma compresión en una etapa (la fórmula con N = 1 y r = p₂/p₁): el gas "
                "sale más caliente y el trabajo es mayor."
                if not isothermal
                else "Con n = 1 partir la compresión no ahorra nada: cada etapa ya es isotérmica.",
                tuple(lines),
            )
        )
    return numbered(steps)


def exponent_steps(
    e: ExponentResult,
    system: UnitSystem,
    p1_Pa: float,
    p2_Pa: float,
    *,
    T1_K: float | None = None,
    T2_K: float | None = None,
    v1: float | None = None,
    v2: float | None = None,
) -> list[ProcedureStep]:
    """El n de dos estados medidos (vademecum §6.1)."""
    if v1 is not None and v2 is not None:
        lines = (
            latex_chain(
                "n",
                frac(r"\ln(p_1/p_2)", r"\ln(v_2/v_1)"),
                frac(rf"\ln {_r(p1_Pa / p2_Pa)}", rf"\ln {_r(v2 / v1)}"),
                _r(e.n, 5),
            ),
        )
    else:
        assert T1_K is not None and T2_K is not None
        a = math.log(T2_K / T1_K) / math.log(p2_Pa / p1_Pa)
        lines = (
            latex_chain(
                r"\frac{n-1}{n}",
                frac(r"\ln(T_2/T_1)", r"\ln(p_2/p_1)"),
                frac(rf"\ln {_r(T2_K / T1_K)}", rf"\ln {_r(p2_Pa / p1_Pa)}"),
                _r(a, 5),
            ),
            latex_chain("n", frac("1", "1 - (n-1)/n"), frac("1", sub("1", _r(a, 5))), _r(e.n, 5)),
        )
    return numbered(
        [
            ProcedureStep(
                "El exponente de dos estados",
                "De p₁·v₁ⁿ = p₂·v₂ⁿ, o de T·p^((1−n)/n) = cte con las temperaturas absolutas "
                f"(vademecum §6.1). {e.text}",
                lines,
            )
        ]
    )
