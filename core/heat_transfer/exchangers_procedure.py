"""Procedimiento didáctico de los intercambiadores (LaTeX), en el sistema de unidades activo.

Cengel y Ghajar (2015), §11-2 (U), §11-4 (LMTD con F) y §11-5 (ε-NTU); la exergía,
como el vademecum (§11.10 y §13.2). La notación es la de Cengel y Ghajar: h el
fluido caliente, c el frío; C = ṁ·c_p.
"""

from __future__ import annotations

import math

from core.heat_transfer.exchangers import (
    ExchangerResult,
    FourTemperatureInputs,
    OverallUResult,
    RatingInputs,
    SizingInputs,
    StreamResult,
)
from core.heat_transfer.procedure_common import frac, n, numbered, q, sub, sum_rows, times
from core.latex import latex_chain, latex_is_wide, latex_number, latex_paren
from core.state_report import ProcedureStep
from core.units_system import UnitSystem, convert_from_si, unit_label

__all__ = ["EPS_FORMULAS", "NTU_FORMULAS", "exchanger_steps", "overall_u_steps"]

_T = "temperature"
_TA = "absolute_temperature"
_DT = "temperature_difference"
_C = "heat_capacity_rate"
_Q = "heat_rate"
_H = "heat_transfer_coefficient"

#: ε(NTU, C_r) de cada tipo (Incropera, tabla 11.3; Cengel y Ghajar, tabla 11-4): los
#: renglones de cada fórmula (las que tienen una variable auxiliar, en dos).
EPS_FORMULAS: dict[str, tuple[str, ...]] = {
    "parallel": (r"\varepsilon = \frac{1 - e^{-NTU\,(1+C_r)}}{1 + C_r}",),
    "counter": (r"\varepsilon = \frac{1 - e^{-NTU\,(1-C_r)}}{1 - C_r\,e^{-NTU\,(1-C_r)}}",),
    "counter_1": (r"\varepsilon = \frac{NTU}{1 + NTU}",),
    "shell": (
        r"\varepsilon_1 = \frac{2}{1 + C_r + s\,\frac{1 + e^{-NTU\,s}}{1 - e^{-NTU\,s}}}",
        r"s = \sqrt{1 + C_r^2}",
    ),
    "shell_n": (
        r"\varepsilon = \frac{a^n - 1}{a^n - C_r}",
        r"a = \frac{1 - \varepsilon_1 C_r}{1 - \varepsilon_1}",
    ),
    "cross_unmixed": (
        r"\begin{aligned}\varepsilon &= \frac{1}{C_r\,NTU} \\ &\quad \cdot \sum_{n=0}^{\infty} "
        r"P_n(NTU)\,P_n(C_r\,NTU)\end{aligned}",
        r"P_n(x) = 1 - e^{-x} \sum_{m=0}^{n} \frac{x^m}{m!}",
    ),
    "cross_unmixed_approx": (
        r"\varepsilon \approx 1 - \exp\left(\frac{NTU^{0.22}}{C_r}\,b\right)",
        r"b = e^{-C_r\,NTU^{0.78}} - 1",
    ),
    "cross_cmax_mixed": (
        r"\varepsilon = \frac{1}{C_r}\left\{1 - \exp\left[-C_r\left(1 - "
        r"e^{-NTU}\right)\right]\right\}",
    ),
    "cross_cmin_mixed": (
        r"\varepsilon = 1 - \exp\left\{-\frac{1}{C_r}\left[1 - "
        r"e^{-C_r\,NTU}\right]\right\}",
    ),
    "phase_change": (r"\varepsilon = 1 - e^{-NTU}",),
}

#: NTU(ε, C_r): la inversa (Incropera, tabla 11.4), en renglones como ``EPS_FORMULAS``.
NTU_FORMULAS: dict[str, tuple[str, ...]] = {
    "parallel": (r"NTU = -\frac{\ln\left[1 - \varepsilon\,(1 + C_r)\right]}{1 + C_r}",),
    "counter": (r"NTU = \frac{1}{C_r - 1}\,\ln\frac{\varepsilon - 1}{\varepsilon\,C_r - 1}",),
    "counter_1": (r"NTU = \frac{\varepsilon}{1 - \varepsilon}",),
    "shell": (
        r"NTU = -\frac{1}{s}\,\ln\frac{E - 1}{E + 1}",
        r"E = \frac{2/\varepsilon - (1 + C_r)}{s}",
    ),
    "shell_n": (
        r"\varepsilon_1 = \frac{F_s - 1}{F_s - C_r}",
        r"F_s = \left(\frac{\varepsilon\,C_r - 1}{\varepsilon - 1}\right)^{1/n}",
    ),
    "cross_cmax_mixed": (r"NTU = -\ln\left[1 + \frac{\ln(1 - \varepsilon\,C_r)}{C_r}\right]",),
    "cross_cmin_mixed": (r"NTU = -\frac{\ln\left[C_r\,\ln(1 - \varepsilon) + 1\right]}{C_r}",),
    "phase_change": (r"NTU = -\ln(1 - \varepsilon)",),
}


def _eps_n(x: float) -> str:
    return latex_number(x, 4)


def _capacity_units_note(system: UnitSystem) -> str:
    """ṁ·c_p en las unidades de U·A (W/K o Btu/(h·°F))."""
    if system == "Técnico":
        return " Con c_p en kJ/(kg·K), ṁ·c_p sale en kW/K: por 1000, en W/K, las unidades de U·A."
    if system == "Inglés":
        return " ṁ·c_p sale en Btu/(s·°F): por 3600 s/h, en Btu/(h·°F), las unidades de U·A."
    return ""


def _latent_factor(system: UnitSystem) -> str:
    """El factor que pasa Q̇/h_fg a kg/s o lb/s."""
    return {"SI": "", "Técnico": r"1000 \cdot ", "Inglés": r"3600 \cdot "}[system]


def _factor_diff(factor: str, diff: str) -> str:
    r"""``C \cdot (a - b)``; si C lleva ×10ⁿ, la resta va en un renglón de continuación."""
    if latex_is_wide(factor) or r"\times" in diff:
        return rf"{factor} \\ &\quad \cdot ({diff})"
    return times(factor, f"({diff})")


def _side(sr: StreamResult, hot: bool) -> str:
    return "h" if hot else "c"


def _capacity_lines(sr: StreamResult, hot: bool, system: UnitSystem) -> list[str]:
    s = _side(sr, hot)
    if sr.phase_change:
        return [rf"C_{s} \to \infty"]
    assert sr.cp_J_per_kgK is not None
    return [
        latex_chain(
            f"C_{s}",
            rf"\dot{{m}}_{s}\,c_{{p,{s}}}",
            rf"{q(sr.m_dot_kg_s, 'mass_flow', system)} \\ &\quad \cdot "
            f"{q(sr.cp_J_per_kgK, 'specific_heat', system)}",
            q(sr.C_W_per_K, _C, system),
        )
    ]


def _cp_text(sr: StreamResult, hot: bool, system: UnitSystem) -> str:
    who = "caliente" if hot else "frío"
    if sr.phase_change:
        return f"El fluido {who} ({sr.name}) cambia de fase a temperatura constante: C → ∞."
    if sr.cp_from_coolprop:
        return (
            f"El c_p del fluido {who} ({sr.name}) sale de CoolProp a su temperatura media "
            f"(T̄ = {_plain(sr.T_mean_K, system)}), que se itera porque la salida depende de c_p."
        )
    return f"El c_p del fluido {who} ({sr.name}) es un dato."


def _plain(T_K: float, system: UnitSystem) -> str:
    return f"{convert_from_si(T_K, _T, system):.4g} {unit_label(_T, system)}".replace(".", ",")


def _capacities_step(r: ExchangerResult, system: UnitSystem) -> ProcedureStep:
    latex = _capacity_lines(r.hot, True, system) + _capacity_lines(r.cold, False, system)
    C_min, C_max = r.C_min_W_per_K, r.C_max_W_per_K
    if r.C_r > 0.0:
        latex.append(
            latex_chain(
                "C_r",
                frac(r"C_{\text{mín}}", r"C_{\text{máx}}"),
                frac(n(C_min, _C, system), n(C_max, _C, system)),
                _eps_n(r.C_r),
            )
        )
    else:
        latex.append(r"C_r = \frac{C_{\text{mín}}}{C_{\text{máx}}} = 0")
    who = "caliente" if r.cmin_side == "hot" else "frío"
    text = (
        " ".join(_cp_text(s, s is r.hot, system) for s in (r.hot, r.cold))
        + _capacity_units_note(system)
        + f" El de menor capacidad (C_mín) es el fluido {who}."
    )
    return ProcedureStep("Capacidades caloríficas", text, tuple(latex))


def _eps_key(r: ExchangerResult) -> str:
    if r.C_r <= 0.0:
        return "phase_change"
    kind = r.arrangement.kind
    if kind == "counter" and abs(1.0 - r.C_r) < 1e-9:
        return "counter_1"
    if kind == "cross_mixed":
        return "cross_cmin_mixed" if r.mixed_is_cmin else "cross_cmax_mixed"
    return kind


def _effectiveness_lines(r: ExchangerResult) -> list[str]:
    """La fórmula ε(NTU, C_r) del tipo con los números."""
    key, ntu, cr = _eps_key(r), r.NTU, r.C_r
    N, Cr, eps = _eps_n(ntu), _eps_n(cr), _eps_n(r.effectiveness)
    if key == "phase_change":
        return [*EPS_FORMULAS[key], latex_chain(r"\varepsilon", rf"1 - e^{{-{N}}}", eps)]
    if key == "parallel":
        return [
            *EPS_FORMULAS[key],
            latex_chain(
                r"\varepsilon",
                frac(rf"1 - e^{{-{N} \cdot {_eps_n(1 + cr)}}}", _eps_n(1 + cr)),
                eps,
            ),
        ]
    if key == "counter_1":
        return [*EPS_FORMULAS[key], latex_chain(r"\varepsilon", frac(N, f"1 + {N}"), eps)]
    if key == "counter":
        e = math.exp(-ntu * (1 - cr))
        return [
            *EPS_FORMULAS[key],
            latex_chain(r"e^{-NTU\,(1-C_r)}", rf"e^{{-{N} \cdot {_eps_n(1 - cr)}}}", _eps_n(e)),
            latex_chain(
                r"\varepsilon", frac(f"1 - {_eps_n(e)}", f"1 - {Cr} \\cdot {_eps_n(e)}"), eps
            ),
        ]
    if key == "shell":
        n_pass = r.arrangement.shell_passes
        ntu1 = ntu / n_pass
        s = math.sqrt(1 + cr * cr)
        e = math.exp(-ntu1 * s)
        e1 = 2.0 / (1 + cr + s * (1 + e) / (1 - e))
        lines = [*EPS_FORMULAS["shell"]]
        if n_pass > 1:
            lines.append(
                latex_chain(r"NTU_1", frac("NTU", "n"), frac(N, str(n_pass)), _eps_n(ntu1))
            )
        nt = "NTU_1" if n_pass > 1 else "NTU"
        lines += [
            latex_chain("s", rf"\sqrt{{1 + {Cr}^2}}", _eps_n(s)),
            latex_chain(rf"e^{{-{nt}\,s}}", rf"e^{{-{_eps_n(ntu1)} \cdot {_eps_n(s)}}}", _eps_n(e)),
            latex_chain(
                r"\varepsilon_1",
                frac(
                    "2",
                    rf"1 + {Cr} + {_eps_n(s)} \cdot \frac{{1 + {_eps_n(e)}}}{{1 - {_eps_n(e)}}}",
                ),
                _eps_n(e1),
            ),
        ]
        if n_pass > 1:
            a = (1 - e1 * cr) / (1 - e1)
            lines += [
                *EPS_FORMULAS["shell_n"],
                latex_chain(
                    "a",
                    frac(rf"1 - {_eps_n(e1)} \cdot {Cr}", f"1 - {_eps_n(e1)}"),
                    _eps_n(a),
                ),
                latex_chain(
                    r"\varepsilon",
                    frac(f"{_eps_n(a)}^{n_pass} - 1", f"{_eps_n(a)}^{n_pass} - {Cr}"),
                    eps,
                ),
            ]
        return lines
    if key == "cross_unmixed":
        approx = r.eps_approx or 0.0
        b = math.exp(-cr * ntu**0.78) - 1.0
        expo = ntu**0.22 / cr * b
        return [
            *EPS_FORMULAS["cross_unmixed"],
            latex_chain(
                r"\varepsilon", r"\varepsilon(NTU,\ C_r)", rf"\varepsilon({N},\ {Cr})", eps
            ),
            *EPS_FORMULAS["cross_unmixed_approx"],
            latex_chain("b", rf"e^{{-{Cr} \cdot {N}^{{0.78}}}} - 1", _eps_n(b)),
            latex_chain(
                r"\frac{NTU^{0.22}}{C_r}\,b",
                rf"\frac{{{N}^{{0.22}}}}{{{Cr}}} \cdot {latex_paren(_eps_n(b))}",
                _eps_n(expo),
            ),
            latex_chain(
                r"\varepsilon_{\text{aprox}}", rf"1 - e^{{{_eps_n(expo)}}}", _eps_n(approx)
            ),
        ]
    if key == "cross_cmax_mixed":
        inner = 1 - math.exp(-ntu)
        return [
            *EPS_FORMULAS[key],
            latex_chain(r"1 - e^{-NTU}", rf"1 - e^{{-{N}}}", _eps_n(inner)),
            latex_chain(
                r"\varepsilon",
                rf"\frac{{1 - e^{{-{Cr} \cdot {_eps_n(inner)}}}}}{{{Cr}}}",
                eps,
            ),
        ]
    inner = 1 - math.exp(-cr * ntu)
    return [
        *EPS_FORMULAS[key],
        latex_chain(r"1 - e^{-C_r\,NTU}", rf"1 - e^{{-{Cr} \cdot {N}}}", _eps_n(inner)),
        latex_chain(r"\varepsilon", rf"1 - e^{{-{_eps_n(inner)}/{Cr}}}", eps),
    ]


def _ntu_inverse_lines(r: ExchangerResult) -> list[str]:
    """NTU(ε, C_r) del tipo con los números (la inversa)."""
    key, eps, cr = _eps_key(r), r.effectiveness, r.C_r
    E_, Cr, N = _eps_n(eps), _eps_n(cr), _eps_n(r.NTU)
    if key == "phase_change":
        return [*NTU_FORMULAS[key], latex_chain("NTU", rf"-\ln(1 - {E_})", N)]
    if key == "parallel":
        return [
            *NTU_FORMULAS[key],
            latex_chain("NTU", frac(rf"-\ln(1 - {E_} \cdot {_eps_n(1 + cr)})", _eps_n(1 + cr)), N),
        ]
    if key == "counter_1":
        return [*NTU_FORMULAS[key], latex_chain("NTU", frac(E_, f"1 - {E_}"), N)]
    if key == "counter":
        return [
            *NTU_FORMULAS[key],
            latex_chain(
                "NTU",
                rf"\frac{{1}}{{{Cr} - 1}} \\ &\quad \cdot "
                rf"\ln\frac{{{E_} - 1}}{{{E_} \cdot {Cr} - 1}}",
                N,
            ),
        ]
    if key == "shell":
        n_pass = r.arrangement.shell_passes
        lines: list[str] = []
        e1 = eps
        if n_pass > 1:
            Fs = ((eps * cr - 1) / (eps - 1)) ** (1.0 / n_pass)
            e1 = (Fs - 1) / (Fs - cr)
            lines += [
                *NTU_FORMULAS["shell_n"],
                latex_chain(
                    "F_s",
                    rf"\left(\frac{{{E_} \cdot {Cr} - 1}}{{{E_} - 1}}\right)^{{1/{n_pass}}}",
                    _eps_n(Fs),
                ),
                latex_chain(
                    r"\varepsilon_1", frac(f"{_eps_n(Fs)} - 1", f"{_eps_n(Fs)} - {Cr}"), _eps_n(e1)
                ),
            ]
        s = math.sqrt(1 + cr * cr)
        E = (2 / e1 - (1 + cr)) / s
        ntu1 = r.NTU / n_pass
        lines += [
            *NTU_FORMULAS["shell"],
            latex_chain("s", rf"\sqrt{{1 + {Cr}^2}}", _eps_n(s)),
            latex_chain("E", frac(rf"2/{_eps_n(e1)} - {_eps_n(1 + cr)}", _eps_n(s)), _eps_n(E)),
            latex_chain(
                "NTU_1" if n_pass > 1 else "NTU",
                rf"-\frac{{1}}{{{_eps_n(s)}}}\,\ln\frac{{{_eps_n(E)} - 1}}{{{_eps_n(E)} + 1}}",
                _eps_n(ntu1),
            ),
        ]
        if n_pass > 1:
            lines.append(latex_chain("NTU", rf"n\,NTU_1 = {n_pass} \cdot {_eps_n(ntu1)}", N))
        return lines
    if key == "cross_cmax_mixed":
        L = _eps_n(math.log(1.0 - eps * cr))
        return [
            *NTU_FORMULAS[key],
            latex_chain(r"\ln(1 - \varepsilon\,C_r)", rf"\ln(1 - {E_} \cdot {Cr})", L),
            latex_chain("NTU", rf"-\ln\left[1 + \frac{{{L}}}{{{Cr}}}\right]", N),
        ]
    if key == "cross_cmin_mixed":
        L = _eps_n(math.log(1.0 - eps))
        return [
            *NTU_FORMULAS[key],
            latex_chain(r"\ln(1 - \varepsilon)", rf"\ln(1 - {E_})", L),
            latex_chain("NTU", frac(rf"-\ln\left[{Cr} \cdot {latex_paren(L)} + 1\right]", Cr), N),
        ]
    return [
        *EPS_FORMULAS["cross_unmixed"],
        rf"\begin{{aligned}}\varepsilon(NTU,\ {Cr}) &= {E_} \\ "
        rf"\Rightarrow NTU &= {N}\end{{aligned}}",
    ]


def _heat_lines(r: ExchangerResult, system: UnitSystem) -> list[str]:
    """Q̇_máx, Q̇ y las salidas (verificación)."""
    Th, Tc = n(r.hot.T_in_K, _T, system), n(r.cold.T_in_K, _T, system)
    Cmin = n(r.C_min_W_per_K, _C, system)
    lines = [
        latex_chain(
            r"\dot{Q}_{\text{máx}}",
            r"C_{\text{mín}}\,(T_{h,\text{ent}} - T_{c,\text{ent}})",
            _factor_diff(Cmin, sub(Th, Tc)),
            q(r.Q_max_W, _Q, system),
        ),
        latex_chain(
            r"\dot{Q}",
            r"\varepsilon\,\dot{Q}_{\text{máx}}",
            times(_eps_n(r.effectiveness), n(r.Q_max_W, _Q, system)),
            q(r.Q_W, _Q, system),
        ),
    ]
    return lines + _outlet_lines(r, system)


def _outlet_line(r: ExchangerResult, sr: StreamResult, system: UnitSystem) -> str:
    """La salida de una corriente (o su caudal, si cambia de fase)."""
    hot = sr is r.hot
    s = _side(sr, hot)
    Qn = n(r.Q_W, _Q, system)
    if sr.phase_change:
        assert sr.h_fg_J_per_kg is not None
        hfg = n(sr.h_fg_J_per_kg, "specific_enthalpy", system)
        return latex_chain(
            rf"\dot{{m}}_{s}",
            frac(r"\dot{Q}", "h_{fg}"),
            frac(Qn, _latent_factor(system) + hfg),
            q(sr.m_dot_kg_s, "mass_flow", system),
        )
    sign = "-" if hot else "+"
    return latex_chain(
        rf"T_{{{s},\text{{sal}}}}",
        rf"T_{{{s},\text{{ent}}}} {sign} \frac{{\dot{{Q}}}}{{C_{s}}}",
        rf"{n(sr.T_in_K, _T, system)} {sign} {frac(Qn, n(sr.C_W_per_K, _C, system))}",
        q(sr.T_out_K, _T, system),
    )


def _outlet_lines(r: ExchangerResult, system: UnitSystem) -> list[str]:
    return [_outlet_line(r, r.hot, system), _outlet_line(r, r.cold, system)]


def _lmtd_lines(r: ExchangerResult, system: UnitSystem) -> list[str]:
    dT1, dT2 = r.end_differences_K
    hi, ho = n(r.hot.T_in_K, _T, system), n(r.hot.T_out_K, _T, system)
    ci, co = n(r.cold.T_in_K, _T, system), n(r.cold.T_out_K, _T, system)
    if r.arrangement.kind == "parallel":
        d1 = (r"T_{h,\text{ent}} - T_{c,\text{ent}}", sub(hi, ci))
        d2 = (r"T_{h,\text{sal}} - T_{c,\text{sal}}", sub(ho, co))
    else:
        d1 = (r"T_{h,\text{ent}} - T_{c,\text{sal}}", sub(hi, co))
        d2 = (r"T_{h,\text{sal}} - T_{c,\text{ent}}", sub(ho, ci))
    a, b = n(dT1, _DT, system), n(dT2, _DT, system)
    lines = [
        latex_chain(r"\Delta T_1", d1[0], d1[1], q(dT1, _DT, system)),
        latex_chain(r"\Delta T_2", d2[0], d2[1], q(dT2, _DT, system)),
    ]
    if math.isclose(dT1, dT2, rel_tol=1e-9):
        lines.append(rf"\Delta T_{{ml}} = \Delta T_1 = \Delta T_2 = {q(r.dT_lm_K, _DT, system)}")
    else:
        lines.append(
            latex_chain(
                r"\Delta T_{ml}",
                frac(r"\Delta T_1 - \Delta T_2", r"\ln(\Delta T_1/\Delta T_2)"),
                frac(sub(a, b), rf"\ln({a}/{b})"),
                q(r.dT_lm_K, _DT, system),
            )
        )
    return lines


def _lmtd_text(r: ExchangerResult) -> str:
    """Qué extremos se restan para la ΔT_ml."""
    if r.arrangement.kind == "parallel":
        return (
            "En flujo paralelo las entradas están juntas y las salidas también: ΔT₁ es la "
            "diferencia a la entrada y ΔT₂ a la salida."
        )
    text = (
        "La de un contracorriente con las mismas cuatro temperaturas: cada extremo junta la "
        "entrada de un fluido con la salida del otro."
    )
    if r.arrangement.kind != "counter":
        text += " En este tipo, F la corrige (Q̇ = U·A·F·ΔT_ml)."
    return text


def _f_lines(r: ExchangerResult) -> tuple[list[str], str]:
    """El factor F = NTU_cc/NTU (y P y R de Cengel y Ghajar)."""
    if r.arrangement.kind in ("counter", "parallel") or r.C_r <= 0.0:
        why = (
            "En contracorriente F = 1."
            if r.arrangement.kind == "counter"
            else "En flujo paralelo se usa la ΔT_ml del paralelo (F = 1)."
            if r.arrangement.kind == "parallel"
            else "Con un fluido que cambia de fase, F = 1 en cualquier tipo."
        )
        return [r"F = 1"], why
    lines: list[str] = []
    PR = r.P_R
    if PR is not None:
        P, R = PR
        tubes = "caliente" if r.arrangement.tubes == "hot" else "frío"
        lines += [
            rf"P = \frac{{t_2 - t_1}}{{T_1 - t_1}} = {_eps_n(P)}",
            rf"R = \frac{{T_1 - T_2}}{{t_2 - t_1}} = {_eps_n(R)}",
        ]
        pr_text = (
            f"P y R como Cengel y Ghajar (t: el fluido de los tubos, acá el {tubes}): con ellos se "
            "lee F en los gráficos de Bowman, Mueller y Nagle (1940)."
        )
    else:
        pr_text = ""
    eps, cr = r.effectiveness, r.C_r
    ntu_cc = _ntu_counter(eps, cr)
    lines += [
        latex_chain(
            r"NTU_{\text{cc}}",
            rf"\frac{{1}}{{{_eps_n(cr)} - 1}} \\ &\quad \cdot "
            rf"\ln\frac{{{_eps_n(eps)} - 1}}{{{_eps_n(eps)} \cdot {_eps_n(cr)} - 1}}",
            _eps_n(ntu_cc),
        ),
        latex_chain(
            "F", frac(r"NTU_{\text{cc}}", "NTU"), frac(_eps_n(ntu_cc), _eps_n(r.NTU)), _eps_n(r.F)
        ),
    ]
    text = (
        f"{pr_text} F es el cociente de los NTU de un contracorriente y de este arreglo para el "
        "mismo ε y C_r: lo que achica a ΔT_ml (Q̇ = U·A·F·ΔT_ml)."
    ).strip()
    return lines, text


def _ntu_counter(eps: float, cr: float) -> float:
    """NTU de un contracorriente para ε y C_r."""
    if abs(1.0 - cr) < 1e-9:
        return eps / (1.0 - eps)
    return math.log((eps - 1.0) / (eps * cr - 1.0)) / (cr - 1.0)


def _exergy_lines(r: ExchangerResult, system: UnitSystem) -> tuple[list[str], str]:
    ex = r.exergy
    T0 = n(ex.T0_K, _TA, system)
    lines: list[str] = []
    for sr, hot, dS in ((r.hot, True, ex.dS_hot_W_per_K), (r.cold, False, ex.dS_cold_W_per_K)):
        s = _side(sr, hot)
        if sr.phase_change:
            sign = "-" if hot else ""
            lines.append(
                latex_chain(
                    rf"\Delta\dot{{S}}_{s}",
                    rf"{sign}\frac{{\dot{{Q}}}}{{T_{{\text{{sat}}}}}}",
                    sign + frac(n(r.Q_W, _Q, system), n(sr.T_in_K, _TA, system)),
                    q(dS, "entropy_rate", system),
                )
            )
            continue
        lines.append(
            latex_chain(
                rf"\Delta\dot{{S}}_{s}",
                rf"C_{s}\,\ln\frac{{T_{{{s},\text{{sal}}}}}}{{T_{{{s},\text{{ent}}}}}}",
                times(
                    n(sr.C_W_per_K, _C, system),
                    rf"\ln\frac{{{n(sr.T_out_K, _TA, system)}}}{{{n(sr.T_in_K, _TA, system)}}}",
                ),
                q(dS, "entropy_rate", system),
            )
        )
    lines += [
        latex_chain(
            r"\dot{S}_{\text{gen}}",
            r"\Delta\dot{S}_h + \Delta\dot{S}_c",
            f"{n(ex.dS_hot_W_per_K, 'entropy_rate', system)} + "
            f"{latex_paren(n(ex.dS_cold_W_per_K, 'entropy_rate', system))}",
            q(ex.S_gen_W_per_K, "entropy_rate", system),
        ),
        latex_chain(
            r"\dot{X}_{\text{dest}}",
            r"T_0\,\dot{S}_{\text{gen}}",
            times(T0, n(ex.S_gen_W_per_K, "entropy_rate", system)),
            q(ex.X_dest_W, _Q, system),
        ),
    ]
    eta = ex.efficiency
    if eta is not None:
        lines.append(
            latex_chain(
                r"\eta_{\text{ex}}",
                frac(r"\Delta\dot{\Psi}_c", r"-\Delta\dot{\Psi}_h"),
                frac(n(ex.dPsi_cold_W, _Q, system), n(-ex.dPsi_hot_W, _Q, system)),
                _eps_n(eta),
            )
        )
    text = (
        "Con c_p constante (vademecum §13.2), ΔṠ = C·ln(T_sal/T_ent) con temperaturas absolutas; "
        "el que cambia de fase, ±Q̇/T_sat. Ẋ_dest = T₀·Ṡ_gen, y el rendimiento exergético del "
        "intercambiador adiabático (vademecum §11.10) es lo que gana el frío sobre lo que pierde "
        "el caliente, con ΔΨ̇ = C·[(T_sal − T_ent) − T₀·ln(T_sal/T_ent)]."
    )
    if eta is None:
        text += " Acá η_ex no se define: el frío no gana exergía (está por debajo de T₀)."
    return lines, text


def _ua_lines(r: ExchangerResult, system: UnitSystem) -> list[str]:
    return [
        latex_chain(
            "UA",
            times(n(r.U_W_per_m2K, _H, system), n(r.A_m2, "area", system)),
            q(r.UA_W_per_K, _C, system),
        ),
        latex_chain(
            "NTU",
            frac("UA", r"C_{\text{mín}}"),
            frac(n(r.UA_W_per_K, _C, system), n(r.C_min_W_per_K, _C, system)),
            _eps_n(r.NTU),
        ),
    ]


def _rating_steps(r: ExchangerResult, system: UnitSystem) -> list[ProcedureStep]:
    f_lines, f_text = _f_lines(r)
    return [
        _capacities_step(r, system),
        ProcedureStep(
            "Número de unidades de transferencia",
            "NTU = U·A/C_mín mide el tamaño del intercambiador (Cengel y Ghajar, §11-5).",
            tuple(_ua_lines(r, system)),
        ),
        ProcedureStep(
            "Efectividad",
            f"La relación ε(NTU, C_r) de un intercambiador «{r.arrangement.name.lower()}» "
            "(Incropera, tabla 11.3; Cengel y Ghajar, tabla 11-4)." + _cross_text(r),
            tuple(_effectiveness_lines(r)),
        ),
        ProcedureStep(
            "Calor y temperaturas de salida",
            "Q̇ = ε·Q̇_máx, y cada salida del balance de su corriente.",
            tuple(_heat_lines(r, system)),
        ),
        ProcedureStep(
            "Lo mismo con la LMTD",
            f_text + " Con F y la ΔT_ml, Q̇ = U·A·F·ΔT_ml da el mismo calor.",
            tuple(
                _lmtd_lines(r, system)
                + f_lines
                + [
                    latex_chain(
                        r"\dot{Q}",
                        r"U\,A\,F\,\Delta T_{ml}",
                        times(
                            n(r.UA_W_per_K, _C, system),
                            _eps_n(r.F),
                            n(r.dT_lm_K, _DT, system),
                        ),
                        q(r.Q_W, _Q, system),
                    )
                ]
            ),
        ),
    ]


def _cross_text(r: ExchangerResult) -> str:
    if _eps_key(r) == "cross_unmixed":
        return (
            " Con los dos fluidos sin mezclar se usa la serie exacta (Mason, 1955); la fórmula "
            "aproximada de las tablas se aparta hasta ~4 %."
        )
    if r.arrangement.kind == "cross_mixed" and r.C_r > 0.0:
        which = "C_mín" if r.mixed_is_cmin else "C_máx"
        return f" El fluido mezclado es el de {which}."
    return ""


def _sizing_target_lines(r: ExchangerResult, system: UnitSystem) -> list[str]:
    i = r.inputs
    assert isinstance(i, SizingInputs)
    if i.target == "Q":
        return [rf"\dot{{Q}} = {q(r.Q_W, _Q, system)}", *_outlet_lines(r, system)]
    target_hot = i.target == "T_hot_out"
    sr, other = (r.hot, r.cold) if target_hot else (r.cold, r.hot)
    T_in, T_out = n(sr.T_in_K, _T, system), n(sr.T_out_K, _T, system)
    diff = sub(T_in, T_out) if target_hot else sub(T_out, T_in)
    symb = (
        r"C_h\,(T_{h,\text{ent}} - T_{h,\text{sal}})"
        if target_hot
        else r"C_c\,(T_{c,\text{sal}} - T_{c,\text{ent}})"
    )
    C_n = n(sr.C_W_per_K, _C, system)
    return [
        latex_chain(r"\dot{Q}", symb, _factor_diff(C_n, diff), q(r.Q_W, _Q, system)),
        _outlet_line(r, other, system),
    ]


def _sizing_steps(r: ExchangerResult, system: UnitSystem) -> list[ProcedureStep]:
    f_lines, f_text = _f_lines(r)
    Th, Tc = n(r.hot.T_in_K, _T, system), n(r.cold.T_in_K, _T, system)
    eps_lines = [
        latex_chain(
            r"\dot{Q}_{\text{máx}}",
            r"C_{\text{mín}}\,(T_{h,\text{ent}} - T_{c,\text{ent}})",
            _factor_diff(n(r.C_min_W_per_K, _C, system), sub(Th, Tc)),
            q(r.Q_max_W, _Q, system),
        ),
        latex_chain(
            r"\varepsilon",
            frac(r"\dot{Q}", r"\dot{Q}_{\text{máx}}"),
            frac(n(r.Q_W, _Q, system), n(r.Q_max_W, _Q, system)),
            _eps_n(r.effectiveness),
        ),
    ]
    area_lines = [
        latex_chain(
            "UA",
            r"NTU\,C_{\text{mín}}",
            times(_eps_n(r.NTU), n(r.C_min_W_per_K, _C, system)),
            q(r.UA_W_per_K, _C, system),
        ),
        latex_chain(
            "A",
            frac("UA", "U"),
            frac(n(r.UA_W_per_K, _C, system), n(r.U_W_per_m2K, _H, system)),
            q(r.A_m2, "area", system),
        ),
    ]
    L = r.tube_length_m
    i = r.inputs
    if L is not None and isinstance(i, SizingInputs) and i.tube_D_m:
        area_lines.append(
            latex_chain(
                "L",
                frac("A", r"\pi\,D"),
                frac(n(r.A_m2, "area", system), rf"\pi \cdot {n(i.tube_D_m, 'length', system)}"),
                q(L, "length", system),
            )
        )
    return [
        ProcedureStep(
            "Calor pedido",
            "El objetivo fija Q̇ y la otra salida sale del balance.",
            tuple(_sizing_target_lines(r, system)),
        ),
        _capacities_step(r, system),
        ProcedureStep(
            "Efectividad pedida",
            "ε = Q̇/Q̇_máx tiene que quedar por debajo del máximo de ese tipo de intercambiador.",
            tuple(eps_lines),
        ),
        ProcedureStep(
            "NTU necesario y área",
            "La inversa NTU(ε, C_r) (Incropera, tabla 11.4) da U·A, y con U el área."
            + _cross_text(r),
            tuple(_ntu_inverse_lines(r) + area_lines),
        ),
        ProcedureStep(
            "Lo mismo con la LMTD",
            f_text
            + " Con F y la ΔT_ml, A = Q̇/(U·F·ΔT_ml) da la misma área (Cengel y Ghajar, §11-4).",
            tuple(
                _lmtd_lines(r, system)
                + f_lines
                + [
                    latex_chain(
                        "A",
                        frac(r"\dot{Q}", r"U\,F\,\Delta T_{ml}"),
                        frac(
                            n(r.Q_W, _Q, system),
                            times(
                                n(r.U_W_per_m2K, _H, system), _eps_n(r.F), n(r.dT_lm_K, _DT, system)
                            ),
                        ),
                        q(r.A_m2, "area", system),
                    )
                ]
            ),
        ),
    ]


def _test_steps(r: ExchangerResult, system: UnitSystem) -> list[ProcedureStep]:
    i = r.inputs
    assert isinstance(i, FourTemperatureInputs)
    dT_h, dT_c = r.hot.dT_K, r.cold.dT_K
    hi, ho = n(r.hot.T_in_K, _T, system), n(r.hot.T_out_K, _T, system)
    ci, co = n(r.cold.T_in_K, _T, system), n(r.cold.T_out_K, _T, system)
    lines = []
    if not r.hot.phase_change:
        lines.append(latex_chain(r"\Delta T_h", sub(hi, ho), q(dT_h, _DT, system)))
    if not r.cold.phase_change:
        lines.append(latex_chain(r"\Delta T_c", sub(co, ci), q(dT_c, _DT, system)))
    if r.C_r > 0.0:
        lines.append(
            latex_chain(
                frac("C_c", "C_h"),
                frac(r"\Delta T_h", r"\Delta T_c"),
                frac(n(dT_h, _DT, system), n(dT_c, _DT, system)),
                _eps_n(dT_h / dT_c),
            )
        )
        lines.append(rf"C_r = {_eps_n(r.C_r)}")
    big = r.hot if (r.cmin_side == "hot") else r.cold
    lines.append(
        latex_chain(
            r"\varepsilon",
            frac(r"\Delta T_{C_{\text{mín}}}", r"T_{h,\text{ent}} - T_{c,\text{ent}}"),
            frac(n(big.dT_K, _DT, system), sub(hi, ci)),
            _eps_n(r.effectiveness),
        )
    )
    who = "caliente" if r.cmin_side == "hot" else "frío"
    first = ProcedureStep(
        "Lo que dicen las temperaturas",
        "Como C_h·ΔT_h = C_c·ΔT_c, el fluido que más cambia de temperatura es el de menor "
        f"capacidad (acá el {who}); su ΔT sobre la máxima posible es ε.",
        tuple(lines),
    )
    f_lines, f_text = _f_lines(r)
    ntu_step = ProcedureStep(
        "NTU y F",
        "La inversa NTU(ε, C_r) del tipo (Incropera, tabla 11.4). " + f_text,
        tuple(_ntu_inverse_lines(r) + f_lines),
    )
    lmtd_step = ProcedureStep(
        "La ΔT media logarítmica", _lmtd_text(r), tuple(_lmtd_lines(r, system))
    )
    if i.U_W_per_m2K is not None:
        heat = [
            latex_chain(
                "UA",
                times(n(r.U_W_per_m2K, _H, system), n(r.A_m2, "area", system)),
                q(r.UA_W_per_K, _C, system),
            ),
            latex_chain(
                r"\dot{Q}",
                r"U\,A\,F\,\Delta T_{ml}",
                times(n(r.UA_W_per_K, _C, system), _eps_n(r.F), n(r.dT_lm_K, _DT, system)),
                q(r.Q_W, _Q, system),
            ),
        ]
        heat_text = "Con U y A, Q̇ = U·A·F·ΔT_ml; los caudales salen de Q̇ y de cada ΔT."
    else:
        side = r.hot if i.known_flow == "hot" else r.cold
        s = _side(side, side is r.hot)
        if side.phase_change:
            assert side.h_fg_J_per_kg is not None
            heat = [
                latex_chain(
                    r"\dot{Q}",
                    rf"\dot{{m}}_{s}\,h_{{fg}}",
                    _latent_factor(system)
                    + times(
                        n(side.m_dot_kg_s, "mass_flow", system),
                        n(side.h_fg_J_per_kg, "specific_enthalpy", system),
                    ),
                    q(r.Q_W, _Q, system),
                )
            ]
        else:
            heat = _capacity_lines(side, side is r.hot, system) + [
                latex_chain(
                    r"\dot{Q}",
                    rf"C_{s}\,\Delta T_{s}",
                    times(n(side.C_W_per_K, _C, system), n(side.dT_K, _DT, system)),
                    q(r.Q_W, _Q, system),
                )
            ]
        heat += [
            latex_chain(
                "UA",
                frac(r"\dot{Q}", r"F\,\Delta T_{ml}"),
                frac(n(r.Q_W, _Q, system), times(_eps_n(r.F), n(r.dT_lm_K, _DT, system))),
                q(r.UA_W_per_K, _C, system),
            ),
            latex_chain(
                "U",
                frac("UA", "A"),
                frac(n(r.UA_W_per_K, _C, system), n(r.A_m2, "area", system)),
                q(r.U_W_per_m2K, _H, system),
            ),
        ]
        heat_text = "El caudal conocido da Q̇, y U·A = Q̇/(F·ΔT_ml)." + _capacity_units_note(system)
    flows: list[str] = []
    for sr, hot in ((r.hot, True), (r.cold, False)):
        if i.U_W_per_m2K is None and (sr is (r.hot if i.known_flow == "hot" else r.cold)):
            continue
        s = _side(sr, hot)
        if sr.phase_change:
            assert sr.h_fg_J_per_kg is not None
            flows.append(
                latex_chain(
                    rf"\dot{{m}}_{s}",
                    frac(r"\dot{Q}", "h_{fg}"),
                    frac(
                        n(r.Q_W, _Q, system),
                        _latent_factor(system) + n(sr.h_fg_J_per_kg, "specific_enthalpy", system),
                    ),
                    q(sr.m_dot_kg_s, "mass_flow", system),
                )
            )
        elif sr.cp_J_per_kgK:
            flows.append(
                latex_chain(
                    rf"\dot{{m}}_{s}",
                    frac(r"\dot{Q}", rf"c_{{p,{s}}}\,\Delta T_{s}"),
                    frac(
                        n(r.Q_W, _Q, system),
                        _latent_factor(system)
                        + times(
                            n(sr.cp_J_per_kgK, "specific_heat", system), n(sr.dT_K, _DT, system)
                        ),
                    ),
                    q(sr.m_dot_kg_s, "mass_flow", system),
                )
            )
    return [
        first,
        ntu_step,
        lmtd_step,
        ProcedureStep("Calor y U", heat_text, tuple(heat)),
        ProcedureStep(
            "Caudales",
            "Cada caudal de su balance: ṁ = Q̇/(c_p·ΔT), o Q̇/h_fg si cambia de fase.",
            tuple(flows),
        ),
    ]


def exchanger_steps(result: ExchangerResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos del problema resuelto (verificación, dimensionamiento o ensayo)."""
    if isinstance(result.inputs, RatingInputs):
        steps = _rating_steps(result, system)
    elif isinstance(result.inputs, SizingInputs):
        steps = _sizing_steps(result, system)
    else:
        steps = _test_steps(result, system)
    ex_lines, ex_text = _exergy_lines(result, system)
    steps.append(ProcedureStep("Exergía", ex_text, tuple(ex_lines)))
    return numbered([s for s in steps if s.latex or s.text])


# ---------------------------------------------------------------------
# U global
# ---------------------------------------------------------------------


def overall_u_steps(result: OverallUResult, system: UnitSystem) -> list[ProcedureStep]:
    """El coeficiente global U con ensuciamiento (Cengel y Ghajar, §11-2)."""
    i = result.inputs
    L = "length"
    steps: list[ProcedureStep] = []
    if i.geometry == "tube":
        steps.append(
            ProcedureStep(
                "Áreas",
                "Las áreas de transferencia de adentro y de afuera del tubo.",
                (
                    latex_chain(
                        "A_i",
                        r"\pi\,D_i\,L",
                        rf"\pi \cdot {times(n(i.D_i_m, L, system), n(i.L_m, L, system))}",
                        q(result.A_i_m2, "area", system),
                    ),
                    latex_chain(
                        "A_o",
                        r"\pi\,D_o\,L",
                        rf"\pi \cdot {times(n(i.D_o_m, L, system), n(i.L_m, L, system))}",
                        q(result.A_o_m2, "area", system),
                    ),
                ),
            )
        )
    else:
        steps.append(
            ProcedureStep(
                "Área",
                "Una pared plana: la misma área de los dos lados.",
                (rf"A = {q(i.A_m2, 'area', system)}",),
            )
        )
    R = "thermal_resistance"
    Ai = n(result.A_i_m2, "area", system)
    Ao = n(result.A_o_m2, "area", system)
    lines = [
        latex_chain(
            r"R_{\text{conv},i}",
            frac("1", r"h_i\,A_i"),
            frac("1", times(n(i.h_i_W_per_m2K, _H, system), Ai)),
            q(1.0 / (i.h_i_W_per_m2K * result.A_i_m2), R, system),
        )
    ]
    if i.R_f_i_m2K_per_W > 0.0:
        lines.append(
            latex_chain(
                r"R_{f,i}",
                frac("R''_{f,i}", "A_i"),
                frac(n(i.R_f_i_m2K_per_W, "fouling_resistance", system), Ai),
                q(i.R_f_i_m2K_per_W / result.A_i_m2, R, system),
            )
        )
    wall = next(x for x in result.resistances if x.kind == "pared")
    k_n = n(i.k_wall_W_per_mK, "thermal_conductivity", system)
    if i.geometry == "tube":
        lines.append(
            latex_chain(
                r"R_{\text{pared}}",
                frac(r"\ln(D_o/D_i)", r"2\pi\,k\,L"),
                frac(
                    rf"\ln({n(i.D_o_m, L, system)}/{n(i.D_i_m, L, system)})",
                    rf"2\pi \cdot {times(k_n, n(i.L_m, L, system))}",
                ),
                q(wall.R_K_per_W, R, system),
            )
        )
    else:
        lines.append(
            latex_chain(
                r"R_{\text{pared}}",
                frac("t", r"k\,A"),
                frac(
                    n(i.thickness_m, L, system),
                    times(n(i.k_wall_W_per_mK, "thermal_conductivity", system), Ai),
                ),
                q(wall.R_K_per_W, R, system),
            )
        )
    if i.R_f_o_m2K_per_W > 0.0:
        lines.append(
            latex_chain(
                r"R_{f,o}",
                frac("R''_{f,o}", "A_o"),
                frac(n(i.R_f_o_m2K_per_W, "fouling_resistance", system), Ao),
                q(i.R_f_o_m2K_per_W / result.A_o_m2, R, system),
            )
        )
    lines.append(
        latex_chain(
            r"R_{\text{conv},o}",
            frac("1", r"h_o\,A_o"),
            frac("1", times(n(i.h_o_W_per_m2K, _H, system), Ao)),
            q(1.0 / (i.h_o_W_per_m2K * result.A_o_m2), R, system),
        )
    )
    steps.append(
        ProcedureStep(
            "Resistencias en serie",
            "Convección de cada lado, el ensuciamiento (R''_f por unidad de área) y la pared.",
            tuple(lines),
        )
    )
    terms = [n(x.R_K_per_W, R, system) for x in result.resistances]
    total = [
        latex_chain(
            r"R_{\text{total}}",
            sum_rows(terms, per_row=1 if any(r"\times" in t for t in terms) else 2),
            q(result.R_total_K_per_W, R, system),
        ),
        latex_chain(
            "UA",
            frac("1", r"R_{\text{total}}"),
            frac("1", n(result.R_total_K_per_W, R, system)),
            q(result.UA_W_per_K, _C, system),
        ),
        latex_chain(
            "U_i" if i.geometry == "tube" else "U",
            frac("UA", "A_i" if i.geometry == "tube" else "A"),
            frac(n(result.UA_W_per_K, _C, system), Ai),
            q(result.U_i_W_per_m2K, _H, system),
        ),
    ]
    if i.geometry == "tube":
        total.append(
            latex_chain(
                "U_o",
                frac("UA", "A_o"),
                frac(n(result.UA_W_per_K, _C, system), Ao),
                q(result.U_o_W_per_m2K, _H, system),
            )
        )
    steps.append(
        ProcedureStep(
            "U global",
            (
                "1/(U·A) = ΣR. U depende del área a la que se refiere: U_i·A_i = U_o·A_o."
                if i.geometry == "tube"
                else "1/(U·A) = ΣR; en una pared plana U es uno solo."
            ),
            tuple(total),
        )
    )
    return numbered(steps)
