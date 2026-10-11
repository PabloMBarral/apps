"""Procedimiento didáctico del segundo principio (LaTeX), en el sistema de unidades activo.

Las fórmulas son las del vademecum: §9.1 (Clausius), §9.2 (rendimiento y Carnot), §9.3
(COP), §10.2 (S_gen del universo), §11.5 (exergía de un calor), §11.8 (trabajo perdido) y
§11.10 (rendimiento exergético). Las temperaturas van absolutas (K o °R): en el Técnico y
el Inglés el primer paso suma 273,15 o 459,67. Los calores y el trabajo van por ciclo o
por unidad de tiempo y la entropía, en la unidad que corresponde (las sustituciones cierran
sin factores, ``core.units_system``).
"""

from __future__ import annotations

from core.balances.balances_procedure import _chain, _model_text, _r, _stack
from core.balances.second_law import (
    CARNOT_PROPS,
    PERFORMANCE_NAMES,
    CarnotResult,
    ClausiusResult,
    HeatPathResult,
    MachineResult,
)
from core.heat_transfer.procedure_common import frac, n, numbered, q, times
from core.latex import latex_chain, latex_paren
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem

__all__ = ["carnot_steps", "clausius_steps", "heat_path_steps", "machine_steps"]

_T = "temperature"
_TA = "absolute_temperature"
_E = "specific_enthalpy"
_S = "specific_entropy"
_P = "pressure"
_V = "specific_volume"
_M = "mass"

#: El LaTeX de cada rendimiento.
_PERF_TEX = {
    "engine": r"\eta",
    "refrigerator": r"\mathrm{COP}_R",
    "heat_pump": r"\mathrm{COP}_B",
}
_REV_TEX = {
    "engine": r"\eta_{\mathrm{Carnot}}",
    "refrigerator": r"\mathrm{COP}_{R,\mathrm{rev}}",
    "heat_pump": r"\mathrm{COP}_{B,\mathrm{rev}}",
}


def _kinds(rates: bool, heat: bool = False) -> tuple[QuantityKind, QuantityKind]:
    """(calor, entropía): por ciclo o por unidad de tiempo (``heat``: en W, como la
    transferencia de calor)."""
    if not rates:
        return "energy", "entropy"
    return ("heat_rate", "entropy_rate") if heat else ("power", "entropy_flow")


def _abs_line(symbol: str, T_K: float, system: UnitSystem) -> str:
    """``T = t + 273,15 = … K`` (Técnico e Inglés) o ``T = … K`` (SI)."""
    if system == "SI":
        return latex_chain(symbol, q(T_K, _TA, system))
    offset = "273.15" if system == "Técnico" else "459.67"
    return latex_chain(symbol, rf"{n(T_K, _T, system)} + {offset}", q(T_K, _TA, system))


def _abs_text(system: UnitSystem) -> str:
    if system == "SI":
        return "Las fórmulas del segundo principio llevan la temperatura absoluta."
    unit = "273,15 a los °C" if system == "Técnico" else "459,67 a los °F"
    return f"Las fórmulas del segundo principio llevan la temperatura absoluta: se suma {unit}."


def _verdict_text(verdict: str, violation: str | None, what: str) -> str:
    if violation:
        return f"**Imposible.** {violation}"
    if verdict == "reversible":
        return f"**Reversible**: {what} = 0."
    return f"**Irreversible**: {what} > 0 (posible, con exergía destruida)."


# ---------------------------------------------------------------------
# Máquina
# ---------------------------------------------------------------------


def machine_steps(r: MachineResult, system: UnitSystem) -> list[ProcedureStep]:
    """Primer principio, rendimiento contra Carnot, Clausius, S_gen y exergía."""
    inp = r.inputs
    kind = inp.kind
    E, S = _kinds(inp.rates)
    dot = r"\dot" if inp.rates else ""

    def sym(name: str) -> str:
        return {"Q_H": rf"{dot}{{Q}}_H", "Q_C": rf"{dot}{{Q}}_C", "W": rf"{dot}{{W}}"}[name]

    QH, QC, W = sym("Q_H"), sym("Q_C"), sym("W")
    p_tex, rev_tex = _PERF_TEX[kind], _REV_TEX[kind]
    perf_name = PERFORMANCE_NAMES[kind]
    steps: list[ProcedureStep] = []

    temps = [_abs_line("T_H", inp.T_H_K, system), _abs_line("T_C", inp.T_C_K, system)]
    T0_default = inp.T_H_K if kind == "refrigerator" else inp.T_C_K
    if abs(r.T0 - T0_default) > 1e-9:
        temps.append(_abs_line("T_0", r.T0, system))
    steps.append(ProcedureStep("Temperaturas absolutas", _abs_text(system), tuple(temps)))

    # la máquina reversible primero si es el dato
    rev_lines = [_rev_line(kind, inp.T_H_K, inp.T_C_K, r.performance_rev, system)]
    rev_text = {
        "engine": "El rendimiento de Carnot (vademecum §9.2), el de una máquina reversible entre "
        "las mismas fuentes:",
        "refrigerator": "El COP de un refrigerador reversible entre las mismas fuentes "
        "(vademecum §9.3):",
        "heat_pump": "El COP de una bomba de calor reversible entre las mismas fuentes "
        "(vademecum §9.3):",
    }[kind]
    if inp.reversible:
        steps.append(
            ProcedureStep(
                "La máquina reversible",
                f"{rev_text} la máquina es reversible, así que {perf_name} es ese.",
                tuple(rev_lines),
            )
        )

    first = _first_law_lines(r, system, QH, QC, W, E)
    steps.append(
        ProcedureStep(
            "Primer principio",
            "El ciclo vuelve a su estado (ΔU = 0), así que el trabajo es la diferencia entre "
            f"los calores, {_w_text(kind)} (vademecum §9.2 y §9.3).",
            tuple(first),
        )
    )
    if "performance" not in inp.data and not inp.reversible:
        steps.append(
            ProcedureStep(
                "Rendimiento" if kind == "engine" else "Coeficiente de operación",
                {
                    "engine": "Lo que se aprovecha (el trabajo) sobre lo que se paga (el calor "
                    "de la fuente caliente):",
                    "refrigerator": "Lo que se aprovecha (el calor que se saca del espacio "
                    "frío) sobre lo que se paga (el trabajo):",
                    "heat_pump": "Lo que se aprovecha (el calor que se entrega) sobre lo que se "
                    "paga (el trabajo):",
                }[kind],
                (_perf_line(r, system, QH, QC, W, E),),
            )
        )
    if not inp.reversible:
        compare = "<" if r.performance < r.performance_rev else ">"
        if r.S_gen == 0.0:
            compare = "="
        steps.append(
            ProcedureStep(
                "La máquina reversible",
                f"{rev_text} ninguna máquina puede superarlo (teorema de Carnot, §9.2.1).",
                (
                    *rev_lines,
                    rf"{p_tex} = {_r(r.performance, 4)} {compare} {rev_tex} = "
                    rf"{_r(r.performance_rev, 4)}",
                ),
            )
        )

    # Clausius y S_gen
    if kind == "engine":
        terms = (frac(QH, "T_H"), frac(QC, "T_C"))
        nums = (
            frac(n(r.Q_H, E, system), n(inp.T_H_K, _TA, system)),
            frac(n(r.Q_C, E, system), n(inp.T_C_K, _TA, system)),
        )
        text = (
            "El ciclo recibe Q_H a T_H y entrega Q_C a T_C: con el signo del vademecum §9.1, "
            "ΣQ/T ≤ 0. Las fuentes ganan lo que pierde el ciclo: S_gen = −ΣQ/T (§10.2)."
        )
    else:
        terms = (frac(QC, "T_C"), frac(QH, "T_H"))
        nums = (
            frac(n(r.Q_C, E, system), n(inp.T_C_K, _TA, system)),
            frac(n(r.Q_H, E, system), n(inp.T_H_K, _TA, system)),
        )
        text = (
            "El ciclo recibe Q_C a T_C y entrega Q_H a T_H: con el signo del vademecum §9.1, "
            "ΣQ/T ≤ 0. Las fuentes ganan lo que pierde el ciclo: S_gen = −ΣQ/T (§10.2)."
        )
    sum_lhs = r"\sum \frac{Q}{T}" if not inp.rates else r"\sum \frac{\dot{Q}}{T}"
    sgen = r"\dot{S}_{\mathrm{gen}}" if inp.rates else r"S_{\mathrm{gen}}"
    steps.append(
        ProcedureStep(
            "Desigualdad de Clausius y entropía generada",
            text,
            (
                _stack(
                    sum_lhs,
                    rf"{terms[0]} - {terms[1]}",
                    rf"{nums[0]} - {nums[1]}",
                    q(r.clausius_sum, S, system),
                ),
                latex_chain(sgen, rf"-{sum_lhs}", q(r.S_gen, S, system)),
            ),
        )
    )

    # exergía
    xd = r"\dot{X}_{\mathrm{dest}}" if inp.rates else r"X_{\mathrm{dest}}"
    lines = [
        latex_chain(
            xd,
            rf"T_0\,{sgen}",
            times(n(r.T0, _TA, system), n(r.S_gen, S, system)),
            q(r.X_dest, E, system),
        )
    ]
    ideal = {
        "engine": (rf"{W}_{{\mathrm{{rev}}}}", rf"{rev_tex}\,{QH}"),
        "refrigerator": (rf"{W}_{{\mathrm{{rev}}}}", frac(QC, rev_tex)),
        "heat_pump": (rf"{W}_{{\mathrm{{rev}}}}", frac(QH, rev_tex)),
    }[kind]
    lines.append(_chain(ideal[0], ideal[1], q(r.W_ideal, E, system)))
    lines.append(
        latex_chain(r"\eta_{\mathrm{ex}}", _eta_ex_formula(kind, QH, QC, W), _r(r.eta_ex, 4))
    )
    what = {
        "engine": "el trabajo de la máquina reversible con el mismo Q_H (el máximo)",
        "refrigerator": "el trabajo de la máquina reversible con el mismo Q_C (el mínimo)",
        "heat_pump": "el trabajo de la máquina reversible con el mismo Q_H (el mínimo)",
    }[kind]
    steps.append(
        ProcedureStep(
            "Exergía destruida",
            f"X_dest = T₀·S_gen (vademecum §11.8); W_rev es {what}, y η_ex, el rendimiento "
            "exergético (§11.10).",
            tuple(lines),
        )
    )
    steps.append(
        ProcedureStep(
            "Veredicto",
            _verdict_text(r.verdict, r.violation, "S_gen"),
            (),
        )
    )
    return numbered(steps)


def _w_text(kind: str) -> str:
    return "W = Q_H − Q_C" if kind == "engine" else "W = Q_H − Q_C (el trabajo que consume)"


def _rev_line(kind: str, T_H: float, T_C: float, value: float, system: UnitSystem) -> str:
    th, tc = n(T_H, _TA, system), n(T_C, _TA, system)
    if kind == "engine":
        return latex_chain(
            _REV_TEX[kind], r"1 - \frac{T_C}{T_H}", rf"1 - \frac{{{tc}}}{{{th}}}", _r(value, 4)
        )
    num = "T_C" if kind == "refrigerator" else "T_H"
    num_n = tc if kind == "refrigerator" else th
    return _chain(
        _REV_TEX[kind],
        frac(num, "T_H - T_C"),
        frac(num_n, rf"{th} - {tc}"),
        _r(value, 4),
    )


def _first_law_lines(
    r: MachineResult, system: UnitSystem, QH: str, QC: str, W: str, E: QuantityKind
) -> list[str]:
    """Lo que falta con W = Q_H − Q_C y, si es dato, la definición del rendimiento."""
    inp = r.inputs
    kind = inp.kind
    data = set(inp.data)
    p = r.performance
    p_tex = _PERF_TEX[kind] if "performance" in data else _REV_TEX[kind]
    p_n = _r(p, 4)
    qh, qc, w = n(r.Q_H, E, system), n(r.Q_C, E, system), n(r.W, E, system)
    QH_q, QC_q, W_q = q(r.Q_H, E, system), q(r.Q_C, E, system), q(r.W, E, system)
    if "performance" not in data and "reversible" not in data:
        if "W" not in data:
            return [_chain(W, rf"{QH} - {QC}", rf"{n(r.Q_H, E, system)} - {latex_paren(qc)}", W_q)]
        if "Q_C" not in data:
            return [_chain(QC, rf"{QH} - {W}", rf"{qh} - {latex_paren(w)}", QC_q)]
        return [_chain(QH, rf"{QC} + {W}", rf"{qc} + {latex_paren(w)}", QH_q)]
    known = next(name for name in ("Q_H", "Q_C", "W") if name in data)
    lines: list[str] = []
    if kind == "engine":
        if known == "Q_H":
            lines.append(_chain(W, rf"{p_tex}\,{QH}", times(p_n, qh), W_q))
        elif known == "W":
            lines.append(_chain(QH, frac(W, p_tex), frac(w, p_n), QH_q))
        else:
            lines.append(_chain(QH, frac(QC, rf"1 - {p_tex}"), frac(qc, rf"1 - {p_n}"), QH_q))
        if known != "Q_C":
            lines.append(_chain(QC, rf"{QH} - {W}", rf"{qh} - {latex_paren(w)}", QC_q))
        else:
            lines.append(_chain(W, rf"{QH} - {QC}", rf"{qh} - {latex_paren(qc)}", W_q))
        return lines
    if kind == "refrigerator":
        if known == "W":
            lines.append(_chain(QC, rf"{p_tex}\,{W}", times(p_n, w), QC_q))
            lines.append(_chain(QH, rf"{QC} + {W}", rf"{qc} + {latex_paren(w)}", QH_q))
        elif known == "Q_C":
            lines.append(_chain(W, frac(QC, p_tex), frac(qc, p_n), W_q))
            lines.append(_chain(QH, rf"{QC} + {W}", rf"{qc} + {latex_paren(w)}", QH_q))
        else:
            lines.append(_chain(W, frac(QH, rf"1 + {p_tex}"), frac(qh, rf"1 + {p_n}"), W_q))
            lines.append(_chain(QC, rf"{QH} - {W}", rf"{qh} - {latex_paren(w)}", QC_q))
        return lines
    if known == "W":
        lines.append(_chain(QH, rf"{p_tex}\,{W}", times(p_n, w), QH_q))
        lines.append(_chain(QC, rf"{QH} - {W}", rf"{qh} - {latex_paren(w)}", QC_q))
    elif known == "Q_H":
        lines.append(_chain(W, frac(QH, p_tex), frac(qh, p_n), W_q))
        lines.append(_chain(QC, rf"{QH} - {W}", rf"{qh} - {latex_paren(w)}", QC_q))
    else:
        lines.append(_chain(W, frac(QC, rf"{p_tex} - 1"), frac(qc, rf"{p_n} - 1"), W_q))
        lines.append(_chain(QH, rf"{QC} + {W}", rf"{qc} + {latex_paren(w)}", QH_q))
    return lines


def _perf_line(
    r: MachineResult, system: UnitSystem, QH: str, QC: str, W: str, E: QuantityKind
) -> str:
    kind = r.inputs.kind
    num_sym, num_v = {
        "engine": (W, r.W),
        "refrigerator": (QC, r.Q_C),
        "heat_pump": (QH, r.Q_H),
    }[kind]
    den_sym, den_v = (QH, r.Q_H) if kind == "engine" else (W, r.W)
    return latex_chain(
        _PERF_TEX[kind],
        frac(num_sym, den_sym),
        frac(n(num_v, E, system), n(den_v, E, system)),
        _r(r.performance, 4),
    )


def _eta_ex_formula(kind: str, QH: str, QC: str, W: str) -> str:
    if kind == "engine":
        return frac(W, rf"{QH}\left(1 - \frac{{T_0}}{{T_H}}\right)")
    if kind == "refrigerator":
        return frac(rf"{QC}\left(\frac{{T_0}}{{T_C}} - 1\right)", W)
    return frac(rf"{QH}\left(1 - \frac{{T_0}}{{T_H}}\right)", W)


# ---------------------------------------------------------------------
# Clausius
# ---------------------------------------------------------------------


def clausius_steps(r: ClausiusResult, system: UnitSystem) -> list[ProcedureStep]:
    """Cada Q/T, la suma, S_gen, el trabajo neto y la exergía."""
    inp = r.inputs
    E, S = _kinds(inp.rates)
    Qs = r"\dot{Q}" if inp.rates else "Q"
    steps: list[ProcedureStep] = []
    temps = tuple(
        _abs_line(f"T_{{{k}}}", res.T_K, system) for k, res in enumerate(inp.reservoirs, 1)
    )
    steps.append(ProcedureStep("Temperaturas absolutas", _abs_text(system), temps))
    lines = []
    for k, (res, term) in enumerate(zip(inp.reservoirs, r.terms, strict=True), 1):
        lines.append(
            latex_chain(
                frac(f"{Qs}_{{{k}}}", f"T_{{{k}}}"),
                frac(n(res.Q, E, system), n(res.T_K, _TA, system)),
                q(term, S, system),
            )
        )
    names = ", ".join(
        f"{k} = {res.label}" if res.label else str(k) for k, res in enumerate(inp.reservoirs, 1)
    )
    steps.append(
        ProcedureStep(
            "Cada fuente",
            f"Q > 0 es el calor que el ciclo recibe de la fuente y Q < 0 el que le entrega "
            f"(vademecum §9.1). Fuentes: {names}.",
            tuple(lines),
        )
    )
    sum_lhs = rf"\sum \frac{{{Qs}}}{{T}}"
    terms = [n(t, S, system) for t in r.terms]
    total = " + ".join(latex_paren(t) if i else t for i, t in enumerate(terms))
    if len(terms) > 2 or any(r"\times" in t for t in terms):
        body = _stack(sum_lhs, *_rows(terms), q(r.clausius_sum, S, system))
    else:
        body = latex_chain(sum_lhs, total, q(r.clausius_sum, S, system))
    sgen = r"\dot{S}_{\mathrm{gen}}" if inp.rates else r"S_{\mathrm{gen}}"
    steps.append(
        ProcedureStep(
            "Desigualdad de Clausius",
            "La suma tiene que ser ≤ 0 (= 0 si el ciclo es reversible). S_gen = −ΣQ/T es la "
            "entropía que ganan las fuentes (vademecum §10.2).",
            (body, latex_chain(sgen, rf"-{sum_lhs}", q(r.S_gen, S, system))),
        )
    )
    W = r"\dot{W}" if inp.rates else "W"
    first = [
        _stack(
            W,
            rf"\sum {Qs}",
            *_rows([n(res.Q, E, system) for res in inp.reservoirs]),
            q(r.W, E, system),
        )
    ]
    if r.efficiency is not None:
        first.append(
            latex_chain(
                r"\eta",
                frac(W, rf"\sum {Qs}_{{\mathrm{{entra}}}}"),
                frac(n(r.W, E, system), n(r.Q_in, E, system)),
                _r(r.efficiency, 4),
            )
        )
    steps.append(
        ProcedureStep(
            "Primer principio",
            "En un ciclo ΔU = 0: el trabajo neto es la suma de los calores (W < 0 si lo consume).",
            tuple(first),
        )
    )
    xd = r"\dot{X}_{\mathrm{dest}}" if inp.rates else r"X_{\mathrm{dest}}"
    steps.append(
        ProcedureStep(
            "Exergía destruida",
            "X_dest = T₀·S_gen (vademecum §11.8). Un ciclo reversible con los mismos calores "
            "daría W + X_dest, la exergía neta de los calores ΣQ·(1 − T₀/T) (§11.5).",
            (
                _abs_line("T_0", inp.T0_K, system),
                latex_chain(
                    xd,
                    rf"T_0\,{sgen}",
                    times(n(inp.T0_K, _TA, system), n(r.S_gen, S, system)),
                    q(r.X_dest, E, system),
                ),
                latex_chain(rf"{W} + {xd}", q(r.X_heat, E, system)),
            ),
        )
    )
    steps.append(ProcedureStep("Veredicto", _verdict_text(r.verdict, r.violation, "S_gen"), ()))
    return numbered(steps)


def _rows(terms: list[str], per_row: int = 2) -> list[str]:
    """Una suma larga en renglones de ``per_row`` términos (de a uno con ×10ⁿ), como un solo
    paso de :func:`_stack`."""
    if any(r"\times" in t for t in terms):
        per_row = 1
    rows = []
    for k in range(0, len(terms), per_row):
        chunk = terms[k : k + per_row]
        body = " + ".join(latex_paren(t) if (j or k) else t for j, t in enumerate(chunk))
        rows.append(body)
    if len(rows) == 1:
        return rows
    return [rows[0] + "".join(rf" \\ &\quad + {row}" for row in rows[1:])]


# ---------------------------------------------------------------------
# Ciclo de Carnot
# ---------------------------------------------------------------------


def carnot_steps(r: CarnotResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los cuatro estados, los calores, el trabajo de cada proceso y el rendimiento."""
    inp = r.inputs
    sub = inp.substance
    closed = inp.flow == "closed"
    e_sym, e_kind = ("u", _E) if closed else ("h", _E)
    m = inp.mass_kg
    steps: list[ProcedureStep] = []
    steps.append(
        ProcedureStep(
            "Temperaturas absolutas",
            _abs_text(system),
            (_abs_line("T_H", inp.T_H_K, system), _abs_line("T_C", inp.T_C_K, system)),
        )
    )
    s1, s2, s3, s4 = r.states
    lines = []
    for label, st, prop in (
        ("1", s1, inp.prop1),
        ("2", s2, inp.prop2),
    ):
        given = {"p": (f"p_{label}", _P), "x": (f"x_{label}", None), "v": (f"v_{label}", _V)}[prop]
        arg = rf"T_H, {given[0]}"
        lines.append(_chain(f"s_{label}", rf"s({arg})", q(st.s, _S, system)))
        lines.append(
            _chain(f"{e_sym}_{label}", rf"{e_sym}({arg})", q(getattr(st, e_sym), e_kind, system))
        )
    data = ", ".join(
        f"{label}: {CARNOT_PROPS[prop]}"
        for label, prop in (("estado 1", inp.prop1), ("estado 2", inp.prop2))
    )
    steps.append(
        ProcedureStep(
            "Estados 1 y 2 (a T_H)",
            f"{_model_text(sub)} Con T_H y otra propiedad ({data}).",
            tuple(lines),
        )
    )
    lines = []
    for label, st, s_from in (("3", s3, "2"), ("4", s4, "1")):
        lines.append(rf"s_{label} = s_{s_from} = {q(st.s, _S, system)}")
        lines.append(
            _chain(
                f"{e_sym}_{label}",
                rf"{e_sym}(T_C, s_{label})",
                q(getattr(st, e_sym), e_kind, system),
            )
        )
        if st.x is not None and sub.kind == "fluid":
            lines.append(latex_chain(f"x_{label}", _r(st.x, 4)))
    steps.append(
        ProcedureStep(
            "Estados 3 y 4 (a T_C)",
            "Las isoentrópicas (2 → 3 y 4 → 1) no cambian s: los estados 3 y 4 son los de T_C "
            "con la entropía de 2 y de 1.",
            tuple(lines),
        )
    )
    Q = "Q"
    ds = q(s2.s - s1.s, _S, system)
    mass = n(m, _M, system)
    steps.append(
        ProcedureStep(
            "Calores (isotermas)",
            "A T constante y reversible, Q = m·T·Δs (vademecum §10.1): el fluido recibe Q_H a T_H "
            "y cede Q_C a T_C con el mismo Δs.",
            (
                latex_chain(r"s_2 - s_1", ds),
                _chain(
                    rf"{Q}_H",
                    r"m\,T_H\,(s_2 - s_1)",
                    times(mass, n(inp.T_H_K, _TA, system), n(s2.s - s1.s, _S, system)),
                    q(r.Q_H, "energy", system),
                ),
                _chain(
                    rf"{Q}_C",
                    r"m\,T_C\,(s_2 - s_1)",
                    times(mass, n(inp.T_C_K, _TA, system), n(s2.s - s1.s, _S, system)),
                    q(r.Q_C, "energy", system),
                ),
            ),
        )
    )
    lines = []
    for p in r.processes:
        i, j = p.start, p.end
        dE = rf"m\,({e_sym}_{j} - {e_sym}_{i})"
        lines.append(
            _stack(
                rf"W_{{{i}{j}}}",
                rf"Q_{{{i}{j}}} - {dE}",
                rf"{n(p.Q, 'energy', system)} - {latex_paren(n(p.dE, 'energy', system))}",
                q(p.W, "energy", system),
            )
        )
    how = (
        "W = Q − ΔU en cada proceso (vademecum §3.1): es el trabajo de frontera del pistón."
        if closed
        else "W = Q − ΔH en cada equipo (vademecum §3.3): caldera y condensador no hacen "
        "trabajo, la turbina lo entrega y el compresor lo consume."
    )
    steps.append(ProcedureStep("Trabajo de cada proceso", how, tuple(lines)))
    terms = [n(p.W, "energy", system) for p in r.processes]
    steps.append(
        ProcedureStep(
            "Trabajo neto y rendimiento",
            "El neto es Q_H − Q_C y el rendimiento, 1 − T_C/T_H: no depende del fluido "
            "(vademecum §9.2).",
            (
                _stack(r"W_{\mathrm{neto}}", *_rows(terms), q(r.W_net, "energy", system)),
                latex_chain(
                    r"\eta",
                    frac(r"W_{\mathrm{neto}}", "Q_H"),
                    frac(n(r.W_net, "energy", system), n(r.Q_H, "energy", system)),
                    _r(r.eta, 4),
                ),
                latex_chain(
                    r"\eta_{\mathrm{Carnot}}",
                    r"1 - \frac{T_C}{T_H}",
                    _r(r.eta_carnot, 4),
                ),
            ),
        )
    )
    return numbered(steps)


# ---------------------------------------------------------------------
# La entropía del calor
# ---------------------------------------------------------------------


def heat_path_steps(r: HeatPathResult, system: UnitSystem) -> list[ProcedureStep]:
    """La S_gen de cada tramo, la total y la exergía del calor a cada temperatura."""
    inp = r.inputs
    E, S = _kinds(inp.rates, heat=True)
    Qs = r"\dot{Q}" if inp.rates else "Q"
    sgen = r"\dot{S}_{\mathrm{gen}}" if inp.rates else r"S_{\mathrm{gen}}"
    steps: list[ProcedureStep] = []
    temps = tuple(_abs_line(f"T_{{{k}}}", T, system) for k, T in enumerate(inp.temperatures, 1))
    names = ", ".join(f"{k} = {name}" for k, name in enumerate(inp.points, 1))
    steps.append(
        ProcedureStep("Temperaturas absolutas", f"{_abs_text(system)} Puntos: {names}.", temps)
    )
    Qn = n(inp.Q, E, system)
    lines = []
    for k, z in enumerate(r.zones, 1):
        sgen_k = rf"\dot{{S}}_{{\mathrm{{gen}},{k}}}" if inp.rates else rf"S_{{\mathrm{{gen}},{k}}}"
        lines.append(
            _stack(
                sgen_k,
                rf"\frac{{{Qs}}}{{T_{{{k + 1}}}}} - \frac{{{Qs}}}{{T_{{{k}}}}}",
                rf"{frac(Qn, n(z.T_out, _TA, system))} - {frac(Qn, n(z.T_in, _TA, system))}",
                q(z.S_gen, S, system),
            )
        )
    zones = ", ".join(f"{k} = {z.name}" for k, z in enumerate(r.zones, 1))
    steps.append(
        ProcedureStep(
            "Entropía generada en cada tramo",
            "Cada tramo no cambia (el calor entra y sale igual): el balance de entropía da "
            f"S_gen = Q/T_sale − Q/T_entra (vademecum §10.3). Tramos: {zones}.",
            tuple(lines),
        )
    )
    last = len(inp.temperatures)
    total = [
        _stack(
            sgen,
            rf"\frac{{{Qs}}}{{T_{{{last}}}}} - \frac{{{Qs}}}{{T_1}}",
            q(r.S_gen, S, system),
        )
    ]
    if len(r.zones) > 1:
        total.insert(
            0, _stack(sgen, *_rows([n(z.S_gen, S, system) for z in r.zones]), q(r.S_gen, S, system))
        )
    steps.append(
        ProcedureStep(
            "Entropía generada total",
            "La suma de los tramos: solo importan la primera y la última temperatura.",
            tuple(total),
        )
    )
    xd = r"\dot{X}_{\mathrm{dest}}" if inp.rates else r"X_{\mathrm{dest}}"
    X = r.exergies
    x_lines = [
        _abs_line("T_0", inp.T0_K, system),
        _stack(
            r"X_{Q,1}",
            rf"{Qs}\left(1 - \frac{{T_0}}{{T_1}}\right)",
            q(X[0], E, system),
        ),
        _stack(
            rf"X_{{Q,{last}}}",
            rf"{Qs}\left(1 - \frac{{T_0}}{{T_{{{last}}}}}\right)",
            q(X[-1], E, system),
        ),
        latex_chain(
            xd,
            rf"T_0\,{sgen}",
            times(n(inp.T0_K, _TA, system), n(r.S_gen, S, system)),
            q(r.X_dest, E, system),
        ),
    ]
    steps.append(
        ProcedureStep(
            "Exergía del calor",
            "La exergía de un calor es Q·(1 − T₀/T) (vademecum §11.5): baja a medida que el calor "
            "pasa a menor temperatura, y lo que pierde es X_dest = T₀·S_gen (§11.8).",
            tuple(x_lines),
        )
    )
    steps.append(ProcedureStep("Veredicto", _verdict_text(r.verdict, r.violation, "S_gen"), ()))
    return numbered(steps)
