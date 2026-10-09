"""Procedimiento didáctico de la conducción (LaTeX), en el sistema de unidades activo.

Cengel y Ghajar (2015), §3-1 a §3-5: las resistencias de la red (pared plana,
cilindro, esfera, convección, contacto y partes en paralelo), la total, Q̇, las
temperaturas de cada cara, el coeficiente global U y el radio crítico.

Las temperaturas de la red se numeran en el sentido del calor: T_∞,1 (el fluido
del lado 1), T₁, T₂… (las caras del sólido, dos en un contacto) y T_∞,2.
"""

from __future__ import annotations

import math

from core.heat_transfer.conduction import (
    INSULATION_K_MAX,
    ConductionResult,
    Resistance,
)
from core.heat_transfer.procedure_common import (
    dimensionless,
    frac,
    n,
    numbered,
    q,
    sub,
    sum_rows,
    times,
)
from core.latex import latex_chain, latex_is_wide, latex_paren
from core.state_report import ProcedureStep
from core.units_system import UnitSystem

__all__ = ["conduction_steps", "node_symbols", "resistance_symbols"]

_R = "thermal_resistance"
_T = "temperature"
_L = "length"
_K = "thermal_conductivity"
_H = "heat_transfer_coefficient"
_PART_LETTERS = "abcdefgh"


def resistance_symbols(result: ConductionResult) -> list[str]:
    r"""El símbolo LaTeX de cada resistencia: R_{\text{conv},1}, R_1, R_{c,1}…"""
    out = []
    for r in result.resistances:
        if r.kind == "convección":
            side = 1 if r is result.resistances[0] else 2
            out.append(rf"R_{{\text{{conv}},{side}}}")
        elif r.kind == "contacto":
            out.append(rf"R_{{c,{_contact_index(r)}}}")
        else:
            out.append(rf"R_{{{(r.layer or 0) + 1}}}")
    return out


def _contact_index(r: Resistance) -> int:
    """La capa que va antes del contacto («Contacto 1–2» → 1)."""
    return int(r.label.split()[1].split("–")[0])


def node_symbols(result: ConductionResult) -> list[str]:
    r"""Las temperaturas de la red en orden: T_{\infty,1}, T_1, T_2, …, T_{\infty,2}."""
    i = result.inputs
    res = result.resistances
    nodes = [r"T_{\infty,1}" if i.inner.kind == "fluid" else "T_1"]
    k = 1 if i.inner.kind == "fluid" else 2
    for j in range(len(res)):
        if j == len(res) - 1 and i.outer.kind == "fluid":
            nodes.append(r"T_{\infty,2}")
        else:
            nodes.append(f"T_{{{k}}}")
            k += 1
    return nodes


def _area_symbol(result: ConductionResult, x_m: float) -> str:
    """A (pared) o A_j de la cara j (cilindro y esfera)."""
    if result.inputs.geometry == "plane":
        return "A"
    return f"A_{{{result.faces_m.index(x_m) + 1}}}"


def _area_value(result: ConductionResult, x_m: float) -> float:
    i = result.inputs
    if i.geometry == "plane":
        return i.area_m2
    if i.geometry == "cylinder":
        return 2.0 * math.pi * x_m * i.length_m
    return 4.0 * math.pi * x_m * x_m


def _R_total_symbol(result: ConductionResult) -> str:
    return r"R_{\text{total}}" if len(result.resistances) > 1 else resistance_symbols(result)[0]


# ---------------------------------------------------------------------
# Pasos
# ---------------------------------------------------------------------


def _geometry_step(result: ConductionResult, system: UnitSystem) -> ProcedureStep:
    i = result.inputs
    if i.geometry == "plane":
        text = (
            "En una **pared plana** el área es la misma en todas las capas: el calor atraviesa "
            "cada una en serie."
        )
        return ProcedureStep("Geometría", text, (rf"A = {q(i.area_m2, 'area', system)}",))
    faces = result.faces_m
    latex = [rf"r_1 = {q(faces[0], _L, system)}"]
    for j, layer in enumerate(i.layers, start=1):
        latex.append(
            latex_chain(
                f"r_{{{j + 1}}}",
                f"r_{{{j}}} + t_{{{j}}}",
                f"{n(faces[j - 1], _L, system)} + {n(layer.thickness_m, _L, system)}",
                q(faces[j], _L, system),
            )
        )
    # las caras de los extremos (U₁ y U₂) y las de la convección y el contacto
    ends = {faces[0], faces[-1]}
    for x in sorted(ends | {r.x_from_m for r in result.resistances if r.kind != "conducción"}):
        j = faces.index(x) + 1
        r_n = n(x, _L, system)
        if i.geometry == "cylinder":
            formula = rf"2\pi\,r_{{{j}}}\,L"
            numbers = rf"2\pi \cdot {times(r_n, n(i.length_m, _L, system))}"
        else:
            formula = rf"4\pi\,r_{{{j}}}^2"
            numbers = rf"4\pi \cdot {r_n}^2"
        latex.append(
            latex_chain(
                _area_symbol(result, x), formula, numbers, q(_area_value(result, x), "area", system)
            )
        )
    if i.geometry == "cylinder":
        text = (
            "En un **cilindro** el área crece con el radio, A = 2π·r·L: las caras de cada capa "
            "tienen radios distintos, y la convección y el contacto usan el área de su cara."
        )
    else:
        text = "En una **esfera** el área crece con el radio al cuadrado, A = 4π·r²."
    return ProcedureStep("Radios y áreas", text, tuple(latex))


def _composite_latex(
    result: ConductionResult, r: Resistance, sym: str, system: UnitSystem
) -> list[str]:
    """La capa con partes en paralelo: k efectiva, R de la capa y R de cada parte."""
    i = result.inputs
    layer = i.layers[r.layer or 0]
    j = (r.layer or 0) + 1
    letters = _PART_LETTERS[: len(layer.parts)]
    terms_f = [rf"f_{{{a}}}\,k_{{{a}}}" for a in letters]
    terms_n = [
        times(dimensionless(p.area_fraction), n(p.k_W_per_mK, _K, system)) for p in layer.parts
    ]
    k_ef = n(layer.k_effective, _K, system)
    latex = [
        latex_chain(
            rf"k_{{{j},\text{{ef}}}}",
            " + ".join(terms_f),
            sum_rows(terms_n, per_row=2),
            q(layer.k_effective, _K, system),
        ),
        latex_chain(
            sym,
            frac(f"L_{{{j}}}", rf"k_{{{j},\text{{ef}}}}\,A"),
            frac(n(layer.thickness_m, _L, system), times(k_ef, n(i.area_m2, "area", system))),
            q(r.R_K_per_W, _R, system),
        ),
    ]
    for a, part, flow in zip(letters, layer.parts, r.parts, strict=True):
        den = times(
            n(part.k_W_per_mK, _K, system),
            dimensionless(part.area_fraction),
            n(i.area_m2, "area", system),
        )
        latex.append(
            latex_chain(
                rf"R_{{{j}{a}}}",
                frac(f"L_{{{j}}}", rf"k_{{{a}}}\,f_{{{a}}}\,A"),
                frac(n(layer.thickness_m, _L, system), den),
                q(flow.R_K_per_W, _R, system),
            )
        )
    return latex


def _conduction_latex(
    result: ConductionResult, r: Resistance, sym: str, system: UnitSystem
) -> list[str]:
    i = result.inputs
    layer = i.layers[r.layer or 0]
    if layer.parts:
        return _composite_latex(result, r, sym, system)
    j = (r.layer or 0) + 1
    k_n = n(layer.k_W_per_mK, _K, system)
    r1, r2 = n(r.x_from_m, _L, system), n(r.x_to_m, _L, system)
    if i.geometry == "plane":
        formula = frac(f"L_{{{j}}}", rf"k_{{{j}}}\,A")
        numbers = frac(n(layer.thickness_m, _L, system), times(k_n, n(i.area_m2, "area", system)))
    elif i.geometry == "cylinder":
        formula = frac(rf"\ln(r_{{{j + 1}}}/r_{{{j}}})", rf"2\pi\,k_{{{j}}}\,L")
        numbers = frac(rf"\ln({r2}/{r1})", rf"2\pi \cdot {times(k_n, n(i.length_m, _L, system))}")
    else:
        formula = frac(f"r_{{{j + 1}}} - r_{{{j}}}", rf"4\pi\,k_{{{j}}}\,r_{{{j}}}\,r_{{{j + 1}}}")
        numbers = frac(f"{r2} - {r1}", rf"4\pi \cdot {times(k_n, r1, r2)}")
    return [latex_chain(sym, formula, numbers, q(r.R_K_per_W, _R, system))]


def _surface_latex(
    result: ConductionResult, r: Resistance, sym: str, system: UnitSystem
) -> list[str]:
    """Convección (1/(h·A)) o contacto (R''_c/A = 1/(h_c·A))."""
    i = result.inputs
    latex = []
    A = _area_value(result, r.x_from_m)
    if r.kind == "convección":
        side = 1 if r is result.resistances[0] else 2
        h = (i.inner if side == 1 else i.outer).h_W_per_m2K or 0.0
        h_sym = f"h_{side}"
    else:
        h = 1.0 / i.layers[_contact_index(r) - 1].contact_m2K_per_W
        h_sym = "h_c"
        latex.append(latex_chain("h_c", frac("1", "R''_c"), q(h, _H, system)))
    latex.append(
        latex_chain(
            sym,
            frac("1", rf"{h_sym}\,{_area_symbol(result, r.x_from_m)}"),
            frac("1", times(n(h, _H, system), n(A, "area", system))),
            q(r.R_K_per_W, _R, system),
        )
    )
    return latex


def _resistances_step(result: ConductionResult, system: UnitSystem) -> ProcedureStep:
    i = result.inputs
    latex: list[str] = []
    for r, sym in zip(result.resistances, resistance_symbols(result), strict=True):
        if r.kind == "conducción":
            latex.extend(_conduction_latex(result, r, sym, system))
        else:
            latex.extend(_surface_latex(result, r, sym, system))
    geo = {
        "plane": "pared plana, R = L/(k·A)",
        "cylinder": "capa cilíndrica, R = ln(r₂/r₁)/(2π·k·L)",
        "sphere": "capa esférica, R = (r₂ − r₁)/(4π·k·r₁·r₂)",
    }[i.geometry]
    text = (
        "Cada elemento entre dos temperaturas es una **resistencia térmica** R = ΔT/Q̇ "
        f"(Cengel y Ghajar, §3-1 y §3-4). Conducción en una {geo}; convección en una "
        "superficie, R = 1/(h·A)."
    )
    if any(r.kind == "contacto" for r in result.resistances):
        text += (
            " El **contacto** entre dos capas es una resistencia R''_c por unidad de área; su "
            "inversa es la conductancia de contacto h_c (§3-2)."
        )
    for layer in i.layers:
        if layer.parts:
            names = "; ".join(
                f"{a} = {p.name}" for a, p in zip(_PART_LETTERS, layer.parts, strict=False)
            )
            text += (
                " La capa compuesta tiene sus partes **en paralelo**, con las caras isotérmicas "
                "(§3-3): 1/R = Σ 1/R_i, que es lo mismo que usar la k efectiva Σ f_i·k_i, con f "
                f"la fracción del área ({names})."
            )
    return ProcedureStep("Resistencias", text, tuple(latex))


def _total_step(result: ConductionResult, system: UnitSystem) -> ProcedureStep | None:
    if len(result.resistances) < 2:
        return None
    numbers = [n(r.R_K_per_W, _R, system) for r in result.resistances]
    latex = latex_chain(
        r"R_{\text{total}}",
        sum_rows(resistance_symbols(result), per_row=3),
        # con ×10ⁿ, un término por renglón (al lado de R_total no entran dos)
        sum_rows(numbers, per_row=1 if any(r"\times" in x for x in numbers) else 2),
        q(result.R_total_K_per_W, _R, system),
    )
    text = "Las resistencias están **en serie** (el mismo Q̇ las atraviesa a todas): se suman."
    return ProcedureStep("Resistencia total", text, (latex,))


def _heat_step(result: ConductionResult, system: UnitSystem) -> ProcedureStep:
    i = result.inputs
    nodes = node_symbols(result)
    R_tot = _R_total_symbol(result)
    QR = times(n(result.Q_W, "heat_rate", system), n(result.R_total_K_per_W, _R, system))
    latex: list[str] = []
    if i.inner.kind == "heat":
        text = (
            "El calor es dato (lo que se genera en el alambre, por ejemplo) y la temperatura del "
            "otro extremo es conocida: la del lado 1 sale de ΔT = Q̇·R."
        )
        latex.append(rf"\dot{{Q}} = {q(result.Q_W, 'heat_rate', system)}")
        latex.append(
            latex_chain(
                nodes[0],
                rf"{nodes[-1]} + \dot{{Q}}\,{R_tot}",
                f"{n(result.T_side2_K, _T, system)} + {QR}",
                q(result.T_side1_K, _T, system),
            )
        )
    elif i.outer.kind == "heat":
        text = (
            "El calor que entra por el lado 2 es dato: va hacia el lado 1, así que Q̇ (del lado 1 "
            "al 2) es negativo."
        )
        latex.append(rf"\dot{{Q}} = {q(result.Q_W, 'heat_rate', system)}")
        latex.append(
            latex_chain(
                nodes[-1],
                rf"{nodes[0]} - \dot{{Q}}\,{R_tot}",
                sub(n(result.T_side1_K, _T, system), QR),
                q(result.T_side2_K, _T, system),
            )
        )
    else:
        text = (
            "Como en un circuito eléctrico, el «potencial» es la temperatura: Q̇ = ΔT/R_total "
            "entre los dos extremos conocidos."
        )
        if result.Q_W < 0.0:
            text += " Sale negativo: el calor va del lado 2 al lado 1."
        dT = sub(n(result.T_side1_K, _T, system), n(result.T_side2_K, _T, system))
        latex.append(
            latex_chain(
                r"\dot{Q}",
                frac(f"{nodes[0]} - {nodes[-1]}", R_tot),
                frac(dT, n(result.R_total_K_per_W, _R, system)),
                q(result.Q_W, "heat_rate", system),
            )
        )
    if result.Q_per_length_W_per_m is not None:
        latex.append(
            latex_chain(
                r"\dot{Q}/L",
                frac(n(result.Q_W, "heat_rate", system), n(i.length_m, _L, system)),
                q(result.Q_per_length_W_per_m, "linear_heat_rate", system),
            )
        )
    return ProcedureStep("Calor", text, tuple(latex))


def _temperatures_step(result: ConductionResult, system: UnitSystem) -> ProcedureStep | None:
    nodes = node_symbols(result)
    res = result.resistances
    Q_n = n(result.Q_W, "heat_rate", system)
    latex: list[str] = []
    # el extremo que no es dato ya salió en el paso del calor
    known_last = result.inputs.outer.kind != "heat"
    for j, (r, sym) in enumerate(zip(res, resistance_symbols(result), strict=True)):
        if j == len(res) - 1 and known_last:
            break
        R_n = n(r.R_K_per_W, _R, system)
        T_prev = n(r.T_from_K, _T, system)
        QR = times(Q_n, R_n)
        # con ×10ⁿ o un Q̇ negativo, Q̇·R va en otro renglón
        numbers = (
            rf"{T_prev} \\ &\quad - {latex_paren(QR)}"
            if latex_is_wide(Q_n, R_n)
            else sub(T_prev, QR)
        )
        latex.append(
            latex_chain(
                nodes[j + 1], rf"{nodes[j]} - \dot{{Q}}\,{sym}", numbers, q(r.T_to_K, _T, system)
            )
        )
    for k, r in enumerate(res):
        j = (r.layer or 0) + 1
        dT = sub(n(r.T_from_K, _T, system), n(r.T_to_K, _T, system))
        for a, flow in zip(_PART_LETTERS, r.parts, strict=False):
            latex.append(
                latex_chain(
                    rf"\dot{{Q}}_{{{j}{a}}}",
                    frac(f"{nodes[k]} - {nodes[k + 1]}", rf"R_{{{j}{a}}}"),
                    frac(dT, n(flow.R_K_per_W, _R, system)),
                    q(flow.Q_W, "heat_rate", system),
                )
            )
    if not latex:
        return None
    text = (
        "Cada resistencia baja la temperatura en Q̇·R: recorriendo la red desde el lado 1 salen "
        "las temperaturas de todas las caras. La mayor caída está en la mayor resistencia."
    )
    if any(r.parts for r in res):
        text += " Por cada parte de la capa compuesta pasa Q̇ = ΔT/R_i, con la misma ΔT."
    return ProcedureStep("Temperaturas", text, tuple(latex))


def _overall_step(result: ConductionResult, system: UnitSystem) -> ProcedureStep:
    i = result.inputs
    R_tot = _R_total_symbol(result)
    R_n = n(result.R_total_K_per_W, _R, system)
    if i.geometry == "plane":
        latex = [
            latex_chain(
                "U",
                frac("1", rf"{R_tot}\,A"),
                frac("1", times(R_n, n(i.area_m2, "area", system))),
                q(result.U_inner_W_per_m2K, _H, system),
            )
        ]
        text = (
            "El **coeficiente global** U resume la red: Q̇ = U·A·ΔT, como la convección "
            "(Cengel y Ghajar, §3-1)."
        )
        return ProcedureStep("Coeficiente global U", text, tuple(latex))
    last = len(result.faces_m)
    latex = [
        latex_chain(
            f"U_{{{idx}}}",
            frac("1", rf"{R_tot}\,A_{{{idx}}}"),
            frac("1", times(R_n, n(A, "area", system))),
            q(U, _H, system),
        )
        for idx, U, A in (
            (1, result.U_inner_W_per_m2K, result.A_inner_m2),
            (last, result.U_outer_W_per_m2K, result.A_outer_m2),
        )
    ]
    text = (
        "El **coeficiente global** U depende del área a la que se refiere: Q̇ = U₁·A₁·ΔT = "
        "U₂·A₂·ΔT, con U₁·A₁ = U₂·A₂ = 1/R_total."
    )
    return ProcedureStep("Coeficiente global U", text, tuple(latex))


def _critical_step(result: ConductionResult, system: UnitSystem) -> ProcedureStep | None:
    r_cr = result.critical_radius_m
    i = result.inputs
    if r_cr is None or i.layers[-1].k_W_per_mK >= INSULATION_K_MAX:
        return None
    k_n = n(i.layers[-1].k_W_per_mK, _K, system)
    h_n = n(i.outer.h_W_per_m2K or 0.0, _H, system)
    if i.geometry == "cylinder":
        formula, numbers = frac("k", "h_2"), frac(k_n, h_n)
    else:
        formula, numbers = frac("2k", "h_2"), frac(rf"2 \cdot {k_n}", h_n)
    latex = [
        latex_chain(r"r_{\text{cr}}", formula, numbers, q(r_cr, _L, system)),
        rf"r_{{{len(result.faces_m)}}} = {q(result.faces_m[-1], _L, system)}",
    ]
    if result.faces_m[-1] < r_cr:
        text = (
            "**Radio crítico** (Cengel y Ghajar, §3-5): el radio exterior es menor que r_cr, así "
            "que agregar aislación aumenta la superficie de convección más de lo que suma de "
            "resistencia, y el calor sube hasta r = r_cr."
        )
    else:
        text = (
            "**Radio crítico** (Cengel y Ghajar, §3-5): el radio exterior supera r_cr, así que más "
            "aislación reduce el calor."
        )
    return ProcedureStep("Radio crítico de aislación", text, tuple(latex))


def conduction_steps(result: ConductionResult, system: UnitSystem) -> list[ProcedureStep]:
    """Los pasos de la red de resistencias, con los valores reemplazados en ``system``."""
    steps: list[ProcedureStep | None] = [
        _geometry_step(result, system),
        _resistances_step(result, system),
        _total_step(result, system),
        _heat_step(result, system),
        _temperatures_step(result, system),
        _overall_step(result, system),
        _critical_step(result, system),
    ]
    return numbered([s for s in steps if s is not None])
