"""Tests de los procedimientos LaTeX de transferencia de calor (Fase 8.1).

Que cada paso esté bien formado en los tres sistemas de unidades y que los
números que se muestran sean los del resultado (el ancho y KaTeX se verifican
aparte, con un navegador).
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest
from scipy.special import i0, i1, k0, k1

from core.heat_transfer import conduction as cd
from core.heat_transfer import convection as cv
from core.heat_transfer import fins as fn
from core.heat_transfer.conduction_procedure import (
    conduction_steps,
    node_symbols,
    resistance_symbols,
)
from core.heat_transfer.convection_procedure import (
    big,
    convection_steps,
    correlation_latex,
    internal_steps,
)
from core.heat_transfer.fins_procedure import fin_steps
from core.heat_transfer.procedure_common import sum_rows, times
from core.latex import latex_quantity
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")
C = 273.15


def _check_steps(steps: list[ProcedureStep]) -> str:
    """Pasos numerados, LaTeX balanceado y sin signos dobles; devuelve todo el LaTeX junto."""
    assert steps
    for k, st in enumerate(steps, 1):
        assert st.title.startswith(f"{k}. ")
        assert st.text
        for tex in st.latex:
            assert tex.count("{") == tex.count("}"), tex
            assert tex.count(r"\left") == tex.count(r"\right"), tex
            assert not re.search(r"[-+]\s*-\s*\d", tex), tex  # «- -3» sin paréntesis
            assert "nan" not in tex and "inf" not in tex.replace(r"\infty", ""), tex
    return "\n".join(t for st in steps for t in st.latex)


def _ends_with(tex: str, value: str) -> bool:
    return tex.replace(r"\end{aligned}", "").rstrip().endswith(value)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.7034, "0.7034"),
        (123.31, "123.3"),
        (1527.4, "1527"),
        (42_183.0, r"42\,180"),
        (1_824_321.0, r"1\,824\,000"),
        (1.824e7, r"1.824\times 10^{7}"),
        (2.4e-5, r"2.4\times 10^{-5}"),
    ],
)
def test_big_numbers(x: float, expected: str) -> None:
    assert big(x) == expected


def test_sum_rows_and_times() -> None:
    assert sum_rows(["a", "b", "c", "d"]) == r"a + b + c \\ &\quad + d"
    assert sum_rows(["1", "-2"]) == "1 + (-2)"
    # con ×10ⁿ, de a dos
    terms = [r"1\times 10^{-5}"] * 3
    assert sum_rows(terms).count(r"\\") == 1
    assert times("-2", "-3", "4") == r"-2 \cdot (-3) \cdot 4"


# ---------------------------------------------------------------------
# Conducción
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(cd.CONDUCTION_EXAMPLES))
@pytest.mark.parametrize("system", SYSTEMS)
def test_conduction_steps(name: str, system: str) -> None:
    r = cd.solve_conduction(cd.CONDUCTION_EXAMPLES[name])
    steps = conduction_steps(r, system)  # type: ignore[arg-type]
    _check_steps(steps)
    assert len(node_symbols(r)) == len(r.resistances) + 1
    assert len(resistance_symbols(r)) == len(r.resistances)
    heat = next(st for st in steps if st.title.endswith("Calor"))
    Q = latex_quantity(r.Q_W, "heat_rate", system)  # type: ignore[arg-type]
    assert any(Q in tex for tex in heat.latex)
    # cada resistencia sale con su valor
    resist = next(st for st in steps if st.title.endswith("Resistencias"))
    joined = "\n".join(resist.latex)
    for res in r.resistances:
        assert latex_quantity(res.R_K_per_W, "thermal_resistance", system) in joined  # type: ignore[arg-type]
    # el radio crítico solo con una aislación exterior en un cilindro o una esfera
    critical = any("Radio crítico" in st.title for st in steps)
    assert critical == (
        r.critical_radius_m is not None and r.inputs.layers[-1].k_W_per_mK < cd.INSULATION_K_MAX
    )


def test_conduction_symbols_with_contact_and_heat() -> None:
    names = list(cd.CONDUCTION_EXAMPLES)
    plates = cd.solve_conduction(cd.CONDUCTION_EXAMPLES[names[4]])
    assert resistance_symbols(plates) == ["R_{1}", "R_{c,1}", "R_{2}"]
    assert node_symbols(plates) == ["T_1", "T_{2}", "T_{3}", "T_{4}"]
    wire = cd.solve_conduction(cd.CONDUCTION_EXAMPLES[names[6]])
    assert node_symbols(wire) == ["T_1", "T_{2}", r"T_{\infty,2}"]
    heat = conduction_steps(wire, "Técnico")[3]
    assert heat.title == "4. Calor"
    assert any(tex.startswith(r"\begin{aligned}T_1 &= T_{\infty,2} + ") for tex in heat.latex)
    # calor dado del lado de afuera: Q̇ < 0 y la T del lado 2 sale en el paso del calor
    outer = replace(
        cd.CONDUCTION_EXAMPLES[names[6]],
        inner=cd.Boundary("fluid", T_K=300.0, h_W_per_m2K=20.0),
        outer=cd.Boundary("heat", Q_W=50.0),
    )
    r = cd.solve_conduction(outer)
    assert r.Q_W == -50.0
    _check_steps(conduction_steps(r, "SI"))


def test_composite_wall_shows_each_part() -> None:
    names = list(cd.CONDUCTION_EXAMPLES)
    r = cd.solve_conduction(cd.CONDUCTION_EXAMPLES[names[3]])
    joined = _check_steps(conduction_steps(r, "Técnico"))
    for a in "abc":
        assert f"R_{{3{a}}}" in joined and rf"\dot{{Q}}_{{3{a}}}" in joined
    assert r"k_{3,\text{ef}}" in joined


# ---------------------------------------------------------------------
# Aletas
# ---------------------------------------------------------------------

_FIN_CASES = (
    [(name, ex.fin, ex.n_fins, ex.base_area_m2) for name, ex in fn.FIN_EXAMPLES.items()]
    + [
        (f"{ex.fin.shape}-{tip}", replace(ex.fin, tip=tip, T_tip_K=None), None, None)
        for ex in list(fn.FIN_EXAMPLES.values())[3:]
        for tip in ("convective", "adiabatic", "corrected", "infinite")
    ]
    + [
        (
            f"{ex.fin.shape}-temperature",
            replace(ex.fin, tip="temperature", T_tip_K=ex.fin.T_inf_K + 0.4 * ex.fin.theta_b),
            None,
            None,
        )
        for ex in list(fn.FIN_EXAMPLES.values())[3:]
    ]
)


@pytest.mark.parametrize(("name", "fin", "n_fins", "base"), _FIN_CASES)
@pytest.mark.parametrize("system", SYSTEMS)
def test_fin_steps(
    name: str, fin: fn.FinInputs, n_fins: int | None, base: float | None, system: str
) -> None:
    r = fn.solve_fin(fin)
    a = fn.solve_fin_array(fn.FinArrayInputs(fin, n_fins, base)) if n_fins else None
    steps = fin_steps(r, system, a)  # type: ignore[arg-type]
    joined = _check_steps(steps)
    Q = latex_quantity(r.Q_W, "heat_rate", system)  # type: ignore[arg-type]
    heat = next(st for st in steps if st.title.endswith("Calor de la aleta"))
    assert any(_ends_with(tex, Q) for tex in heat.latex), name
    assert ("Arreglo" in steps[-1].title) == (a is not None)
    assert (r"\eta_f" in joined) == (r.efficiency is not None)
    if fin.shape == "annular":
        assert "I_0(mr_1)" in joined


@pytest.mark.parametrize("tip", ["corrected", "convective", "adiabatic", "infinite"])
def test_annular_numbers_reproduce_the_heat(tip: str) -> None:
    """Con las funciones de Bessel a 5 cifras, como se muestran, q sale al 0,1 %."""
    ex = list(fn.FIN_EXAMPLES.values())[-1]
    r = fn.solve_fin(replace(ex.fin, tip=tip))
    i = r.inputs
    m, r1 = r.m_per_m, i.r_base_m
    rE = r1 + r.L_c_m

    def five(x: float) -> float:
        return float(f"{x:.5g}")

    I0a, I1a, K0a, K1a = (five(f(m * r1)) for f in (i0, i1, k0, k1))
    I0b, I1b, K0b, K1b = (five(f(m * rE)) for f in (i0, i1, k0, k1))
    beta = i.h_W_per_m2K / (m * i.k_W_per_mK)
    gamma = {
        "corrected": K1b / I1b,
        "adiabatic": K1b / I1b,
        "convective": (K1b - beta * K0b) / (I1b + beta * I0b),
        "infinite": 0.0,
    }[tip]
    phi = (K1a - five(gamma) * I1a) / (K0a + five(gamma) * I0a)
    assert r.M_W * phi == pytest.approx(r.Q_W, rel=1e-3)
    assert r.M_W == pytest.approx(i.k_W_per_mK * 2 * math.pi * r1 * i.thickness_m * m * i.theta_b)


# ---------------------------------------------------------------------
# Convección
# ---------------------------------------------------------------------

_D = {"Re": 2.5e4, "Pr": 0.71, "Ra": 3.0e7, "mu_ratio": 1.2, "D_over_L": 0.01, "heating": 1.0,
      "constant_T": 1.0}  # fmt: skip


@pytest.mark.parametrize("key", list(cv.CORRELATIONS))
def test_every_correlation_substitution_ends_in_its_value(key: str) -> None:
    d = dict(_D)
    if key.startswith("plate_mixed") or key == "plate_turbulent":
        d["Re"] = 8e5
    Nu = cv.CORRELATIONS[key].func(d)
    lines = correlation_latex(key, d, Nu)
    assert lines
    for tex in lines:
        assert tex.count("{") == tex.count("}"), tex
    if key != "tube_laminar":
        assert _ends_with(lines[-1], big(Nu)), key


def test_correlation_lines_use_the_right_branch() -> None:
    low, high = dict(_D, Ra=1e6), dict(_D, Ra=1e10)
    assert "0.54" in correlation_latex("horizontal_upper", low, 1.0)[0]
    assert "0.15" in correlation_latex("horizontal_upper", high, 1.0)[0]
    assert "0.59" in correlation_latex("vertical_mcadams", low, 1.0)[0]
    assert "0.10" in correlation_latex("vertical_mcadams", high, 1.0)[0]
    assert "Pr^{0.3}" in correlation_latex("dittus_boelter", dict(_D, heating=0.0), 1.0)[0]
    assert "4.36" in correlation_latex("tube_laminar", dict(_D, constant_T=0.0), 4.36)[0]


_CONV_CASES = list({**cv.EXTERNAL_EXAMPLES, **cv.NATURAL_EXAMPLES}.items())


@pytest.mark.parametrize(("name", "ex"), _CONV_CASES)
@pytest.mark.parametrize("system", SYSTEMS)
def test_convection_steps(name: str, ex: cv.ConvectionExample, system: str) -> None:
    inp = ex.inputs
    r = cv.solve_external(inp) if isinstance(inp, cv.ExternalFlowInputs) else cv.solve_natural(inp)
    steps = convection_steps(r, system)  # type: ignore[arg-type]
    joined = _check_steps(steps)
    assert latex_quantity(r.Q_W, "heat_rate", system) in joined  # type: ignore[arg-type]
    assert latex_quantity(r.h_W_per_m2K, "heat_transfer_coefficient", system) in joined  # type: ignore[arg-type]
    nusselt = next(st for st in steps if st.title.endswith("Número de Nusselt"))
    assert _ends_with(nusselt.latex[-1], big(r.Nu))
    if r.is_natural:
        assert "Gr" in joined and big(r.Ra) in joined  # type: ignore[arg-type]
    else:
        assert big(r.Re) in joined  # type: ignore[arg-type]


def test_external_procedure_details() -> None:
    mixed = cv.solve_external(cv.ExternalFlowInputs("Air", "plate", 10.0, 333.15, 293.15, 1.0))
    assert r"x_{\text{cr}}" in _check_steps(convection_steps(mixed, "Técnico"))
    sphere = cv.solve_external(cv.ExternalFlowInputs("Air", "sphere", 15.0, 348.15, 293.15, 0.025))
    joined = _check_steps(convection_steps(sphere, "SI"))
    assert r"\frac{\mu_\infty}{\mu_s}" in joined and r"T_\infty = " in joined
    cold = cv.solve_natural(
        cv.NaturalConvectionInputs("Air", "horizontal_plate_up", 278.15, 303.15, 0.5, 0.4)
    )
    joined = _check_steps(convection_steps(cold, "Técnico"))
    assert r"T_\infty - T_s" in joined and r"\frac{A}{P}" in joined


@pytest.mark.parametrize("name", list(cv.INTERNAL_EXAMPLES))
@pytest.mark.parametrize("system", SYSTEMS)
def test_internal_steps(name: str, system: str) -> None:
    r = cv.solve_internal(cv.INTERNAL_EXAMPLES[name].inputs)
    steps = internal_steps(r, system)  # type: ignore[arg-type]
    joined = _check_steps(steps)
    assert latex_quantity(r.T_out_K, "temperature", system) in joined  # type: ignore[arg-type]
    assert latex_quantity(r.Q_W, "heat_rate", system) in joined  # type: ignore[arg-type]
    assert latex_quantity(r.dp_Pa, "pressure_drop", system) in joined  # type: ignore[arg-type]
    assert ("NTU" in joined) == (r.inputs.condition == "constant_T")
    assert ("g_c" in joined) == (system == "Inglés")
    outlet = next(st for st in steps if st.title.endswith("Temperatura de salida"))
    assert ("1000" in outlet.text) == (system == "Técnico")
    assert ("3600" in outlet.text) == (system == "Inglés")
    if r.inputs.condition == "constant_T" and not r.heating:
        assert r"\ln(-" not in joined  # el cociente de ΔT_ml con los valores absolutos


@pytest.mark.parametrize("key", cv.TUBE_CORRELATIONS)
def test_internal_steps_with_each_tube_correlation(key: str) -> None:
    base = next(iter(cv.INTERNAL_EXAMPLES.values())).inputs
    r = cv.solve_internal(replace(base, correlation=key))
    steps = internal_steps(r, "Técnico")
    nusselt = next(st for st in steps if st.title.endswith("Número de Nusselt"))
    if key != "tube_laminar":
        assert _ends_with(nusselt.latex[-1], big(r.Nu))
    assert "fuera de su rango" in nusselt.text or cv.CORRELATIONS[key].in_range(
        cv.dimensionless_groups(r)
    )
