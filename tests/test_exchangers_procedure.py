"""Tests del procedimiento LaTeX de los intercambiadores (Fase 8.2).

Que cada paso esté bien formado en los tres sistemas de unidades y que los
números que se muestran sean los del resultado (el ancho y KaTeX se verifican
aparte, con un navegador).
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from core.heat_transfer import exchangers as hx
from core.heat_transfer.exchangers_procedure import (
    EPS_FORMULAS,
    NTU_FORMULAS,
    exchanger_steps,
    overall_u_steps,
)
from core.latex import latex_number, latex_quantity
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")

ARRANGEMENTS = [
    hx.Arrangement("parallel"),
    hx.Arrangement("counter"),
    hx.Arrangement("shell", 1),
    hx.Arrangement("shell", 2),
    hx.Arrangement("cross_unmixed"),
    hx.Arrangement("cross_mixed", mixed="hot"),
    hx.Arrangement("cross_mixed", mixed="cold"),
]


def _check_steps(steps: list[ProcedureStep]) -> str:
    """Pasos numerados, LaTeX balanceado y sin signos dobles; devuelve todo el LaTeX junto."""
    assert steps
    for k, st in enumerate(steps, 1):
        assert st.title.startswith(f"{k}. ")
        assert st.text
        for tex in st.latex:
            assert tex.count("{") == tex.count("}"), tex
            assert tex.count(r"\left") == tex.count(r"\right"), tex
            assert tex.count(r"\begin{aligned}") == tex.count(r"\end{aligned}"), tex
            assert not re.search(r"[-+]\s*-\s*\d", tex), tex  # «- -3» sin paréntesis
            assert "nan" not in tex and "inf" not in tex.replace(r"\infty", ""), tex
    return "\n".join(t for st in steps for t in st.latex)


def _eps(x: float) -> str:
    return latex_number(x, 4)


# ---------------------------------------------------------------------
# Fórmulas
# ---------------------------------------------------------------------


def test_formulas_have_one_equation_per_line() -> None:
    """Las de dos partes (una variable auxiliar) van en dos renglones: juntas no entraban."""
    for table in (EPS_FORMULAS, NTU_FORMULAS):
        for key, lines in table.items():
            assert isinstance(lines, tuple) and lines, key
            for tex in lines:
                assert r",\quad" not in tex, (key, tex)
                assert tex.count("{") == tex.count("}"), (key, tex)
    assert len(EPS_FORMULAS["shell"]) == 2
    assert len(NTU_FORMULAS["shell_n"]) == 2


# ---------------------------------------------------------------------
# Los ejemplos, en los tres sistemas
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(hx.RATING_EXAMPLES))
def test_rating_steps(name: str, system: str) -> None:
    r = hx.solve_rating(hx.RATING_EXAMPLES[name].inputs)
    steps = exchanger_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert [s.title.split(". ", 1)[1] for s in steps][-1] == "Exergía"
    assert _eps(r.NTU) in tex
    assert _eps(r.effectiveness) in tex
    assert latex_quantity(r.Q_W, "heat_rate", system) in tex  # type: ignore[arg-type]
    for sr in (r.hot, r.cold):
        if sr.phase_change:  # sale a T_sat: lo que se calcula es el caudal que cambia de fase
            assert latex_quantity(sr.m_dot_kg_s, "mass_flow", system) in tex  # type: ignore[arg-type]
        else:
            assert latex_quantity(sr.T_out_K, "temperature", system) in tex  # type: ignore[arg-type]
    assert latex_quantity(r.dT_lm_K, "temperature_difference", system) in tex  # type: ignore[arg-type]
    assert _eps(r.F) in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(hx.SIZING_EXAMPLES))
def test_sizing_steps(name: str, system: str) -> None:
    r = hx.solve_sizing(hx.SIZING_EXAMPLES[name].inputs)
    tex = _check_steps(exchanger_steps(r, system))  # type: ignore[arg-type]
    assert _eps(r.NTU) in tex
    assert latex_quantity(r.A_m2, "area", system) in tex  # type: ignore[arg-type]
    if r.tube_length_m is not None:
        assert latex_quantity(r.tube_length_m, "length", system) in tex  # type: ignore[arg-type]


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(hx.TEST_EXAMPLES))
def test_four_temperature_steps(name: str, system: str) -> None:
    r = hx.solve_four_temperatures(hx.TEST_EXAMPLES[name].inputs)
    tex = _check_steps(exchanger_steps(r, system))  # type: ignore[arg-type]
    assert latex_quantity(r.Q_W, "heat_rate", system) in tex  # type: ignore[arg-type]
    if r.inputs.U_W_per_m2K is None:  # el ensayo da U
        U = latex_quantity(r.U_W_per_m2K, "heat_transfer_coefficient", system)  # type: ignore[arg-type]
        assert U in tex
    else:  # con U dato salen Q̇ y los caudales
        for sr in (r.hot, r.cold):
            assert latex_quantity(sr.m_dot_kg_s, "mass_flow", system) in tex  # type: ignore[arg-type]


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(hx.U_EXAMPLES))
def test_overall_u_steps(name: str, system: str) -> None:
    r = hx.overall_u(hx.U_EXAMPLES[name].inputs)
    tex = _check_steps(overall_u_steps(r, system))  # type: ignore[arg-type]
    assert r"R_{\text{total}}" in tex
    for U in (r.U_i_W_per_m2K, r.U_o_W_per_m2K):
        assert latex_quantity(U, "heat_transfer_coefficient", system) in tex  # type: ignore[arg-type]
    if r.inputs.geometry != "tube":
        assert r"U &= \frac{UA}{A}" in tex and "U_i" not in tex


# ---------------------------------------------------------------------
# Cada tipo, con los números de su fórmula
# ---------------------------------------------------------------------


@pytest.mark.parametrize("arr", ARRANGEMENTS, ids=lambda a: a.name)
def test_every_arrangement_shows_its_formula(arr: hx.Arrangement) -> None:
    base = next(ex for name, ex in hx.RATING_EXAMPLES.items() if "11-9" in name)
    r = hx.solve_rating(replace(base.inputs, arrangement=arr))
    tex = _check_steps(exchanger_steps(r, "SI"))
    key = {
        "parallel": "parallel",
        "counter": "counter",
        "shell": "shell",
        "cross_unmixed": "cross_unmixed",
        "cross_mixed": "cross_cmin_mixed" if r.mixed_is_cmin else "cross_cmax_mixed",
    }[arr.kind]
    for line in EPS_FORMULAS[key]:
        assert line in tex
    if arr.kind == "shell" and arr.shell_passes > 1:
        assert EPS_FORMULAS["shell_n"][0] in tex
    # el ε que se muestra es el del resultado y F = 1 solo en paralelo y contracorriente
    assert _eps(r.effectiveness) in tex
    if arr.kind in ("parallel", "counter"):
        assert "F = 1" in tex
    else:
        assert r"P = \frac{t_2 - t_1}{T_1 - t_1}" in tex
        assert r"R = \frac{T_1 - T_2}{t_2 - t_1}" in tex


@pytest.mark.parametrize("arr", ARRANGEMENTS, ids=lambda a: a.name)
def test_every_arrangement_inverts_ntu(arr: hx.Arrangement) -> None:
    base = hx.SIZING_EXAMPLES[next(iter(hx.SIZING_EXAMPLES))].inputs
    r = hx.solve_sizing(replace(base, arrangement=arr))
    tex = _check_steps(exchanger_steps(r, "Técnico"))
    assert _eps(r.NTU) in tex
    if arr.kind in NTU_FORMULAS:
        assert NTU_FORMULAS[arr.kind][0] in tex
    if arr.kind == "cross_unmixed":
        # la serie no se invierte en forma cerrada: se dice qué ε y C_r se buscan
        assert rf"\varepsilon(NTU,\ {_eps(r.C_r)}) &= {_eps(r.effectiveness)}" in tex


def test_cross_unmixed_shows_the_series_and_the_approximation() -> None:
    name = "Radiador de un auto, flujo cruzado sin mezclar (basado en Cengel y Ghajar 11-6)"
    r = hx.solve_rating(hx.RATING_EXAMPLES[name].inputs)
    tex = _check_steps(exchanger_steps(r, "SI"))
    assert rf"\varepsilon({_eps(r.NTU)},\ {_eps(r.C_r)})" in tex
    assert r"\varepsilon_{\text{aprox}}" in tex
    assert r.eps_approx is not None
    assert _eps(r.eps_approx) in tex


def test_wide_capacity_moves_the_difference_to_its_own_row() -> None:
    """C_mín con ×10ⁿ: la resta (T_h − T_c) va en un renglón de continuación."""
    name = "Agua calentada con agua geotérmica, contracorriente (Cengel y Ghajar 11-8)"
    ex = hx.RATING_EXAMPLES.get(name) or next(iter(hx.RATING_EXAMPLES.values()))
    big = replace(
        ex.inputs,
        hot=replace(ex.inputs.hot, m_dot_kg_s=ex.inputs.hot.m_dot_kg_s * 100),
        cold=replace(ex.inputs.cold, m_dot_kg_s=ex.inputs.cold.m_dot_kg_s * 100),
        A_m2=ex.inputs.A_m2 * 100,
    )
    tex = _check_steps(exchanger_steps(hx.solve_rating(big), "SI"))
    qmax = next(t for t in tex.split("\n") if t.startswith(r"\begin{aligned}\dot{Q}_{\text{máx}}"))
    assert r"\times 10^{" in qmax
    assert r"\\ &\quad \cdot (" in qmax


# ---------------------------------------------------------------------
# Unidades
# ---------------------------------------------------------------------


def test_capacity_and_latent_factors_per_system() -> None:
    """ṁ·c_p y ṁ = Q̇/h_fg cierran con el factor del sistema (1000 en el Técnico, 3600 en el
    Inglés), y el texto lo dice."""
    cond = next(ex for name, ex in hx.RATING_EXAMPLES.items() if "ondensador" in name)
    r = hx.solve_rating(cond.inputs)
    by_system = {s: exchanger_steps(r, s) for s in SYSTEMS}  # type: ignore[arg-type]
    texts = {s: " ".join(st.text for st in steps) for s, steps in by_system.items()}
    assert "por 1000" in texts["Técnico"] and "3600" in texts["Inglés"]
    assert "por 1000" not in texts["SI"] and "3600" not in texts["SI"]
    lat = {s: "\n".join(t for st in steps for t in st.latex) for s, steps in by_system.items()}
    assert r"1000 \cdot " in lat["Técnico"] and r"3600 \cdot " in lat["Inglés"]


def test_exergy_step_closes() -> None:
    """X_dest = T₀·S_gen con los números que se muestran."""
    r = hx.solve_rating(next(iter(hx.RATING_EXAMPLES.values())).inputs)
    steps = exchanger_steps(r, "SI")
    exergy = steps[-1]
    tex = "\n".join(exergy.latex)
    assert latex_quantity(r.exergy.X_dest_W, "heat_rate", "SI") in tex
    assert latex_quantity(r.exergy.S_gen_W_per_K, "entropy_rate", "SI") in tex
