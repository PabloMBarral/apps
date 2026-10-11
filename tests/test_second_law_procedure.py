"""Tests del procedimiento LaTeX del segundo principio (Fase 10.2).

Que cada paso esté bien formado en los tres sistemas de unidades y que los números que se
muestran sean los del resultado (el ancho y KaTeX se verifican aparte, con un navegador:
5787 expresiones, máx. 305 px).
"""

from __future__ import annotations

import re

import pytest

from core.balances import second_law as sl
from core.balances import second_law_procedure as sp
from core.latex import latex_quantity
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")


def _check_steps(steps: list[ProcedureStep]) -> str:
    """Pasos numerados, LaTeX balanceado y sin signos dobles; devuelve todo el LaTeX junto."""
    assert steps
    for k, st in enumerate(steps, 1):
        assert st.title.startswith(f"{k}. ")
        assert st.text
        assert "None" not in st.text and not re.search(r"\be[+-]0", st.text), st.text
        for tex in st.latex:
            assert tex.count("{") == tex.count("}"), tex
            assert tex.count(r"\left") == tex.count(r"\right"), tex
            assert tex.count(r"\begin{aligned}") == tex.count(r"\end{aligned}"), tex
            assert not re.search(r"[-+]\s*-\s*\d", tex), tex  # «+ -3» sin paréntesis
            assert "nan" not in tex and "inf" not in tex, tex
    return "\n".join(t for st in steps for t in st.latex)


def _titles(steps: list[ProcedureStep]) -> list[str]:
    return [s.title.split(". ", 1)[1] for s in steps]


def _q(value: float, kind: str, system: str) -> str:
    return latex_quantity(value, kind, system)  # type: ignore[arg-type]


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(sl.MACHINE_EXAMPLES))
def test_machine_steps(name: str, system: str) -> None:
    r = sl.solve_machine(sl.MACHINE_EXAMPLES[name].inputs)
    steps = sp.machine_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[0] == "Temperaturas absolutas" and titles[-1] == "Veredicto"
    assert "Primer principio" in titles and "La máquina reversible" in titles
    E, S = ("power", "entropy_flow") if r.inputs.rates else ("energy", "entropy")
    assert _q(r.S_gen, S, system) in tex
    assert _q(r.W, E, system) in tex or "W" in r.inputs.data
    assert (r.violation or "") in steps[-1].text


def test_absolute_temperatures_are_shown() -> None:
    r = sl.solve_machine(sl.MACHINE_EXAMPLES["Máquina de Carnot (Çengel §6-10)"].inputs)
    tec = _check_steps(sp.machine_steps(r, "Técnico"))
    eng = _check_steps(sp.machine_steps(r, "Inglés"))
    assert "652 + 273.15" in tec and "925.15" in tec
    assert "459.67" in eng


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(sl.CLAUSIUS_EXAMPLES))
def test_clausius_steps(name: str, system: str) -> None:
    r = sl.solve_clausius(sl.CLAUSIUS_EXAMPLES[name].inputs)
    steps = sp.clausius_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == [
        "Temperaturas absolutas",
        "Cada fuente",
        "Desigualdad de Clausius",
        "Primer principio",
        "Exergía destruida",
        "Veredicto",
    ]
    assert _q(r.clausius_sum, "entropy", system) in tex
    assert _q(r.W, "energy", system) in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(sl.CARNOT_EXAMPLES))
def test_carnot_steps(name: str, system: str) -> None:
    r = sl.solve_carnot(sl.CARNOT_EXAMPLES[name].inputs)
    steps = sp.carnot_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == [
        "Temperaturas absolutas",
        "Estados 1 y 2 (a T_H)",
        "Estados 3 y 4 (a T_C)",
        "Calores (isotermas)",
        "Trabajo de cada proceso",
        "Trabajo neto y rendimiento",
    ]
    assert _q(r.Q_H, "energy", system) in tex
    assert _q(r.W_net, "energy", system) in tex
    letter = "u" if r.inputs.flow == "closed" else "h"
    assert f"{letter}_1" in tex and f"{letter}_4" in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(sl.HEAT_PATH_EXAMPLES))
def test_heat_path_steps(name: str, system: str) -> None:
    r = sl.solve_heat_path(sl.HEAT_PATH_EXAMPLES[name].inputs)
    steps = sp.heat_path_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps)[1] == "Entropía generada en cada tramo"
    S = "entropy_rate" if r.inputs.rates else "entropy"
    E = "heat_rate" if r.inputs.rates else "energy"
    assert _q(r.S_gen, S, system) in tex
    assert _q(r.X_dest, E, system) in tex


def test_wall_steps_have_one_line_per_zone() -> None:
    r = sl.solve_heat_path(sl.HEAT_PATH_EXAMPLES["Pared de ladrillo (Çengel §7-13)"].inputs)
    steps = sp.heat_path_steps(r, "Técnico")
    assert len(steps[1].latex) == 3
    assert "la pared" in steps[1].text
