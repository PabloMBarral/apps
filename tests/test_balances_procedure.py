"""Tests del procedimiento LaTeX de los balances (Fase 10.1).

Que cada paso esté bien formado en los tres sistemas de unidades y que los números que se
muestran sean los del resultado (el ancho y KaTeX se verifican aparte, con un navegador:
15 110 expresiones, máx. 319 px).
"""

from __future__ import annotations

import re

import pytest

from core.balances import balances_procedure as bp
from core.balances import closed as cl
from core.balances import steady_flow as sf
from core.balances import transient as tr
from core.latex import latex_quantity
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")


def _check_steps(steps: list[ProcedureStep]) -> str:
    """Pasos numerados, LaTeX balanceado y sin signos dobles; devuelve todo el LaTeX junto."""
    assert steps
    for k, st in enumerate(steps, 1):
        assert st.title.startswith(f"{k}. ")
        assert st.text
        for bad in ("None", "nan", "e+0", "e-0"):
            assert bad not in st.text, (bad, st.text)
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


# ---------------------------------------------------------------------
# Sistema cerrado
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(cl.CLOSED_EXAMPLES))
def test_closed_steps_are_well_formed(name: str, system: str) -> None:
    r = cl.solve_closed(cl.CLOSED_EXAMPLES[name].inputs)
    steps = bp.closed_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[0] == "Estado 1" and any("rimer principio" in x for x in titles)
    assert titles[-1] in ("Segundo principio", "Comparación con la expansión reversible")
    assert _q(r.S_gen, "entropy", system) in tex
    assert _q(r.Q, "energy", system) in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(cl.EQUILIBRIUM_EXAMPLES))
def test_equilibrium_steps(name: str, system: str) -> None:
    r = cl.solve_equilibrium(cl.EQUILIBRIUM_EXAMPLES[name].inputs)
    steps = bp.equilibrium_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == ["Temperatura final", "Calor", "Entropía de cada parte",
                              "Segundo principio"]  # fmt: skip
    assert _q(r.T_final, "temperature", system) in tex
    assert _q(r.S_gen, "entropy", system) in tex


def test_closed_unit_factors_are_shown() -> None:
    """p·v lleva 100 en el Técnico y 0,18505 en el Inglés (W_b = m·p·Δv)."""
    r = cl.solve_closed(
        cl.CLOSED_EXAMPLES["Vapor de agua calentado a presión constante (Çengel §4-1)"].inputs
    )
    tex = {s: _check_steps(bp.closed_steps(r, s)) for s in SYSTEMS}  # type: ignore[arg-type]
    assert r"100 \cdot" in tex["Técnico"] and r"0.18505 \cdot" in tex["Inglés"]
    assert r"100 \cdot" not in tex["SI"]
    assert r"96.425\ \mathrm{Btu}" in tex["Inglés"]  # W_b del libro: 96,4 Btu
    assert "7.4863" in tex["Inglés"] and "8.3548" in tex["Inglés"]  # v₁ y v₂ de la tabla


def test_closed_impossible_process_shows_the_violation() -> None:
    r = cl.solve_closed(
        cl.ClosedInputs(cl.ideal_gas_substance("air"), "TP", 300.0, 1e5, mass_kg=1.0,
                        process="p_const", end="T", end_value=500.0)
    )  # fmt: skip
    steps = bp.closed_steps(r, "SI")
    assert r.violation is not None and r.violation in steps[-1].text


# ---------------------------------------------------------------------
# Flujo estacionario
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(sf.DEVICE_EXAMPLES))
def test_device_steps_are_well_formed(name: str, system: str) -> None:
    r = sf.solve_device(sf.DEVICE_EXAMPLES[name].inputs)
    steps = bp.device_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[0] == "Entrada" and titles[-1] == "Segundo principio"
    assert "Salida" in titles and "Primer principio" in titles
    assert _q(r.s_gen, "specific_entropy", system) in tex
    if r.w:
        assert _q(r.w, "specific_enthalpy", system) in tex
    if r.q and r.unknown == "q":
        assert _q(r.q, "specific_enthalpy", system) in tex


def test_kinetic_energy_factor() -> None:
    """ω²/2 va dividido por 1000 en el Técnico y por 25 037 en el Inglés (Çengel)."""
    assert bp.ke_factor("SI") == pytest.approx(1.0)
    assert bp.ke_factor("Técnico") == pytest.approx(1000.0)
    assert bp.ke_factor("Inglés") == pytest.approx(25037.0, abs=0.5)
    r = sf.solve_device(sf.DEVICE_EXAMPLES["Turbina de vapor de 5 MW (Çengel §5-4)"].inputs)
    tex = {s: _check_steps(bp.device_steps(r, s)) for s in SYSTEMS}  # type: ignore[arg-type]
    assert r"2 \cdot 1000" in tex["Técnico"] and r"2 \cdot 25\,037" in tex["Inglés"]
    assert "872.38" in tex["Técnico"]  # w del libro: 872,48 kJ/kg con sus tablas


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(sf.MIXING_EXAMPLES))
def test_mixing_steps(name: str, system: str) -> None:
    r = sf.solve_mixing(sf.MIXING_EXAMPLES[name].inputs)
    steps = bp.mixing_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == ["Estados", "Primer principio", "Segundo principio"]
    assert _q(r.S_gen_dot, "entropy_flow", system) in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(sf.EXCHANGER_EXAMPLES))
def test_exchanger_steps(name: str, system: str) -> None:
    r = sf.solve_exchanger(sf.EXCHANGER_EXAMPLES[name].inputs)
    steps = bp.exchanger_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == ["Estados", "Primer principio", "Segundo principio"]
    assert _q(r.Q_transfer, "power", system) in tex
    assert _q(r.S_gen_dot, "entropy_flow", system) in tex


# ---------------------------------------------------------------------
# Régimen transitorio
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(tr.CHARGING_EXAMPLES))
def test_charging_steps(name: str, system: str) -> None:
    r = tr.solve_charging(tr.CHARGING_EXAMPLES[name].inputs)
    steps = bp.charging_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == ["Línea y estado inicial", "Estado final", "Primer principio",
                              "Segundo principio"]  # fmt: skip
    assert _q(r.m2, "mass", system) in tex
    assert _q(r.S_gen, "entropy", system) in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(tr.DISCHARGING_EXAMPLES))
def test_discharging_steps(name: str, system: str) -> None:
    r = tr.solve_discharging(tr.DISCHARGING_EXAMPLES[name].inputs)
    steps = bp.discharging_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == ["Estado inicial", "Estado final", "Entalpía que sale",
                              "Primer principio", "Segundo principio"]  # fmt: skip
    assert _q(r.H_out, "energy", system) in tex
    assert str(tr.INTEGRATION_POINTS) in steps[2].text
