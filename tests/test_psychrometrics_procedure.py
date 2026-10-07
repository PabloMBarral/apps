"""Tests del procedimiento de la psicrometría — Fase 4.

Que los pasos sigan el vademecum §14, que los números que se muestran sean los
del resultado y que el LaTeX esté bien armado en los tres sistemas de unidades
(el ancho y la validez con KaTeX se miden aparte, en un navegador).
"""

from __future__ import annotations

import re

import pytest

from core.hvac import HVAC_EXAMPLES, TOWER_EXAMPLES, solve_cooling_tower, solve_hvac
from core.latex import latex_number
from core.psychrometrics import (
    P_SEA_LEVEL_PA,
    PAIRS,
    STATE_EXAMPLES,
    DeadState,
    moist_air_state,
    state_from_T_phi,
)
from core.psychrometrics_procedure import cooling_tower_steps, hvac_steps, moist_air_steps

C = 273.15
P = P_SEA_LEVEL_PA
SYSTEMS = ["SI", "Técnico", "Inglés"]


def _check_latex(tex: str) -> None:
    assert tex.count("{") == tex.count("}"), tex
    assert "nan" not in tex and "inf" not in tex, tex
    assert not re.search(r"[-+]\s*-\d", tex), tex  # negativos entre paréntesis
    assert r"\begin{aligned}" not in tex or tex.endswith(r"\end{aligned}"), tex


def _state_steps(name: str, system: str):  # type: ignore[no-untyped-def]
    ex = STATE_EXAMPLES[name]
    s = moist_air_state(ex.pressure_Pa, ex.pair, ex.first, ex.second)
    return s, moist_air_steps(
        s,
        system,  # type: ignore[arg-type]
        pair=ex.pair,
        given=(ex.first, ex.second),
        dead=DeadState(ex.T0_K, ex.phi0, ex.pressure_Pa),
        altitude_m=ex.altitude_m,
        room_volume_m3=ex.room_volume_m3,
    )


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(STATE_EXAMPLES))
def test_state_procedure_is_well_formed(name: str, system: str) -> None:
    _, steps = _state_steps(name, system)
    titles = [s.title for s in steps]
    assert titles[0].startswith("Presión total")
    assert titles[1] == "Humedad absoluta y relativa con los datos"
    assert "Entropía" in titles and "Exergía de flujo" in titles
    for step in steps:
        assert step.title and step.text
        for tex in step.latex:
            _check_latex(tex)


def test_state_steps_follow_the_vademecum_order() -> None:
    _, steps = _state_steps("Cengel 14-4: la lectura de la carta", "Técnico")
    assert [s.title for s in steps] == [
        "Presión total",
        "Humedad absoluta y relativa con los datos",
        "Presión del aire seco y grado de saturación",
        "Entalpía",
        "Volumen específico, densidad, constante y calor específico",
        "Temperatura de punto de rocío",
        "Temperatura de bulbo húmedo (saturación adiabática)",
        "Entropía",
        "Exergía de flujo",
    ]


def test_given_quantities_are_not_recomputed() -> None:
    s = moist_air_state(P, ("T", "T_wb"), 25 + C, 15 + C)
    titles = [
        st.title for st in moist_air_steps(s, "Técnico", pair=("T", "T_wb"), given=(25 + C, 15 + C))
    ]
    assert not any("bulbo húmedo" in t for t in titles)
    assert "Temperatura de punto de rocío" in titles
    s = moist_air_state(P, ("T", "h"), 25 + C, 50e3)
    titles = [
        st.title for st in moist_air_steps(s, "Técnico", pair=("T", "h"), given=(25 + C, 50e3))
    ]
    assert "Entalpía" not in titles


@pytest.mark.parametrize("pair", PAIRS)
def test_every_pair_has_its_derivation(pair: tuple[str, str]) -> None:
    ref = state_from_T_phi(P, 30 + C, 0.5)
    v = {
        "T": ref.T_K,
        "phi": ref.phi,
        "T_wb": ref.T_wb_K,
        "T_dp": ref.T_dp_K,
        "omega": ref.omega,
        "h": ref.h_J_per_kg,
    }
    s = moist_air_state(P, pair, v[pair[0]], v[pair[1]])
    for system in SYSTEMS:
        steps = moist_air_steps(s, system, pair=pair, given=(v[pair[0]], v[pair[1]]))  # type: ignore[arg-type]
        derivation = steps[1]
        assert derivation.text
        # el ω que se muestra es el del resultado
        assert any(latex_number(s.omega, 4) in tex for tex in derivation.latex), (pair, system)
        for step in steps:
            for tex in step.latex:
                _check_latex(tex)


def test_wet_bulb_check_shows_both_sides_equal() -> None:
    s = state_from_T_phi(P, 25 + C, 0.75)
    steps = moist_air_steps(s, "Técnico", pair=("T", "phi"), given=(25 + C, 0.75))
    wb = next(st for st in steps if st.title.startswith("Temperatura de bulbo húmedo"))
    joined = " ".join(wb.latex)
    assert joined.count("63.326") >= 2  # h* del balance y h* del aire saturado a T_bh


def test_ice_wet_bulb_uses_the_ice_enthalpy() -> None:
    s = state_from_T_phi(P, -10 + C, 0.6)
    steps = moist_air_steps(s, "Técnico", pair=("T", "phi"), given=(-10 + C, 0.6))
    wb = next(st for st in steps if st.title.startswith("Temperatura de bulbo húmedo"))
    assert "hielo" in wb.text
    assert any("-333.4" in tex for tex in wb.latex)
    dp = next(st for st in steps if st.title.startswith("Temperatura de punto de rocío"))
    assert "escarcha" in dp.title


def test_altitude_and_room_steps() -> None:
    _, steps = _state_steps("La Quiaca, a 3440 m de altura", "Técnico")
    assert steps[0].title == "Presión total a la altura del lugar"
    assert any("3440" in tex for tex in steps[0].latex)
    _, steps = _state_steps("Cengel 14-1: el vapor de agua de una habitación", "Técnico")
    room = steps[-1]
    assert room.title == "Masas de aire seco y de vapor en el recinto"
    assert any("85.56" in tex for tex in room.latex)


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(HVAC_EXAMPLES))
def test_hvac_procedure_is_well_formed(name: str, system: str) -> None:
    r = solve_hvac(HVAC_EXAMPLES[name])
    steps = hvac_steps(r, system)  # type: ignore[arg-type]
    titles = [s.title for s in steps]
    assert titles[0] == "Estado 1 y caudal de aire seco"
    assert titles[1] == "Ambiente para la exergía"
    assert titles[-1] == "Totales"
    process_titles = titles[2:-1]
    assert len(process_titles) == len(r.processes)
    for proc, title in zip(r.processes, process_titles, strict=True):
        assert title.startswith(f"Proceso {proc.index}: {proc.name.lower()}")
        assert f"→ {proc.outlet.number})" in title
    for step in steps:
        for tex in step.latex:
            _check_latex(tex)
        if step.title.startswith("Proceso"):
            assert any(r"\dot X_{\mathrm{dest}}" in tex for tex in step.latex)


def test_coil_step_matches_cengel_14_6() -> None:
    r = solve_hvac(HVAC_EXAMPLES["Cengel 14-6: enfriamiento y deshumidificación"])
    step = hvac_steps(r, "Técnico")[2]
    joined = " ".join(step.latex)
    assert "h_f(14" in joined  # condensado a 14 °C (tabla A-4)
    assert latex_number(r.processes[0].Q_W / 1e3, 5) in joined


def test_mixing_step_shows_the_weighted_averages() -> None:
    r = solve_hvac(HVAC_EXAMPLES["Cengel 14-8: mezcla con aire exterior"])
    step = hvac_steps(r, "Técnico")[2]
    assert step.title == "Proceso 1: mezcla adiabática (1 + 2 → 3)"
    joined = " ".join(step.latex)
    assert latex_number(r.outlet.state.omega, 4) in joined
    assert "recta" in step.text


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(TOWER_EXAMPLES))
def test_tower_procedure(name: str, system: str) -> None:
    r = solve_cooling_tower(TOWER_EXAMPLES[name])
    steps = cooling_tower_steps(r, system)  # type: ignore[arg-type]
    assert [s.title for s in steps] == [
        "Aire que entra (1) y que sale (2)",
        "Agua caliente (3) y enfriada (4)",
        "Balances: caudal de aire y agua de reposición",
        "Rango, aproximación y efectividad",
        "Exergía destruida",
    ]
    for step in steps:
        for tex in step.latex:
            _check_latex(tex)
    if system == "Técnico":
        assert any(latex_number(r.m_dry_air_kg_s, 5) in tex for tex in steps[2].latex)
