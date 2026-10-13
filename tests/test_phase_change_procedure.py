"""Tests del procedimiento LaTeX de la ebullición y la condensación (Fase 11.2)."""

from __future__ import annotations

import dataclasses
import re

import pytest

from core.heat_transfer import boiling as bo
from core.heat_transfer import condensation as co
from core.heat_transfer.phase_change_procedure import boiling_steps, condensation_steps
from core.latex import latex_quantity
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")
C = 273.15
PAN = "Olla de acero inoxidable pulido (Çengel y Ghajar §10-2)"
VERTICAL = "Placa vertical en vapor a 1 atm (Çengel y Ghajar §10-5)"


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
            assert not re.search(r"[-+]\s*-\s*\d", tex), tex
            assert "nan" not in tex and "inf" not in tex, tex
            assert "Pack(" not in tex, tex
    return "\n".join(t for st in steps for t in st.latex)


def _titles(steps: list[ProcedureStep]) -> list[str]:
    return [s.title.split(". ", 1)[1] for s in steps]


def _q(value: float, kind: str, system: str) -> str:
    return latex_quantity(value, kind, system)  # type: ignore[arg-type]


def _pan(**kw: object) -> bo.PoolBoilingResult:
    inputs = dataclasses.replace(bo.BOILING_EXAMPLES[PAN].inputs, **kw)  # type: ignore[arg-type]
    return bo.solve_pool_boiling(inputs)


# ---------------------------------------------------------------------
# Ebullición
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(bo.BOILING_EXAMPLES))
def test_boiling_steps(name: str, system: str) -> None:
    r = bo.solve_pool_boiling(bo.BOILING_EXAMPLES[name].inputs)
    steps = boiling_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _q(r.q_max, "heat_flux", system) in tex
    assert _q(r.Q_W, "heat_rate", system) in tex
    assert _q(r.m_dot, "mass_flow", system) in tex
    assert _q(r.h, "heat_transfer_coefficient", system) in tex
    titles = _titles(steps)
    assert titles[0] == "Propiedades a la saturación"
    assert titles[-1] == "Calor y caudal evaporado"


def test_boiling_step_follows_the_regime() -> None:
    T_sat = _pan().sat.T_sat_K
    cases = {
        "Convección natural": _pan(T_s_K=T_sat + 1.0),
        "Ebullición nucleada (Rohsenow)": _pan(),
        "Transición (interpolada)": _pan(T_s_K=T_sat + 40.0),
        "Ebullición en película (Bromley)": _pan(T_s_K=T_sat + 300.0, emissivity=0.5),
        "Temperatura en el flujo crítico (Rohsenow)": _pan(mode="critical", T_s_K=None),
        "Temperatura de la superficie (Rohsenow)": _pan(mode="flux", T_s_K=None, q_W_per_m2=3e5),
    }
    for title, r in cases.items():
        steps = boiling_steps(r, "Técnico")
        assert title in _titles(steps), (title, _titles(steps))
        _check_steps(steps)
    burnout = bo.solve_pool_boiling(
        bo.BOILING_EXAMPLES["Una resistencia que se quema (ejemplo propio)"].inputs
    )
    film = boiling_steps(burnout, "SI")[-2]
    assert "salta a la película" in film.text
    assert r"\tfrac{3}{4}" in "".join(film.latex)


def test_boiling_shows_the_numbers_of_the_book() -> None:
    """La olla: q̇ = q̇₀·ψ³ con ψ y el resultado de Rohsenow a la vista."""
    r = _pan()
    tex = _check_steps(boiling_steps(r, "SI"))
    assert _q(r.q, "heat_flux", "SI") in tex
    assert "0.149" in tex  # C_cr de la placa grande
    nickel = bo.solve_pool_boiling(
        bo.BOILING_EXAMPLES[
            "Flujo crítico de un calentador niquelado (Çengel y Ghajar §10-2)"
        ].inputs
    )
    tex = _check_steps(boiling_steps(nickel, "Técnico"))
    assert "C_{cr} = 0.12" in tex
    assert _q(nickel.T_s_K, "temperature", "Técnico") in tex


@pytest.mark.parametrize("system", SYSTEMS)
def test_boiling_unit_factors_are_shown(system: str) -> None:
    steps = boiling_steps(_pan(), system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    text = " ".join(s.text for s in steps)
    if system == "SI":
        assert "1000" not in tex and "3600" not in tex and "g_c" not in tex
    elif system == "Técnico":
        assert r"1000 \cdot" in tex and "el 1000" in text
    else:
        assert r"3600 \cdot" in tex and "g_c" in tex and "32.174" in tex
        assert "g_c = 32,174" in text


def test_boiling_film_factor_inside_the_root() -> None:
    r = _pan(T_s_K=_pan().sat.T_sat_K + 300.0)
    for system, factor in (("Técnico", "1000"), ("Inglés", "3600")):
        film = boiling_steps(r, system)[-2]  # type: ignore[arg-type]
        assert any(rf"\left({factor} \cdot \frac" in t for t in film.latex)
        assert "raíz cuarta" in film.text


def test_boiling_plate_film_uses_the_capillary_length() -> None:
    r = _pan(T_s_K=_pan().sat.T_sat_K + 300.0)
    film = boiling_steps(r, "SI")[-2]
    tex = "".join(film.latex)
    assert "L_b" in tex and "Berenson" in film.text


def test_boiling_small_heaters_show_their_c_cr() -> None:
    small = _pan(heater="cylinder", D_m=0.002)
    assert small.critical_case == "cylinder_small"
    assert r"0.12\,L^{*-1/4}" in _check_steps(boiling_steps(small, "SI"))
    sphere = _pan(heater="sphere", D_m=0.004)
    assert sphere.critical_case == "sphere_small"
    assert r"0.227\,L^{*-1/2}" in _check_steps(boiling_steps(sphere, "SI"))


def test_boiling_pr_exponent_only_when_it_is_not_one() -> None:
    water = _check_steps(boiling_steps(_pan(), "SI"))
    assert "^{1}" not in water
    pentane = bo.solve_pool_boiling(
        bo.BOILING_EXAMPLES["n-Pentano sobre cobre pulido (ejemplo propio)"].inputs
    )
    assert "^{1.7}" in _check_steps(boiling_steps(pentane, "SI"))


# ---------------------------------------------------------------------
# Condensación
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(co.CONDENSATION_EXAMPLES))
def test_condensation_steps(name: str, system: str) -> None:
    r = co.solve_condensation(co.CONDENSATION_EXAMPLES[name].inputs)
    steps = condensation_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _q(r.h, "heat_transfer_coefficient", system) in tex
    assert _q(r.Q_W, "heat_rate", system) in tex
    assert _q(r.m_dot, "mass_flow", system) in tex
    assert _q(r.h_fg_star, "specific_enthalpy", system) in tex
    titles = _titles(steps)
    assert titles[:2] == ["Propiedades", "Calor latente corregido"]
    assert ("Espesor de la película" in titles) == (r.delta_m is not None)
    assert ("Condensación en gotas (Griffith)" in titles) == (r.h_dropwise is not None)


def test_condensation_step_follows_geometry_and_regime() -> None:
    base = co.CONDENSATION_EXAMPLES[VERTICAL].inputs
    cases = {
        "Película laminar (Nusselt)": dict(L_m=0.01, T_s_K=99.0 + C),
        "Película con ondas (Kutateladze)": {},
        "Película turbulenta (Labuntsov)": dict(L_m=10.0),
        "Tubo horizontal (Nusselt)": dict(geometry="horizontal_tube", D_m=0.03),
        "Banco de tubos": dict(geometry="tube_bank", D_m=0.03, rows=3, columns=4),
        "Esfera": dict(geometry="sphere", D_m=0.05),
        "Adentro del tubo (Chato)": dict(geometry="inside_tube", D_m=0.03, V_v_m_s=5.0),
    }
    for title, kw in cases.items():
        r = co.solve_condensation(dataclasses.replace(base, **kw))  # type: ignore[arg-type]
        steps = condensation_steps(r, "Inglés")
        assert title in _titles(steps), (title, _titles(steps))
        _check_steps(steps)


def test_condensation_shows_the_book_numbers() -> None:
    r = co.solve_condensation(co.CONDENSATION_EXAMPLES[VERTICAL].inputs)
    tex = _check_steps(condensation_steps(r, "SI"))
    assert r"(4.81 + 3.70\,X)^{0.820}" in tex
    assert "1286" in tex  # Re con ondas (el libro: 1287)
    inclined = co.solve_condensation(
        co.CONDENSATION_EXAMPLES["La misma placa inclinada 30° (Çengel y Ghajar §10-5)"].inputs
    )
    tex = _check_steps(condensation_steps(inclined, "SI"))
    assert r"(\cos 30^\circ)^{1/4}" in tex
    bank = co.solve_condensation(
        co.CONDENSATION_EXAMPLES["Banco de 3 × 4 tubos (Çengel y Ghajar §10-5)"].inputs
    )
    tex = _check_steps(condensation_steps(bank, "SI"))
    assert r"3^{-1/4}" in tex and "N_t" in tex


@pytest.mark.parametrize("system", SYSTEMS)
def test_condensation_unit_factors_are_shown(system: str) -> None:
    r = co.solve_condensation(co.CONDENSATION_EXAMPLES[VERTICAL].inputs)
    steps = condensation_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    if system == "SI":
        assert "1000" not in tex and "3600" not in tex
    else:
        factor = "1000" if system == "Técnico" else "3600"
        assert rf"{factor} \cdot" in tex
        assert any(factor in s.text for s in steps)


def test_condensation_superheat_and_chato_latent_heat() -> None:
    sh = co.solve_condensation(
        co.CONDENSATION_EXAMPLES[
            "Vapor sobrecalentado sobre un tubo vertical (ejemplo propio)"
        ].inputs
    )
    step = condensation_steps(sh, "SI")[1]
    assert r"c_{pv}\,(T_v - T_{\mathrm{sat}})" in "".join(step.latex)
    inside = co.solve_condensation(
        co.CONDENSATION_EXAMPLES["R-134a adentro de un tubo (ejemplo propio)"].inputs
    )
    tex = _check_steps(condensation_steps(inside, "SI"))
    assert r"\tfrac{3}{8}" in tex and r"\mathrm{Re}_v" in tex
