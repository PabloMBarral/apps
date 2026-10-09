"""Tests del procedimiento LaTeX de la radiación (Fase 8.2).

Que cada paso esté bien formado en los tres sistemas de unidades y que los
números que se muestran sean los del resultado (el ancho y KaTeX se verifican
aparte, con un navegador).
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest

from core.heat_transfer import radiation as rd
from core.heat_transfer.radiation_procedure import (
    blackbody_steps,
    constants_latex,
    enclosure_steps,
    surface_balance_steps,
    thermocouple_steps,
    two_surface_steps,
    view_factor_steps,
)
from core.latex import latex_number, latex_quantity
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")


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


def _q(value_si: float, kind: str, system: str) -> str:
    return latex_quantity(value_si, kind, system)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------


def test_constants_in_each_system() -> None:
    """σ = 5,67·10⁻⁸ W/(m²·K⁴) = 0,1712·10⁻⁸ Btu/(h·ft²·°R⁴); C₃ = 2897,8 μm·K = 5216 μm·°R."""
    si, en = constants_latex("SI"), constants_latex("Inglés")
    assert si[0].startswith(r"\sigma = 5.67\times 10^{-8}")
    assert en[0].startswith(r"\sigma = 1.712\times 10^{-9}")
    assert "2897.8" in si[3] and "5216" in en[3]
    assert constants_latex("Técnico") == si  # W y K también en el Técnico


# ---------------------------------------------------------------------
# Los ejemplos, en los tres sistemas
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rd.BLACKBODY_EXAMPLES))
def test_blackbody_steps(name: str, system: str) -> None:
    r = rd.solve_blackbody(rd.BLACKBODY_EXAMPLES[name].inputs)
    tex = _check_steps(blackbody_steps(r, system))  # type: ignore[arg-type]
    assert _q(r.E_b_W_per_m2, "heat_flux", system) in tex
    assert _q(r.lambda_max_m, "wavelength", system) in tex
    assert _q(r.E_blambda_max_W_per_m3, "spectral_emissive_power", system) in tex
    if r.band_fraction is not None:
        assert latex_number(r.band_fraction, 5) in tex
    if r.rows:
        assert latex_number(r.emissivity, 4) in tex
        assert tex.count(r"\varepsilon_") >= len(r.rows)  # un ε_k·Δf_k por banda
    if r.source_rows:
        assert latex_number(r.absorptivity or 0.0, 4) in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rd.VIEW_FACTOR_EXAMPLES))
def test_view_factor_steps(name: str, system: str) -> None:
    r = rd.solve_view_factor(rd.VIEW_FACTOR_EXAMPLES[name].inputs)
    tex = _check_steps(view_factor_steps(r, system))  # type: ignore[arg-type]
    assert latex_number(r.F_ij, 5) in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rd.TWO_SURFACE_EXAMPLES))
def test_two_surface_steps(name: str, system: str) -> None:
    r = rd.solve_two_surface(rd.TWO_SURFACE_EXAMPLES[name].inputs)
    tex = _check_steps(two_surface_steps(r, system))  # type: ignore[arg-type]
    assert _q(abs(r.Q_W), "heat_rate", system) in tex or _q(r.Q_W, "heat_rate", system) in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rd.ENCLOSURE_EXAMPLES))
def test_enclosure_steps(name: str, system: str) -> None:
    r = rd.solve_enclosure(rd.ENCLOSURE_EXAMPLES[name].inputs)
    steps = enclosure_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    for J in r.J_W_per_m2:
        assert _q(J, "heat_flux", system) in tex
    for Q in r.Q_W:
        assert _q(Q, "heat_rate", system) in tex
    for s, T in zip(r.inputs.surfaces, r.T_K, strict=True):
        if s.T_K is None:  # la temperatura que sale (rerradiante o con el calor dado)
            assert _q(T, "absolute_temperature", system) in tex
    # una ecuación por superficie
    radiosity = next(st for st in steps if "radiosidades" in st.title)
    n = len(r.inputs.surfaces)
    assert sum(1 for t in radiosity.latex if not t.startswith("J_") or "E_{b" in t) >= n


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rd.SURFACE_EXAMPLES))
def test_surface_balance_steps(name: str, system: str) -> None:
    r = rd.solve_surface_balance(rd.SURFACE_EXAMPLES[name].inputs)
    tex = _check_steps(surface_balance_steps(r, system))  # type: ignore[arg-type]
    assert _q(r.Q_conv_W, "heat_rate", system) in tex
    assert _q(r.Q_rad_W, "heat_rate", system) in tex
    assert _q(r.h_rad_W_per_m2K, "heat_transfer_coefficient", system) in tex
    if r.Q_solar_W > 0.0:
        assert _q(r.Q_solar_W, "heat_rate", system) in tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rd.THERMOCOUPLE_EXAMPLES))
def test_thermocouple_steps(name: str, system: str) -> None:
    r = rd.solve_thermocouple(rd.THERMOCOUPLE_EXAMPLES[name].inputs)
    tex = _check_steps(thermocouple_steps(r, system))  # type: ignore[arg-type]
    assert _q(r.T_gas_K, "temperature", system) in tex
    assert _q(r.q_rad_W_per_m2, "heat_flux", system) in tex


# ---------------------------------------------------------------------
# Detalles
# ---------------------------------------------------------------------


def test_absolute_temperature_conversion_only_outside_si() -> None:
    """Las fórmulas de la radiación van con T absoluta: en el Técnico y el Inglés se convierte."""
    r = rd.solve_surface_balance(next(iter(rd.SURFACE_EXAMPLES.values())).inputs)
    si = _check_steps(surface_balance_steps(r, "SI"))
    tec = _check_steps(surface_balance_steps(r, "Técnico"))
    eng = _check_steps(surface_balance_steps(r, "Inglés"))
    assert "+ 273.15" not in si and "+ 459.67" not in si
    assert "+ 273.15" in tec
    assert "+ 459.67" in eng


def test_planck_at_the_maximum() -> None:
    """E_bλ,máx = C₁/(λ_máx⁵·(e^{C₂/C₃} − 1)): C₂/C₃ = 4,965 en cualquier sistema."""
    r = rd.solve_blackbody(next(iter(rd.BLACKBODY_EXAMPLES.values())).inputs)
    for system in SYSTEMS:
        tex = _check_steps(blackbody_steps(r, system))  # type: ignore[arg-type]
        assert "e^{4.965}" in tex
    lam_um = r.lambda_max_m * 1e6
    E = rd.C1 * 1e24 / (lam_um**5 * (math.exp(rd.C2 / rd.C3) - 1.0))  # W/(m²·μm)
    assert E == pytest.approx(r.E_blambda_max_W_per_m3 * 1e-6, rel=1e-9)


def test_parallel_plates_numbers_reproduce_the_factor() -> None:
    """F_ij = (a − b)/(2·W_i) con los a y b que se muestran."""
    r = rd.solve_view_factor(rd.ViewFactorInputs("parallel_plates_2d", 0.35, 0.8, 1.7))
    tex = _check_steps(view_factor_steps(r, "SI"))
    a = float(re.search(r"a &= \\sqrt\{\(W_i \+ W_j\)\^2 \+ 4\} .*?&= ([\d.]+)", tex).group(1))
    b = float(re.search(r"b &= \\sqrt\{\(W_j - W_i\)\^2 \+ 4\} .*?&= ([\d.]+)", tex).group(1))
    Wi = 0.35 / 1.7
    assert (a - b) / (2 * Wi) == pytest.approx(r.F_ij, rel=2e-3)


def test_enclosure_with_wide_radiosities_shows_the_differences() -> None:
    """Con J ×10ⁿ (la cavidad de Incropera 13.2 en SI), cada término ya lleva J_i − J_j restado
    para que el renglón entre en un celular; con números cortos se ven las dos J."""
    name = "Cavidad cilíndrica abierta: la potencia del horno (Incropera 13.2)"
    r = rd.solve_enclosure(rd.ENCLOSURE_EXAMPLES[name].inputs)
    si = _check_steps(enclosure_steps(r, "SI"))
    q_lines = [t for t in si.split("\n") if t.startswith(r"\begin{aligned}\dot{Q}_")]
    assert q_lines
    for t in q_lines:
        assert r"\cdot [" in t
        assert not re.search(r"\(\d[\d.]*\\times 10\^\{\d+\} - ", t)  # ninguna resta con ×10ⁿ
    small = rd.solve_enclosure(next(iter(rd.ENCLOSURE_EXAMPLES.values())).inputs)
    tec = _check_steps(enclosure_steps(replace(small), "Técnico"))
    assert re.search(r"\\,\(\d[\d.]* - ", tec)


def test_shield_steps_show_the_shield_temperature() -> None:
    name = "Las mismas con una pantalla de aluminio (Cengel y Ghajar)"
    r = rd.solve_two_surface(rd.TWO_SURFACE_EXAMPLES[name].inputs)
    assert r.shield_T_K
    for system in SYSTEMS:
        tex = _check_steps(two_surface_steps(r, system))  # type: ignore[arg-type]
        for T in r.shield_T_K:
            assert _q(T, "absolute_temperature", system) in tex
