"""Tests del procedimiento LaTeX de los gases reales (Fase 9.2).

Que cada paso esté bien formado en los tres sistemas de unidades y que los números
que se muestran sean los del resultado (el ancho y KaTeX se verifican aparte, con un
navegador: 7021 expresiones, máx. 318 px).
"""

from __future__ import annotations

import re

import pytest

from core.gases import cubic as cb
from core.gases import real as rg
from core.gases import relations as rl
from core.gases.real_procedure import (
    clapeyron_steps,
    compressibility_steps,
    cubic_steps,
    joule_thomson_steps,
    relations_steps,
)
from core.latex import latex_number, latex_quantity
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")
C = 273.15


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
            assert "nan" not in tex and "inf" not in tex.replace(r"\infty", ""), tex
            assert not re.search(r"_(\{[^{}]*\}|\w)_", tex), tex  # doble subíndice
            assert "^2^" not in tex and "}^2}^" not in tex, tex
    return "\n".join(t for st in steps for t in st.latex)


def _titles(steps: list[ProcedureStep]) -> list[str]:
    return [s.title.split(". ", 1)[1] for s in steps]


def _q(value: float, kind: str, system: str) -> str:
    return latex_quantity(value, kind, system)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# El factor de compresibilidad
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rg.REAL_EXAMPLES))
def test_compressibility_steps_are_well_formed(name: str, system: str) -> None:
    r = rg.compressibility(rg.REAL_EXAMPLES[name].inputs)
    steps = compressibility_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == [
        "El fluido y su punto crítico",
        "Las variables reducidas",
        "Como gas ideal (Z = 1)",
        "La carta generalizada: Z⁰ de Lee y Kesler",
        "Lee–Kesler con el factor acéntrico",
        "Van der Waals",
        "Peng–Robinson",
        "El fluido real (CoolProp)",
    ]
    # la incógnita de cada modelo que tiene solución aparece con su unidad
    kind = {"v": "specific_volume", "p": "pressure", "T": "absolute_temperature"}[r.unknown]
    for m in r.models:
        if m.model == "coolprop" or m.Z is None:
            continue
        value = {"v": m.v_m3_per_kg, "p": m.p_Pa, "T": m.T_K}[r.unknown]
        assert _q(value, kind, system) in tex, (m.model, value)  # type: ignore[arg-type]
    assert latex_number(r.Z) in tex  # el Z real


def test_cengel_refrigerant_134a_shows_the_three_roots() -> None:
    r = rg.compressibility(rg.REAL_EXAMPLES["R-134a a 1 MPa y 50 °C (Çengel §3-7)"].inputs)
    steps = compressibility_steps(r, "Técnico")
    vdw = next(s for s in steps if s.title.endswith("Van der Waals"))
    tex = "\n".join(vdw.latex)
    for z in r.model("vdw").roots:
        assert latex_number(z) in tex
    assert r"Z_{\text{vap}}" in tex and r"Z_{\text{líq}}" in tex
    assert "tres raíces" in vdw.text and "vapor" in vdw.text


def test_nitrogen_with_temperature_and_volume_uses_the_reduced_form() -> None:
    """Çengel §3-8 con (T, v): Van der Waals en forma reducida (vademecum §8.4), con
    v_R = 8/3·v′_R."""
    r = rg.compressibility(
        rg.REAL_EXAMPLES["Nitrógeno a 175 K y 0,00375 m³/kg (Çengel §3-8)"].inputs
    )
    steps = compressibility_steps(r, "Técnico")
    tex = _check_steps(steps)
    assert r"\frac{8}{3}\,v'_R" in tex
    assert latex_number(8 / 3 * r.v_R_pseudo) in tex
    assert r"\frac{8\,T_R}{3\,v_R - 1} - \frac{3}{v_R^2}" in tex
    lk0 = next(s for s in steps if "Z⁰" in s.title)
    assert "v′_R" in lk0.text  # el v′_R de Nelson y Obert es dato


def test_pressure_volume_factor_in_the_ideal_step() -> None:
    r = rg.compressibility(
        rg.REAL_EXAMPLES["Aire de un tubo a 200 bar: la temperatura con p y v"].inputs
    )
    for system, factor in (("SI", None), ("Técnico", "100"), ("Inglés", "0.18505")):
        steps = compressibility_steps(r, system)  # type: ignore[arg-type]
        ideal = next(s for s in steps if s.title.endswith("(Z = 1)"))
        tex = "\n".join(ideal.latex)
        if factor:
            assert rf"{factor} \cdot" in tex
            assert "p·v" in ideal.text
        else:
            assert r"\cdot" in tex


def test_model_without_solution_explains_why() -> None:
    r = rg.compressibility(rg.CompressibilityInputs("Water", "Tv", 20.0 + C, 0.0010))
    steps = compressibility_steps(r, "SI")
    vdw = next(s for s in steps if s.title.endswith("Van der Waals"))
    assert "covolumen" in vdw.text and not vdw.latex


def test_lee_kesler_shows_both_fluids() -> None:
    r = rg.compressibility(rg.REAL_EXAMPLES["Metano de un gasoducto a 70 bar y 15 °C"].inputs)
    steps = compressibility_steps(r, "SI")
    lk = next(s for s in steps if "factor acéntrico" in s.title)
    tex = "\n".join(lk.latex)
    lee = rg.lee_kesler(r.T_R, r.p_R, r.fluid.omega, r.phase)
    assert latex_number(lee.Z0) in tex and latex_number(lee.Z_ref) in tex
    assert latex_number(lee.Z) in tex
    assert "n-octano" in lk.text


# ---------------------------------------------------------------------
# Las cúbicas
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(cb.CUBIC_EXAMPLES))
def test_cubic_steps_are_well_formed(name: str, system: str) -> None:
    r = cb.solve_cubic(cb.CUBIC_EXAMPLES[name].inputs)
    steps = cubic_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps)[2] == "Van der Waals: a, b y su punto crítico"
    for sol in (r.vdw, r.pr):
        for root in sol.roots:
            assert latex_number(root.Z) in tex
        assert _q(sol.b, "specific_volume", system) in tex
    assert latex_number(r.omega_vdw, 4) in tex


def test_maxwell_equal_areas_are_shown_equal() -> None:
    r = cb.solve_cubic(cb.CUBIC_EXAMPLES["R-134a a 1 MPa y 50 °C: tres raíces"].inputs)
    steps = cubic_steps(r, "Técnico")
    step = next(s for s in steps if "Maxwell" in s.title)
    sat = r.vdw.saturation
    assert sat is not None
    v3b = 3 * r.vdw.b
    rect = sat.p_sat_Pa / r.fluid.p_cr * (sat.v_g - sat.v_f) / v3b
    tex = "\n".join(step.latex)
    assert tex.count(latex_number(rect)) >= 2  # la integral y el rectángulo dan lo mismo
    assert _q(sat.p_sat_Pa, "pressure", "Técnico") in tex


def test_supercritical_cubic_has_no_saturation_steps() -> None:
    r = cb.solve_cubic(cb.CUBIC_EXAMPLES["Nitrógeno a 175 K y 100 bar (Çengel §3-8)"].inputs)
    steps = cubic_steps(r, "SI")
    _check_steps(steps)
    maxwell = next(s for s in steps if "Maxwell" in s.title)
    assert "Sobre la temperatura crítica" in maxwell.text and not maxwell.latex


def test_negative_kappa_has_no_double_sign() -> None:
    """El helio tiene ω < 0 y κ < 0: α = [1 − |κ|(1 − √T_R)]²."""
    fl = rg.real_fluid("Helium")
    r = cb.solve_cubic(cb.CubicInputs("Helium", 2.0 * fl.p_cr, 1.5 * fl.T_cr))
    tex = _check_steps(cubic_steps(r, "SI"))
    assert rf"\left[1 - {latex_number(abs(r.pr.kappa or 0))}" in tex


# ---------------------------------------------------------------------
# Funciones características y relaciones de Maxwell
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rl.RELATIONS_EXAMPLES))
def test_relations_steps_are_well_formed(name: str, system: str) -> None:
    r = rl.relations(rl.RELATIONS_EXAMPLES[name].inputs)
    steps = relations_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps)[0] == "Los potenciales"
    assert sum("La relación de Maxwell" in t for t in _titles(steps)) == 4
    targets = {
        "u": "temperature_per_volume",
        "h": "temperature_per_pressure",
        "f": "pressure_per_temperature",
        "g": "volume_per_temperature",
    }
    for m in r.maxwell:  # la derivada exacta de cada relación
        assert _q(m.left.exact, targets[m.potential], system) in tex
    assert _q(r.mayer, "specific_heat", system) in tex
    assert _q(r.mu_JT_formula, "temperature_per_pressure", system) in tex


def test_steam_relation_shows_the_table_numbers() -> None:
    """Çengel §12-2 en el Técnico: (7,3804 − 7,7100)/(2·1 bar) = −0,1648 kJ/(kg·K·bar)."""
    r = rl.relations(
        rl.RELATIONS_EXAMPLES["Vapor de agua a 250 °C y 300 kPa (Çengel §12-2)"].inputs
    )
    steps = relations_steps(r, "Técnico")
    g = next(s for s in steps if "g(T, p)" in s.title)
    tex = "\n".join(g.latex)
    assert "7.3804 - 7.71" in tex
    assert r"2 \cdot 1" in tex
    assert latex_number(r.relation("g").left.finite * 1e5 / 1e3) in tex  # en kJ/(kg·K·bar)
    assert "0,01 bar·m³" in g.text  # el pase a m³/(kg·K)


def test_si_shows_the_relation_in_its_own_units() -> None:
    """En el SI J/(kg·K·Pa) = m³/(kg·K): sin el renglón con la fracción de unidades."""
    r = rl.relations(rl.RELATIONS_EXAMPLES["R-134a a 1 MPa y 50 °C"].inputs)
    tex = _check_steps(relations_steps(r, "SI"))
    assert r"\frac{\mathrm{J/(kg\cdot K)}}" not in tex


# ---------------------------------------------------------------------
# Clapeyron
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rl.CLAPEYRON_EXAMPLES))
def test_clapeyron_steps_are_well_formed(name: str, system: str) -> None:
    r = rl.clapeyron(rl.CLAPEYRON_EXAMPLES[name].inputs)
    steps = clapeyron_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == [
        "La saturación a T",
        "La pendiente de la curva de saturación",
        "La ecuación de Clapeyron",
        "Clausius–Clapeyron",
        "Extrapolar la presión de saturación",
    ]
    assert _q(r.h_fg_clapeyron, "specific_enthalpy", system) in tex
    assert _q(r.h_fg_cc, "specific_enthalpy", system) in tex
    assert _q(r.dpdT_finite, "pressure_per_temperature", system) in tex
    assert r.p_sat2_cc is not None and _q(r.p_sat2_cc, "pressure", system) in tex


def test_clapeyron_without_a_second_temperature() -> None:
    r = rl.clapeyron(rl.ClapeyronInputs("Water", 100.0 + C, 1.0))
    steps = clapeyron_steps(r, "SI")
    assert len(steps) == 4


# ---------------------------------------------------------------------
# Joule–Thomson
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(rl.JT_EXAMPLES))
def test_joule_thomson_steps_are_well_formed(name: str, system: str) -> None:
    r = rl.joule_thomson(rl.JT_EXAMPLES[name].inputs)
    steps = joule_thomson_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == [
        "El estado",
        "El coeficiente de Joule–Thomson",
        "La temperatura de inversión",
    ]
    assert _q(r.mu_formula, "temperature_per_pressure", system) in tex
    assert _q(r.mu_finite, "temperature_per_pressure", system) in tex
