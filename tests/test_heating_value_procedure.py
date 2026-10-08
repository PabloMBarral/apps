"""Tests del procedimiento del poder calorífico — Fase 6.

Que los pasos sigan el orden de la página, que los números que se muestran
sean los del resultado y que el LaTeX esté bien armado en los tres sistemas
de unidades (la validez y el ancho con KaTeX se miden aparte, en un navegador).
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from core.combustion.combustion import COMBUSTION_EXAMPLES, solve_combustion
from core.combustion.combustion_procedure import combustion_steps
from core.combustion.fuels import FUELS, Fuel
from core.combustion.heating_value import (
    CORRELATIONS,
    HEATING_VALUE_EXAMPLES,
    HeatingValueInputs,
    estimate_hhv_as_fired,
    solve_heating_value,
)
from core.combustion.heating_value_procedure import correlation_latex, heating_value_steps
from core.latex import latex_number
from core.units_system import convert_from_si

SYSTEMS = ["SI", "Técnico", "Inglés"]


def _check_latex(tex: str) -> None:
    assert tex.count("{") == tex.count("}"), tex
    assert "nan" not in tex and "inf" not in tex, tex
    assert not re.search(r"[-+]\s*-\d", tex), tex
    assert r"\begin{aligned}" not in tex or tex.endswith(r"\end{aligned}"), tex


def _titles(steps) -> list[str]:  # type: ignore[no-untyped-def]
    return [re.sub(r"^\d+\. ", "", s.title) for s in steps]


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(HEATING_VALUE_EXAMPLES))
def test_procedure_is_well_formed(name: str, system: str) -> None:
    result = solve_heating_value(HEATING_VALUE_EXAMPLES[name])
    steps = heating_value_steps(result, system)  # type: ignore[arg-type]
    titles = _titles(steps)
    assert titles[0] == "Análisis en base seca"
    assert titles[1 : 1 + len(result.estimates)] == [e.name for e in result.estimates]
    assert titles[-3:] == [
        "PCS tal cual y seco sin cenizas",
        "Poder calorífico inferior",
        "Desvío respecto de la referencia",
    ]
    for k, step in enumerate(steps, start=1):
        assert step.title.startswith(f"{k}. ")
        assert step.text
        for tex in step.latex:
            _check_latex(tex)


@pytest.mark.parametrize("basis", ["ar", "d", "daf"])
def test_basis_step(basis: str) -> None:
    inp = replace(HEATING_VALUE_EXAMPLES["Lignito Beulah-Zap"], basis=basis)  # type: ignore[arg-type]
    step = heating_value_steps(solve_heating_value(inp), "Técnico")[0]
    tex = "\n".join(step.latex)
    assert r"C_\mathrm{s} &= " in tex
    if basis == "ar":
        assert r"\frac{x_\mathrm{tc}}{1 - W}" in tex
        assert r"\frac{x_\mathrm{tc}}{1 - 0.3224}" in tex
    elif basis == "daf":
        assert r"x_\mathrm{sscz}\,(1 - \mathit{Cz}_\mathrm{s})" in tex
        assert r"\mathit{Cz}_\mathrm{s} &=" not in tex  # las cenizas son el dato
    else:
        assert "1 - W" not in tex
    assert "65.85" in tex  # el C seco de Vorres (1990)


def test_the_numbers_shown_are_the_results() -> None:
    result = solve_heating_value(HEATING_VALUE_EXAMPLES["Bagazo de caña (50 % de humedad)"])
    steps = heating_value_steps(result, "Técnico")
    by_title = {re.sub(r"^\d+\. ", "", s.title): "\n".join(s.latex) for s in steps}
    for e in result.estimates:
        tex = by_title[e.name]
        assert latex_number(e.hhv_d / 1e6, 4) + r"\ \mathrm{MJ/kg}" in tex
        assert latex_number(convert_from_si(e.hhv_d, "specific_enthalpy", "Técnico"), 5) in tex
    main = result.main
    assert main.lhv_ar is not None and main.lhv_d is not None
    lhv = by_title["Poder calorífico inferior"]
    assert latex_number(main.lhv_ar / 1000, 5) in lhv
    assert latex_number(main.lhv_d / 1000, 5) in lhv
    assert "88.13" in lhv  # W* en %
    bases = by_title["PCS tal cual y seco sin cenizas"]
    assert latex_number(main.hhv_ar / 1000, 5) in bases
    dev = by_title["Desvío respecto de la referencia"]
    assert r"\delta_{\text{CyP}} &= \frac{19.42 - 18.99}{18.99} = +2.26\,\%" in dev
    assert r"\delta_{\text{Dulong}}" in dev and r"= -9.4\,\%" in dev


def test_contributions_add_up() -> None:
    """La tabla de aportes de cada correlación suma su PCS (al redondeo de 1 kJ/kg)."""
    inp = HEATING_VALUE_EXAMPLES["Carbón bituminoso Pittsburgh N.º 8"]
    result = solve_heating_value(inp)
    for e in result.estimates:
        analysis = inp.ultimate if CORRELATIONS[e.key].analysis == "ultimate" else inp.proximate
        assert analysis is not None
        tex = correlation_latex(e.key, analysis.pct(), e.hhv_d, "SI")[-2]
        parts = [float(m) for m in re.findall(r"&= (-?[\d.]+)", tex)]
        assert sum(parts[:-1]) == pytest.approx(parts[-1], abs=0.003 * len(parts))
        assert parts[-1] == pytest.approx(e.hhv_d / 1e6, rel=1e-3)


def test_dulong_shows_the_available_hydrogen() -> None:
    result = solve_heating_value(HEATING_VALUE_EXAMPLES["Eucalipto"])
    step = next(s for s in heating_value_steps(result, "SI") if s.title.endswith("Dulong"))
    tex = "\n".join(step.latex)
    assert r"H - \frac{O}{8}" in tex
    assert r"5.92 - \frac{44.18}{8}" in tex
    assert "H - O/8:" in tex
    assert "hidrógeno disponible" in step.text
    assert "Ojo: O = 44,2 %" in step.text


def test_without_hydrogen_there_is_no_pci() -> None:
    result = solve_heating_value(
        HEATING_VALUE_EXAMPLES["Carbón bituminoso Pocahontas N.º 3 (solo inmediato)"]
    )
    step = next(
        s
        for s in heating_value_steps(result, "SI")
        if s.title.endswith("Poder calorífico inferior")
    )
    assert step.latex == ()
    assert "Sin el H" in step.text


def test_without_reference_there_is_no_deviation() -> None:
    inp = HeatingValueInputs.from_basis(
        "d", ultimate_pct={"C": 50, "H": 6, "O": 42, "N": 0.5}, ash_pct=1.5
    )
    titles = _titles(heating_value_steps(solve_heating_value(inp), "Inglés"))
    assert titles[-1] == "Poder calorífico inferior"


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("key", ["channiwala_parikh", "boie", "dulong"])
def test_combustion_procedure_with_estimated_hhv(system: str, key: str) -> None:
    """En /Combustion, un PCS estimado muestra su correlación en el paso del poder calorífico."""
    base = COMBUSTION_EXAMPLES["Bagazo de caña en la caldera de un ingenio"]
    analysis = base.fuel.analysis
    assert analysis is not None
    fuel = Fuel.from_analysis(
        "Bagazo", analysis, estimate_hhv_as_fired(analysis, key), hhv_correlation=key
    )
    steps = combustion_steps(solve_combustion(replace(base, fuel=fuel)), system)  # type: ignore[arg-type]
    step = next(s for s in steps if s.title.endswith("Poder calorífico"))
    assert CORRELATIONS[key].name in step.text
    tex = "\n".join(step.latex)
    assert r"PCS_\mathrm{s}" in tex and r"\text{(dato)}" not in tex
    for t in step.latex:
        _check_latex(t)
    # Sin correlación, como en la 0.21.0: el PCS es un dato.
    plain = combustion_steps(solve_combustion(base), system)  # type: ignore[arg-type]
    plain_step = next(s for s in plain if s.title.endswith("Poder calorífico"))
    assert r"\text{(dato)}" in "\n".join(plain_step.latex)


def test_library_analysis_fuels_keep_their_data() -> None:
    for name in ("Fueloil", "Carbón (Pensilvania)", "Bagazo de caña (50 % de humedad)"):
        assert FUELS[name].hhv_correlation is None
