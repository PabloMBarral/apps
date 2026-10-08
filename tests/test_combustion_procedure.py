"""Tests del procedimiento de la combustión — Fase 5.

Que los pasos sigan el vademecum §16, que los números que se muestran sean
los del resultado y que el LaTeX esté bien armado en los tres sistemas de
unidades (el ancho y la validez con KaTeX se miden aparte, en un navegador).
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from core.combustion.combustion import (
    COMBUSTION_EXAMPLES,
    FLUE_GAS_EXAMPLES,
    CombustionInputs,
    FlueGasInputs,
    solve_combustion,
    solve_flue_gas,
)
from core.combustion.combustion_procedure import combustion_steps, flue_gas_steps, species_latex
from core.combustion.fuels import FUELS
from core.combustion.stoichiometry import AirSpec, Oxidizer
from core.latex import latex_number

SYSTEMS = ["SI", "Técnico", "Inglés"]


def _check_latex(tex: str) -> None:
    assert tex.count("{") == tex.count("}"), tex
    assert "nan" not in tex and "inf" not in tex, tex
    assert not re.search(r"[-+]\s*-\d", tex), tex  # negativos entre paréntesis
    assert not re.search(r"\\cdot\s*-\d", tex), tex
    assert r"\begin{aligned}" not in tex or tex.endswith(r"\end{aligned}"), tex


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(COMBUSTION_EXAMPLES))
def test_combustion_procedure_is_well_formed(name: str, system: str) -> None:
    steps = combustion_steps(solve_combustion(COMBUSTION_EXAMPLES[name]), system)  # type: ignore[arg-type]
    titles = [re.sub(r"^\d+\. ", "", s.title) for s in steps]
    assert titles[:6] == [
        "Combustible",
        "Comburente",
        "Oxígeno y aire teóricos",
        "Exceso de aire",
        "Productos",
        "Humos",
    ]
    assert "Poder calorífico" in titles and "Temperatura adiabática de llama" in titles
    for k, step in enumerate(steps, start=1):
        assert step.title.startswith(f"{k}. ")
        assert step.text
        for tex in step.latex:
            _check_latex(tex)


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(FLUE_GAS_EXAMPLES))
def test_flue_gas_procedure_is_well_formed(name: str, system: str) -> None:
    steps = flue_gas_steps(solve_flue_gas(FLUE_GAS_EXAMPLES[name]), system)  # type: ignore[arg-type]
    titles = [re.sub(r"^\d+\. ", "", s.title) for s in steps]
    assert titles == [
        "Combustible",
        "Comburente",
        "Análisis de los humos",
        "Aire, inquemados y agua",
    ]
    for step in steps:
        for tex in step.latex:
            _check_latex(tex)


def _all_latex(steps) -> str:  # type: ignore[no-untyped-def]
    return "\n".join(t for s in steps for t in s.latex)


def test_the_numbers_shown_are_the_results() -> None:
    r = solve_combustion(
        COMBUSTION_EXAMPLES["Cengel 15-10 y 15-11: metano con 50 % de exceso de aire"]
    )
    tex = _all_latex(combustion_steps(r, "Técnico"))
    assert latex_number(r.stoich.lam, 5) in tex
    assert latex_number(r.stoich.AC, 5) in tex
    assert latex_number(r.flame.T_K - 273.15, 5) in tex
    assert r"802\,562" in tex  # PCI del metano, kJ/kmol (con los h_f de la tabla)
    assert latex_number(r.lhv * r.per_kg / 1e3, 5) in tex
    assert latex_number(r.heat.eta_lhv, 4) in tex  # type: ignore[union-attr, arg-type]
    assert latex_number(r.second_law.S_gen_heat, 5) in tex  # type: ignore[union-attr, arg-type]


def test_reaction_of_a_pure_fuel_is_written() -> None:
    r = solve_combustion(COMBUSTION_EXAMPLES["Cengel 15-1: octano con 20 kmol de O₂"])
    steps = combustion_steps(r, "Técnico")
    products = next(s for s in steps if s.title.endswith("Productos"))
    assert r"\longrightarrow 8\,\mathrm{CO_{2}} + 9\,\mathrm{H_{2}O}" in products.latex[0]
    assert r"\mathrm{C_{8}H_{18}}(\ell) + 95.24" in products.latex[0]


def test_mixtures_sum_the_atoms() -> None:
    r = solve_combustion(CombustionInputs(FUELS["Gas natural típico"]))
    fuel_step = combustion_steps(r, "Técnico")[0]
    assert any(
        tex.startswith(r"\begin{aligned}n_\mathrm{C} &= 0.91 \cdot 1") for tex in fuel_step.latex
    )
    assert any("M_\\mathrm{comb}" in tex for tex in fuel_step.latex)


def test_analysis_fuels_go_per_kg() -> None:
    r = solve_combustion(CombustionInputs(FUELS["Fueloil"], air=AirSpec("excess", 0.15)))
    tecnico = _all_latex(combustion_steps(r, "Técnico"))
    si = _all_latex(combustion_steps(r, "SI"))
    assert r"\mathrm{kmol/kg}" in tecnico
    assert r"\mathrm{mol/kg}" in si
    assert "(dato)" in tecnico  # el PCS es un dato
    assert r"\mathrm{lbmol/lb}" in _all_latex(combustion_steps(r, "Inglés"))


def test_dissociation_shows_kp_checks() -> None:
    r = solve_combustion(COMBUSTION_EXAMPLES["Cengel 15-8: llama adiabática del octano"])
    step = next(s for s in combustion_steps(r, "Técnico") if s.title.endswith("disociación"))
    tex = "\n".join(step.latex)
    assert r"\ln K_p" in tex and "K_p &=" in tex
    assert r"\rightleftharpoons" in tex
    for check in r.kp_checks:
        assert latex_number(check.ln_Kp_tables, 5) in tex


def test_english_units_use_rankine_in_kp() -> None:
    r = solve_combustion(CombustionInputs(FUELS["Metano"]))
    step = next(s for s in combustion_steps(r, "Inglés") if s.title.endswith("disociación"))
    assert latex_number(r.flame_eq.T_K * 1.8, 5) in "\n".join(step.latex)  # type: ignore[union-attr]
    assert "1.9859" in "\n".join(step.latex)  # R_u en Btu/(lbmol·°R)


def test_o2_given_shows_the_formula_for_lambda() -> None:
    inputs = CombustionInputs(FUELS["Gas natural típico"], air=AirSpec("o2_dry", 0.03))
    step = next(
        s
        for s in combustion_steps(solve_combustion(inputs), "Técnico")
        if s.title.endswith("Exceso de aire")
    )
    assert any("y_{\\mathrm{O_2},s}" in tex for tex in step.latex)


def test_humid_air_and_acid_dew_point_steps() -> None:
    inputs = CombustionInputs(
        FUELS["Fueloil"],
        oxidizer=Oxidizer("technical", 303.15, 0.7),
        air=AirSpec("excess", 0.15),
    )
    steps = combustion_steps(solve_combustion(inputs), "Técnico")
    oxidizer = next(s for s in steps if s.title.endswith("Comburente"))
    assert any(tex.startswith(r"\begin{aligned}y_v") for tex in oxidizer.latex)
    dew = next(s for s in steps if s.title.endswith("Punto de rocío"))
    assert any("T_{pr,a}" in tex for tex in dew.latex)
    assert "Verhoff" in dew.text


def test_bomb_procedure_uses_internal_energy() -> None:
    r = solve_combustion(
        COMBUSTION_EXAMPLES["Cengel 15-7: metano y oxígeno en un recipiente rígido"]
    )
    tex = _all_latex(combustion_steps(r, "Técnico"))
    assert "U_r" in tex and "p_2" in tex


def test_orsat_and_o2_methods() -> None:
    orsat = flue_gas_steps(
        solve_flue_gas(FLUE_GAS_EXAMPLES["Cengel 15-4: Orsat del octano"]), "Técnico"
    )
    tex = _all_latex(orsat)
    assert r"y_{\mathrm{N_2},s}" in tex and r"\Delta O" in tex
    o2 = flue_gas_steps(
        solve_flue_gas(FlueGasInputs(FUELS["Gas natural típico"], o2_dry=0.03)), "Técnico"
    )
    assert any(r"\lambda &= 1 + " in t for t in o2[2].latex)


def test_species_latex() -> None:
    assert species_latex("CO2") == r"\mathrm{CO_{2}}"
    assert species_latex("C8H18(l)") == r"\mathrm{C_{8}H_{18}}(\ell)"
    assert species_latex("n-C4H10") == r"\mathrm{n{-}C_{4}H_{10}}"


@pytest.mark.parametrize("name", list(FUELS))
def test_every_fuel_and_variant_builds_the_procedure(name: str) -> None:
    base = CombustionInputs(FUELS[name], air=AirSpec("lambda", 1.2))
    variants = [
        base,
        replace(base, air=AirSpec("lambda", 0.9)),
        replace(base, T_products_K=320.0),
        replace(base, oxidizer=Oxidizer("dry", 288.15, 0.5)),
        replace(base, process="v"),
    ]
    for inputs in variants:
        try:
            r = solve_combustion(inputs)
        except ValueError:
            continue
        for system in SYSTEMS:
            for step in combustion_steps(r, system):  # type: ignore[arg-type]
                for tex in step.latex:
                    _check_latex(tex)
