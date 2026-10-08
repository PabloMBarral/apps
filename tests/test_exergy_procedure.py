"""Tests de core.exergy.exergy_procedure (LaTeX del procedimiento de exergía)."""

from __future__ import annotations

import re
from functools import cache

import pytest

from core.combustion.heating_value import DryUltimate, argonne_coals
from core.cycles import gas_turbine as gtm
from core.cycles import rankine as rk
from core.cycles import refrigeration as rf
from core.exergy import chemical as ch
from core.exergy import physical as ph
from core.exergy import plant as pl
from core.exergy.exergy_procedure import (
    finite_source_steps,
    fuel_steps,
    heat_steps,
    mixture_steps,
    physical_steps,
    plant_steps,
    species_steps,
)
from core.latex import latex_number
from core.state_report import ProcedureStep
from core.units_system import convert_from_si

SYSTEMS = ("SI", "Técnico", "Inglés")


def _check(steps: list[ProcedureStep]) -> None:
    assert steps
    for k, st in enumerate(steps, start=1):
        assert st.title.startswith(f"{k}. ")
        for tex in st.latex:
            assert tex.count("{") == tex.count("}"), tex
            assert "- -" not in tex and "+ -" not in tex, tex
            assert not re.search(r"\b(nan|inf)\b", tex), tex


def _example(name: str):
    ex = ph.PHYSICAL_EXAMPLES[name]
    first = ph.physical_exergy(
        ex.fluid,
        ex.pair,
        ambient=ex.ambient,
        speed_m_per_s=ex.speed_m_per_s,
        height_m=ex.height_m,
        amount=ex.amount,
        mass_kg=ex.mass_kg,
        m_dot_kg_s=ex.m_dot_kg_s,
        **ex.values,
    )
    second = None
    if ex.second:
        pair, values = ex.second
        second = ph.physical_exergy(ex.fluid, pair, ambient=ex.ambient, **values)
    return first, second


@pytest.mark.parametrize("name", list(ph.PHYSICAL_EXAMPLES))
@pytest.mark.parametrize("system", SYSTEMS)
def test_physical_steps(name: str, system: str) -> None:
    first, second = _example(name)
    steps = physical_steps(first, system, second)  # type: ignore[arg-type]
    _check(steps)
    titles = " ".join(s.title for s in steps)
    assert "Estado muerto" in titles and "Exergía de flujo ψ" in titles
    assert "Parte térmica y mecánica" in titles
    if second is not None:
        assert steps[-1].title.endswith("Proceso 1 → 2")
    # el ψ del resultado aparece en el procedimiento
    psi = latex_number(convert_from_si(first.psi_J_per_kg, "specific_enthalpy", system), 5)
    assert any(psi in tex for st in steps for tex in st.latex)


def test_mass_and_flow_totals() -> None:
    tank, _ = _example(next(n for n in ph.PHYSICAL_EXAMPLES if n.startswith("Aire comprimido")))
    steps = physical_steps(tank, "Técnico")
    assert any(r"X &= m\," in tex for st in steps for tex in st.latex)
    wind, _ = _example(next(n for n in ph.PHYSICAL_EXAMPLES if n.startswith("Viento")))
    steps = physical_steps(wind, "Inglés")
    assert any(r"\frac{V^2}{2}" in tex for st in steps for tex in st.latex)
    assert any(r"\dot X &= \dot m" in tex for st in steps for tex in st.latex)


@pytest.mark.parametrize("system", SYSTEMS)
def test_heat_and_finite_source_steps(system: str) -> None:
    for ex in ph.HEAT_EXAMPLES.values():
        _check(heat_steps(ph.heat_exergy(ex.Q_W, ex.T_K, ex.T0_K), system))  # type: ignore[arg-type]
    for ex in ph.FINITE_SOURCE_EXAMPLES.values():
        r = ph.finite_source_exergy(ex.m_kg, ex.c_J_per_kg_K, ex.T_K, ex.T0_K)
        steps = finite_source_steps(r, system)  # type: ignore[arg-type]
        _check(steps)
        assert any(r"\Phi" in tex for tex in steps[0].latex)


def test_finite_source_mc_units() -> None:
    """m·c en kJ/K (Técnico) y Btu/R (Inglés): 500 kg · 0,45 kJ/(kg·K) = 225 kJ/K = 118,5 Btu/R."""
    r = ph.finite_source_exergy(500.0, 450.0, 473.0, 300.0)
    tex = finite_source_steps(r, "Técnico")[0].latex[0]
    assert r"225\ \mathrm{kJ/K}" in tex
    tex = finite_source_steps(r, "Inglés")[0].latex[0]
    assert r"118.48\ \mathrm{Btu/R}" in tex


@pytest.mark.parametrize("key", list(ch.chemical_species()))
def test_species_steps(key: str) -> None:
    r = ch.species_exergy(key)
    for system in SYSTEMS:
        steps = species_steps(r, system)  # type: ignore[arg-type]
        _check(steps)
    titles = " ".join(s.title for s in species_steps(r, "Técnico"))
    assert ("Método de Szargut" in titles) == (r.method is not None)
    assert ("Presión parcial de referencia" in titles) == (r.x_reference is not None)
    assert ("Exergía y poder calorífico" in titles) == (r.lhv_J_per_mol is not None)


def test_methane_method_numbers() -> None:
    steps = species_steps(ch.species_exergy("CH4"), "Técnico")
    method = next(s for s in steps if "Método de Szargut" in s.title)
    joined = " ".join(method.latex)
    assert r"\Delta\bar g_f" in joined and "-50" in joined  # Δg_f ≈ −50,5 MJ/kmol


@pytest.mark.parametrize("system", SYSTEMS)
def test_mixture_steps_with_and_without_condensation(system: str) -> None:
    dry = mixture_steps(ch.mixture_chemical_exergy({"N2": 0.79, "O2": 0.21}), system)  # type: ignore[arg-type]
    _check(dry)
    assert len(dry) == 1
    wet = mixture_steps(
        ch.mixture_chemical_exergy({"N2": 0.72, "O2": 0.12, "CO2": 0.05, "H2O": 0.10, "Ar": 0.01}),
        system,  # type: ignore[arg-type]
    )
    _check(wet)
    assert wet[0].title.endswith("Agua que condensa")


@pytest.mark.parametrize("system", SYSTEMS)
def test_fuel_steps_for_each_branch(system: str) -> None:
    coal = next(c for c in argonne_coals() if c.ultimate is not None)
    cases = {
        "carbón": ch.fuel_chemical_exergy(coal.ultimate, coal.hhv_d, moisture=0.1),
        "madera": ch.fuel_chemical_exergy(
            DryUltimate(C=0.50, H=0.06, O=0.43, N=0.002, A=0.008), 20.0e6, moisture=0.3
        ),
        "líquido": ch.fuel_chemical_exergy(
            DryUltimate(C=0.856, H=0.105, O=0.006, N=0.005, S=0.025, A=0.003),
            45.0e6,
            kind="líquido",
        ),
    }
    for branch, r in cases.items():
        assert r.branch == branch
        steps = fuel_steps(r, system)  # type: ignore[arg-type]
        _check(steps)
        beta = latex_number(r.beta, 5)
        assert any(beta in tex for tex in steps[1].latex)


@cache
def _rankine_plant(name: str) -> pl.PlantExergy:
    return pl.rankine_exergy(rk.solve_rankine(rk.RANKINE_EXAMPLES[name]))


@pytest.mark.parametrize("system", SYSTEMS)
def test_plant_steps_have_a_step_per_component(system: str) -> None:
    plant = _rankine_plant(next(n for n in rk.RANKINE_EXAMPLES if n.startswith("Cengel 10-6")))
    steps = plant_steps(plant, system)  # type: ignore[arg-type]
    _check(steps)
    assert len(steps) == len(plant.components) + 3
    assert steps[-1].title.endswith("Balance de la planta")
    eta = latex_number(100 * plant.efficiency, 4)
    assert any(eta in tex for tex in steps[-1].latex)


def test_plant_steps_for_refrigeration_and_gas_turbine() -> None:
    fridge = pl.refrigeration_exergy(
        rf.solve_refrigeration(next(iter(rf.REFRIGERATION_EXAMPLES.values())))
    )
    steps = plant_steps(fridge, "Técnico")
    _check(steps)
    assert "trabajo" in steps[1].text.lower()
    gt = pl.gas_turbine_plant_exergy(
        gtm.solve_gas_turbine(
            gtm.GAS_TURBINE_EXAMPLES[next(n for n in gtm.GAS_TURBINE_EXAMPLES if "Industrial" in n)]
        ),
        "szargut",
    )
    steps = plant_steps(gt, "SI")
    _check(steps)
    assert r"\dot X_{\mathrm{comb}}" in steps[1].latex[0]
    assert r"\dot X_{\mathrm{aire}}" in steps[1].latex[1]
    escape = next(s for s in steps if "Escape" in s.title)
    assert escape.text.startswith("Pérdida")
