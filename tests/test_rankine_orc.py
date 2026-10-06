"""Tests del ciclo de Rankine orgánico (ORC) — Fase 3.1c.

Otros fluidos de trabajo (R-245fa, R-1233zd(E), isopentano, tolueno, R-134a,
R-1234yf, amoníaco) y el recuperador. Referencias: cálculos a mano con CoolProp
(ecuaciones de estado de Helmholtz) y la clasificación de fluidos secos y
húmedos de Chen, Goswami y Stefanakos (2010).
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import pytest
from CoolProp.CoolProp import PropsSI

from core.cycles.rankine import (
    RANKINE_EXAMPLES,
    RANKINE_FLUIDS,
    FeedwaterHeater,
    RankineInputs,
    Recuperator,
    condenser_cooling_water,
    default_sweep_values,
    fluid_behavior,
    rankine_notes,
    rankine_segments,
    rankine_steps,
    rankine_sweep,
    rankine_to_dict,
    solve_rankine,
    suggested_rankine_inputs,
)

C = 273.15
R245 = "R245fa"
ORC_EXAMPLES = [k for k in RANKINE_EXAMPLES if k.startswith("ORC")]


def _orc(recuperator: float | None = None, **kw) -> RankineInputs:
    return RankineInputs(
        20e5,
        None,
        1.8e5,
        0.8,
        0.7,
        fluid=R245,
        recuperator=None if recuperator is None else Recuperator(recuperator),
        **kw,
    )


def _prop(name: str, k1: str, v1: float, k2: str, v2: float, fluid: str = R245) -> float:
    return PropsSI(name, k1, v1, k2, v2, fluid)


# ---------------------------------------------------------------------
# Ciclo con R-245fa contra el cálculo a mano
# ---------------------------------------------------------------------


def test_orc_states_against_coolprop() -> None:
    r = solve_rankine(_orc())
    h1 = _prop("H", "P", 1.8e5, "Q", 0)
    s1 = _prop("S", "P", 1.8e5, "Q", 0)
    h2 = h1 + (_prop("H", "P", 20e5, "S", s1) - h1) / 0.7
    h3 = _prop("H", "P", 20e5, "Q", 1)
    s3 = _prop("S", "P", 20e5, "Q", 1)
    h4 = h3 - 0.8 * (h3 - _prop("H", "P", 1.8e5, "S", s3))
    assert [s.h_J_per_kg for s in r.states] == pytest.approx([h1, h2, h3, h4], rel=1e-6)
    assert r.eta_th == pytest.approx(((h3 - h4) - (h2 - h1)) / (h3 - h2), rel=1e-6)
    assert r.x_turbine_out is None  # fluido seco: el escape sale sobrecalentado
    assert r.layout.of_kind("boiler")[0].label == "evaporador"  # type: ignore[union-attr]


def test_recuperator_effectiveness_and_balance() -> None:
    r = solve_rankine(_orc(0.8))
    (rec,) = r.of_kind("recuperator")
    st = r.states
    cold_in, cold_out = st[rec.port("fw_in").state], st[rec.port("fw_out").state]
    hot_in, hot_out = st[rec.port("in").state], st[rec.port("out").state]
    q = cold_out.h_J_per_kg - cold_in.h_J_per_kg
    assert hot_in.h_J_per_kg - hot_out.h_J_per_kg == pytest.approx(q, rel=1e-7)
    q_hot = hot_in.h_J_per_kg - _prop("H", "T", cold_in.T_K, "P", hot_out.P_Pa)
    q_cold = _prop("H", "T", hot_in.T_K, "P", cold_out.P_Pa) - cold_in.h_J_per_kg
    assert q == pytest.approx(0.8 * min(q_hot, q_cold), rel=1e-6)
    assert r.q_recuperator_J_per_kg == pytest.approx(q, rel=1e-9)
    assert dict(r.tespy_heaters)["recuperador"] == pytest.approx(q, rel=1e-6)
    plain = solve_rankine(_orc())
    assert r.eta_th > plain.eta_th  # el recuperador mejora el ORC con fluido seco
    assert r.T_mean_in_K > plain.T_mean_in_K
    assert r.w_net_J_per_kg == pytest.approx(plain.w_net_J_per_kg, rel=1e-6)  # mismo trabajo
    assert r.q_in_J_per_kg - r.q_out_J_per_kg == pytest.approx(r.w_net_J_per_kg, rel=1e-8)


def test_recuperator_numbering_and_segments() -> None:
    r = solve_rankine(_orc(0.8))
    assert [label.split(" (")[0] for label in r.layout.labeled()] == [  # type: ignore[union-attr]
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    segments = rankine_segments(r)
    assert (1, 2) in segments and (4, 5) in segments  # los dos lados del recuperador


# ---------------------------------------------------------------------
# Fluidos: clasificación, valores por defecto y ejemplos
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fluid", "behavior"),
    [
        ("Water", "húmedo"),
        ("Ammonia", "húmedo"),
        ("R134a", "isoentrópico"),
        ("R1234yf", "isoentrópico"),
        ("R245fa", "seco"),
        ("R1233zd(E)", "seco"),
        ("Isopentane", "seco"),
        ("Toluene", "seco"),
    ],
)
def test_fluid_behavior(fluid: str, behavior: str) -> None:
    info = fluid_behavior(fluid)
    assert info.behavior == behavior
    assert (info.xi_J_per_kg_K2 > 0) == (info.xi_star > 0)


@pytest.mark.parametrize("fluid", RANKINE_FLUIDS)
def test_suggested_inputs_solve_for_every_fluid(fluid: str) -> None:
    inputs = suggested_rankine_inputs(fluid)
    assert inputs.fluid == fluid
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = solve_rankine(inputs)
    assert 0.05 < r.eta_th < 0.4
    assert (inputs.recuperator is not None) == (fluid_behavior(fluid).behavior == "seco")
    for system in ("SI", "Técnico", "Inglés"):
        for step in rankine_steps(r, system):  # type: ignore[arg-type]
            for t in step.latex:
                assert t.count("{") == t.count("}"), (fluid, step.title, t)


@pytest.mark.parametrize("name", ORC_EXAMPLES)
def test_orc_examples(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = solve_rankine(RANKINE_EXAMPLES[name])
    assert r.W_net_W > 0.4e6  # tamaño de planta real (los ejemplos traen su caudal)
    notes = " ".join(rankine_notes(r))
    expected = {"R245fa": "fluido seco", "Toluene": "fluido seco", "R134a": "casi isoentrópico"}
    assert expected[r.inputs.fluid] in notes


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (replace(_orc(), fluid="Helium"), "no está disponible"),
        (replace(_orc(), T_turbine_in_K=200 + C), "ecuación de estado del R-245fa"),
        (replace(_orc(), T_turbine_in_K=100 + C), "sería líquido"),
        (replace(_orc(), p_boiler_Pa=40e5), "no hay vapor saturado"),
        (RankineInputs(26e5, None, 8e5, fluid="R134a", recuperator=Recuperator(0.8)), "húmedo"),
        (replace(_orc(0.8), heaters=(FeedwaterHeater(6e5),)), "sin calentadores"),
        (_orc(1.0), "entre 0 y 1"),
        (_orc(0.0), "entre 0 y 1"),
    ],
)
def test_invalid_orc_inputs_are_explained(inputs: RankineInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_rankine(inputs)


# ---------------------------------------------------------------------
# Procedimiento, barridos, export y agua de enfriamiento
# ---------------------------------------------------------------------


def test_procedure_names_the_evaporator_and_the_equation_of_state() -> None:
    r = solve_rankine(_orc(0.8))
    steps = rankine_steps(r, "Técnico")
    text = " ".join(s.text for s in steps)
    assert "evaporador" in text and "Cengel no tabula el R-245fa" in text
    assert "caldera" not in text
    rec = next(s for s in steps if s.title.startswith("Recuperador"))
    assert any(r"\varepsilon\,q_{\text{máx}}" in t for t in rec.latex)
    r134 = solve_rankine(RANKINE_EXAMPLES[next(k for k in ORC_EXAMPLES if "R-134a" in k)])
    text134 = " ".join(s.text for s in rankine_steps(r134, "Técnico"))
    assert "Cengel A-12" in text134 and "q_{\\mathrm{evap}}" in " ".join(
        t for s in rankine_steps(r134, "Técnico") for t in s.latex
    )


def test_water_procedure_still_reads_the_steam_tables() -> None:
    r = solve_rankine(RankineInputs(3e6, 350 + C, 75e3))
    text = " ".join(s.text for s in rankine_steps(r, "Técnico"))
    assert "Cengel A-5" in text and "Cengel A-6" in text and "caldera" in text


def test_recuperator_effectiveness_sweep() -> None:
    inputs = _orc(0.8)
    values = default_sweep_values(inputs, "recuperator_eff", n=4)
    points = rankine_sweep(inputs, "recuperator_eff", values)
    assert len(points) == 4
    etas = [p.eta_th for p in points]
    assert etas == sorted(etas)  # más efectividad, más rendimiento
    assert rankine_sweep(_orc(), "recuperator_eff", values) == []  # sin recuperador, nada


@pytest.mark.parametrize("parameter", ["p_boiler", "T_turbine_in", "p_condenser"])
@pytest.mark.parametrize("fluid", ["R245fa", "Toluene", "R134a"])
def test_orc_sweeps_stay_inside_the_fluid_range(fluid: str, parameter: str) -> None:
    inputs = suggested_rankine_inputs(fluid)
    values = default_sweep_values(inputs, parameter, n=5)  # type: ignore[arg-type]
    points = rankine_sweep(inputs, parameter, values)  # type: ignore[arg-type]
    assert len(points) >= 3, (fluid, parameter, values)


def test_export_names_the_fluid() -> None:
    r = solve_rankine(_orc(0.8))
    data = rankine_to_dict(r, "Técnico")
    assert data["fluido"] == R245 and data["fluido_es"] == "R-245fa"
    assert data["ciclo"] == "Rankine orgánico (ORC) simple con recuperador"
    assert "Helmholtz" in data["fuente"]
    assert data["datos"]["recuperador"]["efectividad"] == 0.8
    assert data["resultados"]["q_recuperador"]["valor"] == pytest.approx(
        r.q_recuperator_J_per_kg / 1e3
    )


def test_cooling_water_of_an_orc() -> None:
    r = solve_rankine(_orc(0.8))
    cooling = condenser_cooling_water(r, 20 + C, 25 + C)
    assert cooling.Q_W == pytest.approx(r.Q_out_W)
    assert cooling.m_dot_kg_s == pytest.approx(r.Q_out_W / (4.18e3 * 5.0), rel=5e-3)
