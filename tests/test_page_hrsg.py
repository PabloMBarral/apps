"""Tests de la página de la HRSG con ``streamlit.testing`` (AppTest) — Fase 3.3."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.cycles.hrsg import (
    HRSG_EXAMPLES,
    FlueGas,
    exhaust_composition,
    solve_hrsg,
)

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "10_HRSG.py")
EXAMPLES = list(HRSG_EXAMPLES)
K = "hr_0"  # prefijo de las keys del ejemplo 0 (turbina de gas, vapor sobrecalentado)
STEAM = "ṁ vapor [kg/s]"
STACK = "T chimenea [°C]"


def _new_app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _frame(at: AppTest, column: str):  # noqa: ANN202 — DataFrame de la tabla con esa columna
    return next(df.value for df in at.dataframe if column in df.value.columns)


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert not any("No se pudo dibujar" in w.value for w in at.warning)


def test_default_example_on_load() -> None:
    at = _new_app()
    _no_problems(at)
    assert float(_metric(at, STEAM)) == pytest.approx(15.385, abs=1e-3)
    assert float(_metric(at, STACK)) == pytest.approx(153.0, abs=0.1)
    assert float(_metric(at, "Gases tras el SH [°C]")) == pytest.approx(503.6, abs=0.1)
    assert _metric(at, "Q̇ [kW]") == "50 178"
    assert at.get("plotly_chart")  # el diagrama T–Q
    sections = _frame(at, "Sección")
    assert list(sections["Sección"]) == ["sobrecalentador", "evaporador", "economizador"]
    assert list(_frame(at, "Estado")["Estado"])[-1] == "5 vapor sobrecalentado"
    assert list(_frame(at, "Punto")["Punto"]) == ["a", "b", "c", "d"]
    comparison = _frame(at, "Magnitud")
    assert "Saturado" in comparison.columns
    assert float(comparison["Saturado"][0]) == pytest.approx(22.45, abs=1e-3)


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_example_computes(example: str) -> None:
    at = _new_app()
    at.selectbox(key="hr_example").set_value(example).run()
    _no_problems(at)
    result = solve_hrsg(HRSG_EXAMPLES[example])
    assert float(_metric(at, STEAM)) == pytest.approx(result.m_steam_kg_s, rel=1e-4)
    assert len(_frame(at, "Estado")) == len(result.water)
    assert len(_frame(at, "Sección")) == len(result.sections)


def test_saturated_steam() -> None:
    at = _new_app()
    at.radio(key=f"{K}_steam").set_value("Saturado (sin sobrecalentador)").run()
    _no_problems(at)
    assert float(_metric(at, STEAM)) == pytest.approx(22.450, abs=1e-3)
    assert float(_metric(at, STACK)) == pytest.approx(91.05, abs=0.01)
    assert any(m.label == "Gases tras el EV [°C]" for m in at.metric)
    assert list(_frame(at, "Sección")["Sección"]) == ["evaporador", "economizador"]
    assert len(_frame(at, "Estado")) == 4
    comparison = _frame(at, "Magnitud")
    assert "Sobrecalentado a 565 °C" in comparison.columns  # 25 K bajo los gases, hasta 565 °C
    options = at.selectbox(key="hr_sweep_param").options
    assert "Temperatura del vapor" not in options


def test_excess_air_preset() -> None:
    at = _new_app()
    at.radio(key=f"{K}_comp").set_value("Gases de metano con exceso de aire λ").run()
    at.number_input(key=f"{K}_lambda").set_value(2.5).run()
    _no_problems(at)
    inputs = replace(
        HRSG_EXAMPLES[EXAMPLES[0]], gas=FlueGas.from_fractions(exhaust_composition(2.5))
    )
    expected = solve_hrsg(inputs).m_steam_kg_s
    assert float(_metric(at, STEAM)) == pytest.approx(expected, rel=1e-4)
    assert any(c.value.startswith("En fracción molar: N₂") for c in at.caption)


def test_mass_fractions_give_the_same_boiler() -> None:
    at = _new_app()
    at.radio(key=f"{K}_basis").set_value("Másica (%)").run()
    _no_problems(at)
    n2 = at.number_input(key=f"{K}_mass_N2")
    assert n2.value == pytest.approx(73.4, abs=0.1)  # wᵢ = yᵢ·Mᵢ/M
    assert float(_metric(at, STEAM)) == pytest.approx(15.385, abs=2e-3)  # redondeo al 0,01 %


def test_switching_basis_converts_the_composition() -> None:
    at = _new_app()
    at.number_input(key=f"{K}_molar_H2O").set_value(10.0).run()  # suman 101,58 %
    steam = float(_metric(at, STEAM))
    at.radio(key=f"{K}_basis").set_value("Másica (%)").run()
    _no_problems(at)
    species = ("N2", "O2", "CO2", "H2O", "Ar")
    total = sum(at.number_input(key=f"{K}_mass_{s}").value for s in species)
    assert total == pytest.approx(100.0, abs=0.05)  # la composición cargada, ya normalizada
    assert float(_metric(at, STEAM)) == pytest.approx(steam, abs=3e-3)


def test_composition_is_normalized() -> None:
    at = _new_app()
    at.number_input(key=f"{K}_molar_N2").set_value(80.0).run()
    _no_problems(at)
    assert any("se normalizan a 100 %" in c.value for c in at.caption)


def test_empty_composition_is_explained() -> None:
    at = _new_app()
    for species in ("N2", "O2", "CO2", "H2O", "Ar"):
        at.number_input(key=f"{K}_molar_{species}").set_value(0.0)
    at.run()
    assert at.error and "vacía" in at.error[0].value


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        (f"{K}_pinch@Técnico", 0.0, "pinch tiene que ser positivo"),
        (f"{K}_Tfw@Técnico", 280.0, "más fría que la salida del economizador"),
        (f"{K}_p@Técnico", 250.0, "supera la crítica"),
        (f"{K}_Ts@Técnico", 650.0, "más caliente que los gases"),
    ],
)
def test_invalid_data_are_explained(key: str, value: float, match: str) -> None:
    at = _new_app()
    at.number_input(key=key).set_value(value).run()
    assert not at.exception
    assert at.error and match in at.error[0].value


def test_notes_dew_point_and_zero_approach() -> None:
    at = _new_app()
    at.number_input(key=f"{K}_Tfw@Técnico").set_value(40.0)  # rocío de los gases: 42,6 °C
    at.number_input(key=f"{K}_approach@Técnico").set_value(0.0)
    at.run()
    _no_problems(at)
    notes = " ".join(i.value for i in at.info)
    assert "punto de rocío" in notes and "approach = 0" in notes


@pytest.mark.parametrize(
    ("system", "labels", "unit"),
    [
        ("SI", ("Q̇ [W]", "T chimenea [K]"), r"\mathrm{J/kg}"),
        ("Inglés", ("ṁ vapor [lb/s]", "T chimenea [°F]"), r"\mathrm{Btu/lb}"),
    ],
)
def test_unit_systems(system: str, labels: tuple[str, ...], unit: str) -> None:
    at = _new_app()
    at.selectbox(key="units_system").set_value(system).run()
    _no_problems(at)
    metric_labels = [m.label for m in at.metric]
    assert all(label in metric_labels for label in labels)
    assert any(unit in tex.value for tex in at.latex)
    if system == "SI":
        assert float(_metric(at, "T chimenea [K]")) == pytest.approx(426.2, abs=0.1)


def test_theory_procedure_and_export_are_present() -> None:
    at = _new_app()
    labels = [e.label for e in at.expander]
    assert any("Fórmulas teóricas" in label for label in labels)
    assert any("Procedimiento" in label for label in labels)
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    text = " ".join(md.value for md in theory.markdown)
    assert "§3.3" in text and "§5" in text and "§16" in text and "Kehlhofer" in text
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    titles = " ".join(md.value for md in procedure.markdown)
    assert "Caudal de vapor" in titles and "Punto de rocío" in titles
    assert len(at.get("download_button")) == 2


@pytest.mark.parametrize(
    "variable", ["Pinch", "Presión de evaporación", "Temperatura del agua de alimentación"]
)
def test_sweeps_draw_the_charts(variable: str) -> None:
    at = _new_app()
    at.selectbox(key="hr_sweep_param").set_value(variable).run()
    at.button(key="hr_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) == 4  # T–Q + exergía + caudal de vapor + chimenea
    if variable.startswith("Temperatura del agua"):
        assert any("El caudal de vapor no cambia" in c.value for c in at.caption)
