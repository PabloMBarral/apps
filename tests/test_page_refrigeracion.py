"""Tests de la página de refrigeración con ``streamlit.testing`` (AppTest) — Fase 3.2."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.cycles.refrigeration import REFRIGERATION_EXAMPLES, solve_refrigeration

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "6_Refrigeracion.py")
EXAMPLES = list(REFRIGERATION_EXAMPLES)
K = "rf_0_R134a"  # prefijo de las keys del ejemplo 0 (Cengel 11-1, R-134a)


def _new_app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _states(at: AppTest) -> list[str]:
    frame = next(df.value for df in at.dataframe if "Estado" in df.value.columns)
    return list(frame["Estado"])


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert not any("No se pudo dibujar" in w.value for w in at.warning)


def _compute(at: AppTest) -> AppTest:
    return at.button(key="rf_btn").click().run()


def test_default_example_is_computed_on_load() -> None:
    at = _new_app()
    _no_problems(at)
    assert float(_metric(at, "COP_R")) == pytest.approx(3.97, abs=0.005)  # Cengel 11-1
    assert float(_metric(at, "Q̇_C [kW]")) == pytest.approx(7.18, abs=0.01)
    assert len(_states(at)) == 4
    assert any("148.14 kJ/kg" in i.value for i in at.info)  # referencia de las tablas


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_example_computes(example: str) -> None:
    at = _new_app()
    at.selectbox(key="rf_example").set_value(example).run()
    _no_problems(at)
    inputs = REFRIGERATION_EXAMPLES[example]
    result = solve_refrigeration(inputs)
    assert len(_states(at)) == result.layout.n_states
    label = "COP_B" if inputs.heat_pump else "COP_R"
    assert float(_metric(at, label)) == pytest.approx(result.COP, abs=2e-3)
    has_exergy = any(m.label == "Rendimiento exergético [%]" for m in at.metric)
    assert has_exergy == (inputs.reservoirs is not None)


def test_flash_chamber_from_the_simple_cycle() -> None:
    at = _new_app()
    at.radio(key=f"{K}_cycle").set_value("Dos etapas con cámara").run()
    flash = at.number_input(key=f"{K}_P_fl_p@Técnico")
    assert flash.label.startswith("Presión de la cámara p₆")
    assert flash.value == pytest.approx((1.4 * 8.0) ** 0.5, rel=1e-6)  # media geométrica
    labels = [n.label for n in at.number_input]
    assert any(lab.startswith("Presión del condensador p₄") for lab in labels)
    _compute(at)
    _no_problems(at)
    assert len(_states(at)) == 9
    assert float(_metric(at, "COP_R")) > 3.97


def test_cascade_with_carbon_dioxide_draws_two_diagrams() -> None:
    at = _new_app()
    at.radio(key=f"{K}_cycle").set_value("Cascada").run()
    at.selectbox(key=f"{K}_low").set_value("CarbonDioxide").run()
    at.radio(key=f"{K}_mode").set_value("Temperatura de saturación").run()
    at.number_input(key=f"{K}_T_ev_CarbonDioxide_T@Técnico").set_value(-45.0)
    at.number_input(key=f"{K}_T_co_T@Técnico").set_value(30.0)
    at.number_input(key=f"{K}_T_cl_CarbonDioxide_T@Técnico").set_value(-5.0)
    at.number_input(key=f"{K}_T_eh_CarbonDioxide_T@Técnico").set_value(-10.0)
    _compute(at)
    _no_problems(at)
    assert len(_states(at)) == 8
    frame = next(df.value for df in at.dataframe if "Estado" in df.value.columns)
    assert "Fluido" in frame.columns  # dos refrigerantes
    assert len(at.tabs) >= 2


def test_heat_pump_shows_cop_b() -> None:
    at = _new_app()
    at.radio(key=f"{K}_use").set_value("Bomba de calor").run()
    _compute(at)
    _no_problems(at)
    assert float(_metric(at, "COP_B")) == pytest.approx(4.968, abs=2e-3)  # COP_R + 1


def test_levels_by_saturation_temperature() -> None:
    at = _new_app()
    at.radio(key=f"{K}_mode").set_value("Temperatura de saturación").run()
    at.number_input(key=f"{K}_T_ev_R134a_T@Técnico").set_value(-10.0)
    at.number_input(key=f"{K}_T_co_T@Técnico").set_value(40.0)
    _compute(at)
    _no_problems(at)
    frame = next(df.value for df in at.dataframe if "Estado" in df.value.columns)
    assert float(frame["T [°C]"][0]) == pytest.approx(-10.0, abs=1e-3)
    assert any(c.value.startswith("p de saturación") for c in at.caption)


def test_real_cycle_lowers_the_cop() -> None:
    at = _new_app()
    at.checkbox(key=f"{K}_real").check().run()
    at.number_input(key=f"{K}_eta").set_value(0.8)
    _compute(at)
    _no_problems(at)
    assert float(_metric(at, "COP_R")) < 3.9


def test_capacity_gives_the_mass_flow() -> None:
    at = _new_app()
    at.radio(key=f"{K}_flow").set_value("Capacidad").run()
    _compute(at)
    _no_problems(at)
    assert float(_metric(at, "Q̇_C [kW]")) == pytest.approx(10.0)


def test_second_law_block() -> None:
    at = _new_app()
    at.checkbox(key=f"{K}_ex").check().run()
    _compute(at)
    _no_problems(at)
    eta_ex = float(_metric(at, "Rendimiento exergético [%]"))
    assert 0.0 < eta_ex < 100.0
    titles = " ".join(md.value for md in at.markdown)
    assert "Exergía destruida: válvula de expansión" in titles


def test_invalid_pressures_are_explained() -> None:
    at = _new_app()
    at.number_input(key=f"{K}_P_ev_R134a_p@Técnico").set_value(9.0)  # bar, más que el condensador
    _compute(at)
    assert at.error and "tiene que ser menor" in at.error[0].value


def test_edited_data_ask_for_recalculation() -> None:
    at = _new_app()
    at.number_input(key=f"{K}_eta").set_value(0.85).run()
    assert any("Cambiaste datos" in c.value for c in at.caption)


@pytest.mark.parametrize(
    ("system", "unit"), [("SI", r"\mathrm{J/kg}"), ("Inglés", r"\mathrm{Btu/lb}")]
)
def test_procedure_follows_the_unit_system(system: str, unit: str) -> None:
    at = _new_app()
    at.selectbox(key="units_system").set_value(system).run()
    _no_problems(at)
    assert any(unit in tex.value for tex in at.latex)


def test_theory_procedure_and_export_are_present() -> None:
    at = _new_app()
    labels = [e.label for e in at.expander]
    assert any("Fórmulas teóricas" in label for label in labels)
    assert any("Procedimiento" in label for label in labels)
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    text = " ".join(md.value for md in theory.markdown)
    assert "§9.3" in text and "§10.4" in text and "§11" in text
    assert len(at.get("download_button")) == 2


def test_result_survives_changing_the_diagram() -> None:
    at = _new_app()
    at.selectbox(key="rf_diagram_type").set_value("Ts").run()
    _no_problems(at)
    assert float(_metric(at, "COP_R")) == pytest.approx(3.97, abs=0.005)


def test_sweep_draws_the_charts() -> None:
    at = _new_app()
    at.selectbox(key="rf_sweep_param").set_value("Temperatura de evaporación").run()
    at.button(key="rf_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) >= 3  # diagrama + COP + descarga


def test_flash_pressure_sweep() -> None:
    at = _new_app()
    example = next(e for e in EXAMPLES if e.startswith("Cengel 11-5"))
    at.selectbox(key="rf_example").set_value(example).run()
    at.selectbox(key="rf_sweep_param").set_value("Presión de la cámara").run()
    at.button(key="rf_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) >= 3


def test_changing_the_refrigerant_loads_its_defaults() -> None:
    at = _new_app()
    at.selectbox(key="rf_fluid_0").set_value("Ammonia").run()
    mode = at.radio(key="rf_0_Ammonia_mode")
    assert mode.value == "Temperatura de saturación"  # los datos sugeridos van por T
    _compute(at)
    _no_problems(at)
    assert float(_metric(at, "Q̇_C [kW]")) == pytest.approx(10.0)
