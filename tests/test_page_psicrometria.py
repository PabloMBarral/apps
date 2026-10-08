"""Tests de /Psicrometria: estado, procesos de acondicionamiento y torre (AppTest).

Fase 4. Cada ejemplo de los tres modos, los trece pares de datos, la presión
por altura o dada, el recinto, el ambiente, los sistemas de unidades, los
mensajes al alumno, el procedimiento, la teoría, la exportación y los barridos.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.hvac import HVAC_EXAMPLES, TOWER_EXAMPLES, solve_cooling_tower, solve_hvac
from core.psychrometrics import (
    PAIR_LABELS,
    PAIRS,
    STATE_EXAMPLES,
    moist_air_state,
    pressure_from_altitude,
    state_from_T_phi,
)
from ui.psychro_chart import GRID_CURVE, grid_point_from_selection

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "7_Psicrometria.py")
STATE_NAMES = list(STATE_EXAMPLES)
HVAC_NAMES = list(HVAC_EXAMPLES)
TOWER_NAMES = list(TOWER_EXAMPLES)
OMEGA = "ω [kg/kg a.s.]"
C = 273.15


def _app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert not any("No se pudo dibujar" in w.value for w in at.warning)


def _mode(at: AppTest, mode: str) -> AppTest:
    at.radio(key="ps_mode").set_value(mode).run()
    return at


# ---------------------------------------------------------------------
# Estado del aire húmedo
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", STATE_NAMES)
def test_every_state_example(name: str) -> None:
    at = _app()
    at.selectbox(key="ps_example").set_value(name).run()
    _no_problems(at)
    ex = STATE_EXAMPLES[name]
    state = moist_air_state(ex.pressure_Pa, ex.pair, ex.first, ex.second)
    assert float(_metric(at, OMEGA)) == pytest.approx(state.omega, rel=1e-3)
    assert float(_metric(at, "φ [%]")) == pytest.approx(100 * state.phi, abs=0.01)
    assert len(at.get("plotly_chart")) == 1
    columns = [list(df.value.columns) for df in at.dataframe]
    assert ["Símbolo", "Propiedad", "Valor", "Unidad"] in columns
    assert any("CoolProp (RP-1485)" in cols for cols in columns)
    assert any(c.value.startswith("📘 Sobre este ejemplo") for c in at.caption)
    assert len(at.get("download_button")) == 2


def test_default_example_is_cengel_14_1_with_the_room() -> None:
    at = _app()
    _no_problems(at)
    assert float(_metric(at, "Masa de aire seco [kg]")) == pytest.approx(85.56, abs=0.01)
    assert float(_metric(at, "Masa de vapor [kg]")) == pytest.approx(1.296, abs=0.001)
    assert float(_metric(at, "h [kJ/kg]")) == pytest.approx(63.72, abs=0.01)


@pytest.mark.parametrize("pair", PAIRS)
def test_every_pair_keeps_the_state(pair: tuple[str, str]) -> None:
    """Al cambiar de par, los datos nuevos salen del último estado: el resultado no cambia."""
    at = _app()
    at.selectbox(key="ps_example").set_value(STATE_NAMES[3]).run()  # 14-4
    at.selectbox(key="ps_3_pair").set_value(PAIR_LABELS[pair]).run()
    _no_problems(at)
    assert float(_metric(at, OMEGA)) == pytest.approx(0.01414, abs=2e-5)
    assert float(_metric(at, "T_bh [°C]")) == pytest.approx(23.93, abs=0.01)


def test_altitude_changes_the_humidity_ratio() -> None:
    at = _app()
    at.selectbox(key="ps_example").set_value(STATE_NAMES[3]).run()
    at.radio(key="ps_3_pmode").set_value("Por altura").run()
    at.number_input(key="ps_3_z").set_value(2000.0).run()
    _no_problems(at)
    expected = state_from_T_phi(pressure_from_altitude(2000.0), 35 + C, 0.4)
    assert float(_metric(at, OMEGA)) == pytest.approx(expected.omega, rel=1e-3)
    assert any("atmósfera estándar" in c.value for c in at.caption)
    assert any("veces el vapor" in i.value for i in at.info)


def test_high_pressure_warns_about_the_ideal_model() -> None:
    at = _app()
    at.selectbox(key="ps_example").set_value(STATE_NAMES[3]).run()
    at.radio(key="ps_3_pmode").set_value("Presión dada").run()
    at.number_input(key="ps_3_p@Técnico").set_value(7.0).run()
    _no_problems(at)
    assert any("factor de mejora" in i.value for i in at.info)


def test_invalid_data_show_a_message() -> None:
    at = _app()
    at.selectbox(key="ps_example").set_value(STATE_NAMES[2]).run()  # psicrómetro
    at.number_input(key="ps_2_T_wb@Técnico").set_value(30.0).run()
    assert not at.exception
    assert any("no puede superar" in e.value for e in at.error)


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_other_unit_systems(system: str) -> None:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.session_state["units_system"] = system
    at.run()
    _no_problems(at)
    unit = "J/kg" if system == "SI" else "Btu/lb"
    assert float(_metric(at, f"h [{unit}]")) > 0
    w_label = "ω [lb/lb a.s.]" if system == "Inglés" else OMEGA
    assert float(_metric(at, w_label)) == pytest.approx(0.01515, abs=1e-5)


def test_ambient_and_procedure() -> None:
    at = _app()
    at.number_input(key="ps_0_T0@Técnico").set_value(35.0).run()
    _no_problems(at)
    latex = [e.value for e in at.get("latex")]
    assert any(r"\psi_{qu}" in tex for tex in latex)
    assert any(r"\omega_s" in tex for tex in latex)
    # el ambiente aparece en la carta como el estado 0
    assert any("El 0 es el ambiente" in c.value for c in at.caption)


def test_altitude_sweep() -> None:
    at = _app()
    at.button(key="ps_sweep_btn").click().run()
    _no_problems(at)
    keys = {chart.proto.id for chart in at.get("plotly_chart")}
    assert len(at.get("plotly_chart")) == 4  # la carta y los tres barridos
    assert keys


def test_theory_expander_has_the_formulas() -> None:
    at = _app()
    latex = [e.value for e in at.get("latex")]
    assert any(r"\omega = 0.622" in tex for tex in latex)
    assert any("2.25577" in tex for tex in latex)
    assert any("Aire húmedo (psicrometría)" in m.value for m in at.markdown)


def test_grid_selection_helper() -> None:
    event = {"selection": {"points": [{"curve_number": GRID_CURVE, "x": 25.0, "y": 0.012}]}}
    assert grid_point_from_selection(event, "Técnico") == pytest.approx((298.15, 0.012))
    assert grid_point_from_selection(event, "Inglés")[0] == pytest.approx((25 - 32) / 1.8 + C)
    other = {"selection": {"points": [{"curve_number": 5, "x": 25.0, "y": 0.012}]}}
    assert grid_point_from_selection(other, "Técnico") is None
    assert grid_point_from_selection(None, "Técnico") is None


# ---------------------------------------------------------------------
# Procesos de acondicionamiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", HVAC_NAMES)
def test_every_hvac_example(name: str) -> None:
    at = _mode(_app(), "Procesos de acondicionamiento")
    at.selectbox(key="hv_example").set_value(name).run()
    _no_problems(at)
    result = solve_hvac(HVAC_EXAMPLES[name])
    assert float(_metric(at, "Calor entregado [kW]")) == pytest.approx(
        result.heating_W / 1e3, rel=1e-3, abs=1e-9
    )
    assert float(_metric(at, "Calor quitado [kW]")) == pytest.approx(
        result.cooling_W / 1e3, rel=1e-3, abs=1e-9
    )
    assert len(at.get("plotly_chart")) == 2  # carta y exergía
    states = next(df.value for df in at.dataframe if "Estado" in df.value.columns)
    assert list(states["Estado"]) == [s.number for s in result.streams]
    processes = next(df.value for df in at.dataframe if "Proceso" in df.value.columns)
    assert list(processes["Proceso"]) == [p.label for p in result.processes]
    latex = [e.value for e in at.get("latex")]
    assert sum(r"\dot X_{\mathrm{dest}}" in tex for tex in latex) >= len(result.processes)


def test_hvac_more_processes_and_a_wrong_one() -> None:
    at = _mode(_app(), "Procesos de acondicionamiento")
    at.selectbox(key="hv_example").set_value(HVAC_NAMES[1]).run()  # 14-6
    at.radio(key="hv_1_n").set_value("2").run()
    _no_problems(at)  # el segundo, por defecto, recalienta hasta 25 °C
    assert float(_metric(at, "Calor entregado [kW]")) > 0
    at.selectbox(key="hv_1_p0_type").set_value("Calentamiento o enfriamiento sensible").run()
    assert not at.exception
    assert any("punto de rocío" in e.value for e in at.error)


def test_hvac_mass_flow_and_ambient() -> None:
    at = _mode(_app(), "Procesos de acondicionamiento")
    at.selectbox(key="hv_example").set_value(HVAC_NAMES[1]).run()
    at.radio(key="hv_1_flowkind").set_value("Caudal de aire seco").run()
    at.number_input(key="hv_1_flow_dry_air@Técnico").set_value(1.0).run()
    _no_problems(at)
    from core.hvac import AirFlow, AirInlet

    inputs = HVAC_EXAMPLES[HVAC_NAMES[1]]
    expected = solve_hvac(
        type(inputs)(inputs.p_Pa, AirInlet(303.15, 0.8, AirFlow(1.0, "dry_air")), inputs.processes)
    )
    assert float(_metric(at, "Calor quitado [kW]")) == pytest.approx(
        expected.cooling_W / 1e3, rel=1e-3
    )
    at.checkbox(key="hv_1_dead_inlet").uncheck().run()
    _no_problems(at)


def test_hvac_sweep() -> None:
    at = _mode(_app(), "Procesos de acondicionamiento")
    at.button(key="hv_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) == 4


# ---------------------------------------------------------------------
# Torre de enfriamiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", TOWER_NAMES)
def test_every_tower_example(name: str) -> None:
    at = _mode(_app(), "Torre de enfriamiento")
    at.selectbox(key="ct_example").set_value(name).run()
    _no_problems(at)
    result = solve_cooling_tower(TOWER_EXAMPLES[name])
    assert float(_metric(at, "ṁ aire seco [kg/s]")) == pytest.approx(
        result.m_dry_air_kg_s, rel=1e-3
    )
    assert float(_metric(at, "Reposición [kg/s]")) == pytest.approx(result.makeup_kg_s, rel=1e-3)
    assert len(at.get("plotly_chart")) == 1
    assert len(at.get("download_button")) == 2


def test_tower_wrong_water_outlet() -> None:
    at = _mode(_app(), "Torre de enfriamiento")
    at.number_input(key="ct_0_Twout@Técnico").set_value(10.0).run()
    assert not at.exception
    assert any("bulbo húmedo" in e.value for e in at.error)


def test_tower_sweep() -> None:
    at = _mode(_app(), "Torre de enfriamiento")
    at.button(key="ct_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) == 3
