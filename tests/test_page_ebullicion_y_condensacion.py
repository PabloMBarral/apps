"""Tests de /Ebullicion_y_Condensacion (AppTest).

Fase 11.2. Los dos modos (ebullición en pileta y condensación en película), todos los
ejemplos en los tres sistemas de unidades, los cambios de fluido, de dato de la saturación,
de calentador, de superficie, de dato y de geometría, los errores, la teoría, el
procedimiento y la exportación.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.heat_transfer import boiling as bo
from core.heat_transfer import condensation as co

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "22_Ebullicion_y_Condensacion.py")
C = 273.15
M_BOIL, M_COND = "Ebullición en pileta", "Condensación en película"


def _app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    _no_problems(at)
    return at


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _metric(at: AppTest, prefix: str) -> float:
    value = next(m.value for m in at.metric if m.label.startswith(prefix))
    value = value.split(" ")[0] if not re.match(r"^-?[\d ]+$", value) else value
    return float(value.replace(" ", "").replace(",", ".").replace("−", "-"))


def _mode(at: AppTest, mode: str) -> AppTest:
    at.radio(key="bc_mode").set_value(mode).run()
    _no_problems(at)
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


def _wkey(at: AppTest, key: str) -> str:
    system = at.session_state["units_system"] if "units_system" in at.session_state else "Técnico"
    return f"{key}@{system}"


def _latex(at: AppTest) -> str:
    return "\n".join(str(x.value) for x in at.latex)


# ---------------------------------------------------------------------
# Ebullición
# ---------------------------------------------------------------------


def test_default_is_the_pan_of_the_book() -> None:
    at = _app()
    assert [e.label for e in at.expander] == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert _metric(at, "q̇ [") == pytest.approx(72241, rel=1e-3)  # el libro: 7,21·10⁴
    assert _metric(at, "Q̇ [") == pytest.approx(5106, rel=1e-3)
    assert _metric(at, "ṁ evaporada") == pytest.approx(0.002263, rel=2e-3)
    assert any("ebullición nucleada" in m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 1
    assert len(at.get("download_button")) == 2
    assert r"\dot q_0\,\psi^3" in _latex(at)


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
@pytest.mark.parametrize("index", range(len(bo.BOILING_EXAMPLES)))
def test_every_boiling_example(index: int, system: str) -> None:
    at = _app()
    _system(at, system)
    name = list(bo.BOILING_EXAMPLES)[index]
    at.selectbox(key="eb_example").set_value(name).run()
    _no_problems(at)
    r = bo.solve_pool_boiling(bo.BOILING_EXAMPLES[name].inputs)
    assert any(bo.REGIMES[r.regime] in m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 1


def test_critical_example_and_flux_mode() -> None:
    at = _app()
    at.selectbox(key="eb_example").set_value(list(bo.BOILING_EXAMPLES)[1]).run()
    _no_problems(at)
    assert _metric(at, "T_s [") == pytest.approx(108.9, abs=0.1)  # el libro: 109 °C
    assert _metric(at, "q̇/q̇_máx") == pytest.approx(100.0)
    at.radio(key="eb_1_mode").set_value("flux").run()
    _no_problems(at)
    at.number_input(key=_wkey(at, "eb_1_q")).set_value(5.0e5).run()
    _no_problems(at)
    assert _metric(at, "q̇ [") == pytest.approx(5.0e5, rel=1e-4)
    at.number_input(key=_wkey(at, "eb_1_q")).set_value(1.1e6).run()
    # más que el crítico y sin radiación (ε = 0): la película no llega antes de fundirse
    assert at.error and "se quemaría" in at.error[0].value
    at.number_input(key="eb_1_eps").set_value(0.9).run()
    _no_problems(at)
    assert any("(burnout)" in m.value for m in at.markdown)


def test_burnout_example() -> None:
    at = _app()
    at.selectbox(key="eb_example").set_value("Una resistencia que se quema (ejemplo propio)").run()
    _no_problems(at)
    assert any("(burnout)" in m.value for m in at.markdown)
    assert _metric(at, "T_s [") > 1455.0
    assert any("se quemaría" in i.value for i in at.info)


def test_burnout_without_radiation_is_an_error() -> None:
    at = _app()
    at.selectbox(key="eb_example").set_value("Una resistencia que se quema (ejemplo propio)").run()
    at.number_input(key="eb_4_eps").set_value(0.0).run()
    assert at.error and "se quemaría" in at.error[0].value


def test_surface_from_the_table_and_other_fluids() -> None:
    at = _app()
    at.selectbox(key="eb_0_Water_surf").set_value("Agua – cobre pulido").run()
    _no_problems(at)
    assert any("C_sf = 0.0128" in c.value for c in at.caption)
    for fluid in ("n-Pentane", "R134a", "Nitrogen", "Ammonia"):
        at.selectbox(key="eb_0_fluid").set_value(fluid).run()
        _no_problems(at)
        assert len(at.get("plotly_chart")) == 1
    assert any("No hay C_sf medido" in i.value for i in at.info)


def test_saturation_by_temperature() -> None:
    at = _app()
    at.radio(key="eb_0_by").set_value("La temperatura de saturación").run()
    _no_problems(at)
    at.number_input(key=_wkey(at, "eb_0_Water_Tsat")).set_value(105.0).run()
    _no_problems(at)
    assert any("T_sat" in str(x.value) for x in at.dataframe)  # la tabla de propiedades
    at.number_input(key=_wkey(at, "eb_0_Water_Tsat")).set_value(400.0).run()
    assert at.error and "punto crítico" in at.error[0].value


def test_boiling_errors() -> None:
    at = _app()
    at.number_input(key=_wkey(at, "eb_0_Water_Ts")).set_value(90.0).run()
    assert at.error and "no hierve" in at.error[0].value
    at = _app()
    at.number_input(key=_wkey(at, "eb_0_Water_p")).set_value(300.0).run()
    assert at.error and "crítica" in at.error[0].value


def test_heaters_and_regimes() -> None:
    at = _app()
    at.radio(key="eb_0_heater").set_value("cylinder").run()
    _no_problems(at)
    at.number_input(key=_wkey(at, "eb_0_D")).set_value(5.0).run()
    _no_problems(at)
    at.number_input(key=_wkey(at, "eb_0_Water_Ts")).set_value(160.0).run()
    _no_problems(at)
    assert any("ebullición de transición" in m.value for m in at.markdown)
    at.number_input(key=_wkey(at, "eb_0_Water_Ts")).set_value(400.0).run()
    _no_problems(at)
    assert any("ebullición en película" in m.value for m in at.markdown)
    at.radio(key="eb_0_heater").set_value("sphere").run()
    _no_problems(at)
    at.number_input(key=_wkey(at, "eb_0_Water_Ts")).set_value(100.5).run()
    _no_problems(at)
    assert any("convección natural" in m.value for m in at.markdown)


# ---------------------------------------------------------------------
# Condensación
# ---------------------------------------------------------------------


def test_condensation_default_is_the_vertical_plate() -> None:
    at = _mode(_app(), M_COND)
    assert _metric(at, "h [") == pytest.approx(5838, rel=2e-3)  # el libro: 5848
    assert _metric(at, "Re de la película") == pytest.approx(1286, rel=2e-3)
    assert _metric(at, "Q̇ [") == pytest.approx(6.996e5, rel=2e-3)
    assert any("con ondas" in m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 2
    assert len(at.get("download_button")) == 2


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
@pytest.mark.parametrize("index", range(len(co.CONDENSATION_EXAMPLES)))
def test_every_condensation_example(index: int, system: str) -> None:
    at = _mode(_app(), M_COND)
    _system(at, system)
    name = list(co.CONDENSATION_EXAMPLES)[index]
    at.selectbox(key="ec_example").set_value(name).run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) == 2
    assert at.latex


def test_condensation_book_tubes() -> None:
    at = _mode(_app(), M_COND)
    at.selectbox(key="ec_example").set_value(list(co.CONDENSATION_EXAMPLES)[2]).run()
    _no_problems(at)
    assert _metric(at, "h [") == pytest.approx(9284, rel=2e-3)  # el libro: 9292
    at.selectbox(key="ec_example").set_value(list(co.CONDENSATION_EXAMPLES)[3]).run()
    _no_problems(at)
    assert _metric(at, "h [") == pytest.approx(7054, rel=2e-3)  # el libro: 7060
    assert _metric(at, "h de un tubo") == pytest.approx(9284, rel=2e-3)


def test_condensation_geometries_and_options() -> None:
    at = _mode(_app(), M_COND)
    for geometry in co.GEOMETRIES:
        at.selectbox(key="ec_0_geo").set_value(geometry).run()
        _no_problems(at)
    at.selectbox(key="ec_0_geo").set_value("inclined_plate").run()
    at.number_input(key="ec_0_angle").set_value(60.0).run()
    _no_problems(at)
    at.selectbox(key="ec_0_geo").set_value("inside_tube").run()
    at.checkbox(key="ec_0_hasV").check().run()
    _no_problems(at)
    assert _metric(at, "Re del vapor") > 0
    at.checkbox(key="ec_0_Water_sh").check().run()
    _no_problems(at)
    assert any("sobrecalentado" in i.value for i in at.info)


def test_condensation_errors() -> None:
    at = _mode(_app(), M_COND)
    at.number_input(key=_wkey(at, "ec_0_Water_Ts")).set_value(105.0).run()
    assert at.error and "no condensa" in at.error[0].value
    at = _mode(_app(), M_COND)
    at.selectbox(key="ec_0_fluid").set_value("R134a").run()
    _no_problems(at)
    at.selectbox(key="ec_0_fluid").set_value("Ammonia").run()
    _no_problems(at)
