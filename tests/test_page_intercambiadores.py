"""Tests de /Intercambiadores (AppTest).

Fase 8.2. Los cuatro modos (verificación, dimensionamiento, ensayo y U global),
todos los ejemplos, los cambios de tipo, de fluido, de cambio de fase y de lo que
es dato, los mensajes al alumno, los sistemas de unidades, la teoría, el
procedimiento y la exportación.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.heat_transfer import exchangers as hx

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "16_Intercambiadores.py")

M_RATING = "Verificación: el intercambiador ya está (ε-NTU)"
M_SIZING = "Dimensionamiento: el área que hace falta"
M_TEST = "Ensayo: las cuatro temperaturas"
M_U = "El coeficiente global U"

RATING_NAMES = list(hx.RATING_EXAMPLES)
SIZING_NAMES = list(hx.SIZING_EXAMPLES)
TEST_NAMES = list(hx.TEST_EXAMPLES)
U_NAMES = list(hx.U_EXAMPLES)
C = 273.15


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
    return float(value.replace(" ", "").replace("%", "").replace(",", "."))


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _expanders(at: AppTest) -> list[str]:
    return [e.label for e in at.expander]


def _mode(at: AppTest, mode: str) -> AppTest:
    at.radio(key="hx_mode").set_value(mode).run()
    _no_problems(at)
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


def _wkey(at: AppTest, key: str) -> str:
    """La key real de un ``number_input_si`` (lleva el sistema de unidades)."""
    system = at.session_state["units_system"] if "units_system" in at.session_state else "Técnico"
    return f"{key}@{system}"


def _expected_charts(r: hx.ExchangerResult) -> int:
    """ε–NTU y la comparación siempre; F si no es un doble tubo (y C_r > 0); el perfil en el
    doble tubo."""
    double_pipe = r.arrangement.kind in ("parallel", "counter")
    return 2 + int(not double_pipe and r.C_r > 0.0) + int(double_pipe)


# ---------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------


def test_default_is_the_oil_cooler_of_cengel() -> None:
    at = _app()
    assert at.radio(key="hx_mode").value == M_RATING
    assert _metric(at, "Q̇ [") == pytest.approx(38_400.0, rel=2e-3)
    assert _metric(at, "Efectividad") == pytest.approx(0.462, abs=1e-3)
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert len(at.get("download_button")) == 2
    assert any(c.value.startswith("📘") for c in at.caption)


@pytest.mark.parametrize("name", RATING_NAMES)
def test_every_rating_example(name: str) -> None:
    at = _app()
    at.selectbox(key="xv_example").set_value(name).run()
    _no_problems(at)
    r = hx.solve_rating(hx.RATING_EXAMPLES[name].inputs)
    assert _metric(at, "Q̇ [") == pytest.approx(r.Q_W, rel=1e-4)
    assert _metric(at, "NTU") == pytest.approx(r.NTU, rel=1e-3)
    assert _metric(at, "Factor de corrección F") == pytest.approx(r.F, abs=1e-4)
    assert _charts(at) == _expected_charts(r)
    assert set(r.warnings) == {w.value for w in at.warning}
    streams = at.dataframe[0].value
    assert len(streams) == 2


def test_arrangement_changes() -> None:
    at = _app()
    at.selectbox(key="xv_example").set_value(RATING_NAMES[1]).run()  # contracorriente
    counter = _metric(at, "Q̇ [")
    at.selectbox(key="xv_1_kind").set_value("parallel").run()
    _no_problems(at)
    assert _metric(at, "Q̇ [") < counter
    at.selectbox(key="xv_1_kind").set_value("shell").run()
    at.number_input(key="xv_1_np").set_value(3).run()
    _no_problems(at)
    at.selectbox(key="xv_1_kind").set_value("cross_mixed").run()
    at.radio(key="xv_1_mixed").set_value("cold").run()
    _no_problems(at)
    assert counter > _metric(at, "Q̇ [")


def test_fluid_and_phase_change_inputs() -> None:
    at = _app()
    # el agua de enfriamiento del aceite, con CoolProp a 2 bar
    at.selectbox(key="xv_0_c_fluid").set_value("Water").run()
    _no_problems(at)
    q_coolprop = _metric(at, "Q̇ [")
    assert q_coolprop == pytest.approx(38_400.0, rel=0.01)
    # el aceite condensando (c_p dado: hace falta h_fg)
    at.checkbox(key="xv_0_h_pc").check().run()
    _no_problems(at)
    assert any(m.label.startswith("ṁ que cambia de fase") for m in at.metric)
    assert _metric(at, "C_r") == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Dimensionamiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", SIZING_NAMES)
def test_every_sizing_example(name: str) -> None:
    at = _mode(_app(), M_SIZING)
    at.selectbox(key="xd_example").set_value(name).run()
    _no_problems(at)
    r = hx.solve_sizing(hx.SIZING_EXAMPLES[name].inputs)
    assert _metric(at, "Área A") == pytest.approx(r.A_m2, rel=1e-4)
    if r.tube_length_m is not None:
        assert _metric(at, "Largo de tubo") == pytest.approx(r.tube_length_m, rel=1e-4)
    assert _charts(at) == _expected_charts(r)


def test_sizing_target_and_message() -> None:
    at = _mode(_app(), M_SIZING)
    assert _metric(at, "Largo de tubo") == pytest.approx(108.5, abs=0.1)
    at.radio(key="xd_0_target").set_value("Q").run()
    _no_problems(at)
    # la glicerina con un solo paso de casco no llega (P = ε máx.)
    at.selectbox(key="xd_example").set_value(SIZING_NAMES[1]).run()
    _no_problems(at)
    at.number_input(key="xd_1_np").set_value(1).run()
    assert at.error and "no puede pasar de" in at.error[0].value


# ---------------------------------------------------------------------
# Ensayo
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", TEST_NAMES)
def test_every_test_example(name: str) -> None:
    at = _mode(_app(), M_TEST)
    at.selectbox(key="xe_example").set_value(name).run()
    _no_problems(at)
    r = hx.solve_four_temperatures(hx.TEST_EXAMPLES[name].inputs)
    assert _metric(at, "Q̇ [") == pytest.approx(r.Q_W, rel=1e-4)
    assert _metric(at, "Factor de corrección F") == pytest.approx(r.F, abs=1e-4)
    if r.inputs.U_W_per_m2K is None:  # type: ignore[union-attr]
        assert _metric(at, "U [") == pytest.approx(r.U_W_per_m2K, rel=1e-4)


def test_test_with_a_known_flow() -> None:
    at = _mode(_app(), M_TEST)
    at.selectbox(key="xe_example").set_value(TEST_NAMES[2]).run()  # glicerina, U dato
    _no_problems(at)
    q_with_u = _metric(at, "Q̇ [")
    at.radio(key="xe_2_given").set_value("Un caudal (sale U)").run()
    _no_problems(at)
    assert any(m.label.startswith("U [") for m in at.metric)
    assert _metric(at, "Q̇ [") != pytest.approx(q_with_u)


# ---------------------------------------------------------------------
# U global
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", U_NAMES)
def test_every_u_example(name: str) -> None:
    at = _mode(_app(), M_U)
    at.selectbox(key="xu_example").set_value(name).run()
    _no_problems(at)
    r = hx.overall_u(hx.U_EXAMPLES[name].inputs)
    label = "U_i (área de adentro)" if r.inputs.geometry == "tube" else "U ["
    assert _metric(at, label) == pytest.approx(r.U_i_W_per_m2K, rel=1e-4)
    assert _charts(at) == 1
    table = at.dataframe[0].value
    assert list(table["Resistencia"]) == [x.label for x in r.resistances]


def test_u_geometry_and_fouling() -> None:
    at = _mode(_app(), M_U)
    assert _metric(at, "U_i (área de adentro)") == pytest.approx(399.3, abs=0.2)
    at.selectbox(key="xu_0_fi_sel").set_value("Limpio (sin ensuciamiento)").run()
    at.selectbox(key="xu_0_fo_sel").set_value("Limpio (sin ensuciamiento)").run()
    _no_problems(at)
    assert _metric(at, "U_i (área de adentro)") == pytest.approx(
        _metric(at, "U_i limpio"), rel=1e-4
    )
    at.radio(key="xu_0_geo").set_value("plane").run()
    _no_problems(at)
    assert any(m.label.startswith("U [") for m in at.metric)


# ---------------------------------------------------------------------
# Unidades, teoría y procedimiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_other_unit_systems(system: str) -> None:
    unit = {"SI": "W", "Inglés": "Btu/h"}[system]
    for mode in (M_RATING, M_SIZING, M_TEST):
        at = _system(_mode(_app(), mode), system)
        _no_problems(at)
        label = next(m.label for m in at.metric if m.label.startswith("Q̇ ["))
        assert label == f"Q̇ [{unit}]"
        steps = at.expander[1]
        assert steps.label == "🔬 Procedimiento"
        assert len(steps.get("latex")) >= 8


def test_oil_cooler_in_english_units() -> None:
    at = _system(_app(), "Inglés")
    assert _metric(at, "Q̇ [") == pytest.approx(38_400.0 * 3.412141633, rel=2e-3)


def test_theory_cites_the_sources() -> None:
    at = _app()
    theory = at.expander[0]
    assert theory.label == "📖 Fórmulas teóricas"
    text = " ".join(m.value for m in theory.markdown)
    assert "vademecum" in text and "§11.10" in text and "§13.2" in text
    assert "Çengel y Ghajar" in text and "Incropera" in text and "Mason" in text
    assert len(theory.get("latex")) >= 15
