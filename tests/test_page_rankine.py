"""Tests de la página de Rankine con ``streamlit.testing`` (AppTest) — Fase 3.1a."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.cycles.rankine import RANKINE_EXAMPLES

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "5_Rankine.py")
EXAMPLES = list(RANKINE_EXAMPLES)


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


def test_default_example_is_computed_on_load() -> None:
    at = _new_app()
    _no_problems(at)
    assert float(_metric(at, "η térmico [%]")) == pytest.approx(26.0, abs=0.2)  # Cengel 10-1
    assert _metric(at, "Título a la salida") == "0.886"
    assert len(_states(at)) == 4


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_example_computes(example: str) -> None:
    at = _new_app()
    at.selectbox(key="rk_example").set_value(example).run()
    _no_problems(at)
    expected_states = 6 if RANKINE_EXAMPLES[example].reheat else 4
    assert len(_states(at)) == expected_states


def test_reheat_checkbox_adds_two_states() -> None:
    at = _new_app()
    at.checkbox(key="rk_reheat_0").check().run()
    at.button(key="rk_btn").click().run()
    _no_problems(at)
    assert len(_states(at)) == 6


def test_condenser_label_follows_the_state_numbering() -> None:
    at = _new_app()
    condenser = at.number_input(key="rk_p_lo_0@Técnico")
    assert condenser.label.startswith("Presión del condensador p₄")
    condenser.set_value(0.2).run()
    at.checkbox(key="rk_reheat_0").check().run()
    condenser = at.number_input(key="rk_p_lo_0@Técnico")
    assert condenser.label.startswith("Presión del condensador p₆")  # 5 → 6: turbina de baja
    assert condenser.value == pytest.approx(0.2)  # cambiar el rótulo no pierde el dato


def test_saturated_inlet_hides_the_temperature() -> None:
    at = _new_app()
    at.radio(key="rk_inlet_0").set_value("Vapor saturado seco (x₃ = 1)").run()
    assert not any(n.key == "rk_T_in_0@Técnico" for n in at.number_input)
    at.button(key="rk_btn").click().run()
    _no_problems(at)
    assert _states(at)[2].startswith("3 ")


def test_net_power_gives_the_mass_flow() -> None:
    at = _new_app()
    at.radio(key="rk_flow_0").set_value("Potencia neta").run()
    at.button(key="rk_btn").click().run()
    _no_problems(at)
    assert _metric(at, "Ẇ neto [kW]") == "100 000"  # 100 MW por defecto, sin "1e+05"


def test_invalid_condenser_pressure_is_explained() -> None:
    at = _new_app()
    at.number_input(key="rk_p_lo_0@Técnico").set_value(40.0)  # bar, más que la caldera
    at.button(key="rk_btn").click().run()
    assert at.error and "menor que la de la caldera" in at.error[0].value


def test_edited_data_ask_for_recalculation() -> None:
    at = _new_app()
    at.number_input(key="rk_eta_t_0").set_value(0.85).run()
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
    assert "§9.2" in text and "§10.4" in text
    assert len(at.get("download_button")) == 2


def test_result_survives_changing_the_diagram() -> None:
    at = _new_app()
    at.selectbox(key="rk_diagram_type").set_value("logph").run()
    _no_problems(at)
    assert float(_metric(at, "η térmico [%]")) == pytest.approx(26.0, abs=0.2)


def test_sweep_draws_the_charts() -> None:
    at = _new_app()
    at.selectbox(key="rk_sweep_param").set_value("Presión de la caldera").run()
    at.button(key="rk_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) >= 3  # diagrama + η + título
