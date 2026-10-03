"""Tests de la página de Propiedades con ``streamlit.testing`` (AppTest).

Regresiones de los bugs encontrados en la versión 0.8.0 de la página:

- el cálculo inicial ("t y p" con 0 °C y 1 bar) fallaba con un error de
  CoolProp en inglés (0 °C está debajo del punto triple);
- al cambiar el tipo de diagrama, los resultados desaparecían (solo se
  mostraban en la corrida del submit) y el selector volvía a log p–h;
- elegir p–log v lanzaba una excepción.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.state_report import PAIR_ORDER

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "1_Propiedades.py")


def _new_app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def _submit(at: AppTest) -> AppTest:
    next(b for b in at.button if b.label == "Calcular estado").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _metric(at: AppTest, label_prefix: str) -> str:
    return next(m.value for m in at.metric if m.label.startswith(label_prefix))


def _region_text(at: AppTest) -> str:
    return at.info[0].value


@pytest.fixture(scope="module")
def computed() -> AppTest:
    """Página con el estado por defecto calculado (agua, T-p)."""
    return _submit(_new_app())


def test_page_loads_with_prompt() -> None:
    at = _new_app()
    assert "Calcular estado" in at.info[0].value
    assert not at.metric


def test_default_state_is_valid_superheated_steam(computed: AppTest) -> None:
    # 300 °C y 10 bar: Cengel A-6.
    assert not computed.error
    assert "Vapor sobrecalentado" in _region_text(computed)
    assert _metric(computed, "Temperatura T [°C]") == "300"
    assert _metric(computed, "Entalpía h [kJ/kg]") == "3051.6"


def test_procedure_and_tables_are_rendered(computed: AppTest) -> None:
    assert len(computed.latex) > 5
    assert len(computed.dataframe) >= 5  # 4 grupos de propiedades + saturación


@pytest.mark.parametrize("diagram_type", ["plogv", "logph", "hs", "Ts"])
def test_changing_diagram_keeps_results(diagram_type: str) -> None:
    at = _submit(_new_app())
    at.selectbox(key="prop_diagram_type").set_value(diagram_type).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric, "los resultados desaparecieron al cambiar el diagrama"
    assert at.selectbox(key="prop_diagram_type").value == diagram_type
    assert len(at.get("plotly_chart")) == 1


def test_switching_units_keeps_results() -> None:
    at = _submit(_new_app())
    at.selectbox(key="units_system").set_value("Inglés").run()
    assert not at.exception
    assert _metric(at, "Temperatura T [°F]") == "572"


@pytest.mark.parametrize("pair", PAIR_ORDER)
def test_every_pair_default_computes(pair: str) -> None:
    at = _new_app()
    at.selectbox(key="prop_pair").set_value(pair).run()
    _submit(at)
    assert not at.error, [e.value for e in at.error]
    assert at.metric


def test_student_error_is_shown_in_spanish() -> None:
    at = _new_app()
    at.number_input(key="prop_in_Water_TP_t@Técnico").set_value(0.0)
    _submit(at)
    assert at.error
    assert "punto triple" in at.error[0].value
    assert "hielo" in at.error[0].value


def test_tp_on_saturation_line_warns() -> None:
    at = _new_app()
    at.number_input(key="prop_in_Water_TP_t@Técnico").set_value(100.0)
    at.number_input(key="prop_in_Water_TP_p@Técnico").set_value(1.01325)
    _submit(at)
    assert any("no son independientes" in w.value for w in at.warning)


def test_other_fluid_defaults_compute() -> None:
    at = _new_app()
    at.selectbox(key="prop_fluid").set_value("R134a").run()
    _submit(at)
    assert not at.error
    assert at.metric
