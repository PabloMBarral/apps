"""Tests de :mod:`ui.units_ui` con ``streamlit.testing`` (AppTest).

Regresión: con Streamlit ≥ 1.4x los widgets se identifican solo por su
key, así que un ``number_input`` con key fija conservaba el número al
cambiar de sistema de unidades, pero con la unidad nueva (400 °C pasaba
a leerse como 400 K). ``number_input_si`` ahora usa una key por sistema
y arrastra el valor convertido.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


def _app() -> None:
    import streamlit as st

    from ui.units_ui import number_input_si, render_units_selector

    render_units_selector()
    value_si = number_input_si(
        label="Temperatura", kind="temperature", default_si=673.15, key="t_in"
    )
    st.markdown(f"SI={value_si:.6f}")


def _si_value(at: AppTest) -> float:
    text = next(m.value for m in at.markdown if m.value.startswith("SI="))
    return float(text.removeprefix("SI="))


def test_default_is_shown_in_tecnico() -> None:
    at = AppTest.from_function(_app).run()
    widget = at.number_input(key="t_in@Técnico")
    assert widget.label == "Temperatura [°C]"
    assert widget.value == pytest.approx(400.0)
    assert _si_value(at) == pytest.approx(673.15)


def test_switching_system_keeps_the_physical_value() -> None:
    at = AppTest.from_function(_app).run()
    at.number_input(key="t_in@Técnico").set_value(450.0).run()

    at.selectbox(key="units_system").set_value("SI").run()
    assert at.number_input(key="t_in@SI").value == pytest.approx(723.15)
    assert _si_value(at) == pytest.approx(723.15)

    at.selectbox(key="units_system").set_value("Inglés").run()
    assert at.number_input(key="t_in@Inglés").value == pytest.approx(842.0)
    assert _si_value(at) == pytest.approx(723.15)

    at.selectbox(key="units_system").set_value("Técnico").run()
    assert at.number_input(key="t_in@Técnico").value == pytest.approx(450.0)
