"""Tests de la página de Interpolación con ``streamlit.testing`` (AppTest).

Cubren lo agregado en la Fase 1.7: el expansor `📖 Fórmulas teóricas`
(vademecum §2.1 y §2.2) y la descarga del resultado en CSV / JSON en las
dos pestañas, además de que los ejemplos precargados calculen.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "2_Interpolacion.py")


def _new_app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _click(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def test_theory_expander_cites_the_vademecum() -> None:
    at = _new_app()
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    text = " ".join(md.value for md in theory.markdown)
    assert "§2.1" in text and "§2.2" in text
    assert len(theory.latex) == 4  # simple + las tres de la doble


def test_no_export_before_computing() -> None:
    assert not _new_app().get("download_button")


def test_linear_example_computes_and_exports() -> None:
    at = _click(_new_app(), "lin_btn")  # x = 115 °C, punto medio de la tabla
    assert any("h_f [kJ/kg] = 482.6" in s.value for s in at.success)
    assert len(at.get("download_button")) == 2


@pytest.mark.parametrize("button", ["lin_btn", "bil_btn"])
def test_each_tab_has_its_own_export(button: str) -> None:
    at = _click(_new_app(), button)
    assert len(at.get("download_button")) == 2
    _click(at, "bil_btn" if button == "lin_btn" else "lin_btn")
    assert len(at.get("download_button")) == 4  # dos por pestaña, keys distintas


def test_result_survives_other_widgets() -> None:
    """El resultado (y su export) no desaparece al tocar otro widget."""
    at = _click(_new_app(), "lin_btn")
    at.checkbox(key="bil_extrap").check().run()
    assert any("Resultado" in s.value for s in at.success)
    assert len(at.get("download_button")) == 2
