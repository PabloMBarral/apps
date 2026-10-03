"""Tests de la página de ISO 6976 con ``streamlit.testing`` (AppTest).

Cubren lo agregado en la Fase 1.7: el expansor `📖 Fórmulas teóricas`
(vademecum §16.9 + fórmulas de la norma) y la descarga del resultado en
CSV / JSON, además de que los tres ejemplos del Annex D calculen.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "4_ISO6976.py")

EXAMPLES = (
    "Annex D.2 — 5 componentes (15/15 °C)",
    "Annex D.3 — 5 componentes con vapor de agua (15.55/15.55 °C)",
    "Annex D.4 — 11 componentes (15/15 °C, caso A)",
    "Personalizado (componer manualmente)",
)


def _new_app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _compute(at: AppTest) -> AppTest:
    at.button(key="iso_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def _result_tables(at: AppTest) -> list:
    """Tablas de resultados (el editor de composición también es un dataframe)."""
    return [df for df in at.dataframe if not (df.key or "").startswith("iso_composition_editor")]


def test_theory_expander_cites_the_vademecum() -> None:
    at = _new_app()
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    assert any("§16.9" in md.value for md in theory.markdown)
    assert len(theory.latex) == 5


@pytest.mark.parametrize("example", EXAMPLES)
def test_examples_compute_and_export(example: str) -> None:
    at = _new_app()
    at.selectbox(key="iso_example_label").set_value(example).run()
    assert not at.get("download_button")
    _compute(at)
    assert len(_result_tables(at)) == 2  # intermedios + resultados
    assert len(at.get("download_button")) == 2


def test_changing_example_clears_the_result() -> None:
    at = _compute(_new_app())
    at.selectbox(key="iso_example_label").set_value(EXAMPLES[2]).run()
    assert not at.get("download_button")
    assert not _result_tables(at)
