"""Tests de la página de ISO 6976 con ``streamlit.testing`` (AppTest).

Cubren lo agregado en la Fase 1.7: el expansor `📖 Fórmulas teóricas`
(vademecum §16.9 + fórmulas de la norma) y la descarga del resultado en
CSV / JSON, además de que los ejemplos del Annex D calculen. Desde la 0.30.0, la
matriz de normalización: el caso B del D.4, las u del análisis crudo, la
comparación con la identidad y el error con u(x) que no salen de normalizar.
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
    "Annex D.4 — 11 componentes (15/15 °C, caso B: normalización)",
    "Personalizado (componer manualmente)",
)
CASE_B = EXAMPLES[3]


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
    # vademecum, tres de la norma, la identidad y las tres de la normalización
    assert len(theory.latex) == 8
    assert any("14912" in md.value for md in theory.markdown)


@pytest.mark.parametrize("example", EXAMPLES)
def test_examples_compute_and_export(example: str) -> None:
    at = _new_app()
    at.selectbox(key="iso_example_label").set_value(example).run()
    assert not at.get("download_button")
    _compute(at)
    # intermedios + resultados (+ el análisis crudo y la comparación con la normalización)
    assert len(_result_tables(at)) == (4 if example == CASE_B else 2)
    assert len(at.get("download_button")) == 2


def test_changing_example_clears_the_result() -> None:
    at = _compute(_new_app())
    at.selectbox(key="iso_example_label").set_value(EXAMPLES[2]).run()
    assert not at.get("download_button")
    assert not _result_tables(at)


def test_case_b_uses_the_normalization_matrix() -> None:
    """El caso B del D.4 elige la normalización con las u normalizadas y reproduce las u
    de la norma (§D.4.3.2: u(H_v,G) = 0,016316 MJ/m³)."""
    at = _new_app()
    at.selectbox(key="iso_example_label").set_value(CASE_B).run()
    assert at.radio(key="iso_corr").value == "normalization"
    assert at.radio(key="iso_basis").value == "normalized"
    _compute(at)
    results = _result_tables(at)[1].value
    hv = results[results["magnitud"].str.startswith("H_v,G")].iloc[0]
    assert hv["u (k=1)"] == pytest.approx(0.016316, abs=1e-5)
    raw = _result_tables(at)[2].value
    assert list(raw.columns) == [
        "componente",
        "x",
        "u(y) crudo",
        "u(x) normalizada",
        "r con methane",
    ]
    assert raw.iloc[0]["u(y) crudo"] == pytest.approx(6.3447e-4, rel=1e-4)
    assert raw.iloc[0]["u(x) normalizada"] == pytest.approx(0.000348, rel=1e-9)
    assert raw.iloc[1]["r con methane"] < -0.6  # el etano
    comparison = _result_tables(at)[3].value
    # en el celular, el cambio a la vista (la unidad va con la magnitud)
    assert list(comparison.columns) == ["magnitud", "cambio", "u identidad", "u normalización"]
    assert comparison.iloc[0]["magnitud"] == "H_v,G [MJ/m³]"
    assert comparison.iloc[3]["magnitud"] == "d"
    assert comparison.iloc[0]["cambio"] == "-39 %"
    assert comparison.iloc[0]["u identidad"] == pytest.approx(0.026917, abs=1e-5)
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    assert len(procedure.latex) == 5  # fórmulas, valores y las tres de la normalización


def test_back_to_identity_hides_the_raw_analysis() -> None:
    at = _new_app()
    at.selectbox(key="iso_example_label").set_value(CASE_B).run()
    at.radio(key="iso_corr").set_value("identity").run()
    assert not at.radio(key="iso_corr").value == "normalization"
    assert not [r for r in at.radio if r.key == "iso_basis"]
    _compute(at)
    assert len(_result_tables(at)) == 2
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    assert len(procedure.latex) == 3


def test_raw_basis_with_the_default_example() -> None:
    """Las u del D.2 tomadas como las del análisis crudo: las normalizadas salen un
    poco menores (la del metano, 0,000339 contra 0,000346 cruda)."""
    at = _new_app()
    at.radio(key="iso_corr").set_value("normalization").run()
    at.radio(key="iso_basis").set_value("raw").run()
    _compute(at)
    raw = _result_tables(at)[2].value
    assert raw.iloc[0]["u(y) crudo"] == pytest.approx(0.000346, rel=1e-9)
    assert raw.iloc[0]["u(x) normalizada"] == pytest.approx(0.0003389, rel=1e-3)


def test_a_normalization_error_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un error de la normalización (u(x) que no salen de normalizar, ver
    tests/test_iso6976.py) se muestra sin resultado. AppTest no edita el data_editor:
    el error real se reemplaza por uno con las u(x) de dos componentes distintas."""
    from core.combustion import iso6976

    real = iso6976.normalization_fit

    def inconsistent(composition, basis="normalized"):  # type: ignore[no-untyped-def]
        if basis == "raw":
            return real(composition, basis)
        return real(
            [
                iso6976.GasComponent("methane", 0.95, 0.001),
                iso6976.GasComponent("ethane", 0.05, 0.0005),
            ]
        )

    monkeypatch.setattr(iso6976, "normalization_fit", inconsistent)
    at = _new_app()
    at.radio(key="iso_corr").set_value("normalization").run()
    at.button(key="iso_btn").click().run()
    assert not at.exception
    assert at.error and "iguales" in at.error[0].value
    assert not at.get("download_button")
    # con las del análisis crudo, cualquier u vale
    at.radio(key="iso_basis").set_value("raw").run()
    _compute(at)
    assert len(_result_tables(at)) == 4
