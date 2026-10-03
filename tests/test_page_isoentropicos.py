"""Tests de la página de Isoentrópicos con ``streamlit.testing`` (AppTest).

Regresiones de la versión 0.9.0 de la página:

- con R410A y CO₂ la bomba fallaba con los valores iniciales (errores en
  inglés), y con R134a la turbina "calculaba" a 400 °C, por encima de la
  T máxima de su ecuación de estado (182 °C), sin aviso;
- al cambiar de fluido seguía mostrándose el resultado del fluido
  anterior (y el diagrama mezclaba isolíneas de un fluido con estados de
  otro);
- la tabla mostraba x = -1 fuera de la campana;
- los pasos LaTeX estaban siempre en sistema Técnico (Fase 1.5b).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.fluids import SUPPORTED_FLUIDS

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "3_Isoentropicos.py")
BUTTONS = ("turb_btn", "comp_btn", "pump_btn", "poly_btn")


def _new_app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def _click(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
def test_every_tab_computes_with_defaults(fluid: str) -> None:
    at = _new_app()
    at.selectbox(key="iso_fluid").set_value(fluid).run()
    for key in BUTTONS:
        _click(at, key)
        assert not at.error, (fluid, key, [e.value for e in at.error])
    assert len(at.dataframe) == 4  # una tabla de estados por pestaña


@pytest.mark.parametrize("prefix", ["turb", "comp", "pump"])
def test_inverse_mode_defaults_recover_eta(prefix: str) -> None:
    at = _new_app()
    at.radio(key=f"{prefix}_mode").set_value("Inverso (calcular η_s dados los estados)").run()
    _click(at, f"{prefix}_btn")
    assert not at.error
    eta = float(next(m.value for m in at.metric if m.label == "η_s"))
    assert 0.7 <= eta <= 0.95


def test_result_of_another_fluid_is_not_shown() -> None:
    at = _new_app()
    _click(at, "turb_btn")
    assert any(m.label == "η_s" for m in at.metric)
    at.selectbox(key="iso_fluid").set_value("Air").run()
    assert not any(m.label == "η_s" for m in at.metric)


def test_states_table_shows_region_and_no_minus_one() -> None:
    at = _click(_new_app(), "turb_btn")
    frame = at.dataframe[0].value
    assert list(frame["Estado"]) == [
        "1 (entrada)",
        "2s (salida isoentrópica)",
        "2 (salida real)",
    ]
    assert frame["x [-]"].iloc[0] == "—"  # vapor sobrecalentado
    assert frame["Región"].iloc[2].startswith("Vapor húmedo")


def test_procedure_follows_the_unit_system() -> None:
    at = _new_app()
    at.selectbox(key="units_system").set_value("Inglés").run()
    _click(at, "turb_btn")
    assert any(r"\mathrm{Btu/lb}" in tex.value for tex in at.latex)


def test_liquid_inlet_in_compressor_is_explained() -> None:
    at = _new_app()  # agua: el default del compresor es vapor a 120 °C y 1 bar
    at.number_input(key="comp_in_Water_v1@Técnico").set_value(25.0)  # 25 °C → líquido
    _click(at, "comp_btn")
    assert at.error and "bomba" in at.error[0].value


def test_intercooler_below_saturation_is_explained() -> None:
    at = _new_app()  # agua, 2 etapas de 1 a 9 bar
    at.number_input(key="poly_tic_Water@Técnico").set_value(25.0)
    _click(at, "poly_btn")
    assert at.error and "condensaría" in at.error[0].value


def test_pump_comparison() -> None:
    at = _click(_new_app(), "pump_btn")
    _click(at, "pump_compare_btn")
    labels = [m.label for m in at.metric]
    assert "Error relativo" in labels


def test_theory_and_export_are_present() -> None:
    at = _click(_new_app(), "turb_btn")
    assert any("Fórmulas teóricas" in e.label for e in at.expander)
    assert len(at.get("download_button")) == 2
