"""Tests de la página del ciclo combinado con ``streamlit.testing`` (AppTest) — Fase 3.4."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.cycles.combined import COMBINED_EXAMPLES, CombinedInputs, solve_combined
from core.ideal_gas import AIR_TECHNICAL

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "12_Ciclo_Combinado.py")
EXAMPLES = list(COMBINED_EXAMPLES)
TYPICAL = COMBINED_EXAMPLES[EXAMPLES[0]]
K = "cc_0"  # prefijo de las keys del ejemplo 0 (el ciclo combinado típico)
ETA = "η ciclo combinado [%]"
W_NET = "Ẇ neto [kW]"
STEAM = "ṁ vapor [kg/s]"


def _new_app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _frame(at: AppTest, column: str):  # noqa: ANN202 — DataFrame de la tabla con esa columna
    return next(df.value for df in at.dataframe if column in df.value.columns)


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert not any("No se pudo dibujar" in w.value for w in at.warning)
    assert not any("TESPy no pudo" in w.value for w in at.warning)


def _calculate(at: AppTest) -> AppTest:
    return at.button(key="cc_btn").click().run()


def _assert_result(at: AppTest, inputs: CombinedInputs) -> None:
    _no_problems(at)
    expected = solve_combined(inputs)
    assert float(_metric(at, ETA)) == pytest.approx(expected.eta_th * 100, abs=0.006)
    assert float(_metric(at, STEAM)) == pytest.approx(expected.hrsg.m_steam_kg_s, rel=1e-4)
    assert not any(c.value.startswith("Cambiaste datos") for c in at.caption)


def test_default_example_on_load() -> None:
    at = _new_app()
    _no_problems(at)
    assert _metric(at, ETA) == "55.61"
    assert _metric(at, W_NET) == "306 188"
    assert _metric(at, "Q̇ combustible [kW]") == "550 590"
    assert _metric(at, "η turbina de gas [%]") == "39.27"
    assert _metric(at, "Ẇ turbina de gas [kW]") == "216 229"
    assert _metric(at, "T escape [°C]") == "608.6"
    assert _metric(at, "η ciclo de vapor [%]") == "36.73"
    assert _metric(at, "Ẇ ciclo de vapor [kW]") == "89 958"
    assert float(_metric(at, STEAM)) == pytest.approx(80.488, abs=1e-3)
    # Sankey, T–s de la turbina de gas, T–Q de la HRSG y diagrama del ciclo de vapor
    assert len(at.get("plotly_chart")) == 4
    gas = _frame(at, "Dónde")
    assert list(gas["Estado"]) == ["1", "2s", "2", "3", "4s", "4"]
    assert list(gas["Medio"]) == ["aire"] * 3 + ["gases"] * 3
    sections = _frame(at, "Sección")
    assert list(sections["Sección"]) == ["sobrecalentador", "evaporador", "economizador"]
    steam = _frame(at, "Región")
    assert len(steam) == 7  # con el desaireador
    assert steam["Estado"][4] == "5 (entrada a la turbina)"
    notes = " ".join(i.value for i in at.info)
    assert notes.startswith("El ciclo de vapor sube el rendimiento")
    assert any("λ = 2.636" in c.value for c in at.caption)


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_example_computes(example: str) -> None:
    at = _new_app()
    at.selectbox(key="cc_example").set_value(example).run()
    _assert_result(at, COMBINED_EXAMPLES[example])
    assert any(c.value.startswith("📘 Sobre este ejemplo") for c in at.caption)


def test_cengel_example_is_air_standard_and_designed_by_stack() -> None:
    at = _new_app()
    at.selectbox(key="cc_example").set_value(EXAMPLES[1]).run()
    _no_problems(at)
    assert _metric(at, ETA) == "48.70"
    assert any(m.label == "Q̇ entrada [kW]" for m in at.metric)  # sin combustible
    assert at.radio(key="cc_1_model").value.startswith("Aire estándar")
    assert at.radio(key="cc_1_design").value == "Por temperatura de chimenea"
    assert at.number_input(key="cc_1_Tstack@Técnico").value == pytest.approx(176.85)
    assert any("la chimenea es dato y el pinch sale" in c.value for c in at.caption)
    assert "Temperatura de chimenea" in at.selectbox(key="cc_sweep_param").options


def test_changed_data_wait_for_the_button() -> None:
    at = _new_app()
    at.number_input(key=f"{K}_rp").set_value(20.0).run()
    assert _metric(at, ETA) == "55.61"  # todavía el resultado anterior
    assert any(c.value.startswith("Cambiaste datos") for c in at.caption)
    _calculate(at)
    gt = replace(TYPICAL.gas_turbine, pressure_ratio=20.0)
    _assert_result(at, replace(TYPICAL, gas_turbine=gt))


def test_air_standard_model() -> None:
    at = _new_app()
    at.radio(key=f"{K}_model").set_value("Aire estándar (sin combustible)").run()
    _calculate(at)
    _assert_result(at, replace(TYPICAL, gas_turbine=replace(TYPICAL.gas_turbine, fuel=None)))
    assert any(m.label == "Q̇ entrada [kW]" for m in at.metric)
    assert not any("ṁ combustible" in c.value for c in at.caption)
    control = _frame(at, "TESPy")
    assert "λ" not in list(control["Magnitud"])


def test_technical_air() -> None:
    at = _new_app()
    at.radio(key=f"{K}_air").set_value("Técnico (21 % O₂, 79 % N₂)").run()
    _calculate(at)
    gt = replace(TYPICAL.gas_turbine, air=AIR_TECHNICAL)
    _assert_result(at, replace(TYPICAL, gas_turbine=gt))


def test_saturated_steam() -> None:
    at = _new_app()
    at.radio(key=f"{K}_steam").set_value("Saturado").run()
    assert not any(n.key == f"{K}_Ts@Técnico" for n in at.number_input)
    _calculate(at)
    _assert_result(at, replace(TYPICAL, T_steam_K=None))
    assert list(_frame(at, "Sección")["Sección"]) == ["evaporador", "economizador"]


def test_without_deaerator() -> None:
    at = _new_app()
    at.checkbox(key=f"{K}_da").uncheck().run()
    _calculate(at)
    _assert_result(at, replace(TYPICAL, steam=replace(TYPICAL.steam, deaerator_p_Pa=None)))
    assert len(_frame(at, "Región")) == 4  # bomba, caldera, turbina y condensador


def test_stack_design_starts_from_a_valid_stack_temperature() -> None:
    at = _new_app()
    at.radio(key=f"{K}_design").set_value("Por temperatura de chimenea").run()
    # la chimenea del diseño por pinch (181,6 °C), redondeada hacia arriba a 5 °C
    assert at.number_input(key=f"{K}_Tstack@Técnico").value == pytest.approx(185.0)
    _calculate(at)
    _assert_result(at, replace(TYPICAL, T_stack_K=185.0 + 273.15))
    assert "Temperatura de chimenea" in at.selectbox(key="cc_sweep_param").options
    assert "Pinch" not in at.selectbox(key="cc_sweep_param").options


def test_size_by_net_power() -> None:
    at = _new_app()
    at.radio(key=f"{K}_size").set_value("Potencia neta total").run()
    _calculate(at)
    _no_problems(at)
    assert _metric(at, W_NET) == "300 000"
    assert _metric(at, ETA) == "55.61"  # los rendimientos no cambian con el tamaño


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        (f"{K}_TIT@Técnico", 350.0, "Turbina de gas: La temperatura de entrada a la turbina"),
        (f"{K}_Ts@Técnico", 650.0, "Caldera de recuperación: El vapor"),
        (f"{K}_pda@Técnico", 70.0, "desaireador"),
    ],
)
def test_invalid_data_are_explained(key: str, value: float, match: str) -> None:
    at = _new_app()
    at.number_input(key=key).set_value(value).run()
    _calculate(at)
    assert not at.exception
    assert at.error and match in at.error[0].value


@pytest.mark.parametrize(
    ("system", "labels", "unit"),
    [
        ("SI", ("Ẇ neto [W]", "T escape [K]"), r"\mathrm{J/kg}"),
        ("Inglés", ("ṁ vapor [lb/s]", "T escape [°F]"), r"\mathrm{Btu/lb}"),
    ],
)
def test_unit_systems(system: str, labels: tuple[str, ...], unit: str) -> None:
    at = _new_app()
    at.selectbox(key="units_system").set_value(system).run()
    _no_problems(at)
    metric_labels = [m.label for m in at.metric]
    assert all(label in metric_labels for label in labels)
    assert any(unit in tex.value for tex in at.latex)
    assert _metric(at, ETA) == "55.61"
    if system == "SI":
        assert float(_metric(at, "T escape [K]")) == pytest.approx(881.8, abs=0.1)


def test_tespy_control_matches() -> None:
    at = _new_app()
    control = _frame(at, "TESPy")
    assert list(control.columns) == ["Magnitud", "Gases ideales", "TESPy"]
    pairs = zip(control["Gases ideales"], control["TESPy"], strict=True)
    rows = dict(zip(control["Magnitud"], pairs, strict=True))
    ideal, tespy = (float(v) for v in rows["η turbina de gas [%]"])
    assert tespy == pytest.approx(ideal, abs=0.1)
    ideal, tespy = (float(v) for v in rows["λ"])
    assert tespy == pytest.approx(ideal, rel=3e-3)


def test_theory_procedure_and_export_are_present() -> None:
    at = _new_app()
    labels = [e.label for e in at.expander]
    for name in ("Fórmulas teóricas", "Procedimiento", "Control con TESPy", "Diagrama del ciclo"):
        assert any(name in label for label in labels), name
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    text = " ".join(md.value for md in theory.markdown)
    for cite in ("§3.3", "§5", "§10.4", "§16", "§9-8", "§10-9", "Kehlhofer", "ISO 6976:2016"):
        assert cite in text, cite
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    titles = [md.value for md in procedure.markdown]
    for part in ("Turbina de gas", "Caldera de recuperación", "Ciclo de vapor", "Ciclo combinado"):
        assert f"##### {part}" in titles, part
    assert len(procedure.latex) > 50
    assert len(at.get("download_button")) == 2


@pytest.mark.parametrize(
    ("variable", "log_axis"),
    [
        ("Relación de presiones", True),
        ("Temperatura de entrada a la turbina", False),
        ("Presión del vapor", True),
        ("Pinch", False),
    ],
)
def test_sweeps_draw_the_chart(variable: str, log_axis: bool) -> None:
    at = _new_app()
    at.selectbox(key="cc_sweep_param").set_value(variable).run()
    at.button(key="cc_sweep_btn").click().run()
    _no_problems(at)
    charts = at.get("plotly_chart")
    assert len(charts) == 5  # los cuatro diagramas + el barrido
    sweep = next(c for c in charts if "Rendimientos" in c.proto.spec)
    assert ('"type":"log"' in sweep.proto.spec) == log_axis
    assert any("óptimo" in c.value for c in at.caption) == (variable == "Relación de presiones")
