"""Tests de /Brayton: turbina de gas con interenfriamiento, recalentamiento y regenerador (AppTest).

Fase 3.7. Cada ejemplo, las etapas, el regenerador, las presiones intermedias,
el tamaño por potencia, los mensajes al alumno, los sistemas de unidades, el
procedimiento, la teoría, el control con TESPy, la exportación y los barridos.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.cycles.gas_turbine import (
    GAS_TURBINE_EXAMPLES,
    GasTurbineInputs,
    gas_turbine_tespy,
    solve_gas_turbine,
)

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "13_Brayton.py")
NAMES = list(GAS_TURBINE_EXAMPLES)
ETA = "η turbina de gas [%]"
C = 273.15


def _app(name: str | None = None) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    if name is not None:
        at.selectbox(key="bt_example").set_value(name).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert not any("No se pudo dibujar" in w.value for w in at.warning)


def _check(at: AppTest, inputs: GasTurbineInputs) -> None:
    result = solve_gas_turbine(inputs)
    assert float(_metric(at, ETA)) == pytest.approx(result.eta_th * 100, abs=0.006)
    assert float(_metric(at, "r_bw = w_C/w_T")) == pytest.approx(result.back_work_ratio, abs=6e-4)


def _key(name: str) -> str:
    return f"bt_{NAMES.index(name)}"


@pytest.mark.parametrize("name", NAMES)
def test_every_example(name: str) -> None:
    at = _app(name)
    _no_problems(at)
    inputs = GAS_TURBINE_EXAMPLES[name]
    _check(at, inputs)
    result = solve_gas_turbine(inputs)
    # T–s, comparación de las mejoras y exergía
    assert len(at.get("plotly_chart")) == 3
    states = next(df.value for df in at.dataframe if "Estado" in df.value.columns)
    assert list(states["Estado"]) == [s.label for s in result.states]
    comparison = next(df.value for df in at.dataframe if "Configuración" in df.value.columns)
    assert len(comparison) == 6
    assert sum("◆" in c for c in comparison["Configuración"]) == (
        1 if any(r.current for r in _comparison(inputs)) else 0
    )
    assert any(c.value.startswith("📘 Sobre este ejemplo") for c in at.caption)


def _comparison(inputs: GasTurbineInputs):  # noqa: ANN202
    from core.cycles.gas_turbine import improvement_comparison

    return improvement_comparison(inputs)


def test_default_is_cengel_9_5() -> None:
    at = _app()
    _no_problems(at)
    assert at.selectbox(key="bt_example").value == NAMES[0]
    assert _metric(at, ETA) == "42.54"


def test_stages_and_regenerator_inputs() -> None:
    name = NAMES[0]
    k = _key(name)
    at = _app(name)
    at.radio(key=f"{k}_nc").set_value("2 etapas").run()
    at.radio(key=f"{k}_nt").set_value("2 etapas").run()
    at.checkbox(key=f"{k}_regen").check().run()
    at.number_input(key=f"{k}_eps").set_value(0.9).run()
    at.button(key="bt_btn").click().run()
    _no_problems(at)
    base = GAS_TURBINE_EXAMPLES[name]
    expected = replace(
        base,
        compressor_stages=2,
        turbine_stages=2,
        T_intercool_K=300.0,
        T_reheat_K=1300.0,
        regenerator=0.9,
    )
    _check(at, expected)
    states = next(df.value for df in at.dataframe if "Estado" in df.value.columns)
    assert list(states["Estado"])[-1] == "10"


def test_given_intermediate_pressure() -> None:
    name = NAMES[3]  # 9-8 a): dos etapas de compresión y de expansión
    k = _key(name)
    at = _app(name)
    at.radio(key=f"{k}_pic_mode").set_value("A elección").run()
    at.number_input(key=f"{k}_pic2_0@Técnico").set_value(2.0).run()
    at.button(key="bt_btn").click().run()
    _no_problems(at)
    base = GAS_TURBINE_EXAMPLES[name]
    _check(at, replace(base, T_intercool_K=300.0, T_reheat_K=1300.0, p_intercool_Pa=(2e5,)))


def test_intercooling_temperature_and_losses() -> None:
    name = NAMES[7]  # aeroderivada con interenfriamiento
    k = _key(name)
    at = _app(name)
    at.number_input(key=f"{k}_Tic@Técnico").set_value(40.0).run()
    at.number_input(key=f"{k}_dpic").set_value(5.0).run()
    at.button(key="bt_btn").click().run()
    _no_problems(at)
    base = GAS_TURBINE_EXAMPLES[name]
    _check(at, replace(base, T_intercool_K=40 + C, dp_intercooler=0.05))


def test_size_by_net_power() -> None:
    name = NAMES[5]  # industrial a metano
    k = _key(name)
    at = _app(name)
    at.radio(key=f"{k}_size").set_value("Potencia neta").run()
    at.number_input(key=f"{k}_W@Técnico").set_value(50_000.0).run()  # kW
    at.button(key="bt_btn").click().run()
    _no_problems(at)
    assert _metric(at, "Ẇ neto [kW]").replace(" ", "") == "50000"


def test_changed_data_ask_to_recalculate() -> None:
    name = NAMES[0]
    at = _app(name)
    at.number_input(key=f"{_key(name)}_rp").set_value(10.0).run()
    assert any("tocá «Calcular turbina de gas»" in c.value for c in at.caption)
    _check(at, GAS_TURBINE_EXAMPLES[name])  # sigue el resultado anterior


@pytest.mark.parametrize(
    ("name", "widget", "value", "match"),
    [
        (NAMES[0], "rp", 30.0, None),  # sin regenerador, r_p alta es válida
        (NAMES[2], "rp", 30.0, "El regenerador no sirve"),
        (NAMES[3], "Tic@Técnico", 200.0, "no enfría"),
        (NAMES[3], "Trh@Técnico", 500.0, "no calienta"),
    ],
)
def test_invalid_data_are_explained(
    name: str, widget: str, value: float, match: str | None
) -> None:
    at = _app(name)
    at.number_input(key=f"{_key(name)}_{widget}").set_value(value).run()
    at.button(key="bt_btn").click().run()
    assert not at.exception
    if match is None:
        assert not at.error
    else:
        assert at.error and match in at.error[0].value


def test_procedure_theory_tespy_and_export() -> None:
    name = NAMES[4]  # 9-8 b): con regenerador ideal
    at = _app(name)
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    text = " ".join(md.value for md in theory.markdown)
    assert "Ericsson" in text and "§6.3" in text and "Regenerador" in text
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    titles = " ".join(md.value for md in procedure.markdown)
    for title in (
        "Presiones de cada etapa",
        "Compresor de baja (1 → 2)",
        "Interenfriador (2 → 3)",
        "Turbina de alta (6 → 7)",
        "Regenerador (4 → 5 y 9 → 10)",
        "Recalentador (7 → 8), aire estándar",
        "Balance de exergía",
    ):
        assert title in titles, title
    tespy = next(e for e in at.expander if "Control con TESPy" in e.label)
    # Con un regenerador ideal TESPy no resuelve: lo explica.
    assert any("regenerador ideal" in w.value for w in tespy.warning)
    assert len(at.get("download_button")) == 2


def test_tespy_table_matches() -> None:
    name = NAMES[6]  # microturbina con regenerador
    at = _app(name)
    tespy = next(e for e in at.expander if "Control con TESPy" in e.label)
    table = tespy.dataframe[0].value
    result = solve_gas_turbine(GAS_TURBINE_EXAMPLES[name])
    control = gas_turbine_tespy(result)
    eta_row = table[table["Magnitud"] == "η [%]"].iloc[0]
    assert float(eta_row["Gases ideales"]) == pytest.approx(result.eta_th * 100, abs=1e-3)
    assert float(eta_row["TESPy"]) == pytest.approx(control.eta_th * 100, abs=1e-3)
    assert "T aire del regenerador [°C]" in list(table["Magnitud"])


@pytest.mark.parametrize(
    ("system", "labels"),
    [
        ("SI", ("Ẇ neto [W]", "T escape [K]", "w neto [J/kg]")),
        ("Inglés", ("Ẇ neto [Btu/s]", "T escape [°F]", "w neto [Btu/lb]")),
    ],
)
def test_unit_systems(system: str, labels: tuple[str, ...]) -> None:
    at = _app(NAMES[5])
    at.selectbox(key="units_system").set_value(system).run()
    _no_problems(at)
    metric_labels = [m.label for m in at.metric]
    assert all(label in metric_labels for label in labels), metric_labels


@pytest.mark.parametrize(
    ("name", "variable"),
    [
        (NAMES[0], "Relación de presiones"),
        (NAMES[2], "Relación de presiones"),
        (NAMES[5], "Temperatura de entrada a la turbina"),
        (NAMES[5], "Temperatura ambiente"),
        (NAMES[4], "Cantidad de etapas (compresión y expansión)"),
        (NAMES[6], "Efectividad del regenerador"),
        (NAMES[3], "Presión del interenfriamiento"),
        (NAMES[8], "Presión del recalentamiento"),
    ],
)
def test_sweeps(name: str, variable: str) -> None:
    at = _app(name)
    at.selectbox(key="bt_sweep_param").set_value(variable).run()
    at.button(key="bt_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) == 5  # los 3 del resultado + η y w_neto
