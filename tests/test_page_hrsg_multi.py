"""Tests de la página de la HRSG con dos y tres presiones (AppTest) — Fase 3.5."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.cycles.hrsg_multi import MULTI_HRSG_EXAMPLES, hrsg_exergy, solve_multi_hrsg

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "10_HRSG.py")
TWO_NAMES = [n for n, i in MULTI_HRSG_EXAMPLES.items() if len(i.levels) == 2]
THREE_NAMES = [n for n, i in MULTI_HRSG_EXAMPLES.items() if len(i.levels) == 3]
TWO = MULTI_HRSG_EXAMPLES[TWO_NAMES[0]]
K2 = "hm2_0"  # prefijo de las keys del primer ejemplo de dos presiones
TOTAL = "ṁ vapor total [kg/s]"
STACK = "T chimenea [°C]"
C = 273.15


def _app(levels: str) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    at.radio(key="hr_levels").set_value(levels).run()
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


def test_two_levels() -> None:
    at = _app("2 presiones")
    _no_problems(at)
    assert float(_metric(at, TOTAL)) == pytest.approx(17.594, abs=1e-3)
    assert float(_metric(at, "ṁ vapor de alta [kg/s]")) == pytest.approx(15.265, abs=1e-3)
    assert float(_metric(at, "ṁ vapor de baja [kg/s]")) == pytest.approx(2.3289, abs=1e-4)
    assert float(_metric(at, STACK)) == pytest.approx(103.2, abs=0.05)
    assert float(_metric(at, "η exergético [%]")) == pytest.approx(83.5, abs=0.05)
    assert len(at.get("plotly_chart")) == 3  # T–Q, exergía por caso y por sección
    assert len(_frame(at, "Sección")) == 6
    assert list(_frame(at, "Caldera")["Caldera"]) == [
        "1 presión (solo alta)",
        "2 presiones (alta y baja)",
    ]
    assert len(_frame(at, "Estado")) == 10
    assert list(_frame(at, "Punto")["Punto"]) == list("abcdefh")
    assert any(i.value.startswith("El nivel de baja produce") for i in at.info)


@pytest.mark.parametrize(
    ("levels", "name"),
    [("2 presiones", n) for n in TWO_NAMES] + [("3 presiones", n) for n in THREE_NAMES],
)
def test_every_example_computes(levels: str, name: str) -> None:
    at = _app(levels)
    at.selectbox(key=f"hm{levels[0]}_example").set_value(name).run()
    _no_problems(at)
    result = solve_multi_hrsg(MULTI_HRSG_EXAMPLES[name])
    assert float(_metric(at, TOTAL)) == pytest.approx(result.m_steam_kg_s, rel=1e-4)
    assert float(_metric(at, STACK)) == pytest.approx(result.T_stack_K - C, abs=0.05)
    assert float(_metric(at, "η exergético [%]")) == pytest.approx(
        hrsg_exergy(result).efficiency * 100, abs=0.05
    )
    assert len(_frame(at, "Caldera")) == len(result.levels)
    assert any(c.value.startswith("📘 Sobre este ejemplo") for c in at.caption)


def test_level_inputs_change_the_result() -> None:
    at = _app("2 presiones")
    at.number_input(key=f"{K2}_B_p@Técnico").set_value(3.0).run()
    at.radio(key=f"{K2}_B_steam").set_value("Saturado (sin sobrecalentador)").run()
    _no_problems(at)
    low = replace(TWO.levels[1], p_Pa=3e5, T_steam_K=None)
    result = solve_multi_hrsg(replace(TWO, levels=(TWO.levels[0], low)))
    assert float(_metric(at, STACK)) == pytest.approx(result.T_stack_K - C, abs=0.05)
    assert float(_metric(at, TOTAL)) == pytest.approx(result.m_steam_kg_s, rel=1e-4)


@pytest.mark.parametrize(
    ("levels", "key", "value", "match"),
    [
        ("2 presiones", f"{K2}_A_p@Técnico", 5.0, "tiene que ser menor"),
        ("2 presiones", f"{K2}_A_pinch@Técnico", 0.0, "Nivel de alta: el pinch"),
        ("3 presiones", "hm3_0_M_Ts@Técnico", 300.0, "Nivel de media: el vapor"),
    ],
)
def test_invalid_data_are_explained(levels: str, key: str, value: float, match: str) -> None:
    at = _app(levels)
    at.number_input(key=key).set_value(value).run()
    assert not at.exception
    assert at.error and match in at.error[0].value


def test_theory_procedure_and_export() -> None:
    at = _app("3 presiones")
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    text = " ".join(md.value for md in theory.markdown)
    assert "Dos y tres presiones" in text and "§11" in text and "cap. 5" in text
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    titles = " ".join(md.value for md in procedure.markdown)
    for title in (
        "Nivel de alta: caudal de vapor",
        "Nivel de media: estados del agua",
        "Nivel de baja: economizador",
        "Exergía: cuánto vale el calor recuperado",
    ):
        assert title in titles, title
    assert len(at.get("download_button")) == 2


@pytest.mark.parametrize(
    ("system", "labels"),
    [
        ("SI", ("ṁ vapor total [kg/s]", "T chimenea [K]", "Ẋ destruida [W]")),
        ("Inglés", ("ṁ vapor total [lb/s]", "T chimenea [°F]", "Q̇ [Btu/s]")),
    ],
)
def test_unit_systems(system: str, labels: tuple[str, ...]) -> None:
    at = _app("2 presiones")
    at.selectbox(key="units_system").set_value(system).run()
    _no_problems(at)
    metric_labels = [m.label for m in at.metric]
    assert all(label in metric_labels for label in labels)


@pytest.mark.parametrize(
    "variable",
    [
        "Presión de baja",
        "Presión de alta",
        "Pinch (todos los niveles)",
        "Temperatura del agua de alimentación",
    ],
)
def test_sweeps_draw_the_charts(variable: str) -> None:
    at = _app("2 presiones")
    at.selectbox(key="hm_sweep_param").set_value(variable).run()
    at.button(key="hm_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) == 6  # los 3 del resultado + chimenea, η_II y vapor


def test_single_pressure_shows_its_exergy() -> None:
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    _no_problems(at)
    assert float(_metric(at, "Ẋ destruida [kW]")) == pytest.approx(4311.4, abs=0.5)
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    titles = [md.value for md in procedure.markdown if md.value.startswith("**")]
    exergy = next(i for i, t in enumerate(titles) if "Exergía: cuánto vale" in t)
    dew = next(i for i, t in enumerate(titles) if "Punto de rocío" in t)
    assert exergy == dew - 1  # la exergía va antes del punto de rocío, al final
