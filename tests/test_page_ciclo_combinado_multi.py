"""Tests de /Ciclo_Combinado con dos y tres presiones y recalentamiento (AppTest) — Fase 3.6."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.cycles.combined_multi import (
    COMBINED_MULTI_EXAMPLES,
    MultiCombinedInputs,
    solve_combined_multi,
)

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "12_Ciclo_Combinado.py")
NAMES = list(COMBINED_MULTI_EXAMPLES)
ETA = "η ciclo combinado [%]"
CONFIGS = {(len(i.levels), i.reheat is not None) for i in COMBINED_MULTI_EXAMPLES.values()}
LEVELS = {1: "1 presión", 2: "2 presiones", 3: "3 presiones"}


def _app(levels: int, reheat: bool) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    at.radio(key="cc_levels").set_value(LEVELS[levels])
    at.checkbox(key="cc_reheat").set_value(reheat)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _prefix(name: str) -> str:
    inputs = COMBINED_MULTI_EXAMPLES[name]
    n, rh = len(inputs.levels), inputs.reheat is not None
    return f"cm{n}{'r' if rh else ''}_{NAMES.index(name)}"


def _examples(levels: int, reheat: bool) -> list[str]:
    return [
        n
        for n, i in COMBINED_MULTI_EXAMPLES.items()
        if len(i.levels) == levels and (i.reheat is not None) == reheat
    ]


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert not any("No se pudo dibujar" in w.value for w in at.warning)


def _check(at: AppTest, inputs: MultiCombinedInputs) -> None:
    result = solve_combined_multi(inputs)
    assert float(_metric(at, ETA)) == pytest.approx(result.eta_th * 100, abs=0.006)
    assert float(_metric(at, "x salida turbina")) == pytest.approx(result.steam.x_exhaust, abs=6e-4)


@pytest.mark.parametrize(("levels", "reheat"), sorted(CONFIGS))
def test_each_configuration(levels: int, reheat: bool) -> None:
    at = _app(levels, reheat)
    _no_problems(at)
    name = _examples(levels, reheat)[0]
    inputs = COMBINED_MULTI_EXAMPLES[name]
    _check(at, inputs)
    result = solve_combined_multi(inputs)
    # Sankey, T–s de la TG, T–Q, ciclo de vapor, configuraciones y exergía
    assert len(at.get("plotly_chart")) == 6
    states = next(
        df.value
        for df in at.dataframe
        if "Estado" in df.value.columns
        and "x [-]" in df.value.columns
        and len(df.value) == len(result.steam.states)
    )
    assert list(states["Estado"])[0].startswith("1 · salida del condensador")
    turbines = next(df.value for df in at.dataframe if "Turbina" in df.value.columns)
    assert len(turbines) == len(result.steam.turbines)
    configs = next(df.value for df in at.dataframe if "Configuración" in df.value.columns)
    assert list(configs["Configuración"])[-1].startswith(f"{levels} presi")
    assert any(c.value.startswith("📘 Sobre este ejemplo") for c in at.caption)


@pytest.mark.parametrize("name", NAMES)
def test_every_example(name: str) -> None:
    inputs = COMBINED_MULTI_EXAMPLES[name]
    levels, reheat = len(inputs.levels), inputs.reheat is not None
    at = _app(levels, reheat)
    at.selectbox(key=f"cm_example_{levels}{'r' if reheat else ''}").set_value(name).run()
    _no_problems(at)
    _check(at, inputs)


def test_level_and_reheat_inputs() -> None:
    name = NAMES[0]  # tres presiones con recalentamiento
    base = COMBINED_MULTI_EXAMPLES[name]
    at = _app(3, True)
    k = _prefix(name)
    at.number_input(key=f"{k}_B_p@Técnico").set_value(3.0).run()
    at.number_input(key=f"{k}_rh_T@Técnico").set_value(540.0).run()
    at.button(key="cc_btn").click().run()
    _no_problems(at)
    low = replace(base.levels[2], p_Pa=3e5)
    expected = replace(
        base,
        levels=(*base.levels[:2], low),
        reheat=replace(base.reheat, T_K=540.0 + 273.15),
    )
    _check(at, expected)


def test_reheat_pressure_and_baumann() -> None:
    name = _examples(2, True)[0]
    base = COMBINED_MULTI_EXAMPLES[name]
    at = _app(2, True)
    k = _prefix(name)
    at.number_input(key=f"{k}_rh_p@Técnico").set_value(30.0).run()
    at.checkbox(key=f"{k}_baumann").check().run()
    at.button(key="cc_btn").click().run()
    _no_problems(at)
    expected = replace(base, reheat=replace(base.reheat, p_Pa=30e5), baumann_alpha=1.0)
    _check(at, expected)
    assert any("Regla de Baumann" in i.value for i in at.info)


def test_changed_data_ask_to_recalculate() -> None:
    name = NAMES[0]
    at = _app(3, True)
    at.number_input(key=f"{_prefix(name)}_B_p@Técnico").set_value(3.0).run()
    assert any("tocá «Calcular ciclo combinado»" in c.value for c in at.caption)
    _check(at, COMBINED_MULTI_EXAMPLES[name])  # sigue el resultado anterior


@pytest.mark.parametrize(
    ("levels", "reheat", "key", "value", "match"),
    [
        (2, True, "rh_p@Técnico", 150.0, "Recalentador:"),
        (3, True, "A_Ts@Técnico", 700.0, "Caldera de recuperación:"),
        (3, False, "B_pinch@Técnico", 0.0, "Nivel de baja: el pinch"),
    ],
)
def test_invalid_data_are_explained(
    levels: int, reheat: bool, key: str, value: float, match: str
) -> None:
    at = _app(levels, reheat)
    name = _examples(levels, reheat)[0]
    at.number_input(key=f"{_prefix(name)}_{key}").set_value(value).run()
    at.button(key="cc_btn").click().run()
    assert not at.exception
    assert at.error and match in at.error[0].value


def test_procedure_theory_tespy_and_export() -> None:
    at = _app(3, True)
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    text = " ".join(md.value for md in theory.markdown)
    assert "Dos y tres presiones, recalentamiento" in text and "Baumann" in text
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    titles = " ".join(md.value for md in procedure.markdown)
    for title in (
        "Niveles de alta y de media: caudales",
        "Turbina de alta (3 → 4)",
        "Mezcla con el vapor de media",
        "Recalentador (6 → 7)",
        "Mezcla con el vapor de baja",
        "Exergía del ciclo de fondo",
    ):
        assert title in titles, title
    tespy = next(e for e in at.expander if "Control con TESPy" in e.label)
    assert len(tespy.dataframe) == 2
    steam = tespy.dataframe[1].value
    direct, control = (float(v.replace(" ", "")) for v in steam.iloc[0, 1:3])
    assert control == pytest.approx(direct, rel=3e-3)  # caudal de alta, al 0,3 %
    assert len(at.get("download_button")) == 2


@pytest.mark.parametrize(
    ("system", "labels"),
    [
        ("SI", ("Ẇ neto [W]", "T chimenea [K]", "ṁ vapor de alta [kg/s]")),
        ("Inglés", ("Ẇ neto [Btu/s]", "T chimenea [°F]", "ṁ vapor de media [lb/s]")),
    ],
)
def test_unit_systems(system: str, labels: tuple[str, ...]) -> None:
    at = _app(3, True)
    at.selectbox(key="units_system").set_value(system).run()
    _no_problems(at)
    metric_labels = [m.label for m in at.metric]
    assert all(label in metric_labels for label in labels), metric_labels


@pytest.mark.parametrize(
    ("levels", "reheat", "variable"),
    [
        (3, True, "Presión de alta"),
        (3, True, "Temperatura de recalentamiento"),
        (2, True, "Presión de recalentamiento"),
        (2, False, "Presión de baja"),
    ],
)
def test_sweeps(levels: int, reheat: bool, variable: str) -> None:
    at = _app(levels, reheat)
    at.selectbox(key="cm_sweep_param").set_value(variable).run()
    at.button(key="cm_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) == 8  # los 6 del resultado + η y título


def test_one_pressure_without_reheat_is_the_fase_3_4_page() -> None:
    at = AppTest.from_file(PAGE, default_timeout=300).run()
    _no_problems(at)
    labels = [m.label for m in at.metric]
    assert "x salida turbina" not in labels
    assert "ṁ vapor [kg/s]" in labels
    assert at.radio(key="cc_levels").value == "1 presión"
    assert at.checkbox(key="cc_reheat").value is False


def test_ambient_below_zero() -> None:
    """Hasta la 0.18.0, con el ambiente bajo cero la exergía del ciclo de fondo fallaba."""
    name = NAMES[0]
    at = _app(3, True)
    at.number_input(key=f"{_prefix(name)}_T1@Técnico").set_value(-10.0).run()
    at.button(key="cc_btn").click().run()
    _no_problems(at)
    assert any(md.value.startswith("#### Exergía del ciclo de fondo") for md in at.markdown)
