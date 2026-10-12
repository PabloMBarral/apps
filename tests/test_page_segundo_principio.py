"""Tests de /Segundo_Principio (AppTest).

Fase 10.2. Los cuatro modos (máquinas, Clausius, ciclo de Carnot y la entropía del calor),
todos los ejemplos, los datos de cada máquina, el veredicto, los sistemas de unidades, la
teoría, el procedimiento y la exportación.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.balances import second_law as sl

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "20_Segundo_Principio.py")

M_MACHINE, M_CLAUSIUS, M_CARNOT, M_HEAT = (
    "Máquinas",
    "Clausius",
    "Ciclo de Carnot",
    "Calor y entropía",
)


def _app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    _no_problems(at)
    return at


def _no_problems(at: AppTest, impossible: bool = False) -> None:
    assert not at.exception, [e.value for e in at.exception]
    errors = [e.value for e in at.error]
    if impossible:
        assert errors and all("Imposible" in e for e in errors), errors
    else:
        assert not errors, errors


def _metric(at: AppTest, prefix: str) -> float:
    value = next(m.value for m in at.metric if m.label.startswith(prefix))
    value = value.split("(")[0]
    return float(value.replace(" ", "").replace(",", ".").replace("−", "-"))


def _mode(at: AppTest, mode: str) -> AppTest:
    at.radio(key="sp_mode").set_value(mode).run()
    _no_problems(at)
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


def _wkey(at: AppTest, key: str) -> str:
    system = at.session_state["units_system"] if "units_system" in at.session_state else "Técnico"
    return f"{key}@{system}"


# ---------------------------------------------------------------------
# Máquinas
# ---------------------------------------------------------------------


def test_default_is_the_power_plant() -> None:
    at = _app()
    assert [e.label for e in at.expander] == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert _metric(at, "Trabajo W") == pytest.approx(30000.0)  # kW
    assert len(at.get("plotly_chart")) == 2
    assert len(at.get("download_button")) == 2
    assert any("Irreversible" in i.value for i in at.info)


@pytest.mark.parametrize("name", list(sl.MACHINE_EXAMPLES))
def test_every_machine_example(name: str) -> None:
    at = _app()
    at.selectbox(key="sm_example").set_value(name).run()
    r = sl.solve_machine(sl.MACHINE_EXAMPLES[name].inputs)
    _no_problems(at, impossible=r.verdict == "imposible")
    unit = 1e3  # kW o kJ en el Técnico
    assert _metric(at, "Trabajo W") == pytest.approx(r.W / unit, rel=1e-3)
    assert _metric(at, sl.PERFORMANCE_NAMES[r.kind]) == pytest.approx(r.performance, rel=1e-3)
    if r.verdict == "reversible":
        assert any("Reversible" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(sl.MACHINE_EXAMPLES))
@pytest.mark.parametrize("kind", list(sl.MACHINES))
def test_another_kind_of_machine_starts_possible(kind: str, name: str) -> None:
    """Al cambiar de máquina, los datos por defecto dan una máquina posible."""
    at = _app()
    at.selectbox(key="sm_example").set_value(name).run()
    if sl.MACHINE_EXAMPLES[name].inputs.kind == kind:
        return
    i = list(sl.MACHINE_EXAMPLES).index(name)
    at.radio(key=f"sm_{i}_kind").set_value(kind).run()
    _no_problems(at)
    reversible = "reversible" in sl.MACHINE_EXAMPLES[name].inputs.data  # sigue siéndolo
    boxes = at.success if reversible else at.info
    assert any(("Reversible" if reversible else "Irreversible") in b.value for b in boxes)


@pytest.mark.parametrize("code", ["QH_QC", "QH_W", "QC_W", "QH_p", "QC_p", "W_p", "QH_rev"])
def test_every_pair_of_data(code: str) -> None:
    at = _app()
    at.selectbox(key="sm_0_engine_data").set_value(code).run()
    _no_problems(at)
    assert len(at.metric) == 6


def test_other_ambient_temperature() -> None:
    at = _app()
    at.checkbox(key="sm_0_hasT0").check().run()
    _no_problems(at)
    assert any("no es una de las fuentes" in c.value for c in at.caption)


def test_machine_error_is_explained() -> None:
    at = _app()
    at.number_input(key=_wkey(at, "sm_0_TH")).set_value(10.0).run()  # T_H < T_C
    errors = [e.value for e in at.error]
    assert errors and "más caliente" in errors[0]


# ---------------------------------------------------------------------
# Clausius
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(sl.CLAUSIUS_EXAMPLES))
def test_every_clausius_example(name: str) -> None:
    at = _mode(_app(), M_CLAUSIUS)
    at.selectbox(key="sc_example").set_value(name).run()
    r = sl.solve_clausius(sl.CLAUSIUS_EXAMPLES[name].inputs)
    _no_problems(at, impossible=r.verdict == "imposible")
    assert _metric(at, "ΣQ") == pytest.approx(r.clausius_sum / 1e3, rel=1e-3, abs=1e-9)


def test_more_reservoirs() -> None:
    at = _mode(_app(), M_CLAUSIUS)
    at.number_input(key="sc_0_n").set_value(4).run()
    _no_problems(at)


# ---------------------------------------------------------------------
# Ciclo de Carnot
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(sl.CARNOT_EXAMPLES))
def test_every_carnot_example(name: str) -> None:
    at = _mode(_app(), M_CARNOT)
    at.selectbox(key="sk_example").set_value(name).run()
    _no_problems(at)
    r = sl.solve_carnot(sl.CARNOT_EXAMPLES[name].inputs)
    assert _metric(at, "Rendimiento η") == pytest.approx(r.eta, rel=1e-3)
    assert len(at.get("plotly_chart")) == 3  # T–s, p–v y el trabajo de cada proceso


def test_carnot_with_another_substance() -> None:
    at = _mode(_app(), M_CARNOT)
    at.radio(key="sk_0_sub_kind").set_value("ideal_gas").run()
    _no_problems(at)
    at.radio(key="sk_0_sub_kind").set_value("fluid").run()
    at.selectbox(key="sk_0_sub_fluid").set_value("R134a").run()
    _no_problems(at)


def test_carnot_with_volumes() -> None:
    at = _mode(_app(), M_CARNOT)
    at.selectbox(key="sk_example").set_value("Aire entre 1000 K y 300 K (gas ideal)").run()
    tag = "sk_2_ideal_gas-air-variable"
    at.selectbox(key=f"{tag}_prop1").set_value("v").run()
    at.selectbox(key=f"{tag}_prop2").set_value("v").run()
    _no_problems(at)


# ---------------------------------------------------------------------
# Calor y entropía
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(sl.HEAT_PATH_EXAMPLES))
def test_every_heat_example(name: str) -> None:
    at = _mode(_app(), M_HEAT)
    at.selectbox(key="sq_example").set_value(name).run()
    _no_problems(at)
    r = sl.solve_heat_path(sl.HEAT_PATH_EXAMPLES[name].inputs)
    scale = 1.0 if r.inputs.rates else 1e3
    assert _metric(at, "S") == pytest.approx(r.S_gen / scale, rel=1e-3)
    assert len(at.get("plotly_chart")) == 2


def test_heat_with_more_temperatures() -> None:
    at = _mode(_app(), M_HEAT)
    at.number_input(key="sq_0_n").set_value(3).run()
    _no_problems(at)


def test_heat_from_cold_to_hot_is_impossible() -> None:
    at = _mode(_app(), M_HEAT)
    at.number_input(key=_wkey(at, "sq_0_T1")).set_value(700.0).run()  # sumidero a 700 °C
    _no_problems(at, impossible=True)


# ---------------------------------------------------------------------
# Unidades, teoría y procedimiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_every_mode_in_other_systems(system: str) -> None:
    at = _system(_app(), system)
    _no_problems(at)
    for mode in (M_MACHINE, M_CLAUSIUS, M_CARNOT, M_HEAT):
        _mode(at, mode)


def test_values_survive_a_change_of_system() -> None:
    at = _app()
    at.number_input(key=_wkey(at, "sm_0_TH")).set_value(700.0).run()
    _no_problems(at)
    _system(at, "SI")
    _no_problems(at)
    assert at.number_input(key=_wkey(at, "sm_0_TH")).value == pytest.approx(973.15)


def test_theory_cites_the_vademecum() -> None:
    at = _app()
    theory = at.expander[0]
    text = " ".join(m.value for m in theory.markdown)
    for section in ("§9", "§9.1", "§9.2", "§9.3", "§10.1", "§11.5", "§11.10"):
        assert section in text
    assert len(theory.latex) >= 14


def test_procedure_ends_with_the_verdict() -> None:
    at = _app()
    proc = at.expander[1]
    titles = [m.value for m in proc.markdown if re.match(r"\*\*\d+\. ", m.value)]
    assert any("Clausius" in t for t in titles)
    assert titles[-1].startswith("**7. Veredicto")
