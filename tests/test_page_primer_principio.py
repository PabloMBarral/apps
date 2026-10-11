"""Tests de /Primer_Principio (AppTest).

Fase 10.1. Los tres modos (sistema cerrado con el equilibrio térmico, flujo estacionario con
el dispositivo, la mezcla y el intercambiador, y el régimen transitorio con el llenado y el
vaciado), todos los ejemplos, los cambios de sustancia, de proceso y de dispositivo, el
veredicto del segundo principio, los sistemas de unidades, la teoría, el procedimiento y la
exportación.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.balances import closed as cl
from core.balances import steady_flow as sf
from core.balances import transient as tr

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "19_Primer_Principio.py")

M_CLOSED, M_FLOW, M_TRANS = "Sistema cerrado", "Flujo estacionario", "Régimen transitorio"
C_EQ = "Equilibrio térmico"
F_MIX, F_HX = "Cámara de mezcla", "Intercambiador"
T_DIS = "Vaciado de un tanque"
IMPOSSIBLE = "Compresor de aire que pierde calor (Çengel §5-4)"


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
    return float(value.replace(" ", "").replace("%", "").replace(",", ".").replace("−", "-"))


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _expanders(at: AppTest) -> list[str]:
    return [e.label for e in at.expander]


def _wkey(at: AppTest, key: str) -> str:
    """La key real de un ``number_input_si`` (lleva el sistema de unidades)."""
    system = at.session_state["units_system"] if "units_system" in at.session_state else "Técnico"
    return f"{key}@{system}"


def _mode(at: AppTest, mode: str, radio: str | None = None, sub: str | None = None) -> AppTest:
    at.radio(key="pp_mode").set_value(mode).run()
    _no_problems(at)
    if radio is not None and sub is not None:
        at.radio(key=radio).set_value(sub).run()
        _no_problems(at)
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


# ---------------------------------------------------------------------
# Sistema cerrado
# ---------------------------------------------------------------------


def test_default_is_the_steam_piston_of_cengel() -> None:
    at = _app()
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert _metric(at, "Trabajo de frontera W_b") == pytest.approx(101.73, abs=0.01)  # kJ
    assert _charts(at) == 4  # cascada de energía, de entropía, p–v y T–s
    assert len(at.get("download_button")) == 2
    assert any("Irreversible" in i.value for i in at.info)


def test_book_value_in_english_units() -> None:
    at = _system(_app(), "Inglés")
    _no_problems(at)
    assert _metric(at, "Trabajo de frontera W_b") == pytest.approx(96.4, abs=0.1)  # Btu


@pytest.mark.parametrize("name", list(cl.CLOSED_EXAMPLES))
def test_every_closed_example(name: str) -> None:
    at = _app()
    at.selectbox(key="pc_example").set_value(name).run()
    _no_problems(at)
    assert len(at.metric) == 6
    r = cl.solve_closed(cl.CLOSED_EXAMPLES[name].inputs)
    assert _metric(at, "Calor Q") == pytest.approx(r.Q / 1e3, rel=1e-3, abs=1e-3)


def test_switching_the_substance() -> None:
    at = _app()
    at.radio(key="pc_0_sub_kind").set_value("ideal_gas").run()
    _no_problems(at)
    at.radio(key="pc_0_sub_kind").set_value("incompressible").run()
    _no_problems(at)
    procs = at.selectbox(key="pc_0_incompressible-water-_proc").options
    assert procs == ["Volumen constante (tanque rígido)"]


@pytest.mark.parametrize("proc", list(cl.PROCESSES))
def test_every_process_with_steam(proc: str) -> None:
    at = _app()
    at.selectbox(key="pc_0_fluid-Water-_proc").set_value(proc).run()
    assert not at.exception, [e.value for e in at.exception]


def test_impossible_process_keeps_the_energy_balance() -> None:
    """Calor desde una fuente más fría que el sistema: el veredicto lo dice."""
    at = _app()
    at.checkbox(key="pc_0_has_Tb").uncheck().run()  # el calor viene del ambiente a 77 °F
    _no_problems(at, impossible=True)
    assert len(at.metric) == 6  # el balance de energía se muestra igual


@pytest.mark.parametrize("name", list(cl.EQUILIBRIUM_EXAMPLES))
def test_every_equilibrium_example(name: str) -> None:
    at = _mode(_app(), M_CLOSED, "pc_kind", C_EQ)
    at.selectbox(key="pe_example").set_value(name).run()
    _no_problems(at)
    r = cl.solve_equilibrium(cl.EQUILIBRIUM_EXAMPLES[name].inputs)
    assert _metric(at, "S_gen") == pytest.approx(r.S_gen / 1e3, rel=1e-3)


def test_equilibrium_with_a_reservoir() -> None:
    at = _mode(_app(), M_CLOSED, "pc_kind", C_EQ)
    at.radio(key="pe_0_other").set_value("reservoir").run()
    _no_problems(at)
    assert _metric(at, "Temperatura final") == pytest.approx(25.0, abs=1e-6)


# ---------------------------------------------------------------------
# Flujo estacionario
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(sf.DEVICE_EXAMPLES))
def test_every_device_example(name: str) -> None:
    at = _mode(_app(), M_FLOW)
    at.selectbox(key="pd_example").set_value(name).run()
    _no_problems(at, impossible=name == IMPOSSIBLE)
    r = sf.solve_device(sf.DEVICE_EXAMPLES[name].inputs)
    assert _metric(at, "Caudal ṁ") == pytest.approx(r.m_dot, rel=1e-3)
    assert _metric(at, "s_gen") == pytest.approx(r.s_gen / 1e3, rel=1e-3, abs=1e-6)
    assert _charts(at) >= 2


@pytest.mark.parametrize("device", list(sf.DEVICES))
def test_every_device_with_water(device: str) -> None:
    at = _mode(_app(), M_FLOW)
    at.selectbox(key="pd_example").set_value("Tobera de vapor que pierde calor (Çengel §5-4)").run()
    at.selectbox(key="pd_1_fluid-Water-_dev").set_value(device).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("name", list(sf.MIXING_EXAMPLES))
def test_every_mixing_example(name: str) -> None:
    at = _mode(_app(), M_FLOW, "pf_kind", F_MIX)
    at.selectbox(key="pm_example").set_value(name).run()
    _no_problems(at)
    r = sf.solve_mixing(sf.MIXING_EXAMPLES[name].inputs)
    assert _metric(at, "ṁ₂") == pytest.approx(r.m_dots[1], rel=1e-3)


def test_mixing_with_the_outlet_unknown() -> None:
    at = _mode(_app(), M_FLOW, "pf_kind", F_MIX)
    at.radio(key="pm_0_unknown").set_value("out").run()
    _no_problems(at)
    assert len(at.metric) == 6


@pytest.mark.parametrize("name", list(sf.EXCHANGER_EXAMPLES))
def test_every_exchanger_example(name: str) -> None:
    at = _mode(_app(), M_FLOW, "pf_kind", F_HX)
    at.selectbox(key="px_example").set_value(name).run()
    _no_problems(at)
    r = sf.solve_exchanger(sf.EXCHANGER_EXAMPLES[name].inputs)
    assert _metric(at, "Calor transferido") == pytest.approx(r.Q_transfer / 1e3, rel=1e-3)


@pytest.mark.parametrize("unknown", ["mA", "outA", "outB"])
def test_exchanger_unknowns(unknown: str) -> None:
    at = _mode(_app(), M_FLOW, "pf_kind", F_HX)
    at.radio(key="px_0_unknown").set_value(unknown).run()
    _no_problems(at)


# ---------------------------------------------------------------------
# Régimen transitorio
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(tr.CHARGING_EXAMPLES))
def test_every_charging_example(name: str) -> None:
    at = _mode(_app(), M_TRANS)
    at.selectbox(key="pl_example").set_value(name).run()
    _no_problems(at)
    r = tr.solve_charging(tr.CHARGING_EXAMPLES[name].inputs)
    assert _metric(at, "Masa que entra") == pytest.approx(r.m_in, rel=1e-3)
    assert _charts(at) == 2  # evolución y cascada


def test_charging_an_empty_tank_from_the_air_example() -> None:
    at = _mode(_app(), M_TRANS)
    at.selectbox(key="pl_example").set_value("Tanque con aire que se llena sin calor").run()
    at.radio(key="pl_1_empty").set_value("empty").run()
    _no_problems(at)


@pytest.mark.parametrize("name", list(tr.DISCHARGING_EXAMPLES))
def test_every_discharging_example(name: str) -> None:
    at = _mode(_app(), M_TRANS, "pt_kind", T_DIS)
    at.selectbox(key="pv_example").set_value(name).run()
    _no_problems(at)
    r = tr.solve_discharging(tr.DISCHARGING_EXAMPLES[name].inputs)
    assert _metric(at, "Masa que sale") == pytest.approx(r.m_out, rel=1e-3)


def test_isothermal_discharge_from_the_adiabatic_example() -> None:
    at = _mode(_app(), M_TRANS, "pt_kind", T_DIS)
    at.radio(key="pv_0_mode").set_value("isothermal").run()
    _no_problems(at)
    assert _metric(at, "Calor Q") > 0.0


# ---------------------------------------------------------------------
# Unidades, teoría y procedimiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_every_mode_in_other_systems(system: str) -> None:
    at = _system(_app(), system)
    _no_problems(at)
    for mode, radio, subs in (
        (M_CLOSED, "pc_kind", ["Un proceso", C_EQ]),
        (M_FLOW, "pf_kind", ["Un dispositivo", F_MIX, F_HX]),
        (M_TRANS, "pt_kind", ["Llenado de un tanque", T_DIS]),
    ):
        for sub in subs:
            _mode(at, mode, radio, sub)


def test_values_survive_a_change_of_system() -> None:
    at = _app()
    at.number_input(key=_wkey(at, "pc_0_m")).set_value(9.0).run()
    _no_problems(at)
    _system(at, "SI")
    _no_problems(at)
    assert at.number_input(key=_wkey(at, "pc_0_m")).value == pytest.approx(9.0)


def test_theory_cites_the_vademecum() -> None:
    at = _app()
    theory = at.expander[0]
    text = " ".join(m.value for m in theory.markdown)
    for section in ("§3", "§10", "§13", "§3.5", "§3.6", "§10.4"):
        assert section in text
    assert len(theory.latex) >= 12


def test_procedure_has_the_balances() -> None:
    at = _app()
    proc = at.expander[1]
    titles = [m.value for m in proc.markdown if m.value.startswith("**")]
    assert any("Primer principio" in t for t in titles)
    assert any("Segundo principio" in t for t in titles)
    assert len(proc.latex) >= 8
