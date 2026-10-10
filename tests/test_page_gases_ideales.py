"""Tests de /Gases_Ideales (AppTest).

Fase 9.1. Los tres modos (un gas entre dos estados, mezclas y transformaciones)
con sus submodos, todos los ejemplos, los cambios de par, de base, de proceso y
de dato final, los mensajes al alumno, los sistemas de unidades, la teoría, el
procedimiento y la exportación.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.gases import ideal as ig
from core.gases import mixture as mx
from core.gases import polytropic as pt

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "17_Gases_Ideales.py")

M_GAS = "Un gas entre dos estados"
M_MIX = "Mezclas de gases"
M_PROC = "Transformaciones"
MIX_COMP = "Composición y propiedades"
MIX_ADIA = "Mezcla adiabática (tanque o cámara)"
PROC_ONE = "Un proceso y los cinco caminos"
PROC_STAGED = "Compresión en etapas"
PROC_N = "El exponente n de dos estados"

STATE_NAMES = list(ig.STATE_EXAMPLES)
COMP_NAMES = list(mx.COMPOSITION_EXAMPLES)
MIX_NAMES = list(mx.MIXING_EXAMPLES)
PROC_NAMES = list(pt.PROCESS_EXAMPLES)
C = 273.15


def _app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    _no_problems(at)
    return at


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


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


def _mode(at: AppTest, mode: str, sub: tuple[str, str] | None = None) -> AppTest:
    at.radio(key="ig_mode").set_value(mode).run()
    _no_problems(at)
    if sub is not None:
        at.radio(key=sub[0]).set_value(sub[1]).run()
        _no_problems(at)
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


# ---------------------------------------------------------------------
# Un gas entre dos estados
# ---------------------------------------------------------------------


def test_default_is_air_heated_to_1500_K() -> None:
    at = _app()
    assert at.radio(key="ig_mode").value == M_GAS
    r = ig.solve_state_change(ig.STATE_EXAMPLES[STATE_NAMES[0]].inputs)
    assert _metric(at, "Δh, c_p variable") == pytest.approx(r.variable.dh / 1e3, rel=1e-4)
    assert _metric(at, "Error de Δh") == pytest.approx(100 * r.error("constant", "dh"), abs=0.01)
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert len(at.get("download_button")) == 2
    assert _charts(at) == 1
    assert any(c.value.startswith("📘") for c in at.caption)


@pytest.mark.parametrize("name", STATE_NAMES)
def test_every_state_example(name: str) -> None:
    at = _app()
    at.selectbox(key="gi_example").set_value(name).run()
    _no_problems(at)
    r = ig.solve_state_change(ig.STATE_EXAMPLES[name].inputs)
    assert _metric(at, "Δh, c_p variable") == pytest.approx(r.variable.dh / 1e3, rel=1e-4)
    assert _metric(at, "Δu, c_p variable") == pytest.approx(r.variable.du / 1e3, rel=1e-4)
    assert _metric(at, "Δs, c_p variable") == pytest.approx(r.variable.ds / 1e3, rel=1e-4, abs=1e-6)
    assert [i.value for i in at.info] == list(r.notes)
    states, models = at.dataframe[0].value, at.dataframe[1].value
    assert len(models) == 3
    assert "Z (CoolProp)" in list(states["Propiedad"])


def test_changing_the_pair_keeps_the_state() -> None:
    """Con (p, v) o (T, v) los valores por defecto salen del mismo estado."""
    at = _app()
    before = _metric(at, "Δs, c_p variable")
    at.radio(key="gi_0_1_pair").set_value("pv").run()
    _no_problems(at)
    assert _metric(at, "Δs, c_p variable") == pytest.approx(before, rel=1e-4)
    at.radio(key="gi_0_2_pair").set_value("Tv").run()
    _no_problems(at)
    assert _metric(at, "Δs, c_p variable") == pytest.approx(before, rel=1e-4)


def test_changing_the_gas() -> None:
    at = _app()
    at.selectbox(key="gi_0_gas").set_value("He").run()
    _no_problems(at)
    assert any("monoatómico" in i.value for i in at.info)
    assert _metric(at, "Error de Δh") == pytest.approx(0.0, abs=1e-6)


def test_temperature_out_of_range_is_explained() -> None:
    at = _app()
    at.number_input(key=_wkey(at, "gi_0_2_pT_T")).set_value(9000.0).run()
    assert at.error and "fuera del rango de los polinomios NASA" in at.error[0].value


# ---------------------------------------------------------------------
# Mezclas
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", COMP_NAMES)
def test_every_composition_example(name: str) -> None:
    at = _mode(_app(), M_MIX)
    assert at.radio(key="gm_kind").value == MIX_COMP
    at.selectbox(key="gm_c_example").set_value(name).run()
    _no_problems(at)
    r = mx.solve_mixture(mx.COMPOSITION_EXAMPLES[name].inputs)
    assert _metric(at, "M_M") == pytest.approx(r.M * 1e3, rel=1e-4)
    assert _metric(at, "c_p,M") == pytest.approx(r.cp / 1e3, rel=1e-4)
    assert len(at.dataframe[0].value) == len(r.components)
    assert _charts(at) == 1


def test_changing_the_basis_keeps_the_composition() -> None:
    """Las cantidades de la base nueva salen del último resultado."""
    at = _mode(_app(), M_MIX)
    M = _metric(at, "M_M")
    for basis in ("mole_fraction", "mass_fraction", "moles", "mass"):
        at.radio(key="gm_c0_basis").set_value(basis).run()
        _no_problems(at)
        assert _metric(at, "M_M") == pytest.approx(M, rel=1e-3), basis


def test_one_more_gas_and_the_messages() -> None:
    at = _mode(_app(), M_MIX)
    at.selectbox(key="gm_c_example").set_value("El aire seco (fracciones molares)").run()
    at.number_input(key="gm_c1_n").set_value(5).run()
    _no_problems(at)  # el gas nuevo entra con fracción 0
    assert len(at.dataframe[0].value) == 5
    at.number_input(key="gm_c1_mole_fraction_4").set_value(0.1).run()
    assert at.error and "tienen que sumar 1" in at.error[0].value
    at.number_input(key="gm_c1_mole_fraction_4").set_value(0.0).run()
    at.selectbox(key="gm_c1_1_gas").set_value("N2").run()
    assert at.error and "una sola vez" in at.error[0].value


@pytest.mark.parametrize("name", MIX_NAMES)
def test_every_mixing_example(name: str) -> None:
    at = _mode(_app(), M_MIX, ("gm_kind", MIX_ADIA))
    at.selectbox(key="gm_a_example").set_value(name).run()
    _no_problems(at)
    r = mx.solve_mixing(mx.MIXING_EXAMPLES[name].inputs)
    assert _metric(at, "T final") == pytest.approx(r.T_K - C, rel=1e-4)
    assert _metric(at, "p final") == pytest.approx(r.p_Pa / 1e5, rel=1e-4)
    tank = r.inputs.kind == "tank"
    assert _metric(at, "S_gen" if tank else "Ṡ_gen") == pytest.approx(r.S_gen / 1e3, rel=1e-4)
    assert len(at.dataframe[0].value) == len(r.streams)
    assert _charts(at) == 1
    assert [i.value for i in at.info] == list(r.notes)


def test_mixing_changes_and_messages() -> None:
    at = _mode(_app(), M_MIX, ("gm_kind", MIX_ADIA))
    T_tank = _metric(at, "T final")
    at.radio(key="gm_a0_model").set_value("variable").run()
    _no_problems(at)
    assert _metric(at, "T final") == pytest.approx(T_tank, abs=0.5)
    at.radio(key="gm_a0_kind").set_value("flow").run()  # la salida arranca en la p más baja
    _no_problems(at)
    assert _metric(at, "p final") == pytest.approx(1.0, rel=1e-6)
    at.number_input(key=_wkey(at, "gm_a0_pout")).set_value(1.2).run()
    assert at.error and "no comprime" in at.error[0].value
    at.number_input(key=_wkey(at, "gm_a0_pout")).set_value(1.0).run()
    at.number_input(key="gm_a0_n").set_value(3).run()
    _no_problems(at)
    assert len(at.dataframe[0].value) == 3


# ---------------------------------------------------------------------
# Transformaciones
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", PROC_NAMES)
def test_every_process_example(name: str) -> None:
    at = _mode(_app(), M_PROC)
    assert at.radio(key="gp_kind").value == PROC_ONE
    at.selectbox(key="gp_p_example").set_value(name).run()
    _no_problems(at)
    r = pt.solve_process(pt.PROCESS_EXAMPLES[name].inputs)
    assert _metric(at, "T₂") == pytest.approx(r.state2.T_K - C, rel=1e-4)
    assert _metric(at, "p₂") == pytest.approx(r.state2.p_Pa / 1e5, rel=1e-4)
    assert _metric(at, "q [") == pytest.approx(r.q / 1e3, rel=1e-4, abs=1e-9)
    work = "w [" if r.inputs.system == "closed" else "w_f ["
    assert _metric(at, work) == pytest.approx(
        (r.w if r.inputs.system == "closed" else r.w_f) / 1e3, rel=1e-4, abs=1e-9
    )
    paths = at.dataframe[2].value
    assert len(paths) == 5
    assert _charts(at) == 3
    assert [i.value for i in at.info] == list(r.notes)
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]


def test_cengel_compressor_paths() -> None:
    """Çengel §7-12: 263,2 (isoentrópica), 246,4 (n = 1,3) y 189,2 kJ/kg (isotérmica)."""
    at = _mode(_app(), M_PROC)
    paths = at.dataframe[2].value
    w_f = dict(zip(paths["Proceso"], paths["w_f [kJ/kg]"], strict=True))
    assert float(w_f["Adiabática"]) == pytest.approx(-263.2, abs=0.2)
    assert float(w_f["▸ Politrópica n = 1,3"]) == pytest.approx(-246.4, abs=0.1)
    assert float(w_f["Isoterma"]) == pytest.approx(-189.2, abs=0.1)


def test_process_and_datum_changes() -> None:
    at = _mode(_app(), M_PROC)
    at.radio(key="gp_p0_proc").set_value("isothermal").run()
    _no_problems(at)
    assert _metric(at, "T₂") == pytest.approx(300.0 - C, abs=1e-6)
    at.radio(key="gp_p0_final_isothermal").set_value("ratio").run()
    _no_problems(at)
    at.radio(key="gp_p0_proc").set_value("isochoric").run()  # su dato es p₂ o T₂
    _no_problems(at)
    at.radio(key="gp_p0_sys").set_value("closed").run()
    _no_problems(at)
    assert _metric(at, "w [") == 0.0  # sin trabajo de expansión
    at.radio(key="gp_p0_proc").set_value("polytropic").run()
    at.number_input(key="gp_p0_n").set_value(1.0).run()
    assert at.error and "isoterma" in at.error[0].value


def test_variable_cp_in_the_process() -> None:
    at = _mode(_app(), M_PROC)
    at.selectbox(key="gp_p_example").set_value(PROC_NAMES[1]).run()  # el motor de auto
    _no_problems(at)
    assert _metric(at, "T₂") == pytest.approx(662.81 - C, abs=0.05)
    at.radio(key="gp_p1_model").set_value("constant").run()
    _no_problems(at)
    assert _metric(at, "T₂") > 662.81 - C


def test_staged_compression_of_cengel() -> None:
    at = _mode(_app(), M_PROC, ("gp_kind", PROC_STAGED))
    assert _metric(at, "w_f en 2 etapas") == pytest.approx(-215.36, abs=0.01)
    assert _metric(at, "Ahorro") == pytest.approx(12.6, abs=0.05)
    assert any("interenfriadores" in c.value for c in at.caption)
    assert _charts(at) == 1
    at.number_input(key="gp_s_N").set_value(3).run()
    _no_problems(at)
    assert _metric(at, "w_f en 3 etapas") > -215.36
    assert len(at.dataframe[0].value) == 3
    at.number_input(key="gp_s_N").set_value(1).run()
    _no_problems(at)
    assert _metric(at, "Ahorro") == 0.0


def test_exponent_from_two_states() -> None:
    at = _mode(_app(), M_PROC, ("gp_kind", PROC_N))
    E = pt.EXPONENT_EXAMPLE
    e = pt.exponent_from_states("air", E["p1_Pa"], E["p2_Pa"], T1_K=E["T1_K"], T2_K=E["T2_K"])
    assert _metric(at, "Exponente politrópico n") == pytest.approx(e.n, abs=1e-3)
    assert at.info[0].value == e.text
    at.radio(key="gp_n_data").set_value("v").run()
    _no_problems(at)
    assert _metric(at, "Exponente politrópico n") == pytest.approx(e.n, abs=1e-3)
    at.number_input(key=_wkey(at, "gp_n_p2")).set_value(1.0).run()
    assert at.error and "isóbara" in at.error[0].value


# ---------------------------------------------------------------------
# Unidades
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_every_mode_in_other_systems(system: str) -> None:
    at = _system(_app(), system)
    _no_problems(at)
    assert any(m.label.endswith("[J/kg]") or m.label.endswith("[Btu/lb]") for m in at.metric)
    for mode, sub in (
        (M_MIX, None),
        (M_MIX, ("gm_kind", MIX_ADIA)),
        (M_PROC, None),
        (M_PROC, ("gp_kind", PROC_STAGED)),
        (M_PROC, ("gp_kind", PROC_N)),
    ):
        _mode(at, mode, sub)


def test_values_survive_a_change_of_system() -> None:
    at = _app()
    dh = _metric(at, "Δh, c_p variable")
    _system(at, "Inglés")
    _no_problems(at)
    assert _metric(at, "Δh, c_p variable") == pytest.approx(dh / 2.326, rel=1e-4)
