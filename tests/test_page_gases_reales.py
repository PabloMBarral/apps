"""Tests de /Gases_Reales (AppTest).

Fase 9.2. Los tres modos (el factor de compresibilidad, Van der Waals y Peng–Robinson,
las funciones características con Maxwell, Clapeyron y Joule–Thomson), todos los
ejemplos, los cambios de par y de fluido, los mensajes al alumno, los sistemas de
unidades, la teoría, el procedimiento y la exportación.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.gases import cubic as cb
from core.gases import real as rg
from core.gases import relations as rl

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "18_Gases_Reales.py")

M_Z = "El factor de compresibilidad"
M_CUBIC = "Van der Waals y Peng–Robinson"
M_REL = "Funciones características y Maxwell"
REL_MAXWELL = "Relaciones de Maxwell en un estado"
REL_CLAP = "Clapeyron (el cambio de fase)"
REL_JT = "Joule–Thomson"

Z_NAMES = list(rg.REAL_EXAMPLES)
CUBIC_NAMES = list(cb.CUBIC_EXAMPLES)
REL_NAMES = list(rl.RELATIONS_EXAMPLES)
CLAP_NAMES = list(rl.CLAPEYRON_EXAMPLES)
JT_NAMES = list(rl.JT_EXAMPLES)
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


def _mode(at: AppTest, mode: str, sub: str | None = None) -> AppTest:
    at.radio(key="rg_mode").set_value(mode).run()
    _no_problems(at)
    if sub is not None:
        at.radio(key="rm_kind").set_value(sub).run()
        _no_problems(at)
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


# ---------------------------------------------------------------------
# El factor de compresibilidad
# ---------------------------------------------------------------------


def test_default_is_cengel_refrigerant_134a() -> None:
    at = _app()
    assert at.radio(key="rg_mode").value == M_Z
    r = rg.compressibility(rg.REAL_EXAMPLES[Z_NAMES[0]].inputs)
    assert _metric(at, "Z real") == pytest.approx(r.Z, rel=5e-4)
    assert _metric(at, "v real") == pytest.approx(r.v_m3_per_kg, rel=1e-4)
    assert _metric(at, "Error del gas ideal") == pytest.approx(20.8, abs=0.05)
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert len(at.get("download_button")) == 2
    assert _charts(at) == 2
    assert any(c.value.startswith("📘") for c in at.caption)
    assert len(at.dataframe[0].value) == 6  # los seis modelos


@pytest.mark.parametrize("name", Z_NAMES)
def test_every_compressibility_example(name: str) -> None:
    at = _app()
    at.selectbox(key="rz_example").set_value(name).run()
    _no_problems(at)
    r = rg.compressibility(rg.REAL_EXAMPLES[name].inputs)
    assert _metric(at, "Z real") == pytest.approx(r.Z, rel=5e-4)
    assert [i.value for i in at.info] == list(r.notes)
    missing = [m for m in r.models if m.Z is None]
    assert len(at.warning) == len(missing)
    assert _charts(at) == 2


def test_changing_the_pair_keeps_the_state() -> None:
    """Con (T, v) o (p, v) los valores por defecto salen del mismo estado real."""
    at = _app()
    z = _metric(at, "Z real")
    for pair in ("Tv", "pv"):
        at.radio(key="rz_0_pair").set_value(pair).run()
        _no_problems(at)
        assert _metric(at, "Z real") == pytest.approx(z, abs=2e-4)


def test_changing_the_fluid() -> None:
    at = _app()
    at.selectbox(key="rz_0_fluid").set_value("Nitrogen").run()
    _no_problems(at)
    assert _metric(at, "Z real") == pytest.approx(1.0, abs=0.01)  # N₂ a 10 bar y 50 °C


def test_state_inside_the_dome_is_explained() -> None:
    at = _app()
    idx = Z_NAMES.index("Vapor de agua a 600 °F y 0,51431 ft³/lb (Çengel §3-7)")
    at.selectbox(key="rz_example").set_value(Z_NAMES[idx]).run()
    at.number_input(key=_wkey(at, f"rz_{idx}_Tv_v")).set_value(0.01).run()
    assert at.error and "campana" in at.error[0].value


# ---------------------------------------------------------------------
# Van der Waals y Peng–Robinson
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", CUBIC_NAMES)
def test_every_cubic_example(name: str) -> None:
    at = _mode(_app(), M_CUBIC)
    at.selectbox(key="rw_example").set_value(name).run()
    _no_problems(at)
    r = cb.solve_cubic(cb.CUBIC_EXAMPLES[name].inputs)
    roots = at.dataframe[0].value
    assert len(roots) == len(r.vdw.roots) + len(r.pr.roots)
    assert list(roots["Estable"]).count("✓") == 2  # una estable por cúbica
    assert [i.value for i in at.info] == list(r.notes)
    assert _charts(at) == 3
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]


def test_liquid_that_van_der_waals_calls_vapor_in_the_page() -> None:
    at = _mode(_app(), M_CUBIC)
    at.selectbox(key="rw_example").set_value("R-134a líquido a 10 bar y 20 °C").run()
    _no_problems(at)
    assert any("Para Van der Waals este estado es vapor" in i.value for i in at.info)
    sat = at.dataframe[1].value
    assert list(sat.iloc[:, 0]) == ["Real (CoolProp)", "Van der Waals", "Peng–Robinson"]


def test_cubic_pressure_at_saturation_is_explained() -> None:
    at = _mode(_app(), M_CUBIC)
    p_sat = rl.clapeyron(rl.ClapeyronInputs("R134a", 50.0 + C, 1.0)).p_sat
    at.number_input(key=_wkey(at, "rw_0_p")).set_value(p_sat / 1e5).run()
    assert at.error and "saturación" in at.error[0].value


# ---------------------------------------------------------------------
# Funciones características
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", REL_NAMES)
def test_every_maxwell_example(name: str) -> None:
    at = _mode(_app(), M_REL)
    at.selectbox(key="rm_example").set_value(name).run()
    _no_problems(at)
    r = rl.relations(rl.RELATIONS_EXAMPLES[name].inputs)
    assert _metric(at, "c_p − c_v") == pytest.approx((r.cp - r.cv) / 1e3, rel=1e-3)
    maxwell = at.dataframe[1].value
    assert len(maxwell) == 4
    assert len(at.dataframe[2].value) == 8  # las derivadas de §15.2
    assert _charts(at) == (0 if r.degenerate else 1)


def test_steam_relation_table_like_cengel() -> None:
    at = _mode(_app(), M_REL)
    maxwell = at.dataframe[1].value
    g = maxwell.iloc[3]
    assert "(∂s/∂p)_T" in g["Relación"]
    assert float(g["Izquierda (dif.)"]) == pytest.approx(-0.00165, abs=5e-6)
    assert float(g["Derecha (dif.)"]) == pytest.approx(-0.00159, abs=5e-6)


def test_liquid_with_a_big_step_is_explained() -> None:
    at = _mode(_app(), M_REL)
    name = "Agua líquida a 100 bar y 20 °C"
    idx = REL_NAMES.index(name)
    at.selectbox(key="rm_example").set_value(name).run()
    at.number_input(key=_wkey(at, f"rm_{idx}_dp")).set_value(0.5).run()
    at.number_input(key=_wkey(at, f"rm_{idx}_p")).set_value(1.0).run()
    at.number_input(key=_wkey(at, f"rm_{idx}_dT")).set_value(5.0).run()
    assert at.error and "bar por kelvin" in at.error[0].value


@pytest.mark.parametrize("name", CLAP_NAMES)
def test_every_clapeyron_example(name: str) -> None:
    at = _mode(_app(), M_REL, REL_CLAP)
    at.selectbox(key="rc_example").set_value(name).run()
    _no_problems(at)
    r = rl.clapeyron(rl.CLAPEYRON_EXAMPLES[name].inputs)
    assert _metric(at, "h_fg con Clapeyron") == pytest.approx(r.h_fg_clapeyron / 1e3, rel=1e-4)
    assert [i.value for i in at.info] == list(r.notes)
    assert _charts(at) == 1


@pytest.mark.parametrize("name", JT_NAMES)
def test_every_joule_thomson_example(name: str) -> None:
    at = _mode(_app(), M_REL, REL_JT)
    at.selectbox(key="rj_example").set_value(name).run()
    _no_problems(at)
    r = rl.joule_thomson(rl.JT_EXAMPLES[name].inputs)
    assert _metric(at, "μ_JT [") == pytest.approx(r.mu * 1e5, rel=1e-3)
    assert [i.value for i in at.info] == list(r.notes)
    assert _charts(at) == 1


def test_inversion_curve_cut_by_the_equation_of_state() -> None:
    """El metano: la rama alta de la curva pasa la T máxima de su ecuación de estado."""
    at = _mode(_app(), M_REL, REL_JT)
    at.selectbox(key="rj_0_fluid").set_value("Methane").run()
    _no_problems(at)
    assert any("La línea punteada" in c.value for c in at.caption)
    assert any("temperatura máxima de la ecuación de estado" in i.value for i in at.info)
    assert _charts(at) == 1


# ---------------------------------------------------------------------
# Unidades y teoría
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_every_mode_in_other_systems(system: str) -> None:
    at = _system(_app(), system)
    _no_problems(at)
    unit = "m³/kg" if system == "SI" else "ft³/lb"
    assert any(m.label.endswith(f"[{unit}]") for m in at.metric)
    _mode(at, M_CUBIC)
    for sub in (REL_MAXWELL, REL_CLAP, REL_JT):
        _mode(at, M_REL, sub)


def test_values_survive_a_change_of_system() -> None:
    at = _app()
    z = _metric(at, "Z real")
    _system(at, "Inglés")
    _no_problems(at)
    assert _metric(at, "Z real") == pytest.approx(z, rel=5e-4)


def test_theory_cites_the_vademecum() -> None:
    at = _app()
    theory = at.expander[0]
    text = " ".join(m.value for m in theory.markdown)
    for section in ("§7", "§8", "§15"):
        assert section in text
    assert len(theory.latex) >= 20
