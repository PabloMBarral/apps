"""Tests de /Transferencia_de_Calor (AppTest).

Fase 8.1. Los tres modos (conducción, aletas y convección externa, interna y
natural), todos los ejemplos, los cambios de geometría, de capas, de bordes y de
punta, los mensajes al alumno, los sistemas de unidades, la teoría, el
procedimiento y la exportación.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.heat_transfer import conduction as cd
from core.heat_transfer import convection as cv
from core.heat_transfer import fins as fn

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "14_Transferencia_de_Calor.py")

M_COND = "Conducción: paredes, caños y esferas"
M_FIN = "Aletas"
M_CONV = "Convección: el coeficiente h"
V_EXT = "Forzada externa"
V_INT = "Forzada interna (un tubo)"
V_NAT = "Natural"

COND_NAMES = list(cd.CONDUCTION_EXAMPLES)
FIN_NAMES = list(fn.FIN_EXAMPLES)
EXT_NAMES = list(cv.EXTERNAL_EXAMPLES)
INT_NAMES = list(cv.INTERNAL_EXAMPLES)
NAT_NAMES = list(cv.NATURAL_EXAMPLES)


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
    return float(value.replace(" ", "").replace("%", "").replace(",", "."))


def _metric_label(at: AppTest, prefix: str) -> str:
    return next(m.label for m in at.metric if m.label.startswith(prefix))


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _expanders(at: AppTest) -> list[str]:
    return [e.label for e in at.expander]


def _mode(at: AppTest, mode: str, what: str | None = None) -> AppTest:
    at.radio(key="ht_mode").set_value(mode).run()
    if what is not None:
        at.radio(key="qv_what").set_value(what).run()
    _no_problems(at)
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


def _wkey(at: AppTest, key: str) -> str:
    """La key real de un ``number_input_si`` (lleva el sistema de unidades)."""
    system = at.session_state["units_system"] if "units_system" in at.session_state else "Técnico"
    return f"{key}@{system}"


# ---------------------------------------------------------------------
# Conducción
# ---------------------------------------------------------------------


def test_default_is_the_brick_wall_of_cengel() -> None:
    at = _app()
    assert at.radio(key="ht_mode").value == M_COND
    assert _metric(at, "Q̇ [") == pytest.approx(630.0)
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert _charts(at) == 1  # el perfil (una pared: sin radio crítico)
    assert len(at.get("download_button")) == 2
    assert any(c.value.startswith("📘") for c in at.caption)


@pytest.mark.parametrize("name", COND_NAMES)
def test_every_conduction_example(name: str) -> None:
    at = _app()
    at.selectbox(key="qc_example").set_value(name).run()
    _no_problems(at)
    inputs = cd.CONDUCTION_EXAMPLES[name]
    r = cd.solve_conduction(inputs)
    assert _metric(at, "Q̇ [") == pytest.approx(r.Q_W, rel=1e-4)
    sweep = inputs.geometry != "plane" and inputs.outer.kind == "fluid"
    assert _charts(at) == 1 + int(sweep)
    assert len(at.dataframe) == 1 + int(any(res.parts for res in r.resistances))
    network = at.dataframe[0].value
    assert list(network["Elemento"]) == [res.label for res in r.resistances]
    if r.critical_radius_m is not None:
        assert _metric(at, "Radio crítico") == pytest.approx(1000.0 * r.critical_radius_m, rel=1e-4)


def test_conduction_layers_geometry_and_boundaries() -> None:
    at = _app()
    # una segunda capa con los valores por defecto
    at.number_input(key="qc_0_n").set_value(2).run()
    _no_problems(at)
    assert _metric(at, "Q̇ [") < 630.0
    at.number_input(key="qc_0_n").set_value(1).run()
    # la misma pared como un caño: aparece el radio crítico
    at.radio(key="qc_0_geo").set_value("cylinder").run()
    _no_problems(at)
    assert any(m.label.startswith("U₁") for m in at.metric)
    # calor dado del lado 1 y superficie del lado 2
    at.radio(key="qc_0_geo").set_value("plane").run()
    at.radio(key="qc_0_s1_kind").set_value("heat").run()
    _no_problems(at)
    assert _metric(at, "Q̇ [") == pytest.approx(100.0)
    assert _metric(at, "T del lado 1") == pytest.approx(2.0 + 100.0 * 0.3 / (0.9 * 15.0), rel=1e-4)
    # calor dado en los dos lados: el mensaje
    at.radio(key="qc_0_s2_kind").set_value("heat").run()
    assert at.error and "calor dado en los dos lados" in at.error[0].value


def test_contact_and_composite_layers() -> None:
    at = _app()
    at.selectbox(key="qc_example").set_value(COND_NAMES[4]).run()
    _no_problems(at)
    with_contact = _metric(at, "Q̇ [")
    at.checkbox(key="qc_4_plane_0_c").uncheck().run()
    _no_problems(at)
    assert _metric(at, "Q̇ [") > 2.0 * with_contact
    at.selectbox(key="qc_example").set_value(COND_NAMES[3]).run()
    _no_problems(at)
    frac = at.number_input(key="qc_3_plane_2_p1_f")
    frac.set_value(0.5).run()
    assert at.error and "tienen que sumar 1" in at.error[0].value


def test_wire_sweep_has_a_minimum_at_the_critical_radius() -> None:
    at = _app()
    at.selectbox(key="qc_example").set_value(COND_NAMES[6]).run()
    _no_problems(at)
    assert _metric(at, "T del lado 1") == pytest.approx(105.0, abs=0.1)
    assert any("radio crítico" in m.value.lower() for m in at.info)


# ---------------------------------------------------------------------
# Aletas
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", FIN_NAMES)
def test_every_fin_example(name: str) -> None:
    at = _mode(_app(), M_FIN)
    at.selectbox(key="qa_example").set_value(name).run()
    _no_problems(at)
    ex = fn.FIN_EXAMPLES[name]
    r = fn.solve_fin(ex.fin)
    assert _metric(at, "Q̇ de la aleta") == pytest.approx(r.Q_W, rel=1e-4)
    assert _charts(at) == 2
    assert ("Eficiencia global η_o" in [m.label for m in at.metric]) == (ex.n_fins is not None)
    assert len(at.get("download_button")) == 2
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]


def test_fin_tip_and_shape_changes() -> None:
    at = _mode(_app(), M_FIN)
    at.selectbox(key="qa_example").set_value(FIN_NAMES[3]).run()  # aguja convectiva
    convective = _metric(at, "Q̇ de la aleta")
    at.selectbox(key="qa_3_tip").set_value("adiabatic").run()
    _no_problems(at)
    assert _metric(at, "Q̇ de la aleta") < convective
    at.selectbox(key="qa_3_tip").set_value("temperature").run()
    _no_problems(at)
    assert _metric(at, "T de la punta") == pytest.approx(62.5, abs=1e-6)  # (100 + 25)/2
    at.radio(key="qa_3_shape").set_value("straight").run()
    _no_problems(at)
    at.radio(key="qa_3_shape").set_value("annular").run()
    _no_problems(at)
    at.number_input(key=_wkey(at, "qa_3_k")).set_value(0.0).run()
    assert at.error and "conductividad" in at.error[0].value


# ---------------------------------------------------------------------
# Convección
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", EXT_NAMES)
def test_every_external_example(name: str) -> None:
    at = _mode(_app(), M_CONV, V_EXT)
    at.selectbox(key="qv_ext_example").set_value(name).run()
    _no_problems(at)
    r = cv.solve_external(cv.EXTERNAL_EXAMPLES[name].inputs)
    assert _metric(at, "Q̇ [") == pytest.approx(r.Q_W, rel=1e-4)
    assert _metric(at, "Reynolds") == pytest.approx(r.Re, rel=1e-3)
    assert _charts(at) == 1 + int(len(r.alternatives) > 1)
    assert len(at.get("download_button")) == 2


@pytest.mark.parametrize("name", NAT_NAMES)
def test_every_natural_example(name: str) -> None:
    at = _mode(_app(), M_CONV, V_NAT)
    at.selectbox(key="qv_nat_example").set_value(name).run()
    _no_problems(at)
    r = cv.solve_natural(cv.NATURAL_EXAMPLES[name].inputs)
    assert _metric(at, "Q̇ [") == pytest.approx(r.Q_W, rel=1e-4)
    assert _metric(at, "Nusselt") == pytest.approx(r.Nu, rel=1e-3)


@pytest.mark.parametrize("name", INT_NAMES)
def test_every_internal_example(name: str) -> None:
    at = _mode(_app(), M_CONV, V_INT)
    at.selectbox(key="qv_int_example").set_value(name).run()
    _no_problems(at)
    r = cv.solve_internal(cv.INTERNAL_EXAMPLES[name].inputs)
    assert _metric(at, "T de salida") == pytest.approx(r.T_out_K - 273.15, abs=1e-3)
    assert _metric(at, "Q̇ al fluido") == pytest.approx(r.Q_W, rel=1e-4)
    assert _charts(at) == 3  # el tubo, la comparación y la curva de Nu


def test_external_plate_trip_and_geometry() -> None:
    at = _mode(_app(), M_CONV, V_EXT)
    at.selectbox(key="qv_ext_example").set_value(EXT_NAMES[1]).run()  # placa al viento
    smooth = _metric(at, "h [")
    at.checkbox(key="qv_e1_trip").check().run()
    _no_problems(at)
    assert _metric(at, "h [") > 2.0 * smooth
    at.selectbox(key="qv_e1_geo").set_value("sphere").run()
    _no_problems(at)
    assert any("Whitaker" in c.value for c in at.caption)


def test_internal_condition_and_correlation() -> None:
    at = _mode(_app(), M_CONV, V_INT)
    at.radio(key="qv_i0_cond").set_value("constant_flux").run()
    _no_problems(at)
    assert any(m.label.startswith("T de la pared a la salida") for m in at.metric)
    at.selectbox(key="qv_i0_corr").set_value("dittus_boelter").run()
    _no_problems(at)
    assert any("Dittus y Boelter" in c.value for c in at.caption)


def test_convection_messages() -> None:
    at = _mode(_app(), M_CONV, V_INT)
    # agua a 1 atm con la pared a 150 °C en un tubo largo: hierve adentro
    at.selectbox(key="qv_int_example").set_value(INT_NAMES[0]).run()
    at.number_input(key=_wkey(at, "qv_i0_p")).set_value(1.01325).run()
    at.number_input(key=_wkey(at, "qv_i0_Ts")).set_value(150.0).run()
    assert at.error and "cambia de fase dentro del tubo" in at.error[0].value
    at = _mode(_app(), M_CONV, V_NAT)
    at.number_input(key=_wkey(at, "qv_n0_Ts")).set_value(20.0).run()
    assert at.error and "misma temperatura" in at.error[0].value


# ---------------------------------------------------------------------
# Unidades, teoría y procedimiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_other_unit_systems(system: str) -> None:
    unit = {"SI": "W", "Inglés": "Btu/h"}[system]
    for mode, what in ((M_COND, None), (M_FIN, None), (M_CONV, V_EXT), (M_CONV, V_INT)):
        at = _system(_mode(_app(), mode, what), system)
        _no_problems(at)
        assert any(m.label.endswith(f"[{unit}]") for m in at.metric), mode
        steps = at.expander[1]
        assert steps.label == "🔬 Procedimiento"
        assert len(steps.get("latex")) >= 4


def test_brick_wall_in_english_units() -> None:
    at = _system(_app(), "Inglés")
    assert _metric(at, "Q̇ [") == pytest.approx(630.0 * 3.412141633, rel=1e-4)


def test_theory_cites_the_sources() -> None:
    at = _app()
    theory = at.expander[0]
    assert theory.label == "📖 Fórmulas teóricas"
    text = " ".join(m.value for m in theory.markdown)
    assert "vademecum" in text and "Çengel y Ghajar" in text and "Incropera" in text
    for c in cv.CORRELATIONS.values():
        assert c.name in text
    assert len(theory.get("latex")) >= len(cv.CORRELATIONS) + 10
