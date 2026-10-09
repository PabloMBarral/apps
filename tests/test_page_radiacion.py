"""Tests de /Radiacion (AppTest).

Fase 8.2. Los cinco modos (cuerpo negro, factores de forma, dos superficies,
recintos y radiación con convección o la termocupla), todos los ejemplos, los
cambios de geometría, de bandas, de pantallas y de condiciones, los mensajes al
alumno, los sistemas de unidades, la teoría, el procedimiento y la exportación.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.heat_transfer import radiation as rd

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "15_Radiacion.py")

M_BB = "Cuerpo negro y superficies reales"
M_VF = "Factores de forma"
M_TS = "Dos superficies y pantallas"
M_EN = "Recinto de varias superficies"
M_CR = "Radiación y convección juntas"
W_SURF = "Una superficie que pierde calor"
W_TC = "Una termocupla"

BB_NAMES = list(rd.BLACKBODY_EXAMPLES)
VF_NAMES = list(rd.VIEW_FACTOR_EXAMPLES)
TS_NAMES = list(rd.TWO_SURFACE_EXAMPLES)
EN_NAMES = list(rd.ENCLOSURE_EXAMPLES)
SB_NAMES = list(rd.SURFACE_EXAMPLES)
TC_NAMES = list(rd.THERMOCOUPLE_EXAMPLES)
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
    return float(value.replace(" ", "").replace("%", "").replace(",", "."))


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _expanders(at: AppTest) -> list[str]:
    return [e.label for e in at.expander]


def _mode(at: AppTest, mode: str, what: str | None = None) -> AppTest:
    at.radio(key="rd_mode").set_value(mode).run()
    if what is not None:
        at.radio(key="rc_what").set_value(what).run()
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
# Cuerpo negro
# ---------------------------------------------------------------------


def test_default_is_the_black_ball_of_cengel() -> None:
    at = _app()
    assert at.radio(key="rd_mode").value == M_BB
    assert _metric(at, "E_b = σT⁴") == pytest.approx(23226.0, rel=1e-4)
    assert _metric(at, "E_bλ en λ") == pytest.approx(3846.0, rel=2e-4)
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert _charts(at) == 1
    assert len(at.get("download_button")) == 2
    assert any(c.value.startswith("📘") for c in at.caption)


@pytest.mark.parametrize("name", BB_NAMES)
def test_every_blackbody_example(name: str) -> None:
    at = _app()
    at.selectbox(key="rb_example").set_value(name).run()
    _no_problems(at)
    r = rd.solve_blackbody(rd.BLACKBODY_EXAMPLES[name].inputs)
    assert _metric(at, "E_b = σT⁴") == pytest.approx(r.E_b_W_per_m2, rel=1e-4)
    assert _metric(at, "λ_máx") == pytest.approx(r.lambda_max_m * 1e6, rel=1e-3)
    if r.emissivity is not None:
        assert _metric(at, "Emisividad total") == pytest.approx(r.emissivity, abs=1e-4)
    if r.absorptivity is not None:
        assert _metric(at, "Absortividad") == pytest.approx(r.absorptivity, abs=1e-4)
    assert _charts(at) == 1 + int(r.inputs.T_source_K is not None)
    assert len(at.dataframe) == int(bool(r.rows)) + int(bool(r.source_rows))


def test_bands_change_and_message() -> None:
    at = _app()
    name = next(n for n in BB_NAMES if "12-4" in n)
    idx = BB_NAMES.index(name)
    at.selectbox(key="rb_example").set_value(name).run()
    eps = _metric(at, "Emisividad total")
    assert eps == pytest.approx(0.5206, abs=1e-4)
    # la banda del medio, más emisiva: sube ε
    at.number_input(key=f"rb_{idx}_b1_e").set_value(0.95).run()
    _no_problems(at)
    assert _metric(at, "Emisividad total") > eps
    # un corte que no crece: el mensaje
    at.number_input(key=_wkey(at, f"rb_{idx}_b1_u")).set_value(2.0).run()
    assert at.error and "crecer" in at.error[0].value


def test_band_fraction_and_target() -> None:
    at = _app()
    at.checkbox(key="rb_0_band").check().run()
    at.checkbox(key="rb_0_frac").check().run()
    _no_problems(at)
    assert any(m.label.startswith("Fracción en la banda") for m in at.metric)
    assert any(m.label.startswith("λ de esa fracción") for m in at.metric)


# ---------------------------------------------------------------------
# Factores de forma
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", VF_NAMES)
def test_every_view_factor_example(name: str) -> None:
    at = _mode(_app(), M_VF)
    at.selectbox(key="rv_example").set_value(name).run()
    _no_problems(at)
    r = rd.solve_view_factor(rd.VIEW_FACTOR_EXAMPLES[name].inputs)
    assert _metric(at, "F_ij") == pytest.approx(r.F_ij, abs=1e-4)
    assert _metric(at, "F_ji") == pytest.approx(r.F_ji, abs=1e-4)
    assert _charts(at) == 1


def test_view_factor_geometry_change() -> None:
    at = _mode(_app(), M_VF)
    for g in rd.VIEW_FACTORS:
        at.selectbox(key="rv_0_geo").set_value(g).run()
        _no_problems(at)
        assert 0.0 < _metric(at, "F_ij") <= 1.0


# ---------------------------------------------------------------------
# Dos superficies
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", TS_NAMES)
def test_every_two_surface_example(name: str) -> None:
    at = _mode(_app(), M_TS)
    at.selectbox(key="rt_example").set_value(name).run()
    _no_problems(at)
    r = rd.solve_two_surface(rd.TWO_SURFACE_EXAMPLES[name].inputs)
    assert _metric(at, "Q̇ de 1 a 2") == pytest.approx(r.Q_W, rel=1e-4)
    assert _charts(at) == 1
    network = at.dataframe[0].value
    assert list(network["Resistencia"]) == [x.label for x in r.resistances]


def test_shields_reduce_the_heat() -> None:
    at = _mode(_app(), M_TS)
    without = _metric(at, "Q̇ de 1 a 2")
    assert without == pytest.approx(3625.6, abs=0.1)
    at.number_input(key="rt_0_parallel_plates_ns").set_value(1).run()
    _no_problems(at)
    assert _metric(at, "Q̇ de 1 a 2") == pytest.approx(805.7, abs=0.1)
    assert any(m.label.startswith("T de la pantalla 1") for m in at.metric)
    # una cáscara: los radios tienen que crecer
    at.radio(key="rt_0_geo").set_value("concentric_spheres").run()
    _no_problems(at)
    at.number_input(key="rt_0_concentric_spheres_ns").set_value(1).run()
    at.number_input(key=_wkey(at, "rt_0_concentric_spheres_s0_r")).set_value(500.0).run()
    assert at.error and "crecer" in at.error[0].value


# ---------------------------------------------------------------------
# Recintos
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", EN_NAMES)
def test_every_enclosure_example(name: str) -> None:
    at = _mode(_app(), M_EN)
    at.selectbox(key="re_example").set_value(name).run()
    _no_problems(at)
    r = rd.solve_enclosure(rd.ENCLOSURE_EXAMPLES[name].inputs)
    for s, Q in zip(r.inputs.surfaces, r.Q_W, strict=True):
        assert _metric(at, f"Q̇ de «{s.name}»") == pytest.approx(Q, rel=1e-4, abs=1e-6)
    assert _charts(at) == 1
    assert len(at.dataframe) == 3  # las superficies, F y los pares


def test_enclosure_conditions_and_layouts() -> None:
    at = _mode(_app(), M_EN)
    # la pared aislada del horno de pintura a temperatura dada: deja de ser rerradiante
    at.radio(key="re_0_triangular_duct_s2_cond").set_value("T dada").run()
    _no_problems(at)
    assert _metric(at, "Q̇ de «Pared aislada»") != pytest.approx(0.0, abs=1e-3)
    # otra configuración, con sus valores por defecto
    for layout in rd.ENCLOSURE_LAYOUTS:
        at.selectbox(key="re_0_layout").set_value(layout).run()
        _no_problems(at)
    # un triángulo que no cierra
    at.selectbox(key="re_0_layout").set_value("triangular_duct").run()
    at.number_input(key=_wkey(at, "re_0_triangular_duct_d0")).set_value(5.0).run()
    assert at.error and "triángulo" in at.error[0].value


# ---------------------------------------------------------------------
# Radiación y convección
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", SB_NAMES)
def test_every_surface_example(name: str) -> None:
    at = _mode(_app(), M_CR, W_SURF)
    at.selectbox(key="rc_example").set_value(name).run()
    _no_problems(at)
    r = rd.solve_surface_balance(rd.SURFACE_EXAMPLES[name].inputs)
    assert _metric(at, "Q̇ por radiación") == pytest.approx(r.Q_rad_W, rel=1e-4, abs=1e-3)
    assert _metric(at, "Q̇ neto que pierde") == pytest.approx(r.Q_lost_W, rel=1e-4, abs=1e-3)
    assert _charts(at) == 1


def test_surface_equilibrium_and_sun() -> None:
    at = _mode(_app(), M_CR, W_SURF)
    at.radio(key="rc_0_given").set_value("El calor que recibe (sale la T de equilibrio)").run()
    _no_problems(at)
    assert any(m.label.startswith("T_s de equilibrio") for m in at.metric)
    at.checkbox(key="rc_0_sun").check().run()
    _no_problems(at)
    assert any(m.label.startswith("Q̇ que absorbe del sol") for m in at.metric)


@pytest.mark.parametrize("name", TC_NAMES)
def test_every_thermocouple_example(name: str) -> None:
    at = _mode(_app(), M_CR, W_TC)
    at.selectbox(key="rk_example").set_value(name).run()
    _no_problems(at)
    r = rd.solve_thermocouple(rd.THERMOCOUPLE_EXAMPLES[name].inputs)
    assert _metric(at, "T real del gas") == pytest.approx(r.T_gas_K - C, abs=0.01)
    assert len(at.get("download_button")) == 2


# ---------------------------------------------------------------------
# Unidades, teoría y procedimiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_other_unit_systems(system: str) -> None:
    unit = {"SI": "W", "Inglés": "Btu/h"}[system]
    for mode, what, prefix in (
        (M_TS, None, "Q̇ de 1 a 2"),
        (M_EN, None, "Q̇ de «"),
        (M_CR, W_SURF, "Q̇ por radiación"),
    ):
        at = _system(_mode(_app(), mode, what), system)
        _no_problems(at)
        label = next(m.label for m in at.metric if m.label.startswith(prefix))
        assert label.endswith(f"[{unit}]"), label
        steps = at.expander[1]
        assert steps.label == "🔬 Procedimiento"
        assert len(steps.get("latex")) >= 3


def test_absolute_temperatures_in_english_units() -> None:
    at = _system(_app(), "Inglés")
    _no_problems(at)
    T = at.number_input(key="rb_0_T@Inglés")
    assert T.value == pytest.approx(1440.0)  # 800 K = 1440 °R
    assert _metric(at, "E_b = σT⁴") == pytest.approx(23226.0 * 0.3169983, rel=1e-4)


def test_theory_cites_the_sources() -> None:
    at = _app()
    theory = at.expander[0]
    assert theory.label == "📖 Fórmulas teóricas"
    text = " ".join(m.value for m in theory.markdown)
    assert "vademecum" in text and "Çengel y Ghajar" in text and "Incropera" in text
    assert len(theory.get("latex")) >= 15
