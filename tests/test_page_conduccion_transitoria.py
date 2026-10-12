"""Tests de /Conduccion_Transitoria (AppTest).

Fase 11.1. Los cuatro modos (capacidad concentrada; pared, cilindro y esfera; sólido
semi-infinito y contacto; producto), todos los ejemplos, los cambios de cuerpo, geometría,
condición y forma, la unidad del tiempo, los sistemas de unidades, la teoría, el
procedimiento y la exportación.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.heat_transfer import transient as tr

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "21_Conduccion_Transitoria.py")
C = 273.15

M_LUMP, M_1D, M_SEMI, M_PROD = (
    "Capacidad concentrada",
    "Pared, cilindro largo o esfera",
    "Sólido semi-infinito",
    "Varias dimensiones (producto)",
)
W_CONTACT = "Dos sólidos que se tocan"


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
    value = value.split(" ")[0] if not re.match(r"^-?[\d ]+$", value) else value
    return float(value.replace(" ", "").replace(",", ".").replace("−", "-"))


def _metric_text(at: AppTest, prefix: str) -> str:
    return next(m.value for m in at.metric if m.label.startswith(prefix))


def _mode(at: AppTest, mode: str) -> AppTest:
    at.radio(key="cd_mode").set_value(mode).run()
    _no_problems(at)
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


def _wkey(at: AppTest, key: str) -> str:
    system = at.session_state["units_system"] if "units_system" in at.session_state else "Técnico"
    return f"{key}@{system}"


# ---------------------------------------------------------------------
# Capacidad concentrada
# ---------------------------------------------------------------------


def test_default_is_the_thermocouple() -> None:
    at = _app()
    assert [e.label for e in at.expander] == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert _metric_text(at, "Tiempo t") == "9,94 s"  # el libro: 10 s
    assert _metric(at, "Biot") == pytest.approx(0.001)
    assert len(at.get("plotly_chart")) == 1
    assert len(at.get("download_button")) == 2


@pytest.mark.parametrize("name", list(tr.LUMPED_EXAMPLES))
def test_every_lumped_example(name: str) -> None:
    at = _app()
    at.selectbox(key="tl_example").set_value(name).run()
    _no_problems(at)
    r = tr.solve_lumped(tr.LUMPED_EXAMPLES[name].inputs)
    assert _metric(at, "Biot") == pytest.approx(r.Bi, rel=1e-3)
    if r.inputs.T_target_K is None:
        assert _metric(at, "T a t") == pytest.approx(r.T_K - C, abs=0.01)
    else:
        assert _metric_text(at, "Tiempo t").startswith(tr._time_text(r.time_s).split(" ")[0])


@pytest.mark.parametrize("body", list(tr.BODIES))
def test_every_body(body: str) -> None:
    at = _app()
    at.selectbox(key="tl_0_body").set_value(body).run()
    _no_problems(at)
    assert _metric(at, "Biot") > 0.0
    assert len(at.get("plotly_chart")) == 1


def test_lumped_given_time_and_time_unit() -> None:
    at = _app()
    at.radio(key="tl_0_given").set_value("El tiempo (sale la temperatura)").run()
    _no_problems(at)
    at.selectbox(key="tl_0_t_unit").set_value("min").run()
    _no_problems(at)
    at.number_input(key="tl_0_t@min").set_value(0.5).run()  # 30 s
    _no_problems(at)
    r = tr.solve_lumped(
        tr.LumpedInputs(
            "sphere",
            (0.001,),
            tr.LUMPED_EXAMPLES["Termocupla en un gas (Çengel y Ghajar §4-1)"].inputs.solid,
            210.0,
            25.0 + C,
            200.0 + C,
            time_s=30.0,
        )
    )
    assert _metric(at, "T a t") == pytest.approx(r.T_K - C, abs=0.01)


def test_time_survives_a_change_of_unit() -> None:
    at = _app()
    at.radio(key="tl_0_given").set_value("El tiempo (sale la temperatura)").run()
    at.number_input(key="tl_0_t@s").set_value(90.0).run()
    at.selectbox(key="tl_0_t_unit").set_value("min").run()
    _no_problems(at)
    assert at.number_input(key="tl_0_t@min").value == pytest.approx(1.5)


def test_lumped_error_is_explained() -> None:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    at.number_input(key=_wkey(at, "tl_0_Tt")).set_value(250.0).run()  # fuera de 25 a 200 °C
    assert not at.exception
    assert at.error and "entre la temperatura inicial" in at.error[0].value


def test_custom_material() -> None:
    at = _app()
    at.selectbox(key="tl_0_mat").set_value("Cobre puro").run()
    _no_problems(at)
    copper = _metric(at, "Biot")
    at.selectbox(key="tl_0_mat").set_value("Otro (k, ρ y c a elección)").run()
    _no_problems(at)
    # el «otro» arranca con las propiedades del acero al carbono
    assert _metric(at, "Biot") > copper


# ---------------------------------------------------------------------
# Pared, cilindro y esfera
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(tr.ONE_DIM_EXAMPLES))
def test_every_one_dim_example(name: str) -> None:
    at = _mode(_app(), M_1D)
    at.selectbox(key="tw_example").set_value(name).run()
    _no_problems(at)
    r = tr.solve_one_dimensional(tr.ONE_DIM_EXAMPLES[name].inputs)
    assert _metric(at, "λ₁") == pytest.approx(r.lambda1, rel=1e-4)
    assert _metric(at, "A₁") == pytest.approx(r.A1, rel=1e-4)
    assert len(at.get("plotly_chart")) == 2
    assert at.dataframe  # un término contra la serie


def test_shaft_like_the_book() -> None:
    at = _mode(_app(), M_1D)
    at.selectbox(key="tw_example").set_value("Eje de acero inoxidable (Çengel y Ghajar §4-2)").run()
    _no_problems(at)
    assert _metric(at, "T del centro") == pytest.approx(364.1, abs=0.1)
    assert _metric(at, "Calor entregado Q") == pytest.approx(30124, rel=1e-3)  # kJ/m


@pytest.mark.parametrize("geometry", list(tr.GEOMETRIES))
def test_every_geometry(geometry: str) -> None:
    at = _mode(_app(), M_1D)
    at.radio(key="tw_0_geo").set_value(geometry).run()
    _no_problems(at)
    at.number_input(key="tw_0_pos").set_value(0.5).run()
    _no_problems(at)
    assert any(m.label.startswith("T en la posición") for m in at.metric)


def test_one_dim_short_time_note() -> None:
    at = _mode(_app(), M_1D)
    at.radio(key="tw_0_given").set_value("El tiempo (sale la temperatura)").run()
    at.selectbox(key="tw_0_t_unit").set_value("s").run()
    at.number_input(key="tw_0_t@s").set_value(60.0).run()
    _no_problems(at)
    assert any("τ =" in i.value and "< 0,2" in i.value for i in at.info)


# ---------------------------------------------------------------------
# Semi-infinito y contacto
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(tr.SEMI_INFINITE_EXAMPLES))
def test_every_semi_infinite_example(name: str) -> None:
    at = _mode(_app(), M_SEMI)
    at.selectbox(key="ti_example").set_value(name).run()
    _no_problems(at)
    r = tr.solve_semi_infinite(tr.SEMI_INFINITE_EXAMPLES[name].inputs)
    if r.inputs.T_target_K is not None and r.inputs.depth_m is None:
        assert _metric(at, "Profundidad x") == pytest.approx(r.depth_m * 1000.0, rel=1e-4)
    else:
        assert _metric(at, "T en x y t") == pytest.approx(r.T_K - C, abs=0.01)
    assert len(at.get("plotly_chart")) == 1


@pytest.mark.parametrize("boundary", list(tr.SEMI_BOUNDARIES))
@pytest.mark.parametrize(
    "unknown",
    [
        "La temperatura en x y t",
        "La profundidad a la que llega una temperatura",
        "El tiempo que tarda una temperatura en llegar a x",
    ],
)
def test_every_boundary_and_unknown(boundary: str, unknown: str) -> None:
    at = _mode(_app(), M_SEMI)
    at.selectbox(key="ti_example").set_value(
        "Losa de hormigón con aire caliente (ejemplo propio)"
    ).run()
    idx = list(tr.SEMI_INFINITE_EXAMPLES).index(
        "Losa de hormigón con aire caliente (ejemplo propio)"
    )
    at.radio(key=f"ti_{idx}_bnd").set_value(boundary).run()
    at.radio(key=f"ti_{idx}_unk").set_value(unknown).run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) == 1


@pytest.mark.parametrize("name", list(tr.CONTACT_EXAMPLES))
def test_every_contact_example(name: str) -> None:
    at = _mode(_app(), M_SEMI)
    at.radio(key="ti_what").set_value(W_CONTACT).run()
    at.selectbox(key="tc_example").set_value(name).run()
    _no_problems(at)
    r = tr.solve_contact(tr.CONTACT_EXAMPLES[name].inputs)
    assert _metric(at, "Temperatura de contacto") == pytest.approx(r.T_contact_K - C, abs=0.01)
    assert len(at.get("plotly_chart")) == 1
    at.checkbox(key=f"tc_{list(tr.CONTACT_EXAMPLES).index(name)}_q").uncheck().run()
    _no_problems(at)
    assert not any(m.label.startswith("q̇ de A a B") for m in at.metric)


# ---------------------------------------------------------------------
# Producto
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(tr.PRODUCT_EXAMPLES))
def test_every_product_example(name: str) -> None:
    at = _mode(_app(), M_PROD)
    at.selectbox(key="tp_example").set_value(name).run()
    _no_problems(at)
    r = tr.solve_product(tr.PRODUCT_EXAMPLES[name].inputs)
    assert _metric(at, "T en el punto") == pytest.approx(r.T_K - C, abs=0.01)
    assert len(at.get("plotly_chart")) == 1


@pytest.mark.parametrize("shape", list(tr.PRODUCT_SHAPES))
def test_every_shape(shape: str) -> None:
    at = _mode(_app(), M_PROD)
    at.selectbox(key="tp_0_shape").set_value(shape).run()
    _no_problems(at)
    has_semi = shape in ("semi_cylinder", "semi_plate", "quarter", "corner")
    assert any(m.label.startswith("Q/Q_máx") for m in at.metric) != has_semi


# ---------------------------------------------------------------------
# Unidades, teoría y procedimiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_every_mode_in_other_systems(system: str) -> None:
    at = _system(_app(), system)
    _no_problems(at)
    for mode in (M_LUMP, M_1D, M_SEMI, M_PROD):
        _mode(at, mode)
    at = _mode(at, M_SEMI)
    at.radio(key="ti_what").set_value(W_CONTACT).run()
    _no_problems(at)


def test_values_survive_a_change_of_system() -> None:
    at = _app()
    at.number_input(key=_wkey(at, "tl_0_Tinf")).set_value(300.0).run()
    _no_problems(at)
    _system(at, "SI")
    _no_problems(at)
    assert at.number_input(key=_wkey(at, "tl_0_Tinf")).value == pytest.approx(573.15)


def test_theory_says_the_vademecum_has_no_chapter() -> None:
    at = _app()
    theory = at.expander[0]
    text = " ".join(m.value for m in theory.markdown)
    assert "todavía no tiene un capítulo de transferencia de calor" in text
    for source in ("Çengel y Ghajar", "Incropera", "Heisler", "Carslaw y Jaeger", "Langston"):
        assert source in text
    for section in ("§4-1", "§4-2", "§4-3", "§4-4"):
        assert section in text
    assert len(theory.latex) >= 14


@pytest.mark.parametrize(
    ("mode", "last"),
    [
        (M_LUMP, "Calor"),
        (M_1D, "Serie completa"),
        (M_SEMI, "Profundidad de penetración"),
        (M_PROD, "Calor"),
    ],
)
def test_procedure_steps(mode: str, last: str) -> None:
    at = _mode(_app(), mode)
    proc = at.expander[1]
    titles = [m.value for m in proc.markdown if re.match(r"\*\*\d+\. ", m.value)]
    assert titles and titles[-1].endswith(f"{last}**")
    assert proc.latex
