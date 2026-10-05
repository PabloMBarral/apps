"""Tests de la página de Rankine con ``streamlit.testing`` (AppTest) — Fase 3.1a."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.cycles.rankine import RANKINE_EXAMPLES, solve_rankine

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "5_Rankine.py")
EXAMPLES = list(RANKINE_EXAMPLES)


def _new_app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _states(at: AppTest) -> list[str]:
    frame = next(df.value for df in at.dataframe if "Estado" in df.value.columns)
    return list(frame["Estado"])


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert not any("No se pudo dibujar" in w.value for w in at.warning)


def test_default_example_is_computed_on_load() -> None:
    at = _new_app()
    _no_problems(at)
    assert float(_metric(at, "η térmico [%]")) == pytest.approx(26.0, abs=0.2)  # Cengel 10-1
    assert _metric(at, "Título a la salida") == "0.886"
    assert len(_states(at)) == 4


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_example_computes(example: str) -> None:
    at = _new_app()
    at.selectbox(key="rk_example").set_value(example).run()
    _no_problems(at)
    assert len(_states(at)) == len(solve_rankine(RANKINE_EXAMPLES[example]).states)


def test_reheat_checkbox_adds_two_states() -> None:
    at = _new_app()
    at.checkbox(key="rk_reheat_0").check().run()
    at.button(key="rk_btn").click().run()
    _no_problems(at)
    assert len(_states(at)) == 6


def test_condenser_label_follows_the_state_numbering() -> None:
    at = _new_app()
    condenser = at.number_input(key="rk_p_lo_0@Técnico")
    assert condenser.label.startswith("Presión del condensador p₄")
    condenser.set_value(0.2).run()
    at.checkbox(key="rk_reheat_0").check().run()
    condenser = at.number_input(key="rk_p_lo_0@Técnico")
    assert condenser.label.startswith("Presión del condensador p₆")  # 5 → 6: turbina de baja
    assert condenser.value == pytest.approx(0.2)  # cambiar el rótulo no pierde el dato


def test_saturated_inlet_hides_the_temperature() -> None:
    at = _new_app()
    at.radio(key="rk_inlet_0").set_value("Vapor saturado seco (x = 1)").run()
    assert not any(n.key == "rk_T_in_0@Técnico" for n in at.number_input)
    at.button(key="rk_btn").click().run()
    _no_problems(at)
    assert _states(at)[2].startswith("3 ")


def test_net_power_gives_the_mass_flow() -> None:
    at = _new_app()
    at.radio(key="rk_flow_0").set_value("Potencia neta").run()
    at.button(key="rk_btn").click().run()
    _no_problems(at)
    assert _metric(at, "Ẇ neto [kW]") == "100 000"  # 100 MW por defecto, sin "1e+05"


def test_invalid_condenser_pressure_is_explained() -> None:
    at = _new_app()
    at.number_input(key="rk_p_lo_0@Técnico").set_value(40.0)  # bar, más que la caldera
    at.button(key="rk_btn").click().run()
    assert at.error and "menor que la de la caldera" in at.error[0].value


def test_edited_data_ask_for_recalculation() -> None:
    at = _new_app()
    at.number_input(key="rk_eta_t_0").set_value(0.85).run()
    assert any("Cambiaste datos" in c.value for c in at.caption)


@pytest.mark.parametrize(
    ("system", "unit"), [("SI", r"\mathrm{J/kg}"), ("Inglés", r"\mathrm{Btu/lb}")]
)
def test_procedure_follows_the_unit_system(system: str, unit: str) -> None:
    at = _new_app()
    at.selectbox(key="units_system").set_value(system).run()
    _no_problems(at)
    assert any(unit in tex.value for tex in at.latex)


def test_theory_procedure_and_export_are_present() -> None:
    at = _new_app()
    labels = [e.label for e in at.expander]
    assert any("Fórmulas teóricas" in label for label in labels)
    assert any("Procedimiento" in label for label in labels)
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    text = " ".join(md.value for md in theory.markdown)
    assert "§9.2" in text and "§10.4" in text
    assert len(at.get("download_button")) == 2


def test_result_survives_changing_the_diagram() -> None:
    at = _new_app()
    at.selectbox(key="rk_diagram_type").set_value("logph").run()
    _no_problems(at)
    assert float(_metric(at, "η térmico [%]")) == pytest.approx(26.0, abs=0.2)


def test_sweep_draws_the_charts() -> None:
    at = _new_app()
    at.selectbox(key="rk_sweep_param").set_value("Presión de la caldera").run()
    at.button(key="rk_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) >= 3  # diagrama + η + título


# ---------------------------------------------------------------------
# Regeneración (Fase 3.1b)
# ---------------------------------------------------------------------

_I_10_5 = next(i for i, k in enumerate(EXAMPLES) if k.startswith("Cengel 10-5"))
_I_10_6 = next(i for i, k in enumerate(EXAMPLES) if k.startswith("Cengel 10-6"))


def _example(at: AppTest, index: int) -> AppTest:
    return at.selectbox(key="rk_example").set_value(EXAMPLES[index]).run()


def _labels(at: AppTest) -> list[str]:
    return [n.label for n in at.number_input]


def test_cengel_10_5_labels_follow_the_book_numbering() -> None:
    at = _example(_new_app(), _I_10_5)
    _no_problems(at)
    labels = _labels(at)
    assert any(lab.startswith("Presión de la caldera p₅") for lab in labels)
    assert any(lab.startswith("Presión del condensador p₇") for lab in labels)
    assert any(lab.startswith("Presión de extracción p₆") for lab in labels)
    assert float(_metric(at, "η térmico [%]")) == pytest.approx(46.31, abs=0.02)
    table = next(df.value for df in at.dataframe if "Calentador" in df.value.columns)
    assert list(table["Calentador"]) == ["calentador abierto"]


def test_cengel_10_6_shows_the_book_note_and_the_mixing_chamber() -> None:
    at = _example(_new_app(), _I_10_6)
    _no_problems(at)
    assert len(_states(at)) == 13
    assert any("643,9" in c.value for c in at.caption)  # errata del libro
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    titles = " ".join(md.value for md in procedure.markdown)
    assert "cámara de mezcla" in titles and "fracción de extracción" in titles
    assert at.checkbox(key=f"rk_fwd_{_I_10_6}_2").value is True


@pytest.mark.parametrize("n", [1, 2, 3])
def test_regeneration_with_one_two_or_three_heaters(n: int) -> None:
    at = _new_app()
    at.checkbox(key="rk_regen_0").check().run()
    at.radio(key="rk_nheat_0").set_value(n).run()
    at.button(key="rk_btn").click().run()
    _no_problems(at)
    table = next(df.value for df in at.dataframe if "Calentador" in df.value.columns)
    assert len(table) == n
    assert all(0 < float(y) < 1 for y in table["y = ṁext/ṁ"])


def test_closed_top_heater_offers_the_forward_drain() -> None:
    at = _new_app()
    at.checkbox(key="rk_regen_0").check().run()
    at.radio(key="rk_nheat_0").set_value(2).run()
    assert not any(c.key == "rk_fwd_0_2" for c in at.checkbox)
    at.radio(key="rk_h1_kind_0_2").set_value("Cerrado").run()
    at.checkbox(key="rk_fwd_0_2").check().run()
    at.button(key="rk_btn").click().run()
    _no_problems(at)
    assert any("cámara de mezcla" in label for label in _states(at))


def test_invalid_extraction_pressure_is_explained() -> None:
    at = _new_app()
    at.checkbox(key="rk_regen_0").check().run()
    at.number_input(key="rk_h0_p_0_1@Técnico").set_value(50.0)  # más que la caldera (30 bar)
    at.button(key="rk_btn").click().run()
    assert at.error and "presión de extracción" in at.error[0].value


def test_cooling_water() -> None:
    at = _new_app()
    at.checkbox(key="rk_cw_0").check().run()
    _no_problems(at)
    flow = float(_metric(at, "ṁ agua de enfriamiento [kg/s]"))
    assert flow == pytest.approx(2018.6 / (4.18 * 10), rel=5e-3)  # q_C/(c_p·ΔT), 10-1
    at.number_input(key="rk_cw_out_0@Técnico").set_value(95.0).run()  # T_sat(75 kPa) = 91,8 °C
    assert any("no puede salir más caliente" in w.value for w in at.warning)


def test_extraction_sweep_draws_the_charts() -> None:
    at = _example(_new_app(), _I_10_5)
    at.selectbox(key="rk_sweep_param").set_value(
        "Presión de extracción del calentador abierto"
    ).run()
    at.button(key="rk_sweep_btn").click().run()
    _no_problems(at)
    assert len(at.get("plotly_chart")) >= 3  # diagrama + η + y
