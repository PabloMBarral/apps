"""Tests de /Poder_Calorifico (AppTest).

Fase 6. Todos los ejemplos, las tres bases, los tres modos de análisis, el O y
el CF por diferencia, el H aparte, la referencia (medida en cualquier base o
exacta), la correlación principal, los mensajes al alumno, los sistemas de
unidades, la teoría, el procedimiento, la exportación, la vista de los datos de
validación y el barrido de la humedad.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.combustion.heating_value import (
    CORRELATIONS,
    HEATING_VALUE_EXAMPLES,
    solve_heating_value,
)

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "9_Poder_Calorifico.py")
NAMES = list(HEATING_VALUE_EXAMPLES)
BAGASSE = "Bagazo de caña (50 % de humedad)"
PITTSBURGH = "Carbón bituminoso Pittsburgh N.º 8"
POCAHONTAS = "Carbón bituminoso Pocahontas N.º 3 (solo inmediato)"
METHANE = "Metano (sustancia pura)"
PCS_D = "PCS seco [kJ/kg]"
PCI_AR = "PCI tal cual [kJ/kg]"


def _app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    _no_problems(at)
    return at


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _metric(at: AppTest, label: str) -> float:
    value = next(m.value for m in at.metric if m.label == label)
    return float(value.replace(" ", "").replace("%", ""))


def _example(at: AppTest, name: str) -> str:
    at.selectbox(key="hv_example").set_value(name).run()
    return f"hv_{NAMES.index(name)}"


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


@pytest.mark.parametrize("name", NAMES)
def test_every_example(name: str) -> None:
    at = _app()
    _example(at, name)
    _no_problems(at)
    result = solve_heating_value(HEATING_VALUE_EXAMPLES[name])
    assert _metric(at, PCS_D) == pytest.approx(result.main.hhv_d / 1e3, rel=1e-4)
    if result.main.lhv_ar is not None:
        assert _metric(at, PCI_AR) == pytest.approx(result.main.lhv_ar / 1e3, rel=1e-4)
    assert any(c.value.startswith("📘 Sobre este ejemplo") for c in at.caption)
    assert len(at.get("download_button")) == 2
    assert _charts(at) == 4  # comparación, paridad, error contra el O y humedad
    labels = [e.label for e in at.expander]
    assert labels == [
        "📖 Fórmulas teóricas",
        "🎯 ¿Qué tan buenas son las correlaciones?",
        "🔬 Procedimiento",
        "📊 ¿Cómo cambia con la humedad?",
    ]


def test_default_is_bagasse_with_its_measurement() -> None:
    at = _app()
    assert at.selectbox(key="hv_example").value == BAGASSE
    assert _metric(at, "PCS medido seco [kJ/kg]") == pytest.approx(18990.0)
    assert _metric(at, "Desvío de la principal") == pytest.approx(2.3, abs=0.05)
    assert _metric(at, "Humedad con PCI nulo W*") == pytest.approx(88.1, abs=0.05)
    assert any("H disponible" in i.value for i in at.info)


@pytest.mark.parametrize("basis", ["Tal cual (como se recibe)", "Seca y sin cenizas"])
def test_changing_the_basis_keeps_the_fuel(basis: str) -> None:
    """Al cambiar de base, los datos nuevos salen del último resultado: mismo PCS."""
    at = _app()
    key = _example(at, PITTSBURGH)
    before = _metric(at, PCS_D)
    pci = _metric(at, PCI_AR)
    at.radio(key=f"{key}_basis").set_value(basis).run()
    _no_problems(at)
    assert _metric(at, PCS_D) == pytest.approx(before, rel=1e-4)
    assert _metric(at, PCI_AR) == pytest.approx(pci, rel=1e-4)
    code = "ar" if basis.startswith("Tal") else "daf"
    c = at.number_input(key=f"{key}_{code}_C").value
    expected = 75.50 * (1 - 0.0165) if code == "ar" else 75.50 / (1 - 0.0925)
    assert c == pytest.approx(expected, abs=1e-3)


def test_analysis_modes() -> None:
    at = _app()
    key = _example(at, PITTSBURGH)
    table = at.dataframe[0].value
    assert list(table["Correlación"])[:5] == [c.name for c in CORRELATIONS.values()]
    at.radio(key=f"{key}_an").set_value("Inmediato (próximo)").run()
    _no_problems(at)
    assert at.selectbox(key=f"{key}_main_p").value == "Parikh et al. (2005)"
    assert next(m.value for m in at.metric if m.label == PCI_AR) == "—"
    assert any("no se conoce el H" in i.value for i in at.info)
    at.checkbox(key=f"{key}_hknown").check().run()
    at.number_input(key=f"{key}_Hd").set_value(4.83).run()
    _no_problems(at)
    assert _metric(at, PCI_AR) > 0
    at.radio(key=f"{key}_an").set_value("Elemental (último)").run()
    _no_problems(at)
    assert at.selectbox(key=f"{key}_main_u").value == "Channiwala y Parikh (2002)"


def test_oxygen_and_fixed_carbon_by_difference() -> None:
    at = _app()
    key = _example(at, PITTSBURGH)
    before = _metric(at, PCS_D)
    at.checkbox(key=f"{key}_odiff").check().run()
    _no_problems(at)
    assert f"{key}_d_O" not in [n.key for n in at.number_input]
    assert _metric(at, PCS_D) == pytest.approx(before, rel=1e-4)
    # El CF va por diferencia por defecto (ASTM D3172): no hay campo.
    assert f"{key}_d_FC" not in [n.key for n in at.number_input]
    at.checkbox(key=f"{key}_fcdiff").uncheck().run()
    _no_problems(at)
    assert f"{key}_d_FC" in [n.key for n in at.number_input]


def test_sum_errors_explain_why() -> None:
    at = _app()
    key = _example(at, BAGASSE)
    at.number_input(key=f"{key}_d_C").set_value(60.0).run()
    assert not at.exception
    assert any("tiene que dar 100 %" in e.value for e in at.error)
    at.number_input(key=f"{key}_d_C").set_value(48.64).run()
    _no_problems(at)
    at.checkbox(key=f"{key}_odiff").check().run()
    at.number_input(key=f"{key}_d_C").set_value(95.0).run()
    assert any("O por diferencia daría" in e.value for e in at.error)


def test_reference_in_another_basis() -> None:
    """El PCS medido tal cual (con la humedad) da el mismo desvío que el seco."""
    at = _app()
    key = _example(at, BAGASSE)
    dev = _metric(at, "Desvío de la principal")
    at.selectbox(key=f"{key}_refb").set_value("Tal cual (como se recibe)").run()
    _no_problems(at)
    assert at.number_input(key=f"{key}_refv_ar@Técnico").value == pytest.approx(18990.0 * 0.5)
    assert _metric(at, "Desvío de la principal") == pytest.approx(dev, abs=0.05)
    at.checkbox(key=f"{key}_ref").uncheck().run()
    _no_problems(at)
    assert not any(m.label == "Desvío de la principal" for m in at.metric)


def test_exact_reference_only_for_the_pure_substance() -> None:
    at = _app()
    key = _example(at, METHANE)
    assert any("PCS exacto en base seca" in c.value for c in at.caption)
    assert _metric(at, "PCS exacto seco [kJ/kg]") == pytest.approx(55513.0, rel=1e-4)
    at.number_input(key=f"{key}_d_C").set_value(70.0).run()
    at.number_input(key=f"{key}_d_O").set_value(4.87).run()
    _no_problems(at)
    assert any("ya no corresponde" in c.value for c in at.caption)
    assert not any(m.label == "PCS exacto seco [kJ/kg]" for m in at.metric)


def test_main_correlation() -> None:
    at = _app()
    key = _example(at, BAGASSE)
    at.selectbox(key=f"{key}_main_u").set_value("Boie (1953)").run()
    _no_problems(at)
    result = solve_heating_value(HEATING_VALUE_EXAMPLES[BAGASSE])
    assert _metric(at, PCS_D) == pytest.approx(result.estimate("boie").hhv_d / 1e3, rel=1e-4)


def test_fuel_type_flags_in_the_table() -> None:
    at = _app()
    key = _example(at, BAGASSE)
    table = at.dataframe[0].value
    dulong = table[table["Correlación"] == "Dulong"].iloc[0]
    assert "fuera de su tipo" in dulong["Avisos"] and "O = 42,8 %" in dulong["Avisos"]
    at.selectbox(key=f"{key}_type").set_value("Carbón mineral o lignito").run()
    _no_problems(at)
    table = at.dataframe[0].value
    dulong = table[table["Correlación"] == "Dulong"].iloc[0]
    assert "fuera de su tipo" not in dulong["Avisos"]


def test_bases_table() -> None:
    at = _app()
    _example(at, BAGASSE)
    table = at.dataframe[1].value
    assert list(table["Componente"]) == ["C", "H", "O", "N", "S", "Cenizas", "Humedad"]
    c = table[table["Componente"] == "C"].iloc[0]
    assert float(c["Tal cual [%]"]) == pytest.approx(24.32, abs=0.01)
    assert float(c["Seca [%]"]) == pytest.approx(48.64, abs=0.01)
    assert any("kg de agua por kg seco" in cap.value for cap in at.caption)


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_unit_systems(system: str) -> None:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.session_state["units_system"] = system
    at.run()
    _no_problems(at)
    result = solve_heating_value(HEATING_VALUE_EXAMPLES[BAGASSE])
    if system == "SI":
        assert _metric(at, "PCS seco [J/kg]") == pytest.approx(result.main.hhv_d, rel=1e-4)
        ref_key = "hv_0_refv_d@SI"
        assert at.number_input(key=ref_key).value == pytest.approx(18.99e6)
    else:
        assert _metric(at, "PCS seco [Btu/lb]") == pytest.approx(
            result.main.hhv_d / 2326.0, rel=1e-4
        )
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    assert len(procedure.get("latex")) >= 10


def test_theory_and_procedure() -> None:
    at = _app()
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    tex = [el.value for el in theory.get("latex")]
    assert len(tex) >= 12
    assert any("0.3491" in t for t in tex)
    text = " ".join(m.value for m in theory.markdown)
    assert "§16.9" in text and "ASTM D3180-25" in text and "Ghugare" not in text
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    assert len(procedure.get("latex")) >= 10
    steps = [m.value for m in procedure.markdown if m.value.startswith("**")]
    assert steps[0] == "**1. Análisis en base seca**"


def test_dataset_view() -> None:
    at = _app()
    for label in [c.name for k, c in CORRELATIONS.items() if c.analysis == "ultimate"]:
        at.selectbox(key="hv_ds_corr").set_value(label).run()
        _no_problems(at)
    tables = at.dataframe
    metrics = tables[2].value
    assert list(metrics["Correlación"]) == ["Channiwala y Parikh (2002)", "Boie (1953)", "Dulong"]
    assert list(metrics["AAE"]) == ["4.94 %", "5.23 %", "11.46 %"]
    coals = tables[3].value
    assert len(coals) == 5
    assert list(coals.columns) == ["Carbón", "CyP", "Boie", "Dulong", "Parikh", "Cordero", "Rango"]
    pitt = coals[coals["Carbón"] == "Pittsburgh N.º 8"].iloc[0]
    assert pitt["Parikh"] == "-22.6 %" and pitt["Boie"] == "+0.1 %"


def test_moisture_sweep_caption() -> None:
    at = _app()
    sweep = next(e for e in at.expander if "humedad" in e.label)
    assert any("W* = 88.1 %" in c.value for c in sweep.caption)
