"""Tests de /Combustion: combustión y análisis de humos (AppTest).

Fase 5. Todos los ejemplos de los dos modos, las tres clases de combustible,
cada forma de dar la cantidad de aire, el aire húmedo y precalentado, la
bomba, el equilibrio, los mensajes al alumno, los sistemas de unidades, el
procedimiento, la teoría, la exportación y los barridos.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.combustion.combustion import (
    COMBUSTION_EXAMPLES,
    FLUE_GAS_EXAMPLES,
    solve_combustion,
    solve_flue_gas,
)
from core.combustion.fuels import FUEL_KINDS, FUELS, Fuel
from core.combustion.stoichiometry import AIR_SPEC_KINDS

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "8_Combustion.py")
COMB = list(COMBUSTION_EXAMPLES)
FLUE = list(FLUE_GAS_EXAMPLES)
C = 273.15
BOILER = "Caldera de gas natural (3 % de O₂, humos a 150 °C)"
CONDENSING = "Caldera de condensación (humos a 45 °C)"
METHANE_50 = "Cengel 15-10 y 15-11: metano con 50 % de exceso de aire"
BOMB = "Cengel 15-7: metano y oxígeno en un recipiente rígido"
OCTANE = "Cengel 15-8: llama adiabática del octano"
FUEL_OIL = "Fueloil en una caldera (rocío ácido)"
T_AD = "T_ad completa [°C]"
T_EQ = "T_ad con disociación [°C]"


def _app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    _no_problems(at)
    return at


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert not any("No se pudo dibujar" in w.value for w in at.warning)


def _metric(at: AppTest, label: str) -> float:
    value = next(m.value for m in at.metric if m.label == label)
    return float(value.replace(" ", ""))


def _example(at: AppTest, name: str) -> str:
    """Elige un ejemplo de combustión y devuelve el prefijo de sus keys."""
    at.selectbox(key="cb_example").set_value(name).run()
    return f"cb_{COMB.index(name)}"


def _flue(at: AppTest, name: str) -> str:
    at.radio(key="cb_mode").set_value("Análisis de humos").run()
    at.selectbox(key="fg_example").set_value(name).run()
    return f"fg_{FLUE.index(name)}"


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


# ---------------------------------------------------------------------
# Combustión
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", COMB)
def test_every_combustion_example(name: str) -> None:
    at = _app()
    _example(at, name)
    _no_problems(at)
    result = solve_combustion(COMBUSTION_EXAMPLES[name])
    assert _metric(at, "λ") == pytest.approx(result.stoich.lam, rel=1e-3)
    assert _metric(at, T_AD) == pytest.approx(result.flame.T_K - C, abs=0.6)
    if result.flame_eq is not None:
        assert _metric(at, T_EQ) == pytest.approx(result.flame_eq.T_K - C, abs=0.6)
    assert _metric(at, "PCI [kJ/kg]") == pytest.approx(result.lhv * result.per_kg / 1e3, rel=1e-4)
    assert any(c.value.startswith("📘 Sobre este ejemplo") for c in at.caption)
    assert len(at.get("download_button")) == 2
    has_carbon = result.fuel.elements()["C"] > 0.0
    charts = 1 + has_carbon + (result.heat is not None and result.heat.split is not None)
    assert _charts(at) == charts
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    assert len(procedure.get("latex")) > 20


def test_default_example_is_cengel_15_1() -> None:
    at = _app()
    assert at.selectbox(key="cb_example").value == COMB[0]
    assert _metric(at, "AC [kg/kg]") == pytest.approx(24.05, abs=0.01)
    assert _metric(at, "λ") == pytest.approx(1.6)


def test_methane_with_50_percent_excess_is_cengel_15_10_and_15_11() -> None:
    at = _app()
    _example(at, METHANE_50)
    assert _metric(at, T_AD) == pytest.approx(1789 - C, abs=1.5)
    assert _metric(at, "Calor por kmol [kJ/kmol]") == pytest.approx(871_400, rel=2e-3)
    assert _metric(at, "Agua que condensa [%]") > 70.0
    assert _metric(at, "S_gen con el calor [kJ/(kmol·K)]") == pytest.approx(2746, rel=2e-3)
    assert _metric(at, "X_dest con el calor [kJ/kmol]") == pytest.approx(818_000, rel=2e-3)


def test_condensing_boiler_passes_100_percent_on_the_lhv() -> None:
    at = _app()
    _example(at, CONDENSING)
    _no_problems(at)
    assert _metric(at, "η sobre el PCI [%]") > 100.0
    assert _metric(at, "η sobre el PCS [%]") < 100.0
    assert any("no es un error" in i.value for i in at.info)


def test_fuel_classes() -> None:
    at = _app()  # 15-1: octano líquido
    at.radio(key="cb_0_kind").set_value(FUEL_KINDS["gas"]).run()
    _no_problems(at)
    assert at.selectbox(key="cb_0_gas").value == "Metano"
    assert _metric(at, "PCI [kJ/kg]") == pytest.approx(FUELS["Metano"].lhv_per_kg / 1e3, rel=1e-4)
    at.radio(key="cb_0_kind").set_value(FUEL_KINDS["analysis"]).run()
    _no_problems(at)
    assert _metric(at, "PCI [kJ/kg]") == pytest.approx(FUELS["Fueloil"].lhv_per_kg / 1e3, rel=1e-4)
    at.selectbox(key="cb_0_an").set_value("Carbón (Pensilvania)").run()
    _no_problems(at)
    coal = FUELS["Carbón (Pensilvania)"].lhv_per_kg / 1e3
    assert _metric(at, "PCI [kJ/kg]") == pytest.approx(coal, rel=1e-4)
    at.radio(key="cb_0_kind").set_value(FUEL_KINDS["liquid"]).run()
    at.selectbox(key="cb_0_liq").set_value("C2H5OH(l)").run()
    _no_problems(at)
    ethanol = FUELS["Etanol (líquido)"].lhv_per_kg / 1e3
    assert _metric(at, "PCI [kJ/kg]") == pytest.approx(ethanol, rel=1e-4)


def test_gas_composition_can_be_edited() -> None:
    at = _app()
    key = _example(at, BOILER)
    gkey = f"{key}_gas1"  # el gas natural típico es el segundo de la lista
    at.multiselect(key=f"{gkey}_comp").set_value(["CH4", "H2"]).run()
    at.number_input(key=f"{gkey}_H2").set_value(9.0).run()
    _no_problems(at)
    expected = Fuel.mixture("x", {"CH4": 0.91, "H2": 0.09})
    assert _metric(at, "PCI [kJ/kg]") == pytest.approx(expected.lhv_per_kg / 1e3, rel=1e-4)
    at.number_input(key=f"{gkey}_H2").set_value(19.0).run()
    _no_problems(at)
    assert any("se normalizan" in c.value for c in at.caption)


def test_fuel_errors_explain_why() -> None:
    at = _app()
    key = _example(at, BOILER)
    at.multiselect(key=f"{key}_gas1_comp").set_value(["N2", "CO2"]).run()
    assert not at.exception
    assert any("nada que se queme" in e.value for e in at.error)
    key = _example(at, FUEL_OIL)
    at.number_input(key=f"{key}_an0_C").set_value(80.0).run()
    assert not at.exception
    assert any("tiene que dar 100 %" in e.value for e in at.error)


@pytest.mark.parametrize("kind", list(AIR_SPEC_KINDS))
def test_every_air_spec_keeps_the_result(kind: str) -> None:
    """Al cambiar la forma de dar el aire, el valor nuevo sale del último resultado."""
    at = _app()
    key = _example(at, BOILER)
    lam = _metric(at, "λ")
    at.selectbox(key=f"{key}_spec").set_value(AIR_SPEC_KINDS[kind]).run()  # type: ignore[index]
    _no_problems(at)
    assert _metric(at, "λ") == pytest.approx(lam, rel=1e-4)


def test_air_spec_values_change_the_air() -> None:
    at = _app()
    key = _example(at, METHANE_50)
    at.number_input(key=f"{key}_air_excess").set_value(100.0).run()
    _no_problems(at)
    assert _metric(at, "λ") == pytest.approx(2.0)
    at.selectbox(key=f"{key}_spec").set_value(AIR_SPEC_KINDS["equivalence"]).run()
    at.number_input(key=f"{key}_air_equivalence").set_value(1.25).run()
    _no_problems(at)
    assert _metric(at, "λ") == pytest.approx(0.8)
    assert any("defecto de aire" in i.value for i in at.info)


def test_humid_and_preheated_air() -> None:
    at = _app()
    key = _example(at, BOILER)
    cold = _metric(at, T_AD)
    dew = _metric(at, "T_pr del agua [°C]")
    at.checkbox(key=f"{key}_hot").check().run()
    at.number_input(key=f"{key}_Ta@Técnico").set_value(300.0).run()
    _no_problems(at)
    assert _metric(at, T_AD) > cold + 150.0
    # el vapor del aire es el del ambiente (20 °C y 60 %): el rocío no cambia
    assert _metric(at, "T_pr del agua [°C]") == pytest.approx(dew, abs=0.01)
    assert any("precalentado" in i.value for i in at.info)
    at.number_input(key=f"{key}_phi").set_value(100.0).run()
    _no_problems(at)
    assert _metric(at, "T_pr del agua [°C]") > dew


def test_bomb_at_constant_volume_and_pure_oxygen() -> None:
    at = _app()
    _example(at, BOMB)
    _no_problems(at)
    result = solve_combustion(COMBUSTION_EXAMPLES[BOMB])
    assert result.heat is not None
    assert _metric(at, "p al final de la llama [bar]") == pytest.approx(
        result.flame.p_Pa / 1e5, rel=1e-4
    )
    # J/mol = kJ/kmol; Cengel: 308 730 Btu/lbmol = 718 100 kJ/kmol
    assert _metric(at, "Calor por kmol [kJ/kmol]") == pytest.approx(result.heat.Q_out, rel=1e-5)
    assert _metric(at, "Calor por kmol [kJ/kmol]") == pytest.approx(308_730 * 2.326, rel=1e-3)
    assert any("Oxígeno puro" in r.value for r in at.radio if r.key == "cb_4_ox")
    assert any("oxígeno puro" in i.value for i in at.info)
    assert any("balance de energía interna" in c.value for c in at.caption)


def test_equilibrium_can_be_turned_off() -> None:
    at = _app()
    key = _example(at, OCTANE)
    assert any("Verificación con las constantes" in m.value for m in at.markdown)
    at.checkbox(key=f"{key}_eq").uncheck().run()
    _no_problems(at)
    assert next(m.value for m in at.metric if m.label == T_EQ) == "—"
    assert not any("Verificación con las constantes" in m.value for m in at.markdown)


def test_sulfur_shows_the_acid_dew_point() -> None:
    at = _app()
    key = _example(at, FUEL_OIL)
    stoich = next(df.value for df in at.dataframe if "Símbolo" in df.value.columns)
    assert "T_pr,a" in list(stoich["Símbolo"])
    before = stoich.set_index("Símbolo").loc["T_pr,a", "Valor"]
    at.number_input(key=f"{key}_so3").set_value(5.0).run()
    _no_problems(at)
    stoich = next(df.value for df in at.dataframe if "Símbolo" in df.value.columns)
    assert float(stoich.set_index("Símbolo").loc["T_pr,a", "Valor"]) > float(before)


def test_invalid_data_show_a_message() -> None:
    at = _app()
    key = _example(at, METHANE_50)
    at.number_input(key=f"{key}_Ts@Técnico").set_value(2000.0).run()
    assert not at.exception
    assert any("no habría calor" in e.value for e in at.error)
    key = _example(at, OCTANE)
    at.selectbox(key=f"{key}_spec").set_value(AIR_SPEC_KINDS["lambda"]).run()
    at.number_input(key=f"{key}_air_lambda").set_value(0.5).run()
    assert not at.exception
    assert any("hollín" in e.value for e in at.error)


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_other_unit_systems(system: str) -> None:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.session_state["units_system"] = system
    at.run()
    _example(at, BOILER)
    _no_problems(at)
    result = solve_combustion(COMBUSTION_EXAMPLES[BOILER])
    ratio = "lb/lb" if system == "Inglés" else "kg/kg"
    assert _metric(at, f"AC [{ratio}]") == pytest.approx(result.stoich.AC, rel=1e-3)
    if system == "SI":
        assert _metric(at, "PCI [J/kg]") == pytest.approx(result.lhv * result.per_kg, rel=1e-4)
        assert _metric(at, "T_ad completa [K]") == pytest.approx(result.flame.T_K, abs=1.0)
    else:
        lhv = result.lhv * result.per_kg / 2326.0
        assert _metric(at, "PCI [Btu/lb]") == pytest.approx(lhv, rel=1e-4)
        assert _metric(at, "T_ad completa [°F]") == pytest.approx(
            result.flame.T_K * 1.8 - 459.67, abs=1.0
        )
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    assert len(procedure.get("latex")) > 20


def test_theory_expander_has_the_formulas() -> None:
    at = _app()
    theory = next(e for e in at.expander if "Fórmulas teóricas" in e.label)
    latex = [e.value for e in theory.get("latex")]
    assert any(r"a_s = \frac{\mathrm{O_{2,t}}}" in tex for tex in latex)
    assert any(r"\ln K_p" in tex for tex in latex)
    assert any("2.276" in tex for tex in latex)  # rocío ácido (Verhoff & Banchero)
    assert any("§16" in m.value and "§4.8" in m.value for m in theory.markdown)
    table = next(df.value for df in theory.dataframe)
    assert "h̄_f [kJ/kmol]" in table.columns and len(table) == 15


@pytest.mark.parametrize(
    ("choice", "added"),
    [
        ("Exceso de aire λ", 5),
        ("Temperatura del comburente (precalentamiento)", 1),
        ("Temperatura de los humos a la salida", 2),
    ],
)
def test_sweeps(choice: str, added: int) -> None:
    at = _app()
    _example(at, BOILER)
    before = _charts(at)
    at.selectbox(key="cb_sweep_kind").set_value(choice).run()
    at.button(key="cb_sweep_btn").click().run()
    _no_problems(at)
    assert _charts(at) == before + added
    assert any("La línea punteada marca tus datos" in c.value for c in at.caption)


def test_stack_sweep_needs_constant_pressure() -> None:
    at = _app()
    _example(at, BOMB)
    at.selectbox(key="cb_sweep_kind").set_value("Temperatura de los humos a la salida").run()
    _no_problems(at)
    assert not [b for b in at.button if b.key == "cb_sweep_btn"]
    assert any("presión constante" in c.value for c in at.caption)


# ---------------------------------------------------------------------
# Análisis de humos
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", FLUE)
def test_every_flue_gas_example(name: str) -> None:
    at = _app()
    _flue(at, name)
    _no_problems(at)
    result = solve_flue_gas(FLUE_GAS_EXAMPLES[name])
    assert _metric(at, "λ") == pytest.approx(result.lam, rel=1e-3)
    assert _metric(at, "Aire teórico [%]") == pytest.approx(100 * result.lam, rel=1e-3)
    assert len(at.get("download_button")) == 2
    assert _charts(at) == 1  # el diagrama de combustión
    orsat = any("Balances del Orsat" in m.value for m in at.markdown)
    assert orsat == (result.orsat is not None)
    procedure = next(e for e in at.expander if "Procedimiento" in e.label)
    assert len(procedure.get("latex")) > 5


def test_cengel_15_4_orsat() -> None:
    at = _app()
    _flue(at, FLUE[0])
    assert _metric(at, "Aire teórico [%]") == pytest.approx(130.4, abs=0.1)
    assert any("El balance de oxígeno cierra" in i.value for i in at.info)
    assert any("N₂ por diferencia" in c.value for c in at.caption)


def test_o2_method_and_measurement_errors() -> None:
    at = _app()
    key = _flue(at, FLUE[1])
    lam = _metric(at, "λ")
    at.number_input(key=f"{key}_o2").set_value(6.0).run()
    _no_problems(at)
    assert _metric(at, "λ") > lam
    at.number_input(key=f"{key}_o2").set_value(20.8).run()
    assert not at.exception
    assert any("fuera de rango" in e.value for e in at.error)
    key = _flue(at, FLUE[0])
    at.number_input(key=f"{key}_co2").set_value(80.0).run()
    at.number_input(key=f"{key}_o2orsat").set_value(21.0).run()
    assert not at.exception
    assert any("no pueden sumar 100 %" in e.value for e in at.error)
