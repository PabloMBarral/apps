"""Tests de /Exergia (AppTest).

Fase 7. Los tres modos (física, química y por componente), todos los ejemplos,
los dos ambientes de referencia, los mensajes al alumno, los sistemas de
unidades, la teoría, el procedimiento, la exportación y los datos del último
ciclo que se calculó en otra página.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.combustion.heating_value import HEATING_VALUE_EXAMPLES, hhv_channiwala_parikh
from core.cycles.combined import COMBINED_EXAMPLES, CombinedInputs
from core.cycles.combined_multi import (
    COMBINED_MULTI_EXAMPLES,
    from_combined_with_pinch,
    solve_combined_multi,
)
from core.cycles.gas_turbine import GAS_TURBINE_EXAMPLES, solve_gas_turbine
from core.cycles.rankine import RANKINE_EXAMPLES, solve_rankine
from core.cycles.refrigeration import REFRIGERATION_EXAMPLES, solve_refrigeration
from core.exergy import chemical as ch
from core.exergy import physical as ph
from core.exergy import plant as pl

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "11_Exergia.py")

M_PHYS = "Exergía física: un fluido, un calor o un cuerpo"
M_CHEM = "Exergía química (Szargut)"
M_PLANT = "Una planta, componente por componente (Grassmann)"
F_HEAT = "Un calor a temperatura T"
F_BODY = "Un cuerpo que se enfría o se calienta"
Q_SPECIES = "Una sustancia"
Q_MIX = "Una mezcla de gases"
Q_FUEL = "Un combustible por su análisis elemental"
C_RANKINE = "Ciclo de Rankine"
C_REFRIG = "Refrigeración o bomba de calor"
C_GT = "Turbina de gas (Brayton)"
C_CC = "Ciclo combinado"
CYCLES = (C_RANKINE, C_REFRIG, C_GT, C_CC)
SRC_LAST = "El último que calculaste en la página del ciclo"

PHYS_NAMES = list(ph.PHYSICAL_EXAMPLES)
FUEL_NAMES = [
    n for n, i in HEATING_VALUE_EXAMPLES.items() if i.ultimate is not None and i.fuel_type != "gas"
]
COMBINED_ALL = {
    **{f"Una presión — {n}": i for n, i in COMBINED_EXAMPLES.items()},
    **COMBINED_MULTI_EXAMPLES,
}
PLANT_EXAMPLES = {
    C_RANKINE: RANKINE_EXAMPLES,
    C_REFRIG: REFRIGERATION_EXAMPLES,
    C_GT: GAS_TURBINE_EXAMPLES,
    C_CC: COMBINED_ALL,
}
PLANT_CASES = [(c, n) for c in CYCLES for n in PLANT_EXAMPLES[c]]


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


def _metric_text(at: AppTest, prefix: str) -> str:
    return next(m.value for m in at.metric if m.label.startswith(prefix))


def _expanders(at: AppTest) -> list[str]:
    return [e.label for e in at.expander]


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f} %".replace(".", ",")


def _mode(at: AppTest, mode: str) -> AppTest:
    at.radio(key="ex_mode").set_value(mode).run()
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


# ---------------------------------------------------------------------
# Exergía física
# ---------------------------------------------------------------------


def _physical(name: str) -> tuple[ph.PhysicalExergy, ph.PhysicalExergy | None]:
    ex = ph.PHYSICAL_EXAMPLES[name]
    first = ph.physical_exergy(
        ex.fluid,
        ex.pair,
        ambient=ex.ambient,
        speed_m_per_s=ex.speed_m_per_s,
        height_m=ex.height_m,
        amount=ex.amount,
        mass_kg=ex.mass_kg,
        m_dot_kg_s=ex.m_dot_kg_s,
        **ex.values,
    )
    second = None
    if ex.second:
        pair, values = ex.second
        second = ph.physical_exergy(ex.fluid, pair, ambient=ex.ambient, **values)
    return first, second


def test_default_is_the_steam_of_cengel_10_8() -> None:
    """Cengel 10-8: ψ = 1162 kJ/kg a la entrada de la turbina y Δψ = −713 kJ/kg."""
    at = _app()
    assert at.radio(key="ex_mode").value == M_PHYS
    assert _metric(at, "ψ, exergía de flujo") == pytest.approx(1162.1, abs=0.1)
    assert _metric(at, "Δψ = ψ₂ − ψ₁") == pytest.approx(-713.0, abs=0.1)
    assert _expanders(at) == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    assert _charts(at) == 1  # el estado en el h–s
    assert len(at.get("download_button")) == 2
    assert any(c.value.startswith("📘") for c in at.caption)


@pytest.mark.parametrize("name", PHYS_NAMES)
def test_every_physical_example(name: str) -> None:
    at = _app()
    at.selectbox(key="xf_example").set_value(name).run()
    _no_problems(at)
    first, second = _physical(name)
    assert _metric(at, "ψ, exergía de flujo") == pytest.approx(first.psi_J_per_kg / 1e3, rel=1e-4)
    assert _metric(at, "φ, exergía de la masa") == pytest.approx(
        first.phi_J_per_kg / 1e3, rel=1e-4, abs=1e-3
    )
    if first.X_J is not None:
        assert _metric(at, "X de la masa") == pytest.approx(first.X_J / 1e3, rel=1e-4)
    if first.X_W is not None:
        assert _metric(at, "Ẋ del flujo") == pytest.approx(first.X_W / 1e3, rel=1e-4)
    labels = [m.label for m in at.metric]
    assert any(label.startswith("Δψ") for label in labels) == (second is not None)
    steps = at.expander[-1]
    assert steps.label == "🔬 Procedimiento"
    assert len(steps.get("latex")) > 5


def test_amount_and_kinetic_energy_change_the_totals() -> None:
    at = _app()
    at.radio(key="xf_0_amount").set_value("Un caudal (flujo)").run()
    _no_problems(at)
    psi = _metric(at, "ψ, exergía de flujo")
    assert _metric(at, "Ẋ del flujo") == pytest.approx(psi, rel=1e-4)  # 1 kg/s por defecto
    at.number_input(key="xf_0_V@Técnico").set_value(100.0).run()
    _no_problems(at)
    assert _metric(at, "ψ, exergía de flujo") == pytest.approx(psi, rel=1e-6)
    assert _metric(at, "Ẋ del flujo") == pytest.approx(psi + 5.0, rel=1e-4)  # V²/2 = 5 kJ/kg
    at.radio(key="xf_0_amount").set_value("Una masa (sistema cerrado)").run()
    _no_problems(at)
    assert _metric(at, "X de la masa") > 0.0


def test_another_fluid_starts_from_its_suggested_state() -> None:
    at = _app()
    at.selectbox(key="xf_0_fluid").set_value("R134a").run()
    _no_problems(at)
    assert _metric(at, "ψ, exergía de flujo") != pytest.approx(1162.1, abs=0.1)
    at.selectbox(key="xf_0_pair").set_value("PH").run()
    _no_problems(at)


def test_physical_messages_to_the_student() -> None:
    at = _app()
    at.number_input(key="xf_0_T0@Técnico").set_value(95.0).run()
    assert at.error and "ambiente" in at.error[0].value.lower()
    at = _app()
    at.radio(key="xf_what").set_value(F_HEAT).run()
    at.number_input(key="xh_0_T@Técnico").set_value(-300.0).run()
    assert at.error


@pytest.mark.parametrize("name", list(ph.HEAT_EXAMPLES))
def test_every_heat_example(name: str) -> None:
    at = _app()
    at.radio(key="xf_what").set_value(F_HEAT).run()
    at.selectbox(key="xh_example").set_value(name).run()
    _no_problems(at)
    ex = ph.HEAT_EXAMPLES[name]
    r = ph.heat_exergy(ex.Q_W, ex.T_K, ex.T0_K)
    assert _metric(at, "Ẋ_Q, exergía del calor") == pytest.approx(r.X_W / 1e3, rel=1e-4)
    assert _metric(at, "Factor de Carnot") == pytest.approx(r.carnot, rel=1e-3)
    assert _charts(at) == 1
    assert len(at.get("download_button")) == 2


@pytest.mark.parametrize("name", list(ph.FINITE_SOURCE_EXAMPLES))
def test_every_body_example(name: str) -> None:
    at = _app()
    at.radio(key="xf_what").set_value(F_BODY).run()
    at.selectbox(key="xs_example").set_value(name).run()
    _no_problems(at)
    ex = ph.FINITE_SOURCE_EXAMPLES[name]
    r = ph.finite_source_exergy(ex.m_kg, ex.c_J_per_kg_K, ex.T_K, ex.T0_K)
    assert _metric(at, "Φ, exergía del cuerpo") == pytest.approx(r.Phi_J / 1e3, rel=1e-4)
    assert _charts(at) == 1


def test_iron_block_of_cengel() -> None:
    """Cengel cap. 8: 500 kg de hierro a 200 °C con el ambiente a 27 °C → 8191 kJ."""
    at = _app()
    at.radio(key="xf_what").set_value(F_BODY).run()
    assert _metric(at, "Φ, exergía del cuerpo") == pytest.approx(8191.0, abs=1.0)


@pytest.mark.parametrize("system", ["SI", "Inglés"])
def test_physical_in_other_unit_systems(system: str) -> None:
    at = _system(_app(), system)
    _no_problems(at)
    unit = {"SI": "J/kg", "Inglés": "Btu/lb"}[system]
    label = next(m.label for m in at.metric if m.label.startswith("ψ, exergía de flujo"))
    assert f"[{unit}]" in label
    factor = {"SI": 1.0, "Inglés": 1.0 / 2326.0}[system]
    first, _ = _physical(PHYS_NAMES[0])
    assert _metric(at, "ψ, exergía de flujo") == pytest.approx(
        first.psi_J_per_kg * factor, rel=1e-4
    )


# ---------------------------------------------------------------------
# Exergía química
# ---------------------------------------------------------------------


@pytest.mark.parametrize("model", list(ch.MODELS))
@pytest.mark.parametrize("key", ["CH4", "H2", "CO2", "H2O(l)", "C(s)", "n-C4H10", "C2H5OH(l)"])
def test_species(key: str, model: str) -> None:
    at = _mode(_app(), M_CHEM)
    at.selectbox(key="xq_model").set_value(model).run()
    at.selectbox(key="xq_species").set_value(key).run()
    _no_problems(at)
    r = ch.species_exergy(key, model)  # type: ignore[arg-type]
    assert _metric(at, "ē, exergía química") == pytest.approx(r.e_J_per_mol, rel=1e-5)
    labels = [m.label for m in at.metric]
    assert ("φ = e/PCI" in labels) == (r.ratio_lhv is not None)
    assert ("x⁰⁰ en el aire de referencia" in labels) == (r.x_reference is not None)
    assert any(c.value.startswith(f"Fuente: {r.source}") for c in at.caption)
    assert _charts(at) == 1  # φ de los combustibles
    assert "📋 La tabla completa (los dos modelos)" in _expanders(at)


def test_methane_ratio_and_method_deviation() -> None:
    """φ = 1,036 del metano (Kotas, 1985) y el método de Szargut a menos de 0,2 %."""
    at = _mode(_app(), M_CHEM)
    assert at.selectbox(key="xq_species").value == "CH4"
    assert _metric(at, "φ = e/PCI") == pytest.approx(1.036, abs=0.001)
    assert any("método de Szargut" in i.value for i in at.info)


def test_species_that_the_ahrendts_model_lacks() -> None:
    at = _mode(_app(), M_CHEM)
    at.selectbox(key="xq_model").set_value("ahrendts1980").run()
    _no_problems(at)  # la figura de φ se arma solo con lo que el modelo I tiene
    at.selectbox(key="xq_species").set_value("He").run()
    assert at.error and "elegí" in at.error[0].value
    at.selectbox(key="xq_model").set_value("szargut1988").run()
    _no_problems(at)


@pytest.mark.parametrize("name", list(ch.MIXTURE_EXAMPLES))
def test_every_mixture_example(name: str) -> None:
    at = _mode(_app(), M_CHEM)
    at.radio(key="xq_what").set_value(Q_MIX).run()
    at.selectbox(key="xq_mix_example").set_value(name).run()
    _no_problems(at)
    r = ch.mixture_chemical_exergy(ch.MIXTURE_EXAMPLES[name])
    assert _metric(at, "ē de la mezcla") == pytest.approx(r.e_J_per_mol, rel=1e-4)
    labels = [m.label for m in at.metric]
    assert any(label.startswith("Agua que condensa") for label in labels) == r.condenses
    assert len(at.get("download_button")) == 2


def test_mixture_messages_and_normalization() -> None:
    at = _mode(_app(), M_CHEM)
    at.radio(key="xq_what").set_value(Q_MIX).run()
    at.number_input(key="xq_mix_0_N2").set_value(0.5).run()
    _no_problems(at)
    assert any("se normalizan" in c.value for c in at.caption)
    at.multiselect(key="xq_mix_0_components").set_value([]).run()
    assert at.error


@pytest.mark.parametrize("name", FUEL_NAMES)
def test_every_fuel_example(name: str) -> None:
    at = _mode(_app(), M_CHEM)
    at.radio(key="xq_what").set_value(Q_FUEL).run()
    at.selectbox(key="xq_fuel_example").set_value(name).run()
    _no_problems(at)
    ex = HEATING_VALUE_EXAMPLES[name]
    assert ex.ultimate is not None
    kind = "líquido" if ex.fuel_type == "liquid" else "sólido"
    hhv = (
        ex.reference.hhv_d_J_per_kg
        if ex.reference is not None
        else hhv_channiwala_parikh(ex.ultimate)
    )
    r = ch.fuel_chemical_exergy(ex.ultimate, hhv, moisture=ex.moisture, kind=kind)
    assert _metric(at, "β = e/PCI de la materia seca") == pytest.approx(r.beta, abs=6e-4)
    assert _metric(at, "e química tal cual") == pytest.approx(r.e_J_per_kg / 1e3, rel=2e-4)
    assert any("Szargut y Styrylska" in i.value for i in at.info)


def test_fuel_estimate_and_kind() -> None:
    at = _mode(_app(), M_CHEM)
    at.radio(key="xq_what").set_value(Q_FUEL).run()
    e_measured = _metric(at, "e química tal cual")
    at.checkbox(key="xq_fuel_0_estimate").check().run()
    _no_problems(at)
    assert any(c.value.startswith("PCS seco estimado") for c in at.caption)
    assert _metric(at, "e química tal cual") != pytest.approx(e_measured, rel=1e-6)
    at.radio(key="xq_fuel_0_kind").set_value("Líquido").run()
    _no_problems(at)
    at.number_input(key="xq_fuel_0_O").set_value(80.0).run()
    assert at.error  # la suma pasa del 100 %


# ---------------------------------------------------------------------
# Planta, componente por componente
# ---------------------------------------------------------------------


@cache
def _expected_plant(cycle: str, name: str) -> tuple[pl.PlantExergy, float]:
    inputs = PLANT_EXAMPLES[cycle][name]
    if cycle == C_RANKINE:
        result = solve_rankine(inputs)
        return pl.rankine_exergy(result), result.eta_th
    if cycle == C_REFRIG:
        result = solve_refrigeration(inputs)
        cop = result.COP_B if result.inputs.heat_pump else result.COP_R
        return pl.refrigeration_exergy(result), cop
    if cycle == C_GT:
        result = solve_gas_turbine(inputs)
        return pl.gas_turbine_plant_exergy(result, "szargut"), result.eta_th
    if isinstance(inputs, CombinedInputs):
        inputs = from_combined_with_pinch(inputs)
    result = solve_combined_multi(inputs)
    return pl.combined_plant_exergy(result, "szargut"), result.eta_th


def _plant(at: AppTest, cycle: str, name: str | None = None) -> AppTest:
    _mode(at, M_PLANT)
    at.radio(key="xp_cycle").set_value(cycle).run()
    if name is not None:
        at.selectbox(key=f"xp_{CYCLES.index(cycle)}_example").set_value(name).run()
    return at


@pytest.mark.parametrize(("cycle", "name"), PLANT_CASES)
def test_every_plant_example(cycle: str, name: str) -> None:
    at = _plant(_app(), cycle, name)
    _no_problems(at)
    plant, context = _expected_plant(cycle, name)
    assert _metric_text(at, "Rendimiento exergético η_II") == _pct(plant.efficiency)
    assert _metric(at, "Exergía que entra") == pytest.approx(plant.fuel_W / 1e3, rel=1e-4)
    assert _metric(at, "Exergía destruida") == pytest.approx(
        plant.destroyed_W / 1e3, rel=1e-4, abs=1e-6
    )
    if cycle == C_REFRIG:
        assert _metric(at, "COP") == pytest.approx(context, rel=1e-3)
    else:
        assert _metric_text(at, "η térmico") == _pct(context)
    assert _charts(at) == 2  # Grassmann y ε por componente
    assert len(at.get("download_button")) == 2
    assert "🔬 Procedimiento" in _expanders(at)
    assert "🌊 Exergía de cada corriente" in _expanders(at)
    table = at.dataframe[0].value
    assert list(table["Componente"]) == [c.name for c in plant.components]


def test_rankine_of_cengel_10_8() -> None:
    """Cengel 10-8 (el 10-1 con la fuente a 1600 K y el ambiente a 290 K)."""
    name = next(n for n in RANKINE_EXAMPLES if n.startswith("Cengel 10-1"))
    at = _plant(_app(), C_RANKINE, name)
    table = at.dataframe[0].value
    destroyed = dict(zip(table["Componente"], table["Ẋ_D [kW]"], strict=True))
    caldera = next(k for k in destroyed if k.lower().startswith("caldera"))
    condensador = next(k for k in destroyed if k.lower().startswith("condensador"))
    # por kg de vapor: 1110 kJ/kg en la caldera y 414 kJ/kg en el condensador (Cengel 10-8)
    m = solve_rankine(RANKINE_EXAMPLES[name]).m_dot_kg_s
    assert float(destroyed[caldera]) / m == pytest.approx(1110.0, rel=2e-3)
    assert float(destroyed[condensador]) / m == pytest.approx(414.0, rel=3e-3)


def test_fuel_model_pci_gives_the_thermal_efficiency() -> None:
    """Con e_comb ≈ PCI (vademecum §16.13) el η_II de la turbina de gas es su η térmico."""
    names = list(GAS_TURBINE_EXAMPLES)
    name = next(n for n in names if "metano" in n)
    at = _plant(_app(), C_GT, name)
    key = f"xp_2_{names.index(name)}_model"
    szargut = _metric(at, "Rendimiento exergético η_II")
    at.radio(key=key).set_value("≈ PCI (vademecum §16.13)").run()
    _no_problems(at)
    assert _metric_text(at, "Rendimiento exergético η_II") == _metric_text(at, "η térmico")
    # la exergía química del metano es 4 % mayor que su PCI: η_II un poco menor
    assert szargut < _metric(at, "Rendimiento exergético η_II")


def test_air_standard_has_no_fuel_model() -> None:
    at = _plant(_app(), C_GT)  # Cengel 9-5, aire estándar
    assert not any(r.key == "xp_2_0_model" for r in at.radio)
    assert any(c.value.startswith("Aire estándar") for c in at.caption)


def test_plant_messages_to_the_student() -> None:
    at = _plant(_app(), C_RANKINE)
    at.number_input(key="xp_0_0_TH@Técnico").set_value(300.0).run()  # vapor a 350 °C
    assert at.error and "T_H" in at.error[0].value
    at = _plant(_app(), C_RANKINE)
    at.number_input(key="xp_0_0_TL@Técnico").set_value(5.0).run()  # ambiente a 16,85 °C
    assert at.error and "T_L" in at.error[0].value
    at = _plant(_app(), C_REFRIG)
    at.number_input(key="xp_1_0_TC@Técnico").set_value(-40.0).run()
    assert at.error and "evaporador" in at.error[0].value


def test_rankine_sink_above_the_ambient_is_a_loss() -> None:
    at = _plant(_app(), C_RANKINE)
    _no_problems(at)
    assert _metric(at, "Exergía perdida") == pytest.approx(0.0, abs=1e-9)
    at.number_input(key="xp_0_0_TL@Técnico").set_value(30.0).run()
    _no_problems(at)
    assert _metric(at, "Exergía perdida") > 0.0


@pytest.mark.parametrize(
    ("cycle", "session_key", "inputs"),
    [
        (C_RANKINE, "rk_inputs", list(RANKINE_EXAMPLES.values())[6]),
        (C_REFRIG, "rf_inputs", list(REFRIGERATION_EXAMPLES.values())[3]),
        (C_GT, "bt_inputs", list(GAS_TURBINE_EXAMPLES.values())[2]),
        (C_CC, "cc_inputs", next(i for i in COMBINED_EXAMPLES.values() if i.T_stack_K)),
        (C_CC, "cc_inputs", next(iter(COMBINED_MULTI_EXAMPLES.values()))),
    ],
)
def test_the_last_cycle_of_its_page(cycle: str, session_key: str, inputs: object) -> None:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.session_state[session_key] = inputs
    at.run()
    _plant(at, cycle)
    _no_problems(at)
    cid = CYCLES.index(cycle)
    assert at.radio(key=f"xp_{cid}_source").value == SRC_LAST
    assert any("Con los datos que calculaste" in c.value for c in at.caption)
    name = next(n for n, i in PLANT_EXAMPLES[cycle].items() if i == inputs)
    plant, _ = _expected_plant(cycle, name)
    assert _metric_text(at, "Rendimiento exergético η_II") == _pct(plant.efficiency)
    at.radio(key=f"xp_{cid}_source").set_value("Un ejemplo").run()
    _no_problems(at)


def test_without_a_last_cycle_there_is_no_source_choice() -> None:
    at = _plant(_app(), C_RANKINE)
    assert not any(r.key == "xp_0_source" for r in at.radio)


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_plant_in_every_unit_system(system: str) -> None:
    at = _system(_plant(_app(), C_CC), system)
    _no_problems(at)
    unit = {"SI": "W", "Técnico": "kW", "Inglés": "Btu/s"}[system]
    label = next(m.label for m in at.metric if m.label.startswith("Exergía que entra"))
    assert label.endswith(f"[{unit}]")


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def test_theory_cites_the_vademecum_and_the_sources() -> None:
    at = _app()
    theory = at.expander[0]
    assert theory.label == "📖 Fórmulas teóricas"
    text = " ".join(m.value for m in theory.markdown)
    assert "§11" in text and "§16.13" in text
    for source in ("Szargut", "Ahrendts", "Kotas", "Bejan, Tsatsaronis y Moran", "Grassmann"):
        assert source in text
    assert len(theory.get("latex")) >= 12
