"""Tests de la combustión — Fase 5 (vademecum §16; Cengel cap. 15 y §16-2).

Contra los ejemplos 15-1 a 15-11 de Cengel, el equilibrio de Cantera con los
mismos polinomios NASA-9, ISO 6976 y los balances de elementos, energía y
entropía.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import replace

import pytest

from core.combustion.combustion import (
    COMBUSTION_EXAMPLE_NOTES,
    COMBUSTION_EXAMPLES,
    FLUE_GAS_EXAMPLE_NOTES,
    FLUE_GAS_EXAMPLES,
    CombustionInputs,
    FlueGasInputs,
    air_temperature_sweep,
    combustion_notes,
    combustion_to_dict,
    default_air_temperature_values,
    default_lambda_values,
    default_products_temperature_values,
    flue_gas_notes,
    flue_gas_to_dict,
    lambda_sweep,
    products_temperature_sweep,
    solve_combustion,
    solve_flue_gas,
)
from core.combustion.equilibrium import adiabatic_equilibrium, equilibrium_composition
from core.combustion.fuels import FUELS, Fuel, UltimateAnalysis
from core.combustion.stoichiometry import (
    AirSpec,
    Oxidizer,
    acid_dew_point,
    dry_gas_curves,
    lambda_from_o2,
    orsat_analysis,
    solve_stoichiometry,
)
from core.combustion.thermo import R_U, T_REF_K, species
from core.ideal_gas import AIR_TECHNICAL, combustion_products

C = 273.15
ATM = 101_325.0
OCTANE = FUELS["n-Octano (líquido)"]
METHANE = FUELS["Metano"]


def _solve(name: str):  # type: ignore[no-untyped-def]
    return solve_combustion(COMBUSTION_EXAMPLES[name])


def _atoms(moles: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, n in moles.items():
        for el, a in species(k).formula.items():
            out[el] = out.get(el, 0.0) + n * a
    return out


# ---------------------------------------------------------------------
# Cengel cap. 15
# ---------------------------------------------------------------------


def test_cengel_15_1_octane_with_20_kmol_of_oxygen() -> None:
    r = _solve("Cengel 15-1: octano con 20 kmol de O₂")
    p = r.stoich.products
    assert p["CO2"] == pytest.approx(8.0)
    assert p["H2O"] == pytest.approx(9.0)
    assert p["O2"] == pytest.approx(7.5)
    assert p["N2"] == pytest.approx(20 * 0.79 / 0.21)
    # Cengel: 24,2 kg/kg con M_aire = 29; con el aire técnico (28,85) da 24,05
    assert r.stoich.AC == pytest.approx(24.05, abs=0.01)
    assert r.stoich.AC * 28.97 / r.stoich.M_dry_air / 1000 == pytest.approx(24.2, abs=0.1)


def test_cengel_15_2_ethane_dew_point() -> None:
    r = _solve("Cengel 15-2: etano con 20 % de exceso de aire")
    assert r.stoich.AC == pytest.approx(19.3, abs=0.15)
    assert r.stoich.dew_point_K - C == pytest.approx(52.3, abs=0.3)


def test_cengel_15_3_natural_gas_with_moist_air() -> None:
    r = _solve("Cengel 15-3: gas natural con aire húmedo")
    assert r.stoich.o2_theoretical == pytest.approx(1.465)
    assert r.stoich.dew_point_K - C == pytest.approx(60.9, abs=0.2)


def test_cengel_15_4_orsat() -> None:
    f = solve_flue_gas(FLUE_GAS_EXAMPLES["Cengel 15-4: Orsat del octano"])
    assert f.orsat is not None
    assert f.orsat.fuel_per_100 == pytest.approx(10.90 / 8)
    assert f.lam == pytest.approx(1.31, abs=0.01)  # Cengel: 131 % de aire teórico
    assert f.condensed == pytest.approx(6.59, abs=0.02)  # kmol por kmol de octano
    assert abs(f.orsat.oxygen_residual_relative) < 0.002


def test_cengel_15_6_liquid_propane_with_co() -> None:
    r = _solve("Cengel 15-6: propano líquido con CO en los humos")
    assert r.stoich.products["CO"] == pytest.approx(0.3)
    assert r.stoich.products["CO2"] == pytest.approx(2.7)
    m_air_kg_min = r.stoich.AC * 0.05
    assert m_air_kg_min * 29 / r.stoich.M_dry_air / 1000 == pytest.approx(1.18, abs=0.01)
    assert r.heat is not None
    assert r.heat.Q_out == pytest.approx(363_880, rel=0.01)  # kJ/kmol de propano
    # Cengel: 413 kJ/min con M = 44 kg/kmol; con M = 44,10 da 411
    assert r.total(r.heat.Q_out) * 60 / 1e3 == pytest.approx(413, abs=3)  # kJ/min


def test_cengel_15_7_bomb() -> None:
    r = _solve("Cengel 15-7: metano y oxígeno en un recipiente rígido")
    assert r.heat is not None
    assert r.heat.Q_out / 2.326 == pytest.approx(308_730, rel=0.003)  # Btu por lbmol
    assert r.heat.p_Pa / ATM == pytest.approx(3.35, abs=0.01)


def test_cengel_15_8_adiabatic_flame_of_octane() -> None:
    r = _solve("Cengel 15-8: llama adiabática del octano")
    assert r.flame.T_K == pytest.approx(2395, abs=5)
    r400 = solve_combustion(CombustionInputs(OCTANE, air=AirSpec("theoretical", 4.0)))
    assert r400.flame.T_K == pytest.approx(962, abs=2)


def test_cengel_15_8_c_rich_closes_the_energy_balance_at_2284_K() -> None:
    """Con 90 % de aire (5,5 CO₂ + 2,5 CO) el balance cierra en 2284 K, no en 2236 K."""
    r = solve_combustion(CombustionInputs(OCTANE, air=AirSpec("theoretical", 0.9)))
    p = r.stoich.products
    assert (p["CO2"], p["CO"], p["H2O"]) == pytest.approx((5.5, 2.5, 9.0))
    assert r.flame.T_K == pytest.approx(2284, abs=1)
    held = sum(n * species(k).delta_h(2236.0) for k, n in p.items())
    needed = sum(n * species(k).delta_h(r.flame.T_K) for k, n in p.items())
    assert held < 0.98 * needed


def test_cengel_15_10_and_15_11_methane_with_50_percent_excess() -> None:
    r = _solve("Cengel 15-10 y 15-11: metano con 50 % de exceso de aire")
    assert r.flame.T_K == pytest.approx(1789, abs=2)
    assert r.heat is not None and r.second_law is not None
    assert r.heat.Q_out == pytest.approx(871_400, rel=1e-3)
    assert r.heat.condensed == pytest.approx(2 - 0.4289, abs=1e-3)
    assert r.second_law.S_gen_heat == pytest.approx(2746, rel=1e-3)
    assert r.second_law.X_dest_heat == pytest.approx(818_000, rel=2e-3)


def test_heating_values_of_octane_match_cengel_a27() -> None:
    assert OCTANE.hhv_per_kg / 1e3 == pytest.approx(47_890, rel=5e-4)
    assert OCTANE.lhv_per_kg / 1e3 == pytest.approx(44_430, rel=5e-4)
    # metano: el mismo PCI que la turbina de gas (ISO 6976, 50 027 kJ/kg)
    assert METHANE.lhv_per_kg / 1e3 == pytest.approx(50_027, rel=1e-4)


# ---------------------------------------------------------------------
# Estequiometría
# ---------------------------------------------------------------------


def test_methane_products_match_the_gas_turbine_module() -> None:
    s = solve_stoichiometry(METHANE, Oxidizer(), AirSpec("lambda", 3.0))
    gas, n_air = combustion_products(AIR_TECHNICAL, (1, 4, 0, 0), 3.0)
    assert s.air_moles == pytest.approx(n_air)
    for k, y in gas.mole_fractions.items():
        assert s.wet_fractions.get(k, 0.0) == pytest.approx(y, abs=1e-12)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("lambda", 1.25),
        ("excess", 0.25),
        ("theoretical", 1.25),
        ("equivalence", 0.8),
    ],
)
def test_every_way_of_giving_the_air(kind: str, value: float) -> None:
    s = solve_stoichiometry(FUELS["Gas natural típico"], Oxidizer(), AirSpec(kind, value))  # type: ignore[arg-type]
    assert s.lam == pytest.approx(1.25)


def test_air_fuel_ratio_and_o2_give_back_lambda() -> None:
    fuel = FUELS["Fueloil"]
    ref = solve_stoichiometry(fuel, Oxidizer("dry"), AirSpec("lambda", 1.3))
    by_ac = solve_stoichiometry(fuel, Oxidizer("dry"), AirSpec("AC", ref.AC))
    by_o2 = solve_stoichiometry(fuel, Oxidizer("dry"), AirSpec("o2_dry", ref.dry_fractions["O2"]))
    assert by_ac.lam == pytest.approx(1.3, rel=1e-12)
    assert by_o2.lam == pytest.approx(1.3, rel=1e-10)


def test_o2_formula_for_complete_combustion() -> None:
    """λ = 1 + y·n_seco,s / (O₂,t·(1 − y/y_O₂,aire seco)) con combustión completa."""
    fuel = FUELS["Gas natural típico"]
    air = Oxidizer().composition(ATM)
    stoich = solve_stoichiometry(fuel, Oxidizer(), AirSpec("lambda", 1.0))
    y = 0.03
    expected = 1 + y * stoich.n_dry / (fuel.theoretical_oxygen() * (1 - y / 0.21))
    assert lambda_from_o2(fuel, air, y) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("name", list(FUELS))
def test_element_balance_and_mass_balance(name: str) -> None:
    fuel = FUELS[name]
    s = solve_stoichiometry(fuel, Oxidizer("dry", 300.0, 0.5), AirSpec("lambda", 1.2))
    atoms_in = dict(fuel.elements())
    atoms_in["H"] += 2 * fuel.moisture_mol
    atoms_in["O"] += fuel.moisture_mol
    for k, n in _atoms({k: s.air_moles * y for k, y in s.air.items()}).items():
        atoms_in[k] = atoms_in.get(k, 0.0) + n
    atoms_out = _atoms(s.products)
    for el, n in atoms_in.items():
        assert atoms_out.get(el, 0.0) == pytest.approx(n, rel=1e-12, abs=1e-12)
    # GC = AC + 1 − cenizas + humedad del aire (por kg de combustible)
    vapor = s.air_vapor_moles * species("H2O").M_kg_per_mol / fuel.mass_per_basis
    assert s.GC == pytest.approx(s.AC + 1 - fuel.ash_kg / fuel.mass_per_basis + vapor, rel=1e-9)


def test_rich_mixture_follows_cengel_and_stops_at_soot() -> None:
    s = solve_stoichiometry(METHANE, Oxidizer(), AirSpec("lambda", 0.8))
    assert s.rich
    assert s.products["CO"] + s.products["CO2"] == pytest.approx(1.0)
    assert "O2" not in s.products
    with pytest.raises(ValueError, match="hollín"):
        solve_stoichiometry(OCTANE, Oxidizer(), AirSpec("lambda", 0.6))
    h2 = solve_stoichiometry(FUELS["Hidrógeno"], Oxidizer(), AirSpec("lambda", 0.8))
    assert h2.products["H2"] == pytest.approx(0.2)


def test_co2_max_of_natural_gas() -> None:
    s = solve_stoichiometry(FUELS["Gas natural típico"], Oxidizer(), AirSpec("lambda", 1.3))
    assert s.co2_max == pytest.approx(0.1208, abs=5e-4)
    curves = dry_gas_curves(FUELS["Gas natural típico"], Oxidizer(), [0.8, 1.0, 1.5])
    assert curves["CO2"][1] == pytest.approx(s.co2_max)
    assert curves["O2"][0] == 0.0 and curves["CO"][0] > 0.0


def test_normal_volume_of_methane() -> None:
    """CH₄ con λ = 1: 10,52 kmol de humos por kmol, 22,414 m³/kmol."""
    s = solve_stoichiometry(METHANE, Oxidizer(), AirSpec("lambda", 1.0))
    assert s.n_gas == pytest.approx(1 + 2 + 2 * 0.79 / 0.21)
    assert s.normal_volume * METHANE.M_kg_per_mol == pytest.approx(s.n_gas * 0.0224140, rel=1e-4)


def test_acid_dew_point_verhoff_banchero() -> None:
    """10 % de H₂O y 10 ppm de SO₃ a 1 atm: ≈ 137 °C (Verhoff & Banchero, 1974)."""
    assert acid_dew_point(0.10 * ATM, 10e-6 * ATM) - C == pytest.approx(137.3, abs=0.2)
    assert acid_dew_point(0.10 * ATM, 1e-6 * ATM) < acid_dew_point(0.10 * ATM, 10e-6 * ATM)


def test_orsat_recovers_the_lambda_of_generated_gases() -> None:
    fuel = FUELS["Fueloil"]
    s = solve_stoichiometry(fuel, Oxidizer("dry"), AirSpec("lambda", 1.4), co_fraction=0.01)
    y = s.dry_fractions
    o = orsat_analysis(fuel, Oxidizer("dry"), y["CO2"] + y["SO2"], y["CO"], y["O2"])
    assert o.lam == pytest.approx(1.4, rel=1e-10)
    assert abs(o.oxygen_residual_relative) < 1e-10


def test_o2_and_co_measurement_recover_lambda_and_co() -> None:
    fuel = FUELS["Gas natural típico"]
    s = solve_stoichiometry(fuel, Oxidizer(), AirSpec("lambda", 1.18), co_fraction=0.002)
    y = s.dry_fractions
    f = solve_flue_gas(FlueGasInputs(fuel, o2_dry=y["O2"], co_dry=y["CO"]))
    assert f.lam == pytest.approx(1.18, rel=1e-9)
    assert f.co_fraction == pytest.approx(0.002, rel=1e-8)


# ---------------------------------------------------------------------
# Equilibrio químico
# ---------------------------------------------------------------------

# Cantera 3.2.0 con los mismos polinomios (scratchpad de la Fase 5): T de llama en K.
CANTERA = {
    "metano λ = 1": (CombustionInputs(METHANE), 2223.5792),
    "octano líquido λ = 1": (CombustionInputs(OCTANE), 2263.3788),
    "hidrógeno λ = 1": (CombustionInputs(FUELS["Hidrógeno"]), 2378.0878),
    "metano λ = 0,8 a 10 atm con aire a 700 K": (
        CombustionInputs(
            METHANE,
            oxidizer=Oxidizer("technical", 700.0),
            air=AirSpec("lambda", 0.8),
            p_Pa=10 * ATM,
        ),
        2338.9828,
    ),
    "metano λ = 1 a volumen constante": (CombustionInputs(METHANE, process="v"), 2584.5459),
    "gas natural con aire seco λ = 1,2": (
        CombustionInputs(
            FUELS["Gas natural típico"], oxidizer=Oxidizer("dry"), air=AirSpec("lambda", 1.2)
        ),
        2045.8171,
    ),
}


@pytest.mark.parametrize("case", list(CANTERA))
def test_dissociated_flame_matches_cantera(case: str) -> None:
    inputs, T_cantera = CANTERA[case]
    r = solve_combustion(inputs)
    assert r.flame_eq is not None
    assert r.flame_eq.T_K == pytest.approx(T_cantera, abs=0.01)


def test_methane_equilibrium_composition_matches_cantera() -> None:
    r = solve_combustion(CombustionInputs(METHANE))
    assert r.flame_eq is not None
    y = r.flame_eq.fractions
    assert y["NO"] == pytest.approx(1.851811e-3, rel=1e-4)
    assert y["CO"] == pytest.approx(8.911737e-3, rel=1e-4)


@pytest.mark.parametrize("name", list(COMBUSTION_EXAMPLES))
def test_equilibrium_conserves_atoms_and_energy_and_satisfies_kp(name: str) -> None:
    r = _solve(name)
    assert r.flame_eq is not None, r.eq_error
    atoms_in = _atoms(r.stoich.products)
    atoms_eq = _atoms(r.flame_eq.products)
    for el, n in atoms_in.items():
        assert atoms_eq[el] == pytest.approx(n, rel=1e-6)
    T = r.flame_eq.T_K
    H = sum(n * species(k).h(T) for k, n in r.flame_eq.products.items())
    if r.inputs.process == "p":
        assert H == pytest.approx(r.H_reactants, rel=1e-9, abs=1e-3)
    else:
        U = H - R_U * T * sum(r.flame_eq.products.values())
        assert U == pytest.approx(r.U_reactants, rel=1e-9, abs=1e-3)
    assert r.kp_checks
    for check in r.kp_checks:
        assert check.ln_Kp_composition == pytest.approx(check.ln_Kp_tables, abs=2e-4)


def test_equilibrium_at_fixed_temperature_without_dissociation_when_cold() -> None:
    b = {"C": 1.0, "H": 4.0, "O": 2 * 3.0, "N": 2 * 3.0 * 0.79 / 0.21}
    x = equilibrium_composition(b, 800.0, p_Pa=ATM)
    total = sum(x.values())
    assert x["CO"] / total < 1e-12 and x["CO2"] == pytest.approx(1.0, rel=1e-9)


def test_adiabatic_equilibrium_explains_out_of_range() -> None:
    b = {"C": 1.0, "H": 4.0, "O": 4.0}
    with pytest.raises(ValueError, match="no hay llama"):
        adiabatic_equilibrium(b, -5e6, p_Pa=ATM)


def test_dissociation_lowers_the_flame_and_peaks_slightly_rich() -> None:
    inputs = CombustionInputs(METHANE)
    sweep = lambda_sweep(inputs, [0.9, 0.95, 1.0, 1.05, 1.1])
    t_eq = sweep["T_ad_eq"]
    assert all(e < c for e, c in zip(t_eq, sweep["T_ad"], strict=True) if e is not None)
    assert max(range(5), key=lambda k: t_eq[k]) in (1, 2)  # máximo levemente rico


# ---------------------------------------------------------------------
# Primer y segundo principio
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(COMBUSTION_EXAMPLES))
def test_flame_closes_the_energy_balance(name: str) -> None:
    r = _solve(name)
    T = r.flame.T_K
    H = sum(n * species(k).h(T) for k, n in r.flame.products.items())
    target = r.H_reactants if r.inputs.process == "p" else r.U_reactants
    value = H if r.inputs.process == "p" else H - R_U * T * sum(r.flame.products.values())
    assert value == pytest.approx(target, rel=1e-10, abs=1e-6)


@pytest.mark.parametrize("name", [n for n, i in COMBUSTION_EXAMPLES.items() if i.T_products_K])
def test_energy_split_closes(name: str) -> None:
    r = _solve(name)
    assert r.heat is not None
    if r.heat.split is not None:
        assert r.heat.split.Q_out == pytest.approx(r.heat.Q_out, rel=1e-9)


def test_condensing_boiler_beats_100_percent_on_lhv() -> None:
    r = _solve("Caldera de condensación (humos a 45 °C)")
    assert r.heat is not None and r.heat.eta_lhv is not None
    assert r.heat.condensed > 0.0
    assert r.heat.eta_lhv > 1.0 > r.heat.eta_hhv  # type: ignore[operator]
    dry = _solve("Caldera de gas natural (3 % de O₂, humos a 150 °C)")
    assert dry.heat is not None and dry.heat.condensed == 0.0


def test_second_law_identities() -> None:
    r = _solve("Cengel 15-10 y 15-11: metano con 50 % de exceso de aire")
    sl = r.second_law
    assert sl is not None
    assert sl.S_gen_adiabatic > 0
    assert sl.X_dest_adiabatic == pytest.approx(sl.T0_K * sl.S_gen_adiabatic)
    assert sl.X_dest_adiabatic / sl.lhv == pytest.approx(0.36, abs=0.01)
    assert sl.X_heat == pytest.approx(0.0, abs=1e-6)  # el calor va al ambiente


def test_analysis_fuels_have_no_second_law() -> None:
    r = _solve("Fueloil en una caldera (rocío ácido)")
    assert r.second_law is None
    assert r.stoich.acid_dew_point_K(0.02) > r.stoich.dew_point_K  # type: ignore[operator]


# ---------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (CombustionInputs(METHANE, T_products_K=3000.0), "no habría calor"),
        (CombustionInputs(METHANE, air=AirSpec("lambda", 25.0)), "fuera de rango"),
        (CombustionInputs(METHANE, air=AirSpec("o2_dry", 0.25)), "O₂ de los humos"),
        (CombustionInputs(OCTANE, T_fuel_K=450.0), "solo está tabulado"),
        (
            CombustionInputs(METHANE, T_products_K=400.0, T_sink_K=2500.0),
            "entre −73 °C y la llama",
        ),
        (CombustionInputs(METHANE, co_fraction=1.2), "CO tiene que estar"),
        (CombustionInputs(METHANE, oxidizer=Oxidizer("technical", 360.0, 1.0)), "aire húmedo"),
        (
            CombustionInputs(METHANE, oxidizer=Oxidizer("technical", 283.15, 0.9, 303.15)),
            "punto de rocío",
        ),
        (
            CombustionInputs(METHANE, oxidizer=Oxidizer("technical", 473.15, 0.5, 400.0)),
            "aire ambiente",
        ),
        (CombustionInputs(METHANE, m_fuel_kg_s=-1.0), "caudal"),
        (
            CombustionInputs(
                FUELS["Carbón (Pensilvania)"],
                oxidizer=Oxidizer("oxygen"),
                air=AirSpec("lambda", 1.1),
            ),
            "superaría 6000 K",
        ),
    ],
)
def test_invalid_data_explain_why(inputs: CombustionInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_combustion(inputs)


def test_invalid_fuels_explain_why() -> None:
    with pytest.raises(ValueError, match="suma"):
        Fuel.from_analysis("x", UltimateAnalysis(C=0.5, H=0.1), 30e6)
    with pytest.raises(ValueError, match="nada que se queme"):
        Fuel.mixture("x", {"N2": 0.5, "CO2": 0.5})
    with pytest.raises(ValueError, match="no son de un combustible"):
        Fuel.mixture("x", {"C8H18(l)": 1.0})
    with pytest.raises(ValueError, match="PCI queda negativo"):
        Fuel.from_analysis("x", UltimateAnalysis(C=0.05, H=0.005, W=0.945), 2e6)


# ---------------------------------------------------------------------
# Ejemplos, notas, barridos y export
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(COMBUSTION_EXAMPLES))
def test_every_example_solves_and_has_a_note(name: str) -> None:
    r = _solve(name)
    assert name in COMBUSTION_EXAMPLE_NOTES
    assert combustion_notes(r)


@pytest.mark.parametrize("name", list(FLUE_GAS_EXAMPLES))
def test_every_flue_gas_example(name: str) -> None:
    f = solve_flue_gas(FLUE_GAS_EXAMPLES[name])
    assert name in FLUE_GAS_EXAMPLE_NOTES
    assert flue_gas_notes(f)
    assert 1.0 <= f.lam < 2.0


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("Caldera de condensación (humos a 45 °C)", "no es un error"),
        ("Fueloil en una caldera (rocío ácido)", "rocío ácido"),
        ("Cengel 15-7: metano y oxígeno en un recipiente rígido", "oxígeno puro"),
        ("Cengel 15-8: llama adiabática del octano", "disociación baja la llama"),
        ("Motor a gas: propano a volumen constante", "volumen constante"),
        ("Bagazo de caña en la caldera de un ingenio", "precalentado"),
    ],
)
def test_notes(name: str, fragment: str) -> None:
    notes = combustion_notes(_solve(name))
    assert any(fragment in n for n in notes), notes


def test_sweeps() -> None:
    inputs = COMBUSTION_EXAMPLES["Caldera de gas natural (3 % de O₂, humos a 150 °C)"]
    lam = lambda_sweep(inputs, default_lambda_values(inputs, 9), equilibrium=False)
    assert len(lam["lambda"]) == 9
    k1 = min(range(9), key=lambda k: abs(lam["lambda"][k] - 1.0))  # type: ignore[arg-type]
    assert max(lam["T_ad"]) == pytest.approx(lam["T_ad"][k1], rel=0.02)  # type: ignore[type-var]
    air = air_temperature_sweep(inputs, default_air_temperature_values(5), equilibrium=False)
    assert air["T_ad"] == sorted(air["T_ad"])  # type: ignore[type-var]
    temps = products_temperature_sweep(inputs, default_products_temperature_values(inputs, 12))
    eta = temps["eta_lhv"]
    assert eta == sorted(eta, reverse=True)  # type: ignore[type-var]
    assert temps["condensed_fraction"][0] > 0.0 == temps["condensed_fraction"][-1]  # type: ignore[operator]


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export_is_serializable(system: str) -> None:
    for inputs in COMBUSTION_EXAMPLES.values():
        data = combustion_to_dict(solve_combustion(inputs), system)  # type: ignore[arg-type]
        json.dumps(data)
    for inputs in FLUE_GAS_EXAMPLES.values():
        json.dumps(flue_gas_to_dict(solve_flue_gas(inputs), system))  # type: ignore[arg-type]


def test_results_pickle() -> None:
    r = _solve("Cengel 15-10 y 15-11: metano con 50 % de exceso de aire")
    back = pickle.loads(pickle.dumps(r))
    assert back.flame.T_K == r.flame.T_K
    f = solve_flue_gas(FLUE_GAS_EXAMPLES["Cengel 15-4: Orsat del octano"])
    assert pickle.loads(pickle.dumps(f)).lam == f.lam


def test_preheated_humid_air_keeps_its_vapor() -> None:
    """φ dada a la T del ambiente: precalentar no cambia el vapor (ω constante)."""
    ambient = Oxidizer("technical", 293.15, 0.6)
    hot = Oxidizer("technical", 473.15, 0.6, T_humidity_K=293.15)
    assert hot.T_phi_K == 293.15 and ambient.T_phi_K == 293.15
    assert hot.vapor_fraction(ATM) == ambient.vapor_fraction(ATM)
    cold = solve_combustion(CombustionInputs(METHANE, oxidizer=ambient))
    warm = solve_combustion(CombustionInputs(METHANE, oxidizer=hot))
    assert warm.stoich.products == cold.stoich.products
    assert warm.flame.T_K > cold.flame.T_K + 100.0
    # Sin T_humidity_K, φ = 60 % a 200 °C sería vapor a más de media atmósfera.
    with pytest.raises(ValueError, match="aire húmedo"):
        solve_combustion(CombustionInputs(METHANE, oxidizer=Oxidizer("technical", 473.15, 0.6)))


def test_air_preheating_sweep_with_humid_air_covers_every_temperature() -> None:
    inputs = COMBUSTION_EXAMPLES["Caldera de gas natural (3 % de O₂, humos a 150 °C)"]
    assert inputs.oxidizer.phi > 0.0
    values = default_air_temperature_values()
    sweep = air_temperature_sweep(inputs, values, equilibrium=False)
    assert sweep["T_air"] == values[1:]  # 0 °C queda bajo el rocío del aire a 20 °C y 60 %
    assert sweep["T_ad"] == sorted(sweep["T_ad"])  # type: ignore[type-var]


def test_humid_air_and_pressure_raise_the_dew_point() -> None:
    base = CombustionInputs(METHANE, air=AirSpec("lambda", 1.1))
    dry = solve_combustion(base).stoich.dew_point_K
    humid = solve_combustion(replace(base, oxidizer=Oxidizer("technical", 303.15, 0.9)))
    high_p = solve_combustion(replace(base, p_Pa=10 * ATM))
    assert humid.stoich.dew_point_K > dry  # type: ignore[operator]
    assert high_p.stoich.dew_point_K > dry  # type: ignore[operator]
    assert math.isfinite(T_REF_K)
