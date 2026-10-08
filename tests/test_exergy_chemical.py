"""Tests de core.exergy.chemical.

Szargut et al. (1988), Ahrendts (1980) y Szargut y Styrylska (1964).
"""

from __future__ import annotations

import math
import statistics

import pytest

from core.combustion.heating_value import DryUltimate, argonne_coals, ghugare_dataset
from core.combustion.thermo import R_U
from core.exergy.chemical import (
    MODELS,
    chemical_species,
    fuel_chemical_exergy,
    fuel_exergy_to_dict,
    fuel_ratio_table,
    mixture_chemical_exergy,
    mixture_exergy_to_dict,
    reference_fraction,
    species_exergy,
    species_exergy_to_dict,
    standard_chemical_exergy,
    szargut_method,
)

# Valores verificados contra la tabla A-26 de Moran y Shapiro (kJ/kmol).
MODEL_II = {
    "C(s)": 410_260.0,
    "S(s)": 609_600.0,
    "CH4": 831_650.0,
    "CO2": 19_870.0,
    "H2O": 9_500.0,
    "H2O(l)": 900.0,
    "N2": 720.0,
    "O2": 3_970.0,
    "CO": 275_100.0,
    "C2H6": 1_495_840.0,
    "C8H18(l)": 5_413_100.0,
    "NO": 88_900.0,
    "C2H2": 1_265_800.0,
}
MODEL_I = {
    "C(s)": 404_590.0,
    "S(s)": 598_160.0,
    "CH4": 824_348.0,
    "CO2": 14_176.0,
    "H2O": 8_636.0,
    "H2O(l)": 45.0,
    "N2": 639.0,
}


@pytest.mark.parametrize(("key", "value"), MODEL_II.items())
def test_model_ii_table(key: str, value: float) -> None:
    assert standard_chemical_exergy(key, "szargut1988") == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize(("key", "value"), MODEL_I.items())
def test_model_i_table(key: str, value: float) -> None:
    assert standard_chemical_exergy(key, "ahrendts1980") == pytest.approx(value, rel=1e-9)


def test_table_loads_with_molar_masses() -> None:
    table = chemical_species()
    assert len(table) >= 40
    assert table["CH4"].M_kg_per_mol == pytest.approx(16.0425e-3, rel=1e-4)
    assert table["CH3OH(l)"].atoms == {"C": 1.0, "H": 4.0, "O": 1.0}
    assert table["He"].M_kg_per_mol == pytest.approx(4.0026e-3, rel=1e-4)
    assert set(MODELS) == {"szargut1988", "ahrendts1980"}


# ---------------------------------------------------------------------
# Método de Szargut
# ---------------------------------------------------------------------

_COMPOUNDS = [
    k
    for k, sp in chemical_species().items()
    if sp.nasa_key
    and sp.tabulated_J_per_mol("szargut1988") is not None
    and k not in ("N2", "O2", "H2", "Ar", "NO", "H2O(l)")
]


@pytest.mark.parametrize("key", _COMPOUNDS)
def test_szargut_method_reproduces_the_table(key: str) -> None:
    """Δg_f (polinomios NASA) + Σν·e de los elementos contra la tabla de 1988: ±0,2 %."""
    m = szargut_method(key)
    assert m.e_J_per_mol == pytest.approx(standard_chemical_exergy(key), rel=2e-3)


def test_szargut_method_details_for_methane() -> None:
    m = szargut_method("CH4")
    assert m.dg_f_J_per_mol == pytest.approx(-50.5e3, abs=0.2e3)
    assert m.elements_J_per_mol == pytest.approx(410_260.0 + 2.0 * 236_100.0)
    assert m.hf_J_per_mol - m.T0_K * m.ds_f_J_per_mol_K == pytest.approx(m.dg_f_J_per_mol)


def test_no_and_liquid_water_deviate_more() -> None:
    """NO: NASA (2002) actualizó su h_f; el método da 1,2 % más que la tabla."""
    assert szargut_method("NO").e_J_per_mol / standard_chemical_exergy("NO") - 1 == pytest.approx(
        0.0116, abs=0.002
    )


def test_untabulated_species_use_the_method() -> None:
    iso = species_exergy("i-C4H10")
    assert iso.source == "método de Szargut"
    assert iso.e_J_per_mol == pytest.approx(standard_chemical_exergy("n-C4H10"), rel=3e-3)
    jet = species_exergy("C12H23(l)")
    assert jet.source == "método de Szargut" and jet.ratio_lhv == pytest.approx(1.065, abs=0.01)


def test_reference_elements_have_no_method() -> None:
    for key in ("H2", "O2", "N2", "C(s)", "S(s)"):
        assert species_exergy(key).method is None


def test_method_fails_without_nasa_data() -> None:
    with pytest.raises(ValueError, match="polinomios NASA"):
        szargut_method("He")
    with pytest.raises(ValueError, match="no está en la tabla"):
        species_exergy("unobtainium")


# ---------------------------------------------------------------------
# Gases del aire y combustibles
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "p_kPa"),
    [("N2", 75.78), ("O2", 20.39), ("H2O", 2.2), ("CO2", 0.0335), ("Ar", 0.906)],
)
def test_reference_atmosphere_partial_pressures(key: str, p_kPa: float) -> None:
    """Szargut et al. (1988): e = R̄T₀·ln(p₀/p⁰⁰) con la atmósfera de referencia."""
    x = reference_fraction(key)
    assert x is not None
    assert x * 101.325 == pytest.approx(p_kPa, rel=0.012)
    assert reference_fraction("CH4") is None


def test_fuel_ratios() -> None:
    """φ = e/PCI: ≈ 1,04 en el metano, < 1 en el H₂ y el CO (Kotas, 1985)."""
    ch4 = species_exergy("CH4")
    assert ch4.ratio_lhv == pytest.approx(1.036, abs=0.002)
    assert ch4.ratio_hhv == pytest.approx(0.934, abs=0.002)
    assert species_exergy("H2").ratio_lhv == pytest.approx(0.976, abs=0.002)
    assert species_exergy("CO").ratio_lhv == pytest.approx(0.972, abs=0.002)
    assert species_exergy("C(s)").ratio_lhv == pytest.approx(1.0426, abs=0.001)
    rows = fuel_ratio_table()
    ratios = [r.ratio_lhv for r in rows]
    assert ratios == sorted(ratios)
    assert rows[0].species.key in ("CO", "H2")
    assert species_exergy("CO2").ratio_lhv is None


# ---------------------------------------------------------------------
# Mezclas
# ---------------------------------------------------------------------


def test_dry_air_mixture_by_hand() -> None:
    y = {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004}
    m = mixture_chemical_exergy(y)
    by_hand = sum(x * standard_chemical_exergy(k) for k, x in y.items()) + R_U * 298.15 * sum(
        x * math.log(x) for x in y.values()
    )
    assert m.e_J_per_mol == pytest.approx(by_hand, rel=1e-12)
    assert m.e_J_per_mol == pytest.approx(104.3, abs=0.5)
    assert not m.condenses


def test_reference_atmosphere_has_almost_no_exergy() -> None:
    """El aire de referencia (sin los gases nobles menores) solo tiene la exergía de
    normalizar: sus presiones parciales suman 99,3 kPa y no 101,325, así que e = −R̄T₀·ln Σx⁰⁰."""
    fractions = {k: reference_fraction(k) or 0.0 for k in ("N2", "O2", "H2O", "Ar", "CO2")}
    total = sum(fractions.values())
    m = mixture_chemical_exergy(fractions)
    assert total == pytest.approx(99.31 / 101.325, rel=2e-3)
    assert m.e_J_per_mol == pytest.approx(-R_U * 298.15 * math.log(total), rel=1e-9)


def test_flue_gas_condensation() -> None:
    """Humos con 10 % de agua: a 25 °C y 1 atm el gas queda saturado y el resto condensa."""
    y = {"N2": 0.72, "O2": 0.12, "CO2": 0.05, "H2O": 0.10, "Ar": 0.01}
    wet = mixture_chemical_exergy(y)
    dry = mixture_chemical_exergy(y, condense=False)
    assert wet.condenses and not dry.condenses
    row = next(r for r in wet.rows if r.key == "H2O")
    assert row.x_gas == pytest.approx(wet.x_sat)
    assert wet.x_sat == pytest.approx(3169.9 / 101_325.0, rel=1e-3)
    # balance de moles de agua: lo que queda en el gas + lo que condensa = lo que había
    assert wet.n_gas * wet.x_sat + wet.n_liquid == pytest.approx(0.10, rel=1e-12)
    assert wet.e_J_per_mol != pytest.approx(dry.e_J_per_mol, rel=1e-3)


def test_mixture_validation() -> None:
    with pytest.raises(ValueError, match="no son gases"):
        mixture_chemical_exergy({"C8H18(l)": 1.0})
    with pytest.raises(ValueError, match="negativas"):
        mixture_chemical_exergy({"N2": -1.0, "O2": 2.0})
    with pytest.raises(ValueError, match="vacía"):
        mixture_chemical_exergy({"N2": 0.0})


# ---------------------------------------------------------------------
# Szargut y Styrylska
# ---------------------------------------------------------------------


def _pure(key: str) -> tuple[DryUltimate, float]:
    """El análisis elemental y el PCS (J/kg) de una sustancia pura."""
    sp = chemical_species()[key]
    a = sp.atoms
    M = sp.M_kg_per_mol
    from core.combustion.thermo import atomic_mass

    frac = {el: n * atomic_mass(el) / M for el, n in a.items()}
    u = DryUltimate(C=frac.get("C", 0.0), H=frac.get("H", 0.0), O=frac.get("O", 0.0))
    se = species_exergy(key)
    assert se.hhv_J_per_mol is not None
    return u, se.hhv_J_per_mol / M


@pytest.mark.parametrize(
    ("key", "deviation"),
    [("C8H18(l)", 0.006), ("C2H5OH(l)", 0.012), ("CH3OH(l)", 0.027)],
)
def test_liquid_form_against_pure_liquids(key: str, deviation: float) -> None:
    """La forma de los líquidos contra las exergías exactas de la tabla."""
    u, hhv = _pure(key)
    fe = fuel_chemical_exergy(u, hhv, kind="líquido")
    exact = species_exergy(key).e_J_per_kg
    assert fe.branch == "líquido"
    assert fe.e_J_per_kg / exact - 1.0 == pytest.approx(deviation, abs=0.003)


def test_coal_form_for_graphite() -> None:
    """Con h/c = o/c = n/c = 0, β = 1,0437: el grafito da 410,7 kJ/mol (tabla: 410,26)."""
    hhv_C = 393_510.0 / 12.0107e-3  # −h_f del CO₂ por kg de C
    fe = fuel_chemical_exergy(DryUltimate(C=1.0, H=0.0), hhv_C)
    assert fe.branch == "carbón"
    assert fe.beta == pytest.approx(1.0437)
    assert fe.e_J_per_kg * 12.0107e-3 == pytest.approx(410_260.0, rel=2e-3)


def test_wood_form_continuity_and_biomass_average() -> None:
    """Con o/c → 0 la forma de la madera da casi la de los carbones; en biomasa, e/PCS ≈ 1,05."""
    from core.exergy.chemical import _beta

    assert _beta("madera", 0.08, 0.0, 0.01, 0.0) == pytest.approx(
        _beta("carbón", 0.08, 0.0, 0.01, 0.0), abs=2e-4
    )
    ratios = []
    for smp in ghugare_dataset():
        u = smp.ultimate
        # algunas muestras suman algo más de 100 % (las cenizas negativas quedan en 0)
        if u.C <= 0.0 or u.O / u.C > 2.67 or abs(sum(u.as_dict().values()) - 1.0) > 1e-3:
            continue
        fe = fuel_chemical_exergy(u, smp.hhv_d)
        ratios.append(fe.ratio_hhv)
    assert len(ratios) >= 500
    assert statistics.mean(ratios) == pytest.approx(1.054, abs=0.005)
    assert statistics.median(ratios) == pytest.approx(1.05, abs=0.01)


def test_moisture_and_sulfur_terms() -> None:
    coal = next(c for c in argonne_coals() if c.ultimate is not None and c.ultimate.S > 0.02)
    dry = fuel_chemical_exergy(coal.ultimate, coal.hhv_d)
    wet = fuel_chemical_exergy(coal.ultimate, coal.hhv_d, moisture=0.3)
    # la materia orgánica escala con (1 − W) y la humedad suma su exergía de agua líquida
    assert wet.e_organic_J_per_kg == pytest.approx(0.7 * dry.e_organic_J_per_kg, rel=1e-9)
    assert wet.e_water_J_per_kg == pytest.approx(0.3 * 900.0 / 18.0153e-3, rel=1e-3)
    assert wet.e_sulfur_J_per_kg == pytest.approx(0.7 * dry.e_sulfur_J_per_kg, rel=1e-9)
    # azufre: (e_S − PCI_S)·S, con e_S = 609,6 kJ/mol y PCI_S = 9,26 MJ/kg
    assert dry.lhv_S_J_per_kg == pytest.approx(9.256e6, rel=1e-3)
    assert dry.e_sulfur_J_per_kg == pytest.approx(
        (609_600.0 / 32.065e-3 - dry.lhv_S_J_per_kg) * coal.ultimate.S, rel=1e-9
    )
    assert 1.05 < dry.ratio_lhv < 1.15  # type: ignore[operator]


def test_szargut_styrylska_validation() -> None:
    with pytest.raises(ValueError, match="2,67"):
        fuel_chemical_exergy(DryUltimate(C=0.20, H=0.02, O=0.70, A=0.08), 5.0e6)
    with pytest.raises(ValueError, match="suma"):
        fuel_chemical_exergy(DryUltimate(C=0.5, H=0.06), 20.0e6)
    with pytest.raises(ValueError, match="humedad"):
        fuel_chemical_exergy(DryUltimate(C=0.9, H=0.1), 40.0e6, moisture=1.2)
    with pytest.raises(ValueError, match="PCS"):
        fuel_chemical_exergy(DryUltimate(C=0.9, H=0.1), -1.0)
    near = fuel_chemical_exergy(DryUltimate(C=0.25, H=0.03, O=0.62, A=0.10), 8.0e6)
    assert near.branch == "madera" and any("límite" in w for w in near.warnings)
    wet = fuel_chemical_exergy(DryUltimate(C=0.5, H=0.06, O=0.43, A=0.01), 20.0e6, moisture=0.9)
    assert any("PCI" in w for w in wet.warnings)


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_exports(system: str) -> None:
    d = species_exergy_to_dict(species_exergy("CH4"), system)  # type: ignore[arg-type]
    assert d["fuente"] == "tabla" and "metodo_de_Szargut" in d and "e_sobre_PCI" in d
    m = mixture_exergy_to_dict(
        mixture_chemical_exergy({"CH4": 0.9, "C2H6": 0.05, "N2": 0.05}),
        system,  # type: ignore[arg-type]
    )
    assert "metano" in m["componentes"]
    coal = argonne_coals()[0]
    f = fuel_exergy_to_dict(fuel_chemical_exergy(coal.ultimate, coal.hhv_d), system)  # type: ignore[arg-type]
    assert f["forma_de_la_correlacion"] == "carbón" and f["beta"] > 1.0
