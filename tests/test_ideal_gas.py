"""Tests del modelo de mezclas de gases ideales (aire y gases de combustión) — Fase 3.4.

Referencias: las tablas de gas ideal de Cengel & Boles (A-18 a A-23, Δh̄ entre
298 y 1000 K), las entropías absolutas de NIST-JANAF (Chase, 1998) y el
ejemplo 9-5 de Cengel (compresión isoentrópica del aire con r = 8).
"""

from __future__ import annotations

import math

import pytest
from scipy.integrate import quad

import core.cycles.hrsg as hrsg
from core import ideal_gas
from core.ideal_gas import (
    AIR_DRY,
    AIR_TECHNICAL,
    GAS_SPECIES,
    FlueGas,
    combustion_products,
    exhaust_composition,
    molar_mass,
)

# Δh̄ de 298 a 1000 K (kJ/kmol): Cengel A-18 (N₂), A-19 (O₂), A-20 (CO₂), A-23 (H₂O);
# el argón es monoatómico: c̄_p = 5/2·R_u.
CENGEL_DH = {"N2": 21_460.0, "O2": 22_707.0, "CO2": 33_405.0, "H2O": 25_978.0, "Ar": 14_589.0}
# s̄° a 1000 K y 1 bar (J/(mol·K)), NIST-JANAF.
JANAF_S_1000 = {"N2": 228.170, "O2": 243.578, "CO2": 269.299, "H2O": 232.738}


@pytest.mark.parametrize("species", GAS_SPECIES)
def test_enthalpy_matches_the_textbook_tables(species: str) -> None:
    gas = FlueGas.from_fractions({species: 1.0})
    dh = (gas.h(1000.0) - gas.h(298.15)) * molar_mass(species)  # J/mol
    assert dh == pytest.approx(CENGEL_DH[species], rel=1e-3)


@pytest.mark.parametrize("species", ["N2", "O2", "CO2", "H2O"])
def test_absolute_entropy_matches_janaf(species: str) -> None:
    gas = FlueGas.from_fractions({species: 1.0})
    M = molar_mass(species)
    assert gas.s(298.15, 1e5) * M == pytest.approx(
        ideal_gas.S_STANDARD_J_PER_MOL_K[species], abs=1e-6
    )
    assert gas.s(1000.0, 1e5) * M == pytest.approx(JANAF_S_1000[species], abs=0.02)


def test_s0_is_the_integral_of_cp_over_T() -> None:
    integral, _ = quad(lambda T: AIR_DRY.cp(T) / T, 300.0, 1200.0)
    assert AIR_DRY.s0(1200.0) - AIR_DRY.s0(300.0) == pytest.approx(integral, rel=1e-6)
    assert AIR_DRY.s0(298.15) == pytest.approx(0.0, abs=1e-9)
    assert AIR_DRY.h(298.15) == pytest.approx(0.0, abs=1e-6)


def test_isentropic_compression_like_cengel_9_5() -> None:
    # Aire a 300 K comprimido con r = 8: Cengel da T₂s = 540 K (tabla A-17).
    T2s = AIR_DRY.T_isentropic(300.0, 100e3, 800e3)
    assert T2s == pytest.approx(540.0, abs=0.5)
    # Ida y vuelta: expandir de nuevo hasta la presión inicial.
    assert AIR_DRY.T_isentropic(T2s, 800e3, 100e3) == pytest.approx(300.0, abs=1e-6)
    # La entropía absoluta no cambia en el isoentrópico.
    assert AIR_DRY.s(T2s, 800e3) == pytest.approx(AIR_DRY.s(300.0, 100e3), abs=1e-6)


def test_mixture_entropy_includes_mixing() -> None:
    air = AIR_TECHNICAL
    pure = sum(
        w * FlueGas.from_fractions({s: 1.0}).s(400.0, 1e5)
        for s, w in air.mass_fractions.items()
        if w > 0
    )
    mixing = -ideal_gas.R_U / air.M_kg_per_mol * sum(y * math.log(y) for y in air.y if y > 0)
    assert air.s(400.0, 1e5) == pytest.approx(pure + mixing, rel=1e-10)


def test_temperature_range_depends_on_the_components() -> None:
    assert AIR_TECHNICAL.T_min_K < 100.0  # N₂ y O₂: muy por debajo de 0 °C
    assert AIR_DRY.T_min_K == pytest.approx(216.6, abs=0.1)  # el CO₂ del aire seco
    wet = FlueGas.from_fractions(exhaust_composition(3.0))
    assert wet.T_min_K == pytest.approx(273.17, abs=1e-6)  # el agua
    assert AIR_TECHNICAL.T_from_h(AIR_TECHNICAL.h(250.0)) == pytest.approx(250.0, abs=1e-6)


def test_gases_with_water_below_its_triple_point() -> None:
    """Bajo 0,01 °C el agua sigue como gas ideal con c_p constante (Fase 3.7).

    CoolProp no evalúa el agua por debajo de su punto triple; hace falta para el
    estado muerto de la exergía de los gases con el ambiente bajo cero.
    """
    wet = FlueGas.from_fractions(exhaust_composition(3.0))
    T_m = wet.T_min_K
    assert wet.h(T_m - 1e-7) == pytest.approx(wet.h(T_m), abs=1e-3)
    assert wet.s(T_m - 1e-7, 1e5) == pytest.approx(wet.s(T_m, 1e5), abs=1e-6)
    # c_p casi constante entre −20 y 0 °C: la extrapolación sigue la pendiente del límite.
    assert wet.h(T_m) - wet.h(253.15) == pytest.approx(wet.cp(T_m) * (T_m - 253.15), rel=2e-3)
    assert wet.s0(253.15) < wet.s0(T_m)
    assert ideal_gas._ideal_gas("H2O", (250.0,), "cp")[0] == pytest.approx(
        ideal_gas._ideal_gas("H2O", (T_m,), "cp")[0]
    )


def test_combustion_products_generalize_the_methane_preset() -> None:
    for lam in (1.0, 1.5, 3.0):
        gas, n_air = combustion_products(AIR_TECHNICAL, (1.0, 4.0, 0.0, 0.0), lam)
        expected = exhaust_composition(lam)
        for s in GAS_SPECIES:
            assert gas.mole_fractions[s] == pytest.approx(expected[s], abs=1e-12)
        assert n_air == pytest.approx(lam * 2.0 / 0.21)
    # Con aire seco: el argón y el CO₂ del aire pasan a los gases. CH₄ + 2 O₂ → CO₂ + 2 H₂O
    # no cambia el número de moles: salen n_aire + 1 moles por mol de metano.
    gas, n_air = combustion_products(AIR_DRY, (1.0, 4.0, 0.0, 0.0), 2.0)
    total = n_air + 1.0
    y, y_air = gas.mole_fractions, AIR_DRY.mole_fractions
    assert y["H2O"] * total == pytest.approx(2.0)
    assert y["CO2"] * total == pytest.approx(1.0 + n_air * y_air["CO2"])
    assert y["Ar"] * total == pytest.approx(n_air * y_air["Ar"])
    assert y["O2"] * total == pytest.approx(n_air * y_air["O2"] - 2.0)
    with pytest.raises(ValueError, match="al menos 1"):
        combustion_products(AIR_DRY, (1.0, 4.0, 0.0, 0.0), 0.9)


def test_hrsg_reexports_the_gas_model() -> None:
    assert hrsg.FlueGas is FlueGas
    assert hrsg.exhaust_composition is exhaust_composition
    assert hrsg.molar_mass is molar_mass
    assert hrsg.GAS_SPECIES == GAS_SPECIES
