"""Tests de los polinomios NASA-9 de la combustión — Fase 5.

Contra las tablas de Cengel (A-18 a A-23, que salen de JANAF), el gas ideal de
CoolProp (la base de ``core.ideal_gas``), los poderes caloríficos de ISO
6976:2016 y la tabla de entalpías de formación del vademecum (§16.12).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from core.combustion.thermo import (
    T_REF_K,
    all_species,
    gas_mixture_entropy,
    mixture_enthalpy,
    species,
    water_hfg_J_per_mol,
)
from core.ideal_gas import FlueGas, molar_mass

DATA = Path(__file__).resolve().parents[1] / "data"


def test_the_csv_has_every_species_with_contiguous_ranges() -> None:
    data = all_species()
    assert len(data) == 32
    for sp in data.values():
        for a, b in zip(sp.intervals, sp.intervals[1:], strict=False):
            assert a.T_hi == pytest.approx(b.T_lo)
        if sp.is_gas:
            assert sp.T_max_K == 6000.0
            assert sp.T_min_K <= 300.0


def test_coefficients_are_the_ones_of_nasa_tp_2002() -> None:
    """Muestra de McBride, Zehe & Gordon (2002): el CO₂ entre 200 y 1000 K."""
    co2 = species("CO2")
    a = co2.intervals[0].a
    assert a[:3] == (4.943650540e04, -6.264116010e02, 5.301725240e00)
    assert co2.hf_J_per_mol == -393_510.0
    assert co2.M_kg_per_mol == pytest.approx(0.0440095)


@pytest.mark.parametrize("key", ["CO2", "H2O", "CO", "CH4", "C8H18(l)", "H2O(l)", "OH", "NO"])
def test_enthalpy_at_25C_is_the_formation_enthalpy(key: str) -> None:
    sp = species(key)
    assert sp.h(T_REF_K) == sp.hf_J_per_mol
    assert sp.delta_h(T_REF_K) == 0.0


# Cengel A-18 a A-21 y A-23: h̄(T) − h̄(298 K) en kJ/kmol (= J/mol), de JANAF.
CENGEL = {
    "N2": {1000: 30129 - 8669, 1500: 47073 - 8669, 2000: 64810 - 8669},
    "O2": {1000: 31389 - 8682, 1500: 49292 - 8682, 2000: 67881 - 8682},
    "CO2": {1000: 42769 - 9364, 1500: 71078 - 9364, 2000: 100804 - 9364},
    "CO": {1000: 30355 - 8669, 1500: 47517 - 8669, 2000: 65408 - 8669},
    "H2O": {1000: 35882 - 9904, 1500: 57999 - 9904, 2000: 82593 - 9904},
}


@pytest.mark.parametrize("key", list(CENGEL))
def test_sensible_enthalpy_matches_the_cengel_tables(key: str) -> None:
    # El H₂O de NASA (Woolley, 1987) se separa de JANAF a alta T: hasta 0,5 % a 2000 K.
    tol = 0.006 if key == "H2O" else 5e-4
    for T, dh in CENGEL[key].items():
        assert species(key).delta_h(T) == pytest.approx(dh, rel=tol)


@pytest.mark.parametrize("key", ["N2", "O2", "CO2", "Ar"])
def test_same_as_the_coolprop_ideal_gas_of_core_ideal_gas(key: str) -> None:
    gas = FlueGas.from_fractions({key: 1.0})
    for T in (500.0, 1000.0, 1500.0, 1950.0):
        coolprop = gas.h(T) * molar_mass(key)
        assert species(key).delta_h(T) == pytest.approx(coolprop, rel=3e-4)


def test_cp_is_the_derivative_of_h_and_s() -> None:
    for key in ("CO2", "H2O", "CH4", "OH"):
        sp = species(key)
        for T in (400.0, 1200.0, 2500.0):
            d = 1e-3
            assert (sp.h(T + d) - sp.h(T - d)) / (2 * d) == pytest.approx(sp.cp(T), rel=1e-6)
            ds = (sp.s0(T + d) - sp.s0(T - d)) / (2 * d)
            assert ds == pytest.approx(sp.cp(T) / T, rel=1e-6)


def _iso6976_hc25() -> dict[str, float]:
    with (DATA / "iso6976_calorific_values.csv").open(encoding="utf-8") as fh:
        return {row["name"]: float(row["Hc_25C"]) for row in csv.DictReader(fh)}


@pytest.mark.parametrize(
    ("key", "iso_name", "c", "h"),
    [
        ("CH4", "methane", 1, 4),
        ("C2H6", "ethane", 2, 6),
        ("C3H8", "propane", 3, 8),
        ("n-C4H10", "n-butane", 4, 10),
        ("H2", "hydrogen", 0, 2),
        ("CO", "carbon monoxide", 1, 0),
    ],
)
def test_hhv_matches_iso_6976(key: str, iso_name: str, c: int, h: int) -> None:
    """PCS a 25 °C desde las h_f contra ISO 6976:2016, Tabla 3 (kJ/mol)."""
    hhv = species(key).hf_J_per_mol - (
        c * species("CO2").hf_J_per_mol + h / 2 * species("H2O(l)").hf_J_per_mol
    )
    assert hhv / 1e3 == pytest.approx(_iso6976_hc25()[iso_name], rel=1e-4)


# Tabla de §16.12 del vademecum (Klein & Nellis, 2012, sobre McBride et al., 2002).
VADEMECUM = {
    "CO": (-110_528, 197.648),
    "CO2": (-393_486, 213.774),
    "H2O": (-241_811, 188.818),
    "H2O(l)": (-285_813, 69.938),
    "SO2": (-296_792, 248.207),
    "CH4": (-74_595, 186.360),
    "C2H6": (-83_846, 229.207),
    "C3H8": (-104_674, 270.298),
    "C8H18(l)": (-250_302, 360.833),
    "CH3OH(l)": (-239_004, 126.976),
    "C2H5OH(l)": (-277_402, 159.206),
    "H2": (0, 130.673),
    "O2": (0, 205.137),
    "N2": (0, 191.598),
}


@pytest.mark.parametrize("key", list(VADEMECUM))
def test_formation_enthalpy_and_entropy_close_to_the_vademecum_table(key: str) -> None:
    hf, s = VADEMECUM[key]
    sp = species(key)
    assert sp.hf_J_per_mol == pytest.approx(hf, rel=5e-4, abs=1.0)
    # Las entropías de la tabla son a 1 bar (como las de NASA); las de los combustibles
    # líquidos vienen de otros ajustes (TRC) y difieren hasta 0,9 J/(mol·K).
    assert sp.s0(T_REF_K) == pytest.approx(s, abs=1.0 if key.endswith("(l)") else 0.05)


def test_water_latent_heat_at_25C() -> None:
    """h_fg = 44 004 kJ/kmol ≈ 2442 kJ/kg (vademecum §16.9)."""
    assert water_hfg_J_per_mol() == pytest.approx(44_004.0)
    assert water_hfg_J_per_mol() / species("H2O").M_kg_per_mol / 1e3 == pytest.approx(
        2442.6, abs=0.1
    )


def test_mixture_helpers() -> None:
    moles = {"CO2": 1.0, "H2O": 2.0, "N2": 7.52}
    assert mixture_enthalpy(moles, 1500.0) == pytest.approx(
        sum(n * species(k).h(1500.0) for k, n in moles.items())
    )
    # pura a 1 bar: s = s°
    assert gas_mixture_entropy({"N2": 1.0}, 500.0, 1e5) == pytest.approx(species("N2").s0(500.0))
    # mezcla: cada una a su presión parcial
    s_mix = gas_mixture_entropy({"O2": 0.21, "N2": 0.79}, T_REF_K, 101_325.0)
    expected = sum(
        y * (species(k).s0(T_REF_K) - 8.314462618 * math.log(y * 1.01325))
        for k, y in {"O2": 0.21, "N2": 0.79}.items()
    )
    assert s_mix == pytest.approx(expected)


def test_out_of_range_temperature_explains() -> None:
    with pytest.raises(ValueError, match="solo está tabulado"):
        species("C8H18(l)").check_T(450.0)
    with pytest.raises(ValueError, match="no está en la base"):
        species("C60")
