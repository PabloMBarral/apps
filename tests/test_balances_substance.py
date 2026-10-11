"""Tests de core.balances.substance: fluido real, gas ideal e incompresible (Fase 10.1)."""

from __future__ import annotations

import math

import CoolProp.CoolProp as CP
import pytest

from core.balances import substance as sb

C = 273.15
WATER = sb.fluid_substance("Water")
AIR = sb.ideal_gas_substance("air", "variable")
AIR_C = sb.ideal_gas_substance("air", "constant")
IRON = sb.incompressible_substance("iron")


# ---------------------------------------------------------------------
# Fluido real
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pair", "a", "b"),
    [
        ("TP", 300.0 + C, 1e5),
        ("PX", 3e5, 1.0),
        ("TX", 100.0 + C, 0.5),
        ("PH", 3e5, 2864.88e3),
        ("PS", 1.4e6, 7.0e3),
        ("TS", 200.0 + C, 6.5e3),
        ("PV", 1e6, 0.25),
        ("TV", 400.0 + C, 0.05),
        ("PU", 2e5, 2600e3),
    ],
)
def test_fluid_states_match_coolprop(pair: str, a: float, b: float) -> None:
    st = sb.state(WATER, pair, a, b)  # type: ignore[arg-type]
    assert st.p == pytest.approx(CP.PropsSI("P", "T", st.T, "D", 1.0 / st.v, "Water"), rel=1e-6)
    assert st.h == pytest.approx(st.u + st.p * st.v, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("pair", ["VU", "VS", "VX"])
def test_pairs_with_volume_go_round_trip(pair: str) -> None:
    """v y u (tanque rígido con calor dado), v y s y v y x: CoolProp a bajo nivel."""
    ref = sb.state(WATER, "TX", 150.0 + C, 0.3)
    b = {"VU": ref.u, "VS": ref.s, "VX": 0.3}[pair]
    st = sb.state(WATER, pair, ref.v, b)  # type: ignore[arg-type]
    assert st.T == pytest.approx(ref.T, rel=1e-6)
    assert st.x == pytest.approx(0.3, rel=1e-5)
    assert st.region == "saturated_mixture"


def test_unrestrained_expansion_lands_in_the_dome() -> None:
    """Çengel §4-2: 5 kg de agua a 200 kPa y 25 °C que se expanden al doble de volumen."""
    st1 = sb.state(WATER, "TP", 25.0 + C, 2e5)
    st2 = sb.state(WATER, "TV", 25.0 + C, 2.0 * st1.v)
    assert st2.p / 1e3 == pytest.approx(3.169, abs=0.002)
    assert st2.is_two_phase


# ---------------------------------------------------------------------
# Gas ideal
# ---------------------------------------------------------------------


@pytest.mark.parametrize("sub", [AIR, AIR_C], ids=["variable", "constante"])
@pytest.mark.parametrize("pair", ["PH", "PU", "PS", "TS", "VU", "VS", "PV", "TV"])
def test_ideal_gas_pairs_go_round_trip(sub: sb.Substance, pair: str) -> None:
    ref = sb.state(sub, "TP", 450.0, 3e5)
    a = {"P": ref.p, "T": ref.T, "V": ref.v}[pair[0]]
    b = {"H": ref.h, "U": ref.u, "S": ref.s, "V": ref.v}[pair[1]]
    st = sb.state(sub, pair, a, b)  # type: ignore[arg-type]
    assert st.T == pytest.approx(450.0, rel=1e-9)
    assert st.p == pytest.approx(3e5, rel=1e-9)


def test_ideal_gas_enthalpy_is_u_plus_RT() -> None:
    """h = u + R·T: así el balance de un tanque que se llena cierra (Çengel §5-5)."""
    st = sb.state(AIR, "TP", 600.0, 5e5)
    assert st.h - st.u == pytest.approx(AIR.gas.R * 600.0, rel=1e-12)
    assert st.h - st.u == pytest.approx(st.p * st.v, rel=1e-12)


def test_ideal_gas_against_coolprop_air() -> None:
    """A presión baja, el aire de CoolProp es un gas ideal: Δh y Δs coinciden al 0,5 %."""
    a, b = sb.state(AIR, "TP", 300.0, 1e5), sb.state(AIR, "TP", 600.0, 1e5)
    dh = CP.PropsSI("H", "T", 600.0, "P", 1e5, "Air") - CP.PropsSI("H", "T", 300.0, "P", 1e5, "Air")
    ds = CP.PropsSI("S", "T", 600.0, "P", 1e5, "Air") - CP.PropsSI("S", "T", 300.0, "P", 1e5, "Air")
    assert b.h - a.h == pytest.approx(dh, rel=5e-3)
    assert b.s - a.s == pytest.approx(ds, rel=5e-3)


def test_diffuser_with_constant_cp() -> None:
    """Çengel §5-4 (difusor): aire a 283,15 K y 200 m/s que sale casi quieto: 303 K."""
    st1 = sb.state(AIR_C, "TP", 283.15, 8e4)
    st2 = sb.state(AIR_C, "PH", 8e4, st1.h + 200.0**2 / 2.0)
    assert st2.T == pytest.approx(303.0, abs=0.2)


# ---------------------------------------------------------------------
# Incompresible
# ---------------------------------------------------------------------


def test_incompressible_model() -> None:
    """Vademecum §13: Δu = c·ΔT, Δh = c·ΔT + v·Δp y Δs = c·ln(T₂/T₁)."""
    a = sb.state(IRON, "TP", 300.0, 1e5)
    b = sb.state(IRON, "TP", 500.0, 3e5)
    c, v = IRON.material.c, IRON.material.v
    assert b.u - a.u == pytest.approx(c * 200.0)
    assert b.h - a.h == pytest.approx(c * 200.0 + v * 2e5)
    assert b.s - a.s == pytest.approx(c * math.log(500.0 / 300.0))
    for pair, x in (("PH", b.h), ("PU", b.u), ("PS", b.s)):
        assert sb.state(IRON, pair, 3e5, x).T == pytest.approx(500.0)  # type: ignore[arg-type]


def test_incompressible_water_against_coolprop() -> None:
    """El agua líquida a 1 atm de 20 a 80 °C: el modelo incompresible acierta al 0,5 %."""
    w = sb.incompressible_substance("water")
    a, b = sb.state(w, "TP", 20.0 + C, 101325.0), sb.state(w, "TP", 80.0 + C, 101325.0)
    du = CP.PropsSI("U", "T", 80 + C, "P", 101325, "Water") - CP.PropsSI(
        "U", "T", 20 + C, "P", 101325, "Water"
    )
    ds = CP.PropsSI("S", "T", 80 + C, "P", 101325, "Water") - CP.PropsSI(
        "S", "T", 20 + C, "P", 101325, "Water"
    )
    assert b.u - a.u == pytest.approx(du, rel=5e-3)
    assert b.s - a.s == pytest.approx(ds, rel=5e-3)


def test_custom_material_and_overrides() -> None:
    iron_cengel = sb.incompressible_substance("iron", c=450.0)
    assert iron_cengel.material.c == 450.0 and iron_cengel.material.rho == 7870.0
    custom = sb.incompressible_substance("custom", c=1000.0, rho=2000.0)
    assert custom.material.v == pytest.approx(5e-4)
    assert custom.label == "Material a elección"


# ---------------------------------------------------------------------
# Mensajes al alumno
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sub", "pair", "a", "b", "words"),
    [
        (AIR, "PX", 1e5, 0.5, "un gas ideal no tiene"),
        (IRON, "TV", 300.0, 0.1, "volumen específico del hierro es fijo"),
        (IRON, "TS", 300.0, 10.0, "depende solo de T"),
        (IRON, "PX", 1e5, 0.5, "no tiene título"),
        (WATER, "TP", -1.0, 1e5, "tiene que ser positiva"),
        (WATER, "PX", 1e5, 1.5, "entre 0 y 1"),
        (WATER, "TP", 300.0, 1e12, "máximo de validez"),
        (AIR, "PH", 1e5, 1e9, "fuera del rango de los polinomios"),
    ],
)
def test_student_errors(sub: sb.Substance, pair: str, a: float, b: float, words: str) -> None:
    with pytest.raises(ValueError, match=words):
        sb.state(sub, pair, a, b)  # type: ignore[arg-type]


def test_labels_and_nouns() -> None:
    assert sb.of(WATER) == "del agua"
    assert sb.of(sb.incompressible_substance("glycerin")) == "de la glicerina"
    assert AIR.noun == "el aire" and AIR.label == "Aire (gas ideal, c_p variable)"
    assert sb.state(WATER, "TP", 300.0, 1e5).relabel("2").label == "2"
    with pytest.raises(ValueError, match="no está en la lista"):
        sb.fluid_substance("Unobtainium")
    with pytest.raises(ValueError, match="hacen falta su c y su densidad"):
        sb.incompressible_substance("custom", c=1.0)
