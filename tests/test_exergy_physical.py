"""Tests de core.exergy.physical (vademecum §11.1 a §11.6; Cengel cap. 8 y 10-8)."""

from __future__ import annotations

import math

import pytest
from CoolProp.CoolProp import PropsSI

from core.exergy.physical import (
    FINITE_SOURCE_EXAMPLES,
    HEAT_EXAMPLES,
    PHYSICAL_EXAMPLES,
    Ambient,
    ProcessExergy,
    finite_source_exergy,
    finite_source_notes,
    heat_exergy,
    heat_notes,
    physical_exergy,
    physical_exergy_to_dict,
    physical_notes,
)
from core.fluids import SUPPORTED_FLUIDS, suggested_inputs

BTU = 1055.05585262


def _from_example(name: str):
    ex = PHYSICAL_EXAMPLES[name]
    first = physical_exergy(
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
    if ex.second is not None:
        pair, values = ex.second
        second = physical_exergy(ex.fluid, pair, ambient=ex.ambient, **values)
    return first, second


def _name(prefix: str) -> str:
    return next(n for n in PHYSICAL_EXAMPLES if n.startswith(prefix))


# ---------------------------------------------------------------------
# Contra el libro
# ---------------------------------------------------------------------


def test_cengel_10_8_steam_exergy_at_turbine_inlet_and_outlet() -> None:
    """Cengel 10-8: ψ₄ = 449 kJ/kg con el ambiente a 290 K y 100 kPa."""
    first, second = _from_example(_name("Vapor que entra a la turbina"))
    assert second is not None
    assert second.psi_J_per_kg == pytest.approx(449.0e3, rel=2e-3)
    assert first.psi_J_per_kg == pytest.approx(1162.1e3, rel=1e-3)
    # Δψ de una turbina isoentrópica = −w_T = Δh.
    proc = ProcessExergy(first, second)
    assert proc.dpsi_J_per_kg == pytest.approx(proc.dh_J_per_kg, rel=1e-3)


def test_cengel_compressed_air_tank() -> None:
    """Cengel cap. 8: 200 m³ de aire a 1 MPa y 300 K, ambiente a 100 kPa y 300 K → 281 MJ."""
    first, _ = _from_example(_name("Aire comprimido"))
    assert first.X_J == pytest.approx(281.0e6, rel=3e-3)
    # Con gas ideal: φ = R·T₀·(p₀/p − 1 + ln(p/p₀)) = 120,76 kJ/kg.
    R = 287.0
    phi_ideal = R * 300.0 * (0.1 - 1.0 + math.log(10.0))
    assert first.phi_J_per_kg == pytest.approx(phi_ideal, rel=3e-3)
    assert first.psi_thermal_J_per_kg == pytest.approx(0.0, abs=1e-6)


def test_cengel_r134a_minimum_compressor_work() -> None:
    """Cengel cap. 8: Δψ = 38,0 kJ/kg (el trabajo mínimo) y Δh ≈ 40,3 kJ/kg."""
    first, second = _from_example(_name("R-134a en un compresor"))
    assert second is not None
    proc = ProcessExergy(first, second)
    assert proc.dpsi_J_per_kg == pytest.approx(38.0e3, rel=3e-3)
    assert proc.dh_J_per_kg == pytest.approx(40.3e3, rel=3e-3)


def test_cengel_wind_turbine() -> None:
    """Cengel cap. 8: 10 m/s sobre 1414 kg/s → 70,7 kW (todo es energía cinética)."""
    first, _ = _from_example(_name("Viento"))
    assert first.psi_J_per_kg == pytest.approx(0.0, abs=1e-6)
    assert first.X_W == pytest.approx(70.7e3, rel=1e-3)


def test_cengel_furnace_heat_exergy() -> None:
    """Cengel cap. 8: 3000 Btu/s a 2000 R con el ambiente a 77 °F → 2195 Btu/s."""
    ex = next(e for n, e in HEAT_EXAMPLES.items() if n.startswith("Hogar"))
    r = heat_exergy(ex.Q_W, ex.T_K, ex.T0_K)
    assert r.X_W / BTU == pytest.approx(2195.0, rel=1e-3)
    assert r.X_W + r.anergy_W == pytest.approx(r.Q_W)


def test_cengel_iron_block() -> None:
    """Cengel cap. 8: 500 kg de hierro a 473 K, c = 0,45 kJ/(kg·K), T₀ = 300 K → 8191 kJ."""
    ex = next(e for n, e in FINITE_SOURCE_EXAMPLES.items() if n.startswith("Bloque"))
    r = finite_source_exergy(ex.m_kg, ex.c_J_per_kg_K, ex.T_K, ex.T0_K)
    assert r.Phi_J == pytest.approx(8191.0e3, rel=5e-4)


# ---------------------------------------------------------------------
# Propiedades de la exergía
# ---------------------------------------------------------------------


@pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
def test_dead_state_has_zero_exergy(fluid: str) -> None:
    amb = Ambient(298.15, 101_325.0)
    r = physical_exergy(fluid, "TP", ambient=amb, t=298.15, p=101_325.0)
    assert r.psi_J_per_kg == pytest.approx(0.0, abs=1e-6)
    assert r.phi_J_per_kg == pytest.approx(0.0, abs=1e-6)
    assert r.is_dead_state
    assert any("estado muerto" in n for n in physical_notes(r))


@pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
@pytest.mark.parametrize("pair", ["TP", "PH", "PS", "TV"])
def test_split_adds_up_and_phi_is_non_negative(fluid: str, pair: str) -> None:
    """ψ_T + ψ_M = ψ y φ ≥ 0 en los estados sugeridos de cada fluido."""
    r = physical_exergy(fluid, pair, **suggested_inputs(fluid, pair))  # type: ignore[arg-type]
    assert r.psi_thermal_J_per_kg + r.psi_mechanical_J_per_kg == pytest.approx(
        r.psi_J_per_kg, abs=1e-6 * max(1.0, abs(r.psi_J_per_kg))
    )
    assert r.phi_J_per_kg >= -1e-6
    assert r.psi_thermal_J_per_kg >= -1e-6  # enfriar a p constante hasta T₀ nunca cuesta


def test_psi_and_phi_definitions_against_coolprop() -> None:
    amb = Ambient(290.0, 100_000.0)
    r = physical_exergy("Water", "TP", ambient=amb, t=673.15, p=2.0e6)
    h0 = PropsSI("H", "T", 290.0, "P", 1.0e5, "Water")
    s0 = PropsSI("S", "T", 290.0, "P", 1.0e5, "Water")
    u0 = PropsSI("U", "T", 290.0, "P", 1.0e5, "Water")
    v0 = 1.0 / PropsSI("D", "T", 290.0, "P", 1.0e5, "Water")
    h = PropsSI("H", "T", 673.15, "P", 2.0e6, "Water")
    s = PropsSI("S", "T", 673.15, "P", 2.0e6, "Water")
    u = PropsSI("U", "T", 673.15, "P", 2.0e6, "Water")
    v = 1.0 / PropsSI("D", "T", 673.15, "P", 2.0e6, "Water")
    assert r.psi_J_per_kg == pytest.approx((h - h0) - 290.0 * (s - s0), rel=1e-9)
    assert r.phi_J_per_kg == pytest.approx((u - u0) + 1.0e5 * (v - v0) - 290.0 * (s - s0), rel=1e-9)


def test_kinetic_and_potential_are_pure_exergy() -> None:
    r = physical_exergy("Water", "TP", t=298.15, p=101_325.0, speed_m_per_s=20.0, height_m=100.0)
    assert r.x_flow_J_per_kg == pytest.approx(0.5 * 400.0 + 9.80665 * 100.0, rel=1e-9)
    assert r.x_mass_J_per_kg == pytest.approx(r.x_flow_J_per_kg, rel=1e-9)
    assert any("exergía pura" in n for n in physical_notes(r))


def test_cold_water_has_positive_exergy_and_note() -> None:
    r, _ = _from_example(_name("Agua fría"))
    assert r.psi_J_per_kg > 0.0
    assert r.X_W == pytest.approx(10.0 * r.psi_J_per_kg)
    assert any("más frío que el ambiente" in n for n in physical_notes(r))


def test_low_pressure_steam_notes_vacuum_and_negative_mechanical() -> None:
    r, _ = _from_example(_name("Vapor de baja presión"))
    assert r.psi_mechanical_J_per_kg < 0.0
    assert r.phi_J_per_kg > 5.0 * r.psi_J_per_kg
    notes = " ".join(physical_notes(r))
    assert "mecánica" in notes and "espacio vacío" in notes and "líquido" in notes


def test_two_phase_state_at_T0_has_no_thermal_part() -> None:
    """Una mezcla a T₀: la parte térmica es 0 (Δh = T₀·Δs dentro de la campana)."""
    amb = Ambient(298.15, 101_325.0)
    r = physical_exergy("Water", "TX", ambient=amb, t=298.15, x=0.5)
    assert r.psi_thermal_J_per_kg == pytest.approx(0.0, abs=1e-6)
    assert r.psi_mechanical_J_per_kg == pytest.approx(r.psi_J_per_kg, rel=1e-9)


def test_state_exactly_at_saturation_pressure_of_T0() -> None:
    """Vapor sobrecalentado a p_sat(T₀): el estado intermedio es el líquido saturado."""
    amb = Ambient(298.15, 101_325.0)
    p_sat = PropsSI("P", "T", 298.15, "Q", 0, "Water")
    r = physical_exergy("Water", "TP", ambient=amb, t=373.15, p=p_sat)
    assert math.isfinite(r.psi_thermal_J_per_kg)
    assert r.psi_thermal_J_per_kg + r.psi_mechanical_J_per_kg == pytest.approx(
        r.psi_J_per_kg, rel=1e-9
    )


def test_heat_below_ambient_changes_sign() -> None:
    ex = next(e for n, e in HEAT_EXAMPLES.items() if n.startswith("Cámara"))
    r = heat_exergy(ex.Q_W, ex.T_K, ex.T0_K)
    assert r.X_W < 0.0
    assert r.X_W == pytest.approx(-0.1685e3, rel=1e-3)
    assert any("más fría que el ambiente" in n for n in heat_notes(r))


def test_finite_source_is_non_negative_and_zero_at_T0() -> None:
    hot = finite_source_exergy(1000.0, 4180.0, 353.15, 293.15)
    cold = finite_source_exergy(1000.0, 4180.0, 278.15, 298.15)
    at_T0 = finite_source_exergy(1000.0, 4180.0, 298.15, 298.15)
    assert hot.Phi_J > 0.0 and cold.Phi_J > 0.0
    assert at_T0.Phi_J == pytest.approx(0.0, abs=1e-6)
    assert hot.fraction_of_heat == pytest.approx(0.0902, abs=5e-4)
    assert min(phi for _, phi in hot.curve()) >= -1e-6
    assert any("más frío" in n for n in finite_source_notes(cold))


# ---------------------------------------------------------------------
# Validaciones (mensajes al alumno)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"T0_K": 400.0}, "T₀"),
        ({"T0_K": 200.0}, "T₀"),
        ({"p0_Pa": 1.0e3}, "p₀"),
        ({"p0_Pa": 5.0e7}, "p₀"),
    ],
)
def test_ambient_out_of_range(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Ambient(**kwargs)


def test_invalid_amounts_and_speeds() -> None:
    with pytest.raises(ValueError, match="masa"):
        physical_exergy("Water", "TP", t=300.0, p=1e5, amount="masa", mass_kg=0.0)
    with pytest.raises(ValueError, match="caudal"):
        physical_exergy("Water", "TP", t=300.0, p=1e5, amount="caudal", m_dot_kg_s=-1.0)
    with pytest.raises(ValueError, match="velocidad"):
        physical_exergy("Water", "TP", t=300.0, p=1e5, speed_m_per_s=5000.0)
    with pytest.raises(ValueError, match="positiva"):
        heat_exergy(1.0, -5.0, 298.15)
    with pytest.raises(ValueError, match="masa"):
        finite_source_exergy(0.0, 4180.0, 350.0, 298.15)


def test_process_needs_same_fluid_and_ambient() -> None:
    a = physical_exergy("Water", "TP", t=400.0, p=1e5)
    b = physical_exergy("R134a", "TP", t=300.0, p=1e5)
    with pytest.raises(ValueError, match="mismo fluido"):
        ProcessExergy(a, b)


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PHYSICAL_EXAMPLES))
@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_every_example_exports(name: str, system: str) -> None:
    first, second = _from_example(name)
    data = physical_exergy_to_dict(first, system, second)  # type: ignore[arg-type]
    assert "psi" in data and "phi" in data
    if first.X_J is not None:
        assert "X_masa" in data
    if first.X_W is not None:
        assert "X_flujo" in data
    if second is not None:
        assert "delta_psi" in data
