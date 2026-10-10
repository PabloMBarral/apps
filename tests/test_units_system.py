"""Tests de :mod:`core.units_system` — Fase 1.4.

Cobertura: round-trip exhaustivo SI → sistema → SI por cada
(kind, system) y por valores típicos; puntos de conversión conocidos
(0 K ↔ −459.67 °F, 1 atm ↔ 14.6959 psia, 1 Btu/lb ↔ 2326 J/kg, etc.);
edge cases (T=0, valores grandes); manejo de inputs inválidos.
"""

from __future__ import annotations

import math
import re

import pytest

from core.units_system import (
    DEFAULT_SYSTEM,
    SUPPORTED_SYSTEMS,
    convert_from_si,
    convert_to_si,
    format_quantity,
    parse_user_input,
    unit_label,
)

_KINDS = (
    "temperature",
    "pressure",
    "specific_enthalpy",
    "specific_entropy",
    "specific_volume",
    "specific_heat",
    "temperature_difference",
    "density",
    "speed",
    "dynamic_viscosity",
    "thermal_conductivity",
    "diffusivity",
    "mass_flow",
    "power",
    "volume_flow",
    "molar_enthalpy",
    "molar_entropy",
    "mass",
    "energy",
    "length",
    "heat_transfer_coefficient",
    "heat_flux",
    "thermal_resistance",
    "area",
    "small_length",
    "heat_rate",
    "linear_heat_rate",
    "absolute_temperature",
    "wavelength",
    "wavelength_temperature",
    "spectral_emissive_power",
    "fouling_resistance",
    "entropy_rate",
    "volume",
    "entropy",
    "entropy_flow",
    "molar_mass",
    "amount",
)


class TestDefaults:
    def test_default_system_is_tecnico(self) -> None:
        assert DEFAULT_SYSTEM == "Técnico"

    def test_supported_systems(self) -> None:
        assert SUPPORTED_SYSTEMS == ("SI", "Técnico", "Inglés")


class TestUnitLabels:
    @pytest.mark.parametrize(
        "kind,system,expected",
        [
            ("temperature", "SI", "K"),
            ("temperature", "Técnico", "°C"),
            ("temperature", "Inglés", "°F"),
            ("pressure", "SI", "Pa"),
            ("pressure", "Técnico", "bar"),
            ("pressure", "Inglés", "psia"),
            ("specific_enthalpy", "SI", "J/kg"),
            ("specific_enthalpy", "Técnico", "kJ/kg"),
            ("specific_enthalpy", "Inglés", "Btu/lb"),
            ("specific_entropy", "SI", "J/(kg·K)"),
            ("specific_entropy", "Técnico", "kJ/(kg·K)"),
            ("specific_entropy", "Inglés", "Btu/(lb·°R)"),
            ("specific_volume", "SI", "m³/kg"),
            ("specific_volume", "Técnico", "m³/kg"),
            ("specific_volume", "Inglés", "ft³/lb"),
            ("specific_heat", "SI", "J/(kg·K)"),
            ("specific_heat", "Técnico", "kJ/(kg·K)"),
            ("specific_heat", "Inglés", "Btu/(lb·°R)"),
            ("temperature_difference", "SI", "K"),
            ("temperature_difference", "Técnico", "°C"),
            ("temperature_difference", "Inglés", "°F"),
            ("density", "SI", "kg/m³"),
            ("density", "Técnico", "kg/m³"),
            ("density", "Inglés", "lb/ft³"),
            ("speed", "SI", "m/s"),
            ("speed", "Técnico", "m/s"),
            ("speed", "Inglés", "ft/s"),
            ("dynamic_viscosity", "SI", "Pa·s"),
            ("dynamic_viscosity", "Técnico", "Pa·s"),
            ("dynamic_viscosity", "Inglés", "lb/(ft·s)"),
            ("thermal_conductivity", "SI", "W/(m·K)"),
            ("thermal_conductivity", "Técnico", "W/(m·K)"),
            ("thermal_conductivity", "Inglés", "Btu/(h·ft·°F)"),
            ("diffusivity", "SI", "m²/s"),
            ("diffusivity", "Técnico", "m²/s"),
            ("diffusivity", "Inglés", "ft²/s"),
            ("mass_flow", "SI", "kg/s"),
            ("mass_flow", "Técnico", "kg/s"),
            ("mass_flow", "Inglés", "lb/s"),
            ("power", "SI", "W"),
            ("power", "Técnico", "kW"),
            ("power", "Inglés", "Btu/s"),
            ("volume_flow", "SI", "m³/s"),
            ("volume_flow", "Técnico", "m³/s"),
            ("volume_flow", "Inglés", "ft³/s"),
            ("molar_enthalpy", "SI", "J/mol"),
            ("molar_enthalpy", "Técnico", "kJ/kmol"),
            ("molar_enthalpy", "Inglés", "Btu/lbmol"),
            ("molar_entropy", "SI", "J/(mol·K)"),
            ("molar_entropy", "Técnico", "kJ/(kmol·K)"),
            ("molar_entropy", "Inglés", "Btu/(lbmol·°R)"),
            ("mass", "Técnico", "kg"),
            ("mass", "Inglés", "lb"),
            ("energy", "SI", "J"),
            ("energy", "Técnico", "kJ"),
            ("energy", "Inglés", "Btu"),
            ("length", "Técnico", "m"),
            ("length", "Inglés", "ft"),
            ("heat_transfer_coefficient", "Técnico", "W/(m²·K)"),
            ("heat_transfer_coefficient", "Inglés", "Btu/(h·ft²·°F)"),
            ("heat_flux", "Inglés", "Btu/(h·ft²)"),
            ("thermal_resistance", "SI", "K/W"),
            ("thermal_resistance", "Inglés", "h·°F/Btu"),
            ("area", "Inglés", "ft²"),
            ("small_length", "SI", "m"),
            ("small_length", "Técnico", "mm"),
            ("small_length", "Inglés", "in"),
            ("heat_rate", "Técnico", "W"),
            ("heat_rate", "Inglés", "Btu/h"),
            ("linear_heat_rate", "Inglés", "Btu/(h·ft)"),
            ("absolute_temperature", "Técnico", "K"),
            ("absolute_temperature", "Inglés", "°R"),
            ("wavelength", "Inglés", "μm"),
            ("wavelength_temperature", "SI", "μm·K"),
            ("wavelength_temperature", "Inglés", "μm·°R"),
            ("spectral_emissive_power", "Técnico", "W/(m²·μm)"),
            ("spectral_emissive_power", "Inglés", "Btu/(h·ft²·μm)"),
            ("fouling_resistance", "SI", "m²·K/W"),
            ("fouling_resistance", "Inglés", "h·ft²·°F/Btu"),
            ("entropy_rate", "Técnico", "W/K"),
            ("entropy_rate", "Inglés", "Btu/(h·°R)"),
            ("volume", "Técnico", "m³"),
            ("volume", "Inglés", "ft³"),
            ("entropy", "SI", "J/K"),
            ("entropy", "Técnico", "kJ/K"),
            ("entropy", "Inglés", "Btu/°R"),
            ("entropy_flow", "Técnico", "kW/K"),
            ("entropy_flow", "Inglés", "Btu/(s·°R)"),
            ("molar_mass", "SI", "kg/mol"),
            ("molar_mass", "Técnico", "kg/kmol"),
            ("molar_mass", "Inglés", "lb/lbmol"),
            ("amount", "Técnico", "kmol"),
            ("amount", "Inglés", "lbmol"),
        ],
    )
    def test_label(self, kind: str, system: str, expected: str) -> None:
        assert unit_label(kind, system) == expected  # type: ignore[arg-type]


class TestRoundTrip:
    """Para cada (kind, system) y para una grilla de valores, el ciclo
    ``si → user → si`` debe recuperar el valor SI con tolerancia ≤ 1e-12."""

    _VALUES_PER_KIND: dict[str, tuple[float, ...]] = {
        "temperature": (0.0, 100.0, 273.15, 298.15, 500.0, 1500.0),
        "pressure": (1.0, 1.0e3, 1.0e5, 1.01325e5, 1.0e7),
        "specific_enthalpy": (0.0, 1.0e3, 1.0e6, 2.7e6, 1.0e8),
        "specific_entropy": (0.0, 100.0, 1000.0, 7000.0, 1.0e5),
        "specific_volume": (1.0e-4, 1.0e-3, 1.0, 100.0),
        "specific_heat": (0.0, 100.0, 1000.0, 4186.8, 1.0e5),
        "temperature_difference": (-50.0, 0.0, 1.0, 20.0, 300.0),
        "density": (0.01, 1.0, 1.2, 997.0, 1.0e4),
        "speed": (0.0, 1.0, 340.0, 1500.0),
        "dynamic_viscosity": (1.0e-6, 1.8e-5, 8.9e-4, 1.0),
        "thermal_conductivity": (0.01, 0.026, 0.6, 400.0),
        "diffusivity": (1.0e-8, 1.4e-7, 1.5e-5, 1.0e-3),
        "mass_flow": (0.0, 1.0, 50.0, 600.0),
        "power": (0.0, 1.0, 1.0e3, 2.1e8),
        "volume_flow": (0.0, 1.0e-4, 0.05, 30.0),
        "molar_enthalpy": (-393_510.0, 0.0, 44_004.0, 5.0e6),
        "molar_entropy": (0.0, 69.95, 213.79, 1.0e4),
        "mass": (0.0, 1.0, 2323.0, 1.0e6),
        "energy": (0.0, 1.0, 2.81e8, 1.0e12),
        "length": (0.0, 1.0, 100.0, 3000.0),
        "heat_transfer_coefficient": (0.0, 5.0, 40.0, 1.0e4),
        "heat_flux": (0.0, 1.0, 1.0e3, 1.0e6),
        "thermal_resistance": (0.0, 1.0e-4, 0.94, 50.0),
        "area": (0.0, 1.0e-4, 1.2, 1.0e3),
        "small_length": (0.0, 1.0e-3, 0.025, 0.3),
        "heat_rate": (0.0, 1.0, 630.0, 1.0e6),
        "linear_heat_rate": (0.0, 1.0, 121.0, 1.0e4),
        "absolute_temperature": (0.0, 77.0, 298.15, 5800.0),
        "wavelength": (0.0, 1.0e-7, 2.9e-6, 1.0e-3),
        "wavelength_temperature": (0.0, 1.0e-3, 2.897771955e-3, 0.05),
        "spectral_emissive_power": (0.0, 1.0, 3.846e9, 8.3e13),
        "fouling_resistance": (0.0, 1.0e-4, 9.0e-4, 0.01),
        "entropy_rate": (0.0, 0.5, 120.0, 1.0e5),
        "volume": (0.0, 1.0e-3, 8.011, 1.0e4),
        "entropy": (0.0, 1.0, 4.4e4, 1.0e7),
        "entropy_flow": (0.0, 1.0, 950.0, 1.0e6),
        "molar_mass": (2.016e-3, 0.028966, 0.18),
        "amount": (0.0, 1.0, 361.6, 8000.0),
    }

    @pytest.mark.parametrize("kind", _KINDS)
    @pytest.mark.parametrize("system", SUPPORTED_SYSTEMS)
    def test_si_user_si_recovers(self, kind: str, system: str) -> None:
        for value_si in self._VALUES_PER_KIND[kind]:
            user = convert_from_si(value_si, kind, system)  # type: ignore[arg-type]
            back = convert_to_si(user, kind, system)  # type: ignore[arg-type]
            assert math.isclose(back, value_si, rel_tol=1.0e-12, abs_tol=1.0e-12), (
                f"round-trip failed for kind={kind}, system={system}, "
                f"value_si={value_si!r}: user={user!r}, back={back!r}"
            )

    @pytest.mark.parametrize("kind", _KINDS)
    @pytest.mark.parametrize("system", SUPPORTED_SYSTEMS)
    def test_user_si_user_recovers(self, kind: str, system: str) -> None:
        for value_si in self._VALUES_PER_KIND[kind]:
            user = convert_from_si(value_si, kind, system)  # type: ignore[arg-type]
            back_user = convert_from_si(
                convert_to_si(user, kind, system),
                kind,
                system,  # type: ignore[arg-type]
            )
            assert math.isclose(back_user, user, rel_tol=1.0e-12, abs_tol=1.0e-12)


class TestKnownConversionPoints:
    """Anclas físicas: puntos donde la conversión es exacta o bien tabulada."""

    # --- Temperatura ---

    def test_absolute_zero_kelvin_is_zero(self) -> None:
        assert convert_from_si(0.0, "temperature", "SI") == 0.0

    def test_absolute_zero_in_celsius(self) -> None:
        assert convert_from_si(0.0, "temperature", "Técnico") == pytest.approx(-273.15)

    def test_absolute_zero_in_fahrenheit(self) -> None:
        assert convert_from_si(0.0, "temperature", "Inglés") == pytest.approx(-459.67)

    def test_water_ice_point_celsius(self) -> None:
        assert convert_from_si(273.15, "temperature", "Técnico") == pytest.approx(0.0, abs=1e-12)

    def test_water_ice_point_fahrenheit(self) -> None:
        assert convert_from_si(273.15, "temperature", "Inglés") == pytest.approx(32.0, abs=1e-9)

    def test_room_temp_fahrenheit(self) -> None:
        # 25 °C = 298.15 K = 77 °F (exacto)
        assert convert_from_si(298.15, "temperature", "Inglés") == pytest.approx(77.0, abs=1e-9)

    def test_parse_fahrenheit_to_kelvin(self) -> None:
        # 32 °F → 273.15 K
        assert convert_to_si(32.0, "temperature", "Inglés") == pytest.approx(273.15, abs=1e-9)

    # --- Presión ---

    def test_atmospheric_bar(self) -> None:
        # 101325 Pa = 1.01325 bar
        assert convert_from_si(101325.0, "pressure", "Técnico") == pytest.approx(1.01325, abs=1e-12)

    def test_atmospheric_psia(self) -> None:
        # 101325 Pa ≈ 14.6959487755 psia (NIST factor)
        psia = convert_from_si(101325.0, "pressure", "Inglés")
        assert psia == pytest.approx(14.6959487755, rel=1e-9)

    # --- Entalpía específica ---

    def test_btu_per_lb_exact(self) -> None:
        # 2326 J/kg = 1 Btu/lb exacto
        assert convert_from_si(2326.0, "specific_enthalpy", "Inglés") == pytest.approx(
            1.0, abs=1e-12
        )

    def test_one_btu_per_lb_to_si(self) -> None:
        assert convert_to_si(1.0, "specific_enthalpy", "Inglés") == pytest.approx(2326.0, abs=1e-9)

    def test_kj_per_kg_simple(self) -> None:
        # 2700e3 J/kg = 2700 kJ/kg
        assert convert_from_si(2.7e6, "specific_enthalpy", "Técnico") == pytest.approx(
            2700.0, abs=1e-12
        )

    # --- Entropía específica ---

    def test_btu_per_lb_R_exact(self) -> None:
        # 4186.8 J/(kg·K) = 1 Btu/(lb·°R) exacto
        assert convert_from_si(4186.8, "specific_entropy", "Inglés") == pytest.approx(
            1.0, abs=1e-12
        )

    def test_kj_per_kg_K_simple(self) -> None:
        assert convert_from_si(5000.0, "specific_entropy", "Técnico") == pytest.approx(
            5.0, abs=1e-12
        )

    # --- Volumen específico ---

    def test_ft3_per_lb_conversion(self) -> None:
        # 1 m³/kg ≈ 16.0184633739537 ft³/lb (NIST: 1 ft³ = 0.028316846592 m³)
        v_imperial = convert_from_si(1.0, "specific_volume", "Inglés")
        assert v_imperial == pytest.approx(16.0184633739537, rel=1e-12)

    def test_ft3_per_lb_inverse(self) -> None:
        v_si = convert_to_si(1.0, "specific_volume", "Inglés")
        assert v_si == pytest.approx(0.0624279605761459, rel=1e-12)

    # --- Diferencia de temperatura (vademecum §1.3: sin offset) ---

    def test_delta_t_celsius_equals_kelvin(self) -> None:
        assert convert_from_si(10.0, "temperature_difference", "Técnico") == pytest.approx(10.0)

    def test_delta_t_fahrenheit_has_no_offset(self) -> None:
        # ΔT = 10 K → Δt = 18 °F (no 50 °F: una diferencia no lleva el +32).
        assert convert_from_si(10.0, "temperature_difference", "Inglés") == pytest.approx(18.0)

    # --- Densidad ---

    def test_water_density_in_lb_per_ft3(self) -> None:
        # 1000 kg/m³ ≈ 62.428 lb/ft³ (densidad del agua en tablas inglesas).
        rho = convert_from_si(1000.0, "density", "Inglés")
        assert rho == pytest.approx(62.42796, rel=1e-6)

    # --- Velocidad ---

    def test_one_ft_per_s(self) -> None:
        assert convert_to_si(1.0, "speed", "Inglés") == pytest.approx(0.3048, rel=1e-12)

    # --- Transporte ---

    def test_one_lb_per_ft_s_in_pa_s(self) -> None:
        # 1 lb/(ft·s) = 0.45359237 kg / 0.3048 m / s = 1.488164 Pa·s.
        mu_si = convert_to_si(1.0, "dynamic_viscosity", "Inglés")
        assert mu_si == pytest.approx(1.4881639435695, rel=1e-12)

    def test_one_btu_per_h_ft_f_in_w_per_m_k(self) -> None:
        # 1 Btu_IT/(h·ft·°F) = 1.730735 W/(m·K) (Incropera, tabla de conversión).
        k_si = convert_to_si(1.0, "thermal_conductivity", "Inglés")
        assert k_si == pytest.approx(1.730734666, rel=1e-9)

    def test_one_ft2_per_s_in_m2_per_s(self) -> None:
        assert convert_to_si(1.0, "diffusivity", "Inglés") == pytest.approx(0.09290304, rel=1e-12)

    def test_mass_flow_and_power_are_coherent_with_specific_energy(self) -> None:
        # Potencia = caudal · energía específica sin factores en cada sistema:
        # 1 lb/s · 1 Btu/lb = 1 Btu/s (= 1055.06 W) y 1 kg/s · 1 kJ/kg = 1 kW.
        for system in SUPPORTED_SYSTEMS:
            m = convert_to_si(1.0, "mass_flow", system)
            w = convert_to_si(1.0, "specific_enthalpy", system)
            assert convert_from_si(m * w, "power", system) == pytest.approx(1.0, rel=1e-12)
        assert convert_to_si(1.0, "power", "Inglés") == pytest.approx(1055.05585262, rel=1e-12)


class TestFormatQuantity:
    def test_includes_unit_label_tecnico(self) -> None:
        out = format_quantity(298.15, "temperature", "Técnico")
        assert "°C" in out
        assert "25" in out

    def test_includes_unit_label_si(self) -> None:
        out = format_quantity(298.15, "temperature", "SI")
        assert "K" in out
        assert "298" in out

    def test_includes_unit_label_ingles(self) -> None:
        out = format_quantity(298.15, "temperature", "Inglés")
        assert "°F" in out
        assert "77" in out

    def test_precision_default_is_4_sig_figs(self) -> None:
        # 2700.5 kJ/kg con precision=4 → "2700"
        out = format_quantity(2.7005e6, "specific_enthalpy", "Técnico")
        # `.4g` formato: 2701 (redondea por banker's o half-up)
        assert re.match(r"\d{4}\s*kJ/kg", out)

    def test_precision_custom(self) -> None:
        out = format_quantity(298.15, "temperature", "Técnico", precision=6)
        # 25.0000 °C — formato `.6g` → "25"
        assert "°C" in out

    def test_pressure_atmospheric_tecnico(self) -> None:
        out = format_quantity(101325.0, "pressure", "Técnico")
        assert "bar" in out
        # 1.01325 bar, con precision=4 → "1.013"
        assert "1.013" in out


class TestParseUserInput:
    """``parse_user_input`` es alias de ``convert_to_si``."""

    @pytest.mark.parametrize("kind", _KINDS)
    @pytest.mark.parametrize("system", SUPPORTED_SYSTEMS)
    def test_alias_of_convert_to_si(self, kind: str, system: str) -> None:
        value = 1.234
        a = parse_user_input(value, kind, system)  # type: ignore[arg-type]
        b = convert_to_si(value, kind, system)  # type: ignore[arg-type]
        assert a == b


class TestEdgeCases:
    def test_temperature_zero_kelvin(self) -> None:
        assert convert_from_si(0.0, "temperature", "Inglés") == pytest.approx(-459.67)
        assert convert_to_si(-459.67, "temperature", "Inglés") == pytest.approx(0.0, abs=1e-9)

    def test_pressure_zero(self) -> None:
        for sys in SUPPORTED_SYSTEMS:
            assert convert_from_si(0.0, "pressure", sys) == 0.0
            assert convert_to_si(0.0, "pressure", sys) == 0.0

    def test_specific_enthalpy_zero(self) -> None:
        for sys in SUPPORTED_SYSTEMS:
            assert convert_from_si(0.0, "specific_enthalpy", sys) == 0.0

    def test_large_pressure(self) -> None:
        # 100 MPa = 1e8 Pa
        p_bar = convert_from_si(1.0e8, "pressure", "Técnico")
        assert p_bar == pytest.approx(1000.0, abs=1e-9)

    def test_negative_temperature_celsius(self) -> None:
        # -40 °C → 233.15 K
        assert convert_to_si(-40.0, "temperature", "Técnico") == pytest.approx(233.15, abs=1e-9)

    def test_minus_40_intersection(self) -> None:
        # -40 °C = -40 °F (punto físico clásico)
        assert convert_to_si(-40.0, "temperature", "Inglés") == pytest.approx(233.15, abs=1e-9)


class TestInvalidArgs:
    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="QuantityKind"):
            unit_label("not_a_kind", "SI")  # type: ignore[arg-type]

    def test_unknown_system_raises(self) -> None:
        with pytest.raises(ValueError, match="UnitSystem"):
            unit_label("temperature", "Métrico")  # type: ignore[arg-type]

    def test_convert_with_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="QuantityKind"):
            convert_from_si(1.0, "torque", "SI")  # type: ignore[arg-type]


def test_molar_units_follow_the_tables() -> None:
    """kJ/kmol = J/mol; 1 Btu/lbmol = 2,326 kJ/kmol (Cengel A-26E: s° del CO₂ = 51,07)."""
    assert convert_from_si(-393_520.0, "molar_enthalpy", "Técnico") == -393_520.0
    assert convert_from_si(-393_520.0, "molar_enthalpy", "Inglés") == pytest.approx(
        -169_200, rel=1e-3
    )
    # s° del CO₂: 213,80 kJ/(kmol·K) = 51,07 Btu/(lbmol·°R) (A-26E)
    assert convert_from_si(213.80, "molar_entropy", "Inglés") == pytest.approx(51.07, rel=1e-3)


def test_mass_and_energy_are_coherent_with_specific_energy() -> None:
    """kg · kJ/kg = kJ y lb · Btu/lb = Btu, sin factores (como caudal · trabajo = potencia)."""
    for system in SUPPORTED_SYSTEMS:
        m, e = 2.5, 1.2e5
        product = convert_from_si(m, "mass", system) * convert_from_si(
            e, "specific_enthalpy", system
        )
        assert product == pytest.approx(convert_from_si(m * e, "energy", system), rel=1e-12)
    assert convert_from_si(0.3048, "length", "Inglés") == pytest.approx(1.0)


def test_heat_transfer_units_are_coherent() -> None:
    """h·A·ΔT = Q̇, q''·A = Q̇ y ΔT/R = Q̇ en cada sistema, sin factores (Fase 8)."""
    h, area, dT, R = 25.0, 3.0, 40.0, 0.12
    for system in SUPPORTED_SYSTEMS:
        Q = convert_from_si(h * area * dT, "heat_rate", system)
        product = (
            convert_from_si(h, "heat_transfer_coefficient", system)
            * convert_from_si(area, "area", system)
            * convert_from_si(dT, "temperature_difference", system)
        )
        assert product == pytest.approx(Q, rel=1e-12)
        flux = convert_from_si(h * dT, "heat_flux", system) * convert_from_si(area, "area", system)
        assert flux == pytest.approx(Q, rel=1e-12)
        ratio = convert_from_si(dT, "temperature_difference", system) / convert_from_si(
            R, "thermal_resistance", system
        )
        assert ratio == pytest.approx(convert_from_si(dT / R, "heat_rate", system), rel=1e-12)
        per_length = convert_from_si(h * dT * 0.5, "linear_heat_rate", system) * convert_from_si(
            2.0, "length", system
        )
        assert per_length == pytest.approx(convert_from_si(h * dT, "heat_rate", system), rel=1e-12)
    # 1 Btu/(h·ft²·°F) = 5,678263 W/(m²·K) (Incropera, tabla de conversiones)
    assert 1.0 / convert_from_si(1.0, "heat_transfer_coefficient", "Inglés") == pytest.approx(
        5.678263, rel=1e-6
    )
    assert convert_from_si(0.0254, "small_length", "Inglés") == pytest.approx(1.0)
    assert convert_from_si(0.004, "small_length", "Técnico") == pytest.approx(4.0)


def test_heat_transfer_procedure_units_are_coherent() -> None:
    """Las sustituciones de los procedimientos de la Fase 8 cierran sin factores."""
    h, A, k, P, A_c = 40.0, 0.3, 180.0, 0.016, 2.0e-5
    g, beta, dT, L, nu = 9.80665, 3.1e-3, 50.0, 0.08, 1.75e-5
    f, D, rho, V = 0.03, 0.03, 992.0, 0.24
    for system in SUPPORTED_SYSTEMS:

        def c(x: float, kind: str, system: str = system) -> float:
            return convert_from_si(x, kind, system)  # type: ignore[arg-type]

        # NTU = h·A/(ṁ·c_p): h·A es una capacidad (W/K o Btu/(h·°F))
        assert c(h, "heat_transfer_coefficient") * c(A, "area") == pytest.approx(
            c(h * A, "heat_capacity_rate"), rel=1e-12
        )
        assert c(h * A, "heat_capacity_rate") * c(dT, "temperature_difference") == pytest.approx(
            c(h * A * dT, "heat_rate"), rel=1e-12
        )
        # m = √(h·P/(k·A_c)) de una aleta
        m = math.sqrt(h * P / (k * A_c))
        m_sys = math.sqrt(
            c(h, "heat_transfer_coefficient")
            * c(P, "length")
            / (c(k, "thermal_conductivity") * c(A_c, "area"))
        )
        assert m_sys == pytest.approx(c(m, "inverse_length"), rel=1e-12)
        # Gr = g·β·ΔT·L³/ν² es el mismo número en los tres sistemas
        Gr = g * beta * dT * L**3 / nu**2
        Gr_sys = (
            c(g, "acceleration")
            * c(beta, "expansion_coefficient")
            * c(dT, "temperature_difference")
            * c(L, "length") ** 3
            / c(nu, "diffusivity") ** 2
        )
        assert Gr_sys == pytest.approx(Gr, rel=1e-12)
        # Δp = f·(L/D)·ρ·V²/2 (en el Inglés, dividido por g_c = 32,174 lbm·ft/(lbf·s²))
        dp = f * (5.0 / D) * rho * V**2 / 2.0
        g_c = 32.17404855643 if system == "Inglés" else 1.0
        dp_sys = f * (5.0 / D) * c(rho, "density") * c(V, "speed") ** 2 / (2.0 * g_c)
        assert dp_sys == pytest.approx(c(dp, "pressure_drop"), rel=1e-9)


def test_radiation_and_exchanger_units_are_coherent() -> None:
    """Fase 8.2: σ·T⁴, E_bλ·Δλ, λ·T, R''_f = 1/h y T·Ṡ cierran sin factores."""
    sigma = 5.670374419e-8
    T, lam, R_f, h, S, dT = 800.0, 3.0e-6, 2.0e-4, 1200.0, 35.0, 12.0
    for system in SUPPORTED_SYSTEMS:

        def c(x: float, kind: str, system: str = system) -> float:
            return convert_from_si(x, kind, system)  # type: ignore[arg-type]

        # σ en el sistema: q'' = σ_sys·T_sys⁴
        sigma_sys = c(sigma, "heat_flux") / c(1.0, "absolute_temperature") ** 4
        assert sigma_sys * c(T, "absolute_temperature") ** 4 == pytest.approx(
            c(sigma * T**4, "heat_flux"), rel=1e-12
        )
        # E_bλ [por μm] · Δλ [μm] = q''
        assert c(5.0e9, "spectral_emissive_power") * c(lam, "wavelength") == pytest.approx(
            c(5.0e9 * lam, "heat_flux"), rel=1e-12
        )
        # λ·T con λ en μm y T absoluta
        assert c(lam, "wavelength") * c(T, "absolute_temperature") == pytest.approx(
            c(lam * T, "wavelength_temperature"), rel=1e-12
        )
        # R''_f se suma con 1/h sin factores (U = 1/(1/h + R''_f))
        assert c(1.0 / h, "fouling_resistance") * c(h, "heat_transfer_coefficient") == (
            pytest.approx(1.0, rel=1e-12)
        )
        U_sys = 1.0 / (1.0 / c(h, "heat_transfer_coefficient") + c(R_f, "fouling_resistance"))
        assert U_sys == pytest.approx(
            c(1.0 / (1.0 / h + R_f), "heat_transfer_coefficient"), rel=1e-12
        )
        # T₀·Ṡ_gen = Ẋ_dest (W o Btu/h)
        assert c(T, "absolute_temperature") * c(S, "entropy_rate") == pytest.approx(
            c(T * S, "heat_rate"), rel=1e-12
        )
        # ΔT con °R o °F es lo mismo: Ṡ·ΔT = Q̇
        assert c(S, "entropy_rate") * c(dT, "temperature_difference") == pytest.approx(
            c(S * dT, "heat_rate"), rel=1e-12
        )
    # σ = 0,1714·10⁻⁸ Btu/(h·ft²·°R⁴) (Cengel y Ghajar, tabla 1-1)
    sigma_e = convert_from_si(sigma, "heat_flux", "Inglés") / 1.8**4
    assert sigma_e == pytest.approx(0.1714e-8, rel=2e-3)
    # λ·T del máximo (Wien): 2897,8 μm·K = 5216,0 μm·°R
    assert convert_from_si(2.897771955e-3, "wavelength_temperature", "Inglés") == pytest.approx(
        5216.0, abs=0.1
    )


def test_ideal_gas_units_are_coherent() -> None:
    """Fase 9: cada sustitución de la página de gases ideales cierra sin factores."""
    for system in SUPPORTED_SYSTEMS:

        def si(value: float, kind: str) -> float:
            return convert_to_si(value, kind, system)  # type: ignore[arg-type]

        def back(value: float, kind: str) -> float:
            return convert_from_si(value, kind, system)  # type: ignore[arg-type]

        # m·v = V, m·s = S, ṁ·s = Ṡ y n·M = m
        assert back(si(1.0, "mass") * si(1.0, "specific_volume"), "volume") == pytest.approx(1.0)
        assert back(si(1.0, "mass") * si(1.0, "specific_entropy"), "entropy") == pytest.approx(1.0)
        assert back(si(1.0, "mass_flow") * si(1.0, "specific_entropy"), "entropy_flow") == (
            pytest.approx(1.0)
        )
        assert back(si(1.0, "amount") * si(1.0, "molar_mass"), "mass") == pytest.approx(1.0)
        # T₀·S = X (energía) y T₀·Ṡ = Ẋ (potencia), con T₀ absoluta
        t0 = si(1.0, "absolute_temperature")
        assert back(t0 * si(1.0, "entropy"), "energy") == pytest.approx(1.0)
        assert back(t0 * si(1.0, "entropy_flow"), "power") == pytest.approx(1.0)
        # R = R_u/M y p·V = n·R_u·T
        r = si(1.0, "molar_entropy") / si(1.0, "molar_mass")
        assert back(r, "specific_entropy") == pytest.approx(1.0)
        pv = si(1.0, "pressure") * si(1.0, "volume")
        nrt = si(1.0, "amount") * si(1.0, "molar_entropy") * t0
        # p·V en las unidades de presión y volumen de cada sistema, contra n·R_u·T en energía
        assert back(pv, "energy") / back(nrt, "energy") == pytest.approx(pv / nrt)
    # R_u en cada sistema: 8,314 J/(mol·K) = 8,314 kJ/(kmol·K) = 1,98588 Btu/(lbmol·°R)
    assert convert_from_si(8.314462618, "molar_entropy", "Inglés") == pytest.approx(
        1.98588, rel=1e-5
    )
    assert convert_from_si(0.028966, "molar_mass", "Técnico") == pytest.approx(28.966)
