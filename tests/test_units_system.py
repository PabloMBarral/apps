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
