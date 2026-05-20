"""Tests de :mod:`core.units` — conversiones y normalizadores.

Cubre las funciones de conversión simples (heredadas de Fase 0) y los
normalizadores tolerantes de Fase 1.2. Especialmente importante para
Fase 1.2: el normalizador de magnitudes específicas debe aceptar las
muchas variantes de notación que un alumno puede pegar al copiar
tablas de libros, papers o web.
"""

from __future__ import annotations

import pytest

from core.units import (
    bar_to_pa,
    c_to_k,
    j_to_kj,
    k_to_c,
    kj_to_j,
    normalize_pressure_unit,
    normalize_specific_unit,
    normalize_temperature_unit,
    pa_to_bar,
    parse_header,
    pressure_to_pascal,
    specific_from_si,
    temperature_to_kelvin,
)


class TestSimpleConversions:
    def test_celsius_round_trip(self) -> None:
        assert k_to_c(c_to_k(25.0)) == pytest.approx(25.0)

    def test_celsius_to_kelvin(self) -> None:
        assert c_to_k(0.0) == pytest.approx(273.15)
        assert c_to_k(100.0) == pytest.approx(373.15)

    def test_bar_round_trip(self) -> None:
        assert pa_to_bar(bar_to_pa(2.5)) == pytest.approx(2.5)

    def test_bar_to_pascal(self) -> None:
        assert bar_to_pa(1.01325) == pytest.approx(101325.0)

    def test_kj_round_trip(self) -> None:
        assert j_to_kj(kj_to_j(419.06)) == pytest.approx(419.06)


class TestNormalizeTemperatureUnit:
    @pytest.mark.parametrize(
        "u,expected",
        [
            ("", "C"),
            ("°C", "C"),
            ("°c", "C"),
            ("ºC", "C"),
            ("C", "C"),
            ("c", "C"),
            ("Celsius", "C"),
            ("celsius", "C"),
            ("deg C", "C"),
            ("degC", "C"),
            ("K", "K"),
            ("k", "K"),
            ("Kelvin", "K"),
            ("F", None),
            ("°F", None),
            ("Rankine", None),
        ],
    )
    def test_recognized(self, u: str, expected: str | None) -> None:
        assert normalize_temperature_unit(u) == expected


class TestTemperatureToKelvin:
    def test_celsius(self) -> None:
        assert temperature_to_kelvin(100.0, "°C") == pytest.approx(373.15)

    def test_kelvin_passthrough(self) -> None:
        assert temperature_to_kelvin(373.15, "K") == pytest.approx(373.15)

    def test_unsupported_returns_none(self) -> None:
        assert temperature_to_kelvin(100.0, "°F") is None


class TestNormalizePressureUnit:
    @pytest.mark.parametrize(
        "u,expected",
        [
            ("", "bar"),
            ("bar", "bar"),
            ("Bar", "bar"),
            ("BAR", "bar"),
            ("bar(a)", "bar"),
            ("bara", "bar"),
            ("Pa", "Pa"),
            ("pa", "Pa"),
            ("kPa", "kPa"),
            ("kpa", "kPa"),
            ("MPa", "MPa"),
            ("mpa", "MPa"),
            ("psi", None),
            ("atm", None),
        ],
    )
    def test_recognized(self, u: str, expected: str | None) -> None:
        assert normalize_pressure_unit(u) == expected


class TestPressureToPascal:
    def test_bar(self) -> None:
        assert pressure_to_pascal(1.0, "bar") == pytest.approx(1.0e5)

    def test_atmospheric_kpa(self) -> None:
        assert pressure_to_pascal(101.325, "kPa") == pytest.approx(101325.0)

    def test_mpa(self) -> None:
        assert pressure_to_pascal(0.1, "MPa") == pytest.approx(1.0e5)

    def test_pa_passthrough(self) -> None:
        assert pressure_to_pascal(1.0e5, "Pa") == pytest.approx(1.0e5)

    def test_unsupported(self) -> None:
        assert pressure_to_pascal(1.0, "psi") is None


class TestNormalizeSpecificUnit:
    """Cobertura amplia: kJ/(kg·K) tiene muchas variantes en uso real."""

    @pytest.mark.parametrize(
        "u",
        [
            "kJ/kg",
            "kj/kg",
            "KJ/kg",
            "kJ kg^-1",
            "kJ/(kg·K)",
            "kJ/(kg K)",
            "kJ/(kg.K)",
            "kJ/kg-K",
            "kJ/kg/K",
            "kJ kg^-1 K^-1",
            "kJ/(kgK)",
            "kJ/kgK",
            "kJ/(kg*K)",
        ],
    )
    def test_kilo_variants(self, u: str) -> None:
        assert normalize_specific_unit(u) == "kilo"

    @pytest.mark.parametrize(
        "u",
        [
            "J/kg",
            "j/kg",
            "J kg^-1",
            "J/(kg·K)",
            "J/(kg K)",
            "J/(kg.K)",
            "J/kg-K",
            "J/kg/K",
            "J kg^-1 K^-1",
        ],
    )
    def test_si_variants(self, u: str) -> None:
        assert normalize_specific_unit(u) == "si"

    def test_empty_defaults_to_kilo(self) -> None:
        assert normalize_specific_unit("") == "kilo"

    @pytest.mark.parametrize(
        "u",
        [
            "btu/lb",
            "cal/g",
            "kg/J",
            "xyz",
        ],
    )
    def test_unsupported(self, u: str) -> None:
        assert normalize_specific_unit(u) is None


class TestSpecificFromSi:
    def test_kilo(self) -> None:
        assert specific_from_si(1000.0, "kJ/kg") == pytest.approx(1.0)

    def test_si_passthrough(self) -> None:
        assert specific_from_si(1000.0, "J/kg") == pytest.approx(1000.0)

    def test_middle_dot(self) -> None:
        assert specific_from_si(7000.0, "kJ/(kg·K)") == pytest.approx(7.0)

    def test_kilo_with_dash(self) -> None:
        """Regresión item 1 Fase 1.2: aceptar kJ/kg-K."""
        assert specific_from_si(1000.0, "kJ/kg-K") == pytest.approx(1.0)

    def test_kilo_with_exponent(self) -> None:
        assert specific_from_si(1000.0, "kJ kg^-1 K^-1") == pytest.approx(1.0)

    def test_unsupported(self) -> None:
        assert specific_from_si(1000.0, "btu/lb") is None


class TestParseHeader:
    @pytest.mark.parametrize(
        "header,base,unit",
        [
            ("T [°C]", "t", "°C"),
            ("T [K]", "t", "K"),
            ("P [bar]", "p", "bar"),
            ("h [kJ/kg]", "h", "kJ/kg"),
            ("h_f [kJ/kg]", "hf", "kJ/kg"),
            ("s [kJ/(kg·K)]", "s", "kJ/(kg·K)"),
            ("s_g [J/(kg·K)]", "sg", "J/(kg·K)"),
            ("T", "t", ""),
            ("Temperature (Celsius)", "temperature", "Celsius"),
            ("h (kJ kg^-1)", "h", "kJ kg^-1"),
        ],
    )
    def test_parse(self, header: str, base: str, unit: str) -> None:
        assert parse_header(header) == (base, unit)

    def test_strips_underscore_in_base(self) -> None:
        base, _ = parse_header("h_f [kJ/kg]")
        assert base == "hf"

    def test_blank_header(self) -> None:
        base, unit = parse_header("   ")
        assert base == ""
        assert unit == ""


class TestEndToEndFlowForBilinearBugFromPhase11:
    """Simula el flujo de la página: parsear header, normalizar unidad,
    convertir desde SI. Cubre el caso que reportó el bug."""

    def test_s_kilo_kgk(self) -> None:
        base, unit = parse_header("s [kJ/(kg·K)]")
        assert base == "s"
        # CoolProp devuelve s en J/(kg·K) ≈ 7000 a vapor sobrecalentado típico.
        si_value = 7000.0
        converted = specific_from_si(si_value, unit)
        assert converted == pytest.approx(7.0)

    def test_s_with_dash_notation(self) -> None:
        base, unit = parse_header("s [kJ/kg-K]")
        assert base == "s"
        assert specific_from_si(7000.0, unit) == pytest.approx(7.0)

    def test_h_kg_inverse_notation(self) -> None:
        base, unit = parse_header("h [kJ kg^-1]")
        assert base == "h"
        assert specific_from_si(3000_000.0, unit) == pytest.approx(3000.0)
