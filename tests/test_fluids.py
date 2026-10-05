"""Tests mínimos de :mod:`core.fluids` — Fase 0.

Casos de regresión elegidos lejos de la curva de saturación (donde CoolProp
con par ``PT`` puede devolver resultados ambiguos) y un punto sobre la
saturación accedido por ``PX`` para chequear la temperatura de saturación.
"""

from __future__ import annotations

import math

import pytest

from core.fluids import StatePoint, state_from_pair

WATER = "Water"


class TestWaterSubcooledAt25C1Bar:
    """Líquido subenfriado a 25 °C y 1 bar.

    Valores de referencia (IAPWS-IF97 vía CoolProp):
    h ≈ 104.92 kJ/kg, s ≈ 0.3672 kJ/(kg·K), x = -1 (fuera de la campana).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def state(cls) -> StatePoint:
        return state_from_pair(WATER, "TP", t=298.15, p=1.0e5)

    def test_returns_state_point(self, state: StatePoint) -> None:
        assert isinstance(state, StatePoint)

    def test_temperature_in_kelvin(self, state: StatePoint) -> None:
        assert math.isclose(state.T_K, 298.15, abs_tol=1e-6)

    def test_pressure_in_pascal(self, state: StatePoint) -> None:
        assert math.isclose(state.P_Pa, 1.0e5, rel_tol=1e-6)

    def test_enthalpy(self, state: StatePoint) -> None:
        assert math.isclose(state.h_J_per_kg, 104_920.0, rel_tol=5e-3)

    def test_entropy(self, state: StatePoint) -> None:
        assert math.isclose(state.s_J_per_kg_K, 367.2, rel_tol=5e-3)

    def test_quality_is_outside_two_phase(self, state: StatePoint) -> None:
        assert state.x == pytest.approx(-1.0)


class TestWaterSaturatedVaporAtAtmospheric:
    """Vapor saturado seco a 1.01325 bar — T_sat debe ≈ 100 °C."""

    def test_saturation_temperature(self) -> None:
        state = state_from_pair(WATER, "PX", p=101_325.0, x=1.0)
        # T_sat(1 atm) ≈ 373.124 K según IAPWS-IF97.
        assert math.isclose(state.T_K, 373.124, abs_tol=0.5)

    def test_quality_is_one(self) -> None:
        state = state_from_pair(WATER, "PX", p=101_325.0, x=1.0)
        assert state.x == pytest.approx(1.0)


class TestRoundTripTPtoPH:
    """Round-trip TP → PH: recuperar (T, P) a partir de (P, h)."""

    def test_roundtrip_superheated_steam(self) -> None:
        # 250 °C, 5 bar — vapor sobrecalentado, lejos de la saturación.
        initial = state_from_pair(WATER, "TP", t=523.15, p=5.0e5)
        recovered = state_from_pair(WATER, "PH", p=initial.P_Pa, h=initial.h_J_per_kg)
        assert math.isclose(recovered.T_K, initial.T_K, abs_tol=1e-2)
        assert math.isclose(recovered.P_Pa, initial.P_Pa, rel_tol=1e-9)


class TestInvalidInputsRaiseValueError:
    """Inputs inválidos deben generar mensajes claros para el alumno."""

    def test_negative_pressure(self) -> None:
        with pytest.raises(ValueError, match="presión"):
            state_from_pair(WATER, "TP", t=298.15, p=-1.0e5)

    def test_missing_kwarg(self) -> None:
        with pytest.raises(ValueError, match="requiere argumentos"):
            state_from_pair(WATER, "TP", t=298.15)

    def test_unsupported_pair(self) -> None:
        with pytest.raises(ValueError, match="no soportado"):
            state_from_pair(WATER, "XY", t=298.15, p=1.0e5)  # type: ignore[arg-type]

    def test_quality_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="título"):
            state_from_pair(WATER, "PX", p=1.0e5, x=1.5)


class TestSpecificVolume:
    """``state_from_pair`` completa v = 1/ρ (hace falta para el p–log v)."""

    def test_compressed_liquid_25C_1bar(self) -> None:
        # Cengel & Boles, Tabla A-4: v_f(25 °C) = 0.001003 m³/kg.
        state = state_from_pair(WATER, "TP", t=298.15, p=1.0e5)
        assert state.v_m3_per_kg == pytest.approx(0.001003, rel=1e-3)

    def test_superheated_1MPa_300C(self) -> None:
        # Cengel & Boles, Tabla A-6: 1 MPa, 300 °C → v = 0.25799 m³/kg.
        state = state_from_pair(WATER, "TP", t=573.15, p=1.0e6)
        assert state.v_m3_per_kg == pytest.approx(0.25799, rel=1e-4)

    def test_wet_steam_follows_lever_rule(self) -> None:
        # Cengel & Boles, Tabla A-5 a 100 kPa: v_f = 0.001043, v_g = 1.6941.
        state = state_from_pair(WATER, "PX", p=1.0e5, x=0.5)
        expected = 0.001043 + 0.5 * (1.6941 - 0.001043)
        assert state.v_m3_per_kg == pytest.approx(expected, rel=1e-3)

    def test_manual_state_point_defaults_to_none(self) -> None:
        # Compatibilidad: el código que arma StatePoint a mano sigue andando.
        sp = StatePoint(T_K=300.0, P_Pa=1.0e5, h_J_per_kg=0.0, s_J_per_kg_K=0.0, x=-1.0)
        assert sp.v_m3_per_kg is None


class TestQualitySentinel:
    """CoolProp devuelve x = 10000 para estados monofásicos resueltos con h-s;
    ``state_from_pair`` lo normaliza a la convención documentada (-1)."""

    def test_hs_superheated_reports_minus_one(self) -> None:
        state = state_from_pair(WATER, "HS", h=3.0e6, s=7000.0)
        assert state.x == -1.0

    def test_hs_wet_steam_keeps_quality(self) -> None:
        state = state_from_pair(WATER, "HS", h=2.0e6, s=5000.0)
        assert 0.0 < state.x < 1.0

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (-1.0, -1.0),
            (10000.0, -1.0),
            (0.0, 0.0),
            (1.0, 1.0),
            (0.37, 0.37),
            (-1e-12, 0.0),
            (1.0 + 1e-12, 1.0),
        ],
    )
    def test_normalize_quality(self, raw: float, expected: float) -> None:
        from core.fluids import _normalize_quality

        assert _normalize_quality(raw) == pytest.approx(expected)
