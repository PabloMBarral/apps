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
    def state(self) -> StatePoint:
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
