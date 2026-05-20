"""Tests de :mod:`core.isentropic` — Fase 1.3.

Casos de referencia inspirados en ejemplos clásicos de Cengel (8va ed.),
verificados contra CoolProp para que no dependan de los valores tabulados
exactos del libro (que pueden diferir ligeramente de IAPWS-IF97).
"""

from __future__ import annotations

import pytest

from core.fluids import state_from_pair
from core.isentropic import (
    IsentropicResult,
    compressor_direct,
    compressor_inverse,
    compressor_multistage,
    pump_direct,
    pump_inverse,
    turbine_direct,
    turbine_inverse,
)

# ---------------------------------------------------------------------
# Turbina
# ---------------------------------------------------------------------


class TestTurbineDirect:
    """Turbina de vapor (similar a Cengel Ej. 7-12).

    Estado 1: vapor sobrecalentado a 3 MPa, 400 °C.
    Estado 2: P = 50 kPa, η_s = 0.90 → mezcla húmeda.
    Esperado (orden de magnitud Cengel): w_t ≈ 700-800 kJ/kg.
    """

    def test_returns_result_with_eta_s(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        result = turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.90)
        assert isinstance(result, IsentropicResult)
        assert result.device == "turbine"
        assert result.eta_s == 0.90

    def test_work_in_expected_range(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        result = turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.90)
        w_t_kj = -result.delta_h_real_J_per_kg / 1e3
        assert 700.0 < w_t_kj < 800.0

    def test_real_work_less_than_isentropic(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        result = turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.90)
        # Trabajo neto = -Δh (mayor cuanto más negativo es Δh).
        # Real < isentrópico → |Δh_real| < |Δh_isen|.
        assert abs(result.delta_h_real_J_per_kg) < abs(result.delta_h_isen_J_per_kg)

    def test_outlet_pressure_matches(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        result = turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.90)
        assert result.state_out_real.P_Pa == pytest.approx(5.0e4, rel=1e-6)
        assert result.state_out_isen.P_Pa == pytest.approx(5.0e4, rel=1e-6)

    def test_isentropic_outlet_has_same_entropy_as_inlet(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        result = turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.90)
        assert result.state_out_isen.s_J_per_kg_K == pytest.approx(state_in.s_J_per_kg_K, rel=1e-6)

    def test_steps_contain_latex_and_narrative(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        result = turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.90)
        assert "h_1" in result.steps.formula_latex
        assert "eta_s" in result.steps.formula_latex
        assert "turbina" in result.steps.narrative_es.lower()


class TestTurbineReversible:
    def test_eta_one_recovers_isentropic_outlet(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        result = turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=1.0)
        assert result.state_out_real.h_J_per_kg == pytest.approx(
            result.state_out_isen.h_J_per_kg, rel=1e-9
        )
        assert result.delta_h_real_J_per_kg == pytest.approx(result.delta_h_isen_J_per_kg, rel=1e-9)


class TestTurbineInverse:
    def test_round_trip_recovers_eta_s(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        direct = turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.88)
        inverse = turbine_inverse(
            fluid="Water", state_in=state_in, state_out_real=direct.state_out_real
        )
        assert inverse.eta_s == pytest.approx(0.88, rel=1e-4)
        assert inverse.device == "turbine"


class TestTurbineValidations:
    def test_p_out_equal_to_p_in_raises(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        with pytest.raises(ValueError, match="expande"):
            turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=3.0e6, eta_s=0.9)

    def test_p_out_higher_than_p_in_raises(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        with pytest.raises(ValueError, match="expande"):
            turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e6, eta_s=0.9)

    def test_eta_zero_raises(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        with pytest.raises(ValueError, match="rendimiento isoentrópico"):
            turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.0)

    def test_eta_above_one_raises(self) -> None:
        state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
        with pytest.raises(ValueError, match="rendimiento isoentrópico"):
            turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=1.5)


# ---------------------------------------------------------------------
# Compresor
# ---------------------------------------------------------------------


class TestCompressorDirect:
    """Compresor de aire (similar a Cengel Ej. 7-13/7-14).

    Aire ideal a 100 kPa, 25 °C → 800 kPa con η_s = 0.80.
    Esperado: T_2s ≈ 540 K, T_2 ≈ 600-640 K.
    """

    def test_outlet_temperature_in_range(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        result = compressor_direct(fluid="Air", state_in=state_in, p_out_Pa=8.0e5, eta_s=0.80)
        assert 520.0 < result.state_out_isen.T_K < 560.0
        assert 590.0 < result.state_out_real.T_K < 660.0

    def test_real_work_more_than_isentropic(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        result = compressor_direct(fluid="Air", state_in=state_in, p_out_Pa=8.0e5, eta_s=0.80)
        # Compresor: Δh positivo (consume trabajo). Real > isentrópico.
        assert result.delta_h_real_J_per_kg > result.delta_h_isen_J_per_kg
        assert result.delta_h_real_J_per_kg > 0
        assert result.delta_h_isen_J_per_kg > 0

    def test_reversible_recovers_isentropic(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        result = compressor_direct(fluid="Air", state_in=state_in, p_out_Pa=8.0e5, eta_s=1.0)
        assert result.state_out_real.h_J_per_kg == pytest.approx(
            result.state_out_isen.h_J_per_kg, rel=1e-9
        )


class TestCompressorInverse:
    def test_round_trip(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        direct = compressor_direct(fluid="Air", state_in=state_in, p_out_Pa=8.0e5, eta_s=0.80)
        inverse = compressor_inverse(
            fluid="Air", state_in=state_in, state_out_real=direct.state_out_real
        )
        assert inverse.eta_s == pytest.approx(0.80, rel=1e-4)


class TestCompressorValidations:
    def test_p_out_lower_than_p_in_raises(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=8.0e5)
        with pytest.raises(ValueError, match="comprime"):
            compressor_direct(fluid="Air", state_in=state_in, p_out_Pa=1.0e5, eta_s=0.80)


# ---------------------------------------------------------------------
# Bomba
# ---------------------------------------------------------------------


class TestPumpDirect:
    """Bomba de agua (similar a Cengel Ej. 10-1, Rankine simple).

    Agua sat. líquida a 10 kPa → 15 MPa con η_s = 0.85.
    Esperado: w_p ≈ 17-19 kJ/kg.
    """

    def test_work_in_expected_range(self) -> None:
        state_in = state_from_pair("Water", "PX", p=1.0e4, x=0.0)
        result = pump_direct(fluid="Water", state_in=state_in, p_out_Pa=1.5e7, eta_s=0.85)
        w_p_kj = result.delta_h_real_J_per_kg / 1e3
        assert 16.0 < w_p_kj < 20.0

    def test_reversible_uses_less_work(self) -> None:
        state_in = state_from_pair("Water", "PX", p=1.0e4, x=0.0)
        eta85 = pump_direct(fluid="Water", state_in=state_in, p_out_Pa=1.5e7, eta_s=0.85)
        eta100 = pump_direct(fluid="Water", state_in=state_in, p_out_Pa=1.5e7, eta_s=1.0)
        assert eta100.delta_h_real_J_per_kg < eta85.delta_h_real_J_per_kg

    def test_subcooled_liquid_inlet_accepted(self) -> None:
        # Agua a 25 °C y 1 bar → líquido subenfriado, x = -1.
        state_in = state_from_pair("Water", "TP", t=25 + 273.15, p=1.0e5)
        result = pump_direct(fluid="Water", state_in=state_in, p_out_Pa=10.0e5, eta_s=0.85)
        assert result.device == "pump"


class TestPumpInverse:
    def test_round_trip(self) -> None:
        state_in = state_from_pair("Water", "PX", p=1.0e4, x=0.0)
        direct = pump_direct(fluid="Water", state_in=state_in, p_out_Pa=1.5e7, eta_s=0.85)
        inverse = pump_inverse(
            fluid="Water", state_in=state_in, state_out_real=direct.state_out_real
        )
        assert inverse.eta_s == pytest.approx(0.85, rel=1e-3)


class TestPumpValidations:
    def test_wet_steam_inlet_raises(self) -> None:
        # Mezcla bifásica con x = 0.5 → no es líquido.
        state_in = state_from_pair("Water", "PX", p=1.0e5, x=0.5)
        with pytest.raises(ValueError, match="líquido"):
            pump_direct(fluid="Water", state_in=state_in, p_out_Pa=10.0e5, eta_s=0.85)

    def test_superheated_vapor_inlet_raises(self) -> None:
        # Vapor sobrecalentado a 200 °C, 1 bar: x = -1 pero phase = "gas".
        state_in = state_from_pair("Water", "TP", t=200 + 273.15, p=1.0e5)
        with pytest.raises(ValueError, match="líquido"):
            pump_direct(fluid="Water", state_in=state_in, p_out_Pa=10.0e5, eta_s=0.85)

    def test_p_out_lower_than_p_in_raises(self) -> None:
        state_in = state_from_pair("Water", "PX", p=1.5e7, x=0.0)
        with pytest.raises(ValueError, match="comprime"):
            pump_direct(fluid="Water", state_in=state_in, p_out_Pa=1.0e4, eta_s=0.85)


# ---------------------------------------------------------------------
# Multietapa (politrópico)
# ---------------------------------------------------------------------


class TestCompressorMultistage:
    """Compresor de aire de 3 etapas, 1 bar → 27 bar, η_s = 0.85."""

    def test_three_stages_with_intercooler_uses_less_than_single(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        multi = compressor_multistage(
            fluid="Air",
            state_in=state_in,
            p_out_Pa=27.0e5,
            n_stages=3,
            eta_s_per_stage=0.85,
            intercool=True,
        )
        # Total con intercooler debe ser estrictamente menor que single stage.
        assert multi.total_delta_h_real_J_per_kg < multi.delta_h_single_stage_real_J_per_kg

    def test_three_stages_pressure_ratio_geometric(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        multi = compressor_multistage(
            fluid="Air",
            state_in=state_in,
            p_out_Pa=27.0e5,
            n_stages=3,
            eta_s_per_stage=0.85,
            intercool=True,
        )
        # Π = 27^(1/3) = 3.
        assert multi.pressure_ratio_per_stage == pytest.approx(3.0, rel=1e-6)

    def test_stage_count_and_cooled_flags(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        multi = compressor_multistage(
            fluid="Air",
            state_in=state_in,
            p_out_Pa=27.0e5,
            n_stages=3,
            eta_s_per_stage=0.85,
            intercool=True,
        )
        assert len(multi.stages) == 3
        assert multi.stages[0].cooled_after
        assert multi.stages[1].cooled_after
        # Última etapa nunca tiene intercooler aguas abajo.
        assert not multi.stages[2].cooled_after

    def test_no_intercooler_flag_propagates(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        multi = compressor_multistage(
            fluid="Air",
            state_in=state_in,
            p_out_Pa=27.0e5,
            n_stages=3,
            eta_s_per_stage=0.85,
            intercool=False,
        )
        for stage in multi.stages:
            assert not stage.cooled_after

    def test_one_stage_equivalent_to_single_compressor(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        multi = compressor_multistage(
            fluid="Air",
            state_in=state_in,
            p_out_Pa=8.0e5,
            n_stages=1,
            eta_s_per_stage=0.80,
            intercool=False,
        )
        direct = compressor_direct(fluid="Air", state_in=state_in, p_out_Pa=8.0e5, eta_s=0.80)
        assert multi.total_delta_h_real_J_per_kg == pytest.approx(
            direct.delta_h_real_J_per_kg, rel=1e-9
        )

    def test_outlet_pressure_exact_in_last_stage(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        multi = compressor_multistage(
            fluid="Air",
            state_in=state_in,
            p_out_Pa=27.0e5,
            n_stages=3,
            eta_s_per_stage=0.85,
            intercool=True,
        )
        # La última etapa debe alcanzar exactamente p_out_Pa.
        assert multi.stages[-1].p_out_Pa == pytest.approx(27.0e5, rel=1e-9)


class TestMultistageValidations:
    def test_zero_stages_raises(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=1.0e5)
        with pytest.raises(ValueError, match="n_stages"):
            compressor_multistage(
                fluid="Air",
                state_in=state_in,
                p_out_Pa=8.0e5,
                n_stages=0,
                eta_s_per_stage=0.85,
                intercool=False,
            )

    def test_p_out_lower_raises(self) -> None:
        state_in = state_from_pair("Air", "TP", t=25 + 273.15, p=8.0e5)
        with pytest.raises(ValueError, match="comprime"):
            compressor_multistage(
                fluid="Air",
                state_in=state_in,
                p_out_Pa=1.0e5,
                n_stages=3,
                eta_s_per_stage=0.85,
                intercool=False,
            )
