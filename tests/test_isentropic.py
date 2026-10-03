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


# ---------------------------------------------------------------------
# Fase 1.7 — validaciones nuevas, defaults por fluido, pasos por sistema
# ---------------------------------------------------------------------

from core.fluids import SUPPORTED_FLUIDS, fluid_state_from_pair  # noqa: E402
from core.isentropic import (  # noqa: E402
    isentropic_labeled_states,
    isentropic_steps,
    isentropic_to_dict,
    multistage_labeled_states,
    multistage_steps,
    multistage_to_dict,
    pump_incompressible_comparison,
    suggested_device_inputs,
    summary_csv,
)

_DIRECT = {"turbine": turbine_direct, "compressor": compressor_direct, "pump": pump_direct}
_INVERSE = {"turbine": turbine_inverse, "compressor": compressor_inverse, "pump": pump_inverse}


def _water_turbine() -> IsentropicResult:
    state_in = state_from_pair("Water", "TP", t=400 + 273.15, p=3.0e6)
    return turbine_direct(fluid="Water", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.90)


class TestResultMetadata:
    def test_mode_and_fluid_are_recorded(self) -> None:
        direct = _water_turbine()
        assert direct.mode == "direct" and direct.fluid == "Water"
        inverse = turbine_inverse(
            fluid="Water", state_in=direct.state_in, state_out_real=direct.state_out_real
        )
        assert inverse.mode == "inverse"

    def test_specific_work_is_positive_for_every_device(self) -> None:
        assert _water_turbine().specific_work_J_per_kg > 0.0
        air = state_from_pair("Air", "TP", t=298.15, p=1.0e5)
        comp = compressor_direct(fluid="Air", state_in=air, p_out_Pa=8.0e5, eta_s=0.8)
        assert comp.specific_work_J_per_kg == pytest.approx(comp.delta_h_real_J_per_kg)


class TestNewValidations:
    def test_turbine_outlet_outside_equation_of_state_is_explained(self) -> None:
        # R134a a 400 °C supera su T máxima (182 °C): antes se extrapolaba sin aviso.
        state_in = state_from_pair("R134a", "TP", t=673.15, p=3.0e6)
        with pytest.raises(ValueError, match="rango"):
            turbine_direct(fluid="R134a", state_in=state_in, p_out_Pa=5.0e4, eta_s=0.9)

    def test_compressor_rejects_liquid_inlet(self) -> None:
        liquid = state_from_pair("Water", "TP", t=298.15, p=1.0e5)
        with pytest.raises(ValueError, match="bomba"):
            compressor_direct(fluid="Water", state_in=liquid, p_out_Pa=8.0e5, eta_s=0.8)

    def test_compressor_rejects_saturated_liquid(self) -> None:
        sat_liq = state_from_pair("R134a", "TX", t=263.15, x=0.0)
        with pytest.raises(ValueError, match="líquido"):
            compressor_direct(fluid="R134a", state_in=sat_liq, p_out_Pa=1.0e6, eta_s=0.8)

    def test_compressor_accepts_saturated_vapor(self) -> None:
        sat_vap = state_from_pair("R134a", "TX", t=263.15, x=1.0)
        result = compressor_direct(fluid="R134a", state_in=sat_vap, p_out_Pa=1.0e6, eta_s=0.8)
        assert result.state_out_real.T_K > sat_vap.T_K

    def test_intercooler_cannot_condense_the_fluid(self) -> None:
        # Vapor de agua comprimido de 1 a 9 bar con intercooler a 25 °C: a 3 bar
        # el agua condensaría (T_sat ≈ 133.5 °C).
        steam = state_from_pair("Water", "TP", t=393.15, p=1.0e5)
        with pytest.raises(ValueError, match="condensaría"):
            compressor_multistage(
                fluid="Water",
                state_in=steam,
                p_out_Pa=9.0e5,
                n_stages=2,
                eta_s_per_stage=0.8,
                intercool=True,
                t_intercool_K=298.15,
            )


class TestSuggestedDeviceInputs:
    @pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
    @pytest.mark.parametrize("device", ["turbine", "compressor", "pump"])
    def test_direct_and_inverse_round_trip(self, fluid: str, device: str) -> None:
        d = suggested_device_inputs(fluid, device)  # type: ignore[arg-type]
        state_in = fluid_state_from_pair(fluid, d.pair_in, **d.inlet).to_state_point()
        direct = _DIRECT[device](fluid=fluid, state_in=state_in, p_out_Pa=d.p_out_Pa, eta_s=d.eta_s)
        assert direct.specific_work_J_per_kg > 0.0
        inverse = _INVERSE[device](
            fluid=fluid, state_in=state_in, state_out_real=direct.state_out_real
        )
        assert inverse.eta_s == pytest.approx(d.eta_s, rel=1e-6)

    @pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
    def test_multistage_default_runs(self, fluid: str) -> None:
        d = suggested_device_inputs(fluid, "multistage")
        state_in = fluid_state_from_pair(fluid, d.pair_in, **d.inlet).to_state_point()
        result = compressor_multistage(
            fluid=fluid,
            state_in=state_in,
            p_out_Pa=d.p_out_Pa,
            n_stages=d.n_stages,
            eta_s_per_stage=d.eta_s,
            intercool=True,
            t_intercool_K=d.t_intercool_K,
        )
        assert result.stages[-1].p_out_Pa == pytest.approx(d.p_out_Pa)

    def test_water_defaults_are_the_textbook_ones(self) -> None:
        d = suggested_device_inputs("Water", "turbine")
        assert d.inlet == {"t": 673.15, "p": 3.0e6}
        assert d.p_out_Pa == 5.0e4

    def test_unknown_device(self) -> None:
        with pytest.raises(ValueError, match="desconocido"):
            suggested_device_inputs("Water", "nozzle")  # type: ignore[arg-type]


class TestPumpComparison:
    def test_incompressible_model_is_close_for_water(self) -> None:
        state_in = state_from_pair("Water", "PX", p=1.0e4, x=0.0)
        result = pump_direct(fluid="Water", state_in=state_in, p_out_Pa=1.5e7, eta_s=0.85)
        cmp = pump_incompressible_comparison(result)
        # v1·Δp/η = 0.0010103·(1.5e7 − 1e4)/0.85 ≈ 17.82 kJ/kg (Cengel ej. 10-1).
        assert cmp.w_incompressible_J_per_kg == pytest.approx(17.82e3, rel=2e-3)
        assert cmp.error_pct < 1.0

    def test_only_for_pumps(self) -> None:
        with pytest.raises(ValueError, match="bombas"):
            pump_incompressible_comparison(_water_turbine())


class TestStepsInEverySystem:
    """Fase 1.5b: los pasos se pueden pedir en SI / Técnico / Inglés."""

    def test_tecnico_matches_the_stored_steps(self) -> None:
        result = _water_turbine()
        assert isentropic_steps(result, "Técnico") == result.steps

    @pytest.mark.parametrize(
        "system,unit",
        [("SI", r"\mathrm{J/kg}"), ("Técnico", r"\mathrm{kJ/kg}"), ("Inglés", r"\mathrm{Btu/lb}")],
    )
    def test_units_follow_the_system(self, system: str, unit: str) -> None:
        steps = isentropic_steps(_water_turbine(), system)  # type: ignore[arg-type]
        assert unit in steps.substituted_latex
        assert steps.substituted_latex.count("{") == steps.substituted_latex.count("}")

    def test_ingles_values_are_converted(self) -> None:
        result = _water_turbine()
        steps = isentropic_steps(result, "Inglés")
        # h1 = 3231.7 kJ/kg = 1389.4 Btu/lb.
        assert "1389.4" in steps.substituted_latex
        assert "Btu/lb" in steps.narrative_es

    def test_inverse_and_compressor_branches(self) -> None:
        air = state_from_pair("Air", "TP", t=298.15, p=1.0e5)
        comp = compressor_direct(fluid="Air", state_in=air, p_out_Pa=8.0e5, eta_s=0.8)
        inv = compressor_inverse(fluid="Air", state_in=air, state_out_real=comp.state_out_real)
        assert "w_c" in isentropic_steps(comp, "SI").formula_latex
        assert (
            r"\eta_s &= \frac{h_{2s} - h_1}{h_2 - h_1}" in isentropic_steps(inv, "SI").formula_latex
        )

    def test_multistage_steps_convert_intercooler_temperature(self) -> None:
        air = state_from_pair("Air", "TP", t=298.15, p=1.0e5)
        result = compressor_multistage(
            fluid="Air",
            state_in=air,
            p_out_Pa=2.7e6,
            n_stages=3,
            eta_s_per_stage=0.85,
            intercool=True,
            t_intercool_K=298.15,
        )
        assert "77 °F" in multistage_steps(result, "Inglés").narrative_es
        assert "25 °C" in multistage_steps(result, "Técnico").narrative_es


class TestExport:
    def test_isentropic_dict(self) -> None:
        data = isentropic_to_dict(_water_turbine(), "Técnico")
        assert data["dispositivo"] == "turbina"
        assert data["modo"] == "directo"
        assert data["trabajo_especifico"]["unidad"] == "kJ/kg"
        assert [row["Estado"] for row in data["estados"]] == [
            label for label, _ in isentropic_labeled_states(_water_turbine())
        ]
        # El estado de salida real es vapor húmedo: x definido, región en castellano.
        assert data["estados"][2]["Región"].startswith("Vapor húmedo")

    def test_csv_has_summary_and_states(self) -> None:
        text = summary_csv(isentropic_to_dict(_water_turbine(), "SI"))
        assert text.startswith("# dispositivo,turbina")
        assert "Estado,Fluido,T [K]" in text

    def test_multistage_dict(self) -> None:
        air = state_from_pair("Air", "TP", t=298.15, p=1.0e5)
        result = compressor_multistage(
            fluid="Air",
            state_in=air,
            p_out_Pa=2.7e6,
            n_stages=3,
            eta_s_per_stage=0.85,
            intercool=True,
            t_intercool_K=298.15,
        )
        data = multistage_to_dict(result, "Técnico")
        assert data["etapas"] == 3
        assert len(data["estados"]) == len(multistage_labeled_states(result))
        assert data["ahorro_vs_una_etapa_pct"] == pytest.approx(result.saving_pct)
