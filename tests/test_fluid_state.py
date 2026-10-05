"""Tests del estado completo — :func:`core.fluids.fluid_state_from_pair`.

Valores de referencia de Cengel & Boles (2015), *Thermodynamics: An
Engineering Approach*, 8th ed.: Tablas A-4 (saturación por temperatura),
A-5 (por presión), A-6 (vapor sobrecalentado) y A-7 (líquido comprimido);
y de Cengel & Ghajar, *Heat and Mass Transfer*, Tabla A-9 (propiedades
de transporte del agua saturada). CoolProp implementa IAPWS-95 para el
agua; las tablas usan IAPWS-95/IF97 redondeadas, de ahí las tolerancias.
"""

from __future__ import annotations

import math

import CoolProp
import pytest

from core.fluids import (
    FLUID_NAMES_ES,
    PAIR_KWARGS,
    REGION_LABELS_ES,
    SUPPORTED_FLUIDS,
    FluidState,
    classify_region,
    fluid_limits,
    fluid_state_from_pair,
    saturation_at_pressure,
    saturation_at_temperature,
    state_from_pair,
    suggested_inputs,
)

WATER = "Water"
KJ = 1.0e3


def _c(T_K: float) -> float:
    return T_K - 273.15


@pytest.fixture(scope="module")
def compressed_liquid_5MPa_100C() -> FluidState:
    return fluid_state_from_pair(WATER, "TP", t=373.15, p=5.0e6)


@pytest.fixture(scope="module")
def superheated_1MPa_300C() -> FluidState:
    return fluid_state_from_pair(WATER, "TP", t=573.15, p=1.0e6)


@pytest.fixture(scope="module")
def liquid_25C_1bar() -> FluidState:
    return fluid_state_from_pair(WATER, "TP", t=298.15, p=1.0e5)


# ---------------------------------------------------------------------
# Metadatos
# ---------------------------------------------------------------------


class TestMetadata:
    def test_every_supported_fluid_has_spanish_name(self) -> None:
        assert set(FLUID_NAMES_ES) == set(SUPPORTED_FLUIDS)
        assert FLUID_NAMES_ES["Water"] == "Agua"

    def test_every_region_has_label(self) -> None:
        assert set(REGION_LABELS_ES) == {
            "compressed_liquid",
            "saturated_liquid",
            "saturated_mixture",
            "saturated_vapor",
            "superheated_vapor",
            "supercritical",
        }

    def test_pair_kwargs_cover_ten_pairs(self) -> None:
        assert PAIR_KWARGS["PH"] == ("p", "h")
        assert PAIR_KWARGS["TV"] == ("t", "v")
        assert PAIR_KWARGS["PU"] == ("p", "u")
        assert len(PAIR_KWARGS) == 10


class TestFluidLimits:
    def test_water_triple_and_critical_points(self) -> None:
        lim = fluid_limits(WATER)
        # IAPWS: punto triple 273.16 K / 611.655 Pa; crítico 647.096 K / 22.064 MPa.
        assert lim.T_triple_K == pytest.approx(273.16, abs=1e-6)
        assert lim.P_triple_Pa == pytest.approx(611.655, rel=1e-3)
        assert lim.T_crit_K == pytest.approx(647.096, abs=1e-3)
        assert lim.P_crit_Pa == pytest.approx(22.064e6, rel=1e-6)
        assert lim.v_crit_m3_per_kg == pytest.approx(1.0 / 322.0, rel=1e-6)

    def test_water_gas_constant(self) -> None:
        # R = R_u / M = 8.314462618 / 0.018015268 = 461.52 J/(kg·K).
        assert fluid_limits(WATER).R_J_per_kg_K == pytest.approx(461.52, rel=1e-4)

    def test_purity(self) -> None:
        assert fluid_limits(WATER).is_pure
        assert not fluid_limits("Air").is_pure
        assert not fluid_limits("R410A").is_pure

    def test_unknown_fluid(self) -> None:
        with pytest.raises(ValueError, match="no reconoce"):
            fluid_limits("Unobtainium")


# ---------------------------------------------------------------------
# Saturación (tablas A-4 y A-5)
# ---------------------------------------------------------------------


class TestSaturationAtPressure:
    def test_100_kPa_table_A5(self) -> None:
        sat = saturation_at_pressure(WATER, 100.0e3)
        assert _c(sat.T_sat_K) == pytest.approx(99.61, abs=0.01)
        assert sat.liquid.v_m3_per_kg == pytest.approx(0.001043, rel=1e-3)
        assert sat.vapor.v_m3_per_kg == pytest.approx(1.6941, rel=5e-4)
        assert sat.liquid.u_J_per_kg / KJ == pytest.approx(417.40, abs=0.15)
        assert sat.vapor.u_J_per_kg / KJ == pytest.approx(2505.6, abs=0.15)
        assert sat.liquid.h_J_per_kg / KJ == pytest.approx(417.51, abs=0.15)
        assert sat.h_fg_J_per_kg / KJ == pytest.approx(2257.5, abs=0.15)
        assert sat.vapor.h_J_per_kg / KJ == pytest.approx(2675.0, abs=0.15)
        assert sat.liquid.s_J_per_kg_K / KJ == pytest.approx(1.3028, abs=2e-4)
        assert sat.s_fg_J_per_kg_K / KJ == pytest.approx(6.0562, abs=2e-4)
        assert sat.vapor.s_J_per_kg_K / KJ == pytest.approx(7.3589, abs=2e-4)

    def test_1_MPa_table_A5(self) -> None:
        sat = saturation_at_pressure(WATER, 1.0e6)
        assert _c(sat.T_sat_K) == pytest.approx(179.88, abs=0.01)
        assert sat.vapor.v_m3_per_kg == pytest.approx(0.19436, rel=5e-4)
        assert sat.liquid.h_J_per_kg / KJ == pytest.approx(762.51, abs=0.15)
        assert sat.h_fg_J_per_kg / KJ == pytest.approx(2014.6, abs=0.15)
        assert sat.vapor.s_J_per_kg_K / KJ == pytest.approx(6.5850, abs=2e-4)

    def test_derived_fg_properties(self) -> None:
        sat = saturation_at_pressure(WATER, 1.0e6)
        assert sat.v_fg_m3_per_kg == pytest.approx(sat.vapor.v_m3_per_kg - sat.liquid.v_m3_per_kg)
        assert sat.u_fg_J_per_kg == pytest.approx(sat.vapor.u_J_per_kg - sat.liquid.u_J_per_kg)

    def test_pure_fluid_has_no_glide(self) -> None:
        sat = saturation_at_pressure(WATER, 1.0e6)
        assert sat.glide_K == pytest.approx(0.0, abs=1e-9)
        assert sat.basis == "P"

    def test_pseudo_pure_has_glide(self) -> None:
        # Aire a 1 bar: burbuja ≈ 78.8 K, rocío ≈ 81.6 K.
        assert saturation_at_pressure("Air", 1.0e5).glide_K > 2.0

    def test_above_critical_pressure_raises(self) -> None:
        with pytest.raises(ValueError, match="punto crítico"):
            saturation_at_pressure(WATER, 25.0e6)

    def test_below_triple_pressure_raises(self) -> None:
        with pytest.raises(ValueError, match="punto triple"):
            saturation_at_pressure(WATER, 100.0)


class TestSaturationAtTemperature:
    def test_100C_table_A4(self) -> None:
        sat = saturation_at_temperature(WATER, 373.15)
        assert sat.P_sat_Pa / KJ == pytest.approx(101.42, abs=0.01)
        assert sat.vapor.v_m3_per_kg == pytest.approx(1.6720, rel=5e-4)
        assert sat.liquid.h_J_per_kg / KJ == pytest.approx(419.17, abs=0.15)
        assert sat.h_fg_J_per_kg / KJ == pytest.approx(2256.4, abs=0.15)
        assert sat.vapor.s_J_per_kg_K / KJ == pytest.approx(7.3542, abs=2e-4)
        assert sat.basis == "T"

    def test_above_critical_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="punto crítico"):
            saturation_at_temperature(WATER, 700.0)


# ---------------------------------------------------------------------
# Regiones y propiedades
# ---------------------------------------------------------------------


class TestCompressedLiquid:
    """Tabla A-7: 5 MPa, 100 °C."""

    @pytest.fixture
    def state(self, compressed_liquid_5MPa_100C: FluidState) -> FluidState:
        return compressed_liquid_5MPa_100C

    def test_region(self, state: FluidState) -> None:
        assert state.region == "compressed_liquid"
        assert state.x is None
        assert not state.is_two_phase

    def test_properties_match_table(self, state: FluidState) -> None:
        assert state.v_m3_per_kg == pytest.approx(0.0010410, rel=5e-4)
        assert state.u_J_per_kg / KJ == pytest.approx(417.65, abs=0.15)
        assert state.h_J_per_kg / KJ == pytest.approx(422.85, abs=0.15)
        assert state.s_J_per_kg_K / KJ == pytest.approx(1.3034, abs=2e-4)

    def test_subcooling(self, state: FluidState) -> None:
        # T_sat(5 MPa) = 263.94 °C → 163.94 K de subenfriamiento.
        assert state.subcooling_K == pytest.approx(163.94, abs=0.02)
        assert state.superheat_K is None

    def test_saturation_at_both_anchors(self, state: FluidState) -> None:
        assert state.sat_at_P is not None and state.sat_at_P.basis == "P"
        assert state.sat_at_T is not None
        # A-4: p_sat(100 °C) = 101.42 kPa < 5 MPa → por eso es líquido comprimido.
        assert state.sat_at_T.P_sat_Pa == pytest.approx(101.42e3, rel=1e-4)

    def test_liquid_is_far_from_ideal_gas(self, state: FluidState) -> None:
        assert state.Z < 0.1


class TestSuperheatedVapor:
    """Tabla A-6: 1 MPa, 300 °C."""

    @pytest.fixture
    def state(self, superheated_1MPa_300C: FluidState) -> FluidState:
        return superheated_1MPa_300C

    def test_region(self, state: FluidState) -> None:
        assert state.region == "superheated_vapor"
        assert state.x is None

    def test_properties_match_table(self, state: FluidState) -> None:
        assert state.v_m3_per_kg == pytest.approx(0.25799, rel=2e-4)
        assert state.u_J_per_kg / KJ == pytest.approx(2793.7, abs=0.15)
        assert state.h_J_per_kg / KJ == pytest.approx(3051.6, abs=0.15)
        assert state.s_J_per_kg_K / KJ == pytest.approx(7.1246, abs=2e-4)
        assert state.rho_kg_per_m3 == pytest.approx(1.0 / 0.25799, rel=2e-4)

    def test_superheat(self, state: FluidState) -> None:
        # 300 °C − T_sat(1 MPa) = 300 − 179.88 = 120.12 K.
        assert state.superheat_K == pytest.approx(120.12, abs=0.02)
        assert state.subcooling_K is None

    def test_compressibility_factor(self, state: FluidState) -> None:
        # Z = p·v/(R·T) = 1e6·0.25799/(461.52·573.15) ≈ 0.9753.
        assert state.Z == pytest.approx(0.9753, abs=5e-4)

    def test_caloric_and_transport_defined(self, state: FluidState) -> None:
        assert state.cp_J_per_kg_K is not None and state.cp_J_per_kg_K > 2000.0
        assert state.gamma is not None and 1.2 < state.gamma < 1.4
        assert state.speed_of_sound_m_per_s is not None
        assert state.prandtl == pytest.approx(
            state.viscosity_Pa_s * state.cp_J_per_kg_K / state.conductivity_W_per_m_K
        )

    def test_low_pressure_steam_is_nearly_ideal_gas(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=573.15, p=1.0e3)
        assert state.Z == pytest.approx(1.0, abs=1e-3)


class TestTransportProperties:
    """Cengel & Ghajar, Tabla A-9 (agua a 25 °C): μ = 0.891e-3 kg/(m·s),
    k = 0.607 W/(m·K), Pr = 6.14."""

    @pytest.fixture
    def state(self, liquid_25C_1bar: FluidState) -> FluidState:
        return liquid_25C_1bar

    def test_viscosity(self, state: FluidState) -> None:
        assert state.viscosity_Pa_s == pytest.approx(0.891e-3, rel=1e-2)

    def test_conductivity(self, state: FluidState) -> None:
        assert state.conductivity_W_per_m_K == pytest.approx(0.607, rel=1e-2)

    def test_prandtl(self, state: FluidState) -> None:
        assert state.prandtl == pytest.approx(6.14, rel=1e-2)

    def test_diffusivities(self, state: FluidState) -> None:
        nu = state.kinematic_viscosity_m2_per_s
        alpha = state.thermal_diffusivity_m2_per_s
        assert nu == pytest.approx(state.viscosity_Pa_s * state.v_m3_per_kg)
        # Pr = ν/α.
        assert nu / alpha == pytest.approx(state.prandtl, rel=1e-12)


class TestWetSteam:
    @pytest.mark.parametrize(
        "pair,kwargs",
        [
            ("PX", {"p": 1.0e6, "x": 0.3}),
            ("PH", {"p": 1.0e6, "h": 1.5e6}),
            ("PS", {"p": 1.0e4, "s": 7.0e3}),
            ("TV", {"t": 393.15, "v": 0.5}),
            ("PV", {"p": 2.0e5, "v": 0.5}),
            ("PU", {"p": 2.0e5, "u": 2.0e6}),
            ("HS", {"h": 2.0e6, "s": 5.0e3}),
        ],
    )
    def test_lever_rule_holds_for_every_pair(self, pair: str, kwargs: dict[str, float]) -> None:
        state = fluid_state_from_pair(WATER, pair, **kwargs)  # type: ignore[arg-type]
        assert state.region == "saturated_mixture"
        assert state.x is not None and 0.0 < state.x < 1.0
        sat = state.sat_at_P
        assert sat is not None
        # vademecum §12.2: y = y_f + x·(y_g − y_f), para v, u, h y s.
        x = state.x
        assert state.h_J_per_kg == pytest.approx(sat.liquid.h_J_per_kg + x * sat.h_fg_J_per_kg)
        assert state.s_J_per_kg_K == pytest.approx(
            sat.liquid.s_J_per_kg_K + x * sat.s_fg_J_per_kg_K
        )
        assert state.v_m3_per_kg == pytest.approx(sat.liquid.v_m3_per_kg + x * sat.v_fg_m3_per_kg)
        assert state.u_J_per_kg == pytest.approx(sat.liquid.u_J_per_kg + x * sat.u_fg_J_per_kg)
        assert state.T_K == pytest.approx(sat.T_sat_K, abs=1e-6)

    def test_single_phase_properties_are_undefined(self) -> None:
        state = fluid_state_from_pair(WATER, "PX", p=1.0e5, x=0.5)
        assert state.cp_J_per_kg_K is None
        assert state.cv_J_per_kg_K is None
        assert state.gamma is None
        assert state.speed_of_sound_m_per_s is None
        assert state.viscosity_Pa_s is None
        assert state.prandtl is None
        assert state.kinematic_viscosity_m2_per_s is None
        assert state.thermal_diffusivity_m2_per_s is None
        assert state.superheat_K is None and state.subcooling_K is None

    def test_rigid_tank_quality(self) -> None:
        # Cengel, ej. 3-5 (tipo): tanque rígido a 120 °C con v = 0.5 m³/kg.
        # x = (v − v_f)/(v_g − v_f) = (0.5 − 0.001060)/(0.89133 − 0.001060).
        state = fluid_state_from_pair(WATER, "TV", t=393.15, v=0.5)
        assert state.x == pytest.approx((0.5 - 0.001060) / (0.89133 - 0.001060), rel=1e-3)


class TestSaturationBoundaries:
    def test_saturated_liquid(self) -> None:
        state = fluid_state_from_pair(WATER, "PX", p=1.0e5, x=0.0)
        assert state.region == "saturated_liquid"
        assert state.x == 0.0
        assert state.is_two_phase
        # En los bordes de la campana sí hay cp y transporte (fase saturada).
        assert state.cp_J_per_kg_K == pytest.approx(4215.0, rel=5e-3)
        assert state.viscosity_Pa_s is not None

    def test_saturated_vapor(self) -> None:
        state = fluid_state_from_pair(WATER, "TX", t=373.15, x=1.0)
        assert state.region == "saturated_vapor"
        assert state.x == 1.0
        assert state.v_m3_per_kg == pytest.approx(1.6720, rel=5e-4)


class TestSupercriticalAndBeyondCritical:
    def test_supercritical_fluid(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=773.15, p=30.0e6)
        assert state.region == "supercritical"
        assert state.sat_at_P is None and state.sat_at_T is None
        assert state.superheat_K is None and state.subcooling_K is None

    def test_liquid_above_critical_pressure_is_compressed_liquid(self) -> None:
        # p > p_c y T < T_c: Cengel lo trata como líquido comprimido.
        state = fluid_state_from_pair(WATER, "TP", t=573.15, p=30.0e6)
        assert state.region == "compressed_liquid"
        assert state.sat_at_P is None
        assert state.subcooling_K is None
        assert state.sat_at_T is not None

    def test_gas_above_critical_temperature_is_superheated(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=700.0, p=10.0e6)
        assert state.region == "superheated_vapor"
        assert state.sat_at_T is None
        assert state.superheat_K == pytest.approx(700.0 - 584.15, abs=0.1)


class TestClassifyRegion:
    @pytest.mark.parametrize(
        "phase,quality,expected",
        [
            (CoolProp.iphase_liquid, -1.0, ("compressed_liquid", None)),
            (CoolProp.iphase_supercritical_liquid, -1.0, ("compressed_liquid", None)),
            (CoolProp.iphase_gas, 10000.0, ("superheated_vapor", None)),
            (CoolProp.iphase_supercritical_gas, -1.0, ("superheated_vapor", None)),
            (CoolProp.iphase_supercritical, -1.0, ("supercritical", None)),
            (CoolProp.iphase_twophase, 0.0, ("saturated_liquid", 0.0)),
            (CoolProp.iphase_twophase, 1.0, ("saturated_vapor", 1.0)),
            (CoolProp.iphase_twophase, 0.25, ("saturated_mixture", 0.25)),
        ],
    )
    def test_mapping(self, phase: int, quality: float, expected: tuple) -> None:
        assert classify_region(phase, quality) == expected


# ---------------------------------------------------------------------
# Pares nuevos (T-v, p-v, p-u) y round-trip
# ---------------------------------------------------------------------


class TestNewPairsRoundTrip:
    @pytest.mark.parametrize(
        "T_K,P_Pa",
        [(298.15, 1.0e5), (373.15, 5.0e6), (573.15, 1.0e6), (773.15, 30.0e6)],
    )
    @pytest.mark.parametrize("pair", ["TV", "PV", "PU", "PH", "PS", "TS", "HS"])
    def test_recovers_TP_state(self, T_K: float, P_Pa: float, pair: str) -> None:
        ref = fluid_state_from_pair(WATER, "TP", t=T_K, p=P_Pa)
        values = {
            "t": ref.T_K,
            "p": ref.P_Pa,
            "h": ref.h_J_per_kg,
            "s": ref.s_J_per_kg_K,
            "v": ref.v_m3_per_kg,
            "u": ref.u_J_per_kg,
        }
        k1, k2 = PAIR_KWARGS[pair]
        got = fluid_state_from_pair(WATER, pair, **{k1: values[k1], k2: values[k2]})  # type: ignore[arg-type]
        assert got.T_K == pytest.approx(ref.T_K, abs=1e-4)
        assert got.P_Pa == pytest.approx(ref.P_Pa, rel=1e-5)
        assert got.region == ref.region

    def test_state_from_pair_accepts_new_pairs(self) -> None:
        sp = state_from_pair(WATER, "TV", t=393.15, v=0.5)
        assert 0.0 < sp.x < 1.0
        assert sp.v_m3_per_kg == pytest.approx(0.5)
        sp = state_from_pair(WATER, "PU", p=2.0e5, u=2.0e6)
        assert 0.0 < sp.x < 1.0

    def test_to_state_point(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=573.15, p=1.0e6)
        sp = state.to_state_point()
        assert sp.x == -1.0
        assert sp.v_m3_per_kg == pytest.approx(state.v_m3_per_kg)
        wet = fluid_state_from_pair(WATER, "PX", p=1.0e5, x=0.4).to_state_point()
        assert wet.x == pytest.approx(0.4)


# ---------------------------------------------------------------------
# Validación con mensajes para el alumno
# ---------------------------------------------------------------------


class TestStudentFacingErrors:
    def test_below_triple_point_mentions_ice(self) -> None:
        # Era el default de la página vieja (0 °C, 1 bar) y fallaba en inglés.
        with pytest.raises(ValueError, match="punto triple.*hielo"):
            fluid_state_from_pair(WATER, "TP", t=273.15, p=1.0e5)

    def test_quality_above_critical_pressure(self) -> None:
        with pytest.raises(ValueError, match="presión crítica"):
            fluid_state_from_pair(WATER, "PX", p=30.0e6, x=0.5)

    def test_quality_above_critical_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperatura crítica"):
            fluid_state_from_pair(WATER, "TX", t=700.0, x=0.5)

    def test_quality_below_triple_pressure(self) -> None:
        with pytest.raises(ValueError, match="punto triple"):
            fluid_state_from_pair(WATER, "PX", p=100.0, x=0.5)

    def test_result_outside_equation_of_state(self) -> None:
        # Con la página vieja daba p ≈ 44 549 bar sin ninguna advertencia.
        with pytest.raises(ValueError, match="fuera del rango de validez"):
            fluid_state_from_pair("R134a", "TS", t=273.15, s=0.0)

    def test_inconsistent_pair_is_explained(self) -> None:
        with pytest.raises(ValueError, match="no existe un estado"):
            fluid_state_from_pair(WATER, "HS", h=0.0, s=0.0)

    def test_pressure_above_maximum(self) -> None:
        with pytest.raises(ValueError, match="máximo de validez"):
            fluid_state_from_pair(WATER, "TP", t=573.15, p=2.0e9)

    def test_temperature_above_maximum(self) -> None:
        with pytest.raises(ValueError, match="máximo de validez"):
            fluid_state_from_pair(WATER, "TP", t=2500.0, p=1.0e5)

    def test_non_positive_specific_volume(self) -> None:
        with pytest.raises(ValueError, match="volumen específico"):
            fluid_state_from_pair(WATER, "TV", t=373.15, v=0.0)

    def test_non_finite_input(self) -> None:
        with pytest.raises(ValueError, match="finito"):
            fluid_state_from_pair(WATER, "TP", t=math.nan, p=1.0e5)

    @pytest.mark.parametrize("fluid", ["R410A", "Air"])
    def test_pseudo_pure_quality_with_temperature_is_explained(self, fluid: str) -> None:
        # A T fija, la presión de un pseudo-puro cambia a lo largo de la campana.
        T = fluid_limits(fluid).T_crit_K - 20.0
        with pytest.raises(ValueError, match="pseudo-puro.*p-x"):
            fluid_state_from_pair(fluid, "TX", t=T, x=0.5)

    def test_pseudo_pure_accepts_saturation_ends(self) -> None:
        state = fluid_state_from_pair("R410A", "PX", p=8.0e5, x=1.0)
        assert state.region == "saturated_vapor"
        state = fluid_state_from_pair("Air", "TX", t=100.0, x=0.0)
        assert state.region == "saturated_liquid"


class TestReferenceStateNoise:
    """Donde u, h o s valen 0 por convención, CoolProp devuelve ruido
    (−6.6×10⁻⁸ J/kg) que la página mostraba como "−6.6048×10⁻¹¹ kJ/kg"."""

    def test_water_triple_point_like_table_a4(self) -> None:
        # Cengel A-4, 0.01 °C: u_f = 0.000, h_f = 0.001 kJ/kg, s_f = 0.0000.
        sat = saturation_at_temperature(WATER, 273.16)
        assert sat.liquid.u_J_per_kg == 0.0
        assert sat.liquid.s_J_per_kg_K == 0.0
        assert sat.liquid.h_J_per_kg == pytest.approx(0.6118, rel=1e-3)  # p·v, no es ruido

    def test_liquid_air_at_one_atmosphere(self) -> None:
        state = fluid_state_from_pair("Air", "PX", p=101_325.0, x=0.0)
        assert (state.h_J_per_kg, state.s_J_per_kg_K) == (0.0, 0.0)
        assert state.u_J_per_kg == pytest.approx(-115.8, rel=1e-3)  # u = h − p·v


class TestPseudoPureTwoPhase:
    """Aire y R410A (pseudo-puros: mezclas tratadas como un fluido) en la campana.

    Con deslizamiento de temperatura, el cálculo de CoolProp con p y h no
    reconoce la campana cerca de la línea de burbuja: en el borde devolvía
    "líquido comprimido" y apenas adentro (x ≈ 1e-4 … 1e-2) fallaba con un
    mensaje que decía que el estado no existía. Ahora el título sale de la
    regla de la palanca a p constante y el estado se calcula con (p, x).
    """

    P = 101_325.0  # aire a 1 atm: burbuja 78,90 K, rocío 81,72 K

    _ATTR = {"h": "h_J_per_kg", "s": "s_J_per_kg_K", "v": "v_m3_per_kg", "u": "u_J_per_kg"}

    @pytest.mark.parametrize("x", [0.0, 1e-4, 1e-3, 1e-2, 0.5, 1.0])
    @pytest.mark.parametrize("kw", ["h", "s", "v", "u"])
    def test_air_along_the_dome(self, kw: str, x: float) -> None:
        sat = saturation_at_pressure("Air", self.P)
        y_f = getattr(sat.liquid, self._ATTR[kw])
        y_g = getattr(sat.vapor, self._ATTR[kw])
        value = y_f + x * (y_g - y_f)
        state = fluid_state_from_pair("Air", f"P{kw.upper()}", p=self.P, **{kw: value})
        expected = {0.0: "saturated_liquid", 1.0: "saturated_vapor"}.get(x, "saturated_mixture")
        assert state.region == expected
        assert state.x == pytest.approx(x, abs=1e-9)
        assert sat.liquid.T_K - 1e-9 <= state.T_K <= sat.vapor.T_K + 1e-9

    @pytest.mark.parametrize(("fluid", "p"), [("Air", 1.5e5), ("R410A", 8.0e5)])
    def test_px_inside_the_dome_matches_ph(self, fluid: str, p: float) -> None:
        state = fluid_state_from_pair(fluid, "PX", p=p, x=0.3)
        assert state.region == "saturated_mixture"
        again = fluid_state_from_pair(fluid, "PH", p=p, h=state.h_J_per_kg)
        assert again.x == pytest.approx(0.3)
        assert again.T_K == pytest.approx(state.T_K)

    def test_outside_the_dome_is_unchanged(self) -> None:
        sat = saturation_at_pressure("Air", self.P)
        below = fluid_state_from_pair("Air", "PH", p=self.P, h=sat.liquid.h_J_per_kg - 1000.0)
        above = fluid_state_from_pair("Air", "PH", p=self.P, h=sat.vapor.h_J_per_kg + 1000.0)
        assert below.region == "compressed_liquid"
        assert above.region == "superheated_vapor"


# ---------------------------------------------------------------------
# Valores sugeridos para los formularios
# ---------------------------------------------------------------------


class TestSuggestedInputs:
    @pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
    @pytest.mark.parametrize("pair", sorted(PAIR_KWARGS))
    def test_every_default_is_a_valid_state(self, fluid: str, pair: str) -> None:
        kwargs = suggested_inputs(fluid, pair)  # type: ignore[arg-type]
        assert set(kwargs) == set(PAIR_KWARGS[pair])
        state = fluid_state_from_pair(fluid, pair, **kwargs)  # type: ignore[arg-type]
        assert math.isfinite(state.h_J_per_kg)

    def test_water_defaults_are_textbook_states(self) -> None:
        assert suggested_inputs(WATER, "TP") == {"t": 573.15, "p": 1.0e6}
        state = fluid_state_from_pair(WATER, "PS", **suggested_inputs(WATER, "PS"))
        assert state.region == "saturated_mixture"

    def test_returns_a_copy(self) -> None:
        a = suggested_inputs(WATER, "TP")
        a["t"] = 0.0
        assert suggested_inputs(WATER, "TP")["t"] == 573.15

    def test_unknown_pair(self) -> None:
        with pytest.raises(ValueError, match="no soportado"):
            suggested_inputs(WATER, "XY")  # type: ignore[arg-type]
