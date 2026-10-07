"""Tests del modelo de aire húmedo — Fase 4 (vademecum §14; Cengel §14-1 a §14-5).

Contra las tablas de Cengel (A-4 y A-8), los ejemplos 14-1 a 14-4, la ecuación
de sublimación de IAPWS (2011), la atmósfera estándar de ASHRAE y el modelo de
gas real de CoolProp (``HAPropsSI``, ASHRAE RP-1485).
"""

from __future__ import annotations

import itertools
import json
import math
import threading

import numpy as np
import pytest
from CoolProp.CoolProp import HAPropsSI

from core.psychrometrics import (
    C_P_AIR,
    CHART_PHI_VALUES,
    EPSILON,
    MOLAR_RATIO,
    P_SEA_LEVEL_PA,
    PAIR_LABELS,
    PAIRS,
    R_AIR,
    R_VAPOR,
    STATE_EXAMPLE_NOTES,
    STATE_EXAMPLES,
    T_TRIPLE_K,
    DeadState,
    altitude_from_pressure,
    altitude_sweep,
    chart_grid,
    chart_lines,
    chart_window,
    coolprop_comparison,
    default_altitude_values,
    dew_point,
    moist_air_state,
    moist_air_to_dict,
    moist_enthalpy,
    pressure_from_altitude,
    room_masses,
    saturation_humidity_ratio,
    saturation_pressure,
    saturation_temperature,
    state_from_T_omega,
    state_from_T_phi,
    water_enthalpy,
    water_exergy,
    wet_bulb_temperature,
)

C = 273.15
P = P_SEA_LEVEL_PA


def _values(state):  # type: ignore[no-untyped-def]
    return {
        "T": state.T_K,
        "phi": state.phi,
        "T_wb": state.T_wb_K,
        "T_dp": state.T_dp_K,
        "omega": state.omega,
        "h": state.h_J_per_kg,
    }


# ---------------------------------------------------------------------
# Constantes y saturación
# ---------------------------------------------------------------------


def test_constants_are_the_vademecum_ones() -> None:
    assert C_P_AIR == 1005.0
    assert EPSILON == 0.622
    assert R_AIR == 287.0
    # R_v = R_a·M_a/M_v = 461,5 J/(kg·K) (Cengel A-1: 0,4615), coherente con 1,608.
    assert R_VAPOR == pytest.approx(461.5, abs=0.01)
    assert R_VAPOR / R_AIR == pytest.approx(MOLAR_RATIO)


@pytest.mark.parametrize(
    ("T_C", "p_kPa"),
    [(10.0, 1.2281), (25.0, 3.1698), (30.0, 4.2469), (50.0, 12.352), (100.0, 101.42)],
)
def test_saturation_pressure_matches_table_a4(T_C: float, p_kPa: float) -> None:
    assert saturation_pressure(T_C + C) / 1e3 == pytest.approx(p_kPa, rel=2e-4)


def test_sublimation_pressure_is_iapws_2011() -> None:
    # Valor de verificación de IAPWS R14-08(2011): 230 K → 8,94735 Pa.
    assert saturation_pressure(230.0) == pytest.approx(8.947352740189, rel=1e-9)
    # Continua en el punto triple (hielo y líquido dan 611,657 Pa).
    below = saturation_pressure(T_TRIPLE_K - 1e-9)
    above = saturation_pressure(T_TRIPLE_K)
    assert below == pytest.approx(611.657, rel=1e-8)
    assert above == pytest.approx(below, rel=1e-5)


@pytest.mark.parametrize("T_C", [-40.0, -10.0, -0.5, 0.005, 5.0, 60.0, 150.0])
def test_saturation_temperature_inverts_the_pressure(T_C: float) -> None:
    T = T_C + C
    assert saturation_temperature(saturation_pressure(T)) == pytest.approx(T, abs=1e-6)


def test_water_enthalpy_uses_tables_and_ice() -> None:
    assert water_enthalpy(287.15) / 1e3 == pytest.approx(58.8, abs=0.05)  # h_f a 14 °C (A-4)
    assert water_enthalpy(373.15, "vapor") / 1e3 == pytest.approx(2675.6, abs=0.2)
    assert water_enthalpy(263.15, "ice") / 1e3 == pytest.approx(-333.4 - 21.0)
    with pytest.raises(ValueError, match="punto triple"):
        water_enthalpy(270.0, "liquid")


def test_saturation_works_from_another_thread() -> None:
    """Streamlit corre cada sesión en un hilo: el AbstractState es uno por hilo."""
    out: list[float] = []
    thread = threading.Thread(target=lambda: out.append(saturation_pressure(298.15)))
    thread.start()
    thread.join()
    assert out[0] == pytest.approx(saturation_pressure(298.15))


# ---------------------------------------------------------------------
# Altura
# ---------------------------------------------------------------------


@pytest.mark.parametrize(("z", "p_kPa"), [(0.0, 101.325), (1000.0, 89.875), (3000.0, 70.108)])
def test_pressure_from_altitude_is_ashrae_standard_atmosphere(z: float, p_kPa: float) -> None:
    assert pressure_from_altitude(z) / 1e3 == pytest.approx(p_kPa, abs=0.005)
    assert altitude_from_pressure(pressure_from_altitude(z)) == pytest.approx(z, abs=1e-6)


def test_altitude_sweep_moister_and_lighter_air_up_high() -> None:
    points = altitude_sweep(293.15, 0.5, default_altitude_values())
    assert len(points) == 26
    omegas = [pt.state.omega for pt in points]
    rhos = [pt.state.rho_kg_per_m3 for pt in points]
    wet_bulbs = [pt.state.T_wb_K for pt in points]
    assert all(np.diff(omegas) > 0)  # misma φ: más vapor por kg de aire seco
    assert all(np.diff(rhos) < 0)
    assert all(np.diff(wet_bulbs) < 0)


# ---------------------------------------------------------------------
# Estado: ejemplos de Cengel
# ---------------------------------------------------------------------


def test_cengel_14_1_room_air() -> None:
    s = state_from_T_phi(100.0e3, 25 + C, 0.75)
    assert s.p_a_Pa / 1e3 == pytest.approx(97.62, abs=0.01)
    assert s.omega == pytest.approx(0.0152, abs=1e-4)
    assert s.h_J_per_kg / 1e3 == pytest.approx(63.8, abs=0.15)  # Cengel usa h_g = 2546,5
    m_a, m_v = room_masses(s, 75.0)
    assert m_a == pytest.approx(85.61, rel=1e-3)  # Cengel: T = 298 K
    assert m_v == pytest.approx(1.30, abs=0.01)


def test_cengel_14_2_dew_point() -> None:
    s = state_from_T_phi(P, 20 + C, 0.75)
    assert s.T_dp_K - C == pytest.approx(15.4, abs=0.05)


def test_cengel_14_3_psychrometer() -> None:
    s = moist_air_state(P, ("T", "T_wb"), 25 + C, 15 + C)
    assert s.omega == pytest.approx(0.00653, abs=5e-6)
    assert s.phi == pytest.approx(0.332, abs=5e-4)
    assert s.h_J_per_kg / 1e3 == pytest.approx(41.8, abs=0.06)
    assert s.T_wb_K == pytest.approx(15 + C, abs=1e-6)


def test_cengel_14_4_chart_reading() -> None:
    s = state_from_T_phi(P, 35 + C, 0.40)
    assert s.omega == pytest.approx(0.0142, abs=1e-4)
    assert s.h_J_per_kg / 1e3 == pytest.approx(71.5, abs=0.1)
    assert s.T_wb_K - C == pytest.approx(24.0, abs=0.1)
    assert s.T_dp_K - C == pytest.approx(19.4, abs=0.05)
    assert s.v_m3_per_kg == pytest.approx(0.893, abs=0.001)


def test_properties_follow_the_vademecum_formulas() -> None:
    s = state_from_T_phi(P, 30 + C, 0.6)
    w = s.omega
    assert s.p_v_Pa == pytest.approx(0.6 * saturation_pressure(30 + C))
    assert w == pytest.approx(0.622 * s.p_v_Pa / (P - s.p_v_Pa))
    assert s.h_J_per_kg == pytest.approx(1005 * 30 + w * (2501e3 + 1864 * 30))
    assert s.v_m3_per_kg == pytest.approx(287 * (30 + C) / (P - s.p_v_Pa))
    assert s.rho_kg_per_m3 == pytest.approx((1 + w) / s.v_m3_per_kg)
    assert s.R_J_per_kg_K == pytest.approx(287 * (1 + 1.608 * w) / (1 + w))
    assert s.cp_J_per_kg_K == pytest.approx(1005 + w * 1864)
    assert s.v_moist_m3_per_kg == pytest.approx(s.v_m3_per_kg / (1 + w))
    assert s.mu == pytest.approx(w / saturation_humidity_ratio(30 + C, P))
    assert s.mu < s.phi  # μ = φ·(p − p_vs)/(p − p_v) < φ
    assert s.T_dp_K < s.T_wb_K < s.T_K


# ---------------------------------------------------------------------
# Pares de datos
# ---------------------------------------------------------------------


GRID = [
    (30 + C, 0.5, P),
    (-10 + C, 0.6, P),
    (-30 + C, 0.9, P),
    (5 + C, 0.95, pressure_from_altitude(4500.0)),
    (80 + C, 0.2, P),
    (150 + C, 0.02, P),
    (35 + C, 0.3, 7.0e5),
    (0.005 + C, 0.999, P),
]


@pytest.mark.parametrize(("T", "phi", "p"), GRID)
@pytest.mark.parametrize("pair", PAIRS)
def test_every_pair_reproduces_the_state(
    pair: tuple[str, str], T: float, phi: float, p: float
) -> None:
    ref = state_from_T_phi(p, T, phi)
    v = _values(ref)
    s = moist_air_state(p, pair, v[pair[0]], v[pair[1]])
    assert s.T_K == pytest.approx(ref.T_K, abs=1e-6)
    assert s.omega == pytest.approx(ref.omega, rel=1e-7)
    # en cualquier orden
    s2 = moist_air_state(p, pair[::-1], v[pair[1]], v[pair[0]])
    assert s2.T_K == pytest.approx(ref.T_K, abs=1e-6)


def test_pair_labels_cover_every_pair() -> None:
    assert set(PAIR_LABELS) == set(PAIRS)
    assert len(PAIRS) == 13


@pytest.mark.parametrize("pair", [("omega", "T_dp"), ("h", "T_wb"), ("T", "T")])
def test_dependent_pairs_are_rejected(pair: tuple[str, str]) -> None:
    with pytest.raises(ValueError, match="misma información"):
        moist_air_state(P, pair, 0.01, 0.02)


# ---------------------------------------------------------------------
# Casos límite
# ---------------------------------------------------------------------


def test_dry_air() -> None:
    s = state_from_T_phi(P, 25 + C, 0.0)
    assert s.omega == 0.0 and s.T_dp_K is None and s.s_vapor is None
    assert s.T_wb_K < s.T_K
    # T y T_bh del aire seco vuelven a dar aire seco (sin el error crudo de brentq)
    back = moist_air_state(P, ("T", "T_wb"), 25 + C, s.T_wb_K)
    assert back.omega == 0.0 and back.T_dp_K is None
    assert math.isfinite(s.psi(DeadState(298.15, 0.5)))


def test_saturated_air_has_the_three_temperatures_equal() -> None:
    s = state_from_T_phi(P, 20 + C, 1.0)
    assert s.saturated
    assert s.T_wb_K == pytest.approx(s.T_K, abs=1e-6)
    assert s.T_dp_K == pytest.approx(s.T_K, abs=1e-6)


def test_frost_point_and_ice_bulb_below_zero() -> None:
    s = state_from_T_phi(P, -10 + C, 0.5)
    assert s.dew_over_ice and s.wet_bulb_over_ice
    assert s.T_dp_K - C == pytest.approx(
        HAPropsSI("D", "T", -10 + C, "P", P, "R", 0.5) - C, abs=0.02
    )
    assert s.T_wb_K - C == pytest.approx(
        HAPropsSI("B", "T", -10 + C, "P", P, "R", 0.5) - C, abs=0.02
    )


def test_above_the_boiling_point_the_air_cannot_saturate() -> None:
    s = state_from_T_phi(P, 120 + C, 0.05)
    assert s.omega_s is None and s.mu is None
    assert s.T_wb_K < saturation_temperature(P)


def test_humidity_ratio_and_dew_point_are_consistent() -> None:
    s = state_from_T_phi(P, 28 + C, 0.65)
    assert dew_point(s.omega, P) == pytest.approx(s.T_dp_K)
    assert dew_point(0.0, P) is None
    assert moist_enthalpy(s.T_K, s.omega) == pytest.approx(s.h_J_per_kg)
    assert wet_bulb_temperature(s.T_K, s.omega, P) == pytest.approx(s.T_wb_K)


# ---------------------------------------------------------------------
# Validaciones (mensajes al alumno)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pair", "a", "b", "match"),
    [
        (("T", "phi"), 25 + C, 1.05, "100 %"),
        (("T", "T_wb"), 25 + C, 27 + C, "no puede superar"),
        (("T", "T_wb"), 25 + C, -30 + C, "humedad negativa"),
        (("T", "T_dp"), 25 + C, 26 + C, "sobresaturado"),
        (("T", "h"), 25 + C, 10e3, "negativa"),
        (("T", "omega"), 25 + C, 0.05, "sobresaturado"),
        (("T", "phi"), 300 + C, 0.5, "200 °C"),
        (("T", "phi"), 120 + C, 0.9, "hierve"),
        (("omega", "phi"), 0.01, 0.0, "seco"),
        (("T_wb", "omega"), 15 + C, 0.02, "punto de rocío"),
    ],
)
def test_invalid_data_explain_why(pair: tuple[str, str], a: float, b: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        moist_air_state(P, pair, a, b)


def test_pressure_out_of_range() -> None:
    with pytest.raises(ValueError, match="0,4 y 10 bar"):
        state_from_T_phi(20e3, 298.15, 0.5)


def test_dead_state_needs_some_humidity() -> None:
    with pytest.raises(ValueError, match="infinita"):
        DeadState(298.15, 0.0)


# ---------------------------------------------------------------------
# Comparación con CoolProp (ASHRAE RP-1485)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(("T", "phi", "p"), GRID[:6])
def test_the_ideal_model_is_close_to_coolprop(T: float, phi: float, p: float) -> None:
    s = state_from_T_phi(p, T, phi)
    comp = coolprop_comparison(s, ("T", "phi"), T, phi)
    assert comp.error is None
    rows = {r.key: r for r in comp.rows}
    assert "T" not in rows and "phi" not in rows  # los datos no se comparan
    assert abs(rows["omega"].difference) < 0.012
    assert abs(rows["h"].difference) < 0.01 or abs(rows["h"].coolprop - s.h_J_per_kg) < 400
    assert abs(rows["T_wb"].difference) < 0.06
    assert abs(rows["v"].difference) < 0.002


def test_coolprop_comparison_at_one_atmosphere_shows_the_enhancement_factor() -> None:
    s = state_from_T_phi(P, 25 + C, 0.75)
    rows = {r.key: r for r in coolprop_comparison(s, ("T", "phi"), 25 + C, 0.75).rows}
    # f ≈ 1,004: CoolProp mete un poco más de vapor
    assert 0.003 < rows["omega"].difference < 0.006
    assert abs(rows["T_dp"].difference) < 0.02


def test_coolprop_comparison_without_dew_point_for_dry_air() -> None:
    s = state_from_T_phi(P, 25 + C, 0.0)
    rows = {r.key: r for r in coolprop_comparison(s, ("T", "phi"), 25 + C, 0.0).rows}
    assert rows["T_dp"].model is None and rows["T_dp"].difference is None


# ---------------------------------------------------------------------
# Exergía (vademecum §14.11)
# ---------------------------------------------------------------------


def test_exergy_is_zero_at_the_dead_state_and_positive_elsewhere() -> None:
    dead = DeadState(298.15, 0.5)
    assert state_from_T_phi(P, 298.15, 0.5).psi(dead) == pytest.approx(0.0, abs=1e-9)
    for T, phi in itertools.product((273.15, 298.15, 320.0), (0.05, 0.5, 1.0)):
        s = state_from_T_phi(P, T, phi)
        thermo, mixing = s.psi_parts(dead)
        assert thermo >= -1e-9 and mixing >= -1e-9
        assert s.psi(dead) == pytest.approx(thermo + mixing)


def test_exergy_equals_h_minus_T0_s_for_each_component() -> None:
    """ψ del vademecum = Σ (h − h₀) − T₀·(s − s₀) de cada gas a su presión parcial."""
    dead = DeadState(293.15, 0.4)
    s = state_from_T_phi(P, 308.15, 0.7)
    ref_air = state_from_T_omega(P, dead.T0_K, 0.0)
    # Aire seco y vapor contra el ambiente: el del vapor a p_v0.
    x_air = (
        s.h_air
        - ref_air.h_air
        - dead.T0_K
        * (
            s.s_air
            - (C_P_AIR * math.log(dead.T0_K / 273.15) - R_AIR * math.log((P - dead.p_v0_Pa) / P))
        )
    )
    x_vap = s.h_vapor - dead.h_vapor0 - dead.T0_K * (s.s_vapor - dead.s_vapor0)
    assert s.psi(dead) == pytest.approx(x_air + s.omega * x_vap, rel=2e-3)


def test_liquid_water_has_chemical_exergy_in_unsaturated_air() -> None:
    dry_env = DeadState(298.15, 0.3)
    wet_env = DeadState(298.15, 0.9)
    # R_v·T₀·ln(1/φ₀): más seco el ambiente, más exergía tiene el agua líquida
    assert water_exergy(298.15, "liquid", dry_env) > water_exergy(298.15, "liquid", wet_env) > 0
    assert water_exergy(298.15, "liquid", dry_env) / 1e3 == pytest.approx(
        R_VAPOR * 298.15 * math.log(1 / 0.3) / 1e3, rel=0.02
    )


# ---------------------------------------------------------------------
# Carta
# ---------------------------------------------------------------------


def test_chart_window_defaults_and_growth() -> None:
    w = chart_window(P)
    assert (w.T_min_K, w.T_max_K, w.omega_max) == pytest.approx((273.15, 323.15, 0.030))
    hot = state_from_T_phi(P, 85 + C, 0.3)
    cold = state_from_T_phi(P, -12 + C, 0.5)
    w2 = chart_window(P, (hot, cold))
    assert w2.T_min_K <= cold.T_K - 3 and w2.T_max_K >= hot.T_K + 5
    assert w2.omega_max >= hot.omega
    # a menor presión entra más vapor: la escala de ω crece
    assert chart_window(50e3).omega_max == pytest.approx(0.030 * P / 50e3)


def test_chart_lines_are_on_their_values() -> None:
    w = chart_window(P)
    lines = chart_lines(
        P, w, h_values=(20e3, 50e3, 80e3), v_values=(0.80, 0.85, 0.90), T_wb_values=(290.0,)
    )
    families = {line.family for line in lines}
    assert families == {"phi", "h", "v", "T_wb"}
    assert len([line for line in lines if line.family == "phi"]) == len(CHART_PHI_VALUES)
    for line in lines:
        for T, omega in zip(line.T_K, line.omega, strict=True):
            if omega > w.omega_max * 1.0001:
                continue
            s = state_from_T_omega(P, T, max(omega, 0.0))
            if line.family == "phi":
                assert s.phi == pytest.approx(line.value, rel=1e-6)
            elif line.family == "h":
                assert s.h_J_per_kg == pytest.approx(line.value, abs=1e-6)
            elif line.family == "v":
                assert s.v_m3_per_kg == pytest.approx(line.value, rel=1e-6)
            else:
                assert s.T_wb_K == pytest.approx(line.value, abs=1e-6)
    # cada recta de h nace en la saturación
    h_line = next(line for line in lines if line.family == "h" and line.value == 50e3)
    sat = saturation_humidity_ratio(h_line.T_K[0], P)
    assert h_line.omega[0] == pytest.approx(sat, rel=1e-6)


def test_chart_grid_stays_below_saturation() -> None:
    w = chart_window(P)
    grid = chart_grid(P, w)
    assert len(grid) > 1000
    for T, omega in grid:
        sat = saturation_humidity_ratio(T, P)
        assert sat is None or omega < sat
        assert w.T_min_K <= T <= w.T_max_K and 0 <= omega <= w.omega_max


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(STATE_EXAMPLES))
def test_examples_solve_and_have_a_note(name: str) -> None:
    ex = STATE_EXAMPLES[name]
    s = moist_air_state(ex.pressure_Pa, ex.pair, ex.first, ex.second)
    assert 0.0 <= s.phi <= 1.0
    assert STATE_EXAMPLE_NOTES[name]
    DeadState(ex.T0_K, ex.phi0, ex.pressure_Pa)


def test_la_quiaca_uses_the_altitude() -> None:
    ex = STATE_EXAMPLES["La Quiaca, a 3440 m de altura"]
    assert ex.pressure_Pa / 1e3 == pytest.approx(66.27, abs=0.01)


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export_is_serializable(system: str) -> None:
    s = state_from_T_phi(P, 25 + C, 0.75)
    data = moist_air_to_dict(
        s,
        system,
        pair=("T", "phi"),
        given=(25 + C, 0.75),
        dead=DeadState(298.15, 0.5),
        room_volume_m3=75.0,
    )
    json.dumps(data, ensure_ascii=False)
    assert data["omega"] == pytest.approx(s.omega)
    assert data["recinto"]["m_aire_seco_kg"] > 0
    assert data["psi"]["valor"] >= 0
