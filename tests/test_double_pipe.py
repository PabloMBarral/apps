"""Tests del doble tubo (Fase 12.1) contra Incropera 11.1 y Çengel y Ghajar 11-1."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from core.heat_transfer.design_common import (
    DesignStream,
    FluidProperties,
    darcy_f,
    overall_u_per_length,
    stream_props,
    tube_flow,
)
from core.heat_transfer.double_pipe import (
    DOUBLE_PIPE_EXAMPLES,
    annulus_flow,
    annulus_friction_Re,
    annulus_nu_laminar,
    double_pipe_notes,
    double_pipe_to_dict,
    solve_double_pipe,
    swap_sides,
)

C = 273.15
INCROPERA = DOUBLE_PIPE_EXAMPLES[
    "Enfriador de aceite de una turbina de gas (Incropera, ejemplo 11.1)"
].inputs
CENGEL = DOUBLE_PIPE_EXAMPLES[
    "U de un doble tubo con agua y aceite (Çengel y Ghajar, ejemplo 11-1)"
].inputs
PROCESS = DOUBLE_PIPE_EXAMPLES[
    "Calentador de agua de proceso (caños de acero, con ensuciamiento)"
].inputs
#: El agua de Incropera 11.1 a 35 °C (tabla A.6).
WATER_35 = FluidProperties(994.0, 4178.0, 725e-6, 0.625, name="agua")


# ---------------------------------------------------------------------
# El anular
# ---------------------------------------------------------------------


def test_annulus_table() -> None:
    # Incropera, tabla 8.2 (Kays y Perkins)
    assert annulus_nu_laminar(0.25) == pytest.approx(7.37)
    assert annulus_nu_laminar(1.0) == pytest.approx(4.86)
    # el ejemplo 11.1: d_o/D = 25/45 (el libro lee 5,63)
    assert annulus_nu_laminar(25 / 45) == pytest.approx(5.642, abs=1e-3)
    # Çengel 11-1: 2/3
    assert annulus_nu_laminar(2 / 3) == pytest.approx(5.447, abs=1e-3)
    assert annulus_nu_laminar(0.01) == pytest.approx(17.46)
    values = [annulus_nu_laminar(r) for r in (0.05, 0.1, 0.3, 0.6, 0.9)]
    assert values == sorted(values, reverse=True)


def test_annulus_laminar_friction() -> None:
    # f·Re del anular concéntrico (Shah y London): 64 → 96
    assert annulus_friction_Re(0.0) == 64.0
    assert annulus_friction_Re(1.0) == 96.0
    assert annulus_friction_Re(0.5) == pytest.approx(95.25, abs=0.01)
    assert annulus_friction_Re(1e-6) == pytest.approx(64.0, rel=0.08)
    values = [annulus_friction_Re(k) for k in (0.01, 0.05, 0.1, 0.3, 0.6, 0.99)]
    assert values == sorted(values)


def test_annulus_flow_turbulent_uses_hydraulic_diameter() -> None:
    props = stream_props(DesignStream(300.0, 1.0, fluid="Water", p_Pa=2e5), 300.0, "x")
    a = annulus_flow(props, 2.0, 0.025, 0.05, True, None, "dittus_boelter")
    assert a.D_h_m == pytest.approx(0.025)
    assert a.area_m2 == pytest.approx(math.pi * (0.05**2 - 0.025**2) / 4)
    assert a.Re == pytest.approx(2.0 * 0.025 / (a.area_m2 * props.mu_Pa_s))
    assert a.h_W_per_m2K == pytest.approx(
        0.023 * a.Re**0.8 * props.Pr**0.4 * props.k_W_per_mK / 0.025
    )


# ---------------------------------------------------------------------
# Contra los libros
# ---------------------------------------------------------------------


def test_incropera_11_1() -> None:
    r = solve_double_pipe(INCROPERA)
    assert r.T_cold_out_K - C == pytest.approx(40.2, abs=0.05)
    assert r.Q_W == pytest.approx(8524.0, rel=1e-3)
    assert r.tube.correlation == "dittus_boelter"
    assert r.tube.h_W_per_m2K == pytest.approx(2250.0, rel=0.01)
    assert r.annulus.correlation == "kays_perkins"
    assert r.annulus.Re == pytest.approx(56.0, abs=0.5)
    assert r.annulus.h_W_per_m2K == pytest.approx(38.8, rel=0.005)
    assert r.U_o_W_per_m2K == pytest.approx(38.1, rel=0.006)
    assert r.exchanger.dT_lm_K == pytest.approx(43.2, abs=0.05)
    assert r.L_m == pytest.approx(65.9, rel=0.005)
    # el agua del libro (a 35 °C): h_i = 2250 exacto
    book = solve_double_pipe(replace(INCROPERA, cold=replace(INCROPERA.cold, properties=WATER_35)))
    assert book.tube.Re == pytest.approx(14_050, rel=1e-3)
    assert book.tube.h_W_per_m2K == pytest.approx(2250.0, rel=3e-3)


def test_cengel_11_1() -> None:
    r = solve_double_pipe(CENGEL)
    # medias del libro: 45 °C el agua y 80 °C el aceite
    assert r.tube_props.T_K - C == pytest.approx(45.0, abs=1e-6)
    assert r.annulus_props.T_K - C == pytest.approx(80.0, abs=0.1)
    assert r.tube.velocity_m_s == pytest.approx(1.61, abs=0.01)
    assert r.annulus.Re == pytest.approx(630, rel=0.01)
    assert r.annulus.h_W_per_m2K == pytest.approx(75.2, abs=0.2)
    assert r.U_o_W_per_m2K == pytest.approx(74.5, rel=0.005)


# ---------------------------------------------------------------------
# Consistencia
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(DOUBLE_PIPE_EXAMPLES))
def test_balances_close(name: str) -> None:
    r = solve_double_pipe(DOUBLE_PIPE_EXAMPLES[name].inputs)
    ex = r.exchanger
    i = r.inputs
    Q_hot = i.hot.m_dot_kg_s * ex.hot.cp_J_per_kgK * (i.hot.T_in_K - ex.hot.T_out_K)
    Q_cold = i.cold.m_dot_kg_s * ex.cold.cp_J_per_kgK * (ex.cold.T_out_K - i.cold.T_in_K)
    assert Q_hot == pytest.approx(r.Q_W, rel=1e-9)
    assert Q_cold == pytest.approx(r.Q_W, rel=1e-9)
    # Q̇ = U·A·ΔT_ml (contracorriente, F = 1)
    assert r.U_o_W_per_m2K * r.area_m2 * ex.dT_lm_K == pytest.approx(r.Q_W, rel=1e-6)
    # las medias son las de las salidas
    hot_mean = 0.5 * (i.hot.T_in_K + ex.hot.T_out_K)
    assert (r.annulus_props if r.annulus_stream == "hot" else r.tube_props).T_K == pytest.approx(
        hot_mean, abs=1e-6
    )


def test_sizing_then_rating_gives_back_the_outlets() -> None:
    sized = solve_double_pipe(PROCESS)
    # verificar un equipo con exactamente ese largo: 1 horquilla de L/2
    rated = solve_double_pipe(
        replace(PROCESS, problem="rating", hairpins=1, leg_length_m=sized.L_m / 2.0)
    )
    assert rated.T_cold_out_K == pytest.approx(PROCESS.target_value, abs=1e-5)
    assert rated.T_hot_out_K == pytest.approx(sized.T_hot_out_K, abs=1e-5)
    assert rated.U_o_W_per_m2K == pytest.approx(sized.U_o_W_per_m2K, rel=1e-6)


def test_fouling_and_wall() -> None:
    r = solve_double_pipe(PROCESS)
    kinds = [x.kind for x in r.u_result.resistances]
    assert kinds.count("ensuciamiento") == 2 and "pared" in kinds
    assert r.U_o_clean_W_per_m2K > r.U_o_W_per_m2K
    # el U contra las resistencias, por metro y referido a π·d_o
    R = sum(x.R_K_per_W for x in r.u_result.resistances)
    assert 1.0 / (R * math.pi * PROCESS.d_o_m) == pytest.approx(r.U_o_W_per_m2K)
    # pared delgada: sin resistencia de la pared
    thin = solve_double_pipe(INCROPERA)
    assert "pared" not in [x.kind for x in thin.u_result.resistances]


def test_parallel_flow_needs_more_length() -> None:
    counter = solve_double_pipe(PROCESS)
    parallel = solve_double_pipe(replace(PROCESS, flow="parallel"))
    assert parallel.L_m > counter.L_m


def test_hairpins_and_overdesign() -> None:
    r = solve_double_pipe(INCROPERA)
    assert r.hairpins == 6  # 65,7 m con ramas de 6 m: 6 horquillas (72 m)
    assert r.L_installed_m == pytest.approx(72.0)
    assert r.overdesign == pytest.approx(72.0 / r.L_m - 1.0)
    rated = solve_double_pipe(replace(INCROPERA, problem="rating", hairpins=6))
    assert rated.L_m == pytest.approx(72.0)
    assert rated.T_hot_out_K < 60.0 + C  # más largo que el necesario: enfría más


def test_swapping_sides() -> None:
    swapped = swap_sides(INCROPERA)
    assert swapped is not None
    assert swapped.tube_stream == "hot"
    # el aceite por el tubo de 25 mm sigue laminar y hace falta casi el doble de largo
    assert swapped.L_m > 1.5 * solve_double_pipe(INCROPERA).L_m


def test_notes() -> None:
    notes = double_pipe_notes(solve_double_pipe(INCROPERA))
    assert any("convección en el anular" in n for n in notes)
    assert any("horquillas" in n for n in notes)
    notes = double_pipe_notes(solve_double_pipe(PROCESS))
    assert any("ensuciamiento" in n for n in notes)
    # el aceite a 2,4 m/s en un anular de 1 cm: mucho más Δp que lo admitido
    notes = double_pipe_notes(solve_double_pipe(CENGEL))
    assert any("10 psi" in n for n in notes)


def test_pressure_drops() -> None:
    r = solve_double_pipe(PROCESS)
    t, p = r.tube, r.tube_props
    assert r.dp_tube_Pa == pytest.approx(
        darcy_f(t.Re) * r.L_m / PROCESS.d_i_m * p.rho_kg_per_m3 * t.velocity_m_s**2 / 2
    )
    assert r.dp_annulus_Pa > 0 and r.dp_returns_Pa > 0


# ---------------------------------------------------------------------
# Lo común
# ---------------------------------------------------------------------


def test_tube_flow_correlations() -> None:
    props = stream_props(DesignStream(300.0, 1.0, fluid="Water", p_Pa=2e5), 300.0, "x")
    lam = tube_flow(props, 0.005, 0.02, 2.0, True, None, "auto")
    assert lam.regime == "laminar" and lam.Nu == pytest.approx(3.66)
    assert lam.f == pytest.approx(64.0 / lam.Re)
    turb = tube_flow(props, 0.5, 0.02, 2.0, True, None, "auto")
    assert turb.correlation == "gnielinski" and turb.regime == "turbulento"
    bad = tube_flow(props, 0.5, 0.02, 2.0, True, None, "tube_laminar")
    assert any("no es laminar" in w for w in bad.warnings)
    with pytest.raises(ValueError, match="desconocida"):
        tube_flow(props, 0.5, 0.02, 2.0, True, None, "nada")


def test_overall_u_per_length() -> None:
    u = overall_u_per_length(1000.0, 500.0, 0.02, 0.025, 50.0, 1e-4, 2e-4)
    expected = (
        1 / (1000 * math.pi * 0.02)
        + 1e-4 / (math.pi * 0.02)
        + math.log(0.025 / 0.02) / (2 * math.pi * 50)
        + 2e-4 / (math.pi * 0.025)
        + 1 / (500 * math.pi * 0.025)
    )
    assert u.R_total_K_per_W == pytest.approx(expected)
    assert u.A_o_m2 == pytest.approx(math.pi * 0.025)


# ---------------------------------------------------------------------
# Errores
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"D_m": 0.025}, "no queda anular"),
        ({"d_o_m": 0.02}, "no puede ser menor"),
        ({"leg_length_m": 0.0}, "rama"),
        ({"tube_correlation": "nada"}, "desconocida"),
        ({"target_value": 20.0 + C}, "tiene que quedar entre"),
    ],
)
def test_student_errors(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_double_pipe(replace(INCROPERA, **change))


def test_stream_errors() -> None:
    with pytest.raises(ValueError, match="más caliente"):
        solve_double_pipe(replace(INCROPERA, cold=replace(INCROPERA.cold, T_in_K=120.0 + C)))
    with pytest.raises(ValueError, match="caudal"):
        solve_double_pipe(replace(INCROPERA, cold=replace(INCROPERA.cold, m_dot_kg_s=0.0)))
    bad = FluidProperties(800.0, 2000.0, -1.0, 0.1)
    with pytest.raises(ValueError, match="positivas"):
        solve_double_pipe(replace(INCROPERA, hot=replace(INCROPERA.hot, properties=bad)))
    # agua a 1 atm calentada por encima de 100 °C: hierve en el intercambiador
    with pytest.raises(ValueError, match="hierve"):
        solve_double_pipe(
            replace(
                PROCESS,
                hot=replace(PROCESS.hot, T_in_K=180.0 + C, p_Pa=12e5),
                cold=replace(PROCESS.cold, p_Pa=101_325.0),
                target_value=110.0 + C,
            )
        )


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export_is_json(system: str) -> None:
    r = solve_double_pipe(INCROPERA)
    data = double_pipe_to_dict(r, system)  # type: ignore[arg-type]
    json.dumps(data)
    assert data["horquillas"] == 6
    assert (
        data["tubo"]["Delta_p"]["unidad"] == {"SI": "Pa", "Técnico": "kPa", "Inglés": "psi"}[system]
    )
    assert "aceite" in data["por_el_anular"]
