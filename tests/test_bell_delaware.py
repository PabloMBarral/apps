"""Tests de core.heat_transfer.bell_delaware (Fase 12.2)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from core.heat_transfer.bell_delaware import (
    IDEAL_BANK,
    BellDelawareGeometry,
    bd_areas,
    bd_factors,
    ideal_bank,
    solve_bell_delaware,
    tema_shell_baffle_clearance,
    tema_tube_hole_clearance,
)
from core.heat_transfer.design_common import StreamProps

IN = 0.0254
#: El agua del ejemplo 14-6 de Peters, Timmerhaus y West (2003) a 30 °C; la pared a 40 °C.
WATER_30 = StreamProps(303.15, 995.0, 4178.0, 0.000803, 0.615)
MU_WALL = 0.000657
#: La geometría que reproduce las áreas del libro: corte del 25 %, L_bb = 33,3 mm (cabezal
#: flotante de anillo partido) y 2 pares de tiras; las holguras, las de TEMA.
PETERS_GEO = BellDelawareGeometry(L_bb_m=0.0333, sealing_strips=2)
PETERS = dict(
    props=WATER_30,
    m_dot_kg_s=11.0,
    D_s_m=0.584,
    d_o_m=0.019,
    P_T_m=0.0254,
    angle=90,
    B_m=0.1524,
    baffles=22,
    N_t=405,
    geometry=PETERS_GEO,
    mu_wall_Pa_s=MU_WALL,
)


def _peters(**changes):  # noqa: ANN003, ANN202
    return solve_bell_delaware(**{**PETERS, **changes})


# ---------------------------------------------------------------------
# El banco ideal de Taborek
# ---------------------------------------------------------------------


@pytest.mark.parametrize("angle", [30, 45, 90])
@pytest.mark.parametrize("Re", [5.0, 50.0, 500.0, 5000.0, 50_000.0])
def test_ideal_bank_by_hand(angle: int, Re: float) -> None:
    c = IDEAL_BANK[angle]
    row = next(r for r in c.rows if Re >= r[0])
    a = c.a3 / (1 + 0.14 * Re**c.a4)
    b = c.b3 / (1 + 0.14 * Re**c.b4)
    ib = ideal_bank(Re, angle, 1.25)
    assert ib.j == pytest.approx(row[1] * (1.33 / 1.25) ** a * Re ** row[2], rel=1e-14)
    assert ib.f == pytest.approx(row[3] * (1.33 / 1.25) ** b * Re ** row[4], rel=1e-14)
    assert (ib.a1, ib.a2, ib.b1, ib.b2) == row[1:]


@pytest.mark.parametrize("angle", [30, 45, 90])
@pytest.mark.parametrize("Re", [10.0, 100.0, 1000.0, 10_000.0])
def test_ideal_bank_is_continuous_between_ranges(angle: int, Re: float) -> None:
    below, above = ideal_bank(Re * (1 - 1e-12), angle, 1.25), ideal_bank(Re, angle, 1.25)
    jump_j = abs(above.j / below.j - 1.0)
    jump_f = abs(above.f / below.f - 1.0)
    assert jump_f < 0.01
    # los dos saltos que trae la tabla original de Taborek
    if (angle, Re) == (45, 1000.0):
        assert jump_j == pytest.approx(0.040, abs=0.003)
    elif (angle, Re) == (90, 10_000.0):
        assert jump_j == pytest.approx(0.054, abs=0.003)
    else:
        assert jump_j < 0.01


def test_ideal_bank_rejects_unknown_angle() -> None:
    with pytest.raises(ValueError, match="Ángulo"):
        ideal_bank(1000.0, 60, 1.25)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Holguras de TEMA
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("D_s", "L_sb_in"),
    [(0.3, 1 / 8), (0.584, 3 / 16), (1.2, 1 / 4), (1.5, 5 / 16), (2.0, 3 / 8), (2.5, 7 / 16)],
)
def test_tema_shell_baffle_clearance(D_s: float, L_sb_in: float) -> None:
    assert tema_shell_baffle_clearance(D_s) == pytest.approx(L_sb_in * IN)


def test_tema_tube_hole_clearance() -> None:
    assert tema_tube_hole_clearance(0.75 * IN, 0.3) == pytest.approx(IN / 32)
    assert tema_tube_hole_clearance(0.75 * IN, 1.0) == pytest.approx(IN / 64)
    assert tema_tube_hole_clearance(1.5 * IN, 1.0) == pytest.approx(IN / 32)


# ---------------------------------------------------------------------
# Los factores contra las formas del HEDH (los ejemplos de la biblioteca ht)
# ---------------------------------------------------------------------


def _areas(**changes):  # noqa: ANN003, ANN202
    base = bd_areas(0.584, 0.019, 0.0254, 90, 0.1524, 405, PETERS_GEO)
    return replace(base, **changes)


def test_baffle_cut_and_leakage_factors() -> None:
    fc = bd_factors(_areas(F_c=0.82), 5000.0, 0.1524, 22)
    assert fc.J_c == pytest.approx(1.1404, rel=1e-12)
    # ht.baffle_leakage_Bell(1, 3, 8, "HEDH"): r_s = 1/4, r_lm = 4/8
    fc = bd_factors(_areas(r_s=0.25, r_lm=0.5), 5000.0, 0.1524, 22)
    assert fc.J_l == pytest.approx(0.5530236260777133, rel=1e-12)


def test_bundle_bypass_factor() -> None:
    # ht.bundle_bypassing_Bell(0.5, 5, 25, method="HEDH"): F_sbp = 0,5 y r_ss = 5/25
    a = _areas(F_sbp=0.5, r_ss=0.2)
    assert bd_factors(a, 5000.0, 0.1524, 22).J_b == pytest.approx(0.8483210970579099, rel=1e-12)
    assert bd_factors(a, 50.0, 0.1524, 22).J_b == pytest.approx(0.8372305924553625, rel=1e-12)
    # con r_ss ≥ ½, las tiras cortan todo el bypass
    a = _areas(F_sbp=0.5, r_ss=0.5)
    fc = bd_factors(a, 5000.0, 0.1524, 22)
    assert fc.J_b == 1.0 and fc.R_b == 1.0


def test_unequal_spacing_factor() -> None:
    # ht.unequal_baffle_spacing_Bell(16, 0.1, 0.15, 0.15)
    a = _areas(B_in_m=0.15, B_out_m=0.15)
    assert bd_factors(a, 5000.0, 0.1, 16).J_s == pytest.approx(0.9640087802805195, rel=1e-12)
    assert bd_factors(a, 50.0, 0.1, 16).J_s == pytest.approx(0.97893007745605, rel=1e-12)
    fc = bd_factors(a, 5000.0, 0.1, 16)
    assert fc.R_s_in == pytest.approx((0.1 / 0.15) ** 1.8, rel=1e-12)


def test_laminar_correction_factor() -> None:
    # ht.laminar_correction_Bell(Re, N_c): N_c = (N_tcc + N_tcw)·(N_b + 1) = 8·10 = 80
    a = _areas(N_tcc=6.0, N_tcw=2.0)
    assert bd_factors(a, 30.0, 0.1524, 9).J_r == pytest.approx(0.7267995454361379, rel=1e-12)
    assert bd_factors(a, 15.0, 0.1524, 9).J_r == pytest.approx(0.6877709090698719, rel=1e-12)
    assert bd_factors(a, 150.0, 0.1524, 9).J_r == 1.0
    # nunca menos de 0,4: ht.laminar_correction_Bell(10, 2000)
    a = _areas(N_tcc=150.0, N_tcw=50.0)
    assert bd_factors(a, 10.0, 0.1524, 9).J_r == pytest.approx(0.4)


def test_pressure_drop_factors_by_hand() -> None:
    a = _areas(r_s=0.3, r_lm=0.4, F_sbp=0.2, r_ss=0.1)
    fc = bd_factors(a, 5000.0, 0.1524, 22)
    p = -0.15 * 1.3 + 0.8
    assert fc.R_l == pytest.approx(math.exp(-1.33 * 1.3 * 0.4**p), rel=1e-12)
    assert fc.R_b == pytest.approx(math.exp(-3.7 * 0.2 * (1 - 0.2 ** (1 / 3))), rel=1e-12)
    fc = bd_factors(a, 50.0, 0.1524, 22)
    assert fc.R_b == pytest.approx(math.exp(-4.5 * 0.2 * (1 - 0.2 ** (1 / 3))), rel=1e-12)


# ---------------------------------------------------------------------
# Peters, Timmerhaus y West (2003), ejemplo 14-6
# ---------------------------------------------------------------------


def test_peters_14_6_areas_and_factors() -> None:
    r = _peters()
    a, fc = r.areas, r.factors
    # las áreas del libro: S_m = 0,0255 m² y S_w = 0,03298 m²
    assert a.S_m_m2 == pytest.approx(0.0255, rel=0.005)
    assert a.S_w_m2 == pytest.approx(0.03298, rel=0.005)
    # el h ideal: 5807 W/(m²·K)
    assert r.h_ideal_W_per_m2K == pytest.approx(5807.0, rel=0.01)
    # J_c = 1,013, J_l = 0,56 y J_b = 0,935; el libro no corrige por los extremos
    assert fc.J_c == pytest.approx(1.013, rel=0.02)
    assert fc.J_l == pytest.approx(0.56, rel=0.05)
    assert fc.J_b == pytest.approx(0.935, rel=0.01)
    assert fc.J_s == 1.0 and fc.J_r == 1.0
    # h_o = 5807·1,013·0,56·0,935 = 3080 W/(m²·K)
    assert r.h_W_per_m2K == pytest.approx(3080.0, rel=0.05)
    # el Δp del libro (9,48 kPa) no se reproduce: R_l y R_b dan los valores típicos
    assert 0.3 < fc.R_l < 0.45 and 0.75 < fc.R_b < 0.85
    assert r.dp_Pa == pytest.approx(5912.5, rel=1e-3)


def test_peters_by_hand() -> None:
    r = _peters()
    a, fc, ib = r.areas, r.factors, r.ideal
    D_s, d_o, P_T, B = 0.584, 0.019, 0.0254, 0.1524
    D_ctl = D_s - 0.0333 - d_o
    assert a.S_m_m2 == pytest.approx(B * (0.0333 + D_ctl / P_T * (P_T - d_o)), rel=1e-12)
    theta_ds = 2 * math.acos(0.5)
    theta_ctl = 2 * math.acos(D_s * 0.5 / D_ctl)
    F_w = (theta_ctl - math.sin(theta_ctl)) / (2 * math.pi)
    assert a.F_w == pytest.approx(F_w, rel=1e-12)
    S_wg = math.pi / 4 * D_s**2 * (theta_ds - math.sin(theta_ds)) / (2 * math.pi)
    assert a.S_w_m2 == pytest.approx(S_wg - 405 * F_w * math.pi * d_o**2 / 4, rel=1e-12)
    assert a.N_tcc == pytest.approx(D_s / P_T * 0.5, rel=1e-12)
    assert a.N_tcw == pytest.approx(0.8 / P_T * (D_s * 0.25 - (D_s - D_ctl) / 2), rel=1e-12)
    L_sb, L_tb = 3 / 16 * IN, IN / 32
    assert a.S_sb_m2 == pytest.approx(D_s * L_sb * (2 * math.pi - theta_ds) / 4, rel=1e-12)
    assert a.S_tb_m2 == pytest.approx(
        math.pi / 4 * ((d_o + L_tb) ** 2 - d_o**2) * 405 * (1 - F_w), rel=1e-12
    )
    G = 11.0 / a.S_m_m2
    phi = (0.000803 / MU_WALL) ** 0.14
    Pr = 4178.0 * 0.000803 / 0.615
    assert r.Re == pytest.approx(d_o * G / 0.000803, rel=1e-12)
    assert r.h_ideal_W_per_m2K == pytest.approx(ib.j * 4178.0 * G * Pr ** (-2 / 3) * phi)
    assert r.h_W_per_m2K == pytest.approx(r.h_ideal_W_per_m2K * fc.J_total, rel=1e-12)
    dp_bi = 2 * ib.f * a.N_tcc * G**2 / (995.0 * phi)
    dp_wi = (2 + 0.6 * a.N_tcw) * 11.0**2 / (2 * 995.0 * a.S_m_m2 * a.S_w_m2)
    dp = (
        21 * dp_bi * fc.R_b * fc.R_l
        + 22 * dp_wi * fc.R_l
        + 2 * dp_bi * (1 + a.N_tcw / a.N_tcc) * fc.R_b
    )
    assert r.dp_bi_Pa == pytest.approx(dp_bi, rel=1e-12)
    assert r.dp_wi_Pa == pytest.approx(dp_wi, rel=1e-12)
    assert r.dp_Pa == pytest.approx(dp, rel=1e-12)
    assert r.dp_Pa == pytest.approx(r.dp_cross_Pa + r.dp_window_Pa + r.dp_ends_Pa)


def test_ideal_zones_without_leakage_or_bypass() -> None:
    r = _peters()
    assert r.dp_cross_ideal_Pa > r.dp_cross_Pa
    assert r.dp_window_ideal_Pa > r.dp_window_Pa
    assert r.dp_ends_ideal_Pa > r.dp_ends_Pa
    # sin holguras, sin bypass y con extremos iguales, todos los factores valen 1 (salvo J_c)
    g = BellDelawareGeometry(L_bb_m=0.0, L_sb_m=0.0, L_tb_m=0.0)
    r0 = _peters(geometry=g)
    fc = r0.factors
    assert fc.J_l == 1.0 and fc.R_l == 1.0
    assert fc.J_b == 1.0 and fc.R_b == 1.0
    assert fc.J_s == 1.0 and fc.R_s_in == 1.0 and fc.R_s_out == 1.0
    assert r0.dp_Pa == pytest.approx(
        r0.dp_cross_ideal_Pa + r0.dp_window_ideal_Pa + r0.dp_ends_ideal_Pa
    )


def test_sealing_strips_recover_the_bypass() -> None:
    hs = [_peters(geometry=replace(PETERS_GEO, sealing_strips=n)).factors.J_b for n in range(7)]
    assert all(b > a for a, b in zip(hs, hs[1:6], strict=False))
    assert hs[6] == 1.0  # 6 pares ≥ N_tcc/2 = 5,75


def test_unequal_end_spacing() -> None:
    ends = (3.66 - 21 * 0.1524) / 2
    r = _peters(geometry=replace(PETERS_GEO, B_in_m=ends, B_out_m=ends))
    assert r.factors.J_s < 1.0
    assert r.factors.R_s_in == pytest.approx((0.1524 / ends) ** 1.8)
    assert r.h_W_per_m2K < _peters().h_W_per_m2K
    assert r.dp_ends_Pa < _peters().dp_ends_Pa


def test_laminar_branch() -> None:
    oil = StreamProps(323.15, 870.0, 2000.0, 0.05, 0.13)
    r = solve_bell_delaware(
        oil,
        2.0,
        0.387,
        0.01905,
        0.0254,
        30,
        0.15,
        40,
        176,
        BellDelawareGeometry(L_bb_m=0.012),
        0.08,
    )
    fc, a = r.factors, r.areas
    assert r.Re < 100 and fc.laminar
    assert fc.C_bh == 1.35 and fc.C_bp == 4.5 and fc.n == pytest.approx(1 / 3)
    assert fc.J_r < 1.0
    rho, mu, m = 870.0, 0.05, 2.0
    dp_wi = 26 * mu * m / (rho * math.sqrt(a.S_m_m2 * a.S_w_m2)) * (
        a.N_tcw / (0.0254 - 0.01905) + 0.15 / a.D_w_m**2
    ) + m**2 / (rho * a.S_m_m2 * a.S_w_m2)
    assert r.dp_wi_Pa == pytest.approx(dp_wi, rel=1e-12)


@pytest.mark.parametrize("angle", [30, 45, 90])
def test_pitch_geometry_by_angle(angle: int) -> None:
    a = bd_areas(0.584, 0.019, 0.0254, angle, 0.1524, 405, PETERS_GEO)
    L_pp = {30: 0.866, 45: 0.707, 90: 1.0}[angle] * 0.0254
    L_tp_eff = {30: 1.0, 45: 0.707, 90: 1.0}[angle] * 0.0254
    assert a.L_pp_m == pytest.approx(L_pp)
    assert a.L_tp_eff_m == pytest.approx(L_tp_eff)
    assert a.N_tcc == pytest.approx(0.584 / L_pp * 0.5)


# ---------------------------------------------------------------------
# Errores y avisos
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"baffle_cut": 0.5}, "superponen"),
        ({"baffle_cut": 0.0}, "superponen"),
        ({"L_bb_m": -0.001}, "negativa"),
        ({"L_bb_m": 0.6}, "no entra"),
        ({"L_sb_m": -0.001}, "negativas"),
        ({"L_tb_m": -0.001}, "negativas"),
        ({"sealing_strips": -1}, "negativos"),
        ({"L_pl_m": -0.01}, "negativo"),
        ({"B_in_m": 0.0}, "positivas"),
    ],
)
def test_geometry_errors(changes: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _peters(geometry=replace(PETERS_GEO, **changes))


def test_needs_a_baffle() -> None:
    with pytest.raises(ValueError, match="al menos un bafle"):
        _peters(baffles=0)


def test_warnings() -> None:
    assert not _peters().warnings
    r = _peters(geometry=replace(PETERS_GEO, baffle_cut=0.10))
    assert any("corte" in w for w in r.warnings)
    r = _peters(m_dot_kg_s=1500.0)
    assert any("Taborek" in w for w in r.warnings)
    r = _peters(geometry=replace(PETERS_GEO, L_sb_m=0.03, L_tb_m=0.003))
    assert any("fugas" in w for w in r.warnings)
    # un corte que no llega a los tubos: L_bb grande y corte chico
    r = _peters(geometry=replace(PETERS_GEO, baffle_cut=0.15, L_bb_m=0.2))
    assert r.areas.N_tcw == 0.0 and r.areas.F_w == 0.0
    assert any("no llega a los tubos" in w for w in r.warnings)
