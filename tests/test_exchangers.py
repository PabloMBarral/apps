"""Tests de core.heat_transfer.exchangers (Fase 8.2): ε-NTU, LMTD con F, U y exergía."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import replace

import pytest

from core.export import dict_to_csv
from core.heat_transfer import exchangers as hx
from core.heat_transfer.exchangers import (
    Arrangement,
    FourTemperatureInputs,
    OverallUInputs,
    RatingInputs,
    SizingInputs,
    Stream,
    correction_factor,
    effectiveness,
    eps_counter,
    eps_cross_cmax_mixed,
    eps_cross_cmin_mixed,
    eps_cross_unmixed,
    eps_cross_unmixed_approx,
    eps_parallel,
    eps_shell,
    lmtd,
    max_effectiveness,
    ntu_from_effectiveness,
    overall_u,
    solve_four_temperatures,
    solve_rating,
    solve_sizing,
)

C = 273.15
SYSTEMS = ("SI", "Técnico", "Inglés")
ARRANGEMENTS = (
    Arrangement("parallel"),
    Arrangement("counter"),
    Arrangement("shell", 1),
    Arrangement("shell", 2),
    Arrangement("shell", 3),
    Arrangement("shell", 4),
    Arrangement("cross_unmixed"),
    Arrangement("cross_mixed", mixed="hot"),
    Arrangement("cross_mixed", mixed="cold"),
)


def _example(table: dict[str, hx.ExchangerExample], prefix: str) -> hx.ExchangerExample:
    return next(e for k, e in table.items() if k.startswith(prefix))


def bowman_F(P: float, R: float) -> float:
    """F de 1 casco y 2n pasos de tubo, forma cerrada (Bowman, Mueller y Nagle, 1940)."""
    s = math.sqrt(R * R + 1.0)
    num = s * math.log((1.0 - P) / (1.0 - P * R))
    den = (R - 1.0) * math.log((2.0 - P * (R + 1.0 - s)) / (2.0 - P * (R + 1.0 + s)))
    return num / den


# ---------------------------------------------------------------------
# Relaciones ε-NTU
# ---------------------------------------------------------------------


def test_closed_forms_and_limits() -> None:
    for ntu in (0.1, 1.0, 3.0):
        assert eps_counter(ntu, 1.0) == pytest.approx(ntu / (1.0 + ntu))
        # continuidad en C_r → 1
        assert eps_counter(ntu, 1.0 - 1e-7) == pytest.approx(eps_counter(ntu, 1.0), rel=1e-6)
        assert eps_shell(ntu, 1.0 - 1e-7, 3) == pytest.approx(eps_shell(ntu, 1.0, 3), rel=1e-6)
        # con C_r = 0 todos dan 1 − e^(−NTU)
        for arr in ARRANGEMENTS:
            assert effectiveness(arr, ntu, 0.0) == pytest.approx(1.0 - math.exp(-ntu))
        assert eps_parallel(ntu, 0.0) == pytest.approx(1.0 - math.exp(-ntu))
        assert eps_cross_unmixed(ntu, 0.0) == pytest.approx(1.0 - math.exp(-ntu))
        # C_máx o C_mín mezclado coinciden con C_r = 1
        assert eps_cross_cmax_mixed(ntu, 1.0) == pytest.approx(eps_cross_cmin_mixed(ntu, 1.0))
    # contracorriente ≥ casco ≥ paralelo con el mismo NTU y C_r
    for ntu, cr in ((0.5, 0.5), (2.0, 0.8), (4.0, 1.0)):
        e_cc, e_s1, e_pf = eps_counter(ntu, cr), eps_shell(ntu, cr, 1), eps_parallel(ntu, cr)
        assert e_cc > e_s1 > e_pf
        # más cascos se acercan al contracorriente
        assert eps_shell(ntu, cr, 1) < eps_shell(ntu, cr, 2) < eps_shell(ntu, cr, 4) < e_cc


def test_cross_flow_series_against_kays_and_london() -> None:
    """La serie exacta de los dos fluidos sin mezclar (Mason, 1955)."""
    # Kays y London (1984): NTU = 1, C_r = 1 → ε = 0,476
    assert eps_cross_unmixed(1.0, 1.0) == pytest.approx(0.4762, abs=1e-4)
    # la aproximación de las tablas se aparta menos de 4 %
    worst = 0.0
    for cr in (0.25, 0.5, 0.75, 1.0):
        for ntu in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
            exact, approx = eps_cross_unmixed(ntu, cr), eps_cross_unmixed_approx(ntu, cr)
            worst = max(worst, abs(approx / exact - 1.0))
    assert 0.02 < worst < 0.04
    # con C_r chico tiende a la de un fluido que cambia de fase
    assert eps_cross_unmixed(2.0, 1e-6) == pytest.approx(1.0 - math.exp(-2.0), rel=1e-5)


@pytest.mark.parametrize("arr", ARRANGEMENTS, ids=lambda a: a.name)
@pytest.mark.parametrize("cr", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ntu_round_trip(arr: Arrangement, cr: float) -> None:
    for mixed_is_cmin in (False, True):
        for ntu in (0.1, 0.5, 1.0, 2.0, 4.0):
            eps = effectiveness(arr, ntu, cr, mixed_is_cmin)
            assert 0.0 < eps < max_effectiveness(arr, cr, mixed_is_cmin)
            back = ntu_from_effectiveness(arr, eps, cr, mixed_is_cmin)
            assert back == pytest.approx(ntu, rel=1e-7)


@pytest.mark.parametrize("arr", ARRANGEMENTS, ids=lambda a: a.name)
def test_effectiveness_grows_to_its_maximum(arr: Arrangement) -> None:
    for cr in (0.3, 0.7, 1.0):
        for mixed_is_cmin in (False, True):
            values = [effectiveness(arr, x, cr, mixed_is_cmin) for x in (0.5, 1, 2, 4, 8)]
            assert values == sorted(values)
            eps_max = max_effectiveness(arr, cr, mixed_is_cmin)
            assert values[-1] < eps_max
            if arr.kind not in ("counter", "cross_unmixed"):  # llegan a 1 muy despacio
                assert effectiveness(arr, 60.0, cr, mixed_is_cmin) == pytest.approx(
                    eps_max, rel=1e-6
                )


def test_unreachable_effectiveness_is_explained() -> None:
    arr = Arrangement("shell", 1)
    eps_max = max_effectiveness(arr, 0.75)
    assert eps_max == pytest.approx(2.0 / 3.0, abs=1e-12)  # 11-5 de Cengel y Ghajar
    with pytest.raises(ValueError, match="no puede pasar de 0,6667"):
        ntu_from_effectiveness(arr, 0.70, 0.75)
    with pytest.raises(ValueError, match="contracorriente o con más pasos de casco"):
        ntu_from_effectiveness(Arrangement("parallel"), 0.6, 1.0)
    with pytest.raises(ValueError, match="no es razonable"):
        ntu_from_effectiveness(Arrangement("counter"), 0.9995, 1.0)


def test_correction_factor() -> None:
    # 1 casco: la forma cerrada de Bowman, Mueller y Nagle (1940)
    for P, R in ((0.2, 0.5), (0.4, 1.2), (0.5, 0.8), (0.3, 2.0), (0.25, 3.0)):
        cr = R if R <= 1.0 else 1.0 / R
        eps = P if R <= 1.0 else P * R
        assert correction_factor(Arrangement("shell"), eps, cr) == pytest.approx(
            bowman_F(P, R), rel=1e-9
        )
    # contracorriente, paralelo y cambio de fase: F = 1
    assert correction_factor(Arrangement("counter"), 0.6, 0.8) == 1.0
    assert correction_factor(Arrangement("cross_unmixed"), 0.6, 0.0) == 1.0
    # F ≤ 1 y crece con los pasos de casco
    for cr in (0.5, 1.0):
        F1 = correction_factor(Arrangement("shell", 1), 0.5, cr)
        F2 = correction_factor(Arrangement("shell", 2), 0.5, cr)
        assert F1 < F2 < 1.0
    # el gráfico: F → 1 con P → 0 y decrece con P
    curves = hx.f_curves(Arrangement("shell", 1))
    for P_list, F_list in curves.values():
        assert P_list and F_list[0] > 0.97 and F_list == sorted(F_list, reverse=True)
        assert min(F_list) >= 0.5


def test_effectiveness_curves() -> None:
    curves = hx.effectiveness_curves(Arrangement("shell", 1))
    xs, ys = curves[0.0]
    assert all(y == pytest.approx(1.0 - math.exp(-x)) for x, y in zip(xs, ys, strict=True))
    xs, ys = curves[1.0]
    assert ys == sorted(ys) and ys[0] == 0.0


def test_lmtd() -> None:
    assert lmtd(30.0, 20.0) == pytest.approx(10.0 / math.log(1.5))
    assert lmtd(20.0, 20.0) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="se cruzan"):
        lmtd(10.0, 0.0)


# ---------------------------------------------------------------------
# Ejemplos de Cengel y Ghajar (cap. 11) e Incropera
# ---------------------------------------------------------------------


def test_cengel_11_2_fouling() -> None:
    r = overall_u(_example(hx.U_EXAMPLES, "Doble tubo de acero").inputs)  # type: ignore[arg-type]
    assert r.R_total_K_per_W == pytest.approx(0.0532, rel=2e-3)
    assert r.U_i_W_per_m2K == pytest.approx(399.0, abs=0.5)
    assert r.U_o_W_per_m2K == pytest.approx(315.0, abs=0.5)
    assert r.U_i_W_per_m2K * r.A_i_m2 == pytest.approx(r.U_o_W_per_m2K * r.A_o_m2)
    assert sum(r.share(x) for x in r.resistances) == pytest.approx(1.0)
    assert r.U_i_clean_W_per_m2K > r.U_i_W_per_m2K


def test_cengel_11_4_and_11_8_geothermal_water() -> None:
    sizing = solve_sizing(_example(hx.SIZING_EXAMPLES, "Agua calentada").inputs)  # type: ignore[arg-type]
    assert sizing.Q_W == pytest.approx(301.0e3, rel=1e-3)
    assert sizing.hot.T_out_K - C == pytest.approx(125.1, abs=0.05)
    assert sizing.dT_lm_K == pytest.approx(91.9, abs=0.1)
    assert sizing.tube_length_m == pytest.approx(109.0, rel=5e-3)
    # 11-8: el mismo con ε-NTU
    assert sizing.effectiveness == pytest.approx(0.4286, abs=1e-4)
    assert sizing.NTU == pytest.approx(0.6524, abs=1e-4)
    rating = solve_rating(
        replace(
            _example(hx.RATING_EXAMPLES, "Agua calentada").inputs,  # type: ignore[arg-type]
            A_m2=sizing.A_m2,
        )
    )
    assert rating.cold.T_out_K == pytest.approx(80.0 + C, abs=1e-6)


def test_cengel_11_7_maximum_heat() -> None:
    r = solve_rating(
        RatingInputs(
            Arrangement("counter"),
            Stream("agua caliente", 70.0 + C, 2.0, cp_J_per_kgK=4180.0),
            Stream("agua fría", 10.0 + C, 8.0, cp_J_per_kgK=4180.0),
            1000.0,
            1.0,
        )
    )
    assert r.Q_max_W == pytest.approx(502.0e3, rel=1e-3)
    assert r.cmin_side == "hot"


def test_cengel_11_9_oil_and_water() -> None:
    r = solve_rating(_example(hx.RATING_EXAMPLES, "Aceite enfriado").inputs)  # type: ignore[arg-type]
    assert r.A_m2 == pytest.approx(1.76, abs=0.005)
    assert r.NTU == pytest.approx(0.853, abs=1e-3)
    assert r.C_r == pytest.approx(0.764, abs=1e-3)
    assert r.effectiveness == pytest.approx(0.462, abs=1e-3)  # el libro lee 0,47 del gráfico
    assert r.Q_W == pytest.approx(39.1e3, rel=0.025)
    assert r.cold.T_out_K - C == pytest.approx(65.9, abs=0.05)
    assert r.hot.T_out_K - C == pytest.approx(89.9, abs=0.05)


def test_cengel_11_5_glycerin_two_shell_passes() -> None:
    ex = _example(hx.TEST_EXAMPLES, "Glicerina").inputs
    r = solve_four_temperatures(ex)  # type: ignore[arg-type]
    P, R = r.P_R  # type: ignore[misc]
    assert P == pytest.approx(0.67, abs=0.005) and R == pytest.approx(0.75)
    assert r.F == pytest.approx(0.91, abs=0.002)
    assert r.Q_W == pytest.approx(1830.0, rel=2e-3)
    # con ensuciamiento de 0,0006 m²·K/W afuera de los tubos
    fouled = solve_four_temperatures(replace(ex, U_W_per_m2K=1.0 / (1.0 / 160 + 1.0 / 25 + 0.0006)))  # type: ignore[type-var]
    assert fouled.Q_W == pytest.approx(1.81e3, rel=3e-3)
    # con un solo paso de casco no se llega (ε = ε_máx)
    with pytest.raises(ValueError, match="no puede pasar"):
        solve_four_temperatures(replace(ex, arrangement=Arrangement("shell", 1, tubes="hot")))  # type: ignore[type-var]
    sizing = _example(hx.SIZING_EXAMPLES, "Glicerina").inputs
    s = solve_sizing(sizing)  # type: ignore[arg-type]
    assert s.A_m2 == pytest.approx(math.pi * 0.02 * 60.0, rel=2e-3)
    with pytest.raises(ValueError, match="no puede pasar"):
        solve_sizing(replace(sizing, arrangement=Arrangement("shell", 1, tubes="hot")))  # type: ignore[type-var]


def test_cengel_11_6_radiator() -> None:
    r = solve_four_temperatures(_example(hx.TEST_EXAMPLES, "Radiador").inputs)  # type: ignore[arg-type]
    assert r.F == pytest.approx(0.97, abs=0.002)
    assert r.dT_lm_K == pytest.approx(47.46, abs=0.01)
    assert r.Q_W == pytest.approx(62_925.0, rel=1e-9)
    assert r.U_W_per_m2K == pytest.approx(3346.0, rel=2e-3)
    # la fórmula aproximada daría otro F
    assert r.eps_approx is not None and r.eps_approx < r.effectiveness
    # verificación con ese U: vuelven las temperaturas
    v = solve_rating(_example(hx.RATING_EXAMPLES, "Radiador").inputs)  # type: ignore[arg-type]
    assert v.hot.T_out_K - C == pytest.approx(65.0, abs=0.05)
    assert v.cold.T_out_K - C == pytest.approx(40.0, abs=0.05)


def test_cengel_11_3_condenser() -> None:
    r = solve_four_temperatures(_example(hx.TEST_EXAMPLES, "Condensador").inputs)  # type: ignore[arg-type]
    assert r.dT_lm_K == pytest.approx(11.5, abs=0.05)
    assert r.Q_W == pytest.approx(1.09e6, rel=2e-3)
    assert r.cold.m_dot_kg_s == pytest.approx(32.5, rel=3e-3)
    assert r.hot.m_dot_kg_s == pytest.approx(0.45, rel=5e-3)
    assert r.C_r == 0.0 and r.F == 1.0 and r.P_R is None
    # con CoolProp (h_fg y c_p de tablas IAPWS) da casi lo mismo
    v = solve_rating(_example(hx.RATING_EXAMPLES, "Condensador").inputs)  # type: ignore[arg-type]
    assert v.cold.T_out_K - C == pytest.approx(22.0, abs=0.05)
    assert v.hot.m_dot_kg_s == pytest.approx(0.45, rel=0.01)


def test_incropera_oil_cooler_length() -> None:
    r = solve_sizing(_example(hx.SIZING_EXAMPLES, "Enfriador de aceite").inputs)  # type: ignore[arg-type]
    assert r.Q_W == pytest.approx(8524.0, rel=1e-3)
    assert r.cold.T_out_K - C == pytest.approx(40.2, abs=0.05)
    assert r.dT_lm_K == pytest.approx(43.2, abs=0.05)
    assert r.tube_length_m == pytest.approx(65.9, abs=0.1)


# ---------------------------------------------------------------------
# Coherencia
# ---------------------------------------------------------------------


def _all_results() -> list[hx.ExchangerResult]:
    out = [solve_rating(e.inputs) for e in hx.RATING_EXAMPLES.values()]  # type: ignore[arg-type]
    out += [solve_sizing(e.inputs) for e in hx.SIZING_EXAMPLES.values()]  # type: ignore[arg-type]
    out += [solve_four_temperatures(e.inputs) for e in hx.TEST_EXAMPLES.values()]  # type: ignore[arg-type]
    return out


def test_every_result_closes_its_balances() -> None:
    for r in _all_results():
        # Q̇ = C_h·ΔT_h = C_c·ΔT_c (o ṁ·h_fg)
        for s in (r.hot, r.cold):
            if s.phase_change:
                assert s.m_dot_kg_s * s.h_fg_J_per_kg == pytest.approx(r.Q_W, rel=1e-9)  # type: ignore[operator]
            else:
                assert s.C_W_per_K * s.dT_K == pytest.approx(r.Q_W, rel=1e-9)
        # LMTD: Q̇ = U·A·F·ΔT_ml
        assert r.U_W_per_m2K * r.A_m2 * r.F * r.dT_lm_K == pytest.approx(r.Q_W, rel=1e-9)
        # ε-NTU
        assert r.effectiveness == pytest.approx(r.Q_W / r.Q_max_W, rel=1e-12)
        mixed = r.mixed_is_cmin
        assert effectiveness(r.arrangement, r.NTU, r.C_r, mixed) == pytest.approx(
            r.effectiveness, rel=1e-9
        )
        # exergía: ΔΨ̇_h + ΔΨ̇_c + Ẋ_dest = 0 y Ṡ_gen > 0
        ex = r.exergy
        assert ex.S_gen_W_per_K > 0.0
        assert ex.dPsi_hot_W + ex.dPsi_cold_W + ex.X_dest_W == pytest.approx(0.0, abs=1e-6)
        assert ex.X_dest_W == pytest.approx(ex.T0_K * ex.S_gen_W_per_K)
        assert 0.0 < r.F <= 1.0 + 1e-12


def test_rating_sizing_and_test_agree() -> None:
    for e in hx.RATING_EXAMPLES.values():
        rating = solve_rating(e.inputs)  # type: ignore[arg-type]
        inp: RatingInputs = e.inputs  # type: ignore[assignment]
        if rating.hot.phase_change or rating.cold.phase_change:
            continue
        sizing = solve_sizing(
            SizingInputs(
                inp.arrangement,
                inp.hot,
                inp.cold,
                inp.U_W_per_m2K,
                "T_cold_out",
                rating.cold.T_out_K,
            )
        )
        assert sizing.A_m2 == pytest.approx(inp.A_m2, rel=1e-7)
        test = solve_four_temperatures(
            FourTemperatureInputs(
                inp.arrangement,
                inp.hot,
                replace(inp.cold, m_dot_kg_s=0.0),
                rating.hot.T_out_K,
                rating.cold.T_out_K,
                inp.A_m2,
                inp.U_W_per_m2K,
            )
        )
        assert test.Q_W == pytest.approx(rating.Q_W, rel=1e-7)
        assert test.cold.m_dot_kg_s == pytest.approx(inp.cold.m_dot_kg_s, rel=1e-6)


def test_coolprop_cp_is_taken_at_the_mean_temperature() -> None:
    from core.fluids import fluid_state_from_pair

    r = solve_rating(_example(hx.RATING_EXAMPLES, "Recuperador").inputs)  # type: ignore[arg-type]
    for s in (r.hot, r.cold):
        assert s.cp_from_coolprop
        cp = fluid_state_from_pair("Air", "TP", t=s.T_mean_K, p=s.stream.p_Pa).cp_J_per_kg_K
        assert s.cp_J_per_kgK == pytest.approx(cp, rel=1e-8)
    # el F bajo da un aviso
    assert any("F = " in w for w in r.warnings)


def test_phase_change_is_the_same_in_every_type() -> None:
    base: RatingInputs = _example(hx.RATING_EXAMPLES, "Condensador").inputs  # type: ignore[assignment]
    Qs = [solve_rating(replace(base, arrangement=a)).Q_W for a in ARRANGEMENTS]
    assert max(Qs) == pytest.approx(min(Qs), rel=1e-12)


def test_temperature_profile() -> None:
    for prefix in ("Agua calentada", "El mismo en flujo paralelo", "Condensador"):
        r = solve_rating(_example(hx.RATING_EXAMPLES, prefix).inputs)  # type: ignore[arg-type]
        profile = hx.temperature_profile(r)
        if r.arrangement.kind not in ("parallel", "counter"):
            assert profile is None
            continue
        assert profile is not None
        first, last = profile[0], profile[-1]
        assert first.T_hot_K == pytest.approx(r.hot.T_in_K)
        assert last.T_hot_K == pytest.approx(r.hot.T_out_K, rel=1e-9)
        if r.arrangement.kind == "parallel":
            assert first.T_cold_K == pytest.approx(r.cold.T_in_K)
            assert last.T_cold_K == pytest.approx(r.cold.T_out_K, rel=1e-9)
        else:
            assert first.T_cold_K == pytest.approx(r.cold.T_out_K)
            assert last.T_cold_K == pytest.approx(r.cold.T_in_K, rel=1e-9)
        assert all(p.T_hot_K > p.T_cold_K for p in profile)


def test_type_comparison() -> None:
    r = solve_rating(_example(hx.RATING_EXAMPLES, "Agua calentada").inputs)  # type: ignore[arg-type]
    rows = hx.type_comparison(r)
    Q = {row.arrangement.name: row.Q_W for row in rows}
    assert max(Q.values()) == Q["Doble tubo, contracorriente"]  # type: ignore[type-var]
    assert min(Q.values()) == Q["Doble tubo, flujo paralelo"]  # type: ignore[type-var]
    sizing = solve_sizing(_example(hx.SIZING_EXAMPLES, "Glicerina").inputs)  # type: ignore[arg-type]
    rows = hx.type_comparison(sizing)
    areas = {row.arrangement.name: row.A_m2 for row in rows}
    feasible = [a for a in areas.values() if a is not None]
    assert min(feasible) == areas["Doble tubo, contracorriente"]
    failed = [row for row in rows if row.error]
    assert failed and all(row.A_m2 is None for row in failed)  # 1 casco y paralelo no llegan


# ---------------------------------------------------------------------
# Validaciones y mensajes
# ---------------------------------------------------------------------


def test_validation_messages() -> None:
    hot = Stream("agua", 90.0 + C, 1.0, cp_J_per_kgK=4180.0)
    cold = Stream("aire", 20.0 + C, 1.0, cp_J_per_kgK=1005.0)
    arr = Arrangement("counter")
    with pytest.raises(ValueError, match="tiene que entrar más caliente"):
        solve_rating(RatingInputs(arr, replace(hot, T_in_K=10.0 + C), cold, 100.0, 1.0))
    with pytest.raises(ValueError, match="Los dos fluidos cambian de fase"):
        solve_rating(
            RatingInputs(
                arr,
                replace(hot, phase_change=True, h_fg_J_per_kg=2.2e6),
                replace(cold, phase_change=True, h_fg_J_per_kg=2.4e6),
                100.0,
                1.0,
            )
        )
    with pytest.raises(ValueError, match="U tiene que ser positivo"):
        solve_rating(RatingInputs(arr, hot, cold, 0.0, 1.0))
    with pytest.raises(ValueError, match="área"):
        solve_rating(RatingInputs(arr, hot, cold, 100.0, -1.0))
    with pytest.raises(ValueError, match="caudal del fluido frío"):
        solve_rating(RatingInputs(arr, hot, replace(cold, m_dot_kg_s=0.0), 100.0, 1.0))
    with pytest.raises(ValueError, match="falta el c_p"):
        solve_rating(RatingInputs(arr, hot, replace(cold, cp_J_per_kgK=None), 100.0, 1.0))
    with pytest.raises(ValueError, match="calor latente"):
        solve_rating(RatingInputs(arr, replace(hot, phase_change=True), cold, 100.0, 1.0))
    with pytest.raises(ValueError, match="deslizamiento"):
        solve_rating(
            RatingInputs(
                arr, Stream("R-410A", 40.0 + C, fluid="R410A", phase_change=True), cold, 100.0, 1.0
            )
        )
    # agua a 1 atm calentada a más de 100 °C
    water = Stream("agua", 20.0 + C, 0.2, fluid="Water")
    steam = Stream("gases", 400.0 + C, 2.0, cp_J_per_kgK=1100.0)
    with pytest.raises(ValueError, match="Subí la presión"):
        solve_rating(RatingInputs(arr, steam, water, 500.0, 1.0))
    assert solve_rating(RatingInputs(arr, steam, replace(water, p_Pa=30e5), 500.0, 1.0)).Q_W > 0


def test_sizing_messages() -> None:
    hot = Stream("agua", 90.0 + C, 1.0, cp_J_per_kgK=4180.0)
    cold = Stream("agua", 20.0 + C, 1.0, cp_J_per_kgK=4180.0)
    with pytest.raises(ValueError, match="tiene que quedar entre"):
        solve_sizing(SizingInputs(Arrangement("counter"), hot, cold, 500.0, "T_cold_out", 95.0 + C))
    with pytest.raises(ValueError, match="Q̇_máx"):
        solve_sizing(SizingInputs(Arrangement("counter"), hot, cold, 500.0, "Q", 400e3))
    with pytest.raises(ValueError, match="flujo paralelo"):
        solve_sizing(
            SizingInputs(Arrangement("parallel"), hot, cold, 500.0, "T_cold_out", 60.0 + C)
        )
    with pytest.raises(ValueError, match="no es un objetivo"):
        solve_sizing(
            SizingInputs(
                Arrangement("counter"),
                replace(hot, phase_change=True, h_fg_J_per_kg=2.2e6),
                cold,
                500.0,
                "T_hot_out",
                50.0 + C,
            )
        )


def test_test_mode_messages() -> None:
    hot = Stream("agua", 90.0 + C, cp_J_per_kgK=4180.0)
    cold = Stream("aire", 20.0 + C, cp_J_per_kgK=1005.0)
    cc, pf = Arrangement("counter"), Arrangement("parallel")
    with pytest.raises(ValueError, match="tiene que salir más frío"):
        solve_four_temperatures(FourTemperatureInputs(cc, hot, cold, 95 + C, 40 + C, 1.0, 100.0))
    with pytest.raises(ValueError, match="Falta el caudal"):
        solve_four_temperatures(FourTemperatureInputs(cc, hot, cold, 60 + C, 40 + C, 1.0))
    with pytest.raises(ValueError, match="flujo paralelo"):
        solve_four_temperatures(FourTemperatureInputs(pf, hot, cold, 50 + C, 70 + C, 1.0, 100.0))


def test_overall_u_messages_and_notes() -> None:
    with pytest.raises(ValueError, match="diámetro exterior"):
        overall_u(OverallUInputs("tube", 100.0, 100.0, 15.0, D_i_m=0.02, D_o_m=0.02))
    with pytest.raises(ValueError, match="ensuciamiento"):
        overall_u(
            OverallUInputs(
                "tube", 100.0, 100.0, 15.0, D_i_m=0.02, D_o_m=0.025, R_f_i_m2K_per_W=-1.0
            )
        )
    with pytest.raises(ValueError, match="espesor"):
        overall_u(OverallUInputs("plane", 100.0, 100.0, 15.0))
    gas = overall_u(_example(hx.U_EXAMPLES, "Agua y aire").inputs)  # type: ignore[arg-type]
    notes = " ".join(hx.overall_u_notes(gas))
    assert "aletas" in notes and "convección afuera del tubo" in notes
    fouled = overall_u(_example(hx.U_EXAMPLES, "Doble tubo").inputs)  # type: ignore[arg-type]
    assert "ensuciamiento baja U" in " ".join(hx.overall_u_notes(fouled))


def test_notes() -> None:
    for r in _all_results():
        notes = hx.exchanger_notes(r)
        assert notes and all(isinstance(n, str) and n for n in notes)
        text = " ".join(notes)
        assert "ε = " in text and "Ẋ_dest" in text
        if r.C_r == 0.0:
            assert "C_r = 0" in text
        if r.arrangement.kind == "cross_unmixed":
            assert "serie exacta" in text


def test_export_and_pickle() -> None:
    for r in _all_results():
        for system in SYSTEMS:
            data = hx.exchanger_to_dict(r, system)
            json.dumps(data, ensure_ascii=False)
            assert dict_to_csv(data)
        assert pickle.loads(pickle.dumps(r)) == r
    for e in hx.U_EXAMPLES.values():
        u = overall_u(e.inputs)  # type: ignore[arg-type]
        for system in SYSTEMS:
            assert dict_to_csv(hx.overall_u_to_dict(u, system))
