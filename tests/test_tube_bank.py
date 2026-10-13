"""Tests del banco de tubos (Fase 12.1): Zukauskas contra Incropera y Çengel y Ghajar."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from core.heat_transfer import tube_bank as tb
from core.heat_transfer.tube_bank import (
    TUBE_BANK_EXAMPLES,
    BankProperties,
    compare_arrangements,
    friction_chart,
    max_velocity,
    nu_zukauskas,
    row_correction,
    solve_tube_bank,
    tube_bank_notes,
    tube_bank_to_dict,
    velocity_sweep,
    zukauskas_fit,
)

C = 273.15
INCROPERA = TUBE_BANK_EXAMPLES["Calentador de aire en tresbolillo (Incropera, ejemplo 7.7)"].inputs
CENGEL = TUBE_BANK_EXAMPLES[
    "Aire precalentado con agua geotérmica (Çengel y Ghajar, ejemplo 7-7)"
].inputs
#: Incropera, ejemplo 7.7: aire a 15 °C (todas las propiedades a la entrada), Pr_s a 70 °C.
INCROPERA_PROPS = BankProperties(
    rho_kg_per_m3=1.217,
    mu_Pa_s=14.82e-6 * 1.217,
    k_W_per_mK=0.0253,
    cp_J_per_kgK=1007.0,
    Pr_s=0.701,
    rho_in_kg_per_m3=1.217,
    Pr=0.710,
)
#: Çengel y Ghajar, ejemplo 7-7: aire a 60 °C (supuesta), Pr_s a 120 °C y ρ a 20 °C.
CENGEL_PROPS = BankProperties(
    rho_kg_per_m3=1.059,
    mu_Pa_s=2.008e-5,
    k_W_per_mK=0.02808,
    cp_J_per_kgK=1007.0,
    Pr_s=0.7073,
    rho_in_kg_per_m3=1.204,
    Pr=0.7202,
)


# ---------------------------------------------------------------------
# Las constantes
# ---------------------------------------------------------------------


def test_row_correction_tables() -> None:
    # Çengel y Ghajar, tabla 7-3 (6 filas: entre 5 y 7, como el ejemplo 7-7)
    assert row_correction(6, "aligned", "cengel") == pytest.approx(0.945)
    assert row_correction(7, "staggered", "cengel") == pytest.approx(0.96)
    assert row_correction(1, "staggered", "cengel") == pytest.approx(0.64)
    assert row_correction(16, "aligned", "cengel") == 1.0
    assert row_correction(40, "staggered", "cengel") == 1.0
    # Incropera, tabla 7.6: C₂ hasta 20 filas
    assert row_correction(7, "staggered", "incropera") == pytest.approx(0.95)
    assert row_correction(16, "aligned", "incropera") == pytest.approx(0.99)
    assert row_correction(18, "aligned", "incropera") == pytest.approx(0.995)
    assert row_correction(20, "aligned", "incropera") == 1.0
    # crece con las filas
    values = [row_correction(n, "aligned", "cengel") for n in range(1, 20)]
    assert values == sorted(values)


def test_zukauskas_ranges() -> None:
    f = zukauskas_fit(600.0, "staggered", "cengel", 1.0, 0.7)
    assert (f.C, f.m) == (0.71, 0.5)
    f = zukauskas_fit(5.0e5, "aligned", "cengel", 1.0, 0.7)
    assert (f.C, f.m, f.n) == (0.033, 0.8, 0.4)
    # Incropera, 10² a 10³: cilindro aislado, n según Pr
    f = zukauskas_fit(500.0, "aligned", "incropera", 1.0, 0.7)
    assert f.single_cylinder and (f.C, f.m, f.n) == (0.51, 0.5, 0.37)
    assert zukauskas_fit(500.0, "aligned", "incropera", 1.0, 20.0).n == 0.36
    # tresbolillo con S_T/S_L > 2: 0,40 sin el factor del paso (Incropera)
    f = zukauskas_fit(1.0e4, "staggered", "incropera", 2.5, 0.7)
    assert (f.C, f.p) == (0.40, 0.0)
    f = zukauskas_fit(1.0e4, "staggered", "cengel", 2.5, 0.7)
    assert f.coefficient(2.5) == pytest.approx(0.35 * 2.5**0.2)
    # 0,35·2^0,2 ≈ 0,40: las dos versiones empalman
    assert 0.35 * 2.0**0.2 == pytest.approx(0.40, abs=0.003)


def test_nu_zukauskas_formula() -> None:
    Re, Pr, Prs = 1.0e4, 0.71, 0.70
    expected = 0.27 * Re**0.63 * Pr**0.36 * (Pr / Prs) ** 0.25
    assert nu_zukauskas(Re, Pr, Prs, "aligned", 1.0) == pytest.approx(expected, rel=1e-12)


def test_max_velocity() -> None:
    # Incropera 7.7: por el plano transversal (S_D = 37,7 mm > (S_T + D)/2 = 23,9 mm)
    V, S_D, diagonal = max_velocity(6.0, 0.0164, 0.0313, 0.0343, "staggered")
    assert V == pytest.approx(12.6, abs=0.01)
    assert S_D == pytest.approx(0.0377, abs=1e-4) and not diagonal
    # filas muy juntas: pasa por la diagonal
    V, S_D, diagonal = max_velocity(1.0, 0.020, 0.050, 0.015, "staggered")
    assert diagonal and S_D == pytest.approx(math.hypot(0.015, 0.025))
    assert V == pytest.approx(0.050 / (2.0 * (S_D - 0.020)))
    V, S_D, diagonal = max_velocity(4.5, 0.015, 0.05, 0.05, "aligned")
    assert V == pytest.approx(6.43, abs=0.005) and S_D is None and not diagonal


# ---------------------------------------------------------------------
# Contra los libros
# ---------------------------------------------------------------------


def test_incropera_7_7_with_the_book_properties() -> None:
    r = solve_tube_bank(replace(INCROPERA, properties=INCROPERA_PROPS))
    assert r.V_max_m_s == pytest.approx(12.6, abs=0.01)
    assert r.Re == pytest.approx(13_943, rel=1e-3)
    assert r.F == pytest.approx(0.95)
    # el libro redondea C = 0,35·(S_T/S_L)^⅕ a 0,34: con ese redondeo da 135,6 exacto
    assert r.C == pytest.approx(0.3437, abs=1e-4)
    assert r.h_W_per_m2K * 0.34 / r.C == pytest.approx(135.6, rel=2e-3)
    assert r.h_W_per_m2K == pytest.approx(135.6, rel=0.012)
    assert r.Q_per_length_W_m == pytest.approx(19.4e3, rel=0.015)
    assert r.T_out_K - C == pytest.approx(25.5, abs=0.2)
    # f y χ: el libro los lee de los gráficos (0,35 y 1,04): ±5 %
    assert r.f == pytest.approx(0.35, rel=0.05)
    assert r.chi == pytest.approx(1.04, rel=0.01)
    assert r.dp_Pa == pytest.approx(246.0, rel=0.06)
    assert r.warnings == ()


def test_cengel_7_7_with_the_book_properties() -> None:
    r = solve_tube_bank(replace(CENGEL, properties=CENGEL_PROPS))
    assert r.V_max_m_s == pytest.approx(6.43, abs=0.005)
    assert r.Re == pytest.approx(5086, abs=2)
    assert r.F == pytest.approx(0.945)
    assert r.Nu == pytest.approx(49.3, abs=0.05)
    assert r.h_W_per_m2K == pytest.approx(92.2, abs=0.05)
    assert r.m_dot_kg_s == pytest.approx(2.709, abs=1e-3)
    assert r.T_out_K - C == pytest.approx(29.11, abs=0.01)
    assert r.dT_lm_K == pytest.approx(95.4, abs=0.05)
    assert r.Q_W == pytest.approx(24.9e3, rel=2e-3)
    # S_L/D = 3,33 está fuera del gráfico: se lee la curva de 2,5 (el libro lee 0,16)
    assert any("fuera del gráfico de f" in w for w in r.warnings)
    assert r.chi == 1.0  # alineado cuadrado: el arreglo de referencia
    assert r.f == pytest.approx(0.16, rel=0.12)
    assert r.dp_Pa == pytest.approx(21.0, rel=0.15)


def test_coolprop_against_the_books() -> None:
    # Incropera evalúa a la entrada: con la media y CoolProp, h queda +2 %
    r = solve_tube_bank(INCROPERA)
    assert r.h_W_per_m2K == pytest.approx(135.6, rel=0.03)
    assert r.Q_per_length_W_m == pytest.approx(19.4e3, rel=0.03)
    assert r.dp_Pa == pytest.approx(246.0, rel=0.07)
    # Çengel supone 60 °C, pero la media real es ~25 °C: el aire más denso da h ~5 % mayor
    r = solve_tube_bank(CENGEL)
    assert r.T_mean_K - C == pytest.approx(24.8, abs=0.2)
    assert r.h_W_per_m2K / 92.2 == pytest.approx(1.05, abs=0.01)


# ---------------------------------------------------------------------
# Consistencia
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TUBE_BANK_EXAMPLES))
def test_energy_balance_closes(name: str) -> None:
    r = solve_tube_bank(TUBE_BANK_EXAMPLES[name].inputs)
    i = r.inputs
    assert r.Q_W == pytest.approx(r.m_dot_kg_s * r.cp_J_per_kgK * (r.T_out_K - i.T_in_K), rel=1e-8)
    assert r.T_mean_K == pytest.approx(0.5 * (i.T_in_K + r.T_out_K))
    assert i.T_in_K < r.T_out_K < i.T_s_K


def test_cooling_the_fluid() -> None:
    r = solve_tube_bank(replace(CENGEL, T_in_K=150.0 + C, T_s_K=30.0 + C))
    assert r.Q_W < 0.0
    assert 30.0 + C < r.T_out_K < 150.0 + C


def test_book_properties_are_not_iterated() -> None:
    r = solve_tube_bank(replace(CENGEL, properties=CENGEL_PROPS))
    assert r.iterations == 1
    assert r.Pr == 0.7202 and r.rho_in_kg_per_m3 == 1.204
    assert any("dadas" in n for n in tube_bank_notes(r))


def test_more_velocity_more_h_and_much_more_dp() -> None:
    points = velocity_sweep(INCROPERA)
    assert len(points) == 25
    hs = [p.h_W_per_m2K for p in points]
    dps = [p.dp_Pa for p in points]
    assert hs == sorted(hs) and dps == sorted(dps)
    # al duplicar V: h ~ V^0,6 y Δp ~ V^1,8
    a, b = points[0], points[-1]
    ratio = b.V_m_s / a.V_m_s
    assert 0.5 < math.log(b.h_W_per_m2K / a.h_W_per_m2K) / math.log(ratio) < 0.7
    assert 1.6 < math.log(b.dp_Pa / a.dp_Pa) / math.log(ratio) < 2.0


def test_compare_arrangements() -> None:
    both = compare_arrangements(INCROPERA)
    assert set(both) == {"aligned", "staggered"}
    # mismos pasos: el tresbolillo pierde más presión
    assert both["staggered"].dp_Pa > both["aligned"].dp_Pa


def test_the_other_constants_in_the_notes() -> None:
    r = solve_tube_bank(CENGEL)
    assert any("Incropera" in n for n in tube_bank_notes(r))
    r = solve_tube_bank(replace(CENGEL, V_m_s=0.25))
    if r.fit.single_cylinder:  # pragma: no cover — depende del Re
        assert any("cilindro aislado" in n for n in tube_bank_notes(r))
    r = solve_tube_bank(replace(INCROPERA, V_m_s=0.3))
    assert 100 < r.Re < 1000 and r.fit.single_cylinder
    assert any("cilindro aislado" in n for n in tube_bank_notes(r))
    assert any("Re ≳ 10³" in w for w in r.warnings)


# ---------------------------------------------------------------------
# El gráfico de f y χ
# ---------------------------------------------------------------------


def test_chi_is_one_at_the_reference_arrangements() -> None:
    for Re in (1.0e3, 3.0e3, 2.0e4, 5.0e5):
        assert friction_chart(Re, "aligned", 1.5, 1.5).chi == 1.0
    for Re in (2.0e2, 5.0e3, 5.0e4):
        r = friction_chart(Re, "staggered", 2.0, 2.0 * math.sqrt(3.0) / 2.0)
        assert r.chi == pytest.approx(1.0, abs=1e-9)


def test_f_reads_the_curves() -> None:
    curves = tb.friction_curves("staggered")
    assert sorted(curves) == [1.25, 1.5, 2.0, 2.5]
    # sobre una curva, el valor guardado
    Re, f = curves[1.5][0][10], curves[1.5][1][10]
    assert friction_chart(Re, "staggered", 1.5, 1.5 * math.sqrt(3) / 2).f == pytest.approx(f)
    # entre curvas, lineal en el paso
    mid = friction_chart(1.0e4, "staggered", 1.75, 1.6).f
    lo = friction_chart(1.0e4, "staggered", 1.5, 1.3).f
    hi = friction_chart(1.0e4, "staggered", 2.0, 1.7).f
    assert mid == pytest.approx(0.5 * (lo + hi))
    # f baja con Re y con el paso
    low_Re = friction_chart(1.0e3, "aligned", 1.5, 1.5).f
    assert low_Re > friction_chart(1.0e5, "aligned", 1.5, 1.5).f
    tight = friction_chart(1.0e4, "staggered", 1.25, 1.1).f
    assert tight > friction_chart(1.0e4, "staggered", 2.5, 2.2).f


def test_chart_warnings() -> None:
    r = friction_chart(1.0e4, "staggered", 3.0, 3.0)
    assert any("fuera del gráfico de f" in w for w in r.warnings)
    r = friction_chart(5.0, "staggered", 2.0, 2.0)
    assert any("se prolonga su tramo final" in w for w in r.warnings)
    r = friction_chart(50.0, "aligned", 1.5, 2.0)
    assert any("curvas de Re" in w for w in r.warnings)
    r = friction_chart(1.0e4, "aligned", 1.05, 3.0)
    assert any("χ:" in w for w in r.warnings)
    assert friction_chart(1.0e4, "staggered", 1.5, 1.3).warnings == ()


def test_chart_data_file() -> None:
    text = tb.FRICTION_DATA_PATH.read_text(encoding="utf-8")
    assert text.startswith("# Gráficos del banco de tubos de Zukauskas (1972)")
    data = tb._chart_data()
    assert len(data) == 16
    for (_, _), (xs, ys) in data.items():
        assert list(xs) == sorted(xs) and all(y > 0 for y in ys)


# ---------------------------------------------------------------------
# Errores y avisos
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"S_T_m": 0.015}, "los tubos de una fila se tocan"),
        ({"S_L_m": 0.010}, "dos filas seguidas se tocan"),
        ({"T_s_K": 20.0 + C}, "misma temperatura"),
        ({"V_m_s": 0.0}, "velocidad"),
        ({"N_L": 0}, "al menos una fila"),
        ({"D_m": -0.01}, "diámetro"),
    ],
)
def test_student_errors(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_tube_bank(replace(CENGEL, **change))


def test_staggered_overlap() -> None:
    with pytest.raises(ValueError, match="se superponen"):
        solve_tube_bank(replace(INCROPERA, S_L_m=0.005, S_T_m=0.020))


def test_phase_change_and_wall_boiling() -> None:
    water = TUBE_BANK_EXAMPLES["Agua sobre un banco de tubos calientes"].inputs
    # tubos a 120 °C en agua a 1 atm: hierve en la pared (aviso)
    r = solve_tube_bank(replace(water, p_Pa=101_325.0, T_s_K=120.0 + C))
    assert any("ebullición en la pared" in w for w in r.warnings)
    # agua casi a 100 °C y un banco enorme: saldría hirviendo (error)
    with pytest.raises(ValueError, match="cambia de fase"):
        solve_tube_bank(
            replace(water, p_Pa=101_325.0, T_in_K=99.0 + C, T_s_K=140.0 + C, V_m_s=0.05, N_L=40)
        )


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export_is_json(system: str) -> None:
    r = solve_tube_bank(CENGEL)
    data = tube_bank_to_dict(r, system)  # type: ignore[arg-type]
    json.dumps(data)
    assert data["arreglo"] == "Alineado"
    assert data["Delta_p"]["unidad"] == {"SI": "Pa", "Técnico": "Pa", "Inglés": "lbf/ft²"}[system]
    assert data["Q"]["valor"] == pytest.approx(
        {"SI": 1.0, "Técnico": 1.0, "Inglés": 3600.0 / 1055.05585262}[system] * r.Q_W
    )


@pytest.mark.parametrize("name", list(TUBE_BANK_EXAMPLES))
def test_examples_solve(name: str) -> None:
    ex = TUBE_BANK_EXAMPLES[name]
    r = solve_tube_bank(ex.inputs)
    assert r.Q_W > 0.0 and r.dp_Pa > 0.0
    assert ex.description
    if "Çengel" not in name:
        assert r.warnings == ()
