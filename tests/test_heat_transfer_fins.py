"""Tests de core.heat_transfer.fins (aletas, Fase 8.1).

Las fórmulas cerradas (Incropera, tabla 3.4 y §3.6.4) se contrastan con la
solución numérica de la ecuación de la aleta (``scipy.integrate.solve_bvp``).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_bvp

from core.heat_transfer.fins import (
    FIN_EXAMPLES,
    FinArrayInputs,
    FinInputs,
    efficiency_curve,
    fin_array_notes,
    fin_notes,
    fin_profile,
    fin_to_dict,
    solve_fin,
    solve_fin_array,
    tip_comparison,
)

C = 273.15
NAMES = list(FIN_EXAMPLES)

PIN = FinInputs("pin", 180.0, 100.0, 100.0 + C, 25.0 + C, length_m=0.03, diameter_m=0.005)
STRAIGHT = FinInputs(
    "straight", 200.0, 25.0, 80.0 + C, 25.0 + C, length_m=0.03, thickness_m=0.0015, width_m=0.05
)
ANNULAR = FinInputs(
    "annular", 186.0, 40.0, 180.0 + C, 25.0 + C, length_m=0.005, thickness_m=0.001, r_base_m=0.025
)


def _bvp_uniform(i: FinInputs, tip: str) -> float:
    """q de la aleta de sección constante resolviendo ψ'' = (mL)²·ψ numéricamente.

    Adimensional: ψ = θ/θ_b y ξ = x/L (la forma bien condicionada).
    """
    if i.shape == "pin":
        A_c, P = math.pi * i.diameter_m**2 / 4, math.pi * i.diameter_m
    else:
        A_c, P = i.width_m * i.thickness_m, 2 * (i.width_m + i.thickness_m)
    k, h, tb, L = i.k_W_per_mK, i.h_W_per_m2K, i.theta_b, i.length_m
    mL2 = h * P / (k * A_c) * L * L

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        if tip == "convective":
            other = yb[1] + h * L / k * yb[0]
        elif tip == "adiabatic":
            other = yb[1]
        else:
            other = yb[0] - (i.T_tip_K - i.T_inf_K) / tb  # type: ignore[operator]
        return np.array([ya[0] - 1.0, other])

    xi = np.linspace(0.0, 1.0, 200)
    y0 = np.vstack([np.ones_like(xi), np.zeros_like(xi)])
    sol = solve_bvp(
        lambda _x, y: np.vstack([y[1], mL2 * y[0]]), bc, xi, y0, tol=1e-10, max_nodes=200_000
    )
    assert sol.success, sol.message
    return float(-k * A_c * tb * sol.sol(0.0)[1] / L)


def _bvp_annular(i: FinInputs, tip: str) -> float:
    """q de la aleta anular resolviendo d/dρ(ρ·dψ/dρ) = (m·r₁)²·ρ·ψ, con ρ = r/r₁."""
    k, h, t, tb = i.k_W_per_mK, i.h_W_per_m2K, i.thickness_m, i.theta_b
    r1 = i.r_base_m
    mr1_2 = 2 * h / (k * t) * r1 * r1
    rho2 = (r1 + i.length_m) / r1

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        if tip == "convective":
            other = yb[1] + h * r1 / k * yb[0]
        elif tip == "adiabatic":
            other = yb[1]
        else:
            other = yb[0] - (i.T_tip_K - i.T_inf_K) / tb  # type: ignore[operator]
        return np.array([ya[0] - 1.0, other])

    rho = np.linspace(1.0, rho2, 200)
    y0 = np.vstack([np.ones_like(rho), np.zeros_like(rho)])
    sol = solve_bvp(
        lambda rr, y: np.vstack([y[1], mr1_2 * y[0] - y[1] / rr]),
        bc,
        rho,
        y0,
        tol=1e-10,
        max_nodes=200_000,
    )
    assert sol.success, sol.message
    return float(-k * 2 * math.pi * t * tb * sol.sol(1.0)[1])


# ---------------------------------------------------------------------
# Contra el libro y la solución numérica
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("material", "q", "L_inf"),
    [("cobre", 8.3, 0.19), ("aluminio", 5.6, 0.13), ("inoxidable", 1.6, 0.04)],
)
def test_incropera_long_rods(material: str, q: float, L_inf: float) -> None:
    """Incropera, ej. 3.9: varillas de 5 mm muy largas, h = 100, base a 100 °C y aire a 25 °C."""
    name = next(n for n in NAMES if "Varilla" in n and material in n)
    r = solve_fin(FIN_EXAMPLES[name].fin)
    assert r.Q_W == pytest.approx(q, abs=0.05)
    assert r.L_infinite_m == pytest.approx(L_inf, abs=0.005)
    assert r.efficiency is None
    assert r.effectiveness == pytest.approx(
        math.sqrt(r.inputs.k_W_per_mK * math.pi * 0.005 / (100.0 * math.pi * 0.005**2 / 4))
    )


@pytest.mark.parametrize("base", [PIN, STRAIGHT])
@pytest.mark.parametrize("tip", ["convective", "adiabatic", "temperature"])
def test_uniform_fins_match_the_numerical_solution(base: FinInputs, tip: str) -> None:
    fin = replace(base, tip=tip, T_tip_K=base.T_inf_K + 0.4 * base.theta_b)  # type: ignore[arg-type]
    assert solve_fin(fin).Q_W == pytest.approx(_bvp_uniform(fin, tip), rel=1e-6)


@pytest.mark.parametrize("tip", ["convective", "adiabatic", "temperature"])
def test_annular_fin_matches_the_numerical_solution(tip: str) -> None:
    fin = replace(ANNULAR, tip=tip, T_tip_K=ANNULAR.T_inf_K + 0.9 * ANNULAR.theta_b)  # type: ignore[arg-type]
    assert solve_fin(fin).Q_W == pytest.approx(_bvp_annular(fin, tip), rel=1e-5)
    # una aleta más larga (m·r grande) tampoco desborda
    long = replace(fin, length_m=0.2, k_W_per_mK=20.0)
    assert solve_fin(long).Q_W == pytest.approx(_bvp_annular(long, tip), rel=1e-4)


def test_annular_infinite_is_the_limit_of_a_long_fin() -> None:
    inf = solve_fin(replace(ANNULAR, tip="infinite", k_W_per_mK=10.0))
    long = solve_fin(replace(ANNULAR, tip="adiabatic", k_W_per_mK=10.0, length_m=1.0))
    assert inf.Q_W == pytest.approx(long.Q_W, rel=1e-9)


def test_limits_and_corrected_length() -> None:
    # una aleta convectiva muy larga es una infinita
    long = solve_fin(replace(PIN, length_m=2.0, tip="convective"))
    assert long.Q_W == pytest.approx(solve_fin(replace(PIN, tip="infinite")).Q_W, rel=1e-9)
    # la longitud corregida aproxima la punta convectiva (aleta delgada: < 0,1 %)
    for base in (PIN, STRAIGHT, ANNULAR):
        corr = solve_fin(replace(base, tip="corrected"))
        exact = solve_fin(replace(base, tip="convective"))
        assert corr.Q_W == pytest.approx(exact.Q_W, rel=1e-3)
    # recta con L_c: η = tanh(mL_c)/(mL_c)
    r = solve_fin(replace(STRAIGHT, tip="corrected"))
    assert r.L_c_m == pytest.approx(0.03 + 0.0015 / 2)
    assert r.efficiency == pytest.approx(math.tanh(r.mL) / r.mL, rel=1e-12)


def test_steam_pipe_with_annular_fins() -> None:
    """El caño de Cengel y Ghajar: η = 0,995 exacta y 3689 W por metro con 250 aletas."""
    ex = next(FIN_EXAMPLES[n] for n in NAMES if "aletas anulares" in n)
    fin = solve_fin(ex.fin)
    assert fin.efficiency == pytest.approx(0.9952, abs=1e-4)
    assert fin.Q_W == pytest.approx(11.835, abs=0.005)
    arr = solve_fin_array(FinArrayInputs(ex.fin, ex.n_fins, ex.base_area_m2))  # type: ignore[arg-type]
    assert arr.Q_no_fins_W == pytest.approx(974.0, abs=0.5)
    assert arr.Q_total_W == pytest.approx(3689.0, abs=1.0)
    # η_o = 1 − (N·A_f/A_t)·(1 − η) (Incropera, ec. 3.102)
    frac = ex.n_fins * fin.A_fin_m2 / arr.A_total_m2  # type: ignore[operator]
    assert arr.overall_efficiency == pytest.approx(1 - frac * (1 - fin.efficiency), rel=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_every_example_profile_and_export(name: str) -> None:
    ex = FIN_EXAMPLES[name]
    r = solve_fin(ex.fin)
    xs, Ts = fin_profile(r)
    assert Ts[0] == pytest.approx(ex.fin.T_base_K, rel=1e-12)
    assert Ts[-1] == pytest.approx(r.T_tip_K, rel=1e-9)
    # monótono hacia el fluido
    assert all(a >= b for a, b in zip(Ts, Ts[1:], strict=False))
    assert xs[-1] == pytest.approx(r.L_c_m)
    data = fin_to_dict(r, "Inglés")
    assert data["Q"].endswith("Btu/h")
    assert ex.note
    comparison = tip_comparison(ex.fin)
    assert comparison["infinite"].Q_W >= comparison["convective"].Q_W


def test_temperature_tip_profile_ends_at_the_given_temperature() -> None:
    fin = replace(PIN, tip="temperature", T_tip_K=50.0 + C)
    xs, Ts = fin_profile(solve_fin(fin))
    assert Ts[-1] == pytest.approx(50.0 + C, rel=1e-12)
    ann = replace(ANNULAR, tip="temperature", T_tip_K=150.0 + C)
    xs, Ts = fin_profile(solve_fin(ann))
    assert Ts[-1] == pytest.approx(150.0 + C, rel=1e-9)


def test_efficiency_curve() -> None:
    r = solve_fin(replace(STRAIGHT, tip="corrected"))
    xs, etas = efficiency_curve(r)
    assert etas[0] > etas[-1]
    assert np.interp(r.mL, xs, etas) == pytest.approx(r.efficiency, abs=2e-4)
    ann = solve_fin(replace(ANNULAR, tip="corrected"))
    xs, etas = efficiency_curve(ann)
    assert np.interp(ann.mL, xs, etas) == pytest.approx(ann.efficiency, abs=2e-3)
    # la anular está por debajo de la recta con el mismo mL_c
    straight_eta = math.tanh(xs[40]) / xs[40]
    assert etas[40] <= straight_eta + 1e-9


# ---------------------------------------------------------------------
# Notas y mensajes
# ---------------------------------------------------------------------


def test_notes() -> None:
    thick = solve_fin(replace(STRAIGHT, k_W_per_mK=15.0, h_W_per_m2K=5000.0, thickness_m=0.01))
    notes = " ".join(fin_notes(thick))
    assert "Bi =" in notes and "ε =" in notes
    rod = solve_fin(replace(PIN, length_m=0.5, tip="convective"))
    assert any("se comporta como infinita" in n for n in fin_notes(rod))
    corr = solve_fin(replace(STRAIGHT, tip="corrected"))
    assert any("longitud corregida" in n for n in fin_notes(corr))
    ann = solve_fin(ANNULAR)
    assert any("Bessel" in n for n in fin_notes(ann))
    arr = solve_fin_array(FinArrayInputs(STRAIGHT, 12, 0.003))
    assert "veces" in fin_array_notes(arr)[0]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"k_W_per_mK": 0.0}, "conductividad"),
        ({"h_W_per_m2K": -1.0}, "convección h"),
        ({"T_inf_K": 100.0 + C}, "misma temperatura"),
        ({"length_m": 0.0}, "largo L"),
        ({"diameter_m": 0.0}, "diámetro"),
        ({"tip": "temperature", "T_tip_K": None}, "falta esa temperatura"),
    ],
)
def test_messages(change: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_fin(replace(PIN, **change))


def test_array_messages() -> None:
    with pytest.raises(ValueError, match="no entran"):
        solve_fin_array(FinArrayInputs(STRAIGHT, 1000, 0.003))
    with pytest.raises(ValueError, match="al menos una aleta"):
        solve_fin_array(FinArrayInputs(STRAIGHT, 0, 0.003))
    with pytest.raises(ValueError, match="radio de la base"):
        solve_fin(replace(ANNULAR, r_base_m=0.0))
