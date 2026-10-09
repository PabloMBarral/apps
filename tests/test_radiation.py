"""Tests de core.heat_transfer.radiation (Fase 8.2): cuerpo negro, factores de forma y
recintos."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from core.export import dict_to_csv
from core.heat_transfer import radiation as rd
from core.heat_transfer.radiation import (
    SIGMA,
    BlackbodyInputs,
    EnclosureInputs,
    EnclosureSurface,
    Shield,
    SpectralBand,
    SurfaceBalanceInputs,
    ThermocoupleInputs,
    TwoSurfaceInputs,
    ViewFactorInputs,
    blackbody_fraction,
    fraction_inverse,
    planck,
    solve_blackbody,
    solve_enclosure,
    solve_surface_balance,
    solve_thermocouple,
    solve_two_surface,
    solve_view_factor,
)

SYSTEMS = ("SI", "Técnico", "Inglés")
UM = 1e-6


def _ex(table: dict[str, rd.RadiationExample], prefix: str) -> rd.RadiationExample:
    return next(e for k, e in table.items() if k.startswith(prefix))


# ---------------------------------------------------------------------
# Cuerpo negro
# ---------------------------------------------------------------------


def _fraction_by_quadrature(lambda_T_mK: float) -> float:
    T = 1000.0
    lam = lambda_T_mK / T
    value, _ = quad(lambda x: planck(x, T), 1e-9, lam, limit=500, epsabs=0, epsrel=1e-13)
    return value / (SIGMA * T**4)


@pytest.mark.parametrize(
    "lambda_T_um_K",
    [800, 1000, 1500, 2000, 2898, 4000, 5000, 7000, 10_000, 14_000, 14_388, 14_500, 20_000, 50_000],
)
def test_fraction_series_against_quadrature(lambda_T_um_K: float) -> None:
    lam_T = lambda_T_um_K * UM
    # σ, C₁ y C₂ de CODATA están redondeados a 10 cifras: la cuadratura difiere ~10⁻⁹
    assert blackbody_fraction(lam_T) == pytest.approx(_fraction_by_quadrature(lam_T), abs=5e-9)


def test_fraction_table_and_inverse() -> None:
    # Incropera, tabla 12.1 (y la 12-2 de Cengel y Ghajar)
    assert blackbody_fraction(2898 * UM) == pytest.approx(0.250108, abs=5e-6)
    assert blackbody_fraction(0.0) == 0.0
    for f in (0.01, 0.1, 0.25, 0.5, 0.9, 0.99):
        assert blackbody_fraction(fraction_inverse(f)) == pytest.approx(f, abs=1e-12)
    with pytest.raises(ValueError, match="entre 0 y 1"):
        fraction_inverse(1.0)


def test_planck_wien_and_stefan_boltzmann() -> None:
    T = 1500.0
    total, _ = quad(lambda x: planck(x, T), 1e-8, 1e-3, limit=500)
    assert total == pytest.approx(SIGMA * T**4, rel=1e-6)
    r = solve_blackbody(BlackbodyInputs(T))
    assert r.lambda_max_m * T == pytest.approx(2897.771955 * UM, rel=1e-9)
    # el máximo es el máximo
    assert planck(r.lambda_max_m * 1.01, T) < r.E_blambda_max_W_per_m3
    assert planck(r.lambda_max_m * 0.99, T) < r.E_blambda_max_W_per_m3
    assert planck(1e-9, 300.0) == 0.0  # sin desborde


def test_cengel_12_1_black_ball() -> None:
    r = solve_blackbody(_ex(rd.BLACKBODY_EXAMPLES, "Bola negra").inputs)
    assert r.E_b_W_per_m2 == pytest.approx(23.2e3, rel=2e-3)
    assert r.E_blambda_point_W_per_m3 / 1e6 == pytest.approx(3846.0, abs=1.0)  # type: ignore[operator]
    area = math.pi * 0.2**2
    assert r.E_b_W_per_m2 * area * 300.0 / 1e3 == pytest.approx(876.0, rel=2e-3)  # kJ en 5 min


def test_visible_fractions_and_the_ten_percent() -> None:
    lamp = solve_blackbody(_ex(rd.BLACKBODY_EXAMPLES, "Filamento").inputs)
    assert lamp.band_fraction == pytest.approx(0.0518, abs=5e-4)
    sun = solve_blackbody(_ex(rd.BLACKBODY_EXAMPLES, "El sol").inputs)
    assert sun.band_fraction == pytest.approx(0.426, abs=1e-3)
    assert 0.40 * UM < sun.lambda_max_m < 0.76 * UM
    enclosure = solve_blackbody(_ex(rd.BLACKBODY_EXAMPLES, "Recinto a 2000 K").inputs)
    assert enclosure.lambda_target_m / UM == pytest.approx(1.10, abs=0.005)  # type: ignore[operator]
    ninety = solve_blackbody(replace(enclosure.inputs, fraction_target=0.9))
    assert ninety.lambda_target_m / UM == pytest.approx(4.69, abs=0.005)  # type: ignore[operator]
    assert enclosure.E_blambda_max_W_per_m3 == pytest.approx(4.12e11, rel=2e-3)


def test_band_emissivity_and_selective_absorber() -> None:
    r = solve_blackbody(_ex(rd.BLACKBODY_EXAMPLES, "Superficie con ε").inputs)
    assert r.emissivity == pytest.approx(0.521, abs=5e-4)
    assert r.E_W_per_m2 == pytest.approx(12.1e3, rel=2e-3)
    assert sum(row.fraction for row in r.rows) == pytest.approx(1.0)
    absorber = solve_blackbody(_ex(rd.BLACKBODY_EXAMPLES, "Absorbedor").inputs)
    assert absorber.absorptivity > 0.9 > 0.1 > absorber.emissivity  # type: ignore[operator]
    assert "Kirchhoff" in " ".join(rd.blackbody_notes(absorber))
    # sin bandas es un cuerpo negro
    black = solve_blackbody(BlackbodyInputs(600.0))
    assert black.emissivity is None and black.E_W_per_m2 == black.E_b_W_per_m2
    # con una sola banda de ε constante es una superficie gris (ε = α)
    gray = solve_blackbody(BlackbodyInputs(600.0, bands=(SpectralBand(0.4),), T_source_K=5800))
    assert gray.emissivity == pytest.approx(0.4) and gray.absorptivity == pytest.approx(0.4)


def test_blackbody_messages() -> None:
    with pytest.raises(ValueError, match="crecer"):
        solve_blackbody(BlackbodyInputs(500.0, bands=(SpectralBand(0.5, 3 * UM),
                                                      SpectralBand(0.3, 2 * UM),
                                                      SpectralBand(0.1))))  # fmt: skip
    with pytest.raises(ValueError, match="última banda"):
        solve_blackbody(BlackbodyInputs(500.0, bands=(SpectralBand(0.5, 3 * UM),)))
    with pytest.raises(ValueError, match="entre 0 y 1"):
        solve_blackbody(BlackbodyInputs(500.0, bands=(SpectralBand(1.5),)))
    with pytest.raises(ValueError, match="absoluta positiva"):
        solve_blackbody(BlackbodyInputs(0.0))
    with pytest.raises(ValueError, match="banda"):
        solve_blackbody(BlackbodyInputs(500.0, band_lower_m=2 * UM, band_upper_m=1 * UM))


# ---------------------------------------------------------------------
# Factores de forma
# ---------------------------------------------------------------------


def _mc_parallel_rectangles(X: float, Y: float, L: float, n: int = 1_000_000) -> float:
    rng = np.random.default_rng(7)
    x1, y1, x2, y2 = (rng.random(n) * d for d in (X, Y, X, Y))
    r2 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + L * L
    return float(np.mean(L * L / (math.pi * r2 * r2)) * X * Y)


def _mc_coaxial_disks(ri: float, rj: float, L: float, n: int = 1_000_000) -> float:
    rng = np.random.default_rng(11)
    r1 = ri * np.sqrt(rng.random(n))
    t1 = 2 * math.pi * rng.random(n)
    r2 = rj * np.sqrt(rng.random(n))
    t2 = 2 * math.pi * rng.random(n)
    dx = r1 * np.cos(t1) - r2 * np.cos(t2)
    dy = r1 * np.sin(t1) - r2 * np.sin(t2)
    s2 = dx * dx + dy * dy + L * L
    return float(np.mean(L * L / (math.pi * s2 * s2)) * math.pi * rj * rj)


@pytest.mark.parametrize(
    "inputs,mc",
    [
        (ViewFactorInputs("parallel_rectangles", 1.0, 1.0, 1.0), _mc_parallel_rectangles),
        (ViewFactorInputs("parallel_rectangles", 2.0, 0.5, 0.7), _mc_parallel_rectangles),
        (ViewFactorInputs("coaxial_disks", 0.5, 0.8, 1.0), _mc_coaxial_disks),
    ],
)
def test_3d_view_factors_against_monte_carlo(inputs: ViewFactorInputs, mc) -> None:  # noqa: ANN001
    r = solve_view_factor(inputs)
    assert r.F_ij == pytest.approx(mc(inputs.a, inputs.b, inputs.c), abs=1.5e-3)
    assert r.A_i * r.F_ij == pytest.approx(r.A_j * r.F_ji)


@pytest.mark.parametrize("X,Y,H", [(1.0, 1.0, 1.0), (4.0, 3.0, 2.6), (1.0, 2.0, 0.5)])
def test_perpendicular_rectangles_close_a_box(X: float, Y: float, H: float) -> None:
    """En una caja X × Y × H el piso ve el techo y las cuatro paredes: suman 1.

    Une dos fórmulas independientes (rectángulos paralelos y perpendiculares)."""
    ceiling = solve_view_factor(ViewFactorInputs("parallel_rectangles", X, Y, H)).F_ij
    wall_x = solve_view_factor(ViewFactorInputs("perpendicular_rectangles", X, Y, H)).F_ij
    wall_y = solve_view_factor(ViewFactorInputs("perpendicular_rectangles", Y, X, H)).F_ij
    assert ceiling + 2.0 * wall_x + 2.0 * wall_y == pytest.approx(1.0, abs=1e-12)


def _crossed_strings(a1, a2, b1, b2) -> float:  # noqa: ANN001
    """F de la superficie a1–a2 a la b1–b2 (Hottel): (Σ cruzadas − Σ sin cruzar)/(2·L₁)."""

    def d(p, q) -> float:  # noqa: ANN001
        return math.dist(p, q)

    crossed = d(a1, b2) + d(a2, b1)
    uncrossed = d(a1, b1) + d(a2, b2)
    return (crossed - uncrossed) / (2.0 * d(a1, a2))


def test_2d_view_factors_against_crossed_strings() -> None:
    wi, wj, L = 0.4, 0.6, 0.3
    r = solve_view_factor(ViewFactorInputs("parallel_plates_2d", wi, wj, L))
    expected = _crossed_strings((-wi / 2, 0), (wi / 2, 0), (-wj / 2, L), (wj / 2, L))
    assert r.F_ij == pytest.approx(abs(expected), rel=1e-12)
    r = solve_view_factor(ViewFactorInputs("perpendicular_plates_2d", 1.0, 2.0))
    expected = _crossed_strings((1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 2.0))
    assert r.F_ij == pytest.approx(abs(expected), rel=1e-12)
    # iguales y con un borde común: 1 − sin(45°)
    r = solve_view_factor(ViewFactorInputs("perpendicular_plates_2d", 1.0, 1.0))
    assert r.F_ij == pytest.approx(1.0 - math.sin(math.radians(45.0)))
    # triángulo equilátero: 1/2
    assert solve_view_factor(ViewFactorInputs("three_sided_2d", 1.0, 1.0, 1.0)).F_ij == 0.5


def test_view_factor_limits() -> None:
    # cilindros que se tocan: 1/2 − 1/π
    r = solve_view_factor(ViewFactorInputs("parallel_cylinders_2d", 0.1, 0.0))
    assert r.F_ij == pytest.approx(0.5 - 1.0 / math.pi)
    # tubos que se tocan: el plano solo ve tubos
    assert solve_view_factor(ViewFactorInputs("tube_row_2d", 0.1, 0.1)).F_ij == pytest.approx(1.0)
    # placas o discos muy cerca: casi 1
    assert solve_view_factor(ViewFactorInputs("parallel_rectangles", 10, 10, 0.01)).F_ij > 0.99
    assert solve_view_factor(ViewFactorInputs("coaxial_disks", 1, 1, 0.001)).F_ij > 0.99
    # ejemplos de los libros
    cube = solve_view_factor(_ex(rd.VIEW_FACTOR_EXAMPLES, "Techo y piso").inputs)
    assert cube.F_ij == pytest.approx(0.1998, abs=1e-4)
    cavity = solve_view_factor(_ex(rd.VIEW_FACTOR_EXAMPLES, "El fondo y la boca").inputs)
    assert cavity.F_ij == pytest.approx(0.0557, abs=1e-4)


def test_view_factor_curve_and_messages() -> None:
    for e in rd.VIEW_FACTOR_EXAMPLES.values():
        label, xs, Fs = rd.view_factor_curve(e.inputs)
        assert label and xs == sorted(xs) and all(0.0 <= F <= 1.0 + 1e-12 for F in Fs)
        r = solve_view_factor(e.inputs)
        assert r.F_ij in Fs  # el punto está en la curva
        assert rd.view_factor_notes(r)
    with pytest.raises(ValueError, match="triángulo"):
        solve_view_factor(ViewFactorInputs("three_sided_2d", 1.0, 1.0, 3.0))
    with pytest.raises(ValueError, match="paso"):
        solve_view_factor(ViewFactorInputs("tube_row_2d", 0.2, 0.1))
    with pytest.raises(ValueError, match="positivo"):
        solve_view_factor(ViewFactorInputs("coaxial_disks", 0.0, 1.0, 1.0))


# ---------------------------------------------------------------------
# Dos superficies y pantallas
# ---------------------------------------------------------------------


def test_cengel_parallel_plates_and_shield() -> None:
    r0 = solve_two_surface(_ex(rd.TWO_SURFACE_EXAMPLES, "Placas paralelas").inputs)
    assert r0.Q_W == pytest.approx(3625.0, abs=1.0)
    r1 = solve_two_surface(_ex(rd.TWO_SURFACE_EXAMPLES, "Las mismas con una pantalla").inputs)
    assert r1.Q_W == pytest.approx(806.0, abs=0.5)
    assert r1.Q_no_shields_W == pytest.approx(r0.Q_W)
    # la temperatura de la pantalla cierra los dos lados de la red
    Eb3 = SIGMA * r1.shield_T_K[0] ** 4
    R = [x.R_per_m2 for x in r1.resistances]
    assert (SIGMA * 800**4 - Eb3) / sum(R[:3]) == pytest.approx(r1.Q_W, rel=1e-12)
    assert (Eb3 - SIGMA * 500**4) / sum(R[3:]) == pytest.approx(r1.Q_W, rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_equal_shields_divide_by_n_plus_one(n: int) -> None:
    base = TwoSurfaceInputs("parallel_plates", 700.0, 300.0, 0.5, 0.5)
    shielded = replace(base, shields=tuple(Shield(0.5, 0.5) for _ in range(n)))
    assert solve_two_surface(shielded).Q_W == pytest.approx(
        solve_two_surface(base).Q_W / (n + 1), rel=1e-12
    )


def _two_surface_as_enclosure(A1: float, A2: float, inputs: TwoSurfaceInputs) -> float:
    F = ((0.0, 1.0), (A1 / A2, 1.0 - A1 / A2))
    surfaces = (
        EnclosureSurface("1", A1, inputs.eps1, inputs.T1_K),
        EnclosureSurface("2", A2, inputs.eps2, inputs.T2_K),
    )
    return solve_enclosure(EnclosureInputs(surfaces, F)).Q_W[0]


def test_concentric_geometries_agree_with_the_enclosure() -> None:
    cyl = TwoSurfaceInputs("concentric_cylinders", 600.0, 300.0, 0.3, 0.6, r1_m=0.05, r2_m=0.1)
    A1, A2 = 2 * math.pi * 0.05, 2 * math.pi * 0.1
    assert solve_two_surface(cyl).Q_W == pytest.approx(_two_surface_as_enclosure(A1, A2, cyl))
    sph = TwoSurfaceInputs("concentric_spheres", 77.0, 300.0, 0.02, 0.05, r1_m=0.25, r2_m=0.275)
    A1, A2 = 4 * math.pi * 0.25**2, 4 * math.pi * 0.275**2
    Q = solve_two_surface(sph).Q_W
    assert Q == pytest.approx(_two_surface_as_enclosure(A1, A2, sph))
    expected = SIGMA * A1 * (77**4 - 300**4) / (1 / 0.02 + (1 - 0.05) / 0.05 * (0.25 / 0.275) ** 2)
    assert Q == pytest.approx(expected, rel=1e-12)
    general = TwoSurfaceInputs("general", 600.0, 300.0, 0.3, 0.6, A1_m2=A1, A2_m2=A2, F12=1.0)
    assert solve_two_surface(general).Q_W == pytest.approx(
        _two_surface_as_enclosure(A1, A2, general)
    )


def test_small_object_is_epsilon_times_blackbody() -> None:
    r = solve_two_surface(_ex(rd.TWO_SURFACE_EXAMPLES, "Caño de vapor desnudo").inputs)
    assert r.Q_W == pytest.approx(421.0, abs=0.5)  # Incropera, ejemplo 1.2
    assert "emisividad del recinto no importa" in " ".join(rd.two_surface_notes(r))


def test_two_surface_messages() -> None:
    with pytest.raises(ValueError, match="crecer"):
        solve_two_surface(
            TwoSurfaceInputs("concentric_cylinders", 500, 300, 0.5, 0.5, r1_m=0.1, r2_m=0.05)
        )
    with pytest.raises(ValueError, match="pantallas"):
        solve_two_surface(
            TwoSurfaceInputs("small_object", 500, 300, 0.5, 0.5, shields=(Shield(0.1, 0.1),))
        )
    with pytest.raises(ValueError, match="reciprocidad"):
        solve_two_surface(TwoSurfaceInputs("general", 500, 300, 0.5, 0.5, 2.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="emisividad"):
        solve_two_surface(TwoSurfaceInputs("parallel_plates", 500, 300, 0.0, 0.5))


# ---------------------------------------------------------------------
# Recintos
# ---------------------------------------------------------------------


def test_incropera_paint_oven_with_a_reradiating_wall() -> None:
    r = solve_enclosure(_ex(rd.ENCLOSURE_EXAMPLES, "Horno de pintura").inputs)
    assert r.Q_W[0] == pytest.approx(37.0e3, rel=2e-3)
    assert r.T_K[2] == pytest.approx(1102.0, abs=0.5)
    assert r.Q_W[2] == pytest.approx(0.0, abs=1e-9)
    assert r.J_W_per_m2[2] == pytest.approx(SIGMA * r.T_K[2] ** 4)  # rerradiante: J = E_b
    assert "rerradiante" in " ".join(rd.enclosure_notes(r))


def test_cengel_triangular_duct_with_heat_given() -> None:
    r = solve_enclosure(_ex(rd.ENCLOSURE_EXAMPLES, "Ducto triangular").inputs)
    assert r.T_K[0] == pytest.approx(543.0, abs=0.5)
    assert r.Q_W[0] == pytest.approx(800.0)


def test_incropera_open_cavity() -> None:
    r = solve_enclosure(_ex(rd.ENCLOSURE_EXAMPLES, "Cavidad cilíndrica").inputs)
    assert -r.Q_W[2] == pytest.approx(1830.0, rel=2e-3)
    # todas negras: Q̇_ij = A_i·F_ij·σ(T_i⁴ − T_j⁴)
    S = r.inputs.surfaces
    for i in range(3):
        for j in range(3):
            expected = S[i].area_m2 * r.inputs.F[i][j] * SIGMA * (r.T_K[i] ** 4 - r.T_K[j] ** 4)
            assert r.Q_between(i, j) == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_every_enclosure_balances() -> None:
    for e in rd.ENCLOSURE_EXAMPLES.values():
        r = solve_enclosure(e.inputs)
        scale = max(abs(q) for q in r.Q_W)
        assert sum(r.Q_W) == pytest.approx(0.0, abs=1e-9 * scale)
        for i, s in enumerate(r.inputs.surfaces):
            # cada Q̇_i es la suma de lo que intercambia con las demás
            assert sum(r.Q_between(i, j) for j in range(len(r.inputs.surfaces))) == pytest.approx(
                r.Q_W[i], rel=1e-9, abs=1e-9
            )
            if s.T_K is None and not s.reradiating and s.emissivity < 1.0:
                Eb = r.J_W_per_m2[i] + r.Q_W[i] * (1 - s.emissivity) / (s.area_m2 * s.emissivity)
                assert SIGMA * r.T_K[i] ** 4 == pytest.approx(Eb)


def test_enclosure_messages() -> None:
    a = EnclosureSurface("a", 1.0, 0.5, 500.0)
    b = EnclosureSurface("b", 1.0, 0.5, 300.0)
    with pytest.raises(ValueError, match="sumar 1"):
        solve_enclosure(EnclosureInputs((a, b), ((0.0, 0.9), (1.0, 0.0))))
    with pytest.raises(ValueError, match="reciprocidad"):
        solve_enclosure(EnclosureInputs((a, replace(b, area_m2=2.0)), ((0.0, 1.0), (1.0, 0.0))))
    with pytest.raises(ValueError, match="uno solo"):
        solve_enclosure(EnclosureInputs((replace(a, Q_W=1.0), b), ((0.0, 1.0), (1.0, 0.0))))
    with pytest.raises(ValueError, match="Al menos una"):
        both_q = (replace(a, T_K=None, Q_W=1.0), replace(b, T_K=None, Q_W=-1.0))
        solve_enclosure(EnclosureInputs(both_q, ((0.0, 1.0), (1.0, 0.0))))
    with pytest.raises(ValueError, match="cero absoluto"):
        cold_q = replace(b, T_K=None, Q_W=-1e6)
        solve_enclosure(EnclosureInputs((a, cold_q), ((0.0, 1.0), (1.0, 0.0))))


# ---------------------------------------------------------------------
# Radiación y convección juntas
# ---------------------------------------------------------------------


def test_incropera_1_2_steam_pipe() -> None:
    r = solve_surface_balance(_ex(rd.SURFACE_EXAMPLES, "Caño de vapor").inputs)
    assert r.Q_conv_W == pytest.approx(577.0, abs=0.5)
    assert r.Q_rad_W == pytest.approx(421.0, abs=0.5)
    assert r.Q_lost_W == pytest.approx(998.0, abs=1.0)
    i = r.inputs
    assert r.Q_rad_W == pytest.approx(r.h_rad_W_per_m2K * i.area_m2 * (r.T_s_K - i.T_surr_K))


@pytest.mark.parametrize(
    "alpha,eps,expected",
    [(0.9, 0.9, 306.4), (0.1, 0.1, 34.0), (0.9, 0.1, 574.7), (0.1, 0.9, -234.3)],
)
def test_cengel_solar_surfaces(alpha: float, eps: float, expected: float) -> None:
    base = _ex(rd.SURFACE_EXAMPLES, "Absorbedor gris").inputs
    r = solve_surface_balance(replace(base, alpha_solar=alpha, emissivity=eps))
    assert -r.Q_lost_W == pytest.approx(expected, abs=0.1)


def test_equilibrium_temperature() -> None:
    r = solve_surface_balance(_ex(rd.SURFACE_EXAMPLES, "Resistencia").inputs)
    assert r.Q_lost_W == pytest.approx(1000.0, rel=1e-9)
    assert 400.0 < r.T_s_K < 550.0
    with pytest.raises(ValueError, match="temperatura de la superficie o el calor"):
        solve_surface_balance(SurfaceBalanceInputs(0.5, 10.0, 300.0, 300.0))
    with pytest.raises(ValueError, match="emisividad"):
        solve_surface_balance(SurfaceBalanceInputs(1.5, 10.0, 300.0, 300.0, T_s_K=400.0))


def test_thermocouple() -> None:
    r = solve_thermocouple(_ex(rd.THERMOCOUPLE_EXAMPLES, "Termocupla").inputs)
    assert r.T_gas_K == pytest.approx(715.0, abs=0.1)
    shiny = solve_thermocouple(_ex(rd.THERMOCOUPLE_EXAMPLES, "La misma").inputs)
    assert shiny.error_K == pytest.approx(r.error_K / 6.0)
    assert r.q_rad_W_per_m2 == pytest.approx(r.inputs.h_W_per_m2K * r.error_K)
    assert "menos" in rd.thermocouple_notes(r)[0]
    with pytest.raises(ValueError, match="h tiene que ser positivo"):
        solve_thermocouple(ThermocoupleInputs(650.0, 400.0, 0.6, 0.0))


# ---------------------------------------------------------------------
# Notas, export y pickle
# ---------------------------------------------------------------------


def _all() -> list[tuple[object, object, object]]:
    out: list[tuple[object, object, object]] = []
    for e in rd.BLACKBODY_EXAMPLES.values():
        out.append((solve_blackbody(e.inputs), rd.blackbody_notes, rd.blackbody_to_dict))
    for e in rd.VIEW_FACTOR_EXAMPLES.values():
        out.append((solve_view_factor(e.inputs), rd.view_factor_notes, rd.view_factor_to_dict))
    for e in rd.TWO_SURFACE_EXAMPLES.values():
        out.append((solve_two_surface(e.inputs), rd.two_surface_notes, rd.two_surface_to_dict))
    for e in rd.ENCLOSURE_EXAMPLES.values():
        out.append((solve_enclosure(e.inputs), rd.enclosure_notes, rd.enclosure_to_dict))
    for e in rd.SURFACE_EXAMPLES.values():
        out.append((solve_surface_balance(e.inputs), rd.surface_balance_notes, rd.surface_to_dict))
    for e in rd.THERMOCOUPLE_EXAMPLES.values():
        out.append((solve_thermocouple(e.inputs), rd.thermocouple_notes, rd.thermocouple_to_dict))
    return out


def test_notes_export_and_pickle() -> None:
    for result, notes, to_dict in _all():
        texts = notes(result)  # type: ignore[operator]
        assert texts and all(isinstance(t, str) and t for t in texts)
        for system in SYSTEMS:
            data = to_dict(result, system)  # type: ignore[operator]
            json.dumps(data, ensure_ascii=False)
            assert dict_to_csv(data)
        assert pickle.loads(pickle.dumps(result)) == result
