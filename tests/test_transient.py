"""Tests de la conducción transitoria (Fase 11.1).

Contra Çengel y Ghajar (2015), cap. 4, e Incropera et al. (2007), cap. 5: los enunciados
traen las propiedades, así que los resultados son exactos salvo el redondeo de los libros
(que leen λ₁ y A₁ de una tabla).
"""

from __future__ import annotations

import json
import math
import pickle

import pytest
from scipy.integrate import quad
from scipy.special import erfc, j0, j1

from core.heat_transfer import transient as tr

C = 273.15


def _lumped(name: str) -> tr.LumpedResult:
    return tr.solve_lumped(tr.LUMPED_EXAMPLES[name].inputs)


def _one(name: str) -> tr.OneDimResult:
    return tr.solve_one_dimensional(tr.ONE_DIM_EXAMPLES[name].inputs)


def _semi(name: str) -> tr.SemiInfiniteResult:
    return tr.solve_semi_infinite(tr.SEMI_INFINITE_EXAMPLES[name].inputs)


def _product(name: str) -> tr.ProductResult:
    return tr.solve_product(tr.PRODUCT_EXAMPLES[name].inputs)


# ---------------------------------------------------------------------
# Raíces y coeficientes (Çengel y Ghajar, tabla 4-2)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("geometry", "Bi", "lam", "A"),
    [
        ("wall", 0.1, 0.3111, 1.0161),
        ("wall", 1.0, 0.8603, 1.1191),
        ("wall", 10.0, 1.4289, 1.2620),
        ("wall", 1e9, 1.5708, 1.2732),
        ("cylinder", 0.1, 0.4417, 1.0246),
        ("cylinder", 1.0, 1.2558, 1.2071),
        ("cylinder", 10.0, 2.1795, 1.5677),
        ("cylinder", 1e9, 2.4048, 1.6020),
        ("sphere", 0.1, 0.5423, 1.0298),
        ("sphere", 1.0, 1.5708, 1.2732),
        ("sphere", 10.0, 2.8363, 1.9249),
        ("sphere", 1e9, 3.1416, 2.0000),
    ],
)
def test_first_root_and_coefficient_match_the_table(
    geometry: tr.Geometry1D, Bi: float, lam: float, A: float
) -> None:
    lam1 = tr.eigenvalues(geometry, Bi, 1)[0]
    assert lam1 == pytest.approx(lam, abs=6e-5)
    assert tr.coefficient(geometry, lam1) == pytest.approx(A, abs=6e-5)


@pytest.mark.parametrize("geometry", ["wall", "cylinder", "sphere"])
@pytest.mark.parametrize("Bi", [1e-4, 0.05, 2.0, 80.0])
def test_roots_solve_the_characteristic_equation(geometry: tr.Geometry1D, Bi: float) -> None:
    roots = tr.eigenvalues(geometry, Bi, 8)
    assert all(b > a for a, b in zip(roots, roots[1:], strict=False))  # crecientes
    for lam in roots:
        if geometry == "wall":
            residual = lam * math.tan(lam) - Bi
        elif geometry == "cylinder":
            residual = lam * float(j1(lam)) - Bi * float(j0(lam))
        else:
            residual = 1.0 - lam / math.tan(lam) - Bi
        assert residual == pytest.approx(0.0, abs=1e-8 * max(1.0, Bi))


@pytest.mark.parametrize("geometry", ["wall", "cylinder", "sphere"])
def test_series_is_one_at_the_start(geometry: tr.Geometry1D) -> None:
    """Al principio (τ → 0) el centro todavía está a T_i."""
    assert tr.theta(geometry, 5.0, 1e-3, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert tr.heat_ratio(geometry, 5.0, 1e-3) == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("geometry", ["wall", "cylinder", "sphere"])
@pytest.mark.parametrize("Bi", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("tau", [0.2, 0.5, 2.0])
def test_one_term_is_good_from_tau_02(geometry: tr.Geometry1D, Bi: float, tau: float) -> None:
    """Con τ ≥ 0,2 el primer término se aparta de la serie menos del 2 % (§4-2)."""
    series = tr.theta(geometry, Bi, tau, 0.0)
    one = tr.theta(geometry, Bi, tau, 0.0, terms=1)
    assert one == pytest.approx(series, rel=0.02)


@pytest.mark.parametrize("geometry", ["wall", "cylinder", "sphere"])
@pytest.mark.parametrize("tau", [0.05, 0.3, 1.5])
def test_heat_ratio_is_the_integral_of_the_profile(geometry: tr.Geometry1D, tau: float) -> None:
    """Q/Q_máx = 1 − θ medio, con la media en el volumen de cada geometría."""
    Bi = 2.0
    weight = {"wall": lambda x: 1.0, "cylinder": lambda x: 2.0 * x, "sphere": lambda x: 3.0 * x * x}
    w = weight[geometry]
    mean, _ = quad(lambda x: tr.theta(geometry, Bi, tau, x) * w(x), 0.0, 1.0, limit=200)
    assert tr.heat_ratio(geometry, Bi, tau) == pytest.approx(1.0 - mean, abs=1e-6)


# ---------------------------------------------------------------------
# Capacidad concentrada (§4-1)
# ---------------------------------------------------------------------


def test_thermocouple() -> None:
    r = _lumped("Termocupla en un gas (Çengel y Ghajar §4-1)")
    assert r.Bi == pytest.approx(0.001, rel=0.01)
    assert r.b == pytest.approx(0.462, rel=0.01)
    assert r.time_s == pytest.approx(10.0, abs=0.1)  # el libro: 10 s


def test_time_of_death() -> None:
    r = _lumped("La hora de la muerte (Çengel y Ghajar §4-1)")
    assert r.Lc == pytest.approx(0.0689, rel=1e-3)
    assert r.Bi == pytest.approx(0.89, abs=0.01)
    assert r.time_s / 3600.0 == pytest.approx(12.2, abs=0.05)  # el libro: 12,2 h
    # con Bi = 0,89 la exacta se aparta: el centro sigue más caliente que la media
    assert r.exact_mean_K is not None and r.exact_center_K is not None
    assert r.exact_center_K > r.exact_mean_K > r.T_K
    assert any("no" in n and "uniforme" in n for n in r.notes)


def test_incropera_thermocouple_design() -> None:
    r = _lumped("Termocupla con constante de tiempo de 1 s (Incropera §5.2)")
    assert r.inputs.dims[0] == pytest.approx(0.706e-3, rel=1e-3)
    assert r.time_constant_s == pytest.approx(1.0, rel=1e-9)
    assert r.time_s == pytest.approx(5.2, abs=0.05)  # el libro: 5,2 s


def test_lumped_matches_the_exact_solution_with_small_biot() -> None:
    inp = tr.LumpedInputs("sphere", (0.01,), tr.SOLIDS["copper"], 50.0, 400.0, 300.0, time_s=60.0)
    r = tr.solve_lumped(inp)
    assert r.Bi < 0.01
    assert r.exact_mean_K is not None
    assert r.T_K == pytest.approx(r.exact_mean_K, abs=0.05)


def test_time_constant_is_63_percent() -> None:
    inp = tr.LumpedInputs("cube", (0.02,), tr.SOLIDS["aluminum"], 30.0, 373.15, 293.15, time_s=1.0)
    tau = tr.solve_lumped(inp).time_constant_s
    r = tr.solve_lumped(
        tr.LumpedInputs("cube", (0.02,), tr.SOLIDS["aluminum"], 30.0, 373.15, 293.15, time_s=tau)
    )
    assert 1.0 - r.theta == pytest.approx(0.632, abs=1e-3)
    assert r.Q_J == pytest.approx(r.Q_max_J * (1.0 - math.exp(-1.0)))
    ts, Ts = tr.lumped_curve(r)
    assert all(b <= a for a, b in zip(Ts, Ts[1:], strict=False))  # se enfría


@pytest.mark.parametrize(
    ("body", "dims", "Lc"),
    [
        ("sphere", (0.06,), 0.01),
        ("long_cylinder", (0.04,), 0.01),
        ("plate", (0.02,), 0.01),
        ("cube", (0.06,), 0.01),
        ("box", (1.0, 2.0, 3.0), 6.0 / 22.0),
        ("custom", (2.0, 4.0), 0.5),
    ],
)
def test_characteristic_length(body: tr.Body, dims: tuple[float, ...], Lc: float) -> None:
    V, A = tr.body_volume_area(body, dims)
    assert V / A == pytest.approx(Lc)


# ---------------------------------------------------------------------
# Pared, cilindro y esfera (§4-2)
# ---------------------------------------------------------------------


def test_boiled_egg() -> None:
    r = _one("Huevo hervido (Çengel y Ghajar §4-2)")
    assert r.Bi == pytest.approx(50.0)
    assert r.time_s == pytest.approx(933.0, rel=0.01)  # el libro redondea θ₀
    assert r.T_center_K - C == pytest.approx(70.0, abs=1e-6)
    assert r.time_one_term_s is not None and r.time_one_term_s == pytest.approx(r.time_s, rel=2e-3)


def test_brass_plates() -> None:
    r = _one("Placas de bronce en un horno (Çengel y Ghajar §4-2)")
    assert r.Bi == pytest.approx(0.0218, rel=1e-3)
    assert r.T_K - C == pytest.approx(279.0, abs=1.0)  # el libro: 279 °C
    assert r.T_K == pytest.approx(r.T_one_K, abs=0.01)  # τ grande: un término alcanza


def test_stainless_shaft() -> None:
    r = _one("Eje de acero inoxidable (Çengel y Ghajar §4-2)")
    assert r.Bi == pytest.approx(0.537, abs=1e-3)
    assert r.tau == pytest.approx(1.07, abs=0.01)
    assert r.T_center_K - C == pytest.approx(364.0, abs=0.5)  # el libro: 364 °C
    assert r.Q / 1e3 == pytest.approx(30_120.0, rel=2e-3)  # el libro: 30 120 kJ por metro


def test_steaks_reach_45_and_35_F_together() -> None:
    r = _one("Bifes que no se tienen que escarchar (Çengel y Ghajar §4-2)")
    to_F = lambda T: (T - C) * 1.8 + 32.0  # noqa: E731
    assert to_F(r.T_center_K) == pytest.approx(45.0, abs=1e-6)
    assert to_F(r.T_surface_K) == pytest.approx(35.0, abs=0.1)
    assert r.time_s / 3600.0 == pytest.approx(0.6, abs=0.02)


def test_one_term_warning_for_short_times() -> None:
    inp = tr.OneDimInputs("wall", 0.05, tr.SOLIDS["concrete"], 20.0, 350.0, 300.0, time_s=60.0)
    r = tr.solve_one_dimensional(inp)
    assert r.tau < 0.2
    assert any("no alcanza" in n for n in r.notes)


def test_profiles_and_history() -> None:
    r = _one("Eje de acero inoxidable (Çengel y Ghajar §4-2)")
    profiles = tr.time_profiles(r)
    assert len(profiles) == 5
    t, xs, Ts = profiles[-1]
    assert t == pytest.approx(r.time_s)
    assert Ts[0] == pytest.approx(r.T_center_K)  # ξ = 0 es el centro
    assert Ts[-1] == pytest.approx(r.T_surface_K)
    ts, series, one = tr.center_history(r)
    assert all(b <= a + 1e-9 for a, b in zip(series, series[1:], strict=False))


# ---------------------------------------------------------------------
# Semi-infinito (§4-3)
# ---------------------------------------------------------------------


def test_water_pipes_burial_depth() -> None:
    r = _semi("Caños de agua enterrados (Çengel y Ghajar §4-3)")
    assert r.depth_m == pytest.approx(0.80, abs=0.005)  # el libro: 0,80 m
    assert r.T_K - C == pytest.approx(0.0, abs=1e-6)


def test_incropera_water_main() -> None:
    r = _semi("Caño de agua en el invierno (Incropera §5.7)")
    assert r.depth_m == pytest.approx(0.68, abs=0.005)  # el libro: 0,68 m


@pytest.mark.parametrize(
    ("name", "T_book"),
    [
        ("Bloque de madera al sol (Çengel y Ghajar §4-3)", 149.0),
        ("Bloque de aluminio al sol (Çengel y Ghajar §4-3)", 22.0),
    ],
)
def test_blocks_in_the_sun(name: str, T_book: float) -> None:
    r = _semi(name)
    assert r.T_K - C == pytest.approx(T_book, abs=0.5)
    assert r.energy_per_area == pytest.approx(1250.0 * 1200.0)


def test_temperature_boundary_closed_forms() -> None:
    s = tr.SOLIDS["granite"]
    inp = tr.SemiInfiniteInputs("temperature", s, 300.0, T_s=400.0, depth_m=0.03, time_s=600.0)
    r = tr.solve_semi_infinite(inp)
    eta = 0.03 / (2.0 * math.sqrt(s.alpha * 600.0))
    assert r.T_K == pytest.approx(400.0 - 100.0 * (1.0 - float(erfc(eta))))
    # la energía que entró: la forma cerrada contra la integral del flujo
    integral, _ = quad(lambda t: s.k * 100.0 / math.sqrt(math.pi * s.alpha * t), 0.0, 600.0)
    assert r.energy_per_area == pytest.approx(integral, rel=1e-6)


def test_convection_tends_to_the_surface_temperature_case() -> None:
    s = tr.SOLIDS["concrete"]
    conv = tr.SemiInfiniteInputs(
        "convection", s, 300.0, h=1e7, T_inf=400.0, depth_m=0.02, time_s=3600.0
    )
    temp = tr.SemiInfiniteInputs("temperature", s, 300.0, T_s=400.0, depth_m=0.02, time_s=3600.0)
    assert tr.solve_semi_infinite(conv).T_K == pytest.approx(
        tr.solve_semi_infinite(temp).T_K, abs=1e-3
    )


def test_semi_infinite_inverse_problems_round_trip() -> None:
    s = tr.SOLIDS["oak"]
    base = tr.SemiInfiniteInputs(
        "convection", s, 290.0, h=40.0, T_inf=500.0, depth_m=0.004, time_s=900.0
    )
    T = tr.solve_semi_infinite(base).T_K
    by_depth = tr.SemiInfiniteInputs(
        "convection", s, 290.0, h=40.0, T_inf=500.0, time_s=900.0, T_target_K=T
    )
    by_time = tr.SemiInfiniteInputs(
        "convection", s, 290.0, h=40.0, T_inf=500.0, depth_m=0.004, T_target_K=T
    )
    assert tr.solve_semi_infinite(by_depth).depth_m == pytest.approx(0.004, rel=1e-6)
    assert tr.solve_semi_infinite(by_time).time_s == pytest.approx(900.0, rel=1e-6)


def test_semi_infinite_matches_a_thick_wall_at_short_times() -> None:
    """Al principio la pared no sabe que tiene otra cara: cerca de la superficie es
    semi-infinita (superficie a T_s ↔ h → ∞)."""
    s = tr.SOLIDS["concrete"]
    L, t = 0.5, 600.0
    wall = tr.OneDimInputs("wall", L, s, 1e8, 300.0, 400.0, position=1.0 - 0.01 / L, time_s=t)
    semi = tr.SemiInfiniteInputs("temperature", s, 300.0, T_s=400.0, depth_m=0.01, time_s=t)
    assert tr.solve_one_dimensional(wall).T_K == pytest.approx(
        tr.solve_semi_infinite(semi).T_K, abs=0.05
    )


def test_contact_temperature() -> None:
    al = tr.solve_contact(tr.CONTACT_EXAMPLES["La mano en aluminio (ejemplo propio)"].inputs)
    wood = tr.solve_contact(tr.CONTACT_EXAMPLES["La mano en madera (ejemplo propio)"].inputs)
    assert 20.0 < al.T_contact_K - C < 21.5  # casi la del aluminio
    assert 30.0 < wood.T_contact_K - C < 34.0  # cerca de la piel
    same = tr.solve_contact(tr.ContactInputs(tr.SOLIDS["iron"], 400.0, tr.SOLIDS["iron"], 300.0))
    assert same.T_contact_K == pytest.approx(350.0)


# ---------------------------------------------------------------------
# Multidimensional (§4-4)
# ---------------------------------------------------------------------


def test_short_brass_cylinder_is_the_product() -> None:
    r = _product("Cilindro corto de bronce (Çengel y Ghajar §4-4)")
    s = tr.SOLIDS["brass"]
    wall = tr.theta("wall", 60.0 * 0.06 / s.k, s.alpha * 900.0 / 0.06**2, 0.0)
    cyl = tr.theta("cylinder", 60.0 * 0.05 / s.k, s.alpha * 900.0 / 0.05**2, 0.0)
    assert r.theta == pytest.approx(wall * cyl)
    assert r.T_K - C == pytest.approx(62.7, abs=0.1)
    assert r.heat_ratio is not None and 0.0 < r.heat_ratio < 1.0
    assert r.Q is not None and r.Q > 0.0


def test_langston_rule_with_one_factor_is_the_factor() -> None:
    assert tr._combine_ratios([0.3]) == pytest.approx(0.3)
    assert tr._combine_ratios([0.3, 0.5]) == pytest.approx(0.3 + 0.5 * 0.7)


def test_semi_factor_has_no_heat_ratio() -> None:
    r = _product("Cilindro semi-infinito de aluminio en agua (Çengel y Ghajar §4-4)")
    assert r.heat_ratio is None and r.Q is None
    assert 100.0 < r.T_K - C < 200.0


def test_product_map_has_the_point() -> None:
    r = _product("Cilindro corto de bronce (Çengel y Ghajar §4-4)")
    m = tr.product_map(r)
    assert m is not None
    xs, ys, grid = m
    assert grid[0][0] == pytest.approx(r.T_K)  # el centro es el (0, 0) del mapa


# ---------------------------------------------------------------------
# Errores, export y pickle
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (
            lambda: tr.solve_lumped(
                tr.LumpedInputs("sphere", (0.01,), tr.SOLIDS["iron"], 10.0, 300.0, 300.0, 1.0)
            ),
            "no hay transitorio",
        ),
        (
            lambda: tr.solve_lumped(
                tr.LumpedInputs("sphere", (0.01,), tr.SOLIDS["iron"], 0.0, 300.0, 400.0, 1.0)
            ),
            "h tiene que ser positivo",
        ),
        (
            lambda: tr.solve_lumped(
                tr.LumpedInputs(
                    "sphere", (0.01,), tr.SOLIDS["iron"], 10.0, 300.0, 400.0, T_target_K=450.0
                )
            ),
            "tiene que estar entre",
        ),
        (
            lambda: tr.solve_lumped(
                tr.LumpedInputs("box", (0.01, 0.02), tr.SOLIDS["iron"], 10.0, 300.0, 400.0, 1.0)
            ),
            "hacen falta 3 medidas",
        ),
        (
            lambda: tr.solve_one_dimensional(
                tr.OneDimInputs("wall", 0.1, tr.SOLIDS["iron"], 10.0, 300.0, 400.0, 1.5, 1.0)
            ),
            "y 1 \\(la superficie\\)",
        ),
        (
            lambda: tr.solve_one_dimensional(
                tr.OneDimInputs("sphere", 0.1, tr.SOLIDS["iron"], 10.0, 300.0, 400.0, 0.0, 0.0)
            ),
            "tiempo tiene que ser positivo",
        ),
        (
            lambda: tr.solve_semi_infinite(
                tr.SemiInfiniteInputs("temperature", tr.SOLIDS["soil"], 300.0, T_s=270.0)
            ),
            "Dá dos de tres",
        ),
        (
            lambda: tr.solve_semi_infinite(
                tr.SemiInfiniteInputs(
                    "temperature",
                    tr.SOLIDS["soil"],
                    300.0,
                    T_s=270.0,
                    time_s=3600.0,
                    T_target_K=260.0,
                )
            ),
            "tiene que estar entre",
        ),
        (
            lambda: tr.solve_one_dimensional(
                tr.OneDimInputs(
                    "cylinder", 0.1, tr.Solid("x", 0.0, 1.0, 1.0), 10.0, 300.0, 400.0, time_s=1.0
                )
            ),
            "tienen que ser",
        ),
    ],
)
def test_errors_are_explained(make, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=message):
        make()


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_every_example_exports_and_pickles(system: str) -> None:
    pairs = [
        (tr.LUMPED_EXAMPLES, tr.solve_lumped, tr.lumped_to_dict),
        (tr.ONE_DIM_EXAMPLES, tr.solve_one_dimensional, tr.one_dim_to_dict),
        (tr.SEMI_INFINITE_EXAMPLES, tr.solve_semi_infinite, tr.semi_infinite_to_dict),
        (tr.CONTACT_EXAMPLES, tr.solve_contact, tr.contact_to_dict),
        (tr.PRODUCT_EXAMPLES, tr.solve_product, tr.product_to_dict),
    ]
    for examples, solve, export in pairs:
        for ex in examples.values():
            r = solve(ex.inputs)
            json.dumps(export(r, system))  # type: ignore[arg-type]
            assert pickle.loads(pickle.dumps(r)) == r
            assert r.notes
