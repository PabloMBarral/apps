"""Tests de la condensación en película (Fase 11.2).

Contra Çengel y Ghajar (2015), §10-5: con las propiedades de sus enunciados las correlaciones
dan los números del libro; con las de CoolProp, dentro de 1 % (T_sat de CoolProp a 1 atm es
99,97 °C y el libro usa 100 °C).
"""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import warnings

import pytest

from core.heat_transfer import condensation as co

C = 273.15
G = 9.81

# Ejemplo de la placa vertical: el líquido a 90 °C y el vapor a 100 °C (tabla A-9).
PLATE = {"rho_l": 965.3, "mu_l": 0.315e-3, "nu_l": 0.326e-6, "cp_l": 4206.0, "k_l": 0.675}
# Ejemplo del condensador: el líquido a 35 °C y el vapor a 40 °C.
TUBE = {"rho_l": 994.0, "mu_l": 0.720e-3, "cp_l": 4178.0, "k_l": 0.623}

VERTICAL = "Placa vertical en vapor a 1 atm (Çengel y Ghajar §10-5)"
INCLINED = "La misma placa inclinada 30° (Çengel y Ghajar §10-5)"
HORIZONTAL = "Tubo horizontal del condensador de una central (Çengel y Ghajar §10-5)"
BANK = "Banco de 3 × 4 tubos (Çengel y Ghajar §10-5)"
INSIDE = "R-134a adentro de un tubo (ejemplo propio)"
SUPERHEATED = "Vapor sobrecalentado sobre un tubo vertical (ejemplo propio)"


def _example(name: str, **kw: object) -> co.CondensationResult:
    inputs = co.CONDENSATION_EXAMPLES[name].inputs
    return co.solve_condensation(dataclasses.replace(inputs, **kw))  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Las correlaciones con las propiedades del libro
# ---------------------------------------------------------------------


def test_vertical_plate_with_book_properties() -> None:
    """Placa de 2 m a 80 °C en vapor a 1 atm: h*_fg = 2314 kJ/kg, Re = 1287 (con ondas),
    h = 5848 W/(m²·K), 702 kW y 0,303 kg/s."""
    h_fg = co.h_star(2257e3, PLATE["cp_l"], 20.0)
    assert h_fg == pytest.approx(2314e3, rel=2e-4)
    Re = co.re_wavy(PLATE["k_l"], PLATE["mu_l"], PLATE["nu_l"], h_fg, 20.0, 2.0)
    assert Re == pytest.approx(1287, rel=1e-3)
    h = co.h_wavy(Re, PLATE["k_l"], PLATE["nu_l"])
    assert h == pytest.approx(5848, rel=1e-3)
    Q = h * 6.0 * 20.0
    assert Q == pytest.approx(702e3, rel=2e-3)
    assert Q / h_fg == pytest.approx(0.303, rel=2e-3)
    # inclinada 30°: 677 kW
    assert Q * math.cos(math.radians(30.0)) ** 0.25 == pytest.approx(677e3, rel=2e-3)


def test_horizontal_tube_and_bank_with_book_properties() -> None:
    """Tubo de 3 cm a 30 °C con vapor a 7,38 kPa: h = 9292 W/(m²·K), 8,76 kW y 3,6 g/s por
    metro; con 3 tubos por columna, 7060 W/(m²·K) y 79,8 kW los 12 tubos."""
    h_fg = co.h_star(2407e3, TUBE["cp_l"], 10.0)
    assert h_fg == pytest.approx(2435e3, rel=2e-4)
    h = co.h_horizontal_tube(TUBE["rho_l"], 0.05, h_fg, TUBE["k_l"], TUBE["mu_l"], 10.0, 0.03)
    assert h == pytest.approx(9292, rel=1e-3)
    Q = h * math.pi * 0.03 * 10.0
    assert Q == pytest.approx(8.76e3, rel=2e-3)
    assert Q / h_fg == pytest.approx(3.6e-3, rel=0.01)
    h_N = h * 3**-0.25
    assert h_N == pytest.approx(7060, rel=1e-3)
    assert h_N * 12 * math.pi * 0.03 * 10.0 == pytest.approx(79.8e3, rel=2e-3)


def test_wavy_regime_meets_the_laminar_one() -> None:
    """En Re = 30 la forma con ondas da casi lo mismo que la laminar de Nusselt,
    h = 1,47·k·Re^−⅓·(g/ν²)^⅓ (con ρ_v ≪ ρ_ℓ)."""
    k, nu = 0.675, 0.326e-6
    lam = 1.47 * k * 30.0 ** (-1 / 3) * (G / nu**2) ** (1 / 3)
    assert co.h_wavy(30.0, k, nu) == pytest.approx(lam, rel=5e-3)


def test_turbulent_regime_meets_the_wavy_one() -> None:
    """En Re = 1800 la turbulenta y la de ondas difieren algunos % según Pr."""
    k, nu = 0.675, 0.326e-6
    for Pr in (1.0, 2.0, 5.0):
        ratio = co.h_turbulent(1800.0, k, nu, Pr) / co.h_wavy(1800.0, k, nu)
        assert 0.95 < ratio < 1.15


def test_explicit_reynolds_forms_are_consistent() -> None:
    """Las formas explícitas de Re despejan Re = 4·L·h·ΔT/(μ·h*_fg) con su h; los coeficientes
    redondeados (3,70 por 4/1,08, 0,0690 por 4/58) dejan 0,3 % de diferencia."""
    k, mu, rho, cp, Pr = 0.675, 0.315e-3, 965.3, 4206.0, 1.96
    nu = mu / rho
    h_fg, dT = co.h_star(2257e3, cp, 20.0), 20.0
    for L in (0.5, 2.0):
        Re = co.re_wavy(k, mu, nu, h_fg, dT, L)
        h = co.h_wavy(Re, k, nu)
        assert 4 * L * h * dT / (mu * h_fg) == pytest.approx(Re, rel=3e-3)
    L = 20.0
    Re = co.re_turbulent(k, mu, nu, Pr, h_fg, dT, L)
    h = co.h_turbulent(Re, k, nu, Pr)
    assert 4 * L * h * dT / (mu * h_fg) == pytest.approx(Re, rel=3e-3)


def test_laminar_film_thickness_matches_nusselt() -> None:
    """δ = [4·μ·k·ΔT·x/(g·ρ·Δρ·h*)]^¼ y h = k/δ·(4/3)·… : el h medio de Nusselt es (4/3)·k/δ(L)
    con el coeficiente exacto 0,943 ≈ (4/3)·(1/√2)^½."""
    args = (965.3, 0.6, 2314e3, 0.675, 0.315e-3, 20.0)
    L = 0.05
    delta = co.film_thickness(*args, L)
    h = co.h_laminar_vertical(*args, L)
    assert h == pytest.approx(4.0 / 3.0 * 0.675 / delta, rel=1e-3)


def test_chato_and_sphere_coefficients() -> None:
    args = (994.0, 0.05, 2407e3, 0.623, 0.720e-3, 10.0, 0.03)
    tube = co.h_horizontal_tube(*args)
    assert co.h_sphere(*args) / tube == pytest.approx(0.815 / 0.729)
    h_c = co.h_chato(994.0, 0.05, 2407e3, 4178.0, 0.623, 0.720e-3, 10.0, 0.03)
    expected = (
        0.555
        * (
            G
            * 994.0
            * (994.0 - 0.05)
            * 0.623**3
            / (0.720e-3 * 10.0 * 0.03)
            * (2407e3 + 3 / 8 * 4178.0 * 10.0)
        )
        ** 0.25
    )
    assert h_c == pytest.approx(expected, rel=1e-12)


def test_dropwise_griffith() -> None:
    assert co.dropwise_h(10.0 + C) is None
    assert co.dropwise_h(50.0 + C) == pytest.approx(51_104 + 2044 * 50)
    assert co.dropwise_h(100.0 + C) == pytest.approx(255_504, abs=1)
    assert co.dropwise_h(150.0 + C) == 255_510.0


# ---------------------------------------------------------------------
# Los ejemplos con CoolProp
# ---------------------------------------------------------------------


def test_vertical_plate_with_coolprop() -> None:
    r = _example(VERTICAL)
    assert r.regime == "wavy"
    assert r.Re == pytest.approx(1287, rel=0.01)
    assert r.h == pytest.approx(5848, rel=0.01)
    assert r.Q_W == pytest.approx(702e3, rel=0.01)
    assert r.m_dot == pytest.approx(0.303, rel=0.01)
    assert r.liquid.T_K == pytest.approx((r.sat.T_sat_K + 80.0 + C) / 2)
    Re_balance = 4 * r.inputs.L_m * r.h * r.dT_K / (r.liquid.mu * r.h_fg_star)
    assert r.Re == pytest.approx(Re_balance, rel=3e-3)
    assert r.delta_m is not None and 0.1e-3 < r.delta_m < 0.3e-3
    assert r.h_dropwise is not None and r.h_dropwise > 10 * r.h


def test_inclined_plate_with_coolprop() -> None:
    r, v = _example(INCLINED), _example(VERTICAL)
    assert r.Q_W == pytest.approx(677e3, rel=0.01)
    assert r.h / v.h == pytest.approx(math.cos(math.radians(30.0)) ** 0.25)
    assert any("cos θ" in n for n in r.notes)


def test_horizontal_tube_and_bank_with_coolprop() -> None:
    t, b = _example(HORIZONTAL), _example(BANK)
    assert t.h == pytest.approx(9292, rel=0.01)
    assert t.Q_W == pytest.approx(8.76e3, rel=0.01)
    assert t.m_dot == pytest.approx(3.6e-3, rel=0.01)
    assert t.Re is None and t.delta_m is None
    assert b.h_single == pytest.approx(t.h)
    assert b.h == pytest.approx(7060, rel=0.01)
    assert b.Q_W == pytest.approx(79.8e3, rel=0.01)
    assert b.tubes == 12
    assert any("N^−¼" in n for n in b.notes)


def test_inside_tube_range_of_chato() -> None:
    r = _example(INSIDE)
    assert r.Re_vapor is not None and r.Re_vapor < co.RE_VAPOR_CHATO
    assert any("dentro del rango de Chato" in n for n in r.notes)
    fast = _example(INSIDE, V_v_m_s=5.0)
    assert fast.Re_vapor is not None and fast.Re_vapor > co.RE_VAPOR_CHATO
    assert any("anular" in n for n in fast.notes)
    unknown = _example(INSIDE, V_v_m_s=None)
    assert unknown.Re_vapor is None
    assert any("Cargá la velocidad" in n for n in unknown.notes)


def test_superheat_raises_the_latent_heat() -> None:
    hot, sat = _example(SUPERHEATED), _example(SUPERHEATED, T_v_K=None)
    assert hot.superheat_K == pytest.approx(150.0 + C - hot.sat.T_sat_K)
    assert hot.vapor is not None
    extra = hot.vapor.cp * hot.superheat_K
    assert hot.h_fg_star - sat.h_fg_star == pytest.approx(extra)
    assert hot.h > sat.h  # más h*_fg, película más delgada
    assert hot.dT_K == sat.dT_K  # la fuerza impulsora sigue siendo T_sat − T_s
    assert any("sobrecalentado" in n for n in hot.notes)


def test_regimes_along_the_height() -> None:
    r = _example(VERTICAL)
    c = co.condensation_curve(r)
    assert c.variable == "L" and c.Re is not None
    assert c.regime[0] == "laminar" and c.regime[-1] == "turbulent"
    order = ["laminar", "wavy", "turbulent"]
    assert [order.index(x) for x in c.regime] == sorted(order.index(x) for x in c.regime)
    assert all(b > a for a, b in zip(c.Re, c.Re[1:], strict=False))
    for Re, regime in zip(c.Re, c.regime, strict=True):
        if regime == "laminar":
            assert Re < co.RE_LAMINAR * 1.01
        if regime == "turbulent":
            assert Re > co.RE_TURBULENT * 0.99
    # h baja con el alto mientras la película engorda y vuelve a subir en la turbulenta
    k_min = c.h.index(min(c.h))
    assert 0 < k_min < len(c.h) - 1 and c.regime[k_min + 1] == "turbulent"
    tube = co.condensation_curve(_example(HORIZONTAL))
    assert tube.variable == "D" and tube.Re is None
    assert all(b < a for a, b in zip(tube.h, tube.h[1:], strict=False))


def test_geometry_comparison() -> None:
    plate = co.geometry_comparison(_example(VERTICAL))
    assert [row.key for row in plate] == ["vertical", "inclined_30", "inclined_60"]
    assert plate[0].current and plate[0].h > plate[1].h > plate[2].h
    tube = co.geometry_comparison(_example(BANK))
    keys = [row.key for row in tube]
    assert keys == ["vertical_tube", "horizontal_tube", "tube_bank", "inside_tube"]
    by = {row.key: row for row in tube}
    assert by["tube_bank"].current
    assert by["horizontal_tube"].h > by["tube_bank"].h
    assert by["horizontal_tube"].h > by["vertical_tube"].h  # L/D = 33 > 2,77
    inclined45 = co.geometry_comparison(_example(INCLINED, angle_deg=45.0))
    assert [row.key for row in inclined45] == [
        "vertical",
        "inclined_30",
        "inclined_45",
        "inclined_60",
    ]
    sphere = co.geometry_comparison(_example(HORIZONTAL, geometry="sphere"))
    assert sphere[0].h / sphere[1].h == pytest.approx(0.815 / 0.729)


def test_horizontal_beats_vertical_above_2_77_diameters() -> None:
    """Con película laminar, h_vert/h_horiz = 1,29·(D/L)^¼: iguales con L = 2,77·D."""
    r = _example(HORIZONTAL, T_s_K=39.5 + C, L_m=2.77 * 0.03)
    rows = {row.key: row for row in co.geometry_comparison(r)}
    assert rows["vertical_tube"].regime == "laminar"
    assert rows["vertical_tube"].h / rows["horizontal_tube"].h == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------
# Errores y notas
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"T_s_K": 101.0 + C}, "no condensa"),
        ({"p_Pa": 300e5}, "crítica"),
        ({"T_s_K": -10.0 + C}, "punto triple"),
        ({"T_v_K": 90.0 + C}, "más frío"),
        ({"L_m": 0.0}, "positivos"),
        ({"geometry": "inclined_plate", "angle_deg": 90.0}, "horizontal"),
        ({"fluid": "Air"}, "lista"),
    ],
)
def test_errors(change: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _example(VERTICAL, **change)


def test_tube_errors() -> None:
    with pytest.raises(ValueError, match="diámetro"):
        _example(HORIZONTAL, D_m=0.0)
    with pytest.raises(ValueError, match="largo"):
        _example(HORIZONTAL, L_m=0.0)
    with pytest.raises(ValueError, match="al menos un tubo"):
        _example(BANK, rows=0)
    with pytest.raises(ValueError, match="negativa"):
        _example(INSIDE, V_v_m_s=-1.0)


def test_notes() -> None:
    laminar = _example(VERTICAL, L_m=0.01, T_s_K=99.0 + C)
    assert laminar.regime == "laminar"
    assert any("< 30" in n for n in laminar.notes)
    turbulent = _example(VERTICAL, L_m=10.0)
    assert turbulent.regime == "turbulent"
    assert any("> 1800" in n for n in turbulent.notes)
    near = co.solve_condensation(
        co.CondensationInputs("R134a", 38e5, "horizontal_tube", 80.0 + C, D_m=0.02)
    )
    assert any("punto crítico" in n for n in near.notes)
    tube = _example(SUPERHEATED)
    assert any("D/δ" in n for n in tube.notes)
    for r in (laminar, turbulent, near, tube):
        assert all("e+" not in n and "e-" not in n for n in r.notes)


# ---------------------------------------------------------------------
# Ejemplos, export y pickle
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(co.CONDENSATION_EXAMPLES))
def test_examples_export_and_pickle(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = _example(name)
        co.geometry_comparison(r)
        co.condensation_curve(r, points=20)
    assert pickle.loads(pickle.dumps(r)) == r
    for system in ("SI", "Técnico", "Inglés"):
        data = co.condensation_to_dict(r, system)  # type: ignore[arg-type]
        text = json.dumps(data, ensure_ascii=False)
        assert "NaN" not in text and "Infinity" not in text
    assert co.condensation_to_dict(r, "SI")["Q"]["valor"] == pytest.approx(r.Q_W)


@pytest.mark.parametrize("geometry", list(co.GEOMETRIES))
@pytest.mark.parametrize("fluid", ["Water", "R134a", "Ammonia", "Nitrogen", "n-Pentane"])
def test_every_geometry_and_fluid(geometry: co.Geometry, fluid: str) -> None:
    from core.heat_transfer.phase_change import pressure_limits, saturation_state

    p = 0.2 * pressure_limits(fluid)[1]
    T_sat = saturation_state(fluid, p).T_sat_K
    r = co.solve_condensation(
        co.CondensationInputs(
            fluid,
            p,
            geometry,
            T_sat - 5.0,
            L_m=0.5,
            W_m=1.0,
            D_m=0.02,
            angle_deg=20.0,
            rows=2,
            columns=2,
            V_v_m_s=0.2,
        )
    )
    assert r.h > 0 and r.Q_W > 0 and r.m_dot > 0
    co.geometry_comparison(r)
