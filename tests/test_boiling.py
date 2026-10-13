"""Tests de la ebullición en pileta (Fase 11.2).

Contra Çengel y Ghajar (2015), §10-2: con las propiedades de sus enunciados (tablas A-9 y A-16)
las correlaciones dan los números del libro; con las de CoolProp, dentro de 3 %.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import warnings

import pytest

from core.heat_transfer import boiling as bo
from core.heat_transfer.phase_change import saturation_state

C = 273.15
ATM = 101_325.0

# Agua saturada a 100 °C, como la usan los ejemplos de Çengel y Ghajar (tabla A-9).
BOOK = {
    "mu_l": 0.282e-3,
    "h_fg": 2257e3,
    "rho_l": 957.9,
    "rho_v": 0.6,
    "sigma": 0.0589,
    "cp_l": 4217.0,
    "Pr_l": 1.75,
}


def _example(name: str) -> bo.PoolBoilingResult:
    return bo.solve_pool_boiling(bo.BOILING_EXAMPLES[name].inputs)


PAN = "Olla de acero inoxidable pulido (Çengel y Ghajar §10-2)"
NICKEL = "Flujo crítico de un calentador niquelado (Çengel y Ghajar §10-2)"
FILM = "Resistencia en ebullición en película (Çengel y Ghajar §10-2)"
BURNOUT = "Una resistencia que se quema (ejemplo propio)"


# ---------------------------------------------------------------------
# Las correlaciones con las propiedades del libro
# ---------------------------------------------------------------------


def test_rohsenow_reproduces_the_pan_example() -> None:
    """La olla: ΔT = 8 K y C_sf = 0,0130 → 7,21·10⁴ W/m², 5,10 kW y 2,26 g/s."""
    q = bo.q_rohsenow(**BOOK, C_sf=0.0130, n=1.0, dT=8.0)
    assert q == pytest.approx(7.21e4, rel=2e-3)
    Q = q * math.pi * 0.30**2 / 4.0
    assert Q == pytest.approx(5.10e3, rel=3e-3)
    assert Q / BOOK["h_fg"] == pytest.approx(2.26e-3, rel=3e-3)


def test_critical_flux_reproduces_the_nickel_example() -> None:
    """El calentador de 1 cm: L* = 2,0 (cilindro grande, C_cr = 0,12), q̇_máx = 1,017 MW/m² y
    T_s = 109 °C con C_sf = 0,0060."""
    L_star = bo.length_ratio(0.005, BOOK["rho_l"], BOOK["rho_v"], BOOK["sigma"])
    assert L_star == pytest.approx(2.0, abs=0.01)
    C_cr, case = bo.c_critical("cylinder", L_star)
    assert (C_cr, case) == (0.12, "cylinder_large")
    q_max = bo.q_critical(C_cr, BOOK["h_fg"], BOOK["rho_l"], BOOK["rho_v"], BOOK["sigma"])
    assert q_max == pytest.approx(1.017e6, rel=1e-3)
    dT = bo.dT_rohsenow(**BOOK, C_sf=0.0060, n=1.0, q=q_max)
    assert 100.0 + dT == pytest.approx(109.0, abs=0.1)


def test_film_boiling_reproduces_the_heating_element_example() -> None:
    """La resistencia de 5 mm a 350 °C con ε = 0,05: 922,6 W por metro, con el vapor a 225 °C
    (ρ_v = 0,444, c_pv = 1951, μ_v = 1,749·10⁻⁵ y k_v = 0,0351)."""
    dT = 250.0
    q_f = bo.q_film(0.62, 0.0351, 0.444, BOOK["rho_l"], BOOK["h_fg"], 1951.0, 1.749e-5, 0.005, dT)
    q_r = bo.q_radiation(0.05, 350.0 + C, 100.0 + C)
    assert q_r == pytest.approx(372.6, rel=2e-3)
    assert (q_f + 0.75 * q_r) * math.pi * 0.005 == pytest.approx(922.6, rel=1e-3)


def test_rohsenow_round_trip() -> None:
    for C_sf, n in ((0.013, 1.0), (0.0154, 1.7), (0.0027, 1.7)):
        for dT in (0.5, 5.0, 30.0):
            q = bo.q_rohsenow(**BOOK, C_sf=C_sf, n=n, dT=dT)
            assert bo.dT_rohsenow(**BOOK, C_sf=C_sf, n=n, q=q) == pytest.approx(dT, rel=1e-12)


@pytest.mark.parametrize(
    ("heater", "L_star", "C_cr", "case"),
    [
        ("plate", 50.0, 0.149, "plate"),
        ("cylinder", 1.2, 0.12, "cylinder_large"),
        ("cylinder", 0.5, 0.12 * 0.5**-0.25, "cylinder_small"),
        ("sphere", 10.0, 0.11, "sphere_large"),
        ("sphere", 1.0, 0.227, "sphere_small"),
    ],
)
def test_lienhard_dhir_table(heater: bo.Heater, L_star: float, C_cr: float, case: str) -> None:
    assert bo.c_critical(heater, L_star) == (pytest.approx(C_cr), case)


def test_sphere_constants_meet_at_the_limit() -> None:
    """0,227·L*^−½ llega a 0,11 en L* = 4,26 (en el cilindro la tabla salta de 0,1146 a 0,12)."""
    assert 0.227 * 4.26**-0.5 == pytest.approx(0.11, abs=1e-3)


def test_minimum_flux_formula() -> None:
    q = bo.q_minimum(BOOK["h_fg"], BOOK["rho_l"], BOOK["rho_v"], BOOK["sigma"])
    expected = 0.09 * 0.6 * 2257e3 * (0.0589 * 9.81 * (957.9 - 0.6) / (957.9 + 0.6) ** 2) ** 0.25
    assert q == pytest.approx(expected, rel=1e-12)
    assert q == pytest.approx(1.91e4, rel=5e-3)


def test_film_radiation_combined() -> None:
    """La forma completa de Bromley cumple su ecuación y, con poca radiación, da q̇_película +
    ¾·q̇_rad."""
    for q_f, q_r in ((1e5, 1e3), (1e5, 1e5), (2e5, 1e6)):
        q = bo.film_radiation_combined(q_f, q_r)
        assert q ** (4 / 3) == pytest.approx(q_f ** (4 / 3) + q_r * q ** (1 / 3), rel=1e-9)
    assert bo.film_radiation_combined(1e5, 1e3) == pytest.approx(1e5 + 750.0, rel=1e-5)
    assert bo.film_radiation_combined(1e5, 0.0) == 1e5


def test_surfaces_for() -> None:
    assert len(bo.surfaces_for("Water")) == 8
    assert set(bo.surfaces_for("n-Pentane")) == {"pentane_copper_polished", "pentane_copper_lapped"}
    assert bo.surfaces_for("R134a") == {}
    assert all(s.n == 1.0 for s in bo.surfaces_for("Water").values())
    assert all(s.n == 1.7 for k, s in bo.SURFACES.items() if s.fluid != "Water")


# ---------------------------------------------------------------------
# Los ejemplos con CoolProp
# ---------------------------------------------------------------------


def test_pan_example_with_coolprop() -> None:
    r = _example(PAN)
    assert r.regime == "nucleate"
    assert r.T_s_K == pytest.approx(108.0 + C)
    assert r.q == pytest.approx(7.21e4, rel=0.03)
    assert r.Q_W == pytest.approx(5.10e3, rel=0.03)
    assert r.m_dot == pytest.approx(2.26e-3, rel=0.03)
    assert r.area_m2 == pytest.approx(math.pi * 0.3**2 / 4)
    assert r.h == pytest.approx(r.q / r.dT_K)
    assert r.critical_case == "plate"
    assert r.fraction_of_critical == pytest.approx(r.q / r.q_max)


def test_nickel_example_with_coolprop() -> None:
    r = _example(NICKEL)
    assert r.regime == "nucleate"
    assert r.critical_case == "cylinder_large"
    assert r.q == r.q_max == pytest.approx(1.017e6, rel=0.03)
    assert r.T_s_K - C == pytest.approx(109.0, abs=0.5)
    assert r.dT_K == pytest.approx(r.dT_max)
    assert any("burnout" in n for n in r.notes)


def test_film_example_with_coolprop() -> None:
    r = _example(FILM)
    assert r.regime == "film"
    assert r.film is not None
    assert r.film.vapor.T_K == pytest.approx((350.0 + r.sat.T_sat_K - C) / 2 + C, rel=1e-4)
    per_meter = r.q * math.pi * 0.005
    assert per_meter == pytest.approx(922.6, rel=0.03)
    assert r.Q_W == pytest.approx(per_meter)  # L = 1 m
    assert r.q == pytest.approx(r.film.q_film + 0.75 * r.film.q_rad)


def test_book_properties_vs_coolprop() -> None:
    """Las propiedades de CoolProp a 1 atm contra la tabla A-9 (a 100 °C)."""
    sat = saturation_state("Water", ATM)
    assert sat.T_sat_K - C == pytest.approx(100.0, abs=0.03)
    assert sat.h_fg == pytest.approx(BOOK["h_fg"], rel=1e-3)
    assert sat.rho_l == pytest.approx(BOOK["rho_l"], rel=1e-3)
    assert sat.rho_v == pytest.approx(BOOK["rho_v"], rel=0.01)
    assert sat.sigma == pytest.approx(BOOK["sigma"], rel=2e-3)
    assert sat.mu_l == pytest.approx(BOOK["mu_l"], rel=0.01)
    assert sat.Pr_l == pytest.approx(BOOK["Pr_l"], rel=0.02)


# ---------------------------------------------------------------------
# La curva y los regímenes
# ---------------------------------------------------------------------


def _pan(**kw: object) -> bo.PoolBoilingResult:
    return bo.solve_pool_boiling(
        dataclasses.replace(bo.BOILING_EXAMPLES[PAN].inputs, **kw)  # type: ignore[arg-type]
    )


def test_regimes_along_the_curve() -> None:
    T_sat = _pan().sat.T_sat_K
    assert _pan(T_s_K=T_sat + 1.0).regime == "natural"
    assert _pan(T_s_K=T_sat + 8.0).regime == "nucleate"
    assert _pan(T_s_K=T_sat + 40.0).regime == "transition"
    assert _pan(T_s_K=T_sat + 300.0).regime == "film"


def test_curve_is_continuous_and_shaped_like_nukiyama() -> None:
    r = _pan()
    c = bo.boiling_curve(r)
    assert len(c.dT) >= 240 and c.dT[0] == pytest.approx(bo.DT_MIN_CURVE)
    assert r.dT_max in c.dT and r.dT_min in c.dT and r.dT_onset in c.dT
    assert c.regime[0] == "natural" and c.regime[-1] == "film"
    order = ["natural", "nucleate", "transition", "film"]
    assert [order.index(x) for x in c.regime] == sorted(order.index(x) for x in c.regime)
    for k in range(1, len(c.q)):
        if "transition" in (c.regime[k], c.regime[k - 1]):
            assert c.q[k] <= c.q[k - 1] * (1 + 1e-12)
        else:
            assert c.q[k] >= c.q[k - 1] * (1 - 1e-12)
    # continua en el flujo crítico y en el mínimo
    assert bo.q_rohsenow(
        r.sat.mu_l,
        r.sat.h_fg,
        r.sat.rho_l,
        r.sat.rho_v,
        r.sat.sigma,
        r.sat.cp_l,
        r.sat.Pr_l,
        0.0130,
        1.0,
        r.dT_max,
    ) == pytest.approx(r.q_max, rel=1e-12)
    film = _pan(T_s_K=r.sat.T_sat_K + r.dT_min)
    assert film.regime == "film" and film.q == pytest.approx(r.q_min, rel=1e-6)
    assert c.critical == (r.dT_max, r.q_max)
    assert c.leidenfrost == (r.dT_min, r.q_min)
    assert r.dT_onset < 8.0 < r.dT_max < r.dT_min
    assert c.onset[1] == pytest.approx(_pan(T_s_K=r.sat.T_sat_K + r.dT_onset).natural.q, rel=1e-6)


def test_flux_mode_is_the_inverse_of_the_temperature_mode() -> None:
    r = _pan()
    back = _pan(mode="flux", T_s_K=None, q_W_per_m2=r.q)
    assert back.T_s_K == pytest.approx(r.T_s_K, abs=1e-6)
    assert back.regime == "nucleate"
    low = _pan(mode="flux", T_s_K=None, q_W_per_m2=1.0e3)
    assert low.regime == "natural"
    assert low.q == pytest.approx(1.0e3)
    assert low.natural.q == pytest.approx(1.0e3, rel=1e-6)


def test_critical_mode_is_the_top_of_the_curve() -> None:
    r = _pan(mode="critical", T_s_K=None)
    assert r.q == r.q_max
    assert r.dT_K == r.dT_max


def test_burnout_jumps_to_the_film_branch() -> None:
    r = _example(BURNOUT)
    assert r.burnout and r.regime == "film"
    assert r.film is not None
    assert r.q == pytest.approx(1.1e6) == pytest.approx(r.film.q_total, rel=1e-6)
    assert r.q > r.q_max
    assert r.T_s_K > bo.MELTING_POINTS_K["el níquel"]
    assert any("burnout" in n and "níquel" in n for n in r.notes)
    assert any("forma completa" in n for n in r.notes)  # la radiación manda


def test_burnout_beyond_the_range_is_an_error() -> None:
    inputs = dataclasses.replace(bo.BOILING_EXAMPLES[BURNOUT].inputs, emissivity=0.0)
    with pytest.raises(ValueError, match="se quemaría"):
        bo.solve_pool_boiling(inputs)


def test_minimum_cases() -> None:
    thin = bo.solve_pool_boiling(
        bo.PoolBoilingInputs("Ethanol", 1.23e5, "cylinder", 0.001, 0.013, 1.0, "critical")
    )
    assert thin.minimum_case == "no_transition"
    assert thin.dT_min == thin.dT_max
    assert any("no tiene ebullición de transición" in n for n in thin.notes)
    beyond = bo.solve_pool_boiling(
        bo.PoolBoilingInputs("R134a", 0.3 * 40.59e5, "cylinder", 0.02, 0.013, 1.7, "critical")
    )
    assert beyond.minimum_case == "beyond_range"
    assert any("Leidenfrost" in n and "ecuación de estado" in n for n in beyond.notes)
    c = bo.boiling_curve(beyond)
    assert "film" not in c.regime
    assert c.q[-1] == pytest.approx(beyond.q_min, rel=1e-6)
    assert _pan().minimum_case == "crossing"


# ---------------------------------------------------------------------
# Errores y notas
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"T_s_K": 99.0 + C}, "no hierve"),
        ({"p_Pa": 300e5}, "crítica"),
        ({"p_Pa": 100.0}, "punto triple"),
        ({"C_sf": 0.0}, "C_sf"),
        ({"emissivity": 1.5}, "emisividad"),
        ({"D_m": 0.0}, "diámetro"),
        ({"fluid": "Air"}, "lista"),
        ({"T_s_K": 3000.0 + C}, "demasiado"),
        ({"mode": "flux", "q_W_per_m2": -1.0}, "positivo"),
    ],
)
def test_errors(change: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _pan(**change)


def test_notes() -> None:
    r134a = bo.solve_pool_boiling(
        bo.PoolBoilingInputs("R134a", 5e5, "plate", 0.2, 0.013, 1.7, "temperature", T_s_K=300.0)
    )
    assert any("No hay C_sf medido" in n for n in r134a.notes)
    small = _pan(D_m=0.02)
    assert small.L_star < 27 and any("L* =" in n and "< 27" in n for n in small.notes)
    nat = _pan(T_s_K=_pan().sat.T_sat_K + 1.0)
    assert any("convección natural" in n for n in nat.notes)
    trans = _pan(T_s_K=_pan().sat.T_sat_K + 40.0)
    assert any("interpola en log–log" in n for n in trans.notes)
    film = _pan(T_s_K=_pan().sat.T_sat_K + 300.0)
    assert any("Berenson" in n for n in film.notes)
    for r in (r134a, small, nat, trans, film):
        assert all("e+" not in n and "e-" not in n for n in r.notes)


# ---------------------------------------------------------------------
# Ejemplos, export y pickle
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(bo.BOILING_EXAMPLES))
def test_examples_export_and_pickle(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = _example(name)
        bo.boiling_curve(r)
    assert pickle.loads(pickle.dumps(r)) == r
    for system in ("SI", "Técnico", "Inglés"):
        data = bo.pool_boiling_to_dict(r, system)  # type: ignore[arg-type]
        text = json.dumps(data, ensure_ascii=False)
        assert "NaN" not in text and "Infinity" not in text
        assert data["regimen"] == bo.REGIMES[r.regime]
    assert bo.pool_boiling_to_dict(r, "SI")["q"]["valor"] == pytest.approx(r.q)
