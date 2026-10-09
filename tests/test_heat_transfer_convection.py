"""Tests de core.heat_transfer.convection (correlaciones de convección, Fase 8.1).

Cada correlación se contrasta con un ejemplo publicado (con las propiedades del
libro) o con su límite físico, y con otra correlación independiente en el rango
donde las dos valen. Los ejemplos completos usan CoolProp: el aire tiene k ~2,7 %
mayor y Pr ~3 % menor que la tabla A-15 de Cengel y Ghajar, así que contra el
libro se pide ±3 %.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import replace

import pytest

from core.export import dict_to_csv
from core.heat_transfer import convection as cv
from core.heat_transfer.convection import (
    CORRELATIONS,
    EXTERNAL_EXAMPLES,
    INTERNAL_EXAMPLES,
    NATURAL_EXAMPLES,
    ExternalFlowInputs,
    InternalFlowInputs,
    NaturalConvectionInputs,
    convection_notes,
    convection_to_dict,
    fluid_properties,
    internal_notes,
    internal_to_dict,
    nu_curve,
    solve_external,
    solve_internal,
    solve_natural,
)

C = 273.15
SYSTEMS = ("SI", "Técnico", "Inglés")


def _solve(inputs: ExternalFlowInputs | NaturalConvectionInputs | InternalFlowInputs):  # noqa: ANN202
    if isinstance(inputs, ExternalFlowInputs):
        return solve_external(inputs)
    if isinstance(inputs, NaturalConvectionInputs):
        return solve_natural(inputs)
    return solve_internal(inputs)


ALL_EXAMPLES = {**EXTERNAL_EXAMPLES, **NATURAL_EXAMPLES, **INTERNAL_EXAMPLES}
STEAM_PIPE = next(e for k, e in EXTERNAL_EXAMPLES.items() if k.startswith("Caño de vapor"))
NATURAL_PIPE = next(e for k, e in NATURAL_EXAMPLES.items() if k.startswith("Caño horizontal"))
HEATER = next(e for k, e in INTERNAL_EXAMPLES.items() if k.startswith("Agua calentada con"))


# ---------------------------------------------------------------------
# Correlaciones contra ejemplos publicados (con las propiedades del libro)
# ---------------------------------------------------------------------


def test_cylinder_incropera_example_7_4() -> None:
    """Incropera, ejemplo 7.4: Re = 6071, Pr = 0,700 → Hilpert 37,3 y Churchill–Bernstein 40,6."""
    assert cv.nu_hilpert(6071.0, 0.700) == pytest.approx(37.3, abs=0.05)
    assert cv.nu_churchill_bernstein(6071.0, 0.700) == pytest.approx(40.6, abs=0.05)


def test_cylinder_cengel_steam_pipe_with_table_properties() -> None:
    """Cengel y Ghajar: aire a T_f = 60 °C (ν = 1,896·10⁻⁵, Pr = 0,7202) → Nu = 124."""
    Re = 8.0 * 0.10 / 1.896e-5
    assert Re == pytest.approx(4.219e4, rel=1e-3)
    assert round(cv.nu_churchill_bernstein(Re, 0.7202)) == 124


def test_vertical_plate_incropera_example_9_2() -> None:
    """Incropera, ejemplo 9.2 (pantalla de una chimenea): Ra = 1,813·10⁹ y Pr = 0,690 → 147."""
    assert round(cv.nu_vertical_churchill_chu(1.813e9, 0.690)) == 147


def test_horizontal_cylinder_cengel_hot_water_pipe() -> None:
    """Cengel y Ghajar: Ra = 1,867·10⁶ y Pr = 0,7241 → Nu = 17,40."""
    assert cv.nu_horizontal_cylinder_churchill_chu(1.867e6, 0.7241) == pytest.approx(
        17.40, abs=0.02
    )


def test_dittus_boelter_cengel_resistance_heater() -> None:
    """Cengel y Ghajar: agua a 40 °C, Re = 10 750 y Pr = 4,32 → Nu = 69,4."""
    assert cv.nu_dittus_boelter(10_750.0, 4.32, heating=True) == pytest.approx(69.4, abs=0.05)


def test_laminar_plate_cengel_engine_oil() -> None:
    """Cengel y Ghajar: aceite a 40 °C sobre una placa de 5 m (Re = 4,024·10⁴, Pr = 2962) → 1913."""
    Re = 2.0 * 5.0 / 2.485e-4
    assert cv.nu_plate_laminar(Re, 2962.0) == pytest.approx(1913.0, abs=1.0)


# ---------------------------------------------------------------------
# Límites físicos y coherencia entre correlaciones
# ---------------------------------------------------------------------


def test_physical_limits() -> None:
    # sin flujo, una esfera conduce al infinito: Nu = 2
    assert cv.nu_whitaker(0.0, 0.7, 1.0) == pytest.approx(2.0)
    assert cv.nu_ranz_marshall(0.0, 0.7) == pytest.approx(2.0)
    assert cv.nu_sphere_churchill(0.0, 0.7) == pytest.approx(2.0)
    # tubo largo: la entrada térmica no pesa
    assert cv.nu_hausen(1e5, 1.0, 1e-12) == pytest.approx(3.66, rel=1e-6)
    assert cv.nu_tube_laminar(True) == 3.66
    assert cv.nu_tube_laminar(False) == pytest.approx(4.364, abs=1e-3)
    # la placa vertical sin empuje: el término constante de Churchill y Chu
    assert cv.nu_vertical_churchill_chu(0.0, 0.7) == pytest.approx(0.825**2)
    # la constante 871 empalma la mixta con la laminar en Re_cr = 5·10⁵
    for Pr in (0.7, 7.0):
        lam = cv.nu_plate_laminar(cv.RE_CRITICAL_PLATE, Pr)
        assert cv.nu_plate_mixed(cv.RE_CRITICAL_PLATE, Pr) == pytest.approx(lam, rel=1e-3)
    # Dittus y Boelter: Pr^0,4 si se calienta y Pr^0,3 si se enfría
    ratio = cv.nu_dittus_boelter(1e4, 5.0, True) / cv.nu_dittus_boelter(1e4, 5.0, False)
    assert ratio == pytest.approx(5.0**0.1)
    # Sieder y Tate con la misma viscosidad en la pared
    assert cv.nu_sieder_tate(1e4, 1.0, 1.0) == pytest.approx(0.027 * 1e4**0.8)


@pytest.mark.parametrize(
    ("Re", "row"),
    [(0.5, 0), (3.9, 0), (4.0, 1), (39.0, 1), (100.0, 2), (5000.0, 3), (1e5, 4), (1e6, 4)],
)
def test_hilpert_picks_the_row_of_the_table(Re: float, row: int) -> None:
    _lo, _hi, C_, m = cv._HILPERT[row]
    assert cv.nu_hilpert(Re, 1.0) == pytest.approx(C_ * Re**m)


@pytest.mark.parametrize(
    ("Ra", "row"), [(1e-5, 0), (1e-2, 0), (1.0, 1), (1e3, 2), (1e6, 3), (1e9, 4), (1e13, 4)]
)
def test_morgan_picks_the_row_of_the_table(Ra: float, row: int) -> None:
    _hi, C_, n = cv._MORGAN[row]
    assert cv.nu_morgan(Ra) == pytest.approx(C_ * Ra**n)


def test_independent_correlations_agree_where_both_hold() -> None:
    """Dos correlaciones de autores distintos, dentro de su incertidumbre típica."""
    for Re in (1e2, 1e3, 1e4, 4e4):
        cb, hil = cv.nu_churchill_bernstein(Re, 0.7), cv.nu_hilpert(Re, 0.7)
        assert hil == pytest.approx(cb, rel=0.06), Re
    for Re in (1e4, 3e4, 1e5):
        for Pr in (0.7, 3.0, 10.0):
            gn, db = cv.nu_gnielinski(Re, Pr), cv.nu_dittus_boelter(Re, Pr, True)
            assert db == pytest.approx(gn, rel=0.20), (Re, Pr)
    for Ra in (1e4, 1e6, 1e8, 1e10):
        cc, mc = cv.nu_vertical_churchill_chu(Ra, 0.71), cv.nu_vertical_mcadams(Ra)
        assert mc == pytest.approx(cc, rel=0.15), Ra
    for Ra in (1e5, 1e7, 1e9):
        cc, mo = cv.nu_horizontal_cylinder_churchill_chu(Ra, 0.71), cv.nu_morgan(Ra)
        assert mo == pytest.approx(cc, rel=0.10), Ra


def _colebrook_smooth(Re: float) -> float:
    f = 0.02
    for _ in range(100):
        f = (-2.0 * math.log10(2.51 / (Re * math.sqrt(f)))) ** -2
    return f


@pytest.mark.parametrize("Re", [1e4, 1e5, 1e6, 5e6])
def test_petukhov_matches_colebrook_for_a_smooth_tube(Re: float) -> None:
    assert cv.friction_petukhov(Re) == pytest.approx(_colebrook_smooth(Re), rel=0.02)


def test_correlation_registry() -> None:
    assert set(cv.TUBE_CORRELATIONS) <= set(CORRELATIONS)
    d = {
        "Re": 6e5,  # con Re < 5·10⁵ la mixta de la placa da Nu < 0
        "Pr": 0.9,
        "Ra": 1e7,
        "mu_ratio": 1.1,
        "D_over_L": 0.01,
        "heating": 1.0,
        "constant_T": 1.0,
    }
    for key, c in CORRELATIONS.items():
        assert c.key == key
        assert c.name and c.validity
        assert c.latex.count("{") == c.latex.count("}"), key
        assert c.func(d) > 0.0, key
        assert isinstance(c.in_range(d), bool), key


def test_ranges_tolerate_one_percent() -> None:
    """El Pr del aire de CoolProp (0,708 a 20 °C) entra en el 0,71 de Whitaker."""
    d = {"Re": 1e4, "Pr": 0.708, "mu_ratio": 1.0}
    assert CORRELATIONS["whitaker"].in_range(d)
    assert not CORRELATIONS["whitaker"].in_range({**d, "Pr": 0.69})


# ---------------------------------------------------------------------
# Propiedades
# ---------------------------------------------------------------------


def test_fluid_properties() -> None:
    air = fluid_properties("Air", 300.0, 101_325.0)
    assert air.rho_kg_per_m3 == pytest.approx(1.177, rel=2e-3)
    assert air.Pr == pytest.approx(0.707, rel=5e-3)
    assert air.beta_per_K * 300.0 == pytest.approx(1.0, rel=0.01)  # gas ideal: β = 1/T
    assert not air.is_liquid
    water = fluid_properties("Water", 300.0, 101_325.0)
    # Incropera, tabla A.6 a 300 K: β = 276,1·10⁻⁶ 1/K y Pr = 5,83
    assert water.beta_per_K == pytest.approx(276.1e-6, rel=0.01)
    assert water.Pr == pytest.approx(5.83, rel=0.01)
    assert water.is_liquid
    assert water.nu_m2_per_s == pytest.approx(water.mu_Pa_s / water.rho_kg_per_m3)
    assert water.alpha_m2_per_s == pytest.approx(
        water.k_W_per_mK / (water.rho_kg_per_m3 * water.cp_J_per_kgK)
    )


def test_fluid_properties_messages() -> None:
    with pytest.raises(ValueError, match="no tiene la viscosidad o la conductividad del R-1233zd"):
        fluid_properties("R1233zd(E)", 300.0, 101_325.0)
    with pytest.raises(ValueError, match="punto triple"):
        fluid_properties("Water", 250.0, 101_325.0)


# ---------------------------------------------------------------------
# Convección forzada externa
# ---------------------------------------------------------------------


def test_cengel_steam_pipe_in_wind() -> None:
    r = solve_external(STEAM_PIPE.inputs)
    assert r.correlation == "churchill_bernstein"
    assert r.Re == pytest.approx(4.219e4, rel=5e-3)
    assert r.Nu == pytest.approx(124.0, rel=0.015)
    assert r.Q_W == pytest.approx(1093.0, rel=0.03)  # W por metro de caño
    assert r.Q_W == pytest.approx(r.h_W_per_m2K * math.pi * 0.10 * 1.0 * 100.0)
    assert r.props.T_K == pytest.approx(60.0 + C)  # a la temperatura de película
    assert r.warnings == ()


def test_plate_regimes() -> None:
    base = ExternalFlowInputs("Air", "plate", 10.0, 60.0 + C, 20.0 + C, 1.0, 1.0)
    mixed = solve_external(base)
    assert mixed.Re > cv.RE_CRITICAL_PLATE
    assert mixed.correlation == "plate_mixed"
    laminar = solve_external(replace(base, V_m_per_s=2.0))
    assert laminar.correlation == "plate_laminar"
    # la mixta con Re chico da Nu < 0: no se muestra
    assert "plate_mixed" not in {a.key for a in laminar.alternatives}
    tripped = solve_external(replace(base, tripped=True))
    assert tripped.correlation == "plate_turbulent"
    assert tripped.h_W_per_m2K > 2.0 * mixed.h_W_per_m2K
    notes = " ".join(convection_notes(mixed))
    assert "x_cr" in notes and "veces más" in notes
    assert "Sin el alambre" in " ".join(convection_notes(tripped))
    assert mixed.area_m2 == pytest.approx(1.0)
    # la curva Nu(Re) de cada régimen no cruza Re_cr (la mixta caería a Nu < 0)
    _, xs, nus = nu_curve(mixed)
    assert min(xs) == pytest.approx(cv.RE_CRITICAL_PLATE) and min(nus) > 0.0
    assert mixed.Re in xs
    _, xs, _ = nu_curve(laminar)
    assert max(xs) == pytest.approx(cv.RE_CRITICAL_PLATE) and laminar.Re in xs


def test_sphere_uses_free_stream_properties() -> None:
    i = ExternalFlowInputs("Air", "sphere", 15.0, 75.0 + C, 20.0 + C, 0.025)
    r = solve_external(i)
    assert r.correlation == "whitaker"
    assert r.props.T_K == pytest.approx(i.T_inf_K)
    assert r.mu_s_Pa_s == pytest.approx(fluid_properties("Air", i.T_s_K, i.p_Pa).mu_Pa_s)
    expected = cv.nu_whitaker(r.Re, r.Pr, r.props.mu_Pa_s / r.mu_s_Pa_s)
    assert r.Nu == pytest.approx(expected)
    assert r.area_m2 == pytest.approx(math.pi * 0.025**2)
    keys = [a.key for a in r.alternatives]
    assert keys == ["whitaker", "ranz_marshall"]
    assert r.warnings == ()  # el Pr de CoolProp entra con la tolerancia


def test_cylinder_regime_by_reynolds() -> None:
    base = ExternalFlowInputs("Air", "cylinder", 8.0, 110.0 + C, 10.0 + C, 0.10)
    assert "laminar" in solve_external(base).regime
    assert "turbulenta" in solve_external(replace(base, V_m_per_s=40.0, length_m=0.2)).regime


def test_cold_surface_gains_heat() -> None:
    hot = solve_external(ExternalFlowInputs("Air", "cylinder", 5.0, 60.0 + C, 20.0 + C, 0.05))
    cold = solve_external(ExternalFlowInputs("Air", "cylinder", 5.0, 0.0 + C, 20.0 + C, 0.05))
    assert hot.Q_W > 0.0 > cold.Q_W
    assert cold.q_W_per_m2 == pytest.approx(cold.h_W_per_m2K * (-20.0))


def test_liquid_note_and_wall_boiling_warning() -> None:
    water = solve_external(ExternalFlowInputs("Water", "cylinder", 0.5, 40.0 + C, 20.0 + C, 0.02))
    assert any("líquido" in n for n in convection_notes(water))
    boiling = solve_external(
        ExternalFlowInputs("Water", "cylinder", 0.5, 120.0 + C, 20.0 + C, 0.02)
    )
    assert any("ebullición" in w for w in boiling.warnings)
    condensing = solve_external(
        ExternalFlowInputs("Water", "cylinder", 5.0, 80.0 + C, 150.0 + C, 0.02)
    )
    assert any("condensar" in w for w in condensing.warnings)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"V_m_per_s": 0.0}, "velocidad"),
        ({"length_m": 0.0}, "El diámetro tiene que ser positivo"),
        ({"width_m": -1.0}, "El ancho"),
        ({"T_s_K": -5.0}, "absolutas positivas"),
        ({"p_Pa": 0.0}, "presión"),
        ({"geometry": "cube"}, "Geometría desconocida"),
    ],
)
def test_external_messages(change: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_external(replace(STEAM_PIPE.inputs, **change))


# ---------------------------------------------------------------------
# Convección natural
# ---------------------------------------------------------------------


def test_cengel_hot_water_pipe() -> None:
    r = solve_natural(NATURAL_PIPE.inputs)
    assert r.correlation == "horizontal_cylinder_churchill_chu"
    assert r.Ra == pytest.approx(1.867e6, rel=0.03)
    assert r.Nu == pytest.approx(17.40, rel=0.02)
    assert r.Q_W == pytest.approx(443.0, rel=0.03)
    assert r.Ra == pytest.approx(r.Gr * r.Pr)
    g, p = cv.G_STANDARD, r.props
    Gr = g * p.beta_per_K * 50.0 * 0.08**3 / p.nu_m2_per_s**2
    assert r.Gr == pytest.approx(Gr)


@pytest.mark.parametrize(
    ("geometry", "T_s", "expected"),
    [
        ("horizontal_plate_up", 90.0, "horizontal_upper"),
        ("horizontal_plate_down", 90.0, "horizontal_lower"),
        ("horizontal_plate_up", 5.0, "horizontal_lower"),  # la de arriba de una fría
        ("horizontal_plate_down", 5.0, "horizontal_upper"),  # la de abajo de una fría
    ],
)
def test_horizontal_plates(geometry: str, T_s: float, expected: str) -> None:
    i = NaturalConvectionInputs("Air", geometry, T_s + C, 30.0 + C, 0.6, 0.4)  # type: ignore[arg-type]
    r = solve_natural(i)
    assert r.correlation == expected
    assert r.L_char_m == pytest.approx(0.6 * 0.4 / (2 * (0.6 + 0.4)))  # L = A/P
    assert r.area_m2 == pytest.approx(0.24)
    notes = " ".join(convection_notes(r))
    assert "L = A/P" in notes
    assert ("libre" in notes) == (expected == "horizontal_upper")


def test_upper_face_of_a_hot_plate_transfers_more() -> None:
    up = solve_natural(
        NaturalConvectionInputs("Air", "horizontal_plate_up", 363.15, 303.15, 0.6, 0.6)
    )
    down = replace(up.inputs, geometry="horizontal_plate_down")
    assert up.h_W_per_m2K > 1.5 * solve_natural(down).h_W_per_m2K


def test_natural_regime_and_comparison() -> None:
    vertical = solve_natural(
        NaturalConvectionInputs("Water", "vertical_plate", 50.0 + C, 20.0 + C, 0.3, 0.3)
    )
    assert vertical.regime == "turbulento" and vertical.Ra > 1e9
    notes = " ".join(convection_notes(vertical))
    assert "más que la incertidumbre típica" in notes and "Churchill y Chu" in notes
    air = solve_natural(NaturalConvectionInputs("Air", "vertical_plate", 90.0 + C, 30.0 + C, 0.6))
    assert air.regime == "laminar"
    assert "es la incertidumbre típica" in " ".join(convection_notes(air))


def test_natural_messages() -> None:
    with pytest.raises(ValueError, match="no se dilata"):
        solve_natural(NaturalConvectionInputs("Water", "vertical_plate", 1.0 + C, 3.0 + C, 0.3))
    with pytest.raises(ValueError, match="misma temperatura"):
        solve_natural(NaturalConvectionInputs("Air", "vertical_plate", 300.0, 300.0, 0.3))
    with pytest.raises(ValueError, match="El largo"):
        solve_natural(NaturalConvectionInputs("Air", "sphere", 350.0, 300.0, 0.0))
    with pytest.raises(ValueError, match="Geometría desconocida"):
        solve_natural(NaturalConvectionInputs("Air", "cone", 350.0, 300.0, 0.1))  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Flujo interno
# ---------------------------------------------------------------------


def test_cengel_resistance_heater() -> None:
    """Cengel y Ghajar: Re = 10 750, Nu = 69,4, h = 1460 W/(m²·K), 65 °C y pared a 115 °C."""
    r = solve_internal(HEATER.inputs)
    assert r.correlation == "dittus_boelter"
    assert r.Re == pytest.approx(10_750.0, rel=5e-3)
    assert r.Nu == pytest.approx(69.4, rel=0.01)
    assert r.h_W_per_m2K == pytest.approx(1460.0, rel=0.01)
    assert r.T_out_K - C == pytest.approx(65.0, abs=0.1)
    assert r.T_s_out_K - C == pytest.approx(115.0, abs=1.0)
    assert r.Q_W == pytest.approx(34.6e3, rel=0.005)
    assert r.V_m_per_s == pytest.approx(0.2358, rel=5e-3)
    assert r.warnings == ()


@pytest.mark.parametrize("name", list(INTERNAL_EXAMPLES))
def test_internal_energy_balance(name: str) -> None:
    r = solve_internal(INTERNAL_EXAMPLES[name].inputs)
    i = r.inputs
    # las propiedades a la temperatura media que resulta (la iteración cerró)
    assert r.props.T_K == pytest.approx(r.T_mean_K, abs=1e-6)
    if i.condition == "constant_T":
        assert r.Q_W == pytest.approx(r.h_W_per_m2K * r.area_m2 * r.dT_lm_K, rel=1e-9)
        assert r.T_s_out_K is None
    else:
        assert r.Q_W == pytest.approx(i.q_W_per_m2 * r.area_m2, rel=1e-9)
        assert r.T_s_out_K == pytest.approx(r.T_out_K + i.q_W_per_m2 / r.h_W_per_m2K)
        assert r.dT_lm_K is None
    assert r.heating == (r.Q_W > 0.0)
    assert r.dp_Pa == pytest.approx(
        r.f * i.L_m / i.D_m * r.props.rho_kg_per_m3 * r.V_m_per_s**2 / 2.0
    )
    assert 1 <= r.iterations < 50


def test_cooling_uses_pr_to_the_0_3() -> None:
    ex = next(e for k, e in INTERNAL_EXAMPLES.items() if "enfriada" in k)
    r = solve_internal(ex.inputs)
    assert not r.heating and r.Q_W < 0.0
    assert r.Nu == pytest.approx(0.023 * r.Re**0.8 * r.props.Pr**0.3, rel=1e-12)


def test_internal_auto_selection_and_laminar_limits() -> None:
    lam = InternalFlowInputs("Water", 0.006, 3.0, 0.004, 10.0 + C, "constant_T", T_s_K=40.0 + C)
    r = solve_internal(lam)
    assert r.regime == "laminar" and r.correlation == "hausen"
    assert r.f == pytest.approx(64.0 / r.Re)
    L_h, L_t = r.entry_lengths_m
    assert L_h == pytest.approx(0.05 * r.Re * 0.006)
    assert L_t == pytest.approx(L_h * r.props.Pr)
    assert any("Hausen da" in n for n in internal_notes(r))
    # un tubo muy largo: la entrada no pesa y Hausen tiende a 3,66
    long_tube = solve_internal(replace(lam, L_m=3000.0, T_s_K=12.0 + C))
    assert long_tube.Nu == pytest.approx(3.66, rel=0.01)
    flux = solve_internal(replace(lam, condition="constant_flux", T_s_K=None, q_W_per_m2=300.0))
    assert flux.correlation == "tube_laminar" and flux.Nu == pytest.approx(48.0 / 11.0)
    turb = solve_internal(replace(lam, D_m=0.02, m_dot_kg_s=0.3, L_m=4.0))
    assert turb.regime == "turbulento" and turb.correlation == "gnielinski"
    assert {a.key for a in turb.alternatives} == {"dittus_boelter", "gnielinski", "sieder_tate"}
    transition = solve_internal(replace(lam, D_m=0.01, m_dot_kg_s=0.04, L_m=2.0))
    assert transition.regime == "de transición"
    assert any("transición" in w for w in transition.warnings)


def test_internal_warnings() -> None:
    short = InternalFlowInputs(
        "Water",
        0.05,
        0.3,
        1.0,
        20.0 + C,
        "constant_T",
        T_s_K=40.0 + C,
        correlation="dittus_boelter",
    )
    assert any("L/D < 10" in w for w in solve_internal(short).warnings)
    hausen_flux = InternalFlowInputs(
        "Water", 0.006, 3.0, 0.004, 10.0 + C, "constant_flux", q_W_per_m2=1000.0,
        correlation="hausen",
    )  # fmt: skip
    assert any("fuera de su rango" in w for w in solve_internal(hausen_flux).warnings)
    wall = InternalFlowInputs("Water", 0.025, 1.0, 0.3, 15.0 + C, "constant_T", T_s_K=120.0 + C)
    assert any("ebullición" in w for w in solve_internal(wall).warnings)


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (
            InternalFlowInputs("Water", 0.01, 20.0, 0.01, 20.0 + C, "constant_T", T_s_K=150.0 + C),
            "hierve a 100,0 °C: cambia de fase dentro del tubo.*subí la presión",
        ),
        (
            InternalFlowInputs("Water", 0.05, 20.0, 0.01, 150.0 + C, "constant_T", T_s_K=40.0 + C),
            "condensa a 100,0 °C.*bajá la presión",
        ),
        (
            InternalFlowInputs("Air", 0.01, 20.0, 1e-4, 20.0 + C, "constant_flux", q_W_per_m2=-5e4),
            "por debajo del cero absoluto",
        ),
        (
            InternalFlowInputs(
                "Water", 0.01, 20.0, 0.01, 5.0 + C, "constant_flux", q_W_per_m2=-5e4
            ),
            "Con estos datos la pared del tubo llegaría a .*punto triple",
        ),
    ],
)
def test_internal_phase_and_range_messages(inputs: InternalFlowInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_internal(inputs)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"D_m": 0.0}, "diámetro y el largo"),
        ({"L_m": -1.0}, "diámetro y el largo"),
        ({"m_dot_kg_s": 0.0}, "caudal másico"),
        ({"T_in_K": 0.0}, "temperatura de entrada"),
        ({"condition": "radiation"}, "Condición de pared desconocida"),
        ({"correlation": "colburn"}, "Correlación de tubo desconocida"),
        ({"T_s_K": None}, "Falta la temperatura de la pared"),
        ({"T_s_K": 15.0 + C}, "misma temperatura"),
        ({"condition": "constant_flux", "q_W_per_m2": None}, "Falta el flujo de calor"),
    ],
)
def test_internal_messages(change: dict, match: str) -> None:
    base = InternalFlowInputs("Water", 0.025, 5.0, 0.3, 15.0 + C, "constant_T", T_s_K=120.0 + C)
    with pytest.raises(ValueError, match=match):
        solve_internal(replace(base, **change))


# ---------------------------------------------------------------------
# Todos los ejemplos: notas, curva de Nu, export y pickle
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(ALL_EXAMPLES))
def test_every_example(name: str) -> None:
    r = _solve(ALL_EXAMPLES[name].inputs)
    assert r.warnings == ()  # los ejemplos andan sin avisos
    notes = internal_notes(r) if isinstance(r, cv.InternalFlowResult) else convection_notes(r)
    assert notes and all(isinstance(n, str) and n for n in notes)
    assert all("e+" not in n for n in notes if "Ra = " not in n)  # % legibles
    # la curva pasa por el punto de los datos
    var, xs, nus = nu_curve(r)
    x0 = r.Ra if var == "Ra" else r.Re
    assert xs == sorted(xs) and x0 in xs
    assert nus[xs.index(x0)] == pytest.approx(r.Nu, rel=1e-12)
    assert all(b >= a - 1e-12 for a, b in zip(nus, nus[1:], strict=False))
    # el export
    to_dict = internal_to_dict if isinstance(r, cv.InternalFlowResult) else convection_to_dict
    for system in SYSTEMS:
        data = to_dict(r, system)
        json.dumps(data, ensure_ascii=False)
        assert dict_to_csv(data)
        assert data["Nu"] == r.Nu
    assert pickle.loads(pickle.dumps(r)) == r


# ---------------------------------------------------------------------
# Para la página: los fluidos y el perfil del tubo
# ---------------------------------------------------------------------


def test_convection_fluids_have_transport_properties() -> None:
    fluids = cv.convection_fluids()
    assert "Water" in fluids and "Air" in fluids
    assert "R1233zd(E)" not in fluids  # CoolProp no tiene su viscosidad
    for fluid in fluids:
        fluid_properties(fluid, 350.0, 2.0e5)


@pytest.mark.parametrize("name", list(INTERNAL_EXAMPLES))
def test_tube_profile(name: str) -> None:
    r = solve_internal(INTERNAL_EXAMPLES[name].inputs)
    xs, Tm, Ts = cv.tube_profile(r)
    assert xs[0] == 0.0 and xs[-1] == pytest.approx(r.inputs.L_m)
    assert Tm[0] == pytest.approx(r.inputs.T_in_K)
    assert Tm[-1] == pytest.approx(r.T_out_K, abs=1e-9)
    if r.inputs.condition == "constant_T":
        assert Ts == [r.inputs.T_s_K] * len(xs)
    else:
        q = r.inputs.q_W_per_m2
        assert all(s - m == pytest.approx(q / r.h_W_per_m2K) for s, m in zip(Ts, Tm, strict=True))
        assert Ts[-1] == pytest.approx(r.T_s_out_K)
