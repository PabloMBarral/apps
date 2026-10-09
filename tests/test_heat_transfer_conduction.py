"""Tests de core.heat_transfer.conduction (red de resistencias, Fase 8.1).

Los ejemplos del cap. 3 de Cengel y Ghajar se reproducen con los mismos
datos; el perfil, contra la solución analítica de cada geometría.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from core.heat_transfer.conduction import (
    CONDUCTION_EXAMPLE_NOTES,
    CONDUCTION_EXAMPLES,
    Boundary,
    ConductionInputs,
    Layer,
    ParallelPart,
    conduction_notes,
    conduction_to_dict,
    default_insulation_radii,
    insulation_sweep,
    solve_conduction,
    temperature_profile,
)

C = 273.15
NAMES = list(CONDUCTION_EXAMPLES)


def _example(prefix: str) -> ConductionInputs:
    return CONDUCTION_EXAMPLES[next(n for n in NAMES if n.startswith(prefix))]


# ---------------------------------------------------------------------
# Ejemplos del libro
# ---------------------------------------------------------------------


def test_brick_wall() -> None:
    """Pared de 3 × 5 m, 30 cm, k = 0,9, caras a 16 y 2 °C: 630 W."""
    r = solve_conduction(_example("Pared de ladrillo ("))
    assert r.Q_W == pytest.approx(630.0, rel=1e-12)


def test_single_pane_window() -> None:
    """Cengel y Ghajar: 266 W y la cara interior del vidrio a −2,2 °C."""
    r = solve_conduction(_example("Ventana de un vidrio"))
    assert r.Q_W == pytest.approx(266.0, abs=0.3)
    assert r.surface_temperatures_K[0] - C == pytest.approx(-2.2, abs=0.05)


def test_double_pane_window() -> None:
    """Cengel y Ghajar: 69,2 W y la cara interior a 14,2 °C; el aire es casi toda la resistencia."""
    r = solve_conduction(_example("Ventana de doble vidrio"))
    assert r.Q_W == pytest.approx(69.2, abs=0.06)
    assert r.surface_temperatures_K[0] - C == pytest.approx(14.2, abs=0.05)
    air = next(x for x in r.resistances if "aire" in x.label)
    assert r.share(air) == pytest.approx(0.74, abs=0.005)


def test_composite_wall_with_parallel_bricks() -> None:
    """Cengel y Ghajar: R = 6,87 K/W para una sección de 0,25 m²; 4,37 W; 262 W en 15 m²."""
    inputs = _example("Pared de ladrillos con revoque")
    r = solve_conduction(inputs)
    assert r.Q_W == pytest.approx(262.0, abs=0.5)
    assert r.R_total_K_per_W * 15.0 / 0.25 == pytest.approx(6.87, abs=0.005)
    layer = next(x for x in r.resistances if x.parts)
    assert sum(p.Q_W for p in layer.parts) == pytest.approx(r.Q_W, rel=1e-12)
    # 1/R = Σ 1/R_i
    assert 1.0 / layer.R_K_per_W == pytest.approx(sum(1.0 / p.R_K_per_W for p in layer.parts))
    brick = next(p for p in layer.parts if p.name == "ladrillo")
    assert brick.Q_W / r.Q_W > 0.9  # el ladrillo conduce casi todo


def test_insulated_steam_pipe() -> None:
    """Cengel y Ghajar: 121 W por metro; 0,02 °C en el caño y 284 °C en la aislación."""
    r = solve_conduction(_example("Caño de vapor aislado"))
    assert r.Q_per_length_W_per_m == pytest.approx(121.0, abs=0.3)
    pipe, insulation = (x for x in r.resistances if x.kind == "conducción")
    assert pipe.dT_K == pytest.approx(0.02, abs=0.005)
    assert insulation.dT_K == pytest.approx(284.0, abs=0.5)


def test_insulated_wire_and_critical_radius() -> None:
    """Cengel y Ghajar: 80 W, la interfaz a 105 °C y r_cr = k/h = 12,5 mm."""
    inputs = _example("Alambre eléctrico")
    r = solve_conduction(inputs)
    assert r.Q_W == 80.0
    assert r.T_side1_K - C == pytest.approx(105.0, abs=0.05)
    assert r.critical_radius_m == pytest.approx(0.0125)
    # duplicar el plástico (r₂ = 5,5 mm < r_cr) baja la temperatura de la interfaz
    thicker = replace(inputs, layers=(replace(inputs.layers[0], thickness_m=0.004),))
    assert solve_conduction(thicker).T_side1_K < r.T_side1_K
    # el mínimo del barrido cae en r_cr
    points = insulation_sweep(inputs, [0.0015 + 1e-4 * j for j in range(1, 400)])
    best = min(points, key=lambda p: p.T_side1_K)
    assert best.r_outer_m == pytest.approx(0.0125, abs=1.5e-4)


def test_heat_loss_peaks_at_the_critical_radius() -> None:
    """Con las temperaturas dadas, Q̇ tiene un máximo en r_cr (Cengel y Ghajar, §3-7)."""
    inputs = ConductionInputs(
        geometry="cylinder",
        layers=(Layer("plástico", 0.002, 0.15),),
        inner=Boundary("surface", T_K=90.0 + C),
        outer=Boundary("fluid", T_K=30.0 + C, h_W_per_m2K=12.0),
        length_m=1.0,
        r_inner_m=0.0015,
    )
    points = insulation_sweep(inputs, default_insulation_radii(inputs, n=400))
    best = max(points, key=lambda p: p.Q_W)
    assert best.r_outer_m == pytest.approx(0.0125, rel=0.03)
    # esfera: r_cr = 2k/h
    sphere = replace(inputs, geometry="sphere")
    points = insulation_sweep(sphere, default_insulation_radii(sphere, n=400))
    best = max(points, key=lambda p: p.Q_W)
    assert best.r_outer_m == pytest.approx(0.025, rel=0.03)
    assert solve_conduction(sphere).critical_radius_m == pytest.approx(0.025)


def test_contact_resistance_equivalent_thickness() -> None:
    """Cengel y Ghajar: el contacto de h_c = 11 000 equivale a 2,15 cm de aluminio."""
    r = solve_conduction(_example("Dos placas de aluminio"))
    contact = next(x for x in r.resistances if x.kind == "contacto")
    assert contact.R_K_per_W == pytest.approx(1.0 / 11_000.0)
    notes = " ".join(conduction_notes(r))
    assert "21,5 mm de aluminio" in notes


def test_spherical_tank_heat_flows_inward() -> None:
    """Con el agua helada adentro el calor entra: Q̇ < 0 (del lado 2 al lado 1)."""
    inputs = _example("Tanque esférico")
    r = solve_conduction(inputs)
    r1, r2 = 1.5, 1.52
    R = (
        1.0 / (80.0 * 4.0 * math.pi * r1**2)
        + (r2 - r1) / (4.0 * math.pi * 15.0 * r1 * r2)
        + 1.0 / (15.34 * 4.0 * math.pi * r2**2)
    )
    assert r.Q_W == pytest.approx(-22.0 / R, rel=1e-12)
    assert any("del lado 2 al lado 1" in n for n in conduction_notes(r))


# ---------------------------------------------------------------------
# Propiedades de la red
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_example_is_a_consistent_network(name: str) -> None:
    inputs = CONDUCTION_EXAMPLES[name]
    r = solve_conduction(inputs)
    for a, b in zip(r.resistances, r.resistances[1:], strict=False):
        assert a.T_to_K == pytest.approx(b.T_from_K, rel=1e-12)
    for x in r.resistances:
        assert x.dT_K == pytest.approx(r.Q_W * x.R_K_per_W, rel=1e-9, abs=1e-12)
    assert sum(r.share(x) for x in r.resistances) == pytest.approx(1.0)
    dT = r.T_side1_K - r.T_side2_K
    assert r.U_inner_W_per_m2K * r.A_inner_m2 * dT == pytest.approx(r.Q_W, rel=1e-9)
    assert r.U_outer_W_per_m2K * r.A_outer_m2 * dT == pytest.approx(r.Q_W, rel=1e-9)
    assert name in CONDUCTION_EXAMPLE_NOTES
    data = conduction_to_dict(r, "Técnico")
    assert data["Q"].endswith("W")
    assert len(data["resistencias"]) == len(r.resistances)


def _analytic_T(geometry: str, r_: float, T1: float, T2: float, x1: float, x2: float) -> float:
    """La solución analítica dentro de una capa (Incropera, ec. 3.3, 3.26 y 3.36)."""
    if geometry == "plane":
        return T1 + (T2 - T1) * (r_ - x1) / (x2 - x1)
    if geometry == "cylinder":
        return T1 + (T2 - T1) * math.log(r_ / x1) / math.log(x2 / x1)
    return T1 + (T2 - T1) * (1 / x1 - 1 / r_) / (1 / x1 - 1 / x2)


def _area_at(inputs: ConductionInputs, r_: float) -> float:
    if inputs.geometry == "plane":
        return inputs.area_m2
    if inputs.geometry == "cylinder":
        return 2 * math.pi * r_ * inputs.length_m
    return 4 * math.pi * r_ * r_


@pytest.mark.parametrize("geometry", ["plane", "cylinder", "sphere"])
def test_profile_matches_the_analytical_solution(geometry: str) -> None:
    inputs = ConductionInputs(
        geometry=geometry,  # type: ignore[arg-type]
        layers=(Layer("a", 0.01, 50.0), Layer("b", 0.04, 0.2)),
        inner=Boundary("fluid", T_K=400.0, h_W_per_m2K=200.0),
        outer=Boundary("fluid", T_K=300.0, h_W_per_m2K=10.0),
        area_m2=2.0,
        length_m=3.0,
        r_inner_m=0.05,
    )
    r = solve_conduction(inputs)
    segments = temperature_profile(r, n=21)
    conduction = [x for x in r.resistances if x.kind == "conducción"]
    assert len(segments) == 2
    for seg, res in zip(segments, conduction, strict=True):
        ends = (res.T_from_K, res.T_to_K, res.x_from_m, res.x_to_m)
        for x, T in zip(seg.x_m, seg.T_K, strict=True):
            assert T == pytest.approx(_analytic_T(geometry, x, *ends), rel=1e-12)
        # el flujo local en el medio de la capa es Q̇: −k·A(x)·dT/dx
        x, dx = seg.x_m[10], 1e-7
        slope = (_analytic_T(geometry, x + dx, *ends) - _analytic_T(geometry, x - dx, *ends)) / (
            2 * dx
        )
        k = inputs.layers[seg.layer].k_W_per_mK
        assert -k * _area_at(inputs, x) * slope == pytest.approx(r.Q_W, rel=1e-5)


def test_heat_given_on_the_outer_side() -> None:
    """Con el calor dado del lado 2, Q̇ (del lado 1 al 2) es −Q̇ dado."""
    inputs = ConductionInputs(
        geometry="plane",
        layers=(Layer("pared", 0.1, 1.0),),
        inner=Boundary("surface", T_K=300.0),
        outer=Boundary("heat", Q_W=50.0),
        area_m2=2.0,
    )
    r = solve_conduction(inputs)
    assert r.Q_W == -50.0
    assert r.T_side2_K == pytest.approx(300.0 + 50.0 * 0.1 / 2.0)


def test_notes() -> None:
    pipe = solve_conduction(_example("Caño de vapor aislado"))
    notes = " ".join(conduction_notes(pipe))
    assert "lana de vidrio" in notes and "casi no ofrece resistencia" in notes
    assert "supera el radio crítico" in notes
    wire = solve_conduction(_example("Alambre eléctrico"))
    assert any("menor que el radio crítico" in n for n in conduction_notes(wire))
    # con una capa exterior metálica no se habla del radio crítico
    tank = solve_conduction(_example("Tanque esférico"))
    assert not any("radio crítico" in n for n in conduction_notes(tank))


# ---------------------------------------------------------------------
# Mensajes al alumno
# ---------------------------------------------------------------------

_OK = ConductionInputs(
    geometry="plane",
    layers=(Layer("pared", 0.1, 1.0),),
    inner=Boundary("surface", T_K=300.0),
    outer=Boundary("surface", T_K=280.0),
)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"layers": ()}, "al menos una capa"),
        ({"layers": (Layer("pared", 0.0, 1.0),)}, "espesor positivo"),
        ({"layers": (Layer("pared", 0.1, 0.0),)}, "conductividad"),
        (
            {"layers": (Layer("pared", 0.1, parts=(ParallelPart("a", 1.0, 0.5),)),)},
            "suman 0,5",
        ),
        (
            {"layers": (Layer("pared", 0.1, 1.0, contact_m2K_per_W=1e-4),)},
            "última capa",
        ),
        ({"inner": Boundary("fluid", T_K=300.0, h_W_per_m2K=0.0)}, "h tiene que ser positivo"),
        (
            {"inner": Boundary("heat", Q_W=10.0), "outer": Boundary("heat", Q_W=5.0)},
            "en los dos lados",
        ),
        ({"area_m2": 0.0}, "área"),
        ({"inner": Boundary("surface", T_K=-5.0)}, "absoluta positiva"),
    ],
)
def test_messages(change: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_conduction(replace(_OK, **change))


def test_cylinder_messages() -> None:
    with pytest.raises(ValueError, match="radio interior"):
        solve_conduction(replace(_OK, geometry="cylinder", r_inner_m=0.0))
    composite = Layer("x", 0.1, parts=(ParallelPart("a", 1.0, 1.0),))
    with pytest.raises(ValueError, match="pared plana"):
        solve_conduction(replace(_OK, geometry="cylinder", layers=(composite,), r_inner_m=0.1))
    with pytest.raises(ValueError, match="cilindro o de una esfera"):
        insulation_sweep(_OK, [0.1])
