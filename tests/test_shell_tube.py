"""Tests del casco y tubos con Kern (Fase 12.1)."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from core.heat_transfer.design_common import FluidProperties
from core.heat_transfer.shell_tube import (
    SHELL_TUBE_EXAMPLES,
    baffle_sweep,
    estimate_tube_count,
    kern_equivalent_diameter,
    kern_friction,
    shell_tube_notes,
    shell_tube_to_dict,
    solve_shell_tube,
    tube_inner_diameter,
)

C = 273.15
PETERS = SHELL_TUBE_EXAMPLES["Agua por el casco (Peters, Timmerhaus y West, ejemplo 14-6)"].inputs
KEROSENE = SHELL_TUBE_EXAMPLES["Enfriador de kerosene con agua de río"].inputs
WATER = SHELL_TUBE_EXAMPLES[
    "Intercambiador agua–agua de 1 paso por el casco y 2 por los tubos"
].inputs


# ---------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------


def test_bwg_inner_diameters() -> None:
    # 3/4" BWG 16: 0,620" por dentro; 1" BWG 14: 0,834" (TEMA)
    assert tube_inner_diameter(0.01905, 16) == pytest.approx(0.620 * 0.0254, abs=1e-6)
    assert tube_inner_diameter(0.0254, 14) == pytest.approx(0.834 * 0.0254, abs=1e-6)
    with pytest.raises(ValueError, match="BWG"):
        tube_inner_diameter(0.0254, 30)


def test_equivalent_diameters() -> None:
    # Kern (fig. 28): 3/4" en cuadrado de 1" → 0,95"; en triangular de 1" → 0,73" (la
    # fórmula da 0,720")
    De_sq = kern_equivalent_diameter(0.01905, 0.0254, "square")
    De_tri = kern_equivalent_diameter(0.01905, 0.0254, "triangular")
    assert De_sq / 0.0254 == pytest.approx(0.95, abs=0.005)
    assert De_tri / 0.0254 == pytest.approx(0.73, abs=0.015)


def test_tube_count_estimate() -> None:
    # N_t = 0,785·(CTP/CL)·D_s²/P_T² (Kakaç y Liu), hacia abajo a un múltiplo de los pasos
    n = estimate_tube_count(0.584, 0.0254, "square", 2)
    raw = int(0.785 * 0.90 / 1.0 * 0.584**2 / 0.0254**2)
    assert raw % 2 == 1 and n == raw - 1
    for passes in (1, 2, 4, 6, 8):
        assert estimate_tube_count(0.584, 0.0254, "triangular", passes) % passes == 0
    assert estimate_tube_count(0.584, 0.0254, "square", 1) > n
    assert estimate_tube_count(0.584, 0.0254, "triangular", 2) > n
    assert estimate_tube_count(0.584, 0.0254, "square", 4) < n


def test_kern_friction_fit() -> None:
    assert kern_friction(1.0e4) == pytest.approx(math.exp(0.576 - 0.19 * math.log(1.0e4)))
    assert kern_friction(1.0e4) == pytest.approx(0.3092, abs=1e-4)
    assert kern_friction(1.0e3) > kern_friction(1.0e5)


# ---------------------------------------------------------------------
# Kern a mano y contra Peters, Timmerhaus y West
# ---------------------------------------------------------------------


def test_kern_by_hand_peters_14_6() -> None:
    r = solve_shell_tube(PETERS)
    s = r.shell
    # a mano, con el agua a 30 °C y la pared a 40 °C del libro
    a_s = 0.584 * (0.0254 - 0.019) * 0.1524 / 0.0254
    G = 11.0 / a_s
    De = 4 * (0.0254**2 - math.pi * 0.019**2 / 4) / (math.pi * 0.019)
    Re = G * De / 0.000803
    Pr = 0.000803 * 4178.0 / 0.615
    phi = (0.000803 / 0.000657) ** 0.14
    h = 0.36 * 0.615 / De * Re**0.55 * Pr ** (1 / 3) * phi
    f = math.exp(0.576 - 0.19 * math.log(Re))
    dp = f * G**2 * 0.584 * 23 / (2 * 995.0 * De * phi)
    assert s.a_s_m2 == pytest.approx(a_s, rel=1e-12)
    assert s.G_s == pytest.approx(G, rel=1e-12)
    assert s.Re == pytest.approx(Re, rel=1e-12)
    assert s.phi == pytest.approx(phi, rel=1e-12)
    assert s.h_W_per_m2K == pytest.approx(h, rel=1e-12)
    assert s.dp_Pa == pytest.approx(dp, rel=1e-12)
    # el libro lee j_H y f de los gráficos de Kern: 3369 W/(m²·K) y 16,42 kPa
    assert s.h_W_per_m2K == pytest.approx(3369.0, rel=0.05)
    assert s.dp_Pa == pytest.approx(16_420.0, rel=0.15)


# ---------------------------------------------------------------------
# Balances y el criterio de Kern
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SHELL_TUBE_EXAMPLES))
def test_balances_close(name: str) -> None:
    r = solve_shell_tube(SHELL_TUBE_EXAMPLES[name].inputs)
    ex, i = r.exchanger, r.inputs
    Q_hot = i.hot.m_dot_kg_s * ex.hot.cp_J_per_kgK * (i.hot.T_in_K - ex.hot.T_out_K)
    Q_cold = i.cold.m_dot_kg_s * ex.cold.cp_J_per_kgK * (ex.cold.T_out_K - i.cold.T_in_K)
    assert Q_hot == pytest.approx(r.Q_W, rel=1e-9)
    assert Q_cold == pytest.approx(r.Q_W, rel=1e-9)
    if i.problem == "rating":
        assert r.U_D_W_per_m2K * r.area_m2 * ex.F * ex.dT_lm_K == pytest.approx(r.Q_W, rel=1e-6)
    # el U limpio es el sucio sin el ensuciamiento: 1/U_D = 1/U_C + R_d pedido
    assert 1 / r.U_D_W_per_m2K == pytest.approx(1 / r.U_C_W_per_m2K + r.R_d_required, rel=1e-9)


def test_kern_dirt_factor_criterion() -> None:
    r = solve_shell_tube(KEROSENE)
    ex = r.exchanger
    U_req = r.U_required_W_per_m2K
    assert U_req == pytest.approx(r.Q_W / (r.area_m2 * ex.F * ex.dT_lm_K))
    assert r.R_d_available == pytest.approx(1 / U_req - 1 / r.U_C_W_per_m2K)
    assert r.R_d_available >= r.R_d_required
    assert r.R_d_required == pytest.approx(0.0009 + 0.0002 * 0.01905 / KEROSENE.d_i_m)
    assert r.overdesign == pytest.approx(r.area_m2 / r.area_required_m2 - 1)
    assert r.length_required_m == pytest.approx(r.area_required_m2 / (r.N_t * math.pi * 0.01905))
    assert any("El equipo sirve" in n for n in shell_tube_notes(r))
    # más corto: limpio alcanza pero no tolera el ensuciamiento pedido
    short = solve_shell_tube(replace(KEROSENE, L_m=5.0, baffles=32))
    assert 0 < short.R_d_available < short.R_d_required
    assert any("cuando se ensucie" in n for n in shell_tube_notes(short))
    # mucho más corto: ni limpio
    tiny = solve_shell_tube(replace(KEROSENE, L_m=2.0, baffles=12))
    assert tiny.R_d_available < 0
    assert any("Ni limpio alcanza" in n for n in shell_tube_notes(tiny))


def test_rating_after_check_gives_the_target() -> None:
    r = solve_shell_tube(KEROSENE)
    L_req = r.length_required_m
    rated = solve_shell_tube(replace(KEROSENE, problem="rating", L_m=L_req))
    assert rated.T_hot_out_K == pytest.approx(50.0 + C, abs=1e-4)


def test_uneven_tubes_per_pass_note() -> None:
    r = solve_shell_tube(replace(WATER, tube_count=301, tube_passes=2))
    assert any("no se reparten igual" in n for n in shell_tube_notes(r))
    r = solve_shell_tube(replace(WATER, tube_count=300, tube_passes=2))
    assert not any("no se reparten igual" in n for n in shell_tube_notes(r))


def test_one_tube_pass_is_counterflow() -> None:
    r = solve_shell_tube(PETERS)
    assert r.exchanger.arrangement.kind == "counter"
    assert r.exchanger.F == pytest.approx(1.0)
    r2 = solve_shell_tube(WATER)
    assert r2.exchanger.arrangement.kind == "shell" and r2.exchanger.F < 1.0


def test_tube_side_pressure_drop() -> None:
    r = solve_shell_tube(WATER)
    t, p = r.tube, r.tube_props
    dyn = p.rho_kg_per_m3 * t.velocity_m_s**2 / 2
    assert r.dp_tube_Pa == pytest.approx((t.f * 6.0 * 2 / WATER.d_i_m + 4 * 2) * dyn)
    assert r.dp_tube_returns_Pa == pytest.approx(8 * dyn)
    # la velocidad por paso
    per_pass = r.N_t / 2
    assert t.velocity_m_s == pytest.approx(
        40.0 / (p.rho_kg_per_m3 * per_pass * math.pi * WATER.d_i_m**2 / 4)
    )


def test_baffle_sweep_tradeoff() -> None:
    points = baffle_sweep(WATER)
    assert len(points) >= 10
    hs = [p.h_o_W_per_m2K for p in points]
    dps = [p.dp_shell_Pa for p in points]
    # B crece: h_o baja y Δp baja mucho más
    assert hs == sorted(hs, reverse=True)
    assert dps == sorted(dps, reverse=True)
    assert dps[0] / dps[-1] > 3 * hs[0] / hs[-1]


# ---------------------------------------------------------------------
# Avisos y errores
# ---------------------------------------------------------------------


def test_warnings() -> None:
    viscous = FluidProperties(850.0, 2000.0, 0.05, 0.13, name="aceite pesado")
    r = solve_shell_tube(replace(KEROSENE, hot=replace(KEROSENE.hot, properties=viscous)))
    assert any("fuera del rango de Kern" in w for w in r.warnings)
    r = solve_shell_tube(replace(WATER, B_m=0.05, baffles=119))
    assert any("separación de los bafles" in w for w in r.warnings)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"P_T_m": 0.0254}, "los tubos se tocan"),
        ({"tube_count": 5000}, "no entran"),
        ({"tube_count": 1}, "más pasos"),
        ({"tube_passes": 3}, "1, 2, 4, 6 u 8"),
        ({"baffles": 0}, "al menos un bafle"),
        ({"d_i_m": 0.03}, "menor que el exterior"),
        ({"layout": "hexagonal"}, "desconocido"),
    ],
)
def test_student_errors(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_shell_tube(replace(WATER, **change))


def test_hot_must_be_hotter() -> None:
    with pytest.raises(ValueError, match="más caliente"):
        solve_shell_tube(replace(WATER, cold=replace(WATER.cold, T_in_K=70.0 + C)))


# ---------------------------------------------------------------------
# Export y ejemplos
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export_is_json(system: str) -> None:
    data = shell_tube_to_dict(solve_shell_tube(KEROSENE), system)  # type: ignore[arg-type]
    json.dumps(data)
    assert "R_d_disponible" in data and "U_requerido" in data
    assert (
        data["casco"]["G_s"]["unidad"]
        == {"SI": "kg/(m²·s)", "Técnico": "kg/(m²·s)", "Inglés": "lb/(s·ft²)"}[system]
    )
    assert "kerosene" in data["por_el_casco"]


@pytest.mark.parametrize("name", list(SHELL_TUBE_EXAMPLES))
def test_examples_have_no_warnings(name: str) -> None:
    ex = SHELL_TUBE_EXAMPLES[name]
    r = solve_shell_tube(ex.inputs)
    assert r.warnings == ()
    assert ex.description
