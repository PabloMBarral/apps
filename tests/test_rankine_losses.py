"""Tests del ciclo real de Rankine (pérdidas de carga y de calor) — Fase 3.1c.

Valores de referencia: Çengel & Boles, *Termodinámica*, ejemplo 10-2 (η = 36,1 %,
Ẇ_neto = 18,9 MW con ṁ = 15 kg/s; estados de la figura 10-5), y cálculos a mano
con CoolProp (IAPWS-95) siguiendo los balances del libro (§10-5).
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import pytest
from CoolProp.CoolProp import PropsSI

from core.cycles.rankine import (
    RANKINE_EXAMPLE_NOTES,
    RANKINE_EXAMPLES,
    FeedwaterHeater,
    Losses,
    PipeLoss,
    RankineInputs,
    Reheat,
    condenser_cooling_water,
    rankine_steps,
    rankine_to_dict,
    solve_rankine,
)
from core.cycles.rankine_layout import port_flows

C = 273.15
W = "Water"
CLOSED = "closed"
CENGEL_10_2 = next(k for k in RANKINE_EXAMPLES if k.startswith("Cengel 10-2"))


def _h(**kw: float) -> float:
    (k1, v1), (k2, v2) = kw.items()
    return PropsSI("H", k1, v1, k2, v2, W)


def _s(**kw: float) -> float:
    (k1, v1), (k2, v2) = kw.items()
    return PropsSI("S", k1, v1, k2, v2, W)


def _cengel_10_2() -> RankineInputs:
    return RANKINE_EXAMPLES[CENGEL_10_2]


def _ideal(**losses: float | PipeLoss) -> RankineInputs:
    return RankineInputs(15e6, 600 + C, 10e3, losses=Losses(**losses))  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Cengel 10-2
# ---------------------------------------------------------------------


class TestCengel10_2:
    def test_matches_the_book(self) -> None:
        r = solve_rankine(_cengel_10_2())
        assert r.eta_th == pytest.approx(0.361, abs=5e-4)  # libro: 36,1 %
        assert r.W_net_W == pytest.approx(18.9e6, rel=3e-3)  # libro: 18,9 MW
        assert r.w_net_J_per_kg == pytest.approx(1258.0e3, rel=1e-3)  # libro: 1258,0 kJ/kg
        assert r.w_turbine_J_per_kg == pytest.approx(1277.0e3, rel=1e-3)  # libro: 1277,0
        assert r.w_pump_J_per_kg == pytest.approx(19.0e3, rel=1e-2)  # libro: 19,0 (con v·Δp)
        assert r.q_in_J_per_kg == pytest.approx(3487.5e3, rel=1e-3)  # libro: 3487,5

    def test_states_are_the_ones_of_the_figure(self) -> None:
        r = solve_rankine(_cengel_10_2())
        p = [s.P_Pa for s in r.states]
        T = [s.T_K - C for s in r.states]
        assert p == pytest.approx([9e3, 16e6, 15.9e6, 15.2e6, 15e6, 10e3], rel=1e-9)
        assert T[0] == pytest.approx(38.0, abs=1e-6)
        assert T[2] == pytest.approx(35.0, abs=1e-6)
        assert T[3] == pytest.approx(625.0, abs=1e-6)
        assert T[4] == pytest.approx(600.0, abs=1e-6)
        assert [label for label in r.layout.labels] == [  # type: ignore[union-attr]
            "entrada a la bomba",
            "salida de la bomba",
            "entrada a la caldera",
            "salida de la caldera",
            "entrada a la turbina",
            "salida de la turbina",
        ]

    def test_states_against_coolprop(self) -> None:
        r = solve_rankine(_cengel_10_2())
        h1 = _h(P=9e3, T=38 + C)
        h2 = h1 + (_h(P=16e6, S=_s(P=9e3, T=38 + C)) - h1) / 0.85
        h3 = _h(P=15.9e6, T=35 + C)
        h4 = _h(P=15.2e6, T=625 + C)
        h5 = _h(P=15e6, T=600 + C)
        h6 = h5 - 0.87 * (h5 - _h(P=10e3, S=_s(P=15e6, T=600 + C)))
        got = [s.h_J_per_kg for s in r.states]
        assert got == pytest.approx([h1, h2, h3, h4, h5, h6], rel=1e-6)
        assert r.q_in_J_per_kg == pytest.approx(h4 - h3, rel=1e-6)
        assert r.q_loss_J_per_kg == pytest.approx((h2 - h3) + (h4 - h5), rel=1e-5)

    def test_note_explains_the_figure(self) -> None:
        note = RANKINE_EXAMPLE_NOTES[CENGEL_10_2]
        assert "36,1 %" in note and "35 °C" in note and "38 °C" in note

    def test_procedure_follows_the_book(self) -> None:
        r = solve_rankine(_cengel_10_2())
        steps = rankine_steps(r, "Técnico")
        titles = [s.title for s in steps]
        assert titles[:6] == [
            "Estado 1: salida del condensador",
            "1 → 2: bomba",
            "2 → 3: cañería de alimentación",
            "Estado 5: entrada a la turbina",
            "4 → 5: cañería de vapor",
            "5 → 6: turbina",
        ]
        pump = steps[1]
        assert "cañería de alimentación, la caldera y la cañería de vapor" in pump.text
        assert pump.latex[0].startswith(r"\begin{aligned}p_2 &= p_5 + \sum \Delta p")
        latex = " ".join(t for s in steps for t in s.latex)
        assert r"w_{\mathrm{neto}} &= q_H - q_C - q_{\text{pérd}}" in latex
        assert r"\dot{Q}_{\text{pérd}}" in latex


# ---------------------------------------------------------------------
# Cada pérdida, por separado
# ---------------------------------------------------------------------


def test_without_losses_nothing_changes() -> None:
    plain = RankineInputs(15e6, 600 + C, 10e3)
    assert plain.losses == Losses() and not plain.losses.any
    explicit = RankineInputs(15e6, 600 + C, 10e3, losses=Losses())
    assert solve_rankine(explicit).eta_th == solve_rankine(plain).eta_th


def test_boiler_pressure_drop_costs_pump_work() -> None:
    base = solve_rankine(_ideal())
    lossy = solve_rankine(_ideal(dp_boiler_Pa=1e6))
    assert lossy.states[1].P_Pa == pytest.approx(16e6, rel=1e-9)  # la bomba compensa
    extra = lossy.w_pump_J_per_kg - base.w_pump_J_per_kg
    assert extra == pytest.approx(lossy.states[0].v_m3_per_kg * 1e6, rel=2e-2)  # ≈ v·Δp
    assert lossy.eta_th < base.eta_th
    assert lossy.w_turbine_J_per_kg == pytest.approx(base.w_turbine_J_per_kg, rel=1e-9)


def test_subcooling_lowers_the_condensate_temperature() -> None:
    r = solve_rankine(_ideal(subcooling_K=5.0))
    T_sat = PropsSI("T", "P", 10e3, "Q", 0, W)
    assert r.states[0].T_K == pytest.approx(T_sat - 5.0, abs=1e-6)
    assert r.states[0].x is None  # líquido comprimido
    assert r.T_low_K == pytest.approx(T_sat, abs=1e-6)  # la condensación sigue a T_sat
    assert r.q_in_J_per_kg > solve_rankine(_ideal()).q_in_J_per_kg


def test_condenser_pressure_drop() -> None:
    r = solve_rankine(_ideal(dp_condenser_Pa=2e3))
    assert r.states[0].P_Pa == pytest.approx(8e3, rel=1e-9)
    assert r.states[0].x == pytest.approx(0.0, abs=1e-9)
    assert r.T_low_K == pytest.approx(PropsSI("T", "P", 10e3, "Q", 0, W), abs=1e-6)


def test_adiabatic_pipe_keeps_the_enthalpy() -> None:
    r = solve_rankine(_ideal(steam_pipe=PipeLoss(3e5, 0.0), feed_pipe=PipeLoss(2e5, 0.0)))
    lay = r.layout
    assert lay is not None
    assert r.states[lay.boiler_out].h_J_per_kg == pytest.approx(
        r.states[lay.turbine_inlet].h_J_per_kg, rel=1e-9
    )
    assert r.states[lay.boiler_in].h_J_per_kg == pytest.approx(r.states[1].h_J_per_kg, rel=1e-9)
    assert r.q_loss_J_per_kg == pytest.approx(0.0, abs=1e-3)


def test_saturated_inlet_with_a_lossy_steam_pipe() -> None:
    r = solve_rankine(RankineInputs(8e6, None, 10e3, losses=Losses(steam_pipe=PipeLoss(2e5, 15.0))))
    lay = r.layout
    assert lay is not None
    T_sat = PropsSI("T", "P", 8e6, "Q", 1, W)
    assert r.states[lay.turbine_inlet].x == pytest.approx(1.0)
    assert r.states[lay.boiler_out].T_K == pytest.approx(T_sat + 15.0, abs=1e-6)
    assert r.states[lay.boiler_out].x is None  # la caldera entrega vapor sobrecalentado


def test_reheater_pressure_drop() -> None:
    inputs = RankineInputs(
        15e6, 600 + C, 10e3, reheat=Reheat(4e6, 600 + C), losses=Losses(dp_reheater_Pa=3e5)
    )
    r = solve_rankine(inputs)
    lay = r.layout
    assert lay is not None
    assert r.states[lay.reheat_out].P_Pa == pytest.approx(3.7e6, rel=1e-9)
    assert r.states[lay.reheat_out].T_K == pytest.approx(600 + C, abs=1e-6)
    lossless = solve_rankine(replace(inputs, losses=Losses()))
    assert r.eta_th < lossless.eta_th


def test_closed_heater_water_side_pressure_drop() -> None:
    heaters = (FeedwaterHeater(2e5, CLOSED, dp_Pa=0.5e5), FeedwaterHeater(20e5, CLOSED, dp_Pa=1e5))
    r = solve_rankine(RankineInputs(15e6, 600 + C, 10e3, heaters=heaters))
    low, high = r.of_kind("closed_heater")
    assert r.states[1].P_Pa == pytest.approx(15e6 + 1.5e5, rel=1e-9)  # la bomba compensa todo
    for comp, dp in ((low, 0.5e5), (high, 1e5)):
        p_in = r.states[comp.port("fw_in").state].P_Pa
        assert r.states[comp.port("fw_out").state].P_Pa == pytest.approx(p_in - dp, rel=1e-9)


@pytest.mark.parametrize(
    "inputs",
    [
        RANKINE_EXAMPLES[CENGEL_10_2],
        RankineInputs(
            15e6,
            600 + C,
            10e3,
            0.85,
            0.8,
            reheat=Reheat(4e6, 600 + C),
            heaters=(
                FeedwaterHeater(0.7e5, CLOSED, dp_Pa=0.3e5),
                FeedwaterHeater(5e5),
                FeedwaterHeater(40e5, CLOSED, dp_Pa=1e5),
            ),
            drain_forward=True,
            losses=Losses(
                dp_boiler_Pa=1e6,
                dp_reheater_Pa=2e5,
                dp_condenser_Pa=1e3,
                subcooling_K=3.0,
                feed_pipe=PipeLoss(2e5, 2.0),
                steam_pipe=PipeLoss(3e5, 10.0),
            ),
        ),
    ],
)
def test_first_law_and_mass_balances_with_losses(inputs: RankineInputs) -> None:
    r = solve_rankine(inputs)
    closure = r.q_in_J_per_kg - r.q_out_J_per_kg - r.q_loss_J_per_kg
    assert closure == pytest.approx(r.w_net_J_per_kg, rel=1e-8)
    for comp in r.components:
        m_in = sum(p.fraction for p in comp.inlets)
        m_out = sum(p.fraction for p in comp.outlets)
        assert m_in == pytest.approx(m_out, rel=1e-9), comp.label
    balances = dict(r.tespy_balances)
    pipes = -sum(v for k, v in balances.items() if k.startswith("cañería"))
    assert pipes == pytest.approx(r.q_loss_J_per_kg, rel=1e-6)
    assert r.layout is not None
    flows = port_flows(r.layout)
    ys = r.extraction_fractions
    for ci, comp in enumerate(r.components):
        for pi, port in enumerate(comp.ports):
            value = sum(c * (1.0 if k is None else ys[k]) for k, c in flows[(ci, pi)].items())
            assert value == pytest.approx(port.fraction, rel=1e-7, abs=1e-9), (comp.label, port)


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("losses", "match"),
    [
        (Losses(dp_boiler_Pa=-1e5), "no puede ser negativa"),
        (Losses(feed_pipe=PipeLoss(-1e5, 0.0)), "no puede ser negativa"),
        (Losses(subcooling_K=-2.0), "no puede ser negativo"),
        (Losses(steam_pipe=PipeLoss(0.0, -5.0)), "pierde calor"),
        (Losses(dp_condenser_Pa=10e3), "punto triple"),
        (Losses(subcooling_K=46.0), "congelaría"),
        (Losses(dp_boiler_Pa=2e10), "máximo de validez"),
        (Losses(dp_reheater_Pa=1e5), "no recalienta"),
        (Losses(steam_pipe=PipeLoss(0.0, 1200.0)), "máximo de validez"),
        (Losses(feed_pipe=PipeLoss(0.0, 60.0)), "punto triple"),
    ],
)
def test_invalid_losses_are_explained(losses: Losses, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_rankine(RankineInputs(15e6, 600 + C, 10e3, losses=losses))


def test_reheater_drop_cannot_swallow_the_low_pressure_turbine() -> None:
    inputs = RankineInputs(
        15e6, 600 + C, 10e3, reheat=Reheat(1e5, 600 + C), losses=Losses(dp_reheater_Pa=0.95e5)
    )
    with pytest.raises(ValueError, match="turbina de baja arrancaría"):
        solve_rankine(inputs)


def test_extraction_inside_the_reheater_is_explained() -> None:
    inputs = RankineInputs(
        15e6,
        600 + C,
        10e3,
        reheat=Reheat(4e6, 600 + C),
        heaters=(FeedwaterHeater(3.9e6, CLOSED),),
        losses=Losses(dp_reheater_Pa=2e5),
    )
    with pytest.raises(ValueError, match="cae dentro del recalentador"):
        solve_rankine(inputs)


def test_saturated_inlet_needs_a_superheated_boiler_outlet() -> None:
    inputs = RankineInputs(8e6, None, 10e3, losses=Losses(steam_pipe=PipeLoss(5e6, 1.0)))
    with pytest.raises(ValueError, match="no es vapor"):
        solve_rankine(inputs)


# ---------------------------------------------------------------------
# Procedimiento, export y agua de enfriamiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_procedure_is_well_formed_in_every_system(system: str) -> None:
    r = solve_rankine(_cengel_10_2())
    cooling = condenser_cooling_water(r, 20 + C, 30 + C)
    for step in rankine_steps(r, system, cooling=cooling):  # type: ignore[arg-type]
        for t in step.latex:
            assert t.count("{") == t.count("}"), (step.title, t)
            assert "- -" not in t and "+ -" not in t, (step.title, t)
            assert r"\mathrm{pérd}" not in t  # KaTeX estricto: los acentos van en \text


def test_export_includes_the_losses() -> None:
    r = solve_rankine(_cengel_10_2())
    data = rankine_to_dict(r, "Técnico")
    assert data["ciclo"] == "Rankine simple, ciclo real (con pérdidas)"
    losses = data["datos"]["perdidas"]
    assert losses["dp_caldera"]["valor"] == pytest.approx(7.0)
    assert losses["cañeria_de_vapor"]["dT"]["valor"] == pytest.approx(25.0)
    assert data["resultados"]["q_perdido_en_cañerias"]["valor"] == pytest.approx(
        r.q_loss_J_per_kg / 1e3
    )
    assert "perdidas" not in rankine_to_dict(solve_rankine(_ideal()), "SI")["datos"]


def test_cooling_water_uses_the_condensation_temperature() -> None:
    r = solve_rankine(_ideal(subcooling_K=5.0, dp_condenser_Pa=1e3))
    with pytest.raises(ValueError, match="no puede salir más caliente"):
        condenser_cooling_water(r, 20 + C, r.T_low_K + 0.5)
    cooling = condenser_cooling_water(r, 20 + C, 30 + C)
    assert cooling.Q_W == pytest.approx(r.Q_out_W)


def test_cengel_10_2_solves_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve_rankine(_cengel_10_2())


def test_heat_exchangers_with_friction_are_drawn_almost_on_the_isobar() -> None:
    from core.diagrams import DiagramSpec, build_diagram, segment_between

    r = solve_rankine(_cengel_10_2())
    spec = DiagramSpec(fluid="Water", system="Técnico")
    diagram = build_diagram(spec)
    pts = [s.to_state_point() for s in r.states]
    kind, coords = segment_between(diagram, spec, pts[2], pts[3])  # caldera, 3 → 4
    assert kind == "isobaric"
    assert coords["p"][0] == pytest.approx(15.9e6) and coords["p"][-1] == pytest.approx(15.2e6)
    assert all(b > a for a, b in zip(coords["h"][:-1], coords["h"][1:], strict=True))
    assert segment_between(diagram, spec, pts[5], pts[0])[0] == "isobaric"  # condensador
    assert segment_between(diagram, spec, pts[4], pts[5])[0] == "straight"  # turbina real
