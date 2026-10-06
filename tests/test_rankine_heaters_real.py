"""Tests de los calentadores cerrados reales — Fase 3.1c.

Subenfriador de drenaje (DCA), desrecalentador (TTD < 0) y drenajes bombeados
hacia adelante desde cualquier cerrado (las fracciones quedan acopladas). Las
referencias son balances a mano con CoolProp (IAPWS-95), como en Cengel §10-6.
"""

from __future__ import annotations

import itertools
import warnings

import pytest
from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve

from core.cycles.rankine import (
    RANKINE_EXAMPLE_NOTES,
    RANKINE_EXAMPLES,
    FeedwaterHeater,
    RankineInputs,
    RankineResult,
    Reheat,
    rankine_steps,
    rankine_to_dict,
    solve_rankine,
)
from core.cycles.rankine_layout import port_flows

C = 273.15
W = "Water"
OPEN, CLOSED = "open", "closed"
REAL = next(k for k in RANKINE_EXAMPLES if k.startswith("Calentadores reales"))


def _h(**kw: float) -> float:
    (k1, v1), (k2, v2) = kw.items()
    return PropsSI("H", k1, v1, k2, v2, W)


def _single_closed(**kw) -> RankineInputs:
    return RankineInputs(15e6, 600 + C, 10e3, heaters=(FeedwaterHeater(1.2e6, CLOSED, **kw),))


def _balances_close(r: RankineResult) -> None:
    assert r.q_in_J_per_kg - r.q_out_J_per_kg - r.q_loss_J_per_kg == pytest.approx(
        r.w_net_J_per_kg, rel=1e-8
    )
    for comp in r.components:
        m_in = sum(p.fraction for p in comp.inlets)
        m_out = sum(p.fraction for p in comp.outlets)
        assert m_in == pytest.approx(m_out, rel=1e-9), comp.label
        if comp.kind in ("open_heater", "closed_heater", "mixer", "valve"):  # adiabáticos
            e_in = sum(p.fraction * r.states[p.state].h_J_per_kg for p in comp.inlets)
            e_out = sum(p.fraction * r.states[p.state].h_J_per_kg for p in comp.outlets)
            assert e_in == pytest.approx(e_out, rel=1e-8), comp.label
    assert r.layout is not None
    flows = port_flows(r.layout)
    ys = r.extraction_fractions
    for ci, comp in enumerate(r.components):
        for pi, port in enumerate(comp.ports):
            value = sum(c * (1.0 if k is None else ys[k]) for k, c in flows[(ci, pi)].items())
            assert value == pytest.approx(port.fraction, rel=1e-7, abs=1e-9), (comp.label, port)


# ---------------------------------------------------------------------
# Subenfriador de drenaje
# ---------------------------------------------------------------------


def test_drain_cooler_leaves_the_drain_dca_above_the_water_inlet() -> None:
    r = solve_rankine(_single_closed(dca_K=5.6))
    (heater,) = r.of_kind("closed_heater")
    T_in = r.states[heater.port("fw_in").state].T_K
    drain = r.states[heater.port("drain_out").state]
    assert drain.T_K == pytest.approx(T_in + 5.6, abs=1e-6)
    assert drain.h_J_per_kg == pytest.approx(_h(P=1.2e6, T=T_in + 5.6), rel=1e-9)
    assert drain.x is None  # líquido comprimido (subenfriado)


def test_drain_cooler_needs_less_extraction_by_hand() -> None:
    plain = solve_rankine(_single_closed())
    cooled = solve_rankine(_single_closed(dca_K=5.6))
    (heater,) = cooled.of_kind("closed_heater")
    h = [s.h_J_per_kg for s in cooled.states]
    fw_in, fw_out = heater.port("fw_in").state, heater.port("fw_out").state
    bleed, drain = heater.port("bleed").state, heater.port("drain_out").state
    # Balance de Cengel §10-6 con el drenaje subenfriado: y (h_ext − h_dren) = h_sal − h_ent.
    y = (h[fw_out] - h[fw_in]) / (h[bleed] - h[drain])
    assert cooled.extraction_fractions[0] == pytest.approx(y, rel=1e-6)
    assert cooled.extraction_fractions[0] < plain.extraction_fractions[0]
    assert cooled.eta_th > plain.eta_th  # el drenaje devuelve menos calor al condensador


def test_drain_cooler_with_nothing_to_cool_is_explained() -> None:
    with pytest.raises(ValueError, match="nada que enfriar"):
        solve_rankine(_single_closed(dca_K=150.0))


# ---------------------------------------------------------------------
# Desrecalentador
# ---------------------------------------------------------------------


def test_desuperheater_lets_the_water_leave_above_saturation() -> None:
    inputs = RankineInputs(
        15e6,
        600 + C,
        10e3,
        0.87,
        0.85,
        heaters=(FeedwaterHeater(40e5, CLOSED, -2.0, desuperheater=True),),
    )
    r = solve_rankine(inputs)
    (heater,) = r.of_kind("closed_heater")
    T_sat = PropsSI("T", "P", 40e5, "Q", 0, W)
    out = r.states[heater.port("fw_out").state]
    assert out.T_K == pytest.approx(T_sat + 2.0, abs=1e-6)
    h = [s.h_J_per_kg for s in r.states]
    fw_in, bleed = heater.port("fw_in").state, heater.port("bleed").state
    drain = heater.port("drain_out").state
    y = (out.h_J_per_kg - h[fw_in]) / (h[bleed] - h[drain])  # balance de todo el calentador
    assert r.extraction_fractions[0] == pytest.approx(y, rel=1e-6)
    assert dict(r.tespy_heaters)["calentador cerrado"] == pytest.approx(
        out.h_J_per_kg - h[fw_in], rel=1e-6
    )
    # La zona de condensación deja el agua por debajo de T_sat (el procedimiento lo muestra).
    step = next(s for s in rankine_steps(r, "Técnico") if "fracción de extracción" in s.title)
    T_z = float(step.latex[-1].rsplit("&=", 1)[1].split(r"\ ")[0].strip())
    assert T_z < T_sat - C


@pytest.mark.parametrize(
    ("heater", "match"),
    [
        (FeedwaterHeater(1.2e6, CLOSED, -1.0, desuperheater=True), "vapor sobrecalentado"),
        (FeedwaterHeater(40e5, CLOSED, -200.0, desuperheater=True), "más caliente que la"),
        (FeedwaterHeater(40e5, CLOSED, -1.0), "no puede ser negativa"),
        (FeedwaterHeater(40e5, OPEN, desuperheater=True), "son de los calentadores cerrados"),
        (FeedwaterHeater(40e5, OPEN, dca_K=5.0), "son de los calentadores cerrados"),
        (FeedwaterHeater(40e5, OPEN, drain_forward=True), "son de los calentadores cerrados"),
        (FeedwaterHeater(40e5, CLOSED, dca_K=-1.0), "DCA"),
        (FeedwaterHeater(40e5, CLOSED, dp_Pa=-1e5), "no puede ser negativa"),
    ],
)
def test_invalid_real_heaters_are_explained(heater: FeedwaterHeater, match: str) -> None:
    # 1,2 MPa con η_T = 0,87 desde 15 MPa y 600 °C sale húmedo: no hay qué desrecalentar.
    inputs = RankineInputs(15e6, 600 + C, 10e3, 0.87, 0.85, heaters=(heater,))
    if heater.p_Pa == 1.2e6:
        inputs = RankineInputs(15e6, 450 + C, 10e3, 0.8, 0.85, heaters=(heater,))
    with pytest.raises(ValueError, match=match):
        solve_rankine(inputs)


# ---------------------------------------------------------------------
# Drenajes bombeados hacia adelante desde cualquier cerrado
# ---------------------------------------------------------------------


def _coupled() -> RankineInputs:
    return RankineInputs(
        15e6,
        600 + C,
        10e3,
        heaters=(FeedwaterHeater(2e5, CLOSED, drain_forward=True), FeedwaterHeater(12e5)),
    )


def test_intermediate_pumped_drain_matches_the_coupled_balances_by_hand() -> None:
    r = solve_rankine(_coupled())
    _balances_close(r)
    lay = r.layout
    assert lay is not None
    closed = r.of_kind("closed_heater")[0]
    mixer = r.of_kind("mixer")[0]
    open_heater = r.of_kind("open_heater")[0]
    h = [s.h_J_per_kg for s in r.states]
    h_b1, h_d1 = h[closed.port("bleed").state], h[closed.port("drain_out").state]
    h_i1, h_o1 = h[closed.port("fw_in").state], h[closed.port("fw_out").state]
    h_p = h[mixer.port("drain_in").state]
    h_b2, h_o2 = h[open_heater.port("bleed").state], h[open_heater.port("out").state]

    def balances(v):
        y1, y2, h_mix = v
        return [
            y1 * (h_b1 - h_d1) - (1 - y1 - y2) * (h_o1 - h_i1),  # cerrado de baja
            (1 - y2) * h_mix - (1 - y1 - y2) * h_o1 - y1 * h_p,  # cámara de mezcla
            y2 * h_b2 + (1 - y2) * h_mix - h_o2,  # abierto (sale todo el caudal)
        ]

    y1, y2, h_mix = fsolve(balances, [0.1, 0.1, h_o1], xtol=1e-13)
    assert r.extraction_fractions == pytest.approx((y1, y2), rel=1e-6)
    assert h[mixer.port("out").state] == pytest.approx(h_mix, rel=1e-8)


def test_coupled_procedure_shows_the_system_and_checks_each_balance() -> None:
    r = solve_rankine(_coupled())
    steps = rankine_steps(r, "Técnico")
    titles = [s.title for s in steps]
    assert "Fracciones de extracción: un sistema acoplado" in titles
    assert "Solución del sistema" in titles
    assert sum("verificación del balance" in t for t in titles) == 2
    system = steps[titles.index("Fracciones de extracción: un sistema acoplado")]
    assert len(system.latex) == 3  # dos calentadores y la mezcla
    assert "y₁, y₂ y h" in system.text
    for step in steps:
        for t in step.latex:
            assert t.count("{") == t.count("}"), (step.title, t)
            assert "- -" not in t and "+ -" not in t, (step.title, t)


def test_two_pumped_drains() -> None:
    inputs = RankineInputs(
        15e6,
        600 + C,
        10e3,
        0.87,
        0.85,
        heaters=(
            FeedwaterHeater(2e5, CLOSED, 2.8, drain_forward=True),
            FeedwaterHeater(20e5, CLOSED, 2.8, drain_forward=True),
        ),
    )
    r = solve_rankine(inputs)
    _balances_close(r)
    assert len(r.of_kind("mixer")) == 2 and not r.of_kind("valve")
    assert any("sistema acoplado" in s.title for s in rankine_steps(r, "SI"))


def test_legacy_flag_and_per_heater_flag_give_the_same_cycle() -> None:
    legacy = RankineInputs(
        15e6,
        600 + C,
        10e3,
        reheat=Reheat(4e6, 600 + C),
        heaters=(FeedwaterHeater(0.5e6), FeedwaterHeater(4e6, CLOSED)),
        drain_forward=True,
    )
    per_heater = RankineInputs(
        15e6,
        600 + C,
        10e3,
        reheat=Reheat(4e6, 600 + C),
        heaters=(FeedwaterHeater(0.5e6), FeedwaterHeater(4e6, CLOSED, drain_forward=True)),
    )
    a, b = solve_rankine(legacy), solve_rankine(per_heater)
    assert a.extraction_fractions == pytest.approx(b.extraction_fractions, rel=1e-12)
    assert [s.title for s in rankine_steps(a, "SI")] == [s.title for s in rankine_steps(b, "SI")]


# ---------------------------------------------------------------------
# El ejemplo y la grilla de robustez
# ---------------------------------------------------------------------


def test_real_heaters_example() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = solve_rankine(RANKINE_EXAMPLES[REAL])
    _balances_close(r)
    low, high = r.of_kind("closed_heater")
    T_sat = PropsSI("T", "P", 40e5, "Q", 0, W)
    assert r.states[high.port("fw_out").state].T_K == pytest.approx(T_sat + 1.5, abs=1e-6)
    T_in = r.states[high.port("fw_in").state].T_K
    assert r.states[high.port("drain_out").state].T_K == pytest.approx(T_in + 5.6, abs=1e-6)
    assert r.layout is not None and r.layout.forward_drains == (0,)
    assert "acopladas" in RANKINE_EXAMPLE_NOTES[REAL]
    heaters = rankine_to_dict(r, "Técnico")["datos"]["calentadores"]
    assert heaters[0]["drenaje_bombeado_hacia_adelante"] is True
    assert heaters[2]["desrecalentador"] is True and heaters[2]["DCA"]["valor"] == 5.6
    assert "DCA" not in heaters[0]


def _grid():
    pressures = (0.7e5, 5.0e5, 40.0e5)
    for n in (1, 2, 3):
        for kinds in itertools.product((OPEN, CLOSED), repeat=n):
            closed = [i for i, k in enumerate(kinds) if k == CLOSED]
            for forward in itertools.chain.from_iterable(
                itertools.combinations(closed, r) for r in range(len(closed) + 1)
            ):
                for real in (False, True):
                    heaters = []
                    for i, (p, k) in enumerate(zip(pressures[3 - n :], kinds, strict=True)):
                        if k == OPEN:
                            heaters.append(FeedwaterHeater(p))
                            continue
                        top = p == 40.0e5
                        heaters.append(
                            FeedwaterHeater(
                                p,
                                CLOSED,
                                -1.5 if (real and top) else 2.8,
                                dca_K=5.6 if real else None,
                                desuperheater=real and top,
                                drain_forward=i in forward,
                            )
                        )
                    yield tuple(heaters), real


@pytest.mark.parametrize(("heaters", "real"), list(_grid()))
def test_robustness_grid_real_heaters(heaters, real) -> None:
    inputs = RankineInputs(
        15e6, 600 + C, 10e3, 0.85, 0.8, reheat=Reheat(4e6, 600 + C), heaters=heaters
    )
    try:
        r = solve_rankine(inputs)
    except ValueError as exc:  # rechazado con un mensaje para el alumno
        assert any(word in str(exc) for word in ("calentador", "extracción", "drenaje")), exc
        return
    assert all(0 < y < 1 for y in r.extraction_fractions)
    _balances_close(r)
    assert 0.3 < r.eta_th < 0.5
