"""Tests del Rankine regenerativo (calentadores de agua de alimentación) — Fase 3.1b.

Valores de referencia: Çengel & Boles, *Termodinámica*, ejemplos 10-5 y 10-6,
y cálculos a mano con CoolProp (IAPWS-95) siguiendo los balances del libro.
"""

from __future__ import annotations

import itertools
import pickle
import warnings

import pytest
from CoolProp.CoolProp import PropsSI

from core.cycles.rankine import (
    RANKINE_EXAMPLE_NOTES,
    RANKINE_EXAMPLES,
    FeedwaterHeater,
    RankineInputs,
    RankineResult,
    Reheat,
    solve_rankine,
)

C = 273.15
W = "Water"
OPEN, CLOSED = "open", "closed"


def _h(**kw: float) -> float:
    (k1, v1), (k2, v2) = kw.items()
    return PropsSI("H", k1, v1, k2, v2, W)


def _s(**kw: float) -> float:
    (k1, v1), (k2, v2) = kw.items()
    return PropsSI("S", k1, v1, k2, v2, W)


def _cengel_10_5() -> RankineInputs:
    return RankineInputs(15.0e6, 600 + C, 10.0e3, heaters=(FeedwaterHeater(1.2e6),))


def _cengel_10_6() -> RankineInputs:
    return RankineInputs(
        15.0e6,
        600 + C,
        10.0e3,
        reheat=Reheat(4.0e6, 600 + C),
        heaters=(FeedwaterHeater(0.5e6), FeedwaterHeater(4.0e6, CLOSED)),
        drain_forward=True,
    )


def _closed_to_condenser() -> RankineInputs:
    return RankineInputs(15.0e6, 600 + C, 10.0e3, heaters=(FeedwaterHeater(1.2e6, CLOSED),))


def _three_heaters_real() -> RankineInputs:
    return RankineInputs(
        15.0e6,
        600 + C,
        10.0e3,
        0.85,
        0.8,
        reheat=Reheat(4.0e6, 600 + C),
        heaters=(
            FeedwaterHeater(0.7e5, CLOSED),
            FeedwaterHeater(5.0e5),
            FeedwaterHeater(40.0e5, CLOSED),
        ),
        drain_forward=True,
    )


def _hs(r: RankineResult) -> list[float]:
    return [s.h_J_per_kg for s in r.states]


# ---------------------------------------------------------------------
# Ejemplos de Cengel
# ---------------------------------------------------------------------


class TestCengel10_5:
    """Ideal regenerativo con un calentador abierto a 1,2 MPa (Cengel ej. 10-5)."""

    @classmethod
    def setup_class(cls) -> None:
        cls.r = solve_rankine(_cengel_10_5())

    def test_fraction_and_efficiency(self) -> None:
        (y,) = self.r.extraction_fractions
        assert y == pytest.approx(0.2270, abs=5e-4)  # libro: 0,2270
        assert self.r.eta_th == pytest.approx(0.463, abs=1e-3)  # libro: 46,3 %
        assert len(self.r.states) == 7

    def test_states_match_hand_calculation(self) -> None:
        s1 = _s(P=10e3, Q=0)
        s5 = _s(P=15e6, T=600 + C)
        s3 = _s(P=1.2e6, Q=0)
        hand = [
            _h(P=10e3, Q=0),
            _h(P=1.2e6, S=s1),
            _h(P=1.2e6, Q=0),
            _h(P=15e6, S=s3),
            _h(P=15e6, T=600 + C),
            _h(P=1.2e6, S=s5),
            _h(P=10e3, S=s5),
        ]
        assert _hs(self.r) == pytest.approx(hand, rel=1e-6)
        h = hand
        y = (h[2] - h[1]) / (h[5] - h[1])  # y·h6 + (1 − y)·h2 = h3
        assert self.r.extraction_fractions[0] == pytest.approx(y, rel=1e-6)
        q_in = h[4] - h[3]
        q_out = (1 - y) * (h[6] - h[0])
        assert self.r.eta_th == pytest.approx(1 - q_out / q_in, rel=1e-6)

    def test_regeneration_beats_the_simple_cycle(self) -> None:
        simple = solve_rankine(RankineInputs(15.0e6, 600 + C, 10.0e3))  # Cengel 10-3 c)
        assert self.r.eta_th > simple.eta_th + 0.03
        assert self.r.T_mean_in_K > simple.T_mean_in_K  # sube la T media de aporte de calor

    def test_mixing_generates_entropy(self) -> None:
        # Ideal, pero el calentador abierto mezcla corrientes a distinta T:
        # η queda por debajo de 1 − T_C/T̄_H.
        assert self.r.eta_th < self.r.eta_mean_temperature - 0.01


class TestCengel10_6:
    """Recalentamiento + cerrado a 4 MPa (drenaje bombeado) + abierto a 0,5 MPa."""

    @classmethod
    def setup_class(cls) -> None:
        cls.r = solve_rankine(_cengel_10_6())

    def test_numbering_and_shared_extraction_state(self) -> None:
        lay = self.r.layout
        assert lay is not None and len(self.r.states) == 13
        assert lay.extraction_states[1] == lay.reheat_in == 9  # estado 10
        reheater = self.r.of_kind("reheater")[0]
        y_closed = self.r.extraction_fractions[1]
        assert reheater.port("in").fraction == pytest.approx(1 - y_closed, rel=1e-9)

    def test_matches_the_hand_calculation(self) -> None:
        # Mismos balances que el libro, con la bomba II hasta 15 MPa.
        h = {}
        s1, s3, s6 = _s(P=10e3, Q=0), _s(P=0.5e6, Q=0), _s(P=4e6, Q=0)
        h[1], h[3], h[6] = _h(P=10e3, Q=0), _h(P=0.5e6, Q=0), _h(P=4e6, Q=0)
        h[2], h[4], h[7] = _h(P=0.5e6, S=s1), _h(P=15e6, S=s3), _h(P=15e6, S=s6)
        h[5] = _h(P=15e6, T=PropsSI("T", "P", 4e6, "Q", 0, W))
        s9, s11 = _s(P=15e6, T=600 + C), _s(P=4e6, T=600 + C)
        h[9], h[11] = _h(P=15e6, T=600 + C), _h(P=4e6, T=600 + C)
        h[10], h[12], h[13] = _h(P=4e6, S=s9), _h(P=0.5e6, S=s11), _h(P=10e3, S=s11)
        y = (h[5] - h[4]) / ((h[10] - h[6]) + (h[5] - h[4]))
        z = (1 - y) * (h[3] - h[2]) / (h[12] - h[2])
        h[8] = (1 - y) * h[5] + y * h[7]
        q_in = (h[9] - h[8]) + (1 - y) * (h[11] - h[10])
        q_out = (1 - y - z) * (h[13] - h[1])
        assert _hs(self.r) == pytest.approx([h[k] for k in range(1, 14)], rel=1e-6)
        assert self.r.extraction_fractions == pytest.approx((z, y), rel=1e-6)
        assert self.r.eta_th == pytest.approx(1 - q_out / q_in, rel=1e-6)
        assert (y, z, self.r.eta_th) == pytest.approx((0.1729, 0.1313, 0.4898), abs=5e-4)

    def test_close_to_the_book(self) -> None:
        # Libro: η = 49,2 % (con h₄ = 643,9 kJ/kg; ver RANKINE_EXAMPLE_NOTES).
        assert self.r.eta_th == pytest.approx(0.492, abs=3e-3)
        note = next(v for k, v in RANKINE_EXAMPLE_NOTES.items() if k.startswith("Cengel 10-6"))
        assert "643,9" in note


def test_closed_heater_draining_to_the_condenser() -> None:
    r = solve_rankine(_closed_to_condenser())
    h = _hs(r)
    # Estados: 1 cond, 2 bomba, 3 agua a T_sat(1,2 MPa), 4 turbina, 5 extracción,
    # 6 escape, 7 drenaje, 8 drenaje tras la válvula.
    y = (h[2] - h[1]) / (h[4] - h[6])  # y·(h5 − h7) = h3 − h2
    (y_r,) = r.extraction_fractions
    assert y_r == pytest.approx(y, rel=1e-6)
    assert y_r == pytest.approx(0.2903, abs=5e-4)
    assert r.eta_th == pytest.approx(0.4453, abs=5e-4)
    assert h[7] == pytest.approx(h[6], rel=1e-9)  # la válvula no cambia h
    # El condensador recibe el escape y el drenaje.
    q_out = (1 - y_r) * (h[5] - h[0]) + y_r * (h[7] - h[0])
    assert r.q_out_J_per_kg == pytest.approx(q_out, rel=1e-6)
    # Un abierto a la misma presión rinde más (no tira el drenaje al condensador).
    assert r.eta_th < solve_rankine(_cengel_10_5()).eta_th


def test_closed_heater_cascading_into_the_open_one() -> None:
    r = solve_rankine(
        RankineInputs(
            15.0e6,
            600 + C,
            10.0e3,
            heaters=(FeedwaterHeater(5e5), FeedwaterHeater(40e5, CLOSED)),
        )
    )
    lay = r.layout
    assert lay is not None
    closed = r.of_kind("closed_heater")[0]
    opened = r.of_kind("open_heater")[0]
    h = [s.h_J_per_kg for s in r.states]
    # De mayor a menor presión, como en el libro.
    fw_in, fw_out = closed.port("fw_in").state, closed.port("fw_out").state
    bleed, drain = closed.port("bleed").state, closed.port("drain_out").state
    y_c = (h[fw_out] - h[fw_in]) / (h[bleed] - h[drain])  # todo el caudal pasa por los tubos
    drain_in = opened.port("drain_in").state
    o_in, o_out, o_bleed = (opened.port(r_).state for r_ in ("fw_in", "out", "bleed"))
    y_o = (h[o_out] - h[o_in] - y_c * (h[drain_in] - h[o_in])) / (h[o_bleed] - h[o_in])
    assert r.extraction_fractions == pytest.approx((y_o, y_c), rel=1e-6)


def test_two_closed_heaters_cascade_down_to_the_condenser() -> None:
    r = solve_rankine(
        RankineInputs(
            15.0e6,
            600 + C,
            10.0e3,
            heaters=(FeedwaterHeater(2e5, CLOSED), FeedwaterHeater(10e5, CLOSED)),
        )
    )
    h = [s.h_J_per_kg for s in r.states]
    low, high = r.of_kind("closed_heater")
    y_h = (h[high.port("fw_out").state] - h[high.port("fw_in").state]) / (
        h[high.port("bleed").state] - h[high.port("drain_out").state]
    )
    d_in = low.port("drain_in").state
    d_out = low.port("drain_out").state
    y_l = (
        h[low.port("fw_out").state] - h[low.port("fw_in").state] - y_h * (h[d_in] - h[d_out])
    ) / (h[low.port("bleed").state] - h[d_out])
    assert r.extraction_fractions == pytest.approx((y_l, y_h), rel=1e-6)
    cond = r.of_kind("condenser")[0]
    assert cond.port("drain_in").fraction == pytest.approx(y_l + y_h, rel=1e-9)


def test_two_open_heaters() -> None:
    r = solve_rankine(
        RankineInputs(
            15.0e6, 600 + C, 10.0e3, heaters=(FeedwaterHeater(5e5), FeedwaterHeater(30e5))
        )
    )
    assert r.extraction_fractions == pytest.approx((0.1525, 0.1502), abs=5e-4)
    assert r.eta_th > solve_rankine(_cengel_10_5()).eta_th  # más calentadores, más η
    assert len(r.of_kind("pump")) == 3


# ---------------------------------------------------------------------
# Ciclo real, balances y convención de η_T
# ---------------------------------------------------------------------


def test_real_cycle_follows_one_expansion_line_per_turbine() -> None:
    r = solve_rankine(_three_heaters_real())
    lay = r.layout
    assert lay is not None

    def line(p_in: float, T_in: float, p_out: float) -> float:
        h_in = _h(P=p_in, T=T_in)
        return h_in - 0.85 * (h_in - _h(P=p_out, S=_s(P=p_in, T=T_in)))

    assert r.states[lay.reheat_in].h_J_per_kg == pytest.approx(line(15e6, 600 + C, 4e6), rel=1e-9)
    for idx, p in ((lay.extraction_states[1], 5e5), (lay.extraction_states[0], 0.7e5)):
        assert r.states[idx].h_J_per_kg == pytest.approx(line(4e6, 600 + C, p), rel=1e-9)
    assert r.states[lay.exhaust].h_J_per_kg == pytest.approx(line(4e6, 600 + C, 10e3), rel=1e-9)


@pytest.mark.parametrize(
    "inputs",
    [_cengel_10_5(), _cengel_10_6(), _closed_to_condenser(), _three_heaters_real()],
)
def test_energy_and_mass_balances(inputs: RankineInputs) -> None:
    r = solve_rankine(inputs)
    # Tolerancia de convergencia de TESPy (~1e-3 J/kg en los Merge).
    assert r.q_in_J_per_kg - r.q_out_J_per_kg == pytest.approx(r.w_net_J_per_kg, rel=1e-8)
    for comp in r.components:
        m_in = sum(p.fraction for p in comp.inlets)
        m_out = sum(p.fraction for p in comp.outlets)
        assert m_in == pytest.approx(m_out, rel=1e-9), comp.label
        if comp.kind in ("open_heater", "closed_heater", "mixer", "valve"):  # adiabáticos
            e_in = sum(p.fraction * r.states[p.state].h_J_per_kg for p in comp.inlets)
            e_out = sum(p.fraction * r.states[p.state].h_J_per_kg for p in comp.outlets)
            assert e_in == pytest.approx(e_out, rel=1e-8), comp.label
    balances = dict(r.tespy_balances)
    turbines = -sum(v for k, v in balances.items() if k.startswith("turbina"))
    pumps = sum(v for k, v in balances.items() if k.startswith("bomba"))
    assert turbines == pytest.approx(r.w_turbine_J_per_kg, rel=1e-6)
    assert pumps == pytest.approx(r.w_pump_J_per_kg, rel=1e-6)
    assert balances["caldera"] + balances.get("recalentador", 0.0) == pytest.approx(
        r.q_in_J_per_kg, rel=1e-6
    )
    assert -balances["condensador"] == pytest.approx(r.q_out_J_per_kg, rel=1e-6)
    for label, heat in r.tespy_heaters:
        assert heat > 0, label


def test_saturated_inlet_and_supercritical_boiler() -> None:
    sat = solve_rankine(
        RankineInputs(
            8e6, None, 10e3, heaters=(FeedwaterHeater(5e5), FeedwaterHeater(20e5, CLOSED))
        )
    )
    assert sat.states[sat.layout.turbine_inlet].x == pytest.approx(1.0)  # type: ignore[union-attr]
    assert all(0 < y < 1 for y in sat.extraction_fractions)
    supercritical = solve_rankine(
        RankineInputs(
            25e6,
            600 + C,
            5e3,
            0.9,
            0.85,
            heaters=(FeedwaterHeater(3e5), FeedwaterHeater(60e5, CLOSED, ttd_K=3.0)),
            drain_forward=True,
        )
    )
    closed = supercritical.of_kind("closed_heater")[0]
    T_out = supercritical.states[closed.port("fw_out").state].T_K
    assert T_out == pytest.approx(PropsSI("T", "P", 60e5, "Q", 1, W) - 3.0, abs=1e-6)  # TTD


@pytest.mark.parametrize("p_ext", [8e6, 4e6, 1e6])
def test_extraction_above_at_and_below_the_reheat_pressure(p_ext: float) -> None:
    r = solve_rankine(
        RankineInputs(
            15e6,
            600 + C,
            10e3,
            reheat=Reheat(4e6, 600 + C),
            heaters=(FeedwaterHeater(p_ext, CLOSED),),
        )
    )
    lay = r.layout
    assert lay is not None
    (bleed,) = lay.extraction_states
    if p_ext == 4e6:
        assert bleed == lay.reheat_in  # sale antes de recalentar
    elif p_ext > 4e6:
        assert bleed < lay.reheat_in
    else:
        assert bleed > lay.reheat_out  # type: ignore[operator]
    assert 0 < r.extraction_fractions[0] < 1


# ---------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------


def _with(*heaters: FeedwaterHeater, **kw) -> RankineInputs:
    return RankineInputs(15e6, 600 + C, 10e3, heaters=heaters, **kw)


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (_with(FeedwaterHeater(5e3)), "entre la del condensador"),
        (_with(FeedwaterHeater(16e6)), "entre la del condensador"),
        (_with(FeedwaterHeater(5e6), FeedwaterHeater(5e5)), "tienen que crecer"),
        (_with(FeedwaterHeater(5e5), FeedwaterHeater(5e5, CLOSED)), "no pueden ser iguales"),
        (
            RankineInputs(30e6, 600 + C, 10e3, heaters=(FeedwaterHeater(25e6),)),
            "menor que la crítica",
        ),
        (_with(*(FeedwaterHeater(1e5 * k) for k in range(1, 5))), "hasta 3"),
        (_with(FeedwaterHeater(5e5, CLOSED, ttd_K=-2.0)), "no puede ser negativa"),
        (_with(FeedwaterHeater(5e5, OPEN, ttd_K=2.0)), "no lleva TTD"),
        (
            _with(FeedwaterHeater(5e5, CLOSED), FeedwaterHeater(40e5), drain_forward=True),
            "Solo un calentador cerrado",
        ),
        # Tras la bomba II el agua sale 1,6 K por encima de T_sat(5 bar): el cerrado a
        # 5,05 bar no la puede calentar.
        (_with(FeedwaterHeater(5e5), FeedwaterHeater(5.05e5, CLOSED)), "no puede calentar"),
    ],
)
def test_invalid_heaters_are_explained(inputs: RankineInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_rankine(inputs)


def test_negative_extraction_into_an_open_heater_is_explained() -> None:
    # El drenaje del cerrado de alta le aporta al abierto más calor del que necesita.
    inputs = _with(
        FeedwaterHeater(10e5, CLOSED), FeedwaterHeater(10.5e5), FeedwaterHeater(60e5, CLOSED)
    )
    with pytest.raises(ValueError, match="calentador abierto intermedio necesitaría"):
        solve_rankine(inputs)


# ---------------------------------------------------------------------
# Ejemplos, datos y robustez
# ---------------------------------------------------------------------


def test_regenerative_examples_keep_their_place_and_solve_without_warnings() -> None:
    # Los de la 0.12.0 van del 6 al 8 (los de la 3.1c se suman después).
    names = list(RANKINE_EXAMPLES)
    assert names[0].startswith("Cengel 10-1")
    regen = [n for n in names[:9] if RANKINE_EXAMPLES[n].heaters]
    assert names[6:9] == regen and len(regen) == 3
    for name in regen:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solve_rankine(RANKINE_EXAMPLES[name])


def test_heaters_list_is_stored_as_a_tuple_and_results_pickle() -> None:
    inputs = RankineInputs(15e6, 600 + C, 10e3, heaters=[FeedwaterHeater(1.2e6)])  # type: ignore[arg-type]
    assert isinstance(inputs.heaters, tuple)
    hash(inputs)
    r = solve_rankine(inputs)
    clone = pickle.loads(pickle.dumps(r))
    assert clone.eta_th == pytest.approx(r.eta_th)
    assert clone.extraction_fractions == r.extraction_fractions


def _grid():
    pressures = (0.7e5, 5.0e5, 40.0e5)
    for n in (1, 2, 3):
        for kinds in itertools.product((OPEN, CLOSED), repeat=n):
            heaters = tuple(
                FeedwaterHeater(p, k) for p, k in zip(pressures[3 - n :], kinds, strict=True)
            )
            for reheat in (None, Reheat(4e6, 600 + C)):
                yield heaters, reheat, False
                if kinds[-1] == CLOSED:
                    yield heaters, reheat, True


@pytest.mark.parametrize(("heaters", "reheat", "forward"), list(_grid()))
def test_robustness_grid_real_cycles(heaters, reheat, forward) -> None:
    inputs = RankineInputs(
        15e6, 600 + C, 10e3, 0.85, 0.8, reheat=reheat, heaters=heaters, drain_forward=forward
    )
    r = solve_rankine(inputs)  # converge con status 0 o lo explica la validación
    assert all(0 < y < 1 for y in r.extraction_fractions)
    # Tolerancia de convergencia de TESPy (~1e-3 J/kg en los Merge).
    assert r.q_in_J_per_kg - r.q_out_J_per_kg == pytest.approx(r.w_net_J_per_kg, rel=1e-8)
    assert 0.3 < r.eta_th < 0.5


# ---------------------------------------------------------------------
# Caudales simbólicos, procedimiento y agua de enfriamiento
# ---------------------------------------------------------------------

_PROCEDURE_CASES = [_cengel_10_5(), _cengel_10_6(), _closed_to_condenser(), _three_heaters_real()]


@pytest.mark.parametrize("inputs", _PROCEDURE_CASES)
def test_symbolic_flows_match_the_solved_fractions(inputs: RankineInputs) -> None:
    from core.cycles.rankine_layout import port_flows

    r = solve_rankine(inputs)
    assert r.layout is not None
    flows = port_flows(r.layout)
    ys = r.extraction_fractions
    for ci, comp in enumerate(r.components):
        for pi, port in enumerate(comp.ports):
            value = sum(c * (1.0 if k is None else ys[k]) for k, c in flows[(ci, pi)].items())
            assert value == pytest.approx(port.fraction, rel=1e-7, abs=1e-9), (comp.label, port)


def _last_value(tex: str) -> float:
    """Último número de una cadena ``… &= 0.22707\\end{aligned}``."""
    return float(tex.rsplit("&=", 1)[1].replace(r"\end{aligned}", "").split(r"\ ")[0])


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
@pytest.mark.parametrize("inputs", _PROCEDURE_CASES)
def test_regenerative_procedure(inputs: RankineInputs, system: str) -> None:
    from core.cycles.rankine import condenser_cooling_water, rankine_steps

    r = solve_rankine(inputs)
    cooling = condenser_cooling_water(r, 20 + C, 30 + C)
    steps = rankine_steps(r, system, cooling=cooling)  # type: ignore[arg-type]
    for step in steps:
        for t in step.latex:
            assert t.count("{") == t.count("}"), (step.title, t)
            assert "- -" not in t and "+ -" not in t, (step.title, t)
    titles = [s.title for s in steps]
    balances = [s for s in steps if "fracción de extracción" in s.title]
    assert len(balances) == len(inputs.heaters)
    # De mayor a menor presión, y la y que se muestra es la del resultado.
    names = r.layout.heater_names  # type: ignore[union-attr]
    expected_order = [names[k][0].upper() + names[k][1:] for k in reversed(range(len(names)))]
    assert [s.title.split(":")[0] for s in balances] == expected_order
    for k, step in zip(reversed(range(len(names))), balances, strict=True):
        y_line = next(t for t in reversed(step.latex) if t.startswith(r"\begin{aligned}y"))
        assert _last_value(y_line) == pytest.approx(r.extraction_fractions[k], rel=1e-4)
    assert ("cámara de mezcla" in " ".join(titles)) == inputs.drain_forward
    assert titles[-1] == "Agua de enfriamiento del condensador"
    w_line = next(t for s in steps for t in s.latex if t.startswith(r"\begin{aligned}w_T"))
    assert "&\\quad +" in w_line  # un término por tramo de turbina, cada uno en su renglón


def test_cooling_water() -> None:
    from core.cycles.rankine import condenser_cooling_water

    r = solve_rankine(_cengel_10_5())
    cw = condenser_cooling_water(r, 20 + C, 30 + C)
    dh = _h(P=101325.0, T=30 + C) - _h(P=101325.0, T=20 + C)
    assert cw.m_dot_kg_s == pytest.approx(r.Q_out_W / dh, rel=1e-9)
    assert cw.m_dot_cp_kg_s == pytest.approx(cw.m_dot_kg_s, rel=5e-3)  # c_p = 4,18 kJ/(kg·K)
    assert cw.m_dot_kg_s == pytest.approx(
        1486.7e3 / (4180 * 10), rel=5e-3
    )  # ~35,6 kg por kg de vapor


@pytest.mark.parametrize(
    ("T_in", "T_out", "match"),
    [
        (20.0, 50.0, "no puede salir más caliente"),  # T_sat(10 kPa) = 45,8 °C
        (30.0, 25.0, "tiene que ser mayor"),
        (-5.0, 10.0, "por encima de 0 °C"),
    ],
)
def test_cooling_water_is_validated(T_in: float, T_out: float, match: str) -> None:
    from core.cycles.rankine import condenser_cooling_water

    r = solve_rankine(_cengel_10_5())
    with pytest.raises(ValueError, match=match):
        condenser_cooling_water(r, T_in + C, T_out + C)


# ---------------------------------------------------------------------
# Barridos, export y diagrama
# ---------------------------------------------------------------------


def test_extraction_pressure_sweep_has_an_optimum() -> None:
    from core.cycles.rankine import default_extraction_values, extraction_pressure_sweep

    inputs = _cengel_10_5()
    values = default_extraction_values(inputs, 0)
    assert values[0] > inputs.p_condenser_Pa and values[-1] < inputs.p_boiler_Pa
    points = extraction_pressure_sweep(inputs, 0, values)
    etas = [p.eta_th for p in points]
    best = etas.index(max(etas))
    assert 0 < best < len(etas) - 1  # máximo interior (Cengel §10-6)
    ys = [p.fractions[0] for p in points]
    assert ys == sorted(ys)  # más presión, más extracción


def test_extraction_sweep_does_not_cross_the_reheat_pressure() -> None:
    from core.cycles.rankine import default_extraction_values

    inputs = _cengel_10_6()
    closed = default_extraction_values(inputs, 1)  # a 4 MPa = recalentamiento: del lado de alta
    assert min(closed) == pytest.approx(4e6) and max(closed) < 15e6
    opened = default_extraction_values(inputs, 0)
    assert max(opened) < 4e6


def test_existing_sweeps_stay_clear_of_the_heaters() -> None:
    from core.cycles.rankine import default_sweep_values, rankine_sweep

    inputs = _cengel_10_6()
    p_boiler = default_sweep_values(inputs, "p_boiler")
    assert min(p_boiler) > 4e6
    p_cond = default_sweep_values(inputs, "p_condenser")
    assert max(p_cond) < 0.5e6
    assert len(rankine_sweep(inputs, "p_condenser", p_cond)) == len(p_cond)


def test_export_with_heaters_and_cooling_water() -> None:
    import json

    from core.cycles.rankine import condenser_cooling_water, rankine_to_dict
    from core.export import dict_to_csv

    r = solve_rankine(_cengel_10_6())
    cw = condenser_cooling_water(r, 20 + C, 30 + C)
    data = rankine_to_dict(r, "Técnico", cooling=cw)
    json.dumps(data, ensure_ascii=False)
    assert data["ciclo"] == "Rankine regenerativo con recalentamiento"
    assert [c["tipo"] for c in data["datos"]["calentadores"]] == ["abierto", "cerrado"]
    assert data["datos"]["drenaje_bombeado_hacia_adelante"] is True
    fractions = [f["y"] for f in data["resultados"]["fracciones_de_extraccion"]]
    assert fractions == pytest.approx([0.1313, 0.1729], abs=5e-4)
    assert data["resultados"]["agua_de_enfriamiento"]["caudal"]["unidad"] == "kg/s"
    assert len(data["estados"]) == 13
    assert "resultados.agua_de_enfriamiento.caudal.valor" in dict_to_csv(data)


def test_segments_cover_every_process() -> None:
    from core.cycles.rankine import RANKINE_EXAMPLES, rankine_segments

    simple = solve_rankine(RANKINE_EXAMPLES[next(iter(RANKINE_EXAMPLES))])
    assert rankine_segments(simple) == [(0, 1), (1, 2), (2, 3), (3, 0)]
    r = solve_rankine(_cengel_10_6())
    segs = {(a + 1, b + 1) for a, b in rankine_segments(r)}
    assert {(10, 6), (12, 3), (5, 8), (7, 8), (10, 11), (13, 1)} <= segs  # extracciones y mezcla
    assert len(segs) == 15


def test_segments_overlays_draw_bleeds_as_isobars_and_valves_as_lines() -> None:
    import numpy as np

    from core.cycles.rankine import rankine_segments
    from core.diagrams import DiagramSpec, segment_between, segments_overlays
    from ui.diagrams import get_diagram

    r = solve_rankine(_closed_to_condenser())
    diagram = get_diagram("Water", "Técnico")
    spec = DiagramSpec(fluid="Water", system="Técnico")
    points = [s.to_state_point() for s in r.states]
    pairs = [(points[a], points[b]) for a, b in rankine_segments(r)]
    overlays = segments_overlays(diagram, spec, pairs)
    assert [o.name for o in overlays] == [
        "procesos a p o s constante",
        "uniones rectas (referencia)",
    ]
    assert all(np.isfinite(o.coords_si["s"]).any() for o in overlays)
    # 5 → 7: la extracción condensa a p constante; 7 → 8: la válvula es una recta.
    assert segment_between(diagram, spec, points[4], points[6])[0] == "isobaric"
    assert segment_between(diagram, spec, points[6], points[7])[0] == "straight"
