"""Tests de la HRSG de una presión (diagrama T–Q) — Fase 3.3.

Referencias: el balance de energía a mano (gases ideales con CoolProp, agua
IAPWS-95) y, como control independiente, la misma caldera armada con TESPy
(``HeatExchanger`` en serie y los gases como mezcla, como su tutorial de
turbina de gas).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import replace

import pytest
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from core.cycles.hrsg import (
    GAS_SPECIES,
    HRSG_EXAMPLE_NOTES,
    HRSG_EXAMPLES,
    HRSG_EXAMPLES_EXCESS_AIR,
    FlueGas,
    HRSGInputs,
    default_steam_temperature,
    default_sweep_values,
    exhaust_composition,
    hrsg_notes,
    hrsg_sweep,
    hrsg_to_dict,
    hrsg_tq_profile,
    molar_mass,
    other_steam_option,
    solve_hrsg,
)
from core.cycles.hrsg_procedure import hrsg_steps

C = 273.15
NAMES = list(HRSG_EXAMPLES)
EX_TG, EX_SAT, EX_LOW = NAMES
COOLPROP = {"N2": "Nitrogen", "O2": "Oxygen", "CO2": "CarbonDioxide", "H2O": "Water", "Ar": "Argon"}
TYPICAL = {"N2": 0.7448, "O2": 0.1225, "CO2": 0.0396, "H2O": 0.0842, "Ar": 0.0089}


def _hand(inputs: HRSGInputs) -> dict[str, float]:
    """El balance a mano, independiente del módulo (PropsSI y brentq)."""
    gas = inputs.gas
    w = gas.mass_fractions

    def h_g(T: float) -> float:
        return sum(
            wi
            * (
                PropsSI("H", "T", T, "P", 1.0, COOLPROP[s])
                - PropsSI("H", "T", 298.15, "P", 1.0, COOLPROP[s])
            )
            for s, wi in w.items()
            if wi > 0
        )

    p = inputs.p_steam_Pa
    T_sat = PropsSI("T", "P", p, "Q", 0, "Water")
    h1 = PropsSI("H", "T", inputs.T_feedwater_K, "P", p, "Water")
    h2 = PropsSI("H", "T", T_sat - inputs.approach_K, "P", p, "Water")
    h4 = PropsSI("H", "P", p, "Q", 1, "Water")
    top = h4 if inputs.T_steam_K is None else PropsSI("H", "T", inputs.T_steam_K, "P", p, "Water")
    T_c = T_sat + inputs.pinch_K
    m_g = inputs.m_gas_kg_s
    m_s = m_g * (h_g(inputs.T_gas_in_K) - h_g(T_c)) / (top - h2)
    T_b = brentq(lambda T: h_g(T) - (h_g(inputs.T_gas_in_K) - m_s * (top - h4) / m_g), 300, 1500)
    T_d = brentq(lambda T: h_g(T) - (h_g(T_c) - m_s * (h2 - h1) / m_g), 274, 1500)
    return {"m_s": m_s, "T_b": T_b, "T_d": T_d, "Q": m_s * (top - h1)}


def _tespy(inputs: HRSGInputs) -> dict[str, float]:
    """La misma caldera con TESPy: gases → (SH) → EV → ECO; agua en contracorriente."""
    from tespy.components import HeatExchanger, Sink, Source
    from tespy.connections import Connection

    from core.cycles.tespy_utils import new_network, solve

    p = inputs.p_steam_Pa
    T_sat = PropsSI("T", "P", p, "Q", 0, "Water")
    nw = new_network()
    gin, gout = Source("gases"), Sink("chimenea")
    win, wout = Source("agua de alimentación"), Sink("vapor")
    eco, ev = HeatExchanger("economizador"), HeatExchanger("evaporador")
    hxs = [eco, ev]
    if inputs.superheated:
        sh = HeatExchanger("sobrecalentador")
        hxs.append(sh)
        g1 = Connection(gin, "out1", sh, "in1", label="g1")
        g2 = Connection(sh, "out1", ev, "in1", label="g2")
        s4 = Connection(ev, "out2", sh, "in2", label="w4")
        s5 = Connection(sh, "out2", wout, "in1", label="w5")
        conns = [g1, g2, s4, s5]
    else:
        g1 = Connection(gin, "out1", ev, "in1", label="g1")
        s4 = Connection(ev, "out2", wout, "in1", label="w4")
        conns = [g1, s4]
    g3 = Connection(ev, "out1", eco, "in1", label="g3")
    g4 = Connection(eco, "out1", gout, "in1", label="g4")
    w1 = Connection(win, "out1", eco, "in2", label="w1")
    w2 = Connection(eco, "out2", ev, "in2", label="w2")
    nw.add_conns(*conns, g3, g4, w1, w2)
    for hx in hxs:
        hx.set_attr(pr1=1, pr2=1)
    fluid = {s: w for s, w in inputs.gas.mass_fractions.items() if w > 0}
    g1.set_attr(T=inputs.T_gas_in_K, p=inputs.gas.p_Pa, m=inputs.m_gas_kg_s, fluid=fluid)
    w1.set_attr(T=inputs.T_feedwater_K, p=p, fluid={"water": 1})
    w2.set_attr(td_bubble=inputs.approach_K)
    s4.set_attr(x=1)
    g3.set_attr(T=T_sat + inputs.pinch_K)
    if inputs.superheated:
        s5.set_attr(T=inputs.T_steam_K)
    solve(nw, what="la HRSG")
    out = {"m_s": w1.m.val_SI, "T_d": g4.T.val_SI}
    if inputs.superheated:
        out["T_b"] = g2.T.val_SI
    return out


# ---------------------------------------------------------------------
# Gases: mezcla de gases ideales
# ---------------------------------------------------------------------


def test_composition_is_normalized_and_converted() -> None:
    gas = FlueGas.from_fractions({k: 2 * v for k, v in TYPICAL.items()})
    assert sum(gas.y) == pytest.approx(1.0)
    assert gas.mole_fractions["N2"] == pytest.approx(0.7448 / sum(TYPICAL.values()))
    again = FlueGas.from_fractions(gas.mass_fractions, basis="mass")
    assert again.y == pytest.approx(gas.y, rel=1e-12)
    M = sum(TYPICAL[s] * molar_mass(s) for s in GAS_SPECIES) / sum(TYPICAL.values())
    assert gas.M_kg_per_mol == pytest.approx(M, rel=1e-12)
    assert sum(gas.mass_fractions.values()) == pytest.approx(1.0)


def test_pure_nitrogen_matches_coolprop() -> None:
    gas = FlueGas.from_fractions({"N2": 1.0})
    assert gas.M_kg_per_mol == pytest.approx(0.0280134, rel=1e-5)
    assert gas.R_J_per_kg_K == pytest.approx(296.8, rel=1e-3)
    dh = PropsSI("H", "T", 800, "P", 1.0, "Nitrogen") - PropsSI(
        "H", "T", 298.15, "P", 1.0, "Nitrogen"
    )
    assert gas.h(800.0) == pytest.approx(dh, rel=1e-10)
    assert gas.h(298.15) == pytest.approx(0.0, abs=1e-6)
    assert gas.T_from_h(gas.h(650.0)) == pytest.approx(650.0, abs=1e-6)
    assert gas.cp(500.0) == pytest.approx((gas.h(500.5) - gas.h(499.5)) / 1.0, rel=1e-4)
    assert gas.dew_point_K is None  # sin vapor de agua


def test_dew_point_of_the_flue_gas() -> None:
    gas = FlueGas.from_fractions({"N2": 0.9, "H2O": 0.1})
    assert gas.p_H2O_Pa == pytest.approx(10_132.5)
    assert gas.dew_point_K == pytest.approx(PropsSI("T", "P", 10_132.5, "Q", 1, "Water"))
    assert 45.0 < gas.dew_point_K - C < 47.0


@pytest.mark.parametrize(
    ("fractions", "match"),
    [
        ({"He": 1.0}, "desconocidos"),
        ({"N2": -0.1, "O2": 1.1}, "negativas"),
        ({"N2": 0.0}, "vacía"),
    ],
)
def test_invalid_composition(fractions: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FlueGas.from_fractions(fractions)


def test_methane_exhaust_composition() -> None:
    stoich = exhaust_composition(1.0)
    assert stoich["O2"] == pytest.approx(0.0, abs=1e-15)  # λ = 1: no sobra oxígeno
    # CH₄ + 2/0,21·(0,21 O₂ + 0,79 N₂) → CO₂ + 2 H₂O + 7,524 N₂ (vademecum §16.2)
    assert stoich["CO2"] == pytest.approx(1.0 / (3.0 + 2.0 * 0.79 / 0.21))
    lam3 = exhaust_composition(3.0)
    assert sum(lam3.values()) == pytest.approx(1.0)
    assert lam3["H2O"] == pytest.approx(2.0 * lam3["CO2"])
    assert lam3["O2"] == pytest.approx(0.1353, abs=1e-4)
    with pytest.raises(ValueError, match="al menos 1"):
        exhaust_composition(0.9)


# ---------------------------------------------------------------------
# Balance contra el cálculo a mano y contra TESPy
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_examples_match_the_hand_calculation(name: str) -> None:
    inputs = HRSG_EXAMPLES[name]
    r = solve_hrsg(inputs)
    hand = _hand(inputs)
    assert r.m_steam_kg_s == pytest.approx(hand["m_s"], rel=1e-9)
    assert r.T_stack_K == pytest.approx(hand["T_d"], abs=1e-6)
    assert r.Q_W == pytest.approx(hand["Q"], rel=1e-9)
    if inputs.superheated:
        assert r.T_after_superheater_K == pytest.approx(hand["T_b"], abs=1e-6)


@pytest.mark.parametrize("superheated", [True, False])
def test_tespy_gives_the_same_boiler(superheated: bool) -> None:
    inputs = replace(
        HRSG_EXAMPLES[EX_TG],
        gas=FlueGas.from_fractions(TYPICAL),
        T_steam_K=HRSG_EXAMPLES[EX_TG].T_steam_K if superheated else None,
    )
    r = solve_hrsg(inputs)
    ref = _tespy(inputs)
    # TESPy evalúa cada componente a su presión parcial: la diferencia es < 0,1 %.
    assert r.m_steam_kg_s == pytest.approx(ref["m_s"], rel=1e-3)
    assert r.T_stack_K == pytest.approx(ref["T_d"], abs=0.3)
    if superheated:
        assert r.T_after_superheater_K == pytest.approx(ref["T_b"], abs=0.3)


@pytest.mark.parametrize("name", NAMES)
def test_energy_balances(name: str) -> None:
    r = solve_hrsg(HRSG_EXAMPLES[name])
    assert r.Q_W == pytest.approx(r.Q_gas_W, rel=1e-9)
    assert sum(s.Q_W for s in r.sections) == pytest.approx(r.Q_W, rel=1e-12)
    gas = r.inputs.gas
    for s in r.sections:  # cada sección: lo que ceden los gases lo recibe el agua
        dh = gas.h(s.T_gas_in_K) - gas.h(s.T_gas_out_K)
        assert r.inputs.m_gas_kg_s * dh == pytest.approx(s.Q_W, rel=1e-8)
    assert 0.0 < r.recovery < 1.0
    assert r.T_pinch_gas_K == pytest.approx(r.T_sat_K + r.inputs.pinch_K)
    assert r.water[1].T_K == pytest.approx(r.T_sat_K - r.inputs.approach_K, abs=1e-9)
    sections = {s.name: s for s in r.sections}
    assert sections["evaporador"].dT_cold_K == pytest.approx(r.inputs.pinch_K)
    eco = sections["economizador"]
    assert eco.dT_hot_K == pytest.approx(r.inputs.pinch_K + r.inputs.approach_K, abs=1e-9)
    assert eco.dT_cold_K == pytest.approx(r.T_stack_K - r.inputs.T_feedwater_K)


def test_saturated_steam_gives_more_steam_and_a_colder_stack() -> None:
    superheated = solve_hrsg(HRSG_EXAMPLES[EX_TG])
    saturated = solve_hrsg(replace(HRSG_EXAMPLES[EX_TG], T_steam_K=None))
    assert saturated.m_steam_kg_s > superheated.m_steam_kg_s
    assert saturated.T_stack_K < superheated.T_stack_K
    assert saturated.Q_W > superheated.Q_W
    assert len(saturated.sections) == 2 and saturated.T_after_superheater_K is None
    assert saturated.gas_labels == ("a", "c", "d")


def test_smaller_pinch_gives_more_steam() -> None:
    base = solve_hrsg(HRSG_EXAMPLES[EX_TG])
    low = solve_hrsg(HRSG_EXAMPLES[EX_LOW])
    assert low.m_steam_kg_s > base.m_steam_kg_s
    assert low.T_stack_K < base.T_stack_K


# ---------------------------------------------------------------------
# Diagrama T–Q
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_tq_profile(name: str) -> None:
    r = solve_hrsg(HRSG_EXAMPLES[name])
    prof = hrsg_tq_profile(r)
    assert prof.min_dT_K == pytest.approx(r.inputs.pinch_K, abs=1e-6)
    assert prof.min_dT_Q_W == pytest.approx(prof.boundaries_W[0], rel=1e-9)  # salida del EV
    assert len(prof.boundaries_W) == (2 if r.inputs.superheated else 1)
    Qg, Tg = prof.Q_gas_W, prof.T_gas_K
    assert all(b > a for a, b in zip(Qg, Qg[1:], strict=False))
    assert all(b > a for a, b in zip(Tg, Tg[1:], strict=False))
    assert Qg[0] == pytest.approx(0.0, abs=1e-6) and Qg[-1] == pytest.approx(r.Q_W, rel=1e-9)
    Tw = prof.T_water_K
    assert all(b >= a - 1e-9 for a, b in zip(Tw, Tw[1:], strict=False))
    assert Tw[0] == pytest.approx(r.inputs.T_feedwater_K, abs=1e-6)
    assert Tw[-1] == pytest.approx(r.water[-1].T_K, abs=1e-6)
    assert prof.Q_water_W[-1] == pytest.approx(r.Q_W, rel=1e-9)


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------

BASE = HRSG_EXAMPLES[EX_TG]
AIR = FlueGas.from_fractions({"N2": 0.79, "O2": 0.21})


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (replace(BASE, m_gas_kg_s=0.0), "caudal de gases"),
        (replace(BASE, T_gas_in_K=2500.0), "fuera del rango"),
        (replace(BASE, p_steam_Pa=250e5), "supera la crítica"),
        (replace(BASE, pinch_K=0.0), "pinch tiene que ser positivo"),
        (replace(BASE, approach_K=-1.0), "approach no puede ser negativo"),
        (replace(BASE, T_feedwater_K=280.0 + C), "más fría que la salida del economizador"),
        (replace(BASE, T_gas_in_K=280.0 + C, T_steam_K=None), "no alcanzan para evaporar"),
        (replace(BASE, T_steam_K=270.0 + C), "más caliente que la saturación"),
        (replace(BASE, T_steam_K=600.0 + C), "más caliente que los gases"),
        (
            replace(
                BASE,
                gas=AIR,
                T_gas_in_K=900 + C,
                p_steam_Pa=5e5,
                T_feedwater_K=20 + C,
                pinch_K=2.0,
                T_steam_K=None,
            ),
            "cruce de temperaturas",
        ),
        (
            replace(
                BASE,
                gas=FlueGas.from_fractions({"N2": 0.5, "H2O": 0.5}),
                p_steam_Pa=3e5,
                T_feedwater_K=37 + C,
                pinch_K=5.0,
                T_steam_K=None,
            ),
            "punto de rocío",
        ),
    ],
)
def test_invalid_inputs_are_explained(inputs: HRSGInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_hrsg(inputs)


def test_notes() -> None:
    cold_feed = solve_hrsg(replace(BASE, T_feedwater_K=30 + C))
    assert any("punto de rocío" in n for n in hrsg_notes(cold_feed))
    zero = solve_hrsg(replace(BASE, approach_K=0.0))
    assert any("approach = 0" in n for n in hrsg_notes(zero))
    assert zero.water[1].x == pytest.approx(0.0, abs=1e-9)  # líquido saturado
    assert hrsg_notes(solve_hrsg(BASE)) == []


# ---------------------------------------------------------------------
# Barridos, export y procedimiento
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parameter", "steam_decreases"),
    [("pinch", True), ("T_steam", True), ("T_gas_in", False)],
)
def test_sweeps(parameter: str, steam_decreases: bool) -> None:
    values = default_sweep_values(BASE, parameter, n=5)  # type: ignore[arg-type]
    points = hrsg_sweep(BASE, parameter, values)  # type: ignore[arg-type]
    assert len(points) == 5
    steam = [p.m_steam_kg_s for p in points]
    assert steam == sorted(steam, reverse=steam_decreases)
    for point in points[::2]:
        field = {"pinch": "pinch_K", "T_steam": "T_steam_K", "T_gas_in": "T_gas_in_K"}[parameter]
        r = solve_hrsg(replace(BASE, **{field: point.value_si}))
        assert point.m_steam_kg_s == r.m_steam_kg_s
        assert point.T_stack_K == r.T_stack_K
        assert point.Q_W == r.Q_W


def test_sweep_p_steam() -> None:
    """Más presión: menos vapor y chimenea más caliente, hasta acercarse a la crítica.

    Cerca de la crítica h_fg se achica y el caudal vuelve a subir (con 540 °C,
    el mínimo queda cerca de 100 bar): el barrido no es monótono en general.
    """
    values = default_sweep_values(BASE, "p_steam", n=9)
    assert 1.0e5 <= values[0] < BASE.p_steam_Pa < values[-1] <= 160.0e5
    assert len(hrsg_sweep(BASE, "p_steam", values)) == 9
    low = hrsg_sweep(BASE, "p_steam", [5e5, 10e5, 20e5, 40e5, 60e5])
    steam = [p.m_steam_kg_s for p in low]
    stack = [p.T_stack_K for p in low]
    assert steam == sorted(steam, reverse=True)
    assert stack == sorted(stack)
    near_critical = hrsg_sweep(BASE, "p_steam", [100e5, 200e5])
    assert near_critical[1].m_steam_kg_s > near_critical[0].m_steam_kg_s


def test_other_sweeps_run() -> None:
    for parameter in ("approach", "T_feedwater"):
        values = default_sweep_values(BASE, parameter, n=5)  # type: ignore[arg-type]
        assert len(hrsg_sweep(BASE, parameter, values)) >= 4  # type: ignore[arg-type]
    assert hrsg_sweep(HRSG_EXAMPLES[EX_SAT], "T_steam", [500.0]) == []
    assert default_sweep_values(HRSG_EXAMPLES[EX_SAT], "T_steam") == []


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
@pytest.mark.parametrize("name", NAMES)
def test_export_and_procedure(name: str, system: str) -> None:
    r = solve_hrsg(HRSG_EXAMPLES[name])
    data = hrsg_to_dict(r, system)  # type: ignore[arg-type]
    json.dumps(data, ensure_ascii=False)
    assert data["resultados"]["aprovechamiento"] == pytest.approx(r.recovery)
    assert len(data["agua"]) == len(r.water)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        steps = hrsg_steps(r, system)  # type: ignore[arg-type]
    latex = [t for s in steps for t in s.latex]
    assert all(t.count("{") == t.count("}") for t in latex)
    assert not any("- -" in t or "+ -" in t for t in latex)
    titles = [s.title for s in steps]
    assert titles[0] == "Composición de los gases"
    assert any(t.startswith("Sobrecalentador") for t in titles) == r.inputs.superheated
    if system == "Técnico":
        assert data["resultados"]["calor_total"]["unidad"] == "kW"
        assert any(f"{r.m_steam_kg_s:.5g}" in t for t in latex)


def test_example_metadata() -> None:
    assert set(HRSG_EXAMPLE_NOTES) == set(HRSG_EXAMPLES)
    assert {i.superheated for i in HRSG_EXAMPLES.values()} == {True, False}
    assert math.isclose(sum(HRSG_EXAMPLES[EX_TG].gas.y), 1.0)
    assert HRSG_EXAMPLES[EX_TG].gas == FlueGas.from_fractions(TYPICAL)
    assert set(HRSG_EXAMPLES_EXCESS_AIR) <= set(HRSG_EXAMPLES)
    for name, excess_air in HRSG_EXAMPLES_EXCESS_AIR.items():
        assert HRSG_EXAMPLES[name].gas == FlueGas.from_fractions(exhaust_composition(excess_air))


def test_gas_turbine_example() -> None:
    """Los números del prototipo del plan de la Fase 3.3 (TESPy da lo mismo al 0,02 %)."""
    r = solve_hrsg(HRSG_EXAMPLES[EX_TG])
    assert r.m_steam_kg_s == pytest.approx(15.385, abs=5e-4)
    assert r.T_after_superheater_K - C == pytest.approx(503.55, abs=0.01)
    assert r.T_stack_K - C == pytest.approx(153.05, abs=0.01)
    assert r.Q_W == pytest.approx(50.18e6, rel=1e-3)
    sat = solve_hrsg(replace(HRSG_EXAMPLES[EX_TG], T_steam_K=None))
    assert sat.m_steam_kg_s == pytest.approx(22.450, abs=5e-4)
    assert sat.T_stack_K - C == pytest.approx(91.05, abs=0.01)


def test_other_steam_option() -> None:
    tg = HRSG_EXAMPLES[EX_TG]
    assert other_steam_option(tg) == replace(tg, T_steam_K=None)
    sat = HRSG_EXAMPLES[EX_SAT]
    assert default_steam_temperature(sat) == pytest.approx(475.0 + C)  # 25 K bajo los gases
    other = other_steam_option(sat)
    assert other is not None and other.T_steam_K == pytest.approx(475.0 + C)
    r, r_other = solve_hrsg(sat), solve_hrsg(other)
    assert r_other.m_steam_kg_s < r.m_steam_kg_s and r_other.T_stack_K > r.T_stack_K
    # hasta 565 °C, el límite de los aceros
    hot = replace(sat, T_gas_in_K=700.0 + C)
    assert default_steam_temperature(hot) == pytest.approx(565.0 + C)
    # gases apenas por encima de la saturación: no hay a qué temperatura sobrecalentar
    cold = replace(sat, T_gas_in_K=solve_hrsg(sat).T_sat_K + 30.0)
    assert default_steam_temperature(cold) is None
    assert other_steam_option(cold) is None
