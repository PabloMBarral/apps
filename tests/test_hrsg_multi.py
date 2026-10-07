"""Tests de la HRSG de dos y tres presiones (:mod:`core.cycles.hrsg_multi`) — Fase 3.5.

Referencias: con un nivel, la caldera de la Fase 3.3 (exacto); un cálculo a
mano independiente (PropsSI y brentq) de la cascada; una red de TESPy
(``HeatExchanger`` en serie, cada domo un ``DropletSeparator`` y una ``Pump``
isoentrópica entre niveles), que evalúa cada gas a su presión parcial y
coincide al 0,2 %; y los balances de energía y de exergía, que tienen que
cerrar exactos.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import pytest
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from core.cycles.hrsg import HRSG_EXAMPLES, solve_hrsg
from core.cycles.hrsg_multi import (
    MULTI_HRSG_EXAMPLE_NOTES,
    MULTI_HRSG_EXAMPLES,
    MultiHRSGInputs,
    default_multi_sweep_values,
    from_single,
    hrsg_exergy,
    level_comparison,
    multi_hrsg_notes,
    multi_hrsg_sweep,
    multi_hrsg_to_dict,
    multi_tq_profile,
    solve_multi_hrsg,
)
from core.cycles.hrsg_multi_procedure import multi_hrsg_steps
from core.ideal_gas import FlueGas, exhaust_composition

C = 273.15
NAMES = list(MULTI_HRSG_EXAMPLES)
EX_2P, EX_3P, EX_MODERN, EX_MODERN_2P = NAMES
TWO = MULTI_HRSG_EXAMPLES[EX_2P]
THREE = MULTI_HRSG_EXAMPLES[EX_3P]
COOLPROP = {"N2": "Nitrogen", "O2": "Oxygen", "CO2": "CarbonDioxide", "H2O": "Water", "Ar": "Argon"}


def _hand(inputs: MultiHRSGInputs) -> dict[str, list[float] | float]:
    """La cascada a mano, independiente del módulo (PropsSI y brentq)."""
    w = inputs.gas.mass_fractions

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

    m_g = inputs.m_gas_kg_s
    levels = inputs.levels
    n = len(levels)
    h = h_g(inputs.T_gas_in_K)
    m_up = 0.0
    flows: list[float] = []
    for i, lv in enumerate(levels):
        p = lv.p_Pa
        T_sat = PropsSI("T", "P", p, "Q", 0, "Water")
        h_f = PropsSI("H", "P", p, "Q", 0, "Water")
        h_eco = PropsSI("H", "T", T_sat - lv.approach_K, "P", p, "Water")
        top = (
            PropsSI("H", "P", p, "Q", 1, "Water")
            if lv.T_steam_K is None
            else PropsSI("H", "T", lv.T_steam_K, "P", p, "Water")
        )
        if i == n - 1:
            h_in = PropsSI("H", "T", inputs.T_feedwater_K, "P", p, "Water")
        else:
            s_f = PropsSI("S", "P", levels[i + 1].p_Pa, "Q", 0, "Water")
            h_in = PropsSI("H", "P", p, "S", s_f, "Water")
        h_p = h_g(T_sat + lv.pinch_K)
        m_s = (m_g * (h - h_p) - m_up * (h_f - h_eco)) / (top - h_eco)
        flows.append(m_s)
        m_up += m_s
        h = h_p - m_up * (h_eco - h_in) / m_g
    T_stack = brentq(lambda T: h_g(T) - h, 274.0, 1500.0)
    return {"m": flows, "T_stack": T_stack}


def _tespy(inputs: MultiHRSGInputs, m0: float) -> dict[str, list[float] | float]:
    """La misma cascada con TESPy: gases por todas las secciones; agua de baja a alta."""
    from tespy.components import DropletSeparator, HeatExchanger, Pump, Sink, Source
    from tespy.connections import Connection

    from core.cycles.tespy_utils import new_network, solve

    levels = inputs.levels
    n = len(levels)
    nw = new_network()
    hx: dict[str, HeatExchanger] = {}
    order: list[str] = []
    for i, lv in enumerate(levels):
        for kind in (("SH",) if lv.T_steam_K else ()) + ("EV", "ECO"):
            hx[f"{kind}{i}"] = HeatExchanger(f"{kind}{i}")
            order.append(f"{kind}{i}")
    conns: list[Connection] = []
    gas_in: dict[str, Connection] = {}
    prev, port = Source("gases"), "out1"
    for name in order:
        gas_in[name] = Connection(prev, port, hx[name], "in1", label=f"g_{name}")
        conns.append(gas_in[name])
        prev, port = hx[name], "out1"
    stack = Connection(prev, port, Sink("chimenea"), "in1", label="g_stack")
    conns.append(stack)
    water: dict[str, Connection] = {}
    src, sport = Source("agua de alimentación"), "out1"
    for i in reversed(range(n)):
        lv = levels[i]
        water[f"in{i}"] = Connection(src, sport, hx[f"ECO{i}"], "in2", label=f"w_in{i}")
        water[f"eco{i}"] = Connection(hx[f"ECO{i}"], "out2", hx[f"EV{i}"], "in2", label=f"w_eco{i}")
        conns += [water[f"in{i}"], water[f"eco{i}"]]
        if i == 0:
            target, tport = (hx["SH0"], "in2") if lv.T_steam_K else (Sink("vapor 0"), "in1")
            water["ev0"] = Connection(hx["EV0"], "out2", target, tport, label="w_ev0")
            conns.append(water["ev0"])
        else:
            drum = DropletSeparator(f"domo {i}")
            conns.append(Connection(hx[f"EV{i}"], "out2", drum, "in1", label=f"w_ev{i}"))
            target, tport = (hx[f"SH{i}"], "in2") if lv.T_steam_K else (Sink(f"vapor {i}"), "in1")
            water[f"vap{i}"] = Connection(drum, "out2", target, tport, label=f"w_vap{i}")
            pump = Pump(f"bomba {i}")
            pump.set_attr(eta_s=1.0)
            conns += [water[f"vap{i}"], Connection(drum, "out1", pump, "in1", label=f"w_liq{i}")]
            src, sport = pump, "out1"
        if lv.T_steam_K:
            water[f"sh{i}"] = Connection(
                hx[f"SH{i}"], "out2", Sink(f"vapor sobrecalentado {i}"), "in1", label=f"w_sh{i}"
            )
            conns.append(water[f"sh{i}"])
    nw.add_conns(*conns)
    for h in hx.values():
        h.set_attr(pr1=1, pr2=1)
    fluid = {s: w for s, w in inputs.gas.mass_fractions.items() if w > 0}
    gas_in[order[0]].set_attr(
        T=inputs.T_gas_in_K, p=inputs.gas.p_Pa, m=inputs.m_gas_kg_s, fluid=fluid
    )
    water[f"in{n - 1}"].set_attr(
        T=inputs.T_feedwater_K, p=levels[-1].p_Pa, fluid={"water": 1}, m0=m0
    )
    for i, lv in enumerate(levels):
        T_sat = PropsSI("T", "P", lv.p_Pa, "Q", 0, "Water")
        water[f"eco{i}"].set_attr(td_bubble=lv.approach_K)
        gas_in[f"ECO{i}"].set_attr(T=T_sat + lv.pinch_K)
        if i < n - 1:
            water[f"in{i}"].set_attr(p=lv.p_Pa)
        if lv.T_steam_K:
            water[f"sh{i}"].set_attr(T=lv.T_steam_K)
    water["ev0"].set_attr(x=1)
    solve(nw, what="la HRSG de varias presiones")
    flows = [water["ev0"].m.val_SI] + [water[f"vap{i}"].m.val_SI for i in range(1, n)]
    return {"m": flows, "T_stack": stack.T.val_SI}


# ---------------------------------------------------------------------
# Un nivel = la caldera de la Fase 3.3
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n, i in HRSG_EXAMPLES.items() if not i.by_stack])
def test_one_level_is_the_single_pressure_boiler(name: str) -> None:
    single = solve_hrsg(HRSG_EXAMPLES[name])
    multi = solve_multi_hrsg(from_single(HRSG_EXAMPLES[name]))
    assert multi.m_steam_kg_s == pytest.approx(single.m_steam_kg_s, rel=1e-12)
    assert multi.T_stack_K == pytest.approx(single.T_stack_K, rel=1e-12)
    assert [s.Q_W for s in multi.sections] == pytest.approx([s.Q_W for s in single.sections])
    assert [s.name for s in multi.sections] == [s.name for s in single.sections]
    assert multi.recovery == pytest.approx(single.recovery, rel=1e-12)
    assert multi.W_pumps_W == 0.0


def test_from_single_rejects_the_stack_design() -> None:
    by_stack = replace(HRSG_EXAMPLES[next(iter(HRSG_EXAMPLES))], T_stack_K=430.0)
    with pytest.raises(ValueError, match="por pinch"):
        from_single(by_stack)


# ---------------------------------------------------------------------
# Cálculo: a mano, TESPy y balances
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_cascade_matches_the_hand_calculation(name: str) -> None:
    inputs = MULTI_HRSG_EXAMPLES[name]
    result = solve_multi_hrsg(inputs)
    ref = _hand(inputs)
    assert [lv.m_steam_kg_s for lv in result.levels] == pytest.approx(ref["m"], rel=1e-7)
    assert result.T_stack_K == pytest.approx(ref["T_stack"], abs=1e-4)


@pytest.mark.parametrize(
    "inputs",
    [
        TWO,
        replace(TWO, levels=(TWO.levels[0], replace(TWO.levels[1], T_steam_K=None))),
        THREE,
        replace(
            THREE,
            levels=(THREE.levels[0], *(replace(lv, T_steam_K=None) for lv in THREE.levels[1:])),
        ),
    ],
    ids=["2P", "2P baja saturada", "3P", "3P media y baja saturadas"],
)
def test_tespy_gives_the_same_boiler(inputs: MultiHRSGInputs) -> None:
    result = solve_multi_hrsg(inputs)
    ref = _tespy(inputs, m0=result.m_steam_kg_s)
    assert [lv.m_steam_kg_s for lv in result.levels] == pytest.approx(ref["m"], rel=2e-3)
    assert result.T_stack_K == pytest.approx(ref["T_stack"], abs=0.1)


@pytest.mark.parametrize("name", NAMES)
def test_energy_balances(name: str) -> None:
    result = solve_multi_hrsg(MULTI_HRSG_EXAMPLES[name])
    m_g = result.inputs.m_gas_kg_s
    # Cada sección: lo que ceden los gases lo recibe el agua.
    for section in result.sections:
        gas = result.inputs.gas
        q_gas = m_g * (gas.h(section.T_gas_in_K) - gas.h(section.T_gas_out_K))
        assert section.Q_W == pytest.approx(q_gas, rel=1e-6)
    assert sum(s.Q_W for s in result.sections) == pytest.approx(result.Q_W, rel=1e-9)
    # Del agua de alimentación a las salidas de vapor: Q̇ + bombas.
    h_fw = result.levels[-1].water[0].h_J_per_kg
    gain = sum(lv.m_steam_kg_s * (lv.h_steam_J_per_kg - h_fw) for lv in result.levels)
    assert gain == pytest.approx(result.Q_W + result.W_pumps_W, rel=1e-9)
    # El agua que pasa por cada economizador es la de ese nivel y la de los de más presión.
    total = 0.0
    for lv in result.levels:
        total += lv.m_steam_kg_s
        assert lv.m_water_kg_s == pytest.approx(total)
    # Pinch de cada nivel = el dato, y es el mínimo ΔT de su evaporador.
    for lv in result.levels:
        assert lv.pinch_K == pytest.approx(lv.level.pinch_K)


@pytest.mark.parametrize("name", NAMES)
def test_exergy_balance_closes(name: str) -> None:
    result = solve_multi_hrsg(MULTI_HRSG_EXAMPLES[name])
    x = hrsg_exergy(result)
    assert x.X_gas_in_W == pytest.approx(x.X_water_W + x.X_destroyed_W + x.X_stack_W, rel=1e-9)
    assert x.X_given_W == pytest.approx(x.X_water_W + x.X_destroyed_W, rel=1e-9)
    assert all(d > 0.0 for _, d in x.destroyed_W)  # cada sección genera entropía
    assert [n for n, _ in x.destroyed_W] == [s.name for s in result.sections]
    assert 0.75 < x.efficiency < 0.95


def test_more_levels_cool_the_stack_and_destroy_less_exergy() -> None:
    for inputs in (TWO, THREE, MULTI_HRSG_EXAMPLES[EX_MODERN]):
        rows = level_comparison(inputs)
        results = [r for _, r, _ in rows]
        assert all(r is not None for r in results)
        one, two = results[0], results[1]
        assert one is not None and two is not None
        assert two.T_stack_K < one.T_stack_K - 40.0
        assert two.Q_W > one.Q_W
        assert hrsg_exergy(two).X_destroyed_W < hrsg_exergy(one).X_destroyed_W
        assert hrsg_exergy(two).efficiency > hrsg_exergy(one).efficiency
    three = level_comparison(THREE)[-1][1]
    two = level_comparison(THREE)[1][1]
    assert three is not None and two is not None
    assert hrsg_exergy(three).efficiency > hrsg_exergy(two).efficiency


def test_typical_numbers() -> None:
    two = solve_multi_hrsg(TWO)
    assert [lv.m_steam_kg_s for lv in two.levels] == pytest.approx([15.265, 2.329], abs=2e-3)
    assert two.T_stack_K - C == pytest.approx(103.2, abs=0.1)
    assert two.recovery == pytest.approx(0.857, abs=1e-3)
    assert hrsg_exergy(two).efficiency == pytest.approx(0.835, abs=1e-3)
    three = solve_multi_hrsg(THREE)
    assert three.T_stack_K - C == pytest.approx(97.4, abs=0.1)
    assert [label for label, _, _ in three.gas_points] == list("abcdefhij")


def test_profile_is_a_sawtooth_and_its_minimum_is_the_smallest_pinch() -> None:
    result = solve_multi_hrsg(THREE)
    profile = multi_tq_profile(result)
    kinds = [(s.level, s.kind) for s in profile.segments]
    assert kinds[0] == (2, "economizador")  # desde la chimenea: el economizador de baja
    assert kinds[-1] == (0, "sobrecalentador")
    assert len(profile.boundaries_W) == len(profile.segments) - 1
    assert profile.min_dT_K == pytest.approx(min(lv.level.pinch_K for lv in result.levels))
    # El economizador de media arranca más frío que el vapor de baja que lo precede.
    eco_m = next(s for s in profile.segments if s.level == 1 and s.kind == "economizador")
    ev_b = next(s for s in profile.segments if s.level == 2 and s.kind == "evaporador")
    assert eco_m.T_K[0] == pytest.approx(ev_b.T_K[-1], abs=1.0)  # el domo de baja, bombeado
    assert profile.Q_gas_W[-1] == pytest.approx(result.Q_W, rel=1e-9)


# ---------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------

HP, LP = TWO.levels


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (replace(TWO, levels=(LP, HP)), "tiene que ser menor"),
        (replace(TWO, levels=(HP, replace(LP, p_Pa=76e5, T_steam_K=None))), "muy cerca"),
        (replace(TWO, levels=()), "entre 1 y 3"),
        (replace(TWO, levels=(HP, HP, HP, LP)), "entre 1 y 3"),
        (replace(TWO, T_feedwater_K=160 + C), "más fría"),
        (replace(TWO, levels=(HP, replace(LP, pinch_K=0.0))), "Nivel de baja: el pinch"),
        (replace(TWO, levels=(replace(HP, approach_K=-1.0), LP)), "Nivel de alta: el approach"),
        (replace(TWO, levels=(replace(HP, p_Pa=230e5), LP)), "supera la crítica"),
        (replace(TWO, levels=(HP, replace(LP, T_steam_K=150 + C))), "más caliente que la sat"),
        (replace(TWO, levels=(HP, replace(LP, T_steam_K=250 + C))), "le llegan"),
        (replace(TWO, levels=(replace(HP, pinch_K=310.0), LP)), "no alcanzan"),
        (
            replace(
                THREE,
                levels=(
                    THREE.levels[0],
                    replace(THREE.levels[1], T_steam_K=300 + C),
                    THREE.levels[2],
                ),
            ),
            "Nivel de media: el vapor",
        ),
        (replace(TWO, m_gas_kg_s=0.0), "caudal de gases"),
        (
            replace(
                TWO,
                gas=FlueGas.from_fractions(exhaust_composition(1.05)),  # rocío a 58 °C
                T_feedwater_K=15 + C,
                levels=(HP, replace(LP, p_Pa=1.5e5, T_steam_K=None, pinch_K=3.0)),
            ),
            "punto de rocío",
        ),
    ],
)
def test_invalid_data_are_explained(inputs: MultiHRSGInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_multi_hrsg(inputs)


# ---------------------------------------------------------------------
# Notas, barridos, export y procedimiento
# ---------------------------------------------------------------------


def test_notes() -> None:
    notes = multi_hrsg_notes(solve_multi_hrsg(TWO))
    assert any(n.startswith("El nivel de baja produce") for n in notes)
    tight = replace(
        THREE,
        levels=(THREE.levels[0], replace(THREE.levels[1], T_steam_K=250 + C), THREE.levels[2]),
    )
    assert any("sobrecalentador de media" in n for n in multi_hrsg_notes(solve_multi_hrsg(tight)))
    assert set(MULTI_HRSG_EXAMPLE_NOTES) == set(MULTI_HRSG_EXAMPLES)


def test_sweeps() -> None:
    pinch = multi_hrsg_sweep(TWO, "pinch", default_multi_sweep_values(TWO, "pinch", n=5))
    assert len(pinch) == 5
    stacks = [p.T_stack_K for p in pinch]
    assert stacks == sorted(stacks)  # más pinch, chimenea más caliente
    low = multi_hrsg_sweep(THREE, "p_low", default_multi_sweep_values(THREE, "p_low", n=6))
    assert len(low) == 6
    assert [p.T_stack_K for p in low] == sorted(p.T_stack_K for p in low)
    high = multi_hrsg_sweep(TWO, "p_high", default_multi_sweep_values(TWO, "p_high", n=5))
    assert [p.exergy_efficiency for p in high] == sorted(p.exergy_efficiency for p in high)
    feed = multi_hrsg_sweep(TWO, "T_feedwater", default_multi_sweep_values(TWO, "T_feedwater", 4))
    assert len(feed) == 4 and len(feed[0].m_steam_kg_s) == 2


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
@pytest.mark.parametrize("name", NAMES)
def test_export_and_procedure(name: str, system: str) -> None:
    result = solve_multi_hrsg(MULTI_HRSG_EXAMPLES[name])
    data = multi_hrsg_to_dict(result, system)  # type: ignore[arg-type]
    json.dumps(data, ensure_ascii=False)
    assert len(data["datos"]["niveles"]) == len(result.levels)
    assert data["resultados"]["exergia"]["rendimiento_exergetico"] == pytest.approx(
        hrsg_exergy(result).efficiency
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        steps = multi_hrsg_steps(result, system)  # type: ignore[arg-type]
    titles = [s.title for s in steps]
    assert titles[0] == "Composición de los gases"
    assert sum(t.endswith("economizador") for t in titles) == len(result.levels)
    assert "Exergía: cuánto vale el calor recuperado" in titles
    latex = [t for s in steps for t in s.latex]
    assert all(t.count("{") == t.count("}") for t in latex)
    assert not any("- -" in t or "+ -" in t for t in latex)
