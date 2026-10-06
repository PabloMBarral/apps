"""Tests de la topología y la numeración del Rankine (sin TESPy) — Fases 3.1b y 3.1c."""

from __future__ import annotations

import itertools
from collections import Counter

import pytest

from core.cycles.rankine_layout import (
    MAX_HEATERS,
    FeedwaterHeater,
    PlantLayout,
    heater_names,
    plant_layout,
    port_flows,
)

OPEN = "open"
CLOSED = "closed"


def _ports(layout: PlantLayout, kind: str) -> list[dict[str, list[int]]]:
    """Conexiones de cada componente de ese tipo, con números de estado."""
    out = []
    for c in layout.components:
        if c.kind == kind:
            ports: dict[str, list[int]] = {}
            for p in c.ports:
                ports.setdefault(p.role, []).append(p.state + 1)
            out.append(ports)
    return out


def test_simple_cycle_keeps_the_numbering_of_0_11() -> None:
    lay = plant_layout()
    assert lay.labeled() == (
        "1 (entrada a la bomba)",
        "2 (salida de la bomba)",
        "3 (entrada a la turbina)",
        "4 (salida de la turbina)",
    )
    assert [c.label for c in lay.components] == ["bomba", "caldera", "turbina", "condensador"]


def test_reheat_cycle_keeps_the_numbering_of_0_11() -> None:
    lay = plant_layout(reheat_p_Pa=4.0e6)
    assert lay.labeled() == (
        "1 (entrada a la bomba)",
        "2 (salida de la bomba)",
        "3 (entrada a la turbina de alta)",
        "4 (salida de la turbina de alta)",
        "5 (entrada a la turbina de baja)",
        "6 (salida de la turbina de baja)",
    )
    assert (lay.reheat_in, lay.reheat_out, lay.exhaust) == (3, 4, 5)
    labels = [c.label for c in lay.components]
    assert labels == [
        "bomba",
        "caldera",
        "turbina de alta",
        "recalentador",
        "turbina de baja",
        "condensador",
    ]


def test_cengel_10_5_numbering() -> None:
    # Cengel ej. 10-5: 1 condensador, 2 bomba I, 3 calentador abierto, 4 bomba II,
    # 5 entrada a la turbina, 6 extracción, 7 salida de la turbina.
    lay = plant_layout(heaters=[FeedwaterHeater(1.2e6)])
    assert lay.n_states == 7
    assert _ports(lay, "pump") == [{"in": [1], "out": [2]}, {"in": [3], "out": [4]}]
    assert _ports(lay, "open_heater") == [{"fw_in": [2], "bleed": [6], "out": [3]}]
    assert _ports(lay, "boiler") == [{"in": [4], "out": [5]}]
    assert _ports(lay, "turbine") == [{"in": [5], "out": [6]}, {"in": [6], "out": [7]}]
    assert _ports(lay, "condenser") == [{"in": [7], "out": [1]}]
    assert lay.extraction_states == (5,)
    assert lay.labels[2] == "salida del calentador abierto"


def test_cengel_10_6_numbering() -> None:
    # Cengel ej. 10-6: cerrado a 4 MPa (= recalentamiento) con el drenaje bombeado
    # hacia adelante y abierto a 0,5 MPa; estados 1 a 13 como en el libro.
    lay = plant_layout(
        heaters=[FeedwaterHeater(0.5e6), FeedwaterHeater(4.0e6, CLOSED)],
        reheat_p_Pa=4.0e6,
        drain_forward=True,
    )
    assert lay.n_states == 13
    assert _ports(lay, "pump") == [
        {"in": [1], "out": [2]},
        {"in": [3], "out": [4]},
        {"in": [6], "out": [7]},
    ]
    assert _ports(lay, "open_heater") == [{"fw_in": [2], "bleed": [12], "out": [3]}]
    assert _ports(lay, "closed_heater") == [
        {"fw_in": [4], "fw_out": [5], "bleed": [10], "drain_out": [6]}
    ]
    assert _ports(lay, "mixer") == [{"fw_in": [5], "drain_in": [7], "out": [8]}]
    assert _ports(lay, "boiler") == [{"in": [8], "out": [9]}]
    assert _ports(lay, "reheater") == [{"in": [10], "out": [11]}]
    assert _ports(lay, "turbine") == [
        {"in": [9], "out": [10]},
        {"in": [11], "out": [12]},
        {"in": [12], "out": [13]},
    ]
    assert (lay.turbine_inlet, lay.reheat_in, lay.reheat_out, lay.exhaust) == (8, 9, 10, 12)
    assert lay.extraction_states == (11, 9)  # la extracción a 4 MPa comparte el estado 10


def test_closed_heater_draining_to_the_condenser() -> None:
    lay = plant_layout(heaters=[FeedwaterHeater(1.2e6, CLOSED)])
    assert _ports(lay, "closed_heater") == [
        {"fw_in": [2], "fw_out": [3], "bleed": [5], "drain_out": [7]}
    ]
    assert _ports(lay, "valve") == [{"in": [7], "out": [8]}]
    assert _ports(lay, "condenser") == [{"in": [6], "drain_in": [8], "out": [1]}]
    assert lay.exhaust == 5  # el escape sigue siendo el estado 6, no el último


def test_closed_heater_cascading_into_the_open_one() -> None:
    lay = plant_layout(heaters=[FeedwaterHeater(0.5e6), FeedwaterHeater(4.0e6, CLOSED)])
    (open_heater,) = _ports(lay, "open_heater")
    (closed,) = _ports(lay, "closed_heater")
    (valve,) = _ports(lay, "valve")
    assert valve["in"] == closed["drain_out"]
    assert open_heater["drain_in"] == valve["out"]
    assert "drain_in" not in _ports(lay, "condenser")[0]


def test_two_closed_heaters_cascade_down_to_the_condenser() -> None:
    lay = plant_layout(heaters=[FeedwaterHeater(2.0e5, CLOSED), FeedwaterHeater(1.0e6, CLOSED)])
    low, high = _ports(lay, "closed_heater")
    valve_high, valve_low = _ports(lay, "valve")  # de mayor a menor presión
    assert valve_high["in"] == high["drain_out"] and low["drain_in"] == valve_high["out"]
    assert valve_low["in"] == low["drain_out"]
    assert _ports(lay, "condenser")[0]["drain_in"] == valve_low["out"]
    assert high["fw_in"] == low["fw_out"]  # el agua pasa de uno al otro, sin bomba


def test_extraction_above_the_reheat_pressure_splits_the_high_pressure_turbine() -> None:
    lay = plant_layout(heaters=[FeedwaterHeater(8.0e6, CLOSED)], reheat_p_Pa=4.0e6)
    turbines = [c.label for c in lay.of_kind("turbine")]
    assert turbines == ["turbina de alta (tramo 1)", "turbina de alta (tramo 2)", "turbina de baja"]


def test_heater_names() -> None:
    assert heater_names([FeedwaterHeater(1e5)]) == ("calentador abierto",)
    assert heater_names([FeedwaterHeater(1e5), FeedwaterHeater(1e6, CLOSED)]) == (
        "calentador abierto de baja",
        "calentador cerrado de alta",
    )
    three = [FeedwaterHeater(1e5, CLOSED), FeedwaterHeater(5e5), FeedwaterHeater(4e6, CLOSED)]
    assert heater_names(three)[1] == "calentador abierto intermedio"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"heaters": [FeedwaterHeater(5e5), FeedwaterHeater(5e5, CLOSED)]}, "distintas"),
        ({"heaters": [FeedwaterHeater(5e6), FeedwaterHeater(5e5)]}, "de menor a mayor"),
        ({"heaters": [FeedwaterHeater(1e5 * k) for k in range(1, 5)]}, "hasta 3"),
        (
            {
                "heaters": [FeedwaterHeater(5e5, CLOSED), FeedwaterHeater(4e6)],
                "drain_forward": True,
            },
            "Solo un calentador cerrado",
        ),
    ],
)
def test_layout_errors_are_explained(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plant_layout(**kwargs)


def _configurations():
    pressures = (0.7e5, 5.0e5, 40.0e5)
    for n in range(MAX_HEATERS + 1):
        for kinds in itertools.product((OPEN, CLOSED), repeat=n):
            heaters = [
                FeedwaterHeater(p, k) for p, k in zip(pressures[3 - n :], kinds, strict=True)
            ]
            for reheat in (None, 40.0e5, 10.0e5):
                for forward in (False, True):
                    if forward and (not n or kinds[-1] != CLOSED):
                        continue
                    yield heaters, reheat, forward


@pytest.mark.parametrize(("heaters", "reheat", "forward"), list(_configurations()))
def test_every_state_has_one_producer_and_the_flows_close(heaters, reheat, forward) -> None:
    lay = plant_layout(heaters=heaters, reheat_p_Pa=reheat, drain_forward=forward)
    produced = Counter(p.state for c in lay.components for p in c.outlets)
    consumed = Counter(p.state for c in lay.components for p in c.inlets)
    assert set(produced) == set(range(lay.n_states))
    assert all(v == 1 for v in produced.values())
    for state in range(lay.n_states):
        # Las extracciones van a dos lugares (sigue la turbina y sale la extracción).
        expected = 2 if state in lay.extraction_states else 1
        assert consumed[state] == expected, (lay.labeled()[state], consumed[state])
    assert len(lay.of_kind("open_heater")) + len(lay.of_kind("closed_heater")) == len(heaters)
    assert lay.labels[0] in ("entrada a la bomba", "salida del condensador")


# ---------------------------------------------------------------------
# Fase 3.1c: cañerías con pérdidas, recuperador y drenajes bombeados
# hacia adelante desde cualquier cerrado
# ---------------------------------------------------------------------


def _mass_balances_close(lay: PlantLayout) -> None:
    """Con los caudales simbólicos, lo que entra a cada componente es lo que sale y
    cada extracción reparte el caudal del estado del que sale."""
    flows = port_flows(lay)

    def total(items: list[dict]) -> dict:
        out: dict = {}
        for flow in items:
            for key, coef in flow.items():
                out[key] = out.get(key, 0) + coef
        return {k: v for k, v in out.items() if v}

    for ci, comp in enumerate(lay.components):
        ins = [flows[(ci, pi)] for pi, p in enumerate(comp.ports) if p in comp.inlets]
        outs = [flows[(ci, pi)] for pi, p in enumerate(comp.ports) if p in comp.outlets]
        if comp.kind in ("closed_heater", "recuperator"):  # dos corrientes que no se mezclan
            hot = [flows[(ci, pi)] for pi, p in enumerate(comp.ports) if p.role in ("in", "out")]
            cold = [
                flows[(ci, pi)] for pi, p in enumerate(comp.ports) if p.role in ("fw_in", "fw_out")
            ]
            assert all(f == cold[0] for f in cold) and all(f == hot[0] for f in hot)
            continue
        assert total(ins) == total(outs), comp.label
    produced = {
        p.state: flows[(ci, pi)]
        for ci, c in enumerate(lay.components)
        for pi, p in enumerate(c.ports)
        if p in c.outlets
    }
    consumed: dict[int, list[dict]] = {}
    for ci, c in enumerate(lay.components):
        for pi, p in enumerate(c.ports):
            if p in c.inlets:
                consumed.setdefault(p.state, []).append(flows[(ci, pi)])
    for state, parts in consumed.items():
        assert total(parts) == produced[state], lay.labeled()[state]


def test_cengel_10_2_numbering_with_pipes() -> None:
    # Cengel ej. 10-2 (ciclo real): 1 entrada a la bomba, 2 salida de la bomba,
    # 3 entrada a la caldera, 4 salida de la caldera, 5 entrada a la turbina, 6 salida.
    lay = plant_layout(feed_pipe=True, steam_pipe=True)
    assert lay.labeled() == (
        "1 (entrada a la bomba)",
        "2 (salida de la bomba)",
        "3 (entrada a la caldera)",
        "4 (salida de la caldera)",
        "5 (entrada a la turbina)",
        "6 (salida de la turbina)",
    )
    assert [c.label for c in lay.components] == [
        "bomba",
        "cañería de alimentación",
        "caldera",
        "cañería de vapor",
        "turbina",
        "condensador",
    ]
    assert _ports(lay, "pipe") == [{"in": [2], "out": [3]}, {"in": [4], "out": [5]}]
    assert _ports(lay, "boiler") == [{"in": [3], "out": [4]}]
    assert (lay.boiler_in, lay.boiler_out, lay.turbine_inlet, lay.exhaust) == (2, 3, 4, 5)
    assert lay.condenser_in == lay.exhaust
    _mass_balances_close(lay)


def test_without_pipes_the_boiler_feeds_the_turbine_directly() -> None:
    lay = plant_layout(heaters=[FeedwaterHeater(1.2e6)])
    assert (lay.boiler_in, lay.boiler_out) == (3, 4) and lay.turbine_inlet == 4
    assert not lay.of_kind("pipe")


def test_feed_pipe_after_the_regenerative_line() -> None:
    lay = plant_layout(heaters=[FeedwaterHeater(1.2e6)], feed_pipe=True)
    assert lay.labels[4] == "entrada a la caldera"
    assert _ports(lay, "pipe") == [{"in": [4], "out": [5]}]
    assert _ports(lay, "boiler") == [{"in": [5], "out": [6]}]
    assert lay.extraction_states == (6,)
    _mass_balances_close(lay)


def test_orc_with_recuperator_numbering() -> None:
    # 1 entrada a la bomba, 2 salida de la bomba, 3 salida fría del recuperador,
    # 4 entrada a la turbina, 5 salida de la turbina, 6 salida caliente del recuperador.
    lay = plant_layout(recuperator=True, boiler="evaporador")
    assert lay.labeled() == (
        "1 (entrada a la bomba)",
        "2 (salida de la bomba)",
        "3 (salida del recuperador (líquido))",
        "4 (entrada a la turbina)",
        "5 (salida de la turbina)",
        "6 (salida del recuperador (vapor))",
    )
    assert _ports(lay, "recuperator") == [{"fw_in": [2], "fw_out": [3], "in": [5], "out": [6]}]
    assert _ports(lay, "boiler") == [{"in": [3], "out": [4]}]
    assert _ports(lay, "condenser") == [{"in": [6], "out": [1]}]
    assert (lay.exhaust, lay.condenser_in) == (4, 5)
    assert lay.of_kind("boiler")[0].label == "evaporador"
    _mass_balances_close(lay)


def test_evaporator_name_reaches_the_pipe_labels() -> None:
    lay = plant_layout(feed_pipe=True, steam_pipe=True, boiler="evaporador")
    assert lay.labels[2:4] == ("entrada al evaporador", "salida del evaporador")


def test_intermediate_closed_heater_pumps_its_drain_forward() -> None:
    # Cerrado de baja con el drenaje bombeado a la línea, desaireador (abierto) y cerrado
    # de alta en cascada hacia el abierto.
    heaters = [
        FeedwaterHeater(2.0e5, CLOSED, drain_forward=True),
        FeedwaterHeater(12.0e5),
        FeedwaterHeater(40.0e5, CLOSED),
    ]
    lay = plant_layout(heaters=heaters)
    assert lay.forward_drains == (0,) and not lay.drain_forward
    assert _ports(lay, "closed_heater") == [
        {"fw_in": [2], "fw_out": [3], "bleed": [13], "drain_out": [4]},
        {"fw_in": [8], "fw_out": [9], "bleed": [11], "drain_out": [15]},
    ]
    assert _ports(lay, "mixer") == [{"fw_in": [3], "drain_in": [5], "out": [6]}]
    assert _ports(lay, "open_heater") == [
        {"fw_in": [6], "bleed": [12], "drain_in": [16], "out": [7]}
    ]
    assert [c.label for c in lay.components if c.kind == "pump"] == [
        "bomba I",
        "bomba del drenaje",
        "bomba II",
    ]
    _mass_balances_close(lay)
    flows = port_flows(lay)
    (ci,) = [i for i, c in enumerate(lay.components) if c.kind == "mixer"]
    # Por la línea venía 1 − y₁ − y₂ − y₃ (todo lo que pasó por el condensador) y la
    # mezcla le devuelve y₁.
    assert flows[(ci, 0)] == {None: 1, 0: -1, 1: -1, 2: -1}
    assert flows[(ci, 2)] == {None: 1, 1: -1, 2: -1}


def test_two_pumped_drains_name_each_pump_and_mixer() -> None:
    heaters = [
        FeedwaterHeater(2.0e5, CLOSED, drain_forward=True),
        FeedwaterHeater(20.0e5, CLOSED, drain_forward=True),
    ]
    lay = plant_layout(heaters=heaters)
    assert lay.forward_drains == (0, 1) and lay.drain_forward
    labels = [c.label for c in lay.components]
    assert "bomba del drenaje del calentador cerrado de baja" in labels
    assert "cámara de mezcla del calentador cerrado de alta" in labels
    assert len(labels) == len(set(labels))  # TESPy exige rótulos únicos
    assert not lay.of_kind("valve")
    _mass_balances_close(lay)


def test_legacy_flag_and_per_heater_flag_give_the_same_plant() -> None:
    legacy = plant_layout(
        heaters=[FeedwaterHeater(0.5e6), FeedwaterHeater(4.0e6, CLOSED)], drain_forward=True
    )
    per_heater = plant_layout(
        heaters=[FeedwaterHeater(0.5e6), FeedwaterHeater(4.0e6, CLOSED, drain_forward=True)]
    )
    assert legacy.labels == per_heater.labels
    assert [c.ports for c in legacy.components] == [c.ports for c in per_heater.components]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"heaters": [FeedwaterHeater(5e5, drain_forward=True)]}, "es abierto"),
        ({"heaters": [FeedwaterHeater(5e5)], "recuperator": True}, "sin calentadores"),
    ],
)
def test_new_layout_errors_are_explained(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plant_layout(**kwargs)


def _real_configurations():
    pressures = (0.7e5, 5.0e5, 40.0e5)
    for n in range(MAX_HEATERS + 1):
        for kinds in itertools.product((OPEN, CLOSED), repeat=n):
            closed = [i for i, k in enumerate(kinds) if k == CLOSED]
            for forward in itertools.chain.from_iterable(
                itertools.combinations(closed, r) for r in range(len(closed) + 1)
            ):
                heaters = [
                    FeedwaterHeater(p, k, drain_forward=i in forward)
                    for i, (p, k) in enumerate(zip(pressures[3 - n :], kinds, strict=True))
                ]
                for reheat in (None, 10.0e5):
                    for pipes in ((False, False), (True, True)):
                        yield heaters, reheat, pipes, False
            if n == 0:
                yield [], None, (True, True), True
                yield [], 10.0e5, (False, False), True


@pytest.mark.parametrize(
    ("heaters", "reheat", "pipes", "recuperator"), list(_real_configurations())
)
def test_every_real_plant_numbers_each_state_once_and_balances_mass(
    heaters, reheat, pipes, recuperator
) -> None:
    lay = plant_layout(
        heaters=heaters,
        reheat_p_Pa=reheat,
        feed_pipe=pipes[0],
        steam_pipe=pipes[1],
        recuperator=recuperator,
    )
    produced = Counter(p.state for c in lay.components for p in c.outlets)
    consumed = Counter(p.state for c in lay.components for p in c.inlets)
    assert set(produced) == set(range(lay.n_states))
    assert all(v == 1 for v in produced.values())
    for state in range(lay.n_states):
        expected = 2 if state in lay.extraction_states else 1
        assert consumed[state] == expected, (lay.labeled()[state], consumed[state])
    labels = [c.label for c in lay.components]
    assert len(labels) == len(set(labels))
    _mass_balances_close(lay)
