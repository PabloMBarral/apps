"""Tests de la topología y la numeración del Rankine (sin TESPy) — Fase 3.1b."""

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
