"""Tests de los gráficos del segundo principio (Fase 10.2)."""

from __future__ import annotations

import pytest

from core.balances import second_law as sl
from core.units_system import convert_from_si
from ui import second_law_charts as sc

SYSTEMS = ("SI", "Técnico", "Inglés")


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(sl.MACHINE_EXAMPLES))
def test_limit_figure_puts_the_machine_against_carnot(name: str, system: str) -> None:
    r = sl.solve_machine(sl.MACHINE_EXAMPLES[name].inputs)
    fig = sc.limit_figure(r, system)  # type: ignore[arg-type]
    curve, zone, point = fig.data
    assert zone.fill == "tonexty"  # la zona imposible, arriba de la curva
    T = r.inputs.T_H_K if r.kind == "engine" else r.inputs.T_C_K
    assert point.x[0] == pytest.approx(convert_from_si(T, "temperature", system))  # type: ignore[arg-type]
    assert point.y[0] == pytest.approx(r.performance)
    assert fig.layout.yaxis.range[1] >= r.performance
    color = point.marker.color
    assert (color == "#eb6834") == (r.verdict == "imposible")
    # la curva pasa por el reversible a la temperatura de la máquina
    xs, ys = list(curve.x), list(curve.y)
    x0 = convert_from_si(T, "temperature", system)  # type: ignore[arg-type]
    if xs[0] <= x0 <= xs[-1]:
        k = min(range(len(xs)), key=lambda i: abs(xs[i] - x0))
        assert ys[k] == pytest.approx(r.performance_rev, rel=0.05)


@pytest.mark.parametrize("name", list(sl.HEAT_PATH_EXAMPLES))
def test_heat_exergy_figure(name: str) -> None:
    r = sl.solve_heat_path(sl.HEAT_PATH_EXAMPLES[name].inputs)
    fig = sc.heat_exergy_figure(r, "Técnico")
    bars = fig.data[0]
    kind = "heat_rate" if r.inputs.rates else "energy"
    expected = [convert_from_si(x, kind, "Técnico") for x in r.exergies]
    assert list(bars.y) == pytest.approx(expected)
    assert all(a >= b for a, b in zip(bars.y, bars.y[1:], strict=False))  # baja con T
    # los nombres en renglones cortos y horizontales (a 390 px, girados se mezclaban)
    flat = [label.replace("<br>", " ") for label in bars.x]
    assert all(p in label for p, label in zip(r.inputs.points, flat, strict=True))
    assert fig.layout.xaxis.tickangle == 0
    for label in bars.x:
        for line in label.split("<br>"):
            assert len(line) <= 11 or " " not in line, line


_EDGES = [
    sl.MachineInputs("refrigerator", 300.0, 60.0, W=1e3, performance=0.1),  # T_C < 0,3·T_H
    sl.MachineInputs("heat_pump", 300.0, 60.0, W=1e3, performance=1.1),
    sl.MachineInputs("engine", 300.5, 300.0, Q_H=1e3, performance=1e-4),  # T_H apenas > T_C
]


@pytest.mark.parametrize(
    "inputs",
    [ex.inputs for ex in sl.MACHINE_EXAMPLES.values()] + _EDGES,
)
def test_the_machine_stays_inside_with_its_label(inputs: sl.MachineInputs) -> None:
    r = sl.solve_machine(inputs)
    for system in SYSTEMS:
        fig = sc.limit_figure(r, system)  # type: ignore[arg-type]
        curve, _, point = fig.data
        xs = list(curve.x)
        x0 = point.x[0]
        assert xs[0] <= x0 <= xs[-1]
        # el rótulo va hacia el lado con lugar (a 390 px «esta máquina» se cortaba)
        right_half = x0 - xs[0] > 0.5 * (xs[-1] - xs[0])
        assert point.textposition.endswith("left" if right_half else "right")
