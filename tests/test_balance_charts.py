"""Tests de los gráficos de los balances (Fase 10.1): rótulos, ejes y cascadas."""

from __future__ import annotations

import pytest

from core.balances import closed as cl
from core.balances import steady_flow as sf
from core.balances import transient as tr
from core.balances.common import upper_first
from core.balances.substance import ideal_gas_substance, state
from ui import balance_charts as bc

SYSTEMS = ("SI", "Técnico", "Inglés")


def _value(text: str) -> float:
    """El número de una marca: «0,01», «1000», «10<sup>5</sup>» o «2·10<sup>5</sup>»."""
    text = text.replace("−", "-")
    if "<sup>" not in text:
        return float(text.replace(",", "."))
    mant, _, exp = text.partition("10<sup>")
    m = float(mant.removesuffix("·").replace(",", ".")) if mant else 1.0
    return m * 10.0 ** int(exp.removesuffix("</sup>"))


def test_upper_first_keeps_the_rest() -> None:
    """``str.capitalize`` dejaba «El r-134a»."""
    assert upper_first("el R-134a") == "El R-134a"
    assert upper_first("") == ""


def test_exchanger_legend_keeps_the_fluid_name() -> None:
    r = sf.solve_exchanger(
        sf.EXCHANGER_EXAMPLES["Condensador de R-134a enfriado con agua (Çengel §5-4)"].inputs
    )
    fig = bc.exchanger_tq_figure(r, "SI")
    assert [t.name for t in fig.data] == ["El R-134a (se enfría)", "El agua (se calienta)"]


@pytest.mark.parametrize("system", SYSTEMS)
def test_pv_log_axes_label_the_whole_number(system: str) -> None:
    """Con la campana el p–v abarca varias décadas: cada marca lleva su número completo
    (plotly rotulaba 0,2 o 20 con un «2» suelto)."""
    r = cl.solve_closed(
        cl.CLOSED_EXAMPLES["Vapor de agua que se expande sin calor ni fricción"].inputs
    )
    points = [("1", r.state1), ("2", r.state2)]
    fig = bc.state_diagram("pv", r.inputs.substance, points, [("Proceso", r.path, True)], system)  # type: ignore[arg-type]
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        assert axis.type == "log"
        assert axis.tickvals and len(axis.tickvals) == len(axis.ticktext)
        for value, text in zip(axis.tickvals, axis.ticktext, strict=True):
            assert _value(text) == pytest.approx(value, rel=1e-9), text


def test_pv_narrow_range_keeps_plotly_ticks() -> None:
    """En menos de una década plotly elige marcas con el número completo: no se tocan."""
    r = cl.solve_closed(
        cl.CLOSED_EXAMPLES["Aire que se expande contra la atmósfera (no cuasiestático)"].inputs
    )
    points = [("1", r.state1), ("2", r.state2)]
    fig = bc.state_diagram(
        "pv", r.inputs.substance, points, [("Proceso", r.path, False)], "Técnico"
    )
    assert fig.layout.yaxis.tickvals is None


def test_mid_range_gets_one_two_five() -> None:
    gas = ideal_gas_substance("air")
    a = state(gas, "TP", 300.0, 1e5)
    b = state(gas, "TP", 300.0, 3e6)  # una década y media en p y en v
    fig = bc.state_diagram("pv", gas, [("1", a), ("2", b)], [("Proceso", (a, b), True)], "SI")
    values = list(fig.layout.yaxis.tickvals)
    assert 2e5 in values and 5e5 in values and 1e6 in values
    texts = dict(zip(values, fig.layout.yaxis.ticktext, strict=True))
    assert texts[2e5] == "2·10<sup>5</sup>" and texts[1e6] == "10<sup>6</sup>"


def test_isentropic_label_goes_to_the_left() -> None:
    """2 y 2s quedan cerca: el rótulo de 2s va a la izquierda para no pisar el de 2."""
    r = sf.solve_device(
        sf.DEVICE_EXAMPLES["Rendimiento isoentrópico de una turbina de vapor (Çengel §7-12)"].inputs
    )
    points = [("1", r.state1), ("2", r.state2), ("2s", r.state2s)]
    for kind in ("pv", "Ts"):
        fig = bc.state_diagram(kind, r.inputs.substance, points, [], "SI")  # type: ignore[arg-type]
        marks = next(t for t in fig.data if t.name == "Estados")
        assert list(marks.textposition) == ["top right", "top right", "middle left"]


def test_ideal_gas_has_no_dome() -> None:
    gas = ideal_gas_substance("air")
    a, b = state(gas, "TP", 300.0, 1e5), state(gas, "TP", 600.0, 1e5)
    fig = bc.state_diagram("Ts", gas, [("1", a), ("2", b)], [("Proceso", (a, b), True)], "SI")
    assert "Campana" not in [t.name for t in fig.data]


def test_waterfall_total_is_the_sum() -> None:
    fig = bc.waterfall_figure("Balance", [("Q", 2000.0), ("−W", -500.0)], "ΔU", "energy", "SI")
    wf = fig.data[0]
    assert list(wf.measure) == ["relative", "relative", "total"]
    assert wf.y[-1] == pytest.approx(1500.0)
    assert list(wf.x) == ["Q", "−W", "ΔU"]


def test_evolution_shares_the_mass() -> None:
    r = tr.solve_discharging(
        tr.DISCHARGING_EXAMPLES["Tanque de aire que se vacía sin calor"].inputs
    )
    fig = bc.evolution_figure(r.path, "Técnico", "Vaciado")
    p, T = fig.data
    assert p.x == T.x and len(p.x) == len(r.path)
    assert T.y[0] == pytest.approx(r.state1.T - 273.15) and T.y[-1] == pytest.approx(
        r.state2.T - 273.15
    )
