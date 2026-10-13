"""Tests de los gráficos de la ebullición y la condensación (Fase 11.2)."""

from __future__ import annotations

import pytest

from core.heat_transfer import boiling as bo
from core.heat_transfer import condensation as co
from ui import phase_change_charts as pcc

UNITS = {"q_scale": 1.0, "q_unit": "W/m²", "dT_scale": 1.0, "dT_unit": "K"}


def _boiling(name: str) -> bo.PoolBoilingResult:
    return bo.solve_pool_boiling(bo.BOILING_EXAMPLES[name].inputs)


def _condensation(name: str) -> co.CondensationResult:
    return co.solve_condensation(co.CONDENSATION_EXAMPLES[name].inputs)


def test_boiling_curve_has_one_trace_per_regime_and_the_point() -> None:
    r = _boiling("Olla de acero inoxidable pulido (Çengel y Ghajar §10-2)")
    fig = pcc.boiling_curve_figure(bo.boiling_curve(r), r, **UNITS)
    names = [t.name for t in fig.data]
    assert names[:4] == [
        "convección natural",
        "ebullición nucleada",
        "ebullición de transición (interpolada)",
        "ebullición en película",
    ]
    assert "tu punto" in names
    point = next(t for t in fig.data if t.name == "tu punto")
    assert point.x[0] == pytest.approx(r.dT_K) and point.y[0] == pytest.approx(r.q)
    notable = next(t for t in fig.data if t.name == "puntos notables")
    assert list(notable.text) == ["flujo crítico", "Leidenfrost"]
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert all("e+" not in t and "e-" not in t for t in fig.layout.yaxis.ticktext)
    assert not fig.layout.annotations  # sin burnout


def test_boiling_curve_segments_are_continuous() -> None:
    r = _boiling("Olla de acero inoxidable pulido (Çengel y Ghajar §10-2)")
    fig = pcc.boiling_curve_figure(bo.boiling_curve(r), r, **UNITS)
    lines = [t for t in fig.data if t.mode == "lines"]
    for a, b in zip(lines, lines[1:], strict=False):
        assert b.x[0] == a.x[-1] and b.y[0] == a.y[-1]


def test_burnout_arrow() -> None:
    r = _boiling("Una resistencia que se quema (ejemplo propio)")
    fig = pcc.boiling_curve_figure(bo.boiling_curve(r), r, **UNITS)
    texts = [a.text for a in fig.layout.annotations]
    assert "burnout" in texts
    assert any(a.showarrow for a in fig.layout.annotations)


def test_units_scale_the_axes() -> None:
    r = _boiling("Olla de acero inoxidable pulido (Çengel y Ghajar §10-2)")
    fig = pcc.boiling_curve_figure(
        bo.boiling_curve(r), r, q_scale=2.0, q_unit="X", dT_scale=1.8, dT_unit="°F"
    )
    point = next(t for t in fig.data if t.name == "tu punto")
    assert point.x[0] == pytest.approx(1.8 * r.dT_K) and point.y[0] == pytest.approx(2.0 * r.q)
    assert "[°F]" in fig.layout.xaxis.title.text


def test_condensation_curve_two_panels_for_a_wall() -> None:
    r = _condensation("Placa vertical en vapor a 1 atm (Çengel y Ghajar §10-5)")
    fig = pcc.condensation_curve_figure(
        co.condensation_curve(r, points=40),
        r,
        h_scale=1.0,
        h_unit="W/(m²·K)",
        x_scale=1.0,
        x_unit="m",
    )
    assert fig.layout.yaxis2.title.text == "Re"
    assert fig.layout.yaxis2.type == "log"
    legend = [t.name for t in fig.data if t.showlegend is not False]
    assert {"laminar", "laminar con ondas", "turbulenta", "tus datos"} <= set(legend)
    assert [s.y0 for s in fig.layout.shapes] == [co.RE_LAMINAR, co.RE_TURBULENT]


def test_condensation_curve_against_the_diameter() -> None:
    r = _condensation("Tubo horizontal del condensador de una central (Çengel y Ghajar §10-5)")
    fig = pcc.condensation_curve_figure(
        co.condensation_curve(r, points=30),
        r,
        h_scale=1.0,
        h_unit="W/(m²·K)",
        x_scale=1000.0,
        x_unit="mm",
    )
    assert "Diámetro D [mm]" in fig.layout.xaxis.title.text
    point = next(t for t in fig.data if t.name == "tus datos")
    assert point.x[0] == pytest.approx(30.0)


def test_comparison_highlights_the_data() -> None:
    r = _condensation("Banco de 3 × 4 tubos (Çengel y Ghajar §10-5)")
    rows = co.geometry_comparison(r)
    fig = pcc.comparison_figure(rows, h_scale=1.0, h_unit="W/(m²·K)")
    current = next(t for t in fig.data if t.name == "tus datos")
    assert current.marker.color == "#eb6834"
    assert current.x[0] == pytest.approx(r.h)
    assert len(fig.data[0].x) + len(current.x) == len(rows)
    assert all(
        len(line) <= 22 for name in fig.layout.yaxis.categoryarray for line in name.split("<br>")
    )
