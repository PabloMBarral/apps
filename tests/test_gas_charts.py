"""Tests de ui.gas_charts (Fase 9.1)."""

from __future__ import annotations

import pytest

from core.gases import ideal as ig
from core.gases import mixture as mx
from core.gases import polytropic as pt
from core.units_system import convert_from_si
from ui.gas_charts import (
    PROCESS_COLORS,
    composition_figure,
    cp_figure,
    far_paths,
    mixing_entropy_figure,
    process_label,
    pv_figure,
    staged_figure,
    ts_figure,
    work_figure,
)

C = 273.15


def _trace(fig, name: str):  # noqa: ANN001, ANN202
    return next(t for t in fig.data if t.name == name)


def test_cp_curve_with_the_constant_and_the_mean() -> None:
    g = ig.gas("air")
    fig = cp_figure(g, 300.0, 1500.0, "Técnico")
    curve = _trace(fig, "c_p(T) (polinomio NASA)")
    assert curve.x[0] == pytest.approx(200.0 - C)  # de 200 K
    assert curve.x[-1] == pytest.approx(2000.0 - C)  # a 2000 K
    assert curve.y[0] == pytest.approx(g.cp(200.0) / 1e3)
    const = _trace(fig, "c_p constante (25 °C)")
    assert list(const.y) == pytest.approx([g.cp_ref / 1e3] * 2)
    mean = _trace(fig, "c_p a la T media")
    assert mean.x[0] == pytest.approx(900.0 - C)
    assert mean.y[0] == pytest.approx(g.cp(900.0) / 1e3)
    band = fig.layout.shapes[0]
    assert (band.x0, band.x1) == pytest.approx((300.0 - C, 1500.0 - C))


def test_cp_range_follows_a_hot_process_and_skips_the_band_at_constant_T() -> None:
    g = ig.gas("CO2")
    fig = cp_figure(g, 400.0, 3000.0, "SI")
    assert _trace(fig, "c_p(T) (polinomio NASA)").x[-1] == pytest.approx(3450.0)
    fig = cp_figure(g, 400.0, 400.0, "Inglés")
    assert not fig.layout.shapes  # sin tramo que sombrear


def test_composition_bars_add_to_one_hundred() -> None:
    r = mx.solve_mixture(mx.COMPOSITION_EXAMPLES["Gas natural a 30 °C (fracciones molares)"].inputs)
    fig = composition_figure(r)
    assert len(fig.data) == len(r.components)
    mass = sum(t.x[0] for t in fig.data)
    molar = sum(t.x[1] for t in fig.data)
    assert mass == pytest.approx(100.0) and molar == pytest.approx(100.0)
    # los colores en el orden fijo de la paleta, sin repetir
    colors = [t.marker.color for t in fig.data]
    assert len(set(colors)) == len(colors)
    assert colors[0] == "#2a78d6"
    # las fracciones chicas no llevan rótulo (no entra)
    co2 = next(t for t in fig.data if t.name == "CO₂")
    assert list(co2.text) == ["", ""]


def test_mixing_entropy_by_stream_and_total() -> None:
    name = next(n for n in mx.MIXING_EXAMPLES if n.startswith("Tanque"))
    r = mx.solve_mixing(mx.MIXING_EXAMPLES[name].inputs)
    fig = mixing_entropy_figure(r, "Técnico")
    tp = _trace(fig, "igualar T y p")
    mix = _trace(fig, "mezclar gases distintos")
    assert list(tp.y) == ["1: O₂", "2: N₂", "Total"]
    assert tp.x[-1] == pytest.approx(r.S_gen_TP / 1e3)
    assert mix.x[-1] == pytest.approx(r.S_gen_mix / 1e3)
    assert tp.x[-1] + mix.x[-1] == pytest.approx(r.S_gen / 1e3)
    assert tp.x[0] < 0.0 < tp.x[1]  # el O₂ caliente se enfría: pierde entropía
    assert fig.layout.barmode == "relative"


def _rows(name: str) -> tuple[pt.ComparisonRow, ...]:
    return pt.process_comparison(pt.PROCESS_EXAMPLES[name].inputs)


def test_paths_in_pv_and_ts() -> None:
    name = next(iter(pt.PROCESS_EXAMPLES))  # el compresor de Çengel
    rows = _rows(name)
    fig = pv_figure(rows, "SI")
    lines = [t for t in fig.data if t.mode == "lines"]
    assert len(lines) == sum(row.result is not None for row in rows)
    data = next(row for row in rows if row.is_data)
    data_line = _trace(fig, process_label(data) + " (el dato)")
    assert data_line.line.width == 4
    assert data_line.line.color == PROCESS_COLORS["polytropic"]
    states = next(t for t in fig.data if t.mode == "markers+text")
    assert data.result is not None
    assert states.x[0] == pytest.approx(data.result.state1.v_m3_per_kg)
    assert states.y[1] == pytest.approx(data.result.state2.p_Pa)
    fig = ts_figure(rows, "Técnico")
    states = next(t for t in fig.data if t.mode == "markers+text")
    assert states.y[0] == pytest.approx(300.0 - C)
    adiabatic = _trace(fig, "Adiabática")
    assert adiabatic.x[0] == pytest.approx(adiabatic.x[-1], abs=1e-9)  # vertical en el T–s


def test_work_and_heat_of_each_path() -> None:
    name = next(iter(pt.PROCESS_EXAMPLES))
    rows = _rows(name)
    fig = work_figure(rows, "Técnico", closed=False)
    work = _trace(fig, "trabajo de circulación w_f")
    heat = _trace(fig, "calor q")
    ok = [row for row in rows if row.result is not None and row.process not in far_paths(rows)]
    assert list(work.x) == pytest.approx([row.result.w_f / 1e3 for row in ok])  # type: ignore[union-attr]
    assert list(heat.x) == pytest.approx([row.result.q / 1e3 for row in ok])  # type: ignore[union-attr]
    assert any(label.startswith("▸ ") for label in work.y)
    closed = work_figure(rows, "Técnico", closed=True)
    assert _trace(closed, "trabajo w").x[0] == pytest.approx(ok[0].result.w / 1e3)  # type: ignore[union-attr]


@pytest.mark.parametrize("stages", [1, 3])
def test_staged_compression_paths(stages: int) -> None:
    r = pt.staged_compression(pt.StagedInputs("air", 1e5, 300.0, 9e5, 1.3, stages))
    fig = staged_figure(r, "Inglés")
    assert len(fig.data) == (2 if stages == 1 else 3)
    single = _trace(fig, "una etapa")
    assert single.y[-1] == pytest.approx(convert_from_si(9e5, "pressure", "Inglés"))
    if stages > 1:
        staged = _trace(fig, f"{stages} etapas con interenfriamiento")
        assert staged.y[-1] == pytest.approx(single.y[-1])
        assert staged.x[-1] < single.x[-1]  # sale más fría: menos volumen


def test_far_paths_are_hidden_in_the_ts_and_left_out_of_the_work() -> None:
    """La isócora hasta la p₂ del compresor llega a 9·T₁ y aplastaría a los demás caminos."""
    name = next(iter(pt.PROCESS_EXAMPLES))
    rows = _rows(name)
    assert far_paths(rows) == {"isochoric"}
    ts = ts_figure(rows, "SI")
    assert _trace(ts, "Isócora").visible == "legendonly"
    assert _trace(ts, "Isoterma").visible is True
    pv = pv_figure(rows, "SI")
    assert _trace(pv, "Isócora").visible is True  # en el p–v no molesta (es vertical)
    work = work_figure(rows, "SI", closed=False)
    assert "Isócora" not in _trace(work, "calor q").y
    # con un dato que ya es la isócora no hay nada que ocultar
    tank = next(n for n in pt.PROCESS_EXAMPLES if "tanque rígido" in n)
    assert far_paths(_rows(tank)) == set()
