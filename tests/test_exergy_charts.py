"""Tests de ui.exergy_charts (los gráficos de la página /Exergia)."""

from __future__ import annotations

import pytest

from core.cycles.gas_turbine import GAS_TURBINE_EXAMPLES, solve_gas_turbine
from core.exergy import plant as pl
from ui.exergy_charts import _headline, _wrap_label, grassmann_figure


def test_labels_wrap_without_splitting_a_number_from_its_unit() -> None:
    text = _wrap_label("Escape (gases a 581,4 °C) (pérdida)")
    lines = text.split("<br>")
    assert len(lines) == 2
    assert all(len(line) <= 26 for line in lines)
    assert "581,4 °C" in lines[0]
    assert _wrap_label("Caldera") == "Caldera"


def test_headline_goes_to_two_lines_only_when_long() -> None:
    assert _headline("Trabajo neto", "306 846 kW · 42,8 %") == (
        "<b>Trabajo neto</b> · 306 846 kW · 42,8 %",
        1,
    )
    text, lines = _headline("Frío (exergía del efecto frigorífico)", "21 400 W · 69,1 %")
    assert lines == 2 and "<br>" in text


def test_grassmann_figure_closes_the_balance() -> None:
    name = next(n for n in GAS_TURBINE_EXAMPLES if "secuencial" in n)
    plant = pl.gas_turbine_plant_exergy(solve_gas_turbine(GAS_TURBINE_EXAMPLES[name]))
    rows = pl.grassmann_rows(plant)
    fig = grassmann_figure(
        rows,
        x_in=plant.fuel_W,
        in_label="Exergía que entra",
        product_label="Trabajo neto",
        scale=1e-3,
        unit="kW",
    )
    branches = sum((r.destroyed_W > 0) + (r.loss_W > 0) for r in rows)
    texts = [a.text for a in fig.layout.annotations]
    assert len(texts) == branches + 2  # título, una por rama y el producto
    assert any("Escape" in t and "(pérdida)" in t for t in texts)
    product = plant.fuel_W - sum(r.destroyed_W + r.loss_W for r in rows)
    assert product == pytest.approx(plant.product_W, rel=1e-9)
    assert f"{100 * plant.efficiency:.1f}".replace(".", ",") in texts[-1]
    # más ramas, más alto
    assert fig.layout.height == 120 + 62 * branches
