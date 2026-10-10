"""Tests de ui.real_gas_charts (Fase 9.2)."""

from __future__ import annotations

import math

import pytest

from core.gases import cubic as cb
from core.gases import real as rg
from core.gases import relations as rl
from core.units_system import convert_from_si
from ui.real_gas_charts import (
    MODEL_COLORS,
    clausius_figure,
    convergence_figure,
    fluid_isotherm_figure,
    generalized_chart_figure,
    inversion_figure,
    saturation_figure,
    vdw_isotherms_figure,
    z_isotherm_figure,
)

C = 273.15


def _trace(fig, name: str):  # noqa: ANN001, ANN202
    return next(t for t in fig.data if t.name == name)


def test_generalized_chart_with_the_state() -> None:
    chart = rg.generalized_chart()
    fig = generalized_chart_figure(chart, (0.25, 0.83), (0.25, 0.86))
    iso = [t for t in fig.data if t.legendgroup == "iso"]
    assert len(iso) == len(chart["isotherms"])
    assert sum(t.showlegend is not False for t in iso) == 1  # una sola entrada en la leyenda
    assert len(fig.layout.annotations) == len(chart["isotherms"])  # rótulos directos
    assert fig.layout.xaxis.type == "log"
    # los rótulos de un eje logarítmico van en log₁₀ (adentro de 0,01 a 10)
    for a in fig.layout.annotations:
        assert math.log10(0.009) <= a.x <= math.log10(11.0)
    state = _trace(fig, "el estado (CoolProp)")
    assert (state.x[0], state.y[0]) == pytest.approx((0.25, 0.83))
    assert _trace(fig, "la carta (Z⁰)").y[0] == pytest.approx(0.86)
    no_state = generalized_chart_figure(chart)
    assert all(t.name != "el estado (CoolProp)" for t in no_state.data)


def test_z_isotherm_draws_the_real_fluid_first() -> None:
    data = rg.z_isotherm("R134a", 50.0 + C, 2.5e6)
    fig = z_isotherm_figure(data, "Técnico", (1e6, 0.828))
    assert fig.data[0].name == "real (CoolProp)"  # abajo: no tapa a los modelos
    assert fig.data[0].line.color == MODEL_COLORS["coolprop"]
    assert _trace(fig, "gas ideal").line.dash == "dash"
    assert _trace(fig, "el estado").x[0] == pytest.approx(10.0)  # 1 MPa en bar
    assert fig.data[0].x[-1] == pytest.approx(25.0)


def test_fluid_isotherm_with_the_maxwell_lines() -> None:
    r = cb.solve_cubic(cb.CUBIC_EXAMPLES["R-134a líquido a 10 bar y 20 °C"].inputs)
    data = cb.fluid_isotherms("R134a", r.inputs.T_K)
    fig = fluid_isotherm_figure(data, "SI", (r.v_m3_per_kg, r.inputs.p_Pa), r.fluid.p_cr)
    vdw_line = _trace(fig, "Maxwell, Van der Waals")
    assert vdw_line.showlegend is False and vdw_line.legendgroup == "vdw"
    sat = data["maxwell"]["vdw"]
    assert list(vdw_line.y) == pytest.approx([sat.p_sat_Pa] * 2)
    assert fig.layout.xaxis.type == "log"
    low, high = fig.layout.yaxis.range
    assert low < 0 < r.inputs.p_Pa < high  # se ve el lazo bajo cero


def test_vdw_isotherms_shade_two_equal_areas() -> None:
    iso = cb.vdw_reduced_isotherms((0.8, 0.9, 1.0))
    fig = vdw_isotherms_figure(iso, 0.9)
    areas = [t for t in fig.data if t.legendgroup == "areas"]
    assert len(areas) == 2 and all(t.fill == "toself" for t in areas)
    line = _trace(fig, "Maxwell a T_R = 0,9")
    assert list(line.y) == pytest.approx([iso["maxwell"][0.9][0]] * 2)
    # sin T_R a sombrear (supercrítica) no hay áreas
    assert not [t for t in vdw_isotherms_figure(iso, 1.0).data if t.legendgroup == "areas"]


def test_saturation_curves_figure() -> None:
    fig = saturation_figure(cb.saturation_curves("Water"), "Inglés")
    assert [t.name for t in fig.data][:3] == ["real (CoolProp)", "Van der Waals", "Peng–Robinson"]
    crit = _trace(fig, "punto crítico")
    assert crit.y[0] == pytest.approx(convert_from_si(22.064e6, "pressure", "Inglés"), rel=1e-4)
    assert fig.layout.yaxis.type == "log"


def test_convergence_has_slope_two() -> None:
    r = rl.relations(
        rl.RELATIONS_EXAMPLES["Vapor de agua a 250 °C y 300 kPa (Çengel §12-2)"].inputs
    )
    fig = convergence_figure(r)
    assert len(fig.data) == 5  # las cuatro relaciones y la recta de referencia
    for t in fig.data[:4]:
        assert t.y[0] / t.y[1] == pytest.approx(4.0, rel=0.1)
    ref = _trace(fig, "pendiente 2 (error ∝ Δ²)")
    assert ref.y[0] / ref.y[2] == pytest.approx(16.0)


def test_inversion_figure() -> None:
    curve = rl.inversion_curve("Nitrogen", 30)
    lines = rl.isenthalps("Nitrogen", (200.0, 400.0), 1e5, 400e5)
    fig = inversion_figure(curve, lines, "Técnico", (50e5, 300.0))
    inv = _trace(fig, "curva de inversión (adentro se enfría)")
    assert inv.fill == "toself" and inv.x[0] == inv.x[-1]  # cerrada
    assert sum(t.legendgroup == "h" for t in fig.data) == 2
    state = _trace(fig, "el estado")
    assert (state.x[0], state.y[0]) == pytest.approx((50.0, 300.0 - C))


def test_clausius_figure() -> None:
    r = rl.clapeyron(rl.CLAPEYRON_EXAMPLES["Agua a 100 °C"].inputs)
    assert r.T2 is not None and r.p_sat2 is not None and r.p_sat2_cc is not None
    fig = clausius_figure(rl.clausius_curve(r), "SI", (r.T, r.p_sat), (r.T2, r.p_sat2, r.p_sat2_cc))
    assert _trace(fig, "T₁ (el dato)").y[0] == pytest.approx(r.p_sat)
    second = _trace(fig, "T₂: real y Clausius–Clapeyron")
    assert list(second.y) == pytest.approx([r.p_sat2, r.p_sat2_cc])
    assert fig.layout.yaxis.type == "log"
