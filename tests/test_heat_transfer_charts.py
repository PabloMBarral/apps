"""Tests de ui.heat_transfer_charts (los gráficos de /Transferencia_de_Calor)."""

from __future__ import annotations

import pytest

from core.heat_transfer import conduction as cd
from core.heat_transfer import convection as cv
from core.heat_transfer import fins as fn
from ui.heat_transfer_charts import (
    conduction_profile_figure,
    correlation_figure,
    fin_efficiency_figure,
    fin_profile_figure,
    insulation_figure,
    nu_curve_figure,
    tube_figure,
)

C = 273.15
TECH = {"t_scale": 1.0, "t_offset": -C, "t_unit": "°C"}
NAMES = list(cd.CONDUCTION_EXAMPLES)


def _trace(fig, name: str):  # noqa: ANN001, ANN202
    return next(t for t in fig.data if t.name == name)


def test_profile_of_the_double_window() -> None:
    r = cd.solve_conduction(cd.CONDUCTION_EXAMPLES[NAMES[2]])
    fig = conduction_profile_figure(r, **TECH, x_scale=1000.0, x_unit="mm")
    solid = [t for t in fig.data if t.name == "sólido"]
    assert len(solid) == 3  # una curva por capa
    assert [t.showlegend for t in solid] == [True, False, False]
    # las capas sombreadas, numeradas
    assert len(fig.layout.shapes) == 3
    assert [a.text for a in fig.layout.annotations if a.text.isdigit()] == ["1", "2", "3"]
    # los fluidos: T∞ en los extremos y la caída hasta la cara
    fluids = [t for t in fig.data if t.name.startswith("fluido")]
    assert len(fluids) == 2
    assert fluids[0].y[0] == pytest.approx(20.0) and fluids[1].y[0] == pytest.approx(-10.0)
    assert fluids[0].y[-1] == pytest.approx(r.surface_temperatures_K[0] - C)
    assert fig.layout.xaxis.title.text == "x [mm]"


def test_profile_of_the_contact_and_the_wire() -> None:
    plates = cd.solve_conduction(cd.CONDUCTION_EXAMPLES[NAMES[4]])
    fig = conduction_profile_figure(plates, **TECH, x_scale=1000.0, x_unit="mm")
    jump = _trace(fig, "salto en el contacto")
    contact = plates.resistances[1]
    assert jump.y[0] - jump.y[1] == pytest.approx(contact.dT_K)
    wire = cd.solve_conduction(cd.CONDUCTION_EXAMPLES[NAMES[6]])
    fig = conduction_profile_figure(wire, **TECH, x_scale=1000.0, x_unit="mm")
    assert any(a.text == "Q̇ entra" for a in fig.layout.annotations)
    assert fig.layout.xaxis.title.text == "r [mm]"


def test_insulation_figure_marks_the_critical_radius() -> None:
    inputs = cd.CONDUCTION_EXAMPLES[NAMES[6]]
    r = cd.solve_conduction(inputs)
    points = cd.insulation_sweep(inputs, cd.default_insulation_radii(inputs))
    fig = insulation_figure(
        points,
        r,
        heat_given=True,
        x_scale=1000.0,
        x_unit="mm",
        y_scale=1.0,
        y_offset=-C,
        y_unit="°C",
    )
    curve = _trace(fig, "T del lado 1")
    lowest = min(range(len(curve.y)), key=lambda k: curve.y[k])
    assert curve.x[lowest] == pytest.approx(1000.0 * r.critical_radius_m, rel=0.05)
    assert fig.layout.shapes[0].x0 == pytest.approx(1000.0 * r.critical_radius_m)
    assert _trace(fig, "los datos").y[0] == pytest.approx(r.T_side1_K - C)


def test_insulation_figure_of_a_cold_tank_shows_the_heat_that_enters() -> None:
    inputs = cd.CONDUCTION_EXAMPLES[NAMES[7]]  # el tanque de agua helada: Q̇ < 0
    r = cd.solve_conduction(inputs)
    assert r.Q_W < 0.0
    points = cd.insulation_sweep(inputs, cd.default_insulation_radii(inputs))
    fig = insulation_figure(
        points,
        r,
        heat_given=False,
        x_scale=1.0,
        x_unit="m",
        y_scale=1.0,
        y_offset=0.0,
        y_unit="W",
    )
    curve = _trace(fig, "|Q̇|")
    assert all(y > 0.0 for y in curve.y)
    # con |Q̇| el radio crítico vuelve a ser un máximo
    highest = max(range(len(curve.y)), key=lambda k: curve.y[k])
    assert curve.x[highest] == pytest.approx(r.critical_radius_m, rel=0.05)
    assert _trace(fig, "los datos").y[0] == pytest.approx(-r.Q_W)
    assert fig.layout.yaxis.title.text == "|Q̇| [W]"


def test_fin_figures() -> None:
    ex = fn.FIN_EXAMPLES[list(fn.FIN_EXAMPLES)[4]]  # disipador, longitud corregida
    r = fn.solve_fin(ex.fin)
    others = fn.tip_comparison(ex.fin)
    fig = fin_profile_figure(
        r, others, x_scale=1000.0, x_unit="mm", t_scale=1.0, t_offset=-C, t_unit="°C"
    )
    names = [t.name for t in fig.data]
    assert names[-1] == "longitud corregida"  # la elegida, arriba de las grises
    assert len(names) == 4
    assert fig.data[-1].y[0] == pytest.approx(80.0)
    eta = fin_efficiency_figure(r)
    assert _trace(eta, "la aleta").y[0] == pytest.approx(r.efficiency)
    infinite = fn.solve_fin(fn.FIN_EXAMPLES[list(fn.FIN_EXAMPLES)[0]].fin)
    assert [t.name for t in fin_efficiency_figure(infinite).data] == ["η(m·L_c)"]


def test_convection_figures() -> None:
    ex = next(iter(cv.EXTERNAL_EXAMPLES.values()))
    r = cv.solve_external(ex.inputs)
    var, xs, nus = cv.nu_curve(r)
    fig = nu_curve_figure(var, xs, nus, r.Re, r.Nu, "Churchill y Bernstein (1977)")
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert _trace(fig, "los datos").y[0] == pytest.approx(r.Nu)
    comp = correlation_figure(r.alternatives, r.correlation, h_scale=1.0, h_unit="W/(m²·K)")
    assert [t.name for t in comp.data] == ["en su rango"]
    assert list(comp.data[0].marker.color) == ["#2a78d6", "#6b7280"]  # la usada en azul
    sphere = cv.solve_external(
        cv.ExternalFlowInputs("Air", "sphere", 15.0, 75.0 + C, 20.0 + C, 0.025)
    )
    plate = cv.solve_external(cv.ExternalFlowInputs("Air", "plate", 2.0, 75.0 + C, 25.0 + C, 0.5))
    out = correlation_figure(plate.alternatives, plate.correlation, h_scale=1.0, h_unit="W")
    assert [t.name for t in out.data] == ["en su rango", "fuera de su rango"]
    assert sphere.alternatives


@pytest.mark.parametrize("name", list(cv.INTERNAL_EXAMPLES))
def test_tube_figure(name: str) -> None:
    r = cv.solve_internal(cv.INTERNAL_EXAMPLES[name].inputs)
    fig = tube_figure(r, x_scale=1.0, x_unit="m", t_scale=1.0, t_offset=-C, t_unit="°C")
    fluid = _trace(fig, "fluido T_m")
    assert fluid.y[0] == pytest.approx(r.inputs.T_in_K - C)
    assert fluid.y[-1] == pytest.approx(r.T_out_K - C)
    wall = _trace(fig, "pared T_s")
    if r.inputs.condition == "constant_T":
        assert all(T == pytest.approx(r.inputs.T_s_K - C) for T in wall.y)
    else:
        assert wall.y[-1] == pytest.approx(r.T_s_out_K - C)
