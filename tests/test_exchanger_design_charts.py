"""Tests de ui.exchanger_design_charts (Fases 12.1 y 12.2)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from core.heat_transfer import shell_tube as stt
from core.heat_transfer import tube_bank as tb
from ui.exchanger_design_charts import (
    baffle_sweep_figure,
    dp_zones_figure,
    friction_chart_figure,
    h_cascade_figure,
    methods_sweep_figure,
    velocity_sweep_figure,
)

BANK = tb.TUBE_BANK_EXAMPLES["Calentador de aire en tresbolillo (Incropera, ejemplo 7.7)"].inputs
WATER = stt.SHELL_TUBE_EXAMPLES[
    "Intercambiador agua–agua de 1 paso por el casco y 2 por los tubos"
].inputs


def _trace(fig, name: str):  # noqa: ANN001, ANN202
    return next(t for t in fig.data if t.name == name)


@pytest.mark.parametrize("arrangement", ["aligned", "staggered"])
def test_friction_chart_has_every_curve_and_the_point(arrangement: str) -> None:
    r = tb.solve_tube_bank(replace(BANK, arrangement=arrangement))
    curves = tb.friction_curves(r.inputs.arrangement)
    fig = friction_chart_figure(
        curves, r.inputs.arrangement, Re=r.Re, f=r.f, pitch=r.friction.pitch
    )
    sym = "P_L" if arrangement == "aligned" else "P_T"
    names = [t.name for t in fig.data]
    assert len(fig.data) == len(curves) + 1
    assert all(n.startswith(f"{sym} = ") for n in names[:-1])
    point = fig.data[-1]
    assert point.x[0] == pytest.approx(r.Re) and point.y[0] == pytest.approx(r.f)
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    # las marcas con el número completo (sin un «2» suelto en 20 o 0,2)
    ticks = fig.layout.yaxis
    for value, text in zip(ticks.tickvals, ticks.ticktext, strict=True):
        if 1e-3 <= value < 1e4:
            assert float(text.replace(",", ".")) == pytest.approx(value)
    # Re abarca varias décadas: una marca por potencia de 10, todas como potencias desde
    # 10⁴ (no «10000» y «100000» junto a «10⁶»)
    re_ticks = dict(zip(fig.layout.xaxis.tickvals, fig.layout.xaxis.ticktext, strict=True))
    assert re_ticks[1e3] == "1000"
    assert re_ticks[1e4] == "10⁴" and re_ticks[1e5] == "10⁵" and re_ticks[1e6] == "10⁶"
    # la rampa ordinal: de claro a oscuro con el paso
    colors = [t.line.color for t in fig.data[:-1]]
    assert colors[0] != colors[-1]


def test_velocity_sweep_two_panels_and_slopes() -> None:
    r = tb.solve_tube_bank(BANK)
    points = tb.velocity_sweep(BANK)
    fig = velocity_sweep_figure(
        points,
        V_now=BANK.V_m_s,
        h_now=r.h_W_per_m2K,
        dp_now=r.dp_Pa,
        V_scale=1.0,
        V_unit="m/s",
        h_scale=1.0,
        h_unit="W/(m²·K)",
        dp_scale=1.0,
        dp_unit="Pa",
    )
    h = _trace(fig, "h [W/(m²·K)]")
    dp = _trace(fig, "Δp [Pa]")
    assert h.yaxis == "y" and dp.yaxis == "y2"  # dos paneles, nunca un eje doble
    assert fig.layout.xaxis2.type == "log" and fig.layout.yaxis2.type == "log"
    # en log–log, rectas: h ~ V^0,6 y Δp ~ V^1,8
    slope_h = math.log(h.y[-1] / h.y[0]) / math.log(h.x[-1] / h.x[0])
    slope_dp = math.log(dp.y[-1] / dp.y[0]) / math.log(dp.x[-1] / dp.x[0])
    assert 0.5 < slope_h < 0.8
    assert 1.6 < slope_dp < 2.0
    markers = [t for t in fig.data if t.name == "tus datos"]
    assert len(markers) == 2 and markers[0].x[0] == pytest.approx(BANK.V_m_s)


def test_baffle_sweep_in_other_units() -> None:
    r = stt.solve_shell_tube(WATER)
    points = stt.baffle_sweep(WATER)
    fig = baffle_sweep_figure(
        points,
        B_now=WATER.B_m,
        h_now=r.shell.h_W_per_m2K,
        dp_now=r.shell.dp_Pa,
        B_scale=3.28084,
        B_unit="ft",
        h_scale=0.17611,
        h_unit="Btu/(h·ft²·°F)",
        dp_scale=1.0 / 6894.757,
        dp_unit="psi",
    )
    h = _trace(fig, "h_o [Btu/(h·ft²·°F)]")
    assert h.x[0] == pytest.approx(points[0].B_m * 3.28084)
    assert h.y[0] == pytest.approx(points[0].h_o_W_per_m2K * 0.17611)
    marker = next(t for t in fig.data if t.name == "tus datos")
    assert marker.y[0] == pytest.approx(r.shell.h_W_per_m2K * 0.17611)
    # sin título propio (lo pone la página): la barra de plotly lo tapaba a 390 px
    assert not fig.layout.title.text
    assert "bafles" in fig.layout.xaxis2.title.text


def test_log_ticks_inside_a_narrow_range() -> None:
    # h_o del casco va de ~480 a ~1150 W/(m²·K): con 1-2-5 quedaba una sola marca
    r = stt.solve_shell_tube(WATER)
    points = stt.baffle_sweep(WATER)
    fig = baffle_sweep_figure(
        points,
        B_now=WATER.B_m,
        h_now=r.shell.h_W_per_m2K,
        dp_now=r.shell.dp_Pa,
        B_scale=1.0,
        B_unit="m",
        h_scale=1.0,
        h_unit="W/(m²·K)",
        dp_scale=1.0,
        dp_unit="Pa",
    )
    for axis, ys in (
        (fig.layout.yaxis, [p.h_o_W_per_m2K for p in points]),
        (fig.layout.yaxis2, [p.dp_shell_Pa for p in points]),
        (fig.layout.xaxis2, [p.B_m for p in points]),
    ):
        inside = [v for v in axis.tickvals if min(ys) <= v <= max(ys)]
        assert len(inside) >= 3, (axis.title, inside)
    # los números completos hasta 10⁶, con espacio de miles desde 10⁴
    texts = dict(zip(fig.layout.yaxis2.tickvals, fig.layout.yaxis2.ticktext, strict=True))
    assert texts[20_000.0] == "20 000" and texts[5000.0] == "5000"


# ---------------------------------------------------------------------
# Bell–Delaware (Fase 12.2)
# ---------------------------------------------------------------------

KEROSENE = stt.SHELL_TUBE_EXAMPLES["Enfriador de kerosene con agua de río"].inputs


def _bd(inputs: stt.ShellTubeInputs) -> stt.ShellTubeResult:
    return stt.solve_shell_tube(replace(inputs, shell_method="bell_delaware"))


def test_h_cascade_goes_from_the_ideal_bank_to_h_o() -> None:
    r = _bd(KEROSENE)
    bd = r.bell_delaware
    assert bd is not None
    fc = bd.factors
    factors = [("J_c", fc.J_c), ("J_l", fc.J_l), ("J_b", fc.J_b), ("J_s", fc.J_s), ("J_r", fc.J_r)]
    fig = h_cascade_figure(bd.h_ideal_W_per_m2K, factors, h_scale=1.0, h_unit="W/(m²·K)")
    w = fig.data[0]
    # J_s y J_r valen 1 (extremos iguales, turbulento): no aparecen
    assert list(w.x) == ["h ideal", "J_c", "J_l", "J_b", "h_o"]
    assert list(w.measure) == ["absolute", "relative", "relative", "relative", "total"]
    assert w.y[0] == pytest.approx(bd.h_ideal_W_per_m2K)
    assert w.y[-1] == pytest.approx(bd.h_W_per_m2K)
    # los escalones suman: h_ideal + Σ escalones = h_o
    assert w.y[0] + sum(w.y[1:-1]) == pytest.approx(bd.h_W_per_m2K)
    assert w.text[2] == "×" + f"{fc.J_l:.3g}".replace(".", ",")
    # sin «k» en el eje y el rango arranca en 0
    assert fig.layout.yaxis.exponentformat == "none"
    assert fig.layout.yaxis.range[0] == 0.0
    # en otro sistema: la escala
    fig = h_cascade_figure(bd.h_ideal_W_per_m2K, factors, h_scale=0.1761, h_unit="Btu")
    assert fig.data[0].y[-1] == pytest.approx(0.1761 * bd.h_W_per_m2K)


def test_dp_zones_ideal_next_to_real() -> None:
    bd = _bd(KEROSENE).bell_delaware
    assert bd is not None
    zones = [
        ("Flujo cruzado", bd.dp_cross_ideal_Pa, bd.dp_cross_Pa),
        ("Ventanas", bd.dp_window_ideal_Pa, bd.dp_window_Pa),
        ("Extremos", bd.dp_ends_ideal_Pa, bd.dp_ends_Pa),
    ]
    fig = dp_zones_figure(zones, dp_scale=1e-3, dp_unit="kPa")
    ideal, real = fig.data
    assert ideal.name.startswith("ideal") and real.name == "con fugas y bypass"
    assert fig.layout.barmode == "group"
    assert list(real.y) == pytest.approx([1e-3 * z[2] for z in zones])
    # las fugas y el bypass siempre bajan el Δp
    assert all(i > r for i, r in zip(ideal.y, real.y, strict=True))
    assert all("e" not in t for t in real.text)


def test_methods_sweep_two_panels_both_methods() -> None:
    i = replace(KEROSENE, shell_method="bell_delaware")
    r = stt.solve_shell_tube(i)
    series = {}
    for name, inputs in (("Bell–Delaware", i), ("Kern", replace(i, shell_method="kern"))):
        pts = stt.baffle_sweep(inputs)
        series[name] = (
            [p.B_m for p in pts],
            [p.h_o_W_per_m2K for p in pts],
            [p.dp_shell_Pa for p in pts],
        )
    fig = methods_sweep_figure(
        series,
        x_now=i.B_m,
        h_now=r.h_shell_W_per_m2K,
        dp_now=r.dp_shell_Pa,
        x_title="B [m]",
        h_title="h_o",
        dp_title="Δp",
        hover_x="B",
    )
    names = [t.name for t in fig.data]
    assert names.count("Bell–Delaware") == 2 and names.count("Kern") == 2
    # una sola entrada por método en la leyenda
    shown = [t.name for t in fig.data if t.showlegend is not False]
    assert shown == ["Bell–Delaware", "Kern", "tus datos"]
    kern = [t for t in fig.data if t.name == "Kern"]
    assert all(t.line.dash == "dash" for t in kern)
    assert fig.layout.xaxis2.type == "log" and fig.layout.yaxis2.type == "log"
    # el corte, en un eje lineal, con Kern como una recta
    cut = stt.baffle_cut_sweep(i)
    fig = methods_sweep_figure(
        {
            "Bell–Delaware": (
                [100 * p.baffle_cut for p in cut],
                [p.h_o_W_per_m2K for p in cut],
                [p.dp_shell_Pa for p in cut],
            ),
            "Kern": ([15.0, 45.0], [r.shell.h_W_per_m2K] * 2, [r.shell.dp_Pa] * 2),
        },
        x_now=25.0,
        h_now=r.h_shell_W_per_m2K,
        dp_now=r.dp_shell_Pa,
        x_title="B_c [%]",
        h_title="h_o",
        dp_title="Δp",
        hover_x="B_c",
        log_x=False,
    )
    assert fig.layout.xaxis2.type != "log"
    assert len(fig.data[0].x) == len(cut)
