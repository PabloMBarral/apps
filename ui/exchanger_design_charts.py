"""Gráficos del diseño de intercambiadores — Fases 12.1 y 12.2.

- :func:`friction_chart_figure`: el f de Zukauskas en log–log (Incropera, figs. 7.13 y 7.14;
  Çengel y Ghajar, fig. 7-27), una curva por P_L (alineado) o P_T (tresbolillo), con el
  punto de los datos.
- :func:`velocity_sweep_figure`: h y Δp del banco contra la velocidad de llegada, en dos
  paneles con la velocidad compartida (nunca un eje doble): en log–log son rectas, con
  pendientes de ~0,6 y ~1,8.
- :func:`baffle_sweep_figure`: h_o y Δp del casco contra la separación de los bafles, en
  dos paneles: el compromiso del diseño con Kern.
- :func:`h_cascade_figure` (Bell–Delaware): del h del banco ideal al h_o, un escalón por
  factor (J_c, J_l, J_b, J_s y J_r); los que valen 1 no aparecen.
- :func:`dp_zones_figure` (Bell–Delaware): el Δp del flujo cruzado, de las ventanas y de los
  extremos, cada uno con su barra ideal (sin fugas ni bypass) al lado.
- :func:`methods_sweep_figure`: h_o y Δp contra la separación o contra el corte de los
  bafles, con Kern y Bell–Delaware en los mismos dos paneles (Kern no ve el corte: es una
  recta).

El perfil de temperaturas de un doble tubo y el peso de las resistencias del U salen de
:mod:`ui.exchanger_charts` (Fase 8.2).

Sin título propio: la sección de la página lo nombra (a 390 px la barra de plotly tapaba el
título cuando se tocaba el gráfico).

Colores: la paleta validada para daltonismo del proyecto. La familia de curvas de f va en la
rampa azul ordinal (de claro a oscuro con el paso); el coeficiente h va en azul y la caída
de presión en naranja; el punto de los datos, en tinta con borde blanco. En la cascada, lo
que suma va en azul, lo que resta en naranja y los totales en gris (como las cascadas de
:mod:`ui.balance_charts`); en las zonas del Δp, el ideal en gris y el real en azul; en los
barridos de los dos métodos, Bell–Delaware en azul y Kern en naranja rayado (el color es
el método; el panel, la magnitud).

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en
:mod:`core.heat_transfer.tube_bank` y :mod:`core.heat_transfer.shell_tube`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.heat_transfer.shell_tube import BaffleSweepPoint
from core.heat_transfer.tube_bank import Arrangement, Curve, SweepPoint

__all__ = [
    "baffle_sweep_figure",
    "dp_zones_figure",
    "friction_chart_figure",
    "h_cascade_figure",
    "methods_sweep_figure",
    "velocity_sweep_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_GRAY = "#6b7280"
_LIGHT = "#9ca3af"
_INK = "#1f2937"
_GRID = "#e5e7eb"
#: Rampa azul ordinal (pasos 250 a 650 de la paleta validada).
_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0}
#: El punto de los datos: tinta con borde blanco (se ve sobre el azul y el naranja).
_MARKER = {"size": 13, "color": _INK, "line": {"color": "white", "width": 2}}


def _comma(x: float, fmt: str = ".3g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _tick_text(v: float, full: bool = True) -> str:
    """El rótulo de una marca. Con ``full``, el número completo hasta 10⁶ (con espacio de
    miles desde 10⁴, como :func:`ui.balance_charts.value_axis`); sin él (un eje de muchas
    décadas, con una marca por potencia de 10), potencias desde 10⁴, así no quedan «10000»
    y «100000» junto a «10⁶»."""
    if full and 1e4 <= v < 1e6:
        return f"{v:,.0f}".replace(",", "\u2009")
    if 1e-3 <= v < 1e4:
        return _comma(v, "g")
    exponent = math.floor(math.log10(v) + 1e-9)
    mantissa = v / 10.0**exponent
    head = "" if abs(mantissa - 1.0) < 1e-9 else f"{_comma(mantissa, 'g')}·"
    return f"{head}10{str(exponent).translate(_SUP)}"


def _log_ticks(values: Sequence[float]) -> dict[str, object]:
    """Marcas con el número completo en un eje logarítmico (plotly rotula 20 y 0,2 con un
    «2» suelto): más densas con menos de una década (si no, quedaba una sola marca),
    1-2-5 hasta dos décadas y media y, con más, las potencias de 10 (rotuladas como
    potencias desde 10⁴)."""
    pos = [v for v in values if v > 0.0 and math.isfinite(v)]
    if not pos:
        return {}
    span = math.log10(max(pos) / min(pos))
    lo, hi = math.floor(math.log10(min(pos))), math.ceil(math.log10(max(pos)))
    if span <= 0.5:
        factors: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
    elif span <= 1.2:
        factors = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    elif span <= 2.5:
        factors = (1.0, 2.0, 5.0)
    else:
        factors = (1.0,)
    vals = [f * 10.0**e for e in range(lo - 1, hi + 2) for f in factors]
    full = len(factors) > 1
    return {"tickvals": vals, "ticktext": [_tick_text(v, full) for v in vals]}


def _ramp(k: int, n: int) -> str:
    """El color k de n curvas de la rampa (de claro a oscuro)."""
    if n <= 1:
        return _RAMP[2]
    return _RAMP[round(k * (len(_RAMP) - 1) / (n - 1))]


# ---------------------------------------------------------------------
# El f de Zukauskas
# ---------------------------------------------------------------------


def friction_chart_figure(
    curves: Mapping[float, Curve],
    arrangement: Arrangement,
    *,
    Re: float,
    f: float,
    pitch: float,
) -> go.Figure:
    """f contra Re en log–log, una curva por paso, con el punto de los datos (el f leído
    entre las dos curvas vecinas, antes de la corrección χ)."""
    aligned = arrangement == "aligned"
    sym = "P_L" if aligned else "P_T"
    fig = go.Figure()
    xs_all: list[float] = [Re]
    ys_all: list[float] = [f]
    pitches = sorted(curves)
    for k, p in enumerate(pitches):
        xs, ys = curves[p]
        xs_all += list(xs)
        ys_all += list(ys)
        name = f"{sym} = {_comma(p)}"
        fig.add_trace(
            go.Scatter(
                x=list(xs),
                y=list(ys),
                mode="lines",
                name=name,
                line={"color": _ramp(k, len(pitches)), "width": 2.2},
                hovertemplate=f"{name}<br>Re = %{{x:.4g}}<br>f = %{{y:.3g}}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[Re],
            y=[f],
            mode="markers",
            name=f"tus datos ({sym} = {_comma(pitch)})",
            marker=_MARKER,
            hovertemplate=f"{sym} = {_comma(pitch)}<br>Re = %{{x:.4g}}<br>f = %{{y:.3g}}"
            "<extra></extra>",
        )
    )
    fig.update_xaxes(type="log", title="Re = ρ·V_máx·D/μ", gridcolor=_GRID, **_log_ticks(xs_all))
    fig.update_yaxes(type="log", title="f", gridcolor=_GRID, **_log_ticks(ys_all))
    fig.update_layout(
        height=410,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    return fig


# ---------------------------------------------------------------------
# Barridos: dos paneles con el eje x compartido
# ---------------------------------------------------------------------


def _two_panels(
    x: Sequence[float],
    top: Sequence[float],
    bottom: Sequence[float],
    *,
    x_now: float,
    top_now: float,
    bottom_now: float,
    x_title: str,
    top_title: str,
    bottom_title: str,
    hover_x: str,
) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for row, ys, now, color, name in (
        (1, top, top_now, _BLUE, top_title),
        (2, bottom, bottom_now, _ORANGE, bottom_title),
    ):
        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=list(ys),
                mode="lines",
                name=name,
                line={"color": color, "width": 2.6},
                hovertemplate=f"{hover_x} = %{{x:.4g}}<br>{name} = %{{y:.4g}}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[x_now],
                y=[now],
                mode="markers",
                name="tus datos",
                showlegend=row == 1,
                marker=_MARKER,
                hovertemplate=f"{hover_x} = %{{x:.4g}}<br>{name} = %{{y:.4g}}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(
            type="log",
            title=name,
            gridcolor=_GRID,
            row=row,
            col=1,
            **_log_ticks([*ys, now]),
        )
    ticks = _log_ticks([*x, x_now])
    fig.update_xaxes(type="log", gridcolor=_GRID, row=1, col=1, **ticks)
    fig.update_xaxes(type="log", title=x_title, gridcolor=_GRID, row=2, col=1, **ticks)
    fig.update_layout(
        height=500,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    return fig


def velocity_sweep_figure(
    points: Sequence[SweepPoint],
    *,
    V_now: float,
    h_now: float,
    dp_now: float,
    V_scale: float,
    V_unit: str,
    h_scale: float,
    h_unit: str,
    dp_scale: float,
    dp_unit: str,
) -> go.Figure:
    """h y Δp del banco contra la velocidad de llegada (log–log: rectas)."""
    return _two_panels(
        [p.V_m_s * V_scale for p in points],
        [p.h_W_per_m2K * h_scale for p in points],
        [p.dp_Pa * dp_scale for p in points],
        x_now=V_now * V_scale,
        top_now=h_now * h_scale,
        bottom_now=dp_now * dp_scale,
        x_title=f"Velocidad de llegada V [{V_unit}]",
        top_title=f"h [{h_unit}]",
        bottom_title=f"Δp [{dp_unit}]",
        hover_x="V",
    )


def baffle_sweep_figure(
    points: Sequence[BaffleSweepPoint],
    *,
    B_now: float,
    h_now: float,
    dp_now: float,
    B_scale: float,
    B_unit: str,
    h_scale: float,
    h_unit: str,
    dp_scale: float,
    dp_unit: str,
) -> go.Figure:
    """h_o y Δp del casco contra la separación de los bafles (log–log)."""
    return _two_panels(
        [p.B_m * B_scale for p in points],
        [p.h_o_W_per_m2K * h_scale for p in points],
        [p.dp_shell_Pa * dp_scale for p in points],
        x_now=B_now * B_scale,
        top_now=h_now * h_scale,
        bottom_now=dp_now * dp_scale,
        x_title=f"Separación de los bafles B [{B_unit}]",
        top_title=f"h_o [{h_unit}]",
        bottom_title=f"Δp del casco [{dp_unit}]",
        hover_x="B",
    )


# ---------------------------------------------------------------------
# Bell–Delaware (Fase 12.2)
# ---------------------------------------------------------------------


def _bar_text(x: float) -> str:
    """El valor de una barra con cuatro cifras, sin «k» ni notación exponencial."""
    if x == 0.0 or not math.isfinite(x):
        return "0"
    if not 1e-3 <= abs(x) < 1e9:
        return _tick_text(abs(x)) if x > 0 else "−" + _tick_text(abs(x))
    decimals = max(3 - math.floor(math.log10(abs(x))), 0)
    text = f"{round(x, decimals):,.{decimals}f}"
    return text.replace(",", "\u2009").replace(".", ",").replace("-", "−")


def _value_axis(values: Sequence[float]) -> dict[str, str]:
    """Sin los prefijos de plotly («15k» junto a W/(m²·K)), como
    :func:`ui.balance_charts.value_axis`."""
    top = max((abs(v) for v in values), default=0.0)
    return {"exponentformat": "none" if 1e-3 <= top < 1e6 else "power"}


def h_cascade_figure(
    h_ideal: float,
    factors: Sequence[tuple[str, float]],
    *,
    h_scale: float,
    h_unit: str,
) -> go.Figure:
    """La cascada del coeficiente: h_ideal, un escalón por factor (h·(J − 1), con el h que
    llega a ese factor) y h_o. Los factores que valen 1 no aparecen."""
    labels = ["h ideal"]
    values = [h_ideal * h_scale]
    texts = [_bar_text(h_ideal * h_scale)]
    hovers = [f"h ideal = {_bar_text(h_ideal * h_scale)} {h_unit}"]
    h = peak = h_ideal
    for name, J in factors:
        if abs(J - 1.0) < 1e-9:
            continue
        step = h * (J - 1.0)
        h *= J
        peak = max(peak, h)
        labels.append(name)
        values.append(step * h_scale)
        texts.append(f"×{_comma(J, '.3g')}")
        hovers.append(f"{name} = {_comma(J, '.4g')}: {_bar_text(step * h_scale)} {h_unit}")
    labels.append("h_o")
    values.append(h * h_scale)
    texts.append(_bar_text(h * h_scale))
    hovers.append(f"h_o = {_bar_text(h * h_scale)} {h_unit}")
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            x=labels,
            y=values,
            measure=["absolute"] + ["relative"] * (len(labels) - 2) + ["total"],
            text=texts,
            textposition="outside",
            hovertext=hovers,
            hoverinfo="text",
            increasing={"marker": {"color": _BLUE}},
            decreasing={"marker": {"color": _ORANGE}},
            totals={"marker": {"color": _GRAY}},
            connector={"line": {"color": _LIGHT, "width": 1}},
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=340,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis={
            "title": f"h [{h_unit}]",
            "gridcolor": _GRID,
            "zerolinecolor": _INK,
            "range": [0.0, 1.15 * peak * h_scale],
            **_value_axis(values),
        },
        xaxis={"type": "category"},
        showlegend=False,
        separators=", ",
    )
    return fig


def dp_zones_figure(
    zones: Sequence[tuple[str, float, float]],
    *,
    dp_scale: float,
    dp_unit: str,
) -> go.Figure:
    """El Δp de cada zona del casco (nombre, ideal, real): la barra ideal (sin fugas ni
    bypass: R_l = R_b = 1) al lado de la real."""
    names = [z[0] for z in zones]
    ideal = [z[1] * dp_scale for z in zones]
    real = [z[2] * dp_scale for z in zones]
    fig = go.Figure()
    for label, ys, color in (
        ("ideal (sin fugas ni bypass)", ideal, _LIGHT),
        ("con fugas y bypass", real, _BLUE),
    ):
        fig.add_trace(
            go.Bar(
                x=names,
                y=ys,
                name=label,
                marker={"color": color},
                text=[_bar_text(v) for v in ys],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"{label}<br>%{{x}}: %{{y:.4g}} {dp_unit}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis={
            "title": f"Δp [{dp_unit}]",
            "gridcolor": _GRID,
            "zerolinecolor": _INK,
            "range": [0.0, 1.18 * max([*ideal, *real, 0.0])],
            **_value_axis([*ideal, *real]),
        },
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    return fig


#: El color y el trazo de cada método (el color es el método; el panel, la magnitud).
_METHOD_STYLE: dict[str, dict[str, object]] = {
    "Bell–Delaware": {"color": _BLUE, "width": 2.6},
    "Kern": {"color": _ORANGE, "width": 2.2, "dash": "dash"},
}


def methods_sweep_figure(
    series: Mapping[str, tuple[Sequence[float], Sequence[float], Sequence[float]]],
    *,
    x_now: float,
    h_now: float,
    dp_now: float,
    x_title: str,
    h_title: str,
    dp_title: str,
    hover_x: str,
    log_x: bool = True,
) -> go.Figure:
    """h_o (arriba) y Δp (abajo) de cada método contra la misma variable, en dos paneles con
    el eje x compartido; el punto de los datos va sobre la curva del método elegido."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    xs_all: list[float] = [x_now]
    tops: list[float] = [h_now]
    bottoms: list[float] = [dp_now]
    for name, (xs, hs, dps) in series.items():
        xs_all += list(xs)
        tops += list(hs)
        bottoms += list(dps)
        style = _METHOD_STYLE.get(name, {"color": _INK, "width": 2.0})
        for row, ys, label in ((1, hs, h_title), (2, dps, dp_title)):
            fig.add_trace(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=row == 1,
                    line=style,
                    hovertemplate=f"{name}<br>{hover_x} = %{{x:.4g}}<br>{label} = %{{y:.4g}}"
                    "<extra></extra>",
                ),
                row=row,
                col=1,
            )
    for row, now, label in ((1, h_now, h_title), (2, dp_now, dp_title)):
        fig.add_trace(
            go.Scatter(
                x=[x_now],
                y=[now],
                mode="markers",
                name="tus datos",
                legendgroup="now",
                showlegend=row == 1,
                marker=_MARKER,
                hovertemplate=f"{hover_x} = %{{x:.4g}}<br>{label} = %{{y:.4g}}<extra></extra>",
            ),
            row=row,
            col=1,
        )
    for row, ys, label in ((1, tops, h_title), (2, bottoms, dp_title)):
        fig.update_yaxes(type="log", title=label, gridcolor=_GRID, row=row, col=1, **_log_ticks(ys))
    if log_x:
        ticks = _log_ticks(xs_all)
        fig.update_xaxes(type="log", gridcolor=_GRID, row=1, col=1, **ticks)
        fig.update_xaxes(type="log", title=x_title, gridcolor=_GRID, row=2, col=1, **ticks)
    else:
        fig.update_xaxes(gridcolor=_GRID, row=1, col=1)
        fig.update_xaxes(title=x_title, gridcolor=_GRID, row=2, col=1)
    fig.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    return fig
