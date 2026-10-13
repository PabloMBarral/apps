"""Gráficos del diseño de intercambiadores — Fase 12.1.

- :func:`friction_chart_figure`: el f de Zukauskas en log–log (Incropera, figs. 7.13 y 7.14;
  Çengel y Ghajar, fig. 7-27), una curva por P_L (alineado) o P_T (tresbolillo), con el
  punto de los datos.
- :func:`velocity_sweep_figure`: h y Δp del banco contra la velocidad de llegada, en dos
  paneles con la velocidad compartida (nunca un eje doble): en log–log son rectas, con
  pendientes de ~0,6 y ~1,8.
- :func:`baffle_sweep_figure`: h_o y Δp del casco contra la separación de los bafles, en
  dos paneles: el compromiso del diseño con Kern.

El perfil de temperaturas de un doble tubo y el peso de las resistencias del U salen de
:mod:`ui.exchanger_charts` (Fase 8.2).

Colores: la paleta validada para daltonismo del proyecto. La familia de curvas de f va en la
rampa azul ordinal (de claro a oscuro con el paso); el coeficiente h va en azul y la caída
de presión en naranja; el punto de los datos, en tinta con borde blanco.

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

__all__ = ["baffle_sweep_figure", "friction_chart_figure", "velocity_sweep_figure"]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
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


def _tick_text(v: float) -> str:
    """El rótulo de una marca: el número completo hasta 10⁶ (con espacio de miles desde
    10⁴, como :func:`ui.balance_charts.value_axis`) y potencias de 10 fuera de eso."""
    if 1e4 <= v < 1e6:
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
    1-2-5 hasta dos décadas y media y, con más, las potencias de 10."""
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
    return {"tickvals": vals, "ticktext": [_tick_text(v) for v in vals]}


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
    reference = "alineado cuadrado" if aligned else "tresbolillo equilátero"
    fig.update_xaxes(type="log", title="Re = ρ·V_máx·D/μ", gridcolor=_GRID, **_log_ticks(xs_all))
    fig.update_yaxes(type="log", title="f", gridcolor=_GRID, **_log_ticks(ys_all))
    fig.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": f"El f de Zukauskas ({reference})", "x": 0},
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
    title: str,
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
        height=520,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": title, "x": 0},
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
        title="h y Δp contra la velocidad",
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
        title="h_o y Δp contra los bafles",
        hover_x="B",
    )
