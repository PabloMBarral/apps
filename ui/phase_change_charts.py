"""Gráficos de la ebullición y la condensación — Fase 11.2.

- :func:`boiling_curve_figure`: la curva de ebullición (Nukiyama, 1934) en log–log por
  regímenes, con el inicio de la nucleada, el flujo crítico, el punto de Leidenfrost, el
  punto de los datos y, si se pasa el flujo crítico, el salto a la película («burnout»).
- :func:`condensation_curve_figure`: h contra el alto de la pared (y Re en un segundo panel,
  con los límites de los regímenes), o h contra el diámetro en los tubos y la esfera.
- :func:`comparison_figure`: h de cada geometría con las mismas temperaturas, en un gráfico
  de puntos (las diferencias son de decenas de %; unas barras desde cero las esconderían).

Colores: la paleta validada para daltonismo del proyecto. La nucleada en azul y la película
en naranja (los dos regímenes con correlación), la convección natural en gris y la transición
en gris claro rayado (es una interpolación); en la condensación, los regímenes en azul,
aguamarina y naranja. El punto de los datos va en tinta con borde blanco.

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en
:mod:`core.heat_transfer.boiling` y :mod:`core.heat_transfer.condensation`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.heat_transfer.boiling import REGIMES, BoilingCurve, PoolBoilingResult
from core.heat_transfer.condensation import (
    FILM_REGIMES,
    RE_LAMINAR,
    RE_TURBULENT,
    ComparisonRow,
    CondensationCurve,
    CondensationResult,
)

__all__ = ["boiling_curve_figure", "comparison_figure", "condensation_curve_figure"]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_GRAY = "#6b7280"
_LIGHT_GRAY = "#9ca3af"
_INK = "#1f2937"
_GRID = "#e5e7eb"
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0}
#: El punto de los datos: tinta con borde blanco (se ve sobre el azul y el naranja).
_MARKER = {"size": 13, "color": _INK, "line": {"color": "white", "width": 2}}

_BOILING_STYLE = {
    "natural": {"color": _GRAY, "width": 2.2},
    "nucleate": {"color": _BLUE, "width": 2.8},
    "transition": {"color": _LIGHT_GRAY, "width": 2.2, "dash": "dash"},
    "film": {"color": _ORANGE, "width": 2.8},
}
_FILM_STYLE = {
    "laminar": {"color": _BLUE, "width": 2.6},
    "wavy": {"color": _AQUA, "width": 2.6},
    "turbulent": {"color": _ORANGE, "width": 2.6},
}


def _comma(x: float, fmt: str = ".3g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _tick_text(v: float) -> str:
    if 1e-3 <= v < 1e4:
        return _comma(v, "g")
    exponent = math.floor(math.log10(v) + 1e-9)
    mantissa = v / 10.0**exponent
    head = "" if abs(mantissa - 1.0) < 1e-9 else f"{_comma(mantissa, 'g')}·"
    return f"{head}10{str(exponent).translate(_SUP)}"


def _log_ticks(values: Sequence[float]) -> dict[str, object]:
    """Marcas con el número completo en un eje logarítmico (plotly rotula 20 y 0,2 con un
    «2» suelto): 1-2-5 hasta dos décadas y media y, con más, las potencias de 10."""
    pos = [v for v in values if v > 0.0 and math.isfinite(v)]
    if not pos or math.log10(max(pos) / min(pos)) <= 1.0:
        return {}
    lo, hi = math.floor(math.log10(min(pos))), math.ceil(math.log10(max(pos)))
    factors = (1.0, 2.0, 5.0) if math.log10(max(pos) / min(pos)) <= 2.5 else (1.0,)
    vals = [f * 10.0**e for e in range(lo - 1, hi + 2) for f in factors]
    return {"tickvals": vals, "ticktext": [_tick_text(v) for v in vals]}


def _segments(
    xs: Sequence[float], ys: Sequence[float], labels: Sequence[str]
) -> list[tuple[str, list[float], list[float]]]:
    """Los tramos consecutivos con el mismo régimen; cada tramo arranca en el último punto
    del anterior para que la curva no tenga huecos."""
    out: list[tuple[str, list[float], list[float]]] = []
    for k, (x, y, label) in enumerate(zip(xs, ys, labels, strict=True)):
        if not out or out[-1][0] != label:
            start_x = [xs[k - 1]] if k else []
            start_y = [ys[k - 1]] if k else []
            out.append((label, [*start_x, x], [*start_y, y]))
        else:
            out[-1][1].append(x)
            out[-1][2].append(y)
    return out


def _point(fig: go.Figure, x: float, y: float, name: str, **kwargs: object) -> None:
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            name=name,
            marker=_MARKER,
            **kwargs,
        )
    )


# ---------------------------------------------------------------------
# Ebullición
# ---------------------------------------------------------------------


def boiling_curve_figure(
    curve: BoilingCurve,
    r: PoolBoilingResult,
    *,
    q_scale: float,
    q_unit: str,
    dT_scale: float,
    dT_unit: str,
) -> go.Figure:
    """La curva de ebullición en log–log: q̇ contra ΔT_e = T_s − T_sat por regímenes.

    ``q_scale`` y ``dT_scale`` pasan de W/m² y K a las unidades del sistema.
    """
    fig = go.Figure()
    xs = [x * dT_scale for x in curve.dT]
    ys = [y * q_scale for y in curve.q]
    shown: set[str] = set()
    for regime, sx, sy in _segments(xs, ys, curve.regime):
        name = REGIMES[regime]  # type: ignore[index]
        if regime == "transition":
            name += " (interpolada)"
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                name=name,
                line=_BOILING_STYLE[regime],
                showlegend=regime not in shown,
                legendgroup=regime,
                hovertemplate=f"{REGIMES[regime]}<br>ΔT = %{{x:.4g}} {dT_unit}"  # type: ignore[index]
                f"<br>q̇ = %{{y:.4g}} {q_unit}<extra></extra>",
            )
        )
        shown.add(regime)
    marks = [
        (curve.critical, "flujo crítico", "top center"),
        (curve.leidenfrost, "Leidenfrost", "bottom center"),
    ]
    if r.minimum_case != "crossing":
        marks = marks[:1]
    fig.add_trace(
        go.Scatter(
            x=[p[0] * dT_scale for p, _, _ in marks],
            y=[p[1] * q_scale for p, _, _ in marks],
            mode="markers+text",
            text=[t for _, t, _ in marks],
            textposition=[pos for _, _, pos in marks],
            textfont={"color": _GRAY},
            marker={"size": 8, "color": "white", "line": {"color": _GRAY, "width": 2}},
            name="puntos notables",
            showlegend=False,
            cliponaxis=False,
            hoverinfo="skip",
        )
    )
    x0, y0 = r.dT_K * dT_scale, r.q * q_scale
    _point(
        fig,
        x0,
        y0,
        "tu punto",
        hovertemplate=f"ΔT = %{{x:.4g}} {dT_unit}<br>q̇ = %{{y:.4g}} {q_unit}<extra></extra>",
    )
    if r.burnout:
        fig.add_annotation(
            x=math.log10(x0),
            y=math.log10(y0),
            ax=math.log10(curve.critical[0] * dT_scale),
            ay=math.log10(curve.critical[1] * q_scale),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowwidth=1.6,
            arrowcolor=_INK,
            text="",
        )
        fig.add_annotation(
            x=0.5 * (math.log10(x0) + math.log10(curve.critical[0] * dT_scale)),
            y=math.log10(y0),
            text="burnout",
            showarrow=False,
            yshift=12,
            font={"color": _INK},
        )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": "Curva de ebullición", "x": 0},
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    fig.update_xaxes(
        type="log",
        title=f"ΔT<sub>e</sub> = T<sub>s</sub> − T<sub>sat</sub> [{dT_unit}]",
        gridcolor=_GRID,
        **_log_ticks(xs),
    )
    fig.update_yaxes(type="log", title=f"q̇ [{q_unit}]", gridcolor=_GRID, **_log_ticks(ys))
    return fig


# ---------------------------------------------------------------------
# Condensación
# ---------------------------------------------------------------------


def condensation_curve_figure(
    curve: CondensationCurve,
    r: CondensationResult,
    *,
    h_scale: float,
    h_unit: str,
    x_scale: float,
    x_unit: str,
) -> go.Figure:
    """h contra el alto L (con Re abajo, en su propio panel) o contra el diámetro D."""
    xs = [x * x_scale for x in curve.x]
    hs = [h * h_scale for h in curve.h]
    tall = curve.Re is not None
    x_now = (r.inputs.L_m if tall else r.inputs.D_m) * x_scale
    x_name = "Alto L" if tall else "Diámetro D"
    if tall:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    else:
        fig = go.Figure()
    kw = {"row": 1, "col": 1} if tall else {}
    for regime, sx, sy in _segments(xs, hs, curve.regime):
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                name=FILM_REGIMES[regime],  # type: ignore[index]
                line=_FILM_STYLE[regime],
                legendgroup=regime,
                hovertemplate=f"{x_name} = %{{x:.4g}} {x_unit}<br>h = %{{y:.4g}} {h_unit}"
                "<extra></extra>",
            ),
            **kw,
        )
    fig.add_trace(
        go.Scatter(
            x=[x_now],
            y=[r.h * h_scale],
            mode="markers",
            name="tus datos",
            marker=_MARKER,
            hovertemplate=f"{x_name} = %{{x:.4g}} {x_unit}<br>h = %{{y:.4g}} {h_unit}"
            "<extra></extra>",
        ),
        **kw,
    )
    if tall:
        assert curve.Re is not None and r.Re is not None
        for regime, sx, sy in _segments(xs, list(curve.Re), curve.regime):
            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="lines",
                    name=FILM_REGIMES[regime],  # type: ignore[index]
                    line=_FILM_STYLE[regime],
                    legendgroup=regime,
                    showlegend=False,
                    hovertemplate=f"{x_name} = %{{x:.4g}} {x_unit}<br>Re = %{{y:.4g}}"
                    "<extra></extra>",
                ),
                row=2,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=[x_now],
                y=[r.Re],
                mode="markers",
                name="tus datos",
                showlegend=False,
                marker=_MARKER,
            ),
            row=2,
            col=1,
        )
        # Los límites de los regímenes como trazas: en un eje logarítmico las formas
        # (add_hline) toman la y como exponente y el eje llegaba a 10¹⁸⁰⁰.
        for Re_limit in (RE_LAMINAR, RE_TURBULENT):
            fig.add_trace(
                go.Scatter(
                    x=[xs[0], xs[-1]],
                    y=[Re_limit, Re_limit],
                    mode="lines+text",
                    text=[f"Re = {Re_limit:.0f}", ""],
                    textposition="top right",
                    textfont={"color": _GRAY},
                    line={"color": _GRAY, "width": 1.2, "dash": "dot"},
                    name=f"Re = {Re_limit:.0f}",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )
        Re_values = [*curve.Re, RE_LAMINAR, RE_TURBULENT]
        fig.update_yaxes(
            type="log", title="Re", gridcolor=_GRID, row=2, col=1, **_log_ticks(Re_values)
        )
        fig.update_yaxes(
            title=f"h [{h_unit}]", gridcolor=_GRID, exponentformat="none", row=1, col=1
        )
        fig.update_xaxes(type="log", gridcolor=_GRID, row=1, col=1, **_log_ticks(xs))
        fig.update_xaxes(
            type="log",
            title=f"{x_name} [{x_unit}]",
            gridcolor=_GRID,
            row=2,
            col=1,
            **_log_ticks(xs),
        )
        height = 520
    else:
        fig.update_xaxes(
            type="log", title=f"{x_name} [{x_unit}]", gridcolor=_GRID, **_log_ticks(xs)
        )
        fig.update_yaxes(title=f"h [{h_unit}]", gridcolor=_GRID, exponentformat="none")
        height = 380
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": f"h contra {'el alto' if tall else 'el diámetro'}", "x": 0},
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    return fig


def comparison_figure(rows: Sequence[ComparisonRow], *, h_scale: float, h_unit: str) -> go.Figure:
    """h de cada geometría con las mismas temperaturas (la de los datos, en naranja y con
    «(tus datos)» en el rótulo: sin leyenda, que a 390 px pisaba el título del eje)."""
    names = [_wrap(row.name + (" (tus datos)" if row.current else "")) for row in rows]
    fig = go.Figure()
    for current in (False, True):
        sel = [k for k, row in enumerate(rows) if row.current == current]
        if not sel:
            continue
        fig.add_trace(
            go.Scatter(
                x=[rows[k].h * h_scale for k in sel],
                y=[names[k] for k in sel],
                mode="markers+text",
                text=[_comma(rows[k].h * h_scale, ".4g") for k in sel],
                textposition="middle right",
                cliponaxis=False,
                marker={
                    "size": 13,
                    "color": _ORANGE if current else _BLUE,
                    "line": {"color": _INK, "width": 1},
                },
                name="tus datos" if current else "otras geometrías",
                showlegend=False,
                hovertemplate=f"%{{y}}<br>h = %{{x:.4g}} {h_unit}<extra></extra>",
            )
        )
    top = max(row.h for row in rows) * h_scale
    fig.update_layout(
        height=90 + 52 * len(rows),
        margin={"l": 10, "r": 40, "t": 40, "b": 10},
        title={"text": "La misma pared en otras geometrías", "x": 0},
        showlegend=False,
        separators=", ",
        xaxis={
            "title": f"h [{h_unit}]",
            "gridcolor": _GRID,
            "range": [0, 1.25 * top],
            "exponentformat": "none",
        },
        yaxis={"autorange": "reversed", "categoryorder": "array", "categoryarray": names},
    )
    return fig


def _wrap(text: str, width: int = 22) -> str:
    """Parte un rótulo largo en renglones (``<br>``) de hasta ``width`` caracteres."""
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "<br>".join(lines)
