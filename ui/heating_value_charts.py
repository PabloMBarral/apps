"""Gráficos del poder calorífico por correlaciones — Fase 6.

- :func:`comparison_figure`: el PCS de cada correlación como un punto, con la
  referencia (medida o exacta) como línea vertical. Las diferencias son de
  pocos %: unas barras desde cero las esconderían.
- :func:`parity_figure`: estimado contra medido en las 536 biomasas de
  Ghugare et al. (2014), con la recta y = x, ±10 % y el combustible actual.
- :func:`error_vs_oxygen_figure`: el error de cada muestra contra su O en base
  seca, con la media por tramos: muestra por qué falla Dulong.

Colores: la paleta validada para daltonismo de :mod:`ui.combustion_charts`
(azul para el análisis elemental, naranja para el inmediato y para el
combustible actual). Un punto hueco marca una correlación usada fuera de su
rango o de su tipo de combustible.

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en
:mod:`core.combustion.heating_value`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import plotly.graph_objects as go

from core.combustion.heating_value import DatasetFit

__all__ = [
    "ERROR_RANGE",
    "ComparisonRow",
    "comparison_figure",
    "error_vs_oxygen_figure",
    "parity_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_MUTED = "#6b7280"
_INK = "#1f2937"
_GRID = "#c4c8cf"
#: Leyenda debajo del eje x: en un celular ocupa dos o tres renglones y arriba pisaba el título.
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0}


@dataclass(frozen=True)
class ComparisonRow:
    """Una correlación en el gráfico de comparación (el valor ya en las unidades del sistema)."""

    name: str
    value: float
    analysis: str
    flagged: bool
    deviation: float | None


def _signed(x: float) -> str:
    return f"{'+' if x >= 0 else '−'}{abs(100.0 * x):.1f} %"


def comparison_figure(
    rows: Sequence[ComparisonRow],
    *,
    unit: str,
    reference: float | None = None,
    reference_label: str = "PCS medido",
) -> go.Figure:
    """El PCS de cada correlación como un punto; la referencia, punteada.

    Junto a cada punto va su desvío respecto de la referencia (o el valor, si
    no hay referencia). Azul: análisis elemental; naranja: inmediato; hueco:
    fuera del rango o del tipo de combustible con que se ajustó.
    """
    fig = go.Figure()
    names = [r.name for r in rows]
    legend_done: set[str] = set()
    for row in rows:
        color = _BLUE if row.analysis == "ultimate" else _ORANGE
        group = "elemental" if row.analysis == "ultimate" else "inmediato"
        text = _signed(row.deviation) if row.deviation is not None else f"{row.value:.5g}"
        fig.add_trace(
            go.Scatter(
                x=[row.value],
                y=[row.name],
                mode="markers+text",
                name=f"análisis {group}",
                legendgroup=group,
                showlegend=group not in legend_done,
                marker={
                    "color": "white" if row.flagged else color,
                    "size": 13,
                    "symbol": "circle",
                    "line": {"color": color, "width": 2.5},
                },
                text=[text],
                textposition="middle right",
                textfont={"color": _INK},
                cliponaxis=False,
                hovertemplate=(
                    f"{row.name}<br>PCS = %{{x:.5g}} {unit}"
                    + (f"<br>desvío {text}" if row.deviation is not None else "")
                    + ("<br>fuera de su rango o tipo" if row.flagged else "")
                    + "<extra></extra>"
                ),
            )
        )
        legend_done.add(group)
    if any(r.flagged for r in rows):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="fuera de rango o de tipo",
                marker={
                    "color": "white",
                    "size": 13,
                    "line": {"color": _MUTED, "width": 2.5},
                },
                hoverinfo="skip",
            )
        )
    values = [r.value for r in rows] + ([reference] if reference is not None else [])
    lo, hi = min(values), max(values)
    span = max(hi - lo, 0.05 * hi)
    if reference is not None:
        fig.add_vline(x=reference, line={"color": _INK, "width": 1.5, "dash": "dash"})
        fig.add_annotation(
            x=reference,
            y=1.0,
            yref="paper",
            yanchor="bottom",
            text=reference_label,
            showarrow=False,
            font={"color": _INK, "size": 12},
        )
    fig.update_layout(
        height=110 + 46 * len(rows),
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        xaxis={
            "title": f"PCS en base seca [{unit}]",
            "range": [lo - 0.15 * span, hi + 0.45 * span],
            "gridcolor": "#e5e7eb",
        },
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(names))},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "x": 0},
        separators=". ",
    )
    return fig


def parity_figure(
    fit: DatasetFit,
    *,
    name: str,
    scale: float,
    unit: str,
    current: tuple[float, float] | None = None,
) -> go.Figure:
    """Estimado contra medido en las biomasas de Ghugare et al. (2014).

    ``scale`` pasa de J/kg a las unidades del sistema; ``current`` es el
    (medido, estimado) del combustible de la página, en J/kg.
    """
    measured = [m * scale for m in fit.measured]
    predicted = [p * scale for p in fit.predicted]
    errors = [100.0 * e for e in fit.errors]
    lo = 0.0
    hi = 1.08 * max(*measured, *predicted)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi, hi, lo],
            y=[0.9 * lo, 0.9 * hi, 1.1 * hi, 1.1 * lo],
            mode="lines",
            fill="toself",
            fillcolor="rgba(196, 200, 207, 0.35)",
            line={"width": 0},
            name="±10 %",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="estimado = medido",
            line={"color": _INK, "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=measured,
            y=predicted,
            mode="markers",
            name=f"{fit.n} biomasas",
            marker={"color": _BLUE, "size": 6, "opacity": 0.55},
            customdata=list(zip(fit.names, errors, strict=True)),
            hovertemplate=(
                "%{customdata[0]}<br>medido %{x:.4g}<br>estimado %{y:.4g}"
                "<br>error %{customdata[1]:+.1f} %<extra></extra>"
            ),
        )
    )
    if current is not None:
        fig.add_trace(
            go.Scatter(
                x=[current[0] * scale],
                y=[current[1] * scale],
                mode="markers",
                name="tu combustible",
                marker={
                    "color": _ORANGE,
                    "size": 14,
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 2},
                },
                hovertemplate="tu combustible<br>medido %{x:.4g}<br>estimado %{y:.4g}"
                "<extra></extra>",
            )
        )
    fig.update_layout(
        height=460,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": name, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        xaxis={"title": f"PCS medido [{unit}]", "range": [lo, hi], "gridcolor": "#e5e7eb"},
        yaxis={
            "title": f"PCS estimado [{unit}]",
            "range": [lo, hi],
            "scaleanchor": "x",
            "gridcolor": "#e5e7eb",
        },
        legend=_LEGEND_BELOW,
        separators=". ",
    )
    return fig


def _binned_means(xs: Sequence[float], ys: Sequence[float], width: float) -> tuple[list, list]:
    """Media de ``ys`` por tramos de ``xs`` de ancho ``width`` (con al menos 5 puntos)."""
    groups: dict[int, list[float]] = {}
    for x, y in zip(xs, ys, strict=True):
        groups.setdefault(int(x // width), []).append(y)
    centers, means = [], []
    for k in sorted(groups):
        if len(groups[k]) >= 5:
            centers.append((k + 0.5) * width)
            means.append(sum(groups[k]) / len(groups[k]))
    return centers, means


#: Rango del eje del error [%]: deja afuera solo unas pocas muestras con datos dudosos.
ERROR_RANGE = (-50.0, 40.0)


def error_vs_oxygen_figure(
    fit: DatasetFit, *, name: str, current: tuple[float, float] | None = None
) -> go.Figure:
    """El error de cada biomasa contra su O en base seca, con la media por tramos de 5 %.

    ``current`` es el (O, error) del combustible de la página, en fracciones. El
    eje del error va de −50 a +40 % (:data:`ERROR_RANGE`): fuera quedan menos de 10
    de las 536 muestras (con datos dudosos), que achatarían la nube.
    """
    oxygen = [100.0 * o for o in fit.oxygen]
    errors = [100.0 * e for e in fit.errors]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=oxygen,
            y=errors,
            mode="markers",
            name="cada biomasa",
            marker={"color": _BLUE, "size": 6, "opacity": 0.5},
            customdata=fit.names,
            hovertemplate="%{customdata}<br>O = %{x:.3g} %<br>error %{y:+.1f} %<extra></extra>",
        )
    )
    centers, means = _binned_means(oxygen, errors, 5.0)
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=means,
            mode="lines+markers",
            name="media cada 5 % de O",
            line={"color": _INK, "width": 2.5},
            marker={"size": 6},
            hovertemplate="O ≈ %{x:.3g} %<br>error medio %{y:+.1f} %<extra></extra>",
        )
    )
    if current is not None:
        fig.add_trace(
            go.Scatter(
                x=[100.0 * current[0]],
                y=[100.0 * current[1]],
                mode="markers",
                name="tu combustible",
                marker={
                    "color": _ORANGE,
                    "size": 14,
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 2},
                },
                hovertemplate="tu combustible<br>O = %{x:.3g} %<br>error %{y:+.1f} %"
                "<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line={"color": _MUTED, "width": 1, "dash": "dot"})
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": name, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        xaxis={"title": "O en base seca [%]", "gridcolor": "#e5e7eb"},
        yaxis={
            "title": "error del PCS [%]",
            "range": list(ERROR_RANGE),
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        legend=_LEGEND_BELOW,
        separators=". ",
    )
    return fig
